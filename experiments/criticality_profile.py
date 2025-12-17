"""
JASO Step 1: Layer Criticality Profiling

This script measures the quality impact of:
1. Skipping each layer at each timestep
2. Using sparse attention at each layer/timestep
3. Different sparsity levels

Output: A criticality matrix C[layer, timestep] that guides JASO decisions.
"""

import argparse
import json
import os
from pathlib import Path
from typing import Dict, List, Tuple
import gc

import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
from diffusers import HunyuanVideoPipeline, HunyuanVideoTransformer3DModel, FlowMatchEulerDiscreteScheduler

from svg.logger import logger
from svg.utils.seed import seed_everything


def compute_latent_similarity(latent1: torch.Tensor, latent2: torch.Tensor) -> Dict[str, float]:
    """Compute similarity metrics between two latents."""
    # MSE
    mse = F.mse_loss(latent1, latent2).item()

    # Cosine similarity
    flat1 = latent1.flatten()
    flat2 = latent2.flatten()
    cosine = F.cosine_similarity(flat1.unsqueeze(0), flat2.unsqueeze(0)).item()

    # PSNR (treating latent as signal)
    max_val = max(latent1.abs().max().item(), latent2.abs().max().item())
    psnr = 10 * np.log10(max_val ** 2 / (mse + 1e-10)) if mse > 0 else 100.0

    return {
        "mse": mse,
        "cosine": cosine,
        "psnr": psnr,
    }


class CriticalityProfiler:
    """
    Profiles layer criticality by measuring output deviation when:
    1. A layer is "degraded" (simulated offload delay / sparse attention)
    2. Comparing against full-precision baseline
    """

    def __init__(
        self,
        pipe: HunyuanVideoPipeline,
        num_timesteps: int = 50,
        num_layers: int = 60,
    ):
        self.pipe = pipe
        self.transformer = pipe.transformer
        self.num_timesteps = num_timesteps
        self.num_layers = num_layers

        # Criticality matrix: C[layer, timestep] = importance score
        self.criticality = np.zeros((num_layers, num_timesteps))

        # Hooks for capturing intermediate outputs
        self._hooks = []
        self._layer_outputs = {}

    def _register_hooks(self):
        """Register forward hooks to capture layer outputs."""
        def make_hook(layer_idx):
            def hook(module, input, output):
                if isinstance(output, tuple):
                    self._layer_outputs[layer_idx] = output[0].detach().clone()
                else:
                    self._layer_outputs[layer_idx] = output.detach().clone()
            return hook

        # Hook into transformer blocks
        for idx, block in enumerate(self.transformer.transformer_blocks):
            h = block.register_forward_hook(make_hook(idx))
            self._hooks.append(h)

        for idx, block in enumerate(self.transformer.single_transformer_blocks):
            h = block.register_forward_hook(make_hook(len(self.transformer.transformer_blocks) + idx))
            self._hooks.append(h)

    def _remove_hooks(self):
        """Remove all registered hooks."""
        for h in self._hooks:
            h.remove()
        self._hooks = []
        self._layer_outputs = {}

    def profile_layer_skip_impact(
        self,
        prompt: str,
        height: int = 720,
        width: int = 1280,
        num_frames: int = 129,
        seed: int = 42,
    ) -> np.ndarray:
        """
        Profile the impact of skipping each layer at each timestep.

        Returns: criticality matrix C[layer, timestep]
        """
        seed_everything(seed)

        # Get baseline latent trajectory
        logger.info("Running baseline generation to capture latent trajectory...")
        baseline_latents = self._run_with_latent_capture(
            prompt, height, width, num_frames, seed
        )

        # For each layer, measure impact of degradation at each timestep
        logger.info("Profiling layer criticality...")

        for layer_idx in tqdm(range(self.num_layers), desc="Layers"):
            for timestep_idx in tqdm(range(self.num_timesteps), desc=f"Layer {layer_idx}", leave=False):
                # Run with this layer degraded at this timestep
                degraded_latent = self._run_with_layer_degraded(
                    prompt, height, width, num_frames, seed,
                    degrade_layer=layer_idx,
                    degrade_timestep=timestep_idx,
                    baseline_latents=baseline_latents,
                )

                # Compute deviation from baseline
                if degraded_latent is not None and timestep_idx < len(baseline_latents):
                    metrics = compute_latent_similarity(
                        baseline_latents[timestep_idx],
                        degraded_latent
                    )
                    # Criticality = how much quality drops (higher = more critical)
                    self.criticality[layer_idx, timestep_idx] = 1.0 - metrics["cosine"]

        return self.criticality

    def _run_with_latent_capture(
        self,
        prompt: str,
        height: int,
        width: int,
        num_frames: int,
        seed: int,
    ) -> List[torch.Tensor]:
        """Run generation and capture latent at each timestep."""
        latents_per_step = []

        def callback(pipe, step_idx, timestep, callback_kwargs):
            latents_per_step.append(callback_kwargs["latents"].detach().cpu().clone())
            return callback_kwargs

        seed_everything(seed)

        with torch.no_grad():
            _ = self.pipe(
                prompt=prompt,
                height=height,
                width=width,
                num_frames=num_frames,
                num_inference_steps=self.num_timesteps,
                callback_on_step_end=callback,
                output_type="latent",
            )

        return latents_per_step

    def _run_with_layer_degraded(
        self,
        prompt: str,
        height: int,
        width: int,
        num_frames: int,
        seed: int,
        degrade_layer: int,
        degrade_timestep: int,
        baseline_latents: List[torch.Tensor],
    ) -> torch.Tensor:
        """
        Run generation with a specific layer degraded at a specific timestep.

        Degradation strategies:
        1. Skip the layer entirely (output = input)
        2. Use cached output from previous timestep
        3. Add noise to simulate precision loss
        """
        current_step = [0]
        degraded_output = [None]

        # Strategy: Skip layer by making it identity at target timestep
        original_forward = None
        target_block = None

        if degrade_layer < len(self.transformer.transformer_blocks):
            target_block = self.transformer.transformer_blocks[degrade_layer]
        else:
            single_idx = degrade_layer - len(self.transformer.transformer_blocks)
            if single_idx < len(self.transformer.single_transformer_blocks):
                target_block = self.transformer.single_transformer_blocks[single_idx]

        if target_block is None:
            return None

        original_forward = target_block.forward

        def degraded_forward(*args, **kwargs):
            if current_step[0] == degrade_timestep:
                # Skip this layer - return input unchanged
                hidden_states = args[0] if args else kwargs.get('hidden_states')
                encoder_hidden_states = kwargs.get('encoder_hidden_states', None)
                return hidden_states, encoder_hidden_states
            return original_forward(*args, **kwargs)

        def callback(pipe, step_idx, timestep, callback_kwargs):
            current_step[0] = step_idx
            if step_idx == degrade_timestep:
                degraded_output[0] = callback_kwargs["latents"].detach().cpu().clone()
            return callback_kwargs

        try:
            target_block.forward = degraded_forward
            seed_everything(seed)

            with torch.no_grad():
                _ = self.pipe(
                    prompt=prompt,
                    height=height,
                    width=width,
                    num_frames=num_frames,
                    num_inference_steps=self.num_timesteps,
                    callback_on_step_end=callback,
                    output_type="latent",
                )
        finally:
            target_block.forward = original_forward

        return degraded_output[0]

    def save_profile(self, output_path: str):
        """Save criticality profile to file."""
        np.save(output_path, self.criticality)
        logger.info(f"Saved criticality profile to {output_path}")

        # Also save as JSON for visualization
        json_path = output_path.replace('.npy', '.json')
        profile_data = {
            "num_layers": self.num_layers,
            "num_timesteps": self.num_timesteps,
            "criticality": self.criticality.tolist(),
            "layer_importance": self.criticality.mean(axis=1).tolist(),  # Avg across timesteps
            "timestep_importance": self.criticality.mean(axis=0).tolist(),  # Avg across layers
        }
        with open(json_path, 'w') as f:
            json.dump(profile_data, f, indent=2)
        logger.info(f"Saved criticality JSON to {json_path}")

    @staticmethod
    def load_profile(path: str) -> np.ndarray:
        """Load criticality profile from file."""
        return np.load(path)


def visualize_criticality(criticality: np.ndarray, output_path: str):
    """Generate visualization of criticality matrix."""
    try:
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        # Heatmap
        im = axes[0].imshow(criticality, aspect='auto', cmap='hot')
        axes[0].set_xlabel('Timestep')
        axes[0].set_ylabel('Layer')
        axes[0].set_title('Layer Criticality Matrix')
        plt.colorbar(im, ax=axes[0])

        # Layer importance (averaged across timesteps)
        layer_importance = criticality.mean(axis=1)
        axes[1].bar(range(len(layer_importance)), layer_importance)
        axes[1].set_xlabel('Layer')
        axes[1].set_ylabel('Avg Criticality')
        axes[1].set_title('Layer Importance')

        # Timestep importance (averaged across layers)
        timestep_importance = criticality.mean(axis=0)
        axes[2].plot(timestep_importance)
        axes[2].set_xlabel('Timestep')
        axes[2].set_ylabel('Avg Criticality')
        axes[2].set_title('Timestep Importance')

        plt.tight_layout()
        plt.savefig(output_path, dpi=150)
        plt.close()
        logger.info(f"Saved criticality visualization to {output_path}")

    except ImportError:
        logger.warning("matplotlib not available, skipping visualization")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Profile layer criticality for JASO")
    parser.add_argument("--model_id", type=str, default="tencent/HunyuanVideo")
    parser.add_argument("--prompt", type=str, default="A cat walks on the grass, realistic style")
    parser.add_argument("--height", type=int, default=480)  # Use smaller resolution for profiling
    parser.add_argument("--width", type=int, default=848)
    parser.add_argument("--num_frames", type=int, default=33)  # Fewer frames for profiling
    parser.add_argument("--num_inference_steps", type=int, default=20)  # Fewer steps for profiling
    parser.add_argument("--output_dir", type=str, default="experiments/criticality_profiles")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Load model
    logger.info("Loading model...")
    transformer = HunyuanVideoTransformer3DModel.from_pretrained(
        args.model_id, subfolder="transformer", torch_dtype=torch.bfloat16, revision='refs/pr/18'
    )
    scheduler = FlowMatchEulerDiscreteScheduler(shift=7.0)
    pipe = HunyuanVideoPipeline.from_pretrained(
        args.model_id, transformer=transformer, scheduler=scheduler,
        revision='refs/pr/18', torch_dtype=torch.bfloat16
    )
    pipe.to("cuda")
    pipe.vae.enable_tiling()

    # Profile
    num_layers = len(pipe.transformer.transformer_blocks) + len(pipe.transformer.single_transformer_blocks)
    profiler = CriticalityProfiler(pipe, num_timesteps=args.num_inference_steps, num_layers=num_layers)

    criticality = profiler.profile_layer_skip_impact(
        prompt=args.prompt,
        height=args.height,
        width=args.width,
        num_frames=args.num_frames,
        seed=args.seed,
    )

    # Save results
    profiler.save_profile(os.path.join(args.output_dir, "criticality.npy"))
    visualize_criticality(criticality, os.path.join(args.output_dir, "criticality.png"))

    logger.info("Done!")
