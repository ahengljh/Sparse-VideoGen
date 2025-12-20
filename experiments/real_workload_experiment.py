"""
Real Video DiT Workload Experiment for Memory Synchronization Strategies

This script runs ACTUAL video generation workloads with different memory
synchronization strategies. The only controlled variable is the sync strategy -
all other aspects (model, prompt, resolution, steps) remain constant.

Key difference from simulation:
- Uses real HunyuanVideo model
- Generates actual video output
- Measures memory during real inference

Usage:
    python experiments/real_workload_experiment.py --strategy all --num_trials 3
"""

import argparse
import gc
import json
import math
import os
import sys
import time
from copy import deepcopy
from dataclasses import dataclass, field, asdict
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from termcolor import colored

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from experiments.memory_sync_strategies import (
    SyncStrategy,
    MemoryProfiler,
    MemoryStats,
    BaseSyncStrategy,
    create_strategy,
)


@dataclass
class RealWorkloadConfig:
    """Configuration for real workload experiments"""
    model_id: str = "tencent/HunyuanVideo"
    height: int = 544
    width: int = 960
    num_frames: int = 61
    num_inference_steps: int = 30
    guidance_scale: float = 6.0

    prompt: str = "A cat walks on the grass, realistic style, high quality"
    negative_prompt: str = "low quality, blurry, distorted"

    working_set_size: int = 5
    device: int = 0
    seed: int = 42

    output_dir: str = "outputs/memory_sync_experiments"

    # Use local model path if available (avoids network errors)
    use_local_model: bool = True
    local_model_path: str = "models/HunyuanVideo"


@dataclass
class WorkloadResult:
    """Results from a single workload run"""
    strategy: str
    trial_id: int
    oom_occurred: bool = False
    oom_message: str = ""
    oom_step: int = -1
    oom_block: int = -1

    total_steps: int = 0
    successful_steps: int = 0
    total_blocks: int = 0

    peak_memory_mb: float = 0.0
    memory_samples: List[Dict] = field(default_factory=list)
    step_peak_memories: List[float] = field(default_factory=list)

    total_time_seconds: float = 0.0
    sync_count: int = 0
    load_count: int = 0
    offload_count: int = 0

    video_generated: bool = False
    output_path: str = ""

    @property
    def avg_peak_memory_mb(self) -> float:
        if not self.step_peak_memories:
            return self.peak_memory_mb
        return sum(self.step_peak_memories) / len(self.step_peak_memories)

    @property
    def peak_variance_mb(self) -> float:
        if len(self.step_peak_memories) < 2:
            return 0.0
        mean = sum(self.step_peak_memories) / len(self.step_peak_memories)
        variance = sum((p - mean) ** 2 for p in self.step_peak_memories) / len(self.step_peak_memories)
        return variance ** 0.5


class OffloadingTransformerWrapper:
    """
    Wraps the HunyuanVideo transformer to apply block-level offloading
    with configurable memory synchronization strategies.

    This wrapper intercepts the block execution loop and manages
    GPU memory by offloading blocks not in the working set.
    """

    def __init__(
        self,
        transformer: nn.Module,
        strategy: BaseSyncStrategy,
        working_set_size: int = 5,
        device: int = 0,
    ):
        self.transformer = transformer
        self.strategy = strategy
        self.working_set_size = working_set_size
        self.device = device

        # Get all blocks
        self.double_blocks = list(transformer.transformer_blocks)
        self.single_blocks = list(transformer.single_transformer_blocks)
        self.all_blocks = self.double_blocks + self.single_blocks
        self.num_blocks = len(self.all_blocks)

        # Calculate block sizes
        self.block_sizes_mb = []
        for block in self.all_blocks:
            size = sum(p.numel() * p.element_size() for p in block.parameters())
            self.block_sizes_mb.append(size / (1024 ** 2))
        self.avg_block_size = sum(self.block_sizes_mb) / len(self.block_sizes_mb)

        # Track GPU resident blocks
        self.gpu_blocks: set = set()

        # Statistics
        self.current_step = 0
        self.oom_occurred = False
        self.oom_message = ""
        self.oom_step = -1
        self.oom_block = -1

    def initialize_offload_state(self):
        """Move all blocks to CPU and load initial working set"""
        print(f"  Initializing offload state (working set = {self.working_set_size})...")

        # Move all blocks to CPU first
        for i, block in enumerate(self.all_blocks):
            block.to('cpu')
        self.gpu_blocks.clear()

        gc.collect()
        torch.cuda.empty_cache()

        # Load initial working set
        for i in range(min(self.working_set_size, self.num_blocks)):
            self._load_block(i)

        torch.cuda.synchronize(self.device)
        print(f"  Initial GPU blocks: {sorted(self.gpu_blocks)}")

    def _load_block(self, block_idx: int) -> bool:
        """Load a block to GPU"""
        if block_idx in self.gpu_blocks:
            return True

        try:
            success = self.strategy.load_block(
                self.all_blocks[block_idx],
                self.block_sizes_mb[block_idx]
            )
            if success:
                self.gpu_blocks.add(block_idx)
            return success
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                self.oom_occurred = True
                self.oom_message = f"OOM loading block {block_idx}: {e}"
                self.oom_block = block_idx
                return False
            raise

    def _offload_block(self, block_idx: int) -> bool:
        """Offload a block to CPU"""
        if block_idx not in self.gpu_blocks:
            return True

        try:
            success = self.strategy.offload_block(self.all_blocks[block_idx])
            if success:
                self.gpu_blocks.discard(block_idx)
            return success
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                return False
            raise

    def _prepare_block(self, block_idx: int) -> bool:
        """Ensure block is on GPU, managing working set"""
        if block_idx in self.gpu_blocks:
            self.strategy.pre_forward_sync(block_idx, self.num_blocks)
            return True

        # Need to make room if at capacity
        while len(self.gpu_blocks) >= self.working_set_size:
            # Offload the block furthest behind current position
            to_offload = min(self.gpu_blocks)
            if to_offload >= block_idx:
                # All blocks are ahead, offload the one we won't need soonest
                to_offload = max(self.gpu_blocks)

            if not self._offload_block(to_offload):
                return False

        # Load the needed block
        if not self._load_block(block_idx):
            return False

        self.strategy.pre_forward_sync(block_idx, self.num_blocks)
        return True

    def _finish_block(self, block_idx: int):
        """Called after block forward pass"""
        self.strategy.post_forward_sync(block_idx, self.num_blocks)

    def run_transformer_with_offload(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        temb: torch.Tensor,
        attention_mask: torch.Tensor,
        image_rotary_emb: Tuple,
        timestep: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Execute all transformer blocks with offloading management.
        This replaces the normal block loop in the transformer forward.
        """
        self.current_step = timestep

        # Process double blocks
        for i, block in enumerate(self.double_blocks):
            if not self._prepare_block(i):
                self.oom_step = self.current_step
                raise RuntimeError(self.oom_message or f"Failed to prepare block {i}")

            try:
                hidden_states, encoder_hidden_states = block(
                    hidden_states,
                    encoder_hidden_states,
                    temb,
                    attention_mask,
                    image_rotary_emb,
                    timestep,
                )
            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    self.oom_occurred = True
                    self.oom_message = f"OOM in double block {i} forward: {e}"
                    self.oom_step = self.current_step
                    self.oom_block = i
                    raise
                raise

            self._finish_block(i)

        # Process single blocks
        for i, block in enumerate(self.single_blocks):
            block_idx = len(self.double_blocks) + i

            if not self._prepare_block(block_idx):
                self.oom_step = self.current_step
                raise RuntimeError(self.oom_message or f"Failed to prepare block {block_idx}")

            try:
                hidden_states, encoder_hidden_states = block(
                    hidden_states,
                    encoder_hidden_states,
                    temb,
                    attention_mask,
                    image_rotary_emb,
                    timestep,
                )
            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    self.oom_occurred = True
                    self.oom_message = f"OOM in single block {i} forward: {e}"
                    self.oom_step = self.current_step
                    self.oom_block = block_idx
                    raise
                raise

            self._finish_block(block_idx)

        return hidden_states, encoder_hidden_states

    def get_stats(self) -> Dict[str, Any]:
        """Get current statistics"""
        return {
            'num_blocks': self.num_blocks,
            'num_double_blocks': len(self.double_blocks),
            'num_single_blocks': len(self.single_blocks),
            'working_set_size': self.working_set_size,
            'avg_block_size_mb': self.avg_block_size,
            'gpu_blocks': list(self.gpu_blocks),
            'sync_count': self.strategy.sync_count,
            'load_count': self.strategy.load_count,
            'offload_count': self.strategy.offload_count,
            'oom_occurred': self.oom_occurred,
            'oom_message': self.oom_message,
            'oom_step': self.oom_step,
            'oom_block': self.oom_block,
        }


def create_offloading_forward(wrapper: OffloadingTransformerWrapper):
    """
    Create a new forward function for the transformer that uses offloading.
    This function replaces the original forward to inject our memory management.
    """
    original_transformer = wrapper.transformer

    def offloading_forward(
        self,
        hidden_states: torch.Tensor,
        timestep: torch.LongTensor,
        encoder_hidden_states: torch.Tensor,
        encoder_attention_mask: torch.Tensor,
        pooled_projections: torch.Tensor,
        guidance: torch.Tensor = None,
        attention_kwargs: Optional[Dict[str, Any]] = None,
        return_dict: bool = True,
    ):
        from diffusers.models.modeling_outputs import Transformer2DModelOutput
        from diffusers.utils import USE_PEFT_BACKEND, scale_lora_layers, unscale_lora_layers

        if attention_kwargs is not None:
            attention_kwargs = attention_kwargs.copy()
            lora_scale = attention_kwargs.pop("scale", 1.0)
        else:
            lora_scale = 1.0

        if USE_PEFT_BACKEND:
            scale_lora_layers(self, lora_scale)

        batch_size, num_channels, num_frames, height, width = hidden_states.shape
        p, p_t = self.config.patch_size, self.config.patch_size_t
        post_patch_num_frames = num_frames // p_t
        post_patch_height = height // p
        post_patch_width = width // p

        # 1. RoPE
        image_rotary_emb = self.rope(hidden_states)

        # 2. Conditional embeddings
        temb, token_replace_emb = self.time_text_embed(timestep, pooled_projections, guidance)

        hidden_states = self.x_embedder(hidden_states)
        encoder_hidden_states = self.context_embedder(encoder_hidden_states, timestep, encoder_attention_mask)

        # 3. Attention mask preparation
        latent_sequence_length = hidden_states.shape[1]
        condition_sequence_length = encoder_hidden_states.shape[1]
        sequence_length = latent_sequence_length + condition_sequence_length
        attention_mask = torch.ones(
            batch_size, sequence_length, device=hidden_states.device, dtype=torch.bool
        )
        effective_condition_sequence_length = encoder_attention_mask.sum(dim=1, dtype=torch.int)
        effective_sequence_length = latent_sequence_length + effective_condition_sequence_length
        indices = torch.arange(sequence_length, device=hidden_states.device).unsqueeze(0)
        mask_indices = indices >= effective_sequence_length.unsqueeze(1)
        attention_mask = attention_mask.masked_fill(mask_indices, False)
        attention_mask = attention_mask.unsqueeze(1).unsqueeze(1)

        # 4. Transformer blocks WITH OFFLOADING
        # This is where we inject our memory management
        hidden_states, encoder_hidden_states = wrapper.run_transformer_with_offload(
            hidden_states,
            encoder_hidden_states,
            temb,
            attention_mask,
            image_rotary_emb,
            timestep.item() if hasattr(timestep, 'item') else int(timestep),
        )

        # 5. Output projection
        hidden_states = self.norm_out(hidden_states, temb)
        hidden_states = self.proj_out(hidden_states)

        hidden_states = hidden_states.reshape(
            batch_size, post_patch_num_frames, post_patch_height, post_patch_width, -1, p_t, p, p
        )
        hidden_states = hidden_states.permute(0, 4, 1, 5, 2, 6, 3, 7)
        hidden_states = hidden_states.flatten(6, 7).flatten(4, 5).flatten(2, 3)

        if USE_PEFT_BACKEND:
            unscale_lora_layers(self, lora_scale)

        if not return_dict:
            return (hidden_states,)

        return Transformer2DModelOutput(sample=hidden_states)

    return offloading_forward


def run_real_workload(
    config: RealWorkloadConfig,
    strategy_type: SyncStrategy,
    trial_id: int,
) -> WorkloadResult:
    """
    Run a real video generation workload with the specified sync strategy.
    """
    print(f"\n{'='*70}")
    print(f"REAL WORKLOAD: {strategy_type.value} - Trial {trial_id}")
    print(f"{'='*70}")

    result = WorkloadResult(
        strategy=strategy_type.value,
        trial_id=trial_id,
        total_steps=config.num_inference_steps,
    )

    # Set seed
    torch.manual_seed(config.seed + trial_id)
    torch.cuda.manual_seed(config.seed + trial_id)

    # Clean GPU
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(config.device)

    profiler = MemoryProfiler(device=config.device)

    try:
        # Import diffusers components
        from diffusers import (
            HunyuanVideoPipeline,
            HunyuanVideoTransformer3DModel,
            FlowMatchEulerDiscreteScheduler,
        )
        from diffusers.utils import export_to_video

        # Determine model path (local or HuggingFace)
        import os
        if config.use_local_model and os.path.exists(config.local_model_path):
            model_path = config.local_model_path
            revision = None
            print(f"Loading model from local path: {model_path}")
        else:
            model_path = config.model_id
            revision = 'refs/pr/18'
            print(f"Loading model from HuggingFace: {model_path}")

        # Load transformer
        transformer = HunyuanVideoTransformer3DModel.from_pretrained(
            model_path,
            subfolder="transformer",
            torch_dtype=torch.bfloat16,
            revision=revision,
            local_files_only=config.use_local_model and os.path.exists(config.local_model_path),
        )

        # Create sync strategy
        strategy = create_strategy(
            strategy_type,
            device=config.device,
            safety_margin_mb=800.0,
            memory_threshold_ratio=0.12,
        )

        # Create offloading wrapper
        wrapper = OffloadingTransformerWrapper(
            transformer=transformer,
            strategy=strategy,
            working_set_size=config.working_set_size,
            device=config.device,
        )

        result.total_blocks = wrapper.num_blocks
        print(f"Total blocks: {wrapper.num_blocks} "
              f"({len(wrapper.double_blocks)} double + {len(wrapper.single_blocks)} single)")
        print(f"Average block size: {wrapper.avg_block_size:.1f} MB")

        # Load pipeline
        scheduler = FlowMatchEulerDiscreteScheduler(shift=7.0)
        pipe = HunyuanVideoPipeline.from_pretrained(
            model_path,
            transformer=transformer,
            scheduler=scheduler,
            revision=revision,
            torch_dtype=torch.bfloat16,
            local_files_only=config.use_local_model and os.path.exists(config.local_model_path),
        )
        pipe.vae.enable_tiling()

        # Move non-transformer components to GPU
        # HunyuanVideo has two text encoders: text_encoder (LLaVA) and text_encoder_2 (CLIP)
        pipe.text_encoder.to(f'cuda:{config.device}')
        if hasattr(pipe, 'text_encoder_2') and pipe.text_encoder_2 is not None:
            pipe.text_encoder_2.to(f'cuda:{config.device}')
        pipe.vae.to(f'cuda:{config.device}')

        # Keep transformer embedding layers on GPU
        transformer.rope.to(f'cuda:{config.device}')
        transformer.time_text_embed.to(f'cuda:{config.device}')
        transformer.x_embedder.to(f'cuda:{config.device}')
        transformer.context_embedder.to(f'cuda:{config.device}')
        transformer.norm_out.to(f'cuda:{config.device}')
        transformer.proj_out.to(f'cuda:{config.device}')

        # Initialize offload state
        wrapper.initialize_offload_state()

        # Replace transformer forward with our offloading version
        original_forward = transformer.forward
        transformer.forward = create_offloading_forward(wrapper).__get__(transformer, type(transformer))

        # Memory callback for tracking per-step memory
        step_memories = []

        def memory_callback(pipe, step_idx, timestep, callback_kwargs):
            torch.cuda.synchronize(config.device)
            peak = torch.cuda.max_memory_allocated(config.device) / (1024 ** 2)
            step_memories.append(peak)
            print(f"  Step {step_idx + 1}/{config.num_inference_steps}: "
                  f"peak={peak:.0f}MB, gpu_blocks={sorted(wrapper.gpu_blocks)}", end="\r")
            return callback_kwargs

        print(f"\nGenerating video with {strategy_type.value} strategy...")
        print(f"  Prompt: {config.prompt[:50]}...")
        print(f"  Resolution: {config.width}x{config.height}, {config.num_frames} frames")

        start_time = time.time()
        profiler.reset_peak_stats()

        try:
            output = pipe(
                prompt=config.prompt,
                negative_prompt=config.negative_prompt,
                height=config.height,
                width=config.width,
                num_frames=config.num_frames,
                guidance_scale=config.guidance_scale,
                num_inference_steps=config.num_inference_steps,
                callback_on_step_end=memory_callback,
            )

            result.successful_steps = config.num_inference_steps
            result.video_generated = True

            print()  # Newline after progress

            # Save video
            os.makedirs(config.output_dir, exist_ok=True)
            output_path = os.path.join(
                config.output_dir,
                f"{strategy_type.value}_trial{trial_id}.mp4"
            )
            export_to_video(output.frames[0], output_path, fps=24)
            result.output_path = output_path
            print(f"  Video saved to: {output_path}")

        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                result.oom_occurred = True
                result.oom_message = str(e)
                result.oom_step = wrapper.oom_step
                result.oom_block = wrapper.oom_block
                result.successful_steps = wrapper.current_step
                print(colored(f"\n  OOM Error at step {wrapper.current_step}: {e}", "red"))
            else:
                raise

        result.total_time_seconds = time.time() - start_time

        # Record memory stats
        final_stats = profiler.get_current_stats()
        result.peak_memory_mb = final_stats.peak_memory_mb
        result.step_peak_memories = step_memories

        # Get strategy stats
        wrapper_stats = wrapper.get_stats()
        result.sync_count = wrapper_stats['sync_count']
        result.load_count = wrapper_stats['load_count']
        result.offload_count = wrapper_stats['offload_count']

        # Restore original forward
        transformer.forward = original_forward

        # Cleanup
        del pipe, transformer, wrapper
        gc.collect()
        torch.cuda.empty_cache()

    except Exception as e:
        error_msg = str(e).lower()
        # Distinguish between OOM errors and other errors (like network issues)
        if "out of memory" in error_msg or "cuda" in error_msg:
            result.oom_occurred = True
            result.oom_message = f"OOM: {e}"
        elif "connection" in error_msg or "protocol" in error_msg or "network" in error_msg:
            # Network error - don't count as OOM
            result.oom_occurred = False
            result.oom_message = f"Network error (not OOM): {e}"
        else:
            # Unknown error - be conservative and don't count as OOM
            result.oom_occurred = False
            result.oom_message = f"Error (not OOM): {e}"

        print(colored(f"Error: {e}", "red"))
        import traceback
        traceback.print_exc()

        gc.collect()
        torch.cuda.empty_cache()

    # Print summary
    print(f"\n  {'='*50}")
    print(f"  Trial {trial_id} Summary:")
    print(f"  {'='*50}")
    print(f"  Strategy: {result.strategy}")
    print(f"  OOM: {result.oom_occurred}")
    print(f"  Steps completed: {result.successful_steps}/{result.total_steps}")
    print(f"  Peak memory: {result.peak_memory_mb:.0f} MB ({result.peak_memory_mb/1024:.2f} GB)")
    print(f"  Avg peak: {result.avg_peak_memory_mb:.0f} MB")
    print(f"  Peak std: ±{result.peak_variance_mb:.1f} MB")
    print(f"  Sync count: {result.sync_count}")
    print(f"  Load/Offload: {result.load_count}/{result.offload_count}")
    print(f"  Time: {result.total_time_seconds:.1f}s")
    print(f"  Video generated: {result.video_generated}")

    return result


def print_summary(results: Dict[str, List[WorkloadResult]]):
    """Print experiment summary in paper format"""

    print("\n" + "=" * 80)
    print("EXPERIMENT SUMMARY")
    print("=" * 80)

    # Detailed table
    headers = ["Strategy", "OOM Rate", "Avg Peak", "Peak Std", "Time", "Syncs", "Video"]
    row_format = "{:<18} {:>10} {:>12} {:>12} {:>10} {:>8} {:>8}"

    print(row_format.format(*headers))
    print("-" * 80)

    for strategy_name, trials in results.items():
        oom_count = sum(1 for t in trials if t.oom_occurred)
        oom_rate = f"{oom_count}/{len(trials)} ({100*oom_count/len(trials):.0f}%)"

        successful = [t for t in trials if not t.oom_occurred]
        if successful:
            avg_peak_gb = np.mean([t.avg_peak_memory_mb for t in successful]) / 1024
            std_peak_gb = np.mean([t.peak_variance_mb for t in successful]) / 1024
            avg_time = np.mean([t.total_time_seconds for t in successful])
            avg_sync = np.mean([t.sync_count for t in successful])
            videos = sum(1 for t in successful if t.video_generated)
        else:
            avg_peak_gb = float('nan')
            std_peak_gb = float('nan')
            avg_time = float('nan')
            avg_sync = float('nan')
            videos = 0

        print(row_format.format(
            strategy_name,
            oom_rate,
            f"{avg_peak_gb:.2f}GB",
            f"±{std_peak_gb:.2f}GB",
            f"{avg_time:.1f}s",
            f"{avg_sync:.0f}",
            f"{videos}/{len(trials)}"
        ))

    # Paper format table
    print("\n" + "=" * 80)
    print("TABLE (Paper Format)")
    print("=" * 80)
    print("\n| 传输策略 | OOM率 | 平均峰值内存 | 峰值方差 |")
    print("|----------|-------|-------------|----------|")

    for strategy_name, trials in results.items():
        oom_count = sum(1 for t in trials if t.oom_occurred)
        oom_rate = f"{100*oom_count/len(trials):.0f}%"

        successful = [t for t in trials if not t.oom_occurred]
        if successful:
            avg_peak_gb = np.mean([t.avg_peak_memory_mb for t in successful]) / 1024
            std_peak_gb = np.mean([t.peak_variance_mb for t in successful]) / 1024
        else:
            avg_peak_gb = float('nan')
            std_peak_gb = float('nan')

        display_name = {
            'pure_async': '纯异步',
            'pure_sync': '纯同步',
            'conditional_sync': '条件同步（本文）'
        }.get(strategy_name, strategy_name)

        print(f"| {display_name} | {oom_rate} | {avg_peak_gb:.1f}GB | ±{std_peak_gb:.1f}GB |")


def main():
    parser = argparse.ArgumentParser(description="Real Video DiT Workload Experiment")

    parser.add_argument("--strategy", type=str, default="all",
                       choices=["pure_async", "pure_sync", "conditional_sync", "all"],
                       help="Strategy to test")
    parser.add_argument("--num_trials", type=int, default=3,
                       help="Number of trials per strategy")
    parser.add_argument("--working_set_size", type=int, default=5,
                       help="Number of blocks to keep on GPU")
    parser.add_argument("--num_steps", type=int, default=30,
                       help="Number of inference steps")
    parser.add_argument("--height", type=int, default=544,
                       help="Video height")
    parser.add_argument("--width", type=int, default=960,
                       help="Video width")
    parser.add_argument("--num_frames", type=int, default=61,
                       help="Number of frames")
    parser.add_argument("--output_dir", type=str, default="outputs/memory_sync_experiments",
                       help="Output directory for videos")
    parser.add_argument("--output_json", type=str, default=None,
                       help="Output JSON file for results")
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed")
    parser.add_argument("--model_id", type=str, default="tencent/HunyuanVideo",
                       help="Model ID (HuggingFace)")
    parser.add_argument("--local_model_path", type=str, default="models/HunyuanVideo",
                       help="Local model path (used if exists)")
    parser.add_argument("--no_local_model", action="store_true",
                       help="Force download from HuggingFace instead of using local model")

    args = parser.parse_args()

    if not torch.cuda.is_available():
        print(colored("Error: CUDA is not available!", "red"))
        sys.exit(1)

    # Print GPU info
    props = torch.cuda.get_device_properties(0)
    print(f"\n{'='*70}")
    print(f"GPU: {props.name}")
    print(f"Total Memory: {props.total_memory / (1024**3):.1f} GB")
    print(f"{'='*70}")

    config = RealWorkloadConfig(
        model_id=args.model_id,
        height=args.height,
        width=args.width,
        num_frames=args.num_frames,
        num_inference_steps=args.num_steps,
        working_set_size=args.working_set_size,
        seed=args.seed,
        output_dir=args.output_dir,
        use_local_model=not args.no_local_model,
        local_model_path=args.local_model_path,
    )

    # Determine strategies
    if args.strategy == "all":
        strategies = [
            SyncStrategy.PURE_ASYNC,
            SyncStrategy.PURE_SYNC,
            SyncStrategy.CONDITIONAL_SYNC,
        ]
    else:
        strategies = [SyncStrategy(args.strategy)]

    # Run experiments
    results: Dict[str, List[WorkloadResult]] = {}

    for strategy in strategies:
        strategy_results = []
        for trial in range(args.num_trials):
            result = run_real_workload(config, strategy, trial)
            strategy_results.append(result)
        results[strategy.value] = strategy_results

    # Print summary
    print_summary(results)

    # Save results
    if args.output_json:
        def to_dict(obj):
            if hasattr(obj, '__dataclass_fields__'):
                return {k: to_dict(v) for k, v in asdict(obj).items()}
            elif isinstance(obj, list):
                return [to_dict(v) for v in obj]
            elif isinstance(obj, dict):
                return {k: to_dict(v) for k, v in obj.items()}
            elif isinstance(obj, (np.floating, np.integer)):
                return float(obj)
            else:
                return obj

        output_data = {
            'config': {
                'model_id': config.model_id,
                'height': config.height,
                'width': config.width,
                'num_frames': config.num_frames,
                'num_inference_steps': config.num_inference_steps,
                'working_set_size': config.working_set_size,
            },
            'results': {k: [to_dict(r) for r in v] for k, v in results.items()},
            'timestamp': datetime.now().isoformat(),
        }

        with open(args.output_json, 'w') as f:
            json.dump(output_data, f, indent=2)
        print(f"\nResults saved to {args.output_json}")


if __name__ == "__main__":
    main()
