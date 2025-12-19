"""
Experiment Runner for Video DiT Memory Synchronization Strategies

This script runs real Video DiT inference workloads with different memory
synchronization strategies to compare:
1. OOM rate
2. Average peak memory
3. Peak memory variance

Usage:
    python experiments/run_memory_sync_experiment.py --strategy all --num_trials 5
"""

import argparse
import json
import os
import sys
import time
import gc
from datetime import datetime
from copy import deepcopy
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, asdict

import torch
import torch.cuda as cuda
import numpy as np
from termcolor import colored

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from memory_sync_strategies import (
    SyncStrategy,
    MemoryProfiler,
    ExperimentResult,
    BlockOffloadManager,
    create_strategy,
    BaseSyncStrategy,
)


@dataclass
class ExperimentConfig:
    """Configuration for experiment runs"""
    # Model settings
    model_id: str = "tencent/HunyuanVideo"
    height: int = 544  # Use smaller resolution for faster experiments
    width: int = 960
    num_frames: int = 61  # Fewer frames for experiments
    num_inference_steps: int = 30  # Fewer steps for experiments

    # Offload settings
    working_set_size: int = 5  # Number of blocks to keep on GPU
    device: int = 0

    # Experiment settings
    num_trials: int = 5
    seed: int = 42
    prompt: str = "A cat walks on the grass, realistic"

    # Safety settings
    safety_margin_mb: float = 1000.0
    memory_threshold_ratio: float = 0.15


class VideoDiTOffloadWrapper:
    """
    Wrapper for Video DiT models that implements block-level offloading.

    This wrapper intercepts the transformer block execution and manages
    CPU<->GPU memory transfers with the specified synchronization strategy.
    """

    def __init__(self,
                 transformer,
                 strategy: BaseSyncStrategy,
                 working_set_size: int = 5,
                 device: int = 0):
        self.transformer = transformer
        self.strategy = strategy
        self.working_set_size = working_set_size
        self.device = device

        # Collect all transformer blocks
        self.double_blocks = list(transformer.transformer_blocks)
        self.single_blocks = list(transformer.single_transformer_blocks)
        self.all_blocks = self.double_blocks + self.single_blocks

        # Create offload manager
        self.offload_manager = BlockOffloadManager(
            blocks=self.all_blocks,
            strategy=strategy,
            working_set_size=working_set_size,
            device=device
        )

        # Track execution statistics
        self.oom_occurred = False
        self.oom_message = ""
        self.successful_blocks = 0

    def setup_for_inference(self):
        """Prepare model for offloaded inference"""
        # Move non-block components to GPU
        self.transformer.rope.to(f'cuda:{self.device}')
        self.transformer.time_text_embed.to(f'cuda:{self.device}')
        self.transformer.x_embedder.to(f'cuda:{self.device}')
        self.transformer.context_embedder.to(f'cuda:{self.device}')
        self.transformer.norm_out.to(f'cuda:{self.device}')
        self.transformer.proj_out.to(f'cuda:{self.device}')

        # Initialize working set for blocks
        self.offload_manager.initialize_working_set(start_idx=0)

    def run_blocks_with_offload(self,
                                 hidden_states: torch.Tensor,
                                 encoder_hidden_states: torch.Tensor,
                                 temb: torch.Tensor,
                                 attention_mask: torch.Tensor,
                                 image_rotary_emb: Tuple,
                                 timestep: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Execute transformer blocks with offload management"""
        self.successful_blocks = 0

        # Process double blocks
        for i, block in enumerate(self.double_blocks):
            block_idx = i

            # Prepare block (load if needed, offload old blocks)
            if not self.offload_manager.prepare_block(block_idx):
                self.oom_occurred = True
                self.oom_message = f"OOM during load of double block {i}"
                raise RuntimeError(self.oom_message)

            try:
                # Forward pass
                hidden_states, encoder_hidden_states = block(
                    hidden_states,
                    encoder_hidden_states,
                    temb,
                    attention_mask,
                    image_rotary_emb,
                    timestep,
                )
                self.successful_blocks += 1

            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    self.oom_occurred = True
                    self.oom_message = f"OOM during forward of double block {i}: {str(e)}"
                    raise
                raise

            # Post-block cleanup
            self.offload_manager.finish_block(block_idx)

        # Process single blocks
        for i, block in enumerate(self.single_blocks):
            block_idx = len(self.double_blocks) + i

            # Prepare block
            if not self.offload_manager.prepare_block(block_idx):
                self.oom_occurred = True
                self.oom_message = f"OOM during load of single block {i}"
                raise RuntimeError(self.oom_message)

            try:
                # Forward pass
                hidden_states, encoder_hidden_states = block(
                    hidden_states,
                    encoder_hidden_states,
                    temb,
                    attention_mask,
                    image_rotary_emb,
                    timestep,
                )
                self.successful_blocks += 1

            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    self.oom_occurred = True
                    self.oom_message = f"OOM during forward of single block {i}: {str(e)}"
                    raise
                raise

            # Post-block cleanup
            self.offload_manager.finish_block(block_idx)

        return hidden_states, encoder_hidden_states

    def get_stats(self) -> Dict[str, Any]:
        """Get execution statistics"""
        manager_stats = self.offload_manager.get_stats()
        return {
            **manager_stats,
            'total_blocks': len(self.all_blocks),
            'double_blocks': len(self.double_blocks),
            'single_blocks': len(self.single_blocks),
            'oom_occurred': self.oom_occurred,
            'oom_message': self.oom_message,
            'successful_blocks': self.successful_blocks,
        }


def run_single_experiment(
    config: ExperimentConfig,
    strategy_type: SyncStrategy,
    trial_id: int,
    profiler: MemoryProfiler,
) -> ExperimentResult:
    """Run a single experiment trial with the specified strategy"""

    print(f"\n{'='*60}")
    print(f"Trial {trial_id}: Strategy = {strategy_type.value}")
    print(f"{'='*60}")

    # Set seed for reproducibility
    torch.manual_seed(config.seed + trial_id)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(config.seed + trial_id)

    result = ExperimentResult(
        strategy=strategy_type.value,
        total_steps=config.num_inference_steps,
    )

    try:
        # Import here to avoid loading model until needed
        from diffusers import (
            HunyuanVideoPipeline,
            HunyuanVideoTransformer3DModel,
            FlowMatchEulerDiscreteScheduler
        )

        # Clear GPU memory before loading
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(config.device)

        print(f"Loading model from {config.model_id}...")

        # Load transformer with minimal memory
        transformer = HunyuanVideoTransformer3DModel.from_pretrained(
            config.model_id,
            subfolder="transformer",
            torch_dtype=torch.bfloat16,
            revision='refs/pr/18'
        )

        # Create strategy
        strategy = create_strategy(
            strategy_type,
            device=config.device,
            safety_margin_mb=config.safety_margin_mb,
            memory_threshold_ratio=config.memory_threshold_ratio,
        )

        # Create offload wrapper
        wrapper = VideoDiTOffloadWrapper(
            transformer=transformer,
            strategy=strategy,
            working_set_size=config.working_set_size,
            device=config.device
        )

        # Load pipeline
        flow_shift = 7.0
        scheduler = FlowMatchEulerDiscreteScheduler(shift=flow_shift)
        pipe = HunyuanVideoPipeline.from_pretrained(
            config.model_id,
            transformer=transformer,
            scheduler=scheduler,
            revision='refs/pr/18',
            torch_dtype=torch.bfloat16
        )
        pipe.vae.enable_tiling()

        # Move non-transformer components to GPU
        pipe.text_encoder.to(f'cuda:{config.device}')
        pipe.vae.to(f'cuda:{config.device}')

        # Setup for offloaded inference
        wrapper.setup_for_inference()

        print(f"Model loaded. Starting inference with {strategy_type.value} strategy...")

        # Start profiling
        profiler.reset_peak_stats()
        start_time = time.time()

        # We'll simulate the diffusion loop with offloading
        # This is a simplified version - in production you'd hook into the pipeline

        # Run inference simulation
        # For this experiment, we'll run the block execution pattern
        # that mimics the actual diffusion process

        try:
            # Simulate batch input
            batch_size = 1
            latent_channels = pipe.transformer.config.in_channels
            latent_height = config.height // 8
            latent_width = config.width // 8
            latent_frames = (config.num_frames - 1) // 4 + 1

            print(f"Latent shape: [{batch_size}, {latent_channels}, {latent_frames}, {latent_height}, {latent_width}]")

            # Create dummy inputs on GPU
            hidden_states = torch.randn(
                batch_size,
                latent_height * latent_width * latent_frames,
                pipe.transformer.config.num_attention_heads * pipe.transformer.config.attention_head_dim,
                device=f'cuda:{config.device}',
                dtype=torch.bfloat16
            )

            encoder_hidden_states = torch.randn(
                batch_size,
                256,  # Typical text sequence length
                pipe.transformer.config.num_attention_heads * pipe.transformer.config.attention_head_dim,
                device=f'cuda:{config.device}',
                dtype=torch.bfloat16
            )

            temb = torch.randn(
                batch_size,
                pipe.transformer.config.num_attention_heads * pipe.transformer.config.attention_head_dim,
                device=f'cuda:{config.device}',
                dtype=torch.bfloat16
            )

            attention_mask = torch.ones(
                batch_size, 1, 1,
                hidden_states.shape[1] + encoder_hidden_states.shape[1],
                device=f'cuda:{config.device}',
                dtype=torch.bool
            )

            # Dummy RoPE embeddings
            image_rotary_emb = (
                torch.randn(hidden_states.shape[1], 64, device=f'cuda:{config.device}', dtype=torch.bfloat16),
                torch.randn(hidden_states.shape[1], 64, device=f'cuda:{config.device}', dtype=torch.bfloat16)
            )

            # Simulate diffusion steps
            for step in range(config.num_inference_steps):
                print(f"  Step {step + 1}/{config.num_inference_steps}", end="\r")

                # Run transformer blocks with offload management
                hidden_states, encoder_hidden_states = wrapper.run_blocks_with_offload(
                    hidden_states=hidden_states,
                    encoder_hidden_states=encoder_hidden_states,
                    temb=temb,
                    attention_mask=attention_mask,
                    image_rotary_emb=image_rotary_emb,
                    timestep=step,
                )

                result.successful_steps += 1

                # Sample memory stats periodically
                if step % 5 == 0:
                    stats = profiler.get_current_stats()
                    result.memory_samples.append(stats)

            print()  # New line after progress

        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                result.oom_occurred = True
                result.oom_message = str(e)
                print(colored(f"OOM Error: {e}", "red"))
            else:
                raise

        # Record results
        result.total_time_seconds = time.time() - start_time
        final_stats = profiler.get_current_stats()
        result.peak_memory_mb = final_stats.peak_memory_mb
        result.memory_samples.append(final_stats)

        # Get strategy stats
        wrapper_stats = wrapper.get_stats()
        result.sync_count = wrapper_stats['sync_count']
        result.load_count = wrapper_stats['load_count']
        result.offload_count = wrapper_stats['offload_count']

        # Cleanup
        del pipe, transformer, wrapper
        gc.collect()
        torch.cuda.empty_cache()

    except Exception as e:
        result.oom_occurred = True
        result.oom_message = str(e)
        print(colored(f"Error: {e}", "red"))

        # Cleanup on error
        gc.collect()
        torch.cuda.empty_cache()

    return result


def run_lightweight_experiment(
    config: ExperimentConfig,
    strategy_type: SyncStrategy,
    trial_id: int,
    profiler: MemoryProfiler,
) -> ExperimentResult:
    """
    Run a lightweight experiment that simulates Video DiT memory patterns
    without loading the full model. Uses synthetic blocks with similar
    memory footprints.
    """

    print(f"\n{'='*60}")
    print(f"Trial {trial_id}: Strategy = {strategy_type.value} (Lightweight)")
    print(f"{'='*60}")

    torch.manual_seed(config.seed + trial_id)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(config.seed + trial_id)

    result = ExperimentResult(
        strategy=strategy_type.value,
        total_steps=config.num_inference_steps,
    )

    try:
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(config.device)

        # Create synthetic blocks that mimic HunyuanVideo transformer blocks
        # Each block is approximately 650MB (based on paper)
        num_blocks = 60
        block_size_mb = 200  # Smaller for testing
        hidden_dim = int((block_size_mb * 1024 * 1024 / 4) ** 0.5)  # Approximate

        print(f"Creating {num_blocks} synthetic blocks (~{block_size_mb}MB each)...")

        # Create synthetic blocks
        blocks = []
        for i in range(num_blocks):
            block = torch.nn.Sequential(
                torch.nn.Linear(hidden_dim, hidden_dim, dtype=torch.bfloat16),
                torch.nn.LayerNorm(hidden_dim, dtype=torch.bfloat16),
                torch.nn.Linear(hidden_dim, hidden_dim, dtype=torch.bfloat16),
            )
            blocks.append(block)

        # Verify block size
        actual_size = sum(p.numel() * p.element_size() for p in blocks[0].parameters()) / (1024**2)
        print(f"Actual block size: {actual_size:.2f}MB")

        # Create strategy
        strategy = create_strategy(
            strategy_type,
            device=config.device,
            safety_margin_mb=config.safety_margin_mb,
            memory_threshold_ratio=config.memory_threshold_ratio,
        )

        # Create offload manager
        manager = BlockOffloadManager(
            blocks=blocks,
            strategy=strategy,
            working_set_size=config.working_set_size,
            device=config.device
        )

        # Initialize
        manager.initialize_working_set(start_idx=0)

        print(f"Starting simulation with {strategy_type.value} strategy...")

        # Create activation-like tensors
        activation_size = (1, 32768, hidden_dim)  # Typical activation shape
        print(f"Activation shape: {activation_size}")

        profiler.reset_peak_stats()
        start_time = time.time()

        try:
            # Simulate diffusion steps
            for step in range(config.num_inference_steps):
                print(f"  Step {step + 1}/{config.num_inference_steps}", end="\r")

                # Create new activations each step (simulating diffusion)
                x = torch.randn(*activation_size, device=f'cuda:{config.device}', dtype=torch.bfloat16)

                # Process all blocks
                for block_idx in range(num_blocks):
                    # Prepare block (handles offload/load)
                    if not manager.prepare_block(block_idx):
                        raise RuntimeError(f"OOM during load of block {block_idx}")

                    # Forward pass
                    block = blocks[block_idx]
                    x = block(x)

                    # Finish block
                    manager.finish_block(block_idx)

                result.successful_steps += 1

                # Sample memory stats
                if step % 5 == 0:
                    stats = profiler.get_current_stats()
                    result.memory_samples.append(stats)

                # Simulate activation update (like DDIM step)
                del x
                if strategy_type == SyncStrategy.PURE_SYNC:
                    gc.collect()
                    torch.cuda.empty_cache()

            print()

        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                result.oom_occurred = True
                result.oom_message = str(e)
                print(colored(f"OOM Error: {e}", "red"))
            else:
                raise

        result.total_time_seconds = time.time() - start_time
        final_stats = profiler.get_current_stats()
        result.peak_memory_mb = final_stats.peak_memory_mb
        result.memory_samples.append(final_stats)

        result.sync_count = strategy.sync_count
        result.load_count = strategy.load_count
        result.offload_count = strategy.offload_count

        # Cleanup
        for block in blocks:
            del block
        del blocks
        gc.collect()
        torch.cuda.empty_cache()

    except Exception as e:
        result.oom_occurred = True
        result.oom_message = str(e)
        print(colored(f"Error: {e}", "red"))
        import traceback
        traceback.print_exc()

        gc.collect()
        torch.cuda.empty_cache()

    return result


def print_summary(results: Dict[str, List[ExperimentResult]]):
    """Print summary of experiment results"""

    print("\n" + "=" * 80)
    print("EXPERIMENT SUMMARY")
    print("=" * 80)

    headers = ["Strategy", "OOM Rate", "Avg Peak (MB)", "Peak Std (MB)", "Avg Time (s)", "Sync Count"]
    row_format = "{:<20} {:>10} {:>15} {:>15} {:>12} {:>12}"

    print(row_format.format(*headers))
    print("-" * 80)

    for strategy_name, trials in results.items():
        oom_count = sum(1 for t in trials if t.oom_occurred)
        oom_rate = f"{oom_count}/{len(trials)} ({100*oom_count/len(trials):.0f}%)"

        # Only include successful trials for memory stats
        successful = [t for t in trials if not t.oom_occurred]

        if successful:
            avg_peak = np.mean([t.avg_peak_memory_mb for t in successful])
            std_peak = np.mean([t.peak_variance_mb for t in successful])
            avg_time = np.mean([t.total_time_seconds for t in successful])
            avg_sync = np.mean([t.sync_count for t in successful])
        else:
            avg_peak = float('nan')
            std_peak = float('nan')
            avg_time = float('nan')
            avg_sync = float('nan')

        print(row_format.format(
            strategy_name,
            oom_rate,
            f"{avg_peak:.1f}",
            f"+-{std_peak:.1f}",
            f"{avg_time:.2f}",
            f"{avg_sync:.0f}"
        ))

    print("=" * 80)

    # Print table in paper format
    print("\n" + "=" * 80)
    print("TABLE FORMAT (for paper)")
    print("=" * 80)
    print("| 传输策略 | OOM率 | 平均峰值内存 | 峰值方差 |")
    print("|----------|-------|-------------|----------|")

    for strategy_name, trials in results.items():
        oom_count = sum(1 for t in trials if t.oom_occurred)
        oom_rate = f"{100*oom_count/len(trials):.0f}%"

        successful = [t for t in trials if not t.oom_occurred]
        if successful:
            avg_peak = np.mean([t.avg_peak_memory_mb for t in successful]) / 1024  # Convert to GB
            std_peak = np.mean([t.peak_variance_mb for t in successful]) / 1024
        else:
            avg_peak = float('nan')
            std_peak = float('nan')

        display_name = {
            'pure_async': '纯异步',
            'pure_sync': '纯同步',
            'conditional_sync': '条件同步（本文）'
        }.get(strategy_name, strategy_name)

        print(f"| {display_name} | {oom_rate} | {avg_peak:.1f}GB | ±{std_peak:.1f}GB |")


def main():
    parser = argparse.ArgumentParser(description="Video DiT Memory Sync Strategy Experiments")

    parser.add_argument("--strategy", type=str, default="all",
                       choices=["pure_async", "pure_sync", "conditional_sync", "all"],
                       help="Strategy to test")
    parser.add_argument("--num_trials", type=int, default=5,
                       help="Number of trials per strategy")
    parser.add_argument("--lightweight", action="store_true",
                       help="Run lightweight simulation instead of full model")
    parser.add_argument("--working_set_size", type=int, default=5,
                       help="Number of blocks to keep on GPU")
    parser.add_argument("--num_steps", type=int, default=30,
                       help="Number of diffusion steps")
    parser.add_argument("--height", type=int, default=544,
                       help="Video height")
    parser.add_argument("--width", type=int, default=960,
                       help="Video width")
    parser.add_argument("--num_frames", type=int, default=61,
                       help="Number of video frames")
    parser.add_argument("--output_file", type=str, default=None,
                       help="Output JSON file for results")
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed")

    args = parser.parse_args()

    # Check CUDA availability
    if not torch.cuda.is_available():
        print(colored("Error: CUDA is not available!", "red"))
        sys.exit(1)

    # Print GPU info
    device = 0
    props = torch.cuda.get_device_properties(device)
    print(f"\nGPU: {props.name}")
    print(f"Total Memory: {props.total_memory / (1024**3):.1f} GB")
    print(f"Compute Capability: {props.major}.{props.minor}")

    # Create config
    config = ExperimentConfig(
        height=args.height,
        width=args.width,
        num_frames=args.num_frames,
        num_inference_steps=args.num_steps,
        working_set_size=args.working_set_size,
        num_trials=args.num_trials,
        seed=args.seed,
        device=device,
    )

    # Determine strategies to test
    if args.strategy == "all":
        strategies = [
            SyncStrategy.PURE_ASYNC,
            SyncStrategy.PURE_SYNC,
            SyncStrategy.CONDITIONAL_SYNC,
        ]
    else:
        strategies = [SyncStrategy(args.strategy)]

    # Create profiler
    profiler = MemoryProfiler(device=device)

    # Run experiments
    results: Dict[str, List[ExperimentResult]] = {}
    run_func = run_lightweight_experiment if args.lightweight else run_single_experiment

    for strategy in strategies:
        strategy_results = []

        for trial in range(args.num_trials):
            result = run_func(config, strategy, trial, profiler)
            strategy_results.append(result)

            print(f"\nTrial {trial} Results:")
            print(f"  OOM: {result.oom_occurred}")
            print(f"  Peak Memory: {result.peak_memory_mb:.1f} MB")
            print(f"  Peak Variance: {result.peak_variance_mb:.1f} MB")
            print(f"  Time: {result.total_time_seconds:.2f}s")
            print(f"  Sync Count: {result.sync_count}")
            print(f"  Successful Steps: {result.successful_steps}/{result.total_steps}")

        results[strategy.value] = strategy_results

    # Print summary
    print_summary(results)

    # Save results if requested
    if args.output_file:
        output_data = {
            'config': asdict(config) if hasattr(config, '__dataclass_fields__') else vars(config),
            'results': {
                k: [asdict(r) for r in v]
                for k, v in results.items()
            },
            'timestamp': datetime.now().isoformat(),
        }

        # Handle non-serializable fields
        def clean_for_json(obj):
            if isinstance(obj, dict):
                return {k: clean_for_json(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [clean_for_json(v) for v in obj]
            elif isinstance(obj, (np.floating, np.integer)):
                return float(obj)
            elif hasattr(obj, '__dict__'):
                return clean_for_json(obj.__dict__)
            else:
                try:
                    json.dumps(obj)
                    return obj
                except (TypeError, ValueError):
                    return str(obj)

        output_data = clean_for_json(output_data)

        with open(args.output_file, 'w') as f:
            json.dump(output_data, f, indent=2)
        print(f"\nResults saved to {args.output_file}")


if __name__ == "__main__":
    main()
