#!/usr/bin/env python3
"""
Real Video Generation Test for Memory Sync Protocols

This script runs actual HunyuanVideo inference to test memory sync strategies
under realistic conditions. It generates real videos and measures OOM rates,
sync trigger rates, and performance.

Usage:
    python test_real_generation.py --model-path models/HunyuanVideo --strategy all
    python test_real_generation.py --model-path models/HunyuanVideo --strategy conditional_sync
"""

import argparse
import gc
import json
import os
import sys
import time
from dataclasses import dataclass, asdict, field
from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional, Tuple, Any, Callable
import functools

import torch
import torch.nn as nn

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))


class SyncStrategy(Enum):
    """Memory synchronization strategies"""
    PURE_ASYNC = "pure_async"
    PURE_SYNC = "pure_sync"
    CONDITIONAL_SYNC = "conditional_sync"


@dataclass
class TransferStats:
    """Statistics for block transfers"""
    total_loads: int = 0
    total_offloads: int = 0
    total_syncs: int = 0
    total_ooms: int = 0
    total_load_time_ms: float = 0.0
    total_offload_time_ms: float = 0.0


class BlockOffloadManager:
    """
    Manages block-level offloading for Video DiT transformers.

    This is a simplified version that directly wraps transformer blocks
    to enable GPU<->CPU transfers with different sync strategies.
    """

    def __init__(
        self,
        blocks: List[nn.Module],
        working_set_size: int = 5,
        strategy: SyncStrategy = SyncStrategy.CONDITIONAL_SYNC,
        device: int = 0,
        safety_margin_gb: float = 2.0,
    ):
        self.blocks = blocks
        self.num_blocks = len(blocks)
        self.working_set_size = working_set_size
        self.strategy = strategy
        self.device = torch.device(f'cuda:{device}')
        self.device_id = device
        self.safety_margin_gb = safety_margin_gb

        # Track block locations
        self.block_on_gpu: List[bool] = [True] * self.num_blocks

        # Stats
        self.stats = TransferStats()

        # For conditional sync
        self.recent_failures: List[float] = []
        self.failure_window = 5.0

        # CUDA streams for async operations
        self.load_stream = torch.cuda.Stream(device=device)
        self.offload_stream = torch.cuda.Stream(device=device)

        # Original forward functions
        self._original_forwards: Dict[int, Callable] = {}

    def get_free_memory_gb(self) -> float:
        """Get free GPU memory in GB"""
        props = torch.cuda.get_device_properties(self.device_id)
        reserved = torch.cuda.memory_reserved(self.device_id)
        return (props.total_memory - reserved) / (1024**3)

    def get_block_size_gb(self, block_idx: int) -> float:
        """Estimate block size in GB"""
        block = self.blocks[block_idx]
        total_bytes = sum(p.numel() * p.element_size() for p in block.parameters())
        total_bytes += sum(b.numel() * b.element_size() for b in block.buffers())
        return total_bytes / (1024**3)

    def should_sync(self, block_size_gb: float) -> Tuple[bool, str]:
        """Determine if synchronization is needed"""
        if self.strategy == SyncStrategy.PURE_ASYNC:
            return False, ""

        if self.strategy == SyncStrategy.PURE_SYNC:
            return True, "pure_sync"

        # Conditional sync logic
        free_gb = self.get_free_memory_gb()

        # Condition 1: Low memory
        if free_gb < block_size_gb + self.safety_margin_gb:
            return True, "low_memory"

        # Condition 2: Recent OOM
        now = time.time()
        self.recent_failures = [t for t in self.recent_failures if now - t < self.failure_window]
        if self.recent_failures:
            return True, "recent_failure"

        return False, ""

    def offload_block(self, block_idx: int, force_sync: bool = False) -> bool:
        """Move block from GPU to CPU"""
        if not self.block_on_gpu[block_idx]:
            return True

        block = self.blocks[block_idx]
        block_size_gb = self.get_block_size_gb(block_idx)
        should_sync, _ = self.should_sync(block_size_gb)
        should_sync = should_sync or force_sync

        start = time.time()

        try:
            if should_sync:
                torch.cuda.synchronize(self.device_id)
                self.stats.total_syncs += 1

            block.to('cpu')

            if should_sync:
                torch.cuda.synchronize(self.device_id)
                torch.cuda.empty_cache()

            self.block_on_gpu[block_idx] = False
            self.stats.total_offloads += 1
            self.stats.total_offload_time_ms += (time.time() - start) * 1000

            return True

        except RuntimeError as e:
            if 'out of memory' in str(e).lower():
                self.stats.total_ooms += 1
                self.recent_failures.append(time.time())
                torch.cuda.empty_cache()
                return False
            raise

    def load_block(self, block_idx: int) -> bool:
        """Move block from CPU to GPU"""
        if self.block_on_gpu[block_idx]:
            return True

        block = self.blocks[block_idx]
        block_size_gb = self.get_block_size_gb(block_idx)
        should_sync, reason = self.should_sync(block_size_gb)

        start = time.time()

        try:
            if should_sync:
                torch.cuda.synchronize(self.device_id)
                torch.cuda.empty_cache()
                self.stats.total_syncs += 1

            block.to(self.device)

            if should_sync:
                torch.cuda.synchronize(self.device_id)

            self.block_on_gpu[block_idx] = True
            self.stats.total_loads += 1
            self.stats.total_load_time_ms += (time.time() - start) * 1000

            return True

        except RuntimeError as e:
            if 'out of memory' in str(e).lower():
                self.stats.total_ooms += 1
                self.recent_failures.append(time.time())
                torch.cuda.empty_cache()
                return False
            raise

    def ensure_block_on_gpu(self, block_idx: int) -> bool:
        """Ensure block is on GPU, offloading others if needed"""
        if self.block_on_gpu[block_idx]:
            return True

        # Count blocks on GPU
        gpu_blocks = [i for i, on_gpu in enumerate(self.block_on_gpu) if on_gpu]

        # Offload oldest if at capacity
        while len(gpu_blocks) >= self.working_set_size:
            to_offload = gpu_blocks[0]
            if not self.offload_block(to_offload):
                return False
            gpu_blocks = gpu_blocks[1:]

        return self.load_block(block_idx)

    def initialize(self):
        """Move blocks beyond working set to CPU"""
        print(f"Initializing offloading: keeping {self.working_set_size} blocks on GPU")

        for i in range(self.working_set_size, self.num_blocks):
            self.offload_block(i, force_sync=True)

        torch.cuda.synchronize(self.device_id)
        torch.cuda.empty_cache()

        gpu_count = sum(self.block_on_gpu)
        cpu_count = self.num_blocks - gpu_count
        print(f"  {gpu_count} blocks on GPU, {cpu_count} blocks on CPU")
        print(f"  Free memory: {self.get_free_memory_gb():.2f} GB")

    def wrap_blocks(self):
        """Wrap block forward functions to enable offloading"""
        for idx, block in enumerate(self.blocks):
            self._original_forwards[idx] = block.forward
            block.forward = self._make_wrapped_forward(idx, block.forward)

    def unwrap_blocks(self):
        """Restore original forward functions"""
        for idx, block in enumerate(self.blocks):
            if idx in self._original_forwards:
                block.forward = self._original_forwards[idx]
        self._original_forwards.clear()

        # Move all blocks back to GPU
        for idx in range(self.num_blocks):
            if not self.block_on_gpu[idx]:
                self.blocks[idx].to(self.device)
                self.block_on_gpu[idx] = True

    def _make_wrapped_forward(self, block_idx: int, original_forward: Callable) -> Callable:
        """Create wrapped forward function"""
        @functools.wraps(original_forward)
        def wrapped(*args, **kwargs):
            if not self.ensure_block_on_gpu(block_idx):
                raise RuntimeError(f"Failed to load block {block_idx} to GPU")
            return original_forward(*args, **kwargs)
        return wrapped

    def get_sync_trigger_rate(self) -> float:
        """Calculate sync trigger rate"""
        total = self.stats.total_loads + self.stats.total_offloads
        if total == 0:
            return 0.0
        return self.stats.total_syncs / total


def load_hunyuan_model(model_path: str):
    """Load HunyuanVideo model"""
    from diffusers import (
        HunyuanVideoPipeline,
        HunyuanVideoTransformer3DModel,
        FlowMatchEulerDiscreteScheduler
    )

    print(f"Loading HunyuanVideo from {model_path}...")

    transformer = HunyuanVideoTransformer3DModel.from_pretrained(
        model_path,
        subfolder="transformer",
        torch_dtype=torch.bfloat16,
    )

    scheduler = FlowMatchEulerDiscreteScheduler(shift=7.0)

    pipe = HunyuanVideoPipeline.from_pretrained(
        model_path,
        transformer=transformer,
        scheduler=scheduler,
        torch_dtype=torch.bfloat16
    )

    pipe.vae.enable_tiling()
    pipe.to("cuda")

    # Get block counts
    num_double = len(list(pipe.transformer.transformer_blocks))
    num_single = len(list(pipe.transformer.single_transformer_blocks))
    print(f"Loaded: {num_double} double-stream + {num_single} single-stream = {num_double + num_single} total blocks")

    return pipe


def get_transformer_blocks(pipe) -> List[nn.Module]:
    """Extract all transformer blocks from pipeline"""
    blocks = []

    if hasattr(pipe.transformer, 'transformer_blocks'):
        blocks.extend(list(pipe.transformer.transformer_blocks))
    if hasattr(pipe.transformer, 'single_transformer_blocks'):
        blocks.extend(list(pipe.transformer.single_transformer_blocks))
    if hasattr(pipe.transformer, 'blocks'):
        blocks.extend(list(pipe.transformer.blocks))

    return blocks


@dataclass
class ExperimentResult:
    """Result from one generation attempt"""
    strategy: str
    success: bool
    oom_occurred: bool
    error: Optional[str]
    inference_time_s: float
    peak_memory_gb: float
    sync_trigger_rate: float
    stats: Dict


def run_generation(
    pipe,
    strategy: SyncStrategy,
    working_set_size: int,
    prompt: str,
    height: int,
    width: int,
    num_frames: int,
    num_inference_steps: int,
    output_path: Optional[str] = None,
) -> ExperimentResult:
    """Run a single video generation with offloading"""

    print(f"\n{'='*60}")
    print(f"Strategy: {strategy.value}")
    print(f"Working set: {working_set_size} blocks")
    print(f"Resolution: {height}x{width}, {num_frames} frames, {num_inference_steps} steps")
    print(f"{'='*60}")

    # Reset
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.empty_cache()
    gc.collect()

    # Get blocks
    blocks = get_transformer_blocks(pipe)
    print(f"Found {len(blocks)} transformer blocks")

    # Create manager
    manager = BlockOffloadManager(
        blocks=blocks,
        working_set_size=working_set_size,
        strategy=strategy,
        safety_margin_gb=2.0,
    )

    success = True
    oom_occurred = False
    error = None
    inference_time = 0.0

    try:
        # Initialize offloading
        manager.initialize()
        manager.wrap_blocks()

        print(f"Starting generation...")
        start = time.time()

        # Run generation
        output = pipe(
            prompt=prompt,
            negative_prompt="low quality, blurry, distorted",
            height=height,
            width=width,
            num_frames=num_frames,
            num_inference_steps=num_inference_steps,
            guidance_scale=6.0,
        )

        inference_time = time.time() - start
        print(f"Generation completed in {inference_time:.1f}s")

        # Save video if path provided
        if output_path:
            from diffusers.utils import export_to_video
            os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
            export_to_video(output.frames[0], output_path, fps=24)
            print(f"Saved video to {output_path}")

    except RuntimeError as e:
        if 'out of memory' in str(e).lower():
            oom_occurred = True
            success = False
            error = str(e)[:200]
            print(f"OOM: {error}")
        else:
            success = False
            error = str(e)
            print(f"Error: {error}")
    except Exception as e:
        success = False
        error = str(e)
        print(f"Error: {error}")
    finally:
        # Restore
        manager.unwrap_blocks()

    peak_memory = torch.cuda.max_memory_allocated() / (1024**3)

    # Print stats
    print(f"\nStats:")
    print(f"  Loads: {manager.stats.total_loads}")
    print(f"  Offloads: {manager.stats.total_offloads}")
    print(f"  Syncs: {manager.stats.total_syncs}")
    print(f"  OOMs: {manager.stats.total_ooms}")
    print(f"  Sync rate: {manager.get_sync_trigger_rate()*100:.1f}%")
    print(f"  Peak memory: {peak_memory:.2f} GB")

    return ExperimentResult(
        strategy=strategy.value,
        success=success,
        oom_occurred=oom_occurred,
        error=error,
        inference_time_s=inference_time,
        peak_memory_gb=peak_memory,
        sync_trigger_rate=manager.get_sync_trigger_rate(),
        stats=asdict(manager.stats),
    )


def run_comparison(
    model_path: str,
    strategies: List[SyncStrategy],
    working_set_size: int = 5,
    prompt: str = "A cat walking in a garden, high quality, realistic",
    height: int = 480,
    width: int = 848,
    num_frames: int = 49,
    num_inference_steps: int = 30,
    num_trials: int = 1,
    output_dir: str = "./results/real_generation",
) -> Dict:
    """Run comparison across strategies"""

    print(f"\n{'='*80}")
    print("REAL VIDEO GENERATION - MEMORY SYNC COMPARISON")
    print(f"{'='*80}")
    print(f"Model: {model_path}")
    print(f"Resolution: {height}x{width}, {num_frames} frames")
    print(f"Inference steps: {num_inference_steps}")
    print(f"Working set: {working_set_size} blocks")
    print(f"Strategies: {[s.value for s in strategies]}")
    print(f"Trials: {num_trials}")

    # Load model
    pipe = load_hunyuan_model(model_path)

    results = {}

    for strategy in strategies:
        strategy_results = []

        for trial in range(num_trials):
            print(f"\n{'='*80}")
            print(f"Trial {trial+1}/{num_trials} for {strategy.value}")
            print(f"{'='*80}")

            video_path = os.path.join(
                output_dir,
                f"video_{strategy.value}_trial{trial+1}.mp4"
            ) if output_dir else None

            result = run_generation(
                pipe=pipe,
                strategy=strategy,
                working_set_size=working_set_size,
                prompt=prompt,
                height=height,
                width=width,
                num_frames=num_frames,
                num_inference_steps=num_inference_steps,
                output_path=video_path,
            )

            strategy_results.append(asdict(result))

            # Cool down
            torch.cuda.empty_cache()
            gc.collect()
            time.sleep(2)

        results[strategy.value] = strategy_results

    # Summary
    print(f"\n{'='*80}")
    print("SUMMARY")
    print(f"{'='*80}")
    print(f"{'Strategy':<20} {'Success':<10} {'OOM':<8} {'Sync Rate':<12} {'Time':<10} {'Peak Mem':<12}")
    print(f"{'-'*72}")

    summary = {}
    for strategy, trials in results.items():
        success_rate = sum(1 for t in trials if t['success']) / len(trials)
        oom_rate = sum(1 for t in trials if t['oom_occurred']) / len(trials)
        avg_sync = sum(t['sync_trigger_rate'] for t in trials) / len(trials)
        avg_time = sum(t['inference_time_s'] for t in trials if t['success']) / max(1, sum(1 for t in trials if t['success']))
        avg_mem = sum(t['peak_memory_gb'] for t in trials) / len(trials)

        summary[strategy] = {
            'success_rate': success_rate,
            'oom_rate': oom_rate,
            'avg_sync_rate': avg_sync,
            'avg_time': avg_time,
            'avg_peak_memory': avg_mem,
        }

        print(f"{strategy:<20} {success_rate*100:.0f}%       {oom_rate*100:.0f}%     {avg_sync*100:.1f}%        {avg_time:.1f}s      {avg_mem:.2f} GB")

    # Save results
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        result_file = os.path.join(output_dir, f"results_{timestamp}.json")

        with open(result_file, 'w') as f:
            json.dump({
                'timestamp': timestamp,
                'config': {
                    'model_path': model_path,
                    'height': height,
                    'width': width,
                    'num_frames': num_frames,
                    'num_inference_steps': num_inference_steps,
                    'working_set_size': working_set_size,
                    'num_trials': num_trials,
                },
                'results': results,
                'summary': summary,
            }, f, indent=2)

        print(f"\nResults saved to {result_file}")

    return {'results': results, 'summary': summary}


def main():
    parser = argparse.ArgumentParser(description="Real video generation test")

    parser.add_argument("--model-path", type=str, required=True,
                        help="Path to HunyuanVideo model")
    parser.add_argument("--strategy", type=str, default="all",
                        choices=["pure_async", "pure_sync", "conditional_sync", "all"])
    parser.add_argument("--working-set-size", type=int, default=5)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--width", type=int, default=848)
    parser.add_argument("--num-frames", type=int, default=49)
    parser.add_argument("--num-inference-steps", type=int, default=30)
    parser.add_argument("--num-trials", type=int, default=1)
    parser.add_argument("--prompt", type=str,
                        default="A cat walking in a beautiful garden, high quality")
    parser.add_argument("--output-dir", type=str, default="./results/real_generation")

    args = parser.parse_args()

    if not torch.cuda.is_available():
        print("CUDA not available")
        sys.exit(1)

    # GPU info
    props = torch.cuda.get_device_properties(0)
    print(f"GPU: {props.name}, {props.total_memory / (1024**3):.1f} GB")

    # Strategies
    if args.strategy == "all":
        strategies = [
            SyncStrategy.PURE_ASYNC,
            SyncStrategy.PURE_SYNC,
            SyncStrategy.CONDITIONAL_SYNC,
        ]
    else:
        strategies = [SyncStrategy(args.strategy)]

    run_comparison(
        model_path=args.model_path,
        strategies=strategies,
        working_set_size=args.working_set_size,
        prompt=args.prompt,
        height=args.height,
        width=args.width,
        num_frames=args.num_frames,
        num_inference_steps=args.num_inference_steps,
        num_trials=args.num_trials,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()
