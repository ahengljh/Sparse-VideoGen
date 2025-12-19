#!/usr/bin/env python3
"""
Memory Sync Protocol Experiments with Real Video DiT

This script runs controlled experiments comparing sync strategies under
different memory pressure conditions using actual HunyuanVideo generation.

Experiment Design:
- Vary memory pressure by changing working_set_size (fewer blocks on GPU = more transfers = more pressure)
- Vary memory pressure by adding background tensors
- Compare pure_async, pure_sync, conditional_sync strategies
- Measure OOM rate, sync trigger rate, generation time

Usage:
    python run_memory_experiments.py --model-path models/HunyuanVideo --experiment all
    python run_memory_experiments.py --model-path models/HunyuanVideo --experiment pressure_levels
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

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))


class SyncStrategy(Enum):
    PURE_ASYNC = "pure_async"
    PURE_SYNC = "pure_sync"
    CONDITIONAL_SYNC = "conditional_sync"


@dataclass
class TransferStats:
    total_loads: int = 0
    total_offloads: int = 0
    total_syncs: int = 0
    total_ooms: int = 0


@dataclass
class ExperimentResult:
    experiment_name: str
    strategy: str
    working_set_size: int
    background_memory_gb: float
    success: bool
    oom_occurred: bool
    error: Optional[str]
    inference_time_s: float
    peak_memory_gb: float
    sync_trigger_rate: float
    stats: Dict


class BlockOffloadManager:
    """Manages block-level GPU/CPU transfers with different sync strategies"""

    def __init__(
        self,
        blocks: List[nn.Module],
        working_set_size: int,
        strategy: SyncStrategy,
        device: int = 0,
        safety_margin_gb: float = 1.5,
    ):
        self.blocks = blocks
        self.num_blocks = len(blocks)
        self.working_set_size = working_set_size
        self.strategy = strategy
        self.device = torch.device(f'cuda:{device}')
        self.device_id = device
        self.safety_margin_gb = safety_margin_gb

        self.block_on_gpu = [False] * self.num_blocks
        self.stats = TransferStats()
        self.recent_failures: List[float] = []
        self._original_forwards: Dict[int, Callable] = {}

    def get_free_memory_gb(self) -> float:
        props = torch.cuda.get_device_properties(self.device_id)
        reserved = torch.cuda.memory_reserved(self.device_id)
        return (props.total_memory - reserved) / (1024**3)

    def get_block_size_gb(self, block_idx: int) -> float:
        block = self.blocks[block_idx]
        total_bytes = sum(p.numel() * p.element_size() for p in block.parameters())
        return total_bytes / (1024**3)

    def should_sync(self, block_size_gb: float) -> Tuple[bool, str]:
        if self.strategy == SyncStrategy.PURE_ASYNC:
            return False, ""
        if self.strategy == SyncStrategy.PURE_SYNC:
            return True, "pure_sync"

        # Conditional sync
        free_gb = self.get_free_memory_gb()
        if free_gb < block_size_gb + self.safety_margin_gb:
            return True, "low_memory"

        now = time.time()
        self.recent_failures = [t for t in self.recent_failures if now - t < 5.0]
        if self.recent_failures:
            return True, "recent_failure"

        return False, ""

    def offload_block(self, block_idx: int, force_sync: bool = False) -> bool:
        if not self.block_on_gpu[block_idx]:
            return True

        block = self.blocks[block_idx]
        block_size_gb = self.get_block_size_gb(block_idx)
        should_sync, _ = self.should_sync(block_size_gb)
        should_sync = should_sync or force_sync

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
            return True

        except RuntimeError as e:
            if 'out of memory' in str(e).lower():
                self.stats.total_ooms += 1
                self.recent_failures.append(time.time())
                torch.cuda.empty_cache()
                return False
            raise

    def load_block(self, block_idx: int) -> bool:
        if self.block_on_gpu[block_idx]:
            return True

        block = self.blocks[block_idx]
        block_size_gb = self.get_block_size_gb(block_idx)
        should_sync, _ = self.should_sync(block_size_gb)

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
            return True

        except RuntimeError as e:
            if 'out of memory' in str(e).lower():
                self.stats.total_ooms += 1
                self.recent_failures.append(time.time())
                torch.cuda.empty_cache()
                return False
            raise

    def ensure_block_on_gpu(self, block_idx: int) -> bool:
        if self.block_on_gpu[block_idx]:
            return True

        gpu_blocks = [i for i, on_gpu in enumerate(self.block_on_gpu) if on_gpu]

        while len(gpu_blocks) >= self.working_set_size:
            to_offload = gpu_blocks[0]
            if not self.offload_block(to_offload):
                return False
            gpu_blocks = gpu_blocks[1:]

        return self.load_block(block_idx)

    def initialize(self):
        """Move working set to GPU"""
        self.block_on_gpu = [False] * self.num_blocks

        for i in range(min(self.working_set_size, self.num_blocks)):
            self.blocks[i].to(self.device)
            self.block_on_gpu[i] = True
            torch.cuda.synchronize(self.device_id)

        torch.cuda.empty_cache()

    def wrap_blocks(self):
        for idx, block in enumerate(self.blocks):
            self._original_forwards[idx] = block.forward
            block.forward = self._make_wrapped_forward(idx, block.forward)

    def unwrap_blocks(self):
        for idx, block in enumerate(self.blocks):
            if idx in self._original_forwards:
                block.forward = self._original_forwards[idx]
        self._original_forwards.clear()

        for idx in range(self.num_blocks):
            if self.block_on_gpu[idx]:
                self.blocks[idx].to('cpu')
                self.block_on_gpu[idx] = False
        torch.cuda.empty_cache()

    def _make_wrapped_forward(self, block_idx: int, original_forward: Callable) -> Callable:
        @functools.wraps(original_forward)
        def wrapped(*args, **kwargs):
            if not self.ensure_block_on_gpu(block_idx):
                raise RuntimeError(f"Failed to load block {block_idx}")
            return original_forward(*args, **kwargs)
        return wrapped

    def get_sync_rate(self) -> float:
        total = self.stats.total_loads + self.stats.total_offloads
        return self.stats.total_syncs / total if total > 0 else 0.0


def load_model(model_path: str):
    """Load HunyuanVideo with transformer on CPU"""
    from diffusers import (
        HunyuanVideoPipeline,
        HunyuanVideoTransformer3DModel,
        FlowMatchEulerDiscreteScheduler
    )

    print(f"Loading model from {model_path}...")

    transformer = HunyuanVideoTransformer3DModel.from_pretrained(
        model_path, subfolder="transformer", torch_dtype=torch.bfloat16
    )

    pipe = HunyuanVideoPipeline.from_pretrained(
        model_path,
        transformer=transformer,
        scheduler=FlowMatchEulerDiscreteScheduler(shift=7.0),
        torch_dtype=torch.bfloat16
    )

    pipe.vae.enable_tiling()
    pipe.text_encoder.to("cuda")
    pipe.text_encoder_2.to("cuda")
    pipe.vae.to("cuda")

    blocks = list(pipe.transformer.transformer_blocks) + list(pipe.transformer.single_transformer_blocks)
    print(f"Loaded {len(blocks)} transformer blocks (kept on CPU)")

    return pipe, blocks


def create_background_memory(target_gb: float) -> List[torch.Tensor]:
    """Allocate background tensors to create memory pressure"""
    tensors = []
    if target_gb <= 0:
        return tensors

    print(f"Creating {target_gb:.1f} GB background memory pressure...")

    chunk_elements = 256 * 1024 * 1024  # ~512MB per tensor in bf16
    chunks_needed = int(target_gb * 2)  # 2 chunks per GB

    for _ in range(chunks_needed):
        try:
            t = torch.zeros(chunk_elements, dtype=torch.bfloat16, device='cuda')
            tensors.append(t)
        except RuntimeError:
            break

    actual_gb = sum(t.numel() * 2 for t in tensors) / (1024**3)
    print(f"  Allocated {actual_gb:.2f} GB")
    return tensors


def run_single_generation(
    pipe,
    blocks: List[nn.Module],
    strategy: SyncStrategy,
    working_set_size: int,
    background_memory_gb: float = 0.0,
    height: int = 320,
    width: int = 512,
    num_frames: int = 17,
    num_inference_steps: int = 20,
    prompt: str = "A cat walking",
) -> ExperimentResult:
    """Run single generation with given configuration"""

    torch.cuda.reset_peak_memory_stats()
    torch.cuda.empty_cache()
    gc.collect()

    # Create background memory pressure
    bg_tensors = create_background_memory(background_memory_gb)

    # Create manager
    manager = BlockOffloadManager(
        blocks=blocks,
        working_set_size=working_set_size,
        strategy=strategy,
    )

    success = True
    oom = False
    error = None
    gen_time = 0.0

    try:
        print(f"  Initializing ({working_set_size} blocks to GPU)...")
        manager.initialize()
        manager.wrap_blocks()

        free_mem = manager.get_free_memory_gb()
        print(f"  Free memory before generation: {free_mem:.2f} GB")

        print(f"  Running generation...")
        start = time.time()

        output = pipe(
            prompt=prompt,
            negative_prompt="low quality",
            height=height,
            width=width,
            num_frames=num_frames,
            num_inference_steps=num_inference_steps,
            guidance_scale=6.0,
        )

        gen_time = time.time() - start
        print(f"  Generation completed in {gen_time:.1f}s")

    except RuntimeError as e:
        if 'out of memory' in str(e).lower():
            oom = True
            success = False
            error = "OOM"
            print(f"  OOM occurred!")
        else:
            success = False
            error = str(e)[:100]
            print(f"  Error: {error}")
    except Exception as e:
        success = False
        error = str(e)[:100]
        print(f"  Error: {error}")
    finally:
        manager.unwrap_blocks()
        del bg_tensors
        torch.cuda.empty_cache()

    peak_mem = torch.cuda.max_memory_allocated() / (1024**3)

    return ExperimentResult(
        experiment_name="",
        strategy=strategy.value,
        working_set_size=working_set_size,
        background_memory_gb=background_memory_gb,
        success=success,
        oom_occurred=oom,
        error=error,
        inference_time_s=gen_time,
        peak_memory_gb=peak_mem,
        sync_trigger_rate=manager.get_sync_rate(),
        stats=asdict(manager.stats),
    )


def run_working_set_experiment(
    pipe,
    blocks: List[nn.Module],
    output_dir: str,
) -> Dict:
    """
    Experiment 1: Vary working set size

    Tests how different working set sizes affect OOM rates across strategies.
    Smaller working set = more block transfers = more memory pressure
    """
    print("\n" + "="*80)
    print("EXPERIMENT 1: Working Set Size Variation")
    print("="*80)

    working_set_sizes = [3, 5, 8, 12, 20]
    strategies = [SyncStrategy.PURE_ASYNC, SyncStrategy.PURE_SYNC, SyncStrategy.CONDITIONAL_SYNC]

    results = []

    for ws_size in working_set_sizes:
        for strategy in strategies:
            print(f"\n--- Working set: {ws_size}, Strategy: {strategy.value} ---")

            result = run_single_generation(
                pipe=pipe,
                blocks=blocks,
                strategy=strategy,
                working_set_size=ws_size,
                height=320,
                width=512,
                num_frames=17,
                num_inference_steps=20,
            )
            result.experiment_name = "working_set_variation"
            results.append(asdict(result))

            torch.cuda.empty_cache()
            gc.collect()
            time.sleep(2)

    # Summary
    print("\n" + "="*80)
    print("WORKING SET EXPERIMENT SUMMARY")
    print("="*80)
    print(f"{'WS Size':<10} {'Strategy':<18} {'Success':<10} {'Sync Rate':<12} {'Time':<10} {'Peak Mem':<12}")
    print("-"*72)

    for r in results:
        print(f"{r['working_set_size']:<10} {r['strategy']:<18} "
              f"{'Yes' if r['success'] else 'OOM':<10} "
              f"{r['sync_trigger_rate']*100:.1f}%{'':<8} "
              f"{r['inference_time_s']:.1f}s{'':<6} "
              f"{r['peak_memory_gb']:.2f} GB")

    return {"experiment": "working_set_variation", "results": results}


def run_memory_pressure_experiment(
    pipe,
    blocks: List[nn.Module],
    output_dir: str,
) -> Dict:
    """
    Experiment 2: Vary background memory pressure

    Adds background tensors to simulate real-world memory constraints.
    """
    print("\n" + "="*80)
    print("EXPERIMENT 2: Background Memory Pressure")
    print("="*80)

    pressure_levels = [0.0, 4.0, 8.0, 12.0]  # GB of background memory
    strategies = [SyncStrategy.PURE_ASYNC, SyncStrategy.PURE_SYNC, SyncStrategy.CONDITIONAL_SYNC]
    working_set_size = 5

    results = []

    for pressure in pressure_levels:
        for strategy in strategies:
            print(f"\n--- Pressure: {pressure} GB, Strategy: {strategy.value} ---")

            result = run_single_generation(
                pipe=pipe,
                blocks=blocks,
                strategy=strategy,
                working_set_size=working_set_size,
                background_memory_gb=pressure,
                height=320,
                width=512,
                num_frames=17,
                num_inference_steps=20,
            )
            result.experiment_name = "memory_pressure"
            results.append(asdict(result))

            torch.cuda.empty_cache()
            gc.collect()
            time.sleep(2)

    # Summary
    print("\n" + "="*80)
    print("MEMORY PRESSURE EXPERIMENT SUMMARY")
    print("="*80)
    print(f"{'Pressure':<12} {'Strategy':<18} {'Success':<10} {'Sync Rate':<12} {'Time':<10} {'Peak Mem':<12}")
    print("-"*74)

    for r in results:
        print(f"{r['background_memory_gb']:.1f} GB{'':<5} {r['strategy']:<18} "
              f"{'Yes' if r['success'] else 'OOM':<10} "
              f"{r['sync_trigger_rate']*100:.1f}%{'':<8} "
              f"{r['inference_time_s']:.1f}s{'':<6} "
              f"{r['peak_memory_gb']:.2f} GB")

    return {"experiment": "memory_pressure", "results": results}


def run_resolution_experiment(
    pipe,
    blocks: List[nn.Module],
    output_dir: str,
) -> Dict:
    """
    Experiment 3: Vary resolution

    Higher resolution = more activation memory = more pressure on offloading
    """
    print("\n" + "="*80)
    print("EXPERIMENT 3: Resolution Variation")
    print("="*80)

    resolutions = [
        (256, 384, 9),    # Low
        (320, 512, 17),   # Medium
        (480, 720, 25),   # High
    ]
    strategies = [SyncStrategy.PURE_ASYNC, SyncStrategy.PURE_SYNC, SyncStrategy.CONDITIONAL_SYNC]
    working_set_size = 5

    results = []

    for height, width, frames in resolutions:
        for strategy in strategies:
            print(f"\n--- Resolution: {height}x{width}x{frames}, Strategy: {strategy.value} ---")

            result = run_single_generation(
                pipe=pipe,
                blocks=blocks,
                strategy=strategy,
                working_set_size=working_set_size,
                height=height,
                width=width,
                num_frames=frames,
                num_inference_steps=15,
            )
            result.experiment_name = "resolution"
            results.append(asdict(result))

            torch.cuda.empty_cache()
            gc.collect()
            time.sleep(2)

    # Summary
    print("\n" + "="*80)
    print("RESOLUTION EXPERIMENT SUMMARY")
    print("="*80)
    print(f"{'Resolution':<18} {'Strategy':<18} {'Success':<10} {'Sync Rate':<12} {'Peak Mem':<12}")
    print("-"*70)

    for r in results:
        # Extract resolution from result - need to track this
        print(f"{'See above':<18} {r['strategy']:<18} "
              f"{'Yes' if r['success'] else 'OOM':<10} "
              f"{r['sync_trigger_rate']*100:.1f}%{'':<8} "
              f"{r['peak_memory_gb']:.2f} GB")

    return {"experiment": "resolution", "results": results}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, required=True)
    parser.add_argument("--experiment", type=str, default="all",
                        choices=["working_set", "pressure", "resolution", "all"])
    parser.add_argument("--output-dir", type=str, default="./results/experiments")

    args = parser.parse_args()

    if not torch.cuda.is_available():
        print("CUDA required")
        sys.exit(1)

    props = torch.cuda.get_device_properties(0)
    print(f"GPU: {props.name}, {props.total_memory/(1024**3):.1f} GB")

    # Load model
    pipe, blocks = load_model(args.model_path)

    all_results = {}

    if args.experiment in ["working_set", "all"]:
        result = run_working_set_experiment(pipe, blocks, args.output_dir)
        all_results["working_set"] = result

    if args.experiment in ["pressure", "all"]:
        result = run_memory_pressure_experiment(pipe, blocks, args.output_dir)
        all_results["pressure"] = result

    if args.experiment in ["resolution", "all"]:
        result = run_resolution_experiment(pipe, blocks, args.output_dir)
        all_results["resolution"] = result

    # Save all results
    os.makedirs(args.output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = os.path.join(args.output_dir, f"experiments_{timestamp}.json")

    with open(output_file, 'w') as f:
        json.dump(all_results, f, indent=2)

    print(f"\nResults saved to {output_file}")


if __name__ == "__main__":
    main()
