#!/usr/bin/env python3
"""
Memory Stress Test for Video DiT Sync Protocols

This creates realistic memory pressure patterns matching Video DiT workloads
without requiring actual model downloads. Uses real GPU tensors and PyTorch
operations to demonstrate the sync protocol effectiveness.

The test simulates:
- Block sizes matching HunyuanVideo (~650MB per block)
- Working set management (keeping N blocks on GPU)
- Activation memory patterns during forward passes
- Sequential block traversal (40 steps x 60 blocks = 2400 accesses)

Usage:
    python memory_stress_test.py --strategy all
    python memory_stress_test.py --strategy conditional_sync --num-trials 5
"""

import argparse
import gc
import json
import os
import sys
import time
from dataclasses import dataclass, asdict
from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional, Tuple, Any

import torch
import torch.nn as nn


class SyncStrategy(Enum):
    """Memory synchronization strategies"""
    PURE_ASYNC = "pure_async"
    PURE_SYNC = "pure_sync"
    CONDITIONAL_SYNC = "conditional_sync"


@dataclass
class BlockTransferStats:
    """Statistics for block transfers"""
    load_count: int = 0
    offload_count: int = 0
    sync_count: int = 0
    oom_count: int = 0
    load_time_ms: float = 0.0
    offload_time_ms: float = 0.0


@dataclass
class TrialResult:
    """Result from a single trial"""
    strategy: str
    success: bool
    oom_occurred: bool
    oom_at_step: Optional[int]
    oom_at_block: Optional[int]
    peak_memory_gb: float
    total_time_s: float
    sync_trigger_rate: float
    stats: Dict


class SimulatedBlock(nn.Module):
    """
    Simulated transformer block with realistic memory footprint.

    Each block contains:
    - Attention layers (~200MB)
    - FFN layers (~400MB)
    - LayerNorm and other (~50MB)

    Total: ~650MB per block (similar to HunyuanVideo blocks)
    """

    def __init__(self, hidden_size: int = 3072, intermediate_size: int = 12288):
        super().__init__()

        # Attention-like weights
        self.q_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.k_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.v_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.o_proj = nn.Linear(hidden_size, hidden_size, bias=False)

        # FFN-like weights
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)

        # Norms
        self.norm1 = nn.LayerNorm(hidden_size)
        self.norm2 = nn.LayerNorm(hidden_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Simplified forward for testing
        residual = x
        x = self.norm1(x)
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)
        # Simplified attention (just for memory patterns)
        x = self.o_proj(v)
        x = residual + x

        residual = x
        x = self.norm2(x)
        x = self.gate_proj(x) * self.up_proj(x)
        x = self.down_proj(x)
        x = residual + x

        return x

    def get_memory_size_gb(self) -> float:
        """Calculate total memory size in GB"""
        total = 0
        for p in self.parameters():
            total += p.numel() * p.element_size()
        return total / (1024**3)


class BlockOffloadSimulator:
    """
    Simulates block offloading with different sync strategies.

    This tests the memory management patterns without requiring real model inference.
    """

    def __init__(
        self,
        num_blocks: int = 60,
        working_set_size: int = 5,
        strategy: SyncStrategy = SyncStrategy.CONDITIONAL_SYNC,
        block_size_mb: int = 650,
        device: int = 0,
    ):
        self.num_blocks = num_blocks
        self.working_set_size = working_set_size
        self.strategy = strategy
        self.device = torch.device(f'cuda:{device}')
        self.device_id = device

        # Calculate hidden size to achieve target block size
        # Each block has ~2 * hidden_size^2 + 6 * hidden_size * 4*hidden_size parameters
        # ≈ 26 * hidden_size^2 parameters (bfloat16 = 2 bytes)
        target_params = (block_size_mb * 1024 * 1024) // 2  # bytes / 2 for bf16
        hidden_size = int((target_params / 26) ** 0.5)
        # Round to multiple of 64 for efficiency
        hidden_size = (hidden_size // 64) * 64
        hidden_size = max(hidden_size, 256)  # Minimum size

        self.hidden_size = hidden_size

        # Create blocks
        self.blocks: List[SimulatedBlock] = []
        self.block_locations: List[str] = []  # 'gpu' or 'cpu'

        # CUDA streams
        self.load_stream = torch.cuda.Stream(device=device)
        self.offload_stream = torch.cuda.Stream(device=device)

        # Statistics
        self.stats = BlockTransferStats()

        # Conditional sync state
        self.recent_failures: List[float] = []
        self.failure_window = 5.0
        self.safety_margin_gb = 1.0  # Reduced for tighter memory scenarios

    def initialize(self):
        """Create and place blocks according to working set"""
        print(f"Initializing {self.num_blocks} blocks (hidden_size={self.hidden_size})")

        # Clear any existing blocks
        self.blocks = []
        self.block_locations = []
        torch.cuda.empty_cache()
        gc.collect()

        mem_before = torch.cuda.memory_allocated(self.device_id) / (1024**3)
        print(f"  Memory before init: {mem_before:.2f} GB")

        # Create blocks - always create on CPU first to avoid OOM during init
        for i in range(self.num_blocks):
            # Create on CPU first
            block = SimulatedBlock(
                hidden_size=self.hidden_size,
                intermediate_size=self.hidden_size * 4
            ).to(dtype=torch.bfloat16, device='cpu')

            self.blocks.append(block)
            self.block_locations.append('cpu')

            if (i + 1) % 20 == 0:
                print(f"  Created {i + 1}/{self.num_blocks} blocks on CPU")

        # Now move working set to GPU
        print(f"  Moving working set ({self.working_set_size} blocks) to GPU...")
        for i in range(self.working_set_size):
            self.blocks[i] = self.blocks[i].to(self.device)
            self.block_locations[i] = 'gpu'
            torch.cuda.synchronize(self.device_id)

        # Report memory
        block_size = self.blocks[0].get_memory_size_gb()
        mem_after = torch.cuda.memory_allocated(self.device_id) / (1024**3)
        print(f"  Block size: {block_size * 1024:.1f} MB")
        print(f"  Working set on GPU: {self.working_set_size} blocks = {block_size * self.working_set_size:.2f} GB")
        print(f"  Memory after init: {mem_after:.2f} GB")
        print(f"  Free memory: {self._get_free_memory_gb():.2f} GB")

        torch.cuda.synchronize()

    def _get_free_memory_gb(self) -> float:
        """Get current free GPU memory"""
        props = torch.cuda.get_device_properties(self.device_id)
        reserved = torch.cuda.memory_reserved(self.device_id)
        return (props.total_memory - reserved) / (1024**3)

    def _should_sync(self, block_size_gb: float) -> Tuple[bool, str]:
        """Determine if sync is needed based on strategy"""
        if self.strategy == SyncStrategy.PURE_ASYNC:
            return False, ""

        if self.strategy == SyncStrategy.PURE_SYNC:
            return True, "pure_sync"

        # Conditional sync logic
        free_gb = self._get_free_memory_gb()

        # Condition 1: Memory margin insufficient
        if free_gb < block_size_gb + self.safety_margin_gb:
            return True, "low_memory"

        # Condition 2: Recent failures
        now = time.time()
        self.recent_failures = [t for t in self.recent_failures if now - t < self.failure_window]
        if self.recent_failures:
            return True, "recent_failure"

        return False, ""

    def _offload_block(self, block_idx: int, force_sync: bool = False) -> bool:
        """Offload a block from GPU to CPU"""
        if self.block_locations[block_idx] == 'cpu':
            return True

        block = self.blocks[block_idx]
        block_size_gb = block.get_memory_size_gb()

        should_sync, reason = self._should_sync(block_size_gb)
        should_sync = should_sync or force_sync

        start_time = time.time()

        try:
            if should_sync:
                # Sync before offload
                torch.cuda.synchronize(self.device_id)
                torch.cuda.empty_cache()
                self.stats.sync_count += 1

            # Move to CPU
            if should_sync or self.strategy == SyncStrategy.PURE_SYNC:
                self.blocks[block_idx] = block.to('cpu')
                torch.cuda.synchronize(self.device_id)
            else:
                # Async offload
                with torch.cuda.stream(self.offload_stream):
                    self.blocks[block_idx] = block.to('cpu')

            self.block_locations[block_idx] = 'cpu'

            elapsed = (time.time() - start_time) * 1000
            self.stats.offload_count += 1
            self.stats.offload_time_ms += elapsed

            return True

        except RuntimeError as e:
            if 'out of memory' in str(e).lower():
                self.stats.oom_count += 1
                self.recent_failures.append(time.time())
                torch.cuda.empty_cache()
                return False
            raise

    def _load_block(self, block_idx: int) -> bool:
        """Load a block from CPU to GPU"""
        if self.block_locations[block_idx] == 'gpu':
            return True

        block = self.blocks[block_idx]
        block_size_gb = block.get_memory_size_gb()

        should_sync, reason = self._should_sync(block_size_gb)

        start_time = time.time()

        try:
            if should_sync:
                torch.cuda.synchronize(self.device_id)
                torch.cuda.empty_cache()
                self.stats.sync_count += 1

            if should_sync or self.strategy == SyncStrategy.PURE_SYNC:
                self.blocks[block_idx] = block.to(self.device)
                torch.cuda.synchronize(self.device_id)
            else:
                with torch.cuda.stream(self.load_stream):
                    self.blocks[block_idx] = block.to(self.device)

            self.block_locations[block_idx] = 'gpu'

            elapsed = (time.time() - start_time) * 1000
            self.stats.load_count += 1
            self.stats.load_time_ms += elapsed

            return True

        except RuntimeError as e:
            if 'out of memory' in str(e).lower():
                self.stats.oom_count += 1
                self.recent_failures.append(time.time())
                torch.cuda.empty_cache()
                return False
            raise

    def ensure_block_on_gpu(self, block_idx: int) -> bool:
        """Ensure block is on GPU, offloading others if needed"""
        if self.block_locations[block_idx] == 'gpu':
            return True

        # Find GPU blocks to offload (FIFO)
        gpu_blocks = [i for i, loc in enumerate(self.block_locations) if loc == 'gpu']

        while len(gpu_blocks) >= self.working_set_size:
            oldest = gpu_blocks[0]
            if not self._offload_block(oldest):
                return False
            gpu_blocks.remove(oldest)

        # Load requested block
        return self._load_block(block_idx)

    def run_forward(self, block_idx: int, x: torch.Tensor) -> torch.Tensor:
        """Run forward pass on a block"""
        # Ensure block is on GPU
        if not self.ensure_block_on_gpu(block_idx):
            return None

        # For pure async, sync load stream before using
        if self.strategy == SyncStrategy.PURE_ASYNC:
            self.load_stream.synchronize()

        return self.blocks[block_idx](x)


def run_stress_trial(
    num_blocks: int,
    num_steps: int,
    working_set_size: int,
    strategy: SyncStrategy,
    block_size_mb: int,
    batch_size: int = 1,
    seq_len: int = 1024,
) -> TrialResult:
    """Run a single stress test trial"""

    print(f"\n{'='*60}")
    print(f"STRESS TEST: {strategy.value}")
    print(f"Blocks: {num_blocks}, Steps: {num_steps}")
    print(f"Working set: {working_set_size}, Block size: {block_size_mb} MB")
    print(f"{'='*60}")

    # Reset memory
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.empty_cache()
    gc.collect()

    # Create simulator
    sim = BlockOffloadSimulator(
        num_blocks=num_blocks,
        working_set_size=working_set_size,
        strategy=strategy,
        block_size_mb=block_size_mb,
    )

    sim.initialize()

    # Create input tensor
    x = torch.randn(batch_size, seq_len, sim.hidden_size, dtype=torch.bfloat16, device='cuda')

    oom_occurred = False
    oom_step = None
    oom_block = None
    start_time = time.time()

    try:
        for step in range(num_steps):
            for block_idx in range(num_blocks):
                result = sim.run_forward(block_idx, x)
                if result is None:
                    oom_occurred = True
                    oom_step = step
                    oom_block = block_idx
                    raise RuntimeError(f"OOM at step {step}, block {block_idx}")
                x = result

            if (step + 1) % 10 == 0:
                mem_gb = torch.cuda.max_memory_allocated() / (1024**3)
                print(f"  Step {step + 1}/{num_steps}, Peak memory: {mem_gb:.2f} GB")

    except RuntimeError as e:
        if 'out of memory' in str(e).lower() or 'OOM' in str(e):
            oom_occurred = True
            print(f"OOM occurred: {e}")
        else:
            raise

    total_time = time.time() - start_time
    peak_memory_gb = torch.cuda.max_memory_allocated() / (1024**3)

    # Calculate sync trigger rate and save stats BEFORE cleanup
    total_transfers = sim.stats.load_count + sim.stats.offload_count
    sync_trigger_rate = sim.stats.sync_count / max(1, total_transfers)

    # Save stats before cleanup
    saved_stats = {
        "load_count": sim.stats.load_count,
        "offload_count": sim.stats.offload_count,
        "sync_count": sim.stats.sync_count,
        "oom_count": sim.stats.oom_count,
    }

    # Clean up
    del sim
    del x
    torch.cuda.empty_cache()
    gc.collect()

    return TrialResult(
        strategy=strategy.value,
        success=not oom_occurred,
        oom_occurred=oom_occurred,
        oom_at_step=oom_step,
        oom_at_block=oom_block,
        peak_memory_gb=peak_memory_gb,
        total_time_s=total_time,
        sync_trigger_rate=sync_trigger_rate,
        stats=saved_stats,
    )


def run_comparison(
    strategies: List[SyncStrategy],
    num_blocks: int = 60,
    num_steps: int = 20,
    working_set_size: int = 5,
    block_size_mb: int = 400,  # Smaller for testing on limited GPUs
    num_trials: int = 3,
    output_dir: str = "./results",
) -> Dict:
    """Run comparison across strategies with multiple trials"""

    print(f"\n{'='*80}")
    print("MEMORY SYNC STRESS TEST COMPARISON")
    print(f"{'='*80}")
    print(f"Blocks: {num_blocks}, Steps: {num_steps}")
    print(f"Working set: {working_set_size}, Block size: {block_size_mb} MB")
    print(f"Trials per strategy: {num_trials}")
    print(f"Strategies: {[s.value for s in strategies]}")

    results = {}

    for strategy in strategies:
        strategy_results = []

        for trial in range(num_trials):
            print(f"\n--- Trial {trial + 1}/{num_trials} for {strategy.value} ---")

            result = run_stress_trial(
                num_blocks=num_blocks,
                num_steps=num_steps,
                working_set_size=working_set_size,
                strategy=strategy,
                block_size_mb=block_size_mb,
            )

            strategy_results.append(asdict(result))

            # Delay between trials
            time.sleep(2)
            torch.cuda.empty_cache()
            gc.collect()

        results[strategy.value] = {
            "trials": strategy_results,
            "summary": summarize_trials(strategy_results),
        }

    # Generate comparison summary
    summary = generate_comparison(results)

    # Save results
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = os.path.join(output_dir, f"stress_test_results_{timestamp}.json")

    with open(output_file, 'w') as f:
        json.dump({
            "timestamp": timestamp,
            "config": {
                "num_blocks": num_blocks,
                "num_steps": num_steps,
                "working_set_size": working_set_size,
                "block_size_mb": block_size_mb,
                "num_trials": num_trials,
            },
            "results": results,
            "summary": summary,
        }, f, indent=2)

    print(f"\nResults saved to: {output_file}")

    return {
        "results": results,
        "summary": summary,
        "output_file": output_file,
    }


def summarize_trials(trials: List[Dict]) -> Dict:
    """Summarize results across trials"""
    success_count = sum(1 for t in trials if t["success"])
    oom_count = sum(1 for t in trials if t["oom_occurred"])
    peak_memories = [t["peak_memory_gb"] for t in trials]
    sync_rates = [t["sync_trigger_rate"] for t in trials]
    times = [t["total_time_s"] for t in trials]

    import statistics

    return {
        "success_rate": success_count / len(trials),
        "oom_rate": oom_count / len(trials),
        "peak_memory_mean": statistics.mean(peak_memories) if peak_memories else 0,
        "peak_memory_std": statistics.stdev(peak_memories) if len(peak_memories) > 1 else 0,
        "sync_rate_mean": statistics.mean(sync_rates) if sync_rates else 0,
        "time_mean": statistics.mean(times) if times else 0,
    }


def generate_comparison(results: Dict) -> Dict:
    """Generate comparison summary for paper"""
    summary = {}

    for strategy, data in results.items():
        summary[strategy] = data["summary"]

    # Print formatted table
    print(f"\n{'='*80}")
    print("SUMMARY TABLE FOR PAPER")
    print(f"{'='*80}")
    print(f"{'Strategy':<20} {'OOM Rate':<12} {'Peak Mem':<18} {'Sync Rate':<12} {'Time':<10}")
    print(f"{'-'*72}")

    for strategy, data in summary.items():
        oom_rate = f"{data['oom_rate']*100:.0f}%"
        peak_mem = f"{data['peak_memory_mean']:.2f} +/- {data['peak_memory_std']:.2f} GB"
        sync_rate = f"{data['sync_rate_mean']*100:.1f}%"
        time_str = f"{data['time_mean']:.1f}s"

        print(f"{strategy:<20} {oom_rate:<12} {peak_mem:<18} {sync_rate:<12} {time_str:<10}")

    print(f"{'='*80}")

    return summary


def auto_detect_block_size() -> int:
    """Auto-detect appropriate block size based on GPU memory"""
    if not torch.cuda.is_available():
        return 100

    total_mem_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)

    # Use smaller blocks to allow room for activations and ensure
    # sync strategies can make a difference
    if total_mem_gb >= 40:  # A100-40GB, A6000
        return 400  # Smaller than HunyuanVideo for testing
    elif total_mem_gb >= 24:  # RTX 4090, A5000
        return 200  # More conservative
    elif total_mem_gb >= 16:  # RTX 4070, V100-16GB
        return 150
    else:  # 8GB cards
        return 80


def main():
    parser = argparse.ArgumentParser(description="Memory stress test for sync protocols")

    parser.add_argument(
        "--strategy",
        type=str,
        default="all",
        choices=["pure_async", "pure_sync", "conditional_sync", "all"],
        help="Strategy to test"
    )

    parser.add_argument("--num-blocks", type=int, default=60, help="Number of blocks")
    parser.add_argument("--num-steps", type=int, default=20, help="Number of steps")
    parser.add_argument("--working-set-size", type=int, default=5, help="Working set size")
    parser.add_argument("--block-size-mb", type=int, default=0, help="Block size in MB (0=auto)")
    parser.add_argument("--num-trials", type=int, default=3, help="Trials per strategy")
    parser.add_argument("--output-dir", type=str, default="./results/stress_test", help="Output directory")

    args = parser.parse_args()

    # Check CUDA
    if not torch.cuda.is_available():
        print("ERROR: CUDA not available")
        sys.exit(1)

    # Auto-detect block size if not specified
    if args.block_size_mb == 0:
        args.block_size_mb = auto_detect_block_size()
        print(f"Auto-detected block size: {args.block_size_mb} MB")

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
        strategy_map = {
            "pure_async": SyncStrategy.PURE_ASYNC,
            "pure_sync": SyncStrategy.PURE_SYNC,
            "conditional_sync": SyncStrategy.CONDITIONAL_SYNC,
        }
        strategies = [strategy_map[args.strategy]]

    # Run comparison
    run_comparison(
        strategies=strategies,
        num_blocks=args.num_blocks,
        num_steps=args.num_steps,
        working_set_size=args.working_set_size,
        block_size_mb=args.block_size_mb,
        num_trials=args.num_trials,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()
