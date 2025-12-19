"""
Stress Test for Memory Synchronization Strategies

This script specifically tests the three failure modes described in the paper:
1. Deferred Reclamation - cudaFreeAsync is non-blocking
2. Fragmentation Accumulation - Different sized allocations interleave
3. Peak Overlap - Async operations overlap causing memory spikes

It uses aggressive memory pressure to demonstrate the differences between
synchronization strategies.
"""

import argparse
import gc
import time
import sys
import os
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass, field
import json

import torch
import torch.nn as nn
import numpy as np
from termcolor import colored

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from experiments.memory_sync_strategies import (
    SyncStrategy,
    MemoryProfiler,
    ExperimentResult,
    create_strategy,
)


@dataclass
class StressTestConfig:
    """Configuration for stress tests"""
    device: int = 0
    num_blocks: int = 60  # HunyuanVideo has 60 blocks
    block_size_mb: float = 200  # Simulated block size
    working_set_size: int = 5  # K blocks on GPU
    num_diffusion_steps: int = 40  # T steps
    num_trials: int = 10
    seed: int = 42

    # Memory pressure settings
    activation_size_ratio: float = 0.3  # Activation as ratio of total GPU memory
    variable_activation: bool = True  # Vary activation size to create fragmentation

    # Test specific settings
    target_memory_usage: float = 0.90  # Target 90% GPU usage


class SyntheticTransformerBlock(nn.Module):
    """Synthetic block that mimics HunyuanVideo transformer block memory pattern"""

    def __init__(self, hidden_dim: int, dtype=torch.bfloat16):
        super().__init__()
        # Typical transformer block structure
        self.norm1 = nn.LayerNorm(hidden_dim, dtype=dtype)
        self.attn = nn.Linear(hidden_dim, hidden_dim * 3, dtype=dtype)  # QKV
        self.proj = nn.Linear(hidden_dim, hidden_dim, dtype=dtype)
        self.norm2 = nn.LayerNorm(hidden_dim, dtype=dtype)
        self.mlp_fc1 = nn.Linear(hidden_dim, hidden_dim * 4, dtype=dtype)
        self.mlp_fc2 = nn.Linear(hidden_dim * 4, hidden_dim, dtype=dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Simplified forward pass
        h = self.norm1(x)
        qkv = self.attn(h)
        q, k, v = qkv.chunk(3, dim=-1)
        # Simplified attention (no actual attention computation)
        attn_out = v  # Placeholder
        h = x + self.proj(attn_out)
        h = h + self.mlp_fc2(torch.nn.functional.gelu(self.mlp_fc1(self.norm2(h))))
        return h


def calculate_block_hidden_dim(target_size_mb: float, dtype=torch.bfloat16) -> int:
    """Calculate hidden dimension to achieve target block size"""
    bytes_per_elem = 2 if dtype == torch.bfloat16 else 4

    # Approximate parameter count for SyntheticTransformerBlock:
    # norm1: 2 * hidden_dim
    # attn: hidden_dim * 3 * hidden_dim + 3 * hidden_dim (bias)
    # proj: hidden_dim * hidden_dim + hidden_dim
    # norm2: 2 * hidden_dim
    # mlp_fc1: hidden_dim * 4 * hidden_dim + 4 * hidden_dim
    # mlp_fc2: 4 * hidden_dim * hidden_dim + hidden_dim
    # Total ≈ 8 * hidden_dim^2 + 10 * hidden_dim

    target_bytes = target_size_mb * 1024 * 1024
    target_params = target_bytes / bytes_per_elem

    # Solve: 8 * h^2 + 10 * h = target_params
    # h ≈ sqrt(target_params / 8)
    hidden_dim = int((target_params / 8) ** 0.5)

    # Round to multiple of 64 for efficiency
    hidden_dim = (hidden_dim // 64) * 64
    return max(hidden_dim, 128)


def get_gpu_memory_mb(device: int = 0) -> float:
    """Get total GPU memory in MB"""
    props = torch.cuda.get_device_properties(device)
    return props.total_memory / (1024 ** 2)


def get_free_memory_mb(device: int = 0) -> float:
    """Get free GPU memory in MB"""
    total = get_gpu_memory_mb(device)
    reserved = torch.cuda.memory_reserved(device) / (1024 ** 2)
    return total - reserved


def get_allocated_memory_mb(device: int = 0) -> float:
    """Get allocated GPU memory in MB"""
    return torch.cuda.memory_allocated(device) / (1024 ** 2)


def run_stress_test(
    config: StressTestConfig,
    strategy_type: SyncStrategy,
    trial_id: int,
) -> ExperimentResult:
    """Run a single stress test trial"""

    print(f"\n{'-'*60}")
    print(f"Trial {trial_id}: {strategy_type.value}")
    print(f"{'-'*60}")

    torch.manual_seed(config.seed + trial_id)
    torch.cuda.manual_seed(config.seed + trial_id)

    result = ExperimentResult(
        strategy=strategy_type.value,
        total_steps=config.num_diffusion_steps,
    )

    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(config.device)

    try:
        # Calculate dimensions
        total_memory = get_gpu_memory_mb(config.device)
        hidden_dim = calculate_block_hidden_dim(config.block_size_mb)

        print(f"GPU Memory: {total_memory:.0f} MB")
        print(f"Block hidden dim: {hidden_dim}")

        # Create blocks
        print(f"Creating {config.num_blocks} blocks...")
        blocks = []
        for i in range(config.num_blocks):
            block = SyntheticTransformerBlock(hidden_dim)
            blocks.append(block)

        # Verify block size
        actual_block_size = sum(
            p.numel() * p.element_size() for p in blocks[0].parameters()
        ) / (1024 ** 2)
        print(f"Actual block size: {actual_block_size:.1f} MB")

        # Create strategy
        strategy = create_strategy(
            strategy_type,
            device=config.device,
            safety_margin_mb=500.0,
            memory_threshold_ratio=0.15,
        )

        # Initialize: all blocks on CPU
        for block in blocks:
            block.to('cpu')

        # Load initial working set
        gpu_blocks = set()
        for i in range(config.working_set_size):
            blocks[i].to(f'cuda:{config.device}')
            gpu_blocks.add(i)

        torch.cuda.synchronize()
        print(f"Initial memory usage: {get_allocated_memory_mb(config.device):.0f} MB")

        # Calculate activation size for memory pressure
        activation_budget = total_memory * config.activation_size_ratio
        seq_len = int(activation_budget * 1024 * 1024 / (hidden_dim * 2 * 2))  # bfloat16 = 2 bytes
        seq_len = min(seq_len, 65536)  # Cap at reasonable size
        print(f"Activation sequence length: {seq_len}")

        profiler = MemoryProfiler(device=config.device)
        start_time = time.time()
        peak_memories = []

        try:
            for step in range(config.num_diffusion_steps):
                step_start = time.time()

                # Create activation tensor (simulates x_t)
                if config.variable_activation:
                    # Vary activation size to create fragmentation
                    var_seq_len = int(seq_len * (0.8 + 0.4 * np.random.random()))
                else:
                    var_seq_len = seq_len

                x = torch.randn(
                    1, var_seq_len, hidden_dim,
                    device=f'cuda:{config.device}',
                    dtype=torch.bfloat16
                )

                # Process all blocks
                for block_idx in range(config.num_blocks):
                    # Load block if not on GPU
                    if block_idx not in gpu_blocks:
                        # Offload oldest block if at capacity
                        if len(gpu_blocks) >= config.working_set_size:
                            offload_idx = min(gpu_blocks)
                            if offload_idx < block_idx:  # Only offload if behind
                                success = strategy.offload_block(blocks[offload_idx])
                                if success:
                                    gpu_blocks.discard(offload_idx)
                                else:
                                    raise RuntimeError(f"Failed to offload block {offload_idx}")

                        # Load needed block
                        success = strategy.load_block(blocks[block_idx], actual_block_size)
                        if success:
                            gpu_blocks.add(block_idx)
                        else:
                            raise RuntimeError(f"Failed to load block {block_idx}")

                    # Pre-forward sync
                    strategy.pre_forward_sync(block_idx, config.num_blocks)

                    # Forward pass
                    with torch.no_grad():
                        x = blocks[block_idx](x)

                    # Post-forward sync
                    strategy.post_forward_sync(block_idx, config.num_blocks)

                # Record peak memory for this step
                torch.cuda.synchronize()
                peak = torch.cuda.max_memory_allocated(config.device) / (1024 ** 2)
                peak_memories.append(peak)

                # Simulate DDIM step - new latent
                del x

                # Async strategy doesn't clean up
                if strategy_type == SyncStrategy.PURE_SYNC:
                    gc.collect()
                    torch.cuda.empty_cache()
                elif strategy_type == SyncStrategy.CONDITIONAL_SYNC:
                    if get_free_memory_mb(config.device) < 500:
                        gc.collect()
                        torch.cuda.empty_cache()

                result.successful_steps += 1

                step_time = time.time() - step_start
                current_mem = get_allocated_memory_mb(config.device)
                print(f"  Step {step+1}/{config.num_diffusion_steps}: "
                      f"mem={current_mem:.0f}MB, peak={peak:.0f}MB, "
                      f"time={step_time:.2f}s", end="\r")

            print()  # Newline after progress

        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                result.oom_occurred = True
                result.oom_message = str(e)
                print(colored(f"\nOOM at step {result.successful_steps}: {e}", "red"))
            else:
                raise

        # Record results
        result.total_time_seconds = time.time() - start_time
        result.peak_memory_mb = max(peak_memories) if peak_memories else 0
        result.sync_count = strategy.sync_count
        result.load_count = strategy.load_count
        result.offload_count = strategy.offload_count

        # Create memory samples from peak recordings
        from experiments.memory_sync_strategies import MemoryStats
        for i, peak in enumerate(peak_memories):
            result.memory_samples.append(MemoryStats(
                peak_memory_mb=peak,
                timestamp=i,
            ))

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

    print(f"\nTrial {trial_id} Summary:")
    print(f"  OOM: {result.oom_occurred}")
    print(f"  Steps: {result.successful_steps}/{result.total_steps}")
    print(f"  Peak Memory: {result.peak_memory_mb:.0f} MB")
    print(f"  Peak Variance: {result.peak_variance_mb:.1f} MB")
    print(f"  Sync Count: {result.sync_count}")
    print(f"  Time: {result.total_time_seconds:.2f}s")

    return result


def run_failure_mode_test(
    failure_mode: str,
    config: StressTestConfig,
) -> Dict[str, List[ExperimentResult]]:
    """
    Run tests specifically designed to trigger each failure mode.
    """

    print(f"\n{'='*60}")
    print(f"FAILURE MODE TEST: {failure_mode}")
    print(f"{'='*60}")

    if failure_mode == "deferred_reclamation":
        # Rapid load/offload cycles to trigger delayed memory release
        config.working_set_size = 2  # Very small working set
        config.num_diffusion_steps = 20

    elif failure_mode == "fragmentation":
        # Variable activation sizes to create fragmentation
        config.variable_activation = True
        config.activation_size_ratio = 0.35  # Higher memory pressure

    elif failure_mode == "peak_overlap":
        # Large working set with overlapping operations
        config.working_set_size = 8
        config.activation_size_ratio = 0.25

    results = {}
    strategies = [
        SyncStrategy.PURE_ASYNC,
        SyncStrategy.PURE_SYNC,
        SyncStrategy.CONDITIONAL_SYNC,
    ]

    for strategy in strategies:
        strategy_results = []
        for trial in range(config.num_trials):
            result = run_stress_test(config, strategy, trial)
            strategy_results.append(result)
        results[strategy.value] = strategy_results

    return results


def print_results_table(results: Dict[str, List[ExperimentResult]], title: str = ""):
    """Print formatted results table"""

    if title:
        print(f"\n{title}")
        print("=" * 70)

    print("\n| Strategy | OOM Rate | Avg Peak (GB) | Peak Std (GB) | Sync Rate |")
    print("|----------|----------|---------------|---------------|-----------|")

    for strategy_name, trials in results.items():
        oom_count = sum(1 for t in trials if t.oom_occurred)
        oom_rate = 100 * oom_count / len(trials)

        successful = [t for t in trials if not t.oom_occurred]
        if successful:
            avg_peak_gb = np.mean([t.avg_peak_memory_mb for t in successful]) / 1024
            std_peak_gb = np.mean([t.peak_variance_mb for t in successful]) / 1024

            # Sync rate = syncs per operation
            total_ops = sum(t.load_count + t.offload_count for t in successful)
            total_syncs = sum(t.sync_count for t in successful)
            sync_rate = 100 * total_syncs / total_ops if total_ops > 0 else 0
        else:
            avg_peak_gb = float('nan')
            std_peak_gb = float('nan')
            sync_rate = float('nan')

        display_name = {
            'pure_async': 'Pure Async',
            'pure_sync': 'Pure Sync',
            'conditional_sync': 'Conditional',
        }.get(strategy_name, strategy_name)

        print(f"| {display_name:<8} | {oom_rate:>6.0f}% | {avg_peak_gb:>13.2f} | "
              f"±{std_peak_gb:>12.2f} | {sync_rate:>7.1f}% |")


def main():
    parser = argparse.ArgumentParser(description="Stress Test Memory Sync Strategies")

    parser.add_argument("--mode", type=str, default="all",
                       choices=["all", "standard", "deferred", "fragmentation", "peak_overlap"],
                       help="Test mode")
    parser.add_argument("--num_trials", type=int, default=10,
                       help="Number of trials per strategy")
    parser.add_argument("--num_blocks", type=int, default=60,
                       help="Number of transformer blocks")
    parser.add_argument("--block_size_mb", type=float, default=200,
                       help="Target block size in MB")
    parser.add_argument("--working_set_size", type=int, default=5,
                       help="Number of blocks to keep on GPU")
    parser.add_argument("--num_steps", type=int, default=40,
                       help="Number of diffusion steps")
    parser.add_argument("--output_file", type=str, default=None,
                       help="Output JSON file")
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed")

    args = parser.parse_args()

    if not torch.cuda.is_available():
        print(colored("Error: CUDA is not available!", "red"))
        sys.exit(1)

    # Print GPU info
    props = torch.cuda.get_device_properties(0)
    print(f"\nGPU: {props.name}")
    print(f"Total Memory: {props.total_memory / (1024**3):.1f} GB")

    config = StressTestConfig(
        num_blocks=args.num_blocks,
        block_size_mb=args.block_size_mb,
        working_set_size=args.working_set_size,
        num_diffusion_steps=args.num_steps,
        num_trials=args.num_trials,
        seed=args.seed,
    )

    all_results = {}

    if args.mode in ["all", "standard"]:
        print("\n" + "=" * 60)
        print("STANDARD STRESS TEST")
        print("=" * 60)

        results = {}
        strategies = [
            SyncStrategy.PURE_ASYNC,
            SyncStrategy.PURE_SYNC,
            SyncStrategy.CONDITIONAL_SYNC,
        ]

        for strategy in strategies:
            strategy_results = []
            for trial in range(args.num_trials):
                result = run_stress_test(config, strategy, trial)
                strategy_results.append(result)
            results[strategy.value] = strategy_results

        print_results_table(results, "Standard Test Results")
        all_results["standard"] = results

    if args.mode in ["all", "deferred"]:
        results = run_failure_mode_test("deferred_reclamation", StressTestConfig(
            num_blocks=args.num_blocks,
            block_size_mb=args.block_size_mb,
            num_trials=args.num_trials,
            seed=args.seed,
        ))
        print_results_table(results, "Deferred Reclamation Test Results")
        all_results["deferred_reclamation"] = results

    if args.mode in ["all", "fragmentation"]:
        results = run_failure_mode_test("fragmentation", StressTestConfig(
            num_blocks=args.num_blocks,
            block_size_mb=args.block_size_mb,
            num_trials=args.num_trials,
            seed=args.seed,
        ))
        print_results_table(results, "Fragmentation Test Results")
        all_results["fragmentation"] = results

    if args.mode in ["all", "peak_overlap"]:
        results = run_failure_mode_test("peak_overlap", StressTestConfig(
            num_blocks=args.num_blocks,
            block_size_mb=args.block_size_mb,
            num_trials=args.num_trials,
            seed=args.seed,
        ))
        print_results_table(results, "Peak Overlap Test Results")
        all_results["peak_overlap"] = results

    # Save results
    if args.output_file:
        def serialize_results(results):
            serialized = {}
            for mode, mode_results in results.items():
                serialized[mode] = {}
                for strategy, trials in mode_results.items():
                    serialized[mode][strategy] = []
                    for trial in trials:
                        trial_dict = {
                            'strategy': trial.strategy,
                            'oom_occurred': trial.oom_occurred,
                            'oom_message': trial.oom_message,
                            'peak_memory_mb': trial.peak_memory_mb,
                            'total_time_seconds': trial.total_time_seconds,
                            'sync_count': trial.sync_count,
                            'load_count': trial.load_count,
                            'offload_count': trial.offload_count,
                            'successful_steps': trial.successful_steps,
                            'total_steps': trial.total_steps,
                            'avg_peak_memory_mb': trial.avg_peak_memory_mb,
                            'peak_variance_mb': trial.peak_variance_mb,
                        }
                        serialized[mode][strategy].append(trial_dict)
            return serialized

        with open(args.output_file, 'w') as f:
            json.dump(serialize_results(all_results), f, indent=2)
        print(f"\nResults saved to {args.output_file}")

    # Print final summary table (paper format)
    print("\n" + "=" * 70)
    print("FINAL SUMMARY (Paper Format)")
    print("=" * 70)

    if "standard" in all_results:
        results = all_results["standard"]
        print("\n| 传输策略 | OOM率 | 平均峰值内存 | 峰值方差 |")
        print("|----------|-------|-------------|----------|")

        for strategy_name, trials in results.items():
            oom_count = sum(1 for t in trials if t.oom_occurred)
            oom_rate = f"{100*oom_count/len(trials):.0f}%"

            successful = [t for t in trials if not t.oom_occurred]
            if successful:
                avg_peak = np.mean([t.avg_peak_memory_mb for t in successful]) / 1024
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


if __name__ == "__main__":
    main()
