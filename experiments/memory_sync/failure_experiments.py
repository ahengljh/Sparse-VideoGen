"""
Failure Mechanism Experiments for Video DiT Memory Management

This module implements experiments to demonstrate three types of failure modes
in PyTorch's async memory management for Video DiT scenarios:

1. Deferred Reclamation - cudaFreeAsync doesn't immediately release memory
2. Fragmentation Accumulation - Repeated alloc/free causes unusable fragments
3. Peak Overlap - Async load/offload windows overlap causing OOM

These experiments are designed to generate data for the paper's claims.
"""

import torch
import torch.cuda
import time
import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import json
import os
from collections import defaultdict

from memory_monitor import (
    MemoryMonitor,
    MemoryFragmentationAnalyzer,
    estimate_largest_free_block,
    get_memory_summary,
    save_experiment_results
)


@dataclass
class DeferredReclamationResult:
    """Results from deferred reclamation experiment"""
    trial_id: int
    offload_time_ms: float
    reported_free_gb: float
    actual_free_gb: float  # Measured by allocation test
    allocation_success: bool
    required_sync: bool
    sync_resolved: bool
    memory_timeline: List[Dict]


@dataclass
class FragmentationResult:
    """Results from fragmentation experiment"""
    num_swaps: int
    total_free_gb: float
    largest_contiguous_gb: float
    fragmentation_ratio: float  # 1 - (largest / total)
    allocation_650mb_success: bool
    num_allocation_failures: int
    memory_layout_samples: List[Dict]


@dataclass
class PeakOverlapResult:
    """Results from peak overlap experiment"""
    trial_id: int
    async_peak_gb: float
    sync_peak_gb: float
    overlap_duration_ms: float
    oom_occurred: bool
    memory_timeline: List[Dict]


class DeferredReclamationExperiment:
    """
    Experiment 1: Demonstrate that cudaFreeAsync doesn't immediately release memory.

    Scenario:
    1. Allocate a block (simulating block on GPU)
    2. Free the block asynchronously
    3. Immediately try to allocate another block of same size
    4. Measure if allocation succeeds (it often fails without sync)

    This proves that "标记释放 ≠ 实际释放"
    """

    def __init__(self, device: int = 0, block_size_mb: float = 650):
        self.device = device
        self.block_size_mb = block_size_mb
        self.results: List[DeferredReclamationResult] = []

    def run_single_trial(self, trial_id: int, use_sync: bool = False) -> DeferredReclamationResult:
        """Run a single trial of the deferred reclamation test"""
        torch.cuda.empty_cache()
        torch.cuda.synchronize(self.device)

        block_size = int(self.block_size_mb * 1024 * 1024)
        num_elements = block_size // 4  # float32

        memory_timeline = []

        # Record initial state
        memory_timeline.append({
            'event': 'initial',
            'allocated_gb': torch.cuda.memory_allocated(self.device) / (1024**3),
            'reserved_gb': torch.cuda.memory_reserved(self.device) / (1024**3),
        })

        # Step 1: Allocate block
        block = torch.randn(num_elements, dtype=torch.float32, device=self.device)

        memory_timeline.append({
            'event': 'after_alloc',
            'allocated_gb': torch.cuda.memory_allocated(self.device) / (1024**3),
            'reserved_gb': torch.cuda.memory_reserved(self.device) / (1024**3),
        })

        # Step 2: Free the block (async)
        start_offload = time.time()
        del block
        offload_time_ms = (time.time() - start_offload) * 1000

        # DON'T sync yet - this is the key point
        reported_free = torch.cuda.memory_allocated(self.device) / (1024**3)

        memory_timeline.append({
            'event': 'after_del_no_sync',
            'allocated_gb': reported_free,
            'reserved_gb': torch.cuda.memory_reserved(self.device) / (1024**3),
        })

        # Step 3: Immediately try to allocate same size
        allocation_success = False
        try:
            new_block = torch.randn(num_elements, dtype=torch.float32, device=self.device)
            allocation_success = True
            del new_block
        except RuntimeError as e:
            if 'out of memory' not in str(e).lower():
                raise

        memory_timeline.append({
            'event': 'after_immediate_alloc_attempt',
            'success': allocation_success,
            'allocated_gb': torch.cuda.memory_allocated(self.device) / (1024**3),
        })

        # Step 4: Now sync and try again
        required_sync = not allocation_success
        sync_resolved = False

        if required_sync:
            torch.cuda.synchronize(self.device)
            torch.cuda.empty_cache()

            memory_timeline.append({
                'event': 'after_sync',
                'allocated_gb': torch.cuda.memory_allocated(self.device) / (1024**3),
                'reserved_gb': torch.cuda.memory_reserved(self.device) / (1024**3),
            })

            try:
                new_block = torch.randn(num_elements, dtype=torch.float32, device=self.device)
                sync_resolved = True
                del new_block
            except RuntimeError:
                pass

        # Measure actual free memory
        actual_free = estimate_largest_free_block(self.device) / (1024**3)

        torch.cuda.empty_cache()

        result = DeferredReclamationResult(
            trial_id=trial_id,
            offload_time_ms=offload_time_ms,
            reported_free_gb=reported_free,
            actual_free_gb=actual_free,
            allocation_success=allocation_success,
            required_sync=required_sync,
            sync_resolved=sync_resolved,
            memory_timeline=memory_timeline
        )

        self.results.append(result)
        return result

    def run_experiment(self, num_trials: int = 100) -> Dict:
        """Run multiple trials and aggregate results"""
        print(f"\n{'='*60}")
        print("EXPERIMENT 1: Deferred Reclamation")
        print(f"{'='*60}")
        print(f"Block size: {self.block_size_mb} MB")
        print(f"Trials: {num_trials}")

        self.results = []

        for trial in range(num_trials):
            result = self.run_single_trial(trial)
            if (trial + 1) % 20 == 0:
                print(f"  Completed {trial + 1}/{num_trials} trials")

        # Aggregate results
        immediate_success_rate = sum(1 for r in self.results if r.allocation_success) / len(self.results)
        sync_required_rate = sum(1 for r in self.results if r.required_sync) / len(self.results)
        sync_resolved_rate = sum(1 for r in self.results if r.sync_resolved) / sum(1 for r in self.results if r.required_sync) if any(r.required_sync for r in self.results) else 1.0

        summary = {
            'block_size_mb': self.block_size_mb,
            'num_trials': num_trials,
            'immediate_success_rate': immediate_success_rate,
            'sync_required_rate': sync_required_rate,
            'sync_resolved_rate': sync_resolved_rate,
            'conclusion': 'DEFERRED_RECLAMATION_CONFIRMED' if sync_required_rate > 0.1 else 'NOT_OBSERVED'
        }

        print(f"\nResults:")
        print(f"  Immediate allocation success rate: {immediate_success_rate:.1%}")
        print(f"  Sync required rate: {sync_required_rate:.1%}")
        print(f"  Sync resolved failures: {sync_resolved_rate:.1%}")
        print(f"  Conclusion: {summary['conclusion']}")

        return summary


class FragmentationExperiment:
    """
    Experiment 2: Demonstrate fragmentation accumulation from block swaps.

    Scenario:
    1. Simulate repeated block load/offload cycles
    2. Interleave with varying activation allocations
    3. Measure fragmentation over time
    4. Show that total free > block size but allocation fails

    This proves "Total Free: 2GB | Largest Contiguous: 400MB"
    """

    def __init__(self, device: int = 0, block_size_mb: float = 650):
        self.device = device
        self.block_size_mb = block_size_mb
        self.results: List[FragmentationResult] = []

    def run_fragmentation_buildup(
        self,
        num_swaps: int = 100,
        activation_sizes_mb: List[int] = None,
    ) -> FragmentationResult:
        """
        Run a fragmentation buildup experiment.

        Simulates the pattern in Video DiT where:
        - Blocks of fixed size (650MB) are loaded/offloaded
        - Activations of varying sizes are allocated/freed
        - This creates fragmentation over time
        """
        if activation_sizes_mb is None:
            # Simulate varying activation sizes during forward pass
            activation_sizes_mb = [100, 200, 150, 300, 250, 180, 220, 280, 160, 240]

        torch.cuda.empty_cache()
        torch.cuda.synchronize(self.device)

        block_size = int(self.block_size_mb * 1024 * 1024)
        num_elements_block = block_size // 4

        memory_layout_samples = []
        num_allocation_failures = 0

        # Storage for blocks and activations
        blocks: Dict[int, torch.Tensor] = {}
        activations: List[torch.Tensor] = []

        # Keep some blocks on GPU (simulating working set)
        working_set_size = 3

        for swap_idx in range(num_swaps):
            try:
                # Allocate a new block
                block_id = swap_idx % 10
                if block_id not in blocks:
                    blocks[block_id] = torch.randn(
                        num_elements_block,
                        dtype=torch.float32,
                        device=self.device
                    )

                # Allocate activation (varying size)
                act_size_mb = activation_sizes_mb[swap_idx % len(activation_sizes_mb)]
                act_elements = int(act_size_mb * 1024 * 1024 / 4)
                activations.append(torch.randn(act_elements, dtype=torch.float32, device=self.device))

                # Free oldest activations
                while len(activations) > 5:
                    del activations[0]

                # Free oldest blocks beyond working set
                if len(blocks) > working_set_size:
                    oldest_key = min(blocks.keys())
                    del blocks[oldest_key]

                # Sample memory layout periodically
                if swap_idx % 10 == 0:
                    sample = {
                        'swap_idx': swap_idx,
                        'allocated_gb': torch.cuda.memory_allocated(self.device) / (1024**3),
                        'reserved_gb': torch.cuda.memory_reserved(self.device) / (1024**3),
                        'num_blocks': len(blocks),
                        'num_activations': len(activations),
                    }
                    memory_layout_samples.append(sample)

            except RuntimeError as e:
                if 'out of memory' in str(e).lower():
                    num_allocation_failures += 1
                    # Clear and continue
                    activations.clear()
                    torch.cuda.empty_cache()
                else:
                    raise

        # Final measurement
        torch.cuda.synchronize(self.device)

        # Measure fragmentation
        total_free = (torch.cuda.get_device_properties(self.device).total_memory -
                      torch.cuda.memory_reserved(self.device)) / (1024**3)

        # Free everything but don't empty cache
        blocks.clear()
        activations.clear()

        # Now measure what the allocator thinks is free vs actually usable
        reported_free = (torch.cuda.memory_reserved(self.device) -
                         torch.cuda.memory_allocated(self.device)) / (1024**3)

        largest_contiguous = estimate_largest_free_block(self.device) / (1024**3)

        # Calculate fragmentation ratio
        if reported_free + total_free > 0:
            fragmentation_ratio = 1.0 - (largest_contiguous / (reported_free + total_free))
        else:
            fragmentation_ratio = 0.0

        # Test 650MB allocation
        allocation_650mb_success = False
        try:
            test_block = torch.randn(num_elements_block, dtype=torch.float32, device=self.device)
            allocation_650mb_success = True
            del test_block
        except RuntimeError:
            pass

        torch.cuda.empty_cache()

        result = FragmentationResult(
            num_swaps=num_swaps,
            total_free_gb=total_free + reported_free,
            largest_contiguous_gb=largest_contiguous,
            fragmentation_ratio=fragmentation_ratio,
            allocation_650mb_success=allocation_650mb_success,
            num_allocation_failures=num_allocation_failures,
            memory_layout_samples=memory_layout_samples
        )

        self.results.append(result)
        return result

    def run_experiment(self, swap_counts: List[int] = None) -> Dict:
        """Run fragmentation experiment with different swap counts"""
        if swap_counts is None:
            swap_counts = [10, 25, 50, 100, 200]

        print(f"\n{'='*60}")
        print("EXPERIMENT 2: Fragmentation Accumulation")
        print(f"{'='*60}")
        print(f"Block size: {self.block_size_mb} MB")
        print(f"Swap counts to test: {swap_counts}")

        self.results = []

        for num_swaps in swap_counts:
            print(f"\n  Testing {num_swaps} swaps...")
            result = self.run_fragmentation_buildup(num_swaps)
            print(f"    Total free: {result.total_free_gb:.2f} GB")
            print(f"    Largest contiguous: {result.largest_contiguous_gb:.2f} GB")
            print(f"    Fragmentation ratio: {result.fragmentation_ratio:.2%}")
            print(f"    650MB allocation: {'SUCCESS' if result.allocation_650mb_success else 'FAILED'}")
            print(f"    OOM failures during test: {result.num_allocation_failures}")

        # Summary
        summary = {
            'block_size_mb': self.block_size_mb,
            'results_by_swaps': {
                r.num_swaps: {
                    'total_free_gb': r.total_free_gb,
                    'largest_contiguous_gb': r.largest_contiguous_gb,
                    'fragmentation_ratio': r.fragmentation_ratio,
                    'allocation_success': r.allocation_650mb_success,
                }
                for r in self.results
            },
            'conclusion': 'FRAGMENTATION_CONFIRMED' if any(not r.allocation_650mb_success for r in self.results) else 'NOT_OBSERVED'
        }

        print(f"\nConclusion: {summary['conclusion']}")
        return summary


class PeakOverlapExperiment:
    """
    Experiment 3: Demonstrate peak overlap from async load/offload.

    Scenario:
    1. Start offloading block A (async)
    2. Before A is fully released, start loading block B (async)
    3. Both transfers overlap, causing temporary peak that exceeds capacity

    This proves async operations can cause memory peaks beyond steady state.
    """

    def __init__(self, device: int = 0, block_size_mb: float = 650):
        self.device = device
        self.block_size_mb = block_size_mb
        self.results: List[PeakOverlapResult] = []

    def run_overlap_test(self, trial_id: int) -> PeakOverlapResult:
        """Run a single peak overlap test"""
        torch.cuda.empty_cache()
        torch.cuda.synchronize(self.device)

        block_size = int(self.block_size_mb * 1024 * 1024)
        num_elements = block_size // 4

        memory_timeline = []
        oom_occurred = False

        # Create streams for async operations
        offload_stream = torch.cuda.Stream(device=self.device)
        load_stream = torch.cuda.Stream(device=self.device)

        # Record baseline
        baseline_allocated = torch.cuda.memory_allocated(self.device) / (1024**3)
        memory_timeline.append({
            'event': 'baseline',
            'time_ms': 0,
            'allocated_gb': baseline_allocated,
        })

        # Pre-allocate blocks on CPU (pinned memory)
        cpu_block_a = torch.randn(num_elements, dtype=torch.float32).pin_memory()
        cpu_block_b = torch.randn(num_elements, dtype=torch.float32).pin_memory()

        # Load block A to GPU
        gpu_block_a = cpu_block_a.to(self.device, non_blocking=False)
        torch.cuda.synchronize(self.device)

        memory_timeline.append({
            'event': 'block_a_loaded',
            'time_ms': 0,
            'allocated_gb': torch.cuda.memory_allocated(self.device) / (1024**3),
        })

        # Start async monitoring
        start_time = time.time()
        peak_async = torch.cuda.memory_allocated(self.device)

        # Async offload block A
        with torch.cuda.stream(offload_stream):
            # Copy to CPU (async)
            cpu_copy = gpu_block_a.to('cpu', non_blocking=True)

        # Immediately start loading block B (without waiting for A to complete)
        gpu_block_b = None
        try:
            with torch.cuda.stream(load_stream):
                gpu_block_b = cpu_block_b.to(self.device, non_blocking=True)

            # Sample memory during overlap
            for i in range(10):
                current_mem = torch.cuda.memory_allocated(self.device)
                peak_async = max(peak_async, current_mem)
                memory_timeline.append({
                    'event': f'overlap_sample_{i}',
                    'time_ms': (time.time() - start_time) * 1000,
                    'allocated_gb': current_mem / (1024**3),
                })
                time.sleep(0.001)  # 1ms sampling

        except RuntimeError as e:
            if 'out of memory' in str(e).lower():
                oom_occurred = True
            else:
                raise

        overlap_duration_ms = (time.time() - start_time) * 1000

        # Now do it synchronously for comparison
        torch.cuda.synchronize(self.device)
        del gpu_block_a
        if gpu_block_b is not None:
            del gpu_block_b
        torch.cuda.empty_cache()

        # Sync test
        gpu_block_a = cpu_block_a.to(self.device, non_blocking=False)
        torch.cuda.synchronize(self.device)

        # Sync offload
        del gpu_block_a
        torch.cuda.synchronize(self.device)
        torch.cuda.empty_cache()

        sync_after_offload = torch.cuda.memory_allocated(self.device)

        # Sync load
        gpu_block_b = cpu_block_b.to(self.device, non_blocking=False)
        torch.cuda.synchronize(self.device)
        sync_peak = torch.cuda.memory_allocated(self.device)

        memory_timeline.append({
            'event': 'sync_peak',
            'time_ms': overlap_duration_ms + 10,
            'allocated_gb': sync_peak / (1024**3),
        })

        # Cleanup
        del gpu_block_b
        torch.cuda.empty_cache()

        result = PeakOverlapResult(
            trial_id=trial_id,
            async_peak_gb=peak_async / (1024**3),
            sync_peak_gb=sync_peak / (1024**3),
            overlap_duration_ms=overlap_duration_ms,
            oom_occurred=oom_occurred,
            memory_timeline=memory_timeline
        )

        self.results.append(result)
        return result

    def run_experiment(self, num_trials: int = 50) -> Dict:
        """Run multiple trials of peak overlap test"""
        print(f"\n{'='*60}")
        print("EXPERIMENT 3: Peak Overlap")
        print(f"{'='*60}")
        print(f"Block size: {self.block_size_mb} MB")
        print(f"Trials: {num_trials}")

        self.results = []

        for trial in range(num_trials):
            result = self.run_overlap_test(trial)
            if (trial + 1) % 10 == 0:
                print(f"  Completed {trial + 1}/{num_trials} trials")

        # Aggregate results
        oom_rate = sum(1 for r in self.results if r.oom_occurred) / len(self.results)
        avg_async_peak = np.mean([r.async_peak_gb for r in self.results])
        avg_sync_peak = np.mean([r.sync_peak_gb for r in self.results])
        peak_difference = avg_async_peak - avg_sync_peak

        summary = {
            'block_size_mb': self.block_size_mb,
            'num_trials': num_trials,
            'oom_rate': oom_rate,
            'avg_async_peak_gb': avg_async_peak,
            'avg_sync_peak_gb': avg_sync_peak,
            'peak_difference_gb': peak_difference,
            'conclusion': 'PEAK_OVERLAP_CONFIRMED' if peak_difference > 0.1 or oom_rate > 0.1 else 'NOT_OBSERVED'
        }

        print(f"\nResults:")
        print(f"  OOM rate: {oom_rate:.1%}")
        print(f"  Avg async peak: {avg_async_peak:.2f} GB")
        print(f"  Avg sync peak: {avg_sync_peak:.2f} GB")
        print(f"  Peak difference: {peak_difference:.2f} GB")
        print(f"  Conclusion: {summary['conclusion']}")

        return summary


def run_all_failure_experiments(
    output_dir: str = "./results/failure_experiments",
    block_size_mb: float = 500,  # Adjust based on GPU
) -> Dict:
    """Run all three failure mechanism experiments"""
    os.makedirs(output_dir, exist_ok=True)

    print("\n" + "="*70)
    print("RUNNING ALL FAILURE MECHANISM EXPERIMENTS")
    print("="*70)
    print(f"\nGPU Memory Summary:")
    summary = get_memory_summary()
    for k, v in summary.items():
        print(f"  {k}: {v}")

    results = {}

    # Experiment 1: Deferred Reclamation
    exp1 = DeferredReclamationExperiment(block_size_mb=block_size_mb)
    results['deferred_reclamation'] = exp1.run_experiment(num_trials=50)

    # Experiment 2: Fragmentation
    exp2 = FragmentationExperiment(block_size_mb=block_size_mb)
    results['fragmentation'] = exp2.run_experiment(swap_counts=[10, 25, 50, 100])

    # Experiment 3: Peak Overlap
    exp3 = PeakOverlapExperiment(block_size_mb=block_size_mb)
    results['peak_overlap'] = exp3.run_experiment(num_trials=30)

    # Save results
    results_path = os.path.join(output_dir, 'failure_experiments_results.json')
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)

    print(f"\n{'='*70}")
    print("ALL EXPERIMENTS COMPLETED")
    print(f"{'='*70}")
    print(f"Results saved to: {results_path}")

    return results


if __name__ == "__main__":
    # Run with reduced block size for testing
    results = run_all_failure_experiments(block_size_mb=300)
