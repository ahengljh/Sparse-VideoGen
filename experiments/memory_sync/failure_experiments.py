"""
Failure Mechanism Experiments for Video DiT Memory Management (v2)

This module implements experiments to demonstrate three types of failure modes
in PyTorch's async memory management for Video DiT scenarios.

KEY FIX: The experiments now properly fill GPU memory to create realistic
memory pressure conditions that trigger failure modes.

1. Deferred Reclamation - cudaFreeAsync doesn't immediately release memory
2. Fragmentation Accumulation - Repeated alloc/free causes unusable fragments
3. Peak Overlap - Async load/offload windows overlap causing OOM
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
import gc

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
    base_pressure_gb: float
    block_size_mb: float
    allocation_without_sync_success: bool
    allocation_with_sync_success: bool
    memory_timeline: List[Dict]


@dataclass
class FragmentationResult:
    """Results from fragmentation experiment"""
    num_swaps: int
    base_pressure_gb: float
    total_free_gb: float
    largest_contiguous_gb: float
    fragmentation_ratio: float
    target_allocation_success: bool
    num_allocation_failures: int
    memory_layout_samples: List[Dict]


@dataclass
class PeakOverlapResult:
    """Results from peak overlap experiment"""
    trial_id: int
    base_pressure_gb: float
    block_size_mb: float
    async_peak_gb: float
    sync_peak_gb: float
    oom_in_async: bool
    oom_in_sync: bool
    memory_timeline: List[Dict]


class MemoryPressureManager:
    """
    Manages background memory pressure to simulate realistic VDM conditions.

    In real Video DiT inference:
    - Activations take 12-15GB
    - VAE encoder/decoder takes memory
    - Text encoder embeddings
    - Only 2-4GB left for block swapping
    """

    def __init__(self, device: int = 0):
        self.device = device
        self.pressure_tensors: List[torch.Tensor] = []
        self.target_free_gb: float = 0

    def get_total_memory_gb(self) -> float:
        return torch.cuda.get_device_properties(self.device).total_memory / (1024**3)

    def get_free_memory_gb(self) -> float:
        return (torch.cuda.get_device_properties(self.device).total_memory -
                torch.cuda.memory_reserved(self.device)) / (1024**3)

    def get_allocated_gb(self) -> float:
        return torch.cuda.memory_allocated(self.device) / (1024**3)

    def fill_to_pressure(self, target_free_gb: float, chunk_size_mb: float = 512) -> float:
        """
        Fill GPU memory until only target_free_gb remains.

        Returns actual free memory after filling.
        """
        self.clear()
        torch.cuda.empty_cache()
        torch.cuda.synchronize(self.device)

        self.target_free_gb = target_free_gb
        chunk_elements = int(chunk_size_mb * 1024 * 1024 / 4)  # float32

        while self.get_free_memory_gb() > target_free_gb + chunk_size_mb / 1024:
            try:
                tensor = torch.randn(chunk_elements, dtype=torch.float32, device=self.device)
                self.pressure_tensors.append(tensor)
            except RuntimeError:
                break

        # Fine-tune with smaller chunks
        small_chunk = int(64 * 1024 * 1024 / 4)
        while self.get_free_memory_gb() > target_free_gb + 0.1:
            try:
                tensor = torch.randn(small_chunk, dtype=torch.float32, device=self.device)
                self.pressure_tensors.append(tensor)
            except RuntimeError:
                break

        torch.cuda.synchronize(self.device)
        actual_free = self.get_free_memory_gb()
        print(f"    Memory pressure set: {self.get_allocated_gb():.2f}GB used, {actual_free:.2f}GB free")
        return actual_free

    def clear(self):
        """Release all pressure tensors"""
        self.pressure_tensors.clear()
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.synchronize(self.device)


class DeferredReclamationExperiment:
    """
    Experiment 1: Demonstrate deferred reclamation under memory pressure.

    Key insight: When GPU memory is nearly full, deleting a tensor doesn't
    immediately free the memory. The allocator marks it as available but
    the actual CUDA memory isn't released until synchronization.

    Setup:
    1. Fill GPU to leave only ~1.5x block size free
    2. Allocate a block
    3. Delete the block (no sync)
    4. Immediately try to allocate another block of same size
    5. This often fails because the first block's memory isn't actually free yet
    """

    def __init__(self, device: int = 0):
        self.device = device
        self.pressure_manager = MemoryPressureManager(device)
        self.results: List[DeferredReclamationResult] = []

    def run_single_trial(
        self,
        trial_id: int,
        block_size_mb: float,
        target_free_gb: float
    ) -> DeferredReclamationResult:
        """Run a single trial under memory pressure"""

        memory_timeline = []

        # Step 1: Create memory pressure
        self.pressure_manager.fill_to_pressure(target_free_gb)

        memory_timeline.append({
            'event': 'pressure_set',
            'allocated_gb': self.pressure_manager.get_allocated_gb(),
            'free_gb': self.pressure_manager.get_free_memory_gb(),
        })

        block_elements = int(block_size_mb * 1024 * 1024 / 4)

        # Step 2: Allocate first block
        try:
            block_a = torch.randn(block_elements, dtype=torch.float32, device=self.device)
        except RuntimeError:
            self.pressure_manager.clear()
            return DeferredReclamationResult(
                trial_id=trial_id,
                base_pressure_gb=self.pressure_manager.get_allocated_gb(),
                block_size_mb=block_size_mb,
                allocation_without_sync_success=False,
                allocation_with_sync_success=False,
                memory_timeline=memory_timeline
            )

        memory_timeline.append({
            'event': 'block_a_allocated',
            'allocated_gb': self.pressure_manager.get_allocated_gb(),
            'free_gb': self.pressure_manager.get_free_memory_gb(),
        })

        # Step 3: Delete block A WITHOUT synchronization
        del block_a
        # Explicitly DO NOT call torch.cuda.synchronize() or empty_cache()

        memory_timeline.append({
            'event': 'block_a_deleted_no_sync',
            'allocated_gb': torch.cuda.memory_allocated(self.device) / (1024**3),
            'reserved_gb': torch.cuda.memory_reserved(self.device) / (1024**3),
        })

        # Step 4: Immediately try to allocate block B
        allocation_without_sync_success = False
        try:
            block_b = torch.randn(block_elements, dtype=torch.float32, device=self.device)
            allocation_without_sync_success = True
            del block_b
        except RuntimeError as e:
            if 'out of memory' not in str(e).lower():
                raise

        memory_timeline.append({
            'event': 'immediate_alloc_attempt',
            'success': allocation_without_sync_success,
            'allocated_gb': torch.cuda.memory_allocated(self.device) / (1024**3),
        })

        # Step 5: Now sync and try again
        torch.cuda.synchronize(self.device)
        torch.cuda.empty_cache()

        allocation_with_sync_success = False
        try:
            block_b = torch.randn(block_elements, dtype=torch.float32, device=self.device)
            allocation_with_sync_success = True
            del block_b
        except RuntimeError:
            pass

        memory_timeline.append({
            'event': 'after_sync_alloc_attempt',
            'success': allocation_with_sync_success,
            'allocated_gb': torch.cuda.memory_allocated(self.device) / (1024**3),
        })

        # Cleanup
        self.pressure_manager.clear()

        return DeferredReclamationResult(
            trial_id=trial_id,
            base_pressure_gb=target_free_gb,
            block_size_mb=block_size_mb,
            allocation_without_sync_success=allocation_without_sync_success,
            allocation_with_sync_success=allocation_with_sync_success,
            memory_timeline=memory_timeline
        )

    def run_experiment(
        self,
        block_size_mb: float = 650,
        num_trials: int = 50,
        free_memory_ratios: List[float] = None
    ) -> Dict:
        """
        Run deferred reclamation experiment with varying memory pressure.

        free_memory_ratios: List of (free_memory / block_size) ratios to test
        """
        if free_memory_ratios is None:
            # Test with 1.2x, 1.5x, 2.0x, 3.0x block size free
            free_memory_ratios = [1.2, 1.5, 2.0, 3.0]

        print(f"\n{'='*60}")
        print("EXPERIMENT 1: Deferred Reclamation (with memory pressure)")
        print(f"{'='*60}")
        print(f"Block size: {block_size_mb} MB")
        print(f"Free memory ratios to test: {free_memory_ratios}")

        results_by_ratio = {}

        for ratio in free_memory_ratios:
            target_free_gb = (block_size_mb * ratio) / 1024
            print(f"\n  Testing with {ratio}x block size free ({target_free_gb:.2f} GB)...")

            trial_results = []
            for trial in range(num_trials):
                result = self.run_single_trial(trial, block_size_mb, target_free_gb)
                trial_results.append(result)
                self.results.append(result)

            # Aggregate
            without_sync_success = sum(1 for r in trial_results if r.allocation_without_sync_success)
            with_sync_success = sum(1 for r in trial_results if r.allocation_with_sync_success)

            results_by_ratio[ratio] = {
                'target_free_gb': target_free_gb,
                'without_sync_success_rate': without_sync_success / num_trials,
                'with_sync_success_rate': with_sync_success / num_trials,
                'deferred_reclamation_rate': 1 - (without_sync_success / num_trials),
            }

            print(f"    Without sync success: {without_sync_success}/{num_trials} ({without_sync_success/num_trials:.1%})")
            print(f"    With sync success: {with_sync_success}/{num_trials} ({with_sync_success/num_trials:.1%})")
            print(f"    Deferred reclamation observed: {(1 - without_sync_success/num_trials):.1%}")

        # Summary
        max_deferred = max(r['deferred_reclamation_rate'] for r in results_by_ratio.values())

        summary = {
            'block_size_mb': block_size_mb,
            'num_trials_per_ratio': num_trials,
            'results_by_ratio': results_by_ratio,
            'max_deferred_reclamation_rate': max_deferred,
            'conclusion': 'DEFERRED_RECLAMATION_CONFIRMED' if max_deferred > 0.1 else 'NOT_OBSERVED'
        }

        print(f"\nConclusion: {summary['conclusion']}")
        return summary


class FragmentationExperiment:
    """
    Experiment 2: Demonstrate fragmentation under memory pressure.

    Key insight: When GPU is nearly full and we do repeated alloc/free
    of different sizes (blocks + activations), the memory becomes fragmented.
    Total free might be 2GB but largest contiguous block is only 400MB.
    """

    def __init__(self, device: int = 0):
        self.device = device
        self.pressure_manager = MemoryPressureManager(device)
        self.results: List[FragmentationResult] = []

    def run_fragmentation_test(
        self,
        target_free_gb: float,
        block_size_mb: float,
        num_swaps: int,
        activation_sizes_mb: List[int] = None
    ) -> FragmentationResult:
        """
        Run fragmentation test under memory pressure.

        Simulates the alternating pattern:
        - Allocate block (fixed size)
        - Allocate activation (varying size)
        - Free oldest block
        - Free oldest activation
        - Repeat
        """
        if activation_sizes_mb is None:
            activation_sizes_mb = [50, 100, 75, 150, 125, 80, 110, 90, 130, 70]

        memory_layout_samples = []
        num_failures = 0

        # Set base pressure
        self.pressure_manager.fill_to_pressure(target_free_gb)
        base_allocated = self.pressure_manager.get_allocated_gb()

        # Working storage
        blocks = []
        activations = []
        working_set_blocks = 3
        working_set_acts = 4

        block_elements = int(block_size_mb * 1024 * 1024 / 4)

        for swap_idx in range(num_swaps):
            # Allocate new block
            try:
                block = torch.randn(block_elements, dtype=torch.float32, device=self.device)
                blocks.append(block)
            except RuntimeError:
                num_failures += 1
                torch.cuda.empty_cache()

            # Allocate activation (varying size)
            act_size = activation_sizes_mb[swap_idx % len(activation_sizes_mb)]
            act_elements = int(act_size * 1024 * 1024 / 4)
            try:
                act = torch.randn(act_elements, dtype=torch.float32, device=self.device)
                activations.append(act)
            except RuntimeError:
                num_failures += 1
                torch.cuda.empty_cache()

            # Free oldest to maintain working set
            while len(blocks) > working_set_blocks:
                del blocks[0]
            while len(activations) > working_set_acts:
                del activations[0]

            # Sample periodically
            if swap_idx % 10 == 0:
                memory_layout_samples.append({
                    'swap_idx': swap_idx,
                    'allocated_gb': torch.cuda.memory_allocated(self.device) / (1024**3),
                    'reserved_gb': torch.cuda.memory_reserved(self.device) / (1024**3),
                })

        # Measure final fragmentation state
        # Clear working set but don't empty cache
        blocks.clear()
        activations.clear()

        # DON'T call empty_cache - we want to see fragmentation
        torch.cuda.synchronize(self.device)

        allocated = torch.cuda.memory_allocated(self.device) / (1024**3)
        reserved = torch.cuda.memory_reserved(self.device) / (1024**3)

        # Free in reserved pool (fragmented)
        free_in_pool = reserved - allocated

        # Try to allocate target block size
        target_alloc_success = False
        try:
            test = torch.randn(block_elements, dtype=torch.float32, device=self.device)
            target_alloc_success = True
            del test
        except RuntimeError:
            pass

        # Estimate largest contiguous
        largest = estimate_largest_free_block(self.device) / (1024**3)

        # Calculate fragmentation
        total_free = self.pressure_manager.get_free_memory_gb() + free_in_pool
        frag_ratio = 1.0 - (largest / total_free) if total_free > 0 else 0

        self.pressure_manager.clear()

        return FragmentationResult(
            num_swaps=num_swaps,
            base_pressure_gb=base_allocated,
            total_free_gb=total_free,
            largest_contiguous_gb=largest,
            fragmentation_ratio=frag_ratio,
            target_allocation_success=target_alloc_success,
            num_allocation_failures=num_failures,
            memory_layout_samples=memory_layout_samples
        )

    def run_experiment(
        self,
        block_size_mb: float = 650,
        target_free_gb: float = 3.0,
        swap_counts: List[int] = None
    ) -> Dict:
        """Run fragmentation experiment with varying swap counts"""
        if swap_counts is None:
            swap_counts = [20, 50, 100, 200, 500]

        print(f"\n{'='*60}")
        print("EXPERIMENT 2: Fragmentation Accumulation (with pressure)")
        print(f"{'='*60}")
        print(f"Block size: {block_size_mb} MB")
        print(f"Target free memory: {target_free_gb} GB")
        print(f"Swap counts: {swap_counts}")

        self.results = []
        results_by_swaps = {}

        for num_swaps in swap_counts:
            print(f"\n  Testing {num_swaps} swaps...")

            result = self.run_fragmentation_test(
                target_free_gb=target_free_gb,
                block_size_mb=block_size_mb,
                num_swaps=num_swaps
            )
            self.results.append(result)

            results_by_swaps[num_swaps] = {
                'total_free_gb': result.total_free_gb,
                'largest_contiguous_gb': result.largest_contiguous_gb,
                'fragmentation_ratio': result.fragmentation_ratio,
                'target_alloc_success': result.target_allocation_success,
                'failures': result.num_allocation_failures,
            }

            print(f"    Total free: {result.total_free_gb:.2f} GB")
            print(f"    Largest contiguous: {result.largest_contiguous_gb:.2f} GB")
            print(f"    Fragmentation: {result.fragmentation_ratio:.1%}")
            print(f"    {block_size_mb}MB alloc: {'SUCCESS' if result.target_allocation_success else 'FAILED'}")

        # Determine if fragmentation caused allocation failure
        any_failure = any(not r.target_allocation_success for r in self.results)
        max_frag = max(r.fragmentation_ratio for r in self.results)

        summary = {
            'block_size_mb': block_size_mb,
            'target_free_gb': target_free_gb,
            'results_by_swaps': results_by_swaps,
            'max_fragmentation': max_frag,
            'allocation_failures_observed': any_failure,
            'conclusion': 'FRAGMENTATION_CONFIRMED' if any_failure or max_frag > 0.3 else 'PARTIAL'
        }

        print(f"\nMax fragmentation: {max_frag:.1%}")
        print(f"Conclusion: {summary['conclusion']}")
        return summary


class PeakOverlapExperiment:
    """
    Experiment 3: Demonstrate peak overlap under memory pressure.

    When GPU is nearly full:
    - Async delete block A
    - Async load block B
    - Both are in flight simultaneously
    - Peak = A + B, which can exceed available memory
    """

    def __init__(self, device: int = 0):
        self.device = device
        self.pressure_manager = MemoryPressureManager(device)
        self.results: List[PeakOverlapResult] = []

    def run_single_trial(
        self,
        trial_id: int,
        block_size_mb: float,
        target_free_gb: float
    ) -> PeakOverlapResult:
        """Run single peak overlap test"""

        memory_timeline = []

        # Set pressure - leave room for ~2 blocks
        self.pressure_manager.fill_to_pressure(target_free_gb)
        base_pressure = self.pressure_manager.get_allocated_gb()

        block_elements = int(block_size_mb * 1024 * 1024 / 4)

        # Prepare CPU blocks (pinned for async transfer)
        cpu_block_a = torch.randn(block_elements, dtype=torch.float32).pin_memory()
        cpu_block_b = torch.randn(block_elements, dtype=torch.float32).pin_memory()

        # Load block A
        try:
            gpu_block_a = cpu_block_a.to(self.device)
            torch.cuda.synchronize(self.device)
        except RuntimeError:
            self.pressure_manager.clear()
            return PeakOverlapResult(
                trial_id=trial_id,
                base_pressure_gb=base_pressure,
                block_size_mb=block_size_mb,
                async_peak_gb=0,
                sync_peak_gb=0,
                oom_in_async=True,
                oom_in_sync=False,
                memory_timeline=[]
            )

        memory_timeline.append({
            'event': 'block_a_loaded',
            'allocated_gb': torch.cuda.memory_allocated(self.device) / (1024**3),
        })

        # ASYNC TEST: Delete A and immediately load B
        oom_in_async = False
        async_peak = torch.cuda.memory_allocated(self.device)

        stream_a = torch.cuda.Stream(device=self.device)
        stream_b = torch.cuda.Stream(device=self.device)

        # Start async delete (copy to CPU then delete)
        with torch.cuda.stream(stream_a):
            _ = gpu_block_a.to('cpu', non_blocking=True)

        # Delete reference but memory might not be freed yet
        del gpu_block_a

        # Immediately try to load B (async)
        gpu_block_b = None
        try:
            with torch.cuda.stream(stream_b):
                gpu_block_b = cpu_block_b.to(self.device, non_blocking=True)

            # Sample peak during overlap
            for _ in range(20):
                current = torch.cuda.memory_allocated(self.device)
                async_peak = max(async_peak, current)
                time.sleep(0.0005)

            torch.cuda.synchronize(self.device)

        except RuntimeError as e:
            if 'out of memory' in str(e).lower():
                oom_in_async = True
            else:
                raise

        memory_timeline.append({
            'event': 'async_test_done',
            'peak_gb': async_peak / (1024**3),
            'oom': oom_in_async,
        })

        # Cleanup async test
        if gpu_block_b is not None:
            del gpu_block_b
        torch.cuda.synchronize(self.device)
        torch.cuda.empty_cache()

        # SYNC TEST: Proper sync between delete and load
        oom_in_sync = False
        sync_peak = 0

        try:
            # Load A
            gpu_block_a = cpu_block_a.to(self.device)
            torch.cuda.synchronize(self.device)
            sync_peak = torch.cuda.memory_allocated(self.device)

            # Sync delete A
            del gpu_block_a
            torch.cuda.synchronize(self.device)
            torch.cuda.empty_cache()

            # Now load B
            gpu_block_b = cpu_block_b.to(self.device)
            torch.cuda.synchronize(self.device)
            sync_peak = max(sync_peak, torch.cuda.memory_allocated(self.device))

            del gpu_block_b

        except RuntimeError as e:
            if 'out of memory' in str(e).lower():
                oom_in_sync = True
            else:
                raise

        memory_timeline.append({
            'event': 'sync_test_done',
            'peak_gb': sync_peak / (1024**3),
            'oom': oom_in_sync,
        })

        self.pressure_manager.clear()

        return PeakOverlapResult(
            trial_id=trial_id,
            base_pressure_gb=base_pressure,
            block_size_mb=block_size_mb,
            async_peak_gb=async_peak / (1024**3),
            sync_peak_gb=sync_peak / (1024**3),
            oom_in_async=oom_in_async,
            oom_in_sync=oom_in_sync,
            memory_timeline=memory_timeline
        )

    def run_experiment(
        self,
        block_size_mb: float = 650,
        num_trials: int = 30,
        free_memory_ratios: List[float] = None
    ) -> Dict:
        """Run peak overlap experiment with varying memory pressure"""
        if free_memory_ratios is None:
            # Test with 1.5x, 2.0x, 2.5x, 3.0x block size free
            free_memory_ratios = [1.5, 2.0, 2.5, 3.0]

        print(f"\n{'='*60}")
        print("EXPERIMENT 3: Peak Overlap (with memory pressure)")
        print(f"{'='*60}")
        print(f"Block size: {block_size_mb} MB")
        print(f"Free memory ratios: {free_memory_ratios}")

        results_by_ratio = {}

        for ratio in free_memory_ratios:
            target_free_gb = (block_size_mb * ratio) / 1024
            print(f"\n  Testing with {ratio}x block size free ({target_free_gb:.2f} GB)...")

            trial_results = []
            for trial in range(num_trials):
                result = self.run_single_trial(trial, block_size_mb, target_free_gb)
                trial_results.append(result)
                self.results.append(result)

            async_oom_rate = sum(1 for r in trial_results if r.oom_in_async) / num_trials
            sync_oom_rate = sum(1 for r in trial_results if r.oom_in_sync) / num_trials

            valid_results = [r for r in trial_results if not r.oom_in_async]
            if valid_results:
                avg_async_peak = np.mean([r.async_peak_gb for r in valid_results])
                avg_sync_peak = np.mean([r.sync_peak_gb for r in valid_results])
            else:
                avg_async_peak = 0
                avg_sync_peak = 0

            results_by_ratio[ratio] = {
                'target_free_gb': target_free_gb,
                'async_oom_rate': async_oom_rate,
                'sync_oom_rate': sync_oom_rate,
                'avg_async_peak_gb': avg_async_peak,
                'avg_sync_peak_gb': avg_sync_peak,
                'peak_difference_gb': avg_async_peak - avg_sync_peak,
            }

            print(f"    Async OOM rate: {async_oom_rate:.1%}")
            print(f"    Sync OOM rate: {sync_oom_rate:.1%}")
            print(f"    Peak difference: {avg_async_peak - avg_sync_peak:.2f} GB")

        # Summary
        max_async_oom = max(r['async_oom_rate'] for r in results_by_ratio.values())
        any_sync_oom = any(r['sync_oom_rate'] > 0 for r in results_by_ratio.values())

        summary = {
            'block_size_mb': block_size_mb,
            'num_trials': num_trials,
            'results_by_ratio': results_by_ratio,
            'max_async_oom_rate': max_async_oom,
            'sync_caused_oom': any_sync_oom,
            'conclusion': 'PEAK_OVERLAP_CONFIRMED' if max_async_oom > 0.1 else 'PARTIAL'
        }

        print(f"\nMax async OOM rate: {max_async_oom:.1%}")
        print(f"Sync caused OOM: {any_sync_oom}")
        print(f"Conclusion: {summary['conclusion']}")
        return summary


def run_all_failure_experiments(
    output_dir: str = "./results/failure_experiments",
    block_size_mb: float = None,
) -> Dict:
    """Run all three failure mechanism experiments with proper memory pressure"""
    os.makedirs(output_dir, exist_ok=True)

    print("\n" + "="*70)
    print("FAILURE MECHANISM EXPERIMENTS (with memory pressure)")
    print("="*70)

    summary = get_memory_summary()
    print(f"\nGPU Memory Summary:")
    for k, v in summary.items():
        print(f"  {k}: {v}")

    # Auto-detect block size based on GPU
    if block_size_mb is None:
        total_gb = summary['total_memory_gb']
        # Use ~3% of GPU memory as block size (realistic for 60-block model)
        block_size_mb = total_gb * 1024 * 0.03
        block_size_mb = max(300, min(800, block_size_mb))  # Clamp to reasonable range

    print(f"\nUsing block size: {block_size_mb:.0f} MB")

    results = {}

    # Experiment 1: Deferred Reclamation
    print("\n" + "-"*60)
    exp1 = DeferredReclamationExperiment()
    results['deferred_reclamation'] = exp1.run_experiment(
        block_size_mb=block_size_mb,
        num_trials=20,
        free_memory_ratios=[1.2, 1.5, 2.0, 3.0]
    )

    # Experiment 2: Fragmentation
    print("\n" + "-"*60)
    # Leave enough free for block + some activations
    target_free = (block_size_mb * 4) / 1024  # 4x block size
    exp2 = FragmentationExperiment()
    results['fragmentation'] = exp2.run_experiment(
        block_size_mb=block_size_mb,
        target_free_gb=target_free,
        swap_counts=[20, 50, 100, 200]
    )

    # Experiment 3: Peak Overlap
    print("\n" + "-"*60)
    exp3 = PeakOverlapExperiment()
    results['peak_overlap'] = exp3.run_experiment(
        block_size_mb=block_size_mb,
        num_trials=20,
        free_memory_ratios=[1.5, 2.0, 2.5, 3.0]
    )

    # Save results
    results_path = os.path.join(output_dir, 'failure_experiments_results.json')
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)

    # Print summary table
    print(f"\n{'='*70}")
    print("FAILURE EXPERIMENTS SUMMARY")
    print(f"{'='*70}")
    print(f"{'Experiment':<25} {'Result':<20} {'Key Metric'}")
    print("-"*70)

    dr = results['deferred_reclamation']
    print(f"{'Deferred Reclamation':<25} {dr['conclusion']:<20} "
          f"Max rate: {dr['max_deferred_reclamation_rate']:.1%}")

    fr = results['fragmentation']
    print(f"{'Fragmentation':<25} {fr['conclusion']:<20} "
          f"Max frag: {fr['max_fragmentation']:.1%}")

    po = results['peak_overlap']
    print(f"{'Peak Overlap':<25} {po['conclusion']:<20} "
          f"Max async OOM: {po['max_async_oom_rate']:.1%}")

    print("="*70)
    print(f"Results saved to: {results_path}")

    return results


if __name__ == "__main__":
    run_all_failure_experiments()
