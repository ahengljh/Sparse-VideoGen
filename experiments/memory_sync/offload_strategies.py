"""
Offload Strategies for Video DiT Block-Level Memory Management

This module implements three transfer strategies:
1. Pure Async (baseline) - cudaMemcpyAsync without synchronization
2. Pure Sync - Full synchronization after each transfer
3. Conditional Sync - Adaptive synchronization based on memory state

These strategies are designed to validate the paper's claims about
synchronization necessity in Video DiT inference.
"""

import torch
import torch.nn as nn
import time
from typing import Dict, List, Optional, Tuple, Callable
from dataclasses import dataclass
from enum import Enum
import numpy as np

from memory_monitor import MemoryMonitor, MemorySnapshot, get_memory_summary


class TransferStrategy(Enum):
    PURE_ASYNC = "pure_async"
    PURE_SYNC = "pure_sync"
    CONDITIONAL_SYNC = "conditional_sync"


@dataclass
class TransferStats:
    """Statistics for a single transfer operation"""
    block_id: int
    operation: str  # 'load' or 'offload'
    size_mb: float
    duration_ms: float
    sync_triggered: bool
    memory_before_gb: float
    memory_after_gb: float
    success: bool
    error_msg: str = ""


@dataclass
class StrategyStats:
    """Aggregated statistics for a strategy"""
    strategy: TransferStrategy
    total_transfers: int
    successful_transfers: int
    failed_transfers: int
    oom_count: int
    total_sync_count: int
    sync_trigger_rate: float
    avg_transfer_time_ms: float
    peak_memory_gb: float
    peak_variance_gb: float
    total_time_s: float


class BlockOffloader:
    """
    Manages block-level offloading between GPU and CPU.

    This class simulates the offloading behavior of Video DiT models
    where transformer blocks are swapped between GPU and CPU memory.
    """

    def __init__(
        self,
        device: int = 0,
        cpu_pin_memory: bool = True,
        num_streams: int = 2,
    ):
        self.device = torch.device(f'cuda:{device}')
        self.cpu_device = torch.device('cpu')
        self.cpu_pin_memory = cpu_pin_memory

        # Create CUDA streams for async transfers
        self.load_stream = torch.cuda.Stream(device=device)
        self.offload_stream = torch.cuda.Stream(device=device)
        self.compute_stream = torch.cuda.default_stream(device=device)

        # Block storage
        self.cpu_blocks: Dict[int, torch.Tensor] = {}
        self.gpu_blocks: Dict[int, torch.Tensor] = {}

        # Statistics
        self.transfer_stats: List[TransferStats] = []
        self.sync_count = 0
        self.oom_count = 0

        # Conditional sync parameters
        self.safety_margin_gb = 2.0  # Buffer for safe operation
        self.recent_failure_window = 10
        self.recent_failures: List[float] = []

    def create_block(self, block_id: int, size_mb: float = 650) -> torch.Tensor:
        """Create a simulated block on CPU"""
        num_elements = int(size_mb * 1024 * 1024 / 4)  # float32
        tensor = torch.randn(num_elements, dtype=torch.float32)
        if self.cpu_pin_memory:
            tensor = tensor.pin_memory()
        self.cpu_blocks[block_id] = tensor
        return tensor

    def _get_free_memory_gb(self) -> float:
        """Get current free GPU memory in GB"""
        torch.cuda.synchronize(self.device)
        props = torch.cuda.get_device_properties(self.device)
        reserved = torch.cuda.memory_reserved(self.device)
        return (props.total_memory - reserved) / (1024**3)

    def _get_allocated_memory_gb(self) -> float:
        """Get current allocated GPU memory in GB"""
        return torch.cuda.memory_allocated(self.device) / (1024**3)

    def _should_sync_conditional(
        self,
        block_size_mb: float,
        operation: str
    ) -> Tuple[bool, str]:
        """
        Determine if synchronization is needed based on current state.

        Implements the conditional sync decision logic from the paper:
        1. Memory margin insufficient
        2. Recent allocation failures
        3. Phase transition detected (not implemented in simulation)

        Returns: (should_sync, reason)
        """
        block_size_gb = block_size_mb / 1024
        free_memory_gb = self._get_free_memory_gb()

        # Condition 1: Memory margin insufficient
        if free_memory_gb < block_size_gb + self.safety_margin_gb:
            return True, "memory_margin_insufficient"

        # Condition 2: Recent failures
        current_time = time.time()
        # Clean old failures
        self.recent_failures = [
            t for t in self.recent_failures
            if current_time - t < self.recent_failure_window
        ]
        if len(self.recent_failures) > 0:
            return True, "recent_failures"

        # Condition 3: High fragmentation (estimated)
        allocated = self._get_allocated_memory_gb()
        reserved = torch.cuda.memory_reserved(self.device) / (1024**3)
        if reserved > 0 and (reserved - allocated) / reserved > 0.3:
            return True, "high_fragmentation"

        return False, "no_sync_needed"

    def load_block_async(
        self,
        block_id: int,
        strategy: TransferStrategy = TransferStrategy.PURE_ASYNC
    ) -> TransferStats:
        """Load a block from CPU to GPU using specified strategy"""
        if block_id not in self.cpu_blocks:
            raise ValueError(f"Block {block_id} not found in CPU storage")

        cpu_tensor = self.cpu_blocks[block_id]
        size_mb = cpu_tensor.numel() * 4 / (1024 * 1024)
        memory_before = self._get_allocated_memory_gb()
        sync_triggered = False
        success = True
        error_msg = ""

        start_time = time.time()

        try:
            if strategy == TransferStrategy.PURE_ASYNC:
                # Pure async: no synchronization
                with torch.cuda.stream(self.load_stream):
                    gpu_tensor = cpu_tensor.to(self.device, non_blocking=True)
                self.gpu_blocks[block_id] = gpu_tensor

            elif strategy == TransferStrategy.PURE_SYNC:
                # Pure sync: synchronize after transfer
                with torch.cuda.stream(self.load_stream):
                    gpu_tensor = cpu_tensor.to(self.device, non_blocking=True)
                torch.cuda.synchronize(self.device)
                self.gpu_blocks[block_id] = gpu_tensor
                sync_triggered = True
                self.sync_count += 1

            elif strategy == TransferStrategy.CONDITIONAL_SYNC:
                # Conditional sync: check if sync needed
                should_sync, reason = self._should_sync_conditional(size_mb, 'load')

                if should_sync:
                    # Synchronize before to ensure previous offloads complete
                    torch.cuda.synchronize(self.device)
                    # Trigger memory cleanup
                    torch.cuda.empty_cache()

                with torch.cuda.stream(self.load_stream):
                    gpu_tensor = cpu_tensor.to(self.device, non_blocking=True)

                if should_sync:
                    torch.cuda.synchronize(self.device)
                    sync_triggered = True
                    self.sync_count += 1

                self.gpu_blocks[block_id] = gpu_tensor

        except RuntimeError as e:
            if 'out of memory' in str(e).lower():
                success = False
                error_msg = str(e)
                self.oom_count += 1
                self.recent_failures.append(time.time())
                torch.cuda.empty_cache()
            else:
                raise

        duration_ms = (time.time() - start_time) * 1000
        memory_after = self._get_allocated_memory_gb()

        stats = TransferStats(
            block_id=block_id,
            operation='load',
            size_mb=size_mb,
            duration_ms=duration_ms,
            sync_triggered=sync_triggered,
            memory_before_gb=memory_before,
            memory_after_gb=memory_after,
            success=success,
            error_msg=error_msg
        )
        self.transfer_stats.append(stats)
        return stats

    def offload_block_async(
        self,
        block_id: int,
        strategy: TransferStrategy = TransferStrategy.PURE_ASYNC
    ) -> TransferStats:
        """Offload a block from GPU to CPU using specified strategy"""
        if block_id not in self.gpu_blocks:
            raise ValueError(f"Block {block_id} not found in GPU storage")

        gpu_tensor = self.gpu_blocks[block_id]
        size_mb = gpu_tensor.numel() * 4 / (1024 * 1024)
        memory_before = self._get_allocated_memory_gb()
        sync_triggered = False
        success = True
        error_msg = ""

        start_time = time.time()

        try:
            if strategy == TransferStrategy.PURE_ASYNC:
                # Pure async: just mark for deletion
                with torch.cuda.stream(self.offload_stream):
                    # Copy back to CPU if needed (usually we already have it)
                    pass
                # Delete GPU tensor
                del self.gpu_blocks[block_id]
                # Note: actual memory release is deferred!

            elif strategy == TransferStrategy.PURE_SYNC:
                # Pure sync: ensure completion
                del self.gpu_blocks[block_id]
                torch.cuda.synchronize(self.device)
                torch.cuda.empty_cache()
                sync_triggered = True
                self.sync_count += 1

            elif strategy == TransferStrategy.CONDITIONAL_SYNC:
                should_sync, reason = self._should_sync_conditional(size_mb, 'offload')

                del self.gpu_blocks[block_id]

                if should_sync:
                    torch.cuda.synchronize(self.device)
                    torch.cuda.empty_cache()
                    sync_triggered = True
                    self.sync_count += 1

        except RuntimeError as e:
            success = False
            error_msg = str(e)

        duration_ms = (time.time() - start_time) * 1000
        memory_after = self._get_allocated_memory_gb()

        stats = TransferStats(
            block_id=block_id,
            operation='offload',
            size_mb=size_mb,
            duration_ms=duration_ms,
            sync_triggered=sync_triggered,
            memory_before_gb=memory_before,
            memory_after_gb=memory_after,
            success=success,
            error_msg=error_msg
        )
        self.transfer_stats.append(stats)
        return stats

    def get_strategy_stats(self, strategy: TransferStrategy) -> StrategyStats:
        """Compile statistics for a strategy run"""
        total = len(self.transfer_stats)
        successful = sum(1 for s in self.transfer_stats if s.success)
        failed = total - successful
        total_sync = sum(1 for s in self.transfer_stats if s.sync_triggered)
        sync_rate = total_sync / total if total > 0 else 0.0
        avg_time = np.mean([s.duration_ms for s in self.transfer_stats]) if total > 0 else 0.0

        peak_memory = max(s.memory_after_gb for s in self.transfer_stats) if total > 0 else 0.0
        memory_values = [s.memory_after_gb for s in self.transfer_stats]
        peak_variance = np.var(memory_values) if memory_values else 0.0

        total_time = sum(s.duration_ms for s in self.transfer_stats) / 1000

        return StrategyStats(
            strategy=strategy,
            total_transfers=total,
            successful_transfers=successful,
            failed_transfers=failed,
            oom_count=self.oom_count,
            total_sync_count=total_sync,
            sync_trigger_rate=sync_rate,
            avg_transfer_time_ms=avg_time,
            peak_memory_gb=peak_memory,
            peak_variance_gb=peak_variance,
            total_time_s=total_time
        )

    def reset_stats(self):
        """Reset all statistics"""
        self.transfer_stats = []
        self.sync_count = 0
        self.oom_count = 0
        self.recent_failures = []

    def clear_gpu_blocks(self):
        """Clear all GPU blocks"""
        self.gpu_blocks.clear()
        torch.cuda.empty_cache()


class VideoditBlockSimulator:
    """
    Simulates Video DiT block execution patterns.

    Models the execution pattern:
    - T denoising steps (40-50)
    - N transformer blocks per step (60)
    - K blocks kept on GPU (working set)
    """

    def __init__(
        self,
        num_blocks: int = 60,
        num_steps: int = 40,
        block_size_mb: float = 650,
        working_set_size: int = 5,
        device: int = 0,
    ):
        self.num_blocks = num_blocks
        self.num_steps = num_steps
        self.block_size_mb = block_size_mb
        self.working_set_size = working_set_size
        self.device = device

        self.offloader = BlockOffloader(device=device)

        # Initialize blocks on CPU
        for i in range(num_blocks):
            self.offloader.create_block(i, block_size_mb)

    def simulate_inference(
        self,
        strategy: TransferStrategy,
        monitor: Optional[MemoryMonitor] = None,
        compute_time_ms: float = 10.0,  # Simulated compute time per block
    ) -> StrategyStats:
        """
        Simulate Video DiT inference with given strategy.

        Execution pattern:
        for t in range(T):  # denoising steps
            for i in range(N):  # transformer blocks
                load_block(i) if not on GPU
                compute(block_i)
                offload if needed to maintain working set
        """
        self.offloader.reset_stats()
        self.offloader.clear_gpu_blocks()
        torch.cuda.empty_cache()

        # Track which blocks are on GPU
        gpu_block_ids: List[int] = []

        total_block_accesses = 0
        start_time = time.time()

        try:
            for step in range(self.num_steps):
                if monitor:
                    monitor.record_event(f"step_{step}_start")

                for block_id in range(self.num_blocks):
                    total_block_accesses += 1

                    # Load block if not on GPU
                    if block_id not in gpu_block_ids:
                        # If working set is full, offload oldest block
                        while len(gpu_block_ids) >= self.working_set_size:
                            oldest_id = gpu_block_ids.pop(0)
                            self.offloader.offload_block_async(oldest_id, strategy)

                        # Load new block
                        stats = self.offloader.load_block_async(block_id, strategy)
                        if stats.success:
                            gpu_block_ids.append(block_id)
                        else:
                            # OOM occurred
                            if monitor:
                                monitor.record_oom()
                            # Try to recover
                            torch.cuda.empty_cache()

                    # Simulate computation
                    if compute_time_ms > 0:
                        time.sleep(compute_time_ms / 1000)

                if monitor:
                    monitor.record_event(f"step_{step}_end")

        except Exception as e:
            print(f"Simulation failed: {e}")

        total_time = time.time() - start_time

        stats = self.offloader.get_strategy_stats(strategy)
        stats.total_time_s = total_time

        print(f"\n[{strategy.value}] Simulation completed:")
        print(f"  Total block accesses: {total_block_accesses}")
        print(f"  OOM count: {stats.oom_count}")
        print(f"  Sync trigger rate: {stats.sync_trigger_rate:.2%}")
        print(f"  Peak memory: {stats.peak_memory_gb:.2f} GB")
        print(f"  Peak variance: {stats.peak_variance_gb:.4f} GB²")
        print(f"  Total time: {stats.total_time_s:.2f}s")

        return stats


def run_strategy_comparison(
    num_blocks: int = 60,
    num_steps: int = 10,  # Reduced for testing
    block_size_mb: float = 200,  # Reduced for testing
    working_set_size: int = 5,
    num_trials: int = 3,
) -> Dict[str, List[StrategyStats]]:
    """
    Run comparison of all three strategies.

    Returns statistics for each strategy across multiple trials.
    """
    results = {
        TransferStrategy.PURE_ASYNC.value: [],
        TransferStrategy.PURE_SYNC.value: [],
        TransferStrategy.CONDITIONAL_SYNC.value: [],
    }

    for strategy in [TransferStrategy.PURE_ASYNC, TransferStrategy.PURE_SYNC, TransferStrategy.CONDITIONAL_SYNC]:
        print(f"\n{'='*60}")
        print(f"Testing strategy: {strategy.value}")
        print(f"{'='*60}")

        for trial in range(num_trials):
            print(f"\n  Trial {trial + 1}/{num_trials}")

            simulator = VideoditBlockSimulator(
                num_blocks=num_blocks,
                num_steps=num_steps,
                block_size_mb=block_size_mb,
                working_set_size=working_set_size,
            )

            monitor = MemoryMonitor(interval_ms=50)
            monitor.start()

            try:
                stats = simulator.simulate_inference(strategy, monitor)
                results[strategy.value].append(stats)
            finally:
                monitor.stop()

            # Cleanup between trials
            torch.cuda.empty_cache()
            time.sleep(1)

    return results


def print_comparison_table(results: Dict[str, List[StrategyStats]]):
    """Print comparison table in paper format"""
    print("\n" + "="*80)
    print("QUANTITATIVE COMPARISON RESULTS")
    print("="*80)
    print(f"{'Strategy':<20} {'OOM Rate':<12} {'Peak Memory':<15} {'Peak Variance':<15} {'Sync Rate':<12}")
    print("-"*80)

    for strategy_name, stats_list in results.items():
        if not stats_list:
            continue

        # Calculate averages
        oom_rate = np.mean([s.oom_count > 0 for s in stats_list]) * 100
        peak_memory = np.mean([s.peak_memory_gb for s in stats_list])
        peak_variance = np.mean([s.peak_variance_gb for s in stats_list])
        sync_rate = np.mean([s.sync_trigger_rate for s in stats_list]) * 100

        print(f"{strategy_name:<20} {oom_rate:>8.1f}% {peak_memory:>12.2f} GB {peak_variance:>11.4f} GB² {sync_rate:>8.1f}%")

    print("="*80)


if __name__ == "__main__":
    print("Offload Strategy Comparison Test")
    print("="*50)

    # Run quick test
    results = run_strategy_comparison(
        num_blocks=20,
        num_steps=5,
        block_size_mb=100,
        working_set_size=3,
        num_trials=2,
    )

    print_comparison_table(results)
