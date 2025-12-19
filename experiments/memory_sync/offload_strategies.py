"""
Offload Strategies for Video DiT Block-Level Memory Management (v2)

This module implements three transfer strategies with proper memory pressure:
1. Pure Async (baseline) - cudaMemcpyAsync without synchronization
2. Pure Sync - Full synchronization after each transfer
3. Conditional Sync - Adaptive synchronization based on memory state

KEY FIX: Added memory pressure simulation to create realistic OOM scenarios.
"""

import torch
import torch.nn as nn
import time
import gc
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


class ActivationSimulator:
    """
    Simulates activation memory pressure during Video DiT inference.

    In real inference:
    - Activations take 12-15GB for video generation
    - This leaves only 8-12GB for model weights on 24GB GPU
    - Block offloading must work within this constraint
    """

    def __init__(self, device: int = 0):
        self.device = device
        self.activation_tensors: List[torch.Tensor] = []

    def get_allocated_gb(self) -> float:
        return torch.cuda.memory_allocated(self.device) / (1024**3)

    def get_free_gb(self) -> float:
        props = torch.cuda.get_device_properties(self.device)
        reserved = torch.cuda.memory_reserved(self.device)
        return (props.total_memory - reserved) / (1024**3)

    def allocate_activation_pressure(self, target_free_gb: float) -> float:
        """
        Allocate tensors to leave only target_free_gb available.

        This simulates the activation memory that would be present
        during real Video DiT inference.
        """
        self.clear()
        torch.cuda.empty_cache()

        chunk_size = 256 * 1024 * 1024  # 256MB chunks
        chunk_elements = chunk_size // 4

        while self.get_free_gb() > target_free_gb + 0.3:
            try:
                tensor = torch.randn(chunk_elements, dtype=torch.float32, device=self.device)
                self.activation_tensors.append(tensor)
            except RuntimeError:
                break

        # Fine tune with smaller chunks
        small_chunk = 64 * 1024 * 1024 // 4
        while self.get_free_gb() > target_free_gb + 0.1:
            try:
                tensor = torch.randn(small_chunk, dtype=torch.float32, device=self.device)
                self.activation_tensors.append(tensor)
            except RuntimeError:
                break

        return self.get_free_gb()

    def clear(self):
        """Release all activation tensors"""
        self.activation_tensors.clear()
        gc.collect()
        torch.cuda.empty_cache()


class BlockOffloader:
    """
    Manages block-level offloading between GPU and CPU.
    """

    def __init__(
        self,
        device: int = 0,
        cpu_pin_memory: bool = True,
    ):
        self.device = torch.device(f'cuda:{device}')
        self.device_id = device
        self.cpu_device = torch.device('cpu')
        self.cpu_pin_memory = cpu_pin_memory

        # CUDA streams
        self.load_stream = torch.cuda.Stream(device=device)
        self.offload_stream = torch.cuda.Stream(device=device)

        # Block storage
        self.cpu_blocks: Dict[int, torch.Tensor] = {}
        self.gpu_blocks: Dict[int, torch.Tensor] = {}

        # Statistics
        self.transfer_stats: List[TransferStats] = []
        self.sync_count = 0
        self.oom_count = 0

        # Conditional sync parameters
        self.safety_margin_gb = 1.5
        self.recent_failures: List[float] = []
        self.recent_failure_window = 5.0

    def create_block(self, block_id: int, size_mb: float = 650) -> torch.Tensor:
        """Create a simulated block on CPU"""
        num_elements = int(size_mb * 1024 * 1024 / 4)
        tensor = torch.randn(num_elements, dtype=torch.float32)
        if self.cpu_pin_memory:
            tensor = tensor.pin_memory()
        self.cpu_blocks[block_id] = tensor
        return tensor

    def _get_free_memory_gb(self) -> float:
        props = torch.cuda.get_device_properties(self.device_id)
        reserved = torch.cuda.memory_reserved(self.device_id)
        return (props.total_memory - reserved) / (1024**3)

    def _get_allocated_memory_gb(self) -> float:
        return torch.cuda.memory_allocated(self.device_id) / (1024**3)

    def _should_sync_conditional(self, block_size_mb: float) -> Tuple[bool, str]:
        """Determine if sync is needed based on current state"""
        block_size_gb = block_size_mb / 1024
        free_memory_gb = self._get_free_memory_gb()

        # Condition 1: Memory margin insufficient
        if free_memory_gb < block_size_gb + self.safety_margin_gb:
            return True, "memory_margin"

        # Condition 2: Recent failures
        current_time = time.time()
        self.recent_failures = [
            t for t in self.recent_failures
            if current_time - t < self.recent_failure_window
        ]
        if self.recent_failures:
            return True, "recent_failures"

        return False, ""

    def load_block(
        self,
        block_id: int,
        strategy: TransferStrategy
    ) -> TransferStats:
        """Load a block from CPU to GPU using specified strategy"""
        if block_id not in self.cpu_blocks:
            raise ValueError(f"Block {block_id} not found")

        cpu_tensor = self.cpu_blocks[block_id]
        size_mb = cpu_tensor.numel() * 4 / (1024 * 1024)
        memory_before = self._get_allocated_memory_gb()
        sync_triggered = False
        success = True
        error_msg = ""

        start_time = time.time()

        try:
            if strategy == TransferStrategy.PURE_ASYNC:
                with torch.cuda.stream(self.load_stream):
                    gpu_tensor = cpu_tensor.to(self.device, non_blocking=True)
                self.gpu_blocks[block_id] = gpu_tensor

            elif strategy == TransferStrategy.PURE_SYNC:
                gpu_tensor = cpu_tensor.to(self.device, non_blocking=False)
                torch.cuda.synchronize(self.device_id)
                self.gpu_blocks[block_id] = gpu_tensor
                sync_triggered = True
                self.sync_count += 1

            elif strategy == TransferStrategy.CONDITIONAL_SYNC:
                should_sync, reason = self._should_sync_conditional(size_mb)

                if should_sync:
                    torch.cuda.synchronize(self.device_id)
                    torch.cuda.empty_cache()
                    sync_triggered = True
                    self.sync_count += 1

                gpu_tensor = cpu_tensor.to(self.device, non_blocking=not should_sync)

                if should_sync:
                    torch.cuda.synchronize(self.device_id)

                self.gpu_blocks[block_id] = gpu_tensor

        except RuntimeError as e:
            if 'out of memory' in str(e).lower():
                success = False
                error_msg = "OOM"
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

    def offload_block(
        self,
        block_id: int,
        strategy: TransferStrategy
    ) -> TransferStats:
        """Offload a block from GPU"""
        if block_id not in self.gpu_blocks:
            raise ValueError(f"Block {block_id} not on GPU")

        gpu_tensor = self.gpu_blocks[block_id]
        size_mb = gpu_tensor.numel() * 4 / (1024 * 1024)
        memory_before = self._get_allocated_memory_gb()
        sync_triggered = False

        start_time = time.time()

        if strategy == TransferStrategy.PURE_ASYNC:
            del self.gpu_blocks[block_id]
            # No sync - memory release is deferred

        elif strategy == TransferStrategy.PURE_SYNC:
            del self.gpu_blocks[block_id]
            torch.cuda.synchronize(self.device_id)
            torch.cuda.empty_cache()
            sync_triggered = True
            self.sync_count += 1

        elif strategy == TransferStrategy.CONDITIONAL_SYNC:
            should_sync, _ = self._should_sync_conditional(size_mb)
            del self.gpu_blocks[block_id]

            if should_sync:
                torch.cuda.synchronize(self.device_id)
                torch.cuda.empty_cache()
                sync_triggered = True
                self.sync_count += 1

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
            success=True,
        )
        self.transfer_stats.append(stats)
        return stats

    def get_strategy_stats(self, strategy: TransferStrategy) -> StrategyStats:
        """Compile statistics"""
        total = len(self.transfer_stats)
        successful = sum(1 for s in self.transfer_stats if s.success)
        total_sync = sum(1 for s in self.transfer_stats if s.sync_triggered)

        peak_memory = max((s.memory_after_gb for s in self.transfer_stats), default=0)
        memory_values = [s.memory_after_gb for s in self.transfer_stats]
        peak_variance = float(np.var(memory_values)) if memory_values else 0.0
        avg_time = float(np.mean([s.duration_ms for s in self.transfer_stats])) if total > 0 else 0.0

        return StrategyStats(
            strategy=strategy,
            total_transfers=total,
            successful_transfers=successful,
            failed_transfers=total - successful,
            oom_count=self.oom_count,
            total_sync_count=total_sync,
            sync_trigger_rate=total_sync / total if total > 0 else 0.0,
            avg_transfer_time_ms=avg_time,
            peak_memory_gb=peak_memory,
            peak_variance_gb=peak_variance,
            total_time_s=sum(s.duration_ms for s in self.transfer_stats) / 1000
        )

    def reset_stats(self):
        self.transfer_stats = []
        self.sync_count = 0
        self.oom_count = 0
        self.recent_failures = []

    def clear_gpu_blocks(self):
        self.gpu_blocks.clear()
        torch.cuda.empty_cache()


class VideoditBlockSimulator:
    """
    Simulates Video DiT block execution with memory pressure.

    Key change: Now simulates realistic activation memory pressure
    to create actual OOM scenarios.
    """

    def __init__(
        self,
        num_blocks: int = 60,
        num_steps: int = 40,
        block_size_mb: float = 650,
        working_set_size: int = 5,
        device: int = 0,
        target_free_gb: float = None,  # NEW: target free memory
    ):
        self.num_blocks = num_blocks
        self.num_steps = num_steps
        self.block_size_mb = block_size_mb
        self.working_set_size = working_set_size
        self.device = device

        # Auto-detect target free if not specified
        if target_free_gb is None:
            # Leave enough for working_set + 1 block + small margin
            target_free_gb = (working_set_size + 2) * block_size_mb / 1024

        self.target_free_gb = target_free_gb

        self.offloader = BlockOffloader(device=device)
        self.activation_sim = ActivationSimulator(device=device)

        # Initialize blocks on CPU
        for i in range(num_blocks):
            self.offloader.create_block(i, block_size_mb)

    def simulate_inference(
        self,
        strategy: TransferStrategy,
        monitor: Optional[MemoryMonitor] = None,
        compute_time_ms: float = 1.0,
    ) -> StrategyStats:
        """
        Simulate Video DiT inference with memory pressure.
        """
        self.offloader.reset_stats()
        self.offloader.clear_gpu_blocks()
        self.activation_sim.clear()
        torch.cuda.empty_cache()

        # Create memory pressure to simulate activations
        actual_free = self.activation_sim.allocate_activation_pressure(self.target_free_gb)
        print(f"    Activation pressure set: {actual_free:.2f}GB free (target: {self.target_free_gb:.2f}GB)")

        gpu_block_ids: List[int] = []
        total_block_accesses = 0
        start_time = time.time()

        try:
            for step in range(self.num_steps):
                if monitor:
                    monitor.record_event(f"step_{step}")

                for block_id in range(self.num_blocks):
                    total_block_accesses += 1

                    # Load block if not on GPU
                    if block_id not in gpu_block_ids:
                        # Offload oldest if working set full
                        while len(gpu_block_ids) >= self.working_set_size:
                            oldest_id = gpu_block_ids.pop(0)
                            if oldest_id in self.offloader.gpu_blocks:
                                self.offloader.offload_block(oldest_id, strategy)

                        # Load new block
                        stats = self.offloader.load_block(block_id, strategy)
                        if stats.success:
                            gpu_block_ids.append(block_id)
                        else:
                            if monitor:
                                monitor.record_oom()
                            torch.cuda.empty_cache()

                    # Simulate computation
                    if compute_time_ms > 0:
                        time.sleep(compute_time_ms / 1000)

        except Exception as e:
            print(f"Simulation error: {e}")

        # Cleanup
        self.activation_sim.clear()
        self.offloader.clear_gpu_blocks()
        torch.cuda.empty_cache()

        total_time = time.time() - start_time
        stats = self.offloader.get_strategy_stats(strategy)
        stats.total_time_s = total_time

        print(f"\n[{strategy.value}] Completed:")
        print(f"  Block accesses: {total_block_accesses}")
        print(f"  OOM count: {stats.oom_count}")
        print(f"  Sync rate: {stats.sync_trigger_rate:.1%}")
        print(f"  Peak memory: {stats.peak_memory_gb:.2f} GB")

        return stats


def run_strategy_comparison(
    num_blocks: int = 30,
    num_steps: int = 5,
    block_size_mb: float = 500,
    working_set_size: int = 4,
    target_free_gb: float = None,
    num_trials: int = 3,
) -> Dict[str, List[StrategyStats]]:
    """
    Run comparison of all three strategies with memory pressure.
    """
    results = {
        TransferStrategy.PURE_ASYNC.value: [],
        TransferStrategy.PURE_SYNC.value: [],
        TransferStrategy.CONDITIONAL_SYNC.value: [],
    }

    # Auto-detect target_free if not specified
    if target_free_gb is None:
        # Tight constraint: working set + 1.5 blocks
        target_free_gb = (working_set_size + 1.5) * block_size_mb / 1024

    print(f"\n{'='*60}")
    print("STRATEGY COMPARISON (with memory pressure)")
    print(f"{'='*60}")
    print(f"Blocks: {num_blocks}, Steps: {num_steps}")
    print(f"Block size: {block_size_mb} MB, Working set: {working_set_size}")
    print(f"Target free memory: {target_free_gb:.2f} GB")

    for strategy in [TransferStrategy.PURE_ASYNC, TransferStrategy.PURE_SYNC, TransferStrategy.CONDITIONAL_SYNC]:
        print(f"\n--- {strategy.value} ---")

        for trial in range(num_trials):
            print(f"  Trial {trial + 1}/{num_trials}")

            simulator = VideoditBlockSimulator(
                num_blocks=num_blocks,
                num_steps=num_steps,
                block_size_mb=block_size_mb,
                working_set_size=working_set_size,
                target_free_gb=target_free_gb,
            )

            monitor = MemoryMonitor(interval_ms=50)
            monitor.start()

            try:
                stats = simulator.simulate_inference(strategy, monitor, compute_time_ms=0.5)
                results[strategy.value].append(stats)
            finally:
                monitor.stop()

            torch.cuda.empty_cache()
            time.sleep(0.5)

    return results


def print_comparison_table(results: Dict[str, List[StrategyStats]]):
    """Print results in paper format"""
    print(f"\n{'='*80}")
    print("RESULTS TABLE")
    print(f"{'='*80}")
    print(f"{'Strategy':<20} {'OOM Rate':<12} {'Peak Memory':<15} {'Peak Var':<12} {'Sync Rate':<12}")
    print("-"*80)

    for strategy_name, stats_list in results.items():
        if not stats_list:
            continue

        oom_rate = np.mean([1 if s.oom_count > 0 else 0 for s in stats_list]) * 100
        peak_memory = np.mean([s.peak_memory_gb for s in stats_list])
        peak_var = np.std([s.peak_memory_gb for s in stats_list])
        sync_rate = np.mean([s.sync_trigger_rate for s in stats_list]) * 100

        print(f"{strategy_name:<20} {oom_rate:>8.1f}%    {peak_memory:>10.2f} GB  ±{peak_var:>6.2f} GB  {sync_rate:>8.1f}%")

    print("="*80)


if __name__ == "__main__":
    # Get GPU info
    summary = get_memory_summary()
    print("GPU Info:")
    for k, v in summary.items():
        print(f"  {k}: {v}")

    # Run with tight memory constraints
    total_gb = summary['total_memory_gb']

    # Use ~3% of GPU as block size (realistic)
    block_size_mb = max(300, min(700, total_gb * 1024 * 0.03))

    # Target: leave room for working_set + 1 block
    working_set = 4
    target_free = (working_set + 1.2) * block_size_mb / 1024

    results = run_strategy_comparison(
        num_blocks=20,
        num_steps=3,
        block_size_mb=block_size_mb,
        working_set_size=working_set,
        target_free_gb=target_free,
        num_trials=2,
    )

    print_comparison_table(results)
