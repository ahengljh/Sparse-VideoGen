"""
Memory Synchronization Strategies for Video DiT Block-Level Offloading

This module implements three memory synchronization strategies:
1. Pure Async: Uses async memory operations without explicit synchronization
2. Pure Sync: Synchronizes after every memory operation
3. Conditional Sync: Only synchronizes when memory is tight

Based on the observation that Video DiT has unique execution patterns that
differ from LLMs, requiring specialized memory management.
"""

import torch
import torch.cuda as cuda
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple, Any
from enum import Enum
import time
import gc
import threading
from contextlib import contextmanager


class SyncStrategy(Enum):
    """Memory synchronization strategy types"""
    PURE_ASYNC = "pure_async"
    PURE_SYNC = "pure_sync"
    CONDITIONAL_SYNC = "conditional_sync"


@dataclass
class MemoryStats:
    """Statistics for memory usage tracking"""
    peak_memory_mb: float = 0.0
    current_memory_mb: float = 0.0
    allocated_memory_mb: float = 0.0
    cached_memory_mb: float = 0.0
    free_memory_mb: float = 0.0
    timestamp: float = 0.0


@dataclass
class ExperimentResult:
    """Results from a single experiment run"""
    strategy: str
    oom_occurred: bool = False
    oom_message: str = ""
    peak_memory_mb: float = 0.0
    memory_samples: List[MemoryStats] = field(default_factory=list)
    total_time_seconds: float = 0.0
    sync_count: int = 0
    load_count: int = 0
    offload_count: int = 0
    successful_steps: int = 0
    total_steps: int = 0

    @property
    def peak_variance_mb(self) -> float:
        """Calculate variance of peak memory samples"""
        if len(self.memory_samples) < 2:
            return 0.0
        peaks = [s.peak_memory_mb for s in self.memory_samples]
        mean = sum(peaks) / len(peaks)
        variance = sum((p - mean) ** 2 for p in peaks) / len(peaks)
        return variance ** 0.5  # Return std dev

    @property
    def avg_peak_memory_mb(self) -> float:
        """Average peak memory across samples"""
        if not self.memory_samples:
            return self.peak_memory_mb
        return sum(s.peak_memory_mb for s in self.memory_samples) / len(self.memory_samples)


class MemoryProfiler:
    """Utility for profiling GPU memory usage"""

    def __init__(self, device: int = 0, sample_interval: float = 0.01):
        self.device = device
        self.sample_interval = sample_interval
        self.samples: List[MemoryStats] = []
        self._monitoring = False
        self._monitor_thread: Optional[threading.Thread] = None

    def get_current_stats(self) -> MemoryStats:
        """Get current memory statistics"""
        torch.cuda.synchronize(self.device)

        allocated = torch.cuda.memory_allocated(self.device) / (1024 ** 2)
        cached = torch.cuda.memory_reserved(self.device) / (1024 ** 2)

        # Get device properties for total memory
        props = torch.cuda.get_device_properties(self.device)
        total = props.total_memory / (1024 ** 2)
        free = total - cached

        # Peak memory since last reset
        peak = torch.cuda.max_memory_allocated(self.device) / (1024 ** 2)

        return MemoryStats(
            peak_memory_mb=peak,
            current_memory_mb=allocated,
            allocated_memory_mb=allocated,
            cached_memory_mb=cached,
            free_memory_mb=free,
            timestamp=time.time()
        )

    def reset_peak_stats(self):
        """Reset peak memory statistics"""
        torch.cuda.reset_peak_memory_stats(self.device)

    def start_monitoring(self):
        """Start background memory monitoring"""
        self._monitoring = True
        self.samples = []
        self._monitor_thread = threading.Thread(target=self._monitor_loop)
        self._monitor_thread.daemon = True
        self._monitor_thread.start()

    def stop_monitoring(self) -> List[MemoryStats]:
        """Stop monitoring and return collected samples"""
        self._monitoring = False
        if self._monitor_thread:
            self._monitor_thread.join(timeout=1.0)
        return self.samples

    def _monitor_loop(self):
        """Background monitoring loop"""
        while self._monitoring:
            try:
                stats = self.get_current_stats()
                self.samples.append(stats)
            except Exception:
                pass
            time.sleep(self.sample_interval)

    @contextmanager
    def profile(self):
        """Context manager for profiling a code block"""
        self.reset_peak_stats()
        start_time = time.time()
        self.start_monitoring()
        try:
            yield self
        finally:
            self.stop_monitoring()
            self.total_time = time.time() - start_time


class BaseSyncStrategy(ABC):
    """Base class for memory synchronization strategies"""

    def __init__(self, device: int = 0, safety_margin_mb: float = 1000.0):
        self.device = device
        self.safety_margin_mb = safety_margin_mb
        self.sync_count = 0
        self.load_count = 0
        self.offload_count = 0
        self.recent_failures = 0

        # Create dedicated streams for memory operations
        self.load_stream = torch.cuda.Stream(device=device)
        self.offload_stream = torch.cuda.Stream(device=device)

    def reset_counters(self):
        """Reset operation counters"""
        self.sync_count = 0
        self.load_count = 0
        self.offload_count = 0
        self.recent_failures = 0

    def get_free_memory_mb(self) -> float:
        """Get current free GPU memory in MB"""
        props = torch.cuda.get_device_properties(self.device)
        total = props.total_memory
        reserved = torch.cuda.memory_reserved(self.device)
        return (total - reserved) / (1024 ** 2)

    def get_allocated_memory_mb(self) -> float:
        """Get currently allocated GPU memory in MB"""
        return torch.cuda.memory_allocated(self.device) / (1024 ** 2)

    @abstractmethod
    def load_block(self, block: torch.nn.Module, block_size_mb: float) -> bool:
        """Load a block to GPU. Returns True if successful."""
        pass

    @abstractmethod
    def offload_block(self, block: torch.nn.Module) -> bool:
        """Offload a block to CPU. Returns True if successful."""
        pass

    @abstractmethod
    def pre_forward_sync(self, block_idx: int, total_blocks: int):
        """Called before block forward pass"""
        pass

    @abstractmethod
    def post_forward_sync(self, block_idx: int, total_blocks: int):
        """Called after block forward pass"""
        pass


class AsyncMemoryStrategy(BaseSyncStrategy):
    """
    Pure Async Strategy: Uses async memory operations without explicit synchronization.

    This strategy demonstrates the issues with PyTorch's default async memory management
    for Video DiT workloads:
    - Deferred reclamation: cudaFreeAsync is non-blocking
    - Fragmentation accumulation: Different sized allocations interleave
    - Peak overlap: Async operations can overlap causing memory spikes
    """

    def __init__(self, device: int = 0, **kwargs):
        super().__init__(device, **kwargs)

    def load_block(self, block: torch.nn.Module, block_size_mb: float) -> bool:
        """Load block asynchronously without waiting"""
        try:
            self.load_count += 1
            # Use non_blocking transfer for async behavior
            with torch.cuda.stream(self.load_stream):
                block.to(device=f'cuda:{self.device}', non_blocking=True)
            return True
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                self.recent_failures += 1
                return False
            raise

    def offload_block(self, block: torch.nn.Module) -> bool:
        """Offload block asynchronously without waiting"""
        try:
            self.offload_count += 1
            # Use non_blocking transfer
            with torch.cuda.stream(self.offload_stream):
                block.to(device='cpu', non_blocking=True)
            return True
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                return False
            raise

    def pre_forward_sync(self, block_idx: int, total_blocks: int):
        """No synchronization in async mode"""
        pass

    def post_forward_sync(self, block_idx: int, total_blocks: int):
        """No synchronization in async mode"""
        pass


class SyncMemoryStrategy(BaseSyncStrategy):
    """
    Pure Sync Strategy: Synchronizes after every memory operation.

    This strategy guarantees memory safety by:
    - Ensuring offload operations complete before proceeding
    - Forcing garbage collection and cache trimming
    - Providing deterministic memory state at each point
    """

    def __init__(self, device: int = 0, aggressive_gc: bool = True, **kwargs):
        super().__init__(device, **kwargs)
        self.aggressive_gc = aggressive_gc

    def _full_sync(self):
        """Perform full synchronization with cache cleanup"""
        torch.cuda.synchronize(self.device)
        if self.aggressive_gc:
            gc.collect()
            torch.cuda.empty_cache()
        self.sync_count += 1

    def load_block(self, block: torch.nn.Module, block_size_mb: float) -> bool:
        """Load block with full synchronization"""
        try:
            self.load_count += 1

            # Synchronize before load to ensure previous offloads complete
            self._full_sync()

            # Blocking transfer
            block.to(device=f'cuda:{self.device}')

            # Synchronize after load
            torch.cuda.synchronize(self.device)

            return True
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                self.recent_failures += 1
                # Try cleanup and retry once
                self._full_sync()
                try:
                    block.to(device=f'cuda:{self.device}')
                    torch.cuda.synchronize(self.device)
                    return True
                except RuntimeError:
                    return False
            raise

    def offload_block(self, block: torch.nn.Module) -> bool:
        """Offload block with full synchronization"""
        try:
            self.offload_count += 1

            # Blocking transfer to CPU
            block.to(device='cpu')

            # Full synchronization and cleanup
            self._full_sync()

            return True
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                return False
            raise

    def pre_forward_sync(self, block_idx: int, total_blocks: int):
        """Synchronize before forward pass"""
        torch.cuda.synchronize(self.device)

    def post_forward_sync(self, block_idx: int, total_blocks: int):
        """Synchronize after forward pass"""
        torch.cuda.synchronize(self.device)


class ConditionalSyncMemoryStrategy(BaseSyncStrategy):
    """
    Conditional Sync Strategy: Only synchronizes when memory is tight.

    Key insight: OOM risk is proportional to 1 / (M_free - M_block)
    - When M_free >> M_block: Async is safe
    - When M_free ≈ M_block: Need synchronization

    This strategy balances performance and safety by:
    - Monitoring memory state and recent failures
    - Triggering sync only when necessary
    - Detecting phase transitions (e.g., VAE decode)
    """

    def __init__(self, device: int = 0,
                 memory_threshold_ratio: float = 0.15,
                 failure_window: int = 3,
                 phase_sync_interval: int = 10,
                 **kwargs):
        super().__init__(device, **kwargs)
        self.memory_threshold_ratio = memory_threshold_ratio
        self.failure_window = failure_window
        self.phase_sync_interval = phase_sync_interval
        self.block_counter = 0
        self.phase_counter = 0

    def _should_sync(self, block_size_mb: float) -> bool:
        """Determine if synchronization is needed"""
        free_memory = self.get_free_memory_mb()

        # Condition 1: Memory margin is insufficient
        if free_memory < block_size_mb + self.safety_margin_mb:
            return True

        # Condition 2: Recent allocation failures
        if self.recent_failures > 0:
            return True

        # Condition 3: Phase transition (periodic sync for stability)
        if self.block_counter % self.phase_sync_interval == 0:
            return True

        return False

    def _conditional_sync(self, block_size_mb: float = 0):
        """Perform sync only if conditions are met"""
        if self._should_sync(block_size_mb):
            torch.cuda.synchronize(self.device)
            gc.collect()
            torch.cuda.empty_cache()
            self.sync_count += 1
            # Reset failure counter after successful sync
            self.recent_failures = max(0, self.recent_failures - 1)

    def load_block(self, block: torch.nn.Module, block_size_mb: float) -> bool:
        """Load block with conditional synchronization"""
        try:
            self.load_count += 1
            self.block_counter += 1

            # Check if we need sync before loading
            self._conditional_sync(block_size_mb)

            # Attempt async load first
            with torch.cuda.stream(self.load_stream):
                block.to(device=f'cuda:{self.device}', non_blocking=True)

            # Wait for load stream to complete
            self.load_stream.synchronize()

            return True
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                self.recent_failures += 1
                # Force sync and retry
                torch.cuda.synchronize(self.device)
                gc.collect()
                torch.cuda.empty_cache()
                self.sync_count += 1
                try:
                    block.to(device=f'cuda:{self.device}')
                    return True
                except RuntimeError:
                    return False
            raise

    def offload_block(self, block: torch.nn.Module) -> bool:
        """Offload block with conditional synchronization"""
        try:
            self.offload_count += 1

            # Use async offload
            with torch.cuda.stream(self.offload_stream):
                block.to(device='cpu', non_blocking=True)

            # Conditional sync after offload
            # Estimate block size for sync decision
            block_size = sum(p.numel() * p.element_size() for p in block.parameters()) / (1024**2)
            self._conditional_sync(block_size)

            return True
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                return False
            raise

    def pre_forward_sync(self, block_idx: int, total_blocks: int):
        """Conditional sync before forward pass"""
        # Ensure load stream is complete before forward
        self.load_stream.synchronize()

    def post_forward_sync(self, block_idx: int, total_blocks: int):
        """Conditional sync after forward pass"""
        # Light sync - just ensure computation is done
        # Don't force cache clearing unless memory is tight
        free_memory = self.get_free_memory_mb()
        if free_memory < self.safety_margin_mb:
            torch.cuda.synchronize(self.device)
            self.sync_count += 1


class BlockOffloadManager:
    """
    Manager for block-level offloading with configurable sync strategies.

    Handles the logistics of moving transformer blocks between CPU and GPU
    while maintaining a working set of K blocks on GPU.
    """

    def __init__(self,
                 blocks: List[torch.nn.Module],
                 strategy: BaseSyncStrategy,
                 working_set_size: int = 5,
                 device: int = 0):
        self.blocks = blocks
        self.strategy = strategy
        self.working_set_size = working_set_size
        self.device = device

        # Track which blocks are on GPU
        self.gpu_blocks: set = set()

        # Calculate block sizes
        self.block_sizes_mb = []
        for block in blocks:
            size = sum(p.numel() * p.element_size() for p in block.parameters())
            self.block_sizes_mb.append(size / (1024 ** 2))

        self.avg_block_size_mb = sum(self.block_sizes_mb) / len(self.block_sizes_mb)

    def initialize_working_set(self, start_idx: int = 0):
        """Load initial working set of blocks to GPU"""
        # First, move all blocks to CPU
        for i, block in enumerate(self.blocks):
            block.to('cpu')

        self.gpu_blocks.clear()
        torch.cuda.empty_cache()
        gc.collect()

        # Load working set
        end_idx = min(start_idx + self.working_set_size, len(self.blocks))
        for i in range(start_idx, end_idx):
            success = self.strategy.load_block(self.blocks[i], self.block_sizes_mb[i])
            if success:
                self.gpu_blocks.add(i)
            else:
                raise RuntimeError(f"Failed to load block {i} during initialization")

    def prepare_block(self, block_idx: int) -> bool:
        """
        Ensure block is on GPU, managing working set as needed.
        Returns True if block is ready, False if OOM occurred.
        """
        # If block is already on GPU, just sync
        if block_idx in self.gpu_blocks:
            self.strategy.pre_forward_sync(block_idx, len(self.blocks))
            return True

        # Need to load this block - first offload oldest if at capacity
        if len(self.gpu_blocks) >= self.working_set_size:
            # Find block to offload (oldest/furthest from current)
            offload_idx = self._select_block_to_offload(block_idx)
            if offload_idx is not None:
                success = self.strategy.offload_block(self.blocks[offload_idx])
                if success:
                    self.gpu_blocks.discard(offload_idx)
                else:
                    return False

        # Load the needed block
        success = self.strategy.load_block(self.blocks[block_idx], self.block_sizes_mb[block_idx])
        if success:
            self.gpu_blocks.add(block_idx)
            self.strategy.pre_forward_sync(block_idx, len(self.blocks))
            return True
        return False

    def finish_block(self, block_idx: int):
        """Called after block forward pass completes"""
        self.strategy.post_forward_sync(block_idx, len(self.blocks))

    def _select_block_to_offload(self, current_idx: int) -> Optional[int]:
        """Select which block to offload based on future access pattern"""
        if not self.gpu_blocks:
            return None

        # For sequential access, offload the block furthest behind
        # (Video DiT accesses blocks in sequence: B₁→B₂→...→B_N)
        min_idx = min(self.gpu_blocks)

        # Don't offload blocks we'll need soon
        if min_idx >= current_idx:
            # All GPU blocks are ahead - offload the one we won't need for longest
            max_idx = max(self.gpu_blocks)
            if max_idx > current_idx + self.working_set_size:
                return max_idx
            return None

        return min_idx

    def get_stats(self) -> Dict[str, Any]:
        """Get current manager statistics"""
        return {
            'gpu_blocks': list(self.gpu_blocks),
            'working_set_size': self.working_set_size,
            'sync_count': self.strategy.sync_count,
            'load_count': self.strategy.load_count,
            'offload_count': self.strategy.offload_count,
            'avg_block_size_mb': self.avg_block_size_mb,
        }


def create_strategy(strategy_type: SyncStrategy, device: int = 0, **kwargs) -> BaseSyncStrategy:
    """Factory function to create sync strategy instances"""
    # Common kwargs for base class
    base_kwargs = {
        'safety_margin_mb': kwargs.get('safety_margin_mb', 1000.0),
    }

    if strategy_type == SyncStrategy.PURE_ASYNC:
        return AsyncMemoryStrategy(device=device, **base_kwargs)
    elif strategy_type == SyncStrategy.PURE_SYNC:
        sync_kwargs = {
            **base_kwargs,
            'aggressive_gc': kwargs.get('aggressive_gc', True),
        }
        return SyncMemoryStrategy(device=device, **sync_kwargs)
    elif strategy_type == SyncStrategy.CONDITIONAL_SYNC:
        conditional_kwargs = {
            **base_kwargs,
            'memory_threshold_ratio': kwargs.get('memory_threshold_ratio', 0.15),
            'failure_window': kwargs.get('failure_window', 3),
            'phase_sync_interval': kwargs.get('phase_sync_interval', 10),
        }
        return ConditionalSyncMemoryStrategy(device=device, **conditional_kwargs)
    else:
        raise ValueError(f"Unknown strategy type: {strategy_type}")
