"""
Memory Monitoring Utilities for Video DiT Experiments

This module provides tools for monitoring GPU memory usage patterns,
detecting fragmentation, and tracking allocation/deallocation cycles.
"""

import torch
import time
import threading
import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple, Callable
from collections import deque
import json
import os


@dataclass
class MemorySnapshot:
    """Single memory state snapshot"""
    timestamp: float
    allocated: int  # bytes
    reserved: int   # bytes
    free: int       # bytes
    max_allocated: int
    num_allocs: int
    num_frees: int
    fragmentation_ratio: float  # estimated
    largest_free_block: int  # estimated
    event: str = ""


@dataclass
class AllocationEvent:
    """Record of a single allocation/deallocation"""
    timestamp: float
    operation: str  # 'alloc' or 'free'
    size: int
    address: Optional[int] = None
    stream: Optional[str] = None


@dataclass
class MemoryExperimentResult:
    """Results from a memory experiment"""
    experiment_name: str
    total_duration: float
    snapshots: List[MemorySnapshot]
    allocation_events: List[AllocationEvent]
    oom_occurred: bool = False
    oom_count: int = 0
    peak_memory: int = 0
    peak_memory_variance: float = 0.0
    avg_fragmentation: float = 0.0
    sync_trigger_count: int = 0
    sync_trigger_rate: float = 0.0

    def to_dict(self) -> Dict:
        return {
            'experiment_name': self.experiment_name,
            'total_duration': self.total_duration,
            'oom_occurred': self.oom_occurred,
            'oom_count': self.oom_count,
            'peak_memory_gb': self.peak_memory / (1024**3),
            'peak_memory_variance_gb': self.peak_memory_variance / (1024**3),
            'avg_fragmentation': self.avg_fragmentation,
            'sync_trigger_count': self.sync_trigger_count,
            'sync_trigger_rate': self.sync_trigger_rate,
            'num_snapshots': len(self.snapshots),
        }


class MemoryMonitor:
    """
    Real-time GPU memory monitor with snapshot capture and analysis.

    Usage:
        monitor = MemoryMonitor(device=0, interval_ms=10)
        monitor.start()
        # ... run experiment ...
        monitor.stop()
        results = monitor.get_results()
    """

    def __init__(
        self,
        device: int = 0,
        interval_ms: float = 10,
        max_snapshots: int = 100000
    ):
        self.device = device
        self.interval_ms = interval_ms
        self.max_snapshots = max_snapshots

        self.snapshots: List[MemorySnapshot] = []
        self.allocation_events: List[AllocationEvent] = []

        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._start_time: float = 0
        self._lock = threading.Lock()

        # OOM tracking
        self.oom_count = 0

        # Peak tracking
        self._peak_values: List[int] = []

    def start(self):
        """Start background monitoring"""
        self._running = True
        self._start_time = time.time()
        self._thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self._thread.start()

    def stop(self) -> 'MemoryMonitor':
        """Stop monitoring and return self for chaining"""
        self._running = False
        if self._thread:
            self._thread.join(timeout=1.0)
        return self

    def _monitor_loop(self):
        """Background monitoring loop"""
        while self._running and len(self.snapshots) < self.max_snapshots:
            snapshot = self._capture_snapshot()
            with self._lock:
                self.snapshots.append(snapshot)
                self._peak_values.append(snapshot.allocated)
            time.sleep(self.interval_ms / 1000.0)

    def _capture_snapshot(self, event: str = "") -> MemorySnapshot:
        """Capture current memory state"""
        torch.cuda.synchronize(self.device)

        stats = torch.cuda.memory_stats(self.device)

        allocated = torch.cuda.memory_allocated(self.device)
        reserved = torch.cuda.memory_reserved(self.device)

        # Get device properties
        total_memory = torch.cuda.get_device_properties(self.device).total_memory
        free = total_memory - reserved

        # Estimate fragmentation
        # fragmentation = 1 - (largest_free_block / total_free)
        # We estimate this from allocated vs reserved ratio
        if reserved > 0:
            fragmentation = 1.0 - (allocated / reserved)
        else:
            fragmentation = 0.0

        # Estimate largest free block (heuristic)
        largest_free = free  # conservative estimate

        return MemorySnapshot(
            timestamp=time.time() - self._start_time,
            allocated=allocated,
            reserved=reserved,
            free=free,
            max_allocated=stats.get('allocated_bytes.all.peak', allocated),
            num_allocs=stats.get('num_alloc_retries', 0),
            num_frees=stats.get('num_ooms', 0),
            fragmentation_ratio=fragmentation,
            largest_free_block=largest_free,
            event=event
        )

    def record_event(self, event: str):
        """Record a named event with current memory state"""
        snapshot = self._capture_snapshot(event)
        with self._lock:
            self.snapshots.append(snapshot)

    def record_allocation(self, operation: str, size: int, stream: str = "default"):
        """Record an allocation/deallocation event"""
        event = AllocationEvent(
            timestamp=time.time() - self._start_time,
            operation=operation,
            size=size,
            stream=stream
        )
        with self._lock:
            self.allocation_events.append(event)

    def record_oom(self):
        """Record an OOM event"""
        self.oom_count += 1
        self.record_event(f"OOM_{self.oom_count}")

    def get_results(self, experiment_name: str = "unnamed") -> MemoryExperimentResult:
        """Compile results from monitoring session"""
        with self._lock:
            snapshots = list(self.snapshots)
            events = list(self.allocation_events)
            peak_values = list(self._peak_values)

        if not snapshots:
            return MemoryExperimentResult(
                experiment_name=experiment_name,
                total_duration=0,
                snapshots=[],
                allocation_events=[],
            )

        peak_memory = max(s.allocated for s in snapshots)

        # Calculate peak variance (important for async vs sync comparison)
        if peak_values:
            peak_variance = float(np.var(peak_values))
        else:
            peak_variance = 0.0

        # Calculate average fragmentation
        avg_frag = np.mean([s.fragmentation_ratio for s in snapshots])

        return MemoryExperimentResult(
            experiment_name=experiment_name,
            total_duration=snapshots[-1].timestamp if snapshots else 0,
            snapshots=snapshots,
            allocation_events=events,
            oom_occurred=self.oom_count > 0,
            oom_count=self.oom_count,
            peak_memory=peak_memory,
            peak_memory_variance=peak_variance,
            avg_fragmentation=avg_frag,
        )


class MemoryFragmentationAnalyzer:
    """
    Analyze memory fragmentation patterns specific to Video DiT block offloading.

    Key metrics:
    - Contiguous free block sizes
    - Fragmentation ratio over time
    - Allocation failure prediction
    """

    def __init__(self, device: int = 0):
        self.device = device
        self.history: List[Dict] = []

    def analyze_current_state(self) -> Dict:
        """Capture and analyze current fragmentation state"""
        torch.cuda.synchronize(self.device)

        stats = torch.cuda.memory_stats(self.device)
        allocated = torch.cuda.memory_allocated(self.device)
        reserved = torch.cuda.memory_reserved(self.device)

        # Memory pool stats
        pool_stats = {
            'allocated': allocated,
            'reserved': reserved,
            'free_in_pool': reserved - allocated,
            'active_blocks': stats.get('active.all.current', 0),
            'inactive_blocks': stats.get('inactive_split.all.current', 0),
            'allocation_retries': stats.get('num_alloc_retries', 0),
        }

        # Estimate fragmentation
        if reserved > allocated:
            # Fragmentation ratio: how much reserved memory is unusable
            frag_ratio = (reserved - allocated) / reserved
        else:
            frag_ratio = 0.0

        analysis = {
            'timestamp': time.time(),
            'pool_stats': pool_stats,
            'fragmentation_ratio': frag_ratio,
            'can_allocate_650mb': self._can_allocate(650 * 1024 * 1024),
        }

        self.history.append(analysis)
        return analysis

    def _can_allocate(self, size: int) -> bool:
        """Check if allocation of given size would succeed"""
        try:
            # Try to allocate
            tensor = torch.empty(size // 4, dtype=torch.float32, device=self.device)
            del tensor
            torch.cuda.empty_cache()
            return True
        except RuntimeError:
            return False

    def simulate_block_swap_fragmentation(
        self,
        block_size_mb: int = 650,
        activation_size_mb: int = 2000,
        num_swaps: int = 100,
    ) -> Dict:
        """
        Simulate fragmentation accumulation from repeated block swaps.

        Returns metrics showing how fragmentation builds up.
        """
        block_size = block_size_mb * 1024 * 1024
        activation_size = activation_size_mb * 1024 * 1024

        results = {
            'fragmentation_over_time': [],
            'allocation_failures': 0,
            'total_swaps': num_swaps,
        }

        # Simulate varying activation sizes (as they change during forward pass)
        activation_tensors = []
        block_tensors = []

        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(self.device)

        for i in range(num_swaps):
            try:
                # Simulate block load
                block = torch.empty(
                    block_size // 4,
                    dtype=torch.float32,
                    device=self.device
                )
                block_tensors.append(block)

                # Simulate activation allocation (varying size)
                act_size = activation_size + np.random.randint(-500, 500) * 1024 * 1024
                act_size = max(100 * 1024 * 1024, act_size)
                activation = torch.empty(
                    act_size // 4,
                    dtype=torch.float32,
                    device=self.device
                )
                activation_tensors.append(activation)

                # Simulate block offload (delete oldest blocks)
                if len(block_tensors) > 3:
                    del block_tensors[0]
                if len(activation_tensors) > 2:
                    del activation_tensors[0]

                # Record fragmentation
                analysis = self.analyze_current_state()
                results['fragmentation_over_time'].append(analysis['fragmentation_ratio'])

            except RuntimeError as e:
                if 'out of memory' in str(e).lower():
                    results['allocation_failures'] += 1
                    # Clear and continue
                    torch.cuda.empty_cache()
                else:
                    raise

        # Cleanup
        del block_tensors, activation_tensors
        torch.cuda.empty_cache()

        return results


def estimate_largest_free_block(device: int = 0) -> int:
    """
    Binary search to find the largest allocatable contiguous block.

    This is crucial for determining if a 650MB block can be loaded.
    """
    torch.cuda.synchronize(device)

    total = torch.cuda.get_device_properties(device).total_memory
    reserved = torch.cuda.memory_reserved(device)

    # Binary search for largest allocatable size
    low = 0
    high = total - reserved
    result = 0

    while low <= high:
        mid = (low + high) // 2
        try:
            tensor = torch.empty(mid // 4, dtype=torch.float32, device=device)
            result = mid
            low = mid + 1
            del tensor
        except RuntimeError:
            high = mid - 1

    torch.cuda.empty_cache()
    return result


def get_memory_summary(device: int = 0) -> Dict:
    """Get comprehensive memory summary"""
    torch.cuda.synchronize(device)

    props = torch.cuda.get_device_properties(device)
    stats = torch.cuda.memory_stats(device)

    return {
        'device_name': props.name,
        'total_memory_gb': props.total_memory / (1024**3),
        'allocated_gb': torch.cuda.memory_allocated(device) / (1024**3),
        'reserved_gb': torch.cuda.memory_reserved(device) / (1024**3),
        'free_gb': (props.total_memory - torch.cuda.memory_reserved(device)) / (1024**3),
        'peak_allocated_gb': stats.get('allocated_bytes.all.peak', 0) / (1024**3),
        'num_alloc_retries': stats.get('num_alloc_retries', 0),
        'num_ooms': stats.get('num_ooms', 0),
        'largest_free_block_gb': estimate_largest_free_block(device) / (1024**3),
    }


def save_experiment_results(
    results: MemoryExperimentResult,
    output_dir: str = "./results"
):
    """Save experiment results to JSON"""
    os.makedirs(output_dir, exist_ok=True)

    # Save summary
    summary_path = os.path.join(output_dir, f"{results.experiment_name}_summary.json")
    with open(summary_path, 'w') as f:
        json.dump(results.to_dict(), f, indent=2)

    # Save detailed snapshots
    snapshots_path = os.path.join(output_dir, f"{results.experiment_name}_snapshots.json")
    snapshot_data = [
        {
            'timestamp': s.timestamp,
            'allocated_gb': s.allocated / (1024**3),
            'reserved_gb': s.reserved / (1024**3),
            'free_gb': s.free / (1024**3),
            'fragmentation': s.fragmentation_ratio,
            'event': s.event,
        }
        for s in results.snapshots
    ]
    with open(snapshots_path, 'w') as f:
        json.dump(snapshot_data, f, indent=2)

    print(f"Results saved to {output_dir}/")
    return summary_path, snapshots_path


if __name__ == "__main__":
    # Quick test
    print("Memory Monitor Test")
    print("=" * 50)

    summary = get_memory_summary()
    for k, v in summary.items():
        print(f"  {k}: {v}")
