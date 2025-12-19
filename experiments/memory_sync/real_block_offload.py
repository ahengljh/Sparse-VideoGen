"""
Real Video DiT Block Offloading Integration

This module integrates memory sync protocols with actual HunyuanVideo/Wan model inference.
It wraps the real transformer blocks to enable block-level CPU offloading during inference.

Usage:
    from real_block_offload import OffloadingTransformerWrapper, SyncStrategy

    # Wrap the model
    wrapper = OffloadingTransformerWrapper(
        pipe.transformer,
        working_set_size=5,
        strategy=SyncStrategy.CONDITIONAL_SYNC,
    )

    # Run inference with offloading
    wrapper.enable_offloading()
    output = pipe(prompt="...")
    stats = wrapper.get_stats()
"""

import torch
import torch.nn as nn
import time
import gc
from typing import Dict, List, Tuple, Optional, Callable, Any
from dataclasses import dataclass, field
from enum import Enum
from collections import OrderedDict
import functools


class SyncStrategy(Enum):
    """Memory synchronization strategies"""
    PURE_ASYNC = "pure_async"       # No sync - baseline (unsafe)
    PURE_SYNC = "pure_sync"         # Sync after every transfer (safe but slow)
    CONDITIONAL_SYNC = "conditional_sync"  # Sync only when needed (optimal)


@dataclass
class BlockStats:
    """Statistics for a single block during inference"""
    block_idx: int
    load_count: int = 0
    offload_count: int = 0
    total_load_time_ms: float = 0.0
    total_offload_time_ms: float = 0.0
    sync_triggered_count: int = 0
    oom_count: int = 0


@dataclass
class OffloadStats:
    """Overall offloading statistics"""
    total_blocks: int = 0
    working_set_size: int = 0
    strategy: str = ""
    total_load_count: int = 0
    total_offload_count: int = 0
    total_sync_count: int = 0
    total_oom_count: int = 0
    total_load_time_s: float = 0.0
    total_offload_time_s: float = 0.0
    peak_memory_gb: float = 0.0
    block_stats: Dict[int, BlockStats] = field(default_factory=dict)


class BlockOffloadManager:
    """
    Manages block-level offloading for Video DiT models.

    Handles:
    - Moving blocks between CPU and GPU
    - Maintaining a working set of blocks on GPU
    - Applying different sync strategies
    """

    def __init__(
        self,
        device: int = 0,
        working_set_size: int = 5,
        strategy: SyncStrategy = SyncStrategy.CONDITIONAL_SYNC,
        safety_margin_gb: float = 2.0,
    ):
        self.device = torch.device(f'cuda:{device}')
        self.device_id = device
        self.working_set_size = working_set_size
        self.strategy = strategy
        self.safety_margin_gb = safety_margin_gb

        # Block management
        self.blocks: Dict[int, nn.Module] = {}  # All blocks
        self.gpu_block_ids: List[int] = []  # Blocks currently on GPU
        self.cpu_block_ids: List[int] = []  # Blocks on CPU

        # Statistics
        self.stats = OffloadStats(strategy=strategy.value)
        self.block_stats: Dict[int, BlockStats] = {}

        # Conditional sync state
        self.recent_failures: List[float] = []
        self.failure_window = 5.0  # seconds

        # Streams for async transfers
        self.load_stream = torch.cuda.Stream(device=device)
        self.offload_stream = torch.cuda.Stream(device=device)

    def register_blocks(self, blocks: List[nn.Module]):
        """Register transformer blocks for offloading"""
        for idx, block in enumerate(blocks):
            self.blocks[idx] = block
            self.block_stats[idx] = BlockStats(block_idx=idx)

        self.stats.total_blocks = len(blocks)
        self.stats.working_set_size = self.working_set_size

        # Initially all blocks on GPU
        self.gpu_block_ids = list(range(len(blocks)))
        self.cpu_block_ids = []

    def initialize_offload(self):
        """Move blocks beyond working set to CPU"""
        blocks_to_offload = list(range(self.working_set_size, len(self.blocks)))

        for block_idx in blocks_to_offload:
            self._offload_block(block_idx, force_sync=True)

        torch.cuda.synchronize(self.device_id)
        torch.cuda.empty_cache()

        print(f"Initialized offloading: {len(self.gpu_block_ids)} blocks on GPU, "
              f"{len(self.cpu_block_ids)} blocks on CPU")

    def _get_free_memory_gb(self) -> float:
        """Get current free GPU memory"""
        props = torch.cuda.get_device_properties(self.device_id)
        reserved = torch.cuda.memory_reserved(self.device_id)
        return (props.total_memory - reserved) / (1024**3)

    def _should_sync(self, block_size_gb: float) -> Tuple[bool, str]:
        """Determine if sync is needed based on strategy and state"""
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

    def _estimate_block_size_gb(self, block: nn.Module) -> float:
        """Estimate block size in GB"""
        total_bytes = 0
        for param in block.parameters():
            total_bytes += param.numel() * param.element_size()
        for buffer in block.buffers():
            total_bytes += buffer.numel() * buffer.element_size()
        return total_bytes / (1024**3)

    def _load_block(self, block_idx: int) -> bool:
        """Load a block from CPU to GPU"""
        if block_idx in self.gpu_block_ids:
            return True  # Already on GPU

        if block_idx not in self.cpu_block_ids:
            return False  # Block not registered

        block = self.blocks[block_idx]
        block_size_gb = self._estimate_block_size_gb(block)

        should_sync, reason = self._should_sync(block_size_gb)

        start_time = time.time()

        try:
            if should_sync:
                # Synchronize before loading
                torch.cuda.synchronize(self.device_id)
                torch.cuda.empty_cache()
                self.block_stats[block_idx].sync_triggered_count += 1
                self.stats.total_sync_count += 1

            # Move block to GPU
            if should_sync or self.strategy == SyncStrategy.PURE_SYNC:
                block.to(self.device)
                torch.cuda.synchronize(self.device_id)
            else:
                with torch.cuda.stream(self.load_stream):
                    block.to(self.device)

            # Update tracking
            self.cpu_block_ids.remove(block_idx)
            self.gpu_block_ids.append(block_idx)

            # Update stats
            elapsed_ms = (time.time() - start_time) * 1000
            self.block_stats[block_idx].load_count += 1
            self.block_stats[block_idx].total_load_time_ms += elapsed_ms
            self.stats.total_load_count += 1
            self.stats.total_load_time_s += elapsed_ms / 1000

            return True

        except RuntimeError as e:
            if 'out of memory' in str(e).lower():
                self.block_stats[block_idx].oom_count += 1
                self.stats.total_oom_count += 1
                self.recent_failures.append(time.time())
                torch.cuda.empty_cache()
                return False
            raise

    def _offload_block(self, block_idx: int, force_sync: bool = False) -> bool:
        """Offload a block from GPU to CPU"""
        if block_idx in self.cpu_block_ids:
            return True  # Already on CPU

        if block_idx not in self.gpu_block_ids:
            return False  # Block not on GPU

        block = self.blocks[block_idx]
        block_size_gb = self._estimate_block_size_gb(block)

        should_sync, _ = self._should_sync(block_size_gb)
        should_sync = should_sync or force_sync

        start_time = time.time()

        # Move block to CPU
        if should_sync or self.strategy == SyncStrategy.PURE_SYNC:
            block.to('cpu')
            torch.cuda.synchronize(self.device_id)
            if should_sync:
                torch.cuda.empty_cache()
                self.stats.total_sync_count += 1
        else:
            with torch.cuda.stream(self.offload_stream):
                block.to('cpu')

        # Update tracking
        self.gpu_block_ids.remove(block_idx)
        self.cpu_block_ids.append(block_idx)

        # Update stats
        elapsed_ms = (time.time() - start_time) * 1000
        self.block_stats[block_idx].offload_count += 1
        self.block_stats[block_idx].total_offload_time_ms += elapsed_ms
        self.stats.total_offload_count += 1
        self.stats.total_offload_time_s += elapsed_ms / 1000

        return True

    def ensure_block_on_gpu(self, block_idx: int) -> bool:
        """Ensure a block is on GPU, offloading others if needed"""
        if block_idx in self.gpu_block_ids:
            return True

        # Need to load this block - first check if we need to offload
        while len(self.gpu_block_ids) >= self.working_set_size:
            # Offload the oldest block (FIFO)
            oldest_idx = self.gpu_block_ids[0]
            if not self._offload_block(oldest_idx):
                return False

        # Now load the requested block
        return self._load_block(block_idx)

    def get_stats(self) -> OffloadStats:
        """Get current statistics"""
        self.stats.block_stats = self.block_stats
        self.stats.peak_memory_gb = torch.cuda.max_memory_allocated(self.device_id) / (1024**3)
        return self.stats

    def reset_stats(self):
        """Reset statistics"""
        self.stats = OffloadStats(
            strategy=self.strategy.value,
            total_blocks=len(self.blocks),
            working_set_size=self.working_set_size
        )
        for idx in self.block_stats:
            self.block_stats[idx] = BlockStats(block_idx=idx)
        torch.cuda.reset_peak_memory_stats(self.device_id)


class OffloadingTransformerWrapper:
    """
    Wraps a Video DiT transformer to enable block-level offloading.

    Works with:
    - HunyuanVideoTransformer3DModel
    - WanTransformer3DModel
    - CogVideoXTransformer3DModel
    """

    def __init__(
        self,
        transformer: nn.Module,
        working_set_size: int = 5,
        strategy: SyncStrategy = SyncStrategy.CONDITIONAL_SYNC,
        device: int = 0,
    ):
        self.transformer = transformer
        self.device = device
        self.strategy = strategy
        self.working_set_size = working_set_size

        # Create offload manager
        self.manager = BlockOffloadManager(
            device=device,
            working_set_size=working_set_size,
            strategy=strategy,
        )

        # Find transformer blocks
        self.blocks = self._find_transformer_blocks()
        self.manager.register_blocks(self.blocks)

        # Original forward functions
        self._original_forwards: Dict[int, Callable] = {}
        self._offloading_enabled = False

    def _find_transformer_blocks(self) -> List[nn.Module]:
        """Find all transformer blocks in the model"""
        blocks = []

        # HunyuanVideo pattern
        if hasattr(self.transformer, 'transformer_blocks'):
            blocks.extend(list(self.transformer.transformer_blocks))
        if hasattr(self.transformer, 'single_transformer_blocks'):
            blocks.extend(list(self.transformer.single_transformer_blocks))

        # Wan pattern
        if hasattr(self.transformer, 'blocks'):
            blocks.extend(list(self.transformer.blocks))

        # CogVideoX pattern
        if hasattr(self.transformer, 'transformer_blocks') and not blocks:
            blocks.extend(list(self.transformer.transformer_blocks))

        print(f"Found {len(blocks)} transformer blocks")
        return blocks

    def _create_wrapped_forward(self, block_idx: int, original_forward: Callable) -> Callable:
        """Create a wrapped forward that handles offloading"""
        @functools.wraps(original_forward)
        def wrapped_forward(*args, **kwargs):
            # Ensure block is on GPU before forward
            if self._offloading_enabled:
                success = self.manager.ensure_block_on_gpu(block_idx)
                if not success:
                    raise RuntimeError(f"Failed to load block {block_idx} to GPU")

                # For non-async strategies, transfers are already synchronized
                # For pure async, we intentionally don't sync to demonstrate the safety issues:
                # 1. OOM due to deferred reclamation (measured by this experiment)
                # 2. Potential data corruption from using incomplete transfers
                # The sync strategies are designed to prevent these issues.

            # Run forward
            return original_forward(*args, **kwargs)

        return wrapped_forward

    def enable_offloading(self):
        """Enable block offloading during inference"""
        if self._offloading_enabled:
            return

        # Wrap each block's forward
        for idx, block in enumerate(self.blocks):
            self._original_forwards[idx] = block.forward
            block.forward = self._create_wrapped_forward(idx, block.forward)

        # Initialize - move blocks beyond working set to CPU
        self.manager.initialize_offload()

        self._offloading_enabled = True
        print(f"Offloading enabled with strategy: {self.strategy.value}")

    def disable_offloading(self):
        """Disable offloading and restore original forwards"""
        if not self._offloading_enabled:
            return

        # Restore original forwards
        for idx, block in enumerate(self.blocks):
            if idx in self._original_forwards:
                block.forward = self._original_forwards[idx]

        # Move all blocks back to GPU
        for idx in list(self.manager.cpu_block_ids):
            self.manager._load_block(idx)

        self._original_forwards.clear()
        self._offloading_enabled = False
        print("Offloading disabled")

    def get_stats(self) -> OffloadStats:
        """Get offloading statistics"""
        return self.manager.get_stats()

    def reset_stats(self):
        """Reset statistics"""
        self.manager.reset_stats()

    def print_stats(self):
        """Print formatted statistics"""
        stats = self.get_stats()

        print(f"\n{'='*60}")
        print("OFFLOADING STATISTICS")
        print(f"{'='*60}")
        print(f"Strategy: {stats.strategy}")
        print(f"Total blocks: {stats.total_blocks}")
        print(f"Working set size: {stats.working_set_size}")
        print(f"")
        print(f"Total loads: {stats.total_load_count}")
        print(f"Total offloads: {stats.total_offload_count}")
        print(f"Total syncs: {stats.total_sync_count}")
        print(f"Sync rate: {stats.total_sync_count / max(1, stats.total_load_count + stats.total_offload_count):.1%}")
        print(f"Total OOMs: {stats.total_oom_count}")
        print(f"OOM rate: {stats.total_oom_count / max(1, stats.total_load_count):.1%}")
        print(f"")
        print(f"Load time: {stats.total_load_time_s:.2f}s")
        print(f"Offload time: {stats.total_offload_time_s:.2f}s")
        print(f"Peak memory: {stats.peak_memory_gb:.2f} GB")
        print(f"{'='*60}")


def run_real_inference_test(
    model_name: str = "hunyuan",
    prompt: str = "A cat walking in the garden",
    strategy: SyncStrategy = SyncStrategy.CONDITIONAL_SYNC,
    working_set_size: int = 5,
    num_inference_steps: int = 20,
):
    """
    Run real inference with block offloading.

    This is a template - actual usage depends on model loading code.
    """
    print(f"\n{'='*60}")
    print(f"REAL INFERENCE TEST: {model_name}")
    print(f"{'='*60}")
    print(f"Strategy: {strategy.value}")
    print(f"Working set: {working_set_size} blocks")
    print(f"Steps: {num_inference_steps}")

    # This would be replaced with actual model loading
    print("\nNote: This is a template. To run actual inference:")
    print("1. Load the model using standard HunyuanVideo/Wan pipeline")
    print("2. Create OffloadingTransformerWrapper around pipe.transformer")
    print("3. Call wrapper.enable_offloading()")
    print("4. Run pipe(...) as normal")
    print("5. Call wrapper.print_stats() to see results")

    example_code = '''
# Example usage:
from diffusers import HunyuanVideoPipeline
from real_block_offload import OffloadingTransformerWrapper, SyncStrategy

# Load model
pipe = HunyuanVideoPipeline.from_pretrained("tencent/HunyuanVideo", ...)
pipe.to("cuda")

# Wrap with offloading
wrapper = OffloadingTransformerWrapper(
    pipe.transformer,
    working_set_size=5,
    strategy=SyncStrategy.CONDITIONAL_SYNC,
)
wrapper.enable_offloading()

# Run inference (offloading happens automatically)
output = pipe(
    prompt="A cat walking in the garden",
    num_inference_steps=30,
)

# Print stats
wrapper.print_stats()
'''
    print(f"\n{example_code}")


if __name__ == "__main__":
    run_real_inference_test()
