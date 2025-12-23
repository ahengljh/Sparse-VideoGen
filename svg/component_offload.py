"""
Hybrid Component-Level Offloading for Video Diffusion Models

This module implements a smart hybrid strategy that combines:
1. Layer-level sliding window (like AIO)
2. Component-level optimization within each layer (Attention first, FFN prefetch)

Key insight: We can't pin ALL attention for all 60 layers (would use ~9GB).
Instead, we use a sliding window approach where:
- Attention components are loaded FIRST when a layer enters the window
- FFN components are PREFETCHED during attention computation
- This achieves compute-transfer overlap without excessive memory usage

Memory Strategy:
┌─────────────────────────────────────────────────────────────────────────────┐
│ GPU Memory Layout (24GB example)                                            │
├─────────────────────────────────────────────────────────────────────────────┤
│ Fixed Components (~2GB):                                                    │
│   - VAE, Embedders, RoPE, etc.                                              │
│                                                                             │
│ Sliding Window (~8-12GB):                                                   │
│   - W layers worth of FULL blocks (attention + norm + FFN)                  │
│   - W = 6-8 for 24GB GPU                                                    │
│                                                                             │
│ Activations + Cache (~6-8GB):                                               │
│   - Forward pass activations                                                │
│   - CTCA cache (if enabled)                                                 │
│                                                                             │
│ Within each layer execution:                                                │
│   1. Ensure attention+norm on GPU (load if needed)                          │
│   2. Start prefetching FFN for THIS layer                                   │
│   3. Execute attention (compute-bound, hides FFN transfer)                  │
│   4. Execute FFN (weights now ready)                                        │
│   5. Start prefetching NEXT layer's attention+norm                          │
└─────────────────────────────────────────────────────────────────────────────┘
"""

from __future__ import annotations

import gc
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

import torch
import torch.nn as nn

from .logger import logger
from .timer import time_logging_decorator


@dataclass
class HybridOffloadConfig:
    """Configuration for hybrid component-level offloading."""

    # Device settings
    compute_device: str = "cuda"
    offload_device: str = "cpu"

    # Memory settings
    use_pinned_memory: bool = True

    # Sliding window size (number of FULL layers on GPU)
    # Each layer = attention + norm + FFN
    # For 24GB GPU: 6-8 layers typical
    num_layers_on_gpu: int = 6

    # Prefetching settings
    enable_prefetch: bool = True
    prefetch_count: int = 2  # Prefetch next N layers

    # Component loading order optimization
    # When True: Load attention first, prefetch FFN during attention compute
    # When False: Load entire layer at once (like standard AIO)
    enable_component_prefetch: bool = True

    # Memory management
    empty_cache_frequency: int = 10

    # Debugging
    verbose: bool = False


# Component classification for diffusers HunyuanVideoTransformerBlock
DOUBLE_BLOCK_ATTENTION_COMPONENTS = ['attn']
DOUBLE_BLOCK_FFN_COMPONENTS = ['ff', 'ff_context']
DOUBLE_BLOCK_NORM_COMPONENTS = ['norm1', 'norm1_context', 'norm2', 'norm2_context']

# Single stream blocks
SINGLE_BLOCK_ATTENTION_COMPONENTS = ['attn']
SINGLE_BLOCK_MLP_COMPONENTS = ['proj_out']
SINGLE_BLOCK_NORM_COMPONENTS = ['norm', 'proj_mlp', 'act_mlp']


@dataclass
class LayerMemoryInfo:
    """Memory info for a layer."""
    layer_idx: int
    layer_type: str  # "double" or "single"
    attention_mb: float = 0.0
    ffn_mb: float = 0.0
    norm_mb: float = 0.0
    total_mb: float = 0.0


class HybridOffloadManager:
    """
    Hybrid offload manager with smart component-level optimization.

    This manager:
    1. Uses sliding window for layers (like AIO)
    2. Within each layer, optimizes component loading order
    3. Prefetches FFN during attention computation for overlap
    """

    def __init__(
        self,
        transformer: nn.Module,
        config: HybridOffloadConfig,
        double_blocks_attr: str = "transformer_blocks",
        single_blocks_attr: str = "single_transformer_blocks",
    ):
        self.transformer = transformer
        self.config = config
        self.double_blocks_attr = double_blocks_attr
        self.single_blocks_attr = single_blocks_attr

        # Get layer lists
        self.double_blocks = list(getattr(transformer, double_blocks_attr, []))
        self.single_blocks = list(getattr(transformer, single_blocks_attr, []))
        self.num_double = len(self.double_blocks)
        self.num_single = len(self.single_blocks)
        self.num_layers = self.num_double + self.num_single

        # Analyze memory per layer
        self._layer_memory: Dict[int, LayerMemoryInfo] = {}
        self._analyze_layer_memory()

        # Track what's on GPU
        self._layer_on_gpu: Dict[int, bool] = {i: False for i in range(self.num_layers)}
        self._layers_on_gpu_set: Set[int] = set()

        # For component-level prefetch within a layer
        self._ffn_prefetch_in_progress: Dict[int, bool] = {}
        self._ffn_prefetch_events: Dict[int, torch.cuda.Event] = {}

        # CUDA streams
        self._prefetch_stream: Optional[torch.cuda.Stream] = None
        self._ffn_stream: Optional[torch.cuda.Stream] = None

        # Layer prefetch state
        self._layer_prefetch_in_progress: Dict[int, bool] = {}
        self._layer_prefetch_events: Dict[int, torch.cuda.Event] = {}

        # Statistics
        self.stats = {
            'layer_loads': 0,
            'layer_offloads': 0,
            'prefetch_hits': 0,
            'prefetch_misses': 0,
            'ffn_prefetch_overlaps': 0,
            'cache_clears': 0,
        }

        self._initialized = False
        self._log_counter = 0

    def _analyze_layer_memory(self):
        """Analyze memory usage per layer."""
        logger.info("Analyzing layer memory for hybrid offloading...")

        def get_module_mb(module):
            if module is None:
                return 0.0
            total = sum(p.numel() * p.element_size() for p in module.parameters())
            return total / (1024 * 1024)

        total_attention = 0.0
        total_ffn = 0.0
        total_norm = 0.0

        # Double stream blocks
        for idx, block in enumerate(self.double_blocks):
            info = LayerMemoryInfo(layer_idx=idx, layer_type="double")

            for name in DOUBLE_BLOCK_ATTENTION_COMPONENTS:
                if hasattr(block, name):
                    info.attention_mb += get_module_mb(getattr(block, name))

            for name in DOUBLE_BLOCK_FFN_COMPONENTS:
                if hasattr(block, name):
                    info.ffn_mb += get_module_mb(getattr(block, name))

            for name in DOUBLE_BLOCK_NORM_COMPONENTS:
                if hasattr(block, name):
                    info.norm_mb += get_module_mb(getattr(block, name))

            info.total_mb = info.attention_mb + info.ffn_mb + info.norm_mb
            self._layer_memory[idx] = info

            total_attention += info.attention_mb
            total_ffn += info.ffn_mb
            total_norm += info.norm_mb

        # Single stream blocks
        for idx, block in enumerate(self.single_blocks):
            layer_idx = self.num_double + idx
            info = LayerMemoryInfo(layer_idx=layer_idx, layer_type="single")

            for name in SINGLE_BLOCK_ATTENTION_COMPONENTS:
                if hasattr(block, name):
                    info.attention_mb += get_module_mb(getattr(block, name))

            for name in SINGLE_BLOCK_MLP_COMPONENTS:
                if hasattr(block, name):
                    info.ffn_mb += get_module_mb(getattr(block, name))

            for name in SINGLE_BLOCK_NORM_COMPONENTS:
                if hasattr(block, name):
                    info.norm_mb += get_module_mb(getattr(block, name))

            info.total_mb = info.attention_mb + info.ffn_mb + info.norm_mb
            self._layer_memory[layer_idx] = info

            total_attention += info.attention_mb
            total_ffn += info.ffn_mb
            total_norm += info.norm_mb

        # Log summary
        total = total_attention + total_ffn + total_norm
        logger.info(f"Layer memory analysis complete:")
        logger.info(f"  Total model: {total:.1f}MB ({total/1024:.2f}GB)")
        logger.info(f"  Attention: {total_attention:.1f}MB ({100*total_attention/total:.1f}%)")
        logger.info(f"  FFN: {total_ffn:.1f}MB ({100*total_ffn/total:.1f}%)")
        logger.info(f"  Norm: {total_norm:.1f}MB ({100*total_norm/total:.1f}%)")

        if self.num_layers > 0:
            avg_layer = total / self.num_layers
            window_size = self.config.num_layers_on_gpu
            window_memory = window_size * avg_layer
            logger.info(f"  Per layer average: {avg_layer:.1f}MB")
            logger.info(f"  Window ({window_size} layers): {window_memory:.1f}MB ({window_memory/1024:.2f}GB)")

    def _get_block(self, layer_idx: int) -> nn.Module:
        """Get the block for a given layer index."""
        if layer_idx < self.num_double:
            return self.double_blocks[layer_idx]
        else:
            return self.single_blocks[layer_idx - self.num_double]

    def _is_double_block(self, layer_idx: int) -> bool:
        """Check if layer is a double stream block."""
        return layer_idx < self.num_double

    def _ensure_pinned_memory(self, layer_idx: int):
        """Convert a CPU layer to use pinned memory for faster GPU transfers."""
        block = self._get_block(layer_idx)

        for param in block.parameters():
            if param.device.type == 'cpu' and not param.data.is_pinned():
                try:
                    pinned_tensor = torch.empty_like(param.data, pin_memory=True)
                    pinned_tensor.copy_(param.data)
                    param.data = pinned_tensor
                except Exception:
                    # Pinning can fail if not enough pinnable memory
                    pass

    def prepare_for_inference(self):
        """
        Prepare the model for offloaded inference.

        This sets up CUDA streams, tracking for layers, and pinned memory.
        Layers should already be on CPU (moved by enable_component_offloading).
        """
        if self._initialized:
            return

        logger.info("=" * 70)
        logger.info("[HYBRID-OFFLOAD] Initializing Hybrid Component Offloading")
        logger.info("=" * 70)
        logger.info(f"[HYBRID-OFFLOAD] Strategy: Sliding window with FFN prefetch overlap")
        logger.info(f"[HYBRID-OFFLOAD] Total layers: {self.num_layers}")
        logger.info(f"[HYBRID-OFFLOAD] Window size: {self.config.num_layers_on_gpu} layers")
        logger.info(f"[HYBRID-OFFLOAD] Component prefetch: {self.config.enable_component_prefetch}")
        logger.info("=" * 70)

        # Create CUDA streams
        if self.config.enable_prefetch:
            self._prefetch_stream = torch.cuda.Stream()
        if self.config.enable_component_prefetch:
            self._ffn_stream = torch.cuda.Stream()

        # Check layer locations and set up tracking
        # Layers should already be on CPU from enable_component_offloading
        for idx in range(self.num_layers):
            block = self._get_block(idx)
            first_param = next(block.parameters(), None)
            if first_param is not None:
                is_on_cpu = first_param.device.type == 'cpu'
                self._layer_on_gpu[idx] = not is_on_cpu

                # If on CPU and we want pinned memory, convert to pinned
                if is_on_cpu and self.config.use_pinned_memory:
                    self._ensure_pinned_memory(idx)

        torch.cuda.empty_cache()
        gc.collect()

        self._initialized = True

        num_on_cpu = sum(1 for v in self._layer_on_gpu.values() if not v)
        logger.info(f"[HYBRID-OFFLOAD] {num_on_cpu}/{self.num_layers} layers on CPU with pinned memory")

    def _move_layer_to_cpu(self, layer_idx: int, use_pinned: bool = True):
        """Move entire layer to CPU."""
        block = self._get_block(layer_idx)

        # Always use standard .to() for moving - this ensures proper module tracking
        block.to('cpu')

        # If pinned memory is requested, pin the parameters after moving to CPU
        # Note: Pinned memory helps with faster CPU->GPU transfers
        if use_pinned:
            for param in block.parameters():
                if not param.data.is_pinned():
                    try:
                        pinned = torch.empty_like(param.data, pin_memory=True)
                        pinned.copy_(param.data)
                        param.data = pinned
                    except Exception:
                        # If pinning fails (e.g., not enough pinnable memory), continue without pinning
                        pass

        self._layer_on_gpu[layer_idx] = False
        self._layers_on_gpu_set.discard(layer_idx)

    def _move_layer_to_gpu(self, layer_idx: int, non_blocking: bool = False):
        """Move entire layer to GPU."""
        block = self._get_block(layer_idx)
        block.to(self.config.compute_device, non_blocking=non_blocking)
        self._layer_on_gpu[layer_idx] = True
        self._layers_on_gpu_set.add(layer_idx)
        self.stats['layer_loads'] += 1

    def _move_attention_to_gpu(self, layer_idx: int, non_blocking: bool = False):
        """Move only attention and norm components to GPU (for component-level prefetch)."""
        block = self._get_block(layer_idx)
        is_double = self._is_double_block(layer_idx)

        attn_comps = DOUBLE_BLOCK_ATTENTION_COMPONENTS if is_double else SINGLE_BLOCK_ATTENTION_COMPONENTS
        norm_comps = DOUBLE_BLOCK_NORM_COMPONENTS if is_double else SINGLE_BLOCK_NORM_COMPONENTS

        for name in attn_comps + norm_comps:
            if hasattr(block, name):
                comp = getattr(block, name)
                if comp is not None:
                    comp.to(self.config.compute_device, non_blocking=non_blocking)

    def _move_ffn_to_gpu(self, layer_idx: int, non_blocking: bool = False):
        """Move only FFN components to GPU."""
        block = self._get_block(layer_idx)
        is_double = self._is_double_block(layer_idx)

        ffn_comps = DOUBLE_BLOCK_FFN_COMPONENTS if is_double else SINGLE_BLOCK_MLP_COMPONENTS

        for name in ffn_comps:
            if hasattr(block, name):
                comp = getattr(block, name)
                if comp is not None:
                    comp.to(self.config.compute_device, non_blocking=non_blocking)

    def _start_layer_prefetch(self, layer_idx: int):
        """Start async prefetch of entire layer."""
        if not self.config.enable_prefetch:
            return
        if layer_idx >= self.num_layers:
            return
        if self._layer_on_gpu.get(layer_idx, False):
            return
        if self._layer_prefetch_in_progress.get(layer_idx, False):
            return

        self._layer_prefetch_in_progress[layer_idx] = True
        event = torch.cuda.Event()
        self._layer_prefetch_events[layer_idx] = event

        with torch.cuda.stream(self._prefetch_stream):
            self._move_layer_to_gpu(layer_idx, non_blocking=True)
            event.record()

        if self.config.verbose:
            logger.debug(f"Started prefetch for layer {layer_idx}")

    def _wait_for_layer_prefetch(self, layer_idx: int):
        """Wait for layer prefetch to complete."""
        if layer_idx in self._layer_prefetch_events:
            self._layer_prefetch_events[layer_idx].synchronize()
            del self._layer_prefetch_events[layer_idx]
            self._layer_prefetch_in_progress[layer_idx] = False
            self.stats['prefetch_hits'] += 1
        else:
            self.stats['prefetch_misses'] += 1

    def _start_ffn_prefetch(self, layer_idx: int):
        """Start async prefetch of FFN components (called during attention compute)."""
        if not self.config.enable_component_prefetch:
            return
        if layer_idx >= self.num_layers:
            return
        if self._ffn_prefetch_in_progress.get(layer_idx, False):
            return

        self._ffn_prefetch_in_progress[layer_idx] = True
        event = torch.cuda.Event()
        self._ffn_prefetch_events[layer_idx] = event

        with torch.cuda.stream(self._ffn_stream):
            self._move_ffn_to_gpu(layer_idx, non_blocking=True)
            event.record()

    def _wait_for_ffn_prefetch(self, layer_idx: int):
        """Wait for FFN prefetch to complete."""
        if layer_idx in self._ffn_prefetch_events:
            self._ffn_prefetch_events[layer_idx].synchronize()
            del self._ffn_prefetch_events[layer_idx]
            self._ffn_prefetch_in_progress[layer_idx] = False
            self.stats['ffn_prefetch_overlaps'] += 1

    def _evict_layers_outside_window(self, current_layer: int):
        """Evict layers outside the sliding window."""
        window_size = self.config.num_layers_on_gpu
        window_start = max(0, current_layer - window_size + 1)
        window_end = current_layer

        layers_to_evict = []
        for idx in list(self._layers_on_gpu_set):
            if idx < window_start or idx > window_end:
                layers_to_evict.append(idx)

        for idx in layers_to_evict:
            self._move_layer_to_cpu(idx, use_pinned=self.config.use_pinned_memory)
            self.stats['layer_offloads'] += 1

            if self.config.verbose:
                logger.debug(f"Evicted layer {idx} (window: [{window_start}, {window_end}])")

    @time_logging_decorator("Level 3 - Ensure layer ready (hybrid)")
    def ensure_layer_on_gpu(self, layer_idx: int):
        """
        Ensure layer is on GPU.

        For now, we load the ENTIRE layer at once (like standard AIO).
        The component-level optimization (attention first, FFN prefetch)
        can be enabled later once the basic flow is stable.
        """
        if not self._initialized:
            self.prepare_for_inference()

        # Check if already on GPU
        if self._layer_on_gpu.get(layer_idx, False):
            if self._layer_prefetch_in_progress.get(layer_idx, False):
                self._wait_for_layer_prefetch(layer_idx)
            self._layers_on_gpu_set.add(layer_idx)
        else:
            # Need to load
            if self._layer_prefetch_in_progress.get(layer_idx, False):
                # Prefetch was started - wait for it
                self._wait_for_layer_prefetch(layer_idx)
            else:
                # Load entire layer synchronously
                self._move_layer_to_gpu(layer_idx, non_blocking=False)
                self.stats['prefetch_misses'] += 1

            self._layers_on_gpu_set.add(layer_idx)

        # Evict old layers
        self._evict_layers_outside_window(layer_idx)

        # Start prefetching next layers
        for i in range(1, self.config.prefetch_count + 1):
            next_idx = layer_idx + i
            if next_idx < self.num_layers:
                self._start_layer_prefetch(next_idx)

        # Periodic logging
        self._log_counter += 1
        if self._log_counter <= 3 or self._log_counter % 50 == 0:
            on_gpu = len(self._layers_on_gpu_set)
            logger.info(f"[HYBRID-OFFLOAD] Layer {layer_idx} ready | GPU: {on_gpu} layers")

    @time_logging_decorator("Level 3 - Ensure FFN ready (hybrid)")
    def ensure_ffn_ready(self, layer_idx: int):
        """
        Ensure FFN is ready (called before FFN computation).

        If FFN was being prefetched during attention, wait for it.
        """
        if self._ffn_prefetch_in_progress.get(layer_idx, False):
            self._wait_for_ffn_prefetch(layer_idx)

    @time_logging_decorator("Level 3 - Layer forward complete (hybrid)")
    def layer_forward_complete(self, layer_idx: int):
        """Called after a layer's forward pass is complete."""
        # Periodic cache clear
        if (layer_idx + 1) % self.config.empty_cache_frequency == 0:
            torch.cuda.empty_cache()
            self.stats['cache_clears'] += 1

    def get_statistics(self) -> Dict[str, Any]:
        """Get offloading statistics."""
        total_ops = self.stats['prefetch_hits'] + self.stats['prefetch_misses']
        prefetch_ratio = self.stats['prefetch_hits'] / max(total_ops, 1)

        return {
            **self.stats,
            'prefetch_hit_ratio': prefetch_ratio,
            'num_layers': self.num_layers,
            'layers_on_gpu': len(self._layers_on_gpu_set),
        }

    def print_statistics(self):
        """Print offloading statistics."""
        stats = self.get_statistics()
        print("\n" + "=" * 70)
        print("Hybrid Component Offloading Statistics")
        print("=" * 70)
        print(f"Strategy: Sliding window with FFN prefetch overlap")
        print("-" * 70)
        print(f"Total layers:              {stats['num_layers']}")
        print(f"Window size:               {self.config.num_layers_on_gpu}")
        print(f"Component prefetch:        {self.config.enable_component_prefetch}")
        print("-" * 70)
        print(f"Layer loads:               {stats['layer_loads']}")
        print(f"Layer offloads:            {stats['layer_offloads']}")
        print(f"Prefetch hits:             {stats['prefetch_hits']}")
        print(f"Prefetch misses:           {stats['prefetch_misses']}")
        print(f"Prefetch hit ratio:        {stats['prefetch_hit_ratio']*100:.1f}%")
        print(f"FFN prefetch overlaps:     {stats['ffn_prefetch_overlaps']}")
        print(f"Cache clears:              {stats['cache_clears']}")
        print("=" * 70 + "\n")

    def reset(self):
        """Reset state for a new inference run."""
        self._layer_prefetch_in_progress.clear()
        self._layer_prefetch_events.clear()
        self._ffn_prefetch_in_progress.clear()
        self._ffn_prefetch_events.clear()
        self.stats = {k: 0 for k in self.stats}
        self._log_counter = 0


def create_hybrid_offload_hooks(
    pipe,
    manager: HybridOffloadManager,
) -> Dict[str, Any]:
    """
    Create forward hooks for hybrid offloading.

    This matches the pattern from offload.py:
    - Iterate directly over pipe.transformer.transformer_blocks (the ModuleList)
    - Use explicit factory functions for hook creation

    Args:
        pipe: The HunyuanVideoPipeline
        manager: The HybridOffloadManager

    Returns:
        Dict of hook handles
    """
    handles = {}

    def create_pre_hook(offload_manager: HybridOffloadManager, layer_idx: int):
        """Factory function for pre-hooks (matches offload.py pattern)."""
        def hook(module, args):
            offload_manager.ensure_layer_on_gpu(layer_idx)
            return args
        return hook

    def create_post_hook(offload_manager: HybridOffloadManager, layer_idx: int):
        """Factory function for post-hooks."""
        def hook(module, args, output):
            offload_manager.layer_forward_complete(layer_idx)
            return output
        return hook

    # Install hooks on double stream blocks - iterate over the actual ModuleList
    for idx, block in enumerate(pipe.transformer.transformer_blocks):
        pre_hook = create_pre_hook(manager, idx)
        post_hook = create_post_hook(manager, idx)

        handles[f'double_{idx}_pre'] = block.register_forward_pre_hook(pre_hook)
        handles[f'double_{idx}_post'] = block.register_forward_hook(post_hook)

    # Install hooks on single stream blocks
    num_double = len(pipe.transformer.transformer_blocks)
    for idx, block in enumerate(pipe.transformer.single_transformer_blocks):
        layer_idx = num_double + idx
        pre_hook = create_pre_hook(manager, layer_idx)
        post_hook = create_post_hook(manager, layer_idx)

        handles[f'single_{idx}_pre'] = block.register_forward_pre_hook(pre_hook)
        handles[f'single_{idx}_post'] = block.register_forward_hook(post_hook)

    logger.info(f"Installed {len(handles)} hybrid offload hooks")
    return handles


def enable_component_offloading(
    pipe,
    ffn_layers_on_gpu: int = 6,
    use_pinned_memory: bool = True,
    enable_prefetch: bool = True,
    ffn_prefetch_count: int = 2,
    enable_component_prefetch: bool = False,  # Disabled for now - causes device issues
    verbose: bool = False,
) -> Tuple[HybridOffloadManager, Dict]:
    """
    Enable hybrid component offloading for a HunyuanVideo pipeline.

    This uses a sliding window approach similar to standard AIO:
    1. Keep N layers on GPU at a time (sliding window)
    2. Prefetch next layers while current layer computes
    3. Use pinned memory for faster CPU->GPU transfers

    Args:
        pipe: HunyuanVideoPipeline
        ffn_layers_on_gpu: Number of layers to keep on GPU (sliding window)
        use_pinned_memory: Use pinned CPU memory for faster transfers
        enable_prefetch: Enable async layer prefetching
        ffn_prefetch_count: Number of layers to prefetch ahead
        enable_component_prefetch: (Experimental) Enable FFN prefetch during attention
        verbose: Enable verbose logging

    Returns:
        Tuple of (HybridOffloadManager, hooks_dict)
    """
    config = HybridOffloadConfig(
        num_layers_on_gpu=ffn_layers_on_gpu,
        use_pinned_memory=use_pinned_memory,
        enable_prefetch=enable_prefetch,
        prefetch_count=ffn_prefetch_count,
        enable_component_prefetch=False,  # Force disabled for stability
        verbose=verbose,
    )

    transformer = pipe.transformer

    # Keep other transformer components on GPU FIRST (before moving blocks to CPU)
    # This follows the same order as offload.py
    components_to_keep = ['time_text_embed', 'x_embedder', 'context_embedder',
                          'norm_out', 'proj_out', 'rope']

    logger.info("Moving transformer embedder components to GPU...")
    for name in components_to_keep:
        if hasattr(transformer, name):
            comp = getattr(transformer, name)
            if comp is not None:
                comp.to(config.compute_device)
                first_param = next(comp.parameters(), None)
                if first_param is not None:
                    size_mb = sum(p.numel() * p.element_size() for p in comp.parameters()) / 1024**2
                    logger.info(f"  {name}: {size_mb:.1f}MB → {first_param.device}")

    # Move transformer blocks to CPU using the ModuleList's .to() method
    # This ensures proper PyTorch module tracking
    logger.info("Moving transformer blocks to CPU...")
    transformer.transformer_blocks.to('cpu')
    transformer.single_transformer_blocks.to('cpu')

    # Create manager (will set up tracking for the blocks)
    manager = HybridOffloadManager(transformer, config)

    # Prepare for inference - converts to pinned memory if enabled
    manager.prepare_for_inference()

    # Create and install hooks on blocks - pass pipe to iterate over actual ModuleList
    hooks = create_hybrid_offload_hooks(pipe, manager)

    # CRITICAL: Register a safety hook on the transformer to ensure embedders
    # are on GPU before every forward pass. This prevents device mismatches
    # when the pipeline or other code might move tensors around.
    def ensure_embedders_on_gpu(module, args):
        """Pre-hook to ensure embedders are on GPU before forward."""
        for name in components_to_keep:
            if hasattr(module, name):
                comp = getattr(module, name)
                if comp is not None:
                    first_param = next(comp.parameters(), None)
                    if first_param is not None and first_param.device.type != 'cuda':
                        comp.to('cuda')
                        if verbose:
                            logger.info(f"[HYBRID-OFFLOAD] Safety hook moved {name} to GPU")
        return args

    hook_handle = transformer.register_forward_pre_hook(ensure_embedders_on_gpu)
    hooks['transformer_embedder_safety'] = hook_handle
    logger.info("Registered embedder safety hook on transformer")

    torch.cuda.empty_cache()
    gc.collect()

    # Log memory status
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**3
        reserved = torch.cuda.memory_reserved() / 1024**3
        logger.info(f"GPU memory after setup: {allocated:.2f}GB allocated, {reserved:.2f}GB reserved")

    return manager, hooks


def estimate_memory_savings(
    pipe,
) -> Dict[str, float]:
    """Estimate memory usage for different offloading strategies."""
    transformer = pipe.transformer
    double_blocks = list(transformer.transformer_blocks)
    single_blocks = list(transformer.single_transformer_blocks)

    def get_param_memory(module):
        if module is None:
            return 0.0
        return sum(p.numel() * p.element_size() for p in module.parameters()) / (1024**3)

    # Full model memory
    full_model_gb = get_param_memory(transformer)

    # Per-component memory
    attention_gb = 0.0
    ffn_gb = 0.0
    norm_gb = 0.0

    for block in double_blocks:
        for name in DOUBLE_BLOCK_ATTENTION_COMPONENTS:
            if hasattr(block, name):
                attention_gb += get_param_memory(getattr(block, name))
        for name in DOUBLE_BLOCK_FFN_COMPONENTS:
            if hasattr(block, name):
                ffn_gb += get_param_memory(getattr(block, name))
        for name in DOUBLE_BLOCK_NORM_COMPONENTS:
            if hasattr(block, name):
                norm_gb += get_param_memory(getattr(block, name))

    for block in single_blocks:
        for name in SINGLE_BLOCK_ATTENTION_COMPONENTS:
            if hasattr(block, name):
                attention_gb += get_param_memory(getattr(block, name))
        for name in SINGLE_BLOCK_MLP_COMPONENTS:
            if hasattr(block, name):
                ffn_gb += get_param_memory(getattr(block, name))
        for name in SINGLE_BLOCK_NORM_COMPONENTS:
            if hasattr(block, name):
                norm_gb += get_param_memory(getattr(block, name))

    layer_count = len(double_blocks) + len(single_blocks)
    if layer_count > 0:
        avg_layer_gb = full_model_gb / layer_count
    else:
        avg_layer_gb = 0

    return {
        'full_model_gb': full_model_gb,
        'attention_total_gb': attention_gb,
        'ffn_total_gb': ffn_gb,
        'norm_total_gb': norm_gb,
        'avg_layer_gb': avg_layer_gb,
        'window_6_layers_gb': 6 * avg_layer_gb,
        'window_8_layers_gb': 8 * avg_layer_gb,
    }
