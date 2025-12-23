"""
Fine-Grained Component-Level Offloading for Video Diffusion Models

This module implements Strategy A: Pin Attention, Offload FFN

Key insight: FFN (Feed-Forward Network) dominates memory usage (~70% of each layer)
but is computationally simpler than attention. By keeping attention weights on GPU
and dynamically loading FFN weights, we can:
1. Eliminate transfer latency for attention (compute-bound operation)
2. Overlap FFN transfers with attention computation
3. Achieve ~3x better memory efficiency vs full-layer offloading

Memory layout:
- GPU (Pinned Zone): All attention components across all layers
- GPU (Dynamic Zone): Sliding window of FFN components + activations
- CPU (Pinned Memory): FFN components not in GPU window

Component breakdown per layer:
┌─────────────────────────────────────────────────────────────────────────┐
│ Double Stream Block (~1.3GB)                                            │
├─────────────────────────────────────────────────────────────────────────┤
│ Attention (pinned): ~150MB (12%)                                        │
│   - img_attn_qkv, img_attn_proj, txt_attn_qkv, txt_attn_proj           │
│   - img_attn_q_norm, img_attn_k_norm, txt_attn_q_norm, txt_attn_k_norm │
│                                                                         │
│ Modulation/Norm (pinned): ~50MB (4%)                                    │
│   - img_mod, txt_mod                                                    │
│   - img_norm1, img_norm2, txt_norm1, txt_norm2                         │
│                                                                         │
│ FFN (offloadable): ~900MB (70%)                                         │
│   - img_mlp.fc1 (~300MB), img_mlp.fc2 (~300MB)                         │
│   - txt_mlp.fc1 (~150MB), txt_mlp.fc2 (~150MB)                         │
└─────────────────────────────────────────────────────────────────────────┘
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
class ComponentOffloadConfig:
    """Configuration for component-level offloading."""

    # Device settings
    compute_device: str = "cuda"
    offload_device: str = "cpu"

    # Memory settings
    use_pinned_memory: bool = True

    # FFN sliding window size (number of layers with FFN on GPU)
    # Since FFN is ~500MB/layer for double blocks, 6 layers = ~3GB
    ffn_layers_on_gpu: int = 6

    # Prefetching settings
    enable_prefetch: bool = True
    ffn_prefetch_count: int = 2  # Prefetch FFN for next N layers

    # Memory management
    empty_cache_frequency: int = 10

    # Debugging
    verbose: bool = False


@dataclass
class LayerComponents:
    """Tracks components of a single transformer layer."""
    layer_idx: int
    layer_type: str  # "double" or "single"

    # For double stream blocks
    attention_components: List[str] = field(default_factory=list)
    ffn_components: List[str] = field(default_factory=list)
    norm_components: List[str] = field(default_factory=list)

    # Memory estimates
    attention_memory_mb: float = 0.0
    ffn_memory_mb: float = 0.0
    norm_memory_mb: float = 0.0


# Component classification for Double Stream Blocks (diffusers HunyuanVideoTransformerBlock)
# These are the actual component names used in diffusers implementation
DOUBLE_BLOCK_ATTENTION_COMPONENTS = [
    'attn',  # The attention module
]

DOUBLE_BLOCK_FFN_COMPONENTS = [
    'ff',         # Feed-forward network for hidden states
    'ff_context', # Feed-forward network for context/text
]

DOUBLE_BLOCK_NORM_COMPONENTS = [
    'norm1',          # AdaLayerNormContinuous (contains linear for modulation)
    'norm1_context',  # AdaLayerNormContinuous for context
    'norm2',          # LayerNorm
    'norm2_context',  # LayerNorm for context
]

# Single stream blocks (diffusers HunyuanVideoSingleTransformerBlock)
# Note: Single stream has combined projections, harder to split
SINGLE_BLOCK_ATTENTION_COMPONENTS = [
    'attn',     # Attention module
]

SINGLE_BLOCK_MLP_COMPONENTS = [
    'proj_out',  # Output projection (combines attention and MLP output)
]

SINGLE_BLOCK_NORM_COMPONENTS = [
    'norm',      # AdaLayerNormContinuous
    'proj_mlp',  # MLP up-projection (small, keep with norm)
    'act_mlp',   # Activation (no parameters, but keep for consistency)
]


class ComponentOffloadManager:
    """
    Manages fine-grained component-level offloading.

    Strategy: Pin Attention on GPU, dynamically offload FFN

    This manager:
    1. Keeps all attention components on GPU (zero transfer latency)
    2. Maintains a sliding window of FFN components on GPU
    3. Prefetches FFN components during attention computation
    4. Overlaps FFN transfer with attention compute
    """

    def __init__(
        self,
        transformer: nn.Module,
        config: ComponentOffloadConfig,
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

        # Analyze components for each layer
        self._layer_components: Dict[int, LayerComponents] = {}
        self._analyze_layer_components()

        # Track FFN locations
        self._ffn_on_gpu: Set[int] = set()
        self._ffn_pinned: Dict[int, bool] = {}

        # CUDA streams
        self._prefetch_stream: Optional[torch.cuda.Stream] = None
        self._compute_stream: Optional[torch.cuda.Stream] = None

        # Prefetch state
        self._prefetch_in_progress: Dict[int, bool] = {}
        self._prefetch_events: Dict[int, torch.cuda.Event] = {}

        # Statistics
        self.stats = {
            'attention_pinned_mb': 0.0,
            'ffn_loads': 0,
            'ffn_offloads': 0,
            'prefetch_hits': 0,
            'prefetch_misses': 0,
            'overlap_achieved': 0,
        }

        self._initialized = False
        self._log_counter = 0

    def _analyze_layer_components(self):
        """Analyze and classify components for each layer."""
        logger.info("Analyzing layer components for fine-grained offloading...")

        # Analyze double stream blocks
        for idx, block in enumerate(self.double_blocks):
            components = LayerComponents(
                layer_idx=idx,
                layer_type="double",
                attention_components=DOUBLE_BLOCK_ATTENTION_COMPONENTS.copy(),
                ffn_components=DOUBLE_BLOCK_FFN_COMPONENTS.copy(),
                norm_components=DOUBLE_BLOCK_NORM_COMPONENTS.copy(),
            )

            # Estimate memory for each component type
            for comp_name in components.attention_components:
                if hasattr(block, comp_name):
                    comp = getattr(block, comp_name)
                    components.attention_memory_mb += self._get_module_memory_mb(comp)

            for comp_name in components.ffn_components:
                if hasattr(block, comp_name):
                    comp = getattr(block, comp_name)
                    components.ffn_memory_mb += self._get_module_memory_mb(comp)

            for comp_name in components.norm_components:
                if hasattr(block, comp_name):
                    comp = getattr(block, comp_name)
                    components.norm_memory_mb += self._get_module_memory_mb(comp)

            self._layer_components[idx] = components

        # Analyze single stream blocks
        for idx, block in enumerate(self.single_blocks):
            layer_idx = self.num_double + idx
            components = LayerComponents(
                layer_idx=layer_idx,
                layer_type="single",
                attention_components=SINGLE_BLOCK_ATTENTION_COMPONENTS.copy(),
                ffn_components=SINGLE_BLOCK_MLP_COMPONENTS.copy(),
                norm_components=SINGLE_BLOCK_NORM_COMPONENTS.copy(),
            )

            for comp_name in components.attention_components:
                if hasattr(block, comp_name):
                    comp = getattr(block, comp_name)
                    components.attention_memory_mb += self._get_module_memory_mb(comp)

            for comp_name in components.ffn_components:
                if hasattr(block, comp_name):
                    comp = getattr(block, comp_name)
                    components.ffn_memory_mb += self._get_module_memory_mb(comp)

            for comp_name in components.norm_components:
                if hasattr(block, comp_name):
                    comp = getattr(block, comp_name)
                    components.norm_memory_mb += self._get_module_memory_mb(comp)

            self._layer_components[layer_idx] = components

        # Log analysis results
        total_attn = sum(c.attention_memory_mb for c in self._layer_components.values())
        total_ffn = sum(c.ffn_memory_mb for c in self._layer_components.values())
        total_norm = sum(c.norm_memory_mb for c in self._layer_components.values())

        logger.info(f"Component analysis complete:")
        logger.info(f"  Total Attention: {total_attn:.1f}MB ({total_attn/1024:.2f}GB)")
        logger.info(f"  Total FFN: {total_ffn:.1f}MB ({total_ffn/1024:.2f}GB)")
        logger.info(f"  Total Norm/Mod: {total_norm:.1f}MB ({total_norm/1024:.2f}GB)")

        if len(self._layer_components) > 0:
            sample = list(self._layer_components.values())[0]
            logger.info(f"  Per-layer breakdown (layer 0):")
            logger.info(f"    Attention: {sample.attention_memory_mb:.1f}MB")
            logger.info(f"    FFN: {sample.ffn_memory_mb:.1f}MB")
            logger.info(f"    Norm: {sample.norm_memory_mb:.1f}MB")

    def _get_module_memory_mb(self, module: nn.Module) -> float:
        """Get memory usage of a module in MB."""
        if module is None:
            return 0.0
        total = sum(p.numel() * p.element_size() for p in module.parameters())
        total += sum(b.numel() * b.element_size() for b in module.buffers() if b is not None)
        return total / (1024 * 1024)

    def _get_block(self, layer_idx: int) -> nn.Module:
        """Get the block for a given layer index."""
        if layer_idx < self.num_double:
            return self.double_blocks[layer_idx]
        else:
            return self.single_blocks[layer_idx - self.num_double]

    def prepare_for_inference(self):
        """
        Prepare components for inference.

        This:
        1. Moves all attention components to GPU (pinned)
        2. Moves all FFN components to CPU (with pinned memory)
        3. Sets up CUDA streams for async prefetching
        """
        if self._initialized:
            return

        logger.info("=" * 70)
        logger.info("[COMPONENT-OFFLOAD] Initializing Fine-Grained Component Offloading")
        logger.info("=" * 70)
        logger.info(f"[COMPONENT-OFFLOAD] Strategy: Pin Attention, Offload FFN")
        logger.info(f"[COMPONENT-OFFLOAD] Total layers: {self.num_layers} "
                   f"({self.num_double} double, {self.num_single} single)")
        logger.info(f"[COMPONENT-OFFLOAD] FFN window size: {self.config.ffn_layers_on_gpu} layers")
        logger.info("=" * 70)

        # Create CUDA streams
        if self.config.enable_prefetch:
            self._prefetch_stream = torch.cuda.Stream()
            self._compute_stream = torch.cuda.Stream()

        total_pinned = 0.0

        # Process each layer
        for layer_idx in range(self.num_layers):
            block = self._get_block(layer_idx)
            components = self._layer_components[layer_idx]

            # Move attention and norm components to GPU (permanently pinned)
            for comp_name in components.attention_components + components.norm_components:
                if hasattr(block, comp_name):
                    comp = getattr(block, comp_name)
                    if comp is not None:
                        comp.to(self.config.compute_device)
                        total_pinned += self._get_module_memory_mb(comp)

            # Move FFN components to CPU with pinned memory
            for comp_name in components.ffn_components:
                if hasattr(block, comp_name):
                    comp = getattr(block, comp_name)
                    if comp is not None:
                        self._move_ffn_to_cpu(layer_idx, comp, use_pinned=self.config.use_pinned_memory)

        self.stats['attention_pinned_mb'] = total_pinned

        torch.cuda.empty_cache()
        gc.collect()

        self._initialized = True

        logger.info(f"[COMPONENT-OFFLOAD] Attention pinned on GPU: {total_pinned:.1f}MB "
                   f"({total_pinned/1024:.2f}GB)")
        logger.info(f"[COMPONENT-OFFLOAD] FFN offloaded to CPU with pinned memory")
        logger.info("=" * 70)

    def _move_ffn_to_cpu(
        self,
        layer_idx: int,
        module: nn.Module,
        use_pinned: bool = True
    ):
        """Move FFN module to CPU with optional pinned memory."""
        if use_pinned:
            for param in module.parameters():
                if param.device.type != 'cpu':
                    cpu_tensor = param.data.cpu()
                    if not cpu_tensor.is_pinned():
                        pinned_tensor = torch.empty_like(cpu_tensor, pin_memory=True)
                        pinned_tensor.copy_(cpu_tensor)
                        param.data = pinned_tensor
                    else:
                        param.data = cpu_tensor
            self._ffn_pinned[layer_idx] = True
        else:
            module.to('cpu')
            self._ffn_pinned[layer_idx] = False

        self._ffn_on_gpu.discard(layer_idx)

    def _move_ffn_to_gpu(
        self,
        layer_idx: int,
        non_blocking: bool = False
    ):
        """Move FFN components to GPU."""
        block = self._get_block(layer_idx)
        components = self._layer_components[layer_idx]

        for comp_name in components.ffn_components:
            if hasattr(block, comp_name):
                comp = getattr(block, comp_name)
                if comp is not None:
                    comp.to(self.config.compute_device, non_blocking=non_blocking)

        self._ffn_on_gpu.add(layer_idx)
        self.stats['ffn_loads'] += 1

    def _offload_ffn_from_gpu(self, layer_idx: int):
        """Offload FFN components from GPU to CPU."""
        if layer_idx not in self._ffn_on_gpu:
            return

        block = self._get_block(layer_idx)
        components = self._layer_components[layer_idx]

        for comp_name in components.ffn_components:
            if hasattr(block, comp_name):
                comp = getattr(block, comp_name)
                if comp is not None:
                    self._move_ffn_to_cpu(layer_idx, comp, use_pinned=self.config.use_pinned_memory)

        self._ffn_on_gpu.discard(layer_idx)
        self.stats['ffn_offloads'] += 1

    def _start_ffn_prefetch(self, layer_idx: int):
        """Start async prefetch of FFN for a layer."""
        if not self.config.enable_prefetch:
            return

        if layer_idx >= self.num_layers:
            return

        if layer_idx in self._ffn_on_gpu:
            return  # Already on GPU

        if self._prefetch_in_progress.get(layer_idx, False):
            return  # Already prefetching

        self._prefetch_in_progress[layer_idx] = True

        event = torch.cuda.Event()
        self._prefetch_events[layer_idx] = event

        with torch.cuda.stream(self._prefetch_stream):
            self._move_ffn_to_gpu(layer_idx, non_blocking=True)
            event.record()

        if self.config.verbose:
            logger.debug(f"Started FFN prefetch for layer {layer_idx}")

    def _wait_for_ffn_prefetch(self, layer_idx: int):
        """Wait for FFN prefetch to complete."""
        if layer_idx in self._prefetch_events:
            self._prefetch_events[layer_idx].synchronize()
            del self._prefetch_events[layer_idx]
            self._prefetch_in_progress[layer_idx] = False
            self.stats['prefetch_hits'] += 1
        else:
            self.stats['prefetch_misses'] += 1

    def _evict_oldest_ffn(self, incoming_layer: int):
        """Evict FFN components outside the sliding window."""
        max_on_gpu = self.config.ffn_layers_on_gpu
        current_on_gpu = len(self._ffn_on_gpu)

        if current_on_gpu < max_on_gpu:
            return

        # Sliding window eviction
        window_end = incoming_layer
        window_start = max(0, incoming_layer - max_on_gpu + 1)

        layers_to_evict = []
        for idx in list(self._ffn_on_gpu):
            if idx < window_start or idx > window_end:
                layers_to_evict.append(idx)

        for idx in layers_to_evict:
            self._offload_ffn_from_gpu(idx)

            if self.config.verbose:
                logger.debug(f"Evicted FFN for layer {idx}")

    @time_logging_decorator("Level 3 - Ensure attention ready (component)")
    def ensure_attention_ready(self, layer_idx: int):
        """
        Ensure attention components are ready (they should already be on GPU).

        This is called BEFORE attention computation.
        Since attention is pinned, this is a no-op but serves as a hook point.
        """
        if not self._initialized:
            self.prepare_for_inference()

        # Attention is always on GPU - nothing to do
        # But we can start prefetching FFN for this layer here
        # so it overlaps with attention computation
        if layer_idx not in self._ffn_on_gpu:
            self._start_ffn_prefetch(layer_idx)

        # Periodic logging
        self._log_counter += 1
        if self._log_counter <= 3 or self._log_counter % 50 == 0:
            ffn_on_gpu = len(self._ffn_on_gpu)
            prefetch_count = len([k for k, v in self._prefetch_in_progress.items() if v])
            logger.info(f"[COMPONENT-OFFLOAD] Layer {layer_idx} attention ready | "
                       f"FFN on GPU: {ffn_on_gpu} | Prefetching: {prefetch_count}")

    @time_logging_decorator("Level 3 - Ensure FFN ready (component)")
    def ensure_ffn_ready(self, layer_idx: int):
        """
        Ensure FFN components are on GPU.

        This is called AFTER attention computation but BEFORE FFN computation.
        Ideally, the FFN was already prefetched during attention.
        """
        if layer_idx in self._ffn_on_gpu:
            # Check if prefetch is still in progress
            if self._prefetch_in_progress.get(layer_idx, False):
                self._wait_for_ffn_prefetch(layer_idx)
                self.stats['overlap_achieved'] += 1
            return

        # Need to load synchronously (prefetch miss)
        if self._prefetch_in_progress.get(layer_idx, False):
            self._wait_for_ffn_prefetch(layer_idx)
        else:
            self._move_ffn_to_gpu(layer_idx, non_blocking=False)
            self.stats['prefetch_misses'] += 1

        self._ffn_on_gpu.add(layer_idx)

        # Evict old FFN
        self._evict_oldest_ffn(layer_idx)

        # Start prefetching next layers' FFN
        for i in range(1, self.config.ffn_prefetch_count + 1):
            next_idx = layer_idx + i
            if next_idx < self.num_layers:
                self._start_ffn_prefetch(next_idx)

    @time_logging_decorator("Level 3 - Layer forward complete (component)")
    def layer_forward_complete(self, layer_idx: int):
        """
        Called after a layer's forward pass is complete.
        """
        # Eviction is handled in ensure_ffn_ready
        # Just do periodic cache clearing
        if (layer_idx + 1) % self.config.empty_cache_frequency == 0:
            torch.cuda.empty_cache()

    def get_statistics(self) -> Dict[str, Any]:
        """Get offloading statistics."""
        total_ops = self.stats['prefetch_hits'] + self.stats['prefetch_misses']
        if total_ops > 0:
            prefetch_ratio = self.stats['prefetch_hits'] / total_ops
            overlap_ratio = self.stats['overlap_achieved'] / total_ops
        else:
            prefetch_ratio = 0.0
            overlap_ratio = 0.0

        return {
            **self.stats,
            'prefetch_hit_ratio': prefetch_ratio,
            'overlap_ratio': overlap_ratio,
            'num_layers': self.num_layers,
            'ffn_on_gpu_count': len(self._ffn_on_gpu),
        }

    def print_statistics(self):
        """Print offloading statistics."""
        stats = self.get_statistics()
        print("\n" + "=" * 70)
        print("Fine-Grained Component Offloading Statistics")
        print("=" * 70)
        print(f"Strategy: Pin Attention, Offload FFN")
        print("-" * 70)
        print(f"Total layers:              {stats['num_layers']}")
        print(f"Attention pinned on GPU:   {stats['attention_pinned_mb']:.1f}MB "
              f"({stats['attention_pinned_mb']/1024:.2f}GB)")
        print(f"FFN layers on GPU:         {self.config.ffn_layers_on_gpu}")
        print("-" * 70)
        print(f"FFN loads:                 {stats['ffn_loads']}")
        print(f"FFN offloads:              {stats['ffn_offloads']}")
        print(f"Prefetch hits:             {stats['prefetch_hits']}")
        print(f"Prefetch misses:           {stats['prefetch_misses']}")
        print(f"Prefetch hit ratio:        {stats['prefetch_hit_ratio']*100:.1f}%")
        print("-" * 70)
        print(f"Overlap achieved:          {stats['overlap_achieved']}")
        print(f"Overlap ratio:             {stats['overlap_ratio']*100:.1f}%")
        print("=" * 70 + "\n")

    def reset(self):
        """Reset state for a new inference run."""
        self._prefetch_in_progress.clear()
        self._prefetch_events.clear()
        self.stats = {k: (0.0 if isinstance(v, float) else 0) for k, v in self.stats.items()}


def create_component_offload_hooks(
    manager: ComponentOffloadManager,
) -> Dict[str, Any]:
    """
    Create forward hooks that intercept at attention/FFN boundaries.

    For Double Stream Blocks (diffusers HunyuanVideoTransformerBlock):
    - Pre-hook on block: ensure_attention_ready() (attention weights pinned)
    - Pre-hook before ff: ensure_ffn_ready()

    For Single Stream Blocks (diffusers HunyuanVideoSingleTransformerBlock):
    - Pre-hook on block: ensure_attention_ready()
    - Pre-hook before proj_out: ensure_ffn_ready()
    """
    hooks = {}

    # Double stream blocks
    for idx, block in enumerate(manager.double_blocks):
        # Hook for attention phase - runs at block start
        def make_block_pre_hook(layer_idx: int):
            def hook(module, inputs):
                manager.ensure_attention_ready(layer_idx)
                return inputs
            return hook

        # Hook for FFN phase - runs before ff module
        def make_ffn_pre_hook(layer_idx: int):
            def hook(module, inputs):
                manager.ensure_ffn_ready(layer_idx)
                return inputs
            return hook

        # Hook for completion
        def make_post_hook(layer_idx: int):
            def hook(module, inputs, outputs):
                manager.layer_forward_complete(layer_idx)
                return outputs
            return hook

        # Register pre-hook on the block itself for attention readiness
        h = block.register_forward_pre_hook(make_block_pre_hook(idx))
        hooks[f'double_{idx}_block_pre'] = h

        # Hook the FFN (ff) to catch when we need FFN weights
        if hasattr(block, 'ff'):
            h = block.ff.register_forward_pre_hook(make_ffn_pre_hook(idx))
            hooks[f'double_{idx}_ffn_pre'] = h

        # Post hook on the block itself
        h = block.register_forward_hook(make_post_hook(idx))
        hooks[f'double_{idx}_post'] = h

    # Single stream blocks
    for idx, block in enumerate(manager.single_blocks):
        layer_idx = manager.num_double + idx

        def make_block_pre_hook(layer_idx: int):
            def hook(module, inputs):
                manager.ensure_attention_ready(layer_idx)
                return inputs
            return hook

        def make_ffn_pre_hook(layer_idx: int):
            def hook(module, inputs):
                manager.ensure_ffn_ready(layer_idx)
                return inputs
            return hook

        def make_post_hook(layer_idx: int):
            def hook(module, inputs, outputs):
                manager.layer_forward_complete(layer_idx)
                return outputs
            return hook

        # Register pre-hook on the block for attention readiness
        h = block.register_forward_pre_hook(make_block_pre_hook(layer_idx))
        hooks[f'single_{idx}_block_pre'] = h

        # Hook proj_out for FFN/MLP phase
        if hasattr(block, 'proj_out'):
            h = block.proj_out.register_forward_pre_hook(make_ffn_pre_hook(layer_idx))
            hooks[f'single_{idx}_ffn_pre'] = h

        # Post hook
        h = block.register_forward_hook(make_post_hook(layer_idx))
        hooks[f'single_{idx}_post'] = h

    logger.info(f"Created {len(hooks)} component offload hooks")
    return hooks


def enable_component_offloading(
    pipe,
    ffn_layers_on_gpu: int = 6,
    use_pinned_memory: bool = True,
    enable_prefetch: bool = True,
    ffn_prefetch_count: int = 2,
    verbose: bool = False,
) -> Tuple[ComponentOffloadManager, Dict]:
    """
    Enable fine-grained component offloading for a HunyuanVideo pipeline.

    This implements Strategy A: Pin Attention, Offload FFN

    Args:
        pipe: HunyuanVideoPipeline
        ffn_layers_on_gpu: Number of layers to keep FFN on GPU (sliding window)
        use_pinned_memory: Use pinned CPU memory for faster transfers
        enable_prefetch: Enable async FFN prefetching
        ffn_prefetch_count: Number of layers to prefetch FFN ahead
        verbose: Enable verbose logging

    Returns:
        Tuple of (ComponentOffloadManager, hooks_dict)

    Usage:
        # Pre-encode prompt first
        prompt_embeds = pre_encode_and_offload(pipe, prompt)

        # Enable component offloading
        manager, hooks = enable_component_offloading(pipe, ffn_layers_on_gpu=6)

        # Run inference
        output = pipe(prompt_embeds=..., ...)

        # Print stats
        manager.print_statistics()
    """
    config = ComponentOffloadConfig(
        ffn_layers_on_gpu=ffn_layers_on_gpu,
        use_pinned_memory=use_pinned_memory,
        enable_prefetch=enable_prefetch,
        ffn_prefetch_count=ffn_prefetch_count,
        verbose=verbose,
    )

    transformer = pipe.transformer

    # Create manager
    manager = ComponentOffloadManager(transformer, config)

    # Prepare for inference (moves attention to GPU, FFN to CPU)
    manager.prepare_for_inference()

    # Create and install hooks
    hooks = create_component_offload_hooks(manager)

    # Keep other transformer components on GPU
    components_to_keep = ['time_text_embed', 'x_embedder', 'context_embedder',
                          'norm_out', 'proj_out', 'rope']
    for name in components_to_keep:
        if hasattr(transformer, name):
            comp = getattr(transformer, name)
            if comp is not None:
                comp.to(config.compute_device)

    torch.cuda.empty_cache()
    gc.collect()

    return manager, hooks


def estimate_memory_savings(
    pipe,
) -> Dict[str, float]:
    """
    Estimate memory savings from component offloading.

    Compares:
    1. Full model on GPU
    2. Layer-level offloading (current approach)
    3. Component-level offloading (this approach)
    """
    transformer = pipe.transformer
    double_blocks = list(transformer.transformer_blocks)
    single_blocks = list(transformer.single_transformer_blocks)

    def get_param_memory(module):
        if module is None:
            return 0.0
        return sum(p.numel() * p.element_size() for p in module.parameters()) / (1024**3)

    # Full model memory
    full_model_gb = get_param_memory(transformer)

    # Pinned memory (attention + norms - will stay on GPU)
    pinned_gb = 0.0
    for block in double_blocks:
        for name in DOUBLE_BLOCK_ATTENTION_COMPONENTS + DOUBLE_BLOCK_NORM_COMPONENTS:
            if hasattr(block, name):
                pinned_gb += get_param_memory(getattr(block, name))
    for block in single_blocks:
        for name in SINGLE_BLOCK_ATTENTION_COMPONENTS + SINGLE_BLOCK_NORM_COMPONENTS:
            if hasattr(block, name):
                pinned_gb += get_param_memory(getattr(block, name))

    # FFN memory (will be offloaded)
    ffn_gb = 0.0
    for block in double_blocks:
        for name in DOUBLE_BLOCK_FFN_COMPONENTS:
            if hasattr(block, name):
                ffn_gb += get_param_memory(getattr(block, name))
    for block in single_blocks:
        for name in SINGLE_BLOCK_MLP_COMPONENTS:
            if hasattr(block, name):
                ffn_gb += get_param_memory(getattr(block, name))

    # Layer-level: 6 layers on GPU
    layer_count = len(double_blocks) + len(single_blocks)
    if layer_count > 0:
        avg_layer_gb = full_model_gb / layer_count
        layer_offload_gb = 6 * avg_layer_gb
        avg_ffn_gb = ffn_gb / layer_count
    else:
        avg_layer_gb = 0
        layer_offload_gb = 0
        avg_ffn_gb = 0

    # Component-level: Attention pinned + 6 layers of FFN
    component_offload_gb = pinned_gb + 6 * avg_ffn_gb

    return {
        'full_model_gb': full_model_gb,
        'attention_pinned_gb': pinned_gb,
        'ffn_total_gb': ffn_gb,
        'layer_offload_6layers_gb': layer_offload_gb,
        'component_offload_6layers_gb': component_offload_gb,
        'savings_vs_layer_offload_gb': layer_offload_gb - component_offload_gb,
    }
