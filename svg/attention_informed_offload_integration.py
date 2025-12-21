"""
Integration helpers for Attention-Informed Offloading.

This module provides seamless integration with the existing HunyuanVideo pipeline
and the SAP/CTCA attention processors.
"""

from __future__ import annotations

import gc
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn

from .logger import logger
from .attention_informed_offload import (
    AttentionInformedOffloadManager,
    AttentionInformedOffloadConfig,
    create_attention_informed_offloader,
)

if TYPE_CHECKING:
    from .ctca import CrossTimestepClusterAmortization


# =============================================================================
# Memory Safety Validation
# =============================================================================

def validate_memory_safety(
    transformer: nn.Module,
    config: AttentionInformedOffloadConfig,
    activation_reserve_gb: float = 4.0,
    cuda_overhead_gb: float = 0.5,
) -> Tuple[bool, Dict[str, float]]:
    """
    Validate that the configuration will not cause OOM.

    This function computes the expected memory footprint and compares it
    to available GPU memory.

    Args:
        transformer: Transformer module
        config: Offload configuration
        activation_reserve_gb: Reserved memory for activations/caches
        cuda_overhead_gb: Reserved memory for CUDA runtime

    Returns:
        Tuple of (is_safe, memory_breakdown)
    """
    if not torch.cuda.is_available():
        return False, {"error": "CUDA not available"}

    # Get device properties
    device = torch.cuda.current_device()
    total_memory = torch.cuda.get_device_properties(device).total_memory
    total_gb = total_memory / (1024 ** 3)

    # Current allocations (VAE, embedders, etc.)
    allocated_memory = torch.cuda.memory_allocated()
    allocated_gb = allocated_memory / (1024 ** 3)

    # Estimate layer memory
    double_blocks = list(getattr(transformer, "transformer_blocks", []))
    single_blocks = list(getattr(transformer, "single_transformer_blocks", []))
    all_blocks = double_blocks + single_blocks

    if len(all_blocks) == 0:
        return False, {"error": "No transformer blocks found"}

    # Sample first layer
    layer = all_blocks[0]
    layer_params = sum(p.numel() * p.element_size() for p in layer.parameters())
    layer_buffers = sum(b.numel() * b.element_size() for b in layer.buffers() if b is not None)
    layer_memory_mb = (layer_params + layer_buffers) / (1024 * 1024)
    layer_memory_gb = layer_memory_mb / 1024

    # Compute memory breakdown
    sliding_window_gb = config.num_layers_on_gpu * layer_memory_gb
    prefetch_buffer_gb = config.max_prefetch_count * layer_memory_gb  # Worst case

    total_needed_gb = (
        allocated_gb +
        sliding_window_gb +
        prefetch_buffer_gb +
        activation_reserve_gb +
        cuda_overhead_gb
    )

    # Free memory
    try:
        free_bytes, _ = torch.cuda.mem_get_info()
        free_gb = free_bytes / (1024 ** 3)
    except Exception:
        free_gb = total_gb - allocated_gb

    available_gb = allocated_gb + free_gb

    # Safety check
    is_safe = total_needed_gb <= available_gb * 0.95  # 5% margin

    memory_breakdown = {
        "total_gpu_memory_gb": total_gb,
        "allocated_gb": allocated_gb,
        "free_gb": free_gb,
        "layer_memory_mb": layer_memory_mb,
        "num_layers": len(all_blocks),
        "layers_on_gpu": config.num_layers_on_gpu,
        "sliding_window_gb": sliding_window_gb,
        "prefetch_buffer_gb": prefetch_buffer_gb,
        "activation_reserve_gb": activation_reserve_gb,
        "cuda_overhead_gb": cuda_overhead_gb,
        "total_needed_gb": total_needed_gb,
        "available_gb": available_gb,
        "is_safe": is_safe,
        "safety_margin_gb": available_gb - total_needed_gb,
    }

    return is_safe, memory_breakdown


def suggest_safe_config(
    transformer: nn.Module,
    target_memory_fraction: float = 0.85,
    activation_reserve_gb: float = 4.0,
) -> AttentionInformedOffloadConfig:
    """
    Suggest a safe configuration based on available GPU memory.

    Args:
        transformer: Transformer module
        target_memory_fraction: Fraction of GPU memory to target
        activation_reserve_gb: Reserved memory for activations

    Returns:
        Suggested configuration that should be memory-safe
    """
    device = torch.cuda.current_device()
    total_memory = torch.cuda.get_device_properties(device).total_memory
    total_gb = total_memory / (1024 ** 3)

    allocated_gb = torch.cuda.memory_allocated() / (1024 ** 3)

    # Estimate layer memory
    double_blocks = list(getattr(transformer, "transformer_blocks", []))
    single_blocks = list(getattr(transformer, "single_transformer_blocks", []))
    all_blocks = double_blocks + single_blocks

    if len(all_blocks) == 0:
        logger.warning("No transformer blocks found, using default config")
        return AttentionInformedOffloadConfig()

    layer = all_blocks[0]
    layer_params = sum(p.numel() * p.element_size() for p in layer.parameters())
    layer_memory_gb = layer_params / (1024 ** 3)

    # Available budget
    budget_gb = total_gb * target_memory_fraction - allocated_gb - activation_reserve_gb

    # Calculate safe number of layers
    # Account for sliding window + prefetch buffer
    max_layers = int(budget_gb / layer_memory_gb)
    # Split between sliding window and prefetch
    suggested_layers_on_gpu = max(1, min(max_layers * 2 // 3, 10))
    suggested_prefetch = max(1, min(max_layers // 3, 4))

    logger.info(f"Memory analysis: total={total_gb:.1f}GB, allocated={allocated_gb:.1f}GB, "
                f"budget={budget_gb:.1f}GB, layer={layer_memory_gb*1024:.1f}MB")
    logger.info(f"Suggested: layers_on_gpu={suggested_layers_on_gpu}, "
                f"prefetch_count={suggested_prefetch}")

    return AttentionInformedOffloadConfig(
        num_layers_on_gpu=suggested_layers_on_gpu,
        base_prefetch_count=min(suggested_prefetch, 2),
        max_prefetch_count=suggested_prefetch,
        min_prefetch_count=1,
        use_pinned_memory=True,
        enable_prefetch=True,
        use_priority_eviction=True,
    )


# =============================================================================
# Integration with Attention Processors
# =============================================================================

class AttentionSignalCollector:
    """
    Collects attention signals from SAP/CTCA processors for the offload predictor.

    This class provides hooks that can be registered with attention processors
    to automatically feed signals to the offload manager.
    """

    def __init__(self, offload_manager: AttentionInformedOffloadManager):
        self.offload_manager = offload_manager
        self._layer_start_events: Dict[int, torch.cuda.Event] = {}

    def on_layer_start(self, layer_idx: int):
        """Called when a layer starts processing."""
        event = torch.cuda.Event(enable_timing=True)
        event.record()
        self._layer_start_events[layer_idx] = event

    def on_layer_complete(
        self,
        layer_idx: int,
        attention_density: float,
        timestep: int,
        ctaa_stats: Optional[Dict] = None,
    ):
        """
        Called when a layer completes processing.

        Args:
            layer_idx: Layer index
            attention_density: Computed attention density
            timestep: Current diffusion timestep
            ctaa_stats: Optional CTAA statistics
        """
        # Compute timing
        compute_time_ms = 0.0
        if layer_idx in self._layer_start_events:
            end_event = torch.cuda.Event(enable_timing=True)
            end_event.record()
            torch.cuda.synchronize()
            compute_time_ms = self._layer_start_events[layer_idx].elapsed_time(end_event)
            del self._layer_start_events[layer_idx]

        # Feed to predictor
        self.offload_manager.predictor.record_from_attention(
            layer_idx=layer_idx,
            timestep=timestep,
            attention_density=attention_density,
            compute_time_ms=compute_time_ms,
            ctaa_stats=ctaa_stats,
        )


def patch_attention_processor_for_signals(
    processor,
    signal_collector: AttentionSignalCollector,
):
    """
    Patch an attention processor to collect signals for the offloader.

    This modifies the processor's attention_core_logic to report signals.

    Args:
        processor: SAP or CTCA attention processor
        signal_collector: Signal collector to receive the signals
    """
    original_attention_core_logic = processor.attention_core_logic

    def patched_attention_core_logic(query, key, value, timestep, layer_idx, cu_max_seqlens):
        # Signal start
        signal_collector.on_layer_start(layer_idx)

        # Run original
        output = original_attention_core_logic(query, key, value, timestep, layer_idx, cu_max_seqlens)

        # Try to extract density (processor-specific)
        attention_density = 0.5  # Default
        ctaa_stats = None

        # For SAP/CTCA processors that store density
        if hasattr(processor, '_last_density'):
            attention_density = processor._last_density
        elif hasattr(processor, 'top_p_kmeans'):
            # Estimate from config
            attention_density = processor.top_p_kmeans

        # For CTAA processors
        if hasattr(processor, '_ctaa_video_full_map'):
            try:
                from ..kmeans_utils import get_ctaa_statistics
                ctaa_stats = get_ctaa_statistics()
            except Exception:
                pass

        # Signal complete
        timestep_val = timestep[0].item() if hasattr(timestep, '__getitem__') else timestep
        signal_collector.on_layer_complete(
            layer_idx=layer_idx,
            attention_density=attention_density,
            timestep=int(timestep_val),
            ctaa_stats=ctaa_stats,
        )

        return output

    processor.attention_core_logic = patched_attention_core_logic
    return processor


# =============================================================================
# Pipeline Integration
# =============================================================================

def enable_attention_informed_offloading(
    pipe,
    num_layers_on_gpu: int = 6,
    use_pinned_memory: bool = True,
    enable_prefetch: bool = True,
    use_priority_eviction: bool = True,
    criticality_profile_path: Optional[str] = None,
    ctca_manager: Optional[CrossTimestepClusterAmortization] = None,
    auto_tune: bool = True,
    verbose: bool = False,
) -> Tuple[AttentionInformedOffloadManager, AttentionSignalCollector]:
    """
    Enable attention-informed offloading for a HunyuanVideo pipeline.

    This is the main entry point for integrating attention-informed offloading.

    Args:
        pipe: HunyuanVideoPipeline
        num_layers_on_gpu: Number of layers to keep on GPU
        use_pinned_memory: Use pinned CPU memory
        enable_prefetch: Enable async prefetching
        use_priority_eviction: Use priority-based eviction
        criticality_profile_path: Path to criticality profile
        ctca_manager: Optional CTCA manager for signal integration
        auto_tune: Automatically adjust config for memory safety
        verbose: Enable verbose logging

    Returns:
        Tuple of (offload_manager, signal_collector)

    Usage:
        # After loading pipeline and replacing attention:
        manager, collector = enable_attention_informed_offloading(
            pipe,
            num_layers_on_gpu=6,
            ctca_manager=ctca_manager,
        )

        # Run inference (signals collected automatically)
        output = pipe(...)

        # Print statistics
        manager.print_statistics()
    """
    transformer = pipe.transformer

    # Auto-tune config if requested
    if auto_tune:
        config = suggest_safe_config(transformer)
        config.num_layers_on_gpu = min(config.num_layers_on_gpu, num_layers_on_gpu)
        config.use_pinned_memory = use_pinned_memory
        config.enable_prefetch = enable_prefetch
        config.use_priority_eviction = use_priority_eviction
        config.criticality_profile_path = criticality_profile_path
        config.verbose = verbose
    else:
        config = AttentionInformedOffloadConfig(
            num_layers_on_gpu=num_layers_on_gpu,
            use_pinned_memory=use_pinned_memory,
            enable_prefetch=enable_prefetch,
            use_priority_eviction=use_priority_eviction,
            criticality_profile_path=criticality_profile_path,
            verbose=verbose,
        )

    # Validate memory safety
    is_safe, breakdown = validate_memory_safety(transformer, config)
    if not is_safe:
        logger.warning(f"Configuration may cause OOM! "
                       f"Needed: {breakdown['total_needed_gb']:.1f}GB, "
                       f"Available: {breakdown['available_gb']:.1f}GB")
        if auto_tune:
            # Further reduce layers
            safe_layers = max(1, int(config.num_layers_on_gpu * 0.7))
            config.num_layers_on_gpu = safe_layers
            config.max_prefetch_count = min(config.max_prefetch_count, 2)
            logger.info(f"Reduced to {safe_layers} layers on GPU for safety")
    else:
        logger.info(f"Memory check passed. Margin: {breakdown['safety_margin_gb']:.1f}GB")

    # Create manager
    manager = AttentionInformedOffloadManager(transformer, config)

    # Set CTCA manager if provided
    if ctca_manager is not None:
        manager.set_ctca_manager(ctca_manager)

    # Create signal collector
    collector = AttentionSignalCollector(manager)

    # Move transformer blocks to CPU (pipeline may have loaded to GPU)
    logger.info("Moving transformer blocks to CPU for offloading...")
    for block in manager.all_blocks:
        block.to('cpu')

    # Set up pinned memory
    if use_pinned_memory:
        logger.info("Converting to pinned memory...")
        for idx in range(manager.num_layers):
            manager._ensure_pinned_memory(idx)
            manager._layer_on_gpu[idx] = False

    # Keep embedders and other components on GPU
    components_to_keep = ['timestep_embedding', 'context_embedder', 'x_embedder',
                          'time_text_embed', 'norm_out', 'proj_out']
    for name in components_to_keep:
        if hasattr(transformer, name):
            comp = getattr(transformer, name)
            if comp is not None:
                comp.to('cuda')

    # Initialize manager
    manager.prepare_for_inference()

    torch.cuda.empty_cache()
    gc.collect()

    logger.info(f"Attention-informed offloading enabled with {config.num_layers_on_gpu} "
                f"layers on GPU, priority_eviction={use_priority_eviction}")

    return manager, collector


def create_offload_hooks(
    manager: AttentionInformedOffloadManager,
    collector: AttentionSignalCollector,
) -> Dict[str, Any]:
    """
    Create forward hooks for automatic offload management.

    Returns hooks that can be registered with transformer blocks.
    """
    hooks = {}

    def make_pre_hook(layer_idx: int):
        def pre_hook(module, inputs):
            manager.ensure_layer_on_gpu(layer_idx)
            collector.on_layer_start(layer_idx)
        return pre_hook

    def make_post_hook(layer_idx: int):
        def post_hook(module, inputs, outputs):
            # Extract density from module if available
            density = getattr(module, '_last_attention_density', 0.5)
            manager.layer_forward_complete(layer_idx, attention_density=density)
        return post_hook

    # Create hooks for each layer
    for idx, block in enumerate(manager.all_blocks):
        pre_hook = block.register_forward_pre_hook(make_pre_hook(idx))
        post_hook = block.register_forward_hook(make_post_hook(idx))
        hooks[f'layer_{idx}_pre'] = pre_hook
        hooks[f'layer_{idx}_post'] = post_hook

    return hooks


# =============================================================================
# Convenience: Full Setup
# =============================================================================

def setup_full_attention_informed_offloading(
    pipe,
    prompt: str,
    num_layers_on_gpu: int = 6,
    verbose: bool = False,
) -> Tuple[Dict[str, torch.Tensor], AttentionInformedOffloadManager, Dict]:
    """
    Full setup: pre-encode prompt + enable attention-informed offloading.

    This is a one-stop function for setting up offloading with all optimizations.

    Args:
        pipe: HunyuanVideoPipeline
        prompt: Text prompt to pre-encode
        num_layers_on_gpu: Number of layers on GPU
        verbose: Enable verbose logging

    Returns:
        Tuple of (pre_encoded_embeds, offload_manager, hooks)

    Usage:
        embeds, manager, hooks = setup_full_attention_informed_offloading(
            pipe, prompt="A cat walks..."
        )

        output = pipe(
            prompt_embeds=embeds['prompt_embeds'].cuda(),
            ...
        )

        manager.print_statistics()
    """
    from .offload import pre_encode_and_offload

    # Pre-encode prompt (moves text encoders to GPU, encodes, offloads)
    logger.info("Pre-encoding prompt...")
    pre_encoded = pre_encode_and_offload(pipe, prompt)

    # Enable attention-informed offloading
    logger.info("Setting up attention-informed offloading...")
    manager, collector = enable_attention_informed_offloading(
        pipe,
        num_layers_on_gpu=num_layers_on_gpu,
        verbose=verbose,
    )

    # Create hooks
    hooks = create_offload_hooks(manager, collector)

    return pre_encoded, manager, hooks
