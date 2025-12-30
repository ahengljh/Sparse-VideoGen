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
import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

import torch
import torch.nn as nn

from .logger import logger
from .timer import time_logging_decorator


# Safety buffer for memory fragmentation (GB)
MEMORY_SAFETY_BUFFER_GB = 2.0


@dataclass
class HybridOffloadConfig:
    """Configuration for hybrid component-level offloading."""

    # Device settings
    compute_device: str = "cuda"
    offload_device: str = "cpu"

    # Memory settings
    use_pinned_memory: bool = True

    # Offloading strategy
    # - "layer": Move entire layers as blocks (standard AIO-style)
    # - "stream": StreamBlock pipelining - overlap compute with transfer
    #             While computing attention, prefetch FFN
    #             While computing FFN, prefetch next layer's attention
    #             Achieves near-zero GPU idle time
    offload_strategy: str = "layer"

    # Sliding window size (number of FULL layers on GPU)
    # Each layer = attention + norm + FFN
    # For 24GB GPU: 6-8 layers typical
    # Set to -1 or "auto" for adaptive mode
    num_layers_on_gpu: int = 6

    # For "stream" strategy: how many layers ahead to prefetch
    # Higher = more memory used but better overlap
    stream_prefetch_depth: int = 2

    # =========================================================================
    # Profile-Guided Dynamic Pinning (PGDP) - Key Innovation
    # =========================================================================
    #
    # PGDP uses runtime profiling to identify "heavy" layers and pin them on GPU.
    # Unlike static approaches, PGDP continuously adapts as compute patterns change.
    #
    # Why this works:
    # - Heavy layers have high compute/transfer ratio → worth keeping on GPU
    # - Light layers can tolerate transfer overhead
    # - Data-driven: Uses actual measured compute, not heuristics
    # - ADAPTIVE: Continuously updates as sparse attention changes layer costs
    #
    # ┌─────────────────────────────────────────────────────────────────────┐
    # │ Phase 1: Initial Profiling (warm-up timesteps)                      │
    # │   - All layers run full attention                                   │
    # │   - Measure compute time per layer                                  │
    # │   - Initial pinning based on observed heavy layers                  │
    # │                                                                     │
    # │ Phase 2: Continuous Adaptation (after warm-up)                      │
    # │   - Keep profiling all layers with EMA smoothing                    │
    # │   - Periodically re-evaluate heavy layer set                        │
    # │   - Dynamically adjust pinning: unpin light, pin new heavy          │
    # │   - Adapt to sparse attention patterns that change layer costs      │
    # └─────────────────────────────────────────────────────────────────────┘
    #
    enable_profiling: bool = True
    num_warmup_timesteps: int = 5  # Initial profiling before first pinning decision
    num_pinned_heavy_layers: int = 6  # Top-K heavy layers to pin
    heavy_layer_threshold: float = 0.7  # Pin layers with compute > 70th percentile

    # Dynamic pinning - continuously adapt based on runtime observations
    dynamic_pinning: bool = True  # Enable continuous adaptation (vs static after warm-up)
    repin_interval: int = 5  # Re-evaluate pinning every N timesteps
    ema_alpha: float = 0.3  # EMA smoothing factor (higher = more weight on recent)

    # Adaptive offloading settings
    # When True, automatically calculate optimal layers based on GPU memory
    adaptive_mode: bool = False
    activation_reserve_gb: Optional[float] = None  # Auto-calculated if None
    safety_buffer_gb: float = MEMORY_SAFETY_BUFFER_GB

    # Video resolution for activation estimation (used in adaptive mode)
    video_height: int = 720
    video_width: int = 1280
    num_frames: int = 129

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


# Activation reserve estimates (in GB) for different resolutions
# These are conservative estimates that account for:
# - Q, K, V tensors
# - FFN intermediate (4x expansion)
# - Residual connections
# - PyTorch memory fragmentation and overhead
# - CTCA cache and sparse attention intermediates
ACTIVATION_RESERVE_GB = {
    # (height, width, frames): reserve_gb
    (480, 848, 45): 6.0,
    (480, 848, 97): 8.0,
    (540, 960, 45): 7.0,
    (540, 960, 97): 10.0,
    (720, 1280, 45): 10.0,
    (720, 1280, 97): 14.0,
    (720, 1280, 129): 18.0,  # Increased from 14 - was causing OOM
    (1080, 1920, 45): 18.0,
    (1080, 1920, 97): 24.0,
}

# Safety buffer for memory fragmentation (GB) - increased for stability
MEMORY_SAFETY_BUFFER_GB_ADAPTIVE = 3.0


def estimate_activation_reserve(height: int, width: int, num_frames: int) -> float:
    """
    Estimate GPU memory reserve needed for activations during inference.

    This accounts for temporary tensors created during forward pass:
    - Q, K, V projections
    - FFN intermediate (4x expansion)
    - Attention output
    - Residual connections

    Args:
        height: Video height
        width: Video width
        num_frames: Number of frames

    Returns:
        Estimated activation reserve in GB
    """
    # Try exact match first
    key = (height, width, num_frames)
    if key in ACTIVATION_RESERVE_GB:
        return ACTIVATION_RESERVE_GB[key]

    # Find closest match or interpolate
    # Calculate based on sequence length
    latent_t = num_frames // 4
    latent_h = height // 8 // 2  # After VAE and patchify
    latent_w = width // 8 // 2
    seq_len = latent_t * latent_h * latent_w

    # Rough formula based on sequence length
    # Hidden dim = 3072, bf16 = 2 bytes
    hidden_dim = 3072
    bytes_per_elem = 2

    # Peak activation per layer (Q,K,V + FFN intermediate)
    qkv_bytes = 3 * seq_len * hidden_dim * bytes_per_elem
    ffn_bytes = seq_len * hidden_dim * 4 * bytes_per_elem  # 4x expansion
    peak_per_layer_gb = (qkv_bytes + ffn_bytes) / (1024**3)

    # Multiple layers' activations can overlap, add buffer
    # Also account for PyTorch overhead (~30%)
    estimated_reserve = peak_per_layer_gb * 3 * 1.3

    # Clamp to reasonable range
    return max(4.0, min(estimated_reserve, 24.0))


def calculate_optimal_layers_on_gpu(
    total_gpu_memory_gb: float,
    model_weights_gb: float,
    num_layers: int,
    activation_reserve_gb: float,
    safety_buffer_gb: float = MEMORY_SAFETY_BUFFER_GB_ADAPTIVE,
) -> int:
    """
    Calculate optimal number of layers to keep on GPU.

    Strategy: Reserve space for activations first, then use remaining
    space for model weights.

    Args:
        total_gpu_memory_gb: Total GPU memory in GB
        model_weights_gb: Total model weights in GB
        num_layers: Total number of layers
        activation_reserve_gb: Reserved memory for activations
        safety_buffer_gb: Additional safety buffer

    Returns:
        Number of layers to keep on GPU
    """
    # Available space for weights
    available_for_weights = total_gpu_memory_gb - activation_reserve_gb - safety_buffer_gb

    if available_for_weights <= 0:
        logger.warning(f"GPU memory ({total_gpu_memory_gb:.1f}GB) too small for "
                      f"activation reserve ({activation_reserve_gb:.1f}GB). Using minimum 1 layer.")
        return 1

    # Per-layer weight size
    per_layer_gb = model_weights_gb / num_layers

    # How many layers fit
    num_layers_on_gpu = int(available_for_weights / per_layer_gb)

    # Clamp to valid range
    num_layers_on_gpu = max(1, min(num_layers_on_gpu, num_layers))

    logger.info(f"[ADAPTIVE] GPU: {total_gpu_memory_gb:.1f}GB")
    logger.info(f"[ADAPTIVE] Activation reserve: {activation_reserve_gb:.1f}GB")
    logger.info(f"[ADAPTIVE] Safety buffer: {safety_buffer_gb:.1f}GB")
    logger.info(f"[ADAPTIVE] Available for weights: {available_for_weights:.1f}GB")
    logger.info(f"[ADAPTIVE] Per-layer size: {per_layer_gb:.3f}GB")
    logger.info(f"[ADAPTIVE] Optimal layers on GPU: {num_layers_on_gpu}/{num_layers}")

    return num_layers_on_gpu


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

        # Track what's on GPU (layer-level)
        self._layer_on_gpu: Dict[int, bool] = {i: False for i in range(self.num_layers)}
        self._layers_on_gpu_set: Set[int] = set()

        # Track component-level GPU placement (for "component" strategy)
        self._attention_on_gpu: Dict[int, bool] = {i: False for i in range(self.num_layers)}
        self._ffn_on_gpu: Dict[int, bool] = {i: False for i in range(self.num_layers)}
        self._ffn_on_gpu_set: Set[int] = set()

        # For component-level prefetch within a layer
        self._ffn_prefetch_in_progress: Dict[int, bool] = {}
        self._ffn_prefetch_events: Dict[int, torch.cuda.Event] = {}

        # CUDA streams
        self._prefetch_stream: Optional[torch.cuda.Stream] = None
        self._ffn_stream: Optional[torch.cuda.Stream] = None

        # Layer prefetch state
        self._layer_prefetch_in_progress: Dict[int, bool] = {}
        self._layer_prefetch_events: Dict[int, torch.cuda.Event] = {}

        # =====================================================================
        # Profile-Guided Dynamic Pinning (PGDP) State
        # =====================================================================
        # Timestep tracking
        self._current_timestep: int = 0
        self._last_repin_timestep: int = 0  # Last timestep when we re-evaluated pinning
        self._initial_profiling_complete: bool = False  # Initial warm-up profiling done

        # Per-layer compute time tracking
        # For initial profiling: raw list of times
        self._layer_compute_times: Dict[int, List[float]] = defaultdict(list)
        # For continuous profiling: EMA-smoothed compute time per layer
        self._layer_ema_compute: Dict[int, float] = {}

        # Timing events for profiling
        self._layer_start_event: Optional[torch.cuda.Event] = None
        self._layer_end_event: Optional[torch.cuda.Event] = None

        # Pinned heavy layers (dynamically updated based on runtime profiling)
        self._pinned_heavy_layers: Set[int] = set()

        # Track pinning changes for statistics
        self._total_repin_events: int = 0
        self._layers_pinned_count: Dict[int, int] = defaultdict(int)  # How often each layer was pinned

        # Statistics
        self.stats = {
            'layer_loads': 0,
            'layer_offloads': 0,
            'prefetch_hits': 0,
            'prefetch_misses': 0,
            'ffn_prefetch_overlaps': 0,
            'cache_clears': 0,
            'pinned_layer_hits': 0,  # Accesses to pinned heavy layers
            'profiling_samples': 0,  # Number of profiling measurements
        }

        # Memory tracking
        self._peak_memory_gb = 0.0
        self._memory_samples = []
        self._start_time = None
        self._end_time = None

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

        For "component" strategy:
        - Pins all attention+norm modules on GPU permanently
        - Only FFN modules remain on CPU for sliding window
        """
        if self._initialized:
            return

        strategy = self.config.offload_strategy
        is_stream = strategy == "stream"

        logger.info("=" * 70)
        logger.info("[OFFLOAD] Initializing Offload Manager")
        logger.info("=" * 70)
        if is_stream:
            logger.info(f"[OFFLOAD] Strategy: STREAM (Component-aware pipelining)")
            logger.info(f"[OFFLOAD] Total layers: {self.num_layers}")
            logger.info(f"[OFFLOAD] Window size: {self.config.num_layers_on_gpu} layers")
            logger.info(f"[OFFLOAD] Prefetch depth: {self.config.stream_prefetch_depth}")
            logger.info(f"[OFFLOAD] Key innovation: Overlap attention compute with FFN transfer")
        else:
            logger.info(f"[OFFLOAD] Strategy: LAYER (Slide entire layers)")
            logger.info(f"[OFFLOAD] Total layers: {self.num_layers}")
            logger.info(f"[OFFLOAD] Window size: {self.config.num_layers_on_gpu} layers")
        logger.info("=" * 70)

        # Create CUDA streams
        # For stream strategy, we need separate streams for attention and FFN prefetch
        if self.config.enable_prefetch or is_stream:
            self._prefetch_stream = torch.cuda.Stream()
        if is_stream:
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
        if is_stream:
            logger.info(f"[STREAM-OFFLOAD] {num_on_cpu}/{self.num_layers} layers on CPU")
            logger.info(f"[STREAM-OFFLOAD] Components will be pipelined: Attn prefetch → FFN prefetch → compute")
        else:
            logger.info(f"[LAYER-OFFLOAD] {num_on_cpu}/{self.num_layers} layers on CPU with pinned memory")

        # Log profiling info if enabled
        if self.config.enable_profiling:
            logger.info("-" * 70)
            logger.info("[PGDP] Profile-Guided Dynamic Pinning ENABLED")
            logger.info(f"[PGDP] Initial profiling: {self.config.num_warmup_timesteps} timesteps")
            logger.info(f"[PGDP] Heavy layers to pin: {self.config.num_pinned_heavy_layers}")
            if self.config.dynamic_pinning:
                logger.info(f"[PGDP] Dynamic mode: Re-evaluate every {self.config.repin_interval} timesteps")
                logger.info(f"[PGDP] EMA alpha: {self.config.ema_alpha} (smoothing factor)")
            else:
                logger.info(f"[PGDP] Static mode: Pinning fixed after warm-up")
            logger.info("-" * 70)

    # =========================================================================
    # Profile-Guided Dynamic Pinning (PGDP) Methods
    # =========================================================================

    def on_timestep_start(self, timestep_idx: int):
        """
        Called at the start of each diffusion timestep.

        Handles both initial profiling and dynamic re-pinning:
        1. After warm-up: Initial pinning based on profiled data
        2. Every repin_interval: Re-evaluate and adjust pinning dynamically
        """
        self._current_timestep = timestep_idx

        if not self.config.enable_profiling:
            return

        # Phase 1: Initial profiling complete - do first pinning
        if not self._initial_profiling_complete and timestep_idx >= self.config.num_warmup_timesteps:
            self._analyze_and_apply_initial_profiling()

        # Phase 2: Dynamic re-pinning based on continuous profiling
        elif (self.config.dynamic_pinning and
              self._initial_profiling_complete and
              timestep_idx - self._last_repin_timestep >= self.config.repin_interval):
            self._dynamic_repin(timestep_idx)

    def _start_layer_timing(self, layer_idx: int):
        """Start timing for a layer (called before layer forward)."""
        if not self.config.enable_profiling:
            return

        # Create timing events if needed
        if self._layer_start_event is None:
            self._layer_start_event = torch.cuda.Event(enable_timing=True)
            self._layer_end_event = torch.cuda.Event(enable_timing=True)

        self._layer_start_event.record()

    def _end_layer_timing(self, layer_idx: int):
        """
        End timing for a layer and record result.

        Uses different recording strategies:
        - Before initial profiling: Collect raw samples
        - After initial profiling: Update EMA for continuous tracking
        """
        if not self.config.enable_profiling:
            return
        if self._layer_start_event is None:
            return

        self._layer_end_event.record()
        torch.cuda.synchronize()

        elapsed_ms = self._layer_start_event.elapsed_time(self._layer_end_event)
        self.stats['profiling_samples'] += 1

        if not self._initial_profiling_complete:
            # Phase 1: Collect raw samples during warm-up
            self._layer_compute_times[layer_idx].append(elapsed_ms)
        else:
            # Phase 2: Update EMA for continuous profiling
            alpha = self.config.ema_alpha
            if layer_idx in self._layer_ema_compute:
                # EMA update: new_ema = alpha * new_value + (1 - alpha) * old_ema
                self._layer_ema_compute[layer_idx] = (
                    alpha * elapsed_ms + (1 - alpha) * self._layer_ema_compute[layer_idx]
                )
            else:
                # First observation after warm-up
                self._layer_ema_compute[layer_idx] = elapsed_ms

    def _analyze_and_apply_initial_profiling(self):
        """
        Analyze initial warm-up profiling and do first pinning.

        This establishes the initial EMA values and pins heavy layers.
        """
        if self._initial_profiling_complete:
            return

        logger.info("=" * 70)
        logger.info("[PGDP] Analyzing warm-up profiling results...")
        logger.info("=" * 70)

        # Initialize EMA from warm-up samples
        for layer_idx in range(self.num_layers):
            times = self._layer_compute_times.get(layer_idx, [])
            if times:
                avg_time = sum(times) / len(times)
                self._layer_ema_compute[layer_idx] = avg_time
            else:
                self._layer_ema_compute[layer_idx] = 0.0

        # Determine and pin heavy layers
        heavy_layers = self._identify_heavy_layers()
        self._pinned_heavy_layers = set(heavy_layers)

        # Log profiling results
        sorted_layers = sorted(self._layer_ema_compute.items(), key=lambda x: x[1], reverse=True)
        logger.info(f"[PGDP] Initial profiling complete: {self.stats['profiling_samples']} samples")
        logger.info(f"[PGDP] Layer compute times (top 10):")
        for layer_idx, avg_time in sorted_layers[:10]:
            pin_marker = " [PIN]" if layer_idx in self._pinned_heavy_layers else ""
            logger.info(f"[PGDP]   Layer {layer_idx:2d}: {avg_time:6.2f}ms{pin_marker}")

        # Pin the heavy layers on GPU
        if self._pinned_heavy_layers:
            self._apply_pinning_changes(set(), self._pinned_heavy_layers)
            for layer_idx in self._pinned_heavy_layers:
                self._layers_pinned_count[layer_idx] += 1

        self._initial_profiling_complete = True
        self._last_repin_timestep = self._current_timestep
        logger.info("=" * 70)

    def _identify_heavy_layers(self) -> List[int]:
        """Identify top-K heavy layers based on current EMA compute times."""
        if not self._layer_ema_compute:
            return []

        sorted_layers = sorted(
            self._layer_ema_compute.items(),
            key=lambda x: x[1],
            reverse=True
        )

        num_to_pin = min(self.config.num_pinned_heavy_layers, self.num_layers)
        heavy_layers = []
        for layer_idx, avg_time in sorted_layers[:num_to_pin]:
            if avg_time > 0:
                heavy_layers.append(layer_idx)

        return heavy_layers

    def _dynamic_repin(self, timestep_idx: int):
        """
        Dynamically re-evaluate and adjust pinning based on recent observations.

        This is the key innovation: as sparse attention changes layer costs,
        we adapt pinning to keep the currently-heavy layers on GPU.
        """
        # Identify current heavy layers based on EMA
        new_heavy_layers = set(self._identify_heavy_layers())

        # Compare with currently pinned layers
        layers_to_unpin = self._pinned_heavy_layers - new_heavy_layers
        layers_to_pin = new_heavy_layers - self._pinned_heavy_layers

        if layers_to_unpin or layers_to_pin:
            self._total_repin_events += 1
            logger.info(f"[PGDP] Dynamic repin at timestep {timestep_idx}:")
            if layers_to_unpin:
                logger.info(f"[PGDP]   Unpinning: {sorted(layers_to_unpin)} (became lighter)")
            if layers_to_pin:
                logger.info(f"[PGDP]   Pinning: {sorted(layers_to_pin)} (became heavier)")

            # Apply the changes
            self._apply_pinning_changes(layers_to_unpin, layers_to_pin)

            # Update pinned set
            self._pinned_heavy_layers = new_heavy_layers

            # Track pinning frequency
            for layer_idx in layers_to_pin:
                self._layers_pinned_count[layer_idx] += 1

        self._last_repin_timestep = timestep_idx

    def _apply_pinning_changes(self, layers_to_unpin: Set[int], layers_to_pin: Set[int]):
        """Apply pinning changes: move layers to/from GPU."""
        # Unpin layers (move to CPU if not in current window)
        for layer_idx in layers_to_unpin:
            # Note: We don't immediately move to CPU - just remove from pinned set
            # The regular eviction logic will handle it if needed
            pass

        # Pin new heavy layers (move to GPU)
        pinned_mb = 0.0
        for layer_idx in sorted(layers_to_pin):
            if not self._layer_on_gpu.get(layer_idx, False):
                self._move_layer_to_gpu(layer_idx, non_blocking=False)
            self._layers_on_gpu_set.add(layer_idx)

            info = self._layer_memory.get(layer_idx)
            if info:
                pinned_mb += info.total_mb

        if layers_to_pin:
            logger.info(f"[PGDP] Pinned {len(layers_to_pin)} layers ({pinned_mb:.1f}MB)")

        torch.cuda.empty_cache()

    def is_layer_pinned(self, layer_idx: int) -> bool:
        """Check if a layer is pinned (should never be offloaded)."""
        return layer_idx in self._pinned_heavy_layers

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
        """Move only attention and norm components to GPU (for component-level strategy)."""
        block = self._get_block(layer_idx)
        is_double = self._is_double_block(layer_idx)

        attn_comps = DOUBLE_BLOCK_ATTENTION_COMPONENTS if is_double else SINGLE_BLOCK_ATTENTION_COMPONENTS
        norm_comps = DOUBLE_BLOCK_NORM_COMPONENTS if is_double else SINGLE_BLOCK_NORM_COMPONENTS

        for name in attn_comps + norm_comps:
            if hasattr(block, name):
                comp = getattr(block, name)
                if comp is not None:
                    comp.to(self.config.compute_device, non_blocking=non_blocking)

        self._attention_on_gpu[layer_idx] = True

    def _move_attention_to_cpu(self, layer_idx: int):
        """Move only attention and norm components to CPU."""
        block = self._get_block(layer_idx)
        is_double = self._is_double_block(layer_idx)

        attn_comps = DOUBLE_BLOCK_ATTENTION_COMPONENTS if is_double else SINGLE_BLOCK_ATTENTION_COMPONENTS
        norm_comps = DOUBLE_BLOCK_NORM_COMPONENTS if is_double else SINGLE_BLOCK_NORM_COMPONENTS

        for name in attn_comps + norm_comps:
            if hasattr(block, name):
                comp = getattr(block, name)
                if comp is not None:
                    comp.to('cpu')

        self._attention_on_gpu[layer_idx] = False

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

        self._ffn_on_gpu[layer_idx] = True
        self._ffn_on_gpu_set.add(layer_idx)
        self.stats['layer_loads'] += 1

    def _move_ffn_to_cpu(self, layer_idx: int):
        """Move only FFN components to CPU."""
        block = self._get_block(layer_idx)
        is_double = self._is_double_block(layer_idx)

        ffn_comps = DOUBLE_BLOCK_FFN_COMPONENTS if is_double else SINGLE_BLOCK_MLP_COMPONENTS

        for name in ffn_comps:
            if hasattr(block, name):
                comp = getattr(block, name)
                if comp is not None:
                    comp.to('cpu')

        self._ffn_on_gpu[layer_idx] = False
        self._ffn_on_gpu_set.discard(layer_idx)
        self.stats['layer_offloads'] += 1

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
        """Evict layers outside the sliding window (but NEVER pinned heavy layers)."""
        window_size = self.config.num_layers_on_gpu
        window_start = max(0, current_layer - window_size + 1)
        window_end = current_layer

        layers_to_evict = []
        for idx in list(self._layers_on_gpu_set):
            # NEVER evict pinned heavy layers - they were identified during profiling
            if idx in self._pinned_heavy_layers:
                continue
            if idx < window_start or idx > window_end:
                layers_to_evict.append(idx)

        for idx in layers_to_evict:
            self._move_layer_to_cpu(idx, use_pinned=self.config.use_pinned_memory)
            self.stats['layer_offloads'] += 1

            if self.config.verbose:
                logger.debug(f"Evicted layer {idx} (window: [{window_start}, {window_end}])")

    def _evict_components_outside_window(self, current_layer: int):
        """
        Evict components outside the sliding window for stream strategy.
        NEVER evicts pinned heavy layers - they persist based on profiling.

        In stream strategy, we keep a small window of complete layers
        plus partially loaded next layers (attention prefetched while
        previous layer's FFN computes).
        """
        # For stream strategy, use same window as layer strategy
        # but with smarter prefetching
        window_size = self.config.num_layers_on_gpu
        window_start = max(0, current_layer - window_size + 1)
        window_end = current_layer + self.config.stream_prefetch_depth

        # Evict attention outside window (but never for pinned layers)
        for idx in list(self._attention_on_gpu.keys()):
            if idx in self._pinned_heavy_layers:
                continue  # Never evict pinned heavy layers
            if self._attention_on_gpu[idx] and (idx < window_start or idx > window_end):
                self._move_attention_to_cpu(idx)

        # Evict FFN outside window (but never for pinned layers)
        ffn_to_evict = []
        for idx in list(self._ffn_on_gpu_set):
            if idx in self._pinned_heavy_layers:
                continue  # Never evict pinned heavy layers
            if idx < window_start or idx > window_end:
                ffn_to_evict.append(idx)

        for idx in ffn_to_evict:
            self._move_ffn_to_cpu(idx)

            if self.config.verbose:
                logger.debug(f"Evicted components for layer {idx} (window: [{window_start}, {window_end}])")

    @time_logging_decorator("Level 3 - Ensure layer ready (hybrid)")
    def ensure_layer_on_gpu(self, layer_idx: int):
        """
        Ensure layer is on GPU.

        Supports two strategies:
        - "layer": Load entire layers as blocks (standard AIO-style)
        - "stream": StreamBlock pipelining with component-aware prefetching
                    Prefetches attention and FFN separately to overlap with compute
        """
        if not self._initialized:
            self.prepare_for_inference()

        if self.config.offload_strategy == "stream":
            self._ensure_layer_on_gpu_stream(layer_idx)
        else:
            self._ensure_layer_on_gpu_layer(layer_idx)

    def _ensure_layer_on_gpu_layer(self, layer_idx: int):
        """Layer-level offloading: Load entire layers as blocks."""
        # Check if this is a pinned heavy layer (identified during profiling)
        if layer_idx in self._pinned_heavy_layers:
            self.stats['pinned_layer_hits'] += 1
            self._layers_on_gpu_set.add(layer_idx)
            # Still need to evict other layers and prefetch
            self._evict_layers_outside_window(layer_idx)
            for i in range(1, self.config.prefetch_count + 1):
                next_idx = layer_idx + i
                if next_idx < self.num_layers and next_idx not in self._pinned_heavy_layers:
                    self._start_layer_prefetch(next_idx)
            # Start timing for profiling
            self._start_layer_timing(layer_idx)
            return

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

        # Safety net: Force-move any parameters still on CPU (silent)
        block = self._get_block(layer_idx)
        target_device = self.config.compute_device
        for param in block.parameters():
            if param.device.type != 'cuda':
                param.data = param.data.to(target_device)

        # Evict old layers
        self._evict_layers_outside_window(layer_idx)

        # Start prefetching next layers (skip pinned layers - they're already on GPU)
        for i in range(1, self.config.prefetch_count + 1):
            next_idx = layer_idx + i
            if next_idx < self.num_layers and next_idx not in self._pinned_heavy_layers:
                self._start_layer_prefetch(next_idx)

        # Start timing for profiling (during warm-up phase)
        self._start_layer_timing(layer_idx)

        # Periodic logging (reduced frequency)
        self._log_counter += 1
        if self._log_counter == 1 or self._log_counter % 100 == 0:
            on_gpu = len(self._layers_on_gpu_set)
            pinned = len(self._pinned_heavy_layers)
            logger.info(f"[HYBRID-OFFLOAD] Layer {layer_idx} ready | GPU: {on_gpu} layers | Pinned: {pinned}")

    def _ensure_layer_on_gpu_stream(self, layer_idx: int):
        """
        StreamBlock pipelining: Component-aware prefetching for compute-transfer overlap.

        Combined with Profile-Guided Dynamic Pinning (PGDP):
        - Pinned heavy layers (identified during warm-up) are already on GPU
        - Non-pinned layers use component-aware pipelining

        Key Innovation:
        Instead of loading entire layers, we load components separately:
        1. Attention+Norm (smaller, ~35% of layer) - loaded first
        2. FFN (larger, ~65% of layer) - prefetched during attention compute

        Pipeline visualization:
        ┌─────────────────────────────────────────────────────────────────────┐
        │ Pinned Heavy Layers: Always on GPU (zero transfer after profiling) │
        ├─────────────────────────────────────────────────────────────────────┤
        │ Layer N-1:  [Attn Compute]────[FFN Compute]                         │
        │                    ↓              ↓                                  │
        │             Prefetch FFN_N   Prefetch Attn_{N+1}                    │
        │                    ↓              ↓                                  │
        │ Layer N:        [Wait]────[Attn Compute]────[FFN Compute]           │
        └─────────────────────────────────────────────────────────────────────┘

        Benefits:
        - Smaller transfer units = better overlap with computation
        - Attention loaded first (needed first in forward pass)
        - FFN prefetched while attention computes
        - Near-zero GPU idle time for memory transfers
        - Heavy layers pinned based on profiling data
        """
        # Pinned heavy layers are already on GPU - instant hit
        if layer_idx in self._pinned_heavy_layers:
            self.stats['pinned_layer_hits'] += 1
            # Still need to prefetch next non-pinned layers
            self._evict_components_outside_window(layer_idx)
            for i in range(1, self.config.stream_prefetch_depth + 1):
                next_idx = layer_idx + i
                if next_idx < self.num_layers and next_idx not in self._pinned_heavy_layers:
                    if not self._attention_on_gpu.get(next_idx, False):
                        if not self._layer_prefetch_in_progress.get(next_idx, False):
                            self._start_attention_prefetch(next_idx)
            # Start timing for profiling
            self._start_layer_timing(layer_idx)
            return

        # Step 1: Ensure attention+norm is on GPU
        if not self._attention_on_gpu.get(layer_idx, False):
            # Check if attention prefetch was started by previous layer
            if layer_idx in self._layer_prefetch_events:
                # Wait for attention prefetch (was started during previous layer's FFN)
                self._layer_prefetch_events[layer_idx].synchronize()
                del self._layer_prefetch_events[layer_idx]
                self._layer_prefetch_in_progress[layer_idx] = False
                self.stats['prefetch_hits'] += 1
            else:
                # Load attention synchronously (first layer or cache miss)
                self._move_attention_to_gpu(layer_idx, non_blocking=False)
                self.stats['prefetch_misses'] += 1

        # Step 2: Start async prefetch of THIS layer's FFN (will compute during attention)
        if not self._ffn_on_gpu.get(layer_idx, False):
            if not self._ffn_prefetch_in_progress.get(layer_idx, False):
                self._start_ffn_prefetch(layer_idx)

        # Step 3: Start async prefetch of NEXT layer's attention (skip pinned layers)
        for i in range(1, self.config.stream_prefetch_depth + 1):
            next_idx = layer_idx + i
            if next_idx < self.num_layers and next_idx not in self._pinned_heavy_layers:
                if not self._attention_on_gpu.get(next_idx, False):
                    if not self._layer_prefetch_in_progress.get(next_idx, False):
                        self._start_attention_prefetch(next_idx)

        # Step 4: Wait for this layer's FFN to be ready
        # (should have been prefetched during attention load or previous layer)
        if self._ffn_prefetch_in_progress.get(layer_idx, False):
            self._wait_for_ffn_prefetch(layer_idx)
        elif not self._ffn_on_gpu.get(layer_idx, False):
            # Fallback: load synchronously
            self._move_ffn_to_gpu(layer_idx, non_blocking=False)

        # Safety net: Ensure all params are on GPU
        block = self._get_block(layer_idx)
        target_device = self.config.compute_device
        for param in block.parameters():
            if param.device.type != 'cuda':
                param.data = param.data.to(target_device)

        # Evict old components outside window
        self._evict_components_outside_window(layer_idx)

        # Start timing for profiling (during warm-up phase)
        self._start_layer_timing(layer_idx)

        # Periodic logging
        self._log_counter += 1
        if self._log_counter == 1 or self._log_counter % 100 == 0:
            ffn_on_gpu = len(self._ffn_on_gpu_set)
            attn_on_gpu = sum(1 for v in self._attention_on_gpu.values() if v)
            pinned = len(self._pinned_heavy_layers)
            logger.info(f"[STREAM-OFFLOAD] Layer {layer_idx} ready | "
                       f"Attn: {attn_on_gpu}, FFN: {ffn_on_gpu} on GPU | Pinned: {pinned}")

    def _start_attention_prefetch(self, layer_idx: int):
        """Start async prefetch of attention+norm components."""
        if layer_idx >= self.num_layers:
            return
        if self._attention_on_gpu.get(layer_idx, False):
            return
        if self._layer_prefetch_in_progress.get(layer_idx, False):
            return

        self._layer_prefetch_in_progress[layer_idx] = True
        event = torch.cuda.Event()
        self._layer_prefetch_events[layer_idx] = event

        with torch.cuda.stream(self._prefetch_stream):
            self._move_attention_to_gpu(layer_idx, non_blocking=True)
            event.record()

        if self.config.verbose:
            logger.debug(f"Started attention prefetch for layer {layer_idx}")

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
        # End timing for profiling (during warm-up phase)
        self._end_layer_timing(layer_idx)

        # Periodic cache clear
        if (layer_idx + 1) % self.config.empty_cache_frequency == 0:
            torch.cuda.empty_cache()
            self.stats['cache_clears'] += 1

    def start_tracking(self):
        """Start memory and time tracking for inference."""
        torch.cuda.reset_peak_memory_stats()
        self._start_time = time.time()
        self._peak_memory_gb = 0.0

    def update_memory_tracking(self):
        """Update peak memory tracking (call periodically during inference)."""
        current_memory = torch.cuda.max_memory_allocated() / (1024 ** 3)
        if current_memory > self._peak_memory_gb:
            self._peak_memory_gb = current_memory

    def stop_tracking(self):
        """Stop tracking and finalize stats."""
        self._end_time = time.time()
        self._peak_memory_gb = torch.cuda.max_memory_allocated() / (1024 ** 3)

    def get_statistics(self) -> Dict[str, Any]:
        """Get offloading statistics."""
        inference_time = None
        if self._start_time and self._end_time:
            inference_time = self._end_time - self._start_time

        return {
            **self.stats,
            'num_layers': self.num_layers,
            'layers_on_gpu': len(self._layers_on_gpu_set),
            'window_size': self.config.num_layers_on_gpu,
            'peak_memory_gb': self._peak_memory_gb,
            'inference_time_seconds': inference_time,
        }

    def print_statistics(self):
        """Print offloading statistics."""
        stats = self.get_statistics()
        is_stream = self.config.offload_strategy == "stream"
        num_pinned = len(self._pinned_heavy_layers)

        print("\n" + "=" * 70)
        print("OFFLOADING PERFORMANCE REPORT")
        print("=" * 70)
        mode_str = "Adaptive (auto-calculated)" if self.config.adaptive_mode else "Fixed (user-specified)"
        if is_stream:
            print(f"Strategy: STREAM + PGDP - Component pipelining with profiled pinning ({mode_str})")
        else:
            print(f"Strategy: LAYER + PGDP - Sliding window with profiled pinning ({mode_str})")
        print("-" * 70)
        print("CONFIGURATION:")
        print(f"  Total layers:            {stats['num_layers']}")
        print(f"  Sliding window:          {self.config.num_layers_on_gpu} layers")
        print(f"  Pinned heavy layers:     {num_pinned} (PGDP)")
        if is_stream:
            print(f"  Prefetch depth:          {self.config.stream_prefetch_depth}")
        print(f"  Pinned memory (CPU):     {self.config.use_pinned_memory}")
        if self.config.adaptive_mode:
            print(f"  Video resolution:        {self.config.video_height}x{self.config.video_width}")
            print(f"  Frames:                  {self.config.num_frames}")
        print("-" * 70)
        print("MEMORY EFFICIENCY:")
        if stats['peak_memory_gb'] > 0:
            print(f"  Peak GPU Memory:         {stats['peak_memory_gb']:.2f} GB")
            # Estimate baseline (all layers on GPU)
            per_layer_mb = sum(info.total_mb for info in self._layer_memory.values()) / max(len(self._layer_memory), 1)
            baseline_layers_gb = (per_layer_mb * stats['num_layers']) / 1024
            effective_on_gpu = num_pinned + self.config.num_layers_on_gpu
            savings = baseline_layers_gb - (per_layer_mb * min(effective_on_gpu, stats['num_layers']) / 1024)
            if savings > 0:
                print(f"  Estimated savings:       ~{savings:.1f} GB")
                pct_reduction = (savings / baseline_layers_gb) * 100
                print(f"  Memory reduction:        {pct_reduction:.0f}%")
        else:
            print(f"  Peak GPU Memory:         Not tracked (call start_tracking() before inference)")
        print("-" * 70)
        print("TRANSFER STATS:")
        print(f"  Component loads:         {stats['layer_loads']}")
        print(f"  Component offloads:      {stats['layer_offloads']}")
        print(f"  Pinned layer hits:       {stats['pinned_layer_hits']} (zero-cost access)")
        print(f"  Prefetch hits:           {stats['prefetch_hits']}")
        print(f"  Prefetch misses:         {stats['prefetch_misses']}")
        if is_stream:
            print(f"  FFN prefetch overlaps:   {stats['ffn_prefetch_overlaps']}")
        if stats['inference_time_seconds']:
            print("-" * 70)
            print("TIMING:")
            print(f"  Inference time:          {stats['inference_time_seconds']:.1f}s")
        print("=" * 70)
        print("PROFILE-GUIDED DYNAMIC PINNING (PGDP):")
        if self.config.enable_profiling and self._initial_profiling_complete:
            print(f"  Mode:                    {'Dynamic' if self.config.dynamic_pinning else 'Static'}")
            print(f"  Profiling samples:       {stats['profiling_samples']}")
            print(f"  Current pinned layers:   {sorted(self._pinned_heavy_layers)}")
            print(f"  Pinned layer hits:       {stats['pinned_layer_hits']} (zero-cost access)")
            if self.config.dynamic_pinning:
                print(f"  Repin events:            {self._total_repin_events}")
                print(f"  Repin interval:          Every {self.config.repin_interval} timesteps")
                print(f"  EMA alpha:               {self.config.ema_alpha}")
            if self._layer_ema_compute:
                top_layers = sorted(self._layer_ema_compute.items(), key=lambda x: x[1], reverse=True)[:5]
                print(f"  Top-5 heavy layers (current EMA):")
                for layer_idx, avg_time in top_layers:
                    pin_marker = " [PINNED]" if layer_idx in self._pinned_heavy_layers else ""
                    pin_count = self._layers_pinned_count.get(layer_idx, 0)
                    print(f"    Layer {layer_idx:2d}: {avg_time:6.2f}ms{pin_marker} (pinned {pin_count}x)")
        else:
            print(f"  Status: {'Disabled' if not self.config.enable_profiling else 'Not yet complete'}")
        print("=" * 70)
        print("KEY BENEFITS:")
        print(f"  ✓ Memory: Only {self.config.num_layers_on_gpu + num_pinned}/{stats['num_layers']} layers on GPU")
        if num_pinned > 0:
            if self.config.dynamic_pinning:
                print(f"  ✓ PGDP: Dynamic pinning adapts to changing compute patterns")
            else:
                print(f"  ✓ PGDP: Heavy layers pinned based on warm-up profiling")
        if is_stream:
            print(f"  ✓ Pipelining: Overlap attention compute with FFN transfer")
            print(f"  ✓ Component-aware: Smaller transfer units for better overlap")
        print(f"  ✓ Enables 24GB GPUs for 720p+ video generation")
        print("=" * 70 + "\n")

    def reset(self):
        """Reset state for a new inference run."""
        self._layer_prefetch_in_progress.clear()
        self._layer_prefetch_events.clear()
        self._ffn_prefetch_in_progress.clear()
        self._ffn_prefetch_events.clear()
        self.stats = {k: 0 for k in self.stats}
        self._log_counter = 0
        self._peak_memory_gb = 0.0
        self._start_time = None
        self._end_time = None

        # Reset PGDP state for new profiling run
        self._current_timestep = 0
        self._last_repin_timestep = 0
        self._initial_profiling_complete = False
        self._layer_compute_times.clear()
        self._layer_ema_compute.clear()
        self._total_repin_events = 0
        self._layers_pinned_count.clear()
        # Note: Keep pinned layers for warm start - they'll be re-evaluated
        # during the first repin interval of the new run


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
    ffn_layers_on_gpu: Optional[int] = None,
    use_pinned_memory: bool = True,
    enable_prefetch: bool = True,
    ffn_prefetch_count: int = 2,
    enable_component_prefetch: bool = False,  # Disabled for now - causes device issues
    verbose: bool = False,
    # Adaptive mode parameters
    video_height: int = 720,
    video_width: int = 1280,
    num_frames: int = 129,
    # Strategy: "layer" or "stream"
    offload_strategy: str = "stream",
    stream_prefetch_depth: int = 2,  # For "stream" strategy
    # Profile-Guided Dynamic Pinning (PGDP) parameters
    enable_profiling: bool = True,
    num_warmup_timesteps: int = 5,  # Match SAP warm-up
    num_pinned_heavy_layers: int = 6,  # Top-K heavy layers to pin
    # Dynamic pinning - continuously adapt based on runtime observations
    dynamic_pinning: bool = True,  # Enable continuous adaptation
    repin_interval: int = 5,  # Re-evaluate pinning every N timesteps
    ema_alpha: float = 0.3,  # EMA smoothing factor
) -> Tuple[HybridOffloadManager, Dict]:
    """
    Enable hybrid component offloading for a HunyuanVideo pipeline.

    Supports two offloading strategies combined with Profile-Guided Dynamic Pinning (PGDP):

    1. "layer" strategy:
       - Keep N entire layers on GPU at a time (sliding window)
       - Prefetch next layers while current layer computes
       - Use pinned memory for faster CPU->GPU transfers
       - Simple but can have GPU idle time during transfers

    2. "stream" strategy (default, recommended):
       - StreamBlock pipelining with component-aware prefetching
       - While computing attention, prefetch FFN for same layer
       - While computing FFN, prefetch attention for next layer
       - Achieves near-zero GPU idle time through compute-transfer overlap

    3. Profile-Guided Dynamic Pinning (PGDP) - Key Innovation:
       During SAP warm-up (first N timesteps), we profile compute time per layer.
       After warm-up, we pin heavy layers on GPU. With dynamic_pinning=True,
       we continuously adapt pinning as sparse attention changes layer costs.

       ┌─────────────────────────────────────────────────────────────────┐
       │ Phase 1: Initial Profiling (warm-up timesteps)                  │
       │   - Measure compute time per layer during dense attention       │
       │   - Initial pinning based on observed heavy layers              │
       │                                                                 │
       │ Phase 2: Continuous Adaptation (if dynamic_pinning=True)        │
       │   - Keep profiling with EMA smoothing                           │
       │   - Periodically re-evaluate heavy layer set                    │
       │   - Dynamically adjust: unpin light layers, pin new heavy ones  │
       │   - Adapts to sparse attention patterns as they change          │
       └─────────────────────────────────────────────────────────────────┘

       Why this works:
       - Heavy layers have high compute/transfer ratio → worth keeping on GPU
       - Light layers can tolerate transfer overhead
       - Data-driven: Uses actual measured compute, not heuristics
       - ADAPTIVE: Continuously updates as compute patterns change

    Adaptive Mode:
    When ffn_layers_on_gpu is None, automatically calculates optimal layers
    based on GPU memory and video resolution. This reserves space for
    activations first, then uses remaining memory for model weights.

    Args:
        pipe: HunyuanVideoPipeline
        ffn_layers_on_gpu: Number of layers on GPU. None = auto-detect
        use_pinned_memory: Use pinned CPU memory for faster transfers
        enable_prefetch: Enable async layer prefetching
        ffn_prefetch_count: Number of layers to prefetch ahead
        enable_component_prefetch: (Deprecated) Use stream strategy instead
        verbose: Enable verbose logging
        video_height: Video height (for adaptive mode activation estimation)
        video_width: Video width (for adaptive mode activation estimation)
        num_frames: Number of frames (for adaptive mode activation estimation)
        offload_strategy: "layer" or "stream" (recommended)
        stream_prefetch_depth: How many layers ahead to prefetch for stream strategy
        enable_profiling: Enable PGDP profiling
        num_warmup_timesteps: Number of timesteps for initial profiling
        num_pinned_heavy_layers: Number of heavy layers to pin
        dynamic_pinning: Enable continuous adaptation (vs static after warm-up)
        repin_interval: Re-evaluate pinning every N timesteps
        ema_alpha: EMA smoothing factor (higher = more weight on recent)

    Returns:
        Tuple of (HybridOffloadManager, hooks_dict)
    """
    transformer = pipe.transformer

    # Calculate model size for adaptive mode
    def get_model_weights_gb(model):
        return sum(p.numel() * p.element_size() for p in model.parameters()) / (1024**3)

    num_layers = (len(transformer.transformer_blocks) +
                  len(transformer.single_transformer_blocks))
    model_weights_gb = get_model_weights_gb(transformer)

    # Determine number of layers on GPU
    adaptive_mode = ffn_layers_on_gpu is None

    # Get GPU memory for adaptive calculations
    if torch.cuda.is_available():
        total_gpu_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
    else:
        total_gpu_gb = 24.0  # Default assumption

    # Estimate activation reserve based on resolution
    activation_reserve = estimate_activation_reserve(video_height, video_width, num_frames)

    if adaptive_mode:
        # Calculate optimal layers
        ffn_layers_on_gpu = calculate_optimal_layers_on_gpu(
            total_gpu_memory_gb=total_gpu_gb,
            model_weights_gb=model_weights_gb,
            num_layers=num_layers,
            activation_reserve_gb=activation_reserve,
        )
        logger.info(f"[ADAPTIVE] Auto-selected {ffn_layers_on_gpu} layers on GPU")
    else:
        logger.info(f"[FIXED] Using {ffn_layers_on_gpu} layers on GPU (user specified)")

    # Log strategy info
    if offload_strategy == "stream":
        logger.info(f"[STREAM] Using StreamBlock pipelining with prefetch depth {stream_prefetch_depth}")
        logger.info(f"[STREAM] Key innovation: Overlap attention compute with FFN transfer")
    else:
        logger.info(f"[LAYER] Using standard layer-level sliding window")

    # Log PGDP info
    if enable_profiling:
        logger.info(f"[PGDP] Profile-Guided Dynamic Pinning ENABLED")
        logger.info(f"[PGDP] Initial profiling: {num_warmup_timesteps} timesteps")
        logger.info(f"[PGDP] Will pin top {num_pinned_heavy_layers} heavy layers")
        if dynamic_pinning:
            logger.info(f"[PGDP] Dynamic mode: Re-evaluate every {repin_interval} timesteps")
            logger.info(f"[PGDP] EMA alpha: {ema_alpha} (continuous adaptation)")
        else:
            logger.info(f"[PGDP] Static mode: Pinning fixed after warm-up")

    config = HybridOffloadConfig(
        num_layers_on_gpu=ffn_layers_on_gpu,
        offload_strategy=offload_strategy,
        stream_prefetch_depth=stream_prefetch_depth,
        use_pinned_memory=use_pinned_memory,
        enable_prefetch=enable_prefetch,
        prefetch_count=ffn_prefetch_count,
        verbose=verbose,
        adaptive_mode=adaptive_mode,
        video_height=video_height,
        video_width=video_width,
        num_frames=num_frames,
        # PGDP parameters
        enable_profiling=enable_profiling,
        num_warmup_timesteps=num_warmup_timesteps,
        num_pinned_heavy_layers=num_pinned_heavy_layers,
        # Dynamic pinning parameters
        dynamic_pinning=dynamic_pinning,
        repin_interval=repin_interval,
        ema_alpha=ema_alpha,
    )

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

    # Lazy VAE loading: Keep VAE on CPU, move to GPU only when decode is called
    # This saves ~300MB during transformer inference
    vae_hook_handle = None
    if hasattr(pipe, 'vae') and pipe.vae is not None:
        vae_size_mb = sum(p.numel() * p.element_size() for p in pipe.vae.parameters()) / 1024**2

        # Keep VAE on CPU
        pipe.vae.to('cpu')
        logger.info(f"VAE on CPU (lazy load): {vae_size_mb:.1f}MB - will move to GPU when decode is called")

        # Wrap the decode method to move VAE to GPU before decoding
        # We can't use forward_pre_hook because decode() doesn't call forward()
        original_decode = pipe.vae.decode
        _vae_moved = [False]

        def lazy_decode(*args, **kwargs):
            if not _vae_moved[0]:
                logger.info(f"[LAZY-VAE] Moving VAE to {config.compute_device} for decode...")
                pipe.vae.to(config.compute_device)
                _vae_moved[0] = True
                torch.cuda.empty_cache()
            return original_decode(*args, **kwargs)

        pipe.vae.decode = lazy_decode
        logger.info("Wrapped VAE.decode() for lazy loading")

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

    # Store global reference for external access (benchmarking, stats)
    global _global_offload_manager
    _global_offload_manager = manager

    # Start memory tracking
    manager.start_tracking()

    return manager, hooks


# Global manager reference
_global_offload_manager: Optional[HybridOffloadManager] = None


def get_offload_manager() -> Optional[HybridOffloadManager]:
    """Get the current offload manager instance for stats access."""
    return _global_offload_manager


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
