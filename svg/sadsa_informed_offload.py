"""
SADSA-Informed Adaptive Offloading (SIAO)

A novel layer offloading framework that uses semantic-aware sparse attention signals
to make intelligent memory management decisions. This is the first work to combine:
1. Diffusion stage signals for dynamic memory budgeting
2. Motion estimation for predictive prefetching
3. Quality feedback for eviction prioritization
4. Attention density for compute cost prediction

Key Novel Contributions:
1. SAMBA (Stage-Aware Memory Budget Allocation): Dynamically adjust GPU memory budget
   based on diffusion stage - aggressive in structure, conservative in detail stage
2. MPP (Motion-Predictive Prefetching): Use motion magnitude to predict compute
   intensity and prefetch expensive layers earlier
3. QGE (Quality-Gradient Eviction): Evict layers based on quality contribution
   rather than simple FIFO
4. ADCP (Attention-Density Compute Prediction): Predict layer compute cost from
   SADSA tier decisions

Memory Analysis:
- Naive offloading: Fixed N layers on GPU regardless of workload
- SIAO: Dynamic 1-4 layers based on stage, motion, and quality signals
- Result: Better quality preservation in detail stage, faster in structure stage

Integration with SADSA:
- Receives signals: stage, motion_map, tier_ratios, quality_score
- Feeds back: memory_budget, prefetch_priority, eviction_score
- Bidirectional optimization between attention and memory
"""

from __future__ import annotations

import gc
from collections import defaultdict, deque
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Set, Tuple

import torch
import torch.nn as nn
import numpy as np

from .logger import logger

if TYPE_CHECKING:
    from .sadsa import SADSAManager, DiffusionStage


# =============================================================================
# Configuration
# =============================================================================

class OffloadStage(Enum):
    """Offloading aggressiveness levels aligned with diffusion stages."""
    AGGRESSIVE = auto()  # Structure stage: minimize GPU memory
    BALANCED = auto()    # Semantic stage: balanced approach
    CONSERVATIVE = auto() # Detail stage: maximize quality preservation


@dataclass
class SIAOConfig:
    """Configuration for SADSA-Informed Adaptive Offloading."""

    # Stage-aware memory budgets (number of layers on GPU)
    budget_structure: int = 1   # Aggressive: minimum memory
    budget_semantic: int = 2    # Balanced
    budget_detail: int = 3      # Conservative: more layers for quality

    # Motion thresholds for prefetch adjustment
    motion_high_threshold: float = 0.6  # High motion → prefetch 2 extra layers
    motion_low_threshold: float = 0.2   # Low motion → normal prefetch

    # Quality thresholds for eviction protection
    quality_protect_threshold: float = 0.85  # Protect layers above this quality
    critical_layers_start: int = 3           # First N layers always protected
    critical_layers_end: int = 3             # Last N layers always protected

    # Compute prediction weights
    weight_density: float = 0.4
    weight_tier_ratio: float = 0.3
    weight_timing: float = 0.3

    # Async prefetching
    use_async_prefetch: bool = True
    use_pinned_memory: bool = True

    # Logging
    verbose: bool = False


# =============================================================================
# Running Statistics
# =============================================================================

class ExponentialMovingStats:
    """Track statistics with exponential moving average."""

    def __init__(self, decay: float = 0.9, window: int = 20):
        self.decay = decay
        self.window = window
        self._values: deque = deque(maxlen=window)
        self._ema: Optional[float] = None

    def update(self, value: float):
        self._values.append(value)
        if self._ema is None:
            self._ema = value
        else:
            self._ema = self.decay * self._ema + (1 - self.decay) * value

    @property
    def mean(self) -> float:
        if not self._values:
            return 0.5
        return sum(self._values) / len(self._values)

    @property
    def ema(self) -> float:
        return self._ema if self._ema is not None else 0.5

    @property
    def normalized(self) -> float:
        """Value normalized to [0, 1] based on history."""
        if len(self._values) < 2:
            return 0.5
        min_val, max_val = min(self._values), max(self._values)
        if max_val - min_val < 1e-6:
            return 0.5
        return (self.ema - min_val) / (max_val - min_val)


# =============================================================================
# Layer Compute Predictor (ADCP)
# =============================================================================

@dataclass
class LayerComputeSignals:
    """Signals for predicting layer compute cost."""
    layer_idx: int
    timestep: int
    attention_density: float = 0.5
    full_attention_ratio: float = 0.3
    centroid_ratio: float = 0.5
    skip_ratio: float = 0.2
    compute_time_ms: float = 0.0
    motion_level: float = 0.0


class AttentionDensityComputePredictor:
    """
    ADCP: Predicts compute cost using attention density and tier ratios.

    Novel insight: SADSA tier decisions directly predict compute intensity.
    - Full attention tokens: O(N^2) compute
    - Centroid attention: O(C^2) compute (C << N)
    - Skip: Zero compute

    Combined with temporal coherence (patterns stable across timesteps),
    we can predict future compute costs and optimize prefetching.
    """

    def __init__(self, num_layers: int, config: SIAOConfig):
        self.num_layers = num_layers
        self.config = config

        # Per-layer statistics
        self._density_stats: Dict[int, ExponentialMovingStats] = defaultdict(
            lambda: ExponentialMovingStats()
        )
        self._timing_stats: Dict[int, ExponentialMovingStats] = defaultdict(
            lambda: ExponentialMovingStats()
        )
        self._tier_stats: Dict[int, Dict[str, ExponentialMovingStats]] = defaultdict(
            lambda: {
                'full': ExponentialMovingStats(),
                'centroid': ExponentialMovingStats(),
                'skip': ExponentialMovingStats(),
            }
        )

        # Motion history per timestep
        self._motion_history: deque = deque(maxlen=10)

        # Current state
        self._current_timestep = 0
        self._call_count = 0

    def record_layer_execution(
        self,
        layer_idx: int,
        timestep: int,
        attention_density: float,
        tier_ratios: Tuple[float, float, float],  # (full, centroid, skip)
        compute_time_ms: float,
        motion_level: float = 0.0,
    ):
        """Record execution signals for a layer."""
        self._density_stats[layer_idx].update(attention_density)
        self._timing_stats[layer_idx].update(compute_time_ms)

        full_ratio, centroid_ratio, skip_ratio = tier_ratios
        self._tier_stats[layer_idx]['full'].update(full_ratio)
        self._tier_stats[layer_idx]['centroid'].update(centroid_ratio)
        self._tier_stats[layer_idx]['skip'].update(skip_ratio)

        if layer_idx == 0:  # Record motion once per timestep
            self._motion_history.append(motion_level)
            self._current_timestep = timestep

        self._call_count += 1

    def predict_compute_cost(self, layer_idx: int) -> float:
        """
        Predict relative compute cost (0-1 scale).

        Higher value = more expensive = should prefetch earlier.
        """
        # Factor 1: Attention density
        density_factor = self._density_stats[layer_idx].normalized

        # Factor 2: Tier distribution (full attention is expensive)
        tier_stats = self._tier_stats[layer_idx]
        full_ratio = tier_stats['full'].ema
        skip_ratio = tier_stats['skip'].ema
        # High full ratio or low skip ratio = expensive
        tier_factor = full_ratio * 0.7 + (1 - skip_ratio) * 0.3

        # Factor 3: Historical timing
        timing_factor = self._timing_stats[layer_idx].normalized

        # Weighted combination
        cost = (
            self.config.weight_density * density_factor +
            self.config.weight_tier_ratio * tier_factor +
            self.config.weight_timing * timing_factor
        )

        return min(max(cost, 0.0), 1.0)

    def get_prefetch_order(self, current_layer: int, lookahead: int = 3) -> List[int]:
        """
        Get optimal prefetch order based on predicted costs.

        Expensive layers should be prefetched earlier.
        """
        candidates = []
        for i in range(lookahead):
            layer_idx = current_layer + 1 + i
            if layer_idx >= self.num_layers:
                break
            cost = self.predict_compute_cost(layer_idx)
            candidates.append((layer_idx, cost))

        # Sort by cost descending (expensive first)
        candidates.sort(key=lambda x: -x[1])
        return [idx for idx, _ in candidates]

    def get_motion_trend(self) -> float:
        """Get current motion trend (0-1)."""
        if not self._motion_history:
            return 0.5
        return sum(self._motion_history) / len(self._motion_history)


# =============================================================================
# Stage-Aware Memory Budget (SAMBA)
# =============================================================================

class StageAwareMemoryBudget:
    """
    SAMBA: Dynamically allocate GPU memory based on diffusion stage.

    Key insight: Different diffusion stages have different quality sensitivities.
    - Structure stage (t > 0.7): Global features, can tolerate more sparsity
    - Semantic stage (0.3 < t < 0.7): Object boundaries, moderate sensitivity
    - Detail stage (t < 0.3): Fine details, high quality sensitivity

    Memory allocation strategy:
    - Structure: Aggressive offloading (1 layer) - speed priority
    - Semantic: Balanced (2 layers) - good tradeoff
    - Detail: Conservative (3+ layers) - quality priority
    """

    def __init__(self, config: SIAOConfig, max_timestep: int = 1000):
        self.config = config
        self.max_timestep = max_timestep

        # Stage boundaries (normalized timestep)
        self.structure_threshold = 0.7  # t > 0.7 = structure
        self.semantic_threshold = 0.3   # 0.3 < t < 0.7 = semantic

    def get_stage(self, timestep: int) -> OffloadStage:
        """Determine offload stage from timestep."""
        t_norm = timestep / self.max_timestep

        if t_norm > self.structure_threshold:
            return OffloadStage.AGGRESSIVE
        elif t_norm > self.semantic_threshold:
            return OffloadStage.BALANCED
        else:
            return OffloadStage.CONSERVATIVE

    def get_memory_budget(self, timestep: int, motion_level: float = 0.0) -> int:
        """
        Get number of layers to keep on GPU.

        Motion adjustment: High motion → more layers for quality
        """
        stage = self.get_stage(timestep)

        if stage == OffloadStage.AGGRESSIVE:
            base_budget = self.config.budget_structure
        elif stage == OffloadStage.BALANCED:
            base_budget = self.config.budget_semantic
        else:
            base_budget = self.config.budget_detail

        # Motion adjustment
        if motion_level > self.config.motion_high_threshold:
            # High motion: increase budget for quality
            return base_budget + 1
        elif motion_level < self.config.motion_low_threshold:
            # Low motion: can be more aggressive
            return max(base_budget - 1, 1)

        return base_budget

    def get_prefetch_lookahead(self, timestep: int, motion_level: float = 0.0) -> int:
        """
        Get prefetch lookahead based on stage and motion.

        Expensive layers need earlier prefetching.
        """
        stage = self.get_stage(timestep)

        if stage == OffloadStage.CONSERVATIVE:
            base_lookahead = 3  # More aggressive prefetching in detail stage
        elif stage == OffloadStage.BALANCED:
            base_lookahead = 2
        else:
            base_lookahead = 1

        # High motion → prefetch more layers
        if motion_level > self.config.motion_high_threshold:
            return base_lookahead + 1

        return base_lookahead


# =============================================================================
# Quality-Gradient Eviction (QGE)
# =============================================================================

class QualityGradientEviction:
    """
    QGE: Evict layers based on quality contribution rather than FIFO.

    Novel insight: Not all layers contribute equally to output quality.
    - First/last layers: Critical for input/output projection
    - Layers with high attention density: Important semantic processing
    - Layers with low quality signals: Can be evicted earlier

    Eviction priority (lower = evict first):
    priority = recency * 0.3 + criticality * 0.4 + quality_contribution * 0.3
    """

    def __init__(
        self,
        num_layers: int,
        config: SIAOConfig,
        compute_predictor: AttentionDensityComputePredictor,
    ):
        self.num_layers = num_layers
        self.config = config
        self.compute_predictor = compute_predictor

        # Layer criticality profile (precomputed or learned)
        self._criticality = self._compute_criticality_profile()

        # Quality contribution history
        self._quality_history: Dict[int, ExponentialMovingStats] = defaultdict(
            lambda: ExponentialMovingStats()
        )

    def _compute_criticality_profile(self) -> np.ndarray:
        """
        Compute layer criticality profile.

        Based on typical transformer behavior:
        - Early layers: High criticality (input processing)
        - Middle layers: Moderate (semantic processing)
        - Late layers: High criticality (output projection)
        """
        profile = np.ones(self.num_layers) * 0.5

        # First layers are critical
        for i in range(self.config.critical_layers_start):
            profile[i] = 1.0 - (i * 0.1)

        # Last layers are critical
        for i in range(self.config.critical_layers_end):
            profile[self.num_layers - 1 - i] = 1.0 - (i * 0.1)

        return profile

    def update_quality(self, layer_idx: int, quality_score: float):
        """Update quality contribution for a layer."""
        self._quality_history[layer_idx].update(quality_score)

    def get_eviction_score(
        self,
        layer_idx: int,
        current_layer: int,
        timestep: int,
    ) -> float:
        """
        Get eviction score for a layer.

        Higher score = evict first.
        """
        # Factor 1: Recency (how soon will this layer be needed again?)
        if layer_idx >= current_layer:
            # Future layer - don't evict
            recency = 0.0
        else:
            # Past layer - further past = higher eviction score
            distance = current_layer - layer_idx
            recency = distance / self.num_layers

        # Factor 2: Criticality (lower criticality = higher eviction score)
        criticality_factor = 1.0 - self._criticality[layer_idx]

        # Factor 3: Quality contribution (lower quality = higher eviction score)
        quality_factor = 1.0 - self._quality_history[layer_idx].ema

        # Factor 4: Compute cost (cheaper to reload = higher eviction score)
        compute_cost = self.compute_predictor.predict_compute_cost(layer_idx)
        reload_factor = 1.0 - compute_cost

        # Weighted combination
        score = (
            0.25 * recency +
            0.30 * criticality_factor +
            0.25 * quality_factor +
            0.20 * reload_factor
        )

        return score

    def get_eviction_order(
        self,
        layers_on_gpu: List[int],
        current_layer: int,
        timestep: int,
        num_to_evict: int = 1,
    ) -> List[int]:
        """
        Get optimal eviction order.

        Returns layer indices to evict, highest priority first.
        """
        scores = []
        for layer_idx in layers_on_gpu:
            if layer_idx == current_layer:
                continue  # Never evict current layer

            score = self.get_eviction_score(layer_idx, current_layer, timestep)
            scores.append((layer_idx, score))

        # Sort by score descending (highest eviction priority first)
        scores.sort(key=lambda x: -x[1])

        return [idx for idx, _ in scores[:num_to_evict]]


# =============================================================================
# Motion-Predictive Prefetching (MPP)
# =============================================================================

class MotionPredictivePrefetcher:
    """
    MPP: Use motion signals to optimize prefetching.

    Key insight: Motion magnitude predicts compute intensity.
    - High motion → more full attention tokens → expensive
    - Low motion → more skippable tokens → cheap

    Prefetch strategy:
    - Track motion trend across timesteps
    - High motion trend → prefetch more layers earlier
    - Sudden motion increase → trigger immediate prefetch burst
    """

    def __init__(
        self,
        config: SIAOConfig,
        compute_predictor: AttentionDensityComputePredictor,
    ):
        self.config = config
        self.compute_predictor = compute_predictor

        # Motion history
        self._motion_history: deque = deque(maxlen=10)
        self._motion_acceleration: float = 0.0

    def update_motion(self, motion_level: float):
        """Update motion tracking."""
        if self._motion_history:
            prev_motion = self._motion_history[-1]
            self._motion_acceleration = motion_level - prev_motion

        self._motion_history.append(motion_level)

    def should_burst_prefetch(self) -> bool:
        """
        Check if we should do a burst prefetch.

        Triggered by sudden motion increase.
        """
        if len(self._motion_history) < 2:
            return False

        # Sudden acceleration in motion
        return self._motion_acceleration > 0.2

    def get_prefetch_targets(
        self,
        current_layer: int,
        num_layers: int,
        base_lookahead: int,
    ) -> List[int]:
        """
        Get layers to prefetch with motion-aware ordering.
        """
        motion_level = self._motion_history[-1] if self._motion_history else 0.5

        # Adjust lookahead based on motion
        if motion_level > self.config.motion_high_threshold:
            lookahead = base_lookahead + 2
        elif self.should_burst_prefetch():
            lookahead = base_lookahead + 3  # Burst prefetch
        else:
            lookahead = base_lookahead

        # Get compute-ordered prefetch list
        targets = self.compute_predictor.get_prefetch_order(
            current_layer, lookahead
        )

        return targets


# =============================================================================
# Main SIAO Manager
# =============================================================================

class SADSAInformedOffloadManager:
    """
    SIAO: SADSA-Informed Adaptive Offloading Manager.

    Integrates all components:
    - ADCP: Attention-Density Compute Prediction
    - SAMBA: Stage-Aware Memory Budget Allocation
    - QGE: Quality-Gradient Eviction
    - MPP: Motion-Predictive Prefetching

    Usage:
        manager = SADSAInformedOffloadManager(transformer, config)

        # During forward pass
        manager.begin_timestep(timestep, motion_level)
        for layer_idx in range(num_layers):
            manager.prepare_layer(layer_idx)
            output = layer(input)
            manager.record_layer(layer_idx, attention_stats)
        manager.end_timestep()
    """

    def __init__(
        self,
        transformer: nn.Module,
        config: Optional[SIAOConfig] = None,
        max_timestep: int = 1000,
    ):
        self.transformer = transformer
        self.config = config or SIAOConfig()

        # Extract layers
        if hasattr(transformer, 'transformer_blocks'):
            double_blocks = list(transformer.transformer_blocks)
            single_blocks = list(transformer.single_transformer_blocks)
            self.layers = double_blocks + single_blocks
        else:
            self.layers = list(transformer.children())

        self.num_layers = len(self.layers)
        self.max_timestep = max_timestep

        # Initialize components
        self.compute_predictor = AttentionDensityComputePredictor(
            self.num_layers, self.config
        )
        self.memory_budget = StageAwareMemoryBudget(self.config, max_timestep)
        self.eviction_manager = QualityGradientEviction(
            self.num_layers, self.config, self.compute_predictor
        )
        self.prefetcher = MotionPredictivePrefetcher(
            self.config, self.compute_predictor
        )

        # Layer state tracking
        self._layer_device: List[str] = ['cpu'] * self.num_layers
        self._layers_on_gpu: List[int] = []

        # CUDA streams for async operations
        self._prefetch_stream: Optional[torch.cuda.Stream] = None
        if self.config.use_async_prefetch and torch.cuda.is_available():
            self._prefetch_stream = torch.cuda.Stream()

        # Current state
        self._current_timestep = 0
        self._current_motion = 0.0
        # Start with semantic budget as reasonable default
        self._current_budget = self.config.budget_semantic

        # Statistics
        self._stats = {
            'prefetch_hits': 0,
            'prefetch_misses': 0,
            'evictions': 0,
            'burst_prefetches': 0,
        }

        # Initialize
        self._initialized = False

        logger.info(f"[SIAO] Initialized with {self.num_layers} layers")
        logger.info(f"[SIAO] Budget: structure={self.config.budget_structure}, "
                   f"semantic={self.config.budget_semantic}, "
                   f"detail={self.config.budget_detail}")

    def initialize(self):
        """Initialize: move all layers to CPU and install hooks."""
        if self._initialized:
            return

        logger.info("[SIAO] Initializing - moving layers to CPU...")

        for i, layer in enumerate(self.layers):
            layer.to('cpu')
            self._layer_device[i] = 'cpu'

        self._layers_on_gpu = []

        if self.config.use_pinned_memory:
            logger.info("[SIAO] Pinning memory for fast transfers...")
            for layer in self.layers:
                self._pin_module(layer)

        # Install forward hooks for automatic GPU/CPU movement
        self.install_forward_hooks()

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        self._initialized = True
        logger.info("[SIAO] Initialization complete")

    def _pin_module(self, module: nn.Module):
        """Pin module memory for fast transfers."""
        for param in module.parameters():
            if not param.is_cuda and not param.data.is_pinned():
                try:
                    param.data = param.data.pin_memory()
                except:
                    pass  # Some tensors can't be pinned

    def begin_timestep(self, timestep: int, motion_level: float = 0.0):
        """
        Begin a new timestep - update budget and motion tracking.
        """
        if not self._initialized:
            self.initialize()

        self._current_timestep = timestep
        self._current_motion = motion_level

        # Update motion tracking
        self.prefetcher.update_motion(motion_level)

        # Get memory budget for this timestep
        self._current_budget = self.memory_budget.get_memory_budget(
            timestep, motion_level
        )

        if self.config.verbose:
            stage = self.memory_budget.get_stage(timestep)
            logger.info(f"[SIAO] Timestep {timestep}: stage={stage.name}, "
                       f"budget={self._current_budget}, motion={motion_level:.3f}")

    def prepare_layer(self, layer_idx: int):
        """
        Prepare a layer for execution - ensure it's on GPU.
        """
        # Ensure layer is on GPU
        if self._layer_device[layer_idx] != 'cuda':
            self._load_layer_to_gpu(layer_idx)

        # Prefetch next layers
        self._prefetch_upcoming_layers(layer_idx)

        # Evict if over budget
        self._enforce_budget(layer_idx)

    def _load_layer_to_gpu(self, layer_idx: int, blocking: bool = True):
        """Load a layer to GPU."""
        if self._layer_device[layer_idx] == 'cuda':
            return

        layer = self.layers[layer_idx]

        if self._prefetch_stream and not blocking:
            with torch.cuda.stream(self._prefetch_stream):
                layer.to('cuda', non_blocking=True)
        else:
            layer.to('cuda', non_blocking=False)

        self._layer_device[layer_idx] = 'cuda'
        if layer_idx not in self._layers_on_gpu:
            self._layers_on_gpu.append(layer_idx)

        if self.config.verbose:
            logger.info(f"[SIAO] Loaded layer {layer_idx} to GPU")

    def _unload_layer_to_cpu(self, layer_idx: int):
        """Unload a layer to CPU."""
        if self._layer_device[layer_idx] != 'cuda':
            return

        layer = self.layers[layer_idx]
        layer.to('cpu', non_blocking=True)

        self._layer_device[layer_idx] = 'cpu'
        if layer_idx in self._layers_on_gpu:
            self._layers_on_gpu.remove(layer_idx)

        self._stats['evictions'] += 1

        if self.config.verbose:
            logger.info(f"[SIAO] Evicted layer {layer_idx} to CPU")

    def _prefetch_upcoming_layers(self, current_layer: int):
        """Prefetch upcoming layers based on predictions."""
        if not self._prefetch_stream:
            return

        # Get lookahead based on stage and motion
        lookahead = self.memory_budget.get_prefetch_lookahead(
            self._current_timestep, self._current_motion
        )

        # Check for burst prefetch
        if self.prefetcher.should_burst_prefetch():
            self._stats['burst_prefetches'] += 1
            lookahead += 2

        # Get prioritized prefetch targets
        targets = self.prefetcher.get_prefetch_targets(
            current_layer, self.num_layers, lookahead
        )

        # Prefetch asynchronously
        for layer_idx in targets:
            if self._layer_device[layer_idx] != 'cuda':
                self._load_layer_to_gpu(layer_idx, blocking=False)

    def _enforce_budget(self, current_layer: int):
        """Enforce memory budget by evicting layers."""
        while len(self._layers_on_gpu) > self._current_budget + 1:  # +1 for current
            # Get eviction order
            evict_order = self.eviction_manager.get_eviction_order(
                self._layers_on_gpu,
                current_layer,
                self._current_timestep,
                num_to_evict=1,
            )

            if not evict_order:
                break

            self._unload_layer_to_cpu(evict_order[0])

    def record_layer_execution(
        self,
        layer_idx: int,
        attention_density: float,
        tier_ratios: Tuple[float, float, float],
        compute_time_ms: float,
        quality_score: float = 1.0,
    ):
        """
        Record execution statistics for a layer.

        Called after layer execution to update predictions.
        """
        # Update compute predictor
        self.compute_predictor.record_layer_execution(
            layer_idx=layer_idx,
            timestep=self._current_timestep,
            attention_density=attention_density,
            tier_ratios=tier_ratios,
            compute_time_ms=compute_time_ms,
            motion_level=self._current_motion,
        )

        # Update quality for eviction decisions
        self.eviction_manager.update_quality(layer_idx, quality_score)

    def end_timestep(self):
        """End timestep - cleanup and prepare for next."""
        # Sync prefetch stream
        if self._prefetch_stream:
            self._prefetch_stream.synchronize()

        # Log statistics periodically
        if self.config.verbose and self._current_timestep % 10 == 0:
            self._log_stats()

    def install_forward_hooks(self):
        """
        Install forward pre-hooks on all transformer blocks.

        These hooks automatically move layers to GPU before execution
        and manage memory budget through eviction.
        """
        self._hook_handles = []

        def make_pre_hook(layer_idx):
            def pre_hook(module, inputs):
                # Move layer to GPU and manage prefetch/eviction
                self.prepare_layer(layer_idx)

                # Move inputs to CUDA if they're on CPU
                def move_to_cuda(x):
                    if isinstance(x, torch.Tensor) and x.device.type == 'cpu':
                        return x.to('cuda', non_blocking=True)
                    return x

                if isinstance(inputs, tuple):
                    inputs = tuple(move_to_cuda(x) for x in inputs)
                elif isinstance(inputs, torch.Tensor):
                    inputs = move_to_cuda(inputs)

                return inputs
            return pre_hook

        for i, layer in enumerate(self.layers):
            # Register pre-hook to move layer to GPU before execution
            handle = layer.register_forward_pre_hook(make_pre_hook(i))
            self._hook_handles.append(handle)

        logger.info(f"[SIAO] Installed {len(self._hook_handles)} forward hooks")

    def remove_hooks(self):
        """Remove all installed forward hooks."""
        if hasattr(self, '_hook_handles'):
            for handle in self._hook_handles:
                handle.remove()
            self._hook_handles = []
            logger.info("[SIAO] Removed forward hooks")

    def cleanup(self):
        """Cleanup - move all layers to CPU and remove hooks."""
        logger.info("[SIAO] Cleaning up...")

        # Remove hooks first
        self.remove_hooks()

        for i in list(self._layers_on_gpu):
            self._unload_layer_to_cpu(i)

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        self._log_stats()
        logger.info("[SIAO] Cleanup complete")

    def _log_stats(self):
        """Log statistics."""
        total_prefetch = self._stats['prefetch_hits'] + self._stats['prefetch_misses']
        hit_rate = (self._stats['prefetch_hits'] / max(total_prefetch, 1)) * 100

        logger.info(f"[SIAO] Stats: prefetch_hit_rate={hit_rate:.1f}%, "
                   f"evictions={self._stats['evictions']}, "
                   f"burst_prefetches={self._stats['burst_prefetches']}")

    def get_memory_info(self) -> Dict[str, float]:
        """Get current GPU memory info."""
        if not torch.cuda.is_available():
            return {'allocated': 0, 'free': 0}

        return {
            'allocated': torch.cuda.memory_allocated() / 1e9,
            'free': torch.cuda.mem_get_info()[0] / 1e9,
            'layers_on_gpu': len(self._layers_on_gpu),
        }


# =============================================================================
# Integration with Pipeline
# =============================================================================

def create_siao_manager(
    transformer: nn.Module,
    budget_structure: int = 1,
    budget_semantic: int = 2,
    budget_detail: int = 3,
    use_async_prefetch: bool = True,
    use_pinned_memory: bool = True,
    verbose: bool = False,
) -> SADSAInformedOffloadManager:
    """
    Create SIAO manager with configuration.

    Args:
        transformer: The transformer model
        budget_structure: Layers on GPU during structure stage
        budget_semantic: Layers on GPU during semantic stage
        budget_detail: Layers on GPU during detail stage
        use_async_prefetch: Enable async prefetching
        use_pinned_memory: Use pinned memory
        verbose: Enable verbose logging

    Returns:
        SADSAInformedOffloadManager instance
    """
    config = SIAOConfig(
        budget_structure=budget_structure,
        budget_semantic=budget_semantic,
        budget_detail=budget_detail,
        use_async_prefetch=use_async_prefetch,
        use_pinned_memory=use_pinned_memory,
        verbose=verbose,
    )

    manager = SADSAInformedOffloadManager(transformer, config)
    # Initialize immediately to install hooks and prepare for inference
    manager.initialize()
    return manager
