"""
Compute-Memory Co-Adaptation (CMCA) Framework: Compute Cost Model

This module provides a principled cost model for video diffusion inference that
exploits temporal coherence across timesteps. The key insight is that sparse
attention patterns (from SAP/CTCA/CTAA) directly predict layer compute costs,
enabling intelligent memory management decisions.

Theoretical Foundation:
-----------------------
In video diffusion, latent features z_t evolve slowly:
    ||z_t - z_{t-1}||₂ / ||z_t||₂ ≈ 0.01-0.05

This temporal coherence manifests in three predictable ways:
1. Cluster Stability: P(cluster(token,t) = cluster(token,t-1)) > 0.95
2. Attention Stability: ||A_t - A_{t-1}||_F / ||A_t||_F < 0.1  
3. Compute Stability: |cost(L,t) - cost(L,t-1)| / cost(L,t) < 0.2

Cost Model:
-----------
Total layer cost decomposes as:

    Cost(l, t) = α·C_attn(l,t) + β·C_cluster(l,t) + γ·C_transfer(l,t)

Where:
- C_attn: Attention cost (function of density and tier distribution)
- C_cluster: Clustering cost (function of recluster vs. reuse decision)
- C_transfer: Transfer cost (function of layer size and bandwidth overlap)

This model enables:
- Predictive prefetching: Start loading expensive layers earlier
- Optimal pinning: Pin layers that would cause stalls if transferred
- Adaptive scheduling: Allocate memory budget based on predicted costs
"""

from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Set, Tuple

import torch
import torch.nn as nn

from .logger import logger

if TYPE_CHECKING:
    from .ctca import CrossTimestepClusterAmortization


# =============================================================================
# Running Statistics with Exponential Moving Average
# =============================================================================

class RunningEMA:
    """
    Maintains exponential moving average with windowed history for normalization.
    
    The EMA provides smooth estimates that adapt to changing conditions while
    the history window enables z-score normalization for cross-layer comparison.
    """
    
    def __init__(self, alpha: float = 0.3, window: int = 50):
        """
        Args:
            alpha: EMA smoothing factor. Higher = more weight on recent observations.
                   Default 0.3 balances responsiveness with stability.
            window: History window size for min/max normalization.
        """
        self.alpha = alpha
        self.window = window
        self._history: deque = deque(maxlen=window)
        self._ema: Optional[float] = None
        self._count: int = 0
    
    def update(self, value: float) -> float:
        """Update with new observation and return current EMA."""
        self._history.append(value)
        self._count += 1
        
        if self._ema is None:
            self._ema = value
        else:
            self._ema = self.alpha * value + (1 - self.alpha) * self._ema
        
        return self._ema
    
    @property
    def value(self) -> float:
        """Current EMA value."""
        return self._ema if self._ema is not None else 0.5
    
    @property
    def count(self) -> int:
        """Number of observations."""
        return self._count
    
    def get_normalized(self) -> float:
        """
        Return value normalized to [0, 1] based on observed min/max.
        Returns 0.5 if insufficient history.
        """
        if len(self._history) < 2:
            return 0.5
        
        min_val = min(self._history)
        max_val = max(self._history)
        
        if max_val - min_val < 1e-6:
            return 0.5
        
        return (self.value - min_val) / (max_val - min_val)
    
    def get_percentile_rank(self) -> float:
        """
        Return the percentile rank of current EMA within history.
        Useful for identifying heavy/light layers relative to distribution.
        """
        if len(self._history) < 2 or self._ema is None:
            return 0.5
        
        below_count = sum(1 for v in self._history if v < self._ema)
        return below_count / len(self._history)


# =============================================================================
# Layer Cost Signal Collection
# =============================================================================

@dataclass
class LayerCostSignals:
    """
    Signals collected from a layer that inform compute cost prediction.
    
    These signals come from different sources:
    - Attention density: From sparse attention mask analysis
    - CTCA decision: From cluster amortization module
    - CTAA tiers: From hierarchical attention tier assignment
    - Timing: From CUDA event profiling
    """
    layer_idx: int
    timestep: int
    
    # Attention pattern signals
    attention_density: float = 0.5  # Fraction of attended token pairs [0, 1]
    
    # CTCA signals
    ctca_reclustered: bool = False  # True if full K-means ran (expensive)
    ctca_quality: float = 1.0       # Cluster quality metric
    
    # CTAA tier distribution (should sum to 1.0)
    tier_full_ratio: float = 0.3     # Fraction getting full attention
    tier_centroid_ratio: float = 0.5 # Fraction getting centroid attention
    tier_skip_ratio: float = 0.2     # Fraction skipped
    
    # Measured timing (ms)
    compute_time_ms: float = 0.0
    
    def __post_init__(self):
        """Validate signal ranges."""
        self.attention_density = max(0.0, min(1.0, self.attention_density))
        self.tier_full_ratio = max(0.0, min(1.0, self.tier_full_ratio))


# =============================================================================
# Compute Cost Predictor
# =============================================================================

@dataclass
class CostModelConfig:
    """Configuration for the compute cost model."""
    
    # Weight factors for cost components (should sum to 1.0)
    weight_attention: float = 0.40   # Weight for attention density signal
    weight_clustering: float = 0.30  # Weight for CTCA decision signal
    weight_timing: float = 0.30      # Weight for historical timing signal
    
    # EMA parameters
    ema_alpha: float = 0.3           # Smoothing factor for EMA updates
    history_window: int = 50         # Window size for normalization
    
    # Cost scaling factors (relative costs)
    recluster_cost_factor: float = 1.0   # Full K-means = high cost
    reuse_cost_factor: float = 0.05      # Centroid update only = low cost
    
    # Tier cost factors (attention type relative costs)
    full_attention_cost: float = 1.0
    centroid_attention_cost: float = 0.1
    skip_attention_cost: float = 0.0
    
    # Transfer time estimation (empirical, for 24GB GPU)
    layer_transfer_ms: float = 50.0  # Typical layer CPU->GPU transfer time
    
    def __post_init__(self):
        """Normalize weights."""
        total = self.weight_attention + self.weight_clustering + self.weight_timing
        if abs(total - 1.0) > 0.01:
            self.weight_attention /= total
            self.weight_clustering /= total
            self.weight_timing /= total


class ComputeCostPredictor:
    """
    Predicts layer compute costs using attention pattern signals.
    
    This predictor exploits temporal coherence in video diffusion:
    - Patterns at timestep t predict patterns at timestep t+1
    - Combine multiple signals for robust prediction
    - Normalize across layers for fair comparison
    
    The prediction is used by the offload manager to:
    1. Prioritize prefetching for expensive layers
    2. Pin heavy layers on GPU (they'd cause stalls otherwise)
    3. Evict light layers first (they're cheap to reload)
    
    Mathematical Model:
    ------------------
    Predicted cost combines three normalized signals:
    
        cost(l, t) = α · density(l) + β · cluster_cost(l) + γ · timing(l)
    
    Where:
    - density(l): Normalized attention density (higher = more compute)
    - cluster_cost(l): Expected clustering cost (recluster = 1.0, reuse = 0.05)
    - timing(l): Normalized historical timing
    
    All signals are normalized to [0, 1] for fair combination.
    """
    
    def __init__(
        self,
        num_layers: int,
        config: Optional[CostModelConfig] = None,
    ):
        """
        Initialize the cost predictor.
        
        Args:
            num_layers: Total number of transformer layers
            config: Cost model configuration
        """
        self.num_layers = num_layers
        self.config = config or CostModelConfig()
        
        # Per-layer signal tracking with EMA
        self._density_ema: Dict[int, RunningEMA] = {}
        self._timing_ema: Dict[int, RunningEMA] = {}
        
        # CTCA state tracking (for predicting recluster decisions)
        self._calls_since_recluster: Dict[int, int] = defaultdict(int)
        self._last_quality: Dict[int, float] = defaultdict(lambda: 1.0)
        
        # Global statistics for cross-layer normalization
        self._global_density = RunningEMA(alpha=self.config.ema_alpha)
        self._global_timing = RunningEMA(alpha=self.config.ema_alpha)
        
        # CTCA manager reference (set externally)
        self.ctca_manager: Optional[CrossTimestepClusterAmortization] = None
        
        # Current timestep tracking
        self._current_timestep: int = 0
        self._total_samples: int = 0
        
        # Initialize EMA trackers for all layers
        for layer_idx in range(num_layers):
            self._density_ema[layer_idx] = RunningEMA(
                alpha=self.config.ema_alpha,
                window=self.config.history_window
            )
            self._timing_ema[layer_idx] = RunningEMA(
                alpha=self.config.ema_alpha,
                window=self.config.history_window
            )
    
    def set_ctca_manager(self, ctca_manager: CrossTimestepClusterAmortization):
        """Set CTCA manager for accessing cluster quality signals."""
        self.ctca_manager = ctca_manager
    
    def on_timestep_start(self, timestep: int):
        """Called at the start of each diffusion timestep."""
        self._current_timestep = timestep
    
    def record_signals(self, signals: LayerCostSignals):
        """
        Record observed signals after layer execution.
        
        This updates our statistical models for future predictions.
        
        Args:
            signals: Collected signals from layer execution
        """
        layer_idx = signals.layer_idx
        
        # Update density EMA
        self._density_ema[layer_idx].update(signals.attention_density)
        self._global_density.update(signals.attention_density)
        
        # Update timing EMA
        if signals.compute_time_ms > 0:
            self._timing_ema[layer_idx].update(signals.compute_time_ms)
            self._global_timing.update(signals.compute_time_ms)
        
        # Update CTCA state
        if signals.ctca_reclustered:
            self._calls_since_recluster[layer_idx] = 0
        else:
            self._calls_since_recluster[layer_idx] += 1
        
        self._last_quality[layer_idx] = signals.ctca_quality
        self._total_samples += 1
    
    def record_from_attention(
        self,
        layer_idx: int,
        timestep: int,
        attention_density: float,
        compute_time_ms: float,
        ctca_reclustered: bool = False,
        ctca_quality: float = 1.0,
        tier_ratios: Optional[Tuple[float, float, float]] = None,
    ):
        """
        Convenience method to record signals from attention processor.
        
        Args:
            layer_idx: Layer index
            timestep: Current diffusion timestep
            attention_density: Fraction of attended token pairs
            compute_time_ms: Measured compute time in milliseconds
            ctca_reclustered: Whether full K-means ran
            ctca_quality: Cluster quality score
            tier_ratios: Optional (full, centroid, skip) ratios from CTAA
        """
        if tier_ratios is not None:
            full_ratio, centroid_ratio, skip_ratio = tier_ratios
        else:
            full_ratio, centroid_ratio, skip_ratio = 0.3, 0.5, 0.2
        
        signals = LayerCostSignals(
            layer_idx=layer_idx,
            timestep=timestep,
            attention_density=attention_density,
            ctca_reclustered=ctca_reclustered,
            ctca_quality=ctca_quality,
            tier_full_ratio=full_ratio,
            tier_centroid_ratio=centroid_ratio,
            tier_skip_ratio=skip_ratio,
            compute_time_ms=compute_time_ms,
        )
        self.record_signals(signals)
    
    def predict_cost(self, layer_idx: int, timestep: Optional[int] = None) -> float:
        """
        Predict relative compute cost for a layer.
        
        Returns a value in [0, 1] where:
        - 0 = very cheap (sparse, reuses clusters, historically fast)
        - 1 = very expensive (dense, will recluster, historically slow)
        
        Higher cost layers should be prefetched earlier and pinned longer.
        
        Args:
            layer_idx: Layer to predict cost for
            timestep: Target timestep (default: current)
        
        Returns:
            Predicted cost in [0, 1]
        """
        if timestep is None:
            timestep = self._current_timestep
        
        cfg = self.config
        
        # Component 1: Attention density (higher density = more FLOPs)
        if layer_idx in self._density_ema and self._density_ema[layer_idx].count > 0:
            density_factor = self._density_ema[layer_idx].get_normalized()
        else:
            density_factor = 0.5  # Default: assume average
        
        # Component 2: CTCA clustering cost prediction
        cluster_factor = self._predict_cluster_cost(layer_idx)
        
        # Component 3: Historical timing
        if layer_idx in self._timing_ema and self._timing_ema[layer_idx].count > 0:
            timing_factor = self._timing_ema[layer_idx].get_normalized()
        else:
            timing_factor = 0.5  # Default: assume average
        
        # Weighted combination
        cost = (
            cfg.weight_attention * density_factor +
            cfg.weight_clustering * cluster_factor +
            cfg.weight_timing * timing_factor
        )
        
        return max(0.0, min(1.0, cost))
    
    def _predict_cluster_cost(self, layer_idx: int) -> float:
        """
        Predict clustering cost based on CTCA state.
        
        Uses calls-since-recluster and quality to predict whether
        the next call will trigger expensive full K-means.
        """
        cfg = self.config
        
        # Check CTCA manager if available
        if self.ctca_manager is not None:
            cache = self.ctca_manager.get_cache(layer_idx)
            
            if cache is None:
                # No cache = will need full clustering
                return cfg.recluster_cost_factor
            
            # Get cache state
            quality = cache.get_average_quality()
            calls_since = cache.calls_since_full_cluster
            
            # Predict: will next call trigger recluster?
            # CTCA reclusters when: quality < threshold OR calls >= max_interval
            quality_threshold = 0.8  # From CTCAConfig defaults
            max_interval = 10
            
            if calls_since >= max_interval - 1:
                # Will hit max interval next call
                return cfg.recluster_cost_factor
            elif quality < quality_threshold:
                # Quality dropped, likely recluster
                return cfg.recluster_cost_factor * 0.9
            elif quality < quality_threshold + 0.1:
                # Quality borderline
                return (cfg.recluster_cost_factor + cfg.reuse_cost_factor) / 2
            else:
                # Should reuse
                return cfg.reuse_cost_factor
        
        # Fallback to simple heuristic if no CTCA manager
        calls_since = self._calls_since_recluster.get(layer_idx, 0)
        quality = self._last_quality.get(layer_idx, 1.0)
        
        if calls_since == 0:
            # Just reclustered, next should reuse
            return cfg.reuse_cost_factor
        elif calls_since >= 8:
            # Approaching max interval
            return cfg.recluster_cost_factor * 0.8
        elif quality < 0.85:
            return cfg.recluster_cost_factor * 0.7
        else:
            return cfg.reuse_cost_factor
    
    def predict_transfer_stall(
        self,
        layer_idx: int,
        available_overlap_ms: float,
    ) -> float:
        """
        Predict memory transfer stall time for a layer.
        
        Stall = max(0, transfer_time - available_overlap_time)
        
        A layer causes stall if it takes longer to transfer than
        the available compute time to hide the transfer.
        
        Args:
            layer_idx: Layer index
            available_overlap_ms: Time available to overlap transfer (ms)
        
        Returns:
            Predicted stall time in milliseconds
        """
        transfer_time = self.config.layer_transfer_ms
        return max(0.0, transfer_time - available_overlap_ms)
    
    def get_layer_rankings(
        self,
        exclude_layers: Optional[Set[int]] = None,
        top_k: Optional[int] = None,
    ) -> List[Tuple[int, float]]:
        """
        Get layers ranked by predicted compute cost (highest first).
        
        Useful for:
        - Deciding which layers to pin (top-K expensive)
        - Prioritizing prefetch order
        
        Args:
            exclude_layers: Layers to exclude from ranking
            top_k: If set, return only top K layers
        
        Returns:
            List of (layer_idx, predicted_cost) sorted by cost descending
        """
        exclude = exclude_layers or set()
        
        rankings = []
        for layer_idx in range(self.num_layers):
            if layer_idx in exclude:
                continue
            cost = self.predict_cost(layer_idx)
            rankings.append((layer_idx, cost))
        
        # Sort by cost descending (expensive first)
        rankings.sort(key=lambda x: x[1], reverse=True)
        
        if top_k is not None:
            return rankings[:top_k]
        
        return rankings
    
    def identify_heavy_layers(self, k: int) -> List[int]:
        """
        Identify top-K heavy layers based on predicted cost.
        
        These layers should be pinned on GPU to avoid stalls.
        
        Args:
            k: Number of heavy layers to identify
        
        Returns:
            List of layer indices (sorted by cost, heaviest first)
        """
        rankings = self.get_layer_rankings(top_k=k)
        return [layer_idx for layer_idx, _ in rankings]
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get predictor statistics for debugging/logging."""
        
        # Calculate statistics
        layer_costs = [self.predict_cost(l) for l in range(self.num_layers)]
        avg_cost = sum(layer_costs) / max(len(layer_costs), 1)
        
        heavy_layers = self.identify_heavy_layers(k=min(10, self.num_layers))
        
        return {
            'total_samples': self._total_samples,
            'current_timestep': self._current_timestep,
            'avg_predicted_cost': avg_cost,
            'heavy_layers': heavy_layers,
            'global_avg_density': self._global_density.value,
            'global_avg_timing_ms': self._global_timing.value,
            'config': {
                'weight_attention': self.config.weight_attention,
                'weight_clustering': self.config.weight_clustering,
                'weight_timing': self.config.weight_timing,
            }
        }
    
    def reset(self):
        """Reset all state for new generation."""
        for ema in self._density_ema.values():
            ema._history.clear()
            ema._ema = None
            ema._count = 0
        
        for ema in self._timing_ema.values():
            ema._history.clear()
            ema._ema = None
            ema._count = 0
        
        self._calls_since_recluster.clear()
        self._last_quality.clear()
        self._global_density = RunningEMA(alpha=self.config.ema_alpha)
        self._global_timing = RunningEMA(alpha=self.config.ema_alpha)
        self._current_timestep = 0
        self._total_samples = 0


# =============================================================================
# Stall-Minimizing Pinning Optimizer
# =============================================================================

class StallMinimizingPinner:
    """
    Optimizes layer pinning to minimize total stall time.
    
    Problem Formulation:
    -------------------
    Given:
    - L layers with predicted costs cost[l]
    - GPU memory budget allowing K pinned layers
    - Layer transfer time T_transfer
    - Available overlap time for each layer
    
    Minimize:
        Total_Stall = Σ max(0, T_transfer - overlap_time[l]) for l not pinned
    
    Subject to:
        |Pinned_Layers| ≤ K
    
    Greedy Solution:
    ---------------
    Since pinning a layer eliminates its potential stall, we should pin
    the layers that would cause the most stall if not pinned. This is
    equivalent to pinning layers with:
        stall_if_not_pinned[l] = max(0, T_transfer - overlap_time[l])
    
    Where overlap_time[l] is the compute time of the previous layer
    (during which we can overlap the transfer).
    
    Approximation:
    -------------
    We approximate overlap_time[l] ≈ compute_time[l-1] using our
    cost predictor, scaled by average layer compute time.
    """
    
    def __init__(
        self,
        cost_predictor: ComputeCostPredictor,
        num_layers: int,
        transfer_time_ms: float = 50.0,
        avg_compute_time_ms: float = 100.0,
    ):
        """
        Initialize the stall-minimizing pinner.
        
        Args:
            cost_predictor: The compute cost predictor
            num_layers: Total number of layers
            transfer_time_ms: Typical CPU->GPU transfer time per layer
            avg_compute_time_ms: Average layer compute time
        """
        self.cost_predictor = cost_predictor
        self.num_layers = num_layers
        self.transfer_time_ms = transfer_time_ms
        self.avg_compute_time_ms = avg_compute_time_ms
    
    def compute_stall_if_not_pinned(self, layer_idx: int) -> float:
        """
        Compute potential stall time if layer is not pinned.
        
        Stall occurs when transfer time exceeds available overlap time.
        The overlap time is approximately the compute time of previous layer.
        
        Args:
            layer_idx: Layer index
        
        Returns:
            Potential stall time in milliseconds
        """
        if layer_idx == 0:
            # First layer has no previous layer to overlap with
            # At timestep start, we have inter-timestep gap, assume some overlap
            overlap_time = self.avg_compute_time_ms * 0.5
        else:
            # Overlap time = compute time of previous layer
            prev_cost = self.cost_predictor.predict_cost(layer_idx - 1)
            overlap_time = prev_cost * self.avg_compute_time_ms * 2.0  # Scale to ms
        
        return max(0.0, self.transfer_time_ms - overlap_time)
    
    def select_layers_to_pin(self, k: int) -> List[int]:
        """
        Select top-K layers to pin based on stall minimization.
        
        Strategy: Pin layers that would cause the most stall if not pinned.
        This is different from just pinning expensive layers - we also
        consider whether there's enough overlap time to hide the transfer.
        
        Args:
            k: Maximum number of layers to pin
        
        Returns:
            List of layer indices to pin (sorted by stall potential, highest first)
        """
        stall_scores = []
        
        for layer_idx in range(self.num_layers):
            stall = self.compute_stall_if_not_pinned(layer_idx)
            # Also factor in compute cost - expensive layers benefit more from pinning
            cost = self.cost_predictor.predict_cost(layer_idx)
            
            # Combined score: high stall + high cost = should pin
            score = stall + cost * self.avg_compute_time_ms * 0.5
            stall_scores.append((layer_idx, score))
        
        # Sort by score descending
        stall_scores.sort(key=lambda x: x[1], reverse=True)
        
        return [layer_idx for layer_idx, _ in stall_scores[:k]]
    
    def estimate_total_stall(
        self,
        pinned_layers: Set[int],
        num_timesteps: int = 50,
    ) -> float:
        """
        Estimate total stall time for a pinning configuration.
        
        Args:
            pinned_layers: Set of layer indices that are pinned
            num_timesteps: Number of diffusion timesteps
        
        Returns:
            Estimated total stall time in milliseconds
        """
        total_stall = 0.0
        
        for layer_idx in range(self.num_layers):
            if layer_idx not in pinned_layers:
                stall = self.compute_stall_if_not_pinned(layer_idx)
                total_stall += stall * num_timesteps
        
        return total_stall


# =============================================================================
# Factory Function
# =============================================================================

def create_cost_predictor(
    num_layers: int,
    weight_attention: float = 0.40,
    weight_clustering: float = 0.30,
    weight_timing: float = 0.30,
    ema_alpha: float = 0.3,
) -> ComputeCostPredictor:
    """
    Create a compute cost predictor with specified configuration.
    
    Args:
        num_layers: Number of transformer layers
        weight_attention: Weight for attention density signal
        weight_clustering: Weight for CTCA clustering signal
        weight_timing: Weight for historical timing signal
        ema_alpha: EMA smoothing factor
    
    Returns:
        Configured ComputeCostPredictor instance
    """
    config = CostModelConfig(
        weight_attention=weight_attention,
        weight_clustering=weight_clustering,
        weight_timing=weight_timing,
        ema_alpha=ema_alpha,
    )
    
    return ComputeCostPredictor(num_layers, config)
