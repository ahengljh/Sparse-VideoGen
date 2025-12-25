"""
SADSA: Semantic-Aware Dynamic Sparse Attention for Video Diffusion

This module implements four key components:
1. STIS (Semantic Token Importance Scoring): Score tokens by text-relevance and motion
2. DSAS (Diffusion-Stage Adaptive Sparsity): Stage-specific attention thresholds
3. MCAR (Motion-Conditioned Attention Routing): Route high-motion regions to full attention
4. QPTC (Quality-Preserving Temporal Coherence): Detect and prevent quality degradation

Memory-efficient design:
- Minimal state storage (only last timestep cached)
- CPU offloading for non-critical data
- Aggressive cache cleanup
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from enum import Enum

import torch
import torch.nn.functional as F

from .logger import logger
from .timer import time_logging_decorator


# =============================================================================
# Configuration
# =============================================================================

class DiffusionStage(Enum):
    """Stages of diffusion process with different sparsity requirements."""
    STRUCTURE = "structure"  # t > 0.7: Global structure, aggressive sparsity OK
    SEMANTIC = "semantic"    # 0.3 < t < 0.7: Semantic formation, balanced
    DETAIL = "detail"        # t < 0.3: Fine details, conservative sparsity


@dataclass
class StageConfig:
    """Configuration for each diffusion stage."""
    stage: DiffusionStage
    t_start: float  # Normalized timestep start (higher = earlier in diffusion)
    t_end: float    # Normalized timestep end

    # Attention thresholds
    p_full: float           # Fraction for full attention tier
    p_total: float          # Fraction for full + centroid tiers

    # Clustering parameters
    recluster_interval: int  # Max calls before forced recluster
    quality_threshold: float # Quality threshold for recluster trigger

    # Motion sensitivity (higher = more aggressive full attention for motion)
    motion_sensitivity: float = 1.0


# Default stage configurations
DEFAULT_STAGE_CONFIGS = [
    StageConfig(
        stage=DiffusionStage.STRUCTURE,
        t_start=0.7, t_end=1.0,
        p_full=0.50,  # Aggressive: 50% full
        p_total=0.90, # 40% centroid, 10% skip
        recluster_interval=15,
        quality_threshold=0.65,
        motion_sensitivity=0.5,  # Less sensitive to motion
    ),
    StageConfig(
        stage=DiffusionStage.SEMANTIC,
        t_start=0.3, t_end=0.7,
        p_full=0.70,  # Balanced
        p_total=0.95,
        recluster_interval=8,
        quality_threshold=0.75,
        motion_sensitivity=1.0,
    ),
    StageConfig(
        stage=DiffusionStage.DETAIL,
        t_start=0.0, t_end=0.3,
        p_full=0.85,  # Conservative: 85% full
        p_total=0.98, # Only 13% centroid, 2% skip
        recluster_interval=5,
        quality_threshold=0.85,
        motion_sensitivity=1.5,  # Very sensitive to motion
    ),
]


@dataclass
class SADSAConfig:
    """Main configuration for SADSA framework."""

    # Stage configurations
    stage_configs: List[StageConfig] = field(default_factory=lambda: DEFAULT_STAGE_CONFIGS)

    # Clustering parameters
    num_q_centroids: int = 400
    num_k_centroids: int = 1000
    kmeans_init_iters: int = 50
    kmeans_step_iters: int = 2

    # Motion detection
    motion_threshold_high: float = 0.6  # High motion -> full attention
    motion_threshold_low: float = 0.15  # Low motion -> can skip

    # Token importance weights
    motion_importance_weight: float = 0.6
    edge_importance_weight: float = 0.4

    # Quality monitoring
    quality_window_size: int = 3  # Timesteps to track for quality trend
    quality_drop_threshold: float = 0.12  # Trigger conservative mode if quality drops

    # Memory management
    max_cached_timesteps: int = 2  # Only cache last N timesteps

    # Logging
    verbose: bool = False


# =============================================================================
# DSAS: Diffusion-Stage Adaptive Sparsity
# =============================================================================

class DiffusionStageAdaptiveSparsity:
    """
    Adapts sparsity thresholds based on diffusion stage.

    Key insight: Early timesteps focus on global structure (can be sparse),
    late timesteps add fine details (need full attention).
    """

    def __init__(self, stage_configs: List[StageConfig]):
        self.stage_configs = sorted(stage_configs, key=lambda x: -x.t_start)

    def get_stage(self, timestep: int, max_timestep: int = 1000) -> StageConfig:
        """Get configuration for current diffusion stage."""
        t_normalized = timestep / max_timestep

        for config in self.stage_configs:
            if config.t_end <= t_normalized <= config.t_start:
                return config

        # Default to most conservative (detail stage)
        return self.stage_configs[-1]

    def get_thresholds(
        self,
        timestep: int,
        max_timestep: int = 1000,
        motion_level: float = 0.0,
    ) -> Tuple[float, float]:
        """
        Get (p_full, p_total) thresholds adapted to stage and motion.

        Args:
            timestep: Current diffusion timestep
            max_timestep: Maximum timestep value
            motion_level: Average motion level (0-1)

        Returns:
            (p_full, p_total) thresholds
        """
        config = self.get_stage(timestep, max_timestep)

        # Adjust thresholds based on motion
        # High motion -> increase p_full (more full attention)
        motion_boost = motion_level * config.motion_sensitivity * 0.15

        p_full = min(config.p_full + motion_boost, 0.95)
        p_total = min(config.p_total + motion_boost * 0.5, 0.99)

        return p_full, p_total


# =============================================================================
# MCAR: Motion-Conditioned Attention Routing
# =============================================================================

class MotionConditionedRouter:
    """
    Routes tokens to attention tiers based on motion magnitude.

    Key insight: High-motion regions need full attention to avoid blur,
    static regions can use approximation.
    """

    def __init__(
        self,
        motion_threshold_high: float = 0.6,
        motion_threshold_low: float = 0.15,
    ):
        self.motion_threshold_high = motion_threshold_high
        self.motion_threshold_low = motion_threshold_low

        # Cache for motion estimation (only keep last timestep)
        self._prev_latent: Optional[torch.Tensor] = None
        self._motion_cache: Optional[torch.Tensor] = None

    @time_logging_decorator("Level 4 - Motion estimation")
    def estimate_motion(
        self,
        current_latent: torch.Tensor,  # [B, C, T, H, W] or [B, S, D]
        is_video_format: bool = True,
    ) -> torch.Tensor:
        """
        Estimate motion magnitude per spatial/temporal location.

        Returns motion map normalized to [0, 1].
        """
        if self._prev_latent is None:
            self._prev_latent = current_latent.detach()
            # First call: assume medium motion everywhere
            if is_video_format:
                B, C, T, H, W = current_latent.shape
                return torch.full((B, T * H * W), 0.3, device=current_latent.device)
            else:
                B, S, D = current_latent.shape
                return torch.full((B, S), 0.3, device=current_latent.device)

        # Compute temporal difference
        diff = (current_latent - self._prev_latent).abs()

        if is_video_format:
            # Aggregate across channels: [B, C, T, H, W] -> [B, T, H, W]
            motion = diff.mean(dim=1)
            B, T, H, W = motion.shape
            motion = motion.view(B, T * H * W)
        else:
            # [B, S, D] -> [B, S]
            motion = diff.mean(dim=-1)

        # Normalize per-sample
        motion_max = motion.amax(dim=-1, keepdim=True).clamp(min=1e-6)
        motion = motion / motion_max

        # Update cache
        self._prev_latent = current_latent.detach()
        self._motion_cache = motion

        return motion

    def get_motion_masks(
        self,
        motion_map: torch.Tensor,  # [B, S]
    ) -> Dict[str, torch.Tensor]:
        """
        Classify tokens into motion-based tiers.

        Returns masks for routing decisions.
        """
        high_motion = motion_map > self.motion_threshold_high
        low_motion = motion_map < self.motion_threshold_low
        medium_motion = ~(high_motion | low_motion)

        return {
            'force_full': high_motion,      # Must use full attention
            'prefer_centroid': medium_motion, # Centroid is OK
            'allow_skip': low_motion,        # Can potentially skip
        }

    def get_motion_level(self) -> float:
        """Get average motion level from last estimation."""
        if self._motion_cache is None:
            return 0.3
        return self._motion_cache.mean().item()

    def reset(self):
        """Reset motion cache for new video."""
        self._prev_latent = None
        self._motion_cache = None


# =============================================================================
# QPTC: Quality-Preserving Temporal Coherence
# =============================================================================

class QualityPreservingCoherence:
    """
    Monitors quality signals and adapts thresholds to prevent degradation.

    Key insight: Track trajectory smoothness and attention stability;
    sudden changes indicate potential quality issues.
    """

    def __init__(
        self,
        window_size: int = 3,
        quality_drop_threshold: float = 0.12,
    ):
        self.window_size = window_size
        self.quality_drop_threshold = quality_drop_threshold

        # Quality history (kept small for memory)
        self._quality_history: List[float] = []
        self._density_history: Dict[int, List[float]] = {}  # layer_idx -> densities

        # Latent trajectory tracking (only keep last 2)
        self._latent_velocities: List[float] = []

    def compute_trajectory_smoothness(
        self,
        current_latent: torch.Tensor,
        prev_latent: Optional[torch.Tensor],
        prev_prev_latent: Optional[torch.Tensor],
    ) -> float:
        """
        Measure smoothness of latent trajectory.

        Smooth trajectory = consistent generation.
        Sudden jumps = potential quality issues.
        """
        if prev_latent is None or prev_prev_latent is None:
            return 1.0

        # Compute velocities
        v_current = (current_latent - prev_latent).norm()
        v_prev = (prev_latent - prev_prev_latent).norm()

        # Compute acceleration (change in velocity)
        acceleration = abs(v_current - v_prev)

        # Smoothness: high if acceleration is low relative to velocity
        smoothness = 1.0 / (1.0 + acceleration / (v_prev + 1e-6))

        return min(smoothness.item(), 1.0)

    def update_density_signal(self, layer_idx: int, density: float):
        """Record density for stability tracking."""
        if layer_idx not in self._density_history:
            self._density_history[layer_idx] = []

        history = self._density_history[layer_idx]
        history.append(density)

        # Keep only recent history
        if len(history) > self.window_size * 2:
            self._density_history[layer_idx] = history[-self.window_size:]

    def compute_density_stability(self, layer_idx: int) -> float:
        """Check if density patterns are stable."""
        if layer_idx not in self._density_history:
            return 1.0

        history = self._density_history[layer_idx]
        if len(history) < 2:
            return 1.0

        import numpy as np
        densities = history[-self.window_size:]
        if len(densities) < 2:
            return 1.0

        mean_d = np.mean(densities)
        std_d = np.std(densities)

        # Stability: 1 if std is low relative to mean
        stability = 1.0 / (1.0 + std_d / (mean_d + 1e-6))
        return min(stability, 1.0)

    def estimate_quality(
        self,
        smoothness: float,
        density_stability: float,
    ) -> float:
        """Combined quality estimate."""
        quality = 0.5 * smoothness + 0.5 * density_stability

        self._quality_history.append(quality)
        if len(self._quality_history) > self.window_size * 2:
            self._quality_history = self._quality_history[-self.window_size:]

        return quality

    def should_use_conservative_mode(self) -> bool:
        """Check if quality is degrading and conservative mode is needed."""
        if len(self._quality_history) < self.window_size:
            return False

        recent = self._quality_history[-self.window_size:]

        # Check for declining trend
        trend = recent[-1] - recent[0]
        if trend < -self.quality_drop_threshold:
            return True

        # Check for absolute low quality
        if recent[-1] < 0.5:
            return True

        return False

    def get_threshold_adjustment(self) -> Tuple[float, float]:
        """
        Get adjustment to apply to thresholds.

        Returns (p_full_boost, p_total_boost) to add to base thresholds.
        """
        if self.should_use_conservative_mode():
            return (0.15, 0.03)  # Boost full attention significantly
        return (0.0, 0.0)

    def reset(self):
        """Reset for new video."""
        self._quality_history.clear()
        self._density_history.clear()
        self._latent_velocities.clear()


# =============================================================================
# STIS: Semantic Token Importance Scoring
# =============================================================================

class SemanticTokenImportanceScorer:
    """
    Score tokens by importance based on motion and edge density.

    Note: Cross-attention based text relevance is handled separately
    in the attention processor since it requires access to cross-attention maps.
    """

    def __init__(
        self,
        motion_weight: float = 0.6,
        edge_weight: float = 0.4,
    ):
        self.motion_weight = motion_weight
        self.edge_weight = edge_weight

    @time_logging_decorator("Level 4 - Edge density")
    def compute_edge_density(
        self,
        latent: torch.Tensor,  # [B, C, T, H, W] or [B, S, D]
        is_video_format: bool = True,
    ) -> torch.Tensor:
        """
        Estimate edge/detail density per location.

        High-frequency regions (edges, textures) need more attention.
        """
        if not is_video_format:
            # For sequence format, use local variance as proxy
            B, S, D = latent.shape
            # Compute local variance in feature space
            mean = latent.mean(dim=-1, keepdim=True)
            variance = ((latent - mean) ** 2).mean(dim=-1)
            # Normalize
            var_max = variance.amax(dim=-1, keepdim=True).clamp(min=1e-6)
            return variance / var_max

        B, C, T, H, W = latent.shape

        # Compute spatial gradients as edge proxy
        # Sobel-like filter approximation
        dx = (latent[:, :, :, :, 1:] - latent[:, :, :, :, :-1]).abs()
        dy = (latent[:, :, :, 1:, :] - latent[:, :, :, :-1, :]).abs()

        # Pad to original size
        dx = F.pad(dx, (0, 1, 0, 0, 0, 0), mode='replicate')
        dy = F.pad(dy, (0, 0, 0, 1, 0, 0), mode='replicate')

        # Combine gradients
        edges = (dx + dy).mean(dim=1)  # [B, T, H, W]
        edges = edges.view(B, T * H * W)

        # Normalize
        edge_max = edges.amax(dim=-1, keepdim=True).clamp(min=1e-6)
        return edges / edge_max

    def compute_importance(
        self,
        motion_map: torch.Tensor,  # [B, S]
        edge_map: torch.Tensor,    # [B, S]
    ) -> torch.Tensor:
        """
        Combine motion and edge signals into importance score.
        """
        importance = (
            self.motion_weight * motion_map +
            self.edge_weight * edge_map
        )

        # Normalize to [0, 1]
        imp_max = importance.amax(dim=-1, keepdim=True).clamp(min=1e-6)
        return importance / imp_max


# =============================================================================
# SADSA Manager: Unified Interface
# =============================================================================

class SADSAManager:
    """
    Unified manager for Semantic-Aware Dynamic Sparse Attention.

    Coordinates all components:
    - DSAS: Stage-adaptive thresholds
    - MCAR: Motion-conditioned routing
    - QPTC: Quality preservation
    - STIS: Token importance scoring

    Usage:
        sadsa = SADSAManager(config)

        for timestep in timesteps:
            sadsa.on_timestep_begin(timestep, current_latent)

            for layer_idx in layers:
                p_full, p_total = sadsa.get_adaptive_thresholds(layer_idx, timestep)
                motion_masks = sadsa.get_motion_masks()

                # ... do attention ...

                sadsa.on_layer_complete(layer_idx, density)
    """

    def __init__(self, config: Optional[SADSAConfig] = None):
        self.config = config or SADSAConfig()

        # Initialize components
        self.dsas = DiffusionStageAdaptiveSparsity(self.config.stage_configs)
        self.mcar = MotionConditionedRouter(
            motion_threshold_high=self.config.motion_threshold_high,
            motion_threshold_low=self.config.motion_threshold_low,
        )
        self.qptc = QualityPreservingCoherence(
            window_size=self.config.quality_window_size,
            quality_drop_threshold=self.config.quality_drop_threshold,
        )
        self.stis = SemanticTokenImportanceScorer(
            motion_weight=self.config.motion_importance_weight,
            edge_weight=self.config.edge_importance_weight,
        )

        # State
        self._current_timestep = 0
        self._max_timestep = 1000
        self._current_motion_level = 0.3
        self._current_motion_masks: Optional[Dict[str, torch.Tensor]] = None
        self._current_importance: Optional[torch.Tensor] = None

        # Latent history for quality tracking (kept minimal)
        self._latent_t_minus_1: Optional[torch.Tensor] = None
        self._latent_t_minus_2: Optional[torch.Tensor] = None

        # Stats
        self._stats = {
            'conservative_mode_triggers': 0,
            'full_attention_boosts': 0,
            'layers_processed': 0,
        }

        self._log_counter = 0

    def on_timestep_begin(
        self,
        timestep: int,
        current_latent: Optional[torch.Tensor] = None,
        max_timestep: int = 1000,
    ):
        """
        Called at start of each diffusion timestep.

        Updates motion estimates and quality tracking.
        """
        self._current_timestep = timestep
        self._max_timestep = max_timestep

        if current_latent is not None:
            # Determine format
            is_video = current_latent.dim() == 5  # [B, C, T, H, W]

            # Estimate motion
            motion_map = self.mcar.estimate_motion(current_latent, is_video)
            self._current_motion_level = motion_map.mean().item()
            self._current_motion_masks = self.mcar.get_motion_masks(motion_map)

            # Compute edge density for importance
            edge_map = self.stis.compute_edge_density(current_latent, is_video)
            self._current_importance = self.stis.compute_importance(motion_map, edge_map)

            # Update quality tracking
            smoothness = self.qptc.compute_trajectory_smoothness(
                current_latent,
                self._latent_t_minus_1,
                self._latent_t_minus_2,
            )

            # Shift latent history
            self._latent_t_minus_2 = self._latent_t_minus_1
            # Store on CPU to save GPU memory
            self._latent_t_minus_1 = current_latent.detach().cpu() if current_latent is not None else None

            # Log stage info periodically
            self._log_counter += 1
            if self._log_counter <= 3 or self._log_counter % 10 == 0:
                stage = self.dsas.get_stage(timestep, max_timestep)
                logger.info(f"[SADSA] Timestep {timestep}: Stage={stage.stage.value}, "
                           f"Motion={self._current_motion_level:.2f}")

    def get_adaptive_thresholds(
        self,
        layer_idx: int,
        timestep: Optional[int] = None,
    ) -> Tuple[float, float]:
        """
        Get (p_full, p_total) thresholds adapted to current state.

        Considers:
        - Diffusion stage
        - Motion level
        - Quality feedback
        """
        if timestep is None:
            timestep = self._current_timestep

        # Get stage-based thresholds with motion adjustment
        p_full, p_total = self.dsas.get_thresholds(
            timestep,
            self._max_timestep,
            self._current_motion_level,
        )

        # Apply quality-based adjustment
        q_boost_full, q_boost_total = self.qptc.get_threshold_adjustment()
        if q_boost_full > 0:
            self._stats['conservative_mode_triggers'] += 1

        p_full = min(p_full + q_boost_full, 0.95)
        p_total = min(p_total + q_boost_total, 0.99)

        return p_full, p_total

    def get_stage_config(self, timestep: Optional[int] = None) -> StageConfig:
        """Get current stage configuration."""
        if timestep is None:
            timestep = self._current_timestep
        return self.dsas.get_stage(timestep, self._max_timestep)

    def get_motion_masks(self) -> Optional[Dict[str, torch.Tensor]]:
        """Get motion-based routing masks."""
        return self._current_motion_masks

    def get_token_importance(self) -> Optional[torch.Tensor]:
        """Get token importance scores."""
        return self._current_importance

    def should_force_full_attention(self, layer_idx: int) -> bool:
        """
        Check if this layer should use full attention regardless of clustering.

        Used for critical early layers or when quality is severely degraded.
        """
        stage = self.get_stage_config()

        # First few layers in detail stage should use full attention
        if stage.stage == DiffusionStage.DETAIL and layer_idx < 3:
            return True

        # If quality is very low, use full attention
        if self.qptc.should_use_conservative_mode():
            if len(self.qptc._quality_history) > 0:
                if self.qptc._quality_history[-1] < 0.4:
                    return True

        return False

    def on_layer_complete(self, layer_idx: int, density: float):
        """Called after layer attention is complete."""
        self._stats['layers_processed'] += 1

        # Update quality tracking
        self.qptc.update_density_signal(layer_idx, density)

        # Compute quality estimate
        stability = self.qptc.compute_density_stability(layer_idx)

        # Use last known smoothness (approximate)
        smoothness = 0.8 if len(self.qptc._latent_velocities) == 0 else 0.7
        self.qptc.estimate_quality(smoothness, stability)

    def reset(self):
        """Reset all state for new video generation."""
        self.mcar.reset()
        self.qptc.reset()
        self._current_timestep = 0
        self._current_motion_level = 0.3
        self._current_motion_masks = None
        self._current_importance = None
        self._latent_t_minus_1 = None
        self._latent_t_minus_2 = None
        self._log_counter = 0

    def get_statistics(self) -> Dict:
        """Get SADSA statistics."""
        return {
            **self._stats,
            'current_motion_level': self._current_motion_level,
            'conservative_mode_active': self.qptc.should_use_conservative_mode(),
        }

    def print_statistics(self):
        """Print SADSA statistics."""
        stats = self.get_statistics()
        print("\n" + "=" * 60)
        print("SADSA (Semantic-Aware Dynamic Sparse Attention) Statistics")
        print("=" * 60)
        print(f"Layers processed:            {stats['layers_processed']}")
        print(f"Conservative mode triggers:  {stats['conservative_mode_triggers']}")
        print(f"Full attention boosts:       {stats['full_attention_boosts']}")
        print(f"Current motion level:        {stats['current_motion_level']:.2f}")
        print(f"Conservative mode active:    {stats['conservative_mode_active']}")
        print("=" * 60 + "\n")


# =============================================================================
# Factory Function
# =============================================================================

def create_sadsa_manager(
    # Stage thresholds
    structure_p_full: float = 0.50,
    semantic_p_full: float = 0.70,
    detail_p_full: float = 0.85,
    # Motion detection
    motion_threshold_high: float = 0.6,
    motion_threshold_low: float = 0.15,
    # Quality
    quality_drop_threshold: float = 0.12,
    # Clustering
    num_q_centroids: int = 400,
    num_k_centroids: int = 1000,
    verbose: bool = False,
) -> SADSAManager:
    """
    Factory function to create configured SADSA manager.
    """
    stage_configs = [
        StageConfig(
            stage=DiffusionStage.STRUCTURE,
            t_start=0.7, t_end=1.0,
            p_full=structure_p_full,
            p_total=min(structure_p_full + 0.40, 0.95),
            recluster_interval=15,
            quality_threshold=0.65,
            motion_sensitivity=0.5,
        ),
        StageConfig(
            stage=DiffusionStage.SEMANTIC,
            t_start=0.3, t_end=0.7,
            p_full=semantic_p_full,
            p_total=min(semantic_p_full + 0.25, 0.98),
            recluster_interval=8,
            quality_threshold=0.75,
            motion_sensitivity=1.0,
        ),
        StageConfig(
            stage=DiffusionStage.DETAIL,
            t_start=0.0, t_end=0.3,
            p_full=detail_p_full,
            p_total=min(detail_p_full + 0.13, 0.99),
            recluster_interval=5,
            quality_threshold=0.85,
            motion_sensitivity=1.5,
        ),
    ]

    config = SADSAConfig(
        stage_configs=stage_configs,
        num_q_centroids=num_q_centroids,
        num_k_centroids=num_k_centroids,
        motion_threshold_high=motion_threshold_high,
        motion_threshold_low=motion_threshold_low,
        quality_drop_threshold=quality_drop_threshold,
        verbose=verbose,
    )

    return SADSAManager(config)
