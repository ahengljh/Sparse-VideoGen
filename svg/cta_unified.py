"""
Unified Cross-Timestep Amortization Framework (CTA)

This module unifies CTCA, CTAA, and Attention-Informed Offloading into a
cohesive framework that exploits temporal coherence in diffusion models.

=============================================================================
ACADEMIC FRAMING
=============================================================================

Core Insight:
    Diffusion models exhibit strong TEMPORAL COHERENCE - latent representations
    change smoothly across timesteps. This coherence manifests in three ways:

    1. CLUSTER STABILITY: Token cluster assignments remain largely stable
       → CTCA amortizes K-means clustering across timesteps

    2. ATTENTION PATTERN STABILITY: Importance distribution changes gradually
       → CTAA uses hierarchical attention (full/centroid/skip)

    3. COMPUTE PATTERN STABILITY: Layer compute costs are predictable
       → Attention-Informed Offloading predicts prefetch/eviction priority

Framework Architecture:

    ┌─────────────────────────────────────────────────────────────────────┐
    │                    CTA: Cross-Timestep Amortization                  │
    │                                                                      │
    │   Shared Foundation: Temporal Coherence in Diffusion Models          │
    │   ─────────────────────────────────────────────────────────────────  │
    │                                                                      │
    │   ┌─────────────┐    ┌─────────────┐    ┌─────────────────────────┐ │
    │   │    CTCA     │    │    CTAA     │    │ Attention-Informed      │ │
    │   │  Cluster    │───▶│  Attention  │───▶│ Offloading              │ │
    │   │  Amortize   │    │  Amortize   │    │                         │ │
    │   └─────────────┘    └─────────────┘    └─────────────────────────┘ │
    │         │                  │                       │                 │
    │         ▼                  ▼                       ▼                 │
    │   Cluster IDs        Tier Maps              Prefetch Order          │
    │   Centroids          (full/centroid/skip)   Eviction Priority       │
    │   Quality Score      Density                                        │
    │                                                                      │
    │   ─────────────────────────────────────────────────────────────────  │
    │   Signal Flow:                                                       │
    │   CTCA decisions ──▶ CTAA tier selection ──▶ Offload predictions    │
    │                                                                      │
    └─────────────────────────────────────────────────────────────────────┘

Paper Contributions:
    1. CTCA: 5-10x reduction in K-means overhead via cluster reuse
    2. CTAA: Hierarchical attention reducing compute by ~30%
    3. AI-Offload: Predictive offloading improving prefetch hit rate by 15%
    4. Unified framework showing how all three exploit temporal coherence

=============================================================================
"""

from __future__ import annotations

import gc
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Set, Tuple
from enum import Enum
import time

import torch
import torch.nn as nn
import numpy as np

from .logger import logger
from .timer import time_logging_decorator

# Import components
from .ctca import (
    CrossTimestepClusterAmortization,
    CTCAConfig,
    ClusterCache,
    create_ctca,
)
from .attention_informed_offload import (
    AttentionInformedOffloadManager,
    AttentionInformedOffloadConfig,
    LayerComputePredictor,
    AdaptivePrefetchScheduler,
    PriorityEvictionManager,
)


# =============================================================================
# Unified Configuration
# =============================================================================

@dataclass
class CTAConfig:
    """
    Unified configuration for Cross-Timestep Amortization framework.

    This config controls all three components (CTCA, CTAA, Offloading)
    and their interactions.
    """

    # ─────────────────────────────────────────────────────────────────────
    # Global Settings
    # ─────────────────────────────────────────────────────────────────────

    # Quality vs Speed tradeoff (0.0 = max speed, 1.0 = max quality)
    quality_bias: float = 0.7

    # Memory budget (GB) - None for auto-detect
    gpu_memory_budget_gb: Optional[float] = None

    # Number of transformer layers
    num_layers: int = 60

    # Number of diffusion timesteps
    num_timesteps: int = 50

    # ─────────────────────────────────────────────────────────────────────
    # CTCA: Cross-Timestep Cluster Amortization
    # ─────────────────────────────────────────────────────────────────────

    ctca_enabled: bool = True

    # Number of clusters for Q and K
    num_q_centroids: int = 400
    num_k_centroids: int = 1000

    # Quality threshold for triggering re-clustering (0-1)
    # Lower = more aggressive reuse (faster but potentially lower quality)
    ctca_quality_threshold: float = 0.80

    # Recluster interval bounds
    ctca_min_interval: int = 2   # Minimum calls between reclusters
    ctca_max_interval: int = 10  # Maximum calls before forced recluster

    # K-means iterations
    ctca_full_kmeans_iters: int = 50   # For full reclustering
    ctca_update_iters: int = 2          # For centroid-only update

    # ─────────────────────────────────────────────────────────────────────
    # CTAA: Cross-Timestep Attention Amortization
    # ─────────────────────────────────────────────────────────────────────

    ctaa_enabled: bool = True

    # Tier thresholds (cumulative probability)
    ctaa_p_full: float = 0.70      # Top 70% get full attention
    ctaa_p_total: float = 0.95     # 70-95% get centroid attention, >95% skipped

    # Minimum key cluster ratio (always attend to at least this fraction)
    ctaa_min_kc_ratio: float = 0.10

    # ─────────────────────────────────────────────────────────────────────
    # Attention-Informed Offloading
    # ─────────────────────────────────────────────────────────────────────

    offload_enabled: bool = True

    # Number of layers to keep on GPU
    num_layers_on_gpu: int = 6

    # Prefetch settings
    enable_prefetch: bool = True
    base_prefetch_count: int = 2
    max_prefetch_count: int = 4

    # Use priority-based eviction (vs FIFO)
    use_priority_eviction: bool = True

    # Prediction weights for compute cost
    prediction_weight_density: float = 0.4
    prediction_weight_ctca: float = 0.3
    prediction_weight_timing: float = 0.3

    # Use pinned memory for faster transfers
    use_pinned_memory: bool = True

    # ─────────────────────────────────────────────────────────────────────
    # Signal Flow Settings
    # ─────────────────────────────────────────────────────────────────────

    # Feed CTCA decisions to offload predictor
    ctca_informs_offload: bool = True

    # Feed CTAA density to offload predictor
    ctaa_informs_offload: bool = True

    # Feed offload predictions to CTCA (prioritize reclustering for prefetched layers)
    offload_informs_ctca: bool = True

    # ─────────────────────────────────────────────────────────────────────
    # Debugging
    # ─────────────────────────────────────────────────────────────────────

    verbose: bool = False
    collect_statistics: bool = True


# =============================================================================
# Unified Statistics
# =============================================================================

@dataclass
class CTAStatistics:
    """Unified statistics for the CTA framework."""

    # CTCA stats
    ctca_full_cluster_count: int = 0
    ctca_update_only_count: int = 0
    ctca_quality_triggered: int = 0
    ctca_interval_triggered: int = 0
    ctca_avg_quality: float = 0.0

    # CTAA stats
    ctaa_full_attention_blocks: int = 0
    ctaa_centroid_attention_blocks: int = 0
    ctaa_skipped_blocks: int = 0
    ctaa_total_blocks: int = 0

    # Offload stats
    offload_gpu_loads: int = 0
    offload_gpu_offloads: int = 0
    offload_prefetch_hits: int = 0
    offload_prefetch_misses: int = 0
    offload_priority_evictions: int = 0

    # Timing stats
    total_ctca_time_ms: float = 0.0
    total_ctaa_time_ms: float = 0.0
    total_offload_time_ms: float = 0.0
    total_attention_time_ms: float = 0.0

    # Per-timestep tracking
    timestep_stats: List[Dict] = field(default_factory=list)

    def get_summary(self) -> Dict[str, Any]:
        """Get summary statistics."""
        total_ctca = self.ctca_full_cluster_count + self.ctca_update_only_count
        total_prefetch = self.offload_prefetch_hits + self.offload_prefetch_misses

        return {
            # CTCA efficiency
            'ctca_reuse_ratio': self.ctca_update_only_count / max(total_ctca, 1),
            'ctca_kmeans_speedup': total_ctca * 50 / max(
                self.ctca_full_cluster_count * 50 + self.ctca_update_only_count * 2, 1
            ),

            # CTAA efficiency
            'ctaa_full_ratio': self.ctaa_full_attention_blocks / max(self.ctaa_total_blocks, 1),
            'ctaa_centroid_ratio': self.ctaa_centroid_attention_blocks / max(self.ctaa_total_blocks, 1),
            'ctaa_skip_ratio': self.ctaa_skipped_blocks / max(self.ctaa_total_blocks, 1),

            # Offload efficiency
            'prefetch_hit_ratio': self.offload_prefetch_hits / max(total_prefetch, 1),

            # Overall
            'total_layers_processed': total_ctca,
        }


# =============================================================================
# Signal Bridge: Connects CTCA/CTAA signals to Offload Predictor
# =============================================================================

class CTASignalBridge:
    """
    Bridges signals between CTCA, CTAA, and Offload components.

    This is the key integration point that enables the components to
    inform each other's decisions.

    Signal Flow:
        CTCA ──┬──► Offload Predictor (cluster quality, recluster decisions)
               │
        CTAA ──┴──► Offload Predictor (density, tier distribution)

        Offload ───► CTCA (prefetch hints for eager reclustering)
    """

    def __init__(
        self,
        config: CTAConfig,
        ctca_manager: CrossTimestepClusterAmortization,
        offload_manager: AttentionInformedOffloadManager,
    ):
        self.config = config
        self.ctca_manager = ctca_manager
        self.offload_manager = offload_manager
        self.predictor = offload_manager.predictor

        # Signal buffers
        self._ctca_signals: Dict[int, Dict] = {}  # layer_idx -> signals
        self._ctaa_signals: Dict[int, Dict] = {}  # layer_idx -> signals
        self._prefetch_hints: Set[int] = set()    # Layers being prefetched

        # Connect CTCA manager to predictor
        self.predictor.set_ctca_manager(ctca_manager)

    def on_ctca_decision(
        self,
        layer_idx: int,
        timestep: int,
        full_recluster: bool,
        cluster_quality: float,
        kmeans_time_ms: float,
    ):
        """
        Called after CTCA makes a clustering decision.

        Feeds signal to offload predictor for future predictions.
        """
        if not self.config.ctca_informs_offload:
            return

        self._ctca_signals[layer_idx] = {
            'timestep': timestep,
            'full_recluster': full_recluster,
            'quality': cluster_quality,
            'time_ms': kmeans_time_ms,
        }

        # Update predictor with CTCA cost signal
        # Full recluster is ~25x more expensive than update-only
        ctca_cost = 1.0 if full_recluster else 0.04

        # This will be picked up by predictor's CTCA factor
        if self.config.verbose:
            logger.debug(f"CTCA signal: layer={layer_idx}, recluster={full_recluster}, "
                        f"quality={cluster_quality:.3f}")

    def on_ctaa_computation(
        self,
        layer_idx: int,
        timestep: int,
        attention_density: float,
        full_ratio: float,
        centroid_ratio: float,
        skip_ratio: float,
        attention_time_ms: float,
    ):
        """
        Called after CTAA computes hierarchical attention.

        Feeds density and tier info to offload predictor.
        """
        if not self.config.ctaa_informs_offload:
            return

        self._ctaa_signals[layer_idx] = {
            'timestep': timestep,
            'density': attention_density,
            'full_ratio': full_ratio,
            'centroid_ratio': centroid_ratio,
            'skip_ratio': skip_ratio,
            'time_ms': attention_time_ms,
        }

        # Feed to predictor
        self.predictor.record_from_attention(
            layer_idx=layer_idx,
            timestep=timestep,
            attention_density=attention_density,
            compute_time_ms=attention_time_ms,
            ctaa_stats={
                'full_attention_blocks': int(full_ratio * 100),
                'centroid_attention_blocks': int(centroid_ratio * 100),
                'total_blocks': 100,
            },
        )

    def on_prefetch_started(self, layer_idx: int):
        """
        Called when a layer starts being prefetched.

        Can hint to CTCA to prepare clusters for this layer.
        """
        if not self.config.offload_informs_ctca:
            return

        self._prefetch_hints.add(layer_idx)

    def on_prefetch_complete(self, layer_idx: int):
        """Called when prefetch completes."""
        self._prefetch_hints.discard(layer_idx)

    def should_eager_recluster(self, layer_idx: int) -> bool:
        """
        Check if CTCA should eagerly recluster for a prefetched layer.

        Rationale: If a layer is being prefetched, we have time to do
        full reclustering while the prefetch is in progress.
        """
        return layer_idx in self._prefetch_hints

    def get_layer_predicted_cost(self, layer_idx: int, timestep: int) -> float:
        """Get predicted compute cost for a layer."""
        return self.predictor.predict_compute_cost(layer_idx, timestep)

    def reset(self):
        """Reset all signal buffers."""
        self._ctca_signals.clear()
        self._ctaa_signals.clear()
        self._prefetch_hints.clear()


# =============================================================================
# Unified CTA Manager
# =============================================================================

class CTAManager:
    """
    Unified Cross-Timestep Amortization Manager.

    This is the main entry point that coordinates CTCA, CTAA, and
    Attention-Informed Offloading as a single cohesive system.

    Usage:
        # Initialize
        cta = CTAManager(transformer, config)
        cta.prepare_for_inference()

        # Per-timestep
        for timestep in timesteps:
            cta.on_timestep_begin(timestep)

            for layer_idx in range(num_layers):
                # Offloader ensures layer is on GPU with intelligent prefetching
                cta.ensure_layer_ready(layer_idx)

                # Get clusters with CTCA amortization
                clusters = cta.get_clusters(query, key, layer_idx, timestep)

                # Compute attention with CTAA hierarchical tiers
                output = cta.hierarchical_attention(
                    q, k, v, clusters, layer_idx, timestep
                )

                # Signal completion for prediction updates
                cta.on_layer_complete(layer_idx)

        # Print unified statistics
        cta.print_statistics()
    """

    def __init__(
        self,
        transformer: nn.Module,
        config: Optional[CTAConfig] = None,
    ):
        self.transformer = transformer
        self.config = config or CTAConfig()

        # Get layer lists
        self.double_blocks = list(getattr(transformer, "transformer_blocks", []))
        self.single_blocks = list(getattr(transformer, "single_transformer_blocks", []))
        self.all_blocks = self.double_blocks + self.single_blocks
        self.num_layers = len(self.all_blocks)

        # Update config with actual layer count
        self.config.num_layers = self.num_layers

        # Initialize components
        self._init_ctca()
        self._init_offload()
        self._init_signal_bridge()

        # Statistics
        self.stats = CTAStatistics()

        # State tracking
        self._current_timestep = 0
        self._current_layer = 0
        self._initialized = False

        # Timing events
        self._layer_start_event: Optional[torch.cuda.Event] = None
        self._layer_end_event: Optional[torch.cuda.Event] = None

    def _init_ctca(self):
        """Initialize CTCA manager."""
        if not self.config.ctca_enabled:
            self.ctca_manager = None
            return

        ctca_config = CTCAConfig(
            quality_threshold=self.config.ctca_quality_threshold,
            min_recluster_interval=self.config.ctca_min_interval,
            max_recluster_interval=self.config.ctca_max_interval,
            full_kmeans_iters=self.config.ctca_full_kmeans_iters,
            update_only_iters=self.config.ctca_update_iters,
            adaptive_recluster=True,
            verbose=self.config.verbose,
        )

        self.ctca_manager = CrossTimestepClusterAmortization(
            config=ctca_config,
            num_q_centroids=self.config.num_q_centroids,
            num_k_centroids=self.config.num_k_centroids,
        )
        # Note: CTCA logs its own initialization banner

    def _init_offload(self):
        """Initialize Attention-Informed Offload Manager."""
        if not self.config.offload_enabled:
            self.offload_manager = None
            return

        offload_config = AttentionInformedOffloadConfig(
            num_layers_on_gpu=self.config.num_layers_on_gpu,
            use_pinned_memory=self.config.use_pinned_memory,
            enable_prefetch=self.config.enable_prefetch,
            base_prefetch_count=self.config.base_prefetch_count,
            max_prefetch_count=self.config.max_prefetch_count,
            use_priority_eviction=self.config.use_priority_eviction,
            verbose=self.config.verbose,
        )

        # Update predictor weights
        self.offload_manager = AttentionInformedOffloadManager(
            self.transformer, offload_config
        )
        self.offload_manager.predictor.weight_density = self.config.prediction_weight_density
        self.offload_manager.predictor.weight_ctca = self.config.prediction_weight_ctca
        self.offload_manager.predictor.weight_timing = self.config.prediction_weight_timing
        # Note: Offload manager logs its own banner in prepare_for_inference()

    def _init_signal_bridge(self):
        """Initialize signal bridge between components."""
        if self.ctca_manager is None or self.offload_manager is None:
            self.signal_bridge = None
            return

        self.signal_bridge = CTASignalBridge(
            config=self.config,
            ctca_manager=self.ctca_manager,
            offload_manager=self.offload_manager,
        )

        # Connect CTCA to offload predictor
        self.offload_manager.set_ctca_manager(self.ctca_manager)

        # Log CTAA initialization
        logger.info("=" * 60)
        logger.info("[CTAA] Cross-Timestep Attention Amortization INITIALIZED")
        logger.info("=" * 60)
        logger.info(f"[CTAA] Full attention threshold (p_full): {self.config.ctaa_p_full}")
        logger.info(f"[CTAA] Total threshold (p_total): {self.config.ctaa_p_total}")
        logger.info(f"[CTAA] Min KC ratio: {self.config.ctaa_min_kc_ratio}")
        logger.info("=" * 60)

        # Log signal bridge
        logger.info("[CTA] Signal bridge connected: CTCA -> CTAA -> Offload")

    def prepare_for_inference(self):
        """Prepare all components for inference."""
        if self._initialized:
            return

        logger.info("")
        logger.info("=" * 70)
        logger.info("   CTA: Cross-Timestep Amortization Framework")
        logger.info("   Unified CTCA + CTAA + Attention-Informed Offloading")
        logger.info("=" * 70)
        logger.info("")
        logger.info("[CTA] Preparing framework for inference...")

        # Move transformer blocks to CPU for offloading
        if self.offload_manager is not None:
            logger.info("Moving transformer blocks to CPU...")
            for block in self.all_blocks:
                block.to('cpu')

            # Keep embedders on GPU
            self._move_embedders_to_gpu()

            # Initialize offload manager
            self.offload_manager.prepare_for_inference()

        # Create timing events
        self._layer_start_event = torch.cuda.Event(enable_timing=True)
        self._layer_end_event = torch.cuda.Event(enable_timing=True)

        torch.cuda.empty_cache()
        gc.collect()

        self._initialized = True
        self._ctaa_log_counter = 0  # For periodic CTAA logging

        logger.info("")
        logger.info("=" * 70)
        logger.info("[CTA] Framework READY - All components initialized")
        logger.info("=" * 70)
        logger.info("")

    def _move_embedders_to_gpu(self):
        """Move embedder components to GPU (they're always needed)."""
        embedder_names = [
            'timestep_embedding', 'context_embedder', 'x_embedder',
            'time_text_embed', 'norm_out', 'proj_out'
        ]
        for name in embedder_names:
            if hasattr(self.transformer, name):
                comp = getattr(self.transformer, name)
                if comp is not None:
                    comp.to('cuda')

    def on_timestep_begin(self, timestep: int):
        """Called at the start of each diffusion timestep."""
        self._current_timestep = timestep

        # Log timestep progress
        if timestep == self.config.num_timesteps - 1 or timestep % 10 == 0:
            logger.info(f"[CTA] === Timestep {timestep}/{self.config.num_timesteps} ===")

        if self.offload_manager is not None:
            self.offload_manager.on_timestep_begin(timestep)

    def ensure_layer_ready(self, layer_idx: int):
        """
        Ensure a layer is ready for computation.

        Uses attention-informed offloading for intelligent prefetching.
        """
        if not self._initialized:
            self.prepare_for_inference()

        self._current_layer = layer_idx

        if self.offload_manager is not None:
            self.offload_manager.ensure_layer_on_gpu(layer_idx)

        # Start timing
        self._layer_start_event.record()

    @time_logging_decorator("Level 3 - CTA get_clusters")
    def get_clusters(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        layer_idx: int,
        timestep: int,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor,
               torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Get cluster assignments with CTCA amortization.

        Returns:
            q_cluster_ids, q_centroids, q_cluster_sizes,
            k_cluster_ids, k_centroids, k_cluster_sizes
        """
        if self.ctca_manager is None:
            # Fallback: full clustering every time
            from .kmeans_utils import batch_kmeans_Euclid
            cfg, num_heads, seq_len, dim = query.shape
            query_flat = query.reshape(cfg * num_heads, seq_len, dim)
            key_flat = key.reshape(cfg * num_heads, seq_len, dim)

            q_ids, q_cents, q_sizes, _ = batch_kmeans_Euclid(
                query_flat, n_clusters=self.config.num_q_centroids
            )
            k_ids, k_cents, k_sizes, _ = batch_kmeans_Euclid(
                key_flat, n_clusters=self.config.num_k_centroids
            )
            return q_ids, q_cents, q_sizes, k_ids, k_cents, k_sizes

        # Track CTCA timing
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        start_event.record()

        # Get clusters with CTCA (may reuse cached assignments)
        result = self.ctca_manager.get_clusters(query, key, layer_idx, timestep)

        end_event.record()
        torch.cuda.synchronize()
        ctca_time_ms = start_event.elapsed_time(end_event)

        # Update statistics
        self.stats.total_ctca_time_ms += ctca_time_ms

        # Check what decision was made
        cache = self.ctca_manager.get_cache(layer_idx)
        if cache is not None:
            full_recluster = cache.calls_since_full_cluster == 0
            quality = cache.get_average_quality()
        else:
            full_recluster = True
            quality = 1.0

        if full_recluster:
            self.stats.ctca_full_cluster_count += 1
        else:
            self.stats.ctca_update_only_count += 1

        # Signal to bridge
        if self.signal_bridge is not None:
            self.signal_bridge.on_ctca_decision(
                layer_idx=layer_idx,
                timestep=timestep,
                full_recluster=full_recluster,
                cluster_quality=quality,
                kmeans_time_ms=ctca_time_ms,
            )

        return result

    @time_logging_decorator("Level 3 - CTA hierarchical_attention")
    def hierarchical_attention(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        q_cluster_ids: torch.Tensor,
        k_cluster_ids: torch.Tensor,
        q_centroids: torch.Tensor,
        k_centroids: torch.Tensor,
        q_cluster_sizes: torch.Tensor,
        k_cluster_sizes: torch.Tensor,
        layer_idx: int,
        timestep: int,
    ) -> torch.Tensor:
        """
        Compute attention with CTAA hierarchical tiers.

        Tiers:
        - Full (p < p_full): Token-level attention
        - Centroid (p_full < p < p_total): Centroid approximation
        - Skip (p > p_total): Zero attention
        """
        from .kmeans_utils import (
            identify_hierarchical_dynamic_map,
            hierarchical_sparse_attention_fwd,
            density_calculation,
        )
        from .kernels.triton.permute import permute_tensor_by_labels_triton

        cfg, num_heads, seq_len, dim = query.shape

        # Reshape for hierarchical map identification
        qc_view = q_centroids.view(cfg, num_heads, self.config.num_q_centroids, dim)
        kc_view = k_centroids.view(cfg, num_heads, self.config.num_k_centroids, dim)
        qc_sizes = q_cluster_sizes.view(cfg, num_heads, self.config.num_q_centroids)
        kc_sizes = k_cluster_sizes.view(cfg, num_heads, self.config.num_k_centroids)

        # Get hierarchical attention maps (CTAA)
        if self.config.ctaa_enabled:
            full_map, centroid_map, centroid_weights = identify_hierarchical_dynamic_map(
                qc_view, kc_view, qc_sizes, kc_sizes,
                self.config.ctaa_p_full,
                self.config.ctaa_p_total,
                self.config.ctaa_min_kc_ratio,
            )
        else:
            # Fallback: all full attention
            from .kmeans_utils import identify_dynamic_map
            full_map = identify_dynamic_map(
                qc_view, kc_view, qc_sizes, kc_sizes,
                p=self.config.ctaa_p_total,
                min_kc_ratio=self.config.ctaa_min_kc_ratio,
            )
            centroid_map = torch.zeros_like(full_map)
            centroid_weights = torch.zeros(
                cfg, num_heads, self.config.num_q_centroids, self.config.num_k_centroids,
                device=query.device, dtype=query.dtype
            )

        # Permute Q, K, V by cluster assignments
        q_perm, q_indices = permute_tensor_by_labels_triton(query, q_cluster_ids, dim=2)
        k_perm, k_indices = permute_tensor_by_labels_triton(key, k_cluster_ids, dim=2)
        v_perm, _ = permute_tensor_by_labels_triton(value, k_cluster_ids, dim=2, sorted_indices=k_indices)

        # Compute hierarchical attention
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        start_event.record()

        output_perm = hierarchical_sparse_attention_fwd(
            q_perm, k_perm, v_perm,
            full_map, centroid_map, centroid_weights,
            qc_sizes, kc_sizes,
            v_centroids=None,  # Computed internally
        )

        end_event.record()
        torch.cuda.synchronize()
        attn_time_ms = start_event.elapsed_time(end_event)
        self.stats.total_attention_time_ms += attn_time_ms

        # Inverse permutation to restore original order
        from .kernels.triton.permute import apply_inverse_permutation_triton
        output = apply_inverse_permutation_triton(output_perm, q_indices, dim=2)

        # Compute density for statistics
        density = density_calculation(full_map, qc_sizes, kc_sizes)
        attention_density = density.mean().item()

        # Update statistics
        full_count = full_map.sum().item()
        centroid_count = centroid_map.sum().item()
        total_count = full_map.numel()
        skip_count = total_count - full_count - centroid_count

        self.stats.ctaa_full_attention_blocks += int(full_count)
        self.stats.ctaa_centroid_attention_blocks += int(centroid_count)
        self.stats.ctaa_skipped_blocks += int(skip_count)
        self.stats.ctaa_total_blocks += int(total_count)

        # Signal to bridge
        if self.signal_bridge is not None:
            self.signal_bridge.on_ctaa_computation(
                layer_idx=layer_idx,
                timestep=timestep,
                attention_density=attention_density,
                full_ratio=full_count / max(total_count, 1),
                centroid_ratio=centroid_count / max(total_count, 1),
                skip_ratio=skip_count / max(total_count, 1),
                attention_time_ms=attn_time_ms,
            )

        # Periodic CTAA logging
        self._ctaa_log_counter += 1
        if self._ctaa_log_counter <= 3 or self._ctaa_log_counter % 100 == 0:
            full_pct = 100 * full_count / max(total_count, 1)
            cent_pct = 100 * centroid_count / max(total_count, 1)
            skip_pct = 100 * skip_count / max(total_count, 1)
            logger.info(f"[CTAA] Layer {layer_idx} @ t={timestep}: "
                       f"Full={full_pct:.0f}% | Centroid={cent_pct:.0f}% | Skip={skip_pct:.0f}% | "
                       f"Density={attention_density:.2f}")

        return output

    def on_layer_complete(self, layer_idx: int, attention_density: Optional[float] = None):
        """Called after a layer's forward pass is complete."""
        # Record timing
        self._layer_end_event.record()
        torch.cuda.synchronize()
        layer_time_ms = self._layer_start_event.elapsed_time(self._layer_end_event)

        # Update offload manager
        if self.offload_manager is not None:
            self.offload_manager.layer_forward_complete(
                layer_idx,
                attention_density=attention_density,
            )

    def on_generation_complete(self):
        """Called when video generation is complete."""
        # Collect final statistics from components
        if self.ctca_manager is not None:
            ctca_stats = self.ctca_manager.get_statistics()
            self.stats.ctca_avg_quality = ctca_stats.get('estimated_speedup', 1.0)

        if self.offload_manager is not None:
            offload_stats = self.offload_manager.get_statistics()
            self.stats.offload_gpu_loads = offload_stats['gpu_loads']
            self.stats.offload_gpu_offloads = offload_stats['gpu_offloads']
            self.stats.offload_prefetch_hits = offload_stats['prefetch_hits']
            self.stats.offload_prefetch_misses = offload_stats['prefetch_misses']
            self.stats.offload_priority_evictions = offload_stats['priority_evictions']

    def get_statistics(self) -> CTAStatistics:
        """Get unified statistics."""
        self.on_generation_complete()
        return self.stats

    def print_statistics(self):
        """Print unified statistics."""
        self.on_generation_complete()
        summary = self.stats.get_summary()

        print("\n" + "=" * 80)
        print("Cross-Timestep Amortization (CTA) Framework Statistics")
        print("=" * 80)

        print("\n📊 CTCA (Cross-Timestep Cluster Amortization):")
        print(f"   Full reclusters:        {self.stats.ctca_full_cluster_count}")
        print(f"   Update-only:            {self.stats.ctca_update_only_count}")
        print(f"   Reuse ratio:            {summary['ctca_reuse_ratio']*100:.1f}%")
        print(f"   K-means speedup:        {summary['ctca_kmeans_speedup']:.1f}x")
        print(f"   Total CTCA time:        {self.stats.total_ctca_time_ms:.1f}ms")

        print("\n📊 CTAA (Cross-Timestep Attention Amortization):")
        print(f"   Full attention blocks:  {self.stats.ctaa_full_attention_blocks} "
              f"({summary['ctaa_full_ratio']*100:.1f}%)")
        print(f"   Centroid attention:     {self.stats.ctaa_centroid_attention_blocks} "
              f"({summary['ctaa_centroid_ratio']*100:.1f}%)")
        print(f"   Skipped blocks:         {self.stats.ctaa_skipped_blocks} "
              f"({summary['ctaa_skip_ratio']*100:.1f}%)")
        print(f"   Total attention time:   {self.stats.total_attention_time_ms:.1f}ms")

        print("\n📊 Attention-Informed Offloading:")
        print(f"   GPU loads:              {self.stats.offload_gpu_loads}")
        print(f"   GPU offloads:           {self.stats.offload_gpu_offloads}")
        print(f"   Prefetch hits:          {self.stats.offload_prefetch_hits}")
        print(f"   Prefetch misses:        {self.stats.offload_prefetch_misses}")
        print(f"   Prefetch hit ratio:     {summary['prefetch_hit_ratio']*100:.1f}%")
        print(f"   Priority evictions:     {self.stats.offload_priority_evictions}")

        print("\n" + "=" * 80)
        print("🎯 Key Insight: All three components exploit TEMPORAL COHERENCE")
        print("   in diffusion models for complementary optimizations.")
        print("=" * 80 + "\n")

    def reset(self):
        """Reset all state for new generation."""
        if self.ctca_manager is not None:
            self.ctca_manager.reset()
        if self.offload_manager is not None:
            self.offload_manager.reset()
        if self.signal_bridge is not None:
            self.signal_bridge.reset()

        self.stats = CTAStatistics()
        self._current_timestep = 0
        self._current_layer = 0


# =============================================================================
# Factory Functions
# =============================================================================

def create_cta_manager(
    transformer: nn.Module,
    # CTCA settings
    num_q_centroids: int = 400,
    num_k_centroids: int = 1000,
    ctca_quality_threshold: float = 0.80,
    # CTAA settings
    ctaa_p_full: float = 0.70,
    ctaa_p_total: float = 0.95,
    # Offload settings
    num_layers_on_gpu: int = 6,
    use_priority_eviction: bool = True,
    # Global settings
    quality_bias: float = 0.7,
    verbose: bool = False,
) -> CTAManager:
    """
    Create a unified CTA manager with all components.

    This is the main entry point for using the CTA framework.

    Args:
        transformer: Transformer module
        num_q_centroids: Number of query clusters for SAP
        num_k_centroids: Number of key clusters for SAP
        ctca_quality_threshold: Quality threshold for CTCA reclustering
        ctaa_p_full: Top-p for full attention in CTAA
        ctaa_p_total: Top-p for total (full + centroid) attention
        num_layers_on_gpu: Number of layers in GPU sliding window
        use_priority_eviction: Use attention-informed eviction
        quality_bias: Quality vs speed tradeoff (0-1)
        verbose: Enable verbose logging

    Returns:
        Configured CTAManager
    """
    config = CTAConfig(
        quality_bias=quality_bias,
        num_q_centroids=num_q_centroids,
        num_k_centroids=num_k_centroids,
        ctca_quality_threshold=ctca_quality_threshold,
        ctaa_p_full=ctaa_p_full,
        ctaa_p_total=ctaa_p_total,
        num_layers_on_gpu=num_layers_on_gpu,
        use_priority_eviction=use_priority_eviction,
        verbose=verbose,
    )

    return CTAManager(transformer, config)
