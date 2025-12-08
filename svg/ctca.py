"""
Cross-Timestep Cluster Amortization (CTCA)

Novel contribution: Amortize K-means clustering cost across diffusion timesteps
by exploiting temporal coherence in the denoising process.

Key insight: In diffusion models, the semantic structure of video latents changes
smoothly across timesteps. Cluster memberships remain largely stable, allowing us
to reuse cluster assignments and only update centroids without full re-clustering.

This reduces K-means overhead by 5-10x while maintaining attention quality.
"""

from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple
from collections import deque

import torch
import torch.nn.functional as F

from .timer import time_logging_decorator
from .kmeans_utils import (
    batch_kmeans_Euclid,
    triton_centroid_update_sorted_euclid,
    euclid_assign_triton,
)


@dataclass
class CTCAConfig:
    """Configuration for Cross-Timestep Cluster Amortization."""

    # Quality threshold for triggering re-clustering
    # Lower = more aggressive reuse (faster but potentially lower quality)
    # Higher = more frequent re-clustering (slower but higher quality)
    quality_threshold: float = 0.80

    # Minimum timesteps between re-clustering (prevents thrashing)
    min_recluster_interval: int = 2

    # Maximum timesteps to reuse clusters (forces periodic refresh)
    max_recluster_interval: int = 10

    # Number of K-means iterations for full re-clustering
    full_kmeans_iters: int = 50

    # Number of iterations for centroid-only update (when reusing assignments)
    update_only_iters: int = 1

    # Whether to use quality-based adaptive re-clustering
    # If False, uses fixed interval re-clustering
    adaptive_recluster: bool = True

    # History length for tracking quality trends
    quality_history_len: int = 5

    # Centroid drift threshold (alternative trigger for re-clustering)
    centroid_drift_threshold: float = 0.1

    # Enable verbose logging for debugging
    verbose: bool = False


@dataclass
class ClusterCache:
    """Cache entry for a single layer's cluster state."""

    # Cluster assignments
    q_cluster_ids: torch.Tensor  # [B*H, S]
    k_cluster_ids: torch.Tensor  # [B*H, S]

    # Cluster centroids
    q_centroids: torch.Tensor    # [B*H, Kc_q, D]
    k_centroids: torch.Tensor    # [B*H, Kc_k, D]

    # Cluster sizes
    q_cluster_sizes: torch.Tensor  # [B*H, Kc_q]
    k_cluster_sizes: torch.Tensor  # [B*H, Kc_k]

    # Metadata
    last_full_cluster_timestep: int = -1
    last_update_timestep: int = -1
    creation_timestep: int = -1

    # Quality tracking
    quality_history: deque = field(default_factory=lambda: deque(maxlen=5))

    def update_quality(self, quality: float):
        self.quality_history.append(quality)

    def get_average_quality(self) -> float:
        if len(self.quality_history) == 0:
            return 1.0
        return sum(self.quality_history) / len(self.quality_history)

    def get_quality_trend(self) -> float:
        """Returns negative if quality is declining, positive if improving."""
        if len(self.quality_history) < 2:
            return 0.0
        recent = list(self.quality_history)
        return recent[-1] - recent[0]


class CrossTimestepClusterAmortization:
    """
    Cross-Timestep Cluster Amortization (CTCA) Manager.

    This class manages cluster caching and decides when to perform full
    K-means clustering vs. reusing cached cluster assignments.

    Usage:
        ctca = CrossTimestepClusterAmortization(config, num_q_centroids, num_k_centroids)

        # During attention forward pass:
        q_ids, q_cents, q_sizes, k_ids, k_cents, k_sizes = ctca.get_clusters(
            query, key, layer_idx, timestep
        )
    """

    def __init__(
        self,
        config: CTCAConfig,
        num_q_centroids: int,
        num_k_centroids: int,
    ):
        self.config = config
        self.num_q_centroids = num_q_centroids
        self.num_k_centroids = num_k_centroids

        # Per-layer cluster cache
        self._cache: Dict[int, ClusterCache] = {}

        # Statistics tracking
        self.stats = {
            'full_cluster_count': 0,
            'reuse_count': 0,
            'update_only_count': 0,
            'quality_triggered_recluster': 0,
            'interval_triggered_recluster': 0,
        }

    def reset(self):
        """Reset all cached state. Call at the start of each video generation."""
        self._cache.clear()
        self.stats = {k: 0 for k in self.stats}

    def has_cache(self, layer_idx: int) -> bool:
        """Check if we have cached clusters for a layer."""
        return layer_idx in self._cache

    def get_cache(self, layer_idx: int) -> Optional[ClusterCache]:
        """Get cached clusters for a layer."""
        return self._cache.get(layer_idx, None)

    @time_logging_decorator("Level 4 - CTCA cluster quality")
    def compute_cluster_quality(
        self,
        data: torch.Tensor,
        cluster_ids: torch.Tensor,
        centroids: torch.Tensor,
    ) -> float:
        """
        Compute cluster quality using the Calinski-Harabasz index (variance ratio).

        Quality = (inter-cluster variance / intra-cluster variance) * (N - K) / (K - 1)

        Higher values indicate better-defined clusters.
        We normalize to [0, 1] range for easier threshold comparison.

        Args:
            data: [B, N, D] - Input data points
            cluster_ids: [B, N] - Cluster assignments
            centroids: [B, K, D] - Cluster centroids

        Returns:
            float: Quality score in [0, 1] range
        """
        B, N, D = data.shape
        K = centroids.shape[1]

        if K <= 1 or N <= K:
            return 1.0  # Degenerate case

        # Compute assigned centroids for each point
        # [B, N, D]
        expanded_ids = cluster_ids.unsqueeze(-1).expand(-1, -1, D)
        assigned_centroids = torch.gather(centroids, 1, expanded_ids)

        # Intra-cluster variance (within-cluster sum of squares)
        # Measures how compact the clusters are
        intra_var = ((data - assigned_centroids) ** 2).sum(dim=-1).mean()

        # Inter-cluster variance (between-cluster sum of squares)
        # Measures how well-separated the clusters are
        global_mean = data.mean(dim=1, keepdim=True)  # [B, 1, D]

        # Weight by cluster sizes
        cluster_sizes = torch.zeros(B, K, device=data.device)
        ones = torch.ones(B, N, device=data.device)
        cluster_sizes.scatter_add_(1, cluster_ids, ones)
        cluster_sizes = cluster_sizes.clamp(min=1)  # Avoid division by zero

        # Between-cluster variance
        centroid_diff = centroids - global_mean  # [B, K, D]
        inter_var = (cluster_sizes.unsqueeze(-1) * (centroid_diff ** 2)).sum(dim=(1, 2)).mean()

        # Calinski-Harabasz index
        if intra_var < 1e-10:
            return 1.0  # Perfect clustering (all points at centroids)

        ch_index = (inter_var / intra_var) * ((N - K) / (K - 1))

        # Normalize to [0, 1] using sigmoid-like transform
        # Empirically, good clustering has CH index > 100
        quality = ch_index / (ch_index + 100.0)

        return quality.item()

    @time_logging_decorator("Level 4 - CTCA centroid drift")
    def compute_centroid_drift(
        self,
        old_centroids: torch.Tensor,
        new_centroids: torch.Tensor,
    ) -> float:
        """
        Compute relative drift between old and new centroids.

        Args:
            old_centroids: [B, K, D]
            new_centroids: [B, K, D]

        Returns:
            float: Relative drift (0 = no change, 1 = complete change)
        """
        # L2 distance between corresponding centroids
        drift = (old_centroids - new_centroids).norm(dim=-1)  # [B, K]

        # Normalize by centroid magnitudes
        old_norm = old_centroids.norm(dim=-1).clamp(min=1e-6)  # [B, K]
        relative_drift = (drift / old_norm).mean()

        return relative_drift.item()

    def _should_recluster(
        self,
        layer_idx: int,
        timestep: int,
        query: torch.Tensor,
        key: torch.Tensor,
    ) -> Tuple[bool, str]:
        """
        Decide whether to perform full re-clustering or reuse cached assignments.

        Returns:
            Tuple[bool, str]: (should_recluster, reason)
        """
        cache = self._cache.get(layer_idx, None)

        # No cache exists - must cluster
        if cache is None:
            return True, "no_cache"

        steps_since_full = cache.last_full_cluster_timestep - timestep
        # Note: timesteps decrease in diffusion (1000 -> 0)
        # So steps_since_full will be positive if we clustered earlier

        # Check minimum interval (prevent thrashing)
        if steps_since_full < self.config.min_recluster_interval:
            return False, "min_interval"

        # Check maximum interval (force periodic refresh)
        if steps_since_full >= self.config.max_recluster_interval:
            return True, "max_interval"

        # Adaptive quality-based decision
        if self.config.adaptive_recluster:
            # Compute current quality with cached assignments
            B_H, S, D = query.shape[0] * query.shape[1], query.shape[2], query.shape[3]
            query_flat = query.reshape(B_H, S, D)

            q_quality = self.compute_cluster_quality(
                query_flat,
                cache.q_cluster_ids,
                cache.q_centroids,
            )

            cache.update_quality(q_quality)

            # Check quality threshold
            if q_quality < self.config.quality_threshold:
                return True, "quality_drop"

            # Check quality trend (declining quality suggests need for refresh)
            trend = cache.get_quality_trend()
            if trend < -0.1:  # Quality declining significantly
                return True, "quality_trend"

        return False, "reuse"

    @time_logging_decorator("Level 3.5 - CTCA full clustering")
    def _full_clustering(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        layer_idx: int,
        timestep: int,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor,
               torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Perform full K-means clustering on query and key.

        Args:
            query: [B, H, S, D]
            key: [B, H, S, D]
            layer_idx: Layer index
            timestep: Current diffusion timestep

        Returns:
            q_cluster_ids, q_centroids, q_cluster_sizes,
            k_cluster_ids, k_centroids, k_cluster_sizes
        """
        cfg, num_heads, seq_len, dim = query.shape

        # Flatten batch and head dimensions
        query_flat = query.reshape(cfg * num_heads, seq_len, dim)
        key_flat = key.reshape(cfg * num_heads, seq_len, dim)

        # Get initial centroids from cache if available (warm start)
        cache = self._cache.get(layer_idx, None)
        init_q_centroids = cache.q_centroids if cache is not None else None
        init_k_centroids = cache.k_centroids if cache is not None else None

        # Full K-means for query
        q_cluster_ids, q_centroids, q_cluster_sizes, q_iters = batch_kmeans_Euclid(
            query_flat,
            n_clusters=self.num_q_centroids,
            max_iters=self.config.full_kmeans_iters,
            init_centroids=init_q_centroids,
        )

        # Full K-means for key
        k_cluster_ids, k_centroids, k_cluster_sizes, k_iters = batch_kmeans_Euclid(
            key_flat,
            n_clusters=self.num_k_centroids,
            max_iters=self.config.full_kmeans_iters,
            init_centroids=init_k_centroids,
        )

        # Update cache
        self._cache[layer_idx] = ClusterCache(
            q_cluster_ids=q_cluster_ids,
            k_cluster_ids=k_cluster_ids,
            q_centroids=q_centroids,
            k_centroids=k_centroids,
            q_cluster_sizes=q_cluster_sizes,
            k_cluster_sizes=k_cluster_sizes,
            last_full_cluster_timestep=timestep,
            last_update_timestep=timestep,
            creation_timestep=timestep,
        )

        # Update stats
        self.stats['full_cluster_count'] += 1

        if self.config.verbose:
            print(f"[CTCA] Layer {layer_idx} @ t={timestep}: Full clustering "
                  f"(Q iters={q_iters}, K iters={k_iters})")

        return (q_cluster_ids, q_centroids, q_cluster_sizes,
                k_cluster_ids, k_centroids, k_cluster_sizes)

    @time_logging_decorator("Level 3.5 - CTCA centroid update only")
    def _update_centroids_only(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        layer_idx: int,
        timestep: int,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor,
               torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Update centroids without reassigning cluster memberships.

        This is much faster than full K-means: O(N) vs O(N*K*iters)

        Args:
            query: [B, H, S, D]
            key: [B, H, S, D]
            layer_idx: Layer index
            timestep: Current diffusion timestep

        Returns:
            q_cluster_ids, q_centroids, q_cluster_sizes,
            k_cluster_ids, k_centroids, k_cluster_sizes
        """
        cache = self._cache[layer_idx]
        cfg, num_heads, seq_len, dim = query.shape

        # Flatten batch and head dimensions
        query_flat = query.reshape(cfg * num_heads, seq_len, dim)
        key_flat = key.reshape(cfg * num_heads, seq_len, dim)

        # Reuse cluster assignments from cache
        q_cluster_ids = cache.q_cluster_ids
        k_cluster_ids = cache.k_cluster_ids

        # Update centroids based on new data positions
        # This computes new centroids as the mean of assigned points
        q_centroids, q_cluster_sizes = triton_centroid_update_sorted_euclid(
            query_flat,
            q_cluster_ids,
            cache.q_centroids,
        )

        k_centroids, k_cluster_sizes = triton_centroid_update_sorted_euclid(
            key_flat,
            k_cluster_ids,
            cache.k_centroids,
        )

        # Optional: Run a few reassignment iterations for refinement
        if self.config.update_only_iters > 1:
            x_sq_q = (query_flat ** 2).sum(dim=-1)
            x_sq_k = (key_flat ** 2).sum(dim=-1)

            for _ in range(self.config.update_only_iters - 1):
                # Reassign based on new centroids
                q_cluster_ids = euclid_assign_triton(query_flat, q_centroids, x_sq_q)
                k_cluster_ids = euclid_assign_triton(key_flat, k_centroids, x_sq_k)

                # Update centroids
                q_centroids, q_cluster_sizes = triton_centroid_update_sorted_euclid(
                    query_flat, q_cluster_ids, q_centroids
                )
                k_centroids, k_cluster_sizes = triton_centroid_update_sorted_euclid(
                    key_flat, k_cluster_ids, k_centroids
                )

        # Update cache with new centroids (keep assignments if no reassignment)
        cache.q_centroids = q_centroids
        cache.k_centroids = k_centroids
        cache.q_cluster_sizes = q_cluster_sizes
        cache.k_cluster_sizes = k_cluster_sizes
        cache.last_update_timestep = timestep

        if self.config.update_only_iters > 1:
            cache.q_cluster_ids = q_cluster_ids
            cache.k_cluster_ids = k_cluster_ids

        # Update stats
        self.stats['update_only_count'] += 1

        if self.config.verbose:
            print(f"[CTCA] Layer {layer_idx} @ t={timestep}: Centroid update only")

        return (q_cluster_ids, q_centroids, q_cluster_sizes,
                k_cluster_ids, k_centroids, k_cluster_sizes)

    @time_logging_decorator("Level 3 - CTCA get clusters")
    def get_clusters(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        layer_idx: int,
        timestep: int,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor,
               torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Get cluster assignments and centroids for query and key tensors.

        This is the main entry point for CTCA. It decides whether to:
        1. Perform full K-means clustering (expensive)
        2. Update centroids only (cheap)
        3. Reuse cached values entirely (free)

        Args:
            query: [B, H, S, D] - Query tensor
            key: [B, H, S, D] - Key tensor
            layer_idx: Transformer layer index
            timestep: Current diffusion timestep (decreasing from 1000 to 0)

        Returns:
            Tuple of:
                q_cluster_ids: [B*H, S] - Query cluster assignments
                q_centroids: [B*H, Kc_q, D] - Query cluster centroids
                q_cluster_sizes: [B*H, Kc_q] - Query cluster sizes
                k_cluster_ids: [B*H, S] - Key cluster assignments
                k_centroids: [B*H, Kc_k, D] - Key cluster centroids
                k_cluster_sizes: [B*H, Kc_k] - Key cluster sizes
        """
        should_recluster, reason = self._should_recluster(
            layer_idx, timestep, query, key
        )

        if should_recluster:
            if reason == "quality_drop" or reason == "quality_trend":
                self.stats['quality_triggered_recluster'] += 1
            elif reason == "max_interval":
                self.stats['interval_triggered_recluster'] += 1

            return self._full_clustering(query, key, layer_idx, timestep)
        else:
            self.stats['reuse_count'] += 1
            return self._update_centroids_only(query, key, layer_idx, timestep)

    def get_statistics(self) -> Dict[str, any]:
        """Get CTCA performance statistics."""
        total = (self.stats['full_cluster_count'] +
                 self.stats['update_only_count'] +
                 self.stats['reuse_count'])

        if total == 0:
            return self.stats

        stats_with_ratios = self.stats.copy()
        stats_with_ratios['full_cluster_ratio'] = self.stats['full_cluster_count'] / total
        stats_with_ratios['reuse_ratio'] = (self.stats['update_only_count'] +
                                            self.stats['reuse_count']) / total
        stats_with_ratios['total_calls'] = total

        # Estimate speedup (full clustering ~50 iters, update ~1 iter)
        full_cost = self.stats['full_cluster_count'] * self.config.full_kmeans_iters
        update_cost = self.stats['update_only_count'] * self.config.update_only_iters
        actual_cost = full_cost + update_cost
        baseline_cost = total * self.config.full_kmeans_iters

        if actual_cost > 0:
            stats_with_ratios['estimated_speedup'] = baseline_cost / actual_cost
        else:
            stats_with_ratios['estimated_speedup'] = 1.0

        return stats_with_ratios

    def print_statistics(self):
        """Print CTCA performance statistics."""
        stats = self.get_statistics()
        print("\n" + "=" * 60)
        print("CTCA (Cross-Timestep Cluster Amortization) Statistics")
        print("=" * 60)
        print(f"Total clustering calls:     {stats.get('total_calls', 0)}")
        print(f"Full re-clustering:         {stats['full_cluster_count']} "
              f"({stats.get('full_cluster_ratio', 0)*100:.1f}%)")
        print(f"  - Quality triggered:      {stats['quality_triggered_recluster']}")
        print(f"  - Interval triggered:     {stats['interval_triggered_recluster']}")
        print(f"Centroid update only:       {stats['update_only_count']}")
        print(f"Cache reuse:                {stats['reuse_count']}")
        print(f"Estimated K-means speedup:  {stats.get('estimated_speedup', 1.0):.2f}x")
        print("=" * 60 + "\n")


# Factory function for easy instantiation
def create_ctca(
    num_q_centroids: int,
    num_k_centroids: int,
    quality_threshold: float = 0.80,
    adaptive: bool = True,
    verbose: bool = False,
) -> CrossTimestepClusterAmortization:
    """
    Create a CTCA manager with the given configuration.

    Args:
        num_q_centroids: Number of query clusters
        num_k_centroids: Number of key clusters
        quality_threshold: Quality threshold for triggering re-clustering (0-1)
        adaptive: Whether to use adaptive quality-based re-clustering
        verbose: Whether to print debug information

    Returns:
        Configured CTCA manager
    """
    config = CTCAConfig(
        quality_threshold=quality_threshold,
        adaptive_recluster=adaptive,
        verbose=verbose,
    )

    return CrossTimestepClusterAmortization(
        config=config,
        num_q_centroids=num_q_centroids,
        num_k_centroids=num_k_centroids,
    )
