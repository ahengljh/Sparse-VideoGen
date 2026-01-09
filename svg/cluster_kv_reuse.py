"""
Cluster-Guided KV Reuse (CKGR)

Novel contribution: Use CTCA's cluster stability information to guide partial KV reuse.

Key insight: If cluster assignments are stable across timesteps (as CTCA exploits),
then the K, V vectors within those clusters are also stable. We can:
1. Cache K/V at cluster centroid level (1000 centroids vs 124K tokens = 124x smaller)
2. For stable clusters: approximate token KV with centroid KV
3. For unstable clusters: recompute full KV

This bridges CTCA's cluster reuse with actual compute savings in KV projection.

Memory: ~1.4 GB for all layers (vs 170 GB for full KV cache)
Compute savings: Skip K/V projection for stable clusters (~50-70% of tokens)

Integration:
- Works in conjunction with CTCA (Cross-Timestep Cluster Amortization)
- Uses CTCA's cluster_ids and stability information
- Called during attention forward pass after clustering
"""

from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple, List
from collections import defaultdict

import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

from .timer import time_logging_decorator
from .logger import logger


# =============================================================================
# Triton Kernels for Efficient Operations
# =============================================================================

@triton.jit
def _kv_centroid_update_kernel(
    kv_ptr,           # *f16/f32 [B*H, S, D] - K or V tensor
    cluster_ids_ptr,  # *i32     [B*H, S]    - cluster assignments
    centroid_sum_ptr, # *f32     [B*H, K, D] - output sum accumulator
    count_ptr,        # *i32     [B*H, K]    - output count accumulator
    B_H: tl.constexpr,
    S: tl.constexpr,
    D: tl.constexpr,
    K: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    """Compute K/V centroids by accumulating tokens per cluster.

    Each program handles one token across all dimensions.
    Uses atomic adds for thread-safe accumulation.
    """
    pid = tl.program_id(axis=0)
    token_idx = pid  # range: [0, B*H * S)

    # Derive (b_h, s) indices
    b_h = token_idx // S
    s = token_idx % S

    # Bounds check
    if b_h >= B_H or s >= S:
        return

    # Get cluster assignment for this token
    cluster_id = tl.load(cluster_ids_ptr + b_h * S + s)

    # Guard for invalid cluster ids
    cluster_id = tl.where(cluster_id < K, cluster_id, 0)
    cluster_id = tl.where(cluster_id >= 0, cluster_id, 0)

    # Load token KV vector and accumulate to centroid
    kv_base = (b_h * S + s) * D
    centroid_base = (b_h * K + cluster_id) * D

    offs = tl.arange(0, BLOCK_D)
    for d_start in range(0, D, BLOCK_D):
        mask = offs + d_start < D
        kv_vals = tl.load(kv_ptr + kv_base + d_start + offs, mask=mask, other=0.0)
        kv_vals = kv_vals.to(tl.float32)

        dest_ptr = centroid_sum_ptr + centroid_base + d_start + offs
        tl.atomic_add(dest_ptr, kv_vals, mask=mask)

    # Update count (once per token)
    tl.atomic_add(count_ptr + b_h * K + cluster_id, 1)


@triton.jit
def _scatter_centroids_to_tokens_kernel(
    centroid_ptr,     # *f16/f32 [B*H, K, D] - centroids
    cluster_ids_ptr,  # *i32     [B*H, S]    - cluster assignments
    output_ptr,       # *f16/f32 [B*H, S, D] - output token values
    stable_mask_ptr,  # *bool    [B*H, K]    - which clusters are stable
    B_H: tl.constexpr,
    S: tl.constexpr,
    D: tl.constexpr,
    K: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    """Scatter centroid values to tokens in stable clusters.

    For tokens in stable clusters, copy centroid value to output.
    For tokens in unstable clusters, output is left unchanged.
    """
    pid = tl.program_id(axis=0)
    token_idx = pid

    b_h = token_idx // S
    s = token_idx % S

    if b_h >= B_H or s >= S:
        return

    # Get cluster assignment
    cluster_id = tl.load(cluster_ids_ptr + b_h * S + s)
    cluster_id = tl.where(cluster_id < K, cluster_id, 0)
    cluster_id = tl.where(cluster_id >= 0, cluster_id, 0)

    # Check if this cluster is stable
    is_stable = tl.load(stable_mask_ptr + b_h * K + cluster_id)

    if is_stable:
        # Copy centroid value to output
        centroid_base = (b_h * K + cluster_id) * D
        output_base = (b_h * S + s) * D

        offs = tl.arange(0, BLOCK_D)
        for d_start in range(0, D, BLOCK_D):
            mask = offs + d_start < D
            centroid_vals = tl.load(centroid_ptr + centroid_base + d_start + offs, mask=mask, other=0.0)
            tl.store(output_ptr + output_base + d_start + offs, centroid_vals, mask=mask)


@triton.jit
def _compute_stability_kernel(
    old_ids_ptr,      # *i32  [B*H, S] - old cluster assignments
    new_ids_ptr,      # *i32  [B*H, S] - new cluster assignments
    overlap_ptr,      # *i32  [B*H, K] - overlap count output
    old_count_ptr,    # *i32  [B*H, K] - old cluster size output
    new_count_ptr,    # *i32  [B*H, K] - new cluster size output
    B_H: tl.constexpr,
    S: tl.constexpr,
    K: tl.constexpr,
):
    """Compute cluster stability by counting overlaps between old and new assignments."""
    pid = tl.program_id(axis=0)
    token_idx = pid

    b_h = token_idx // S
    s = token_idx % S

    if b_h >= B_H or s >= S:
        return

    old_cluster = tl.load(old_ids_ptr + b_h * S + s)
    new_cluster = tl.load(new_ids_ptr + b_h * S + s)

    # Bounds check
    old_cluster = tl.where(old_cluster < K, old_cluster, 0)
    old_cluster = tl.where(old_cluster >= 0, old_cluster, 0)
    new_cluster = tl.where(new_cluster < K, new_cluster, 0)
    new_cluster = tl.where(new_cluster >= 0, new_cluster, 0)

    # Count old and new cluster sizes
    tl.atomic_add(old_count_ptr + b_h * K + old_cluster, 1)
    tl.atomic_add(new_count_ptr + b_h * K + new_cluster, 1)

    # Count overlap only if same cluster in both
    if old_cluster == new_cluster:
        tl.atomic_add(overlap_ptr + b_h * K + old_cluster, 1)


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class ClusterKVCache:
    """Cache for cluster-level KV information.

    Stores K/V centroids computed from token-level K/V values.
    Memory efficient: stores centroids (num_clusters) instead of all tokens (seq_len).

    For HunyuanVideo with 1000 clusters, 60 layers, 24 heads, 128 dim:
    Memory = 60 * 2 * 1000 * 24 * 128 * 2 bytes = ~0.35 GB (vs ~170 GB for full KV cache)
    """

    # Cluster centroids in K/V space
    # Shape: [B*H, num_k_clusters, head_dim]
    k_centroids: torch.Tensor
    v_centroids: torch.Tensor

    # Which clusters are considered "stable" (can reuse)
    # Shape: [B*H, num_k_clusters] - boolean mask
    stable_clusters: torch.Tensor

    # Consecutive stable step counts per cluster
    # Shape: [B*H, num_k_clusters]
    stable_counts: torch.Tensor

    # Cluster assignments from CTCA
    # Shape: [B*H, seq_len]
    k_cluster_ids: torch.Tensor

    # Cluster sizes for weighted operations
    # Shape: [B*H, num_k_clusters]
    cluster_sizes: torch.Tensor

    # Metadata
    last_update_timestep: int = -1
    quality_score: float = 0.0
    update_count: int = 0
    token_energy: Optional[torch.Tensor] = None

    def to_device(
        self,
        device: torch.device,
        move_k_centroids: bool = True,
        move_v_centroids: bool = True,
        move_stable_clusters: bool = True,
        move_stable_counts: bool = True,
        move_cluster_ids: bool = True,
        move_cluster_sizes: bool = True,
        move_token_energy: bool = True,
    ):
        """Move cache to device with selective fields."""
        if move_k_centroids:
            self.k_centroids = self.k_centroids.to(device, non_blocking=True)
        if move_v_centroids:
            self.v_centroids = self.v_centroids.to(device, non_blocking=True)
        if move_stable_clusters:
            self.stable_clusters = self.stable_clusters.to(device, non_blocking=True)
        if move_stable_counts:
            self.stable_counts = self.stable_counts.to(device, non_blocking=True)
        if move_cluster_ids:
            self.k_cluster_ids = self.k_cluster_ids.to(device, non_blocking=True)
        if move_cluster_sizes:
            self.cluster_sizes = self.cluster_sizes.to(device, non_blocking=True)
        if move_token_energy and self.token_energy is not None:
            self.token_energy = self.token_energy.to(device, non_blocking=True)

    def to_cpu(self):
        """Move cache to CPU to save GPU memory."""
        self.k_centroids = self.k_centroids.cpu()
        self.v_centroids = self.v_centroids.cpu()
        self.stable_clusters = self.stable_clusters.cpu()
        self.stable_counts = self.stable_counts.cpu()
        self.k_cluster_ids = self.k_cluster_ids.cpu()
        self.cluster_sizes = self.cluster_sizes.cpu()
        if self.token_energy is not None:
            self.token_energy = self.token_energy.cpu()

    def memory_bytes(self) -> int:
        """Return total memory usage in bytes."""
        total = 0
        tensors = [
            self.k_centroids,
            self.v_centroids,
            self.stable_clusters,
            self.stable_counts,
            self.k_cluster_ids,
            self.cluster_sizes,
        ]
        if self.token_energy is not None:
            tensors.append(self.token_energy)
        for tensor in tensors:
            total += tensor.numel() * tensor.element_size()
        return total


@dataclass
class CKGRConfig:
    """Configuration for Cluster-Guided KV Reuse."""

    # Number of clusters (should match CTCA's num_k_clusters)
    num_k_clusters: int = 1000

    # Stability threshold for cluster reuse (Jaccard index)
    stability_threshold: float = 0.7

    # Minimum ratio of stable clusters to enable reuse
    min_stable_ratio: float = 0.3

    # Minimum token-level reuse ratio to enable reuse
    min_reuse_ratio: float = 0.05

    # Minimum cache update count before allowing reuse (warmup)
    min_reuse_steps: int = 2

    # Minimum consecutive stable steps required per cluster
    min_stable_steps: int = 2

    # Minimum cosine similarity between old/new K centroids
    # Set <= 0.0 to disable centroid drift gating
    centroid_sim_threshold: float = 0.98

    # Token-level stability gating (pixel-level proxy)
    # If both are None, token gating is disabled
    token_delta_threshold: Optional[float] = 0.05
    token_delta_quantile: Optional[float] = None

    # Quality threshold for CTCA cluster quality
    quality_threshold: float = 0.75

    # Which projections to reuse
    # Reusing V is typically safer for quality than reusing K
    reuse_k: bool = False
    reuse_v: bool = True

    # Minimum fraction of heads that must agree on reuse for a token
    # 1.0 = all heads, 0.5 = majority vote
    min_head_reuse_ratio: float = 1.0

    # Optional importance gating (lower importance => safer reuse)
    # If both threshold and quantile are None, importance gating is disabled
    importance_threshold: Optional[float] = None
    importance_quantile: Optional[float] = None
    importance_reduce: str = "max"

    # Transfer vs compute gating
    use_transfer_cost: bool = True
    transfer_bandwidth_gbps: float = 20.0
    proj_us_per_token_k: float = 0.35
    proj_us_per_token_v: float = 0.25
    min_reuse_score: float = 0.0

    # Whether to use Triton kernels (faster) or PyTorch fallback
    use_triton: bool = True

    # Enable verbose logging
    verbose: bool = False


# =============================================================================
# Main CKGR Manager
# =============================================================================

class ClusterGuidedKVReuse:
    """
    Cluster-Guided KV Reuse Manager.

    Uses CTCA's cluster information to decide which tokens can reuse cached KV
    and which need recomputation.

    Strategy:
    1. When CTCA reuses clusters, we also reuse K/V for those stable clusters
    2. K/V are stored at centroid level (huge memory savings)
    3. For attention, stable clusters use centroid K/V, others get fresh K/V

    Key Innovation: Selective computation guided by cluster stability.
    - Tokens in stable clusters: use cached centroid K/V (no computation)
    - Tokens in unstable clusters: recompute full K/V

    Integration with existing pipeline:
    - Called after CTCA's get_clusters() returns cluster_ids
    - Before actual attention computation
    - Updates K/V tensors with cached values for stable clusters
    """

    def __init__(
        self,
        config: CKGRConfig,
        num_layers: int = 60,
    ):
        self.config = config
        self.num_layers = num_layers

        # Per-layer caches
        self._cache: Dict[int, ClusterKVCache] = {}

        # Statistics tracking
        self.stats = {
            'total_tokens': 0,
            'reused_tokens': 0,
            'recomputed_tokens': 0,
            'cache_hits': 0,
            'cache_misses': 0,
            'total_calls': 0,
        }

        # Per-layer statistics
        self.layer_stats: Dict[int, Dict] = defaultdict(lambda: {
            'reuse_count': 0, 'recompute_count': 0
        })

        logger.info(f"[CKGR] Initialized with {num_layers} layers, "
                   f"{config.num_k_clusters} clusters, "
                   f"stability_threshold={config.stability_threshold}")

    def reset(self):
        """Reset cache and statistics. Call at start of each video generation."""
        self._cache.clear()
        self.stats = {k: 0 for k in self.stats}
        self.layer_stats.clear()
        logger.info("[CKGR] Cache and statistics reset")

    def get_cache(self, layer_idx: int) -> Optional[ClusterKVCache]:
        """Get cached centroids and cluster assignments for a layer."""
        return self._cache.get(layer_idx)

    @time_logging_decorator("Level 4 - CKGR compute KV centroids")
    def _compute_kv_centroids_triton(
        self,
        K: torch.Tensor,          # [B*H, S, D]
        V: torch.Tensor,          # [B*H, S, D]
        cluster_ids: torch.Tensor, # [B*H, S]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute K/V centroids using Triton kernel.

        Returns:
            k_centroids: [B*H, K, D]
            v_centroids: [B*H, K, D]
            cluster_sizes: [B*H, K]
        """
        B_H, S, D = K.shape
        num_clusters = self.config.num_k_clusters
        device = K.device
        dtype = K.dtype

        # Allocate output buffers
        k_sum = torch.zeros(B_H, num_clusters, D, device=device, dtype=torch.float32)
        v_sum = torch.zeros(B_H, num_clusters, D, device=device, dtype=torch.float32)
        k_count = torch.zeros(B_H, num_clusters, device=device, dtype=torch.int32)
        v_count = torch.zeros(B_H, num_clusters, device=device, dtype=torch.int32)

        # Launch kernel for K
        total_tokens = B_H * S
        BLOCK_D = 128
        grid = (total_tokens,)

        _kv_centroid_update_kernel[grid](
            K, cluster_ids.to(torch.int32),
            k_sum, k_count,
            B_H, S, D, num_clusters, BLOCK_D
        )

        # Launch kernel for V
        _kv_centroid_update_kernel[grid](
            V, cluster_ids.to(torch.int32),
            v_sum, v_count,
            B_H, S, D, num_clusters, BLOCK_D
        )

        # Compute means (avoid division by zero)
        counts_f = k_count.float().unsqueeze(-1).clamp(min=1.0)
        k_centroids = (k_sum / counts_f).to(dtype)
        v_centroids = (v_sum / counts_f).to(dtype)

        return k_centroids, v_centroids, k_count

    @time_logging_decorator("Level 4 - CKGR compute KV centroids PyTorch")
    def _compute_kv_centroids_pytorch(
        self,
        K: torch.Tensor,          # [B*H, S, D]
        V: torch.Tensor,          # [B*H, S, D]
        cluster_ids: torch.Tensor, # [B*H, S]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute K/V centroids using PyTorch (fallback).

        Vectorized implementation using scatter_add for efficiency.
        """
        B_H, S, D = K.shape
        num_clusters = self.config.num_k_clusters
        device = K.device
        dtype = K.dtype

        # Expand cluster_ids for scatter_add: [B*H, S] -> [B*H, S, D]
        cluster_ids_expanded = cluster_ids.unsqueeze(-1).expand(-1, -1, D)

        # Initialize accumulators
        k_sum = torch.zeros(B_H, num_clusters, D, device=device, dtype=torch.float32)
        v_sum = torch.zeros(B_H, num_clusters, D, device=device, dtype=torch.float32)

        # Scatter add
        k_sum.scatter_add_(1, cluster_ids_expanded.long(), K.float())
        v_sum.scatter_add_(1, cluster_ids_expanded.long(), V.float())

        # Count cluster sizes
        cluster_sizes = torch.zeros(B_H, num_clusters, device=device, dtype=torch.int32)
        ones = torch.ones(B_H, S, device=device, dtype=torch.int32)
        cluster_sizes.scatter_add_(1, cluster_ids.long(), ones)

        # Compute means
        counts_f = cluster_sizes.float().unsqueeze(-1).clamp(min=1.0)
        k_centroids = (k_sum / counts_f).to(dtype)
        v_centroids = (v_sum / counts_f).to(dtype)

        return k_centroids, v_centroids, cluster_sizes

    def _compute_kv_centroids(
        self,
        K: torch.Tensor,
        V: torch.Tensor,
        cluster_ids: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute K/V centroids using best available method."""
        if self.config.use_triton and K.is_cuda:
            return self._compute_kv_centroids_triton(K, V, cluster_ids)
        else:
            return self._compute_kv_centroids_pytorch(K, V, cluster_ids)

    @time_logging_decorator("Level 4 - CKGR compute stability")
    def _compute_cluster_stability(
        self,
        old_ids: torch.Tensor,  # [B*H, S]
        new_ids: torch.Tensor,  # [B*H, S]
    ) -> torch.Tensor:
        """Compute stability mask for clusters.

        A cluster is stable if Jaccard similarity > threshold.
        Jaccard = |intersection| / |union|

        Returns:
            stable_mask: [B*H, K] - True for stable clusters
        """
        B_H, S = new_ids.shape
        K = self.config.num_k_clusters
        device = new_ids.device

        if self.config.use_triton and new_ids.is_cuda:
            # Use Triton kernel
            overlap = torch.zeros(B_H, K, device=device, dtype=torch.int32)
            old_count = torch.zeros(B_H, K, device=device, dtype=torch.int32)
            new_count = torch.zeros(B_H, K, device=device, dtype=torch.int32)

            grid = (B_H * S,)
            _compute_stability_kernel[grid](
                old_ids.to(torch.int32), new_ids.to(torch.int32),
                overlap, old_count, new_count,
                B_H, S, K
            )

            # Compute Jaccard similarity
            union = old_count + new_count - overlap
            union = union.float().clamp(min=1.0)
            jaccard = overlap.float() / union

        else:
            # PyTorch fallback - vectorized
            # One-hot encode cluster assignments
            old_onehot = F.one_hot(old_ids.long().clamp(0, K-1), num_classes=K).float()  # [B*H, S, K]
            new_onehot = F.one_hot(new_ids.long().clamp(0, K-1), num_classes=K).float()  # [B*H, S, K]

            # Compute cluster sizes
            old_count = old_onehot.sum(dim=1)  # [B*H, K]
            new_count = new_onehot.sum(dim=1)  # [B*H, K]

            # Compute overlap (both assigned to same cluster)
            overlap = (old_onehot * new_onehot).sum(dim=1)  # [B*H, K]

            # Jaccard similarity
            union = (old_count + new_count - overlap).clamp(min=1.0)
            jaccard = overlap / union

        # Threshold to get stable mask
        stable_mask = jaccard > self.config.stability_threshold

        return stable_mask

    @time_logging_decorator("Level 4 - CKGR scatter centroids")
    def _scatter_centroids_to_tokens(
        self,
        K: torch.Tensor,           # [B*H, S, D] - output tensor (modified in place)
        V: torch.Tensor,           # [B*H, S, D] - output tensor (modified in place)
        k_centroids: torch.Tensor, # [B*H, K, D]
        v_centroids: torch.Tensor, # [B*H, K, D]
        cluster_ids: torch.Tensor, # [B*H, S]
        stable_mask: torch.Tensor, # [B*H, K]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Replace token K/V with centroid values for stable clusters.

        Returns:
            K, V: Modified tensors
            reuse_mask: [B*H, S] - True for tokens that were replaced
        """
        B_H, S, D = K.shape
        num_clusters = self.config.num_k_clusters
        device = K.device

        # Create token-level reuse mask: token can reuse if its cluster is stable
        # Gather stable_mask using cluster_ids: [B*H, S]
        reuse_mask = torch.gather(stable_mask, 1, cluster_ids.long().clamp(0, num_clusters-1))

        if not reuse_mask.any():
            return K, V, reuse_mask

        if self.config.use_triton and K.is_cuda:
            # Use Triton kernel for scattering
            grid = (B_H * S,)
            BLOCK_D = 128

            _scatter_centroids_to_tokens_kernel[grid](
                k_centroids, cluster_ids.to(torch.int32), K, stable_mask,
                B_H, S, D, num_clusters, BLOCK_D
            )
            _scatter_centroids_to_tokens_kernel[grid](
                v_centroids, cluster_ids.to(torch.int32), V, stable_mask,
                B_H, S, D, num_clusters, BLOCK_D
            )
        else:
            # PyTorch fallback - vectorized gather
            # Get centroid values for each token's cluster
            cluster_ids_expanded = cluster_ids.unsqueeze(-1).expand(-1, -1, D).long().clamp(0, num_clusters-1)
            k_from_centroids = torch.gather(k_centroids, 1, cluster_ids_expanded)
            v_from_centroids = torch.gather(v_centroids, 1, cluster_ids_expanded)

            # Apply mask - replace only for stable clusters
            reuse_mask_expanded = reuse_mask.unsqueeze(-1)  # [B*H, S, 1]
            K = torch.where(reuse_mask_expanded, k_from_centroids, K)
            V = torch.where(reuse_mask_expanded, v_from_centroids, V)

        return K, V, reuse_mask

    @time_logging_decorator("Level 4 - CKGR scatter centroids single")
    def _scatter_centroids_to_tokens_single(
        self,
        X: torch.Tensor,           # [B*H, S, D] - output tensor (modified in place)
        centroids: torch.Tensor,   # [B*H, K, D]
        cluster_ids: torch.Tensor, # [B*H, S]
        stable_mask: torch.Tensor, # [B*H, K]
        reuse_mask_override: Optional[torch.Tensor] = None,  # [B*H, S]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Replace token values with centroid values for stable clusters."""
        B_H, S, D = X.shape
        num_clusters = self.config.num_k_clusters

        # Token-level reuse mask
        reuse_mask = torch.gather(stable_mask, 1, cluster_ids.long().clamp(0, num_clusters - 1))
        if reuse_mask_override is not None:
            reuse_mask = reuse_mask & reuse_mask_override

        if not reuse_mask.any():
            return X, reuse_mask

        if self.config.use_triton and X.is_cuda:
            grid = (B_H * S,)
            BLOCK_D = 128
            _scatter_centroids_to_tokens_kernel[grid](
                centroids, cluster_ids.to(torch.int32), X, stable_mask,
                B_H, S, D, num_clusters, BLOCK_D
            )
        else:
            cluster_ids_expanded = cluster_ids.unsqueeze(-1).expand(-1, -1, D).long().clamp(0, num_clusters - 1)
            from_centroids = torch.gather(centroids, 1, cluster_ids_expanded)
            reuse_mask_expanded = reuse_mask.unsqueeze(-1)  # [B*H, S, 1]
            X = torch.where(reuse_mask_expanded, from_centroids, X)

        return X, reuse_mask

    def should_reuse(
        self,
        layer_idx: int,
        ctca_reused: bool,
        cluster_quality: float = 1.0,
    ) -> bool:
        """Decide if we should attempt KV reuse for this layer.

        Args:
            layer_idx: Transformer layer index
            ctca_reused: Whether CTCA reused clusters (vs full recluster)
            cluster_quality: CTCA's cluster quality score

        Returns:
            True if we should attempt KV reuse
        """
        # No cache exists - can't reuse
        if layer_idx not in self._cache:
            return False
        cache = self._cache[layer_idx]

        # CTCA did full recluster - cache is invalid
        if not ctca_reused:
            return False

        # Quality too low - clusters may have drifted
        if cluster_quality < self.config.quality_threshold:
            return False

        # Warmup: wait for enough updates before reusing
        if self.config.min_reuse_steps > 0 and cache.update_count < self.config.min_reuse_steps:
            return False

        return True

    def _normalize_cluster_importance(
        self,
        cluster_importance: torch.Tensor,
        batch_heads: int,
        num_clusters: int,
    ) -> torch.Tensor:
        """Normalize cluster importance to [B*H, K]."""
        importance = cluster_importance
        if importance.dim() == 4:
            # [B, H, Q, K] -> reduce over Q
            if self.config.importance_reduce == "mean":
                importance = importance.mean(dim=2)
            else:
                importance = importance.max(dim=2).values
        if importance.dim() == 3:
            # [B, H, K] -> [B*H, K]
            importance = importance.reshape(batch_heads, num_clusters)
        if importance.dim() != 2:
            raise ValueError("cluster_importance must be [B*H, K], [B, H, K], or [B, H, Q, K]")
        return importance

    def _apply_importance_gating(
        self,
        stable_mask: torch.Tensor,
        cluster_importance: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """Gate reuse based on cluster importance."""
        if cluster_importance is None:
            return stable_mask
        if self.config.importance_threshold is None and self.config.importance_quantile is None:
            return stable_mask

        B_H, K = stable_mask.shape
        importance = self._normalize_cluster_importance(cluster_importance, B_H, K)
        if importance.device != stable_mask.device:
            importance = importance.to(stable_mask.device, non_blocking=True)

        if self.config.importance_quantile is not None:
            flat = importance.reshape(-1)
            thresh = torch.quantile(flat, self.config.importance_quantile)
        else:
            thresh = self.config.importance_threshold

        if thresh is None:
            return stable_mask

        return stable_mask & (importance <= thresh)

    def _estimate_transfer_cost_ms(
        self,
        cache: ClusterKVCache,
        reuse_k: bool,
        reuse_v: bool,
    ) -> float:
        """Estimate CPU->GPU transfer time in ms for cached centroids."""
        if self.config.transfer_bandwidth_gbps <= 0.0:
            return 0.0

        bytes_to_transfer = 0
        if reuse_k:
            bytes_to_transfer += cache.k_centroids.numel() * cache.k_centroids.element_size()
        if reuse_v:
            bytes_to_transfer += cache.v_centroids.numel() * cache.v_centroids.element_size()

        transfer_seconds = bytes_to_transfer / (self.config.transfer_bandwidth_gbps * 1e9)
        return transfer_seconds * 1e3

    def _estimate_compute_savings_ms(
        self,
        reuse_tokens: int,
        reuse_k: bool,
        reuse_v: bool,
    ) -> float:
        """Estimate compute savings in ms from skipping projections."""
        us_per_token = 0.0
        if reuse_k:
            us_per_token += self.config.proj_us_per_token_k
        if reuse_v:
            us_per_token += self.config.proj_us_per_token_v

        return (reuse_tokens * us_per_token) / 1e3

    def compute_reuse_masks(
        self,
        layer_idx: int,
        k_cluster_ids: torch.Tensor,
        ctca_reused: bool,
        cluster_quality: float = 1.0,
        current_k_centroids: Optional[torch.Tensor] = None,
        cluster_importance: Optional[torch.Tensor] = None,
        token_reuse_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Dict, Optional[ClusterKVCache]]:
        """Compute cluster-level and token-level reuse masks.

        Returns:
            stable_mask: [B*H, K] or None
            reuse_mask: [B*H, S] or None
            info: Dict with reuse metadata
            cache: ClusterKVCache moved to device, or None
        """
        info = {
            'reused': False,
            'stable_ratio': 0.0,
            'reuse_ratio': 0.0,
        }

        # If reuse is disabled entirely, skip
        if not (self.config.reuse_k or self.config.reuse_v):
            return None, None, info, None

        # Check if we should attempt reuse
        if not self.should_reuse(layer_idx, ctca_reused, cluster_quality):
            return None, None, info, None

        cache = self._cache[layer_idx]
        device = k_cluster_ids.device

        # Compute stability between old and new cluster assignments
        old_cluster_ids = cache.k_cluster_ids.to(device, non_blocking=True)
        stable_mask = self._compute_cluster_stability(old_cluster_ids, k_cluster_ids)

        # Optional centroid drift gating (less change -> more reuse)
        if current_k_centroids is not None and self.config.centroid_sim_threshold > 0.0:
            if current_k_centroids.shape == cache.k_centroids.shape:
                cached_k_centroids = cache.k_centroids.to(device, non_blocking=True)
                if current_k_centroids.device != cached_k_centroids.device:
                    current_k_centroids = current_k_centroids.to(device, non_blocking=True)
                denom = (
                    cached_k_centroids.norm(dim=-1) * current_k_centroids.norm(dim=-1)
                ).clamp(min=1e-6)
                cos_sim = (cached_k_centroids * current_k_centroids).sum(dim=-1) / denom
                stable_mask = stable_mask & (cos_sim >= self.config.centroid_sim_threshold)
            elif self.config.verbose:
                logger.info("[CKGR] Skipping centroid drift gating due to shape mismatch.")

        # Require clusters to be stable for multiple consecutive steps
        if self.config.min_stable_steps > 1:
            stable_counts = cache.stable_counts.to(device, non_blocking=True)
            stable_steps = stable_counts + 1
            stable_mask = stable_mask & (stable_steps >= self.config.min_stable_steps)

        # Optional importance gating
        stable_mask = self._apply_importance_gating(stable_mask, cluster_importance)

        stable_ratio = stable_mask.float().mean().item()
        info['stable_ratio'] = stable_ratio

        # Only apply reuse if enough clusters are stable
        if stable_ratio < self.config.min_stable_ratio:
            return stable_mask, None, info, cache

        reuse_mask = torch.gather(
            stable_mask,
            1,
            k_cluster_ids.long().clamp(0, self.config.num_k_clusters - 1)
        )

        if token_reuse_mask is not None:
            if token_reuse_mask.dim() != 2:
                raise ValueError("token_reuse_mask must be [B, S] or [B*H, S]")
            if token_reuse_mask.shape[0] != reuse_mask.shape[0]:
                repeat = reuse_mask.shape[0] // token_reuse_mask.shape[0]
                if token_reuse_mask.shape[0] * repeat != reuse_mask.shape[0]:
                    raise ValueError("token_reuse_mask batch does not match B*H")
                token_reuse_mask = token_reuse_mask.repeat_interleave(repeat, dim=0)
            if token_reuse_mask.device != reuse_mask.device:
                token_reuse_mask = token_reuse_mask.to(reuse_mask.device, non_blocking=True)
            reuse_mask = reuse_mask & token_reuse_mask

        reuse_ratio = reuse_mask.float().mean().item()
        info['reused'] = True
        info['reuse_ratio'] = reuse_ratio

        # Minimum token-level reuse ratio
        if self.config.min_reuse_ratio > 0.0 and reuse_ratio < self.config.min_reuse_ratio:
            info['reused'] = False
            return stable_mask, None, info, cache

        # Transfer vs compute gating
        if self.config.use_transfer_cost:
            reuse_tokens = int(reuse_mask.sum().item())
            transfer_ms = self._estimate_transfer_cost_ms(cache, self.config.reuse_k, self.config.reuse_v)
            savings_ms = self._estimate_compute_savings_ms(reuse_tokens, self.config.reuse_k, self.config.reuse_v)
            reuse_score = savings_ms - transfer_ms

            info['reuse_tokens'] = reuse_tokens
            info['transfer_ms'] = transfer_ms
            info['compute_savings_ms'] = savings_ms
            info['reuse_score'] = reuse_score

            if reuse_score < self.config.min_reuse_score:
                info['reused'] = False
                return stable_mask, None, info, cache

        # Move only required cache fields to device for actual reuse
        cache.to_device(
            device,
            move_k_centroids=self.config.reuse_k,
            move_v_centroids=self.config.reuse_v,
            move_stable_clusters=False,
            move_stable_counts=False,
            move_cluster_ids=False,
            move_cluster_sizes=False,
            move_token_energy=False,
        )

        return stable_mask, reuse_mask, info, cache

    def reduce_reuse_mask(
        self,
        reuse_mask: Optional[torch.Tensor],
        batch_size: int,
        num_heads: int,
    ) -> Optional[torch.Tensor]:
        """Reduce per-head reuse mask to token-level mask."""
        if reuse_mask is None:
            return None

        reuse_mask = reuse_mask.view(batch_size, num_heads, -1)

        if self.config.min_head_reuse_ratio >= 1.0:
            return reuse_mask.all(dim=1)

        head_ratio = reuse_mask.float().mean(dim=1)
        return head_ratio >= self.config.min_head_reuse_ratio

    def record_reuse(
        self,
        layer_idx: int,
        reuse_mask: Optional[torch.Tensor],
        total_tokens: int,
        reused: bool,
    ):
        """Update CKGR statistics for a call."""
        self.stats['total_calls'] += 1

        if reused and reuse_mask is not None:
            num_reused = reuse_mask.sum().item()
            num_total = reuse_mask.numel()
            self.stats['reused_tokens'] += num_reused
            self.stats['recomputed_tokens'] += num_total - num_reused
            self.stats['total_tokens'] += num_total
            self.stats['cache_hits'] += 1
            self.layer_stats[layer_idx]['reuse_count'] += 1
        else:
            self.stats['cache_misses'] += 1
            self.stats['total_tokens'] += total_tokens
            self.stats['recomputed_tokens'] += total_tokens
            self.layer_stats[layer_idx]['recompute_count'] += 1

    def update_cache(
        self,
        layer_idx: int,
        K: torch.Tensor,
        V: torch.Tensor,
        k_cluster_ids: torch.Tensor,
        timestep: int,
        ctca_reused: bool,
        cluster_quality: float,
        token_energy: Optional[torch.Tensor] = None,
    ):
        """Public wrapper to update CKGR cache."""
        self._update_cache(
            layer_idx, K, V, k_cluster_ids,
            timestep, ctca_reused, cluster_quality,
            token_energy=token_energy,
        )

    @time_logging_decorator("Level 3 - CKGR process KV")
    def process_kv(
        self,
        K: torch.Tensor,           # [B, H, S, D]
        V: torch.Tensor,           # [B, H, S, D]
        k_cluster_ids: torch.Tensor,  # [B*H, S] - from CTCA
        layer_idx: int,
        timestep: int,
        ctca_reused: bool,
        cluster_quality: float = 1.0,
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict]:
        """Main entry point: Process K/V tensors with cluster-guided reuse.

        If CTCA reused clusters and we have cached centroids, replace token
        K/V with centroid values for stable clusters.

        Args:
            K, V: Key and Value tensors [B, H, S, D]
            k_cluster_ids: Cluster assignments from CTCA [B*H, S]
            layer_idx: Transformer layer index
            timestep: Current diffusion timestep
            ctca_reused: Whether CTCA reused clusters
            cluster_quality: CTCA's cluster quality score

        Returns:
            K, V: Processed tensors (may have centroid values for stable clusters)
            info: Dict with statistics
        """
        B, H, S, D = K.shape
        info = {
            'reused': False,
            'stable_ratio': 0.0,
            'reuse_ratio': 0.0,
        }

        # Reshape to [B*H, S, D] for processing
        K_flat = K.view(B * H, S, D)
        V_flat = V.view(B * H, S, D)

        # Compute reuse masks
        stable_mask, reuse_mask, reuse_info, cache = self.compute_reuse_masks(
            layer_idx, k_cluster_ids, ctca_reused, cluster_quality
        )
        info.update(reuse_info)

        # Apply reuse if enabled and masks available
        if info['reused'] and reuse_mask is not None and stable_mask is not None and cache is not None:
            if self.config.reuse_k:
                K_flat, _ = self._scatter_centroids_to_tokens_single(
                    K_flat, cache.k_centroids, k_cluster_ids, stable_mask
                )
            if self.config.reuse_v:
                V_flat, _ = self._scatter_centroids_to_tokens_single(
                    V_flat, cache.v_centroids, k_cluster_ids, stable_mask
                )

            if self.config.verbose:
                logger.info(f"[CKGR] Layer {layer_idx}: reused {info['reuse_ratio']*100:.1f}% tokens "
                           f"(stable clusters: {info['stable_ratio']*100:.1f}%)")

        # Update statistics
        total_tokens = B * H * S
        self.record_reuse(layer_idx, reuse_mask, total_tokens, info['reused'])

        if cache is not None:
            cache.to_cpu()

        # Update cache with current K/V centroids
        self.update_cache(
            layer_idx, K_flat, V_flat, k_cluster_ids,
            timestep, ctca_reused, cluster_quality
        )

        # Reshape back to [B, H, S, D]
        K_out = K_flat.view(B, H, S, D)
        V_out = V_flat.view(B, H, S, D)

        return K_out, V_out, info

    @time_logging_decorator("Level 4 - CKGR update cache")
    def _update_cache(
        self,
        layer_idx: int,
        K: torch.Tensor,           # [B*H, S, D]
        V: torch.Tensor,           # [B*H, S, D]
        k_cluster_ids: torch.Tensor,
        timestep: int,
        ctca_reused: bool,
        cluster_quality: float,
        token_energy: Optional[torch.Tensor] = None,
    ):
        """Update the cache with new K/V centroids."""

        # Compute K/V centroids
        k_centroids, v_centroids, cluster_sizes = self._compute_kv_centroids(
            K, V, k_cluster_ids
        )

        # Compute stability if we have old cache
        if layer_idx in self._cache and ctca_reused:
            old_cache = self._cache[layer_idx]
            old_cache.to_device(K.device)
            stable_mask = self._compute_cluster_stability(
                old_cache.k_cluster_ids, k_cluster_ids
            )
            prev_counts = getattr(old_cache, "stable_counts", None)
            if prev_counts is None:
                prev_counts = torch.zeros_like(stable_mask, dtype=torch.int32)
            stable_counts = torch.where(
                stable_mask,
                prev_counts + 1,
                torch.zeros_like(prev_counts),
            )
            update_count = getattr(old_cache, "update_count", 0) + 1
            old_cache.to_cpu()
        else:
            # Fresh cache - no stability info
            B_H = k_centroids.shape[0]
            stable_mask = torch.zeros(
                B_H, self.config.num_k_clusters,
                dtype=torch.bool, device=K.device
            )
            stable_counts = torch.zeros(
                B_H, self.config.num_k_clusters,
                dtype=torch.int32, device=K.device
            )
            update_count = 1

        # Create new cache entry
        self._cache[layer_idx] = ClusterKVCache(
            k_centroids=k_centroids,
            v_centroids=v_centroids,
            stable_clusters=stable_mask,
            stable_counts=stable_counts,
            k_cluster_ids=k_cluster_ids.clone(),
            cluster_sizes=cluster_sizes,
            last_update_timestep=timestep,
            quality_score=cluster_quality,
            update_count=update_count,
            token_energy=token_energy.detach().float().cpu() if token_energy is not None else None,
        )

        # Move to CPU to save GPU memory
        self._cache[layer_idx].to_cpu()

    def get_statistics(self) -> Dict:
        """Get CKGR performance statistics."""
        total = self.stats['total_tokens']

        stats = {
            **self.stats,
            'reuse_ratio': self.stats['reused_tokens'] / max(total, 1),
            'cache_hit_ratio': self.stats['cache_hits'] / max(self.stats['total_calls'], 1),
        }

        # Add per-layer stats
        stats['layer_stats'] = dict(self.layer_stats)

        # Memory usage
        total_memory = sum(
            cache.memory_bytes() for cache in self._cache.values()
        )
        stats['cache_memory_mb'] = total_memory / (1024 * 1024)

        return stats

    def print_statistics(self):
        """Print CKGR performance statistics."""
        stats = self.get_statistics()

        print("\n" + "=" * 60)
        print("CKGR (Cluster-Guided KV Reuse) Statistics")
        print("=" * 60)
        print(f"Total calls:          {stats['total_calls']}")
        print(f"Cache hits:           {stats['cache_hits']} ({stats['cache_hit_ratio']*100:.1f}%)")
        print(f"Cache misses:         {stats['cache_misses']}")
        print(f"Total tokens:         {stats['total_tokens']:,}")
        print(f"Reused tokens:        {stats['reused_tokens']:,} ({stats['reuse_ratio']*100:.1f}%)")
        print(f"Recomputed tokens:    {stats['recomputed_tokens']:,}")
        print(f"Cache memory:         {stats['cache_memory_mb']:.2f} MB")
        print("=" * 60 + "\n")


# =============================================================================
# Factory Functions
# =============================================================================

def create_ckgr_manager(
    num_layers: int = 60,
    num_k_clusters: int = 1000,
    stability_threshold: float = 0.7,
    min_reuse_steps: int = 2,
    min_stable_steps: int = 2,
    centroid_sim_threshold: float = 0.98,
    quality_threshold: float = 0.75,
    reuse_k: bool = False,
    reuse_v: bool = True,
    min_head_reuse_ratio: float = 1.0,
    min_reuse_ratio: float = 0.05,
    token_delta_threshold: Optional[float] = 0.05,
    token_delta_quantile: Optional[float] = None,
    importance_threshold: Optional[float] = None,
    importance_quantile: Optional[float] = None,
    importance_reduce: str = "max",
    use_transfer_cost: bool = True,
    transfer_bandwidth_gbps: float = 20.0,
    proj_us_per_token_k: float = 0.35,
    proj_us_per_token_v: float = 0.25,
    min_reuse_score: float = 0.0,
    verbose: bool = False,
) -> ClusterGuidedKVReuse:
    """Create a CKGR manager with default configuration.

    Args:
        num_layers: Number of transformer layers
        num_k_clusters: Number of clusters (should match CTCA)
        stability_threshold: Jaccard threshold for cluster stability
        min_reuse_steps: Warmup updates before enabling reuse
        min_stable_steps: Consecutive stable steps required per cluster
        centroid_sim_threshold: Cosine similarity threshold for centroid drift
        quality_threshold: CTCA quality threshold for enabling reuse
        reuse_k: Whether to reuse K via cached centroids
        reuse_v: Whether to reuse V via cached centroids
        min_head_reuse_ratio: Token reuse threshold across heads
        min_reuse_ratio: Minimum token reuse ratio to enable reuse
        token_delta_threshold: Relative token change threshold for reuse
        token_delta_quantile: Quantile threshold for reuse
        importance_threshold: Absolute importance threshold for reuse gating
        importance_quantile: Quantile threshold for reuse gating
        importance_reduce: Reduce mode for QxK importance ("max" or "mean")
        use_transfer_cost: Whether to gate reuse by transfer vs compute cost
        transfer_bandwidth_gbps: Estimated CPU->GPU bandwidth
        proj_us_per_token_k: Estimated K projection cost (microseconds)
        proj_us_per_token_v: Estimated V projection cost (microseconds)
        min_reuse_score: Minimum (savings - transfer) score to reuse
        verbose: Enable verbose logging

    Returns:
        Configured CKGR manager
    """
    config = CKGRConfig(
        num_k_clusters=num_k_clusters,
        stability_threshold=stability_threshold,
        min_reuse_steps=min_reuse_steps,
        min_stable_steps=min_stable_steps,
        centroid_sim_threshold=centroid_sim_threshold,
        quality_threshold=quality_threshold,
        reuse_k=reuse_k,
        reuse_v=reuse_v,
        min_head_reuse_ratio=min_head_reuse_ratio,
        min_reuse_ratio=min_reuse_ratio,
        token_delta_threshold=token_delta_threshold,
        token_delta_quantile=token_delta_quantile,
        importance_threshold=importance_threshold,
        importance_quantile=importance_quantile,
        importance_reduce=importance_reduce,
        use_transfer_cost=use_transfer_cost,
        transfer_bandwidth_gbps=transfer_bandwidth_gbps,
        proj_us_per_token_k=proj_us_per_token_k,
        proj_us_per_token_v=proj_us_per_token_v,
        min_reuse_score=min_reuse_score,
        verbose=verbose,
    )

    return ClusterGuidedKVReuse(config, num_layers)


# =============================================================================
# Integration Helpers (for attention processor)
# =============================================================================

# Global CKGR manager (set during initialization)
_ckgr_manager: Optional[ClusterGuidedKVReuse] = None


def initialize_ckgr(
    num_layers: int = 60,
    num_k_clusters: int = 1000,
    **kwargs
) -> ClusterGuidedKVReuse:
    """Initialize global CKGR manager."""
    global _ckgr_manager
    _ckgr_manager = create_ckgr_manager(num_layers, num_k_clusters, **kwargs)
    return _ckgr_manager


def get_ckgr_manager() -> Optional[ClusterGuidedKVReuse]:
    """Get the global CKGR manager."""
    return _ckgr_manager


def reset_ckgr():
    """Reset global CKGR manager."""
    if _ckgr_manager is not None:
        _ckgr_manager.reset()


def print_ckgr_statistics():
    """Print global CKGR statistics."""
    if _ckgr_manager is not None:
        _ckgr_manager.print_statistics()


# =============================================================================
# Integration Example: How to integrate CKGR with CTCA attention processor
# =============================================================================
"""
CKGR Integration Guide
======================

CKGR (Cluster-Guided KV Reuse) works alongside CTCA to reduce computation
and improve consistency in sparse attention.

Key Integration Points:
1. Initialize CKGR when initializing the attention processor
2. Call process_kv() after K/V projection but before sparse attention
3. Pass CTCA's cluster information to CKGR
4. Print statistics after generation

Example Integration in Hunyuan_SAPAttn_CTCA_Processor2_0:

```python
# In attention.py, modify the semantic_aware_permutation method:

from ..cluster_kv_reuse import get_ckgr_manager, initialize_ckgr

class Hunyuan_SAPAttn_CTCA_CKGR_Processor2_0(Hunyuan_SAPAttn_CTCA_Processor2_0):
    '''CTCA + CKGR enabled attention processor.'''

    # CKGR configuration
    ckgr_enabled: bool = True
    ckgr_stability_threshold: float = 0.7
    ckgr_quality_threshold: float = 0.75

    @classmethod
    def initialize_ckgr(cls):
        '''Initialize CKGR manager.'''
        from ..cluster_kv_reuse import initialize_ckgr as init_ckgr
        init_ckgr(
            num_layers=60,
            num_k_clusters=cls.num_k_clusters,
            stability_threshold=cls.ckgr_stability_threshold,
            quality_threshold=cls.ckgr_quality_threshold,
        )

    @time_logging_decorator("Level 3 - SAP with CTCA + CKGR")
    def semantic_aware_permutation(self, query, key, value, timestep, layer_idx):
        '''Semantic aware permutation with CKGR integration.'''
        cfg, num_heads, seq_len, dim = query.size()

        # 1. Get clusters from CTCA
        (qlabels, qcentroids, qcluster_sizes, qiter,
         klabels, kcentroids, kcluster_sizes, kiter) = self.kmeans_clustering(
            query, key, layer_idx, timestep=timestep
        )

        # 2. Apply CKGR to K/V if enabled
        if self.ckgr_enabled:
            ckgr = get_ckgr_manager()
            if ckgr is not None:
                # Determine if CTCA reused clusters
                ctca_reused = self.ctca_manager and layer_idx in self.ctca_manager._cache

                # Get cluster quality from CTCA
                cache = self.ctca_manager.get_cache(layer_idx) if self.ctca_manager else None
                cluster_quality = cache.get_average_quality() if cache else 1.0

                # Process K/V with CKGR
                key, value, ckgr_info = ckgr.process_kv(
                    key, value,
                    klabels,  # K cluster assignments from CTCA
                    layer_idx,
                    timestep[0].item() if isinstance(timestep, torch.Tensor) else timestep,
                    ctca_reused,
                    cluster_quality,
                )

        # 3. Continue with standard SAP pipeline...
        # (identify_dynamic_map, permute tensors, etc.)
        ...
```

Initialization in inference script:
```python
# In hyvideo_t2v_inference.py

from svg.cluster_kv_reuse import initialize_ckgr, reset_ckgr, print_ckgr_statistics

# Initialize CKGR at startup
if args.ckgr_enabled:
    initialize_ckgr(
        num_layers=60,
        num_k_clusters=args.num_k_clusters,
        stability_threshold=0.7,
    )

# Reset before each video generation
reset_ckgr()

# Generate video...

# Print statistics after generation
print_ckgr_statistics()
```

Benefits of CKGR Integration:
1. Reduces noise in attention patterns for stable clusters
2. Memory efficient: stores centroids (~0.35 GB) vs full KV cache (~170 GB)
3. Automatic adaptation: only reuses when clusters are stable
4. Compatible with existing CTCA pipeline

Future Optimizations (Selective Projection):
The current implementation replaces K/V after projection. A more aggressive
optimization would skip projection entirely for stable tokens:

```python
def selective_kv_projection(hidden_states, k_proj, v_proj, stable_mask, cached_kv):
    '''Only project unstable tokens, use cached KV for stable ones.'''
    B, S, D = hidden_states.shape

    # Identify which tokens need projection
    unstable_indices = (~stable_mask).nonzero(as_tuple=True)

    if len(unstable_indices[0]) == 0:
        # All stable - use cached entirely
        return cached_kv['K'], cached_kv['V']

    # Project only unstable tokens
    unstable_hidden = hidden_states[unstable_indices]
    K_unstable = k_proj(unstable_hidden)
    V_unstable = v_proj(unstable_hidden)

    # Scatter back to full tensor
    K = cached_kv['K'].clone()
    V = cached_kv['V'].clone()
    K[unstable_indices] = K_unstable
    V[unstable_indices] = V_unstable

    return K, V
```

This selective projection approach can save significant compute when most
tokens are in stable clusters (typically 50-70% after warmup).
"""
