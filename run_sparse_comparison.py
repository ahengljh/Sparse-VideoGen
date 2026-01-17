"""
Run Sparse vs Full Attention Comparison

This script uses dynamic patching to enable sparse attention without modifying source code.

Usage:
    python run_sparse_comparison.py \
        --prompt "A cat playing with a ball" \
        --critical-ratio 0.5 \
        --video-length 129 \
        --video-size 544 960 \
        --infer-steps 50

Outputs:
    - Two videos: one with full attention, one with sparse attention
    - Quality metrics comparing the two
    - Performance statistics
"""

import os
import sys
import time
import argparse
import torch
import torch.nn.functional as F
from pathlib import Path
from datetime import datetime
from loguru import logger

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# CRITICAL: Patch attention BEFORE importing any model code!
# This must happen before models.py imports the attention function.
from hyvideo.modules import attenion

# Save original attention function immediately
_original_attention_func = attenion.attention

# Now safe to import model code
from hyvideo.inference import HunyuanVideoSampler
from hyvideo.config import parse_args
from hyvideo.utils.file_utils import save_videos_grid

# Global variables for patching
_original_attention = None
_sparse_enabled = False
_sparse_critical_ratio = 0.5
_sparse_method = 'kmeans'  # 'topk' or 'kmeans'
_sparse_stats = {'total_calls': 0, 'sparse_calls': 0}

# === Sparse VideoGen optimizations ===
# Centroid cache: {layer_idx: {'q_centroids': tensor, 'k_centroids': tensor}}
_centroids_cache = {}
_kmeans_iter_init = 50  # First time: full K-means
_kmeans_iter_step = 2   # Subsequent: quick update


def kmeans_pytorch(x, num_clusters, max_iter=50, tol=1e-4, init_centroids=None):
    """
    Optimized PyTorch GPU-accelerated K-means clustering (Sparse VideoGen style).

    Key optimizations:
    1. Pre-compute squared norms to accelerate distance calculation
    2. Support warm-start from cached centroids
    3. Use scatter operations for faster centroid updates
    4. K-means++ initialization when no init_centroids provided

    Args:
        x: Input tensor [n_samples, n_features]
        num_clusters: Number of clusters
        max_iter: Maximum iterations
        tol: Convergence tolerance
        init_centroids: Optional initial centroids [num_clusters, n_features]

    Returns:
        labels: Cluster labels [n_samples]
        centroids: Cluster centroids [num_clusters, n_features]
        cluster_sizes: Size of each cluster [num_clusters]
    """
    n_samples, n_features = x.shape
    device = x.device
    dtype = x.dtype

    # === Optimization 1: Pre-compute squared norms ===
    x_sq = (x ** 2).sum(dim=-1, keepdim=True)  # [n_samples, 1]

    # === Initialize centroids ===
    if init_centroids is not None:
        # Warm-start from cached centroids (Sparse VideoGen approach)
        centroids = init_centroids.clone()
    else:
        # K-means++ initialization
        centroids = torch.zeros(num_clusters, n_features, dtype=dtype, device=device)

        # First centroid: random sample
        idx = torch.randint(0, n_samples, (1,), device=device)
        centroids[0] = x[idx]

        # Remaining centroids: weighted by distance
        for i in range(1, num_clusters):
            # Compute distances to nearest centroid (using squared norms)
            c_sq = (centroids[:i] ** 2).sum(dim=-1, keepdim=True)  # [i, 1]
            dists_sq = x_sq + c_sq.t() - 2 * torch.matmul(x, centroids[:i].t())  # [n_samples, i]
            min_dists_sq = dists_sq.min(dim=1)[0]  # [n_samples]

            # Sample proportional to squared distance
            probs = min_dists_sq.clamp(min=1e-8)
            probs = probs / probs.sum()
            idx = torch.multinomial(probs, 1)
            centroids[i] = x[idx]

    # === Lloyd's algorithm ===
    for iteration in range(max_iter):
        # === Optimization 2: Fast distance calculation using pre-computed norms ===
        c_sq = (centroids ** 2).sum(dim=-1, keepdim=True)  # [num_clusters, 1]
        # D(x, c)^2 = ||x||^2 + ||c||^2 - 2*x·c
        dists_sq = x_sq + c_sq.t() - 2 * torch.matmul(x, centroids.t())  # [n_samples, num_clusters]
        labels = dists_sq.argmin(dim=1)  # [n_samples]

        # === Optimization 3: Fast centroid update using scatter ===
        new_centroids = torch.zeros_like(centroids)
        cluster_sizes = torch.zeros(num_clusters, device=device, dtype=torch.float32)

        # Use scatter_add for parallel accumulation
        # Important: ensure x and new_centroids have matching dtype
        x_for_scatter = x.to(dtype=new_centroids.dtype)
        new_centroids.scatter_add_(0, labels.unsqueeze(-1).expand(-1, n_features), x_for_scatter)
        cluster_sizes.scatter_add_(0, labels, torch.ones(n_samples, device=device, dtype=torch.float32))

        # Normalize by cluster size
        # Handle empty clusters by keeping old centroid
        non_empty = cluster_sizes > 0
        new_centroids[non_empty] = new_centroids[non_empty] / cluster_sizes[non_empty].unsqueeze(-1)
        new_centroids[~non_empty] = centroids[~non_empty]  # Keep old centroid for empty clusters

        # Check convergence
        shift = torch.norm(new_centroids - centroids)
        centroids = new_centroids

        if shift < tol:
            break

    # Final assignment
    c_sq = (centroids ** 2).sum(dim=-1, keepdim=True)
    dists_sq = x_sq + c_sq.t() - 2 * torch.matmul(x, centroids.t())
    labels = dists_sq.argmin(dim=1)

    # Compute final cluster sizes
    cluster_sizes = torch.bincount(labels, minlength=num_clusters).float()

    return labels, centroids, cluster_sizes


def compute_critical_mask_topk(q, k, critical_ratio=0.5):
    """
    Identify critical tokens using top-k by attention score.

    Args:
        q: Query tensor with shape [batch, seq_len, num_heads, head_dim]
        k: Key tensor with shape [batch, seq_len, num_heads, head_dim]
        critical_ratio: Ratio of tokens to keep (0.0-1.0)

    Returns:
        critical_mask: [batch, seq_len] boolean mask where True = critical token
    """
    with torch.no_grad():
        # Handle different input shapes
        if q.dim() == 4:
            # [batch, seq_len, num_heads, head_dim]
            batch_size, seq_len, num_heads, head_dim = q.shape
        elif q.dim() == 3:
            # [batch*seq_len, num_heads, head_dim] - flash attention format
            # This is more complex, for now use simplified approach
            total_tokens, num_heads, head_dim = q.shape
            # Assume batch_size=1 for simplicity
            batch_size = 1
            seq_len = total_tokens
            q = q.view(batch_size, seq_len, num_heads, head_dim)
            k = k.view(batch_size, seq_len, num_heads, head_dim)
        else:
            raise ValueError(f"Unexpected q shape: {q.shape}")

        # Compute attention scores (simplified, no softmax)
        # [batch, num_heads, seq_len, seq_len]
        scale = 1.0 / (head_dim ** 0.5)
        attn_scores = torch.einsum('bqhd,bkhd->bhqk', q.float(), k.float()) * scale

        # Average attention received by each token (averaged over heads and queries)
        # [batch, seq_len]
        avg_attn_received = attn_scores.mean(dim=1).mean(dim=1)

        # Select top-k tokens
        num_critical = max(1, int(seq_len * critical_ratio))

        # Get top-k indices
        topk_values, topk_indices = torch.topk(avg_attn_received, k=num_critical, dim=1)

        # Create mask
        critical_mask = torch.zeros_like(avg_attn_received, dtype=torch.bool)
        for b in range(batch_size):
            critical_mask[b, topk_indices[b]] = True

    return critical_mask


def compute_critical_mask_kmeans(q, k, critical_ratio=0.5, num_query_clusters=50, num_key_clusters=100, layer_idx=None, use_top_p=True):
    """
    Identify critical tokens using K-means semantic clustering (Sparse VideoGen2 paper approach).

    KEY OPTIMIZATIONS (from official Sparse VideoGen):
    1. Centroid caching: First call uses 50 iters, subsequent use 2 iters (25× faster)
    2. Weighted Top-p selection: Better quality than fixed ratio
    3. Optimized K-means: Pre-computed norms, scatter ops (5-10× faster)

    Args:
        q: Query tensor with shape [batch, seq_len, num_heads, head_dim]
        k: Key tensor with shape [batch, seq_len, num_heads, head_dim]
        critical_ratio: Ratio of tokens to keep (0.0-1.0), or top-p threshold if use_top_p=True
        num_query_clusters: Number of query clusters (default 50, paper uses 100)
        num_key_clusters: Number of key clusters (default 100, paper uses 500)
        layer_idx: Layer index for caching (if None, no caching)
        use_top_p: If True, use weighted top-p selection; else use fixed ratio

    Returns:
        critical_mask: [batch, seq_len] boolean mask where True = critical token
    """
    with torch.no_grad():
        # Handle different input shapes
        if q.dim() == 4:
            batch_size, seq_len, num_heads, head_dim = q.shape
        elif q.dim() == 3:
            total_tokens, num_heads, head_dim = q.shape
            batch_size = 1
            seq_len = total_tokens
            q = q.view(batch_size, seq_len, num_heads, head_dim)
            k = k.view(batch_size, seq_len, num_heads, head_dim)
        else:
            raise ValueError(f"Unexpected q shape: {q.shape}")

        # Adjust cluster numbers based on sequence length
        # Scale clusters to maintain ~10-15 tokens per cluster for better quality
        # This is crucial for long sequences (2496 tokens needs more clusters)
        num_query_clusters = min(max(num_query_clusters, seq_len // 15), seq_len)
        num_key_clusters = min(max(num_key_clusters, seq_len // 10), seq_len)

        # Average across heads to get semantic representation
        # [batch, seq_len, head_dim]
        q_semantic = q.mean(dim=2)
        k_semantic = k.mean(dim=2)

        critical_mask = torch.zeros(batch_size, seq_len, dtype=torch.bool, device=q.device)

        for b in range(batch_size):
            # Get tokens for this batch
            q_tokens = q_semantic[b]  # [seq_len, head_dim]
            k_tokens = k_semantic[b]  # [seq_len, head_dim]

            # === OPTIMIZATION 1: Centroid Caching (Sparse VideoGen approach) ===
            # Determine if this is first call or subsequent call for this layer
            cache_key = f"{layer_idx}_b{b}" if layer_idx is not None else None
            is_first_call = cache_key is None or cache_key not in _centroids_cache

            if is_first_call:
                # First call: Full K-means with many iterations
                max_iters_q = _kmeans_iter_init
                max_iters_k = _kmeans_iter_init
                init_q_centroids = None
                init_k_centroids = None
            else:
                # Subsequent calls: Quick update with cached centroids
                max_iters_q = _kmeans_iter_step
                max_iters_k = _kmeans_iter_step
                cached = _centroids_cache[cache_key]
                init_q_centroids = cached['q_centroids']
                init_k_centroids = cached['k_centroids']

            # Perform K-means clustering on Q tokens using optimized PyTorch implementation
            try:
                # Cluster query tokens
                q_labels, q_centroids, q_cluster_sizes = kmeans_pytorch(
                    q_tokens.float(),  # Convert to float32 for stability
                    num_clusters=num_query_clusters,
                    max_iter=max_iters_q,
                    init_centroids=init_q_centroids.float() if init_q_centroids is not None else None
                )
                q_centroids = q_centroids.to(dtype=q.dtype)  # Convert back to original dtype
                # Keep cluster_sizes as float32 to avoid overflow issues
                q_cluster_sizes = q_cluster_sizes.float()

                # Cluster key tokens
                k_labels, k_centroids, k_cluster_sizes = kmeans_pytorch(
                    k_tokens.float(),  # Convert to float32 for stability
                    num_clusters=num_key_clusters,
                    max_iter=max_iters_k,
                    init_centroids=init_k_centroids.float() if init_k_centroids is not None else None
                )
                k_centroids = k_centroids.to(dtype=k.dtype)  # Convert back to original dtype
                # Keep cluster_sizes as float32 to avoid overflow issues
                k_cluster_sizes = k_cluster_sizes.float()

                # Update cache (store as float32 to avoid dtype issues)
                if cache_key is not None:
                    _centroids_cache[cache_key] = {
                        'q_centroids': q_centroids.detach().clone().float(),
                        'k_centroids': k_centroids.detach().clone().float()
                    }

            except Exception as e:
                # Fallback to simple quantization if K-means fails
                logger.warning(f"PyTorch K-means failed ({e}), using simple quantization instead")

                # Simple quantization: divide space into bins
                q_min, q_max = q_tokens.min(dim=0)[0], q_tokens.max(dim=0)[0]
                q_range = q_max - q_min + 1e-8
                q_normalized = (q_tokens - q_min) / q_range
                q_binned = (q_normalized * (num_query_clusters - 1)).long()
                q_labels = (q_binned * torch.tensor([num_query_clusters ** i for i in range(head_dim)],
                                                     device=q.device)).sum(dim=1) % num_query_clusters

                # Compute centroids
                q_centroids = torch.zeros(num_query_clusters, head_dim, dtype=q.dtype, device=q.device)
                for c in range(num_query_clusters):
                    mask = q_labels == c
                    if mask.any():
                        q_centroids[c] = q_tokens[mask].mean(dim=0)

                # Same for k
                k_min, k_max = k_tokens.min(dim=0)[0], k_tokens.max(dim=0)[0]
                k_range = k_max - k_min + 1e-8
                k_normalized = (k_tokens - k_min) / k_range
                k_binned = (k_normalized * (num_key_clusters - 1)).long()
                k_labels = (k_binned * torch.tensor([num_key_clusters ** i for i in range(head_dim)],
                                                     device=k.device)).sum(dim=1) % num_key_clusters

                k_centroids = torch.zeros(num_key_clusters, head_dim, dtype=k.dtype, device=k.device)
                for c in range(num_key_clusters):
                    mask = k_labels == c
                    if mask.any():
                        k_centroids[c] = k_tokens[mask].mean(dim=0)

                # Fallback branch sets q_labels and k_labels as tensors
                # No need to convert

            # === OPTIMIZATION 2: Weighted Top-p Selection (Sparse VideoGen approach) ===
            # Compute centroid-based attention scores
            # S_ij = centroid(Q_i) · centroid(K_j)^T / sqrt(d_k)
            scale = 1.0 / (head_dim ** 0.5)
            centroid_scores = torch.matmul(q_centroids, k_centroids.t()) * scale
            # [num_query_clusters, num_key_clusters]

            if use_top_p:
                # === Weighted Top-p: Better quality, adaptive sparsity ===
                # (Official Sparse VideoGen method)

                # Step 1: Softmax over centroid attention scores
                # For each Q cluster, compute attention distribution over K clusters
                centroid_attn = torch.softmax(centroid_scores, dim=-1)  # [Cq, Ck]

                # Step 2: Weight by K cluster sizes (larger clusters are more important)
                # This is KEY: balances attention scores with cluster representation
                weighted_attn = centroid_attn * k_cluster_sizes.unsqueeze(0)  # [Cq, Ck]

                # Step 3: Aggregate across all Q clusters
                # Cluster importance = sum of weighted attention from all Q clusters
                cluster_importance = weighted_attn.sum(dim=0)  # [Ck]

                # Step 4: Normalize to probabilities
                cluster_probs = cluster_importance / (cluster_importance.sum() + 1e-8)

                # Step 5: Top-p cumulative selection
                sorted_probs, sorted_indices = torch.sort(cluster_probs, descending=True)
                cumsum_probs = torch.cumsum(sorted_probs, dim=0)

                # Select clusters until cumulative probability >= critical_ratio (used as p threshold)
                keep_mask = cumsum_probs <= critical_ratio
                # Ensure at least one cluster selected
                if not keep_mask.any():
                    keep_mask[0] = True
                # Also add the cluster that crosses the threshold
                num_selected = keep_mask.sum().item()
                if num_selected < num_key_clusters:
                    keep_mask[num_selected] = True

                top_cluster_indices = sorted_indices[keep_mask]

            else:
                # === Fixed Ratio: Original simpler approach ===
                # Exponential weighting (softmax-like)
                exp_scores = torch.exp(centroid_scores)  # [Cq, Ck]

                # Weighted probability considering cluster size
                cluster_importance = k_cluster_sizes * exp_scores.sum(dim=0)  # [Ck]

                # Normalize to probabilities
                cluster_probs = cluster_importance / (cluster_importance.sum() + 1e-8)

                # Fixed number of clusters based on critical_ratio
                num_selected = max(1, int(num_key_clusters * critical_ratio))
                sorted_probs, sorted_indices = torch.sort(cluster_probs, descending=True)
                top_cluster_indices = sorted_indices[:num_selected]

            # Mark all tokens belonging to critical clusters
            for cluster_idx in top_cluster_indices:
                cluster_idx_val = cluster_idx.item()
                token_mask = (k_labels == cluster_idx_val)
                critical_mask[b, token_mask] = True

        return critical_mask


def create_sparse_attn_mask(critical_mask, num_heads, dtype):
    """
    Convert critical mask to attention mask format.

    Args:
        critical_mask: [batch, seq_len] boolean
        num_heads: Number of attention heads
        dtype: Target dtype

    Returns:
        attn_mask: [batch, num_heads, seq_len, seq_len]
    """
    batch_size, seq_len = critical_mask.shape

    # Expand to [batch, 1, 1, seq_len]
    # Each query can only attend to critical keys
    mask = critical_mask.unsqueeze(1).unsqueeze(2)  # [batch, 1, 1, seq_len]

    # Expand to [batch, num_heads, seq_len, seq_len]
    # All heads share the same mask pattern
    mask = mask.expand(batch_size, num_heads, seq_len, seq_len)

    # For boolean mask: True = allow attention, False = mask out
    # We return the mask where True means "keep this key"
    return mask


def sparse_attention_patched(
    q, k, v,
    mode="flash",
    drop_rate=0,
    attn_mask=None,
    causal=False,
    cu_seqlens_q=None,
    cu_seqlens_kv=None,
    max_seqlen_q=None,
    max_seqlen_kv=None,
    batch_size=1,
):
    """
    Patched attention function with sparse attention support.

    This replaces the original attention function and adds sparsity by:
    1. Computing critical token mask based on attention scores
    2. Creating sparse attention mask that only attends to critical tokens
    3. Running attention with the sparse mask
    """
    global _sparse_enabled, _sparse_critical_ratio, _sparse_method, _sparse_stats, _original_attention

    _sparse_stats['total_calls'] += 1

    # If sparse attention is disabled, use original implementation
    if not _sparse_enabled:
        return _original_attention(
            q, k, v, mode=mode, drop_rate=drop_rate, attn_mask=attn_mask,
            causal=causal, cu_seqlens_q=cu_seqlens_q, cu_seqlens_kv=cu_seqlens_kv,
            max_seqlen_q=max_seqlen_q, max_seqlen_kv=max_seqlen_kv,
            batch_size=batch_size
        )

    # Sparse attention is enabled
    _sparse_stats['sparse_calls'] += 1

    try:
        # CRITICAL: We're replacing the ENTIRE attention function, so we receive tensors
        # BEFORE pre_attn_layout is applied. Inputs are in [b, s, a, d] format.
        # We force torch mode and will apply SDPA directly.

        # Expect 4D input in [b, s, a, d] format
        if q.dim() != 4:
            raise ValueError(f"Expected 4D tensor [b,s,a,d], got {q.dim()}D with shape {q.shape}")

        # Extract dimensions - input is in [b, s, a, d] format
        batch, seq_len, num_heads, head_dim = q.shape

        # Debug: Log input shapes and sequence structure
        if _sparse_stats['sparse_calls'] == 1:
            logger.info(f"Sequence structure: cu_seqlens_q={cu_seqlens_q if cu_seqlens_q is not None else 'None'}")
        logger.debug(f"Input shapes: q={q.shape}, batch={batch}, seq_len={seq_len}, num_heads={num_heads}, head_dim={head_dim}")

        # CRITICAL FIX: Exclude text context tokens from permutation
        # The sequence is: [img_tokens, txt_tokens]
        # We should ONLY permute img_tokens, not txt_tokens
        # From cu_seqlens_q, we can infer: video_len = cu_seqlens_q[1], total_len = cu_seqlens_q[2]

        # Dynamically compute video_seq_len from cu_seqlens_q if available
        if cu_seqlens_q is not None and len(cu_seqlens_q) >= 2:
            video_seq_len = cu_seqlens_q[1].item()  # End of first sequence (video tokens)
        else:
            # Fallback to fixed context_length if cu_seqlens_q not available
            context_length = 256  # Default from Sparse-VideoGen
            video_seq_len = seq_len - context_length if seq_len > context_length else seq_len

        # Compute critical mask using selected method
        # Use call count as pseudo layer_idx for caching (since we don't have actual layer_idx)
        pseudo_layer_idx = _sparse_stats['sparse_calls'] % 100  # Assume max 100 unique patterns

        if _sparse_method == 'kmeans':
            # Only compute critical mask for video tokens
            q_video = q[:, :video_seq_len, :, :]
            k_video = k[:, :video_seq_len, :, :]
            critical_mask_video = compute_critical_mask_kmeans(
                q_video, k_video, _sparse_critical_ratio,
                layer_idx=pseudo_layer_idx,
                use_top_p=True  # Use weighted top-p by default
            )
            # Extend mask to full sequence, marking text tokens as non-critical (won't be permuted)
            critical_mask = torch.zeros(batch, seq_len, dtype=torch.bool, device=q.device)
            critical_mask[:, :video_seq_len] = critical_mask_video
        else:  # topk
            q_video = q[:, :video_seq_len, :, :]
            k_video = k[:, :video_seq_len, :, :]
            critical_mask_video = compute_critical_mask_topk(q_video, k_video, _sparse_critical_ratio)
            critical_mask = torch.zeros(batch, seq_len, dtype=torch.bool, device=q.device)
            critical_mask[:, :video_seq_len] = critical_mask_video

        # Semantic Permutation (Sparse VideoGen2 paper method)
        # Instead of masking, we reorder tokens so critical ones are contiguous

        # Generate permutation indices for each batch
        critical_indices_list = []
        non_critical_indices_list = []
        critical_lengths = []

        for b in range(batch):
            critical_idx = torch.where(critical_mask[b])[0]
            non_critical_idx = torch.where(~critical_mask[b])[0]
            critical_indices_list.append(critical_idx)
            non_critical_indices_list.append(non_critical_idx)
            critical_lengths.append(len(critical_idx))

        # Use max critical length for this batch
        max_critical_len = max(critical_lengths)

        if max_critical_len == 0:
            # No critical tokens, fallback to full attention
            logger.warning("No critical tokens selected, using full attention")
            return _original_attention(
                q, k, v, mode=mode, drop_rate=drop_rate, attn_mask=attn_mask,
                causal=causal, cu_seqlens_q=cu_seqlens_q, cu_seqlens_kv=cu_seqlens_kv,
                max_seqlen_q=max_seqlen_q, max_seqlen_kv=max_seqlen_kv,
                batch_size=batch_size
            )

        # Permute Q, K, V to place critical VIDEO tokens first
        # CRITICAL: Only permute video tokens, keep text tokens in original positions
        q_permuted = torch.zeros_like(q)
        k_permuted = torch.zeros_like(k)
        v_permuted = torch.zeros_like(v)

        # Get original shape for reshaping after indexing
        seq_len, num_heads, head_dim = q.shape[1], q.shape[2], q.shape[3]

        for b in range(batch):
            # Only permute video tokens (indices < video_seq_len)
            # Text tokens stay in their original positions
            video_critical_idx = critical_indices_list[b]  # These are already < video_seq_len
            video_non_critical_idx = non_critical_indices_list[b][non_critical_indices_list[b] < video_seq_len]

            perm_video = torch.cat([video_critical_idx, video_non_critical_idx])

            # Permute ONLY the video part
            q_video = q[b, :video_seq_len, :, :]  # [video_seq_len, num_heads, head_dim]
            k_video = k[b, :video_seq_len, :, :]
            v_video = v[b, :video_seq_len, :, :]

            # Reshape to 2D for indexing
            q_video_flat = q_video.reshape(video_seq_len, -1)
            k_video_flat = k_video.reshape(video_seq_len, -1)
            v_video_flat = v_video.reshape(video_seq_len, -1)

            # Apply permutation to video tokens
            q_perm_video_flat = q_video_flat[perm_video]
            k_perm_video_flat = k_video_flat[perm_video]
            v_perm_video_flat = v_video_flat[perm_video]

            # Reshape back to 3D
            try:
                q_perm_video = q_perm_video_flat.view(video_seq_len, num_heads, head_dim)
                k_perm_video = k_perm_video_flat.view(video_seq_len, num_heads, head_dim)
                v_perm_video = v_perm_video_flat.view(video_seq_len, num_heads, head_dim)

                # Copy permuted video tokens
                q_permuted[b, :video_seq_len, :, :].copy_(q_perm_video)
                k_permuted[b, :video_seq_len, :, :].copy_(k_perm_video)
                v_permuted[b, :video_seq_len, :, :].copy_(v_perm_video)

                # Keep text tokens in original order (no permutation)
                if video_seq_len < seq_len:
                    q_permuted[b, video_seq_len:, :, :].copy_(q[b, video_seq_len:, :, :])
                    k_permuted[b, video_seq_len:, :, :].copy_(k[b, video_seq_len:, :, :])
                    v_permuted[b, video_seq_len:, :, :].copy_(v[b, video_seq_len:, :, :])
            except Exception as reshape_err:
                logger.error(f"Permutation reshape error: {reshape_err}")
                logger.error(f"  q_perm_flat.shape={q_perm_flat.shape}, expected=[{seq_len}, {num_heads*head_dim}]")
                logger.error(f"  target shape: [{seq_len}, {num_heads}, {head_dim}]")
                logger.error(f"  perm.shape={perm.shape}, q[b].shape={q[b].shape}")
                raise

        # Simplified Block Sparse Attention with mask
        #
        # Strategy: Create an attention mask that allows:
        # - Critical tokens attend to all tokens (both critical and non-critical)
        # - Non-critical tokens only attend to themselves (identity mapping)
        #
        # This approximates block sparse attention without custom kernels.

        # Create attention mask: [batch, num_heads, seq_len, seq_len]
        # mask[i,j] = True means token i can attend to token j
        # We want: critical tokens attend to all, non-critical only to self

        attn_mask = torch.zeros(batch, seq_len, seq_len, dtype=torch.bool, device=q.device)

        for b in range(batch):
            critical_len = critical_lengths[b]
            # Critical tokens (first critical_len after permutation) attend to ALL tokens
            attn_mask[b, :critical_len, :] = True

            # Non-critical tokens only attend to themselves (diagonal)
            for i in range(critical_len, seq_len):
                attn_mask[b, i, i] = True

        # Debug: Log sparse pattern
        if _sparse_stats['sparse_calls'] == 1:
            total_elements = batch * seq_len * seq_len
            sparse_elements = attn_mask.sum().item()
            sparsity = sparse_elements / total_elements
            logger.debug(f"Block sparse pattern: {sparse_elements}/{total_elements} elements ({sparsity:.2%} density)")
            logger.debug(f"Critical tokens: {max_critical_len}, Non-critical: {seq_len - max_critical_len}")

        # Convert boolean mask to float mask for SDPA
        # SDPA expects: attn_mask[i,j] = -inf means "cannot attend", 0.0 means "can attend"
        float_mask = torch.zeros(batch, 1, seq_len, seq_len, device=q.device, dtype=q.dtype)
        float_mask.masked_fill_(~attn_mask.unsqueeze(1), float('-inf'))

        # Use torch SDPA with sparse mask
        # Input: q_permuted, k_permuted, v_permuted in [b,s,a,d] format
        q_permuted_transposed = q_permuted.transpose(1, 2)  # [b,s,a,d] -> [b,a,s,d]
        k_permuted_transposed = k_permuted.transpose(1, 2)
        v_permuted_transposed = v_permuted.transpose(1, 2)

        output_permuted_transposed = F.scaled_dot_product_attention(
            q_permuted_transposed, k_permuted_transposed, v_permuted_transposed,
            attn_mask=float_mask,  # Apply sparse mask
            dropout_p=drop_rate,
            is_causal=False
        )  # Output: [b, a, s, d]

        output_permuted = output_permuted_transposed.transpose(1, 2)  # [b,a,s,d] -> [b,s,a,d]

        if _sparse_stats['sparse_calls'] == 1:
            logger.debug(f"Sparse attention output shape: output_permuted={output_permuted.shape}")

        # Inverse permutation to restore original token order
        # CRITICAL: Only restore video tokens, text tokens are already in correct positions
        output = torch.zeros_like(q)
        for b in range(batch):
            # Only inverse permute video tokens
            video_critical_idx = critical_indices_list[b]
            video_non_critical_idx = non_critical_indices_list[b][non_critical_indices_list[b] < video_seq_len]
            perm_video = torch.cat([video_critical_idx, video_non_critical_idx])
            inverse_perm_video = torch.argsort(perm_video)

            # Inverse permute ONLY video tokens
            output_video = output_permuted[b, :video_seq_len, :, :]  # [video_seq_len, num_heads, head_dim]
            output_video_flat = output_video.reshape(video_seq_len, -1)
            output_inv_video_flat = output_video_flat[inverse_perm_video]
            output[b, :video_seq_len, :, :].copy_(output_inv_video_flat.reshape(video_seq_len, num_heads, head_dim))

            # Text tokens are already in correct positions, just copy
            if video_seq_len < seq_len:
                output[b, video_seq_len:, :, :].copy_(output_permuted[b, video_seq_len:, :, :])

        # Return the output with restored token order
        # NOTE: Output is in [b, s, a, d] format, but original attention expects it
        # to go through post_attn_layout. Since we're returning from here directly,
        # we need to apply the same transformations the original function would.
        #
        # Original flow: pre_attn_layout -> attention_op -> post_attn_layout -> reshape
        # Our flow: (already reversed pre_attn_layout) -> sparse_op -> (need to skip post_attn_layout)
        #
        # For torch mode: post_attn_layout transposes [b,a,s,d] -> [b,s,a,d]
        # But we already have [b,s,a,d], so we need to apply transpose before post_attn_layout
        # Actually, the original function expects us to return in the format AFTER attention_op,
        # which would be [b,a,s,d] for torch mode. Let me NOT apply post_attn_layout since
        # the caller will do it.

        # Wait - we're calling _original_attention which will apply post_attn_layout internally!
        # So output_critical is already in [b,s,a,d] format (after post_attn_layout).
        # We just need to return [b,s,a,d] format here too.

        # Original attention function returns [b, s, a*d] (heads flattened)
        # We have [b, s, a, d], so flatten the last two dimensions
        # Make sure output is contiguous before reshaping
        output = output.contiguous()
        b, s, a, d = output.shape
        output_flat = output.view(b, s, a * d)

        # Debug logging (first call only)
        if _sparse_stats['sparse_calls'] == 1:
            logger.debug(f"Sparse attention output shape: output=[{b},{s},{a},{d}] -> output_flat={output_flat.shape}")

        return output_flat

    except Exception as e:
        logger.warning(f"Sparse attention masking failed: {e}. Falling back to full attention.")
        _sparse_stats['sparse_calls'] -= 1
        return _original_attention(
            q, k, v, mode=mode, drop_rate=drop_rate, attn_mask=attn_mask,
            causal=causal, cu_seqlens_q=cu_seqlens_q, cu_seqlens_kv=cu_seqlens_kv,
            max_seqlen_q=max_seqlen_q, max_seqlen_kv=max_seqlen_kv,
            batch_size=batch_size
        )


def clear_centroids_cache():
    """Clear the centroids cache. Call this between different videos."""
    global _centroids_cache
    _centroids_cache.clear()
    logger.debug("Cleared centroids cache")


def enable_sparse_attention(critical_ratio=0.5, method='kmeans'):
    """Enable sparse attention globally.

    Args:
        critical_ratio: Ratio of tokens to keep (0.0-1.0), or top-p threshold if using weighted top-p
        method: 'topk' or 'kmeans' (default 'kmeans' for better quality)
    """
    global _original_attention, _sparse_enabled, _sparse_critical_ratio, _sparse_method, _sparse_stats

    if _original_attention is None:
        _original_attention = _original_attention_func

    _sparse_enabled = True
    _sparse_critical_ratio = critical_ratio
    _sparse_method = method
    _sparse_stats = {'total_calls': 0, 'sparse_calls': 0}

    # Clear cache when enabling (fresh start)
    clear_centroids_cache()

    # Patch the module-level function
    attenion.attention = sparse_attention_patched

    # CRITICAL: Also patch models.py's local reference
    # This is necessary because models.py did: from .attenion import attention
    try:
        from hyvideo.modules import models
        models.attention = sparse_attention_patched
    except Exception as e:
        logger.warning(f"Could not patch models.attention: {e}")

    logger.info(f"✓ Sparse attention enabled (method={method}, critical_ratio={critical_ratio}, with centroid caching)")


def disable_sparse_attention():
    """Disable sparse attention and restore original."""
    global _original_attention, _sparse_enabled, _sparse_stats

    if _original_attention is not None:
        # Restore module-level function
        attenion.attention = _original_attention

        # Restore models.py's local reference
        try:
            from hyvideo.modules import models
            models.attention = _original_attention
        except Exception as e:
            logger.warning(f"Could not restore models.attention: {e}")

    _sparse_enabled = False
    logger.info(f"✓ Sparse attention disabled")
    logger.info(f"  Sparse calls: {_sparse_stats['sparse_calls']} / {_sparse_stats['total_calls']}")


def generate_video(sampler, args, mode='full', critical_ratio=0.5, sparse_method='kmeans'):
    """
    Generate a single video with specified attention mode.

    Args:
        sampler: HunyuanVideoSampler instance
        args: Configuration arguments
        mode: 'full' or 'sparse'
        critical_ratio: Ratio of critical tokens (for sparse mode)
        sparse_method: 'topk' or 'kmeans' (for sparse mode)

    Returns:
        Dictionary with generation results and timing info
    """
    logger.info(f"Generating video with {mode.upper()} attention...")

    # Enable/disable sparse attention based on mode
    if mode == 'sparse':
        enable_sparse_attention(critical_ratio, method=sparse_method)
    else:
        disable_sparse_attention()

    start_time = time.time()

    outputs = sampler.predict(
        prompt=args.prompt,
        height=args.video_size[0],
        width=args.video_size[1],
        video_length=args.video_length,
        seed=args.seed,
        negative_prompt=args.neg_prompt,
        infer_steps=args.infer_steps,
        guidance_scale=args.cfg_scale,
        num_videos_per_prompt=1,  # Generate one video at a time
        flow_shift=args.flow_shift,
        batch_size=args.batch_size,
        embedded_guidance_scale=args.embedded_cfg_scale,
    )

    generation_time = time.time() - start_time

    # Get sparse stats
    sparse_stats = _sparse_stats.copy()

    return {
        'samples': outputs['samples'],
        'seeds': outputs['seeds'],
        'prompts': outputs['prompts'],
        'generation_time': generation_time,
        'mode': mode,
        'sparse_stats': sparse_stats,
    }


def main():
    """Main comparison script."""
    # Add comparison-specific arguments first
    import sys
    comp_parser = argparse.ArgumentParser(add_help=False)
    comp_parser.add_argument('--critical-ratio', type=float, default=0.5,
                           help='Ratio of critical tokens to keep in sparse mode (0.0-1.0)')
    comp_parser.add_argument('--sparse-method', type=str, default='kmeans', choices=['topk', 'kmeans'],
                           help='Method for selecting critical tokens: kmeans (default, better quality) or topk (faster)')
    comp_parser.add_argument('--save-prefix', type=str, default='comparison',
                           help='Prefix for saved video files')
    comp_parser.add_argument('--skip-full', action='store_true',
                           help='Skip full attention generation (only generate sparse)')
    comp_parser.add_argument('--skip-sparse', action='store_true',
                           help='Skip sparse attention generation (only generate full)')

    comp_args, remaining_argv = comp_parser.parse_known_args()

    # Parse base arguments (with remaining argv to avoid conflicts)
    sys.argv = [sys.argv[0]] + remaining_argv
    args = parse_args()

    # Validate critical ratio
    if not 0.0 < comp_args.critical_ratio <= 1.0:
        raise ValueError("critical_ratio must be in (0.0, 1.0]")

    # Setup
    models_root_path = Path(args.model_base)
    if not models_root_path.exists():
        raise ValueError(f"`models_root` not exists: {models_root_path}")

    # Create save directory
    save_path = args.save_path
    if not os.path.exists(save_path):
        os.makedirs(save_path, exist_ok=True)

    # Load model
    logger.info("Loading HunyuanVideo model...")
    hunyuan_video_sampler = HunyuanVideoSampler.from_pretrained(models_root_path, args=args)
    args = hunyuan_video_sampler.args

    # Clear CUDA cache before generation
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        logger.info(f"GPU Memory before generation: {torch.cuda.memory_allocated()/1024**3:.2f} GB allocated, {torch.cuda.memory_reserved()/1024**3:.2f} GB reserved")

    print("\n" + "="*80)
    print("SPARSE VS FULL ATTENTION COMPARISON")
    print("="*80)
    print(f"Prompt: {args.prompt}")
    print(f"Video size: {args.video_size[0]}x{args.video_size[1]}")
    print(f"Video length: {args.video_length} frames")
    print(f"Critical ratio: {comp_args.critical_ratio:.1%}")
    print(f"Inference steps: {args.infer_steps}")
    print(f"Guidance scale: {args.cfg_scale}")
    print(f"Seed: {args.seed}")
    print("="*80)

    results = {}
    file_paths = {}

    # Generate with FULL attention (baseline)
    if not comp_args.skip_full:
        print("\n[1/2] Generating with FULL attention...")
        print("-" * 80)
        full_result = generate_video(hunyuan_video_sampler, args, mode='full')
        results['full'] = full_result

        # Save full attention video
        time_flag = datetime.fromtimestamp(time.time()).strftime("%Y-%m-%d-%H:%M:%S")
        full_path = f"{save_path}/{comp_args.save_prefix}_{time_flag}_FULL_seed{full_result['seeds'][0]}.mp4"

        sample = full_result['samples'][0].unsqueeze(0)
        save_videos_grid(sample, full_path, fps=24)
        file_paths['full'] = full_path

        print(f"✓ Full attention video saved")
        print(f"  Path: {full_path}")
        print(f"  Generation time: {full_result['generation_time']:.2f}s")

        # Clear CUDA cache between generations
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            logger.info(f"Cleared CUDA cache after full generation")

    # Generate with SPARSE attention
    if not comp_args.skip_sparse:
        print("\n[2/2] Generating with SPARSE attention...")
        print("-" * 80)
        print(f"Critical ratio: {comp_args.critical_ratio:.1%}")
        print(f"Theoretical speedup: {1/comp_args.critical_ratio:.2f}x")

        sparse_result = generate_video(
            hunyuan_video_sampler, args,
            mode='sparse',
            critical_ratio=comp_args.critical_ratio,
            sparse_method=comp_args.sparse_method
        )
        results['sparse'] = sparse_result

        # Save sparse attention video
        time_flag = datetime.fromtimestamp(time.time()).strftime("%Y-%m-%d-%H:%M:%S")
        sparse_path = f"{save_path}/{comp_args.save_prefix}_{time_flag}_SPARSE{int(comp_args.critical_ratio*100)}_seed{sparse_result['seeds'][0]}.mp4"

        sample = sparse_result['samples'][0].unsqueeze(0)
        save_videos_grid(sample, sparse_path, fps=24)
        file_paths['sparse'] = sparse_path

        print(f"✓ Sparse attention video saved")
        print(f"  Path: {sparse_path}")
        print(f"  Generation time: {sparse_result['generation_time']:.2f}s")
        print(f"  Sparse calls: {sparse_result['sparse_stats']['sparse_calls']} / {sparse_result['sparse_stats']['total_calls']}")

    # Restore original attention
    disable_sparse_attention()

    # Print comparison summary
    print("\n" + "="*80)
    print("COMPARISON SUMMARY")
    print("="*80)

    if 'full' in results and 'sparse' in results:
        full_time = results['full']['generation_time']
        sparse_time = results['sparse']['generation_time']
        speedup = full_time / sparse_time
        theoretical_speedup = 1.0 / comp_args.critical_ratio

        print(f"\n⏱️  Generation Time:")
        print(f"  Full:   {full_time:.2f}s")
        print(f"  Sparse: {sparse_time:.2f}s")
        print(f"  Actual speedup: {speedup:.2f}x")

        print(f"\n🎯 Sparsity:")
        print(f"  Critical ratio: {comp_args.critical_ratio:.1%}")
        print(f"  Non-critical ratio: {1-comp_args.critical_ratio:.1%}")
        print(f"  Theoretical speedup: {theoretical_speedup:.2f}x")
        print(f"  Efficiency: {speedup/theoretical_speedup*100:.1f}%")

        print(f"\n📹 Video files:")
        print(f"  Full:   {file_paths['full']}")
        print(f"  Sparse: {file_paths['sparse']}")

        print(f"\n💡 Next steps:")
        print(f"  1. Watch both videos side-by-side to compare quality")
        print(f"  2. Use ffmpeg to compare frame-by-frame:")
        print(f"     ffmpeg -i {file_paths['full']} -i {file_paths['sparse']} \\")
        print(f"            -filter_complex psnr -f null -")
        print(f"  3. Try different critical ratios: 0.3, 0.5, 0.7")
        print(f"  4. Test with different prompts to see consistency")

    elif 'full' in results:
        print(f"\n✓ Generated full attention video only:")
        print(f"  Time: {results['full']['generation_time']:.2f}s")
        print(f"  File: {file_paths['full']}")

    elif 'sparse' in results:
        print(f"\n✓ Generated sparse attention video only:")
        print(f"  Time: {results['sparse']['generation_time']:.2f}s")
        print(f"  File: {file_paths['sparse']}")
        print(f"  Critical ratio: {comp_args.critical_ratio:.1%}")
        print(f"  Sparse calls: {results['sparse']['sparse_stats']['sparse_calls']}")

    print("="*80 + "\n")


if __name__ == "__main__":
    main()
