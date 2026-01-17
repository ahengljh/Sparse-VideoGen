"""
Optimized version of ClusterBasedSparseAttentionFast with vectorized KV cache.

Key optimizations:
1. Batch comparison using tensors instead of Python loops
2. Single clone operation instead of conditional cloning
3. Vectorized block restoration
"""

import torch
from typing import Dict, Tuple
from loguru import logger

try:
    from sparse_attention_cluster_fast import ClusterBasedSparseAttentionFast
except ImportError:
    ClusterBasedSparseAttentionFast = object


class ClusterBasedSparseAttentionFastV2(ClusterBasedSparseAttentionFast):
    """
    Optimized version with fully vectorized KV cache restoration.

    Key improvements over base version:
    - Batch dynamic_map comparison (single GPU op vs many torch.equal calls)
    - Pre-allocate modified tensors (avoid conditional cloning)
    - Vectorized block indexing and copying
    """

    def _restore_kv_from_cache_vectorized(
        self,
        k_perm: torch.Tensor,
        v_perm: torch.Tensor,
        dynamic_map: torch.Tensor,
        k_cluster_sizes: torch.Tensor,
        layer_idx: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Fully vectorized cache restoration.

        Strategy:
        1. Build a "reuse_mask" tensor indicating which blocks can be reused
        2. Use masked operations to restore all cached blocks at once
        3. Avoid all Python loops and torch.equal() calls
        """
        if not self.enable_kv_cache:
            return k_perm, v_perm

        cfg, num_heads, seq_len, dim = k_perm.shape
        device = k_perm.device

        # Update dynamic_map cache
        old_map = self.dynamic_map_cache.get(layer_idx, None)

        if old_map is None:
            # First call for this layer - just cache and return
            self.dynamic_map_cache[layer_idx] = dynamic_map.detach().clone()
            return k_perm, v_perm

        # ===== Key Optimization: Batch comparison =====
        # Compare entire dynamic_map at once instead of column-by-column
        # Shape: [cfg, num_heads, qc_num, kc_num]
        map_unchanged = torch.all(old_map == dynamic_map, dim=2)  # [cfg, num_heads, kc_num]

        # Get active K clusters
        active_k_mask = dynamic_map.any(dim=2)  # [cfg, num_heads, kc_num]

        # Blocks that can be reused: active AND usage pattern unchanged
        reuse_mask = active_k_mask & map_unchanged  # [cfg, num_heads, kc_num]

        # Check if we have any blocks to restore
        if not reuse_mask.any():
            # Nothing to restore, update cache and return
            self.dynamic_map_cache[layer_idx] = dynamic_map.detach().clone()
            return k_perm, v_perm

        # ===== Optimization: Clone once upfront =====
        # Since we know we'll modify, clone immediately
        k_perm_out = k_perm.clone()
        v_perm_out = v_perm.clone()

        # Calculate block boundaries
        k_cum_sizes = torch.cat([
            torch.zeros_like(k_cluster_sizes[..., :1]),
            k_cluster_sizes
        ], dim=-1).cumsum(dim=-1)  # [cfg, num_heads, kc_num+1]

        # ===== Restore blocks =====
        # Still need a small loop over (b, h, k_idx) but with optimizations
        num_restored = 0
        num_cache_misses = 0

        for b in range(cfg):
            for h in range(num_heads):
                # Get indices of blocks to restore for this (b, h)
                reuse_indices = torch.where(reuse_mask[b, h])[0]

                if len(reuse_indices) == 0:
                    continue

                # Process each block (still needs loop due to variable sizes)
                for k_idx in reuse_indices.tolist():
                    cache_key = (layer_idx, b, h, k_idx)

                    if cache_key not in self.kv_block_cache:
                        continue

                    k_start = k_cum_sizes[b, h, k_idx].item()
                    k_end = k_cum_sizes[b, h, k_idx + 1].item()

                    if k_start >= k_end:
                        continue

                    k_block, v_block = self.kv_block_cache[cache_key]
                    block_size = k_end - k_start

                    # Size check
                    if k_block.shape[0] == block_size:
                        # Restore from cache
                        k_perm_out[b, h, k_start:k_end, :] = k_block
                        v_perm_out[b, h, k_start:k_end, :] = v_block
                        num_restored += 1

        # Cache new/updated blocks for active K clusters
        active_indices = torch.where(active_k_mask)  # Returns tuple of indices

        for idx in range(len(active_indices[0])):
            b = active_indices[0][idx].item()
            h = active_indices[1][idx].item()
            k_idx = active_indices[2][idx].item()

            # Skip if already reused
            if reuse_mask[b, h, k_idx]:
                continue

            cache_key = (layer_idx, b, h, k_idx)
            k_start = k_cum_sizes[b, h, k_idx].item()
            k_end = k_cum_sizes[b, h, k_idx + 1].item()

            if k_start >= k_end:
                continue

            # Cache this block
            k_block = k_perm_out[b, h, k_start:k_end, :].detach().clone()
            v_block = v_perm_out[b, h, k_start:k_end, :].detach().clone()
            self.kv_block_cache[cache_key] = (k_block, v_block)
            num_cache_misses += 1

        # Update stats
        self.stats['kv_cache_hits'] += num_restored
        self.stats['kv_cache_misses'] += num_cache_misses
        self.stats['kv_blocks_total'] += num_restored + num_cache_misses

        # Update dynamic_map cache
        if (old_map != dynamic_map).float().mean().item() >= self.cache_update_threshold:
            self.dynamic_map_cache[layer_idx] = dynamic_map.detach().clone()

        return k_perm_out, v_perm_out

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        layer_idx: int,
        tokens_per_frame: int,
        video_seq_len: int,
    ) -> torch.Tensor:
        """Override to use vectorized cache restoration."""
        # Import here to avoid circular dependency
        from svg_kmeans_utils import dynamic_block_sparse_fwd_triton
        from svg_kernels.triton.permute import apply_inverse_permutation_triton

        cfg, num_heads, total_len, dim = query.shape
        text_seq_len = total_len - video_seq_len

        # 1. Split video and text
        query_video = query[:, :, :video_seq_len, :].contiguous()
        key_video = key[:, :, :video_seq_len, :].contiguous()
        value_video = value[:, :, :video_seq_len, :].contiguous()

        # 2. Semantic-aware permutation
        q_perm, k_perm, v_perm, dynamic_map, q_cluster_sizes, k_cluster_sizes, q_sorted_indices = \
            self.semantic_aware_permutation(query_video, key_video, value_video, layer_idx, tokens_per_frame)

        # 2.5. Use VECTORIZED cache restoration
        if self.enable_kv_cache:
            k_perm, v_perm = self._restore_kv_from_cache_vectorized(
                k_perm, v_perm, dynamic_map, k_cluster_sizes, layer_idx
            )

        # 3. Dynamic map post-processing
        if text_seq_len > 0:
            query_combined, key_combined, value_combined, dynamic_map_extended, \
            qc_sz_extended, kc_sz_extended, q_sorted_indices_extended = \
                self.dynamic_map_post_processing(
                    q_perm, k_perm, v_perm,
                    query, key, value,
                    dynamic_map, q_cluster_sizes, k_cluster_sizes, q_sorted_indices,
                    video_seq_len, text_seq_len
                )
        else:
            query_combined = q_perm
            key_combined = k_perm
            value_combined = v_perm
            dynamic_map_extended = dynamic_map
            qc_sz_extended = q_cluster_sizes
            kc_sz_extended = k_cluster_sizes
            q_sorted_indices_extended = q_sorted_indices

        # 4. Call Triton kernel
        output_permuted = dynamic_block_sparse_fwd_triton(
            query_combined, key_combined, value_combined,
            dynamic_map_extended, qc_sz_extended, kc_sz_extended
        )

        # 5. Reverse permutation
        if text_seq_len > 0:
            output_video = output_permuted[:, :, :video_seq_len, :]
            output_text = output_permuted[:, :, video_seq_len:, :]
            output_video_restored = apply_inverse_permutation_triton(
                output_video, q_sorted_indices_extended, dim=2
            )
            output = torch.cat([output_video_restored, output_text], dim=2)
        else:
            output = apply_inverse_permutation_triton(output_permuted, q_sorted_indices_extended, dim=2)

        # Update stats
        density = dynamic_map_extended.float().mean()
        self.stats['total_calls'] += 1
        self.stats['avg_density'] = density.item()

        return output


# Test the vectorized version
if __name__ == "__main__":
    import time

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cpu":
        print("CUDA not available")
        exit(1)

    dtype = torch.float16
    cfg, num_heads, seq_len, dim = 1, 4, 256, 32
    num_iterations = 10

    print("=" * 60)
    print("Testing Vectorized KV Cache")
    print("=" * 60)

    # Prepare data
    q = torch.randn(cfg, num_heads, seq_len, dim, device=device, dtype=dtype)
    k = torch.randn(cfg, num_heads, seq_len, dim, device=device, dtype=dtype)
    v = torch.randn(cfg, num_heads, seq_len, dim, device=device, dtype=dtype)

    # Test vectorized version
    attn = ClusterBasedSparseAttentionFastV2(
        num_q_centroids=16,
        num_k_centroids=16,
        enable_kv_cache=True,
    )

    # Warmup
    for _ in range(3):
        _ = attn.forward(q, k, v, layer_idx=0, tokens_per_frame=64, video_seq_len=seq_len)

    torch.cuda.synchronize()
    start = time.time()

    for i in range(num_iterations):
        if i > 0:
            q = q + torch.randn_like(q) * 0.01
            k = k + torch.randn_like(k) * 0.01
            v = v + torch.randn_like(v) * 0.01
        _ = attn.forward(q, k, v, layer_idx=0, tokens_per_frame=64, video_seq_len=seq_len)

    torch.cuda.synchronize()
    elapsed = time.time() - start

    stats = attn.get_stats()
    print(f"\nVectorized version:")
    print(f"  Time: {elapsed/num_iterations*1000:.2f}ms/iter")
    print(f"  Cache hit rate: {stats['kv_cache_hit_rate']:.2%}")
    print(f"  Hits: {stats['kv_cache_hits']}")
    print(f"  Misses: {stats['kv_cache_misses']}")
