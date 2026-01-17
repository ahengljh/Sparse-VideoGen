"""
Ultra-aggressive KV cache: Cache entire K/V tensors when dynamic_map is stable.

Key idea:
- Instead of block-level cache, cache the ENTIRE permuted K/V tensors
- Only when dynamic_map changes significantly, recompute everything
- Trades granularity for speed

This is much faster but less flexible than block-level caching.
"""

import torch
from typing import Dict, Tuple
from loguru import logger

try:
    from sparse_attention_cluster_fast import ClusterBasedSparseAttentionFast
except ImportError:
    ClusterBasedSparseAttentionFast = object


class ClusterBasedSparseAttentionFastV3(ClusterBasedSparseAttentionFast):
    """
    Aggressive whole-tensor KV caching.

    When dynamic_map is stable (< threshold change):
    - Directly reuse cached K/V tensors (no cloning, no block-wise operations)
    - Skip all cache restoration logic

    When dynamic_map changes:
    - Recompute and cache new K/V tensors
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Whole-tensor cache instead of block-level
        self.kv_tensor_cache = {}  # {layer_idx: (k_perm, v_perm, dynamic_map)}

    def _try_reuse_kv_tensors(
        self,
        k_perm: torch.Tensor,
        v_perm: torch.Tensor,
        dynamic_map: torch.Tensor,
        layer_idx: int,
    ) -> tuple[torch.Tensor, torch.Tensor, bool]:
        """
        Try to reuse entire cached K/V tensors.

        Returns:
            (k_perm, v_perm, cache_hit)
        """
        if not self.enable_kv_cache:
            return k_perm, v_perm, False

        if layer_idx not in self.kv_tensor_cache:
            # First call - cache and return
            self.kv_tensor_cache[layer_idx] = (
                k_perm.detach().clone(),
                v_perm.detach().clone(),
                dynamic_map.detach().clone()
            )
            self.stats['kv_cache_misses'] += 1
            return k_perm, v_perm, False

        # Check if dynamic_map changed
        cached_k, cached_v, cached_map = self.kv_tensor_cache[layer_idx]

        # Fast comparison: compute change ratio
        if cached_map.shape != dynamic_map.shape:
            # Shape changed - need to recompute
            self.kv_tensor_cache[layer_idx] = (
                k_perm.detach().clone(),
                v_perm.detach().clone(),
                dynamic_map.detach().clone()
            )
            self.stats['kv_cache_misses'] += 1
            return k_perm, v_perm, False

        # ===== Single GPU operation for comparison =====
        change_ratio = (cached_map != dynamic_map).float().mean().item()

        if change_ratio < self.cache_update_threshold:
            # Cache hit! Return cached tensors directly
            self.stats['kv_cache_hits'] += 1
            # Return cached tensors WITHOUT cloning (zero overhead!)
            return cached_k, cached_v, True
        else:
            # Dynamic map changed too much - update cache
            self.kv_tensor_cache[layer_idx] = (
                k_perm.detach().clone(),
                v_perm.detach().clone(),
                dynamic_map.detach().clone()
            )
            self.stats['kv_cache_misses'] += 1
            return k_perm, v_perm, False

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        layer_idx: int,
        tokens_per_frame: int,
        video_seq_len: int,
    ) -> torch.Tensor:
        """Override to use whole-tensor cache."""
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

        # 2.5. Try to reuse entire cached K/V tensors
        if self.enable_kv_cache:
            k_perm, v_perm, cache_hit = self._try_reuse_kv_tensors(
                k_perm, v_perm, dynamic_map, layer_idx
            )
            # If cache hit, k_perm and v_perm are directly from cache (no cloning!)

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

    def reset_cache(self):
        """Override to clear tensor cache."""
        super().reset_cache()
        self.kv_tensor_cache = {}


# Quick test
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
    print("Testing Whole-Tensor KV Cache (V3)")
    print("=" * 60)

    q = torch.randn(cfg, num_heads, seq_len, dim, device=device, dtype=dtype)
    k = torch.randn(cfg, num_heads, seq_len, dim, device=device, dtype=dtype)
    v = torch.randn(cfg, num_heads, seq_len, dim, device=device, dtype=dtype)

    attn = ClusterBasedSparseAttentionFastV3(
        num_q_centroids=16,
        num_k_centroids=16,
        enable_kv_cache=True,
        cache_update_threshold=0.05,  # Only update if > 5% of dynamic_map changed
    )

    # Warmup
    for _ in range(3):
        _ = attn.forward(q, k, v, layer_idx=0, tokens_per_frame=64, video_seq_len=seq_len)

    torch.cuda.synchronize()
    start = time.time()

    for i in range(num_iterations):
        if i > 0:
            # Small perturbation - dynamic_map should stay mostly stable
            q = q + torch.randn_like(q) * 0.01
            k = k + torch.randn_like(k) * 0.01
            v = v + torch.randn_like(v) * 0.01
        _ = attn.forward(q, k, v, layer_idx=0, tokens_per_frame=64, video_seq_len=seq_len)

    torch.cuda.synchronize()
    elapsed = time.time() - start

    stats = attn.get_stats()
    print(f"\nWhole-tensor cache (V3):")
    print(f"  Time: {elapsed/num_iterations*1000:.2f}ms/iter")
    print(f"  Cache hit rate: {stats.get('kv_cache_hit_rate', 0):.2%}")
    print(f"  Hits: {stats.get('kv_cache_hits', 0)}")
    print(f"  Misses: {stats.get('kv_cache_misses', 0)}")
    print(f"\nKey advantage: Zero overhead when cache hits (no cloning, no loops)")
