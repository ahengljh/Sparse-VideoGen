"""
V4: Temporal Smoothing Cache

Key improvement over V3:
- Uses Exponential Moving Average (EMA) to smooth K/V updates
- Reduces sudden changes in cached K/V tensors
- May improve both quality and cache hit rate
"""

import torch
from typing import Dict, Tuple
from loguru import logger

try:
    from sparse_attention_cluster_fast import ClusterBasedSparseAttentionFast
except ImportError:
    ClusterBasedSparseAttentionFast = object


class ClusterBasedSparseAttentionFastV4(ClusterBasedSparseAttentionFast):
    """
    V4 with temporal smoothing for K/V cache.

    Instead of abruptly switching between cached and new K/V,
    we use EMA to smooth the transition.
    """

    def __init__(self, *args, ema_alpha=0.3, **kwargs):
        """
        Args:
            ema_alpha: Smoothing factor (0-1)
                      - 0.0 = never update (always use old cache)
                      - 1.0 = no smoothing (same as V3)
                      - 0.3 = 30% new + 70% old (recommended)
        """
        super().__init__(*args, **kwargs)
        self.ema_alpha = ema_alpha

        # Separate caches for temporal smoothing
        self.k_smooth_cache = {}  # {layer_idx: (k_smoothed, dynamic_map)}
        self.v_smooth_cache = {}  # {layer_idx: (v_smoothed, dynamic_map)}

    def _try_reuse_kv_tensors(
        self,
        k_perm: torch.Tensor,
        v_perm: torch.Tensor,
        dynamic_map: torch.Tensor,
        layer_idx: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Try to reuse K/V with temporal smoothing.
        """
        if not self.enable_k_cache and not self.enable_v_cache:
            return k_perm, v_perm

        k_result = k_perm
        v_result = v_perm

        # ===== K cache with temporal smoothing =====
        if self.enable_k_cache:
            if layer_idx not in self.k_smooth_cache:
                # First call - initialize cache
                self.k_smooth_cache[layer_idx] = (
                    k_perm.detach().clone(),
                    dynamic_map.detach().clone()
                )
                self.stats['k_cache_misses'] += 1
            else:
                cached_k, cached_map = self.k_smooth_cache[layer_idx]

                # Shape check
                if cached_map.shape != dynamic_map.shape:
                    self.k_smooth_cache[layer_idx] = (
                        k_perm.detach().clone(),
                        dynamic_map.detach().clone()
                    )
                    self.stats['k_cache_misses'] += 1
                else:
                    # Compare dynamic_map
                    change_ratio = (cached_map != dynamic_map).float().mean().item()

                    if change_ratio < self.cache_update_threshold:
                        # Cache hit - use cached K directly
                        k_result = cached_k
                        self.stats['k_cache_hits'] += 1
                    else:
                        # Cache miss - smooth update
                        # EMA: k_new = alpha * k_current + (1-alpha) * k_cached
                        k_smoothed = self.ema_alpha * k_perm + (1 - self.ema_alpha) * cached_k

                        # Update cache with smoothed value
                        self.k_smooth_cache[layer_idx] = (
                            k_smoothed.detach().clone(),
                            dynamic_map.detach().clone()
                        )

                        k_result = k_smoothed
                        self.stats['k_cache_misses'] += 1

        # ===== V cache with temporal smoothing =====
        if self.enable_v_cache:
            if layer_idx not in self.v_smooth_cache:
                # First call - initialize cache
                self.v_smooth_cache[layer_idx] = (
                    v_perm.detach().clone(),
                    dynamic_map.detach().clone()
                )
                self.stats['v_cache_misses'] += 1
            else:
                cached_v, cached_map = self.v_smooth_cache[layer_idx]

                # Shape check
                if cached_map.shape != dynamic_map.shape:
                    self.v_smooth_cache[layer_idx] = (
                        v_perm.detach().clone(),
                        dynamic_map.detach().clone()
                    )
                    self.stats['v_cache_misses'] += 1
                else:
                    # Compare dynamic_map
                    change_ratio = (cached_map != dynamic_map).float().mean().item()

                    if change_ratio < self.cache_update_threshold:
                        # Cache hit - use cached V directly
                        v_result = cached_v
                        self.stats['v_cache_hits'] += 1
                    else:
                        # Cache miss - smooth update
                        v_smoothed = self.ema_alpha * v_perm + (1 - self.ema_alpha) * cached_v

                        # Update cache with smoothed value
                        self.v_smooth_cache[layer_idx] = (
                            v_smoothed.detach().clone(),
                            dynamic_map.detach().clone()
                        )

                        v_result = v_smoothed
                        self.stats['v_cache_misses'] += 1

        # Update legacy combined stats
        if self.enable_k_cache and self.enable_v_cache:
            k_hit = (k_result is not k_perm)
            v_hit = (v_result is not v_perm)
            if k_hit and v_hit:
                self.stats['kv_cache_hits'] += 1
            else:
                self.stats['kv_cache_misses'] += 1

        return k_result, v_result

    def reset_cache(self):
        """Override to clear smooth caches."""
        super().reset_cache()
        self.k_smooth_cache = {}
        self.v_smooth_cache = {}


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
    print("Testing V4: Temporal Smoothing Cache")
    print("=" * 60)

    # Test different EMA alpha values
    for ema_alpha in [0.1, 0.3, 0.5, 1.0]:
        print(f"\n{'='*60}")
        print(f"EMA Alpha = {ema_alpha} (alpha=1.0 is equivalent to V3)")
        print(f"{'='*60}")

        q = torch.randn(cfg, num_heads, seq_len, dim, device=device, dtype=dtype)
        k = torch.randn(cfg, num_heads, seq_len, dim, device=device, dtype=dtype)
        v = torch.randn(cfg, num_heads, seq_len, dim, device=device, dtype=dtype)

        attn = ClusterBasedSparseAttentionFastV4(
            num_q_centroids=16,
            num_k_centroids=16,
            enable_k_cache=True,
            enable_v_cache=True,
            cache_update_threshold=0.05,
            ema_alpha=ema_alpha,
        )

        # Warmup
        for _ in range(3):
            _ = attn.forward(q, k, v, layer_idx=0, tokens_per_frame=64, video_seq_len=seq_len)

        # Benchmark
        torch.cuda.synchronize()
        start = time.time()

        for i in range(num_iterations):
            q = q + torch.randn_like(q) * 0.01
            k = k + torch.randn_like(k) * 0.01
            v = v + torch.randn_like(v) * 0.01
            _ = attn.forward(q, k, v, layer_idx=0, tokens_per_frame=64, video_seq_len=seq_len)

        torch.cuda.synchronize()
        elapsed = time.time() - start

        stats = attn.get_stats()
        print(f"\nPerformance:")
        print(f"  Time: {elapsed/num_iterations*1000:.2f} ms/iter")
        print(f"\nCache Statistics:")
        print(f"  K cache hit rate: {stats.get('k_cache_hit_rate', 0):.2%}")
        print(f"  V cache hit rate: {stats.get('v_cache_hit_rate', 0):.2%}")

        if ema_alpha == 1.0:
            print(f"\nNote: alpha=1.0 means NO smoothing (equivalent to V3)")
        else:
            print(f"\nNote: alpha={ema_alpha} means {ema_alpha*100:.0f}% new + {(1-ema_alpha)*100:.0f}% old")
            print(f"      Lower alpha = smoother transitions but potentially stale cache")
