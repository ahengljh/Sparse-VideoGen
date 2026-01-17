"""
Cluster-based Sparse Attention using Sparse-VideoGen's optimized kernels

This uses the actual Sparse-VideoGen implementation with Triton kernels
for maximum performance.
"""

import torch
import torch.nn.functional as F
from loguru import logger
from typing import Optional, Dict, Tuple

# Import Sparse-VideoGen's optimized functions
from svg_kmeans_utils import (
    batch_kmeans_Euclid,
    identify_dynamic_map,
    density_calculation,
    dynamic_block_sparse_fwd_triton,
)
from svg_kernels.triton.permute import (
    permute_tensor_by_labels_triton,
    apply_inverse_permutation_triton,
)


class ClusterBasedSparseAttentionFast:
    """
    Fast cluster-based sparse attention using Sparse-VideoGen's Triton kernels.

    This replaces the slow PyTorch implementation with optimized Triton kernels.
    """

    def __init__(
        self,
        num_q_centroids: int = 64,
        num_k_centroids: int = 64,
        top_p_kmeans: float = 0.9,
        min_kc_ratio: float = 0.1,
        kmeans_iter_init: int = 10,
        kmeans_iter_step: int = 3,
        enable_first_frame_sink: bool = True,
        enable_kv_cache: bool = False,
        enable_k_cache: bool = None,  # None means follow enable_kv_cache
        enable_v_cache: bool = None,  # None means follow enable_kv_cache
        cache_update_threshold: float = 0.05,
        enable_dynamic_threshold: bool = False,
        base_threshold: float = 0.03,
        max_threshold: float = 0.15,
        total_layers: int = 60,
        total_steps: int = 30,
    ):
        """
        Args:
            num_q_centroids: Number of query clusters
            num_k_centroids: Number of key clusters
            top_p_kmeans: Keep top-p cumulative probability of cluster pairs
            min_kc_ratio: Minimum ratio of key clusters to keep
            kmeans_iter_init: K-means iterations for initialization
            kmeans_iter_step: K-means iterations for subsequent steps
            enable_first_frame_sink: Whether to ensure first frame attends to all
            enable_kv_cache: Whether to enable KV caching (both K and V, ~14% speedup)
            enable_k_cache: Whether to enable K caching only (None means follow enable_kv_cache)
            enable_v_cache: Whether to enable V caching only (None means follow enable_kv_cache)
            cache_update_threshold: Dynamic map change threshold for cache invalidation (default 0.05)
            enable_dynamic_threshold: Enable adaptive threshold based on layer and step
            base_threshold: Base threshold for early layers/steps (default 0.03)
            max_threshold: Max threshold for late layers/steps (default 0.15)
            total_layers: Total number of layers (for normalization, default 60)
            total_steps: Total inference steps (for normalization, default 30)
        """
        self.num_q_centroids = num_q_centroids
        self.num_k_centroids = num_k_centroids
        self.top_p_kmeans = top_p_kmeans
        self.min_kc_ratio = min_kc_ratio
        self.kmeans_iter_init = kmeans_iter_init
        self.kmeans_iter_step = kmeans_iter_step
        self.enable_first_frame_sink = enable_first_frame_sink

        # Fine-grained cache control
        self.enable_kv_cache = enable_kv_cache
        self.enable_k_cache = enable_k_cache if enable_k_cache is not None else enable_kv_cache
        self.enable_v_cache = enable_v_cache if enable_v_cache is not None else enable_kv_cache
        self.cache_update_threshold = cache_update_threshold

        # Dynamic threshold control
        self.enable_dynamic_threshold = enable_dynamic_threshold
        self.base_threshold = base_threshold
        self.max_threshold = max_threshold
        self.total_layers = total_layers
        self.total_steps = total_steps
        self.current_step = 0  # Will be updated externally

        # Cache for centroids across layers
        self.centroids_initialized = {}
        self.q_centroids_cache = {}
        self.k_centroids_cache = {}

        # KV cache (when enabled) - using whole-tensor strategy for best performance
        # Separate K and V caches for fine-grained control
        self.k_tensor_cache = {}   # {layer_idx: (k_perm, dynamic_map)}
        self.v_tensor_cache = {}   # {layer_idx: (v_perm, dynamic_map)}

        # Legacy: combined KV cache (for backward compatibility)
        self.kv_tensor_cache = {}  # {layer_idx: (k_perm, v_perm, dynamic_map)}

        # Legacy block-level cache (kept for compatibility, not used by default)
        self.kv_block_cache = {}  # {(layer_idx, b, h, k_idx): (k_block, v_block)}
        self.dynamic_map_cache = {}  # {layer_idx: dynamic_map}

        # Log cache configuration
        if self.enable_k_cache or self.enable_v_cache:
            cache_mode = []
            if self.enable_k_cache:
                cache_mode.append("K")
            if self.enable_v_cache:
                cache_mode.append("V")
            logger.info("Tensor caching enabled: {} (whole-tensor strategy). "
                       "Cached tensors will be reused when dynamic_map change < {:.1%}".format(
                           "+".join(cache_mode), cache_update_threshold))

        # Statistics
        self.stats = {
            'total_calls': 0,
            'avg_density': 0.0,
            'kmeans_iters': 0,
            'k_cache_hits': 0,
            'k_cache_misses': 0,
            'v_cache_hits': 0,
            'v_cache_misses': 0,
            # Legacy stats (for backward compatibility)
            'kv_cache_hits': 0,
            'kv_cache_misses': 0,
            'kv_blocks_recomputed': 0,
            'kv_blocks_total': 0,
        }

    def kmeans_clustering(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        layer_idx: int
    ) -> Tuple:
        """
        Perform K-means clustering on query and key using Sparse-VideoGen's Triton kernels.

        Args:
            query: (cfg, num_heads, seq_len, dim)
            key: (cfg, num_heads, seq_len, dim)
            layer_idx: current layer index

        Returns:
            Tuple of (q_labels, q_centroids, q_sizes, q_iter,
                     k_labels, k_centroids, k_sizes, k_iter)
        """
        cfg, num_heads, seq_len, dim = query.shape

        # Reshape to (cfg*num_heads, seq_len, dim) for batched K-means
        q_flat = query.reshape(cfg * num_heads, seq_len, dim)
        k_flat = key.reshape(cfg * num_heads, seq_len, dim)

        # Check if centroids are initialized for this layer
        if layer_idx not in self.centroids_initialized:
            # Initial clustering with more iterations
            q_labels, q_centroids, q_sizes, q_iter = batch_kmeans_Euclid(
                q_flat,
                n_clusters=self.num_q_centroids,
                max_iters=self.kmeans_iter_init,
                init_centroids=None,
            )
            k_labels, k_centroids, k_sizes, k_iter = batch_kmeans_Euclid(
                k_flat,
                n_clusters=self.num_k_centroids,
                max_iters=self.kmeans_iter_init,
                init_centroids=None,
            )

            # Cache centroids
            self.q_centroids_cache[layer_idx] = q_centroids
            self.k_centroids_cache[layer_idx] = k_centroids
            self.centroids_initialized[layer_idx] = True

            logger.debug(f"Layer {layer_idx}: Initialized centroids with {q_iter}/{k_iter} K-means iters")
        else:
            # Use cached centroids as initialization (warm start)
            q_labels, q_centroids, q_sizes, q_iter = batch_kmeans_Euclid(
                q_flat,
                n_clusters=self.num_q_centroids,
                max_iters=self.kmeans_iter_step,
                init_centroids=self.q_centroids_cache[layer_idx],
            )
            k_labels, k_centroids, k_sizes, k_iter = batch_kmeans_Euclid(
                k_flat,
                n_clusters=self.num_k_centroids,
                max_iters=self.kmeans_iter_step,
                init_centroids=self.k_centroids_cache[layer_idx],
            )

            # Update cache
            self.q_centroids_cache[layer_idx] = q_centroids
            self.k_centroids_cache[layer_idx] = k_centroids

        self.stats['kmeans_iters'] = (q_iter + k_iter) / 2

        return q_labels, q_centroids, q_sizes, q_iter, k_labels, k_centroids, k_sizes, k_iter

    def semantic_aware_permutation(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        layer_idx: int,
        tokens_per_frame: int,
    ) -> Tuple:
        """
        Perform semantic-aware permutation using K-means clustering and Triton kernels.

        Args:
            query: (cfg, num_heads, seq_len, dim)
            key: (cfg, num_heads, seq_len, dim)
            value: (cfg, num_heads, seq_len, dim)
            layer_idx: current layer index
            tokens_per_frame: number of tokens per frame

        Returns:
            Tuple of (q_perm, k_perm, v_perm, dynamic_map,
                     q_cluster_sizes, k_cluster_sizes, q_sorted_indices)
        """
        cfg, num_heads, seq_len, dim = query.shape

        # 1. K-means clustering
        q_labels, q_centroids, q_sizes, q_iter, k_labels, k_centroids, k_sizes, k_iter = \
            self.kmeans_clustering(query, key, layer_idx)

        # 2. Reshape cluster info
        q_cluster_sizes = q_sizes.view(cfg, num_heads, self.num_q_centroids)
        k_cluster_sizes = k_sizes.view(cfg, num_heads, self.num_k_centroids)
        q_centroids_reshaped = q_centroids.view(cfg, num_heads, self.num_q_centroids, dim)
        k_centroids_reshaped = k_centroids.view(cfg, num_heads, self.num_k_centroids, dim)

        # 3. Identify dynamic map (which cluster pairs should attend)
        dynamic_map = identify_dynamic_map(
            q_centroids_reshaped,
            k_centroids_reshaped,
            q_cluster_sizes,
            k_cluster_sizes,
            self.top_p_kmeans,
            self.min_kc_ratio,
        )

        # 4. Add first-frame sink if enabled
        if self.enable_first_frame_sink and tokens_per_frame > 0:
            # First frame clusters should attend to all key clusters
            first_frame_clusters = tokens_per_frame // (seq_len // self.num_q_centroids + 1)
            first_frame_clusters = min(first_frame_clusters, self.num_q_centroids)
            dynamic_map[:, :, :first_frame_clusters, :] = True

        # 5. Permute query, key, value by cluster labels using Triton kernels
        # NOTE: Sparse-VideoGen's permute_tensor_by_labels_triton expects dim==2
        # So we keep tensors as (cfg, num_heads, seq_len, dim) and permute dim=2

        # Permute using fast Triton kernels (dim=2 for sequence dimension in 4D tensor)
        q_permuted, q_sorted_indices = permute_tensor_by_labels_triton(query, q_labels, dim=2)
        k_permuted, k_sorted_indices = permute_tensor_by_labels_triton(key, k_labels, dim=2)
        v_permuted, _ = permute_tensor_by_labels_triton(value, k_labels, dim=2)

        return q_permuted, k_permuted, v_permuted, dynamic_map, q_cluster_sizes, k_cluster_sizes, q_sorted_indices

    def _should_recompute_k_block(
        self,
        layer_idx: int,
        b: int,
        h: int,
        k_idx: int,
        current_dynamic_map: torch.Tensor
    ) -> bool:
        """Check if a K block needs recomputation based on dynamic_map changes."""
        if not self.enable_kv_cache:
            return True

        cache_key = (layer_idx, b, h, k_idx)
        if cache_key not in self.kv_block_cache:
            return True

        if layer_idx not in self.dynamic_map_cache:
            return True

        old_map = self.dynamic_map_cache[layer_idx]
        old_usage = old_map[b, h, :, k_idx]
        new_usage = current_dynamic_map[b, h, :, k_idx]

        if torch.equal(old_usage, new_usage):
            return False

        self.stats['kv_blocks_recomputed'] += 1
        return True

    def _update_dynamic_map_cache(self, layer_idx: int, dynamic_map: torch.Tensor):
        """Update the dynamic_map cache for change tracking."""
        if not self.enable_kv_cache:
            return

        old_map = self.dynamic_map_cache.get(layer_idx, None)
        if old_map is not None:
            changed = (old_map != dynamic_map).float().mean().item()
            if changed < self.cache_update_threshold:
                # Silent - this is the common case when caching is effective
                return

        self.dynamic_map_cache[layer_idx] = dynamic_map.detach().clone()

    def _get_adaptive_threshold(self, layer_idx: int) -> float:
        """
        Calculate adaptive cache threshold based on layer index and current step.

        Strategy:
        - Early layers (0-30%) + Early steps (0-30%): strict threshold (base)
        - Late layers (70-100%) + Late steps (70-100%): relaxed threshold (max)
        - Gradual transition in between

        Returns:
            Adaptive threshold value
        """
        if not self.enable_dynamic_threshold:
            return self.cache_update_threshold

        # Normalize layer index (0.0 = first layer, 1.0 = last layer)
        layer_ratio = layer_idx / max(self.total_layers - 1, 1)

        # Normalize step index (0.0 = first step, 1.0 = last step)
        step_ratio = self.current_step / max(self.total_steps - 1, 1)

        # Combined stability factor (average of layer and step)
        # Higher = more stable = can use higher threshold
        stability_factor = (layer_ratio + step_ratio) / 2.0

        # Non-linear scaling: use sqrt to be more conservative early on
        # This makes threshold increase slowly at first, then faster
        stability_factor = stability_factor ** 0.7

        # Calculate adaptive threshold
        threshold = self.base_threshold + (self.max_threshold - self.base_threshold) * stability_factor

        return threshold

    def set_current_step(self, step: int):
        """Update current inference step for dynamic threshold calculation."""
        self.current_step = step

    def _try_reuse_kv_tensors(
        self,
        k_perm: torch.Tensor,
        v_perm: torch.Tensor,
        dynamic_map: torch.Tensor,
        layer_idx: int,
        tokens_per_frame: int = 0,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Try to reuse cached K and/or V tensors (hybrid whole-tensor + first-frame strategy).

        This is the optimized V3 implementation with fine-grained K/V control and
        first-frame sink support.

        Strategy:
        - Cache K and/or V tensors per layer based on configuration
        - Compare dynamic_map change with dynamic threshold
        - Return cached tensors when change < threshold

        First-frame sink support (if enabled):
        - When cache hit occurs, replace first frame tokens with fresh values
        - This ensures first frame is always up-to-date while other frames benefit from cache
        - Allows dynamic threshold to work for all frames

        Args:
            k_perm: Permuted K tensor (cfg, num_heads, seq_len, dim)
            v_perm: Permuted V tensor (cfg, num_heads, seq_len, dim)
            dynamic_map: (cfg, num_heads, qc_num, kc_num)
            layer_idx: current layer index
            tokens_per_frame: number of tokens per frame (0 if unknown)

        Returns:
            (k_perm, v_perm) - hybrid result with first frame fresh, others potentially cached
        """
        # Quick exit if no caching enabled
        if not self.enable_k_cache and not self.enable_v_cache:
            return k_perm, v_perm

        k_result = k_perm
        v_result = v_perm

        # Determine if we need to handle first-frame specially
        use_first_frame_sink = self.enable_first_frame_sink and tokens_per_frame > 0
        first_frame_tokens = tokens_per_frame if use_first_frame_sink else 0

        # ===== Try to reuse K tensor =====
        if self.enable_k_cache:
            if layer_idx not in self.k_tensor_cache:
                # First call - cache K
                self.k_tensor_cache[layer_idx] = (
                    k_perm.detach().clone(),
                    dynamic_map.detach().clone()
                )
                self.stats['k_cache_misses'] += 1
            else:
                # Check if we can reuse cached K
                cached_k, cached_map = self.k_tensor_cache[layer_idx]

                # Shape check
                if cached_map.shape != dynamic_map.shape:
                    # Shape changed - update cache
                    self.k_tensor_cache[layer_idx] = (
                        k_perm.detach().clone(),
                        dynamic_map.detach().clone()
                    )
                    self.stats['k_cache_misses'] += 1
                else:
                    # Compare dynamic_map
                    change_ratio = (cached_map != dynamic_map).float().mean().item()

                    # Get adaptive threshold (may vary by layer and step)
                    threshold = self._get_adaptive_threshold(layer_idx)

                    if change_ratio < threshold:
                        # Cache hit! Reuse cached K
                        if use_first_frame_sink:
                            # Hybrid: use cached K but replace first frame with fresh values
                            k_result = cached_k.clone()
                            k_result[:, :, :first_frame_tokens, :] = k_perm[:, :, :first_frame_tokens, :]
                        else:
                            # Full cache hit
                            k_result = cached_k
                        self.stats['k_cache_hits'] += 1
                    else:
                        # Dynamic map changed - update cache
                        self.k_tensor_cache[layer_idx] = (
                            k_perm.detach().clone(),
                            dynamic_map.detach().clone()
                        )
                        self.stats['k_cache_misses'] += 1

        # ===== Try to reuse V tensor =====
        if self.enable_v_cache:
            if layer_idx not in self.v_tensor_cache:
                # First call - cache V
                self.v_tensor_cache[layer_idx] = (
                    v_perm.detach().clone(),
                    dynamic_map.detach().clone()
                )
                self.stats['v_cache_misses'] += 1
            else:
                # Check if we can reuse cached V
                cached_v, cached_map = self.v_tensor_cache[layer_idx]

                # Shape check
                if cached_map.shape != dynamic_map.shape:
                    # Shape changed - update cache
                    self.v_tensor_cache[layer_idx] = (
                        v_perm.detach().clone(),
                        dynamic_map.detach().clone()
                    )
                    self.stats['v_cache_misses'] += 1
                else:
                    # Compare dynamic_map
                    change_ratio = (cached_map != dynamic_map).float().mean().item()

                    # Get adaptive threshold (may vary by layer and step)
                    threshold = self._get_adaptive_threshold(layer_idx)

                    if change_ratio < threshold:
                        # Cache hit! Reuse cached V
                        if use_first_frame_sink:
                            # Hybrid: use cached V but replace first frame with fresh values
                            v_result = cached_v.clone()
                            v_result[:, :, :first_frame_tokens, :] = v_perm[:, :, :first_frame_tokens, :]
                        else:
                            # Full cache hit
                            v_result = cached_v
                        self.stats['v_cache_hits'] += 1
                    else:
                        # Dynamic map changed - update cache
                        self.v_tensor_cache[layer_idx] = (
                            v_perm.detach().clone(),
                            dynamic_map.detach().clone()
                        )
                        self.stats['v_cache_misses'] += 1

        # Update legacy combined stats for backward compatibility
        if self.enable_k_cache and self.enable_v_cache:
            # Both K and V enabled - update legacy KV stats
            k_hit = (k_result is not k_perm)
            v_hit = (v_result is not v_perm)
            if k_hit and v_hit:
                self.stats['kv_cache_hits'] += 1
            else:
                self.stats['kv_cache_misses'] += 1

        return k_result, v_result

    def dynamic_map_post_processing(
        self,
        q_perm: torch.Tensor,
        k_perm: torch.Tensor,
        v_perm: torch.Tensor,
        query_full: torch.Tensor,
        key_full: torch.Tensor,
        value_full: torch.Tensor,
        dyn_map: torch.Tensor,
        qc_sz_s: torch.Tensor,
        kc_sz_s: torch.Tensor,
        q_sorted_indices: torch.Tensor,
        video_length: int,
        text_length: int,
    ):
        """
        Post-process dynamic map to enable video-text interaction.

        Following Sparse-VideoGen's implementation exactly:
        - Video part: uses cluster sparse (already permuted)
        - Text part: treated as additional clusters with dense attention patterns
        - Video can attend to text, text can attend to video

        Args:
            q_perm, k_perm, v_perm: Permuted video tokens (cfg, num_heads, video_len, dim)
            query_full, key_full, value_full: Full tensors including text (cfg, num_heads, total_len, dim)
            dyn_map: Dynamic map for video clusters (cfg, num_heads, num_q_clusters, num_k_clusters)
            qc_sz_s, kc_sz_s: Cluster sizes (cfg, num_heads, num_clusters)
            q_sorted_indices: Sorting indices for inverse permutation (num_heads, video_len)
            video_length: Number of video tokens
            text_length: Number of text tokens

        Returns:
            Updated query, key, value, dyn_map, qc_sz_s, kc_sz_s, q_sorted_indices
        """
        cfg, num_heads, total_len, dim = query_full.shape

        # 1. Create new tensors with permuted video + original text
        # Strategy: Put permuted video tokens at the beginning, text tokens at the end
        query_combined = query_full.clone()
        key_combined = key_full.clone()
        value_combined = value_full.clone()

        # Replace video part with permuted version
        query_combined[:, :, :video_length, :] = q_perm
        key_combined[:, :, :video_length, :] = k_perm
        value_combined[:, :, :video_length, :] = v_perm

        # Text tokens remain in their original positions (video_length:)
        # No need to modify, they're already there

        # 2. Extend dynamic map to include text clusters
        # Text is treated as 1 additional cluster (or could split into prompt/unprompt)
        # For simplicity, we treat all text as one cluster
        # dyn_map shape: (cfg, num_heads, num_q_clusters, num_k_clusters)
        # Add 1 row and 1 column for the text cluster

        # Pad dynamic map: add 1 cluster for text in both dimensions
        dyn_map = F.pad(dyn_map, (0, 1, 0, 1), value=False)

        # Text cluster (last row) can attend to all video clusters + itself
        dyn_map[:, :, -1, :] = True  # Text queries attend to all keys

        # All video clusters can attend to text cluster
        dyn_map[:, :, :, -1] = True  # All queries can attend to text keys

        # 3. Update cluster sizes to include text cluster
        qc_sz_s = F.pad(qc_sz_s, (0, 1), value=0)
        qc_sz_s[:, :, -1] = text_length  # Text cluster size

        kc_sz_s = F.pad(kc_sz_s, (0, 1), value=0)
        kc_sz_s[:, :, -1] = text_length  # Text cluster size

        # 4. Update sorted indices to include text tokens
        # q_sorted_indices shape: (num_heads, video_length)
        # We need to add text token indices at the end
        q_sorted_indices = F.pad(q_sorted_indices, (0, text_length), value=0)

        # Text tokens are not permuted, they stay in order
        # Indices: video_length, video_length+1, ..., video_length+text_length-1
        text_indices = torch.arange(
            video_length, video_length + text_length,
            device=q_sorted_indices.device,
            dtype=q_sorted_indices.dtype
        )
        q_sorted_indices[:, video_length:] = text_indices

        # Add batch dimension if needed (Sparse-VideoGen does this)
        if q_sorted_indices.dim() == 2:
            q_sorted_indices = q_sorted_indices.unsqueeze(0)

        return query_combined, key_combined, value_combined, dyn_map, qc_sz_s, kc_sz_s, q_sorted_indices

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        layer_idx: int,
        tokens_per_frame: int,
        video_seq_len: int = None,
    ) -> torch.Tensor:
        """
        Forward pass for cluster-based sparse attention using Sparse-VideoGen kernels.

        Now supports video-text interaction via dynamic_map_post_processing!

        Args:
            query: (cfg, num_heads, total_seq_len, dim) - includes video + text
            key: (cfg, num_heads, total_seq_len, dim) - includes video + text
            value: (cfg, num_heads, total_seq_len, dim) - includes video + text
            layer_idx: current layer index
            tokens_per_frame: number of tokens per frame
            video_seq_len: number of video tokens (if None, assumes all tokens are video)

        Returns:
            output: (cfg, num_heads, total_seq_len, dim)
        """
        cfg, num_heads, total_seq_len, dim = query.shape

        # Determine video and text lengths
        if video_seq_len is None:
            video_seq_len = total_seq_len  # All tokens are video
        text_seq_len = total_seq_len - video_seq_len

        # 1. Extract video part for clustering
        query_video = query[:, :, :video_seq_len, :].contiguous()
        key_video = key[:, :, :video_seq_len, :].contiguous()
        value_video = value[:, :, :video_seq_len, :].contiguous()

        # 2. Semantic-aware permutation on VIDEO ONLY
        q_perm, k_perm, v_perm, dynamic_map, q_cluster_sizes, k_cluster_sizes, q_sorted_indices = \
            self.semantic_aware_permutation(query_video, key_video, value_video, layer_idx, tokens_per_frame)

        # 2.5. Try to reuse cached K/V tensors (whole-tensor strategy - V3)
        # Support fine-grained K/V caching
        if self.enable_k_cache or self.enable_v_cache:
            k_perm, v_perm = self._try_reuse_kv_tensors(
                k_perm, v_perm, dynamic_map, layer_idx, tokens_per_frame
            )

        # 3. Dynamic map post-processing to enable video-text interaction
        if text_seq_len > 0:
            # Have text tokens - use post-processing to combine video and text
            query_combined, key_combined, value_combined, dynamic_map_extended, \
            qc_sz_extended, kc_sz_extended, q_sorted_indices_extended = \
                self.dynamic_map_post_processing(
                    q_perm, k_perm, v_perm,
                    query, key, value,
                    dynamic_map, q_cluster_sizes, k_cluster_sizes, q_sorted_indices,
                    video_seq_len, text_seq_len
                )

            # 4. Block sparse attention on combined sequence
            output_permuted = dynamic_block_sparse_fwd_triton(
                query_combined, key_combined, value_combined,
                dynamic_map_extended, qc_sz_extended, kc_sz_extended
            )

            # 5. Apply inverse permutation
            output = apply_inverse_permutation_triton(output_permuted, q_sorted_indices_extended, dim=2)

            # Update statistics
            density = density_calculation(dynamic_map_extended, qc_sz_extended, kc_sz_extended)
        else:
            # No text tokens - just video
            # 4. Block sparse attention on permuted video tokens
            output_permuted = dynamic_block_sparse_fwd_triton(
                q_perm, k_perm, v_perm, dynamic_map, q_cluster_sizes, k_cluster_sizes
            )

            # 5. Apply inverse permutation
            output = apply_inverse_permutation_triton(output_permuted, q_sorted_indices, dim=2)

            # Update statistics
            density = density_calculation(dynamic_map, q_cluster_sizes, k_cluster_sizes)

        self.stats['total_calls'] += 1
        self.stats['avg_density'] = density.mean().item()

        return output

    def get_stats(self) -> Dict:
        """Get statistics about the clustering and sparsity."""
        stats = {
            'total_calls': self.stats['total_calls'],
            'avg_density': self.stats['avg_density'],
            'avg_sparsity': 1.0 - self.stats['avg_density'],
            'kmeans_iters': self.stats['kmeans_iters'],
            'num_q_centroids': self.num_q_centroids,
            'num_k_centroids': self.num_k_centroids,
            'top_p_kmeans': self.top_p_kmeans,
        }

        # Add fine-grained K/V cache statistics
        if self.enable_k_cache or self.enable_v_cache:
            # K cache stats
            if self.enable_k_cache:
                k_total = self.stats['k_cache_hits'] + self.stats['k_cache_misses']
                k_hit_rate = self.stats['k_cache_hits'] / max(1, k_total)
                stats.update({
                    'k_cache_enabled': True,
                    'k_cache_hit_rate': k_hit_rate,
                    'k_cache_hits': self.stats['k_cache_hits'],
                    'k_cache_misses': self.stats['k_cache_misses'],
                })
            else:
                stats['k_cache_enabled'] = False

            # V cache stats
            if self.enable_v_cache:
                v_total = self.stats['v_cache_hits'] + self.stats['v_cache_misses']
                v_hit_rate = self.stats['v_cache_hits'] / max(1, v_total)
                stats.update({
                    'v_cache_enabled': True,
                    'v_cache_hit_rate': v_hit_rate,
                    'v_cache_hits': self.stats['v_cache_hits'],
                    'v_cache_misses': self.stats['v_cache_misses'],
                })
            else:
                stats['v_cache_enabled'] = False

            # Legacy combined KV stats (for backward compatibility)
            if self.enable_k_cache and self.enable_v_cache:
                total_accesses = self.stats['kv_cache_hits'] + self.stats['kv_cache_misses']
                hit_rate = self.stats['kv_cache_hits'] / max(1, total_accesses)
                stats.update({
                    'kv_cache_enabled': True,
                    'kv_cache_hit_rate': hit_rate,
                    'kv_cache_hits': self.stats['kv_cache_hits'],
                    'kv_cache_misses': self.stats['kv_cache_misses'],
                })
            else:
                stats['kv_cache_enabled'] = (self.enable_k_cache or self.enable_v_cache)
        else:
            stats.update({
                'k_cache_enabled': False,
                'v_cache_enabled': False,
                'kv_cache_enabled': False,
            })

        return stats

    def reset_cache(self):
        """Reset all caches (useful when starting new video generation)."""
        self.centroids_initialized = {}
        self.q_centroids_cache = {}
        self.k_centroids_cache = {}
        self.kv_tensor_cache = {}  # Legacy combined KV cache
        self.k_tensor_cache = {}   # Fine-grained K cache
        self.v_tensor_cache = {}   # Fine-grained V cache
        self.kv_block_cache = {}  # Legacy (unused)
        self.dynamic_map_cache = {}  # Legacy (unused)
        self.stats = {
            'total_calls': 0,
            'avg_density': 0.0,
            'kmeans_iters': 0,
            'kv_cache_hits': 0,
            'kv_cache_misses': 0,
            'k_cache_hits': 0,
            'k_cache_misses': 0,
            'v_cache_hits': 0,
            'v_cache_misses': 0,
            'kv_blocks_recomputed': 0,
            'kv_blocks_total': 0,
        }
