"""
Cluster-based Sparse Attention for HunyuanVideo

This script demonstrates the advanced cluster-based sparse attention mode,
which uses K-means clustering to identify which token blocks should attend to each other.

Features:
- GPU-accelerated K-means clustering using Sparse-VideoGen's Triton kernels
- Semantic-aware token permutation for efficient block-sparse computation
- Dynamic cluster pair selection via top-p thresholding
- Sparse-VideoGen style warmup (optional): early timesteps use dense, later use sparse
- First-frame sink mechanism for global information flow

Basic usage:
    python run_sparse_cluster.py \\
        --prompt "A cat walks on the grass" \\
        --video-length 17 \\
        --video-size 256 448 \\
        --infer-steps 50 \\
        --seed 42 \\
        --flow-reverse \\
        --num-q-centroids 64 \\
        --num-k-centroids 64 \\
        --top-p-kmeans 0.9

With Sparse-VideoGen warmup:
    python run_sparse_cluster.py \\
        --prompt "A cat walks on the grass" \\
        --video-length 17 \\
        --video-size 256 448 \\
        --infer-steps 50 \\
        --seed 42 \\
        --flow-reverse \\
        --num-q-centroids 64 \\
        --num-k-centroids 64 \\
        --top-p-kmeans 0.9 \\
        --enable-warmup \\
        --warmup-steps-ratio 0.2 \\
        --zero-step-kmeans-init
"""

import argparse
import os
import time
from pathlib import Path
from loguru import logger
from datetime import datetime
import torch
import torch.nn.functional as F

from hyvideo.utils.file_utils import save_videos_grid
from hyvideo.config import parse_args
from hyvideo.inference import HunyuanVideoSampler

# Import cluster-based sparse attention (fast version with Triton kernels)
from sparse_attention_cluster_fast import ClusterBasedSparseAttentionFast


# Global variables for attention patching
_cluster_sparse_attn = None
_original_attention_func = None
_sparse_enabled = False
_sparse_stats = {
    'total_calls': 0,
    'sparse_calls': 0,
    'avg_density': 0.0,
}

# Global variables for warmup (Sparse-VideoGen style)
_current_step_index = None  # Track current inference step index (0, 1, 2, ...)
_total_inference_steps = None  # Total inference steps
_total_layers = 60  # HunyuanVideo has ~60 layers
_warmup_config = {
    'enabled': False,
    'first_times_fp_ratio': 0.075,  # First 7.5% of steps use dense (Sparse-VideoGen default)
    'first_layers_fp_ratio': 0.025,  # First 2.5% of layers use dense (Sparse-VideoGen default)
    'zero_step_kmeans_init': False,  # Initialize centroids during warmup
    'num_fp_steps': 0,  # Number of warmup steps (calculated)
    'num_fp_layers': 0,  # Number of warmup layers (calculated)
}


def patch_attention_for_cluster_sparse(
    num_q_centroids=64,
    num_k_centroids=64,
    top_p_kmeans=0.9,
    min_kc_ratio=0.1,
    kmeans_iter_init=10,
    kmeans_iter_step=3,
    video_length=17,
    video_size=(256, 448),
    vae='884-16c-hy',
    enable_warmup=False,
    first_times_fp_ratio=0.075,
    first_layers_fp_ratio=0.025,
    zero_step_kmeans_init=False,
    total_inference_steps=30,
    enable_kv_cache=False,
    enable_k_cache=None,
    enable_v_cache=None,
    cache_update_threshold=0.05,
    enable_dynamic_threshold=False,
    base_threshold=0.03,
    max_threshold=0.15,
    enable_first_frame_sink=False,
):
    """
    Patch HunyuanVideo's attention function to use cluster-based sparse attention.

    Args:
        enable_warmup: Enable Sparse-VideoGen style warmup (early steps/layers use dense)
        first_times_fp_ratio: Ratio of early steps that use dense (default: 0.075 = 7.5%)
        first_layers_fp_ratio: Ratio of early layers that use dense (default: 0.025 = 2.5%)
        zero_step_kmeans_init: Initialize K-means centroids during warmup phase
        total_inference_steps: Total number of inference steps (for warmup calculation)
        enable_first_frame_sink: Enable first-frame sink (first frame uses dense + skips cache)
    """
    global _cluster_sparse_attn, _original_attention_func, _sparse_enabled, _sparse_stats, _warmup_config
    global _total_inference_steps, _total_layers

    # Calculate video tokens
    height, width = video_size
    if '884' in str(vae):
        latents_t = (video_length - 1) // 4 + 1
    elif '888' in str(vae):
        latents_t = (video_length - 1) // 8 + 1
    else:
        latents_t = video_length

    latents_h = height // 8
    latents_w = width // 8

    patch_size = [1, 2, 2]
    rope_t = latents_t // patch_size[0]
    rope_h = latents_h // patch_size[1]
    rope_w = latents_w // patch_size[2]

    tokens_per_frame = rope_h * rope_w
    num_frames = rope_t

    logger.info(f"Video config: {num_frames} frames × {tokens_per_frame} tokens/frame = {num_frames * tokens_per_frame} video tokens")

    # Set warmup configuration (Sparse-VideoGen style)
    _total_inference_steps = total_inference_steps
    _warmup_config['enabled'] = enable_warmup
    _warmup_config['first_times_fp_ratio'] = first_times_fp_ratio
    _warmup_config['first_layers_fp_ratio'] = first_layers_fp_ratio
    _warmup_config['zero_step_kmeans_init'] = zero_step_kmeans_init

    # Calculate warmup steps and layers (following Sparse-VideoGen logic)
    import math
    _warmup_config['num_fp_steps'] = math.floor(first_times_fp_ratio * total_inference_steps)
    _warmup_config['num_fp_layers'] = math.floor(first_layers_fp_ratio * _total_layers)

    logger.info(f"Warmup configuration:")
    logger.info(f"  Inference steps warmup: {_warmup_config['num_fp_steps']} / {total_inference_steps} ({first_times_fp_ratio*100:.1f}%)")
    logger.info(f"  Layer warmup: {_warmup_config['num_fp_layers']} / {_total_layers} ({first_layers_fp_ratio*100:.1f}%)")

    # Create cluster-based sparse attention instance (FAST version with Triton)
    _cluster_sparse_attn = ClusterBasedSparseAttentionFast(
        num_q_centroids=num_q_centroids,
        num_k_centroids=num_k_centroids,
        top_p_kmeans=top_p_kmeans,
        min_kc_ratio=min_kc_ratio,
        kmeans_iter_init=kmeans_iter_init,
        kmeans_iter_step=kmeans_iter_step,
        enable_first_frame_sink=enable_first_frame_sink,
        enable_kv_cache=enable_kv_cache,
        enable_k_cache=enable_k_cache,
        enable_v_cache=enable_v_cache,
        cache_update_threshold=cache_update_threshold,
        enable_dynamic_threshold=enable_dynamic_threshold,
        base_threshold=base_threshold,
        max_threshold=max_threshold,
        total_layers=_total_layers,
        total_steps=total_inference_steps,
    )

    # Patch the attention function
    try:
        from hyvideo.modules import attenion
        _original_attention_func = attenion.attention

        def cluster_sparse_attention_wrapper(
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
            global _sparse_stats, _current_step_index, _warmup_config

            _sparse_stats['total_calls'] += 1

            # Debug: log first few calls
            if _sparse_stats['total_calls'] <= 5:
                logger.info(f"Attention wrapper called #{_sparse_stats['total_calls']}: q.shape={q.shape}, mode={mode}, q.ndim={q.ndim}")

            # Determine layer index (based on total_calls, not sparse_calls to avoid dead loop)
            layer_idx = _sparse_stats['total_calls'] % _total_layers

            # Check warmup conditions (Sparse-VideoGen style)
            use_dense_attention = False
            warmup_reason = None

            if _warmup_config['enabled']:
                # Layer warmup: early layers use dense
                if layer_idx < _warmup_config['num_fp_layers']:
                    use_dense_attention = True
                    warmup_reason = f"early layer ({layer_idx} < {_warmup_config['num_fp_layers']})"

                # Step warmup: early inference steps use dense
                if _current_step_index is not None and _current_step_index < _warmup_config['num_fp_steps']:
                    use_dense_attention = True
                    warmup_reason = f"early step ({_current_step_index} < {_warmup_config['num_fp_steps']})"

            if use_dense_attention:
                # Use original dense attention during warmup
                if _sparse_stats['total_calls'] <= 5:
                    logger.info(f"Warmup active: Using DENSE attention ({warmup_reason})")

                # During warmup with zero_step_kmeans_init, initialize centroids
                # (but don't use sparse attention yet)
                if _warmup_config['zero_step_kmeans_init'] and len(q.shape) == 4:
                    try:
                        # Extract dimensions
                        batch, seq_len, num_heads, head_dim = q.shape
                        video_seq_len = num_frames * tokens_per_frame

                        # Transpose to (batch, num_heads, seq_len, head_dim)
                        q_transposed = q.transpose(1, 2)
                        k_transposed = k.transpose(1, 2)

                        # Extract video part only
                        q_video = q_transposed[:, :, :video_seq_len, :].contiguous()
                        k_video = k_transposed[:, :, :video_seq_len, :].contiguous()

                        # Run K-means clustering to initialize centroids
                        # This doesn't return output, just initializes the centroids cache
                        _ = _cluster_sparse_attn.kmeans_clustering(q_video, k_video, layer_idx)

                        # Log first few initializations
                        if _sparse_stats['total_calls'] <= 3:
                            logger.info(f"  Zero-step K-means init: Initialized centroids for layer {layer_idx}")

                    except Exception as e:
                        # Don't fail if initialization fails, just log warning
                        if _sparse_stats['total_calls'] <= 3:
                            logger.warning(f"  Zero-step K-means init failed: {e}")

                return _original_attention_func(
                    q, k, v, mode, drop_rate, attn_mask, causal,
                    cu_seqlens_q, cu_seqlens_kv, max_seqlen_q, max_seqlen_kv, batch_size
                )

            # Handle input: q is (batch, seq_len, num_heads, head_dim) before pre_attn_layout
            if len(q.shape) == 4:
                if _sparse_stats['total_calls'] <= 5:
                    logger.info(f"  -> Entering cluster sparse path (4D tensor detected)")
                batch, seq_len, num_heads, head_dim = q.shape

                # Calculate video token length (following Sparse-VideoGen approach)
                # video_seq_len = num_frames * tokens_per_frame (already calculated during patch)
                video_seq_len = num_frames * tokens_per_frame

                # Log sequence info on first call
                if _sparse_stats['sparse_calls'] == 0:
                    logger.info(f"Sequence structure:")
                    logger.info(f"  Total seq_len: {seq_len}")
                    logger.info(f"  Video tokens: {video_seq_len} ({num_frames} frames × {tokens_per_frame} tokens/frame)")
                    logger.info(f"  Text tokens: {seq_len - video_seq_len}")

                # Transpose to (batch, num_heads, seq_len, head_dim) for processing
                q_transposed = q.transpose(1, 2)
                k_transposed = k.transpose(1, 2)
                v_transposed = v.transpose(1, 2)

                # Apply cluster-based sparse attention to ENTIRE SEQUENCE (video + text)
                # The forward method now handles video-text interaction internally via dynamic_map_post_processing
                try:
                    output_transposed = _cluster_sparse_attn.forward(
                        q_transposed, k_transposed, v_transposed,
                        layer_idx=layer_idx,
                        tokens_per_frame=tokens_per_frame,
                        video_seq_len=video_seq_len,  # Pass video length for post-processing
                    )

                    # Transpose back to (batch, seq_len, num_heads, head_dim)
                    output = output_transposed.transpose(1, 2)

                    # Reshape to (batch, seq_len, num_heads*head_dim) to match original attention output
                    output = output.reshape(batch, seq_len, -1)

                    _sparse_stats['sparse_calls'] += 1
                    stats = _cluster_sparse_attn.get_stats()
                    _sparse_stats['avg_density'] = stats['avg_density']

                    # Log first few calls
                    if _sparse_stats['sparse_calls'] <= 3:
                        text_count = seq_len - video_seq_len
                        logger.info(f"Cluster sparse attention applied (with video-text interaction):")
                        logger.info(f"  Video tokens: {video_seq_len} (cluster sparse, {num_frames} frames)")
                        if text_count > 0:
                            logger.info(f"  Text tokens: {text_count} (treated as 1 additional cluster)")
                            logger.info(f"  Video ↔ Text interaction: ENABLED via dynamic_map_post_processing")
                        logger.info(f"  Video clusters: Q={num_q_centroids}, K={num_k_centroids}")
                        logger.info(f"  Overall density: {stats['avg_density']:.2%}")
                        logger.info(f"  Overall sparsity: {stats['avg_sparsity']:.2%}")
                        logger.info(f"  K-means iters: {stats['kmeans_iters']:.1f}")

                        # Show cache info if enabled
                        if stats.get('k_cache_enabled'):
                            logger.info(f"  K cache: ENABLED (hit rate: {stats['k_cache_hit_rate']:.1%})")
                        if stats.get('v_cache_enabled'):
                            logger.info(f"  V cache: ENABLED (hit rate: {stats['v_cache_hit_rate']:.1%})")

                    return output

                except Exception as e:
                    logger.warning(f"Cluster sparse attention failed: {e}, falling back to dense")
                    import traceback
                    traceback.print_exc()
            else:
                if _sparse_stats['total_calls'] <= 5:
                    logger.info(f"  -> Skipping cluster sparse (q.shape={q.shape} is not 4D), using dense fallback")

            # Fall back to original implementation
            return _original_attention_func(
                q, k, v, mode, drop_rate, attn_mask, causal,
                cu_seqlens_q, cu_seqlens_kv, max_seqlen_q, max_seqlen_kv, batch_size
            )

        attenion.attention = cluster_sparse_attention_wrapper

        # CRITICAL: Also patch models.py's local reference
        from hyvideo.modules import models
        models.attention = cluster_sparse_attention_wrapper

        _sparse_enabled = True

        logger.info("✓ Cluster-based sparse attention enabled:")
        logger.info(f"  Q centroids: {num_q_centroids}")
        logger.info(f"  K centroids: {num_k_centroids}")
        logger.info(f"  Top-p: {top_p_kmeans}")
        logger.info(f"  Min K ratio: {min_kc_ratio}")
        logger.info(f"  K-means init iters: {kmeans_iter_init}")
        logger.info(f"  K-means step iters: {kmeans_iter_step}")

        # Verify patching
        logger.info(f"  Attention function: {attenion.attention.__name__}")
        logger.info(f"  Models attention: {models.attention.__name__}")

        # Patch diffusion pipeline to track step index (for warmup and dynamic threshold)
        if enable_warmup or enable_dynamic_threshold:
            try:
                # We need to track which inference step we're on (0, 1, 2, ...)
                # This is different from timestep value (1000, 950, 900, ...)
                # We'll patch the scheduler's step function to increment step_index

                from hyvideo.diffusion.schedulers import FlowMatchDiscreteScheduler

                if hasattr(FlowMatchDiscreteScheduler, 'step'):
                    _original_step = FlowMatchDiscreteScheduler.step

                    def step_with_index_increment(self, *args, **kwargs):
                        global _current_step_index, _cluster_sparse_attn
                        result = _original_step(self, *args, **kwargs)
                        if _current_step_index is not None:
                            _current_step_index += 1
                            # Update step index in attention module for dynamic threshold
                            if _cluster_sparse_attn is not None and hasattr(_cluster_sparse_attn, 'set_current_step'):
                                _cluster_sparse_attn.set_current_step(_current_step_index)
                        return result

                    FlowMatchDiscreteScheduler.step = step_with_index_increment
                    logger.info(f"  Step tracking: ENABLED (patched FlowMatchDiscreteScheduler.step)")

            except Exception as e:
                logger.warning(f"Could not patch step tracking: {e}")
                logger.warning(f"  Step-based warmup may not work correctly")

        # Log warmup settings
        if enable_warmup:
            logger.info(f"  Warmup: ENABLED (Sparse-VideoGen style)")
            logger.info(f"    - Step warmup: first {_warmup_config['num_fp_steps']} steps use DENSE ({first_times_fp_ratio*100:.1f}%)")
            logger.info(f"    - Layer warmup: first {_warmup_config['num_fp_layers']} layers use DENSE ({first_layers_fp_ratio*100:.1f}%)")
            if zero_step_kmeans_init:
                logger.info(f"    - Zero-step K-means initialization: ON")
        else:
            logger.info(f"  Warmup: Disabled (always sparse)")

    except Exception as e:
        logger.error(f"Failed to patch attention: {e}")
        raise


def unpatch_attention():
    """Restore original attention function."""
    global _original_attention_func, _sparse_enabled, _sparse_stats

    if _original_attention_func is not None:
        try:
            from hyvideo.modules import attenion, models
            attenion.attention = _original_attention_func
            models.attention = _original_attention_func
            _sparse_enabled = False

            avg_density = _sparse_stats['avg_density']
            logger.info("✓ Cluster-based sparse attention disabled")
            logger.info(f"  Total calls: {_sparse_stats['total_calls']}")
            logger.info(f"  Sparse calls: {_sparse_stats['sparse_calls']}")
            logger.info(f"  Average density: {avg_density:.2%}")
            logger.info(f"  Average sparsity: {(1.0 - avg_density):.2%}")

        except Exception as e:
            logger.warning(f"Could not unpatch attention: {e}")


def main():
    # Parse cluster-specific arguments first
    import sys
    cluster_parser = argparse.ArgumentParser(add_help=False)
    cluster_parser.add_argument('--num-q-centroids', type=int, default=64, help='Number of query clusters')
    cluster_parser.add_argument('--num-k-centroids', type=int, default=64, help='Number of key clusters')
    cluster_parser.add_argument('--top-p-kmeans', type=float, default=0.9, help='Top-p for cluster selection')
    cluster_parser.add_argument('--min-kc-ratio', type=float, default=0.1, help='Minimum key cluster ratio')
    cluster_parser.add_argument('--kmeans-iter-init', type=int, default=10, help='K-means iterations for init')
    cluster_parser.add_argument('--kmeans-iter-step', type=int, default=3, help='K-means iterations per step')
    cluster_parser.add_argument('--enable-warmup', action='store_true', help='Enable Sparse-VideoGen style warmup')
    cluster_parser.add_argument('--first-times-fp-ratio', type=float, default=0.075, help='Ratio of early steps using dense (default: 0.075 = 7.5%)')
    cluster_parser.add_argument('--first-layers-fp-ratio', type=float, default=0.025, help='Ratio of early layers using dense (default: 0.025 = 2.5%)')
    cluster_parser.add_argument('--zero-step-kmeans-init', action='store_true', help='Initialize K-means during warmup phase')
    cluster_parser.add_argument('--enable-kv-cache', action='store_true', help='Enable both K and V caching (legacy, same as --enable-k-cache --enable-v-cache)')
    cluster_parser.add_argument('--enable-k-cache', action='store_true', help='Enable K tensor caching only')
    cluster_parser.add_argument('--enable-v-cache', action='store_true', help='Enable V tensor caching only')
    cluster_parser.add_argument('--cache-update-threshold', type=float, default=0.05, help='Dynamic map change threshold for cache updates (default: 0.05 = 5%%)')
    cluster_parser.add_argument('--enable-dynamic-threshold', action='store_true', help='Enable adaptive threshold that increases with layer and step (recommended)')
    cluster_parser.add_argument('--base-threshold', type=float, default=0.03, help='Base threshold for early layers/steps (default: 0.03 = 3%%)')
    cluster_parser.add_argument('--max-threshold', type=float, default=0.15, help='Max threshold for late layers/steps (default: 0.15 = 15%%)')
    cluster_parser.add_argument('--enable-first-frame-sink', action='store_true', help='Enable first-frame sink: first frame uses dense attention and skips KV cache (recommended)')

    cluster_args, remaining_argv = cluster_parser.parse_known_args()

    # Filter out empty strings from remaining args
    remaining_argv = [arg for arg in remaining_argv if arg.strip() != '']

    # Temporarily set sys.argv to remaining args for parse_args
    original_argv = sys.argv
    sys.argv = [sys.argv[0]] + remaining_argv

    # Parse HunyuanVideo arguments
    args = parse_args()

    # Restore original argv
    sys.argv = original_argv

    # Add cluster arguments to args
    args.num_q_centroids = cluster_args.num_q_centroids
    args.num_k_centroids = cluster_args.num_k_centroids
    args.top_p_kmeans = cluster_args.top_p_kmeans
    args.min_kc_ratio = cluster_args.min_kc_ratio
    args.kmeans_iter_init = cluster_args.kmeans_iter_init
    args.kmeans_iter_step = cluster_args.kmeans_iter_step
    args.enable_warmup = cluster_args.enable_warmup
    args.first_times_fp_ratio = cluster_args.first_times_fp_ratio
    args.first_layers_fp_ratio = cluster_args.first_layers_fp_ratio
    args.zero_step_kmeans_init = cluster_args.zero_step_kmeans_init
    args.enable_kv_cache = cluster_args.enable_kv_cache
    args.enable_k_cache = cluster_args.enable_k_cache
    args.enable_v_cache = cluster_args.enable_v_cache
    args.cache_update_threshold = cluster_args.cache_update_threshold
    args.enable_dynamic_threshold = cluster_args.enable_dynamic_threshold
    args.base_threshold = cluster_args.base_threshold
    args.max_threshold = cluster_args.max_threshold
    args.enable_first_frame_sink = cluster_args.enable_first_frame_sink

    logger.info(f"Arguments: {args}")

    # Setup paths
    models_root_path = Path(args.model_base)
    if not models_root_path.exists():
        raise ValueError(f"Model path not found: {models_root_path}")

    save_path = args.save_path if args.save_path_suffix == "" else f'{args.save_path}_{args.save_path_suffix}'
    os.makedirs(save_path, exist_ok=True)

    # Load models FIRST
    logger.info("Loading HunyuanVideo models...")
    hunyuan_video_sampler = HunyuanVideoSampler.from_pretrained(models_root_path, args=args)
    args = hunyuan_video_sampler.args

    # THEN enable cluster-based sparse attention (after model loading)
    patch_attention_for_cluster_sparse(
        num_q_centroids=args.num_q_centroids,
        num_k_centroids=args.num_k_centroids,
        top_p_kmeans=args.top_p_kmeans,
        min_kc_ratio=args.min_kc_ratio,
        kmeans_iter_init=args.kmeans_iter_init,
        kmeans_iter_step=args.kmeans_iter_step,
        video_length=args.video_length,
        video_size=args.video_size,
        vae=args.vae,
        enable_warmup=args.enable_warmup,
        first_times_fp_ratio=args.first_times_fp_ratio,
        first_layers_fp_ratio=args.first_layers_fp_ratio,
        zero_step_kmeans_init=args.zero_step_kmeans_init,
        total_inference_steps=args.infer_steps,
        enable_kv_cache=args.enable_kv_cache,
        enable_k_cache=args.enable_k_cache,
        enable_v_cache=args.enable_v_cache,
        cache_update_threshold=args.cache_update_threshold,
        enable_dynamic_threshold=args.enable_dynamic_threshold,
        base_threshold=args.base_threshold,
        max_threshold=args.max_threshold,
        enable_first_frame_sink=args.enable_first_frame_sink,
    )

    # Generate video
    logger.info(f"Generating video with cluster-based sparse attention...")
    start_time = time.time()

    outputs = hunyuan_video_sampler.predict(
        prompt=args.prompt,
        height=args.video_size[0],
        width=args.video_size[1],
        video_length=args.video_length,
        seed=args.seed,
        negative_prompt=args.neg_prompt,
        infer_steps=args.infer_steps,
        guidance_scale=args.cfg_scale,
        num_videos_per_prompt=args.num_videos,
        flow_shift=args.flow_shift,
        batch_size=args.batch_size,
        embedded_guidance_scale=args.embedded_cfg_scale,
    )

    generation_time = time.time() - start_time

    # Disable sparse attention
    unpatch_attention()

    # Save video
    samples = outputs['samples']
    if 'LOCAL_RANK' not in os.environ or int(os.environ['LOCAL_RANK']) == 0:
        for i, sample in enumerate(samples):
            sample = samples[i].unsqueeze(0)
            time_flag = datetime.fromtimestamp(time.time()).strftime("%Y-%m-%d-%H:%M:%S")
            filename = f"{time_flag}_CLUSTER_seed{outputs['seeds'][i]}.mp4"
            save_path_full = os.path.join(save_path, filename)
            save_videos_grid(sample, save_path_full, fps=24)

            logger.info(f"✓ Video saved: {save_path_full}")
            logger.info(f"  Generation time: {generation_time:.2f}s")
            logger.info(f"  Cluster config: Q={args.num_q_centroids}, K={args.num_k_centroids}, top_p={args.top_p_kmeans}")

            stats = _cluster_sparse_attn.get_stats()
            logger.info(f"  Final stats:")
            logger.info(f"    - Density: {stats['avg_density']:.2%}")
            logger.info(f"    - Sparsity: {stats['avg_sparsity']:.2%}")
            logger.info(f"    - K-means iters: {stats['kmeans_iters']:.1f}")

            # Print cache statistics if enabled
            if stats.get('k_cache_enabled') or stats.get('v_cache_enabled'):
                logger.info(f"  Cache statistics:")
                if stats.get('k_cache_enabled'):
                    logger.info(f"    - K cache hit rate: {stats['k_cache_hit_rate']:.2%}")
                    logger.info(f"    - K cache hits: {stats['k_cache_hits']}")
                    logger.info(f"    - K cache misses: {stats['k_cache_misses']}")
                if stats.get('v_cache_enabled'):
                    logger.info(f"    - V cache hit rate: {stats['v_cache_hit_rate']:.2%}")
                    logger.info(f"    - V cache hits: {stats['v_cache_hits']}")
                    logger.info(f"    - V cache misses: {stats['v_cache_misses']}")


if __name__ == "__main__":
    main()
