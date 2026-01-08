import os

import torch

from ...logger import logger
from .attention import (
    Hunyuan_SAPAttn_Processor2_0,
    Hunyuan_SAPAttn_CTCA_Processor2_0,
    Hunyuan_SVGAttn_Processor2_0,
    HunyuanVideoAttnProcessor2_0_FlashAttention,
    prepare_flexattention,
)
from .custom_models import replace_sparse_forward
from .utils import get_attention_mask, sparsity_to_width


def replace_hyvideo_flashattention(pipe):
    """
    Replace the FSDP + masked attention with flash attention + varlen. Crucial for inference efficiency.
    """
    for layer_idx, m in enumerate(pipe.transformer.transformer_blocks):
        self_attn = m.attn
        self_attn.processor = HunyuanVideoAttnProcessor2_0_FlashAttention(layer_idx=layer_idx)
        print(f"Replaced FlashAttention implementation in double stream transformer block {layer_idx}")

    for layer_idx, m in enumerate(pipe.transformer.single_transformer_blocks):
        self_attn = m.attn
        self_attn.processor = HunyuanVideoAttnProcessor2_0_FlashAttention(
            layer_idx=layer_idx + len(pipe.transformer.transformer_blocks)
        )
        print(f"Replaced FlashAttention implementation in single stream transformer block {layer_idx}")


def replace_hyvideo_attention(
    pipe,
    height,
    width,
    num_frames,
    prompt_length,
    first_layers_fp,
    first_times_fp,
    pattern="SVG",  # Default to SVG for backward compatibility
    # SVG specific, but provide defaults for general call signature
    num_sampled_rows=64,
    sample_mse_max_row=10000,
    sparsity=0.25,
    # Pattern dispatcher and KMEANS_BLOCK specific args
    num_q_centroids=None,
    num_k_centroids=None,
    top_p_kmeans=None,
    min_kc_ratio=0,
    logging_file=None,
    kmeans_iter_init=0,
    kmeans_iter_step=0,
    zero_step_kmeans_init=False,
    # CTCA (Cross-Timestep Cluster Amortization) specific args
    ctca_quality_threshold=0.80,
    ctca_adaptive=True,
    ctca_min_interval=2,
    ctca_max_interval=10,
    ctca_verbose=False,
    # CKGR (Cluster-Guided KV Reuse) specific args
    ckgr_enabled=False,
    ckgr_stability_threshold=0.7,
    ckgr_min_reuse_steps=2,
    ckgr_min_stable_steps=2,
    ckgr_centroid_sim_threshold=0.98,
    ckgr_quality_threshold=0.75,
    ckgr_verbose=False,
    ckgr_reuse_k=False,
    ckgr_reuse_v=True,
    ckgr_min_head_reuse_ratio=1.0,
):

    cfg_size, num_head, head_dim, dtype, device = 1, 24, 128, torch.bfloat16, "cuda"
    context_length, num_frame = 256, 1 + num_frames // 4  # TODO: Make it more formal
    frame_size = height * width // 256  # TODO: Make it more formal

    if pattern == "SVG":
        masks = ["spatial", "temporal"]

        # Calculation
        spatial_width = temporal_width = sparsity_to_width(sparsity, context_length, num_frame, frame_size)

        print(f"Spatial_width: {spatial_width}, Temporal_width: {temporal_width}. Sparsity: {sparsity}")

        AttnModule = Hunyuan_SVGAttn_Processor2_0

        # These might be needed by the processor if it has to adapt to sequence dimensions
        AttnModule.prompt_length = prompt_length
        AttnModule.context_length = context_length
        AttnModule.num_frame = num_frame
        AttnModule.frame_size = frame_size

        AttnModule.num_sampled_rows = num_sampled_rows
        AttnModule.sample_mse_max_row = sample_mse_max_row
        AttnModule.attention_masks = [
            get_attention_mask(mask_name, sample_mse_max_row, context_length, num_frame, frame_size)
            for mask_name in masks
        ]
        AttnModule.first_layers_fp = first_layers_fp
        AttnModule.first_times_fp = first_times_fp

        block_mask = prepare_flexattention(
            cfg_size,
            num_head,
            head_dim,
            dtype,
            device,
            context_length,
            prompt_length,
            num_frame,
            frame_size,
            diag_width=spatial_width,
            multiplier=temporal_width,
        )
        AttnModule.block_mask = block_mask
        replace_sparse_forward()

        logger.info("Flexattn block_mask prepared.")
        logger.info(block_mask)

        for layer_idx, m in enumerate(pipe.transformer.transformer_blocks):
            self_attn = m.attn
            self_attn.processor = AttnModule(layer_idx=layer_idx)
            print(f"Replaced Sparse VideoGen block for Double Stream Transformer at layer {layer_idx}")

        for layer_idx, m in enumerate(pipe.transformer.single_transformer_blocks):
            self_attn = m.attn
            self_attn.processor = AttnModule(layer_idx=layer_idx + len(pipe.transformer.transformer_blocks))
            print(
                f"Replaced Sparse VideoGen block for Single Stream Transformer at layer {layer_idx + len(pipe.transformer.transformer_blocks)}"
            )

    elif pattern in ["SAP"]:

        # Pass K-means specific parameters to the processor's constructor or set them as attributes
        # The processor itself will handle the K-means logic internally
        logger.info(
            f"Configuring KMEANS_BLOCK attention with QC: {num_q_centroids}, KC: {num_k_centroids}, P: {top_p_kmeans}, min_kc_ratio: {min_kc_ratio}"
        )

        # Make dir and clear the logging file
        if logging_file is not None:
            os.makedirs(os.path.dirname(logging_file), exist_ok=True)
            with open(logging_file, "w") as f:
                f.write("")

        AttnModule = Hunyuan_SAPAttn_Processor2_0

        AttnModule.first_layers_fp = first_layers_fp
        AttnModule.first_times_fp = first_times_fp
        AttnModule.logging_file = logging_file

        # These might be needed by the processor if it has to adapt to sequence dimensions
        AttnModule.prompt_length = prompt_length
        AttnModule.context_length = context_length
        AttnModule.num_frame = num_frame
        AttnModule.frame_size = frame_size

        AttnModule.num_q_centroids = num_q_centroids
        AttnModule.num_k_centroids = num_k_centroids
        AttnModule.top_p_kmeans = top_p_kmeans
        AttnModule.min_kc_ratio = min_kc_ratio
        AttnModule.kmeans_iter_init = kmeans_iter_init
        AttnModule.kmeans_iter_step = kmeans_iter_step
        AttnModule.zero_step_kmeans_init = zero_step_kmeans_init

        replace_sparse_forward()

        for layer_idx, m in enumerate(pipe.transformer.transformer_blocks):
            self_attn = m.attn
            self_attn.processor = AttnModule(layer_idx=layer_idx)
            print(f"Replaced Semantic Aware Permutation block for Double Stream Transformer at layer {layer_idx}")

        for layer_idx, m in enumerate(pipe.transformer.single_transformer_blocks):
            self_attn = m.attn
            self_attn.processor = AttnModule(layer_idx=layer_idx + len(pipe.transformer.transformer_blocks))
            print(
                f"Replaced Semantic Aware Permutation block for Single Stream Transformer at layer {layer_idx + len(pipe.transformer.transformer_blocks)}"
            )

    elif pattern == "SAP_CTCA":
        # SAP with Cross-Timestep Cluster Amortization (CTCA)
        # This variant reduces K-means overhead by reusing cluster assignments across timesteps

        logger.info(
            f"Configuring SAP with CTCA: QC={num_q_centroids}, KC={num_k_centroids}, "
            f"P={top_p_kmeans}, min_kc_ratio={min_kc_ratio}"
        )
        logger.info(
            f"CTCA config: quality_threshold={ctca_quality_threshold}, adaptive={ctca_adaptive}, "
            f"min_interval={ctca_min_interval}, max_interval={ctca_max_interval}"
        )

        # Make dir and clear the logging file
        if logging_file is not None:
            os.makedirs(os.path.dirname(logging_file), exist_ok=True)
            with open(logging_file, "w") as f:
                f.write("")

        AttnModule = Hunyuan_SAPAttn_CTCA_Processor2_0

        # Standard SAP configuration
        AttnModule.first_layers_fp = first_layers_fp
        AttnModule.first_times_fp = first_times_fp
        AttnModule.logging_file = logging_file

        AttnModule.prompt_length = prompt_length
        AttnModule.context_length = context_length
        AttnModule.num_frame = num_frame
        AttnModule.frame_size = frame_size

        AttnModule.num_q_centroids = num_q_centroids
        AttnModule.num_k_centroids = num_k_centroids
        AttnModule.top_p_kmeans = top_p_kmeans
        AttnModule.min_kc_ratio = min_kc_ratio
        AttnModule.kmeans_iter_init = kmeans_iter_init
        AttnModule.kmeans_iter_step = kmeans_iter_step
        AttnModule.zero_step_kmeans_init = zero_step_kmeans_init

        # CTCA-specific configuration
        AttnModule.ctca_quality_threshold = ctca_quality_threshold
        AttnModule.ctca_adaptive = ctca_adaptive
        AttnModule.ctca_min_interval = ctca_min_interval
        AttnModule.ctca_max_interval = ctca_max_interval
        AttnModule.ctca_verbose = ctca_verbose

        # CKGR-specific configuration
        AttnModule.ckgr_enabled = ckgr_enabled
        AttnModule.ckgr_stability_threshold = ckgr_stability_threshold
        AttnModule.ckgr_min_reuse_steps = ckgr_min_reuse_steps
        AttnModule.ckgr_min_stable_steps = ckgr_min_stable_steps
        AttnModule.ckgr_centroid_sim_threshold = ckgr_centroid_sim_threshold
        AttnModule.ckgr_quality_threshold = ckgr_quality_threshold
        AttnModule.ckgr_verbose = ckgr_verbose
        AttnModule.ckgr_reuse_k = ckgr_reuse_k
        AttnModule.ckgr_reuse_v = ckgr_reuse_v
        AttnModule.ckgr_min_head_reuse_ratio = ckgr_min_head_reuse_ratio

        # Initialize CTCA manager (shared across all layers)
        AttnModule.initialize_ctca()

        # Initialize CKGR manager if enabled
        if ckgr_enabled:
            AttnModule.initialize_ckgr()

        replace_sparse_forward()

        for layer_idx, m in enumerate(pipe.transformer.transformer_blocks):
            self_attn = m.attn
            self_attn.processor = AttnModule(layer_idx=layer_idx)
            print(f"Replaced SAP+CTCA block for Double Stream Transformer at layer {layer_idx}")

        for layer_idx, m in enumerate(pipe.transformer.single_transformer_blocks):
            self_attn = m.attn
            self_attn.processor = AttnModule(layer_idx=layer_idx + len(pipe.transformer.transformer_blocks))
            print(
                f"Replaced SAP+CTCA block for Single Stream Transformer at layer {layer_idx + len(pipe.transformer.transformer_blocks)}"
            )

    else:
        assert pattern == "dense", f"Invalid pattern: {pattern}"


def get_ctca_processor():
    """
    Get the CTCA processor class for external access to statistics.

    Usage:
        processor = get_ctca_processor()
        processor.print_ctca_statistics()
    """
    return Hunyuan_SAPAttn_CTCA_Processor2_0


def reset_ctca():
    """Reset CTCA state for a new video generation."""
    Hunyuan_SAPAttn_CTCA_Processor2_0.reset_ctca()


def print_ctca_statistics():
    """Print CTCA performance statistics."""
    Hunyuan_SAPAttn_CTCA_Processor2_0.print_ctca_statistics()


def print_ctaa_statistics():
    """Print CTAA (Cross-Timestep Attention Amortization) statistics."""
    from ...kmeans_utils import print_ctaa_statistics as _print_ctaa_statistics
    _print_ctaa_statistics()


def reset_ctaa_statistics():
    """Reset CTAA statistics for a new video generation."""
    from ...kmeans_utils import reset_ctaa_statistics as _reset_ctaa_statistics
    _reset_ctaa_statistics()


# CKGR (Cluster-Guided KV Reuse) helper functions
def reset_ckgr():
    """Reset CKGR state for a new video generation."""
    Hunyuan_SAPAttn_CTCA_Processor2_0.reset_ckgr()


def print_ckgr_statistics():
    """Print CKGR performance statistics."""
    Hunyuan_SAPAttn_CTCA_Processor2_0.print_ckgr_statistics()
