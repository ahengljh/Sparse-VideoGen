import os

import torch

from ...logger import logger
from .attention import (
    Hunyuan_SAPAttn_Processor2_0,
    Hunyuan_SADSAAttn_Processor2_0,
    Hunyuan_SVGAttn_Processor2_0,
    HunyuanVideoAttnProcessor2_0_FlashAttention,
    prepare_flexattention,
    setup_sadsa_attention,
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
    # SADSA specific args
    structure_p_full=0.50,
    semantic_p_full=0.70,
    detail_p_full=0.85,
    motion_threshold_high=0.6,
    motion_threshold_low=0.15,
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

    elif pattern == "SADSA":
        # SADSA: Semantic-Aware Dynamic Sparse Attention
        # Stage-adaptive thresholds + motion-aware routing + quality preservation
        logger.info("=" * 60)
        logger.info("[SADSA] Setting up Semantic-Aware Dynamic Sparse Attention")
        logger.info("=" * 60)

        # Make dir and clear the logging file
        if logging_file is not None:
            os.makedirs(os.path.dirname(logging_file), exist_ok=True)
            with open(logging_file, "w") as f:
                f.write("")

        # Use the setup function which handles all configuration
        setup_sadsa_attention(
            pipe=pipe,
            structure_p_full=structure_p_full,
            semantic_p_full=semantic_p_full,
            detail_p_full=detail_p_full,
            motion_threshold_high=motion_threshold_high,
            motion_threshold_low=motion_threshold_low,
            num_q_centroids=num_q_centroids or 400,
            num_k_centroids=num_k_centroids or 1000,
            kmeans_iter_init=kmeans_iter_init or 50,
            kmeans_iter_step=kmeans_iter_step or 2,
            min_kc_ratio=min_kc_ratio,
            first_layers_fp=first_layers_fp,
            first_times_fp=first_times_fp,
            num_frame=num_frame,
            frame_size=frame_size,
            context_length=context_length,
            prompt_length=prompt_length,
            max_timestep=1000,
            logging_file=logging_file,
            verbose=True,
        )

        replace_sparse_forward()

    else:
        assert pattern == "dense", f"Invalid pattern: {pattern}. Valid patterns: SVG, SAP, SADSA, dense"
