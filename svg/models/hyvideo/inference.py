import os
from typing import Optional, Dict, List, Tuple

import torch

from ...logger import logger
from .attention import (
    KVReuseConfig,
    VideoKReuseConfig,
    Hunyuan_SAPAttn_Processor2_0,
    Hunyuan_SVGAttn_Processor2_0,
    HunyuanVideoAttnProcessor2_0_FlashAttention,
    prepare_flexattention,
)
from .custom_models import replace_sparse_forward
from .utils import get_attention_mask, sparsity_to_width


def replace_hyvideo_flashattention(
    pipe,
    kv_reuse_config: Optional[KVReuseConfig] = None,
    video_k_reuse_config: Optional[VideoKReuseConfig] = None,
):
    """
    Replace the FSDP + masked attention with flash attention + varlen. Crucial for inference efficiency.
    """
    if kv_reuse_config is not None:
        HunyuanVideoAttnProcessor2_0_FlashAttention.kv_reuse_cfg = kv_reuse_config
    if video_k_reuse_config is not None:
        HunyuanVideoAttnProcessor2_0_FlashAttention.video_k_reuse_cfg = video_k_reuse_config

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
    kv_reuse_config: Optional[KVReuseConfig] = None,
    video_k_reuse_config: Optional[VideoKReuseConfig] = None,
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
):

    cfg_size, num_head, head_dim, dtype, device = 1, 24, 128, torch.bfloat16, "cuda"
    context_length, num_frame = 256, 1 + num_frames // 4  # TODO: Make it more formal
    frame_size = height * width // 256  # TODO: Make it more formal

    if pattern == "SVG":
        if kv_reuse_config is not None:
            Hunyuan_SVGAttn_Processor2_0.kv_reuse_cfg = kv_reuse_config
        if video_k_reuse_config is not None:
            Hunyuan_SVGAttn_Processor2_0.video_k_reuse_cfg = video_k_reuse_config

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
        if kv_reuse_config is not None:
            Hunyuan_SAPAttn_Processor2_0.kv_reuse_cfg = kv_reuse_config
        if video_k_reuse_config is not None:
            Hunyuan_SAPAttn_Processor2_0.video_k_reuse_cfg = video_k_reuse_config

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

    else:
        assert pattern == "dense", f"Invalid pattern: {pattern}"


def collect_kv_reuse_stats(pipe) -> Tuple[Dict[str, int], List[Dict[str, int]]]:
    totals = {"hits": 0, "misses": 0, "layers": 0}
    per_layer = []

    def _collect_from_block(block, layer_idx):
        processor = block.attn.processor
        if not hasattr(processor, "get_kv_reuse_stats"):
            return
        stats = processor.get_kv_reuse_stats()
        hits = int(stats.get("hits", 0))
        misses = int(stats.get("misses", 0))
        if hits == 0 and misses == 0:
            return
        totals["hits"] += hits
        totals["misses"] += misses
        totals["layers"] += 1
        per_layer.append({"layer": layer_idx, "hits": hits, "misses": misses})

    for layer_idx, block in enumerate(pipe.transformer.transformer_blocks):
        _collect_from_block(block, layer_idx)

    offset = len(pipe.transformer.transformer_blocks)
    for layer_idx, block in enumerate(pipe.transformer.single_transformer_blocks):
        _collect_from_block(block, layer_idx + offset)

    return totals, per_layer


def collect_video_k_reuse_stats(pipe) -> Tuple[Dict[str, int], List[Dict[str, int]]]:
    totals = {
        "hits": 0,
        "misses": 0,
        "cached_blocks": 0,
        "stable_blocks": 0,
        "critical_blocks": 0,
        "layers": 0,
    }
    per_layer = []

    def _collect_from_block(block, layer_idx):
        processor = block.attn.processor
        if not hasattr(processor, "get_video_k_reuse_stats"):
            return
        stats = processor.get_video_k_reuse_stats()
        hits = int(stats.get("hits", 0))
        misses = int(stats.get("misses", 0))
        cached_blocks = int(stats.get("cached_blocks", 0))
        stable_blocks = int(stats.get("stable_blocks", 0))
        critical_blocks = int(stats.get("critical_blocks", 0))
        if hits == 0 and misses == 0 and cached_blocks == 0 and stable_blocks == 0 and critical_blocks == 0:
            return
        totals["hits"] += hits
        totals["misses"] += misses
        totals["cached_blocks"] += cached_blocks
        totals["stable_blocks"] += stable_blocks
        totals["critical_blocks"] += critical_blocks
        totals["layers"] += 1
        per_layer.append(
            {
                "layer": layer_idx,
                "hits": hits,
                "misses": misses,
                "cached_blocks": cached_blocks,
                "stable_blocks": stable_blocks,
                "critical_blocks": critical_blocks,
            }
        )

    for layer_idx, block in enumerate(pipe.transformer.transformer_blocks):
        _collect_from_block(block, layer_idx)

    offset = len(pipe.transformer.transformer_blocks)
    for layer_idx, block in enumerate(pipe.transformer.single_transformer_blocks):
        _collect_from_block(block, layer_idx + offset)

    return totals, per_layer


def collect_video_k_reuse_metrics(pipe) -> List[Dict[str, object]]:
    metrics = []

    def _collect_from_block(block, layer_idx):
        processor = block.attn.processor
        if not hasattr(processor, "get_video_k_reuse_step_stats"):
            return
        for entry in processor.get_video_k_reuse_step_stats():
            record = dict(entry)
            record["layer"] = layer_idx
            metrics.append(record)

    for layer_idx, block in enumerate(pipe.transformer.transformer_blocks):
        _collect_from_block(block, layer_idx)

    offset = len(pipe.transformer.transformer_blocks)
    for layer_idx, block in enumerate(pipe.transformer.single_transformer_blocks):
        _collect_from_block(block, layer_idx + offset)

    return metrics
