import os
from typing import Optional

import torch

from ...logger import logger
from ...layer_offload import (
    OffloadConfig,
    OffloadedTransformerBlocks,
    enable_layer_offloading,
    get_gpu_memory_info,
)
from ...sadsa_informed_offload import (
    SIAOConfig,
    SADSAInformedOffloadManager,
    create_siao_manager,
)
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


def setup_sadsa_with_offloading(
    pipe,
    height: int,
    width: int,
    num_frames: int,
    prompt_length: int,
    first_layers_fp: int = 0,
    first_times_fp: float = 1001,
    # SADSA specific
    structure_p_full: float = 0.50,
    semantic_p_full: float = 0.70,
    detail_p_full: float = 0.85,
    motion_threshold_high: float = 0.6,
    motion_threshold_low: float = 0.15,
    # Clustering params
    num_q_centroids: int = 50,
    num_k_centroids: int = 200,
    min_kc_ratio: float = 0,
    kmeans_iter_init: int = 50,
    kmeans_iter_step: int = 2,
    # Offloading params
    enable_offload: bool = True,
    layers_on_gpu: int = 1,
    use_pinned_memory: bool = True,
    async_prefetch: bool = True,
    # Logging
    logging_file: Optional[str] = None,
    verbose: bool = False,
) -> Optional[OffloadedTransformerBlocks]:
    """
    Set up SADSA (Semantic-Aware Dynamic Sparse Attention) with layer offloading.

    This is the recommended configuration for running HunyuanVideo on 24GB GPUs.
    Combines:
    1. SADSA: Stage-adaptive sparse attention for quality + efficiency
    2. Layer offloading: Sequential layer execution to fit in GPU memory

    Memory Analysis (HunyuanVideo 720p, 129 frames):
    - Without offloading: ~26GB GPU memory (model) + ~4GB (KV cache) = OOM on 24GB
    - With SADSA only: ~26GB GPU memory (model) + ~1-2GB (sparse KV) = Still OOM
    - With SADSA + Offloading: ~4GB (1-2 layers) + ~1-2GB (sparse KV) = ~6-8GB total

    Args:
        pipe: HunyuanVideoPipeline instance
        height: Video height
        width: Video width
        num_frames: Number of frames
        prompt_length: Length of text prompt tokens
        first_layers_fp: Number of first layers to always use full precision attention
        first_times_fp: Timestep threshold for full precision warmup
        structure_p_full: p_full for structure stage (t > 0.7)
        semantic_p_full: p_full for semantic stage (0.3 < t < 0.7)
        detail_p_full: p_full for detail stage (t < 0.3)
        motion_threshold_high: Motion threshold for forcing full attention
        motion_threshold_low: Motion threshold for allowing skip
        num_q_centroids: Number of query centroids for clustering
        num_k_centroids: Number of key centroids for clustering
        min_kc_ratio: Minimum ratio of key centroids to keep
        kmeans_iter_init: K-means iterations for initialization
        kmeans_iter_step: K-means iterations per step
        enable_offload: Whether to enable layer offloading
        layers_on_gpu: Number of layers to keep on GPU (1=min memory, 2=better latency)
        use_pinned_memory: Use pinned memory for fast transfers
        async_prefetch: Asynchronously prefetch next layer
        logging_file: Path to log file
        verbose: Enable verbose logging

    Returns:
        OffloadedTransformerBlocks instance if offloading enabled, None otherwise

    Example:
        pipe = HunyuanVideoPipeline.from_pretrained(...)

        # Set up SADSA + offloading for 24GB GPU
        offloaded = setup_sadsa_with_offloading(
            pipe, height=720, width=1280, num_frames=129,
            prompt_length=256, enable_offload=True
        )

        # Generate video
        output = pipe(prompt="...", ...)

        # Cleanup
        if offloaded:
            offloaded.cleanup()
    """
    logger.info("=" * 70)
    logger.info("[SADSA+Offload] Setting up Semantic-Aware Dynamic Sparse Attention")
    logger.info("[SADSA+Offload] with Layer Offloading for 24GB GPU inference")
    logger.info("=" * 70)

    # Calculate video dimensions
    context_length = 256
    num_frame = 1 + num_frames // 4
    frame_size = height * width // 256

    logger.info(f"[SADSA+Offload] Video config: {height}x{width}, {num_frames} frames")
    logger.info(f"[SADSA+Offload] Latent config: context={context_length}, frames={num_frame}, frame_size={frame_size}")

    # Make dir and clear the logging file
    if logging_file is not None:
        os.makedirs(os.path.dirname(logging_file), exist_ok=True)
        with open(logging_file, "w") as f:
            f.write("")

    # Step 1: Replace with FlashAttention first
    logger.info("[SADSA+Offload] Step 1/3: Installing FlashAttention processors...")
    replace_hyvideo_flashattention(pipe)

    # Step 2: Set up SADSA attention
    logger.info("[SADSA+Offload] Step 2/3: Setting up SADSA attention...")
    setup_sadsa_attention(
        pipe=pipe,
        structure_p_full=structure_p_full,
        semantic_p_full=semantic_p_full,
        detail_p_full=detail_p_full,
        motion_threshold_high=motion_threshold_high,
        motion_threshold_low=motion_threshold_low,
        num_q_centroids=num_q_centroids,
        num_k_centroids=num_k_centroids,
        kmeans_iter_init=kmeans_iter_init,
        kmeans_iter_step=kmeans_iter_step,
        min_kc_ratio=min_kc_ratio,
        first_layers_fp=first_layers_fp,
        first_times_fp=first_times_fp,
        num_frame=num_frame,
        frame_size=frame_size,
        context_length=context_length,
        prompt_length=prompt_length,
        max_timestep=1000,
        logging_file=logging_file,
        verbose=verbose,
    )

    # Step 3: Replace sparse forward (for custom transformer blocks)
    replace_sparse_forward()

    # Step 4: Enable layer offloading if requested
    offload_manager = None
    if enable_offload:
        logger.info("[SADSA+Offload] Step 3/3: Enabling SIAO (SADSA-Informed Adaptive Offloading)...")
        logger.info("[SADSA+Offload] Novel features enabled:")
        logger.info("[SADSA+Offload]   - SAMBA: Stage-Aware Memory Budget Allocation")
        logger.info("[SADSA+Offload]   - MPP: Motion-Predictive Prefetching")
        logger.info("[SADSA+Offload]   - QGE: Quality-Gradient Eviction")
        logger.info("[SADSA+Offload]   - ADCP: Attention-Density Compute Prediction")

        # Create SIAO manager with stage-aware budgets
        offload_manager = create_siao_manager(
            transformer=pipe.transformer,
            budget_structure=1,   # Aggressive in structure stage
            budget_semantic=2,    # Balanced in semantic stage
            budget_detail=layers_on_gpu + 1,  # Conservative in detail stage
            use_async_prefetch=async_prefetch,
            use_pinned_memory=use_pinned_memory,
            verbose=verbose,
        )

        # Store reference on transformer
        pipe.transformer._siao_manager = offload_manager
    else:
        logger.info("[SADSA+Offload] Step 3/3: Layer offloading disabled (full GPU mode)")

    # Report memory status
    mem = get_gpu_memory_info()
    logger.info(f"[SADSA+Offload] Setup complete!")
    logger.info(f"[SADSA+Offload] GPU memory: {mem['allocated']:.2f}GB allocated, {mem['free']:.2f}GB free")
    logger.info("=" * 70)

    return offload_manager


def estimate_memory_requirements(
    height: int = 720,
    width: int = 1280,
    num_frames: int = 129,
    enable_offload: bool = True,
    enable_sadsa: bool = True,
) -> dict:
    """
    Estimate GPU memory requirements for video generation.

    Returns a dict with memory estimates in GB.
    """
    # HunyuanVideo model parameters
    model_params_gb = 26.0  # ~13B params in bf16

    # Per-layer memory (60 total layers)
    per_layer_gb = model_params_gb / 60  # ~0.43 GB per layer

    # Latent dimensions
    num_frame = 1 + num_frames // 4
    frame_size = height * width // 256
    seq_len = num_frame * frame_size + 256  # + context length

    # Attention memory (Q, K, V per layer)
    # [batch, heads, seq_len, head_dim] * 3 * bf16
    batch_size = 1
    num_heads = 24
    head_dim = 128
    attn_per_layer_gb = (batch_size * num_heads * seq_len * head_dim * 3 * 2) / 1e9

    # With SADSA, attention is sparse (~20-50% density on average)
    sadsa_reduction = 0.35 if enable_sadsa else 1.0

    # Compute total
    if enable_offload:
        # Only 1-2 layers on GPU at a time
        model_memory = per_layer_gb * 2  # 2 layers for prefetching
    else:
        model_memory = model_params_gb

    attention_memory = attn_per_layer_gb * sadsa_reduction

    # Activations and misc
    activation_memory = 2.0 if enable_offload else 4.0

    total = model_memory + attention_memory + activation_memory

    return {
        'model_memory_gb': model_memory,
        'attention_memory_gb': attention_memory,
        'activation_memory_gb': activation_memory,
        'total_estimated_gb': total,
        'offload_enabled': enable_offload,
        'sadsa_enabled': enable_sadsa,
        'fits_24gb': total < 22,  # Leave 2GB headroom
    }
