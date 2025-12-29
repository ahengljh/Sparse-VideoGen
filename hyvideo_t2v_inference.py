import argparse
import json
import os
from glob import glob
import math
from copy import deepcopy

from termcolor import colored
import numpy as np
import torch
from diffusers import HunyuanVideoPipeline, HunyuanVideoTransformer3DModel, FlowMatchEulerDiscreteScheduler
from diffusers.utils import load_image, export_to_video

from dataloader import load_prompt_or_image
from svg.timer import print_operator_log_data
from svg.utils.seed import seed_everything
from svg.models.hyvideo.inference import (
    replace_hyvideo_flashattention,
    replace_hyvideo_attention,
    print_ctca_statistics,
    print_ctaa_statistics,
    reset_ctca,
    reset_ctaa_statistics,
)
from svg.models.hyvideo.utils import get_prompt_length
from svg.offload import pre_encode_and_offload
from svg.component_offload import enable_component_offloading, estimate_memory_savings

from svg.logger import logger

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate video from text prompt using Wan-Diffuser")
    parser.add_argument("--model_id", type=str, default="tencent/HunyuanVideo", help="Model ID to use for generation")
    parser.add_argument("--data_path", type=str, default=None, help="Path of VBench I2V data suite")
    parser.add_argument("--prompt", type=str, default=None, help="Text prompt for video generation")
    parser.add_argument("--negative_prompt", type=str, default=None, help="Negative text prompt to avoid certain features")

    parser.add_argument("--prompt_source", type=str, default="prompt", choices=["prompt", "T2V_Hyv_VBench", "T2V_Hyv_Web", "T2V_Xingyang_Motion", "T2V_Xingyang_VBench"], help="Source of the prompt")
    parser.add_argument("--prompt_idx", type=int, default=0, help="Index of the prompt")

    parser.add_argument("--height", type=int, default=720, help="Height of the generated video")
    parser.add_argument("--width", type=int, default=1280, help="Width of the generated video")
    parser.add_argument("--num_frames", type=int, default=129, help="Number of frames in the generated video")
    parser.add_argument("--num_inference_steps", type=int, default=50, help="Number of denoising steps in the generated video")
    parser.add_argument("--resolution", type=str, default="720p", choices=["480p", "720p"], help="Resolution of the generated video")
    parser.add_argument("--output_file", type=str, default="output.mp4", help="Output video file name")
    parser.add_argument("--logging_file", type=str, default=None, help="Path to the logging file.")
    parser.add_argument("--seed", type=int, default=0, help="Random seed for generation")
    parser.add_argument("--skip_existing", action="store_true", help="Skip generating existing output files")

    parser.add_argument("--pattern", type=str, default="dense", choices=["SVG", "dense", "SAP", "SAP_CTCA"])
    parser.add_argument("--first_layers_fp", type=float, default=0.025, help="Only works for best config. Leave the 0, 1, 2, 40, 41 layers in FP")
    parser.add_argument("--first_times_fp", type=float, default=0.075, help="Only works for best config. Leave the first 10% timestep in FP")

    # SVG1 specific
    parser.add_argument("--num_sampled_rows", type=int, default=64, help="The number of sampled rows")
    parser.add_argument("--sample_mse_max_row", type=int, default=10000, help="The maximum number of rows in attention mask. Prevent OOM.")
    parser.add_argument("--sparsity", type=float, default=0.25, help="The sparsity of the striped attention pattern. Accepts one or two float values.")

    # SVG2 (SAP) specific
    parser.add_argument("--num_q_centroids", "--qc", type=int, default=50, help="Number of query centroids for SAP.")
    parser.add_argument("--num_k_centroids", "--kc", type=int, default=200, help="Number of key centroids for SAP.")
    parser.add_argument("--top_p_kmeans", type=float, default=0.9, help="Top-p threshold for block selection in SAP.")
    parser.add_argument("--min_kc_ratio", type=float, default=0, help="At least this proportion of key blocks to keep per query block in SAP.")
    parser.add_argument("--kmeans_iter_init", type=int, default=0, help="Number of KMeans iterations for initialization in SAP.")
    parser.add_argument("--kmeans_iter_step", type=int, default=0, help="Number of KMeans iterations for other diffusion steps in SAP.")
    parser.add_argument("--zero_step_kmeans_init", action="store_true", help="Initialize the centroids for the first step in SAP, not after warmup.")

    # CTCA (Cross-Timestep Cluster Amortization) specific - only for SAP_CTCA pattern
    parser.add_argument("--ctca_quality_threshold", type=float, default=0.80, help="Quality threshold for triggering re-clustering (0-1). Lower = more aggressive reuse.")
    parser.add_argument("--ctca_adaptive", action="store_true", default=True, help="Use adaptive quality-based re-clustering.")
    parser.add_argument("--ctca_no_adaptive", action="store_false", dest="ctca_adaptive", help="Disable adaptive re-clustering, use fixed interval.")
    parser.add_argument("--ctca_min_interval", type=int, default=2, help="Minimum timesteps between re-clustering.")
    parser.add_argument("--ctca_max_interval", type=int, default=10, help="Maximum timesteps to reuse clusters before forced refresh.")
    parser.add_argument("--ctca_verbose", action="store_true", help="Enable verbose CTCA logging.")

    # Dynamic Offloading - enables running on smaller GPUs (e.g., 4090 24GB)
    parser.add_argument("--enable_offload", action="store_true", help="Enable dynamic layer offloading to run on smaller GPUs.")
    parser.add_argument("--offload_strategy", type=str, default="layer", choices=["layer", "component"],
                        help="Offloading strategy: 'layer' (full layer offload) or 'component' (pin attention, offload FFN).")
    parser.add_argument("--offload_num_layers", type=int, default=None, help="Number of transformer layers to keep on GPU (sliding window size). None=auto-detect based on GPU memory and resolution.")
    parser.add_argument("--offload_auto", action="store_true", help="Enable adaptive offloading that auto-calculates optimal layers based on GPU memory and video resolution.")
    parser.add_argument("--offload_max_memory_gb", type=float, default=None, help="Auto-tune num_layers based on memory budget (e.g., 20.0 for 24GB GPU).")
    parser.add_argument("--offload_pinned_memory", action="store_true", default=True, help="Use pinned CPU memory for faster transfers.")
    parser.add_argument("--offload_no_pinned_memory", action="store_false", dest="offload_pinned_memory", help="Disable pinned memory.")
    parser.add_argument("--offload_prefetch", action="store_true", default=True, help="Enable async prefetching of next layer.")
    parser.add_argument("--offload_no_prefetch", action="store_false", dest="offload_prefetch", help="Disable async prefetching.")
    parser.add_argument("--offload_verbose", action="store_true", help="Enable verbose offloading logging.")

    args = parser.parse_args()

    seed_everything(args.seed)

    # In some cases it will raise RuntimeError: cusolver error: CUSOLVER_STATUS_INTERNAL_ERROR
    torch.backends.cuda.preferred_linalg_library(backend="magma")
    
    if args.skip_existing:
        if os.path.exists(args.output_file):
            logger.info(f"Output file {args.output_file} already exists. Skipping generation.")
            exit(0)

    #########################################################
    # Load the model
    #########################################################
    import gc

    # Load transformer (loads to CPU by default from from_pretrained)
    if args.enable_offload:
        logger.info("Loading model for offload mode...")
    transformer = HunyuanVideoTransformer3DModel.from_pretrained(
        args.model_id, subfolder="transformer", torch_dtype=torch.bfloat16, revision='refs/pr/18'
    )

    flow_shift = 7.0
    scheduler = FlowMatchEulerDiscreteScheduler(shift=flow_shift)

    # Load pipeline
    pipe = HunyuanVideoPipeline.from_pretrained(
        args.model_id, transformer=transformer, scheduler=scheduler,
        revision='refs/pr/18', torch_dtype=torch.bfloat16
    )

    if args.enable_offload:
        # Immediately move everything to CPU to free GPU memory
        # This is crucial for 24GB GPUs - diffusers may load to GPU during from_pretrained
        logger.info("Moving all pipeline components to CPU to free GPU memory...")
        pipe.to('cpu')
        gc.collect()
        torch.cuda.empty_cache()
        logger.info(f"Pipeline on CPU. GPU memory: {torch.cuda.memory_allocated()/1024**3:.2f}GB")

    pipe.vae.enable_tiling()

    #########################################################
    # Setup device placement (with optional offloading)
    # Note: If offloading is enabled, we pre-encode the prompt first,
    # then offload text encoders to CPU to save ~14GB of GPU memory.
    #########################################################
    offload_manager = None
    offload_hooks = None
    pre_encoded_embeds = None  # Will store pre-computed prompt embeddings if offloading

    if args.enable_offload:
        # For offloading, we need to:
        # 1. First move to CUDA to enable text encoding
        # 2. Pre-encode will be done later after prompt is loaded
        logger.info("Offload mode: Pipeline will use pre-encoded embeddings")
    else:
        # Standard mode: load everything to GPU
        pipe.to("cuda")

    config = pipe.transformer.config

    #########################################################
    # Translate the percentage of warmup of layers and timesteps to the actual layers and timesteps
    #########################################################
    ref_scheduler = deepcopy(pipe.scheduler)
    ref_scheduler.set_timesteps(args.num_inference_steps)
    ref_timesteps = ref_scheduler.timesteps
    total_layers = config.num_layers + config.num_single_layers
    
    num_fp_timesteps = math.floor(args.first_times_fp * args.num_inference_steps)
    num_fp_layers = math.floor(args.first_layers_fp * total_layers)
    if num_fp_timesteps > 0:
        args.first_times_fp = ref_scheduler.timesteps[num_fp_timesteps - 1] - 1
    else:
        args.first_times_fp = 1001 # 1000 is the first timestep
    args.first_layers_fp = num_fp_layers
    
    logger.info(f"Warmup of Timesteps: {num_fp_timesteps} / {args.num_inference_steps} || {args.first_times_fp} / 1000 use FP")
    logger.info(f"Warmup of Layers: {num_fp_layers} / {total_layers} use FP")
    
    #########################################################
    # Load the prompt and image path
    #########################################################
    args.prompt, _ = load_prompt_or_image(args.prompt_source, args.prompt_idx, args.prompt, None)

    if args.prompt is None:
        print(colored("Using default prompt", "red"))
        args.prompt = "A cat walks on the grass, realistic"

    if args.negative_prompt is None:
        args.negative_prompt = "Aerial view, aerial view, overexposed, low quality, deformation, a poor composition, bad hands, bad teeth, bad eyes, bad limbs, distortion"
    
    prompt_length = get_prompt_length(pipe, args.prompt)
    print(f"Prompt length: {prompt_length}")

    #########################################################
    # Pre-encode prompt and setup offloading (if enabled)
    # This must happen BEFORE replacing attention
    #########################################################
    if args.enable_offload:
        logger.info("Pre-encoding prompt (text encoders temporarily on GPU)...")
        # Pre-encode prompt - this moves text encoders to GPU, encodes, then offloads to CPU
        # Note: HunyuanVideoPipeline.encode_prompt doesn't support negative_prompt
        pre_encoded_embeds = pre_encode_and_offload(
            pipe,
            prompt=args.prompt,
            device="cuda",
            dtype=torch.bfloat16,
        )

        # Choose offloading strategy
        # Both "layer" and "component" strategies use the unified enable_component_offloading
        # - "layer": Slide entire layers through a window (standard AIO-style)
        # - "component": Pin ALL attention on GPU, slide only FFN (fine-grained)
        strategy_desc = {
            "layer": "Slide entire layers (standard AIO)",
            "component": "Pin attention on GPU, slide FFN only (fine-grained)",
        }
        logger.info(f"Setting up offloading: {strategy_desc.get(args.offload_strategy, args.offload_strategy)}")

        # Show memory analysis
        savings = estimate_memory_savings(pipe)
        logger.info(f"Memory analysis:")
        logger.info(f"  Full model: {savings['full_model_gb']:.2f}GB")
        logger.info(f"  Attention: {savings['attention_total_gb']:.2f}GB")
        logger.info(f"  FFN: {savings['ffn_total_gb']:.2f}GB")
        logger.info(f"  Norm: {savings['norm_total_gb']:.2f}GB")
        logger.info(f"  Per layer: {savings['avg_layer_gb']*1024:.1f}MB")

        # Determine layers on GPU: None for auto, or user-specified value
        if args.offload_auto or args.offload_num_layers is None:
            ffn_layers = None  # Auto-detect based on GPU memory and resolution
            logger.info("Using ADAPTIVE offloading (auto-detect layers based on GPU memory)")
        else:
            ffn_layers = args.offload_num_layers
            logger.info(f"  Window ({ffn_layers} layers): {ffn_layers * savings['avg_layer_gb']:.2f}GB")

        offload_manager, offload_hooks = enable_component_offloading(
            pipe,
            ffn_layers_on_gpu=ffn_layers,
            use_pinned_memory=args.offload_pinned_memory,
            enable_prefetch=args.offload_prefetch,
            ffn_prefetch_count=2,
            verbose=args.offload_verbose,
            # Pass video resolution for adaptive mode activation estimation
            video_height=args.height,
            video_width=args.width,
            num_frames=args.num_frames,
            # Strategy: "layer" (slide whole layers) or "component" (pin attention, slide FFN)
            offload_strategy=args.offload_strategy,
        )

    #########################################################
    # Replace the attention
    #########################################################
    replace_hyvideo_flashattention(pipe)

    if args.pattern == "SVG":
        replace_hyvideo_attention(
            pipe,
            args.height,
            args.width,
            args.num_frames,
            prompt_length,
            first_layers_fp=args.first_layers_fp,
            first_times_fp=args.first_times_fp,
            pattern=args.pattern,
            # SVG specific
            num_sampled_rows=args.num_sampled_rows,
            sample_mse_max_row=args.sample_mse_max_row,
            sparsity=args.sparsity,
        )
    elif args.pattern == "SAP":
        replace_hyvideo_attention(
            pipe,
            args.height,
            args.width,
            args.num_frames,
            prompt_length,
            first_layers_fp=args.first_layers_fp,
            first_times_fp=args.first_times_fp,
            pattern=args.pattern,
            # SAP specific
            num_q_centroids=args.num_q_centroids,
            num_k_centroids=args.num_k_centroids,
            top_p_kmeans=args.top_p_kmeans,
            min_kc_ratio=args.min_kc_ratio,
            logging_file=args.logging_file,
            kmeans_iter_init=args.kmeans_iter_init,
            kmeans_iter_step=args.kmeans_iter_step,
            zero_step_kmeans_init=args.zero_step_kmeans_init,
        )
    elif args.pattern == "SAP_CTCA":
        # SAP with Cross-Timestep Cluster Amortization
        # This variant reduces K-means overhead by reusing cluster assignments
        replace_hyvideo_attention(
            pipe,
            args.height,
            args.width,
            args.num_frames,
            prompt_length,
            first_layers_fp=args.first_layers_fp,
            first_times_fp=args.first_times_fp,
            pattern=args.pattern,
            # SAP specific
            num_q_centroids=args.num_q_centroids,
            num_k_centroids=args.num_k_centroids,
            top_p_kmeans=args.top_p_kmeans,
            min_kc_ratio=args.min_kc_ratio,
            logging_file=args.logging_file,
            kmeans_iter_init=args.kmeans_iter_init,
            kmeans_iter_step=args.kmeans_iter_step,
            zero_step_kmeans_init=args.zero_step_kmeans_init,
            # CTCA specific
            ctca_quality_threshold=args.ctca_quality_threshold,
            ctca_adaptive=args.ctca_adaptive,
            ctca_min_interval=args.ctca_min_interval,
            ctca_max_interval=args.ctca_max_interval,
            ctca_verbose=args.ctca_verbose,
        )
    else:
        assert args.pattern == "dense", f"Invalid pattern: {args.pattern}"
        
    # Print time logger
    for layer_idx, block in enumerate(pipe.transformer.transformer_blocks):
        block.register_forward_hook(print_operator_log_data)
    for layer_idx, block in enumerate(pipe.transformer.single_transformer_blocks):
        block.register_forward_hook(print_operator_log_data)

    #########################################################
    # Generate the video
    #########################################################
    # Reset CTAA statistics for fresh tracking
    reset_ctaa_statistics()

    if pre_encoded_embeds is not None:
        # Use pre-computed embeddings (offload mode)
        # Note: Embedders will be moved to GPU by the forward pre-hook registered in enable_offloading
        logger.info("Using pre-computed prompt embeddings...")
        output = pipe(
            prompt_embeds=pre_encoded_embeds['prompt_embeds'].cuda(),
            pooled_prompt_embeds=pre_encoded_embeds['pooled_prompt_embeds'].cuda(),
            prompt_attention_mask=pre_encoded_embeds['prompt_attention_mask'].cuda(),
            height=args.height,
            width=args.width,
            num_frames=args.num_frames,
            guidance_scale=6.0,
            num_inference_steps=args.num_inference_steps,
        ).frames[0]
    else:
        # Standard mode - encode prompt on-the-fly
        output = pipe(
            prompt=args.prompt,
            negative_prompt=args.negative_prompt,
            height=args.height,
            width=args.width,
            num_frames=args.num_frames,
            guidance_scale=6.0,
            num_inference_steps=args.num_inference_steps,
        ).frames[0]

    # Create parent directory for output file if it doesn't exist
    output_dir = os.path.dirname(args.output_file)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    export_to_video(output, args.output_file, fps=24)

    # Print CTCA and CTAA statistics if using SAP_CTCA pattern
    if args.pattern == "SAP_CTCA":
        print_ctca_statistics()
        print_ctaa_statistics()

    # Print offloading statistics if enabled
    if offload_manager is not None:
        offload_manager.print_statistics()

    logger.info(f"Video saved to {args.output_file}")
