import argparse
import gc
import json
import os
import time
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
from svg.models.hyvideo.attention import VideoKReuseConfig
from svg.models.hyvideo.inference import (
    replace_hyvideo_flashattention,
    replace_hyvideo_attention,
)
from svg.models.hyvideo.utils import get_prompt_length
from svg.offload import enable_offloading, pre_encode_and_offload

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

    parser.add_argument("--pattern", type=str, default="dense", choices=["SVG", "dense", "SAP"])
    parser.add_argument("--first_layers_fp", type=float, default=0.025, help="Only works for best config. Leave the 0, 1, 2, 40, 41 layers in FP")
    parser.add_argument("--first_times_fp", type=float, default=0.075, help="Only works for best config. Leave the first 10% timestep in FP")

    # SVG1 specific
    parser.add_argument("--num_sampled_rows", type=int, default=64, help="The number of sampled rows")
    parser.add_argument("--sample_mse_max_row", type=int, default=10000, help="The maximum number of rows in attention mask. Prevent OOM.")
    parser.add_argument("--sparsity", type=float, default=0.25, help="The sparsity of the striped attention pattern. Accepts one or two float values.")

    # SVG2 specific
    parser.add_argument("--num_q_centroids", "--qc", type=int, default=50, help="Number of query centroids for SAP.")
    parser.add_argument("--num_k_centroids", "--kc", type=int, default=200, help="Number of key centroids for SAP.")
    parser.add_argument("--top_p_kmeans", type=float, default=0.9, help="Top-p threshold for block selection in SAP.")
    parser.add_argument("--min_kc_ratio", type=float, default=0, help="At least this proportion of key blocks to keep per query block in SAP.")
    parser.add_argument("--kmeans_iter_init", type=int, default=0, help="Number of KMeans iterations for initialization in SAP.")
    parser.add_argument("--kmeans_iter_step", type=int, default=0, help="Number of KMeans iterations for other diffusion steps in SAP.")
    parser.add_argument("--zero_step_kmeans_init", action="store_true", help="Initialize the centroids for the first step in SAP, not after warmup.")

    # Attention output caching across diffusion steps
    parser.add_argument("--video_k_reuse", action="store_true", help="Cache attention output for reuse across steps.")
    parser.add_argument("--video_k_reuse_warmup_steps", type=int, default=2, help="Warmup steps before caching begins.")
    parser.add_argument("--video_k_reuse_start_step", type=int, default=4, help="Start reusing cached output after this step.")
    parser.add_argument("--video_k_reuse_interval", type=int, default=2, help="Refresh cache every N steps.")
    parser.add_argument("--video_k_reuse_layer_stride", type=int, default=1, help="Apply caching every N layers.")
    parser.add_argument("--video_k_reuse_max_layers", type=int, default=60, help="Cap number of layers that cache.")
    parser.add_argument("--video_k_reuse_layers", type=str, default=None, help="Comma-separated list of layer indices to cache.")
    parser.add_argument("--video_k_reuse_metrics", action="store_true", help="Collect per-step reuse metrics.")
    parser.add_argument(
        "--video_k_reuse_metrics_jsonl",
        type=str,
        default=None,
        help="Write per-step metrics as JSONL to this path.",
    )

    # Dynamic Offloading - enables running on smaller GPUs (e.g., 4090 24GB)
    parser.add_argument("--enable_offload", action="store_true", help="Enable dynamic layer offloading to run on smaller GPUs.")
    parser.add_argument("--offload_num_layers", type=int, default=6, help="Number of transformer layers to keep on GPU (sliding window size). Higher=faster but more VRAM. Recommended: 4-8 for 24GB, 10-15 for 40GB+.")
    parser.add_argument("--offload_max_memory_gb", type=float, default=None, help="Auto-tune num_layers based on memory budget (e.g., 20.0 for 24GB GPU).")
    parser.add_argument("--offload_pinned_memory", action="store_true", default=True, help="Use pinned CPU memory for faster transfers.")
    parser.add_argument("--offload_no_pinned_memory", action="store_false", dest="offload_pinned_memory", help="Disable pinned memory.")
    parser.add_argument("--offload_prefetch", action="store_true", default=True, help="Enable async prefetching of next layer.")
    parser.add_argument("--offload_no_prefetch", action="store_false", dest="offload_prefetch", help="Disable async prefetching.")
    parser.add_argument("--offload_verbose", action="store_true", help="Enable verbose offloading logging.")
    parser.add_argument("--offload_auto", action="store_true", help="Auto-tune offloading window based on available VRAM.")
    parser.add_argument("--offload_max_fraction", type=float, default=0.90, help="When --offload_auto is set, target this fraction of total VRAM.")
    parser.add_argument("--offload_activation_reserve_gb", type=float, default=4.0, help="Reserve VRAM for activations/caches when auto-tuning offload.")
    parser.add_argument("--offload_cuda_overhead_gb", type=float, default=0.5, help="Extra VRAM headroom for CUDA workspaces when auto-tuning offload.")
    parser.add_argument("--offload_auto_allow_increase", action="store_true", help="Allow auto-tune to increase num_layers_on_gpu above --offload_num_layers.")

    # GPU memory limit — for reproducing 24GB consumer GPU conditions on larger GPUs
    parser.add_argument("--max_gpu_memory_gb", type=float, default=None, help="Restrict PyTorch CUDA memory to this many GB (e.g., 24.0). Simulates consumer GPU on larger hardware.")

    args = parser.parse_args()

    seed_everything(args.seed)

    # Enforce GPU memory limit before any CUDA allocation
    if args.max_gpu_memory_gb is not None and torch.cuda.is_available():
        total_bytes = torch.cuda.get_device_properties(0).total_memory
        total_gb = total_bytes / (1024 ** 3)
        fraction = min(1.0, args.max_gpu_memory_gb / total_gb)
        torch.cuda.set_per_process_memory_fraction(fraction)
        logger.info(f"GPU memory restricted to {args.max_gpu_memory_gb:.1f}GB ({fraction*100:.0f}% of {total_gb:.1f}GB)")

    # In some cases it will raise RuntimeError: cusolver error: CUSOLVER_STATUS_INTERNAL_ERROR
    torch.backends.cuda.preferred_linalg_library(backend="magma")
    
    if args.skip_existing:
        if os.path.exists(args.output_file):
            logger.info(f"Output file {args.output_file} already exists. Skipping generation.")
            exit(0)

    #########################################################
    # Load the model
    #########################################################
    if args.enable_offload:
        logger.info("Loading model for offload mode...")
    transformer = HunyuanVideoTransformer3DModel.from_pretrained(
        args.model_id, subfolder="transformer", torch_dtype=torch.bfloat16, revision='refs/pr/18'
    )
    flow_shift = 7.0
    scheduler = FlowMatchEulerDiscreteScheduler(shift=flow_shift)
    pipe = HunyuanVideoPipeline.from_pretrained(
        args.model_id, transformer=transformer, scheduler=scheduler,
        revision='refs/pr/18', torch_dtype=torch.bfloat16
    )
    if args.enable_offload:
        # Immediately move everything to CPU to free GPU memory
        # This is crucial for 24GB GPUs - diffusers may load to GPU during from_pretrained
        logger.info("Moving all pipeline components to CPU to free GPU memory...")
        pipe.to("cpu")
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

        # Now enable transformer layer offloading (text encoders already on CPU)
        logger.info("Setting up transformer layer offloading...")
        offload_manager, offload_hooks = enable_offloading(
            pipe,
            use_pinned_memory=args.offload_pinned_memory,
            enable_prefetch=args.offload_prefetch,
            num_layers_on_gpu=args.offload_num_layers,
            max_memory_gb=args.offload_max_memory_gb,
            auto_tune_layers_on_gpu=args.offload_auto,
            max_memory_fraction=args.offload_max_fraction,
            activation_reserve_gb=args.offload_activation_reserve_gb,
            cuda_overhead_gb=args.offload_cuda_overhead_gb,
            auto_tune_allow_increase=args.offload_auto_allow_increase,
            verbose=args.offload_verbose,
        )

    #########################################################
    # Replace the attention
    #########################################################
    video_k_reuse_config = None
    if args.video_k_reuse:
        layer_indices = None
        if args.video_k_reuse_layers:
            layer_indices = tuple(
                int(x.strip()) for x in args.video_k_reuse_layers.split(",") if x.strip()
            )
        warmup_steps = max(2, args.video_k_reuse_warmup_steps)
        start_step = max(warmup_steps, args.video_k_reuse_start_step)
        video_k_reuse_config = VideoKReuseConfig(
            enabled=True,
            warmup_steps=warmup_steps,
            start_step=start_step,
            interval=max(1, args.video_k_reuse_interval),
            layer_stride=max(1, args.video_k_reuse_layer_stride),
            max_layers=max(1, args.video_k_reuse_max_layers),
            layer_indices=layer_indices,
            metrics_enabled=bool(args.video_k_reuse_metrics),
        )

    # Note: video_k_reuse_config is NOT passed to attention processors anymore.
    # Block-level caching is handled in the transformer forward loop instead,
    # which enables skipping weight loading entirely for cached blocks.
    replace_hyvideo_flashattention(
        pipe,
    )

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
    else:
        assert args.pattern == "dense", f"Invalid pattern: {args.pattern}"
        # For dense pattern, still need sparse forward replacement for block-level caching
        if video_k_reuse_config is not None:
            from svg.models.hyvideo.custom_models import replace_sparse_forward
            replace_sparse_forward()

    # Set up block-level output caching on the transformer
    if video_k_reuse_config is not None:
        pipe.transformer._block_cache_cfg = video_k_reuse_config
        pipe.transformer.reset_block_cache()
        logger.info(f"Block-level output caching enabled: {video_k_reuse_config}")

    # Print time logger
    for layer_idx, block in enumerate(pipe.transformer.transformer_blocks):
        block.register_forward_hook(print_operator_log_data)
    for layer_idx, block in enumerate(pipe.transformer.single_transformer_blocks):
        block.register_forward_hook(print_operator_log_data)

    #########################################################
    # Generate the video
    #########################################################
    # Per-step GPU memory tracking
    gpu_memory_trajectory = []

    def _gpu_memory_callback(pipe_self, step, timestep, callback_kwargs):
        if torch.cuda.is_available():
            allocated_mb = torch.cuda.memory_allocated() / (1024 ** 2)
            reserved_mb = torch.cuda.memory_reserved() / (1024 ** 2)
            peak_mb = torch.cuda.max_memory_allocated() / (1024 ** 2)
            elapsed = time.perf_counter() - t_start
            gpu_memory_trajectory.append({
                "step": step,
                "elapsed_s": round(elapsed, 2),
                "allocated_mb": round(allocated_mb, 1),
                "reserved_mb": round(reserved_mb, 1),
                "peak_mb": round(peak_mb, 1),
            })
        return callback_kwargs

    # Reset peak memory tracking and start wall-clock timer
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
    t_start = time.perf_counter()

    if pre_encoded_embeds is not None:
        # Use pre-computed embeddings (offload mode)
        # Note: Embedders will be moved to GPU by the forward pre-hook registered in enable_offloading
        logger.info("Using pre-computed prompt embeddings...")
        output = pipe(
            prompt_embeds=pre_encoded_embeds["prompt_embeds"].cuda(),
            pooled_prompt_embeds=pre_encoded_embeds["pooled_prompt_embeds"].cuda(),
            prompt_attention_mask=pre_encoded_embeds["prompt_attention_mask"].cuda(),
            height=args.height,
            width=args.width,
            num_frames=args.num_frames,
            guidance_scale=6.0,
            num_inference_steps=args.num_inference_steps,
            callback_on_step_end=_gpu_memory_callback,
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
            callback_on_step_end=_gpu_memory_callback,
        ).frames[0]

    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t_end = time.perf_counter()
    wall_clock_s = t_end - t_start
    peak_gpu_mb = torch.cuda.max_memory_allocated() / (1024 ** 2) if torch.cuda.is_available() else 0

    logger.info(f"Inference wall-clock: {wall_clock_s:.2f}s | Peak GPU memory: {peak_gpu_mb:.0f}MB")

    # Create parent directory for output file if it doesn't exist
    output_dir = os.path.dirname(args.output_file)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    export_to_video(output, args.output_file, fps=24)

    # Print block-level output caching statistics
    if args.video_k_reuse:
        block_stats = pipe.transformer.get_block_cache_stats()
        total_ops = block_stats["hits"] + block_stats["misses"]
        hit_rate = (block_stats["hits"] / total_ops) * 100 if total_ops > 0 else 0.0
        logger.info(
            f"Block cache stats: hits={block_stats['hits']} misses={block_stats['misses']} "
            f"hit_rate={hit_rate:.1f}%"
        )

    if args.video_k_reuse and args.video_k_reuse_metrics:
        # Write block-level timing
        block_timing = pipe.transformer.get_block_cache_timing()
        if block_timing:
            timing_path = args.output_file + ".block_timing.jsonl"
            with open(timing_path, "w", encoding="utf-8") as handle:
                for entry in block_timing:
                    handle.write(json.dumps(entry, sort_keys=True) + "\n")
            logger.info(f"Block cache timing: {len(block_timing)} entries -> {timing_path}")

    # Print offloading statistics if enabled
    if offload_manager is not None:
        offload_manager.print_statistics()

    #########################################################
    # Write per-run summary JSON (timing, memory, config)
    # Written alongside the video as <output_file>.run.json
    #########################################################
    run_summary = {
        "output_file": args.output_file,
        "wall_clock_s": round(wall_clock_s, 2),
        "peak_gpu_mb": round(peak_gpu_mb, 0),
        "height": args.height,
        "width": args.width,
        "num_frames": args.num_frames,
        "num_inference_steps": args.num_inference_steps,
        "pattern": args.pattern,
        "seed": args.seed,
        "offload": args.enable_offload,
        "offload_num_layers": args.offload_num_layers if args.enable_offload else None,
        "video_k_reuse": args.video_k_reuse,
    }
    if args.video_k_reuse:
        block_stats = pipe.transformer.get_block_cache_stats()
        run_summary.update({
            "kv_warmup_steps": args.video_k_reuse_warmup_steps,
            "kv_start_step": args.video_k_reuse_start_step,
            "kv_interval": args.video_k_reuse_interval,
            "kv_layer_stride": args.video_k_reuse_layer_stride,
            "kv_max_layers": args.video_k_reuse_max_layers,
            "block_cache_hits": block_stats["hits"],
            "block_cache_misses": block_stats["misses"],
        })

    # Add GPU memory trajectory to run summary
    if gpu_memory_trajectory:
        run_summary["gpu_memory_trajectory"] = gpu_memory_trajectory

    summary_path = args.output_file + ".run.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(run_summary, f, indent=2)
    logger.info(f"Run summary written to {summary_path}")

    # Also save GPU memory trajectory as a separate CSV for easy plotting
    if gpu_memory_trajectory:
        gpu_csv_path = args.output_file + ".gpu_memory.csv"
        import csv as csv_mod
        with open(gpu_csv_path, "w", newline="") as f:
            writer = csv_mod.DictWriter(f, fieldnames=["step", "elapsed_s", "allocated_mb", "reserved_mb", "peak_mb"])
            writer.writeheader()
            writer.writerows(gpu_memory_trajectory)
        logger.info(f"GPU memory trajectory written to {gpu_csv_path}")
