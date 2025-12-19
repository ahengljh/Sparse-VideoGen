#!/usr/bin/env python3
"""
Real Workload Memory Sync Test

This script runs actual Video DiT inference with different memory synchronization
strategies to validate the paper's claims about the necessity of sync protocols.

Usage:
    # Test with HunyuanVideo
    python run_real_workload_test.py --model hunyuan --strategy all

    # Test with Wan
    python run_real_workload_test.py --model wan --strategy all

    # Single strategy test
    python run_real_workload_test.py --model hunyuan --strategy conditional_sync
"""

import argparse
import gc
import json
import os
import sys
import time
from copy import deepcopy
from dataclasses import dataclass, asdict
from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional, Tuple, Any

import torch
import torch.nn as nn

# Add parent directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

from real_block_offload import (
    SyncStrategy,
    BlockOffloadManager,
    OffloadingTransformerWrapper,
    OffloadStats,
)


@dataclass
class ExperimentConfig:
    """Configuration for a single experiment run"""
    model_type: str  # "hunyuan" or "wan"
    strategy: SyncStrategy
    working_set_size: int
    num_inference_steps: int
    height: int
    width: int
    num_frames: int
    prompt: str


@dataclass
class ExperimentResult:
    """Results from a single experiment run"""
    config: Dict
    success: bool
    oom_occurred: bool
    error_message: Optional[str]
    total_time_s: float
    inference_time_s: float
    stats: Optional[Dict]
    peak_memory_gb: float
    sync_trigger_rate: float


def get_gpu_memory_info() -> Dict[str, float]:
    """Get current GPU memory statistics"""
    if not torch.cuda.is_available():
        return {"error": "CUDA not available"}

    props = torch.cuda.get_device_properties(0)
    reserved = torch.cuda.memory_reserved(0)
    allocated = torch.cuda.memory_allocated(0)

    return {
        "total_gb": props.total_memory / (1024**3),
        "reserved_gb": reserved / (1024**3),
        "allocated_gb": allocated / (1024**3),
        "free_gb": (props.total_memory - reserved) / (1024**3),
    }


def load_hunyuan_pipeline(model_id: str = None):
    """Load HunyuanVideo pipeline"""
    from diffusers import (
        HunyuanVideoPipeline,
        HunyuanVideoTransformer3DModel,
        FlowMatchEulerDiscreteScheduler
    )

    # Default to local path, fallback to HuggingFace
    if model_id is None:
        local_path = os.path.join(os.path.dirname(__file__), "../../models/HunyuanVideo")
        if os.path.exists(local_path):
            model_id = local_path
        else:
            model_id = "tencent/HunyuanVideo"

    print(f"Loading HunyuanVideo model from {model_id}...")

    # Check if loading from local path (no revision needed)
    is_local = os.path.exists(model_id)

    if is_local:
        transformer = HunyuanVideoTransformer3DModel.from_pretrained(
            model_id,
            subfolder="transformer",
            torch_dtype=torch.bfloat16,
        )
    else:
        transformer = HunyuanVideoTransformer3DModel.from_pretrained(
            model_id,
            subfolder="transformer",
            torch_dtype=torch.bfloat16,
            revision='refs/pr/18'
        )

    flow_shift = 7.0
    scheduler = FlowMatchEulerDiscreteScheduler(shift=flow_shift)

    if is_local:
        pipe = HunyuanVideoPipeline.from_pretrained(
            model_id,
            transformer=transformer,
            scheduler=scheduler,
            torch_dtype=torch.bfloat16
        )
    else:
        pipe = HunyuanVideoPipeline.from_pretrained(
            model_id,
            transformer=transformer,
            scheduler=scheduler,
            revision='refs/pr/18',
            torch_dtype=torch.bfloat16
        )

    pipe.vae.enable_tiling()
    pipe.to("cuda")

    # Count blocks
    num_double = len(list(pipe.transformer.transformer_blocks))
    num_single = len(list(pipe.transformer.single_transformer_blocks))
    print(f"Loaded HunyuanVideo: {num_double} double-stream + {num_single} single-stream blocks")

    return pipe


def load_wan_pipeline(model_id: str = None):
    """Load Wan pipeline"""
    from diffusers import AutoencoderKLWan, WanPipeline
    from diffusers.schedulers.scheduling_unipc_multistep import UniPCMultistepScheduler

    # Default to local path, fallback to HuggingFace
    if model_id is None:
        local_path = os.path.join(os.path.dirname(__file__), "../../models/Wan2.1-T2V-14B")
        if os.path.exists(local_path):
            model_id = local_path
        else:
            model_id = "Wan-AI/Wan2.1-T2V-14B-Diffusers"

    print(f"Loading Wan model from {model_id}...")

    vae = AutoencoderKLWan.from_pretrained(
        model_id,
        subfolder="vae",
        torch_dtype=torch.float32
    )

    flow_shift = 5.0
    scheduler = UniPCMultistepScheduler(
        prediction_type="flow_prediction",
        use_flow_sigmas=True,
        num_train_timesteps=1000,
        flow_shift=flow_shift
    )

    pipe = WanPipeline.from_pretrained(
        model_id,
        vae=vae,
        torch_dtype=torch.bfloat16
    )
    pipe.scheduler = scheduler
    pipe.to("cuda")

    num_blocks = len(list(pipe.transformer.blocks))
    print(f"Loaded Wan: {num_blocks} transformer blocks")

    return pipe


def run_single_experiment(
    pipe,
    config: ExperimentConfig,
) -> ExperimentResult:
    """Run a single experiment with the given configuration"""

    print(f"\n{'='*60}")
    print(f"Running experiment: {config.strategy.value}")
    print(f"Working set size: {config.working_set_size}")
    print(f"Resolution: {config.height}x{config.width}, {config.num_frames} frames")
    print(f"{'='*60}")

    # Reset memory stats
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.empty_cache()
    gc.collect()

    start_time = time.time()
    oom_occurred = False
    error_message = None
    stats = None
    inference_time = 0.0

    try:
        # Create wrapper with offloading
        wrapper = OffloadingTransformerWrapper(
            pipe.transformer,
            working_set_size=config.working_set_size,
            strategy=config.strategy,
            device=0,
        )

        # Enable offloading
        wrapper.enable_offloading()

        # Memory before inference
        mem_before = get_gpu_memory_info()
        print(f"Memory before inference: {mem_before['allocated_gb']:.2f} GB allocated")

        # Run inference
        inference_start = time.time()

        if config.model_type == "hunyuan":
            output = pipe(
                prompt=config.prompt,
                negative_prompt="low quality, blurry",
                height=config.height,
                width=config.width,
                num_frames=config.num_frames,
                guidance_scale=6.0,
                num_inference_steps=config.num_inference_steps,
            )
        else:  # wan
            output = pipe(
                prompt=config.prompt,
                negative_prompt="low quality, blurry",
                height=config.height,
                width=config.width,
                num_frames=config.num_frames,
                guidance_scale=5.0,
                num_inference_steps=config.num_inference_steps,
            )

        inference_time = time.time() - inference_start

        # Get stats
        offload_stats = wrapper.get_stats()
        stats = {
            "total_blocks": offload_stats.total_blocks,
            "working_set_size": offload_stats.working_set_size,
            "strategy": offload_stats.strategy,
            "total_load_count": offload_stats.total_load_count,
            "total_offload_count": offload_stats.total_offload_count,
            "total_sync_count": offload_stats.total_sync_count,
            "total_oom_count": offload_stats.total_oom_count,
            "total_load_time_s": offload_stats.total_load_time_s,
            "total_offload_time_s": offload_stats.total_offload_time_s,
            "peak_memory_gb": offload_stats.peak_memory_gb,
        }

        # Print stats
        wrapper.print_stats()

        # Disable offloading and restore model
        wrapper.disable_offloading()

    except RuntimeError as e:
        if 'out of memory' in str(e).lower():
            oom_occurred = True
            error_message = str(e)
            print(f"OOM occurred: {error_message[:200]}")
        else:
            error_message = str(e)
            print(f"Error: {error_message}")
    except Exception as e:
        error_message = str(e)
        print(f"Unexpected error: {error_message}")

    total_time = time.time() - start_time
    peak_memory_gb = torch.cuda.max_memory_allocated() / (1024**3)

    # Calculate sync trigger rate
    sync_trigger_rate = 0.0
    if stats:
        total_transfers = stats["total_load_count"] + stats["total_offload_count"]
        if total_transfers > 0:
            sync_trigger_rate = stats["total_sync_count"] / total_transfers

    return ExperimentResult(
        config=asdict(config),
        success=not oom_occurred and error_message is None,
        oom_occurred=oom_occurred,
        error_message=error_message,
        total_time_s=total_time,
        inference_time_s=inference_time,
        stats=stats,
        peak_memory_gb=peak_memory_gb,
        sync_trigger_rate=sync_trigger_rate,
    )


def run_strategy_comparison(
    model_type: str,
    strategies: List[SyncStrategy],
    working_set_size: int = 5,
    num_inference_steps: int = 20,
    height: int = 480,
    width: int = 848,
    num_frames: int = 49,
    prompt: str = "A cat walking in the garden, realistic",
    output_dir: str = "./results",
    model_path: Optional[str] = None,
) -> Dict[str, Any]:
    """Run comparison across multiple sync strategies"""

    print(f"\n{'='*80}")
    print("MEMORY SYNC STRATEGY COMPARISON - REAL WORKLOAD TEST")
    print(f"{'='*80}")
    print(f"Model: {model_type}")
    if model_path:
        print(f"Model path: {model_path}")
    print(f"Resolution: {height}x{width}, {num_frames} frames")
    print(f"Inference steps: {num_inference_steps}")
    print(f"Working set size: {working_set_size} blocks")
    print(f"Strategies to test: {[s.value for s in strategies]}")

    # Load model
    if model_type == "hunyuan":
        pipe = load_hunyuan_pipeline(model_path)
    elif model_type == "wan":
        pipe = load_wan_pipeline(model_path)
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    results = {}

    for strategy in strategies:
        config = ExperimentConfig(
            model_type=model_type,
            strategy=strategy,
            working_set_size=working_set_size,
            num_inference_steps=num_inference_steps,
            height=height,
            width=width,
            num_frames=num_frames,
            prompt=prompt,
        )

        # Run experiment
        result = run_single_experiment(pipe, config)
        results[strategy.value] = asdict(result)

        # Clean up between runs
        torch.cuda.empty_cache()
        gc.collect()
        time.sleep(2)  # Give GPU time to settle

    # Generate summary
    summary = generate_summary(results)

    # Save results
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    output_file = os.path.join(output_dir, f"real_workload_results_{timestamp}.json")
    with open(output_file, 'w') as f:
        json.dump({
            "timestamp": timestamp,
            "model_type": model_type,
            "config": {
                "height": height,
                "width": width,
                "num_frames": num_frames,
                "num_inference_steps": num_inference_steps,
                "working_set_size": working_set_size,
            },
            "results": results,
            "summary": summary,
        }, f, indent=2)

    print(f"\nResults saved to: {output_file}")

    return {
        "results": results,
        "summary": summary,
        "output_file": output_file,
    }


def generate_summary(results: Dict[str, Dict]) -> Dict:
    """Generate summary statistics for paper"""

    summary = {}

    for strategy, result in results.items():
        summary[strategy] = {
            "success": result["success"],
            "oom_occurred": result["oom_occurred"],
            "peak_memory_gb": result["peak_memory_gb"],
            "sync_trigger_rate": result["sync_trigger_rate"],
            "inference_time_s": result["inference_time_s"],
        }

        if result["stats"]:
            summary[strategy]["total_loads"] = result["stats"]["total_load_count"]
            summary[strategy]["total_offloads"] = result["stats"]["total_offload_count"]
            summary[strategy]["total_syncs"] = result["stats"]["total_sync_count"]
            summary[strategy]["oom_count"] = result["stats"]["total_oom_count"]

    # Print formatted summary
    print(f"\n{'='*80}")
    print("SUMMARY FOR PAPER")
    print(f"{'='*80}")
    print(f"{'Strategy':<20} {'OOM':<8} {'Peak Mem':<12} {'Sync Rate':<12} {'Time':<10}")
    print(f"{'-'*70}")

    for strategy, data in summary.items():
        oom_str = "Yes" if data.get("oom_occurred", False) else "No"
        peak_mem = f"{data.get('peak_memory_gb', 0):.2f} GB"
        sync_rate = f"{data.get('sync_trigger_rate', 0)*100:.1f}%"
        time_str = f"{data.get('inference_time_s', 0):.1f}s"

        print(f"{strategy:<20} {oom_str:<8} {peak_mem:<12} {sync_rate:<12} {time_str:<10}")

    print(f"{'='*80}")

    return summary


def main():
    parser = argparse.ArgumentParser(
        description="Run real workload memory sync tests"
    )

    parser.add_argument(
        "--model",
        type=str,
        default="hunyuan",
        choices=["hunyuan", "wan"],
        help="Model to test"
    )

    parser.add_argument(
        "--model-path",
        type=str,
        default=None,
        help="Path to local model directory (e.g., models/HunyuanVideo)"
    )

    parser.add_argument(
        "--strategy",
        type=str,
        default="all",
        choices=["pure_async", "pure_sync", "conditional_sync", "all"],
        help="Sync strategy to test"
    )

    parser.add_argument(
        "--working-set-size",
        type=int,
        default=5,
        help="Number of blocks to keep on GPU"
    )

    parser.add_argument(
        "--num-inference-steps",
        type=int,
        default=20,
        help="Number of denoising steps"
    )

    parser.add_argument(
        "--height",
        type=int,
        default=480,
        help="Video height"
    )

    parser.add_argument(
        "--width",
        type=int,
        default=848,
        help="Video width"
    )

    parser.add_argument(
        "--num-frames",
        type=int,
        default=49,
        help="Number of video frames"
    )

    parser.add_argument(
        "--prompt",
        type=str,
        default="A cat walking in a beautiful garden, high quality, realistic",
        help="Text prompt for video generation"
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default="./results/real_workload",
        help="Output directory for results"
    )

    args = parser.parse_args()

    # Determine strategies to test
    if args.strategy == "all":
        strategies = [
            SyncStrategy.PURE_ASYNC,
            SyncStrategy.PURE_SYNC,
            SyncStrategy.CONDITIONAL_SYNC,
        ]
    else:
        strategy_map = {
            "pure_async": SyncStrategy.PURE_ASYNC,
            "pure_sync": SyncStrategy.PURE_SYNC,
            "conditional_sync": SyncStrategy.CONDITIONAL_SYNC,
        }
        strategies = [strategy_map[args.strategy]]

    # Check CUDA availability
    if not torch.cuda.is_available():
        print("ERROR: CUDA is not available. GPU required for these experiments.")
        sys.exit(1)

    # Print GPU info
    mem_info = get_gpu_memory_info()
    print(f"\nGPU Memory: {mem_info['total_gb']:.1f} GB total, {mem_info['free_gb']:.1f} GB free")

    # Run experiments
    run_strategy_comparison(
        model_type=args.model,
        strategies=strategies,
        working_set_size=args.working_set_size,
        num_inference_steps=args.num_inference_steps,
        height=args.height,
        width=args.width,
        num_frames=args.num_frames,
        prompt=args.prompt,
        output_dir=args.output_dir,
        model_path=args.model_path,
    )


if __name__ == "__main__":
    main()
