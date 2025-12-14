#!/usr/bin/env python3
"""
Experiment runner for CTAA evaluation.

Usage:
    python experiments/run_experiments.py --suite quick_validation
    python experiments/run_experiments.py --suite parameter_sweep
    python experiments/run_experiments.py --suite quality_evaluation
    python experiments/run_experiments.py --suite ablation_study

    # Custom single experiment
    python experiments/run_experiments.py --config ctaa_default --prompt "A cat walking"
"""

import argparse
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch

from experiments.configs import (
    BASELINE, DEFAULT_CTAA, CONSERVATIVE_CTAA, AGGRESSIVE_CTAA,
    QUICK_VALIDATION_SUITE, PARAMETER_SWEEP_SUITE,
    QUALITY_EVALUATION_SUITE, ABLATION_SUITE,
    ExperimentConfig, ExperimentSuite,
)


def setup_ctaa_config(config: ExperimentConfig):
    """
    Configure CTAA parameters before running inference.
    Must be called before creating the pipeline.
    """
    from svg.models.hyvideo.attention import Hunyuan_SAPAttn_CTCA_Processor2_0

    Hunyuan_SAPAttn_CTCA_Processor2_0.ctaa_enabled = config.ctaa_enabled
    Hunyuan_SAPAttn_CTCA_Processor2_0.ctaa_p_full = config.ctaa_p_full
    Hunyuan_SAPAttn_CTCA_Processor2_0.ctaa_p_total = config.ctaa_p_total

    print(f"[Config] CTAA enabled: {config.ctaa_enabled}")
    print(f"[Config] CTAA p_full: {config.ctaa_p_full}")
    print(f"[Config] CTAA p_total: {config.ctaa_p_total}")


def get_gpu_memory_stats():
    """Get current GPU memory statistics."""
    if torch.cuda.is_available():
        return {
            'allocated_gb': torch.cuda.memory_allocated() / 1024**3,
            'reserved_gb': torch.cuda.memory_reserved() / 1024**3,
            'max_allocated_gb': torch.cuda.max_memory_allocated() / 1024**3,
        }
    return {}


def reset_gpu_memory_stats():
    """Reset GPU memory tracking."""
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.empty_cache()


def collect_ctaa_stats():
    """Collect CTAA statistics after a run."""
    from svg.kmeans_utils import get_ctaa_statistics
    return get_ctaa_statistics()


def collect_ctca_stats():
    """Collect CTCA statistics after a run."""
    from svg.models.hyvideo.attention import Hunyuan_SAPAttn_CTCA_Processor2_0
    return Hunyuan_SAPAttn_CTCA_Processor2_0.get_ctca_statistics()


def run_single_experiment(
    config: ExperimentConfig,
    prompt: str,
    output_dir: str,
    num_inference_steps: int = 30,
    resolution: str = "480p",
    seed: int = 42,
    model_id: str = "tencent/HunyuanVideo",
) -> dict:
    """
    Run a single experiment and collect metrics.

    Returns:
        Dictionary containing timing, memory, and quality metrics.
    """
    import gc
    from diffusers import HunyuanVideoPipeline, FlowMatchEulerDiscreteScheduler
    from diffusers.utils import export_to_video
    from diffusers.models import HunyuanVideoTransformer3DModel

    from svg.models.hyvideo.inference import (
        replace_hyvideo_flashattention,
        replace_hyvideo_attention,
        reset_ctca,
        reset_ctaa_statistics,
    )
    from svg.offload import enable_offloading, pre_encode_and_offload, OffloadConfig
    from svg.utils.seed import seed_everything

    # Set resolution
    if resolution == "480p":
        height, width = 480, 848
    elif resolution == "720p":
        height, width = 720, 1280
    else:
        height, width = 480, 848

    num_frames = 61 if resolution == "480p" else 129

    # Setup output paths
    os.makedirs(output_dir, exist_ok=True)
    video_path = os.path.join(output_dir, f"{config.name}.mp4")
    metrics_path = os.path.join(output_dir, f"{config.name}_metrics.json")

    results = {
        'config': config.to_dict(),
        'prompt': prompt,
        'resolution': resolution,
        'num_inference_steps': num_inference_steps,
        'seed': seed,
        'video_path': video_path,
    }

    # Setup CTAA configuration
    setup_ctaa_config(config)

    # Reset statistics
    reset_gpu_memory_stats()

    # Load pipeline
    print(f"\n{'='*60}")
    print(f"Running experiment: {config.name}")
    print(f"Prompt: {prompt[:50]}...")
    print(f"{'='*60}")

    load_start = time.time()

    # Load transformer separately (matching main inference script)
    transformer = HunyuanVideoTransformer3DModel.from_pretrained(
        model_id, subfolder="transformer", torch_dtype=torch.bfloat16, revision='refs/pr/18'
    )

    # Create scheduler
    flow_shift = 7.0
    scheduler = FlowMatchEulerDiscreteScheduler(shift=flow_shift)

    # Load pipeline with transformer and scheduler
    pipe = HunyuanVideoPipeline.from_pretrained(
        model_id, transformer=transformer, scheduler=scheduler,
        revision='refs/pr/18', torch_dtype=torch.bfloat16
    )

    # Move to CPU first to free GPU memory (for offload mode)
    pipe.to('cpu')
    gc.collect()
    torch.cuda.empty_cache()

    pipe.vae.enable_tiling()

    load_time = time.time() - load_start
    results['model_load_time'] = load_time

    # Setup offloading
    offload_config = OffloadConfig(
        enabled=True,
        num_layers_on_gpu=8,
        enable_prefetch=True,
    )
    offload_manager = enable_offloading(pipe, offload_config)

    # Pre-encode prompts
    pre_encoded_embeds = pre_encode_and_offload(pipe, prompt, None)

    # Replace attention
    replace_hyvideo_flashattention(pipe)
    replace_hyvideo_attention(pipe, pattern="SAP_CTCA")

    # Reset statistics before generation
    reset_ctca()
    reset_ctaa_statistics()

    # Set seed
    seed_everything(seed)

    # Run generation
    gen_start = time.time()
    output = pipe(
        prompt_embeds=pre_encoded_embeds['prompt_embeds'].cuda(),
        pooled_prompt_embeds=pre_encoded_embeds['pooled_prompt_embeds'].cuda(),
        prompt_attention_mask=pre_encoded_embeds['prompt_attention_mask'].cuda(),
        height=height,
        width=width,
        num_frames=num_frames,
        guidance_scale=6.0,
        num_inference_steps=num_inference_steps,
    ).frames[0]
    gen_time = time.time() - gen_start
    results['generation_time'] = gen_time

    # Collect statistics
    results['gpu_memory'] = get_gpu_memory_stats()
    results['ctca_stats'] = collect_ctca_stats()
    results['ctaa_stats'] = collect_ctaa_stats()

    # Save video
    export_to_video(output, video_path, fps=24)
    results['video_saved'] = True

    # Save metrics
    with open(metrics_path, 'w') as f:
        json.dump(results, f, indent=2)

    # Cleanup
    del pipe
    torch.cuda.empty_cache()

    print(f"\nExperiment {config.name} completed:")
    print(f"  Generation time: {gen_time:.2f}s")
    print(f"  Peak GPU memory: {results['gpu_memory'].get('max_allocated_gb', 0):.2f}GB")
    if results['ctaa_stats'].get('total_blocks', 0) > 0:
        print(f"  CTAA full ratio: {results['ctaa_stats'].get('full_ratio', 0)*100:.1f}%")
        print(f"  CTAA skip ratio: {results['ctaa_stats'].get('skip_ratio', 0)*100:.1f}%")

    return results


def run_experiment_suite(suite: ExperimentSuite, output_base_dir: str = "experiment_results",
                         model_id: str = "tencent/HunyuanVideo"):
    """
    Run a full experiment suite.
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    suite_dir = os.path.join(output_base_dir, f"{suite.name}_{timestamp}")
    os.makedirs(suite_dir, exist_ok=True)

    print(f"\n{'#'*60}")
    print(f"Running experiment suite: {suite.name}")
    print(f"Total experiments: {suite.total_experiments()}")
    print(f"Output directory: {suite_dir}")
    print(f"Model: {model_id}")
    print(f"{'#'*60}\n")

    all_results = []

    for prompt_idx, prompt in enumerate(suite.prompts):
        prompt_slug = prompt[:30].replace(" ", "_").replace(",", "")
        prompt_dir = os.path.join(suite_dir, f"prompt_{prompt_idx:02d}_{prompt_slug}")

        for config in suite.configs:
            for run_idx in range(suite.num_runs):
                if suite.num_runs > 1:
                    run_dir = os.path.join(prompt_dir, f"run_{run_idx}")
                else:
                    run_dir = prompt_dir

                seed = 42 + run_idx  # Different seed for each run

                try:
                    result = run_single_experiment(
                        config=config,
                        prompt=prompt,
                        output_dir=run_dir,
                        num_inference_steps=suite.num_inference_steps,
                        resolution=suite.resolution,
                        seed=seed,
                        model_id=model_id,
                    )
                    result['prompt_idx'] = prompt_idx
                    result['run_idx'] = run_idx
                    all_results.append(result)
                except Exception as e:
                    print(f"ERROR in experiment {config.name}: {e}")
                    all_results.append({
                        'config': config.to_dict(),
                        'prompt': prompt,
                        'error': str(e),
                    })

    # Save suite summary
    summary_path = os.path.join(suite_dir, "suite_summary.json")
    with open(summary_path, 'w') as f:
        json.dump({
            'suite_name': suite.name,
            'timestamp': timestamp,
            'total_experiments': len(all_results),
            'results': all_results,
        }, f, indent=2)

    print(f"\n{'#'*60}")
    print(f"Suite completed! Results saved to: {suite_dir}")
    print(f"{'#'*60}")

    return all_results


def main():
    parser = argparse.ArgumentParser(description="Run CTAA experiments")
    parser.add_argument(
        "--suite",
        type=str,
        choices=["quick_validation", "parameter_sweep", "quality_evaluation", "ablation_study"],
        help="Pre-defined experiment suite to run"
    )
    parser.add_argument("--config", type=str, help="Single config name to run")
    parser.add_argument("--prompt", type=str, help="Single prompt for custom run")
    parser.add_argument("--output_dir", type=str, default="experiment_results")
    parser.add_argument("--steps", type=int, default=30)
    parser.add_argument("--resolution", type=str, default="480p", choices=["480p", "720p"])
    parser.add_argument("--model_id", type=str, default="tencent/HunyuanVideo",
                        help="Model ID or local path to HunyuanVideo model")

    args = parser.parse_args()

    if args.suite:
        # Run pre-defined suite
        suites = {
            "quick_validation": QUICK_VALIDATION_SUITE,
            "parameter_sweep": PARAMETER_SWEEP_SUITE,
            "quality_evaluation": QUALITY_EVALUATION_SUITE,
            "ablation_study": ABLATION_SUITE,
        }
        suite = suites[args.suite]
        run_experiment_suite(suite, args.output_dir, model_id=args.model_id)

    elif args.config and args.prompt:
        # Run single custom experiment
        configs = {
            "baseline": BASELINE,
            "ctaa_default": DEFAULT_CTAA,
            "ctaa_conservative": CONSERVATIVE_CTAA,
            "ctaa_aggressive": AGGRESSIVE_CTAA,
        }
        if args.config not in configs:
            print(f"Unknown config: {args.config}")
            print(f"Available: {list(configs.keys())}")
            return

        run_single_experiment(
            config=configs[args.config],
            prompt=args.prompt,
            output_dir=args.output_dir,
            num_inference_steps=args.steps,
            resolution=args.resolution,
            model_id=args.model_id,
        )
    else:
        parser.print_help()
        print("\nExamples:")
        print("  python experiments/run_experiments.py --suite quick_validation")
        print("  python experiments/run_experiments.py --config baseline --prompt 'A cat walking'")


if __name__ == "__main__":
    main()
