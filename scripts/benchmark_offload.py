#!/usr/bin/env python3
"""
Benchmark script to quantify offloading benefits.

Measures:
1. Peak GPU memory usage
2. Inference time
3. Prefetch efficiency (hits vs misses)
4. Memory savings

Usage:
    python scripts/benchmark_offload.py --with-offload
    python scripts/benchmark_offload.py --without-offload
"""

import argparse
import time
import torch
import subprocess
import threading
import json
from pathlib import Path


def monitor_gpu_memory(interval=0.1, stop_event=None, memory_samples=None):
    """Monitor GPU memory usage in a background thread."""
    while not stop_event.is_set():
        try:
            result = subprocess.run(
                ['nvidia-smi', '--query-gpu=memory.used', '--format=csv,noheader,nounits'],
                capture_output=True, text=True
            )
            memory_mb = int(result.stdout.strip().split('\n')[0])
            memory_samples.append(memory_mb)
        except Exception:
            pass
        time.sleep(interval)


def run_inference(args):
    """Run inference and collect metrics."""
    from diffusers import HunyuanVideoPipeline
    from svg.component_offload import (
        enable_component_offloading,
        get_offload_manager
    )

    # Start memory monitoring
    memory_samples = []
    stop_event = threading.Event()
    monitor_thread = threading.Thread(
        target=monitor_gpu_memory,
        args=(0.1, stop_event, memory_samples)
    )
    monitor_thread.start()

    # Load pipeline
    print("Loading pipeline...")
    load_start = time.time()
    pipe = HunyuanVideoPipeline.from_pretrained(
        args.model_id,
        torch_dtype=torch.bfloat16,
    )

    if args.with_offload:
        print("Enabling component offloading...")
        enable_component_offloading(
            pipe,
            ffn_layers_on_gpu=args.num_layers,
            use_pinned_memory=True,
            enable_prefetch=True,
            ffn_prefetch_count=2,
        )
    else:
        print("Moving full model to GPU (no offloading)...")
        pipe.to("cuda")

    load_time = time.time() - load_start

    # Warm-up
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()

    # Run inference
    print("Running inference...")
    inference_start = time.time()

    video = pipe(
        prompt=args.prompt,
        height=480,
        width=848,
        num_frames=args.num_frames,
        num_inference_steps=args.steps,
        generator=torch.Generator("cuda").manual_seed(42),
    ).frames[0]

    torch.cuda.synchronize()
    inference_time = time.time() - inference_start

    # Stop monitoring
    stop_event.set()
    monitor_thread.join()

    # Collect metrics
    peak_memory_pytorch = torch.cuda.max_memory_allocated() / (1024 ** 3)  # GB
    peak_memory_nvidia = max(memory_samples) / 1024 if memory_samples else 0  # GB

    metrics = {
        "mode": "with_offload" if args.with_offload else "baseline",
        "num_layers_on_gpu": args.num_layers if args.with_offload else "all",
        "load_time_seconds": round(load_time, 2),
        "inference_time_seconds": round(inference_time, 2),
        "peak_memory_pytorch_gb": round(peak_memory_pytorch, 2),
        "peak_memory_nvidia_gb": round(peak_memory_nvidia, 2),
        "num_frames": args.num_frames,
        "inference_steps": args.steps,
    }

    # Get offload stats if available
    if args.with_offload:
        manager = get_offload_manager()
        if manager:
            # Stop tracking and print stats
            manager.stop_tracking()
            manager.print_statistics()

            metrics["offload_stats"] = {
                "layer_loads": manager.stats['layer_loads'],
                "layer_offloads": manager.stats['layer_offloads'],
            }
            metrics["peak_memory_offload_gb"] = round(manager._peak_memory_gb, 2)

    return metrics


def main():
    parser = argparse.ArgumentParser(description="Benchmark offloading performance")
    parser.add_argument("--model-id", default="tencent/HunyuanVideo", help="Model ID")
    parser.add_argument("--prompt", default="A cat walks on the grass.", help="Prompt")
    parser.add_argument("--num-frames", type=int, default=45, help="Number of frames")
    parser.add_argument("--steps", type=int, default=30, help="Inference steps")
    parser.add_argument("--num-layers", type=int, default=6, help="Layers on GPU (with offload)")

    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--with-offload", action="store_true", help="Run with offloading")
    group.add_argument("--without-offload", action="store_true", help="Run without offloading")

    parser.add_argument("--output", default="benchmark_results.json", help="Output file")

    args = parser.parse_args()

    print("=" * 60)
    print("OFFLOADING BENCHMARK")
    print("=" * 60)
    print(f"Mode: {'With Offload' if args.with_offload else 'Baseline (No Offload)'}")
    print(f"Frames: {args.num_frames}, Steps: {args.steps}")
    if args.with_offload:
        print(f"Layers on GPU: {args.num_layers}")
    print("=" * 60)

    metrics = run_inference(args)

    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    print(json.dumps(metrics, indent=2))

    # Append to results file
    results_file = Path(args.output)
    if results_file.exists():
        with open(results_file) as f:
            all_results = json.load(f)
    else:
        all_results = []

    all_results.append(metrics)

    with open(results_file, 'w') as f:
        json.dump(all_results, f, indent=2)

    print(f"\nResults saved to {args.output}")

    # Print comparison if both modes exist
    if len(all_results) >= 2:
        baseline = next((r for r in all_results if r['mode'] == 'baseline'), None)
        offload = next((r for r in all_results if r['mode'] == 'with_offload'), None)

        if baseline and offload:
            print("\n" + "=" * 60)
            print("COMPARISON")
            print("=" * 60)
            mem_saved = baseline['peak_memory_nvidia_gb'] - offload['peak_memory_nvidia_gb']
            mem_pct = mem_saved / baseline['peak_memory_nvidia_gb'] * 100
            time_overhead = offload['inference_time_seconds'] - baseline['inference_time_seconds']
            time_pct = time_overhead / baseline['inference_time_seconds'] * 100

            print(f"Memory Saved: {mem_saved:.1f} GB ({mem_pct:.1f}% reduction)")
            print(f"Time Overhead: {time_overhead:.1f}s ({time_pct:.1f}% slower)")


if __name__ == "__main__":
    main()
