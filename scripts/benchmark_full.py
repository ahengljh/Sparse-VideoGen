#!/usr/bin/env python3
"""
Full benchmark script for SAP_CTCA + Component Offloading.

Measures the complete system performance including:
1. SAP sparse attention with CTCA
2. Component-level offloading
3. Memory and time metrics

Usage:
    # Full system (SAP_CTCA + Offloading)
    python scripts/benchmark_full.py --mode full

    # Offloading only (no sparse attention)
    python scripts/benchmark_full.py --mode offload-only

    # Baseline (no optimizations, requires 40GB+ GPU)
    python scripts/benchmark_full.py --mode baseline
"""

import argparse
import json
import os
import subprocess
import sys
import threading
import time
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
    """Run inference using the actual inference script."""

    # Start memory monitoring
    memory_samples = []
    stop_event = threading.Event()
    monitor_thread = threading.Thread(
        target=monitor_gpu_memory,
        args=(0.1, stop_event, memory_samples)
    )
    monitor_thread.start()

    # Build command based on mode
    cmd = [
        sys.executable, "hyvideo_t2v_inference.py",
        "--model_id", args.model_id,
        "--prompt", args.prompt,
        "--resolution", "540p",  # Use lower res for faster benchmarking
        "--num_frames", str(args.num_frames),
        "--num_inference_steps", str(args.steps),
        "--output_file", f"benchmark_{args.mode}.mp4",
        "--seed", "42",
    ]

    if args.mode == "full":
        # Full system: SAP_CTCA + Offloading
        cmd.extend([
            "--pattern", "SAP_CTCA",
            "--num_q_centroids", "100",
            "--num_k_centroids", "400",
            "--top_p_kmeans", "0.95",
            "--min_kc_ratio", "0.05",
            "--first_layers_fp", "0.1",
            "--first_times_fp", "0.15",
            "--kmeans_iter_init", "10",
            "--kmeans_iter_step", "3",
            "--ctca_quality_threshold", "0.88",
            "--ctca_min_interval", "1",
            "--ctca_max_interval", "5",
            "--enable_offload",
            "--offload_strategy", "component",
            "--offload_num_layers", str(args.num_layers),
            "--offload_pinned_memory",
            "--offload_prefetch",
        ])
    elif args.mode == "offload-only":
        # Offloading only, no sparse attention
        cmd.extend([
            "--pattern", "full",  # Full attention
            "--enable_offload",
            "--offload_strategy", "component",
            "--offload_num_layers", str(args.num_layers),
            "--offload_pinned_memory",
            "--offload_prefetch",
        ])
    elif args.mode == "sparse-only":
        # SAP_CTCA only, no offloading
        cmd.extend([
            "--pattern", "SAP_CTCA",
            "--num_q_centroids", "100",
            "--num_k_centroids", "400",
            "--top_p_kmeans", "0.95",
            "--min_kc_ratio", "0.05",
            "--first_layers_fp", "0.1",
            "--first_times_fp", "0.15",
            "--kmeans_iter_init", "10",
            "--kmeans_iter_step", "3",
            "--ctca_quality_threshold", "0.88",
            "--ctca_min_interval", "1",
            "--ctca_max_interval", "5",
        ])
    # baseline mode: no extra flags

    print(f"Running: {' '.join(cmd)}")
    print("-" * 60)

    start_time = time.time()
    result = subprocess.run(cmd, capture_output=False)
    inference_time = time.time() - start_time

    # Stop monitoring
    stop_event.set()
    monitor_thread.join()

    peak_memory_gb = max(memory_samples) / 1024 if memory_samples else 0

    metrics = {
        "mode": args.mode,
        "num_layers_on_gpu": args.num_layers if args.mode in ["full", "offload-only"] else "all",
        "inference_time_seconds": round(inference_time, 2),
        "peak_memory_nvidia_gb": round(peak_memory_gb, 2),
        "num_frames": args.num_frames,
        "inference_steps": args.steps,
        "return_code": result.returncode,
    }

    return metrics


def main():
    parser = argparse.ArgumentParser(description="Benchmark full SAP_CTCA + Offloading system")
    parser.add_argument("--model-id", default="tencent/HunyuanVideo", help="Model ID")
    parser.add_argument("--prompt", default="A cat walks on the grass.", help="Prompt")
    parser.add_argument("--num-frames", type=int, default=45, help="Number of frames")
    parser.add_argument("--steps", type=int, default=30, help="Inference steps")
    parser.add_argument("--num-layers", type=int, default=6, help="Layers on GPU (with offload)")
    parser.add_argument("--mode", choices=["full", "offload-only", "sparse-only", "baseline"],
                        default="full", help="Benchmark mode")
    parser.add_argument("--output", default="benchmark_full_results.json", help="Output file")

    args = parser.parse_args()

    print("=" * 60)
    print("FULL SYSTEM BENCHMARK")
    print("=" * 60)
    print(f"Mode: {args.mode}")
    print(f"  full        = SAP_CTCA + Component Offloading")
    print(f"  offload-only = Offloading only (full attention)")
    print(f"  sparse-only = SAP_CTCA only (no offloading)")
    print(f"  baseline    = No optimizations")
    print("-" * 60)
    print(f"Frames: {args.num_frames}, Steps: {args.steps}")
    if args.mode in ["full", "offload-only"]:
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

    # Print comparison table if multiple modes exist
    modes = {r['mode']: r for r in all_results}
    if len(modes) >= 2:
        print("\n" + "=" * 60)
        print("COMPARISON TABLE")
        print("=" * 60)
        print(f"{'Mode':<15} {'Memory (GB)':<15} {'Time (s)':<15}")
        print("-" * 60)
        for mode, result in sorted(modes.items()):
            print(f"{mode:<15} {result['peak_memory_nvidia_gb']:<15} {result['inference_time_seconds']:<15}")

        # Calculate savings if full and baseline exist
        if 'full' in modes and 'baseline' in modes:
            full = modes['full']
            baseline = modes['baseline']
            mem_saved = baseline['peak_memory_nvidia_gb'] - full['peak_memory_nvidia_gb']
            mem_pct = mem_saved / baseline['peak_memory_nvidia_gb'] * 100 if baseline['peak_memory_nvidia_gb'] > 0 else 0
            time_diff = full['inference_time_seconds'] - baseline['inference_time_seconds']
            time_pct = time_diff / baseline['inference_time_seconds'] * 100 if baseline['inference_time_seconds'] > 0 else 0

            print("-" * 60)
            print(f"Full vs Baseline:")
            print(f"  Memory Saved: {mem_saved:.1f} GB ({mem_pct:.1f}% reduction)")
            print(f"  Time Diff: {time_diff:+.1f}s ({time_pct:+.1f}%)")


if __name__ == "__main__":
    main()
