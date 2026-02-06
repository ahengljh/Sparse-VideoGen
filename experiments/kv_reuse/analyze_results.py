#!/usr/bin/env python3
"""
Aggregate and summarize all KV reuse experiment results.

Usage:
    python experiments/kv_reuse/analyze_results.py --result_root result/kv_reuse_paper

Produces:
    - Console tables for each experiment group
    - CSV files in result/kv_reuse_paper/summary/ for LaTeX import
"""
import argparse
import csv
import json
import os
import sys
from collections import defaultdict
from glob import glob
from pathlib import Path


def load_jsonl(path):
    entries = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                entries.append(json.loads(line))
    return entries


def mean(vals):
    return sum(vals) / len(vals) if vals else 0.0


def std(vals):
    m = mean(vals)
    return (sum((v - m) ** 2 for v in vals) / len(vals)) ** 0.5 if len(vals) > 1 else 0.0


def fmt(val, prec=4):
    return f"{val:.{prec}f}"


# =========================================================================
# 1. Quality metrics (PSNR, SSIM, LPIPS, MSE)
# =========================================================================
def summarize_quality(quality_root, summary_dir):
    """Aggregate quality JSONL files into per-config summaries."""
    print("\n" + "=" * 72)
    print("QUALITY METRICS (vs Dense baseline)")
    print("=" * 72)

    results = {}
    for jsonl_path in sorted(glob(os.path.join(quality_root, "**", "*.jsonl"), recursive=True)):
        rel = os.path.relpath(jsonl_path, quality_root)
        config_name = rel.replace(".jsonl", "").replace("/", "__")
        entries = load_jsonl(jsonl_path)
        if not entries:
            continue

        psnr = [e["PSNR"] for e in entries if "PSNR" in e]
        ssim = [e["SSIM"] for e in entries if "SSIM" in e]
        lpips_vals = [e["LPIPS"] for e in entries if "LPIPS" in e]
        mse = [e["MSE"] for e in entries if "MSE" in e]

        results[config_name] = {
            "n": len(entries),
            "PSNR": mean(psnr),
            "PSNR_std": std(psnr),
            "SSIM": mean(ssim),
            "SSIM_std": std(ssim),
            "LPIPS": mean(lpips_vals),
            "LPIPS_std": std(lpips_vals),
            "MSE": mean(mse),
            "MSE_std": std(mse),
        }

    if not results:
        print("  No quality results found.")
        return

    # Print table
    header = f"{'Config':<45} {'N':>3} {'PSNR':>10} {'SSIM':>10} {'LPIPS':>10} {'MSE':>12}"
    print(header)
    print("-" * len(header))
    for config_name in sorted(results):
        r = results[config_name]
        print(
            f"{config_name:<45} {r['n']:>3} "
            f"{fmt(r['PSNR'], 2)}±{fmt(r['PSNR_std'], 2):>5} "
            f"{fmt(r['SSIM'], 4)}±{fmt(r['SSIM_std'], 4):>6} "
            f"{fmt(r['LPIPS'], 4)}±{fmt(r['LPIPS_std'], 4):>6} "
            f"{fmt(r['MSE'], 6)}±{fmt(r['MSE_std'], 6):>8}"
        )

    # Write CSV
    csv_path = os.path.join(summary_dir, "quality_summary.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["config", "n", "PSNR", "PSNR_std", "SSIM", "SSIM_std", "LPIPS", "LPIPS_std", "MSE", "MSE_std"])
        for config_name in sorted(results):
            r = results[config_name]
            writer.writerow([config_name, r["n"], r["PSNR"], r["PSNR_std"], r["SSIM"], r["SSIM_std"], r["LPIPS"], r["LPIPS_std"], r["MSE"], r["MSE_std"]])
    print(f"\n  Saved to {csv_path}")


# =========================================================================
# 2. KV reuse metrics (hit rates, reuse rates, mode distribution)
# =========================================================================
def summarize_kv_metrics(metrics_root, summary_dir):
    """Aggregate per-step KV reuse JSONL metrics."""
    print("\n" + "=" * 72)
    print("KV REUSE METRICS")
    print("=" * 72)

    results = {}
    for jsonl_path in sorted(glob(os.path.join(metrics_root, "**", "*.jsonl"), recursive=True)):
        rel = os.path.relpath(jsonl_path, metrics_root)
        # Group by config (directory), aggregate across prompts/seeds
        config_name = str(Path(rel).parent)
        if config_name not in results:
            results[config_name] = {
                "files": 0,
                "entries": 0,
                "total_tokens": 0,
                "reused_tokens": 0,
                "computed_tokens": 0,
                "hits": 0,
                "misses": 0,
                "full_k_count": 0,
                "modes": defaultdict(int),
                "change_ratios": [],
            }

        entries = load_jsonl(jsonl_path)
        r = results[config_name]
        r["files"] += 1
        r["entries"] += len(entries)

        for e in entries:
            r["total_tokens"] += e.get("total_tokens", 0)
            r["reused_tokens"] += e.get("reused_tokens", 0)
            r["computed_tokens"] += e.get("computed_tokens", 0)
            r["hits"] += e.get("hits", 0)
            r["misses"] += e.get("misses", 0)
            if e.get("full_k"):
                r["full_k_count"] += 1
            mode = e.get("mode", "unknown")
            r["modes"][mode] += 1
            cr = e.get("change_ratio")
            if cr is not None:
                r["change_ratios"].append(cr)

    if not results:
        print("  No KV reuse metrics found.")
        return

    header = f"{'Config':<40} {'Files':>5} {'Entries':>7} {'TokenReuse%':>11} {'HitRate%':>9} {'FullK%':>7} {'PartialReuse':>12}"
    print(header)
    print("-" * len(header))

    csv_rows = []
    for config_name in sorted(results):
        r = results[config_name]
        total_ops = r["hits"] + r["misses"]
        token_reuse = (r["reused_tokens"] / r["total_tokens"] * 100) if r["total_tokens"] > 0 else 0
        hit_rate = (r["hits"] / total_ops * 100) if total_ops > 0 else 0
        full_k_pct = (r["full_k_count"] / r["entries"] * 100) if r["entries"] > 0 else 0
        partial = r["modes"].get("partial_reuse", 0)

        print(
            f"{config_name:<40} {r['files']:>5} {r['entries']:>7} "
            f"{token_reuse:>10.1f}% {hit_rate:>8.1f}% {full_k_pct:>6.1f}% {partial:>12}"
        )
        csv_rows.append({
            "config": config_name,
            "files": r["files"],
            "entries": r["entries"],
            "total_tokens": r["total_tokens"],
            "reused_tokens": r["reused_tokens"],
            "token_reuse_pct": round(token_reuse, 2),
            "hits": r["hits"],
            "misses": r["misses"],
            "hit_rate_pct": round(hit_rate, 2),
            "full_k_pct": round(full_k_pct, 2),
            "partial_reuse_count": partial,
            "modes": json.dumps(dict(r["modes"]), sort_keys=True),
        })

    # Mode distribution detail
    print("\nMode Distribution:")
    for config_name in sorted(results):
        r = results[config_name]
        total = r["entries"] if r["entries"] > 0 else 1
        mode_str = ", ".join(f"{m}={c}({c/total*100:.0f}%)" for m, c in sorted(r["modes"].items()))
        print(f"  {config_name}: {mode_str}")

    csv_path = os.path.join(summary_dir, "kv_reuse_metrics_summary.csv")
    if csv_rows:
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=csv_rows[0].keys())
            writer.writeheader()
            writer.writerows(csv_rows)
        print(f"\n  Saved to {csv_path}")


# =========================================================================
# 3. Per-step reuse trajectory (for figures)
# =========================================================================
def export_per_step_trajectory(metrics_root, summary_dir):
    """Export per-step reuse data averaged across prompts/seeds for plotting."""
    print("\n" + "=" * 72)
    print("PER-STEP TRAJECTORIES (for figures)")
    print("=" * 72)

    for jsonl_dir in sorted(glob(os.path.join(metrics_root, "**", ""), recursive=True)):
        jsonl_files = glob(os.path.join(jsonl_dir, "*.jsonl"))
        if not jsonl_files:
            continue

        config_name = os.path.relpath(jsonl_dir, metrics_root).rstrip("/")

        # Collect per-step data across all files
        step_data = defaultdict(lambda: {"reused": [], "total": [], "mode": []})
        for jsonl_path in jsonl_files:
            entries = load_jsonl(jsonl_path)
            for e in entries:
                step = e.get("step", -1)
                step_data[step]["reused"].append(e.get("reused_tokens", 0))
                step_data[step]["total"].append(e.get("total_tokens", 0))
                step_data[step]["mode"].append(e.get("mode", "unknown"))

        if not step_data:
            continue

        csv_path = os.path.join(summary_dir, f"trajectory_{config_name.replace('/', '_')}.csv")
        os.makedirs(os.path.dirname(csv_path), exist_ok=True)
        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["step", "avg_reuse_pct", "n_samples", "dominant_mode"])
            for step in sorted(step_data.keys()):
                d = step_data[step]
                total = sum(d["total"])
                reused = sum(d["reused"])
                pct = (reused / total * 100) if total > 0 else 0
                # Dominant mode
                mode_counts = defaultdict(int)
                for m in d["mode"]:
                    mode_counts[m] += 1
                dominant = max(mode_counts, key=mode_counts.get)
                writer.writerow([step, round(pct, 2), len(d["total"]), dominant])

        print(f"  {config_name}: {len(step_data)} steps -> {csv_path}")


# =========================================================================
# Main
# =========================================================================
def main():
    parser = argparse.ArgumentParser(description="Analyze KV reuse experiment results")
    parser.add_argument("--result_root", type=str, default="result/kv_reuse_paper")
    args = parser.parse_args()

    result_root = args.result_root
    quality_root = os.path.join(result_root, "quality")
    metrics_root = os.path.join(result_root, "metrics")
    summary_dir = os.path.join(result_root, "summary")
    os.makedirs(summary_dir, exist_ok=True)

    print(f"Result root: {result_root}")
    print(f"Quality dir: {quality_root}")
    print(f"Metrics dir: {metrics_root}")
    print(f"Summary dir: {summary_dir}")

    summarize_quality(quality_root, summary_dir)
    summarize_kv_metrics(metrics_root, summary_dir)
    export_per_step_trajectory(metrics_root, summary_dir)

    print("\n" + "=" * 72)
    print("DONE. Summary CSVs written to:", summary_dir)
    print("=" * 72)


if __name__ == "__main__":
    main()
