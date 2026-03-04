#!/usr/bin/env python3
"""
Aggregate and summarize all KV reuse experiment results.

Usage:
    python experiments/kv_reuse/analyze_results.py --result_root result/kv_reuse_paper

Produces:
    - Console tables for each experiment group
    - CSV files in result/kv_reuse_paper/summary/ for LaTeX import
    - Timing/memory summary with speedup ratios
"""
import argparse
import csv
import json
import os
import re
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
# 4. Timing and memory (from .run.json files)
# =========================================================================
def _parse_run_json_tag(result_root, run_json_path):
    """Extract experiment tag from a .run.json path.

    Convention: result_root/<tag>/<cfg_tag>/<pid>-<seed>.mp4.run.json
    Returns (tag, cfg_tag, pid, seed) or None if unparseable.
    """
    rel = os.path.relpath(run_json_path, result_root)
    # Strip the .mp4.run.json suffix to get <tag>/<cfg_tag>/<pid>-<seed>
    base = rel
    for suffix in (".run.json", ".mp4"):
        if base.endswith(suffix):
            base = base[: -len(suffix)]
    parts = base.split("/")
    if len(parts) < 2:
        return None
    tag = parts[0]
    cfg_tag = parts[1] if len(parts) >= 3 else ""
    filename = parts[-1]
    # Parse <pid>-<seed> from filename
    m = re.match(r"^(\d+)-(\d+)$", filename)
    if m:
        pid, seed = m.group(1), m.group(2)
    else:
        pid, seed = filename, ""
    return tag, cfg_tag, pid, seed


def summarize_timing_memory(result_root, summary_dir):
    """Aggregate .run.json files into timing/memory summary tables."""
    print("\n" + "=" * 72)
    print("TIMING & MEMORY (from .run.json)")
    print("=" * 72)

    run_files = sorted(glob(os.path.join(result_root, "**", "*.run.json"), recursive=True))
    if not run_files:
        print("  No .run.json files found.")
        return

    # Group by (tag, cfg_tag)
    groups = defaultdict(list)
    for rj in run_files:
        parsed = _parse_run_json_tag(result_root, rj)
        if parsed is None:
            continue
        tag, cfg_tag, pid, seed = parsed
        try:
            with open(rj, "r") as f:
                data = json.load(f)
        except (json.JSONDecodeError, IOError):
            continue
        data["_pid"] = pid
        data["_seed"] = seed
        groups[(tag, cfg_tag)].append(data)

    if not groups:
        print("  No parseable .run.json files found.")
        return

    # --- Per-group summary ---
    header = (
        f"{'Tag':<25} {'Config':<15} {'N':>3} "
        f"{'Time(s)':>12} {'PeakGPU(MB)':>14} "
        f"{'Pattern':<7} {'KVReuse':>7}"
    )
    print(header)
    print("-" * len(header))

    csv_rows = []
    for (tag, cfg_tag) in sorted(groups):
        entries = groups[(tag, cfg_tag)]
        times = [e["wall_clock_s"] for e in entries if "wall_clock_s" in e]
        gpus = [e["peak_gpu_mb"] for e in entries if "peak_gpu_mb" in e]
        patterns = set(e.get("pattern", "?") for e in entries)
        kv = any(e.get("video_k_reuse", False) for e in entries)

        t_mean, t_std = mean(times), std(times)
        g_mean, g_std = mean(gpus), std(gpus)
        pat = "/".join(sorted(patterns))

        print(
            f"{tag:<25} {cfg_tag:<15} {len(entries):>3} "
            f"{fmt(t_mean, 1)}±{fmt(t_std, 1):>5}s "
            f"{fmt(g_mean, 0)}±{fmt(g_std, 0):>5}MB "
            f"{pat:<7} {'yes' if kv else 'no':>7}"
        )

        csv_rows.append({
            "tag": tag,
            "cfg_tag": cfg_tag,
            "n": len(entries),
            "time_mean_s": round(t_mean, 2),
            "time_std_s": round(t_std, 2),
            "peak_gpu_mean_mb": round(g_mean, 0),
            "peak_gpu_std_mb": round(g_std, 0),
            "pattern": pat,
            "kv_reuse": kv,
        })

    csv_path = os.path.join(summary_dir, "timing_memory_summary.csv")
    if csv_rows:
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=csv_rows[0].keys())
            writer.writeheader()
            writer.writerows(csv_rows)
        print(f"\n  Saved to {csv_path}")

    # --- Speedup table: pair baseline vs +KV variant by (pid, seed, cfg_tag) ---
    _compute_speedups(groups, summary_dir)


def _compute_speedups(groups, summary_dir):
    """Compute pairwise speedup ratios between baseline and +KV variants."""
    # Define pairs: (baseline_tag, kv_tag, label)
    pairs = [
        ("sap", "sap_kv_reuse", "SAP+KV vs SAP"),
        ("sap", "sap_k_only", "SAP+Konly vs SAP"),
        ("svg", "svg_kv_reuse", "SVG+KV vs SVG"),
        ("dense", "dense_kv_reuse", "Dense+KV vs Dense"),
        ("sap_only", "sap_kv_reuse", "SAP+KV vs SAP (comp)"),
        ("svg_only", "svg_kv_reuse", "SVG+KV vs SVG (comp)"),
        ("scale_sap", "scale_sap_kv", "ScaleSAP+KV vs ScaleSAP"),
    ]

    print("\n" + "-" * 72)
    print("SPEEDUP RATIOS (baseline / variant)")
    print("-" * 72)

    csv_rows = []
    found_any = False

    for base_tag, kv_tag, label in pairs:
        # Collect all cfg_tags where both tags exist
        base_cfgs = {cfg for (t, cfg) in groups if t == base_tag}
        kv_cfgs = {cfg for (t, cfg) in groups if t == kv_tag}
        common_cfgs = sorted(base_cfgs & kv_cfgs)
        if not common_cfgs:
            continue

        found_any = True
        for cfg_tag in common_cfgs:
            base_entries = groups[(base_tag, cfg_tag)]
            kv_entries = groups[(kv_tag, cfg_tag)]

            # Index by (pid, seed) for pairing
            base_by_key = {(e["_pid"], e["_seed"]): e for e in base_entries}
            kv_by_key = {(e["_pid"], e["_seed"]): e for e in kv_entries}

            speedups = []
            mem_deltas = []
            for key in sorted(set(base_by_key) & set(kv_by_key)):
                b = base_by_key[key]
                k = kv_by_key[key]
                bt = b.get("wall_clock_s", 0)
                kt = k.get("wall_clock_s", 0)
                if kt > 0 and bt > 0:
                    speedups.append(bt / kt)
                bg = b.get("peak_gpu_mb", 0)
                kg = k.get("peak_gpu_mb", 0)
                if bg > 0:
                    mem_deltas.append(kg - bg)

            if not speedups:
                continue

            sp_mean, sp_std = mean(speedups), std(speedups)
            md_mean = mean(mem_deltas) if mem_deltas else 0

            # Also compute absolute time savings
            base_times = [base_by_key[k]["wall_clock_s"] for k in sorted(set(base_by_key) & set(kv_by_key))]
            kv_times = [kv_by_key[k]["wall_clock_s"] for k in sorted(set(base_by_key) & set(kv_by_key))]

            print(
                f"  {label:<30} [{cfg_tag}]  "
                f"speedup={fmt(sp_mean, 2)}x±{fmt(sp_std, 2)}  "
                f"base={fmt(mean(base_times), 1)}s  variant={fmt(mean(kv_times), 1)}s  "
                f"mem_delta={md_mean:+.0f}MB  n={len(speedups)}"
            )

            csv_rows.append({
                "label": label,
                "cfg_tag": cfg_tag,
                "n_pairs": len(speedups),
                "speedup_mean": round(sp_mean, 4),
                "speedup_std": round(sp_std, 4),
                "base_time_mean_s": round(mean(base_times), 2),
                "variant_time_mean_s": round(mean(kv_times), 2),
                "mem_delta_mean_mb": round(md_mean, 0),
            })

    if not found_any:
        print("  No matching baseline/variant pairs found for speedup calculation.")
        return

    csv_path = os.path.join(summary_dir, "speedup_summary.csv")
    if csv_rows:
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=csv_rows[0].keys())
            writer.writeheader()
            writer.writerows(csv_rows)
        print(f"\n  Saved to {csv_path}")


# =========================================================================
# 5. Speed-Quality Trade-off Analysis (ablation sweeps)
# =========================================================================
def analyze_speed_quality_tradeoff(result_root, quality_root, summary_dir):
    """For each ablation sweep, pair timing with quality to show trade-offs."""
    print("\n" + "=" * 72)
    print("SPEED-QUALITY TRADE-OFF (ablation sweeps)")
    print("=" * 72)

    tradeoff_dir = os.path.join(summary_dir, "tradeoff")
    os.makedirs(tradeoff_dir, exist_ok=True)

    # Collect timing from ablation .run.json files
    ablation_timing = {}  # abl_name -> list of wall_clock_s
    run_files = sorted(glob(os.path.join(result_root, "ablations", "**", "*.run.json"), recursive=True))
    for rj in run_files:
        try:
            with open(rj, "r") as f:
                data = json.load(f)
        except (json.JSONDecodeError, IOError):
            continue
        # Parse ablation name from path: ablations/<abl_name>/<cfg_tag>/<pid>-<seed>.mp4.run.json
        rel = os.path.relpath(rj, os.path.join(result_root, "ablations"))
        parts = rel.split("/")
        if len(parts) < 2:
            continue
        abl_name = parts[0]
        if abl_name not in ablation_timing:
            ablation_timing[abl_name] = []
        ablation_timing[abl_name].append(data.get("wall_clock_s", 0))

    # Collect SAP baseline timing
    sap_times = []
    for rj in sorted(glob(os.path.join(result_root, "sap", "**", "*.run.json"), recursive=True)):
        try:
            with open(rj, "r") as f:
                data = json.load(f)
            sap_times.append(data.get("wall_clock_s", 0))
        except (json.JSONDecodeError, IOError):
            continue
    sap_mean_time = mean(sap_times) if sap_times else 0

    # Collect quality from ablation quality files
    ablation_quality = {}  # abl_name -> {PSNR: [...], SSIM: [...], LPIPS: [...]}
    quality_files = sorted(glob(os.path.join(quality_root, "ablations", "**", "*.jsonl"), recursive=True))
    for qf in quality_files:
        rel = os.path.relpath(qf, os.path.join(quality_root, "ablations"))
        parts = rel.split("/")
        if not parts:
            continue
        # abl_name_vs_sap/cfg_tag.jsonl -> extract abl_name
        abl_dir = parts[0]
        abl_name = abl_dir.replace("_vs_sap", "")
        entries = load_jsonl(qf)
        if abl_name not in ablation_quality:
            ablation_quality[abl_name] = {"PSNR": [], "SSIM": [], "LPIPS": []}
        for e in entries:
            if "PSNR" in e:
                ablation_quality[abl_name]["PSNR"].append(e["PSNR"])
            if "SSIM" in e:
                ablation_quality[abl_name]["SSIM"].append(e["SSIM"])
            if "LPIPS" in e:
                ablation_quality[abl_name]["LPIPS"].append(e["LPIPS"])

    if not ablation_timing or not ablation_quality:
        print("  No ablation data found.")
        return

    # Define sweep groups for organized trade-off plots
    sweep_groups = {
        "Block Size": {"prefix": "block_size_", "param_extract": lambda n: int(n.replace("block_size_", ""))},
        "Warmup Steps": {"prefix": "warmup_", "param_extract": lambda n: int(n.replace("warmup_", ""))},
        "Refresh Interval": {"prefix": "interval_", "param_extract": lambda n: int(n.replace("interval_", ""))},
        "Max Cached Blocks": {"prefix": "max_blocks_", "param_extract": lambda n: int(n.replace("max_blocks_", ""))},
        "K-only vs KV": {"names": ["k_only", "kv_joint"]},
    }

    # Build trade-off CSV
    csv_rows = []
    for abl_name in sorted(set(ablation_timing) & set(ablation_quality)):
        times = ablation_timing[abl_name]
        q = ablation_quality[abl_name]
        speedup = sap_mean_time / mean(times) if mean(times) > 0 and sap_mean_time > 0 else 0
        row = {
            "ablation": abl_name,
            "n_runs": len(times),
            "time_mean_s": round(mean(times), 2),
            "time_std_s": round(std(times), 2),
            "speedup_vs_sap": round(speedup, 4),
            "PSNR": round(mean(q["PSNR"]), 2) if q["PSNR"] else None,
            "SSIM": round(mean(q["SSIM"]), 4) if q["SSIM"] else None,
            "LPIPS": round(mean(q["LPIPS"]), 4) if q["LPIPS"] else None,
        }
        csv_rows.append(row)
        print(f"  {abl_name:<25} speedup={speedup:.3f}x  PSNR={row['PSNR']}  SSIM={row['SSIM']}  LPIPS={row['LPIPS']}  time={row['time_mean_s']}s")

    csv_path = os.path.join(tradeoff_dir, "ablation_tradeoff.csv")
    if csv_rows:
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=csv_rows[0].keys())
            writer.writeheader()
            writer.writerows(csv_rows)
        print(f"\n  Saved to {csv_path}")

    # Generate trade-off plots per sweep group
    _plot_tradeoff_sweeps(csv_rows, sweep_groups, sap_mean_time, tradeoff_dir)


def _plot_tradeoff_sweeps(csv_rows, sweep_groups, sap_mean_time, tradeoff_dir):
    """Generate speed vs quality trade-off scatter plots for each ablation sweep."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("  matplotlib not available, skipping trade-off plots.")
        return

    rows_by_name = {r["ablation"]: r for r in csv_rows}

    for group_name, group_def in sweep_groups.items():
        # Find matching ablation names
        if "names" in group_def:
            names = [n for n in group_def["names"] if n in rows_by_name]
            labels = names
        else:
            prefix = group_def["prefix"]
            names = sorted([n for n in rows_by_name if n.startswith(prefix)],
                           key=group_def["param_extract"])
            labels = [str(group_def["param_extract"](n)) for n in names]

        if len(names) < 2:
            continue

        # Filter to names that have quality data
        valid_names = [n for n in names if rows_by_name[n]["PSNR"] is not None]
        if len(valid_names) < 2:
            continue

        valid_labels = [labels[names.index(n)] for n in valid_names]
        speedups = [rows_by_name[n]["speedup_vs_sap"] for n in valid_names]
        psnrs = [rows_by_name[n]["PSNR"] for n in valid_names]
        ssims = [rows_by_name[n]["SSIM"] for n in valid_names]
        lpips_vals = [rows_by_name[n]["LPIPS"] for n in valid_names]

        fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

        def _scatter(ax, ys, color, ylabel, title):
            ax.scatter(speedups, ys, c=color, s=80, zorder=5)
            for i, label in enumerate(valid_labels):
                ax.annotate(label, (speedups[i], ys[i]), textcoords="offset points",
                            xytext=(5, 5), fontsize=8)
            ax.axvline(x=1.0, color="gray", linestyle="--", alpha=0.5)
            ax.set_xlabel("Speedup vs SAP")
            ax.set_ylabel(ylabel)
            ax.set_title(f"{group_name}: {title}")
            ax.grid(True, alpha=0.3)

        _scatter(axes[0], psnrs, "steelblue", "PSNR (dB) vs SAP", "Speedup vs PSNR")
        _scatter(axes[1], ssims, "darkorange", "SSIM vs SAP", "Speedup vs SSIM")
        _scatter(axes[2], lpips_vals, "forestgreen", "LPIPS vs SAP (lower=better)", "Speedup vs LPIPS")

        plt.suptitle(f"Speed-Quality Trade-off: {group_name}", fontsize=13, fontweight="bold")
        plt.tight_layout()
        plot_name = f"tradeoff_{group_name.lower().replace(' ', '_')}.png"
        plot_path = os.path.join(tradeoff_dir, plot_name)
        plt.savefig(plot_path, dpi=150)
        plt.close()
        print(f"  Plot saved: {plot_path}")


# =========================================================================
# 6. Combined Trade-off Pareto (Exp 6: SAP + KV reuse joint sweep)
# =========================================================================
def analyze_combined_tradeoff(result_root, quality_root, summary_dir):
    """Analyze the combined SAP + KV reuse trade-off experiment (06_tradeoff.sh).
    Produces a Pareto frontier plot of speedup vs quality across all levels."""
    print("\n" + "=" * 72)
    print("COMBINED SPEED-QUALITY TRADE-OFF (Exp 6: Pareto frontier)")
    print("=" * 72)

    tradeoff_dir = os.path.join(summary_dir, "tradeoff")
    os.makedirs(tradeoff_dir, exist_ok=True)

    # Collect timing for each level
    level_timing = {}
    for rj in sorted(glob(os.path.join(result_root, "tradeoff", "**", "*.run.json"), recursive=True)):
        rel = os.path.relpath(rj, os.path.join(result_root, "tradeoff"))
        parts = rel.split("/")
        if len(parts) < 2:
            continue
        level = parts[0]
        try:
            with open(rj, "r") as f:
                data = json.load(f)
        except (json.JSONDecodeError, IOError):
            continue
        if level not in level_timing:
            level_timing[level] = []
        level_timing[level].append(data.get("wall_clock_s", 0))

    # Collect quality for each level (vs Dense)
    level_quality = {}
    for qf in sorted(glob(os.path.join(quality_root, "tradeoff", "**", "*.jsonl"), recursive=True)):
        rel = os.path.relpath(qf, os.path.join(quality_root, "tradeoff"))
        parts = rel.split("/")
        if not parts:
            continue
        level = parts[0].replace("_vs_dense", "")
        entries = load_jsonl(qf)
        if level not in level_quality:
            level_quality[level] = {"PSNR": [], "SSIM": [], "LPIPS": []}
        for e in entries:
            if "PSNR" in e:
                level_quality[level]["PSNR"].append(e["PSNR"])
            if "SSIM" in e:
                level_quality[level]["SSIM"].append(e["SSIM"])
            if "LPIPS" in e:
                level_quality[level]["LPIPS"].append(e["LPIPS"])

    if not level_timing:
        print("  No trade-off experiment data found (run 06_tradeoff.sh first).")
        return

    # Get Dense oracle time as baseline
    dense_times = level_timing.get("L0_dense", [])
    dense_mean_time = mean(dense_times) if dense_times else 0

    # Build summary table
    header = f"{'Level':<35} {'N':>3} {'Time(s)':>10} {'Speedup':>8} {'PSNR':>8} {'SSIM':>8} {'LPIPS':>8}"
    print(header)
    print("-" * len(header))

    csv_rows = []
    for level in sorted(level_timing.keys()):
        times = level_timing[level]
        q = level_quality.get(level, {"PSNR": [], "SSIM": [], "LPIPS": []})
        speedup = dense_mean_time / mean(times) if mean(times) > 0 and dense_mean_time > 0 else 0

        row = {
            "level": level,
            "n_runs": len(times),
            "time_mean_s": round(mean(times), 2),
            "time_std_s": round(std(times), 2),
            "speedup_vs_dense": round(speedup, 4),
            "PSNR": round(mean(q["PSNR"]), 2) if q["PSNR"] else None,
            "SSIM": round(mean(q["SSIM"]), 4) if q["SSIM"] else None,
            "LPIPS": round(mean(q["LPIPS"]), 4) if q["LPIPS"] else None,
        }
        csv_rows.append(row)
        psnr_str = f"{row['PSNR']:.2f}" if row["PSNR"] is not None else "N/A"
        ssim_str = f"{row['SSIM']:.4f}" if row["SSIM"] is not None else "N/A"
        lpips_str = f"{row['LPIPS']:.4f}" if row["LPIPS"] is not None else "N/A"
        print(f"  {level:<33} {len(times):>3} {mean(times):>9.1f}s {speedup:>7.2f}x {psnr_str:>8} {ssim_str:>8} {lpips_str:>8}")

    csv_path = os.path.join(tradeoff_dir, "combined_tradeoff.csv")
    if csv_rows:
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=csv_rows[0].keys())
            writer.writeheader()
            writer.writerows(csv_rows)
        print(f"\n  Saved to {csv_path}")

    # Pareto frontier plot
    _plot_pareto_frontier(csv_rows, tradeoff_dir)


def _plot_pareto_frontier(csv_rows, tradeoff_dir):
    """Generate a Pareto frontier scatter plot: speedup vs PSNR/SSIM/LPIPS."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("  matplotlib not available, skipping Pareto plot.")
        return

    # Filter rows with valid quality data (skip L0_dense — it's the reference)
    rows = [r for r in csv_rows if r["PSNR"] is not None and r["level"] != "L0_dense"]
    if len(rows) < 2:
        print("  Not enough data points for Pareto plot.")
        return

    levels = [r["level"] for r in rows]
    speedups = [r["speedup_vs_dense"] for r in rows]
    psnrs = [r["PSNR"] for r in rows]
    ssims = [r["SSIM"] for r in rows]
    lpips_vals = [r["LPIPS"] for r in rows]

    # Short labels
    short_labels = []
    for lv in levels:
        sl = lv.replace("_sap_kv_", "\nKV:").replace("_sap_", "\nSAP:").replace("L", "L")
        short_labels.append(lv.split("_", 1)[0])  # Just L1, L2, etc.

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    # Speedup vs PSNR
    axes[0].scatter(speedups, psnrs, c="steelblue", s=100, zorder=5, edgecolors="navy")
    for i, label in enumerate(short_labels):
        axes[0].annotate(label, (speedups[i], psnrs[i]), textcoords="offset points",
                         xytext=(6, 6), fontsize=9, fontweight="bold")
    axes[0].set_xlabel("Speedup vs Dense", fontsize=11)
    axes[0].set_ylabel("PSNR (dB) vs Dense", fontsize=11)
    axes[0].set_title("Speedup vs PSNR")
    axes[0].grid(True, alpha=0.3)

    # Speedup vs SSIM
    axes[1].scatter(speedups, ssims, c="darkorange", s=100, zorder=5, edgecolors="saddlebrown")
    for i, label in enumerate(short_labels):
        axes[1].annotate(label, (speedups[i], ssims[i]), textcoords="offset points",
                         xytext=(6, 6), fontsize=9, fontweight="bold")
    axes[1].set_xlabel("Speedup vs Dense", fontsize=11)
    axes[1].set_ylabel("SSIM vs Dense", fontsize=11)
    axes[1].set_title("Speedup vs SSIM")
    axes[1].grid(True, alpha=0.3)

    # Speedup vs LPIPS
    axes[2].scatter(speedups, lpips_vals, c="forestgreen", s=100, zorder=5, edgecolors="darkgreen")
    for i, label in enumerate(short_labels):
        axes[2].annotate(label, (speedups[i], lpips_vals[i]), textcoords="offset points",
                         xytext=(6, 6), fontsize=9, fontweight="bold")
    axes[2].set_xlabel("Speedup vs Dense", fontsize=11)
    axes[2].set_ylabel("LPIPS vs Dense (lower=better)", fontsize=11)
    axes[2].set_title("Speedup vs LPIPS")
    axes[2].grid(True, alpha=0.3)

    # Add legend with level descriptions
    level_desc = {
        "L1": "SAP only",
        "L2": "SAP + KV conserv.",
        "L3": "SAP + KV default",
        "L4": "SAP + KV aggr.",
        "L5": "SAP aggr. + KV def.",
        "L6": "SAP aggr. + KV aggr.",
        "L7": "SAP v.aggr. + KV aggr.",
        "L8": "SAP extreme + KV extreme",
    }
    legend_text = "\n".join(f"{k}: {v}" for k, v in level_desc.items())
    fig.text(0.02, 0.02, legend_text, fontsize=7, fontfamily="monospace",
             verticalalignment="bottom", bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))

    plt.suptitle("Speed-Quality Pareto Frontier (SAP + KV Reuse Combined)", fontsize=13, fontweight="bold")
    plt.tight_layout(rect=[0.0, 0.15, 1.0, 0.95])
    plot_path = os.path.join(tradeoff_dir, "pareto_frontier.png")
    plt.savefig(plot_path, dpi=150)
    plt.close()
    print(f"  Pareto plot saved: {plot_path}")


# =========================================================================
# 7. GPU Memory Trajectory (from .run.json or .gpu_memory.csv)
# =========================================================================
def summarize_gpu_memory_trajectories(result_root, summary_dir):
    """Aggregate GPU memory trajectories and generate per-step CSV + plots."""
    print("\n" + "=" * 72)
    print("GPU MEMORY TRAJECTORIES (per-step)")
    print("=" * 72)

    # Collect trajectories from .run.json files
    run_files = sorted(glob(os.path.join(result_root, "**", "*.run.json"), recursive=True))
    if not run_files:
        print("  No .run.json files found.")
        return

    # Group trajectories by (tag, cfg_tag)
    trajectories = defaultdict(list)
    for rj in run_files:
        parsed = _parse_run_json_tag(result_root, rj)
        if parsed is None:
            continue
        tag, cfg_tag, pid, seed = parsed
        try:
            with open(rj, "r") as f:
                data = json.load(f)
        except (json.JSONDecodeError, IOError):
            continue
        traj = data.get("gpu_memory_trajectory")
        if traj:
            trajectories[(tag, cfg_tag)].append({
                "pid": pid, "seed": seed, "trajectory": traj
            })

    if not trajectories:
        print("  No GPU memory trajectories found in .run.json files.")
        return

    # Export aggregated per-step CSVs (averaged across runs)
    gpu_summary_dir = os.path.join(summary_dir, "gpu_memory")
    os.makedirs(gpu_summary_dir, exist_ok=True)

    summary_rows = []
    for (tag, cfg_tag) in sorted(trajectories):
        runs = trajectories[(tag, cfg_tag)]
        # Aggregate by step
        step_data = defaultdict(lambda: {"allocated": [], "reserved": [], "peak": [], "elapsed": []})
        for run in runs:
            for entry in run["trajectory"]:
                step = entry["step"]
                step_data[step]["allocated"].append(entry["allocated_mb"])
                step_data[step]["reserved"].append(entry["reserved_mb"])
                step_data[step]["peak"].append(entry["peak_mb"])
                step_data[step]["elapsed"].append(entry["elapsed_s"])

        csv_name = f"gpu_trajectory_{tag}_{cfg_tag}.csv"
        csv_path = os.path.join(gpu_summary_dir, csv_name)
        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["step", "n_runs", "avg_allocated_mb", "std_allocated_mb",
                             "avg_reserved_mb", "avg_peak_mb", "avg_elapsed_s"])
            for step in sorted(step_data.keys()):
                d = step_data[step]
                writer.writerow([
                    step, len(d["allocated"]),
                    round(mean(d["allocated"]), 1), round(std(d["allocated"]), 1),
                    round(mean(d["reserved"]), 1),
                    round(mean(d["peak"]), 1),
                    round(mean(d["elapsed"]), 2),
                ])

        n_steps = len(step_data)
        all_alloc = [v for d in step_data.values() for v in d["allocated"]]
        avg_alloc = mean(all_alloc)
        max_alloc = max(all_alloc) if all_alloc else 0
        min_alloc = min(all_alloc) if all_alloc else 0

        print(f"  {tag}/{cfg_tag}: {len(runs)} runs, {n_steps} steps, "
              f"avg={avg_alloc:.0f}MB, min={min_alloc:.0f}MB, max={max_alloc:.0f}MB -> {csv_name}")

        summary_rows.append({
            "tag": tag, "cfg_tag": cfg_tag, "n_runs": len(runs), "n_steps": n_steps,
            "avg_allocated_mb": round(avg_alloc, 1),
            "min_allocated_mb": round(min_alloc, 1),
            "max_allocated_mb": round(max_alloc, 1),
        })

    # Write summary CSV
    summary_csv = os.path.join(summary_dir, "gpu_memory_summary.csv")
    if summary_rows:
        with open(summary_csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=summary_rows[0].keys())
            writer.writeheader()
            writer.writerows(summary_rows)
        print(f"\n  Summary saved to {summary_csv}")

    # Generate comparison plots
    _plot_gpu_memory_comparison(trajectories, gpu_summary_dir)


def _plot_gpu_memory_comparison(trajectories, gpu_summary_dir):
    """Generate GPU memory comparison plots for matching cfg_tags."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("  matplotlib not available, skipping GPU memory plots.")
        return

    # Group by cfg_tag for comparison plots
    by_cfg = defaultdict(dict)
    for (tag, cfg_tag), runs in trajectories.items():
        by_cfg[cfg_tag][tag] = runs

    for cfg_tag, tag_runs in sorted(by_cfg.items()):
        if len(tag_runs) < 2:
            continue  # Need at least 2 methods to compare

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

        for tag, runs in sorted(tag_runs.items()):
            # Average trajectory across runs
            step_alloc = defaultdict(list)
            step_elapsed = defaultdict(list)
            for run in runs:
                for entry in run["trajectory"]:
                    step_alloc[entry["step"]].append(entry["allocated_mb"])
                    step_elapsed[entry["step"]].append(entry["elapsed_s"])

            steps = sorted(step_alloc.keys())
            avg_alloc = [mean(step_alloc[s]) for s in steps]
            avg_elapsed = [mean(step_elapsed[s]) for s in steps]

            label = tag.replace("_", " ").title()
            ax1.plot(steps, avg_alloc, label=label, marker=".", markersize=3)
            ax2.plot(avg_elapsed, avg_alloc, label=label, marker=".", markersize=3)

        ax1.set_xlabel("Denoising Step")
        ax1.set_ylabel("GPU Memory Allocated (MB)")
        ax1.set_title(f"GPU Memory vs Step ({cfg_tag})")
        ax1.legend(fontsize=8)
        ax1.grid(True, alpha=0.3)

        ax2.set_xlabel("Elapsed Time (s)")
        ax2.set_ylabel("GPU Memory Allocated (MB)")
        ax2.set_title(f"GPU Memory vs Time ({cfg_tag})")
        ax2.legend(fontsize=8)
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        plot_path = os.path.join(gpu_summary_dir, f"gpu_memory_comparison_{cfg_tag}.png")
        plt.savefig(plot_path, dpi=150)
        plt.close()
        print(f"  Plot saved: {plot_path}")


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

    summarize_timing_memory(result_root, summary_dir)
    summarize_quality(quality_root, summary_dir)
    summarize_kv_metrics(metrics_root, summary_dir)
    export_per_step_trajectory(metrics_root, summary_dir)
    analyze_speed_quality_tradeoff(result_root, quality_root, summary_dir)
    analyze_combined_tradeoff(result_root, quality_root, summary_dir)
    summarize_gpu_memory_trajectories(result_root, summary_dir)

    print("\n" + "=" * 72)
    print("DONE. Summary CSVs written to:", summary_dir)
    print("=" * 72)


if __name__ == "__main__":
    main()
