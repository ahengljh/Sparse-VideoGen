#!/usr/bin/env python3
"""
Analyze and visualize CTAA experiment results.

Creates:
- Performance comparison tables
- Quality vs speedup trade-off plots
- Parameter sensitivity analysis

Usage:
    python experiments/analyze_results.py --results_dir experiment_results/parameter_sweep_xxx
"""

import argparse
import json
import os
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

# Optional imports for visualization
try:
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    print("Warning: matplotlib not available. Install with: pip install matplotlib")

try:
    import pandas as pd
    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False
    print("Warning: pandas not available. Tables will be printed in basic format.")


def load_suite_results(results_dir: str) -> Dict:
    """Load suite summary from experiment results directory."""
    summary_path = Path(results_dir) / "suite_summary.json"
    if not summary_path.exists():
        raise FileNotFoundError(f"Suite summary not found: {summary_path}")

    with open(summary_path) as f:
        return json.load(f)


def load_quality_results(results_dir: str) -> Optional[Dict]:
    """Load quality evaluation results if available."""
    quality_path = Path(results_dir) / "quality_evaluation.json"
    if quality_path.exists():
        with open(quality_path) as f:
            return json.load(f)
    return None


def extract_metrics(results: List[Dict]) -> Dict[str, Dict]:
    """
    Extract and organize metrics from experiment results.

    Returns:
        Dict mapping config_name -> aggregated metrics
    """
    config_metrics = {}

    for result in results:
        if 'error' in result:
            continue

        config = result['config']
        config_name = config['name']

        if config_name not in config_metrics:
            config_metrics[config_name] = {
                'config': config,
                'generation_times': [],
                'peak_memory_gb': [],
                'ctaa_full_ratio': [],
                'ctaa_centroid_ratio': [],
                'ctaa_skip_ratio': [],
                'ctca_speedup': [],
            }

        metrics = config_metrics[config_name]
        metrics['generation_times'].append(result.get('generation_time', 0))

        if 'gpu_memory' in result:
            metrics['peak_memory_gb'].append(result['gpu_memory'].get('max_allocated_gb', 0))

        if 'ctaa_stats' in result and result['ctaa_stats'].get('total_blocks', 0) > 0:
            stats = result['ctaa_stats']
            metrics['ctaa_full_ratio'].append(stats.get('full_ratio', 0))
            metrics['ctaa_centroid_ratio'].append(stats.get('centroid_ratio', 0))
            metrics['ctaa_skip_ratio'].append(stats.get('skip_ratio', 0))

        if 'ctca_stats' in result:
            # Estimate CTCA speedup from reuse ratio
            stats = result['ctca_stats']
            total = stats.get('total_calls', 1)
            full = stats.get('full_recluster', total)
            if full > 0:
                metrics['ctca_speedup'].append(total / full)

    return config_metrics


def compute_theoretical_speedup(full_ratio: float, centroid_ratio: float, skip_ratio: float,
                                centroid_cost: float = 0.01) -> float:
    """
    Compute theoretical attention speedup from CTAA ratios.

    Args:
        full_ratio: Fraction of blocks with full attention
        centroid_ratio: Fraction of blocks with centroid attention
        skip_ratio: Fraction of blocks skipped
        centroid_cost: Relative cost of centroid attention (default: 1% of full)

    Returns:
        Theoretical speedup factor
    """
    effective_cost = full_ratio + centroid_ratio * centroid_cost + skip_ratio * 0
    if effective_cost > 0:
        return 1.0 / effective_cost
    return 1.0


def print_performance_table(config_metrics: Dict[str, Dict]):
    """Print performance comparison table."""
    print("\n" + "=" * 100)
    print("PERFORMANCE COMPARISON TABLE")
    print("=" * 100)

    headers = ["Config", "p_full", "p_total", "Time (s)", "Memory (GB)",
               "Full %", "Centroid %", "Skip %", "Theo. Speedup"]
    col_widths = [20, 8, 8, 12, 12, 10, 12, 10, 14]

    # Print header
    header_str = ""
    for h, w in zip(headers, col_widths):
        header_str += f"{h:<{w}}"
    print(header_str)
    print("-" * 100)

    # Sort by config name
    for config_name in sorted(config_metrics.keys()):
        m = config_metrics[config_name]
        config = m['config']

        time_mean = np.mean(m['generation_times']) if m['generation_times'] else 0
        time_std = np.std(m['generation_times']) if len(m['generation_times']) > 1 else 0
        mem_mean = np.mean(m['peak_memory_gb']) if m['peak_memory_gb'] else 0

        full_mean = np.mean(m['ctaa_full_ratio']) * 100 if m['ctaa_full_ratio'] else 100
        centroid_mean = np.mean(m['ctaa_centroid_ratio']) * 100 if m['ctaa_centroid_ratio'] else 0
        skip_mean = np.mean(m['ctaa_skip_ratio']) * 100 if m['ctaa_skip_ratio'] else 0

        theo_speedup = compute_theoretical_speedup(
            full_mean / 100, centroid_mean / 100, skip_mean / 100
        )

        time_str = f"{time_mean:.1f}" + (f" +/- {time_std:.1f}" if time_std > 0 else "")

        row = [
            config_name[:18],
            f"{config['ctaa_p_full']:.2f}",
            f"{config['ctaa_p_total']:.2f}",
            time_str,
            f"{mem_mean:.2f}",
            f"{full_mean:.1f}%",
            f"{centroid_mean:.1f}%",
            f"{skip_mean:.1f}%",
            f"{theo_speedup:.2f}x",
        ]

        row_str = ""
        for val, w in zip(row, col_widths):
            row_str += f"{val:<{w}}"
        print(row_str)

    print("=" * 100)


def print_quality_table(quality_results: Dict):
    """Print quality comparison table."""
    if not quality_results or 'evaluations' not in quality_results:
        print("No quality results available")
        return

    evaluations = quality_results['evaluations']

    print("\n" + "=" * 80)
    print("QUALITY COMPARISON TABLE (vs Baseline)")
    print("=" * 80)

    headers = ["Config", "PSNR (dB)", "SSIM", "LPIPS"]
    col_widths = [25, 20, 20, 20]

    header_str = ""
    for h, w in zip(headers, col_widths):
        header_str += f"{h:<{w}}"
    print(header_str)
    print("-" * 80)

    # Group by config
    config_quality = {}
    for eval_result in evaluations:
        config = eval_result.get('config_name', 'unknown')
        if config not in config_quality:
            config_quality[config] = {'psnr': [], 'ssim': [], 'lpips': []}

        config_quality[config]['psnr'].append(eval_result['metrics']['psnr']['mean'])
        if 'ssim' in eval_result['metrics']:
            config_quality[config]['ssim'].append(eval_result['metrics']['ssim']['mean'])
        if 'lpips' in eval_result['metrics']:
            config_quality[config]['lpips'].append(eval_result['metrics']['lpips']['mean'])

    for config_name in sorted(config_quality.keys()):
        q = config_quality[config_name]

        psnr_str = f"{np.mean(q['psnr']):.2f} +/- {np.std(q['psnr']):.2f}" if q['psnr'] else "N/A"
        ssim_str = f"{np.mean(q['ssim']):.4f} +/- {np.std(q['ssim']):.4f}" if q['ssim'] else "N/A"
        lpips_str = f"{np.mean(q['lpips']):.4f} +/- {np.std(q['lpips']):.4f}" if q['lpips'] else "N/A"

        row = [config_name[:23], psnr_str, ssim_str, lpips_str]
        row_str = ""
        for val, w in zip(row, col_widths):
            row_str += f"{val:<{w}}"
        print(row_str)

    print("=" * 80)
    print("Reference: PSNR > 30dB = good, SSIM > 0.95 = excellent, LPIPS < 0.1 = very similar")
    print("=" * 80)


def plot_speedup_vs_quality(config_metrics: Dict, quality_results: Optional[Dict],
                            output_path: str = "speedup_vs_quality.png"):
    """Create speedup vs quality trade-off plot."""
    if not HAS_MATPLOTLIB:
        print("Matplotlib required for plotting")
        return

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Extract data
    configs = []
    speedups = []
    psnr_values = []
    ssim_values = []
    lpips_values = []

    for config_name, m in config_metrics.items():
        if not m['ctaa_full_ratio']:
            continue

        full_mean = np.mean(m['ctaa_full_ratio'])
        centroid_mean = np.mean(m['ctaa_centroid_ratio'])
        skip_mean = np.mean(m['ctaa_skip_ratio'])
        speedup = compute_theoretical_speedup(full_mean, centroid_mean, skip_mean)

        configs.append(config_name)
        speedups.append(speedup)

    # Get quality metrics if available
    if quality_results and 'evaluations' in quality_results:
        config_quality = {}
        for eval_result in quality_results['evaluations']:
            config = eval_result.get('config_name', 'unknown')
            if config not in config_quality:
                config_quality[config] = {'psnr': [], 'ssim': [], 'lpips': []}
            config_quality[config]['psnr'].append(eval_result['metrics']['psnr']['mean'])
            if 'ssim' in eval_result['metrics']:
                config_quality[config]['ssim'].append(eval_result['metrics']['ssim']['mean'])
            if 'lpips' in eval_result['metrics']:
                config_quality[config]['lpips'].append(eval_result['metrics']['lpips']['mean'])

        for config_name in configs:
            if config_name in config_quality:
                psnr_values.append(np.mean(config_quality[config_name]['psnr']))
                ssim_values.append(np.mean(config_quality[config_name]['ssim']) if config_quality[config_name]['ssim'] else None)
                lpips_values.append(np.mean(config_quality[config_name]['lpips']) if config_quality[config_name]['lpips'] else None)
            else:
                psnr_values.append(None)
                ssim_values.append(None)
                lpips_values.append(None)

    # Plot 1: Speedup vs PSNR
    ax1 = axes[0]
    valid_idx = [i for i, p in enumerate(psnr_values) if p is not None]
    if valid_idx:
        x = [speedups[i] for i in valid_idx]
        y = [psnr_values[i] for i in valid_idx]
        labels = [configs[i] for i in valid_idx]

        ax1.scatter(x, y, s=100, c='blue', alpha=0.7)
        for i, label in enumerate(labels):
            ax1.annotate(label, (x[i], y[i]), fontsize=8, ha='center', va='bottom')

    ax1.set_xlabel('Theoretical Speedup')
    ax1.set_ylabel('PSNR (dB)')
    ax1.set_title('Speedup vs PSNR')
    ax1.grid(True, alpha=0.3)

    # Plot 2: Speedup vs SSIM
    ax2 = axes[1]
    valid_idx = [i for i, s in enumerate(ssim_values) if s is not None]
    if valid_idx:
        x = [speedups[i] for i in valid_idx]
        y = [ssim_values[i] for i in valid_idx]

        ax2.scatter(x, y, s=100, c='green', alpha=0.7)

    ax2.set_xlabel('Theoretical Speedup')
    ax2.set_ylabel('SSIM')
    ax2.set_title('Speedup vs SSIM')
    ax2.grid(True, alpha=0.3)

    # Plot 3: Speedup vs LPIPS
    ax3 = axes[2]
    valid_idx = [i for i, l in enumerate(lpips_values) if l is not None]
    if valid_idx:
        x = [speedups[i] for i in valid_idx]
        y = [lpips_values[i] for i in valid_idx]

        ax3.scatter(x, y, s=100, c='red', alpha=0.7)

    ax3.set_xlabel('Theoretical Speedup')
    ax3.set_ylabel('LPIPS (lower is better)')
    ax3.set_title('Speedup vs LPIPS')
    ax3.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Plot saved to: {output_path}")


def plot_parameter_sensitivity(config_metrics: Dict, output_path: str = "parameter_sensitivity.png"):
    """Create parameter sensitivity heatmap."""
    if not HAS_MATPLOTLIB:
        print("Matplotlib required for plotting")
        return

    # Extract p_full, p_total, and speedup
    p_full_values = sorted(set(m['config']['ctaa_p_full'] for m in config_metrics.values()))
    p_total_values = sorted(set(m['config']['ctaa_p_total'] for m in config_metrics.values()))

    # Create speedup matrix
    speedup_matrix = np.zeros((len(p_full_values), len(p_total_values)))
    speedup_matrix[:] = np.nan

    for config_name, m in config_metrics.items():
        if not m['ctaa_full_ratio']:
            continue

        p_full = m['config']['ctaa_p_full']
        p_total = m['config']['ctaa_p_total']

        full_mean = np.mean(m['ctaa_full_ratio'])
        centroid_mean = np.mean(m['ctaa_centroid_ratio'])
        skip_mean = np.mean(m['ctaa_skip_ratio'])
        speedup = compute_theoretical_speedup(full_mean, centroid_mean, skip_mean)

        i = p_full_values.index(p_full)
        j = p_total_values.index(p_total)
        speedup_matrix[i, j] = speedup

    fig, ax = plt.subplots(figsize=(10, 8))

    im = ax.imshow(speedup_matrix, cmap='YlOrRd', aspect='auto')
    ax.set_xticks(range(len(p_total_values)))
    ax.set_xticklabels([f"{p:.2f}" for p in p_total_values])
    ax.set_yticks(range(len(p_full_values)))
    ax.set_yticklabels([f"{p:.2f}" for p in p_full_values])

    ax.set_xlabel('p_total')
    ax.set_ylabel('p_full')
    ax.set_title('Theoretical Speedup vs CTAA Parameters')

    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Speedup')

    # Add text annotations
    for i in range(len(p_full_values)):
        for j in range(len(p_total_values)):
            if not np.isnan(speedup_matrix[i, j]):
                ax.text(j, i, f"{speedup_matrix[i, j]:.1f}x",
                       ha='center', va='center', fontsize=10)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Plot saved to: {output_path}")


def plot_attention_distribution(config_metrics: Dict, output_path: str = "attention_distribution.png"):
    """Create stacked bar chart of attention type distribution."""
    if not HAS_MATPLOTLIB:
        print("Matplotlib required for plotting")
        return

    configs = []
    full_ratios = []
    centroid_ratios = []
    skip_ratios = []

    for config_name in sorted(config_metrics.keys()):
        m = config_metrics[config_name]
        if not m['ctaa_full_ratio']:
            continue

        configs.append(config_name.replace('ctaa_', '').replace('_', '\n'))
        full_ratios.append(np.mean(m['ctaa_full_ratio']) * 100)
        centroid_ratios.append(np.mean(m['ctaa_centroid_ratio']) * 100)
        skip_ratios.append(np.mean(m['ctaa_skip_ratio']) * 100)

    if not configs:
        print("No CTAA data available for plotting")
        return

    fig, ax = plt.subplots(figsize=(12, 6))

    x = np.arange(len(configs))
    width = 0.6

    p1 = ax.bar(x, full_ratios, width, label='Full Attention', color='#e74c3c')
    p2 = ax.bar(x, centroid_ratios, width, bottom=full_ratios, label='Centroid Attention', color='#f39c12')
    p3 = ax.bar(x, skip_ratios, width, bottom=np.array(full_ratios) + np.array(centroid_ratios),
                label='Skipped', color='#27ae60')

    ax.set_ylabel('Percentage of Block Pairs')
    ax.set_xlabel('Configuration')
    ax.set_title('CTAA Attention Type Distribution')
    ax.set_xticks(x)
    ax.set_xticklabels(configs, fontsize=8)
    ax.legend(loc='upper right')
    ax.set_ylim(0, 100)

    # Add percentage labels
    for i, (f, c, s) in enumerate(zip(full_ratios, centroid_ratios, skip_ratios)):
        ax.text(i, f/2, f'{f:.0f}%', ha='center', va='center', fontsize=8, color='white')
        ax.text(i, f + c/2, f'{c:.0f}%', ha='center', va='center', fontsize=8, color='black')
        ax.text(i, f + c + s/2, f'{s:.0f}%', ha='center', va='center', fontsize=8, color='white')

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Plot saved to: {output_path}")


def generate_latex_table(config_metrics: Dict, quality_results: Optional[Dict]) -> str:
    """Generate LaTeX table for academic paper."""
    lines = [
        r"\begin{table}[h]",
        r"\centering",
        r"\caption{CTAA Performance and Quality Comparison}",
        r"\label{tab:ctaa_results}",
        r"\begin{tabular}{lcccccc}",
        r"\toprule",
        r"Config & $p_{full}$ & $p_{total}$ & Full \% & Skip \% & Speedup & PSNR (dB) \\",
        r"\midrule",
    ]

    # Get quality data
    quality_by_config = {}
    if quality_results and 'evaluations' in quality_results:
        for eval_result in quality_results['evaluations']:
            config = eval_result.get('config_name', 'unknown')
            if config not in quality_by_config:
                quality_by_config[config] = []
            quality_by_config[config].append(eval_result['metrics']['psnr']['mean'])

    for config_name in sorted(config_metrics.keys()):
        m = config_metrics[config_name]
        config = m['config']

        full_mean = np.mean(m['ctaa_full_ratio']) * 100 if m['ctaa_full_ratio'] else 100
        skip_mean = np.mean(m['ctaa_skip_ratio']) * 100 if m['ctaa_skip_ratio'] else 0
        speedup = compute_theoretical_speedup(
            full_mean / 100,
            np.mean(m['ctaa_centroid_ratio']) if m['ctaa_centroid_ratio'] else 0,
            skip_mean / 100
        )

        psnr = np.mean(quality_by_config.get(config_name, [0])) if config_name in quality_by_config else "-"
        psnr_str = f"{psnr:.2f}" if isinstance(psnr, float) else psnr

        display_name = config_name.replace('_', r'\_')
        line = f"{display_name} & {config['ctaa_p_full']:.2f} & {config['ctaa_p_total']:.2f} & "
        line += f"{full_mean:.1f} & {skip_mean:.1f} & {speedup:.2f}x & {psnr_str} \\\\"
        lines.append(line)

    lines.extend([
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ])

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Analyze CTAA experiment results")
    parser.add_argument("--results_dir", type=str, required=True, help="Experiment results directory")
    parser.add_argument("--output_dir", type=str, default=None, help="Output directory for plots")
    parser.add_argument("--latex", action="store_true", help="Generate LaTeX table")
    parser.add_argument("--no_plots", action="store_true", help="Skip plot generation")

    args = parser.parse_args()

    # Load results
    print(f"Loading results from: {args.results_dir}")
    suite_results = load_suite_results(args.results_dir)
    quality_results = load_quality_results(args.results_dir)

    # Extract metrics
    config_metrics = extract_metrics(suite_results['results'])

    # Print tables
    print_performance_table(config_metrics)
    if quality_results:
        print_quality_table(quality_results)

    # Generate plots
    if not args.no_plots and HAS_MATPLOTLIB:
        output_dir = args.output_dir or args.results_dir
        os.makedirs(output_dir, exist_ok=True)

        plot_attention_distribution(
            config_metrics,
            os.path.join(output_dir, "attention_distribution.png")
        )
        plot_parameter_sensitivity(
            config_metrics,
            os.path.join(output_dir, "parameter_sensitivity.png")
        )
        if quality_results:
            plot_speedup_vs_quality(
                config_metrics, quality_results,
                os.path.join(output_dir, "speedup_vs_quality.png")
            )

    # Generate LaTeX
    if args.latex:
        latex = generate_latex_table(config_metrics, quality_results)
        print("\n" + "=" * 60)
        print("LATEX TABLE")
        print("=" * 60)
        print(latex)

        latex_path = os.path.join(args.output_dir or args.results_dir, "results_table.tex")
        with open(latex_path, 'w') as f:
            f.write(latex)
        print(f"\nLaTeX saved to: {latex_path}")


if __name__ == "__main__":
    main()
