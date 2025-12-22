#!/usr/bin/env python3
"""
CTA Density Log Analyzer

Analyzes JSONL density logs from CTA framework runs.
Provides per-layer, per-timestep, and overall statistics.

Usage:
    python scripts/analyze_density_log.py --input logs/density_log.jsonl
    python scripts/analyze_density_log.py --input logs/density_log.jsonl --output report.txt
    python scripts/analyze_density_log.py --input logs/density_log.jsonl --plot

Input format (JSONL):
    {"timestep": 999, "layer": 0, "avg_density": 0.45, "density": [0.3, 0.5, ...]}
"""

import argparse
import json
import os
import sys
from collections import defaultdict
from typing import Dict, List, Tuple, Any, Optional
from dataclasses import dataclass, field

try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False
    print("Warning: numpy not available. Using basic Python math.")

try:
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False


# =============================================================================
# Data Structures
# =============================================================================

@dataclass
class DensityEntry:
    """Single density log entry."""
    timestep: int
    layer: int
    avg_density: float
    density: List[float] = field(default_factory=list)


@dataclass
class LayerStats:
    """Statistics for a single layer across all timesteps."""
    layer_idx: int
    densities: List[float] = field(default_factory=list)
    timesteps: List[int] = field(default_factory=list)

    def mean(self) -> float:
        return sum(self.densities) / len(self.densities) if self.densities else 0.0

    def std(self) -> float:
        if len(self.densities) < 2:
            return 0.0
        mean = self.mean()
        variance = sum((d - mean) ** 2 for d in self.densities) / len(self.densities)
        return variance ** 0.5

    def min(self) -> float:
        return min(self.densities) if self.densities else 0.0

    def max(self) -> float:
        return max(self.densities) if self.densities else 0.0


@dataclass
class TimestepStats:
    """Statistics for a single timestep across all layers."""
    timestep: int
    densities: List[float] = field(default_factory=list)
    layers: List[int] = field(default_factory=list)

    def mean(self) -> float:
        return sum(self.densities) / len(self.densities) if self.densities else 0.0

    def std(self) -> float:
        if len(self.densities) < 2:
            return 0.0
        mean = self.mean()
        variance = sum((d - mean) ** 2 for d in self.densities) / len(self.densities)
        return variance ** 0.5


# =============================================================================
# Loading and Parsing
# =============================================================================

def load_density_log(filepath: str) -> List[DensityEntry]:
    """Load density log from JSONL file."""
    entries = []

    with open(filepath, 'r') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue

            try:
                data = json.loads(line)
                entry = DensityEntry(
                    timestep=int(data.get('timestep', 0)),
                    layer=int(data.get('layer', 0)),
                    avg_density=float(data.get('avg_density', 0.0)),
                    density=data.get('density', []),
                )
                entries.append(entry)
            except json.JSONDecodeError as e:
                print(f"Warning: Could not parse line {line_num}: {e}")
            except (KeyError, TypeError, ValueError) as e:
                print(f"Warning: Invalid data on line {line_num}: {e}")

    return entries


# =============================================================================
# Analysis Functions
# =============================================================================

def compute_layer_stats(entries: List[DensityEntry]) -> Dict[int, LayerStats]:
    """Compute statistics grouped by layer."""
    layer_data: Dict[int, LayerStats] = {}

    for entry in entries:
        if entry.layer not in layer_data:
            layer_data[entry.layer] = LayerStats(layer_idx=entry.layer)

        layer_data[entry.layer].densities.append(entry.avg_density)
        layer_data[entry.layer].timesteps.append(entry.timestep)

    return layer_data


def compute_timestep_stats(entries: List[DensityEntry]) -> Dict[int, TimestepStats]:
    """Compute statistics grouped by timestep."""
    timestep_data: Dict[int, TimestepStats] = {}

    for entry in entries:
        if entry.timestep not in timestep_data:
            timestep_data[entry.timestep] = TimestepStats(timestep=entry.timestep)

        timestep_data[entry.timestep].densities.append(entry.avg_density)
        timestep_data[entry.timestep].layers.append(entry.layer)

    return timestep_data


def compute_overall_stats(entries: List[DensityEntry]) -> Dict[str, Any]:
    """Compute overall statistics."""
    if not entries:
        return {}

    all_densities = [e.avg_density for e in entries]
    all_timesteps = sorted(set(e.timestep for e in entries))
    all_layers = sorted(set(e.layer for e in entries))

    mean_density = sum(all_densities) / len(all_densities)
    variance = sum((d - mean_density) ** 2 for d in all_densities) / len(all_densities)
    std_density = variance ** 0.5

    # Compute theoretical speedup from density
    # Density = fraction of attention blocks computed
    # Speedup = 1 / density (theoretical, assuming linear scaling)
    theoretical_speedup = 1.0 / mean_density if mean_density > 0 else 1.0

    # Compute density trend (early vs late timesteps)
    mid_timestep = (max(all_timesteps) + min(all_timesteps)) // 2
    early_entries = [e for e in entries if e.timestep > mid_timestep]
    late_entries = [e for e in entries if e.timestep <= mid_timestep]

    early_density = sum(e.avg_density for e in early_entries) / len(early_entries) if early_entries else 0
    late_density = sum(e.avg_density for e in late_entries) / len(late_entries) if late_entries else 0

    return {
        'total_entries': len(entries),
        'num_timesteps': len(all_timesteps),
        'num_layers': len(all_layers),
        'timestep_range': (min(all_timesteps), max(all_timesteps)),
        'layer_range': (min(all_layers), max(all_layers)),
        'mean_density': mean_density,
        'std_density': std_density,
        'min_density': min(all_densities),
        'max_density': max(all_densities),
        'theoretical_speedup': theoretical_speedup,
        'compute_reduction': (1.0 - mean_density) * 100,
        'early_timestep_density': early_density,
        'late_timestep_density': late_density,
        'density_trend': 'increasing' if late_density > early_density else 'decreasing',
    }


def identify_critical_layers(layer_stats: Dict[int, LayerStats], threshold: float = 0.8) -> List[int]:
    """Identify layers with high average density (potential bottlenecks)."""
    critical = []
    for layer_idx, stats in layer_stats.items():
        if stats.mean() > threshold:
            critical.append(layer_idx)
    return sorted(critical)


def identify_efficient_layers(layer_stats: Dict[int, LayerStats], threshold: float = 0.3) -> List[int]:
    """Identify layers with low average density (most efficient)."""
    efficient = []
    for layer_idx, stats in layer_stats.items():
        if stats.mean() < threshold:
            efficient.append(layer_idx)
    return sorted(efficient)


# =============================================================================
# Report Generation
# =============================================================================

def generate_report(entries: List[DensityEntry]) -> str:
    """Generate comprehensive text report."""
    lines = []

    # Header
    lines.append("=" * 80)
    lines.append("  CTA DENSITY LOG ANALYSIS REPORT")
    lines.append("  Sparse Attention Pattern Analysis")
    lines.append("=" * 80)
    lines.append("")

    # Overall statistics
    overall = compute_overall_stats(entries)

    lines.append("OVERALL STATISTICS")
    lines.append("-" * 40)
    lines.append(f"  Total log entries:     {overall['total_entries']:>8}")
    lines.append(f"  Number of timesteps:   {overall['num_timesteps']:>8}")
    lines.append(f"  Number of layers:      {overall['num_layers']:>8}")
    lines.append(f"  Timestep range:        {overall['timestep_range'][0]} -> {overall['timestep_range'][1]}")
    lines.append(f"  Layer range:           {overall['layer_range'][0]} -> {overall['layer_range'][1]}")
    lines.append("")

    lines.append("DENSITY METRICS")
    lines.append("-" * 40)
    lines.append(f"  Mean density:          {overall['mean_density']:>8.4f}")
    lines.append(f"  Std deviation:         {overall['std_density']:>8.4f}")
    lines.append(f"  Min density:           {overall['min_density']:>8.4f}")
    lines.append(f"  Max density:           {overall['max_density']:>8.4f}")
    lines.append("")

    lines.append("EFFICIENCY ANALYSIS")
    lines.append("-" * 40)
    lines.append(f"  Theoretical Speedup:   {overall['theoretical_speedup']:>8.2f}x")
    lines.append(f"  Compute Reduction:     {overall['compute_reduction']:>8.1f}%")
    lines.append("")
    lines.append(f"  Early timestep density:{overall['early_timestep_density']:>8.4f}")
    lines.append(f"  Late timestep density: {overall['late_timestep_density']:>8.4f}")
    lines.append(f"  Density trend:         {overall['density_trend']}")
    lines.append("")

    # Per-layer analysis
    layer_stats = compute_layer_stats(entries)
    lines.append("=" * 80)
    lines.append("PER-LAYER ANALYSIS")
    lines.append("=" * 80)
    lines.append("")

    # Table header
    lines.append(f"{'Layer':>6} {'Mean':>10} {'Std':>10} {'Min':>10} {'Max':>10} {'Calls':>8}")
    lines.append("-" * 60)

    for layer_idx in sorted(layer_stats.keys()):
        stats = layer_stats[layer_idx]
        lines.append(f"{layer_idx:>6} {stats.mean():>10.4f} {stats.std():>10.4f} "
                    f"{stats.min():>10.4f} {stats.max():>10.4f} {len(stats.densities):>8}")

    lines.append("")

    # Identify critical and efficient layers
    critical_layers = identify_critical_layers(layer_stats)
    efficient_layers = identify_efficient_layers(layer_stats)

    if critical_layers:
        lines.append(f"  High-density layers (>0.8): {critical_layers}")
        lines.append("    -> These layers use more full attention, potential bottlenecks")
    else:
        lines.append("  No high-density layers (>0.8) - good sparsity across all layers")

    lines.append("")

    if efficient_layers:
        lines.append(f"  Low-density layers (<0.3): {efficient_layers}")
        lines.append("    -> These layers benefit most from sparse attention")
    lines.append("")

    # Per-timestep analysis (summary)
    timestep_stats = compute_timestep_stats(entries)
    lines.append("=" * 80)
    lines.append("PER-TIMESTEP SUMMARY")
    lines.append("=" * 80)
    lines.append("")

    timesteps = sorted(timestep_stats.keys(), reverse=True)

    # Show first 5 and last 5 timesteps
    if len(timesteps) > 10:
        show_timesteps = timesteps[:5] + ['...'] + timesteps[-5:]
    else:
        show_timesteps = timesteps

    lines.append(f"{'Timestep':>10} {'Mean Density':>15} {'Std':>10} {'Layers':>8}")
    lines.append("-" * 50)

    for ts in show_timesteps:
        if ts == '...':
            lines.append("       ...")
        else:
            stats = timestep_stats[ts]
            lines.append(f"{ts:>10} {stats.mean():>15.4f} {stats.std():>10.4f} {len(stats.layers):>8}")

    lines.append("")

    # Interpretation
    lines.append("=" * 80)
    lines.append("INTERPRETATION")
    lines.append("=" * 80)
    lines.append("")

    if overall['mean_density'] < 0.3:
        lines.append("  [EXCELLENT] Very low average density - high sparsity achieved")
        lines.append(f"  Theoretical speedup of {overall['theoretical_speedup']:.2f}x on attention")
    elif overall['mean_density'] < 0.5:
        lines.append("  [GOOD] Moderate density - reasonable sparsity")
        lines.append(f"  Theoretical speedup of {overall['theoretical_speedup']:.2f}x on attention")
    elif overall['mean_density'] < 0.7:
        lines.append("  [MODERATE] Higher density - less sparsity than ideal")
        lines.append("  Consider adjusting p_full/p_total thresholds")
    else:
        lines.append("  [REVIEW] High density - most attention is computed fully")
        lines.append("  Sparse attention may not be providing significant benefit")

    lines.append("")

    if overall['density_trend'] == 'increasing':
        lines.append("  Density increases during denoising (early -> late timesteps)")
        lines.append("  This is typical: later timesteps need finer detail attention")
    else:
        lines.append("  Density decreases during denoising")
        lines.append("  Early timesteps use more full attention for coarse structure")

    lines.append("")
    lines.append("=" * 80)

    return "\n".join(lines)


def generate_json_summary(entries: List[DensityEntry]) -> Dict[str, Any]:
    """Generate JSON summary."""
    overall = compute_overall_stats(entries)
    layer_stats = compute_layer_stats(entries)
    timestep_stats = compute_timestep_stats(entries)

    return {
        'overall': overall,
        'per_layer': {
            layer_idx: {
                'mean': stats.mean(),
                'std': stats.std(),
                'min': stats.min(),
                'max': stats.max(),
                'num_calls': len(stats.densities),
            }
            for layer_idx, stats in layer_stats.items()
        },
        'per_timestep': {
            ts: {
                'mean': stats.mean(),
                'std': stats.std(),
                'num_layers': len(stats.layers),
            }
            for ts, stats in timestep_stats.items()
        },
        'critical_layers': identify_critical_layers(layer_stats),
        'efficient_layers': identify_efficient_layers(layer_stats),
    }


# =============================================================================
# Visualization
# =============================================================================

def create_plots(entries: List[DensityEntry], output_dir: str):
    """Create visualization plots."""
    if not HAS_MATPLOTLIB:
        print("Warning: matplotlib not available for plotting")
        return

    os.makedirs(output_dir, exist_ok=True)

    layer_stats = compute_layer_stats(entries)
    timestep_stats = compute_timestep_stats(entries)

    # Plot 1: Density heatmap (timestep x layer)
    create_density_heatmap(entries, os.path.join(output_dir, 'density_heatmap.png'))

    # Plot 2: Per-layer density distribution
    create_layer_distribution(layer_stats, os.path.join(output_dir, 'layer_density.png'))

    # Plot 3: Density over timesteps
    create_timestep_trend(timestep_stats, os.path.join(output_dir, 'timestep_trend.png'))

    # Plot 4: Layer density box plot
    create_layer_boxplot(layer_stats, os.path.join(output_dir, 'layer_boxplot.png'))


def create_density_heatmap(entries: List[DensityEntry], output_path: str):
    """Create heatmap of density across timesteps and layers."""
    if not entries:
        return

    timesteps = sorted(set(e.timestep for e in entries), reverse=True)
    layers = sorted(set(e.layer for e in entries))

    # Create matrix
    matrix = [[0.0] * len(layers) for _ in range(len(timesteps))]
    ts_to_idx = {ts: i for i, ts in enumerate(timesteps)}
    layer_to_idx = {l: i for i, l in enumerate(layers)}

    for entry in entries:
        i = ts_to_idx[entry.timestep]
        j = layer_to_idx[entry.layer]
        matrix[i][j] = entry.avg_density

    fig, ax = plt.subplots(figsize=(14, 8))

    im = ax.imshow(matrix, cmap='RdYlGn_r', aspect='auto', vmin=0, vmax=1)

    # Labels
    ax.set_xlabel('Layer Index')
    ax.set_ylabel('Timestep')
    ax.set_title('Attention Density Heatmap (Layer × Timestep)')

    # Ticks
    if len(layers) <= 20:
        ax.set_xticks(range(len(layers)))
        ax.set_xticklabels(layers)
    else:
        ax.set_xticks(range(0, len(layers), 5))
        ax.set_xticklabels([layers[i] for i in range(0, len(layers), 5)])

    if len(timesteps) <= 20:
        ax.set_yticks(range(len(timesteps)))
        ax.set_yticklabels(timesteps)
    else:
        ax.set_yticks(range(0, len(timesteps), 5))
        ax.set_yticklabels([timesteps[i] for i in range(0, len(timesteps), 5)])

    # Colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Density (0=sparse, 1=full)')

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Saved: {output_path}")


def create_layer_distribution(layer_stats: Dict[int, LayerStats], output_path: str):
    """Create bar chart of mean density per layer."""
    layers = sorted(layer_stats.keys())
    means = [layer_stats[l].mean() for l in layers]
    stds = [layer_stats[l].std() for l in layers]

    fig, ax = plt.subplots(figsize=(14, 6))

    colors = ['#27ae60' if m < 0.3 else '#f39c12' if m < 0.6 else '#e74c3c' for m in means]
    bars = ax.bar(layers, means, color=colors, yerr=stds, capsize=2, alpha=0.8)

    ax.set_xlabel('Layer Index')
    ax.set_ylabel('Mean Density')
    ax.set_title('Mean Attention Density per Layer')
    ax.set_ylim(0, 1)
    ax.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5, label='50% threshold')

    # Legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#27ae60', label='Low (<0.3) - Efficient'),
        Patch(facecolor='#f39c12', label='Medium (0.3-0.6)'),
        Patch(facecolor='#e74c3c', label='High (>0.6) - Dense'),
    ]
    ax.legend(handles=legend_elements, loc='upper right')

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Saved: {output_path}")


def create_timestep_trend(timestep_stats: Dict[int, TimestepStats], output_path: str):
    """Create line plot of density over timesteps."""
    timesteps = sorted(timestep_stats.keys(), reverse=True)
    means = [timestep_stats[ts].mean() for ts in timesteps]
    stds = [timestep_stats[ts].std() for ts in timesteps]

    fig, ax = plt.subplots(figsize=(12, 6))

    ax.plot(range(len(timesteps)), means, 'b-', linewidth=2, label='Mean Density')
    ax.fill_between(range(len(timesteps)),
                    [m - s for m, s in zip(means, stds)],
                    [m + s for m, s in zip(means, stds)],
                    alpha=0.3, color='blue', label='±1 Std')

    ax.set_xlabel('Inference Step (0=start, N=end)')
    ax.set_ylabel('Mean Density')
    ax.set_title('Attention Density Trend During Diffusion')
    ax.set_ylim(0, 1)
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Saved: {output_path}")


def create_layer_boxplot(layer_stats: Dict[int, LayerStats], output_path: str):
    """Create box plot of density distribution per layer."""
    layers = sorted(layer_stats.keys())

    # Sample layers if too many
    if len(layers) > 30:
        sample_indices = list(range(0, len(layers), len(layers) // 20 + 1))
        layers = [layers[i] for i in sample_indices]

    data = [layer_stats[l].densities for l in layers]

    fig, ax = plt.subplots(figsize=(14, 6))

    bp = ax.boxplot(data, labels=layers, patch_artist=True)

    # Color by median
    for i, patch in enumerate(bp['boxes']):
        median = layer_stats[layers[i]].mean()
        if median < 0.3:
            patch.set_facecolor('#27ae60')
        elif median < 0.6:
            patch.set_facecolor('#f39c12')
        else:
            patch.set_facecolor('#e74c3c')
        patch.set_alpha(0.7)

    ax.set_xlabel('Layer Index')
    ax.set_ylabel('Density')
    ax.set_title('Density Distribution per Layer')
    ax.set_ylim(0, 1)

    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Saved: {output_path}")


# =============================================================================
# Main Entry Point
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Analyze CTA density logs",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Analyze log file and print report
    python scripts/analyze_density_log.py --input logs/density_log.jsonl

    # Save report to file
    python scripts/analyze_density_log.py --input logs/density_log.jsonl --output report.txt

    # Generate plots
    python scripts/analyze_density_log.py --input logs/density_log.jsonl --plot

    # Output JSON summary
    python scripts/analyze_density_log.py --input logs/density_log.jsonl --json
"""
    )

    parser.add_argument("--input", "-i", type=str, required=True,
                       help="Input JSONL file with density logs")
    parser.add_argument("--output", "-o", type=str, default=None,
                       help="Output file for text report")
    parser.add_argument("--json", action="store_true",
                       help="Output JSON summary instead of text")
    parser.add_argument("--plot", action="store_true",
                       help="Generate visualization plots")
    parser.add_argument("--plot_dir", type=str, default="./density_analysis_plots",
                       help="Directory for plot output")

    args = parser.parse_args()

    # Load data
    if not os.path.exists(args.input):
        print(f"Error: Input file not found: {args.input}")
        sys.exit(1)

    print(f"Loading density log from: {args.input}")
    entries = load_density_log(args.input)

    if not entries:
        print("Error: No valid entries found in log file")
        sys.exit(1)

    print(f"Loaded {len(entries)} log entries")

    # Generate output
    if args.json:
        summary = generate_json_summary(entries)
        output = json.dumps(summary, indent=2)
    else:
        output = generate_report(entries)

    # Save or print
    if args.output:
        with open(args.output, 'w') as f:
            f.write(output)
        print(f"Report saved to: {args.output}")
    else:
        print(output)

    # Generate plots
    if args.plot:
        create_plots(entries, args.plot_dir)
        print(f"\nPlots saved to: {args.plot_dir}/")


if __name__ == "__main__":
    main()
