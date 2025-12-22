#!/usr/bin/env python3
"""
CTA Framework Results Analyzer

Analyzes JSON output from CTA (Cross-Timestep Amortization) framework runs.
Provides detailed statistics for CTCA, CTAA, and Attention-Informed Offloading.

Usage:
    python scripts/analyze_cta_results.py --input results.json
    python scripts/analyze_cta_results.py --input results.json --output analysis_report.txt
    python scripts/analyze_cta_results.py --input results.json --plot

The input JSON should contain CTA framework statistics (exported by the inference script).
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from datetime import datetime

# Optional imports
try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False
    print("Warning: numpy not available. Some analysis features disabled.")

try:
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False


# =============================================================================
# Data Classes for Type Safety
# =============================================================================

@dataclass
class CTCAStats:
    """Statistics for Cross-Timestep Cluster Amortization."""
    full_cluster_count: int = 0
    update_only_count: int = 0
    quality_triggered_recluster: int = 0
    interval_triggered_recluster: int = 0
    total_calls: int = 0
    reuse_ratio: float = 0.0
    estimated_speedup: float = 1.0
    total_time_ms: float = 0.0

    @classmethod
    def from_dict(cls, d: Dict) -> 'CTCAStats':
        return cls(
            full_cluster_count=d.get('full_cluster_count', 0),
            update_only_count=d.get('update_only_count', 0),
            quality_triggered_recluster=d.get('quality_triggered_recluster', 0),
            interval_triggered_recluster=d.get('interval_triggered_recluster', 0),
            total_calls=d.get('total_calls', 0),
            reuse_ratio=d.get('reuse_ratio', d.get('update_only_ratio', 0.0)),
            estimated_speedup=d.get('estimated_speedup', 1.0),
            total_time_ms=d.get('total_time_ms', 0.0),
        )


@dataclass
class CTAAStats:
    """Statistics for Cross-Timestep Attention Amortization."""
    full_attention_blocks: int = 0
    centroid_attention_blocks: int = 0
    skipped_blocks: int = 0
    total_blocks: int = 0
    full_ratio: float = 0.0
    centroid_ratio: float = 0.0
    skip_ratio: float = 0.0
    average_density: float = 0.0
    total_time_ms: float = 0.0

    @classmethod
    def from_dict(cls, d: Dict) -> 'CTAAStats':
        total = d.get('total_blocks', 1)
        return cls(
            full_attention_blocks=d.get('full_attention_blocks', 0),
            centroid_attention_blocks=d.get('centroid_attention_blocks', 0),
            skipped_blocks=d.get('skipped_blocks', 0),
            total_blocks=total,
            full_ratio=d.get('full_ratio', d.get('full_attention_blocks', 0) / max(total, 1)),
            centroid_ratio=d.get('centroid_ratio', d.get('centroid_attention_blocks', 0) / max(total, 1)),
            skip_ratio=d.get('skip_ratio', d.get('skipped_blocks', 0) / max(total, 1)),
            average_density=d.get('average_density', 0.0),
            total_time_ms=d.get('total_time_ms', 0.0),
        )


@dataclass
class OffloadStats:
    """Statistics for Attention-Informed Offloading."""
    gpu_loads: int = 0
    gpu_offloads: int = 0
    prefetch_hits: int = 0
    prefetch_misses: int = 0
    prefetch_hit_ratio: float = 0.0
    priority_evictions: int = 0
    fifo_evictions: int = 0
    cache_clears: int = 0
    num_layers: int = 60
    layers_on_gpu: int = 6
    layer_memory_mb: float = 0.0

    @classmethod
    def from_dict(cls, d: Dict) -> 'OffloadStats':
        hits = d.get('prefetch_hits', 0)
        misses = d.get('prefetch_misses', 0)
        total = hits + misses
        return cls(
            gpu_loads=d.get('gpu_loads', 0),
            gpu_offloads=d.get('gpu_offloads', 0),
            prefetch_hits=hits,
            prefetch_misses=misses,
            prefetch_hit_ratio=hits / max(total, 1),
            priority_evictions=d.get('priority_evictions', 0),
            fifo_evictions=d.get('fifo_evictions', 0),
            cache_clears=d.get('cache_clears', 0),
            num_layers=d.get('num_layers', 60),
            layers_on_gpu=d.get('layers_on_gpu', d.get('num_layers_on_gpu', 6)),
            layer_memory_mb=d.get('layer_memory_mb', 0.0),
        )


@dataclass
class CTAResults:
    """Complete CTA Framework results."""
    # Run info
    timestamp: str = ""
    prompt: str = ""
    resolution: str = ""
    num_frames: int = 0
    num_timesteps: int = 0

    # Component stats
    ctca: Optional[CTCAStats] = None
    ctaa: Optional[CTAAStats] = None
    offload: Optional[OffloadStats] = None

    # Overall metrics
    total_generation_time_s: float = 0.0
    peak_memory_gb: float = 0.0

    @classmethod
    def from_dict(cls, d: Dict) -> 'CTAResults':
        result = cls(
            timestamp=d.get('timestamp', ''),
            prompt=d.get('prompt', ''),
            resolution=d.get('resolution', ''),
            num_frames=d.get('num_frames', 0),
            num_timesteps=d.get('num_timesteps', 0),
            total_generation_time_s=d.get('total_generation_time_s', d.get('generation_time', 0.0)),
            peak_memory_gb=d.get('peak_memory_gb', 0.0),
        )

        if 'ctca' in d or 'ctca_stats' in d:
            result.ctca = CTCAStats.from_dict(d.get('ctca', d.get('ctca_stats', {})))

        if 'ctaa' in d or 'ctaa_stats' in d:
            result.ctaa = CTAAStats.from_dict(d.get('ctaa', d.get('ctaa_stats', {})))

        if 'offload' in d or 'offload_stats' in d:
            result.offload = OffloadStats.from_dict(d.get('offload', d.get('offload_stats', {})))

        return result


# =============================================================================
# Analysis Functions
# =============================================================================

def compute_ctca_efficiency(stats: CTCAStats) -> Dict[str, float]:
    """Compute CTCA efficiency metrics."""
    total = stats.full_cluster_count + stats.update_only_count
    if total == 0:
        return {'reuse_ratio': 0.0, 'speedup': 1.0, 'time_saved_ratio': 0.0}

    reuse_ratio = stats.update_only_count / total

    # Full clustering ~50 iters, update ~2 iters
    full_cost = stats.full_cluster_count * 50
    update_cost = stats.update_only_count * 2
    baseline_cost = total * 50
    actual_cost = full_cost + update_cost

    speedup = baseline_cost / max(actual_cost, 1)
    time_saved_ratio = 1.0 - (actual_cost / baseline_cost) if baseline_cost > 0 else 0.0

    return {
        'reuse_ratio': reuse_ratio,
        'speedup': speedup,
        'time_saved_ratio': time_saved_ratio,
        'quality_triggers': stats.quality_triggered_recluster,
        'interval_triggers': stats.interval_triggered_recluster,
    }


def compute_ctaa_efficiency(stats: CTAAStats) -> Dict[str, float]:
    """Compute CTAA efficiency metrics."""
    if stats.total_blocks == 0:
        return {'theoretical_speedup': 1.0, 'compute_reduction': 0.0}

    # Centroid attention ~1% cost, skipped ~0% cost
    effective_cost = stats.full_ratio + stats.centroid_ratio * 0.01
    theoretical_speedup = 1.0 / max(effective_cost, 0.01)
    compute_reduction = 1.0 - effective_cost

    return {
        'theoretical_speedup': theoretical_speedup,
        'compute_reduction': compute_reduction,
        'full_pct': stats.full_ratio * 100,
        'centroid_pct': stats.centroid_ratio * 100,
        'skip_pct': stats.skip_ratio * 100,
    }


def compute_offload_efficiency(stats: OffloadStats) -> Dict[str, float]:
    """Compute offloading efficiency metrics."""
    total_loads = stats.prefetch_hits + stats.prefetch_misses

    return {
        'prefetch_hit_ratio': stats.prefetch_hit_ratio,
        'prefetch_miss_ratio': 1.0 - stats.prefetch_hit_ratio,
        'priority_eviction_ratio': stats.priority_evictions / max(stats.gpu_offloads, 1),
        'total_transfers': stats.gpu_loads + stats.gpu_offloads,
        'memory_per_layer_mb': stats.layer_memory_mb,
        'gpu_window_size': stats.layers_on_gpu,
    }


def compute_overall_metrics(results: CTAResults) -> Dict[str, Any]:
    """Compute overall framework metrics."""
    metrics = {
        'generation_time_s': results.total_generation_time_s,
        'peak_memory_gb': results.peak_memory_gb,
        'resolution': results.resolution,
        'frames': results.num_frames,
        'timesteps': results.num_timesteps,
    }

    # Compute combined speedup estimate
    ctca_speedup = 1.0
    ctaa_speedup = 1.0

    if results.ctca:
        efficiency = compute_ctca_efficiency(results.ctca)
        ctca_speedup = efficiency['speedup']
        metrics['ctca_speedup'] = ctca_speedup

    if results.ctaa:
        efficiency = compute_ctaa_efficiency(results.ctaa)
        ctaa_speedup = efficiency['theoretical_speedup']
        metrics['ctaa_speedup'] = ctaa_speedup

    # Combined attention speedup (CTCA affects clustering, CTAA affects attention compute)
    # These are roughly multiplicative for attention component
    metrics['combined_attention_speedup'] = ctca_speedup * ctaa_speedup

    return metrics


# =============================================================================
# Report Generation
# =============================================================================

def generate_text_report(results: CTAResults) -> str:
    """Generate detailed text report."""
    lines = []

    # Header
    lines.append("=" * 80)
    lines.append("  CTA FRAMEWORK ANALYSIS REPORT")
    lines.append("  Cross-Timestep Amortization: CTCA + CTAA + AI-Offload")
    lines.append("=" * 80)
    lines.append("")

    # Run info
    lines.append("RUN INFORMATION")
    lines.append("-" * 40)
    if results.timestamp:
        lines.append(f"  Timestamp:      {results.timestamp}")
    if results.prompt:
        lines.append(f"  Prompt:         {results.prompt[:60]}...")
    lines.append(f"  Resolution:     {results.resolution}")
    lines.append(f"  Frames:         {results.num_frames}")
    lines.append(f"  Timesteps:      {results.num_timesteps}")
    lines.append(f"  Generation Time:{results.total_generation_time_s:.2f}s")
    lines.append(f"  Peak Memory:    {results.peak_memory_gb:.2f} GB")
    lines.append("")

    # CTCA Analysis
    if results.ctca:
        lines.append("=" * 80)
        lines.append("CTCA: Cross-Timestep Cluster Amortization")
        lines.append("=" * 80)

        stats = results.ctca
        efficiency = compute_ctca_efficiency(stats)

        lines.append("")
        lines.append("  Clustering Decisions:")
        lines.append(f"    Full reclusters:     {stats.full_cluster_count:>6}")
        lines.append(f"    Centroid updates:    {stats.update_only_count:>6}")
        lines.append(f"    Total calls:         {stats.full_cluster_count + stats.update_only_count:>6}")
        lines.append("")
        lines.append("  Trigger Analysis:")
        lines.append(f"    Quality-triggered:   {stats.quality_triggered_recluster:>6}")
        lines.append(f"    Interval-triggered:  {stats.interval_triggered_recluster:>6}")
        lines.append("")
        lines.append("  Efficiency Metrics:")
        lines.append(f"    Cluster Reuse Ratio: {efficiency['reuse_ratio']*100:>6.1f}%")
        lines.append(f"    K-means Speedup:     {efficiency['speedup']:>6.2f}x")
        lines.append(f"    Time Saved:          {efficiency['time_saved_ratio']*100:>6.1f}%")

        if stats.total_time_ms > 0:
            lines.append(f"    Total CTCA Time:     {stats.total_time_ms:>6.1f}ms")
        lines.append("")

        # Interpretation
        lines.append("  Interpretation:")
        if efficiency['reuse_ratio'] > 0.8:
            lines.append("    [EXCELLENT] High cluster reuse - temporal coherence well exploited")
        elif efficiency['reuse_ratio'] > 0.6:
            lines.append("    [GOOD] Moderate cluster reuse - consider lowering quality threshold")
        else:
            lines.append("    [REVIEW] Low cluster reuse - video may have high motion/variation")
        lines.append("")

    # CTAA Analysis
    if results.ctaa:
        lines.append("=" * 80)
        lines.append("CTAA: Cross-Timestep Attention Amortization")
        lines.append("=" * 80)

        stats = results.ctaa
        efficiency = compute_ctaa_efficiency(stats)

        lines.append("")
        lines.append("  Attention Tier Distribution:")
        lines.append(f"    Full attention:      {stats.full_attention_blocks:>8} ({efficiency['full_pct']:.1f}%)")
        lines.append(f"    Centroid attention:  {stats.centroid_attention_blocks:>8} ({efficiency['centroid_pct']:.1f}%)")
        lines.append(f"    Skipped:             {stats.skipped_blocks:>8} ({efficiency['skip_pct']:.1f}%)")
        lines.append(f"    Total blocks:        {stats.total_blocks:>8}")
        lines.append("")
        lines.append("  Efficiency Metrics:")
        lines.append(f"    Theoretical Speedup: {efficiency['theoretical_speedup']:>6.2f}x")
        lines.append(f"    Compute Reduction:   {efficiency['compute_reduction']*100:>6.1f}%")

        if stats.average_density > 0:
            lines.append(f"    Average Density:     {stats.average_density:>6.3f}")
        if stats.total_time_ms > 0:
            lines.append(f"    Total Attention Time:{stats.total_time_ms:>6.1f}ms")
        lines.append("")

        # Interpretation
        lines.append("  Interpretation:")
        if efficiency['skip_pct'] > 20:
            lines.append("    [EXCELLENT] High skip ratio - significant attention savings")
        elif efficiency['centroid_pct'] > 30:
            lines.append("    [GOOD] Good centroid usage - balanced quality/speed")
        else:
            lines.append("    [CONSERVATIVE] Most attention is full - consider raising p_full")
        lines.append("")

    # Offload Analysis
    if results.offload:
        lines.append("=" * 80)
        lines.append("AI-OFFLOAD: Attention-Informed Layer Offloading")
        lines.append("=" * 80)

        stats = results.offload
        efficiency = compute_offload_efficiency(stats)

        lines.append("")
        lines.append("  Memory Configuration:")
        lines.append(f"    Total layers:        {stats.num_layers:>6}")
        lines.append(f"    GPU window size:     {stats.layers_on_gpu:>6}")
        if stats.layer_memory_mb > 0:
            lines.append(f"    Memory per layer:    {stats.layer_memory_mb:>6.1f}MB")
            lines.append(f"    GPU memory used:     {stats.layers_on_gpu * stats.layer_memory_mb:>6.1f}MB")
        lines.append("")
        lines.append("  Transfer Statistics:")
        lines.append(f"    GPU loads:           {stats.gpu_loads:>6}")
        lines.append(f"    GPU offloads:        {stats.gpu_offloads:>6}")
        lines.append(f"    Total transfers:     {efficiency['total_transfers']:>6}")
        lines.append("")
        lines.append("  Prefetch Performance:")
        lines.append(f"    Prefetch hits:       {stats.prefetch_hits:>6}")
        lines.append(f"    Prefetch misses:     {stats.prefetch_misses:>6}")
        lines.append(f"    Hit ratio:           {efficiency['prefetch_hit_ratio']*100:>6.1f}%")
        lines.append("")
        lines.append("  Eviction Statistics:")
        lines.append(f"    Priority evictions:  {stats.priority_evictions:>6}")
        lines.append(f"    FIFO evictions:      {stats.fifo_evictions:>6}")
        lines.append(f"    Cache clears:        {stats.cache_clears:>6}")
        lines.append("")

        # Interpretation
        lines.append("  Interpretation:")
        if efficiency['prefetch_hit_ratio'] > 0.9:
            lines.append("    [EXCELLENT] Very high prefetch hit rate - minimal memory stalls")
        elif efficiency['prefetch_hit_ratio'] > 0.7:
            lines.append("    [GOOD] Good prefetch hit rate - some optimization possible")
        else:
            lines.append("    [REVIEW] Low prefetch hit rate - consider increasing prefetch count")
        lines.append("")

    # Overall Summary
    lines.append("=" * 80)
    lines.append("OVERALL SUMMARY")
    lines.append("=" * 80)

    overall = compute_overall_metrics(results)
    lines.append("")
    lines.append("  Combined Efficiency:")
    if 'ctca_speedup' in overall:
        lines.append(f"    CTCA K-means Speedup:    {overall['ctca_speedup']:.2f}x")
    if 'ctaa_speedup' in overall:
        lines.append(f"    CTAA Attention Speedup:  {overall['ctaa_speedup']:.2f}x")
    if 'combined_attention_speedup' in overall:
        lines.append(f"    Combined Attention:      {overall['combined_attention_speedup']:.2f}x")
    lines.append("")

    # Key insight
    lines.append("  Key Insight:")
    lines.append("    All three CTA components exploit TEMPORAL COHERENCE in diffusion:")
    lines.append("    - CTCA: Cluster assignments remain stable across timesteps")
    lines.append("    - CTAA: Attention patterns change gradually")
    lines.append("    - Offload: Compute costs are predictable from recent history")
    lines.append("")
    lines.append("=" * 80)

    return "\n".join(lines)


def generate_json_summary(results: CTAResults) -> Dict:
    """Generate JSON summary for programmatic access."""
    summary = {
        'run_info': {
            'timestamp': results.timestamp,
            'prompt': results.prompt,
            'resolution': results.resolution,
            'num_frames': results.num_frames,
            'num_timesteps': results.num_timesteps,
            'generation_time_s': results.total_generation_time_s,
            'peak_memory_gb': results.peak_memory_gb,
        },
    }

    if results.ctca:
        summary['ctca'] = {
            'raw_stats': vars(results.ctca),
            'efficiency': compute_ctca_efficiency(results.ctca),
        }

    if results.ctaa:
        summary['ctaa'] = {
            'raw_stats': vars(results.ctaa),
            'efficiency': compute_ctaa_efficiency(results.ctaa),
        }

    if results.offload:
        summary['offload'] = {
            'raw_stats': vars(results.offload),
            'efficiency': compute_offload_efficiency(results.offload),
        }

    summary['overall'] = compute_overall_metrics(results)

    return summary


# =============================================================================
# Visualization
# =============================================================================

def create_plots(results: CTAResults, output_dir: str):
    """Create visualization plots."""
    if not HAS_MATPLOTLIB:
        print("Warning: matplotlib not available for plotting")
        return

    os.makedirs(output_dir, exist_ok=True)

    # Figure 1: CTCA pie chart
    if results.ctca:
        fig, ax = plt.subplots(figsize=(8, 6))
        stats = results.ctca

        labels = ['Full Recluster', 'Centroid Update']
        sizes = [stats.full_cluster_count, stats.update_only_count]
        colors = ['#e74c3c', '#27ae60']
        explode = (0, 0.05)

        ax.pie(sizes, explode=explode, labels=labels, colors=colors,
               autopct='%1.1f%%', shadow=True, startangle=90)
        ax.set_title('CTCA: Clustering Decision Distribution')

        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'ctca_distribution.png'), dpi=150)
        plt.close()
        print(f"Saved: {output_dir}/ctca_distribution.png")

    # Figure 2: CTAA stacked bar
    if results.ctaa:
        fig, ax = plt.subplots(figsize=(10, 6))
        stats = results.ctaa

        categories = ['Attention Tiers']
        full = [stats.full_ratio * 100]
        centroid = [stats.centroid_ratio * 100]
        skip = [stats.skip_ratio * 100]

        x = range(len(categories))
        width = 0.5

        ax.bar(x, full, width, label='Full Attention', color='#e74c3c')
        ax.bar(x, centroid, width, bottom=full, label='Centroid Attention', color='#f39c12')
        ax.bar(x, skip, width, bottom=[f+c for f, c in zip(full, centroid)],
               label='Skipped', color='#27ae60')

        ax.set_ylabel('Percentage')
        ax.set_title('CTAA: Attention Tier Distribution')
        ax.set_xticks(x)
        ax.set_xticklabels(categories)
        ax.legend()
        ax.set_ylim(0, 100)

        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'ctaa_distribution.png'), dpi=150)
        plt.close()
        print(f"Saved: {output_dir}/ctaa_distribution.png")

    # Figure 3: Offload prefetch performance
    if results.offload:
        fig, ax = plt.subplots(figsize=(8, 6))
        stats = results.offload

        labels = ['Prefetch Hit', 'Prefetch Miss']
        sizes = [stats.prefetch_hits, stats.prefetch_misses]
        colors = ['#27ae60', '#e74c3c']

        ax.pie(sizes, labels=labels, colors=colors,
               autopct='%1.1f%%', shadow=True, startangle=90)
        ax.set_title('AI-Offload: Prefetch Performance')

        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'offload_prefetch.png'), dpi=150)
        plt.close()
        print(f"Saved: {output_dir}/offload_prefetch.png")

    # Figure 4: Overall speedup summary
    overall = compute_overall_metrics(results)

    fig, ax = plt.subplots(figsize=(10, 6))

    speedups = []
    labels = []

    if 'ctca_speedup' in overall:
        speedups.append(overall['ctca_speedup'])
        labels.append('CTCA\n(K-means)')

    if 'ctaa_speedup' in overall:
        speedups.append(overall['ctaa_speedup'])
        labels.append('CTAA\n(Attention)')

    if 'combined_attention_speedup' in overall:
        speedups.append(overall['combined_attention_speedup'])
        labels.append('Combined\n(CTCA × CTAA)')

    if speedups:
        x = range(len(speedups))
        colors = ['#3498db', '#2ecc71', '#9b59b6'][:len(speedups)]

        bars = ax.bar(x, speedups, color=colors)
        ax.set_ylabel('Speedup (x)')
        ax.set_title('CTA Framework: Estimated Speedup by Component')
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.axhline(y=1.0, color='gray', linestyle='--', alpha=0.5, label='Baseline (1x)')

        # Add value labels
        for bar, val in zip(bars, speedups):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                   f'{val:.2f}x', ha='center', va='bottom', fontsize=12)

        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'cta_speedup_summary.png'), dpi=150)
        plt.close()
        print(f"Saved: {output_dir}/cta_speedup_summary.png")


# =============================================================================
# Main Entry Point
# =============================================================================

def load_results(input_path: str) -> CTAResults:
    """Load results from JSON file."""
    with open(input_path, 'r') as f:
        data = json.load(f)
    return CTAResults.from_dict(data)


def main():
    parser = argparse.ArgumentParser(
        description="Analyze CTA Framework results",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Analyze a single result file
    python scripts/analyze_cta_results.py --input results.json

    # Save report to file
    python scripts/analyze_cta_results.py --input results.json --output report.txt

    # Generate plots
    python scripts/analyze_cta_results.py --input results.json --plot --plot_dir ./plots

    # Output JSON summary
    python scripts/analyze_cta_results.py --input results.json --json
"""
    )

    parser.add_argument("--input", "-i", type=str, required=True,
                       help="Input JSON file with CTA results")
    parser.add_argument("--output", "-o", type=str, default=None,
                       help="Output file for text report")
    parser.add_argument("--json", action="store_true",
                       help="Output JSON summary instead of text")
    parser.add_argument("--plot", action="store_true",
                       help="Generate visualization plots")
    parser.add_argument("--plot_dir", type=str, default="./cta_analysis_plots",
                       help="Directory for plot output")

    args = parser.parse_args()

    # Load results
    if not os.path.exists(args.input):
        print(f"Error: Input file not found: {args.input}")
        sys.exit(1)

    print(f"Loading results from: {args.input}")
    results = load_results(args.input)

    # Generate output
    if args.json:
        summary = generate_json_summary(results)
        output = json.dumps(summary, indent=2)
    else:
        output = generate_text_report(results)

    # Save or print
    if args.output:
        with open(args.output, 'w') as f:
            f.write(output)
        print(f"Report saved to: {args.output}")
    else:
        print(output)

    # Generate plots
    if args.plot:
        create_plots(results, args.plot_dir)


if __name__ == "__main__":
    main()
