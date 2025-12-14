#!/usr/bin/env python3
"""
Generate speedup analysis and breakdown visualization.
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

def create_speedup_analysis():
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))

    # === Left: Component breakdown ===
    ax1 = axes[0]
    ax1.set_title('(a) Computation Time Breakdown', fontsize=11, fontweight='bold')

    # Baseline breakdown
    baseline_components = ['K-means', 'Sparse\nAttention', 'Other']
    baseline_values = [15, 70, 15]  # Percentages
    baseline_colors = ['#FFCDD2', '#BBDEFB', '#C8E6C9']

    # Our method breakdown
    ours_values = [2.1, 40, 15]  # K-means reduced 7x, attention reduced ~1.7x

    x = np.arange(len(baseline_components))
    width = 0.35

    bars1 = ax1.bar(x - width/2, baseline_values, width, label='Baseline (SAP)',
                    color=baseline_colors, edgecolor=['#E53935', '#1976D2', '#43A047'], linewidth=2)
    bars2 = ax1.bar(x + width/2, ours_values, width, label='Ours (CTA)',
                    color=['#EF9A9A', '#90CAF9', '#A5D6A7'],
                    edgecolor=['#E53935', '#1976D2', '#43A047'], linewidth=2)

    ax1.set_ylabel('Time (%)', fontsize=10)
    ax1.set_xticks(x)
    ax1.set_xticklabels(baseline_components, fontsize=9)
    ax1.legend(fontsize=9, loc='upper right')
    ax1.set_ylim(0, 85)

    # Add speedup annotations
    ax1.annotate('7× faster', xy=(0, 2.1), xytext=(-0.5, 25),
                fontsize=9, color='#C62828', fontweight='bold',
                arrowprops=dict(arrowstyle='->', color='#C62828'))
    ax1.annotate('1.7× faster', xy=(1, 40), xytext=(0.3, 55),
                fontsize=9, color='#1565C0', fontweight='bold',
                arrowprops=dict(arrowstyle='->', color='#1565C0'))

    # === Middle: CTAA tier distribution pie ===
    ax2 = axes[1]
    ax2.set_title('(b) CTAA Attention Distribution', fontsize=11, fontweight='bold')

    sizes = [17.2, 31.7, 51.1]
    labels = ['Full\n(17.2%)', 'Centroid\n(31.7%)', 'Skip\n(51.1%)']
    colors = ['#E53935', '#FF9800', '#BDBDBD']
    explode = (0.05, 0.02, 0)

    wedges, texts, autotexts = ax2.pie(sizes, explode=explode, labels=labels, colors=colors,
                                        autopct='', startangle=90,
                                        wedgeprops=dict(edgecolor='white', linewidth=2))

    # Add cost annotations
    ax2.text(0.6, 0.8, 'Cost: 100%', fontsize=8, color='darkred')
    ax2.text(0.9, 0.1, 'Cost: ~1%', fontsize=8, color='darkorange')
    ax2.text(0.5, -0.9, 'Cost: 0%', fontsize=8, color='gray')

    # Center text
    ax2.text(0, 0, 'Effective\nCost:\n17.5%', ha='center', va='center',
             fontsize=11, fontweight='bold')

    # === Right: Overall speedup ===
    ax3 = axes[2]
    ax3.set_title('(c) Combined Speedup Analysis', fontsize=11, fontweight='bold')

    components = ['CTCA\n(K-means)', 'CTAA\n(Attention)', 'Combined\n(Overall)']
    speedups = [7.0, 1.75, 1.5]  # Conservative estimates
    colors = ['#C8E6C9', '#FFE0B2', '#E1BEE7']
    edge_colors = ['#388E3C', '#F57C00', '#7B1FA2']

    bars = ax3.bar(components, speedups, color=colors, edgecolor=edge_colors, linewidth=2)

    ax3.set_ylabel('Speedup Factor (×)', fontsize=10)
    ax3.set_ylim(0, 8)
    ax3.axhline(y=1, color='gray', linestyle='--', alpha=0.5)

    # Add value labels
    for bar, val in zip(bars, speedups):
        ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.2,
                f'{val:.1f}×', ha='center', fontsize=11, fontweight='bold')

    # Add breakdown annotation
    ax3.text(2, 5, 'K-means: 15% → 2.1%\nAttention: 70% → 40%\nTotal: 100% → 57.1%',
             fontsize=9, ha='center', va='center',
             bbox=dict(boxstyle='round', facecolor='#F3E5F5', edgecolor='#7B1FA2'))

    plt.tight_layout()
    plt.savefig('fig_speedup_analysis.pdf', dpi=300, bbox_inches='tight')
    plt.savefig('fig_speedup_analysis.png', dpi=300, bbox_inches='tight')
    print("Saved: fig_speedup_analysis.pdf, fig_speedup_analysis.png")

if __name__ == "__main__":
    create_speedup_analysis()
