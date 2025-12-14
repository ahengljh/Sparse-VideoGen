#!/usr/bin/env python3
"""
Generate overview architecture diagram for CTCA/CTAA paper.
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np

def create_overview_diagram():
    fig, ax = plt.subplots(1, 1, figsize=(14, 8))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 8)
    ax.axis('off')

    # Colors
    color_input = '#E3F2FD'
    color_ctca = '#C8E6C9'
    color_ctaa = '#FFE0B2'
    color_output = '#F3E5F5'
    color_cache = '#FFECB3'

    # Title
    ax.text(7, 7.5, 'Cross-Timestep Amortization (CTA) Architecture',
            fontsize=16, fontweight='bold', ha='center')

    # Input block
    input_box = FancyBboxPatch((0.5, 5), 2.5, 1.5, boxstyle="round,pad=0.05",
                                facecolor=color_input, edgecolor='#1976D2', linewidth=2)
    ax.add_patch(input_box)
    ax.text(1.75, 5.75, 'Input\nLatent $\\mathbf{z}_t$', ha='center', va='center', fontsize=11)

    # CTCA block
    ctca_box = FancyBboxPatch((4, 4.5), 3, 2.5, boxstyle="round,pad=0.05",
                               facecolor=color_ctca, edgecolor='#388E3C', linewidth=2)
    ax.add_patch(ctca_box)
    ax.text(5.5, 6.2, 'CTCA', fontsize=12, fontweight='bold', ha='center')
    ax.text(5.5, 5.5, 'Cross-Timestep\nCluster Amortization', ha='center', va='center', fontsize=10)
    ax.text(5.5, 4.8, '• Reuse K-means\n• Adaptive update', ha='center', va='center', fontsize=9)

    # CTAA block
    ctaa_box = FancyBboxPatch((8, 4.5), 3, 2.5, boxstyle="round,pad=0.05",
                               facecolor=color_ctaa, edgecolor='#F57C00', linewidth=2)
    ax.add_patch(ctaa_box)
    ax.text(9.5, 6.2, 'CTAA', fontsize=12, fontweight='bold', ha='center')
    ax.text(9.5, 5.5, 'Cross-Timestep\nAttention Amortization', ha='center', va='center', fontsize=10)
    ax.text(9.5, 4.8, '• Hierarchical tiers\n• Centroid attention', ha='center', va='center', fontsize=9)

    # Output block
    output_box = FancyBboxPatch((12, 5), 1.5, 1.5, boxstyle="round,pad=0.05",
                                 facecolor=color_output, edgecolor='#7B1FA2', linewidth=2)
    ax.add_patch(output_box)
    ax.text(12.75, 5.75, 'Output\n$\\mathbf{z}_{t-1}$', ha='center', va='center', fontsize=11)

    # Cache block (below)
    cache_box = FancyBboxPatch((4, 1.5), 7, 1.5, boxstyle="round,pad=0.05",
                                facecolor=color_cache, edgecolor='#FFA000', linewidth=2)
    ax.add_patch(cache_box)
    ax.text(7.5, 2.5, 'Cluster Cache (CPU)', fontsize=11, fontweight='bold', ha='center')
    ax.text(7.5, 1.9, 'Centroids | Assignments | Quality Metrics', ha='center', fontsize=10)

    # Arrows
    arrow_props = dict(arrowstyle='->', color='#424242', lw=2,
                       connectionstyle='arc3,rad=0')

    # Input -> CTCA
    ax.annotate('', xy=(4, 5.75), xytext=(3, 5.75),
                arrowprops=arrow_props)

    # CTCA -> CTAA
    ax.annotate('', xy=(8, 5.75), xytext=(7, 5.75),
                arrowprops=arrow_props)
    ax.text(7.5, 6.1, 'Clusters', fontsize=9, ha='center')

    # CTAA -> Output
    ax.annotate('', xy=(12, 5.75), xytext=(11, 5.75),
                arrowprops=arrow_props)

    # CTCA <-> Cache
    ax.annotate('', xy=(5.5, 4.5), xytext=(5.5, 3),
                arrowprops=dict(arrowstyle='<->', color='#388E3C', lw=2))
    ax.text(5.2, 3.7, 'Update/\nReuse', fontsize=8, ha='center')

    # CTAA <-> Cache
    ax.annotate('', xy=(9.5, 4.5), xytext=(9.5, 3),
                arrowprops=dict(arrowstyle='<->', color='#F57C00', lw=2))
    ax.text(9.8, 3.7, 'V-centroids', fontsize=8, ha='center')

    # Timestep loop indicator
    ax.annotate('', xy=(1.75, 5), xytext=(12.75, 4.3),
                arrowprops=dict(arrowstyle='->', color='#9E9E9E', lw=1.5,
                               connectionstyle='arc3,rad=-0.3', linestyle='--'))
    ax.text(7, 4.0, 'Next timestep $t-1$', fontsize=9, ha='center', color='#757575')

    # Legend
    legend_elements = [
        mpatches.Patch(facecolor=color_ctca, edgecolor='#388E3C', label='CTCA: 7× K-means speedup'),
        mpatches.Patch(facecolor=color_ctaa, edgecolor='#F57C00', label='CTAA: ~5.7× attention reduction'),
        mpatches.Patch(facecolor=color_cache, edgecolor='#FFA000', label='CPU Cache: ~3GB memory savings'),
    ]
    ax.legend(handles=legend_elements, loc='lower left', fontsize=10)

    plt.tight_layout()
    plt.savefig('fig_overview.pdf', dpi=300, bbox_inches='tight')
    plt.savefig('fig_overview.png', dpi=300, bbox_inches='tight')
    print("Saved: fig_overview.pdf, fig_overview.png")

if __name__ == "__main__":
    create_overview_diagram()
