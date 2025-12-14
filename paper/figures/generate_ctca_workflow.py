#!/usr/bin/env python3
"""
Generate CTCA workflow diagram showing cluster reuse across timesteps.
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, Circle, Rectangle
import numpy as np

def create_ctca_workflow():
    fig, ax = plt.subplots(1, 1, figsize=(12, 7))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 7)
    ax.axis('off')

    # Colors
    color_full = '#E53935'  # Red for full clustering
    color_update = '#43A047'  # Green for centroid update only
    color_cache = '#FFF9C4'

    # Title
    ax.text(6, 6.7, 'CTCA: Cross-Timestep Cluster Amortization',
            fontsize=14, fontweight='bold', ha='center')

    # Timeline
    timesteps = ['$t=T$', '$t=T-1$', '$t=T-2$', '$t=T-3$', '...', '$t=2$', '$t=1$']
    x_positions = [1, 2.5, 4, 5.5, 7, 8.5, 10]

    # Draw timeline
    ax.plot([0.5, 10.5], [5, 5], 'k-', lw=2)
    for i, (x, t) in enumerate(zip(x_positions, timesteps)):
        ax.plot([x, x], [4.9, 5.1], 'k-', lw=2)
        ax.text(x, 4.5, t, ha='center', fontsize=10)

    # Clustering decisions
    decisions = ['Full', 'Update', 'Update', 'Update', '', 'Full', 'Update']
    colors = [color_full, color_update, color_update, color_update, 'white', color_full, color_update]

    for i, (x, decision, color) in enumerate(zip(x_positions, decisions, colors)):
        if decision:
            circle = Circle((x, 5.7), 0.25, facecolor=color, edgecolor='black', lw=1.5)
            ax.add_patch(circle)
            ax.text(x, 5.7, decision[0], ha='center', va='center', fontsize=9, fontweight='bold',
                   color='white' if color == color_full else 'black')

    # Arrow showing interval
    ax.annotate('', xy=(5.5, 6.2), xytext=(1, 6.2),
                arrowprops=dict(arrowstyle='<->', color='#1976D2', lw=2))
    ax.text(3.25, 6.5, '$\\Delta_{max}$ interval', ha='center', fontsize=10, color='#1976D2')

    # Quality check illustration
    ax.text(6, 3.8, 'Decision Logic:', fontsize=11, fontweight='bold', ha='center')

    # Decision tree
    decision_box = FancyBboxPatch((2, 2.5), 8, 1.2, boxstyle="round,pad=0.03",
                                   facecolor='#E3F2FD', edgecolor='#1976D2', linewidth=1.5)
    ax.add_patch(decision_box)

    ax.text(6, 3.3, 'if (cache empty) OR ($\\Delta \\geq \\Delta_{max}$) OR (Quality < $\\tau_q$):',
            fontsize=9, ha='center', family='monospace')
    ax.text(6, 2.9, '→ Full K-means clustering', fontsize=9, ha='center', color=color_full, fontweight='bold')
    ax.text(6, 2.6, 'else: → Centroid update only (reuse assignments)',
            fontsize=9, ha='center', color=color_update, fontweight='bold')

    # Statistics box
    stats_box = FancyBboxPatch((1.5, 0.5), 4, 1.5, boxstyle="round,pad=0.03",
                                facecolor='#E8F5E9', edgecolor='#388E3C', linewidth=1.5)
    ax.add_patch(stats_box)
    ax.text(3.5, 1.75, 'Typical Statistics', fontsize=10, fontweight='bold', ha='center')
    ax.text(3.5, 1.35, 'Full clustering: ~11%', fontsize=9, ha='center')
    ax.text(3.5, 1.0, 'Centroid update: ~89%', fontsize=9, ha='center')
    ax.text(3.5, 0.65, 'K-means speedup: 7×', fontsize=9, ha='center', fontweight='bold', color='#2E7D32')

    # Complexity box
    complexity_box = FancyBboxPatch((6.5, 0.5), 4, 1.5, boxstyle="round,pad=0.03",
                                     facecolor='#FFF3E0', edgecolor='#F57C00', linewidth=1.5)
    ax.add_patch(complexity_box)
    ax.text(8.5, 1.75, 'Complexity Reduction', fontsize=10, fontweight='bold', ha='center')
    ax.text(8.5, 1.35, 'Full: $\\mathcal{O}(N \\cdot C \\cdot I)$', fontsize=9, ha='center')
    ax.text(8.5, 1.0, 'Update: $\\mathcal{O}(N \\cdot C)$', fontsize=9, ha='center')
    ax.text(8.5, 0.65, 'Savings: $I$ iterations (~10×)', fontsize=9, ha='center', color='#E65100')

    # Legend
    legend_elements = [
        mpatches.Patch(facecolor=color_full, edgecolor='black', label='Full K-means'),
        mpatches.Patch(facecolor=color_update, edgecolor='black', label='Centroid update only'),
    ]
    ax.legend(handles=legend_elements, loc='upper right', fontsize=10)

    plt.tight_layout()
    plt.savefig('fig_ctca_workflow.pdf', dpi=300, bbox_inches='tight')
    plt.savefig('fig_ctca_workflow.png', dpi=300, bbox_inches='tight')
    print("Saved: fig_ctca_workflow.pdf, fig_ctca_workflow.png")

if __name__ == "__main__":
    create_ctca_workflow()
