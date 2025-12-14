#!/usr/bin/env python3
"""
Generate temporal coherence visualization showing smooth changes across timesteps.
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, Circle, Ellipse
import numpy as np

def create_temporal_coherence():
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # === Left: Cluster evolution across timesteps ===
    ax1 = axes[0]
    ax1.set_title('(a) Cluster Stability Across Timesteps', fontsize=12, fontweight='bold')
    ax1.set_xlim(0, 10)
    ax1.set_ylim(0, 8)

    # Colors for clusters
    cluster_colors = ['#E53935', '#1E88E5', '#43A047', '#FB8C00']

    # Draw timesteps
    timesteps = ['$t=T$', '$t=T-1$', '$t=T-2$', '$t=T-3$']
    y_positions = [6.5, 4.5, 2.5, 0.5]

    np.random.seed(42)

    for t_idx, (label, y) in enumerate(zip(timesteps, y_positions)):
        ax1.text(0.3, y + 0.5, label, fontsize=10, fontweight='bold', va='center')

        # Draw cluster regions (ellipses)
        base_positions = [(2.5, 0.5), (4.5, 0.3), (6.5, 0.4), (8.5, 0.5)]

        for c_idx, (bx, by) in enumerate(base_positions):
            # Add small random drift to show temporal evolution
            drift_x = np.random.randn() * 0.1 * t_idx
            drift_y = np.random.randn() * 0.05 * t_idx

            x = bx + drift_x
            y_pos = y + by + drift_y

            # Draw cluster ellipse
            ellipse = Ellipse((x, y_pos + 0.3), 1.2, 0.6,
                             facecolor=cluster_colors[c_idx], alpha=0.4,
                             edgecolor=cluster_colors[c_idx], linewidth=2)
            ax1.add_patch(ellipse)

            # Draw some tokens within cluster
            n_tokens = 5
            for _ in range(n_tokens):
                tx = x + np.random.randn() * 0.25
                ty = y_pos + 0.3 + np.random.randn() * 0.12
                ax1.plot(tx, ty, 'o', color=cluster_colors[c_idx], markersize=4)

        # Show boundary token migration (only between adjacent timesteps)
        if t_idx > 0:
            # Arrow showing a token that migrated
            ax1.annotate('', xy=(3.3, y + 0.8), xytext=(3.0, y + 1.5),
                        arrowprops=dict(arrowstyle='->', color='gray', lw=1, alpha=0.5))

    # Key insight annotation
    ax1.text(5, -0.3, 'Key insight: Only boundary tokens shift; core assignments stable',
             ha='center', fontsize=9, style='italic')

    ax1.axis('off')

    # === Right: Quantitative stability ===
    ax2 = axes[1]
    ax2.set_title('(b) Cluster Assignment Stability', fontsize=12, fontweight='bold')

    # Simulated data showing assignment stability
    timesteps_range = np.arange(1, 51)
    np.random.seed(123)

    # High stability (most tokens stay in same cluster)
    stability = 100 - 2 * np.log1p(timesteps_range) + np.random.randn(50) * 0.5
    stability = np.clip(stability, 85, 100)

    ax2.plot(timesteps_range, stability, 'b-', lw=2, label='Assignment stability')
    ax2.fill_between(timesteps_range, stability, 85, alpha=0.3, color='blue')

    # Quality threshold line
    ax2.axhline(y=80, color='r', linestyle='--', lw=1.5, label='Quality threshold ($\\tau_q$)')

    # Reclustering points
    recluster_points = [1, 11, 21, 31, 41]
    for rp in recluster_points:
        ax2.axvline(x=rp, color='green', linestyle=':', alpha=0.7)
        ax2.plot(rp, 100, 'g^', markersize=8)

    ax2.set_xlabel('Diffusion Timestep', fontsize=10)
    ax2.set_ylabel('Cluster Assignment Stability (%)', fontsize=10)
    ax2.set_xlim(0, 51)
    ax2.set_ylim(75, 102)
    ax2.legend(loc='lower left', fontsize=9)
    ax2.grid(True, alpha=0.3)

    # Annotation for reclustering
    ax2.annotate('Full reclustering\n(every ~10 steps)', xy=(11, 100), xytext=(20, 95),
                fontsize=9, ha='center',
                arrowprops=dict(arrowstyle='->', color='green', lw=1))

    plt.tight_layout()
    plt.savefig('fig_temporal_coherence.pdf', dpi=300, bbox_inches='tight')
    plt.savefig('fig_temporal_coherence.png', dpi=300, bbox_inches='tight')
    print("Saved: fig_temporal_coherence.pdf, fig_temporal_coherence.png")

if __name__ == "__main__":
    create_temporal_coherence()
