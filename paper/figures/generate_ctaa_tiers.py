#!/usr/bin/env python3
"""
Generate CTAA hierarchical attention tiers visualization.
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, Rectangle
import numpy as np

def create_ctaa_tiers():
    fig, axes = plt.subplots(1, 3, figsize=(14, 5))

    # Colors
    color_full = '#E53935'  # Red
    color_centroid = '#FF9800'  # Orange
    color_skip = '#BDBDBD'  # Gray

    # === Left: Attention matrix with tiers ===
    ax1 = axes[0]
    ax1.set_title('(a) Hierarchical Attention Map', fontsize=12, fontweight='bold')

    # Create sample attention pattern
    np.random.seed(42)
    n_q, n_k = 10, 12  # Query and Key clusters

    # Generate importance-based pattern
    importance = np.random.rand(n_q, n_k)
    importance = importance / importance.sum(axis=1, keepdims=True)  # Normalize rows

    # Create tier assignments based on cumulative importance
    tiers = np.zeros((n_q, n_k), dtype=int)  # 0=skip, 1=centroid, 2=full
    for i in range(n_q):
        sorted_idx = np.argsort(importance[i])[::-1]
        cumsum = np.cumsum(importance[i, sorted_idx])
        for rank, j in enumerate(sorted_idx):
            if cumsum[rank] <= 0.7:
                tiers[i, j] = 2  # Full
            elif cumsum[rank] <= 0.95:
                tiers[i, j] = 1  # Centroid
            else:
                tiers[i, j] = 0  # Skip

    # Draw attention matrix
    for i in range(n_q):
        for j in range(n_k):
            if tiers[i, j] == 2:
                color = color_full
            elif tiers[i, j] == 1:
                color = color_centroid
            else:
                color = color_skip
            rect = Rectangle((j, n_q - 1 - i), 1, 1, facecolor=color, edgecolor='white', lw=0.5)
            ax1.add_patch(rect)

    ax1.set_xlim(0, n_k)
    ax1.set_ylim(0, n_q)
    ax1.set_xlabel('Key Clusters ($C_K$)', fontsize=10)
    ax1.set_ylabel('Query Clusters ($C_Q$)', fontsize=10)
    ax1.set_aspect('equal')

    # === Middle: Tier explanation ===
    ax2 = axes[1]
    ax2.set_title('(b) Attention Tiers', fontsize=12, fontweight='bold')
    ax2.axis('off')
    ax2.set_xlim(0, 10)
    ax2.set_ylim(0, 10)

    # Full attention tier
    full_box = FancyBboxPatch((0.5, 7), 9, 2, boxstyle="round,pad=0.05",
                               facecolor=color_full, edgecolor='darkred', linewidth=2, alpha=0.8)
    ax2.add_patch(full_box)
    ax2.text(5, 8.3, 'Full Token Attention', ha='center', fontsize=11, fontweight='bold', color='white')
    ax2.text(5, 7.5, '$p \\leq p_{full}$ (17.2%)', ha='center', fontsize=10, color='white')
    ax2.text(5, 7.1, 'Complexity: $\\mathcal{O}(B^2)$', ha='center', fontsize=9, color='white')

    # Centroid attention tier
    centroid_box = FancyBboxPatch((0.5, 4), 9, 2, boxstyle="round,pad=0.05",
                                   facecolor=color_centroid, edgecolor='darkorange', linewidth=2, alpha=0.8)
    ax2.add_patch(centroid_box)
    ax2.text(5, 5.3, 'Centroid Attention', ha='center', fontsize=11, fontweight='bold', color='white')
    ax2.text(5, 4.5, '$p_{full} < p \\leq p_{total}$ (31.7%)', ha='center', fontsize=10, color='white')
    ax2.text(5, 4.1, 'Complexity: $\\mathcal{O}(1)$', ha='center', fontsize=9, color='white')

    # Skip tier
    skip_box = FancyBboxPatch((0.5, 1), 9, 2, boxstyle="round,pad=0.05",
                               facecolor=color_skip, edgecolor='gray', linewidth=2)
    ax2.add_patch(skip_box)
    ax2.text(5, 2.3, 'Skip (No Attention)', ha='center', fontsize=11, fontweight='bold')
    ax2.text(5, 1.5, '$p > p_{total}$ (51.1%)', ha='center', fontsize=10)
    ax2.text(5, 1.1, 'Complexity: $\\mathcal{O}(0)$', ha='center', fontsize=9)

    # === Right: Computation comparison ===
    ax3 = axes[2]
    ax3.set_title('(c) Computation Savings', fontsize=12, fontweight='bold')

    # Bar chart data
    categories = ['Full\nAttention', 'Centroid\nAttention', 'Skip']
    baseline_cost = [100, 100, 100]  # Baseline would be 100% for all
    our_cost = [17.2, 0.3, 0]  # Our method: full=17.2%, centroid=~0.3% (1% of 31.7%), skip=0%

    x = np.arange(len(categories))
    width = 0.35

    bars1 = ax3.bar(x - width/2, baseline_cost, width, label='Baseline (SAP)', color='#90CAF9', edgecolor='#1976D2')
    bars2 = ax3.bar(x + width/2, our_cost, width, label='Ours (CTAA)', color='#A5D6A7', edgecolor='#388E3C')

    ax3.set_ylabel('Relative Compute Cost (%)', fontsize=10)
    ax3.set_xticks(x)
    ax3.set_xticklabels(categories, fontsize=9)
    ax3.legend(fontsize=9)
    ax3.set_ylim(0, 120)

    # Add value labels
    for bar, val in zip(bars2, our_cost):
        ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2,
                f'{val:.1f}%', ha='center', fontsize=9, fontweight='bold')

    # Add total savings annotation
    ax3.text(1, 90, 'Total effective cost:\n17.5% → 5.7× speedup',
             ha='center', fontsize=10, fontweight='bold',
             bbox=dict(boxstyle='round', facecolor='#E8F5E9', edgecolor='#388E3C'))

    plt.tight_layout()
    plt.savefig('fig_ctaa_tiers.pdf', dpi=300, bbox_inches='tight')
    plt.savefig('fig_ctaa_tiers.png', dpi=300, bbox_inches='tight')
    print("Saved: fig_ctaa_tiers.pdf, fig_ctaa_tiers.png")

if __name__ == "__main__":
    create_ctaa_tiers()
