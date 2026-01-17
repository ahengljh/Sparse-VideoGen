"""
Test script for cluster-based sparse attention

This script tests the cluster-based sparse attention implementation
to ensure it works correctly before running full video generation.
"""

import torch
from loguru import logger

from kmeans_cluster_utils import (
    batch_kmeans_euclidean,
    identify_dynamic_map,
    permute_tensor_by_labels,
    apply_inverse_permutation,
    density_calculation,
)
from sparse_attention_cluster import ClusterBasedSparseAttention


def test_kmeans():
    """Test K-means clustering"""
    logger.info("Testing K-means clustering...")

    B, N, D = 2, 100, 64
    K = 10

    x = torch.randn(B, N, D, device='cuda')

    # Test clustering
    labels, centroids, sizes, n_iter = batch_kmeans_euclidean(
        x, n_clusters=K, max_iters=10, verbose=True
    )

    logger.info(f"  Labels shape: {labels.shape}")
    logger.info(f"  Centroids shape: {centroids.shape}")
    logger.info(f"  Cluster sizes: {sizes}")
    logger.info(f"  Iterations: {n_iter}")

    # Verify all points are assigned
    assert labels.shape == (B, N)
    assert centroids.shape == (B, K, D)
    assert (labels >= 0).all() and (labels < K).all()
    assert (sizes.sum(dim=-1) == N).all()

    logger.info("✓ K-means test passed")


def test_dynamic_map():
    """Test dynamic map calculation"""
    logger.info("Testing dynamic map...")

    B, H, qc_num, kc_num, D = 1, 8, 16, 16, 64

    q_centroids = torch.randn(B, H, qc_num, D, device='cuda')
    k_centroids = torch.randn(B, H, kc_num, D, device='cuda')
    q_sizes = torch.randint(10, 50, (B, H, qc_num), device='cuda')
    k_sizes = torch.randint(10, 50, (B, H, kc_num), device='cuda')

    # Test dynamic map
    dynamic_map = identify_dynamic_map(
        q_centroids, k_centroids, q_sizes, k_sizes,
        top_p=0.9, min_kc_ratio=0.1
    )

    logger.info(f"  Dynamic map shape: {dynamic_map.shape}")
    density = density_calculation(dynamic_map, q_sizes, k_sizes)
    logger.info(f"  Density: {density.mean().item():.2%}")
    logger.info(f"  Sparsity: {(1 - density.mean().item()):.2%}")

    # Verify shape and type
    assert dynamic_map.shape == (B, H, qc_num, kc_num)
    assert dynamic_map.dtype == torch.bool
    assert 0.0 < density.mean().item() < 1.0

    logger.info("✓ Dynamic map test passed")


def test_permutation():
    """Test token permutation"""
    logger.info("Testing permutation...")

    # Test with actual usage pattern from sparse_attention_cluster
    BH, N, D = 16, 100, 64  # (cfg*num_heads, seq_len, dim)
    K = 10

    x = torch.randn(BH, N, D, device='cuda')

    # Cluster (produces BH, N labels)
    labels, _, _, _ = batch_kmeans_euclidean(x, n_clusters=K, max_iters=5)

    # In sparse_attention_cluster, we do: q_flat.unsqueeze(1), then dim=1
    # So labels is (BH, N), tensor is (BH, 1, N, D), permute along dim=1
    # But labels has 2 dims (BH, N), tensor has 4 dims
    # We need labels to match up to the permute dimension

    # Simple test: permute the last dimension where we have labels
    x_perm, sorted_indices = permute_tensor_by_labels(x, labels, dim=1)

    # Inverse permute
    x_restored = apply_inverse_permutation(x_perm, sorted_indices, dim=1)

    # Verify restoration
    diff = (x - x_restored).abs().max()
    logger.info(f"  Max difference after restore: {diff.item():.6f}")

    assert diff < 1e-5, f"Permutation restoration failed: {diff}"

    logger.info("✓ Permutation test passed")


def test_cluster_sparse_attention():
    """Test full cluster-based sparse attention"""
    logger.info("Testing cluster-based sparse attention...")

    cfg, num_heads, seq_len, dim = 1, 8, 224, 64  # Small test case
    tokens_per_frame = 16  # 14 frames × 16 tokens/frame

    q = torch.randn(cfg, num_heads, seq_len, dim, device='cuda')
    k = torch.randn(cfg, num_heads, seq_len, dim, device='cuda')
    v = torch.randn(cfg, num_heads, seq_len, dim, device='cuda')

    # Create cluster sparse attention
    cluster_attn = ClusterBasedSparseAttention(
        num_q_centroids=16,
        num_k_centroids=16,
        top_p_kmeans=0.9,
        kmeans_iter_init=5,
        kmeans_iter_step=2,
    )

    # Test forward pass
    output = cluster_attn.forward(q, k, v, layer_idx=0, tokens_per_frame=tokens_per_frame)

    # Verify output
    assert output.shape == q.shape
    logger.info(f"  Output shape: {output.shape}")

    # Get stats
    stats = cluster_attn.get_stats()
    logger.info(f"  Sparsity: {stats['avg_sparsity']:.2%}")
    logger.info(f"  Density: {stats['avg_density']:.2%}")
    logger.info(f"  K-means iters: {stats['kmeans_iters']:.1f}")

    # Test second call (should use cached centroids)
    output2 = cluster_attn.forward(q, k, v, layer_idx=0, tokens_per_frame=tokens_per_frame)
    assert output2.shape == q.shape

    stats2 = cluster_attn.get_stats()
    logger.info(f"  Second call sparsity: {stats2['avg_sparsity']:.2%}")

    logger.info("✓ Cluster sparse attention test passed")


def test_memory_efficiency():
    """Test memory efficiency compared to dense attention"""
    logger.info("Testing memory efficiency...")

    cfg, num_heads, seq_len, dim = 1, 8, 1000, 64

    q = torch.randn(cfg, num_heads, seq_len, dim, device='cuda')
    k = torch.randn(cfg, num_heads, seq_len, dim, device='cuda')
    v = torch.randn(cfg, num_heads, seq_len, dim, device='cuda')

    # Measure dense attention memory
    torch.cuda.reset_peak_memory_stats()
    scores = torch.matmul(q, k.transpose(-1, -2)) / (dim ** 0.5)
    attn = torch.softmax(scores, dim=-1)
    dense_output = torch.matmul(attn, v)
    dense_memory = torch.cuda.max_memory_allocated() / 1024**2

    logger.info(f"  Dense attention memory: {dense_memory:.2f} MB")

    # Measure cluster sparse attention memory
    torch.cuda.reset_peak_memory_stats()
    cluster_attn = ClusterBasedSparseAttention(
        num_q_centroids=32,
        num_k_centroids=32,
        top_p_kmeans=0.9,
        kmeans_iter_init=5,
    )
    sparse_output = cluster_attn.forward(q, k, v, layer_idx=0, tokens_per_frame=100)
    sparse_memory = torch.cuda.max_memory_allocated() / 1024**2

    logger.info(f"  Sparse attention memory: {sparse_memory:.2f} MB")
    logger.info(f"  Memory reduction: {(1 - sparse_memory/dense_memory)*100:.1f}%")

    logger.info("✓ Memory efficiency test passed")


def main():
    """Run all tests"""
    logger.info("=" * 60)
    logger.info("Cluster-Based Sparse Attention Test Suite")
    logger.info("=" * 60)

    if not torch.cuda.is_available():
        logger.error("CUDA not available! Tests require GPU.")
        return

    try:
        test_kmeans()
        logger.info("")

        test_dynamic_map()
        logger.info("")

        test_permutation()
        logger.info("")

        test_cluster_sparse_attention()
        logger.info("")

        test_memory_efficiency()
        logger.info("")

        logger.info("=" * 60)
        logger.info("✓ All tests passed!")
        logger.info("=" * 60)

    except Exception as e:
        logger.error(f"Test failed: {e}")
        import traceback
        traceback.print_exc()
        raise


if __name__ == "__main__":
    main()
