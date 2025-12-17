"""
JASO: Joint Attention-Sparsity-Offloading Optimizer

This module implements the JASO framework that jointly optimizes:
1. Which layers to keep on GPU (offloading decisions)
2. What sparsity level to use per layer (attention sparsity)
3. When to recompute vs reuse (temporal amortization)

All decisions are guided by:
- Layer criticality profile C[layer, timestep]
- Memory budget constraint
- Quality threshold
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import numpy as np
import torch


@dataclass
class JASOConfig:
    """Configuration for JASO optimizer."""

    # Memory constraints
    gpu_memory_budget_gb: float = 20.0  # Target GPU memory usage
    layer_memory_gb: float = 0.5  # Approximate memory per layer on GPU

    # Quality constraints
    quality_threshold: float = 0.95  # Minimum quality (1.0 = baseline)

    # Offloading parameters
    min_layers_on_gpu: int = 4  # Always keep at least this many layers
    max_layers_on_gpu: int = 60  # Maximum layers on GPU

    # Sparsity parameters
    sparsity_levels: List[float] = field(default_factory=lambda: [0.0, 0.3, 0.5, 0.7])
    # 0.0 = full attention, 0.7 = 70% sparse

    # Temporal amortization
    enable_ctca: bool = True
    enable_ctaa: bool = True
    ctca_quality_threshold: float = 0.80
    ctaa_p_full: float = 0.7
    ctaa_p_total: float = 0.95

    # Adaptive parameters
    adaptive_recluster_interval: int = 5  # Recompute decisions every N timesteps


@dataclass
class JASODecision:
    """Decision output from JASO optimizer for a single timestep."""

    timestep: int

    # Offloading decisions: which layers to keep on GPU
    layers_on_gpu: List[int]  # Layer indices to keep on GPU
    layers_to_prefetch: List[int]  # Layers to prefetch for next step

    # Sparsity decisions: per-layer sparsity level
    layer_sparsity: Dict[int, float]  # layer_idx -> sparsity level

    # Amortization decisions
    reuse_clusters: Dict[int, bool]  # layer_idx -> whether to reuse clusters
    attention_tier: Dict[int, str]  # layer_idx -> "full" / "centroid" / "skip"

    # Estimated metrics
    estimated_memory_gb: float
    estimated_quality: float


class JASOOptimizer:
    """
    Joint Attention-Sparsity-Offloading Optimizer.

    Uses criticality profile and runtime constraints to make optimal decisions.
    """

    def __init__(
        self,
        config: JASOConfig,
        criticality_profile: Optional[np.ndarray] = None,
        num_layers: int = 60,
        num_timesteps: int = 50,
    ):
        self.config = config
        self.num_layers = num_layers
        self.num_timesteps = num_timesteps

        # Load or initialize criticality profile
        if criticality_profile is not None:
            self.criticality = criticality_profile
        else:
            # Default: uniform criticality (no prior knowledge)
            self.criticality = np.ones((num_layers, num_timesteps)) * 0.5

        # Precompute layer importance rankings per timestep
        self._precompute_rankings()

        # Runtime state
        self._current_timestep = 0
        self._previous_decisions: List[JASODecision] = []
        self._cluster_cache: Dict[int, torch.Tensor] = {}

    def _precompute_rankings(self):
        """Precompute layer importance rankings for each timestep."""
        self.layer_rankings = []
        for t in range(self.num_timesteps):
            # Sort layers by criticality (descending)
            rankings = np.argsort(-self.criticality[:, t])
            self.layer_rankings.append(rankings.tolist())

    def reset(self):
        """Reset optimizer state for new generation."""
        self._current_timestep = 0
        self._previous_decisions = []
        self._cluster_cache = {}

    def get_decision(
        self,
        timestep: int,
        current_memory_gb: float,
        quality_feedback: Optional[float] = None,
    ) -> JASODecision:
        """
        Get JASO decision for current timestep.

        Args:
            timestep: Current diffusion timestep index
            current_memory_gb: Current GPU memory usage
            quality_feedback: Optional quality metric from previous step

        Returns:
            JASODecision with all optimization decisions
        """
        self._current_timestep = timestep

        # Step 1: Determine how many layers we can fit on GPU
        available_memory = self.config.gpu_memory_budget_gb - current_memory_gb
        max_layers = min(
            int(available_memory / self.config.layer_memory_gb),
            self.config.max_layers_on_gpu
        )
        max_layers = max(max_layers, self.config.min_layers_on_gpu)

        # Step 2: Select most critical layers for GPU
        layer_rankings = self.layer_rankings[timestep]
        layers_on_gpu = layer_rankings[:max_layers]

        # Step 3: Determine sparsity levels based on criticality and GPU residency
        layer_sparsity = {}
        attention_tier = {}
        for layer_idx in range(self.num_layers):
            criticality = self.criticality[layer_idx, timestep]

            if layer_idx in layers_on_gpu:
                # On GPU: can afford lower sparsity for critical layers
                if criticality > 0.7:
                    layer_sparsity[layer_idx] = 0.0  # Full attention
                    attention_tier[layer_idx] = "full"
                elif criticality > 0.4:
                    layer_sparsity[layer_idx] = 0.3  # Light sparsity
                    attention_tier[layer_idx] = "centroid"
                else:
                    layer_sparsity[layer_idx] = 0.5  # Moderate sparsity
                    attention_tier[layer_idx] = "centroid"
            else:
                # Off GPU: must use high sparsity to compensate for offload cost
                if criticality > 0.5:
                    layer_sparsity[layer_idx] = 0.5
                    attention_tier[layer_idx] = "centroid"
                else:
                    layer_sparsity[layer_idx] = 0.7
                    attention_tier[layer_idx] = "skip"

        # Step 4: Determine cluster reuse (CTCA decisions)
        reuse_clusters = {}
        for layer_idx in range(self.num_layers):
            if timestep == 0:
                reuse_clusters[layer_idx] = False  # Always compute fresh at t=0
            elif self.config.enable_ctca:
                # Reuse if quality feedback is good and not at recluster interval
                should_recompute = (
                    timestep % self.config.adaptive_recluster_interval == 0 or
                    (quality_feedback is not None and quality_feedback < self.config.ctca_quality_threshold)
                )
                reuse_clusters[layer_idx] = not should_recompute
            else:
                reuse_clusters[layer_idx] = False

        # Step 5: Determine prefetch targets for next timestep
        if timestep + 1 < self.num_timesteps:
            next_rankings = self.layer_rankings[timestep + 1]
            # Prefetch layers that will be needed but aren't currently on GPU
            layers_to_prefetch = [
                l for l in next_rankings[:max_layers]
                if l not in layers_on_gpu
            ][:3]  # Prefetch up to 3 layers
        else:
            layers_to_prefetch = []

        # Step 6: Estimate memory and quality
        estimated_memory = len(layers_on_gpu) * self.config.layer_memory_gb
        estimated_quality = self._estimate_quality(layers_on_gpu, layer_sparsity, timestep)

        decision = JASODecision(
            timestep=timestep,
            layers_on_gpu=layers_on_gpu,
            layers_to_prefetch=layers_to_prefetch,
            layer_sparsity=layer_sparsity,
            reuse_clusters=reuse_clusters,
            attention_tier=attention_tier,
            estimated_memory_gb=estimated_memory,
            estimated_quality=estimated_quality,
        )

        self._previous_decisions.append(decision)
        return decision

    def _estimate_quality(
        self,
        layers_on_gpu: List[int],
        layer_sparsity: Dict[int, float],
        timestep: int,
    ) -> float:
        """Estimate quality based on decisions."""
        total_criticality = 0
        preserved_criticality = 0

        for layer_idx in range(self.num_layers):
            c = self.criticality[layer_idx, timestep]
            total_criticality += c

            # Quality preservation based on GPU residency and sparsity
            sparsity = layer_sparsity.get(layer_idx, 0.5)
            on_gpu = layer_idx in layers_on_gpu

            if on_gpu:
                # Better quality when on GPU
                quality_factor = 1.0 - (sparsity * 0.1)  # Small penalty for sparsity
            else:
                # Penalty for offloading
                quality_factor = 0.9 - (sparsity * 0.15)

            preserved_criticality += c * quality_factor

        return preserved_criticality / (total_criticality + 1e-10)

    def get_statistics(self) -> Dict:
        """Get optimization statistics."""
        if not self._previous_decisions:
            return {}

        avg_layers_on_gpu = np.mean([len(d.layers_on_gpu) for d in self._previous_decisions])
        avg_memory = np.mean([d.estimated_memory_gb for d in self._previous_decisions])
        avg_quality = np.mean([d.estimated_quality for d in self._previous_decisions])

        # Count attention tiers
        tier_counts = {"full": 0, "centroid": 0, "skip": 0}
        for d in self._previous_decisions:
            for tier in d.attention_tier.values():
                tier_counts[tier] += 1

        total_tiers = sum(tier_counts.values())

        return {
            "avg_layers_on_gpu": avg_layers_on_gpu,
            "avg_memory_gb": avg_memory,
            "avg_quality": avg_quality,
            "attention_tier_distribution": {
                k: v / total_tiers for k, v in tier_counts.items()
            },
            "total_decisions": len(self._previous_decisions),
        }

    def print_statistics(self):
        """Print optimization statistics."""
        stats = self.get_statistics()
        if not stats:
            print("No statistics available (no decisions made)")
            return

        print("\n" + "=" * 50)
        print("JASO Optimization Statistics")
        print("=" * 50)
        print(f"Average layers on GPU: {stats['avg_layers_on_gpu']:.1f}")
        print(f"Average memory usage: {stats['avg_memory_gb']:.2f} GB")
        print(f"Average quality estimate: {stats['avg_quality']:.3f}")
        print(f"Attention tier distribution:")
        for tier, ratio in stats['attention_tier_distribution'].items():
            print(f"  {tier}: {ratio*100:.1f}%")
        print("=" * 50 + "\n")


class JASOScheduler:
    """
    Integrates JASO optimizer with the diffusion pipeline.

    Provides hooks for the attention processor and offload manager.
    """

    def __init__(self, optimizer: JASOOptimizer):
        self.optimizer = optimizer
        self._current_decision: Optional[JASODecision] = None

    def on_timestep_begin(self, timestep: int, current_memory_gb: float):
        """Called at the beginning of each timestep."""
        self._current_decision = self.optimizer.get_decision(
            timestep=timestep,
            current_memory_gb=current_memory_gb,
        )
        return self._current_decision

    def should_layer_be_on_gpu(self, layer_idx: int) -> bool:
        """Check if a layer should be on GPU."""
        if self._current_decision is None:
            return True
        return layer_idx in self._current_decision.layers_on_gpu

    def get_layer_sparsity(self, layer_idx: int) -> float:
        """Get sparsity level for a layer."""
        if self._current_decision is None:
            return 0.0
        return self._current_decision.layer_sparsity.get(layer_idx, 0.0)

    def should_reuse_clusters(self, layer_idx: int) -> bool:
        """Check if clusters should be reused for a layer."""
        if self._current_decision is None:
            return False
        return self._current_decision.reuse_clusters.get(layer_idx, False)

    def get_attention_tier(self, layer_idx: int) -> str:
        """Get attention tier for a layer."""
        if self._current_decision is None:
            return "full"
        return self._current_decision.attention_tier.get(layer_idx, "full")

    def get_prefetch_layers(self) -> List[int]:
        """Get layers to prefetch."""
        if self._current_decision is None:
            return []
        return self._current_decision.layers_to_prefetch

    def on_generation_complete(self):
        """Called when generation is complete."""
        self.optimizer.print_statistics()
        self.optimizer.reset()


def create_jaso_optimizer(
    criticality_path: Optional[str] = None,
    num_layers: int = 60,
    num_timesteps: int = 50,
    gpu_memory_budget_gb: float = 20.0,
    quality_threshold: float = 0.95,
) -> Tuple[JASOOptimizer, JASOScheduler]:
    """
    Factory function to create JASO optimizer and scheduler.

    Args:
        criticality_path: Path to criticality profile .npy file
        num_layers: Number of transformer layers
        num_timesteps: Number of diffusion timesteps
        gpu_memory_budget_gb: GPU memory budget in GB
        quality_threshold: Minimum quality threshold

    Returns:
        Tuple of (optimizer, scheduler)
    """
    # Load criticality profile if provided
    criticality = None
    if criticality_path is not None:
        try:
            criticality = np.load(criticality_path)
            print(f"Loaded criticality profile from {criticality_path}")
        except Exception as e:
            print(f"Warning: Could not load criticality profile: {e}")

    config = JASOConfig(
        gpu_memory_budget_gb=gpu_memory_budget_gb,
        quality_threshold=quality_threshold,
    )

    optimizer = JASOOptimizer(
        config=config,
        criticality_profile=criticality,
        num_layers=num_layers,
        num_timesteps=num_timesteps,
    )

    scheduler = JASOScheduler(optimizer)

    return optimizer, scheduler
