"""
Experiment configurations for CTAA (Cross-Timestep Attention Amortization) evaluation.

Experiment Design:
1. Baseline: Full attention (no CTAA)
2. CTAA variants: Different (p_full, p_total) combinations
3. Ablation: CTCA only vs CTCA+CTAA

Quality Metrics:
- PSNR (Peak Signal-to-Noise Ratio)
- SSIM (Structural Similarity Index)
- LPIPS (Learned Perceptual Image Patch Similarity)

Performance Metrics:
- Wall-clock time
- GPU memory peak
- Attention compute ratio (full/centroid/skip)
"""

from dataclasses import dataclass
from typing import List, Optional
import itertools


@dataclass
class ExperimentConfig:
    """Configuration for a single experiment run."""
    name: str
    ctaa_enabled: bool
    ctaa_p_full: float  # Top-p for full attention
    ctaa_p_total: float  # Top-p including centroid attention
    description: str = ""

    def to_dict(self):
        return {
            'name': self.name,
            'ctaa_enabled': self.ctaa_enabled,
            'ctaa_p_full': self.ctaa_p_full,
            'ctaa_p_total': self.ctaa_p_total,
            'description': self.description,
        }


# Baseline configuration (no CTAA - full attention everywhere)
BASELINE = ExperimentConfig(
    name="baseline",
    ctaa_enabled=False,
    ctaa_p_full=1.0,
    ctaa_p_total=1.0,
    description="Baseline: Full attention (no CTAA optimization)"
)

# Default CTAA configuration
DEFAULT_CTAA = ExperimentConfig(
    name="ctaa_default",
    ctaa_enabled=True,
    ctaa_p_full=0.7,
    ctaa_p_total=0.95,
    description="Default CTAA: p_full=0.7, p_total=0.95"
)

# Conservative CTAA (higher quality, less speedup)
CONSERVATIVE_CTAA = ExperimentConfig(
    name="ctaa_conservative",
    ctaa_enabled=True,
    ctaa_p_full=0.85,
    ctaa_p_total=0.98,
    description="Conservative CTAA: More full attention blocks"
)

# Aggressive CTAA (lower quality, more speedup)
AGGRESSIVE_CTAA = ExperimentConfig(
    name="ctaa_aggressive",
    ctaa_enabled=True,
    ctaa_p_full=0.5,
    ctaa_p_total=0.9,
    description="Aggressive CTAA: More centroid/skip blocks"
)


def generate_parameter_sweep_configs() -> List[ExperimentConfig]:
    """
    Generate configurations for parameter sweep experiments.

    Sweep parameters:
    - p_full: [0.5, 0.6, 0.7, 0.8, 0.9]
    - p_total: [0.9, 0.95, 0.98, 1.0]

    Constraint: p_full <= p_total
    """
    configs = [BASELINE]  # Always include baseline

    p_full_values = [0.5, 0.6, 0.7, 0.8, 0.9]
    p_total_values = [0.9, 0.95, 0.98, 1.0]

    for p_full, p_total in itertools.product(p_full_values, p_total_values):
        if p_full <= p_total:  # Valid constraint
            config = ExperimentConfig(
                name=f"ctaa_pf{int(p_full*100)}_pt{int(p_total*100)}",
                ctaa_enabled=True,
                ctaa_p_full=p_full,
                ctaa_p_total=p_total,
                description=f"CTAA sweep: p_full={p_full}, p_total={p_total}"
            )
            configs.append(config)

    return configs


def generate_ablation_configs() -> List[ExperimentConfig]:
    """
    Generate configurations for ablation study.

    Compares:
    1. Baseline (no optimizations)
    2. CTCA only (clustering reuse, no attention optimization)
    3. CTCA + CTAA (full optimization)
    """
    return [
        BASELINE,
        ExperimentConfig(
            name="ctca_only",
            ctaa_enabled=False,
            ctaa_p_full=1.0,
            ctaa_p_total=1.0,
            description="CTCA only: Cluster reuse without attention optimization"
        ),
        DEFAULT_CTAA,
    ]


# Test prompts for evaluation
TEST_PROMPTS = [
    # Motion-focused prompts
    "A cat walking gracefully through a garden with flowers",
    "Ocean waves crashing on a sandy beach at sunset",
    "A person running through a city street at night",

    # Detail-focused prompts
    "Close-up of a butterfly landing on a flower",
    "Raindrops falling on a window with city lights in background",

    # Scene complexity prompts
    "A busy marketplace with people walking and vendors selling goods",
    "A forest with sunlight streaming through the trees and birds flying",
]

# Quick test prompts (for fast iteration)
QUICK_TEST_PROMPTS = [
    "A cat walking in a garden",
    "Ocean waves on a beach",
]


@dataclass
class ExperimentSuite:
    """Collection of experiments to run."""
    name: str
    configs: List[ExperimentConfig]
    prompts: List[str]
    num_inference_steps: int = 30
    resolution: str = "480p"
    num_runs: int = 1  # Number of runs per config for timing variance

    def total_experiments(self) -> int:
        return len(self.configs) * len(self.prompts) * self.num_runs


# Pre-defined experiment suites
QUICK_VALIDATION_SUITE = ExperimentSuite(
    name="quick_validation",
    configs=[BASELINE, DEFAULT_CTAA],
    prompts=QUICK_TEST_PROMPTS[:1],
    num_inference_steps=20,
    resolution="480p",
    num_runs=1,
)

PARAMETER_SWEEP_SUITE = ExperimentSuite(
    name="parameter_sweep",
    configs=generate_parameter_sweep_configs(),
    prompts=QUICK_TEST_PROMPTS,
    num_inference_steps=50,  # Match bash script for quality
    resolution="480p",
    num_runs=1,
)

QUALITY_EVALUATION_SUITE = ExperimentSuite(
    name="quality_evaluation",
    configs=[BASELINE, CONSERVATIVE_CTAA, DEFAULT_CTAA, AGGRESSIVE_CTAA],
    prompts=TEST_PROMPTS,
    num_inference_steps=50,
    resolution="720p",
    num_runs=1,
)

ABLATION_SUITE = ExperimentSuite(
    name="ablation_study",
    configs=generate_ablation_configs(),
    prompts=TEST_PROMPTS[:3],
    num_inference_steps=30,
    resolution="480p",
    num_runs=3,  # Multiple runs for timing variance
)
