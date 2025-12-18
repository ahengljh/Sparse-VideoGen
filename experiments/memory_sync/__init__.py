"""
Memory Synchronization Protocol Experiments for Video DiT

This package contains experiments to validate the paper's claims about
the necessity of synchronization protocols for Video DiT memory management.

Modules:
    memory_monitor: GPU memory monitoring utilities
    offload_strategies: Async/Sync/Conditional transfer implementations
    failure_experiments: Three failure mechanism experiments
    execution_pattern_analysis: VDM vs LLM pattern comparison
    quantitative_comparison: Main results table generation
    conditional_sync_experiment: Sync strategy validation
    real_model_experiment: Integration with actual Video DiT models

Quick Start:
    # Run all experiments
    python run_all_experiments.py --all

    # Run specific experiment
    python run_all_experiments.py --quantitative

    # Run with custom parameters
    python run_all_experiments.py --quantitative --num-trials 10
"""

from .memory_monitor import (
    MemoryMonitor,
    MemorySnapshot,
    MemoryExperimentResult,
    MemoryFragmentationAnalyzer,
    get_memory_summary,
    estimate_largest_free_block,
    save_experiment_results,
)

from .offload_strategies import (
    TransferStrategy,
    BlockOffloader,
    VideoditBlockSimulator,
    StrategyStats,
    run_strategy_comparison,
)

__all__ = [
    'MemoryMonitor',
    'MemorySnapshot',
    'MemoryExperimentResult',
    'MemoryFragmentationAnalyzer',
    'get_memory_summary',
    'estimate_largest_free_block',
    'save_experiment_results',
    'TransferStrategy',
    'BlockOffloader',
    'VideoditBlockSimulator',
    'StrategyStats',
    'run_strategy_comparison',
]
