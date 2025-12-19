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
    real_block_offload: Real model block offloading integration
    run_real_workload_test: Full inference testing with sync protocols
    memory_stress_test: Lightweight stress testing for memory patterns

Quick Start:
    # Run stress test (no model download required)
    python memory_stress_test.py --strategy all

    # Run with real model inference
    python run_real_workload_test.py --model hunyuan --strategy all

    # Run all simulated experiments
    python run_all_experiments.py --all
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

from .real_block_offload import (
    SyncStrategy,
    BlockOffloadManager,
    OffloadingTransformerWrapper,
    BlockStats,
    OffloadStats,
)

__all__ = [
    # Memory monitoring
    'MemoryMonitor',
    'MemorySnapshot',
    'MemoryExperimentResult',
    'MemoryFragmentationAnalyzer',
    'get_memory_summary',
    'estimate_largest_free_block',
    'save_experiment_results',
    # Simulated offload strategies
    'TransferStrategy',
    'BlockOffloader',
    'VideoditBlockSimulator',
    'StrategyStats',
    'run_strategy_comparison',
    # Real model offloading
    'SyncStrategy',
    'BlockOffloadManager',
    'OffloadingTransformerWrapper',
    'BlockStats',
    'OffloadStats',
]
