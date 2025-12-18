"""
Quantitative Comparison Experiment Framework

This module runs systematic comparisons of memory management strategies
to generate the data for the paper's quantitative results table:

| 传输策略 | OOM率 | 平均峰值内存 | 峰值方差 |
|----------|-------|-------------|----------|
| 纯异步 | 73% | 23.1GB | ±2.3GB |
| 纯同步 | 0% | 21.8GB | ±0.2GB |
| 条件同步（本文） | 0% | 22.4GB | ±0.4GB |

The experiment systematically varies:
- Block size
- Working set size
- Number of steps
- Memory pressure conditions
"""

import torch
import time
import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, asdict
import json
import os
from enum import Enum
import traceback

from memory_monitor import MemoryMonitor, get_memory_summary, save_experiment_results
from offload_strategies import (
    TransferStrategy,
    BlockOffloader,
    VideoditBlockSimulator,
    StrategyStats,
    print_comparison_table
)


@dataclass
class ExperimentConfig:
    """Configuration for a single experiment run"""
    name: str
    num_blocks: int
    num_steps: int
    block_size_mb: float
    working_set_size: int
    num_trials: int
    compute_time_ms: float = 5.0  # Simulated compute time


@dataclass
class TrialResult:
    """Result from a single trial"""
    trial_id: int
    strategy: str
    oom_occurred: bool
    oom_count: int
    peak_memory_gb: float
    memory_variance_gb: float
    sync_trigger_count: int
    sync_trigger_rate: float
    total_time_s: float
    successful_transfers: int
    failed_transfers: int


@dataclass
class ExperimentResult:
    """Aggregated results from an experiment"""
    config: ExperimentConfig
    strategy: str
    trials: List[TrialResult]

    # Aggregated metrics
    oom_rate: float
    avg_peak_memory_gb: float
    peak_memory_std_gb: float
    avg_memory_variance_gb: float
    avg_sync_trigger_rate: float
    avg_time_s: float

    def to_table_row(self) -> Dict:
        """Format for paper table"""
        return {
            'strategy': self.strategy,
            'oom_rate': f"{self.oom_rate * 100:.1f}%",
            'peak_memory_gb': f"{self.avg_peak_memory_gb:.1f}GB",
            'peak_variance_gb': f"±{self.peak_memory_std_gb:.1f}GB",
            'sync_rate': f"{self.avg_sync_trigger_rate * 100:.1f}%",
        }


class QuantitativeExperimentRunner:
    """
    Runs systematic quantitative comparison experiments.

    Generates data for the paper's main results table.
    """

    def __init__(self, device: int = 0, output_dir: str = "./results/quantitative"):
        self.device = device
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

        self.results: Dict[str, List[ExperimentResult]] = {}

    def run_single_trial(
        self,
        config: ExperimentConfig,
        strategy: TransferStrategy,
        trial_id: int,
    ) -> TrialResult:
        """Run a single trial with given configuration and strategy"""
        # Clear GPU state
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(self.device)
        time.sleep(0.5)  # Let GPU settle

        # Create simulator
        simulator = VideoditBlockSimulator(
            num_blocks=config.num_blocks,
            num_steps=config.num_steps,
            block_size_mb=config.block_size_mb,
            working_set_size=config.working_set_size,
            device=self.device,
        )

        # Run with monitoring
        monitor = MemoryMonitor(device=self.device, interval_ms=10)
        monitor.start()

        oom_count = 0
        start_time = time.time()

        try:
            stats = simulator.simulate_inference(
                strategy=strategy,
                monitor=monitor,
                compute_time_ms=config.compute_time_ms
            )
            oom_count = stats.oom_count
        except RuntimeError as e:
            if 'out of memory' in str(e).lower():
                oom_count += 1
            else:
                print(f"Error in trial: {e}")
                traceback.print_exc()

        total_time = time.time() - start_time
        monitor.stop()

        # Get results
        monitor_results = monitor.get_results(f"{config.name}_{strategy.value}_{trial_id}")

        # Calculate variance from memory samples
        if monitor_results.snapshots:
            memory_values = [s.allocated / (1024**3) for s in monitor_results.snapshots]
            memory_variance = float(np.var(memory_values))
        else:
            memory_variance = 0.0

        return TrialResult(
            trial_id=trial_id,
            strategy=strategy.value,
            oom_occurred=oom_count > 0,
            oom_count=oom_count,
            peak_memory_gb=monitor_results.peak_memory / (1024**3),
            memory_variance_gb=memory_variance,
            sync_trigger_count=monitor_results.sync_trigger_count,
            sync_trigger_rate=stats.sync_trigger_rate if hasattr(stats, 'sync_trigger_rate') else 0,
            total_time_s=total_time,
            successful_transfers=stats.successful_transfers if hasattr(stats, 'successful_transfers') else 0,
            failed_transfers=stats.failed_transfers if hasattr(stats, 'failed_transfers') else 0,
        )

    def run_experiment(
        self,
        config: ExperimentConfig,
        strategies: List[TransferStrategy] = None
    ) -> Dict[str, ExperimentResult]:
        """Run experiment for all strategies"""
        if strategies is None:
            strategies = [
                TransferStrategy.PURE_ASYNC,
                TransferStrategy.PURE_SYNC,
                TransferStrategy.CONDITIONAL_SYNC,
            ]

        print(f"\n{'='*70}")
        print(f"EXPERIMENT: {config.name}")
        print(f"{'='*70}")
        print(f"Config:")
        print(f"  Blocks: {config.num_blocks}")
        print(f"  Steps: {config.num_steps}")
        print(f"  Block size: {config.block_size_mb} MB")
        print(f"  Working set: {config.working_set_size}")
        print(f"  Trials: {config.num_trials}")

        results = {}

        for strategy in strategies:
            print(f"\n--- Strategy: {strategy.value} ---")
            trials = []

            for trial_id in range(config.num_trials):
                print(f"  Trial {trial_id + 1}/{config.num_trials}...", end=" ")
                try:
                    result = self.run_single_trial(config, strategy, trial_id)
                    trials.append(result)
                    print(f"Peak: {result.peak_memory_gb:.2f}GB, "
                          f"OOM: {result.oom_occurred}, "
                          f"Sync: {result.sync_trigger_rate:.1%}")
                except Exception as e:
                    print(f"FAILED: {e}")

                # Cleanup between trials
                torch.cuda.empty_cache()
                time.sleep(0.5)

            if trials:
                # Aggregate results
                oom_rate = sum(1 for t in trials if t.oom_occurred) / len(trials)
                peak_memories = [t.peak_memory_gb for t in trials]
                variances = [t.memory_variance_gb for t in trials]
                sync_rates = [t.sync_trigger_rate for t in trials]
                times = [t.total_time_s for t in trials]

                results[strategy.value] = ExperimentResult(
                    config=config,
                    strategy=strategy.value,
                    trials=trials,
                    oom_rate=oom_rate,
                    avg_peak_memory_gb=np.mean(peak_memories),
                    peak_memory_std_gb=np.std(peak_memories),
                    avg_memory_variance_gb=np.mean(variances),
                    avg_sync_trigger_rate=np.mean(sync_rates),
                    avg_time_s=np.mean(times),
                )

        return results

    def run_main_comparison(
        self,
        block_size_mb: float = None,
        num_trials: int = 10,
    ) -> Dict:
        """
        Run the main comparison experiment for the paper.

        Uses parameters that will stress test memory management.
        """
        # Auto-detect block size if not specified
        if block_size_mb is None:
            summary = get_memory_summary(self.device)
            total_memory_gb = summary['total_memory_gb']
            # Use block size that's significant but allows some blocks on GPU
            block_size_mb = total_memory_gb * 1024 / 20  # ~5% of GPU memory

        print(f"\n{'='*70}")
        print("MAIN QUANTITATIVE COMPARISON")
        print(f"{'='*70}")
        print(f"GPU: {torch.cuda.get_device_name(self.device)}")
        print(f"Total memory: {torch.cuda.get_device_properties(self.device).total_memory / (1024**3):.1f} GB")
        print(f"Block size: {block_size_mb:.0f} MB")

        config = ExperimentConfig(
            name="main_comparison",
            num_blocks=60,
            num_steps=10,  # Reduced for reasonable experiment time
            block_size_mb=block_size_mb,
            working_set_size=5,
            num_trials=num_trials,
            compute_time_ms=2.0,
        )

        results = self.run_experiment(config)
        self.results['main_comparison'] = results

        # Print table
        self._print_results_table(results)

        # Save results
        self._save_results('main_comparison', results)

        return results

    def run_stress_test(
        self,
        block_size_mb: float = None,
        num_trials: int = 5,
    ) -> Dict:
        """
        Run stress test with high memory pressure.

        Designed to maximize OOM occurrences for async strategy.
        """
        if block_size_mb is None:
            summary = get_memory_summary(self.device)
            # Larger blocks = more pressure
            block_size_mb = summary['total_memory_gb'] * 1024 / 10  # 10% of GPU

        config = ExperimentConfig(
            name="stress_test",
            num_blocks=30,
            num_steps=5,
            block_size_mb=block_size_mb,
            working_set_size=3,
            num_trials=num_trials,
            compute_time_ms=1.0,
        )

        print(f"\n{'='*70}")
        print("STRESS TEST (High Memory Pressure)")
        print(f"{'='*70}")

        results = self.run_experiment(config)
        self.results['stress_test'] = results

        self._print_results_table(results)
        self._save_results('stress_test', results)

        return results

    def run_scaling_analysis(
        self,
        block_sizes_mb: List[float] = None,
        num_trials: int = 3,
    ) -> Dict:
        """
        Analyze how results scale with block size.

        Generates data for understanding the relationship between
        block size and OOM risk.
        """
        if block_sizes_mb is None:
            summary = get_memory_summary(self.device)
            total_mb = summary['total_memory_gb'] * 1024
            block_sizes_mb = [total_mb / 40, total_mb / 20, total_mb / 15, total_mb / 10]

        print(f"\n{'='*70}")
        print("SCALING ANALYSIS")
        print(f"{'='*70}")

        all_results = {}

        for block_size in block_sizes_mb:
            config = ExperimentConfig(
                name=f"scale_{int(block_size)}mb",
                num_blocks=20,
                num_steps=5,
                block_size_mb=block_size,
                working_set_size=3,
                num_trials=num_trials,
                compute_time_ms=1.0,
            )

            results = self.run_experiment(config)
            all_results[int(block_size)] = results

        # Summary
        print(f"\n{'='*70}")
        print("SCALING SUMMARY")
        print(f"{'='*70}")
        print(f"{'Block Size':<12} {'Async OOM':<12} {'Sync OOM':<12} {'Cond OOM':<12}")
        print("-" * 50)

        for block_size, results in all_results.items():
            async_oom = results.get('pure_async', ExperimentResult(
                config=config, strategy='pure_async', trials=[],
                oom_rate=0, avg_peak_memory_gb=0, peak_memory_std_gb=0,
                avg_memory_variance_gb=0, avg_sync_trigger_rate=0, avg_time_s=0
            )).oom_rate * 100
            sync_oom = results.get('pure_sync', ExperimentResult(
                config=config, strategy='pure_sync', trials=[],
                oom_rate=0, avg_peak_memory_gb=0, peak_memory_std_gb=0,
                avg_memory_variance_gb=0, avg_sync_trigger_rate=0, avg_time_s=0
            )).oom_rate * 100
            cond_oom = results.get('conditional_sync', ExperimentResult(
                config=config, strategy='conditional_sync', trials=[],
                oom_rate=0, avg_peak_memory_gb=0, peak_memory_std_gb=0,
                avg_memory_variance_gb=0, avg_sync_trigger_rate=0, avg_time_s=0
            )).oom_rate * 100

            print(f"{block_size:>8} MB  {async_oom:>8.1f}%    {sync_oom:>8.1f}%    {cond_oom:>8.1f}%")

        self._save_results('scaling_analysis', all_results)
        return all_results

    def _print_results_table(self, results: Dict[str, ExperimentResult]):
        """Print results in paper table format"""
        print(f"\n{'='*80}")
        print("RESULTS TABLE (Paper Format)")
        print(f"{'='*80}")
        print(f"{'策略':<20} {'OOM率':<12} {'平均峰值内存':<15} {'峰值方差':<15} {'同步率':<12}")
        print("-" * 80)

        for strategy_name, result in results.items():
            print(f"{strategy_name:<20} "
                  f"{result.oom_rate * 100:>8.1f}%    "
                  f"{result.avg_peak_memory_gb:>10.2f} GB    "
                  f"±{result.peak_memory_std_gb:>8.2f} GB    "
                  f"{result.avg_sync_trigger_rate * 100:>8.1f}%")

        print("=" * 80)

    def _save_results(self, name: str, results: Dict):
        """Save results to JSON"""
        output_path = os.path.join(self.output_dir, f"{name}_results.json")

        # Convert to serializable format
        serializable = {}
        for key, value in results.items():
            if isinstance(value, ExperimentResult):
                serializable[key] = {
                    'strategy': value.strategy,
                    'oom_rate': value.oom_rate,
                    'avg_peak_memory_gb': value.avg_peak_memory_gb,
                    'peak_memory_std_gb': value.peak_memory_std_gb,
                    'avg_memory_variance_gb': value.avg_memory_variance_gb,
                    'avg_sync_trigger_rate': value.avg_sync_trigger_rate,
                    'avg_time_s': value.avg_time_s,
                    'num_trials': len(value.trials),
                    'trials': [asdict(t) for t in value.trials],
                }
            elif isinstance(value, dict):
                serializable[key] = self._serialize_nested(value)
            else:
                serializable[key] = value

        with open(output_path, 'w') as f:
            json.dump(serializable, f, indent=2)

        print(f"Results saved to: {output_path}")

    def _serialize_nested(self, d: Dict) -> Dict:
        """Recursively serialize nested dicts"""
        result = {}
        for k, v in d.items():
            if isinstance(v, ExperimentResult):
                result[k] = {
                    'strategy': v.strategy,
                    'oom_rate': v.oom_rate,
                    'avg_peak_memory_gb': v.avg_peak_memory_gb,
                    'peak_memory_std_gb': v.peak_memory_std_gb,
                }
            elif isinstance(v, dict):
                result[k] = self._serialize_nested(v)
            else:
                result[k] = v
        return result


def run_full_quantitative_analysis(
    output_dir: str = "./results/quantitative",
    num_trials: int = 5,
) -> Dict:
    """
    Run complete quantitative analysis suite.

    This generates all data needed for the paper's quantitative claims.
    """
    print("\n" + "="*70)
    print("FULL QUANTITATIVE ANALYSIS SUITE")
    print("="*70)

    # Print GPU info
    summary = get_memory_summary()
    print("\nGPU Information:")
    for k, v in summary.items():
        print(f"  {k}: {v}")

    runner = QuantitativeExperimentRunner(output_dir=output_dir)
    results = {}

    # 1. Main comparison
    print("\n" + "-"*50)
    print("1. MAIN COMPARISON")
    print("-"*50)
    results['main'] = runner.run_main_comparison(num_trials=num_trials)

    # 2. Stress test
    print("\n" + "-"*50)
    print("2. STRESS TEST")
    print("-"*50)
    results['stress'] = runner.run_stress_test(num_trials=num_trials)

    # 3. Scaling analysis
    print("\n" + "-"*50)
    print("3. SCALING ANALYSIS")
    print("-"*50)
    results['scaling'] = runner.run_scaling_analysis(num_trials=max(2, num_trials // 2))

    # Final summary
    print("\n" + "="*70)
    print("ANALYSIS COMPLETE")
    print("="*70)
    print(f"\nResults saved to: {output_dir}/")

    return results


if __name__ == "__main__":
    # Run with reduced trials for testing
    run_full_quantitative_analysis(num_trials=3)
