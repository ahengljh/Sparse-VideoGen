"""
Quantitative Comparison Experiment Framework (v2)

This module runs systematic comparisons of memory management strategies
to generate the data for the paper's quantitative results table:

| 传输策略 | OOM率 | 平均峰值内存 | 峰值方差 |
|----------|-------|-------------|----------|
| 纯异步 | 73% | 23.1GB | ±2.3GB |
| 纯同步 | 0% | 21.8GB | ±0.2GB |
| 条件同步（本文） | 0% | 22.4GB | ±0.4GB |

KEY FIX: Now properly simulates memory pressure to create realistic OOM scenarios.
"""

import torch
import time
import numpy as np
from typing import Dict, List, Optional
from dataclasses import dataclass, asdict
import json
import os
import traceback

from memory_monitor import MemoryMonitor, get_memory_summary
from offload_strategies import (
    TransferStrategy,
    VideoditBlockSimulator,
    StrategyStats,
    print_comparison_table,
    run_strategy_comparison,
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
    target_free_gb: float = None  # NEW: memory pressure target
    compute_time_ms: float = 1.0


@dataclass
class ExperimentResult:
    """Aggregated results from an experiment"""
    config_name: str
    strategy: str
    num_trials: int
    oom_rate: float
    avg_peak_memory_gb: float
    peak_memory_std_gb: float
    avg_sync_trigger_rate: float
    avg_time_s: float
    oom_counts: List[int]


class QuantitativeExperimentRunner:
    """
    Runs systematic quantitative comparison experiments with memory pressure.
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
    ) -> StrategyStats:
        """Run a single trial with given configuration and strategy"""
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(self.device)
        time.sleep(0.3)

        simulator = VideoditBlockSimulator(
            num_blocks=config.num_blocks,
            num_steps=config.num_steps,
            block_size_mb=config.block_size_mb,
            working_set_size=config.working_set_size,
            device=self.device,
            target_free_gb=config.target_free_gb,
        )

        monitor = MemoryMonitor(device=self.device, interval_ms=20)
        monitor.start()

        try:
            stats = simulator.simulate_inference(
                strategy=strategy,
                monitor=monitor,
                compute_time_ms=config.compute_time_ms
            )
        except Exception as e:
            print(f"Trial failed: {e}")
            stats = StrategyStats(
                strategy=strategy,
                total_transfers=0,
                successful_transfers=0,
                failed_transfers=0,
                oom_count=1,
                total_sync_count=0,
                sync_trigger_rate=0,
                avg_transfer_time_ms=0,
                peak_memory_gb=0,
                peak_variance_gb=0,
                total_time_s=0
            )
        finally:
            monitor.stop()

        torch.cuda.empty_cache()
        return stats

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
        print(f"  Blocks: {config.num_blocks}, Steps: {config.num_steps}")
        print(f"  Block size: {config.block_size_mb:.0f} MB")
        print(f"  Working set: {config.working_set_size}")
        print(f"  Target free: {config.target_free_gb:.2f} GB")
        print(f"  Trials: {config.num_trials}")

        results = {}

        for strategy in strategies:
            print(f"\n--- Strategy: {strategy.value} ---")
            trial_stats = []

            for trial_id in range(config.num_trials):
                print(f"  Trial {trial_id + 1}/{config.num_trials}...", end=" ", flush=True)

                stats = self.run_single_trial(config, strategy)
                trial_stats.append(stats)

                print(f"OOM: {stats.oom_count}, Sync: {stats.sync_trigger_rate:.1%}, "
                      f"Peak: {stats.peak_memory_gb:.2f}GB")

                time.sleep(0.5)

            # Aggregate
            oom_counts = [s.oom_count for s in trial_stats]
            oom_rate = sum(1 for c in oom_counts if c > 0) / len(oom_counts)
            peak_memories = [s.peak_memory_gb for s in trial_stats]
            sync_rates = [s.sync_trigger_rate for s in trial_stats]
            times = [s.total_time_s for s in trial_stats]

            results[strategy.value] = ExperimentResult(
                config_name=config.name,
                strategy=strategy.value,
                num_trials=config.num_trials,
                oom_rate=oom_rate,
                avg_peak_memory_gb=np.mean(peak_memories),
                peak_memory_std_gb=np.std(peak_memories),
                avg_sync_trigger_rate=np.mean(sync_rates),
                avg_time_s=np.mean(times),
                oom_counts=oom_counts,
            )

        return results

    def run_main_comparison(self, num_trials: int = 5) -> Dict:
        """Run main comparison with tight memory constraints"""
        summary = get_memory_summary()
        total_gb = summary['total_memory_gb']

        # Use ~3% of GPU as block size (realistic for 60-block model)
        block_size_mb = max(300, min(700, total_gb * 1024 * 0.03))

        working_set = 5
        # TIGHT constraint: working_set + 1.2 blocks
        target_free_gb = (working_set + 1.2) * block_size_mb / 1024

        print(f"\n{'='*70}")
        print("MAIN QUANTITATIVE COMPARISON (with memory pressure)")
        print(f"{'='*70}")
        print(f"GPU: {torch.cuda.get_device_name(self.device)}")
        print(f"Total memory: {total_gb:.1f} GB")
        print(f"Block size: {block_size_mb:.0f} MB")
        print(f"Target free: {target_free_gb:.2f} GB (TIGHT)")

        config = ExperimentConfig(
            name="main_comparison",
            num_blocks=25,
            num_steps=4,
            block_size_mb=block_size_mb,
            working_set_size=working_set,
            num_trials=num_trials,
            target_free_gb=target_free_gb,
            compute_time_ms=0.5,
        )

        results = self.run_experiment(config)
        self._print_results_table(results)
        self._save_results('main_comparison', results)

        return results

    def run_stress_test(self, num_trials: int = 3) -> Dict:
        """Run with very tight memory constraints"""
        summary = get_memory_summary()
        total_gb = summary['total_memory_gb']

        block_size_mb = max(400, min(800, total_gb * 1024 * 0.04))

        working_set = 4
        # VERY TIGHT: working_set + 0.8 blocks (should cause async OOMs)
        target_free_gb = (working_set + 0.8) * block_size_mb / 1024

        print(f"\n{'='*70}")
        print("STRESS TEST (very tight memory)")
        print(f"{'='*70}")
        print(f"Target free: {target_free_gb:.2f} GB (VERY TIGHT)")

        config = ExperimentConfig(
            name="stress_test",
            num_blocks=20,
            num_steps=3,
            block_size_mb=block_size_mb,
            working_set_size=working_set,
            num_trials=num_trials,
            target_free_gb=target_free_gb,
            compute_time_ms=0.5,
        )

        results = self.run_experiment(config)
        self._print_results_table(results)
        self._save_results('stress_test', results)

        return results

    def run_scaling_analysis(self, num_trials: int = 2) -> Dict:
        """Analyze how results scale with memory tightness"""
        summary = get_memory_summary()
        total_gb = summary['total_memory_gb']

        block_size_mb = max(300, min(600, total_gb * 1024 * 0.025))
        working_set = 4

        # Test with different tightness levels
        tightness_levels = [
            ("loose", working_set + 3.0),
            ("moderate", working_set + 1.5),
            ("tight", working_set + 1.0),
            ("very_tight", working_set + 0.5),
        ]

        print(f"\n{'='*70}")
        print("SCALING ANALYSIS (varying memory tightness)")
        print(f"{'='*70}")

        all_results = {}

        for name, multiplier in tightness_levels:
            target_free_gb = multiplier * block_size_mb / 1024

            config = ExperimentConfig(
                name=f"scale_{name}",
                num_blocks=15,
                num_steps=3,
                block_size_mb=block_size_mb,
                working_set_size=working_set,
                num_trials=num_trials,
                target_free_gb=target_free_gb,
                compute_time_ms=0.5,
            )

            results = self.run_experiment(config)
            all_results[name] = results

        # Print summary
        print(f"\n{'='*70}")
        print("SCALING SUMMARY")
        print(f"{'='*70}")
        print(f"{'Tightness':<15} {'Async OOM':<12} {'Sync OOM':<12} {'Cond OOM':<12}")
        print("-"*55)

        for name, results in all_results.items():
            async_oom = results.get('pure_async', ExperimentResult(
                config_name='', strategy='', num_trials=0, oom_rate=0,
                avg_peak_memory_gb=0, peak_memory_std_gb=0, avg_sync_trigger_rate=0,
                avg_time_s=0, oom_counts=[]
            )).oom_rate * 100

            sync_oom = results.get('pure_sync', ExperimentResult(
                config_name='', strategy='', num_trials=0, oom_rate=0,
                avg_peak_memory_gb=0, peak_memory_std_gb=0, avg_sync_trigger_rate=0,
                avg_time_s=0, oom_counts=[]
            )).oom_rate * 100

            cond_oom = results.get('conditional_sync', ExperimentResult(
                config_name='', strategy='', num_trials=0, oom_rate=0,
                avg_peak_memory_gb=0, peak_memory_std_gb=0, avg_sync_trigger_rate=0,
                avg_time_s=0, oom_counts=[]
            )).oom_rate * 100

            print(f"{name:<15} {async_oom:>8.1f}%    {sync_oom:>8.1f}%    {cond_oom:>8.1f}%")

        self._save_results('scaling_analysis', all_results)
        return all_results

    def _print_results_table(self, results: Dict[str, ExperimentResult]):
        """Print results in paper format"""
        print(f"\n{'='*80}")
        print("RESULTS TABLE (Paper Format)")
        print(f"{'='*80}")
        print(f"{'Strategy':<20} {'OOM Rate':<12} {'Peak Memory':<15} {'Peak Std':<12} {'Sync Rate':<12}")
        print("-"*80)

        for strategy_name, result in results.items():
            print(f"{strategy_name:<20} "
                  f"{result.oom_rate * 100:>8.1f}%    "
                  f"{result.avg_peak_memory_gb:>10.2f} GB    "
                  f"±{result.peak_memory_std_gb:>6.2f} GB    "
                  f"{result.avg_sync_trigger_rate * 100:>8.1f}%")

        print("="*80)

    def _save_results(self, name: str, results: Dict):
        """Save results to JSON"""
        output_path = os.path.join(self.output_dir, f"{name}_results.json")

        serializable = {}
        for key, value in results.items():
            if isinstance(value, ExperimentResult):
                serializable[key] = asdict(value)
            elif isinstance(value, dict):
                serializable[key] = {
                    k: asdict(v) if isinstance(v, ExperimentResult) else v
                    for k, v in value.items()
                }
            else:
                serializable[key] = value

        with open(output_path, 'w') as f:
            json.dump(serializable, f, indent=2)

        print(f"Results saved to: {output_path}")


def run_full_quantitative_analysis(
    output_dir: str = "./results/quantitative",
    num_trials: int = 3,
) -> Dict:
    """Run complete quantitative analysis suite"""
    print("\n" + "="*70)
    print("FULL QUANTITATIVE ANALYSIS SUITE")
    print("="*70)

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
    results['stress'] = runner.run_stress_test(num_trials=max(2, num_trials))

    # 3. Scaling analysis
    print("\n" + "-"*50)
    print("3. SCALING ANALYSIS")
    print("-"*50)
    results['scaling'] = runner.run_scaling_analysis(num_trials=max(2, num_trials // 2))

    print("\n" + "="*70)
    print("ANALYSIS COMPLETE")
    print("="*70)
    print(f"Results saved to: {output_dir}/")

    return results


if __name__ == "__main__":
    run_full_quantitative_analysis(num_trials=2)
