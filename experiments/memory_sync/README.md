# Video DiT Memory Synchronization Protocol - Experiment Suite

This experiment suite validates the claims made in the paper about the necessity of synchronization protocols for Video DiT memory management.

## Paper Section: "The Memory Management Gap for Video DiT"

### Core Claims to Validate

1. **Video DiT has unique execution characteristics** different from LLMs
2. **PyTorch's async memory management** causes failures in VDM scenarios
3. **Three failure mechanisms** occur: deferred reclamation, fragmentation, peak overlap
4. **Synchronization protocol is necessary** for reliable execution
5. **Conditional sync** achieves safety with ~15% sync trigger rate

## Experiment Structure

```
experiments/memory_sync/
├── memory_monitor.py          # GPU memory monitoring utilities
├── offload_strategies.py      # Async/Sync/Conditional transfer implementations
├── failure_experiments.py     # Three failure mechanism experiments
├── execution_pattern_analysis.py  # VDM vs LLM pattern comparison
├── quantitative_comparison.py # Main results table generation
├── conditional_sync_experiment.py # Sync strategy validation
├── real_model_experiment.py   # Integration with actual Video DiT models
├── run_all_experiments.py     # Main experiment runner
└── README.md                  # This file
```

## Quick Start

### Run All Experiments

```bash
cd experiments/memory_sync
python run_all_experiments.py --all
```

### Run Specific Experiments

```bash
# Execution pattern analysis (VDM vs LLM)
python run_all_experiments.py --execution-pattern

# Failure mechanism experiments
python run_all_experiments.py --failure-mechanisms

# Quantitative comparison (main results table)
python run_all_experiments.py --quantitative

# Conditional sync validation
python run_all_experiments.py --conditional-sync
```

### Custom Parameters

```bash
# More trials for statistical significance
python run_all_experiments.py --quantitative --num-trials 20

# Custom block size
python run_all_experiments.py --failure-mechanisms --block-size-mb 500

# Custom output directory
python run_all_experiments.py --all --output-dir ./my_results
```

## Expected Results

### 1. Execution Pattern Analysis

Validates that Video DiT has:
- **Deterministic sequential access**: Blocks execute in strict order B₁→B₂→...→Bₙ
- **No cross-step reuse**: x_t ≠ x_{t-1}, no KV-cache
- **High-frequency traversal**: 40 steps × 60 blocks = 2400 block accesses

### 2. Failure Mechanism Experiments

Demonstrates three failure modes:

| Failure Type | Description | Measurement |
|--------------|-------------|-------------|
| Deferred Reclamation | cudaFreeAsync doesn't immediately release | Allocation failure rate after del |
| Fragmentation | Repeated swaps cause unusable fragments | Total free vs largest contiguous |
| Peak Overlap | Async load/offload windows overlap | Peak memory during concurrent transfers |

### 3. Quantitative Comparison (Main Results Table)

Target format for paper:

| Strategy | OOM Rate | Peak Memory | Peak Variance |
|----------|----------|-------------|---------------|
| Pure Async | ~73% | ~23.1GB | ±2.3GB |
| Pure Sync | 0% | ~21.8GB | ±0.2GB |
| Conditional Sync | 0% | ~22.4GB | ±0.4GB |

### 4. Conditional Sync Validation

Validates that conditional sync achieves:
- **0% OOM rate** (safety guarantee)
- **~15% sync trigger rate** (efficiency)
- Proper phase-aware behavior (VAE decode, final output)

## GPU Requirements

- **Minimum**: 8GB VRAM (with reduced block sizes)
- **Recommended**: 24GB VRAM (RTX 4090, A100)
- **Ideal**: 40GB+ (A100-40GB, A6000)

The experiments auto-detect GPU memory and adjust block sizes accordingly.

## Output Format

Results are saved to timestamped directories:

```
results/
└── experiment_run_20241218_120000/
    ├── execution_pattern/
    │   └── execution_pattern_analysis.json
    ├── failure_mechanisms/
    │   └── failure_experiments_results.json
    ├── quantitative/
    │   ├── main_comparison_results.json
    │   ├── stress_test_results.json
    │   └── scaling_analysis_results.json
    ├── conditional_sync/
    │   └── conditional_sync_results.json
    ├── all_results.json
    └── paper_summary.md
```

## Paper Data Generation

The `paper_summary.md` file in each run contains formatted results ready for paper inclusion.

### Key Metrics to Report

1. **Execution Pattern**
   - Block access count: `num_steps × num_blocks`
   - KV-cache reuse ratio: 0% (VDM) vs >90% (LLM)
   - Weight amortization: 1.0x (VDM) vs Nx (LLM)

2. **Failure Mechanisms**
   - Deferred reclamation: % of immediate allocations that fail
   - Fragmentation ratio: 1 - (largest_contiguous / total_free)
   - Peak overlap: async_peak - sync_peak

3. **Strategy Comparison**
   - OOM rate per strategy
   - Peak memory (mean ± std)
   - Sync trigger rate

## Extending Experiments

### Adding New Strategies

Edit `offload_strategies.py`:

```python
class TransferStrategy(Enum):
    PURE_ASYNC = "pure_async"
    PURE_SYNC = "pure_sync"
    CONDITIONAL_SYNC = "conditional_sync"
    YOUR_NEW_STRATEGY = "your_strategy"  # Add here
```

### Custom Memory Policies

Edit `conditional_sync_experiment.py`:

```python
class ConditionalSyncPolicy:
    def should_sync(self, block_size_gb, operation):
        # Implement your policy logic
        pass
```

### Real Model Integration

Use `real_model_experiment.py` as a template to integrate with actual model inference.

## Troubleshooting

### OOM During Experiments

Reduce block size:
```bash
python run_all_experiments.py --failure-mechanisms --block-size-mb 200
```

### Slow Experiments

Reduce trials:
```bash
python run_all_experiments.py --quantitative --num-trials 3
```

### Import Errors

Ensure you're in the correct directory:
```bash
cd /path/to/Sparse-VideoGen/experiments/memory_sync
python run_all_experiments.py --all
```

## Citation

If you use these experiments in your research, please cite the paper.

## License

Same as the main Sparse-VideoGen repository.
