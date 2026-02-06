#!/usr/bin/env bash
# ============================================================================
# Master script: run all KV reuse experiments
#
# Usage:
#   bash experiments/kv_reuse/run_all.sh              # Run everything
#   bash experiments/kv_reuse/run_all.sh 1             # Run only experiment 1
#   bash experiments/kv_reuse/run_all.sh 1 2           # Run experiments 1 & 2
#   bash experiments/kv_reuse/run_all.sh analyze        # Only run analysis
#
# Environment overrides:
#   SEEDS="42"                      # Quick run with 1 seed
#   PROMPT_IDS="1 7"                # Quick run with 2 prompts
#   CUDA_VISIBLE_DEVICES=0,1        # Multi-GPU (one job per GPU)
#
# Recommended execution order by priority:
#   Priority 1 (main results):   01_baselines.sh
#   Priority 2 (ablations):      02_ablations.sh
#   Priority 3 (scalability):    03_scalability.sh
#   Priority 4 (composability):  04_composability.sh
#   Priority 5 (timing):         05_timing.sh
#   Final:                        analyze_results.py
# ============================================================================
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
source "${SCRIPT_DIR}/config.sh"

EXPERIMENTS="${@:-1 2 3 4 5 analyze}"

for exp in $EXPERIMENTS; do
    case "$exp" in
        1)
            log "========== EXPERIMENT 1: BASELINES =========="
            bash "${SCRIPT_DIR}/01_baselines.sh"
            ;;
        2)
            log "========== EXPERIMENT 2: ABLATIONS =========="
            bash "${SCRIPT_DIR}/02_ablations.sh"
            ;;
        3)
            log "========== EXPERIMENT 3: SCALABILITY =========="
            bash "${SCRIPT_DIR}/03_scalability.sh"
            ;;
        4)
            log "========== EXPERIMENT 4: COMPOSABILITY =========="
            bash "${SCRIPT_DIR}/04_composability.sh"
            ;;
        5)
            log "========== EXPERIMENT 5: TIMING =========="
            bash "${SCRIPT_DIR}/05_timing.sh"
            ;;
        analyze)
            log "========== ANALYSIS =========="
            python "${SCRIPT_DIR}/analyze_results.py" --result_root "${RESULT_ROOT}"
            ;;
        *)
            echo "Unknown experiment: $exp"
            echo "Valid: 1 2 3 4 5 analyze"
            exit 1
            ;;
    esac
done

log "All requested experiments complete."
