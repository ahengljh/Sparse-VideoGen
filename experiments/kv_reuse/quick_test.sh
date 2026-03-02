#!/usr/bin/env bash
# ============================================================================
# Quick test: minimal evaluation for early-stage validation.
# Generates ~1 second videos (25 frames at 24fps) with 1 prompt, 1 seed.
# Compares SAP baseline vs SAP+KV reuse (our method).
# Always uses offloading with 24GB GPU memory limit.
#
# Usage:
#   bash experiments/kv_reuse/quick_test.sh
#   PROMPT_IDS="1 7" bash experiments/kv_reuse/quick_test.sh   # 2 prompts
#
# Output:
#   - Videos in result/kv_reuse_paper/{sap,sap_kv_reuse}/
#   - Per-run timing + memory in .run.json
#   - Quality metrics in result/kv_reuse_paper/quality/
#   - KV reuse metrics in result/kv_reuse_paper/metrics/
#   - Summary CSVs in result/kv_reuse_paper/summary/
# ============================================================================
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

# Force quick mode
export QUICK_MODE=1
source "${SCRIPT_DIR}/config.sh"

HEIGHT=$DEFAULT_HEIGHT; WIDTH=$DEFAULT_WIDTH
NUM_FRAMES=$DEFAULT_NUM_FRAMES; RESOLUTION=$DEFAULT_RESOLUTION
CFG_TAG="${RESOLUTION}_${NUM_FRAMES}f"

log "=== Quick Test ($CFG_TAG, GPU limit ${GPU_MEMORY_LIMIT_GB}GB) ==="
log "Prompts: $PROMPT_IDS | Seeds: $SEEDS | Frames: $NUM_FRAMES"

# ---------- 1. SAP baseline ----------
log "--- SAP (baseline) ---"
for seed in $SEEDS; do
for pid in $PROMPT_IDS; do
    PROMPT_TEXT=$(cat "${PROJECT_ROOT}/examples/${pid}/prompt.txt")
    OUTPUT_FILE="${RESULT_ROOT}/sap/${CFG_TAG}/${pid}-${seed}.mp4"
    log "[sap] prompt=$pid seed=$seed"
    run_inference --seed "$seed" "${SAP_ARGS[@]}"
done
done

# ---------- 2. SAP + KV reuse (our method) ----------
log "--- SAP + KV Reuse ---"
for seed in $SEEDS; do
for pid in $PROMPT_IDS; do
    PROMPT_TEXT=$(cat "${PROJECT_ROOT}/examples/${pid}/prompt.txt")
    OUTPUT_FILE="${RESULT_ROOT}/sap_kv_reuse/${CFG_TAG}/${pid}-${seed}.mp4"
    METRICS_JSONL="${METRICS_ROOT}/sap_kv_reuse/${CFG_TAG}/${pid}-${seed}.jsonl"
    mkdir -p "$(dirname "$METRICS_JSONL")"
    log "[sap_kv_reuse] prompt=$pid seed=$seed"
    run_inference --seed "$seed" \
        "${SAP_ARGS[@]}" \
        "${KV_METRICS_ARGS[@]}" \
        --video_k_reuse_metrics_jsonl "$METRICS_JSONL"
done
done

# ---------- 3. Quality metrics (SAP+KV vs SAP) ----------
log "--- Quality Metrics ---"
for seed in $SEEDS; do
for pid in $PROMPT_IDS; do
    compute_quality \
        "${RESULT_ROOT}/sap/${CFG_TAG}/${pid}-${seed}.mp4" \
        "${RESULT_ROOT}/sap_kv_reuse/${CFG_TAG}/${pid}-${seed}.mp4" \
        "${QUALITY_ROOT}/sap_kv_reuse_vs_sap/${CFG_TAG}.jsonl" \
        "$pid" "$seed" || true
done
done

# ---------- 4. Run analysis ----------
log "--- Analysis ---"
python "${SCRIPT_DIR}/analyze_results.py" --result_root "${RESULT_ROOT}"

log "=== Quick Test Complete ==="
