#!/usr/bin/env bash
# ============================================================================
# Shared configuration for all KV reuse experiments
# Source this file from other scripts: source experiments/kv_reuse/config.sh
# ============================================================================

# --- Paths ---
export PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
export RESULT_ROOT="${PROJECT_ROOT}/result/kv_reuse_paper"
export METRICS_ROOT="${RESULT_ROOT}/metrics"
export QUALITY_ROOT="${RESULT_ROOT}/quality"

# --- Model ---
export MODEL_ID="tencent/HunyuanVideo"

# --- Prompts ---
# Use the 7 built-in example prompts for main experiments.
# Override PROMPT_IDS to use a subset (e.g., for quick smoke tests).
export PROMPT_IDS="${PROMPT_IDS:-1 2 3 4 5 6 7}"

# --- Seeds (3 seeds for statistical significance) ---
export SEEDS="${SEEDS:-42 123 456}"

# --- Inference ---
export INFER_STEPS=50

# --- Sparse attention defaults (SVG / SAP) ---
export FIRST_TIMES_FP=0.1
export FIRST_LAYERS_FP=0.03
export SVG_SPARSITY=0.25
export SVG_SAMPLED_ROWS=64
export SAP_QC=400
export SAP_KC=1000
export SAP_TOP_P=0.9
export SAP_MIN_KC_RATIO=0.10
export SAP_KMEANS_INIT=50
export SAP_KMEANS_STEP=2

# --- KV reuse defaults ---
export KV_BLOCK_SIZE=64
export KV_MAX_BLOCKS=64
export KV_WARMUP=4
export KV_START_STEP=6
export KV_INTERVAL=2
export KV_LAYER_STRIDE=8
export KV_MAX_LAYERS=8

# --- GPU ---
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

# --- Helpers ---
run_inference() {
    # Usage: run_inference <extra_args...>
    # Expects: $OUTPUT_FILE, $PROMPT_TEXT, $HEIGHT, $WIDTH, $NUM_FRAMES, $RESOLUTION
    python "${PROJECT_ROOT}/hyvideo_t2v_inference.py" \
        --model_id "${MODEL_ID}" \
        --prompt "${PROMPT_TEXT}" \
        --height "${HEIGHT}" \
        --width "${WIDTH}" \
        --num_frames "${NUM_FRAMES}" \
        --num_inference_steps "${INFER_STEPS}" \
        --resolution "${RESOLUTION}" \
        --output_file "${OUTPUT_FILE}" \
        --skip_existing \
        "$@"
}

compute_quality() {
    # Usage: compute_quality <ref_video> <test_video> <output_jsonl> <prompt_idx> <seed>
    local ref="$1" test="$2" out="$3" pidx="$4" seed="$5"
    if [[ ! -f "$ref" ]] || [[ ! -f "$test" ]]; then
        echo "SKIP quality (missing video): ref=$ref test=$test"
        return 1
    fi
    python "${PROJECT_ROOT}/svg/utils/metric.py" \
        --video1_path "$ref" \
        --video2_path "$test" \
        --output_path "$out" \
        --prompt_idx "$pidx" \
        --seed "$seed"
}

timestamp() {
    date "+%Y-%m-%d %H:%M:%S"
}

log() {
    echo "[$(timestamp)] $*"
}
