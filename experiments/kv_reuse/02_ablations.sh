#!/usr/bin/env bash
# ============================================================================
# Experiment 2: Ablation Studies
# Isolates each attention output caching design choice on top of SAP at 720p/49f.
# Reference: SAP baseline from 01_baselines.
#
# Produces the data for Table: Ablation Study in the paper.
# ============================================================================
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
source "${SCRIPT_DIR}/config.sh"

HEIGHT=$DEFAULT_HEIGHT; WIDTH=$DEFAULT_WIDTH
NUM_FRAMES=$DEFAULT_NUM_FRAMES; RESOLUTION=$DEFAULT_RESOLUTION
CFG_TAG="${RESOLUTION}_${NUM_FRAMES}f"

# Use a subset for ablations to save compute
ABLATION_PROMPTS="${ABLATION_PROMPTS:-1 7}"
ABLATION_SEEDS="${ABLATION_SEEDS:-42}"

run_ablation() {
    # Usage: run_ablation <ablation_name> <extra_kv_args...>
    local abl_name="$1"; shift
    for seed in $ABLATION_SEEDS; do
    for pid in $ABLATION_PROMPTS; do
        PROMPT_TEXT=$(cat "${PROJECT_ROOT}/examples/${pid}/prompt.txt")
        OUTPUT_FILE="${RESULT_ROOT}/ablations/${abl_name}/${CFG_TAG}/${pid}-${seed}.mp4"
        METRICS_JSONL="${METRICS_ROOT}/ablations/${abl_name}/${CFG_TAG}/${pid}-${seed}.jsonl"
        mkdir -p "$(dirname "$METRICS_JSONL")"
        log "[ablation:${abl_name}] prompt=$pid seed=$seed"
        run_inference \
            --seed "$seed" \
            "${SAP_ARGS[@]}" \
            --video_k_reuse \
            --video_k_reuse_metrics \
            --video_k_reuse_metrics_jsonl "$METRICS_JSONL" \
            "$@"
    done
    done
}

quality_vs_sap() {
    local abl_name="$1"
    for seed in $ABLATION_SEEDS; do
    for pid in $ABLATION_PROMPTS; do
        ref="${RESULT_ROOT}/sap/${CFG_TAG}/${pid}-${seed}.mp4"
        test="${RESULT_ROOT}/ablations/${abl_name}/${CFG_TAG}/${pid}-${seed}.mp4"
        qout="${QUALITY_ROOT}/ablations/${abl_name}_vs_sap/${CFG_TAG}.jsonl"
        mkdir -p "$(dirname "$qout")"
        compute_quality "$ref" "$test" "$qout" "$pid" "$seed" || true
    done
    done
}

# ============================================================================
# A. Refresh interval sweep (controls reuse frequency)
# ============================================================================
log "=== Ablation A: Refresh Interval ==="
for iv in 1 2 3 4; do
    run_ablation "interval_${iv}" \
        --video_k_reuse_warmup_steps "$KV_WARMUP" \
        --video_k_reuse_start_step "$KV_START_STEP" \
        --video_k_reuse_interval "$iv" \
        --video_k_reuse_layer_stride "$KV_LAYER_STRIDE" \
        --video_k_reuse_max_layers "$KV_MAX_LAYERS"
    quality_vs_sap "interval_${iv}"
done

# ============================================================================
# B. Layer stride sweep (controls which layers cache)
# ============================================================================
log "=== Ablation B: Layer Stride ==="
for stride in 1 2 4 8; do
    run_ablation "layer_stride_${stride}" \
        --video_k_reuse_warmup_steps "$KV_WARMUP" \
        --video_k_reuse_start_step "$KV_START_STEP" \
        --video_k_reuse_interval "$KV_INTERVAL" \
        --video_k_reuse_layer_stride "$stride" \
        --video_k_reuse_max_layers "$KV_MAX_LAYERS"
    quality_vs_sap "layer_stride_${stride}"
done

# ============================================================================
# C. Start step sweep (controls when caching begins)
# ============================================================================
log "=== Ablation C: Start Step ==="
for ss in 2 4 6 10; do
    ws=$((ss > 2 ? ss - 2 : 2))  # warmup = start_step - 2 (minimum 2)
    run_ablation "start_step_${ss}" \
        --video_k_reuse_warmup_steps "$ws" \
        --video_k_reuse_start_step "$ss" \
        --video_k_reuse_interval "$KV_INTERVAL" \
        --video_k_reuse_layer_stride "$KV_LAYER_STRIDE" \
        --video_k_reuse_max_layers "$KV_MAX_LAYERS"
    quality_vs_sap "start_step_${ss}"
done

# ============================================================================
# D. Max layers sweep (controls how many layers cache)
# ============================================================================
log "=== Ablation D: Max Layers ==="
for ml in 10 20 30 60; do
    run_ablation "max_layers_${ml}" \
        --video_k_reuse_warmup_steps "$KV_WARMUP" \
        --video_k_reuse_start_step "$KV_START_STEP" \
        --video_k_reuse_interval "$KV_INTERVAL" \
        --video_k_reuse_layer_stride "$KV_LAYER_STRIDE" \
        --video_k_reuse_max_layers "$ml"
    quality_vs_sap "max_layers_${ml}"
done

log "02_ablations.sh complete."
