#!/usr/bin/env bash
# ============================================================================
# Experiment 2: Ablation Studies
# Isolates each KV reuse design choice on top of SAP at 480p/49f.
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
ABLATION_PROMPTS="${ABLATION_PROMPTS:-1 3 5 7}"
ABLATION_SEEDS="${ABLATION_SEEDS:-42 123 456}"

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
# A. K-only vs KV reuse
# ============================================================================
log "=== Ablation A: K-only vs KV ==="
run_ablation "kv_joint" \
    --video_k_reuse_block_size "$KV_BLOCK_SIZE" \
    --video_k_reuse_max_blocks "$KV_MAX_BLOCKS" \
    --video_k_reuse_warmup_steps "$KV_WARMUP" \
    --video_k_reuse_start_step "$KV_START_STEP" \
    --video_k_reuse_interval "$KV_INTERVAL" \
    --video_k_reuse_layer_stride "$KV_LAYER_STRIDE" \
    --video_k_reuse_max_layers "$KV_MAX_LAYERS"
quality_vs_sap "kv_joint"

run_ablation "k_only" \
    --no_video_kv_reuse_v \
    --video_k_reuse_block_size "$KV_BLOCK_SIZE" \
    --video_k_reuse_max_blocks "$KV_MAX_BLOCKS" \
    --video_k_reuse_warmup_steps "$KV_WARMUP" \
    --video_k_reuse_start_step "$KV_START_STEP" \
    --video_k_reuse_interval "$KV_INTERVAL" \
    --video_k_reuse_layer_stride "$KV_LAYER_STRIDE" \
    --video_k_reuse_max_layers "$KV_MAX_LAYERS"
quality_vs_sap "k_only"

# ============================================================================
# B. Block size sweep
# ============================================================================
log "=== Ablation B: Block Size ==="
for bs in 32 64 128; do
    run_ablation "block_size_${bs}" \
        --video_k_reuse_block_size "$bs" \
        --video_k_reuse_max_blocks "$KV_MAX_BLOCKS" \
        --video_k_reuse_warmup_steps "$KV_WARMUP" \
        --video_k_reuse_start_step "$KV_START_STEP" \
        --video_k_reuse_interval "$KV_INTERVAL" \
        --video_k_reuse_layer_stride "$KV_LAYER_STRIDE" \
        --video_k_reuse_max_layers "$KV_MAX_LAYERS"
    quality_vs_sap "block_size_${bs}"
done

# ============================================================================
# C. Warmup steps sweep
# ============================================================================
log "=== Ablation C: Warmup Steps ==="
for ws in 2 4 6 8; do
    ss=$((ws + 2))  # start_step = warmup + 2
    run_ablation "warmup_${ws}" \
        --video_k_reuse_block_size "$KV_BLOCK_SIZE" \
        --video_k_reuse_max_blocks "$KV_MAX_BLOCKS" \
        --video_k_reuse_warmup_steps "$ws" \
        --video_k_reuse_start_step "$ss" \
        --video_k_reuse_interval "$KV_INTERVAL" \
        --video_k_reuse_layer_stride "$KV_LAYER_STRIDE" \
        --video_k_reuse_max_layers "$KV_MAX_LAYERS"
    quality_vs_sap "warmup_${ws}"
done

# ============================================================================
# D. Refresh interval sweep
# ============================================================================
log "=== Ablation D: Refresh Interval ==="
for iv in 1 2 4 8; do
    run_ablation "interval_${iv}" \
        --video_k_reuse_block_size "$KV_BLOCK_SIZE" \
        --video_k_reuse_max_blocks "$KV_MAX_BLOCKS" \
        --video_k_reuse_warmup_steps "$KV_WARMUP" \
        --video_k_reuse_start_step "$KV_START_STEP" \
        --video_k_reuse_interval "$iv" \
        --video_k_reuse_layer_stride "$KV_LAYER_STRIDE" \
        --video_k_reuse_max_layers "$KV_MAX_LAYERS"
    quality_vs_sap "interval_${iv}"
done

# ============================================================================
# E. Max cached blocks sweep
# ============================================================================
log "=== Ablation E: Max Cached Blocks ==="
for mb in 16 32 64 128; do
    run_ablation "max_blocks_${mb}" \
        --video_k_reuse_block_size "$KV_BLOCK_SIZE" \
        --video_k_reuse_max_blocks "$mb" \
        --video_k_reuse_warmup_steps "$KV_WARMUP" \
        --video_k_reuse_start_step "$KV_START_STEP" \
        --video_k_reuse_interval "$KV_INTERVAL" \
        --video_k_reuse_layer_stride "$KV_LAYER_STRIDE" \
        --video_k_reuse_max_layers "$KV_MAX_LAYERS"
    quality_vs_sap "max_blocks_${mb}"
done

# ============================================================================
# F. Layer stride / max layers sweep
# ============================================================================
log "=== Ablation F: Layer Selection ==="
for stride in 4 8 16; do
for ml in 4 8 16; do
    run_ablation "layer_s${stride}_m${ml}" \
        --video_k_reuse_block_size "$KV_BLOCK_SIZE" \
        --video_k_reuse_max_blocks "$KV_MAX_BLOCKS" \
        --video_k_reuse_warmup_steps "$KV_WARMUP" \
        --video_k_reuse_start_step "$KV_START_STEP" \
        --video_k_reuse_interval "$KV_INTERVAL" \
        --video_k_reuse_layer_stride "$stride" \
        --video_k_reuse_max_layers "$ml"
    quality_vs_sap "layer_s${stride}_m${ml}"
done
done

log "02_ablations.sh complete."
