#!/usr/bin/env bash
# ============================================================================
# Experiment 2: Ablation Studies
# Isolates each design choice using a fixed prompt subset at 720p/129f.
# ============================================================================
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
source "${SCRIPT_DIR}/config.sh"

HEIGHT=720; WIDTH=1280; NUM_FRAMES=129; RESOLUTION="720p"

# Use a subset for ablations to save compute (override with env var)
ABLATION_PROMPTS="${ABLATION_PROMPTS:-1 3 5 7}"
ABLATION_SEEDS="${ABLATION_SEEDS:-42 123 456}"

run_ablation() {
    # Usage: run_ablation <ablation_name> <extra_args...>
    local abl_name="$1"; shift
    for seed in $ABLATION_SEEDS; do
    for pid in $ABLATION_PROMPTS; do
        PROMPT_TEXT=$(cat "${PROJECT_ROOT}/examples/${pid}/prompt.txt")
        OUTPUT_FILE="${RESULT_ROOT}/ablations/${abl_name}/720p_129f/${pid}-${seed}.mp4"
        METRICS_JSONL="${METRICS_ROOT}/ablations/${abl_name}/720p_129f/${pid}-${seed}.jsonl"
        mkdir -p "$(dirname "$METRICS_JSONL")"
        log "[ablation:${abl_name}] prompt=$pid seed=$seed"
        run_inference \
            --seed "$seed" \
            --pattern dense \
            --video_k_reuse \
            --video_k_reuse_metrics \
            --video_k_reuse_metrics_jsonl "$METRICS_JSONL" \
            "$@"
    done
    done
}

quality_vs_dense() {
    local abl_name="$1"
    for seed in $ABLATION_SEEDS; do
    for pid in $ABLATION_PROMPTS; do
        ref="${RESULT_ROOT}/dense/720p_129f/${pid}-${seed}.mp4"
        test="${RESULT_ROOT}/ablations/${abl_name}/720p_129f/${pid}-${seed}.mp4"
        qout="${QUALITY_ROOT}/ablations/${abl_name}_vs_dense/720p_129f.jsonl"
        mkdir -p "$(dirname "$qout")"
        compute_quality "$ref" "$test" "$qout" "$pid" "$seed" || true
    done
    done
}

# ============================================================================
# A. K-only vs KV reuse  (K-only already in 01_baselines)
# ============================================================================
log "=== Ablation: K-only vs KV ==="
# KV default is the main result; K-only from 01_baselines
# Just ensure the ablation directory also has them for comparison scripts
run_ablation "kv_reuse_on" \
    --video_k_reuse_block_size "$KV_BLOCK_SIZE" \
    --video_k_reuse_max_blocks "$KV_MAX_BLOCKS" \
    --video_k_reuse_warmup_steps "$KV_WARMUP" \
    --video_k_reuse_start_step "$KV_START_STEP" \
    --video_k_reuse_interval "$KV_INTERVAL" \
    --video_k_reuse_layer_stride "$KV_LAYER_STRIDE" \
    --video_k_reuse_max_layers "$KV_MAX_LAYERS"

run_ablation "kv_reuse_off" \
    --no_video_kv_reuse_v \
    --video_k_reuse_block_size "$KV_BLOCK_SIZE" \
    --video_k_reuse_max_blocks "$KV_MAX_BLOCKS" \
    --video_k_reuse_warmup_steps "$KV_WARMUP" \
    --video_k_reuse_start_step "$KV_START_STEP" \
    --video_k_reuse_interval "$KV_INTERVAL" \
    --video_k_reuse_layer_stride "$KV_LAYER_STRIDE" \
    --video_k_reuse_max_layers "$KV_MAX_LAYERS"

quality_vs_dense "kv_reuse_on"
quality_vs_dense "kv_reuse_off"

# ============================================================================
# B. Block size sweep
# ============================================================================
log "=== Ablation: Block Size ==="
for bs in 16 32 64 128 256; do
    run_ablation "block_size_${bs}" \
        --video_k_reuse_block_size "$bs" \
        --video_k_reuse_max_blocks "$KV_MAX_BLOCKS" \
        --video_k_reuse_warmup_steps "$KV_WARMUP" \
        --video_k_reuse_start_step "$KV_START_STEP" \
        --video_k_reuse_interval "$KV_INTERVAL" \
        --video_k_reuse_layer_stride "$KV_LAYER_STRIDE" \
        --video_k_reuse_max_layers "$KV_MAX_LAYERS"
    quality_vs_dense "block_size_${bs}"
done

# ============================================================================
# C. Warmup steps sweep
# ============================================================================
log "=== Ablation: Warmup Steps ==="
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
    quality_vs_dense "warmup_${ws}"
done

# ============================================================================
# D. Start step sweep
# ============================================================================
log "=== Ablation: Start Step ==="
for ss in 4 6 8 12; do
    run_ablation "start_step_${ss}" \
        --video_k_reuse_block_size "$KV_BLOCK_SIZE" \
        --video_k_reuse_max_blocks "$KV_MAX_BLOCKS" \
        --video_k_reuse_warmup_steps "$KV_WARMUP" \
        --video_k_reuse_start_step "$ss" \
        --video_k_reuse_interval "$KV_INTERVAL" \
        --video_k_reuse_layer_stride "$KV_LAYER_STRIDE" \
        --video_k_reuse_max_layers "$KV_MAX_LAYERS"
    quality_vs_dense "start_step_${ss}"
done

# ============================================================================
# E. Refresh interval sweep
# ============================================================================
log "=== Ablation: Refresh Interval ==="
for iv in 1 2 4 8; do
    run_ablation "interval_${iv}" \
        --video_k_reuse_block_size "$KV_BLOCK_SIZE" \
        --video_k_reuse_max_blocks "$KV_MAX_BLOCKS" \
        --video_k_reuse_warmup_steps "$KV_WARMUP" \
        --video_k_reuse_start_step "$KV_START_STEP" \
        --video_k_reuse_interval "$iv" \
        --video_k_reuse_layer_stride "$KV_LAYER_STRIDE" \
        --video_k_reuse_max_layers "$KV_MAX_LAYERS"
    quality_vs_dense "interval_${iv}"
done

# ============================================================================
# F. Max cached blocks sweep
# ============================================================================
log "=== Ablation: Max Cached Blocks ==="
for mb in 16 32 64 128; do
    run_ablation "max_blocks_${mb}" \
        --video_k_reuse_block_size "$KV_BLOCK_SIZE" \
        --video_k_reuse_max_blocks "$mb" \
        --video_k_reuse_warmup_steps "$KV_WARMUP" \
        --video_k_reuse_start_step "$KV_START_STEP" \
        --video_k_reuse_interval "$KV_INTERVAL" \
        --video_k_reuse_layer_stride "$KV_LAYER_STRIDE" \
        --video_k_reuse_max_layers "$KV_MAX_LAYERS"
    quality_vs_dense "max_blocks_${mb}"
done

# ============================================================================
# G. EMA alpha sweep
# ============================================================================
log "=== Ablation: EMA Alpha ==="
for alpha in 0.5 0.8 1.0; do
    aname=$(echo "$alpha" | tr '.' 'p')
    run_ablation "ema_alpha_${aname}" \
        --video_k_reuse_block_size "$KV_BLOCK_SIZE" \
        --video_k_reuse_max_blocks "$KV_MAX_BLOCKS" \
        --video_k_reuse_warmup_steps "$KV_WARMUP" \
        --video_k_reuse_start_step "$KV_START_STEP" \
        --video_k_reuse_interval "$KV_INTERVAL" \
        --video_k_reuse_ema_alpha "$alpha" \
        --video_k_reuse_layer_stride "$KV_LAYER_STRIDE" \
        --video_k_reuse_max_layers "$KV_MAX_LAYERS"
    quality_vs_dense "ema_alpha_${aname}"
done

# ============================================================================
# H. Layer stride / max layers sweep
# ============================================================================
log "=== Ablation: Layer Stride & Max Layers ==="
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
    quality_vs_dense "layer_s${stride}_m${ml}"
done
done

log "02_ablations.sh complete."
