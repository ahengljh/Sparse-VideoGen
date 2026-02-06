#!/usr/bin/env bash
# ============================================================================
# Experiment 3: Scalability Study
# Dense vs KV reuse across resolutions (480p, 720p) and frame counts (33, 65, 129)
# ============================================================================
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
source "${SCRIPT_DIR}/config.sh"

SCALE_PROMPTS="${SCALE_PROMPTS:-1 3 5 7}"
SCALE_SEEDS="${SCALE_SEEDS:-42 123 456}"

# Resolution configs: "tag height width resolution num_frames"
CONFIGS=(
    "480p_33f   480  854  480p  33"
    "480p_65f   480  854  480p  65"
    "480p_129f  480  854  480p  129"
    "720p_33f   720  1280 720p  33"
    "720p_65f   720  1280 720p  65"
    "720p_129f  720  1280 720p  129"
)

for cfg_line in "${CONFIGS[@]}"; do
    read -r cfg_tag HEIGHT WIDTH RESOLUTION NUM_FRAMES <<< "$cfg_line"

    for seed in $SCALE_SEEDS; do
    for pid in $SCALE_PROMPTS; do
        PROMPT_TEXT=$(cat "${PROJECT_ROOT}/examples/${pid}/prompt.txt")

        # --- Dense ---
        tag="scale_dense"
        OUTPUT_FILE="${RESULT_ROOT}/${tag}/${cfg_tag}/${pid}-${seed}.mp4"
        log "[${tag}/${cfg_tag}] prompt=$pid seed=$seed"
        run_inference \
            --seed "$seed" \
            --pattern dense

        # --- KV reuse ---
        tag="scale_kv_reuse"
        OUTPUT_FILE="${RESULT_ROOT}/${tag}/${cfg_tag}/${pid}-${seed}.mp4"
        METRICS_JSONL="${METRICS_ROOT}/${tag}/${cfg_tag}/${pid}-${seed}.jsonl"
        mkdir -p "$(dirname "$METRICS_JSONL")"
        log "[${tag}/${cfg_tag}] prompt=$pid seed=$seed"
        run_inference \
            --seed "$seed" \
            --pattern dense \
            --video_k_reuse \
            --video_k_reuse_block_size "$KV_BLOCK_SIZE" \
            --video_k_reuse_max_blocks "$KV_MAX_BLOCKS" \
            --video_k_reuse_warmup_steps "$KV_WARMUP" \
            --video_k_reuse_start_step "$KV_START_STEP" \
            --video_k_reuse_interval "$KV_INTERVAL" \
            --video_k_reuse_layer_stride "$KV_LAYER_STRIDE" \
            --video_k_reuse_max_layers "$KV_MAX_LAYERS" \
            --video_k_reuse_metrics \
            --video_k_reuse_metrics_jsonl "$METRICS_JSONL"
    done
    done

    # --- Quality ---
    for seed in $SCALE_SEEDS; do
    for pid in $SCALE_PROMPTS; do
        ref="${RESULT_ROOT}/scale_dense/${cfg_tag}/${pid}-${seed}.mp4"
        test="${RESULT_ROOT}/scale_kv_reuse/${cfg_tag}/${pid}-${seed}.mp4"
        qout="${QUALITY_ROOT}/scale_kv_reuse_vs_dense/${cfg_tag}.jsonl"
        mkdir -p "$(dirname "$qout")"
        compute_quality "$ref" "$test" "$qout" "$pid" "$seed" || true
    done
    done
done

log "03_scalability.sh complete."
