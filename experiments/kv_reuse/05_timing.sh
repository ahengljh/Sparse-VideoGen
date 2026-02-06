#!/usr/bin/env bash
# ============================================================================
# Experiment 5: Wall-clock Timing
# Dedicated timing runs with CUDA sync — uses a single prompt, 3 seeds,
# captures stdout to log files for extract_time.py parsing.
# Run AFTER baselines (so model downloads are cached).
# ============================================================================
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
source "${SCRIPT_DIR}/config.sh"

# Single prompt for timing (reduces variance from prompt length)
TIMING_PROMPT_ID="${TIMING_PROMPT_ID:-7}"
TIMING_SEEDS="${TIMING_SEEDS:-42 123 456}"
TIMING_LOG_DIR="${RESULT_ROOT}/timing_logs"
mkdir -p "$TIMING_LOG_DIR"

PROMPT_TEXT=$(cat "${PROJECT_ROOT}/examples/${TIMING_PROMPT_ID}/prompt.txt")

# Configs: "tag height width resolution num_frames extra_args..."
declare -A TIMING_CONFIGS

# --- 720p ---
TIMING_CONFIGS["dense_720p_129f"]="720 1280 720p 129 --pattern dense"
TIMING_CONFIGS["k_only_720p_129f"]="720 1280 720p 129 --pattern dense --video_k_reuse --no_video_kv_reuse_v --video_k_reuse_block_size $KV_BLOCK_SIZE --video_k_reuse_max_blocks $KV_MAX_BLOCKS --video_k_reuse_warmup_steps $KV_WARMUP --video_k_reuse_start_step $KV_START_STEP --video_k_reuse_interval $KV_INTERVAL --video_k_reuse_layer_stride $KV_LAYER_STRIDE --video_k_reuse_max_layers $KV_MAX_LAYERS"
TIMING_CONFIGS["kv_reuse_720p_129f"]="720 1280 720p 129 --pattern dense --video_k_reuse --video_k_reuse_block_size $KV_BLOCK_SIZE --video_k_reuse_max_blocks $KV_MAX_BLOCKS --video_k_reuse_warmup_steps $KV_WARMUP --video_k_reuse_start_step $KV_START_STEP --video_k_reuse_interval $KV_INTERVAL --video_k_reuse_layer_stride $KV_LAYER_STRIDE --video_k_reuse_max_layers $KV_MAX_LAYERS"

# --- 480p ---
TIMING_CONFIGS["dense_480p_129f"]="480 854 480p 129 --pattern dense"
TIMING_CONFIGS["kv_reuse_480p_129f"]="480 854 480p 129 --pattern dense --video_k_reuse --video_k_reuse_block_size $KV_BLOCK_SIZE --video_k_reuse_max_blocks $KV_MAX_BLOCKS --video_k_reuse_warmup_steps $KV_WARMUP --video_k_reuse_start_step $KV_START_STEP --video_k_reuse_interval $KV_INTERVAL --video_k_reuse_layer_stride $KV_LAYER_STRIDE --video_k_reuse_max_layers $KV_MAX_LAYERS"

# --- 720p shorter ---
TIMING_CONFIGS["dense_720p_33f"]="720 1280 720p 33 --pattern dense"
TIMING_CONFIGS["kv_reuse_720p_33f"]="720 1280 720p 33 --pattern dense --video_k_reuse --video_k_reuse_block_size $KV_BLOCK_SIZE --video_k_reuse_max_blocks $KV_MAX_BLOCKS --video_k_reuse_warmup_steps $KV_WARMUP --video_k_reuse_start_step $KV_START_STEP --video_k_reuse_interval $KV_INTERVAL --video_k_reuse_layer_stride $KV_LAYER_STRIDE --video_k_reuse_max_layers $KV_MAX_LAYERS"

TIMING_CONFIGS["dense_720p_65f"]="720 1280 720p 65 --pattern dense"
TIMING_CONFIGS["kv_reuse_720p_65f"]="720 1280 720p 65 --pattern dense --video_k_reuse --video_k_reuse_block_size $KV_BLOCK_SIZE --video_k_reuse_max_blocks $KV_MAX_BLOCKS --video_k_reuse_warmup_steps $KV_WARMUP --video_k_reuse_start_step $KV_START_STEP --video_k_reuse_interval $KV_INTERVAL --video_k_reuse_layer_stride $KV_LAYER_STRIDE --video_k_reuse_max_layers $KV_MAX_LAYERS"

# --- Composability timing ---
TIMING_CONFIGS["svg_720p_129f"]="720 1280 720p 129 --pattern SVG --num_sampled_rows $SVG_SAMPLED_ROWS --sparsity $SVG_SPARSITY --first_times_fp $FIRST_TIMES_FP --first_layers_fp $FIRST_LAYERS_FP"
TIMING_CONFIGS["svg_kv_720p_129f"]="720 1280 720p 129 --pattern SVG --num_sampled_rows $SVG_SAMPLED_ROWS --sparsity $SVG_SPARSITY --first_times_fp $FIRST_TIMES_FP --first_layers_fp $FIRST_LAYERS_FP --video_k_reuse --video_k_reuse_block_size $KV_BLOCK_SIZE --video_k_reuse_max_blocks $KV_MAX_BLOCKS --video_k_reuse_warmup_steps $KV_WARMUP --video_k_reuse_start_step $KV_START_STEP --video_k_reuse_interval $KV_INTERVAL --video_k_reuse_layer_stride $KV_LAYER_STRIDE --video_k_reuse_max_layers $KV_MAX_LAYERS"

for config_name in $(echo "${!TIMING_CONFIGS[@]}" | tr ' ' '\n' | sort); do
    cfg_str="${TIMING_CONFIGS[$config_name]}"
    read -r HEIGHT WIDTH RESOLUTION NUM_FRAMES <<< "$(echo "$cfg_str" | awk '{print $1, $2, $3, $4}')"
    EXTRA_ARGS="$(echo "$cfg_str" | cut -d' ' -f5-)"

    for seed in $TIMING_SEEDS; do
        OUTPUT_FILE="${RESULT_ROOT}/timing/${config_name}/${TIMING_PROMPT_ID}-${seed}.mp4"
        LOGFILE="${TIMING_LOG_DIR}/${config_name}_seed${seed}.log"
        mkdir -p "$(dirname "$OUTPUT_FILE")"

        log "[timing:${config_name}] seed=$seed -> $LOGFILE"

        # Run with stdout/stderr captured for timing extraction
        run_inference --seed "$seed" $EXTRA_ARGS 2>&1 | tee "$LOGFILE"
    done

    # Extract average time across seeds
    log "[timing:${config_name}] Extracting average time..."
    python "${PROJECT_ROOT}/svg/utils/extract_time.py" \
        -f "${TIMING_LOG_DIR}/${config_name}_seed"*.log \
        -n "$INFER_STEPS" || true
done

log "05_timing.sh complete."
