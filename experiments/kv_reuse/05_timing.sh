#!/usr/bin/env bash
# ============================================================================
# Experiment 5: Wall-clock Timing
# Dedicated timing runs with CUDA sync — uses a single prompt, 3 seeds,
# captures stdout to log files for extract_time.py parsing.
# Run AFTER baselines (so model downloads are cached).
# All runs use offloading.
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

# Helper: run a timing configuration across all seeds
run_timing() {
    local config_name="$1"; shift
    # Remaining args: height width resolution num_frames extra_args...
    local h="$1" w="$2" res="$3" nf="$4"; shift 4

    HEIGHT="$h"; WIDTH="$w"; RESOLUTION="$res"; NUM_FRAMES="$nf"

    for seed in $TIMING_SEEDS; do
        OUTPUT_FILE="${RESULT_ROOT}/timing/${config_name}/${TIMING_PROMPT_ID}-${seed}.mp4"
        LOGFILE="${TIMING_LOG_DIR}/${config_name}_seed${seed}.log"
        mkdir -p "$(dirname "$OUTPUT_FILE")"

        log "[timing:${config_name}] seed=$seed -> $LOGFILE"

        # Run with stdout/stderr captured for timing extraction
        run_inference --seed "$seed" "$@" 2>&1 | tee "$LOGFILE"
    done

    # Extract average time across seeds
    log "[timing:${config_name}] Extracting average time..."
    python "${PROJECT_ROOT}/svg/utils/extract_time.py" \
        -f "${TIMING_LOG_DIR}/${config_name}_seed"*.log \
        -n "$INFER_STEPS" || true
}

# ---- 720p / 129 frames ----
log "=== Timing: 720p 129f ==="
run_timing "dense_720p_129f"      720 1280 720p 129 --pattern dense
run_timing "k_only_720p_129f"     720 1280 720p 129 --pattern dense \
    "${KV_REUSE_ARGS[@]}" --no_video_kv_reuse_v
run_timing "kv_reuse_720p_129f"   720 1280 720p 129 --pattern dense \
    "${KV_REUSE_ARGS[@]}"

# ---- 720p / 65 frames ----
log "=== Timing: 720p 65f ==="
run_timing "dense_720p_65f"       720 1280 720p 65  --pattern dense
run_timing "kv_reuse_720p_65f"    720 1280 720p 65  --pattern dense \
    "${KV_REUSE_ARGS[@]}"

# ---- 720p / 33 frames ----
log "=== Timing: 720p 33f ==="
run_timing "dense_720p_33f"       720 1280 720p 33  --pattern dense
run_timing "kv_reuse_720p_33f"    720 1280 720p 33  --pattern dense \
    "${KV_REUSE_ARGS[@]}"

# ---- 480p / 129 frames ----
log "=== Timing: 480p 129f ==="
run_timing "dense_480p_129f"      480 854  480p 129 --pattern dense
run_timing "kv_reuse_480p_129f"   480 854  480p 129 --pattern dense \
    "${KV_REUSE_ARGS[@]}"

# ---- Composability timing (720p/129f) ----
log "=== Timing: Composability ==="
run_timing "svg_720p_129f"        720 1280 720p 129 "${SVG_ARGS[@]}"
run_timing "svg_kv_720p_129f"     720 1280 720p 129 \
    "${SVG_ARGS[@]}" "${KV_REUSE_ARGS[@]}"

log "05_timing.sh complete."
