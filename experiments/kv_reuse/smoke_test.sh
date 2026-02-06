#!/usr/bin/env bash
# ============================================================================
# Smoke test: quick sanity check that all configs run without errors.
# Uses 1 prompt, 1 seed, 480p, 33 frames — should finish in minutes.
# Run this FIRST before committing to the full experiment suite.
# ============================================================================
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
source "${SCRIPT_DIR}/config.sh"

HEIGHT=480; WIDTH=854; NUM_FRAMES=33; RESOLUTION="480p"
SMOKE_SEED=42
SMOKE_PID=7
PROMPT_TEXT=$(cat "${PROJECT_ROOT}/examples/${SMOKE_PID}/prompt.txt")

SMOKE_DIR="${RESULT_ROOT}/smoke_test"

run_smoke() {
    local name="$1"; shift
    OUTPUT_FILE="${SMOKE_DIR}/${name}.mp4"
    log "[smoke:${name}] Starting..."
    local start_time=$(date +%s)
    run_inference --seed "$SMOKE_SEED" "$@"
    local end_time=$(date +%s)
    local elapsed=$((end_time - start_time))
    log "[smoke:${name}] Done in ${elapsed}s -> ${OUTPUT_FILE}"
}

KV_BASE=(
    --video_k_reuse
    --video_k_reuse_block_size "$KV_BLOCK_SIZE"
    --video_k_reuse_max_blocks "$KV_MAX_BLOCKS"
    --video_k_reuse_warmup_steps "$KV_WARMUP"
    --video_k_reuse_start_step "$KV_START_STEP"
    --video_k_reuse_interval "$KV_INTERVAL"
    --video_k_reuse_layer_stride "$KV_LAYER_STRIDE"
    --video_k_reuse_max_layers "$KV_MAX_LAYERS"
    --video_k_reuse_metrics
)

log "=== Smoke Test ==="

# 1. Dense (ground truth)
run_smoke "dense" --pattern dense

# 2. K-only reuse
run_smoke "k_only" --pattern dense "${KV_BASE[@]}" --no_video_kv_reuse_v \
    --video_k_reuse_metrics_jsonl "${SMOKE_DIR}/k_only_metrics.jsonl"

# 3. KV reuse
run_smoke "kv_reuse" --pattern dense "${KV_BASE[@]}" \
    --video_k_reuse_metrics_jsonl "${SMOKE_DIR}/kv_reuse_metrics.jsonl"

# 4. SVG + KV reuse
run_smoke "svg_kv" \
    --pattern SVG \
    --num_sampled_rows "$SVG_SAMPLED_ROWS" \
    --sparsity "$SVG_SPARSITY" \
    --first_times_fp "$FIRST_TIMES_FP" \
    --first_layers_fp "$FIRST_LAYERS_FP" \
    "${KV_BASE[@]}" \
    --video_k_reuse_metrics_jsonl "${SMOKE_DIR}/svg_kv_metrics.jsonl"

# 5. Quality check
log "Quality: kv_reuse vs dense..."
compute_quality \
    "${SMOKE_DIR}/dense.mp4" \
    "${SMOKE_DIR}/kv_reuse.mp4" \
    "${SMOKE_DIR}/quality_kv_vs_dense.jsonl" \
    "$SMOKE_PID" "$SMOKE_SEED" || true

log "Quality: k_only vs dense..."
compute_quality \
    "${SMOKE_DIR}/dense.mp4" \
    "${SMOKE_DIR}/k_only.mp4" \
    "${SMOKE_DIR}/quality_konly_vs_dense.jsonl" \
    "$SMOKE_PID" "$SMOKE_SEED" || true

# Print summary
log "=== Smoke Test Results ==="
log "Videos:"
ls -lh "${SMOKE_DIR}"/*.mp4 2>/dev/null || echo "  (none)"
log "Metrics:"
for mf in "${SMOKE_DIR}"/*_metrics.jsonl; do
    if [[ -f "$mf" ]]; then
        entries=$(wc -l < "$mf")
        log "  $(basename "$mf"): ${entries} entries"
    fi
done
log "Quality:"
for qf in "${SMOKE_DIR}"/quality_*.jsonl; do
    if [[ -f "$qf" ]]; then
        log "  $(basename "$qf"):"
        cat "$qf"
    fi
done

log "Smoke test complete."
