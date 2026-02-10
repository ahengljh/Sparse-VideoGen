#!/usr/bin/env bash
# ============================================================================
# Experiment 1: Baselines & Main Comparisons
# Generates videos for Dense, K-only reuse, and KV reuse at 720p/129f.
# All runs use sliding-window offloading (--enable_offload) for 24GB GPUs.
#
# Produces the data for Table: Main Results in the paper.
# ============================================================================
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
source "${SCRIPT_DIR}/config.sh"

HEIGHT=720; WIDTH=1280; NUM_FRAMES=129; RESOLUTION="720p"

# ---------- Dense baseline (reference) ----------
log "=== Dense Baseline (with offload) ==="
for seed in $SEEDS; do
for pid in $PROMPT_IDS; do
    PROMPT_TEXT=$(cat "${PROJECT_ROOT}/examples/${pid}/prompt.txt")
    tag="dense"
    OUTPUT_FILE="${RESULT_ROOT}/${tag}/720p_129f/${pid}-${seed}.mp4"
    log "[$tag] prompt=$pid seed=$seed"
    run_inference \
        --seed "$seed" \
        --pattern dense
done
done

# ---------- K-only reuse ----------
log "=== K-only Reuse ==="
for seed in $SEEDS; do
for pid in $PROMPT_IDS; do
    PROMPT_TEXT=$(cat "${PROJECT_ROOT}/examples/${pid}/prompt.txt")
    tag="k_only_reuse"
    OUTPUT_FILE="${RESULT_ROOT}/${tag}/720p_129f/${pid}-${seed}.mp4"
    METRICS_JSONL="${METRICS_ROOT}/${tag}/720p_129f/${pid}-${seed}.jsonl"
    mkdir -p "$(dirname "$METRICS_JSONL")"
    log "[$tag] prompt=$pid seed=$seed"
    run_inference \
        --seed "$seed" \
        --pattern dense \
        "${KV_REUSE_ARGS[@]}" \
        --no_video_kv_reuse_v \
        --video_k_reuse_metrics \
        --video_k_reuse_metrics_jsonl "$METRICS_JSONL" \
        --video_k_reuse_verbose
done
done

# ---------- KV reuse (joint K+V, default) ----------
log "=== KV Reuse (K+V) ==="
for seed in $SEEDS; do
for pid in $PROMPT_IDS; do
    PROMPT_TEXT=$(cat "${PROJECT_ROOT}/examples/${pid}/prompt.txt")
    tag="kv_reuse"
    OUTPUT_FILE="${RESULT_ROOT}/${tag}/720p_129f/${pid}-${seed}.mp4"
    METRICS_JSONL="${METRICS_ROOT}/${tag}/720p_129f/${pid}-${seed}.jsonl"
    mkdir -p "$(dirname "$METRICS_JSONL")"
    log "[$tag] prompt=$pid seed=$seed"
    run_inference \
        --seed "$seed" \
        --pattern dense \
        "${KV_REUSE_ARGS[@]}" \
        --video_k_reuse_metrics \
        --video_k_reuse_metrics_jsonl "$METRICS_JSONL" \
        --video_k_reuse_verbose
done
done

# ---------- Quality: compare each reuse variant against dense ----------
log "=== Computing quality metrics ==="
for seed in $SEEDS; do
for pid in $PROMPT_IDS; do
    ref="${RESULT_ROOT}/dense/720p_129f/${pid}-${seed}.mp4"

    for tag in k_only_reuse kv_reuse; do
        test="${RESULT_ROOT}/${tag}/720p_129f/${pid}-${seed}.mp4"
        qout="${QUALITY_ROOT}/${tag}_vs_dense/720p_129f.jsonl"
        mkdir -p "$(dirname "$qout")"
        compute_quality "$ref" "$test" "$qout" "$pid" "$seed" || true
    done
done
done

log "01_baselines.sh complete."
