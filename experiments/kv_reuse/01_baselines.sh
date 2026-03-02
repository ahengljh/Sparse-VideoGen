#!/usr/bin/env bash
# ============================================================================
# Experiment 1: Main Results
# SAP is the primary baseline (prior work). Shows KV reuse speedup on top.
# Dense is included as quality oracle (1 seed only to save compute).
# All runs use offloading. 480p / 49 frames (~2s video).
#
# Produces the data for Table: Main Results in the paper.
# Each run auto-generates <video>.run.json with wall-clock time + peak GPU MB.
# ============================================================================
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
source "${SCRIPT_DIR}/config.sh"

HEIGHT=$DEFAULT_HEIGHT; WIDTH=$DEFAULT_WIDTH
NUM_FRAMES=$DEFAULT_NUM_FRAMES; RESOLUTION=$DEFAULT_RESOLUTION
CFG_TAG="${RESOLUTION}_${NUM_FRAMES}f"

# Dense oracle uses only 1 seed (just a quality reference, not timed)
DENSE_SEED="${DENSE_SEED:-42}"

# ---------- Dense baseline (quality oracle, 1 seed) ----------
log "=== Dense Baseline (quality oracle) ==="
for pid in $PROMPT_IDS; do
    PROMPT_TEXT=$(cat "${PROJECT_ROOT}/examples/${pid}/prompt.txt")
    tag="dense"
    OUTPUT_FILE="${RESULT_ROOT}/${tag}/${CFG_TAG}/${pid}-${DENSE_SEED}.mp4"
    log "[$tag] prompt=$pid seed=$DENSE_SEED"
    run_inference --seed "$DENSE_SEED" --pattern dense
done

# ---------- SAP (primary baseline) ----------
log "=== SAP (baseline) ==="
for seed in $SEEDS; do
for pid in $PROMPT_IDS; do
    PROMPT_TEXT=$(cat "${PROJECT_ROOT}/examples/${pid}/prompt.txt")
    tag="sap"
    OUTPUT_FILE="${RESULT_ROOT}/${tag}/${CFG_TAG}/${pid}-${seed}.mp4"
    log "[$tag] prompt=$pid seed=$seed"
    run_inference --seed "$seed" "${SAP_ARGS[@]}"
done
done

# ---------- SAP + K-only reuse ----------
log "=== SAP + K-only Reuse ==="
for seed in $SEEDS; do
for pid in $PROMPT_IDS; do
    PROMPT_TEXT=$(cat "${PROJECT_ROOT}/examples/${pid}/prompt.txt")
    tag="sap_k_only"
    OUTPUT_FILE="${RESULT_ROOT}/${tag}/${CFG_TAG}/${pid}-${seed}.mp4"
    METRICS_JSONL="${METRICS_ROOT}/${tag}/${CFG_TAG}/${pid}-${seed}.jsonl"
    mkdir -p "$(dirname "$METRICS_JSONL")"
    log "[$tag] prompt=$pid seed=$seed"
    run_inference --seed "$seed" \
        "${SAP_ARGS[@]}" \
        "${KV_METRICS_ARGS[@]}" \
        --no_video_kv_reuse_v \
        --video_k_reuse_metrics_jsonl "$METRICS_JSONL"
done
done

# ---------- SAP + KV reuse (full, our method) ----------
log "=== SAP + KV Reuse ==="
for seed in $SEEDS; do
for pid in $PROMPT_IDS; do
    PROMPT_TEXT=$(cat "${PROJECT_ROOT}/examples/${pid}/prompt.txt")
    tag="sap_kv_reuse"
    OUTPUT_FILE="${RESULT_ROOT}/${tag}/${CFG_TAG}/${pid}-${seed}.mp4"
    METRICS_JSONL="${METRICS_ROOT}/${tag}/${CFG_TAG}/${pid}-${seed}.jsonl"
    mkdir -p "$(dirname "$METRICS_JSONL")"
    log "[$tag] prompt=$pid seed=$seed"
    run_inference --seed "$seed" \
        "${SAP_ARGS[@]}" \
        "${KV_METRICS_ARGS[@]}" \
        --video_k_reuse_metrics_jsonl "$METRICS_JSONL"
done
done

# ---------- Dense + KV reuse (shows KV reuse works without sparse attention) ----------
log "=== Dense + KV Reuse ==="
for seed in $SEEDS; do
for pid in $PROMPT_IDS; do
    PROMPT_TEXT=$(cat "${PROJECT_ROOT}/examples/${pid}/prompt.txt")
    tag="dense_kv_reuse"
    OUTPUT_FILE="${RESULT_ROOT}/${tag}/${CFG_TAG}/${pid}-${seed}.mp4"
    METRICS_JSONL="${METRICS_ROOT}/${tag}/${CFG_TAG}/${pid}-${seed}.jsonl"
    mkdir -p "$(dirname "$METRICS_JSONL")"
    log "[$tag] prompt=$pid seed=$seed"
    run_inference --seed "$seed" \
        --pattern dense \
        "${KV_METRICS_ARGS[@]}" \
        --video_k_reuse_metrics_jsonl "$METRICS_JSONL"
done
done

# ---------- SVG ----------
log "=== SVG ==="
for seed in $SEEDS; do
for pid in $PROMPT_IDS; do
    PROMPT_TEXT=$(cat "${PROJECT_ROOT}/examples/${pid}/prompt.txt")
    tag="svg"
    OUTPUT_FILE="${RESULT_ROOT}/${tag}/${CFG_TAG}/${pid}-${seed}.mp4"
    log "[$tag] prompt=$pid seed=$seed"
    run_inference --seed "$seed" "${SVG_ARGS[@]}"
done
done

# ---------- SVG + KV reuse ----------
log "=== SVG + KV Reuse ==="
for seed in $SEEDS; do
for pid in $PROMPT_IDS; do
    PROMPT_TEXT=$(cat "${PROJECT_ROOT}/examples/${pid}/prompt.txt")
    tag="svg_kv_reuse"
    OUTPUT_FILE="${RESULT_ROOT}/${tag}/${CFG_TAG}/${pid}-${seed}.mp4"
    METRICS_JSONL="${METRICS_ROOT}/${tag}/${CFG_TAG}/${pid}-${seed}.jsonl"
    mkdir -p "$(dirname "$METRICS_JSONL")"
    log "[$tag] prompt=$pid seed=$seed"
    run_inference --seed "$seed" \
        "${SVG_ARGS[@]}" \
        "${KV_METRICS_ARGS[@]}" \
        --video_k_reuse_metrics_jsonl "$METRICS_JSONL"
done
done

# ---------- Quality: compare all variants against SAP baseline ----------
log "=== Computing quality metrics (vs SAP baseline) ==="
for seed in $SEEDS; do
for pid in $PROMPT_IDS; do
    ref="${RESULT_ROOT}/sap/${CFG_TAG}/${pid}-${seed}.mp4"
    for tag in sap_k_only sap_kv_reuse dense_kv_reuse svg svg_kv_reuse; do
        test="${RESULT_ROOT}/${tag}/${CFG_TAG}/${pid}-${seed}.mp4"
        qout="${QUALITY_ROOT}/${tag}_vs_sap/${CFG_TAG}.jsonl"
        mkdir -p "$(dirname "$qout")"
        compute_quality "$ref" "$test" "$qout" "$pid" "$seed" || true
    done
done
done

# ---------- Quality: also compare against dense oracle ----------
log "=== Computing quality metrics (vs Dense oracle) ==="
for pid in $PROMPT_IDS; do
    ref="${RESULT_ROOT}/dense/${CFG_TAG}/${pid}-${DENSE_SEED}.mp4"
    for tag in sap sap_kv_reuse dense_kv_reuse svg svg_kv_reuse; do
        test="${RESULT_ROOT}/${tag}/${CFG_TAG}/${pid}-${DENSE_SEED}.mp4"
        qout="${QUALITY_ROOT}/${tag}_vs_dense/${CFG_TAG}.jsonl"
        mkdir -p "$(dirname "$qout")"
        compute_quality "$ref" "$test" "$qout" "$pid" "$DENSE_SEED" || true
    done
done

log "01_baselines.sh complete."
