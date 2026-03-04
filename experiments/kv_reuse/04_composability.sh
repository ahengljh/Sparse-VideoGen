#!/usr/bin/env bash
# ============================================================================
# Experiment 4: Composability with Sparse Attention
# Shows KV reuse is orthogonal to SVG / SAP sparse attention patterns.
# All runs use offloading. 480p / 49 frames.
#
# Produces the data for Table: Composability in the paper.
# ============================================================================
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
source "${SCRIPT_DIR}/config.sh"

HEIGHT=$DEFAULT_HEIGHT; WIDTH=$DEFAULT_WIDTH
NUM_FRAMES=$DEFAULT_NUM_FRAMES; RESOLUTION=$DEFAULT_RESOLUTION
CFG_TAG="${RESOLUTION}_${NUM_FRAMES}f"

COMP_PROMPTS="${COMP_PROMPTS:-1 7}"
COMP_SEEDS="${COMP_SEEDS:-42}"

# ---- SVG only ----
log "=== SVG only ==="
for seed in $COMP_SEEDS; do
for pid in $COMP_PROMPTS; do
    PROMPT_TEXT=$(cat "${PROJECT_ROOT}/examples/${pid}/prompt.txt")
    tag="svg_only"
    OUTPUT_FILE="${RESULT_ROOT}/${tag}/${CFG_TAG}/${pid}-${seed}.mp4"
    log "[$tag] prompt=$pid seed=$seed"
    run_inference --seed "$seed" "${SVG_ARGS[@]}"
done
done

# ---- SVG + KV reuse ----
log "=== SVG + KV Reuse ==="
for seed in $COMP_SEEDS; do
for pid in $COMP_PROMPTS; do
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

# ---- SAP only ----
log "=== SAP only ==="
for seed in $COMP_SEEDS; do
for pid in $COMP_PROMPTS; do
    PROMPT_TEXT=$(cat "${PROJECT_ROOT}/examples/${pid}/prompt.txt")
    tag="sap_only"
    OUTPUT_FILE="${RESULT_ROOT}/${tag}/${CFG_TAG}/${pid}-${seed}.mp4"
    log "[$tag] prompt=$pid seed=$seed"
    run_inference --seed "$seed" "${SAP_ARGS[@]}"
done
done

# ---- SAP + KV reuse ----
log "=== SAP + KV Reuse ==="
for seed in $COMP_SEEDS; do
for pid in $COMP_PROMPTS; do
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

# ---- Quality: everything vs its base (SVG or SAP) ----
log "=== Computing quality metrics ==="
for seed in $COMP_SEEDS; do
for pid in $COMP_PROMPTS; do
    # SVG+KV vs SVG
    compute_quality \
        "${RESULT_ROOT}/svg_only/${CFG_TAG}/${pid}-${seed}.mp4" \
        "${RESULT_ROOT}/svg_kv_reuse/${CFG_TAG}/${pid}-${seed}.mp4" \
        "${QUALITY_ROOT}/svg_kv_vs_svg/${CFG_TAG}.jsonl" \
        "$pid" "$seed" || true

    # SAP+KV vs SAP
    compute_quality \
        "${RESULT_ROOT}/sap_only/${CFG_TAG}/${pid}-${seed}.mp4" \
        "${RESULT_ROOT}/sap_kv_reuse/${CFG_TAG}/${pid}-${seed}.mp4" \
        "${QUALITY_ROOT}/sap_kv_vs_sap/${CFG_TAG}.jsonl" \
        "$pid" "$seed" || true
done
done

log "04_composability.sh complete."
