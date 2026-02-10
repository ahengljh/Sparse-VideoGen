#!/usr/bin/env bash
# ============================================================================
# Experiment 4: Composability with Sparse Attention
# Shows KV reuse is orthogonal to SVG / SAP sparse attention patterns.
# All runs use offloading.
#
# Produces the data for Table: Composability in the paper.
# ============================================================================
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
source "${SCRIPT_DIR}/config.sh"

HEIGHT=720; WIDTH=1280; NUM_FRAMES=129; RESOLUTION="720p"

COMP_PROMPTS="${COMP_PROMPTS:-1 3 5 7}"
COMP_SEEDS="${COMP_SEEDS:-42 123 456}"

KV_METRICS_ARGS=(
    "${KV_REUSE_ARGS[@]}"
    --video_k_reuse_metrics
    --video_k_reuse_verbose
)

# ---- SVG only ----
log "=== SVG only ==="
for seed in $COMP_SEEDS; do
for pid in $COMP_PROMPTS; do
    PROMPT_TEXT=$(cat "${PROJECT_ROOT}/examples/${pid}/prompt.txt")
    tag="svg_only"
    OUTPUT_FILE="${RESULT_ROOT}/${tag}/720p_129f/${pid}-${seed}.mp4"
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
    OUTPUT_FILE="${RESULT_ROOT}/${tag}/720p_129f/${pid}-${seed}.mp4"
    METRICS_JSONL="${METRICS_ROOT}/${tag}/720p_129f/${pid}-${seed}.jsonl"
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
    OUTPUT_FILE="${RESULT_ROOT}/${tag}/720p_129f/${pid}-${seed}.mp4"
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
    OUTPUT_FILE="${RESULT_ROOT}/${tag}/720p_129f/${pid}-${seed}.mp4"
    METRICS_JSONL="${METRICS_ROOT}/${tag}/720p_129f/${pid}-${seed}.jsonl"
    mkdir -p "$(dirname "$METRICS_JSONL")"
    log "[$tag] prompt=$pid seed=$seed"
    run_inference --seed "$seed" \
        "${SAP_ARGS[@]}" \
        "${KV_METRICS_ARGS[@]}" \
        --video_k_reuse_metrics_jsonl "$METRICS_JSONL"
done
done

# ---- Quality: everything vs dense ----
log "=== Computing quality metrics ==="
for seed in $COMP_SEEDS; do
for pid in $COMP_PROMPTS; do
    ref="${RESULT_ROOT}/dense/720p_129f/${pid}-${seed}.mp4"
    for tag in svg_only svg_kv_reuse sap_only sap_kv_reuse; do
        test="${RESULT_ROOT}/${tag}/720p_129f/${pid}-${seed}.mp4"
        qout="${QUALITY_ROOT}/${tag}_vs_dense/720p_129f.jsonl"
        mkdir -p "$(dirname "$qout")"
        compute_quality "$ref" "$test" "$qout" "$pid" "$seed" || true
    done
done
done

log "04_composability.sh complete."
