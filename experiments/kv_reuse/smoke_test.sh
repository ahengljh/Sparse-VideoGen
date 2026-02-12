#!/usr/bin/env bash
# ============================================================================
# Smoke test: quick sanity check that all configs run without errors.
# Uses 1 prompt, 1 seed, 480p, 33 frames — should finish in minutes.
# Run this FIRST before committing to the full experiment suite.
# All runs use offloading (via config.sh run_inference).
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
    run_inference --seed "$SMOKE_SEED" "$@"
    # Timing is now in the .run.json file
    if [[ -f "${OUTPUT_FILE}.run.json" ]]; then
        log "[smoke:${name}] Done. $(python3 -c "import json; d=json.load(open('${OUTPUT_FILE}.run.json')); print(f'time={d[\"wall_clock_s\"]}s peak_gpu={d[\"peak_gpu_mb\"]:.0f}MB')")"
    else
        log "[smoke:${name}] Done -> ${OUTPUT_FILE}"
    fi
}

log "=== Smoke Test ==="

# 1. SAP (baseline)
run_smoke "sap" "${SAP_ARGS[@]}"

# 2. SAP + K-only reuse
run_smoke "sap_k_only" \
    "${SAP_ARGS[@]}" \
    "${KV_REUSE_ARGS[@]}" --no_video_kv_reuse_v \
    --video_k_reuse_metrics \
    --video_k_reuse_metrics_jsonl "${SMOKE_DIR}/sap_k_only_metrics.jsonl"

# 3. SAP + KV reuse (full method)
run_smoke "sap_kv_reuse" \
    "${SAP_ARGS[@]}" \
    "${KV_REUSE_ARGS[@]}" \
    --video_k_reuse_metrics \
    --video_k_reuse_metrics_jsonl "${SMOKE_DIR}/sap_kv_reuse_metrics.jsonl"

# 4. SVG + KV reuse
run_smoke "svg_kv" \
    "${SVG_ARGS[@]}" \
    "${KV_REUSE_ARGS[@]}" \
    --video_k_reuse_metrics \
    --video_k_reuse_metrics_jsonl "${SMOKE_DIR}/svg_kv_metrics.jsonl"

# 5. Quality check
log "Quality: sap_kv_reuse vs sap..."
compute_quality \
    "${SMOKE_DIR}/sap.mp4" \
    "${SMOKE_DIR}/sap_kv_reuse.mp4" \
    "${SMOKE_DIR}/quality_kv_vs_sap.jsonl" \
    "$SMOKE_PID" "$SMOKE_SEED" || true

# Print summary
log "=== Smoke Test Results ==="
log "Run summaries:"
for rj in "${SMOKE_DIR}"/*.run.json; do
    if [[ -f "$rj" ]]; then
        log "  $(basename "$rj"): $(python3 -c "import json; d=json.load(open('$rj')); print(f'pattern={d.get(\"pattern\",\"?\")} kv_reuse={d.get(\"video_k_reuse\",False)} time={d[\"wall_clock_s\"]}s gpu={d[\"peak_gpu_mb\"]:.0f}MB')" 2>/dev/null || cat "$rj")"
    fi
done
log "Metrics:"
for mf in "${SMOKE_DIR}"/*_metrics.jsonl; do
    if [[ -f "$mf" ]]; then
        entries=$(wc -l < "$mf")
        log "  $(basename "$mf"): ${entries} entries"
    fi
done

log "Smoke test complete."
