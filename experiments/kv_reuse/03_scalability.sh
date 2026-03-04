#!/usr/bin/env bash
# ============================================================================
# Experiment 3: Scalability Study
# SAP vs SAP+KV reuse across resolutions and frame counts.
# Shows speedup scales with token count. All runs use offloading.
#
# Produces the data for Figure/Table: Scalability in the paper.
# Wall-clock time and peak GPU memory are in each <video>.run.json.
# ============================================================================
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
source "${SCRIPT_DIR}/config.sh"

SCALE_PROMPTS="${SCALE_PROMPTS:-1 7}"
SCALE_SEEDS="${SCALE_SEEDS:-42}"

# Resolution configs: "tag height width resolution num_frames"
CONFIGS=(
    "720p_33f   720  1280 720p  33"
    "720p_49f   720  1280 720p  49"
    "720p_65f   720  1280 720p  65"
    "720p_129f  720  1280 720p  129"
)

for cfg_line in "${CONFIGS[@]}"; do
    read -r cfg_tag HEIGHT WIDTH RESOLUTION NUM_FRAMES <<< "$cfg_line"

    for seed in $SCALE_SEEDS; do
    for pid in $SCALE_PROMPTS; do
        PROMPT_TEXT=$(cat "${PROJECT_ROOT}/examples/${pid}/prompt.txt")

        # --- SAP baseline ---
        tag="scale_sap"
        OUTPUT_FILE="${RESULT_ROOT}/${tag}/${cfg_tag}/${pid}-${seed}.mp4"
        log "[${tag}/${cfg_tag}] prompt=$pid seed=$seed"
        run_inference --seed "$seed" "${SAP_ARGS[@]}"

        # --- SAP + KV reuse ---
        tag="scale_sap_kv"
        OUTPUT_FILE="${RESULT_ROOT}/${tag}/${cfg_tag}/${pid}-${seed}.mp4"
        METRICS_JSONL="${METRICS_ROOT}/${tag}/${cfg_tag}/${pid}-${seed}.jsonl"
        mkdir -p "$(dirname "$METRICS_JSONL")"
        log "[${tag}/${cfg_tag}] prompt=$pid seed=$seed"
        run_inference --seed "$seed" \
            "${SAP_ARGS[@]}" \
            "${KV_METRICS_ARGS[@]}" \
            --video_k_reuse_metrics_jsonl "$METRICS_JSONL"
    done
    done

    # --- Quality: SAP+KV vs SAP ---
    for seed in $SCALE_SEEDS; do
    for pid in $SCALE_PROMPTS; do
        ref="${RESULT_ROOT}/scale_sap/${cfg_tag}/${pid}-${seed}.mp4"
        test="${RESULT_ROOT}/scale_sap_kv/${cfg_tag}/${pid}-${seed}.mp4"
        qout="${QUALITY_ROOT}/scale_sap_kv_vs_sap/${cfg_tag}.jsonl"
        mkdir -p "$(dirname "$qout")"
        compute_quality "$ref" "$test" "$qout" "$pid" "$seed" || true
    done
    done
done

log "03_scalability.sh complete."
