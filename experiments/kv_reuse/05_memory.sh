#!/usr/bin/env bash
# ============================================================================
# Experiment 5: Memory Feasibility ("Can It Run?")
# Demonstrates that offloading is required for 24GB GPUs.
# Without offloading → OOM. With offloading → runs.
# With offloading + sparse + KV reuse → runs faster.
#
# Produces the data for Table: Memory Feasibility in the paper.
# Uses a single prompt and seed to minimize wasted compute on OOM runs.
# ============================================================================
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
source "${SCRIPT_DIR}/config.sh"

MEM_PID="${MEM_PID:-7}"
MEM_SEED="${MEM_SEED:-42}"
PROMPT_TEXT=$(cat "${PROJECT_ROOT}/examples/${MEM_PID}/prompt.txt")
MEM_DIR="${RESULT_ROOT}/memory"
MEM_LOG="${MEM_DIR}/feasibility.log"
mkdir -p "$MEM_DIR"

# Configs to test: "tag height width resolution num_frames"
MEM_CONFIGS=(
    "720p_49f   720  1280 720p  49"
    "720p_129f  720  1280 720p  129"
)

log "=== Memory Feasibility Study ===" | tee "$MEM_LOG"
log "GPU: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo 'unknown')" | tee -a "$MEM_LOG"

for cfg_line in "${MEM_CONFIGS[@]}"; do
    read -r cfg_tag HEIGHT WIDTH RESOLUTION NUM_FRAMES <<< "$cfg_line"
    log "--- Config: ${cfg_tag} ---" | tee -a "$MEM_LOG"

    # ---- 1. No offloading, dense (expected: OOM on 24GB) ----
    tag="no_offload_dense"
    OUTPUT_FILE="${MEM_DIR}/${tag}_${cfg_tag}.mp4"
    log "[${tag}/${cfg_tag}] Attempting without offloading (expect OOM)..." | tee -a "$MEM_LOG"
    if run_inference_no_offload --seed "$MEM_SEED" --pattern dense 2>&1 | tee -a "$MEM_LOG"; then
        log "[${tag}/${cfg_tag}] SUCCESS (unexpected on 24GB)" | tee -a "$MEM_LOG"
    else
        log "[${tag}/${cfg_tag}] FAILED (OOM expected on 24GB)" | tee -a "$MEM_LOG"
    fi

    # ---- 2. Offloading + dense ----
    tag="offload_dense"
    OUTPUT_FILE="${MEM_DIR}/${tag}_${cfg_tag}.mp4"
    log "[${tag}/${cfg_tag}] With offloading, dense attention..." | tee -a "$MEM_LOG"
    if run_inference_offload --seed "$MEM_SEED" --pattern dense 2>&1 | tee -a "$MEM_LOG"; then
        log "[${tag}/${cfg_tag}] SUCCESS" | tee -a "$MEM_LOG"
    else
        log "[${tag}/${cfg_tag}] FAILED" | tee -a "$MEM_LOG"
    fi

    # ---- 3. Offloading + SAP ----
    tag="offload_sap"
    OUTPUT_FILE="${MEM_DIR}/${tag}_${cfg_tag}.mp4"
    log "[${tag}/${cfg_tag}] With offloading + SAP..." | tee -a "$MEM_LOG"
    if run_inference_offload --seed "$MEM_SEED" "${SAP_ARGS[@]}" 2>&1 | tee -a "$MEM_LOG"; then
        log "[${tag}/${cfg_tag}] SUCCESS" | tee -a "$MEM_LOG"
    else
        log "[${tag}/${cfg_tag}] FAILED" | tee -a "$MEM_LOG"
    fi

    # ---- 4. Offloading + SAP + KV reuse (full system) ----
    tag="offload_sap_kv"
    OUTPUT_FILE="${MEM_DIR}/${tag}_${cfg_tag}.mp4"
    log "[${tag}/${cfg_tag}] With offloading + SAP + KV reuse..." | tee -a "$MEM_LOG"
    if run_inference_offload --seed "$MEM_SEED" \
        "${SAP_ARGS[@]}" "${KV_REUSE_ARGS[@]}" 2>&1 | tee -a "$MEM_LOG"; then
        log "[${tag}/${cfg_tag}] SUCCESS" | tee -a "$MEM_LOG"
    else
        log "[${tag}/${cfg_tag}] FAILED" | tee -a "$MEM_LOG"
    fi
done

# ---- Summary: collect run.json timing + memory for all successful runs ----
log "=== Summary ===" | tee -a "$MEM_LOG"
log "Run summaries:" | tee -a "$MEM_LOG"
for rj in "${MEM_DIR}"/*.run.json; do
    if [[ -f "$rj" ]]; then
        log "  $(basename "$rj"):" | tee -a "$MEM_LOG"
        cat "$rj" | tee -a "$MEM_LOG"
    fi
done

log "05_memory.sh complete." | tee -a "$MEM_LOG"
