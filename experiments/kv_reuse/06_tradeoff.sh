#!/usr/bin/env bash
# ============================================================================
# Experiment 6: Speed-Quality Trade-off (Combined Parameter Sweep)
#
# Jointly varies SAP aggressiveness and KV reuse aggressiveness to show
# a Pareto frontier of speedup vs video quality.
#
# Parameters varied:
#   SAP:      KC (num_k_centroids), top_p
#   KV reuse: interval, max_layers
#
# Levels from conservative (highest quality) to aggressive (fastest):
#   L0: Dense (no SAP, no KV reuse) — quality oracle
#   L1: SAP only (default KC=1000, top_p=0.9) — baseline
#   L2: SAP + KV conservative (interval=1, max_layers=4)
#   L3: SAP + KV default (interval=2, max_layers=8)
#   L4: SAP + KV aggressive (interval=4, max_layers=16)
#   L5: SAP aggressive (KC=500, top_p=0.8) + KV default
#   L6: SAP aggressive + KV aggressive
#   L7: SAP very aggressive (KC=300, top_p=0.7) + KV aggressive
#   L8: SAP extreme (KC=200, top_p=0.6) + KV extreme (interval=8, max_layers=20)
#
# All runs: 720p/49f, quality compared to Dense oracle.
# ============================================================================
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
source "${SCRIPT_DIR}/config.sh"

HEIGHT=$DEFAULT_HEIGHT; WIDTH=$DEFAULT_WIDTH
NUM_FRAMES=$DEFAULT_NUM_FRAMES; RESOLUTION=$DEFAULT_RESOLUTION
CFG_TAG="${RESOLUTION}_${NUM_FRAMES}f"

TRADEOFF_PROMPTS="${TRADEOFF_PROMPTS:-1 7}"
TRADEOFF_SEEDS="${TRADEOFF_SEEDS:-42}"

DENSE_SEED="42"

# ============================================================================
# Helper: run a trade-off configuration
# METRICS_JSONL is set inside the loop per run, so callers should NOT pass it.
# ============================================================================
run_tradeoff() {
    local level_name="$1"; shift
    for seed in $TRADEOFF_SEEDS; do
    for pid in $TRADEOFF_PROMPTS; do
        PROMPT_TEXT=$(cat "${PROJECT_ROOT}/examples/${pid}/prompt.txt")
        OUTPUT_FILE="${RESULT_ROOT}/tradeoff/${level_name}/${CFG_TAG}/${pid}-${seed}.mp4"
        local metrics_jsonl="${METRICS_ROOT}/tradeoff/${level_name}/${CFG_TAG}/${pid}-${seed}.jsonl"
        mkdir -p "$(dirname "$metrics_jsonl")"
        log "[tradeoff:${level_name}] prompt=$pid seed=$seed"
        run_inference --seed "$seed" "$@" \
            --video_k_reuse_metrics_jsonl "$metrics_jsonl"
    done
    done
}

# Same as run_tradeoff but without metrics_jsonl (for non-KV levels)
run_tradeoff_no_metrics() {
    local level_name="$1"; shift
    for seed in $TRADEOFF_SEEDS; do
    for pid in $TRADEOFF_PROMPTS; do
        PROMPT_TEXT=$(cat "${PROJECT_ROOT}/examples/${pid}/prompt.txt")
        OUTPUT_FILE="${RESULT_ROOT}/tradeoff/${level_name}/${CFG_TAG}/${pid}-${seed}.mp4"
        log "[tradeoff:${level_name}] prompt=$pid seed=$seed"
        run_inference --seed "$seed" "$@"
    done
    done
}

quality_vs_dense() {
    local level_name="$1"
    for pid in $TRADEOFF_PROMPTS; do
        ref="${RESULT_ROOT}/tradeoff/L0_dense/${CFG_TAG}/${pid}-${DENSE_SEED}.mp4"
        for seed in $TRADEOFF_SEEDS; do
            test="${RESULT_ROOT}/tradeoff/${level_name}/${CFG_TAG}/${pid}-${seed}.mp4"
            qout="${QUALITY_ROOT}/tradeoff/${level_name}_vs_dense/${CFG_TAG}.jsonl"
            mkdir -p "$(dirname "$qout")"
            compute_quality "$ref" "$test" "$qout" "$pid" "$seed" || true
        done
    done
}

# Shared KV reuse base args (no interval/max_layers — those vary per level)
KV_BASE=(
    --video_k_reuse
    --video_k_reuse_warmup_steps "$KV_WARMUP"
    --video_k_reuse_start_step "$KV_START_STEP"
    --video_k_reuse_layer_stride "$KV_LAYER_STRIDE"
    --video_k_reuse_metrics
)

# ============================================================================
# L0: Dense (quality oracle, only seed=42)
# ============================================================================
log "=== L0: Dense (quality oracle) ==="
for pid in $TRADEOFF_PROMPTS; do
    PROMPT_TEXT=$(cat "${PROJECT_ROOT}/examples/${pid}/prompt.txt")
    OUTPUT_FILE="${RESULT_ROOT}/tradeoff/L0_dense/${CFG_TAG}/${pid}-${DENSE_SEED}.mp4"
    log "[tradeoff:L0_dense] prompt=$pid seed=$DENSE_SEED"
    run_inference --seed "$DENSE_SEED" --pattern dense
done

# L1: SAP only (no KV reuse)
log "=== L1: SAP only ==="
run_tradeoff_no_metrics "L1_sap" "${SAP_ARGS[@]}"

# L2: SAP + KV conservative
log "=== L2: SAP + KV conservative ==="
run_tradeoff "L2_sap_kv_conservative" \
    "${SAP_ARGS[@]}" "${KV_BASE[@]}" \
    --video_k_reuse_interval 1 --video_k_reuse_max_layers 4

# L3: SAP + KV default
log "=== L3: SAP + KV default ==="
run_tradeoff "L3_sap_kv_default" \
    "${SAP_ARGS[@]}" "${KV_BASE[@]}" \
    --video_k_reuse_interval 2 --video_k_reuse_max_layers 8

# L4: SAP + KV aggressive
log "=== L4: SAP + KV aggressive ==="
run_tradeoff "L4_sap_kv_aggressive" \
    "${SAP_ARGS[@]}" "${KV_BASE[@]}" \
    --video_k_reuse_interval 4 --video_k_reuse_max_layers 16

# Helper for non-default SAP params
sap_custom_args() {
    local kc="$1" tp="$2"
    echo "--pattern SAP --num_q_centroids $SAP_QC --num_k_centroids $kc"
    echo "--top_p_kmeans $tp --min_kc_ratio $SAP_MIN_KC_RATIO"
    echo "--kmeans_iter_init $SAP_KMEANS_INIT --kmeans_iter_step $SAP_KMEANS_STEP"
    echo "--zero_step_kmeans_init --first_times_fp $FIRST_TIMES_FP --first_layers_fp $FIRST_LAYERS_FP"
}

# L5: SAP aggressive (KC=500, top_p=0.8) + KV default
log "=== L5: SAP aggressive + KV default ==="
run_tradeoff "L5_sap_aggr_kv_default" \
    $(sap_custom_args 500 0.8) "${KV_BASE[@]}" \
    --video_k_reuse_interval 2 --video_k_reuse_max_layers 8

# L6: SAP aggressive + KV aggressive
log "=== L6: SAP aggressive + KV aggressive ==="
run_tradeoff "L6_sap_aggr_kv_aggr" \
    $(sap_custom_args 500 0.8) "${KV_BASE[@]}" \
    --video_k_reuse_interval 4 --video_k_reuse_max_layers 16

# L7: SAP very aggressive (KC=300, top_p=0.7) + KV aggressive
log "=== L7: SAP very aggressive + KV aggressive ==="
run_tradeoff "L7_sap_vaggr_kv_aggr" \
    $(sap_custom_args 300 0.7) "${KV_BASE[@]}" \
    --video_k_reuse_interval 4 --video_k_reuse_max_layers 16

# L8: SAP extreme (KC=200, top_p=0.6) + KV extreme
log "=== L8: SAP extreme + KV extreme ==="
run_tradeoff "L8_sap_extreme_kv_extreme" \
    $(sap_custom_args 200 0.6) "${KV_BASE[@]}" \
    --video_k_reuse_interval 8 --video_k_reuse_max_layers 20

# ============================================================================
# Quality: all levels vs Dense oracle
# ============================================================================
log "=== Computing quality metrics (all levels vs Dense) ==="
for level in L1_sap L2_sap_kv_conservative L3_sap_kv_default L4_sap_kv_aggressive \
             L5_sap_aggr_kv_default L6_sap_aggr_kv_aggr L7_sap_vaggr_kv_aggr \
             L8_sap_extreme_kv_extreme; do
    quality_vs_dense "$level"
done

log "06_tradeoff.sh complete."
