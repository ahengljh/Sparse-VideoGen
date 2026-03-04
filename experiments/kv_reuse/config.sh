#!/usr/bin/env bash
# ============================================================================
# Shared configuration for all KV reuse experiments
# Source this file from other scripts: source experiments/kv_reuse/config.sh
#
# Key design decisions:
#   - Baselines: Dense (quality oracle), SAP, SVG (existing methods)
#   - Our contributions: KV reuse + dynamic layer offloading
#   - Offloading only for Exp 5 (Memory Feasibility); other exps use full GPU
#   - Default: 720p, 49 frames (~2s video)
#   - Each run automatically produces <video>.run.json with timing + memory
# ============================================================================

# --- Paths ---
export PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
export RESULT_ROOT="${PROJECT_ROOT}/result/kv_reuse_paper"
export METRICS_ROOT="${RESULT_ROOT}/metrics"
export QUALITY_ROOT="${RESULT_ROOT}/quality"

# --- Model ---
export MODEL_ID="tencent/HunyuanVideo"

# --- Quick mode: minimal runs for early-stage validation ---
# Set QUICK_MODE=1 to use 1 prompt, 1 seed, 25 frames (~1s at 24fps)
export QUICK_MODE="${QUICK_MODE:-0}"

if [[ "$QUICK_MODE" == "1" ]]; then
    export PROMPT_IDS="${PROMPT_IDS:-7}"
    export SEEDS="${SEEDS:-42}"
    export DEFAULT_NUM_FRAMES="${DEFAULT_NUM_FRAMES:-25}"
    log_quick_note="[QUICK MODE] 1 prompt, 1 seed, ${DEFAULT_NUM_FRAMES} frames"
else
    export PROMPT_IDS="${PROMPT_IDS:-1 7}"
    export SEEDS="${SEEDS:-42}"
    export DEFAULT_NUM_FRAMES="${DEFAULT_NUM_FRAMES:-49}"
    log_quick_note=""
fi

# --- Inference ---
export INFER_STEPS="${INFER_STEPS:-50}"

# --- Default resolution: 720p ---
export DEFAULT_HEIGHT=720
export DEFAULT_WIDTH=1280
export DEFAULT_RESOLUTION="720p"

# --- Offloading (only for Exp 5: Memory Feasibility) ---
# On H200 (141GB), offloading is NOT needed for normal runs.
# Only enable for memory feasibility experiments that simulate 24GB consumer GPUs.
export OFFLOAD_NUM_LAYERS="${OFFLOAD_NUM_LAYERS:-6}"
export GPU_MEMORY_LIMIT_GB="${GPU_MEMORY_LIMIT_GB:-24}"

# --- Sparse attention defaults (SVG / SAP) ---
export FIRST_TIMES_FP=0.1
export FIRST_LAYERS_FP=0.03
export SVG_SPARSITY=0.25
export SVG_SAMPLED_ROWS=64
export SAP_QC=400
export SAP_KC=1000
export SAP_TOP_P=0.9
export SAP_MIN_KC_RATIO=0.10
export SAP_KMEANS_INIT=50
export SAP_KMEANS_STEP=2

# --- KV reuse defaults ---
export KV_BLOCK_SIZE=64
export KV_MAX_BLOCKS=64
export KV_WARMUP=4
export KV_START_STEP=6
export KV_INTERVAL=2
export KV_LAYER_STRIDE=8
export KV_MAX_LAYERS=8

# --- GPU ---
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

# --- Common argument arrays ---
OFFLOAD_ARGS=(
    --enable_offload
    --offload_num_layers "$OFFLOAD_NUM_LAYERS"
    --offload_pinned_memory
    --offload_prefetch
    --max_gpu_memory_gb "$GPU_MEMORY_LIMIT_GB"
)

KV_REUSE_ARGS=(
    --video_k_reuse
    --video_k_reuse_block_size "$KV_BLOCK_SIZE"
    --video_k_reuse_max_blocks "$KV_MAX_BLOCKS"
    --video_k_reuse_warmup_steps "$KV_WARMUP"
    --video_k_reuse_start_step "$KV_START_STEP"
    --video_k_reuse_interval "$KV_INTERVAL"
    --video_k_reuse_layer_stride "$KV_LAYER_STRIDE"
    --video_k_reuse_max_layers "$KV_MAX_LAYERS"
)

KV_METRICS_ARGS=(
    "${KV_REUSE_ARGS[@]}"
    --video_k_reuse_metrics
)

SVG_ARGS=(
    --pattern SVG
    --num_sampled_rows "$SVG_SAMPLED_ROWS"
    --sparsity "$SVG_SPARSITY"
    --first_times_fp "$FIRST_TIMES_FP"
    --first_layers_fp "$FIRST_LAYERS_FP"
)

SAP_ARGS=(
    --pattern SAP
    --num_q_centroids "$SAP_QC"
    --num_k_centroids "$SAP_KC"
    --top_p_kmeans "$SAP_TOP_P"
    --min_kc_ratio "$SAP_MIN_KC_RATIO"
    --kmeans_iter_init "$SAP_KMEANS_INIT"
    --kmeans_iter_step "$SAP_KMEANS_STEP"
    --zero_step_kmeans_init
    --first_times_fp "$FIRST_TIMES_FP"
    --first_layers_fp "$FIRST_LAYERS_FP"
)

# --- Helpers ---
run_inference() {
    # Usage: run_inference <extra_args...>
    # Expects: $OUTPUT_FILE, $PROMPT_TEXT, $HEIGHT, $WIDTH, $NUM_FRAMES, $RESOLUTION
    # No offloading — uses full GPU memory (H200 141GB). Much faster.
    python "${PROJECT_ROOT}/hyvideo_t2v_inference.py" \
        --model_id "${MODEL_ID}" \
        --prompt "${PROMPT_TEXT}" \
        --height "${HEIGHT}" \
        --width "${WIDTH}" \
        --num_frames "${NUM_FRAMES}" \
        --num_inference_steps "${INFER_STEPS}" \
        --resolution "${RESOLUTION}" \
        --output_file "${OUTPUT_FILE}" \
        --skip_existing \
        "$@"
}

# run_inference_offload: WITH offloading + 24GB limit.
# Used only for Exp 5 (Memory Feasibility) to simulate consumer GPU.
run_inference_offload() {
    python "${PROJECT_ROOT}/hyvideo_t2v_inference.py" \
        --model_id "${MODEL_ID}" \
        --prompt "${PROMPT_TEXT}" \
        --height "${HEIGHT}" \
        --width "${WIDTH}" \
        --num_frames "${NUM_FRAMES}" \
        --num_inference_steps "${INFER_STEPS}" \
        --resolution "${RESOLUTION}" \
        --output_file "${OUTPUT_FILE}" \
        --skip_existing \
        "${OFFLOAD_ARGS[@]}" \
        "$@"
}

# run_inference_no_offload: NO offloading, NO memory limit.
# Used for memory feasibility tests (expected to OOM on 24GB GPUs).
run_inference_no_offload() {
    python "${PROJECT_ROOT}/hyvideo_t2v_inference.py" \
        --model_id "${MODEL_ID}" \
        --prompt "${PROMPT_TEXT}" \
        --height "${HEIGHT}" \
        --width "${WIDTH}" \
        --num_frames "${NUM_FRAMES}" \
        --num_inference_steps "${INFER_STEPS}" \
        --resolution "${RESOLUTION}" \
        --output_file "${OUTPUT_FILE}" \
        "$@"
}

compute_quality() {
    # Usage: compute_quality <ref_video> <test_video> <output_jsonl> <prompt_idx> <seed>
    local ref="$1" test="$2" out="$3" pidx="$4" seed="$5"
    if [[ ! -f "$ref" ]] || [[ ! -f "$test" ]]; then
        echo "SKIP quality (missing video): ref=$ref test=$test"
        return 1
    fi
    python "${PROJECT_ROOT}/svg/utils/metric.py" \
        --video1_path "$ref" \
        --video2_path "$test" \
        --output_path "$out" \
        --prompt_idx "$pidx" \
        --seed "$seed"
}

timestamp() {
    date "+%Y-%m-%d %H:%M:%S"
}

log() {
    echo "[$(timestamp)] $*"
}
