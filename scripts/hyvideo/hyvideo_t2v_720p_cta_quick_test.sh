#!/bin/bash

# =============================================================================
# Quick Test: CTA Framework (fewer steps for fast validation)
# =============================================================================
#
# This is a quick test version with reduced inference steps for faster
# validation of the CTA framework. Use the full script for quality output.
#
# =============================================================================

set -e

# Quick test settings
HEIGHT=720
WIDTH=1280
NUM_FRAMES=129
NUM_INFERENCE_STEPS=20  # Reduced for quick testing (use 50 for quality)

# SAP settings
NUM_Q_CENTROIDS=400
NUM_K_CENTROIDS=1000

# CTCA settings
CTCA_QUALITY_THRESHOLD=0.80

# Offloading (for 24GB GPU)
OFFLOAD_NUM_LAYERS=6

# Output
OUTPUT_DIR="outputs/cta_quick_test"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUTPUT_FILE="${OUTPUT_DIR}/quick_test_${TIMESTAMP}.mp4"
LOG_FILE="${OUTPUT_DIR}/log_${TIMESTAMP}.jsonl"

# Prompt
PROMPT="${1:-A beautiful sunset over the ocean, waves gently rolling, cinematic}"

mkdir -p "${OUTPUT_DIR}"

echo "============================================================"
echo "CTA Quick Test (${NUM_INFERENCE_STEPS} steps)"
echo "============================================================"
echo "Prompt: ${PROMPT}"
echo ""

export TIME_BENCH=1

python hyvideo_t2v_inference.py \
    --model_id "tencent/HunyuanVideo" \
    --height ${HEIGHT} \
    --width ${WIDTH} \
    --num_frames ${NUM_FRAMES} \
    --num_inference_steps ${NUM_INFERENCE_STEPS} \
    --pattern SAP_CTCA \
    --num_q_centroids ${NUM_Q_CENTROIDS} \
    --num_k_centroids ${NUM_K_CENTROIDS} \
    --top_p_kmeans 0.9 \
    --min_kc_ratio 0.10 \
    --kmeans_iter_init 50 \
    --kmeans_iter_step 2 \
    --ctca_quality_threshold ${CTCA_QUALITY_THRESHOLD} \
    --ctca_min_interval 2 \
    --ctca_max_interval 10 \
    --ctca_adaptive \
    --first_times_fp 0.10 \
    --first_layers_fp 0.03 \
    --enable_offload \
    --offload_num_layers ${OFFLOAD_NUM_LAYERS} \
    --offload_prefetch \
    --offload_pinned_memory \
    --prompt "${PROMPT}" \
    --output_file "${OUTPUT_FILE}" \
    --logging_file "${LOG_FILE}" \
    --seed 42

echo ""
echo "============================================================"
echo "Quick test complete!"
echo "Output: ${OUTPUT_FILE}"
echo "============================================================"
