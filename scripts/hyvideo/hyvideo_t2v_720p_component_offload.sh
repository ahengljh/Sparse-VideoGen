#!/bin/bash
#
# Example script demonstrating hybrid component-level offloading with SAP_CTCA
#
# Strategy: Sliding Window with Cross-Timestep Cluster Amortization (CTCA)
# - Uses sliding window of full layers (like standard AIO)
# - SAP_CTCA reuses cluster assignments across timesteps for efficiency
# - Tuned parameters for good quality-speed balance
#
# This is especially useful for GPUs with limited VRAM (e.g., RTX 4090 24GB)
# where we want to maximize GPU utilization while staying within memory limits.
#

set -e

# Default parameters
MODEL_ID="${MODEL_ID:-tencent/HunyuanVideo}"
PROMPT="${PROMPT:-A cat walks on the grass, realistic style.}"
OUTPUT_FILE="${OUTPUT_FILE:-output_component_offload.mp4}"
RESOLUTION="${RESOLUTION:-720p}"
NUM_FRAMES="${NUM_FRAMES:-129}"
SEED="${SEED:-42}"

# Offloading parameters
# FFN_LAYERS_ON_GPU: Number of layers to keep on GPU (sliding window)
# Higher = faster but more VRAM, Lower = slower but less VRAM
# Recommended: 4-8 for 24GB GPU, 10-15 for 40GB+ GPU
FFN_LAYERS_ON_GPU="${FFN_LAYERS_ON_GPU:-6}"

echo "=============================================="
echo "SAP_CTCA with Component Offloading"
echo "=============================================="
echo "Strategy: Cross-Timestep Cluster Amortization"
echo ""
echo "Model: $MODEL_ID"
echo "Resolution: $RESOLUTION"
echo "Frames: $NUM_FRAMES"
echo "Layers on GPU: $FFN_LAYERS_ON_GPU"
echo "Output: $OUTPUT_FILE"
echo "=============================================="

python hyvideo_t2v_inference.py \
    --model_id "$MODEL_ID" \
    --prompt "$PROMPT" \
    --resolution "$RESOLUTION" \
    --num_frames "$NUM_FRAMES" \
    --output_file "$OUTPUT_FILE" \
    --seed "$SEED" \
    --pattern SAP_CTCA \
    --num_q_centroids 100 \
    --num_k_centroids 400 \
    --top_p_kmeans 0.95 \
    --min_kc_ratio 0.05 \
    --first_layers_fp 0.1 \
    --first_times_fp 0.15 \
    --kmeans_iter_init 10 \
    --kmeans_iter_step 3 \
    --ctca_quality_threshold 0.88 \
    --ctca_min_interval 1 \
    --ctca_max_interval 5 \
    --enable_offload \
    --offload_strategy component \
    --offload_num_layers "$FFN_LAYERS_ON_GPU" \
    --offload_pinned_memory \
    --offload_prefetch

echo ""
echo "=============================================="
echo "Generation complete!"
echo "Output saved to: $OUTPUT_FILE"
echo "=============================================="
