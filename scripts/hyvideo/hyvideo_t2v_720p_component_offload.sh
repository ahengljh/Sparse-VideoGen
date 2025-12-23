#!/bin/bash
#
# Example script demonstrating fine-grained component-level offloading
#
# Strategy: Pin Attention, Offload FFN
# - Keeps attention weights (~30% of model) permanently on GPU
# - Dynamically loads FFN weights (~70% of model)
# - Achieves better memory efficiency than full-layer offloading
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
# FFN_LAYERS_ON_GPU: Number of layers to keep FFN on GPU (sliding window)
# Higher = faster but more VRAM, Lower = slower but less VRAM
# Recommended: 4-8 for 24GB GPU, 10-15 for 40GB+ GPU
FFN_LAYERS_ON_GPU="${FFN_LAYERS_ON_GPU:-6}"

echo "=============================================="
echo "Fine-Grained Component Offloading Demo"
echo "=============================================="
echo "Strategy: Pin Attention on GPU, Offload FFN"
echo ""
echo "Model: $MODEL_ID"
echo "Resolution: $RESOLUTION"
echo "Frames: $NUM_FRAMES"
echo "FFN layers on GPU: $FFN_LAYERS_ON_GPU"
echo "Output: $OUTPUT_FILE"
echo "=============================================="

python hyvideo_t2v_inference.py \
    --model_id "$MODEL_ID" \
    --prompt "$PROMPT" \
    --resolution "$RESOLUTION" \
    --num_frames "$NUM_FRAMES" \
    --output_file "$OUTPUT_FILE" \
    --seed "$SEED" \
    --pattern SAP \
    --num_q_centroids 50 \
    --num_k_centroids 200 \
    --top_p_kmeans 0.9 \
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
