#!/bin/bash
#
# Example script demonstrating hybrid component-level offloading with SAP_CTCA
#
# Supports two offloading strategies:
#
# 1. "layer" strategy (default):
#    - Slide entire layers through a window
#    - Standard AIO-style offloading
#
# 2. "component" strategy (fine-grained):
#    - Pin ALL attention+norm modules on GPU permanently (~35% of each layer)
#    - Only slide FFN modules through a window (~65% of each layer)
#    - Reduces transfer overhead since attention is smaller
#    - Better for scenarios with frequent attention reuse
#
# Combined with SAP_CTCA for sparse attention.
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

# Offloading strategy:
# - "layer": Slide entire layers (standard AIO-style)
# - "component": Pin attention on GPU, slide only FFN (fine-grained)
OFFLOAD_STRATEGY="${OFFLOAD_STRATEGY:-component}"

# Offloading mode:
# - Set OFFLOAD_AUTO=1 for adaptive mode (auto-detect based on GPU memory)
# - Set FFN_LAYERS_ON_GPU to a number for fixed mode
OFFLOAD_AUTO="${OFFLOAD_AUTO:-1}"
FFN_LAYERS_ON_GPU="${FFN_LAYERS_ON_GPU:-}"

echo "=============================================="
echo "SAP_CTCA with Component Offloading"
echo "=============================================="
echo "Strategy: Cross-Timestep Cluster Amortization"
echo ""
echo "Model: $MODEL_ID"
echo "Resolution: $RESOLUTION"
echo "Frames: $NUM_FRAMES"
echo "Offload Strategy: $OFFLOAD_STRATEGY"
if [ "$OFFLOAD_STRATEGY" = "component" ]; then
    echo "  → Pin ALL attention on GPU, slide FFN only"
else
    echo "  → Slide entire layers"
fi
if [ "$OFFLOAD_AUTO" = "1" ]; then
    echo "Offload Mode: ADAPTIVE (auto-detect layers)"
else
    echo "Layers on GPU: $FFN_LAYERS_ON_GPU"
fi
echo "Output: $OUTPUT_FILE"
echo "=============================================="

# Build offload arguments
OFFLOAD_ARGS="--enable_offload --offload_strategy $OFFLOAD_STRATEGY --offload_pinned_memory --offload_prefetch"
if [ "$OFFLOAD_AUTO" = "1" ]; then
    OFFLOAD_ARGS="$OFFLOAD_ARGS --offload_auto"
elif [ -n "$FFN_LAYERS_ON_GPU" ]; then
    OFFLOAD_ARGS="$OFFLOAD_ARGS --offload_num_layers $FFN_LAYERS_ON_GPU"
fi

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
    $OFFLOAD_ARGS

echo ""
echo "=============================================="
echo "Generation complete!"
echo "Output saved to: $OUTPUT_FILE"
echo "=============================================="
