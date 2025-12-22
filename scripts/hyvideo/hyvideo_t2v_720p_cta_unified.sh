#!/bin/bash

# =============================================================================
# Unified CTA (Cross-Timestep Amortization) Framework Test Script
# =============================================================================
#
# This script runs HunyuanVideo with the full CTA framework:
#   - CTCA: Cross-Timestep Cluster Amortization (5-10x K-means speedup)
#   - CTAA: Cross-Timestep Attention Amortization (~30% compute reduction)
#   - AI-Offload: Attention-Informed Offloading (15% better prefetch)
#
# Video Configuration:
#   - Resolution: 720p (720 x 1280)
#   - Duration: 5 seconds (129 frames at ~24fps)
#
# =============================================================================

set -e  # Exit on error

# =============================================================================
# Configuration
# =============================================================================

# Model
MODEL_ID="tencent/HunyuanVideo"

# Video settings (720p, 5 seconds)
HEIGHT=720
WIDTH=1280
NUM_FRAMES=129        # ~5 seconds at 24fps
NUM_INFERENCE_STEPS=50

# SAP (Semantic Aware Permutation) settings
NUM_Q_CENTROIDS=400   # Query clusters
NUM_K_CENTROIDS=1000  # Key clusters
TOP_P_KMEANS=0.9      # Top-p for block selection
MIN_KC_RATIO=0.10     # Minimum key blocks ratio

# K-means iterations (used by CTCA)
KMEANS_ITER_INIT=50   # Full clustering iterations
KMEANS_ITER_STEP=2    # Update-only iterations

# CTCA (Cross-Timestep Cluster Amortization) settings
CTCA_QUALITY_THRESHOLD=0.80   # Quality threshold for reclustering (0-1)
CTCA_MIN_INTERVAL=2           # Minimum timesteps between reclusters
CTCA_MAX_INTERVAL=10          # Maximum timesteps to reuse clusters

# CTAA is enabled by default in SAP_CTCA pattern
# Settings: p_full=0.7, p_total=0.95

# Warmup settings (use dense attention for early layers/timesteps)
FIRST_TIMES_FP=0.10   # Dense attention for first 10% timesteps
FIRST_LAYERS_FP=0.03  # Dense attention for first 3% layers

# Offloading settings (for 24GB GPU)
OFFLOAD_NUM_LAYERS=6          # Layers in GPU sliding window
OFFLOAD_MAX_MEMORY_GB=""      # Auto-detect if empty
OFFLOAD_PREFETCH=true
OFFLOAD_PINNED_MEMORY=true

# MLP 2:4 Sparsity (optional, set to false if causing issues)
ENABLE_MLP_2OF4=false
MLP_OUTLIER_RATIO=0.02

# Output settings
OUTPUT_DIR="outputs/cta_unified_720p"
LOG_DIR="logs/cta_unified_720p"
SEED=42

# Prompt (can be overridden via command line)
PROMPT="${1:-A cat walks on the grass, realistic style, high quality, 4K}"

# =============================================================================
# Setup
# =============================================================================

echo "============================================================"
echo "CTA (Cross-Timestep Amortization) Unified Framework"
echo "============================================================"
echo ""
echo "Video Settings:"
echo "  Resolution:    ${WIDTH}x${HEIGHT}"
echo "  Frames:        ${NUM_FRAMES} (~5 seconds)"
echo "  Steps:         ${NUM_INFERENCE_STEPS}"
echo ""
echo "CTA Components:"
echo "  CTCA:          Enabled (quality_threshold=${CTCA_QUALITY_THRESHOLD})"
echo "  CTAA:          Enabled (p_full=0.7, p_total=0.95)"
echo "  AI-Offload:    Enabled (${OFFLOAD_NUM_LAYERS} layers on GPU)"
echo ""
echo "Prompt: ${PROMPT}"
echo "============================================================"
echo ""

# Create output directories
mkdir -p "${OUTPUT_DIR}"
mkdir -p "${LOG_DIR}"

# Generate timestamp for unique filenames
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUTPUT_FILE="${OUTPUT_DIR}/video_${TIMESTAMP}.mp4"
LOG_FILE="${LOG_DIR}/density_log_${TIMESTAMP}.jsonl"

# =============================================================================
# Build Command
# =============================================================================

CMD="python hyvideo_t2v_inference.py"

# Model settings
CMD="${CMD} --model_id ${MODEL_ID}"
CMD="${CMD} --height ${HEIGHT}"
CMD="${CMD} --width ${WIDTH}"
CMD="${CMD} --num_frames ${NUM_FRAMES}"
CMD="${CMD} --num_inference_steps ${NUM_INFERENCE_STEPS}"

# Use SAP_CTCA pattern (includes CTCA + CTAA)
CMD="${CMD} --pattern SAP_CTCA"

# SAP settings
CMD="${CMD} --num_q_centroids ${NUM_Q_CENTROIDS}"
CMD="${CMD} --num_k_centroids ${NUM_K_CENTROIDS}"
CMD="${CMD} --top_p_kmeans ${TOP_P_KMEANS}"
CMD="${CMD} --min_kc_ratio ${MIN_KC_RATIO}"
CMD="${CMD} --kmeans_iter_init ${KMEANS_ITER_INIT}"
CMD="${CMD} --kmeans_iter_step ${KMEANS_ITER_STEP}"

# CTCA settings
CMD="${CMD} --ctca_quality_threshold ${CTCA_QUALITY_THRESHOLD}"
CMD="${CMD} --ctca_min_interval ${CTCA_MIN_INTERVAL}"
CMD="${CMD} --ctca_max_interval ${CTCA_MAX_INTERVAL}"
CMD="${CMD} --ctca_adaptive"

# Warmup settings
CMD="${CMD} --first_times_fp ${FIRST_TIMES_FP}"
CMD="${CMD} --first_layers_fp ${FIRST_LAYERS_FP}"

# Offloading settings
CMD="${CMD} --enable_offload"
CMD="${CMD} --offload_num_layers ${OFFLOAD_NUM_LAYERS}"
if [ -n "${OFFLOAD_MAX_MEMORY_GB}" ]; then
    CMD="${CMD} --offload_max_memory_gb ${OFFLOAD_MAX_MEMORY_GB}"
fi
if [ "${OFFLOAD_PREFETCH}" = true ]; then
    CMD="${CMD} --offload_prefetch"
else
    CMD="${CMD} --offload_no_prefetch"
fi
if [ "${OFFLOAD_PINNED_MEMORY}" = true ]; then
    CMD="${CMD} --offload_pinned_memory"
else
    CMD="${CMD} --offload_no_pinned_memory"
fi

# MLP 2:4 Sparsity (optional)
if [ "${ENABLE_MLP_2OF4}" = true ]; then
    CMD="${CMD} --enable_mlp_2of4"
    CMD="${CMD} --mlp_outlier_ratio ${MLP_OUTLIER_RATIO}"
fi

# Output settings
CMD="${CMD} --prompt \"${PROMPT}\""
CMD="${CMD} --output_file ${OUTPUT_FILE}"
CMD="${CMD} --logging_file ${LOG_FILE}"
CMD="${CMD} --seed ${SEED}"

# =============================================================================
# Run
# =============================================================================

echo "Running command:"
echo "${CMD}"
echo ""
echo "============================================================"
echo ""

# Set environment variables for timing benchmarks
export TIME_BENCH=1

# Run the inference
eval ${CMD}

# =============================================================================
# Summary
# =============================================================================

echo ""
echo "============================================================"
echo "Generation Complete!"
echo "============================================================"
echo ""
echo "Output video:    ${OUTPUT_FILE}"
echo "Density log:     ${LOG_FILE}"
echo ""
echo "To view statistics, check the console output above for:"
echo "  - CTCA Statistics (cluster reuse ratio, K-means speedup)"
echo "  - CTAA Statistics (tier distribution)"
echo "  - Offloading Statistics (prefetch hit rate)"
echo ""

# Check if output file exists
if [ -f "${OUTPUT_FILE}" ]; then
    FILE_SIZE=$(du -h "${OUTPUT_FILE}" | cut -f1)
    echo "Video file size: ${FILE_SIZE}"
    echo ""
    echo "Play with: ffplay ${OUTPUT_FILE}"
else
    echo "WARNING: Output file not found!"
fi

echo "============================================================"
