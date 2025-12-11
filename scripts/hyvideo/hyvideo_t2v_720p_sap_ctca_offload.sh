#!/bin/bash

# SAP with CTCA + Dynamic Offloading for Small GPUs (e.g., RTX 4090 24GB)
#
# This script runs HunyuanVideo with:
# 1. SAP (Semantic-Aware Permutation) for sparse attention
# 2. CTCA (Cross-Timestep Cluster Amortization) to reduce K-means overhead
# 3. Dynamic Layer Offloading to fit on 24GB GPUs
#
# Without offloading, HunyuanVideo needs ~25-30GB VRAM
# With offloading, it can run on 24GB GPUs (4090, 3090)

# Model configuration
model_id="tencent/HunyuanVideo"
resolution="720p"
height=720
width=1280
num_frames=129
num_inference_steps=50

# SAP configuration
qc_kmeans=400                    # Number of query centroids
kc_kmeans=1000                   # Number of key centroids
top_p_k=0.9                      # Top-p threshold for block selection
min_kc_ratio=0.10                # Minimum key blocks ratio
kmeans_iter_init=50              # K-means iterations for initialization
kmeans_iter_step=2               # K-means iterations per step

# CTCA configuration
ctca_quality_threshold=0.80      # Quality threshold for re-clustering
ctca_min_interval=2              # Minimum timesteps between re-clustering
ctca_max_interval=10             # Maximum timesteps to reuse clusters

# Warmup configuration
first_times_fp=0.1               # Dense attention for first 10% timesteps
first_layers_fp=0.03             # Dense attention for first 3% layers

# Offloading configuration (sliding window approach)
# Memory breakdown for HunyuanVideo:
#   - Total transformer: ~13GB (60 layers @ ~217MB each)
#   - VAE: ~300MB (kept on GPU for decoding)
#   - Activations: ~2-4GB (depends on resolution/frames)
#   - CUDA overhead: ~500MB
#
# For 24GB GPU (4090/3090), we have ~20GB usable with headroom:
#   20GB - 4GB (activations) - 0.3GB (VAE) - 0.5GB (overhead) = ~15GB for layers
#   15GB / 0.217GB per layer = ~69 layers (but we only have 60)
#   So with this setup, we could keep all 60 layers, but we use sliding window
#   for safety margin and to leave room for attention caching.
#
# Recommended settings:
#   - 24GB GPU (4090/3090): 6-10 layers (safe), up to 15 (aggressive)
#   - 40GB+ GPU (A100/A6000): 15-30 layers
#   - Use max_memory_gb for auto-tuning
offload_num_layers=8             # Keep 8 layers on GPU (~1.7GB)
# offload_max_memory_gb=20       # Alternative: auto-tune based on memory budget

# Text encoder offload mode:
#   - "sequential": Move to GPU for encoding, then release (saves ~3-5GB) [RECOMMENDED]
#   - "keep": Keep on GPU always (uses more memory)
#   - "cpu": Keep on CPU (may cause issues)
text_encoder_mode="sequential"

# Output configuration
output_dir="outputs/hyvideo_sap_ctca_offload"
logging_dir="logs/hyvideo_sap_ctca_offload"

# Prompt
prompt="A cat walks on the grass, realistic style, high quality"

# Create output directories
mkdir -p $output_dir
mkdir -p $logging_dir

# Run inference with offloading enabled
python hyvideo_t2v_inference.py \
    --model_id $model_id \
    --pattern SAP_CTCA \
    --height $height \
    --width $width \
    --num_frames $num_frames \
    --num_inference_steps $num_inference_steps \
    --prompt "$prompt" \
    --output_file "${output_dir}/output.mp4" \
    --logging_file "${logging_dir}/density_log.jsonl" \
    --first_times_fp $first_times_fp \
    --first_layers_fp $first_layers_fp \
    --num_q_centroids $qc_kmeans \
    --num_k_centroids $kc_kmeans \
    --top_p_kmeans $top_p_k \
    --min_kc_ratio $min_kc_ratio \
    --kmeans_iter_init $kmeans_iter_init \
    --kmeans_iter_step $kmeans_iter_step \
    --ctca_quality_threshold $ctca_quality_threshold \
    --ctca_min_interval $ctca_min_interval \
    --ctca_max_interval $ctca_max_interval \
    --ctca_adaptive \
    --enable_offload \
    --offload_num_layers $offload_num_layers \
    --offload_text_encoder_mode $text_encoder_mode \
    --offload_pinned_memory \
    --offload_prefetch \
    --seed 42

echo ""
echo "============================================"
echo "Done! Video saved to ${output_dir}/output.mp4"
echo "Logs saved to ${logging_dir}/"
echo "============================================"
