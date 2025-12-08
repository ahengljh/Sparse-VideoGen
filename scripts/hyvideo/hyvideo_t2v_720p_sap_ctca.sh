#!/bin/bash

# SAP with Cross-Timestep Cluster Amortization (CTCA)
# This script runs HunyuanVideo with SAP+CTCA for accelerated video generation
# CTCA reduces K-means clustering overhead by 5-10x by reusing cluster assignments

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
kmeans_iter_step=2               # K-means iterations per step (when not using CTCA cache)

# CTCA configuration
ctca_quality_threshold=0.80      # Quality threshold (0-1) for triggering re-clustering
ctca_min_interval=2              # Minimum timesteps between re-clustering
ctca_max_interval=10             # Maximum timesteps to reuse clusters
# ctca_adaptive is enabled by default
# ctca_verbose can be added for debugging

# Warmup configuration
first_times_fp=0.1               # Dense attention for first 10% timesteps
first_layers_fp=0.03             # Dense attention for first 3% layers

# Output configuration
output_dir="outputs/hyvideo_sap_ctca"
logging_dir="logs/hyvideo_sap_ctca"

# Prompt
prompt="A cat walks on the grass, realistic style, high quality"

# Create output directories
mkdir -p $output_dir
mkdir -p $logging_dir

# Run inference
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
    --seed 42

echo "Done! Video saved to ${output_dir}/output.mp4"
echo "CTCA statistics and density logs saved to ${logging_dir}/"
