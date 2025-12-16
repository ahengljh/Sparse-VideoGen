#!/bin/bash

# SAP + CTCA + CTAA + Dynamic Offload (auto)
#
# This is the same as hyvideo_t2v_720p_sap_ctca_offload.sh, but with an
# auto-tuned offload window based on *current* available VRAM.
#
# Use this as a baseline script to compare against:
# - hyvideo_t2v_720p_sap_ctca_offload.sh (fixed window)
# - hyvideo_t2v_720p_sap_ctca_offload_auto_mlp2of4.sh (auto + MLP 2:4)

set -e

# Memory-safety knobs (override externally if desired)
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export SVG_ROPE_CHUNK_SIZE="${SVG_ROPE_CHUNK_SIZE:-4096}"
export SVG_ROPE_FORCE_FP32="${SVG_ROPE_FORCE_FP32:-1}"

# Model configuration
model_id="tencent/HunyuanVideo"
height=720
width=1280
num_frames=129
num_inference_steps=50

# SAP configuration
qc_kmeans=400
kc_kmeans=1000
top_p_k=0.9
min_kc_ratio=0.10
kmeans_iter_init=50
kmeans_iter_step=2

# CTCA configuration
ctca_quality_threshold=0.80
ctca_min_interval=2
ctca_max_interval=10

# Warmup configuration
first_times_fp=0.1
first_layers_fp=0.03

# Offload configuration
offload_num_layers=8              # fallback window (also acts as a cap unless --offload_auto_allow_increase is set)
offload_max_fraction=0.90         # target fraction of total VRAM
offload_activation_reserve_gb=4.0 # reserve for activations/caches
offload_cuda_overhead_gb=0.5      # reserve for CUDA workspaces

# Output configuration
output_dir="outputs/hyvideo_sap_ctca_offload_auto"
logging_dir="logs/hyvideo_sap_ctca_offload_auto"

# Prompt
prompt="A cat walks on the grass, realistic style, high quality"

mkdir -p "${output_dir}" "${logging_dir}"

python hyvideo_t2v_inference.py \
  --model_id "${model_id}" \
  --pattern SAP_CTCA \
  --height "${height}" \
  --width "${width}" \
  --num_frames "${num_frames}" \
  --num_inference_steps "${num_inference_steps}" \
  --prompt "${prompt}" \
  --output_file "${output_dir}/output.mp4" \
  --logging_file "${logging_dir}/density_log.jsonl" \
  --first_times_fp "${first_times_fp}" \
  --first_layers_fp "${first_layers_fp}" \
  --num_q_centroids "${qc_kmeans}" \
  --num_k_centroids "${kc_kmeans}" \
  --top_p_kmeans "${top_p_k}" \
  --min_kc_ratio "${min_kc_ratio}" \
  --kmeans_iter_init "${kmeans_iter_init}" \
  --kmeans_iter_step "${kmeans_iter_step}" \
  --ctca_quality_threshold "${ctca_quality_threshold}" \
  --ctca_min_interval "${ctca_min_interval}" \
  --ctca_max_interval "${ctca_max_interval}" \
  --ctca_adaptive \
  --enable_offload \
  --offload_auto \
  --offload_num_layers "${offload_num_layers}" \
  --offload_max_fraction "${offload_max_fraction}" \
  --offload_activation_reserve_gb "${offload_activation_reserve_gb}" \
  --offload_cuda_overhead_gb "${offload_cuda_overhead_gb}" \
  --offload_pinned_memory \
  --offload_prefetch \
  --seed 42

echo ""
echo "============================================"
echo "Done! Video saved to ${output_dir}/output.mp4"
echo "Logs saved to ${logging_dir}/"
echo "============================================"
