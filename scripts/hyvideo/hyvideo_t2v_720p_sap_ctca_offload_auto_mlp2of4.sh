#!/bin/bash

# SAP + CTCA + CTAA + Dynamic Offload (auto) + Training-free MLP 2:4 Sparsity (outlier split)
#
# This script runs HunyuanVideo with:
# 1) SAP_CTCA attention (CTCA reduces K-means overhead; CTAA enabled by default in the CTCA processor)
# 2) Dynamic transformer layer offloading (auto-tuned to your GPU's available VRAM)
# 3) Training-free MLP 2:4 structured sparsity with an outlier-dense split
#
# Notes:
# - Semi-structured sparse GEMM acceleration depends on your PyTorch build.
#   If unsupported, the code falls back to using dense pruned weights (still runs).
# - Auto offload uses current free VRAM (torch.cuda.mem_get_info) and reserves
#   headroom for activations/workspaces; tune offload_* vars if you see OOM.

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

# SAP configuration (quality-oriented defaults)
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

# Warmup configuration (keep early steps/layers dense)
first_times_fp=0.1
first_layers_fp=0.03

# Offload configuration
# Fallback window (also acts as a cap unless --offload_auto_allow_increase is set)
offload_num_layers=8
offload_max_fraction=0.90          # Target fraction of total VRAM
offload_activation_reserve_gb=4.0  # Reserve for activations/caches
offload_cuda_overhead_gb=0.5       # Reserve for CUDA workspaces

# MLP 2:4 sparsity configuration (training-free)
mlp_outlier_ratio=0.02             # Keep top 2% output rows dense per Linear
mlp_outlier_metric="l2"            # "l2" or "maxabs"

# Output configuration
output_dir="outputs/hyvideo_sap_ctca_offload_auto_mlp2of4"
logging_dir="logs/hyvideo_sap_ctca_offload_auto_mlp2of4"

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
  --enable_mlp_2of4 \
  --mlp_outlier_ratio "${mlp_outlier_ratio}" \
  --mlp_outlier_metric "${mlp_outlier_metric}" \
  --seed 42

echo ""
echo "============================================"
echo "Done! Video saved to ${output_dir}/output.mp4"
echo "Logs saved to ${logging_dir}/"
echo "============================================"
