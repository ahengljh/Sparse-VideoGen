#!/bin/bash

# =============================================================================
# SAP + CTCA + Dynamic Offloading for Small GPUs (e.g., RTX 4090 24GB)
# =============================================================================
# This script runs HunyuanVideo with:
# 1. SAP (Semantic-Aware Permutation) for sparse attention
# 2. CTCA (Cross-Timestep Cluster Amortization) to reduce K-means overhead
# 3. Dynamic Layer Offloading to fit on 24GB GPUs
#
# Without offloading, HunyuanVideo needs ~25-30GB VRAM
# With offloading, it can run on 24GB GPUs (4090, 3090)
#
# Metrics Monitoring:
# - CTCA statistics (full re-clustering vs centroid-only updates)
# - Offloading statistics (GPU loads, prefetch hit rate)
# - Memory usage and total inference time
# =============================================================================

set -e  # Exit on error

# Timestamp for this run
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
RUN_ID="ctca_offload_${TIMESTAMP}"

# =============================================================================
# Model Configuration
# =============================================================================
model_id="tencent/HunyuanVideo"
resolution="720p"
height=720
width=1280
num_frames=129
num_inference_steps=50

# =============================================================================
# SAP Configuration
# =============================================================================
qc_kmeans=400                    # Number of query centroids
kc_kmeans=1000                   # Number of key centroids
top_p_k=0.9                      # Top-p threshold for block selection
min_kc_ratio=0.10                # Minimum key blocks ratio
kmeans_iter_init=50              # K-means iterations for initialization
kmeans_iter_step=2               # K-means iterations per step

# =============================================================================
# CTCA Configuration (Cross-Timestep Cluster Amortization)
# =============================================================================
ctca_quality_threshold=0.80      # Quality threshold for re-clustering
ctca_min_interval=2              # Minimum calls between re-clustering
ctca_max_interval=10             # Maximum calls to reuse clusters
ctca_adaptive=true               # Use quality-based adaptive re-clustering
ctca_verbose=false               # Enable verbose CTCA logging

# =============================================================================
# Offloading Configuration
# =============================================================================
enable_offload=true              # Enable dynamic layer offloading
offload_pinned_memory=true       # Use pinned memory for faster transfers
offload_prefetch=true            # Enable prefetching for next layer

# =============================================================================
# Warmup Configuration
# =============================================================================
first_times_fp=0.1               # Dense attention for first 10% timesteps
first_layers_fp=0.03             # Dense attention for first 3% layers

# =============================================================================
# Output Configuration
# =============================================================================
output_dir="outputs/hyvideo_sap_ctca_offload"
logging_dir="logs/hyvideo_sap_ctca_offload"
metrics_file="${logging_dir}/metrics_${RUN_ID}.log"

# Prompt
prompt="${1:-A cat walks on the grass, realistic style, high quality}"

# =============================================================================
# Setup
# =============================================================================
mkdir -p "$output_dir"
mkdir -p "$logging_dir"

echo "============================================================"
echo "SAP + CTCA + Offloading Video Generation"
echo "============================================================"
echo "Run ID:          ${RUN_ID}"
echo "Timestamp:       $(date)"
echo "Resolution:      ${width}x${height}"
echo "Frames:          ${num_frames}"
echo "Steps:           ${num_inference_steps}"
echo "Q Centroids:     ${qc_kmeans}"
echo "K Centroids:     ${kc_kmeans}"
echo "CTCA Quality:    ${ctca_quality_threshold}"
echo "CTCA Min Intv:   ${ctca_min_interval}"
echo "CTCA Max Intv:   ${ctca_max_interval}"
echo "Offloading:      ${enable_offload}"
echo "Pinned Memory:   ${offload_pinned_memory}"
echo "Prefetch:        ${offload_prefetch}"
echo "Prompt:          ${prompt}"
echo "============================================================"

# Save configuration to metrics file
cat > "$metrics_file" << EOF
# CTCA + Offload Run Configuration
# Run ID: ${RUN_ID}
# Timestamp: $(date -Iseconds)

[Configuration]
model_id=${model_id}
resolution=${resolution}
height=${height}
width=${width}
num_frames=${num_frames}
num_inference_steps=${num_inference_steps}
qc_kmeans=${qc_kmeans}
kc_kmeans=${kc_kmeans}
top_p_k=${top_p_k}
min_kc_ratio=${min_kc_ratio}
ctca_quality_threshold=${ctca_quality_threshold}
ctca_min_interval=${ctca_min_interval}
ctca_max_interval=${ctca_max_interval}
ctca_adaptive=${ctca_adaptive}
enable_offload=${enable_offload}
offload_pinned_memory=${offload_pinned_memory}
offload_prefetch=${offload_prefetch}
prompt=${prompt}

[Timing]
EOF

# Record start time
start_time=$(date +%s.%N)
echo "start_time=$(date -Iseconds)" >> "$metrics_file"

# =============================================================================
# Build Command
# =============================================================================
cmd="python hyvideo_t2v_inference.py \
    --model_id $model_id \
    --pattern SAP_CTCA \
    --height $height \
    --width $width \
    --num_frames $num_frames \
    --num_inference_steps $num_inference_steps \
    --prompt \"$prompt\" \
    --output_file \"${output_dir}/output_${RUN_ID}.mp4\" \
    --logging_file \"${logging_dir}/density_${RUN_ID}.jsonl\" \
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
    --seed 42"

# Add optional flags
if [ "$ctca_adaptive" = true ]; then
    cmd="$cmd --ctca_adaptive"
fi

if [ "$ctca_verbose" = true ]; then
    cmd="$cmd --ctca_verbose"
fi

if [ "$enable_offload" = true ]; then
    cmd="$cmd --enable_offload"
fi

if [ "$offload_pinned_memory" = true ]; then
    cmd="$cmd --offload_pinned_memory"
fi

if [ "$offload_prefetch" = true ]; then
    cmd="$cmd --offload_prefetch"
fi

# =============================================================================
# Run Inference and Capture Output
# =============================================================================
echo ""
echo "Running inference with offloading..."
echo ""

# Run and capture output
temp_output="${logging_dir}/output_${RUN_ID}.txt"
eval "$cmd" 2>&1 | tee "$temp_output"

# Record end time
end_time=$(date +%s.%N)
echo "end_time=$(date -Iseconds)" >> "$metrics_file"

# Calculate duration
duration=$(echo "$end_time - $start_time" | bc)
echo "duration_seconds=${duration}" >> "$metrics_file"

# =============================================================================
# Parse and Save CTCA Statistics
# =============================================================================
echo "" >> "$metrics_file"
echo "[CTCA_Statistics]" >> "$metrics_file"

# Extract CTCA statistics from output
if grep -q "CTCA.*Statistics" "$temp_output"; then
    total_calls=$(grep "Total clustering calls:" "$temp_output" | awk '{print $NF}' || echo "N/A")
    full_recluster=$(grep "Full re-clustering:" "$temp_output" | awk '{print $3}' || echo "N/A")
    full_recluster_pct=$(grep "Full re-clustering:" "$temp_output" | grep -oP '\d+\.\d+%' || echo "N/A")
    quality_triggered=$(grep "Quality triggered:" "$temp_output" | awk '{print $NF}' || echo "N/A")
    interval_triggered=$(grep "Interval triggered:" "$temp_output" | awk '{print $NF}' || echo "N/A")
    centroid_only=$(grep "Centroid update only:" "$temp_output" | awk '{print $4}' || echo "N/A")
    centroid_only_pct=$(grep "Centroid update only:" "$temp_output" | grep -oP '\d+\.\d+%' || echo "N/A")
    kmeans_speedup=$(grep "Estimated K-means speedup:" "$temp_output" | awk '{print $NF}' || echo "N/A")

    echo "total_clustering_calls=${total_calls}" >> "$metrics_file"
    echo "full_reclustering=${full_recluster}" >> "$metrics_file"
    echo "full_reclustering_pct=${full_recluster_pct}" >> "$metrics_file"
    echo "quality_triggered=${quality_triggered}" >> "$metrics_file"
    echo "interval_triggered=${interval_triggered}" >> "$metrics_file"
    echo "centroid_update_only=${centroid_only}" >> "$metrics_file"
    echo "centroid_update_only_pct=${centroid_only_pct}" >> "$metrics_file"
    echo "kmeans_speedup=${kmeans_speedup}" >> "$metrics_file"
fi

# Extract Dynamic Offloading statistics
if grep -q "Dynamic Layer Offloading" "$temp_output"; then
    echo "" >> "$metrics_file"
    echo "[Offloading_Statistics]" >> "$metrics_file"

    total_layers=$(grep "Total layers:" "$temp_output" | awk '{print $NF}' || echo "N/A")
    layers_on_gpu=$(grep "Layers kept on GPU:" "$temp_output" | awk '{print $NF}' || echo "N/A")
    layer_memory=$(grep "Layer memory:" "$temp_output" | awk '{print $3}' || echo "N/A")
    gpu_loads=$(grep "GPU loads:" "$temp_output" | awk '{print $NF}' || echo "N/A")
    gpu_offloads=$(grep "GPU offloads:" "$temp_output" | awk '{print $NF}' || echo "N/A")
    window_slides=$(grep "Window slides:" "$temp_output" | awk '{print $NF}' || echo "N/A")
    prefetch_hits=$(grep "Prefetch hits:" "$temp_output" | awk '{print $NF}' || echo "N/A")
    prefetch_misses=$(grep "Prefetch misses:" "$temp_output" | awk '{print $NF}' || echo "N/A")
    prefetch_ratio=$(grep "Prefetch hit ratio:" "$temp_output" | awk '{print $NF}' || echo "N/A")
    cache_clears=$(grep "Cache clears:" "$temp_output" | awk '{print $NF}' || echo "N/A")

    echo "total_layers=${total_layers}" >> "$metrics_file"
    echo "layers_on_gpu=${layers_on_gpu}" >> "$metrics_file"
    echo "layer_memory=${layer_memory}" >> "$metrics_file"
    echo "gpu_loads=${gpu_loads}" >> "$metrics_file"
    echo "gpu_offloads=${gpu_offloads}" >> "$metrics_file"
    echo "window_slides=${window_slides}" >> "$metrics_file"
    echo "prefetch_hits=${prefetch_hits}" >> "$metrics_file"
    echo "prefetch_misses=${prefetch_misses}" >> "$metrics_file"
    echo "prefetch_hit_ratio=${prefetch_ratio}" >> "$metrics_file"
    echo "cache_clears=${cache_clears}" >> "$metrics_file"
fi

# =============================================================================
# Summary
# =============================================================================
echo ""
echo "============================================================"
echo "                    RUN SUMMARY"
echo "============================================================"
echo "Run ID:              ${RUN_ID}"
echo "Duration:            ${duration} seconds"
echo "Output Video:        ${output_dir}/output_${RUN_ID}.mp4"
echo "Metrics Log:         ${metrics_file}"
echo "Density Log:         ${logging_dir}/density_${RUN_ID}.jsonl"
echo "Raw Output:          ${temp_output}"
echo "------------------------------------------------------------"

if [ -n "$kmeans_speedup" ] && [ "$kmeans_speedup" != "N/A" ]; then
    echo "CTCA Performance:"
    echo "  Total Calls:       ${total_calls}"
    echo "  Full Re-cluster:   ${full_recluster} (${full_recluster_pct})"
    echo "  Centroid Only:     ${centroid_only} (${centroid_only_pct})"
    echo "  Quality Triggers:  ${quality_triggered}"
    echo "  Interval Triggers: ${interval_triggered}"
    echo "  K-means Speedup:   ${kmeans_speedup}"

    # Performance assessment
    speedup_num=$(echo "$kmeans_speedup" | tr -d 'x')
    if (( $(echo "$speedup_num >= 5" | bc -l) )); then
        echo "  Assessment:        EXCELLENT (>= 5x speedup)"
    elif (( $(echo "$speedup_num >= 3" | bc -l) )); then
        echo "  Assessment:        GOOD (3-5x speedup)"
    elif (( $(echo "$speedup_num >= 2" | bc -l) )); then
        echo "  Assessment:        MODERATE (2-3x speedup)"
    else
        echo "  Assessment:        NEEDS TUNING (< 2x speedup)"
        echo ""
        echo "  Suggestions:"
        echo "  - Increase ctca_max_interval (current: ${ctca_max_interval})"
        echo "  - Lower ctca_quality_threshold (current: ${ctca_quality_threshold})"
    fi
    echo ""
fi

if [ -n "$prefetch_ratio" ] && [ "$prefetch_ratio" != "N/A" ]; then
    echo "Offloading Performance:"
    echo "  Total Layers:      ${total_layers}"
    echo "  Layers on GPU:     ${layers_on_gpu}"
    echo "  GPU Loads:         ${gpu_loads}"
    echo "  GPU Offloads:      ${gpu_offloads}"
    echo "  Prefetch Hits:     ${prefetch_hits}"
    echo "  Prefetch Misses:   ${prefetch_misses}"
    echo "  Prefetch Ratio:    ${prefetch_ratio}"

    # Offload efficiency assessment
    ratio_num=$(echo "$prefetch_ratio" | tr -d '%')
    if (( $(echo "$ratio_num >= 80" | bc -l) )); then
        echo "  Assessment:        EXCELLENT (>= 80% prefetch hits)"
    elif (( $(echo "$ratio_num >= 60" | bc -l) )); then
        echo "  Assessment:        GOOD (60-80% prefetch hits)"
    elif (( $(echo "$ratio_num >= 40" | bc -l) )); then
        echo "  Assessment:        MODERATE (40-60% prefetch hits)"
    else
        echo "  Assessment:        NEEDS TUNING (< 40% prefetch hits)"
    fi
fi

echo "============================================================"
echo ""

# =============================================================================
# JSON Summary
# =============================================================================
json_file="${logging_dir}/summary_${RUN_ID}.json"
cat > "$json_file" << EOF
{
    "run_id": "${RUN_ID}",
    "timestamp": "$(date -Iseconds)",
    "config": {
        "resolution": "${width}x${height}",
        "frames": ${num_frames},
        "steps": ${num_inference_steps},
        "q_centroids": ${qc_kmeans},
        "k_centroids": ${kc_kmeans},
        "ctca_quality_threshold": ${ctca_quality_threshold},
        "ctca_min_interval": ${ctca_min_interval},
        "ctca_max_interval": ${ctca_max_interval},
        "offload_enabled": ${enable_offload},
        "offload_pinned": ${offload_pinned_memory},
        "offload_prefetch": ${offload_prefetch}
    },
    "ctca_results": {
        "total_clustering_calls": "${total_calls}",
        "full_reclustering": "${full_recluster}",
        "centroid_update_only": "${centroid_only}",
        "quality_triggered": "${quality_triggered}",
        "interval_triggered": "${interval_triggered}",
        "kmeans_speedup": "${kmeans_speedup}"
    },
    "offload_results": {
        "total_layers": "${total_layers}",
        "layers_on_gpu": "${layers_on_gpu}",
        "gpu_loads": "${gpu_loads}",
        "gpu_offloads": "${gpu_offloads}",
        "prefetch_hits": "${prefetch_hits}",
        "prefetch_misses": "${prefetch_misses}",
        "prefetch_ratio": "${prefetch_ratio}"
    },
    "timing": {
        "duration_seconds": ${duration}
    },
    "outputs": {
        "video": "${output_dir}/output_${RUN_ID}.mp4",
        "metrics": "${metrics_file}",
        "density_log": "${logging_dir}/density_${RUN_ID}.jsonl"
    }
}
EOF

echo "JSON summary saved to: ${json_file}"
