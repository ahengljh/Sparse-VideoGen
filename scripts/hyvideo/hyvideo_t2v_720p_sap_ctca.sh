#!/bin/bash

# =============================================================================
# SAP with Cross-Timestep Cluster Amortization (CTCA)
# =============================================================================
# This script runs HunyuanVideo with SAP+CTCA for accelerated video generation
# CTCA reduces K-means clustering overhead by 5-10x by reusing cluster assignments
#
# Metrics Monitoring:
# - CTCA statistics (full re-clustering vs centroid-only updates)
# - K-means speedup estimation
# - Quality trigger events
# - Total inference time
# =============================================================================

set -e  # Exit on error

# Timestamp for this run
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
RUN_ID="ctca_${TIMESTAMP}"

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
ctca_quality_threshold=0.80      # Quality threshold (0-1) for triggering re-clustering
ctca_min_interval=2              # Minimum calls between re-clustering (prevents thrashing)
ctca_max_interval=10             # Maximum calls to reuse clusters (forces refresh)
ctca_adaptive=true               # Use quality-based adaptive re-clustering
ctca_verbose=false               # Enable verbose CTCA logging

# =============================================================================
# Warmup Configuration
# =============================================================================
first_times_fp=0.1               # Dense attention for first 10% timesteps
first_layers_fp=0.03             # Dense attention for first 3% layers

# =============================================================================
# Output Configuration
# =============================================================================
output_dir="outputs/hyvideo_sap_ctca"
logging_dir="logs/hyvideo_sap_ctca"
metrics_file="${logging_dir}/metrics_${RUN_ID}.log"

# Prompt
prompt="${1:-A cat walks on the grass, realistic style, high quality}"

# =============================================================================
# Setup
# =============================================================================
mkdir -p "$output_dir"
mkdir -p "$logging_dir"

echo "============================================================"
echo "SAP + CTCA Video Generation"
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
echo "Prompt:          ${prompt}"
echo "============================================================"

# Save configuration to metrics file
cat > "$metrics_file" << EOF
# CTCA Run Configuration
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
kmeans_iter_init=${kmeans_iter_init}
kmeans_iter_step=${kmeans_iter_step}
ctca_quality_threshold=${ctca_quality_threshold}
ctca_min_interval=${ctca_min_interval}
ctca_max_interval=${ctca_max_interval}
ctca_adaptive=${ctca_adaptive}
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

# =============================================================================
# Run Inference and Capture Output
# =============================================================================
echo ""
echo "Running inference..."
echo ""

# Run and capture both stdout and stderr, while also displaying to terminal
# Tee to a temporary file for parsing CTCA statistics
temp_output="${logging_dir}/output_${RUN_ID}.txt"
eval "$cmd" 2>&1 | tee "$temp_output"

# Record end time
end_time=$(date +%s.%N)
echo "end_time=$(date -Iseconds)" >> "$metrics_file"

# Calculate duration (use awk instead of bc for portability)
duration=$(awk "BEGIN {printf \"%.2f\", $end_time - $start_time}")
echo "duration_seconds=${duration}" >> "$metrics_file"

# =============================================================================
# Parse and Save CTCA Statistics
# =============================================================================
echo "" >> "$metrics_file"
echo "[CTCA_Statistics]" >> "$metrics_file"

# Extract CTCA statistics from output
if grep -q "CTCA.*Statistics" "$temp_output"; then
    # Parse key metrics
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

# Extract Dynamic Offloading statistics if present
if grep -q "Dynamic Layer Offloading" "$temp_output"; then
    echo "" >> "$metrics_file"
    echo "[Offloading_Statistics]" >> "$metrics_file"

    gpu_loads=$(grep "GPU loads:" "$temp_output" | awk '{print $NF}' || echo "N/A")
    gpu_offloads=$(grep "GPU offloads:" "$temp_output" | awk '{print $NF}' || echo "N/A")
    prefetch_hits=$(grep "Prefetch hits:" "$temp_output" | awk '{print $NF}' || echo "N/A")
    prefetch_misses=$(grep "Prefetch misses:" "$temp_output" | awk '{print $NF}' || echo "N/A")
    prefetch_ratio=$(grep "Prefetch hit ratio:" "$temp_output" | awk '{print $NF}' || echo "N/A")

    echo "gpu_loads=${gpu_loads}" >> "$metrics_file"
    echo "gpu_offloads=${gpu_offloads}" >> "$metrics_file"
    echo "prefetch_hits=${prefetch_hits}" >> "$metrics_file"
    echo "prefetch_misses=${prefetch_misses}" >> "$metrics_file"
    echo "prefetch_hit_ratio=${prefetch_ratio}" >> "$metrics_file"
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

    # Performance assessment (use awk for float comparison)
    speedup_num=$(echo "$kmeans_speedup" | tr -d 'x')
    assessment=$(awk -v s="$speedup_num" 'BEGIN {
        if (s >= 5) print "EXCELLENT"
        else if (s >= 3) print "GOOD"
        else if (s >= 2) print "MODERATE"
        else print "NEEDS_TUNING"
    }')

    if [ "$assessment" = "EXCELLENT" ]; then
        echo "  Assessment:        EXCELLENT (>= 5x speedup)"
    elif [ "$assessment" = "GOOD" ]; then
        echo "  Assessment:        GOOD (3-5x speedup)"
    elif [ "$assessment" = "MODERATE" ]; then
        echo "  Assessment:        MODERATE (2-3x speedup)"
    else
        echo "  Assessment:        NEEDS TUNING (< 2x speedup)"
        echo ""
        echo "  Suggestions:"
        echo "  - Increase ctca_max_interval (current: ${ctca_max_interval})"
        echo "  - Lower ctca_quality_threshold (current: ${ctca_quality_threshold})"
    fi
fi

echo "============================================================"
echo ""

# =============================================================================
# Optional: Create a simple JSON summary for programmatic access
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
        "ctca_max_interval": ${ctca_max_interval}
    },
    "results": {
        "duration_seconds": ${duration},
        "total_clustering_calls": "${total_calls}",
        "full_reclustering": "${full_recluster}",
        "centroid_update_only": "${centroid_only}",
        "quality_triggered": "${quality_triggered}",
        "interval_triggered": "${interval_triggered}",
        "kmeans_speedup": "${kmeans_speedup}"
    },
    "outputs": {
        "video": "${output_dir}/output_${RUN_ID}.mp4",
        "metrics": "${metrics_file}",
        "density_log": "${logging_dir}/density_${RUN_ID}.jsonl"
    }
}
EOF

echo "JSON summary saved to: ${json_file}"
