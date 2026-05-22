#!/bin/bash

set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
python_bin="${PYTHON_BIN:-python}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/matplotlib}"

# Edit these paths before running. Add more entries to compare multiple pruned models.
# This script runs the static zero-column collapse analysis in debug/analyze_pruned_moe.py.
model_paths=(
    "/path/to/pruned_model"
    # "/path/to/another_pruned_model"
)

output_dir="${OUTPUT_DIR:-$repo_root/debug_outputs/pruned_moe_analysis}"
trust_remote_code="${TRUST_REMOTE_CODE:-1}"
dpi="${DPI:-180}"
significant_threshold="${SIGNIFICANT_THRESHOLD:-0.8}"

if [ "${#model_paths[@]}" -eq 0 ]; then
    echo "No model paths configured. Edit model_paths in scripts/analyze_pruned_moe.sh."
    exit 1
fi

for model_path in "${model_paths[@]}"; do
    if [ ! -d "$model_path" ]; then
        echo "Model directory not found: $model_path"
        exit 1
    fi
done

mkdir -p "$output_dir"

cmd=(
    "$python_bin"
    "$repo_root/debug/analyze_pruned_moe.py"
    "--output-dir" "$output_dir"
    "--dpi" "$dpi"
    "--significant-threshold" "$significant_threshold"
)

if [ "$trust_remote_code" = "1" ]; then
    cmd+=("--trust-remote-code")
fi

for model_path in "${model_paths[@]}"; do
    cmd+=("--model" "$model_path")
done

echo "Writing significant sparse-structure analysis artifacts to: $output_dir"
printf 'Analyzing model: %s\n' "${model_paths[@]}"
echo "Using significant sparse threshold: $significant_threshold"
echo "Key outputs: summary.txt, depth_projection_summary.txt, layer_projection_significant_zero_col.png, expert_significant_zero_col_heatmap_*.png, weight_overview_heatmap.png"

"${cmd[@]}"
