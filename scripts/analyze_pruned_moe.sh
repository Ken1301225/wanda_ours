#!/bin/bash

set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
python_bin="${PYTHON_BIN:-python}"

# Edit these paths before running. Add more entries to compare multiple pruned models.
model_paths=(
    "/path/to/pruned_model"
    # "/path/to/another_pruned_model"
)

output_dir="${OUTPUT_DIR:-$repo_root/debug_outputs/pruned_moe_analysis}"
trust_remote_code="${TRUST_REMOTE_CODE:-1}"
max_pattern_plots="${MAX_PATTERN_PLOTS:-9}"
dpi="${DPI:-180}"
pattern_max_side="${PATTERN_MAX_SIDE:-512}"

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
    "--max-pattern-plots" "$max_pattern_plots"
    "--dpi" "$dpi"
    "--pattern-max-side" "$pattern_max_side"
)

if [ "$trust_remote_code" = "1" ]; then
    cmd+=("--trust-remote-code")
fi

for model_path in "${model_paths[@]}"; do
    cmd+=("--model" "$model_path")
done

echo "Writing analysis artifacts to: $output_dir"
printf 'Analyzing model: %s\n' "${model_paths[@]}"

"${cmd[@]}"
