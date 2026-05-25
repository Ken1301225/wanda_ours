#!/bin/bash

set -euo pipefail

# Keep paths without trailing '/' to avoid dynamic-module cache key collisions.
model="${MODEL:-deepseek-ai/DeepSeek-V2-Lite}"
sparsity_ratio="${SPARSITY_RATIO:-0.75}"
cuda_device="${CUDA_DEVICE:-1}"
seed="${SEED:-0}"
nsamples="${NSAMPLES:-512}"
run_root="${RUN_ROOT:-outputs/deepseekv2/moe_wanda}"
output_dir="${OUTPUT_DIR:-${run_root}/output}"
checkpoint_dir="${CHECKPOINT_DIR:-${run_root}/ckpt}"

# Set CUDA device visibility
# export CUDA_HOME=/path/to/cuda
# export PATH=$CUDA_HOME/bin:$PATH
export CUDA_VISIBLE_DEVICES="$cuda_device"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-$HOME/.cache/huggingface/datasets}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-$HOME/.cache/huggingface/hub}"

mkdir -p "$output_dir" "$checkpoint_dir"

# Define function to run python command
run_python_command () {
    python main_dsv2.py \
    --model "$model" \
    --prune_method "$1" \
    --sparsity_ratio "$sparsity_ratio" \
    --sparsity_type "$2" \
    --seed "$seed" \
    --save "$3" \
    --save_model "$4" \
    --nsamples "$nsamples"
}



echo "Running with MoE-Wanda pruning method"
run_python_command "moe_wanda" "unstructured" "$output_dir" "$checkpoint_dir"
echo "Finished MoE-Wanda pruning method"
