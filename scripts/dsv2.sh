#!/bin/bash

set -euo pipefail

# Edit these shared roots to match your environment.
MODEL_ROOT="${MODEL_ROOT:-/path/to/model}"
HF_CACHE_ROOT="${HF_CACHE_ROOT:-/path/to/huggingface}"
RUN_ROOT="${RUN_ROOT:-/path/to/experiments/deepseekv2/moe_wanda}"

# Keep paths without trailing '/' to avoid dynamic-module cache key collisions.
model="${MODEL:-${MODEL_ROOT}/models--deepseek-ai--DeepSeek-V2-Lite/snapshots/604d5664dddd88a0433dbae533b7fe9472482de0}"
sparsity_ratio="${SPARSITY_RATIO:-0.75}"
cuda_device="${CUDA_DEVICE:-1}"
seed="${SEED:-0}"
nsamples="${NSAMPLES:-512}"
output_dir="${OUTPUT_DIR:-${RUN_ROOT}/output}"
checkpoint_dir="${CHECKPOINT_DIR:-${RUN_ROOT}/ckpt}"

# Set CUDA device visibility
# export CUDA_HOME=/path/to/cuda
# export PATH=$CUDA_HOME/bin:$PATH
export CUDA_VISIBLE_DEVICES="$cuda_device"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-${HF_CACHE_ROOT}/datasets}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-${HF_CACHE_ROOT}/hub}"

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
