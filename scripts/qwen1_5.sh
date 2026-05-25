#!/bin/bash

set -euo pipefail

model="${MODEL:-Qwen/Qwen1.5-MoE-A2.7B}"
sparsity_ratio="${SPARSITY_RATIO:-0.75}"
cuda_device="${CUDA_DEVICE:-0}"
seed="${SEED:-0}"
down_proj_max_col_zero_ratio="${DOWN_PROJ_MAX_COL_ZERO_RATIO:-0.8}"
timestamp=$(date +"%Y%m%d_%H%M%S")

run_root="${RUN_ROOT:-outputs/qwen1_5/moe_wanda}"
output_dir="${OUTPUT_DIR:-${run_root}/output_${timestamp}}"
checkpoint_dir="${CHECKPOINT_DIR:-${run_root}/ckpt_${timestamp}}"

export CUDA_VISIBLE_DEVICES="$cuda_device"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-$HOME/.cache/huggingface/datasets}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-$HOME/.cache/huggingface/hub}"

mkdir -p "$output_dir" "$checkpoint_dir"

python main.py \
    --model "$model" \
    --prune_method moe_wanda \
    --sparsity_ratio "$sparsity_ratio" \
    --sparsity_type unstructured \
    --seed "$seed" \
    --down_proj_max_col_zero_ratio "$down_proj_max_col_zero_ratio" \
    --save "$output_dir" \
    --save_model "$checkpoint_dir" \
    --nsamples 128
