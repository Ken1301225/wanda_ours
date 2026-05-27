#!/bin/bash

set -euo pipefail

# Edit these shared roots to match your environment.
MODEL_ROOT="${MODEL_ROOT:-/data1/ldk/model}"
HF_CACHE_ROOT="${HF_CACHE_ROOT:-/data1/ldk/huggingface}"
RUN_ROOT="${RUN_ROOT:-/data1/ldk/SPNN/qwen1_5/moe_wanda}"

model="${MODEL:-${MODEL_ROOT}/Qwen1.5/models--Qwen--Qwen1.5-MoE-A2.7B/snapshots/1a758c50ecb6350748b9ce0a99d2352fd9fc11c9}"
sparsity_ratio="${SPARSITY_RATIO:-0.75}"
cuda_device="${CUDA_DEVICE:-0}"
seed="${SEED:-0}"
timestamp=$(date +"%Y%m%d_%H%M%S")

output_dir="${OUTPUT_DIR:-${RUN_ROOT}/output_${timestamp}}"
checkpoint_dir="${CHECKPOINT_DIR:-${RUN_ROOT}/ckpt_${timestamp}}"

export CUDA_VISIBLE_DEVICES="$cuda_device"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-${HF_CACHE_ROOT}/datasets}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-${HF_CACHE_ROOT}/hub}"

mkdir -p "$output_dir" "$checkpoint_dir"

python main.py \
    --model "$model" \
    --prune_method moe_wanda \
    --sparsity_ratio "$sparsity_ratio" \
    --sparsity_type unstructured \
    --seed "$seed" \
    --save "$output_dir" \
    --save_model "$checkpoint_dir" \
    --nsamples 128
