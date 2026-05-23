#!/bin/bash

model="/data1/ldk/model/Qwen1.5/models--Qwen--Qwen1.5-MoE-A2.7B/snapshots/1a758c50ecb6350748b9ce0a99d2352fd9fc11c9/"
sparsity_ratio=0.75
cuda_device=0
seed=0
down_proj_max_col_zero_ratio=0.8
timestamp=$(date +"%Y%m%d_%H%M%S")

export CUDA_VISIBLE_DEVICES="$cuda_device"
export HF_DATASETS_CACHE="/data1/ldk/huggingface/datasets"
export HF_HUB_CACHE="/data1/ldk/huggingface/hub"

output_dir="/data1/ldk/SPNN/qwen1_5/moe_wanda/output_${timestamp}/"
checkpoint_dir="/data1/ldk/SPNN/qwen1_5/moe_wanda/ckpt_${timestamp}/"

mkdir -p "$output_dir" "$checkpoint_dir"

if [ ! -d "$model" ]; then
    echo "Model path not found: $model"
    exit 1
fi

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
