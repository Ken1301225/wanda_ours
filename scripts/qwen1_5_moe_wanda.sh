#!/bin/bash

# Set common variables
base_dir="/data1/data/kangborui/gujinrui/wanda"
hf_home="$base_dir/huggingface"
model_repo="Qwen/Qwen1.5-MoE-A2.7B"
model="$hf_home/hub/models--Qwen--Qwen1.5-MoE-A2.7B"
sparsity_ratio=0.5
cuda_device=5
seed=0

# Set CUDA device visibility
export CUDA_VISIBLE_DEVICES=$cuda_device
export HF_HOME="$hf_home"
export HF_HUB_CACHE="$hf_home/hub"
export HF_DATASETS_CACHE="$hf_home/datasets"

mkdir -p "$HF_HUB_CACHE" "$HF_DATASETS_CACHE" "$base_dir/output" "$base_dir/checkpoints"

if [ ! -d "$model" ]; then
    echo "Model cache not found: $model"
    echo "Download with: huggingface-cli download $model_repo --local-dir $model"
    exit 1
fi

run_python_command () {
    python main.py \
    --model $model \
    --prune_method wanda \
    --sparsity_ratio $sparsity_ratio \
    --sparsity_type $1 \
    --seed $seed \
    --save $2 \
    --save_model $3 \
    --nsamples 128
}

echo "Running Qwen1.5 MoE Wanda pruning"
run_python_command "unstructured" "$base_dir/output/qwen1_5_moe_wanda" "$base_dir/checkpoints/qwen1_5_moe_wanda"
echo "Finished Qwen1.5 MoE Wanda pruning"
