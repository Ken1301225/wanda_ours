#!/bin/bash

# Set common variables
base_dir="/data1/data/kangborui/gujinrui/wanda"
hf_home="$base_dir/huggingface"
model_repo="Qwen/Qwen1.5-MoE-A2.7B"
model="$hf_home/hub/models--Qwen--Qwen1.5-MoE-A2.7B"
sparsity_ratio=0.75
cuda_device=5
seed=0
routing_mode="dense_softmax"
routing_power=2.0

# Set CUDA device visibility
# export CUDA_HOME=/data1/ldk/env/dkllm
# export PATH=$CUDA_HOME/bin:$PATH
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


# Define function to run python command
run_python_command () {
    python main.py \
    --model $model \
    --prune_method $1 \
    --sparsity_ratio $sparsity_ratio \
    --sparsity_type $2 \
    --seed $seed \
    --save $3 \
    --save_model $4 \
    --nsamples 128 \
    --moe_wanda_routing_mode $routing_mode \
    --moe_wanda_routing_power $routing_power
}



echo "Running with MoE-Wanda pruning method"
run_python_command "moe_wanda" "unstructured" "$base_dir/output/moe_wanda" "$base_dir/checkpoints/moe_wanda"
echo "Finished MoE-Wanda pruning method"
