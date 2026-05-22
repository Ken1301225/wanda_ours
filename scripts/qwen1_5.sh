#!/bin/bash

# Set common variables
model="/data1/ldk/model/Qwen1.5/models--Qwen--Qwen1.5-MoE-A2.7B/snapshots/1a758c50ecb6350748b9ce0a99d2352fd9fc11c9/"
sparsity_ratio=0.3
cuda_device=1
seed=0
timestamp=$(date +"%Y%m%d_%H%M%S")

# Set CUDA device visibility
# export CUDA_HOME=/data1/ldk/env/dkllm
# export PATH=$CUDA_HOME/bin:$PATH
export CUDA_VISIBLE_DEVICES=$cuda_device
export HF_DATASETS_CACHE="/data1/ldk/huggingface/datasets"
export HF_HUB_CACHE="/data1/ldk/huggingface/hub"


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
    --nsamples 128
}



echo "Running with wanda pruning method"
run_python_command "wanda" "unstructured" "/data1/ldk/nlp/wanda_moe/output_${timestamp}/" "/data1/ldk/nlp/wanda_moe/ckpt_${timestamp}/"
# run_python_command "sparsegpt" "unstructured" "/data1/ldk/SPNN/qwen1_5/wanda/output9/" "/data1/ldk/SPNN/qwen1_5/wanda/ckpt9/"
# run_python_command "ablate_wanda_seq" "unstructured" "/data1/ldk/SPNN/qwen1_5/wanda/output10/" "/data1/ldk/SPNN/qwen1_5/wanda/ckpt10/"
# run_python_command "wanda" "unstructured" 
echo "Finished wanda pruning method"

