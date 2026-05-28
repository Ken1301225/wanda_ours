#!/bin/bash

set -euo pipefail

# Edit these shared roots to match your environment.
MODEL_ROOT="${MODEL_ROOT:-/data1/ldk/model}"
HF_CACHE_ROOT="${HF_CACHE_ROOT:-/data1/ldk/huggingface}"
RUN_ROOT="${RUN_ROOT:-/data1/ldk/nlp/wanda_moe_dense_soft}"

MODEL_REPO="${MODEL_REPO:-Qwen/Qwen1.5-MoE-A2.7B}"
MODEL="${MODEL:-${MODEL_ROOT}/Qwen1.5/models--Qwen--Qwen1.5-MoE-A2.7B/snapshots/1a758c50ecb6350748b9ce0a99d2352fd9fc11c9}"
SPARSITY_RATIO="${SPARSITY_RATIO:-0.5}"
CUDA_DEVICE="${CUDA_DEVICE:-0}"
SEED="${SEED:-0}"
NSAMPLES="${NSAMPLES:-128}"
ROUTING_MODE="${ROUTING_MODE:-dense_softmax}"
ROUTING_POWER="${ROUTING_POWER:-1.5}"
timestamp=$(date +"%Y%m%d_%H%M%S")

OUTPUT_DIR="${OUTPUT_DIR:-${RUN_ROOT}/output_${timestamp}}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-${RUN_ROOT}/ckpt_${timestamp}}"
CLUSTER_EXPERTS="${CLUSTER_EXPERTS:-false}"
CLUSTER_K="${CLUSTER_K:-15}"

export CUDA_VISIBLE_DEVICES="$CUDA_DEVICE"
export HF_HOME="${HF_HOME:-$HF_CACHE_ROOT}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-${HF_CACHE_ROOT}/datasets}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-${HF_CACHE_ROOT}/hub}"

mkdir -p "$OUTPUT_DIR" "$CHECKPOINT_DIR"

CLUSTER_ARGS=(--moe_wanda_cluster_k "$CLUSTER_K")
if [ "$CLUSTER_EXPERTS" = "true" ]; then
    CLUSTER_ARGS+=(--moe_wanda_cluster_experts)
fi

if [ ! -d "$MODEL" ]; then
    echo "Model cache not found: $MODEL"
    echo "Download with: huggingface-cli download $MODEL_REPO --local-dir $MODEL"
    exit 1
fi

python main.py \
    --model "$MODEL" \
    --prune_method moe_wanda \
    --sparsity_ratio "$SPARSITY_RATIO" \
    --sparsity_type unstructured \
    --seed "$SEED" \
    --save "$OUTPUT_DIR" \
    --save_model "$CHECKPOINT_DIR" \
    --nsamples "$NSAMPLES" \
    --moe_wanda_routing_mode "$ROUTING_MODE" \
    --moe_wanda_routing_power "$ROUTING_POWER" \
    "${CLUSTER_ARGS[@]}"
