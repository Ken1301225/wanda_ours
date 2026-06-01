#!/bin/bash

set -euo pipefail

# Edit these shared roots to match your environment.
MODEL_ROOT="${MODEL_ROOT:-/data1/ldk/huggingface/hub}"
HF_CACHE_ROOT="${HF_CACHE_ROOT:-/data1/ldk/huggingface}"
RUN_ROOT="${RUN_ROOT:-/data1/ldk/nlp/deepseekv2/wanda_moe}"

# Keep paths without trailing '/' to avoid dynamic-module cache key collisions.
MODEL="${MODEL:-${MODEL_ROOT}/models--deepseek-ai--DeepSeek-V2-Lite/snapshots/604d5664dddd88a0433dbae533b7fe9472482de0}"
SPARSITY_RATIO="${SPARSITY_RATIO:-0.75}"
CUDA_DEVICE="${CUDA_DEVICE:-1}"
SEED="${SEED:-0}"
NSAMPLES="${NSAMPLES:-512}"
ROUTING_MODE="${ROUTING_MODE:-dense_softmax}"
ROUTING_POWER="${ROUTING_POWER:-2.0}"
CLUSTER_EXPERTS="${CLUSTER_EXPERTS:-false}"
CLUSTER_K="${CLUSTER_K:-15}"
SAVE_MODEL="${SAVE_MODEL:-false}"
DIAGNOSTICS="${DIAGNOSTICS:-true}"
OUTPUT_DIR="${OUTPUT_DIR:-${RUN_ROOT}/output}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-${RUN_ROOT}/ckpt}"

export CUDA_VISIBLE_DEVICES="$CUDA_DEVICE"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-${HF_CACHE_ROOT}/datasets}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-${HF_CACHE_ROOT}/hub}"

mkdir -p "$OUTPUT_DIR"

CLUSTER_ARGS=(--moe_wanda_cluster_k "$CLUSTER_K")
if [ "$CLUSTER_EXPERTS" = "true" ]; then
    CLUSTER_ARGS+=(--moe_wanda_cluster_experts)
fi

DIAGNOSTIC_ARGS=()
if [ "$DIAGNOSTICS" = "false" ]; then
    DIAGNOSTIC_ARGS=(--no_diagnostics)
fi

SAVE_ARGS=()
if [ "$SAVE_MODEL" = "true" ]; then
    mkdir -p "$CHECKPOINT_DIR"
    SAVE_ARGS=(--save_model "$CHECKPOINT_DIR")
fi

python main_dsv2.py \
    --model "$MODEL" \
    --prune_method moe_wanda \
    --sparsity_ratio "$SPARSITY_RATIO" \
    --sparsity_type unstructured \
    --seed "$SEED" \
    --save "$OUTPUT_DIR" \
    --nsamples "$NSAMPLES" \
    --moe_wanda_routing_mode "$ROUTING_MODE" \
    --moe_wanda_routing_power "$ROUTING_POWER" \
    "${CLUSTER_ARGS[@]}" \
    "${DIAGNOSTIC_ARGS[@]}" \
    "${SAVE_ARGS[@]}"
