#!/usr/bin/env bash
set -euo pipefail

MODEL_ROOT="${MODEL_ROOT:-/data1/ldk/huggingface/hub}"
HF_CACHE_ROOT="${HF_CACHE_ROOT:-/data1/ldk/huggingface}"
RUN_ROOT="${RUN_ROOT:-/data1/ldk/nlp/deepseekv2/wanda_moe_multi_sparsity}"

MODEL_REPO="${MODEL_REPO:-deepseek-ai/DeepSeek-V2-Lite}"
MODEL="${MODEL:-${MODEL_ROOT}/models--deepseek-ai--DeepSeek-V2-Lite/snapshots/604d5664dddd88a0433dbae533b7fe9472482de0}"
SPARSITY_RATIOS="${SPARSITY_RATIOS:-0.5}"
SPARSITY_TYPE="${SPARSITY_TYPE:-2:4}"
CUDA_DEVICE="${CUDA_DEVICE:-3}"
SEED="${SEED:-0}"
NSAMPLES="${NSAMPLES:-512}"
ROUTING_MODE="${ROUTING_MODE:-dense_softmax}"
ROUTING_POWER="${ROUTING_POWER:-1.0}"
CLUSTER_EXPERTS="${CLUSTER_EXPERTS:-true}"
CLUSTER_K="${CLUSTER_K:-5}"
SAVE_MODEL="${SAVE_MODEL:-true}"
DIAGNOSTICS="${DIAGNOSTICS:-true}"
SKIP_EXISTING="${SKIP_EXISTING:-true}"

timestamp=$(date +"%Y%m%d_%H%M%S")
RUN_NAME="${RUN_NAME:-multi_sparsity_${timestamp}}"
RUN_DIR="${RUN_DIR:-${RUN_ROOT}/${RUN_NAME}}"

export CUDA_VISIBLE_DEVICES="$CUDA_DEVICE"
export HF_HOME="${HF_HOME:-$HF_CACHE_ROOT}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-${HF_CACHE_ROOT}/datasets}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-${HF_CACHE_ROOT}/hub}"

if [[ ! -d "$MODEL" ]]; then
  echo "Model cache not found: $MODEL" >&2
  echo "Download with: huggingface-cli download $MODEL_REPO --local-dir $MODEL" >&2
  exit 1
fi

mkdir -p "$RUN_DIR"

CLUSTER_ARGS=(--moe_wanda_cluster_k "$CLUSTER_K")
if [[ "$CLUSTER_EXPERTS" == "true" ]]; then
  CLUSTER_ARGS+=(--moe_wanda_cluster_experts)
fi

DIAGNOSTIC_ARGS=()
if [[ "$DIAGNOSTICS" == "false" ]]; then
  DIAGNOSTIC_ARGS=(--no_diagnostics)
fi

for sparsity_ratio in $SPARSITY_RATIOS; do
  sparsity_label="${sparsity_ratio/./p}"
  case_dir="${RUN_DIR}/sparsity_${sparsity_label}"
  output_dir="${case_dir}/output"
  checkpoint_dir="${case_dir}/ckpt"
  result_file="${output_dir}/log_moe_wanda.txt"

  if [[ "$SKIP_EXISTING" == "true" && -s "$result_file" ]]; then
    echo "===== sparsity_${sparsity_label} ====="
    echo "Skip existing result: $result_file"
    continue
  fi

  mkdir -p "$output_dir"

  cmd=(
    python main_dsv2.py
    --model "$MODEL"
    --prune_method moe_wanda
    --sparsity_ratio "$sparsity_ratio"
    --sparsity_type "$SPARSITY_TYPE"
    --seed "$SEED"
    --save "$output_dir"
    --nsamples "$NSAMPLES"
    --moe_wanda_routing_mode "$ROUTING_MODE"
    --moe_wanda_routing_power "$ROUTING_POWER"
    "${CLUSTER_ARGS[@]}"
    "${DIAGNOSTIC_ARGS[@]}"
  )

  if [[ "$SAVE_MODEL" == "true" ]]; then
    mkdir -p "$checkpoint_dir"
    cmd+=(--save_model "$checkpoint_dir")
  fi

  echo "===== sparsity_${sparsity_label} ====="
  printf '%q ' "${cmd[@]}"
  echo
  "${cmd[@]}"
done

echo "Run directory: $RUN_DIR"
