#!/usr/bin/env bash
set -euo pipefail

MODEL_ROOT="${MODEL_ROOT:-/data1/ldk/model}"
HF_CACHE_ROOT="${HF_CACHE_ROOT:-/data1/ldk/huggingface}"
RUN_ROOT="${RUN_ROOT:-/data1/ldk/nlp/wanda_moe_ablation}"

MODEL_REPO="${MODEL_REPO:-Qwen/Qwen1.5-MoE-A2.7B}"
MODEL="${MODEL:-${MODEL_ROOT}/Qwen1.5/models--Qwen--Qwen1.5-MoE-A2.7B/snapshots/1a758c50ecb6350748b9ce0a99d2352fd9fc11c9}"
SPARSITY_RATIO="${SPARSITY_RATIO:-0.5}"
SPARSITY_TYPE="${SPARSITY_TYPE:-unstructured}"
CUDA_DEVICE="${CUDA_DEVICE:-0}"
SEED="${SEED:-0}"
NSAMPLES="${NSAMPLES:-128}"
SAVE_MODEL="${SAVE_MODEL:-false}"
DRY_RUN="${DRY_RUN:-false}"

ROUTING_POWERS="${ROUTING_POWERS:-0.5 1.0 1.5 2.0}"
CLUSTER_KS="${CLUSTER_KS:-5 10 15 30}"

timestamp=$(date +"%Y%m%d_%H%M%S")
ABLATION_ROOT="${ABLATION_ROOT:-${RUN_ROOT}/ablation_${timestamp}}"

export CUDA_VISIBLE_DEVICES="$CUDA_DEVICE"
export HF_HOME="${HF_HOME:-$HF_CACHE_ROOT}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-${HF_CACHE_ROOT}/datasets}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-${HF_CACHE_ROOT}/hub}"

if [[ "$DRY_RUN" != "true" && ! -d "$MODEL" ]]; then
  echo "Model cache not found: $MODEL" >&2
  echo "Download with: huggingface-cli download $MODEL_REPO --local-dir $MODEL" >&2
  exit 1
fi

run_case() {
  local case_name="$1"
  local routing_mode="$2"
  local routing_power="$3"
  local cluster_experts="$4"
  local cluster_k="$5"

  local output_dir="${ABLATION_ROOT}/${case_name}/output"
  local checkpoint_dir="${ABLATION_ROOT}/${case_name}/ckpt"
  if [[ "$DRY_RUN" != "true" ]]; then
    mkdir -p "$output_dir"
  fi

  local cmd=(
    python main.py
    --model "$MODEL"
    --prune_method moe_wanda
    --sparsity_ratio "$SPARSITY_RATIO"
    --sparsity_type "$SPARSITY_TYPE"
    --seed "$SEED"
    --save "$output_dir"
    --nsamples "$NSAMPLES"
    --moe_wanda_routing_mode "$routing_mode"
    --moe_wanda_routing_power "$routing_power"
    --moe_wanda_cluster_k "$cluster_k"
  )

  if [[ "$cluster_experts" == "true" ]]; then
    cmd+=(--moe_wanda_cluster_experts)
  fi

  if [[ "$SAVE_MODEL" == "true" ]]; then
    if [[ "$DRY_RUN" != "true" ]]; then
      mkdir -p "$checkpoint_dir"
    fi
    cmd+=(--save_model "$checkpoint_dir")
  fi

  echo "===== $case_name ====="
  printf '%q ' "${cmd[@]}"
  echo
  if [[ "$DRY_RUN" != "true" ]]; then
    "${cmd[@]}"
  fi
}

# 1. Routing statistics ablation: top-k routed tokens vs all-token dense softmax.
run_case "routing_topk_p2_no_cluster" "topk" "2.0" "false" "15"
run_case "routing_dense_softmax_p2_no_cluster" "dense_softmax" "2.0" "false" "15"

# 2. Routing power ablation under dense all-token routing.
for power in $ROUTING_POWERS; do
  run_case "routing_power_${power}_dense_no_cluster" "dense_softmax" "$power" "false" "15"
done

# 3. Expert clustering ablation with fixed score definition.
for cluster_k in $CLUSTER_KS; do
  run_case "cluster_k${cluster_k}_dense_p1_5" "dense_softmax" "1.5" "true" "$cluster_k"
done

echo "Ablation root: $ABLATION_ROOT"
