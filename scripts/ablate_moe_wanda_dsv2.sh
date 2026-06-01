#!/usr/bin/env bash
set -euo pipefail

MODEL_ROOT="${MODEL_ROOT:-/data1/ldk/huggingface/hub}"
HF_CACHE_ROOT="${HF_CACHE_ROOT:-/data1/ldk/huggingface}"
RUN_ROOT="${RUN_ROOT:-/data1/ldk/SPNN/deepseekv2/moe_wanda_ablation}"

MODEL_REPO="${MODEL_REPO:-deepseek-ai/DeepSeek-V2-Lite}"
MODEL="${MODEL:-${MODEL_ROOT}/models--deepseek-ai--DeepSeek-V2-Lite/snapshots/604d5664dddd88a0433dbae533b7fe9472482de0}"
SPARSITY_RATIO="${SPARSITY_RATIO:-0.75}"
SPARSITY_TYPE="${SPARSITY_TYPE:-unstructured}"
CUDA_DEVICES="${CUDA_DEVICES:-${CUDA_DEVICE:-1}}"
MAX_PARALLEL="${MAX_PARALLEL:-1}"
SEED="${SEED:-0}"
NSAMPLES="${NSAMPLES:-512}"
SAVE_MODEL="${SAVE_MODEL:-false}"
DRY_RUN="${DRY_RUN:-false}"
DIAGNOSTICS="${DIAGNOSTICS:-false}"

ROUTING_POWERS="${ROUTING_POWERS:-0.5 1.0 1.5 2.0}"
CLUSTER_KS="${CLUSTER_KS:-5 10 15 30}"

timestamp=$(date +"%Y%m%d_%H%M%S")
ABLATION_ROOT="${ABLATION_ROOT:-${RUN_ROOT}/ablation_${timestamp}}"

export HF_HOME="${HF_HOME:-$HF_CACHE_ROOT}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-${HF_CACHE_ROOT}/datasets}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-${HF_CACHE_ROOT}/hub}"

read -r -a GPU_QUEUE <<< "$CUDA_DEVICES"
if [[ "${#GPU_QUEUE[@]}" -eq 0 ]]; then
  echo "CUDA_DEVICES must contain at least one GPU id." >&2
  exit 1
fi
if (( MAX_PARALLEL < 1 )); then
  echo "MAX_PARALLEL must be >= 1." >&2
  exit 1
fi

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
  local gpu_id="$6"

  local output_dir="${ABLATION_ROOT}/${case_name}/output"
  local checkpoint_dir="${ABLATION_ROOT}/${case_name}/ckpt"
  if [[ "$DRY_RUN" != "true" ]]; then
    mkdir -p "$output_dir"
  fi

  local cmd=(
    python main_dsv2.py
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

  if [[ "$DIAGNOSTICS" == "false" ]]; then
    cmd+=(--no_diagnostics)
  fi

  if [[ "$SAVE_MODEL" == "true" ]]; then
    if [[ "$DRY_RUN" != "true" ]]; then
      mkdir -p "$checkpoint_dir"
    fi
    cmd+=(--save_model "$checkpoint_dir")
  fi

  echo "===== $case_name ====="
  echo "GPU: $gpu_id"
  printf 'CUDA_VISIBLE_DEVICES=%q ' "$gpu_id"
  printf '%q ' "${cmd[@]}"
  echo
  if [[ "$DRY_RUN" != "true" ]]; then
    CUDA_VISIBLE_DEVICES="$gpu_id" "${cmd[@]}"
  fi
}

running_jobs=0
case_index=0

submit_case() {
  local case_name="$1"
  local routing_mode="$2"
  local routing_power="$3"
  local cluster_experts="$4"
  local cluster_k="$5"
  local gpu_id="${GPU_QUEUE[$((case_index % ${#GPU_QUEUE[@]}))]}"
  case_index=$((case_index + 1))

  if [[ "$DRY_RUN" == "true" ]]; then
    run_case "$case_name" "$routing_mode" "$routing_power" "$cluster_experts" "$cluster_k" "$gpu_id"
    return
  fi

  run_case "$case_name" "$routing_mode" "$routing_power" "$cluster_experts" "$cluster_k" "$gpu_id" &
  running_jobs=$((running_jobs + 1))
  if (( running_jobs >= MAX_PARALLEL )); then
    wait -n
    running_jobs=$((running_jobs - 1))
  fi
}

# 1. Routing statistics ablation: top-k routed tokens vs all-token dense softmax.
submit_case "routing_topk_p2_no_cluster" "topk" "2.0" "false" "15"
submit_case "routing_dense_softmax_p2_no_cluster" "dense_softmax" "2.0" "false" "15"

# 2. Routing power ablation under dense all-token routing.
for power in $ROUTING_POWERS; do
  power_label="${power/./p}"
  submit_case "routing_power_${power_label}_dense_no_cluster" "dense_softmax" "$power" "false" "15"
done

# 3. Expert clustering ablation with fixed score definition.
for cluster_k in $CLUSTER_KS; do
  submit_case "cluster_k${cluster_k}_dense_p1_5" "dense_softmax" "1.5" "true" "$cluster_k"
done

if [[ "$DRY_RUN" != "true" ]]; then
  wait
fi

echo "Ablation root: $ABLATION_ROOT"
