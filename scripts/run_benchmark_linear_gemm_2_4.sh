#!/usr/bin/env bash
set -euo pipefail

DENSE_MODEL="${DENSE_MODEL:-/data1/ldk/model/Qwen1.5/models--Qwen--Qwen1.5-MoE-A2.7B/snapshots/1a758c50ecb6350748b9ce0a99d2352fd9fc11c9/}"
PRUNED_MODEL="${PRUNED_MODEL:-/data1/ldk/nlp/wanda_moe_cluster/ckpt_20260529_155156/}"
CACHE_DIR="${CACHE_DIR:-/data1/ldk/huggingface}"
BATCH_SIZE="${BATCH_SIZE:-32}"
PROMPT_LENGTH="${PROMPT_LENGTH:-512}"
TOKENS="${TOKENS:-$((BATCH_SIZE * PROMPT_LENGTH))}"
DTYPE="${DTYPE:-bfloat16}"
DEVICE="${DEVICE:-cuda}"
BACKEND="${BACKEND:-cusparselt}"
CUSPARSELT_ALG_ID="${CUSPARSELT_ALG_ID:-0}"
WARMUP="${WARMUP:-3}"
ITERS="${ITERS:-20}"
MAX_MODULES_PER_GROUP="${MAX_MODULES_PER_GROUP:-20}"
OUT_DIR="${OUT_DIR:-benchmark_projection_table}"

mkdir -p "$OUT_DIR"

if [[ -z "$DENSE_MODEL" || -z "$PRUNED_MODEL" ]]; then
  echo "Usage: DENSE_MODEL=/path/to/dense PRUNED_MODEL=/path/to/pruned_2_4 bash scripts/run_benchmark_linear_gemm_2_4.sh" >&2
  exit 1
fi

python scripts/benchmark_linear_gemm_2_4.py \
  --dense-model "$DENSE_MODEL" \
  --pruned-model "$PRUNED_MODEL" \
  --cache-dir "$CACHE_DIR" \
  --device "$DEVICE" \
  --tokens "$TOKENS" \
  --dtype "$DTYPE" \
  --backend "$BACKEND" \
  --cusparselt-alg-id "$CUSPARSELT_ALG_ID" \
  --warmup "$WARMUP" \
  --iters "$ITERS" \
  --max-modules-per-group "$MAX_MODULES_PER_GROUP" \
  --output-json "$OUT_DIR/linear_gemm_speedup.json" \
  --output-markdown "$OUT_DIR/linear_gemm_speedup.md"
