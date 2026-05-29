#!/usr/bin/env bash
set -euo pipefail

BASE_MODEL="${BASE_MODEL:-}"
PRUNED_MODEL="${PRUNED_MODEL:-}"
CACHE_DIR="${CACHE_DIR:-llm_weights}"
BATCH_SIZE="${BATCH_SIZE:-1}"
PROMPT_LENGTHS="${PROMPT_LENGTHS:-128,512,1024}"
DECODE_STEPS="${DECODE_STEPS:-32}"
DTYPE="${DTYPE:-bfloat16}"
WARMUP="${WARMUP:-3}"
ITERS="${ITERS:-10}"
OUT_DIR="${OUT_DIR:-benchmark_model_inference}"

mkdir -p "$OUT_DIR"

if [[ -z "$BASE_MODEL" || -z "$PRUNED_MODEL" ]]; then
  echo "Usage: BASE_MODEL=/path/to/dense PRUNED_MODEL=/path/to/2_4_pruned bash scripts/run_benchmark_model_inference.sh" >&2
  exit 1
fi

python scripts/benchmark_model_inference.py \
  --model "$BASE_MODEL" \
  --cache-dir "$CACHE_DIR" \
  --batch-size "$BATCH_SIZE" \
  --prompt-lengths "$PROMPT_LENGTHS" \
  --decode-steps "$DECODE_STEPS" \
  --dtype "$DTYPE" \
  --warmup "$WARMUP" \
  --iters "$ITERS" \
  --output-json "$OUT_DIR/dense_baseline.json"

python scripts/benchmark_model_inference.py \
  --model "$PRUNED_MODEL" \
  --cache-dir "$CACHE_DIR" \
  --batch-size "$BATCH_SIZE" \
  --prompt-lengths "$PROMPT_LENGTHS" \
  --decode-steps "$DECODE_STEPS" \
  --dtype "$DTYPE" \
  --warmup "$WARMUP" \
  --iters "$ITERS" \
  --output-json "$OUT_DIR/pruned_2_4_mask_dense.json"

python scripts/benchmark_model_inference.py \
  --model "$PRUNED_MODEL" \
  --cache-dir "$CACHE_DIR" \
  --batch-size "$BATCH_SIZE" \
  --prompt-lengths "$PROMPT_LENGTHS" \
  --decode-steps "$DECODE_STEPS" \
  --dtype "$DTYPE" \
  --warmup "$WARMUP" \
  --iters "$ITERS" \
  --semi-structured-sparse \
  --sparse-scope moe_experts \
  --output-json "$OUT_DIR/pruned_2_4_sparse_kernel.json"
