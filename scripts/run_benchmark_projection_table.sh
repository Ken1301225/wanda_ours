#!/usr/bin/env bash
set -euo pipefail

DENSE_MODEL="${DENSE_MODEL:-}"
PRUNED_MODEL="${PRUNED_MODEL:-}"
CACHE_DIR="${CACHE_DIR:-llm_weights}"
BATCH_SIZE="${BATCH_SIZE:-1}"
PROMPT_LENGTH="${PROMPT_LENGTH:-512}"
DTYPE="${DTYPE:-bfloat16}"
WARMUP="${WARMUP:-3}"
ITERS="${ITERS:-10}"
DEVICE="${DEVICE:-cuda}"
SPARSE_SCOPE="${SPARSE_SCOPE:-all_linear}"
USE_SPARSE_KERNEL="${USE_SPARSE_KERNEL:-true}"
OUT_DIR="${OUT_DIR:-benchmark_projection_table}"

mkdir -p "$OUT_DIR"

if [[ -z "$DENSE_MODEL" || -z "$PRUNED_MODEL" ]]; then
  echo "Usage: DENSE_MODEL=/path/to/dense PRUNED_MODEL=/path/to/pruned_2_4 bash scripts/run_benchmark_projection_table.sh" >&2
  exit 1
fi

SPARSE_ARGS=()
if [[ "$USE_SPARSE_KERNEL" == "true" ]]; then
  SPARSE_ARGS+=(--semi-structured-sparse --sparse-scope "$SPARSE_SCOPE")
fi

python scripts/benchmark_projection_table.py \
  --dense-model "$DENSE_MODEL" \
  --pruned-model "$PRUNED_MODEL" \
  --cache-dir "$CACHE_DIR" \
  --batch-size "$BATCH_SIZE" \
  --prompt-length "$PROMPT_LENGTH" \
  --dtype "$DTYPE" \
  --device "$DEVICE" \
  --warmup "$WARMUP" \
  --iters "$ITERS" \
  --output-json "$OUT_DIR/projection_speedup.json" \
  --output-markdown "$OUT_DIR/projection_speedup.md" \
  "${SPARSE_ARGS[@]}"
