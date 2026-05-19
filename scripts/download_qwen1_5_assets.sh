#!/bin/bash

set -euo pipefail

BASE_DIR="${BASE_DIR:-/data1/data/kangborui/gujinrui/wanda}"
HF_HOME_DIR="$BASE_DIR/huggingface"
HF_HUB_DIR="$HF_HOME_DIR/hub"
HF_DATASETS_DIR="$HF_HOME_DIR/datasets"
MODEL_REPO="${MODEL_REPO:-Qwen/Qwen1.5-MoE-A2.7B}"
MODEL_DIR="$HF_HUB_DIR/models--Qwen--Qwen1.5-MoE-A2.7B"
CONDA_ENV_NAME="${CONDA_ENV_NAME:-prune_llm}"
MAKE_ARCHIVE="${MAKE_ARCHIVE:-0}"

mkdir -p "$MODEL_DIR" "$HF_DATASETS_DIR" "$BASE_DIR/archives"

export HF_HOME="$HF_HOME_DIR"
export HF_HUB_CACHE="$HF_HUB_DIR"
export HF_DATASETS_CACHE="$HF_DATASETS_DIR"

echo "Downloading model to $MODEL_DIR"
conda run -n "$CONDA_ENV_NAME" huggingface-cli download "$MODEL_REPO" --local-dir "$MODEL_DIR"

echo "Prefetching datasets into $HF_DATASETS_DIR"
conda run -n "$CONDA_ENV_NAME" python -c "from datasets import load_dataset; load_dataset('wikitext', 'wikitext-2-raw-v1', split='train'); load_dataset('wikitext', 'wikitext-2-raw-v1', split='test'); load_dataset('allenai/c4', name='en', data_files={'train': 'en/c4-train.00000-of-01024.json.gz', 'validation': 'en/c4-validation.00000-of-00008.json.gz'}, verification_mode='no_checks')"

if [ "$MAKE_ARCHIVE" = "1" ]; then
    ARCHIVE_PATH="$BASE_DIR/archives/qwen1_5_assets_$(date +%Y%m%d_%H%M%S).tar"
    echo "Packing model and datasets into $ARCHIVE_PATH"
    tar -cf "$ARCHIVE_PATH" -C "$BASE_DIR" huggingface
    echo "Archive ready: $ARCHIVE_PATH"
fi

echo "Done. Model dir: $MODEL_DIR"
echo "Done. HF cache dir: $HF_HOME_DIR"
