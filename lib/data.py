# Code adapted from https://github.com/IST-DASLab/sparsegpt/blob/master/datautils.py
import os
import numpy as np
import random
import torch
from datasets import load_dataset
import glob

# Set seed for reproducibility
def set_seed(seed):
    np.random.seed(seed)
    torch.random.manual_seed(seed)

# Wrapper for tokenized input IDs
class TokenizerWrapper:
    def __init__(self, input_ids):
        self.input_ids = input_ids


def _default_hf_hub_cache():
    hf_hub_cache = os.environ.get("HF_HUB_CACHE")
    if hf_hub_cache:
        return hf_hub_cache

    hf_cache_root = os.environ.get("HF_CACHE_ROOT")
    if hf_cache_root:
        return os.path.join(hf_cache_root, "hub")

    return "/data1/ldk/huggingface/hub"


def _find_local_dataset_files(dataset_repo, pattern):
    hub_cache = _default_hf_hub_cache()
    cache_repo = f"datasets--{dataset_repo.replace('/', '--')}"
    search_pattern = os.path.join(hub_cache, cache_repo, "snapshots", "*", pattern)
    return sorted(glob.glob(search_pattern))


def _load_local_wikitext_split(split):
    files = _find_local_dataset_files(
        "Salesforce/wikitext",
        os.path.join("wikitext-2-raw-v1", f"{split}-*.parquet"),
    )
    if not files:
        raise ValueError(f"Offline fallback failed: local WikiText file not found for split={split}.")
    return load_dataset("parquet", data_files={split: files}, split=split)

# Load and process wikitext2 dataset
def get_wikitext2(nsamples, seed, seqlen, tokenizer):
    # Load train and test datasets
    try:
        traindata = _load_local_wikitext_split('train')
        testdata = _load_local_wikitext_split('test')
    except Exception:
        traindata = load_dataset('wikitext', 'wikitext-2-raw-v1', split='train')
        testdata = load_dataset('wikitext', 'wikitext-2-raw-v1', split='test')

    # Encode datasets
    trainenc = tokenizer(" ".join(traindata['text']), return_tensors='pt')
    testenc = tokenizer("\n\n".join(testdata['text']), return_tensors='pt')

    # Generate samples from training set
    random.seed(seed)
    trainloader = []
    for _ in range(nsamples):
        i = random.randint(0, trainenc.input_ids.shape[1] - seqlen - 1)
        j = i + seqlen
        inp = trainenc.input_ids[:, i:j]
        tar = inp.clone()
        tar[:, :-1] = -100
        trainloader.append((inp, tar))
    return trainloader, testenc

def _find_local_c4_file(filename):
    candidates = _find_local_dataset_files("allenai/c4", os.path.join("en", filename))
    return candidates[-1] if candidates else None


def _load_c4_split_from_local(split, filename):
    local_file = _find_local_c4_file(filename)
    if local_file is None:
        raise ValueError(
            f"Offline fallback failed: local C4 file not found for split={split}, filename={filename}."
        )
    return load_dataset("json", data_files={split: local_file}, split=split)

# Load and process c4 dataset
def get_c4(nsamples, seed, seqlen, tokenizer):
    # Load train and validation datasets
    # traindata = load_dataset('allenai/c4', 'allenai--c4', data_files={'train': 'en/c4-train.00000-of-01024.json.gz'}, split='train')
    # valdata = load_dataset('allenai/c4', 'allenai--c4', data_files={'validation': 'en/c4-validation.00000-of-00008.json.gz'}, split='validation')
    # 一次性将 train 和 validation 喂给 data_files，满足库的完整性检查
    try:
        traindata = _load_c4_split_from_local('train', 'c4-train.00000-of-01024.json.gz')
        valdata = _load_c4_split_from_local('validation', 'c4-validation.00000-of-00008.json.gz')
    except Exception:
        dataset = load_dataset(
            "allenai/c4", 
            name="en", 
            data_files={
                "train": "en/c4-train.00000-of-01024.json.gz",
                "validation": "en/c4-validation.00000-of-00008.json.gz"
            },
            verification_mode="no_checks"
        )

        traindata = dataset["train"]
        valdata = dataset["validation"]

    # Generate samples from training set
    random.seed(seed)
    trainloader = []
    for _ in range(nsamples):
        while True:
            i = random.randint(0, len(traindata) - 1)
            trainenc = tokenizer(traindata[i]['text'], return_tensors='pt')
            if trainenc.input_ids.shape[1] > seqlen:
                break
        i = random.randint(0, trainenc.input_ids.shape[1] - seqlen - 1)
        j = i + seqlen
        inp = trainenc.input_ids[:, i:j]
        tar = inp.clone()
        tar[:, :-1] = -100
        trainloader.append((inp, tar))

    # Prepare validation dataset
    valenc = tokenizer(' '.join(valdata[:1100]['text']), return_tensors='pt')
    valenc = valenc.input_ids[:, :(256 * seqlen)]
    valenc = TokenizerWrapper(valenc)
    return trainloader, valenc

# Function to select the appropriate loader based on dataset name
def get_loaders(name, nsamples=128, seed=0, seqlen=2048, tokenizer=None):
    if 'wikitext2' in name:
        return get_wikitext2(nsamples, seed, seqlen, tokenizer)
    if "c4" in name:
        return get_c4(nsamples, seed, seqlen, tokenizer)
