#!/usr/bin/env python
import argparse
import json
import time
from pathlib import Path


TABLE_ROWS = ("up/gate_proj", "down_proj")


def parse_int_list(value):
    values = []
    for item in value.split(","):
        item = item.strip()
        if not item:
            continue
        parsed = int(item)
        if parsed <= 0:
            raise ValueError(f"values must be positive, got {parsed}")
        values.append(parsed)
    if not values:
        raise ValueError("at least one value is required")
    return values


def classify_projection(name):
    if ".experts." not in name:
        return None
    if name.endswith(("up_proj", "gate_proj")):
        return "up/gate_proj"
    if name.endswith("down_proj"):
        return "down_proj"
    return None


def build_parser():
    parser = argparse.ArgumentParser(
        description="Paper-style 2:4 benchmark: measure GEMM latency for MoE expert Linear layers."
    )
    parser.add_argument("--dense-model", required=True, help="Original dense HF model path.")
    parser.add_argument("--pruned-model", required=True, help="Already-pruned 2:4 HF checkpoint path.")
    parser.add_argument("--cache-dir", default="llm_weights")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", choices=["float16", "bfloat16"], default="float16")
    parser.add_argument("--tokens", default="1", help="Comma-separated GEMM M sizes, e.g. 1,8,64.")
    parser.add_argument("--warmup", type=int, default=25)
    parser.add_argument("--iters", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-modules-per-group", type=int, default=16)
    parser.add_argument("--backend", choices=["cutlass", "cusparselt"], default="cusparselt")
    parser.add_argument("--cusparselt-alg-id", type=int, default=0)
    parser.add_argument("--skip-2-4-check", action="store_true")
    parser.add_argument("--output-json", default=None)
    parser.add_argument("--output-markdown", default=None)
    return parser


def _import_runtime():
    try:
        import torch
        import torch.nn as nn
        import torch.nn.functional as F
        from transformers import AutoModelForCausalLM
    except ImportError as exc:
        raise RuntimeError("PyTorch and transformers are required for this benchmark.") from exc
    return torch, nn, F, AutoModelForCausalLM


def _dtype_from_name(torch, dtype_name):
    if dtype_name == "float16":
        return torch.float16
    if dtype_name == "bfloat16":
        return torch.bfloat16
    raise ValueError(f"unsupported dtype: {dtype_name}")


def _sync(torch, device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _load_model(model_path, args):
    torch, _, _, AutoModelForCausalLM = _import_runtime()
    dtype = _dtype_from_name(torch, args.dtype)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=dtype,
        cache_dir=args.cache_dir,
        low_cpu_mem_usage=True,
        device_map="auto" if args.device == "cuda" else None,
        trust_remote_code=True,
    )
    model.eval()
    return model


def _module_device(module):
    return module.weight.device


def _collect_expert_linears(model, max_modules_per_group):
    _, nn, _, _ = _import_runtime()
    groups = {row: [] for row in TABLE_ROWS}
    for name, module in model.named_modules():
        if not isinstance(module, nn.Linear):
            continue
        group = classify_projection(name)
        if group is None:
            continue
        if len(groups[group]) < max_modules_per_group:
            groups[group].append((name, module))
    return groups


def _has_2_4_pattern(torch, weight):
    dense_weight = weight.detach()
    if dense_weight.shape[1] % 4 != 0:
        return False
    blocks = dense_weight.reshape(dense_weight.shape[0], dense_weight.shape[1] // 4, 4)
    return bool(torch.all((blocks == 0).sum(dim=-1) == 2).item())


def _to_sparse_weight(weight, backend, cusparselt_alg_id):
    try:
        import torch.sparse.semi_structured as semi_structured
        from torch.sparse import to_sparse_semi_structured
    except ImportError as exc:
        raise RuntimeError("torch.sparse.to_sparse_semi_structured is unavailable.") from exc

    previous_force_cutlass = semi_structured.SparseSemiStructuredTensor._FORCE_CUTLASS
    previous_alg_id = getattr(semi_structured.SparseSemiStructuredTensorCUSPARSELT, "_DEFAULT_ALG_ID", 0)
    semi_structured.SparseSemiStructuredTensor._FORCE_CUTLASS = backend == "cutlass"
    if backend == "cusparselt":
        semi_structured.SparseSemiStructuredTensorCUSPARSELT._DEFAULT_ALG_ID = int(cusparselt_alg_id)
    try:
        return to_sparse_semi_structured(weight.detach())
    finally:
        semi_structured.SparseSemiStructuredTensor._FORCE_CUTLASS = previous_force_cutlass
        semi_structured.SparseSemiStructuredTensorCUSPARSELT._DEFAULT_ALG_ID = previous_alg_id


def _time_linear(torch, F, weight, tokens, warmup, iters):
    device = weight.device
    dtype = weight.dtype
    in_features = weight.shape[1]
    x = torch.randn(tokens, in_features, device=device, dtype=dtype)

    with torch.inference_mode():
        for _ in range(warmup):
            F.linear(x, weight)
    _sync(torch, device)

    if device.type == "cuda":
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        with torch.inference_mode():
            for _ in range(iters):
                F.linear(x, weight)
        end.record()
        _sync(torch, device)
        return start.elapsed_time(end) / iters

    start_time = time.perf_counter()
    with torch.inference_mode():
        for _ in range(iters):
            F.linear(x, weight)
    return ((time.perf_counter() - start_time) * 1000.0) / iters


def _average_group_latency(model, args, tokens, sparse):
    torch, _, F, _ = _import_runtime()
    groups = _collect_expert_linears(model, args.max_modules_per_group)
    latencies = {}
    conversion = {"converted": 0, "skipped": []}
    for group, modules in groups.items():
        group_latencies = []
        for name, module in modules:
            weight = module.weight
            if sparse:
                if not args.skip_2_4_check and not _has_2_4_pattern(torch, weight):
                    conversion["skipped"].append({"name": name, "reason": "not_2_4"})
                    continue
                try:
                    weight = _to_sparse_weight(weight, args.backend, args.cusparselt_alg_id)
                    conversion["converted"] += 1
                except Exception as exc:
                    conversion["skipped"].append({"name": name, "reason": str(exc)})
                    continue
            group_latencies.append(_time_linear(torch, F, weight, tokens, args.warmup, args.iters))
        latencies[group] = sum(group_latencies) / len(group_latencies) if group_latencies else 0.0
    return latencies, conversion


def build_comparison(dense_ms, sparse_ms):
    comparison = {}
    for group in TABLE_ROWS:
        dense_value = dense_ms.get(group, 0.0)
        sparse_value = sparse_ms.get(group, 0.0)
        speedup = 0.0 if sparse_value == 0 else dense_value / sparse_value
        comparison[group] = {
            "dense_ms": dense_value,
            "sparse_ms": sparse_value,
            "speedup": speedup,
        }
    return comparison


def format_markdown_table(comparison):
    lines = [
        "| MoE Layer | Dense | 2:4 | Speedup |",
        "|---|---:|---:|---:|",
    ]
    for group in TABLE_ROWS:
        row = comparison[group]
        lines.append(
            f"| {group} | {row['dense_ms']:.2f} | {row['sparse_ms']:.2f} | {row['speedup']:.2f}x |"
        )
    return "\n".join(lines)


def run_benchmark(args):
    torch, _, _, _ = _import_runtime()
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available.")
    torch.manual_seed(args.seed)

    tokens_list = parse_int_list(args.tokens)
    dense_model = _load_model(args.dense_model, args)
    pruned_model = _load_model(args.pruned_model, args)

    results = {
        "dense_model": args.dense_model,
        "pruned_model": args.pruned_model,
        "backend": args.backend,
        "cusparselt_alg_id": args.cusparselt_alg_id,
        "dtype": args.dtype,
        "max_modules_per_group": args.max_modules_per_group,
        "by_tokens": {},
    }
    for tokens in tokens_list:
        dense_ms, _ = _average_group_latency(dense_model, args, tokens, sparse=False)
        sparse_ms, conversion = _average_group_latency(pruned_model, args, tokens, sparse=True)
        comparison = build_comparison(dense_ms, sparse_ms)
        results["by_tokens"][str(tokens)] = {
            "comparison": comparison,
            "conversion": conversion,
        }
    return results


def _print_results(results):
    for tokens, payload in results["by_tokens"].items():
        print(f"\n# tokens={tokens} backend={results['backend']}")
        print(format_markdown_table(payload["comparison"]))
        conversion = payload["conversion"]
        print(f"converted={conversion['converted']} skipped={len(conversion['skipped'])}")


def main():
    parser = build_parser()
    args = parser.parse_args()
    results = run_benchmark(args)
    _print_results(results)
    if args.output_json:
        Path(args.output_json).write_text(json.dumps(results, indent=2, sort_keys=True) + "\n")
    if args.output_markdown:
        first_tokens = str(parse_int_list(args.tokens)[0])
        table = format_markdown_table(results["by_tokens"][first_tokens]["comparison"])
        Path(args.output_markdown).write_text(table + "\n")


if __name__ == "__main__":
    main()
