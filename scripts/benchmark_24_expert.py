#!/usr/bin/env python
import argparse
import json
import math
import time
from pathlib import Path


def parse_token_counts(value):
    counts = []
    for item in value.split(","):
        item = item.strip()
        if not item:
            continue
        count = int(item)
        if count <= 0:
            raise ValueError(f"token counts must be positive, got {count}")
        counts.append(count)
    if not counts:
        raise ValueError("at least one token count is required")
    return counts


def build_parser():
    parser = argparse.ArgumentParser(
        description="Benchmark dense, 2:4 mask-only, and semi-structured sparse expert MLP kernels."
    )
    parser.add_argument("--hidden-size", type=int, default=2048)
    parser.add_argument("--intermediate-size", type=int, default=1408)
    parser.add_argument("--token-counts", type=str, default="1,4,8,16,32,64,128,256")
    parser.add_argument("--sparsity-type", choices=["2:4", "4:8"], default="2:4")
    parser.add_argument("--dtype", choices=["float16", "bfloat16"], default="float16")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--warmup", type=int, default=25)
    parser.add_argument("--iters", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--modes",
        nargs="+",
        choices=["dense", "mask_dense", "sparse"],
        default=["dense", "mask_dense", "sparse"],
    )
    parser.add_argument(
        "--benchmarks",
        nargs="+",
        choices=["gate_proj", "up_proj", "down_proj", "expert_mlp"],
        default=["gate_proj", "up_proj", "down_proj", "expert_mlp"],
    )
    parser.add_argument("--output-json", type=str, default=None)
    return parser


def _import_torch():
    try:
        import torch
        import torch.nn.functional as F
    except ImportError as exc:
        raise RuntimeError("PyTorch is required to run this benchmark.") from exc
    return torch, F


def _parse_nm(sparsity_type):
    n, m = sparsity_type.split(":")
    return int(n), int(m)


def _dtype_from_name(torch, name):
    if name == "float16":
        return torch.float16
    if name == "bfloat16":
        return torch.bfloat16
    raise ValueError(f"unsupported dtype: {name}")


def _make_nm_weight(weight, prune_n, prune_m):
    torch, _ = _import_torch()
    metric = weight.detach().abs().float()
    mask = torch.zeros_like(metric, dtype=torch.bool)
    usable_cols = (metric.shape[1] // prune_m) * prune_m
    if usable_cols != metric.shape[1]:
        raise ValueError(
            f"input dimension {metric.shape[1]} must be divisible by prune_m={prune_m}"
        )

    for start in range(0, metric.shape[1], prune_m):
        block = metric[:, start : start + prune_m]
        indices = torch.topk(block, prune_n, dim=1, largest=False)[1]
        mask.scatter_(1, start + indices, True)

    sparse_weight = weight.clone()
    sparse_weight[mask] = 0
    return sparse_weight


def _to_sparse_semi_structured(weight):
    try:
        from torch.sparse import to_sparse_semi_structured
    except ImportError as exc:
        raise RuntimeError("torch.sparse.to_sparse_semi_structured is unavailable.") from exc
    return to_sparse_semi_structured(weight)


def _sync(torch, device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _time_cuda(torch, device, fn, warmup, iters):
    for _ in range(warmup):
        fn()
    _sync(torch, device)

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(iters):
            fn()
        end.record()
        _sync(torch, device)
        elapsed_ms = start.elapsed_time(end)
        peak_bytes = torch.cuda.max_memory_allocated(device)
    else:
        start_time = time.perf_counter()
        for _ in range(iters):
            fn()
        elapsed_ms = (time.perf_counter() - start_time) * 1000.0
        peak_bytes = None

    return elapsed_ms / iters, peak_bytes


def _linear_flops(tokens, in_features, out_features):
    return 2 * tokens * in_features * out_features


def _expert_flops(tokens, hidden_size, intermediate_size):
    return (
        _linear_flops(tokens, hidden_size, intermediate_size)
        + _linear_flops(tokens, hidden_size, intermediate_size)
        + _linear_flops(tokens, intermediate_size, hidden_size)
    )


def _make_weights(torch, hidden_size, intermediate_size, dtype, device, seed):
    torch.manual_seed(seed)
    scale = 1.0 / math.sqrt(hidden_size)
    gate = torch.randn(intermediate_size, hidden_size, device=device, dtype=dtype) * scale
    up = torch.randn(intermediate_size, hidden_size, device=device, dtype=dtype) * scale
    down = torch.randn(hidden_size, intermediate_size, device=device, dtype=dtype) * scale
    return {"gate_proj": gate, "up_proj": up, "down_proj": down}


def _prepare_mode_weights(weights, mode, prune_n, prune_m):
    if mode == "dense":
        return weights

    masked = {
        name: _make_nm_weight(weight, prune_n, prune_m)
        for name, weight in weights.items()
    }
    if mode == "mask_dense":
        return masked
    if mode == "sparse":
        return {
            name: _to_sparse_semi_structured(weight)
            for name, weight in masked.items()
        }
    raise ValueError(f"unsupported mode: {mode}")


def _make_benchmark_fn(torch, F, benchmark, x_hidden, x_intermediate, weights):
    if benchmark == "gate_proj":
        return lambda: F.linear(x_hidden, weights["gate_proj"])
    if benchmark == "up_proj":
        return lambda: F.linear(x_hidden, weights["up_proj"])
    if benchmark == "down_proj":
        return lambda: F.linear(x_intermediate, weights["down_proj"])
    if benchmark == "expert_mlp":
        return lambda: F.linear(
            F.silu(F.linear(x_hidden, weights["gate_proj"])) * F.linear(x_hidden, weights["up_proj"]),
            weights["down_proj"],
        )
    raise ValueError(f"unsupported benchmark: {benchmark}")


def _benchmark_flops(benchmark, tokens, hidden_size, intermediate_size):
    if benchmark in {"gate_proj", "up_proj"}:
        return _linear_flops(tokens, hidden_size, intermediate_size)
    if benchmark == "down_proj":
        return _linear_flops(tokens, intermediate_size, hidden_size)
    if benchmark == "expert_mlp":
        return _expert_flops(tokens, hidden_size, intermediate_size)
    raise ValueError(f"unsupported benchmark: {benchmark}")


def run_benchmark(args):
    torch, F = _import_torch()
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available. Use --device cpu only for functional smoke tests.")

    device = torch.device(args.device)
    dtype = _dtype_from_name(torch, args.dtype)
    prune_n, prune_m = _parse_nm(args.sparsity_type)
    token_counts = parse_token_counts(args.token_counts)

    results = []
    for mode in args.modes:
        base_weights = None
        try:
            base_weights = _make_weights(
                torch,
                args.hidden_size,
                args.intermediate_size,
                dtype,
                device,
                args.seed,
            )
            mode_weights = _prepare_mode_weights(base_weights, mode, prune_n, prune_m)
            if mode != "dense":
                del base_weights
                base_weights = None
                _sync(torch, device)
                if device.type == "cuda":
                    torch.cuda.empty_cache()
        except Exception as exc:
            results.append({"mode": mode, "status": "error", "error": str(exc)})
            continue

        for tokens in token_counts:
            x_hidden = torch.randn(tokens, args.hidden_size, device=device, dtype=dtype)
            x_intermediate = torch.randn(tokens, args.intermediate_size, device=device, dtype=dtype)
            for benchmark in args.benchmarks:
                fn = _make_benchmark_fn(torch, F, benchmark, x_hidden, x_intermediate, mode_weights)
                try:
                    latency_ms, peak_bytes = _time_cuda(torch, device, fn, args.warmup, args.iters)
                    flops = _benchmark_flops(benchmark, tokens, args.hidden_size, args.intermediate_size)
                    results.append(
                        {
                            "mode": mode,
                            "benchmark": benchmark,
                            "tokens": tokens,
                            "latency_ms": latency_ms,
                            "tflops": flops / (latency_ms / 1000.0) / 1e12,
                            "peak_memory_mb": None if peak_bytes is None else peak_bytes / (1024**2),
                            "status": "ok",
                        }
                    )
                except Exception as exc:
                    results.append(
                        {
                            "mode": mode,
                            "benchmark": benchmark,
                            "tokens": tokens,
                            "status": "error",
                            "error": str(exc),
                        }
                    )
        del mode_weights
        if base_weights is not None:
            del base_weights
        _sync(torch, device)
        if device.type == "cuda":
            torch.cuda.empty_cache()
    return results


def _print_results(results):
    headers = ["mode", "benchmark", "tokens", "latency_ms", "tflops", "peak_memory_mb", "status"]
    print("\t".join(headers))
    for row in results:
        values = []
        for key in headers:
            value = row.get(key)
            if isinstance(value, float):
                value = f"{value:.4f}"
            values.append("" if value is None else str(value))
        print("\t".join(values))
        if row.get("status") == "error":
            print(f"# error {row['mode']}: {row.get('error', '')}")


def main():
    parser = build_parser()
    args = parser.parse_args()
    results = run_benchmark(args)
    _print_results(results)
    if args.output_json:
        Path(args.output_json).write_text(json.dumps(results, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
