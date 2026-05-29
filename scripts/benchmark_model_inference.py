#!/usr/bin/env python
import argparse
import json
import time
from pathlib import Path


MOE_EXPERT_SUFFIXES = ("gate_proj", "up_proj", "down_proj")


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


def build_parser():
    parser = argparse.ArgumentParser(description="Benchmark full-model causal LM inference throughput.")
    parser.add_argument("--model", required=True, help="HF model path or pruned checkpoint path.")
    parser.add_argument("--cache-dir", default="llm_weights")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", choices=["float16", "bfloat16"], default="bfloat16")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--prompt-lengths", default="128,512,1024")
    parser.add_argument("--decode-steps", type=int, default=32)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--iters", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--trust-remote-code", action="store_true", default=True)
    parser.add_argument(
        "--semi-structured-sparse",
        action="store_true",
        help="Convert eligible 2:4 Linear weights to PyTorch semi-structured sparse tensors before benchmarking.",
    )
    parser.add_argument(
        "--sparse-scope",
        choices=["moe_experts", "all_linear"],
        default="moe_experts",
        help="Which Linear weights to convert when --semi-structured-sparse is enabled.",
    )
    parser.add_argument(
        "--skip-2-4-check",
        action="store_true",
        help="Skip explicit 2:4 legality checks before sparse conversion.",
    )
    parser.add_argument("--output-json", default=None)
    return parser


def _import_runtime():
    try:
        import torch
        import torch.nn as nn
        from transformers import AutoModelForCausalLM
    except ImportError as exc:
        raise RuntimeError("PyTorch and transformers are required for full-model benchmarking.") from exc
    return torch, nn, AutoModelForCausalLM


def _dtype_from_name(torch, dtype_name):
    if dtype_name == "float16":
        return torch.float16
    if dtype_name == "bfloat16":
        return torch.bfloat16
    raise ValueError(f"unsupported dtype: {dtype_name}")


def _sync(torch, device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _is_moe_expert_linear(name):
    return ".experts." in name and name.endswith(MOE_EXPERT_SUFFIXES)


def _has_2_4_pattern(torch, weight):
    dense_weight = weight.detach()
    if dense_weight.shape[1] % 4 != 0:
        return False
    blocks = dense_weight.reshape(dense_weight.shape[0], dense_weight.shape[1] // 4, 4)
    zeros_per_block = (blocks == 0).sum(dim=-1)
    return bool(torch.all(zeros_per_block == 2).item())


def _convert_to_semi_structured_sparse(model, sparse_scope, skip_2_4_check):
    torch, nn, _ = _import_runtime()
    try:
        from torch.sparse import to_sparse_semi_structured
    except ImportError as exc:
        raise RuntimeError("torch.sparse.to_sparse_semi_structured is unavailable.") from exc

    converted = []
    skipped = []
    for name, module in model.named_modules():
        if not isinstance(module, nn.Linear):
            continue
        if sparse_scope == "moe_experts" and not _is_moe_expert_linear(name):
            continue
        if not skip_2_4_check and not _has_2_4_pattern(torch, module.weight):
            skipped.append({"name": name, "reason": "not_2_4"})
            continue
        try:
            sparse_weight = to_sparse_semi_structured(module.weight.detach())
            module.weight = nn.Parameter(sparse_weight, requires_grad=False)
            converted.append(name)
        except Exception as exc:
            skipped.append({"name": name, "reason": str(exc)})
    return {"converted_count": len(converted), "converted": converted, "skipped": skipped}


def _load_model(args):
    torch, _, AutoModelForCausalLM = _import_runtime()
    dtype = _dtype_from_name(torch, args.dtype)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=dtype,
        cache_dir=args.cache_dir,
        low_cpu_mem_usage=True,
        device_map="auto" if args.device == "cuda" else None,
        trust_remote_code=args.trust_remote_code,
    )
    model.eval()
    return model


def _model_device(model, requested_device):
    torch, _, _ = _import_runtime()
    if requested_device != "cuda":
        return torch.device(requested_device)
    return next(model.parameters()).device


def _make_inputs(torch, model, batch_size, prompt_length, device):
    vocab_size = int(getattr(model.config, "vocab_size", 32000))
    return torch.randint(0, vocab_size, (batch_size, prompt_length), device=device)


def _time_call(torch, device, fn, warmup, iters):
    with torch.inference_mode():
        for _ in range(warmup):
            fn()
    _sync(torch, device)

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        with torch.inference_mode():
            for _ in range(iters):
                fn()
        end.record()
        _sync(torch, device)
        elapsed_ms = start.elapsed_time(end)
        peak_memory = torch.cuda.max_memory_allocated(device)
    else:
        start_time = time.perf_counter()
        with torch.inference_mode():
            for _ in range(iters):
                fn()
        elapsed_ms = (time.perf_counter() - start_time) * 1000.0
        peak_memory = None
    return elapsed_ms / iters, peak_memory


def _prefill_fn(model, input_ids):
    return lambda: model(input_ids=input_ids, use_cache=True)


def _decode_once(model, input_ids):
    import torch

    with torch.inference_mode():
        outputs = model(input_ids=input_ids, use_cache=True)
        past_key_values = outputs.past_key_values
        next_token = torch.argmax(outputs.logits[:, -1:, :], dim=-1)
    return lambda: model(input_ids=next_token, past_key_values=past_key_values, use_cache=True)


def _decode_loop_fn(model, input_ids, decode_steps):
    import torch

    def run():
        outputs = model(input_ids=input_ids, use_cache=True)
        past_key_values = outputs.past_key_values
        next_token = torch.argmax(outputs.logits[:, -1:, :], dim=-1)
        for _ in range(decode_steps):
            outputs = model(input_ids=next_token, past_key_values=past_key_values, use_cache=True)
            past_key_values = outputs.past_key_values
            next_token = torch.argmax(outputs.logits[:, -1:, :], dim=-1)
        return outputs

    return run


def run_benchmark(args):
    torch, _, _ = _import_runtime()
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available.")

    torch.manual_seed(args.seed)
    model = _load_model(args)
    device = _model_device(model, args.device)

    sparse_report = None
    if args.semi_structured_sparse:
        sparse_report = _convert_to_semi_structured_sparse(
            model,
            args.sparse_scope,
            args.skip_2_4_check,
        )

    prompt_lengths = parse_int_list(args.prompt_lengths)
    results = {
        "model": args.model,
        "batch_size": args.batch_size,
        "dtype": args.dtype,
        "semi_structured_sparse": bool(args.semi_structured_sparse),
        "sparse_scope": args.sparse_scope,
        "sparse_report": sparse_report,
        "measurements": [],
    }

    for prompt_length in prompt_lengths:
        input_ids = _make_inputs(torch, model, args.batch_size, prompt_length, device)

        prefill_ms, prefill_peak = _time_call(
            torch,
            device,
            _prefill_fn(model, input_ids),
            args.warmup,
            args.iters,
        )
        results["measurements"].append(
            {
                "phase": "prefill",
                "prompt_length": prompt_length,
                "decode_steps": 0,
                "latency_ms": prefill_ms,
                "tokens_per_second": (args.batch_size * prompt_length) / (prefill_ms / 1000.0),
                "peak_memory_mb": None if prefill_peak is None else prefill_peak / (1024**2),
            }
        )

        decode_ms, decode_peak = _time_call(
            torch,
            device,
            _decode_once(model, input_ids),
            args.warmup,
            args.iters,
        )
        results["measurements"].append(
            {
                "phase": "decode_one_token_with_cache",
                "prompt_length": prompt_length,
                "decode_steps": 1,
                "latency_ms": decode_ms,
                "tokens_per_second": args.batch_size / (decode_ms / 1000.0),
                "peak_memory_mb": None if decode_peak is None else decode_peak / (1024**2),
            }
        )

        loop_ms, loop_peak = _time_call(
            torch,
            device,
            _decode_loop_fn(model, input_ids, args.decode_steps),
            max(1, args.warmup // 2),
            max(1, args.iters // 2),
        )
        generated_tokens = args.batch_size * args.decode_steps
        results["measurements"].append(
            {
                "phase": "decode_loop",
                "prompt_length": prompt_length,
                "decode_steps": args.decode_steps,
                "latency_ms": loop_ms,
                "tokens_per_second": generated_tokens / (loop_ms / 1000.0),
                "peak_memory_mb": None if loop_peak is None else loop_peak / (1024**2),
            }
        )
    return results


def _print_results(results):
    sparse_report = results.get("sparse_report")
    if sparse_report is not None:
        print(
            f"sparse_conversion converted={sparse_report['converted_count']} "
            f"skipped={len(sparse_report['skipped'])}"
        )
    print("phase\tprompt_length\tdecode_steps\tlatency_ms\ttokens_per_second\tpeak_memory_mb")
    for row in results["measurements"]:
        peak = row["peak_memory_mb"]
        print(
            f"{row['phase']}\t{row['prompt_length']}\t{row['decode_steps']}\t"
            f"{row['latency_ms']:.4f}\t{row['tokens_per_second']:.4f}\t"
            f"{'' if peak is None else f'{peak:.2f}'}"
        )


def main():
    parser = build_parser()
    args = parser.parse_args()
    results = run_benchmark(args)
    _print_results(results)
    if args.output_json:
        Path(args.output_json).write_text(json.dumps(results, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
