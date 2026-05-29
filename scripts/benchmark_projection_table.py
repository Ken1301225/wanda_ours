#!/usr/bin/env python
import argparse
import json
import time
from collections import defaultdict
from pathlib import Path

from scripts.benchmark_model_inference import (
    _convert_to_semi_structured_sparse,
    _import_runtime,
    _load_model,
    _make_inputs,
    _model_device,
    _sync,
)


TABLE_ROWS = ("q/k/v/o_proj", "up/gate_proj", "down_proj")


def build_parser():
    parser = argparse.ArgumentParser(
        description="Benchmark dense vs pruned 2:4 model projection latency and print a paper-style table."
    )
    parser.add_argument("--dense-model", required=True, help="Original dense HF model path.")
    parser.add_argument("--pruned-model", required=True, help="Already-pruned 2:4 HF checkpoint path.")
    parser.add_argument("--cache-dir", default="llm_weights")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", choices=["float16", "bfloat16"], default="bfloat16")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--prompt-length", type=int, default=512)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--iters", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--semi-structured-sparse",
        action="store_true",
        help="Convert the pruned checkpoint's eligible 2:4 weights to PyTorch semi-structured sparse tensors.",
    )
    parser.add_argument(
        "--sparse-scope",
        choices=["moe_experts", "all_linear"],
        default="all_linear",
        help="Which Linear weights to convert when --semi-structured-sparse is enabled.",
    )
    parser.add_argument("--skip-2-4-check", action="store_true")
    parser.add_argument("--output-json", default=None)
    parser.add_argument("--output-markdown", default=None)
    return parser


class ProjectionTimer:
    def __init__(self, torch, nn, device):
        self.torch = torch
        self.nn = nn
        self.device = device
        self.enabled = False
        self.handles = []
        self.events = defaultdict(list)
        self.cpu_starts = {}
        self.cpu_elapsed_ms = defaultdict(float)

    @staticmethod
    def group_for_name(name):
        if name.endswith(("q_proj", "k_proj", "v_proj", "o_proj")):
            return "q/k/v/o_proj"
        if name.endswith(("up_proj", "gate_proj")):
            return "up/gate_proj"
        if name.endswith("down_proj"):
            return "down_proj"
        return None

    def attach(self, model):
        for name, module in model.named_modules():
            if not isinstance(module, self.nn.Linear):
                continue
            group = self.group_for_name(name)
            if group is None:
                continue
            self.handles.append(module.register_forward_pre_hook(self._make_pre_hook(group)))
            self.handles.append(module.register_forward_hook(self._make_post_hook(group)))

    def remove(self):
        for handle in self.handles:
            handle.remove()
        self.handles.clear()

    def reset(self):
        self.events.clear()
        self.cpu_starts.clear()
        self.cpu_elapsed_ms.clear()

    def _make_pre_hook(self, group):
        def hook(module, inputs):
            if not self.enabled:
                return
            if self.device.type == "cuda":
                start = self.torch.cuda.Event(enable_timing=True)
                start.record()
                self.events[group].append([start, None])
            else:
                self.cpu_starts[id(module)] = (group, time.perf_counter())

        return hook

    def _make_post_hook(self, group):
        def hook(module, inputs, output):
            if not self.enabled:
                return
            if self.device.type == "cuda":
                end = self.torch.cuda.Event(enable_timing=True)
                end.record()
                self.events[group][-1][1] = end
            else:
                started = self.cpu_starts.pop(id(module), None)
                if started is not None:
                    started_group, start_time = started
                    self.cpu_elapsed_ms[started_group] += (time.perf_counter() - start_time) * 1000.0

        return hook

    def totals_ms(self):
        totals = {row: 0.0 for row in TABLE_ROWS}
        if self.device.type == "cuda":
            _sync(self.torch, self.device)
            for group, pairs in self.events.items():
                totals[group] = sum(start.elapsed_time(end) for start, end in pairs if end is not None)
        else:
            for group, value in self.cpu_elapsed_ms.items():
                totals[group] = value
        return totals


def _make_args(model_path, base_args):
    return argparse.Namespace(
        model=model_path,
        cache_dir=base_args.cache_dir,
        device=base_args.device,
        dtype=base_args.dtype,
        trust_remote_code=True,
    )


def _load_benchmark_model(model_path, args, convert_sparse):
    model = _load_model(_make_args(model_path, args))
    sparse_report = None
    if convert_sparse:
        sparse_report = _convert_to_semi_structured_sparse(
            model,
            args.sparse_scope,
            args.skip_2_4_check,
        )
    return model, sparse_report


def _measure_projection_groups(model, args):
    torch, nn, _ = _import_runtime()
    device = _model_device(model, args.device)
    input_ids = _make_inputs(torch, model, args.batch_size, args.prompt_length, device)
    timer = ProjectionTimer(torch, nn, device)
    timer.attach(model)
    try:
        with torch.inference_mode():
            for _ in range(args.warmup):
                model(input_ids=input_ids, use_cache=True)
        _sync(torch, device)

        timer.reset()
        timer.enabled = True
        with torch.inference_mode():
            for _ in range(args.iters):
                model(input_ids=input_ids, use_cache=True)
        timer.enabled = False
        totals = timer.totals_ms()
    finally:
        timer.enabled = False
        timer.remove()
    return {group: value / args.iters for group, value in totals.items()}


def build_comparison(dense_ms, pruned_ms):
    comparison = {}
    for group in TABLE_ROWS:
        dense_value = dense_ms.get(group, 0.0)
        pruned_value = pruned_ms.get(group, 0.0)
        speedup = 0.0 if pruned_value == 0 else dense_value / pruned_value
        comparison[group] = {
            "dense_ms": dense_value,
            "pruned_ms": pruned_value,
            "speedup": speedup,
        }
    return comparison


def format_markdown_table(comparison):
    lines = [
        "| LLaMA Layer | Dense | 2:4 | Speedup |",
        "|---|---:|---:|---:|",
    ]
    for group in TABLE_ROWS:
        row = comparison[group]
        lines.append(
            f"| {group} | {row['dense_ms']:.2f} | {row['pruned_ms']:.2f} | {row['speedup']:.2f}x |"
        )
    return "\n".join(lines)


def run_benchmark(args):
    torch, _, _ = _import_runtime()
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available.")
    torch.manual_seed(args.seed)

    dense_model, _ = _load_benchmark_model(args.dense_model, args, convert_sparse=False)
    dense_ms = _measure_projection_groups(dense_model, args)
    del dense_model
    if args.device == "cuda":
        torch.cuda.empty_cache()

    pruned_model, sparse_report = _load_benchmark_model(
        args.pruned_model,
        args,
        convert_sparse=args.semi_structured_sparse,
    )
    pruned_ms = _measure_projection_groups(pruned_model, args)
    comparison = build_comparison(dense_ms, pruned_ms)
    return {
        "dense_model": args.dense_model,
        "pruned_model": args.pruned_model,
        "semi_structured_sparse": bool(args.semi_structured_sparse),
        "sparse_scope": args.sparse_scope,
        "sparse_report": sparse_report,
        "batch_size": args.batch_size,
        "prompt_length": args.prompt_length,
        "iters": args.iters,
        "comparison": comparison,
    }


def main():
    parser = build_parser()
    args = parser.parse_args()
    results = run_benchmark(args)
    table = format_markdown_table(results["comparison"])
    print(table)
    if results.get("sparse_report") is not None:
        report = results["sparse_report"]
        print(f"\nsparse_conversion converted={report['converted_count']} skipped={len(report['skipped'])}")
    if args.output_json:
        Path(args.output_json).write_text(json.dumps(results, indent=2, sort_keys=True) + "\n")
    if args.output_markdown:
        Path(args.output_markdown).write_text(table + "\n")


if __name__ == "__main__":
    main()
