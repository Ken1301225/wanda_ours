import argparse
import json
import os
from importlib.metadata import PackageNotFoundError, version


def _safe_version(name):
    try:
        return version(name)
    except PackageNotFoundError:
        return "unavailable"


def _safe_cuda_device_count():
    try:
        import torch
    except ImportError:
        return 0

    return torch.cuda.device_count()


def append_run_diagnostic(args, payload):
    diagnostics_path = getattr(args, "diagnostics_path", None)
    if not diagnostics_path:
        return

    with open(diagnostics_path, "a") as f:
        f.write(json.dumps(payload, sort_keys=True) + "\n")


def append_run_summary(args, line):
    diagnostics_summary_path = getattr(args, "diagnostics_summary_path", None)
    if not diagnostics_summary_path:
        return

    with open(diagnostics_summary_path, "a") as f:
        print(line, file=f, flush=True)


def prepare_run_outputs(args):
    if not args.save or not getattr(args, "diagnostics", True):
        return

    os.makedirs(args.save, exist_ok=True)
    args.diagnostics_path = os.path.join(
        args.save, f"diagnostics_{args.prune_method}.jsonl"
    )
    args.diagnostics_summary_path = os.path.join(
        args.save, f"diagnostics_{args.prune_method}.txt"
    )

    open(args.diagnostics_path, "w").close()
    open(args.diagnostics_summary_path, "w").close()

    append_run_diagnostic(
        args,
        {
            "event": "run_start",
            "model": args.model,
            "prune_method": args.prune_method,
            "seed": args.seed,
            "nsamples": args.nsamples,
            "sparsity_ratio": args.sparsity_ratio,
            "sparsity_type": args.sparsity_type,
            "use_variant": bool(args.use_variant),
            "moe_wanda_routing_mode": args.moe_wanda_routing_mode,
            "moe_wanda_routing_power": args.moe_wanda_routing_power,
            "moe_wanda_cluster_experts": bool(args.moe_wanda_cluster_experts),
            "moe_wanda_cluster_k": args.moe_wanda_cluster_k,
            "diagnostics": bool(args.diagnostics),
            "save": args.save,
            "save_model": args.save_model,
            "torch_version": _safe_version("torch"),
            "transformers_version": _safe_version("transformers"),
            "accelerate_version": _safe_version("accelerate"),
            "cuda_device_count": _safe_cuda_device_count(),
        },
    )
    append_run_summary(
        args,
        f"run_start method={args.prune_method} model={args.model} sparsity={args.sparsity_ratio} nsamples={args.nsamples} routing_mode={args.moe_wanda_routing_mode} routing_power={args.moe_wanda_routing_power} cluster_experts={bool(args.moe_wanda_cluster_experts)} cluster_k={args.moe_wanda_cluster_k}",
    )


def get_llm(model_name, cache_dir="llm_weights"):
    import torch
    from transformers import AutoModelForCausalLM

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        cache_dir=cache_dir,
        low_cpu_mem_usage=True,
        device_map="auto",
        trust_remote_code=True,
    )
    model.seqlen = 2048
    # model.seqlen = model.config.max_position_embeddings
    return model


def build_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, help="MoE model")
    parser.add_argument(
        "--seed", type=int, default=0, help="Seed for sampling the calibration data."
    )
    parser.add_argument(
        "--nsamples", type=int, default=128, help="Number of calibration samples."
    )
    parser.add_argument(
        "--sparsity_ratio", type=float, default=0, help="Sparsity level"
    )
    parser.add_argument(
        "--sparsity_type", type=str, choices=["unstructured", "4:8", "2:4"]
    )
    parser.add_argument("--prune_method", type=str, choices=["moe_wanda"])
    parser.add_argument("--cache_dir", default="llm_weights", type=str)
    parser.add_argument(
        "--use_variant",
        action="store_true",
        help="whether to use the wanda variant described in the appendix",
    )
    parser.add_argument(
        "--moe_wanda_routing_mode",
        type=str,
        choices=["topk", "dense_softmax"],
        default="dense_softmax",
        help="Routing statistics mode for MoE-Wanda calibration.",
    )
    parser.add_argument(
        "--moe_wanda_routing_power",
        type=float,
        default=2.0,
        help="Exponent p used in routing-weight scaling g_e(x)^p.",
    )
    parser.add_argument(
        "--moe_wanda_cluster_experts",
        action="store_true",
        help="Cluster experts by router-logit traces and prune within each cluster.",
    )
    parser.add_argument(
        "--moe_wanda_cluster_k",
        type=int,
        default=15,
        help="Fixed number of expert clusters per MoE layer when clustering is enabled.",
    )
    parser.add_argument("--save", type=str, default=None, help="Path to save results.")
    parser.add_argument(
        "--save_model", type=str, default=None, help="Path to save the pruned model."
    )
    parser.add_argument(
        "--no_diagnostics",
        action="store_false",
        dest="diagnostics",
        help="Disable detailed pruning diagnostics and only write the final result log.",
    )
    parser.set_defaults(diagnostics=True)

    parser.add_argument("--eval_zero_shot", action="store_true")
    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()

    import numpy as np
    import torch

    from lib.eval import eval_ppl
    from lib.prune import check_sparsity, prune_moe_wanda
    from lib.tokenizer import load_tokenizer

    print("torch", version("torch"))
    print("transformers", version("transformers"))
    print("accelerate", version("accelerate"))
    print("# of gpus: ", torch.cuda.device_count())

    if args.model is not None:
        args.model = args.model.rstrip("/")
    if args.save_model is not None:
        args.save_model = args.save_model.rstrip("/")

    prepare_run_outputs(args)

    # Setting seeds for reproducibility
    np.random.seed(args.seed)
    torch.random.manual_seed(args.seed)

    # Handling n:m sparsity
    prune_n, prune_m = 0, 0
    if args.sparsity_type != "unstructured":
        assert (
            args.sparsity_ratio == 0.5
        ), "sparsity ratio must be 0.5 for structured N:M sparsity"
        prune_n, prune_m = map(int, args.sparsity_type.split(":"))

    model_name = args.model.split("/")[-1]
    print(f"loading llm model {args.model}")
    model = get_llm(args.model, args.cache_dir)
    model.eval()
    tokenizer = load_tokenizer(args.model, use_fast=False, trust_remote_code=True)

    device = torch.device("cuda:0")
    print("use device ", device)

    if args.sparsity_ratio != 0:
        print("pruning starts")
        prune_moe_wanda(
            args, model, tokenizer, device, prune_n=prune_n, prune_m=prune_m
        )

    ################################################################
    print("*" * 30)
    sparsity_ratio = check_sparsity(model)
    print(f"sparsity sanity check {sparsity_ratio:.4f}")
    print("*" * 30)
    ################################################################
    ppl_test = eval_ppl(args, model, tokenizer, device)
    print(f"wikitext perplexity {ppl_test}")

    append_run_diagnostic(
        args,
        {
            "event": "run_complete",
            "model": args.model,
            "prune_method": args.prune_method,
            "actual_sparsity": float(sparsity_ratio),
            "ppl_test": float(ppl_test),
        },
    )
    append_run_summary(
        args,
        f"run_complete actual_sparsity={sparsity_ratio:.6f} ppl_test={ppl_test:.6f}",
    )

    if args.save:
        if not os.path.exists(args.save):
            os.makedirs(args.save)
        save_filepath = os.path.join(args.save, f"log_{args.prune_method}.txt")
        with open(save_filepath, "w") as f:
            print("method\tactual_sparsity\tppl_test", file=f, flush=True)
            print(
                f"{args.prune_method}\t{sparsity_ratio:.4f}\t{ppl_test:.4f}",
                file=f,
                flush=True,
            )

    if args.save_model:
        model.save_pretrained(args.save_model)
        tokenizer.save_pretrained(args.save_model)


if __name__ == "__main__":
    main()
