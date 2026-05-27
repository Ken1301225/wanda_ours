import argparse
import os 
import sys
from importlib.metadata import version

def disable_deepspeed_probe_for_save():
    try:
        import accelerate.utils.other as accel_other
        accel_other.is_deepspeed_available = lambda: False
    except Exception:
        pass
def get_llm(model_name, cache_dir="llm_weights"):
    import torch
    from transformers import AutoModelForCausalLM

    model = AutoModelForCausalLM.from_pretrained(
        model_name, 
        torch_dtype=torch.bfloat16,
        # cache_dir=cache_dir, 
        device_map="auto",
        trust_remote_code=True,
    )
    model.seqlen = 1024 
    # model.seqlen = model.config.max_position_embeddings 
    return model


def build_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, help='MoE model')
    parser.add_argument('--seed', type=int, default=0, help='Seed for sampling the calibration data.')
    parser.add_argument('--nsamples', type=int, default=128, help='Number of calibration samples.')
    parser.add_argument('--sparsity_ratio', type=float, default=0, help='Sparsity level')
    parser.add_argument("--sparsity_type", type=str, choices=["unstructured", "4:8", "2:4"])
    parser.add_argument("--prune_method", type=str, choices=["moe_wanda"])
    parser.add_argument("--cache_dir", default="llm_weights", type=str )
    parser.add_argument('--use_variant', action="store_true", help="whether to use the wanda variant described in the appendix")
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
    parser.add_argument('--save', type=str, default=None, help='Path to save results.')
    parser.add_argument('--save_model', type=str, default=None, help='Path to save the pruned model.')

    parser.add_argument("--eval_zero_shot", action="store_true")
    return parser


def main():
    import numpy as np
    import torch

    from lib.prune_dsv2 import check_sparsity, prune_moe_wanda
    from lib.tokenizer import load_tokenizer

    print('torch', version('torch'))
    print('transformers', version('transformers'))
    print('accelerate', version('accelerate'))
    print('# of gpus: ', torch.cuda.device_count())

    parser = build_parser()
    args = parser.parse_args()

    # Normalize path-like arguments to avoid dynamic-module cache collisions.
    if args.model is not None:
        args.model = args.model.rstrip("/")
    if args.save_model is not None:
        args.save_model = args.save_model.rstrip("/")

    # Setting seeds for reproducibility
    np.random.seed(args.seed)
    torch.random.manual_seed(args.seed)

    # Handling n:m sparsity
    prune_n, prune_m = 0, 0
    if args.sparsity_type != "unstructured":
        assert args.sparsity_ratio == 0.5, "sparsity ratio must be 0.5 for structured N:M sparsity"
        prune_n, prune_m = map(int, args.sparsity_type.split(":"))

    model_name = args.model.split("/")[-1]
    print(f"loading llm model {args.model}")
    model = get_llm(args.model, args.cache_dir)
    print(model)
    model.eval()
    tokenizer = load_tokenizer(args.model, use_fast=False, trust_remote_code=True)

    device = torch.device("cuda:0")
    print("use device ", device)

    if args.sparsity_ratio != 0:
        print("pruning starts")
        prune_moe_wanda(args, model, tokenizer, device, prune_n=prune_n, prune_m=prune_m)

    ################################################################
    print("*"*30)
    sparsity_ratio = check_sparsity(model)
    print(f"sparsity sanity check {sparsity_ratio:.4f}")
    print("*"*30)
    ################################################################
    # ppl_test = eval_ppl(args, model, tokenizer, device)
    # print(f"wikitext perplexity {ppl_test}")

    if args.save:
        if not os.path.exists(args.save):
            os.makedirs(args.save)
        save_filepath = os.path.join(args.save, f"log_{args.prune_method}.txt")
        with open(save_filepath, "w") as f:
            print("method\tactual_sparsity\tppl_test", file=f, flush=True)
            print(f"{args.prune_method}\t{sparsity_ratio:.4f}", file=f, flush=True) #\t{ppl_test:.4f}"


    if args.save_model:
        module_name = model.__class__.__module__
        module_file = sys.modules[module_name].__file__ if module_name in sys.modules else "<unknown>"
        print(f"save source module: {module_name}")
        print(f"save source file: {module_file}")
        disable_deepspeed_probe_for_save()
        model.save_pretrained(args.save_model)
        tokenizer.save_pretrained(args.save_model)

if __name__ == '__main__':
    main()
