import time 
import heapq 
import json
import torch 
import torch.nn as nn 
from .sparsegpt import SparseGPT 
from .moe_wanda import apply_column_zero_ratio_guard, attach_moe_wanda_hooks, build_moe_wanda_mask, build_moe_wanda_metric, filter_moe_expert_linears
from .data import get_loaders 

from .ablate import AblateGPT 


def _append_diagnostic(args, payload):
    diagnostics_path = getattr(args, "diagnostics_path", None)
    if not diagnostics_path:
        return

    with open(diagnostics_path, "a") as f:
        f.write(json.dumps(payload, sort_keys=True) + "\n")


def _append_diagnostic_summary(args, line):
    diagnostics_summary_path = getattr(args, "diagnostics_summary_path", None)
    if not diagnostics_summary_path:
        return

    with open(diagnostics_summary_path, "a") as f:
        print(line, file=f, flush=True)


def _parse_module_location(name):
    parts = name.split(".")
    info = {
        "module_name": name,
        "projection": parts[-1] if parts else name,
        "layer": None,
        "expert": None,
    }

    if "layers" in parts:
        idx = parts.index("layers")
        if idx + 1 < len(parts):
            try:
                info["layer"] = int(parts[idx + 1])
            except ValueError:
                pass

    if "experts" in parts:
        idx = parts.index("experts")
        if idx + 1 < len(parts):
            try:
                info["expert"] = int(parts[idx + 1])
            except ValueError:
                pass

    return info


def _safe_stat(tensor, op):
    if tensor is None or tensor.numel() == 0:
        return None
    return float(op(tensor).item())


def _safe_std(tensor):
    if tensor is None or tensor.numel() == 0:
        return None
    return float(torch.std(tensor, unbiased=False).item())


def _collect_mask_diagnostics(name, weight, W_mask, W_metric):
    location = _parse_module_location(name)
    mask_ratio = float(W_mask.float().mean().item())
    pruned_metric = W_metric[W_mask]
    kept_metric = W_metric[~W_mask]
    pre_zero_rows = float((weight == 0).all(dim=1).float().mean().item())
    pre_zero_cols = float((weight == 0).all(dim=0).float().mean().item())
    pre_sparsity = float((weight == 0).float().mean().item())
    weight_norm_before = float(weight.float().norm().item())

    return {
        **location,
        "mask_ratio": mask_ratio,
        "pre_sparsity": pre_sparsity,
        "pre_zero_row_ratio": pre_zero_rows,
        "pre_zero_col_ratio": pre_zero_cols,
        "weight_norm_before": weight_norm_before,
        "metric_mean": _safe_stat(W_metric, torch.mean),
        "metric_std": _safe_std(W_metric),
        "metric_min": _safe_stat(W_metric, torch.min),
        "metric_max": _safe_stat(W_metric, torch.max),
        "pruned_metric_mean": _safe_stat(pruned_metric, torch.mean),
        "pruned_metric_max": _safe_stat(pruned_metric, torch.max),
        "kept_metric_mean": _safe_stat(kept_metric, torch.mean),
        "kept_metric_min": _safe_stat(kept_metric, torch.min),
    }


def _finalize_weight_diagnostics(module_diag, weight):
    module_diag.update(
        {
            "post_sparsity": float((weight == 0).float().mean().item()),
            "post_zero_row_ratio": float((weight == 0).all(dim=1).float().mean().item()),
            "post_zero_col_ratio": float((weight == 0).all(dim=0).float().mean().item()),
            "weight_norm_after": float(weight.float().norm().item()),
        }
    )
    return module_diag


def _summarize_modules(module_diagnostics):
    projection_summary = {}
    for proj in sorted({entry["projection"] for entry in module_diagnostics}):
        entries = [entry for entry in module_diagnostics if entry["projection"] == proj]
        projection_summary[proj] = {
            "module_count": len(entries),
            "mean_post_sparsity": float(sum(entry["post_sparsity"] for entry in entries) / len(entries)),
            "mean_post_zero_col_ratio": float(sum(entry["post_zero_col_ratio"] for entry in entries) / len(entries)),
            "max_post_zero_col_ratio": float(max(entry["post_zero_col_ratio"] for entry in entries)),
            "mean_post_zero_row_ratio": float(sum(entry["post_zero_row_ratio"] for entry in entries) / len(entries)),
            "max_post_zero_row_ratio": float(max(entry["post_zero_row_ratio"] for entry in entries)),
            "mean_metric_std": float(sum(entry["metric_std"] or 0.0 for entry in entries) / len(entries)),
        }

    top_zero_cols = sorted(
        module_diagnostics,
        key=lambda entry: entry["post_zero_col_ratio"],
        reverse=True,
    )[:5]

    return projection_summary, [
        {
            "module_name": entry["module_name"],
            "layer": entry["layer"],
            "expert": entry["expert"],
            "projection": entry["projection"],
            "post_zero_col_ratio": entry["post_zero_col_ratio"],
            "post_zero_row_ratio": entry["post_zero_row_ratio"],
            "mask_ratio": entry["mask_ratio"],
        }
        for entry in top_zero_cols
    ]


def _layer_delta_stats(layer, inps, outs, args, attention_mask=None, position_ids=None, position_embeddings=None):
    delta_sq_sum = 0.0
    base_sq_sum = 0.0
    sample_count = 0

    for j in range(args.nsamples):
        with torch.no_grad():
            if position_embeddings is None:
                new_out = layer(inps[j].unsqueeze(0), attention_mask=attention_mask, position_ids=position_ids)[0]
            else:
                new_out = layer(
                    inps[j].unsqueeze(0),
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    position_embeddings=position_embeddings,
                )[0]

        old_out = outs[j]
        delta = (new_out - old_out).float()
        delta_sq_sum += float(delta.pow(2).mean().item())
        base_sq_sum += float(old_out.float().pow(2).mean().item())
        sample_count += 1
        outs[j] = new_out

    delta_rms = (delta_sq_sum / max(sample_count, 1)) ** 0.5
    base_rms = (base_sq_sum / max(sample_count, 1)) ** 0.5
    relative_delta_rms = delta_rms / max(base_rms, 1e-12)
    return delta_rms, base_rms, relative_delta_rms


def _collect_moe_wanda_stats(layer, subset, inps, args, attention_mask=None, position_ids=None, position_embeddings=None, outs=None):
    moe_collectors, moe_groups, moe_handles = attach_moe_wanda_hooks(layer, subset)
    try:
        for j in range(args.nsamples):
            with torch.no_grad():
                layer_out = layer(
                    inps[j].unsqueeze(0),
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    position_embeddings=position_embeddings,
                )[0]
            if outs is not None:
                outs[j] = layer_out
    finally:
        for handle in moe_handles:
            handle.remove()

    return moe_collectors, moe_groups


def _build_moe_wanda_mask(name, W_metric, args, prune_n=0, prune_m=0):
    W_mask = (torch.zeros_like(W_metric) == 1)
    if prune_n != 0:
        for ii in range(W_metric.shape[1]):
            if ii % prune_m == 0:
                tmp = W_metric[:, ii:(ii + prune_m)].float()
                W_mask.scatter_(1, ii + torch.topk(tmp, prune_n, dim=1, largest=False)[1], True)
    else:
        sort_res = torch.sort(W_metric, dim=-1, stable=True)

        if args.use_variant:
            tmp_metric = torch.cumsum(sort_res[0], dim=1)
            sum_before = W_metric.sum(dim=1)

            alpha = 0.4
            alpha_hist = [0.0, 0.8]
            W_mask, cur_sparsity = return_given_alpha(alpha, sort_res, W_metric, tmp_metric, sum_before)
            while (torch.abs(cur_sparsity - args.sparsity_ratio) > 0.001) and (alpha_hist[1] - alpha_hist[0] >= 0.001):
                if cur_sparsity > args.sparsity_ratio:
                    alpha_new = (alpha + alpha_hist[0]) / 2.0
                    alpha_hist[1] = alpha
                else:
                    alpha_new = (alpha + alpha_hist[1]) / 2.0
                    alpha_hist[0] = alpha

                alpha = alpha_new
                W_mask, cur_sparsity = return_given_alpha(alpha, sort_res, W_metric, tmp_metric, sum_before)
            print(f"alpha found {alpha} sparsity {cur_sparsity:.6f}")
        else:
            indices = sort_res[1][:, :int(W_metric.shape[1] * args.sparsity_ratio)]
            W_mask.scatter_(1, indices, True)

    if (
        prune_n == 0
        and name.endswith("down_proj")
        and getattr(args, "down_proj_max_col_zero_ratio", 1.0) < 1.0
    ):
        W_mask = apply_column_zero_ratio_guard(
            W_mask,
            W_metric,
            max_zero_ratio=args.down_proj_max_col_zero_ratio,
        )
    return W_mask


def _apply_moe_wanda_pruning(layer, subset, collectors, groups, args, prune_n=0, prune_m=0, layer_index=None, total_layers=None):
    module_diagnostics = []

    for module_idx, name in enumerate(subset, start=1):
        if layer_index is not None and total_layers is not None:
            print(f"pruning layer {layer_index + 1}/{total_layers} module {module_idx}/{len(subset)} name {name}")
        else:
            print(f"pruning module {module_idx}/{len(subset)} name {name}")
        weight = subset[name].weight.data
        W_metric = build_moe_wanda_metric(layer, name, subset[name], collectors, groups)
        if W_metric is None:
            raise RuntimeError(f"Missing MoE-Wanda metric for expert module: {name}")

        W_mask = _build_moe_wanda_mask(name, W_metric, args, prune_n=prune_n, prune_m=prune_m)

        module_diag = _collect_mask_diagnostics(name, weight, W_mask, W_metric)
        subset[name].weight.data[W_mask] = 0
        _finalize_weight_diagnostics(module_diag, subset[name].weight.data)
        module_diagnostics.append(module_diag)
        _append_diagnostic(args, {"event": "module_prune", **module_diag})

    return module_diagnostics

def find_layers(module, layers=[nn.Linear], name=''):
    """
    Recursively find the layers of a certain type in a module.

    Args:
        module (nn.Module): PyTorch module.
        layers (list): List of layer types to find.
        name (str): Name of the module.

    Returns:
        dict: Dictionary of layers of the given type(s) within the module.
    """
    if type(module) in layers and '.experts.' in name:
        return {name: module}
    res = {}
    for name1, child in module.named_children():
        res.update(find_layers(
            child, layers=layers, name=name + '.' + name1 if name != '' else name1
        ))
    return res

def check_sparsity(model):
    use_cache = model.config.use_cache 
    model.config.use_cache = False 

    layers = model.model.layers
    count = 0 
    total_params = 0
    for i in range(len(layers)):
        layer = layers[i]
        subset = find_layers(layer)

        sub_count = 0
        sub_params = 0
        for name in subset:
            W = subset[name].weight.data
            count += (W==0).sum().item()
            total_params += W.numel()

            sub_count += (W==0).sum().item()
            sub_params += W.numel()
            
        if sub_params < 1e-8:
            sub_params = 1e-8
        print(f"layer {i} sparsity {float(sub_count)/sub_params:.6f}")

    model.config.use_cache = use_cache 
    if total_params < 1e-8:
        total_params = 1e-8
    return float(count)/total_params 

def prepare_calibration_input(model, dataloader, device,nsamples=128):
    use_cache = model.config.use_cache
    model.config.use_cache = False
    layers = model.model.layers


    dtype = next(iter(model.parameters())).dtype
    inps = torch.zeros((nsamples, model.seqlen, model.config.hidden_size), dtype=dtype, device=device)
    inps.requires_grad = False
    cache = {'i': 0, 'attention_mask': None, "position_ids": None,"position_embeddings" : None}

    class Catcher(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module
        def forward(self, inp, **kwargs):
            inps[cache['i']] = inp
            cache['i'] += 1
            cache['attention_mask'] = kwargs['attention_mask']
            cache['position_ids'] = kwargs['position_ids']
            cache['position_embeddings'] = kwargs['position_embeddings']

            raise ValueError
    layers[0] = Catcher(layers[0])
    for batch in dataloader:
        try:
            model(batch[0].to(device))
        except ValueError:
            pass 
    layers[0] = layers[0].module

    outs = torch.zeros_like(inps)
    attention_mask = cache['attention_mask']
    position_ids = cache['position_ids']
    model.config.use_cache = use_cache
    position_embeddings = cache['position_embeddings']


    return inps, outs, attention_mask, position_ids ,position_embeddings 

def return_given_alpha(alpha, sort_res, W_metric, tmp_metric, sum_before):
    thres_cumsum = sum_before * alpha 
    sort_mask = tmp_metric <= thres_cumsum.reshape((-1,1))
    thres = torch.gather(sort_res[0], dim=1, index=sort_mask.sum(dim=1, keepdims=True)-1)
    W_mask = (W_metric <= thres)
    cur_sparsity = (W_mask==True).sum() / W_mask.numel()
    return W_mask, cur_sparsity

def prune_magnitude(args, model, tokenizer, device=torch.device("cuda:0"), prune_n=0, prune_m=0):
    layers = model.model.layers 

    for i in range(len(layers)):
        layer = layers[i]
        subset = find_layers(layer)

        for name in subset:
            W = subset[name].weight.data 
            W_metric = torch.abs(W)
            if prune_n != 0:
                W_mask = (torch.zeros_like(W)==1)
                for ii in range(W_metric.shape[1]):
                    if ii % prune_m == 0:
                        tmp = W_metric[:,ii:(ii+prune_m)].float()
                        W_mask.scatter_(1,ii+torch.topk(tmp, prune_n,dim=1, largest=False)[1], True)
            else:
                thresh = torch.sort(W_metric.flatten().cuda())[0][int(W.numel()*args.sparsity_ratio)].cpu()
                W_mask = (W_metric<=thresh)

            W[W_mask] = 0

def prune_moe_wanda(args, model, tokenizer, device=torch.device("cuda:0"), prune_n=0, prune_m=0):
    use_cache = model.config.use_cache 
    model.config.use_cache = False 

    print("loading calibdation data")
    dataloader, _ = get_loaders("c4",nsamples=args.nsamples,seed=args.seed,seqlen=model.seqlen,tokenizer=tokenizer)
    print("dataset loading complete")
    with torch.no_grad():
        inps, outs, attention_mask, position_ids ,position_embeddings  = prepare_calibration_input(model, dataloader, device, args.nsamples)

    layers = model.model.layers
    for i in range(len(layers)):
        layer = layers[i]
        subset = filter_moe_expert_linears(find_layers(layer))
        print(f"collecting MoE-Wanda statistics for layer {i + 1}/{len(layers)} ({len(subset)} modules)")

        if f"model.layers.{i}" in model.hf_device_map:   ## handle the case for llama-30B and llama-65B, when the device map has multiple GPUs;
            dev = model.hf_device_map[f"model.layers.{i}"]
            inps, outs, attention_mask, position_ids, position_embeddings = inps.to(dev), outs.to(dev), attention_mask.to(dev), position_ids.to(dev) , position_embeddings.to(dev)

        moe_collectors, moe_groups = _collect_moe_wanda_stats(
            layer,
            subset,
            inps,
            args,
            attention_mask=attention_mask,
            position_ids=position_ids,
            position_embeddings=position_embeddings,
            outs=outs,
        )

        module_diagnostics = _apply_moe_wanda_pruning(
            layer,
            subset,
            moe_collectors,
            moe_groups,
            args,
            prune_n=prune_n,
            prune_m=prune_m,
            layer_index=i,
            total_layers=len(layers),
        )

        delta_rms, base_rms, relative_delta_rms = _layer_delta_stats(
            layer,
            inps,
            outs,
            args,
            attention_mask=attention_mask,
            position_ids=position_ids,
            position_embeddings=position_embeddings,
        )
        projection_summary, top_zero_cols = _summarize_modules(module_diagnostics)
        top_module_name = top_zero_cols[0]["module_name"] if top_zero_cols else "none"
        top_module_ratio = top_zero_cols[0]["post_zero_col_ratio"] if top_zero_cols else 0.0
        layer_diag = {
            "event": "layer_summary",
            "layer": i,
            "module_count": len(module_diagnostics),
            "delta_rms": float(delta_rms),
            "output_rms_before": float(base_rms),
            "relative_delta_rms": float(relative_delta_rms),
            "projection_summary": projection_summary,
            "top_zero_col_modules": top_zero_cols,
        }
        _append_diagnostic(args, layer_diag)
        _append_diagnostic_summary(
            args,
            f"layer={i} delta_rms={delta_rms:.6f} relative_delta_rms={relative_delta_rms:.6f} top_zero_col_module={top_module_name} top_zero_col_ratio={top_module_ratio:.6f}",
        )
        inps, outs = outs, inps

    model.config.use_cache = use_cache 
    torch.cuda.empty_cache()


@torch.no_grad()
def prune_sparsegpt(args, model, tokenizer, dev, prune_n=0, prune_m=0):
    ## SparseGPT code available at: https://github.com/IST-DASLab/sparsegpt/tree/f5c25005a61f96a0933ca2f95705a963585aafaa
    print('Starting ...')
    dataloader, _ = get_loaders("c4",nsamples=args.nsamples,seed=args.seed,seqlen=model.seqlen,tokenizer=tokenizer)

    use_cache = model.config.use_cache
    model.config.use_cache = False
    layers = model.model.layers

    if "model.embed_tokens" in model.hf_device_map:
        dev = model.hf_device_map["model.embed_tokens"]

    dtype = next(iter(model.parameters())).dtype
    inps = torch.zeros(
        (args.nsamples, model.seqlen, model.config.hidden_size), dtype=dtype, device=dev
    )
    cache = {'i': 0, 'attention_mask': None, "position_ids": None,"position_embeddings" : None}

    class Catcher(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module
        def forward(self, inp, **kwargs):
            inps[cache['i']] = inp
            cache['i'] += 1
            cache['attention_mask'] = kwargs['attention_mask']
            cache['position_ids'] = kwargs['position_ids']
            cache['position_embeddings'] = kwargs['position_embeddings']
            raise ValueError
    layers[0] = Catcher(layers[0])
    for batch in dataloader:
        try:
            model(batch[0].to(dev))
        except ValueError:
            pass
    layers[0] = layers[0].module
    torch.cuda.empty_cache()

    outs = torch.zeros_like(inps)
    attention_mask = cache['attention_mask']
    position_ids = cache['position_ids']
    position_embeddings = cache['position_embeddings'] 

    print('Ready.')

    for i in range(len(layers)):
        layer = layers[i]
        if f"model.layers.{i}" in model.hf_device_map:
            dev = model.hf_device_map[f"model.layers.{i}"]
            print(f"layer {i} device {dev}")
            inps, outs, attention_mask, position_ids,position_embeddings = inps.to(dev), outs.to(dev), attention_mask.to(dev), position_ids.to(dev), position_embeddings.to(dev)

        subset = find_layers(layer)

        gpts = {}
        for name in subset:
            gpts[name] = SparseGPT(subset[name])

        def add_batch(name):
            def tmp(_, inp, out):
                gpts[name].add_batch(inp[0].data, out.data)
            return tmp

        handles = []
        for name in gpts:
            handles.append(subset[name].register_forward_hook(add_batch(name)))

        for j in range(args.nsamples):
            outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask, position_ids=position_ids,position_embeddings=position_embeddings)[0]
        for h in handles:
            h.remove()

        for name in gpts:
            print(i, name)
            print('Pruning ...')

            gpts[name].fasterprune(args.sparsity_ratio, prune_n=prune_n, prune_m=prune_m, percdamp=0.01, blocksize=128)
            gpts[name].free()

        for j in range(args.nsamples):
            outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask, position_ids=position_ids,position_embeddings=position_embeddings)[0]

        layers[i] = layer 
        torch.cuda.empty_cache()

        inps, outs = outs, inps

    model.config.use_cache = use_cache
    torch.cuda.empty_cache()



@torch.no_grad()
def prune_ablate(args, model, tokenizer, dev, prune_n=0, prune_m=0):
    ## SparseGPT code available at: https://github.com/IST-DASLab/sparsegpt/tree/f5c25005a61f96a0933ca2f95705a963585aafaa
    print('Starting ...')
    dataloader, _ = get_loaders("c4",nsamples=args.nsamples,seed=args.seed,seqlen=model.seqlen,tokenizer=tokenizer)

    use_cache = model.config.use_cache
    model.config.use_cache = False
    layers = model.model.layers

    if "model.embed_tokens" in model.hf_device_map:
        dev = model.hf_device_map["model.embed_tokens"]

    dtype = next(iter(model.parameters())).dtype
    inps = torch.zeros(
        (args.nsamples, model.seqlen, model.config.hidden_size), dtype=dtype, device=dev
    )
    cache = {'i': 0, 'attention_mask': None, "position_ids": None,"position_embeddings" : None}

    class Catcher(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module
        def forward(self, inp, **kwargs):
            inps[cache['i']] = inp
            cache['i'] += 1
            cache['attention_mask'] = kwargs['attention_mask']
            cache['position_ids'] = kwargs['position_ids']
            cache['position_embeddings'] = kwargs['position_embeddings']
            raise ValueError
    layers[0] = Catcher(layers[0])
    for batch in dataloader:
        try:
            model(batch[0].to(dev))
        except ValueError:
            pass
    layers[0] = layers[0].module
    torch.cuda.empty_cache()

    outs = torch.zeros_like(inps)
    attention_mask = cache['attention_mask']
    position_ids = cache['position_ids']
    position_embeddings = cache['position_embeddings'] 

    print('Ready.')

    for i in range(len(layers)):
        layer = layers[i]
        if f"model.layers.{i}" in model.hf_device_map:
            dev = model.hf_device_map[f"model.layers.{i}"]
            print(f"layer {i} device {dev}")
            inps, outs, attention_mask, position_ids,position_embeddings = inps.to(dev), outs.to(dev), attention_mask.to(dev), position_ids.to(dev), position_embeddings.to(dev)
        subset = find_layers(layer)

        gpts = {}
        for name in subset:
            gpts[name] = AblateGPT(subset[name])

        moe_collectors, moe_groups, moe_handles = attach_moe_wanda_hooks(layer, subset)

        def add_batch(name):
            def tmp(_, inp, out):
                gpts[name].add_batch(inp[0].data, out.data)
            return tmp

        handles = []
        for name in gpts:
            handles.append(subset[name].register_forward_hook(add_batch(name)))

        for j in range(args.nsamples):
            outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask, position_ids=position_ids,position_embeddings=position_embeddings)[0]
        for h in handles:
            h.remove()
        for h in moe_handles:
            h.remove()

        module_diagnostics = []
        for name in gpts:
            print(i, name)
            print('Pruning ...')

            weight = subset[name].weight.data

            if args.prune_method == "ablate_wanda_seq":
                prune_mask = build_moe_wanda_mask(
                    layer,
                    name,
                    subset[name],
                    moe_collectors,
                    moe_groups,
                    args.sparsity_ratio,
                    prune_n=prune_n,
                    prune_m=prune_m,
                )
                if prune_mask is None:
                    prune_mask = gpts[name].get_wanda_mask(args.sparsity_ratio, prune_n, prune_m)
            elif args.prune_method == "ablate_mag_seq":
                prune_mask = gpts[name].get_mag_mask(args.sparsity_ratio, prune_n, prune_m)
            elif "iter" in args.prune_method:
                prune_mask = None 

            wanda_metric = None
            if "wanda" in args.prune_method:
                wanda_metric = build_moe_wanda_metric(
                    layer,
                    name,
                    subset[name],
                    moe_collectors,
                    moe_groups,
                )

            if wanda_metric is None:
                if "wanda" in args.prune_method:
                    wanda_metric = torch.abs(weight) * torch.sqrt(gpts[name].scaler_row.reshape((1,-1)))
                elif "mag" in args.prune_method:
                    wanda_metric = torch.abs(weight)

            if prune_mask is not None and wanda_metric is not None:
                module_diag = _collect_mask_diagnostics(name, weight, prune_mask, wanda_metric)
            else:
                full_metric = wanda_metric if wanda_metric is not None else torch.abs(weight)
                module_diag = _collect_mask_diagnostics(name, weight, torch.zeros_like(weight, dtype=torch.bool), full_metric)
                module_diag["mask_ratio"] = None

            gpts[name].fasterprune(
                args,
                args.sparsity_ratio,
                mask=prune_mask,
                prune_n=prune_n,
                prune_m=prune_m,
                percdamp=0.01,
                blocksize=128,
                wanda_metric=wanda_metric,
            )
            _finalize_weight_diagnostics(module_diag, subset[name].weight.data)
            module_diagnostics.append(module_diag)
            _append_diagnostic(args, {"event": "module_prune", **module_diag})
            gpts[name].free()

        delta_rms, base_rms, relative_delta_rms = _layer_delta_stats(
            layer,
            inps,
            outs,
            args,
            attention_mask=attention_mask,
            position_ids=position_ids,
            position_embeddings=position_embeddings,
        )
        projection_summary, top_zero_cols = _summarize_modules(module_diagnostics)
        top_module_name = top_zero_cols[0]["module_name"] if top_zero_cols else "none"
        top_module_ratio = top_zero_cols[0]["post_zero_col_ratio"] if top_zero_cols else 0.0
        layer_diag = {
            "event": "layer_summary",
            "layer": i,
            "module_count": len(module_diagnostics),
            "delta_rms": float(delta_rms),
            "output_rms_before": float(base_rms),
            "relative_delta_rms": float(relative_delta_rms),
            "projection_summary": projection_summary,
            "top_zero_col_modules": top_zero_cols,
        }
        _append_diagnostic(args, layer_diag)
        _append_diagnostic_summary(
            args,
            f"layer={i} delta_rms={delta_rms:.6f} relative_delta_rms={relative_delta_rms:.6f} top_zero_col_module={top_module_name} top_zero_col_ratio={top_module_ratio:.6f}",
        )

        layers[i] = layer 
        torch.cuda.empty_cache()

        inps, outs = outs, inps

    model.config.use_cache = use_cache
    torch.cuda.empty_cache()
