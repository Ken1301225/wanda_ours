try:
    import torch
except ImportError:  # pragma: no cover - exercised only in lightweight test environments.
    torch = None


MOE_EXPERT_LINEAR_SUFFIXES = {"gate_proj", "up_proj", "down_proj"}


def _resolve_submodule(module, path):
    if not path:
        return module

    current = module
    for part in path.split("."):
        if part not in current._modules:
            return None
        current = current._modules[part]
    return current


def _split_expert_name(name):
    expert_marker = ".experts."
    if expert_marker not in name:
        return None, None, None

    parent_prefix, tail = name.split(expert_marker, 1)
    if "." not in tail:
        return None, None, None
    expert_idx, suffix = tail.split(".", 1)
    expert_prefix = f"{parent_prefix}{expert_marker}{expert_idx}"
    return parent_prefix, expert_prefix, suffix


def is_moe_expert_linear(name):
    _, expert_prefix, suffix = _split_expert_name(name)
    return expert_prefix is not None and suffix in MOE_EXPERT_LINEAR_SUFFIXES


def filter_moe_expert_linears(subset):
    return {name: module for name, module in subset.items() if is_moe_expert_linear(name)}

def apply_column_zero_ratio_guard(W_mask, W_metric, max_zero_ratio):
    if max_zero_ratio >= 1.0:
        return W_mask
    if max_zero_ratio < 0.0:
        raise ValueError(f"max_zero_ratio must be in [0, 1], got {max_zero_ratio}")

    rows, cols = W_mask.shape
    if rows == 0 or cols == 0:
        return W_mask

    device = W_mask.device
    guarded_mask = W_mask.clone()
    max_pruned_per_col = int(rows * max_zero_ratio + 1e-9)
    pruned_counts = W_mask.sum(dim=0)
    excess = (pruned_counts - max_pruned_per_col).clamp_min(0)
    released_count = int(excess.sum().item())

    if released_count == 0:
        return guarded_mask

    row_ids = torch.arange(rows, device=device, dtype=torch.long).unsqueeze(1).expand(rows, cols)

    pruned_scores = torch.where(guarded_mask, W_metric, torch.full_like(W_metric, float("-inf")))
    pruned_order = torch.argsort(pruned_scores, dim=0, descending=True)
    pruned_rank = torch.empty_like(pruned_order)
    pruned_rank.scatter_(0, pruned_order, row_ids)
    release_mask = guarded_mask & (pruned_rank < excess.unsqueeze(0))
    guarded_mask = guarded_mask & ~release_mask

    pruned_counts = guarded_mask.sum(dim=0)
    capacity = (max_pruned_per_col - pruned_counts).clamp_min(0)
    selectable_kept = (~guarded_mask) & (capacity.unsqueeze(0) > 0)

    kept_scores = torch.where(selectable_kept, W_metric, torch.full_like(W_metric, float("inf")))
    kept_order = torch.argsort(kept_scores, dim=0)
    kept_rank = torch.empty_like(kept_order)
    kept_rank.scatter_(0, kept_order, row_ids)
    selectable_mask = selectable_kept & (kept_rank < capacity.unsqueeze(0))

    flat_scores = torch.where(selectable_mask, W_metric, torch.full_like(W_metric, float("inf"))).reshape(-1)
    reprune_scores, reprune_indices = torch.topk(flat_scores, k=released_count, largest=False)

    if not torch.isfinite(reprune_scores).all():
        raise ValueError(
            f"Column zero-ratio guard is infeasible for mask shape {tuple(W_mask.shape)} at max_zero_ratio={max_zero_ratio}"
        )

    reprune_mask = torch.zeros_like(guarded_mask, dtype=torch.bool).reshape(-1)
    reprune_mask[reprune_indices] = True
    guarded_mask = guarded_mask | reprune_mask.reshape(rows, cols)

    return guarded_mask


def _flatten_tokens(x):
    if x.ndim <= 2:
        return x
    return x.reshape(-1, x.shape[-1])


def _activation_derivative(act_fn, x):
    act_name = getattr(act_fn, "__class__", type(act_fn)).__name__.lower()
    fn_name = getattr(act_fn, "__name__", "").lower()
    if "silu" in act_name or "swish" in act_name or "silu" in fn_name or "swish" in fn_name:
        sig = torch.sigmoid(x)
        return sig * (1 + x * (1 - sig))

    raise ValueError(f"Unsupported activation for MoE-Wanda gate derivative: {act_name or fn_name}")


def _build_mask_from_metric(W_metric, sparsity_ratio, prune_n=0, prune_m=0):
    W_mask = torch.zeros_like(W_metric, dtype=torch.bool)
    if prune_n != 0:
        for ii in range(W_metric.shape[1]):
            if ii % prune_m == 0:
                tmp = W_metric[:, ii : (ii + prune_m)].float()
                W_mask.scatter_(1, ii + torch.topk(tmp, prune_n, dim=1, largest=False)[1], True)
        return W_mask

    indices = torch.sort(W_metric, dim=-1, stable=True)[1][:, : int(W_metric.shape[1] * sparsity_ratio)]
    W_mask.scatter_(1, indices, True)
    return W_mask


class ExpertMoments:
    def __init__(self, expert_module):
        self.act_fn = getattr(expert_module, "act_fn", None)
        self.hidden_dim = expert_module.gate_proj.weight.shape[1]
        self.intermediate_dim = expert_module.gate_proj.weight.shape[0]

        self.down_joint_sum = torch.zeros(self.intermediate_dim, dtype=torch.float32)
        self.up_joint_sum = torch.zeros(self.intermediate_dim, self.hidden_dim, dtype=torch.float32)
        self.gate_joint_sum = torch.zeros(self.intermediate_dim, self.hidden_dim, dtype=torch.float32)

        self.current_routing = None
        self.current_gate_output = None
        self.current_up_output = None

        self.pending_routing = []

    def queue_routing(self, routing_weights):
        if routing_weights is None or routing_weights.numel() == 0:
            return
        self.pending_routing.append(routing_weights.detach())

    def expert_pre_hook(self, _, __):
        if not self.pending_routing:
            raise RuntimeError("Missing routing weights for expert forward during MoE-Wanda statistics collection.")
        routing = self.pending_routing.pop(0)
        self.current_routing = routing.float().pow(2).reshape(-1, 1)
        self.current_gate_output = None
        self.current_up_output = None

    def gate_hook(self, _, inp, out):
        if self.current_routing is None:
            return

        x = _flatten_tokens(inp[0].detach().float())
        a = _flatten_tokens(out.detach().float())
        phi_sq = self.act_fn(a).pow(2)
        x_sq = x.pow(2)
        weighted_phi_sq = phi_sq * self.current_routing
        self.up_joint_sum += weighted_phi_sq.t().matmul(x_sq).cpu()
        self.current_gate_output = a

    def up_hook(self, _, inp, out):
        if self.current_routing is None:
            return
        self.current_up_output = _flatten_tokens(out.detach().float())

    def down_pre_hook(self, _, inp):
        if self.current_routing is None:
            return

        h = _flatten_tokens(inp[0].detach().float())
        self.down_joint_sum += (h.pow(2) * self.current_routing).sum(dim=0).cpu()

        if self.current_gate_output is None or self.current_up_output is None:
            raise RuntimeError("Incomplete expert activations for MoE-Wanda gate statistics collection.")

    def finalize_gate_stats(self, x):
        if self.current_routing is None:
            return
        if self.current_gate_output is None or self.current_up_output is None:
            raise RuntimeError("Incomplete expert activations for MoE-Wanda gate statistics collection.")

        x = _flatten_tokens(x.detach().float())
        slope_sq = _activation_derivative(self.act_fn, self.current_gate_output).pow(2)
        gate_scale_sq = self.current_up_output.pow(2) * slope_sq
        weighted_gate_scale_sq = gate_scale_sq * self.current_routing
        self.gate_joint_sum += weighted_gate_scale_sq.t().matmul(x.pow(2)).cpu()

        self.current_routing = None
        self.current_gate_output = None
        self.current_up_output = None


class MoeGroupState:
    def __init__(self):
        self.total_tokens = 0


def _parse_gate_output(gate_output):
    if torch.is_tensor(gate_output):
        return gate_output, None, None

    if not isinstance(gate_output, (tuple, list)):
        return None, None, None

    tensors = [item for item in gate_output if torch.is_tensor(item)]
    if not tensors:
        return None, None, None

    if len(tensors) == 1:
        return tensors[0], None, None

    first, second = tensors[0], tensors[1]
    if first.dtype in (torch.int8, torch.int16, torch.int32, torch.int64):
        return None, first, second
    if second.dtype in (torch.int8, torch.int16, torch.int32, torch.int64):
        return None, second, first
    return tensors[0], None, None


def _compute_topk_from_module(moe_module, hidden_states):
    hidden_states = hidden_states.detach()
    gate_output = moe_module.gate(hidden_states)
    router_logits, selected_experts, routing_weights = _parse_gate_output(gate_output)

    if selected_experts is not None and routing_weights is not None:
        return selected_experts.detach(), routing_weights.detach().float()

    if router_logits is None:
        raise RuntimeError("Unsupported MoE gate output for MoE-Wanda statistics collection.")

    top_k = getattr(moe_module, "top_k", None)
    if top_k is None:
        top_k = getattr(moe_module, "num_experts_per_tok", None)
    if top_k is None and hasattr(moe_module, "config"):
        top_k = getattr(moe_module.config, "num_experts_per_tok", None)
    if top_k is None:
        raise RuntimeError("Unable to infer top-k routing parameter for MoE-Wanda statistics collection.")

    if hasattr(moe_module, "route_tokens_to_experts"):
        reshaped_logits = router_logits
        if hidden_states.ndim == 2:
            reshaped_logits = router_logits.unsqueeze(0)
        selected_experts, routing_weights = moe_module.route_tokens_to_experts(reshaped_logits)
        return selected_experts.detach(), routing_weights.detach().float()

    routing_probs = torch.softmax(router_logits.float(), dim=-1)
    routing_weights, selected_experts = torch.topk(routing_probs, top_k, dim=-1)
    if getattr(moe_module, "norm_topk_prob", False):
        routing_weights = routing_weights / routing_weights.sum(dim=-1, keepdim=True).clamp_min_(1e-12)
    return selected_experts.detach(), routing_weights.detach().float()


def _queue_qwen_routing(moe_module, hidden_states, expert_map):
    flat_hidden = _flatten_tokens(hidden_states)
    selected_experts, routing_weights = _compute_topk_from_module(moe_module, flat_hidden)
    selected_experts = selected_experts.reshape(-1, selected_experts.shape[-1])
    routing_weights = routing_weights.reshape(-1, routing_weights.shape[-1])

    expert_mask = torch.nn.functional.one_hot(selected_experts, num_classes=moe_module.num_experts).permute(2, 1, 0)
    for expert_idx, collector in expert_map.items():
        mask = expert_mask[expert_idx]
        topk_pos, token_idx = torch.where(mask)
        collector.queue_routing(routing_weights[token_idx, topk_pos])

    return flat_hidden.shape[0]


def _queue_deepseek_routing(moe_module, hidden_states, expert_map):
    flat_hidden = _flatten_tokens(hidden_states)
    selected_experts, routing_weights = _compute_topk_from_module(moe_module, hidden_states)
    selected_experts = selected_experts.reshape(-1, selected_experts.shape[-1])
    routing_weights = routing_weights.reshape(-1, routing_weights.shape[-1])

    flat_selected = selected_experts.reshape(-1)
    flat_routing = routing_weights.reshape(-1)
    idxs = flat_selected.argsort()
    sorted_experts = flat_selected[idxs]
    sorted_routing = flat_routing[idxs]

    for expert_idx, collector in expert_map.items():
        collector.queue_routing(sorted_routing[sorted_experts == expert_idx])

    return flat_hidden.shape[0]


def attach_moe_wanda_hooks(layer, subset):
    collectors = {}
    groups = {}
    handles = []

    for name in subset:
        parent_prefix, expert_prefix, suffix = _split_expert_name(name)
        if expert_prefix is None:
            continue

        if expert_prefix not in collectors:
            expert_module = _resolve_submodule(layer, expert_prefix)
            if expert_module is None:
                continue
            collectors[expert_prefix] = ExpertMoments(expert_module)

        groups.setdefault(parent_prefix, {"experts": {}, "state": MoeGroupState()})
        expert_idx = int(expert_prefix.split(".experts.", 1)[1].split(".", 1)[0])
        groups[parent_prefix]["experts"][expert_idx] = collectors[expert_prefix]

    seen_modules = set()
    for expert_prefix, collector in collectors.items():
        expert_module = _resolve_submodule(layer, expert_prefix)
        if expert_module is not None and expert_module not in seen_modules:
            handles.append(expert_module.register_forward_pre_hook(collector.expert_pre_hook))
            seen_modules.add(expert_module)

        gate_proj = _resolve_submodule(layer, f"{expert_prefix}.gate_proj")
        if gate_proj is not None and gate_proj not in seen_modules:
            handles.append(gate_proj.register_forward_hook(collector.gate_hook))
            seen_modules.add(gate_proj)

        up_proj = _resolve_submodule(layer, f"{expert_prefix}.up_proj")
        if up_proj is not None and up_proj not in seen_modules:
            handles.append(up_proj.register_forward_hook(collector.up_hook))
            seen_modules.add(up_proj)

        down_proj = _resolve_submodule(layer, f"{expert_prefix}.down_proj")
        if down_proj is not None and down_proj not in seen_modules:
            handles.append(down_proj.register_forward_pre_hook(collector.down_pre_hook))
            seen_modules.add(down_proj)

        if expert_module is not None and expert_module not in seen_modules:
            seen_modules.add(expert_module)

        def make_expert_post_hook(local_collector):
            def expert_post_hook(module, inp, out):
                local_collector.finalize_gate_stats(inp[0])
            return expert_post_hook

        handles.append(expert_module.register_forward_hook(make_expert_post_hook(collector)))

    for parent_prefix, payload in groups.items():
        moe_module = _resolve_submodule(layer, parent_prefix)
        if moe_module is None:
            continue

        expert_map = payload["experts"]
        state = payload["state"]

        def make_moe_pre_hook(local_module, local_expert_map, local_state):
            def moe_pre_hook(module, inp):
                hidden_states = inp[0]
                module_name = type(local_module).__name__.lower()
                if "deepseek" in module_name or hasattr(local_module, "moe_infer"):
                    token_count = _queue_deepseek_routing(local_module, hidden_states, local_expert_map)
                else:
                    token_count = _queue_qwen_routing(local_module, hidden_states, local_expert_map)
                local_state.total_tokens += token_count
            return moe_pre_hook

        handles.append(moe_module.register_forward_pre_hook(make_moe_pre_hook(moe_module, expert_map, state)))

    return collectors, groups, handles


def build_moe_wanda_metric(layer, name, linear_module, collectors, groups):
    parent_prefix, expert_prefix, suffix = _split_expert_name(name)
    if expert_prefix is None or expert_prefix not in collectors or parent_prefix not in groups:
        return None

    total_tokens = groups[parent_prefix]["state"].total_tokens
    if total_tokens <= 0:
        return None

    collector = collectors[expert_prefix]
    W = linear_module.weight.data.float().abs()
    denom = float(total_tokens)

    if suffix == "down_proj":
        moment = (collector.down_joint_sum / denom).to(linear_module.weight.device).clamp_min_(0)
        return W * torch.sqrt(moment).reshape(1, -1)

    if suffix == "up_proj":
        moment = (collector.up_joint_sum / denom).to(linear_module.weight.device).clamp_min_(0)
        return W * torch.sqrt(moment)

    if suffix == "gate_proj":
        moment = (collector.gate_joint_sum / denom).to(linear_module.weight.device).clamp_min_(0)
        return W * torch.sqrt(moment)

    return None


def build_moe_wanda_mask(layer, name, linear_module, collectors, groups, sparsity_ratio, prune_n=0, prune_m=0):
    W_metric = build_moe_wanda_metric(layer, name, linear_module, collectors, groups)
    if W_metric is None:
        return None
    return _build_mask_from_metric(W_metric, sparsity_ratio, prune_n=prune_n, prune_m=prune_m)
