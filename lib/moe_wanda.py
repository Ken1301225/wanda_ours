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


def _apply_routing_power(routing_weights, routing_power):
    if routing_power <= 0:
        raise ValueError(f"MoE-Wanda routing power must be > 0, got {routing_power}")
    routing_weights = routing_weights.detach().float()
    if routing_power == 1:
        return routing_weights
    return routing_weights.pow(routing_power)


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


def _build_global_mask_from_flat_metric(flat_metric, sparsity_ratio):
    prune_count = int(flat_metric.numel() * sparsity_ratio)
    flat_mask = torch.zeros_like(flat_metric, dtype=torch.bool)
    if prune_count <= 0:
        return flat_mask

    indices = torch.argsort(flat_metric.float(), stable=True)[:prune_count]
    flat_mask[indices] = True
    return flat_mask


def _normalize_router_trace_matrix(trace_matrix):
    trace_matrix = trace_matrix.float()
    trace_matrix = trace_matrix - trace_matrix.mean(dim=1, keepdim=True)
    norms = torch.norm(trace_matrix, p=2, dim=1, keepdim=True)
    normalized = trace_matrix / norms.clamp_min_(1e-12)
    normalized = torch.where(norms > 1e-12, normalized, torch.zeros_like(normalized))
    return normalized


def _run_fixed_kmeans(features, cluster_k, seed=0, max_iters=25):
    num_points = features.shape[0]
    if cluster_k <= 1 or num_points <= 1:
        return torch.zeros(num_points, dtype=torch.long)
    if cluster_k >= num_points:
        return torch.arange(num_points, dtype=torch.long)

    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed))
    perm = torch.randperm(num_points, generator=generator)
    centroids = features[perm[:cluster_k]].clone()
    labels = None

    for _ in range(max_iters):
        similarity = features.matmul(centroids.t())
        new_labels = torch.argmax(similarity, dim=1)
        if labels is not None and torch.equal(labels, new_labels):
            break
        labels = new_labels

        centroid_rows = []
        for cluster_idx in range(cluster_k):
            members = features[labels == cluster_idx]
            if members.numel() == 0:
                centroid_rows.append(features[perm[cluster_idx % num_points]])
                continue

            centroid = members.mean(dim=0)
            norm = torch.norm(centroid, p=2)
            if norm > 1e-12:
                centroid = centroid / norm
            centroid_rows.append(centroid)
        centroids = torch.stack(centroid_rows, dim=0)

    return labels if labels is not None else torch.zeros(num_points, dtype=torch.long)


def build_expert_clusters(groups, cluster_k, seed=0):
    cluster_assignments = {}
    cluster_summaries = {}

    for parent_prefix, payload in groups.items():
        expert_ids = sorted(payload["experts"])
        state = payload["state"]
        if not state.router_logits_trace:
            raise RuntimeError("Missing router logits needed for expert clustering.")

        router_logits = torch.cat(state.router_logits_trace, dim=0)
        expert_trace_matrix = router_logits.t()[expert_ids]
        features = _normalize_router_trace_matrix(expert_trace_matrix)
        effective_k = min(int(cluster_k), len(expert_ids))
        labels = _run_fixed_kmeans(features, effective_k, seed=seed)

        assignments = {
            expert_id: int(labels[idx].item())
            for idx, expert_id in enumerate(expert_ids)
        }
        cluster_sizes = {}
        for cluster_id in range(effective_k):
            cluster_sizes[cluster_id] = sum(
                1 for assigned_cluster in assignments.values() if assigned_cluster == cluster_id
            )

        cluster_assignments[parent_prefix] = assignments
        cluster_summaries[parent_prefix] = {
            "cluster_k_requested": int(cluster_k),
            "cluster_count": int(effective_k),
            "expert_count": len(expert_ids),
            "cluster_sizes": cluster_sizes,
        }

    return cluster_assignments, cluster_summaries


def build_cluster_global_masks(metrics_by_name, cluster_by_name, sparsity_ratio, prune_n=0, prune_m=0):
    if prune_n != 0:
        return {
            name: _build_mask_from_metric(metric, sparsity_ratio, prune_n=prune_n, prune_m=prune_m)
            for name, metric in metrics_by_name.items()
        }

    grouped_names = {}
    for name, metric in metrics_by_name.items():
        parent_prefix, _, suffix = _split_expert_name(name)
        if suffix is None or parent_prefix is None:
            raise ValueError(f"Expected MoE expert module name, got {name}")
        if name not in cluster_by_name:
            raise KeyError(f"Missing cluster assignment for {name}")
        key = (parent_prefix, cluster_by_name[name], suffix)
        grouped_names.setdefault(key, []).append((name, metric))

    masks = {}
    for _, grouped_metrics in grouped_names.items():
        flat_metric = torch.cat([metric.reshape(-1) for _, metric in grouped_metrics], dim=0)
        flat_mask = _build_global_mask_from_flat_metric(flat_metric, sparsity_ratio)

        start = 0
        for name, metric in grouped_metrics:
            end = start + metric.numel()
            masks[name] = flat_mask[start:end].reshape_as(metric)
            start = end

    return masks


class ExpertMoments:
    def __init__(self, expert_module):
        self.expert_module = expert_module
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

    def queue_routing(self, routing_weights, routing_power=2.0):
        if routing_weights is None or routing_weights.numel() == 0:
            return
        self.pending_routing.append((routing_weights.detach(), routing_power))

    def accumulate_dense(self, hidden_states, routing_weights, routing_power=2.0):
        if routing_weights is None or routing_weights.numel() == 0:
            return

        x = _flatten_tokens(hidden_states.detach())
        routing_scaled = _apply_routing_power(routing_weights, routing_power).reshape(-1, 1)
        if x.shape[0] != routing_scaled.shape[0]:
            raise RuntimeError("Dense MoE-Wanda routing weights do not align with token count.")

        device = self.expert_module.gate_proj.weight.device
        dtype = self.expert_module.gate_proj.weight.dtype
        expert_input = x.to(device=device, dtype=dtype)
        x_sq = x.to(device=device, dtype=torch.float32).pow(2)

        gate_output = _flatten_tokens(self.expert_module.gate_proj(expert_input).detach()).float()
        up_output = _flatten_tokens(self.expert_module.up_proj(expert_input).detach()).float()
        activated_gate = self.act_fn(gate_output)
        weighted_phi_sq = activated_gate.pow(2) * routing_scaled.to(device)
        self.up_joint_sum += weighted_phi_sq.t().matmul(x_sq).cpu()

        expert_hidden = activated_gate * up_output
        self.down_joint_sum += (expert_hidden.pow(2) * routing_scaled.to(device)).sum(dim=0).cpu()

        slope_sq = _activation_derivative(self.act_fn, gate_output).pow(2)
        gate_scale_sq = up_output.pow(2) * slope_sq
        self.gate_joint_sum += (gate_scale_sq * routing_scaled.to(device)).t().matmul(x_sq).cpu()

    def expert_pre_hook(self, _, __):
        if not self.pending_routing:
            raise RuntimeError("Missing routing weights for expert forward during MoE-Wanda statistics collection.")
        routing_weights, routing_power = self.pending_routing.pop(0)
        self.current_routing = _apply_routing_power(routing_weights, routing_power).reshape(-1, 1)
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
        self.router_logits_trace = []


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


def _compute_router_logits(moe_module, hidden_states):
    hidden_states = hidden_states.detach()
    gate_output = moe_module.gate(hidden_states)
    router_logits, _, _ = _parse_gate_output(gate_output)

    if router_logits is None:
        raise RuntimeError("Router logits are required for clustered MoE-Wanda pruning.")

    return router_logits.detach().float()


def _compute_dense_routing_probs(moe_module, hidden_states):
    hidden_states = hidden_states.detach()
    gate_output = moe_module.gate(hidden_states)
    router_logits, _, _ = _parse_gate_output(gate_output)

    if router_logits is None:
        raise RuntimeError(
            "Dense-softmax MoE-Wanda statistics require router logits from moe_module.gate(...)."
        )

    return torch.softmax(router_logits.float(), dim=-1)


def _queue_qwen_routing(moe_module, hidden_states, expert_map, routing_power=2.0):
    flat_hidden = _flatten_tokens(hidden_states)
    selected_experts, routing_weights = _compute_topk_from_module(moe_module, flat_hidden)
    selected_experts = selected_experts.reshape(-1, selected_experts.shape[-1])
    routing_weights = routing_weights.reshape(-1, routing_weights.shape[-1])

    expert_mask = torch.nn.functional.one_hot(selected_experts, num_classes=moe_module.num_experts).permute(2, 1, 0)
    for expert_idx, collector in expert_map.items():
        mask = expert_mask[expert_idx]
        topk_pos, token_idx = torch.where(mask)
        collector.queue_routing(routing_weights[token_idx, topk_pos], routing_power=routing_power)

    return flat_hidden.shape[0]


def _queue_deepseek_routing(moe_module, hidden_states, expert_map, routing_power=2.0):
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
        collector.queue_routing(sorted_routing[sorted_experts == expert_idx], routing_power=routing_power)

    return flat_hidden.shape[0]


def _accumulate_dense_routing(flat_hidden, routing_probs, expert_map, routing_power=2.0):
    routing_probs = routing_probs.reshape(-1, routing_probs.shape[-1])
    for expert_idx, collector in expert_map.items():
        collector.accumulate_dense(flat_hidden, routing_probs[:, expert_idx], routing_power=routing_power)
    return flat_hidden.shape[0]


def attach_moe_wanda_hooks(layer, subset, routing_mode="topk", routing_power=2.0, collect_router_logits=False):
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

    if routing_mode not in {"topk", "dense_softmax"}:
        raise ValueError(f"Unsupported MoE-Wanda routing mode: {routing_mode}")
    if routing_power <= 0:
        raise ValueError(f"MoE-Wanda routing power must be > 0, got {routing_power}")

    if routing_mode == "topk":
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
                flat_hidden = _flatten_tokens(hidden_states)
                if collect_router_logits:
                    router_logits_input = hidden_states if ("deepseek" in module_name or hasattr(local_module, "moe_infer")) else flat_hidden
                    router_logits = _compute_router_logits(local_module, router_logits_input)
                    local_state.router_logits_trace.append(router_logits.reshape(-1, router_logits.shape[-1]).cpu())
                if routing_mode == "dense_softmax":
                    if "deepseek" in module_name or hasattr(local_module, "moe_infer"):
                        routing_probs = _compute_dense_routing_probs(local_module, hidden_states)
                    else:
                        routing_probs = _compute_dense_routing_probs(local_module, flat_hidden)
                    token_count = _accumulate_dense_routing(
                        flat_hidden, routing_probs, local_expert_map, routing_power=routing_power
                    )
                elif "deepseek" in module_name or hasattr(local_module, "moe_infer"):
                    token_count = _queue_deepseek_routing(
                        local_module, hidden_states, local_expert_map, routing_power=routing_power
                    )
                else:
                    token_count = _queue_qwen_routing(
                        local_module, hidden_states, local_expert_map, routing_power=routing_power
                    )
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

    down_proj = _resolve_submodule(layer, f"{expert_prefix}.down_proj")
    if down_proj is None:
        return None
    down_norm = torch.norm(down_proj.weight.data.float(), p=2, dim=0).to(linear_module.weight.device)

    if suffix == "up_proj":
        moment = (collector.up_joint_sum / denom).to(linear_module.weight.device).clamp_min_(0)
        return W * down_norm.reshape(-1, 1) * torch.sqrt(moment)

    if suffix == "gate_proj":
        moment = (collector.gate_joint_sum / denom).to(linear_module.weight.device).clamp_min_(0)
        return W * down_norm.reshape(-1, 1) * torch.sqrt(moment)

    return None


def build_moe_wanda_mask(layer, name, linear_module, collectors, groups, sparsity_ratio, prune_n=0, prune_m=0):
    W_metric = build_moe_wanda_metric(layer, name, linear_module, collectors, groups)
    if W_metric is None:
        return None
    return _build_mask_from_metric(W_metric, sparsity_ratio, prune_n=prune_n, prune_m=prune_m)
