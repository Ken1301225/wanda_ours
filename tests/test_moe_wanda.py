import types
import unittest

from lib.moe_wanda import (
    attach_moe_wanda_hooks,
    build_cluster_global_masks,
    build_moe_wanda_metric,
    filter_moe_expert_linears,
    is_moe_expert_linear,
)

try:
    import torch
    import torch.nn as nn
except ImportError:  # pragma: no cover - depends on local runtime.
    torch = None
    nn = None


if torch is not None:
    class DummyExpert(nn.Module):
        def __init__(self):
            super().__init__()
            self.gate_proj = nn.Linear(2, 3, bias=False)
            self.up_proj = nn.Linear(2, 3, bias=False)
            self.down_proj = nn.Linear(3, 2, bias=False)
            self.act_fn = torch.nn.functional.silu

        def forward(self, x):
            hidden = self.act_fn(self.gate_proj(x)) * self.up_proj(x)
            return self.down_proj(hidden)


    class DummyMoe(nn.Module):
        def __init__(self):
            super().__init__()
            self.num_experts = 2
            self.top_k = 1
            self.gate = nn.Linear(2, 2, bias=False)
            self.experts = nn.ModuleList([DummyExpert(), DummyExpert()])

        def forward(self, hidden_states):
            flat_hidden = hidden_states.reshape(-1, hidden_states.shape[-1])
            router_logits = self.gate(flat_hidden)
            routing_probs = torch.softmax(router_logits, dim=-1)
            routing_weights, selected_experts = torch.topk(routing_probs, self.top_k, dim=-1)

            outputs = torch.zeros(
                flat_hidden.shape[0],
                self.experts[0].down_proj.weight.shape[0],
                dtype=flat_hidden.dtype,
            )
            for expert_idx, expert in enumerate(self.experts):
                token_mask = selected_experts.squeeze(-1) == expert_idx
                if token_mask.any():
                    expert_out = expert(flat_hidden[token_mask])
                    outputs[token_mask] = expert_out * routing_weights[token_mask]
            return outputs.reshape(hidden_states.shape)


    class DummyLayer(nn.Module):
        def __init__(self):
            super().__init__()
            self.mlp = nn.Module()
            self.mlp.experts = nn.ModuleList([DummyExpert()])


    class DummySparseMoeLayer(nn.Module):
        def __init__(self):
            super().__init__()
            self.mlp = DummyMoe()

        def forward(self, x):
            return self.mlp(x)


class MoeWandaTests(unittest.TestCase):
    def test_is_moe_expert_linear(self):
        self.assertTrue(is_moe_expert_linear("mlp.experts.0.gate_proj"))
        self.assertTrue(is_moe_expert_linear("mlp.experts.0.up_proj"))
        self.assertTrue(is_moe_expert_linear("mlp.experts.0.down_proj"))
        self.assertFalse(is_moe_expert_linear("self_attn.q_proj"))
        self.assertFalse(is_moe_expert_linear("mlp.experts.0"))

    def test_filter_moe_expert_linears_keeps_only_expert_projections(self):
        subset = {
            "self_attn.q_proj": object(),
            "mlp.experts.0.gate_proj": object(),
            "mlp.experts.0.up_proj": object(),
            "mlp.experts.0.down_proj": object(),
            "mlp.shared_expert.gate_proj": object(),
        }

        filtered = filter_moe_expert_linears(subset)

        self.assertEqual(
            set(filtered),
            {
                "mlp.experts.0.gate_proj",
                "mlp.experts.0.up_proj",
                "mlp.experts.0.down_proj",
            },
        )

    @unittest.skipIf(torch is None, "torch is not installed in the current test runtime")
    def test_build_moe_wanda_metric_returns_none_for_non_expert_modules(self):
        layer = DummyLayer()

        metric = build_moe_wanda_metric(
            layer,
            "self_attn.q_proj",
            nn.Linear(2, 2, bias=False),
            collectors={},
            groups={},
        )

        self.assertIsNone(metric)

    @unittest.skipIf(torch is None, "torch is not installed in the current test runtime")
    def test_build_moe_wanda_metric_for_down_proj_uses_routing_weighted_moment(self):
        layer = DummyLayer()
        down_proj = layer.mlp.experts[0].down_proj
        down_proj.weight.data = torch.tensor([[1.0, -2.0, 3.0], [-4.0, 5.0, -6.0]])

        collectors = {
            "mlp.experts.0": types.SimpleNamespace(
                down_joint_sum=torch.tensor([4.0, 9.0, 16.0]),
                up_joint_sum=torch.zeros(3, 2),
                gate_joint_sum=torch.zeros(3, 2),
            )
        }
        groups = {"mlp": {"state": types.SimpleNamespace(total_tokens=4)}}

        metric = build_moe_wanda_metric(
            layer,
            "mlp.experts.0.down_proj",
            down_proj,
            collectors=collectors,
            groups=groups,
        )

        expected = down_proj.weight.data.abs() * torch.tensor([[1.0, 1.5, 2.0]])
        self.assertTrue(torch.allclose(metric, expected))

    @unittest.skipIf(torch is None, "torch is not installed in the current test runtime")
    def test_attach_moe_wanda_hooks_dense_softmax_collects_unselected_expert_stats(self):
        layer = DummySparseMoeLayer()
        layer.mlp.gate.weight.data = torch.tensor([[3.0, 0.0], [-3.0, 0.0]])

        subset = {
            "mlp.experts.0.gate_proj": layer.mlp.experts[0].gate_proj,
            "mlp.experts.0.up_proj": layer.mlp.experts[0].up_proj,
            "mlp.experts.0.down_proj": layer.mlp.experts[0].down_proj,
            "mlp.experts.1.gate_proj": layer.mlp.experts[1].gate_proj,
            "mlp.experts.1.up_proj": layer.mlp.experts[1].up_proj,
            "mlp.experts.1.down_proj": layer.mlp.experts[1].down_proj,
        }
        hidden_states = torch.tensor([[[1.0, 0.5], [2.0, -0.5]]])

        collectors, _, handles = attach_moe_wanda_hooks(layer, subset, routing_mode="dense_softmax")
        try:
            layer(hidden_states)
        finally:
            for handle in handles:
                handle.remove()

        unselected_expert = collectors["mlp.experts.1"]
        self.assertGreater(unselected_expert.down_joint_sum.sum().item(), 0.0)
        self.assertGreater(unselected_expert.up_joint_sum.sum().item(), 0.0)
        self.assertGreater(unselected_expert.gate_joint_sum.sum().item(), 0.0)

    @unittest.skipIf(torch is None, "torch is not installed in the current test runtime")
    def test_attach_moe_wanda_hooks_dense_softmax_respects_routing_power(self):
        layer = DummySparseMoeLayer()
        layer.mlp.gate.weight.data = torch.tensor([[1.0, 0.0], [-1.0, 0.0]])

        subset = {
            "mlp.experts.0.gate_proj": layer.mlp.experts[0].gate_proj,
            "mlp.experts.0.up_proj": layer.mlp.experts[0].up_proj,
            "mlp.experts.0.down_proj": layer.mlp.experts[0].down_proj,
            "mlp.experts.1.gate_proj": layer.mlp.experts[1].gate_proj,
            "mlp.experts.1.up_proj": layer.mlp.experts[1].up_proj,
            "mlp.experts.1.down_proj": layer.mlp.experts[1].down_proj,
        }
        hidden_states = torch.tensor([[[1.0, 0.5], [2.0, -0.5]]])

        collectors_p1, _, handles_p1 = attach_moe_wanda_hooks(
            layer, subset, routing_mode="dense_softmax", routing_power=1.0
        )
        try:
            layer(hidden_states)
        finally:
            for handle in handles_p1:
                handle.remove()

        collectors_p2, _, handles_p2 = attach_moe_wanda_hooks(
            layer, subset, routing_mode="dense_softmax", routing_power=2.0
        )
        try:
            layer(hidden_states)
        finally:
            for handle in handles_p2:
                handle.remove()

        expert0_p1 = collectors_p1["mlp.experts.0"].down_joint_sum.sum().item()
        expert0_p2 = collectors_p2["mlp.experts.0"].down_joint_sum.sum().item()
        self.assertGreater(expert0_p1, expert0_p2)

    @unittest.skipIf(torch is None, "torch is not installed in the current test runtime")
    def test_build_cluster_global_masks_redistributes_budget_within_cluster(self):
        metrics_by_name = {
            "mlp.experts.0.up_proj": torch.tensor([[10.0, 9.0], [8.0, 7.0]]),
            "mlp.experts.1.up_proj": torch.tensor([[4.0, 3.0], [2.0, 1.0]]),
            "mlp.experts.2.up_proj": torch.tensor([[6.0, 5.0], [0.4, 0.3]]),
            "mlp.experts.3.up_proj": torch.tensor([[0.2, 0.1], [0.05, 0.01]]),
        }
        cluster_by_name = {
            "mlp.experts.0.up_proj": 0,
            "mlp.experts.1.up_proj": 0,
            "mlp.experts.2.up_proj": 1,
            "mlp.experts.3.up_proj": 1,
        }

        masks = build_cluster_global_masks(metrics_by_name, cluster_by_name, sparsity_ratio=0.5)

        self.assertEqual(int(masks["mlp.experts.0.up_proj"].sum().item()), 0)
        self.assertEqual(int(masks["mlp.experts.1.up_proj"].sum().item()), 4)
        self.assertEqual(int(masks["mlp.experts.2.up_proj"].sum().item()), 0)
        self.assertEqual(int(masks["mlp.experts.3.up_proj"].sum().item()), 4)
