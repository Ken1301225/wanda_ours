import types
import unittest

from lib.moe_wanda import (
    build_moe_wanda_metric,
    filter_moe_expert_linears,
    is_moe_expert_linear,
    split_moe_pruning_stages,
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


    class DummyLayer(nn.Module):
        def __init__(self):
            super().__init__()
            self.mlp = nn.Module()
            self.mlp.experts = nn.ModuleList([DummyExpert()])


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

    def test_split_moe_pruning_stages_separates_upstream_and_downstream(self):
        subset = {
            "mlp.experts.0.gate_proj": object(),
            "mlp.experts.0.up_proj": object(),
            "mlp.experts.0.down_proj": object(),
            "mlp.experts.1.up_proj": object(),
            "mlp.experts.1.down_proj": object(),
        }

        upstream, downstream = split_moe_pruning_stages(subset)

        self.assertEqual(
            set(upstream),
            {
                "mlp.experts.0.gate_proj",
                "mlp.experts.0.up_proj",
                "mlp.experts.1.up_proj",
            },
        )
        self.assertEqual(
            set(downstream),
            {
                "mlp.experts.0.down_proj",
                "mlp.experts.1.down_proj",
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
    def test_build_moe_wanda_metric_for_up_proj_uses_local_h_stability_only(self):
        layer = DummyLayer()
        expert = layer.mlp.experts[0]
        expert.up_proj.weight.data = torch.tensor([[1.0, -2.0], [-3.0, 4.0], [5.0, -6.0]])
        expert.down_proj.weight.data = torch.tensor([[10.0, 20.0, 30.0], [40.0, 50.0, 60.0]])

        collectors = {
            "mlp.experts.0": types.SimpleNamespace(
                down_joint_sum=torch.zeros(3),
                up_joint_sum=torch.tensor([[4.0, 9.0], [16.0, 25.0], [36.0, 49.0]]),
                gate_joint_sum=torch.zeros(3, 2),
            )
        }
        groups = {"mlp": {"state": types.SimpleNamespace(total_tokens=4)}}

        metric = build_moe_wanda_metric(
            layer,
            "mlp.experts.0.up_proj",
            expert.up_proj,
            collectors=collectors,
            groups=groups,
        )

        expected = expert.up_proj.weight.data.abs() * torch.tensor([[1.0, 1.5], [2.0, 2.5], [3.0, 3.5]])
        self.assertTrue(torch.allclose(metric, expected))
