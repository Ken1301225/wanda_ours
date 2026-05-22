import unittest

from debug.analyze_pruned_moe import (
    aggregate_expert_rows,
    aggregate_layer_projection_rows,
    build_parser,
    parse_expert_module_name,
    select_pattern_samples,
)


class AnalyzePrunedMoeTests(unittest.TestCase):
    def test_parser_accepts_multiple_models(self):
        parser = build_parser()

        args = parser.parse_args(
            [
                "--model",
                "/tmp/model_a",
                "--model",
                "/tmp/model_b",
                "--output-dir",
                "/tmp/out",
            ]
        )

        self.assertEqual(args.model, ["/tmp/model_a", "/tmp/model_b"])
        self.assertEqual(args.output_dir, "/tmp/out")
        self.assertEqual(args.max_pattern_plots, 9)

    def test_parse_expert_module_name(self):
        parsed = parse_expert_module_name("model.layers.7.mlp.experts.13.down_proj")

        self.assertEqual(
            parsed,
            {
                "layer": 7,
                "expert": 13,
                "projection": "down_proj",
            },
        )
        self.assertIsNone(parse_expert_module_name("model.layers.7.self_attn.q_proj"))

    def test_aggregate_expert_rows(self):
        rows = [
            {
                "model_label": "model_a",
                "layer": 0,
                "expert": 0,
                "projection": "gate_proj",
                "sparsity": 0.5,
                "zero_row_ratio": 0.25,
                "zero_col_ratio": 0.10,
            },
            {
                "model_label": "model_a",
                "layer": 0,
                "expert": 0,
                "projection": "up_proj",
                "sparsity": 0.75,
                "zero_row_ratio": 0.50,
                "zero_col_ratio": 0.20,
            },
        ]

        aggregated = aggregate_expert_rows(rows)

        self.assertEqual(len(aggregated), 1)
        self.assertEqual(aggregated[0]["projection_count"], 2)
        self.assertAlmostEqual(aggregated[0]["mean_sparsity"], 0.625)
        self.assertAlmostEqual(aggregated[0]["max_zero_col_ratio"], 0.20)

    def test_aggregate_layer_projection_rows(self):
        rows = [
            {
                "model_label": "model_a",
                "layer": 0,
                "expert": 0,
                "projection": "down_proj",
                "sparsity": 0.5,
                "zero_row_ratio": 0.0,
                "zero_col_ratio": 0.1,
            },
            {
                "model_label": "model_a",
                "layer": 0,
                "expert": 1,
                "projection": "down_proj",
                "sparsity": 0.75,
                "zero_row_ratio": 0.2,
                "zero_col_ratio": 0.4,
            },
        ]

        aggregated = aggregate_layer_projection_rows(rows)

        self.assertEqual(len(aggregated), 1)
        self.assertAlmostEqual(aggregated[0]["mean_sparsity"], 0.625)
        self.assertAlmostEqual(aggregated[0]["max_zero_row_ratio"], 0.2)
        self.assertEqual(aggregated[0]["expert_count"], 2)

    def test_select_pattern_samples_balances_projections(self):
        rows = [
            {
                "model_label": "model_a",
                "module_name": "model.layers.0.mlp.experts.0.gate_proj",
                "projection": "gate_proj",
                "zero_col_ratio": 0.4,
                "zero_row_ratio": 0.1,
                "sparsity": 0.5,
            },
            {
                "model_label": "model_a",
                "module_name": "model.layers.0.mlp.experts.0.up_proj",
                "projection": "up_proj",
                "zero_col_ratio": 0.3,
                "zero_row_ratio": 0.2,
                "sparsity": 0.6,
            },
            {
                "model_label": "model_a",
                "module_name": "model.layers.0.mlp.experts.0.down_proj",
                "projection": "down_proj",
                "zero_col_ratio": 0.2,
                "zero_row_ratio": 0.8,
                "sparsity": 0.7,
            },
            {
                "model_label": "model_a",
                "module_name": "model.layers.1.mlp.experts.1.down_proj",
                "projection": "down_proj",
                "zero_col_ratio": 0.9,
                "zero_row_ratio": 0.1,
                "sparsity": 0.8,
            },
        ]

        selected = select_pattern_samples(rows, max_pattern_plots=3)

        self.assertEqual(len(selected), 3)
        self.assertEqual({row["projection"] for row in selected}, {"gate_proj", "up_proj", "down_proj"})


if __name__ == "__main__":
    unittest.main()
