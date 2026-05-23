import unittest

from debug.analyze_pruned_moe import (
    aggregate_projection_overlap_rows,
    aggregate_expert_projection_rows,
    aggregate_layer_depth_projection_rows,
    aggregate_layer_projection_rows,
    build_parser,
    compute_mask_overlap_metrics,
    compute_significant_zero_ratios,
    format_ascii_table,
    parse_expert_module_name,
    select_weight_overview_rows,
)


class AnalyzePrunedMoeTests(unittest.TestCase):
    def test_parser_accepts_multiple_models_and_threshold(self):
        parser = build_parser()

        args = parser.parse_args(
            [
                "--reference-model",
                "/tmp/reference_model",
                "--model",
                "/tmp/model_a",
                "--model",
                "/tmp/model_b",
                "--output-dir",
                "/tmp/out",
            ]
        )

        self.assertEqual(args.reference_model, "/tmp/reference_model")
        self.assertEqual(args.model, ["/tmp/model_a", "/tmp/model_b"])
        self.assertEqual(args.output_dir, "/tmp/out")
        self.assertAlmostEqual(args.significant_threshold, 0.8)

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

    def test_compute_significant_zero_ratios(self):
        weight = [
            [0.0, 1.0, 0.0, 1.0, 1.0],
            [0.0, 0.0, 0.0, 1.0, 1.0],
            [0.0, 0.0, 0.0, 1.0, 1.0],
            [0.0, 0.0, 0.0, 0.0, 0.0],
            [1.0, 0.0, 1.0, 1.0, 1.0],
        ]

        stats = compute_significant_zero_ratios(weight, significant_threshold=0.8)

        self.assertAlmostEqual(stats["sparsity"], 14 / 25)
        self.assertAlmostEqual(stats["significant_zero_col_ratio"], 3 / 5)
        self.assertAlmostEqual(stats["significant_zero_row_ratio"], 1 / 5)
        self.assertAlmostEqual(stats["max_col_zero_fraction"], 4 / 5)
        self.assertAlmostEqual(stats["max_row_zero_fraction"], 1.0)

    def test_compute_significant_zero_ratios_respects_exact_threshold_boundary(self):
        weight = [
            [0.0, 0.0, 1.0, 1.0, 1.0],
            [0.0, 0.0, 1.0, 1.0, 1.0],
            [0.0, 0.0, 1.0, 1.0, 1.0],
            [0.0, 0.0, 1.0, 1.0, 1.0],
            [1.0, 1.0, 1.0, 1.0, 1.0],
        ]

        stats = compute_significant_zero_ratios(weight, significant_threshold=0.8)

        self.assertAlmostEqual(stats["significant_zero_col_ratio"], 2 / 5)
        self.assertAlmostEqual(stats["significant_zero_row_ratio"], 0.0)
        self.assertAlmostEqual(stats["max_col_zero_fraction"], 0.8)

    def test_compute_mask_overlap_metrics(self):
        reference_weight = [
            [0.0, 1.0, 0.0],
            [1.0, 0.0, 1.0],
        ]
        candidate_weight = [
            [0.0, 1.0, 1.0],
            [1.0, 0.0, 0.0],
        ]

        metrics = compute_mask_overlap_metrics(reference_weight, candidate_weight)

        self.assertAlmostEqual(metrics["reference_zero_ratio"], 0.5)
        self.assertAlmostEqual(metrics["candidate_zero_ratio"], 0.5)
        self.assertAlmostEqual(metrics["mask_match_ratio"], 4 / 6)
        self.assertAlmostEqual(metrics["pruned_overlap_ratio"], 2 / 6)
        self.assertAlmostEqual(metrics["pruned_jaccard"], 2 / 4)

    def test_aggregate_projection_overlap_rows(self):
        rows = [
            {
                "model_label": "new_model",
                "reference_label": "old_model",
                "projection": "down_proj",
                "mask_match_ratio": 0.8,
                "pruned_overlap_ratio": 0.4,
                "pruned_jaccard": 0.5,
            },
            {
                "model_label": "new_model",
                "reference_label": "old_model",
                "projection": "down_proj",
                "mask_match_ratio": 0.6,
                "pruned_overlap_ratio": 0.2,
                "pruned_jaccard": 0.25,
            },
        ]

        aggregated = aggregate_projection_overlap_rows(rows)

        self.assertEqual(len(aggregated), 1)
        self.assertEqual(aggregated[0]["module_count"], 2)
        self.assertAlmostEqual(aggregated[0]["mean_mask_match_ratio"], 0.7)
        self.assertAlmostEqual(aggregated[0]["mean_pruned_overlap_ratio"], 0.3)
        self.assertAlmostEqual(aggregated[0]["mean_pruned_jaccard"], 0.375)

    def test_aggregate_layer_projection_rows_uses_significant_metrics(self):
        rows = [
            {
                "model_label": "model_a",
                "layer": 0,
                "expert": 0,
                "projection": "down_proj",
                "significant_zero_col_ratio": 0.25,
                "significant_zero_row_ratio": 0.10,
            },
            {
                "model_label": "model_a",
                "layer": 0,
                "expert": 1,
                "projection": "down_proj",
                "significant_zero_col_ratio": 0.75,
                "significant_zero_row_ratio": 0.50,
            },
        ]

        aggregated = aggregate_layer_projection_rows(rows)

        self.assertEqual(len(aggregated), 1)
        self.assertEqual(aggregated[0]["expert_count"], 2)
        self.assertAlmostEqual(aggregated[0]["mean_significant_zero_col_ratio"], 0.5)
        self.assertAlmostEqual(aggregated[0]["max_significant_zero_row_ratio"], 0.5)

    def test_aggregate_layer_depth_projection_rows_splits_shallow_mid_deep(self):
        rows = [
            {"model_label": "model_a", "layer": 0, "expert": 0, "projection": "gate_proj", "significant_zero_col_ratio": 0.1},
            {"model_label": "model_a", "layer": 1, "expert": 0, "projection": "gate_proj", "significant_zero_col_ratio": 0.2},
            {"model_label": "model_a", "layer": 2, "expert": 0, "projection": "gate_proj", "significant_zero_col_ratio": 0.3},
            {"model_label": "model_a", "layer": 3, "expert": 0, "projection": "gate_proj", "significant_zero_col_ratio": 0.4},
            {"model_label": "model_a", "layer": 4, "expert": 0, "projection": "gate_proj", "significant_zero_col_ratio": 0.5},
            {"model_label": "model_a", "layer": 5, "expert": 0, "projection": "gate_proj", "significant_zero_col_ratio": 0.6},
        ]

        aggregated = aggregate_layer_depth_projection_rows(rows)

        self.assertEqual([row["depth_bucket"] for row in aggregated], ["shallow", "mid", "deep"])
        self.assertEqual([row["module_count"] for row in aggregated], [2, 2, 2])
        self.assertAlmostEqual(aggregated[0]["mean_significant_zero_col_ratio"], 0.15)
        self.assertAlmostEqual(aggregated[1]["mean_significant_zero_col_ratio"], 0.35)
        self.assertAlmostEqual(aggregated[2]["mean_significant_zero_col_ratio"], 0.55)

    def test_aggregate_expert_projection_rows_computes_significant_metrics(self):
        rows = [
            {
                "model_label": "model_a",
                "layer": 0,
                "expert": 1,
                "projection": "up_proj",
                "significant_zero_col_ratio": 0.0,
                "significant_zero_row_ratio": 0.0,
            },
            {
                "model_label": "model_a",
                "layer": 1,
                "expert": 1,
                "projection": "up_proj",
                "significant_zero_col_ratio": 0.5,
                "significant_zero_row_ratio": 0.25,
            },
            {
                "model_label": "model_a",
                "layer": 2,
                "expert": 1,
                "projection": "up_proj",
                "significant_zero_col_ratio": 0.25,
                "significant_zero_row_ratio": 0.5,
            },
        ]

        aggregated = aggregate_expert_projection_rows(rows)

        self.assertEqual(len(aggregated), 1)
        self.assertEqual(aggregated[0]["layer_count"], 3)
        self.assertAlmostEqual(aggregated[0]["mean_significant_zero_col_ratio"], 0.25)
        self.assertAlmostEqual(aggregated[0]["max_significant_zero_col_ratio"], 0.5)
        self.assertAlmostEqual(aggregated[0]["significant_col_collapse_incidence"], 2 / 3)
        self.assertAlmostEqual(aggregated[0]["mean_significant_zero_row_ratio"], 0.25)
        self.assertAlmostEqual(aggregated[0]["significant_row_collapse_incidence"], 2 / 3)

    def test_format_ascii_table(self):
        table = format_ascii_table(
            rows=[
                {"projection": "gate_proj", "mean": 0.125, "max": 0.5},
                {"projection": "down_proj", "mean": 0.250, "max": 0.75},
            ],
            columns=[
                ("projection", "projection"),
                ("mean", "mean"),
                ("max", "max"),
            ],
            float_precision=3,
        )

        self.assertIn("| projection | mean  | max   |", table)
        self.assertIn("| gate_proj  | 0.125 | 0.500 |", table)
        self.assertIn("| down_proj  | 0.250 | 0.750 |", table)

    def test_select_weight_overview_rows_prefers_highest_significant_zero_col_ratio(self):
        rows = [
            {"layer": 0, "expert": 1, "projection": "gate_proj", "significant_zero_col_ratio": 0.2},
            {"layer": 1, "expert": 2, "projection": "gate_proj", "significant_zero_col_ratio": 0.7},
            {"layer": 2, "expert": 3, "projection": "gate_proj", "significant_zero_col_ratio": 0.3},
            {"layer": 3, "expert": 4, "projection": "up_proj", "significant_zero_col_ratio": 0.4},
            {"layer": 4, "expert": 5, "projection": "up_proj", "significant_zero_col_ratio": 0.9},
            {"layer": 5, "expert": 6, "projection": "down_proj", "significant_zero_col_ratio": 0.8},
        ]

        selected = select_weight_overview_rows(rows)

        self.assertEqual(selected[("shallow", "gate_proj")]["expert"], 2)
        self.assertEqual(selected[("deep", "up_proj")]["expert"], 5)
        self.assertEqual(selected[("deep", "down_proj")]["expert"], 6)


if __name__ == "__main__":
    unittest.main()
