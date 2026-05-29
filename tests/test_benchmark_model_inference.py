import unittest
from pathlib import Path

from scripts.benchmark_model_inference import build_parser, parse_int_list
from scripts.benchmark_projection_table import (
    ProjectionTimer,
    build_parser as build_table_parser,
    format_markdown_table,
)


class BenchmarkModelInferenceTests(unittest.TestCase):
    def test_parse_int_list(self):
        self.assertEqual(parse_int_list("1,16,128"), [1, 16, 128])

    def test_parser_accepts_model_benchmark_options(self):
        parser = build_parser()

        args = parser.parse_args(
            [
                "--model",
                "/tmp/model",
                "--batch-size",
                "2",
                "--prompt-lengths",
                "128,512",
                "--decode-steps",
                "16",
                "--semi-structured-sparse",
                "--sparse-scope",
                "moe_experts",
                "--sparse-backend",
                "cusparselt",
                "--cusparselt-alg-id",
                "1",
            ]
        )

        self.assertEqual(args.model, "/tmp/model")
        self.assertEqual(args.batch_size, 2)
        self.assertEqual(parse_int_list(args.prompt_lengths), [128, 512])
        self.assertEqual(args.decode_steps, 16)
        self.assertTrue(args.semi_structured_sparse)
        self.assertEqual(args.sparse_scope, "moe_experts")
        self.assertEqual(args.sparse_backend, "cusparselt")
        self.assertEqual(args.cusparselt_alg_id, 1)

    def test_example_script_compares_dense_mask_and_sparse_kernel(self):
        script = Path("scripts/run_benchmark_model_inference.sh").read_text()

        self.assertIn("BASE_MODEL", script)
        self.assertIn("PRUNED_MODEL", script)
        self.assertIn("dense_baseline.json", script)
        self.assertIn("pruned_2_4_mask_dense.json", script)
        self.assertIn("pruned_2_4_sparse_kernel.json", script)
        self.assertIn("--semi-structured-sparse", script)

    def test_projection_table_parser_accepts_dense_and_pruned_models(self):
        parser = build_table_parser()

        args = parser.parse_args(
            [
                "--dense-model",
                "/tmp/dense",
                "--pruned-model",
                "/tmp/pruned",
                "--semi-structured-sparse",
                "--sparse-backend",
                "cusparselt",
            ]
        )

        self.assertEqual(args.dense_model, "/tmp/dense")
        self.assertEqual(args.pruned_model, "/tmp/pruned")
        self.assertTrue(args.semi_structured_sparse)
        self.assertEqual(args.sparse_backend, "cusparselt")

    def test_projection_timer_groups_expected_module_names(self):
        self.assertIsNone(ProjectionTimer.group_for_name("model.layers.0.self_attn.q_proj"))
        self.assertIsNone(ProjectionTimer.group_for_name("model.layers.0.self_attn.o_proj"))
        self.assertEqual(ProjectionTimer.group_for_name("model.layers.0.mlp.experts.0.up_proj"), "up/gate_proj")
        self.assertEqual(ProjectionTimer.group_for_name("model.layers.0.mlp.experts.0.gate_proj"), "up/gate_proj")
        self.assertEqual(ProjectionTimer.group_for_name("model.layers.0.mlp.experts.0.down_proj"), "down_proj")
        self.assertIsNone(ProjectionTimer.group_for_name("model.layers.0.mlp.shared_expert.up_proj"))
        self.assertIsNone(ProjectionTimer.group_for_name("model.embed_tokens"))

    def test_format_markdown_table_matches_requested_columns(self):
        table = format_markdown_table(
            {
                "up/gate_proj": {"dense_ms": 9.82, "pruned_ms": 6.10, "speedup": 1.61},
                "down_proj": {"dense_ms": 9.92, "pruned_ms": 6.45, "speedup": 1.54},
            }
        )

        self.assertIn("| LLaMA Layer | Dense | 2:4 | Speedup |", table)
        self.assertNotIn("q/k/v/o_proj", table)
        self.assertIn("| up/gate_proj | 9.82 | 6.10 | 1.61x |", table)

    def test_projection_table_example_script_loads_existing_checkpoints(self):
        script = Path("scripts/run_benchmark_projection_table.sh").read_text()

        self.assertIn("DENSE_MODEL", script)
        self.assertIn("PRUNED_MODEL", script)
        self.assertIn("scripts/benchmark_projection_table.py", script)
        self.assertIn("SPARSE_BACKEND", script)
        self.assertNotIn("python main.py", script)


if __name__ == "__main__":
    unittest.main()
