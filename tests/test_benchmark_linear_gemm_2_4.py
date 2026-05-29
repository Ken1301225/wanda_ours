import unittest
from pathlib import Path

from scripts.benchmark_linear_gemm_2_4 import (
    classify_projection,
    format_markdown_table,
    parse_int_list,
    build_parser,
)


class BenchmarkLinearGemm24Tests(unittest.TestCase):
    def test_parse_int_list(self):
        self.assertEqual(parse_int_list("1,8,64"), [1, 8, 64])

    def test_parser_accepts_paper_style_options(self):
        parser = build_parser()

        args = parser.parse_args(
            [
                "--dense-model",
                "/tmp/dense",
                "--pruned-model",
                "/tmp/pruned",
                "--tokens",
                "1",
                "--backend",
                "cusparselt",
            ]
        )

        self.assertEqual(args.dense_model, "/tmp/dense")
        self.assertEqual(args.pruned_model, "/tmp/pruned")
        self.assertEqual(parse_int_list(args.tokens), [1])
        self.assertEqual(args.backend, "cusparselt")

    def test_classify_projection_keeps_only_moe_expert_linears(self):
        self.assertEqual(classify_projection("model.layers.0.mlp.experts.0.up_proj"), "up/gate_proj")
        self.assertEqual(classify_projection("model.layers.0.mlp.experts.0.gate_proj"), "up/gate_proj")
        self.assertEqual(classify_projection("model.layers.0.mlp.experts.0.down_proj"), "down_proj")
        self.assertIsNone(classify_projection("model.layers.0.self_attn.q_proj"))
        self.assertIsNone(classify_projection("model.layers.0.mlp.shared_expert.up_proj"))

    def test_format_markdown_table_matches_paper_style_columns(self):
        table = format_markdown_table(
            {
                "up/gate_proj": {"dense_ms": 9.82, "sparse_ms": 6.10, "speedup": 1.61},
                "down_proj": {"dense_ms": 9.92, "sparse_ms": 6.45, "speedup": 1.54},
            }
        )

        self.assertIn("| MoE Layer | Dense | 2:4 | Speedup |", table)
        self.assertIn("| up/gate_proj | 9.82 | 6.10 | 1.61x |", table)
        self.assertIn("| down_proj | 9.92 | 6.45 | 1.54x |", table)

    def test_example_script_runs_linear_gemm_benchmark_only(self):
        script = Path("scripts/run_benchmark_linear_gemm_2_4.sh").read_text()

        self.assertIn("scripts/benchmark_linear_gemm_2_4.py", script)
        self.assertIn("DENSE_MODEL", script)
        self.assertIn("PRUNED_MODEL", script)
        self.assertNotIn("benchmark_model_inference.py", script)


if __name__ == "__main__":
    unittest.main()
