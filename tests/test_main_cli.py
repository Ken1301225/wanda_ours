import json
import tempfile
import unittest
from pathlib import Path

from main import build_parser as build_main_parser, prepare_run_outputs
from main_dsv2 import build_parser as build_dsv2_parser


class MainCliTests(unittest.TestCase):
    def test_parser_accepts_moe_wanda_only(self):
        for build_parser in (build_main_parser, build_dsv2_parser):
            parser = build_parser()

            args = parser.parse_args(
                [
                    "--model",
                    "dummy",
                    "--sparsity_type",
                    "unstructured",
                    "--prune_method",
                    "moe_wanda",
                ]
            )

            self.assertEqual(args.prune_method, "moe_wanda")

    def test_parser_rejects_legacy_wanda_name(self):
        for build_parser in (build_main_parser, build_dsv2_parser):
            parser = build_parser()

            with self.assertRaises(SystemExit):
                parser.parse_args(
                    [
                        "--model",
                        "dummy",
                        "--sparsity_type",
                        "unstructured",
                        "--prune_method",
                        "wanda",
                    ]
                )

    def test_parser_accepts_dense_softmax_routing_mode(self):
        for build_parser in (build_main_parser, build_dsv2_parser):
            parser = build_parser()

            args = parser.parse_args(
                [
                    "--model",
                    "dummy",
                    "--sparsity_type",
                    "unstructured",
                    "--prune_method",
                    "moe_wanda",
                    "--moe_wanda_routing_mode",
                    "dense_softmax",
                ]
            )

            self.assertEqual(args.moe_wanda_routing_mode, "dense_softmax")

    def test_parser_accepts_moe_wanda_routing_power(self):
        for build_parser in (build_main_parser, build_dsv2_parser):
            parser = build_parser()

            args = parser.parse_args(
                [
                    "--model",
                    "dummy",
                    "--sparsity_type",
                    "unstructured",
                    "--prune_method",
                    "moe_wanda",
                    "--moe_wanda_routing_power",
                    "1.0",
                ]
            )

            self.assertEqual(args.moe_wanda_routing_power, 1.0)

    def test_parser_accepts_moe_wanda_cluster_flags(self):
        for build_parser in (build_main_parser, build_dsv2_parser):
            parser = build_parser()

            args = parser.parse_args(
                [
                    "--model",
                    "dummy",
                    "--sparsity_type",
                    "unstructured",
                    "--prune_method",
                    "moe_wanda",
                    "--moe_wanda_cluster_experts",
                    "--moe_wanda_cluster_k",
                    "15",
                ]
            )

            self.assertTrue(args.moe_wanda_cluster_experts)
            self.assertEqual(args.moe_wanda_cluster_k, 15)

    def test_prepare_run_outputs_writes_routing_settings(self):
        parser = build_main_parser()

        with tempfile.TemporaryDirectory() as tmpdir:
            args = parser.parse_args(
                [
                    "--model",
                    "dummy",
                    "--sparsity_type",
                    "unstructured",
                    "--prune_method",
                    "moe_wanda",
                    "--save",
                    tmpdir,
                    "--moe_wanda_routing_mode",
                    "dense_softmax",
                    "--moe_wanda_routing_power",
                    "1.0",
                    "--moe_wanda_cluster_experts",
                    "--moe_wanda_cluster_k",
                    "15",
                ]
            )

            prepare_run_outputs(args)

            diagnostics_path = Path(tmpdir) / "diagnostics_moe_wanda.jsonl"
            summary_path = Path(tmpdir) / "diagnostics_moe_wanda.txt"
            self.assertTrue(diagnostics_path.exists())
            self.assertTrue(summary_path.exists())

            run_start = json.loads(diagnostics_path.read_text().strip())
            self.assertEqual(run_start["moe_wanda_routing_mode"], "dense_softmax")
            self.assertEqual(run_start["moe_wanda_routing_power"], 1.0)
            self.assertTrue(run_start["moe_wanda_cluster_experts"])
            self.assertEqual(run_start["moe_wanda_cluster_k"], 15)

            summary = summary_path.read_text().strip()
            self.assertIn("routing_mode=dense_softmax", summary)
            self.assertIn("routing_power=1.0", summary)
            self.assertIn("cluster_experts=True", summary)
            self.assertIn("cluster_k=15", summary)

    def test_moe_scripts_forward_routing_flags(self):
        for relpath in (
            "scripts/qwen1_5.sh",
            "scripts/qwen1_5_moe_wanda.sh",
            "scripts/dsv2.sh",
        ):
            script_text = Path(relpath).read_text()
            self.assertIn("--moe_wanda_routing_mode", script_text, msg=relpath)
            self.assertIn("--moe_wanda_routing_power", script_text, msg=relpath)
            self.assertIn("--moe_wanda_cluster_experts", script_text, msg=relpath)
            self.assertIn("--moe_wanda_cluster_k", script_text, msg=relpath)
            self.assertIn("CLUSTER_EXPERTS", script_text, msg=relpath)
            self.assertIn("CLUSTER_K", script_text, msg=relpath)

    def test_moe_scripts_use_anti_collapse_style_path_roots(self):
        expected_tokens = {
            "scripts/qwen1_5.sh": ["MODEL_ROOT", "HF_CACHE_ROOT", "RUN_ROOT", "OUTPUT_DIR", "CHECKPOINT_DIR"],
            "scripts/qwen1_5_moe_wanda.sh": ["MODEL_ROOT", "HF_CACHE_ROOT", "RUN_ROOT", "OUTPUT_DIR", "CHECKPOINT_DIR"],
            "scripts/dsv2.sh": ["MODEL_ROOT", "HF_CACHE_ROOT", "RUN_ROOT", "OUTPUT_DIR", "CHECKPOINT_DIR"],
        }

        for relpath, tokens in expected_tokens.items():
            script_text = Path(relpath).read_text()
            for token in tokens:
                self.assertIn(token, script_text, msg=f"{relpath} missing {token}")

    def test_qwen_moe_wanda_ablation_script_covers_routing_power_and_clustering(self):
        script_text = Path("scripts/ablate_moe_wanda_qwen1_5.sh").read_text()

        self.assertIn("run_case", script_text)
        self.assertIn("topk", script_text)
        self.assertIn("dense_softmax", script_text)
        self.assertIn("0.5", script_text)
        self.assertIn("1.0", script_text)
        self.assertIn("1.5", script_text)
        self.assertIn("2.0", script_text)
        self.assertIn("CLUSTER_KS", script_text)
        self.assertIn("--moe_wanda_cluster_experts", script_text)
