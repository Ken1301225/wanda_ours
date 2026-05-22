from main import build_parser as build_main_parser
from main import prepare_run_outputs
from main_dsv2 import build_parser as build_dsv2_parser
from types import SimpleNamespace
import tempfile
import unittest
import sys
from unittest.mock import patch


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

    def test_prepare_run_outputs_writes_run_start_without_global_torch_import(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            args = SimpleNamespace(
                save=tmpdir,
                model="dummy",
                prune_method="moe_wanda",
                seed=0,
                nsamples=128,
                sparsity_ratio=0.5,
                sparsity_type="unstructured",
                use_variant=False,
                save_model=None,
            )

            torch_stub = SimpleNamespace(cuda=SimpleNamespace(device_count=lambda: 0))

            with patch("main.version", return_value="test-version"), patch.dict(sys.modules, {"torch": torch_stub}):
                prepare_run_outputs(args)

            with open(args.diagnostics_path, "r") as f:
                content = f.read()

            self.assertIn('"event": "run_start"', content)
