from main import build_parser as build_main_parser
from main_dsv2 import build_parser as build_dsv2_parser
import unittest


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
