import unittest

from scripts.benchmark_24_expert import build_parser, parse_token_counts


class Benchmark24ExpertTests(unittest.TestCase):
    def test_parse_token_counts(self):
        self.assertEqual(parse_token_counts("1,4,16"), [1, 4, 16])

    def test_parser_defaults_to_dense_mask_and_sparse_modes(self):
        parser = build_parser()

        args = parser.parse_args([])

        self.assertEqual(args.modes, ["dense", "mask_dense", "sparse"])
        self.assertEqual(args.sparsity_type, "2:4")
        self.assertEqual(args.token_counts, "1,4,8,16,32,64,128,256")

    def test_parser_accepts_moe_like_shapes(self):
        parser = build_parser()

        args = parser.parse_args(
            [
                "--hidden-size",
                "2048",
                "--intermediate-size",
                "1408",
                "--token-counts",
                "8,32",
                "--modes",
                "dense",
                "sparse",
            ]
        )

        self.assertEqual(args.hidden_size, 2048)
        self.assertEqual(args.intermediate_size, 1408)
        self.assertEqual(parse_token_counts(args.token_counts), [8, 32])
        self.assertEqual(args.modes, ["dense", "sparse"])


if __name__ == "__main__":
    unittest.main()
