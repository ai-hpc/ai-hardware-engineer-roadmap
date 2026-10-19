import json
from pathlib import Path
import unittest

from budget import FORMATS, kv_bytes, linear_intensity, matvec, projection_shapes, quantized_matrix_bytes, report


class BudgetTests(unittest.TestCase):
    def setUp(self):
        self.config = json.loads(Path(__file__).with_name("qwen3-config.json").read_text())

    def test_projection_dimensions_are_not_assumed_hidden_size(self):
        shapes = projection_shapes(self.config)
        self.assertEqual(shapes["q"], (4096, 2560))
        self.assertEqual(shapes["k"], (1024, 2560))
        self.assertEqual(shapes["o"], (2560, 4096))
        self.assertEqual(shapes["down"], (2560, 9728))

    def test_fused_projection_preserves_results(self):
        q, k, v, x = [[1, 2], [3, 4]], [[5, 6]], [[7, 8]], [2, -1]
        separate = matvec(q, x) + matvec(k, x) + matvec(v, x)
        fused = matvec(q + k + v, x)
        self.assertEqual(fused, separate)
        self.assertEqual(fused, [0, 2, 4, 6])

    def test_tied_embeddings_still_require_full_lm_head(self):
        table, hidden = [[1, 0], [0, 1], [2, -1]], [2, 3]
        self.assertEqual(table[1], [0, 1])  # input is one row lookup
        self.assertEqual(matvec(table, hidden), [2, 3, 1])  # output reads all rows
        result = report(self.config)
        self.assertEqual(result["unique_parameters_tied"], 4022468096)
        self.assertEqual(result["unique_parameters_if_untied"] - result["unique_parameters_tied"],
                         result["lm_head_macs_per_decode_step"])

    def test_quantized_storage_counts_metadata(self):
        self.assertEqual(FORMATS["Q4_0"][0], 2)
        self.assertEqual(quantized_matrix_bytes(1, 256, "Q4_K"), 144)
        self.assertEqual(quantized_matrix_bytes(1, 256, "Q6_K"), 210)
        self.assertEqual(8 * 210 / 256, 6.5625)
        with self.assertRaises(ValueError):
            quantized_matrix_bytes(2, 257, "Q4_K")

    def test_kv_units_batch_and_context(self):
        self.assertEqual(kv_bytes(self.config, 4096), 603979776)
        self.assertEqual(kv_bytes(self.config, 4096, batch=2), 2 * kv_bytes(self.config, 4096))
        self.assertEqual(kv_bytes(self.config, 8192, bytes_per_element=1), kv_bytes(self.config, 4096))

    def test_arithmetic_intensity_has_precision_and_batch(self):
        fp16 = linear_intensity(4096, 2560, 1)
        int4 = linear_intensity(4096, 2560, 1, weight_bytes=.5)
        self.assertLess(fp16, 1)
        self.assertGreater(int4, 3.9)
        self.assertGreater(linear_intensity(4096, 2560, 32), 30 * fp16)

    def test_cache_lowers_optimistic_bound(self):
        result = report(self.config)
        self.assertLess(result["optimistic_weights_plus_full_kv_read_tokens_per_s"],
                        result["optimistic_weight_only_tokens_per_s"])
        self.assertEqual(result["kv_fp16_max_config_context_bytes"], 36 * 1024 ** 3)

    def test_invalid_shapes_fail(self):
        for bad in (0, -1, True, 2.5):
            with self.assertRaises(ValueError):
                kv_bytes(self.config, bad)
        with self.assertRaises(ValueError):
            projection_shapes(dict(self.config, num_key_value_heads=7))
        with self.assertRaises(ValueError):
            matvec([[1, 2]], [1])


if __name__ == "__main__":
    unittest.main()
