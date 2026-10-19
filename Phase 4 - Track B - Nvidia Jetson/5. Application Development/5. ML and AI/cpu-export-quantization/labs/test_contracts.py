import unittest
from contracts import affine_quantize, kv_bytes, chunk_deadline_seconds


class ContractTests(unittest.TestCase):
    def test_rounding_clipping_and_zero(self):
        self.assertEqual(affine_quantize(0, .1), (0, 0))
        self.assertEqual(affine_quantize(20, .1), (127, 12.700000000000001))
        self.assertEqual(affine_quantize(.25, .1)[0], 2)  # ties-to-even
        for x in (-1.27, -.341, 0, .67, 1.27):
            self.assertLessEqual(abs(affine_quantize(x, .01)[1] - x), .005000001)
        with self.assertRaises(ValueError):
            affine_quantize(1, 0)

    def test_kv_counts_keys_and_values_and_heads(self):
        self.assertEqual(kv_bytes(32, 8, 128, 4096), 536870912)
        self.assertEqual(kv_bytes(32, 8, 128, 4096, sequences=2), 1073741824)
        with self.assertRaises(ValueError):
            kv_bytes(32, 0, 128, 4096)

    def test_remaining_actions_not_full_prediction_horizon(self):
        self.assertAlmostEqual(chunk_deadline_seconds(10, 30, .05), .2833333333333333)
        self.assertEqual(chunk_deadline_seconds(0, 30, .05), 0)


if __name__ == '__main__':
    unittest.main()
