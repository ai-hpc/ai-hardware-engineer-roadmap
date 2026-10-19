import unittest
from accounting import bandwidth, validate_signatures


class AccountingTest(unittest.TestCase):
    def test_eight_rank_hand_calculation(self):
        self.assertEqual(bandwidth(1_000_000_000, .004, 8),
                         {"algbw_GBps": 250.0, "busbw_GBps": 437.5})

    def test_gather_uses_total_output_size(self):
        # Two ranks, each contributes 8 bytes: total output is 16 bytes.
        self.assertAlmostEqual(bandwidth(16, 1e-6, 2, "all_gather")["busbw_GBps"], .008)
        self.assertEqual(bandwidth(16, 1e-6, 1)["busbw_GBps"], 0)

    def test_invalid_units_and_rank_contracts(self):
        for args in [(0, 1, 2), (1, 0, 2), (1, float("nan"), 2), (1, 1, 0)]:
            with self.assertRaises(ValueError): bandwidth(*args)
        with self.assertRaises(ValueError):
            validate_signatures([([2], "float64", "sum"), ([3], "float64", "sum")])
        self.assertTrue(validate_signatures([([2], "float64", "sum")] * 2))


if __name__ == "__main__": unittest.main()
