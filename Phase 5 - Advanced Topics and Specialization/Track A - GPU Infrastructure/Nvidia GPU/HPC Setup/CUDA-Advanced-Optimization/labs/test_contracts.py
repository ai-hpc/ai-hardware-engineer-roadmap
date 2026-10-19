import math
import unittest
from contracts import claimed_tiles, faulty_padded_layernorm, layernorm, pointwise, traffic


class NumericalContracts(unittest.TestCase):
    def test_hand_layernorm_and_padding(self):
        # Mean2, population variance2/3, epsilon1/3 -> denominator1.
        for block in [3, 4, 8, 32]:
            self.assertEqual(layernorm([1., 2., 3.], block, epsilon=1/3), [-1., 0., 1.])

    def test_padding_regression_detects_old_error(self):
        correct = layernorm([1., 2., 4.], 4)
        faulty = faulty_padded_layernorm([1., 2., 4.], 4)
        self.assertGreater(max(abs(a-b) for a,b in zip(correct, faulty)), .1)

    def test_constant_and_singleton_rows(self):
        self.assertEqual(layernorm([7.], 32), [0.])
        self.assertEqual(layernorm([7.] * 5, 8), [0.] * 5)

    def test_normalization_validation(self):
        for row, block, epsilon in [([], 4, 1e-5), ([1., math.nan], 4, 1e-5),
                                    ([1., 2.], 1, 1e-5), ([1.], 1, 0.),
                                    ([1.], 1, math.inf), ([1.], 1.5, 1e-5)]:
            with self.assertRaises(ValueError): layernorm(row, block, epsilon)

    def test_fusion_preserves_fixed_dropout_decisions(self):
        a, b = pointwise([-3., 0., 2., 4.], [1., 2., 3., 4.], [True, False, False, True], .25)
        self.assertEqual(a, b)
        self.assertEqual(a[1:3], [2., 3.])
        self.assertAlmostEqual(a[3], .5 * 4 * (1 + math.erf(4 / math.sqrt(2))) / .75 + 4)

    def test_dropout_validation(self):
        for probability in [-1., 1., math.nan]:
            with self.assertRaises(ValueError): pointwise([1.], [0.], [True], probability)
        with self.assertRaises(ValueError): pointwise([1.], [], [True], .1)
        with self.assertRaises(ValueError): pointwise([1.], [0.], [1], .1)

    def test_traffic_counts_include_residual(self):
        self.assertEqual(traffic(1000, 4), {"separate_bytes": 28000, "fused_bytes": 12000})
        self.assertEqual(traffic(0), {"separate_bytes": 0, "fused_bytes": 0})
        for n in [-1, 1.5, True]:
            with self.assertRaises(ValueError): traffic(n)

    def test_tiles_cover_each_element_once(self):
        for n in [0, 1, 31, 32, 33, 255, 256, 257, 1003]:
            for workers in [1, 2, 3, 32]:
                flattened = [v for group in claimed_tiles(n, 256, workers) for v in group]
                self.assertEqual(sorted(flattened), list(range(n)))
                self.assertEqual(len(flattened), len(set(flattened)))

    def test_tile_validation(self):
        for args in [(-1, 32, 1), (1, 0, 1), (1, 32, 0), (1, 2.5, 1)]:
            with self.assertRaises(ValueError): claimed_tiles(*args)


if __name__ == "__main__":
    unittest.main()
