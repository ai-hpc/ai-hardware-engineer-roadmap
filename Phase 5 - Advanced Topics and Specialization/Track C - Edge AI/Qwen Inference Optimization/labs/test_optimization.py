import math
import unittest

from optimization import (column_major_gemm, dequantize_rows, matvec, quantize_rows,
                          required_replicas, rescale_equivalent, rope, row_parallel,
                          speculative_expected_tokens, speculative_monte_carlo, speculative_rate)


class OptimizationTests(unittest.TestCase):
    def test_rope_preserves_norm_at_long_position(self):
        x = [1, 2, 3, 4, 5, 6, 7, 8]
        self.assertAlmostEqual(sum(a*a for a in x), sum(a*a for a in rope(x, 200000, 5e6)), places=10)
        self.assertEqual(rope(x, 0), x)

    def test_rope_relative_position_identity(self):
        q, k, p, t = [1, 2, -3, 4], [2, -1, 4, 3], 13, 29
        a = sum(x*y for x, y in zip(rope(q, p), rope(k, t)))
        b = sum(x*y for x, y in zip(q, rope(k, t-p)))
        self.assertAlmostEqual(a, b, places=11)

    def test_rope_layout_error_is_visible_at_nonzero_position(self):
        x, angle = [1., 2., 3., 4.], 1.
        wrong_first = x[0]*math.cos(angle)-x[1]*math.sin(angle)
        self.assertGreater(abs(rope(x, 1)[0]-wrong_first), .5)

    def test_quantizer_zero_rows_and_error_bound(self):
        weight = [[0, 0, 0], [-1, .2, 2]]
        q, scales = quantize_rows(weight)
        actual = dequantize_rows(q, scales)
        self.assertEqual(actual[0], weight[0])
        self.assertLessEqual(max(abs(a-b) for a,b in zip(actual[1], weight[1])), scales[1]/2 + 1e-12)

    def test_scale_corruption_changes_output(self):
        w = [[-1, 2], [-.01, .02]]
        q, s = quantize_rows(w)
        correct, bad = matvec(dequantize_rows(q, s), [1, 1]), matvec(dequantize_rows(q, s[::-1]), [1, 1])
        self.assertGreater(max(abs(a-b) for a,b in zip(correct,bad)), .9 * max(map(abs, correct)))
        self.assertAlmostEqual(bad[0], correct[0] / 100)

    def test_channel_scaling_preserves_float_operation(self):
        w, x, scales = [[1, -2, 3], [4, 5, -6]], [2, 3, 4], [.5, 10, 2]
        self.assertEqual(matvec(w, x), rescale_equivalent(w, x, scales))

    def test_tensor_parallel_sums_not_averages(self):
        w, x = [[1, 2, 3, 4], [2, 1, 0, -1]], [1, 2, -1, 3]
        combined, partials = row_parallel(w, x, 2)
        self.assertEqual(combined, matvec(w, x))
        self.assertNotEqual([value / 2 for value in combined], matvec(w, x))
        self.assertNotEqual(partials[0], combined)

    def test_speculation_extremes(self):
        self.assertEqual(speculative_expected_tokens(0, 4), 1)
        self.assertEqual(speculative_expected_tokens(1, 4), 5)
        self.assertEqual(speculative_expected_tokens(.4, 0), 1)

    def test_speculation_counts_accepted_prefix_not_independent_total(self):
        expected = speculative_expected_tokens(.55, 5)
        self.assertAlmostEqual(expected, 2.1607096875)
        self.assertLess(abs(speculative_monte_carlo(.55, 5)-expected), .02)
        self.assertNotAlmostEqual(expected, 1+5*.55)

    def test_speculation_includes_draft_overhead(self):
        self.assertLess(speculative_rate(.2, 5, 5, 32.5), 1000/30)
        self.assertGreater(speculative_rate(.9, 5, 1, 32.5), 1000/30)

    def test_capacity_headroom_changes_replica_count(self):
        self.assertEqual(required_replicas(347.222222, 400, .7), 2)
        self.assertEqual(required_replicas(347.222222, 400, 1), 1)
        with self.assertRaises(ValueError):
            required_replicas(10, 0)

    def test_non_square_padded_column_major_gemm(self):
        # A=[[1,2,3],[4,5,6]], B=[[7,8],[9,10],[11,12]], leading dims 3,4,3.
        a = [1,4,999,2,5,999,3,6,999]
        b = [7,9,11,999,8,10,12,999]
        c = [float('nan'),float('nan'),123,float('nan'),float('nan'),123]
        out = column_major_gemm(a,b,c,2,2,3,3,4,3)
        self.assertEqual(out,[58,139,123,64,154,123])
        # Incorrect leading dimensions must not quietly match the oracle.
        self.assertNotEqual(column_major_gemm(a,b,c,2,2,3,2,3,3),out)

    def test_beta_requires_initialized_input(self):
        out=column_major_gemm([2],[3],[5],1,1,1,1,1,1,alpha=2,beta=3)
        self.assertEqual(out,[27])

    def test_invalid_contracts(self):
        with self.assertRaises(ValueError): rope([1,2,3],0)
        with self.assertRaises(ValueError): row_parallel([[1,2,3]],[1,2,3],2)
        with self.assertRaises(ValueError): speculative_expected_tokens(.4,-1)
        with self.assertRaises(ValueError): quantize_rows([[math.nan]])
        with self.assertRaises(ValueError): column_major_gemm([1],[2],[0],2,1,1,2,1,2)


if __name__ == "__main__":
    unittest.main()
