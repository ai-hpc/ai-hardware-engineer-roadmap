import math
import unittest

from quality_metrics import aggregate, forward_kl, log_softmax, masked_nll, sliding_targets


class QualityMetricsTests(unittest.TestCase):
    def test_stability_and_shift_invariance(self):
        self.assertAlmostEqual(log_softmax([10000, 10000])[0], -math.log(2))
        for a, b in zip(log_softmax([0, 2]), log_softmax([1000, 1002])):
            self.assertAlmostEqual(a, b)

    def test_mask_and_weighted_aggregation(self):
        total, count = masked_nll([[0, 0], [10000, -10000], [0, 0]], [0, None, 1])
        self.assertEqual(count, 2)
        self.assertAlmostEqual(total, 2 * math.log(2))
        self.assertEqual(aggregate([(2, 2), (4, 1)])["mean_nll_nats"], 2)
        self.assertNotEqual(aggregate([(2, 2), (4, 1)])["mean_nll_nats"], (1 + 4) / 2)

    def test_sliding_windows_score_every_target_once(self):
        self.assertEqual([len(x) for x in sliding_targets(10, 6, 3)], [5, 3, 1])
        for length in range(2, 24):
            for window in range(2, 10):
                for stride in range(1, window):
                    self.assertEqual(sum(sliding_targets(length, window, stride), []), list(range(1, length)))

    def test_data_nll_can_improve_while_teacher_kl_is_positive(self):
        p, q = [0.8, 0.2], [0.9, 0.1]
        self.assertGreater(forward_kl(p, q), 0)
        self.assertLess(-math.log(q[0]) + math.log(p[0]), 0)
        teacher_ce_delta = sum(a * (-math.log(b) + math.log(a)) for a, b in zip(p, q))
        self.assertAlmostEqual(teacher_ce_delta, forward_kl(p, q))

    def test_kl_support_and_identity(self):
        self.assertEqual(forward_kl([0, 1], [0, 1]), 0)
        self.assertEqual(forward_kl([0.5, 0.5], [0, 1]), math.inf)

    def test_ppl_is_not_bounded_by_vocabulary_size(self):
        summary = aggregate([masked_nll([[math.log(.001), math.log(.999)]], [0])])
        self.assertAlmostEqual(summary["perplexity"], 1000)

    def test_invalid_contracts_fail(self):
        for p, q in [([.5], [.5]), ([1], [1, 0]), ([math.nan], [1]), ([-1, 2], [.5, .5])]:
            with self.assertRaises(ValueError):
                forward_kl(p, q)
        for parts in [[], [(1, 0)], [(-1, 1)], [(math.inf, 1)]]:
            with self.assertRaises(ValueError):
                aggregate(parts)
        with self.assertRaises(ValueError):
            masked_nll([[0, 1]], [2])
        with self.assertRaises(ValueError):
            sliding_targets(10, 4, 4)


if __name__ == "__main__":
    unittest.main()
