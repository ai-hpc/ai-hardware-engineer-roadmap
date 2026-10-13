import math
import unittest

from contracts import (assert_disjoint_groups, chunk_with_mask, denormalize,
                       paired_task_bootstrap, usable_chunk_indices)
from vla_lab import centroid_tokens, dataset, experiment, loss_gradient, predict


class GroundingTests(unittest.TestCase):
    def test_centroids_and_state_contract(self):
        sample = dataset(1, 19)[0]
        tokens = centroid_tokens(sample.image)
        expected = tuple(tokens[sample.instruction][d] - sample.state[d] for d in range(2))
        self.assertEqual(sample.target, expected)
        shifted = dataset(1, 19)[0]
        shifted.state = (sample.state[0] + 0.3, sample.state[1])
        logits = [[0.0] * 3 for _ in range(3)]
        self.assertAlmostEqual(predict(logits, sample)[0] - predict(logits, shifted)[0], 0.3)

    def test_gradient_against_finite_difference(self):
        sample = dataset(1, 9)[0]
        logits = [[0.1, -0.2, 0.4] for _ in range(3)]
        _, analytic = loss_gradient(logits, sample)
        row, epsilon = logits[sample.instruction], 1e-6
        for c in range(3):
            original = row[c]
            row[c] = original + epsilon
            plus, _ = loss_gradient(logits, sample)
            row[c] = original - epsilon
            minus, _ = loss_gradient(logits, sample)
            row[c] = original
            self.assertAlmostEqual(analytic[c], (plus - minus) / (2 * epsilon), places=7)

    def test_training_and_language_intervention(self):
        result = experiment(train_count=90, test_count=45, epochs=10)
        self.assertLess(result["test_mse_per_coordinate"], 0.02)
        self.assertLess(result["test_mse_per_coordinate"], result["uniform_attention_mse"] / 10)
        self.assertGreater(result["wrong_instruction_mse"], result["test_mse_per_coordinate"] * 10)

    def test_missing_color_rejected(self):
        with self.assertRaises(ValueError):
            centroid_tokens([[[0, 0, 0]] * 2 for _ in range(2)])

    def test_pixel_range_rejected(self):
        with self.assertRaises(ValueError):
            centroid_tokens([[[1e308] * 3] * 2 for _ in range(2)])


class ContractTests(unittest.TestCase):
    def test_action_units(self):
        self.assertEqual(denormalize([0.5, -1], [0, 1], [0.02, 0.1]), [0.01, 0.9])
        for scales in ([0, 1], [1], [math.nan, 1]):
            with self.assertRaises(ValueError):
                denormalize([0, 0], [0, 0], scales)

    def test_denormalization_overflow(self):
        with self.assertRaises(ValueError):
            denormalize([1e308], [0], [2])

    def test_episode_padding_and_mask(self):
        records = [{"episode": e, "time": t, "action": [a]}
                   for e, t, a in [("a", 0, 1), ("a", 1, 2), ("b", 0, 99)]]
        self.assertEqual(chunk_with_mask(records, 1, 3), ([[2], [2], [2]], [1, 0, 0]))
        records[1]["time"] = 0
        with self.assertRaises(ValueError):
            chunk_with_mask(records, 0, 2)

    def test_split_leakage(self):
        assert_disjoint_groups(["session1"], ["session2"])
        with self.assertRaises(ValueError):
            assert_disjoint_groups(["session1"], ["session1"])

    def test_chunk_delay_and_expiry(self):
        self.assertEqual(usable_chunk_indices(0, 0.07, 0.07, 0.02, 8, 0.2), [4, 5, 6, 7])
        self.assertEqual(usable_chunk_indices(0, 0.07, 0.07, 0.02, 8, 0.1), [4, 5])
        self.assertEqual(usable_chunk_indices(0, 0.07, 0.12, 0.02, 8, 0.1), [])
        with self.assertRaises(ValueError):
            usable_chunk_indices(1, 0.5, 2, 0.02, 8, 1)

    def test_schedule_exact_boundaries(self):
        self.assertEqual(usable_chunk_indices(0, 0.04, 0.04, 0.02, 8, 0.1), [2, 3, 4, 5])
        self.assertEqual(usable_chunk_indices(0, 0.1, 0.1, 0.02, 8, 0.1), [5])

    def test_bootstrap_pairs_and_macro_weighting(self):
        rows = [{"task": "a", "seed": i, "reference": 0, "candidate": 1} for i in range(9)]
        rows += [{"task": "b", "seed": 0, "reference": 1, "candidate": 0}]
        result = paired_task_bootstrap(rows, 100)
        self.assertEqual(result["macro_delta"], 0)  # episode-weighted delta would be .8
        self.assertEqual(result["ci_low"], -1)
        self.assertEqual(result["ci_high"], 1)
        with self.assertRaises(ValueError):
            paired_task_bootstrap(rows + [rows[0]], 100)


if __name__ == "__main__":
    unittest.main()
