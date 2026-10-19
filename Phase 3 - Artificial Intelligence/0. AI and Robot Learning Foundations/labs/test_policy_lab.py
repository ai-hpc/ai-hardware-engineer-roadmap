"""Checks for mathematical correctness, held-out data, and meaningful fixture failures."""

import math
import unittest

from policy_lab import (Sample, evaluate, features, fit_scaler, gradient_error,
                        make_dataset, rollout, run, split_episodes, validate_splits)


class PolicyLabTests(unittest.TestCase):
    def test_hand_worked_prediction_and_rmse(self):
        rows = [Sample(0, 0.5, 0.2, 0.28), Sample(1, -0.5, 0.2, -0.52)]
        score = evaluate(rows, lambda e, v: 0.8 * e - 0.6 * v)
        self.assertLess(score["pooled_action_rmse_m_s2"], 1e-12)

    def test_scaling_has_known_values_and_constant_feature_rule(self):
        rows = [Sample(i, float(i), 2.0, 0.0) for i in (1, 2, 3)]
        scaler = fit_scaler(rows)
        self.assertEqual(scaler[0], [2.0, 2.0])
        self.assertEqual(scaler[1][1], 1.0)
        self.assertAlmostEqual(features(4, 2, scaler)[0], math.sqrt(6))

    def test_episode_overlap_rejected(self):
        parts = split_episodes(make_dataset())
        validate_splits(parts)
        parts[1].append(parts[0][0])
        with self.assertRaisesRegex(ValueError, "Episode leakage"):
            validate_splits(parts)

    def test_all_data_assigned_once(self):
        rows = make_dataset()
        parts = split_episodes(rows)
        self.assertEqual(sum(map(len, parts)), len(rows))
        self.assertEqual([len({r.episode for r in p}) for p in parts], [24, 6, 6])

    def test_gradient_against_finite_difference(self):
        rows = make_dataset()[:20]
        self.assertLess(gradient_error([0.4, -0.3, 0.2], rows, fit_scaler(rows)), 1e-7)

    def test_episode_and_frame_weighting_are_distinct(self):
        rows = [Sample(0, 0, 0, 0), Sample(1, 0, 0, 2), Sample(1, 0, 0, 2)]
        result = evaluate(rows, lambda e, v: 0)
        self.assertAlmostEqual(result["mean_episode_action_rmse_m_s2"], 1.0)
        self.assertAlmostEqual(result["pooled_action_rmse_m_s2"], math.sqrt(8 / 3))

    def test_teacher_rollout_and_observation_bias(self):
        teacher = lambda e, v: 0.8 * e - 0.6 * v
        self.assertEqual(rollout(teacher)["successes"], 12)
        self.assertEqual(rollout(teacher, error_bias=0.6)["successes"], 0)

    def test_nonfinite_action_rejected(self):
        with self.assertRaisesRegex(ValueError, "non-finite"):
            rollout(lambda e, v: float("nan"))

    def test_noise_is_reproducible_and_preserves_paired_initial_conditions(self):
        teacher = lambda e, v: 0.8 * e - 0.6 * v
        clean = rollout(teacher)
        noisy = rollout(teacher, observation_noise_std=0.1)
        self.assertEqual(noisy, rollout(teacher, observation_noise_std=0.1))
        for a, b in zip(clean["per_episode"], noisy["per_episode"]):
            self.assertEqual(a["initial_position_m"], b["initial_position_m"])
            self.assertEqual(a["initial_velocity_m_s"], b["initial_velocity_m_s"])
        self.assertNotEqual(clean["per_episode"][0]["final_position_m"], noisy["per_episode"][0]["final_position_m"])
        with self.assertRaises(ValueError):
            rollout(teacher, observation_noise_std=-0.1)

    def test_full_learning_experiment(self):
        report = run()
        self.assertTrue(report["leakage_fault_caught"])
        learned = report["test"]["pooled_action_rmse_m_s2"]
        baseline = report["test_zero_baseline"]["pooled_action_rmse_m_s2"]
        self.assertLess(learned, 0.04)
        self.assertLess(learned, baseline / 2)
        self.assertGreater(report["test_error_bias_0_6_m"]["pooled_action_rmse_m_s2"], learned * 5)
        self.assertEqual(report["rollout"]["successes"], 12)
        self.assertEqual(report["rollout_error_bias_0_6_m"]["successes"], 0)


if __name__ == "__main__":
    unittest.main()
