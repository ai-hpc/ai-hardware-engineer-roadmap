"""Numerical targets catch episode leakage, timeout bias, and PPO sign errors."""

import math
import unittest

from returns import Transition, clipped_surrogate, discounted_chunk_target, gae


class ReturnsTests(unittest.TestCase):
    def test_terminal_ignores_arbitrary_next_value(self):
        advantage, target = gae([Transition(2, .5, 1000, terminated=True)], .9, .95)
        self.assertEqual(advantage, [1.5])
        self.assertEqual(target, [2])

    def test_timeout_bootstraps_final_state(self):
        advantage, target = gae([Transition(2, 1, 4, truncated=True)], .9, .95)
        self.assertAlmostEqual(advantage[0], 4.6)
        self.assertAlmostEqual(target[0], 5.6)

    def test_terminal_takes_precedence_when_both_flags_are_true(self):
        _, target = gae([Transition(2, 1, 4, terminated=True, truncated=True)], .9, .95)
        self.assertEqual(target, [2])

    def test_no_advantage_leakage_across_reset(self):
        prefix = [Transition(1, .5, 1), Transition(2, 1, 4, truncated=True)]
        a, _ = gae(prefix + [Transition(3, 9, 0, terminated=True)], .9, .8)
        b, _ = gae(prefix + [Transition(900, -200, 0, terminated=True)], .9, .8)
        self.assertEqual(a[:2], b[:2])
        self.assertAlmostEqual(a[0], 4.712)

    def test_rollout_cut_keeps_bootstrap(self):
        _, target = gae([Transition(2, 1, 4)], .9, 1.)
        self.assertAlmostEqual(target[0], 5.6)

    def test_lambda_zero_equals_one_step_td(self):
        advantage, _ = gae([Transition(1, .5, 1), Transition(2, 1, 4, truncated=True)], .9, 0.)
        self.assertAlmostEqual(advantage[0], 1.4)
        self.assertAlmostEqual(advantage[1], 4.6)

    def test_lambda_one_full_episode_matches_monte_carlo(self):
        rewards = [1., -2., 3.]
        values = [.7, 1.1, -.4]
        transitions = [Transition(rewards[i], values[i], values[i+1] if i < 2 else 0.,
                                  terminated=i == 2) for i in range(3)]
        _, targets = gae(transitions, .9, 1.)
        # Independent direct sum; no backward recurrence or value predictions.
        expected = [sum(.9 ** k * r for k, r in enumerate(rewards[i:])) for i in range(3)]
        for a, b in zip(targets, expected):
            self.assertAlmostEqual(a, b)

    def test_wrong_reset_observation_changes_target(self):
        _, correct = gae([Transition(2, 1, 4, truncated=True)], .9, .8)
        _, wrong = gae([Transition(2, 1, 9, truncated=True)], .9, .8)
        self.assertAlmostEqual(wrong[0] - correct[0], 4.5)

    def test_invalid_inputs(self):
        for transitions, gamma, lam in (([], .9, .9), ([Transition(math.nan, 1, 2)], .9, .9),
                                        ([Transition(1, 1, 2)], 1.1, .9)):
            with self.assertRaises(ValueError):
                gae(transitions, gamma, lam)


class PpoTests(unittest.TestCase):
    def test_positive_advantage_caps_improvement(self):
        self.assertAlmostEqual(clipped_surrogate(1.5, 2), 2.4)
        self.assertAlmostEqual(clipped_surrogate(.5, 2), 1.)

    def test_negative_advantage_uses_other_side(self):
        self.assertAlmostEqual(clipped_surrogate(.5, -2), -1.6)
        self.assertAlmostEqual(clipped_surrogate(1.5, -2), -3.)

    def test_zero_advantage_zero_objective(self):
        self.assertEqual(clipped_surrogate(10, 0), 0)


class ChunkTargets(unittest.TestCase):
    def test_duration_discount_matches_nested_one_step_backups(self):
        target = discounted_chunk_target([1., 2.], 10., .9)
        self.assertAlmostEqual(target, 1 + .9 * (2 + .9 * 10))
        self.assertAlmostEqual(target, 10.9)

    def test_terminal_and_shortened_chunk(self):
        self.assertAlmostEqual(discounted_chunk_target([1., 2.], 1000., .9, True), 2.8)
        self.assertAlmostEqual(discounted_chunk_target([1.], 10., .9), 10.)
        self.assertAlmostEqual(discounted_chunk_target([1.], 10., .9, True), 1.)

    def test_discount_endpoints(self):
        self.assertEqual(discounted_chunk_target([1., 2., 3.], 10., 0.), 1.)
        self.assertEqual(discounted_chunk_target([1., 2., 3.], 10., 1.), 16.)

    def test_invalid_or_unexecuted_chunk_rejected(self):
        for rewards,value,gamma in [([],1.,.9),([math.nan],1.,.9),([1.],math.inf,.9),([1.],1.,1.1)]:
            with self.assertRaises(ValueError): discounted_chunk_target(rewards,value,gamma)


if __name__ == "__main__":
    unittest.main()
