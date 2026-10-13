import math
import unittest
from robotics_contracts import (
    rates, rotate_xyzw, partial_reset, wilson_interval, command_is_fresh,
)


class TimingTests(unittest.TestCase):
    def test_vector_rate_does_not_multiply_robot_clock(self):
        r = rates(16, 100, 2, 1/120, 2)
        self.assertEqual(r["transitions_per_s"], 800)
        self.assertEqual(r["vector_physics_steps_per_s"], 100)
        self.assertAlmostEqual(r["per_environment_real_time_factor"], 5/6)
        self.assertAlmostEqual(r["policy_period_s"], 1/60)

    def test_invalid_measurements_fail(self):
        for wall in (0, -1, float("nan")):
            with self.assertRaises(ValueError):
                rates(16, 100, 2, 1/120, wall)


class CoordinateTests(unittest.TestCase):
    def test_known_axis_and_inverse(self):
        q = (0, 0, math.sqrt(.5), math.sqrt(.5))
        v = rotate_xyzw(q, (1, 0, 0))
        for got, expected in zip(v, (0, 1, 0)):
            self.assertAlmostEqual(got, expected)
        restored = rotate_xyzw((0, 0, -q[2], q[3]), v)
        for got, expected in zip(restored, (1, 0, 0)):
            self.assertAlmostEqual(got, expected)

    def test_nonunit_rejected(self):
        with self.assertRaises(ValueError):
            rotate_xyzw((0, 0, 0, 2), (1, 0, 0))


class ResetTests(unittest.TestCase):
    def test_partial_reset_preserves_other_envs_and_input(self):
        before = [[1, 2], [3, 4], [5, 6]]
        after = partial_reset(before, [1], [0, 0])
        self.assertEqual(after, [[1, 2], [0, 0], [5, 6]])
        after[0][0] = 99
        self.assertEqual(before, [[1, 2], [3, 4], [5, 6]])

    def test_invalid_id_and_dimensions(self):
        for ids in ([3], [0, 0], [-1]):
            with self.assertRaises(ValueError):
                partial_reset([[1, 2]], ids, [0, 0])
        with self.assertRaises(ValueError):
            partial_reset([[1, 2]], [0], [0])


class EvaluationTests(unittest.TestCase):
    def test_wilson_includes_uncertainty_at_extremes(self):
        low, high = wilson_interval(80, 100)
        self.assertAlmostEqual(low, .7111708, places=5)
        self.assertAlmostEqual(high, .8666331, places=5)
        self.assertGreater(wilson_interval(0, 100)[1], 0)
        self.assertLess(wilson_interval(100, 100)[0], 1)
        with self.assertRaises(ValueError):
            wilson_interval(0, 0)

    def test_expired_future_and_previous_episode_commands(self):
        self.assertTrue(command_is_fresh(1, .95, .1, 2, 2))
        self.assertFalse(command_is_fresh(1, .5, .1, 2, 2))
        self.assertFalse(command_is_fresh(1, 1.1, .1, 2, 2))
        self.assertFalse(command_is_fresh(1, .95, .1, 2, 1))
        with self.assertRaises(ValueError):
            command_is_fresh(float("nan"), .95, .1, 2, 2)


if __name__ == "__main__":
    unittest.main()
