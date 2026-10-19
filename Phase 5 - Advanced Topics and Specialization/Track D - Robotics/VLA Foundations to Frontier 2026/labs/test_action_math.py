import math
import statistics
import unittest

from action_math import (SIGMA, alpha_schedule, ddim_indices, ddim_step,
                         diffusion_epsilon, distribution_metrics, flow_velocity,
                         oracle_sample, target_cdf)


class ActionMathTests(unittest.TestCase):
    def test_perfect_epsilon_recovers_clean_target_across_skipped_levels(self):
        clean, epsilon = 0.6, -1.0
        current, following = 0.25, 0.8
        x = math.sqrt(current) * clean + math.sqrt(1 - current) * epsilon
        expected = math.sqrt(following) * clean + math.sqrt(1 - following) * epsilon
        self.assertAlmostEqual(ddim_step(x, epsilon, current, following), expected)
        self.assertAlmostEqual(ddim_step(x, epsilon, current, 1.0), clean)
        self.assertGreater(abs(ddim_step(x, clean, current, 1.0) - clean), 1.0)

    def test_schedule_and_jumps_include_both_endpoints_without_duplicate_steps(self):
        schedule = alpha_schedule()
        self.assertEqual(schedule[0], 1.0)
        self.assertEqual(schedule[-1], 1e-4)
        self.assertTrue(all(a > b for a, b in zip(schedule, schedule[1:])))
        # More cosine levels can hit the fixed floor before the endpoint,
        # yielding alpha_k=1 (beta_k=0) and invalid equal-endpoint DDIM jumps.
        with self.assertRaisesRegex(ValueError, "duplicates levels"):
            alpha_schedule(158)
        for count in (1, 3, 8, 32, 64):
            levels = ddim_indices(64, count)
            self.assertEqual((levels[0], levels[-1], len(levels)), (64, 0, count + 1))
            self.assertTrue(all(a > b for a, b in zip(levels, levels[1:])))

    def test_oracle_epsilon_matches_numerical_score_of_mixture_density(self):
        def log_density(x, alpha):
            variance = alpha * SIGMA**2 + 1 - alpha
            return math.log(sum(math.exp(-(x - math.sqrt(alpha) * m)**2 / (2 * variance))
                                for m in (-1, 1)))
        for x in (-1.3, -0.2, 0.0, 0.5, 1.4):
            for alpha in (0.1, 0.5, 0.9):
                step = 1e-5
                score = (log_density(x + step, alpha) - log_density(x - step, alpha)) / (2 * step)
                self.assertAlmostEqual(diffusion_epsilon(x, alpha, "either"),
                                       -math.sqrt(1 - alpha) * score, places=7)

    def test_flow_field_matches_derivative_of_known_gaussian_marginals(self):
        # The exact marginal path is X(t)=t*mean+sqrt((1-t)^2+t²σ²)*Z.
        for mean, label in ((-1, "left"), (1, "right")):
            for t in (0.0, 0.2, 0.7, 1.0):
                for z in (-2, 0, 1.5):
                    scale = math.sqrt((1 - t)**2 + t*t*SIGMA**2)
                    derivative = mean + (-(1 - t) + t * SIGMA**2) / scale * z
                    self.assertAlmostEqual(flow_velocity(t * mean + scale * z, t, label), derivative)

    def test_oracle_sampling_tracks_distribution_and_detects_wrong_direction(self):
        noise = [statistics.NormalDist().inv_cdf((i + 0.5) / 512) for i in range(512)]
        for method in ("flow", "ddim"):
            for label in ("left", "right", "either"):
                samples = [oracle_sample(label, z, 64, method) for z in noise]
                metric = distribution_metrics(samples, label)
                self.assertLess(metric["cdf_distance"], 0.05)
                self.assertLess(metric["between_modes_fraction"], 0.01)
        reversed_values = [oracle_sample("left", z, 32, "flow", True) for z in noise]
        self.assertGreater(distribution_metrics(reversed_values, "left")["cdf_distance"], 0.5)

    def test_mean_is_wrong_distribution_for_either(self):
        metric = distribution_metrics([0.0] * 10, "either")
        self.assertEqual(metric["mean"], 0.0)
        self.assertEqual(metric["between_modes_fraction"], 1.0)
        self.assertEqual(metric["cdf_distance"], 0.5)
        self.assertAlmostEqual(target_cdf(0, "either"), 0.5)
        self.assertAlmostEqual(target_cdf(-1, "left"), 0.5)

    def test_underflow_stability_and_invalid_contracts(self):
        self.assertTrue(math.isfinite(diffusion_epsilon(1000, 0.99, "either")))
        self.assertTrue(math.isfinite(flow_velocity(-1000, 0.99, "either")))
        bad_calls = [lambda: ddim_indices(64, 65), lambda: ddim_indices(64, True),
                     lambda: alpha_schedule(0), lambda: ddim_step(0, 0, 0.5, 0.2),
                     lambda: diffusion_epsilon(0, 1, "left"),
                     lambda: flow_velocity(0, float("nan"), "left"),
                     lambda: oracle_sample("either", 0, 0),
                     lambda: oracle_sample("invalid", 0),
                     lambda: distribution_metrics([], "left"),
                     lambda: distribution_metrics([float("inf")], "left")]
        for call in bad_calls:
            with self.assertRaises(ValueError):
                call()


if __name__ == "__main__":
    unittest.main()
