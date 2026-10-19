import sys
from pathlib import Path
import unittest
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
from estimation_extensions import (ekf_range_update, unscented_square, yaw_error_update,
                                   nullspace_project, rts_scalar)


class TestExtensions(unittest.TestCase):
    def test_range_example_and_unobservable_tangent(self):
        x, p = ekf_range_update([3., 4.], np.eye(2), 5.4, .25)
        np.testing.assert_allclose(x, [3.192, 4.256])
        tangent = np.array([-.8, .6])
        self.assertAlmostEqual(float(tangent @ p @ tangent), 1.)
        with self.assertRaises(ValueError):
            ekf_range_update([0., 0.], np.eye(2), 1., .25)

    def test_unscented_square_matches_gaussian_exact_moments(self):
        for mean, variance in [(2., 1.), (0., 2.), (-3., .4)]:
            actual = unscented_square(mean, variance)
            expected = mean**2 + variance, 2*variance**2 + 4*mean**2*variance
            np.testing.assert_allclose(actual, expected, atol=1e-12)

    def test_yaw_boundary_uses_short_residual(self):
        angle, variance = yaw_error_update(np.deg2rad(179), .04, np.deg2rad(-179), .04)
        self.assertAlmostEqual(abs(angle), np.pi)
        self.assertAlmostEqual(variance, .02)

    def test_whitened_nullspace_eliminates_feature_and_preserves_unit_noise(self):
        hf = np.ones((3, 1))
        hs = np.array([[1., 0.], [0., 1.], [1., 1.]])
        cov = np.diag([1., 4., 9.])
        r = np.array([.1, -.2, .4])
        a, h, n = nullspace_project(r, hs, hf, cov)
        b, _, _ = nullspace_project(r + (hf * 7).ravel(), hs, hf, cov)
        np.testing.assert_allclose(a, b, atol=1e-12)
        np.testing.assert_allclose(n @ n.T, np.eye(2), atol=1e-12)
        self.assertEqual(h.shape, (2, 2))

    def test_rts_future_information_changes_past_only_in_smoother(self):
        means, variances = rts_scalar([0., 1.], [1., 1.], [0., 0.], [1., 2.], [1.])
        np.testing.assert_allclose(means, [.5, 1.])
        np.testing.assert_allclose(variances, [.75, 1.])


if __name__ == '__main__':
    unittest.main()
