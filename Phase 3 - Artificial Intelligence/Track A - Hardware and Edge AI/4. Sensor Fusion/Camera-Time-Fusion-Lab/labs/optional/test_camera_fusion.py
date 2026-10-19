import sys
from pathlib import Path
import unittest
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
from contracts import Measurement
from camera_fusion import (project_board, recover_position, predict, estimate, generate, run)


class TestCameraFusion(unittest.TestCase):
    def test_noiseless_geometry_and_wrong_scale_low_residual(self):
        pixels = project_board(2., np.random.default_rng(1), 0.)
        z, residual = recover_position(pixels)
        self.assertAlmostEqual(z, 2., places=6)
        self.assertLess(residual, 1e-6)
        bad_z, bad_residual = recover_position(pixels, 1.1)
        self.assertAlmostEqual(bad_z, 1.4, places=6)
        self.assertLess(bad_residual, 1e-6)

    def test_invalid_geometry_and_backward_prediction_fail(self):
        with self.assertRaises(ValueError):
            recover_position(np.zeros((3, 2)))
        with self.assertRaises(ValueError):
            project_board(9., np.random.default_rng(1))
        with self.assertRaises(ValueError):
            predict(np.zeros(2), np.eye(2), -.01)

    def test_continuous_noise_composes_over_substeps(self):
        x, p = np.array([2., .3]), np.eye(2)
        a, b = predict(x, p, .2)
        a, b = predict(a, b, .3)
        c, d = predict(x, p, .5)
        np.testing.assert_allclose(a, c, atol=1e-12)
        np.testing.assert_allclose(b, d, atol=1e-12)

    def test_replay_never_uses_future_arrivals(self):
        _, events, _ = generate()
        a = estimate(events, 2.)
        b = estimate([e for e in events if e.arrival_time <= 2.], 2.)
        np.testing.assert_allclose(a[0], b[0])
        np.testing.assert_allclose(a[1], b[1])
        self.assertGreaterEqual(np.linalg.eigvalsh(a[1]).min(), -1e-12)

    def test_zero_delay_same_filter_and_camera_outlier_gate(self):
        _, events, _ = generate(delay=0.)
        a = estimate(events, 4.)
        b = estimate(events, 4., use_arrival_time=True)
        np.testing.assert_allclose(a[0], b[0], atol=1e-12)
        np.testing.assert_allclose(a[1], b[1], atol=1e-12)
        self.assertGreaterEqual(a[2], 1)

    def test_delayed_replay_beats_wrong_timestamp_for_fixture(self):
        report, _, _ = run()
        methods = report['methods']
        self.assertLess(methods['capture_time_replay']['rmse_m'], methods['arrival_time']['rmse_m'])
        self.assertLess(methods['capture_time_replay']['rmse_m'], methods['wheel_only']['rmse_m'])
        self.assertEqual(report['counts']['outputs_per_method'], 241)
        self.assertGreater(methods['capture_time_replay']['final_rejected_camera'], 0)


if __name__ == '__main__':
    unittest.main()
