import unittest
from contracts import Measurement, visible_measurements


class TestContracts(unittest.TestCase):
    def test_late_sensor_cannot_change_earlier_output(self):
        m = Measurement(1, 1.4, 'camera', 2, .01, 0)
        self.assertEqual(visible_measurements([m], 1.3), [])
        self.assertEqual(visible_measurements([m], 1.4), [m])

    def test_capture_order_and_duplicate_timestamps(self):
        a = Measurement(1, 2, 'camera', 2, .01, 4)
        b = Measurement(1, 1, 'wheel', .5, .01, 3)
        self.assertEqual(visible_measurements([a, b], 2), [b, a])

    def test_invalid_units_clock_and_covariance_fail(self):
        for args in [(2, 1, 'camera', 0, .1, 0), (0, 0, 'camera', 0, 0, 0),
                     (0, 0, 'camera', float('nan'), .1, 0),
                     (0, 0, 'unknown', 0, .1, 0)]:
            with self.assertRaises(ValueError):
                Measurement(*args)


if __name__ == '__main__':
    unittest.main()
