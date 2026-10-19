import math
import unittest

from energy_thermal import Sample, incremental_energy, integrate, per_success, thermal_step


class EnergyTests(unittest.TestCase):
    def test_irregular_sample_integral(self):
        result = integrate([Sample(0, 6), Sample(1, 10), Sample(3, 10), Sample(4, 6)],
                           maximum_gap_s=2)
        self.assertEqual(result, {"duration_s": 4, "energy_j": 36, "average_w": 9})
        # Arithmetic mean 8 W is incorrect for these irregular sample times.
        self.assertNotEqual(result["average_w"], 8)

    def test_linear_power_independent_of_sampling_density(self):
        coarse = integrate([Sample(0, 2), Sample(5, 12)], maximum_gap_s=5)
        fine = integrate([Sample(t, 2 + 2*t) for t in (0, .1, .9, 2, 4.8, 5)], maximum_gap_s=3)
        self.assertAlmostEqual(coarse["energy_j"], 35)
        self.assertAlmostEqual(fine["energy_j"], 35)

    def test_bad_time_and_gap_fail(self):
        for points in ([Sample(0, 1)], [Sample(1, 1), Sample(1, 2)],
                       [Sample(1, 1), Sample(0, 2)], [Sample(0, 1), Sample(3, 2)]):
            with self.subTest(points=points), self.assertRaises(ValueError):
                integrate(points, maximum_gap_s=2)

    def test_gap_boundary_tolerates_only_timestamp_roundoff(self):
        result = integrate([Sample(.1, 2), Sample(.4, 2)], maximum_gap_s=.3)
        self.assertAlmostEqual(result["energy_j"], .6)
        epoch = 1_700_000_000.0
        result = integrate([Sample(epoch, 2), Sample(epoch + .1, 2)], maximum_gap_s=.1)
        self.assertAlmostEqual(result["energy_j"], .2, places=6)
        for timestamps in ((.1, .40001), (epoch, epoch + .10001)):
            gap = .3 if timestamps[0] == .1 else .1
            with self.subTest(timestamps=timestamps), self.assertRaises(ValueError):
                integrate([Sample(t, 2) for t in timestamps], maximum_gap_s=gap)
        with self.assertRaises(ValueError):
            integrate([Sample(1e20, 2), Sample(math.nextafter(1e20, math.inf), 2)],
                      maximum_gap_s=.1)

    def test_nonfinite_negative_and_bool_fail(self):
        for value in (math.nan, math.inf, -1, True):
            with self.subTest(value=value), self.assertRaises(ValueError):
                integrate([Sample(0, 1), Sample(1, value)], maximum_gap_s=2)
        with self.assertRaises(ValueError):
            integrate([Sample(0, 1), Sample(math.nan, 2)], maximum_gap_s=2)

    def test_success_denominator(self):
        result = per_success(36, 200, 180)
        self.assertEqual(result["joules_per_valid_on_time_result"], .2)
        self.assertEqual(result["success_fraction"], .9)
        self.assertGreater(per_success(28, 200, 100)["joules_per_valid_on_time_result"], .2)
        for attempts, successes in ((0, 0), (10, 11), (10, -1), (10, True), (10, 2.5)):
            with self.subTest(counts=(attempts, successes)), self.assertRaises(ValueError):
                per_success(36, attempts, successes)

    def test_incremental_is_separate_and_not_clipped(self):
        self.assertEqual(incremental_energy(36, 4, 4), 20)
        self.assertEqual(incremental_energy(10, 4, 4), -6)


class ThermalTests(unittest.TestCase):
    def test_one_time_constant(self):
        actual = thermal_step(35, 35, 15, 2, 40, 80)
        self.assertAlmostEqual(actual, 53.96361676485673)

    def test_exact_subdivision(self):
        whole = thermal_step(35, 35, 15, 2, 40, 240)
        split = 35
        for _ in range(24):
            split = thermal_step(split, 35, 15, 2, 40, 10)
        self.assertAlmostEqual(whole, split)
        self.assertLess(whole, 65)

    def test_equilibrium_cooling_and_zero_time(self):
        self.assertEqual(thermal_step(65, 35, 15, 2, 40, 10000), 65)
        self.assertEqual(thermal_step(52, 35, 15, 2, 40, 0), 52)
        self.assertAlmostEqual(thermal_step(65, 35, 0, 2, 40, 80), 35 + 30/math.e)

    def test_invalid_physics(self):
        for kwargs in ({"power_w": -1}, {"resistance_c_per_w": 0},
                       {"capacity_j_per_c": -2}, {"duration_s": -1},
                       {"initial_c": math.nan}):
            args = dict(initial_c=35, ambient_c=35, power_w=15,
                        resistance_c_per_w=2, capacity_j_per_c=40, duration_s=80)
            args.update(kwargs)
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                thermal_step(**args)


if __name__ == "__main__":
    unittest.main()
