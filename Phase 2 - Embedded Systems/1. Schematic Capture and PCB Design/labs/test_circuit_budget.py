import itertools
import math
import unittest
from dataclasses import replace
from circuit_budget import Bus, assess, crossing, parallel, rise_time, scenarios, waveform


class CircuitTests(unittest.TestCase):
    def test_nominal_math_and_parallel_loading(self):
        self.assertAlmostEqual(rise_time(2200, 100e-12, 3.3)*1e9, 186.405529, places=5)
        self.assertEqual(parallel((2200,)*4), 550)
        self.assertTrue(assess(Bus())['passes_model'])
        self.assertFalse(assess(scenarios()['four_breakouts'])['checks']['sink_guarantee'])

    def test_nominal_success_does_not_cover_tolerance(self):
        report = assess(scenarios()['nominal_only'])
        self.assertLess(report['nominal_rise_ns_no_leakage'], 300)
        self.assertGreater(report['worst_rise_ns'], 300)
        self.assertFalse(report['passes_model'])

    def test_numerical_waveform_converges_to_analytic(self):
        r, c, v = 2200, 100e-12, 3.3
        expected = rise_time(r, c, v)
        errors = []
        for divisor in (20, 200, 2000):
            values = waveform(r, c, v, r*c/divisor)
            estimate = crossing(values, .7*v) - crossing(values, .3*v)
            errors.append(abs(estimate - expected))
        self.assertGreater(errors[0], errors[1])
        self.assertGreater(errors[1], errors[2])
        self.assertLess(errors[2] / expected, .001)

    def test_independent_tolerance_corners_enclosed(self):
        bus = replace(Bus(), pullups_ohm=(3300, 4700), leakage_sink_ua=5)
        report = assess(bus)
        for signs in itertools.product((-.05, .05), repeat=2):
            r = parallel(tuple(v*(1+s) for v, s in zip(bus.pullups_ohm, signs)))
            for c, v in itertools.product((80e-12, 120e-12), (3.135, 3.465)):
                self.assertLessEqual(rise_time(r, c, v, 5e-6)*1e9, report['worst_rise_ns']+1e-10)
                self.assertLessEqual((v-.4)/r*1000, report['worst_current_at_low_max_ma']+1e-10)

    def test_probe_loading_changes_rise_but_not_sink(self):
        before = assess(Bus())
        after = assess(replace(Bus(), capacitance_min_pf=120, capacitance_max_pf=160))
        self.assertFalse(after['checks']['rise'])
        self.assertEqual(before['worst_current_at_low_max_ma'], after['worst_current_at_low_max_ma'])

    def test_leakage_can_prevent_threshold_crossing(self):
        report = assess(scenarios()['leakage_fault'])
        self.assertIsNone(report['worst_rise_ns'])
        self.assertFalse(report['checks']['high_level'])
        self.assertFalse(report['reaches_70_percent'])
        self.assertTrue(math.isinf(rise_time(3300, 100e-12, 3.135, .001)))

    def test_bad_units_or_integration_settings_rejected(self):
        for kwargs in [{'pullups_ohm': ()}, {'resistor_tolerance': 1}, {'rail_min_v': 4},
                       {'capacitance_min_pf': -10}, {'leakage_sink_ua': float('nan')},
                       {'rise_limit_ns': True}]:
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                Bus(**kwargs)
        for step in (0, 1, 1e-20):
            with self.assertRaises(ValueError):
                waveform(2200, 100e-12, 3.3, step)


if __name__ == '__main__':
    unittest.main()
