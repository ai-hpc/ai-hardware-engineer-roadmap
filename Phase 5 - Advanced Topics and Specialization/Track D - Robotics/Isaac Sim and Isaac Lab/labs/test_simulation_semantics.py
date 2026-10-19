import math
import unittest
from simulation_semantics import actuator_response, identify_actuator, paired_regressions, report


class IdentificationTests(unittest.TestCase):
    def test_exact_step_response_and_delay(self):
        got=actuator_response([1]*8, .02, .08, 2)
        self.assertEqual(got[:3], [0,0,0])
        self.assertAlmostEqual(got[3], 1-math.exp(-.25))
        self.assertAlmostEqual(got[8], 1-math.exp(-1.5))

    def test_delay_identified_separately_from_time_constant(self):
        result=report()
        self.assertEqual(result["fitted"]["delay_steps"], 2)
        self.assertAlmostEqual(result["fitted"]["tau_s"], .08, delta=.001)
        self.assertLess(result["held_out_rmse"], .001)
        no_delay=next(x for x in result["delay_candidates"] if x["delay_steps"]==0)
        self.assertGreater(no_delay["one_step_mse"], 100*result["fitted"]["one_step_mse"])

    def test_unexcited_data_is_not_an_identification_success(self):
        with self.assertRaises(ValueError): identify_actuator([0]*8, [0]*9, .02)

    def test_invalid_signals(self):
        for dt,tau,delay in [(0,.1,0),(.1,0,0),(.1,.1,-1),(.1,.1,True)]:
            with self.assertRaises(ValueError): actuator_response([1],dt,tau,delay)
        with self.assertRaises(ValueError): identify_actuator([1]*8, [1]*8, .02)


class PairedTests(unittest.TestCase):
    def test_perfect_small_sample_does_not_certify_one_percent_margin(self):
        result=paired_regressions(dict.fromkeys(range(100), True),dict.fromkeys(range(100), True))
        self.assertFalse(result["gate_passed"])
        self.assertGreater(result["one_sided_95_upper_regression_probability"],.02)

    def test_sufficient_zero_regression_sample_passes_declared_margin(self):
        self.assertTrue(paired_regressions(dict.fromkeys(range(1000), True),dict.fromkeys(range(1000), True))["gate_passed"])

    def test_matching_totals_can_hide_paired_regressions(self):
        result=paired_regressions({"a":True,"b":False},{"a":False,"b":True})
        self.assertEqual((result["regressions"],result["improvements"]),(1,1))
        self.assertFalse(result["gate_passed"])

    def test_missing_scenario_and_invalid_labels_fail(self):
        for a,b in [({},{}),({"a":True},{"b":True}),({"a":1},{"a":True})]:
            with self.assertRaises(ValueError): paired_regressions(a,b)


if __name__ == "__main__": unittest.main()
