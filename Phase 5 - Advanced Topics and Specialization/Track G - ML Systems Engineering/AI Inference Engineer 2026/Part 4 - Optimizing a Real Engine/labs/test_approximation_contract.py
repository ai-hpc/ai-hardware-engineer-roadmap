"""Counterexamples to a claimed expert-threshold quality guarantee.

Reuses Part 1's CPU expert combine. Models the reviewed skip predicate only;
these tests do not execute CUDA or claim that K3 produces these routing values.
"""
import importlib.util
import unittest
from pathlib import Path

source = Path(__file__).resolve().parents[2] / 'Part 1 - Fundamentals/labs/inference_contracts.py'
spec = importlib.util.spec_from_file_location('part1_contracts_for_part4', source)
contracts = importlib.util.module_from_spec(spec)
spec.loader.exec_module(contracts)


def combine(weights, values, position, threshold=0.08, minimum_position=16384):
    # The same normalized weights feed both paths; skipping does not renormalize.
    experts = [lambda x, value=v: value for v in values]
    live = position >= minimum_position
    masked = [lambda x, expert=f, weight=w: 0 if live and weight < threshold else expert(x)
              for f, w in zip(experts, weights)]
    return contracts.expert_reference([0], [list(range(len(weights)))], [weights], masked)[0]


class ApproximationCounterexamples(unittest.TestCase):
    def test_uniform_top16_crosses_depth_gate_and_drops_entire_output(self):
        weights, values = [1/16]*16, [1.0]*16
        self.assertEqual(combine(weights, values, 16383), 1.0)
        self.assertEqual(combine(weights, values, 16384), 0.0)
        self.assertEqual(combine(weights, values, 131072, threshold=0), 1.0)

    def test_small_dropped_weight_can_dominate_output(self):
        exact = combine([0.01, 0.99], [1000, 1], 20000, threshold=0)
        approx = combine([0.01, 0.99], [1000, 1], 20000)
        self.assertAlmostEqual(exact, 10.99)
        self.assertAlmostEqual(approx, 0.99)
        self.assertGreater(abs(exact-approx)/abs(exact), 0.9)
        # One weight below 0.08 does not imply relative output error <= 0.08.

    def test_threshold_comparison_is_strict(self):
        self.assertAlmostEqual(combine([0.08, 0.92], [10, 1], 16384), 1.72)
        self.assertAlmostEqual(combine([0.079, 0.921], [10, 1], 16384), 0.921)

    def test_cancellation_makes_relative_error_large(self):
        weights, values = [0.01, 0.99], [99.0, -1.0]
        self.assertAlmostEqual(combine(weights, values, 16384, threshold=0), 0.0)
        self.assertAlmostEqual(combine(weights, values, 16384), -0.99)


if __name__ == '__main__':
    unittest.main(verbosity=2)
