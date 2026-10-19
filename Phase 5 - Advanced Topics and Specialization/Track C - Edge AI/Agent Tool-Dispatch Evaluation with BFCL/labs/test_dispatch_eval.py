from copy import deepcopy
import json
from pathlib import Path
import unittest

from dispatch_eval import InvalidCall, canonical, catalog_hash, equivalent, evaluate, paired_counts, parse_and_validate, wilson


class DispatchTests(unittest.TestCase):
    def setUp(self):
        here = Path(__file__).parent
        self.fixtures = json.loads((here / "fixtures.json").read_text())
        self.good = json.loads((here / "responses-good.json").read_text())
        self.bad = json.loads((here / "responses-regression.json").read_text())

    def test_reference_and_injected_regressions(self):
        reference, candidate = evaluate(self.fixtures, self.good), evaluate(self.fixtures, self.bad)
        self.assertEqual(reference["passed"], 6)
        self.assertTrue(reference["gate_passed"])
        self.assertEqual(candidate["passed"], 2)
        self.assertFalse(candidate["gate_passed"])
        self.assertEqual(paired_counts(reference, candidate),
                         {"both_pass": 2, "regressions": 4, "improvements": 0, "both_fail": 0})

    def test_parallel_is_a_multiset(self):
        calls = self.fixtures[2]["turns"][0]["calls"]
        self.assertTrue(equivalent(calls[::-1], calls, parallel=True))
        self.assertFalse(equivalent(calls[::-1], calls, parallel=False))
        self.assertFalse(equivalent(calls + calls[:1], calls, parallel=True))

    def test_boolean_is_not_an_integer_argument(self):
        raw = '[{"name":"set_level","arguments":{"room":"office","level":true}}]'
        with self.assertRaisesRegex(InvalidCall, "wrong_argument_type"):
            parse_and_validate(raw)

    def test_duplicates_and_nonfinite_values_fail(self):
        for raw in ('[{"name":"add","name":"set_level","arguments":{}}]',
                    '[{"name":"add","arguments":{"a":NaN,"b":1}}]',
                    '[{"name":"add","arguments":{"a":1e999,"b":1}}]',
                    '[{"name":"add","arguments":{"a":' + '1' * 400 + ',"b":1}}]'):
            with self.assertRaises(InvalidCall):
                parse_and_validate(raw)

    def test_code_and_repairs_are_not_silently_accepted(self):
        for raw in ("__import__('os').system('true')", "```json\n[]\n```", "[{'name': 'add'}]", "[1,]"):
            with self.assertRaises(InvalidCall):
                parse_and_validate(raw)

    def test_unknown_missing_extra_and_range(self):
        calls = [{"name": "delete_all", "arguments": {}},
                 {"name": "add", "arguments": {"a": 1}},
                 {"name": "add", "arguments": {"a": 1, "b": 2, "c": 3}},
                 {"name": "set_level", "arguments": {"room": "office", "level": 101}}]
        for call in calls:
            with self.assertRaises(InvalidCall):
                parse_and_validate(canonical([call]))

    def test_missing_extra_duplicate_case_or_stale_catalog_cannot_pass(self):
        bad = deepcopy(self.good)
        del bad["outputs"]["simple"]
        with self.assertRaises(ValueError):
            evaluate(self.fixtures, bad)
        bad = deepcopy(self.good)
        bad["outputs"]["unrequested"] = ["[]"]
        with self.assertRaises(ValueError):
            evaluate(self.fixtures, bad)
        bad = deepcopy(self.good)
        bad["catalog_sha256"] = "0" * 64
        with self.assertRaises(ValueError):
            evaluate(self.fixtures, bad)
        with self.assertRaises(ValueError):
            evaluate(self.fixtures + self.fixtures[:1], self.good)

    def test_turn_order_and_history_matter(self):
        bad = deepcopy(self.good)
        bad["outputs"]["turns"].reverse()
        result = evaluate(self.fixtures, bad)
        row = next(row for row in result["cases"] if row["id"] == "turns")
        self.assertFalse(row["passed"])
        self.assertIn("simulated_outcome_mismatch", row["errors"])

    def test_empty_or_incomplete_evaluations_fail(self):
        with self.assertRaises(ValueError):
            evaluate([], {"catalog_sha256": catalog_hash(), "outputs": {}})
        bad = deepcopy(self.good)
        bad["outputs"]["turns"] = bad["outputs"]["turns"][:1]
        self.assertFalse(evaluate(self.fixtures, bad)["gate_passed"])

    def test_small_sample_uncertainty_is_visible(self):
        low, high = wilson(1, 1)
        self.assertLess(low, .21)
        self.assertAlmostEqual(high, 1)
        low, high = wilson(50, 100)
        self.assertLess(low, .41)
        self.assertGreater(high, .59)


if __name__ == "__main__":
    unittest.main()
