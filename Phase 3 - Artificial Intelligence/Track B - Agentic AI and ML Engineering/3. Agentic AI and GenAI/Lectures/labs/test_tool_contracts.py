import unittest
from tool_contracts import CallBudget, dispatch, parse_arguments


class ToolContractTests(unittest.TestCase):
    def test_arithmetic(self):
        self.assertEqual(dispatch("add_integers", {"a": -4, "b": 9}), {"ok": True, "sum": 5})

    def test_strict_types_and_bounds(self):
        for v in [True, "2", 1.5, 1000001, float("nan")]:
            self.assertFalse(dispatch("add_integers", {"a": v, "b": 2})["ok"])

    def test_unknown_and_extra_fields(self):
        self.assertEqual(dispatch("eval", {"code": "1+1"})["error"], "unknown-tool")
        self.assertFalse(dispatch("add_integers", {"a": 1, "b": 2, "command": "unused"})["ok"])

    def test_duplicate_json_rejected(self):
        with self.assertRaisesRegex(ValueError, "duplicate-key"):
            parse_arguments('{"a":1,"a":2,"b":3}')

    def test_json_object_and_size(self):
        for raw in ["[]", " " * 4097]:
            with self.assertRaises(ValueError): parse_arguments(raw)

    def test_duplicate_id_across_turns(self):
        b = CallBudget(2); b.admit(["call-1"])
        with self.assertRaisesRegex(ValueError, "duplicate-call-id"): b.admit(["call-1"])

    def test_batch_admission_atomic(self):
        b = CallBudget(2); b.admit(["call-1"])
        with self.assertRaisesRegex(ValueError, "tool-call-budget"): b.admit(["call-2", "call-3"])
        self.assertEqual(b.seen, {"call-1"})

    def test_duplicate_same_batch(self):
        b = CallBudget()
        with self.assertRaisesRegex(ValueError, "duplicate-call-id"): b.admit(["x", "x"])
        self.assertFalse(b.seen)


if __name__ == "__main__": unittest.main()
