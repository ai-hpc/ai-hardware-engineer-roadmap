"""Offline teaching evaluator. Custom fixtures; NOT the official BFCL scorer.

Only an in-memory light simulator and arithmetic run. No generated code is eval'd.
"""
import argparse
from collections import Counter, defaultdict
from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path

CATALOG = {
    "set_level": {"room": {"type": "string", "enum": ["kitchen", "office"]},
                  "level": {"type": "integer", "minimum": 0, "maximum": 100}},
    "add": {"a": {"type": "number"}, "b": {"type": "number"}},
}


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def catalog_hash():
    return hashlib.sha256(canonical(CATALOG).encode()).hexdigest()


class InvalidCall(ValueError):
    pass


def unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise InvalidCall("duplicate_json_key")
        result[key] = value
    return result


def reject_constant(value):
    raise InvalidCall("nonfinite_number")


def parse_and_validate(raw):
    try:
        calls = json.loads(raw, object_pairs_hook=unique_object, parse_constant=reject_constant)
    except json.JSONDecodeError as exc:
        raise InvalidCall("invalid_json") from exc
    if not isinstance(calls, list):
        raise InvalidCall("expected_call_list")
    for call in calls:
        if not isinstance(call, dict) or set(call) != {"name", "arguments"}:
            raise InvalidCall("invalid_call_envelope")
        if not isinstance(call["name"], str) or call["name"] not in CATALOG:
            raise InvalidCall("unknown_tool")
        args, schema = call["arguments"], CATALOG[call["name"]]
        if not isinstance(args, dict):
            raise InvalidCall("invalid_arguments")
        if set(schema) - set(args):
            raise InvalidCall("missing_argument")
        if set(args) - set(schema):
            raise InvalidCall("extra_argument")
        for name, spec in schema.items():
            value, kind = args[name], spec["type"]
            valid = ((kind == "string" and isinstance(value, str))
                     or (kind == "integer" and type(value) is int)
                     or (kind == "number" and type(value) in (int, float)))
            if not valid:
                raise InvalidCall("wrong_argument_type")
            if kind in ("number", "integer"):
                try:
                    finite = math.isfinite(value)
                except OverflowError:
                    finite = False
                if not finite:
                    raise InvalidCall("number_outside_finite_float_range")
            if ("enum" in spec and value not in spec["enum"]
                    or "minimum" in spec and value < spec["minimum"]
                    or "maximum" in spec and value > spec["maximum"]):
                raise InvalidCall("argument_out_of_range")
    return calls


def equivalent(actual, expected, parallel=False):
    # Multiset, not set: duplicate calls must never disappear from scoring.
    a, b = [canonical(x) for x in actual], [canonical(x) for x in expected]
    return Counter(a) == Counter(b) if parallel else a == b


def execute(calls, state):
    """Input is already validated. State is a local dictionary, never a device."""
    outputs = []
    for call in calls:
        args = call["arguments"]
        if call["name"] == "set_level":
            state[args["room"]] = args["level"]
            outputs.append({"level": args["level"]})
        elif call["name"] == "add":
            outputs.append(args["a"] + args["b"])
    return outputs


def wilson(successes, count, z=1.96):
    if count <= 0 or not 0 <= successes <= count:
        raise ValueError("nonempty sample and valid success count required")
    p = successes / count
    denominator = 1 + z * z / count
    center = (p + z * z / (2 * count)) / denominator
    radius = z * math.sqrt(p * (1 - p) / count + z * z / (4 * count * count)) / denominator
    return [max(0, center - radius), min(1, center + radius)]


def evaluate(fixtures, responses):
    ids = [case["id"] for case in fixtures]
    if not ids or len(set(ids)) != len(ids):
        raise ValueError("fixture IDs must be nonempty and unique")
    if responses.get("catalog_sha256") != catalog_hash():
        raise ValueError("catalog fingerprint mismatch")
    if set(responses.get("outputs", {})) != set(ids):
        raise ValueError("response IDs must exactly cover fixtures")
    rows = []
    for case in fixtures:
        state, expected_state = {}, {}
        raw_turns = responses["outputs"][case["id"]]
        errors, parsed_all, matched_all = [], True, True
        if not isinstance(raw_turns, list) or len(raw_turns) != len(case["turns"]):
            errors.append("turn_count_mismatch")
            parsed_all = matched_all = False
        else:
            for turn, raw in zip(case["turns"], raw_turns):
                expected = parse_and_validate(canonical(turn["calls"]))
                expected_outputs = execute(expected, expected_state)
                try:
                    if not isinstance(raw, str):
                        raise InvalidCall("expected_raw_string")
                    calls = parse_and_validate(raw)
                except InvalidCall as exc:
                    errors.append(str(exc))
                    parsed_all = matched_all = False
                    continue
                if not equivalent(calls, expected, turn.get("parallel", False)):
                    errors.append("call_mismatch")
                    matched_all = False
                actual_outputs = execute(calls, state)
                if turn.get("parallel", False):
                    result_match = Counter(map(canonical, actual_outputs)) == Counter(map(canonical, expected_outputs))
                else:
                    result_match = actual_outputs == expected_outputs
                if state != expected_state or not result_match:
                    errors.append("simulated_outcome_mismatch")
        rows.append({"id": case["id"], "category": case["category"],
                     "parsed_and_valid": parsed_all, "exact_calls": matched_all,
                     "passed": not errors, "errors": sorted(set(errors))})
    grouped = defaultdict(list)
    for row in rows:
        grouped[row["category"]].append(row["passed"])
    categories = {name: {"passed": sum(values), "total": len(values),
                         "accuracy": sum(values) / len(values),
                         "wilson_95_interval": wilson(sum(values), len(values))}
                  for name, values in sorted(grouped.items())}
    return {"benchmark": "course custom fixtures, not BFCL scores", "cases": rows,
            "categories": categories, "passed": sum(r["passed"] for r in rows),
            "total": len(rows), "gate_passed": all(r["passed"] for r in rows)}


def paired_counts(reference, candidate):
    ref = {row["id"]: row["passed"] for row in reference["cases"]}
    new = {row["id"]: row["passed"] for row in candidate["cases"]}
    if set(ref) != set(new):
        raise ValueError("paired comparisons require identical case IDs")
    return {"both_pass": sum(ref[k] and new[k] for k in ref),
            "regressions": sum(ref[k] and not new[k] for k in ref),
            "improvements": sum(not ref[k] and new[k] for k in ref),
            "both_fail": sum(not ref[k] and not new[k] for k in ref)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--responses", type=Path, default=Path(__file__).with_name("responses-good.json"))
    args = parser.parse_args()
    fixtures = json.loads(Path(__file__).with_name("fixtures.json").read_text())
    result = evaluate(fixtures, json.loads(args.responses.read_text()))
    print(json.dumps(result, indent=2))
    return 0 if result["gate_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
