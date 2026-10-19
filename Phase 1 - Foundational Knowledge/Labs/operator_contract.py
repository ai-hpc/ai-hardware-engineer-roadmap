#!/usr/bin/env python3
"""CPU reference for the numbers, shapes, and bytes lesson; standard library only."""

import argparse
import json
import math
import platform
import unittest
from pathlib import Path


WEIGHTS = [[1.0, -0.5, 0.25], [-1.0, 0.75, 0.5]]
INPUT = [2.0, -1.0, 0.5]
WEIGHT_SCALE = 0.25
INPUT_SCALE = 0.5


def quantize(values, scale):
    """Signed int8, zero point 0, nearest-even rounding, then saturation."""
    if not math.isfinite(scale) or scale <= 0:
        raise ValueError("scale must be finite and positive")
    if any(not math.isfinite(value) for value in values):
        raise ValueError("inputs must be finite")
    quantized = []
    for value in values:
        scaled = value / scale
        # Clamp before round so finite values with a tiny scale can saturate
        # even when the floating-point division overflows to infinity.
        quantized.append(127 if scaled >= 127 else -128 if scaled <= -128 else round(scaled))
    return quantized


def matvec(weights, vector):
    if not vector or not weights or any(len(row) != len(vector) for row in weights):
        raise ValueError("expected nonempty weights[M,K] and vector[K]")
    return [sum(w * x for w, x in zip(row, vector)) for row in weights]


def traffic_budget(rows, columns, batch):
    """Analytical compulsory traffic, assuming weights are read once per batch."""
    if any(type(size) is not int or size <= 0 for size in (rows, columns, batch)):
        raise ValueError("shape dimensions must be positive integers")
    weight_bytes = rows * columns  # one byte per int8 weight
    input_bytes = batch * columns  # one byte per int8 input
    output_bytes = batch * rows * 4  # four bytes per int32 accumulator
    total_bytes = weight_bytes + input_bytes + output_bytes
    operations = 2 * batch * rows * columns  # one multiply + one add per MAC
    return {
        "shape_B_M_K": [batch, rows, columns],
        "weight_bytes": weight_bytes,
        "input_bytes": input_bytes,
        "output_bytes": output_bytes,
        "total_bytes": total_bytes,
        "integer_operations": operations,
        "operations_per_byte": operations / total_bytes,
        "assumption": "weights and inputs read once; outputs written once; metadata excluded",
    }


def reference_report():
    q_weights = [quantize(row, WEIGHT_SCALE) for row in WEIGHTS]
    q_input = quantize(INPUT, INPUT_SCALE)
    accumulators = matvec(q_weights, q_input)
    recovered = [value * WEIGHT_SCALE * INPUT_SCALE for value in accumulators]
    expected = matvec(WEIGHTS, INPUT)
    return {
        "execution_status": "CPU numerical reference and analytical traffic model",
        "python": platform.python_version(),
        "contract": {
            "weights_shape": [2, 3], "input_shape": [3], "output_shape": [2],
            "weight_scale": WEIGHT_SCALE, "input_scale": INPUT_SCALE,
            "zero_point": 0, "rounding": "nearest, ties to even",
            "saturation_range": [-128, 127], "accumulator_type": "int32 contract",
            "implementation_note": "Python integers are unbounded; this example fits int32",
        },
        "quantized_weights": q_weights, "quantized_input": q_input,
        "accumulators": accumulators, "recovered_output": recovered,
        "floating_reference": expected,
        "maximum_absolute_error": max(abs(a - b) for a, b in zip(recovered, expected)),
        "traffic_examples": [traffic_budget(1024, 1024, b) for b in (1, 8)],
    }


class ContractTests(unittest.TestCase):
    def test_worked_example(self):
        report = reference_report()
        self.assertEqual(report["accumulators"], [21, -20])
        self.assertEqual(report["recovered_output"], [2.625, -2.5])
        self.assertEqual(report["maximum_absolute_error"], 0)

    def test_nearest_even_and_saturation(self):
        self.assertEqual(quantize([0.5, 1.5, -0.5, -1.5, 200, -200], 1),
                         [0, 2, 0, -2, 127, -128])
        self.assertEqual(quantize([1.0, -1.0], 1e-320), [127, -128])

    def test_shape_mismatch_is_rejected(self):
        with self.assertRaises(ValueError):
            matvec(WEIGHTS, [1, 2])
        with self.assertRaises(ValueError):
            matvec([[1, 2], [3]], [1, 2])

    def test_invalid_quantization_is_rejected(self):
        for scale in (0, -1, float("inf"), float("nan")):
            with self.assertRaises(ValueError):
                quantize(INPUT, scale)
        with self.assertRaises(ValueError):
            quantize([float("nan")], 1)

    def test_missing_input_scale_changes_the_result(self):
        report = reference_report()
        wrong = [value * WEIGHT_SCALE for value in report["accumulators"]]
        self.assertNotEqual(wrong, report["floating_reference"])

    def test_batch_reuses_weights_in_the_model(self):
        one = traffic_budget(1024, 1024, 1)
        eight = traffic_budget(1024, 1024, 8)
        self.assertEqual(one["total_bytes"], 1053696)
        self.assertEqual(eight["weight_bytes"], one["weight_bytes"])
        self.assertGreater(eight["operations_per_byte"], one["operations_per_byte"])
        with self.assertRaises(ValueError):
            traffic_budget(1024, 1024, 0)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.self_test:
        result = unittest.TextTestRunner(verbosity=2).run(
            unittest.defaultTestLoader.loadTestsFromTestCase(ContractTests))
        if not result.wasSuccessful():
            raise SystemExit(1)
    rendered = json.dumps(reference_report(), indent=2, allow_nan=False) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    else:
        print(rendered, end="")


if __name__ == "__main__":
    main()
