#!/usr/bin/env python3
"""Optional pinned tinygrad CPU execution; deliberately separate from stdlib tests."""

import argparse
import hashlib
from importlib.metadata import version
import json
import math
import os
from pathlib import Path
import platform
import statistics
import subprocess
import sys
import time

from reference import fixture, fold_batchnorm, max_error, reference


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repeats", type=int, default=30)
    parser.add_argument("--debug", type=int, choices=(0, 2, 4), default=0)
    args = parser.parse_args()
    if not 5 <= args.repeats <= 200:
        parser.error("--repeats must be in [5, 200]")
    # These are documented tinygrad settings, set before importing the optional package.
    os.environ["DEBUG"] = str(args.debug)
    os.environ["DEV"] = "CPU"
    os.environ["BEAM"] = "0"
    try:
        package_version = version("tinygrad")
    except Exception:
        parser.error("Install the pinned optional requirements in a virtual environment first")
    if package_version != "0.14.0":
        parser.error(f"This exercise is pinned to tinygrad 0.14.0; found {package_version}")
    from tinygrad import Tensor, TinyJit, Device, dtypes
    from tinygrad.helpers import GlobalCounters

    compiler_command = os.environ.get("CC", "clang")
    compiler_version = subprocess.run([compiler_command, "--version"], check=True,
                                      capture_output=True, text=True, timeout=10).stdout.splitlines()[0]
    cpu_model = platform.processor() or "unknown"
    cpuinfo = Path("/proc/cpuinfo")
    if cpuinfo.exists():
        cpu_model = next((line.split(":", 1)[1].strip() for line in cpuinfo.read_text().splitlines()
                          if line.startswith("model name")), cpu_model)
    tensor = lambda values: Tensor(values, device="CPU", dtype=dtypes.float32).realize()
    results = []
    for shape in [(1, 3, 4), (3, 5, 7), (8, 16, 9)]:
        case = fixture(*shape)
        expected = reference(**case)
        fw, fb = fold_batchnorm(**{k: v for k, v in case.items() if k != "inputs"})
        x = tensor(case["inputs"])
        w, b, mu, var, gamma, beta = [tensor(case[name]) for name in
                                     ("weights", "bias", "mean", "variance", "gamma", "beta")]
        wf, bf = tensor(fw), tensor(fb)
        eps = case["epsilon"]

        def materialized(inp):
            z = (inp @ w + b).realize()
            normalized = ((z - mu) * (var + eps).rsqrt() * gamma + beta).realize()
            return normalized.relu().realize()

        def lazy(inp):
            z = inp @ w + b
            return ((z - mu) * (var + eps).rsqrt() * gamma + beta).relu().realize()

        def folded(inp):
            return (inp @ wf + bf).relu().realize()

        for name, program in [("materialized", materialized), ("lazy", lazy), ("folded", folded)]:
            fn = TinyJit(program)
            started = time.perf_counter_ns()
            # First calls execute, capture, and replay. Excluded from warm timing.
            for _ in range(3):
                output = fn(x)
                Device["CPU"].synchronize()
            cold_ms = (time.perf_counter_ns() - started) / 1e6
            actual = output.tolist()
            error = max_error(actual, expected)
            scale = max(abs(v) for row in expected for v in row)
            tolerance = 2e-5 + 2e-5 * scale
            if error > tolerance:
                raise AssertionError(f"{name} {shape}: error {error} exceeds {tolerance}")
            # New input values with the same shape must change the result on replay.
            alternate = [[value + 0.25 for value in row] for row in case["inputs"]]
            alternate_result = fn(tensor(alternate)).tolist()
            alternate_error = max_error(alternate_result, reference(**dict(case, inputs=alternate)))
            if alternate_error > tolerance:
                raise AssertionError("JIT replay failed on changed input values")
            samples, counts = [], []
            for _ in range(args.repeats):
                before = GlobalCounters.kernel_count
                started = time.perf_counter_ns()
                fn(x)
                Device["CPU"].synchronize()
                samples.append((time.perf_counter_ns() - started) / 1000)
                counts.append(GlobalCounters.kernel_count - before)
            results.append({"shape_B_K_N": list(shape), "variant": name,
                            "max_absolute_error": error, "max_alternate_input_error": alternate_error,
                            "atol_plus_rtol_max_reference": tolerance,
                            "first_three_calls_ms": cold_ms, "warm_call_us": samples,
                            "median_warm_call_us": statistics.median(samples),
                            "p95_warm_call_us": sorted(samples)[math.ceil(0.95 * len(samples)) - 1],
                            "reported_kernels_per_warm_call": counts})

    directory = Path(__file__).resolve().parent
    report = {"execution_status": "measured CPU compiled execution on synthetic tensors",
              "tinygrad_version": package_version, "python": sys.version.split()[0],
              "platform": platform.platform(), "device": str(Device.DEFAULT),
              "cpu_model": cpu_model, "compiler_command": compiler_command,
              "compiler_version": compiler_version,
              "compiler_target_arguments": getattr(Device["CPU"].compiler, "args", []),
              "fixture_seed": 13,
              "dtype": "float32", "beam": 0, "repeats": args.repeats,
              "timing_boundary": "host TinyJit call plus CPU synchronization; excludes input creation and output copy",
              "hashes": {name: hashlib.sha256((directory / name).read_bytes()).hexdigest()
                         for name in ("reference.py", "optional_tinygrad.py", "requirements-tinygrad.txt")},
              "results": results}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    for row in results:
        print(f"{row['shape_B_K_N']} {row['variant']:12} error={row['max_absolute_error']:.3g} "
              f"median_us={row['median_warm_call_us']:.2f} kernels={sorted(set(row['reported_kernels_per_warm_call']))}")
    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()
