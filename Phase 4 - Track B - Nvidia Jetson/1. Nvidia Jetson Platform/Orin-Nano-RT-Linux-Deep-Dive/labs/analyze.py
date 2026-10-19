"""Inspect a finite timing trace; never interpret observed maxima as WCET."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


def nearest_rank(values: list[int], probability: float) -> int:
    if not values or not 0 < probability <= 1:
        raise ValueError("need samples and 0 < probability <= 1")
    return sorted(values)[math.ceil(probability * len(values)) - 1]


def summarize(trace: dict) -> dict:
    period = trace["period_ns"]
    start = trace["start_ns"]
    samples = trace["samples"]
    if type(period) is not int or period <= 0 or type(start) is not int or not samples:
        raise ValueError("invalid period, start, or empty trace")
    lateness, response, misses = [], [], 0
    last_finish = start
    for index, sample in enumerate(samples):
        release, wake, finish = (sample[key] for key in ("release_ns", "wake_ns", "finish_ns"))
        if any(type(value) is not int for value in (release, wake, finish)):
            raise ValueError("timestamps must be integer nanoseconds")
        if release != start + (index + 1) * period:
            raise ValueError("release times must lie on the nominal periodic grid")
        if wake < release or finish < wake or wake < last_finish:
            raise ValueError("timestamps violate causal ordering")
        lateness.append(wake - release)
        response.append(finish - release)
        misses += finish > release + period
        last_finish = finish
    def stats(values: list[int]) -> dict:
        return {"p50_us": nearest_rank(values, .50) / 1000,
                "p99_us": nearest_rank(values, .99) / 1000,
                "max_observed_us": max(values) / 1000}
    return {"samples": len(samples), "deadline_misses": misses,
            "deadline_miss_fraction": misses / len(samples),
            "wake_lateness": stats(lateness), "completion_response": stats(response),
            "final_wake_lateness_us": lateness[-1] / 1000,
            "interpretation": "finite observations on this host; no WCET or Jetson claim"}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trace", type=Path)
    args = parser.parse_args()
    print(json.dumps(summarize(json.loads(args.trace.read_text())), indent=2))


if __name__ == "__main__":
    main()
