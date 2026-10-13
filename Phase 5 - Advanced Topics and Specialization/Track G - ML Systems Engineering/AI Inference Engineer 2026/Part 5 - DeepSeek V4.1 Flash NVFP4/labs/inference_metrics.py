#!/usr/bin/env python3
"""Summarize finite client request logs; TPOT percentiles are not ITL percentiles."""
import argparse
import json
import math
from pathlib import Path


def finite(value, name):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(name + " must be a finite number")
    return value


def percentile(values, probability):
    """Nearest-rank percentile, or None for no observations."""
    if not values:
        return None
    return sorted(values)[max(0, math.ceil(probability * len(values)) - 1)]


def summarize(rows, ttft_slo_s, tpot_slo_s):
    for value, name in ((ttft_slo_s, "TTFT SLO"), (tpot_slo_s, "TPOT SLO")):
        if finite(value, name) <= 0:
            raise ValueError(name + " must be positive")
    measured, ids = [], set()
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError("each JSONL row must be an object")
        if row.get("phase") not in ("warmup", "measurement"):
            raise ValueError("every request must declare phase")
        request_id = row.get("request_id")
        if not isinstance(request_id, str) or not request_id or request_id in ids:
            raise ValueError("request_id must be a unique nonempty string")
        ids.add(request_id)
        if row["phase"] == "warmup":
            continue
        if row.get("status") not in ("ok", "error", "timeout"):
            raise ValueError("status must be ok, error, or timeout")
        start = finite(row.get("start_s"), "start_s")
        end = finite(row.get("end_s"), "end_s")
        if end < start:
            raise ValueError("end_s cannot precede start_s")
        count = row.get("output_tokens")
        if type(count) is not int or count < 0:
            raise ValueError("output_tokens must be a nonnegative integer")
        first = row.get("first_token_s")
        if first is not None:
            first = finite(first, "first_token_s")
            if not start <= first <= end:
                raise ValueError("first_token_s outside request interval")
        if row["status"] == "ok" and (count < 1 or first is None):
            raise ValueError("completed request needs at least one token and first_token_s")
        measured.append(row)
    if not measured:
        raise ValueError("no measured requests")
    window = max(r["end_s"] for r in measured) - min(r["start_s"] for r in measured)
    if window <= 0:
        raise ValueError("measurement window must be positive")
    ttfts, tpots, completed_tokens, good_tokens, good_requests = [], [], 0, 0, 0
    successes = 0
    for row in measured:
        if row["status"] != "ok":
            continue
        successes += 1
        n = row["output_tokens"]
        ttft = row["first_token_s"] - row["start_s"]
        tpot = (row["end_s"] - row["first_token_s"]) / (n - 1) if n > 1 else None
        ttfts.append(ttft)
        if tpot is not None:
            tpots.append(tpot)
        completed_tokens += n
        if ttft <= ttft_slo_s and (tpot is None or tpot <= tpot_slo_s):
            good_tokens += n
            good_requests += 1
    return {
        "measurement_requests": len(measured), "completed_requests": successes,
        "failed_requests": len(measured) - successes,
        "failure_rate": (len(measured) - successes) / len(measured),
        "window_s": window, "completed_output_tokens": completed_tokens,
        "completed_output_tokens_per_s": completed_tokens / window,
        "slo_passing_requests": good_requests, "good_output_tokens_per_s": good_tokens / window,
        "ttft_samples": len(ttfts), "tpot_samples": len(tpots),
        "ttft_p50_s": percentile(ttfts, .50), "ttft_p95_s": percentile(ttfts, .95),
        "request_average_tpot_p50_s": percentile(tpots, .50),
        "request_average_tpot_p95_s": percentile(tpots, .95),
        "percentile_method": "nearest rank; successful requests only; failures reported separately",
    }


def demo_rows():
    return [
        {"request_id": "warm", "phase": "warmup"},
        {"request_id": "a", "phase": "measurement", "status": "ok",
         "start_s": 0.0, "first_token_s": .2, "end_s": .8, "output_tokens": 4},
        {"request_id": "b", "phase": "measurement", "status": "ok",
         "start_s": 1.0, "first_token_s": 1.5, "end_s": 2.0, "output_tokens": 2},
        {"request_id": "c", "phase": "measurement", "status": "timeout",
         "start_s": 2.0, "first_token_s": None, "end_s": 3.0, "output_tokens": 0},
    ]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--input", type=Path)
    group.add_argument("--demo", action="store_true")
    parser.add_argument("--ttft-slo-s", type=float, default=.4)
    parser.add_argument("--tpot-slo-s", type=float, default=.3)
    args = parser.parse_args()
    try:
        rows = demo_rows() if args.demo else [
            json.loads(line) for line in args.input.read_text().splitlines() if line.strip()]
        result = summarize(rows, args.ttft_slo_s, args.tpot_slo_s)
    except (OSError, ValueError) as exc:
        parser.error(str(exc))
    result["execution_status"] = "synthetic fixture" if args.demo else "input log analysis"
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
