#!/usr/bin/env python3
"""Deterministic sensor/worker queue simulation; no hardware or third-party packages."""

import argparse
import heapq
import json
import math
import platform
import unittest
from collections import Counter, deque
from dataclasses import dataclass, replace
from pathlib import Path


@dataclass(frozen=True)
class Observation:
    epoch: int
    sequence: int
    captured_ms: int
    values: tuple
    unit: str = "m"
    frame: str = "base_link"


class FreshnessGate:
    """Single ordered completion stream, one clock domain, explicit reset epoch."""

    def __init__(self, epoch=1, deadline_ms=40):
        self.epoch = epoch
        self.deadline_ms = deadline_ms
        self.last_sequence = -1

    def check(self, observation, completed_ms):
        if observation.epoch != self.epoch:
            return "wrong_epoch"
        if observation.unit != "m" or observation.frame != "base_link":
            return "wrong_contract"
        if len(observation.values) != 3 or not all(math.isfinite(x) for x in observation.values):
            return "invalid_values"
        if observation.captured_ms > completed_ms:
            return "future_timestamp"
        if observation.sequence <= self.last_sequence:
            return "duplicate_or_reordered"
        self.last_sequence = observation.sequence
        if completed_ms - observation.captured_ms > self.deadline_ms:
            return "stale"
        return "publish"


def nearest_rank(values, percentile):
    """For this small lesson, use sorted[ceil(p*N)-1], not interpolated percentiles."""
    return sorted(values)[math.ceil(percentile * len(values)) - 1] if values else None


def simulate(policy, samples=10, capture_period_ms=10, service_ms=25, deadline_ms=40):
    if policy not in ("fifo", "latest"):
        raise ValueError("unknown queue policy")
    if any(type(v) is not int or v <= 0 for v in
           (samples, capture_period_ms, service_ms, deadline_ms)):
        raise ValueError("counts and time intervals must be positive integers")
    capacity = 3 if policy == "fifo" else 1
    events = []
    for seq in range(samples):
        obs = Observation(1, seq, seq * capture_period_ms, (1.0, 0.0, 0.0))
        # Arrival precedes completion at equal times; the tie rule is part of this fixture.
        heapq.heappush(events, (obs.captured_ms, 0, seq, obs))
    pending = deque()
    busy = False
    gate = FreshnessGate(deadline_ms=deadline_ms)
    trace, dropped = [], []
    high_water = 0
    while events:
        now, event_type, _, obs = heapq.heappop(events)
        if event_type == 0:
            if len(pending) == capacity:
                if policy == "fifo":
                    dropped.append({"sequence": obs.sequence, "reason": "queue_full"})
                    continue
                replaced = pending.popleft()
                dropped.append({"sequence": replaced.sequence, "reason": "replaced_by_newer"})
            pending.append(obs)
            high_water = max(high_water, len(pending))
        else:
            busy = False
            status = gate.check(obs, now)
            trace.append({
                "sequence": obs.sequence, "captured_ms": obs.captured_ms,
                "completed_ms": now, "age_ms": now - obs.captured_ms,
                "status": status,
            })
        if not busy and pending:
            next_obs = pending.popleft()
            busy = True
            heapq.heappush(events, (now + service_ms, 1, next_obs.sequence, next_obs))
    ages = [row["age_ms"] for row in trace]
    published_ages = [row["age_ms"] for row in trace if row["status"] == "publish"]
    return {
        "policy": policy, "generated": samples, "completed": len(trace),
        "status_counts": dict(Counter(row["status"] for row in trace)),
        "dropped": len(dropped), "waiting_capacity": capacity,
        "queue_high_water": high_water,
        "all_completed_age_ms": {"p50": nearest_rank(ages, 0.50),
                                 "p95": nearest_rank(ages, 0.95), "max": max(ages)},
        "published_max_age_ms": max(published_ages) if published_ages else None,
        "trace": trace, "drop_trace": dropped,
    }


class FreshnessTests(unittest.TestCase):
    def setUp(self):
        self.obs = Observation(1, 0, 0, (1.0, 0.0, 0.0))

    def test_deadline_is_inclusive(self):
        self.assertEqual(FreshnessGate().check(self.obs, 40), "publish")
        self.assertEqual(FreshnessGate().check(self.obs, 41), "stale")

    def test_future_timestamp_is_rejected(self):
        self.assertEqual(FreshnessGate().check(replace(self.obs, captured_ms=50), 40),
                         "future_timestamp")

    def test_duplicate_and_reordered_are_rejected(self):
        gate = FreshnessGate()
        self.assertEqual(gate.check(replace(self.obs, sequence=2), 20), "publish")
        self.assertEqual(gate.check(replace(self.obs, sequence=2), 20), "duplicate_or_reordered")
        self.assertEqual(gate.check(self.obs, 20), "duplicate_or_reordered")

    def test_reset_blocks_old_completions(self):
        gate = FreshnessGate(epoch=2)
        self.assertEqual(gate.check(self.obs, 20), "wrong_epoch")
        self.assertEqual(gate.check(replace(self.obs, epoch=2), 20), "publish")

    def test_values_units_and_frames_are_checked(self):
        for obs in (replace(self.obs, values=(1, 2)),
                    replace(self.obs, values=(float("nan"), 0, 0))):
            self.assertEqual(FreshnessGate().check(obs, 20), "invalid_values")
        for obs in (replace(self.obs, unit="mm"), replace(self.obs, frame="camera_link")):
            self.assertEqual(FreshnessGate().check(obs, 20), "wrong_contract")

    def test_accounting_and_freshness_for_both_policies(self):
        for policy in ("fifo", "latest"):
            report = simulate(policy)
            self.assertEqual(report["generated"], report["completed"] + report["dropped"])
            self.assertLessEqual(report["queue_high_water"], report["waiting_capacity"])
            self.assertLessEqual(report["published_max_age_ms"], 40)
        self.assertGreater(simulate("fifo")["status_counts"].get("stale", 0), 0)
        self.assertGreater(simulate("latest")["status_counts"].get("publish", 0),
                           simulate("fifo")["status_counts"].get("publish", 0))

    def test_fast_worker_and_slow_worker(self):
        fast = simulate("latest", service_ms=5)
        self.assertEqual(fast["status_counts"], {"publish": 10})
        self.assertEqual(fast["dropped"], 0)
        slow = simulate("latest", service_ms=45)
        self.assertEqual(slow["status_counts"].get("publish", 0), 0)
        self.assertIsNone(slow["published_max_age_ms"])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--service-ms", type=int, default=25)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.service_ms <= 0:
        parser.error("--service-ms must be positive")
    if args.self_test:
        result = unittest.TextTestRunner(verbosity=2).run(
            unittest.defaultTestLoader.loadTestsFromTestCase(FreshnessTests))
        if not result.wasSuccessful():
            raise SystemExit(1)
    report = {
        "execution_status": "synthetic discrete-event fixture; no wall-clock measurement",
        "python": platform.python_version(),
        "settings": {"samples": 10, "capture_period_ms": 10,
                     "service_ms": args.service_ms, "deadline_ms": 40,
                     "clock_domain": "one simulated monotonic clock"},
        "policies": [simulate(policy, service_ms=args.service_ms) for policy in ("fifo", "latest")],
    }
    rendered = json.dumps(report, indent=2, allow_nan=False) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    else:
        print(rendered, end="")


if __name__ == "__main__":
    main()
