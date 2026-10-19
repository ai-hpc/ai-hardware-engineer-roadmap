#!/usr/bin/env python3
"""Train the existing tiny policy, then simulate capture-to-action timing.

Integer simulation timestamps are not measured inference latency. No hardware or
network is used. The fallback assumes an independent current velocity encoder.
"""
import argparse
from collections import deque
from dataclasses import asdict, dataclass
import hashlib
import json
import math
from pathlib import Path
import platform
import random
import statistics

import policy_lab


@dataclass(frozen=True)
class Observation:
    session: int
    sequence: int
    captured_ms: int
    position_m: float
    velocity_m_s: float


@dataclass(frozen=True)
class Timing:
    name: str
    inference_ms: int
    queue_mode: str = "latest"
    check_age: bool = True
    max_age_ms: int = 250
    transport_ms: int = 20
    sensor_period_ms: int = 50
    dt_ms: int = 10
    duration_ms: int = 20_000
    queue_capacity: int = 64
    outage_start_ms: int = 3_000
    outage_end_ms: int = 3_500

    def __post_init__(self):
        if self.queue_mode not in ("latest", "fifo"):
            raise ValueError("queue_mode must be latest or fifo")
        for name in ("inference_ms", "max_age_ms", "sensor_period_ms", "dt_ms",
                     "duration_ms", "queue_capacity"):
            value = getattr(self, name)
            if type(value) is not int or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        for name in ("transport_ms", "outage_start_ms", "outage_end_ms"):
            value = getattr(self, name)
            if type(value) is not int or value < 0:
                raise ValueError(f"{name} must be a nonnegative integer")
        if self.outage_end_ms < self.outage_start_ms:
            raise ValueError("outage end precedes start")
        for name in ("inference_ms", "max_age_ms", "transport_ms", "sensor_period_ms",
                     "duration_ms", "outage_start_ms", "outage_end_ms"):
            if getattr(self, name) % self.dt_ms:
                raise ValueError(f"{name} must align with dt_ms in this discrete fixture")


class CommandGate:
    def __init__(self, session, max_age_ms=250, check_age=True):
        if type(session) is not int or session < 0:
            raise ValueError("session must be a nonnegative integer")
        if type(max_age_ms) is not int or max_age_ms <= 0:
            raise ValueError("max_age_ms must be a positive integer")
        self.session, self.max_age_ms, self.check_age = session, max_age_ms, check_age
        self.last_sequence = -1
        self.observation = None
        self.acceleration = 0.0

    def accept(self, observation, now_ms, acceleration):
        """Rejected input does not renew the previous command's expiry."""
        if type(now_ms) is not int or now_ms < 0:
            raise ValueError("now_ms must be a nonnegative integer")
        if type(observation.session) is not int or observation.session != self.session:
            return "wrong_session"
        if type(observation.sequence) is not int or observation.sequence <= self.last_sequence:
            return "old_sequence"
        if type(observation.captured_ms) is not int or observation.captured_ms < 0:
            return "invalid_timestamp"
        if observation.captured_ms > now_ms:
            return "future_timestamp"
        if self.observation and observation.captured_ms < self.observation.captured_ms:
            return "backward_timestamp"
        if not all(math.isfinite(value) for value in (
                observation.position_m, observation.velocity_m_s, acceleration)):
            return "nonfinite"
        if self.check_age and now_ms - observation.captured_ms > self.max_age_ms:
            return "stale"
        self.last_sequence = observation.sequence
        self.observation = observation
        self.acceleration = max(-1.0, min(1.0, acceleration))
        return "accepted"

    def action_for_interval(self, now_ms, dt_ms, current_velocity):
        if type(now_ms) is not int or now_ms < 0 or type(dt_ms) is not int or dt_ms <= 0:
            raise ValueError("invalid integration interval")
        if not math.isfinite(current_velocity):
            raise ValueError("fallback encoder velocity must be finite")
        valid = self.observation is not None
        if valid and now_ms < self.observation.captured_ms:
            raise ValueError("time moved before the accepted observation")
        if valid and self.check_age:
            valid = now_ms + dt_ms <= self.observation.captured_ms + self.max_age_ms
        if valid:
            return self.acceleration, "policy"
        # Local damping is an explicit assumed capability, not a universal safe stop.
        return max(-1.0, min(1.0, -2.0 * current_velocity)), "fallback"


def simulate(policy, timing, initial=(1.0, 0.0), session=0, retain_trace=False):
    if len(initial) != 2 or not all(math.isfinite(v) for v in initial):
        raise ValueError("initial position and velocity must be finite")
    position, velocity = initial
    gate = CommandGate(session, timing.max_age_ms, timing.check_age)
    transport, pending = deque(), deque()
    active = None
    sequence = 0
    accepted = rejected = discarded = fallback_steps = clipped = completed = 0
    max_queue = 0
    completion_ages = []
    applied_ages = []
    applied_sequences = set()
    trace = []
    max_abs_position = abs(position)
    for now in range(0, timing.duration_ms, timing.dt_ms):
        if now % timing.sensor_period_ms == 0:
            if not timing.outage_start_ms <= now < timing.outage_end_ms:
                observation = Observation(session, sequence, now, position, velocity)
                transport.append((now + timing.transport_ms, observation))
            sequence += 1
        while transport and transport[0][0] <= now:
            _, observation = transport.popleft()
            if timing.queue_mode == "latest":
                discarded += len(pending)
                pending.clear()
                pending.append(observation)
            elif len(pending) < timing.queue_capacity:
                pending.append(observation)
            else:
                discarded += 1  # FIFO drop-new behavior is part of the stated fixture.
        max_queue = max(max_queue, len(pending))
        if active is not None and active[0] <= now:
            _, observation = active
            acceleration = policy(-observation.position_m, observation.velocity_m_s)
            result = gate.accept(observation, now, acceleration)
            completion_ages.append(now - observation.captured_ms)
            completed += 1
            if result == "accepted":
                accepted += 1
                clipped += abs(acceleration) > 1.0
            else:
                rejected += 1
            active = None
        if active is None and pending:
            observation = pending.popleft()
            active = (now + timing.inference_ms, observation)
        acceleration, source = gate.action_for_interval(now, timing.dt_ms, velocity)
        if source == "fallback":
            fallback_steps += 1
            age = None
        else:
            age = now - gate.observation.captured_ms
            # Include the end of the held interval in the freshness evidence.
            applied_ages.append(age + timing.dt_ms)
            applied_sequences.add(gate.observation.sequence)
        velocity += acceleration * timing.dt_ms / 1000
        position += velocity * timing.dt_ms / 1000  # semi-implicit Euler
        max_abs_position = max(max_abs_position, abs(position))
        if retain_trace:
            trace.append({"time_ms": now + timing.dt_ms, "position_m": position,
                          "velocity_m_s": velocity, "acceleration_m_s2": acceleration,
                          "source": source, "observation_age_start_ms": age,
                          "pending_observations": len(pending)})
    report = {"session": session, "initial_position_m": initial[0],
              "initial_velocity_m_s": initial[1], "final_position_m": position,
              "final_velocity_m_s": velocity,
              "success": abs(position) < 0.1 and abs(velocity) < 0.1,
              "max_abs_position_m": max_abs_position,
              "completed_inferences": completed, "accepted_commands": accepted,
              "distinct_commands_applied": len(applied_sequences),
              "applied_policy_intervals": len(applied_ages),
              "rejected_commands": rejected, "discarded_observations": discarded,
              "max_pending_observations": max_queue, "clipped_actions": clipped,
              "fallback_fraction": fallback_steps / (timing.duration_ms // timing.dt_ms),
              "max_completion_age_ms": max(completion_ages, default=None),
              "max_applied_age_ms": max(applied_ages, default=None)}
    if retain_trace:
        report["trace"] = trace
    return report


def scenarios():
    return [Timing("fast_latest", 10, transport_ms=0),
            Timing("slow_fifo_without_age_check", 120, queue_mode="fifo", check_age=False),
            Timing("slow_fifo_with_age_check", 120, queue_mode="fifo"),
            Timing("slow_latest_with_age_check", 120),
            Timing("over_budget_latest", 350)]


def run(seed=7, episodes=12, retain_trace=True):
    if type(episodes) is not int or not 1 <= episodes <= 100:
        raise ValueError("episodes must be an integer from 1 to 100")
    trained = policy_lab.run(seed=seed)
    weights = trained["weights_normalized_inputs"]
    scaler = (trained["scaler_training_mean"], trained["scaler_training_std"])
    policy = lambda error, velocity: policy_lab.predict(weights, error, velocity, scaler)
    rng = random.Random(501)
    initials = [(rng.uniform(-1.4, 1.4), rng.uniform(-0.3, 0.3)) for _ in range(episodes)]
    results = []
    for timing in scenarios():
        rows = [simulate(policy, timing, initial, i, retain_trace and i == 0)
                for i, initial in enumerate(initials)]
        results.append({"timing": asdict(timing), "successes": sum(row["success"] for row in rows),
                        "episodes": episodes,
                        "mean_abs_final_position_m": statistics.fmean(abs(row["final_position_m"]) for row in rows),
                        "per_episode": rows})
    return {"scope": "Actual CPU training and numerical simulation; imposed timing, no hardware",
            "python": platform.python_version(), "dataset_seed": seed, "rollout_seed": 501,
            "training_dataset_sha256": trained["dataset_sha256"],
            "split_episode_ids": trained["split_episode_ids"],
            "weights_normalized_inputs": weights, "scaler_mean": scaler[0], "scaler_std": scaler[1],
            "held_out_action_rmse_m_s2": trained["test"]["pooled_action_rmse_m_s2"],
            "held_out_zero_action_rmse_m_s2": trained["test_zero_baseline"]["pooled_action_rmse_m_s2"],
            "selection": "Weights/rate selected on validation demonstrations; fixed timing scenarios",
            "success_contract": "At 20 seconds: |position| < 0.1 m and |velocity| < 0.1 m/s; terminal condition only",
            "success_caveat": "Fallback alone can satisfy the terminal condition; inspect applied_policy_intervals and distinct_commands_applied, not only accepted_commands, before attributing success to the model",
            "assumptions": ["Perfect instantaneous position/velocity capture", "One inference worker",
                "20-second semi-implicit Euler cart; no inertia model beyond integrated acceleration",
                "Independent current velocity encoder available to fallback damping",
                "Fixed 500 ms sensor outage; no network or physical actuator"],
            "source_sha256": {name: hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
                              for name in ("policy_lab.py", "timed_policy_lab.py")},
            "scenarios": results}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--episodes", type=int, default=12)
    parser.add_argument("--output", type=Path, default=Path("timed-policy-report.json"))
    parser.add_argument("--no-trace", action="store_true")
    args = parser.parse_args()
    if not 1 <= args.episodes <= 100:
        parser.error("episodes must be from 1 to 100")
    report = run(args.seed, args.episodes, not args.no_trace)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(f"Held-out action RMSE: {report['held_out_action_rmse_m_s2']:.6f} m/s^2")
    for row in report["scenarios"]:
        print(f"{row['timing']['name']}: {row['successes']}/{row['episodes']} successful episodes; "
              f"mean |final position|={row['mean_abs_final_position_m']:.4f} m")
    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()
