#!/usr/bin/env python3
"""Synthetic energy accounting and a one-node thermal model; no device access."""
from __future__ import annotations

import json
import math
from dataclasses import dataclass


@dataclass(frozen=True)
class Sample:
    seconds: float
    watts: float


def finite(value: float, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"{name} must be a finite number")
    return float(value)


def integrate(samples: list[Sample], *, maximum_gap_s: float) -> dict:
    """Trapezoidal integral for one measurement boundary, with gap rejection."""
    gap = finite(maximum_gap_s, "maximum gap")
    if gap <= 0 or len(samples) < 2:
        raise ValueError("positive maximum gap and at least two samples required")
    points = [(finite(s.seconds, "timestamp"), finite(s.watts, "power")) for s in samples]
    if any(p < 0 for _, p in points):
        raise ValueError("this consumption-only contract rejects negative power")
    energy = 0.0
    for (t0, p0), (t1, p1) in zip(points, points[1:]):
        dt = t1 - t0
        # Account only for representation/subtraction roundoff at the inputs'
        # resolution. This is not permission to bridge a missing meter sample.
        roundoff = 4 * max(math.ulp(t0), math.ulp(t1), math.ulp(gap))
        if roundoff >= gap or not math.isfinite(dt) or dt <= 0 or dt > gap + roundoff:
            raise ValueError("timestamps must increase without an excessive gap")
        energy += (p0 + p1) * 0.5 * dt
    duration = points[-1][0] - points[0][0]
    return {"duration_s": duration, "energy_j": energy, "average_w": energy / duration}


def per_success(energy_j: float, attempted: int, valid_on_time: int) -> dict:
    energy = finite(energy_j, "energy")
    if energy < 0 or any(type(n) is not int for n in (attempted, valid_on_time)):
        raise ValueError("nonnegative energy and integer counts required")
    if not 0 < valid_on_time <= attempted:
        raise ValueError("at least one success, with successes <= attempts, required")
    return {"attempted": attempted, "valid_on_time": valid_on_time,
            "success_fraction": valid_on_time / attempted,
            "joules_per_valid_on_time_result": energy / valid_on_time}


def incremental_energy(total_j: float, idle_w: float, duration_s: float) -> float:
    values = [finite(v, n) for v, n in
              ((total_j, "total energy"), (idle_w, "idle power"), (duration_s, "duration"))]
    if any(v < 0 for v in values):
        raise ValueError("nonnegative energy, power and duration required")
    # Preserve negative results: they indicate a mismatched/noisy baseline,
    # not free energy. Do not silently clip or substitute this for total energy.
    return values[0] - values[1] * values[2]


def thermal_step(initial_c: float, ambient_c: float, power_w: float,
                 resistance_c_per_w: float, capacity_j_per_c: float,
                 duration_s: float) -> float:
    """Exact constant-power solution of C*dT/dt = P - (T-Ta)/R."""
    t0, ta, p, r, c, dt = [finite(v, n) for v, n in
                         ((initial_c, "initial temperature"), (ambient_c, "ambient"),
                          (power_w, "power"), (resistance_c_per_w, "resistance"),
                          (capacity_j_per_c, "capacity"), (duration_s, "duration"))]
    if p < 0 or r <= 0 or c <= 0 or dt < 0:
        raise ValueError("require P >= 0, R > 0, C > 0 and time >= 0")
    equilibrium = ta + p * r
    return equilibrium + (t0 - equilibrium) * math.exp(-dt / (r * c))


def demonstration() -> dict:
    samples = [Sample(0, 6), Sample(1, 10), Sample(3, 10), Sample(4, 6)]
    first = integrate(samples, maximum_gap_s=2)
    second = integrate([Sample(0, 7), Sample(4, 7)], maximum_gap_s=4)
    return {
        "evidence_kind": "synthetic arithmetic, not measured Jetson power",
        "boundary": "hypothetical complete system DC input",
        "samples": [vars(s) for s in samples],
        "configuration_a": {**first, **per_success(first["energy_j"], 200, 180),
                            "idle_baseline_w": 4,
                            "incremental_energy_j": incremental_energy(first["energy_j"], 4,
                                                                        first["duration_s"])},
        "configuration_b": {**second, **per_success(second["energy_j"], 200, 100)},
        "thermal": {"ambient_c": 35, "power_w": 15, "resistance_c_per_w": 2,
                    "capacity_j_per_c": 40, "time_constant_s": 80,
                    "equilibrium_c": 65,
                    "temperature_after_80_s_c": thermal_step(35, 35, 15, 2, 40, 80),
                    "temperature_after_240_s_c": thermal_step(35, 35, 15, 2, 40, 240)},
    }


if __name__ == "__main__":
    print(json.dumps(demonstration(), indent=2, allow_nan=False))
