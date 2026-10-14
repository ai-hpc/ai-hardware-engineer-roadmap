#!/usr/bin/env python3
"""CPU teaching models; analytical fixtures, not VLA or silicon benchmarks.

Uses decimal GB/GB/s and binary MiB explicitly. No external dependencies.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from decimal import Decimal
import json
import math

MIB = 1024 ** 2


def integer(name: str, value: int, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return value


def finite(name: str, value: float, minimum: float = 0) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be numeric")
    if not math.isfinite(value) or value < minimum:
        raise ValueError(f"{name} must be finite and >= {minimum}")
    return float(value)


def storage_bytes(elements: int, bits: int, group: int | None = None,
                  scale_bits: int = 0, tensor_scale_bytes: int = 0) -> int:
    """One synthetic tensor; partial final groups allocate their full payload.

    Without group, values are bit-packed with a final byte rounded up.
    Does not model row alignment, swizzles, multiple tensors, or repacking.
    """
    integer("elements", elements, 1)
    integer("bits", bits, 1)
    integer("scale_bits", scale_bits)
    integer("tensor_scale_bytes", tensor_scale_bytes)
    if group is None:
        if scale_bits or tensor_scale_bytes:
            raise ValueError("specify group for this lab's scale-bearing layout")
        return (elements * bits + 7) // 8
    integer("group", group, 1)
    blocks = (elements + group - 1) // group
    return blocks * ((group * bits + 7) // 8 + (scale_bits + 7) // 8) + tensor_scale_bytes


def traffic_bytes(weights: int, repeats: int, resident: int = 0,
                  cold: bool = True) -> int:
    """Ideal retained subset; cold pays one fill, warm assumes a valid prior fill."""
    integer("weights", weights, 1)
    integer("repeats", repeats, 1)
    integer("resident", resident)
    if resident > weights:
        raise ValueError("resident bytes exceed the weight allocation")
    if not isinstance(cold, bool):
        raise ValueError("cold must be boolean")
    return repeats * (weights - resident) + (resident if cold else 0)


def memory_fixture(sram_mib: int = 0, reserve_mib: int = 0,
                   bandwidth_gbps: float = 120, cold: bool = True,
                   nvfp4_metadata: bool = False) -> dict:
    integer("sram_mib", sram_mib)
    integer("reserve_mib", reserve_mib)
    bandwidth_gbps = finite("bandwidth_gbps", bandwidth_gbps)
    if bandwidth_gbps == 0 or reserve_mib > sram_mib:
        raise ValueError("bandwidth must be positive; reserve cannot exceed SRAM")
    backbone = storage_bytes(3_000_000_000, 4, 16, 8, 4) if nvfp4_metadata else storage_bytes(3_000_000_000, 4)
    expert = storage_bytes(300_000_000, 8)
    resident = min((sram_mib - reserve_mib) * MIB, expert)
    traffic = backbone + traffic_bytes(expert, 10, resident, cold)
    return {
        "evidence": "analytical synthetic fixture; weight traffic only",
        "layout": "one synthetic 1D NVFP4 backbone tensor" if nvfp4_metadata else "packed payload only; quantization metadata omitted",
        "residency": "cold: one retained-subset fill included" if cold else "warm: retained subset already valid; prior fill excluded",
        "backbone_weight_bytes": backbone,
        "expert_weight_bytes": expert,
        "expert_resident_bytes": resident,
        "weight_capacity_bytes": backbone + expert,
        "weight_traffic_bytes_per_evaluation": traffic,
        "sustained_bandwidth_GB_per_s": bandwidth_gbps,
        "weight_bandwidth_floor_ms": traffic / (bandwidth_gbps * 1e9) * 1000,
        "required_weight_GB_per_s_at_20_Hz": traffic * 20 / 1e9,
        "excluded": ["activation/state/sensor traffic", "bank and port conflicts", "compute", "dispatch", "repacking", "coherence", "power and area"],
    }


def tile_utilization(m: int, n: int, k: int, tm: int, tn: int, tk: int) -> float:
    """Fraction of issued scalar products useful under full padded tile execution."""
    for name, value in zip(("m", "n", "k", "tm", "tn", "tk"), (m, n, k, tm, tn, tk)):
        integer(name, value, 1)
    padded = math.prod(((d + t - 1) // t) * t for d, t in ((m, tm), (n, tn), (k, tk)))
    return m * n * k / padded


@dataclass(frozen=True)
class Job:
    name: str
    release_ms: float
    work_ms: float
    deadline_ms: float


def schedule(jobs: list[Job], quantum_ms: float | None = None,
             dispatch_ms: float = 0) -> dict:
    """One abstract shared server: FIFO whole jobs, or EDF at legal tile boundaries.

    No dependencies, DMA concurrency, Linux scheduling, or hardware preemption
    modeled. Dispatch overhead is charged to each selected segment.
    Timing arithmetic uses the decimal spelling of validated inputs, so binary
    float roundoff cannot delay a release or fabricate an equality-boundary miss.
    """
    if not jobs or len({j.name for j in jobs}) != len(jobs):
        raise ValueError("provide jobs with unique names")
    dispatch_ms = finite("dispatch_ms", dispatch_ms)
    if quantum_ms is not None and finite("quantum_ms", quantum_ms) == 0:
        raise ValueError("quantum must be positive")
    for j in jobs:
        finite("release", j.release_ms)
        if finite("work", j.work_ms) == 0:
            raise ValueError("work must be positive")
        if finite("deadline", j.deadline_ms) < j.release_ms:
            raise ValueError("deadline precedes release")
    times = {j.name: tuple(Decimal(str(v)) for v in (j.release_ms, j.work_ms, j.deadline_ms)) for j in jobs}
    overhead = Decimal(str(dispatch_ms))
    quantum = Decimal(str(quantum_ms)) if quantum_ms is not None else None
    remaining = {j.name: times[j.name][1] for j in jobs}
    finished, timeline = {}, []
    time_ms = Decimal(0)
    while remaining:
        ready = [j for j in jobs if j.name in remaining and times[j.name][0] <= time_ms]
        if not ready:
            time_ms = min(times[j.name][0] for j in jobs if j.name in remaining)
            continue
        job = min(ready, key=lambda j: (times[j.name][2], times[j.name][0], j.name) if quantum is not None else (times[j.name][0], j.name))
        duration = min(remaining[job.name], quantum) if quantum is not None else remaining[job.name]
        start = time_ms
        time_ms += overhead + duration
        timeline.append({"job": job.name, "start_ms": float(start), "end_ms": float(time_ms), "dispatch_ms": dispatch_ms})
        remaining[job.name] -= duration
        if remaining[job.name] == 0:
            del remaining[job.name]
            finished[job.name] = {"finish_ms": float(time_ms), "response_ms": float(time_ms-times[job.name][0]), "missed": time_ms > times[job.name][2]}
    return {"evidence": "abstract shared-resource schedule", "policy": "EDF at tile boundaries" if quantum_ms is not None else "FIFO nonpreemptive", "timeline": timeline, "jobs": finished}


@dataclass(frozen=True)
class Attempt:
    episode: str
    success: bool
    energy_j: float
    capture_ms: float
    usable_ms: float
    consumed_ms: float


def summarize(attempts: list[Attempt], deadline_ms: float,
              max_age_ms: float) -> dict:
    """One record per attempt; all timestamps share a declared common clock.

    One instrumented action per synthetic attempt. Real episodes require
    per-action timing records plus a separate per-episode energy/outcome table.
    """
    if not attempts or len({x.episode for x in attempts}) != len(attempts):
        raise ValueError("provide unique, nonempty attempt records")
    finite("deadline", deadline_ms)
    finite("max_age", max_age_ms)
    for x in attempts:
        if not isinstance(x.success, bool):
            raise ValueError("success must be boolean")
        finite("energy_j", x.energy_j)
        for t in (x.capture_ms, x.usable_ms, x.consumed_ms):
            finite("timestamp", t)
        if not x.capture_ms <= x.usable_ms <= x.consumed_ms:
            raise ValueError("timestamps must be capture <= usable <= consumed")
    latencies = sorted(Decimal(str(x.usable_ms)) - Decimal(str(x.capture_ms)) for x in attempts)
    ages = [Decimal(str(x.consumed_ms)) - Decimal(str(x.capture_ms)) for x in attempts]
    energy = sum(x.energy_j for x in attempts)
    successes = sum(x.success for x in attempts)
    def quantile(q): return float(latencies[math.ceil(q * len(latencies)) - 1])
    return {
        "attempts": len(attempts), "successes": successes,
        "success_rate": successes / len(attempts),
        "total_compute_system_energy_j": energy,
        "energy_per_attempt_j": energy / len(attempts),
        "energy_per_success_j": energy / successes if successes else None,
        "energy_per_success_status": "defined" if successes else "undefined: no successful task",
        "latency_ms": {"p50": quantile(.5), "p95": quantile(.95), "p99": quantile(.99), "max": float(max(latencies))},
        "deadline_misses": sum(t > Decimal(str(deadline_ms)) for t in latencies),
        "stale_at_consumption": sum(t > Decimal(str(max_age_ms)) for t in ages),
        "quantile_method": "nearest rank; tiny fixture does not characterize production tails",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    mem = commands.add_parser("memory", help="reproduce the proposal's traffic calculation")
    mem.add_argument("--sram-mib", type=int, default=0)
    mem.add_argument("--reserve-mib", type=int, default=0)
    mem.add_argument("--bandwidth-gbps", type=float, default=120)
    mem.add_argument("--warm", action="store_true")
    mem.add_argument("--nvfp4-metadata", action="store_true")
    commands.add_parser("sweep", help="sweep ideal residency and sustained bandwidth")
    commands.add_parser("tiles", help="compare padded tile arithmetic, not achieved occupancy")
    sched = commands.add_parser("schedule", help="compare blocking with legal scheduling boundaries")
    sched.add_argument("--dispatch-ms", type=float, default=0)
    commands.add_parser("evaluate", help="include failed attempts, freshness, and all-attempt energy")
    args = parser.parse_args()
    try:
        if args.command == "memory":
            result = memory_fixture(args.sram_mib, args.reserve_mib, args.bandwidth_gbps, not args.warm, args.nvfp4_metadata)
        elif args.command == "sweep":
            result = [memory_fixture(s, 8, b) for s in (16, 32, 64) for b in (60, 120, 180)]
        elif args.command == "tiles":
            result = [{"m": m, "tile_m": t, "useful_padded_fraction": tile_utilization(m, 256, 256, t, 64, 64)} for m in (8, 257) for t in (16, 32, 64)]
        elif args.command == "schedule":
            jobs = [Job("vision", 0, 18, 40), Job("action", 2, 3, 8)]
            result = [schedule(jobs, dispatch_ms=args.dispatch_ms), schedule(jobs, 3, args.dispatch_ms)]
        else:
            result = summarize([Attempt("a", True, 10, 0, 35, 40), Attempt("b", False, 12, 100, 160, 190), Attempt("c", True, 8, 200, 245, 260)], 50, 70)
    except ValueError as error:
        parser.error(str(error))
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
