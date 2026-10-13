"""Small offline VLA data/evaluation/timing contracts, not a robot safety layer."""
import math
import random
from statistics import mean


def denormalize(action, center, scale):
    if not action or not len(action) == len(center) == len(scale):
        raise ValueError("nonempty matching dimensions required")
    if any(not math.isfinite(v) for vector in (action, center, scale) for v in vector):
        raise ValueError("nonfinite action or statistics")
    if any(v <= 0 for v in scale):
        raise ValueError("scales must be positive")
    result = [a * s + c for a, c, s in zip(action, center, scale)]
    if any(not math.isfinite(value) for value in result):
        raise ValueError("denormalization overflow")
    return result


def chunk_with_mask(records, start, horizon):
    """Records: {episode, time, action}; right-pad terminal action, mask padding.

    Input episodes must be contiguous and each episode's timestamps strictly
    increasing. The helper validates the full input before slicing. Sampling
    at fixed time intervals and action-frame conversion belong upstream.
    """
    if horizon < 1 or not 0 <= start < len(records):
        raise ValueError("invalid chunk bounds")
    seen = set()
    previous = None
    dimension = len(records[0]["action"])
    for record in records:
        episode, timestamp, action = record["episode"], record["time"], record["action"]
        if not math.isfinite(timestamp) or not action or len(action) != dimension:
            raise ValueError("bad timestamp or dimension")
        if any(not math.isfinite(v) for v in action):
            raise ValueError("nonfinite action")
        if previous is None or episode != previous["episode"]:
            if episode in seen:
                raise ValueError("episodes must be contiguous")
            seen.add(episode)
        elif timestamp <= previous["time"]:
            raise ValueError("timestamps must increase within episode")
        previous = record
    selected = []
    for record in records[start:start + horizon]:
        if record["episode"] != records[start]["episode"]:
            break
        selected.append(list(record["action"]))
    valid = len(selected)
    selected.extend([selected[-1][:] for _ in range(horizon - valid)])
    return selected, [1] * valid + [0] * (horizon - valid)


def assert_disjoint_groups(train_groups, test_groups):
    overlap = set(train_groups) & set(test_groups)
    if overlap:
        raise ValueError(f"split leakage in {len(overlap)} group(s)")


def usable_chunk_indices(observation_time, arrival_time, now, period, horizon, max_age):
    """Offline schedule: omit actions whose intended start already passed.

    All timestamps must use ONE monotonically increasing clock and one epoch.
    Chunk index i targets observation_time + i*period. Future-start actions are
    returned only if observation age at the intended start is <= max_age.
    Revalidate at actual execution if dispatch is late. This deliberately does
    not interpolate or hold a past action.
    """
    values = (observation_time, arrival_time, now, period, max_age)
    if any(not math.isfinite(v) for v in values) or period <= 0 or max_age < 0 or horizon < 1:
        raise ValueError("invalid schedule")
    if observation_time > arrival_time or arrival_time > now:
        raise ValueError("clock order violation")
    if now - observation_time > max_age:
        return []
    return [i for i in range(horizon)
            if observation_time + i * period >= now - 1e-12
            and i * period <= max_age + 1e-12]


def paired_task_bootstrap(rows, samples=2000, seed=17):
    """Task-macro delta and percentile CI, resampling paired task means.

    rows: {task, seed, reference: 0|1, candidate: 0|1}. Fixed seed sets must
    match by construction in each row. This treats observed within-task rates
    as fixed and estimates between-task uncertainty, NOT a hierarchical CI.
    """
    if not rows or samples < 100:
        raise ValueError("need rows and >= 100 bootstrap draws")
    groups, seen = {}, set()
    for row in rows:
        key = (row["task"], row["seed"])
        if key in seen:
            raise ValueError("duplicate task/seed pair")
        seen.add(key)
        if row["reference"] not in (0, 1) or row["candidate"] not in (0, 1):
            raise ValueError("success must be binary")
        groups.setdefault(row["task"], []).append(row["candidate"] - row["reference"])
    deltas = [mean(values) for values in groups.values()]
    if len(deltas) < 2:
        raise ValueError("at least two tasks required for task bootstrap")
    rng = random.Random(seed)
    draws = sorted(mean(rng.choices(deltas, k=len(deltas))) for _ in range(samples))
    return {"macro_delta": mean(deltas), "ci_low": draws[int(0.025 * samples)],
            "ci_high": draws[min(samples - 1, int(0.975 * samples))],
            "tasks": len(deltas), "pairs": len(rows),
            "resampling_unit": "task mean; within-task rates treated as fixed"}
