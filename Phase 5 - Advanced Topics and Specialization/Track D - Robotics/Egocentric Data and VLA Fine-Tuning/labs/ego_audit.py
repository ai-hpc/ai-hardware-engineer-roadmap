"""Audit egocentric episode metadata before it reaches a training set (pure Python, CPU).

An "episode" here is a dict such as:
  {"episode_id": "e001", "participant": "p07", "location": "kitchen_3",
   "timestamps_s": [...], "verb": "pour", "noun": "kettle",
   "content_hash": "sha256 of the decoded frames"}
The checks mirror what goes wrong in practice: broken clocks, duplicated clips,
participants or rooms leaking across splits, and a label distribution with a long tail.
"""
from __future__ import annotations

import math
import statistics
from collections import Counter, defaultdict


def timestamp_report(timestamps_s: list[float], fps: float, tolerance_s: float = 1e-4) -> dict:
    """Frame-timing check in the spirit of LeRobot's load-time `tolerance_s` test:
    consecutive frames must be 1/fps apart within tolerance. Also catches
    non-monotonic clocks (a common symptom of mixing host and device time)."""
    period = 1.0 / fps
    gaps = [b - a for a, b in zip(timestamps_s, timestamps_s[1:])]
    non_monotonic = sum(1 for g in gaps if g <= 0.0)
    off_period = sum(1 for g in gaps if abs(g - period) > tolerance_s)
    dropped = sum(max(0, round(g / period) - 1) for g in gaps if g > 0.0)
    return {
        "frames": len(timestamps_s),
        "duration_s": (timestamps_s[-1] - timestamps_s[0]) if len(timestamps_s) > 1 else 0.0,
        "non_monotonic": non_monotonic,
        "off_period": off_period,
        "estimated_dropped_frames": dropped,
        "ok": non_monotonic == 0 and off_period == 0,
    }


def duration_stats(episodes: list[dict], fps: float) -> dict:
    durations = [len(e["timestamps_s"]) / fps for e in episodes]
    if not durations:
        return {"count": 0}
    durations.sort()
    return {
        "count": len(durations),
        "total_hours": sum(durations) / 3600.0,
        "mean_s": statistics.fmean(durations),
        "median_s": statistics.median(durations),
        "min_s": durations[0],
        "max_s": durations[-1],
    }


def find_duplicates(episodes: list[dict]) -> list[list[str]]:
    """Groups of episode ids that share a content hash (re-uploads, overlapping clips)."""
    groups: dict[str, list[str]] = defaultdict(list)
    for e in episodes:
        groups[e["content_hash"]].append(e["episode_id"])
    return [sorted(ids) for ids in groups.values() if len(ids) > 1]


def split_leakage(splits: dict[str, list[dict]], key: str) -> dict[str, list[str]]:
    """Values of `key` (participant, location, kitchen...) that appear in more than one split.
    Any non-empty result means the test set is not measuring generalization to new people
    or places, whatever the episode ids say."""
    seen: dict[str, set[str]] = defaultdict(set)
    for split_name, eps in splits.items():
        for e in eps:
            seen[e[key]].add(split_name)
    return {value: sorted(names) for value, names in seen.items() if len(names) > 1}


def group_split(episodes: list[dict], key: str, fractions: dict[str, float], seed: int = 0) -> dict[str, list[dict]]:
    """Assign whole groups (all episodes sharing `key`) to one split, so no group leaks.
    Deterministic: groups are ordered by a seeded hash, then filled to the target fractions."""
    import hashlib

    groups: dict[str, list[dict]] = defaultdict(list)
    for e in episodes:
        groups[e[key]].append(e)
    order = sorted(groups, key=lambda g: hashlib.sha256(f"{seed}:{g}".encode()).hexdigest())
    total = len(episodes)
    names = list(fractions)
    out: dict[str, list[dict]] = {n: [] for n in names}
    idx = 0
    for g in order:
        # put the group in the first split still below its target size
        while idx < len(names) - 1 and len(out[names[idx]]) >= fractions[names[idx]] * total:
            idx += 1
        out[names[idx]].extend(groups[g])
    return out


def label_distribution(episodes: list[dict], field: str) -> dict:
    counts = Counter(e[field] for e in episodes)
    n = sum(counts.values())
    probs = [c / n for c in counts.values()]
    entropy = -sum(p * math.log2(p) for p in probs if p > 0)
    ranked = counts.most_common()
    head = ranked[: max(1, len(ranked) // 10)]
    return {
        "classes": len(counts),
        "entropy_bits": entropy,
        "max_entropy_bits": math.log2(len(counts)) if counts else 0.0,
        "top10pct_share": sum(c for _, c in head) / n if n else 0.0,
        "singletons": sum(1 for c in counts.values() if c == 1),
        "counts": dict(ranked),
    }


def cohens_kappa(labels_a: list[str], labels_b: list[str]) -> float:
    """Agreement between two annotators beyond chance. 1 = perfect, 0 = chance level."""
    if len(labels_a) != len(labels_b) or not labels_a:
        raise ValueError("need two equal-length, non-empty label lists")
    n = len(labels_a)
    observed = sum(a == b for a, b in zip(labels_a, labels_b)) / n
    ca, cb = Counter(labels_a), Counter(labels_b)
    expected = sum(ca[k] * cb.get(k, 0) for k in ca) / (n * n)
    if expected == 1.0:
        return 1.0
    return (observed - expected) / (1.0 - expected)
