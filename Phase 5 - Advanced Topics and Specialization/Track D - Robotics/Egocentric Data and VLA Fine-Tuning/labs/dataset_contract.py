"""Dataset schema, statistics and normalization contracts for VLA fine-tuning (pure Python, CPU).

This mirrors the parts of a LeRobot v3.0 dataset that decide whether fine-tuning works:
a feature schema (meta/info.json), per-feature statistics (meta/stats.json: min, max,
mean, std, count and q01..q99 quantiles) and the normalization a policy applies with them.
Stacks disagree on the normalization: openpi uses mean/std for pi0 but q01/q99 quantiles
for pi0-FAST and pi0.5; LeRobot's pi0.5 needs q01/q99 in stats.json; GR00T defaults to
min-max; OpenVLA(-OFT) maps q01..q99 to [-1, 1]. Getting this wrong fails silently.
"""
from __future__ import annotations

import math

QUANTILES = (0.01, 0.10, 0.50, 0.90, 0.99)
# Same wording as LeRobot v0.6.1, so a search for the real error finds this check too.
QUANTILES_ERROR = "QUANTILES normalization mode requires q01 and q99 stats"


# --------------------------------------------------------------------------- schema
def _is_number(v) -> bool:
    # bool is a subclass of int in Python; a True/False in a float vector is a bug, not 1.0/0.0
    return isinstance(v, (int, float)) and not isinstance(v, bool)


def validate_frame(frame: dict, schema: dict) -> list[str]:
    """Return a list of problems (empty = valid). schema: name -> {"dtype", "shape"}."""
    problems = []
    for name in sorted(set(schema) - set(frame)):
        problems.append(f"missing feature {name!r}")
    for name in sorted(set(frame) - set(schema)):
        problems.append(f"unexpected feature {name!r}")
    for name, spec in schema.items():
        if name not in frame:
            continue
        value = frame[name]
        if spec["dtype"] in ("float32", "float64"):
            if _is_number(value) and list(spec["shape"]) == [1]:
                value = [value]            # a scalar feature such as `timestamp` has shape [1]
            if not isinstance(value, (list, tuple)) or len(value) != spec["shape"][0]:
                problems.append(f"{name}: expected shape {spec['shape']}, got {len(value) if isinstance(value, (list, tuple)) else type(value).__name__}")
            elif any(not _is_number(v) or not math.isfinite(v) for v in value):
                problems.append(f"{name}: non-finite or non-numeric value")
        elif spec["dtype"] == "string" and not isinstance(value, str):
            problems.append(f"{name}: expected string")
        elif spec["dtype"] == "int64" and (isinstance(value, bool) or not isinstance(value, int)):
            problems.append(f"{name}: expected int")
    return problems


def check_timestamps(timestamps_s: list[float], fps: float, tolerance_s: float = 1e-4) -> bool:
    """Consecutive frames must be 1/fps apart within tolerance (LeRobot's load-time rule)."""
    return all(abs((b - a) - 1.0 / fps) <= tolerance_s for a, b in zip(timestamps_s, timestamps_s[1:]))


# --------------------------------------------------------------------------- statistics
def _quantile(sorted_vals: list[float], q: float) -> float:
    """Linear-interpolation quantile on sorted data. LeRobot estimates quantiles from
    5000-bin histograms instead, so expect small differences against its stats.json."""
    if not sorted_vals:
        raise ValueError("empty")
    pos = q * (len(sorted_vals) - 1)
    lo = math.floor(pos)
    hi = min(lo + 1, len(sorted_vals) - 1)
    return sorted_vals[lo] + (pos - lo) * (sorted_vals[hi] - sorted_vals[lo])


def feature_stats(vectors: list[list[float]]) -> dict:
    """Per-dimension min, max, mean, std (population), count and quantiles."""
    if not vectors:
        raise ValueError("no data")
    dims = len(vectors[0])
    cols = [[v[d] for v in vectors] for d in range(dims)]
    out = {"count": len(vectors), "min": [], "max": [], "mean": [], "std": []}
    out.update({f"q{int(round(q * 100)):02d}": [] for q in QUANTILES})
    for col in cols:
        s = sorted(col)
        mean = math.fsum(col) / len(col)
        var = math.fsum((x - mean) ** 2 for x in col) / len(col)
        out["min"].append(s[0])
        out["max"].append(s[-1])
        out["mean"].append(mean)
        out["std"].append(math.sqrt(var))
        for q in QUANTILES:
            out[f"q{int(round(q * 100)):02d}"].append(_quantile(s, q))
    return out


def aggregate_stats(parts: list[dict]) -> dict:
    """Merge per-episode stats exactly for count/min/max/mean/std (parallel variance formula).
    Quantiles cannot be merged exactly from summaries: they are count-weighted averages here
    and flagged approximate. Recompute from data (or a histogram) when they matter."""
    total = sum(p["count"] for p in parts)
    dims = len(parts[0]["mean"])
    merged = {"count": total, "min": [], "max": [], "mean": [], "std": [], "quantiles_approximate": True}
    for d in range(dims):
        mean = math.fsum(p["count"] * p["mean"][d] for p in parts) / total
        # E[x^2] = var + mean^2 for each part, combined by counts
        ex2 = math.fsum(p["count"] * (p["std"][d] ** 2 + p["mean"][d] ** 2) for p in parts) / total
        merged["min"].append(min(p["min"][d] for p in parts))
        merged["max"].append(max(p["max"][d] for p in parts))
        merged["mean"].append(mean)
        merged["std"].append(math.sqrt(max(0.0, ex2 - mean * mean)))
    for key in (k for k in parts[0] if k.startswith("q") and k[1:].isdigit()):
        merged[key] = [math.fsum(p["count"] * p[key][d] for p in parts) / total for d in range(dims)]
    return merged


# --------------------------------------------------------------------------- normalization
def normalize(x: list[float], stats: dict, mode: str, eps: float = 1e-8) -> list[float]:
    if mode == "MEAN_STD":
        return [(v - m) / (s + eps) for v, m, s in zip(x, stats["mean"], stats["std"])]
    if mode == "MIN_MAX":          # -> [-1, 1]
        return [2.0 * (v - lo) / (hi - lo + eps) - 1.0 for v, lo, hi in zip(x, stats["min"], stats["max"])]
    if mode == "QUANTILES":        # q01..q99 -> [-1, 1]; outliers fall outside
        if "q01" not in stats or "q99" not in stats:
            raise ValueError(QUANTILES_ERROR)
        return [2.0 * (v - lo) / (hi - lo + eps) - 1.0 for v, lo, hi in zip(x, stats["q01"], stats["q99"])]
    if mode == "IDENTITY":
        return list(x)
    raise ValueError(f"unknown mode {mode}")


def unnormalize(y: list[float], stats: dict, mode: str, eps: float = 1e-8) -> list[float]:
    if mode == "MEAN_STD":
        return [v * (s + eps) + m for v, m, s in zip(y, stats["mean"], stats["std"])]
    if mode in ("MIN_MAX", "QUANTILES"):
        lo_key, hi_key = ("min", "max") if mode == "MIN_MAX" else ("q01", "q99")
        if mode == "QUANTILES" and ("q01" not in stats or "q99" not in stats):
            raise ValueError(QUANTILES_ERROR)
        return [(v + 1.0) / 2.0 * (hi - lo + eps) + lo for v, lo, hi in zip(y, stats[lo_key], stats[hi_key])]
    if mode == "IDENTITY":
        return list(y)
    raise ValueError(f"unknown mode {mode}")


def stats_parity(train: dict, deploy: dict, keys=("mean", "std", "min", "max", "q01", "q99"),
                 rel_tol: float = 1e-6) -> list[str]:
    """Differences between the stats a checkpoint was trained with and the stats a deployment
    loads. Any entry here means the robot receives actions on the wrong scale. q01/q99 are
    compared by default because quantile-normalized stacks (pi0.5, pi0-FAST, OpenVLA-OFT)
    use only those two."""
    issues = []
    for k in keys:
        if k not in train or k not in deploy:
            if (k in train) != (k in deploy):
                issues.append(f"{k}: present in only one of train/deploy")
            continue
        if len(train[k]) != len(deploy[k]):
            issues.append(f"{k}: dimension {len(train[k])} vs {len(deploy[k])}")
            continue
        for d, (a, b) in enumerate(zip(train[k], deploy[k])):
            if not math.isclose(a, b, rel_tol=rel_tol, abs_tol=1e-12):
                issues.append(f"{k}[{d}]: {a} vs {b}")
    return issues
