"""Original synthetic fixture: a point robot with constant braking deceleration.

This is a supervised approximation exercise, not real collision/safety data.
Units: distance m, speed m/s, deceleration m/s^2, response latency s.
"""
from dataclasses import asdict, dataclass, replace
import hashlib
import json
import math
import random
import statistics

FEATURES = ("distance_m", "speed_m_s", "deceleration_m_s2", "latency_s")
MARGIN_M = 0.2


@dataclass(frozen=True)
class Sample:
    episode: int
    distance_m: float
    speed_m_s: float
    deceleration_m_s2: float
    latency_s: float
    stop_required: int

    def features(self):
        return [getattr(self, key) for key in FEATURES]


def stopping_distance(speed, deceleration, latency):
    if not all(math.isfinite(x) for x in (speed, deceleration, latency)):
        raise ValueError("nonfinite physical input")
    if speed < 0 or deceleration <= 0 or latency < 0:
        raise ValueError("need speed/latency >=0 and deceleration >0")
    return speed * latency + speed * speed / (2 * deceleration) + MARGIN_M


def make_dataset(seed=17, episodes=144, frames=24):
    rng, rows = random.Random(seed), []
    for episode in range(episodes):
        deceleration, latency = rng.uniform(0.8, 3.5), rng.uniform(0.05, 0.45)
        for _ in range(frames):
            speed, distance = rng.uniform(0.1, 4.0), rng.uniform(0.2, 8.0)
            stop = int(distance <= stopping_distance(speed, deceleration, latency))
            rows.append(Sample(episode, distance, speed, deceleration, latency, stop))
    return rows


def split_episodes(rows, seed=23):
    ids = sorted({row.episode for row in rows})
    if len(ids) < 12:
        raise ValueError("need at least twelve episodes")
    random.Random(seed).shuffle(ids)
    n = len(ids) // 6
    partitions = (set(ids[:-2*n]), set(ids[-2*n:-n]), set(ids[-n:]))
    result = [[row for row in rows if row.episode in group] for group in partitions]
    validate_splits(result)
    return result


def validate_splits(parts):
    seen = set()
    for rows in parts:
        current = {row.episode for row in rows}
        if not current or seen & current:
            raise ValueError("empty partition or episode leakage")
        seen |= current


def fit_scaler(training_rows):
    if not training_rows:
        raise ValueError("empty training set")
    columns = list(zip(*(row.features() for row in training_rows)))
    return [statistics.fmean(c) for c in columns], [statistics.pstdev(c) or 1.0 for c in columns]


def sensor_shift(rows, distance_bias_m):
    if not math.isfinite(distance_bias_m):
        raise ValueError("nonfinite sensor bias")
    # Observations change; physical ground truth and episode IDs remain fixed.
    return [replace(row, distance_m=row.distance_m + distance_bias_m) for row in rows]


def dataset_sha256(rows):
    encoded = json.dumps([asdict(row) for row in rows], sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()
