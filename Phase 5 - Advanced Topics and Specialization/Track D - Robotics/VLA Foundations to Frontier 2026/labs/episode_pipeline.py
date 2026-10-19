"""Inspectable 2-D dataset conversion and offline action dispatch; no robot I/O.

Time is integer microseconds in an explicit episode clock. Source commands are
piecewise constant velocities or interval deltas converted to such velocities.
The latter is an assumption, not information recoverable from arbitrary logs.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path


SCHEMA = "base_xy_delta_m/v1"
CLOCK = "simulation_us"
PERIOD_US = 20_000


def require(condition, message):
    if not condition:
        raise ValueError(message)


def integer(value, name, minimum=0):
    require(type(value) is int and value >= minimum, f"invalid {name}")
    return value


def vector(value, name):
    require(isinstance(value, (tuple, list)) and len(value) == 2, f"invalid {name} shape")
    require(all(type(x) in (int, float) and math.isfinite(x) for x in value),
            f"nonfinite or nonnumeric {name}")
    return tuple(float(x) for x in value)


def digest(value):
    data = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(data.encode()).hexdigest()


def validate_source(source):
    require(source["mode"] in ("interval_delta", "velocity"), "unknown action mode")
    require(source["unit"] == ("mm" if source["mode"] == "interval_delta" else "m/s"),
            "source mode/unit mismatch")
    require(source["frame"] in ("camera_xy", "base_xy"), "unknown source frame")
    integer(source["period_us"], "source period", 1)
    require(source["interpolation"] == "constant_velocity_within_interval",
            "resampling requires an explicit interpolation assumption")
    r = source["rotation_to_base"]
    require(isinstance(r, list) and len(r) == 2, "rotation shape")
    r = [vector(row, "rotation row") for row in r]
    require(all(abs(sum(r[k][i] * r[k][j] for k in range(2)) - (i == j)) < 1e-10
                for i in range(2) for j in range(2)), "rotation is not orthonormal")
    require(abs(r[0][0] * r[1][1] - r[0][1] * r[1][0] - 1) < 1e-10,
            "rotation must preserve handedness")
    if source["frame"] == "base_xy":
        require(r == [(1.0, 0.0), (0.0, 1.0)], "base frame must use identity rotation")
    return r


def causal_frame(frames, decision_us, max_age_us):
    """A captured image is usable only after its arrival in this same clock."""
    candidates = [f for f in frames
                  if f["capture_us"] <= decision_us and f["ready_us"] <= decision_us]
    if not candidates:
        return None
    frame = max(candidates, key=lambda f: f["capture_us"])
    return frame if decision_us - frame["capture_us"] <= max_age_us else None


def convert_episode(episode, source, target_period_us=20_000, max_age_us=25_000):
    rotation = validate_source(source)
    integer(target_period_us, "target period", 1)
    integer(max_age_us, "observation age bound")
    require(episode["clock"] == CLOCK, "unknown episode clock")
    require(isinstance(episode["instruction"], str) and episode["instruction"].strip(),
            "instruction must be a nonempty string")
    steps, frames = episode["steps"], episode["frames"]
    require(len(steps) >= 2, "need initial and final observations")
    require(type(episode["terminal"]) is bool, "terminal flag must be Boolean")
    period = source["period_us"]
    for i, step in enumerate(steps):
        integer(step["time_us"], "step time")
        require(step["time_us"] == i * period, "source grid mismatch or missing step")
        require(type(step["is_last"]) is bool and step["is_last"] == (i == len(steps) - 1),
                "last observation marker mismatch")
        vector(step["state_base_m"], "state")
        if step["is_last"]:
            require(step["action"] is None, "final observation has no action target")
        else:
            vector(step["action"], "source action")
    seen, previous_capture = set(), -1
    for frame in frames:
        integer(frame["id"], "frame ID")
        integer(frame["capture_us"], "capture timestamp")
        integer(frame["ready_us"], "arrival timestamp")
        require(frame["id"] not in seen, "duplicate frame ID")
        require(previous_capture < frame["capture_us"] <= frame["ready_us"],
                "camera timestamps must be ordered and causal")
        require(frame["clock"] == episode["clock"], "camera clock mismatch")
        rgb = frame["rgb_2x2"]
        require(isinstance(rgb, list) and len(rgb) == 12 and
                all(type(x) is int and 0 <= x <= 255 for x in rgb), "RGB fixture shape/range")
        seen.add(frame["id"])
        previous_capture = frame["capture_us"]
    duration = steps[-1]["time_us"]
    require(duration % target_period_us == 0, "partial final target interval is unsupported")
    require(duration // target_period_us <= 10_000, "fixture exceeds bounded conversion size")
    velocities = []
    for step in steps[:-1]:
        values = vector(step["action"], "source action")
        factor = 1_000 / period if source["mode"] == "interval_delta" else 1.0
        velocities.append(tuple(sum(rotation[d][k] * values[k] * factor for k in range(2))
                                for d in range(2)))
    rows = []
    for start in range(0, duration, target_period_us):
        end = start + target_period_us
        delta = [0.0, 0.0]
        for index in range(start // period, (end - 1) // period + 1):
            overlap = min(end, (index + 1) * period) - max(start, index * period)
            for d in range(2):
                delta[d] += velocities[index][d] * overlap / 1_000_000
        frame = causal_frame(frames, start, max_age_us)
        # State uses a causal hold. Interpolating using a later measurement would
        # leak future state into this observation, even if command labels may
        # legitimately integrate future demonstrated actions.
        state = steps[start // period]["state_base_m"]
        delta = list(vector(delta, "converted delta"))
        rows.append({"episode": episode["id"], "session": episode["session"],
                     "origin": episode["origin"], "time_us": start, "action": delta,
                     "schema": SCHEMA, "period_us": target_period_us,
                     "instruction": episode["instruction"],
                     "terminated": end == duration and episode["terminal"],
                     "truncated": end == duration and not episode["terminal"],
                     "state_base_m": list(state), "observation_valid": frame is not None,
                     "frame_id": frame["id"] if frame else None,
                     "rgb_2x2": list(frame["rgb_2x2"]) if frame else None,
                     "frame_age_us": start - frame["capture_us"] if frame else None})
    return rows


def convert_bundle(bundle):
    require(bundle["schema"] == "course_episode_fixture/v1", "unknown fixture schema")
    ids, origins, splits, converted = set(), {}, {}, []
    for episode in bundle["episodes"]:
        require(episode["id"] not in ids, "duplicate episode ID")
        ids.add(episode["id"])
        split = episode["split"]
        require(split in ("train", "test"), "unknown split")
        require(episode["session"] not in splits or splits[episode["session"]] == split,
                "session leaks between splits")
        require(episode["origin"] not in origins, "duplicated source episode provenance")
        splits[episode["session"]] = split
        origins[episode["origin"]] = episode["id"]
        rows = convert_episode(episode, bundle["sources"][episode["source"]])
        converted.extend(dict(row, split=split) for row in rows)
    require(converted, "empty dataset")
    return converted


def fit_normalizer(rows):
    training = [row for row in rows if row["split"] == "train" and row["observation_valid"]]
    require(training, "no valid training examples")
    require(all(row["schema"] == SCHEMA and row["period_us"] == PERIOD_US for row in training),
            "normalizer requires the canonical action schema and cadence")
    actions = [vector(row["action"], "training action") for row in training]
    center = [sum(a[d] for a in actions) / len(actions) for d in range(2)]
    std = [math.sqrt(sum((a[d] - center[d]) ** 2 for a in actions) / len(actions))
           for d in range(2)]
    normalizer = {"schema": SCHEMA, "period_us": PERIOD_US, "center": center,
                  "scale": [s if s > 1e-12 else 1.0 for s in std],
                  "constant": [s <= 1e-12 for s in std], "count": len(actions),
                  "training_rows_sha256": digest(training)}
    return dict(normalizer, sha256=digest(normalizer))


def normalize(action, stats):
    return [(a - m) / s for a, m, s in zip(vector(action, "action"), stats["center"], stats["scale"])]


def denormalize(action, stats):
    require(stats["sha256"] == digest({k: v for k, v in stats.items() if k != "sha256"}),
            "normalizer content/hash mismatch")
    require(stats["schema"] == SCHEMA, "normalizer action schema mismatch")
    center, scale = vector(stats["center"], "center"), vector(stats["scale"], "scale")
    require(all(s > 0 for s in scale), "normalizer scale must be positive")
    return vector([a * s + m for a, m, s in zip(vector(action, "normalized action"), center, scale)],
                  "decoded action")


def chunk(rows, index, horizon):
    integer(index, "chunk index")
    integer(horizon, "horizon", 1)
    require(index < len(rows) and horizon <= 100, "chunk bounds")
    selected = []
    for row in rows[index:index + horizon]:
        if row["episode"] != rows[index]["episode"]:
            break
        selected.append(list(row["action"]))
    valid = len(selected)
    return selected + [selected[-1][:] for _ in range(horizon - valid)], [1] * valid + [0] * (horizon - valid)


def masked_mse(predictions, targets, mask):
    require(len(predictions) == len(targets) == len(mask), "loss shape mismatch")
    require(all(type(m) is int and m in (0, 1) for m in mask) and any(mask), "invalid or empty mask")
    numerator = 0.0
    for prediction, target, valid in zip(predictions, targets, mask):
        prediction, target = vector(prediction, "prediction"), vector(target, "target")
        numerator += valid * sum((p - t) ** 2 for p, t in zip(prediction, target))
    return numerator / (2 * sum(mask))


class Executor:
    """Offline exact-start dispatcher, with an explicitly zero-delta fallback.

    There is no interpolation, physical controller, continuity enforcement, or
    RTC flow inpainting. A new result replaces only unexecuted future targets.
    """
    def __init__(self, normalizer, max_age_us=100_000):
        self.normalizer = normalizer
        self.max_age_us = integer(max_age_us, "maximum age")
        self.epoch = 0
        self.reset()

    def reset(self):
        self.epoch += 1
        self.now = self.last_dispatch = self.last_request = self.last_observation = -1
        self.queue = {}

    def clock(self, now):
        integer(now, "dispatch time")
        require(now >= self.now, "clock moved backward without a reset")
        self.now = now

    def submit(self, message, now):
        self.clock(now)
        integer(message["epoch"], "reset epoch", 1)
        require(message["epoch"] == self.epoch, "old or unknown reset epoch")
        require(message["clock"] == CLOCK and message["schema"] == SCHEMA, "clock/schema mismatch")
        require(message["normalizer_sha256"] == self.normalizer["sha256"], "wrong normalizer artifact")
        request = integer(message["request"], "request ID")
        obs = integer(message["observation_us"], "observation timestamp")
        ready = integer(message["ready_us"], "ready timestamp")
        period = integer(message["period_us"], "action period", 1)
        require(period == self.normalizer["period_us"] == PERIOD_US, "action cadence mismatch")
        require(obs <= ready <= now, "future observation/result or reversed clock order")
        require(request > self.last_request and obs >= self.last_observation, "out-of-order result")
        actions = message["normalized_actions"]
        require(1 <= len(actions) <= 32, "bounded action horizon required")
        physical = [denormalize(a, self.normalizer) for a in actions]
        require(all(abs(x) <= 0.05 for a in physical for x in a), "fixture delta bound exceeded")
        indices = [i for i in range(len(actions)) if obs + i * period >= now
                   and obs + i * period > self.last_dispatch and i * period <= self.max_age_us]
        require(indices, "fully expired chunk")
        # Validate the complete candidate before mutating accepted identity or queue.
        candidate = {t: v for t, v in self.queue.items() if t >= now}
        for i in indices:
            candidate[obs + i * period] = (request, obs, i, physical[i])
        require(len(candidate) <= 32, "action queue capacity exceeded")
        self.queue = candidate
        self.last_request, self.last_observation = request, obs
        return indices

    def dispatch(self, now):
        self.clock(now)
        require(now > self.last_dispatch, "duplicate dispatch tick")
        missed = sum(t < now for t in self.queue)
        self.queue = {t: v for t, v in self.queue.items() if t >= now}
        item = self.queue.pop(now, None)
        self.last_dispatch = now
        if item is None or now - item[1] > self.max_age_us:
            return {"time_us": now, "status": "fallback", "action_m": [0.0, 0.0], "missed": missed}
        request, obs, index, action = item
        return {"time_us": now, "status": "execute", "action_m": list(action),
                "request": request, "index": index, "age_us": now - obs, "missed": missed}


def message(stats, epoch=1, request=1, observation_us=0, ready_us=70_000, action=(0.0, 0.01)):
    return {"epoch": epoch, "request": request, "observation_us": observation_us,
            "ready_us": ready_us, "period_us": 20_000, "clock": CLOCK, "schema": SCHEMA,
            "normalizer_sha256": stats["sha256"],
            "normalized_actions": [normalize(action, stats) for _ in range(8)]}


def replay(stats):
    executor, trace = Executor(stats), []
    first = message(stats)
    trace.append({"event": "late_result", "accepted_indices": executor.submit(first, 70_000)})
    trace.extend(executor.dispatch(t) for t in (70_000, 80_000))
    second = message(stats, request=2, observation_us=60_000, ready_us=85_000, action=(0.0, 0.02))
    trace.append({"event": "replacement", "accepted_indices": executor.submit(second, 85_000)})
    try:
        executor.submit(first, 90_000)
    except ValueError as error:
        trace.append({"event": "old_result", "rejection": str(error)})
    trace.append(executor.dispatch(100_000))
    executor.reset()
    try:
        executor.submit(second, 0)
    except ValueError as error:
        trace.append({"event": "reset_then_old_result", "rejection": str(error)})
    zero = message(stats, epoch=2, ready_us=10_000, action=(0.0, 0.0))
    trace.append({"event": "zero_action", "accepted_indices": executor.submit(zero, 10_000)})
    trace.extend(executor.dispatch(t) for t in (20_000, 60_000, 120_000))
    return trace


def run(path):
    raw = path.read_bytes()
    rows = convert_bundle(json.loads(raw))
    stats = fit_normalizer(rows)
    targets, mask = chunk(rows, 2, 4)
    predictions = [[0.0, 0.0] for _ in targets]
    return {"scope": "synthetic RGB episode conversion and offline exact-start action dispatch",
            "fixture_sha256": hashlib.sha256(raw).hexdigest(), "rows": rows,
            "normalizer": stats, "boundary_chunk": {"targets": targets, "mask": mask},
            "masked_zero_prediction_mse_m2": masked_mse(predictions, targets, mask),
            "trace": replay(stats)}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixture", type=Path, default=Path(__file__).with_name("episode-fixture.json"))
    args = parser.parse_args()
    print(json.dumps(run(args.fixture), indent=2, allow_nan=False))
