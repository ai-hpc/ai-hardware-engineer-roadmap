"""Atomic observation/action protocol for the simulator-only push exercise.

The simulator owns target integration, last-applied action, epoch and monotonic time.
The policy echoes identifiers; it never compares clocks from different processes.
"""
from dataclasses import dataclass
import hashlib
import json
import math

JOINT_NAMES = tuple(f"panda_joint{i}" for i in range(1, 8)) + (
    "panda_finger_joint1", "panda_finger_joint2")
ARM_NAMES = JOINT_NAMES[:7]


class Rejected(ValueError):
    pass


def integer(value, label):
    if type(value) is not int or value < 0:
        raise Rejected(label + " must be a nonnegative integer")
    return value


def vector(value, length, label):
    if not isinstance(value, (list, tuple)) or len(value) != length:
        raise Rejected(label + " has the wrong shape")
    result = []
    for x in value:
        try:
            valid = type(x) in (int, float) and math.isfinite(x)
        except OverflowError:
            valid = False
        if not valid:
            raise Rejected(label + " must contain finite numbers")
        result.append(float(x))
    return tuple(result)


def strict_loads(raw):
    def unique(pairs):
        result = {}
        for k, value in pairs:
            if k in result:
                raise Rejected("duplicate JSON key")
            result[k] = value
        return result

    def nonfinite(_):
        raise Rejected("nonfinite JSON constant")

    try:
        value = json.loads(raw, object_pairs_hook=unique, parse_constant=nonfinite)
    except (ValueError, TypeError) as exc:
        raise Rejected("invalid JSON") from exc
    if type(value) is not dict:
        raise Rejected("message must be a JSON object")
    return value


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def observation(value):
    fields = {"version", "episode", "sequence", "issued_ns", "expires_ns", "frame",
              "names", "q", "qd", "cube", "goal", "target", "last_action"}
    if type(value) is not dict or set(value) != fields:
        raise Rejected("observation schema")
    if type(value["version"]) is not int or value["version"] != 1:
        raise Rejected("observation version")
    for label in ("episode", "sequence", "issued_ns", "expires_ns"):
        integer(value[label], label)
    if value["expires_ns"] <= value["issued_ns"]:
        raise Rejected("nonpositive observation lifetime")
    if (value["frame"] != "base" or not isinstance(value["names"], (list, tuple))
            or tuple(value["names"]) != JOINT_NAMES):
        raise Rejected("frame or joint order")
    for label, length in (("q", 9), ("qd", 9), ("cube", 3), ("goal", 3),
                          ("target", 7), ("last_action", 7)):
        vector(value[label], length, label)
    if any(abs(x) > 1 for x in value["last_action"]):
        raise Rejected("last action outside [-1,1]")
    return value


def digest(obs):
    return hashlib.sha256(canonical(observation(obs)).encode()).hexdigest()


def policy_observation(obs, q0):
    """The 31-value feed-forward policy contract; last_action means last applied."""
    observation(obs)
    q0 = vector(q0, 9, "q0")
    return ([q - zero for q, zero in zip(obs["q"], q0)]
            + [.1 * v for v in obs["qd"]] + list(obs["cube"]) + list(obs["goal"])
            + list(obs["last_action"]))


def command_for(obs, action):
    observation(obs)
    return {"version": 1, "episode": obs["episode"], "sequence": obs["sequence"],
            "issued_ns": obs["issued_ns"], "observation_sha256": digest(obs),
            "names": list(ARM_NAMES), "action": list(vector(action, 7, "action"))}


@dataclass(frozen=True)
class Applied:
    target: tuple
    action: tuple
    age_ns: int


class CommandGate:
    def __init__(self, lower, upper, scale=.05):
        self.lower, self.upper = vector(lower, 7, "lower"), vector(upper, 7, "upper")
        self.scale = vector([scale], 1, "scale")[0]
        if self.scale <= 0 or any(lo >= hi for lo, hi in zip(self.lower, self.upper)):
            raise Rejected("invalid scale or limits")
        self.episode = -1
        self.sequence = -1
        self.pending = None

    def reset(self, episode, target):
        integer(episode, "episode")
        values = vector(target, 7, "target")
        if episode <= self.episode:
            raise Rejected("episode must increase across resets")
        if any(not lo <= q <= hi for q, lo, hi in zip(values, self.lower, self.upper)):
            raise Rejected("reset target exceeds limits")
        self.episode, self.sequence = episode, -1
        self.target, self.last_action, self.pending = values, (0.0,) * 7, None

    def issue(self, q, qd, cube, goal, now_ns, ttl_ns):
        integer(now_ns, "now_ns")
        integer(ttl_ns, "ttl_ns")
        if self.episode < 0 or ttl_ns == 0:
            raise Rejected("reset first and use positive ttl")
        candidate = {"version": 1, "episode": self.episode, "sequence": self.sequence+1,
                     "issued_ns": now_ns, "expires_ns": now_ns+ttl_ns, "frame": "base",
                     "names": list(JOINT_NAMES), "q": list(vector(q, 9, "q")),
                     "qd": list(vector(qd, 9, "qd")), "cube": list(vector(cube, 3, "cube")),
                     "goal": list(vector(goal, 3, "goal")), "target": list(self.target),
                     "last_action": list(self.last_action)}
        observation(candidate)
        self.sequence += 1
        # Own a serialized snapshot: a caller mutating its message cannot change the gate.
        self.pending = strict_loads(canonical(candidate))
        return candidate

    def accept(self, command, now_ns):
        integer(now_ns, "now_ns")
        expected = {"version", "episode", "sequence", "issued_ns", "observation_sha256", "names", "action"}
        if type(command) is not dict or set(command) != expected:
            raise Rejected("command schema")
        if type(command["version"]) is not int or command["version"] != 1:
            raise Rejected("command version")
        for key in ("episode", "sequence", "issued_ns"):
            integer(command[key], key)
        if command["episode"] != self.episode:
            raise Rejected("previous or unknown episode")
        if self.pending is None:
            raise Rejected("no pending observation (duplicate or unsolicited command)")
        obs = self.pending
        if (command["sequence"] != obs["sequence"] or command["issued_ns"] != obs["issued_ns"]
                or command["observation_sha256"] != digest(obs)):
            raise Rejected("command does not answer the pending observation")
        if not obs["issued_ns"] <= now_ns <= obs["expires_ns"]:
            raise Rejected("future or expired command")
        if not isinstance(command["names"], (list, tuple)) or tuple(command["names"]) != ARM_NAMES:
            raise Rejected("command joint order")
        action = vector(command["action"], 7, "action")
        if any(abs(a) > 1 for a in action):
            raise Rejected("action exceeds normalized limits")
        target = tuple(max(lo, min(hi, q + self.scale*a))
                       for q, a, lo, hi in zip(self.target, action, self.lower, self.upper))
        result = Applied(target, action, now_ns - obs["issued_ns"])
        self.target, self.last_action, self.pending = target, action, None
        return result

    def expire(self):
        """Hold the last bounded target and discard the request. No new action is integrated."""
        self.pending = None
