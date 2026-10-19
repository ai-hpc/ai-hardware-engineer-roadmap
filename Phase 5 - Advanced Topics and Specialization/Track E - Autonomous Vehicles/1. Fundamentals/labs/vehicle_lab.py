"""Finite CPU vehicle/replay teaching models. No hardware or network interfaces."""
from dataclasses import asdict, dataclass, replace
import argparse
import hashlib
import json
import math
from pathlib import Path


def finite(*values):
    return all(isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x) for x in values)


def clamp(x, low, high):
    return max(low, min(high, x))


@dataclass(frozen=True)
class State:
    x: float = 0.0
    y: float = 1.0
    yaw: float = 0.0
    speed: float = 5.0


def bicycle_step(state, steering, acceleration, dt, wheelbase=2.8):
    """Rear axle reference, x forward/y left; held road-wheel angle and acceleration.

    Integrate distance exactly for constant acceleration, then its constant-curvature
    arc. Braking stops at zero speed: this forward-only model cannot reverse.
    No tire slip, load transfer, road banking, steering dynamics or collision model.
    """
    if not finite(*asdict(state).values(), steering, acceleration, dt, wheelbase):
        raise ValueError("finite numeric inputs required")
    if state.speed < 0 or dt <= 0 or wheelbase <= 0 or abs(steering) >= math.pi / 2:
        raise ValueError("invalid model domain")
    moving_dt = min(dt, -state.speed / acceleration) if acceleration < 0 else dt
    distance = state.speed * moving_dt + .5 * acceleration * moving_dt ** 2
    speed = max(0., state.speed + acceleration * dt)
    curvature = math.tan(steering) / wheelbase
    angle = curvature * distance
    # Midpoint/sinc expression avoids cancellation for nearly straight travel.
    sinc = math.sin(angle / 2) / (angle / 2) if angle else 1.
    dx = distance * sinc * math.cos(state.yaw + angle / 2)
    dy = distance * sinc * math.sin(state.yaw + angle / 2)
    return State(state.x + dx, state.y + dy, state.yaw + angle, speed)


def pure_pursuit(state, target, wheelbase=2.8):
    """Road-wheel radians toward a world-frame target ahead of the rear axle."""
    if len(target) != 2 or not finite(*asdict(state).values(), *target, wheelbase) or wheelbase <= 0:
        raise ValueError("invalid geometry")
    dx, dy = target[0] - state.x, target[1] - state.y
    forward = math.cos(state.yaw) * dx + math.sin(state.yaw) * dy
    left = -math.sin(state.yaw) * dx + math.cos(state.yaw) * dy
    if forward <= 0:
        raise ValueError("target must be ahead in the vehicle frame")
    return math.atan2(2 * wheelbase * left, forward * forward + left * left)


def stopping_distance(speed, delay, deceleration):
    if not finite(speed, delay, deceleration) or speed < 0 or delay < 0 or deceleration <= 0:
        raise ValueError("invalid stopping parameters")
    return speed * delay + speed * speed / (2 * deceleration)


def time_to_collision(gap, ego_speed, lead_speed):
    if not finite(gap, ego_speed, lead_speed) or gap < 0 or min(ego_speed, lead_speed) < 0:
        raise ValueError("invalid relative-motion parameters")
    if gap == 0:
        return 0.
    closing = ego_speed - lead_speed
    return gap / closing if closing > 0 else None


@dataclass(frozen=True)
class Observation:
    frame_id: int
    capture_s: float
    ready_s: float
    state: State
    clock: str = "simulation-epoch-1"
    frame: str = "world_xy_yaw_rear_axle"
    schema: str = "course-state-v1"


class ObservationGate:
    """Track accepted acquisition identity, not just fresh publication time."""
    def __init__(self, max_age=.15):
        if not finite(max_age) or max_age <= 0:
            raise ValueError("max_age must be positive")
        self.max_age = max_age
        self.last_frame = -1
        self.last_capture = -math.inf

    def check(self, observation, now):
        o = observation
        if not finite(now, o.capture_s, o.ready_s, *asdict(o.state).values()):
            return "nonfinite"
        if not isinstance(o.frame_id, int) or isinstance(o.frame_id, bool) or o.frame_id < 0:
            return "invalid_frame_id"
        if o.clock != "simulation-epoch-1":
            return "wrong_clock"
        if o.frame != "world_xy_yaw_rear_axle" or o.schema != "course-state-v1":
            return "wrong_schema_or_frame"
        if o.state.speed < 0:
            return "invalid_state"
        if o.capture_s > now + 1e-12 or o.ready_s < o.capture_s - 1e-12:
            return "invalid_time_order"
        if o.ready_s > now + 1e-12:
            return "not_ready"
        if now - o.capture_s > self.max_age + 1e-12:
            return "stale"
        if o.frame_id <= self.last_frame or o.capture_s < self.last_capture:
            return "out_of_order"
        self.last_frame, self.last_capture = o.frame_id, o.capture_s
        return "accepted"


MODEL_CONTRACT = {
    "model": "course-linear-steering-v1", "input_names": ["lateral_error_m", "heading_error_rad"],
    "input_shape": [1, 2], "dtype": "float32", "output_shape": [1, 1],
    "output_units": "road_wheel_rad", "frame": "rear_axle_flu", "max_abs_output": .35,
}


def decode_model_output(output, manifest):
    """A deliberately small independent deployment boundary, not openpilot's schema."""
    if manifest != MODEL_CONTRACT:
        raise ValueError("model/preprocessing/units contract mismatch")
    if not isinstance(output, list) or len(output) != 1 or not isinstance(output[0], list) or len(output[0]) != 1:
        raise ValueError("expected output [1,1]")
    value = output[0][0]
    if not finite(value) or abs(value) > manifest["max_abs_output"]:
        raise ValueError("invalid or out-of-domain steering prediction")
    return value


def replay_faults():
    good = Observation(10, .95, .98, State())
    cases = [
        ("fresh", good, 1.),
        ("fresh_publish_old_capture", replace(good, frame_id=11, capture_s=.6, ready_s=.99), 1.),
        ("future_capture", replace(good, frame_id=11, capture_s=1.02, ready_s=1.03), 1.),
        ("wrong_clock", replace(good, frame_id=11, clock="wall-clock"), 1.),
        ("duplicate_frame", good, 1.01),
        ("wrong_frame", replace(good, frame_id=11, frame="camera_optical"), 1.),
        ("nonfinite_state", replace(good, frame_id=11, state=State(y=math.nan)), 1.),
        ("not_ready", replace(good, frame_id=11, ready_s=1.1), 1.),
        ("recovery", replace(good, frame_id=12, capture_s=1.01, ready_s=1.02), 1.03),
    ]
    gate = ObservationGate()
    return [{"case": name, "frame_id": o.frame_id, "age_s": now-o.capture_s,
             "result": gate.check(o, now)} for name, o, now in cases]


def simulate(delay=0., seconds=8., dt=.05):
    if not finite(delay, seconds, dt) or delay < 0 or not 0 < dt <= .1 or not dt <= seconds <= 60:
        raise ValueError("bounded simulation requires delay >=0, 0<dt<=.1 and dt<=seconds<=60")
    if round(seconds/dt)>20000:
        raise ValueError("at most 20000 simulation steps are supported")
    state, gate, steering = State(), ObservationGate(), 0.
    queue, trace = [], []
    for tick in range(round(seconds / dt)):
        now = tick * dt
        queue.append(Observation(tick, now, now + delay, state))
        available = [o for o in queue if o.ready_s <= now + 1e-12]
        queue = [o for o in queue if o.ready_s > now + 1e-12]
        observation = max(available, key=lambda o: o.frame_id) if available else None
        result = gate.check(observation, now) if observation else "missing"
        if result == "accepted":
            desired = pure_pursuit(observation.state, (observation.state.x + 8., 0.))
            acceleration = clamp(6. - observation.state.speed, -3., 2.)
        else:
            # Classroom fallback only: bounded braking with steering toward zero.
            # This is not a vehicle's validated minimal-risk maneuver.
            desired, acceleration = 0., -3.
        desired = clamp(desired, -.35, .35)
        steering += clamp(desired - steering, -.5 * dt, .5 * dt)
        lateral_accel = state.speed ** 2 * math.tan(steering) / 2.8
        state = bicycle_step(state, steering, acceleration, dt)
        trace.append({"time_s": (tick + 1) * dt, **asdict(state), "steering_rad": steering,
                      "acceleration_m_s2": acceleration, "lateral_accel_m_s2": lateral_accel,
                      "observation_status": result})
    return {"delay_s": delay, "dt_s": dt, "steps": len(trace),
            "rms_lateral_error_m": math.sqrt(sum(t['y']**2 for t in trace)/len(trace)),
            "final_state": asdict(state),
            "accepted": sum(t['observation_status'] == 'accepted' for t in trace),
            "stale": sum(t['observation_status'] == 'stale' for t in trace),
            "max_abs_lateral_accel_m_s2": max(abs(t['lateral_accel_m_s2']) for t in trace),
            "trace": trace}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--delay', type=float, default=0.)
    parser.add_argument('--seconds', type=float, default=8.)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    report = {"scope": "CPU teaching models; no CARLA, openpilot or vehicle execution",
              "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              "stopping_distance_10m_s_100ms_4m_s2_m": stopping_distance(10., .1, 4.),
              "closing_ttc_20m_10_vs_6m_s_s": time_to_collision(20., 10., 6.),
              "fault_replay": replay_faults(), "simulation": simulate(args.delay, args.seconds)}
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
    summary = {**report, "simulation": {k:v for k,v in report['simulation'].items() if k != 'trace'}}
    print(json.dumps(summary, indent=2, allow_nan=False))


if __name__ == '__main__':
    main()
