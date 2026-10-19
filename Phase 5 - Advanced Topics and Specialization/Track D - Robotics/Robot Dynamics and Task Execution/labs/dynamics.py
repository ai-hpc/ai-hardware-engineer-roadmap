"""One actuated point-mass pendulum; angles start at the downward vertical."""
from dataclasses import dataclass
import math


def finite(value, name):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"{name} must be finite")
    return value


@dataclass(frozen=True)
class Pendulum:
    mass: float = 2.0
    length: float = 0.5
    damping: float = 0.2
    gravity: float = 9.81

    def __post_init__(self):
        for name in ("mass", "length", "gravity"):
            if finite(getattr(self, name), name) <= 0:
                raise ValueError(f"{name} must be positive")
        if finite(self.damping, "damping") < 0:
            raise ValueError("damping must be nonnegative")

    @property
    def inertia(self):
        return self.mass * self.length ** 2

    def gravity_torque(self, angle):
        return self.mass * self.gravity * self.length * math.sin(finite(angle, "angle"))

    def energy(self, angle, velocity):
        finite(angle, "angle")
        finite(velocity, "velocity")
        return (0.5 * self.inertia * velocity ** 2
                + self.mass * self.gravity * self.length * (1 - math.cos(angle)))

    def acceleration(self, angle, velocity, applied_torque):
        finite(velocity, "velocity")
        finite(applied_torque, "applied_torque")
        return (applied_torque - self.damping * velocity - self.gravity_torque(angle)) / self.inertia


def required_motor_current(joint_torque, reduction, efficiency, torque_constant):
    """Ideal static transmission relation; excludes motor/gear dynamics and friction."""
    finite(joint_torque, "joint_torque")
    for value, name in ((reduction, "reduction"), (efficiency, "efficiency"),
                        (torque_constant, "torque_constant")):
        if finite(value, name) <= 0:
            raise ValueError(f"{name} must be positive")
    if efficiency > 1:
        raise ValueError("efficiency must not exceed one")
    return joint_torque / (reduction * efficiency * torque_constant)


def pd_rollout(model, target, *, compensate=True, torque_limit=12.0,
               kp=8.0, kd=3.8, dt=0.005, steps=1200, initial=(0.0, 0.0)):
    """Semi-implicit Euler fixture; no latency, contacts, gears, or motor dynamics."""
    finite(target, "target")
    for value, name in ((torque_limit, "torque_limit"), (kp, "kp"), (kd, "kd"), (dt, "dt")):
        if finite(value, name) <= 0:
            raise ValueError(f"{name} must be positive")
    if type(steps) is not int or not 1 <= steps <= 100000:
        raise ValueError("steps must be an integer in 1..100000")
    angle, velocity = initial
    finite(angle, "initial angle")
    finite(velocity, "initial velocity")
    trace = []
    for step in range(steps):
        requested = kp * (target - angle) - kd * velocity
        if compensate:
            requested += model.gravity_torque(angle)
        applied = max(-torque_limit, min(torque_limit, requested))
        acceleration = model.acceleration(angle, velocity, applied)
        velocity += acceleration * dt
        angle += velocity * dt
        if not math.isfinite(angle) or not math.isfinite(velocity):
            raise ValueError("integration diverged")
        trace.append({"time_s": (step + 1) * dt, "angle_rad": angle,
                      "velocity_rad_s": velocity, "requested_torque_Nm": requested,
                      "applied_torque_Nm": applied})
    return trace
