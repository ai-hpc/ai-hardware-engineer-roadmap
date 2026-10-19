#!/usr/bin/env python3
"""Optional MuJoCo 3.16 CPU crosscheck of the point-mass pendulum lesson."""
import argparse
from dataclasses import dataclass, asdict
import hashlib
import json
import math
from pathlib import Path
import platform

from dynamics import Pendulum, finite, pd_rollout


@dataclass(frozen=True)
class SphericalBob(Pendulum):
    """Reference adaptation: same COM/gravity, plus a solid sphere's own inertia."""
    radius: float = .01

    def __post_init__(self):
        super().__post_init__()
        if finite(self.radius, "radius") <= 0:
            raise ValueError("radius must be positive")

    @property
    def inertia(self):
        return super().inertia + .4 * self.mass * self.radius ** 2


def make_engine(plant, *, dt=.005, radius=.01, torque_limit=12., implicit_damping=True):
    import mujoco
    for name, value in (("dt", dt), ("radius", radius), ("torque_limit", torque_limit)):
        if finite(value, name) <= 0:
            raise ValueError(f"{name} must be positive")
    if type(implicit_damping) is not bool:
        raise ValueError("implicit_damping must be boolean")
    xml = f'''<mujoco model="course_pendulum">
      <option timestep="{dt:.17g}" gravity="0 0 {-plant.gravity:.17g}" integrator="Euler">
        <flag eulerdamp="{'enable' if implicit_damping else 'disable'}"/>
      </option>
      <worldbody><body name="pendulum">
        <joint name="hinge" type="hinge" axis="0 1 0" damping="{plant.damping:.17g}"/>
        <geom name="bob" type="sphere" pos="0 0 {-plant.length:.17g}"
          size="{radius:.17g}" mass="{plant.mass:.17g}" contype="0" conaffinity="0"/>
      </body></worldbody>
      <actuator><motor joint="hinge" gear="1" ctrllimited="true"
        ctrlrange="{-torque_limit:.17g} {torque_limit:.17g}" forcelimited="true"
        forcerange="{-torque_limit:.17g} {torque_limit:.17g}"/></actuator>
    </mujoco>'''
    model = mujoco.MjModel.from_xml_string(xml)
    if (model.nq, model.nv, model.nu) != (1, 1, 1):
        raise ValueError("expected a single actuated hinge")
    return model, mujoco.MjData(model), xml


def check_state(data):
    import mujoco
    warnings = [mujoco.mjtWarning(i).name for i, item in enumerate(data.warning) if item.number]
    if warnings or not all(math.isfinite(float(v)) for v in (data.qpos[0], data.qvel[0])):
        raise ValueError("invalid MuJoCo episode: " + ", ".join(warnings))


def reset(model, data, initial):
    import mujoco
    if len(initial) != 2:
        raise ValueError("initial state is (angle_rad, velocity_rad_s)")
    for value in initial:
        finite(value, "initial state")
    mujoco.mj_resetData(model, data)
    data.qpos[0], data.qvel[0] = initial
    mujoco.mj_forward(model, data)
    check_state(data)


def joint_inertia(model, data):
    import mujoco
    import numpy as np
    dense = np.empty((model.nv, model.nv))
    # MuJoCo 3.16 signature; older examples taking data.qM use a different API.
    mujoco.mj_fullM(model, data, dense)
    return float(dense[0, 0])


def native_rollout(plant, target, *, compensate=True, torque_limit=12., kp=8., kd=3.8,
                   dt=.005, steps=1200, initial=(0., 0.), radius=.01, implicit_damping=True):
    import mujoco
    finite(target, "target")
    for name, value in (("kp", kp), ("kd", kd)):
        if finite(value, name) <= 0:
            raise ValueError(f"{name} must be positive")
    if type(steps) is not int or not 1 <= steps <= 100000:
        raise ValueError("steps must be an integer in 1..100000")
    model, data, xml = make_engine(plant, dt=dt, radius=radius, torque_limit=torque_limit,
                                  implicit_damping=implicit_damping)
    reset(model, data, initial)
    inertia = joint_inertia(model, data)
    trace = []
    for step in range(steps):
        angle, velocity = float(data.qpos[0]), float(data.qvel[0])
        requested = kp * (target - angle) - kd * velocity
        if compensate:
            requested += plant.gravity_torque(angle)
        finite(requested, "requested torque")
        data.ctrl[0] = max(-torque_limit, min(torque_limit, requested))
        mujoco.mj_step(model, data)
        check_state(data)
        if not math.isclose(float(data.time), (step + 1) * dt, rel_tol=1e-10, abs_tol=1e-9):
            raise ValueError("simulator time disagrees with step count")
        trace.append({"time_s": float(data.time), "angle_rad": float(data.qpos[0]),
                      "velocity_rad_s": float(data.qvel[0]), "requested_torque_Nm": requested,
                      "applied_torque_Nm": float(data.qfrc_actuator[0])})
    return {"trace": trace, "joint_inertia_kg_m2": inertia, "xml": xml}


def difference(first, second, key):
    if len(first) != len(second) or not first:
        raise ValueError("compare equal, nonempty trajectories")
    return max(abs(a[key] - b[key]) for a, b in zip(first, second))


def run():
    import mujoco
    plant, bob, target = Pendulum(), SphericalBob(), math.pi / 2
    probes = []
    model, data, _ = make_engine(plant)
    for angle in (-math.pi / 2, -.4, 0., .4, math.pi / 2):
        reset(model, data, (angle, .7))
        data.ctrl[0] = .3
        mujoco.mj_forward(model, data)
        check_state(data)
        expected_acceleration = bob.acceleration(angle, .7, .3)
        observed = float(data.qacc[0])
        if not math.isclose(observed, expected_acceleration, rel_tol=1e-12, abs_tol=1e-12):
            raise ValueError("independent continuous acceleration check failed")
        probes.append({"angle_rad": angle, "velocity_rad_s": .7, "applied_torque_Nm": .3,
                       "native_inertia_kg_m2": joint_inertia(model, data),
                       "native_gravity_bias_Nm": float(data.qfrc_bias[0]),
                       "native_damping_torque_Nm": float(data.qfrc_passive[0]),
                       "native_acceleration_rad_s2": observed,
                       "finite_bob_expected_acceleration_rad_s2": expected_acceleration})
    cases = []
    for name, compensate, limit in (("PD_only", False, 12.), ("gravity_compensated", True, 12.),
                                     ("insufficient_torque", True, 8.)):
        args = {"compensate": compensate, "torque_limit": limit}
        point_trace = pd_rollout(plant, target, **args)
        native = native_rollout(plant, target, **args)
        trace = native["trace"]
        # Match BOTH physical inertia and explicit damping before demanding parity.
        finite_reference = pd_rollout(bob, target, **args)
        explicit = native_rollout(plant, target, implicit_damping=False, **args)
        parity = {key: difference(finite_reference, explicit["trace"], key)
                  for key in ("angle_rad", "velocity_rad_s")}
        if max(parity.values()) > 1e-10:
            raise ValueError("finite-bob explicit-damping reference mismatch")
        cases.append({"name": name, "compensate": compensate, "torque_limit_Nm": limit,
                      "native_final_error_rad": target - trace[-1]["angle_rad"],
                      "native_final_velocity_rad_s": trace[-1]["velocity_rad_s"],
                      "point_mass_final_error_rad": target - point_trace[-1]["angle_rad"],
                      "max_point_mass_vs_native_angle_difference_rad": difference(point_trace, trace, "angle_rad"),
                      "max_point_mass_vs_native_velocity_difference_rad_s": difference(point_trace, trace, "velocity_rad_s"),
                      "max_native_applied_torque_Nm": max(abs(row["applied_torque_Nm"]) for row in trace),
                      "native_clipped_steps": sum(abs(row["requested_torque_Nm"]) > limit for row in trace),
                      "finite_bob_explicit_damping_max_difference": parity,
                      "native_samples_each_0_5_s": trace[99::100],
                      "implicit_damping_model_xml": native["xml"],
                      "explicit_damping_model_xml": explicit["xml"]})
    directory = Path(__file__).resolve().parent
    paths = [directory / "dynamics.py", Path(__file__).resolve(),
             directory / "optional/test_mujoco_pendulum.py"]
    return {"evidence": "actual MuJoCo CPU dynamics; one hinge and a finite sphere; no contacts or hardware",
            "mujoco": mujoco.__version__, "python": platform.python_version(), "platform": platform.platform(),
            "source_sha256": {str(path.relative_to(directory)): hashlib.sha256(path.read_bytes()).hexdigest()
                              for path in paths},
            "point_mass_model": asdict(plant), "sphere_radius_m": bob.radius,
            "point_mass_inertia_kg_m2": plant.inertia, "finite_bob_inertia_kg_m2": bob.inertia,
            "dt_s": .005, "steps_per_rollout": 1200, "target_rad": target,
            "initial_state": [0., 0.], "kp_Nm_per_rad": 8., "kd_Nm_s_per_rad": 3.8,
            "dynamics_probes": probes, "cases": cases}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = run()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    for case in report["cases"]:
        print(case["name"], "native final error:", case["native_final_error_rad"], "rad",
              "max point-model angle difference:", case["max_point_mass_vs_native_angle_difference_rad"], "rad")
    print("Saved", args.output)


if __name__ == "__main__":
    main()
