#!/usr/bin/env python3
"""Actual headless MuJoCo execution of the saved scalar policy; no physical robot."""
import argparse
from dataclasses import asdict, dataclass
import hashlib
import json
import math
from pathlib import Path
import platform
import random
import statistics
import time

import policy_lab


@dataclass(frozen=True)
class Dynamics:
    name: str = "matched_100ms"
    mass_kg: float = 1.0
    assumed_mass_kg: float = 1.0
    damping_n_s_m: float = 0.0
    force_limit_n: float = 3.0
    physics_dt_s: float = 0.1
    policy_period_s: float = 0.1
    duration_s: float = 20.0

    def __post_init__(self):
        for name in ("mass_kg", "assumed_mass_kg", "force_limit_n", "physics_dt_s",
                     "policy_period_s", "duration_s"):
            value = getattr(self, name)
            if type(value) not in (int, float) or not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
        if (type(self.damping_n_s_m) not in (int, float)
                or not math.isfinite(self.damping_n_s_m) or self.damping_n_s_m < 0):
            raise ValueError("damping must be finite and nonnegative")
        for value in (self.policy_period_s / self.physics_dt_s,
                      self.duration_s / self.policy_period_s):
            if (not math.isfinite(value) or value < 1
                    or not math.isclose(value, round(value), rel_tol=0, abs_tol=1e-9)):
                raise ValueError("periods must contain a positive integer number of steps")
        if self.duration_s / self.physics_dt_s > 100_000:
            raise ValueError("teaching run exceeds 100000 physics steps per episode")


def load_policy(path):
    report = json.loads(Path(path).read_text())
    if not isinstance(report, dict):
        raise ValueError("saved policy must be a JSON object")
    source_hash = hashlib.sha256(Path(policy_lab.__file__).read_bytes()).hexdigest()
    if report.get("script_sha256") != source_hash:
        raise ValueError("policy source identity changed; regenerate with policy_lab.py")
    values = []
    for key, size in (("weights_normalized_inputs", 3), ("scaler_training_mean", 2),
                      ("scaler_training_std", 2)):
        value = report.get(key)
        if (not isinstance(value, list) or len(value) != size
                or any(type(v) not in (int, float) or not math.isfinite(v) for v in value)):
            raise ValueError(f"invalid {key}")
        values.append(value)
    if any(v <= 0 for v in values[2]):
        raise ValueError("scaler must be positive")
    weights, means, scales = values
    return lambda error, velocity: policy_lab.predict(weights, error, velocity, (means, scales))


def model_xml(config):
    """The physical model identity; policy adapter and episode length are separate."""
    # No contact surfaces or free rotations: one translational degree of freedom.
    return f'''<mujoco model="course_cart">
      <option timestep="{config.physics_dt_s:.17g}" gravity="0 0 0" integrator="Euler"/>
      <worldbody><body name="cart">
        <joint name="slide" type="slide" axis="1 0 0" damping="{config.damping_n_s_m:.17g}"/>
        <geom type="box" size="0.1 0.1 0.1" mass="{config.mass_kg:.17g}" contype="0" conaffinity="0"/>
      </body></worldbody>
      <actuator><motor name="force" joint="slide" gear="1" ctrllimited="true"
        ctrlrange="{-config.force_limit_n:.17g} {config.force_limit_n:.17g}" forcelimited="true"
        forcerange="{-config.force_limit_n:.17g} {config.force_limit_n:.17g}"/></actuator>
    </mujoco>'''


def make_model(config):
    import mujoco
    xml = model_xml(config)
    model = mujoco.MjModel.from_xml_string(xml)
    if (model.nq, model.nv, model.nu) != (1, 1, 1):
        raise ValueError("unexpected cart dimensions")
    return model, mujoco.MjData(model), xml


def reject_warnings(data):
    """Auto-reset can turn an unstable state into a finite, apparently successful one."""
    import mujoco
    warnings = [mujoco.mjtWarning(i).name for i, warning in enumerate(data.warning)
                if warning.number]
    if warnings:
        raise ValueError("MuJoCo warning invalidates episode: " + ", ".join(warnings))


def reset(model, data, initial):
    import mujoco
    if len(initial) != 2 or any(not math.isfinite(v) for v in initial):
        raise ValueError("initial position and velocity must be finite")
    mujoco.mj_resetData(model, data)
    data.qpos[0], data.qvel[0] = initial
    mujoco.mj_forward(model, data)
    reject_warnings(data)


def control_force(raw_acceleration, config):
    if not math.isfinite(raw_acceleration):
        raise ValueError("nonfinite policy action")
    acceleration = max(-1.0, min(1.0, raw_acceleration))
    requested = config.assumed_mass_kg * acceleration
    applied = max(-config.force_limit_n, min(config.force_limit_n, requested))
    return applied, acceleration != raw_acceleration, applied != requested


def simulate(policy, config, initial, *, engine=None):
    """Run one episode; an optional engine must come unmodified from make_model."""
    import mujoco
    model, data, xml = make_model(config) if engine is None else engine
    if xml != model_xml(config):
        raise ValueError("reused engine does not match the requested physical model")
    reset(model, data, initial)
    substeps = round(config.policy_period_s / config.physics_dt_s)
    decisions = round(config.duration_s / config.policy_period_s)
    trace = [{"time_s": 0.0, "position_m": float(data.qpos[0]),
              "velocity_m_s": float(data.qvel[0]), "force_n": 0.0}]
    action_clips = force_clips = 0
    for decision in range(decisions):
        raw = policy(-float(data.qpos[0]), float(data.qvel[0]))
        force, action_clip, force_clip = control_force(raw, config)
        action_clips += action_clip
        force_clips += force_clip
        data.ctrl[0] = force
        for _ in range(substeps):
            mujoco.mj_step(model, data)
            reject_warnings(data)
            if not all(math.isfinite(float(v)) for v in (data.qpos[0], data.qvel[0])):
                raise ValueError("nonfinite simulator state")
        expected_time = (decision + 1) * config.policy_period_s
        if not math.isclose(data.time, expected_time, rel_tol=1e-10, abs_tol=1e-8):
            raise ValueError("simulated clock disagrees with physics steps")
        trace.append({"time_s": float(data.time), "position_m": float(data.qpos[0]),
                      "velocity_m_s": float(data.qvel[0]), "force_n": force})
    outside = [i for i, row in enumerate(trace)
               if abs(row["position_m"]) >= 0.1 or abs(row["velocity_m_s"]) >= 0.1]
    settling_index = (max(outside) + 1) if outside else 0
    return {"initial": list(initial), "final_position_m": float(data.qpos[0]),
            "final_velocity_m_s": float(data.qvel[0]),
            "success": bool(abs(data.qpos[0]) < 0.1 and abs(data.qvel[0]) < 0.1),
            "sampled_settling_time_s": trace[settling_index]["time_s"] if settling_index < len(trace) else None,
            "mean_abs_position_m": statistics.fmean(abs(r["position_m"]) for r in trace[1:]),
            "policy_calls": decisions, "physics_steps": decisions * substeps,
            "acceleration_clips": action_clips, "force_clips": force_clips,
            "trace": trace, "xml_sha256": hashlib.sha256(xml.encode()).hexdigest()}


def analytical_trace(policy, initial, dt=0.1, steps=200):
    position, velocity = initial
    out = [(position, velocity)]
    for _ in range(steps):
        acceleration = max(-1.0, min(1.0, policy(-position, velocity)))
        velocity += acceleration * dt
        position += velocity * dt
        out.append((position, velocity))
    return out


def run(policy_path):
    import mujoco
    policy = load_policy(policy_path)
    rng = random.Random(101)
    initials = [(rng.uniform(-1.4, 1.4), rng.uniform(-0.3, 0.3)) for _ in range(12)]
    configs = [Dynamics(), Dynamics("substeps_10ms", physics_dt_s=0.01),
               Dynamics("mass3_wrong_adapter", mass_kg=3, physics_dt_s=0.01),
               Dynamics("mass3_correct_adapter", mass_kg=3, assumed_mass_kg=3, physics_dt_s=0.01),
               Dynamics("mass3_force_limited", mass_kg=3, assumed_mass_kg=3, force_limit_n=0.5, physics_dt_s=0.01),
               Dynamics("damping_without_compensation", damping_n_s_m=1, physics_dt_s=0.01)]
    cases=[]
    started=time.perf_counter()
    for config in configs:
        engine=make_model(config)
        episodes=[simulate(policy, config, initial, engine=engine) for initial in initials]
        parity=None
        if config.name == "matched_100ms":
            parity=max(abs(row[key]-expected[j]) for ep, initial in zip(episodes,initials)
                       for row,expected in zip(ep["trace"],analytical_trace(policy,initial))
                       for j,key in enumerate(("position_m","velocity_m_s")))
            if parity > 1e-10:raise ValueError("matched analytical/simulator parity failed")
        first_trace=episodes[0]["trace"]
        for episode in episodes:del episode["trace"]
        cases.append({"config":asdict(config),"successes":sum(bool(e["success"]) for e in episodes),
                      "mean_abs_position_m":statistics.fmean(e["mean_abs_position_m"] for e in episodes),
                      "force_clips":sum(e["force_clips"] for e in episodes),
                      "analytical_max_absolute_error":parity,"first_episode_trace":first_trace,"episodes":episodes})
    return {"evidence":"actual MuJoCo CPU dynamics; synthetic 1-DOF model and learned scalar policy; no rendering or hardware",
            "mujoco":mujoco.__version__,"python":platform.python_version(),"platform":platform.platform(),
            "policy_sha256":hashlib.sha256(Path(policy_path).read_bytes()).hexdigest(),
            "source_sha256":hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "rollout_seed":101,"episodes_per_case":12,"cases":cases,
            "simulation_wall_seconds":time.perf_counter()-started}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy",type=Path,required=True,help="JSON produced by policy_lab.py")
    parser.add_argument("--output",type=Path,required=True)
    args=parser.parse_args()
    report=run(args.policy)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(report,indent=2,allow_nan=False)+"\n")
    for case in report["cases"]:
        print(case["config"]["name"],f'{case["successes"]}/12',f'mean_abs_position={case["mean_abs_position_m"]:.6f} m',f'force_clips={case["force_clips"]}')
    print("Saved",args.output)

if __name__ == "__main__":main()
