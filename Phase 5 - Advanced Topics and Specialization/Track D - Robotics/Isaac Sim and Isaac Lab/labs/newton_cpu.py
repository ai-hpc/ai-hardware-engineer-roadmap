#!/usr/bin/env python3
"""Actual standalone Newton 1.5.2 / Warp 1.16 CPU checks; no Isaac or renderer."""
import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path


def experiment():
    import numpy as np
    import newton
    import warp as wp

    wp.init()
    # Explicit convention from this Newton release; no implicit target-layout migration.
    newton.use_coord_layout_targets = True
    with wp.ScopedDevice("cpu"):
        def build(floor=False, worlds=1):
            template = newton.ModelBuilder()
            if floor:
                template.add_ground_plane()
            body = template.add_body(xform=wp.transform(
                p=wp.vec3(0, 0, 2), q=wp.quat_identity()))
            template.add_shape_box(body, hx=.05, hy=.05, hz=.05)
            builder = newton.ModelBuilder()
            builder.replicate(template, world_count=worlds)
            model = builder.finalize()
            state, other = model.state(), model.state()
            newton.eval_fk(model, model.joint_q, model.joint_qd, state)
            solver = newton.solvers.SolverXPBD(model, iterations=8)
            pipeline = newton.CollisionPipeline(model) if floor else None
            contacts = pipeline.contacts() if pipeline else None
            return model, state, other, solver, pipeline, contacts

        def run(floor, worlds, steps, dt):
            model, state, other, solver, pipeline, contacts = build(floor, worlds)
            control = model.control()
            initial = state.body_q.numpy().copy()
            for _ in range(steps):
                state.clear_forces()
                if pipeline:
                    pipeline.collide(state, contacts)
                solver.step(state, other, control, contacts, dt)
                state, other = other, state
            wp.synchronize()
            positions = state.body_q.numpy().copy()
            velocities = state.body_qd.numpy().copy()
            assert np.isfinite(positions).all() and np.isfinite(velocities).all()
            return initial, positions, velocities

        free = []
        for steps, dt in [(10, .01), (20, .005)]:
            _, pose, velocity = run(False, 1, steps, dt)
            # XPBD's ballistic update is semi-implicit Euler: v changes before x.
            discrete_z = 2 - 9.81 * dt * dt * steps * (steps + 1) / 2
            analytic_z = 2 - .5 * 9.81 * (steps * dt) ** 2
            assert abs(float(pose[0, 2]) - discrete_z) < 3e-6
            assert abs(float(velocity[0, 2]) + 9.81 * steps * dt) < 3e-6
            free.append({"steps": steps, "dt_s": dt, "z_m": float(pose[0, 2]),
                         "semi_implicit_z_m": discrete_z, "continuous_z_m": analytic_z,
                         "continuous_error_m": abs(float(pose[0, 2]) - analytic_z)})
        assert free[1]["continuous_error_m"] < .51 * free[0]["continuous_error_m"]

        _, single, single_v = run(True, 1, 480, 1 / 240)
        _, paired, paired_v = run(True, 2, 480, 1 / 240)
        # World replicas occupy identical coordinates. Cross-world contacts would corrupt this.
        assert paired.shape == (2, 7)
        assert np.max(np.abs(paired - single)) < 2e-5
        assert np.max(np.abs(paired_v - single_v)) < 2e-5
        assert np.max(np.abs(paired[:, 2] - .05)) < .003
        assert np.max(np.abs(paired_v)) < .02
        return {"scope": "actual CPU XPBD physics; no Kit, Isaac Lab, MJWarp, CUDA or rendering",
                "newton": importlib.metadata.version("newton"), "warp": wp.__version__,
                "numpy": np.__version__, "device": str(wp.get_device()),
                "freefall": free, "resting_worlds": 2,
                "resting_z_m": paired[:, 2].tolist(),
                "world_replication_max_pose_difference": float(np.max(np.abs(paired-single))),
                "maximum_resting_velocity": float(np.max(np.abs(paired_v))),
                "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = json.dumps(experiment(), indent=2, allow_nan=False) + "\n"
    if args.output:
        args.output.write_text(report)
    print(report, end="")
