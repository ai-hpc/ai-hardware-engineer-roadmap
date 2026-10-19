#!/usr/bin/env python3
"""Optional Isaac Sim 6.1 atomic push bridge; source-reviewed, not simulator-executed.

This is a simulator experiment. Asset/drive/material parity with training must be
measured before comparing learned-policy outcomes. It has no physical robot backend.
"""
import argparse
import json
import math
from pathlib import Path
import time
from capstone_protocol import CommandGate, JOINT_NAMES, Rejected, canonical, strict_loads
from ros2_capstone_replay import isolate_ros


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", required=True, type=Path)
    parser.add_argument("--domain-id", type=int, required=True)
    parser.add_argument("--namespace", required=True)
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--lockstep", action="store_true")
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--output", type=Path, default=Path("push-bridge-result.json"))
    args = parser.parse_args()
    if args.episodes <= 0:
        parser.error("episodes must be positive")
    isolate_ros(args.domain_id)
    contract = json.loads(args.contract.read_text())
    if tuple(contract["joint_names"]) != JOINT_NAMES:
        raise ValueError("runtime joint contract mismatch")
    dt, decimation = contract["physics_dt"], contract["decimation"]
    if (type(dt) not in (int, float) or not math.isfinite(dt) or dt <= 0
            or type(decimation) is not int or decimation <= 0):
        raise ValueError("invalid timestep or decimation")
    period_ns = round(dt * decimation * 1e9)
    ttl_ns = 500_000_000 if args.lockstep else period_ns
    limits = contract["limits"]
    gate = CommandGate([row[0] for row in limits], [row[1] for row in limits], contract["action_scale"])

    from isaacsim import SimulationApp
    app = SimulationApp({"headless": args.headless})
    import numpy as np
    import isaacsim.core.experimental.utils.app as app_utils
    import isaacsim.core.experimental.utils.stage as stage_utils
    from isaacsim.core.experimental.objects import Cube, GroundPlane
    from isaacsim.core.experimental.prims import Articulation, GeomPrim, RigidPrim
    from isaacsim.core.simulation_manager import SimulationManager
    from isaacsim.storage.native import get_assets_root_path
    app_utils.enable_extension("isaacsim.ros2.bridge")
    app.update()
    import rclpy
    from rclpy.node import Node
    from std_msgs.msg import String

    stage_utils.create_new_stage()
    SimulationManager.setup_simulation(dt=dt, device="cuda:0")
    GroundPlane("/World/GroundPlane")
    stage_utils.add_reference_to_stage(
        get_assets_root_path() + "/Isaac/IsaacLab/Robots/FrankaEmika/franka_panda.usda",
        "/World/Robot", variants=[("Physics", "physx"), ("Colliders", "primitives")])
    robot = Articulation("/World/Robot")
    robot.set_solver_iteration_counts(8, 0)
    robot.set_enabled_self_collisions(False)
    Cube("/World/Cube", sizes=.05, positions=[[.5, 0, .025]])
    GeomPrim("/World/Cube", apply_collision_apis=True)
    cube = RigidPrim("/World/Cube", masses=[.2])
    rclpy.init(args=[])
    node = Node("push_simulator", namespace=args.namespace, enable_rosout=False)
    publisher = node.create_publisher(String, "observation", 1)
    log, accepted = [], []

    def receive(message):
        try:
            result = gate.accept(strict_loads(message.data), time.monotonic_ns())
            accepted.append(result.age_ns)
            log.append({"episode": gate.episode, "sequence": gate.sequence,
                        "event": "accepted", "age_ns": result.age_ns})
        except Rejected as error:
            log.append({"episode": gate.episode, "event": "rejected", "reason": str(error)})

    subscription = node.create_subscription(String, "normalized_action", receive, 1)
    try:
        deadline = time.monotonic()+10
        while not (publisher.get_subscription_count() and subscription.get_publisher_count()):
            if time.monotonic() > deadline:
                raise TimeoutError("start the policy node with the same domain and namespace")
            rclpy.spin_once(node, timeout_sec=.02)
        app_utils.play(); app.update()
        names = list(robot.dof_names)
        if len(names) != 9 or set(names) != set(JOINT_NAMES):
            raise ValueError("asset joints differ from checkpoint")
        read_order = [names.index(name) for name in JOINT_NAMES]
        arm_ids = read_order[:7]
        finger_id = [names.index("panda_finger_joint1")]
        rng = np.random.default_rng(7)
        outcomes, physics_steps, missed, fallback = [], 0, 0, 0
        start = time.monotonic_ns()
        for episode in range(args.episodes):
            canonical_q = np.asarray(contract["q0"], dtype=float).copy()
            canonical_q[:7] += rng.uniform(-.05, .05, 7)
            canonical_q[7:] = 0
            canonical_q[:7] = np.clip(canonical_q[:7], gate.lower, gate.upper)
            q_init = np.array([canonical_q[JOINT_NAMES.index(name)] for name in names])
            robot.set_dof_positions(q_init[None])
            robot.set_dof_velocities(np.zeros((1, 9)))
            robot.set_dof_position_targets(q_init[None])
            gate.reset(episode, canonical_q[:7].tolist())
            xy = rng.uniform([.4, -.15], [.6, .15])
            angle = rng.uniform(-np.pi, np.pi)
            goal = np.r_[xy + rng.uniform(.1, .2)*np.array([np.cos(angle), np.sin(angle)]), .025]
            goal[0] = np.clip(goal[0], .3, .7)
            cube.set_world_poses(positions=[[*xy, .025]], orientations=[[1, 0, 0, 0]])
            cube.set_velocities(np.zeros((1, 3)), np.zeros((1, 3)))
            pacing_origin = time.monotonic_ns()  # reset/setup time is reported separately from step deadlines.
            for step in range(round(6/dt)):
                if step % decimation == 0:
                    if gate.pending is not None:
                        fallback += 1; gate.expire()
                    q = robot.get_dof_positions().numpy()[0, read_order]
                    qd = robot.get_dof_velocities().numpy()[0, read_order]
                    position = cube.get_world_poses()[0].numpy()[0]
                    obs = gate.issue(q.tolist(), qd.tolist(), position.tolist(), goal.tolist(),
                                     time.monotonic_ns(), ttl_ns)
                    publisher.publish(String(data=canonical(obs)))
                    if args.lockstep:
                        while gate.pending is not None:
                            if time.monotonic_ns() > obs["expires_ns"]:
                                gate.expire()
                                raise TimeoutError("lockstep response missing; no old command was applied")
                            rclpy.spin_once(node, timeout_sec=.002)
                rclpy.spin_once(node, timeout_sec=0)
                robot.set_dof_position_targets([gate.target], dof_indices=arm_ids)
                robot.set_dof_position_targets([[0.0]], dof_indices=finger_id)
                SimulationManager.step(); physics_steps += 1
                if not args.lockstep:
                    slack = pacing_origin + round((step+1)*dt*1e9) - time.monotonic_ns()
                    if slack > 0: time.sleep(slack/1e9)
                    else: missed += 1
            if gate.pending is not None: fallback += 1
            gate.expire()
            final = cube.get_world_poses()[0].numpy()[0]
            outcomes.append(bool(np.linalg.norm(final[:2]-goal[:2]) < .03))
        elapsed = (time.monotonic_ns()-start)/1e9
        samples = sorted(accepted)
        percentile = lambda p: samples[max(0, math.ceil(p*len(samples))-1)] if samples else None
        report = {"scope": "Isaac simulator experiment; asset parity still requires separate validation",
                  "lockstep": args.lockstep, "episodes": len(outcomes), "successes": sum(outcomes),
                  "physics_steps": physics_steps, "wall_s": elapsed, "rtf": physics_steps*dt/elapsed,
                  "step_deadline_misses": missed, "expired_requests_held": fallback,
                  "accepted_actions": len(accepted), "age_p50_ns": percentile(.5), "age_p99_ns": percentile(.99),
                  "events": log}
        args.output.write_text(json.dumps(report, indent=2, allow_nan=False)+"\n")
        print(json.dumps({key:value for key,value in report.items() if key != "events"}, indent=2))
    finally:
        node.destroy_node(); rclpy.try_shutdown(); app.close()


if __name__ == "__main__": main()
