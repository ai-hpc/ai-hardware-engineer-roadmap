"""Run bounded CPU fixtures for dynamics, optimal control, motion, and grasping."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import platform

from dynamics import Pendulum, pd_rollout, required_motor_current
from optimal_control import finite_action_mpc, finite_lqr, lqr_rollout, step
from task_execution import (compose_pose, cubic_duration, evaluate_lift,
                            required_finger_force, segment_clearance)


def experiments():
    model = Pendulum()
    target = math.pi / 2
    pendulum_runs = {}
    for name, compensate, limit in (("PD_only", False, 12.0),
                                     ("gravity_compensated", True, 12.0),
                                     ("insufficient_torque", True, 8.0)):
        trace = pd_rollout(model, target, compensate=compensate, torque_limit=limit)
        pendulum_runs[name] = {
            "final_error_rad": target - trace[-1]["angle_rad"],
            "final_velocity_rad_s": trace[-1]["velocity_rad_s"],
            "max_applied_torque_Nm": max(abs(row["applied_torque_Nm"]) for row in trace),
            "clipped_steps": sum(abs(row["requested_torque_Nm"]) > limit for row in trace),
            "samples_each_0_5_s": trace[99::100],
        }
    gains, value = finite_lqr(0.1, 40)
    lqr = lqr_rollout((1.0, 0.0), gains, 0.1)
    clipped = lqr_rollout((1.0, 0.0), gains, 0.1, acceleration_limit=1.0)
    plan = finite_action_mpc((0.35, 0.5))
    state, mpc_trace = (0.35, 0.5), []
    for tick in range(10):
        current = finite_action_mpc(state)
        if current is None:
            mpc_trace.append({"tick": tick, "state": state, "outcome": "no_feasible_plan"})
            break
        action = current["actions"][0]
        state = step(state, action, 0.2)
        mpc_trace.append({"tick": tick, "action": action, "state": state})
    obstacle = (0.0, 0.0)
    path = [(-1.0, 0.0), (-1.0, 0.6), (1.0, 0.6), (1.0, 0.0)]
    object_before = (0.6, 0.2, 0.0, math.pi / 2)
    gripper_before = compose_pose(object_before, (0.1, 0.0, 0.05, 0.0))
    gripper_after = (*gripper_before[:2], gripper_before[2] + 0.2, gripper_before[3])
    object_after = (*object_before[:2], 0.2, object_before[3])
    return {
        "dynamics": {"model": vars(model), "inertia_kg_m2": model.inertia,
                     "required_horizontal_torque_Nm": model.gravity_torque(target),
                     "ideal_motor_current_A": required_motor_current(model.gravity_torque(target), 50, .8, .1),
                     "dt_s": .005, "steps": 1200, "runs": pendulum_runs},
        "optimal_control": {"dt_s": .1, "horizon": 40, "first_gain": gains[0],
                            "initial_value_matrix": value,
                            "unconstrained_cost": lqr["cost"],
                            "unconstrained_final_state": lqr["final_state"],
                            "largest_requested_acceleration": max(abs(row["requested_acceleration"]) for row in lqr["trace"]),
                            "clipped_cost": clipped["cost"], "clipped_final_state": clipped["final_state"],
                            "mpc_initial_plan": plan, "mpc_receding_trace": mpc_trace,
                            "mpc_final_position_error_m": state[0] - .45,
                            "mpc_final_velocity_m_s": state[1],
                            "mpc_settled": abs(state[0] - .45) < .02 and abs(state[1]) < .05,
                            "unavoidable_boundary_violation_has_plan": finite_action_mpc((.45, .5)) is not None},
        "motion": {"straight_segment_clearance_m": segment_clearance((-1., 0.), (1., 0.), obstacle, .2, .1),
                   "detour_segment_clearances_m": [segment_clearance(a, b, obstacle, .2, .1)
                                                    for a, b in zip(path, path[1:])],
                   "one_metre_cubic_duration_s": cubic_duration(1., .5, 1.),
                   "drifted_segment_clearance_m": segment_clearance((-1., .25), (1., .25), obstacle, .2, .1)},
        "grasp": {"desired_world_gripper_pose": gripper_before,
                  "required_normal_force_each_finger_N": required_finger_force(.5, .4, 1.5),
                  "object_moves_with_hand": evaluate_lift(gripper_before, object_before, gripper_after, object_after),
                  "hand_moves_object_stays": evaluate_lift(gripper_before, object_before, gripper_after, object_before)},
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("experiment", choices=("all", "dynamics", "optimal_control", "motion", "grasp"))
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    results = experiments()
    if args.experiment != "all":
        results = {args.experiment: results[args.experiment]}
    directory = Path(__file__).parent
    report = {"execution_status": "synthetic CPU models; no robot, GPU, or contact simulator",
              "python": platform.python_version(),
              "source_sha256": {path.name: hashlib.sha256(path.read_bytes()).hexdigest()
                                for path in sorted(directory.glob("*.py")) if not path.name.startswith("test_")},
              "results": results}
    encoded = json.dumps(report, indent=2, allow_nan=False)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded + "\n", encoding="utf-8")
    print(encoded)


if __name__ == "__main__":
    main()
