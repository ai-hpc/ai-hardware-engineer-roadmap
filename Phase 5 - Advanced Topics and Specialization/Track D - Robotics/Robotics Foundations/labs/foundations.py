"""Small, inspectable robotics models. Standard library only; no physical robot I/O.

Run: python3 foundations.py all --output results.json
All reported experiments are synthetic numerical fixtures, not robot benchmarks.
"""

from __future__ import annotations

import argparse
import heapq
import json
import math
from pathlib import Path
from typing import Iterable


def finite(*values: float) -> None:
    if not all(math.isfinite(value) for value in values):
        raise ValueError("values must be finite")


def transform_point(pose: tuple[float, float, float],
                    point: tuple[float, float]) -> tuple[float, float]:
    """T_AB: pose of frame B in A; point in B -> point in A. Metres/radians."""
    tx, ty, yaw = pose
    x, y = point
    finite(tx, ty, yaw, x, y)
    c, s = math.cos(yaw), math.sin(yaw)
    return c * x - s * y + tx, s * x + c * y + ty


def inverse_pose(pose: tuple[float, float, float]) -> tuple[float, float, float]:
    """Return T_BA from T_AB; translation must also be rotated."""
    x, y, yaw = pose
    finite(x, y, yaw)
    c, s = math.cos(yaw), math.sin(yaw)
    return -c * x - s * y, s * x - c * y, -yaw


def compose_pose(ab: tuple[float, float, float],
                 bc: tuple[float, float, float]) -> tuple[float, float, float]:
    """T_AC = T_AB T_BC, with the rightmost transform applied first."""
    finite(*ab, *bc)
    x, y = transform_point(ab, bc[:2])
    return x, y, ab[2] + bc[2]


def validate_age(stamp: float, now: float, max_age: float) -> float:
    """Require timestamps from one clock; reject future and stale observations."""
    finite(stamp, now, max_age)
    if max_age < 0:
        raise ValueError("max_age must be nonnegative")
    age = now - stamp
    if age < 0 or age > max_age:
        raise ValueError("observation is future-dated or stale")
    return age


def link_lengths(l1: float, l2: float) -> None:
    finite(l1, l2)
    if l1 <= 0 or l2 <= 0:
        raise ValueError("link lengths must be positive")


def fk_2r(q1: float, q2: float, l1: float = 1., l2: float = 1.) -> tuple[float, float]:
    """Planar two-revolute-joint arm; q2 is relative to link 1."""
    link_lengths(l1, l2)
    finite(q1, q2)
    return (l1 * math.cos(q1) + l2 * math.cos(q1 + q2),
            l1 * math.sin(q1) + l2 * math.sin(q1 + q2))


def ik_2r(x: float, y: float, l1: float = 1., l2: float = 1.,
          elbow: int = 1) -> tuple[float, float]:
    """Analytical position IK. Does not check joint limits or collisions.

    The equal-link folded-origin case has infinitely many q1 solutions; return 0.
    """
    link_lengths(l1, l2)
    finite(x, y)
    if elbow not in (-1, 1):
        raise ValueError("elbow must be -1 or 1")
    radius = math.hypot(x, y)
    if radius > l1 + l2 + 1e-12 or radius < abs(l1 - l2) - 1e-12:
        raise ValueError("target is outside the reachable annulus")
    cosine = (x * x + y * y - l1 * l1 - l2 * l2) / (2 * l1 * l2)
    q2 = elbow * math.acos(max(-1., min(1., cosine)))
    if x == 0. and y == 0. and l1 == l2:
        return 0., q2
    q1 = math.atan2(y, x) - math.atan2(l2 * math.sin(q2), l1 + l2 * math.cos(q2))
    return q1, q2


def jacobian_2r(q1: float, q2: float, l1: float = 1., l2: float = 1.) -> tuple:
    """Rows are x/y, columns are joint 1/2; xdot = J qdot."""
    link_lengths(l1, l2)
    finite(q1, q2)
    return ((-l1 * math.sin(q1) - l2 * math.sin(q1 + q2), -l2 * math.sin(q1 + q2)),
            (l1 * math.cos(q1) + l2 * math.cos(q1 + q2), l2 * math.cos(q1 + q2)))


def drive_step(pose: tuple[float, float, float], left: float, right: float,
               track: float, dt: float) -> tuple[float, float, float]:
    """Exact ideal differential-drive arc for constant wheel LINEAR speeds.

    left/right in m/s, track is wheel separation in m, dt in s; no wheel slip.
    """
    finite(*pose, left, right, track, dt)
    if track <= 0 or dt <= 0:
        raise ValueError("track and dt must be positive")
    v, omega = (left + right) / 2, (right - left) / track
    half_turn = omega * dt / 2
    sinc = math.sin(half_turn) / half_turn if half_turn != 0 else 1.
    distance = v * dt * sinc
    heading = pose[2] + half_turn
    return (pose[0] + distance * math.cos(heading),
            pose[1] + distance * math.sin(heading), pose[2] + omega * dt)


def control_rollout(kp: float, dt: float, steps: int = 80, delay: int = 0,
                    velocity_limit: float | None = None) -> list[float]:
    """Toy position integrator, target 1 m, initial position 0 m.

    x[k+1] = x[k] + dt * kp * (1 - x[k-delay]). Before k=0, x=0.
    A real robot's dynamics, contact, motor limits and estimator are absent.
    """
    finite(kp, dt)
    if kp <= 0 or dt <= 0 or steps < 1 or delay < 0:
        raise ValueError("invalid control configuration")
    if velocity_limit is not None:
        finite(velocity_limit)
        if velocity_limit <= 0:
            raise ValueError("velocity_limit must be positive")
    positions = [0.]
    for k in range(steps):
        observed = positions[max(0, k - delay)]
        velocity = kp * (1. - observed)
        if velocity_limit is not None:
            velocity = max(-velocity_limit, min(velocity_limit, velocity))
        positions.append(positions[-1] + dt * velocity)
    return positions


def project(point: tuple[float, float, float], fx: float, fy: float,
            cx: float, cy: float) -> tuple[float, float]:
    """Ideal rectified pinhole camera. Optical x right, y down, z forward."""
    x, y, z = point
    finite(x, y, z, fx, fy, cx, cy)
    if z <= 0 or fx <= 0 or fy <= 0:
        raise ValueError("depth and focal lengths must be positive")
    return fx * x / z + cx, fy * y / z + cy


def unproject(pixel: tuple[float, float], depth: float, fx: float, fy: float,
              cx: float, cy: float) -> tuple[float, float, float]:
    """Depth is optical-axis z in metres, not Euclidean range."""
    u, v = pixel
    finite(u, v, depth, fx, fy, cx, cy)
    if depth <= 0 or fx <= 0 or fy <= 0:
        raise ValueError("depth and focal lengths must be positive")
    return (u - cx) * depth / fx, (v - cy) * depth / fy, depth


def scalar_update(mean: float, variance: float, measurement: float,
                  measurement_variance: float) -> tuple[float, float, float]:
    """Independent unbiased scalar Gaussian prior/measurement of one quantity."""
    finite(mean, variance, measurement, measurement_variance)
    if variance < 0 or measurement_variance <= 0:
        raise ValueError("prior variance must be >=0 and measurement variance >0")
    gain = variance / (variance + measurement_variance)
    posterior = mean + gain * (measurement - mean)
    # Joseph form retains the nonnegative terms explicitly.
    posterior_variance = (1 - gain) ** 2 * variance + gain ** 2 * measurement_variance
    return posterior, posterior_variance, gain


Cell = tuple[int, int]


def inflate_cells(obstacles: Iterable[Cell], width: int, height: int,
                  radius: int) -> set[Cell]:
    """Square footprint of (2*radius+1) cells; outside the map is occupied.

    This conservative grid exercise is not a continuous collision checker.
    """
    if width < 1 or height < 1 or radius < 0:
        raise ValueError("invalid map size or radius")
    obstacles = set(obstacles)
    return {(x, y) for x in range(width) for y in range(height)
            if any((x + dx, y + dy) in obstacles
                   or not (0 <= x + dx < width and 0 <= y + dy < height)
                   for dx in range(-radius, radius + 1)
                   for dy in range(-radius, radius + 1))}


def astar(width: int, height: int, obstacles: Iterable[Cell], start: Cell,
          goal: Cell) -> list[Cell] | None:
    """Four-neighbour unit-cost grid A* with Manhattan-distance heuristic."""
    if width < 1 or height < 1:
        raise ValueError("map dimensions must be positive")
    blocked = set(obstacles)

    def valid(cell: Cell) -> bool:
        return 0 <= cell[0] < width and 0 <= cell[1] < height and cell not in blocked

    if not valid(start) or not valid(goal):
        raise ValueError("start and goal must be free in-bounds cells")

    def heuristic(cell: Cell) -> int:
        return abs(goal[0] - cell[0]) + abs(goal[1] - cell[1])

    frontier = [(heuristic(start), 0, start)]
    cost = {start: 0}
    parent: dict[Cell, Cell] = {}
    while frontier:
        _, distance, current = heapq.heappop(frontier)
        if distance != cost[current]:
            continue  # An improved route superseded this queue entry.
        if current == goal:
            path = [current]
            while current != start:
                current = parent[current]
                path.append(current)
            return path[::-1]
        x, y = current
        for neighbour in ((x + 1, y), (x, y + 1), (x - 1, y), (x, y - 1)):
            candidate = distance + 1
            if valid(neighbour) and candidate < cost.get(neighbour, math.inf):
                cost[neighbour] = candidate
                parent[neighbour] = current
                heapq.heappush(frontier, (candidate + heuristic(neighbour), candidate, neighbour))
    return None


def experiments() -> dict:
    world_base = (1., 2., math.pi / 2)
    base_camera = (.2, 0., 0.)
    world_camera = compose_pose(world_base, base_camera)
    point_world = transform_point(world_camera, (1., 0.))
    kinematic_target = (1., 1.)
    branches = [ik_2r(*kinematic_target, elbow=e) for e in (1, -1)]

    def control_summary(delay: int, limit: float | None = None) -> dict:
        positions = control_rollout(12., .1, delay=delay, velocity_limit=limit)
        return {"delay_steps": delay, "velocity_limit_m_s": limit,
                "final_error_m": 1 - positions[-1],
                "peak_absolute_error_m": max(abs(1 - x) for x in positions),
                "last_ten_positions_m": positions[-10:]}

    obstacles = {(3, y) for y in range(7) if y != 3}
    point_path = astar(7, 7, obstacles, (1, 3), (5, 3))
    inflated = inflate_cells(obstacles, 7, 7, 1)
    body_path = astar(7, 7, inflated, (1, 3), (5, 3))
    return {
        "frames": {"point_world_m": point_world,
                   "point_camera_roundtrip_m": transform_point(inverse_pose(world_camera), point_world),
                   "observation_age_s": validate_age(9.96, 10., .05)},
        "kinematics": {"ik_branches_rad": branches,
                       "fk_roundtrips_m": [fk_2r(*q) for q in branches],
                       "straight_arm_jacobian_m_rad": jacobian_2r(0., 0.),
                       "drive_after_1s_m_rad": drive_step((0., 0., 0.), .1, .3, .4, 1.)},
        "control": {"kp_s_inverse": 12., "dt_s": .1, "steps": 80,
                    "instantaneous": control_summary(0),
                    "one_step_delay": control_summary(1),
                    "one_step_delay_with_limit": control_summary(1, 1.)},
        "perception": {"pixel_uv": project((.1, .05, 1.), 500., 500., 320., 240.),
                       "point_optical_m": unproject((370., 265.), 1., 500., 500., 320., 240.),
                       "posterior_mean_variance_gain": scalar_update(1., .04, 1.3, .01)},
        "planning": {"point_path": point_path,
                     "point_path_length_cells": len(point_path) - 1 if point_path else None,
                     "square_footprint_path": body_path,
                     "inflated_blocked_cells": sorted(inflated)},
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("experiment", choices=("all", "frames", "kinematics", "control", "perception", "planning"))
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = experiments()
    if args.experiment != "all":
        result = {args.experiment: result[args.experiment]}
    result = {"execution_status": "synthetic numerical fixture", "results": result}
    encoded = json.dumps(result, indent=2, allow_nan=False)
    if args.output:
        args.output.write_text(encoded + "\n", encoding="utf-8")
    print(encoded)


if __name__ == "__main__":
    main()
