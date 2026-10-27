"""Turn tracked human hands into robot-style actions (pure Python, CPU).

Input per video frame (all in the SLAM world frame, metres):
  T_world_wrist  : SE3 wrist pose (e.g. from MPS hand tracking chained through T_world_device)
  thumb_tip      : 3D point
  index_tip      : 3D point
Output at the control rate:
  relative end-effector actions (translation + rotation vector, expressed in the
  current end-effector frame) and a gripper opening in [0, 1].
Relative actions are what GR00T N1.7 and UMI-style policies share across embodiments:
they do not depend on where the world origin is, only on how the hand moved.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

from ego_geometry import SE3, Vec3, interpolate_pose, quat_to_axis_angle


@dataclass(frozen=True)
class HandFrame:
    t: float               # seconds, device clock
    T_world_wrist: SE3
    thumb_tip: Vec3
    index_tip: Vec3


@dataclass(frozen=True)
class GripperMap:
    """Thumb-index distance -> opening in [0, 1] (0 = closed). Calibrate per person."""
    closed_m: float = 0.015
    open_m: float = 0.09

    def opening(self, thumb: Vec3, index: Vec3) -> float:
        d = math.dist(thumb, index)
        x = (d - self.closed_m) / (self.open_m - self.closed_m)
        return min(1.0, max(0.0, x))


def resample(frames: list[HandFrame], rate_hz: float) -> list[HandFrame]:
    """Interpolate hand frames onto a uniform control-rate clock (slerp for rotation)."""
    if len(frames) < 2:
        return list(frames)
    out: list[HandFrame] = []
    t, dt, j = frames[0].t, 1.0 / rate_hz, 0
    while t <= frames[-1].t + 1e-12:
        while j < len(frames) - 2 and frames[j + 1].t < t:
            j += 1
        a, b = frames[j], frames[j + 1]
        u = 0.0 if b.t == a.t else min(1.0, max(0.0, (t - a.t) / (b.t - a.t)))
        lerp = lambda p, q: tuple(x + u * (y - x) for x, y in zip(p, q))  # noqa: E731
        out.append(HandFrame(t, interpolate_pose(a.T_world_wrist, b.T_world_wrist, u),
                             lerp(a.thumb_tip, b.thumb_tip), lerp(a.index_tip, b.index_tip)))
        t += dt
    return out


def eef_poses(frames: list[HandFrame], T_wrist_eef: SE3, T_base_world: SE3) -> list[SE3]:
    """Wrist pose -> virtual gripper pose, expressed in the robot (or canonical) base frame."""
    return [T_base_world @ f.T_world_wrist @ T_wrist_eef for f in frames]


def relative_actions(poses: list[SE3]) -> list[tuple[Vec3, Vec3]]:
    """Action k moves pose k to pose k+1, expressed in the frame of pose k:
    delta = T_k^-1 * T_{k+1}  ->  (translation, rotation vector)."""
    actions = []
    for a, b in zip(poses, poses[1:]):
        d = a.inverse() @ b
        actions.append((d.t, quat_to_axis_angle(d.q)))
    return actions


def moving_average(values: list[Vec3], window: int) -> list[Vec3]:
    if window <= 1:
        return list(values)
    half = window // 2
    out = []
    for i in range(len(values)):
        lo, hi = max(0, i - half), min(len(values), i + half + 1)
        n = hi - lo
        out.append(tuple(sum(v[k] for v in values[lo:hi]) / n for k in range(3)))  # type: ignore[arg-type]
    return out


@dataclass(frozen=True)
class Limits:
    workspace_min: Vec3 = (0.2, -0.4, 0.0)    # robot base frame, metres
    workspace_max: Vec3 = (0.8, 0.4, 0.6)
    max_speed_m_s: float = 0.5
    max_ang_speed_rad_s: float = 2.0


def feasibility_report(poses: list[SE3], actions: list[tuple[Vec3, Vec3]], rate_hz: float, limits: Limits) -> dict:
    """Count where the retargeted motion leaves what the robot can do. A clean human clip
    can still be infeasible for a given arm: report it instead of silently clipping."""
    outside = sum(
        1 for p in poses
        if any(not (lo <= x <= hi) for x, lo, hi in zip(p.t, limits.workspace_min, limits.workspace_max))
    )
    too_fast = sum(1 for tr, _ in actions if math.hypot(*tr) * rate_hz > limits.max_speed_m_s)
    spin = sum(1 for _, rv in actions if math.hypot(*rv) * rate_hz > limits.max_ang_speed_rad_s)
    n = max(1, len(poses))
    return {
        "poses": len(poses),
        "outside_workspace": outside,
        "too_fast_steps": too_fast,
        "too_fast_rotation_steps": spin,
        "feasible_fraction": 1.0 - outside / n,
        "ok": outside == 0 and too_fast == 0 and spin == 0,
    }


def retarget_episode(frames: list[HandFrame], rate_hz: float, T_wrist_eef: SE3, T_base_world: SE3,
                     gripper: GripperMap = GripperMap(), limits: Limits = Limits()) -> dict:
    """Full pipeline: resample -> EEF poses in base frame -> relative actions + gripper -> report."""
    rs = resample(frames, rate_hz)
    poses = eef_poses(rs, T_wrist_eef, T_base_world)
    actions = relative_actions(poses)
    grip = [gripper.opening(f.thumb_tip, f.index_tip) for f in rs]
    return {
        "times": [f.t for f in rs],
        "poses": poses,
        "actions": [(*tr, *rv, grip[i + 1]) for i, (tr, rv) in enumerate(actions)],  # 7-D: dxyz, rotvec, gripper
        "report": feasibility_report(poses, actions, rate_hz, limits),
    }
