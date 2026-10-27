"""Frames and camera models for egocentric data (pure Python, CPU).

Conventions used throughout the course labs:
  * Quaternions are (w, x, y, z), unit norm. Libraries differ (some use x, y, z, w), so
    check each library's convention and convert explicitly at the boundary.
  * A transform T_a_b maps points expressed in frame b into frame a:
        p_a = T_a_b * p_b
    so chains read right to left: T_world_camera = T_world_device * T_device_camera.
  * Camera frame: +z forward along the optical axis, +x right, +y down.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

Vec3 = tuple[float, float, float]
Quat = tuple[float, float, float, float]  # (w, x, y, z)


# --------------------------------------------------------------------------- quaternions
def quat_normalize(q: Quat) -> Quat:
    n = math.sqrt(sum(c * c for c in q))
    if n == 0.0:
        raise ValueError("zero quaternion")
    return tuple(c / n for c in q)  # type: ignore[return-value]


def quat_mul(a: Quat, b: Quat) -> Quat:
    aw, ax, ay, az = a
    bw, bx, by, bz = b
    return (aw * bw - ax * bx - ay * by - az * bz,
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw)


def quat_conj(q: Quat) -> Quat:
    w, x, y, z = q
    return (w, -x, -y, -z)


def quat_rotate(q: Quat, v: Vec3) -> Vec3:
    qv = (0.0, *v)
    w, x, y, z = quat_mul(quat_mul(q, qv), quat_conj(q))
    return (x, y, z)


def quat_from_axis_angle(axis: Vec3, angle: float) -> Quat:
    n = math.sqrt(sum(c * c for c in axis))
    if n == 0.0:
        return (1.0, 0.0, 0.0, 0.0)
    s = math.sin(angle / 2.0) / n
    return (math.cos(angle / 2.0), axis[0] * s, axis[1] * s, axis[2] * s)


def quat_to_axis_angle(q: Quat) -> Vec3:
    """Rotation vector (axis * angle), angle in [0, pi]."""
    w, x, y, z = quat_normalize(q)
    if w < 0.0:  # take the short way round
        w, x, y, z = -w, -x, -y, -z
    s = math.sqrt(max(0.0, 1.0 - w * w))
    angle = 2.0 * math.atan2(s, w)
    if s < 1e-12:
        return (0.0, 0.0, 0.0)
    return (x / s * angle, y / s * angle, z / s * angle)


def quat_slerp(a: Quat, b: Quat, t: float) -> Quat:
    a, b = quat_normalize(a), quat_normalize(b)
    dot = sum(x * y for x, y in zip(a, b))
    if dot < 0.0:
        b, dot = tuple(-c for c in b), -dot  # type: ignore[assignment]
    if dot > 0.9995:  # nearly parallel: lerp and renormalize
        return quat_normalize(tuple(x + t * (y - x) for x, y in zip(a, b)))  # type: ignore[arg-type]
    theta = math.acos(dot)
    s0 = math.sin((1.0 - t) * theta) / math.sin(theta)
    s1 = math.sin(t * theta) / math.sin(theta)
    return tuple(s0 * x + s1 * y for x, y in zip(a, b))  # type: ignore[return-value]


def quat_angle_between(a: Quat, b: Quat) -> float:
    dot = abs(sum(x * y for x, y in zip(quat_normalize(a), quat_normalize(b))))
    return 2.0 * math.acos(min(1.0, dot))


# --------------------------------------------------------------------------- rigid transforms
@dataclass(frozen=True)
class SE3:
    """Rigid transform T_a_b: rotation q (wxyz) then translation t, both expressed in frame a."""
    q: Quat = (1.0, 0.0, 0.0, 0.0)
    t: Vec3 = (0.0, 0.0, 0.0)

    def apply(self, p: Vec3) -> Vec3:
        r = quat_rotate(self.q, p)
        return (r[0] + self.t[0], r[1] + self.t[1], r[2] + self.t[2])

    def __matmul__(self, other: "SE3") -> "SE3":
        """Composition: (T_a_b @ T_b_c) = T_a_c."""
        return SE3(quat_normalize(quat_mul(self.q, other.q)), self.apply(other.t))

    def inverse(self) -> "SE3":
        qi = quat_conj(self.q)
        ti = quat_rotate(qi, self.t)
        return SE3(qi, (-ti[0], -ti[1], -ti[2]))


def interpolate_pose(a: SE3, b: SE3, t: float) -> SE3:
    """Linear translation, spherical rotation."""
    return SE3(quat_slerp(a.q, b.q, t), tuple(x + t * (y - x) for x, y in zip(a.t, b.t)))  # type: ignore[arg-type]


# --------------------------------------------------------------------------- camera models
@dataclass(frozen=True)
class Pinhole:
    fx: float
    fy: float
    cx: float
    cy: float

    def project(self, p: Vec3) -> tuple[float, float] | None:
        x, y, z = p
        if z <= 0.0:
            return None  # behind the camera
        return (self.fx * x / z + self.cx, self.fy * y / z + self.cy)

    def unproject(self, uv: tuple[float, float]) -> Vec3:
        """Unit ray direction for a pixel."""
        x = (uv[0] - self.cx) / self.fx
        y = (uv[1] - self.cy) / self.fy
        n = math.sqrt(x * x + y * y + 1.0)
        return (x / n, y / n, 1.0 / n)


@dataclass(frozen=True)
class KannalaBrandt:
    """Equidistant fisheye with polynomial radial distortion (the KB / "KannalaBrandtK3" family).

    theta_d = theta * (1 + k1 theta^2 + k2 theta^4 + k3 theta^6 + k4 theta^8),
    r_pixels = f * theta_d. Unlike a pinhole, points beyond 90 degrees off-axis still
    project, which is why head-mounted wide-angle cameras use fisheye models.
    Project Aria's Fisheye624 adds more radial terms plus tangential and thin-prism
    distortion; this simplified model shows the mechanism, not the exact Aria model.
    """
    f: float
    cx: float
    cy: float
    k: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 0.0)

    def _distort(self, theta: float) -> float:
        t2 = theta * theta
        k1, k2, k3, k4 = self.k
        return theta * (1.0 + t2 * (k1 + t2 * (k2 + t2 * (k3 + t2 * k4))))

    def project(self, p: Vec3) -> tuple[float, float] | None:
        x, y, z = p
        r_xy = math.hypot(x, y)
        if r_xy == 0.0 and z <= 0.0:
            return None
        theta = math.atan2(r_xy, z)
        if r_xy < 1e-15:
            return (self.cx, self.cy)
        rd = self.f * self._distort(theta)
        return (self.cx + rd * x / r_xy, self.cy + rd * y / r_xy)

    def unproject(self, uv: tuple[float, float], iters: int = 20) -> Vec3:
        """Unit ray direction; inverts the distortion polynomial with Newton's method."""
        du, dv = uv[0] - self.cx, uv[1] - self.cy
        rd = math.hypot(du, dv) / self.f
        if rd < 1e-15:
            return (0.0, 0.0, 1.0)
        theta = rd
        k1, k2, k3, k4 = self.k
        for _ in range(iters):
            t2 = theta * theta
            f_val = self._distort(theta) - rd
            df = 1.0 + 3 * k1 * t2 + 5 * k2 * t2 ** 2 + 7 * k3 * t2 ** 3 + 9 * k4 * t2 ** 4
            step = f_val / df
            theta -= step
            if abs(step) < 1e-14:
                break
        s = math.sin(theta)
        return (s * du / math.hypot(du, dv), s * dv / math.hypot(du, dv), math.cos(theta))


def camera_point_to_world(p_camera: Vec3, T_world_device: SE3, T_device_camera: SE3) -> Vec3:
    """Chain used for every head-mounted measurement: camera -> device (calibration) -> world (SLAM)."""
    return (T_world_device @ T_device_camera).apply(p_camera)
