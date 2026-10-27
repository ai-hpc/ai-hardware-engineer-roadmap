import math
import unittest

from ego_geometry import (SE3, KannalaBrandt, Pinhole, camera_point_to_world, quat_angle_between,
                          quat_from_axis_angle, quat_rotate, quat_to_axis_angle)
from retarget import GripperMap, HandFrame, Limits, relative_actions, resample, retarget_episode


def close(a, b, tol=1e-9):
    return all(abs(x - y) <= tol for x, y in zip(a, b))


class TestFrames(unittest.TestCase):
    def test_inverse_and_composition(self):
        T = SE3(quat_from_axis_angle((0, 0, 1), 0.7), (0.3, -0.2, 1.1))
        p = (0.5, 0.1, -0.4)
        self.assertTrue(close((T.inverse() @ T).apply(p), p))
        U = SE3(quat_from_axis_angle((1, 0, 0), -0.4), (0.0, 0.5, 0.0))
        self.assertTrue(close((T @ U).apply(p), T.apply(U.apply(p))))

    def test_camera_to_world_chain(self):
        # Device rotated 90 deg about z and moved 1 m along x; camera 10 cm in front of the device.
        T_world_device = SE3(quat_from_axis_angle((0, 0, 1), math.pi / 2), (1.0, 0.0, 0.0))
        T_device_camera = SE3((1.0, 0.0, 0.0, 0.0), (0.1, 0.0, 0.0))
        p = camera_point_to_world((0.0, 0.0, 0.0), T_world_device, T_device_camera)
        self.assertTrue(close(p, (1.0, 0.1, 0.0)))

    def test_rotation_vector_roundtrip(self):
        q = quat_from_axis_angle((0.2, -0.5, 0.8), 1.3)
        rv = quat_to_axis_angle(q)
        self.assertAlmostEqual(math.hypot(*rv), 1.3, places=12)


class TestCameraModels(unittest.TestCase):
    def test_pinhole_roundtrip_and_behind_camera(self):
        cam = Pinhole(600, 600, 320, 240)
        p = (0.1, -0.05, 0.8)
        ray = cam.unproject(cam.project(p))
        n = math.sqrt(sum(c * c for c in p))
        self.assertTrue(close(ray, tuple(c / n for c in p), 1e-12))
        self.assertIsNone(cam.project((0.0, 0.0, -1.0)))

    def test_fisheye_roundtrip_wide_angle(self):
        cam = KannalaBrandt(240.0, 704.0, 704.0, (0.03, -0.01, 0.002, -0.0003))
        # 80 degrees off-axis: far outside any useful pinhole field of view
        theta = math.radians(80)
        p = (math.sin(theta) * 0.6, math.sin(theta) * 0.8, math.cos(theta))
        ray = cam.unproject(cam.project(p))
        self.assertTrue(close(ray, p, 1e-9))

    def test_wrong_intrinsics_are_detectable(self):
        """Negative case: the same pixel unprojected with the wrong focal length lands far away."""
        true_cam = KannalaBrandt(240.0, 704.0, 704.0)
        wrong_cam = KannalaBrandt(300.0, 704.0, 704.0)
        p = (0.4, 0.2, 0.6)
        n = math.sqrt(sum(c * c for c in p))
        uv = true_cam.project(p)
        err = math.acos(max(-1.0, min(1.0, sum(a * b / n for a, b in zip(wrong_cam.unproject(uv), p)))))
        self.assertGreater(math.degrees(err), 5.0)


def synthetic_reach(n=31, fps=30.0, dx_per_s=0.15, yaw_per_s=0.3, close_at=0.6):
    """A hand moving along +x while yawing, closing the grip partway through."""
    frames = []
    for i in range(n):
        t = i / fps
        q = quat_from_axis_angle((0, 0, 1), yaw_per_s * t)
        wrist = SE3(q, (0.4 + dx_per_s * t, 0.0, 0.2))
        gap = 0.08 if t < close_at else 0.02
        thumb = wrist.apply((0.05, gap / 2, 0.0))
        index = wrist.apply((0.05, -gap / 2, 0.0))
        frames.append(HandFrame(t, wrist, thumb, index))
    return frames


class TestRetarget(unittest.TestCase):
    def test_constant_motion_gives_constant_relative_actions(self):
        frames = resample(synthetic_reach(), 10.0)
        poses = [f.T_world_wrist for f in frames]
        acts = relative_actions(poses)
        mags = [math.hypot(*tr) for tr, _ in acts]
        rots = [math.hypot(*rv) for _, rv in acts]
        self.assertLess(max(mags) - min(mags), 1e-9)
        self.assertAlmostEqual(rots[0], 0.3 / 10.0, places=9)

    def test_resample_keeps_endpoints(self):
        frames = synthetic_reach()
        rs = resample(frames, 10.0)
        self.assertAlmostEqual(rs[0].t, 0.0)
        self.assertAlmostEqual(rs[-1].t, frames[-1].t, places=9)
        self.assertLess(quat_angle_between(rs[-1].T_world_wrist.q, frames[-1].T_world_wrist.q), 1e-9)

    def test_gripper_mapping(self):
        g = GripperMap(closed_m=0.015, open_m=0.09)
        self.assertEqual(g.opening((0, 0, 0), (0.0, 0.0, 0.01)), 0.0)
        self.assertEqual(g.opening((0, 0, 0), (0.0, 0.0, 0.2)), 1.0)
        self.assertAlmostEqual(g.opening((0, 0, 0), (0.0, 0.0, 0.0525)), 0.5)

    def test_episode_is_feasible_in_correct_frame(self):
        out = retarget_episode(synthetic_reach(), 10.0, SE3(), SE3())
        self.assertTrue(out["report"]["ok"], out["report"])
        self.assertEqual(len(out["actions"][0]), 7)
        # A hand closed around an object keeps a ~2 cm thumb-index gap, so it maps near (not to) 0.
        self.assertLess(out["actions"][-1][6], 0.1)          # closed at the end
        self.assertGreater(out["actions"][0][6], 0.5)      # open at the start

    def test_swapped_calibration_is_caught(self):
        """Negative case: using T_world_base where T_base_world is needed puts the hand
        outside the robot workspace, and the report says so."""
        T_base_world = SE3(quat_from_axis_angle((0, 0, 1), math.pi / 2), (0.0, -0.6, 0.0))
        ok = retarget_episode(synthetic_reach(), 10.0, SE3(), T_base_world,
                              limits=Limits(workspace_min=(-0.2, -1.0, 0.0), workspace_max=(0.2, 0.0, 0.6)))
        swapped = retarget_episode(synthetic_reach(), 10.0, SE3(), T_base_world.inverse(),
                                   limits=Limits(workspace_min=(-0.2, -1.0, 0.0), workspace_max=(0.2, 0.0, 0.6)))
        self.assertTrue(ok["report"]["ok"], ok["report"])
        self.assertGreater(swapped["report"]["outside_workspace"], 0)

    def test_speed_limit_flags_fast_motion(self):
        fast = retarget_episode(synthetic_reach(dx_per_s=1.5), 10.0, SE3(), SE3(),
                                limits=Limits(workspace_min=(-2, -2, -2), workspace_max=(2, 2, 2)))
        self.assertGreater(fast["report"]["too_fast_steps"], 0)

    def test_rotation_matches_quat_rotate(self):
        q = quat_from_axis_angle((0, 1, 0), 0.5)
        self.assertTrue(close(quat_rotate(q, (0, 1, 0)), (0, 1, 0)))


if __name__ == "__main__":
    unittest.main()
