"""Optional actual MuJoCo checks; run with the shared hashed physics environment."""
from contextlib import chdir
import math
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import mujoco
from dynamics import Pendulum, pd_rollout
from mujoco_pendulum import (SphericalBob, difference, joint_inertia, make_engine,
                             native_rollout, reset)


class NativePendulumTests(unittest.TestCase):
    def test_inertia_gravity_sign_and_center_of_mass(self):
        model, data, _ = make_engine(Pendulum())
        for angle in (-math.pi / 2, 0., math.pi / 2):
            with self.subTest(angle=angle):
                reset(model, data, (angle, 0.))
                self.assertAlmostEqual(joint_inertia(model, data), .50008, places=12)
                self.assertAlmostEqual(data.qfrc_bias[0], 9.81 * math.sin(angle), places=12)
                self.assertAlmostEqual(data.geom_xpos[0, 0], -.5 * math.sin(angle), places=12)
                self.assertAlmostEqual(data.geom_xpos[0, 2], -.5 * math.cos(angle), places=12)

    def test_acceleration_and_static_holding_torque(self):
        plant = Pendulum()
        model, data, _ = make_engine(plant)
        reset(model, data, (.4, .7))
        data.ctrl[0] = .3
        mujoco.mj_forward(model, data)
        self.assertAlmostEqual(data.qfrc_passive[0], -.14, places=12)
        self.assertAlmostEqual(data.qacc[0], (.3 - .14 - 9.81 * math.sin(.4)) / .50008, places=12)
        reset(model, data, (math.pi / 2, 0.))
        data.ctrl[0] = 9.81
        mujoco.mj_step(model, data)
        self.assertAlmostEqual(data.qvel[0], 0., places=12)
        self.assertAlmostEqual(data.qpos[0], math.pi / 2, places=12)

    def test_explicit_and_implicit_damping_steps_are_distinct(self):
        dt, angle, velocity, torque = .05, .4, .7, .3
        for implicit in (False, True):
            with self.subTest(implicit=implicit):
                model, data, _ = make_engine(Pendulum(), dt=dt, implicit_damping=implicit)
                reset(model, data, (angle, velocity))
                data.ctrl[0] = torque
                mujoco.mj_step(model, data)
                if implicit:
                    expected = (velocity + dt * (torque - 9.81 * math.sin(angle)) / .50008) / (1 + dt * .2 / .50008)
                else:
                    expected = velocity + dt * (torque - .2 * velocity - 9.81 * math.sin(angle)) / .50008
                self.assertAlmostEqual(data.qvel[0], expected, places=12)
                self.assertAlmostEqual(data.qpos[0], angle + dt * expected, places=12)

    def test_native_actuator_clamps_without_python_clipping(self):
        model, data, _ = make_engine(Pendulum(), torque_limit=8.)
        reset(model, data, (math.pi / 2, 0.))
        data.ctrl[0] = 100.
        mujoco.mj_step(model, data)
        self.assertEqual(data.qfrc_actuator[0], 8.)
        self.assertLess(data.qvel[0], 0.)

    def test_full_matched_finite_bob_reference(self):
        target = math.pi / 2
        expected = pd_rollout(SphericalBob(), target)
        actual = native_rollout(Pendulum(), target, implicit_damping=False)["trace"]
        self.assertLess(difference(expected, actual, "angle_rad"), 1e-11)
        self.assertLess(difference(expected, actual, "velocity_rad_s"), 1e-11)

    def test_native_pd_gravity_compensation_and_insufficient_torque(self):
        target = math.pi / 2
        plain = native_rollout(Pendulum(), target, compensate=False)["trace"]
        compensated = native_rollout(Pendulum(), target)["trace"]
        limited = native_rollout(Pendulum(), target, torque_limit=8.)["trace"]
        self.assertGreater(target - plain[-1]["angle_rad"], .8)
        self.assertLess(abs(target - compensated[-1]["angle_rad"]), .001)
        self.assertGreater(abs(limited[-1]["velocity_rad_s"]), .2)
        self.assertLessEqual(max(abs(row["applied_torque_Nm"]) for row in limited), 8.)

    def test_reset_clears_forces_and_instability_is_rejected(self):
        model, data, _ = make_engine(Pendulum())
        data.ctrl[0], data.qfrc_applied[0] = 3., 2.
        mujoco.mj_step(model, data)
        reset(model, data, (.4, -.2))
        self.assertEqual(data.time, 0.)
        self.assertEqual(data.ctrl[0], 0.)
        self.assertEqual(data.qfrc_applied[0], 0.)
        with tempfile.TemporaryDirectory() as directory, chdir(directory):
            with self.assertRaisesRegex(ValueError, "invalid MuJoCo episode"):
                native_rollout(Pendulum(), 0., initial=(0., 1e12), steps=1)


if __name__ == "__main__":
    unittest.main()
