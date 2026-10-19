"""Actual MuJoCo checks against independent mechanics and timing contracts."""
from pathlib import Path
from contextlib import chdir
import sys
import tempfile
import unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import mujoco
import mujoco_bridge as bridge


class PhysicsBridgeTests(unittest.TestCase):
    def test_one_step_force_over_mass_and_updated_velocity_position(self):
        cfg=bridge.Dynamics(mass_kg=2, physics_dt_s=.05, policy_period_s=.05, duration_s=.05)
        model,data,_=bridge.make_model(cfg)
        bridge.reset(model,data,(0,0));data.ctrl[0]=.5;mujoco.mj_step(model,data)
        self.assertAlmostEqual(data.qvel[0],.0125)
        self.assertAlmostEqual(data.qpos[0],.000625)
        self.assertAlmostEqual(data.qfrc_actuator[0],.5)

    def test_engine_limit_operates_even_without_python_clipping(self):
        model,data,_=bridge.make_model(bridge.Dynamics(force_limit_n=1))
        data.ctrl[0]=10;mujoco.mj_step(model,data)
        self.assertAlmostEqual(data.qfrc_actuator[0],1)
        self.assertAlmostEqual(data.qvel[0],.1)

    def test_reset_clears_time_controls_external_force_and_state(self):
        model,data,_=bridge.make_model(bridge.Dynamics())
        data.ctrl[0]=2;data.qfrc_applied[0]=5;mujoco.mj_step(model,data)
        bridge.reset(model,data,(.3,-.4))
        self.assertEqual(float(data.time),0)
        self.assertEqual(float(data.ctrl[0]),0)
        self.assertEqual(float(data.qfrc_applied[0]),0)
        self.assertEqual(float(data.qpos[0]),.3)
        self.assertEqual(float(data.qvel[0]),-.4)
        self.assertAlmostEqual(data.xpos[1,0],.3)

    def test_entire_matched_trajectory_against_scalar_reference(self):
        policy=lambda e,v:.8*e-.6*v
        initial=(1.2,-.2)
        result=bridge.simulate(policy,bridge.Dynamics(),initial)
        expected=bridge.analytical_trace(policy,initial)
        self.assertEqual(len(result['trace']),len(expected))
        for row,(x,v) in zip(result['trace'],expected):
            self.assertAlmostEqual(row['position_m'],x,places=12)
            self.assertAlmostEqual(row['velocity_m_s'],v,places=12)

    def test_substeps_change_integration_without_more_policy_calls(self):
        coarse=bridge.simulate(lambda e,v:1,bridge.Dynamics(duration_s=.1),(0,0))
        fine=bridge.simulate(lambda e,v:1,bridge.Dynamics(physics_dt_s=.01,duration_s=.1),(0,0))
        self.assertEqual(coarse['policy_calls'],1);self.assertEqual(fine['policy_calls'],1)
        self.assertEqual(coarse['physics_steps'],1);self.assertEqual(fine['physics_steps'],10)
        self.assertAlmostEqual(coarse['final_position_m'],.01)
        self.assertAlmostEqual(fine['final_position_m'],.0055)
        self.assertAlmostEqual(fine['final_velocity_m_s'],.1)
        # Constant-acceleration exact position is .005m after .1s.
        self.assertLess(abs(fine['final_position_m']-.005),abs(coarse['final_position_m']-.005))

    def test_mass_adapter_and_force_saturation_change_acceleration(self):
        wrong=bridge.simulate(lambda e,v:.5,bridge.Dynamics(mass_kg=3,duration_s=.1),(0,0))
        correct=bridge.simulate(lambda e,v:.5,bridge.Dynamics(mass_kg=3,assumed_mass_kg=3,duration_s=.1),(0,0))
        limited=bridge.simulate(lambda e,v:.5,bridge.Dynamics(mass_kg=3,assumed_mass_kg=3,force_limit_n=.5,duration_s=.1),(0,0))
        self.assertAlmostEqual(wrong['final_velocity_m_s'],.05/3)
        self.assertAlmostEqual(correct['final_velocity_m_s'],.05)
        self.assertAlmostEqual(limited['final_velocity_m_s'],.05/3)
        self.assertEqual(limited['force_clips'],1)

    def test_euler_joint_damping_is_implicit(self):
        result=bridge.simulate(lambda e,v:0,bridge.Dynamics(mass_kg=2,damping_n_s_m=4,duration_s=.1),(0,1))
        self.assertAlmostEqual(result['final_velocity_m_s'],1/(1+.1*4/2))
        self.assertAlmostEqual(result['final_position_m'],.1/(1+.1*4/2))

    def test_nonfinite_policy_action_rejected_before_stepping(self):
        cfg=bridge.Dynamics();engine=bridge.make_model(cfg)
        with self.assertRaises(ValueError):bridge.simulate(lambda e,v:float('nan'),cfg,(1,0),engine=engine)
        self.assertEqual(engine[1].time,0)

    def test_simulator_auto_reset_is_failure_not_success(self):
        # MuJoCo normally resets huge finite states to zero during mj_step.
        # The clock still advances to .1, so finite-state/time checks alone miss it.
        cfg=bridge.Dynamics(duration_s=.1)
        with tempfile.TemporaryDirectory() as directory, chdir(directory):
            for initial in ((1e12,0),(0,1e12)):
                with self.subTest(initial=initial), self.assertRaisesRegex(ValueError,"MuJoCo warning"):
                    bridge.simulate(lambda e,v:0,cfg,initial)

    def test_cached_physics_must_match_but_policy_adapter_can_change(self):
        cfg=bridge.Dynamics(mass_kg=3,duration_s=.1)
        engine=bridge.make_model(cfg)
        wrong=bridge.Dynamics(duration_s=.1)
        with self.assertRaisesRegex(ValueError,"reused engine"):
            bridge.simulate(lambda e,v:1,wrong,(0,0),engine=engine)
        changed_adapter=bridge.Dynamics(mass_kg=3.0,assumed_mass_kg=3,duration_s=.1)
        result=bridge.simulate(lambda e,v:1,changed_adapter,(0,0),engine=engine)
        self.assertAlmostEqual(result['final_velocity_m_s'],.1)

    def test_success_and_sampled_settling_boundary_are_strict(self):
        cfg=bridge.Dynamics(duration_s=.2)
        outside=bridge.simulate(lambda e,v:0,cfg,(.1,0))
        inside=bridge.simulate(lambda e,v:0,cfg,(.099,0))
        crossing=bridge.simulate(lambda e,v:0,cfg,(.101,-.05))
        self.assertFalse(outside['success'])
        self.assertIsNone(outside['sampled_settling_time_s'])
        self.assertTrue(inside['success'])
        self.assertEqual(inside['sampled_settling_time_s'],0)
        self.assertTrue(crossing['success'])
        self.assertAlmostEqual(crossing['sampled_settling_time_s'],.1)

if __name__ == '__main__':unittest.main()
