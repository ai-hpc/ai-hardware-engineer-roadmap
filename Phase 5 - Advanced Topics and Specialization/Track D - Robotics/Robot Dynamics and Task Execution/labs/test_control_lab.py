import math
import unittest

from dynamics import Pendulum, pd_rollout, required_motor_current
from optimal_control import finite_action_mpc, finite_lqr, lqr_rollout, position_range, step
from task_execution import (compose_pose, cubic_duration, cubic_sample, evaluate_lift,
                            inverse_pose, required_finger_force, segment_clearance)


class DynamicsTests(unittest.TestCase):
    def test_horizontal_gravity_and_current_hand_calculation(self):
        model = Pendulum()
        self.assertEqual(model.inertia, .5)
        self.assertAlmostEqual(model.gravity_torque(math.pi / 2), 9.81)
        self.assertAlmostEqual(model.acceleration(math.pi / 2, 0, 0), -19.62)
        self.assertAlmostEqual(required_motor_current(9.81, 50, .8, .1), 2.4525)

    def test_gravity_is_potential_energy_derivative(self):
        model, angle, epsilon = Pendulum(), .7, 1e-6
        derivative = (model.energy(angle + epsilon, 0) - model.energy(angle - epsilon, 0)) / (2 * epsilon)
        self.assertAlmostEqual(derivative, model.gravity_torque(angle), places=7)

    def test_compensation_holds_initial_equilibrium(self):
        trace = pd_rollout(Pendulum(), math.pi / 2, initial=(math.pi / 2, 0), steps=50)
        self.assertTrue(all(abs(row["velocity_rad_s"]) < 1e-12 for row in trace))

    def test_torque_limit_prevents_horizontal_equilibrium(self):
        trace = pd_rollout(Pendulum(), math.pi / 2, torque_limit=8, initial=(math.pi / 2, 0), steps=1)
        self.assertEqual(trace[0]["applied_torque_Nm"], 8)
        self.assertLess(trace[0]["velocity_rad_s"], 0)

    def test_compensation_removes_PD_static_error_in_fixture(self):
        model, target = Pendulum(), math.pi / 2
        plain = pd_rollout(model, target, compensate=False)[-1]
        compensated = pd_rollout(model, target)[-1]
        self.assertGreater(abs(target - plain["angle_rad"]), .5)
        self.assertLess(abs(target - compensated["angle_rad"]), .001)

    def test_invalid_model_and_nonfinite_targets(self):
        for values in ({"mass": 0}, {"length": -1}, {"damping": -1}, {"gravity": float("nan")}):
            with self.assertRaises(ValueError):
                Pendulum(**values)
        with self.assertRaises(ValueError):
            pd_rollout(Pendulum(), float("nan"))
        with self.assertRaises(ValueError):
            required_motor_current(1, 50, 1.2, .1)


class OptimalControlTests(unittest.TestCase):
    def test_exact_held_acceleration_update(self):
        self.assertEqual(step((2., 3.), 4., .5), (4., 5.))

    def test_one_step_gain_against_hand_calculation(self):
        gains, _ = finite_lqr(.1, 1)
        self.assertAlmostEqual(gains[0][0], .05 / .11025)
        self.assertAlmostEqual(gains[0][1], .105 / .11025)

    def test_value_function_matches_executed_finite_horizon_cost(self):
        gains, p = finite_lqr(.1, 40)
        x = (1.2, -.3)
        result = lqr_rollout(x, gains, .1)
        expected = sum(x[i] * p[i][j] * x[j] for i in range(2) for j in range(2))
        self.assertAlmostEqual(result["cost"], expected, places=9)
        self.assertAlmostEqual(p[0][1], p[1][0], places=10)

    def test_first_action_minimizes_independent_two_step_objective(self):
        # Minimize the second action analytically, then finite-difference the
        # complete two-step cost around the first action selected by Riccati.
        state, dt = (.4, -.2), .1
        gains, _ = finite_lqr(dt, 2)
        optimum = -sum(k * x for k, x in zip(gains[0], state))
        def cost(first):
            x1 = step(state, first, dt)
            second = -(.05 * x1[0] + .105 * x1[1]) / .11025
            x2 = step(x1, second, dt)
            return (10*state[0]**2 + state[1]**2 + .1*first**2
                    + 10*x1[0]**2 + x1[1]**2 + .1*second**2
                    + 10*x2[0]**2 + x2[1]**2)
        self.assertAlmostEqual((cost(optimum + 1e-5) - cost(optimum - 1e-5)) / 2e-5, 0, places=8)
        self.assertLess(cost(optimum), cost(optimum + .5))

    def test_clipping_is_not_the_unconstrained_optimum(self):
        gains, _ = finite_lqr(.1, 40)
        reference = lqr_rollout((1., 0.), gains, .1)
        clipped = lqr_rollout((1., 0.), gains, .1, acceleration_limit=1.)
        self.assertGreater(clipped["cost"], reference["cost"])

    def test_mpc_brakes_before_boundary(self):
        plan = finite_action_mpc((.35, .5))
        self.assertIsNotNone(plan)
        self.assertEqual(plan["actions"][0], -1)
        self.assertTrue(all(state[0] <= .5 for state in plan["states"]))

    def test_mpc_reports_infeasible_instead_of_returning_zero_action(self):
        self.assertIsNone(finite_action_mpc((.45, .5)))

    def test_continuous_constraint_catches_hidden_turning_point(self):
        self.assertAlmostEqual(step((.49, .2), -1., .4)[0], .49)
        self.assertAlmostEqual(position_range((.49, .2), -1., .4)[1], .51)
        self.assertIsNone(finite_action_mpc((.49, .2), dt=.4, horizon=1, actions=(-1.,)))

    def test_invalid_cost_and_unbounded_search_rejected(self):
        with self.assertRaises(ValueError):
            finite_lqr(.1, 10, r=0)
        with self.assertRaises(ValueError):
            finite_lqr(.1, 10, q=(-1, 1))
        with self.assertRaises(ValueError):
            finite_action_mpc((0, 0), horizon=8, actions=tuple(range(10)))


class ExecutionTests(unittest.TestCase):
    def test_free_endpoints_do_not_establish_free_segment(self):
        self.assertGreater(segment_clearance((-1., 0), (-1., 0), (0., 0), .2, .1), 0)
        self.assertGreater(segment_clearance((1., 0), (1., 0), (0., 0), .2, .1), 0)
        self.assertAlmostEqual(segment_clearance((-1., 0), (1., 0), (0., 0), .2, .1), -.3)

    def test_nearest_point_is_clamped_to_segment(self):
        self.assertAlmostEqual(segment_clearance((1., 0), (2., 0), (0., 0), .2, .1), .7)

    def test_clearance_margin_and_tracking_error(self):
        self.assertAlmostEqual(segment_clearance((-1., .6), (1., .6), (0., 0), .2, .1), .3)
        self.assertLess(segment_clearance((-1., .25), (1., .25), (0., 0), .2, .1), 0)
        self.assertLess(segment_clearance((-1., .6), (1., .6), (0., 0), .2, .1, .31), 0)

    def test_cubic_endpoint_and_exact_peak_constraints(self):
        duration = cubic_duration(1., .5, 1.)
        self.assertEqual(duration, 3.)
        start, velocity0, acceleration0 = cubic_sample((0., 0), (1., 0), duration, 0)
        end, velocity1, _ = cubic_sample((0., 0), (1., 0), duration, duration)
        _, middle_velocity, _ = cubic_sample((0., 0), (1., 0), duration, duration/2)
        self.assertEqual(start, (0., 0))
        self.assertEqual(end, (1., 0))
        self.assertEqual(velocity0, (0., 0))
        self.assertEqual(velocity1, (0., 0))
        self.assertAlmostEqual(math.hypot(*middle_velocity), .5)
        self.assertAlmostEqual(math.hypot(*acceleration0), 2/3)

    def test_grasp_transform_and_roundtrip(self):
        pose = (.6, .2, 0, math.pi/2)
        grasp = compose_pose(pose, (.1, 0, .05, 0))
        for a, b in zip(grasp[:3], (.6, .3, .05)):
            self.assertAlmostEqual(a, b)
        relative = compose_pose(inverse_pose(pose), grasp)
        for a, b in zip(relative, (.1, 0, .05, 0)):
            self.assertAlmostEqual(a, b)

    def test_finger_force_is_per_contact_not_total(self):
        self.assertAlmostEqual(required_finger_force(.5, .4, 1.5), 9.196875)
        self.assertAlmostEqual(required_finger_force(.5, .2, 1.5), 18.39375)

    def test_hand_endpoint_success_can_hide_object_failure(self):
        before, hand = (0, 0, 0, 0), (0, 0, .05, 0)
        after_hand = (0, 0, .25, 0)
        good = evaluate_lift(hand, before, after_hand, (0, 0, .2, 0))
        bad = evaluate_lift(hand, before, after_hand, before)
        self.assertTrue(good["success"])
        self.assertFalse(bad["success"])
        self.assertAlmostEqual(bad["retention_error"]["translation_m"], .2)

    def test_retention_without_lift_is_not_task_success(self):
        result = evaluate_lift((0,0,.05,0), (0,0,0,0), (0,0,.05,0), (0,0,0,0))
        self.assertTrue(result["retained"])
        self.assertFalse(result["success"])

    def test_invalid_geometry_and_friction(self):
        with self.assertRaises(ValueError):
            segment_clearance((0,0), (1,0), (0,0), -1, .1)
        with self.assertRaises(ValueError):
            required_finger_force(.5, 0)
        with self.assertRaises(ValueError):
            cubic_sample((0,0), (1,0), 0, 0)


if __name__ == "__main__":
    unittest.main()
