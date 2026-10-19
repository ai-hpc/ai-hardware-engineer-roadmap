import math
from dataclasses import replace
import unittest
from vehicle_lab import (State, Observation, ObservationGate, MODEL_CONTRACT, bicycle_step,
                         pure_pursuit, stopping_distance, time_to_collision, decode_model_output,
                         replay_faults, simulate)


class VehicleContracts(unittest.TestCase):
    def test_straight_constant_acceleration(self):
        s = bicycle_step(State(0, 0, 0, 2), 0, 1, 2)
        self.assertEqual(s, State(6, 0, 0, 4))

    def test_exact_quarter_circle(self):
        s = bicycle_step(State(0, 0, 0, 1), math.atan(2.8 / 4), 0, 2 * math.pi)
        self.assertAlmostEqual(s.x, 4)
        self.assertAlmostEqual(s.y, 4)
        self.assertAlmostEqual(s.yaw, math.pi / 2)

    def test_braking_does_not_reverse(self):
        self.assertEqual(bicycle_step(State(0, 0, 0, 2), 0, -4, 2), State(.5, 0, 0, 0))

    def test_steering_and_target_sign(self):
        self.assertAlmostEqual(pure_pursuit(State(0, 0, 0, 1), (4, 2)), math.atan(.56))
        self.assertLess(pure_pursuit(State(), (8, 0)), 0)
        with self.assertRaises(ValueError): pure_pursuit(State(), (-8, 0))

    def test_rotated_vehicle_geometry(self):
        self.assertAlmostEqual(pure_pursuit(State(0, 0, math.pi/2, 1), (-2, 4)), math.atan(.56))

    def test_stopping_and_closing_speed(self):
        self.assertEqual(stopping_distance(10, .1, 4), 13.5)
        self.assertEqual(time_to_collision(20, 10, 6), 5)
        self.assertIsNone(time_to_collision(20, 6, 10))
        self.assertEqual(time_to_collision(0, 6, 10), 0)

    def test_domains(self):
        for args in [(State(speed=-1), 0, 0, .1), (State(), math.pi/2, 0, .1), (State(), 0, 0, 0)]:
            with self.assertRaises(ValueError): bicycle_step(*args)
        for args in [(1, 0, 0), (-1, .1, 4), (1, math.nan, 4)]:
            with self.assertRaises(ValueError): stopping_distance(*args)

    def test_fault_replay_known_answers(self):
        self.assertEqual([r['result'] for r in replay_faults()],
                         ['accepted', 'stale', 'invalid_time_order', 'wrong_clock', 'out_of_order',
                          'wrong_schema_or_frame', 'nonfinite', 'not_ready', 'accepted'])

    def test_capture_age_boundary_and_timestamp_regression(self):
        gate = ObservationGate(.15)
        self.assertEqual(gate.check(Observation(2, .85, .99, State()), 1), 'accepted')
        self.assertEqual(gate.check(Observation(3, .84, .99, State()), 1), 'stale')
        self.assertEqual(gate.check(Observation(3, .849, .99, State()), .999), 'out_of_order')

    def test_rejected_frame_does_not_poison_recovery(self):
        gate = ObservationGate()
        self.assertEqual(gate.check(Observation(999, -9, 1, State()), 1), 'stale')
        self.assertEqual(gate.check(Observation(2, 1, 1, State()), 1), 'accepted')

    def test_invalid_identity_and_publish_before_capture(self):
        gate = ObservationGate();o = Observation(1, 1, 1, State())
        self.assertEqual(gate.check(replace(o, frame_id=True), 1), 'invalid_frame_id')
        self.assertEqual(gate.check(replace(o, ready_s=.9), 1), 'invalid_time_order')

    def test_model_units_shape_nonfinite_and_range(self):
        self.assertEqual(decode_model_output([[-.12]], MODEL_CONTRACT), -.12)
        for value in [[-.12], [[1.]], [[math.nan]], [[True]], [[.1, .2]]]:
            with self.assertRaises(ValueError): decode_model_output(value, MODEL_CONTRACT)
        with self.assertRaises(ValueError):
            decode_model_output([[-.12]], {**MODEL_CONTRACT, 'output_units':'steering_wheel_deg'})

    def test_baseline_tracks_and_respects_rate_bound(self):
        r = simulate()
        self.assertEqual(r['accepted'], 160)
        self.assertLess(abs(r['final_state']['y']), .03)
        previous = 0.
        for sample in r['trace']:
            self.assertLessEqual(abs(sample['steering_rad'] - previous), .025 + 1e-12)
            self.assertLessEqual(abs(sample['steering_rad']), .35)
            previous = sample['steering_rad']

    def test_stale_run_stops_but_does_not_complete_lane_task(self):
        r = simulate(.2)
        self.assertEqual(r['accepted'], 0)
        self.assertEqual(r['stale'], 156)
        self.assertEqual(r['final_state']['speed'], 0)
        self.assertEqual(r['final_state']['y'], 1)
        self.assertAlmostEqual(r['final_state']['x'], 25/6)


if __name__ == '__main__':
    unittest.main()
