import math
import unittest

from timed_policy_lab import CommandGate, Observation, Timing, run, simulate


class GateTests(unittest.TestCase):
    def observation(self, sequence=0, captured=0, session=4, position=1.0):
        return Observation(session, sequence, captured, position, 0.0)

    def test_deadline_covers_whole_held_action_interval(self):
        gate = CommandGate(4, 250)
        self.assertEqual(gate.accept(self.observation(), 200, -0.5), "accepted")
        self.assertEqual(gate.action_for_interval(240, 10, 0.0), (-0.5, "policy"))
        self.assertEqual(gate.action_for_interval(250, 10, 0.0), (0.0, "fallback"))

    def test_exact_deadline_acceptance_does_not_create_extra_lease(self):
        gate = CommandGate(4, 250)
        self.assertEqual(gate.accept(self.observation(), 250, -0.5), "accepted")
        self.assertEqual(gate.action_for_interval(250, 10, 0.0)[1], "fallback")
        self.assertEqual(CommandGate(4).accept(self.observation(), 251, -0.5), "stale")

    def test_duplicate_does_not_refresh_expiry(self):
        gate = CommandGate(4)
        observation = self.observation()
        gate.accept(observation, 100, -0.5)
        self.assertEqual(gate.accept(observation, 240, 0.9), "old_sequence")
        self.assertEqual(gate.action_for_interval(250, 10, 0)[1], "fallback")
        self.assertEqual(gate.acceleration, -0.5)

    def test_old_session_and_future_time_are_rejected(self):
        gate = CommandGate(4)
        self.assertEqual(gate.accept(self.observation(session=3), 100, 0), "wrong_session")
        self.assertEqual(gate.accept(self.observation(session=True), 100, 0), "wrong_session")
        self.assertEqual(gate.accept(self.observation(captured=101), 100, 0), "future_timestamp")
        self.assertIsNone(gate.observation)

    def test_nonfinite_values_do_not_replace_last_valid_command(self):
        gate = CommandGate(4)
        gate.accept(self.observation(), 100, -0.5)
        for value in (math.nan, math.inf, -math.inf):
            self.assertEqual(gate.accept(self.observation(1, 100, position=value), 100, 0), "nonfinite")
            self.assertEqual(gate.accept(self.observation(1, 100), 100, value), "nonfinite")
        self.assertEqual(gate.last_sequence, 0)

    def test_higher_sequence_cannot_move_capture_time_backwards(self):
        gate = CommandGate(4)
        gate.accept(self.observation(3, 100), 150, 0)
        self.assertEqual(gate.accept(self.observation(4, 90), 160, 0), "backward_timestamp")

    def test_actions_are_clipped_and_fallback_is_damping(self):
        gate = CommandGate(4)
        gate.accept(self.observation(), 100, 100)
        self.assertEqual(gate.action_for_interval(100, 10, 0), (1.0, "policy"))
        self.assertEqual(gate.action_for_interval(300, 10, 0.2), (-0.4, "fallback"))
        self.assertEqual(gate.action_for_interval(300, 10, -5), (1.0, "fallback"))

    def test_invalid_time_and_fallback_encoder_are_rejected(self):
        gate = CommandGate(4)
        for now, dt, velocity in ((-1, 10, 0), (0, 0, 0), (0, 10, math.nan)):
            with self.assertRaises(ValueError):
                gate.action_for_interval(now, dt, velocity)


class SimulationTests(unittest.TestCase):
    def test_two_steps_match_hand_calculated_semi_implicit_damping(self):
        timing = Timing("hand", 100, transport_ms=1000, sensor_period_ms=100,
                        dt_ms=100, max_age_ms=300, duration_ms=200,
                        outage_start_ms=0, outage_end_ms=0)
        report = simulate(lambda e, v: 99, timing, (0, 1), retain_trace=True)
        self.assertAlmostEqual(report["trace"][0]["position_m"], 0.09)
        self.assertAlmostEqual(report["final_position_m"], 0.17)
        self.assertAlmostEqual(report["final_velocity_m_s"], 0.8)
        self.assertEqual(report["accepted_commands"], 0)

    def test_capture_transport_compute_timestamps(self):
        timing = Timing("latency", 120, duration_ms=300, outage_start_ms=0, outage_end_ms=0)
        report = simulate(lambda e, v: -0.5, timing, retain_trace=True)
        first = next(row for row in report["trace"] if row["source"] == "policy")
        self.assertEqual(first["time_ms"], 150)  # interval [140,150): capture0 + transport20 + compute120
        self.assertEqual(first["observation_age_start_ms"], 140)
        self.assertLessEqual(report["max_applied_age_ms"], 250)

    def test_queue_capacity_and_freshness_are_different_contracts(self):
        expert = lambda e, v: 0.8 * e - 0.6 * v
        fifo = simulate(expert, Timing("fifo", 120, queue_mode="fifo", check_age=False))
        latest = simulate(expert, Timing("latest", 120))
        self.assertEqual(fifo["max_pending_observations"], 64)
        self.assertGreater(fifo["max_applied_age_ms"], 5000)
        self.assertEqual(latest["max_pending_observations"], 1)
        self.assertLessEqual(latest["max_applied_age_ms"], 250)
        self.assertGreater(latest["discarded_observations"], 0)

    def test_over_budget_computation_has_no_accepted_policy_commands(self):
        report = simulate(lambda e, v: e, Timing("too_slow", 350))
        self.assertEqual(report["accepted_commands"], 0)
        self.assertGreater(report["rejected_commands"], 0)
        self.assertEqual(report["fallback_fraction"], 1)
        self.assertIsNone(report["max_applied_age_ms"])

    def test_exact_deadline_results_can_be_accepted_without_ever_being_applied(self):
        timing = Timing("at_deadline", 250, transport_ms=0, duration_ms=1000,
                        outage_start_ms=0, outage_end_ms=0)
        report = simulate(lambda e, v: -0.5, timing)
        self.assertEqual(report["accepted_commands"], 3)
        self.assertEqual(report["applied_policy_intervals"], 0)
        self.assertEqual(report["distinct_commands_applied"], 0)
        self.assertEqual(report["fallback_fraction"], 1)
        self.assertIsNone(report["max_applied_age_ms"])

    def test_simultaneous_arrival_replaces_pending_before_worker_selects_next(self):
        calls = []
        def policy(error, velocity):
            calls.append((error, velocity))
            return 0.0
        timing = Timing("event_tie", 200, transport_ms=0, sensor_period_ms=100,
                        dt_ms=100, max_age_ms=1000, duration_ms=500,
                        outage_start_ms=0, outage_end_ms=0)
        report = simulate(policy, timing, initial=(0, 1))
        self.assertEqual(len(calls), 2)
        self.assertEqual(calls[0], (0, 1))
        # Before first result at t=200: bounded fallback gave x=.17, v=.8.
        # Arrival at t=200 replaces t=100 before the worker starts its next job.
        self.assertAlmostEqual(calls[1][0], -0.17)
        self.assertAlmostEqual(calls[1][1], 0.8)
        self.assertEqual(report["discarded_observations"], 2)
        self.assertEqual(report["distinct_commands_applied"], 2)
        self.assertEqual(report["applied_policy_intervals"], 3)

    def test_sensor_outage_expires_commands_and_recovers(self):
        report = simulate(lambda e, v: 0.8*e-0.6*v, Timing("outage", 10, transport_ms=0), retain_trace=True)
        during = [row for row in report["trace"] if 3300 <= row["time_ms"] < 3500]
        after = [row for row in report["trace"] if 3600 <= row["time_ms"] < 3700]
        self.assertTrue(all(row["source"] == "fallback" for row in during))
        self.assertTrue(all(row["source"] == "policy" for row in after))

    def test_timing_and_initial_state_validation(self):
        for kwargs in ({"inference_ms": 0}, {"inference_ms": 125},
                       {"inference_ms": 120, "queue_mode": "unbounded"},
                       {"inference_ms": 120, "transport_ms": -1}):
            with self.assertRaises(ValueError):
                Timing("invalid", **kwargs)
        with self.assertRaises(ValueError):
            simulate(lambda e, v: 0, Timing("valid", 10), (math.nan, 0))


class TrainedIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.report = run(retain_trace=False)

    def test_training_partition_and_pairing_are_preserved(self):
        groups = [set(ids) for ids in self.report["split_episode_ids"].values()]
        self.assertEqual(sum(map(len, groups)), len(set.union(*groups)))
        reference = [(row["initial_position_m"], row["initial_velocity_m_s"])
                     for row in self.report["scenarios"][0]["per_episode"]]
        for scenario in self.report["scenarios"]:
            self.assertEqual(reference, [(row["initial_position_m"], row["initial_velocity_m_s"])
                                        for row in scenario["per_episode"]])

    def test_low_prediction_error_does_not_imply_queue_success(self):
        rows = {row["timing"]["name"]: row for row in self.report["scenarios"]}
        self.assertLess(self.report["held_out_action_rmse_m_s2"], 0.03)
        self.assertEqual(rows["fast_latest"]["successes"], 12)
        self.assertEqual(rows["slow_latest_with_age_check"]["successes"], 12)
        self.assertEqual(rows["slow_fifo_without_age_check"]["successes"], 0)
        # Final tolerance can be met by fallback alone; do not credit the model.
        over_budget = rows["over_budget_latest"]
        self.assertGreater(over_budget["successes"], 0)
        self.assertTrue(all(row["accepted_commands"] == 0 for row in over_budget["per_episode"]))


if __name__ == "__main__":
    unittest.main(verbosity=2)
