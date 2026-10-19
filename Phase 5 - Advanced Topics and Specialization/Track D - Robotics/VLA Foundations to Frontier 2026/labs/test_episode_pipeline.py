import copy
import json
import math
from pathlib import Path
import unittest

from episode_pipeline import (CLOCK, Executor, causal_frame, chunk, convert_bundle,
                              convert_episode, denormalize, fit_normalizer, masked_mse,
                              message, normalize, replay)


def fixture():
    return json.loads(Path(__file__).with_name("episode-fixture.json").read_text())


class DatasetTests(unittest.TestCase):
    def test_conversion_matches_physical_motion_across_rates(self):
        bundle = fixture()
        rows = convert_bundle(bundle)
        self.assertEqual([r["action"] for r in rows[:4]],
                         [[0.0, 0.01], [0.0, 0.0], [0.0, 0.01], [0.0, 0.01]])
        self.assertEqual([r["action"] for r in rows[4:8]],
                         [[0.0, 0.01], [0.0, 0.01], [0.0, 0.0], [0.0, 0.0]])
        # Same original recording, different output interval: integrated motion
        # is unchanged. Naively repeating 40ms deltas would double this result.
        episode = bundle["episodes"][0]
        coarse = convert_episode(episode, bundle["sources"][episode["source"]], 40_000)
        self.assertEqual([r["action"] for r in coarse], [[0.0, 0.01], [0.0, 0.02]])
        self.assertAlmostEqual(sum(r["action"][1] for r in coarse), 0.03)

    def test_causal_alignment_rejects_captured_but_unreceived_frame(self):
        frames = fixture()["episodes"][0]["frames"]
        self.assertEqual(causal_frame(frames, 20_000, 25_000)["id"], 0)
        self.assertEqual(causal_frame(frames, 40_000, 25_000)["id"], 2)
        self.assertEqual(causal_frame(frames, 60_000, 25_000)["id"], 2)
        self.assertIsNone(causal_frame(frames, 61_000, 25_000))
        self.assertIsNone(causal_frame(frames, -1, 25_000))

    def test_state_is_causal_hold_not_future_interpolation(self):
        rows = convert_bundle(fixture())
        self.assertEqual(rows[5]["time_us"], 20_000)
        self.assertEqual(rows[5]["state_base_m"], [0.0, 0.0])
        self.assertEqual(rows[6]["state_base_m"], [0.0, 0.02])
        self.assertEqual(rows[5]["instruction"], "move in base positive y")
        self.assertEqual(rows[5]["rgb_2x2"], fixture()["episodes"][1]["frames"][0]["rgb_2x2"])

    def test_normalizer_ignores_held_out_actions(self):
        rows = convert_bundle(fixture())
        stats = fit_normalizer(rows)
        self.assertEqual(stats["count"], 8)
        self.assertAlmostEqual(stats["center"][1], 0.00625)
        self.assertAlmostEqual(stats["scale"][1], math.sqrt(0.0000234375))
        self.assertEqual(stats["constant"], [True, False])
        self.assertEqual(stats["scale"][0], 1.0)
        for row in rows:
            if row["split"] == "test":
                row["action"] = [900.0, -1000.0]
        self.assertEqual(fit_normalizer(rows), stats)
        for actual, expected in zip(denormalize(normalize([0.0, 0.02], stats), stats), (0.0, 0.02)):
            self.assertAlmostEqual(actual, expected, places=14)
        rows[0]["period_us"] = 40_000
        with self.assertRaisesRegex(ValueError, "cadence"):
            fit_normalizer(rows)

    def test_missing_observation_is_excluded_from_fit(self):
        rows = convert_bundle(fixture())
        rows[0]["observation_valid"] = False
        self.assertEqual(fit_normalizer(rows)["count"], 7)

    def test_final_observation_not_an_action_and_chunk_never_crosses_reset(self):
        rows = convert_bundle(fixture())
        self.assertEqual(len(rows), 12)
        targets, mask = chunk(rows, 2, 4)
        self.assertEqual(targets, [[0.0, 0.01]] * 4)
        self.assertEqual(mask, [1, 1, 0, 0])
        predictions = [[0.0, 0.0]] * 2 + [[100.0, -100.0]] * 2
        self.assertAlmostEqual(masked_mse(predictions, targets, mask), 0.00005)
        with self.assertRaisesRegex(ValueError, "empty mask"):
            masked_mse(predictions, targets, [0] * 4)
        bundle = fixture()
        bundle["episodes"][0]["steps"][-1]["action"] = [0, 0]
        with self.assertRaisesRegex(ValueError, "no action target"):
            convert_bundle(bundle)

    def test_session_leak_and_copied_provenance_detected(self):
        bundle = fixture()
        bundle["episodes"][2]["session"] = "morning"
        with self.assertRaisesRegex(ValueError, "session leaks"):
            convert_bundle(bundle)
        bundle = fixture()
        bundle["episodes"][2]["origin"] = bundle["episodes"][0]["origin"]
        with self.assertRaisesRegex(ValueError, "duplicated source"):
            convert_bundle(bundle)

    def test_bad_units_reflection_and_unstated_interpolation_fail(self):
        for field, value in (("unit", "m"), ("rotation_to_base", [[1, 0], [0, -1]]),
                             ("interpolation", "nearest_sample")):
            with self.subTest(field=field):
                bundle = fixture()
                bundle["sources"]["camera_deltas"][field] = value
                with self.assertRaises(ValueError):
                    convert_bundle(bundle)

    def test_duplicate_frame_clock_jump_and_nonfinite_action_fail(self):
        for problem in ("duplicate", "clock", "timestamp", "nan"):
            with self.subTest(problem=problem):
                bundle = fixture()
                episode = bundle["episodes"][0]
                if problem == "duplicate":
                    episode["frames"][1]["id"] = 0
                elif problem == "clock":
                    episode["frames"][1]["clock"] = "wall_us"
                elif problem == "timestamp":
                    episode["steps"][1]["time_us"] = 0
                else:
                    episode["steps"][1]["action"] = [float("nan"), 0]
                with self.assertRaises(ValueError):
                    convert_bundle(bundle)

    def test_truncation_still_has_a_final_observation_without_action(self):
        bundle = fixture()
        bundle["episodes"][0]["terminal"] = False
        rows = convert_bundle(bundle)
        self.assertEqual(len(rows), 12)
        self.assertTrue(rows[3]["truncated"])
        self.assertFalse(rows[3]["terminated"])
        self.assertTrue(rows[7]["terminated"])
        self.assertFalse(rows[7]["truncated"])


class ExecutionTests(unittest.TestCase):
    def setUp(self):
        self.stats = fit_normalizer(convert_bundle(fixture()))
        self.executor = Executor(self.stats)

    def test_late_start_expiry_and_exact_boundary(self):
        self.assertEqual(self.executor.submit(message(self.stats), 70_000), [4, 5])
        self.assertEqual(self.executor.dispatch(70_000)["status"], "fallback")
        accepted = self.executor.dispatch(80_000)
        self.assertEqual((accepted["index"], accepted["age_us"]), (4, 80_000))
        self.assertEqual(self.executor.dispatch(100_000)["status"], "execute")
        self.assertEqual(self.executor.dispatch(120_000)["status"], "fallback")

    def test_zero_action_is_valid_and_skipped_ticks_are_counted(self):
        zero = message(self.stats, ready_us=0, action=(0.0, 0.0))
        self.executor.submit(zero, 0)
        result = self.executor.dispatch(40_000)
        self.assertEqual(result["status"], "execute")
        for value in result["action_m"]:
            self.assertAlmostEqual(value, 0.0, places=14)
        self.assertEqual(result["missed"], 2)

    def test_out_of_order_reply_cannot_replace_newer_result(self):
        new = message(self.stats, request=2, observation_us=60_000, ready_us=80_000,
                      action=(0.0, 0.02))
        self.executor.submit(new, 80_000)
        old = message(self.stats)
        with self.assertRaisesRegex(ValueError, "out-of-order"):
            self.executor.submit(old, 90_000)
        result = self.executor.dispatch(100_000)
        self.assertEqual(result["request"], 2)
        self.assertAlmostEqual(result["action_m"][0], 0.0, places=14)
        self.assertAlmostEqual(result["action_m"][1], 0.02, places=14)

    def test_invalid_newer_message_does_not_poison_acceptance(self):
        bad = message(self.stats, request=999)
        bad["normalized_actions"][0][0] = float("nan")
        with self.assertRaises(ValueError):
            self.executor.submit(bad, 70_000)
        self.assertEqual(self.executor.last_request, -1)
        self.assertFalse(self.executor.queue)
        self.executor.submit(message(self.stats, request=1), 70_000)
        self.assertEqual(self.executor.dispatch(80_000)["request"], 1)

    def test_reset_discards_buffer_and_rejects_late_previous_epoch(self):
        old = message(self.stats)
        self.executor.submit(old, 70_000)
        self.executor.reset()
        self.assertFalse(self.executor.queue)
        with self.assertRaisesRegex(ValueError, "reset epoch"):
            self.executor.submit(old, 0)
        self.assertEqual(self.executor.dispatch(0)["status"], "fallback")
        new = message(self.stats, epoch=2, ready_us=10_000)
        self.executor.submit(new, 10_000)
        self.assertEqual(self.executor.dispatch(20_000)["status"], "execute")

    def test_wrong_artifact_clock_future_expired_and_bound_fail(self):
        for field, value, now in (("normalizer_sha256", "wrong", 70_000),
                                  ("clock", "wall_us", 70_000),
                                  ("period_us", 40_000, 70_000),
                                  ("observation_us", 80_000, 70_000),
                                  ("ready_us", 80_000, 70_000),
                                  ("ready_us", 120_000, 120_000)):
            with self.subTest(field=field, value=value):
                candidate = message(self.stats)
                candidate[field] = value
                with self.assertRaises(ValueError):
                    Executor(self.stats).submit(candidate, now)
        with self.assertRaisesRegex(ValueError, "bound"):
            Executor(self.stats).submit(message(self.stats, action=(0.0, 0.1)), 70_000)

    def test_normalizer_corruption_is_detected_even_with_copied_hash(self):
        bad = copy.deepcopy(self.stats)
        bad["scale"][1] *= 10
        with self.assertRaisesRegex(ValueError, "content/hash"):
            Executor(bad).submit(message(self.stats), 70_000)

    def test_duplicate_dispatch_and_backwards_clock_fail(self):
        self.executor.dispatch(20_000)
        with self.assertRaisesRegex(ValueError, "duplicate dispatch"):
            self.executor.dispatch(20_000)
        with self.assertRaisesRegex(ValueError, "backward"):
            self.executor.dispatch(0)

    def test_full_replay_has_executed_zeros_and_separate_fallback(self):
        trace = replay(self.stats)
        statuses = [r["status"] for r in trace if "status" in r]
        self.assertEqual(statuses, ["fallback", "execute", "execute", "execute", "execute", "fallback"])
        self.assertEqual(sum("rejection" in r for r in trace), 2)


if __name__ == "__main__":
    unittest.main()
