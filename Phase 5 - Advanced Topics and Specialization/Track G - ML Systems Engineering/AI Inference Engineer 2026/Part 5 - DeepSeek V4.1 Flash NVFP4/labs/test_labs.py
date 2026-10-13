import unittest
from checkpoint_audit import audit
from fp4_layout import decode, pack, unpack, reconstruct
from inference_metrics import demo_rows, summarize


class CheckpointTests(unittest.TestCase):
    def test_counts_shards_not_tensors_and_unknown_size(self):
        result = audit({"text_config": {"num_hidden_layers": 40}},
                       {"weight_map": {"a": "one", "b": "one", "c": "two"}})
        self.assertEqual(result["unique_shards"], 2)
        self.assertEqual(result["layers"], 40)
        self.assertIsNone(result["declared_payload_bytes"])
        self.assertEqual(result["runtime_compatibility"], "not established")

    def test_payload_units(self):
        result = audit({}, {"weight_map": {"x": "one"},
                            "metadata": {"total_size": 2**30}})
        self.assertEqual(result["declared_payload_gib"], 1)

    def test_invalid_size(self):
        for size in (-1, True, 2.5):
            with self.assertRaises(ValueError):
                audit({}, {"weight_map": {"x": "one"}, "metadata": {"total_size": size}})


class FormatTests(unittest.TestCase):
    def test_all_nibbles_and_asymmetric_order(self):
        codes = list(range(16))
        self.assertEqual(unpack(pack(codes)), codes)
        self.assertEqual(pack([1, 10]), bytes([0xA1]))
        self.assertEqual(decode(15), -6)

    def test_wrong_scale_changes_reconstruction(self):
        self.assertEqual(reconstruct([1, 10], [2.0], 2), [1, -2])
        self.assertNotEqual(reconstruct([1, 10], [2.0], 2),
                            reconstruct([1, 10], [4.0], 2))

    def test_rejects_incomplete_and_invalid_layout(self):
        with self.assertRaises(ValueError):
            pack([1])
        with self.assertRaises(ValueError):
            reconstruct([1, 2, 3], [1.0], 2)
        with self.assertRaises(ValueError):
            reconstruct([1, 2], [float("nan")], 2)


class MetricsTests(unittest.TestCase):
    def test_failures_stay_in_window_and_denominator(self):
        r = summarize(demo_rows(), .4, .3)
        self.assertEqual(r["measurement_requests"], 3)
        self.assertEqual(r["completed_output_tokens_per_s"], 2)
        self.assertAlmostEqual(r["good_output_tokens_per_s"], 4/3)
        self.assertAlmostEqual(r["failure_rate"], 1/3)

    def test_one_token_has_no_tpot(self):
        r = summarize([{"request_id": "one", "phase": "measurement", "status": "ok",
                        "start_s": 0, "first_token_s": 1, "end_s": 1, "output_tokens": 1}], 2, 1)
        self.assertIsNone(r["request_average_tpot_p95_s"])
        self.assertEqual(r["slo_passing_requests"], 1)

    def test_all_failures_are_not_zero_latency_success(self):
        r = summarize([demo_rows()[-1]], .4, .3)
        self.assertEqual(r["good_output_tokens_per_s"], 0)
        self.assertIsNone(r["ttft_p95_s"])
        self.assertEqual(r["failure_rate"], 1)

    def test_invalid_time_nan_duplicate_and_empty(self):
        cases = [[], [dict(demo_rows()[1], end_s=-1)],
                 [dict(demo_rows()[1], first_token_s=float("nan"))],
                 [demo_rows()[1], demo_rows()[1]]]
        for rows in cases:
            with self.assertRaises(ValueError):
                summarize(rows, .4, .3)

    def test_partial_failed_tokens_excluded(self):
        row = dict(demo_rows()[-1], output_tokens=99, first_token_s=2.2)
        r = summarize([row], .4, .3)
        self.assertEqual(r["completed_output_tokens"], 0)


if __name__ == "__main__":
    unittest.main()
