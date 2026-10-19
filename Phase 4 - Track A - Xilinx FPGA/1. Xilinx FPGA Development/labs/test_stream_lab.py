"""Integration tests run Icarus, including faults a waveform-only demo misses."""
import unittest

from run_stream_lab import FAULTS, run, tool_paths


class StreamContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            tool_paths()
        except RuntimeError as error:
            raise unittest.SkipTest(str(error)) from error

    def test_directed_and_random_backpressure(self):
        for seed in (1, 17, 2026):
            with self.subTest(seed=seed):
                result = run(seed)
                self.assertEqual(result["status"], "passed")
                self.assertEqual(result["accepted"], result["delivered"] + result["flushed"])
                self.assertEqual(result["flushed"], 1)
                for metric in ("stalls", "replacements", "saturated"):
                    self.assertGreater(result[metric], 0)

    def test_scoreboard_rejects_protocol_arithmetic_and_reset_faults(self):
        for fault, (_, _, assertion) in FAULTS.items():
            with self.subTest(fault=fault):
                result = run(fault=fault)
                self.assertEqual(result["status"], "fault_detected")
                self.assertEqual(result["assertion"], assertion)


if __name__ == "__main__":
    unittest.main()
