import copy
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

from analyze import nearest_rank, summarize

HERE = Path(__file__).resolve().parent


class TraceTests(unittest.TestCase):
    def setUp(self):
        self.trace = {"start_ns": 0, "period_ns": 1000, "samples": [
            {"release_ns": 1000, "wake_ns": 1100, "finish_ns": 1300},
            {"release_ns": 2000, "wake_ns": 2900, "finish_ns": 3000},
            {"release_ns": 3000, "wake_ns": 3100, "finish_ns": 4100}]}

    def test_deadline_equality_is_on_time_and_late_completion_is_miss(self):
        result = summarize(self.trace)
        self.assertEqual(result["deadline_misses"], 1)
        self.assertEqual(result["completion_response"]["max_observed_us"], 1.1)
        self.assertEqual(result["wake_lateness"]["p99_us"], .9)

    def test_quantile_convention(self):
        self.assertEqual(nearest_rank(list(range(1, 101)), .99), 99)
        self.assertEqual(nearest_rank([42], .99), 42)
        for values, probability in (([], .5), ([1], 0), ([1], 2)):
            with self.assertRaises(ValueError): nearest_rank(values, probability)

    def test_corrupt_traces_rejected(self):
        for key, value in (("release_ns", 999), ("wake_ns", 999), ("finish_ns", 1050)):
            trace = copy.deepcopy(self.trace)
            trace["samples"][0][key] = value
            with self.assertRaises(ValueError): summarize(trace)
        self.trace["samples"] = []
        with self.assertRaises(ValueError): summarize(self.trace)

    def test_relative_sleep_adds_work_to_nominal_drift(self):
        # Idealized exact timer: 1000ns sleep + 200ns work. This is a model.
        relative = {"start_ns": 0, "period_ns": 1000, "samples": [
            {"release_ns": (i+1)*1000, "wake_ns": (i+1)*1000+i*200,
             "finish_ns": (i+1)*1000+(i+1)*200} for i in range(10)]}
        self.assertEqual(summarize(relative)["final_wake_lateness_us"], 1.8)
        self.assertEqual(summarize(relative)["deadline_misses"], 5)


@unittest.skipUnless(sys.platform.startswith("linux") and shutil.which("cc"),
                     "native timing experiment needs Linux and a C compiler")
class NativeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.directory = tempfile.TemporaryDirectory()
        cls.binary = str(Path(cls.directory.name) / "periodic")
        subprocess.run(["cc", "-std=c11", "-O2", "-Wall", "-Wextra", "-Werror",
                        str(HERE / "periodic.c"), "-o", cls.binary], check=True,
                       capture_output=True, text=True)

    @classmethod
    def tearDownClass(cls):
        cls.directory.cleanup()

    def test_actual_timers_produce_valid_complete_traces(self):
        for mode in ("absolute", "relative"):
            result = subprocess.run([self.binary, mode, "12", "1000", "100"],
                                    check=True, capture_output=True, text=True, timeout=10)
            trace = json.loads(result.stdout)
            self.assertEqual(trace["mode"], mode)
            self.assertEqual(trace["clock"], "CLOCK_MONOTONIC")
            self.assertEqual(summarize(trace)["samples"], 12)
            # Timing tails depend on the host. No fixed latency or speedup assertion.

    def test_invalid_and_excessive_work_rejected(self):
        for args in (("wrong", "10", "1000", "100"), ("absolute", "0", "1000", "0"),
                     ("absolute", "10", "0", "0"), ("relative", "10", "1000", "-1"),
                     ("absolute", "nan", "1000", "0"), ("absolute", "2000", "100000", "0")):
            result = subprocess.run([self.binary, *args], capture_output=True, timeout=5)
            self.assertEqual(result.returncode, 2)


if __name__ == "__main__":
    unittest.main()
