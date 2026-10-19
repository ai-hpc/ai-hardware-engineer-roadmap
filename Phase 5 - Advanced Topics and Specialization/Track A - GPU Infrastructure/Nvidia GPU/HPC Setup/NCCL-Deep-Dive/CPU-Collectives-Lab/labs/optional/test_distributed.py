"""Optional real-framework integration tests, excluded from stdlib discovery."""
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

LAB = Path(__file__).resolve().parents[1]


class DistributedTest(unittest.TestCase):
    def run_case(self, *args):
        with tempfile.TemporaryDirectory() as directory:
            report = Path(directory) / "report.json"
            env = dict(os.environ, OMP_NUM_THREADS="1", GLOO_SOCKET_IFNAME="lo")
            proc = subprocess.run([sys.executable, "-m", "torch.distributed.run",
                                   "--standalone", "--nproc-per-node=2", str(LAB / "distributed_demo.py"),
                                   "--output", str(report), *args], env=env,
                                  capture_output=True, text=True, timeout=45)
            self.assertEqual(proc.returncode, 0, proc.stdout + proc.stderr)
            return json.loads(report.read_text())

    def test_unequal_batches_need_sample_weighting(self):
        result = self.run_case()
        self.assertAlmostEqual(result["serial_gradient"][0], -47.5)
        self.assertAlmostEqual(result["serial_gradient"][1], -13.5)
        self.assertAlmostEqual(result["comparisons"]["local_mean"]["max_gradient_error"], 5.708333333333336)
        self.assertLess(result["comparisons"]["global_sample_mean"]["max_parameter_error"], 1e-12)

    def test_equal_batches_recover_rank_average(self):
        result = self.run_case("--equal-batches")
        self.assertLess(result["comparisons"]["local_mean"]["max_gradient_error"], 1e-12)

    def test_mismatched_contract_is_rejected_on_both_ranks(self):
        result = self.run_case("--mismatch")
        self.assertIn("rejected_before_data_collective", result)


if __name__ == "__main__": unittest.main()
