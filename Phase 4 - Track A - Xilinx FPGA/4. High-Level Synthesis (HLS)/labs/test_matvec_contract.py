"""Actual host C++ execution; no HLS synthesis or FPGA timing is implied."""
from pathlib import Path
import os
import shutil
import subprocess
import tempfile
import unittest

HERE = Path(__file__).resolve().parent


@unittest.skipUnless(shutil.which("g++"), "g++ required for native C++ lab")
class MatvecContractTests(unittest.TestCase):
    def run_variant(self, flags):
        with tempfile.TemporaryDirectory(prefix="course-hls-contract-") as tmp:
            binary = Path(tmp) / "verify"
            command = ["g++", "-std=c++17", "-Wall", "-Wextra", "-Werror", *flags,
                       str(HERE / "verify_contract.cpp"), "-o", str(binary)]
            subprocess.run(command, check=True, capture_output=True, text=True, timeout=30)
            env = dict(os.environ, ASAN_OPTIONS="detect_leaks=1:halt_on_error=1",
                       UBSAN_OPTIONS="halt_on_error=1")
            return subprocess.run([str(binary)], capture_output=True, text=True,
                                  timeout=10, env=env)

    def test_scalar_reference_all_shapes_and_boundaries(self):
        result = self.run_variant(["-O2"])
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("verified_shapes=512", result.stdout)

    def test_address_and_undefined_behavior_sanitizers(self):
        result = self.run_variant(["-O1", "-g", "-fsanitize=address,undefined",
                                   "-fno-omit-frame-pointer", "-fno-pie", "-no-pie"])
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_ragged_and_saturation_mutants_are_detected(self):
        for define, failure in [("COURSE_MUTATE_DROP_TAIL", "ragged fixture mismatch"),
                                ("COURSE_MUTATE_WRAP_OUTPUT", "positive saturation mismatch")]:
            with self.subTest(mutation=define):
                result = self.run_variant(["-O2", "-D" + define])
                self.assertNotEqual(result.returncode, 0)
                self.assertIn(failure, result.stderr)


if __name__ == "__main__":
    unittest.main()
