import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest

HERE = Path(__file__).resolve().parent
CXX = os.environ.get("CXX", "g++")


@unittest.skipUnless(shutil.which(CXX), "C++17 compiler required for the native queue lab")
class SpscTests(unittest.TestCase):
    def compile_and_run(self, flags):
        with tempfile.TemporaryDirectory(prefix="course-spsc-") as directory:
            binary = Path(directory) / "queue"
            build = subprocess.run([CXX, "-std=c++17", "-Wall", "-Wextra", "-Werror",
                                    "-pthread", *flags, str(HERE / "spsc_demo.cpp"), "-o", str(binary)],
                                   capture_output=True, text=True, timeout=60)
            self.assertEqual(build.returncode, 0, build.stderr)
            return subprocess.run([str(binary)], capture_output=True, text=True, timeout=30)

    def test_optimized_native_reference_and_threads(self):
        result = self.compile_and_run(["-O2"])
        self.assertEqual(result.returncode, 0, result.stderr)
        evidence = json.loads(result.stdout)
        self.assertEqual(evidence["sequential_operations"], 300000)
        self.assertEqual(evidence["concurrent_items"], 400000)
        self.assertIsInstance(evidence["indices_lock_free_on_this_host"], bool)

    def test_address_and_undefined_behavior_sanitizers(self):
        result = self.compile_and_run(["-O1", "-g", "-fsanitize=address,undefined", "-fno-omit-frame-pointer"])
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_missing_full_check_mutant_is_detected(self):
        result = self.compile_and_run(["-O2", "-DCOURSE_MUTATE_SKIP_FULL_CHECK"])
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("full queue accepted overwrite", result.stderr)


if __name__ == "__main__":
    unittest.main()
