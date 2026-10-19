"""Native C correctness checks; these do not claim a Yocto package/image build."""

import json
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest


class ObservationPackageTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        compiler = shutil.which("cc")
        if compiler is None:
            raise unittest.SkipTest("native C compiler unavailable")
        cls.temp = tempfile.TemporaryDirectory(prefix="observation-package-")
        cls.addClassCleanup(cls.temp.cleanup)
        cls.binary = Path(cls.temp.name) / "observation-check"
        source = Path(__file__).parent / "meta-robot-course/recipes-apps/observation-check/files/observation-check.c"
        subprocess.run([compiler, "-std=c11", "-Wall", "-Wextra", "-Werror", "-O2",
                        str(source), "-o", str(cls.binary)], check=True, capture_output=True)

    def run_case(self, *arguments):
        return subprocess.run([str(self.binary), *arguments], text=True, capture_output=True)

    def test_fresh(self):
        result = self.run_case("100", "125", "40")
        self.assertEqual(result.returncode, 0)
        self.assertEqual(json.loads(result.stdout), {"status": "fresh", "age_ms": 25})

    def test_inclusive_deadline(self):
        result = self.run_case("100", "140", "40")
        self.assertEqual(result.returncode, 0)
        self.assertEqual(json.loads(result.stdout)["age_ms"], 40)

    def test_stale(self):
        result = self.run_case("100", "141", "40")
        self.assertEqual(result.returncode, 2)
        self.assertEqual(json.loads(result.stdout), {"status": "stale", "age_ms": 41})

    def test_future_timestamp(self):
        result = self.run_case("150", "140", "40")
        self.assertEqual(result.returncode, 65)
        self.assertEqual(json.loads(result.stdout), {"status": "clock_error"})

    def test_invalid_integer_inputs(self):
        for value in ("-1", "+1", "1.5", " 1", "NaN", "", "18446744073709551616"):
            with self.subTest(value=value):
                result = self.run_case(value, "140", "40")
                self.assertEqual(result.returncode, 64)
                self.assertEqual(result.stdout, "")

    def test_uint64_boundary(self):
        result = self.run_case("18446744073709551614", "18446744073709551615", "1")
        self.assertEqual(result.returncode, 0)
        self.assertEqual(json.loads(result.stdout)["age_ms"], 1)

    def test_argument_count(self):
        self.assertEqual(self.run_case("0", "1").returncode, 64)


if __name__ == "__main__":
    unittest.main()
