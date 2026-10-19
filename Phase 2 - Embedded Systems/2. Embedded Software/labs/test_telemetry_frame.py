"""Compile/run native C, sanitizers, and two deliberately incorrect implementations."""
from pathlib import Path
import os
import shutil
import subprocess
import tempfile
import unittest

HERE = Path(__file__).resolve().parent


class TelemetryFrameTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cc = shutil.which(os.environ.get("CC", "cc"))
        if not cls.cc:
            raise unittest.SkipTest("A C11 compiler is required")

    def compile_run(self, replacement=None, sanitized=False):
        with tempfile.TemporaryDirectory(prefix="course-frame-") as directory:
            folder = Path(directory)
            source = (HERE / "telemetry_frame.c").read_text()
            if replacement:
                old, new = replacement
                self.assertEqual(source.count(old), 1, "mutation anchor changed")
                source = source.replace(old, new)
            path = folder / "candidate.c"
            path.write_text(source)
            binary = folder / "frame-test"
            flags = ["-std=c11", "-Wall", "-Wextra", "-Wconversion", "-Werror"]
            if sanitized:
                flags += ["-O1", "-g", "-fno-omit-frame-pointer", "-fsanitize=address,undefined"]
            else:
                flags += ["-O2"]
            build = subprocess.run([self.cc, *flags, "-I", str(HERE), str(path),
                                    str(HERE / "telemetry_frame_test.c"), "-o", str(binary)],
                                   text=True, capture_output=True, timeout=30)
            self.assertEqual(build.returncode, 0, build.stdout + build.stderr)
            environment = dict(os.environ, ASAN_OPTIONS="detect_leaks=1:halt_on_error=1",
                               UBSAN_OPTIONS="halt_on_error=1:print_stacktrace=1")
            return subprocess.run([str(binary)], text=True, capture_output=True,
                                  timeout=20, env=environment)

    def test_actual_c_parser_boundary_corruption_recovery_and_wrap(self):
        result = self.compile_run()
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn("99 payload fixtures/all byte splits", result.stdout)

    def test_address_and_undefined_behavior_sanitizers(self):
        result = self.compile_run(sanitized=True)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn("PASS:", result.stdout)

    def test_checker_rejects_crc_bypass(self):
        result = self.compile_run((
            "if (expected_crc != actual_crc)",
            "if (actual_crc == UINT16_C(0xdead) && expected_crc == UINT16_C(0xbeef))"))
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("TELEMETRY_BAD_CRC", result.stderr)

    def test_checker_rejects_future_timestamp_acceptance(self):
        result = self.compile_run((
            "elapsed < UINT32_C(0x80000000) && elapsed <= maximum_age",
            "(elapsed <= maximum_age || elapsed == UINT32_MAX)"))
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("telemetry_age_valid(99, 100, 20)", result.stderr)


if __name__ == "__main__":
    unittest.main()
