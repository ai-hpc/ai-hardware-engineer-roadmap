"""Compile and execute native code; no simulated SIMD implementation."""
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest

HERE = Path(__file__).resolve().parent

class NativeSimdTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.compiler = shutil.which("g++") or shutil.which("clang++")
        if not cls.compiler:
            raise unittest.SkipTest("C++ compiler unavailable")
        cls.temp = tempfile.TemporaryDirectory(prefix="course-simd-")
        cls.addClassCleanup(cls.temp.cleanup)
        cls.directory = Path(cls.temp.name)
        cls.binary = cls.build("native")

    @classmethod
    def build(cls, name, *flags):
        binary = cls.directory / name
        result = subprocess.run([cls.compiler, "-std=c++17", "-O2", "-Wall", "-Wextra",
            "-ffp-contract=off", *flags, str(HERE / "simd_contracts.cpp"), "-o", str(binary)],
            text=True, capture_output=True, timeout=90)
        if result.returncode:
            raise AssertionError(result.stdout + result.stderr)
        return binary

    def run_case(self, mode, binary=None):
        result = subprocess.run([str(binary or self.binary), mode], text=True,
            capture_output=True, timeout=30,
            env={**os.environ, "ASAN_OPTIONS": "detect_leaks=1:halt_on_error=1",
                 "UBSAN_OPTIONS": "halt_on_error=1"})
        if result.returncode == 77:
            self.skipTest(result.stdout.strip())
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn("PASS " + mode, result.stdout)

    def test_scalar_reference_lengths_and_guards(self): self.run_case("scalar")
    def test_vector_lengths_offsets_tails_and_guards(self): self.run_case("lengths")
    def test_shuffle_permute_hadd_and_sum(self): self.run_case("lanes")
    def test_blend_uses_sign_bit(self): self.run_case("blend")
    def test_fma_retains_single_rounding(self): self.run_case("fma")
    def test_conversion_rounding_contract(self): self.run_case("conversion")
    def test_nan_and_signed_zero_operand_order(self): self.run_case("nan")

    def test_address_and_undefined_behavior_sanitizers(self):
        binary = self.build("sanitized", "-O1", "-g", "-fsanitize=address,undefined",
                            "-fno-omit-frame-pointer", "-fno-pie", "-no-pie")
        self.run_case("scalar", binary)
        self.run_case("all", binary)

    def assert_mutant_rejected(self, macro, mode, diagnostic):
        self.run_case(mode)  # skip explicitly if the host lacks the vector ISA
        binary = self.build(macro.lower(), "-D" + macro)
        result = subprocess.run([str(binary), mode], capture_output=True, text=True, timeout=30)
        self.assertEqual(result.returncode, 1, result.stdout + result.stderr)
        self.assertIn(diagnostic, result.stderr)

    def test_missing_tail_mutant_is_detected(self):
        self.assert_mutant_rejected("COURSE_MUTANT_MISSING_TAIL", "lengths", "tail/value mismatch")

    def test_half_reverse_mutant_is_detected(self):
        self.assert_mutant_rejected("COURSE_MUTANT_FULL_REVERSE", "lanes", "full reverse")

    def test_real_aarch64_compiler_output(self):
        clang = shutil.which("clang++")
        if not clang:
            self.skipTest("Clang cross compiler unavailable; ARM code not executed")
        assembly = self.directory / "neon.s"
        result = subprocess.run([clang, "--target=aarch64-none-elf", "-ffreestanding",
            "-O2", "-S", str(HERE / "neon_contracts.cpp"), "-o", str(assembly)],
            capture_output=True, text=True, timeout=30)
        self.assertEqual(result.returncode, 0, result.stderr)
        text = assembly.read_text()
        # Two FADDP operations for a four-float reduction; FCVTZS truncates toward zero.
        self.assertEqual(text.count("\tfaddp\t"), 2, text)
        self.assertIn("\tfcvtzs\t", text)

if __name__ == "__main__":
    unittest.main(verbosity=2)
