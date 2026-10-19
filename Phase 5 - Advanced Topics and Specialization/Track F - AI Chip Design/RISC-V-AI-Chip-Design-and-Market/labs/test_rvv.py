"""Real RISC-V binaries in QEMU user mode; emulator timing is not chip timing."""
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest

HERE = Path(__file__).resolve().parent


class RvvExecutionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.clang = shutil.which(os.environ.get("COURSE_RVV_CLANG", "clang"))
        cls.linker = shutil.which(os.environ.get("COURSE_RVV_LLD", "ld.lld")) or shutil.which("ld.lld-18")
        cls.qemu = shutil.which(os.environ.get("COURSE_RVV_QEMU", "qemu-riscv64"))
        if not all((cls.clang, cls.linker, cls.qemu)):
            raise unittest.SkipTest("RVV execution needs Clang, LLD and qemu-riscv64; see lab setup")
        cls.temp = tempfile.TemporaryDirectory(prefix="course-rvv-")
        cls.addClassCleanup(cls.temp.cleanup)
        cls.directory = Path(cls.temp.name)
        cls.binary = cls.build("rvv")

    @classmethod
    def build(cls, name, *extra):
        binary = cls.directory / name
        command = [cls.clang, "--target=riscv64-linux-gnu", "-march=rv64gcv", "-mabi=lp64d",
                   "-O2", "-ffreestanding", "-fno-builtin", "-fno-vectorize", "-fno-slp-vectorize",
                   "-fno-stack-protector", "-nostdlib", "-static", "-Wl,--no-relax",
                   "-Wl,-e,_start", "-fuse-ld=" + cls.linker, *extra,
                   str(HERE / "rvv_contract.c"), "-o", str(binary)]
        result = subprocess.run(command, capture_output=True, text=True, timeout=60)
        if result.returncode:
            raise AssertionError(result.stdout + result.stderr)
        return binary

    @classmethod
    def execute(cls, binary, width):
        return subprocess.run([cls.qemu, "-cpu", f"rv64,v=true,vlen={width},elen=64,vext_spec=v1.0",
                               str(binary)], capture_output=True, text=True, timeout=30)

    def test_same_binary_runs_at_three_vector_lengths(self):
        for width in (128, 256, 512):
            with self.subTest(vlen=width):
                result = self.execute(self.binary, width)
                self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
                self.assertIn("PASS: 784", result.stdout)

    def test_fixed_eight_lane_mutant_only_passes_matching_width(self):
        binary = self.build("fixed-width", "-DCOURSE_MUTANT_FIXED_WIDTH")
        for width, expected_pass in ((128, False), (256, True), (512, False)):
            with self.subTest(vlen=width):
                result = self.execute(binary, width)
                self.assertEqual(result.returncode == 0, expected_pass, result.stdout + result.stderr)
                self.assertIn("PASS:" if expected_pass else "FAIL:", result.stdout)

    def test_full_width_tail_mutant_is_rejected(self):
        binary = self.build("wrong-tail", "-DCOURSE_MUTANT_TAIL")
        for width in (128, 256, 512):
            with self.subTest(vlen=width):
                result = self.execute(binary, width)
                self.assertNotEqual(result.returncode, 0)
                self.assertIn("FAIL:", result.stdout)

    def test_generated_program_contains_real_vector_instructions(self):
        assembly = self.directory / "rvv.s"
        result = subprocess.run([self.clang, "--target=riscv64-linux-gnu", "-march=rv64gcv",
            "-mabi=lp64d", "-O2", "-ffreestanding", "-fno-builtin", "-fno-vectorize",
            "-fno-slp-vectorize", "-S", str(HERE / "rvv_contract.c"), "-o", str(assembly)],
            capture_output=True, text=True, timeout=30)
        self.assertEqual(result.returncode, 0, result.stderr)
        text = assembly.read_text()
        for instruction in ("vsetvli", "vle32.v", "vadd.vv", "vse32.v"):
            self.assertIn(instruction, text)


if __name__ == "__main__":
    unittest.main(verbosity=2)
