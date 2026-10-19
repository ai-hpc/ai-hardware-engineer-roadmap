"""Compile and execute the OpenCL kernel on an installed CPU/GPU implementation."""
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest


class OpenCLExampleTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        compiler = shutil.which(os.environ.get("CXX", "g++"))
        if compiler is None:
            raise unittest.SkipTest("C++ compiler missing")
        cls.directory = tempfile.TemporaryDirectory(prefix="course-opencl-")
        cls.addClassCleanup(cls.directory.cleanup)
        work = Path(cls.directory.name)
        probe = work / "probe.cpp"
        probe.write_text("#define CL_TARGET_OPENCL_VERSION 120\n#include <CL/cl.h>\nint main() { cl_uint n; return clGetPlatformIDs(0, nullptr, &n); }\n")
        probe_build = subprocess.run([compiler, str(probe), "-lOpenCL", "-o", str(work / "probe")],
                                     text=True, capture_output=True, timeout=30)
        if probe_build.returncode:
            raise unittest.SkipTest("OpenCL headers/loader development files unavailable: " + probe_build.stderr[-300:])
        cls.binary = work / "vector_add_opencl"
        subprocess.run([compiler, "-O2", "-std=c++17", "-Wall", "-Wextra", "-Werror",
                        str(Path(__file__).with_name("vector_add_opencl.cpp")), "-lOpenCL", "-o", str(cls.binary)],
                       text=True, capture_output=True, timeout=30, check=True)
        result = subprocess.run([str(cls.binary), "1"], text=True, capture_output=True, timeout=60)
        if result.returncode and any(message in result.stderr for message in
                ("OpenCL status -1001", "no OpenCL platform", "no OpenCL device")):
            raise unittest.SkipTest("No OpenCL device implementation installed; a loader alone is insufficient")
        if result.returncode:
            raise RuntimeError("OpenCL smoke test failed:\n" + result.stdout + result.stderr)

    def run_example(self, *arguments):
        return subprocess.run([str(self.binary), *map(str, arguments)],
                              text=True, capture_output=True, timeout=60)

    def test_actual_kernel_with_empty_short_and_uneven_lengths(self):
        result = self.run_example("--self-test")
        self.assertEqual(result.returncode, 0, result.stderr)
        for n in (0, 1, 7, 63, 64, 65, 255, 256, 257, 1025):
            self.assertIn(f"PASS n={n} ", result.stdout)
        self.assertIn("kernel_event_ns=", result.stdout)

    def test_invalid_kernel_surfaces_build_log(self):
        result = self.run_example("--inject-build-error")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("BUILD_LOG:", result.stderr)
        self.assertIn("build program failed", result.stderr)

    def test_rejects_invalid_lengths_before_launch(self):
        for argument in ("-1", "1000001", "4x", "nan"):
            with self.subTest(argument=argument):
                result = self.run_example(argument)
                self.assertNotEqual(result.returncode, 0)
                self.assertIn("length must be", result.stderr)


if __name__ == "__main__":
    unittest.main()
