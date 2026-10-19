import os
from pathlib import Path
import resource
import shutil
import subprocess
import sys
import tempfile
import unittest

SOURCE = Path(__file__).resolve().parent.parent / "Lectures/demos/rt-demo/rt_demo.c"


@unittest.skipUnless(sys.platform.startswith("linux") and shutil.which("gcc"),
                     "Linux and GCC required for the pthread API demo")
class RuntimeDemoTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.directory = tempfile.TemporaryDirectory(prefix="course-rt-demo-")
        cls.addClassCleanup(cls.directory.cleanup)
        cls.binary = Path(cls.directory.name) / "rt_demo"
        build = subprocess.run(["gcc", "-std=c17", "-Wall", "-Wextra", "-Werror", "-O2", "-pthread",
                                str(SOURCE), "-o", str(cls.binary)], capture_output=True, text=True, timeout=30)
        if build.returncode: raise AssertionError(build.stderr)

    def run_demo(self, *args, **kwargs):
        return subprocess.run([str(self.binary), *args], capture_output=True, text=True,
                              timeout=5, **kwargs)

    def test_default_and_obtained_affinity(self):
        result = self.run_demo()
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("Final counter = 10000", result.stdout)
        self.assertIn("memory_locked=0", result.stdout)
        allowed = sorted(x for x in os.sched_getaffinity(0) if x < 1024)
        if not allowed: self.skipTest("allowed CPU outside this demo's fixed cpu_set_t")
        result = self.run_demo("--cpu", str(allowed[0]))
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("obtained_count=1", result.stdout)

    def test_rejects_partial_malformed_and_unknown_options(self):
        for text in ("", "1junk", "2,", ",2", "3-1", "-1", "1024", "1,,2", "1-2x"):
            with self.subTest(text=text):
                result = self.run_demo("--cpu", text)
                self.assertNotEqual(result.returncode, 0)
                self.assertIn("Invalid CPU list", result.stderr)
        self.assertNotEqual(self.run_demo("--unknown").returncode, 0)
        self.assertNotEqual(self.run_demo("unexpected-positional").returncode, 0)

    @unittest.skipIf(os.geteuid() == 0, "root may bypass the deliberately restricted memory-lock limit")
    def test_failed_memory_lock_is_not_reported_as_success(self):
        def restrict_child():
            resource.setrlimit(resource.RLIMIT_MEMLOCK, (0, 0))
        result = self.run_demo("--lock-memory", preexec_fn=restrict_child)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("mlockall", result.stderr)
        self.assertNotIn("Final counter", result.stdout)


if __name__ == "__main__":
    unittest.main()
