"""Compile the shipped C++ examples, then check ownership and argument boundaries."""
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest

SOURCES = Path(__file__).resolve().parent.parent


class ParallelExampleTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        compiler = shutil.which(os.environ.get("CXX", "g++"))
        if compiler is None:
            raise unittest.SkipTest("C++ compiler missing; install g++ and oneTBB development files")
        cls.directory = tempfile.TemporaryDirectory(prefix="course-parallel-")
        cls.addClassCleanup(cls.directory.cleanup)
        cls.work = Path(cls.directory.name)
        probe = cls.work / "probe.cpp"
        probe.write_text("#include <oneapi/tbb/global_control.h>\n#include <omp.h>\nint main() { return 0; }\n")
        flags = ["-O2", "-std=c++17", "-fopenmp", "-pthread", "-Wall", "-Wextra", "-Werror"]
        result = subprocess.run([compiler, *flags, str(probe), "-ltbb", "-o", str(cls.work / "probe")],
                                text=True, capture_output=True, timeout=30)
        if result.returncode:
            raise unittest.SkipTest("OpenMP/oneTBB development toolchain unavailable: " + result.stderr[-500:])
        cls.binaries = {}
        for name in ("fib_benchmark", "video_pipeline_template", "concurrency_contracts"):
            binary = cls.work / name
            subprocess.run([compiler, *flags, str(SOURCES / f"{name}.cpp"), "-ltbb", "-o", str(binary)],
                           text=True, capture_output=True, timeout=60, check=True)
            cls.binaries[name] = binary

    def execute(self, name, *arguments):
        return subprocess.run([str(self.binaries[name]), *map(str, arguments)],
                              text=True, capture_output=True, timeout=30)

    def test_fibonacci_against_independent_reference(self):
        result = self.execute("fib_benchmark", "--self-test")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("PASS 27", result.stdout)

    def test_fibonacci_small_inputs_and_equal_thread_budget(self):
        for n, expected in ((0, 0), (1, 1), (2, 1), (10, 55)):
            with self.subTest(n=n):
                result = self.execute("fib_benchmark", n, 1, 2, 1)
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertIn(f"fib({n})={expected} ", result.stdout)
                self.assertIn("observed_openmp_team=1", result.stdout)

    def test_fibonacci_rejects_invalid_arguments(self):
        for arguments in ((-1,), (43,), ("2x",), (5, 0), (5, 2, 1), (5, 2, 20, 0)):
            with self.subTest(arguments=arguments):
                self.assertNotEqual(self.execute("fib_benchmark", *arguments).returncode, 0)

    def test_pipeline_keyed_join_and_limits(self):
        result = self.execute("video_pipeline_template", "--self-test")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("deliberate FIFO-join mismatch", result.stdout)
        self.assertIn("PASS frames=100 delivered=100", result.stdout)

    def test_pipeline_makes_progress_with_one_worker(self):
        result = self.execute("video_pipeline_template", 31, 4, 1)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("frames=31 delivered=31", result.stdout)

    def test_pipeline_rejects_invalid_arguments(self):
        for arguments in ((-1,), ("3x",), (5, 0), (5, 1, 0)):
            with self.subTest(arguments=arguments):
                self.assertNotEqual(self.execute("video_pipeline_template", *arguments).returncode, 0)

    def test_real_concurrency_contracts_and_isolation_counterexample(self):
        result = self.execute("concurrency_contracts", "--self-test")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("PASS all 6 concurrency contracts", result.stdout)
        self.assertIn("concurrent regions=2", result.stdout)

    def test_concurrency_example_rejects_unknown_cli(self):
        self.assertEqual(self.execute("concurrency_contracts", "--unknown").returncode, 2)


class OpenMPPatternTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        compiler = shutil.which(os.environ.get("CXX", "g++"))
        if compiler is None:
            raise unittest.SkipTest("C++ compiler missing for OpenMP patterns")
        cls.directory = tempfile.TemporaryDirectory(prefix="course-openmp-patterns-")
        cls.addClassCleanup(cls.directory.cleanup)
        work = Path(cls.directory.name)
        probe = work / "probe.cpp"
        probe.write_text("#include <omp.h>\nint main() { return omp_get_max_threads() < 1; }\n")
        flags = ["-O2", "-std=c++17", "-fopenmp", "-Wall", "-Wextra", "-Werror"]
        result = subprocess.run([compiler, *flags, str(probe), "-o", str(work / "probe")],
                                text=True, capture_output=True, timeout=30)
        if result.returncode:
            raise unittest.SkipTest("OpenMP development toolchain unavailable: " + result.stderr[-500:])
        cls.binary = work / "openmp_patterns"
        subprocess.run([compiler, *flags, str(SOURCES / "openmp_patterns.cpp"), "-o", str(cls.binary)],
                       text=True, capture_output=True, timeout=60, check=True)

    def run_patterns(self, limit):
        # Pin the environment: a user's process-wide limit must not silently
        # turn the multi-thread regression into a one-thread test.
        environment = dict(os.environ, OMP_THREAD_LIMIT=str(limit), OMP_DYNAMIC="FALSE")
        return subprocess.run([str(self.binary)], env=environment,
                              text=True, capture_output=True, timeout=30)

    def test_private_lifetimes_reductions_barriers_and_task_counts(self):
        result = self.run_patterns(4)
        self.assertEqual(result.returncode, 0, result.stderr)
        for team in (1, 2, 4):
            self.assertIn(f"PASS requested={team} observed_team={team}:", result.stdout)
        self.assertIn("team=2 parallel=2 serial=4", result.stdout)

    def test_runtime_thread_limit_is_observed(self):
        result = self.run_patterns(1)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("PASS requested=4 observed_team=1:", result.stdout)
        self.assertIn("two-thread counterexample not exercised", result.stdout)


if __name__ == "__main__":
    unittest.main()
