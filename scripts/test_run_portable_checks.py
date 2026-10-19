"""Exercise archive discovery and actual child-process failure behavior."""
import subprocess
import shutil
import tempfile
import unittest
from pathlib import Path

from run_portable_checks import discover_suites, run_suite


class PortableRunnerTests(unittest.TestCase):
    def make_file(self, root, relative, content=""):
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
        return path

    def test_archive_discovery_prunes_generated_and_private_trees(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for relative in (
                "scripts/test_tool.py", "Phase 1 - Basics/Labs/test_math.py",
                "Phase 2 - Systems/Module/labs/test_contract.py",
                "docs/Phase 1 - Basics/Labs/test_duplicate.py",
                "_site/Phase 1 - Basics/Labs/test_duplicate.py",
                ".codex/Phase 1 - Basics/Labs/test_internal.py",
                "Phase 1 - Basics/.venv/labs/test_dependency.py",
                "Phase 1 - Basics/build/labs/test_generated.py",
                "Phase 1 - Basics/labs/.hidden/test_private.py",
                "Other/labs/test_unrelated.py",
                "Projects/unrelated/labs/test_unrelated.py",
                "Projects/jetson-llm-runtime/labs/test_runtime.py",
            ):
                self.make_file(root, relative)
            self.assertEqual(discover_suites(root), [
                Path("Phase 1 - Basics/Labs"),
                Path("Phase 2 - Systems/Module/labs"),
                Path("Projects/jetson-llm-runtime/labs"), Path("scripts")])

    @unittest.skipUnless(shutil.which("git"), "Git-specific discovery needs Git")
    def test_checkout_respects_gitignore_and_includes_new_tests(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            subprocess.run(["git", "init", "-q", str(root)], check=True)
            self.make_file(root, ".gitignore", "Phase 1 - Basics/Ignored/\ntest_ignored.py\n")
            self.make_file(root, "Phase 1 - Basics/Labs/test_new.py",
                           "import unittest\nclass New(unittest.TestCase):\n"
                           "    def test_new(self): self.assertTrue(True)\n")
            self.make_file(root, "Phase 1 - Basics/Labs/test_ignored.py",
                           "raise RuntimeError('ignored file must not run')\n")
            self.make_file(root, "Phase 1 - Basics/Ignored/labs/test_ignore.py")
            self.assertEqual(discover_suites(root), [Path("Phase 1 - Basics/Labs")])
            self.assertTrue(run_suite(root, Path("Phase 1 - Basics/Labs"), 5,
                                      quiet=True)["passed"])

    def test_empty_suite_fails_instead_of_reporting_validation(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.make_file(root, "labs/test_empty.py", "# No test cases yet.\n")
            result = run_suite(root, Path("labs"), 5, quiet=True)
            self.assertFalse(result["passed"])
            self.assertEqual(result["result"], "exit status 5")

    def test_custom_suite_cannot_pass_without_executing_methods(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.make_file(root, "labs/test_no_execution.py",
                           "import unittest\n"
                           "class EmptyRun(unittest.TestSuite):\n"
                           "    def countTestCases(self): return 1\n"
                           "    def run(self, result, debug=False): return result\n"
                           "def load_tests(loader, tests, pattern): return EmptyRun()\n")
            result = run_suite(root, Path("labs"), 5, quiet=True)
            self.assertFalse(result["passed"])
            self.assertEqual(result["status"], "failed")
            self.assertEqual(result["tests_discovered"], 1)
            self.assertEqual(result["tests_executed"], 0)
            self.assertEqual(result["result"], "exit status 5")

    def test_early_stop_cannot_hide_a_required_failing_check(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.make_file(root, "labs/test_stop.py",
                           "import unittest\nclass Early(unittest.TestCase):\n"
                           "    def test_1_stop(self): self._outcome.result.stop()\n"
                           "    def test_2_required(self): self.fail('required check')\n")
            result = run_suite(root, Path("labs"), 5, quiet=True)
            self.assertFalse(result["passed"])
            self.assertEqual(result["status"], "failed")
            self.assertEqual(result["tests_discovered"], 2)
            self.assertEqual(result["tests_executed"], 1)
            self.assertTrue(result["stopped_early"])
            self.assertEqual(result["result"], "exit status 6")

    def test_setup_errors_and_failed_subtests_are_not_passes(self):
        bodies = {
            "setup": ("import unittest\nclass Setup(unittest.TestCase):\n"
                      "    @classmethod\n    def setUpClass(cls):\n"
                      "        raise RuntimeError('fixture failed')\n"
                      "    def test_required(self): pass\n"),
            "subtest": ("import unittest\nclass Sub(unittest.TestCase):\n"
                        "    def test_cases(self):\n"
                        "        with self.subTest(i=0): self.assertEqual(1,2)\n"),
        }
        for name, content in bodies.items():
            with self.subTest(name=name), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                self.make_file(root, "labs/test_failure.py", content)
                result = run_suite(root, Path("labs"), 5, quiet=True)
                self.assertFalse(result["passed"])
                self.assertEqual(result["status"], "failed")
                self.assertEqual(result["errors"] + result["failures"], 1)

    def test_failed_assertion_propagates(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.make_file(root, "labs/test_failure.py",
                           "import unittest\nclass Failure(unittest.TestCase):\n"
                           "    def test_wrong(self): self.assertEqual(1, 2)\n")
            result = run_suite(root, Path("labs"), 5, quiet=True)
            self.assertFalse(result["passed"])
            self.assertEqual(result["result"], "exit status 1")
            self.assertEqual(result["tests_executed"], 1)
            self.assertEqual(result["failures"], 1)

    def test_hung_suite_is_stopped(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.make_file(root, "labs/test_hang.py",
                           "import time, unittest\nclass Hang(unittest.TestCase):\n"
                           "    def test_slow(self): time.sleep(30)\n")
            result = run_suite(root, Path("labs"), .1, quiet=True)
            self.assertFalse(result["passed"])
            self.assertEqual(result["result"], "timeout")

    def test_passing_suite_runs_from_archive(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.make_file(root, "labs/test_sum.py",
                           "import unittest\nclass Sum(unittest.TestCase):\n"
                           "    def test_sum(self): self.assertEqual(sum([1, 2]), 3)\n")
            self.assertTrue(run_suite(root, Path("labs"), 5, quiet=True)["passed"])

    def test_all_skipped_suite_does_not_claim_a_validation_pass(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.make_file(root, "labs/test_optional.py",
                           "import unittest\nclass Optional(unittest.TestCase):\n"
                           "    @classmethod\n    def setUpClass(cls):\n"
                           "        raise unittest.SkipTest('tool unavailable')\n"
                           "    def test_optional(self): pass\n")
            result = run_suite(root, Path("labs"), 5, quiet=True)
            self.assertFalse(result["passed"])
            self.assertEqual(result["status"], "skipped")
            self.assertEqual(result["tests_executed"], 0)
            self.assertEqual(result["skipped_groups"], 1)
            self.assertEqual(result["tests_discovered"], 1)

    def test_one_skipped_class_does_not_hide_a_real_passing_test(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.make_file(root, "labs/test_mixed.py",
                           "import unittest\nclass Optional(unittest.TestCase):\n"
                           "    @classmethod\n    def setUpClass(cls):\n"
                           "        raise unittest.SkipTest('tool unavailable')\n"
                           "    def test_optional(self): pass\n"
                           "class Executed(unittest.TestCase):\n"
                           "    def test_actual(self): self.assertEqual(2 + 2, 4)\n")
            result = run_suite(root, Path("labs"), 5, quiet=True)
            self.assertTrue(result["passed"])
            self.assertEqual(result["tests_executed"], 1)
            self.assertEqual(result["skipped_groups"], 1)
            self.assertEqual(result["tests_discovered"], 2)

    def test_skipped_method_is_not_counted_as_execution(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.make_file(root, "labs/test_mixed.py",
                           "import unittest\nclass Mixed(unittest.TestCase):\n"
                           "    def test_actual(self): self.assertEqual(2 + 2, 4)\n"
                           "    @unittest.skip('optional runtime unavailable')\n"
                           "    def test_optional(self): pass\n")
            result = run_suite(root, Path("labs"), 5, quiet=True)
            self.assertTrue(result["passed"])
            self.assertEqual(result["tests_run"], 2)
            self.assertEqual(result["tests_executed"], 1)
            self.assertEqual(result["tests_skipped"], 1)
            self.assertEqual(result["skipped_groups"], 0)
            self.assertEqual(result["skip_reasons"][0]["reason"], "optional runtime unavailable")

    def test_successful_exit_during_import_is_not_a_test_pass(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.make_file(root, "labs/test_exit.py", "raise SystemExit(0)\n")
            result = run_suite(root, Path("labs"), 5, quiet=True)
            self.assertFalse(result["passed"])
            self.assertEqual(result["status"], "failed")
            self.assertIn("without a test-result record", result["result"])

    def test_skipped_subtests_do_not_subtract_executed_methods(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.make_file(root, "labs/test_subtest.py",
                           "import unittest\nclass Partial(unittest.TestCase):\n"
                           "    def test_cases(self):\n"
                           "        self.assertEqual(2+2,4)\n"
                           "        for i in range(3):\n"
                           "            with self.subTest(i=i): self.skipTest('optional variant')\n")
            result = run_suite(root, Path("labs"), 5, quiet=True)
            self.assertTrue(result["passed"])
            self.assertEqual(result["tests_executed"], 1)
            self.assertEqual(result["tests_skipped"], 0)
            self.assertEqual(result["skipped_groups"], 0)
            self.assertEqual(result["skipped_subtests"], 3)

    @unittest.skipUnless(__import__('os').name == 'posix', 'POSIX process-group cleanup')
    def test_timeout_stops_native_child_processes_too(self):
        import time
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            child = "import time,pathlib;time.sleep(.7);pathlib.Path('orphan.txt').write_text('alive')"
            self.make_file(root, "labs/test_child.py",
                           "import subprocess,sys,time,unittest\n"
                           "class Parent(unittest.TestCase):\n"
                           "    def test_child(self):\n"
                           f"        subprocess.Popen([sys.executable,'-c',{child!r}])\n"
                           "        time.sleep(30)\n")
            result = run_suite(root, Path("labs"), .2, quiet=True)
            self.assertEqual(result["result"], "timeout")
            time.sleep(.8)
            self.assertFalse((root / 'orphan.txt').exists())

    @unittest.skipUnless(shutil.which('git'), 'Git-specific discovery needs Git')
    def test_broken_git_checkout_does_not_fallback_to_ignored_files(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.make_file(root, '.git', 'gitdir: missing-directory\n')
            self.make_file(root, 'Phase 1 - Basics/labs/test_private.py')
            with self.assertRaises(subprocess.CalledProcessError): discover_suites(root)


if __name__ == "__main__":
    unittest.main()
