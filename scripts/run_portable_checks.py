#!/usr/bin/env python3
"""Run the curriculum's CPU lab suites without a GPU or network.

Each scripts/, Phase */.../labs/, or bundled runtime labs/ directory containing test_*.py is a separate
unittest suite. Separate processes prevent identically named lesson helpers from
shadowing each other. New, nonignored files are included before they are committed.
Git checkouts respect .gitignore; extracted archives use filesystem discovery.
Generated trees and hidden directories are excluded in both cases. Run from any
directory; --list prints the discovered suites without running them.
Most exercises need only Python; optional compiler/runtime prerequisites are
declared by their suites. A wholly skipped suite is reported separately from a
validation pass, and a run with no passing suite exits unsuccessfully.
"""
from __future__ import annotations

import argparse
import json
import os
import signal
from pathlib import Path
import subprocess
import sys
import tempfile
import time


ROOT = Path(__file__).resolve().parents[1]
EXCLUDED_DIRECTORIES = {
    "docs", "_site", "site", "build", "dist", "node_modules", "__pycache__",
    "venv", "env", "env_isaaclab",
}


def source_path(path: Path) -> bool:
    """Exclude generated trees and private/editor environments from discovery."""
    return not any(part.startswith(".") or part in EXCLUDED_DIRECTORIES
                   for part in path.parts)


def candidate_files(root: Path) -> list[Path]:
    # Archives have no Git metadata; Git itself is optional for the CPU route.
    # A worktree's .git is a file, so test existence rather than is_dir().
    if (root / ".git").exists():
        result = subprocess.run(
            ["git", "-C", str(root), "ls-files", "-z", "--cached", "--others",
             "--exclude-standard", "--", "*.py"],
            capture_output=True, text=True, check=True,
        )
        return [Path(name) for name in result.stdout.split("\0") if name]
    files = []
    for directory, dirs, names in os.walk(root):
        dirs[:] = [name for name in dirs
                   if not name.startswith(".") and name not in EXCLUDED_DIRECTORIES]
        files.extend((Path(directory) / name).relative_to(root)
                     for name in names if name.startswith("test_") and name.endswith(".py"))
    return files


def discover_suites(root: Path) -> list[Path]:
    suites = set()
    for path in candidate_files(root):
        if not source_path(path) or not path.name.startswith("test_") or path.suffix != ".py":
            continue
        is_tool_suite = path.parent == Path("scripts")
        is_course_tree = path.parts[0].startswith("Phase ") or path.parts[:2] == ("Projects", "jetson-llm-runtime")
        is_lab_suite = is_course_tree and path.parent.name.lower() == "labs"
        if not (is_tool_suite or is_lab_suite):
            continue
        if (root / path).is_file():
            suites.add(path.parent)
    return sorted(suites)


SUITE_PROGRAM = """\
import json
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(sys.argv[1]).resolve()))
suite = unittest.TestSuite(unittest.defaultTestLoader.loadTestsFromName(name)
                           for name in json.loads(sys.argv[2]))
discovered = suite.countTestCases()
if discovered == 0:
    print('No tests discovered: an empty suite is not validation.', file=sys.stderr)
    raise SystemExit(5)
class RecordingResult(unittest.TextTestResult):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.started_methods = {}  # retain objects so CPython cannot reuse their ids

    def startTest(self, test):
        self.started_methods[id(test)] = test
        super().startTest(test)

result = unittest.TextTestRunner(resultclass=RecordingResult).run(suite)
skipped_methods = sum(id(test) in result.started_methods for test, _ in result.skipped)
skipped_subtests = sum(isinstance(test, unittest.TestCase)
                       and id(test) not in result.started_methods for test, _ in result.skipped)
Path(sys.argv[3]).write_text(json.dumps({
    'tests_discovered': discovered,
    'tests_run': result.testsRun,
    'tests_executed': result.testsRun - skipped_methods,
    'tests_skipped': skipped_methods,
    'skipped_groups': len(result.skipped) - skipped_methods - skipped_subtests,
    'skipped_subtests': skipped_subtests,
    'failures': len(result.failures),
    'errors': len(result.errors),
    'expected_failures': len(result.expectedFailures),
    'unexpected_successes': len(result.unexpectedSuccesses),
    'stopped_early': bool(result.shouldStop),
    'skip_reasons': [{'test': str(test), 'reason': reason} for test, reason in result.skipped],
}), encoding='utf-8')
if result.wasSuccessful():
    if result.shouldStop:
        print('Suite stopped early: remaining checks were not validated.', file=sys.stderr)
        raise SystemExit(6)
    if result.testsRun == skipped_methods:
        # A skipped setUpClass is an _ErrorHolder and adds no executed method.
        if result.skipped:
            raise SystemExit(77)
        print('No test methods executed: a custom suite is not validation.', file=sys.stderr)
        raise SystemExit(5)
raise SystemExit(0 if result.wasSuccessful() else 1)
"""


def run_suite(root: Path, suite: Path, timeout: float, *, quiet: bool = False) -> dict:
    # Load only selected files, not ignored or generated tests that happen to
    # share a directory with a real course suite.
    modules = sorted({path.stem for path in candidate_files(root)
                      if path.parent == suite and source_path(path)
                      and path.name.startswith("test_") and path.suffix == ".py"
                      and (root / path).is_file()})
    start = time.monotonic()
    metrics = {}
    with tempfile.TemporaryDirectory(prefix="course-test-result-") as directory:
        result_file = Path(directory) / "result.json"
        command = [sys.executable, "-c", SUITE_PROGRAM, str(suite), json.dumps(modules), str(result_file)]
        process = subprocess.Popen(command, cwd=root, start_new_session=os.name == "posix",
                                   stdout=subprocess.DEVNULL if quiet else None,
                                   stderr=subprocess.DEVNULL if quiet else None)
        try:
            returncode = process.wait(timeout=timeout)
            passed = returncode == 0
            if returncode == 77:
                status, reason = "skipped", "all tests skipped; no checks executed"
            else:
                status = "passed" if passed else "failed"
                reason = "passed" if passed else f"exit status {returncode}"
        except subprocess.TimeoutExpired:
            # Native compiler/runtime tests can create grandchildren. Killing only
            # the Python parent leaves those workers running after the timeout.
            if os.name == "posix":
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
            else:
                process.kill()
            process.wait()
            passed, status, reason = False, "failed", "timeout"
            if not quiet:
                print(f"Suite exceeded {timeout:g} seconds.", file=sys.stderr)
        if result_file.exists():
            try:
                metrics = json.loads(result_file.read_text(encoding="utf-8"))
                required = ("tests_discovered", "tests_run", "tests_executed", "tests_skipped",
                            "skipped_groups", "skipped_subtests", "failures", "errors")
                if any(type(metrics.get(key)) is not int or metrics[key] < 0 for key in required):
                    raise ValueError("invalid test-result counts")
            except (ValueError, AttributeError, OSError) as error:
                metrics = {}
                passed, status, reason = False, "failed", f"invalid test-result record: {error}"
        elif status in {"passed", "skipped"}:
            # sys.exit(0) during import is not a successful unittest execution.
            passed, status, reason = False, "failed", "child exited without a test-result record"

    return {"suite": str(suite), "status": status, "passed": passed, "result": reason,
            "elapsed_seconds": round(time.monotonic() - start, 3), **metrics}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--list", action="store_true", help="print suites and exit")
    parser.add_argument("--timeout", type=float, default=120,
                        help="maximum seconds per suite (default: 120)")
    parser.add_argument("--json", type=Path, help="save suite results as JSON")
    args = parser.parse_args()
    if not 0 < args.timeout < float("inf"):
        parser.error("--timeout must be finite and greater than zero")
    try:
        suites = discover_suites(ROOT)
    except (OSError, subprocess.CalledProcessError) as error:
        parser.error(f"test discovery failed: {error}")
    if not suites:
        parser.error("no portable test suites found")
    if args.list:
        print("\n".join(map(str, suites)))
        return 0
    results = []
    for suite in suites:
        print(f"\n=== {suite} ===", flush=True)
        results.append(run_suite(ROOT, suite, args.timeout))
    passed = sum(result["status"] == "passed" for result in results)
    skipped = sum(result["status"] == "skipped" for result in results)
    failed = sum(result["status"] == "failed" for result in results)
    print(f"\n{passed}/{len(results)} portable suites passed; {skipped} skipped; {failed} failed.")
    executed = sum(result.get("tests_executed", 0) for result in results)
    skipped_tests = sum(result.get("tests_skipped", 0) for result in results)
    skipped_groups = sum(result.get("skipped_groups", 0) for result in results)
    skipped_subtests = sum(result.get("skipped_subtests", 0) for result in results)
    print(f"{executed} test methods executed; {skipped_tests} test methods, "
          f"{skipped_groups} groups and {skipped_subtests} subtests skipped.")
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    return int(failed > 0 or passed == 0)


if __name__ == "__main__":
    raise SystemExit(main())
