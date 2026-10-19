# Yocto lab: a small package with an observable contract

Use this application throughout the [sixteen Yocto lessons](../Lecture/README.md). It needs no model download, GPU, or sensor. The package is intentionally small enough to understand before building an operating system around it.

## What the program does

`observation-check CAPTURE_MS COMPLETE_MS LIMIT_MS` accepts nonnegative integer milliseconds from one clock domain. It reports the age and whether it meets an inclusive limit. It returns `0` for fresh, `2` for stale, `64` for malformed input, and `65` when capture time is after completion time. These statuses are an explicit application contract, not operating-system timing guarantees.

The checker does not sample clocks, synchronize devices, validate frames, supervise a motor, or keep state across invocations. The earlier [observation freshness lab](../../../Labs/Observation-Freshness.md) teaches those broader queue/identity contracts. This lesson packages one small, testable component.

## Native checks

**Tested here:** Ubuntu 24.04.4 x86-64, Python 3.12.3, GCC 13.3.0. Run from the repository root:

```bash
python3 -m unittest discover -s 'Phase 2 - Embedded Systems/3. Embedded Linux/Yocto/Labs' -p 'test_*.py' -v
cc -std=c11 -Wall -Wextra -Werror -O2 'Phase 2 - Embedded Systems/3. Embedded Linux/Yocto/Labs/meta-robot-course/recipes-apps/observation-check/files/observation-check.c' -o /tmp/observation-check
/tmp/observation-check 100 125 40
/tmp/observation-check 100 141 40
printf 'exit status: %s\n' "$?"
```

The suite compiles a temporary native executable and runs seven tests. If `cc` is unavailable it reports a skip, which is not a passing native compilation. Expected command output is fresh/age 25, stale/age 41, and exit status 2. The executable rejects malformed numbers, negative values, and values exceeding uint64; it checks ordering before subtracting timestamps to avoid unsigned underflow.

The negative exit codes are intentional. If your shell uses `set -e`, run the test suite for all cases or handle those expected statuses explicitly.

## Supplied layer

```text
meta-robot-course/
  conf/layer.conf
  recipes-apps/observation-check/
    observation-check_1.0.bb
    files/observation-check.c
    files/LICENSE
  recipes-core/images/robot-course-image.bb
```

The layer declares Scarthgap compatibility. The application recipe compiles with the target `${CC}` and installs under `${D}${bindir}`; the image recipe selects its runtime package. Copy this layer into a workspace with **no spaces in its path** before adding it. Follow [Lecture 5](../Lecture/Lecture-05.md) for the pinned baseline and [Lecture 6](../Lecture/Lecture-06.md) for exact layer commands.

## Validation record

The review used upstream Poky tag `yocto-5.0.20`, resolved to commit `64e69ed23703f6358ec431d3f5f1f8483f974cae`, with BitBake 2.8.1. A temporary no-space copy of the supplied layer was added to a `qemux86-64`/`poky` configuration. Missing `chrpath`, `diffstat`, and `lz4c` host tools were supplied in an isolated temporary tools directory; this detail matters when reproducing the environment.

| Check | Actual result |
|---|---|
| Native application suite | 7 tests passed |
| Upstream BitBake parser self-tests | 14 tests passed |
| `bitbake -p` with supplied layer | 922 recipes parsed, 1880 targets, 47 skipped, 0 errors |
| Recipe discovery | `observation-check` version 1.0 visible |
| Effective image package list | Included `observation-check` |
| `bitbake -c unpack observation-check` | Recipe QA, fetch, and unpack tasks succeeded |
| Target compile/package, full image, SDK, QEMU/board boot | **Not run** |

The unpack exercise also fetched the configured uninative shim; “no image build” does not mean “no network access.” Parsing and source tasks validate a limited part of the metadata path. They do not certify target code generation, package QA, image composition, or boot.

The [machine-readable record](validation.json) names the versions, completed task boundary, and SHA-256 hashes of the supplied layer inputs. Record your own results after modifying them. The full target acceptance cases are in the [capstone](../Lecture/Lecture-14.md).

**Stopping condition:** once native behavior and the metadata path are understood, choose whether you have the resources for the image route. You can complete the native/metadata milestone without starting a multi-hour image build.
