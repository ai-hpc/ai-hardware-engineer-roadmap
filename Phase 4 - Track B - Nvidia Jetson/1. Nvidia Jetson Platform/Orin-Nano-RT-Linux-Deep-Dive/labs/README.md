# Lab: measure a periodic loop without changing the kernel

A controller that runs for 200 microseconds and then sleeps for 2 milliseconds has a period of at least 2.2 milliseconds. Repeating that pattern makes it fall behind a 2 millisecond schedule even when every sleep works perfectly. This lab makes the difference between a **period**, a **wake-up delay**, and a **completion deadline** visible.

Allow 30–45 minutes. Use Linux, a C compiler, and Python 3.10 or newer. No Jetson, GPU, administrator privileges, real-time scheduler, or extra Python packages are required. The native program changes no system settings. It allocates a finite trace before the loop and prints after the loop.

## 1. Predict the ideal traces

Number jobs from one. Let the start time be zero, period P=2 ms, and work C=0.2 ms.

| Job | Nominal release | Absolute-timer wake, ideal | Relative-timer wake, ideal |
|---|---:|---:|---:|
| 1 | 2 ms | 2 ms | 2 ms |
| 2 | 4 ms | 4 ms | 4.2 ms |
| 3 | 6 ms | 6 ms | 6.4 ms |
| 100 | 200 ms | 200 ms | 219.8 ms |

Absolute timers target `start + job*P`. The deliberately naive relative implementation sleeps P after the previous work completes. Its ideal drift at job n is `(n-1)*C`. Actual OS scheduling adds delay to both. An absolute target prevents this particular accumulation of work time; it does not guarantee the thread runs immediately when the timer expires.

Both implementations are judged against the **same nominal periodic grid**. The relative loop's reported lateness therefore includes accumulated schedule drift. It is not a measurement of the operating system's per-sleep wake-up latency.

## 2. Build and run both modes

From the repository root:

```bash
cd 'Phase 4 - Track B - Nvidia Jetson/1. Nvidia Jetson Platform/Orin-Nano-RT-Linux-Deep-Dive/labs'
cc -std=c11 -O2 -Wall -Wextra -Werror periodic.c -o /tmp/course-periodic
/tmp/course-periodic absolute 100 2000 200 > /tmp/course-absolute.json
/tmp/course-periodic relative 100 2000 200 > /tmp/course-relative.json
python3 analyze.py /tmp/course-absolute.json
python3 analyze.py /tmp/course-relative.json
python3 -m unittest discover -s . -p test_timing.py -v
```

Arguments are mode, iteration count, period in microseconds, and synthetic work in microseconds. Work is a bounded wall-clock busy loop, so a preemption can extend it; this is not an instruction-count workload. The program rejects invalid values and configurations whose nominal duration exceeds ten seconds. External scheduling delays can still make the actual run longer.

The [C program](periodic.c) uses `CLOCK_MONOTONIC`. It retries interrupted absolute sleeps with the original target and interrupted relative sleeps with the returned remaining duration. `clock_nanosleep` returns an error number directly; checking only `errno` would be wrong. See the [Linux API reference](https://man7.org/linux/man-pages/man2/clock_nanosleep.2.html).

## 3. Read the measurements

For each job:

```text
wake lateness       = wake - nominal release
completion response = finish - nominal release
deadline            = nominal release + period
deadline miss       = finish > deadline
```

A completion exactly at its deadline is counted as on time. This is an explicit convention, not a universal controller requirement. Change the deadline model if your application has a different relative deadline.

The [analyzer](analyze.py) checks timestamp ordering and release spacing before computing statistics. Its p99 is the nearest-rank sample quantile: sort N values and select rank `ceil(0.99*N)`, counting ranks from one. With only 100 samples, p99 is the 99th value. A small run is good for understanding the mechanism and poor for estimating rare tails.

An actual run on the shared x86-64 Linux host on **2026-10-18**, using ordinary scheduler policy 0 (`SCHED_OTHER`), produced:

| Observation | Absolute | Relative |
|---|---:|---:|
| Samples | 100 | 100 |
| Wake lateness p99 | 697.360 µs | 30,290.388 µs |
| Largest observed completion response | 1,255.304 µs | 30,779.976 µs |
| Completion misses against the 2 ms grid | 0 | 94 |
| Final wake lateness | 94.593 µs | 30,579.935 µs |

These values describe that run, not expected results or a Jetson benchmark. The raw [absolute trace](absolute-example.json), [relative trace](relative-example.json), and [compiler/environment record](validation.json) are included so the table can be recomputed. Zero observed misses does not prove that a later run will have zero misses.

## 4. Change one assumption

1. Set work to zero. The relative loop can still drift because each iteration adds scheduling and loop overhead.
2. Set work to 2500 µs with a 2000 µs period and 20 iterations. The offered work exceeds the period. Absolute timers cannot repair this overload.
3. Increase the period while keeping work fixed. Describe the remaining slack in the same units as the completion response.
4. Repeat several times under normal activity. Save all runs, including the worst one; do not select the most attractive result.

In the absolute implementation, a target already in the past returns immediately. Consecutive overdue jobs may therefore run back to back. That behavior is intentional for this experiment. A robot needs an explicit policy: skip obsolete work, process the newest observation, degrade functionality, or enter its designed stop state. Executing an accumulated queue of old control actions is often inappropriate.

## 5. Explain what you learned

Submit the two raw traces, their summaries, the compiler version, kernel version, and answers:

- **Why can an absolute loop miss a deadline?** Its thread can be delayed, and work or blocking can exceed the available budget.
- **Does a 1 ns timestamp unit imply 1 ns scheduling accuracy?** No. Representation, clock resolution, measurement overhead, and scheduling accuracy differ.
- **Can these samples establish WCET?** No. They cover finite conditions on one host; they do not bound all permitted executions.
- **Should the test assert “absolute p99 < 100 µs”?** No. That would make correctness depend on an arbitrary host's scheduling. The tests check actual execution, trace completeness, causality, invalid inputs, deadline arithmetic, and quantile conventions.

Continue to the [real-time Linux guide](../Guide.md) to connect this experiment to scheduling, interrupts, locks, and a robot's full sensor-to-actuator path.
