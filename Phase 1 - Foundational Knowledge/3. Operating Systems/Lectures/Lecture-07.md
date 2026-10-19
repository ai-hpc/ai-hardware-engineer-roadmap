# Lecture 7: Real-Time Linux — Mechanisms, Measurements, and Deadlines

## Start with the deadline

A controller receives a sample at time 0 and must issue a command by 2 ms. Suppose its assumed budget is 0.15 ms for wakeup, 0.60 ms CPU computation, 0.40 ms device work, 0.20 ms communication, and 0.25 ms interference. Their sum is 1.60 ms, leaving 0.40 ms margin. These are **example allocations**, not measurements of Linux, Jetson, or a robot. A 0.7 ms queue wait consumes the margin and misses the deadline even if the computation itself remains fast.

Real-time design concerns timely completion under a stated workload and fault model. A hard deadline treats a miss as a requirement violation; a soft deadline permits a defined loss of quality. Neither name specifies a universal number of microseconds. An automotive integrity classification is not a scheduling-latency threshold.

Distinguish three quantities:

- **Wakeup latency:** actual execution time minus intended wakeup time.
- **Execution demand:** CPU time required by the job, with specified inputs and cache state.
- **Response time:** completion minus release, including waiting and device dependencies.

A timer benchmark measures the first. It does not measure the entire control path or establish its worst-case execution time.

## What PREEMPT_RT changes

The useful question is which contexts can prevent a more urgent runnable thread from executing. PREEMPT_RT reduces important non-preemptible regions, but raw locks, low-level interrupt entry, firmware, device stalls, and hardware contention remain. There is no universal “PREEMPT_RT <50 µs” guarantee. Kernel version, architecture, configuration, firmware, drivers, and load all matter.

| Mechanism | Relevant change | Remaining obligation |
|---|---|---|
| `spinlock_t` on PREEMPT_RT | Uses an RT-mutex-based implementation; contended acquisition can block | Audit context and lock ordering; this does not authorize arbitrary sleeping operations inside every critical section |
| `raw_spinlock_t` | Retains raw spinning/non-preemptible semantics | Keep sections bounded; do not sleep |
| Most device IRQ handlers | Forced threading makes substantial work schedulable | Some handlers stay in hard-IRQ context; identify flags and actual thread priorities |
| Softirq work | RT changes execution and locking context | Do not assume all softirqs always execute only in `ksoftirqd` |

Read the kernel's [real-time theory](https://docs.kernel.org/core-api/real-time/theory.html) and [RT/non-RT differences](https://docs.kernel.org/core-api/real-time/differences.html). A driver that works with one preemption model still needs an audit for another. In particular, a high-priority FIFO thread spinning for a lower-priority callback on the same CPU can prevent the callback from ever completing.

`CONFIG_HZ_1000` controls a periodic tick configuration; it is not a guarantee of 1 ms timer accuracy or a prerequisite for all high-resolution timers. Tick suppression also has conditions. Learn the relevant kernel configuration instead of copying a list of boot flags.

## Priority can help one task and starve its dependency

Assume a control thread uses FIFO priority 80 and waits for input produced by an IRQ thread at 50. Blocking while waiting lets the dependency run. Busy-waiting indefinitely at 80 on their only allowed CPU prevents it from running. Raising the consumer to 99 makes this failure worse.

Write the dependency graph before assigning priorities. Include device IRQ threads, network receive work, logging, memory reclaim, GPU completion, and the task that provides the input. FIFO tasks also remain subject to CPU eligibility and configured real-time bandwidth controls. Priority inheritance helps a lock dependency; it does not solve an arbitrary polling loop or a deadlock.

## Inspect before changing the machine

These are read-only commands; availability depends on the host:

```bash
uname -a
cat /proc/cmdline
chrt -p $$
taskset -pc $$
cat /sys/kernel/tracing/available_tracers
```

If tracefs is unavailable or permission is denied, record that. An unavailable observation is not evidence that the feature is disabled. Save the kernel configuration and board/firmware versions using your distribution's supported inspection path.

For the runnable C API exercise, use [rt-demo](demos/rt-demo/README.md). Its normal mode checks affinity and synchronization. Optional FIFO and memory-lock requests fail visibly when denied. The exercise was compiled and executed on a Linux host; it is **not** an RT benchmark and does not establish a control deadline.

## A bounded timer experiment

The following is a **documented target experiment, not a run performed for this lesson**. Install the distribution's `rt-tests`, inspect `cyclictest --help`, and run on a disposable development target with the intended workload. The first command avoids requesting a FIFO policy and asks the tool not to tune power management:

```bash
cyclictest --default-system --policy=other --threads=1 --interval=1000 --duration=60 --quiet
```

For a separately configured RT experiment, use a justified priority, permissions, CPU placement, and optional memory locking. Do not jump to priority 99. Record the exact command and failures. See the maintained [cyclictest manual](https://manpages.debian.org/testing/rt-tests/cyclictest.8.en.html) for version-specific options.

Report sample count, duration, interval, min/mean/max, histogram range and overflow counts, workload, CPU placement, temperature, clock behavior, and trace overhead. A histogram ending at 200 µs must not silently discard a 400 µs observation. An observed maximum over 60 seconds is not a proven upper bound over the product's lifetime. A quiet machine and a loaded machine answer different questions.

**Check:** 100,000 observations have maximum 85 µs against a 100 µs wakeup allocation. Has a 2 ms end-to-end deadline been proved? **Answer:** no. This run supports an empirical wakeup result under its conditions; execution, blocking, transfers, rare disturbances, and the full path remain to be evaluated.

## Diagnose the source of a miss

Start with timestamps that share a documented clock domain. Split sensor acquisition, ready-to-run delay, execution, queueing, and completion. Hardware timestamps may require clock conversion before subtraction.

Use available ftrace tracers or scheduler events to investigate an observed delay. There is no generic tracer named `latency`: inspect `available_tracers` and choose an actually available tracer such as `wakeup_rt`, `irqsoff`, or `preemptoff` that answers your question. RT configuration changes which tracers are useful. Save and restore tracing settings on a test system; recording itself changes the workload. The [ftrace documentation](https://docs.kernel.org/trace/ftrace.html) explains each tracer's observation boundary.

On x86, firmware System Management Interrupts can stop normal OS execution while a clock source continues advancing. Such a pause can therefore appear in a wakeup-latency measurement. Hardware-latency tools investigate unexplained execution gaps; a gap alone does not prove a particular SMI source. Do not transfer x86 SMI terminology directly to an Arm board. Read the kernel's [hardware-latency detector explanation](https://docs.kernel.org/trace/hwlat_detector.html).

## Memory and isolation are contracts, not switches

Preallocate bounded working sets, check allocation/locking errors, and touch the pages and thread stacks that the critical path will use. `mlockall` flags, limits, mapping types, later allocations, and `MCL_ONFAULT` change the contract; see [memory locking](https://man7.org/linux/man-pages/man2/mlock.2.html). It does not reserve CPU time, finish GPU work, or make arbitrary future allocation safe. The demo deliberately describes its limited stack-touch coverage.

CPU affinity is per thread, and isolation has several separate controls; continue to [Lecture 8](Lecture-08.md). Frequency policy, thermal limits, firmware, and power budgets must be measured together. Disabling power management or memory policies without evidence can worsen the real deployment.

## Completion exercise

Create a one-page timing contract for a simulated robot pipeline. Specify the release event, deadline, input size, queue bound, overload policy, and clock domains. Allocate budgets whose sum fits the deadline, then introduce a queue delay that breaks it. Label every number **assumed**, **measured**, or **bounded by analysis**. Explain which measurement could falsify your design and what the system does after a missed deadline. A kernel configuration alone is not the completion artifact.
