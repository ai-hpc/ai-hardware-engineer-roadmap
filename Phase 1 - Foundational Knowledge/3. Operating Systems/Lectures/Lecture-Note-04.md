# Review 4: Scheduling, Real-Time Behavior, and CPU Placement

Use this page after [Lecture 6](Lecture-06.md), [Lecture 7](Lecture-07.md), and [Lecture 8](Lecture-08.md). Try the questions before revealing or reading the answers. Return after two days and one week with a new example from your project.

## 1. Separate CPU time from response time

A job executes for 3 ms, waits runnable for 2 ms, waits for a device for 4 ms, and then executes for 1 ms. Its CPU demand is **4 ms** and its response time is **10 ms**, assuming these intervals cover the complete sequential path. A 6 ms execution budget does not imply that an 8 ms response deadline is met.

Draw the timeline and label the waits. If CPU and device work overlap, use the dependency graph rather than blindly summing overlapping durations.

## 2. Choose a policy from the workload

**Question:** Does switching every worker to FIFO priority 99 make a robot safer?

**Answer:** No. A high-priority worker can starve the lower-priority thread that produces its input. Define critical work, release conditions, blocking, CPU eligibility, and budgets. Normal fair scheduling, FIFO/RR, and deadline scheduling provide different contracts. Kernel settings and permissions must be recorded; none schedules GPU operations by itself.

**Question:** What did EEVDF change?

**Answer:** Modern Linux fair scheduling uses eligibility and virtual deadlines to choose runnable work. A virtual deadline is part of fair-service accounting, not the robot's wall-clock actuation deadline. Refer to the current kernel's implementation and [Lecture 6](Lecture-06.md), rather than treating one historical CFS timeslice as a universal constant.

## 3. Interpret an RT measurement

**Question:** A 60-second timer run has maximum wakeup latency 40 µs. Has the platform proved a 50 µs worst-case bound?

**Answer:** No. It has an observed maximum over the recorded workload, duration, configuration, and instrumentation. Repeatability and stress coverage help evaluation; they do not turn finite testing into a proof. Measure the complete sensor-to-command path separately.

**Question:** Does PREEMPT_RT make every instruction preemptible?

**Answer:** No. It reduces important non-preemptible regions and threads much interrupt work. Raw critical sections, some IRQ paths, firmware, devices, and contention remain. Ordinary `spinlock_t` has different semantics under RT; consult [Lecture 9](Lecture-09.md) before moving driver code between configurations.

## 4. Explain CPU placement without overclaiming

**Question:** A worker's allowed mask is `{2, 3}`. Is it pinned to one CPU?

**Answer:** No. It can run on either allowed CPU, subject to effective constraints. A single-CPU mask restricts this thread more tightly, but does not reserve exclusive use of that CPU or its shared caches.

**Question:** Is `isolcpus` plus `nohz_full` equivalent to bare metal?

**Answer:** No. Load balancing, tick suppression, RCU callbacks, IRQ placement, housekeeping, and task eligibility are separate controls. Cpuset partitions offer supported runtime isolation mechanisms where available. Verify the actual effective sets and remaining kernel activity.

**Question:** Does the `performance` governor disable thermal throttling?

**Answer:** No. CPU frequency policy, driver behavior, thermal limits, firmware, and idle states all matter. Record sustained conditions alongside latency.

## 5. Produce a small evidence package

Run the normal, unprivileged [C thread/API demo](demos/rt-demo/README.md), including one permitted CPU chosen from the actual affinity set. Save compiler/version, command, requested and effective affinity, and successful counter result. Run malformed-argument cases and explain why silently continuing after a failed affinity or lock request would invalidate a benchmark claim.

Then prepare one timing budget with an explicit queue bound and overload policy. Label every quantity as assumed, measured, or analytically bounded. Finish by explaining why a functioning API demo is useful evidence but cannot certify a robot's real-time behavior.
