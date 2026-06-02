# Lecture 6: CPU Scheduling: CFS, EEVDF & Real-Time Classes

## Overview

The CPU scheduler decides which task runs next and for how long. In a simple world with one task, there is no scheduling problem. In a real AI system with `camerad`, `modeld`, `controlsd`, telemetry loggers, and kernel workers all competing for CPU time, the scheduler's choices directly determine whether inference completes within the frame deadline or misses it by 5 ms. The core challenge is: how do you give every process a fair share of CPU while ensuring that safety-critical real-time tasks — like CAN bus writes and model inference — are never delayed by background work? The mental model is a **strict hierarchy of queues**: real-time tasks are checked first, always, before the fair scheduler even gets a turn. For an AI hardware engineer, knowing how to assign the right scheduling class, set the right priority, and verify the result is the difference between a demo that sometimes glitches and a production system that meets its deadlines.

---

## Scheduler Class Hierarchy

Linux scheduler classes are checked in **strict priority order** — a higher class **always preempts** a lower one:

```
Scheduler Class Priority Hierarchy
┌────────────────────────────────────────────────────────┐
│  stop_sched_class        ← HIGHEST PRIORITY            │
│  CPU migration, stop-machine operations                │
│  (internal kernel use only)                            │
├────────────────────────────────────────────────────────┤
│  dl_sched_class                                        │
│  SCHED_DEADLINE — CBS/EDF; periodic RT tasks          │
│  Example: modeld at 30fps, sensor pipeline             │
├────────────────────────────────────────────────────────┤
│  rt_sched_class                                        │
│  SCHED_FIFO, SCHED_RR — static priority 1–99          │
│  Example: controlsd CAN writes, IMU read loop         │
├────────────────────────────────────────────────────────┤
│  fair_sched_class                                      │
│  SCHED_NORMAL / SCHED_BATCH                           │
│  CFS (< Linux 6.6) or EEVDF (>= Linux 6.6)           │
│  Example: most userspace processes, glibc, Python     │
├────────────────────────────────────────────────────────┤
│  idle_sched_class        ← LOWEST PRIORITY             │
│  SCHED_IDLE — below nice +19                          │
│  Example: telemetry logging, log compression          │
└────────────────────────────────────────────────────────┘
         ↑ Higher class always preempts lower class ↑
```

A single `SCHED_FIFO` task at priority 1 **preempts every CFS/EEVDF task** on the system. There is **no cooperative override** — the kernel enforces it unconditionally.

> **Key Insight:** The scheduler class check happens at every wakeup and preemption point. When `controlsd` (SCHED_FIFO, priority 50) wakes up because a CAN frame arrived, the kernel immediately preempts whatever `SCHED_NORMAL` task was running — even if that task is in the middle of a Python interpreter loop. This unconditional preemption is what makes real-time scheduling deterministic.

---

## CFS: Completely Fair Scheduler (Linux 2.6.23 – 6.5)

**CFS** models an "ideal CPU" running all runnable tasks simultaneously at 1/N speed. Replaced by EEVDF in Linux 6.6.

### vruntime and the Red-Black Tree

Each task accumulates **virtual runtime** weighted by its scheduling weight:

```
vruntime += actual_runtime x (NICE_0_WEIGHT / task_weight)
```

Think of vruntime as a **debt tracker**: tasks with more CPU time accumulated have higher debt. The scheduler always gives CPU time to the task with the **least debt** (lowest vruntime). **Nice values** change the debt accumulation rate: a nice -5 task accumulates debt 3x slower than a nice 0 task, so it gets 3x more CPU share.

Tasks are stored in a **red-black tree** keyed by `vruntime`. The scheduler always picks the leftmost node (minimum vruntime): O(log n) insert/delete, O(1) pick-next.

```
CFS Red-Black Tree (sorted by vruntime)
                    [vruntime=100]
                   /              \
          [vruntime=50]      [vruntime=200]
          /          \
   [vruntime=20] [vruntime=80]
         ↑
    leftmost node = next task to run
```

### Nice Values and Weights

| Nice | Weight | CPU share vs nice-0 |
|---|---|---|
| -20 | 88761 | ~88x baseline |
| -5 | 3121 | ~3x baseline |
| 0 | 1024 | Baseline |
| +10 | 110 | ~1/9 baseline |
| +19 | 15 | ~1/68 baseline |

`weight = 1024 / (1.25 ^ nice)` — each step is a 25% change in CPU allocation.

### CFS Tuning Parameters

```bash
/proc/sys/kernel/sched_latency_ns          # scheduling period (default 6 ms)
/proc/sys/kernel/sched_min_granularity_ns  # minimum slice (default 0.75 ms)
```

With 8 tasks at nice 0: each gets 6 ms / 8 = 0.75 ms. **CFS weakness**: a newly woken latency-sensitive task may **wait up to `sched_latency_ns`** if many tasks have lower vruntime.

> **Common Pitfall:** A freshly woken inference thread can be delayed up to `sched_latency_ns` (6 ms by default) under CFS if other tasks have lower vruntime. This is the classic CFS "wakeup latency" problem. If `modeld` wakes up after waiting for a camera frame and 7 other tasks have lower vruntime, it waits up to 6 ms before running. This is why inference threads that need deterministic latency should use `SCHED_FIFO` or `SCHED_DEADLINE` rather than relying on CFS.

Now that we understand CFS's limitations, let's look at its replacement and why it improves tail latency for AI workloads.

---

## EEVDF: Earliest Eligible Virtual Deadline First (Linux 6.6+)

**EEVDF** replaces CFS entirely in Linux 6.6. CFS code is removed from the kernel tree. EEVDF keeps fairness (like CFS) but fixes CFS’s main weakness: a task that just woke up often has to wait behind tasks that have been running and thus have *lower* vruntime, even when the woken task is latency‑sensitive. EEVDF instead uses **eligibility** and **virtual deadlines** so that a task that is “owed” CPU runs as soon as it is allowed to, without being stuck behind others’ vruntime history.

### The Problem EEVDF Solves (CFS Recap)

Under CFS, the scheduler always runs the task with the **smallest vruntime**. When a task sleeps (e.g. waiting for a camera frame), it stops accumulating vruntime. Other runnable tasks keep running and their vruntime grows. When the sleeping task wakes up, its vruntime is *older* (smaller) than no one’s — it’s just *not the smallest* among the many tasks that have been running. So CFS may not pick it until after several other tasks get their turn, and the woken task can see **wakeup-to-run latency** of up to the full scheduling period (e.g. 6 ms). For a 30 Hz inference pipeline, that extra jitter is unacceptable. EEVDF changes the selection rule so that “how much CPU am I owed?” and “when is my next deadline?” matter more than raw vruntime order.

### Key Concepts (Plain Language)

- **lag**  
  How much CPU time a task is **owed** compared to an “ideal” fair share. If the ideal scheduler would have given the task 10 ms by now but it only got 5 ms (e.g. because it was sleeping), **lag = +5 ms** (it is behind). If it got 12 ms, **lag = −2 ms** (it is ahead).  
  - **Positive lag** → task is behind; it *should* get CPU soon.  
  - **Zero or negative lag** → task has had at least its fair share; others can go first.

- **eligible**  
  A task is **eligible** when it is allowed to run from a fairness point of view. In EEVDF, that means its “virtual start time” is at or before the current virtual time, which corresponds to **lag ≥ 0**: the task is not ahead of its fair share.  
  - So: **eligible ≈ “has non-negative lag (lag ≥ 0)” in the intuition** — actually eligibility is defined via virtual time, but the effect is: if a task has already had *more* than its fair share (negative lag), it is **not** eligible and is not considered for selection until virtual time catches up.

- **virtual deadline**  
  Each runnable task gets a **virtual deadline**: a point in virtual time by which it “should” get its next slice of CPU. The kernel computes this from the task’s `sched_slice` (how much virtual time it gets per scheduling round) and the current vruntime. A task that needs to be more responsive gets a smaller slice and thus an **earlier** virtual deadline.

- **Selection rule**  
  Among **all runnable tasks**, first drop any that are **not eligible**. Among the **eligible** ones, run the task with the **earliest virtual deadline**. So: **Earliest Eligible Virtual Deadline First**.

### Why “Eligible” Matters

If the scheduler only chose “earliest deadline,” a task that had already consumed *more* than its fair share (negative lag) could still have the earliest deadline and keep getting selected, starving others. The **eligible** filter prevents that: only tasks that are not ahead of their fair share (i.e. eligible) are candidates. So we get both **fairness** (only eligible tasks run) and **good latency** (among those, the one with the earliest deadline runs next).

### EEVDF Selection Logic — Step by Step

At each scheduling decision, the kernel has a set of runnable tasks. For each task it knows (conceptually) **lag** and **virtual deadline**.

1. **Compute eligibility**  
   For each runnable task, check if it is **eligible** (virtual start time ≤ current virtual time; in the “who is owed CPU?” view: not ahead of its fair share). Mark ineligible tasks (e.g. lag &lt; 0) as **not** candidates.

2. **Among eligible tasks only**  
   Ignore ineligible tasks for this decision.

3. **Pick earliest virtual deadline**  
   Among the eligible tasks, choose the one whose **virtual deadline** is smallest (earliest in virtual time). That task runs next.

**Example:**

```
EEVDF Selection — at current virtual time T

  Task A:  lag = +5 ms (owed CPU)     deadline = T + 2 ms   → eligible, deadline in 2 ms
  Task B:  lag = +1 ms (owed CPU)     deadline = T + 8 ms   → eligible, deadline in 8 ms
  Task C:  lag = −2 ms (ahead)        deadline = T + 1 ms   → NOT eligible (already got extra CPU)

Eligible set: {A, B}. Earliest deadline among them: A (T+2 ms).
→ EEVDF picks Task A.
```

Task C has the earliest deadline (T+1 ms) but is **ineligible**, so it is not considered. That preserves fairness. Between A and B, A has the earlier deadline, so A runs — which matches the fact that A is more “behind” (larger positive lag).

### Why EEVDF Improves Tail Latency (Especially for AI Workloads)

- **CFS:** A freshly woken task (e.g. inference thread after a frame arrives) often has vruntime larger than that of tasks that have been running. So CFS runs those others first, and the woken task can wait up to the full scheduling period (e.g. 6 ms). That shows up as **high tail latency** and jitter.

- **EEVDF:** When the inference thread wakes up, it typically has **positive lag** (it was blocked, so it is “owed” CPU). So it is **eligible**. Its virtual deadline is set from its `sched_slice`. Among all eligible tasks, the scheduler picks the **earliest deadline**. So the woken inference thread runs as soon as it is eligible and has the earliest deadline — **without** having to “catch up” in vruntime behind long‑running background tasks. That reduces wakeup-to-run latency and tail latency in mixed workloads (inference + logging, telemetry, etc.).

So: **EEVDF improves tail latency** because scheduling is driven by “who is owed CPU?” (eligibility) and “who has the tightest deadline?” (virtual deadline), not by who has the smallest vruntime. A freshly woken, latency‑sensitive task is often both owed CPU and given an early deadline, so it gets scheduled quickly.

### Inspecting EEVDF (and CFS) Behavior

```bash
cat /proc/[pid]/sched | grep slice    # per-task sched_slice; smaller = more responsive
```

Smaller `sched_slice` means the task gets a shorter virtual-time slice per round and an earlier virtual deadline, so it tends to be more responsive under EEVDF.

### Where EEVDF vs CFS Is Used

- **Linux 6.6+** (mainline): EEVDF is the default fair scheduler; CFS code is removed.
- **Jetson JetPack 6.x** (kernel 5.15): still uses CFS.
- **Yocto Scarthgap** and other 6.6+‑based BSPs: EEVDF is the default.

> **Key Insight:** EEVDF’s improvement over CFS for AI workloads is that a freshly woken inference thread that is “owed” CPU (positive lag) is scheduled as soon as it is eligible and has the earliest deadline, regardless of other tasks’ vruntime history. Under CFS, the same thread could be delayed while background tasks with lower vruntime run first. EEVDF’s eligibility + earliest‑deadline rule avoids that and reduces tail latency for mixed workloads (inference, logging, telemetry) on the same cores.

---

## SCHED_FIFO

`rt_sched_class`; static priority 1–99 (99 = highest).

- Runs until it **voluntarily blocks**, calls `sched_yield()`, or is preempted by a higher-priority RT task
- **No time slice** within a priority level — a misbehaving task at prio 99 starves everything below it
- Suitable for tasks with **well-understood, bounded CPU usage**: CAN bus writes, IMU read loops

```bash
chrt -f 50 ./controlsd                   # launch with SCHED_FIFO priority 50
chrt -f -p 70 $(pgrep modeld)            # change priority of running process
chrt -p $(pgrep camerad)                 # query scheduler class and priority
```

### RT Throttling

```bash
cat /proc/sys/kernel/sched_rt_runtime_us   # default 950000 (950 ms)
cat /proc/sys/kernel/sched_rt_period_us    # default 1000000 (1 s) — 95% CPU cap for all RT tasks
```

RT tasks are collectively **throttled to 95% of CPU** by default — non-RT tasks retain at least 5%. Setting `sched_rt_runtime_us = -1` **disables throttling entirely**; used in AV/robotics setups where all RT tasks have known bounded runtime and starvation of non-RT is acceptable.

> **Common Pitfall:** A runaway `SCHED_FIFO` task at high priority that never blocks will starve all lower-priority tasks, including the shell, SSH daemon, and monitoring tools. This can make the system impossible to recover without a hard reboot. Always test `SCHED_FIFO` tasks for correctness (they must block periodically on I/O or `nanosleep`) before running at high priority on a production system. The 5% RT throttling (`sched_rt_runtime_us`) exists as a safety net — disabling it requires confidence that all RT tasks are well-behaved.

---

## SCHED_RR

Same as SCHED_FIFO plus a time slice.

- Within the same priority level, tasks **round-robin** after their slice expires
- Slice length: `/proc/sys/kernel/sched_rr_timeslice_ms` (default 100 ms)
- Useful when **multiple equal-priority RT tasks** must share time without cooperative yielding

---

## SCHED_DEADLINE

`dl_sched_class`; uses **Constant Bandwidth Server (CBS)** with **Earliest Deadline First (EDF)**.

### Parameters

```c
struct sched_attr attr = {
    .size           = sizeof(attr),
    .sched_policy   = SCHED_DEADLINE,
    .sched_runtime  = 5000000,    /* 5 ms: CPU budget consumed before forced descheduling */
    .sched_deadline = 16666666,   /* 16.7 ms: relative deadline from period start (must finish by here) */
    .sched_period   = 16666666,   /* 16.7 ms: period — activates once per period (60 fps) */
};
sched_setattr(0, &attr, 0);       /* requires CAP_SYS_NICE */
```

### Properties

- **Admission control**: kernel rejects `sched_setattr()` with `EBUSY` if adding this task makes sum(runtime/period) > 1.0 on the CPU set — a hard schedulability guarantee
- **Budget enforcement**: task is descheduled after consuming its `runtime` budget; replenished at the next period — misbehaving tasks cannot starve others
- **No static priority**: the kernel's EDF logic dynamically orders tasks by absolute deadline

```bash
# 30fps inference: 10ms budget, 33ms deadline, 33ms period
chrt -d --sched-runtime 10000000 --sched-deadline 33333333 --sched-period 33333333 0 ./modeld
```

```
SCHED_DEADLINE Timeline (30fps, 10ms budget, 33ms period)
t=0          t=10ms       t=16ms       t=33ms       t=43ms
│            │            │            │            │
├────────────┤░░░░░░░░░░░░├────────────┤░░░░░░░░░░░░├──
│ modeld     │ idle/other │ modeld     │ idle/other │
│ runs up to │ period     │ runs up to │ period     │
│ 10ms budget│ continues  │ 10ms budget│ continues  │
└────────────┘            └────────────┘

Legend: ── = modeld running  ░ = other tasks running / modeld done early
```

> **Key Insight:** `SCHED_DEADLINE`'s admission control is a formal schedulability proof at the kernel level. When you call `sched_setattr()` with deadline parameters, the kernel checks whether the sum of all deadline tasks' `runtime/period` ratios still fits within the CPU's capacity. If it doesn't, `EBUSY` is returned. This means the kernel can mathematically guarantee that all admitted deadline tasks will meet their deadlines — something no priority-based scheme (SCHED_FIFO) can provide. This is the correct scheduling class for periodic inference pipelines.

> **Common Pitfall:** Setting `sched_runtime` too low causes `SIGXCPU` to be sent to the task when it exceeds its budget, or the task is simply descheduled early. If `modeld` sometimes finishes in 8 ms but occasionally spikes to 12 ms, setting `sched_runtime = 10ms` will cause occasional early termination. Use profiling (`perf sched latency`, `bpftrace`) to measure the 99th-percentile execution time and set `sched_runtime` to at least that value with some headroom.

---

## Scheduler Inspection

```bash
cat /proc/[pid]/sched                    # vruntime, nr_voluntary_switches, nr_involuntary_switches
schedtool [pid]                          # scheduler class, priority, affinity (human-readable)

# Trace scheduler decisions
trace-cmd record -e sched_switch -e sched_wakeup ./workload
trace-cmd report | head -100
# Shows exact sequence of task switches and wakeups — identifies which task preempted which

# Per-task scheduling latency report
perf sched record -- sleep 5
perf sched latency                       # avg/max wakeup-to-run latency per task
# Most useful field: max wakeup latency — if >1ms on inference thread, investigate

# Run queue latency histogram (eBPF; no recompile needed)
runqlat -m 10                            # histogram in milliseconds, 10 second window
# Shows time tasks spend waiting on run queue before getting CPU
```

### /proc/[pid]/sched Key Fields

| Field | Meaning |
|---|---|
| `nr_voluntary_switches` | Times task gave up CPU willingly (blocking I/O, sleep) |
| `nr_involuntary_switches` | Times task was preempted (slice expired, higher-priority task woke) |
| `se.load.weight` | CFS scheduling weight (derived from nice value) |
| `se.vruntime` | Accumulated virtual runtime |
| `policy` | Scheduler policy integer: 0=NORMAL, 1=FIFO, 2=RR, 6=DEADLINE |
| `prio` | Effective priority: 100=RT prio 99; 120=nice 0; 139=nice 19 |

High `nr_involuntary_switches` on `modeld` indicates **CFS preemption** — first signal to elevate to `SCHED_FIFO` or `SCHED_DEADLINE`.

> **Key Insight:** `nr_involuntary_switches` in `/proc/[pid]/sched` is the diagnostic canary for CFS preemption problems. If this counter grows quickly while `modeld` is running inference, it means the scheduler is forcibly removing `modeld` from the CPU before it finishes — because other tasks have lower vruntime or higher priority. The fix is to elevate `modeld` to `SCHED_FIFO` or `SCHED_DEADLINE`. Check this field first before reaching for more complex profiling tools.

---

## Summary

| Policy | Class | Priority range | Time slice | Preemptible by | Use case |
|---|---|---|---|---|---|
| `SCHED_DEADLINE` | `dl_sched_class` | EDF/CBS dynamic | Per runtime budget | Higher-deadline DL task | Periodic RT: modeld, sensor pipeline |
| `SCHED_FIFO` | `rt_sched_class` | 1–99 (99 highest) | None | Higher RT priority | Hard RT: CAN writes, actuation thread |
| `SCHED_RR` | `rt_sched_class` | 1–99 | `sched_rr_timeslice_ms` | Higher RT priority | Equal-priority RT sharing |
| `SCHED_NORMAL` | `fair_sched_class` | nice -20 to +19 | `sched_latency_ns / n` | Any RT or DL task | General processes, background work |
| `SCHED_IDLE` | `idle_sched_class` | Below nice +19 | CFS/EEVDF slice | Everything else | Telemetry, log compression |

### Conceptual Review

- **Why does a single `SCHED_FIFO` task at priority 1 preempt all `SCHED_NORMAL` tasks?** Scheduler classes are checked in strict descending priority order at every scheduling decision. `rt_sched_class` is checked before `fair_sched_class`. As long as any runnable RT task exists, the fair scheduler never runs. This is the kernel's unconditional guarantee that RT tasks get CPU over normal tasks.
- **What is vruntime and why does CFS use it?** vruntime is the amount of CPU time a task has received, normalized by the task's weight (nice value). CFS picks the task with the lowest vruntime — the one that has received the least fair share. Nice values affect how fast vruntime accumulates: a nice -5 task's vruntime grows 3x slower, so it gets ~3x more CPU share.
- **What is the fundamental weakness of CFS for latency-sensitive tasks?** A freshly woken task may have a higher vruntime than other runnable tasks (because it was sleeping while they accumulated low vruntime). CFS must wait until this task's turn comes around in the scheduling period — up to `sched_latency_ns` (6 ms default). EEVDF fixes this with the eligibility + earliest-deadline-first rule.
- **What does `SCHED_DEADLINE` admission control guarantee?** When `sched_setattr()` is called, the kernel checks whether sum(runtime/period) across all deadline tasks on the CPU set is ≤ 1.0. If yes, it admits the task and guarantees all admitted tasks will meet their deadlines. If no, it returns `EBUSY`. This is a mathematically proven schedulability guarantee.
- **When should you use `SCHED_FIFO` vs `SCHED_DEADLINE`?** Use `SCHED_FIFO` for tasks that run in short, bounded bursts triggered by hardware events (CAN writes, IMU reads) where the completion time is always short and you just need strict priority. Use `SCHED_DEADLINE` for periodic tasks with known CPU budgets per period (inference at 30fps) where you want the kernel to enforce the budget and provide formal schedulability guarantees.
- **What does a high `nr_involuntary_switches` in `/proc/[pid]/sched` indicate?** The task is being preempted by the scheduler — either its time slice expired or a higher-priority task woke up. For an inference thread under CFS, this means the scheduler is removing it mid-inference. The fix is to move to `SCHED_FIFO` or reduce the number of competing tasks on the same CPU via `cpuset` isolation.

---

## AI Hardware Connection

- `SCHED_DEADLINE` maps directly to periodic inference tasks: `modeld` at 30fps declares `runtime=10ms, deadline=33ms, period=33ms`; the kernel's admission control proves schedulability and enforces the budget — no user-space watchdog required.
- `SCHED_FIFO` at priority 50–70 is standard for openpilot `controlsd`: CAN bus writes at 100Hz must not be delayed by CFS jitter, which can exceed 5 ms on an untuned multi-process system.
- EEVDF (Linux 6.6) reduces tail latency for mixed workloads — relevant when TensorRT inference, sensor reading, and logging share the same Jetson without full CPU isolation; newly woken inference threads are scheduled sooner than under CFS's vruntime ordering.
- `chrt -f 50 $(pgrep modeld)` is a standard production tuning step on openpilot and Autoware-based AV stacks; for persistent configuration use systemd unit options `CPUSchedulingPolicy=fifo` and `CPUSchedulingPriority=50`.
- `rt_throttling` disabled (`sched_rt_runtime_us = -1`) is used in safety-certified AV ECU deployments where all RT tasks have formally verified bounded CPU usage and non-RT starvation is mitigated by running telemetry at `SCHED_IDLE`.
- `perf sched latency` after a field test surfaces scheduler-induced delays invisible in application-level timing — the primary diagnostic when inference latency increases in deployment vs. bench testing on the same hardware.

### Real example in openpilot (this repo)

The openpilot code in this roadmap implements the same scheduling concepts from this lecture:

| Lecture concept | Openpilot implementation |
|------------------|---------------------------|
| **SCHED_FIFO** / `chrt -f` | **`common/util.cc`**: `set_realtime_priority(int level)` uses `sched_setscheduler(tid, SCHED_FIFO, &sa)`. The comment states it is *"equivalent to the 'chrt' command"*. Priority is 1–99 (same as Lecture-06). |
| **CPU affinity** (pin task to cores) | **`common/util.cc`**: `set_core_affinity(std::vector<int> cores)` uses `sched_setaffinity(tid, ...)` to pin the calling thread to the given cores — avoids migration and cache thrash for RT threads. |
| **Declarations** | **`common/util.h`**: Declares `set_realtime_priority(int level)` and `set_core_affinity(std::vector<int> cores)`. |

Relevant snippets:

- **`openpilot/common/util.cc`** (lines 36–61): `set_realtime_priority()` — gets TID via `gettid`, fills `sched_param` with `sched_priority`, calls `sched_setscheduler(tid, SCHED_FIFO, &sa)`.
- **`openpilot/common/util.cc`** (lines 63–76): `set_core_affinity()` — builds `cpu_set_t`, calls `sched_setaffinity(tid, ...)`.

Processes like `controlsd` and `modeld` (or their launchers) call these helpers at startup to run with SCHED_FIFO and pinned cores so CAN writes and inference meet deadlines. Search the openpilot tree for `set_realtime_priority` and `set_core_affinity` to find exact call sites.
