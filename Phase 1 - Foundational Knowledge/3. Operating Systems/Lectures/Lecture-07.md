# Lecture 7: Real-Time Linux: PREEMPT_RT & Determinism

## Overview

The core problem this lecture addresses is: how do you make Linux respond to external events within a guaranteed time bound? Standard Linux is designed for throughput — it defers low-priority work but gives no hard promises about when any particular task will run. Real-time systems flip this priority: worst-case latency matters more than average throughput. The mental model to carry here is that of a factory production line with strict time slots — a single missed slot can shut down the entire line, even if the average pace is fine. For an AI hardware engineer, this matters directly: autonomous vehicle control loops, robot servo controllers, and safety watchdogs all require bounded response time. A neural network that computes a correct result 5ms too late is as dangerous as one that computes a wrong result.

---

## Real-Time Definitions

Real-time does **not mean fast**. It means **bounded worst-case latency** — a deadline that must always be met.

- **Soft RT**: missing a deadline degrades quality (e.g., dropped audio frame, delayed video render)
- **Hard RT**: missing a deadline is a system failure (e.g., motor overshoot, brake actuation too late)
- Metric of interest: **worst-case latency (WCL)**, not average; a single 1ms spike violates a 500µs hard deadline even if the average is 10µs

> **Key Insight:** PREEMPT_RT does not make Linux faster — it makes Linux more *predictable*. The goal is bounded worst-case latency, not minimum average latency. A system with average 5µs latency but occasional 500µs spikes fails hard-RT requirements. A system with a consistent 80µs latency passes.

---

## Linux Preemption Models

Configured at build time via `CONFIG_PREEMPT_*`:

| Config | Preemptibility | Typical Worst-Case Latency | Use Case |
|---|---|---|---|
| `PREEMPT_NONE` | Not preemptible (yield points only) | >1ms | Servers, throughput workloads |
| `PREEMPT_VOLUNTARY` | `might_resched()` points added | ~500µs | General desktop Linux |
| `PREEMPT` | Most kernel paths preemptible | ~100–200µs | Interactive desktop |
| `PREEMPT_RT` | Fully preemptible kernel | <50µs achievable | Robotics, motor control, AV |

`PREEMPT_RT` was mainlined in **Linux 6.12** (released November 2024). Prior to that it was a long-running out-of-tree patch series maintained by Thomas Gleixner and Ingo Molnar.

Think of each preemption model as controlling how many "no interrupt" zones exist in the kernel. `PREEMPT_NONE` has many; `PREEMPT_RT` has almost none.

---

## PREEMPT_RT Internals

The PREEMPT_RT patch series achieves **full preemptibility** through three core mechanisms: converting **spinlocks to sleeping locks**, moving **interrupt handlers into schedulable threads**, and making **softirq processing preemptible**. Each mechanism attacks a different source of latency.

### Spinlock Conversion

Under `PREEMPT_RT`, `spinlock_t` is converted to a sleeping `rtmutex`:

- **Pre-RT behavior**: `spin_lock()` disables preemption and busy-waits; no other task can run on that CPU while holding the lock
- **RT behavior**: `spin_lock()` puts the task to sleep if the lock is contended; a higher-priority task can preempt and run
- **`raw_spinlock_t`**: the escape hatch — remains a true non-preemptible spinlock; reserved for very short hardware-critical sections (e.g., per-CPU counter updates, arch interrupt entry code)

The following diagram shows the behavioral difference before and after the conversion:

```
  PREEMPT_NONE / PREEMPT spinlock behavior:
  ┌──────────────────────────────────────────────────┐
  │  CPU 0                                           │
  │  spin_lock(&L)  ── preemption OFF ──────────────►│
  │  [busy-waits if contended]                       │
  │  spin_unlock(&L) ── preemption ON               │
  │                                                  │
  │  HIGH-priority task: CANNOT RUN during spin      │
  └──────────────────────────────────────────────────┘

  PREEMPT_RT rtmutex behavior:
  ┌──────────────────────────────────────────────────┐
  │  CPU 0                     CPU 1                 │
  │  spin_lock(&L)             HIGH-prio task wakes  │
  │  [sleeps if contended] ──► [preempts, runs now]  │
  │  [woken when L released]                         │
  │  spin_unlock(&L)                                 │
  │                                                  │
  │  HIGH-priority task: CAN RUN; no blocked CPU     │
  └──────────────────────────────────────────────────┘
```

> **Key Insight:** The spinlock-to-rtmutex conversion is what makes PREEMPT_RT fundamentally different from `CONFIG_PREEMPT`. With a true spinlock, the CPU is occupied and no higher-priority task can run on it, no matter what. With an rtmutex, the CPU is free while a low-priority task waits for the lock.

### Hardirq Threading

- IRQ handlers become **threaded kernel threads** under RT
- Default thread priority: `SCHED_FIFO`, priority 50
- `IRQF_NO_THREAD` flag: opt-out for specific handlers that cannot sleep (e.g., the timer interrupt hardware path)
- Consequence: an RT task at priority 99 can preempt an IRQ handler running at priority 50

This is a conceptual shift. In standard Linux, interrupts are sacred — they preempt everything. Under PREEMPT_RT, interrupts are just high-priority threads, subject to the same scheduler rules as any other task.

### Softirq Handling

- Softirqs run in `ksoftirqd` kernel threads (one per CPU)
- `ksoftirqd` is a normal preemptible task; RT tasks can preempt it at any point
- Prevents softirq storms from blocking high-priority RT tasks for unbounded time

### Sleeping in Kernel

- Pre-RT: many kernel code paths had "cannot sleep here" constraints because spinlocks held preemption off
- With RT: sleeping is safe in most contexts because `spinlock_t` is now a sleeping lock
- Remaining `raw_spinlock_t` sections still cannot sleep; these must be kept extremely short

> **Common Pitfall:** Driver developers who call `msleep()` or `schedule()` while holding a `raw_spinlock_t` will cause a kernel BUG on PREEMPT_RT kernels. If your driver was written for non-RT and uses spinlocks throughout, audit every critical section to ensure it contains no sleep points.

---

## Measuring Scheduling Latency

To verify that your RT configuration achieves the required latency bounds, you need a measurement tool. **`cyclictest`** is the standard tool for measuring **RT scheduling latency**.

```bash
# Basic: 1 thread, SCHED_FIFO priority 99, nanosleep, 1ms interval, 10000 loops
cyclictest -t1 -p 99 -n -i 1000 -l 10000

# Histogram mode: 60-second run, histogram up to 200µs buckets
cyclictest --histogram=200 -D 60s -p 99 -n
```

The **histogram mode** is critical for safety analysis. You are not just interested in average latency — you need to know the **entire distribution, especially the tail**. A single data point in the 400µs bucket can fail a 300µs requirement.

Targets by application domain:

| Domain | Max Acceptable Latency |
|---|---|
| AV planning loop (soft RT) | <100µs |
| Robotics servo control | <500µs |
| Motor drive control (hard RT) | <50µs |
| Safety-critical (ASIL-B certified) | <20µs |

---

## Latency Sources

Understanding where latency comes from lets you target fixes effectively. Sources fall into two categories: software sources you can control and hardware sources that require firmware or platform changes.

### Quantifiable Software Sources

- **IRQ disable sections**: `raw_spinlock_t` hold time, hardware register access sequences; goal: <1µs
- **Memory allocation on hot path**: `GFP_ATOMIC` bypasses direct reclaim but still acquires zone locks
- **Cache misses / TLB shootdowns**: IPI to flush remote TLBs on large SMP systems adds ~10–30µs
- **CPU frequency transitions**: `schedutil` governor can step frequency mid-task; transition adds up to ~200µs

### SMI (System Management Interrupts)

SMIs are the most dangerous latency source because they are invisible to the OS and impossible to prevent from software alone.

- Generated by BIOS/UEFI/BMC firmware for power management, thermal throttling, ECC memory scrubbing
- **Invisible to the OS**: CPU enters SMM (ring -2), OS clock stops, OS cannot observe or account for this time
- Typical impact: 50–300µs per SMI event; some platforms fire 10–100 SMIs per second
- **Detection tool**: `hwlatdetect` — polls a hardware timer in a tight loop; large polling gaps indicate SMI

```bash
# Run for 60 seconds; report any gap larger than 20 microseconds
hwlatdetect --duration=60s --threshold=20
# If this reports violations, the platform firmware must be tuned — software alone cannot fix SMI latency
```

`hwlatdetect` works by **monopolizing a CPU** and measuring gaps between hardware timer reads. Any gap larger than the polling interval indicates **something invisible (an SMI) stole CPU time**.

> **Common Pitfall:** On many x86 server platforms, ECC memory scrubbing generates SMIs every few seconds. On some BIOS versions this cannot be disabled and adds 100–300µs spikes. This must be discovered during bring-up, not after ASIL certification testing begins.

### NUMA and Hardware Effects

- Cross-NUMA memory access adds ~100ns per LLC miss on two-socket servers
- CPU C-state exit latency: C1 ~1µs, C6 ~100µs; use `idle=poll` or `intel_idle.max_cstate=1` to eliminate
- Turbo boost frequency settling after idle exit adds variable latency; `performance` governor avoids this

---

## RT Tuning Checklist

Tuning RT latency is a layered process: kernel configuration sets the foundation, boot parameters isolate the hardware, and runtime settings and application code finalize the setup.

### Kernel Configuration

```
CONFIG_PREEMPT_RT=y    # enable fully preemptible kernel
CONFIG_HZ_1000=y       # 1ms timer tick (higher resolution)
CONFIG_NO_HZ_FULL=y    # enable tickless operation on isolated CPUs
CONFIG_RCU_NOCB_CPU=y  # enable RCU callback offloading
```

### Boot Parameters

```
isolcpus=2,3 nohz_full=2,3 rcu_nocbs=2,3 irqaffinity=0,1
```

- `isolcpus=`: removes CPUs from kernel scheduler pool; tasks must be explicitly placed with `taskset`
- `nohz_full=`: disables the periodic scheduler tick on isolated CPUs (tickless operation)
- `rcu_nocbs=`: offloads RCU callbacks to `rcuoc` kthreads running on non-isolated CPUs
- `irqaffinity=`: routes all hardware IRQs away from isolated CPUs

These four parameters **work together**. `isolcpus` removes the CPU from the scheduler. `nohz_full` stops the kernel from interrupting it every millisecond. `rcu_nocbs` stops RCU from firing callbacks on it. `irqaffinity` stops hardware interrupts from landing on it. Together, they create a **nearly bare-metal execution environment** for your RT task.

### Runtime Tuning

```bash
# Fix CPU frequency to maximum; eliminates frequency ramp-up jitter
cpupower frequency-set -g performance

# Disable NUMA auto-balancing; page migrations cause TLB shootdown jitter
echo 0 > /proc/sys/kernel/numa_balancing

# Disable transparent hugepages; async promotions cause unpredictable latency
echo never > /sys/kernel/mm/transparent_hugepage/enabled
```

### Application-Level RT Setup

```c
mlockall(MCL_CURRENT | MCL_FUTURE);  // pin ALL current + future pages in RAM
// Without this, the kernel can evict pages to swap at any time
// A single major page fault during inference adds 1-10ms of latency

// Set SCHED_FIFO policy with explicit priority
struct sched_param param = { .sched_priority = 80 };
sched_setscheduler(0, SCHED_FIFO, &param);
// SCHED_FIFO: once running, this task runs until it yields or is preempted by higher priority
// Priority 80 leaves room for priority 81-99 tasks (e.g., hardirq threads at 50)

// Pre-fault thread stack before entering RT loop
char stack_probe[8192];
memset(stack_probe, 0, sizeof(stack_probe));
// The kernel allocates stack pages lazily; touching them now forces physical allocation
// Without this, the first deep function call in the RT loop triggers a minor page fault

// Pre-allocate ALL buffers before entering the real-time loop
// No malloc() calls inside the RT loop
// malloc() calls brk()/mmap() which may trigger page faults and take kernel locks
```

This setup sequence must happen before entering the real-time loop. Think of it as the "pre-flight check" — everything that could cause latency is triggered deliberately during initialization, so it cannot happen unexpectedly during the time-critical phase.

> **Common Pitfall:** Calling `mlockall()` without subsequently pre-touching all pages only prevents future eviction — pages not yet faulted in are still demand-paged on first access. Always follow `mlockall()` with a `memset` or similar touch of all buffers you will use in the RT loop.

---

## ftrace Latency Tracer

When `cyclictest` shows a latency spike and you need to find the exact kernel code path responsible, `ftrace` provides the answer.

```bash
echo 0 > /sys/kernel/tracing/tracing_on         # stop tracing while configuring
echo latency > /sys/kernel/tracing/current_tracer  # select the latency tracer
echo 1 > /sys/kernel/tracing/tracing_on         # start tracing
# ... wait for or trigger a latency event ...
cat /sys/kernel/tracing/trace                   # read the captured trace
```

Output includes: timestamp, worst-case wakeup latency, and full kernel call stack from wakeup trigger to task resumption. Identifies the exact function responsible for the worst observed latency spike.

> **Key Insight:** `cyclictest` tells you *that* a latency violation occurred. `ftrace` tells you *where* in the kernel the time was spent. You need both tools: cyclictest to confirm the system meets its deadline, and ftrace to diagnose violations when it doesn't.

---

## QNX: Commercial RTOS Reference

QNX represents the alternative design point: rather than retrofitting a general-purpose OS for real-time, build RT in from the start.

- Microkernel architecture: drivers, filesystems, and network stack run as isolated user-space processes
- Fully preemptive from initial design — no retrofit required, unlike PREEMPT_RT on Linux
- Deployment: QNX CAR platform, BlackBerry IVY, medical devices, avionics flight management
- Adaptive partitioning scheduler: CPU time budget reserved per partition with guaranteed minimums even under overload
- Relevance: QNX + hypervisor pairing (e.g., on NXP S32G, TI TDA4VM) isolates safety-certified RTOS from Linux ADAS stack on the same SoC

In practice, modern AV platforms often run **both**: QNX handles **safety-critical real-time control** (brakes, steering), while Linux runs the **AI inference stack**. A **hypervisor** provides hardware isolation between the two.

---

## Summary

| Config | Max Latency (Typical) | Suitable For | Tradeoff |
|---|---|---|---|
| `PREEMPT_NONE` | >1ms | Batch compute, servers | Highest throughput |
| `PREEMPT_VOLUNTARY` | ~500µs | General Linux | Minimal overhead |
| `PREEMPT` | ~100–200µs | Desktop, light RT | Moderate overhead |
| `PREEMPT_RT` | <50µs | Robotics, AV, motor control | Small throughput reduction |
| QNX | <10µs | Hard RT, safety-certified | Commercial license required |

### Conceptual Review

- **Why doesn't "fast" mean "real-time"?** A system can have low average latency but occasional spikes of 500µs. RT requires a *bounded* worst case — the spike must be eliminated, not just made rare.
- **What does PREEMPT_RT actually change in the kernel?** It converts `spinlock_t` to sleeping `rtmutex`, moves IRQ handlers into schedulable threads, and makes softirq processing preemptible — eliminating the three main sources of unbounded kernel hold time.
- **Why is `raw_spinlock_t` still needed in a PREEMPT_RT kernel?** Some hardware-critical paths (e.g., timer interrupt entry, per-CPU counter updates) genuinely cannot sleep. `raw_spinlock_t` preserves true spinlock semantics for those rare cases and must be kept to sub-microsecond hold times.
- **What does `isolcpus` actually do?** It removes a CPU from the kernel's general scheduler pool at boot time. No task is placed on that CPU unless explicitly assigned via `taskset`. Combined with `nohz_full` and `rcu_nocbs`, it creates near-bare-metal execution for pinned RT tasks.
- **Why run `hwlatdetect` before `cyclictest`?** SMI events are invisible to cyclictest — the CPU disappears from the OS perspective. If hwlatdetect shows violations, no amount of kernel tuning will fix them. Platform firmware must be changed first.
- **Why call `mlockall()` and then memset all buffers?** `mlockall(MCL_FUTURE)` prevents future eviction but does not fault in pages that haven't been accessed yet. Pre-touching forces physical allocation, eliminating both minor and major page faults from the RT execution path.

---

## AI Hardware Connection

- `PREEMPT_RT` is required for openpilot `controlsd`: CAN frame writes to the vehicle bus must complete within 10ms or the safety watchdog triggers a controlled disengage; scheduler jitter cannot be allowed to violate this bound
- `cyclictest` histogram output feeds directly into ISO 26262 ASIL-B timing analysis; the histogram tail (worst observed latency) must fall within the allocated WCET budget for each safety function
- `isolcpus` + `nohz_full` on Jetson Orin (e.g., cores 4–11 for DNN inference, 0–3 for OS) prevents Linux background jitter from appearing as latency spikes in the inference timing loop
- `mlockall(MCL_CURRENT|MCL_FUTURE)` is mandatory in any real-time inference process; a single major page fault during model execution can add 1–10ms of unexpected latency, violating hard deadlines
- `hwlatdetect` is run during ECU bring-up to locate SMI sources on embedded x86 platforms; firmware vendors must bound or eliminate SMI latency to achieve ASIL certification
- `rcu_nocbs=` on isolated inference cores removes RCU callback invocations that would otherwise appear as random multi-microsecond latency spikes inside the inference loop timing window
