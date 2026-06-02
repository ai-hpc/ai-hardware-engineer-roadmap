# Lecture Note 04 (L7, L8, L9): Real-Time Linux, Multi-Core Scheduling & Synchronization

---

## Big Picture: Why This Matters

Imagine a robot arm that must stop within 2 ms when it detects a person. If the OS sometimes delays your “stop” command by 5 ms because it was busy with something else, the system is unsafe — even if it’s usually fast. **Predictability** (worst-case behavior) matters more than average speed.

This note is about three things that make systems predictable:

| Topic | The question it answers |
|-------|-------------------------|
| **Part 1: Real-Time Linux** | “How do I make the OS respond within a guaranteed time?” |
| **Part 2: Multi-Core & CPU Placement** | “How do I keep my critical task on the right CPU and avoid random delays from the scheduler?” |
| **Part 3: Synchronization** | “How do I let multiple threads share data safely without races or deadlocks?” |

**One-line summary:** You need a predictable kernel (Part 1), predictable *placement* of your task on CPUs (Part 2), and correct use of locks so high-priority work isn’t stuck behind low-priority work (Part 3).

---

## How This Note Is Organized

1. **Part 1 — Real-Time Linux:** Bounded response time: preemption models, PREEMPT_RT, measuring and tuning latency.
2. **Part 2 — Multi-Core Scheduling:** Controlling which task runs on which CPU; affinity, isolation, NUMA.
3. **Part 3 — Synchronization:** Spinlocks, mutexes, read/write locks, seqlocks, completions, and lockdep.

**Terms you’ll see:**  
- *Latency* = delay between “something happened” and “we responded.”  
- *Preemption* = a higher-priority task interrupting the current one.  
- *Kernel* = the core OS code that runs with full hardware access.  
- *IRQ* = hardware interrupt (e.g. device finished DMA, timer tick).
- *Top half* = the first part of interrupt handling: the IRQ handler that runs immediately when the interrupt fires; must be very short (e.g. acknowledge hardware, queue work). *Bottom half* = deferred work run after the top half (softirqs, tasklets, workqueues); can do more work without holding the CPU in interrupt context.
- *Critical section* = code that touches shared data and must run with a lock held.

**How to use this note:** Read the **Big Picture** and **How This Note Is Organized** first. Then go to the part you need (Real-Time, Multi-Core, or Synchronization). Each part has short **Context** lines at the start of sections to explain *why* the topic matters before the details.

---

# Part 1: Real-Time Linux (PREEMPT_RT & Determinism)

**Context:** Normal Linux is tuned for throughput (get lots of work done). Real-time is tuned for **guaranteed** response: “no matter what, we respond within X microseconds.”

---

## What “Real-Time” Really Means

**Real-time does not mean “fast.”** It means **bounded worst-case latency**: there is a deadline, and the system must *always* meet it.

| Type | Meaning | Example |
|------|--------|--------|
| **Soft RT** | Missing a deadline hurts quality | Dropped audio frame, delayed video |
| **Hard RT** | Missing a deadline is a failure | Motor overshoot, brake too late |

The metric that matters is **worst-case latency**, not average. One 1 ms spike breaks a 500 µs hard deadline even if the average is 10 µs.

> **Takeaway:** PREEMPT_RT does not make Linux faster; it makes it more **predictable**. You care about the worst delay, not the average.

---

## Linux Preemption Models

**Preemption** means: “Can a higher-priority task interrupt the one currently running?”  
- **No preemption:** The kernel can run for a long time without giving the CPU to your RT task.  
- **Full preemption (PREEMPT_RT):** The kernel can be interrupted almost everywhere, so your RT task can run soon.

This is chosen when the kernel is **built**, via `CONFIG_PREEMPT_*`:

| Config | Can kernel be preempted? | Typical worst-case delay | Good for |
|--------|---------------------------|---------------------------|----------|
| `PREEMPT_NONE` | Almost no | >1 ms | Servers, max throughput |
| `PREEMPT_VOLUNTARY` | Only at specific yield points | ~500 µs | General desktop |
| `PREEMPT` | Most kernel code | ~100–200 µs | Interactive desktop |
| `PREEMPT_RT` | Almost entire kernel | <50 µs possible | Robotics, motor control, AV |

**PREEMPT_RT** was merged into mainline in **Linux 6.12** (late 2024). Before that it was a long-standing out-of-tree patch.

**Simple picture:** Think of the kernel as having “no entry” zones where nothing can interrupt. PREEMPT_NONE has many such zones; PREEMPT_RT has almost none, so your RT task can run sooner.

---

## How PREEMPT_RT Works (Three Main Changes)

In a normal kernel, three things can block your RT task for a long time: spinlocks (CPU busy-waiting), interrupt handlers, and softirqs. PREEMPT_RT fixes each one.

### 1. Spinlocks Become “Sleeping” Locks (rtmutex)

A **spinlock** is a lock where, if the lock is busy, the CPU just waits in a loop (“spins”) instead of doing something else. In the normal kernel that also disables preemption, so your RT task can’t run on that CPU until the lock is released.

- **Normal kernel:** `spin_lock()` = busy-wait + preemption off. That CPU is “stuck” until the lock is released.
- **PREEMPT_RT:** `spin_lock()` is replaced by a **sleeping** lock (rtmutex). If the lock is busy, the task **sleeps** (gives up the CPU), so a higher-priority task can run. When the lock is free, the task is woken.
- **`raw_spinlock_t`:** The exception — stays a real spinlock for tiny hardware-critical sections (e.g. timer interrupt entry). Never sleep while holding this.

**In plain English:** Under RT, most “spinlocks” no longer hog the CPU; they let the scheduler run your RT task. That’s why worst-case latency drops.

### 2. Interrupt Handlers as Threads (Hardirq Threading)

**Context:** Linux splits interrupt handling into a **top half** (the IRQ handler that runs immediately when the interrupt fires — must be very short, e.g. acknowledge hardware and queue work) and a **bottom half** (deferred work such as softirqs or tasklets). In normal Linux, when a hardware interrupt (IRQ) fires, the CPU runs the **top half** immediately and it can’t be preempted by your RT task. That can add unpredictable delay.

- Under PREEMPT_RT, **IRQ handlers (top half) run as normal kernel threads** (default priority 50).
- So an RT task at priority 99 **can preempt** an IRQ handler. Interrupts are no longer “untouchable”; they obey the same scheduler.
- Handlers that truly cannot sleep keep the old behavior with `IRQF_NO_THREAD`.

### 3. Softirqs in Preemptible Threads (Bottom Half)

**Context:** The **bottom half** is where the kernel does deferred work after an interrupt (e.g. network RX processing, block I/O completion). One common form is **softirqs**. In the normal kernel they can run for a long time and block your RT task.

- Under PREEMPT_RT, softirqs run inside `ksoftirqd` **threads** (one per CPU), which are normal preemptible tasks.
- So RT tasks can preempt softirq work and avoid long, unbounded delays from “softirq storms.”

---

## Measuring Scheduling Latency: cyclictest

**Why measure?** You need to *prove* that worst-case delay stays under your requirement (e.g. 300 µs). **cyclictest** is the standard tool: it wakes up at a fixed interval and measures how late the wakeup actually was.

```bash
# Basic: 1 thread, highest RT priority, 1 ms interval, 10000 loops
cyclictest -t1 -p 99 -n -i 1000 -l 10000

# Histogram: 60 s run, buckets up to 200 µs (see full distribution, including worst case)
cyclictest --histogram=200 -D 60s -p 99 -n
```

**Why use histogram mode?** The **average** latency can look fine while a few rare spikes break your deadline. The histogram shows the full distribution — including the worst case. One sample in the 400 µs bucket means you fail a 300 µs requirement.

**Rough targets by domain** (max acceptable delay):

| Domain | Max acceptable latency |
|--------|------------------------|
| AV planning (soft RT) | <100 µs |
| Robotics servo | <500 µs |
| Motor control (hard RT) | <50 µs |
| Safety-critical (e.g. ASIL-B) | <20 µs |

---

## Where Latency Comes From

When cyclictest shows a spike, the delay came from somewhere. Fixing it means knowing the possible sources.

### Software (you can improve)

- **IRQ-off / raw_spinlock sections** — Keep under ~1 µs.
- **Memory allocation on hot path** — Even `GFP_ATOMIC` can take locks.
- **Cache misses / TLB shootdowns** — IPIs on big SMP systems can add ~10–30 µs.
- **CPU frequency changes** — e.g. `schedutil` stepping frequency can add up to ~200 µs.

### SMI (System Management Interrupts) — Often Invisible

**Context:** Some interrupts are handled by the **firmware** (BIOS/UEFI), not the OS. The CPU enters a special mode (SMM); the OS’s clock effectively stops. The OS cannot see or fix this.

- SMIs are used for power management, thermal throttling, ECC memory scrubbing, etc.
- Typical cost: 50–300 µs per event; some platforms fire 10–100 SMIs per second.
- **Tool:** `hwlatdetect` — runs a tight loop reading a hardware timer; if there’s a big gap between reads, something invisible (SMI) stole the CPU.

```bash
hwlatdetect --duration=60s --threshold=20
```

If this reports violations, **software tuning alone is not enough**; firmware/BIOS may need changes (e.g. ECC scrubbing).

### Hardware / Platform

- **NUMA:** On multi-socket machines, accessing memory on another socket is slower. Cross-socket access adds cost on every cache miss.
- **CPU idle (C-states):** Deeper sleep (e.g. C6) saves power but takes longer to wake (e.g. ~100 µs). For RT, limit depth with `idle=poll` or `intel_idle.max_cstate=1`.
- **Turbo / frequency:** If the CPU changes frequency in the middle of your task, timing becomes unpredictable. Use the `performance` governor so frequency stays at max.

---

## Real-Time Tuning Checklist

Tuning is layered: **kernel config** sets the base, **boot parameters** reserve and isolate CPUs, **runtime** settings fix frequency and memory behavior, and **your application** must lock memory and avoid allocations in the RT loop.

### Kernel config (examples)

```
CONFIG_PREEMPT_RT=y
CONFIG_HZ_1000=y
CONFIG_NO_HZ_FULL=y
CONFIG_RCU_NOCB_CPU=y
```

### Boot parameters (example: CPUs 2,3 isolated)

```
isolcpus=2,3 nohz_full=2,3 rcu_nocbs=2,3 irqaffinity=0,1
```

| Parameter | Effect |
|-----------|--------|
| `isolcpus=` | These CPUs are not used by the general scheduler; only tasks you pin go there. |
| `nohz_full=` | No periodic timer tick on these CPUs (fewer interruptions). |
| `rcu_nocbs=` | RCU callbacks run on other CPUs, not on isolated ones. |
| `irqaffinity=` | Hardware IRQs go to CPUs 0,1 only, not to isolated CPUs. |

### Runtime

```bash
cpupower frequency-set -g performance
echo 0 > /proc/sys/kernel/numa_balancing
echo never > /sys/kernel/mm/transparent_hugepage/enabled
```

### In your RT application (before the real-time loop)

**Context:** The kernel can swap your process’s memory to disk or allocate it lazily. A single **page fault** (bringing a page from disk or allocating it) in the RT loop can add 1–10 ms and break your deadline. So you do all “risky” work before entering the time-critical loop.

- **Lock all memory:** `mlockall(MCL_CURRENT | MCL_FUTURE)` — tells the kernel not to swap your pages out. (If a page is swapped out and you touch it, you get a major fault = huge delay.)
- **Use SCHED_FIFO:** e.g. priority 80 — your task is scheduled as real-time and can preempt normal tasks.
- **Pre-fault stack and buffers:** Touch every page you’ll use (e.g. `memset` stack and buffers). That forces the kernel to allocate them *now* so no fault happens later in the RT loop.
- **No malloc in the RT loop:** Allocate everything before the loop; inside the loop, no `malloc` (it can take locks and trigger faults).

> **Pitfall:** `mlockall()` only prevents *eviction*; it doesn’t load pages that were never touched. You must **touch** the pages (e.g. with `memset`) so they are actually in RAM.

---

## Finding What Caused a Latency Spike: ftrace

- **cyclictest** tells you *that* a spike happened (and how big it was).
- **ftrace** (latency tracer) tells you *where* in the kernel the time was spent — which function was running during the delay.

```bash
echo 0 > /sys/kernel/tracing/tracing_on
echo latency > /sys/kernel/tracing/current_tracer
echo 1 > /sys/kernel/tracing/tracing_on
# ... run workload until spike ...
cat /sys/kernel/tracing/trace
```

You get timestamps, worst-case wakeup latency, and the kernel call stack from wakeup to task run.

---

## QNX as a Contrast (Commercial RTOS)

- **Microkernel:** drivers, filesystems, network run as separate user processes.
- **Designed for RT from the start** (no “retrofit” like PREEMPT_RT).
- Used in QNX CAR, medical, avionics. Often **paired with a hypervisor**: QNX for safety-critical control (brakes, steering), Linux for AI stack; hypervisor keeps them isolated on the same SoC.

---

# Part 2: Multi-Core Scheduling, CPU Affinity & isolcpus

**Context:** On a multi-core system, the Linux scheduler tries to spread work evenly across CPUs. It may **move your task** from one CPU to another. Each move flushes that task’s caches and can add hundreds of microseconds of jitter. For real-time and inference you want **fixed assignment**: “this task always runs on these CPUs, and nothing else runs there.”

---

## The Problem

With many CPUs, the scheduler constantly moves tasks to “balance load.” That’s good for **fairness** and **throughput** but bad for **latency predictability**: migrations flush caches and add hundreds of microseconds. For RT and inference you want **assigned seats** — certain tasks on certain CPUs, with minimal interference from the OS or other tasks.

---

## SMP Scheduler in One Paragraph

**SMP** = symmetric multi-processing (multiple CPUs). Linux keeps a **runqueue** (list of runnable tasks) per CPU. A **load balancer** periodically tries to equalize load: move tasks from busy CPUs to idle ones, preferring same core → same package → same NUMA node. For throughput this is good; for RT and inference you usually **don’t** want the scheduler to touch your critical cores — you isolate them and pin your task there.

---

## CPU Topology (Why Placement Matters)

**Context:** Not all “CPUs” are equal. Two logical CPUs can be **SMT siblings** (hyperthreads) on the same physical core — they share L1/L2 cache and execution units. Putting two heavy threads on the same physical core can make both slower due to contention.

Rough hierarchy:

```
Socket (package) → Die → Physical core → SMT threads (e.g. 2 logical CPUs per core)
```

**Rule of thumb:** For inference, often **one thread per physical core** is better than using both SMT siblings for two heavy threads.

**Inspect topology:**

```bash
lscpu
lscpu -e
cat /sys/devices/system/cpu/cpu0/topology/core_id
cat /sys/devices/system/cpu/cpu0/topology/core_cpus_list
```

---

## CPU Affinity (Pinning a Task to CPUs)

**Affinity** = which CPUs a task is *allowed* to run on. Without setting it, the scheduler can run your task on any CPU and may migrate it; with affinity, your task only runs on the CPUs you specify.

**In C:**

```c
cpu_set_t mask;
CPU_ZERO(&mask);
CPU_SET(2, &mask); CPU_SET(3, &mask);
sched_setaffinity(pid, sizeof(mask), &mask);
```

**From shell:**

```bash
taskset -c 2,3 ./inference_app
taskset -cp 2,3 <pid>
```

**Why pin?** Fewer migrations → warmer caches, less TLB flush cost, more predictable timing.

**Important:** Affinity only restricts *your* task. It does **not** stop other tasks or the kernel from using those CPUs. For full isolation you need **isolcpus** (and related boot options) so the scheduler doesn’t put anyone else there.

---

## isolcpus: Reserve CPUs for Your RT/Inference Tasks

**Context:** Affinity says “my task runs only on CPUs 2,3.” But the scheduler can still put *other* tasks (daemons, kernel threads) on 2 and 3, which causes cache pollution and jitter. **isolcpus** says “CPUs 2 and 3 are *off limits* to the general scheduler”; only tasks you explicitly pin there will run there.

- **`isolcpus=2,3`** — At boot, CPUs 2 and 3 are **removed from the general scheduler**. No normal task is placed there unless you pin it with `taskset` or `sched_setaffinity`.
- Usually combined with:
  - **`nohz_full=2,3`** — No periodic timer tick on those CPUs (fewer interruptions).
  - **`rcu_nocbs=2,3`** — RCU callbacks run on other CPUs, not on 2,3.
  - **`irqaffinity=0,1`** — Hardware IRQs are routed only to CPUs 0,1, not to 2,3.

**Result:** Isolated CPUs run almost only what you pin there; OS jitter can drop from hundreds of µs to well under 10 µs.

| Mechanism | What it does | When it applies |
|-----------|----------------|------------------|
| `sched_setaffinity` / taskset | Restrict one task to a set of CPUs | Per process |
| `isolcpus` | Remove CPUs from general pool | Boot time |
| cpuset cgroup | Restrict a group of processes to CPUs (and NUMA nodes) | Runtime (e.g. containers) |
| numactl | Bind process to NUMA node(s) and CPUs | Per run |

---

## CPU Frequency: Use “performance” Governor

Variable frequency causes **timing jitter**. Use a fixed max frequency on RT/inference cores:

```bash
cpupower frequency-set -g performance
```

On mobile SoCs (e.g. Jetson), thermal throttling can still change frequency; monitor scaling_cur_freq under load.

---

## cpuset Cgroups (Runtime CPU Partitioning)

Without rebooting, you can assign a **group** of processes to specific CPUs (and NUMA nodes) using cpusets:

```bash
mkdir /sys/fs/cgroup/cpuset/inference
echo "4-7" > /sys/fs/cgroup/cpuset/inference/cpuset.cpus
echo "0"   > /sys/fs/cgroup/cpuset/inference/cpuset.mems
echo <pid> > /sys/fs/cgroup/cpuset/inference/cgroup.procs
```

Kubernetes can do similar things with `cpu_manager_policy=static` and `topologyManagerPolicy=single-numa-node` for guaranteed pods.

---

## Cache and Intel CAT (Optional)

**Context:** L1/L2 caches are per-core; **L3 (LLC)** is shared by all cores in a package. So even if your inference task is pinned to one core, another task on another core can **evict** your data from L3, causing extra cache misses and latency.

- **Intel CAT (Cache Allocation Technology / RDT):** Lets you partition L3 “ways” between groups (e.g. “inference” vs “OS”). The OS and other workloads then cannot use the ways reserved for inference, so they can’t evict your hot data. Configure via `resctrl` or tools like `pqos`.

---

## NUMA Affinity

**Context:** **NUMA** (Non-Uniform Memory Access) means that on multi-socket or some SoC systems, memory is attached to a specific “node” (e.g. socket). Accessing memory on *your* node is faster than accessing memory on another node. So **which NUMA node** your process’s memory comes from matters; cross-NUMA access costs more on every cache miss.

```bash
numactl --cpunodebind=0 --membind=0 ./inference
numastat -p <pid>
nvidia-smi topo -m   # CPU–GPU topology
```

Turn off **AutoNUMA** on latency-sensitive systems: `echo 0 > /proc/sys/kernel/numa_balancing`. Otherwise page migrations cause TLB shootdowns and jitter.

---

# Part 3: Synchronization — Spinlocks, Mutexes, RW Locks & Seqlocks

**Context:** When multiple threads (or CPUs) touch the same data, they must coordinate. Otherwise one thread might read half-updated data or two writers might corrupt memory. That’s a **race condition**. Synchronization primitives are the tools that enforce rules like “only one writer at a time” or “many readers OK, one writer exclusive.”

---

## The Problem: Race Conditions

When two threads use the same data without coordination, the result can depend on **timing** — a **race condition**. Synchronization primitives ensure that only one writer (or many readers, depending on the primitive) access shared state at a time.

**Critical section** = the piece of code that touches shared data. We want three properties:

1. **Mutual exclusion** — At most one thread in the critical section at a time.
2. **Progress** — If no one is inside, a waiting thread eventually gets in.
3. **Bounded waiting** — No thread waits forever (no starvation).

---

## Spinlock (`spinlock_t`)

- **Behavior:** To acquire the lock, the CPU **busy-waits** (spins) in a loop until the lock is free. No context switch. While the lock is held, preemption is disabled on that CPU.
- **Use when:** The critical section is **very short** (< ~1 µs) and you might be in **interrupt context** (where you **cannot sleep** — e.g. inside an IRQ handler).
- **When sharing data with an IRQ handler:** Use `spin_lock_irqsave` / `spin_unlock_irqrestore`. This disables local interrupts while you hold the lock so the handler can’t run on the same CPU and try to take the same lock (which would deadlock).

```c
spin_lock_irqsave(&lock, flags);
/* critical section */
spin_unlock_irqrestore(&lock, flags);
```

**Rule:** Never sleep (no `kmalloc(GFP_KERNEL)`, `msleep()`, etc.) while holding a spinlock. On PREEMPT_RT, `spinlock_t` becomes a sleeping lock (rtmutex); `raw_spinlock_t` stays a real spinlock — do not sleep with it.

---

## Mutex (`struct mutex`)

- **Behavior:** If the lock is busy, the task **blocks** (sleeps) and the scheduler runs other tasks. No spinning — the CPU is free to do other work.
- **Use when:** Longer critical sections, and **only in process context** (not in interrupt handlers, because interrupts cannot sleep).
- **Variants:** `mutex_trylock` (don’t block, return success/failure), `mutex_lock_interruptible` (can be interrupted by a signal).

**rtmutex:** A mutex with **priority inheritance**. If high-priority task H is waiting on a lock held by low-priority task L, the kernel temporarily **boosts L’s priority** to H’s level so L runs sooner, releases the lock sooner, and H can proceed. This avoids **priority inversion** (H stuck behind L, which is stuck behind medium-priority tasks). Under PREEMPT_RT, many kernel “spinlocks” are actually rtmutexes.

---

## Read/Write Semaphore (`rw_semaphore`)

- **Behavior:** Many **readers** can hold the lock at the same time, OR one **writer** exclusively — never both. Good when reads are frequent and writes are rare.
- **Use when:** Reads are much more frequent than writes (e.g. config tables, model weights). Many threads can read in parallel; only one can write.
- **APIs:** `down_read` / `up_read`, `down_write` / `up_write`; `downgrade_write` turns a write lock into a read lock without releasing (so no other writer can slip in).
- **Context:** Process context only (it’s a sleeping lock). For short read-heavy sections in **IRQ context** the kernel has `rwlock_t` (spin-based, no sleep).

---

## Seqlock (`seqlock_t`)

- **Behavior:** **Writers** never block — they just increment a sequence counter and write. **Readers** read the sequence number, copy the data, then check the sequence again; if it changed (a write happened), they **retry**. So writers are never delayed by readers.
- **Use when:** Writes are **rare**, reads are frequent, and you need **writers to never wait** (e.g. a high-priority sensor thread updating a timestamp; readers can retry).
- **Limitation:** Do **not** use for data that contains **pointers** that might be freed and reused (reader could dereference a freed pointer before retry detects the change). For pointer-heavy structures use **RCU** instead. Seqlock is for plain data (numbers, structs without pointers).

---

## Completion (`struct completion`)

- **Behavior:** A one-shot “event happened” signal. One thread calls `wait_for_completion()` and blocks; another thread (or an interrupt handler) later calls `complete()` to wake it. No lock protecting shared data — just “wait until X is done.”
- **APIs:** `wait_for_completion`, `wait_for_completion_timeout`, `complete` (wake one) / `complete_all` (wake all); `reinit_completion` to reuse after `complete_all`.
- **Use when:** A single “done” event (DMA finished, kthread started, firmware loaded). Cleaner and less error-prone than using a semaphore for one-shot signaling.

---

## lockdep — Catch Deadlocks and Bad Lock Use

- **What it does:** The kernel’s **lock dependency tracker**. It records the order in which locks are taken and detects **potential deadlocks** (e.g. CPU 0 holds A and wants B, CPU 1 holds B and wants A) and **invalid use** (e.g. taking a sleeping lock in interrupt context). It can report these the *first* time a bad ordering is seen, before an actual hang happens.
- **Enable:** `CONFIG_PROVE_LOCKING=y`, `CONFIG_LOCK_STAT=y`. Use during development; disable in production (adds overhead).
- **Output:** Warnings in `dmesg` with full stack traces. `cat /proc/lock_stat` shows per-lock contention statistics.

---

# Summary Tables

## Preemption / RT

| Config | Typical worst-case | Best for |
|--------|---------------------|----------|
| PREEMPT_NONE | >1 ms | Servers |
| PREEMPT_VOLUNTARY | ~500 µs | General |
| PREEMPT | ~100–200 µs | Desktop |
| PREEMPT_RT | <50 µs | Robotics, AV, motor control |
| QNX (RTOS) | <10 µs | Hard RT, safety-certified |

## Synchronization Primitives

**Quick “when to use what”:** Very short section + IRQ? → spinlock. Longer section, process only? → mutex. Many readers, few writers? → rw_semaphore. Writer must never wait? → seqlock. One-shot “event done”? → completion.

| Primitive | Context | Blocks? | Priority inheritance? | Best for |
|-----------|---------|--------|------------------------|----------|
| spinlock_t | Process + IRQ | No (spin) | No | Very short CS, IRQ-shared data |
| raw_spinlock_t | Process + IRQ | No (spin) | No | Hardware-critical, must not sleep |
| mutex | Process | Yes | No | Longer CS in process context |
| rtmutex | Process | Yes | Yes | RT, PREEMPT_RT |
| rw_semaphore | Process | Yes | No | Read-heavy, longer CS |
| seqlock_t | Process + IRQ (writer) | Writer no; reader retries | No | Rare writes, frequent reads |
| completion | Process | Yes | No | One-shot event |

## CPU / Isolation Mechanisms

| Mechanism | Scope | When | Tool / where |
|-----------|--------|------|----------------|
| sched_setaffinity | Per task | Runtime | taskset |
| isolcpus | Per CPU | Boot | Kernel cmdline |
| nohz_full, rcu_nocbs, irqaffinity | Per CPU | Boot | Kernel cmdline |
| cpuset cgroup | Per cgroup | Runtime | cgset, Kubernetes |
| cpufreq performance | Per CPU | Runtime | cpupower |
| numactl | Per process / node | Per run | numactl |

---

# AI Hardware / Edge Relevance

- **PREEMPT_RT** is used where control loops (e.g. openpilot `controlsd`) must complete within a hard window (e.g. 10 ms); cyclictest histograms feed into timing analysis for safety (e.g. ISO 26262 ASIL-B).
- **isolcpus + nohz_full** on Jetson Orin (e.g. cores 4–11 for inference, 0–3 for OS) keeps OS jitter out of the inference latency budget.
- **mlockall** and pre-faulting are standard for RT inference; a single major page fault can add 1–10 ms and break deadlines.
- **hwlatdetect** during bring-up finds SMI-induced latency; fixing it often requires firmware/BIOS, not just kernel tuning.
- **CPU affinity + isolcpus** for ROS2 RT callback groups and inference threads give deterministic placement and stable WCET.
- **NUMA binding** (`numactl`, cpuset mems) keeps inference and GPU on the same socket and avoids cross-NUMA penalty.
- **Intel CAT** can reserve LLC ways for inference so OS activity doesn’t evict model weights and cause latency spikes.
- **Spinlocks** in camera/DMA ISRs protect frame indices (short, no sleep); **rw_semaphore** for weight hot-reload (many readers, one writer); **seqlock** for sensor timestamps (writer never blocked); **completion** for “DMA done” in accelerator drivers.
- **rtmutex** under PREEMPT_RT gives priority inheritance across the kernel, avoiding priority inversion in AV/robotics stacks.
- **lockdep** during driver development catches lock-order and interrupt-context bugs in camera, ISP, and DMA code before deployment.

---

*Combines Lectures L7, L8, L9 (Real-Time Linux, Multi-Core Scheduling & isolcpus, Synchronization).*
