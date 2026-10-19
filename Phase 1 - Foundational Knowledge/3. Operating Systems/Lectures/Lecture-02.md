# Lecture 2: Processes, task_struct & the Linux Process Model

## Overview

A running AI system is not a single program — it is a collection of competing processes that must share a CPU, memory, and hardware devices without interfering with each other. The core challenge this lecture addresses is: how does the Linux kernel track every running program and safely multiplex the hardware among them? The mental model to carry forward is that every running entity in Linux — whether a process, a thread, or a kernel worker — is represented by one structure: `task_struct`. Understanding this structure is understanding how the kernel sees your code. For an AI hardware engineer, this matters because scheduler class, CPU affinity, cgroup membership, and memory layout are all fields in `task_struct`, and tuning inference pipeline performance means knowing which knobs map to which fields.

---

## The Process Abstraction

A **process** is a program in execution. It combines three orthogonal components:

- **Virtual CPU**: register state (PC, SP, general-purpose registers) saved in `task_struct` during preemption
- **Virtual memory**: address space — text, data, heap, stack, and memory-mapped regions, described by `mm_struct`
- **Resources**: file descriptor table, signal handlers, sockets, cgroup membership — all reachable from `task_struct`

**Threads** are processes that share `mm_struct` and `files_struct` but have independent stacks and register state. Both use `task_struct`, but Linux also maintains thread groups and explicitly shared resources; sharing an address space alone does not define every POSIX thread behavior.

> **Thread identity:** Each schedulable thread has its own task structure and TID. Threads in one thread group share a TGID, returned by `getpid()` in their PID namespace. Affinity and scheduling policy are set through supported APIs, not by writing internal kernel fields.

---

## task_struct Key Fields

`task_struct` is defined in `include/linux/sched.h`. Its size and layout depend on version, architecture, and configuration; only conceptual fields relevant to AI/embedded work are listed here.

```
task_struct — The kernel's representation of a running task
┌─────────────────────────────────────────────────────────┐
│  pid       — unique thread ID (gettid())                │
│  tgid      — thread group ID; all threads share this    │
│              (getpid() returns tgid)                    │
│  state     — TASK_RUNNING / TASK_INTERRUPTIBLE / etc.   │
├─────────────────────────────────────────────────────────┤
│  mm ──────────────────────────────> mm_struct           │
│                                     (virtual address    │
│                                      space, page table) │
├─────────────────────────────────────────────────────────┤
│  files ────────────────────────────> files_struct       │
│                                     (open FD table;     │
│                                      shared by threads) │
├─────────────────────────────────────────────────────────┤
│  sched_class ──> rt / fair / dl / idle / stop           │
│  se     — CFS/EEVDF entity (vruntime, load weight)      │
│  rt     — RT entity (static priority, time slice)       │
│  dl     — DEADLINE entity (runtime, deadline, period)   │
├─────────────────────────────────────────────────────────┤
│  cgroups ──────────────────────────> css_set            │
│  cpus_mask  — CPU affinity bitmask                      │
└─────────────────────────────────────────────────────────┘
```

| Field | Type | Purpose |
|---|---|---|
| `pid` | `pid_t` | Thread identity; userspace numbering depends on PID namespace |
| `tgid` | `pid_t` | Thread group ID — shared across all threads; returned by `getpid()` |
| `__state` in modern kernels | `unsigned int` | Current run state (TASK_RUNNING, TASK_INTERRUPTIBLE, etc.) |
| `mm` | `struct mm_struct *` | Virtual memory descriptor; NULL for kernel threads |
| `fs` | `struct fs_struct *` | Filesystem root and working directory |
| `files` | `struct files_struct *` | Open file descriptor table |
| `signal` / `sighand` | Shared signal bookkeeping / handler table | Distinct structures with thread-group sharing rules |
| `sched_class` | pointer | Which scheduler class handles this task |
| `se` | `struct sched_entity` | CFS/EEVDF entity: vruntime, load weight, slice |
| `rt` | `struct sched_rt_entity` | RT entity: static priority, time slice |
| `dl` | `struct sched_dl_entity` | DEADLINE entity: runtime, deadline, period budgets |
| `cgroups` | `struct css_set *` | cgroup v2 membership; pointer to CSS set |
| `cpus_mask` | `cpumask_t` | Allowed CPU affinity; set via `sched_setaffinity()` |

---

## Process States

Understanding **process states** is essential for debugging. The state field in `task_struct` tells you exactly what the kernel thinks a process is doing at any moment. This is the information visible in the `ps` command's `STAT` column.

| State | Macro | Wakeable by signal? | `ps` letter | Example cause |
|---|---|---|---|---|
| Running or runnable | `TASK_RUNNING` | — | R | On CPU or on a run queue |
| Interruptible sleep | `TASK_INTERRUPTIBLE` | Yes | S | Waiting for I/O, event, or timer |
| Uninterruptible sleep | `TASK_UNINTERRUPTIBLE` | No | D | Some kernel I/O or resource waits; depends on the actual path |
| Killable | `TASK_KILLABLE` | Fatal signals | D | Uninterruptible but yields to kill |
| Stopped | `__TASK_STOPPED` | SIGCONT | T | SIGSTOP or debugger attach |
| Zombie | `EXIT_ZOMBIE` | — | Z | Exited; awaiting parent `wait()` call |
| Dead task state | `TASK_DEAD` | — | — | Task no longer scheduled; final object reclamation has additional lifetime rules |

`D` is evidence of an uninterruptible or killable wait, not proof of a broken device. Inspect the blocked stack, timeout policy, and dependencies. V4L2 dequeue is not universally uninterruptible; drivers must preserve DMA lifetime even when a userspace wait is interrupted.

```text
runnable ↔ running → interruptible/killable/uninterruptible wait
                        ↓ condition and wakeup rules
                    runnable again
running → exit → zombie status retained → parent/subreaper wait reaps it
```

A task awakened from a wait becomes runnable, not immediately executing. Exiting a wait and stopping a device are separate operations.

> **Reaping:** A short-lived zombie state is normal between child exit and parent collection. A parent that never reaps can accumulate zombies. The child has exited and freed its memory, but the kernel keeps a minimal `task_struct` entry until the parent calls `wait()` to collect the exit code. If a parent process never calls `wait()`, zombies accumulate and eventually exhaust the PID namespace. In openpilot, process supervisors must reap all child processes.

---

## fork / exec / wait

The **fork/exec/wait** trio is the fundamental mechanism for creating new processes in Unix. Understanding this sequence is also key to understanding why openpilot's **multi-process architecture** works efficiently.

### fork() and Copy-on-Write

`fork()` creates a child as a structural copy of the parent. Physical memory is **not** copied immediately — **Copy-on-Write** defers allocation:

```
fork() — Copy-on-Write Memory Model
┌─────────────┐   fork()   ┌─────────────┐
│   Parent    │ ─────────> │    Child    │
│   Process   │            │   Process   │
│             │            │             │
│ page table: │            │ page table: │
│  0x1000 ──────────────────────> [RO]  │  ← same physical page
│  0x2000 ──────────────────────> [RO]  │    marked read-only
│  0x3000 ──────────────────────> [RO]  │
└─────────────┘            └─────────────┘
                                  │
                           write to 0x2000
                                  │
                                  ▼
                           ┌─────────────┐
                           │ PAGE FAULT  │
                           │ kernel      │
                           │ allocates   │
                           │ new page    │
                           │ child 0x2000│
                           │ → new page  │
                           └─────────────┘
```

The sequence when `fork()` is called:

1. **Kernel copies `task_struct`**: a new `task_struct` is allocated and populated from the parent's, with a new PID.
2. **`mm_struct` is duplicated**: the new task gets its own virtual address space descriptor, but the page table entries point to the same physical pages as the parent.
3. **Pages marked read-only**: the uses copy-on-write protection for appropriate private writable mappings. Shared mappings intentionally remain shared, and special mappings have their own fork behavior.
4. **Child returns 0, parent returns child PID**: both resume execution from the instruction after `fork()`.
5. **On first write**: a page fault fires. The kernel allocates a new physical page, copies the content, and updates only the writing task's page table. This is the actual "copy" — deferred until necessary.
6. **Unmodified read-only code can remain shared**: do not generalize this to writable mappings or every special mapping.

CoW avoids eagerly copying all payload pages, but `fork()` is not O(1) in address-space size: VMAs, page-table work, and other resources still cost time. Large mappings, pinned memory, and later writes can be expensive. Do not infer a particular current application process-launch method without checking its pinned source.

### exec() and wait()

`execve()` **replaces the current address space** with a new ELF binary. File descriptors whose `FD_CLOEXEC` flag is clear survive a successful exec; `O_CLOEXEC` sets that descriptor flag when opening. `waitpid()` reaps a **zombie child**, reclaiming its `task_struct`. Without `wait()`, zombies accumulate and eventually exhaust PID space.

If a parent exits before reaping, **orphan children** are reparented to an appropriate child subreaper or PID-namespace init process, which must reap them.

After `fork()` in a multithreaded process, the child should use only async-signal-safe operations before `execve()`. Check exec failure and use `_exit()` on its error path. In the parent, handle `waitpid()` interruption and inspect `WIFEXITED`/`WIFSIGNALED` before interpreting the status.

Illustrative lifecycle: the executable name below is a fictional example, not an openpilot installation path.

```
The fork / exec / wait lifecycle
┌─────────┐
│ Parent  │
│ Process │
└────┬────┘
     │ fork()
     ├──────────────────────────────────┐
     │                                  ▼
     │                           ┌─────────────┐
     │ (continues running)       │    Child    │
     │                           │  (PID = N)  │
     │                           └──────┬──────┘
     │                                  │ execve("/usr/bin/camerad")
     │                                  ▼
     │                           ┌─────────────┐
     │                           │  camerad    │
     │                           │  (new ELF)  │
     │                           └──────┬──────┘
     │                                  │ exit(0)
     │                                  ▼
     │                           ┌─────────────┐
     │ waitpid(N, &status, 0) ←─ │   ZOMBIE    │
     │ (reaps child)             │  (PID = N)  │
     ▼                           └─────────────┘
┌─────────┐
│ Parent  │
│(continues│
└─────────┘
```

---

## clone() and Threads

Linux libc implementations use clone-family interfaces for process/thread creation; the exact choice of `clone`, `clone3`, or architecture-specific paths depends on libc and the platform. The **flags argument** determines what the new task shares with its parent.

| Flag | Effect |
|---|---|
| `CLONE_VM` | Share `mm_struct`; sharing alone does not create the same thread group |
| `CLONE_FILES` | Share open file descriptor table |
| `CLONE_SIGHAND` | Share signal handlers |
| `CLONE_NEWPID` | New PID namespace — child is PID 1 inside it |
| `CLONE_NEWNET` | New network namespace — isolated interface/routing table |
| `CLONE_NEWNS` | New mount namespace — isolated filesystem view |

A POSIX-thread implementation uses additional flags and bookkeeping, including `CLONE_THREAD` for the thread group and TLS/child-TID handling. Use `pthread_create()` unless deliberately implementing a threading runtime. `getpid()` returns `tgid` (same for all threads in a process); `gettid()` returns the unique per-thread `pid`.

> **Key Insight:** The fact that threads and processes are the same structure (`task_struct`) means the scheduler treats them identically. A thread at `SCHED_FIFO` priority 80 will preempt a process at priority 50 just as readily as it preempts another thread at priority 50. CPU affinity, cgroup membership, and scheduling class are per-`task_struct` — meaning you can set different scheduling policies for different threads within the same process.

---

## Linux Namespaces

**Namespaces** partition kernel resources so a set of processes sees an isolated view. They are the foundation of containers.

| Namespace | Isolates | Container use |
|---|---|---|
| `pid` | Process ID numbering | Container init appears as PID 1 |
| `mnt` | Filesystem mount tree | Container-private root filesystem |
| `net` | Network interfaces, routes, iptables | Per-container networking |
| `uts` | Hostname and domain name | Container-specific hostname |
| `ipc` | System V IPC, POSIX message queues | IPC isolation between containers |
| `user` | UID/GID mappings | Rootless containers |
| `cgroup` | cgroup root view | Nested cgroup hierarchies |
| `time` | Clock offsets | Time namespace per container |

```bash
ls -l /proc/self/ns            # inspect the current process; read-only
```

GPU access in a container needs the correct devices, permissions, and compatible driver/runtime libraries. Device node names differ by platform and BSP; use the supported container toolkit configuration. See [Lecture 23](Lecture-23.md).

Now that we understand how processes are created and isolated, let's look at how the kernel limits what resources they can consume — cgroups.

---

## cgroups v2: Resource Control

**Unified hierarchy** at `/sys/fs/cgroup/`. All controllers (cpu, memory, io, cpuset) attach to the same hierarchy tree.

| Controller | Key file | Example value | Effect |
|---|---|---|---|
| cpu | `cpu.max` | `50000 100000` | 50% of one CPU (quota µs / period µs) |
| cpuset | `cpuset.cpus` | `0-3` | Restrict to cores 0–3 |
| cpuset | `cpuset.mems` | `0` | Restrict to NUMA node 0 |
| memory | `memory.max` | `4G` | Reclaim/limit enforcement; unresolved pressure can invoke cgroup OOM |
| memory | `memory.swap.max` | `0` | Disable swap for this group |
| io | `io.max` | `8:0 rbps=104857600` | 100 MiB/s read cap for that actual device, when supported |
| pids | `pids.max` | `512` | Limit fork bombs in untrusted containers |

```bash
cat /proc/[pid]/cgroup              # cgroup membership path for a process
cat /sys/fs/cgroup/[path]/cpu.stat  # throttled_usec, nr_throttled — detect throttling
```

Kubernetes uses **cgroup v2** to enforce CPU and memory limits on inference pods. A pod with `cpu.max = 200000 1000000` (20% of one core) will have `modeld` **throttled** if it exceeds that budget.

Compare changes in `cpu.stat` with the workload timeline. Throttling counters show budget exhaustion; they do not prove that a particular latency spike has no other cause. CPU and memory placement must be evaluated using effective masks and the actual NUMA topology. An empty requested cpuset can inherit an ancestor's effective mask; do not assume a fixed remote-memory latency.

---

## Context Switch Mechanics

Now that we understand how the kernel tracks tasks and their resources, let's look at the operation that switches execution between them — the **context switch**.

`context_switch()` is in `kernel/sched/core.c`. It performs two distinct operations:

1. **`switch_mm_irqs_off()`** — install the new process's **virtual address space**. Address-space changes can involve CR3 on x86 or TTBR registers on ARM64; switching between threads sharing an mm can avoid replacing the address space. Exact paths include architecture-specific optimizations. This is the step that makes the new process's memory visible and hides the old process's memory. Every memory access after this point goes through the new page table.

2. **`switch_to()`** — save the outgoing task's callee-saved registers (rbx, rbp, r12–r15 on x86; x19–x28, fp, lr on ARM64) and stack pointer to its `task_struct`, then restore the incoming task's saved registers. When `switch_to()` returns, the CPU is executing in the context of the new task.

3. **Resume** — the new task resumes at the exact instruction where it was last preempted, as if nothing happened. Its register state, stack, and virtual memory are all restored.

```
Context Switch Timeline
┌──────────────┐                    ┌──────────────┐
│  Task A      │                    │  Task B      │
│  (running)   │                    │  (waiting)   │
└──────┬───────┘                    └──────┬───────┘
       │                                   │
       │ scheduler tick / block            │
       │                                   │
       ▼                                   │
┌─────────────────────────────────┐        │
│  context_switch(A → B)          │        │
│  1. switch_mm: write TTBR0/CR3  │        │
│  2. switch_to: save A's regs    │        │
│               restore B's regs  │        │
└─────────────────────┬───────────┘        │
                      │                    │
                      └───────────────────►│
                                           │  (Task B resumes here)
                                           ▼
                                    ┌──────────────┐
                                    │  Task B      │
                                    │  (running)   │
                                    └──────────────┘
```

**TLB cost:** ASID/PCID tags can preserve translations across address-space switches, subject to invalidation and rollover rules. Cache footprint, mitigations, CPU migration, and run-queue contention also affect measured cost. There is no universal 1–10 µs context-switch bound. Distinguish a 1 kHz loop's 1 ms period from a 100 Hz loop's 10 ms period. Use the [RT demo](demos/rt-demo/README.md) for actual wakeup measurements with an explicit observation boundary.

---

## /proc/[pid]/ Runtime Inspection

| Path | Contents |
|---|---|
| `/proc/[pid]/maps` | Virtual memory regions: address, permissions, backing file |
| `/proc/[pid]/smaps` | Per-region RSS and PSS; identifies memory waste and sharing |
| `/proc/[pid]/status` | State, VmRSS, threads, capability sets |
| `/proc/[pid]/fd/` | Symlinks to open files, sockets, V4L2 device nodes |
| `/proc/[pid]/sched` | CFS/EEVDF: vruntime, nr_voluntary_switches, se.load.weight |
| `/proc/[pid]/wchan` | Kernel function where task is currently sleeping |
| `/proc/[pid]/cgroup` | cgroup v2 membership path |
| `/proc/[pid]/oom_score` | OOM killer score; higher value killed first under memory pressure |
| `/proc/[pid]/oom_score_adj` | Writable: tune OOM priority (-1000 = never kill, +1000 = kill first) |

OOM selection depends on the eligible scope and policy. `oom_score_adj` changes preference; it does not reserve memory or guarantee a service survives every failure. Prefer measured memory budgets and explicit overload behavior to blindly protecting an inference process while exhausting the rest of the system.

---

## Summary

| State | Macro | Wakeable? | Example cause |
|---|---|---|---|
| Running / runnable | `TASK_RUNNING` | — | On CPU or waiting on run queue |
| Interruptible sleep | `TASK_INTERRUPTIBLE` | Yes (signal) | Blocked on `read()`, `epoll_wait()` |
| Uninterruptible sleep | `TASK_UNINTERRUPTIBLE` | No | Some kernel I/O or resource waits |
| Killable | `TASK_KILLABLE` | Fatal signals | A killable kernel wait |
| Stopped | `__TASK_STOPPED` | SIGCONT | Debugger, SIGSTOP |
| Zombie | `EXIT_ZOMBIE` | — | Awaiting parent `waitpid()` |

### Conceptual review and practice

**Why use CoW?** It avoids copying private payload pages until a write requires it. Fork still incurs metadata/page-table work; subsequent writes can allocate and copy.

**Is a zombie using no memory at all?** No. Its address space has been released, but exit status and kernel bookkeeping remain until reaped. It does not execute application work.

**Does affinity create a dedicated CPU?** No. It limits eligible logical CPUs. Other tasks, interrupts, shared caches, and memory traffic may still interfere. Orin's Cortex-A78AE cores should not be described as a big/LITTLE pair of unlike CPU clusters.

**What does D state prove?** Only a class of kernel wait at the observation instant. Trace the blocked path and its dependencies before diagnosing hardware failure.

Read `/proc/self/status`, `/proc/self/cgroup`, and `/proc/self/maps`. Then run the native [SPSC lab](../labs/SPSC-Queue.md), which creates real producer/consumer threads. Explain which resources those threads share and which state—registers, stack, scheduling identity—belongs to each thread.
