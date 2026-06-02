# Lecture 2: Processes, task_struct & the Linux Process Model

## Overview

A running AI system is not a single program — it is a collection of competing processes that must share a CPU, memory, and hardware devices without interfering with each other. The core challenge this lecture addresses is: how does the Linux kernel track every running program and safely multiplex the hardware among them? The mental model to carry forward is that every running entity in Linux — whether a process, a thread, or a kernel worker — is represented by one structure: `task_struct`. Understanding this structure is understanding how the kernel sees your code. For an AI hardware engineer, this matters because scheduler class, CPU affinity, cgroup membership, and memory layout are all fields in `task_struct`, and tuning inference pipeline performance means knowing which knobs map to which fields.

---

## The Process Abstraction

A **process** is a program in execution. It combines three orthogonal components:

- **Virtual CPU**: register state (PC, SP, general-purpose registers) saved in `task_struct` during preemption
- **Virtual memory**: address space — text, data, heap, stack, and memory-mapped regions, described by `mm_struct`
- **Resources**: file descriptor table, signal handlers, sockets, cgroup membership — all reachable from `task_struct`

**Threads** are processes that share `mm_struct` and `files_struct` but have independent stacks and register state. Linux makes **no kernel distinction** between "process" and "thread" — both are represented by `task_struct`.

> **Key Insight:** Linux has no separate "thread" concept at the kernel level. A thread is simply a task that shares its `mm_struct` (address space) with another task. This design simplifies the scheduler but means every thread has its own `task_struct`, its own PID (visible via `gettid()`), and its own scheduler entity. When you pin CPU affinity for a thread, you are writing to that thread's `task_struct.cpus_mask`.

---

## task_struct Key Fields

`task_struct` is defined in `include/linux/sched.h`. It is large (~5 KB); only the fields relevant to AI/embedded work are listed here.

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
| `pid` | `pid_t` | Process ID — unique per thread in the system |
| `tgid` | `pid_t` | Thread group ID — shared across all threads; returned by `getpid()` |
| `state` | `unsigned int` | Current run state (TASK_RUNNING, TASK_INTERRUPTIBLE, etc.) |
| `mm` | `struct mm_struct *` | Virtual memory descriptor; NULL for kernel threads |
| `fs` | `struct fs_struct *` | Filesystem root and working directory |
| `files` | `struct files_struct *` | Open file descriptor table |
| `signal` | `struct signal_struct *` | Signal handlers, pending signals, process group |
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
| Uninterruptible sleep | `TASK_UNINTERRUPTIBLE` | No | D | DMA wait, kernel I/O path (V4L2, NVMe) |
| Killable | `TASK_KILLABLE` | SIGKILL only | D | Uninterruptible but yields to kill |
| Stopped | `__TASK_STOPPED` | SIGCONT | T | SIGSTOP or debugger attach |
| Zombie | `EXIT_ZOMBIE` | — | Z | Exited; awaiting parent `wait()` call |
| Dead | `TASK_DEAD` | — | — | Fully reclaimed after parent reaps |

`D` state in `ps` indicates a process **blocked inside a kernel I/O path**. Persistent `D` state is a **driver hang indicator** — common during V4L2 buffer dequeue failures or NVMe timeout.

```
Process State Machine
                   ┌───────────────────────────────┐
                   │                               │
           schedule()                      preempted / slice expires
                   │                               │
                   ▼                               │
           ┌─────────────┐     blocks on I/O   ┌──┴──────────┐
  fork() → │ TASK_RUNNING│ ─────────────────→  │ TASK_INTER- │
  exec()   │  (runnable) │ ←─────────────────  │  RUPTIBLE   │
           └──────┬──────┘    signal / event   └─────────────┘
                  │                                    │
        kernel DMA/I/O path                      SIGKILL only
                  │                                    ▼
                  ▼                          ┌──────────────────┐
           ┌────────────┐                   │  TASK_KILLABLE   │
           │  TASK_UN-  │                   └──────────────────┘
           │INTERRUPTIBLE│
           └──────┬──────┘
                  │
               exit()
                  ▼
           ┌────────────┐    parent wait()   ┌──────────┐
           │EXIT_ZOMBIE │ ─────────────────→ │TASK_DEAD │
           └────────────┘                   └──────────┘
```

> **Key Insight:** `TASK_UNINTERRUPTIBLE` exists because some kernel operations — particularly DMA transfers and hardware I/O — cannot be safely interrupted mid-way. If a process waiting on `VIDIOC_DQBUF` (V4L2 dequeue buffer) could be killed at any point, the DMA engine might write into freed memory. The `D` state is the kernel saying "I'm in the middle of a hardware operation; please wait." A persistent `D` state means the hardware never completed its operation.

> **Common Pitfall:** A zombie process (`Z` in `ps`) is not a bug in the child — it is a bug in the parent. The child has exited and freed its memory, but the kernel keeps a minimal `task_struct` entry until the parent calls `wait()` to collect the exit code. If a parent process never calls `wait()`, zombies accumulate and eventually exhaust the PID namespace. In openpilot, process supervisors must reap all child processes.

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
3. **Pages marked read-only**: the kernel marks all shared pages read-only in both parent and child page tables.
4. **Child returns 0, parent returns child PID**: both resume execution from the instruction after `fork()`.
5. **On first write**: a page fault fires. The kernel allocates a new physical page, copies the content, and updates only the writing task's page table. This is the actual "copy" — deferred until necessary.
6. **Code pages are never copied**: read-only text segments (the program's executable code) are genuinely shared forever, never duplicated.

CoW makes `fork()` **fast even for large processes**. openpilot's multi-process architecture relies on this: `camerad`, `modeld`, `plannerd`, and `controlsd` each fork from a common base without duplicating megabytes of shared library code.

### exec() and wait()

`execve()` **replaces the current address space** with a new ELF binary. File descriptors without `O_CLOEXEC` survive across exec. `waitpid()` reaps a **zombie child**, reclaiming its `task_struct`. Without `wait()`, zombies accumulate and eventually exhaust PID space.

If a parent exits before reaping, **orphan children** are reparented to PID 1 (systemd), which calls `wait()` internally.

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

`clone()` is the underlying syscall behind **both** `fork()` and `pthread_create()`. The **flags argument** determines what the new task shares with its parent.

| Flag | Effect |
|---|---|
| `CLONE_VM` | Share `mm_struct` — both tasks use the same address space (thread) |
| `CLONE_FILES` | Share open file descriptor table |
| `CLONE_SIGHAND` | Share signal handlers |
| `CLONE_NEWPID` | New PID namespace — child is PID 1 inside it |
| `CLONE_NEWNET` | New network namespace — isolated interface/routing table |
| `CLONE_NEWNS` | New mount namespace — isolated filesystem view |

A thread is simply a task created with `CLONE_VM | CLONE_FILES | CLONE_SIGHAND`. `getpid()` returns `tgid` (same for all threads in a process); `gettid()` returns the unique per-thread `pid`.

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
ls -la /proc/[pid]/ns/          # inspect namespace membership of a running process
unshare --pid --fork bash       # launch shell in new PID namespace
```

GPU device files (`/dev/nvidia0`, `/dev/nvhost-ctrl`) must be bind-mounted into the container's mount namespace for CUDA to initialize inside containers.

> **Common Pitfall:** When running TensorRT or CUDA inside a Docker container, the container has its own mount namespace. The NVIDIA runtime must bind-mount `/dev/nvidia*` and `/dev/nvhost-*` into the container. If this fails silently, CUDA will report "no devices found" even though the host can see the GPU. Always check `docker run --gpus all` or the NVIDIA container runtime configuration before chasing a CUDA driver bug.

Now that we understand how processes are created and isolated, let's look at how the kernel limits what resources they can consume — cgroups.

---

## cgroups v2: Resource Control

**Unified hierarchy** at `/sys/fs/cgroup/`. All controllers (cpu, memory, io, cpuset) attach to the same hierarchy tree.

| Controller | Key file | Example value | Effect |
|---|---|---|---|
| cpu | `cpu.max` | `50000 100000` | 50% of one CPU (quota µs / period µs) |
| cpuset | `cpuset.cpus` | `0-3` | Restrict to cores 0–3 |
| cpuset | `cpuset.mems` | `0` | Restrict to NUMA node 0 |
| memory | `memory.max` | `4G` | OOM-kill if exceeded |
| memory | `memory.swap.max` | `0` | Disable swap for this group |
| io | `io.max` | `8:0 rbps=104857600` | 100 MB/s read on device 8:0 |
| pids | `pids.max` | `512` | Limit fork bombs in untrusted containers |

```bash
cat /proc/[pid]/cgroup              # cgroup membership path for a process
cat /sys/fs/cgroup/[path]/cpu.stat  # throttled_usec, nr_throttled — detect throttling
```

Kubernetes uses **cgroup v2** to enforce CPU and memory limits on inference pods. A pod with `cpu.max = 200000 1000000` (20% of one core) will have `modeld` **throttled** if it exceeds that budget.

> **Key Insight:** `cpu.stat`'s `throttled_usec` field is the smoking gun for cgroup-induced latency. If your inference pod shows consistent 2–3 ms latency spikes and `throttled_usec` is climbing, the Kubernetes CPU limit is the bottleneck — not the model, not the GPU, not the scheduler. This is the first file to check after `perf` and `bpftrace` show CPU stalls in the inference thread.

> **Common Pitfall:** Setting `cpuset.cpus` without also setting `cpuset.mems` on a NUMA system can lead to memory being allocated from the wrong NUMA node. This causes cross-node memory traffic that adds ~100 ns per cache line miss. Always pair CPU affinity with NUMA memory node pinning for latency-sensitive inference workloads on multi-socket servers.

---

## Context Switch Mechanics

Now that we understand how the kernel tracks tasks and their resources, let's look at the operation that switches execution between them — the **context switch**.

`context_switch()` is in `kernel/sched/core.c`. It performs two distinct operations:

1. **`switch_mm_irqs_off()`** — install the new process's **virtual address space**. On x86 this writes CR3 (the page table base register); on ARM64 it writes TTBR0_EL1. This is the step that makes the new process's memory visible and hides the old process's memory. Every memory access after this point goes through the new page table.

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

**TLB cost**: ARM64 uses **ASID-tagged TLBs** — switching between tasks with valid ASIDs avoids a full TLB flush. x86 uses **PCID** for the same purpose. Context switch overhead: 1–10 µs depending on cache state and whether the TLB must be flushed. For a 1 kHz control loop (`controlsd` at 100 Hz CAN output), scheduler jitter must stay well below 1 ms.

> **Key Insight:** The TLB (Translation Lookaside Buffer) is a hardware cache that stores recent virtual-to-physical address translations. Without ASID tags, every context switch would require flushing the TLB entirely — that is, invalidating all cached translations — because the new process has a completely different address space. ASID tags let the hardware distinguish "translation for process A" from "translation for process B," so old entries remain valid and the new process can hit the TLB immediately. This is why ASID exhaustion (when all 256 or 65536 ASID slots fill up) forces a TLB flush and adds latency.

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

> **Common Pitfall:** Under memory pressure, the OOM killer selects the process with the highest `oom_score` to terminate. By default, large-memory processes score highest. On a Jetson running both `modeld` and a data logging service, the OOM killer may terminate `modeld` rather than the logger if `modeld` has a larger RSS. Set `oom_score_adj = -500` on critical inference processes to protect them. Conversely, set `oom_score_adj = +500` on non-critical logging processes so they are killed first.

---

## Summary

| State | Macro | Wakeable? | Example cause |
|---|---|---|---|
| Running / runnable | `TASK_RUNNING` | — | On CPU or waiting on run queue |
| Interruptible sleep | `TASK_INTERRUPTIBLE` | Yes (signal) | Blocked on `read()`, `epoll_wait()` |
| Uninterruptible sleep | `TASK_UNINTERRUPTIBLE` | No | DMA wait, `VIDIOC_DQBUF` in driver |
| Killable | `TASK_KILLABLE` | SIGKILL only | NFS soft mount wait |
| Stopped | `__TASK_STOPPED` | SIGCONT | Debugger, SIGSTOP |
| Zombie | `EXIT_ZOMBIE` | — | Awaiting parent `waitpid()` |

### Conceptual Review

- **Why does Linux use a single `task_struct` for both processes and threads?** Simplicity and consistency. The scheduler, OOM killer, cgroup accounting, and CPU affinity mechanisms all operate on `task_struct` without needing special cases for threads. The distinction between process and thread is entirely in which fields are shared (`mm`, `files`) via the `clone()` flags.
- **What is a zombie process and why does it exist?** A zombie is a process that has called `exit()` but whose parent has not yet called `wait()`. The kernel keeps a minimal `task_struct` so the parent can retrieve the child's exit status. Zombies consume a PID slot but no CPU or memory. They accumulate when a parent fails to reap its children.
- **Why does `fork()` use Copy-on-Write instead of immediately copying memory?** Copying the entire address space at `fork()` time would be prohibitively slow for large processes. Most `fork()+exec()` pairs never write to the parent's pages at all — `execve()` replaces the address space immediately. CoW defers the copy cost to the moment it is actually needed.
- **What does `TASK_UNINTERRUPTIBLE` mean in practice?** The process is blocked inside a kernel code path (typically a hardware I/O operation) that cannot be safely interrupted. You cannot kill a process in this state with SIGKILL — only when the kernel I/O path completes (or times out) will the process become killable. Persistent `D` state means the hardware is hung.
- **How does `clone()` relate to `fork()` and `pthread_create()`?** Both `fork()` and `pthread_create()` are implemented in terms of the `clone()` syscall. `fork()` calls `clone()` with no sharing flags (new `mm`, new `files`). `pthread_create()` calls `clone()` with `CLONE_VM | CLONE_FILES | CLONE_SIGHAND` (shared address space, shared file descriptors, shared signal handlers).
- **What is CPU affinity and why does it matter for inference?** `task_struct.cpus_mask` is a bitmask of CPUs the task is allowed to run on. Pinning `modeld` to the big cluster (e.g., Cortex-A78AE on Orin) prevents the scheduler from migrating it to a LITTLE core mid-inference. Migration causes cache invalidation and pipeline stalls; affinity eliminates this variability.

---

## AI Hardware Connection

- `task_struct.sched_class` determines the scheduler for each task; assigning `modeld` to `SCHED_FIFO` switches it to `rt_sched_class`, preventing CFS/EEVDF jitter from delaying frame processing by up to 5 ms on an untuned system.
- `cpuset.cpus` in cgroup v2 pins inference processes to isolated cores, preventing migration to cores shared with interrupt handlers; on Jetson Orin, the big-cluster Cortex-A78AE cores are typically reserved for `modeld` and `camerad`.
- `cpu.max` throttles background processes (telemetry, logging) to a fixed quota so inference threads retain burst CPU headroom — directly writable at `/sys/fs/cgroup/[pod]/cpu.max` in Kubernetes.
- openpilot's `camerad`, `modeld`, `sensord`, `plannerd`, and `controlsd` run as separate processes; CoW fork semantics give each an independent `mm_struct`, enabling crash isolation without corrupting sibling address spaces.
- `TASK_UNINTERRUPTIBLE` appears in camera and DMA driver code paths — a persistent `D`-state process in `/proc/[pid]/wchan` pointing to `v4l2_dqbuf` or `nvdla_submit` immediately identifies the stalled hardware interface.
- PID namespaces in Kubernetes inference pods isolate service process trees; `/proc/[pid]/cgroup` on the host maps any guest PID to its pod's resource accounting group for OOM investigation.
