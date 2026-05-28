# Operating Systems (Phase 1 §3)

<div class="course-identity operating-systems" markdown="1">
<div class="course-identity__icon">OS</div>
<div markdown="1">
<p class="course-identity__eyebrow">Module 3 · Operating Systems</p>
<p class="course-identity__title">Learn the runtime layer that schedules processes, memory, files, drivers, and devices.</p>
<p class="course-identity__meta">Artifact: OS behavior lab · Measure: syscalls, context switches, memory, contention</p>
</div>
</div>


**Primary source:** [Caltech CS124 Spring 2024](https://users.cms.caltech.edu/~donnie/cs124/lectures/) (Donnie Pinkston).

This section is the **OS theory + Linux-shaped practice** layer of Phase 1. It sits after [**§2 — Computer Architecture**](../2.%20Computer%20Architecture%20and%20Hardware/Guide.md) (you need CPU/memory context) and before [**§4 — C++ and Parallel Computing**](../4.%20C%2B%2B%20and%20Parallel%20Computing/Guide.md) (you will use processes, threads, and VM ideas when writing host and CUDA code).

**Why it matters for AI hardware:** Deployed stacks run on Linux (Jetson, ADAS, servers) or smaller RTOSes. Scheduling, virtual memory, I/O, and the user/kernel boundary are what you tune when chasing latency, debugging drivers, or reasoning about zero-copy camera → GPU paths.

---

## How this folder is laid out

Everything you need is **under this directory**—there is no separate “side” tree next to `Lectures/`.

| Location | Role |
|----------|------|
| **`Lectures/Lecture-NN.md`** | Main notes for **26** topics (see index below). |
| **`Lectures/Lecture-Note-NN.md`** | Optional deeper notes where present. |
| **`Lectures/demos/rt-demo/`** | Small **user-space** C demo tying **Lectures 1–9** to runnable code ([README](Lectures/demos/rt-demo/README.md)). |
| **`Final-Test-Problems.md`** | Five integration problems with explicit lecture cross-links. |

**Study flow:** Read lectures in the thematic order below → use **rt-demo** after you finish **L9** (optional but useful) → attempt **Final-Test-Problems** near the end of the section.

---

## Course snapshot

| Aspect | Details |
|--------|---------|
| **Base curriculum** | Caltech CS124 — concepts, slides, and pacing |
| **Write-ups in this repo** | **26** markdown lectures (including capstone-style **L25** and **L26** extensions aligned with this roadmap) |
| **Prerequisites** | C programming; Phase 1 §2 (architecture / memory hierarchy) |
| **Classic hands-on (external)** | [Pintos](https://web.stanford.edu/class/cs140/projects/pintos/) — Stanford teaching kernel (threads, user programs, VM, files) |

---

## Recommended order (by theme)

Follow this order rather than skipping around; later lectures assume earlier vocabulary.

1. **System shape & boot (L1–L5)** — history, UNIX I/O, traps and structure, microkernels, firmware/boot.
2. **Concurrency in the kernel model (L6–L9)** — processes, threads, interrupt context, synchronization and deadlock.  
   *Optional:* build and run **[rt-demo](Lectures/demos/rt-demo/README.md)** here to connect syscalls, scheduling, affinity, and pthread primitives to real code.
3. **Scheduling (L10–L12)** — advanced sync, CPU scheduling, real-time and Linux schedulers.
4. **User/kernel boundary (L13–L14)** — system calls, signals.
5. **Memory (L15–L20)** — virtual memory, page tables, replacement, thrashing, Pintos VM discussion.
6. **Filesystems (L21–L24)** — allocation, locking, SSDs, Pintos FS design, journaling.
7. **Integration (L25–L26)** — Yocto capstone tie-in; eBPF / observability.

---

<a id="lecture-index"></a>
## Lecture index (all topics)

| # | Topic | Key concepts |
|:-:|-------|----------------|
| [1](Lectures/Lecture-01.md) | Introduction & OS history | Mainframes → time-sharing, virtualization, RTOS |
| [2](Lectures/Lecture-02.md) | OS components & UNIX I/O | Syscalls, kernel/user mode, fds, pipes, shells |
| [3](Lectures/Lecture-03.md) | Traps, interrupts & structure | Traps/faults, preemption, monolithic vs microkernel |
| [4](Lectures/Lecture-04.md) | Microkernels & exokernels | Mach, L4, IPC, hybrid kernels |
| [5](Lectures/Lecture-05.md) | Bootstrap & firmware | BIOS/UEFI, ACPI, chain loading; **Linux kernel lectures** tie-in to DT, modules |
| [6](Lectures/Lecture-06.md) | Process abstraction | States, PCB, context switch, queues |
| [7](Lectures/Lecture-07.md) | Threads | User vs kernel threads, models, Amdahl |
| [8](Lectures/Lecture-08.md) | Kernel stacks & interrupts | Reentrancy, interrupt context, critical sections |
| [9](Lectures/Lecture-09.md) | Synchronization & deadlock | Peterson, spinlocks, semaphores, deadlock |
| [10](Lectures/Lecture-10.md) | Advanced synchronization | RCU, rwlocks, lock granularity |
| [11](Lectures/Lecture-11.md) | Process scheduling | FCFS, RR, SJF, priority, MLQs |
| [12](Lectures/Lecture-12.md) | Real-time & Linux schedulers | EDF, rate-monotonic, CFS |
| [13](Lectures/Lecture-13.md) | System calls | Trap path, arguments, pointer checks |
| [14](Lectures/Lecture-14.md) | UNIX signals | Handlers, masks, sigreturn |
| [15](Lectures/Lecture-15.md) | Virtual memory & MMU | Paging, TLB, demand paging |
| [16](Lectures/Lecture-16.md) | Page tables & COW | PTE bits, fork/vfork |
| [17](Lectures/Lecture-17.md) | Frame tables & replacement | FIFO, optimal, Belady |
| [18](Lectures/Lecture-18.md) | Replacement policies | LRU, clock, working set |
| [19](Lectures/Lecture-19.md) | Pintos VM design | Project-oriented discussion |
| [20](Lectures/Lecture-20.md) | Allocation & thrashing | Global vs local, working set |
| [21](Lectures/Lecture-21.md) | Filesystems | Directories, inodes, allocation |
| [22](Lectures/Lecture-22.md) | File locking & SSDs | flock, FTL, TRIM |
| [23](Lectures/Lecture-23.md) | Pintos file system design | Project-oriented discussion |
| [24](Lectures/Lecture-24.md) | Journaling filesystems | Crash consistency |
| [25](Lectures/Lecture-25.md) | Capstone: custom Linux images | Yocto; pairs with Phase 2 embedded Linux |
| [26](Lectures/Lecture-26.md) | eBPF deep dive | Verifier, maps, CO-RE, XDP, observability |

---

## Practice

1. **[Final-Test-Problems.md](Final-Test-Problems.md)** — five problems (kernel build, PREEMPT_RT, ext4, boot chain, modules + DT); each points to specific lectures.
2. **[Lectures/demos/rt-demo/](Lectures/demos/rt-demo/README.md)** — one program for **L1–L9** (scheduling, affinity, `mlockall`, pthread sync). Requires Linux or WSL.

---

## Resources

- **Slides:** [CS124 lectures (PDF)](https://users.cms.caltech.edu/~donnie/cs124/lectures/)
- **Textbook:** *Operating System Concepts* (Silberschatz, Galvin, Gagne)
- **Kernel narrative:** [Linux Insides](https://0xax.gitbooks.io/linux-insides/)
- **Real fork (openpilot / comma):** [agnos-kernel-sdm845](https://github.com/commaai/agnos-kernel-sdm845), [agnos-builder](https://github.com/commaai/agnos-builder) — [**AGNOS Guide**](../../Phase%205%20-%20Advanced%20Topics%20and%20Specialization/5.%20Autonomous%20Vehicles/2.%20openpilot%20Reference%20Stack/agnos/Guide.md) maps **all 26** lecture topics to kernel paths and fork-specific changes.

---

## AI / embedded relevance

| OS idea | Typical AI / edge use |
|---------|------------------------|
| **Real-time scheduling** | Inference deadlines, sensor pipelines |
| **Virtual memory & DMA** | Weights in RAM, camera → accelerator buffers |
| **fds & I/O** | Cameras, CAN, logging |
| **Process / thread model** | Separate daemons for capture, inference, control |
| **User vs kernel** | Drivers vs userspace ML runtimes |
| **eBPF** | Profiling, scheduler analysis, XDP filtering |
