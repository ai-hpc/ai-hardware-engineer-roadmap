# Operating Systems (Phase 1 §3)

<div class="course-identity operating-systems" markdown="1">
<div class="course-identity__icon">OS</div>
<div markdown="1">
<p class="course-identity__eyebrow">Module 3 · Operating Systems</p>
<p class="course-identity__title">Learn the runtime layer that schedules processes, memory, files, drivers, and devices.</p>
<p class="course-identity__meta">Artifact: OS behavior lab · Measure: syscalls, context switches, memory, contention</p>
</div>
</div>


This course explains the Linux mechanisms between your VLA runtime and the devices it uses. Start after [computer architecture](../2.%20Computer%20Architecture%20and%20Hardware/Guide.md); use its memory hierarchy model to distinguish scheduling delay, CPU page faults, device DMA, and accelerator execution.

The repository contains 26 original roadmap lectures plus eight optional consolidation notes. [Caltech CS124](https://users.cms.caltech.edu/~donnie/cs124/lectures/) and [Stanford Pintos](https://web.stanford.edu/class/cs140/projects/pintos/) are supplementary OS resources; their topic numbering is not the numbering below.

## Study sequence and evidence

| Lectures | Question to answer | Artifact carried into the next group |
|---|---|---|
| 1–5: kernel, processes, interrupts, syscalls, boot | How does an application request a device operation? | Annotated boot handoff diagram; an actual boot/IRQ trace when a target is available |
| 6–11: scheduling and synchronization | Why did a ready task miss its deadline? | Periodic-worker trace with execution, waiting, and contention separated |
| 12–16: virtual memory, allocation, DMA, NUMA | Which address and ownership domain contains each buffer? | Buffer table: CPU virtual address, DMA address, lifetime, synchronization |
| 17–20: drivers and I/O | How is work submitted, completed, and reclaimed? | Device probe map and completion timeline |
| 21–24: storage, OTA, containers, deployed systems | What survives restart, isolation, and resource pressure? | Recoverable image/service configuration and failure log |
| 25–26: Yocto and eBPF integration | Can another engineer reproduce and diagnose the system? | Pinned metadata inputs and a compiled tracer; runtime tracing is a separate milestone |

Use [rt-demo](Lectures/demos/rt-demo/README.md) after L9 and the [native bounded SPSC queue](labs/SPSC-Queue.md) with L10. The queue tests full/empty/wrap behavior, actual threaded transfer, lifetime checks, and an intentional overwrite defect. Start without elevated scheduling privileges; compare load conditions before changing policy. A good latency histogram is evidence for the tested workload, not proof of a hard real-time bound. Record kernel version/configuration because implementation details and available tracing features vary.

The [BPF build lab](labs/BPF-Build-Lab.md) adds an optional native-tool milestone after L26: compile a real object/skeleton/reader, inspect its sections, and test input rejection without attaching to the kernel. It complements the executed SPSC and timer exercises. Follow the [review/evidence matrix](Review-Matrix.md) to see which concepts have native checks and which still need target hardware or privileged runtime validation.

<a id="lecture-index"></a>
## Lecture index

| # | Actual lecture in this repository |
|---|---|
| [1](Lectures/Lecture-01.md) | Modern OS Architecture & the Linux Kernel |
| [2](Lectures/Lecture-02.md) | Processes, task_struct & the Linux Process Model |
| [3](Lectures/Lecture-03.md) | Interrupts, Exceptions & Bottom Halves |
| [4](Lectures/Lecture-04.md) | System Calls, vDSO & eBPF |
| [5](Lectures/Lecture-05.md) | Kernel Modules, Boot Process & Device Tree |
| [6](Lectures/Lecture-06.md) | CPU Scheduling: CFS, EEVDF & Real-Time Classes |
| [7](Lectures/Lecture-07.md) | Real-Time Linux: PREEMPT_RT & Determinism |
| [8](Lectures/Lecture-08.md) | Multi-Core Scheduling, CPU Affinity & isolcpus |
| [9](Lectures/Lecture-09.md) | Synchronization: Spinlocks, Mutexes, RW Locks & Seqlocks |
| [10](Lectures/Lecture-10.md) | Lock-Free Programming: RCU, Atomics & Memory Ordering |
| [11](Lectures/Lecture-11.md) | Deadlock, Priority Inversion & PI Mutexes |
| [12](Lectures/Lecture-12.md) | Virtual Memory & the Linux Memory Model |
| [13](Lectures/Lecture-13.md) | Page Tables, TLBs & Huge Pages |
| [14](Lectures/Lecture-14.md) | Memory Allocation: SLUB, kmalloc & CMA |
| [15](Lectures/Lecture-15.md) | DMA, IOMMU & GPU Memory Management |
| [16](Lectures/Lecture-16.md) | NUMA Topology & HPC Memory Optimization |
| [17](Lectures/Lecture-17.md) | Linux Device Driver Model & Device Tree |
| [18](Lectures/Lecture-18.md) | Character Drivers, Interrupt-Driven I/O & V4L2 |
| [19](Lectures/Lecture-19.md) | Modern I/O: io_uring, DMA-BUF & Zero-Copy Pipelines |
| [20](Lectures/Lecture-20.md) | PCIe, NVMe & GPU Driver Architecture |
| [21](Lectures/Lecture-21.md) | Filesystems: ext4, btrfs, F2FS & overlayfs |
| [22](Lectures/Lecture-22.md) | Embedded Storage: eMMC, UFS, NVMe & OTA Partitioning |
| [23](Lectures/Lecture-23.md) | Containers, cgroups v2 & NVIDIA Container Runtime |
| [24](Lectures/Lecture-24.md) | OS for AI Systems: L4T, openpilot OS & RT Tuning |
| [25](Lectures/Lecture-25.md) | Capstone Project — Custom Linux Images with Yocto |
| [26](Lectures/Lecture-26.md) | eBPF — Programmable Kernel Observability & Networking |

## Integration lab: follow one observation to one action

Model a camera producer, inference worker, and controller as separate stages with bounded queues. Put a monotonic timestamp and sequence number on every item. Deliberately delay the inference worker; show whether the controller receives an old observation, a dropped item, or an explicit timeout. Explain the queue policy and account for each buffer until its final consumer releases it.

A complete submission includes timestamps for capture, dequeue, device submission, device completion, and command publication; p50/p95/p99 latency and deadline misses; and one trace demonstrating the cause of an outlier. State which stages are simulated. This becomes the runtime contract for [embedded Linux](../../Phase%202%20-%20Embedded%20Systems/3.%20Embedded%20Linux/Guide.md) and the host-side baseline for [parallel computing](../4.%20C%2B%2B%20and%20Parallel%20Computing/Guide.md).

## Reference and review

- [Final practice problems with answer key](Final-Test-Problems.md).
- [Linux DMA API guide](https://docs.kernel.org/core-api/dma-api-howto.html): address translation, mapping, ownership, and memory ordering.
- [Linux EEVDF scheduler documentation](https://docs.kernel.org/scheduler/sched-eevdf.html): distinguish fair scheduling from deadline guarantees.
- [AGNOS code-reading guide](../../Phase%205%20-%20Advanced%20Topics%20and%20Specialization/Track%20E%20-%20Autonomous%20Vehicles/2.%20openpilot%20Reference%20Stack/agnos/Guide.md): compare lecture concepts with a real vendor kernel and image build.

Use the [shared completion rubric](../../Learning-Guide.md#completion-rubric) to separate explanation, reproduced experiments, and hardware work still to do.
