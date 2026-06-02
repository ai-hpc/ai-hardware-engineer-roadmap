# Lecture Note 01 (L1, L2): OS Architecture & the Linux Kernel; Processes, task_struct & the Linux Process Model

**Combines:** Lecture L1 (Modern OS Architecture & the Linux Kernel) and Lecture L2 (Processes, task_struct & the Linux Process Model).

---

## How This Note Is Organized

1. **Part 1 (L1) — OS & kernel:** OS roles; privilege levels (x86 Rings, ARM EL0–EL3); monolithic kernel; versioning; /proc, /sys; kernel modules; source layout; AI platforms.
2. **Part 2 (L2) — Processes & task_struct:** Process abstraction; task_struct and key fields; process states; fork/exec/wait and COW; clone() and threads; namespaces; cgroups v2; context switch; /proc/[pid] inspection.

---

# Part 1 (L1): Modern OS Architecture & the Linux Kernel

**Context:** The OS is the **trusted referee**: it owns hardware, enforces protection, and provides abstractions. User code runs at **Ring 3 / EL0**; kernel at **Ring 0 / EL1**. On Jetson, TensorRT runs at EL0 and reaches hardware only via the kernel.

---

## OS: Three Roles

| Role | Meaning |
|------|--------|
| Resource manager | CPU time, memory, I/O, network across processes |
| Abstraction layer | Uniform interfaces (files, sockets, VM) over hardware |
| Protection boundary | Isolates processes and kernel; enforced in hardware |

---

## Privilege Levels

**x86:** Ring 0 = kernel (all instructions); Ring 3 = user (privileged instruction → fault). Switch: SYSCALL → Ring 0; SYSRET → Ring 3.

**ARM64 (AArch64):** EL0 = user (TensorRT, ROS2); EL1 = kernel (MMU, drivers); EL2 = hypervisor (KVM); EL3 = secure monitor (TrustZone, PSCI). Jetson: kernel at EL1; NVIDIA firmware at EL3.

---

## Linux Kernel: Monolithic + Modules

**Monolithic:** core and drivers in one address space at Ring 0/EL1; fast in-kernel calls; a **crashing driver can panic the system**. Contrast: **microkernels** (QNX) run drivers in separate processes. **Loadable modules** (`.ko`) add drivers at runtime without rebuilding the kernel.

**Subsystems:** kernel/ (scheduler, signals), mm/ (memory), drivers/ (GPU, V4L2, NVMe, PCIe), fs/ (VFS), net/, arch/. **Versioning:** mainline → stable → LTS (2–6 years). AI platforms pin to LTS (e.g. Jetson 5.10/6.1, Yocto 5.15/6.6) for BSP and driver stability.

---

## /proc and /sys

**/proc:** Virtual filesystem; reads invoke kernel code. Examples: cpuinfo, meminfo, interrupts, cmdline; per-process: maps, status, fd, wchan. **/sys (sysfs):** Device/bus hierarchy; class (net, thermal, gpio), cgroup, firmware/devicetree. On Jetson, thermal zones and GPU state are in /sys.

---

## Kernel Modules

`insmod`/`rmmod`/`modprobe`; `lsmod`, `modinfo`. Module init/exit; `MODULE_DEVICE_TABLE(of, ...)` for udev auto-load on Device Tree match. Compile against running kernel headers; DKMS for out-of-tree (e.g. NVIDIA, FPGA).

---

# Part 2 (L2): Processes, task_struct & the Linux Process Model

**Context:** Every runnable entity is a `task_struct`. **Threads are tasks** that share `mm_struct` and `files_struct`. Scheduler class, affinity, cgroups live in task_struct; **tuning inference = tuning these fields**.

---

## Process Abstraction

Process = program in execution: **virtual CPU** (registers in task_struct), **virtual memory** (mm_struct), **resources** (files, signals, cgroups). Thread = task with shared mm and files; kernel does not distinguish “thread” vs “process” beyond clone flags.

---

## task_struct Key Fields

`pid` (thread ID, gettid()), `tgid` (process ID, getpid()), `state`, `mm`, `files`, `sched_class`, `se` (CFS/EEVDF), `rt` (RT), `dl` (DEADLINE), `cgroups`, `cpus_mask`. Scheduler and affinity act on this structure.

---

## Process States

TASK_RUNNING (R), TASK_INTERRUPTIBLE (S), TASK_UNINTERRUPTIBLE (D — e.g. DMA, VIDIOC_DQBUF), TASK_KILLABLE, STOPPED (T), EXIT_ZOMBIE (Z). **Persistent D = driver/hardware hang**. **Zombie** = parent has not called wait().

---

## fork / exec / wait; COW

`fork()` creates child; physical pages not copied — **Copy-on-Write**: pages shared read-only until first write, then copy. `execve()` replaces address space. `waitpid()` reaps zombie. CoW makes fork() O(1) for address space size; openpilot multi-process relies on it.

---

## clone() and Threads

`clone(CLONE_VM | CLONE_FILES | CLONE_SIGHAND)` = **thread** (shared mm, files). `getpid()` = tgid; `gettid()` = per-thread pid. **Scheduler treats all tasks alike**; per-task affinity and scheduling class.

---

## Namespaces and cgroups v2

**Namespaces:** pid, mnt, net, uts, ipc, user, cgroup, time — isolate view of resources; basis of containers. **cgroups v2** (unified at /sys/fs/cgroup/): cpu.max, cpuset.cpus/mems, memory.max, io.max, pids.max. Kubernetes uses them for pod limits; cpu.stat throttled_usec indicates CPU throttling.

---

## Context Switch

`switch_mm` (install new page table, CR3/TTBR0); `switch_to` (save/restore registers). **ASID/PCID avoid full TLB flush**. Cost **~1–10 µs**. /proc/[pid]/sched, wchan, cgroup, oom_score for inspection.

---

## Summary

| L1 | OS roles; Rings/EL; monolithic kernel; /proc, /sys; modules; LTS. |
| L2 | task_struct; states; fork/exec/wait; COW; clone/threads; namespaces; cgroups; context switch. |

---

## AI Hardware Connection

- L1: L4T/AGNOS/Yocto pin kernel for driver and ABI stability; /proc/interrupts and /sys/thermal for diagnostics; monolithic design means driver crash can panic system — IOMMU and testing matter.
- L2: SCHED_FIFO and cpus_mask for modeld; cgroup cpu.max and throttled_usec for pod throttling; CoW for multi-process; TASK_UNINTERRUPTIBLE in V4L2/NVMe paths; oom_score_adj to protect inference from OOM.

---

*Combines Lectures L1, L2 (OS Architecture & Linux Kernel; Processes, task_struct & Process Model).*
