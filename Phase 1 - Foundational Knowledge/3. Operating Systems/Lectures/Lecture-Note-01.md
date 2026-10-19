# Lecture Note 01: Kernel Architecture and Processes

Consolidates [Lecture 1](Lecture-01.md) and [Lecture 2](Lecture-02.md). Use this as retrieval practice after reading the lectures.

## Trace a request through protection boundaries

A process asks for kernel-managed operations through a syscall ABI. The kernel validates and establishes access, but subsequent access to an authorized mapping may happen without another syscall. CPU user/kernel privilege is distinct from an IOMMU's device-DMA protection. Neither stops an arbitrary buggy kernel driver from writing privileged CPU memory.

Linux uses a monolithic kernel with loadable modules. A module runs as kernel code; it is not an isolated process. A BSP couples a particular kernel/configuration with firmware, modules, hardware descriptions, and userspace components. R35/JetPack 5 has a 5.10 baseline; R36/JetPack 6 has a 5.15 baseline. Check exact newer releases and support schedules rather than assuming every LTS lasts six years. Yocto selects a kernel through recipes and machine/BSP configuration.

**Question:** Two processes map address `0x100000`. Do they share a buffer?

**Answer:** The address alone cannot tell you. Each address space has its own mapping; backing pages could differ or be deliberately shared. `/proc/PID/maps` describes virtual mappings, not a portable physical/DMA address.

## Distinguish task, thread group, and shared resources

Each schedulable thread has a `task_struct` and TID. Threads in a thread group share a TGID, which `getpid()` reports in their PID namespace. Their stacks/register state differ while resources such as the address space and descriptor table are commonly shared. `CLONE_VM` alone does not establish all POSIX-thread semantics; thread-group/TLS/exit bookkeeping also matters. Use pthreads for ordinary application threads.

Task state answers whether work is running/runnable or waiting. A wakeup makes a task runnable; another task or interrupt can still delay its execution. D state identifies a class of kernel wait, not proof of a hung camera. Blocking V4L2 dequeue is not universally an uninterruptible wait.

**Question:** Does changing affinity reserve a CPU?

**Answer:** No. It restricts eligible CPUs. Other tasks, IRQs and shared-memory/cache traffic can still interfere. Inspect effective masks and the actual resource policy.

## Follow process lifetime

`fork()` creates a new process with copy-on-write behavior for appropriate private mappings. It still incurs metadata and page-table work. Shared mappings retain their sharing contract. `execve()` replaces the program image; descriptor `FD_CLOEXEC` flags control which fds close. A parent/subreaper uses `waitpid()` or a related interface to collect exit status.

A zombie no longer runs or retains its normal address space, but kernel bookkeeping remains. Orphans can be reparented to a subreaper or namespace init; not necessarily the host's systemd. In a multithreaded process, the forked child has restricted safe operations before exec.

**Question:** Why can fork followed immediately by exec be cheaper than copying every payload page?

**Answer:** CoW defers private data copies, and exec replaces the old mappings before most writes occur. That avoids eager copying; it does not make fork constant-time.

## Inspect and explain

Read `/proc/self/status`, `/proc/self/maps`, `/proc/self/cgroup`, and `/proc/self/ns`. Identify a mapping, eligible CPU list, namespace identity, and cgroup membership. Different container/host views are expected.

Run the [SPSC lab](../labs/SPSC-Queue.md). Explain why its producer and consumer have separate execution state but share the queue, and why sharing memory requires an ownership/order protocol. Record kernel and compiler versions alongside results.
