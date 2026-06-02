# Lecture Note 02 (L3, L4): Interrupts, Exceptions & Bottom Halves; System Calls, vDSO & eBPF

**Combines:** Lecture L3 (Interrupts, Exceptions & Bottom Halves) and Lecture L4 (System Calls, vDSO & eBPF).

---

## How This Note Is Organized

1. **Part 1 (L3) — Interrupts & bottom halves:** Interrupts vs exceptions; GIC/APIC; MSI/MSI-X; top half vs bottom half; softirq, tasklet, workqueue; interrupt flow and latency.
2. **Part 2 (L4) — System calls & vDSO:** Syscall path (x86/ARM64); overhead and mitigations; vDSO for time; key syscalls (mmap, ioctl); eBPF for observability.

---

# Part 1 (L3): Interrupts, Exceptions & Bottom Halves

**Context:** Hardware signals the CPU asynchronously (camera frame, GPU done, packet). Kernel uses a **top half** (fast, acknowledge HW, schedule work) and **bottom half** (do work without holding IRQs disabled). Misuse causes dropped frames and latency spikes.

---

## Interrupts vs Exceptions

| Type | Origin | Sync? | Example |
|------|--------|-------|--------|
| Hardware interrupt | Device IRQ | No | NIC, GPU done, camera frame |
| Software interrupt | INT/SVC | Yes | Syscall, breakpoint |
| Exception (fault) | CPU error | Yes | Page fault, div-by-zero |
| Exception (abort) | Unrecoverable | Yes | Machine check |

**Fault:** re-execute instruction after fix. **Trap:** advance after handler. **Abort:** no return.

---

## Interrupt Controllers

**ARM GIC v3:** SGI (0–15, IPI), PPI (16–31, per-CPU), SPI (32–1019, devices). Priority 0–255; PMR masks. **x86 APIC:** Local APIC + I/O APIC; IDT; TPR. **MSI/MSI-X (PCIe):** Message-signaled interrupts; MSI-X gives **per-queue vectors and CPU affinity** — used by NVMe, GPU, NICs for scalability.

---

## Top Half vs Bottom Half

**Top half (ISR):** Runs with IRQs disabled on local CPU; must be very short (acknowledge HW, save pointer, schedule bottom half); cannot sleep. **Bottom half:** Softirq, tasklet, or workqueue; can do real work; some can sleep (workqueue). Long top half blocks other IRQs and adds latency.

---

## Bottom-Half Mechanisms

**Softirq:** Statically allocated; runs in ksoftirqd or after hardirq; cannot sleep. **Tasklet:** Built on softirq; one at a time per CPU. **Workqueue:** Process context; can sleep; for heavier work (e.g. block I/O). Threaded IRQs run handler in a kthread — preemptible under PREEMPT_RT.

---

# Part 2 (L4): System Calls, vDSO & eBPF

**Context:** User code reaches kernel only via **syscalls** (toll booth). Cost: mode switch, Spectre/Meltdown mitigations, TLB effects (**~100–400 ns round-trip**). **vDSO** avoids syscall for time; **eBPF** observes kernel without changing code.

---

## Syscall Path

x86-64: SYSCALL → entry_SYSCALL_64 → sys_call_table[rax] → SYSRET. ARM64: SVC → el0_svc → sys_call_table. `strace -c` / `strace -T -e mmap,ioctl` to count and time syscalls.

---

## Syscall Overhead

**Mode switch ~50–150 ns**; mitigations (KPTI, IBRS, retpoline) add ~50–200 ns on x86; ARM64 lighter. At 200 fps × 4 ioctls = 1600/s × 300 ns ≈ 0.5 ms/s; **batching and zero-copy** (mmap, io_uring) reduce crossings.

---

## vDSO

Kernel maps a **read-only page** with time (and a few other) helpers. `clock_gettime(CLOCK_MONOTONIC)`, `gettimeofday()` can resolve **in userspace** (~10–20 ns) without SYSCALL. Use CLOCK_MONOTONIC for timing and sensor fusion; **CLOCK_REALTIME can jump** (NTP).

---

## Key Syscalls: mmap, ioctl

**mmap:** MAP_SHARED for zero-copy shm; MAP_HUGETLB for large buffers and TLB reduction; CUDA pinned and Jetson unified memory use mmap on device nodes. **ioctl:** Device control (V4L2, GPU); nearly all device-specific ops; path: sys_ioctl → driver ioctl_ops (e.g. VIDIOC_DQBUF blocks until frame).

---

## eBPF

Programs run in kernel (**verified, JIT**); attach to kprobes, tracepoints, uprobes, XDP. **Observability without modifying kernel or app**: profile syscalls, scheduler, driver paths. bpftrace, BCC, libbpf; **production-safe instrumentation**.

---

## Summary

| L3 | IRQ vs exception; GIC/APIC; MSI-X; top/bottom half; softirq/tasklet/workqueue; keep top half &lt;1 µs. |
| L4 | Syscall path and cost; vDSO for time; mmap/ioctl; eBPF for observability. |

---

## AI Hardware Connection

- L3: Camera/GPU/NVMe pipelines depend on IRQ flow; MSI-X per queue for NVMe and GPU; long ISR blocks RT tasks — use bottom half for work.
- L4: V4L2 ioctl cost in camera path; vDSO for timestamps; eBPF to trace inference and driver latency without code changes.

---

*Combines Lectures L3, L4 (Interrupts, Exceptions & Bottom Halves; System Calls, vDSO & eBPF).*
