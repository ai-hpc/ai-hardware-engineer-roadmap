# Lecture 4: System Calls, vDSO & eBPF

## Overview

Every time user-space code needs the kernel to do something — open a file, allocate memory, send a packet, talk to a GPU — it crosses the hardware privilege boundary via a system call. The core challenge this lecture addresses is: how does this boundary crossing work, how expensive is it, and how can you observe what is happening inside the kernel without modifying source code? The mental model is a **toll booth**: every syscall is a controlled crossing from user space into kernel space, with a fixed cost whether the work takes 1 ns or 1 ms. For an AI hardware engineer, this matters because a 200 fps camera pipeline can spend measurable CPU time just crossing the toll booth for V4L2 ioctls, and eBPF is the instrument you use to measure and diagnose the kernel's internal behavior in production without changing a single line of application code.

---

## The System Call Interface

**System calls** are the only sanctioned path for user-space code to request kernel services. User space **cannot** directly access hardware registers, allocate physical memory, or change scheduler policy — it asks the kernel via the **syscall ABI**.

### Call Path — x86-64

```
Syscall Path — x86-64
┌─────────────────────────────────────────────────────────┐
│  User Space (Ring 3)                                    │
│                                                         │
│  Application code                                       │
│      │ calls read(fd, buf, len)                         │
│      ▼                                                  │
│  glibc wrapper                                          │
│      │ mov $0, %eax    (syscall number for read = 0)    │
│      │ mov fd, %rdi    (first argument)                 │
│      │ mov buf, %rsi   (second argument)                │
│      │ mov len, %rdx   (third argument)                 │
│      │ SYSCALL         ← hardware mode switch           │
└──────┼──────────────────────────────────────────────────┘
       │ Ring 3 → Ring 0 (hardware enforced)
┌──────┼──────────────────────────────────────────────────┐
│  Kernel Space (Ring 0)                                  │
│      ▼                                                  │
│  entry_SYSCALL_64                                       │
│      │ saves registers to kernel stack                  │
│      │ looks up sys_call_table[RAX]                     │
│      ▼                                                  │
│  sys_read()                                             │
│      │ does actual work (VFS, page cache, driver)       │
│      │ return value → RAX                               │
│      │ SYSRET          ← hardware mode switch back      │
└──────┼──────────────────────────────────────────────────┘
       │ Ring 0 → Ring 3 (hardware enforced)
┌──────┼──────────────────────────────────────────────────┐
│  User Space (Ring 3)                                    │
│      ▼                                                  │
│  glibc wrapper returns to application                   │
└─────────────────────────────────────────────────────────┘
```

ARM64 uses `SVC #0` → `VBAR_EL1 + 0x400` → `el0_svc` → `sys_call_table[x8]()` → `ERET`.

```bash
strace -c ./inference_app    # count syscalls by type and cumulative time
strace -T -e mmap,ioctl ./camerad  # trace specific calls with per-call duration
```

---

## Syscall Overhead

| Cost component | Typical penalty |
|---|---|
| Mode switch (SYSCALL/SYSRET) | 50–150 ns |
| Spectre/Meltdown mitigations (IBRS, retpoline, KPTI) | 50–200 ns |
| TLB and cache effects (KPTI flushes user-space TLB entries) | 20–100 ns |
| Total round-trip (minimal kernel work) | 100–400 ns |

ARM64 mitigations (CSV2, SSBS) are **lighter than x86**. At 200 fps across 2 cameras, each frame requiring 4 V4L2 ioctls = 1600 syscalls/s × 300 ns = 0.5 ms/s pure **mode-switch overhead**. **Batching and zero-copy** (`mmap`, `io_uring`) reduce crossing frequency.

> **Key Insight:** Spectre and Meltdown mitigations (KPTI, IBRS, retpoline) roughly doubled syscall overhead on x86 compared to pre-2018 systems. The kernel flushes or isolates page table entries on each user↔kernel transition to prevent speculative execution from leaking kernel memory. ARM64 mitigations are architecturally lighter — one reason embedded AI platforms often prefer ARM for latency-sensitive workloads. If you are porting code from x86 benchmarks, do not assume syscall overhead is similar.

Now that we understand the cost of crossing the privilege boundary, let's look at how the kernel avoids that crossing for the most frequently called time functions.

---

## vDSO: Kernel Calls Without Kernel Entry

The **virtual Dynamic Shared Object (vDSO)** is a read-only ELF page mapped by the kernel into every process address space at startup. Selected time functions read from a shared `vvar` data page that the kernel updates — no `SYSCALL` instruction, no mode transition, no TLB flush.

Think of the vDSO as a memo that the kernel leaves in your address space: "Here is the current time, updated continuously by the kernel. Read it directly without asking me."

| Function | Full syscall | vDSO |
|---|---|---|
| `clock_gettime(CLOCK_MONOTONIC)` | ~200 ns | ~10–20 ns |
| `clock_gettime(CLOCK_REALTIME)` | ~200 ns | ~10–20 ns |
| `gettimeofday()` | ~200 ns | ~10–20 ns |
| `getcpu()` | ~150 ns | ~5 ns |

Calling through glibc **automatically uses vDSO** when available — no application change required. Use `CLOCK_MONOTONIC` for inter-process timing (not subject to NTP adjustments); use `CLOCK_MONOTONIC_RAW` for hardware-only monotonic counter.

```bash
# Verify vDSO is mapped
cat /proc/[pid]/maps | grep vdso
# Check which symbols are exported
nm /proc/[pid]/map_files/[vdso-range] 2>/dev/null | grep " T "
```

> **Common Pitfall:** `CLOCK_REALTIME` is adjusted by NTP and can jump backward. For timestamping sensor data or measuring inter-frame intervals, always use `CLOCK_MONOTONIC` (or `CLOCK_MONOTONIC_RAW` to also exclude PTP/NTP rate adjustments). Using `CLOCK_REALTIME` for sensor fusion timestamps creates subtle synchronization bugs when NTP makes a correction mid-run.

---

## Key Syscalls for AI and Embedded Systems

Understanding the most important syscalls for AI hardware work means understanding what happens under the hood of every V4L2 camera operation, GPU command submission, and zero-copy buffer transfer.

### mmap / munmap

```c
/* Zero-copy shared memory between processes */
int fd = shm_open("/sensor_ring", O_CREAT | O_RDWR, 0600);
/* shm_open creates an anonymous file in /dev/shm backed by tmpfs */
void *buf = mmap(NULL, SIZE, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
/* MAP_SHARED: writes are visible to all processes that mapped this fd */
/* No copy is ever made — all processes share the same physical pages */

/* Huge page mapping for large GPU staging buffer */
void *huge = mmap(NULL, 2*1024*1024, PROT_READ | PROT_WRITE,
                  MAP_PRIVATE | MAP_ANONYMOUS | MAP_HUGETLB, -1, 0);
/* MAP_HUGETLB: allocate 2 MB huge pages instead of 4 KB pages */
/* Reduces TLB pressure: 1 TLB entry covers 2 MB instead of 4 KB */
/* Requires vm.nr_hugepages to be pre-allocated in /proc/sys/vm/ */
```

- `MAP_HUGETLB`: reduces TLB pressure for large staging buffers; requires `vm.nr_hugepages`
- `CUDA cudaMallocHost()` calls `mmap` on `/dev/nvidia*` for pinned host memory
- On Jetson unified memory, `cudaMalloc` uses `mmap` on `/dev/nvhost-as-gpu`

### ioctl

**Primary device control interface**; nearly all hardware-specific operations use it.

```
Syscall Path for V4L2 Camera Buffer Dequeue
┌─────────────┐
│  camerad    │  ioctl(fd, VIDIOC_DQBUF, &buf)
└──────┬──────┘
       │ SYSCALL (ioctl number)
       ▼
┌─────────────────────────────────────────────────────────┐
│  sys_ioctl() → vfs_ioctl() → v4l2_ioctl()              │
│  → video_device.ioctl_ops.vidioc_dqbuf()                │
│  → driver's dequeue function                           │
│  → blocks in TASK_UNINTERRUPTIBLE until frame arrives  │
│  → returns buffer with frame pointer                   │
└─────────────────────────────────────────────────────────┘
       │ SYSRET
       ▼
┌─────────────┐
│  camerad    │  buf.m.userptr now points to frame data
└─────────────┘
```

| ioctl | Device | Purpose |
|---|---|---|
| `VIDIOC_QBUF` / `VIDIOC_DQBUF` | V4L2 camera | Queue / dequeue capture buffer |
| `VIDIOC_STREAMON` / `STREAMOFF` | V4L2 | Start / stop streaming |
| `DRM_IOCTL_GEM_*` | DRM/KMS | Buffer allocation, display scanout |
| `NVGPU_IOCTL_CHANNEL_ALLOC_GPFIFO` | NVIDIA GPU (`/dev/nvhost-gpu`) | Allocate GPU command queue |
| `RPMSG_CREATE_EPT_IOCTL` | RPMsg | IPC with Cortex-M coprocessor (i.MX8) |
| Custom `_IOWR(MAGIC, N, struct)` | FPGA PCIe driver | Submit inference workload to accelerator |

### epoll

**O(1) per-event multiplexing** across many file descriptors. Used in openpilot's `cereal` messaging layer to multiplex CAN frames, camera V4L2 events, IMU data, and model output events.

```c
int epfd = epoll_create1(EPOLL_CLOEXEC);
/* EPOLL_CLOEXEC: close the epoll fd automatically on exec() */
struct epoll_event ev = { .events = EPOLLIN | EPOLLET, .data.fd = camera_fd };
/* EPOLLET: edge-triggered — notify once when data arrives, not repeatedly */
epoll_ctl(epfd, EPOLL_CTL_ADD, camera_fd, &ev);
int n = epoll_wait(epfd, events, MAX_EVENTS, timeout_ms);
/* blocks until at least one fd is ready; returns number of ready events */
```

**Edge-triggered mode** (`EPOLLET`) is preferred for latency-sensitive paths — **one wakeup per edge**, no repeated notifications for unread data.

### prctl

```
PR_SET_NAME          Name thread; visible in ps/top/htop/perf
PR_SET_TIMERSLACK    Reduce timer coalescing (set to 1 ns for RT threads; default 50 µs)
PR_SET_SECCOMP       Apply seccomp-BPF filter to current thread
PR_SET_NO_NEW_PRIVS  Prevent privilege escalation across exec()
```

> **Key Insight:** `PR_SET_TIMERSLACK` is a hidden source of timing jitter. The default 50 µs timer slack allows the kernel to coalesce nearby timer expirations to save power. An RT thread waiting for a 1 ms timer may be woken up to 50 µs late. Setting `prctl(PR_SET_TIMERSLACK, 1)` (1 ns) disables coalescing for that thread and removes this source of jitter. This is standard practice for `controlsd` and any CAN write thread on openpilot.

### sched_setattr — SCHED_DEADLINE

```c
struct sched_attr attr = {
    .size           = sizeof(attr),
    .sched_policy   = SCHED_DEADLINE,
    .sched_runtime  = 5000000,     /* 5ms budget per period — CPU time consumed before descheduling */
    .sched_deadline = 16666666,    /* 16.7ms relative deadline — must complete by this time */
    .sched_period   = 16666666,    /* 16.7ms period — one activation per period (60fps) */
};
sched_setattr(0, &attr, 0);        /* 0 = self; requires CAP_SYS_NICE */
```

### perf_event_open

Accesses **hardware performance counters** from userspace. Used by `perf stat`, Nsight Systems, and VTune.

```c
struct perf_event_attr pe = {
    .type   = PERF_TYPE_HARDWARE,
    .config = PERF_COUNT_HW_CACHE_MISSES,  /* LLC miss counter */
    .disabled = 1,                          /* start disabled; enable manually */
};
int fd = perf_event_open(&pe, 0, -1, -1, 0);  /* measure self (pid=0) */
ioctl(fd, PERF_EVENT_IOC_ENABLE, 0);          /* start counting */
/* ... inference workload runs here ... */
read(fd, &count, sizeof(count));               /* read accumulated miss count */
```

On Jetson, ARM PMU counters measure LLC miss rate during DNN inference — guides INT8 tiling decisions.

### memfd_create

Creates an **anonymous file-backed memory region** — usable as shared memory without a filesystem path. Used in openpilot's `msgq` for **zero-copy IPC** between `modeld` and `controlsd`.

```c
int fd = memfd_create("shared_tensor", MFD_CLOEXEC);
/* Creates an anonymous file in kernel memory — no filesystem path needed */
ftruncate(fd, TENSOR_SIZE);
/* Sets the file size; backing pages are allocated lazily on first access */
void *ptr = mmap(NULL, TENSOR_SIZE, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
/* Map the shared region into this process's address space */
/* pass fd over Unix socket to second process for its own mmap */
/* both processes now share the same physical pages — zero-copy IPC */
```

This pattern allows `modeld` to write inference outputs directly into a shared buffer that `controlsd` reads without any copy. The `fd` is passed over a Unix domain socket using `SCM_RIGHTS`.

> **Common Pitfall:** `memfd_create` buffers live in RAM (tmpfs). If the inference tensor is large (e.g., 50 MB feature map), creating and destroying shared tensors at 30 fps consumes significant RAM bandwidth from repeated page faults. Pre-allocate the memfd buffers once at startup and reuse them, rather than creating new ones per frame.

---

## eBPF: Kernel-Attached Verified Programs

**eBPF** programs run inside the kernel with verified safety: the kernel verifier checks bounds, loop termination, and pointer types before JIT-compiling to native code. No kernel module, no recompile. Think of eBPF as a read-only microscope you can attach to any kernel function at runtime — it observes without modifying.

### Architecture

```
eBPF Program Lifecycle
┌──────────────────────────────────────────────────────────────┐
│  Development                                                 │
│  C source → clang + libbpf → BPF bytecode (.o file)         │
└───────────────────────────────┬──────────────────────────────┘
                                │ load via bpf() syscall
                                ▼
┌──────────────────────────────────────────────────────────────┐
│  Kernel Verification                                         │
│  → bounds checker (no out-of-bounds memory access)          │
│  → loop termination verifier (no infinite loops)            │
│  → pointer type checker (no arbitrary kernel ptr dereference)│
│  → JIT compile to native code                               │
└───────────────────────────────┬──────────────────────────────┘
                                │ attach to hook
                                ▼
┌──────────────────────────────────────────────────────────────┐
│  Runtime                                                     │
│  Event fires (syscall / tracepoint / kprobe / XDP packet)   │
│  → eBPF program runs in kernel context                      │
│  → writes results to BPF maps (ring buffer / hash / array)  │
└───────────────────────────────┬──────────────────────────────┘
                                │ user-space reads maps
                                ▼
┌──────────────────────────────────────────────────────────────┐
│  User Space                                                  │
│  bpftrace / BCC tool reads BPF maps                         │
│  → latency histograms, counts, traces                       │
└──────────────────────────────────────────────────────────────┘
```

### Attachment Points

| Hook | Use |
|---|---|
| `kprobe` / `kretprobe` | Any kernel function entry/return — driver internals |
| `tracepoint` | Stable kernel tracepoints: `sched:sched_switch`, `irq:irq_handler_entry`, `block:block_rq_issue` |
| `uprobe` | User-space function: TensorRT engine execution, glibc allocations |
| `XDP` | Pre-stack packet processing in driver context — 100GbE line rate filtering |
| `TC` (traffic control) | Post-stack egress/ingress; used for latency tagging |

### BCC / bpftrace Production Tools

```bash
runqlat                                   # scheduler run-queue latency histogram
# First tool to run when inference latency is inconsistent

offcputime -p $(pgrep modeld) 10          # where modeld spends time blocked off-CPU
# Shows what kernel function modeld is sleeping in and for how long

# Count context switches per process
bpftrace -e 'tracepoint:sched:sched_switch { @[comm] = count(); }'
# High counts on modeld → investigate CFS preemption; consider SCHED_FIFO

# Trace ioctl latency from camerad
bpftrace -e '
  tracepoint:syscalls:sys_enter_ioctl /comm == "camerad"/ { @s[tid] = nsecs; }
  tracepoint:syscalls:sys_exit_ioctl  /comm == "camerad"/ {
    @us = hist((nsecs - @s[tid]) / 1000); delete(@s[tid]); }'
# Outputs histogram of VIDIOC_DQBUF latency in microseconds

execsnoop        # trace new process executions
opensnoop        # trace file opens (useful to find what config camerad loads)
biolatency       # block I/O latency histogram (model loading from NVMe)
```

> **Key Insight:** eBPF is uniquely powerful because it attaches to production systems without any source code changes, recompilation, or restart. You can instrument `modeld` running on a Jetson in the field, collect a latency histogram of `VIDIOC_DQBUF` calls, and detach the probe — all while the inference pipeline continues running. This is impossible with traditional profilers that require either source instrumentation or stopping the process.

> **Common Pitfall:** eBPF programs that use `kprobe` hooks are fragile across kernel versions — kernel internal function names and signatures change between releases. Use `tracepoint` hooks instead when possible; tracepoints are stable ABIs defined in `Documentation/trace/tracepoints.rst`. The `syscalls:sys_enter_ioctl` tracepoint will work on any Linux kernel version.

---

## seccomp: Syscall Filtering

A **BPF program** evaluated on every syscall; returns `ALLOW`, `ERRNO(N)`, `KILL`, or `TRAP`. Applied with `prctl(PR_SET_SECCOMP, SECCOMP_MODE_FILTER, &prog)`.

Docker's default seccomp profile blocks ~44 syscalls (`kexec_load`, `ptrace`, `mount`, `unshare`, etc.). A TensorRT inference container needs fewer than 50 syscalls; a tight allowlist eliminates the rest. Prevents post-exploit lateral movement in containerized AI deployments.

```
seccomp Decision Flow
┌─────────────┐
│  Application│  calls read(fd, buf, len)
└──────┬──────┘
       │ SYSCALL
       ▼
┌──────────────────────────────────┐
│  seccomp-BPF filter              │
│  checks syscall number against   │
│  the program's allow-list        │
│                                  │
│  read (0)?  → ALLOW → continue  │
│  ptrace(101)? → KILL             │
│  kexec_load? → ERRNO(EPERM)     │
└──────────────────────────────────┘
```

---

## Linux Capabilities

Root privilege is split into **~40 fine-grained grants**. **Drop capabilities** after setup.

| Capability | Grants |
|---|---|
| `CAP_SYS_NICE` | `sched_setscheduler()`, `sched_setattr()` — set RT priority without root |
| `CAP_IPC_LOCK` | `mlockall()` — lock all memory pages; eliminates RT page-fault latency |
| `CAP_NET_ADMIN` | Network configuration, raw sockets, eBPF TC programs |
| `CAP_SYS_RAWIO` | Direct hardware I/O, PCIe MMIO access, FPGA register writes |
| `CAP_PERFMON` | `perf_event_open()` with hardware counters |

```bash
setcap cap_sys_nice+ep /opt/inference/modeld    # grant RT capability; no sudo at runtime
grep Cap /proc/$(pgrep modeld)/status           # inspect effective capability mask
```

> **Key Insight:** `setcap` writes the capability into the ELF binary's extended attributes. When `modeld` executes, the kernel reads these attributes and grants the capabilities without root access. This is safer than running `modeld` as root because the process only has the specific capabilities it needs — it cannot, for example, mount filesystems (`CAP_SYS_ADMIN`) or load kernel modules (`CAP_SYS_MODULE`). Principle of least privilege applied to system calls.

> **Common Pitfall:** `setcap` grants are lost when a binary is replaced (e.g., by a software update). A Makefile or install script that copies a new binary without re-applying `setcap` will silently break RT scheduling. Always include the `setcap` call in the install step and verify with `getcap /opt/inference/modeld` after deployment.

---

## Summary

| Mechanism | Kernel entry? | Latency | Primary use |
|---|---|---|---|
| Syscall (SYSCALL / SVC) | Yes | 100–400 ns | All kernel services |
| vDSO (`clock_gettime`) | No | 10–20 ns | High-frequency timestamping |
| `mmap` (after setup) | No (page faults only) | Sub-ns when cached | Zero-copy buffers, shared memory |
| eBPF (JIT, kernel hook) | Runs in kernel | < 1 µs per probe | Production profiling, syscall filtering |
| seccomp filter | Per-syscall check | 5–20 ns overhead | Sandbox; attack surface reduction |
| vDSO `getcpu()` | No | ~5 ns | Determine current CPU for lockless rings |

### Conceptual Review

- **Why does crossing from user space to kernel space cost 100–400 ns?** The CPU must save registers, switch page tables (KPTI), flush TLB entries, and execute Spectre/Meltdown mitigations on x86. The actual kernel work may be trivial, but the transition itself has unavoidable hardware costs. ARM64 mitigations are architecturally lighter, which is one reason embedded AI platforms prefer ARM.
- **What is the vDSO and why is it faster than a syscall?** The vDSO is a kernel-maintained ELF page mapped into every process. It contains functions like `clock_gettime` that read from a shared `vvar` page the kernel updates continuously. No `SYSCALL` instruction is needed — the function executes entirely in user space, reducing latency from ~200 ns to ~15 ns.
- **What is the difference between `epoll` level-triggered and edge-triggered modes?** Level-triggered: `epoll_wait` returns as long as there is data available (can return the same fd multiple times). Edge-triggered: `epoll_wait` returns only when new data arrives (one notification per event). Edge-triggered is preferred for high-throughput paths like V4L2 events because it avoids repeated wakeups when the consumer is slower than the producer.
- **What does eBPF's kernel verifier check?** It verifies that the program has no out-of-bounds memory accesses, will always terminate (no infinite loops), and does not dereference arbitrary kernel pointers. Only programs that pass all checks are JIT-compiled and attached. This is what makes eBPF safe to run in production kernel context without source changes.
- **Why use `memfd_create` instead of POSIX shared memory (`shm_open`)?** `memfd_create` files are anonymous — they have no filesystem path and are automatically cleaned up when all file descriptors referencing them are closed. `shm_open` creates a file in `/dev/shm` that persists after process exit (until explicitly unlinked), which can leak shared memory between runs.
- **What is seccomp and why does it matter for inference containers?** seccomp filters the syscalls a process is allowed to make. A TensorRT inference container only needs ~50 syscalls (`mmap`, `ioctl`, `read`, `write`, `epoll_wait`, etc.). Blocking everything else (including `ptrace`, `kexec_load`, `mount`) means a compromised inference process cannot escalate privilege or persist on the system — the kernel refuses the dangerous syscall before any exploit code runs.

---

## AI Hardware Connection

- vDSO `clock_gettime(CLOCK_MONOTONIC)` provides µs-accurate sensor timestamping at ~15 ns per call — essential for synchronizing camera frames, IMU samples, and CAN messages in openpilot's sensor fusion pipeline without kernel entry overhead.
- `ioctl(VIDIOC_DQBUF)` is the hot path in every V4L2 camera pipeline; measuring its latency distribution with `bpftrace` directly quantifies camera-to-model input delay without modifying or recompiling `camerad`.
- `bpftrace runqlat` is the first diagnostic for inference latency spikes — 5 ms scheduler run-queue latency on the model thread appears immediately in the histogram without code changes or kernel module insertion.
- `CAP_SYS_NICE` via `setcap` allows `modeld` to set its own RT scheduling policy without running as root, compatible with container security policies and Kubernetes `securityContext.capabilities`.
- seccomp allowlists on TensorRT inference containers block syscalls like `ptrace`, `kexec_load`, and `mount` that are meaningless for inference but provide significant post-exploit paths in AV deployments.
- `perf_event_open` with ARM PMU events on Jetson Orin measures LLC miss rate during DNN inference — cache miss data guides INT8 quantization and layer tiling decisions to reduce the memory working set below the L2/L3 cache size.
