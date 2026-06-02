# Lecture 26: eBPF — Programmable Kernel Observability & Networking

## Overview

Lecture 4 introduced eBPF as a "read-only microscope" for production profiling. This lecture goes deep: how the eBPF virtual machine works, what the verifier actually checks, how to write programs from scratch with libbpf, and how to use eBPF for AI system observability — tracing GPU driver latency, measuring DMA throughput, profiling scheduler decisions on inference threads, and building custom network filtering for autonomous vehicle telemetry. The core challenge is: how do you build **production-safe, low-overhead, kernel-level instrumentation** without writing kernel modules or rebooting? The mental model is that eBPF turns the Linux kernel into a **programmable platform** — you inject small verified programs at specific hook points, and the kernel executes them at native speed every time that hook fires. For an AI hardware engineer, eBPF is the difference between "inference sometimes takes 30 ms instead of 10 ms and we don't know why" and a precise histogram showing that VIDIOC_DQBUF blocks for 20 ms when the ISP is processing HDR frames.

---

## eBPF Virtual Machine

eBPF is not "a tracing tool." It is a **general-purpose in-kernel virtual machine** with a RISC-like instruction set, verified execution, and JIT compilation to native code. Tools like `bpftrace` and BCC are frontends — they generate eBPF bytecode that runs on this VM.

### Register File

| Register | Purpose |
|---|---|
| `r0` | Return value (also: helper function return) |
| `r1`–`r5` | Function arguments (caller-saved) |
| `r6`–`r9` | Callee-saved (preserved across helper calls) |
| `r10` | Frame pointer (read-only; points to 512-byte stack) |

The eBPF ISA has **11 registers** — deliberately similar to ARM64's calling convention. This makes JIT compilation to ARM (Jetson, mobile SoCs) very efficient: eBPF registers map **nearly 1:1 to hardware registers**.

### Instruction Set

```
eBPF Instruction Format (64-bit):
┌──────────┬──────┬──────┬──────────┬────────────────────────────────┐
│  opcode  │ dst  │ src  │  offset  │           immediate            │
│  8 bits  │4 bits│4 bits│ 16 bits  │           32 bits              │
└──────────┴──────┴──────┴──────────┴────────────────────────────────┘
```

| Category | Instructions | Example |
|---|---|---|
| ALU (64-bit) | `add`, `sub`, `mul`, `div`, `mod`, `or`, `and`, `xor`, `lsh`, `rsh`, `arsh`, `neg` | `r0 += r1` |
| ALU (32-bit) | Same ops with `w` suffix | `w0 += w1` (32-bit add) |
| Memory | `ldx`, `stx`, `st` (1/2/4/8 byte) | `r0 = *(u64 *)(r1 + 16)` |
| Branch | `jeq`, `jne`, `jgt`, `jge`, `jlt`, `jle`, `jset` | `if r0 > r1 goto +5` |
| Call | `call imm` | `call bpf_ktime_get_ns` |
| Exit | `exit` | Return `r0` to caller |
| Atomic | `lock xadd`, `lock cmpxchg`, `lock xchg` | Atomic counter increment |

**No floating point.** eBPF has integer-only ALU. Latency histograms use integer nanoseconds; throughput calculations use integer bytes/sec. This is by design — floating-point exceptions in kernel context are dangerous and non-deterministic.

### JIT Compilation

The kernel JIT-compiles eBPF bytecode to **native machine instructions** at load time:

| Architecture | JIT Quality | Notes |
|---|---|---|
| x86-64 | Excellent | Near 1:1 mapping; eBPF was designed with x86 in mind |
| ARM64 (AArch64) | Excellent | Jetson Orin, Qualcomm SoCs — register mapping is natural |
| ARM32 | Good | Older embedded, some Cortex-A devices |
| RISC-V | Good | Growing support; relevant for open-source AI chips |

After JIT, eBPF programs run at **native instruction speed** — not interpreted. The overhead per probe hit is typically **50–200 ns**, dominated by the function call trampoline, not the eBPF instructions.

---

## The Verifier: Why eBPF Is Safe

Before any eBPF program executes, the kernel verifier performs **static analysis** on every possible execution path. This is what makes eBPF safe for production — a buggy eBPF program is **rejected at load time, never at runtime**.

### Verification Checks

| Check | What It Prevents |
|---|---|
| **Reachability** | All instructions must be reachable; no dead code hiding malicious paths |
| **No unreachable instructions** | Program must terminate via `exit` on all paths |
| **Bounded loops** | Loops must have provably bounded iteration count (since Linux 5.3: bounded loops allowed; before that, no loops at all) |
| **Memory safety** | Every pointer dereference is checked for bounds; no arbitrary kernel memory access |
| **Type tracking** | Registers are tracked as `NOT_INIT`, `SCALAR`, `PTR_TO_MAP_VALUE`, `PTR_TO_CTX`, etc. — can't use a scalar as a pointer |
| **Stack bounds** | Stack accesses must be within the 512-byte frame; no buffer overflow |
| **Helper argument types** | Each helper function specifies expected argument types; verifier checks they match |
| **Privilege level** | Unprivileged users can only run cgroup/socket programs; `CAP_BPF` or `CAP_SYS_ADMIN` required for tracing |

### Verification Example

```c
// This program PASSES verification:
SEC("tracepoint/syscalls/sys_enter_ioctl")
int trace_ioctl(struct trace_event_raw_sys_enter *ctx) {
    u64 ts = bpf_ktime_get_ns();           // helper call — verifier knows return type
    u32 pid = bpf_get_current_pid_tgid();   // another safe helper
    bpf_map_update_elem(&start_ts, &pid, &ts, BPF_ANY);  // map access — verifier checks key/value sizes
    return 0;                                // explicit exit
}

// This program FAILS verification:
SEC("tracepoint/syscalls/sys_enter_ioctl")
int bad_program(struct trace_event_raw_sys_enter *ctx) {
    char *ptr = (char *)0xffff888000000000;  // arbitrary kernel pointer
    char c = *ptr;                            // REJECTED: direct kernel memory access
    return 0;
}
// verifier error: "R1 type=scalar expected=fp"
```

### Complexity Limits

| Limit | Value (Linux 6.x) | Purpose |
|---|---|---|
| Max instructions | 1,000,000 | Prevent excessive verification time |
| Max verified states | 64 per instruction | Bound verifier memory |
| Stack size | 512 bytes | Fixed; no dynamic allocation |
| Tail calls depth | 33 | Prevent infinite recursion |
| Max map entries | Configurable per map | Memory budget |
| Helper call nesting | 8 | Bound stack depth |

> **Key Insight:** The verifier is the reason eBPF is trusted in production. Unlike a kernel module — which can dereference any pointer, corrupt any data structure, and crash the system — an eBPF program is mathematically proven safe before it runs. The trade-off is expressiveness: you cannot write arbitrary kernel code in eBPF. But for observability and networking, the restricted model is sufficient and the safety guarantee is invaluable.

---

## BPF Maps: Kernel↔User Data Structures

BPF maps are **shared data structures** between eBPF programs (kernel side) and user-space applications. They are the primary mechanism for getting data out of eBPF programs.

| Map Type | Use Case | AI/Embedded Example |
|---|---|---|
| `BPF_MAP_TYPE_HASH` | Key-value store | Per-PID ioctl latency tracking |
| `BPF_MAP_TYPE_ARRAY` | Fixed-size indexed array | Per-CPU counters for interrupt frequency |
| `BPF_MAP_TYPE_RINGBUF` | Lock-free SPSC ring buffer (preferred) | Stream of timestamped events to user space |
| `BPF_MAP_TYPE_PERF_EVENT_ARRAY` | Per-CPU perf event ring | Legacy event streaming (prefer ringbuf) |
| `BPF_MAP_TYPE_PERCPU_HASH` | Per-CPU hash (no locking) | Concurrent per-CPU statistics |
| `BPF_MAP_TYPE_LRU_HASH` | Auto-evicting hash | Track recent connections without unbounded growth |
| `BPF_MAP_TYPE_STACK_TRACE` | Kernel/user stack capture | Profile where modeld spends time in kernel |
| `BPF_MAP_TYPE_PROG_ARRAY` | Tail call dispatch table | Chain eBPF programs for complex tracing logic |
| `BPF_MAP_TYPE_BLOOM_FILTER` | Probabilistic set membership | Fast PID filtering for tracing |

### Ring Buffer (BPF_MAP_TYPE_RINGBUF)

The ring buffer is the **preferred mechanism for streaming events** from kernel to user space. It uses a **single shared buffer** (not per-CPU), supports variable-length records, and has excellent cache behavior.

```c
// Kernel side (eBPF program)
struct {
    __uint(type, BPF_MAP_TYPE_RINGBUF);
    __uint(max_entries, 256 * 1024);      // 256 KB ring buffer
} events SEC(".maps");

struct event {
    u64 timestamp_ns;
    u32 pid;
    u32 latency_us;
    char comm[16];
};

SEC("tracepoint/syscalls/sys_exit_ioctl")
int trace_ioctl_exit(struct trace_event_raw_sys_exit *ctx) {
    struct event *e;
    e = bpf_ringbuf_reserve(&events, sizeof(*e), 0);
    if (!e) return 0;  // ring full — drop event (safe)

    e->timestamp_ns = bpf_ktime_get_ns();
    e->pid = bpf_get_current_pid_tgid() >> 32;
    e->latency_us = /* computed from start timestamp */;
    bpf_get_current_comm(&e->comm, sizeof(e->comm));

    bpf_ringbuf_submit(e, 0);  // publish to user space
    return 0;
}
```

```c
// User side (C application using libbpf)
static int handle_event(void *ctx, void *data, size_t data_sz) {
    struct event *e = data;
    printf("%-16s pid=%-6d latency=%u us\n", e->comm, e->pid, e->latency_us);
    return 0;
}

struct ring_buffer *rb = ring_buffer__new(bpf_map__fd(skel->maps.events),
                                          handle_event, NULL, NULL);
while (!stop) {
    ring_buffer__poll(rb, 100 /* timeout ms */);
}
```

> **Key Insight:** `BPF_MAP_TYPE_RINGBUF` replaced `BPF_MAP_TYPE_PERF_EVENT_ARRAY` as the preferred event streaming mechanism. The ring buffer is a single shared buffer (not per-CPU), so events are globally ordered by timestamp — critical for correlating camera frame events with GPU completion events across different CPUs. Per-CPU perf buffers require user-space merging and sorting, which adds latency and complexity.

---

## Writing eBPF Programs with libbpf

**libbpf** is the canonical C library for loading and interacting with eBPF programs. It provides the "CO-RE" (Compile Once — Run Everywhere) mechanism that makes eBPF programs portable across kernel versions.

### CO-RE: Compile Once, Run Everywhere

The problem: **kernel data structures change between versions**. A `struct task_struct` field might be at offset 1248 in kernel 5.10 and offset 1264 in kernel 6.1. Without CO-RE, you'd need to compile eBPF programs per kernel version.

CO-RE solves this with **BTF (BPF Type Format)** — a compact type metadata embedded in the kernel that describes the layout of all structures at runtime.

```c
// Without CO-RE — fragile, breaks across kernel versions:
u32 pid = *(u32 *)((char *)task + 1248);  // hardcoded offset!

// With CO-RE — portable:
u32 pid = BPF_CORE_READ(task, tgid);
// At load time, libbpf reads kernel BTF and adjusts the offset automatically
```

### Complete libbpf Program: Tracing V4L2 ioctl Latency

This example measures the **latency of every V4L2 ioctl call** from `camerad` — the camera pipeline process in openpilot.

**BPF program (`ioctl_lat.bpf.c`):**

```c
#include "vmlinux.h"
#include <bpf/bpf_helpers.h>
#include <bpf/bpf_tracing.h>
#include <bpf/bpf_core_read.h>

#define MAX_ENTRIES 10240
#define TASK_COMM_LEN 16

struct {
    __uint(type, BPF_MAP_TYPE_HASH);
    __uint(max_entries, MAX_ENTRIES);
    __type(key, u32);          // tid
    __type(value, u64);        // start timestamp
} start SEC(".maps");

struct {
    __uint(type, BPF_MAP_TYPE_RINGBUF);
    __uint(max_entries, 256 * 1024);
} events SEC(".maps");

struct event {
    u64 ts;
    u32 pid;
    u32 tid;
    u64 latency_ns;
    int ret;
    unsigned int cmd;
    char comm[TASK_COMM_LEN];
};

// Filter: only trace ioctl from "camerad"
static __always_inline bool should_trace(void) {
    char comm[TASK_COMM_LEN];
    bpf_get_current_comm(&comm, sizeof(comm));
    // Compare first 7 chars: "camerad"
    return comm[0] == 'c' && comm[1] == 'a' && comm[2] == 'm' &&
           comm[3] == 'e' && comm[4] == 'r' && comm[5] == 'a' &&
           comm[6] == 'd';
}

SEC("tracepoint/syscalls/sys_enter_ioctl")
int trace_ioctl_enter(struct trace_event_raw_sys_enter *ctx) {
    if (!should_trace()) return 0;

    u32 tid = (u32)bpf_get_current_pid_tgid();
    u64 ts = bpf_ktime_get_ns();
    bpf_map_update_elem(&start, &tid, &ts, BPF_ANY);
    return 0;
}

SEC("tracepoint/syscalls/sys_exit_ioctl")
int trace_ioctl_exit(struct trace_event_raw_sys_exit *ctx) {
    if (!should_trace()) return 0;

    u32 tid = (u32)bpf_get_current_pid_tgid();
    u64 *tsp = bpf_map_lookup_elem(&start, &tid);
    if (!tsp) return 0;

    struct event *e = bpf_ringbuf_reserve(&events, sizeof(*e), 0);
    if (!e) {
        bpf_map_delete_elem(&start, &tid);
        return 0;
    }

    u64 now = bpf_ktime_get_ns();
    e->ts = now;
    e->pid = bpf_get_current_pid_tgid() >> 32;
    e->tid = tid;
    e->latency_ns = now - *tsp;
    e->ret = ctx->ret;
    bpf_get_current_comm(&e->comm, sizeof(e->comm));

    bpf_ringbuf_submit(e, 0);
    bpf_map_delete_elem(&start, &tid);
    return 0;
}

char LICENSE[] SEC("license") = "GPL";
```

**User-space loader (`ioctl_lat.c`):**

```c
#include <stdio.h>
#include <signal.h>
#include <bpf/libbpf.h>
#include "ioctl_lat.skel.h"  // auto-generated by bpftool gen skeleton

static volatile bool running = true;
static void sig_handler(int sig) { running = false; }

static int handle_event(void *ctx, void *data, size_t data_sz) {
    struct event *e = data;
    printf("%-8.3f %-16s pid=%-6d tid=%-6d latency=%.3f ms ret=%d\n",
           (double)e->ts / 1e9, e->comm, e->pid, e->tid,
           (double)e->latency_ns / 1e6, e->ret);
    return 0;
}

int main(void) {
    signal(SIGINT, sig_handler);

    // Open, load, and attach BPF programs
    struct ioctl_lat_bpf *skel = ioctl_lat_bpf__open_and_load();
    if (!skel) { fprintf(stderr, "Failed to load BPF\n"); return 1; }

    ioctl_lat_bpf__attach(skel);

    // Set up ring buffer polling
    struct ring_buffer *rb = ring_buffer__new(
        bpf_map__fd(skel->maps.events), handle_event, NULL, NULL);

    printf("Tracing camerad ioctl latency... Ctrl-C to stop.\n");
    while (running) {
        ring_buffer__poll(rb, 100);
    }

    ring_buffer__free(rb);
    ioctl_lat_bpf__destroy(skel);
    return 0;
}
```

**Build:**
```bash
# Generate vmlinux.h from kernel BTF
bpftool btf dump file /sys/kernel/btf/vmlinux format c > vmlinux.h

# Compile BPF program
clang -g -O2 -target bpf -D__TARGET_ARCH_arm64 -c ioctl_lat.bpf.c -o ioctl_lat.bpf.o

# Generate skeleton header
bpftool gen skeleton ioctl_lat.bpf.o > ioctl_lat.skel.h

# Compile user-space loader
gcc -o ioctl_lat ioctl_lat.c -lbpf -lelf -lz
```

---

## eBPF Program Types & Hook Points

eBPF programs attach to **different kernel subsystems**. Each program type has a specific context structure and set of available helpers.

### Tracing Program Types

| Program Type | Hook | Context | Use Case |
|---|---|---|---|
| `BPF_PROG_TYPE_TRACEPOINT` | Static kernel tracepoints | `struct trace_event_raw_*` | Stable, portable: `sched:sched_switch`, `irq:irq_handler_entry` |
| `BPF_PROG_TYPE_KPROBE` | Any kernel function | `struct pt_regs` (registers) | Deep driver tracing: GPU command submit, DMA engine |
| `BPF_PROG_TYPE_KRETPROBE` | Kernel function return | `struct pt_regs` | Measure function duration |
| `BPF_PROG_TYPE_FENTRY` / `FEXIT` | BPF trampoline (faster) | Function arguments directly | Low-overhead tracing (5.5+); preferred over kprobe |
| `BPF_PROG_TYPE_UPROBE` | User-space function | `struct pt_regs` | Trace TensorRT API calls, Python function entry |
| `BPF_PROG_TYPE_RAW_TRACEPOINT` | Raw tracepoint (no formatting) | Raw `struct` | Lowest overhead tracing |
| `BPF_PROG_TYPE_PERF_EVENT` | Hardware PMU / software events | `struct bpf_perf_event_data` | Profile cache misses, branch mispredictions |

### Networking Program Types

| Program Type | Hook | Use Case |
|---|---|---|
| `BPF_PROG_TYPE_XDP` | NIC driver (before sk_buff) | Line-rate packet filtering, DDoS mitigation |
| `BPF_PROG_TYPE_SCHED_CLS` (TC) | Traffic control ingress/egress | Packet modification, QoS, latency tagging |
| `BPF_PROG_TYPE_CGROUP_SKB` | Per-cgroup socket buffer | Container-level network policy |
| `BPF_PROG_TYPE_SK_MSG` | Socket message level | Transparent proxy, service mesh |
| `BPF_PROG_TYPE_SOCK_OPS` | TCP event callbacks | Per-connection congestion control |

### Security & Scheduling Program Types

| Program Type | Hook | Use Case |
|---|---|---|
| `BPF_PROG_TYPE_LSM` | Linux Security Module hooks | Runtime security policy enforcement |
| `BPF_PROG_TYPE_STRUCT_OPS` | Kernel struct ops replacement | Custom TCP congestion control, custom scheduler (sched_ext) |

### fentry/fexit vs. kprobe (Performance)

```
kprobe overhead:     ~100–200 ns per hit
fentry/fexit overhead: ~10–50 ns per hit  (4–10× faster)
```

`fentry`/`fexit` use **BPF trampolines** — the kernel patches the function prologue to call the eBPF program directly, avoiding the `int3` breakpoint mechanism that kprobes use. **Always prefer `fentry`/`fexit` on kernels ≥ 5.5.**

```c
// fentry — trace entry to the v4l2_ioctl kernel function
SEC("fentry/video_ioctl2")
int BPF_PROG(trace_v4l2_entry, struct file *file, unsigned int cmd, unsigned long arg) {
    // Direct access to function arguments — no pt_regs parsing needed
    u32 tid = (u32)bpf_get_current_pid_tgid();
    u64 ts = bpf_ktime_get_ns();
    bpf_map_update_elem(&start, &tid, &ts, BPF_ANY);
    return 0;
}

// fexit — trace return
SEC("fexit/video_ioctl2")
int BPF_PROG(trace_v4l2_exit, struct file *file, unsigned int cmd, unsigned long arg, int ret) {
    // 'ret' is the return value — available as the last argument
    // ... compute latency, emit event ...
    return 0;
}
```

---

## eBPF for AI System Observability

### 1. GPU Driver Latency Tracing

Trace NVIDIA GPU driver ioctls to measure command submission and completion latency on Jetson:

```bash
# Trace all ioctls to /dev/nvhost-gpu and /dev/nvgpu
bpftrace -e '
tracepoint:syscalls:sys_enter_ioctl
/comm == "modeld" || comm == "camerad"/ {
    @start[tid] = nsecs;
    @cmd[tid] = args->cmd;
}

tracepoint:syscalls:sys_exit_ioctl
/comm == "modeld" || comm == "camerad"/ {
    if (@start[tid]) {
        $lat = (nsecs - @start[tid]) / 1000;
        @latency_us = hist($lat);
        if ($lat > 1000) {
            printf("SLOW ioctl: %s tid=%d cmd=0x%x lat=%d us ret=%d\n",
                   comm, tid, @cmd[tid], $lat, args->ret);
        }
        delete(@start[tid]);
        delete(@cmd[tid]);
    }
}'
```

### 2. Scheduler Analysis for RT Inference Threads

```bash
# How long does modeld wait in the run queue before getting CPU time?
bpftrace -e '
tracepoint:sched:sched_wakeup /args->comm == "modeld"/ {
    @wake[args->pid] = nsecs;
}

tracepoint:sched:sched_switch /args->next_comm == "modeld"/ {
    if (@wake[args->next_pid]) {
        @runq_latency_us = hist((nsecs - @wake[args->next_pid]) / 1000);
        delete(@wake[args->next_pid]);
    }
}'
# If the histogram shows a long tail (>1ms), modeld is being preempted.
# Solutions: SCHED_FIFO, isolcpus, or SCHED_DEADLINE
```

### 3. DMA Transfer Profiling

```bash
# Trace DMA-related functions in the kernel
# Useful for understanding camera → model data pipeline
bpftrace -e '
kprobe:dma_map_page { @dma_map[comm] = count(); }
kretprobe:dma_map_page { @dma_map_lat = hist(nsecs - @start[tid]); }
'

# Memory-mapped I/O tracing for FPGA accelerators
bpftrace -e '
kprobe:pci_iomap { printf("PCI IOMAP: %s bar=%d\n", comm, arg1); }
'
```

### 4. Inference Pipeline End-to-End Latency

Build a custom tracer that measures the full pipeline: camera frame arrival → ISP processing → model inference → control output:

```bash
# Trace the full openpilot pipeline
bpftrace -e '
uprobe:/data/openpilot/selfdrive/modeld/modeld:run_model {
    @model_start[tid] = nsecs;
}
uretprobe:/data/openpilot/selfdrive/modeld/modeld:run_model {
    if (@model_start[tid]) {
        @model_latency_ms = hist((nsecs - @model_start[tid]) / 1000000);
        delete(@model_start[tid]);
    }
}'
```

### 5. Memory Allocation Tracking

```bash
# Track large allocations from AI processes (potential OOM debugging)
bpftrace -e '
tracepoint:kmem:mm_page_alloc /args->order >= 4/ {
    printf("%s pid=%d order=%d (%d KB)\n",
           comm, pid, args->order, (1 << args->order) * 4);
    @large_allocs[comm] = count();
}'

# Track CMA (Contiguous Memory Allocator) for camera DMA buffers
bpftrace -e '
tracepoint:cma:cma_alloc_start { @cma_start[tid] = nsecs; }
tracepoint:cma:cma_alloc_finish {
    @cma_latency = hist((nsecs - @cma_start[tid]) / 1000);
    delete(@cma_start[tid]);
}'
```

---

## XDP: Programmable Packet Processing

**XDP (eXpress Data Path)** runs eBPF programs at the **NIC driver level**, before the kernel allocates `sk_buff` structures. This enables packet processing at **millions of packets per second** with zero memory allocation overhead.

### XDP Actions

| Return Code | Action | Packets/sec (10GbE) |
|---|---|---|
| `XDP_DROP` | Drop packet at NIC | ~24 Mpps |
| `XDP_PASS` | Send to normal kernel stack | ~1–5 Mpps |
| `XDP_TX` | Bounce packet back out same NIC | ~20 Mpps |
| `XDP_REDIRECT` | Send to different NIC, CPU, or socket | ~15 Mpps |

### XDP for AV Telemetry Filtering

In autonomous vehicle deployments, the compute unit receives **high-bandwidth sensor data** (cameras, LiDAR, radar) over Ethernet. XDP can filter and prioritize this traffic **without kernel overhead**:

```c
SEC("xdp")
int xdp_sensor_filter(struct xdp_md *ctx) {
    void *data = (void *)(long)ctx->data;
    void *data_end = (void *)(long)ctx->data_end;

    struct ethhdr *eth = data;
    if (data + sizeof(*eth) > data_end) return XDP_DROP;

    // Pass camera frames (specific EtherType or VLAN)
    if (eth->h_proto == htons(0x88B5))  // IEEE 802.1 local experimental
        return XDP_PASS;

    // Drop non-essential traffic (telemetry, debug) under CPU pressure
    struct iphdr *ip = data + sizeof(*eth);
    if (data + sizeof(*eth) + sizeof(*ip) > data_end) return XDP_DROP;

    // High-priority: LiDAR data (UDP port 2368 — Velodyne default)
    if (ip->protocol == IPPROTO_UDP) {
        struct udphdr *udp = (void *)ip + sizeof(*ip);
        if ((void *)udp + sizeof(*udp) > data_end) return XDP_DROP;
        if (udp->dest == htons(2368)) return XDP_PASS;
    }

    return XDP_DROP;  // drop everything else
}
```

---

## sched_ext: Custom Schedulers with eBPF

Since Linux 6.12, **sched_ext** allows writing **CPU schedulers as eBPF programs** — a revolutionary capability for AI workload optimization.

```c
// Define scheduling callbacks as eBPF programs
SEC("struct_ops/enqueue")
void BPF_PROG(enqueue, struct task_struct *p, u64 enq_flags) {
    // Custom logic: if this is modeld, enqueue to high-priority DSQ
    if (is_inference_task(p))
        scx_bpf_dispatch(p, HIGH_PRIO_DSQ, SCX_SLICE_DFL, enq_flags);
    else
        scx_bpf_dispatch(p, DEFAULT_DSQ, SCX_SLICE_DFL, enq_flags);
}

SEC("struct_ops/dispatch")
void BPF_PROG(dispatch, s32 cpu, struct task_struct *prev) {
    // Drain high-priority DSQ first (inference tasks)
    scx_bpf_consume(HIGH_PRIO_DSQ);
    // Then default
    scx_bpf_consume(DEFAULT_DSQ);
}
```

This enables:
- **Inference-priority scheduling**: modeld always gets CPU before non-critical tasks
- **Core-affinity policies**: pin inference threads to specific cores without `isolcpus`
- **Latency-aware scheduling**: preempt background tasks when camera frame arrives
- **Custom load balancing**: distribute AI pipeline stages across cores based on workload characteristics

> **Key Insight:** sched_ext means you can prototype a custom scheduler for your AI workload, test it in production, and iterate — all without writing a kernel module or rebooting. For autonomous driving systems where scheduling determinism is safety-critical, this is transformative: you can build a scheduler that guarantees `modeld` never waits more than 100 µs for CPU time, and prove it with eBPF tracing.

---

## eBPF Tooling Ecosystem

| Tool | Level | Use Case |
|---|---|---|
| **bpftrace** | One-liners, scripts | Quick investigation, ad-hoc tracing |
| **BCC** (BPF Compiler Collection) | Python + embedded C | Pre-built tools (`runqlat`, `biolatency`, `offcputime`), custom tools |
| **libbpf** | C library | Production tools, CO-RE, maximum performance |
| **libbpf-rs** | Rust bindings | Safe eBPF tooling in Rust |
| **cilium/ebpf** (Go) | Go library | Kubernetes networking, cloud-native tools |
| **bpftool** | CLI utility | Inspect loaded programs, dump maps, generate skeletons |
| **Aya** | Rust eBPF framework | Write eBPF programs in Rust |

### bpftool Commands

```bash
# List all loaded eBPF programs
bpftool prog list

# Show details of a specific program
bpftool prog show id 42

# Dump JIT-compiled instructions
bpftool prog dump jited id 42

# List all BPF maps
bpftool map list

# Dump map contents
bpftool map dump id 5

# Generate vmlinux.h for CO-RE development
bpftool btf dump file /sys/kernel/btf/vmlinux format c > vmlinux.h

# Generate skeleton header from compiled BPF object
bpftool gen skeleton my_program.bpf.o > my_program.skel.h
```

---

## Hands-On Exercises

1. **Scheduler latency histogram:** Use `bpftrace` to measure run-queue latency for a specific process (e.g., a Python inference script running a model). Compare latency under default CFS vs. `SCHED_FIFO` (use `chrt -f 50` to set priority). Produce histograms for both and explain the difference.

2. **ioctl latency tracer:** Write a complete libbpf CO-RE program that traces all `ioctl` calls from a specified PID, records the ioctl command number and latency, and streams events via ring buffer to a user-space process that prints them. Build and test on your development machine.

3. **Memory allocation profiler:** Use `bpftrace` to trace `kmalloc` calls larger than 4096 bytes from a specific process. Count how many large allocations happen during model loading vs. inference. Propose optimizations to reduce allocation pressure during inference.

4. **XDP packet filter:** Write an XDP program that counts packets by IP protocol (TCP, UDP, ICMP) and drops ICMP packets. Attach it to a virtual interface (`veth`) and test with `ping` and `iperf3`. Verify that ICMP drops while TCP/UDP passes.

5. **Custom BCC tool:** Write a BCC tool that traces the latency of `read()` and `write()` syscalls from processes that have "model" in their name. Output a per-second summary with p50, p90, p99 latencies. Use this to characterize the I/O pattern during model loading.

6. **sched_ext exploration (Linux 6.12+):** Build and run one of the example `sched_ext` schedulers from the Linux kernel source (`tools/sched_ext/`). Compare the scheduling behavior of `scx_simple` vs. `scx_central` under a mixed workload (inference + background compilation). Measure tail latency for the inference process under each scheduler.

---

## Key Takeaways

| Concept | Why It Matters for AI Hardware |
|---|---|
| eBPF virtual machine | Programmable kernel — safe, JIT-compiled, native speed |
| Verifier | Production safety guarantee — buggy programs rejected at load, not runtime |
| BPF maps (ring buffer) | Zero-copy event streaming from kernel to user space |
| CO-RE (libbpf) | Write once, run on any kernel version — portable observability |
| fentry/fexit | 10× faster than kprobe — low-overhead production tracing |
| XDP | Packet processing at NIC level — millions of pps for sensor data |
| sched_ext | Custom CPU schedulers via eBPF — inference-priority scheduling |
| bpftrace one-liners | First responder tool — diagnose latency in 30 seconds |

---

## Resources

* **[BPF Performance Tools](https://www.brendangregg.com/bpf-performance-tools-book.html) by Brendan Gregg:** The definitive book on eBPF for system performance analysis. Covers 150+ BCC/bpftrace tools with real-world examples.
* **[Learning eBPF](https://isovalent.com/books/learning-ebpf/) by Liz Rice (O'Reilly):** Practical introduction to eBPF programming with libbpf and Go.
* **[libbpf-bootstrap](https://github.com/libbpf/libbpf-bootstrap):** Minimal scaffolding for building CO-RE eBPF programs — start here for custom tools.
* **[bpftrace Reference Guide](https://github.com/bpftrace/bpftrace/blob/master/docs/reference_guide.md):** Complete one-liner and script syntax.
* **[Brendan Gregg's eBPF Page](https://www.brendangregg.com/ebpf.html):** Updated collection of eBPF tools, use cases, and performance analysis methodology.
* **[Linux kernel: tools/sched_ext/](https://git.kernel.org/pub/scm/linux/kernel/git/torvalds/linux.git/tree/tools/sched_ext):** sched_ext example schedulers and documentation.
* **[Cilium eBPF Documentation](https://docs.cilium.io/en/stable/bpf/):** Comprehensive eBPF/XDP networking reference.
* **[BTF and CO-RE](https://nakryiko.com/posts/bpf-core-reference-guide/):** Andrii Nakryiko's guide to writing portable eBPF programs.
