# Lecture 16: NUMA Topology & HPC Memory Optimization

## Overview

Multi-socket servers are the standard platform for large-scale AI training and high-throughput inference. These machines do not treat all RAM as equal: memory physically attached to the CPU on socket 0 is fast to access from socket 0, but slow to access from socket 1. This non-uniform memory access — **NUMA** — means that where data lives in physical RAM is as important as what the data is. The mental model is geography: local memory is like a file on your desk, remote memory is like a file in another office building — same data, but retrieving it takes much longer. For an AI hardware engineer, NUMA awareness determines whether multi-GPU training saturates memory bandwidth or spends half its time waiting for cross-socket data transfers. Getting NUMA placement right is often the difference between achieving theoretical bandwidth and suffering a silent 2× slowdown.

---

## NUMA Architecture

**NUMA** (Non-Uniform Memory Access) is the memory topology of multi-socket servers. Each CPU socket (**NUMA node**) has a directly attached DRAM bank accessible at **local latency and full bandwidth**. Accessing DRAM attached to a different socket requires traversing the socket interconnect (Intel QPI/UPI, AMD Infinity Fabric, IBM X-Bus), incurring additional latency and reduced bandwidth.

```
2-Socket NUMA Server Topology

Socket 0 (NUMA Node 0)          Socket 1 (NUMA Node 1)
┌─────────────────────┐         ┌─────────────────────┐
│  CPU cores 0–23     │         │  CPU cores 24–47    │
│  L3 Cache (60MB)    │         │  L3 Cache (60MB)    │
│  Memory Controller  │         │  Memory Controller  │
└────────┬────────────┘         └────────┬────────────┘
         │                               │
    ┌────▼────┐   QPI/UPI link      ┌────▼────┐
    │ DDR5    │ <─────────────────> │ DDR5    │
    │ (192GB) │   ~100 GB/s         │ (192GB) │
    │ ~80 ns  │   ~150-200 ns       │ ~80 ns  │
    │ local   │   cross-socket      │ local   │
    └─────────┘                     └─────────┘
         │                               │
    ┌────▼────┐                     ┌────▼────┐
    │ GPU0    │                     │ GPU2    │
    │ GPU1    │                     │ GPU3    │
    │ (PCIe)  │                     │ (PCIe)  │
    └─────────┘                     └─────────┘
```

| Access type | Latency | Bandwidth (DDR5 2-socket Xeon) |
|-------------|---------|-------------------------------|
| Local DRAM | ~80 ns | ~200 GB/s per socket |
| Remote DRAM (1 QPI hop) | ~150–200 ns | ~100 GB/s |
| Remote DRAM (2 hops, 4-socket) | ~250+ ns | ~60 GB/s |

A process running on socket 0 that accesses data allocated on socket 1 pays the **remote penalty** on every cache miss — 2× to 3× the latency and half the bandwidth of local access.

> **Key Insight:** NUMA penalties are silent. There is no error, no warning, no visible failure — the program just runs slower. A 2× bandwidth reduction on model weight access directly translates to 2× longer matrix multiply time. `numastat -p` is the tool that makes this invisible problem visible.

### Topology Discovery

Before optimizing NUMA placement, you need to understand the topology of the specific machine:

```bash
numactl --hardware       # nodes, CPUs per node, memory per node, distance matrix
lstopo                   # graphical topology: sockets, cores, L3 cache, NUMA nodes, PCIe devices
numastat                 # per-node allocation hit/miss counters (system-wide)
cat /sys/devices/system/node/node0/distance   # raw NUMA distance factors
```

**Distance matrix**: local access is 10 (normalized); remote 1-hop typically 20–40; 2-hop 60–80.

A distance of 20 means cross-socket access has roughly **2× the latency** of local access. This number directly predicts performance loss for NUMA-unaware workloads.

---

## Linux NUMA Memory Policies

Linux provides several allocation policies that control which NUMA node a new page is allocated from. Understanding the default behavior is critical because it is often wrong for AI workloads.

### Default: First-Touch

The kernel allocates a page on the NUMA node of the CPU that first **faults** (accesses) it. This is efficient when the **initializing thread is the consuming thread**. It becomes a problem when a main thread on node 0 initializes a data structure later consumed exclusively by threads on node 1.

```
First-Touch Bug Pattern (common in AI frameworks):

Main thread (node 0):                Worker thread (node 1):
  weights = malloc(14GB)               [waiting for weights]
  memset(weights, 0, 14GB)  ←── pages allocated on node 0
  [signals worker thread]
                                       [begins inference]
                                       [reads weights]
                                           ↓
                                       Every cache miss crosses QPI!
                                       Effective bandwidth halved.
```

Critical pattern for AI workloads:
```c
/* Wrong: main thread (node 0) initializes, worker (node 1) consumes */
float *weights = malloc(model_size);
memset(weights, 0, model_size);   /* allocated on node 0 */

/* Correct: initialize on the thread that will use the data */
/* Pin worker thread to node 1, then fault the pages */
numa_run_on_node(1);
numa_set_membind(node1_mask);
float *weights = malloc(model_size);
memset(weights, 0, model_size);   /* allocated on node 1 */
```

> **Common Pitfall:** PyTorch DataLoader workers that run on node 0 initialize weight tensors on node 0's pages, even if the GPU and inference workers are on node 1. This creates a persistent, silent bandwidth penalty for the lifetime of the process. Always pin the initializing thread to the correct node before the first `memset` or tensor fill.

### Memory Policy Types

| Policy | Mode constant | Behavior |
|--------|--------------|----------|
| Default (first-touch) | `MPOL_DEFAULT` | Inherit from parent; local node preferred |
| Bind | `MPOL_BIND` | Strictly allocate only from named nodes; fail if full |
| Preferred | `MPOL_PREFERRED` | Prefer named node; fall back to others if needed |
| Interleave | `MPOL_INTERLEAVE` | Round-robin pages across named nodes |

### System Calls

```c
/* Per-process policy */
set_mempolicy(MPOL_INTERLEAVE, &all_nodes_mask, max_node);

/* Per-VMA policy (overrides process policy for this address range) */
mbind(addr, len, MPOL_BIND, &node0_mask, max_node, MPOL_MF_MOVE);
```

`MPOL_MF_MOVE`: migrate existing pages in the VMA to the target node immediately.

The `mbind()` call with `MPOL_MF_MOVE` is powerful: it moves **already-allocated pages** to the target node. This allows correcting a first-touch placement bug **without restarting the process**.

---

## numactl: Command-Line Policy Control

`numactl` applies NUMA policies to a process without modifying its source code. It is the primary tool for quick experiments and production deployments:

```bash
# Bind process to CPUs and memory of node 0 (GPU on node 0 socket)
numactl --cpunodebind=0 --membind=0 ./inference_server

# Interleave allocations across all nodes (bandwidth-bound all-reduce)
numactl --interleave=all ./allreduce_benchmark

# Prefer node 1 but allow fallback (mixed workload)
numactl --preferred=1 ./data_loader

# Interleave across nodes 0 and 1 only
numactl --interleave=0,1 ./matmul_benchmark
```

Using `--cpunodebind` and `--membind` together ensures both CPU execution and memory allocation happen on the **same node** — collocating compute and data to avoid cross-socket transfers.

---

## libnuma API (Programmatic Control)

For applications that need runtime topology awareness, `libnuma` provides programmatic access to NUMA policies:

```c
#include <numa.h>

int node = numa_node_of_cpu(sched_getcpu());   /* which node am I on? */
void *p = numa_alloc_onnode(size, node);       /* allocate on specific node */
numa_free(p, size);                            /* release */

numa_bind(nodemask);                           /* bind calling thread's CPU+memory */
numa_set_interleave_mask(&all_nodes);          /* interleave for future allocs */
numa_set_membind(&node_mask);                  /* memory-only bind */
```

The `numa_node_of_cpu(sched_getcpu())` pattern is the runtime equivalent of asking "which NUMA node is my current CPU core on?" — use it before allocating large buffers to ensure local placement.

---

## AutoNUMA Balancing

The kernel's **automatic NUMA balancer** periodically scans process page tables, temporarily removes present bits from PTEs, and re-faults pages to observe which CPU accesses which pages. **"Hot" pages migrate** to the NUMA node of the accessing CPU.

```
AutoNUMA mechanism:
1. Kernel scanner removes Present bit from PTEs (makes pages "not present")
2. Next access to that page causes a page fault
3. Kernel records: "CPU on node X faulted page Y"
4. If page Y is on node Z ≠ X, kernel migrates it to node X
5. Result: hot pages move to the node where they are accessed most
```

- Enable/disable: `echo 1 > /proc/sys/kernel/numa_balancing`
- Overhead: ~1% CPU for scanning; migration stalls the faulting thread briefly
- Beneficial for long-running processes with stable, localizable working sets
- Harmful for latency-sensitive inference: unpredictable migration bursts cause tail latency spikes

**Recommendation**: disable AutoNUMA (`numa_balancing=0` in kernel parameters) on real-time and latency-sensitive inference servers. Use explicit `mbind`/`numactl` instead.

> **Common Pitfall:** AutoNUMA is enabled by default. On a latency-sensitive inference server, the background page scanner causes periodic TLB shootdowns (to clear Present bits) and migration stalls (while pages move between nodes). These appear as unpredictable tail latency spikes in p99/p999 inference latency histograms. Disable AutoNUMA and set placement explicitly.

---

## Multi-GPU NUMA Affinity

GPU topology interacts directly with NUMA topology. In a 2-socket server, GPU cards are connected to **one socket's PCIe root complex**. Every CPU-to-GPU memory transfer from a remote socket crosses the socket interconnect, adding **60–100 ns of latency** per cache line.

```bash
nvidia-smi topo -m    # GPU-GPU and CPU-GPU interconnect topology
```

Example 2-socket output:
```
        GPU0  GPU1  GPU2  GPU3  CPU Affinity
GPU0     X    NV4   SYS   SYS   0-23
GPU1    NV4    X    SYS   SYS   0-23
GPU2    SYS   SYS    X    NV4   24-47
GPU3    SYS   SYS   NV4    X    24-47
```

`SYS` = traverses QPI/UPI (slow); `NV4` = NVLink 4 (fast). Bind the inference process to the CPU affinity range collocated with the target GPU to avoid `SYS` crossings.

The `SYS` links cross the QPI/UPI interconnect. For GPU0/GPU1, use CPU cores 0–23 (node 0). For GPU2/GPU3, use CPU cores 24–47 (node 1). Mixing these creates cross-socket PCIe traffic that halves effective H2D/D2H transfer bandwidth.

Correct binding for GPU0:
```bash
numactl --cpunodebind=0 --membind=0 ./inference_server --gpu=0
```

> **Key Insight:** `nvidia-smi topo -m` is the first command to run on a new multi-GPU server. The topology matrix tells you exactly which CPU cores and memory to bind to each GPU. This single configuration decision can double or halve effective memory bandwidth for H2D transfers.

---

## HPC Memory Optimization Patterns

With the NUMA concepts in place, there are two primary optimization patterns depending on whether the workload is latency-bound or bandwidth-bound.

### Binding (Latency-Bound Inference)

Pin all model weight allocations and the inference process to the **NUMA node local to the GPU**. Use first-touch by the pinned worker thread, or `mbind(MPOL_BIND)` after allocation.

```bash
echo 256 > /sys/devices/system/node/node0/hugepages/hugepages-2048kB/nr_hugepages
numactl --cpunodebind=0 --membind=0 ./inferenced
```

The `MPOL_BIND` policy ensures that if node 0 is out of memory, the allocation fails rather than silently falling back to node 1. This prevents surprise performance degradation under memory pressure.

### Interleaving (Bandwidth-Bound Training)

For GEMM operations that exceed single-node memory bandwidth, **interleave the weight matrix across all nodes**. Aggregate bandwidth = N × per-node bandwidth.

```bash
numactl --interleave=all ./training_process
```

Interleaving spreads pages round-robin across all nodes. A 4-node system with 200 GB/s per node delivers ~800 GB/s aggregate bandwidth with interleaving — versus 200 GB/s from a single node. This is the correct mode for distributed training all-reduce passes where the bottleneck is memory throughput.

> **Key Insight:** Binding and interleaving are opposites. Binding maximizes locality for latency-sensitive workloads (each access hits local DRAM). Interleaving maximizes total bandwidth for throughput-bound workloads (all DRAM banks contribute). Choosing the wrong mode for the workload type can cut performance in half.

### Huge Pages + NUMA

Pre-allocate huge pages per NUMA node for model weight buffers. HugeTLBFS supports per-node allocation via `/sys/devices/system/node/nodeN/hugepages/`. Combine with `madvise(MADV_WILLNEED)` to pre-fault pages local before inference starts.

```bash
# Allocate 256 huge pages on node 0 specifically
echo 256 > /sys/devices/system/node/node0/hugepages/hugepages-2048kB/nr_hugepages
```

Per-node huge page allocation guarantees the huge pages come from the correct NUMA node. System-wide huge page allocation has no locality guarantee.

---

## Memory Bandwidth Measurement

```bash
stream                                               # STREAM TRIAD benchmark per node
numastat -p $(pidof inferenced)                      # per-process NUMA hit/miss stats
perf stat -e cache-misses,LLC-load-misses ./workload # LLC miss rate (proxy for remote access)
perf mem record -- ./workload && perf mem report     # memory access latency profiling
```

`numastat -p` fields:
- `numa_hit`: pages allocated on the intended node
- `numa_miss`: pages allocated on a different node due to pressure
- `numa_foreign`: pages intended for this node but allocated elsewhere

A `numa_miss` rate above 5% is a strong signal that the process is experiencing NUMA placement failures — either due to memory pressure on the preferred node or incorrect policy configuration.

---

## Summary

| Policy | Behavior | Best for | Command | Caveat |
|--------|----------|----------|---------|--------|
| Default (first-touch) | Alloc on faulting CPU's node | General workloads | (implicit) | Wrong if initializer ≠ consumer |
| `MPOL_BIND` | Strict: only named nodes | Latency-sensitive inference | `numactl --membind=N` | OOM if node is full |
| `MPOL_INTERLEAVE` | Round-robin across nodes | Bandwidth-bound GEMM, all-reduce | `numactl --interleave=all` | Higher latency per access |
| `MPOL_PREFERRED` | Prefer node; allow fallback | Mixed workloads | `numactl --preferred=N` | May silently go remote |
| AutoNUMA | Migrate hot pages automatically | Long-running stable workloads | `echo 1 > .../numa_balancing` | Unpredictable latency spikes |

### Conceptual Review

- **What makes NUMA performance problems hard to detect?** There are no errors or warnings — the program produces correct results, just slower. Remote memory access is transparent to the application. Only `numastat -p` (showing `numa_miss` counts) or `perf mem report` (showing remote access latency) reveals the problem.

- **Why is the first-touch policy problematic for AI workloads?** AI frameworks often initialize tensors on a main or loader thread, then use them on inference worker threads. If the threads run on different NUMA nodes, all weight data is on the wrong node for the lifetime of the process.

- **When should you use `MPOL_INTERLEAVE` instead of `MPOL_BIND`?** When the workload is bandwidth-bound rather than latency-bound — specifically, when model weights are larger than a single node's memory bandwidth can stream at inference speed, or during distributed training all-reduce.

- **What does `nvidia-smi topo -m` tell you and why does it matter?** It shows the interconnect between each GPU pair and the CPU affinity of each GPU. `SYS` connections traverse the slow QPI/UPI link. This tells you exactly which CPU cores and memory nodes to bind to for each GPU to avoid cross-socket penalties.

- **Why should AutoNUMA be disabled on latency-sensitive inference servers?** AutoNUMA scans page tables and triggers migrations at unpredictable intervals, causing TLB shootdowns and brief thread stalls. These show up as p99/p999 tail latency spikes in inference latency histograms.

- **What is the difference between `numactl --membind` and `numactl --preferred`?** `--membind` fails (OOM) if the specified node cannot satisfy the allocation. `--preferred` falls back to other nodes silently. Use `--membind` when you require local allocation guarantees; use `--preferred` when best-effort locality is acceptable.

---

## AI Hardware Connection

- Multi-GPU inference servers require NUMA binding so that CPU pre-processing threads, pinned host memory, and the target GPU share the same PCIe root complex, avoiding QPI/UPI penalty on every H2D transfer
- First-touch initialization must occur on the thread pinned to the correct NUMA node; PyTorch DataLoader workers that initialize weight tensors on the wrong node silently degrade bandwidth by 2× for the lifetime of the process
- NUMA interleave across all nodes maximizes aggregate memory bandwidth for distributed training all-reduce passes where the bottleneck is memory throughput rather than latency
- `numastat -p` is the primary diagnostic for identifying remote memory access in a deployed inference pipeline; `numa_miss` values above 5% indicate a NUMA placement bug
- AutoNUMA must be disabled on latency-sensitive inference servers to prevent tail latency spikes caused by background page migration competing with inference kernel execution
- Jetson Orin has a single NUMA node (unified CPU-GPU DRAM); NUMA policies do not apply, but CPU-cluster affinity between the Cortex-A78 cluster and Cortex-X1 cores still affects LLC sharing and bandwidth
