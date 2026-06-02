# Lecture 8: Multi-Core Scheduling, CPU Affinity & isolcpus

## Overview

The core problem this lecture addresses is: when you have multiple CPUs and multiple tasks, how do you control *which task runs on which CPU*, and why does that control matter for performance? On a modern server or SoC, the Linux scheduler makes CPU placement decisions dozens of times per second. Those decisions are excellent for fairness but terrible for latency predictability — the scheduler can migrate a task between CPUs at any time, flushing its warm cache and adding hundreds of microseconds of overhead. The mental model to carry here is that of assigned seating on a plane: the default (open seating) works for most passengers, but for critical roles you need reserved, guaranteed seats. For an AI hardware engineer, controlling CPU placement is the difference between a neural network inference pipeline with predictable 2ms latency and one with random 10ms spikes caused by the OS moving threads around.

---

## SMP Scheduler Architecture

Linux SMP scheduler uses **per-CPU runqueues** (`struct rq`) to **reduce contention**:

- `load_balance()` triggered by: idle CPU detection, periodic `rebalance_domains()`, and explicit migration requests
- Load metric: sum of task weights (CFS nice-weighted) on each runqueue
- `migration/N` kernel threads execute the actual task moves on behalf of the scheduler
- Goal: equalize load across CPUs while respecting topology constraints (prefer same-core > same-package > same-NUMA node)

```
  SMP Load Balancing Overview:

  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐    ┌──────────────┐
  │   CPU 0      │    │   CPU 1      │    │   CPU 2      │    │   CPU 3      │
  │  runqueue    │    │  runqueue    │    │  runqueue    │    │  runqueue    │
  │  [T1][T2]   │    │  [T3][T4]   │    │  [ ]         │    │  [T5]        │
  │  [T5]       │    │             │    │              │    │              │
  └──────┬───────┘    └──────┬───────┘    └──────┬───────┘    └──────┬───────┘
         │                   │                   │                   │
         └───────────────────┴─── load_balance() ┴───────────────────┘
                         (equalizes task counts across CPUs)
```

While **load balancing** is beneficial for throughput workloads, it is **harmful for real-time and inference workloads** where cache warmth and latency predictability matter more than fairness.

---

## CPU Topology Hierarchy

```
Physical Package (Socket)
  └── DIE
        └── MC (physical core)
              └── SMT Thread (hyperthreading — 2 logical CPUs per core)
```

Linux models this as a `struct sched_domain` hierarchy. **Imbalance threshold and migration cost** increase at higher domain levels; the scheduler is **reluctant to migrate across NUMA boundaries**.

```
  Cache Sharing by Topology Level:

  ┌─────────────────────────────────────────────────┐
  │  NUMA Node 0 (Socket 0)                         │
  │  ┌───────────────────┐  ┌───────────────────┐   │
  │  │  Physical Core 0  │  │  Physical Core 1  │   │
  │  │  ┌────┐  ┌────┐  │  │  ┌────┐  ┌────┐  │   │
  │  │  │CPU0│  │CPU1│  │  │  │CPU2│  │CPU3│  │   │
  │  │  └────┘  └────┘  │  │  └────┘  └────┘  │   │
  │  │  Shared L1+L2    │  │  Shared L1+L2    │   │
  │  └─────────┬─────────┘  └─────────┬─────────┘   │
  │            └──────────┬────────────┘             │
  │                 Shared L3 (LLC)                  │
  └─────────────────────────────────────────────────┘
```

### Topology Inspection

```bash
lscpu                                                     # summary + per-CPU table
lscpu -e                                                  # extended per-CPU table
cat /sys/devices/system/cpu/cpu0/topology/core_id         # physical core ID
cat /sys/devices/system/cpu/cpu0/topology/physical_package_id  # socket ID
cat /sys/devices/system/cpu/cpu0/topology/thread_siblings # sibling SMT bitmask
cat /sys/devices/system/cpu/cpu0/topology/core_cpus_list  # all logical CPUs on this core
```

**SMT siblings** share L1/L2 caches and execution units. Inference workloads should **pin to physical cores** (one thread per core), not to both siblings of an SMT pair, to avoid **resource contention**.

> **Key Insight:** Two hyperthreads on the same physical core share the L1 instruction cache, L1 data cache, and integer/FP execution units. Running two competing inference threads on sibling SMT CPUs can be *slower* than running them on separate physical cores. Always verify with profiling before assuming hyperthreading helps.

---

## CPU Affinity

**Affinity mask**: bitmask specifying which CPUs a task is allowed to execute on. Without pinning, the scheduler is free to **migrate a task to any CPU at any time**.

```c
// System call interface
cpu_set_t mask;
CPU_ZERO(&mask);
CPU_SET(2, &mask); CPU_SET(3, &mask);  // allow execution only on CPUs 2 and 3
sched_setaffinity(pid, sizeof(mask), &mask);
sched_getaffinity(pid, sizeof(mask), &mask);
```

```bash
taskset -c 2,3 ./inference_app      # launch process with affinity to CPUs 2 and 3
taskset -cp 2,3 <pid>               # apply affinity to an already-running process
cat /proc/<pid>/status | grep Cpus_allowed
```

Benefits of affinity pinning:
- Eliminates scheduler-induced migrations; preserves L1/L2 cache warmth
- Avoids TLB flush cost on migration (~thousands of cycles on x86)
- Reduces latency variance; makes timing more predictable

| Mechanism | Kernel Interface | Granularity | Persistence | Tool |
|---|---|---|---|---|
| CPU affinity | `sched_setaffinity` | Per-task | Until process exits | `taskset` |
| `isolcpus` | Boot parameter | Per-CPU | Boot-time | kernel cmdline |
| cpuset cgroup | `/sys/fs/cgroup/cpuset` | Per-cgroup | Until reconfigured | `cgset`, k8s |
| `numactl` | `numactl --cpunodebind` | Per-NUMA node | Per-invocation | `numactl` |
| Intel CAT | `/sys/fs/resctrl` | Per-LLC-way | Until reconfigured | `pqos`, `resctrl` |

> **Common Pitfall:** `sched_setaffinity` restricts where a task *can* run, but it does not prevent other tasks from running on those same CPUs. If you pin your inference thread to CPU 2 but don't isolate CPU 2, OS daemons and kernel threads can still run there and evict your cache. Affinity pinning alone is not sufficient — it must be combined with `isolcpus` for full isolation.

---

## isolcpus — Removing CPUs from the General Scheduler

`isolcpus=` is a **kernel boot parameter**. CPUs listed are **removed from the general scheduling pool** permanently at boot. No task is placed on an isolated CPU unless **explicitly assigned** via `taskset` or `sched_setaffinity`.

Complementary parameters for full isolation:

```
isolcpus=2,3,4,5 nohz_full=2,3,4,5 rcu_nocbs=2,3,4,5 irqaffinity=0,1
```

| Parameter | Effect |
|---|---|
| `isolcpus=N` | Removes CPU N from general scheduler pool |
| `nohz_full=N` | Disables periodic scheduler tick on CPU N (tickless) |
| `rcu_nocbs=N` | Offloads RCU callbacks off CPU N to `rcuoc` kthreads |
| `irqaffinity=0,1` | Routes all hardware IRQs to CPUs 0 and 1 only |

```
  CPU Isolation Layout (4-core example):

  ┌───────────────────────────────────────────────────────┐
  │  CPU 0          CPU 1          CPU 2          CPU 3   │
  │  ┌──────────┐   ┌──────────┐   ┌──────────┐   ┌──────────┐ │
  │  │ OS/Kernel│   │ OS/Kernel│   │  RT/AI   │   │  RT/AI   │ │
  │  │ tasks    │   │ tasks    │   │  thread  │   │  thread  │ │
  │  │ IRQs ✓   │   │ IRQs ✓   │   │  IRQs ✗  │   │  IRQs ✗  │ │
  │  │ tick ✓   │   │ tick ✓   │   │  tick ✗  │   │  tick ✗  │ │
  │  │ RCU ✓    │   │ RCU ✓    │   │  RCU ✗   │   │  RCU ✗   │ │
  │  └──────────┘   └──────────┘   └──────────┘   └──────────┘ │
  │  ◄── General scheduler pool ──►◄───── Isolated ─────────►  │
  └───────────────────────────────────────────────────────┘
```

**Combined effect**: OS jitter reduced from ~200µs worst-case to **<10µs on isolated cores**. After boot, verify with `cyclictest` and confirm no IRQs are delivered to isolated CPUs via `cat /proc/interrupts`.

`tuna`: command-line tool that wraps `isolcpus` and IRQ affinity management for runtime CPU shielding configuration.

> **Key Insight:** `isolcpus` is a boot-time declaration that says "these CPUs are reserved." `nohz_full` turns off the timer tick (which would otherwise interrupt the isolated CPU every 1ms). `rcu_nocbs` removes RCU callbacks. Together they push the OS almost entirely off the isolated CPUs. Without all three, isolation is incomplete.

---

## CPU Frequency Scaling

**Variable CPU frequency** is another source of **latency jitter**. When a CPU transitions from a low power state to high frequency, instructions per second changes mid-task, making timing analysis unreliable.

| Governor | Behavior | Use Case |
|---|---|---|
| `performance` | Always at maximum frequency | Deterministic RT / inference latency |
| `powersave` | Always at minimum frequency | Battery-constrained idle systems |
| `schedutil` | Tracks CFS utilization signal | General throughput workloads |
| `ondemand` | Ramps on load, scales down on idle | Legacy desktop |

```bash
# Set performance governor on all CPUs
echo performance > /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor
# Or via cpupower
cpupower frequency-set -g performance
```

Variable frequency causes timing jitter; frequency transitions add up to 200µs of latency. **Always use `performance` governor** on RT and inference cores.

> **Common Pitfall:** On mobile SoCs (Jetson, Qualcomm), the `performance` governor may conflict with thermal throttling. Monitor `cat /sys/devices/system/cpu/cpu*/cpufreq/scaling_cur_freq` during load tests to confirm the frequency stays fixed. If thermal limits are hit, the CPU throttles regardless of governor setting.

---

## cpuset Cgroup Controller

While `isolcpus` works at boot time, **`cpuset` cgroups** provide a **runtime mechanism** to assign process groups to CPU subsets **without a reboot**.

```bash
mkdir /sys/fs/cgroup/cpuset/inference
echo "4-7" > /sys/fs/cgroup/cpuset/inference/cpuset.cpus   # assign CPUs 4-7 to this group
echo "0"   > /sys/fs/cgroup/cpuset/inference/cpuset.mems   # bind to NUMA node 0 memory
echo <pid> > /sys/fs/cgroup/cpuset/inference/cgroup.procs  # add process to the group
```

This creates a CPU partition: all threads in the `inference` cgroup are scheduled only on CPUs 4–7, and their memory allocations come from NUMA node 0.

Kubernetes uses `cpu_manager_policy=static` to allocate exclusive CPUs to **Guaranteed QoS** pods via cpuset internally. Pair with `topologyManagerPolicy=single-numa-node` to align CPU and GPU PCIe topology for GPU pods.

---

## Cache Topology and Intel CAT

CPU affinity and isolation control which threads run where, but they don't prevent **shared cache pollution**. Even a pinned inference thread shares the **L3 LLC** with OS daemons running on other cores. **Intel CAT** addresses this.

### Cache Sharing

- L1 (32–64 KB), L2 (256 KB–1 MB): private per physical core
- L3 / LLC (8–64 MB+): shared across all cores in a package
- SMT siblings share L1 + L2 + execution units; avoid co-locating competing workloads on same physical core

```bash
perf stat -e cache-misses,cache-references ./inference    # measure LLC miss rate
```

### Intel Cache Allocation Technology (CAT / RDT)

Partitions LLC ways between workloads using Capacity Bitmasks (CBM). Prevents co-located OS daemons from evicting hot model weights.

```
  Intel CAT LLC Partitioning (16-way cache):

  ┌───┬───┬───┬───┬───┬───┬───┬───┬───┬───┬───┬───┬───┬───┬───┬───┐
  │ 0 │ 1 │ 2 │ 3 │ 4 │ 5 │ 6 │ 7 │ 8 │ 9 │10 │11 │12 │13 │14 │15 │  LLC ways
  └───┴───┴───┴───┴───┴───┴───┴───┴───┴───┴───┴───┴───┴───┴───┴───┘
  ◄───────────────────────────►◄──────────────────────────────────►
      Inference group (ways 0-7)         OS/other (ways 8-15)
      CBM: 0x00FF                        CBM: 0xFF00
```

```bash
mount -t resctrl resctrl /sys/fs/resctrl
mkdir /sys/fs/resctrl/inference
# Allocate LLC ways 0–3 (4 ways) exclusively to inference group
echo "L3:0=0x000f" > /sys/fs/resctrl/inference/schemata
# 0x000f = binary 0000 0000 0000 1111 = ways 0,1,2,3
echo <pid> > /sys/fs/resctrl/inference/tasks
```

Management tool: `pqos` (Intel RDT OSS tools). Also supports Memory Bandwidth Allocation (MBA) to limit DRAM bandwidth consumed by background workloads.

> **Key Insight:** Without Intel CAT, a single `kworker` or `systemd-journald` burst can flush 30–50% of the LLC. The next inference forward pass then pays L3-miss penalties on every weight access. CAT gives model weights a reserved cache partition that OS activity cannot touch.

---

## NUMA Affinity

On multi-socket servers and SoCs with asymmetric memory topology, the **NUMA node from which memory is allocated** matters as much as which CPU runs the task. **Cross-NUMA memory access** adds latency on every cache miss.

```bash
# Pin process to socket 0 CPUs + socket 0 memory
numactl --cpunodebind=0 --membind=0 ./inference

# Show CPU <-> GPU PCIe topology
nvidia-smi topo -m

# Show per-process NUMA access stats
numastat -p <pid>

# Show NUMA hit/miss counters
numastat
```

**AutoNUMA** (`/proc/sys/kernel/numa_balancing`): **disable on latency-sensitive inference nodes**. Automatic page migration causes **TLB shootdown IPIs** that add jitter spikes.

> **Common Pitfall:** If the GPU PCIe root port attaches to socket 0 and your inference process allocates memory on socket 1 (the OS default when socket 0 is under pressure), every DMA transfer suffers cross-NUMA latency. Always confirm NUMA binding with `numastat -p <pid>` after launching inference and verify the `Numa_miss` counter is near zero.

---

## Summary

| Mechanism | Kernel/User? | Granularity | Persistence | Tool |
|---|---|---|---|---|
| `sched_setaffinity` | Kernel syscall | Per-task | Process lifetime | `taskset` |
| `isolcpus` | Kernel boot param | Per-CPU | Boot-time | kernel cmdline |
| `nohz_full` | Kernel boot param | Per-CPU | Boot-time | kernel cmdline |
| `rcu_nocbs` | Kernel boot param | Per-CPU | Boot-time | kernel cmdline |
| cpuset cgroup | Kernel cgroup | Per-cgroup | Dynamic | `cgset`, kubectl |
| `cpufreq performance` | Kernel driver | Per-CPU | Dynamic | `cpupower` |
| Intel CAT | Kernel resctrl | Per-LLC-way | Dynamic | `pqos` |
| `numactl` | Userspace wrapper | Per-NUMA node | Per-invocation | `numactl` |

### Conceptual Review

- **Why is CPU affinity alone not enough to eliminate jitter?** Affinity prevents the target task from migrating away, but it does not prevent other tasks from running on the pinned CPU. `isolcpus` must be used to exclude all other tasks from the CPU.
- **What is the difference between `isolcpus` and `sched_setaffinity`?** `isolcpus` removes a CPU from the kernel's general scheduler pool — the scheduler will never place an ordinary task there. `sched_setaffinity` restricts one specific task to a set of CPUs, but those CPUs remain in the general pool.
- **Why disable `nohz_full` on non-isolated CPUs?** The timer tick is needed for accurate scheduler accounting on CPUs running multiple tasks. Only isolated CPUs benefit from tickless operation.
- **Why avoid SMT siblings for competing workloads?** Two hyperthreads on the same physical core share L1 cache and execution units. A competing thread evicts the inference thread's cache lines and competes for FP units, causing unpredictable slowdowns.
- **What problem does Intel CAT solve that CPU affinity does not?** Affinity controls which CPU runs which task. CAT controls which LLC cache *ways* are available to which task. Even with affinity pinning, an OS daemon on a different core can evict model weights from the shared LLC.
- **Why disable AutoNUMA on inference nodes?** AutoNUMA migrates pages between NUMA nodes to improve locality, but the migration itself causes TLB shootdown IPIs on all CPUs that map the migrated page. This adds latency spikes that are invisible to the task but visible in latency histograms.

---

## AI Hardware Connection

- `isolcpus=4-11 nohz_full=4-11 rcu_nocbs=4-11` on Jetson Orin dedicates 8 ARM cores exclusively to AI inference threads; the OS and sensor drivers run on cores 0–3, preventing OS jitter from entering the inference latency budget
- Kubernetes `cpu_manager_policy=static` uses cpuset cgroups to give TensorRT inference pods exclusive physical CPU cores, eliminating noisy-neighbor scheduler interference in multi-tenant inference clusters
- CPU affinity pinning for ROS2 real-time callback groups ensures deterministic execution of LiDAR processing and motion control nodes; scheduler migration would invalidate WCET measurements
- NUMA-aware memory binding (`numactl -m 0`) reduces first-inference latency on multi-socket servers where the GPU PCIe root port attaches to socket 0; model weights in socket-1 memory incur cross-NUMA fetch overhead on every forward pass
- Intel CAT partitions LLC so OS daemons cannot evict hot model layer weights during inference; without CAT, a `kworker` burst can flush 30–50% of LLC, causing a spike of LLC-miss latency at the start of the next forward pass
- `cpufreq performance` governor is non-negotiable on AV edge compute: frequency ramp-up delay from a low-power C-state can add 100–200µs to the first GPU kernel launch after an idle period
