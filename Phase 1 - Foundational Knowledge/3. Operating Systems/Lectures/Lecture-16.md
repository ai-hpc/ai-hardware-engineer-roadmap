# Lecture 16: NUMA placement and memory access cost

NUMA means that access cost can depend on which CPU or device uses which memory. The useful question is not simply “which socket owns this process?” It is where its threads run, where its pages reside, and which paths its transfers use.

**Prerequisites:** [allocation](Lecture-14.md) and [DMA](Lecture-15.md). Plan 60–90 minutes for topology and placement inspection; hardware performance comparison is a separate experiment. A single-node machine can complete the reasoning exercises without pretending to measure remote-node behavior.

## 1. Discover nodes instead of assuming one per socket

A socket can expose several NUMA nodes; memory-only nodes and other heterogeneous arrangements also exist. A node is a locality/placement abstraction, not a synonym for physical CPU socket. NUMA locality can affect latency, available bandwidth, and contention, but there is no universal “remote memory is twice as slow” factor. [Linux physical memory organization](https://docs.kernel.org/mm/physical_memory.html).

Read-only Linux inventory:

```bash
lscpu
numactl --hardware
numactl --show
cat /proc/self/status
```

`numactl` may require the distribution’s package. In `/proc/self/status`, compare allowed CPUs and memory nodes with the machine-wide topology; a container or batch job may have a restricted subset. The command observes its own process, not another worker automatically.

A dependency-free Python alternative:

```python
from pathlib import Path
import os
print('allowed CPUs:', sorted(os.sched_getaffinity(0)))
for node in sorted(Path('/sys/devices/system/node').glob('node[0-9]*')):
    print(node.name, 'CPUs:', (node / 'cpulist').read_text().strip(),
          'distance row:', (node / 'distance').read_text().strip())
```

Distance values are relative topology information. A reported local distance 10 and remote distance 20 do not establish measured nanoseconds, achievable bandwidth, or an exact 2× application slowdown.

## 2. Separate CPU affinity from memory policy

CPU affinity restricts where a thread may execute. Memory policy influences eligible/preferred placement of applicable allocations. Setting only one does not automatically align the other. Placement can also be constrained by cpusets, mapping type, existing pages, and kernel policy.

| Policy idea | Use to investigate | Limit |
|---|---|---|
| Default/local behavior | Allocation near the faulting CPU under applicable policy | Fallback and existing/shared mappings matter |
| Preferred node(s) | Favor selected locality | Fallback is allowed under the policy |
| Bind | Restrict eligible allocation nodes | Can fail under pressure; does not create memory |
| Interleave | Distribute pages across eligible nodes | Does not guarantee ideal aggregate bandwidth |

Policies have thread, process inheritance, and mapping scopes. Consult the exact API and current installed version. [Linux NUMA memory policy](https://docs.kernel.org/admin-guide/mm/numa_memory_policy.html), [numactl project and manual](https://github.com/numactl/numactl).

**First-touch example:** one thread allocates an untouched anonymous range; a worker pinned to a selected CPU writes it. Those write faults commonly establish private page backing under the applicable policy. If the main thread first initializes the whole range elsewhere, the later worker may inherit remote pages. Reading a demand-zero mapping and writing it are not necessarily equivalent allocation events.

Setting a new policy after pages already exist does not automatically move all of them. Migration requires the relevant flags/API, eligibility, permissions, and successful results. Record the placement after changing policy instead of assuming success.

## 3. Inspect placement without confusing it with traffic

`/proc/PID/numa_maps` describes mapping policy and node page counts where available. `numastat -p PID` summarizes per-process memory placement; it is not a direct counter of every remote CPU load. System NUMA allocation statistics also differ from measured cache-miss traffic.

Use the actual numeric PID of your test process. Record the mapping of interest, its size, page size, and node counts before and after initialization. Explain any shared/file-backed mappings separately from the anonymous working set.

**Worked placement calculation:** a 64 MiB range with 4 KiB pages has `64 × 2^20 / 4096 = 16,384` pages. If its placement report shows 12,288 on node 0 and 4,096 on node 1, that is 75% and 25% of those pages. It does not prove 25% of accesses are remote: access frequency and the executing CPUs are additional inputs.

**Check:** a 1 MiB hot subset may dominate accesses even if most of the allocated 64 MiB is elsewhere. Page count and access count answer different questions.

## 4. Design a controlled placement experiment

Choose a CPU-only workload with a known result and a working set larger than the relevant caches. Separate initialization from the timed region. Record thread placement, memory placement, compiler/build, problem size, repeated-run method, and background load.

Documented launch template, after selecting actual permitted node IDs:

```text
numactl --cpunodebind=NODE --membind=NODE ./YOUR_TEST_PROGRAM
```

The uppercase fields are placeholders; the course does not supply a fabricated universal node number or benchmark result. Compare local placement, a deliberate remote placement on a suitable multi-node machine, and an interleaved policy if the workload can use it. Keep the work and correctness check identical.

Publish throughput and latency distributions alongside placement evidence. A halved memory bandwidth does not necessarily double matrix-multiply runtime: compute intensity, cache reuse, overlapping work, and other bottlenecks affect the result. A compute-bound GEMM may be insensitive to the particular memory change.

## 5. Include GPUs and other devices in the topology

A PCIe device connects through a specific root complex/switch path. Its host buffers and feeding CPU threads can therefore have different locality costs. On supported NVIDIA server systems, `nvidia-smi topo -m` helps describe those relationships; it is not a replacement for measured transfer behavior or a universal Jetson command.

Keep CPU preprocessing, host-buffer initialization, pinning/registration, GPU transfer, and computation as separate stages. A remote host buffer may add traffic across an interconnect, but the penalty depends on the platform and path. Do not claim that every transfer pays a fixed number of nanoseconds per cache line or halves bandwidth.

PyTorch DataLoader workers commonly prepare input batches; do not assume they initialize all model weights. Identify the actual allocation/initialization call sites. A pinned buffer’s physical placement and transfer contract need evidence, and pinning does not automatically repair locality.

## 6. Treat migration and huge pages as experiments

Automatic NUMA balancing can sample access and migrate work/pages under the kernel’s policy. It may improve a workload or introduce overhead. Record whether it is enabled and compare under controlled conditions before changing a host-wide setting. “Always disable it for inference” is not a measured conclusion.

Huge pages reduce some translation overhead while changing allocation, fragmentation, and placement constraints. Reserving a pool on a node is not sufficient proof that a particular mapping uses that pool. `MADV_WILLNEED` is advice, not an unconditional prefault or locality guarantee. Inspect actual mapping page size and node placement. [Linux huge-page documentation](https://docs.kernel.org/admin-guide/mm/hugetlbpage.html).

**Completion:** submit a topology/allowed-mask record, the 16,384-page calculation, one mapping’s placement, and a controlled performance experiment plan or actual raw results. Distinguish page placement, remote accesses, interface bandwidth, and application time. Continue with [the device model](Lecture-17.md).
