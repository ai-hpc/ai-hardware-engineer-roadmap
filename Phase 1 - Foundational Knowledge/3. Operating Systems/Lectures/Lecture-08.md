# Lecture 8: CPU Placement, Affinity, and Isolation

## The problem: eligibility is different from exclusivity

A perception worker is allowed on logical CPUs `{2, 3}`. It may migrate between them. A single-CPU mask `{2}` removes that choice while CPU 2 remains available, but other threads may also run on CPU 2. Neither mask reserves cache capacity, memory bandwidth, GPU execution, or a deadline.

Pinning can help locality and repeatability; it can also strand work behind a busy CPU. Compare measured workloads instead of assuming all scheduler migration is harmful. Migration does not literally flush every cache, and its cost depends on topology and working set.

## Inspect actual topology

```bash
lscpu
lscpu -e
cat /proc/self/status
cat /proc/self/cgroup
```

Read `Cpus_allowed_list` and `Mems_allowed_list` in the status output. CPU numbers identify logical CPUs, not physical core numbers. Core IDs may repeat across sockets. SMT sibling relationships, cache sharing, heterogeneous CPU capacities, and NUMA nodes are platform properties; a socket need not equal a NUMA node, and a core need not have two SMT threads.

For an available CPU, inspect its directory under `/sys/devices/system/cpu/cpuN/topology/`, especially `thread_siblings_list`, `core_id`, and `physical_package_id`. Missing files are a platform limitation to record. Never copy a Jetson CPU range from a different module's core count.

**Worked placement:** suppose CPUs 0/4 share one physical core and 1/5 share another. Two compute workers on 0 and 4 compete for that core's resources. Moving one to 1 changes the physical resources available; moving it to 5 uses the same second core. Whether two SMT threads improve throughput still needs measurement.

## Affinity is per thread

Linux `sched_setaffinity` applies to a thread ID. `taskset` can launch a process with an inherited mask, but existing threads may already have different masks; `taskset -a` addresses all threads of a process when supported. Libraries can create workers or change affinity later. Verify each relevant thread rather than inspecting only the process leader.

The effective set is constrained by online CPUs, cpuset restrictions, and other platform rules. An empty permitted intersection fails. Always check return values and read back the result. The [Linux affinity API](https://man7.org/linux/man-pages/man2/sched_setaffinity.2.html) explains these restrictions and inheritance.

This Linux/Python experiment changes only its own process and restores its mask; it needs no root access. It is a functional placement exercise, not a benchmark:

```python
import os

original = os.sched_getaffinity(0)
chosen = min(original)
try:
    os.sched_setaffinity(0, {chosen})
    actual = os.sched_getaffinity(0)
    assert actual == {chosen}
    print("requested", chosen, "allowed", sorted(actual))
finally:
    os.sched_setaffinity(0, original)
```

For a compiled version with malformed-mask tests and checked thread APIs, run [rt-demo](demos/rt-demo/README.md). Never assume CPU 0 is available inside a container or batch allocation.

## Separate the isolation mechanisms

| Mechanism | What it controls | What it does not establish |
|---|---|---|
| Affinity mask | Where a particular thread may run | Exclusive use of those CPUs |
| cpuset controller | CPU and memory-node eligibility for a group | Complete IRQ/firmware/device isolation |
| Isolated cpuset partition, when supported/configured | Scheduler load-balancing participation for that partition | Absence of every kernel activity |
| `nohz_full` | Adaptive tick suppression under required conditions | No interrupts, timers, or scheduler activity |
| `rcu_nocbs` | Callback offloading mechanism | Correct placement of every offload thread by itself |
| IRQ affinity | Placement for interrupts that honor that control | Relocation of all managed or per-CPU interrupts |

The kernel documents `isolcpus` as deprecated in favor of cpusets for relevant isolation use cases. Its flags have different meanings; `domain` isolation cannot simply be undone through the same boot parameter at runtime. Do not describe it as deleting CPUs from the scheduler. Explicitly eligible tasks and unavoidable kernel activity can still run. Consult [kernel parameters](https://docs.kernel.org/admin-guide/kernel-parameters.html) and [adaptive ticks](https://docs.kernel.org/timers/no_hz.html).

Allocate housekeeping capacity for device completion, networking, storage, RCU, and management. If a control task relies on a callback that cannot get CPU time, isolating the control task more aggressively can break progress. Reserve capacity using the workload's dependencies, then observe interrupt counters and thread placement under load.

## cgroup v2: requested versus effective resources

The old `/sys/fs/cgroup/cpuset/...` layout is a cgroup v1 example; do not assume it on a v2 host. On v2, the controller must be available and enabled in the relevant hierarchy. Delegation, ancestor constraints, partition state, and process placement rules matter.

Inspect the hierarchy before any configuration:

```bash
findmnt -t cgroup2
cat /sys/fs/cgroup/cgroup.controllers
cat /sys/fs/cgroup/cgroup.subtree_control
```

For your actual group, compare `cpuset.cpus` with `cpuset.cpus.effective`, and `cpuset.mems` with `cpuset.mems.effective`. A configured string is not enough to demonstrate that the group obtained those resources. A normal cpuset can overlap other groups; an isolated partition has additional validity and exclusivity rules. Use the [cgroup v2 cpuset documentation](https://docs.kernel.org/admin-guide/cgroup-v2.html#cpuset) or your service manager's supported configuration path. This lesson does not modify the host hierarchy.

**Check:** a child requests CPUs 2–5, but its parent's effective set is 2–3. Does the child obtain four CPUs? **Answer:** no; child eligibility cannot exceed the ancestor's restrictions. Inspect effective values and configuration errors.

## Frequency, thermal state, and shared resources

A generic `performance` governor requests the highest frequency allowed by its policy. It does not override firmware, thermal throttling, power limits, or every driver's autonomous behavior. `powersave` semantics also depend on the driver. Frequency and idle-state exit are related but distinct mechanisms; governor selection alone does not remove idle exit latency. See [CPU performance scaling](https://docs.kernel.org/admin-guide/pm/cpufreq.html).

Run a warm steady-state workload and record temperature, power limits, effective frequency observations, and task latency together. Do not treat a brief cool-board result as sustained performance. CPU pinning also does not stop another core from consuming shared memory bandwidth.

On supported systems, resource-control interfaces can restrict cache allocation and monitor bandwidth. Nonoverlapping allocation masks require correct configuration across the relevant groups; they do not pin arbitrary model weights permanently in cache or guarantee a miss rate. Generic `perf` events such as `cache-misses` must be interpreted using the actual PMU event mapping, not relabeled as precise LLC statistics without evidence. Read [resctrl](https://docs.kernel.org/filesystems/resctrl.html).

For memory placement, continue to [Lecture 16](Lecture-16.md): `numastat -p` describes placement/accounting, not a trace of every remote memory access. CPU pinning alone does not move existing pages.

## A controlled learning experiment

Run the same CPU workload with its default allowed set, one selected logical CPU, and two separate physical cores if available. Keep input, software, thread count, and measurement duration fixed. Save effective masks, topology, wall time, CPU time, migrations/context switches when observable, and output correctness. Repeat under a documented competing load.

Do not require pinning to win. A useful result explains why it helped, hurt, or made no measurable difference. For a robot pipeline, also record queue age and missed deadlines: a throughput improvement can hide older sensor data.
