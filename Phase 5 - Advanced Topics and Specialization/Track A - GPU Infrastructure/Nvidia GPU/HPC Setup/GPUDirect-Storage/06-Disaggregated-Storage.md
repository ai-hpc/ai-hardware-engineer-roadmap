# 06 — Shared storage: capacity, bandwidth, and failure domains

Disaggregated storage lets compute and storage capacity grow independently, but
it also introduces shared resources and network dependencies. Design it from
workload measurements rather than a promise of linear scaling.

**Prerequisites:** [data path](01-Architecture-and-Data-Path.md) and
[measurement](05-Performance-Tuning.md). **Artifact:** a storage plan with capacity,
bandwidth, redundancy, and failure assumptions. Allow 1–2 hours for analysis;
network deployment is an optional hardware exercise.

## What remote storage changes

```text
application -> filesystem/block layer -> transport -> storage target -> media
                                      -> network and target failure domains
```

NVMe over Fabrics extends NVMe over a fabric; its transports and implementation
support depend on the selected stack. The official
[NVM Express specifications](https://nvmexpress.org/specifications/)
define the protocol families. A file-reading application can often retain its
logical API while the underlying path changes. That does not guarantee unchanged
latency, supported GDS mode, error behavior, or code/configuration requirements.

Local caching and shared storage can coexist. The useful comparison is among
actual workload designs: dataset copies, cache reuse, access concurrency,
checkpoint write bursts, recovery, and operational cost.

## A capacity and bandwidth plan you can calculate

The following **hypothetical inputs** are teaching values, not OpenFlex or another
product's specifications:

- 32 GPU workers need 10 decimal GB/s each during an I/O-heavy phase;
- each storage unit supplies a measured 75 GB/s useful throughput for that pattern;
- each unit has 300 TB of usable capacity after its internal protection;
- the dataset is 500 TB and an additional replication policy requires 1.5× storage.

Bandwidth requires `ceil(320 / 75) = 5` units. Capacity requires
`ceil(500 × 1.5 / 300) = 3`. Under these assumptions the initial count is 5.
It is only feasible if the network, client paths, placement, and software can
carry the demand. Avoid applying a redundancy multiplier twice if it is already
included in the usable-capacity input.

```python
from math import ceil

workers = 32
worker_GB_s = 10.0
unit_GB_s = 75.0
usable_unit_TB = 300.0
stored_TB = 500.0 * 1.5
units = max(ceil(workers * worker_GB_s / unit_GB_s),
            ceil(stored_TB / usable_unit_TB))
print(units)  # 5, under the stated assumptions
```

Use `ceil`, not `int(x) + 1`: the latter overallocates when the requirement is an
exact integer. Check that capacities and throughput inputs are positive before
turning this calculation into a reusable tool.

**Follow-up:** if one unit must fail without missing the same peak bandwidth target,
six units are an initial N+1 arithmetic candidate. That still requires suitable
data placement, failover behavior, network headroom, and measured degraded-mode
performance. Simply buying six independent units does not establish resilience.

## Why linear scale-out can stop

Storage devices, controller paths, switch uplinks, clients, metadata services,
CPU processing, and the application can all constrain scaling. Adding two equal
storage units doubles their nominal aggregate capacity; it does not guarantee
that each training job sees twice the throughput.

Compare these cases with the same dataset and correctness check:

| Experiment | What it separates |
|---|---|
| One client, one target | Baseline path and integrity |
| Several clients, one target | Target and shared-link contention |
| One client, several targets | Client-side aggregation limits |
| Several clients, several targets | Placement, fairness, and scale-out behavior |
| One controlled unavailable path | Recovery behavior and remaining capacity |

Report the slowest worker and completion-time distribution when synchronized
training makes stragglers important. Aggregate bandwidth alone can hide one worker
that repeatedly stalls an entire step.

## Network configuration belongs to the selected stack

RoCE deployments require deliberate congestion, queue, priority, and loss-recovery
configuration. Do not treat “lossless network” as an unconditional guarantee or
copy a switch CLI for a different network operating system. Use the selected NIC,
firmware, switch, and driver documentation, and validate the agreed traffic class
end to end.

For a controlled lab, begin with read-only inventory and the documented diagnostic
commands. Then establish transport connectivity, a mounted test path, payload
integrity, and finally the intended GDS mode. Keep network throughput and
application/file-I/O throughput as distinct measurements.

Use the official [nvme-cli project](https://github.com/linux-nvme/nvme-cli) for
command semantics and the [GDS configuration guide](https://docs.nvidia.com/gpudirect-storage/configuration-guide/index.html)
for supported storage-path testing. Discovery, connection, formatting, and
multi-host mounting are different operations; this lesson does not require
formatting a device or changing a production network.

## Apply it to robot learning

A robot-learning dataset may contain many small compressed images, metadata,
actions, and episode boundaries. A sequential tensor-read benchmark omits file
lookup, decoding, sample grouping, and random access. Measure the loader that
preserves episode identity and timestamp alignment. A faster loader that pairs
an image with the wrong action is incorrect.

**Independent task:** design a local-cache plus shared-storage loader for a
recorded-data experiment. Specify the cache key, dataset revision, eviction and
invalidation rules, behavior after a partial download, and checksum verification.
Measure repeated-epoch and first-epoch behavior separately if hardware is available.

**Self-check with answer:** does shared storage eliminate failure? No. It changes
which failures are coupled. Storage and network failure can affect multiple compute
nodes; replication, failover, backups, and recovery tests address different risks.

**Exit criteria:** calculate the two sizing constraints, identify at least one
shared bottleneck, and distinguish analytical sizing from measured support and
performance. [Return to the course index](README.md).
