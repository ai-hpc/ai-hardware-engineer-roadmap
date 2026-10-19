# 05 — Measure the storage path before tuning it

Optimize useful input delivery to the application. A high read bandwidth is only
helpful if it removes a real bottleneck without corrupting data or delaying the
consumer.

**Prerequisites:** [correct API use](04-libcufile-API.md), basic bandwidth units,
and a declared workload. **Artifact:** an integrity-checked comparison with raw
samples and a justified bottleneck explanation. The examples below are analytical
exercises; no GPU/storage throughput was measured in this course revision.

## 1. Build a bound from the actual topology

For a single flow, a first bound is the minimum capacity of the resources it
must traverse. Shared resources constrain the *aggregate* of competing flows.
Do not divide a system-level number by GPU count without checking the topology.

**Worked hypothetical example:** four readers share a 12 GB/s storage array and
a 20 GB/s network path. Each GPU-facing path can carry 25 GB/s. Aggregate storage
throughput cannot exceed 12 GB/s. If four equally weighted readers receive an
equal share, their analytical ceiling is 3 GB/s each. Equal sharing is an
assumption, not a measurement or a scheduling guarantee.

Count decimal GB/s and GiB/s consistently. `100 Gb/s / 8 = 12.5 GB/s` is a raw
bit-to-byte conversion; protocol overhead and contention lower useful payload
throughput. CPU decode, memory copies, decompression, and augmentation can become
bottlenecks after storage is improved.

## 2. Establish a valid baseline

Create a dedicated test file containing known data. Avoid a sparse or easily
compressible fixture unless that is the workload you intend to study. Keep the
same payload, access pattern, allocation, and timing boundaries between paths.

Record the operating-system and storage cache conditions. Repeating a small file
can measure a cache rather than physical media. Do not label a run “cold” because
it is the first iteration of your Python loop. Document how cache conditions were
established; system-wide cache changes affect other workloads and are not part of
this portable exercise.

For each run, separate:

| Measurement | Includes |
|---|---|
| Initialization | Context, driver, file and buffer setup |
| Steady-state I/O | Submission through verified completion |
| Consumer-ready latency | I/O plus required layout conversion, decode, and synchronization |
| End-to-end step time | Input pipeline and the actual downstream computation |

Use the installed `gdsio` tools and their own help as described in NVIDIA's
[benchmarking/configuration guide](https://docs.nvidia.com/gpudirect-storage/configuration-guide/index.html).
Do not invent a `gds_bandwidth` command line or copy flags from a different tool
version. Save the exact command and tool version with the results.

## 3. Change one variable at a time

| Sweep | Keep fixed | What the result can reveal |
|---|---|---|
| Logical transfer size | Total bytes, pattern, path, buffers | Per-request overhead versus sustained transfer |
| Concurrency | Size, total bytes, path | Whether more outstanding work hides latency or creates contention |
| Alignment | Logical payload and integrity comparison | Which implementation path or extra work changes |
| Registration reuse | Same buffers and requests | Setup amortization versus memory/registration pressure |
| Direct versus compatibility path | Same data and timing boundary | Whether the selected path improves this workload |
| One versus several readers | Same per-reader workload | Shared links, fairness, CPU/NUMA or storage bottlenecks |

**Worked queue estimate:** a hypothetical path capable of 20 GB/s with 200 μs
request latency needs about `20e9 × 200e-6 = 4,000,000 bytes` in flight to cover that
bandwidth-delay product. Four 1 MiB requests hold about 4.19 MB. This suggests an
initial experiment; it does not prove that queue depth 4 saturates a real system.
Service time, software overhead, resource limits, and variability still matter.

## 4. Draw overlap instead of assuming it

For two buffers, a valid schedule looks like:

```text
buffer A: read 0 -> compute 0 -> read 2 -> compute 2
buffer B:          read 1 -> compute 1 -> read 3
                  ^ overlap is legal only while dependencies are satisfied
```

The next read may overlap computation on another buffer if the hardware and API
support it. The producer must not overwrite a buffer before its consumer finishes.
A host-synchronous read can overlap previously launched GPU computation, but it
still blocks its host thread. Stream-ordered asynchronous I/O has a different
submission/completion contract.

**Predict:** if I/O takes 4 ms and independent compute takes 6 ms, ideal pipelining
can approach a 6 ms initiation interval after startup, instead of 10 ms serial
service. It does not make a single item's dependency chain shorter than 10 ms.
Measure whether shared resources or synchronization prevent this ideal overlap.

## 5. Report evidence without invented targets

Use a table populated from your own raw samples:

| Path / configuration | Payload integrity | I/O path evidence | I/O time | Consumer-ready time | Step time | CPU use |
|---|---|---|---|---|---|---|
| Baseline | record | record | measure | measure | measure | measure |
| One changed variable | record | record | measure | measure | measure | measure |

A successful API call is not direct-path evidence. A fixed “3.2× faster” figure
without workload, versions, hardware, and raw measurements is not an appropriate
acceptance target. State variation across trials, cache conditions, and whether
initialization is amortized. Explain a result that failed to improve performance
as carefully as a speedup.

## 6. Self-checks and transfer

- **Registration is expensive; should every buffer be registered?** Reuse can
  amortize cost, but registration consumes resources. Measure the intended pool
  and lifetime; there is no universal one-millisecond cost or ideal pool size.
- **GPU memory bandwidth is much larger than storage bandwidth; can it be ignored?**
  It may be a loose bound for an isolated I/O test. Concurrent kernels and copies
  can still contend for memory resources in the complete application.
- **Does twice the storage hardware double throughput?** Only if storage was the
  limiting resource and the remaining links/software can serve the added demand.
- **How does this apply to robot learning?** Replay datasets and checkpoint loads
  can benefit from a better input path, but small online camera observations have
  different sizes, deadlines, and processing steps. Measure the actual workload.

**Independent task:** choose a recorded robot-data loader. Count raw bytes, image
or tensor decoding, batching, and device transfers. Identify one measured
bottleneck, change one factor, and show both integrity and end-to-end timing.

[Course index](README.md) · [Distributed storage](06-Disaggregated-Storage.md) ·
[Official best practices](https://docs.nvidia.com/gpudirect-storage/best-practices-guide/index.html)
