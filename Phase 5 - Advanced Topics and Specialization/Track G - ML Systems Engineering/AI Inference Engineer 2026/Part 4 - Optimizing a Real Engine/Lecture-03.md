# Part 4 · Lecture 03 — Diagnosis: Launch-Bound, Bandwidth-Bound, or Comm-Bound?

**Practice:** [CPU source lab](CPU-Source-Lab.md). Historical GPU results are source reports; use the **Answers** after attempting the self-check.


## Overview

Optimization starts with competing explanations, then measurements that distinguish them. Low throughput alone does not tell you whether a kernel saturates HBM, waits for memory, launches too little parallel work, waits on another rank, or spends time between launches.

The historical SparkInfer measurements below belong to different revisions. Do not combine an early launch count with a later graph-captured throughput as if they described one execution.

## 1. Start with the ceiling you cannot beat

A bandwidth bound is useful only when the byte model matches the executed workload. For rank r, let Bᵣ be bytes transferred from HBM for one token and BWᵣ its available bandwidth. A lower bound on a synchronous phase is:

```text
phase_time >= max_r(B_r / BW_r)
```

Add serial phases separately. Overlap changes how phase bounds combine; do not blindly add every profiler duration. Aggregate node bandwidth is appropriate only when useful traffic is balanced and concurrent across ranks.

Count routed expert weights, shared/dense weights, KV or recurrent state, activation traffic, quantization metadata, and replicated reads. Requested load bytes and HBM bytes differ because of caches. A tensor's file size is not automatically its runtime format or its bytes read per token.

**Worked estimate.** Suppose each of eight ranks actually reads 2 GB during a phase and can sustain at most 4.8 TB/s. The optimistic phase bound is 2e9/4.8e12 = **0.417 ms**. If the measured phase takes 4 ms, that gap is an investigation target. It does not prove that quantization cannot help: memory latency, small grids, poor coalescing, cache misses, or other work may prevent bandwidth saturation.

A tempting but invalid K3 calculation multiplies an old “50B active” label by 553 GiB/2.8T, then divides aggregate HBM bandwidth by that result. The official card reports **104B activated**, and global mean bytes/parameter does not describe the heterogeneous active tensors. Build the byte ledger from tensor types and routing instead.

At batch 1, a simple BF16 matrix-vector product performs about two FLOPs per two-byte weight: approximately **1 FLOP/byte**, ignoring other traffic. At 0.21 stored bytes/weight, the weight-only arithmetic intensity would be about **9.5 FLOPs/byte**, before unpacking, scales, and other traffic. “Batch 1 means intensity 1” is not format-independent.

## 2. Price the collective, because everyone assumes it is the problem

The source's early `ExpertsOnly` configuration reports:

```text
92 MoE layers × 1 all-reduce/layer = 92 calls/token
payload = 3584 F32 elements × 4 = 14,336 bytes = 14 KiB
92 × 58.7 us = 5.4004 ms/token
5.4004 / 281.6 = 1.92% of the early measured token time
```

This makes communication a small component **in that configuration**. The head-sharded path later uses 185 collectives; faster kernels also make the same communication cost a larger fraction. Re-measure after structural changes.

Do not multiply payload bytes into a latency that already measures the complete call. A 14 KiB collective and a 7 MiB collective reportedly took 58.7 and 85 μs: 512 times the payload for 1.45 times the latency. That suggests a substantial fixed component over this range. Fit `latency ≈ alpha + bytes/beta` only as a local model, and report topology, rank count, dtype, synchronization, and concurrency. There is no universal size below which every collective is latency-bound.

## 3. Estimate the launch bill

The early source estimates around 30 kernels/layer × 93 layers = 2790 launches per rank/token. Dividing a 16.6 ms token budget by 2790 gives **5.95 μs per launch slot**. That is a budget, not a measurement of a kernel or host API call. Eight ranks, multiple streams, graph replay, and overlap make this distinction essential.

Collect a timeline containing CPU submission, kernels, copies, collectives, and explicit synchronization. Ask whether the GPU's critical path waits for work the CPU has not submitted. Compare eager and captured execution with identical work and properly completed timing intervals.

| Observation | Useful next measurement |
|---|---|
| Gaps between dependent kernels | CPU launch trace, dependency waits, stream events, and copies |
| One long kernel with low bandwidth | Grid, resident warps, instructions, stalls, and actual HBM bytes |
| Rank finishes early and waits | Per-rank routing/load distribution and collective arrival times |
| Removing a synchronization appears faster | Check that the timing interval still waits for completion |
| Captured path improves while kernels are unchanged | Paired end-to-end timing and launch-gap reduction |

Summing durations across overlapping GPUs/streams does not produce wall-clock time. Keep the dependency path and measurement scope explicit. A graph can help even when kernels last 100 μs; the size of the benefit must be measured, not inferred from a fixed “under 10 μs” rule.

## 4. Occupancy — the ceiling hidden inside “the kernel is slow”

On a 132-SM H200, a 12-block grid can place work on at most 12 SMs at once: **9.1% grid coverage**. This is not “9.1% occupancy.” Occupancy means resident warps divided by the hardware maximum per SM. Block size, registers, shared memory, and architectural limits constrain it.

A single 1024-thread block can expose more work than a 128-thread block while still using only one SM. A larger grid can increase memory-level parallelism, but more blocks can also add redundant reads or reduce locality. Find an additional independent axis—context slices, value tiles, or expert/FFN bands—then measure the complete operation including its merge.

The relevant source cases are the KDA value tiling in [Lecture 04](Lecture-04.md) and MLA splitting in [Lecture 06](Lecture-06.md). Their general lesson is to expose useful parallel work without silently changing the data dependencies.

## 5. The diagnosis changes with context depth

The source reports an early comparison:

| Engine | Context 64 | Context 131,072 |
|---|---:|---:|
| llama.cpp reference | 18.32 tok/s | 18.44 tok/s |
| SparkInfer candidate | 10.34 tok/s | 1.00 tok/s |

The candidate's depth-dependent work clearly warrants investigation. The table alone does not identify its cause. Both engines use compressed latent MLA state; the candidate's older 576-F32 layout is not an uncompressed full-head cache. Cache dtype, representation, reuse across heads, grid shape, and context splitting all affect cost.

A nearly flat reference curve over two measurements does not establish O(1) attention. Also inspect how context was populated. The candidate's seek-based synthetic cache does not represent a real 128k prefix, especially when routing and expert skipping depend on values. Separate kernel timing from real-prefix quality and end-to-end serving claims.

## 6. Amdahl's law, with receipts

The record gives these 16-layer timings before and after MoE work:

| Configuration | Before | After |
|---|---:|---:|
| One rank | 196.65 ms | 60.66 ms |
| Eight ranks | 80.61 ms | 54.33 ms |
| Scaling factor | 2.44× | 1.1165× |

Absolute eight-rank performance improves even while scaling relative to one rank gets worse. That is possible whenever the remaining work parallelizes less well.

Under the **idealized** model `T(N)=S+P/N`, solve:

```text
S = (8*T8 - T1)/7       P = T1-S
before: S=64.03 ms, P=132.62 ms, serial fraction=32.56%
after:  S=53.43 ms, P=  7.23 ms, serial fraction=88.07%
```

These pairs do **not** establish “the parallel component shrank exactly 3× while the serial component stayed fixed.” The fitted terms change differently, and the fit absorbs communication, kernel efficiency, and other overhead. More rank counts and phase measurements are needed for causal attribution.

**A clean hypothetical example:** a phase is 70% of a one-rank workload and becomes three times faster. New time is `0.30 + 0.70/3 = 0.5333` of the old time. If only that remaining 0.2333 portion parallelizes, the new serial fraction is **56.25%** and ideal eight-way scaling is `1/(0.5625 + 0.4375/8) = 1.620×`. This derivation has explicit assumptions, unlike inferring a causal model from two endpoint ratios.

## 7. The diagnostic sequence

1. Pin model, weights, runtime, flags, hardware, input, and timing boundaries.
2. Build per-rank byte estimates and distinguish requested traffic from measured HBM traffic.
3. Count collectives from the actual forward path; measure their relevant payloads.
4. Inspect CPU/GPU timelines and state which gaps are on the critical path.
5. Inspect grid coverage, resident-warps constraints, and repeated memory traffic.
6. Sweep context and batch over deployment-relevant ranges using declared cache initialization.
7. Write one falsifiable prediction, make one change, and re-measure the whole operation.

Do the arithmetic on a laptop. Actual GPU timing, counters, and collective measurements require the relevant hardware; a CPU model cannot establish them.

## Lab — diagnose before you touch anything

Begin with the [CPU source lab](CPU-Source-Lab.md): run the real shard and metadata checks, then calculate the memory and Amdahl examples above. Connect them to the existing [Part 1 contract lab](../Part%201%20-%20Fundamentals/labs/README.md).

If you have a supported GPU workload, add `DIAGNOSIS.md` containing one measured number for each candidate bottleneck, a trace or raw timing file, your predicted intervention, and the observation that would disprove it. If you have no GPU, label the trace-dependent conclusions **unmeasured** and finish the source/math exercise; renting an eight-GPU node is not required to learn this section.

## Self-check

1. Does being 1000× below a naive bandwidth bound prove memory is irrelevant?
2. Calculate the collective share from 92 calls at 58.7 μs in a 281.6 ms step.
3. What does 16.6 ms / 2790 tell you, and what does it not tell you?
4. Distinguish a 12-block grid on 132 SMs from CUDA occupancy.
5. Can a flat two-point context curve prove attention has constant complexity?
6. Derive the 56.25% serial fraction and 1.620× scaling in the hypothetical example.

## Answers

1. No. Check the byte model, replication, actual traffic, memory latency, grid size, and other phases. A loose upper bound does not locate a bottleneck.
2. 5.4004 ms, or 1.92%; valid for that call pattern and measured state.
3. A 5.95 μs average budget per counted slot, not measured host overhead or kernel duration. Ranks, overlap, graph mode, and dependencies matter.
4. At most 9.1% of SMs receive those blocks simultaneously; occupancy concerns resident warps per SM and needs resource information.
5. No. Other costs can dominate, and the two workloads may use different timing or cache-population methods.
6. New serial share = 0.30/(0.30+0.70/3)=0.5625. Apply ideal Amdahl scaling to the remaining parallel fraction, obtaining 1.620×.

## Sources and next step

The historical timings come from the pinned [tensor-parallel notes](https://github.com/gittensor-ai-lab/sparkinfer-k3/blob/7689cc77e0e18d4b4f1937cf2a189dcd5a8d90f3/docs/tensor-parallel.md) and [reference ledger](https://github.com/gittensor-ai-lab/sparkinfer-k3/blob/7689cc77e0e18d4b4f1937cf2a189dcd5a8d90f3/bench/scripts/reference.lock). They are not GPU measurements reproduced by this course refresh.

Continue to [Lecture 04 — Launch geometry](Lecture-04.md), or return to [Part 4](README.md).
