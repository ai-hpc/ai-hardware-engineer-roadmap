# 02 — Collective algorithms and honest bandwidth accounting

**Outcome:** calculate collective bytes and explain why a normalized benchmark number differs from a link specification. Prerequisite: [collective semantics](01-Fundamentals.md). Start with the [CPU collectives lab](CPU-Collectives-Lab/Guide.md) if ranks are unfamiliar. Source review: 2026-10-18; no H200 benchmark was run for this lesson.

## A rank is a participant, not necessarily a machine

For an elementwise SUM AllReduce, every rank starts with its own array and finishes with the sum of all arrays. Three inputs `[1,2]`, `[10,20]`, `[100,200]` produce `[111,222]` at all three ranks. SUM does **not** divide by the rank count. A training framework may subsequently average gradients.

AllGather concatenates contributions; ReduceScatter reduces and distributes disjoint pieces of the result. Agree on rank order, element type, sizes, operation, and communicator before discussing speed.

## Ring: reduce pieces, then distribute them

Split each rank's S-byte array into N equal chunks. During reduce-scatter, each round sends one chunk to the next rank and adds a received contribution. After N−1 rounds, every rank owns one completely reduced chunk. During all-gather, N−1 further rounds distribute those finished chunks. Each rank sends `2(N−1)S/N` bytes in this idealized ring.

For two ranks holding `[1,2]` and `[10,20]`: after reduce-scatter, rank 0 can own `[11]`, rank 1 `[22]`; after all-gather both own `[11,22]`. This small example makes the reduction and replication distinct.

The factor `(N−1)/N` describes data movement. It is **not** a utilization guarantee. Launch overhead, memory traffic, routing, contention and protocol costs remain.

## Trees and hardware reductions

A tree reduces toward a root and broadcasts the reduced result back. Its dependency depth grows roughly as log₂N, whereas a simple ring has N−1 stages per phase. NCCL's double-tree approach splits the payload across complementary trees to distribute the forwarding load; it does not mean “reduce-scatter on tree A, all-gather on tree B.” Neither tree structure guarantees 100% bandwidth.

NCCL also has algorithms for compatible NVLink reduction hardware and network plugins. Algorithm selection depends on operation, sizes, topology, NCCL version and enabled features. There is no universal “larger than 1 MB means Ring” rule. Keep defaults for a baseline; isolate an override only when testing a hypothesis. [NVIDIA's double-tree explanation](https://developer.nvidia.com/blog/massively-scale-deep-learning-training-nccl-2-4/), [NCCL configuration](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/env.html).

## Units before conclusions

Use decimal `GB/s = 10⁹ bytes/s`. `1 GiB = 2³⁰ bytes`. A network advertised as 400 Gb/s has a 50 GB/s line-rate ceiling **per direction**, before overhead. H200 SXM advertises 900 GB/s aggregate bidirectional NVLink bandwidth **per GPU**; that is about 450 GB/s in each direction, shared across its links. It is not 900 GB/s separately to each of seven peers. NVSwitch provides switched connectivity; it does not create seven independent per-GPU bandwidth budgets. [H200 specifications](https://www.nvidia.com/en-us/data-center/h200/).

For `nccl-tests`, with elapsed time t and size S as defined for that operation:

| Operation | Algorithm bandwidth | Bus bandwidth normalization |
|---|---|---|
| AllReduce | S/t | algbw × 2(N−1)/N |
| AllGather | S/t, S = total output across contributions | algbw × (N−1)/N |
| ReduceScatter | S/t, S = total input array per rank | algbw × (N−1)/N |

Bus bandwidth is an operation-normalized metric, not a direct count from physical link counters. Its point-to-point accounting model needs additional care with in-network reductions, concurrent paths and shared NICs. Compare like operations, rank counts and hardware conventions. [Official nccl-tests accounting](https://github.com/NVIDIA/nccl-tests/blob/master/doc/PERFORMANCE.md).

**Worked example, invented timing for arithmetic only:** 8 ranks, S=1,000,000,000 bytes, t=4 ms gives `algbw=250 GB/s`, `busbw=437.5 GB/s`. For a simplified ring with 450 GB/s available one-way bandwidth and no overhead, `t ≥ 1.75×1e9/450e9 = 3.889 ms`. This bound does not predict what a real server will measure.

## A reproducible hardware measurement

These commands require a compatible CUDA/NCCL installation and eight allocated GPUs. They were source-reviewed, not run here. Record the repository SHA with the output; use an approved revision in a reproducible environment.

```bash
git clone https://github.com/NVIDIA/nccl-tests
cd nccl-tests
git rev-parse HEAD
make -j MPI=0 CUDA_HOME=/usr/local/cuda NCCL_HOME=/usr
./build/all_reduce_perf -b 1M -e 256M -f 2 -g 8 -n 20 -w 5
./build/all_gather_perf -b 1M -e 256M -f 2 -g 8 -n 20 -w 5
./build/reduce_scatter_perf -b 1M -e 256M -f 2 -g 8 -n 20 -w 5
```

Start small because several input/output/validation buffers may coexist. Inspect output column units; use raw bytes and timing when checking arithmetic. Require zero `#wrong` before interpreting performance. Save in-place and out-of-place results separately. Record clocks, power limit, temperature, topology, runtime versions and concurrent jobs. There is no universal passing GB/s threshold.

## Compute and communication must use the same workload

For a dense 70-billion-parameter transformer processing **8,192 global tokens**, the approximate `6PT` training estimate is `3.44064×10¹⁵ FLOPs`. At 989.5 TFLOP/s dense BF16 per H200 SXM, an impossible-perfect single-GPU compute lower bound is **3.477 seconds**, or **0.435 seconds** evenly divided across eight GPUs. Actual throughput is lower; attention costs, recomputation, optimizer work and parallelization change the model. The dense peak must not be replaced with the 1,979 TFLOP/s structured-sparse figure.

A separate hypothetical AllReduce of 140 GB per rank over an ideal eight-rank 450 GB/s ring costs `1.75×140/450 = 0.544 seconds`. That does not describe a feasible unsharded 70B training deployment on one 141 GB GPU: weights alone approach capacity, before gradients and optimizer states. Use this example to check units, then derive the actual sharded communication schedule.

ZeRO-3 reduces persistent memory by sharding parameters, gradients and optimizer state. It adds parameter gathers and does not reduce all communication to S/N. Overlap can hide only communication whose data dependencies permit concurrent work. A rough `max(compute,communication)` bound assumes perfect overlap; it is not an achieved step time.

## Check your understanding

1. Why is 900/25 an unfair NVLink-to-HDR comparison? **Answer:** the numerator is bidirectional aggregate and the denominator one-direction line rate; match direction and scope first (450/25=18), then account for shared links and overhead.
2. Does busbw=437.5 GB/s prove every NVLink transfers at that rate? **Answer:** no; it is a normalized collective result.
3. If all-gather contributes 8 bytes from each of 2 ranks, what S belongs in the table? **Answer:** 16 bytes, the complete output.
4. Which measurement distinguishes overhead from bandwidth? **Answer:** sweep message size; a near-constant small-message time suggests overhead, while a large-message time proportional to bytes suggests a bandwidth limit.

Submit one hand calculation, a CPU correctness report, and—when hardware is available—a raw sweep with a falsifiable explanation for one change in slope.
