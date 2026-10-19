# Lecture 06 — MoE placement and communication

**Prerequisites:** Lecture 05 and expert/tensor parallelism. **Time:** 3–4 hours; multiple GPUs for measurement. **Outcome:** choose placement from topology and observed routing.

## Placement is part of the algorithm

Tensor parallelism partitions tensor operations. Expert parallelism assigns experts to ranks and dispatches token activations to them. Shared components may remain replicated. Data parallelism may replicate an entire serving group. Draw these groups explicitly; multiplying every CLI parallel-size value is not a reliable rank count.

For each phase, estimate dispatched tokens, hidden dimension, activation bytes, selected experts, and whether combine traffic returns to the sender. Real runtimes may fuse, compress, or overlap transfers, so validate the byte model with collectives and traces.

## Worked imbalance example

Four ranks process 100, 100, 100, and 220 equal-cost expert assignments. Ignoring overlap and communication, the iteration follows the 220-work rank, not the average 130. The mean utilization implied by this simplified model is 130/220 ≈ 59%. This is an illustrative workload, not a DeepSeek routing measurement.

Measure max/mean tokens per rank, rank latency, expert occupancy, and dispatch/combine time. Text and image-containing workloads can route differently; use both if both are in your deployment contract.

## Lab

1. Record the GPU topology and links actually available, including cross-node boundaries.
2. Validate a small collective baseline before testing the model.
3. Compare two **supported** placements at fixed checkpoint, request mix, and offered load.
4. Separate startup redistribution from steady-state dispatch.
5. Compare p95 request latency, completed output throughput, per-rank peak memory, and communication time.
6. Repeat at low and high concurrency. Keep three runs per condition.

The [NCCL documentation](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/index.html) is the primary reference for collectives and diagnostics. A collective bandwidth microbenchmark does not by itself predict small-message latency or full-model performance.

## Offload extension

If Engram or experts are offloaded, record useful bytes, actual transferred bytes, cache hits, prefetch behavior, host capacity, and storage queue depth. Random lookups can trigger much larger page transfers than the requested row. Test cold and warm cases separately, and report the cold-start cost.

**Reduced exercise:** use an invented token-to-expert assignment matrix to calculate rank imbalance and compare two placements. It teaches the scheduling model but cannot validate interconnect performance.

For an executable follow-up, the [CPU MoE training and dispatch lab](../../../Track%20A%20-%20GPU%20Infrastructure/Nvidia%20GPU/HPC%20Setup/Long-Context-MoE-Foundation-Training/CPU-Training-Lab.md) runs actual two-process dispatch and combination against a single-process reference. Include its empty-source and idle-expert cases: zero useful work does not remove a rank's collective obligations. This tiny network teaches routing and communication correctness; it is not a DeepSeek loader or a measurement of a GPU fabric.

## Self-check

- Can more GPUs increase latency? Yes: smaller local work and extra synchronization/traffic may outweigh saved compute.
- Does higher all-to-all throughput imply better request latency? Not necessarily; latency, contention, and overlap matter.
- Which memory number determines feasibility? The worst rank's peak, including replicated and temporary state.

**Exit artifact:** topology diagram, group definitions, rank statistics, and a justified placement decision. Include a condition under which you would choose a different placement.

[Previous](Lecture-05.md) · [Overview](README.md) · [Next: cache behavior](Lecture-07.md)
