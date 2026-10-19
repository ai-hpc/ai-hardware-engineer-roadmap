# NCCL Deep Dive — NVIDIA Collective Communications Library

NCCL (pronounced "Nickel") is the **core communication engine** that makes multi-GPU AI training possible. On NVIDIA deployments configured with the NCCL process-group backend, these libraries use NCCL for many collectives. PyTorch also supports other backends; verify the selected backend and execution path.

Understanding NCCL at depth means understanding **why your training runs fast or slow**, and how to fix it when it isn't.

## What NCCL Solves

For an AllReduce over `n` ranks, nccl-tests reports `algbw = S / t` and `busbw = algbw × 2(n−1)/n`, where `S` is the payload per rank. For a hypothetical 1 GB payload completed in 10 ms on 8 ranks, these are 100 GB/s and 175 GB/s respectively. Those numbers are definitions, not measurements from this repository. Neither can be equated blindly with a bidirectional NVLink datasheet number. [nccl-tests bandwidth definitions](https://github.com/NVIDIA/nccl-tests/blob/master/doc/PERFORMANCE.md).

Start with correctness and a message-size sweep, then inspect topology and algorithm selection. NCCL may use NVLink, PCIe, NICs, or staging depending on the system; CPU launch/control work still exists. A universal “250× faster” example hides the very bottleneck this course teaches you to measure.

## Topic Index

| # | Topic | Key Questions Answered |
|---|---|---|
| 01 | [Fundamentals](./01-Fundamentals.md) | What are collectives? What does each operation do? |
| 02 | [Algorithms & Bandwidth](./02-Algorithms-and-Bandwidth.md) | How does Ring AllReduce work? How do payload bandwidth, normalized bus bandwidth, and link limits differ? |
| 03 | [Framework Integration](./03-Framework-Integration.md) | How do PyTorch/DeepSpeed/Megatron use NCCL internally? |
| 04 | [Configuration & Tuning](./04-Configuration-and-Tuning.md) | Which env vars matter? How to tune for H200 vs PCIe? |
| 05 | [Multi-Node Clusters](./05-Multi-Node-Clusters.md) | InfiniBand, SHARP offload, hierarchical AllReduce |
| 06 | [Debugging](./06-Debugging.md) | Hangs, timeouts, topology mismatches — how to fix them |
| 07 | [Trillion-Parameter Scale](./07-Trillion-Parameter-Scale.md) | How NCCL + tensor/pipeline parallelism trains 1T+ models |

## Quick Reference

- **Training slow, GPUs idle?** → [04-Configuration-and-Tuning](./04-Configuration-and-Tuning.md)
- **NCCL hang / timeout?** → [06-Debugging](./06-Debugging.md)
- **Understanding Ring AllReduce math?** → [02-Algorithms-and-Bandwidth](./02-Algorithms-and-Bandwidth.md)
- **Building multi-node cluster?** → [05-Multi-Node-Clusters](./05-Multi-Node-Clusters.md)
- **Training 70B+ models?** → [07-Trillion-Parameter-Scale](./07-Trillion-Parameter-Scale.md)

## Start on a CPU

Run the [two-process collectives and DDP lab](CPU-Collectives-Lab/Guide.md): test SUM/AllGather, catch an incompatible shape, and compare unequal-batch gradients with a serial reference before using a GPU cluster.

## Practice and assessment

**Entry diagnostic (prerequisite check):** Explain rank, communicator, collective semantics, and the physical link path. If this is unfamiliar, use the prerequisite material above or the [roadmap diagnostics](../../../../../Learning-Guide.md#diagnose-before-reading) before starting the lab.

1. **Baseline lab:** Run a correctness-checked collective baseline over several message sizes.
2. **Guided variation:** Change rank placement and inspect a controlled communication failure in a test job.
3. **Independent transfer:** repeat the method on a new input or configuration and explain where your original assumptions stop applying.

**Evidence to submit:** Latency, algorithm/bus bandwidth definitions, topology, scaling, and diagnostic logs. Keep raw results and reproduction steps; distinguish measured results from estimates or work not run.

**Self-check:** Which observation would disprove your explanation of the result? Identify a competing explanation and the smallest experiment that separates them. A passing answer names a concrete predicted difference, not only another tool to try.

**Exit criteria:** reproduce the baseline, demonstrate the negative/variation check, and meet the [shared completion rubric](../../../../../Learning-Guide.md#completion-rubric). Use the [lesson cycle](../../../../../Learning-Guide.md#a-repeatable-lesson-cycle) for retrieval practice and spaced review.
