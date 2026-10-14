# Chapter 4: Multi-GPU and GB200 NVL72 Placement

**Previous:** [Single-GPU deployment](03-Single-B200-Qwen-Inference.md) · **Next:** [Kernel engineering](05-Blackwell-Kernel-Engineering.md)

## Place tensors before counting GPUs

Multi-GPU execution solves a concrete constraint: model capacity, request throughput, or latency. Begin with Chapter 3's largest-rank memory budget and measured single-group baseline. NVLink adds a fast communication path; it does not merge allocations or eliminate collectives.

A GB200 Superchip combines **one Grace CPU and two Blackwell GPUs**. The NVL72 configuration combines 36 Grace CPUs and 72 GPUs in a scale-up domain. CPU–GPU coherency over NVLink-C2C is distinct from GPU-to-GPU NVLink connectivity. Remote HBM, local HBM, and Grace memory have different paths and costs. [NVIDIA GB200 system overview](https://docs.nvidia.com/multi-node-nvlink-systems/multi-node-tuning-guide/overview.html).

## Choose a parallelism strategy with an explicit communication model

| Strategy | Memory/compute benefit | Communication to measure | Common bad assumption |
|---|---|---|---|
| Replicas | More independent request capacity | Request routing; optional cache sharing | Replication makes an oversized model fit |
| Tensor parallelism | Splits selected matrices across ranks | Per-layer reductions/gathers and synchronization | More ranks always reduce batch-one latency |
| Pipeline parallelism | Splits layer stages | Activation transfer and pipeline bubbles | Every stage has equal compute/memory |
| Expert parallelism | Places MoE experts across ranks | Token dispatch/combine, routing skew | Active parameters determine total resident expert memory |
| CPU memory/offload | Additional capacity | Movement and synchronization on the actual path | Coherency makes CPU memory as fast as HBM |

The legal TP degree depends on the model and engine's partition rules. Check attention-head, KV-head, hidden-dimension, and kernel constraints. A rack with 72 GPUs does not imply TP=72 is supported or appropriate. Several smaller TP groups plus data-parallel replicas may produce better goodput.

## Worked break-even model

Let a measured single-group decode step take `T_compute`. A candidate TP split changes it to approximately:

`T_step = T_partitioned_compute + T_collectives + T_launch/sync + T_other`

If compute falls from an illustrative 20 ms to 12 ms but newly exposed communication costs 10 ms, the candidate is slower. These numbers are a teaching example, not B200 measurements. Measure overlap: simply summing all profiler event durations can double-count simultaneous compute and communication.

For an 8-rank AllReduce of payload `S`, nccl-tests' normalized bus bandwidth is `(S/t) × 2(7/8)`. Link specifications often aggregate send and receive directions, while a collective's algorithm bandwidth describes useful payload divided by time. Keep the definitions beside the plot. [NCCL bandwidth definitions](https://github.com/NVIDIA/nccl-tests/blob/master/doc/PERFORMANCE.md).

## Grace memory is a separate tier

A coherent CPU/GPU address model removes some programming complexity but does not remove transfer cost, placement effects, page behavior, or bandwidth contention. Do not use the bidirectional NVLink-C2C link rating as guaranteed LPDDR read bandwidth. A KV offload experiment must record which pages live where, the migration policy, and observed GPU stalls.

A useful experiment compares no offload, a fixed offload fraction, and a lower-concurrency HBM-only configuration. Report capacity gained and latency lost. If the working set is repeatedly fetched across the slower link, a model that fits may still violate its deadline.

## Lab: grow a serving group, then grow replicas

1. Save device/rank/NIC/CPU affinity mapping and peer-access results.
2. Run correctness-checked collective tests across representative message sizes. Distinguish a healthy physical fabric from a healthy serving configuration.
3. Test one supported TP group at fixed prompts, context, output length, and concurrency. Record per-rank memory and collective time.
4. Increase TP only where legal; then compare the same total GPU count arranged as more replicas of the smaller TP group.
5. Add load until the latency objective is missed. Count failures, queueing delay, and imbalance.
6. On an isolated test job, stop a worker and document failure detection and recovery. A restarted worker is not necessarily a resumed stateful request.

**Deliverable:** a placement matrix showing GPUs per group, replica count, per-rank peak memory, useful goodput, p99 latency, and failure behavior. Carry it into Chapter 6 for admission and recovery policy.

## Translate the lesson to a custom SoC

A wide matrix array can be underused when DMA or SRAM banking serializes its input. A large memory sum can hide an inaccessible or slow bank. Reuse the same placement-and-communication reasoning for accelerator tiles, CPU fallbacks, sensor buffers, and wireless contention. Draw the path before assuming a bandwidth improvement helps the control loop.
