# 05 — Warp specialization: pipeline ownership and reuse

**Outcome:** explain producer/consumer roles and prove when a shared-memory stage may be reused. This lesson is a source-reviewed design exercise; it does not claim a locally executed Hopper kernel.

## 1. Separate roles to expose overlap

Warp specialization assigns different work to different warps or groups: for example, some issue tile loads while others compute. It can overlap operations with different bottlenecks. It does not guarantee that producers or consumers remain busy, and it is not limited to one architecture.

Uniform-role kernels can also overlap loading and computation through software pipelining, multiple warps and asynchronous operations. Specialization is a scheduling choice to compare against an already competent baseline.

```text
producer: acquire empty stage -> issue transfer -> establish ready state
consumer: wait for ready stage -> compute -> wait for use to finish -> release stage
```

The last wait is essential when computation itself is asynchronous. “The consumer issued an instruction” and “the stage may be overwritten” are different events.

## 2. Hopper-specific mechanisms

Hopper provides TMA for supported asynchronous tensor transfers and WGMMA operations involving a warpgroup of 128 threads. Specific instruction shapes, layouts, types and participation rules come from the selected architecture's ISA and library implementation.

A thread still issues the transfer; descriptors, alignment, transaction completion and barriers still need management. The transfer is not free, and a producer role does not remove memory bandwidth limits. Ampere also supports asynchronous copy mechanisms; specialization as a concept predates Hopper's TMA/WGMMA combination.

Sources: [Hopper architecture](https://developer.nvidia.com/blog/nvidia-hopper-architecture-in-depth/) and [PTX ISA](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html). Keep Hopper mechanisms distinct from newer Blackwell matrix and memory mechanisms when reading current CUTLASS code.

## 3. Work a two-buffer schedule

Let each load take 3 time units and each compute take 5, with independent resources and no overhead. Four tiles take `4×(3+5)=32` units without overlap. An ideal pipeline takes `3 + 4×5 = 23` units. The model assumes enough buffering and correct dependencies; it is not measured hardware timing.

| Stage | Buffer A | Buffer B |
|---|---|---|
| Fill | Load tile 0 | Empty |
| First compute | Consume tile 0 | Load tile 1 |
| Next | Load tile 2 only after tile 0 is released | Consume tile 1 |
| Next | Consume tile 2 | Load tile 3 only after tile 1 is released |
| Drain | Released | Consume tile 3, then release |

Changing only the buffer index modulo two does not implement this protocol. A producer that laps a slow consumer can overwrite live data. Each reuse needs a matching generation/phase and completion condition.

**Checkpoint:** if compute is asynchronous, when may the producer load tile 2 into A? After the consumer's relevant operations that read A have completed and the release is visible under the chosen synchronization mechanism.

## 4. State the collective contract

For each barrier or pipeline primitive, write:

- the participating threads and expected arrivals;
- the transfer bytes or transactions tracked, if applicable;
- the current phase/generation;
- which operation establishes data readiness;
- which operation establishes that consumers are finished;
- how the final partial tile and pipeline drain are handled.

A generic `barrier.arrive()` and a comment saying “wait for phase” are not a correct implementation. TMA transfer completion and asynchronous matrix-use completion are different obligations. Use the exact library's API and a complete example for the selected architecture.

The [CUDA asynchronous-execution documentation](https://docs.nvidia.com/cuda/cuda-programming-guide/) and [CCCL pipeline API](https://nvidia.github.io/cccl/libcudacxx/extended_api/synchronization_primitives/pipeline.html) explain the mechanisms. A high-level protocol diagram alone cannot validate memory visibility or device progress.

## 5. Trace an established implementation

Use a pinned [CUTLASS](https://github.com/NVIDIA/cutlass) revision and an architecture-appropriate example. Record:

| Decision | What to find in source |
|---|---|
| Target and instruction family | Build target and selected collective |
| Roles | Producer/consumer participants |
| Tile and layout | Shapes, alignment and swizzle requirements |
| Stage storage | Number of buffers and shared-memory footprint |
| Ready/reuse protocol | Full and empty barriers, phases and waits |
| Resource budget | Registers, shared memory and occupancy |
| Epilogue | Accumulator conversion, output shape and tail stores |

Do not copy a partial template alias and call it a complete GEMM. CUTLASS/CuTe APIs evolve, and architecture-specific examples may require targets such as `sm_90a`. Compile and run the complete selected example with its stated toolchain before altering the schedule.

## 6. Understand the FlashAttention-3 example accurately

[FlashAttention-3](https://arxiv.org/abs/2407.08608) combines Hopper asynchronous operations, warp specialization and overlap across attention suboperations. Its schedule should be explained from the paper and a pinned implementation. A cartoon with exactly one QK warpgroup and one PV warpgroup is not a universal kernel layout.

When comparing FA2 and FA3, preserve head dimension, sequence lengths, causal mask, dtype, GPU, clock/power settings and timing convention. Explain whether FLOPs count masked work and whether the hardware ceiling is dense or sparse. A paper's result on one configuration is not a guaranteed table for every H100/H200.

This course has no local Hopper result for this lesson. Do not infer TFLOP/s, utilization or speedup from the ideal two-stage timing example.

## 7. When specialization helps or hurts

Potential benefit comes from exposed overlap, reduced instruction overhead for one role, or a better resource allocation. Potential costs include dedicated producer threads, barrier overhead, more shared-memory stages, register pressure and reduced occupancy.

Small workloads may not amortize setup. A bandwidth-saturated workload cannot exceed its relevant bandwidth ceiling just by changing roles. A compute-intensive workload can still stall because a producer cannot supply the next tile. The decision requires a trace and resource measurements.

**Independent exercise:** draw a three-buffer schedule for load time 7 and compute time 3. Identify the limiting stage. Can additional buffering change its steady-state rate if the same single transfer resource still needs 7 units per tile? No; it can absorb variation or hide a limited gap, but cannot remove that throughput bound.

**Completion:** submit an annotated source path, a ready/reuse timeline, a tail/drain case, and a falsifiable performance prediction. If hardware is available, add compiled/runtime correctness and measured traces. Otherwise mark those outcomes not run and use the [CPU compiler experiment](../../../../../Phase%204%20-%20Track%20C%20-%20DL%20Inference%20Optimization/00%20-%20CPU%20Compiler%20Lab/Guide.md) for an actual executed optimization study.
