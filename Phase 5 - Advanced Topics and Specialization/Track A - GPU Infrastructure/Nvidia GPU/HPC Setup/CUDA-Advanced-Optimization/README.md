# CUDA Advanced Optimization — Deep Dive

Study five optimization techniques through their correctness conditions, data movement and measured bottlenecks. Start with streams, memory ownership, array indexing and the CPU compiler lab before using architecture-specific mechanisms.

**Begin with the [correctness lab](labs/README.md).** Nine CPU checks run without CUDA. Complete CUDA examples were compiled with CUDA 13.4.92; GPU execution is explicitly unvalidated because no usable CUDA device was available. The five lessons distinguish executable source, mathematical models and source-reviewed designs. Allow 45–60 minutes per lesson plus toolchain setup.

## Why These Techniques Matter

Each technique addresses a different measured bottleneck. Begin with a trace that distinguishes CPU launch gaps, memory traffic, compute utilization, and synchronization stalls. There is no universal ladder from 30% to 85% of peak: the relevant ceiling depends on arithmetic intensity, precision, shape, and memory level.

A graph replay still has a launch cost. A persistent kernel amortizes repeated dispatch but consumes resident resources and needs correct work-queue termination. Fusion can reduce HBM traffic while increasing register pressure. Cooperative launch and synchronization have explicit residency and participation requirements; validate them on the target GPU. [CUDA programming model](https://docs.nvidia.com/cuda/archive/13.0.0/cuda-c-programming-guide/index.html).

## Topic Index

| # | Topic | What It Solves |
|---|---|---|
| 01 | [CUDA Graphs](./01-CUDA-Graphs.md) | Repeated host submission can limit short operations |
| 02 | [Cooperative Groups](./02-Cooperative-Groups.md) | Thread block boundary limits synchronization flexibility |
| 03 | [Persistent Kernels](./03-Persistent-Kernels.md) | Reuse blocks across finite work items |
| 04 | [Kernel Fusion](./04-Kernel-Fusion.md) | Separate kernels waste HBM bandwidth on intermediate results |
| 05 | [Warp Specialization](./05-Warp-Specialization.md) | Assign roles and coordinate asynchronous buffer reuse |

## How They Relate

```
CUDA Graphs          → reduces CPU↔GPU interface overhead
Cooperative Groups   → enables flexible intra-kernel synchronization
Persistent Kernels   → amortizes dispatch across multiple work items
Kernel Fusion        → reduces HBM round-trips between operations
Warp Specialization  → overlaps compute and memory within a single kernel

Combined designs:
  Trace the actual algorithm and target implementation.
  Benchmark the combination; do not multiply isolated speedups.
```

## Quick Navigation

- **LLM inference latency too high?** → [01-CUDA-Graphs](./01-CUDA-Graphs.md)
- **Writing a custom reduction/scan?** → [02-Cooperative-Groups](./02-Cooperative-Groups.md)
- **Kernel launch overhead visible in profile?** → [03-Persistent-Kernels](./03-Persistent-Kernels.md)
- **GPU memory bandwidth bottleneck?** → [04-Kernel-Fusion](./04-Kernel-Fusion.md)
- **Want to write FlashAttention-style kernels?** → [05-Warp-Specialization](./05-Warp-Specialization.md)

## Practice and assessment

**Entry diagnostic (prerequisite check):** Name a kernel bottleneck with a roofline or trace rather than utilization alone. If this is unfamiliar, use the prerequisite material above or the [roadmap diagnostics](../../../../../Learning-Guide.md#diagnose-before-reading) before starting the lab.

1. **Baseline lab:** Optimize one measured CUDA operator after establishing numerical parity.
2. **Guided variation:** Test shapes outside the tuned case and show a deliberately faulty optimization is rejected.
3. **Independent transfer:** repeat the method on a new input or configuration and explain where your original assumptions stop applying.

**Evidence to submit:** Correctness, speed across shapes, memory traffic, occupancy/register use, and trace evidence. Keep raw results and reproduction steps; distinguish measured results from estimates or work not run.

**Self-check:** Which observation would disprove your explanation of the result? Identify a competing explanation and the smallest experiment that separates them. A passing answer names a concrete predicted difference, not only another tool to try.

**Exit criteria:** reproduce the baseline, demonstrate the negative/variation check, and meet the [shared completion rubric](../../../../../Learning-Guide.md#completion-rubric). Use the [lesson cycle](../../../../../Learning-Guide.md#a-repeatable-lesson-cycle) for retrieval practice and spaced review.
