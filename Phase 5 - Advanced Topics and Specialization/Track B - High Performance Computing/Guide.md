# Phase 5 — Track B: High Performance Computing

<div class="course-identity hpc" markdown="1">
<div class="course-identity__icon">HPC</div>
<div markdown="1">
<p class="course-identity__eyebrow">Track 5B · HPC / CUDA-X</p>
<p class="course-identity__title">Use GPU-accelerated libraries and performance tools to build high-throughput compute systems.</p>
<p class="course-identity__meta">Artifact: CUDA-X benchmark · Measure: FLOP/s, bandwidth, speedup, efficiency</p>
</div>
</div>


**Timeline:** Ongoing — study each category as your projects demand it.

**Prerequisites:** Phase 1 §4 (C++/CUDA), Phase 4 Track B (Jetson, CUDA runtime), Phase 4 Track C (ML compiler + DL inference optimization). Phase 5A is needed for distributed-library projects, not for the first single-GPU lab.

---

## What this track covers

High Performance Computing goes beyond single-GPU programming and cluster setup (covered in Phase 5A GPU Infrastructure). This track covers the **GPU-accelerated library ecosystem** and **domain-specific HPC applications** that run on top of that infrastructure.

| Sub-track | Focus | Guide |
|-----------|-------|-------|
| **CUDA-X Libraries** | The full NVIDIA GPU-accelerated library suite: math (cuBLAS, cuFFT), DL (cuDNN, CUTLASS, TensorRT), data (RAPIDS), vision (DALI, CV-CUDA), communication (NCCL, NVSHMEM), parallel algorithms (Thrust, CUB), and 40+ more | [CUDA-X Libraries →](CUDA-X%20Libraries/Guide.md) |

---

## How to use this track

1. **Start with one correct single-GPU library benchmark.** Add [Phase 5A — GPU Infrastructure](../Track%20A%20-%20GPU%20Infrastructure/Guide.md) when the chosen workload actually needs multiple GPUs or nodes.
2. **Then [CUDA-X Libraries](CUDA-X%20Libraries/Guide.md)** — the comprehensive reference for every GPU-accelerated library. Organized by category (math, DL, data, vision, communication) with role-based learning paths and hands-on projects.

## Relationship to other tracks

| This track provides | Other tracks use it for |
|--------------------|------------------------|
| cuBLAS, cuDNN, CUTLASS | Phase 4C (compiler targets), Phase 5F (AI Chip Design — what hardware must accelerate) |
| NCCL, NVSHMEM | Phase 5A (GPU Infrastructure — multi-GPU communication) |
| TensorRT, FlashInfer | Phase 4B §8 (Jetson runtime), Phase 4C Part 2 (inference optimization) |
| RAPIDS (cuDF, cuML, cuGraph) | Data pipeline acceleration for any ML workflow |
| CV-CUDA, DALI, Video Codec SDK | Phase 5E (Autonomous Vehicles — perception pipelines) |

## Make a library call into a hardware requirement

Select one VLA operation from a trace: a projection GEMM, normalization, resize, or reduction. Establish its shape/layout/dtype contract, then compare a library implementation with a correct reference. Include packing, transfer, plan creation, and synchronization when they are on the real inference path; report warm kernel time separately.

Pass the best measured implementation and the remaining bottleneck to the compiler/accelerator project. A custom accelerator should improve a real constrained pipeline, so comparing only against an untuned Python loop is insufficient evidence.

## Practice and assessment

**Entry diagnostic (prerequisite check):** Choose a library primitive and explain its data layout and synchronization contract. If this is unfamiliar, use the prerequisite material above or the [roadmap diagnostics](../../Learning-Guide.md#diagnose-before-reading) before starting the lab.

1. **Baseline lab:** Compare a library-backed operation with a correct baseline on representative shapes.
2. **Guided variation:** Change shape/batch and separate setup or plan creation from execution.
3. **Independent transfer:** repeat the method on a new input or configuration and explain where your original assumptions stop applying.

**Evidence to submit:** Numerical error, throughput, latency, memory, setup cost, and hardware/library versions. Keep raw results and reproduction steps; distinguish measured results from estimates or work not run.

**Self-check:** Which observation would disprove your explanation of the result? Identify a competing explanation and the smallest experiment that separates them. A passing answer names a concrete predicted difference, not only another tool to try.

**Exit criteria:** reproduce the baseline, demonstrate the negative/variation check, and meet the [shared completion rubric](../../Learning-Guide.md#completion-rubric). Use the [lesson cycle](../../Learning-Guide.md#a-repeatable-lesson-cycle) for retrieval practice and spaced review.
