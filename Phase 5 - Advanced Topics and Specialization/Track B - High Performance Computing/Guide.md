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

**Prerequisites:** Phase 1 §4 (C++/CUDA), Phase 4 Track B (Jetson, CUDA runtime), Phase 4 Track C (ML compiler + DL inference optimization), Phase 5A (GPU Infrastructure).

---

## What this track covers

High Performance Computing goes beyond single-GPU programming and cluster setup (covered in Phase 5A GPU Infrastructure). This track covers the **GPU-accelerated library ecosystem** and **domain-specific HPC applications** that run on top of that infrastructure.

| Sub-track | Focus | Guide |
|-----------|-------|-------|
| **CUDA-X Libraries** | The full NVIDIA GPU-accelerated library suite: math (cuBLAS, cuFFT), DL (cuDNN, CUTLASS, TensorRT), data (RAPIDS), vision (DALI, CV-CUDA), communication (NCCL, NVSHMEM), parallel algorithms (Thrust, CUB), and 40+ more | [CUDA-X Libraries →](CUDA-X%20Libraries/Guide.md) |

---

## How to use this track

1. **Complete [Phase 5A — GPU Infrastructure](../Track%20A%20-%20GPU%20Infrastructure/Guide.md)** first — covers Nvidia/AMD GPU clusters, multi-GPU networking, Slurm/K8s, distributed training.
2. **Then [CUDA-X Libraries](CUDA-X%20Libraries/Guide.md)** — the comprehensive reference for every GPU-accelerated library. Organized by category (math, DL, data, vision, communication) with role-based learning paths and hands-on projects.

## Relationship to other tracks

| This track provides | Other tracks use it for |
|--------------------|------------------------|
| cuBLAS, cuDNN, CUTLASS | Phase 4C (compiler targets), Phase 5F (AI Chip Design — what hardware must accelerate) |
| NCCL, NVSHMEM | Phase 5A (GPU Infrastructure — multi-GPU communication) |
| TensorRT, FlashInfer | Phase 4B §8 (Jetson runtime), Phase 4C Part 2 (inference optimization) |
| RAPIDS (cuDF, cuML, cuGraph) | Data pipeline acceleration for any ML workflow |
| CV-CUDA, DALI, Video Codec SDK | Phase 5E (Autonomous Vehicles — perception pipelines) |
