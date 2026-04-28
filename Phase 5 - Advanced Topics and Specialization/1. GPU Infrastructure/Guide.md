# 1. GPU Infrastructure (Phase 5)

**Timeline:** 12–24 months (fundamentals); 24–48 months for advanced phase.

**Prerequisites:** Phase 4 Track B (Jetson, CUDA stack), Phase 4 Track C (ML compiler + DL inference optimization).

---

## What this track covers

**HPC** = high-performance computing: solving problems that need massive compute and memory by using many machines (or many GPUs) working together. In AI, "HPC" usually means **large-scale training** and **high-throughput inference** on GPU clusters, not single workstations.

This track is organized around the main accelerator platforms you will evaluate in real HPC and large-scale AI work:

| Sub-track | Focus | Guide |
|-----------|-------|-------|
| **Nvidia GPU** | CUDA, NCCL, NVLink/NVSwitch, InfiniBand, GPUDirect, Slurm/K8s, multi-GPU/multi-node clusters | [Nvidia GPU →](Nvidia%20GPU/Guide.md) |
| **AMD GPU** | ROCm, HIP, RCCL, AMD Instinct (MI300X), RDNA/CDNA architecture, porting CUDA workloads | [AMD GPU →](AMD%20GPU/Guide.md) |
| **Accelerator Platform Evaluation** | Practical comparison of NVIDIA GPUs, AMD GPUs, and Google accelerators for training, inference, scaling, tooling, and cost/performance tradeoffs | [Accelerator Platform Evaluation →](Accelerator%20Platform%20Evaluation/Guide.md) |

For the full CUDA-X library ecosystem (cuBLAS, cuDNN, CUTLASS, TensorRT, NCCL, RAPIDS, and 40+ more), see **[Phase 5B — High Performance Computing](../2.%20High%20Performance%20Computing/Guide.md)**.

---

## How to use this track

1. **Start with [Nvidia GPU](Nvidia%20GPU/Guide.md)** — the dominant ecosystem for AI HPC. Covers fundamentals, virtualization, interconnects, storage, distributed training, and performance modeling.
2. **Add [AMD GPU](AMD%20GPU/Guide.md)** — for portability, alternative hardware, and understanding the growing AMD AI ecosystem (MI300X, ROCm 6+).
3. **Use [Accelerator Platform Evaluation](Accelerator%20Platform%20Evaluation/Guide.md)** when you need to choose between NVIDIA, AMD, and Google accelerator paths for real workloads and real cluster operations.

These tracks assume you've completed Phase 4 Track C (compiler + inference optimization). The HPC content here focuses on **infrastructure, multi-GPU scaling, and distributed systems** — the compiler and kernel optimization skills come from Track C.
