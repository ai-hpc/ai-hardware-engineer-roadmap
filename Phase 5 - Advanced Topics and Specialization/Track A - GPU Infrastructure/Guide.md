# Phase 5 — Track A: GPU Infrastructure

<div class="course-identity gpu-infra" markdown="1">
<div class="course-identity__icon">GPU</div>
<div markdown="1">
<p class="course-identity__eyebrow">Track 5A · GPU Infrastructure</p>
<p class="course-identity__title">Operate multi-GPU systems, interconnects, distributed runtimes, and accelerator platforms.</p>
<p class="course-identity__meta">Artifact: GPU cluster/runbook · Measure: scaling, bandwidth, utilization, cost</p>
</div>
</div>


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
| **Distributed AI Interconnects** | vLLM, PyTorch distributed, UCX, UCC, NCCL/RCCL, topology-aware multi-node inference, and broken-fabric debugging | [Distributed AI Interconnects →](Distributed%20AI%20Interconnects/Guide.md) |
| **Accelerator Platform Evaluation** | Practical comparison of NVIDIA GPUs, AMD GPUs, and Google accelerators for training, inference, scaling, tooling, and cost/performance tradeoffs | [Accelerator Platform Evaluation →](Accelerator%20Platform%20Evaluation/Guide.md) |

For the full CUDA-X library ecosystem (cuBLAS, cuDNN, CUTLASS, TensorRT, NCCL, RAPIDS, and 40+ more), see **[Phase 5B — High Performance Computing](../Track%20B%20-%20High%20Performance%20Computing/Guide.md)**.

---

## How to use this track

1. **Start with [Nvidia GPU](Nvidia%20GPU/Guide.md)** — the dominant ecosystem for AI HPC. Covers fundamentals, virtualization, interconnects, storage, distributed training, and performance modeling.
2. **Add [AMD GPU](AMD%20GPU/Guide.md)** — for portability, alternative hardware, and understanding the growing AMD AI ecosystem (MI300X, ROCm 6+).
3. **Study [Distributed AI Interconnects](Distributed%20AI%20Interconnects/Guide.md)** when your project crosses vLLM, PyTorch distributed, UCX/UCC, NCCL/RCCL, and non-trivial physical topologies.
4. **Use [Accelerator Platform Evaluation](Accelerator%20Platform%20Evaluation/Guide.md)** when you need to choose between NVIDIA, AMD, and Google accelerator paths for real workloads and real cluster operations.

These tracks assume you've completed Phase 4 Track C (compiler + inference optimization). The HPC content here focuses on **infrastructure, multi-GPU scaling, and distributed systems** — the compiler and kernel optimization skills come from Track C.

## Follow the workload, then choose the infrastructure

Start with one versioned model and a single-device trace from Phase 4. Record which constraint forces expansion: model residency, training state, dataset input, request throughput, or a latency deadline. If none does, a cluster is an optional comparative study rather than a prerequisite for the SoC project.

Follow this chain: [NVIDIA platform](Nvidia%20GPU/Guide.md) → topology and per-rank memory → [collective behavior](Nvidia%20GPU/HPC%20Setup/NCCL-Deep-Dive/README.md) → [placement experiment](Distributed%20AI%20Interconnects/Guide.md) → [platform decision](Accelerator%20Platform%20Evaluation/Guide.md). Port a representative kernel to AMD when portability helps test whether a claimed bottleneck is architectural or software-specific.

**Handoff to chip design:** provide an operator/shape distribution, memory-traffic trace, communication budget, and bottleneck that remains after software tuning. Cluster throughput alone does not define the robot's observation-to-action deadline.

## Practice and assessment

**Entry diagnostic (prerequisite check):** Draw GPU/CPU/NIC topology and explain bandwidth versus message latency. If this is unfamiliar, use the prerequisite material above or the [roadmap diagnostics](../../Learning-Guide.md#diagnose-before-reading) before starting the lab.

1. **Baseline lab:** Establish compute, memory-transfer, and collective baselines on available hardware.
2. **Guided variation:** Change placement or message size and test a recoverable worker failure.
3. **Independent transfer:** repeat the method on a new input or configuration and explain where your original assumptions stop applying.

**Evidence to submit:** Topology, achieved bandwidth/latency, scaling efficiency, error logs, and configuration. Keep raw results and reproduction steps; distinguish measured results from estimates or work not run.

**Self-check:** Which observation would disprove your explanation of the result? Identify a competing explanation and the smallest experiment that separates them. A passing answer names a concrete predicted difference, not only another tool to try.

**Exit criteria:** reproduce the baseline, demonstrate the negative/variation check, and meet the [shared completion rubric](../../Learning-Guide.md#completion-rubric). Use the [lesson cycle](../../Learning-Guide.md#a-repeatable-lesson-cycle) for retrieval practice and spaced review.
