# Blackwell B200 for Qwen Transformer Inference

A 6-chapter special course on running Qwen-class transformer models on NVIDIA's **Blackwell B200** generation: single B200, GB200 superchip, and the NVL72 rack-scale system. Inference only — no training, no fine-tuning.

This series is a Blackwell counterpart to the [Qwen Inference Optimization series](../../../../Track%20C%20-%20Edge%20AI/Qwen%20Inference%20Optimization/README.md) in Edge AI. That series covered Qwen3-4B on Jetson and Qwen2.5-72B on 4×H100. This series picks up where the H100 left off: **what changes when you move Qwen2.5-72B (and bigger) to Blackwell.**

The course follows one chain: inventory the real GPU and fabric, inspect checkpoint numerics, derive a per-GPU memory budget, measure a serving group, then evaluate scale and recovery. Published DGX B200 configurations provide 180 GB per GPU; other Blackwell products require their own specifications. NVL72 is a multi-GPU fabric, not one CUDA device with uniformly accessible memory. [DGX B200 hardware guide](https://docs.nvidia.com/dgx/dgxb200-user-guide/introduction-to-dgxb200.html).

**Review date: 2026-10-14.** The numerical/deployment chapters distinguish teaching calculations from measurements. No throughput, quality loss, or runtime compatibility is claimed without a named artifact and reproducible run.

| Chapter | Title | Focus |
|---|---|---|
| 01 | [Blackwell Architecture](01-Blackwell-Architecture.md) | Dual-die package (one CUDA device), HBM3e, 5th-gen tensor cores and Tensor Memory, NVL72 |
| 02 | [FP4 Numerics & Checkpoint Contracts](02-FP4-Numerics-Transformer-Engine.md) | MXFP4 vs NVFP4, scale metadata, storage accounting, quality validation |
| 03 | [Qwen on a Single B200](03-Single-B200-Qwen-Inference.md) | Qwen2.5-72B fitting in one GPU, layout, single vs dual instance, speculative decoding |
| 04 | [Multi-B200 and NVL72](04-Multi-B200-NVL72.md) | GB200 superchip, NVL72 architecture, legal TP degrees, replicas, placement, NCCL |
| 05 | [Blackwell Kernel Engineering](05-Blackwell-Kernel-Engineering.md) | `tcgen05` + Tensor Memory, Blackwell TMA modes, warp specialization, attention kernels for SM100, CUTLASS |
| 06 | [Production Serving on Blackwell](06-Production-Serving-on-Blackwell.md) | compatibility experiment, goodput, admission, recovery, upgrade evidence |

## Prerequisites

* [Phase 5 — Edge LLM Inference Internals](../../../../Track%20C%20-%20Edge%20AI/Edge%20LLM%20Inference%20Internals/Lecture-01.md) — GEMV vs GEMM roofline math.
* [Phase 5 — Qwen Inference Optimization (full series)](../../../../Track%20C%20-%20Edge%20AI/Qwen%20Inference%20Optimization/README.md) — especially Lecture 04 (Qwen2.5-72B Multi-GPU FP16 on H100).
* [Phase 5 — NCCL Deep Dive](../NCCL-Deep-Dive/README.md) — collective math for the multi-GPU chapters.
* [Phase 5 — CUDA Advanced Optimization](../CUDA-Advanced-Optimization/README.md) — TMA, persistent kernels, warp specialization patterns.

## Scope

* **In scope:** Qwen2.5-72B-Instruct, other explicitly identified Qwen checkpoints (dense or MoE); hypothetical sizes are capacity exercises, not released models on B200 silicon. FP4/FP6/FP8/FP16 dtypes. Single-GPU through NVL72 rack-scale.
* **Out of scope:** training, fine-tuning, LoRA. Hopper (H100/H200) recipes — see the H100 chapter of the Qwen series. AMD MI300X — different chapter, different fabric, different software stack.

## A note on dates and software versions

Use a release whose support matrix matches the exact GPU architecture, model, and format. Save tool/container revisions and test a supported small model first. Hopper FlashAttention-3 is not a blanket prescription for SM100 attention; Chapter 5 studies target-specific kernels. Compare the public examples with your installed CLI help before running commands.

## Practice and assessment

**Entry diagnostic (prerequisite check):** Distinguish FP4 hardware capability from checkpoint-format and kernel support. If this is unfamiliar, use the prerequisite material above or the [roadmap diagnostics](../../../../../Learning-Guide.md#diagnose-before-reading) before starting the lab.

1. **Baseline lab:** Audit a supported model format and establish a correct Blackwell serving baseline.
2. **Guided variation:** Compare precision or placement only after a fixed quality gate passes.
3. **Independent transfer:** repeat the method on a new input or configuration and explain where your original assumptions stop applying.

**Evidence to submit:** Checkpoint/runtime revisions, quality, TTFT/TPOT, useful throughput, memory, and dispatched kernels. Keep raw results and reproduction steps; distinguish measured results from estimates or work not run.

**Self-check:** Which observation would disprove your explanation of the result? Identify a competing explanation and the smallest experiment that separates them. A passing answer names a concrete predicted difference, not only another tool to try.

**Exit criteria:** reproduce the baseline, demonstrate the negative/variation check, and meet the [shared completion rubric](../../../../../Learning-Guide.md#completion-rubric). Use the [lesson cycle](../../../../../Learning-Guide.md#a-repeatable-lesson-cycle) for retrieval practice and spaced review.
