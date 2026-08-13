# Blackwell B200 for Qwen Transformer Inference

A 6-chapter special course on running Qwen-class transformer models on NVIDIA's **Blackwell B200** generation: single B200, GB200 superchip, and the NVL72 rack-scale system. Inference only — no training, no fine-tuning.

This series is a Blackwell counterpart to the [Qwen Inference Optimization series](../../../../Track%20C%20-%20Edge%20AI/Qwen%20Inference%20Optimization/README.md) in Edge AI. That series covered Qwen3-4B on Jetson and Qwen2.5-72B on 4×H100. This series picks up where the H100 left off: **what changes when you move Qwen2.5-72B (and bigger) to Blackwell.**

The short version: a single B200 holds 192 GB of HBM3e at 8 TB/s and supports FP4 tensor cores, so a Qwen2.5-72B-class model fits in **one** GPU with bandwidth-bound decode throughput that requires 4–8 H100s to match. The NVL72 rack scales the same model to extreme batch + long context with one coherent memory image. Software has to keep up — Transformer Engine 2, FP4 microscaling, FlashAttention-3 on 5th-gen tensor cores, TMA enhancements, persistent kernels.

| Chapter | Title | Focus |
|---|---|---|
| 01 | [Blackwell Architecture](01-Blackwell-Architecture.md) | Dual-die package, NVLink-C2C, HBM3e, 5th-gen tensor cores, NVL72 |
| 02 | [FP4 Numerics & Transformer Engine 2](02-FP4-Numerics-Transformer-Engine.md) | MX-FP4/FP6/FP8 microscaling, dynamic per-block quant, quality vs bandwidth |
| 03 | [Qwen on a Single B200](03-Single-B200-Qwen-Inference.md) | Qwen2.5-72B fitting in one die, layout, per-die TP via NVLink-C2C |
| 04 | [Multi-B200 and NVL72](04-Multi-B200-NVL72.md) | GB200 superchip, NVL72 architecture, TP=16/72, NCCL on NVLink-5 |
| 05 | [Blackwell Kernel Engineering](05-Blackwell-Kernel-Engineering.md) | 5th-gen WGMMA, TMA-2, async warp specialization, FlashAttention-3, CUTLASS 4 |
| 06 | [Production Serving on Blackwell](06-Production-Serving-on-Blackwell.md) | TRT-LLM 0.20+, vLLM Blackwell backend, FP4 in prod, benchmarks, cost |

## Prerequisites

* [Phase 5 — Edge LLM Inference Internals](../../../../Track%20C%20-%20Edge%20AI/Edge%20LLM%20Inference%20Internals/Lecture-01.md) — GEMV vs GEMM roofline math.
* [Phase 5 — Qwen Inference Optimization (full series)](../../../../Track%20C%20-%20Edge%20AI/Qwen%20Inference%20Optimization/README.md) — especially Lecture 04 (Qwen2.5-72B Multi-GPU FP16 on H100).
* [Phase 5 — NCCL Deep Dive](../NCCL-Deep-Dive/README.md) — collective math for the multi-GPU chapters.
* [Phase 5 — CUDA Advanced Optimization](../CUDA-Advanced-Optimization/README.md) — TMA, persistent kernels, warp specialization patterns.

## Scope

* **In scope:** Qwen2.5-72B-Instruct, Qwen3-32B/A3B (MoE), and hypothetical Qwen3-300B+ class models on B200 silicon. FP4/FP6/FP8/FP16 dtypes. Single-GPU through NVL72 rack-scale.
* **Out of scope:** training, fine-tuning, LoRA. Hopper (H100/H200) recipes — see the H100 chapter of the Qwen series. AMD MI300X — different chapter, different fabric, different software stack.

## A note on dates and software versions

This series assumes Blackwell-aware software stacks from mid-2026: CUDA 13, cuBLAS 13, TensorRT-LLM 0.20+, vLLM Blackwell backend (PR series merged Q1 2026), Transformer Engine 2.x, FlashAttention 3.x. If you're reading on an earlier toolchain, expect missing kernels and slower-than-quoted numbers.
