# Part 2 · Lecture 02 — Hopper Hardware Story: H100, H200, Transformer Engine, FP8

## Overview

The Hopper architecture (Compute Capability 9.0) is the silicon every 2024–2026 production inference deployment runs on by default. Understanding what it provides — and what its successor (Blackwell) inherits and changes — is the foundation of every Part 2 lecture.

This lecture covers:

1. The H100 SXM5 silicon — SM count, tensor cores, HBM3, NVLink 4, NVSwitch fabric.
2. H200 — same silicon, more memory, faster memory.
3. **Transformer Engine** — the FP8 mixed-precision system that defines Hopper's inference advantage.
4. **TMA (Tensor Memory Accelerator)** — asynchronous global → shared memory copies that hide HBM latency.
5. **WGMMA (Warp-Group Matrix Multiply Accumulate)** — Hopper's async tensor core operation.
6. CUDA / cuDNN / FlashAttention-4 — the software stack that exposes the silicon.
7. What this means for Llama 3.3 70B and Qwen 2.5 72B specifically.

By the end you should be able to explain why H200 is "the same H100 with better memory" produces a 30–45% inference speedup on these models — and why TensorRT-LLM with FP8 unlocks the second half of the silicon's potential.

---

## 1. H100 SXM5 silicon — what's in the box

* **132 SMs** (Streaming Multiprocessors). Each SM has 128 FP32 cores, 64 FP64 cores, 4 tensor cores.
* **4th-generation tensor cores** — BF16, FP16, FP8 (E4M3, E5M2), TF32, INT8.
* **80 GB HBM3** at **3.35 TB/s** total bandwidth across 5 HBM stacks.
* **NVLink 4** — 18 NVLink connections per GPU at 50 GB/s each → **900 GB/s** aggregate per GPU.
* **NVSwitch v3 fabric** — fully-connected 8-GPU domains (HGX H100 server), all-to-all 900 GB/s simultaneous.
* **L2 cache** — 50 MB on-chip, shared across all SMs.
* **PCIe Gen5** — 128 GB/s aggregate to the host.

The H100 was the first NVIDIA chip with **native FP8 tensor cores**. This is the single most important inference feature it shipped.

### 1.1 Per-SM features

* **228 KB shared memory + L1** per SM (configurable split).
* **Tensor Memory Accelerator (TMA)** — async DMA engine inside the SM (more in §4).
* **Distributed Shared Memory (DSM)** — direct SM-to-SM access via shared memory, used by FlashAttention 4.
* **Thread Block Clusters** — groups of cooperating thread blocks that can share data via DSM.

These features collectively let kernels overlap matmul compute with memory movement and reach close to the 989 BF16 TFLOPs peak.

---

## 2. H200 — same compute, better memory

H200 is the same H100 die — same SM count, same FLOPs, same NVLink, same NVSwitch. The differences:

| Spec | H100 SXM5 | H200 SXM5 | Δ |
|------|-----------|-----------|---|
| HBM type | HBM3 | HBM3e | newer generation |
| HBM capacity | 80 GB | 141 GB | +76% |
| HBM bandwidth | 3.35 TB/s | 4.80 TB/s | +43% |
| Everything else | (same) | (same) | — |

For our two Part 2 models:

* Llama 3.3 70B at FP16 is 140 GB — *fits on a single H200*. On H100 it requires INT4 or TP=2.
* Qwen 2.5 72B at FP16 is 144 GB — slightly over H200; INT4 or FP8 makes it comfortable.
* At FP8 (70 GB / 72 GB) both fit on a single H200 with KV headroom.

The H200's main inference impact:

1. **Single-GPU 70B-class deployment becomes practical** at FP16 or FP8 with reasonable batch sizes.
2. **Decode TPOT improves ~30–43%** on bandwidth-bound workloads (per Part 1 Lecture 03 roofline).
3. **Long-context serving (128K) becomes practical** without aggressive KV quantization.

H100 is still the workhorse for batch / training. **H200 is the workhorse for chat / decode-dominant workloads.**

---

## 3. Transformer Engine — FP8 mixed precision

**The single most important Hopper software feature for inference.**

Transformer Engine (TE) is NVIDIA's mixed-precision library that uses FP8 tensor cores while keeping FP16/BF16 accumulators and statistics, producing FP16-quality output with FP8 throughput.

### 3.1 The two FP8 formats

| Format | Sign | Exponent bits | Mantissa bits | Range | Use |
|--------|------|---------------|----------------|-------|-----|
| **E4M3** | 1 | 4 | 3 | ±448 | weights, activations (default) |
| **E5M2** | 1 | 5 | 2 | ±57,344 | gradients, KV cache |

E4M3 has more mantissa (precision) and less range. Used where precision matters: weights, forward-pass activations.
E5M2 has more range and less precision. Used where range matters: backward-pass gradients, KV cache (values can be large).

### 3.2 Per-tensor scaling

To use FP8 effectively, each tensor needs a **scale factor** to map its actual dynamic range into FP8's representable range. TE manages this scale factor history:

```text
for each forward pass:
  for each tensor:
    compute max_abs(tensor) → scale = max_repr / max_abs
    store fp8_tensor = tensor / scale × max_repr
    record scale in tensor metadata for matmul
```

Per-tensor scaling is the common case. Per-channel and per-block scaling are options for higher accuracy at higher overhead.

### 3.3 FP8 GEMM via tensor cores

A Hopper tensor core executes:

```text
D = matmul(A_fp8, B_fp8) accumulated in FP32, output written as FP16/BF16 or FP8
```

* **Throughput:** 1,979 TFLOPs FP8 vs 989 TFLOPs BF16 — exactly 2× on H100 / H200.
* **Accuracy:** with proper per-tensor scaling, end-to-end loss is within ~0.1 pp of BF16 on most benchmarks.
* **HBM read:** 1 byte per parameter vs 2 bytes for BF16 — halves bandwidth pressure.

For decode (bandwidth-bound), FP8 weights produce a near-2× throughput improvement on Hopper. **This is the single most important precision drop on Hopper.**

### 3.4 TensorRT-LLM, vLLM, SGLang FP8 paths

* **TensorRT-LLM** — earliest and most mature FP8 path. Uses NVIDIA's TE directly. Best parity, best throughput.
* **vLLM 0.22+** — FP8 weight + activation supported via Marlin kernels (community) and TE integration. Approaching TRT-LLM throughput.
* **SGLang 0.5+** — FP8 support landing; not as mature as the other two for dense models. Catching up on MoE where it focuses its effort.

For Llama 3.3 70B or Qwen 2.5 72B on Hopper, FP8 production deployment in mid-2026 generally means TRT-LLM unless you have a specific reason to use vLLM (multi-modal, multi-LoRA, OpenAI-API flexibility).

---

## 4. TMA — Tensor Memory Accelerator

Hopper introduced an **async DMA engine inside each SM** that loads global memory into shared memory without blocking compute.

Pre-Hopper, loading a matmul tile required threads to issue `ld.global` instructions and synchronize — the SM was waiting on HBM. TMA fires the load, the SM goes back to computing the *previous* tile, and the load completes asynchronously.

### 4.1 Why it matters for inference

* **FFN matmuls** are tiled. Each tile load is ~64–128 KB. Without TMA, this load blocks the SM. With TMA, it overlaps with the previous tile's compute.
* **FlashAttention 4** uses TMA aggressively. It is the key reason FA3 is ~1.5–2× faster than FA2 on Hopper for the same operation.
* **GEMM throughput** on H100 reaches ~90% of peak only when TMA + WGMMA are used together. Older kernels stuck at 60–70%.

You do not write TMA code directly as an inference engineer. You ensure your runtime uses kernels that use it — vLLM 0.22+, SGLang 0.5+, TRT-LLM, FlashAttention 4 all do.

---

## 5. WGMMA — Warp-Group Matrix Multiply Accumulate

The Hopper async tensor-core instruction. Replaces the older WMMA / MMA instructions from Volta / Ampere.

Key properties:

* **Async** — issues the matmul, threads continue, completion fence later.
* **Warp-group** scope (4 warps cooperating) — larger granular operation, fewer launch overheads.
* **Operand staging** — can read directly from shared memory or registers.

For an inference engineer this is again a kernel-internals concern. The user-visible signal is "does this runtime use Hopper-native kernels?" — if yes, you're getting FA3 + WGMMA + TMA + FP8 together. If you're running an old vLLM 0.5.x release without WGMMA kernels, you're leaving 30–40% on the table.

**Verify your kernels:** check Nsight Compute → look at the instruction mix. If you see `WGMMA` instructions, you're on the Hopper path. If you only see older `MMA` instructions, your kernel is pre-Hopper.

---

## 6. The software stack

The Hopper hardware features are only useful through the software stack:

| Layer | Component | Version (this lecture) |
|-------|-----------|------------------------|
| Driver | NVIDIA driver | R555+ for full Hopper feature set |
| CUDA | CUDA toolkit | 12.6+ |
| cuBLAS | matmul library | latest with CUDA 13.3 |
| cuDNN | DNN primitives | 9.x (Hopper-optimized) |
| FlashAttention | attention kernels | FA3 (3.0+) |
| Transformer Engine | FP8 mixed precision | 1.10+ |
| NCCL | collectives | 2.20+ |
| Triton | kernel DSL | 3.0+ |

If any of these is older, you're not getting the full Hopper inference advantage. **The first checklist item on every new Hopper deployment: verify the driver and CUDA versions.** Outdated drivers regress FP8 performance silently.

---

## 7. What this means for Llama 3.3 70B and Qwen 2.5 72B

Bringing it back to the anchor pair:

### 7.1 H100 80G (single GPU)

* **FP16:** doesn't fit (140 GB).
* **FP8:** fits tight (~70 GB weights), KV cache pressure forces FP8 KV. Practical for low-batch chat.
* **INT4:** fits (~35 GB), more KV headroom, batch up to 16 at 4K context.

Recipe: **INT4 weights, FP16 activations, FP16 KV**. For higher throughput: **FP8 weights/activations** via TRT-LLM.

### 7.2 H200 141G (single GPU)

* **FP16:** fits both (140 GB / 144 GB), tight on KV.
* **FP8:** comfortable, ~70 GB weights leaves ~70 GB for KV + activations. Best single-GPU recipe.
* **INT4:** comfortable, large batch / long context.

Recipe: **FP8 weights/activations via TRT-LLM**, FP8 KV for long context. This is the H200's sweet spot.

### 7.3 4× H100 80G (TP=4)

* Per-GPU weight share ~35 GB at FP16. Comfortable.
* Cross-GPU all-reduce per layer dominates if NCCL isn't tuned. We will profile this in Lecture 04.

Recipe: **FP16 or FP8 weights/activations**, FP16 KV. TP=4 for production chat.

### 7.4 8× H200 141G (TP=8)

* Per-GPU weight share ~18 GB. Comfortable for very long context or very high batch.
* All-reduce is more frequent (TP=8 doubles the collective frequency vs TP=4). Net throughput per GPU sometimes lower than TP=4.

Recipe: **FP16 native** for max parity, INT4 for max throughput. Best for *batch* workloads where the long-context KV headroom matters.

---

## Lab — verify your Hopper software stack and measure FP8 vs FP16 on one GEMM

Goal: confirm your stack uses Hopper-native kernels, then measure the FP8 vs FP16 speedup on a real matmul shape from Llama 3.3 70B.

1. **Print versions** — driver, CUDA, cuDNN, FlashAttention, TE. Commit a `versions.txt` to the bench repo.
2. **Pick one FFN matmul shape** from Llama 3.3 70B at batch=16: `(16, 8192) × (8192, 28672)`.
3. **Measure** with `torch.matmul`:
   * FP16 reference.
   * BF16 reference.
   * FP8 via TE (`te.Linear`).
4. **Profile with Nsight Compute** — confirm WGMMA instructions on the FP8 path, FA3 if attention is in scope.
5. **Compare measured TFLOPs** to peak. Achieved efficiency = measured / peak.

Pass criterion: you see 1.6–1.9× FP8 speedup over FP16 on the FFN shape, and your stack uses WGMMA + TMA. If not, upgrade something.

---

## Self-check

1. The H200 has the same compute as H100 but 43% more HBM bandwidth. For Llama 3.3 70B INT4 decode at batch=1 (bandwidth-bound), what speedup do you predict? For prefill at 2K tokens (compute-bound)?
2. Why is E4M3 used for weights and E5M2 for KV cache in many FP8 deployments?
3. A teammate ships a vLLM deployment using kernels that show only legacy MMA instructions in Nsight Compute (no WGMMA). Estimate the performance left on the table. Why is upgrading the runtime the highest-leverage action?
4. Your driver is R535, CUDA 12.2. Your FP8 path is 1.3× faster than FP16 instead of the expected 1.8×. What's most likely wrong?
5. For Qwen 2.5 72B on 1× H200 with FP8 weights + FP8 KV + 16K context, estimate (a) weight memory, (b) KV memory at batch=8, (c) headroom for activations. Will this configuration fit?

---

## References

* NVIDIA H100 Tensor Core GPU Architecture whitepaper — [nvidia.com/en-us/data-center/h100/](https://www.nvidia.com/en-us/data-center/h100/)
* NVIDIA H200 product page — [nvidia.com/en-us/data-center/h200/](https://www.nvidia.com/en-us/data-center/h200/)
* NVIDIA Transformer Engine documentation — [docs.nvidia.com/deeplearning/transformer-engine/user-guide/](https://docs.nvidia.com/deeplearning/transformer-engine/user-guide/)
* "FlashAttention-4: Fast and Accurate Attention with Asynchrony and Low-precision" — [arXiv:2407.08608](https://arxiv.org/abs/2407.08608)
* CUDA 12.x Hopper Programming Guide — [docs.nvidia.com/cuda/](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html)
* NCCL documentation — [docs.nvidia.com/deeplearning/nccl/](https://docs.nvidia.com/deeplearning/nccl/)

Cross-references:

* [Phase 5 → GPU Infrastructure → Blackwell-B200-Qwen-Inference → 01 Blackwell Architecture](../../../1.%20GPU%20Infrastructure/Nvidia%20GPU/HPC%20Setup/Blackwell-B200-Qwen-Inference/01-Blackwell-Architecture.md) — successor architecture
* [Phase 5 → GPU Infrastructure → CUDA-Advanced-Optimization → 05 Warp Specialization](../../../1.%20GPU%20Infrastructure/Nvidia%20GPU/HPC%20Setup/CUDA-Advanced-Optimization/05-Warp-Specialization.md) — async warp patterns

---

## Current as of 2026-06

Stack versions pinned: driver R580+, CUDA 13.0+, FA3 3.0+, TE 2.x. Refresh when CUDA 13 / FA4 / TE 2.x lands or when a Hopper successor (post-Blackwell) ships.

---

## Next

* Next: [Lecture 03 — Quantizing Llama 3.3 70B and Qwen 2.5 72B](Lecture-03.md)
* Previous: [Lecture 01 — Anatomy of a 70B-class dense model](Lecture-01.md)
* Up: [Part 2 — Dense at Hopper](README.md)
