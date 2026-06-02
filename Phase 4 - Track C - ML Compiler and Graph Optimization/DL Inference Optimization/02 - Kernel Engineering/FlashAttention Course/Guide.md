# FlashAttention — A Systems / Kernel Course

<div class="course-identity auto-course" style="--course-accent: #16a34a; --course-accent-rgb: 22, 163, 74;" markdown="1">
<div class="course-identity__icon">FSKC</div>
<div markdown="1">
<p class="course-identity__eyebrow">Deep Dive · Compiler Track</p>
<p class="course-identity__title">Specialized course identity for FlashAttention — A Systems / Kernel Course.</p>
<p class="course-identity__meta">Artifact: compiler/inference optimization · Measure: op count, memory, latency</p>
</div>
</div>


**Parent:** [02 — Kernel Engineering](../Guide.md)

**Format:** 10 lectures. Theory → repo / code reading → small lab. Every lecture ships an artifact (notebook, benchmark script, kernel sketch, correctness test, or patch).

**Why this course exists.** Most "FlashAttention" material online stops at "tiling makes it faster." That is not enough to ship attention kernels. By the end of this course you can read the FlashAttention repository, explain the IO math, debug correctness, benchmark kernels, and modify or integrate attention paths for real LLM inference and training.

---

## What you will be able to do at the end

- Derive the IO complexity of standard attention vs FlashAttention and predict where each one is compute- or memory-bound on a given GPU.
- Implement the online-softmax recurrence in plain NumPy and prove numerical equivalence to a one-shot softmax over the full attention matrix.
- Read `flash-attention/csrc` and trace a `flash_attn_func(...)` call from Python into the launched CUDA kernel.
- Explain the warp / block / sequence / head partitioning differences between FA1, FA2, and FA3.
- Stand up a correctness harness that compares your kernel against PyTorch's reference SDPA with controlled tolerance bands.
- Patch one focused path (e.g. a varlen mask, a small numerics fix, an alternative dispatch) and produce a benchmark + report.

---

## Prerequisites

- Phase 4 Track B — Jetson, CUDA basics, TensorRT.
- Phase 4 Track C Unit 01 — Graph and operator optimization.
- Comfort with linear algebra at the level of multi-head attention, softmax, and matrix multiply.
- A workstation with at least one NVIDIA GPU, CUDA 12.x, PyTorch built against that CUDA, and a clone of `Dao-AILab/flash-attention`.

If you do not have a GPU locally, a single L4 / A10 / 3090 in a cloud instance is enough for lectures 1–7. Lectures 8–10 benefit from H100/H200 access for the FA3 / FP8 lab.

---

## Syllabus

<div class="lecture-map" markdown>

| # | Lecture | Lab artifact |
|---|---------|--------------|
| 1 | [Attention bottleneck + roofline](Lecture%2001%20-%20Attention%20Bottleneck%20and%20Roofline.md) | Notebook: per-shape arithmetic intensity table; roofline plot for one GPU |
| 2 | [Online softmax + numerical correctness](Lecture%2002%20-%20Online%20Softmax%20and%20Numerical%20Correctness.md) | NumPy script proving blockwise-LSE equals one-shot softmax bit-for-bit in fp64; tolerance plot for fp16 / bf16 |
| 3 | [FlashAttention-1 algorithm](Lecture%2003%20-%20FlashAttention-1%20Algorithm.md) | Pseudocode walkthrough + an HBM-byte counter that matches the FA1 paper's IO complexity bound |
| 4 | [GPU kernel performance basics](Lecture%2004%20-%20GPU%20Kernel%20Performance%20Basics.md) | Nsight Compute capture of a coalesced vs uncoalesced copy kernel; warp-reduction microbenchmark |
| 5 | [Repo anatomy + Python / CUDA API](Lecture%2005%20-%20Repo%20Anatomy%20and%20Python%20CUDA%20API.md) | Annotated `flash_attn_func` call trace from Python to launched kernel; minimal `setup.py`-style build recipe |
| 6 | [FlashAttention-2](Lecture%2006%20-%20FlashAttention-2.md) | Benchmark table: FA1 vs FA2 vs PyTorch SDPA across seqlen / head-dim sweep |
| 7 | [Backward pass + validation](Lecture%2007%20-%20Backward%20Pass%20and%20Validation.md) | Correctness harness that compares dQ/dK/dV against a reference with documented tolerances |
| 8 | [Inference path (KV cache, decode)](Lecture%2008%20-%20Inference%20Path%20KV%20Cache%20and%20Decode.md) | Microbenchmark of decode-step latency with and without paged KV; RoPE / GQA sanity check |
| 9 | [Hopper / FA3 / FA4](Lecture%2009%20-%20Hopper%20FA3%20FA4.md) | FA3 vs FA2 benchmark on H100 / H200; identify the WGMMA + TMA + warp-specialization paths in the source |
| 10 | [Capstone](Lecture%2010%20-%20Capstone.md) | One focused kernel-path change + benchmark + correctness + write-up |

</div>

---

## How to use this course

- Do the lectures in order. Each one builds primitives the next assumes.
- Read the listed source files **inside the repo** before reading the lecture's "Build it" section. The lectures are tour guides, not replacements.
- Keep every lab artifact in a single `flash-attn-course/` working directory so you have a personal benchmark archive at the end.
- Treat correctness as a first-class output: every benchmark must have a paired correctness test.

---

## Core sources

- Repo: <https://github.com/Dao-AILab/flash-attention>
- FA1 paper: <https://arxiv.org/abs/2205.14135>
- FA2 paper: <https://tridao.me/publications/flash2/flash2.pdf>
- FA3 blog: <https://tridao.me/blog/2024/flash3/>
- CUTLASS / CuTe: <https://github.com/NVIDIA/cutlass>
- PTX ISA: <https://docs.nvidia.com/cuda/parallel-thread-execution/>
- Nsight Compute: <https://docs.nvidia.com/nsight-compute/>

---

## Role mapping

- **MTS Kernels / DL Inference Optimization Engineer** — direct skill match. The capstone artifact is in the form a hiring manager can read.
- **HPC / Distributed AI Engineer** — needed for understanding the kernel layer that NCCL and friends sit on top of.
- **LLM Inference Engineer (vLLM, TensorRT-LLM, SGLang)** — Lectures 8–10 map directly to the decode-path optimizations these stacks ship.

---

## What this course is not

- Not a CUDA-from-scratch course. Lecture 4 reviews the strict minimum; everything deeper is referenced to Track B and to NVIDIA's own docs.
- Not a transformer-architecture course. We assume you know what Q, K, V, and softmax(QKᵀ/√d)·V are; we focus on how to make that line of math run fast on real hardware.
- Not exhaustive. We intentionally skip FlashAttention forks for non-NVIDIA hardware and quantized-only variants. After this course you should be able to read those repos on your own.
