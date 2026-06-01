# Part 1 · Lecture 03 — Roofline, Bandwidth, and the Memory Hierarchy

## Overview

Lecture 02 established that decode is bandwidth-bound and prefill is compute-bound. This lecture turns those statements into a quantitative model — the **roofline** — that lets you predict, from a model + GPU pair alone, what the achievable performance ceiling is for any kernel in your workload.

A roofline plot answers one specific question:

> *Given a kernel's arithmetic intensity (FLOPs per byte read from HBM), what is the upper bound on its achievable throughput on this GPU?*

The answer is: the minimum of (intensity × bandwidth) and (peak FLOPs). That ceiling is **physical** — no kernel can exceed it on that hardware. If your measured kernel is far below the ceiling, you have engineering room. If it is at the ceiling, the only way forward is changing the workload (precision drop, larger batches, prefix cache) or the hardware.

This lecture builds the roofline model from first principles:

1. The **memory hierarchy** on Hopper and Blackwell — what each level costs and what holds the working set.
2. The **roofline equation** and how to compute the ridge point for any GPU.
3. **Worked rooflines** for H100, H200, B200, Jetson Orin Nano Super 8 GB.
4. How to **map a kernel onto a roofline** — measure arithmetic intensity, locate it on the chart, identify the ceiling.
5. What roofline cannot tell you — and the next two layers of analysis after it.

By the end you should be able to plot the roofline for any GPU from its datasheet, place any matmul-shaped kernel on it, and predict the performance ceiling before running it.

---

## 1. The memory hierarchy

A GPU is a bandwidth pyramid. Closer to the SM means faster but smaller. Modern Hopper / Blackwell SMs look like this:

```text
                 ┌─────────────────────────────┐
                 │  Registers (per thread)     │  256 × 32-bit = ~1 KB / thread
                 │   ~10s of TB/s effective    │  total per SM: ~256 KB
                 └─────────────────────────────┘
                                │
                 ┌─────────────────────────────┐
                 │  Shared memory / L1 (per SM)│  H100/B200: ~228 KB usable per SM
                 │  ~25 TB/s aggregate         │
                 └─────────────────────────────┘
                                │
                 ┌─────────────────────────────┐
                 │  L2 cache (per GPU)         │  H100: 50 MB · B200: 60+ MB
                 │  ~5 TB/s effective          │
                 └─────────────────────────────┘
                                │
                 ┌─────────────────────────────┐
                 │  HBM (per GPU)              │  H100 80 GB HBM3 @ 3.35 TB/s
                 │                             │  H200 141 GB HBM3e @ 4.8 TB/s
                 │                             │  B200 192 GB HBM3e @ 8.0 TB/s
                 └─────────────────────────────┘
                                │
                 ┌─────────────────────────────┐
                 │  NVLink (cross-GPU)         │  H100 NVLink 4: 900 GB/s per GPU
                 │                             │  B200 NVLink 5: 1.8 TB/s per GPU
                 └─────────────────────────────┘
                                │
                 ┌─────────────────────────────┐
                 │  PCIe / fabric (cross-node) │  PCIe Gen5: 64 GB/s per direction
                 │                             │  IB NDR: 400 Gb/s · XDR: 800 Gb/s
                 └─────────────────────────────┘
```

What lives at each level for LLM inference:

* **Registers + shared memory** — the active matrix tile during a matmul, attention scores being softmaxed. The SM scheduler partitions these between warps.
* **L2 cache** — recently-touched activations, KV cache for the current token if seq is short, frequently-accessed weight tiles. Good kernels reuse L2.
* **HBM** — the weights (gigabytes), the KV cache (gigabytes at long context), activations between layers. **This is the level the roofline cares about.**
* **NVLink** — cross-GPU all-reduce traffic in tensor parallelism, KV transfer in disaggregated P/D.
* **PCIe / IB** — cross-node. Off the critical path for single-replica serving; on the critical path for distributed training and large-cluster inference.

The roofline equation assumes HBM is the bandwidth bottleneck. That assumption is correct for >95% of LLM inference kernels at batch=1.

---

## 2. The roofline equation

A roofline plot has two axes:

* **X axis: arithmetic intensity** (FLOPs per byte read from HBM)
* **Y axis: achieved throughput** (TFLOPs/s)

Two ceilings:

* **Memory ceiling** — `throughput ≤ bandwidth × arithmetic_intensity`
* **Compute ceiling** — `throughput ≤ peak_FLOPs`

The **ridge point** is where they meet:

```text
ridge_point = peak_FLOPs / bandwidth
```

For any kernel with arithmetic intensity below the ridge point: **bandwidth-bound** (memory is the ceiling).
For any kernel above the ridge point: **compute-bound** (FLOPs are the ceiling).

```text
   throughput
       ▲
peak ──┼───────────────────  ← compute ceiling (peak TFLOPs)
       │              ╱
       │           ╱       ← linear region (bandwidth × intensity)
       │        ╱
       │     ╱
       │  ╱
       │╱
       └────┴──────────────► arithmetic intensity (FLOPs/byte)
            ↑
         ridge point
```

For LLM inference, the question is *always*: where is this kernel on this chart? If far below the ceiling, optimization is possible. If at the ceiling, the only progress is to **change the X coordinate** (raise arithmetic intensity by batching, fusion, or precision drop) or **the chart** (different GPU).

---

## 3. Worked rooflines

### 3.1 H100 SXM (80 GB HBM3)

* Peak BF16/FP16 TFLOPs: 989 (tensor cores)
* Peak FP8 TFLOPs: 1,979 (tensor cores)
* HBM3 bandwidth: 3.35 TB/s

Ridge points:

* FP16: 989 × 10¹² / 3.35 × 10¹² = **~295 FLOPs/byte**
* FP8: 1,979 × 10¹² / 3.35 × 10¹² = **~591 FLOPs/byte**

A decode step at batch=1 has arithmetic intensity ≈ 2/bytes_per_param = ~1 FLOP/byte at FP16. That is ~300× below the ridge — deeply bandwidth-bound.

A prefill of a 4K-token prompt with batch=1 has intensity in the hundreds of FLOPs/byte — at or above the ridge, compute-bound.

### 3.2 H200 SXM (141 GB HBM3e)

* Peak BF16/FP16 TFLOPs: 989 (unchanged from H100)
* Peak FP8 TFLOPs: 1,979 (unchanged)
* HBM3e bandwidth: 4.80 TB/s

Ridge points:

* FP16: 989 / 4.80 = **~206 FLOPs/byte** (43% lower than H100)
* FP8: 1,979 / 4.80 = **~412 FLOPs/byte**

H200 lowers the ridge point because bandwidth grew without FLOPs growing. **Bandwidth-bound kernels are 43% faster on H200 than H100** at the same precision. Compute-bound kernels are unchanged — H200's win is decode, not prefill. This is why H200 is the right pick for chat workloads and H100 is the right pick for embedding / batch.

### 3.3 B200 SXM (192 GB HBM3e)

* Peak BF16 TFLOPs: 2,250 (tensor cores, ~2.3× H100)
* Peak FP8 TFLOPs: 4,500 (~2.3× H100)
* Peak FP4 TFLOPs: 9,000 (Transformer Engine 2, native FP4)
* HBM3e bandwidth: 8.0 TB/s

Ridge points:

* FP16: 2,250 / 8.0 = **~281 FLOPs/byte**
* FP8: 4,500 / 8.0 = **~562 FLOPs/byte**
* FP4: 9,000 / 8.0 = **~1,125 FLOPs/byte**

Blackwell's FP4 path moves the ridge point dramatically higher — meaning at FP4 the compute ceiling is far above where most kernels sit. **For bandwidth-bound decode, the FP4 win is in the X coordinate**: arithmetic intensity doubles vs FP8 because bytes per param halve.

### 3.4 Jetson Orin Nano Super 8 GB (edge cross-reference)

* Peak BF16/FP16 TFLOPs: ~67 (Ampere SM 8.7 tensor cores)
* Peak INT8 TFLOPs: ~134
* LPDDR5 bandwidth: ~102 GB/s (0.1 TB/s)

Ridge points:

* FP16: 67 × 10¹² / 102 × 10⁹ = **~657 FLOPs/byte**
* INT8: 134 × 10¹² / 102 × 10⁹ = **~1,314 FLOPs/byte**

Notice the ridge point on Orin Nano is *much higher* than on Hopper — bandwidth is the relatively scarce resource. A 4B-class model decoding at batch=1 on Orin Nano hits the bandwidth wall extremely hard (this is the entire premise of the [Qwen Inference Optimization](../../../3.%20Edge%20AI/Qwen%20Inference%20Optimization/README.md) special course). Batching almost never helps on edge because each user is alone.

### 3.5 Cross-GPU summary

| GPU | Peak FP16 | HBM BW | FP16 Ridge | FP8 Ridge | FP4 Ridge |
|-----|-----------|--------|------------|-----------|-----------|
| Jetson Orin Nano Super 8G | 67 TFLOPs | 102 GB/s | 657 | (INT8: 1314) | — |
| H100 SXM | 989 TFLOPs | 3.35 TB/s | 295 | 591 | — |
| H200 SXM | 989 TFLOPs | 4.80 TB/s | 206 | 412 | — |
| B200 SXM | 2,250 TFLOPs | 8.0 TB/s | 281 | 562 | 1,125 |

**The takeaway:** the ridge point is a property of the (GPU, precision) pair. Choose precision to match the regime your kernel is in.

---

## 4. Mapping a kernel onto a roofline

For each kernel in your workload:

### 4.1 Compute its arithmetic intensity

```text
intensity = FLOPs_per_call / bytes_read_from_HBM_per_call
```

For a `(M, K) @ (K, N)` matmul on `M × K + K × N + M × N` matrix in FP16:

```text
FLOPs       = 2 × M × N × K
bytes       = 2 × (M × K + K × N + M × N)   [if everything streams from HBM]
intensity   = M × N × K / (M × K + K × N + M × N)
```

For a "fat" matmul (M, N, K all large): intensity ≈ MNK / (MK + KN + MN) → grows with the matrix size.
For a "skinny" matmul (decode, M=1): intensity ≈ K × N / (K + KN + N) ≈ 1 in FP16. **Bandwidth-bound.**

### 4.2 Locate it on the chart

Draw a vertical line at the kernel's intensity. The achievable ceiling is where that line meets the lower of the two ceilings:

* If intensity < ridge: ceiling = bandwidth × intensity (bandwidth-bound)
* If intensity > ridge: ceiling = peak FLOPs (compute-bound)

### 4.3 Compare to measured

Run the kernel, measure achieved TFLOPs/s. The ratio (achieved / ceiling) is the **efficiency**.

* > 80%: the kernel is well-optimized; further work must change the kernel's intensity (precision drop, batching, fusion).
* 30–80%: room for kernel improvement — bad memory access pattern, no Tensor Memory Accelerator usage, insufficient occupancy.
* < 30%: usually something structural — Python overhead between launches, no CUDA Graph, kernel launch latency, wrong precision.

### 4.4 The two failure modes the roofline does *not* catch

Roofline is necessary but not sufficient:

1. **Scheduler / Python overhead** — between kernel launches there is CPU time. Roofline assumes 100% kernel time. For very short decodes (<1 ms), the Python loop can dominate.
2. **Warp execution efficiency** — even at high occupancy, divergent branches or memory access patterns can mean each warp does less actual work. Nsight Compute gives this as `sm__sass_thread_inst_executed_per_inst_executed.ratio`.

If your kernel is at 50% efficiency and the roofline says you should be at 80%, run Nsight Compute and check both of these before trying to rewrite the kernel.

---

## 5. The roofline mental check

Five questions a roofline-literate engineer asks before changing anything:

1. **What is my kernel's arithmetic intensity?** Compute it from the math, not measurement first.
2. **Where is the ridge point on this GPU at this precision?**
3. **Is my kernel above or below the ridge?** This tells me which ceiling I am up against.
4. **What is my measured efficiency vs the ceiling?**
5. **Which of the four levers** moves it: precision (changes intensity), batching (changes intensity), fusion (changes bytes), GPU (changes ceiling)?

The answer to question 5 is the entire content of Parts 2 and 3 of this course.

---

## Lab — plot the roofline for your target GPU and three kernels

Goal: produce a single chart, committed to your benchmark repo, with three points on it.

1. **Pick a target GPU** — H100, H200, or whatever you have.
2. **Draw the roofline at FP16, FP8, and INT4** for that GPU. Use the datasheet numbers.
3. **Pick three kernels** from your benchmark:
   * A decode step at batch=1 on a 4B-class model.
   * A prefill step at 2K tokens on the same model.
   * The FFN gate+up matmul at batch=64 (decode regime, larger batch).
4. **Measure** each kernel's FLOPs/sec achieved and HBM bytes read (Nsight Compute → `gpu__time_duration` + `dram__bytes`). Compute arithmetic intensity and achieved TFLOPs.
5. **Plot** the three points on the roofline. Annotate which ceiling each one is bound by.

Pass criterion: the chart goes in your benchmark report. Anyone reading it understands which kernels are bandwidth-bound and which are compute-bound on this GPU.

---

## Self-check

1. The H200 has the same peak FP16 TFLOPs as the H100 but 43% more HBM bandwidth. For a workload that is 80% decode (bandwidth-bound) and 20% prefill (compute-bound), what is the expected end-to-end speedup of H200 over H100? Sketch the math.
2. You run a decode kernel and measure 480 GB/s of HBM read on an H200 (4.8 TB/s peak). What is your bandwidth efficiency? What are two likely reasons the kernel is at this level instead of 80%+?
3. A teammate proposes doubling batch size from 64 to 128 for a 70B-class decode. The roofline says the kernel is at the compute ceiling at batch=64. Predict the throughput change. Defend in one sentence.
4. Why does the FP4 ridge point on B200 (~1125 FLOPs/byte) feel "too high to ever hit"? What workload shape would actually reach it?
5. On Jetson Orin Nano Super 8 GB at INT8 the ridge point is ~1314 FLOPs/byte. A Qwen3-4B decoded at batch=1 has intensity ~2. What is the upper bound on tokens/sec from the roofline alone? What single change (without changing hardware) gets closest to it?

---

## References

* "Roofline: An Insightful Visual Performance Model for Multicore Architectures" — Williams et al., 2009 — the original paper, still authoritative
* NVIDIA H100 architecture whitepaper — [nvidia.com/en-us/data-center/h100/](https://www.nvidia.com/en-us/data-center/h100/)
* NVIDIA H200 product page — [nvidia.com/en-us/data-center/h200/](https://www.nvidia.com/en-us/data-center/h200/)
* NVIDIA B200 / Blackwell whitepaper — [nvidia.com/en-us/data-center/technologies/blackwell-architecture/](https://www.nvidia.com/en-us/data-center/technologies/blackwell-architecture/)
* "Liger Kernel: Efficient Triton Kernels for LLM Training" — practical fused-kernel reading — [arXiv:2410.10989](https://arxiv.org/abs/2410.10989)
* "FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness" — [arXiv:2205.14135](https://arxiv.org/abs/2205.14135) — IO-aware ≈ roofline-aware

Cross-references:

* [Phase 5 → GPU Infrastructure → Blackwell-B200-Qwen-Inference → 01 Blackwell Architecture](../../../1.%20GPU%20Infrastructure/Nvidia%20GPU/HPC%20Setup/Blackwell-B200-Qwen-Inference/01-Blackwell-Architecture.md)
* [Phase 5 → GPU Infrastructure → CUDA Advanced Optimization → 04 Kernel Fusion](../../../1.%20GPU%20Infrastructure/Nvidia%20GPU/HPC%20Setup/CUDA-Advanced-Optimization/04-Kernel-Fusion.md)

---

## Current as of 2026-06

GPU peak numbers from NVIDIA's published datasheets for H100, H200, B200 SXM. Update if NVIDIA releases corrected peaks or if new precision modes ship (FP6, FP3).

---

## Next

* Next: [Lecture 04 — The precision stack, FP16 → FP8 → FP4 → INT4](Lecture-04.md)
* Previous: [Lecture 02 — Transformer execution, from tokens to bits](Lecture-02.md)
* Up: [Part 1 — Fundamentals](README.md)
