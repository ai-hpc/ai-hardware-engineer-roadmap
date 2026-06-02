# Part 3 · Lecture 02 — Blackwell Hardware Story: B200, B300, GB200 NVL72, Transformer Engine 2, FP4

## Overview

**Blackwell** (Compute Capability 10.0) is the inference silicon for **2025–2027 trillion-parameter MoE serving**. It builds on Hopper but changes **three things** that matter for MoE inference:

1. **FP4 native arithmetic** via Transformer Engine 2 — doubles FP8 throughput, halves HBM bytes for weights.
2. **NVLink 5 + larger NVLink domains (NVL72)** — interconnect for 72 GPUs in one coherent fabric, enabling EP at scale.
3. **HBM3e at 8 TB/s, 192-288 GB per GPU** — the bandwidth and capacity to hold sharded MoE experts close to the compute.

This lecture covers the Blackwell stack as it applies to MoE inference:

1. **Blackwell silicon** — dual-die package, SM architecture, tensor cores.
2. **B200 vs B300** — capacity and bandwidth differences.
3. **Grace+Blackwell (GB200)** — coherent CPU-GPU memory and what it enables.
4. **GB200 NVL72** — the 72-GPU NVLink domain that is the production MoE target.
5. **Transformer Engine 2 and FP4** — microscaling format, scale management.
6. **NVLink 5** — fabric bandwidth, latency, all-to-all primitives.
7. **What changes vs Hopper** for the runtime layer.

By the end you should be able to map DeepSeek V3.1 and Qwen3-MoE 235B-A22B onto a B200 / B300 / GB200 NVL72 deployment and justify the precision + interconnect choices.

---

## 1. Blackwell silicon — what's in the box

### 1.1 The package

Blackwell ships as a **dual-die** package. Each die is comparable to a full Hopper chip; they are connected via a chip-to-chip interconnect (NV-C2C) at **10 TB/s**. To software, the dual-die package **looks like one GPU** with twice the SMs and twice the HBM stacks.

* **Total SMs (B200):** 208 (104 per die × 2)
* **Total tensor cores:** 832 (4 per SM)
* **HBM stacks:** 8 (4 per die)
* **L2 cache:** ~60 MB on-chip (per package)
* **NVLink:** 18 links at 100 GB/s each → 1.8 TB/s per GPU (NVLink 5)

### 1.2 Tensor cores

The 5th-generation tensor cores support:

* BF16, FP16, FP8 (inherited from Hopper)
* **FP6** (E3M2, E2M3 — partial support)
* **FP4** (E2M1, MX-FP4 microscaling — new native operation)
* INT8, INT4 (weight-only)

Peak throughput at each precision:

| Precision | Peak TFLOPs (B200) | Ratio vs FP16 |
|-----------|---------------------|----------------|
| BF16/FP16 | 2,250 | 1.0× |
| FP8 | 4,500 | 2.0× |
| FP6 | 4,500 (same as FP8) | 2.0× |
| FP4 | 9,000 | 4.0× |

**FP4 is the headline:** 4× the FP16 throughput. For MoE workloads where weights dominate HBM bandwidth, FP4 also halves the bytes-per-param read. Both effects compound.

### 1.3 HBM

* **B200 HBM3e:** 192 GB at 8.0 TB/s
* **B300 HBM3e:** 288 GB at ~8.0 TB/s (capacity bump, bandwidth comparable)

The bandwidth ratio vs Hopper:

* H100 HBM3: 3.35 TB/s → B200: 8.0 TB/s — **2.4× more bandwidth**
* H200 HBM3e: 4.80 TB/s → B200: 8.0 TB/s — **1.67× more bandwidth**

For decode (bandwidth-bound), Blackwell is **much faster at the same precision** than Hopper, before FP4 even enters the discussion.

---

## 2. B200 vs B300

The two SKUs ship at different points in 2025–2026:

| Spec | B200 SXM | B300 SXM |
|------|----------|----------|
| HBM capacity | 192 GB | 288 GB |
| HBM bandwidth | 8.0 TB/s | ~8.0 TB/s |
| Peak FP16 TFLOPs | 2,250 | ~2,500 (slightly higher) |
| Peak FP4 TFLOPs | 9,000 | ~10,000 |
| TDP | 1000 W | 1200 W |
| NVLink | NVLink 5 | NVLink 5 |

**B200** is the primary 2025 SKU. **B300** lands in late 2025 / early 2026 with a capacity bump (necessary for the largest MoEs at high batch).

For DeepSeek V3.1 at FP4 (350 GB total):

* B200 192 GB: requires 2× B200 minimum, weights split.
* B300 288 GB: requires 2× B300 minimum.
* For practical batch sizes and KV cache headroom: 4× B200 / B300 or larger.

For Qwen3-MoE 235B-A22B at FP4 (118 GB):

* B200 192 GB: **fits on a single GPU** with KV headroom up to ~16K context.
* B300 288 GB: fits comfortably with long context.

---

## 3. Grace + Blackwell (GB200)

The GB200 superchip pairs **one Grace CPU** with **two Blackwell GPUs** (B200 dies × 2 per Blackwell = 4 dies per GB200) on a single board with **coherent memory**.

### 3.1 Coherent CPU-GPU memory

* The Grace CPU has up to 480 GB of LPDDR5x.
* The two Blackwell GPUs have 192 GB HBM3e each (384 GB total per GB200).
* NV-C2C links the CPU memory to the GPU memory at 900 GB/s in each direction.
* **The GPU can read CPU memory directly** — pinned, large pages, no explicit copy.

For MoE inference, this enables **CPU-resident expert offload**: experts that are rarely activated can live in CPU memory, fetched on demand. This is an **active research topic**; production runtimes are starting to use it but it's not the default.

### 3.2 Why GB200 matters

The GB200 superchip is the **building block of NVL72**. One NVL72 rack has 36 GB200 superchips = 72 Blackwell GPUs + 36 Grace CPUs, all in one NVLink domain.

For 671B MoE serving at scale, **NVL72 is the practical target**. Single-server (8× B200) deployments work for **smaller MoEs and lower scales**.

---

## 4. GB200 NVL72 — the production MoE target

```text
NVL72 Rack:
  - 36× GB200 superchips
  - 72× Blackwell GPUs (B200 × 36 boards × 2 GPUs)
  - 36× Grace CPUs
  - 13.5 TB aggregate HBM3e (72 × 192 GB)
  - 17 TB aggregate Grace LPDDR5x
  - NVLink Switch v4: full all-to-all 1.8 TB/s per GPU
  - Aggregate NVLink bandwidth: ~130 TB/s
  - Liquid cooled
  - ~100 kW power
```

For DeepSeek V3.1 671B at FP4 (350 GB):

* Fits across 4-8 GPUs (one TP+EP partition).
* The other ~64 GPUs in NVL72 host more partitions (data-parallel replicas), enabling massive concurrency.

For an inference workload:

* **Single replica:** 8-16 GPUs with TP × EP combinations.
* **Multiple replicas per NVL72:** 4-8 replicas, each serving independent traffic, sharing the NVLink fabric.

NVL72 also enables **disaggregated prefill / decode** (Lecture 04) by partitioning the 72 GPUs into a prefill pool and a decode pool, connected by the NVLink fabric for KV transfer.

### 4.1 NVL72 vs HGX H100 8-GPU box

The leap is large:

| Property | 8× H100 HGX | GB200 NVL72 |
|----------|-------------|-------------|
| GPUs in domain | 8 | 72 |
| Per-GPU HBM | 80 GB HBM3 | 192 GB HBM3e |
| Per-GPU HBM BW | 3.35 TB/s | 8.0 TB/s |
| NVLink per GPU | 900 GB/s | 1.8 TB/s |
| Total HBM | 640 GB | 13.5 TB |
| Total NVLink BW | 7.2 TB/s | 130 TB/s |

**The 9× larger NVLink domain is what makes MoE EP practical at full scale.** On 8× H100, EP=8 means each GPU holds 1/8 of experts; on NVL72, EP=64 means each GPU holds 1/64. Token routing is cheaper because fewer tokens per expert per step; load balancing is easier.

---

## 5. Transformer Engine 2 and FP4

The most important Blackwell software feature for inference.

### 5.1 What FP4 stores

**E2M1 format:** 1 sign + 2 exponent + 1 mantissa = 4 bits. Representable values: {0, ±0.5, ±1, ±1.5, ±2, ±3, ±4, ±6}. That's 8 values plus signs.

This is a **very coarse** representation. Used alone, parity would be terrible. The trick is **microscaling**.

### 5.2 Microscaling (MX-FP4)

Each block of 32 consecutive elements shares a **shared scale factor** (typically an FP8 E8M0 — just the exponent). The actual value of element i is:

```text
value_i = fp4_value_i × 2^(shared_scale)
```

This gives FP4 **effective dynamic range similar to FP8** — the scale provides 256 possible exponents per block, the FP4 mantissa provides 8 distinct values within that range.

The format is standardized by the Open Compute Project (OCP) under the name **MX-FP4** ([opencompute.org](https://www.opencompute.org/)).

### 5.3 Scale management in TE2

Transformer Engine 2 manages scales automatically:

* **Per-tensor scales** for weights (statistical max-abs over the tensor).
* **Per-block scales** within a tensor (the microscaling factor).
* **Online recalibration** for activations during inference (similar to FP8 in TE1).

This is the FP4-equivalent of TE1's FP8 amax history. For deployment you typically don't manage scales by hand — the TE2 library does it.

### 5.4 Where FP4 hurts

FP4 weights and activations are well-studied. **FP4 KV cache is more aggressive** and typically loses 1–3 pp parity on long-context evals even with per-block scaling. **FP8 KV is the production default**; FP4 KV is for memory-pressure recovery only.

### 5.5 Throughput impact

For Qwen3-MoE 235B-A22B at FP4 on B200 vs FP8 on H200:

| Precision | Hardware | Active param weight read per token | Decode TPOT (batch=1) |
|-----------|----------|--------------------------------------|------------------------|
| FP8 | 4× H200 | 22 GB | ~7 ms |
| FP4 | 4× B200 | 11 GB | ~2 ms (bandwidth ceiling) |

Roughly **3-4× decode improvement** going FP8→FP4 on this model, dominated by the bandwidth gain (8 TB/s vs 4.8 TB/s) combined with the halved precision.

In practice, real runtime efficiency reduces this to ~2.5×. Still very large.

---

## 6. NVLink 5 — the fabric

### 6.1 Per-GPU bandwidth

* **NVLink 5:** 100 GB/s per link, 18 links per GPU → 1.8 TB/s per GPU.
* **NVLink Switch v4:** fully-connected within an NVL72 domain.

This is **2× the per-GPU NVLink bandwidth of Hopper**. For **all-to-all communication** (the dominant cost in MoE EP), this directly translates to **2× faster expert dispatching**.

### 6.2 All-to-all primitives

NCCL provides:

* `nccl_alltoall` — every rank sends one buffer per other rank, one receive per other rank. The canonical MoE primitive.
* `nccl_alltoallv` — variable-size sends, used when tokens-per-expert is uneven.
* `nccl_send` / `nccl_recv` — point-to-point.

On NVL72, an `alltoall` with 64 GPUs and 1 MB per pair is ~1 ms. For MoE EP, this is the per-layer cost; with 61–94 layers, total all-to-all per token can be substantial. We'll diagnose this in Lecture 03.

### 6.3 Latency

Latency for a single 4 KB transfer between any pair of NVL72 GPUs is **sub-microsecond**. For very small transfers (a few hundred bytes — e.g. a single token's hidden state to one expert), the **latency floor dominates over bandwidth**. This is why **batching helps MoE EP** just as much as it helps dense decode.

---

## 7. What changes vs Hopper

For an MoE inference engineer moving from Hopper to Blackwell:

1. **Precision floor drops to FP4.** Plan recipes for both FP8 (Hopper-compatible) and FP4 (Blackwell-only). TRT-LLM's FP4 path is the most mature.
2. **NVLink domain grows from 8 to 72 GPUs.** EP partitioning scales to higher degrees; token routing becomes more efficient per layer.
3. **HBM capacity doubles per GPU** (80→192 → 288 GB across generations). MoE experts that fit on 8× H100 can fit on 2-4× B200/B300.
4. **All-to-all bandwidth doubles** (900 GB/s → 1.8 TB/s per GPU). EP scaling efficiency improves.
5. **Disaggregated P/D becomes practical** within a single NVLink domain — the KV transfer cost between prefill and decode GPUs is small enough.
6. **Grace coherent memory** enables CPU-side expert offload — emerging pattern, not the default yet.

The Hopper recipes from Part 2 (TP + continuous batching + paged KV + speculation + prefix cache) carry forward. The Blackwell-specific recipes (FP4, EP at NVL72 scale, P/D disaggregation, MLA-aware kernels for DeepSeek) are *additions* on top.

---

## Lab — verify your Blackwell stack and bench one matmul at FP4

Goal: same as Lecture 02 from Part 2, but for Blackwell.

1. **Print versions** — driver R580+, CUDA 13.3+ (12.8 was the introduction), cuDNN 9.x, FA4, TE 2.15+.
2. **Pick one FFN matmul shape** from DeepSeek V3.1 or Qwen3-MoE 235B-A22B per-expert FFN — e.g., `(64, 7168) × (7168, 2048)` for DeepSeek.
3. **Measure** with `torch.matmul`:
   * BF16 reference.
   * FP8 via TE2 (`te.Linear` with `fp8_recipe="hybrid"`).
   * FP4 via TE2 (`te.Linear` with `fp8_recipe="fp4"`).
4. **Profile with Nsight Compute** — confirm Blackwell-native instructions (WGMMA + new FP4 ops).
5. **Compare measured TFLOPs** to peak (2,250 BF16 / 4,500 FP8 / 9,000 FP4). Compute efficiency.

Pass criterion: you see ~1.8× FP4 over FP8 and ~3-4× FP4 over BF16 on the FFN shape.

---

## Self-check

1. The H200 has 4.8 TB/s HBM and 990 BF16 TFLOPs (ridge ~206 FLOPs/byte). The B200 has 8.0 TB/s HBM and 2250 BF16 TFLOPs (ridge ~281 FLOPs/byte). For a decode kernel with arithmetic intensity 4 FLOPs/byte at FP8, which GPU has the higher ceiling, and by how much?
2. For DeepSeek V3.1 at FP4 (350 GB weights), how many B200s are needed to hold the full model with 50% headroom for KV + activations + scheduler state at concurrency 32? Show the math.
3. Why does Grace coherent memory matter specifically for MoE rather than for dense? Sketch one workload that uses it.
4. Microscaling FP4 uses per-block scale factors. Why does this help compared to per-tensor scaling for FP4 weights specifically?
5. NVLink 5 doubles per-GPU bandwidth vs NVLink 4. For an MoE EP=8 workload that spends 25% of step time in all-to-all, what is the expected reduction in step time on Blackwell vs Hopper (just from the NVLink improvement)?

---

## References

* NVIDIA Blackwell architecture page — [nvidia.com/en-us/data-center/technologies/blackwell-architecture/](https://www.nvidia.com/en-us/data-center/technologies/blackwell-architecture/)
* NVIDIA GB200 NVL72 product page — [nvidia.com/en-us/data-center/gb200-nvl72/](https://www.nvidia.com/en-us/data-center/gb200-nvl72/)
* Transformer Engine documentation (including FP4) — [docs.nvidia.com/deeplearning/transformer-engine/](https://docs.nvidia.com/deeplearning/transformer-engine/)
* MX (microscaling) format specification — [opencompute.org](https://www.opencompute.org/) (under MX format OCP standard)
* NVLink Switch system whitepapers — [nvidia.com/en-us/data-center/nvlink/](https://www.nvidia.com/en-us/data-center/nvlink/)
* "FP4 Quantization for LLMs" recent papers — search arXiv for "FP4 quantization" in 2024–2025

Cross-references:

* [Phase 5 → GPU Infrastructure → Blackwell-B200-Qwen-Inference](../../../1.%20GPU%20Infrastructure/Nvidia%20GPU/HPC%20Setup/Blackwell-B200-Qwen-Inference/README.md) — Blackwell + Qwen courses going deeper
* [Part 1 → Lecture 03 — Roofline](../Part%201%20-%20Fundamentals/Lecture-03.md) — for Blackwell ridge points

---

## Current as of 2026-06

Blackwell SKUs pinned: B200 SXM (192 GB), B300 SXM (288 GB), GB200 superchip, GB200 NVL72. NVLink 5 (1.8 TB/s per GPU), HBM3e (8.0 TB/s). Refresh when Vera Rubin (post-Blackwell) lands or B400 ships.

---

## Next

* Next: [Lecture 03 — Expert parallelism (EP) and the gating hot path](Lecture-03.md)
* Previous: [Lecture 01 — Anatomy of a modern MoE](Lecture-01.md)
* Up: [Part 3 — MoE at Blackwell](README.md)
