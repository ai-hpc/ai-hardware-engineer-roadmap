# Chapter 1: Blackwell B200 Architecture for Transformer Inference

## Overview

Blackwell B200 is the first NVIDIA datacenter GPU built from **two reticle-sized dies** on a single package, linked by a 10 TB/s on-package interconnect. It carries 192 GB of HBM3e at 8 TB/s, adds **5th-generation tensor cores with native FP4/FP6** support, and slots into a rack-scale NVL72 fabric that puts 72 B200 GPUs and 36 Grace CPUs behind one coherent memory image.

For Qwen-class transformer inference the headline number is bandwidth: **8 TB/s per GPU is 2.4× H100 SXM5 and ~1.9× H200**. Decode is bandwidth-bound; the bandwidth doubles; tok/s roughly doubles. The architectural decisions that make 8 TB/s usable — dual-die scaling, NVLink-C2C between dies, tensor cores that match the new bandwidth — are what this chapter is about.

By the end you should be able to:

* Sketch the B200 package and explain how dual-die affects software (TP=2 inside one card).
* Quote per-GPU and aggregate bandwidth, capacity, and compute numbers for Qwen-class workloads.
* Compare B200 against H100, H200, and the still-shipping L40S on relevant inference metrics.
* Place GB200 superchip and NVL72 rack in the same mental model.

---

## 1. The Package: Two Dies, One GPU

```
       ┌────────────────────────────────────────┐
       │                  B200                  │
       │   ┌────────────┐ NVLink-C2C ┌────────────┐│
       │   │   Die 0    │◄══════════►│   Die 1    ││
       │   │ (104 B txn)│  10 TB/s   │ (104 B txn)││
       │   │  4×HBM3e   │            │  4×HBM3e   ││
       │   │  96 GB     │            │  96 GB     ││
       │   │  4 TB/s    │            │  4 TB/s    ││
       │   └────────────┘            └────────────┘│
       │                                          │
       │   208 B transistors total                │
       │   192 GB HBM3e total                     │
       │   ~8 TB/s aggregate HBM bandwidth        │
       │   NVLink 5 external: 1.8 TB/s            │
       └────────────────────────────────────────┘
```

Two reticle-limit dies (~800 mm² each) are co-packaged on a CoWoS-L substrate. Each die has its own SM array, its own L2 cache, and four HBM3e stacks (24 GB each, 1 TB/s each → 4 TB/s per die).

The two dies are connected by **NVLink-C2C** — a wide, low-latency, on-package link that delivers **10 TB/s bidirectional**. For comparison: H100 SXM's NVLink 4 between two cards is 900 GB/s. So inside one B200, the die-to-die bandwidth is more than 10× a single H100-to-H100 NVLink. The two dies present themselves to CUDA as **one GPU**, but software that's aware of the partitioning can do something the H100 generation could not — run **tensor-parallel inference across two dies inside a single GPU**.

### 1.1 Why this matters for Qwen inference

For Qwen2.5-72B-Instruct at FP4 (~36 GB weights), the model fits comfortably on **one** die. The KV cache, activations, and CUDA context take another 10–20 GB. You can run the whole model on one die and keep the second die free for either:

* A second concurrent model instance (doubles serving throughput at the cost of higher per-request latency under load).
* TP=2 of the same model across the two dies, doubling decode bandwidth and shrinking per-stream latency.
* A draft model for speculative decoding (Qwen3-8B drafting Qwen2.5-72B target).

This is a regime that simply doesn't exist on H100/H200 — there, a 70B-class model in FP4/FP8 fits on one card, but you can't subdivide the card further. On B200 the dual-die architecture is a software-visible degree of freedom.

---

## 2. HBM3e: The Bandwidth Story

| Spec | B200 | H200 | H100 SXM |
|---|---|---|---|
| HBM capacity | 192 GB | 141 GB | 80 GB |
| HBM bandwidth | ~8 TB/s | ~4.8 TB/s | ~3.35 TB/s |
| L2 cache total | ~120 MB | ~50 MB | ~50 MB |
| SM count | ~160 (across both dies) | ~132 | 132 |

For decode-bound LLM inference, the relevant comparison is **bandwidth × dtype efficiency**:

| Configuration | Effective bytes/token (Qwen2.5-72B) | Theoretical tok/s (one GPU) |
|---|---|---|
| H100 SXM, FP8 weights, FP16 KV | ~75 GB | ~45 |
| H200, FP8 weights, FP16 KV | ~75 GB | ~64 |
| H100 SXM, FP16 weights | ~145 GB | ~23 |
| H200, FP16 weights | ~145 GB | ~33 |
| **B200, FP4 weights, FP8 KV** | **~36 GB** | **~220** |
| B200, FP8 weights, FP16 KV | ~75 GB | ~106 |
| B200, FP16 weights | ~145 GB | ~55 |

The bandwidth jump and the FP4 capability together produce roughly **5–6×** the single-stream decode rate of an H100, with **the same single-GPU footprint**. This is the deployment-economics inflection point that makes Blackwell interesting for serving 70B-class chat at scale.

---

## 3. 5th-Generation Tensor Cores

Tensor cores have been the dominant matmul accelerator on NVIDIA GPUs since Volta. Blackwell's 5th-gen iteration adds:

* **Native FP4** (E2M1 with shared block scale) — 4 bits per element.
* **Native FP6** (E2M3 / E3M2) — 6 bits.
* **MX (microscaling) formats** — sub-block scales (per-32-element block), shared exponent — the OCP-MX standard.
* **Refined async warp-group MMA (WGMMA)** — Hopper-era WGMMA is extended with better tile sizes, lower register pressure on the FP4/FP6 paths.

For Qwen inference, the most important number is **throughput at FP4**: a B200 die delivers ~2.25 PFLOPS dense FP4, ~4.5 PFLOPS with structured sparsity. That's roughly 8× H100 FP16 dense throughput, on the same physical area budget.

The compute is rarely the bottleneck for decode (still bandwidth-bound), but it dominates **prefill** and **batched-decode at high concurrency**. A prefill of 4 k tokens on Qwen2.5-72B that takes ~250 ms on H100 takes ~70–90 ms on B200 with FP4 weights and FP8 activations.

### 3.1 What "5th-gen" actually means at the kernel level

* WGMMA instructions still take warp groups of 4 warps (128 threads) but with new tile shapes that match FP4 better: `64×16×64` for FP4 vs `64×16×16` for FP16.
* The Transformer Engine 2 manages scale factors per block automatically, transparently to the CUDA kernel.
* The `tcgen05.mma` instruction (Blackwell-only) brings a faster path with reduced register pressure for FP4/FP6 — but requires CUTLASS 4 / cuBLAS 13 to be exposed.
* Sparsity support is now 2:4 *and* 4:8 structured.

You as an LLM inference engineer rarely write WGMMA by hand. Production runtimes (TRT-LLM, vLLM, SGLang) compile down to these instructions via Triton-MLIR, CUTLASS templates, or hand-tuned kernels — but you should know they exist when reading kernel traces.

---

## 4. NVLink-C2C and the "GPU as Two GPUs" Pattern

Inside a B200, the two dies are NVLink-C2C-connected. CUDA exposes them as one device for compatibility, but Blackwell-aware code paths can:

* Pin tensors to a specific die (Die-0 HBM or Die-1 HBM) via NUMA-style placement.
* Do tensor-parallel collectives **across the two dies** using NVLink-C2C as the fabric — 10 TB/s of "free" intra-package bandwidth that doesn't touch the external NVLink.

For Qwen2.5-72B TP=2 inside one B200:

```
Per layer collectives:
  AllReduce post-attention:  d_model floats × 2 bytes
  AllReduce post-FFN:        d_model floats × 2 bytes
  = ~32 KB per layer × 80 layers = 2.56 MB / token

NVLink-C2C bandwidth: 10 TB/s
Roundtrip latency: ~150 ns
Per-token NVLink time: ~0.0003 ms (negligible)
```

In other words: **intra-package TP is essentially free**. The collective time is so small that you get the bandwidth doubling effect of TP=2 without the usual NCCL/NVLink penalty. This is unique to Blackwell.

---

## 5. NVLink 5 and the Off-Package Fabric

For multi-GPU deployments, B200 supports **NVLink 5** at **1.8 TB/s per GPU bidirectional** (across all 18 NVLink-5 ports). That's 2× the per-GPU NVLink bandwidth of H100 SXM.

In an HGX B200 8-GPU board (the direct H100 SXM successor):

```
8 × B200 each with NVLink-5 → NVSwitch
NVSwitch fabric: 130 TB/s aggregate (full bisection)
Total HBM: 8 × 192 GB = 1.54 TB
Total HBM bandwidth: 8 × 8 TB/s = 64 TB/s
```

This is the natural deployment for Qwen2.5-72B-class serving when you want both **high batch throughput** and **high single-stream latency**. With TP=8 you partition the model across 8 B200s; each holds ~1/8th of weights and KV. Aggregate decode bandwidth supports tens of thousands of concurrent streams at long context.

---

## 6. GB200 Superchip — Coherent Grace Memory

Above the single B200 sits the **GB200 superchip**: two B200 GPUs plus one **Grace** ARM CPU on the same NVLink-C2C fabric, with **coherent memory** between them.

```
┌──────────────────────────────────────────────────────────┐
│                       GB200 Superchip                    │
│  ┌─────────────┐   NVLink-C2C   ┌────────────────────┐   │
│  │   Grace     │◄══════════════►│   B200 (Die 0+1)   │   │
│  │ 72-core ARM │    900 GB/s    │  192 GB HBM3e      │   │
│  │ 480 GB LPDDR│    coherent    │                    │   │
│  └─────────────┘                └────────────────────┘   │
│  ┌─────────────┐                ┌────────────────────┐   │
│  │   Grace     │◄══════════════►│   B200 (Die 0+1)   │   │
│  │ 72-core ARM │                │  192 GB HBM3e      │   │
│  │ 480 GB LPDDR│                │                    │   │
│  └─────────────┘                └────────────────────┘   │
│                                                          │
│  Aggregate: 384 GB HBM3e + 960 GB LPDDR5X = 1.34 TB     │
│             coherent across GPUs and CPUs                │
└──────────────────────────────────────────────────────────┘
```

For LLM serving, the relevant property is that **Grace LPDDR memory is GPU-addressable at memory-coherent speeds** (~900 GB/s per Grace). This effectively gives you a tiered memory system:

* Hot tier: HBM3e (8 TB/s, 192 GB per GPU)
* Warm tier: Grace LPDDR (900 GB/s, 480 GB per Grace)

Use cases for the warm tier in inference:

* Spill KV cache for long-context sequences that exceed HBM budget.
* Hold a large MoE expert pool with only the active experts in HBM.
* Cache prefill prefixes / system prompts across many requests.

---

## 7. NVL72 — The Rack as a Computer

The largest Blackwell unit is the **NVL72** rack: **72 B200 GPUs + 36 Grace CPUs** in a single liquid-cooled chassis, fully NVSwitch-fabric-connected.

```
NVL72 Rack:
  - 36 GB200 superchips
    = 72 × B200 + 36 × Grace
  - 18 compute trays × 2 GB200 per tray
  - 9 NVSwitch trays
  - Liquid cooling, ~120 kW per rack
  - Total HBM: 72 × 192 GB = 13.8 TB HBM3e
  - Total Grace LPDDR: 36 × 480 GB = 17.3 TB
  - Total coherent memory: ~30 TB
  - Aggregate HBM bandwidth: 72 × 8 TB/s = 576 TB/s
  - NVLink-5 fabric: 130 TB/s bisection
```

This is the platform for two regimes:

1. **Massive serving** — Qwen2.5-72B at TP=8 replicated 9× across the rack, batch >> 1000, optimized for tokens/sec/$ and tokens/sec/W.
2. **Single-model giants** — frontier 500B+/1T models at TP=72, where you simply can't fit the weights on fewer GPUs.

For the Qwen lineup specifically, NVL72 is overkill for Qwen2.5-72B (which fits on a single B200) but **exactly right** for hypothetical Qwen3-300B+ class models, Qwen-Max-class dense models, or large Qwen-MoE deployments.

---

## 8. Comparison Table — When Does B200 Win?

| Workload | H100 SXM | H200 | B200 | Winner |
|---|---|---|---|---|
| Single-stream Qwen2.5-72B FP16 decode | ~22 tok/s (TP=4) | ~33 tok/s (TP=4) | ~55 tok/s (TP=2) | B200 |
| Single-stream Qwen2.5-72B FP4 decode | n/a (no FP4) | n/a | ~220 tok/s (one GPU) | **B200** |
| Batched serving, B=64, Qwen2.5-72B | ~3,500 tok/s aggregate (8×H100) | ~4,800 tok/s (8×H200) | ~10,000 tok/s (8×B200) | B200 |
| Prefill 32k Qwen2.5-72B | ~2.5 s (TP=4) | ~1.8 s | ~0.6 s (one GPU FP4) | B200 |
| Qwen2.5-72B single-GPU fit | no | barely (FP4 emulated) | yes (FP4 native) | B200 |
| Qwen3-4B edge | n/a (overkill) | n/a | n/a | use Jetson |
| Long-context 131k decode | KV spills early | KV fits with room | KV fits easily | B200 |
| Per-token cost at high load | baseline | ~0.75× | ~0.35× | B200 |

The honest summary: **B200 wins on every Qwen2.5-72B-class workload that can use FP4 in production.** The one case where it doesn't dominate is fine-tuning-only or pre-FP4-validation pipelines where you're still doing FP16 — there, the bandwidth jump still helps but the multiple is smaller (2× not 5×).

---

## 9. Reading a B200 in `nvidia-smi`

Practical orientation:

```bash
$ nvidia-smi --query-gpu=name,memory.total,memory.used,utilization.gpu,power.draw \
             --format=csv,noheader

NVIDIA B200, 196608 MiB, 38912 MiB, 87 %, 925.43 W
```

* **196 608 MiB ≈ 192 GB** — confirms one B200.
* Topology output (`nvidia-smi topo -m`) shows NVLink-5 connectivity between cards; on an HGX B200 you should see `NV18` (18-link NVLink 5) between every GPU pair.
* The two dies inside one B200 are not separately addressable by default in `nvidia-smi`; tools like `nvbandwidth` and CUDA Multi-Instance GPU APIs can probe the C2C link directly.

---

## Key Takeaways

| Takeaway | Why it matters |
|---|---|
| B200 is two dies on one package, linked by 10 TB/s NVLink-C2C | Enables intra-package TP=2 — a regime that doesn't exist on H100/H200 |
| 192 GB HBM3e at 8 TB/s | 2.4× H100 bandwidth; decode tok/s roughly tracks this |
| Native FP4 tensor cores | A 70B-class Qwen fits in one GPU at ~36 GB weights |
| NVLink-C2C collective time is negligible (~0.0003 ms/token) | Intra-package TP is effectively free |
| NVL72 rack = 13.8 TB HBM + 30 TB coherent memory | Enables Qwen-300B+ class TP=72 deployments |
| GB200 superchip exposes Grace LPDDR as coherent warm tier | Long-context KV, MoE expert pools, prefix caches can spill cleanly |
| Production runtimes need Blackwell support (CUDA 13, cuBLAS 13, TRT-LLM 0.20+) | Older stacks emulate FP4 or fall back to FP8 paths |

---

## Resources

* **[NVIDIA Blackwell Architecture Whitepaper](https://resources.nvidia.com/en-us-blackwell-architecture):** Official architecture deep dive.
* **[GB200 NVL72 Product Page](https://www.nvidia.com/en-us/data-center/gb200-nvl72/):** Rack-scale platform specs.
* **[HGX B200 Platform Brief](https://www.nvidia.com/en-us/data-center/hgx/):** 8-GPU board spec for non-rack deployments.
* **[OCP Microscaling Formats Specification (MX-FP4/FP6/FP8)](https://www.opencompute.org/documents/ocp-microscaling-formats-mx-v1-0-spec-final-pdf):** The standard behind the new numeric formats.
* **[NVLink-C2C Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/):** Updated for Blackwell die-aware programming.
* **[Chapter 2 — FP4 Numerics & Transformer Engine 2](02-FP4-Numerics-Transformer-Engine.md):** The next chapter in this series.
