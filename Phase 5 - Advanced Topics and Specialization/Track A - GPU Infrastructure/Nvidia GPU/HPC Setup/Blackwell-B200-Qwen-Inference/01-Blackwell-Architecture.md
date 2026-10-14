# Chapter 1: Blackwell B200 Architecture for Transformer Inference

## Overview

Blackwell B200 is the first NVIDIA datacenter GPU built from **two reticle-sized dies** on a single package, linked by a 10 TB/s on-package interconnect. It carries 192 GB of HBM3e at 8 TB/s, adds **5th-generation tensor cores with native FP4/FP6** support, and slots into a rack-scale NVL72 fabric that puts 72 B200 GPUs and 36 Grace CPUs behind one coherent memory image.

For Qwen-class transformer inference the headline number is bandwidth: **8 TB/s per GPU is 2.4× H100 SXM5 and ~1.9× H200**. Decode is bandwidth-bound; the bandwidth doubles; tok/s roughly doubles. The architectural decisions that make 8 TB/s usable — dual-die scaling, NVLink-C2C between dies, tensor cores that match the new bandwidth — are what this chapter is about.

By the end you should be able to:

* Sketch the B200 package and explain what the dual-die design does and does not change for software (it is one CUDA device).
* Quote per-GPU and aggregate bandwidth, capacity, and compute numbers for Qwen-class workloads.
* Compare B200 against H100, H200, and the still-shipping L40S on relevant inference metrics.
* Place GB200 superchip and NVL72 rack in the same mental model.

---

## 1. The Package: Two Dies, One GPU

```
       ┌────────────────────────────────────────┐
       │                  B200                  │
       │   ┌────────────┐ die-to-die ┌────────────┐│
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

The two dies are connected by a **10 TB/s die-to-die interconnect** on the package. This is not NVLink-C2C, which is the separate 900 GB/s Grace↔GPU link in Grace Hopper and GB200. For comparison, H100 SXM's NVLink 4 is 900 GB/s per GPU. The two dies present themselves to CUDA as **one GPU**: one device, one address space, one pool of 148 SMs. CUDA gives you no control over which die a kernel's blocks run on or which die's HBM an allocation lands in. The hardware spreads both across the dies, and cross-die traffic travels over the 10 TB/s link transparently. What software can observe is non-uniformity. L2 is partitioned between the dies, and an access served by the far die's partition has higher latency than one served locally (Chips and Cheese measured this on B200).

### 1.1 Why this matters for Qwen inference

For Qwen2.5-72B-Instruct at FP4 (~36 GB weights), the model fits comfortably in **one B200**. The KV cache, activations and CUDA context take another 10–20 GB, which leaves most of the 192 GB free. The spare capacity is a GPU-level resource, not a spare die. Typical uses:

* A much larger KV cache and batch, which raises serving throughput.
* A second model instance in the same GPU: two processes sharing the device, or MIG partitions where your driver and configuration support them.
* A draft model for speculative decoding (for example, Qwen3-8B drafting for Qwen2.5-72B).

None of these places work on a particular die, and you cannot run "tensor parallelism across the two dies" of one B200. Tensor parallelism splits a model across separate GPUs (Chapter 4). The dual-die design matters for capacity and bandwidth (192 GB at ~8 TB/s in one device), not as a software-visible partition.

---

## 2. HBM3e: The Bandwidth Story

| Spec | B200 | H200 | H100 SXM |
|---|---|---|---|
| HBM capacity | 192 GB | 141 GB | 80 GB |
| HBM bandwidth | ~8 TB/s | ~4.8 TB/s | ~3.35 TB/s |
| L2 cache total | 126 MB (partitioned across the two dies) | ~50 MB | ~50 MB |
| SM count | 148 (74 enabled per die) | 132 | 132 |

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
* **A new MMA instruction family, `tcgen05`** — replaces Hopper's `wgmma` (which is `sm_90a`-only and does not run on Blackwell). Accumulators live in a dedicated per-SM **Tensor Memory (TMEM)** instead of registers, one thread issues each MMA, and a pair of SMs can cooperate on a single MMA (Chapter 5 §1).

For Qwen inference, the most important number is **throughput at FP4**: a B200 die delivers ~2.25 PFLOPS dense FP4, ~4.5 PFLOPS with structured sparsity. That's roughly 8× H100 FP16 dense throughput, on the same physical area budget.

The compute is rarely the bottleneck for decode (still bandwidth-bound), but it dominates **prefill** and **batched-decode at high concurrency**. A prefill of 4 k tokens on Qwen2.5-72B that takes ~250 ms on H100 takes ~70–90 ms on B200 with FP4 weights and FP8 activations.

### 3.1 What "5th-gen" actually means at the kernel level

* There is no Blackwell WGMMA. `wgmma.mma_async` exists only on `sm_90a`. Blackwell kernels issue `tcgen05.mma`, and the tile shape and operand formats are encoded in a 32-bit instruction descriptor rather than in the mnemonic.
* The Transformer Engine 2 manages scale factors per block automatically, transparently to the CUDA kernel.
* `tcgen05.mma` is the path to Blackwell's tensor-core throughput. It needs Blackwell targets (`sm_100a` / `sm_100f`, CUDA 12.8 or newer) and libraries built for them (cuBLAS, CUTLASS 3.8+).
* Sparsity support is now 2:4 *and* 4:8 structured.

You as an LLM inference engineer rarely write `tcgen05` PTX by hand. Production runtimes (TRT-LLM, vLLM, SGLang) compile down to these instructions via Triton-MLIR, CUTLASS templates, or hand-tuned kernels — but you should know they exist when reading kernel traces.

---

## 4. One GPU, Two Dies: What Software Can and Cannot Do

It is tempting to treat a B200 as two GPUs in one package, but CUDA does not expose it that way:

* **One device.** `nvidia-smi` and `cudaGetDeviceCount` show one GPU per B200. Kernels are scheduled across all 148 SMs, and allocations come from the combined 192 GB.
* **No die affinity.** There is no CUDA API to pin a kernel, a stream or an allocation to die 0 or die 1. So you cannot place half a model on each die, and you cannot run tensor-parallel collectives "across the two dies". Tensor parallelism splits a model across GPUs, connected by NVLink 5 (§5 and Chapter 4).
* **Isolation, not placement.** If you need two isolated slices of one B200, use MIG where your driver and configuration support it, or two processes sharing the GPU. Neither gives you control over which die runs what.
* **What you can see.** Some memory accesses are slower than others. L2 is split into partitions on the two dies, and hits in the far partition cost more latency. For most inference kernels this is second-order next to HBM bandwidth. It shows up as variance in latency microbenchmarks, not as something you schedule around.

The practical rule: size and parallelize B200 deployments as single GPUs with 192 GB and ~8 TB/s each. The die-to-die link is the reason a single kernel can treat both dies' SMs and HBM as one pool. It is not a free intra-package fabric you can program.

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
* The two dies inside one B200 are not separately addressable: `nvidia-smi` and CUDA show one GPU. MIG instances, where supported, are isolation slices, not a handle on the physical dies.

---

## Key Takeaways

| Takeaway | Why it matters |
|---|---|
| B200 is two dies on one package, linked by a 10 TB/s die-to-die interconnect | CUDA still sees one GPU with 148 SMs and 192 GB; there is no die-level placement or intra-package TP |
| 192 GB HBM3e at 8 TB/s | 2.4× H100 bandwidth; decode tok/s roughly tracks this |
| Native FP4 tensor cores | A 70B-class Qwen fits in one GPU at ~36 GB weights |
| L2 is 126 MB, partitioned across the dies | Far-partition hits cost extra latency; you cannot control placement |
| NVL72 rack = 13.8 TB HBM + 30 TB coherent memory | Enables Qwen-300B+ class TP=72 deployments |
| GB200 superchip exposes Grace LPDDR as coherent warm tier | Long-context KV, MoE expert pools, prefix caches can spill cleanly |
| Production runtimes need Blackwell support (CUDA 13, cuBLAS 13, TRT-LLM 0.20+) | Older stacks emulate FP4 or fall back to FP8 paths |

---

## Resources

* **[NVIDIA Blackwell Architecture Whitepaper](https://resources.nvidia.com/en-us-blackwell-architecture):** Official architecture deep dive.
* **[GB200 NVL72 Product Page](https://www.nvidia.com/en-us/data-center/gb200-nvl72/):** Rack-scale platform specs.
* **[HGX B200 Platform Brief](https://www.nvidia.com/en-us/data-center/hgx/):** 8-GPU board spec for non-rack deployments.
* **[OCP Microscaling Formats Specification (MX-FP4/FP6/FP8)](https://www.opencompute.org/documents/ocp-microscaling-formats-mx-v1-0-spec-final-pdf):** The standard behind the new numeric formats.
* **[Blackwell Tuning Guide](https://docs.nvidia.com/cuda/blackwell-tuning-guide/):** Compute capability 10.0 specifics, including the 126 MB L2.
* **[Chapter 2 — FP4 Numerics & Transformer Engine 2](02-FP4-Numerics-Transformer-Engine.md):** The next chapter in this series.
