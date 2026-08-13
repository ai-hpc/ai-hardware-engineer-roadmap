# Chapter 4: Multi-B200, GB200 Superchip, and NVL72

## Overview

Single-B200 deployment covers Qwen2.5-72B at FP4 cleanly. **It doesn't cover:** dense models larger than 192 GB, MoE models with huge expert pools, extreme-batch long-context serving, or sub-30 ms p99 latency on 72B-class chat at scale. For all of those you go multi-B200 — HGX B200 (8-GPU board), GB200 superchip (2 B200 + 1 Grace, coherent), or NVL72 (72 B200 + 36 Grace in one rack).

This chapter is the multi-GPU scaling story for Qwen-class workloads on Blackwell. It covers the fabric, the TP/PP choices, the NCCL hot path, and the new memory tiers that GB200/NVL72 unlock.

By the end you should be able to:

* Pick the right Blackwell platform (HGX vs GB200 vs NVL72) for a given Qwen workload.
* Compute NCCL bandwidth in the decode hot path for TP=8 / TP=16 / TP=72.
* Use Grace LPDDR coherent memory for KV spill, prefix caching, or MoE expert offload.
* Predict serving throughput for Qwen2.5-72B and hypothetical Qwen3-300B+ at rack scale.

---

## 1. The Blackwell Platform Ladder

```
┌────────────────────────────────────────────────────────────────┐
│  Tier 1: Single B200 (~$30k chip + carrier)                    │
│    192 GB HBM  ·  8 TB/s  ·  1.8 TB/s NVLink-5                 │
│    Fits Qwen2.5-72B-MX-FP4 entirely                            │
└────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌────────────────────────────────────────────────────────────────┐
│  Tier 2: HGX B200 (8 × B200, SXM)                              │
│    1.54 TB HBM total                                           │
│    64 TB/s aggregate HBM bandwidth                             │
│    NVSwitch fabric, 130 TB/s bisection                         │
│    Drop-in HGX H100/H200 replacement                           │
└────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌────────────────────────────────────────────────────────────────┐
│  Tier 3: GB200 Superchip (2 × B200 + 1 Grace, NVLink-C2C)      │
│    384 GB HBM + 480 GB LPDDR5X coherent                        │
│    Grace ↔ B200: 900 GB/s coherent                             │
│    Building block of NVL72 racks                               │
└────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌────────────────────────────────────────────────────────────────┐
│  Tier 4: NVL72 Rack (36 × GB200, liquid-cooled)                │
│    72 × B200 + 36 × Grace                                      │
│    13.8 TB HBM + 17.3 TB LPDDR = ~30 TB coherent               │
│    576 TB/s aggregate HBM bandwidth                            │
│    NVLink-5 fabric: 130 TB/s bisection                         │
│    ~120 kW per rack                                            │
└────────────────────────────────────────────────────────────────┘
```

**Decision rule for Qwen workloads:**

| Workload | Right tier |
|---|---|
| Single-user low-latency Qwen2.5-72B | Tier 1 (single B200, intra-package TP=2) |
| Multi-tenant chat, 100s of concurrent users | Tier 2 (HGX B200 8-GPU) |
| Long-context KV spill / prefix cache / MoE expert pool | Tier 3 (GB200, uses Grace coherent memory) |
| Frontier model serving (Qwen3-300B+ / Qwen-MoE) | Tier 4 (NVL72) |
| Trillion-parameter dense models | Tier 4 (NVL72), TP=72 |

Most production Qwen2.5-72B serving lives at **Tier 1 or Tier 2**. NVL72 is overkill for current Qwen lineups but inevitable for what's coming.

---

## 2. HGX B200 — The 8-GPU Standard

The HGX B200 board is the direct successor to HGX H100 SXM and HGX H200. Eight B200 GPUs on a board, all connected through NVSwitch-5.

```
       ┌─────────┐    ┌─────────┐    ┌─────────┐    ┌─────────┐
       │ B200 #0 │    │ B200 #1 │    │ B200 #2 │    │ B200 #3 │
       └────┬────┘    └────┬────┘    └────┬────┘    └────┬────┘
            │              │              │              │
            └──────────────┴───┬──────────┴──────────────┘
                               │
                       ┌───────┴───────┐
                       │   NVSwitch    │   (130 TB/s bisection)
                       │     fabric    │
                       └───────┬───────┘
                               │
            ┌──────────────────┼──────────────────┐
            │                  │                  │
       ┌────┴────┐    ┌────────┴────┐    ┌────────┴────┐    ┌─────────┐
       │ B200 #4 │    │  B200 #5    │    │  B200 #6    │    │ B200 #7 │
       └─────────┘    └─────────────┘    └─────────────┘    └─────────┘

  Aggregate: 1.54 TB HBM3e, 64 TB/s HBM bandwidth, 130 TB/s NVLink fabric
```

For Qwen2.5-72B at **TP=8** the model partitions across all 8 cards:

* Each GPU holds 1/8 of weights (~5.6 GB at MX-FP4-mixed).
* Each GPU holds 1/8 of KV per stored sequence.
* AllReduce after attention and FFN happens on NVLink-5 / NVSwitch.

The key change vs HGX H100: **collective overhead drops dramatically**. NVLink-5 is 2× the per-GPU bandwidth of NVLink-4, and NVSwitch-5 has full bisection at 130 TB/s. For Qwen2.5-72B per-token:

```
Per layer collectives (TP=8):
  AllReduce post-attention:  d_model=8192 × 2 bytes = 16 KB
  AllReduce post-FFN:        d_model=8192 × 2 bytes = 16 KB
                                                    = 32 KB / layer
× 80 layers                                         = 2.56 MB / token

NVLink-5 bandwidth per GPU: 1.8 TB/s
Time per 16-KB AllReduce on NVLink-5: ~4 µs (vs ~10 µs on NVLink-4)
Per-token NVLink-5 time: 2 × 80 × 4 µs = 0.64 ms
```

That 0.64 ms is the new latency floor on collectives. Compare to H100 SXM TP=8's ~1.6 ms floor. The collective overhead is now a smaller fraction of the (faster) compute step, so amortized scaling is even cleaner.

### 2.1 Expected HGX B200 numbers on Qwen2.5-72B

| Metric | TP=4 | TP=8 |
|---|---|---|
| Single-stream decode | ~450 tok/s | ~620 tok/s |
| Batch=32 aggregate decode | ~9,500 tok/s | ~14,000 tok/s |
| Batch=128 aggregate decode | ~22,000 tok/s | ~32,000 tok/s |
| TTFT @ 2k prompt, B=1 | ~15 ms | ~12 ms |
| TTFT @ 32k prompt, B=8 | ~280 ms | ~180 ms |

A single HGX B200 8-GPU box handles **production traffic equivalent to 4–6 HGX H100 8-GPU boxes** for Qwen-72B-class serving. Operationally and economically, this is the inflection point.

---

## 3. The Tensor Parallel Cadence Beyond TP=8

Above TP=8 you start losing the natural alignment with `n_kv_heads = 8` for Qwen2.5-72B. Options:

* **TP=16** — pair off KV heads (each pair of GPUs shares one KV head's slice). Doubles KV memory cost per slice (it's replicated), worth it only when you need the per-stream latency.
* **TP=72** — for frontier models, requires PP-like fragmentation or expert-parallelism (MoE). Pure dense Qwen2.5-72B at TP=72 wastes resources; reserved for Qwen3-300B+ class.

The general rule: **TP should divide `n_kv_heads` evenly.** Qwen2.5-72B with `n_kv_heads = 8` is happy at TP ∈ {1, 2, 4, 8}. Hypothetical Qwen3-MoE with 16 KV heads would extend to TP=16 cleanly.

---

## 4. GB200 Superchip and Coherent Grace Memory

Above HGX sits the **GB200 superchip**: 2 × B200 + 1 × Grace ARM CPU, with NVLink-C2C between the Grace and the B200s.

```
┌──────────────────────────────────────────────────────────┐
│                    GB200 Superchip                       │
│  ┌────────────────────┐   NVLink-C2C   ┌──────────────┐ │
│  │   B200 (2 dies)    │◄══════════════►│   Grace      │ │
│  │   192 GB HBM3e     │   900 GB/s     │   72-core    │ │
│  │   8 TB/s           │   coherent     │   480 GB     │ │
│  │                    │                │   LPDDR5X    │ │
│  └────────────────────┘                └──────────────┘ │
│  ┌────────────────────┐                ┌──────────────┐ │
│  │   B200 (2 dies)    │◄══════════════►│   Grace      │ │
│  │   192 GB HBM3e     │                │   72-core    │ │
│  │   8 TB/s           │                │   480 GB     │ │
│  │                    │                │   LPDDR5X    │ │
│  └────────────────────┘                └──────────────┘ │
└──────────────────────────────────────────────────────────┘
```

The crucial property: **Grace LPDDR is GPU-addressable at memory-coherent speeds**. This creates a two-tier memory hierarchy for inference:

* **Hot tier:** HBM3e — 8 TB/s, 192 GB per B200.
* **Warm tier:** Grace LPDDR — 900 GB/s, 480 GB per Grace.

### 4.1 What you put in the warm tier

Three production-relevant uses for warm-tier Grace memory:

**(a) Long-context KV spill.** When a sequence's KV exceeds HBM budget, page the cold blocks (oldest tokens, lowest-attention-weight blocks) to Grace LPDDR. Hot blocks (recent tokens, high-attention ones) stay in HBM. Production frameworks handle this with policies like LRU or attention-weight-based eviction.

```
Single sequence at 256k context:
  KV per layer per token (FP8) = 2 KB
  × 80 layers × 256k = 41 GB
  → All in HBM? Yes, but eats 20% of budget for one sequence.
  → Spill cold blocks to Grace: keep latest 32k in HBM, page older.
```

**(b) Prefix cache for shared system prompts.** Multi-tenant deployments often share a long system prompt across many requests (RAG context, agent system message, persona). Compute KV once, store in Grace, paste into HBM for each new request. Cuts TTFT for prompt-shared traffic by 10–100×.

**(c) MoE expert pool offload.** For Qwen-MoE variants (or hypothetical Qwen3-MoE), only ~10% of experts are active per token. Keep all expert weights in Grace; stream active experts to HBM each step. Trade some bandwidth for the ability to host much larger expert pools per GPU.

### 4.2 Numbers for warm-tier KV

A worked example: Qwen2.5-72B serving 16 concurrent users at 128k average context.

```
KV per user @ MX-FP8: 80 layers × 2 × 8 KV-heads × 128 head-dim × 128k × 1 byte
                    = 21 GB per user
16 users × 21 GB = 336 GB

Single-B200 HBM budget for KV: ~120 GB
GB200 superchip HBM (2 B200): ~240 GB
GB200 Grace LPDDR: 960 GB

Strategy:
  - Hot 16k tokens per user in HBM: 16 × 2.6 GB = 42 GB (fits HBM)
  - Cold 112k tokens per user in Grace LPDDR: 16 × 18.4 GB = 295 GB (fits Grace)
  - Total: 337 GB, comfortable.

Bandwidth cost per decode:
  Hot KV read (HBM): ~2.6 GB × 16 = 42 GB at 8 TB/s = 5 ms aggregate
  Cold KV read (Grace): ~18.4 GB × 16 = 295 GB at 900 GB/s = 330 ms aggregate
  → Cold-tier reads dominate if you re-read every step.
```

This is why production frameworks **don't** re-read cold KV every step. Instead, recompute KV scores for the cold tier sparingly (e.g., every 16 tokens), or use **sliding-window attention** for the cold tier so it never gets re-read at full bandwidth. Both patterns are how 256k+ context becomes affordable on GB200.

---

## 5. NVL72 — The Rack-Scale Computer

```
NVL72 Rack:
  - 36 GB200 superchips (72 B200, 36 Grace)
  - 18 compute trays, each holding 2 GB200
  - 9 NVSwitch trays
  - Liquid cooling loop
  - ~120 kW power, ~3000 lb
  - 13.8 TB HBM3e + 17.3 TB Grace LPDDR = ~30 TB coherent
  - 576 TB/s aggregate HBM bandwidth
  - 130 TB/s NVLink-5 bisection (full crossbar through NVSwitch)
```

The defining property: **all 72 GPUs are NVLink-5 peers.** Any GPU can read another GPU's HBM at full NVLink-5 speed. Any GPU can read any Grace LPDDR with the same coherent protocol. The entire rack presents as one logical accelerator with 30 TB of memory and 576 TB/s of bandwidth.

### 5.1 What NVL72 is actually for

**Not** for serving Qwen2.5-72B in default config. That fits in one B200. Putting it on NVL72 wastes 71 cards.

**Yes** for:

* **Hypothetical Qwen3-Max / Qwen-1T** — dense models bigger than the GB200's 384 GB. TP=72 puts a 1T-parameter model in one logical device.
* **Massive Qwen-MoE serving** — total expert pool in 100s of GB, with router-driven activation choosing ~10% per token. NVL72 holds the entire MoE in HBM for full bandwidth.
* **Trillion-token concurrent context** — 10,000 concurrent users at 32k context each. Aggregate KV = 320 TB at FP16 (210 TB at FP8). Mostly Grace-resident, with hot tiers in HBM.
* **Extreme-batch frontier serving** — batch=2048+ on Qwen-class large models with continuous batching.

### 5.2 NCCL on NVLink-5 fabric

NVL72's NCCL collectives use the NVSwitch fabric. For TP=72 on a 1T-parameter model (`d_model = 16384`):

```
Per-layer AllReduce: 16384 floats × 2 bytes = 32 KB
× ~100 layers (1T-class) = 3.2 MB / token

NVLink-5 bandwidth in the rack: 130 TB/s bisection
Time per 32-KB AllReduce: ~10 µs at TP=72 (latency-bound)
Per-token NCCL time: 2 × 100 × 10 µs = 2 ms

Compute time per token estimate: ~25 ms
NCCL fraction: 2/27 ≈ 7%
```

Healthy. Above ~TP=72 you'd start being collective-latency-bound, but within one NVL72 rack you have headroom.

---

## 6. Choosing the Right Platform — The Decision Matrix

| Qwen Workload | Recommended Platform | Why |
|---|---|---|
| Qwen2.5-72B, single-user low-latency chat | Single B200, intra-package TP=2 | 280 tok/s in one chip |
| Qwen2.5-72B, multi-tenant chat (100s users) | HGX B200 (1 box, TP=8) | 32k tok/s aggregate, simplest |
| Qwen2.5-72B, multi-tenant with 128k+ context | GB200 superchip | Grace LPDDR for KV spill |
| Qwen-MoE serving | GB200 (small) or NVL72 (large) | Grace holds expert pool |
| Qwen3-Max-class dense (hypothetical 300B+) | NVL72 partial, TP=16-32 | Doesn't fit smaller platforms |
| Frontier 1T+ dense or MoE | NVL72 full rack | TP=72 with NVSwitch crossbar |
| Trillion-token concurrent serving | NVL72 multi-rack | Need coherent memory at scale |
| Research/training adjacency | NVL72 — same hardware as DGX SuperPOD | Reuses the training fleet |

---

## 7. Cost Economics — When Multi-GPU Pays Off

Approximate mid-2026 pricing (chip + integration, not list, not on-prem TCO):

| Platform | Cost | Per-GPU effective cost |
|---|---|---|
| Single B200 carrier | ~$45k | $45k |
| HGX B200 8-GPU board | ~$280k | $35k |
| GB200 superchip | ~$100k | $50k |
| NVL72 rack | ~$3.0M | $42k |

For pure Qwen2.5-72B serving where one B200 suffices, **single-GPU deployment has the best $/tok/s**. For workloads that need multi-GPU collectives (frontier models, extreme concurrency), HGX B200 amortizes the multi-GPU premium across throughput. NVL72 is only economic when the workload demands its scale; for anything smaller, an 8-GPU HGX is cheaper per token served.

The rough heuristic: **stay on the smallest platform that fits your workload at the throughput you need.** Multi-GPU is for "the model literally doesn't fit" or "I need more aggregate bandwidth than 8 TB/s." Don't scale up because it's available.

---

## 8. Diagnostics for Multi-B200 Deployments

```bash
# 1. Confirm NVLink topology
nvidia-smi topo -m
# HGX B200 expected: NV18 (18-link NVLink-5) between every GPU pair

# 2. Confirm NCCL using NVLink not PCIe
NCCL_DEBUG=INFO NCCL_DEBUG_SUBSYS=ALL trtllm-bench ...
# Look for: "NCCL INFO Channel 00 : ... [NVLink]"

# 3. Check NCCL bandwidth in isolation
nccl-tests/build/all_reduce_perf -b 16K -e 1M -f 2 -g 8
# Healthy on HGX B200: 32 KB AllReduce ≤ 6 µs

# 4. On GB200, check Grace coherent path
numactl --hardware
# Should show 2 NUMA nodes per superchip with low cross-node distance

# 5. NVL72: check rack-level NVLink reach
nvidia-smi nvlink --status -l 0
# All 18 links should be Active with 100 Gbps each
```

If any of these reveal degraded connectivity, single-GPU performance can be fine while multi-GPU collapses. Always validate the fabric before bench-marking the model.

---

## Key Takeaways

| Takeaway | Why it matters |
|---|---|
| HGX B200 8-GPU is the natural Qwen2.5-72B serving platform | One box replaces 4–6 HGX H100 boxes for the same throughput |
| GB200 adds 480 GB of coherent Grace LPDDR per GPU | Unlocks long-context KV spill, prefix caching, MoE pools |
| NVL72 is rack-scale: 30 TB coherent memory, 576 TB/s HBM | The platform for frontier dense and MoE models |
| TP should divide `n_kv_heads` — 8 for Qwen2.5-72B | TP=1,2,4,8 are the natural Qwen-72B configurations |
| NVLink-5 collective latency is half NVLink-4's | Collective floor drops to ~0.6 ms/token on HGX B200 |
| Cost-efficient deployment = smallest platform that fits | Don't scale up just because the rack is available |
| Always validate the fabric before benchmarking the model | NCCL on PCIe destroys multi-GPU performance silently |

---

## Resources

* **[NVIDIA GB200 NVL72 Architecture Brief](https://www.nvidia.com/en-us/data-center/gb200-nvl72/):** Official rack-scale platform.
* **[HGX B200 Datasheet](https://www.nvidia.com/en-us/data-center/hgx/):** 8-GPU board spec.
* **[NVIDIA NCCL Documentation](https://docs.nvidia.com/deeplearning/nccl/):** Updated for NVLink-5 / NVSwitch-5.
* **[Chapter 3 — Single-B200 Qwen Inference](03-Single-B200-Qwen-Inference.md):** When you don't need multi-GPU.
* **[Chapter 5 — Blackwell Kernel Engineering](05-Blackwell-Kernel-Engineering.md):** The kernels backing these collectives.
* **[Phase 5 — NCCL Deep Dive](../NCCL-Deep-Dive/README.md):** Companion deep dive on the collective layer.
