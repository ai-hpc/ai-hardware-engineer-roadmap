# Part 3 · Lecture 04 — Disaggregated Prefill / Decode: Mooncake, Splitwise, DistServe

## Overview

The serving stack from Part 2 Lecture 05 (continuous batching, paged KV, prefix cache, speculation) treats every GPU as homogeneous — every GPU does prefill *and* decode. The 2024–2025 generation of inference research showed this is suboptimal: **prefill is compute-bound, decode is bandwidth-bound, and they have different ideal GPUs and different ideal precisions.**

**Disaggregated prefill / decode (P/D disaggregation)** dedicates one pool of GPUs to prefill and another to decode, transferring the populated KV cache between them. Each pool is optimized for its phase. The result: **2–3× $/MTok improvement** at the cost of cluster complexity.

This lecture covers:

1. **The cost-economics argument** for disaggregation.
2. **Mooncake** — Moonshot's published P/D architecture.
3. **Splitwise** — Microsoft Research's earlier proposal.
4. **DistServe** — academic baseline + open-source implementation.
5. **KV transfer mechanics** — bandwidth, latency, scheduling.
6. **When disaggregation wins** — workload shape, model architecture, cluster size.
7. **NVL72 disaggregation** — practical Blackwell-scale deployment.

By the end you should be able to predict, for a given workload + model + cluster, whether disaggregation pays off, and roughly by how much.

---

## 1. The cost-economics argument

For Llama 3.3 70B FP8 chat workload on 4× H200, colocated continuous batching:

* Each GPU does prefill *and* decode.
* During a long prefill (4K-token prompt, ~250 ms on TP=4), the GPU is at high tensor-core utilization (~70%) but low HBM read utilization (~30%).
* During subsequent decode steps (per-token bandwidth-bound), the GPU is at high HBM utilization (~80%) but low tensor-core utilization (~5%).

In each phase, **~70% of the *other* dimension is wasted**. The GPU is paying for both compute and bandwidth but **only fully using one at a time**.

### 1.1 The disaggregation insight

Run two clusters:

* **Prefill pool** — GPUs optimized for compute. Cheaper GPUs with less HBM bandwidth (e.g., H100 instead of H200) work well because the prefill is FLOP-bound.
* **Decode pool** — GPUs optimized for bandwidth (H200, B200). Compute is not the bottleneck.

The user's request flow:

```text
1. Request arrives → prefill pool
2. Prefill GPU runs the full prefill → produces full KV cache
3. KV cache transferred to decode pool over NVLink (or RDMA)
4. Decode GPU emits tokens
5. New request can join decode batch immediately (separate from any prefill)
```

### 1.2 The economics

If H100s are 2/3 the cost of H200s and prefill is compute-bound:

* **Colocated:** all GPUs are H200 (~$3/hour). 70% efficiency = effective $4.3/hour.
* **Disaggregated:** prefill pool is H100 ($2/hour, near 95% utilization); decode pool is H200 (95% utilization). Average $2.5/hour effective.

Roughly **30-40% $/MTok improvement**. Real measurements from **Mooncake** show similar numbers.

For MoE on Blackwell, the gain is larger because:

* Active params (decode) are different from total params (prefill needs all experts loaded).
* All-to-all communication is more efficient on dedicated decode clusters with constant batch shape.

---

## 2. Mooncake — Moonshot AI's published architecture

[Mooncake](https://arxiv.org/abs/2407.00079) is the production P/D system Moonshot uses to serve Kimi (their flagship LLM). The 2024 paper revealed several engineering insights:

### 2.1 Architecture

```text
                ┌─── Prefill pool ────┐
   user req ───►│  H100 × 8 GPUs      │
                │  TP=8               │
                │  FP8 weights        │
                └─────────┬───────────┘
                          │ KV transfer
                          ▼
                ┌─── Decode pool ─────┐
                │  H200 × 8 GPUs      │
                │  TP=8               │
                │  FP8 weights        │
                │  FP8 KV cache       │
                └─────────────────────┘
```

* **Prefill pool**: lower-cost GPUs (H100 instead of H200). Compute matters more than bandwidth.
* **Decode pool**: higher-bandwidth GPUs. Bandwidth matters more than compute.
* **KV transfer**: via fast network or NVLink fabric.

### 2.2 Key optimizations

1. **KV cache layout** — laid out for cross-pool transfer efficiency. Block-aligned, contiguous per layer.
2. **Prefix cache** — implemented in the decode pool (where it can serve many users). Cross-request prefix matching uses a global index.
3. **Latency-aware scheduling** — when decode pool is at capacity, slow down prefill admission rather than build queues.
4. **KV reuse** — prefill pool computes the full KV; if a similar request arrives later, decode-pool prefix cache can short-circuit the transfer.

### 2.3 Reported gains

From the paper:

* 75% throughput improvement vs colocated at the same hardware cost.
* p99 TTFT improved ~40%.
* Long-context (128K) workloads benefit most.

These are **workload-specific**. Mooncake's gains assume **chat-shape traffic with substantial prefix sharing**.

---

## 3. Splitwise — Microsoft Research

[Splitwise](https://arxiv.org/abs/2311.18677) (2024) is the academic precursor to Mooncake. Same basic insight (split prefill from decode), with different engineering choices:

* **Hardware**: A100 / H100 mix, focused on cost optimization across heterogeneous hardware.
* **KV transfer**: RDMA over InfiniBand, lower bandwidth than NVLink.
* **Scheduling**: lock-step prefill-then-decode rather than asynchronous.

Splitwise's reported gains were similar (~30-50% throughput improvement), but the engineering is older and the NVLink-fabric approach Mooncake uses is more bandwidth-efficient.

---

## 4. DistServe — academic + open-source

[DistServe](https://arxiv.org/abs/2401.09670) (2024) is the academic baseline that formalized many of these ideas. It comes with an open-source reference implementation, integrated into vLLM (experimental) and SGLang (production).

Key ideas:

* **Goodput metric** — defines "useful" throughput as requests-meeting-SLO per second, not raw tokens/sec.
* **Disaggregation only helps if the workload SLO-mix benefits.** Pure throughput workloads don't gain.
* **Heterogeneous-pool scheduling** — explicit modeling of which GPU class belongs to which pool.

DistServe is the conceptual foundation for the disaggregation features now landing in vLLM V1 and SGLang.

---

## 5. KV transfer mechanics

The hardest engineering problem in disaggregation.

### 5.1 The volume

For DeepSeek V3.1 (MLA) at 8K context:

```text
KV per request = 8192 tokens × 69 KB / token ≈ 560 MB
```

For Llama 3.3 70B (GQA) at 8K context:

```text
KV per request = 8192 tokens × 320 KB / token ≈ 2.6 GB
```

For Qwen3-MoE 235B-A22B (GQA, 94 layers, 4 KV heads):

```text
KV per request = 8192 × 192 KB ≈ 1.6 GB
```

**MoE with MLA (DeepSeek) has much smaller KV transfer than GQA models.** This is another way MLA's compressed KV pays off.

### 5.2 The bandwidth

Between two GPUs in the same NVLink domain (NVL72):

* NVLink 5: 1.8 TB/s
* 500 MB transfer: ~0.3 ms
* 2.6 GB transfer: ~1.5 ms

Both are **small relative to the prefill time** (which is hundreds of milliseconds for long prompts). KV transfer cost is **not the bottleneck** within a single NVLink domain.

Between two GPUs across NVLink domains (different NVL72 racks):

* InfiniBand NDR (400 Gb/s) or XDR (800 Gb/s): 50-100 GB/s effective.
* 500 MB: ~10 ms
* 2.6 GB: ~50 ms

Across racks, **KV transfer becomes substantial relative to prefill time** for short prompts. Cross-rack disaggregation is only worth it for long prompts (>4K tokens).

### 5.3 Scheduling

A well-designed disaggregated system pipelines:

1. Prefill GPU does layer 1 → starts transferring layer 1 KV while computing layer 2.
2. Decode GPU receives layer 1 → ready to use as soon as full prefill arrives.
3. By the time prefill finishes layer 61, layers 1–60 are already on the decode GPU.

Effective KV transfer time ≈ **time of last layer's transfer, not all layers** — order-of-magnitude smaller than naive.

This is the SGLang / Mooncake approach. Implementations differ.

---

## 6. When disaggregation wins

A decision table:

| Workload property | Disaggregation gain | Notes |
|-------------------|----------------------|-------|
| Long prompts (>2K) | high | prefill cost matters; compute-bound win |
| Short prompts (<512) | minimal | prefill is small; latency floor dominates |
| Mixed traffic (chat) | moderate | depends on prefill/decode ratio in the mix |
| Heavy prefix sharing | high if decode-pool cache | cache hit reduces prefill volume |
| Pure batch / offline | low | no SLO, colocated continuous-batching wins |
| MoE | higher than dense | active params differ between prefill/decode |
| Hopper-only deployment | low | no FP4 to differentiate decode GPU |
| NVL72 | high | KV transfer near-free within domain |
| Cross-rack | low to moderate | KV transfer is large fraction of TTFT |

### 6.1 The simple test

If your workload has:

* Average prompt length > 2K tokens
* Average output length 50-500 tokens
* Multiple-rack cluster of 16+ GPUs
* Sufficient prefix sharing

→ disaggregation likely wins 20-50% on $/MTok.

If your workload is **short-prompt agent loops or pure batch**, **stay colocated** — the engineering complexity isn't worth it.

---

## 7. NVL72 disaggregation in practice

For DeepSeek V3.1 on NVL72:

```text
72 GPUs total
- Prefill pool: 16 GPUs (TP=2 × EP=8), holding full 671B at FP4
- Decode pool: 48 GPUs (multiple replicas: TP=2 × EP=8, run 3 replicas)
- 8 GPUs: shared infrastructure (scheduler, prefix cache, monitoring)
```

KV transfer between prefill and decode pools is within the NVLink fabric — ~1-3 ms even for 4K-context KVs.

Reported throughput improvement vs colocated (from SGLang benchmarks on similar setups):

* +35-50% tokens/sec at the same hardware cost.
* p99 TTFT improvement: 30-40%.

### 7.1 Configuration in SGLang

```bash
# Prefill server
python -m sglang.launch_server \
    --model-path deepseek-ai/DeepSeek-V3.1 \
    --tp-size 2 --ep-size 8 \
    --disaggregation-mode prefill \
    --port 30001

# Decode server (run multiple)
python -m sglang.launch_server \
    --model-path deepseek-ai/DeepSeek-V3.1 \
    --tp-size 2 --ep-size 8 \
    --disaggregation-mode decode \
    --port 30002 \
    --connect-to-prefill <prefill-host>:30001
```

vLLM V1 has experimental disaggregation support; the production-ready option as of mid-2026 is SGLang.

### 7.2 When NVL72 doesn't pay

For low-concurrency workloads or short-prompt agent loops, NVL72 disaggregation overhead exceeds the gain. **Stay colocated** on 8× B200 or smaller deployments.

---

## Lab — measure disaggregation on Qwen3-MoE

Goal: produce the disaggregation cost-economics report.

1. **Hardware** — NVL72 partition, ideally 16 GPUs (or 8× B200 cluster).
2. **Model** — Qwen3-MoE 235B-A22B FP4 (DeepSeek if you have the cluster).
3. **Runtime** — SGLang 0.5+ with disaggregation.
4. **Baseline** — colocated continuous batching on 8× B200 with concurrency=64.
5. **Candidate** — 4× B200 prefill + 4× B200 decode, same concurrency.
6. **Measure** — throughput, p99 TTFT, p99 TPOT for chat-shape and long-context-shape workloads.
7. **Compute** — $/MTok for each. Use the same hardware cost per replica.

Pass criterion: you can defend whether to ship disaggregation for a specific product workload with measured numbers.

---

## Self-check

1. For Llama 3.3 70B GQA at 8K context, KV transfer is ~2.6 GB. For DeepSeek V3.1 MLA at 8K, it's ~560 MB. Why does MLA help disaggregation specifically?
2. A teammate proposes disaggregating a pure batch workload (no SLO). Defend or reject in two sentences.
3. NVLink 5 makes KV transfer within NVL72 trivial. Why does cross-rack disaggregation still struggle?
4. Mooncake reports 75% throughput improvement. What workload assumptions make this plausible, and where might it fail?
5. At 8× B200 with continuous batching, your p99 TTFT is 800 ms (target 500 ms). You consider disaggregation. What's the first measurement that decides whether to commit?

---

## References

* Mooncake — [arXiv:2407.00079](https://arxiv.org/abs/2407.00079)
* Splitwise — [arXiv:2311.18677](https://arxiv.org/abs/2311.18677)
* DistServe — [arXiv:2401.09670](https://arxiv.org/abs/2401.09670)
* SGLang disaggregation guide — [sgl-project.github.io](https://sgl-project.github.io/)
* "TetriInfer" (recent disaggregation work) — [arXiv:2401.11181](https://arxiv.org/abs/2401.11181)

Cross-references:

* [Part 1 → Lecture 05 — Runtime landscape (disaggregation note)](../Part%201%20-%20Fundamentals/Lecture-05.md)
* [Part 2 → Lecture 05 — Modern serving stack](../Part%202%20-%20Dense%20at%20Hopper/Lecture-05.md) — for the colocated baseline

---

## Current as of 2026-06

Mooncake / DistServe / Splitwise as the canonical 2024–2025 papers. SGLang 0.5+ has production-quality disaggregation; vLLM V1 experimental. Refresh when major new P/D papers land or when vLLM stabilizes the disaggregation API.

---

## Next

* Next: [Lecture 05 — Production MoE serving: MTP speculation, constrained decode, cost model](Lecture-05.md)
* Previous: [Lecture 03 — Expert parallelism](Lecture-03.md)
* Up: [Part 3 — MoE at Blackwell](README.md)
