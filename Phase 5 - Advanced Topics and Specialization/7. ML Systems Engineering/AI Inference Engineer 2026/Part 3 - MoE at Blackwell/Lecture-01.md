# Part 3 · Lecture 01 — Anatomy of a Modern MoE: DeepSeek V3.1 and Qwen3-MoE 235B-A22B

## Overview

Mixture-of-Experts changes the inference economics in one specific way: **total parameters and active parameters become two different numbers**, and the runtime's job is to use the second one well.

A dense 72B model on Hopper does 72B params of work per token. A 235B-A22B MoE does 235B of params *worth of memory* but only 22B of *compute per token* — because the **gating network routes each token to a small fraction of experts**. The **HBM pressure looks like a 235B model**; the **FLOPs look like a 22B model**. This shifts the bandwidth/compute ratio sharply: MoE decode is *even more* bandwidth-bound than dense decode at the same active-param scale.

This lecture walks the two anchor models side by side:

* **DeepSeek V3.1** — 671B total / 37B active, MLA attention, MTP head, 256+1 experts, top-8 routing.
* **Qwen3-MoE 235B-A22B** — 235B total / 22B active, GQA attention, 128 experts, top-8 routing.

Topics:

1. The shared MoE skeleton — what every modern MoE ships.
2. **MLA** — DeepSeek's KV compression, the most distinctive 2025 architectural choice.
3. **MTP** — DeepSeek's native multi-token prediction, free speculation built into the model.
4. The expert layer — count, sizing, top-k routing, shared experts.
5. The two models in concrete numbers — config, total params, active params, KV per token, decode bandwidth ceiling.
6. What changes in the inference graph compared to dense (Part 2 Lecture 01).
7. Why MoE inference economics differ — and what the runtime has to do differently.

By the end you should be able to read either model's config and predict its inference cost shape on Blackwell — total HBM footprint, per-token decode bandwidth, per-token compute, and where the runtime has to choose differently from a dense deployment.

---

## 1. The shared MoE skeleton

Both DeepSeek V3.1 and Qwen3-MoE 235B-A22B implement the same canonical mixture-of-experts block. The transformer layer changes only in the FFN:

```text
input hidden states (h)
       │
       ▼
   RMSNorm
       │
       ▼
   Attention block   (MLA for DeepSeek; GQA for Qwen3-MoE)
       │
       ▼
   residual add
       │
       ▼
   RMSNorm
       │
       ▼
   ┌─────────────── MoE FFN ─────────────────┐
   │  gating network: g(h) → expert scores   │
   │  topk(scores, k=8) → selected experts   │
   │  for each selected expert e:            │
   │      out_e = expert_e(h)                │
   │  output = sum_e (gate_score_e × out_e)  │
   │  (+ shared expert output, DeepSeek)     │
   └─────────────────────────────────────────┘
       │
       ▼
   residual add → next layer
```

The compute pattern:

* **The gating network** is a small Linear layer (hidden → num_experts) that runs every step. It is cheap (FLOPs negligible) but **introduces a token-routing decision** every layer.
* **Each token activates k experts** out of N — top-8 of 256 for DeepSeek, top-8 of 128 for Qwen3-MoE. So 8/256 ≈ 3.1% of expert capacity is touched per token, or 8/128 = 6.25% for Qwen3-MoE.
* **The shared expert** (DeepSeek only) is a small Linear path that runs for *every* token. It absorbs the "common knowledge" that all tokens need.
* **The output** is a weighted sum of the selected experts' outputs.

What "active" means in the param count: **only the experts the token actually touched** count toward the per-token compute. The other 248 (or 120) experts **sit in HBM but contribute nothing** this step.

---

## 2. MLA — Multi-head Latent Attention (DeepSeek)

The most distinctive 2024–2025 architectural choice. **MLA compresses the KV cache by ~15× compared to full MHA, and ~4× compared to the GQA configs typical of 70B-class models (§2.2)**, which is the single biggest inference win at long context.

### 2.1 The idea

Standard GQA stores `K, V ∈ R^(L × h_kv × head_dim)` per token — the explicit key/value vectors. MLA instead stores a small **latent vector** `c_kv ∈ R^(L × d_c)` per token, where `d_c << h_kv × head_dim`.

At attention time, K and V are *reconstructed from the latent*:

```text
K_t = W_uK · c_t      (W_uK: d_c → h_kv × head_dim, low-rank up-projection)
V_t = W_uV · c_t
```

The latent `c_t` is the stored cache. K and V are recomputed on-the-fly each decode step. This is a **memory/compute tradeoff**: more compute (a small matmul per layer per step) for **dramatically less KV memory**.

One subtlety: **the rotary embedding cannot ride inside the latent**. RoPE applies a position-dependent rotation to each key, and that rotation does not commute with the low-rank up-projection `W_uK` — folding position into the compressed latent would mean re-rotating the cached value for every new query position. DeepSeek's fix is a **decoupled rotary key**: a small 64-dim key per token (`qk_rope_head_dim = 64`), rotated once and cached alongside the latent. The latent stays position-free; the rotary key carries the position.

### 2.2 KV bytes per token — MLA vs GQA

For DeepSeek V3 / V3.1:

| Spec | Value |
|------|-------|
| `kv_lora_rank` (d_c) | 512 |
| `qk_rope_head_dim` (decoupled rotary key) | 64 |
| `num_hidden_layers` | 61 |
| `bytes_per_element` | 2 (FP16/BF16) |

```text
mla_kv_bytes_per_token = num_layers × (d_c + qk_rope_head_dim) × bytes
                       = 61 × (512 + 64) × 2
                       = 70,272 bytes
                       ≈ 69 KB / token
```

For a hypothetical 64-head, 8-KV-head, head-dim-128 GQA model with 61 layers:

```text
gqa_kv_bytes_per_token = 2 (K + V) × 61 × 8 × 128 × 2
                       = 249,856 bytes
                       ≈ 244 KB / token
```

MLA is **~3.5× smaller** than equivalent GQA — and DeepSeek V3 has **more layers (61) than its dense competitors** so the absolute savings are larger.

At 128K context, DeepSeek V3.1 needs **~9 GB of KV cache per request**, versus ~30 GB+ for a GQA equivalent. This is what makes **long-context serving practical without aggressive KV quantization**.

### 2.3 What MLA costs

* Extra matmul per decode step: `W_uK · c_t` and `W_uV · c_t`. Two small matmuls per layer per step.
* On Blackwell at FP4, these are cheap — the compute fits comfortably in the spare cycles between the FFN matmul and the attention.
* The runtime needs MLA-aware kernels — FlashAttention 4 with the MLA shape, or specialized kernels in SGLang's DeepSeek path.

vLLM, SGLang, and TensorRT-LLM all support MLA as of mid-2026, with SGLang having the most mature DeepSeek-specific kernels.

---

## 3. MTP — Multi-Token Prediction (DeepSeek)

DeepSeek V3 introduced MTP as a *training-time* objective: predict the next *k* tokens, not just the next one. At inference time, the model has learned to also emit predictions for positions +1, +2, ..., +k.

### 3.1 How it changes inference

```text
Standard decode step:           emits one token
With MTP active:                emits up to k tokens, with confidence scores

Verification:
  At step t, target model has emitted tokens t+1, t+2, t+3 (k=3)
  Continue from t+3 unless any of them is rejected by sampling logic
```

This is **native speculative decoding** built into the model — no separate draft model needed.

For DeepSeek V3.1:

* Acceptance rate (rough): 60-80% at k=3 (sample-dependent).
* Effective throughput: 1.6-2.5× decode speedup with no additional draft cost.
* Works in vLLM 0.22+ (`speculative_config={"method": "deepseek_mtp"}`) and SGLang.

### 3.2 Why this matters for inference engineering

MTP makes DeepSeek's decode throughput on Blackwell roughly equivalent to a dense model 2× smaller than its active param count would suggest. **At constant cost, DeepSeek V3.1 with MTP serves ~3× more tokens than a 37B dense model would.** That math is what makes DeepSeek's $/MTok economics competitive with closed flagship models in 2025-2026.

### 3.3 Qwen3-MoE has no native MTP

Qwen3-MoE 235B-A22B was **not trained with the MTP objective**. It uses **a lightweight EAGLE-3 head (or Medusa heads)** that reuses the target model's hidden states. Acceptance rates are similar (70-80% at k=4) but the engineering layer is different: a separately trained head, separate runtime path, separate fine-tuning.

This is one of the cleanest "architectural choice → inference recipe" contrasts in the course.

---

## 4. The expert layer

### 4.1 DeepSeek V3.1 expert configuration

From `config.json`:

| Field | Value |
|-------|-------|
| `n_routed_experts` | 256 |
| `n_shared_experts` | 1 |
| `num_experts_per_tok` | 8 |
| `moe_intermediate_size` | 2048 (per expert FFN) |
| `routed_scaling_factor` | 2.5 |
| `first_k_dense_replace` | 3 (first 3 layers are dense, no MoE) |

So:

* 61 total layers; first 3 are standard dense FFN, the remaining 58 use MoE.
* Each MoE layer has 256 + 1 = 257 experts.
* Each token routes to 8 of 256 routed experts plus the 1 shared expert.
* Each expert is a SwiGLU FFN with intermediate size 2048 (much smaller than DeepSeek V3's dense FFN, which is 18432 wide — ~9× larger).

Param count per MoE layer:

```text
Per expert: 3 × hidden × moe_intermediate (gate + up + down)
          = 3 × 7168 × 2048
          = 44M params
Per layer (routed): 256 × 44M = 11.2B
Per layer (shared): 1 × 44M = 44M
Per layer (gating): hidden × num_experts = 7168 × 256 = 1.8M
Per layer total: ~11.3B

58 MoE layers + 3 dense layers + attention layers + embed/LM head
≈ 671B total params
```

Per-token active params:

```text
Per token, per MoE layer:
  Attention: ~110M (MLA-shaped attention)
  8 routed experts × 44M + 1 shared × 44M + gating = ~352M + ~44M + ~2M ≈ ~398M
  Per-layer active: ~510M

58 MoE layers × 510M + 3 dense layers × dense-FFN-size + attention + embed
≈ 37B active params per token
```

Which matches the published "37B active." 

### 4.2 Qwen3-MoE 235B-A22B expert configuration

From `config.json`:

| Field | Value |
|-------|-------|
| `num_experts` | 128 |
| `num_experts_per_tok` | 8 |
| `moe_intermediate_size` | 1536 |
| (no shared experts) | — |

So:

* All layers are MoE (no first-k-dense-replace).
* 128 experts per layer, top-8 routing.
* Per-expert FFN intermediate 1536.
* No shared expert.

The result: ~235B total params, ~22B active per token.

### 4.3 What this means for the runtime

The runtime has to:

* **Hold all experts in HBM** — both models need their full expert weights resident across the cluster.
* **Route every token through the gating network** — small but per-layer per-token.
* **Move tokens to their selected experts** — this is the all-to-all communication problem when experts are partitioned across GPUs (Lecture 03).
* **Compute only the active experts' FFN** — 8 of 256 (3.1%) for DeepSeek, 8 of 128 (6.25%) for Qwen3-MoE.

A dense model has a simple "1 GPU, all the weights" mental model. An MoE has "all experts everywhere or partition them, and route tokens at runtime." **The complexity moves from compute density to communication and scheduling.**

---

## 5. The two models in concrete numbers

### 5.1 DeepSeek V3.1

| Spec | Value |
|------|-------|
| Total params | 671B |
| Active params (per token) | 37B |
| Layers | 61 (3 dense + 58 MoE) |
| Hidden size | 7168 |
| Attention | MLA, 128 heads (Q), `q_lora_rank=1536`, `kv_lora_rank=512` |
| Vocab | 129,280 |
| Context | 128K (extended via YaRN) |
| KV bytes/token (FP16) | ~69 KB |
| Native speculation | MTP |
| Total HBM (BF16) | ~1.4 TB |
| Total HBM (FP8) | ~700 GB |
| Total HBM (FP4) | ~350 GB |

### 5.2 Qwen3-MoE 235B-A22B

| Spec | Value |
|------|-------|
| Total params | 235B |
| Active params (per token) | 22B |
| Layers | 94 |
| Hidden size | 4096 |
| Attention | GQA, 64 Q heads, 4 KV heads (per Qwen3 release) |
| Vocab | 151,936 |
| Context | 256K (Instruct-2507) |
| KV bytes/token (FP16) | 2 × 94 × 4 × 128 × 2 ≈ 192 KB |
| Native speculation | None (use EAGLE-3 / Medusa) |
| Total HBM (BF16) | ~470 GB |
| Total HBM (FP8) | ~235 GB |
| Total HBM (FP4) | ~118 GB |

### 5.3 Side by side

| Property | DeepSeek V3.1 | Qwen3-MoE 235B-A22B |
|----------|---------------|----------------------|
| Total / active params | 671B / 37B | 235B / 22B |
| Active param ratio | 5.5% | 9.4% |
| Attention | MLA (compressed KV) | GQA (4 KV heads) |
| KV bytes/token | ~69 KB | ~192 KB |
| Layers | 61 | 94 |
| Native speculation | MTP (yes) | none (EAGLE-3 external) |
| Experts | 256 + 1 shared | 128 |
| HBM at FP4 | ~350 GB | ~118 GB |

DeepSeek has more total params but compressed KV; Qwen has fewer total but more KV per token. Choosing one over the other depends on:

* Long-context priorities → DeepSeek MLA wins.
* HBM-constrained deployment → Qwen3-MoE 235B-A22B fits more easily.
* Need MTP speculation out of the box → DeepSeek.
* Multilingual coverage (Chinese especially) → both competitive; Qwen3 slightly stronger on Chinese benchmarks.

---

## 6. Inference graph changes vs dense

Compared to Part 2's dense models, the inference graph adds:

### 6.1 New stages per layer

```text
Dense block FFN:                       MoE block FFN:
  RMSNorm                                RMSNorm
  gate matmul + up matmul                gating linear → top-k
  SwiGLU                                 token → expert assignment
  down matmul                            (if EP) all-to-all dispatch
  residual                               per-expert FFN computation (only selected)
                                         (if EP) all-to-all combine
                                         weighted sum of expert outputs
                                         (+ shared expert path, DeepSeek)
                                         residual
```

The extra stages:

1. **Gating computation** — small but per-token per-layer.
2. **Token routing** — assigns each token to its experts, builds the dispatch table.
3. **(EP-only)** All-to-all communication to move tokens to their experts and back.
4. **Per-expert FFN** — same FFN math as dense, but only on a subset of tokens per expert.
5. **Output combination** — weighted sum of expert outputs.
6. **(MTP)** Multi-token head produces k+1 logit predictions per step.

### 6.2 The new bottleneck candidates

* **Gating compute** — usually small but can become measurable on Hopper-class hardware at high batch sizes.
* **All-to-all communication** — the dominant new bottleneck (Lecture 03).
* **Expert load imbalance** — if some experts get many more tokens than others, the slowest expert sets the step time.
* **KV cache attention** — for DeepSeek MLA, the up-projection compute adds; for Qwen3-MoE, the standard GQA attention cost.

---

## 7. Why MoE economics differ

A simplified cost model:

```text
$/MTok ≈ (replica_cost_per_hour × hours_per_MTok)
       = (replica_cost) / (output_tokens/sec × 3600 / 10^6)
```

For dense (Part 2): the model has to do P × 2 FLOPs per token, where P is the model size, mostly read from HBM at decode time.

For MoE: the model has to *hold* P_total in HBM but only *do* P_active × 2 FLOPs per token. Decode bandwidth still pays for the active params' weights, but the HBM cost is much higher.

### 7.1 The bandwidth math

Decode at batch=1 for DeepSeek V3.1 at FP4 on one B200 (just hypothetically, ignoring it doesn't fit):

```text
bytes read per decode step:
  Active weights: 37B × 0.5 bytes (FP4) = 18.5 GB
  KV cache (128K, FP16 MLA): 9 GB
  Total: ~27.5 GB

Decode time on B200 (8 TB/s HBM):
  ≈ 27.5 GB / 8 TB/s ≈ 3.4 ms / token
  ≈ ~300 tokens/sec at batch=1 (theoretical ceiling)
```

In practice on GB200 NVL72 with proper batching:

* MoE serving wins from batching because the cross-token expert routing amortizes the all-to-all cost.
* Bandwidth ceiling per GPU is more strict than dense because total HBM held is high.
* Real measurements at SGLang on GB200 NVL72: ~600-1000 tok/s/GPU at concurrency 64 for DeepSeek V3.1 FP4.

### 7.2 The $/MTok comparison

The headline numbers (mid-2026 ballpark; replicate in your lab):

| Model | Hardware | Active | Throughput (tok/s/GPU) | $/MTok |
|-------|----------|--------|------------------------|--------|
| Llama 3.3 70B FP8 | 4× H100 | 70B | ~580 | ~$1.20 |
| Qwen 2.5 72B FP8 | 4× H100 | 72B | ~550 | ~$1.26 |
| Qwen3-MoE 235B-A22B FP4 | 8× B200 | 22B | ~1200 | ~$1.27 |
| DeepSeek V3.1 FP4 + MTP | 16× B200 (NVL72) | 37B | ~1500 (effective with MTP) | ~$1.02 |

**MoE on Blackwell is the cost-economics winner** once the cluster scale is available — but it **requires the cluster scale**. For a single-replica deployment, **dense Hopper still wins**.

---

## Lab — derive both configs and produce a side-by-side cost table

Goal: extend the benchmark repo with MoE-aware cost modeling. Cap of one day.

1. **Download both `config.json` files** from official Hugging Face repos.
2. **Compute, programmatically**, for each model:
   * Total params, active params per token.
   * Per-token KV cache bytes (MLA-aware for DeepSeek, GQA-aware for Qwen3-MoE).
   * HBM footprint at BF16, FP8, FP4 (weights + KV at 128K, batch=1).
   * Decode bandwidth ceiling on B200 single GPU (assuming weights fit, batch=1).
3. **Render a comparison table** with both models, three precisions.
4. **Predict** at what batch size the decode workload moves from bandwidth-bound to compute-bound (using Part 1 Lecture 03 ridge-point math, FP4 ceiling ~1125 FLOPs/byte on B200).
5. **Pick** an HBM-feasible deployment shape for each model (single B200, 2× B200, 4× B200, 8× B200, or NVL72) at FP4 + FP8 KV with concurrency targets of 16, 64, 256.

Pass criterion: the report can be reproduced by another engineer from public configs, and your predictions match measured numbers in Lectures 02-05.

---

## Self-check

1. MLA reduces DeepSeek V3.1's per-token KV from ~244 KB (hypothetical GQA equivalent) to ~69 KB. At 128K context, how many concurrent requests fit in the KV cache budget of one B200 (192 GB) versus the GQA hypothetical?
2. MTP gives DeepSeek native speculation at ~2× effective decode throughput. Why is it specifically not transferable to Qwen3-MoE without retraining?
3. A teammate proposes serving DeepSeek V3.1 on 8× H200 instead of 8× B200 to save cost. Without running it, predict the throughput drop. What's the bottleneck?
4. Why does Qwen3-MoE 235B-A22B have 94 layers and DeepSeek V3.1 only 61, but DeepSeek's active params (37B) is *higher* than Qwen3's (22B)?
5. For a long-context (128K) chat product at batch=8, which model has the larger total HBM footprint at FP4? Show the math.

---

## References

* DeepSeek V3 technical report — [arXiv:2412.19437](https://arxiv.org/abs/2412.19437)
* DeepSeek V3.1 release notes — [github.com/deepseek-ai/DeepSeek-V3](https://github.com/deepseek-ai/DeepSeek-V3)
* DeepSeek V3.1 model card — [huggingface.co/deepseek-ai/DeepSeek-V3.1](https://huggingface.co/deepseek-ai/DeepSeek-V3.1)
* MLA — described in DeepSeek V2 paper [arXiv:2405.04434](https://arxiv.org/abs/2405.04434)
* DeepSeek-MoE paper — [arXiv:2401.06066](https://arxiv.org/abs/2401.06066)
* Multi-token prediction — [arXiv:2404.19737](https://arxiv.org/abs/2404.19737)
* Qwen3 technical report — [qwenlm.github.io/blog/qwen3/](https://qwenlm.github.io/blog/qwen3/)
* Qwen3-MoE 235B-A22B model card — [huggingface.co/Qwen/Qwen3-235B-A22B](https://huggingface.co/Qwen/Qwen3-235B-A22B) (model name as released)
* "Mixture of Experts Explained" — [huggingface.co/blog/moe](https://huggingface.co/blog/moe) — accessible primer
* Switch Transformers — [arXiv:2101.03961](https://arxiv.org/abs/2101.03961) — the foundational MoE paper

Cross-references:

* [Part 2 → Lecture 01 — Anatomy of a 70B-class dense model](../Part%202%20-%20Dense%20at%20Hopper/Lecture-01.md) — for direct dense-vs-MoE comparison
* [Phase 5 → GPU Infrastructure → Long-Context-MoE-Foundation-Training → 04 MoE Fundamentals](../../../1.%20GPU%20Infrastructure/Nvidia%20GPU/HPC%20Setup/Long-Context-MoE-Foundation-Training/04-MoE-Fundamentals.md) — training-side perspective on MoE

---

## Current as of 2026-06

Configs pinned from the official model cards: DeepSeek V3.1 (2025-08) and Qwen3-235B-A22B-Instruct-2507 (2025-07). Refresh when DeepSeek V4 or Qwen4-MoE ships, or when a competing MoE family with different architectural choices becomes the production standard.

---

## Next

* Next: [Lecture 02 — Blackwell hardware story](Lecture-02.md)
* Up: [Part 3 — MoE at Blackwell](README.md)
