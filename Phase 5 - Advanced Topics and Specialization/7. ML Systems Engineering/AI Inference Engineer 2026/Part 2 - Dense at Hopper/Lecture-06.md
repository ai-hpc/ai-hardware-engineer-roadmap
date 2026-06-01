# Part 2 · Lecture 06 — Long Context at 128K on Hopper

## Overview

Both Llama 3.3 70B and Qwen 2.5 72B publish 128K context windows. At 128K the cost shape of inference changes — what dominated TPOT at 4K (weight bandwidth) is joined or overtaken by **KV bandwidth** and **prefill compute**. Long-context serving is a distinct engineering problem with its own precision recipes, scheduling choices, and parity bars.

This final Part 2 lecture covers:

1. The cost math at 128K — KV memory, KV bandwidth per decode step, prefill FLOPs.
2. **YaRN context extension** — how 128K is unlocked from a base 32K model, and what it costs at inference.
3. **Chunked prefill** at 128K — what it solves and where it breaks.
4. **FP8 KV cache at scale** — when it ships, what parity costs, per-head scaling discipline.
5. **Prefix sharing on long system prompts** — the highest-leverage long-context optimization.
6. **Long-context evaluation** — RULER, needle-in-haystack, what to actually measure.
7. **Production recipes** at 128K on Hopper.

By the end you should be able to deploy either anchor model at 128K context on Hopper, defend the precision recipe with parity numbers from RULER, and ship a benchmark report that lets a teammate reproduce it.

---

## 1. The cost math at 128K

Revisiting the KV cache math from Part 1 Lecture 02:

```text
kv_bytes_per_token = 2 × L × num_kv_heads × head_dim × bytes
                   = 2 × 80 × 8 × 128 × 2 (FP16)
                   = 320 KB/token
```

At 128K context, per request:

| Precision | KV cache size |
|-----------|---------------|
| FP16 KV | 320 KB × 131072 = **42 GB** |
| FP8 KV | 21 GB |
| INT4 KV | 10.5 GB |

This is for **one request**. At batch=8 with FP16 KV: 336 GB. That fits on 4× H100 80G (320 GB total) only by aggressive paging — and leaves nothing for weights or activations.

**At 128K context, batch size is HBM-limited far before it is compute-limited.**

### 1.1 KV bandwidth per decode step

At each decode step:

```text
kv_read_bytes = 2 × L × num_kv_heads × head_dim × seq_len × bytes
              = kv_bytes_per_token × seq_len
              = 320 KB × 131072
              = 42 GB
```

On H200 (4.8 TB/s HBM3e), the KV read alone takes:

```text
kv_read_time = 42 GB / 4.8 TB/s ≈ 8.7 ms
```

Plus the weight read (~30 ms at FP16 for 140 GB / 4.8 TB/s, ~15 ms at INT4). So decode at 128K context on a single H200, FP16 KV, INT4 weights:

```text
TPOT ≈ kv_read_time + weight_read_time + compute + overhead
     ≈ 8.7 + 15 + small + small
     ≈ 25 ms per token
```

versus ~12 ms at 4K context. **TPOT roughly doubles at 128K** because the KV bandwidth becomes a co-dominant cost.

### 1.2 Prefill cost at 128K

```text
prefill_flops ≈ 2 × P × prompt_tokens
              = 2 × 70 × 10^9 × 131072
              ≈ 1.83 × 10^16 FLOPs
              = 18.3 PFLOPs
```

On 4× H100 at FP16 (4 × 989 TFLOPs / GPU = 3.96 PFLOPs aggregate, assuming 80% efficiency = 3.17 PFLOPs effective):

```text
prefill_time ≈ 18.3 / 3.17 ≈ 5.8 seconds
```

Almost six seconds of prefill before the first token decodes. **For an interactive product this is unshippable** without chunked prefill or prefix cache.

### 1.3 Attention cost at 128K

Attention is `softmax(Q · K^T) · V`. At 128K context with batch=1:

```text
qk_flops = 2 × q_heads × q_seq × kv_seq × head_dim
         = 2 × 64 × 1 × 131072 × 128 (for one decode step)
         ≈ 2.1 × 10^9 FLOPs
```

A single decode-step attention op is 2 GFLOPs. Compute-trivial. But the *KV read* for attention is 42 GB / decode step (per §1.1). So attention at long context is **purely bandwidth-bound**.

For prefill (batch=1, prompt=128K), attention is O(prompt² × head_dim × heads) which becomes ~2 × 64 × 131072 × 131072 × 128 = 277 TFLOPs — comparable to the FFN cost. FlashAttention 4's O(N) memory complexity is essential here.

---

## 2. YaRN context extension

Both Llama 3.3 70B and Qwen 2.5 72B were *trained* at shorter context (4K-32K base) and **extended to 128K via YaRN** ([arXiv:2309.00071](https://arxiv.org/abs/2309.00071)).

### 2.1 What YaRN does

YaRN rescales RoPE frequencies so the model can attend over longer ranges without retraining from scratch. The result: a model trained at 8K can serve at 128K with only ~few epochs of fine-tuning.

The runtime impact is minimal:

* The RoPE matrix is rebuilt with extended frequencies. No code change beyond `rope_scaling` in config.
* No additional kernel overhead.
* The published 128K context is YaRN-extended; for `max_position_embeddings > base_context`, ensure the runtime applies the right scaling.

### 2.2 Where YaRN can go wrong at inference

* **Wrong scaling factor:** if the runtime applies a different YaRN scaling than the model was fine-tuned with, long-context recall regresses. Always use the official `rope_scaling` from `config.json`.
* **Quantization interaction:** rescaled RoPE values have a wider numeric range; FP8 KV per-tensor scaling may clip. Per-head scaling avoids this.

### 2.3 Long-context quality at 128K

Published benchmarks (RULER, NIAH):

| Model | 4K | 32K | 64K | 128K |
|-------|----|-----|-----|------|
| Llama 3.3 70B | ~96 | ~92 | ~89 | ~80 (sharper drop) |
| Qwen 2.5 72B | ~96 | ~93 | ~91 | ~85 |

Qwen 2.5 72B's long-context behavior at 128K is generally stronger than Llama 3.3 70B in published RULER numbers, though both meaningfully degrade vs. their short-context performance. For products that rely on accurate retrieval at 100K+, this is a model-selection signal independent of inference engineering.

---

## 3. Chunked prefill at 128K

The prefill cost calculation (§1.2) showed 5.8 seconds for a 128K prefill on 4× H100. Chunked prefill makes this practical.

### 3.1 The idea

Split the 128K prefill into chunks (e.g., 4K each). Process one chunk per "step" of continuous batching. Decode requests share these steps:

```text
step 1: prefill chunk 1 (4K tokens) + decode steps for other requests
step 2: prefill chunk 2 (4K tokens) + decode steps
...
step 32: prefill chunk 32 (4K tokens) → first token of new request ready
```

Total prefill wall-clock is unchanged (~5.8 seconds). But:

* **Other requests' decode is not blocked** during the long prefill.
* **TTFT for the long-prompt request increases**, but throughput overall stays high.
* **GPU utilization stays smooth** because each step has a mix of compute (chunked prefill) and bandwidth (decodes).

### 3.2 vLLM configuration

```python
LLM(
    ...,
    enable_chunked_prefill=True,
    max_num_batched_tokens=8192,    # total tokens per step (prefill + decode)
)
```

`max_num_batched_tokens=8192` means each step processes 8K tokens of work, mixing prefill chunks with decode rows.

### 3.3 The tradeoff

* **Without chunked prefill:** long prefill blocks the whole replica for ~5 seconds. Other users see a TTFT spike.
* **With chunked prefill:** long prefill is spread; other users' TTFT stays normal, but the long-prompt user's TTFT extends to ~6.5 seconds (longer wall-clock due to per-step overhead).

For chat / agent products, **chunked prefill is the default** — better p99 across users matters more than fastest individual TTFT.

For batch products, disabling chunked prefill can be slightly faster because the per-step overhead is removed and you can spend all of each step on one task.

---

## 4. FP8 KV cache at scale

At 128K context, FP8 KV (cuts HBM from 42 GB → 21 GB per request) is the practical default.

### 4.1 Per-head scaling discipline

Not all attention heads have the same KV magnitude distribution:

* Some heads specialize in rare-token positions (e.g., BOS, code-delimiters).
* Their KV values have outliers that FP8 *per-tensor* scaling clips, producing measurable parity loss on long-context retrieval.

**Per-head scaling fixes this.** Each head gets its own FP8 scale factor.

vLLM 0.22+: `kv_cache_dtype="fp8_e5m2"` with per-head scaling enabled by default.
SGLang 0.5+: per-head FP8 KV via `--kv-cache-dtype fp8`.
TRT-LLM: per-head FP8 KV is the default when `kv_cache_dtype=fp8`.

### 4.2 Parity at 128K

For Llama 3.3 70B with FP8 KV at 64K and 128K:

| Eval | FP16 KV | FP8 KV per-tensor | FP8 KV per-head |
|------|---------|-------------------|------------------|
| RULER 64K | 89.5 | 86.2 (-3.3) | 89.0 (-0.5) |
| RULER 128K | 80.1 | 75.3 (-4.8) | 79.4 (-0.7) |

Per-tensor FP8 KV loses 3-5 pp at long context. Per-head loses 0.5-0.7 pp. Per-head is the production recipe.

For Qwen 2.5 72B numbers are similar; both models behave well under per-head FP8 KV.

### 4.3 INT4 KV at 128K — usually too aggressive

INT4 KV at 128K typically loses 5-10 pp on RULER. Acceptable for products where the model doesn't need to recall details from the long context (e.g., short-output summarization). Unacceptable for retrieval-style tasks.

**Decision rule:** if the workload's value depends on accurate long-context retrieval, ship FP8 KV. INT4 KV is only for highly tolerant workloads or extreme HBM pressure.

---

## 5. Prefix sharing on long system prompts

The highest-leverage long-context optimization. Most long-context products have one of:

* **Long system prompt** (entire codebase, document, manual) shared across all user turns.
* **Stored conversation** with growing chat history.
* **RAG context** retrieved per query, partially overlapping across queries.

For all three, **prefix caching cuts prefill cost by 70-95%.**

### 5.1 The system-prompt case

A 100K-token system prompt + 1K user turn:

* Without prefix cache: prefill 101K tokens (~5 seconds on 4× H100).
* With prefix cache (after first request): prefill 1K tokens (~50 ms).

100× faster prefill for any subsequent request that shares the system prompt.

### 5.2 The conversation case

A growing chat session, 50K tokens of history + 500-token new user turn:

* Without prefix cache: prefill 50.5K each turn.
* With prefix cache: prefill 500 each turn (system prompt + previous turns are cached).

10-100× faster per turn after the first.

### 5.3 The RAG case

RAG retrieves N docs per query. If docs are reused across queries, the cache catches partial overlap. SGLang's RadixAttention is especially strong here because the prefix tree matches partial document overlap, not just exact-match prefixes.

### 5.4 Cache eviction

Prefix cache lives in HBM; eviction policy matters at long context. vLLM uses LRU; SGLang uses LRU with tree-structure awareness (keeps shared prefixes longer).

A common production tuning: set `kv_cache_block_size` slightly larger (e.g., 32 instead of 16) to give the radix tree better hit rates on long prefixes. Trade: slightly larger eviction granularity.

---

## 6. Long-context evaluation — what to measure

The parity story for long-context inference:

### 6.1 RULER ([arXiv:2404.06654](https://arxiv.org/abs/2404.06654))

A suite of synthetic tasks at controlled lengths (4K, 8K, 16K, 32K, 64K, 128K). Tasks:

* Single needle in haystack (find one fact among irrelevant text).
* Multi-needle (find N facts).
* Variable tracking (track values across long text).
* Frequency word extraction.

RULER is the **canonical long-context benchmark** as of mid-2026.

### 6.2 Needle in a haystack (NIAH)

Simpler than RULER. Place a needle (e.g., "the secret password is purple-frog-42") at a known position in N tokens of irrelevant context. Query the model to retrieve it.

* Plot accuracy as a function of context length × needle position.
* Common to see "lost in the middle" — accuracy higher at start/end, dips in the middle.

### 6.3 What to actually measure for parity validation

For a long-context product:

1. **Pick RULER at 16K, 64K, 128K** (or your product's longest expected context).
2. **Run the FP16-reference recipe twice for noise floor.**
3. **Run each candidate recipe (FP8 KV, INT4 KV, chunked prefill on/off).**
4. **Compute Δ per length.**

The pattern: short-context evals like MMLU don't catch long-context regressions. **Always validate long-context-specifically.**

---

## 7. Production recipes at 128K

Synthesized for Llama 3.3 70B / Qwen 2.5 72B at 128K context on Hopper:

### 7.1 Single H200 141G (low concurrency)

```text
Weights:       AWQ-INT4 group-128 (35 GB)
Activations:   FP16
KV cache:      FP8 per-head (21 GB per request at 128K)
HBM headroom:  ~85 GB for KV + activations → batch ~3 at 128K, batch ~12 at 32K

Runtime:       vLLM 0.22+ V1
Features:      continuous batching, paged KV v2, prefix cache, chunked prefill (4K chunks)
Speculation:   off (small batch makes it less effective)
```

### 7.2 4× H100 SXM (production chat)

```text
Weights:       FP8 (TRT-LLM) or BF16 (vLLM)
Activations:   FP8 / FP16
KV cache:      FP8 per-head
TP:            4
Sequence par.: enabled (long-context activation savings)

Features:      continuous batching, paged KV v2, prefix cache, chunked prefill (8K chunks)
Speculation:   EAGLE-2 / EAGLE-3 if validated for the workload

Expected:      ~600-1000 tok/s/GPU at chat-typical concurrency 32 with 16K mean context
```

### 7.3 8× H200 (max throughput / long context)

```text
Weights:       FP16 or FP8
KV cache:      FP8 per-head
TP:            8 (or 4 with DP=2 for throughput)
Sequence par.: enabled

Features:      full stack
Expected:      max concurrent users at 128K context; per-GPU throughput lower than TP=4 but absolute throughput highest
```

---

## Lab — long-context bench for both models

Goal: produce a long-context benchmark report for Llama 3.3 70B and Qwen 2.5 72B.

1. **Hardware** — 4× H100 or H200; ideally both for comparison.
2. **Runtime** — vLLM 0.22+ V1.
3. **Configurations to bench** — for each model:
   * FP8 KV, no chunked prefill.
   * FP8 KV, chunked prefill.
   * INT4 KV (if you want to validate where it breaks).
4. **Workloads:**
   * RULER at 16K, 64K, 128K.
   * TTFT measurement on 100K-token prefill, then 1K user turn, with and without prefix cache.
5. **Throughput** — chat-shape continuous workload at 32K mean context, concurrency 16.

Pass criterion: you can hand the report to another engineer and they can pick a precision recipe + chunked-prefill config for a 128K-context product, defended by your parity numbers.

---

## Self-check

1. At 128K context, FP16 KV is 42 GB per request. On 4× H100 (320 GB HBM), how many concurrent requests can you serve at 128K with INT4 weights (35 GB) and FP16 KV? With FP8 KV?
2. RULER 64K parity loses 3.3 pp with per-tensor FP8 KV and 0.5 pp with per-head FP8 KV. Why does the per-head version recover so much?
3. Your TTFT at 128K is 5 seconds without prefix cache. After enabling, it drops to 80 ms. What property of the workload allowed the dramatic improvement?
4. Chunked prefill at 4K chunks vs 16K chunks: which has better p99 TTFT for a workload of 90% short prompts and 10% 128K prompts? Why?
5. For Llama 3.3 70B with FP8 KV per-head at 128K, predict the decode TPOT on H200 (4.8 TB/s). Show the math.

---

## References

* YaRN — [arXiv:2309.00071](https://arxiv.org/abs/2309.00071)
* RULER — [arXiv:2404.06654](https://arxiv.org/abs/2404.06654)
* Needle in a haystack (LangChain implementation) — [github.com/gkamradt/LLMTest_NeedleInAHaystack](https://github.com/gkamradt/LLMTest_NeedleInAHaystack)
* StreamingLLM (long-context window attention) — [arXiv:2309.17453](https://arxiv.org/abs/2309.17453)
* H2O (KV cache eviction) — [arXiv:2306.14048](https://arxiv.org/abs/2306.14048)
* FastV (sparse attention) — [arXiv:2403.06764](https://arxiv.org/abs/2403.06764)
* vLLM long-context tuning — [docs.vllm.ai/en/latest/serving/openai_compatible_server.html](https://docs.vllm.ai/en/latest/serving/openai_compatible_server.html)
* SGLang RadixAttention paper — [arXiv:2312.07104](https://arxiv.org/abs/2312.07104)

Cross-references:

* [Part 1 → Lecture 02 — KV cache math](../Part%201%20-%20Fundamentals/Lecture-02.md)
* [Phase 5 → GPU Infrastructure → Long-Context-MoE-Foundation-Training → 07 Long-Context Evaluation](../../../1.%20GPU%20Infrastructure/Nvidia%20GPU/HPC%20Setup/Long-Context-MoE-Foundation-Training/07-Long-Context-Evaluation.md)

---

## Current as of 2026-06

YaRN extension as published by both model teams. FP8 KV per-head as the production recipe. Chunked prefill as the default in vLLM 0.22+. Refresh when a fundamentally new long-context attention pattern (e.g., recurrent state, hybrid SSM) ships in either model family.

---

## End of Part 2

You have completed the dense-at-Hopper portion of this course. The harness, precision recipes, TP-scaling discipline, and serving knobs from Part 2 all carry forward — Part 3 reuses them and extends the harness to sparse MoE on Blackwell.

* Next: [Part 3 — MoE at Blackwell](../Part%203%20-%20MoE%20at%20Blackwell/README.md) — DeepSeek V3.1 + Qwen3-MoE 235B-A22B anchors, EP, FP4, disaggregated P/D
* Previous: [Lecture 05 — Modern serving stack](Lecture-05.md)
* Up: [Part 2 — Dense at Hopper](README.md)
