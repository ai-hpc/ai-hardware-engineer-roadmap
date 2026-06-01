# Part 2 · Lecture 05 — The Modern Serving Stack: Continuous Batching, Paged KV, Prefix Cache, Speculation

## Overview

A 70B dense model on Hopper, partitioned by TP and quantized by AWQ-INT4 or FP8, will give you the *single-replica peak*. To get to a production-grade serving system you need the four modern serving features that turned LLM inference from a research demo into an industry:

1. **Continuous batching** — dynamically mix prefill and decode in one batch, recomposed every step.
2. **PagedAttention v2** — paged, block-level KV cache management that eliminates HBM fragmentation.
3. **Prefix caching / RadixAttention** — share KV entries across requests with overlapping prefixes.
4. **Speculative decoding** — generate multiple tokens per forward pass via a draft model.

Each one independently moved a major metric by 2–10× when it shipped (2023–2025). Together they define the 2026 serving baseline. A deployment that doesn't use them is shipping at 10–50× the cost it could.

This lecture applies each to Llama 3.3 70B and Qwen 2.5 72B on 4–8× H100/H200, with concrete numbers and the runtime configs.

By the end you should be able to enable each feature for either anchor model, predict the metric movement, and verify with measurement.

---

## 1. Continuous batching

The breakthrough of 2023. The mental model:

### 1.1 The problem static batching solves badly

Before continuous batching, a request was dispatched as a self-contained job:

```text
batch of 8 requests arrives → forward pass → all 8 finish (or pad) → next batch starts
```

Problems:

* If request 1 generates 50 tokens and request 8 generates 500 tokens, the batch waits for request 8 — request 1 sits at the bottom of the batch idle.
* Throughput is bounded by the longest-running request.
* GPU utilization is poor — frequent rebuilds.

### 1.2 Continuous batching reframes the problem

Each request is decomposed into **steps**. At each step, the scheduler picks which sequences participate in the batch:

```text
step t:    batch = [req1.t=8, req2.t=12, req3.t=0(prefill), req4.t=5, ...]
step t+1:  req1 finishes → batch = [req2.t=13, req3.t=1, req4.t=6, req5.t=0(prefill), ...]
step t+2:  req5 prefill finishes → batch = [req2.t=14, req3.t=2, req4.t=7, req5.t=1, ...]
```

* New requests join the batch as soon as a slot opens.
* Finished requests leave immediately.
* Throughput approaches the single-batch peak continuously.

### 1.3 Mixing prefill and decode

The subtle hard part: prefill is compute-bound, decode is memory-bound. Mixing them in the same step can stall the batch.

vLLM 0.22+ V1 uses **chunked prefill** — prefill is split into chunks the size of a decode batch row. Prefill becomes "just another decode-shaped step." This keeps the batch balanced and the GPU at high utilization.

### 1.4 Throughput impact

For Llama 3.3 70B on 4× H100 with mixed chat traffic (mean prompt 1024, mean output 256, concurrency 32):

| Scheduling | Throughput tok/s/GPU | Notes |
|------------|----------------------|-------|
| Static batch | ~140 | dominated by stragglers |
| Continuous batching | ~300 | 2.1× improvement |
| Continuous + chunked prefill | ~360 | another 20% from smooth batch composition |

Continuous batching is on by default in vLLM, SGLang, and TRT-LLM. There is no good reason to disable it.

---

## 2. PagedAttention v2 — block-level KV memory

The second breakthrough. The mental model:

### 2.1 The KV memory fragmentation problem

Before paging, each request reserved KV memory contiguously. If you allocated 4096 tokens of KV and the request finished at token 312, the rest (3784 tokens × 320 KB ≈ 1.2 GB) was wasted until the request released.

Across many requests, HBM looks Swiss-cheese: full of holes. Effective capacity might be 50% of physical.

### 2.2 PagedAttention v1

Each request's KV is split into fixed-size **blocks** (e.g., 16 tokens per block). Blocks are allocated from a pool as the sequence grows.

* No fragmentation — every block is fully used or available.
* Effective HBM capacity for KV: 95–98% of physical (down from 50–70%).
* Cost: ~3% kernel overhead from indirect block addressing.

### 2.3 PagedAttention v2

Added in vLLM 0.6+. Improvements:

* Smaller blocks (4 or 8 tokens) for finer-grained allocation.
* Better cache utilization in the attention kernel.
* Lower overhead.

For Llama 3.3 70B / Qwen 2.5 72B at 128K context with batch=8:

| Memory mode | Effective KV capacity | Max concurrent requests at 128K |
|-------------|----------------------|----------------------------------|
| Contiguous | ~50% | ~3 |
| PagedAttention v1 | ~95% | ~6 |
| PagedAttention v2 | ~97% | ~6 (smoother allocation) |

**Without paging, long-context serving on Hopper is not practical.** vLLM, SGLang, TRT-LLM, and llama.cpp all implement paged KV.

### 2.4 The shared kernel layer — FlashInfer

A 2025 shift worth knowing: the paged-attention (and sampling) kernels are increasingly *not* each runtime's bespoke CUDA. **FlashInfer** ([arXiv:2501.01005](https://arxiv.org/abs/2501.01005), MLSys 2025; Apache-2.0, `flashinfer-ai`) is an open-source, JIT-compiled attention + sampling **kernel engine** that vLLM, SGLang, TensorRT-LLM, and MLC-LLM all build on. It provides:

* **Paged / ragged attention kernels** for prefill *and* decode over block-sparse (page-table) KV layouts — i.e., the kernels that make PagedAttention fast, including the indirect block addressing referenced above.
* **Sorting-free top-k / top-p sampling kernels** (the sampling step, not just attention).
* **Customizable, JIT-compiled attention variants** (different masks, RoPE shapes, head geometries, FP8 KV) without hand-writing CUDA per case.

Why this matters to you: a kernel-level win (a new attention variant, a Blackwell-tuned path) lands across *multiple* runtimes at once, and "which attention backend" becomes a config/version knob to pin and profile — not a black box. When you set `VLLM_ATTENTION_BACKEND=FLASHINFER` or SGLang's `--attention-backend flashinfer`, this is the engine you are selecting.

---

## 3. Prefix caching — RadixAttention and friends

The 2024 breakthrough. Many requests share prefixes — system prompts, RAG retrievals, conversation history. Recomputing KV for these prefixes is pure waste.

### 3.1 Simple prefix cache (vLLM)

A request prefix is hashed. If the hash matches an existing KV block sequence in the cache, the cached KV is reused. The request only needs to compute KV for its *new* tokens.

* For a chat product with 1000-token system prompts and 100-token user turns: prefix cache hit saves 90% of prefill cost.
* For a RAG product where each query embeds 5 retrieved docs: variable hit rate depending on retrieval overlap.

### 3.2 RadixAttention (SGLang)

A more general structure: a **radix tree** indexed by prefix tokens. Any common prefix between any two requests hits the cache, not just the system prompt.

```text
        ┌── "What is..."  (5 requests share this 7-token prefix)
        │
"You are a helpful   ┤
 assistant. Respond  │
 in English. "       └── "Translate..." (3 requests)

Both branches share the system prompt KV — only the branch-specific tokens
needed fresh prefill.
```

* **Wins for RAG, agent, multi-turn chat** where prefixes branch into many requests.
* **Win factor:** 2-10× prefill throughput at high prefix-overlap workloads.

### 3.3 Measurable impact

For Qwen 2.5 72B on a customer-support chat product (long system prompt + tool catalog):

| Prefix cache | TTFT | Prefill cost |
|--------------|------|--------------|
| Off | 480 ms | full |
| vLLM hash prefix cache | 90 ms | 19% of full |
| SGLang RadixAttention | 85 ms | 18% of full |

For a one-paragraph system prompt the impact is smaller (~30-50% TTFT improvement). For long system prompts + tool catalogs (typical agent / chat) the impact is dramatic.

### 3.4 Configuration

vLLM:

```python
LLM(model="...", enable_prefix_caching=True)
```

SGLang: on by default via RadixAttention.

TRT-LLM 1.3+: supported via `--use_paged_context_fmha`, `kv_cache_reuse=True` at runtime.

---

## 4. Speculative decoding

The 2024–2025 breakthrough on decode throughput. The mental model:

### 4.1 The idea

A small **draft model** proposes the next k tokens. The full **target model** verifies all k in one forward pass:

```text
draft model: emits 5 candidate tokens (cheap, fast)
target model: prefills "previous tokens + 5 candidates" and gets logits for each position
              accept tokens 1..n where draft prediction matches target's argmax
              reject from token n+1
```

* Best case: draft predicts all 5 correctly → 5 tokens generated in 1 target forward pass.
* Worst case: draft mispredicts token 1 → 1 token (no win).
* Average for well-paired draft+target: **acceptance rate 60-80%**, giving 2-4× effective throughput.

### 4.2 Draft model choice

For Llama 3.3 70B target:

* **Llama 3.2 1B Instruct** as draft — 70× smaller, ~0.5 ms/token draft latency, acceptance ~65%.
* **Llama 3.2 3B Instruct** as draft — 23× smaller, ~1.5 ms/token, acceptance ~75%.

For Qwen 2.5 72B target:

* **Qwen 2.5 1.5B Instruct** as draft — ~75% acceptance.
* **Qwen 2.5 3B Instruct** as draft — ~80% acceptance.

The pattern: draft from the **same family** as target. Cross-family drafting (Llama draft, Qwen target) loses acceptance.

### 4.3 EAGLE / EAGLE-2 / EAGLE-3

[EAGLE](https://arxiv.org/abs/2401.15077) and successors use a *learned* speculation head trained on the target model's distribution. EAGLE-3 ([arXiv:2503.01840](https://arxiv.org/abs/2503.01840)) is the 2025 state-of-the-art:

* Acceptance rates 75-85% across categories.
* No separate draft model — the head is fused with the target.
* 2.5-3.5× effective decode throughput for chat workloads.

vLLM and SGLang both support EAGLE-2 / EAGLE-3 as of mid-2026.

### 4.4 When speculation hurts

* **Very low batch** (1-2 requests): speculation adds latency without saving much because verification is short.
* **Very short decodes** (<10 tokens output): the draft cost amortizes poorly.
* **High-acceptance regimes** are exceptional; for low-acceptance workloads (very creative tasks, non-English code), speculation can be 0.7× of non-speculative.

**Validate per workload.** A 2.5× speedup on chat may not transfer to a JSON-emitting agent.

### 4.5 Throughput impact

For Llama 3.3 70B on 4× H100 FP8 chat workload (concurrency 32, prompt 1024, output 256):

| Decoding | Throughput tok/s/GPU |
|----------|----------------------|
| Greedy autoregressive | ~580 |
| Llama 3.2 1B draft + Llama 3.3 70B target | ~870 (1.5×) |
| EAGLE-3 head | ~1450 (2.5×) |

Speculation is the biggest "free" decode optimization on Hopper for chat workloads.

---

## 5. The four-feature stack

Combined for Llama 3.3 70B FP8 on 4× H100, chat workload:

```text
baseline (vLLM 0.22, no advanced features):       ~250 tok/s/GPU
+ continuous batching:                           ~580 tok/s/GPU  (already on by default)
+ PagedAttention v2:                             ~610 tok/s/GPU  (3-5% from less fragmentation)
+ prefix cache:                                  ~700 tok/s/GPU  (15-20% for long system prompts)
+ EAGLE-3 speculation:                           ~1450 tok/s/GPU (2.07× from speculation)
```

The composed stack is ~6× the no-feature baseline. **This is the inference-engineering edge over a naive deployment.**

---

## 6. Configuration cheat sheet — vLLM

```python
from vllm import LLM, SamplingParams

llm = LLM(
    model="meta-llama/Llama-3.3-70B-Instruct",
    tensor_parallel_size=4,
    dtype="float16",
    
    # continuous batching: on by default in V1, no flag needed
    
    # PagedAttention v2: on by default
    block_size=16,                     # default; smaller = finer paging
    
    # prefix cache: enable explicitly
    enable_prefix_caching=True,
    
    # speculation: pick one
    speculative_model="meta-llama/Llama-3.2-1B-Instruct",
    num_speculative_tokens=5,
    # OR EAGLE-3 once landed:
    # speculative_config={"method": "eagle3", ...}
    
    # chunked prefill (recommended for long prompts):
    enable_chunked_prefill=True,
    max_num_batched_tokens=8192,
    
    # KV quantization (optional):
    kv_cache_dtype="fp8_e5m2",
    
    # serving config:
    gpu_memory_utilization=0.92,
    max_num_seqs=128,
)
```

For Qwen 2.5 72B replace the model path and use `Qwen/Qwen2.5-1.5B-Instruct` as the draft.

---

## 7. Order to enable

A pragmatic order:

1. **Continuous batching + PagedAttention v2** — already on by default; verify with a profile.
2. **Prefix caching** — biggest single TTFT win; enable, measure.
3. **Chunked prefill** — enable if your prompts are >2K tokens; measure TTFT impact.
4. **FP8 KV cache** — if at long context or large batch; measure parity.
5. **Speculation** — last because acceptance-rate measurement requires the rest of the stack to be stable. Measure acceptance with a draft model first; switch to EAGLE-3 once the workload is profiled.

---

## Lab — enable each feature, measure each metric movement

Goal: isolate each feature's contribution to throughput and TTFT on a fixed workload.

1. **Baseline** — Llama 3.3 70B FP8 on 4× H100, vLLM 0.22, default config except features-off.
2. **Add continuous batching + PagedAttention** — measure throughput delta.
3. **Add prefix cache** — measure TTFT delta on a workload with 1500-token system prompt + 100-token user turns.
4. **Add chunked prefill** — measure TTFT delta on a workload with 8K-token prompts.
5. **Add FP8 KV cache** — measure HBM saved, validate parity on RULER at 32K.
6. **Add a 1B-class draft model** — measure throughput delta, measure acceptance rate.
7. **Plot** the cumulative throughput across all features.

Pass criterion: you have a chart that shows each feature's individual contribution and the cumulative result. Each contribution is defended by a measurement.

---

## Self-check

1. A teammate measures prefix caching and reports a 2× throughput improvement. You suspect the test is biased. What workload property would inflate prefix-cache gains?
2. EAGLE-3 acceptance is 70% on chat. On a tool-call agent workload it's 45%. Why might agent acceptance be lower, and what does that suggest about whether to ship EAGLE-3 for the agent product?
3. PagedAttention v2 vs v1: under what workload would you measurably notice the v2 win?
4. Continuous batching mixes prefill and decode. For a workload with mean prompt 2048 and mean output 16 (agent shape), would you enable chunked prefill? Why?
5. Your draft model is Llama 3.2 1B and target is Qwen 2.5 72B. Acceptance is 35%. What is the first experiment?

---

## References

* vLLM PagedAttention paper — [arXiv:2309.06180](https://arxiv.org/abs/2309.06180)
* SGLang RadixAttention paper — [arXiv:2312.07104](https://arxiv.org/abs/2312.07104)
* FlashInfer (attention/sampling kernel engine) — [arXiv:2501.01005](https://arxiv.org/abs/2501.01005) (MLSys 2025) · [github.com/flashinfer-ai/flashinfer](https://github.com/flashinfer-ai/flashinfer)
* Speculative Decoding original — [arXiv:2211.17192](https://arxiv.org/abs/2211.17192)
* EAGLE — [arXiv:2401.15077](https://arxiv.org/abs/2401.15077)
* EAGLE-2 — [arXiv:2406.16858](https://arxiv.org/abs/2406.16858)
* EAGLE-3 — [arXiv:2503.01840](https://arxiv.org/abs/2503.01840)
* Medusa heads — [arXiv:2401.10774](https://arxiv.org/abs/2401.10774)
* DistServe (P/D disaggregation, contrast) — [arXiv:2401.09670](https://arxiv.org/abs/2401.09670)
* "Orca: A Distributed Serving System for Transformer-Based Generative Models" — OSDI 2022 — pre-vLLM origin of continuous batching ideas

Cross-references:

* [Part 1 → Lecture 05 — Runtime landscape](../Part%201%20-%20Fundamentals/Lecture-05.md)
* [Phase 5 → Edge AI → Agent Tool-Dispatch Evaluation with BFCL](../../../3.%20Edge%20AI/Agent%20Tool-Dispatch%20Evaluation%20with%20BFCL/Lecture-01.md) — parity gating for speculation

---

## Current as of 2026-06

Features pinned: vLLM 0.22 V1 (continuous batching, PagedAttention v2, prefix cache, EAGLE-2/3 supported), SGLang 0.5 (RadixAttention), TRT-LLM 1.3 (in-flight batching, KV reuse). Refresh when EAGLE-4 or successor lands, or when a fundamentally new scheduling method (post-continuous-batching) ships.

---

## Next

* Next: [Lecture 06 — Long context at 128K on Hopper](Lecture-06.md)
* Previous: [Lecture 04 — Tensor parallelism on 8× H100/H200](Lecture-04.md)
* Up: [Part 2 — Dense at Hopper](README.md)
