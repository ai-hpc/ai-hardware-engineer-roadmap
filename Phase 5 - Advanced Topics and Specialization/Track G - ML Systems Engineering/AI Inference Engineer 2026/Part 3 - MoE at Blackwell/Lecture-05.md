# Part 3 · Lecture 05 — Production MoE Serving: MTP Speculation, Constrained Decode, Cost Model

## Overview

This is the **capstone of the course**. The previous Part 3 lectures established the architecture (Lecture 01), the silicon (Lecture 02), the parallelism (Lecture 03), and the serving topology (Lecture 04). This lecture brings them together into a **defended, production-shippable inference deployment** for DeepSeek V3.1 and Qwen3-MoE 235B-A22B on Blackwell, with a `$/MTok` cost model that another engineer can reproduce.

Topics:

1. **MTP as native speculation** for DeepSeek — configuration, acceptance rate, throughput.
2. **EAGLE-3 for Qwen3-MoE** — the non-native speculation path with draft model.
3. **Constrained decoding** — XGrammar / Outlines at MoE scale, tool-call use cases.
4. **The production stack** — full configuration for each model on NVL72.
5. **Cost model** — `$/MTok` derivation, what each precision / EP / disaggregation choice buys.
6. **Capstone benchmark** — the report your final repo must contain.

By the end you should be able to deploy either MoE model to production-shape traffic on a Blackwell cluster, defend every recipe choice with parity and throughput numbers, and produce a `$/MTok` cost defense for the business.

---

## 1. MTP — native speculation for DeepSeek V3.1

DeepSeek V3 introduced multi-token prediction (MTP) as a training-time objective. At inference time, the model emits **k+1 logit predictions per forward pass**, not just one.

### 1.1 The inference flow

```text
Standard autoregressive decode:
  step 1: emit token N+1
  step 2: emit token N+2
  step 3: emit token N+3
  → 3 forward passes for 3 tokens

MTP decode (k=3):
  step 1: emit token N+1, predict N+2 and N+3
  Verification: accept N+1 (always, it's the standard output)
                accept N+2 if its prediction matches what step 2 would have produced
                accept N+3 if both N+2 accepted AND N+3 matches
  → 1 forward pass for up to 3 tokens
```

The model's MTP heads produce the predictions. The **verification logic** (does the predicted token match the standard output) lives in the inference runtime. The **acceptance rate determines the effective speedup**.

### 1.2 Acceptance rates

Real-world for DeepSeek V3.1:

* Chat: 65-75% on token N+2; 50-65% on N+3.
* Code generation: 55-65% on N+2 (harder to predict).
* JSON / tool-call: 80-90% (highly predictable structure).

Average effective tokens per pass: ~1.8-2.2 across diverse traffic.

### 1.3 Configuration

vLLM 0.22+:

```python
LLM(
    model="deepseek-ai/DeepSeek-V3.1",
    speculative_config={
        "method": "deepseek_mtp",
        "num_speculative_tokens": 3,
    },
    ...
)
```

SGLang:

```bash
python -m sglang.launch_server \
    --model-path deepseek-ai/DeepSeek-V3.1 \
    --speculative-algorithm DeepseekMTP \
    --speculative-num-steps 3 \
    ...
```

### 1.4 Throughput impact

For DeepSeek V3.1 at FP4 on 16× B200 NVL72 partition, chat workload:

| Decoding | Throughput tok/s/replica |
|----------|--------------------------|
| Greedy autoregressive | ~6,500 |
| MTP k=2 | ~10,000 (1.5×) |
| MTP k=3 | ~13,000 (2.0×) |

MTP is the **largest single throughput win** on DeepSeek's deployment stack.

---

## 2. EAGLE-3 for Qwen3-MoE

Qwen3-MoE 235B-A22B has no native MTP. The 2026 production speculation path is **EAGLE-3** ([arXiv:2503.01840](https://arxiv.org/abs/2503.01840)) with a separately-trained or distilled draft model.

### 2.1 EAGLE-3 mechanics

Unlike a separate draft LLM, EAGLE-3 trains a **lightweight speculation head** that sits on top of the target model. The head **reuses the target's hidden states** and predicts next-k tokens.

* No separate draft model HBM cost.
* Acceptance rates 75-85% across categories.
* 2.5-3.5× effective decode throughput.

For Qwen3-MoE 235B-A22B, an EAGLE-3 head trained specifically for this model is published by the SGLang community.

### 2.2 Configuration

SGLang:

```bash
python -m sglang.launch_server \
    --model-path Qwen/Qwen3-235B-A22B \
    --speculative-algorithm EAGLE3 \
    --speculative-draft-model-path Qwen/Qwen3-235B-A22B-EAGLE3 \
    --speculative-num-steps 3 \
    --speculative-eagle-topk 1 \
    ...
```

### 2.3 Throughput impact

For Qwen3-MoE 235B-A22B at FP4 on 8× B200, chat workload:

| Decoding | Throughput tok/s/replica |
|----------|--------------------------|
| Greedy autoregressive | ~4,500 |
| EAGLE-3 k=3 | ~10,500 (2.3×) |

Comparable to DeepSeek + MTP at a similar effective speedup ratio.

---

## 3. Constrained decoding at MoE scale

Structured output (tool calls, JSON, code) requires the model to emit **specifically-shaped sequences**. **Constrained decoding** uses a grammar (or regex / FSM) to restrict the sampler to **legal tokens only**.

### 3.1 XGrammar vs Outlines

| Library | Approach | Where it lives |
|---------|----------|----------------|
| XGrammar | compile grammar to FSM, apply at logit level | SGLang (native), vLLM (integration) |
| Outlines | regex / Pydantic, slower for complex grammars | vLLM, llama.cpp |

**XGrammar is the fastest production option** in mid-2026. SGLang ships it natively.

### 3.2 BFCL parity at MoE FP4

The [BFCL evaluation lecture](../../../Track%20C%20-%20Edge%20AI/Agent%20Tool-Dispatch%20Evaluation%20with%20BFCL/Lecture-01.md) framework applies to MoE inference exactly as to dense. At MoE FP4, tool-call accuracy is the parity gate that decides if FP4 ships.

For DeepSeek V3.1:

| Precision | BFCL (simple) | BFCL (multi-turn) | Δ vs BF16 |
|-----------|---------------|-------------------|-----------|
| BF16 reference | 88.5 | 79.2 | — |
| FP8 | 88.1 | 78.7 | -0.4 / -0.5 |
| FP4 | 87.0 | 77.4 | -1.5 / -1.8 |
| FP4 + grammar-constrained decoding | 88.2 | 78.9 | -0.3 / -0.3 |

Grammar-constrained decoding **recovers most of the FP4 parity loss on structured-output workloads**. This is the recipe for agent products at FP4.

### 3.3 Configuration

SGLang:

```python
from sglang import RuntimeEndpoint, ChatTemplate

# Define schema
schema = {
    "type": "object",
    "properties": {
        "tool": {"type": "string", "enum": ["home_control", "search", "memory"]},
        "args": {"type": "object"},
    },
    "required": ["tool", "args"],
}

# Request with structured output
response = client.chat.completions.create(
    model=endpoint,
    messages=...,
    response_format={"type": "json_schema", "json_schema": {"schema": schema}},
)
```

XGrammar applies at the logit-sampling level. Throughput cost is minimal (~5% slower than unconstrained decoding) for well-formed grammars.

---

## 4. The production stack

Synthesizing everything from Part 3:

### 4.1 DeepSeek V3.1 production recipe

```text
Hardware:           NVL72 partition, 16× B200 (TP=2 × EP=8)
Weights:            FP4 (MX-FP4 via TE2)
Activations:        FP8
KV cache:           BF16 (MLA-compressed; small, no precision drop needed)
Attention:          MLA-aware kernels (SGLang or vLLM 0.22+)
Speculation:        MTP k=3
Serving:            SGLang 0.5+ V1
Features:           continuous batching, paged KV, RadixAttention prefix cache,
                    chunked prefill (8K chunks), XGrammar (if agent workload)
Disaggregation:     enabled if cluster has > 16 GPUs available; SGLang P/D mode

Expected throughput: ~13,000 tok/s/replica (with MTP)
Expected $/MTok:    ~$1.9-2.0 raw replica cost (16× B200 @ ~$5.50/GPU-hr — derived in §5.2)
```

### 4.2 Qwen3-MoE 235B-A22B production recipe

```text
Hardware:           8× B200 (EP=8) for single-replica; 16-32× for multi-replica
Weights:            FP4 (MX-FP4)
Activations:        FP8
KV cache:           FP8 per-head (GQA, 4 KV heads, 94 layers — meaningful at long context)
Speculation:        EAGLE-3 k=3 (with the model-specific draft head)
Serving:            SGLang 0.5+ V1
Features:           continuous batching, paged KV, prefix cache, chunked prefill,
                    XGrammar
Disaggregation:     usually not worth it at this scale; ship colocated

Expected throughput: ~10,500 tok/s/replica
Expected $/MTok:    ~$1.1-1.2 raw replica cost (8× B200 @ ~$5.50/GPU-hr)
```

---

## 5. The cost model

Deriving `$/MTok` is the **final exit-criterion deliverable**.

### 5.1 The formula

```text
$/MTok = (replica_cost_per_hour × 10^6) / (3600 × output_tokens_per_sec)
```

Inputs:

* **replica_cost_per_hour** — GPU cost × number of GPUs in a replica + amortized infrastructure (network, storage, scheduler).
* **output_tokens_per_sec** — measured throughput at the workload's typical concurrency.

### 5.2 GB200 NVL72 cost model

Approximate 2026 cloud rates:

| GPU class | $/hour (one GPU) |
|-----------|------------------|
| H100 SXM | ~$2.50 |
| H200 SXM | ~$3.50 |
| B200 SXM | ~$5.50 |
| B300 SXM | ~$7.00 |
| GB200 (per Blackwell GPU) | ~$5.50 (similar to B200 SXM on hyperscalers) |

A 16-GPU replica of DeepSeek V3.1:

* Hardware: 16 × $5.50 = $88/hour
* Network + scheduler: ~$2/hour
* Total: ~$90/hour

If throughput is 13,000 tok/s/replica (with MTP):

```text
$/MTok = ($90 × 10^6) / (3600 × 13,000)
       = $90,000,000 / 46,800,000
       ≈ $1.92 per million tokens
```

Now compare: the published rate for DeepSeek V3.1 API ($/MTok) is ~$0.30 input and ~$1.10 output (varies by provider) — **below** this raw single-replica estimate. That gap tells you real deployments run at much higher utilization and batching than this example assumes, on top of provider-scale economics (prefix caching, traffic mixing across replicas, committed-hardware pricing). The $1.92 is a teaching anchor, not the truth of an optimized fleet.

**For the inference engineer's defense:** show the raw $/MTok at full utilization. The product team multiplies by overhead.

### 5.3 What moves the cost model

| Lever | Effect on $/MTok | Range |
|-------|------------------|-------|
| FP4 vs FP8 | -30 to -40% | precision drop on weights |
| MTP / EAGLE | -40 to -55% | effective throughput multiplier |
| Disaggregation (NVL72 in-domain) | -20 to -35% | hardware optimization per phase |
| Concurrency 16 → 64 | -50 to -65% | batch amortization |
| Concurrency 64 → 256 | -10 to -20% | diminishing returns |
| EP=8 → EP=16 | -10 to -20% | less per-GPU pressure |

A naive deployment at FP8 + greedy decoding + concurrency 16 + EP=2 might cost $4-6/MTok (greedy alone drops the replica to ~6,500 tok/s, or ~$3.85/MTok; FP8 and the low concurrency push it further). The same model at FP4 + MTP + EP=8 + concurrency 64 hits $2 or less. **The optimization stack moves the cost model by 2-3×.**

---

## 6. The capstone benchmark — what your repo must contain

The **final deliverable** of this course. Your benchmark repo contains:

### 6.1 Reproducibility layer

* Exact hardware (GPU SKU, driver, CUDA, cuDNN, FA, TE versions).
* Exact runtime versions and commit hashes.
* Exact model hashes from Hugging Face.
* Calibration data manifests (if AWQ used).

### 6.2 Parity reports

For each model × precision recipe:

* Reference noise floor (FP16/BF16 across two seeds).
* Candidate Δ on the workload's eval set.
* Failure-mode analysis if parity exceeded budget.

### 6.3 Throughput tables

For each (model × runtime × precision × EP × concurrency) cell:

* TTFT p50/p95/p99.
* TPOT p50/p95/p99.
* Throughput tok/s/replica and tok/s/GPU.
* Effective $/MTok at the cost model from §5.

### 6.4 Profiles

At least one Nsight Systems profile for:

* TP all-reduce dominated regime (Part 2).
* EP all-to-all dominated regime (Part 3).
* MTP / EAGLE speculation in flight.

### 6.5 The narrative

A 3-page markdown summary explaining:

* The workload and SLO targets.
* The recipe choice for each (model, hardware) pair.
* The $/MTok defense.
* What you would change if the cluster scale doubled.

This narrative is what a senior engineer reviews before signing off on the deployment. **It is the capstone.**

---

## Lab — produce the capstone report

Goal: the final capstone benchmark report for either DeepSeek V3.1 or Qwen3-MoE 235B-A22B.

1. **Pick one model** and one workload class (chat or agent).
2. **Define the SLO** — TTFT, TPOT, throughput targets.
3. **Hardware** — what you have access to (8× B200, NVL72 partition, etc.).
4. **Build matrix** — at least four configurations covering different precision × EP × speculation choices.
5. **Bench each** with parity validation.
6. **Compute $/MTok** for each.
7. **Write the narrative** — recommend a ship recipe with defended numbers.

Pass criterion: another engineer can clone the repo, run `make bench`, and reproduce your numbers within ±10%.

---

## Self-check

1. MTP delivers 2× decode speedup for DeepSeek V3.1 with no extra HBM. EAGLE-3 delivers 2.3× speedup for Qwen3-MoE with a small EAGLE-head HBM cost. Why might you still pick EAGLE-3 for a code-gen workload even if MTP-DeepSeek is available?
2. Your XGrammar-constrained decoding adds 5% to TPOT. Your BFCL accuracy improves 4 pp at FP4. Does this trade ship for an agent product?
3. At NVL72 single-replica DeepSeek V3.1 EP=64 FP4 with MTP, predict your $/MTok if the per-GPU cost is $5.50/hour and throughput is 20,000 tok/s/replica.
4. A product team wants 0.5s TTFT and 30ms TPOT for a chat product. Which model + recipe do you ship: DeepSeek V3.1 NVL72 or Qwen3-MoE 235B-A22B 8× B200? Defend in two sentences.
5. The cost model says FP4 saves 35%, MTP saves 50%, EP=16 saves 15%. Why is the total savings not 100%?

---

## References

* DeepSeek V3 MTP — [arXiv:2412.19437](https://arxiv.org/abs/2412.19437)
* "Multi-token Prediction" — [arXiv:2404.19737](https://arxiv.org/abs/2404.19737)
* EAGLE-3 — [arXiv:2503.01840](https://arxiv.org/abs/2503.01840)
* XGrammar — [arXiv:2411.15100](https://arxiv.org/abs/2411.15100)
* Outlines — [github.com/dottxt-ai/outlines](https://github.com/dottxt-ai/outlines)
* BFCL — [gorilla.cs.berkeley.edu/leaderboard.html](https://gorilla.cs.berkeley.edu/leaderboard.html)
* SGLang DeepSeek serving guide — [sgl-project.github.io](https://sgl-project.github.io/)
* DeepSeek V3.1 official inference guide — [github.com/deepseek-ai/DeepSeek-V3](https://github.com/deepseek-ai/DeepSeek-V3)

Cross-references:

* [Phase 5 → Edge AI → Agent Tool-Dispatch Evaluation with BFCL](../../../Track%20C%20-%20Edge%20AI/Agent%20Tool-Dispatch%20Evaluation%20with%20BFCL/Lecture-01.md)
* [Part 2 → Lecture 05 — Modern serving stack](../Part%202%20-%20Dense%20at%20Hopper/Lecture-05.md)
* [Phase 5 → ML Systems Engineering Guide → Capstone Options](../../Guide.md#capstone-options)

---

## Current as of 2026-06

MTP and EAGLE-3 as canonical 2025–2026 speculation paths. XGrammar as the standard constrained-decode library. SGLang as the production serving runtime for MoE. NVL72 cost model from current cloud pricing. Refresh when new speculation methods land or when pricing shifts significantly.

---

## End of Part 3 — End of Course

Congratulations. You have completed AI Inference Engineer 2026.

The proof is the benchmark repo. If another engineer can run it and reproduce your numbers, you are the senior inference engineer the course was designed to produce.

* Back to: [AI Inference Engineer 2026 — Overview](../README.md)
* Up: [Phase 5 → Track G — ML Systems Engineering](../../Guide.md)
* Refresh Log: [REFRESH-LOG.md](../REFRESH-LOG.md)
