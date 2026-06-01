# Part 2 · Lecture 03 — Quantizing Llama 3.3 70B and Qwen 2.5 72B

## Overview

Part 1 Lecture 04 introduced the precision stack and the major quantization methods at the concept level. This lecture applies them to the two anchor models on Hopper hardware and produces concrete, defended recipes.

The Llama 3.3 70B ↔ Qwen 2.5 72B pair is uniquely useful here because:

* They share architecture (Lecture 01) so the same methods apply.
* They differ in one published anomaly (arXiv:2408.15301) — the Llama-3-70B activation-outlier sensitivity — so we get to walk a real "this method works here but not there" case.
* Both are widely-deployed, so primary benchmarks are abundant.

This lecture covers:

1. The precision recipes that ship for each model in 2026.
2. **AWQ on Llama 3.3 70B and Qwen 2.5 72B** — calibration data choice, group size, validation.
3. **GPTQ vs AWQ** — when each wins and why.
4. **The Llama-3-70B W8A8 anomaly** — what it is, how QuaRot/SpinQuant fix it.
5. **FP8 on Hopper** — TensorRT-LLM's path and how to validate parity.
6. **KV cache quantization** — FP8 KV per-head scaling, when INT4 KV is unsafe.
7. **The full parity validation methodology** — eval sets, budgets, failure modes.

By the end you should be able to produce a defended, parity-verified precision recipe for either model on H100 or H200, document it, and ship.

---

## 1. The 2026 production recipes

Starting points, in order of increasing precision aggression:

| Recipe | Llama 3.3 70B | Qwen 2.5 72B | Hardware fit |
|--------|---------------|----------------|--------------|
| **Safe baseline** | FP16/BF16 weights/activations/KV | same | 4-8× H100 (TP) or 1× H200 |
| **FP8 throughput** | FP8 E4M3 weights+activations, FP16 KV (TRT-LLM) | same | 1× H200 or 2× H100 NVL |
| **FP8 full** | FP8 weights+activations+E5M2 KV | same | 1× H200 (best single-GPU) |
| **AWQ-INT4 mainstream** | AWQ-INT4 group-128 weights, FP16 activations, FP16 KV | same (validated) | 1× H100 or 1× H200 (high batch) |
| **AWQ-INT4 + FP8 KV** | AWQ-INT4 weights, FP16 activations, FP8 KV | same | 1× H200 long context |
| **W4A8 (advanced)** | QuaRot/SpinQuant W4A8 | QuaRot/SpinQuant W4A8 | 1× H100/H200 |
| **W4A4 (research)** | SpinQuant W4A4 (parity TBD per workload) | same | Hopper edge; Blackwell native (Part 3) |

The recipes that ship in 2026 production for these two models are usually one of: **FP8 full (TRT-LLM, max throughput)** or **AWQ-INT4 (max cost-efficiency)**. The remaining recipes are tradeoffs at the margins.

---

## 2. AWQ on Llama 3.3 70B and Qwen 2.5 72B

AWQ ([arXiv:2306.00978](https://arxiv.org/abs/2306.00978)) is the default starting point for INT4. The pipeline:

```text
1. Load FP16/BF16 reference model
2. Pick calibration data — 256–1024 prompts representative of deployment
3. For each Linear layer:
   a. Compute per-channel activation magnitudes from calibration data
   b. Identify top ~1% channels with largest activation
   c. Compute per-channel scale factor s such that:
      W' = W × s    (the important channels become larger in weight space)
      X' = X / s    (and proportionally smaller in activation space)
   d. Quantize W' to INT4 with group-128 scaling
   e. Keep scale factor for the dequant kernel
4. Output a quantized model + scale factors
```

The key insight: pre-scaling weights moves "important" rows into the high-magnitude region of INT4's representable range, where quantization is finer. The activation side absorbs the inverse scale without precision loss because activations are computed at FP16 anyway.

### 2.1 Calibration data choice

The most-underrated decision in AWQ. The calibration set should match the **deployment distribution**:

* **English chat product:** 512+ prompts from WikiText or your own product logs.
* **Multilingual product (Qwen 2.5 72B + Chinese)** : include Chinese-language prompts at the same proportion as production traffic. Calibrating Qwen 2.5 72B on only English calibration data produces measurably worse parity on Chinese eval.
* **Agent / tool-use product:** include tool-call examples. Calibrating without them produces a weight quant that systematically biases against rare action tokens — exactly the bug the [BFCL lecture](../../../3.%20Edge%20AI/Agent%20Tool-Dispatch%20Evaluation%20with%20BFCL/Lecture-01.md) calls out.

**Rule:** if you cannot defend the calibration set's representativeness, you cannot defend the quant.

### 2.2 Group size

`group_size=128` is the default. It means every 128 contiguous weights share one scale factor.

* Smaller groups (64 or 32) → finer-grained scaling, slightly better parity, slightly larger storage and slower kernels.
* Larger groups (256, "channel-wise") → coarser, faster, lower parity.

For 70B-class models, **group=128 is the universal default**. Drop to 64 only if a specific axis fails parity.

### 2.3 AWQ on Llama 3.3 70B — typical result

| Recipe | MMLU | HumanEval | BFCL (simple) | Δ vs FP16 |
|--------|------|-----------|----------------|-----------|
| FP16 reference | 82.0 | 73.8 | 88.2 | — |
| AWQ-INT4 group-128, English calibration | 81.2 | 72.5 | 86.9 | -0.8 / -1.3 / -1.3 |
| AWQ-INT4 group-128, mixed calibration | 81.5 | 72.9 | 87.4 | -0.5 / -0.9 / -0.8 |

Numbers approximate; replicate in your lab. **The mixed-calibration recipe is the production-shippable one.**

### 2.4 AWQ on Qwen 2.5 72B — typical result

| Recipe | MMLU | CEval (Chinese) | HumanEval | Δ vs FP16 |
|--------|------|------------------|-----------|-----------|
| FP16 reference | 86.1 | 83.9 | 87.2 | — |
| AWQ-INT4 group-128, English-only calibration | 85.3 | 81.5 | 86.6 | -0.8 / **-2.4** / -0.6 |
| AWQ-INT4 group-128, English+Chinese calibration | 85.6 | 83.1 | 86.8 | -0.5 / -0.8 / -0.4 |

The English-only calibration loses 2.4 pp on Chinese eval — well outside any reasonable parity budget. **This is the most common quantization mistake on Qwen models in production:** developers use English calibration data because it's what's available, and ship a model that silently underperforms on the language they specifically picked Qwen for.

---

## 3. GPTQ vs AWQ

[GPTQ](https://arxiv.org/abs/2210.17323) is the older method, quantizing one weight at a time using a second-order error metric (OBS).

**Differences:**

| Property | GPTQ | AWQ |
|----------|------|-----|
| Method | OBS error minimization | Activation-aware scaling |
| Per-channel sensitivity handling | implicit | explicit |
| Calibration data sensitivity | medium | high (more sensitive to bad calibration) |
| Parity on agent workloads | ~ | typically 0.3–1 pp better |
| Parity on translation / multilingual | ~ | ~ |
| Inference kernel | identical (Marlin / standard INT4 GEMM) | identical |
| Quantization time | slow (hours for 70B) | similar |

**Recommendation:**

* Default to **AWQ** for 7B–70B dense LLMs. If parity fails on a specific axis, try GPTQ as a fallback.
* For models with severe outliers (Llama-3-70B family for activation-quant paths, see §4), neither AWQ nor GPTQ alone is sufficient — use QuaRot or SpinQuant.

---

## 4. The Llama-3-70B W8A8 anomaly

[arXiv:2408.15301](https://arxiv.org/abs/2408.15301) documented a specific failure mode of Llama-3-70B-class models under per-channel W8A8 quantization. The architectural quirk (which carries over to Llama 3.3 70B since the architecture is unchanged) is:

* A small number of MLP channels in late layers have activations with **outliers 50–100× larger than the bulk**.
* Standard per-tensor activation quantization clips these severely, producing a 3–5 pp MMLU drop.
* Per-channel scaling helps but still leaves a 2–3 pp gap.

### 4.1 Workarounds for Llama 3.3 70B

In order of increasing complexity:

1. **Stay at W4A16** — AWQ-INT4 weights with FP16 activations. Dodges the activation-quantization problem entirely. **This is what most production deployments do.**
2. **Use FP8 instead of INT8** — the FP8 dynamic range (E5M2: ±57,000) is large enough to represent the outliers without clipping. TensorRT-LLM's FP8 path on Llama 3.3 70B has no measurable parity issue.
3. **QuaRot W4A8** — Hadamard rotation invariantly spreads the outliers across channels. Parity recovers.
4. **SpinQuant W4A8** — learnable rotation, slightly better parity than QuaRot at higher pipeline complexity.

**Decision tree:**

```
Need INT8 activation quant for any reason? ─► QuaRot or SpinQuant
                                              │
Want max throughput on Hopper? ──────────► TRT-LLM FP8 (no anomaly)
                                              │
Want max cost-efficiency? ───────────────► AWQ-INT4 weights + FP16 activations
```

### 4.2 Does the anomaly apply to Qwen 2.5 72B?

The same paper benchmarked Qwen models and found they tolerate W8A8 better than Llama-3 — no equivalent severe outlier pattern. So W8A8 on Qwen 2.5 72B is viable; W8A8 on Llama 3.3 70B is not without rotation.

**This is the most concrete architecture-specific quantization insight from the Llama vs Qwen comparison.** Both models look similar in their config.json, but Llama-3 has this one quirky property that changes which precision paths are practical.

---

## 5. FP8 on Hopper — the TensorRT-LLM path

For maximum throughput on Hopper, FP8 weights + FP8 activations is the recipe. The mature path is TensorRT-LLM.

### 5.1 The TRT-LLM build

```bash
# Quantize the model
python examples/llama/quantize.py \
    --model_dir /path/to/llama-3.3-70b \
    --output_dir ./fp8_checkpoint \
    --dtype float16 \
    --qformat fp8 \
    --kv_cache_dtype fp8 \
    --calib_size 512

# Build the engine
trtllm-build \
    --checkpoint_dir ./fp8_checkpoint \
    --output_dir ./engine \
    --gemm_plugin float16 \
    --use_fp8_context_fmha enable \
    --max_input_len 8192 \
    --max_output_len 4096 \
    --max_batch_size 64 \
    --tp_size 4
```

The engine plan is precomputed for the chosen (precision, batch range, sequence length range, TP size). Changing any of these requires a rebuild.

### 5.2 Expected throughput on Hopper

On 4× H100 SXM (TP=4), Llama 3.3 70B:

| Precision | TPOT (batch=1) | TPOT (batch=64) | Throughput tok/s/GPU |
|-----------|----------------|------------------|----------------------|
| BF16 (vLLM) | ~38 ms | ~24 ms | ~660 tok/s/GPU |
| FP8 (TRT-LLM) | ~22 ms | ~13 ms | ~1200 tok/s/GPU |

FP8 ≈ 1.7–1.8× throughput improvement. Parity drop on MMLU: ~0.2 pp (well within budget).

For Qwen 2.5 72B numbers are similar but ~5% slower at the same batch due to the slightly larger model.

### 5.3 vLLM 0.7+ FP8 path

vLLM's FP8 implementation is approaching TRT-LLM in 2026. Roughly 90% of TRT-LLM throughput for ~50% of the deployment friction (no engine compile step). For chat products where iteration matters, vLLM FP8 is often the practical pick.

---

## 6. KV cache quantization

The KV cache is the second precision decision after weights, especially at long context.

### 6.1 FP8 KV cache

FP8 E5M2 (range ±57000) is the safe default for KV.

* **Cuts KV bytes in half** — at 128K context this is 21 GB instead of 42 GB per request.
* **Halves KV read bandwidth** during decode — meaningful at long context.
* **Parity drop:** ~0.1–0.3 pp on most evals if per-head scaling is used.

**Per-head scaling** matters. Per-tensor scaling can clip outlier heads (rare-token specialization heads). vLLM 0.7+, SGLang 0.4+, and TRT-LLM all support per-head FP8 KV.

### 6.2 INT4 KV cache

Aggressive. Cuts KV by 4×.

* Allows 4× longer context or 4× more concurrent requests per HBM.
* **Parity loss:** 1–3 pp on long-context evals (RULER, needle-in-haystack). On chat/agent at short context it's usually fine.
* **Always validate at the actual context length the product will use.** Short-context parity does not predict long-context behavior.

For Llama 3.3 70B / Qwen 2.5 72B at 128K context, FP8 KV is the production default. INT4 KV is a research-mode option.

---

## 7. The full parity validation methodology

The discipline from Part 1 Lecture 04, applied:

### 7.1 Pin the reference

For Llama 3.3 70B: BF16 weights, FP16 KV, official tokenizer, fixed seed list.
For Qwen 2.5 72B: same.

Hash the weights. The hash goes in the bench report. Any quant candidate is judged against this exact reference.

### 7.2 Pin the eval set

Per workload class:

* **Chat:** MMLU (subset of 500 questions, fixed seed), GSM8K (200 problems), HumanEval if code-relevant.
* **Multilingual (Qwen):** CEval, MMLU-CN, FLORES (translation if relevant).
* **Agent:** [BFCL](../../../3.%20Edge%20AI/Agent%20Tool-Dispatch%20Evaluation%20with%20BFCL/Lecture-01.md) at the categories you serve (simple, multiple, irrelevance, multi-turn).
* **Long context:** RULER at 16K / 64K / 128K.

### 7.3 The candidate run

1. Apply quantization.
2. Re-run identical eval. Compute Δ per category.
3. Inspect failures: sample 20 newly-failing items, categorize.

### 7.4 The parity budget

For Llama 3.3 70B and Qwen 2.5 72B in a production chat product, a defensible budget:

| Metric | Floor (reference) | Budget vs ref |
|--------|-------------------|----------------|
| MMLU | 82.0 / 86.1 | -1.0 pp |
| HumanEval | 73.8 / 87.2 | -1.5 pp |
| BFCL (simple) | 88.2 / ~88 | -2.0 pp |
| BFCL (irrelevance) | ~80 / ~80 | -2.0 pp |
| RULER 64K | ~94 / ~92 | -1.5 pp |
| Per-language eval (Qwen Chinese) | 83.9 | -1.5 pp |

Budgets are *workload-defended*, not generic. A code-product cannot afford -1.5 pp HumanEval; for them tighten to -0.5. A non-multilingual product can ignore CEval.

### 7.5 Failure modes specific to these models

* **Llama 3.3 70B W8A8 → MMLU drops 3 pp.** Switch to W4A16 or QuaRot.
* **Qwen 2.5 72B AWQ-INT4 with English-only calibration → CEval drops 2 pp.** Fix calibration data.
* **Either at INT4 KV at 64K+ context → RULER drops 3-5 pp.** Switch to FP8 KV.
* **FP8 KV per-tensor → BFCL drops 1 pp on rare-action tokens.** Switch to per-head FP8 KV.

These four are well-documented and recoverable. Anything you cannot recover within budget → re-pick precision recipe.

---

## Lab — produce defended recipes for both models

Goal: ship a precision recipe with parity-validated numbers for each of Llama 3.3 70B and Qwen 2.5 72B. Cap of two days.

1. **Reference runs** — BF16 / FP16 baseline for both on fixed eval set. Two seeds each → noise floor.
2. **Candidate A — AWQ-INT4 group-128** with appropriate calibration (English for Llama, English+Chinese for Qwen). Re-run eval.
3. **Candidate B — FP8 weights+activations via TRT-LLM**. Re-run eval.
4. **Candidate C — AWQ-INT4 + FP8 KV** at 64K context. Re-run eval (long-context-relevant evals only).
5. **For Llama 3.3 70B only — Candidate D — try W8A8 with QuaRot.** Compare with W4A16 baseline.
6. **Produce a recipe report** per model:
   * Precision recipe.
   * Per-eval-category Δ vs reference (with parity budget marked).
   * Per-knob throughput improvement (TPOT, throughput tok/s/GPU).
   * Recommended ship recipe with two-sentence justification.

Pass criterion: another engineer reading the report agrees the ship recipe is the right tradeoff for the stated workload class.

---

## Self-check

1. A teammate wants to deploy Llama 3.3 70B at W8A8 on H100 for max throughput. Defend or reject in two sentences, citing the specific arXiv result.
2. Why does AWQ on Qwen 2.5 72B with English-only calibration fail on CEval specifically? Walk the chain: calibration → weight scaling → which channels are protected → which language's tokens use those channels.
3. For Llama 3.3 70B at 128K context, FP8 weights + FP8 KV would put you at what total HBM (one request, no batching)? Will it fit on one H200?
4. Your AWQ-INT4 Llama 3.3 70B passes MMLU and HumanEval but BFCL (simple) drops 2.5 pp. What's the most likely failure mode and what is the first experiment to fix it?
5. For a Chinese-language chat product, between (a) AWQ-INT4 Qwen 2.5 72B at H100 and (b) FP8 Llama 3.3 70B at H200, which would you pick on cost? On quality? Be specific about the tradeoffs.

---

## References

* AWQ — [arXiv:2306.00978](https://arxiv.org/abs/2306.00978)
* GPTQ — [arXiv:2210.17323](https://arxiv.org/abs/2210.17323)
* "The Uniqueness of LLaMA3-70B Series with Per-Channel Quantization" — [arXiv:2408.15301](https://arxiv.org/abs/2408.15301)
* QuaRot — [arXiv:2404.00456](https://arxiv.org/abs/2404.00456)
* SpinQuant — [arXiv:2405.16406](https://arxiv.org/abs/2405.16406)
* TensorRT-LLM FP8 documentation — [nvidia.github.io/TensorRT-LLM/](https://nvidia.github.io/TensorRT-LLM/)
* AutoAWQ — [github.com/casper-hansen/AutoAWQ](https://github.com/casper-hansen/AutoAWQ)
* MMLU benchmark — [github.com/hendrycks/test](https://github.com/hendrycks/test)
* HumanEval — [github.com/openai/human-eval](https://github.com/openai/human-eval)
* RULER long-context eval — [arXiv:2404.06654](https://arxiv.org/abs/2404.06654)

Cross-references:

* [Part 1 → Lecture 04 — The precision stack](../Part%201%20-%20Fundamentals/Lecture-04.md)
* [Phase 5 → Edge AI → Qwen Inference Optimization → Lecture 02 — Quantizing Qwen3-4B](../../../3.%20Edge%20AI/Qwen%20Inference%20Optimization/Lecture-02.md)
* [Phase 5 → Edge AI → Agent Tool-Dispatch Evaluation with BFCL](../../../3.%20Edge%20AI/Agent%20Tool-Dispatch%20Evaluation%20with%20BFCL/Lecture-01.md) — parity for agent workloads

---

## Current as of 2026-06

AWQ, GPTQ, QuaRot, SpinQuant are the pinned methods. FP8 (E4M3 / E5M2) on Hopper via TE / TRT-LLM. Refresh when a method ships with materially better-than-AWQ parity on these specific models, or when FP4-on-Hopper paths land (none expected — that's Blackwell).

---

## Next

* Next: [Lecture 04 — Single-node multi-GPU serving (TP)](Lecture-04.md)
* Previous: [Lecture 02 — Hopper hardware story](Lecture-02.md)
* Up: [Part 2 — Dense at Hopper](README.md)
