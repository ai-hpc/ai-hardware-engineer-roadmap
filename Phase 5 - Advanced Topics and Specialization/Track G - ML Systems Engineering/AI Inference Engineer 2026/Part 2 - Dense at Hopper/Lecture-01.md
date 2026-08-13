# Part 2 · Lecture 01 — Anatomy of a 70B-Class Dense Model: Llama 3.3 70B vs Qwen 2.5 72B

## Overview

Two models, one architecture family, two production deployments — this lecture takes the side-by-side comparison from a model-card-level summary down to inference-graph tensor shapes and concrete cost numbers per stage.

Both Llama 3.3 70B (Meta, 2024-12) and Qwen 2.5 72B (Alibaba, 2024-09) are **dense decoder-only transformers** that share:

* 80 transformer layers
* GQA with 64 query heads and 8 KV heads (head_dim 128)
* RoPE positional encoding, RMSNorm, SwiGLU FFN
* 128K context window (reached by different routes: Meta's `llama3` RoPE frequency scaling for Llama 3.3, YaRN for Qwen 2.5)

They are, in fact, **dimensionally almost identical** — same hidden size (8192), same 80 layers, same GQA geometry. They differ in three smaller places that matter for inference engineering:

1. **Vocabulary / tokenizer** — Llama 3.3 uses a tiktoken-derived BPE at ~128K vocab; Qwen 2.5 uses its own 152K BPE optimized for multilingual (especially Chinese) content. The bigger embed + LM-head matrices add ≈0.4B params to the 70B-vs-72B gap (most of the gap is the ~3% wider FFN, ≈1.8B — see §5), and tokenization efficiency differs by ~20–30% on Chinese.
2. **QKV bias** — Qwen keeps the bias terms on Q, K, V projections; Llama is bias-free. Tiny memory footprint, modest impact on long-context extrapolation behavior.
3. **FFN width** — Qwen's intermediate size is *slightly* larger (29568 vs Llama's 28672, ~3%). Real, but minor — not the "Qwen is 50% wider" you'll see in secondary sources (which misquote Qwen as 12288 hidden / 49152 FFN; §2.1 shows why that fails a back-of-envelope check).

This lecture covers:

1. The shared architecture — what every modern dense LLM ships in 2025–2026.
2. The four differences and their inference-cost impact.
3. KV cache cost per token, derived from `config.json` for both.
4. Tensor shapes per layer — exactly what gets matmul'd.
5. Parameter accounting — where the 70B and 72B labels come from and what they hide.
6. Practical impact — how runtime picks (vLLM / SGLang / TRT-LLM), quantization, and deployment shape change between the two.

By the end you should be able to read the `config.json` for either model and predict, in concrete numbers, what each forward pass step costs in HBM bytes and FLOPs.

---

> 🧠 **See it in 3D.** This lecture's architecture is rendered to scale in the [**LLM Inference Visualizer**](https://github.com/ai-hpc/llm-inference-viz) — switch between Qwen 2.5 72B and Llama 3.3 70B, hover each stage (embedding → RMSNorm → GQA → RoPE → softmax → SwiGLU), and watch the roofline mark every op memory- or compute-bound.

---

## 1. The shared architecture

Both models implement the same canonical 2024+ decoder-only design:

```text
input tokens (vocab → embeddings)
       │
       ▼
   ┌─────────────────────────────────────────────────┐
   │ for layer in 1..80:                             │
   │   ┌──── attention block ──────────────────────┐ │
   │   │  RMSNorm                                  │ │
   │   │  Q, K, V projections (GQA: 64Q, 8KV)      │ │
   │   │  RoPE on Q and K                          │ │
   │   │  scaled-dot-product attention (masked)    │ │
   │   │  output projection                        │ │
   │   │  residual add                             │ │
   │   └───────────────────────────────────────────┘ │
   │   ┌──── feed-forward block ───────────────────┐ │
   │   │  RMSNorm                                  │ │
   │   │  gate projection, up projection           │ │
   │   │  SwiGLU (silu(gate) * up)                 │ │
   │   │  down projection                          │ │
   │   │  residual add                             │ │
   │   └───────────────────────────────────────────┘ │
   └─────────────────────────────────────────────────┘
       │
       ▼
   RMSNorm + LM head (output projection to vocab)
       │
       ▼
   logits → sampler → next token
```

Both:

* Use **GQA** (8 KV heads shared across 64 query heads → group size 8).
* Use **RoPE** for position encoding, extended to 128K by different routes — Llama 3.3 via Meta's `"rope_type": "llama3"` frequency scaling plus long-context continued pretraining; Qwen 2.5 via YaRN applied at inference over a 32K native window.
* Use **RMSNorm** with epsilon ≈ 1e-6 / 1e-5.
* Use **SwiGLU** in the FFN (gate, up, down projections).
* Are bidirectional-positional via RoPE, **causally masked** for decoding.
* Tie or do not tie the LM head with input embeddings — Llama 3.3 70B does **not** tie (separate output matrix), Qwen 2.5 72B does **not** tie either. (The 4B-class Qwen3 *does* tie, but 72B does not.)

This is the architecture **every 7B–72B dense LLM ships today**. Mastering it = portable.

---

## 1.1 RMSNorm — what it actually does

Applied twice per block (pre-attention, pre-MLP), so 160 times across 80 layers. Worth understanding cold, not just naming.

### The problem it solves

Neural networks produce activations that drift in scale as they pass through layers:

```text
Layer 1 output: small values (~0.1)
Layer 40 output: large values (~50)
Layer 80 output: exploding or vanishing
```

Without normalization, **gradients explode or vanish** and training fails for deep stacks.

### The formula

```text
RMSNorm(x) = γ ⊙ (x / RMS(x))

where RMS(x) = sqrt( (1/d) · Σ xᵢ² )
```

### Intuition: RMS = "typical magnitude"

Take `x = [2, -2, 1, -1]`:

```text
Step 1 — square (removes sign): [4, 4, 1, 1]
Step 2 — average:               (4+4+1+1)/4 = 2.5
Step 3 — square root:           √2.5 ≈ 1.58
```

Why square? Because positive and negative values cancel if you average directly:

```text
[+5, -5] → average = 0  ❌  (looks empty; the signal is actually strong)
[+5, -5] → RMS     = 5  ✔  (captures the true energy)
```

RMS measures **signal energy**, not center.

### The two-step operation

**Step A — measure scale:** divide by RMS → vector now has stable unit magnitude.

**Step B — re-scale (learned):** multiply by learned weight `γ` → the model decides how loud each layer should be.

```text
Before:  x = [3, -4]    → RMS = √((9+16)/2) ≈ 3.54  (magnitude uncontrolled)
After:   x / RMS ≈ [0.85, -1.13]                     (stable ~unit magnitude)
With γ:  γ ⊙ [0.85, -1.13]                           (model-learned scale)
```

> **Analogy:** think of each token vector as an audio signal. RMS = volume meter. RMSNorm = automatic gain control. No matter how loud the layer gets, it keeps the signal at a stable, learnable loudness.

### Why not LayerNorm?

LayerNorm also subtracts the mean (centers the distribution). In practice, centering adds computation without consistent benefit for transformer activations — the scale matters far more than the center. RMSNorm drops the centering:

```text
LayerNorm:  center (subtract mean) + scale
RMSNorm:    scale only
```

Result: **simpler, ~10% faster, essentially the same quality** for deep decoders. All modern dense LLMs (Llama, Qwen, Mistral, Gemma) use RMSNorm for this reason.

### What this means for inference engineering

* **Cost:** elementwise, bandwidth-bound at decode (reads activations once, writes once). Negligible FLOPs — ~0.5 FLOP/B, well below the roofline ridge. Two calls per block × 80 blocks = 160 calls per token, but each is cheap.
* **ε (epsilon):** `1e-6` for Qwen 2.5 72B, `1e-5` for Llama 3.3 70B. Adds to the denominator to prevent division by zero when activations are near-zero. Inference-irrelevant in practice.
* **γ weights:** 1 vector of `d_model` floats per RMSNorm call × 2 per block × 80 blocks = 160 × 8192 ≈ 1.3M params — tiny fraction of 72B, but each must be loaded from HBM at decode.

---

## 2. The four differences

### 2.1 Width — nearly identical (and a cautionary tale)

| Model | hidden (d) | intermediate (d_ff) | d_ff / d |
|-------|-----------|---------------------|----------|
| Llama 3.3 70B | 8192 | 28672 | 3.5 |
| Qwen 2.5 72B | 8192 | 29568 | 3.6 |

Same hidden dimension; Qwen's FFN is only ~3% wider. Per-layer decode costs (batch=1) are therefore nearly equal:

| Cost | Llama 3.3 70B | Qwen 2.5 72B | Ratio |
|------|----------------|----------------|-------|
| FFN gate+up+down HBM read | 3 × d × d_ff × bytes ≈ 3 × 8192 × 28672 × 2 ≈ 1.41 GB FP16 | 3 × 8192 × 29568 × 2 ≈ 1.45 GB FP16 | Qwen 1.03× |
| FFN matmul FLOPs (per token) | 6 × d × d_ff ≈ 1.41 GFLOP | 6 × 8192 × 29568 ≈ 1.45 GFLOP | Qwen 1.03× |
| Attention QKVO proj HBM | ≈ 302 MB | ≈ 302 MB | 1.0× |

> ⚠️ **Common misquote.** Many secondary sources list Qwen 2.5 72B as **12288 hidden / 49152 FFN**, implying it is "50% wider" than Llama. That is wrong — 12288 is GPT-3's width, not Qwen's. The fastest way to catch it: 12288 hidden with a 49152 FFN across 80 layers would weigh in at **~160B+ parameters**, not 72B. When a config doesn't reconcile with the parameter count on the box, distrust the config. §4 derives the real shapes straight from the official `config.json`.

### 2.2 Attention head geometry — identical

| Model | num_q_heads | num_kv_heads | head_dim |
|-------|------------|--------------|----------|
| Llama 3.3 70B | 64 | 8 | 128 |
| Qwen 2.5 72B | 64 | 8 | 128 |

Identical. This means:

* **KV cache per token is identical** between the two models (320 KB/token at FP16, per Part 1 Lecture 02).
* **Attention computation per token is identical** — the attention matmul depends on Q × K^T at (h_q, head_dim) shape, same for both.

For a long-context workload, the two models have the **same KV memory pressure**. The differentiator is the **FFN cost**.

### 2.3 QKV bias

| Model | Q bias | K bias | V bias |
|-------|--------|--------|--------|
| Llama 3.3 70B | absent | absent | absent |
| Qwen 2.5 72B | present | present | present |

Memory cost: 8192 + 1024 + 1024 = 10,240 floats × 80 layers ≈ 819K floats ≈ 1.6 MB at FP16. Negligible.

Inference cost: one extra add per matmul. Negligible on modern hardware.

**Why does it matter?** It is a small architectural commitment Qwen made because the team observed that bias terms in QKV projections help long-context extrapolation behavior. The cost is essentially zero, so it ships. For an inference engineer it is a one-line config flag (`use_bias` or `attention_bias` in HF transformers).

### 2.4 Vocabulary and tokenizer

| Model | Vocab size | Tokenizer base |
|-------|-----------|----------------|
| Llama 3.3 70B | 128,256 | tiktoken-derived BPE |
| Qwen 2.5 72B | 152,064 | Qwen BPE (multilingual + code optimized) |

Differences with inference impact:

* **Embedding matrix size:** Llama 128256 × 8192 ≈ 1.05B params; Qwen 152064 × 8192 ≈ 1.25B params (≈ 2.5 GB FP16 each for embed and LM head, untied). Qwen's larger vocab adds ≈0.4B params to the 72B-vs-70B parameter gap — real, but the ~3% wider FFN contributes the bulk (≈1.8B; see §5).
* **LM head matrix:** same sizes as embeddings (untied).
* **Tokenization efficiency** — for a given text:
  * English: Llama's tokenizer is ~5% more efficient than Qwen's.
  * Chinese: Qwen's tokenizer is ~25–30% more efficient.
  * Code: Qwen's is ~10% more efficient.
* **For deployment latency:** the same Chinese input produces fewer tokens with Qwen → fewer decode steps → lower wall-clock for the same response content. This is a real win in Chinese-language products.

---

## 3. KV cache per token, derived from config

A senior engineer derives this on a whiteboard. Both models:

```text
kv_bytes_per_token = 2 × L × num_kv_heads × head_dim × bytes
                   = 2 × 80 × 8 × 128 × 2 (FP16)
                   = 327,680 bytes
                   ≈ 320 KB / token
```

| Context | KV cache @ FP16 | @ FP8 | @ INT4 |
|---------|------------------|-------|--------|
| 4,096   | 1.3 GB          | 0.65 GB | 0.33 GB |
| 32,768  | 10.5 GB         | 5.25 GB | 2.6 GB  |
| 131,072 | 42 GB           | 21 GB   | 10.5 GB |

**Per request.** This is the same for both models. **Long-context serving forces the FP8-KV decision** regardless of which model you pick from this pair.

---

## 4. Tensor shapes per layer — exactly what gets matmul'd

For a forward pass step (single token, decode batch=1):

### 4.1 Llama 3.3 70B

| Tensor | Shape | Size FP16 |
|--------|-------|-----------|
| `attn_q.weight` | (8192, 8192) | 134 MB |
| `attn_k.weight` | (8192, 1024) | 17 MB |
| `attn_v.weight` | (8192, 1024) | 17 MB |
| `attn_o.weight` | (8192, 8192) | 134 MB |
| `ffn_gate.weight` | (8192, 28672) | 470 MB |
| `ffn_up.weight` | (8192, 28672) | 470 MB |
| `ffn_down.weight` | (28672, 8192) | 470 MB |
| Per-layer total | — | **~1.7 GB** |
| × 80 layers | — | **~136 GB FP16** |
| + embed + LM head | 128256 × 8192 × 2 × 2 | **+ 4 GB** |
| **Total** | | **~140 GB FP16** |

### 4.2 Qwen 2.5 72B

From the official `config.json` (`hidden_size: 8192`, `intermediate_size: 29568`). Note `attn_q` projects to `num_q_heads × head_dim = 64 × 128 = 8192`, and K/V to `8 × 128 = 1024`:

| Tensor | Shape | Size FP16 |
|--------|-------|-----------|
| `attn_q.weight` | (8192, 8192) | 134 MB |
| `attn_k.weight` | (8192, 1024) | 17 MB |
| `attn_v.weight` | (8192, 1024) | 17 MB |
| `attn_o.weight` | (8192, 8192) | 134 MB |
| `ffn_gate.weight` | (8192, 29568) | 485 MB |
| `ffn_up.weight` | (8192, 29568) | 485 MB |
| `ffn_down.weight` | (29568, 8192) | 485 MB |
| QKV biases | (8192 + 1024 + 1024) × 2 | ~25 KB |
| Per-layer total | — | **~1.76 GB** |
| × 80 layers | — | **~141 GB FP16** |
| + embed + LM head | 152064 × 8192 × 2 × 2 | **+ 5 GB** |
| **Total** | | **~146 GB FP16** |

Almost the same per-layer footprint as Llama 3.3 70B (~1.7 GB) — the FFN is just ~3% larger. The whole 72B-vs-70B gap is then a slightly wider FFN (~3% per layer) + a larger vocab (152K vs 128K → ~1 GB more in embed + LM head) + negligible QKV biases.

> 🔍 **This reconciles — the 12288 myth didn't.** ~1.76 GB/layer × 80 + ~5 GB embeddings ≈ **146 GB ≈ 73B params × 2 B**, matching the "72B" on the box. The 12288 / 49152 figure would have given ~328 GB (~164B params) — and that mismatch is exactly the tell. **Always derive from the published `config.json`, then sanity-check against the advertised parameter count.** Third-party summaries get widths wrong surprisingly often.

**Takeaway:** Llama 3.3 70B and Qwen 2.5 72B are *architecturally near-identical* dense decoders. Their real inference-relevant differences are:

* **Tokenizer efficiency** — fewer tokens for Chinese / code with Qwen → fewer decode steps for the same content (the biggest practical win).
* **Vocab size** — Qwen's 152K vs Llama's 128K adds ~1 GB to embed + LM head.
* **FFN width and QKV biases** — both real but negligible (~3% per layer; ~2 MB).
* **Training data / post-training quality** — the dominant *behavioral* difference, invisible to the inference graph.

---

## 5. Parameter accounting — where 70B and 72B come from

Quick sanity check using the corrected numbers.

### Llama 3.3 70B

```text
Per-layer attention (Q + K + V + O):
  Q: 8192 × 8192 = 67M
  K: 8192 × 1024 = 8.4M
  V: 8192 × 1024 = 8.4M
  O: 8192 × 8192 = 67M
  Attention total: ~151M

Per-layer FFN (gate + up + down):
  gate: 8192 × 28672 = 235M
  up:   8192 × 28672 = 235M
  down: 28672 × 8192 = 235M
  FFN total: ~705M

Per-layer total: ~856M
× 80 layers: ~68.5B

Embeddings + LM head: 128256 × 8192 × 2 = ~2.1B (untied)
RMSNorm: ~negligible

Total: ~70.6B
```

Matches "70B" within rounding.

### Qwen 2.5 72B

```text
Per-layer (slightly larger):
  Attention same: ~151M
  FFN: 8192 × 29568 × 3 = ~727M
  Per-layer: ~878M
× 80 layers: ~70.2B

Embeddings + LM head: 152064 × 8192 × 2 = ~2.5B
QKV biases: ~2 MB (negligible at param count)

Total: ~72.7B
```

Matches "72B" within rounding.

The ~2B parameter difference between the two labels comes mostly from the ~3% wider FFN (≈22M more per layer × 80 ≈ 1.8B); the larger vocab adds ≈0.4B in embed + LM head. Architecturally, treat them as nearly identical for inference engineering purposes.

---

## 6. Practical impact — what changes between deploying these two

### 6.1 Runtime picks

* Both supported as first-class models in **vLLM**, **SGLang**, **TensorRT-LLM**, **llama.cpp** as of mid-2026.
* No runtime-specific behavior differs meaningfully between the two.
* `transformers` config differences: `attention_bias=True` for Qwen, `tie_word_embeddings=False` for both.

### 6.2 Quantization

* **AWQ-INT4 works well on both.** Calibration sets should match the deployment language distribution.
* **The arXiv:2408.15301 W8A8 anomaly applies to Llama 3.3 70B.** Qwen 2.5 72B is more W8A8-tolerant by the same paper's measurements.
* For W4A4 / FP4, both benefit from QuaRot or SpinQuant. We will walk this in Lecture 03.

### 6.3 Deployment shape on common hardware

| Hardware | Llama 3.3 70B | Qwen 2.5 72B | Notes |
|----------|----------------|----------------|-------|
| 1× H100 80G | INT4 only, tight | INT4 only, very tight | KV cache pressure limits batch |
| 1× H200 141G | FP8 with small batch, INT4 with batch | Same | H200 is the sweet spot for single-GPU 70B-class |
| 2× H100 NVL (TP=2) | FP8 with batch | Same | ~35 GB/GPU at FP8 fits comfortably with KV |
| 4× H100 80G (TP=4) | FP16/BF16 native | FP16/BF16 native | 35–36B/GPU, comfortable |
| 8× H100/H200 (TP=8) | FP16, large batch, long context | Same | production sweet spot for max throughput |

For Llama 3.3 70B at 32K context, FP8 weights + FP8 KV on 2× H100 NVL is the cost-effective recipe in 2026. Qwen 2.5 72B benefits from the same recipe with no surprises (slightly more memory due to vocab).

### 6.4 Tokenizer-driven cost difference

For an English-only chat product the two models have **nearly identical $/MTok** at the same recipe. For a Chinese-language product Qwen 2.5 72B emits **~25% fewer tokens** for the same response — meaning the *effective* $/MTok is ~25% lower. This is the **largest engineering difference** between the two for many product contexts.

---

## Lab — derive both configs from disk and produce a side-by-side cost report

Goal: a Markdown report in your benchmark repo with side-by-side cost numbers.

1. **Download both `config.json` files** from the official Hugging Face repos.
2. **Compute, programmatically:**
   * Per-layer weight memory at FP16, FP8, INT4.
   * Total parameter count (verify matches 70B / 72B labels).
   * KV bytes per token at FP16, FP8, INT4.
   * Embedding + LM head memory.
3. **Render a comparison table** with both models, all three precisions.
4. **Predict** total HBM at four scenarios: (batch=1, ctx=4K) / (batch=1, ctx=128K) / (batch=16, ctx=4K) / (batch=16, ctx=32K).
5. **Decide** which hardware × precision recipe each scenario forces. Write down the reasoning for each.

Pass criterion: your report can be reproduced by another engineer from the same configs, and the predictions match measured numbers (Lecture 02 will validate them on real H100/H200).

---

## Self-check

1. The W8A8 Llama-3-70B anomaly (arXiv:2408.15301) is well-documented. Does the same anomaly likely apply to Qwen 2.5 72B? Why or why not, given what you now know about the architectural similarities?
2. A teammate proposes deploying Qwen 2.5 72B FP16 on 4× H100 80G for an English-language chat product. Without running it: does it fit? Show the KV cache + weight memory math.
3. For a Chinese-language chat product at 4× H100, would you pick Llama 3.3 70B INT4 or Qwen 2.5 72B INT4? Justify in two sentences using tokenizer efficiency.
4. Both models share the same KV head structure (8 KV heads × head_dim 128). What is the minimum HBM you would budget for KV cache alone at batch=64, context=8K, FP8 KV?
5. The vanilla Qwen 2.5 72B has `hidden_size=8192`, not the 12288 some secondary sources cite. What is the lesson for an inference engineer reading product specs from non-primary sources?

---

## References

* Llama 3.3 70B model card — [huggingface.co/meta-llama/Llama-3.3-70B-Instruct](https://huggingface.co/meta-llama/Llama-3.3-70B-Instruct)
* Qwen 2.5 72B model card — [huggingface.co/Qwen/Qwen2.5-72B-Instruct](https://huggingface.co/Qwen/Qwen2.5-72B-Instruct)
* Qwen 2.5 technical report — [arXiv:2412.15115](https://arxiv.org/abs/2412.15115)
* "The Uniqueness of LLaMA3-70B Series with Per-Channel Quantization" — [arXiv:2408.15301](https://arxiv.org/abs/2408.15301)
* GQA paper — [arXiv:2305.13245](https://arxiv.org/abs/2305.13245)
* RoPE paper — [arXiv:2104.09864](https://arxiv.org/abs/2104.09864)
* YaRN — [arXiv:2309.00071](https://arxiv.org/abs/2309.00071) — context extension method used by Qwen 2.5 (32K native → 128K at inference); Llama 3.1/3.3 instead use Meta's `"rope_type": "llama3"` frequency scaling + long-context continued pretraining

Cross-references:

* [Part 1 → Lecture 02 — Transformer execution](../Part%201%20-%20Fundamentals/Lecture-02.md)
* [Phase 5 → Edge AI → Qwen Inference Optimization → Lecture 01 — Architecture Deep Dive](../../../Track%20C%20-%20Edge%20AI/Qwen%20Inference%20Optimization/Lecture-01.md) — Qwen 4B/72B side-by-side (different focus, related material)

---

## Current as of 2026-06

Configs pinned from the official Hugging Face cards at the time of writing. Refresh if Meta or Alibaba publishes a v2 / point release of either model with architectural changes.

---

## Next

* Next: [Lecture 02 — Hopper hardware story](Lecture-02.md)
* Up: [Part 2 — Dense at Hopper](README.md)
