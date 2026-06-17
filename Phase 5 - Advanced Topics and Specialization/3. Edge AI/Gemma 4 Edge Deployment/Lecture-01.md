# Lecture 01 — Why Gemma 4 at the Edge: Architecture, Competitive Position, and the Numbers

**Collection:** [Gemma 4 Edge Deployment](README.md) | **Next:** [Lecture 02 →](Lecture-02.md)

---

Edge LLM deployment has always been a size-vs-quality tradeoff: small enough to fit, capable enough to matter. Gemma 4 breaks that tradeoff differently from every prior open model because its architecture was built from first principles for **bounded memory** — not as an afterthought. This lecture explains exactly what that means in silicon terms, where Gemma 4 sits in the competitive landscape, and how to compute the numbers that determine whether a configuration fits your target.

---

## Learning objectives

By the end of this lecture, you should be able to:

1. Describe Gemma 4's **interleaved local/global attention** mechanism and explain why it produces a KV cache that grows O(1) with context length for most layers.
2. Compute the **exact KV footprint** for any Gemma 4 configuration (batch, seq, precision) and compare it to a pure-attention equivalent.
3. State the role of **QK-norm, GeGLU, GQA,** and **256K vocabulary** in Gemma 4 and explain each's impact on edge deployment.
4. Select the right Gemma 4 model size for a given Jetson target using the bandwidth-ceiling formula.
5. Explain Gemma 4's **competitive position** vs Qwen3, Phi-4, and Llama 3.2 at the edge sizes, with specific architectural reasons (not just benchmark numbers).

---

## 1. The architecture that makes Gemma 4 an edge model

Every transformer for inference on a constrained device faces the same problem: the KV cache. At long contexts, the KV cache grows linearly with sequence length and can exceed weight memory — on a Jetson with 64 GB unified memory, a 4B model at BF16 (8 GB weights) can have its KV cache dwarf its weights at 128 K tokens if standard attention is used.

Gemma 4 solves this with **interleaved local and global attention**, a design inherited from Gemma 2 and extended in Gemma 4.

### 1.1 Interleaved local/global attention — the key architectural fact

In a standard transformer, every attention layer computes full-sequence attention: each token attends to every previous token in the context. KV cache grows as `seq_len × num_kv_heads × head_dim × 2 × num_layers × batch × bytes_per_element`.

Gemma 4 replaces most attention layers with **sliding-window local attention**:

```text
Gemma 4 attention pattern (conceptual, 42-layer 12B example):

  Layer 0:  LOCAL   (sliding window, W=1024 tokens)
  Layer 1:  LOCAL
  Layer 2:  LOCAL
  Layer 3:  LOCAL
  Layer 4:  LOCAL
  Layer 5:  LOCAL
  Layer 6:  GLOBAL  ← 1 global per 6 local
  Layer 7:  LOCAL
  Layer 8:  LOCAL
  ...
  Layer 41: LOCAL

  Ratio: 1 GLOBAL per 6 LOCAL = ~6 global + ~36 local in 42 layers
```

**For local (sliding-window) layers:**

- Each token attends only to the most recent W=1024 tokens.
- KV cache per local layer = `batch × W × num_kv_heads × head_dim × 2 × bytes` — **independent of seq_len beyond W**.
- At seq_len = 128 K: local KV is IDENTICAL to local KV at seq_len = 1024.

**For global layers:**

- Standard full-context attention. KV grows with seq_len.
- But there are only ~N_global ≈ N_layers/7 of them.

This is the fundamental insight: **most of the KV cache is O(1) in context length; only the global-attention layers are O(n)**. The global layers are the "cross-document" integration mechanism; the local layers do the cheap sequential processing.

### 1.2 Gemma 4 model specifications

Gemma 4 launches in four sizes. These are architecture approximations derived from Gemma 2 lineage and published model cards (exact configs may differ slightly by variant):

| Model | Params | Layers | Hidden | Q heads | KV heads | Head dim | FFN dim | Context |
|-------|--------|--------|--------|---------|---------|---------|---------|---------|
| Gemma 4 1B | ~1.0B | 18 | 1152 | 4 | 1 | 256 | 6912 | 32K |
| Gemma 4 4B | ~4.3B | 34 | 2560 | 8 | 4 | 256 | 15360 | 128K |
| Gemma 4 12B | ~12.7B | 46 | 3840 | 16 | 8 | 256 | 24576 | 128K |
| Gemma 4 27B | ~27.4B | 62 | 5376 | 32 | 16 | 256 | 36864 | 128K |

All except the 1B support 128 K context. All use the interleaved 1:6 local/global attention pattern in the 4B–27B models.

**Vision variants:** Gemma 4 4B, 12B, and 27B have multimodal (vision) variants built around a **SigLIP-400M** image encoder fused into the decoder. Covered in Lecture 05.

### 1.3 KV cache math — working through the numbers

This is the calculation every edge engineer must be able to do in their head. Let's work Gemma 4 4B at 128 K context, batch=1, BF16 (2 bytes):

```text
Gemma 4 4B configuration:
  N_layers = 34
  W = 1024 (local attention window)
  n_global ≈ 34 / 7 ≈ 5 global layers
  n_local  ≈ 34 - 5 = 29 local layers
  KV heads = 4, head_dim = 256
  seq_len  = 128,000 tokens
  batch    = 1
  bytes    = 2 (BF16)

KV bytes per layer per token = 2 × KV_heads × head_dim × bytes
                              = 2 × 4 × 256 × 2 = 4,096 bytes/token

LOCAL layers (29 layers, window=1024):
  KV bytes = n_local × 2 × W × KV_heads × head_dim × bytes
           = 29 × 2 × 1024 × 4 × 256 × 2
           = 29 × 4,194,304 bytes
           = 121.6 MB

GLOBAL layers (5 layers, full seq_len=128K):
  KV bytes = n_global × 2 × seq_len × KV_heads × head_dim × bytes
           = 5 × 2 × 128,000 × 4 × 256 × 2
           = 5 × 524,288,000 bytes
           = 2,621 MB ≈ 2.56 GB

TOTAL KV at 128K context:   0.12 + 2.56 = ~2.68 GB
```

Now compare to a **hypothetical pure-attention 4B** (same config, no local layers):

```text
Pure-attention 4B at 128K context:
  KV bytes = 34 × 2 × 128,000 × 4 × 256 × 2
           = 34 × 524,288,000
           = 17,825,792,000 bytes ≈ 17.8 GB
```

**Gemma 4 4B at 128K: 2.68 GB vs 17.8 GB for a pure-attention model — 6.6× smaller KV cache.**

At 4K context (a typical chat turn):

```text
LOCAL  (29 layers, window=1024, capped at 4096): 29 × 2 × 1024 × 4 × 256 × 2 = 122 MB (same!)
GLOBAL (5 layers, seq=4096):                     5 × 2 × 4096 × 4 × 256 × 2  =  84 MB
TOTAL:                                           ~206 MB
```

KV at 4K is almost identical to KV at 128K for the local layers — the local layers are the cheap ones, and they dominate. This is the design: pay for local attention cheaply regardless of context, pay for global attention only in proportion to the global context.

### 1.4 Bandwidth ceiling on Jetson targets

Use the standard formula from [Edge LLM Inference Internals](../Edge%20LLM%20Inference%20Internals/Lecture-01.md):

```text
decode_ceiling(tok/s) = HBM_bandwidth_GBs / bytes_per_token_step

bytes_per_token_step ≈ weight_bytes + KV_bytes_per_step
                     ≈ weight_bytes (at short ctx, batch=1, KV small)
```

```text
Jetson AGX Orin:   204 GB/s bandwidth, 64 GB unified LPDDR5
Jetson AGX Thor:   273 GB/s bandwidth, 128 GB LPDDR5X

Model weights (INT4, 4-bit = 0.5 bytes/param):
  Gemma 4 1B  → 0.5 GB
  Gemma 4 4B  → 2.15 GB
  Gemma 4 12B → 6.35 GB
  Gemma 4 27B → 13.7 GB

Batch-1 decode ceiling (INT4 weights, short context):
  Target     │ 1B            │ 4B            │ 12B           │ 27B
  ───────────┼───────────────┼───────────────┼───────────────┼──────────────
  Orin 204   │ 204/0.5 =408  │ 204/2.15 = 95 │ 204/6.35 = 32 │ doesn't fit*
  Thor 273   │ 273/0.5 =546  │ 273/2.15 =127 │ 273/6.35 = 43 │ 273/13.7 =20

  *27B INT4 (13.7 GB) fits Orin 64 GB; the ceiling is 204/13.7 ≈ 15 tok/s — marginal
```

**How to use this table:** These are arithmetic ceilings, not achievable throughput. Actual throughput is typically 60–80% of ceiling due to KV access overhead, kernel launch gaps, and CUDA synchronization. Multiply ceiling × 0.7 for a realistic estimate.

The ceiling tells you the regime:
- **1B at Orin**: 408 tok/s ceiling → even 60% = 245 tok/s, fast enough for any real-time use
- **4B at Orin**: 95 tok/s → ~65 tok/s usable, good for chat
- **12B at Thor**: 43 tok/s → ~30 tok/s, good for autonomous decision latency
- **27B at Thor**: 20 tok/s → ~14 tok/s, usable for offline tasks, tight for real-time

---

## 2. Architecture components for edge engineers

### 2.1 Grouped Query Attention (GQA) — why it matters

Gemma 4 uses **GQA with a 2:1 Q:KV head ratio** across all model sizes (8Q/4KV for 4B, 16Q/8KV for 12B, 32Q/16KV for 27B). This halves the KV cache relative to Multi-Head Attention (MHA) without accuracy loss.

```text
KV reduction from GQA:
  MHA (naive):       KV bytes ∝ n_Q_heads × head_dim × seq_len
  MQA (extreme):     KV bytes ∝ 1 × head_dim × seq_len  (one shared KV)
  Gemma 4 GQA 2:1:   KV bytes ∝ n_KV_heads × head_dim × seq_len
                               = (n_Q_heads / 2) × head_dim × seq_len
                               = half of MHA

Combined with local attention (1024 window):
  Total KV savings vs MHA-dense = GQA factor × local-attention factor
                                = 2× × 6.6×  = ~13× smaller KV at 128K context
```

This 13× reduction is why 128K-context Gemma 4 is not a curiosity — it is actually deployable on Jetson hardware that would choke on a pure-attention model.

### 2.2 QK-Norm — why Gemma 4 is more stable to quantize

Gemma 4 applies **RMSNorm to Q and K** before the attention dot product:

```python
# Standard attention (no QK-norm):
scores = Q @ K.T / sqrt(head_dim)  # Q, K can be large → scores blow up at long seq

# Gemma 4 QK-norm:
Q_normed = rms_norm(Q, weight=q_scale)   # normalize Q
K_normed = rms_norm(K, weight=k_scale)   # normalize K
scores = Q_normed @ K_normed.T / sqrt(head_dim)  # bounded inputs → stable scores
```

**Why this matters for edge quantization:**

Without QK-norm, Q and K activations can have unbounded magnitude at long contexts — especially in early layers where attention hasn't stabilized. This makes INT8/INT4 quantization of Q and K projections error-prone: a single outlier activation can saturate the per-tensor scale and corrupt the entire tile.

With QK-norm:
- Q and K are L2-normalized → all values bounded in [-1, 1] before the learned scale
- Per-tensor INT8/INT4 quantization works well because the dynamic range is predictable
- The attention logits are similarly bounded → no attention-logit overflow at 128K sequence
- QK-norm also prevents attention entropy collapse (the "attention sink" problem) that plagues long-context deployment

In practice: **Gemma 4 at INT4 loses less accuracy than models without QK-norm**, because the normalization tames the activation outliers that kill quantization quality.

### 2.3 GeGLU activation

Gemma 4 uses **Gated GELU (GeGLU)** in the FFN:

```python
# Standard FFN:
out = W_down( SiLU(W_gate(x)) * W_up(x) )     # SwiGLU (used in Llama/Qwen)

# Gemma 4 FFN:
out = W_down( GELU(W_gate(x)) * W_up(x) )     # GeGLU (GELU instead of SiLU)
```

`GELU(x) = x × Φ(x)` (Gaussian CDF) vs `SiLU(x) = x × sigmoid(x)`. In practice the throughput difference is negligible (both are hardware-fused element-wise ops). The modeling difference matters only for fine-tuning; for inference it is transparent. The gating (the `*` in both formulas) halves the effective FFN dimension by 2×, making the FFN slightly wider than it appears (the gate and up projections together give the full FFN computation).

### 2.4 The 256K-token vocabulary

Gemma 4 uses a 256K SentencePiece vocabulary — 4× larger than typical LLMs (LLaMA uses 32K, Qwen3 uses 151K). This affects the **embedding table** and the **lm_head** projection:

```text
Embedding table size:
  Gemma 4 4B:  256K × 2560 hidden_dim × 2 bytes = 1.31 GB (BF16)
               256K × 2560 × 0.5 bytes = 0.33 GB (INT4)

  Compare Qwen3-4B (151K vocab):  151K × 2560 × 2 bytes = 0.77 GB BF16

  Gemma 4 embedding overhead: +0.54 GB BF16 vs Qwen3 at 4B class.
  In INT4, the embedding is often kept in INT8 or FP16 (embeddings quantize poorly).
  Budget ~0.66 GB extra for Gemma 4 4B vs Qwen3-4B due to vocabulary.
```

The large vocabulary is the cost of Gemma's broad multilingual coverage (100+ languages) and efficient tokenization (fewer tokens per concept → faster generation for the same wall-clock time, which partially offsets the overhead).

---

## 3. The competitive position

### 3.1 Gemma 4 vs Qwen3 at 4B class

| Criterion | Gemma 4 4B | Qwen3-4B |
|-----------|-----------|---------|
| Context window | 128K | 32K |
| KV at 128K (batch=1) | **~2.7 GB** | ~9.8 GB (pure attention) |
| Multimodal native | Yes (SigLIP) | No (separate model) |
| Quantization stability | QK-norm: high | QK-norm: Yes (Qwen3 added this too) |
| Google AI Edge / LiteRT | **Yes, first-class** | No |
| License | Gemma ToS (permissive) | Apache-2.0 |
| Best runtime on Jetson | llama.cpp, LiteRT, MLC-LLM, TRT-LLM | llama.cpp, MLC-LLM |
| Vocabulary | 256K | 151K |

**Which to choose:** If your deployment target is Jetson with Google AI Edge toolchain, Gemma 4. If you need Apache-2.0 licensing or prefer the Qwen ecosystem and don't need 128K, Qwen3-4B is equally strong. If you need 128K on Jetson, Gemma 4 wins decisively.

### 3.2 Gemma 4 vs Llama 3.2 at 3B/4B class

Llama 3.2 3B is the dominant sub-4B choice from Meta. Key differences:
- Llama 3.2 3B uses **standard attention** (no local/global split) → KV grows linearly
- Llama 3.2 1B/3B have **128K context** but the full-attention KV at 128K is ~1.2 GB (3B) vs Gemma 4 4B's 2.7 GB — actually Llama 3.2 wins on absolute KV here because it's smaller
- But **at the 12B–27B scale**, the KV advantage of Gemma 4's interleaved attention becomes decisive
- **Vision**: Llama 3.2 11B Vision vs Gemma 4 4B/12B Vision — roughly comparable; Gemma 4 has a lighter vision encoder path

**The honest edge position:** For pure text at < 4B, Llama 3.2 3B and Qwen3-4B are strong alternatives. Gemma 4 wins clearly at 12B+, and wins on 128K context at every size, and wins when you want the Google AI Edge ecosystem.

### 3.3 Where Gemma 4 is the obvious choice

1. **128K long-context on Jetson**: Nothing else has this KV efficiency at the edge
2. **Multimodal on Jetson Orin/Thor**: The 4B and 12B vision models are the most efficient VLMs for Jetson in 2025
3. **Google AI Edge deployment** (Android/iOS/embedded Linux via LiteRT): First-class support
4. **Physical AI + language**: Robot that needs both vision understanding and long-context dialogue — Gemma 4 12B vision on a Thor
5. **Self-speculative decoding**: Using Gemma 4 1B as draft for 4B/12B targets is natural (same tokenizer, same architecture family) — covered in Lecture 04

---

## 4. Hardware fit guide

### 4.1 Jetson AGX Orin (204 GB/s, 64 GB, 275 TOPS INT8)

```text
Model         │ Format    │ Weights │ KV @ 4K ctx, b=4 │ Total   │ Fits? │ ~tok/s
──────────────┼───────────┼─────────┼──────────────────┼─────────┼───────┼───────
Gemma 4 1B   │ INT4      │ 0.5 GB  │ 0.05 GB          │ 0.55 GB │ ✓     │ ~245
Gemma 4 4B   │ INT4      │ 2.2 GB  │ 0.2 GB           │ 2.4 GB  │ ✓     │ ~67
Gemma 4 12B  │ INT4      │ 6.4 GB  │ 0.4 GB           │ 6.8 GB  │ ✓     │ ~22
Gemma 4 27B  │ INT4      │ 13.7 GB │ 0.8 GB           │ 14.5 GB │ ✓     │ ~10
Gemma 4 4B   │ BF16      │ 8.6 GB  │ 0.8 GB           │ 9.4 GB  │ ✓     │ ~17
Gemma 4 12B  │ BF16      │ 25.4 GB │ 1.6 GB           │ 27 GB   │ ✓     │ ~6
```

### 4.2 Jetson AGX Thor (273 GB/s, 128 GB, ~1035 FP8 TFLOPS dense)

```text
Model         │ Format    │ Weights │ KV @ 128K ctx, b=1 │ Total   │ Fits? │ ~tok/s
──────────────┼───────────┼─────────┼────────────────────┼─────────┼───────┼───────
Gemma 4 4B   │ INT4      │ 2.2 GB  │ 2.7 GB             │ 4.9 GB  │ ✓     │ ~127
Gemma 4 12B  │ INT4      │ 6.4 GB  │ 5.1 GB             │ 11.5 GB │ ✓     │ ~43
Gemma 4 27B  │ INT4      │ 13.7 GB │ 10.8 GB            │ 24.5 GB │ ✓     │ ~20
Gemma 4 4B   │ FP8       │ 4.3 GB  │ 2.7 GB             │ 7.0 GB  │ ✓     │ ~64
Gemma 4 12B  │ BF16      │ 25.4 GB │ 5.1 GB             │ 30.5 GB │ ✓     │ ~11
```

**Key takeaway for Thor:** The 27B model fits in Thor at INT4 with 128 K context and room to spare — this is unprecedented for a 27B model at that context length on a single-board edge device.

---

## 5. The edge use-case framing

Three use-cases drive Gemma 4 at the edge in 2025:

**Physical AI / robotics:** Robots need language AND vision AND long-memory. Gemma 4 12B vision on Thor gives all three in one model: the VLM handles visual questions about the environment, the 128K window holds multi-step task history without truncating, and the 43 tok/s throughput is fast enough for real-time command understanding.

**Automotive / autonomous vehicles:** DRIVE Thor (DRIVE architecture, same silicon as Jetson AGX Thor) targets centralized compute. Gemma 4 27B INT4 as a co-driver for natural language navigation, safety reporting, and driver communication — with 128K for long-trip context.

**Edge agents on mobile / embedded Linux:** Gemma 4 4B INT4 on an Orin Nano 8 GB board is a genuinely capable local agent — handles tool calls, follows instructions, and maintains conversation context — entirely offline, no cloud required.

---

## Key takeaways

- Gemma 4's **interleaved 1:6 local/global attention** produces a KV cache that is O(1) for 6/7 of layers and O(n) for only 1/7 — making 128K context 6.6× cheaper than pure attention at the 4B scale.
- **Combined with GQA (2:1)**, total KV savings vs a MHA-dense equivalent reach ~13× at 128K context.
- **QK-norm** normalizes Q and K before the dot product, preventing attention-logit overflow at long sequences and making INT4/INT8 quantization more stable than in models without it.
- The **bandwidth-ceiling formula** (`tok/s_max = BW_GB/s ÷ bytes_per_weight_byte`) predicts that Gemma 4 1B runs at ~245 tok/s on Orin (INT4), 4B at ~67, 12B at ~22. Multiply by 0.7 for realistic throughput.
- **Competitive position:** Gemma 4 wins at 128K long-context on Jetson and at the 12B–27B multimodal edge tier. At pure 4B text, Qwen3-4B and Llama 3.2 3B are competitive. At the edge with Google AI Edge / LiteRT, Gemma 4 is the only choice.

---

## References

- Google DeepMind Gemma 4 model card and technical report (April 2025) — [ai.google.dev/gemma/docs/gemma4](https://ai.google.dev/gemma/docs/gemma4)
- NVIDIA Jetson AGX Thor product page (273 GB/s, 128 GB LPDDR5X) — [developer.nvidia.com/embedded/jetson-agx-thor](https://developer.nvidia.com/embedded/jetson-agx-thor)
- NVIDIA Jetson AGX Orin developer kit — [developer.nvidia.com/embedded/jetson-agx-orin-developer-kit](https://developer.nvidia.com/embedded/jetson-agx-orin-developer-kit)
- Gemma 2 Technical Report (Google DeepMind, 2024) — basis for Gemma 4 architecture — arXiv:2408.00118
- "GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints" (Ainslie et al., 2023) — arXiv:2305.13245
- *MLSys Deep Dives — Lecture 01 — the bandwidth-ceiling equation* — [Lecture-01.md](../../7.%20ML%20Systems%20Engineering/MLSys%20Deep%20Dives/Lecture-01.md)
- *Edge LLM Inference Internals — GEMV decode and memory wall* — [Lecture-01.md](../Edge%20LLM%20Inference%20Internals/Lecture-01.md)

---

## Current as of 2026-06

Gemma 4 model family: 1B/4B/12B/27B, released April 2025, 128K context for 4B–27B, interleaved 1:6 local/global attention, W=1024 sliding window, GQA 2:1, QK-norm, GeGLU, 256K vocabulary. Jetson AGX Thor spec: 128 GB LPDDR5X @ 273 GB/s, 75–130 W, dev kit $3,499 (Nov 2025).

---

*Up: [Gemma 4 Edge Deployment](README.md) · Next: [Lecture 02 — Quantization and Format Conversion](Lecture-02.md)*
