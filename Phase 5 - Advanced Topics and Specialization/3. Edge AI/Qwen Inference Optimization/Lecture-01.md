# Lecture 1: Qwen Architecture Deep Dive — Qwen3-4B and Qwen2.5-72B Side by Side

## Overview

Before you can optimize Qwen inference, you need to know what's actually in the weights. This lecture is a complete walk of the architecture for two specific releases:

* **Qwen3-4B-Instruct** — Alibaba's 2026 small-instruct model. The size most people will actually run on edge.
* **Qwen2.5-72B-Instruct** — the 2024 large-instruct workhorse, still the most-deployed 70B-class open-weight model.

Both are decoder-only causal transformers in the same family. They share design choices (GQA, RoPE-NeoX, SwiGLU FFN, RMSNorm, BPE with 151 936 vocab) but differ in scale, embedding tying, and rotary configuration in ways that change your inference recipe.

By the end you should be able to:

* Map every byte in a Qwen GGUF / safetensors file to a specific role in the forward pass.
* Compute the parameter count and per-tensor memory footprint from `config.json` alone.
* Explain why Qwen3-4B ties embeddings and Qwen2.5-72B does not — and what that costs at the LM head.
* Read a GEMV trace and identify the layer/block/projection from M and K alone.

---

## 1. Family Lineage (the short version)

```
Qwen 1.5  ──► Qwen2  ──► Qwen2.5  ──► Qwen3
 (2023)     (2024)      (2024)       (2026)
   │          │            │            │
   │          │            │            ├─ "thinking" mode (CoT toggleable)
   │          │            │            ├─ better tool use
   │          │            ├─ YaRN extended context (128 k)
   │          │            ├─ improved post-training
   │          ├─ GQA standard, dual rope-base for short/long ctx
   │          │
   ├─ original release, MHA only
```

Both models we cover are **decoder-only**, **causal**, **bidirectional-positional via RoPE**, **left-padded for batch**, **uses ChatML-style template** (`<|im_start|>role\n…<|im_end|>`).

---

## 2. The Config That Drives Everything

`config.json` for each model:

| Field | Qwen3-4B-Instruct | Qwen2.5-72B-Instruct |
|---|---|---|
| `hidden_size` (d_model) | 2560 | 8192 |
| `intermediate_size` (FFN) | 6912 | 29568 |
| `num_hidden_layers` | 36 | 80 |
| `num_attention_heads` | 32 | 64 |
| `num_key_value_heads` | 8 (GQA group=4) | 8 (GQA group=8) |
| `head_dim` | 128 | 128 |
| `vocab_size` | 151 936 | 152 064 |
| `max_position_embeddings` | 40 960 (262 144 w/ YaRN) | 32 768 (131 072 w/ YaRN) |
| `rope_theta` | 1 000 000 | 1 000 000 |
| `rope_scaling` | YaRN (rel) | YaRN (rel) |
| `tie_word_embeddings` | **true** | **false** |
| `rms_norm_eps` | 1e-6 | 1e-6 |
| `torch_dtype` (release) | BF16 | BF16 |

These twelve numbers determine every shape you'll see in a GEMV trace.

### 2.1 Deriving shapes by hand

Given the config, every tensor's shape is mechanical:

| Tensor | Qwen3-4B | Qwen2.5-72B | Math |
|---|---|---|---|
| `token_embd.weight` | 151936 × 2560 | 152064 × 8192 | `vocab × d` |
| Per layer: `attn_norm.weight` | 2560 | 8192 | `d` |
| `attn_q.weight` (W_Q) | 2560 × 4096 | 8192 × 8192 | `d × (n_heads · head_dim)` |
| `attn_k.weight` (W_K) | 2560 × 1024 | 8192 × 1024 | `d × (n_kv_heads · head_dim)` |
| `attn_v.weight` (W_V) | 2560 × 1024 | 8192 × 1024 | same as K (GQA) |
| `attn_q.bias` | 4096 | 8192 | Qwen keeps QKV bias |
| `attn_k.bias` | 1024 | 1024 |  |
| `attn_v.bias` | 1024 | 1024 |  |
| `attn_o.weight` (W_O) | 4096 × 2560 | 8192 × 8192 | `(n_heads · head_dim) × d` |
| `ffn_norm.weight` | 2560 | 8192 | `d` |
| `ffn_gate.weight` (W_g) | 2560 × 6912 | 8192 × 29568 | `d × intermediate` |
| `ffn_up.weight` (W_u) | 2560 × 6912 | 8192 × 29568 | `d × intermediate` |
| `ffn_down.weight` (W_d) | 6912 × 2560 | 29568 × 8192 | `intermediate × d` |
| `output_norm.weight` | 2560 | 8192 | `d` |
| `output.weight` (LM head) | **tied** | 152064 × 8192 | (only Qwen2.5-72B has a separate one) |

### 2.2 Parameter count sanity check

For Qwen3-4B:

```
Embeddings (tied):       151 936 · 2560                   = 389  M
Per layer attention:     2 · 2560·4096 + 2·2560·1024      ≈   26 M
  +  bias                4096 + 2·1024                    =   ~ 6 K (negligible)
Per layer FFN:           3 · 2560 · 6912                  ≈   53 M
Per layer norms:         2 · 2560                         ≈    5 K
Per layer total:                                          ≈   79 M
× 36 layers:                                              ≈ 2.84 B
+ final norm:                                             negligible

Total: 389 M (embed) + 2.84 B (layers) ≈ 3.23 B params
Marketing name "4B" — close enough, depending on bias and norm accounting.
```

For Qwen2.5-72B:

```
Embeddings (untied — 2 copies):  2 · 152 064 · 8192     ≈ 2.49 B
Per layer attention:             8192·8192 + 2·8192·1024
                                 + 8192·8192            ≈ 151  M
Per layer FFN:                   3 · 8192 · 29 568      ≈ 727  M
Per layer total:                                        ≈ 878  M
× 80 layers:                                            ≈ 70.2 B
Total:                                                  ≈ 72.7 B params ✓
```

The huge embedding contribution to the 72B model (2.49 B params, ~3.4% of total) is exactly why **not tying** them is a defensible choice at that scale — but it's also why some custom Qwen2.5-72B fine-tunes you'll find on HuggingFace tie them after-the-fact to save 1.24 B params of LM head weight.

---

## 3. Attention Block: GQA in Both, Different Pressure

Both models use **Grouped-Query Attention** with 8 KV heads. What changes is the Q-head count.

### 3.1 Qwen3-4B-Instruct

```
n_heads     = 32
n_kv_heads  = 8
group_size  = 32 / 8 = 4    ← every 4 Q heads share 1 K, 1 V
head_dim    = 128
```

KV cache per token:

```
2 (K and V) · 8 heads · 128 head_dim · 2 bytes (FP16)
 = 4096 bytes per token per layer
 × 36 layers
 = 147 456 bytes per token
 × 4096 ctx = 576 MB
```

### 3.2 Qwen2.5-72B-Instruct

```
n_heads     = 64
n_kv_heads  = 8
group_size  = 64 / 8 = 8    ← every 8 Q heads share 1 K, 1 V
head_dim    = 128
```

KV cache per token:

```
2 · 8 · 128 · 2 = 4096 bytes per token per layer
× 80 layers     = 327 680 bytes per token
× 32 768 ctx    = 10.0 GB
× 131 072 ctx   = 40.0 GB  ← long-context regime, single-stream
```

The 72B's wider Q (64 heads) costs you compute, but its KV per token is only **2.2× larger** than the 4B's because both use 8 KV heads. That's the whole point of GQA: bound the KV cache regardless of Q-head count.

### 3.3 The attention block flow (identical in both models)

```
x[d]
 │
 ├─► RMSNorm(eps=1e-6)
 │      │
 │      ├─► [W_Q + b_Q] → q[n_heads · head_dim]
 │      ├─► [W_K + b_K] → k[n_kv_heads · head_dim]
 │      └─► [W_V + b_V] → v[n_kv_heads · head_dim]
 │
 │     RoPE(q, k)   ← rotary with theta=1e6, NeoX layout
 │
 │     Append k,v to KV cache
 │
 │     attn_out = softmax(q · Kᵀ / √head_dim) · V
 │     (Q has 4× or 8× more heads than K/V — heads in the same group
 │      attend to the same K/V, then concatenate independently)
 │
 │     attn_out → [W_O] → o[d]
 ▼
x + o   (residual)
```

The Qwen-specific detail you'll forget once and regret: **Qwen keeps QKV bias** (`b_Q, b_K, b_V`). Llama and Mistral don't. Runtimes that strip bias on import will silently break Qwen models. Verify your loader.

---

## 4. RoPE — Rotary Position Embedding (the long version)

Position encoding is how the model knows token order. Qwen uses **RoPE** (Rotary Position Embedding) — same family as Llama, Mistral, Phi, Gemma. This section gets long because RoPE is where the subtlest Qwen-specific inference bugs live, and where every long-context technique (YaRN, NTK-aware, Dual-Chunk) plugs in.

### 4.1 Why rotation instead of addition

Older architectures (original Transformer, BERT) added a learned position vector to each token embedding:

```
x'_i = x_i + p_i        # element-wise add a position vector
```

Two problems for inference:

1. The `p_i` vectors are learned per-position — sequences longer than the training context have no defined behavior.
2. Position information mixes into the residual stream and is consumed by every downstream op including FFN — wasting bandwidth on dimensions whose only job is to say "where am I."

RoPE does the opposite: **rotate Q and K only**, in place, by an angle proportional to position. The **2D rotation matrix** is the high school one:

```
R(mθ) = [ cos(mθ)  -sin(mθ) ]
        [ sin(mθ)   cos(mθ) ]
```

The algebraic property that makes this work:

```
⟨ R(mθ)·Q_m , R(nθ)·K_n ⟩  =  ⟨ Q_m , R((n - m)θ)·K_n ⟩
```

The inner product (which is what attention uses) depends only on the **relative** position `(n − m)`. The model never sees absolute coordinates, only relative geometry.

Practical consequences for an inference engineer:

* Position info never enters the FFN — Q and K are the only tensors touched.
* Extending context past training is "just" picking the right rotation angles for unseen positions.
* The **rotated** K is what gets cached, not the raw K. RoPE happens **before** KV insertion.

### 4.2 Where RoPE plugs in

```
hidden_state ──► RMSNorm ──┬─► q_proj ──► Q ──┐
                            ├─► k_proj ──► K ──┼─► RoPE(Q, K, pos)
                            └─► v_proj ──► V   │           │
                                               │           ▼
                                               │   write rotated K and V
                                               │   to KV cache at pos
                                               │
                              attention(Q,     ▼
                                        K_cache, V_cache) → o_proj
```

The RoPE kernel does:

1. Read the per-head Q and K (shape `[n_heads, head_dim]` and `[n_kv_heads, head_dim]`).
2. Read the precomputed `cos[pos]` and `sin[pos]` slices for the current position.
3. Apply the per-pair rotation in registers.
4. Write Q and K back, usually in place.

Look at your `rope_kernel`'s ptxas line from the JLLM build:

```
Used 15 registers, used 0 barriers, 389 bytes cmem[0]
```

That's an embarrassingly parallel signature — one thread per (head × pair) element, no shared memory, no synchronization. The kernel is bandwidth-bound on Q+K traffic, which is tiny compared to the QKV projection's weight read that just preceded it.

### 4.3 Pair layout — NeoX vs original

The rotation is applied to **pairs** of coordinates along the head dimension. Two conventions:

```
"Original" (Llama-1, GPT-NeoX research code): pairs are interleaved
   indices: [0, 1, 2, 3, …, d-2, d-1]
   pair (0,1), (2,3), (4,5), …, (d-2, d-1)

"NeoX-style" (Qwen, Llama-2/3, Mistral, Phi, Gemma): pairs split halves
   indices: [0, 1, …, d/2-1] | [d/2, d/2+1, …, d-1]
   pair (0, d/2), (1, d/2+1), (2, d/2+2), …, (d/2-1, d-1)
```

If your kernel was written for the original layout and you point it at a Qwen model, the math runs, **no error fires**, and the first ~5–10 generated tokens look plausible because position 0–10 angles are tiny and the corruption is small. Past that, the model goes off the rails. This is the canonical **"passes unit tests, fails integration tests"** RoPE bug.

Verify by inspecting which two elements of `head_dim` your kernel multiplies by the same `cos`. If they are adjacent (indices `2i` and `2i+1`), you've got the original layout — wrong for Qwen.

### 4.4 The angle — why `rope_theta = 1 000 000`

Each pair index `i ∈ [0, head_dim/2)` rotates at frequency:

```
θ_i = base ^ (-2i / head_dim)
```

where `base` is `rope_theta` from `config.json`. Qwen uses `base = 1e6`, **not** Llama-1's `10 000`.

| `base` | Slowest pair `i=0` | Fastest pair `i = head_dim/2 - 1`, head_dim=128 |
|---|---|---|
| 10 000 | 1.0 rad/pos | ~1e-4 rad/pos |
| 1 000 000 | 1.0 rad/pos | ~1e-6 rad/pos |

The larger base **spreads the frequencies further apart**. High-frequency pairs (small `i`) rotate fast — they encode fine-grained local order. Low-frequency pairs (large `i`) rotate slowly — they encode coarse "far away" order.

With `base = 1e6`, the slowest pair barely rotates over the entire 32 k training context. This is **the design choice that lets Qwen2.5 and Qwen3 extend to 131 k tokens without retraining** — the slow channels were trained in a regime where they almost don't change, so extending them slightly stays in-distribution.

Llama-1 with `base = 10 000` exhausts the slow channels much earlier — which is why Llama-1 couldn't do long context without surgery.

### 4.5 PyTorch reference (NeoX layout, matches Qwen)

```python
import torch

def precompute_cos_sin(head_dim: int, max_pos: int, base: float,
                       device, dtype=torch.float32):
    """Build cos/sin tables once at startup. Tables are tiny (max_pos x head_dim/2)."""
    inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2,
                                            device=device, dtype=torch.float32)
                               / head_dim))                 # [head_dim/2]
    positions = torch.arange(max_pos, device=device, dtype=torch.float32)
    freqs = torch.outer(positions, inv_freq)                # [max_pos, head_dim/2]
    return freqs.cos().to(dtype), freqs.sin().to(dtype)


def rope_neox(x, cos, sin, position_ids):
    """
    x:            [B, n_heads, seq_len, head_dim]
    cos, sin:     [max_pos, head_dim/2]
    position_ids: [B, seq_len]
    """
    half = x.shape[-1] // 2
    x1 = x[..., :half]      # first half of head_dim
    x2 = x[..., half:]      # second half
    cos_p = cos[position_ids].unsqueeze(1)   # [B, 1, seq_len, head_dim/2]
    sin_p = sin[position_ids].unsqueeze(1)
    rot1 = x1 * cos_p - x2 * sin_p
    rot2 = x2 * cos_p + x1 * sin_p
    return torch.cat([rot1, rot2], dim=-1)


# In the attention block:
q = rope_neox(q, cos, sin, pos_ids)
k = rope_neox(k, cos, sin, pos_ids)
# V is NOT rotated — only Q and K.
```

For inference specifically:

* The `cos/sin` tables are precomputed once for `max_pos = max_position_embeddings` and live in constant or read-only memory. Tiny — at `head_dim=128` and `max_pos=131072` they total ~16 MB combined, FP16.
* During **decode** at position `n`, you index a single row `cos[n], sin[n]` — no broadcast across batch needed, and certainly no recomputation.
* The rotated K is what gets appended to the KV cache. If your runtime caches **pre-rotation** K, every attention step recomputes RoPE on the entire prefix — silent O(N²) blow-up.

### 4.6 Extending context without retraining

When you push past the trained context (32 k for Qwen2.5-72B, 40 k for Qwen3-4B native), the slow frequencies start producing rotations the model never saw during training. Without correction, perplexity explodes past the boundary. Several techniques fix this **at inference time only**, no fine-tuning required:

#### 4.6.1 Position Interpolation (PI)
Scale all position indices by `1/k` so the model effectively sees compressed positions. Works mechanically; loses precision because every pair gets the same scaling — including the fast pairs that were already fine. Used in early Llama-2 long-context tunes; not used in Qwen.

#### 4.6.2 Dynamic NTK-Aware RoPE
At inference, **scale `rope_theta` upward as the sequence grows past training context**:

```
base_effective = base · ( k · seq_len / orig_max - (k - 1) ) ^ (head_dim / (head_dim - 2))
```

The trick: high-frequency pairs (which were trained on many rotations) keep behavior; low-frequency pairs get a stretched effective range. Used in early Qwen and Qwen2 long-context variants. The "dynamic" part means the scaling depends on actual input length, not a fixed factor.

#### 4.6.3 YaRN — the production technique for Qwen2.5 / Qwen3
YaRN refines NTK-aware in two ways:

* **Per-frequency policy.** Fast pairs use plain extrapolation (no scaling). Slow pairs use interpolation (rescale). Mid pairs blend smoothly. The boundaries are controlled by `beta_fast` and `beta_slow` in config.
* **Attention temperature correction.** When you rescale frequencies, the variance of attention logits changes — the softmax gets sharper or flatter. YaRN multiplies attention scores by `1/√t(s)` where `t(s)` depends on the scaling factor. Compensates for the entropy drift.

Both models in this series ship YaRN configuration:

```json
"rope_scaling": {
  "type": "yarn",
  "factor": 4.0,
  "original_max_position_embeddings": 32768,
  "attention_factor": 0.1,
  "beta_fast": 32,
  "beta_slow": 1
}
```

The runtime reads these at startup, adjusts the `cos/sin` precompute, and folds the attention scale into the softmax. **Zero per-token cost** — you get long context for free as long as you wire it correctly.

#### 4.6.4 Dual-Chunk RoPE (for 100 k+)
For the experimental long-context Qwen variants and "Qwen-Long" builds, even YaRN starts to degrade past ~100 k. Dual-Chunk attention splits the computation into:

* **Local chunks** — within a window of N tokens, standard rotated RoPE applies.
* **Global path** — cross-chunk attention uses **rescaled** angles that compress the absolute range, so long-distance relationships stay in-distribution.

This is implemented as a custom attention kernel, not a pure RoPE patch. As of mid-2026, LMDeploy/TurboMind ships it for Qwen-LongContext; mainline vLLM and TRT-LLM rely on YaRN for the production Qwen2.5/Qwen3 lineup.

### 4.7 The `config.json` cheat sheet

The fields you actually look at when wiring up RoPE for inference:

| Field | Purpose | Qwen3-4B-Instruct | Qwen2.5-72B-Instruct |
|---|---|---|---|
| `rope_theta` | Base frequency `θ_base` | 1 000 000 | 1 000 000 |
| `rope_scaling.type` | Scaling strategy | `yarn` | `yarn` |
| `rope_scaling.factor` | Context multiplier | 4–8 | 4 |
| `rope_scaling.original_max_position_embeddings` | Native training context | 32 768 | 32 768 |
| `rope_scaling.beta_fast` / `beta_slow` | YaRN frequency boundaries | 32 / 1 | 32 / 1 |
| `max_position_embeddings` | Max allowed context after scaling | 40 960–262 144 | 131 072 |
| `head_dim` | Determines `head_dim/2` rotated pairs | 128 | 128 |

### 4.8 The four checks every inference runtime must pass

1. **Pair layout is NeoX** — pairs are `(i, i + head_dim/2)`, not `(2i, 2i+1)`.
2. **`rope_scaling` is read and applied** — not just `rope_theta`. If your runtime only honors `rope_theta`, you get a model that runs at 32 k and breaks past that.
3. **`cos/sin` tables cover the full extended range** — precompute for `max_position_embeddings`, not `original_max_position_embeddings`.
4. **Rotation happens before KV insertion** — the cache stores rotated K. If you see `kv_append` *before* `rope` in a trace, the runtime has the order wrong.

A 30-second integration test: feed a ~50 000-token prompt and ask for a 20-token completion. If grammatical and on-topic, YaRN is wired correctly. If it produces gibberish past ~10 generated tokens, RoPE scaling is broken.

### 4.9 RoPE in the GEMV trace

A correctly-instrumented trace for a single decode step on layer 0 shows:

```
[GEMV-GPU #0] type=12 M=4096 K=2560     ← Q projection
[GEMV-GPU #1] type=12 M=1024 K=2560     ← K projection
[GEMV-GPU #2] type=14 M=1024 K=2560     ← V projection
[rope] applied at position N             ← rotates Q and K in place
[kv_append] cached K, V at position N    ← cached K is post-rotation
[attention] flash_attention_decode       ← attends over [0..N]
```

If you can't see RoPE in your trace at all, either the kernel is silent (most are) or it's been folded into the QKV-projection kernel as a fused epilogue (vLLM does this — `qkv_proj_with_rope_kernel`). Fusing RoPE into QKV saves one launch and one Q+K pass through memory — small win, but free if you have the kernel.

---

## 5. FFN: SwiGLU with Asymmetric Sizing

Both models use the **SwiGLU** FFN variant:

```
ffn(x) = down( silu(gate(x)) ⊙ up(x) )
```

In code:

```python
def ffn(x, W_gate, W_up, W_down):
    g = x @ W_gate                  # [d] → [intermediate]
    u = x @ W_up                    # [d] → [intermediate]
    h = torch.nn.functional.silu(g) * u
    return h @ W_down               # [intermediate] → [d]
```

**Three GEMVs per layer**, intermediate dimension is **2.7× d_model** (Qwen3-4B: 2560 → 6912; Qwen2.5-72B: 8192 → 29568). That ratio is fixed by the choice to keep FFN-vs-attention parameter balance close to 2:1.

The FFN is the **larger** of the two blocks by parameter count:

| | Qwen3-4B per layer | Qwen2.5-72B per layer |
|---|---|---|
| Attention params | 26 M | 151 M |
| FFN params       | 53 M | 727 M |
| FFN / Attention  | 2.0× | 4.8× |

Two takeaways:
1. The 72B is **even more FFN-dominated** than the 4B. Optimizing FFN matters more at scale.
2. Quantization of FFN-down hurts quality more than any other tensor (this is empirically true across LLM families). Q6_K or per-tensor-skip is the standard recipe.

---

## 6. Normalization and Residual Structure

Both models use:

* **Pre-norm RMSNorm** (not LayerNorm, not Post-norm).
* `eps = 1e-6`.
* **Weight only**, no bias on the norm.

The fused residual+RMSNorm kernel pattern (which JLLM names `fused_rmsnorm_residual_kernel`) is the standard implementation: read residual stream, accumulate variance, scale, write. Doing this as one kernel rather than three is ~3× faster on bandwidth-bound hardware.

```
y = x · scale / sqrt(mean(x²) + eps) · w
```

There are 2 norms per layer plus a final output norm:

```
Total norms in Qwen3-4B:  2 · 36 + 1 = 73 norm tensors
Total norms in Qwen2.5-72B: 2 · 80 + 1 = 161 norm tensors
```

Norm weights are tiny (~`d` floats each) — store FP32 even when the model is FP16/INT4. The numerical noise floor from FP16 norm weights is non-trivial.

---

## 7. Tokenizer

Both models use the same family of tokenizers — Qwen's BPE built on top of `tiktoken`-style byte-level encoding:

| | Qwen3-4B | Qwen2.5-72B |
|---|---|---|
| `vocab_size` | 151 936 | 152 064 |
| Encoding | BPE, byte-level | BPE, byte-level |
| Multilingual | Yes (heavy CJK + Latin) | Yes |
| Special tokens | `<\|im_start\|>` = 151 644, `<\|im_end\|>` = 151 645, etc. | same IDs |

The vocab is enormous compared to Llama (32 k), which has two consequences:

1. **LM head is huge.** For Qwen3-4B with tied embeddings, you pay it once (in `token_embd`). For Qwen2.5-72B with untied, the final GEMV is `8192 × 152 064 = 1.25 B` parameters — at FP16 that's a single GEMV reading **2.5 GB of weights per token**. On an A100 80GB at ~2 TB/s, that's ~1.25 ms just for the LM head.

2. **Token efficiency.** A typical English sentence is ~30% fewer tokens than Llama's tokenizer encodes it as, ~50% fewer for Chinese. This directly improves perceived tok/s on translated workloads. Your apples-to-apples benchmarks must account for this when comparing across families.

### 7.1 Chat template

```
<|im_start|>system
You are a helpful assistant.<|im_end|>
<|im_start|>user
Hello, this is testing.<|im_end|>
<|im_start|>assistant
<think>

</think>

Hello! It seems like ...<|im_end|>
```

Qwen3 introduces the `<think>…</think>` block — visible CoT that runtime can show or hide. For inference optimization this changes nothing structurally; it's still a token stream. But it does mean a non-trivial fraction of decoded tokens may be inside the `<think>` block — invisible to the end user but consuming the same bandwidth.

---

## 8. Mapping Qwen Tensors to GGUF / safetensors Names

When debugging a runtime trace you need to match names. The mapping:

| HuggingFace safetensors | GGUF | Role |
|---|---|---|
| `model.embed_tokens.weight` | `token_embd.weight` | input embedding |
| `model.layers.N.input_layernorm.weight` | `blk.N.attn_norm.weight` | pre-attention norm |
| `model.layers.N.self_attn.q_proj.weight` | `blk.N.attn_q.weight` | W_Q |
| `model.layers.N.self_attn.q_proj.bias` | `blk.N.attn_q.bias` | b_Q (Qwen has these) |
| `model.layers.N.self_attn.k_proj.weight` | `blk.N.attn_k.weight` | W_K |
| `model.layers.N.self_attn.v_proj.weight` | `blk.N.attn_v.weight` | W_V |
| `model.layers.N.self_attn.o_proj.weight` | `blk.N.attn_output.weight` | W_O |
| `model.layers.N.post_attention_layernorm.weight` | `blk.N.ffn_norm.weight` | pre-FFN norm |
| `model.layers.N.mlp.gate_proj.weight` | `blk.N.ffn_gate.weight` | W_g |
| `model.layers.N.mlp.up_proj.weight` | `blk.N.ffn_up.weight` | W_u |
| `model.layers.N.mlp.down_proj.weight` | `blk.N.ffn_down.weight` | W_d |
| `model.norm.weight` | `output_norm.weight` | final norm |
| `lm_head.weight` | `output.weight` (absent if tied) | LM head |

Counting tensor entries:

* Qwen3-4B (tied): `1 + (2 + 4 + 4 + 4 - 1) · 36 + 1 = 1 + 13 · 36 + 1 = 470`? In practice the count depends on whether biases are stored separately and whether tied LM head appears. The JLLM trace shows **398 tensors** for Qwen3-4B-AWQ — biases on QKV (3·36 = 108 extra) push that up vs Llama-style models.

* Qwen2.5-72B FP16 untied: ~`2 + 13 · 80 + 1 = 1043` tensors, ~145 GB on disk.

---

## 9. Final Comparison

| | Qwen3-4B-Instruct (Q4_K_M) | Qwen2.5-72B-Instruct (FP16) |
|---|---|---|
| On-disk size | ~2.4 GB | ~145 GB |
| Active params per token | 4 B | 72 B |
| Effective bytes/weight read per decode | ~0.6 (Q4_K average) | 2 (FP16) |
| Bytes/token (weights) | 2.4 GB | 145 GB |
| Roofline tok/s on Orin Nano (50 GB/s) | ~20 (theoretical) | n/a (doesn't fit) |
| Roofline tok/s on 8×H100 SXM (~3.35 TB/s each, TP=8 → effective 26.8 TB/s) | n/a | ~185 |
| KV per token (8 KV heads · 128 head_dim · 2 layers·side) | 147 KB / token | 320 KB / token |
| KV at max context | 576 MB (4 k) / 9.4 GB (64 k) | 10 GB (32 k) / 40 GB (131 k) |
| Embeddings tied? | Yes (saves 389 MB) | No (1.24 B LM head params) |
| Where it runs | Jetson Orin Nano 8GB, M2 Mac, iPhone Pro NPUs | 4–8 × A100/H100, 8 × L40S |

The next lecture starts where the difference is biggest — quantization, which only the 4B undergoes.

---

## Hands-On Exercises

1. **Verify the shapes.** Download Qwen3-4B-Instruct from HuggingFace. Open `config.json`, compute every per-tensor shape from the formulas in §2.1, then load with `transformers` and assert each `param.shape` matches. Anything that doesn't match is either a runtime bug or your config-reading bug.

2. **Param-count sanity.** Compute the total parameter count for Qwen2.5-72B-Instruct from `config.json` alone. Compare with `sum(p.numel() for p in model.parameters())` from a real load. They should agree within 0.1%; if not, you've miscounted norms or biases.

3. **Trace ↔ tensor mapping.** Take the JLLM log from the previous lecture (`[GEMV-GPU #0] type=12 M=4096 K=2560`) and prove from `config.json` that this **must** be the Q projection of `blk.0`. Repeat for #1 and #2. Then predict what the next 4 GEMVs in `blk.0` should be (shapes and types).

4. **KV cache calculator.** Build a small Python utility that takes `(n_layers, n_kv_heads, head_dim, kv_dtype, ctx)` and returns KV cache bytes. Use it to plot KV size vs context length for both Qwen3-4B and Qwen2.5-72B from 1 k to 128 k tokens. Mark the GPU-memory budget for Orin Nano, L40S 48GB, A100 80GB, H100 80GB.

5. **RoPE flavor check.** Pick a runtime (llama.cpp, MLC, or your own) and confirm via inspection that its `rope_kernel` uses the NeoX (split) layout for Qwen. The fastest verification: dump the rotation matrix it applies and compare against `cos(theta_i) ± sin(theta_i)` patterns in either order.

6. **YaRN integration test.** Take Qwen3-4B-Instruct and feed it a 50 000-token document with a single distinctive fact buried near token 45 000 ("the password is `puffin-spruce-47`"). Ask the model at the end: "What was the password?" If YaRN is wired correctly the model retrieves it; if not, the answer is gibberish or a refusal. Run this with `--rope-scaling yarn` enabled and disabled to confirm which knob in your runtime matters.

7. **Implement RoPE from scratch.** Copy the §4.5 PyTorch reference. Load Qwen3-4B and replace `transformers`'s built-in RoPE with your implementation. Run a few prompts and confirm bit-identical output to the reference implementation. Then deliberately swap to original-layout pairs (`(2i, 2i+1)` instead of `(i, i+d/2)`) and observe the first ~10 tokens look reasonable before output collapses.

8. **Per-frequency YaRN inspection.** Plot the `cos[pos]` table values for `pos ∈ [0, 32k, 100k, 131k]` and pair indices `i ∈ [0, 16, 32, 48, 63]`. With YaRN applied, fast pairs should look the same as without YaRN; slow pairs should show smooth interpolation. If they don't, YaRN's per-frequency policy isn't being applied.

---

## Key Takeaways

| Takeaway | Why it matters |
|---|---|
| Twelve numbers in `config.json` determine every tensor shape | You can predict every GEMV in a trace without loading the model |
| GQA bounds the KV cache regardless of Q-head count | KV scales with `n_kv_heads · n_layers`, not `n_heads` |
| Qwen keeps QKV biases — Llama/Mistral don't | Loaders that strip bias silently break Qwen |
| RoPE-NeoX layout (pairs are split halves, not interleaved) | Wrong layout = silent corruption past ~10 generated tokens |
| `rope_theta = 1e6` spreads frequencies for long-context headroom | The single config value that makes 131 k context possible without retraining |
| YaRN, not raw `rope_theta`, is what enables 131 k context | Runtimes that only read `rope_theta` silently break long inputs |
| Cache stores rotated K, not raw K | RoPE must run before `kv_append`, or attention is silently wrong |
| Tied embeddings on Qwen3-4B, untied on Qwen2.5-72B | LM head is one of the largest GEMVs in the 72B; cheap in 4B |
| FFN dominates parameters more than attention | Optimizing FFN paths buys more than optimizing attention |
| `<think>` blocks consume the same bandwidth as visible output | Real-world decode budgets must include hidden CoT |

---

## Resources

* **[Qwen3 Technical Report (2026)](https://arxiv.org/abs/2505.09388):** Official architecture and post-training description.
* **[Qwen2.5 Technical Report (2024)](https://arxiv.org/abs/2412.15115):** The 72B paper with all configs.
* **[Hugging Face — Qwen/Qwen3-4B-Instruct](https://huggingface.co/Qwen/Qwen3-4B-Instruct-2507):** Weights, tokenizer, generation config.
* **[Hugging Face — Qwen/Qwen2.5-72B-Instruct](https://huggingface.co/Qwen/Qwen2.5-72B-Instruct):** Weights and `config.json`.
* **[RoFormer: Enhanced Transformer with Rotary Position Embedding (Su et al., 2021)](https://arxiv.org/abs/2104.09864):** The original RoPE paper.
* **[YaRN: Efficient Context Window Extension](https://arxiv.org/abs/2309.00071):** The math behind both models' long-context behavior.
* **["Scaling Laws of RoPE-based Extrapolation" (NTK-aware analysis)](https://arxiv.org/abs/2310.05209):** Why `rope_theta` matters and how dynamic-NTK works.
* **["Dual Chunk Attention" (Qwen-LongContext team, 2024)](https://arxiv.org/abs/2402.17463):** The 100 k+ context technique used in Qwen-Long variants.
* **[Hugging Face Transformers — `modeling_qwen2.py`](https://github.com/huggingface/transformers/blob/main/src/transformers/models/qwen2/modeling_qwen2.py):** Production-grade RoPE-NeoX reference (`apply_rotary_pos_emb`).
* **["GQA: Training Generalized Multi-Query Transformer Models"](https://arxiv.org/abs/2305.13245):** Original GQA paper.
* **[llama.cpp GGUF format spec](https://github.com/ggerganov/ggml/blob/master/docs/gguf.md):** Tensor naming and on-disk layout.
* **[Phase 5 — Edge LLM Inference Internals](../Edge%20LLM%20Inference%20Internals/Lecture-01.md):** The prerequisite roofline lecture.
