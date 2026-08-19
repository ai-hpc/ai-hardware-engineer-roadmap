# Module 09 — KV Cache & Long Context

**Collection:** [Hardware-Aware LLM Quantization](README.md) | **Previous:** [← Module 08](Lecture-08.md) | **Next:** [Module 10 →](Lecture-10.md)

---

Every ledger so far has treated the KV cache as a footnote. At 4 K context it is one. At 262 K it is **co-dominant with the entire model's weights**, and the correct optimization strategy inverts.

This module computes the inversion point exactly, and shows that a 262 K context target on a 32 GB card is not a quantization problem you can solve by compressing weights harder — it is an architecture-and-KV-format problem with a hard feasibility boundary.

---

## Learning objectives

By the end of this module you should be able to:

1. Compute KV cache bytes per token from a model's GQA geometry.
2. Compute the **crossover context length** where KV traffic equals weight traffic.
3. Predict tok/s as a function of context length.
4. Determine whether a given (model, context, VRAM) triple is **feasible at all**, before running anything.
5. Explain why KV quantization is a different problem from weight quantization — online, per-head, and latency-critical.

---

## 1. KV cache arithmetic

The cache stores one K and one V vector per **KV head** (not per query head — that is what GQA buys), per layer, per token:

```text
   KV_bytes_per_token  =  2  ×  n_layers  ×  n_kv_heads  ×  head_dim  ×  bytes_per_element
                          │                  │
                     K and V          GQA groups, NOT query heads
```

For the case-study geometry (48 layers, `head_dim` 128 — with `n_kv_heads` shown both ways since the published inventory does not pin it):

| `n_kv` | Elements/token | BF16 | FP8 | NVFP4 |
|---:|---:|---:|---:|---:|
| 8 | 98,304 | **192 KiB** | 96 KiB | 54 KiB |
| 4 | 49,152 | 96 KiB | **48 KiB** | 27 KiB |

Two things follow immediately, and both are structural:

* **KV cost is linear in `n_kv_heads`.** Halving KV heads halves both capacity and traffic — a bigger lever than any KV quantization format. This is why GQA (and MLA) exist, and why architecture choices dominate format choices here.
* **KV cost is independent of `d_model` and `d_ff`.** A wider MLP costs weights, not cache. Weight optimization and KV optimization are **decoupled problems**.

---

## 2. The crossover: when KV traffic overtakes the weights

At every decode step, attention reads the **entire** cache for all previous tokens. So KV traffic grows linearly with context while weight traffic stays flat:

```text
   B_token(L)  =  W  +  L × KV_bytes_per_token
                  │      └──────────────────┘
             15.88 GB      grows with context
```

Setting the two terms equal gives the crossover:

```text
   L_crossover  =  W / KV_bytes_per_token
```

| `n_kv` | KV format | KV/token | **Crossover** |
|---:|---|---:|---:|
| 8 | BF16 | 192 KiB | **80,770 tokens** |
| 8 | FP8 | 96 KiB | 161,540 |
| 8 | NVFP4 | 54 KiB | 287,182 |
| 4 | BF16 | 96 KiB | 161,540 |
| 4 | FP8 | 48 KiB | 323,079 |
| 4 | NVFP4 | 27 KiB | 574,363 |

```text
   traffic
     ▲
     │                                          ╱ KV (grows with L)
     │                                       ╱
     │                                    ╱
  W ─┼─────────────────────────────────╳──────────────  weights (flat)
     │                              ╱   │
     │                           ╱      └── crossover
     │                        ╱
     └──────────────────────────────────────────────▶  context length L

   L < crossover  :  a WEIGHT-dominated problem  →  Modules 04–07 apply
   L > crossover  :  a KV-dominated problem      →  weight quantization stops helping
```

> **With BF16 KV and 8 KV heads, the crossover is ~81 K tokens.** Past that point, further weight quantization is attacking the smaller of the two terms. If you serve at 262 K, you have been on the wrong side of this line for most of your context window.

---

## 3. Throughput versus context

Predicted decode throughput at the measured `BW_eff = 1296 GB/s`, with 4 KV heads and FP8 KV:

| Context | KV traffic | `B_token` | **tok/s** | vs short context |
|---:|---:|---:|---:|---:|
| 0 | 0.00 GB | 15.88 GB | **81.6** | — |
| 4 K | 0.20 GB | 16.08 GB | 80.6 | −1 % |
| 32 K | 1.61 GB | 17.49 GB | 74.1 | −9 % |
| 128 K | 6.44 GB | 22.32 GB | 58.1 | −29 % |
| **262 K** | **12.88 GB** | **28.76 GB** | **45.1** | **−45 %** |

**Decode throughput nearly halves across the context window** — even with FP8 KV and only 4 KV heads, which is already an aggressive configuration.

This has a direct consequence for benchmarking honesty: **a tok/s number without a context length is meaningless.** A benchmark run at 4 K and deployed at 262 K overstates real throughput by ~80 %. [Module 12](Lecture-12.md) treats context as a mandatory reported variable for exactly this reason.

---

## 4. Feasibility: does 262 K fit at all?

Capacity, not traffic, is the harder constraint. The budget:

```text
   32 GB card                    =  29.80 GiB
   − weights                      −  18.80 GiB
   − workspace / activations      −   1.00 GiB  (approx.)
   ────────────────────────────────────────────
   available for KV               ≈  10.00 GiB
```

Now the 262 144-token cache under each configuration:

| `n_kv` | Format | 262 K cache | Fits in 10 GiB? |
|---:|---|---:|:---:|
| 8 | BF16 | 48.00 GiB | ✗ (4.8× over) |
| 8 | FP8 | 24.00 GiB | ✗ (2.4× over) |
| 8 | NVFP4 | 13.50 GiB | ✗ (1.35× over) |
| 4 | BF16 | 24.00 GiB | ✗ |
| 4 | FP8 | 12.00 GiB | ✗ (marginal) |
| **4** | **NVFP4** | **6.75 GiB** | **✓** |

```text
   262 K context on a 32 GB card with 18.8 GiB of weights requires
   BOTH  4 KV heads  AND  4-bit KV.  Every other combination overflows.
```

And notice what does **not** help: quantizing the vision tower (−0.86 GiB) or the embeddings (−1.18 GiB) buys VRAM but not nearly enough to rescue an 8-KV-head BF16 configuration, which is 38 GiB over budget.

**This is where the "capacity optimization ≠ bandwidth optimization" distinction from [Module 01](Lecture-01.md) finally pays off in the other direction.** Evicting the vision tower and quantizing embeddings gained you *zero* tok/s — but at 262 K they are 2.04 GiB of headroom, and headroom is exactly what is scarce. **The same change is worthless for one objective and valuable for the other.** That is the whole reason the framework scores four factors instead of one.

Practical order of operations at long context:

```text
   1. reduce n_kv_heads          ← architecture; biggest lever, but requires the model to have it
   2. quantize KV to FP8/FP4     ← format; 2–3.6× capacity AND traffic
   3. evict Class B/C tensors    ← vision tower, unused adapters: pure capacity
   4. quantize embeddings        ← pure capacity
   5. quantize weights harder    ← LAST; you are attacking the smaller term
```

Note that the list is **exactly inverted** from the short-context priority order in [Module 01 §8](Lecture-01.md).

---

## 5. KV quantization is a different engineering problem

Weight quantization is offline, one-shot, and you can spend an hour per tensor. KV quantization is none of those things:

```text
   WEIGHTS                          KV CACHE
   ──────────────────────────       ───────────────────────────────────
   quantized once, offline          quantized ONLINE, every token
   full calibration available       no future data — must be causal
   error is static                  error accumulates over the context
   cost amortized to zero           cost is IN the decode critical path
```

Four consequences that shape any workable design:

**1. Scaling must be per-head and per-token.** These are the legal axes from [Module 06 §4](Lecture-06.md) — a per-token scale on the K vector factors out of the attention logit, so it is free. A per-channel scale across `head_dim` does not.

**2. The quantizer must be cheap.** It runs on the decode critical path. An MSE grid search ([Module 05](Lecture-05.md)) is out of the question; absmax or a running percentile is what you can afford.

**3. Errors persist.** A weight error is the same on every token. A KV error is written once and then **re-read for every subsequent token in the sequence** — an early-token quantization error influences all 262 K downstream steps. This argues for keeping the first few tokens' KV in higher precision.

**4. Which connects directly to attention sinks.** [Module 06 §3](Lecture-06.md) showed the first token carries massive activations and absorbs surplus attention mass. It is also the entry that gets re-read most often. Both arguments point the same way:

```text
   keep the first N tokens' KV in BF16/FP8  (N ≈ 4–128, cheap: 128 tokens × 48 KiB = 6 MB)
   quantize the rest to FP4

   → protects the sink mechanism
   → protects the most-re-read entries
   → costs ~0.06 % of the cache
```

This "sink-preserving KV quantization" pattern is common in current long-context systems, and both halves of its justification are things you derived in earlier modules.

---

## 6. The long-context ledger

Extend [Module 04's](Lecture-04.md) ledger with the context term:

```python
def ledger_at_context(B_token_weights_GB, n_layers, n_kv, head_dim,
                      kv_bytes_per_elem, context_len,
                      vram_GB=32.0, weights_GiB=18.80, workspace_GiB=1.0,
                      bw_eff_GBs=1296):
    GiB = 1024**3
    kv_per_token = 2 * n_layers * n_kv * head_dim * kv_bytes_per_elem   # bytes
    kv_traffic   = context_len * kv_per_token / 1e9                     # GB per decode step
    kv_capacity  = context_len * kv_per_token / GiB                     # GiB resident

    b_token   = B_token_weights_GB + kv_traffic
    available = vram_GB * 1e9 / GiB - weights_GiB - workspace_GiB

    return {
        "kv_per_token_KiB": kv_per_token / 1024,
        "crossover_tokens": B_token_weights_GB * 1e9 / kv_per_token,
        "B_token_GB":       b_token,
        "predicted_tps":    bw_eff_GBs / b_token,
        "kv_capacity_GiB":  kv_capacity,
        "fits":             kv_capacity <= available,
        "headroom_GiB":     available - kv_capacity,
    }
```

Run it across your full context range **before** choosing a KV format. It answers the feasibility question in microseconds, and infeasible configurations are extremely common at 100 K+ on consumer cards.

---

## Checkpoint

You should now be able to:

1. Compute KV bytes/token from GQA geometry and explain why `d_model` does not appear.
2. Compute the crossover context length and interpret which side of it you are on.
3. Predict the tok/s decay curve across a context range.
4. Determine feasibility of a (model, context, VRAM) triple before running anything.
5. Give both reasons for keeping sink-token KV in higher precision.
6. Explain why the long-context optimization order is the inverse of the short-context one.

---

## Ship it

Produce a **context sweep** for your deployment: KV bytes/token for your geometry in each format; the crossover length for each; a tok/s-vs-context table from 0 to your maximum; a feasibility table with headroom; and a written statement of which side of the crossover your **p50 and p99 serving context lengths** fall on.

If p99 is past the crossover, your next optimization is KV, not weights — regardless of what the rest of this course made you want to do.

---

## Current as of

* **Timeless:** KV arithmetic, the crossover derivation, the capacity/traffic distinction, per-token scaling legality, the sink-preservation argument.
* **Case-study pins:** `W = 15.88 GB`, `BW_eff = 1296 GB/s`, 48 layers / `head_dim` 128, 32 GB VRAM. `n_kv_heads` is **not** pinned by the published inventory — both 8 and 4 are tabulated; substitute your model's actual value.
* **Refresh surface:** runtime support for FP4 KV cache is less mature than for FP8. Verify what your serving stack actually implements before planning around 4-bit KV; an unsupported format is an infeasible plan, not an aggressive one.

---

**Next:** [Module 10 — Speculative Decoding →](Lecture-10.md)
