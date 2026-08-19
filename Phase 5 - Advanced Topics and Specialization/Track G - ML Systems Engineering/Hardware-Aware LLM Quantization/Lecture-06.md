# Module 06 — Activation Outliers

**Collection:** [Hardware-Aware LLM Quantization](README.md) | **Previous:** [← Module 05](Lecture-05.md) | **Next:** [Module 07 →](Lecture-07.md)

---

Weights are a fixed, inspectable, well-behaved tensor you can spend an hour calibrating. Activations are a **distribution over inputs you have not seen yet**, and they contain structures that are pathological for any fixed-scale quantizer.

This module explains those structures, why they exist (they are not defects — the model built them on purpose), and what actually works. It also justifies the recommendation from [Module 02](Lecture-02.md): at batch 1, keep activations in BF16 and spend your effort elsewhere.

---

## Learning objectives

By the end of this module you should be able to:

1. Contrast weight and activation distributions and explain why the latter are harder.
2. Distinguish **systematic channel outliers** from **rare token spikes**, and name the different remedy each requires.
3. Explain **massive activations** and **attention sinks**, and why removing them breaks the model.
4. State which scaling axes are mathematically legal for a GEMM, and why per-channel activation scaling requires a weight correction.
5. Decide whether W4A4 is worth it for a given operating point.

---

## 1. Two different tensors

```text
   WEIGHTS                                ACTIVATIONS
   ────────────────────────────           ──────────────────────────────
   fixed after training                   change with every input
   roughly Gaussian, zero-mean            heavy-tailed, structured
   dynamic range ~10²                     dynamic range 10³–10⁵
   calibrate once, offline                must be handled ONLINE
   outliers scattered                     outliers in FIXED CHANNELS
   you can spend an hour per tensor       you have microseconds
```

Weight distributions are close to what quantization theory assumes. Activation distributions are not, and the gap is not small — it is the difference between a well-conditioned problem and a badly-conditioned one.

---

## 2. Structure one: systematic channel outliers

In a trained transformer, a **small, consistent set of hidden dimensions carries values 10–100× larger than the rest**, in the same channels, for essentially every token.

```text
   hidden dimension index (d_model = 8192)
   0        1000      2000      3000      4000      5000      6000      7000
   │         │         │         │         │         │         │         │
   ▁▁▁▂▁▁▁▁▁▁█▁▁▁▁▁▂▁▁▁▁▁▁▁▁▁▁▁▁▁▁█▁▁▁▁▁▁▁▁▁▁▂▁▁▁▁▁▁▁▁▁▁▁▁▁█▁▁▁▁▁▁▁▁▁▁▁▁▁
            ▲                     ▲                          ▲
            └── the SAME channels, on nearly EVERY token ────┘
```

The consequence for a per-tensor scale is immediate. If three channels out of 8192 are 50× larger than the rest, a per-tensor absmax scale is set by those three, and the other 8189 channels are compressed into the bottom 2 % of the representable grid. With E2M1's eight magnitudes, that means **almost everything quantizes to zero** — the [Module 02 §4](Lecture-02.md) annihilation, applied to 99.96 % of the tensor.

**Because this structure is consistent, it is fixable.** SmoothQuant ([Module 05](Lecture-05.md)) migrates the per-channel magnitude into the weights, where it can be absorbed. Per-channel activation scaling would also fix it — if it were legal, which brings us to §4.

---

## 3. Structure two: rare token spikes (and why they are load-bearing)

The second structure is different in kind. A **small number of token positions** — very often the first token — produce activations with magnitudes thousands of times the median, concentrated in a couple of dimensions. These are the **massive activations** described in the recent literature, and they are closely tied to **attention sinks**.

```text
   activation magnitude by token position
   ▲
   │ █
   │ █
   │ █                                                    ← position 0 (often BOS):
   │ █                                                       magnitude 1000×+ the rest
   │ █
   │ █ ▁ ▂ ▁ ▁ ▂ ▁ ▁ ▁ ▂ ▁ ▁ ▁ ▁ ▂ ▁ ▁ ▁ ▁ ▁ ▂ ▁ ▁ ▁ ▁
   └──┴─┴─┴─┴─┴─┴─┴─┴─┴─┴─┴─┴─┴─┴─┴─┴─┴─┴─┴─┴─┴─┴─┴─┴──▶ token position
      0 1 2 3 ...
```

**Why the model builds them.** Softmax attention must distribute a total probability mass of exactly 1 across the keys, on every head, at every position — even when a head has nothing it wants to attend to. The model needs a "none of the above" option, so it designates a token (usually the first) as an **attention sink** and dumps surplus mass there. The massive activation is the mechanism that makes that token reliably attractive.

**Two consequences that matter enormously here:**

1. **You cannot clip these away.** They are not noise. Suppress the sink and attention mass gets redistributed onto tokens the head was deliberately ignoring, which corrupts the output. Quantization schemes that aggressively clip activations damage exactly this mechanism, and the damage shows up as incoherence at long context rather than as a perplexity change.

2. **Percentile calibration will not catch them.** They occur on a fraction of a percent of positions. A 99.9th-percentile threshold computed over a calibration set clips the sink. This is a case where the standard recipe is actively wrong.

```text
   channel outliers   →  SYSTEMATIC   →  migrate them (SmoothQuant)         ✅
   token spikes       →  RARE + LOAD-BEARING  →  must be REPRESENTED, not clipped
                                             →  needs per-token dynamic scaling
                                                or keep-in-higher-precision
```

**This is why the two structures need different remedies, and why treating "activation outliers" as one problem fails.**

---

## 4. Which scaling axes are legal

This is the part most treatments skip, and it explains why activation quantization is constrained in a way weight quantization is not.

For `Y = X · W` with `X : [T, K]` and `W : [K, N]`:

```text
   Y[t, n]  =  Σ_k  X[t, k] · W[k, n]
```

**Per-token scaling of X (per row) — LEGAL.**

```text
   X[t, :] = a_t · X̂[t, :]     ⟹     Y[t, n] = a_t · Σ_k X̂[t,k] W[k,n]
```

`a_t` factors cleanly out of the whole output row. You can dequantize afterwards. **This is free**, and it is why per-token dynamic activation scaling is the standard.

**Per-output-channel scaling of W (per column) — LEGAL.**

```text
   W[:, n] = b_n · Ŵ[:, n]     ⟹     Y[t, n] = b_n · Σ_k X[t,k] Ŵ[k,n]
```

Factors out of the output column. Also free.

**Per-input-channel scaling of X (per column, index `k`) — NOT LEGAL alone.**

```text
   X[t, k] = c_k · X̂[t, k]     ⟹     Y[t,n] = Σ_k c_k X̂[t,k] W[k,n]
```

`c_k` sits **inside the summation over `k`**. It does not factor out. You cannot undo it after the GEMM.

```text
   The reduction axis is the one you cannot scale freely.
   ──────────────────────────────────────────────────────────
   And it is exactly the axis where the channel outliers live.
```

The only way to use a per-input-channel factor is to cancel it on the other operand:

```text
   Y = (X · diag(c)⁻¹) · (diag(c) · W)
```

which is **precisely SmoothQuant** ([Module 05 §4.3](Lecture-05.md)). Now you can see it is not a heuristic — it is the unique legal construction for putting a per-input-channel factor into the computation.

So the toolkit is:

| Structure | Legal remedy |
|---|---|
| Channel outliers (reduction axis) | SmoothQuant-style migration into W, or fine-grained blocks |
| Token spikes (row axis) | per-token dynamic scaling — free and effective |
| Both | per-token scaling **and** migration, plus small blocks |

---

## 5. Why block scaling helps, and how much

NVFP4's 16-element blocks provide a third defence that needs no algebra at all: **an outlier only contaminates its own block.**

```text
   PER-TENSOR scale, one 50× outlier in 8192 values:
   ├──────────────────────── all 8192 values share one window ─────────────────────┤
        → 8191 values crushed into the bottom 2 % of the grid  → mass annihilation

   BLOCK-16 scale, same outlier:
   ├──16──┤├──16──┤├──16──┤ ... ├──16──┤
                    ▲
                    └─ only THESE 16 values are affected;  16/8192 = 0.2 % of the tensor
```

The damage is bounded by the block size, not by the tensor size — a **512× reduction in blast radius** for `d_model = 8192`. This is the deeper reason fine-grained block formats displaced per-tensor INT8: not the bit-width, the **blast radius**.

It is also why the NVFP4-vs-MXFP4 block-size difference (16 vs 32) matters more for activations than for weights. Activations are where the outliers are.

---

## 6. Is W4A4 worth it? — the operating-point answer

Everything above is the cost side. Here is the benefit side, from [Module 01](Lecture-01.md)'s roofline.

**At batch 1:**

```text
   weight traffic per token     ≈  15.88 GB
   activation traffic per token ≈  a few hundred KB
                                   ────────────────
   activation share of B_token  ≈  0.01 %
```

Quantizing activations to FP4 reduces `B_token` by ~0.005 %. The compute benefit (2× tensor-core throughput over FP8) applies to a workload sitting **263× below the ridge point**. So:

```text
   W4A4 benefit at batch 1  ≈  0
   W4A4 risk    at batch 1  =  every structure in this module
```

> **Verdict for single-user decode on RTX 5090: do not quantize activations.** Use W4A16 (NVFP4 weights, BF16 activations). This is not caution — it is the roofline.

**When W4A4 does pay:**

| Condition | Why |
|---|---|
| Large-batch serving (B ≳ 100) | approaching the ridge; compute starts to bind |
| Long-context prefill | prefill is compute-bound by construction |
| Fused kernels where activations are re-read | activation traffic stops being negligible |
| Memory-capacity pressure from activation buffers | large batch × long sequence |

Note that **speculative decoding raises your effective batch** to `K+1` per verification pass ([Module 10](Lecture-10.md)). At `K = 3` that is batch 4 — still 60× below the ridge. Speculation does not change this verdict.

---

## 7. Measure it yourself

Do not take any of this on faith for your model. The diagnostic is thirty lines:

```python
import torch
from collections import defaultdict

stats = defaultdict(list)

def make_hook(name):
    def hook(_mod, inp, _out):
        x = inp[0].detach().float()                  # [batch, seq, hidden]
        stats[name].append({
            "per_channel_absmax": x.abs().amax(dim=(0, 1)).cpu(),   # [hidden]
            "per_token_absmax":   x.abs().amax(dim=-1).flatten().cpu(),
            "median":             x.abs().median().item(),
        })
    return hook

for name, mod in model.named_modules():
    if isinstance(mod, torch.nn.Linear):
        mod.register_forward_hook(make_hook(name))

# ... run 32 calibration sequences ...

for name, records in stats.items():
    ch  = torch.stack([r["per_channel_absmax"] for r in records]).amax(0)
    tok = torch.cat([r["per_token_absmax"] for r in records])
    med = sum(r["median"] for r in records) / len(records)

    print(f"{name:50s}  "
          f"channel_ratio={ch.max()/ch.median():7.1f}  "   # >10 → channel outliers
          f"token_ratio={tok.max()/tok.median():8.1f}  "   # >100 → massive activations
          f"outlier_channels={(ch > 10*ch.median()).sum().item():4d}")
```

Read it like this:

| Signal | Threshold | Diagnosis | Remedy |
|---|---|---|---|
| `channel_ratio` | > 10 | systematic channel outliers | SmoothQuant / smaller blocks |
| `token_ratio` | > 100 | massive activations / sinks | per-token dynamic scaling; **never clip** |
| `outlier_channels` | small (1–10) | classic transformer structure | expected; not a bug |
| both low | — | this layer is easy | quantize it freely |

Run this **before** choosing a method. It is the diagnosis that [Module 05 §4](Lecture-05.md)'s selection tree requires, and it takes ten minutes.

---

## Checkpoint

You should now be able to:

1. Name the two activation-outlier structures and give the different remedy each needs.
2. Explain why attention sinks exist and why clipping them corrupts long-context behavior.
3. Prove which GEMM scaling axes are legal, and derive SmoothQuant's form from the illegal one.
4. Quantify the blast-radius advantage of block-16 scaling over per-tensor scaling.
5. Justify W4A16 over W4A4 at batch 1 from the roofline, and name the conditions that flip the verdict.
6. Read a channel/token ratio table and pick a method from it.

---

## Ship it

Produce an **activation profile** for one model: the per-layer table from §7, a histogram of per-channel absmax for the worst layer, a plot of per-token absmax versus position (showing the sink at position 0), and a one-paragraph diagnosis naming which structures your model has and which remedy you selected.

If your `token_ratio` exceeds 100 and your calibration used a 99.9th percentile, **you have already found a bug** — and that is the artifact.

---

## Current as of

* **Timeless:** the two outlier structures, the legal-axis proof, the blast-radius argument, the batch-1 W4A4 verdict.
* **2026 understanding:** massive activations and their link to attention sinks are established results (StreamingLLM's sink observation; the massive-activations line of work showing a handful of dimensions acting as learned attention biases). Treat "do not clip the sink" as settled practice.
* **Refresh surface:** whether runtimes expose per-token dynamic activation scaling for NVFP4 on `sm_120`. If they do not, W4A4 is off the table for you regardless of the analysis here.

---

**Next:** [Module 07 — Layer Sensitivity →](Lecture-07.md)
