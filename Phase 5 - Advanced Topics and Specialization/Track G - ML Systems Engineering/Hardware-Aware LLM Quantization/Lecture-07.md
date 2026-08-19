# Module 07 — Layer Sensitivity: Why Q/K Break While O and MLP Survive

**Collection:** [Hardware-Aware LLM Quantization](README.md) | **Previous:** [← Module 06](Lecture-06.md) | **Next:** [Module 08 →](Lecture-08.md)

---

Every practitioner eventually discovers the same empirical rule: **quantize the MLP and output projection freely; touch Q and K at your peril.** It is usually passed on as folklore, sometimes with a hand-wave about activation outliers.

The hand-wave is wrong, and knowing why it is wrong is worth real throughput. The mechanism is not in the *distribution* of Q/K values — it is in the **operator that consumes them**. Quantization error into a linear operator averages down. Quantization error into a softmax gets **exponentiated**.

This module derives that, connects it to the measured case-study result, and gives you a protocol for measuring sensitivity directly instead of guessing it from statistics.

---

## Learning objectives

By the end of this module you should be able to:

1. Derive why per-element quantization error *averages down* through a linear projection but *amplifies* through attention.
2. Quantify the attention-weight distortion produced by a given relative error on Q/K.
3. Explain why GQA makes K the worst traffic-to-damage ratio in the entire model.
4. Explain why **sensitivity is not predicted by outlier magnitude**, and what that implies for method selection.
5. Run a leave-one-out sensitivity sweep and turn it into a precision-allocation table.

---

## 1. The two fates of a quantization error

Both paths start identically: a weight tensor is quantized, producing per-element relative error `η ≈ 0.10` for NVFP4 ([Module 02 §5](Lecture-02.md)).

### Path A — through a linear projection (O, MLP, V)

The output is a dot product of length `K`. Errors are approximately zero-mean and independent, so they add in quadrature while the signal adds linearly:

```text
   relative output error  ≈  η / √K
```

| Layer | `K` | Output error from η = 0.10 |
|---|---:|---:|
| O projection | 8192 | **0.11 %** |
| MLP down projection | 13,870 | **0.09 %** |

Then the result is **added into the residual stream**, which by mid-network has accumulated a large norm. An additive perturbation of 0.1 % on one contribution to a large running sum is diluted further still.

```text
   x  ←  x  +  f(x) + δ            δ/|x| ≪ 0.1 %
         └── large ──┘  └ tiny ┘

   LINEAR · ADDITIVE · DILUTED BY THE RESIDUAL · AVERAGES OVER K TERMS
```

**This is why MLP and O projections tolerate 4-bit quantization well.** It is not that their weights are special. It is that four separate mechanisms all work in your favour.

### Path B — through attention scores (Q, K)

Now the same error enters a dot product that is then **exponentiated**.

```text
   s_i  =  (q · k_i) / √d_head              attention logit
   p    =  softmax(s)                       attention weights
```

Perturb `q → q + δ` and `k_i → k_i + ε_i`:

```text
   Δs_i  =  (δ · k_i  +  q · ε_i) / √d_head
```

Here is the critical step. For `q, k` with iid components, both `q·k` and `δ·k` scale as `σ²√d`, so the *relative* logit error is `≈ η√2` — **the √K averaging benefit does not appear**, because the error is relative to a quantity that is itself a dot product of the same length. The averaging that saved Path A cancels out.

So the **absolute** logit error is:

```text
   Δs  ≈  η · √2 · |s|
```

And softmax turns absolute logit differences into **multiplicative** weight ratios:

```text
   p_i / p_j  =  exp(s_i − s_j)      ⟹      a shift Δs changes the ratio by  exp(Δs)
```

| η | typical \|logit\| | Δs | **attention weight distorted by** |
|---:|---:|---:|---:|
| 0.05 | 5 | 0.35 | **×1.42** |
| 0.05 | 10 | 0.71 | **×2.03** |
| 0.05 | 20 | 1.41 | **×4.11** |
| 0.10 | 10 | 1.41 | **×4.11** |
| 0.10 | 20 | 2.83 | **×16.9** |

```text
   PATH A  (O, MLP)          PATH B  (Q, K)
   ─────────────────         ─────────────────────────
   error / √K                error × exp(·)
   0.10 → 0.001              0.10 → attention mass moves by 2–17×
   AVERAGES DOWN             AMPLIFIES
```

**That is the whole answer.** Three orders of magnitude separate the two fates, and the cause is the operator, not the data.

Two corollaries worth internalizing:

* **Sensitivity grows with logit magnitude**, which grows with `d_head` and over the course of training. Bigger, better-trained models are *more* Q/K-sensitive, not less.
* **V is on Path A, not Path B.** V is consumed by a weighted sum, not a softmax. `V` quantizes like `O` — freely. When people say "attention is sensitive" they mean Q and K specifically; lumping V in with them costs you traffic for no reason.

---

## 2. RoPE turns magnitude error into phase error

Rotary position embedding rotates `(q_{2j}, q_{2j+1})` pairs by angle `m·θ_j` at position `m`. The attention logit becomes a sum over pairs of

```text
   r_q,j · r_k,j · cos( (m − n)·θ_j  +  φ_j )
```

where `r` are pair magnitudes and `φ` the relative phase. Quantization perturbs the Cartesian components, which perturbs **both** the magnitude and the **angle** of each pair:

```text
   (x, y) ──quantize──▶ (x + δ, y)          angular error  Δφ ≈ δ / r
                                             ▲
                    small-magnitude pairs suffer LARGE angular error
```

Attention reads the phase. So NVFP4's per-block scaling interacts with RoPE in a specific way: a 16-element block spans 8 rotary pairs, and the block scale is set by the largest of them — so low-magnitude pairs inside a high-magnitude block get both coarse magnitude resolution *and* large angular error.

This yields a **testable prediction**:

> Q/K quantization damage should **grow with context length**, because the `(m−n)·θ_j` term makes attention more phase-sensitive at large relative positions.

Do not take that on faith — §5's protocol includes the sweep that checks it. If your model is destined for 262 K context ([Module 09](Lecture-09.md)), a Q/K sensitivity measured at 4 K is not the number that governs deployment.

---

## 3. GQA makes K uniquely bad

Now add the traffic side. For the case-study geometry (`d_model = 8192`, 48 layers, 8 KV heads × 128, `d_ff ≈ 13,870`):

| Tensor | Params/layer | Share of body | Traffic | **Share of `B_token`** |
|---|---:|---:|---:|---:|
| MLP (3 mats) | 340.9 M | 69.3 % | 9.20 GB | **57.96 %** |
| Q | 67.1 M | 13.6 % | 1.81 GB | **11.41 %** |
| O | 67.1 M | 13.6 % | 1.81 GB | **11.41 %** |
| **K + V** | **16.8 M** | **3.4 %** | **0.45 GB** | **2.85 %** |

With grouped-query attention, `K` and `V` are shared across `G = n_q / n_kv = 8` query heads. So:

```text
   K is the SMALLEST tensor group in the model  (2.85 % of B_token, and V is half of that)
   K feeds 8 query heads                        (one error → 8 corrupted attention distributions)
   K enters via Path B                          (exponential amplification)
```

```text
                 traffic saved          behavioral damage
   MLP    ████████████████████████       ▏
   O      ████▌                          ▏
   Q      ████▌                          ████████
   K      ▊                              ████████████████████
   V      ▊                              ▏
```

**K has the worst reward/risk ratio of any tensor in the model, by a wide margin.** Quantizing K to 4 bits buys you under 1.5 % of your byte budget and pays for it with amplified, GQA-multiplied attention distortion. This is not a close call.

---

## 4. The measured case study

The prediction meets the data:

```text
   MLP + O quantized          147.87 tok/s     acceptance 2.792
   MLP + O + QKV quantized    150.73 tok/s     acceptance 2.546
                              ──────────       ────────────────
                              +1.9 % speed     −8.8 % acceptance
```

Both numbers match theory:

* **Speed: +1.9 %.** Q + K + V together are 14.3 % of `B_token`; converting them from an already-compressed baseline yields a small single-digit gain. The table in §3 bounds the possible win, and the measurement lands inside it.
* **Behavior: −8.8 % acceptance.** Path B amplification, multiplied by GQA sharing. Acceptance length is a direct read on how much the target distribution moved ([Module 08](Lecture-08.md)) — and it moved a lot.

**The trade is 1.9 % throughput for 8.8 % acceptance, and it is a bad one.** Worse than it looks, in fact: [Module 10](Lecture-10.md) shows that acceptance feeds back into throughput, so part of that 1.9 % is given straight back.

The correct configuration keeps **MLP + O + V** in NVFP4 and leaves **Q + K** wide. Cost: ~11 % of `B_token` left uncompressed. Benefit: the entire acceptance regression avoided.

---

## 5. Sensitivity ≠ outlier magnitude

Here is the finding that resolves the confusion, and it is the most important practical point in the module. **The Q/K degradation is not explained by activation-outlier statistics.** You can profile Q/K with [Module 06's](Lecture-06.md) tooling and find unremarkable channel and token ratios — and they will still be the most damaging tensors to quantize.

The reason is that these measure different things:

```text
   outlier statistics    ──▶  predict QUANTIZATION ERROR      (how wrong the tensor becomes)
   sensitivity           ──▶  error  ×  DOWNSTREAM AMPLIFICATION
                                        └── the operator's doing, invisible to any
                                            statistic computed on the tensor itself
```

```text
   sensitivity_i  =  quantization_error_i   ×   amplification_i
                     └── measurable from ──┘    └── a property of the OPERATOR,
                         the tensor alone            not of the tensor ──┘
```

Q/K sit at modest error and enormous amplification. Embeddings sit at high error and near-zero amplification (a gather feeds a residual stream). **Neither is predicted by looking at the tensor.**

> **The practical rule: never infer sensitivity from weight or activation statistics. Measure it end-to-end.** Statistics tell you which *method* to use ([Module 05](Lecture-05.md)); only an ablation tells you which *tensors* to quantize.

### The measurement protocol

```python
# Leave-one-out: quantize everything EXCEPT the group under test.
# The DELTA against all-quantized is that group's contribution to the damage.
GROUPS = ["mlp", "o_proj", "q_proj", "k_proj", "v_proj", "lm_head", "embed"]

baseline_all = evaluate(quantize(model, groups=GROUPS))     # everything quantized
results = {}
for g in GROUPS:
    held_out = [x for x in GROUPS if x != g]
    m = quantize(model, groups=held_out)
    results[g] = {
        "kl_vs_fp16":   mean_kl(m, reference),          # Module 08
        "acceptance":   acceptance_length(m, drafter),  # Module 08 / 10
        "traffic_saved": traffic_of(g),                 # Module 04
        "recovery":     baseline_all.kl - mean_kl(m, reference),
    }
```

Then rank by the ratio that actually matters:

```text
                       traffic_saved_i
   value_i  =  ────────────────────────────
                 KL_increase_i  +  ϵ
```

A representative outcome — **your model's numbers will differ, which is the point of measuring**:

| Group | Traffic saved | ΔKL | Δacceptance | Value | Verdict |
|---|---:|---:|---:|---:|---|
| MLP | 57.96 % | low | small | **highest** | quantize |
| O | 11.41 % | low | small | high | quantize |
| V | ~1.4 % | low | small | moderate | quantize |
| lm_head | 16.0 % | low–moderate | small | high | quantize (FP8 first) |
| Q | 11.41 % | **high** | **large** | low | **keep wide** |
| K | ~1.4 % | **highest** | **largest** | **lowest** | **keep wide** |
| Embeddings | ~0 % | low | none | **zero** | pointless either way |

That table is the direct input to [Module 11](Lecture-11.md)'s allocation solver.

---

## Checkpoint

You should now be able to:

1. Derive `η/√K` for Path A and `exp(η√2·|s|)` for Path B, and explain why the averaging benefit vanishes in the second.
2. Compute the attention-weight distortion for a given η and logit magnitude.
3. Explain why V quantizes like O rather than like K.
4. Explain why GQA makes K the worst reward/risk tensor in the model.
5. State why outlier statistics do not predict sensitivity, and what they *are* good for.
6. Design a leave-one-out sweep and rank groups by traffic-per-unit-KL.

---

## Ship it

Produce a **sensitivity table** for your model: every tensor group, its traffic share from [Module 04's](Lecture-04.md) ledger, its ΔKL and Δacceptance from a leave-one-out sweep, and the resulting value ratio. Add the context-length sweep from §2 — measure Q/K sensitivity at 4 K and at your maximum context and report whether the predicted growth appears.

Then state your allocation and, more importantly, **the two groups you decided not to quantize and why.**

---

## Current as of

* **Timeless:** the Path A / Path B derivation, the softmax amplification bound, the GQA sharing argument, sensitivity ≠ outlier magnitude.
* **Case-study pins:** the 147.87/2.792 vs 150.73/2.546 comparison; the per-tensor traffic shares are derived from the [Module 04](Lecture-04.md) reconstruction with an assumed 48-layer geometry — recompute for your own config rather than reusing the percentages.
* **Open/testable:** the prediction that Q/K sensitivity grows with context length follows from the RoPE phase argument and should be verified per model, not assumed.

---

**Next:** [Module 08 — Behavior Preservation →](Lecture-08.md)
