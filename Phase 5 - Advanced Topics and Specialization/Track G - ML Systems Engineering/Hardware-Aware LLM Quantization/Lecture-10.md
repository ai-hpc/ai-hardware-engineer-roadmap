# Module 10 — Speculative Decoding and the Acceptance Tax

**Collection:** [Hardware-Aware LLM Quantization](README.md) | **Previous:** [← Module 09](Lecture-09.md) | **Next:** [Module 11 →](Lecture-11.md)

---

Speculative decoding is the only technique in this course that changes the throughput equation itself rather than one of its terms. Everything else reduces `B_token`. Speculation emits **multiple tokens per weight-read**, which multiplies the ceiling.

It also creates a feedback loop that quietly destroys most quantization wins, and this module's central result quantifies it: on the case-study model, **quantizing QKV delivered an 11.8 % byte win and realized 1.9 % — the acceptance tax consumed 84 % of it.**

---

## Learning objectives

By the end of this module you should be able to:

1. State the rejection-sampling rule and explain why speculative decoding is **lossless**.
2. Derive the speedup model `S = τ / (1 + K·c)` and choose an optimal draft depth.
3. Explain how quantizing the **target** and the **drafter** each affect acceptance, differently.
4. Compute the **acceptance tax** on a quantization change and decide whether it is worth shipping.
5. Explain why speculation changes which optimization is correct.

---

## 1. The mechanism, and why it is lossless

```text
   1. DRAFT     cheap model q proposes K tokens autoregressively
   2. VERIFY    target model p scores all K+1 positions in ONE forward pass
   3. ACCEPT    walk left to right; accept token x with probability min(1, p(x)/q(x))
   4. ON REJECT sample a correction from the residual  (p(x) − q(x))⁺ / Σ(p − q)⁺
   5. EMIT      the accepted prefix + one correction token
```

Step 4 is what makes the whole thing free of quality cost:

```text
   ┌──────────────────────────────────────────────────────────────────┐
   │  The output distribution is EXACTLY p — the target's own          │
   │  distribution — regardless of how bad the drafter q is.           │
   │  A bad drafter costs SPEED (low acceptance), never QUALITY.       │
   └──────────────────────────────────────────────────────────────────┘
```

That guarantee is why speculation is safe to deploy and why it can be used as a *measurement instrument* ([Module 08](Lecture-08.md)) — the thing being measured is not perturbed by measuring it.

The economics work because **verifying K+1 tokens costs almost the same as verifying one.** Decode is memory-bound ([Module 01](Lecture-01.md)); the weights are read once regardless of how many positions ride along. K+1 positions is a batch of K+1 — still ~60× below the ridge point.

### Draft sources

| Source | How it works | Typical α |
|---|---|---|
| **Small draft model** | separate 0.5–1 B model | moderate; tokenizer must match |
| **MTP head** | extra prediction heads trained with the model | high — trained on the same distribution |
| **EAGLE-style** | autoregressive head over the target's *features*, not tokens | highest |
| **N-gram / prompt lookup** | retrieve continuations from context | high for repetitive/code text, free |

The case study uses an **MTP head** — 0.85 GB, Class C in [Module 04's](Lecture-04.md) ledger. It is on the decode path only when speculation is enabled, and that conditionality is exactly what made the ledger change shape in Module 04 §5.

---

## 2. The speedup model

From [Module 08 §4](Lecture-08.md), with per-token acceptance `α` and draft depth `K`:

```text
   τ  =  1 + α + α² + ... + α^K  =  (1 − α^{K+1}) / (1 − α)          expected tokens per cycle
```

One cycle costs one target pass plus `K` draft passes. With `c = T_draft / T_target`:

```text
                 τ
   S  =  ─────────────────
           1  +  K · c
```

```text
   NUMERATOR   τ      grows with α and K, but SATURATES (α^K → 0)
   DENOMINATOR 1+K·c  grows LINEARLY with K, forever
                      ⟹ an interior optimum in K always exists
```

Optimal `K` by regime:

| α | c | **K\*** | S |
|---:|---:|---:|---:|
| 0.70 | 0.05 | 6 | 2.35 |
| 0.70 | 0.10 | 4 | 1.98 |
| 0.70 | 0.20 | 3 | 1.58 |
| **0.785** | **0.10** | **5** | **2.38** |
| 0.85 | 0.05 | 10 | 3.70 |
| 0.85 | 0.20 | 5 | 2.08 |

Two operational rules fall out:

* **A cheaper drafter buys depth.** Halving `c` from 0.10 to 0.05 raises `K*` from 5 to 7 at α = 0.785 and lifts speedup ~24 %. This is a strong argument for keeping the MTP head *small*, and — as §4 shows — a weak argument for quantizing it.
* **`K` must be re-tuned after any change to α.** A quantization that lowers acceptance also lowers the optimal draft depth. Teams that fix `K` in a config file and then quantize are running a suboptimal `K` and blaming the quantization.

---

## 3. The feedback loop

Here is the interaction the rest of this course has been building toward.

```text
   quantize the target
        │
        ├──▶  B_token ↓   ──▶  each target pass is FASTER      ──▶  tok/s ↑
        │
        └──▶  p moves    ──▶  TV(p,q) ↑  ──▶  α ↓  ──▶  τ ↓   ──▶  tok/s ↓
                                          (Module 08 §4)

   NET = (byte gain)  ×  (acceptance loss)
```

Combining with the throughput equation:

```text
              BW_eff        τ(α)
   tok/s  =  ────────  ×  ──────────
              B_token      1 + K·c
              └──────┘     └────────┘
              Modules 1–7   this module
```

**Both factors move when you quantize the target, and they move in opposite directions.**

### Quantifying the tax on the case study

```text
   MLP + O quantized          147.87 tok/s     τ = 2.792
   MLP + O + QKV quantized    150.73 tok/s     τ = 2.546
```

Decompose the observed change:

```text
   observed tok/s ratio        =  150.73 / 147.87  =  1.0193
   τ ratio                     =  2.546  / 2.792   =  0.9119
   ⟹ implied B_token ratio     =  1.0193 / 0.9119  =  1.1178
                                   (B_token fell 10.5 %)
```

Cross-check against [Module 07's](Lecture-07.md) traffic table — Q + K + V are 14.26 % of `B_token`, and BF16 → NVFP4 removes `(1 − 0.5625/2) = 71.9 %` of their bytes:

```text
   predicted B_token reduction  =  14.26 % × 71.9 %  =  10.25 %
   implied from measurement                          =  10.5 %      ✓
```

The model closes to within 0.25 points. So:

```text
   ┌────────────────────────────────────────────────────────────────┐
   │   potential gain (if acceptance had held)   :   +11.78 %       │
   │   realized gain                             :    +1.93 %       │
   │   ─────────────────────────────────────────────────────────    │
   │   ACCEPTANCE TAX  =  83.6 % of the byte win, consumed          │
   └────────────────────────────────────────────────────────────────┘
```

**Six sevenths of the throughput win was paid back as lost acceptance** — and that is *before* counting the behavioral cost of a target distribution that moved by 0.082 in total variation ([Module 08](Lecture-08.md)).

A configuration whose sole justification is "+1.9 % tok/s" is not worth a measurable behavior regression. **The acceptance tax converts what looks like a marginal win into a clear loss**, and you can only see it if you measure acceptance alongside throughput.

> This is the general lesson: **in a speculative system, throughput and behavior are not independent axes.** Damaging the target's distribution costs you speed directly, through α. The metrics you thought were in tension are partly aligned — which is good news, because it means the honest choice is usually also the fast one.

---

## 4. Quantizing the drafter is a different decision

Target and drafter sit on opposite sides of the equation:

| | Quantize the **target** | Quantize the **drafter** |
|---|---|---|
| Effect on `B_token` | large — it is the model | small — MTP head is 0.85 GB |
| Effect on `c` | none | reduces it → allows deeper `K` |
| Effect on α | reduces (p moves) | reduces (q moves) |
| Effect on **output quality** | **yes — p is the output distribution** | **none — losslessness holds** |

Two consequences:

**1. The drafter has no quality budget to protect.** By the losslessness guarantee, a degraded drafter cannot change the output distribution. So drafter quantization is a **pure speed/acceptance trade** with no behavioral risk — a much easier decision than target quantization.

**2. But the drafter is small, so the win is small.** The MTP head is 0.85 GB against a 15.88 GB target. Quantizing it to NVFP4 saves 0.61 GB per draft step, reduces `c` modestly, and costs acceptance. Using the numbers from [Module 04 §5](Lecture-04.md), the MTP head is ~4.6 % of traffic at `K = 3`.

```text
   Do NOT quantize the drafter aggressively.
   It is 4.6 % of the traffic and 100 % of the acceptance rate.
```

The asymmetry is sharp: the drafter's *only* job is to agree with the target. Damaging it attacks the one quantity — α — that multiplies your entire throughput. Keep the drafter at BF16 or FP8 unless you have measured that a lower precision holds acceptance.

---

## 5. Speculation changes which optimization is correct

[Module 04 §5](Lecture-04.md) showed bandwidth utilization dropping from 72 % to ~55 % when speculation is enabled. Now the reason is clear: speculation amortizes one weight-read across ~2.9 emitted tokens, so the same weights support far more tokens per second.

```text
   NO SPECULATION           SPECULATION ENABLED
   ────────────────         ─────────────────────────────
   72 % of peak BW          ~55 % of peak BW
   bandwidth-bound          NOT bandwidth-bound
   → remove bytes           → raise α, and fix the kernels
```

| Lever | Value without speculation | Value with speculation |
|---|---|---|
| Quantize weights further | high | **diminishing** — you are at 55 % of peak |
| Raise acceptance α | n/a | **highest** — multiplies everything |
| Tune draft depth `K` | n/a | high, and free |
| Improve kernel efficiency | high | **high** — the 45 % gap is not bytes |
| Reduce KV traffic (long ctx) | high | high — [Module 09](Lecture-09.md) |

**In the DSpark configuration, the ranked next actions are: (1) fix the kernels, (2) tune `K`, (3) raise α, (4) quantize `lm_head`. Further body quantization is fifth** — and quantizing Q/K is negative-value once the acceptance tax is counted.

That ordering is the practical payoff of the whole course, and note that it is not what any single measurement would have told you.

---

## Checkpoint

You should now be able to:

1. State the rejection-sampling rule and explain the losslessness guarantee.
2. Derive `S = τ/(1+K·c)` and explain why an interior optimum in `K` exists.
3. Decompose an observed throughput change into byte and acceptance factors.
4. Compute the acceptance tax and use it to reject a marginal configuration.
5. Explain why the drafter has no quality budget but should still not be quantized hard.
6. Explain why enabling speculation reorders the optimization priorities.

---

## Ship it

Build an **acceptance-aware benchmark harness** that reports, for every configuration:

```text
   config | B_token | τ | α | K | c | S | tok/s | predicted tok/s | byte gain % | realized % | tax %
```

with `predicted tok/s = BW_eff/B_token × τ/(1+Kc)`. Then run the `K` sweep at fixed quantization and the quantization sweep at fixed `K`, and report:

* your measured `K*`, and whether it moved after quantization
* the acceptance tax for each quantization step
* **any configuration where the tax exceeded 50 % of the byte win** — those are your rejected candidates, and listing them is the point

---

## Current as of

* **Timeless:** the rejection-sampling rule and losslessness, `τ = (1−α^{K+1})/(1−α)`, `S = τ/(1+K·c)`, the acceptance-tax decomposition.
* **Case-study pins:** 147.87 tok/s @ τ 2.792 and 150.73 @ τ 2.546; the derived 10.5 % `B_token` reduction and 83.6 % acceptance tax. The α values assume `K = 3`; the tax computation is independent of `K` since it uses τ ratios directly.
* **2026 drafting methods:** MTP heads, EAGLE-3-style feature-level drafting, and n-gram/prompt-lookup are the current families. See [MLSys Deep Dives — Lecture 06](../MLSys%20Deep%20Dives/Lecture-06.md) for the systems treatment.

---

**Next:** [Module 11 — Hardware-Aware AutoQuant →](Lecture-11.md)
