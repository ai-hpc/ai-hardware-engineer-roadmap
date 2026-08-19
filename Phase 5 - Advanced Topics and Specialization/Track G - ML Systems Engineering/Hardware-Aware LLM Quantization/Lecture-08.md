# Module 08 — Behavior Preservation

**Collection:** [Hardware-Aware LLM Quantization](README.md) | **Previous:** [← Module 07](Lecture-07.md) | **Next:** [Module 09 →](Lecture-09.md)

---

Every module so far has been about making the model faster. This one is about proving you did not break it — which is the harder half, because the failure is silent. A quantization that damages the model does not crash, does not warn, and frequently does not move perplexity.

The headline result of this module is an exact identity that turns speculative decoding's acceptance rate into a **free, online, statistically-grounded measurement of how far your quantized model drifted.** It is the cheapest sharp instrument available, and most teams already have it running and are not reading it.

---

## Learning objectives

By the end of this module you should be able to:

1. Order the behavioral metrics by sensitivity and cost, and say which to run when.
2. Explain why perplexity is a *rough* screen and what it misses.
3. Use mean KL vs. the FP16 reference and top-1 agreement as the working gate.
4. **Prove** that speculative acceptance rate equals `1 − TV(p, q)`, and use it to measure distribution drift.
5. Compute the number of tokens needed to detect a given acceptance change, and avoid reporting noise.

---

## 1. The metric ladder

```text
   CHEAP, INSENSITIVE                                    EXPENSIVE, DECISIVE
   ─────────────────────────────────────────────────────────────────────────▶

   weight MSE ──▶ layer-output MSE ──▶ perplexity ──▶ mean KL ──▶ top-1 ──▶ acceptance ──▶ benchmarks
        │              │                    │            │          │           │             │
     seconds        minutes             ~10 min      ~10 min    ~10 min     ~minutes       hours
     tells you      tells you           weak         THE        intuitive   sharp,         ground
     nothing        which layer         signal       GATE       and cheap   free,          truth
     about behavior  moved                                                   online
```

Two rules govern the ladder:

* **Never promote on a lower rung alone.** Weight MSE is a debugging aid, not evidence.
* **Never skip to benchmarks.** They are the ground truth and far too slow to steer a search over dozens of quantization configurations. Use them to *confirm* a decision, not to make it.

---

## 2. Why perplexity is a weak instrument

Perplexity is `exp(mean NLL)` over a corpus ([Logprobs, Perplexity & KL Divergence — Lecture 03](../Logprobs,%20Perplexity%20and%20KL%20Divergence/Lecture-03.md)). Its weakness for quantization grading is structural:

```text
   PPL is a MEAN over tens of thousands of tokens.

   Most tokens are easy: "of the", "  return", "). "  →  the model is confident
   and stays confident after quantization. These dominate the mean and DILUTE
   the signal from the tokens where the damage actually lands.
```

```text
   PPL  16.21  →  16.28     ( +0.4 % )    "looks fine, ship it"
   but underneath:
      98 % of tokens        : unchanged
       2 % of tokens        : top-1 prediction FLIPPED
      long-context behavior : degraded (Module 07 §2)
      tool-call formatting  : intermittently broken
```

Three specific blind spots:

| Blind spot | Why PPL misses it |
|---|---|
| **Tail behavior** | rare-but-critical tokens are averaged away |
| **Long context** | PPL is usually measured with a short sliding window |
| **Distribution shape** | PPL only reads the probability of the *observed* token; the rest of the distribution can deform freely |

That last one is the killer, and it is exactly what KL fixes.

> Use PPL as a **smoke test**: a large jump means something is badly wrong. A small jump means almost nothing. Never ship on PPL alone.

---

## 3. The working gate: KL and top-1 agreement

Compare your quantized model `q` against the unquantized reference `p` **on the same inputs**, comparing full next-token distributions rather than just the observed token:

```text
   D_KL(p ‖ q)  =  Σ_x  p(x) · log( p(x) / q(x) )
```

This reads the *entire* distribution, which is what makes it strictly more informative than perplexity. Report three numbers together:

```python
def grade_quantization(ref_model, quant_model, dataset, top_k=1):
    kls, agree, probs_ref, probs_q = [], [], [], []
    for batch in dataset:
        with torch.no_grad():
            lp = torch.log_softmax(ref_model(batch).logits.float(),   dim=-1)
            lq = torch.log_softmax(quant_model(batch).logits.float(), dim=-1)
        p = lp.exp()
        kls.append((p * (lp - lq)).sum(-1).flatten())          # per-token KL, nats
        agree.append((lp.argmax(-1) == lq.argmax(-1)).flatten())
        probs_ref.append(p.max(-1).values.flatten())
        probs_q.append(lq.exp().gather(-1, lp.argmax(-1, keepdim=True)).squeeze(-1).flatten())

    kl = torch.cat(kls)
    return {
        "mean_kl":      kl.mean().item(),
        "p99_kl":       kl.quantile(0.99).item(),      # ← the tail PPL hides
        "top1_agree":   torch.cat(agree).float().mean().item(),
        "prob_rms":     (torch.cat(probs_ref) - torch.cat(probs_q)).pow(2).mean().sqrt().item(),
    }
```

Working thresholds (calibrate to your own tolerance, but these are defensible starting points):

| Mean KL (nats) | Top-1 agreement | Verdict |
|---:|---:|---|
| < 0.01 | > 99 % | indistinguishable — ship |
| 0.01 – 0.05 | 97–99 % | acceptable for most uses; confirm on benchmarks |
| 0.05 – 0.15 | 93–97 % | noticeable; only for aggressive throughput targets |
| > 0.15 | < 93 % | broken — find which layer, using [Module 07's](Lecture-07.md) sweep |

**Always report `p99_kl` alongside `mean_kl`.** A configuration with mean KL 0.02 and p99 KL 3.0 is damaging a small set of tokens catastrophically — the exact failure perplexity conceals.

---

## 4. Acceptance length — the identity that makes it rigorous

Now the sharpest cheap instrument you have.

In speculative decoding, a draft model `q` proposes a token, and the target `p` accepts it with probability `min(1, p(x)/q(x))` ([Module 10](Lecture-10.md)). The **overall** acceptance probability is:

```text
   α  =  Σ_x  q(x) · min( 1,  p(x)/q(x) )
      =  Σ_x  min( q(x),  p(x) )
      =  1  −  TV(p, q)
```

using `Σ min(p,q) = 1 − ½Σ|p−q| = 1 − TV(p,q)`.

```text
   ┌──────────────────────────────────────────────────────────────┐
   │   acceptance rate  =  1  −  total variation distance          │
   │                                between target and draft       │
   └──────────────────────────────────────────────────────────────┘
```

This is exact, not an approximation. And it has a consequence that is easy to miss:

> If you hold the **drafter fixed** and quantize only the **target**, then any change in acceptance rate is a **direct measurement of how far the target's distribution moved** — in total variation, over the real serving distribution, computed for free while serving.

You are already running this measurement. You may just not have known it was a distribution-drift meter.

### Recovering α from acceptance length

Runtimes usually report **acceptance length** `τ` (mean tokens emitted per target forward pass) rather than the per-token rate. With `K` draft tokens per cycle:

```text
   τ  =  1 + α + α² + ... + α^K  =  (1 − α^{K+1}) / (1 − α)
```

Invert numerically to recover `α`, then `TV = 1 − α`. Applying this to the case-study measurements:

| Config | τ | α (K=3) | **TV(p, q)** | Δ TV vs baseline |
|---|---:|---:|---:|---:|
| Baseline (target unquantized) | 2.886 | 0.7852 | 0.2148 | — |
| MLP + O quantized | 2.792 | 0.7636 | 0.2364 | **+0.0216** |
| MLP + O + QKV quantized | 2.546 | 0.7034 | 0.2966 | **+0.0818** |

**Adding QKV moved the target distribution nearly four times as far as quantizing MLP + O did** (`+0.082` vs `+0.022` in TV), while [Module 07](Lecture-07.md) showed it bought only 1.9 % throughput. The acceptance number was telling you that the whole time.

> `K` matters for the absolute α values (at `K=2` the same τ implies α ≈ 0.96; at `K=4`, α ≈ 0.72), so **always report your draft depth alongside τ.** The *ranking* of configurations is unaffected by `K`, which is why acceptance works as a comparator even when you are unsure of the drafter's exact configuration.

### Why this beats perplexity

| | Perplexity | Acceptance length |
|---|---|---|
| Reads | probability of the observed token | full distribution, via TV |
| Distribution | a static corpus | **your actual serving traffic** |
| Cost | a separate evaluation run | **free — already running** |
| Latency to signal | ~10 minutes | **continuous, online** |
| Sensitivity to tail damage | poor (averaged away) | good (rejections concentrate there) |

The one caveat: **acceptance is a comparison against your drafter, not against truth.** If you quantize the drafter too, both distributions move and the measurement is confounded. Keep the drafter fixed across a quantization sweep — this is a controlled-experiment requirement, and [Module 12](Lecture-12.md) treats it as such.

---

## 5. How many tokens before you believe the number?

Acceptance is a Bernoulli process, so the standard error on `α` from `N` draft tokens is `√(α(1−α)/N)`:

| α | Target SE | Draft tokens needed |
|---:|---:|---:|
| 0.75 | 0.010 | ~1,900 |
| 0.75 | 0.005 | ~7,500 |
| 0.75 | 0.002 | ~46,900 |
| 0.80 | 0.005 | ~6,400 |

To claim a **1-point** acceptance difference between two configurations you need roughly **8,000 draft tokens per configuration** — a couple of minutes of generation. To claim a 0.4-point difference you need ~47,000.

```text
   Reported: "acceptance dropped from 2.886 to 2.871"
   Measured over: 500 tokens
   ⟹  SE ≈ 0.019 on α;  the difference is INSIDE the noise.
       This is not a result. It is a coin flip with a decimal point.
```

Most published acceptance comparisons do not state their sample size. Yours should — and if the delta is inside two standard errors, report it as "no detectable difference", not as a small improvement.

---

## 6. The gate, assembled

Run this in order and stop at the first failure:

```text
   1.  layer-output MSE          seconds   →  did the quantization run correctly at all?
   2.  perplexity                ~10 min   →  smoke test; large jump = something is broken
   3.  mean KL + p99 KL          ~10 min   →  THE GATE (thresholds in §3)
   4.  top-1 agreement           free      →  intuitive cross-check on the same run
   5.  acceptance length         minutes   →  drift measured on real traffic; ≥8k draft tokens
   6.  task benchmarks           hours     →  confirm the decision; never steer with it
   7.  long-context probe        hours     →  Modules 07 §2, 09 — DO NOT SKIP if you serve long context
```

Step 7 is the one teams omit and regret. Every metric above except the last is typically measured at short context, and [Module 07's](Lecture-07.md) RoPE argument predicts that Q/K damage *grows* with context length. A configuration that passes at 4 K can fail at 262 K, and nothing in steps 1–6 will warn you.

---

## Checkpoint

You should now be able to:

1. Order the metric ladder and state each rung's cost and sensitivity.
2. Explain the three things perplexity structurally cannot see.
3. Set KL and top-1 thresholds, and explain why `p99_kl` must accompany `mean_kl`.
4. Derive `α = 1 − TV(p, q)` from the rejection-sampling rule.
5. Convert an acceptance length and draft depth into a TV distance.
6. Compute the sample size needed to support an acceptance claim.

---

## Ship it

Build a **quant grader** that emits one row per configuration:

```text
   config | B_token | tok/s | PPL | mean_KL | p99_KL | top1% | τ | α | TV | n_draft_tokens | verdict
```

Run it across the [Module 07](Lecture-07.md) leave-one-out configurations. The deliverable is the table **plus a written promotion rule** fixed in advance — the KL and acceptance thresholds you will ship at, committed *before* you see the results. That ordering is what makes it an experiment rather than a rationalization.

---

## Current as of

* **Timeless:** the metric ladder, the KL gate, `α = 1 − TV(p,q)`, the `τ = (1−α^{K+1})/(1−α)` relation, the Bernoulli sample-size arithmetic.
* **Prerequisite depth:** [Logprobs, Perplexity & KL Divergence](../Logprobs,%20Perplexity%20and%20KL%20Divergence/README.md) derives the information-theoretic identities used here; [Lecture 05](../Logprobs,%20Perplexity%20and%20KL%20Divergence/Lecture-05.md) covers llama.cpp's `--kl-divergence` grading mode.
* **Case-study pins:** τ values 2.886 / 2.792 / 2.546. The α and TV columns assume draft depth `K = 3`; recompute for your own `K`.

---

**Next:** [Module 09 — KV Cache & Long Context →](Lecture-09.md)
