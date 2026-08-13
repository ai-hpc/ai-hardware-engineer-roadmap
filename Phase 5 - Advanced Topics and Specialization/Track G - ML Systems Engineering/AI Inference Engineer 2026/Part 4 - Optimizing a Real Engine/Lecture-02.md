# Part 4 · Lecture 02 — The Scoreboard: A Benchmark That Cannot Be Gamed

## Overview

This is the lecture most optimization courses skip, and it is the one that decides whether the rest of the work counted.

The case-study repository has roughly **forty merged pull requests whose only purpose was to fix the measurement**. Not the engine — the *ruler*. Each closed one specific way a number could look better than reality. That ratio is not a sign of a badly-built harness; it is what happens when a benchmark is put under real incentive pressure and the failures are recorded instead of quietly patched.

By the end of this lecture you should be able to design a performance gate for your own workload that survives three distinct adversaries: **an honest engineer fooling themselves**, **a hostile contributor optimizing for the metric**, and **the hardware itself being non-deterministic**. Those need different defenses, and confusing them is why most internal benchmarks are decorative.

---

## 1. Why the scoreboard comes first

The intuitive order is: build the thing, then measure it. That order fails for a reason that has nothing to do with laziness.

**A benchmark is a specification of what you will optimize.** Every hour of engineering attention flows toward the number on the board. If the number is wrong, the attention is wrong — and you will not find out from the number, because the number will be going up.

The case study's canonical instance, from [Lecture 01](Lecture-01.md):

```text
   the harness measured        ctx 64
   the slot was named          KIMI_K3_H200X8_IQ1S_SPARKINFER_128
   the contribution guide said 128k
   the hardware saw            64

                        ctx 64      ctx 131,072
   llama.cpp             18.32          18.44
   sparkinfer            10.34           1.00      <- the real gap
                         1.8x            18x
```

Weeks of kernel work were graded at a context nobody runs. The engine was not lying and neither was anyone else — three artifacts disagreed and none of them was authoritative. [PR #51](https://github.com/gittensor-ai-lab/sparkinfer-k3/pull/51) ("*score at 128k, the context this repo actually targets*") fixed it, and the frontier promptly *dropped* from 9.63 to 1.00 because it had started measuring the real thing.

> **A frontier that drops when you fix the harness is the harness working.** If correcting a measurement never costs you a number you liked, you have not corrected a measurement.

There is a second, subtler reason to build the scoreboard first: **you cannot retroactively grade a ladder.** The repo can audit its own history row by row only because each row was sealed when it happened. Reconstructing 46 rungs afterwards from memory and re-runs is not possible on rented hardware you no longer hold.

---

## 2. Gate order — correctness *before* speed

The single most important structural decision in this harness is the order of two checks.

```text
   kimi_k3_eval.sh
     │
     ├─ 1. ACCURACY GATE   top-1 ≥ 0.95   AND   mean KL ≤ 0.05
     │      │                     against the captured reference,
     │      │                     identical weights, identical token ids
     │      └─ fail → REJECT.  the speed number is never even reported.
     │
     ├─ 2. SIGNIFICANCE GATE   gain > 2% of the current frontier
     │      └─ fail → label `none`  (inside measurement noise)
     │
     └─ 3. TIER   min(delta / llama_ref, delta / frontier)  →  xs | s | m | l | xl
```

The rule stated plainly: *a speedup that erodes parity is not a speedup.* A faster kernel that changes the model's output is worth **zero**, and the harness enforces that by refusing to compute a tier at all rather than by trading accuracy against speed on some curve.

This matters more for inference engines than for almost any other software, because — as [Lecture 10](Lecture-10.md) develops at length — the natural failure mode of a broken inference kernel is **fluent, plausible, wrong output**. There is no crash to stop you. If the speed gate runs first, corruption reads as a win, and the fastest configuration in your sweep is the most broken one. That is not hypothetical here: [Lecture 10 §3](Lecture-10.md) covers a PR that measured **169.72 tok/s** because two live defects made it read the wrong rows.

### 2.1 What "correctness" means when there is no second implementation

There is no independent reimplementation of Kimi K3 to check against, and the target quantization's *own* top-1 against full precision is only 90.4%. So absolute quality numbers are meaningless here. Correctness is defined narrowly and honestly:

> **Agreement with the reference engine on identical weights and identical token ids.** Nothing else.

That definition has real consequences the repo documents rather than hides:

* **Two reference-server flags are mandatory, not tuning.** `--no-context-shift`, because K3 is a hybrid recurrent architecture that llama.cpp cannot context-shift — a long eval dies mid-run without it. And `--no-jinja`, because the gate posts raw token ids and a chat template would prepend tokens the candidate never saw.
* **The residual KL is known and accepted.** Mean KLD sits ~400× above the 1e-5 bar you would expect from two runs of the *same* implementation, from an identified cause: K3 keeps f32 activations where ggml quantizes them before a quantized mat-vec. The instruction to contributors is exactly right — *do not go hunting it as a bug; do not make it worse.*
* **The gate cannot run at the scored context.** Capturing a reference at 131,072 tokens is prohibitively expensive, so parity is graded at short depths. The untested region is stated out loud (§7).

### 2.2 Top-1 and KL do different jobs

A subtlety worth internalizing, because it generalizes to any parity gate you build:

| | What it is | Behaviour |
|---|---|---|
| **top-1 ≥ 0.95** | `argmax_ref == argmax_ours` on one logit row per probe | Effectively **boolean**. Any bar in (0, 1] behaves identically. All 48 top-1 values in the sealed log are exactly `1.0`. |
| **mean KL ≤ 0.05** | Sum over all 163,840 vocab entries | The **graded** gate. Moves long before an argmax flips. Merged runs sit at **0.004–0.008**. |

So the pair is not redundant: top-1 says *did the decision change*, KL says *how far is the distribution drifting*, and KL is the one that gives you warning. Merged runs sitting an order of magnitude under the bar is what a healthy margin looks like — and is why [PR #147](https://github.com/gittensor-ai-lab/sparkinfer-k3/pull/147) could raise the top-1 bar to 0.95 without breaking anything.

This is the mechanism developed from first principles in [Logprobs, Perplexity & KL Divergence — Lecture 05](../../Logprobs,%20Perplexity%20and%20KL%20Divergence/Lecture-05.md). If the pairing feels arbitrary, read that first.

### 2.3 Grade the worst depth, not the average

The gate probes **seven context depths** — 4, 128, 256, 512, 1024, 2048, 4096 — as nested prefixes of one document, and takes the **worst**, not the mean.

Until [PR #116](https://github.com/gittensor-ai-lab/sparkinfer-k3/pull/116) it was a single 4-token prompt graded on one next-token distribution. That gate could not distinguish "correct" from "correct only while the KV cache is nearly empty" — which is precisely the bug class that a KV-cache, attention, routing, or LM-head change introduces.

Two consequences the contribution guide spells out:

* A regression at one depth fails the gate even if every other depth is perfect.
* Parity is additionally **ratcheted against `main` measured in the same round, per depth** ([PR #83](https://github.com/gittensor-ai-lab/sparkinfer-k3/pull/83)) — so drift is visible even while it is legal.

And then the correction to that ratchet, which is a good lesson in over-tightening: [PRs #124](https://github.com/gittensor-ai-lab/sparkinfer-k3/pull/124) and [#126](https://github.com/gittensor-ai-lab/sparkinfer-k3/pull/126) established that **below the 0.05 bar, a ratio is never a regression.** A change that moves KL from 0.001 to 0.004 is 4× "worse" and still forty times inside the bar. Flagging it trains contributors to ignore the flag. The final rule: `accuracy-regression` applies only when a depth is both ≥2× main **and** at or over the bar — at which point the absolute check rejects anyway.

> **Steal this.** A ratcheting gate needs an absolute floor below which ratios are ignored, or it generates noise proportional to how good you have become.

---

## 3. Measuring speed on hardware that will not sit still

Correctness is deterministic: fixed weights, fixed inputs, greedy decode ⇒ exact, and you verify by re-running. Speed is not. Clocks drift, thermals vary, boxes differ. The two halves of the gate therefore need entirely different trust models, and the repo separates them explicitly.

The techniques, each traceable to the failure that motivated it:

**Interleave both builds on the same box.** `main` and the PR are built and benched on the same node, interleaved, and scored as a **same-box delta**, so box-to-box hardware variance cancels. How much variance? The repo records `main` reading **18.14 pinned vs 18.88 measured** when the box changed — several percent, which is larger than an `S` tier.

**Never cross-compare boxes.** The llama.cpp pin was originally **16.7026**, a single rep on a different machine. Corrected to **18.4435** — 3-rep median on the box that scores every PR ([PR #101](https://github.com/gittensor-ai-lab/sparkinfer-k3/pull/101)). The note in `reference.lock` is exact about what happened: *"Nothing about anyone's measured speed changed; the yardstick was wrong."* Because the tier basis is `delta / llama_ref`, raising it made every tier ~9.4% smaller.

**A single rep is not a measurement.** The same lock file records a "−8% decode falloff with depth" that turned out to be *a measurement artefact* — one rep (`llama-bench -r 1`) on a different box. Re-measured with 3 reps on the scoring box, llama.cpp holds essentially flat with depth (18.32 → 18.44). The artefact had been flattering: it made the reference look like it degraded with context when it was the candidate that fell off a cliff.

**Warm the page cache.** [PR #92](https://github.com/gittensor-ai-lab/sparkinfer-k3/pull/92) — "*warm the page cache so the frontier is not measured on a cold box*." Loading 553 GiB from cold disk versus warm cache is not a small effect, and whichever build runs first eats it.

**Rebuild from scratch with a pinned compiler.** [PR #93](https://github.com/gittensor-ai-lab/sparkinfer-k3/pull/93). A reused build directory means you are partly measuring the previous run's artifacts.

**Wall clock is a bound, not the reading.** [PR #61](https://github.com/gittensor-ai-lab/sparkinfer-k3/pull/61), refined by [#68](https://github.com/gittensor-ai-lab/sparkinfer-k3/pull/68) and [#121](https://github.com/gittensor-ai-lab/sparkinfer-k3/pull/121) which sized the margin to the observed jitter. Total wall time can *falsify* a reported tok/s — a claim implying more work than the elapsed time allows is impossible — but it cannot *be* the reading, because it includes load, warmup, and teardown.

**Distinguish a flaky node from a regression.** [PRs #118](https://github.com/gittensor-ai-lab/sparkinfer-k3/pull/118) and [#82](https://github.com/gittensor-ai-lab/sparkinfer-k3/pull/82): a busy or dying box must cause a *retry with a diagnosis*, not a failed round and not a recorded regression. Otherwise infrastructure noise enters the ladder as engineering history, and every future reader has to guess which rows are real.

---

## 4. The tier function, and why it has two denominators

Past the 2% significance gate, the label is:

```text
   tier_basis = min( delta / llama_ref ,  delta / frontier )

   xs  < 3.5%        s  3.5–6%        m  6–10%        l  10–18%        xl  > 18%
```

Taking the **worse** of two bases means the tier **can never exceed the speedup actually measured**. The two terms bind in opposite regimes, and that is the entire design:

| Regime | Which term is smaller | What it prevents |
|---|---|---|
| Frontier **below** the reference | `delta / llama_ref` | An immature engine minting `xl`s from low-hanging fruit. Doubling 1 tok/s is a 100% gain and ~5% of the reference. |
| Frontier **past** the reference | `delta / frontier` | Cheap tiers once you lead. At 2.2× ahead, `xl` costs a real **18% over `main`** — not 18% of a reference `main` already beat. |

The failure this replaced is instructive. Tier credit used to be capped at *twice* the measured gain — invisible while the engine was behind, and then, once the frontier passed 2× the reference, it became the whole rule and **`xl` was costing 9%**. [PR #122](https://github.com/gittensor-ai-lab/sparkinfer-k3/pull/122) capped credit at the measured speedup and **re-scored the entire ladder from the sealed receipts.** Being able to do that retroactive re-score, correctly, is the payoff for sealing every round.

### 4.1 When to switch the denominator off

When the scored metric moved to prefill ([Lecture 09](Lecture-09.md)), the llama anchor was **disabled** and the tier became `delta / frontier` alone. The reasoning is worth following closely, because it is a genuinely non-obvious modelling point.

The tier buckets are *fractions of the reference*. llama.cpp's two numbers are **7.8× apart** — 18.44 decode versus 143.88 prefill. So leaving the anchor on would make a tier cost **2.7× more on prefill than on decode**: an `l` would need +27.1% over the frontier instead of +10.0%.

> Nothing about prefill work is 2.7× harder. That number came from llama.cpp's shape, not from ours.

And a second reason: at that point llama.cpp *batched* the prompt and SparkInfer walked it token by token, so `delta / 143.88` sized a gain against **a feature that had not been built** rather than against the work in the PR. The anchor is scheduled to flip back on once both engines are doing the same thing — at which point it means what it says again.

`pct_of_llama` is still recorded on every run; it stopped being the tier basis, not the target. The verdict JSON carries **both** — `pct_over_frontier` (the honest measured speedup), `pct_of_llama` (the tier basis), and `scored_context` (which context earned it).

---

## 5. The frontier — a number with an owner and a provenance

The frontier is the best figure measured on `main` by a sealed round. Four properties, each earned:

**Raise-only.** So a slow box cannot deflate it and mint tiers for everyone behind it.

**Re-measured every round, not trusted from a pin** ([PR #50](https://github.com/gittensor-ai-lab/sparkinfer-k3/pull/50)). A pin is a claim about `main` that someone made once.

**Never hand-filled.** `check_reference_lock.py` is a required CI job: `0` means "not measured" and is always allowed, but **any non-zero baseline must trace to a committed measurement JSON** for the same node *and* context. The repo's justification is the sharpest sentence in its docs on this subject:

> *Downstream, a hand-filled baseline is indistinguishable from a measured one.*

**Quant- and node-qualified by name.** `KIMI_K3_H200X8_IQ1S_LLAMA_128K`, not `LLAMA_128K`. The three milestone nodes run the same quant at the same context, so a single shared slot would silently overwrite one milestone's reference with another's ([PR #24](https://github.com/gittensor-ai-lab/sparkinfer-k3/pull/24) fixed reading the unqualified slot).

### 5.1 The stale-pin lesson, twice — and the asymmetry nobody expects

A stale frontier caused two *different* problems, and the difference between them is the actual lesson.

**As a tier basis, stale-low is self-correcting.** It over-scores the next PR by whatever the gap is, and the next round re-measures and fixes it. Annoying, bounded.

**As a regression guard, stale-low is dangerous.** The decode guard's floor is 1% under the pinned value. The slot read **46.48** while `main` actually did ~56.8 — so it would have admitted a decode collapse to **46.02**, a **19% regression**, while cheerfully reporting that the guard passed. [PR #134](https://github.com/gittensor-ai-lab/sparkinfer-k3/pull/134) set it to 56.8 by hand for exactly this reason, and documented the risk of doing so (the slot is raise-only, so if `main` really measures under the floor, every PR fails the guard until a human intervenes).

> **The same number in two roles needs two different staleness tolerances.** A basis wants to be *current*; a guard wants to be *conservative in the safe direction*. Work out which yours is.

**And stale-low as a published target is expensive.** The prefill frontier was hand-seeded at 40.35 and attributed to the wrong commit. The bot always measured the real frontier each round, so nothing was mis-scored — but three PRs optimized against the *published* 40.35 and reported +25.4%, +24.5%, +5.3%, which against the real 53.02 were **−4.6%, −5.3%, −19.8%**. Only the one PR that measured `main` itself landed within 0.23% of the bot. [PRs #140](https://github.com/gittensor-ai-lab/sparkinfer-k3/pull/140) and [#146](https://github.com/gittensor-ai-lab/sparkinfer-k3/pull/146) made a stale pin *say so*, and the rule that came out of it is short:

> A seeded frontier is a claim about `main` that nobody measured. If one is unavoidable, measure the **current head**, and say which commit.

### 5.2 Measured vs attested

At the time of writing, the prefill frontier pin holds at **69.02** while the engine demonstrably does **99.68** on the node — same box, same weights, output bit-identical to the per-token path. The pin is deliberately *not* raised, because no sealed round has measured it.

That distinction is the whole point of a ledger: **the tier basis does not move on a number the log cannot show you.** The repo also records the consequence rather than discovering it later — with `main` faster than its own pin, the claim gate is temporarily *lenient*, and that is accepted deliberately as the lesser failure versus an unattested pin.

---

## 6. Twelve holes, and the PRs that closed them

Grouped by what was being exploited. Every one of these is a real merged fix; the titles are quoted because they are better than a paraphrase.

**The candidate must not grade itself.**

| PR | What it closed |
|---|---|
| [#56](https://github.com/gittensor-ai-lab/sparkinfer-k3/pull/56) | "*stop the PR's own binary deciding its tier*" |
| [#41](https://github.com/gittensor-ai-lab/sparkinfer-k3/pull/41) | "*grade with the harness from `main`, not the PR's copy*" |
| [#18](https://github.com/gittensor-ai-lab/sparkinfer-k3/pull/18) | "*guard the scoring function, not just its outputs*" |
| [#35](https://github.com/gittensor-ai-lab/sparkinfer-k3/pull/35) | "*pin the tier basis to trusted `main` and bind the result to the PR*" |

The general principle: **the thing being measured cannot supply any part of the measuring apparatus.** In this repo that is enforced structurally — `sensitive-paths-guard` is a *required status check* covering `label.py`, the measurement chain, `reference.lock`, the accuracy gate, and `bench/results/`. It runs on `pull_request_target`, so it applies to fork PRs and cannot be disabled by editing the workflow in your own branch. Harness improvements are welcome — via an issue, not a PR that would score itself.

**The verdict must be re-derived, not reported.** A node run posts `/eval RESULT_JSON {…}`; the workflow **recomputes** the tier from the reported measurements and from `reference.lock` *on the protected branch*, not from the payload ([#32](https://github.com/gittensor-ai-lab/sparkinfer-k3/pull/32) closed "answer key, verdict overwrite, dead validator"). A posted label cannot set a payout.

**A claim must be backed by a number.**

| PR | What it closed |
|---|---|
| [#139](https://github.com/gittensor-ai-lab/sparkinfer-k3/pull/139) | "*a ticked box must be backed by a measured number*" |
| [#132](https://github.com/gittensor-ai-lab/sparkinfer-k3/pull/132) | "*require a prefill claim and a decode-guard claim, not just a node tick*" |
| [#141](https://github.com/gittensor-ai-lab/sparkinfer-k3/pull/141) | "*skip a PR whose claimed prefill cannot beat the frontier*" — do not spend node hours on an arithmetic impossibility |
| [#44](https://github.com/gittensor-ai-lab/sparkinfer-k3/pull/44) | "*stop the bot claiming success it never verified*" |

**Some numbers are impossible.** [PR #130](https://github.com/gittensor-ai-lab/sparkinfer-k3/pull/130) added a **plausibility ceiling** — a claimed result above 5× the reference is rejected as implausible rather than recorded. This feels like an odd thing to need until you have seen a corrupted kernel post a spectacular number. The ceiling is not a statement about what is achievable; it is a statement that **a result too good to be true should be *checked*, not banked.** [Lecture 10](Lecture-10.md) has the case it caught.

**Scheduling is part of the measurement.** [#91](https://github.com/gittensor-ai-lab/sparkinfer-k3/pull/91) serve the oldest PR first, not the newest. [#85](https://github.com/gittensor-ai-lab/sparkinfer-k3/pull/85) never merge a PR that did not beat the frontier. [#99](https://github.com/gittensor-ai-lab/sparkinfer-k3/pull/99) label conflicting PRs instead of skipping them silently. [#79](https://github.com/gittensor-ai-lab/sparkinfer-k3/pull/79) stop booking the node for PRs that cannot be scored — and stop refusing the ones that can. [#62](https://github.com/gittensor-ai-lab/sparkinfer-k3/pull/62) close the draft-then-ready evasion.

And the constraint that makes marginal scoring coherent at all: **one open PR per contributor.** Two open PRs from one author are both measured against a baseline the other is about to move — the marginal-gain number only means something if they land one at a time.

**Originality is a measurement problem too.** `copycat-guard` fingerprints every diff against merged history by containment, with tiered consequences:

| containment | action |
|---|---|
| ≥ 90% | denylist + close |
| 80–90% | comment + auto-close |
| 70–80% | label for **semantic review** — never closes, never blocks |
| < 70% | ignored |

The 70–80% band exists because overlap alone cannot separate a renamed copy from an independent contributor who touched the same hot function. The asymmetry is stated explicitly: *a missed copycat costs one PR's emission; a false block costs a real contributor their access permanently.* Size your false-positive tolerance to the cost of the false positive, not to your confidence in the detector.

---

## 7. Document the asymmetries — and the honest boundaries

The most credible thing in this harness is not a defense. It is the set of places where it writes down what it *does not* prove.

**The two sides are not measured the same way, and that is in the lock file.** llama.cpp prefills 131,072 real tokens (`llama-bench -d`). SparkInfer had no prefill path — a genuine fill is ~10 hours of sequential `forward_token` calls — so it allocates the cache at 131,072, leaves it **zeroed**, and seeks position.

The justification is specific: **decode cost is data-independent** — the MLA reduction is dense whether the entries are activations or zeros — so the *timing* is faithful. The *correctness* is not, which is exactly why the accuracy gate runs separately at short context against a real capture, and never at 128k.

Then the defense against abusing that shortcut: a PR that stubbed `--seek` would score ctx-64 decode as 128k, roughly a free 10×. So the harness **refuses any run where the bench did not announce the seek it performed.**

> **Steal this pattern.** Any measurement shortcut needs three things written next to it: (1) what it makes faithful, (2) what it does *not*, and (3) the assertion that stops it being abused. A shortcut with only (1) is a hole.

**Stated limits, verbatim in spirit:**

* *"4096 is not the 131,072 you are scored at."* The parity gate's untested region is "everything past 4k" — with the opt-in depths on, past 32k. Not "nothing", and not "everything".
* Receipt verification is gated behind a repository variable that **is not set**, and the workflow prints a notice saying so on every run. Until it is on, a tier proves the arithmetic was recomputed on trusted inputs; **it does not prove the measurement happened.**
* Deep parity depths (8192/16384/32768) exist and are committed but default off, because 32768 costs 812 s per measured build against 136 s for 4096 — the deep pass runs through the decode path one token at a time. The cost is stated; the tradeoff is a choice, not an oversight.
* `--merge-admin` can land a change with no human reading it, within stated bounds. Documented, the repo says, "*because a contributor is entitled to know what can merge their work.*"

There is also a genuine structural limit on speed attestation, which the project's trust document states rather than papers over: **there is no cryptographic proof of a benchmark number.** Correctness is deterministic and can be re-run or sealed in a TEE; speed is not, and is made trustworthy by *reproduction and consensus* within a tolerance. Claiming more than that would be the dishonest option.

---

## 8. The checklist

Distilled from the forty-odd fixes above. Each line is traceable to something that actually went wrong.

**Order and scope**

1. Correctness gate runs **before** the speed gate, and a failure suppresses the speed number entirely.
2. Grade the **worst** case across a sweep, not the average.
3. A ratcheting comparison has an **absolute floor** below which ratios are ignored.
4. The **scored configuration is the one you ship** — verify the harness actually reaches it.

**Isolation**

5. The candidate supplies **no part** of the measuring apparatus. Enforce with a required check, not a convention.
6. The verdict is **re-derived** from measurements on trusted inputs, never accepted as reported.
7. Both sides built from scratch, same box, **interleaved**, pinned compiler, warm cache, recorded clocks.

**Provenance**

8. Every non-zero baseline **traces to a committed measurement**. Zero means "not measured" and is always legal.
9. Baselines are **named** for every axis their value depends on — node, quant, context.
10. The frontier is **raise-only** and re-measured each round.
11. A pin that has gone stale **says so**.
12. Distinguish **measured** from **attested**, and let the basis move only on the latter.

**Adversaries and reality**

13. Claims must be **backed by numbers**, and impossible claims rejected before they cost node hours.
14. Infrastructure failure produces a **retry with a diagnosis**, never a recorded regression.
15. Every measurement shortcut is documented with what it makes faithful, what it does not, and the assertion preventing abuse.
16. Write down what the gate **does not prove.**

---

## Lab — build the scoreboard before you optimize anything

Goal: a harness you would be willing to have audited. No engine changes in this lab.

1. **Pin the reference.** Repo, commit, weights hash, exact command line, in one committed file. If it lives in someone else's repository, add the assertion that fails when it moves.
2. **Capture a parity reference** at ≥5 nested context depths. Commit the logits or a hash of them.
3. **Write the gate**, in this order: parity (worst depth, absolute bar + ratchet-with-floor) → significance (a % of your frontier, sized to your measured noise) → tier.
4. **Measure your noise floor first.** Run `main` against itself 5+ times, interleaved. The spread *is* your significance threshold; if you picked 2% and your spread is 4%, your gate is decorative.
5. **Add the provenance check.** A CI job that fails if any non-zero baseline does not trace to a committed measurement. Then try to cheat it and fix what you find.
6. **Make one deliberate corruption.** Break a kernel so output is wrong but plausible, and confirm the gate rejects it *before* reporting speed. If it reports speed first, your order is wrong.
7. **Write `LIMITS.md`.** Three things your gate does not prove. Be specific about the untested region.

Pass criterion: a colleague can read your harness and name the measurement it is *most* vulnerable to — and it is a hole you already documented in `LIMITS.md`.

---

## Self-check

1. Your gate rejects on parity before reporting speed. A contributor argues this wastes node hours, since most PRs pass parity anyway. Answer them in three sentences.
2. Your frontier is stale-low by 15%. Give the consequence if it is (a) the tier basis, (b) a regression guard whose floor is 1% below it. Which is worse and why?
3. Tier = `min(delta/reference, delta/frontier)`. Compute the `xl` threshold (>18%) in absolute tok/s for a frontier at 10 tok/s and again at 50 tok/s, with the reference at 18.44. Which term binds in each case?
4. You must measure decode at 128k but cannot afford to prefill 128k of real tokens. Describe the shortcut, the property that makes it faithful for *timing*, why it is not faithful for *correctness*, and the assertion that stops someone stubbing it.
5. A PR reports 5.2× over the frontier. Your ceiling is 5×, so it is rejected as implausible. The author insists it is real. What exactly do you do next — and what would change your mind?
6. Your accuracy ratchet flags every PR that moves KL from 0.001 to 0.003. After two weeks nobody reads the flag. Diagnose the design error and fix it in one rule.
7. Why does "one open PR per contributor" follow logically from scoring marginal gains against a merged frontier?

---

## References

* **SparkInfer-K3 measurement chain** — [`CONTRIBUTING.md`](https://github.com/gittensor-ai-lab/sparkinfer-k3/blob/main/CONTRIBUTING.md) (gate order, tier bands, the prefill-anchor reasoning), [`bench/scripts/reference.lock`](https://github.com/gittensor-ai-lab/sparkinfer-k3/blob/main/bench/scripts/reference.lock) (pinned baselines and the commentary this lecture quotes), [`EVAL-TRUST.md`](https://github.com/gittensor-ai-lab/sparkinfer-k3/blob/main/EVAL-TRUST.md) (the deterministic-vs-non-deterministic trust split).
* **The hardening PRs** — [pull requests labeled `eval:*` and the `eval:`/`fix(eval):` series](https://github.com/gittensor-ai-lab/sparkinfer-k3/pulls?q=is%3Apr+is%3Aclosed) on `gittensor-ai-lab/sparkinfer-k3`. Reading twenty of them in order is the best available substitute for making the mistakes yourself.
* **KL divergence and top-1 agreement as a quantization gate** — llama.cpp's `--kl-divergence` / `--kl-divergence-base`; derived in [Logprobs, Perplexity & KL Divergence — Lecture 05](../../Logprobs,%20Perplexity%20and%20KL%20Divergence/Lecture-05.md).
* **"Which Quantization Should I Use?"** — [arXiv:2601.14277](https://arxiv.org/abs/2601.14277) — the systematic finding that intrinsic metrics (PPL, mean KLD) are *necessary but not sufficient*; a useful caution on §2's narrow definition of correctness.
* **Goodhart's law**, in its original form: when a measure becomes a target, it ceases to be a good measure. Every PR in §6 is an instance.

Cross-references:

* [Phase 5 → ML Systems Engineering Guide → Stage 0: Measurement Discipline](../../Guide.md#stage-0-measurement-discipline) — the stage this lecture is the field report for.
* [Part 1 Lecture 01 — The 2026 inference engineer's mental model](../Part%201%20-%20Fundamentals/Lecture-01.md) — TTFT / TPOT / throughput definitions assumed here.

---

## Current as of 2026-08

SparkInfer-K3 at `7689cc7`. Gate: top-1 ≥ 0.95, mean KL ≤ 0.05, seven depths, worst-of. Significance 2% of frontier. Tiers `xs` <3.5% / `s` 3.5–6% / `m` 6–10% / `l` 10–18% / `xl` >18%. Scored metric prefill @ 32k with the llama anchor disabled; decode @ 128k as a 1% regression guard. Receipt verification off (`REQUIRE_EVAL_RECEIPT` unset). The *rules* are the durable content here; the thresholds are one project's calibration.

---

## Next

* Next: [Lecture 03 — Diagnosis: launch-bound, bandwidth-bound, or comm-bound?](Lecture-03.md)
* Previous: [Lecture 01 — The workload, the baseline, and the ladder](Lecture-01.md)
* Up: [Part 4 — Optimizing a Real Engine](README.md)
