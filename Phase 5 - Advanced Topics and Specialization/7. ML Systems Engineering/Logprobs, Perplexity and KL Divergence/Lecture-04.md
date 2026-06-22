# Lecture 04 - KL Divergence: The Gap Between Two Distributions

**Collection:** [Logprobs, Perplexity & KL Divergence](README.md) | **Previous:** [← Lecture 03](Lecture-03.md) | **Next:** [Lecture 05](Lecture-05.md)

---

Two lectures ago we earned the left side of the master identity — cross-entropy `H(p,q)`, the loss you train on, and entropy `H(p)`, the floor it can never beat. This lecture earns the third and final term: the **KL divergence** `D_KL(p ‖ q)`, the part of the identity that is nobody's fault but yours.

`H(p,q) = H(p) + D_KL(p ‖ q)`. Read it as an accounting statement. Entropy is the bill reality sends you — the data's own uncertainty, which no model can refund. KL divergence is the surcharge you add on top by using the wrong distribution. It is **non-negative, zero only when your model is exactly right, and entirely avoidable in principle.** Every cross-entropy you have ever measured was entropy plus this surcharge, and you cannot tell the two apart from the loss number alone — which is why a falling loss can mean "the data got easier" rather than "my model got better."

That is the abstract reading. The operational one is what makes KL the load-bearing metric of this phase: it is the **universal "distance from the model I trust."** Quantization asks "how far did INT4 drift from FP16?" Distillation asks "how far is the student from the teacher?" RLHF asks "how far has the tuned policy strayed from the base?" Speculative decoding asks "how close is the draft to the target?" All four are one number — `D_KL` between two next-token distributions — and the rest of this lecture builds it carefully enough that you can compute it, prove its properties, and know which of its two faces (forward or reverse) any given technique is actually optimizing.

---

## Learning objectives

By the end of this lecture, you should be able to:

1. Define `D_KL(p ‖ q)` and state its operational meaning: the **expected extra nats** you pay coding samples from `p` with a code built for `q`.
2. Prove the three properties that matter — `D_KL ≥ 0` (Gibbs/Jensen), `= 0` iff `p = q`, and **asymmetry** — and explain why KL is a *divergence*, not a distance.
3. Derive the master identity `D_KL(p ‖ q) = H(p,q) − H(p)` and conclude that **MLE is forward-KL minimization**, tying Lectures 02–03 together.
4. Distinguish **forward** (mass-covering, zero-avoiding) from **reverse** (mode-seeking, zero-forcing) KL, and map each common technique to the one it minimizes.
5. Reach for a **symmetric** cousin — Jensen–Shannon, total variation — when the application genuinely needs symmetry, and know JSD's bound and metric properties.
6. Compute **per-token KL** between two LM logit vectors in PyTorch without falling into the `F.kl_div` argument-order trap, and read mean corpus KLD as a single faithfulness number (full treatment in [Lecture 05](Lecture-05.md)).

---

## 1. Definition and meaning

For two distributions `p` and `q` over the same support, the **Kullback–Leibler divergence of `q` from `p`** is

```text
  D_KL(p ‖ q) = Σ_x p(x) log( p(x) / q(x) ) = E_p[ log p(x) − log q(x) ]
```

Three things to read off immediately:

- **The expectation is under `p`.** You weight every term by how often *reality* (the trusted distribution) produces `x`, not by what the model `q` believes. This is the entire source of the asymmetry in §2 — swap which distribution holds the pen and you get a different number.
- **It is a difference of log-probabilities, averaged.** `log p − log q` is the per-event surprise gap: at each outcome, how many more nats of surprise does `q` assign than the truth `p` does? Average that gap under `p` and you have the divergence.
- **Units follow the log.** Natural log gives nats; `log2` gives bits. We default to nats and write `bits = log2` only when comparing to a perplexity in bits-per-token.

The operational meaning is a coding statement, and it is the one to memorize. Shannon's source-coding theorem says the cheapest code for a source `p` uses about `−log p(x)` nats per symbol `x`, for an average of `H(p)`. Suppose you instead build your code assuming the distribution is `q` — so you spend `−log q(x)` nats on symbol `x` — but the symbols keep arriving from the real source `p`. Your average cost is now `H(p,q) = −Σ p log q`, the cross-entropy. The **excess** over the optimal `H(p)` is

```text
  H(p,q) − H(p) = E_p[ −log q ] − E_p[ −log p ] = E_p[ log p − log q ] = D_KL(p ‖ q).
```

So `D_KL(p ‖ q)` is the **expected number of extra nats per symbol you waste by coding `p`-data with a `q`-optimal code.** Equivalently — and this is the phrase to keep — it is *the surprise penalty for believing `q` when reality is `p`.* When `q = p` the penalty is zero: you built the right code. When `q` puts low probability on something `p` produces often, that term `p(x) log(p(x)/q(x))` blows up — you are paying dearly, in surprise, every time the event you under-modeled occurs. We just proved the master identity in passing; §3 does it again slowly, from the sums, because it is worth seeing twice.

---

## 2. Properties, with proofs

Four properties define how you are allowed to use KL. The first two you must be able to prove; the second two are what stop you from misusing it.

### (a) Non-negativity: `D_KL(p ‖ q) ≥ 0`

This is **Gibbs' inequality**, and it falls out of Jensen's inequality in three lines. Jensen says that for a *concave* function `φ` (and `log` is concave), `E[φ(Y)] ≤ φ(E[Y])`. Apply it to `Y = q(x)/p(x)` under the expectation taken over `p`:

```text
  −D_KL(p ‖ q) = E_p[ log( q(x) / p(x) ) ]              (flip the sign inside the log)

              ≤  log( E_p[ q(x) / p(x) ] )               (Jensen: log is concave)

               = log( Σ_x p(x) · q(x)/p(x) )             (write out the expectation)

               = log( Σ_x q(x) )                          (the p(x) cancels)

               = log( 1 ) = 0.                            (q is a distribution)

  Therefore  −D_KL(p ‖ q) ≤ 0,  i.e.  D_KL(p ‖ q) ≥ 0.   ∎
```

The cancellation `Σ_x p(x)·q(x)/p(x) = Σ_x q(x) = 1` is the whole trick: the `p`-weighting in the expectation is exactly what is needed to collapse the ratio back to `q`'s total mass. (Strictly, the sum runs over `x` where `p(x) > 0`; if `q(x) = 0` there, the ratio diverges and `D_KL = +∞` — a real possibility for LMs, addressed in §6.)

This single inequality is the backbone of the course. Combined with the identity of §3 it says `H(p,q) ≥ H(p)`: **cross-entropy can never beat entropy.** No model, however good, drives the loss below the data's own uncertainty floor, because the gap *is* a KL and KL is non-negative.

### (b) Zero iff equal: `D_KL(p ‖ q) = 0 ⟺ p = q` (almost everywhere)

Jensen's inequality is an *equality* exactly when the random variable inside is constant `p`-almost-everywhere. Here that variable is `q(x)/p(x)`, so equality requires `q(x)/p(x) = c` (constant) wherever `p(x) > 0`. Summing, `Σ q(x) = c·Σ p(x)`, i.e. `1 = c·1`, so `c = 1` and `q(x) = p(x)` on the support of `p`. KL is zero precisely when the two distributions coincide — there is no other way to score a perfect zero.

### (c) Asymmetry: `D_KL(p ‖ q) ≠ D_KL(q ‖ p)` in general

KL is **not symmetric**, and the difference is not a rounding error — the two orderings answer different questions and can differ by orders of magnitude. A minimal numerical witness over a two-event support:

```text
  p = (0.9, 0.1),  q = (0.5, 0.5)        (natural log)

  D_KL(p ‖ q) = 0.9·log(0.9/0.5) + 0.1·log(0.1/0.5)
              = 0.9·(0.5878)   + 0.1·(−1.6094)
              = 0.5290 − 0.1609 = 0.368 nats

  D_KL(q ‖ p) = 0.5·log(0.5/0.9) + 0.5·log(0.5/0.1)
              = 0.5·(−0.5878)  + 0.5·(1.6094)
              = −0.2939 + 0.8047 = 0.511 nats

  0.368 ≠ 0.511.                          ∎
```

The numbers differ because the averaging weights differ: `D_KL(p ‖ q)` weights the surprise gap by `p`, `D_KL(q ‖ p)` weights it by `q`. §4 turns this asymmetry into the single most consequential design distinction in the lecture.

### (d) Not a metric — so call it a *divergence*

A metric needs symmetry and the triangle inequality. KL has **neither**: it fails symmetry by (c), and there is no constant `c` for which `D_KL(p ‖ r) ≤ c·(D_KL(p ‖ q) + D_KL(q ‖ r))` holds in general. It can also be `+∞` while the distributions are clearly "close" in any visual sense (one zero in the wrong place). So it is a **divergence**: a non-negative, identity-of-indiscernibles measure of dissimilarity that is *not* a distance. Speak precisely — "the KL divergence between," never "the KL distance" — because the asymmetry is load-bearing, and §5 exists exactly for the cases where you do need a true symmetric distance.

---

## 3. The master identity, proven

We met the identity informally in §1 and in Lecture 02. Here it is from the definitions, term by term:

```text
  D_KL(p ‖ q) = Σ_x p(x) log( p(x) / q(x) )

              = Σ_x p(x) [ log p(x) − log q(x) ]              (log of a quotient)

              = Σ_x p(x) log p(x)  −  Σ_x p(x) log q(x)       (split the sum)

              = −H(p)              −  ( −H(p,q) )             (defs of H(p), H(p,q))

              = H(p,q) − H(p).                                 ∎

  Rearranged:   H(p,q) = H(p) + D_KL(p ‖ q).
```

Now the consequence that justifies the whole course. In supervised LM training, `p` is the **data distribution** — for next-token prediction it is effectively a one-hot delta on the observed next token, fixed by the dataset. `q` is your **model**. The training loss is the cross-entropy `H(p_data, q_model)` averaged over the corpus (Lecture 02). Split it with the identity:

```text
  loss = H(p_data, q_model) = H(p_data) + D_KL(p_data ‖ q_model)
                              └── fixed ──┘   └─── the only part you can move ───┘
```

`H(p_data)` does not depend on your parameters — it is a property of the data alone. **So minimizing cross-entropy loss is *exactly* minimizing the forward KL `D_KL(p_data ‖ q_model)`.** Maximum-likelihood estimation, the objective behind essentially every pretrained LM, *is* forward-KL minimization; the additive `H(p_data)` is just an offset you cannot influence and therefore ignore. This is why two of our three numbers are not really separate goals: drive the perplexity down (Lecture 03), and you are mechanically shrinking the forward KL from the data to your model. The metric you report and the divergence you minimize are the same act, viewed through the identity.

---

## 4. Forward vs reverse KL

Because KL is asymmetric (§2c), `D_KL(p ‖ q)` and `D_KL(q ‖ p)` are *different objectives with different solutions*, and choosing the wrong one quietly determines whether your model spreads out or collapses. Fix `p` as the trusted target and let `q` be the thing you optimize.

**Forward KL — `D_KL(p ‖ q)` — is mass-covering / zero-avoiding.** The expectation is under `p`, so every `x` with `p(x) > 0` contributes a term `p(x) log(p(x)/q(x))`. If `q(x) → 0` where `p(x) > 0`, that term `→ +∞`. The optimizer is therefore **terrified of putting zero mass anywhere `p` has mass** — it would rather smear `q` thin to cover all of `p`'s support than leave a hole. Where `p(x) = 0`, forward KL is indifferent to `q(x)` (the term is `0·log0 = 0`), so `q` is free to leak mass into empty regions. Net effect: `q` becomes **broad, averaging, mode-covering.**

**Reverse KL — `D_KL(q ‖ p)` — is mode-seeking / zero-forcing.** Now the expectation is under `q`, and the penalized term is `q(x) log(q(x)/p(x))`. Wherever `p(x)` is small but `q(x)` is not, the ratio is large and you pay. The cheapest escape is to **set `q(x) = 0` there** — `0·log0 = 0` costs nothing. So reverse KL aggressively forces `q` to zero outside `p`'s high-density regions, and `q` collapses onto **one mode**, ignoring the rest of `p` entirely.

The classic intuition is a **bimodal `p`** (two separated bumps) approximated by a single Gaussian `q`:

```text
   p:   /\        /\          two modes, a valley between

   forward  D_KL(p ‖ q):   q straddles BOTH bumps — one wide, flat
                           Gaussian centered in the valley, covering
                           all of p's mass (and the empty middle too).
                           "Cover everything p does."

   reverse  D_KL(q ‖ p):   q snaps onto ONE bump — a narrow Gaussian
                           sitting on a single mode, ignoring the other.
                           "Be confidently right somewhere p is."
```

Neither is "correct" — they encode different risk preferences. Forward KL never abandons a region the truth visits (good for a generative model that must not assign zero probability to real data); reverse KL produces sharp, decisive, sometimes overconfident behavior (good when you want the policy to commit). Which one a technique uses is a design fact worth memorizing:

| Technique | Minimizes | Personality | Why |
|---|---|---|---|
| **MLE / pretraining** | forward `D_KL(p_data ‖ q)` | mass-covering | §3 — it *is* the cross-entropy loss; cannot zero out real tokens |
| **Knowledge distillation** | forward `D_KL(p_teacher ‖ q_student)` | mass-covering | student matches the teacher's full soft distribution (Lecture 05) |
| **Label smoothing / forward-KL fine-tune** | forward | mass-covering | same family as MLE against a softened target |
| **Variational inference (ELBO)** | reverse `D_KL(q ‖ p_post)` | mode-seeking | tractable: expectation under the `q` you control |
| **RLHF / policy optimization (PPO, GRPO)** | reverse `D_KL(q_policy ‖ p_ref)` | mode-seeking | KL-regularize the *sampling* policy toward the reference (Lecture 05) |

A useful mnemonic: you minimize **forward** KL when you must not *miss* anything the target does (coverage), and **reverse** KL when you must not *invent* anything the target doesn't (concentration). Pretraining and distillation are coverage problems; alignment and variational approximation are concentration problems.

---

## 5. Symmetric cousins

Sometimes the application genuinely wants a single symmetric number — "how different are these two distributions?" with no privileged reference. KL refuses to answer that question (it always privileges the first argument). Three standard tools do.

**Jensen–Shannon divergence (JSD)** symmetrizes KL by routing both through their mixture `m = ½(p + q)`:

```text
  m = ½(p + q)

  JSD(p, q) = ½ D_KL(p ‖ m) + ½ D_KL(q ‖ m)
```

Its properties are exactly the ones KL lacks. It is **symmetric** by construction: swapping `p` and `q` leaves `m` unchanged and swaps the two terms. It is **always finite** — because `m(x) ≥ ½ p(x)` and `m(x) ≥ ½ q(x)`, neither ratio inside can diverge even if `p` and `q` have disjoint support (the failure mode that sends raw KL to `+∞`). It is **bounded**: `0 ≤ JSD(p, q) ≤ log 2` in nats (`≤ 1` in bits), reaching `log 2` exactly when `p` and `q` have disjoint support. And critically, **`√JSD` is a true metric** — it satisfies symmetry and the triangle inequality — so when you need to *cluster*, *embed*, or *threshold* distributions with a real distance, `√JSD` is the principled choice where KL is not.

**Total variation distance (TV)** is the other workhorse, and the most intuitive:

```text
  TV(p, q) = ½ Σ_x | p(x) − q(x) |
```

TV is a genuine metric, lives in `[0, 1]`, and reads directly as "the largest gap in probability either distribution assigns to any event." It is linked to KL by **Pinsker's inequality**, `TV(p, q) ≤ √( ½ D_KL(p ‖ q) )`, which lets a KL bound certify a TV bound — useful when a worst-case probability gap is what you actually care about.

When to reach for which: use **plain KL** whenever there is a privileged reference and you want the coding/surprise interpretation — quantization (FP16 is the reference), distillation (the teacher is the reference), RLHF (the base policy is the reference). That covers almost everything in this phase, which is why the rest of the course is KL, not JSD. Reach for **JSD or TV** only when the two distributions are genuinely peers (comparing two independently trained models, A/B-ing two checkpoints with no "ground truth," or needing a bounded, symmetric, never-infinite score for a dashboard).

---

## 6. KL between two LM next-token distributions

Now the concrete object this course is built to compute. At a single position, a language model emits a logit vector over the vocabulary; softmax turns it into a next-token distribution. Given two models — a **reference** `p` (say FP16) and a **candidate** `q` (say INT4) — looking at the *same* context, the per-token KL is the KL between their two vocabulary distributions:

```text
  per-token  D_KL(p ‖ q) = Σ_{v ∈ vocab} p(v) log( p(v) / q(v) )
```

with `p = softmax(reference_logits)` and `q = softmax(candidate_logits)`. Average this over every token position in a corpus and you get **mean KLD** — a single scalar that says, in nats per token, how far the candidate's next-token beliefs drift from the reference's across real text. This is precisely the faithfulness number llama.cpp reports to grade quantization; Lecture 05 gives it the full treatment (alongside top-token agreement and probability RMS). One subtlety the formula makes explicit: if the candidate assigns *zero* to a token the reference finds plausible, that term is `+∞`. Real softmax output is never exactly zero, but after low-bit quantization or aggressive top-k truncation a token can get close enough that the per-token KL spikes — which is informative, not a bug: it is the metric loudly flagging a token the quant abandoned.

In PyTorch, the function is `F.kl_div`, and it has a notorious calling convention that is the single most common source of wrong KL numbers:

```python
import torch
import torch.nn.functional as F

# ref_logits, cand_logits: shape [T, V] — one logit row per token position.
# Convention here: p = reference (trusted), q = candidate, computing D_KL(p ‖ q).

def per_token_kl(ref_logits: torch.Tensor, cand_logits: torch.Tensor) -> torch.Tensor:
    """Returns a length-T tensor of D_KL(p ‖ q) in NATS, one value per position."""
    log_p = F.log_softmax(ref_logits,  dim=-1)   # log p  (reference)
    log_q = F.log_softmax(cand_logits, dim=-1)   # log q  (candidate)
    p     = log_p.exp()                          # p
    # D_KL(p ‖ q) = Σ p (log p − log q), summed over vocab, per token:
    return (p * (log_p - log_q)).sum(dim=-1)     # [T]

# --- The same thing via F.kl_div, demonstrating the two gotchas ---
# GOTCHA 1 (argument order): F.kl_div(input, target) computes
#     Σ target * (log target − input),
# i.e. it expects INPUT = log q (the model/candidate, already log'd) and
#      TARGET = p (the reference). The order is (log-q, p) — *backwards* from
#      how we write D_KL(p ‖ q). Passing (log_p, q) silently computes the
#      REVERSE KL and you will never see an error, only a wrong number.
# GOTCHA 2 (log_target): by default target is taken as a PROBABILITY. If your
#      target is already in log-space, you MUST pass log_target=True, or kl_div
#      will exp() it a second time and return garbage.

def per_token_kl_via_F(ref_logits, cand_logits):
    log_p = F.log_softmax(ref_logits,  dim=-1)   # log p  (reference / TARGET)
    log_q = F.log_softmax(cand_logits, dim=-1)   # log q  (candidate / INPUT)
    # input = log_q, target = log_p, reduction='none', log_target=True:
    kl = F.kl_div(log_q, log_p, reduction="none", log_target=True)  # [T, V]
    return kl.sum(dim=-1)                          # [T]  == per_token_kl(...)

# mean corpus KLD — the single faithfulness number (Lecture 05):
#   mean_kld = torch.cat([per_token_kl(r, c) for r, c in batches]).mean()
```

The takeaway from the gotchas: `F.kl_div`'s first argument is the **log of `q`** and its second is **`p`**, the opposite of the `D_KL(p ‖ q)` reading order, and `log_target=True` is mandatory when your target is already a log-prob. When in doubt, compute it explicitly as `(p * (log_p - log_q)).sum(-1)` — it is the same arithmetic, reads exactly like the math, and cannot be silently reversed.

> **Hardware lens:** computing KL is strictly heavier than computing perplexity, and the cost is *memory*, not flops. A plain PPL run only needs the candidate's log-prob at the *one* realized token per position — a scalar. KL needs the **entire vocabulary distribution from *both* models** at every position: two full `[T, V]` logit tensors, roughly **2× the logit memory/storage** of a PPL pass (and `V` is 32K–256K, so a full-corpus logit dump is large). The standard pattern is to compute and **cache the FP16 reference logits once** — write `log_softmax(ref_logits)` to disk for the whole eval corpus — then **stream each candidate quant against that cached reference**, so the expensive full-precision forward pass is paid a single time and every subsequent quant comparison is one cheap forward pass plus a vector op. This is exactly how `--kl-divergence` graders are built to stay tractable.

> **2026 update:** KL divergence — not perplexity alone — is now the preferred *intrinsic* metric for grading quantization. llama.cpp exposes it directly via `--kl-divergence`, and the field has converged on mean KLD (plus top-token agreement) correlating better with perceived quality than a PPL delta, which is "rough" by comparison (Lecture 05). The same term reappears across alignment: **PPO, DPO, and GRPO all carry an explicit KL penalty** toward a reference policy. RLHF objectives churn — the algorithm of the month changes — but the **KL term is the invariant across every one of them**, which is the whole reason this lecture sits where it does in the course. Lecture 05 turns both observations loose on real graders and real RLHF losses.

---

## Current as of

June 2026. The mathematics is settled (Kullback–Leibler 1951; Gibbs' inequality; Jensen). What is dated is the tooling: llama.cpp's `--kl-divergence` quant grader and the current consensus that mean KLD + top-token agreement track quality better than PPL alone, plus the RLHF landscape (PPO/DPO/GRPO) whose KL penalty is the stable core. Re-verify grader flags and the reigning policy-optimization method before quoting specifics; the identity and its proofs do not expire.
