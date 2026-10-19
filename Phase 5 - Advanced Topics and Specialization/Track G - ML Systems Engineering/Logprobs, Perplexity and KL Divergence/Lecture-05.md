# Lecture 05 - Where It All Lands: Quantization, Distillation, RLHF & Speculative Decoding

**Collection:** [Logprobs, Perplexity & KL Divergence](README.md) | **Previous:** [← Lecture 04](Lecture-04.md) | **Next:** [Course index](README.md)

---

The preceding lectures connect cross-entropy, perplexity, and KL. This lecture applies them to quantization, distillation, alignment, and speculative decoding. These techniques use different distributions and objectives: do not substitute a reference-model KL for data NLL, or treat a small numerical drift as a complete task-quality verdict.

Quantization can measure drift from a reference model. Distillation may train against teacher distributions. Some alignment methods regularize policy drift; the precise penalty depends on the objective and configuration. Exact speculative sampling preserves the target distribution under its acceptance/correction assumptions, while runtime and draft quality determine whether it accelerates execution.

The throughline for a senior engineer: **`p` is always the distribution you trust, `q` is always the distribution you can afford.** Data, teacher, FP16 reference, reference policy, target model — those are `p`. Student, quant, tuned policy, draft model — those are `q`. Read every section below by asking *which `p`, which `q`, and is the KL being minimized or measured?*

---

## Learning objectives

By the end of this lecture, you should be able to:

1. State, for each of the four techniques, **which distribution is `p`, which is `q`, and whether the KL is being minimized (a training objective) or measured (a quality gate)**.
2. Grade a quantization the way llama.cpp's `--kl-divergence` mode does — **mean/median `D_KL`, token-probability RMS, top-1/top-k agreement, and PPL ratio** — and explain why PPL alone is "rough."
3. Cite the **2026 finding** that quantization *format* matters more than nominal bit-width and that intrinsic metrics are necessary but **not sufficient**, and translate it into a ship/no-ship rule.
4. Derive knowledge distillation as **forward-KL matching at temperature `T`**, explain "dark knowledge" and the `T²` gradient scaling, and say why forward KL makes the student *mode-covering*.
5. Derive the optimal unrestricted policy for a KL-regularized reward objective, then distinguish that derivation from finite-data DPO training and configurable PPO/GRPO regularization.
6. Prove that **speculative decoding's accepted samples are distributed exactly as the target `p`**, and relate acceptance length to the draft-target KL gap.
7. Build a **capstone quant grader**: FP16 reference vs quantized model over WikiText-2, reporting the full metric panel plus a thresholded `verdict()`.

---

## 1. The identity as a toolkit

The whole course collapses to one line you should be able to write blind:

```text
   H(p, q)   =   H(p)   +   D_KL(p ‖ q)          and          PPL = exp(H(p, q))
   ───────       ────       ────────────
   what q pays   the floor   the avoidable waste — what every
   per token     (p's own    technique below either MINIMIZES
   (mean −log q) entropy)    (training) or MEASURES (a gate)
```

For fixed prefixes, teacher-reference KL is `E_p_reference[log p_reference − log q]`. Held-out NLL change is `E_data[log p_reference − log q]`. The expectations differ. A quantized model can improve finite-sample data PPL while having positive teacher KL. The [CPU lab](Lab-01-Masked-Quality-Metrics.md) supplies a two-token counterexample. Report both measurements with their distributions, masks, and denominators.

Here is the map for the rest of the lecture — four techniques, one column each:

| Technique | `p` (trusted) | `q` (cheap) | The KL | Minimized or measured? |
|-----------|---------------|-------------|--------|------------------------|
| Quantization grading | FP16 logits | INT4/INT8 logits | `D_KL(p_fp16 ‖ q_quant)` | **measured** (a gate) |
| Knowledge distillation | teacher | student | `D_KL(p_teacher ‖ q_student)` at `T` | **minimized** (the loss) |
| RLHF / alignment | reference policy `π_ref` | tuned policy `π_θ` | `D_KL(π_θ ‖ π_ref)` | **constrained** (a leash) |
| Speculative decoding | target model `p` | draft model `q` | the `p`/`q` gap (KL-like) | **neither** — lossless; gap sets *speed* |

> **Hardware lens:** every metric in this lecture is a **gate before something ships** — a quant, a distilled student, a tuned policy, a draft head. Crucially, computing the gate is a **forward pass**, not a training run: you stream the weights once over an eval set and read logprobs off the output. That is *orders of magnitude* cheaper than the full downstream benchmark suite, which is why these intrinsic numbers are the first filter — they kill the obviously-broken candidates before you spend GPU-days on MMLU/GSM8K/IFEval.

---

## 2. Quantization grading — the deepest gate

Quantization replaces FP16 weights with a lower-precision format (INT8, INT4, the K-quants and I-quants in llama.cpp) to shrink memory footprint and bandwidth — the dominant cost in memory-bound decode. The question every time: **did it break the model, or just dent it?** The instinct is to compute perplexity on both and compare. That works, but it is *rough*, and understanding *why* is the heart of this course.

### 2.1 Why PPL ratio is only a proxy

PPL is a **single scalar averaged over the whole corpus**. It collapses the entire next-token distribution at every position into one number (`exp` of mean NLL on the *held-out token*). Two failure modes hide inside a near-equal PPL:

- The quant can be **right on average but wrong in the tails** — it keeps the top token's probability about right (so NLL on the actual next token barely moves) while badly mangling the *rest* of the distribution. PPL never looks at the mass it put on tokens that didn't occur.
- The quant can **trade errors that cancel** across positions — over-confident here, under-confident there — leaving the mean untouched while the per-token distribution is visibly degraded.

KL divergence does not collapse this way: `D_KL(p_fp16 ‖ q_quant)` is computed over the **full distribution at every token**, then averaged. It sees the tails. That is the whole reason llama.cpp added a KL mode.

### 2.2 What `llama.cpp --kl-divergence` reports

The workflow is two passes. First, run the **FP16 reference** over a corpus (WikiText-2 by convention) with `--kl-divergence-base` to dump its per-token logits to a file. Then run the **quantized** model with `--kl-divergence`, pointing at that base file; it streams the quant's logits, aligns them token-for-token with the cached FP16 distribution, and reports:

| Metric | Definition | What it tells you |
|--------|-----------|-------------------|
| **Mean `D_KL(p_fp16 ‖ q_quant)`** | average over tokens of `Σ_v p_fp16(v)·log(p_fp16(v)/q_quant(v))` | overall distributional drift; the headline number |
| **Median `D_KL`** | the 50th-percentile per-token KL | typical-case drift, robust to a few pathological tokens |
| **Max / 99th-pct `D_KL`** | worst per-token KL | catches a quant that is fine on average but catastrophic on rare tokens |
| **RMS Δ token-prob** | `sqrt(E[(q_quant(x) − p_fp16(x))²])` on the realized token | **the std of the Gaussian noise quantization injects onto token probabilities** — a directly interpretable "how jittery did probs get" |
| **Top-1 agreement %** | fraction of tokens where `argmax q_quant == argmax p_fp16` | how often greedy decoding picks the *same* token — tracks perceived quality closely |
| **Top-k agreement %** | fraction where the FP16 top-1 is within the quant's top-k | softer agreement; useful when sampling, not greedy |
| **PPL ratio** | `PPL(quant) / PPL(fp16)` | the classic proxy; keep it, but don't trust it alone |

The RMS framing is the one to internalize: quantization is, to first order, **additive noise on the logits**, and the RMS Δ token-prob is literally the standard deviation of that noise after it propagates through softmax. A quant with RMS Δ of 0.002 is whispering; one at 0.02 is shouting.

### 2.3 Scope empirical quantization results correctly

[Which Quantization Should I Use?](https://arxiv.org/abs/2601.14277v1), submitted January 11, 2026, evaluates llama.cpp formats on Llama-3.1-8B-Instruct using task scores, perplexity, CPU throughput, and storage. The study covers one model/runtime setup, not evidence that a single KL threshold or format ranking generalizes to all models and GPUs.

Use numerical checks to catch conversion defects, then evaluate representative held-out tasks and hardware performance. Predeclare tolerances from your use case. Compare rare failures as well as means. A small average PPL or KL difference does not certify reasoning, tool use, or robot action quality; neither metric is itself a direct measurement of those outcomes.

---

## 3. Knowledge distillation — minimizing the teacher-student KL

Distillation (Hinton, Vinyals & Dean, 2015) trains a small **student** `q` to imitate a large **teacher** `p`. The student doesn't learn from the hard one-hot labels alone; it learns from the teacher's **full soft distribution**, which carries far more information per example.

### 3.1 The objective is forward KL at temperature `T`

Soften both distributions with temperature `T` (divide logits by `T` before softmax), then match:

```text
   L_distill  =  D_KL( p_teacher^(T)  ‖  q_student^(T) )
              =  Σ_v  p_T(v) · [ log p_T(v) − log q_T(v) ]

   where  p_T(v) = softmax(z_teacher / T)[v],   q_T(v) = softmax(z_student / T)[v]
```

This is **forward KL** — `p` (teacher) in the first slot. Because `D_KL(p ‖ q) = H(p, q) − H(p)` and `H(p)` (the teacher's entropy) is fixed during student training, minimizing this KL is identical to minimizing the **cross-entropy of the student against the teacher's soft labels** — the exact same `H(p, q)` from Lecture 02, now with the teacher playing the role of "the data."

### 3.2 Dark knowledge

The reason soft targets beat hard labels is **"dark knowledge"**: the *relative* probabilities the teacher assigns to the **wrong** classes. A teacher that says "dog 0.9, wolf 0.08, cat 0.0002, car 1e-9" has told the student that this image is *much* more wolf-like than cat-like and nothing like a car — a rich similarity structure that a one-hot label `dog=1` annihilates. Temperature `T > 1` *amplifies* this signal by flattening the distribution, pulling the tiny logits up into a range where their ratios carry gradient. For LLMs the analogue is the full next-token distribution: the teacher's ranking of *plausible continuations* is the dark knowledge, and matching it is why a distilled model can feel far more capable than its parameter count suggests.

### 3.3 The `T²` gradient-scaling note

When you soften logits by `T`, the gradient of the soft-target cross-entropy w.r.t. the student logits scales like `1/T²`. So if you blend distillation with a hard-label loss, you **multiply the distillation term by `T²`** to keep the two gradients on a comparable scale across temperatures:

```text
   L = α · T² · D_KL(p_T ‖ q_T)   +   (1 − α) · CE(hard_label, q_{T=1})
       └──────── soft, scaled ────┘       └──── hard, unscaled ────┘
```

Forget the `T²` and your distillation signal silently shrinks as you raise `T`, exactly when you wanted it to matter more.

### 3.4 Forward KL → mode-covering

From Lecture 04: **forward KL `D_KL(p ‖ q)` is mode-covering (zero-avoiding).** Wherever the teacher `p` puts mass, the term `p(v)·log(p(v)/q(v))` blows up if the student `q(v) → 0`, so the student is *forced* to keep probability everywhere the teacher does — it **smooths over the teacher's mass** rather than collapsing onto the single most likely continuation. That is usually what you want from a generative student (coverage, diversity, calibrated tails), and it is the opposite of the reverse-KL, mode-*seeking* behavior you would get if you swapped the arguments — a contrast that becomes the entire story in §4.

For the engineering practice — data pipelines, intermediate-feature matching, when distillation beats pruning or quantization for a given compression budget — see the compression treatment in [Practical Machine Learning (CS329P) — Lecture 10](../../../Phase%203%20-%20Artificial%20Intelligence/Track%20B%20-%20Agentic%20AI%20and%20ML%20Engineering/4.%20ML%20Engineering%20and%20MLOps/Practical%20Machine%20Learning%20%28CS329P%29/Lecture-10.md), which *uses* this KL without deriving it. This section is the derivation it points back to.

---

## 4. RLHF / alignment — reward minus a KL leash

Reinforcement learning from human feedback fine-tunes a policy `π_θ` to maximize a learned reward `r(x, y)`. Left unconstrained, the optimizer **reward-hacks**: it finds degenerate outputs that score high under the imperfect reward model — repetition, sycophancy, exploit phrases, eventually gibberish — and collapses the distribution. The fix is a **KL leash** to the reference (pre-RLHF) policy `π_ref`:

```text
   maximize_θ   E_{x, y∼π_θ}[ r(x, y) ]   −   β · D_KL( π_θ(·|x)  ‖  π_ref(·|x) )
                └──── chase reward ────┘        └──── but don't stray from the trusted model ────┘
```

Here the KL direction is `D_KL(π_θ ‖ π_ref)`: the tuned policy is in the expectation slot. A positive coefficient penalizes divergence but does not guarantee safety, prevent every form of reward hacking, or establish task competence. Evaluate those properties separately.

### 4.1 DPO is the closed-form optimum of exactly this objective

The elegant result (Rafailov et al., 2023): the **KL-regularized reward objective has a closed-form optimal policy**, and inverting it lets you skip the RL loop entirely. The optimum is the reference policy *reweighted* by exponentiated reward:

```text
   π*(y|x)  =  (1/Z(x)) · π_ref(y|x) · exp( r(x, y) / β )         ← optimum of the KL-leashed objective

   ⇒ solve for the implicit reward:   r(x, y)  =  β · log( π*(y|x) / π_ref(y|x) )  +  β·log Z(x)
```

Substituting the optimal-policy relation into a Bradley–Terry preference model leads to the DPO objective on preference pairs. The closed form describes an unrestricted policy for a specified reward and reference; fitting a parameterized language model on finite preference data remains an optimization problem. DPO training does not automatically achieve that ideal optimum. See the [original DPO paper](https://arxiv.org/abs/2305.18290).

### 4.2 Inspect the actual alignment objective

| Method | Relationship to reference KL |
|---|---|
| PPO used for RLHF | Often adds a reference-policy KL reward penalty; PPO itself is a policy optimization algorithm and does not require this specific penalty. |
| DPO | Derives a preference objective using a KL-regularized reward formulation and reference log-probability ratios. |
| GRPO | May include an explicit reference KL term; configuration and implementation decide whether it is active. |

As checked 2026-10-18, [TRL's GRPO documentation](https://huggingface.co/docs/trl/grpo_trainer) specifies `beta=0.0` by default, disabling the KL penalty. Record the actual version, coefficient, estimator, token/sequence normalization, and masks. There is no invariant nonzero “KL leash” across every alignment method or run.

---

## 5. Speculative decoding — why it's *lossless*

Speculative decoding is the odd one out: it involves a draft model `q` and a target model `p`, the gap between them is KL-like, and yet **the output distribution is provably identical to the target's**. No approximation, no quality loss — only speed. Here is why.

### 5.1 The acceptance rule is a logprob-ratio test

A cheap **draft** `q` proposes a token `x`. The expensive **target** `p` scores it. Accept or repair:

```text
   draft proposes x ~ q(·)
   accept x   with probability   min(1, p(x) / q(x))
   on reject: resample x' ~ norm( max(0, p − q) )      ← the normalized residual
```

The accept probability `min(1, p(x)/q(x))` is a **ratio of probabilities** — i.e., a difference of logprobs, `log p(x) − log q(x)`, thresholded at zero. This course has been about reading exactly that quantity. The genius is the **repair on rejection**: when you reject, you don't fall back to the target's full distribution (that would double-count the mass the draft already got right) — you resample from `max(0, p − q)`, the mass the target wanted that the draft *under*-proposed, renormalized.

### 5.2 Proof that accepted samples are distributed exactly as `p`

```text
   P(output = x) = P(accept on the draft) + P(reject, then resample to x)

   ① draft proposes x and it is accepted:
        q(x) · min(1, p(x)/q(x))  =  min(q(x), p(x))

   ② draft proposes ANY token, it is rejected, then the residual resamples to x:
        P(reject) · norm(max(0, p−q))[x]
        P(reject) = Σ_y q(y)·(1 − min(1, p(y)/q(y))) = Σ_y (q(y) − min(q(y),p(y))) = Σ_y max(0, q(y)−p(y))
        the residual max(0, p−q) sums to  Σ_y max(0, p(y)−q(y))  = the SAME total (both equal 1 − Σ min(p,q))
        ⇒ the P(reject) factor and the residual's normalizer CANCEL, leaving exactly  max(0, p(x) − q(x))

   total:  min(q(x), p(x)) + max(0, p(x) − q(x))  =  p(x)        ∎   (for ANY draft q)
```

The two cases sum to `p(x)` by the algebraic identity `min(a,b) + max(0, b−a) = b`. The draft `q` **cancels out completely** — it appears nowhere in the result. That is the formal content of "lossless": for *any* draft, good or garbage, the emitted token is distributed exactly as the target. A bad draft never corrupts the output; it only wastes work.

### 5.3 The KL-like gap sets the *speed*, not the quality

So what does the draft quality buy you? **Acceptance rate**, and therefore **expected accept-length** (tokens emitted per target forward pass). The closer `q` is to `p`, the more often `min(1, p/q) = 1` and the token survives — a **better-aligned draft → higher acceptance → more tokens per memory pass → more speedup**. The governing gap is exactly a divergence between draft and target: where `q` matches `p`, acceptance is high; where `q` diverges, you eat rejections and re-drafts. This is the one place in the lecture where the KL is **neither minimized nor measured as a gate** — it is the lever on *throughput*, with quality nailed to the target by the proof above.

This is the algorithm's-eye view of what you met as a *systems* technique in [MLSys Deep Dives — Lecture 06](../MLSys%20Deep%20Dives/Lecture-06.md) (EAGLE-3, DFlash, the verify-kernel stack, and acceptance length as the governing metric), and as an *on-device measurement* in the [Gemma 4 Edge Deployment — Lecture 06](../../Track%20C%20-%20Edge%20AI/Gemma%204%20Edge%20Deployment/Lecture-06.md) MTP-drafter lecture, where acceptance length is profiled on real silicon. Same acceptance rule, three altitudes: the math here, the systems there, the hardware there.

> **Hardware lens:** draft/reference distribution comparisons can help select candidates, but acceptance and speed are different measurements. Exact single-token speculative acceptance depends on distribution overlap, `Σ min(p,q)=1−TV(p,q)`. End-to-end speed also depends on draft cost, verification cost, accepted length, batch size, and hardware. Measure each; high top-1 agreement alone does not guarantee useful speedup.

---

## 6. Capstone — build a quant grader

Tie it together. Load an FP16 **reference** and a **quantized** model, run both over WikiText-2, and report the panel from §2 plus a thresholded verdict. Realistic Hugging Face + PyTorch; the only subtlety is a correct sliding window (Lecture 03) and caching the reference logits.

```python
import torch, torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset

REF_ID   = "meta-llama/Llama-3.1-8B"          # p  — FP16 reference (trusted)
QUANT_ID = "path/to/llama-3.1-8b-q4_k_m"      # q  — quantized model (cheap)
STRIDE, WINDOW = 512, 2048                      # sliding-window PPL (Lecture 03)

def load(model_id, dtype):
    tok = AutoTokenizer.from_pretrained(model_id)
    m   = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=dtype,
                                               device_map="cuda").eval()
    return tok, m

@torch.no_grad()
def grade(ref_id=REF_ID, quant_id=QUANT_ID):
    tok, ref   = load(ref_id, torch.float16)
    _,   quant = load(quant_id, torch.float16)   # quantized weights, fp16 compute
    text = "\n\n".join(load_dataset("wikitext", "wikitext-2-raw-v1", split="test")["text"])
    ids  = tok(text, return_tensors="pt").input_ids.cuda()

    nll_ref = nll_q = kl_sum = rms_sq = 0.0
    top1_hits = n_tok = 0
    prev = 0
    for begin in range(0, ids.size(1), STRIDE):
        end = min(begin + WINDOW, ids.size(1))
        trg_len = end - prev                       # only score the NEW tokens (no double count)
        chunk   = ids[:, begin:end]
        labels  = chunk.clone(); labels[:, :-trg_len] = -100

        # --- one forward pass each; the reference logits are computed once here and reused ---
        lr = ref(chunk).logits[0, :-1].float()     # p over the vocab, per position
        lq = quant(chunk).logits[0, :-1].float()   # q over the vocab, per position
        tgt = chunk[0, 1:]                          # the realized next tokens
        mask = labels[0, 1:] != -100                # positions we actually score

        logp = F.log_softmax(lr[mask], dim=-1)      # log p
        logq = F.log_softmax(lq[mask], dim=-1)      # log q
        p, q = logp.exp(), logq.exp()
        gold = tgt[mask]

        # cross-entropies on the realized token -> the two PPLs
        nll_ref += -logp[torch.arange(gold.numel()), gold].sum().item()
        nll_q   += -logq[torch.arange(gold.numel()), gold].sum().item()
        # mean D_KL(p_fp16 || q_quant): full-distribution, summed over vocab then tokens
        kl_sum  += (p * (logp - logq)).sum().item()
        # RMS token-prob change on the realized token (the Gaussian-noise std)
        rms_sq  += ((q[torch.arange(gold.numel()), gold]
                     - p[torch.arange(gold.numel()), gold]) ** 2).sum().item()
        # top-1 agreement: same argmax?
        top1_hits += (lr[mask].argmax(-1) == lq[mask].argmax(-1)).sum().item()
        n_tok     += gold.numel()
        prev = end
        if end == ids.size(1): break

    import math
    return {
        "ppl_fp16":      math.exp(nll_ref / n_tok),
        "ppl_quant":     math.exp(nll_q   / n_tok),
        "ppl_ratio":     math.exp(nll_q / n_tok) / math.exp(nll_ref / n_tok),
        "mean_kld":      kl_sum / n_tok,                  # nats; D_KL(p_fp16 || q_quant)
        "top1_agree_pct": 100.0 * top1_hits / n_tok,
        "rms_dprob":     math.sqrt(rms_sq / n_tok),       # std of injected prob noise
    }

def verdict(m, kld_ship=0.02, kld_warn=0.10, top1_ship=99.0, ppl_warn=1.01):
    """Intrinsic gate only — necessary, NOT sufficient (see 2026 finding)."""
    if m["mean_kld"] <= kld_ship and m["top1_agree_pct"] >= top1_ship \
                                  and m["ppl_ratio"] <= ppl_warn:
        return "SHIP (intrinsic clean) — still confirm on a reasoning/instruction eval"
    if m["mean_kld"] >= kld_warn or m["top1_agree_pct"] < 95.0:
        return "REJECT — quant broke the distribution (high KLD / low top-1 agreement)"
    return "INVESTIGATE — borderline; run GSM8K/MMLU-Pro/IFEval before deciding"

if __name__ == "__main__":
    m = grade()
    for k, v in m.items():
        print(f"{k:>16}: {v:.4f}")
    print("verdict:", verdict(m))
```

The thresholds are deliberately conservative starting points (`mean_kld` in nats; tune per model family). Note what `verdict()` does **not** claim: a "SHIP" is an *intrinsic* pass that explicitly tells you to confirm downstream — the §2.3 lesson encoded in code. The grader is the fast filter; the reasoning eval is the verdict.

> **Hardware lens:** this entire grader is **two forward passes over a few hundred KB of text** — minutes on one GPU — versus GPU-*days* for a full benchmark suite. That asymmetry is why intrinsic grading is the first gate in every serious quantization or drafting pipeline: cache the FP16 reference logits once, sweep every candidate quant/draft against them cheaply, and only escalate the survivors to the expensive downstream evals.

---

## Current as of

**June 2026.** The mathematics is settled (Shannon 1948; Kullback–Leibler 1951; Hinton et al. 2015; Rafailov et al. 2023). What is *current* is the quantization-evaluation practice, and it has converged hard this cycle:

- **The 2026 quant-eval findings (the headline).** Per **arXiv 2601.14277** ("Which Quantization Should I Use? A Systematic Evaluation of llama.cpp Quantization on Llama-3.1-8B," Jan 2026): **(1) format beats nominal bit-width** — rank quants by their packing scheme (K-quant/I-quant, importance-weighted, mixed-precision blocks), never by the number in the name; a good 4-bit format outperforms a sloppy higher-bit one. **(2) intrinsic metrics are necessary but not sufficient** — quants with near-equal PPL *and* near-equal mean KLD can still diverge on reasoning/instruction benchmarks, so the intrinsic panel is the fast filter and a downstream eval is the verdict. The field consensus is now **"KLD over PPL"** for the intrinsic gate: mean `D_KL(p_fp16 ‖ q_quant)` and top-1 agreement track perceived quality better than the PPL delta.
- **Tooling.** `llama.cpp`'s `--kl-divergence` / `--kl-divergence-base` flags report the full panel (mean/median/max KLD, RMS Δ token-prob, top-1/top-k agreement, PPL ratio); flag names and output formatting drift across releases — verify against your build.
- **Alignment.** Record the objective and configuration. Reference KL may be explicit, implicit in a derivation, or absent. A DPO optimal-policy derivation is not proof that finite-data training reaches that optimum.
- **Speculative decoding.** Losslessness is a theorem, not a trend — it does not age. The frontier (EAGLE-3, DFlash, and the MiMo + TileRT throughput milestone) is about driving the draft-target gap *down* to raise acceptance length; those throughput numbers are vendor-reported and tracked in [MLSys Deep Dives — Lecture 06](../MLSys%20Deep%20Dives/Lecture-06.md).
