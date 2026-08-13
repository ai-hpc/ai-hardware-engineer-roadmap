# Lecture 01 - Logprobs: What a Language Model Actually Emits

**Collection:** [Logprobs, Perplexity & KL Divergence](README.md) | **Previous:** [← Course index](README.md) | **Next:** [Lecture 02](Lecture-02.md)

---

A language model does not emit text. It emits, at every step, a probability distribution over its entire vocabulary — and the single number you read off that distribution for the token that actually appeared is the **log-probability**, the `log q(x_t | x_<t)`. Sampling, greedy argmax, beam search, the text in your terminal — all of it is downstream cosmetics applied to that distribution. The logprob is the model's own opinion, in its own currency, before any of those cosmetics run.

This is the atom of the entire course. Cross-entropy (Lecture 02) is the mean of these numbers. Perplexity (Lecture 03) is the exponential of that mean. KL divergence (Lecture 04) is a weighted difference of two of them. Quantization grading, distillation loss, the RLHF KL penalty, the speculative-decoding accept/reject test (Lecture 05) — every one of those is an aggregation of per-token logprobs. If you read logprobs fluently, the rest of the course is bookkeeping; if you don't, every later metric is a black box you can only trust on faith.

So this lecture does one thing exhaustively: it takes you from the raw logit vector the LM head produces, through softmax and the log, to the number you can extract from OpenAI, Hugging Face, vLLM, or llama.cpp — and it explains why the model, the kernels, and your evaluation harness all prefer to do their arithmetic in log-space rather than probability-space.

---

## Learning objectives

By the end of this lecture you should be able to:

1. Trace the LM-head pipeline `logits → softmax → probabilities → log → logprobs`, and explain why `log_softmax` fuses the last three steps into one numerically stable kernel.
2. Justify log-space on two independent grounds — **additivity** (joint sequence logprob = sum of token logprobs) and **numerical stability** (a long-sequence probability underflows float; its log does not) — and state the log-sum-exp trick.
3. Distinguish a **token logprob** from a **sequence logprob**, and apply **length normalization** (mean / per-token logprob) when comparing candidates of different lengths.
4. Predict the effect of **temperature** on logprobs and entropy, including the `T → 0` (argmax) and `T → ∞` (uniform) limits.
5. Extract logprobs from OpenAI, Hugging Face Transformers, vLLM, and llama.cpp, and explain what `top_logprobs` truncation costs you.
6. Name the systems that consume logprobs — confidence, ranking/best-of-n, guided decoding, evaluation, hallucination signals — and place the LM head correctly on the inference roofline.

---

## 1. From logits to logprobs

A transformer's body produces, for the current position, a single hidden vector `h ∈ R^{d}` where `d` is the model dimension (4096 for Llama-3-8B, 3584 for Qwen2.5-7B). The **LM head** — usually a single weight matrix `W_U ∈ R^{V × d}`, where `V` is the vocabulary size — projects that hidden vector to one real number per vocabulary entry:

```text
logits z = W_U · h          z ∈ R^{V}
```

`z` is the **logit vector**. Its entries are unbounded real numbers (positive and negative); they are *not* probabilities and do not sum to anything in particular. To turn them into a distribution `q(·)` over the vocab, apply the **softmax**:

```text
                 exp(z_i)
q_i = softmax(z)_i = ─────────────         (i indexes the vocabulary)
                    Σ_j exp(z_j)
```

The denominator `Z = Σ_j exp(z_j)` is the **partition function** / normalizer; it forces `Σ_i q_i = 1` and `q_i > 0`. Now take the log to get the **logprob** of entry `i`:

```text
log q_i = z_i − log Σ_j exp(z_j) = z_i − log Z
```

That last line is the whole game and worth staring at: **a logprob is just the token's own logit minus one shared normalizer.** Every token at a given step subtracts the *same* `log Z`. The expensive, distribution-wide part of the computation is that single scalar; everything else is the token's raw score.

Doing it as `softmax` then `log` is wasteful and numerically fragile (you exponentiate, normalize, then take a log — three chances to lose precision). Every framework therefore offers a fused **`log_softmax`** that computes `z_i − log Z` directly:

```text
log_softmax(z)_i = z_i − logsumexp(z)
logsumexp(z)     = log Σ_j exp(z_j)
```

The table below names the four quantities you will be juggling for the rest of the course.

| Quantity | Symbol | Shape | Range | Sums to |
|---|---|---|---|---|
| Hidden state | `h` | `[d]` | unbounded | — |
| Logits | `z = W_U h` | `[V]` | unbounded | — |
| Probabilities | `q = softmax(z)` | `[V]` | `(0, 1)` | `1` |
| Logprobs | `log q = log_softmax(z)` | `[V]` | `(−∞, 0]` | — |

Note the range of a logprob: it is `≤ 0` always, because a probability is `≤ 1` and `log` of something `≤ 1` is `≤ 0`. A logprob of `0` means probability `1` (certainty); a very negative logprob means a token the model considered nearly impossible.

---

## 2. Why log-space: additivity, stability, log-sum-exp

There are two independent reasons the model, the loss function, and your eval harness all live in log-space. Either one alone would justify it; together they make probability-space arithmetic a beginner's mistake.

**Reason 1 — additivity.** A language model factorizes the probability of a sequence by the chain rule, and probabilities of successive tokens **multiply**:

```text
q(x_1 ... x_T) = Π_{t=1}^{T} q(x_t | x_<t)
```

Take the log and the product becomes a **sum** — this is the single most useful identity in the course:

```text
log q(x_1 ... x_T) = Σ_{t=1}^{T} log q(x_t | x_<t)
```

The joint **sequence logprob is just the sum of the per-token logprobs.** Sums are cheap, differentiable, and stable; products of many small numbers are none of those. Every "score this sequence" operation — re-ranking, best-of-n, beam search — is an addition in log-space.

**Reason 2 — numerical stability / underflow.** Per-token probabilities are routinely small (an average token might sit around `q ≈ 0.05`). Multiply a hundred of them and you are deep below the floor of IEEE float:

```text
typical per-token prob ≈ 0.05
100-token sequence prob ≈ 0.05^100 ≈ 1e-130   →  underflows to 0.0 in float32/float16
same sequence in log-space:
  Σ log(0.05) = 100 × (−3.0) = −300.0          →  a perfectly ordinary float
```

float32 underflows to zero below roughly `1e-38`; float16 below roughly `6e-5`. A realistic sentence's probability is unrepresentable as a number — but its **logprob, around −100 to −300, is mundane.** Log-space does not merely tidy the arithmetic; it is the only space in which the quantity *exists*.

**The log-sum-exp trick.** Computing `log Z = log Σ_j exp(z_j)` naively also overflows: if any logit is large (say `z_j = 90`), `exp(90) ≈ 1.2e39` blows past float32's max (`~3.4e38`). The fix is to subtract the max logit `m = max_j z_j` before exponentiating, then add it back:

```text
logsumexp(z) = m + log Σ_j exp(z_j − m),    where m = max_j z_j
```

Every shifted exponent `z_j − m` is `≤ 0`, so every `exp(...)` is in `(0, 1]` — no overflow — and the largest term is exactly `1`, so no underflow-to-zero of the whole sum either. This identity is exact (the `m` cancels algebraically), and it is what is actually inside `torch.logsumexp`, `F.log_softmax`, and every CUDA softmax kernel. You will never call it by hand, but when a fused softmax kernel shows up on a profiler, this is the arithmetic it is protecting.

---

## 3. Token logprob vs sequence logprob, and length normalization

Keep two quantities mentally distinct:

| Term | Definition | Typical magnitude |
|---|---|---|
| Token logprob | `ℓ_t = log q(x_t \| x_<t)` — one number, for one realized token | `−0.01` to `−15` |
| Sequence logprob | `L = Σ_{t=1}^{T} ℓ_t` — the sum over the generated tokens | `−10` to `−several hundred` |

The sequence logprob has a built-in trap: **it is monotonically non-increasing in length.** Every additional token adds another `≤ 0` term, so a longer sequence almost always has a more-negative (lower) total logprob than a shorter one — *regardless of quality*. If you rank candidate generations by raw sequence logprob, you are mostly ranking by brevity.

The fix when candidates differ in length is **length normalization** — divide by the token count to get the **mean (per-token) logprob**:

```text
mean logprob  =  L / T  =  (1/T) Σ_{t=1}^{T} log q(x_t | x_<t)
```

This per-token average is exactly the negative of the **cross-entropy** the model assigns to the sequence — the bridge to Lecture 02 — and its exponential of the negative is **perplexity** (Lecture 03). So "mean logprob," "negative NLL," and "−log PPL" are three names for one number; you will meet all three.

When to use which:

- **Comparing fixed-length continuations** (e.g., multiple-choice answer tokens of equal length, or scoring the same target string under two models): raw sequence logprob is fine and correct.
- **Comparing free-form generations of different lengths** (best-of-n sampling, beam hypotheses): normalize, or short hypotheses win on length alone. Beam search implementations expose a `length_penalty` exponent `α` and divide by `T^α` precisely to tune this; `α = 1` is plain mean-logprob, `α = 0` is raw sum.

```python
import torch.nn.functional as F

# token_logprobs: 1-D tensor of per-token logprobs for ONE candidate
seq_logprob  = token_logprobs.sum().item()              # ranks short candidates higher
mean_logprob = token_logprobs.mean().item()             # length-fair; = -NLL
length_penalized = seq_logprob / (len(token_logprobs) ** 0.7)   # beam-style, alpha=0.7
```

---

## 4. Temperature: reshaping the distribution before you read it

**Temperature** `T` rescales the logits *before* the softmax, dividing every logit by `T`:

```text
q_i(T) = softmax(z / T)_i = exp(z_i / T) / Σ_j exp(z_j / T)
```

The effect is to stretch or compress the gaps between logits, which sharpens or flattens the resulting distribution:

| Regime | Effect on distribution | Effect on entropy | Effect on logprobs |
|---|---|---|---|
| `T < 1` (e.g. 0.7) | **sharpens** — gaps widen, top token dominates | entropy ↓ | top-token logprob ↑ (toward 0), tail ↓ |
| `T = 1` | the model's native distribution | native | native logprobs |
| `T > 1` (e.g. 1.5) | **flattens** — gaps shrink, mass spreads | entropy ↑ | top-token logprob ↓, tail ↑ |
| `T → 0⁺` | **argmax** — all mass on the single largest logit | entropy → 0 | top logprob → 0, all others → −∞ |
| `T → ∞` | **uniform** over the vocab | entropy → `log V` (maximal) | every logprob → `−log V` |

```text
logits z = [4.0, 2.0, 1.0]   (3-way toy vocab)

T = 0.5 : q = [0.980, 0.018, 0.002]   sharper, near-argmax
T = 1.0 : q = [0.844, 0.114, 0.042]   native
T = 2.0 : q = [0.629, 0.231, 0.140]   flatter, toward uniform
```

Two warnings that matter for the rest of the course. First, **temperature is a property of how you read the distribution, not of the model.** The logits are fixed; `T` is a sampling-time knob. Second — and this is the one people forget — **when you compute logprobs for evaluation, perplexity, or KL, you almost always want `T = 1`**, the model's true distribution. Reporting perplexity computed at `T = 0.7` is a category error: you measured a distribution the model never claimed. The same applies to KL between two models — compare them both at `T = 1` or the number is meaningless. Temperature belongs to generation; the information-theoretic metrics belong to the untouched distribution.

---

## 5. Getting logprobs in practice

The math is identical everywhere; the APIs are not. Here is the landscape.

| Runtime | How you get logprobs | Truncation behavior | Notes |
|---|---|---|---|
| **OpenAI API** | `logprobs=True`, optional `top_logprobs=k` (k ≤ 20) on chat completions | returns only the chosen token's logprob plus top-`k` alternatives — **not the full vocab** | you cannot reconstruct the full distribution or an exact normalizer from this |
| **HF Transformers** | `out = model(input_ids); F.log_softmax(out.logits, dim=-1)`, then `gather` the target ids | none — you hold the **full** `[seq, V]` logits/logprobs in memory | the reference implementation; everything else is an approximation of this |
| **vLLM** | sampling param `logprobs=k` (generated tokens) and `prompt_logprobs=k` (prompt tokens) | top-`k` per position, like OpenAI; `k` configurable | `prompt_logprobs` is how you score/evaluate a fixed string at serving speed |
| **llama.cpp** | server `n_probs` / `logprobs` field; CLI `--logits-all` to keep logits for *every* position (not just the last) | top-`k` probs per token; `--logits-all` enables full-sequence scoring (perplexity) | `--logits-all` is what the `llama-perplexity` tool needs |

The single most important distinction in that table: **hosted APIs truncate to top-`k` to save bandwidth.** A full logprob vector over Llama-3's 128k vocab is 128k float32 values = **512 KB per token**; Gemma's 256k vocab doubles that to **1 MB per token**. Streaming that for every generated token would dwarf the text payload, so OpenAI/vLLM hand back only the top ~20. That is enough for confidence and ranking, but it is **not** enough to compute an exact cross-model KL divergence (Lecture 04), which needs the whole distribution — a recurring constraint you will hit the moment you try to grade one model against another over an API.

When you control the model, do it the reference way — full `log_softmax`, then gather the realized token. Here is a complete, runnable computation of the **sequence logprob** of a string under a HF causal LM, with the off-by-one shift handled correctly:

```python
import torch, torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

name = "meta-llama/Meta-Llama-3-8B"           # V = 128,256
tok  = AutoTokenizer.from_pretrained(name)
model = AutoModelForCausalLM.from_pretrained(name, torch_dtype=torch.float16,
                                             device_map="auto").eval()

text = "The capital of France is Paris."
ids  = tok(text, return_tensors="pt").input_ids.to(model.device)   # [1, T]

with torch.no_grad():
    logits = model(ids).logits                 # [1, T, V]

# Position t predicts token t+1, so align logits[:, :-1] with targets ids[:, 1:].
logprobs = F.log_softmax(logits[:, :-1, :].float(), dim=-1)         # [1, T-1, V]
targets  = ids[:, 1:]                                               # [1, T-1]
token_lp = logprobs.gather(-1, targets.unsqueeze(-1)).squeeze(-1)   # [1, T-1]

seq_logprob  = token_lp.sum().item()           # additivity: sum of token logprobs
mean_logprob = token_lp.mean().item()          # = -NLL  (Lecture 02)
ppl          = torch.exp(-token_lp.mean()).item()   # = perplexity (Lecture 03)

print(f"sequence logprob = {seq_logprob:.3f} nats")
print(f"mean logprob/tok = {mean_logprob:.3f}   PPL = {ppl:.2f}")
for t, lp in zip(targets[0].tolist(), token_lp[0].tolist()):
    print(f"  {tok.decode([t])!r:>12} : {lp:+.3f}")
```

Three things in that snippet are load-bearing and trip up nearly everyone:

- **The shift.** `logits[:, :-1]` against `ids[:, 1:]`. The logit at position `t` is the prediction *of* token `t+1`; misalign by one and every number is garbage. This same shift defines teacher forcing in Lecture 02.
- **`.float()` before `log_softmax`.** The forward pass runs in float16 for speed, but `logsumexp` over 128k entries accumulates rounding error; upcast to float32 for the normalizer or your logprobs drift by `~0.1` nat — enough to move a perplexity comparison.
- **`gather`, not argmax.** You want the logprob of the token that *actually appeared* in the string, not of the model's preferred token. Scoring is about the realized sequence; that is the whole point of a logprob.

---

## 6. What logprobs power

Everything past this lecture is a way of summarizing logprobs. The major consumers:

- **Confidence / uncertainty.** The chosen token's logprob (or `exp` of it, the probability) is the model's stated confidence. Low confidence, or high per-step entropy `H = −Σ q_i log q_i`, flags spots where the model is guessing — used directly in selective prediction and abstention.
- **Candidate ranking / scoring.** Re-ranking, **best-of-n**, beam search, and self-consistency all score whole sequences by (length-normalized) sequence logprob and keep the best. The arithmetic is the additivity identity from §2.
- **Constrained / guided decoding.** Grammar- and schema-constrained decoders (JSON mode, regex, function-call schemas) work by **masking logits** — setting disallowed tokens' logits to `−∞` *before* softmax, so their probability is exactly `0` and the normalizer redistributes mass over the legal set.
- **Evaluation.** Multiple-choice benchmarks (MMLU-style) often score each option by the logprob the model assigns to its tokens and pick the argmax option — no generation, just logprobs. This is faster and lower-variance than sampling answers.
- **Hallucination / faithfulness signals.** A sharp drop in token logprob, or a spike in entropy, mid-generation is a usable (if noisy) signal that the model has left supported ground — the basis of several uncertainty-based hallucination detectors.

And the through-line of this course: **perplexity (Lecture 03) is `exp` of the mean negative logprob, and KL divergence (Lecture 04) is an expectation of a difference of logprobs between two models.** Both are aggregations of exactly the per-token numbers you just learned to extract. Master the atom and the molecules assemble themselves.

> **Hardware lens:** the LM head is a `[d × V]` GEMM — for Llama-3-8B that is `4096 × 128,256`, about **525M parameters in the unembedding alone**, and at decode it runs as a GEMV (`h` is a single `[d]` vector). The output is the full `[V]` logit vector — 128k values for Llama-3, **256k for Gemma 2/3** — which is why models with huge vocabularies pay a real bandwidth tax at the head, and why fused `log_softmax` and (during training) chunked cross-entropy kernels exist to avoid materializing the full `[seq, V]` float32 tensor. Extracting *full* logprobs means reading that entire vector off the device every step; `top_logprobs` exists precisely to truncate it and save the 512 KB–1 MB-per-token transfer. If "GEMV," "bandwidth-bound," and "roofline" are not yet reflexes, read the prerequisite: [Edge LLM Inference Internals — Lecture 01](../../Track%20C%20-%20Edge%20AI/Edge%20LLM%20Inference%20Internals/Lecture-01.md).

> **2026 update:** logprobs are no longer a debugging curiosity — they are production signal. Eval harnesses score multiple-choice with them; guardrail and routing layers read token confidence and entropy to decide when to escalate to a larger model or refuse; uncertainty-based hallucination detectors key off logprob drops. At the same time access is tightening: several hosted reasoning models restrict or omit logprob output entirely (the reasoning trace is hidden, and so are its logprobs), and structured-output / guided-decoding paths **mask logits** before you ever see them — so the logprobs you read back already reflect the constraint mask, not the model's unconstrained opinion. When you need an exact cross-model KL, you still need the full vocab distribution, which means a model you host yourself; the API's top-20 will not do it.

---

## Current as of

June 2026. The mathematics of this lecture is permanent: softmax, `log_softmax`, the log-sum-exp trick, additivity of sequence logprobs, and the temperature limits (`T→0` argmax, `T→∞` uniform) are settled and will read identically in a decade. What moves is the tooling and the access policy. Vocabulary sizes have crept up (Llama-3 at 128k, Gemma at 256k, with 256k now common for multilingual models), which steadily raises the bandwidth cost of full-vector logprob extraction. API surfaces drift — OpenAI's `top_logprobs` cap, vLLM's `prompt_logprobs`, and llama.cpp's `--logits-all` are current as written, but hosted providers increasingly gate logprob access behind tiers or withhold it for hidden-reasoning models. Treat the equations as load-bearing and re-check the API column of §5 against current provider docs before you depend on it.
