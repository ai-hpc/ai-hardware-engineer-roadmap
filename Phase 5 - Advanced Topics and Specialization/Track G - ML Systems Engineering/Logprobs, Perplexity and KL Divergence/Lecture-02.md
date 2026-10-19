# Lecture 02 - Entropy, Cross-Entropy & Negative Log-Likelihood

**Collection:** [Logprobs, Perplexity & KL Divergence](README.md) | **Previous:** [← Lecture 01](Lecture-01.md) | **Next:** [Lecture 03](Lecture-03.md)

---

The number scrolling past in your training logs — `loss: 2.041`, `loss: 1.998`, `loss: 1.973` — is not a generic "error." It is a specific quantity with a name, a unit, and a physical floor: it is the **cross-entropy** between the data distribution and your model, measured in **nats**. Lecture 01 gave you the raw material, `log q(x_t | x_<t)`, the log-probability the model assigns to each token. This lecture earns the loss that aggregates those logprobs into the single scalar your optimizer chases.

By the end you will be able to read that scalar three ways at once: as the **expected surprise** the data inflicts on your model (cross-entropy), as the sum of a part you can never remove (the data's own **entropy**) and a part you can (the **KL gap**), and as the exponent under the perplexity you will report in Lecture 03. The master identity `H(p,q) = H(p) + D_KL(p ‖ q)` is the spine of this entire course, and this lecture is where it stops being a definition and starts being the thing you stare at when a loss curve plateaus.

We work in nats (natural log) throughout, because that is what PyTorch's `cross_entropy` returns and what your logs print. Bits (`log2`) appear only where the coding interpretation makes them the natural unit; the conversion is a constant factor `1 nat = 1/ln 2 ≈ 1.4427 bits`, and we will be explicit every time we switch.

---

## Learning objectives

By the end of this lecture you should be able to:

* Define **surprise** `−log p(x)` and **entropy** `H(p) = E_p[−log p]` as expected surprise, and explain why the uniform distribution maximizes entropy and a peaked distribution minimizes it.
* Define **cross-entropy** `H(p,q) = −Σ p log q`, give its optimal-coding interpretation, and prove that `H(p,q) ≥ H(p)` always.
* Show that for next-token prediction each observed hard target is represented as one-hot, so the cross-entropy loss **collapses to the mean negative log-likelihood** `mean_t[−log q(x_t | x_<t)]` — the number in your logs.
* State and explain the **master identity** `H(p,q) = H(p) + D_KL(p ‖ q)`, and identify entropy as the **floor** the loss can never beat.
* Convert a loss in nats to perplexity, and read a training/eval curve in terms of irreducible vs. closable error.

---

## 1. Surprise and entropy

Start with a single event. If an outcome `x` has probability `p(x)`, its **surprise** (or *information content*) is

```text
  surprise(x) = −log p(x)
```

This is the only function (up to the base of the log) that is additive over independent events and monotone decreasing in probability: a certain event (`p = 1`) carries zero surprise, an impossible one carries infinite surprise, and two independent events surprise you by the sum of their individual surprises (because `−log(p·p') = −log p − log p'`). The base of the log sets the unit: `log2` gives **bits**, natural `log` gives **nats**.

**Entropy** is the *expected* surprise of a distribution — how surprised you should expect to be, on average, by a draw from `p`:

```text
  H(p) = E_p[−log p] = − Σ_x p(x) log p(x)
```

It is the **irreducible uncertainty** of the source: the average information you receive per sample, and the floor on how few nats per symbol any code can use (Shannon's source-coding theorem). Two limiting cases anchor the intuition:

* **Uniform distribution maximizes entropy.** With `N` equally likely outcomes, `p(x) = 1/N`, so `H = −Σ (1/N) log(1/N) = log N`. Nothing is more uncertain than "all outcomes equally likely"; this is the maximum entropy any distribution over `N` symbols can have.
* **A peaked distribution has low entropy.** If one outcome has probability near 1 and the rest near 0, almost every draw is the expected one, so the average surprise is near 0. A deterministic source (`p = 1` on one symbol) has `H = 0`.

A worked tiny example fixes the units. Take a 4-symbol alphabet `{a, b, c, d}`:

| Distribution | p(a) | p(b) | p(c) | p(d) | H (nats) | H (bits) |
|---|---|---|---|---|---|---|
| Uniform | 0.25 | 0.25 | 0.25 | 0.25 | `ln 4 ≈ 1.386` | `log2 4 = 2.000` |
| Peaked | 0.97 | 0.01 | 0.01 | 0.01 | `≈ 0.168` | `≈ 0.242` |
| Deterministic | 1.00 | 0.00 | 0.00 | 0.00 | `0.000` | `0.000` |

For the uniform row, every symbol carries `−ln 0.25 = 1.386` nats of surprise, and since they are equally likely the expectation is the same `1.386` nats — exactly `log N`. For the peaked row, the dominant symbol carries only `−ln 0.97 ≈ 0.030` nats and is drawn 97% of the time, dragging the average down to `0.168`. Read in bits, the uniform distribution needs a full 2 bits/symbol; the peaked one needs a quarter of that. **Entropy is the answer to "how many nats per symbol does this source fundamentally cost?"** — and no model, however good, can describe it for less.

---

## 2. Cross-entropy

Entropy assumes you know the true distribution `p` and built the optimal code for it. **Cross-entropy** asks the realistic question: you built your code for *your model* `q`, but the symbols actually arrive from `p`. How many nats per symbol do you pay now?

```text
  H(p, q) = E_p[−log q] = − Σ_x p(x) log q(x)
```

**The coding interpretation.** The ideal description length for a symbol under `q` is `−log2 q(x)` bits. An individual binary prefix code uses integer lengths, so this is not generally an exact codeword length. Block/arithmetic coding approaches the expected rate. Averaging the ideal lengths over symbols drawn from `p` gives cross-entropy; using a mismatched `q` adds KL relative to the ideal rate for `p`.

**Why `H(p, q) ≥ H(p)` always.** The cross-entropy can never beat the entropy: the best possible code is the one matched to the true distribution. The shortfall is exactly the KL divergence (Lecture 04 proves `H(p,q) − H(p) = D_KL(p ‖ q) ≥ 0` via Gibbs' inequality), but the bound itself is intuitive — you cannot describe a source for fewer nats than its own entropy, so coding for any `q ≠ p` can only cost more. Equality holds **iff** `q = p` everywhere `p` has support. This single inequality is why training has a floor, which §4 makes precise.

A concrete reading: keep `p` uniform over `{a,b,c,d}` (`H(p) = 1.386` nats) but suppose the model `q` is confidently wrong, `q = (0.7, 0.1, 0.1, 0.1)`. Then

```text
  H(p, q) = −Σ p log q
          = −0.25 (ln 0.7 + ln 0.1 + ln 0.1 + ln 0.1)
          = −0.25 (−0.357 − 2.303 − 2.303 − 2.303)
          ≈ 1.816 nats
```

You pay `1.816` nats instead of the `1.386` floor — `0.430` nats of pure waste, which is precisely `D_KL(p ‖ q)`. The model's overconfidence in `a` costs you every time `b`, `c`, or `d` actually shows up.

---

## 3. The training loss IS cross-entropy

Here is the move that makes this lecture matter for systems work. In language modeling we never have a soft data distribution `p` over the vocabulary; for a given position `t`, the dataset simply *contains* one actual next token, call it `y_t`. The empirical data distribution at that position is therefore a **one-hot**: `p(y_t) = 1` and `p(x) = 0` for every other token `x`.

Plug a one-hot `p` into cross-entropy and the sum collapses — every term is multiplied by zero except the one at the true token:

```text
  H(p, q) = − Σ_x p(x) log q(x)
          = − 1 · log q(y_t)              ← only the true-token term survives
          = − log q(y_t | x_<t)           ← the NLL of the true token
```

So at a single position the cross-entropy **is** the negative log-likelihood of the actual next token — nothing more. Averaging over every position in the batch gives the scalar your optimizer minimizes:

```text
  loss = mean_t [ − log q(y_t | x_<t) ]   = empirical cross-entropy = mean NLL
```

This is why "cross-entropy loss," "negative log-likelihood," and "the language-modeling loss" are three names for the same number. (Note the one-hot is the *empirical* `p`; the *true* `p` for a position is generally soft — many tokens could plausibly continue — and that gap is what entropy in §4 captures. Distillation, Lecture 05, replaces the one-hot with a teacher's soft `p`.)

**Teacher forcing and shifted labels.** During training the model predicts position `t` while conditioning on the *ground-truth* prefix `x_<t` rather than its own past samples — this is **teacher forcing**. Mechanically it means the targets are the inputs **shifted by one**: `logits[:, :-1]` predicts `labels[:, 1:]`. Padding is excluded with an ignore index. Prompt tokens are excluded only for objectives that score completions; ordinary language-model pretraining usually scores text throughout. Divide by the number of included targets after shifting. Every position is scored in parallel against the single token that actually followed.

The PyTorch identity below is the whole §3 in code — `F.cross_entropy` on logits is *exactly* the mean of the gathered `−log_softmax` at the true tokens:

```python
import torch
import torch.nn.functional as F

torch.manual_seed(0)
B, T, V = 2, 5, 32000            # batch, seq, vocab
logits = torch.randn(B, T, V)
labels = torch.randint(0, V, (B, T))

# --- Shifted labels (teacher forcing): predict token t+1 from tokens <= t ---
shift_logits = logits[:, :-1, :].reshape(-1, V)   # [(B*(T-1)), V]
shift_labels = labels[:, 1:].reshape(-1)          # [(B*(T-1))]

# Path A: the library's fused cross-entropy (mean reduction, in nats)
loss_builtin = F.cross_entropy(shift_logits, shift_labels)

# Path B: cross-entropy spelled out = mean of gathered negative log-softmax
logq = F.log_softmax(shift_logits, dim=-1)                 # log q over vocab
nll  = -logq.gather(1, shift_labels[:, None]).squeeze(1)   # -log q(true token)
loss_manual = nll.mean()

print(loss_builtin.item(), loss_manual.item())   # identical to fp precision
assert torch.allclose(loss_builtin, loss_manual, atol=1e-6)
```

`F.cross_entropy` internally fuses `log_softmax` + `nll_loss`; it never asks you to softmax first (doing so and then taking `log` is numerically worse — see Lecture 01's log-sum-exp). The two paths agree to floating-point precision because they *are* the same computation.

> **Hardware lens:** computing this loss materializes a `[batch × seq × vocab]` logit tensor and an equally large `log_softmax` — for a 256k-vocab model at `batch·seq = 8192`, that is `8192 × 256000 × 2 bytes ≈ 4.2 GB` in `bf16` for the logits *alone*, frequently the single largest activation in the whole training step and a major source of activation memory and HBM bandwidth. The fix is to never materialize the full tensor: fused / **chunked cross-entropy** kernels (Liger-Kernel's fused linear+CE, **cut-cross-entropy**, FlashCE) compute the loss and its gradient block-by-block over the vocabulary, keeping only running reductions. Vocabularies of 128k–256k (Llama 3, Gemma, Qwen) make this a first-order cost, not a footnote.

---

## 4. The master identity

Everything above converges on one equation. Cross-entropy decomposes exactly into entropy plus KL divergence:

```text
  H(p, q)  =  H(p)  +  D_KL(p ‖ q)
   ▲           ▲          ▲
   │           │          └─ the gap: avoidable error the model CAN close
   │           └──────────── irreducible entropy of the data: the FLOOR
   └──────────────────────── the loss you actually train (mean −log q)
```

The algebra is one line — split the log inside KL and recognize the two pieces (the full Gibbs-inequality proof that `D_KL ≥ 0` is Lecture 04):

```text
  D_KL(p ‖ q) = Σ p log(p/q) = Σ p log p − Σ p log q = −H(p) + H(p, q)
  ⇒  H(p, q) = H(p) + D_KL(p ‖ q),     with  D_KL(p ‖ q) ≥ 0.
```

Read physically: **your loss is the data's own entropy plus the distance from your model to the data.** Training can only ever move the second term. No amount of optimization, scale, or data drives the loss below `H(p)` — that is the **floor**, the genuine ambiguity in language (many tokens can legitimately follow "The capital of France is"). A plateau does not establish `D_KL(p ‖ q) → 0`: model capacity, optimization, finite data, or a distribution shift can leave a substantial gap. The population identity does not make true conditional entropy observable from one finite validation set. A model that reached loss `= H(p)` would be *perfect* — it would assign exactly the data's true conditional probabilities — and `D_KL(p ‖ q) = 0` is the only way to get there.

Quantization grading can measure `D_KL(p_reference ‖ q_quant)` on common prefixes. That is distinct from the change in held-out data NLL: `E_data[log p_reference − log q_quant]`. These become equal only when the token expectation is taken under the reference distribution. Distillation may directly minimize teacher KL; alignment may use a policy KL regularizer. Always name both distributions and the averaging population.

---

## 5. Reading a training / eval curve

Your loss is in **nats per token**. The single most useful reflex is to exponentiate it: because `PPL = exp(H(p, q)) = exp(mean NLL)` (derived in full in Lecture 03), the loss and the perplexity are the same information in two units.

| Mean NLL (nats/token) | `PPL = exp(mean NLL)` |
|---|---:|
| `0` | `1` |
| `1` | `2.718` |
| `2` | `7.389` |
| `3` | `20.086` |
| `5` | `148.413` |
| `ln V` | `V` |

These are mathematical conversions, not universal quality bands. Compare the same text, tokenizer, masking, and context policy. A uniform predictor gives `ln V`; random initialization need not produce uniform logits. A deviation from `ln V` is a diagnostic to investigate, not proof of leakage or a broken loss.

Two reporting conventions to keep straight:

* **Per-token vs. per-sequence.** The printed loss is almost always the **per-token** mean (`reduction='mean'` divides by the token count). A *per-sequence* sum (`reduction='sum'` then divide by batch) is sensitive to sequence length and is the wrong thing to compare across runs with different `seq_len`. Always confirm the denominator before comparing two curves; a "lower loss" that merely used shorter sequences is an artifact.
* **What "good" looks like is relative.** Absolute loss is only meaningful against a *fixed tokenizer and dataset*. Two models with different tokenizers spread the same text across different numbers of tokens, so their per-token losses are not comparable — the fix is **bits-per-byte**, normalizing to the underlying UTF-8 bytes (Lecture 03). Until then, treat raw cross-entropy as comparable only *within* a tokenizer family.

> **Implementation note, checked 2026-10-18:** fused linear/cross-entropy implementations can avoid materializing full-vocabulary logits. Availability and memory savings depend on the framework, shape, dtype, and kernel. See the primary [Cut Cross-Entropy implementation](https://github.com/apple/ml-cross-entropy). Choose the reporting unit explicitly; bits-per-byte provides a common text denominator, while token NLL remains useful under a fixed tokenizer.

---

## Current as of

June 2026. The mathematics of this lecture is permanent: surprise, entropy, cross-entropy, the master identity `H(p,q) = H(p) + D_KL(p ‖ q)`, and the collapse of cross-entropy to mean NLL under a one-hot target are Shannon-1948 results that no framework revision will touch — the loss in your logs has meant exactly this since the first neural language model. What moves is the **tooling**: which fused/chunked cross-entropy kernel is fastest (Liger-Kernel, cut-cross-entropy, FlashCE and their successors), where the fused-linear+CE boundary lands in a given framework, and whether teams headline their numbers in nats, bits-per-token, or bits-per-byte. Treat the identity as bedrock and the kernel names and reporting unit as the parts to re-check on each refresh.
