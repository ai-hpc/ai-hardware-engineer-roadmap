# Lecture 03 - Perplexity: The Exponential of Confusion

**Collection:** [Logprobs, Perplexity & KL Divergence](README.md) | **Previous:** [← Lecture 02](Lecture-02.md) | **Next:** [Lecture 04](Lecture-04.md)

---

Lecture 02 left you holding the loss: cross-entropy `H(p, q) = −Σ p log q`, the mean negative log-likelihood your model pays per token. That number is in nats, it lives somewhere between 1.8 and 2.5 for a decent modern LLM, and almost nobody quotes it. They quote **perplexity** instead — the same quantity, run through `exp`. Perplexity is the single most-cited number in language modeling, the one in every quantization table and every "our model beats theirs" tweet. It is also the most misused, because three different failure modes quietly invalidate the comparison people are trying to make with it.

Perplexity is not a new idea. It is `exp(H(p, q))` and nothing more — a monotone re-skinning of the loss you already train on. The reason it earns its own lecture is that the `exp` buys you an *interpretation* (an effective branching factor — how many equally-likely tokens the model is, in effect, choosing between) and the reporting *conventions* around it (tokenizer normalization, sliding windows, corpus choice) carry every trap that turns a clean metric into a wrong conclusion.

This lecture derives perplexity straight from Lecture 02, gives you the branching-factor reading with a worked micro-example, then spends most of its length on the three things that make practitioners report it wrong: tokenizer dependence (fixed by bits-per-byte), the long-document scoring problem (fixed by the sliding window), and the temptation to treat perplexity as ground truth for output quality (it is a rough proxy — Lecture 05 does the rigorous version). By the end, perplexity is your first-line acceptance test for a quantized model, and you know exactly when to stop trusting it.

---

## Learning objectives

By the end of this lecture, you should be able to:

1. Derive `PPL = exp(mean NLL) = exp(H(p, q)) = 2^(bits per token)` directly from the Lecture 02 loss, and read it as a geometric mean of `1/q(x_t)`.
2. Interpret perplexity as an **effective branching factor**, and compute it by hand on a tiny example.
3. Explain why perplexity is **not comparable across tokenizers**, and convert to **bits-per-byte (BPB)** so it is.
4. Implement the **HF strided sliding-window** perplexity loop and explain why naive chunking inflates the number.
5. Name the standard corpora (WikiText-2/103, C4) and the context/stride you must report for a result to be reproducible.
6. Use perplexity as the **canonical quantization gate** (PPL delta of INT4 vs FP16), and state precisely why it is a *rough* proxy — deferring the rigorous grader to Lecture 05.

---

## 1. Definition and derivation

Start from where Lecture 02 ended. For a held-out sequence `x_1 … x_N`, teacher-forced under the model `q`, the loss is the mean negative log-likelihood:

```text
    L  =  mean NLL  =  (1/N) Σ_t [ −log q(x_t | x_<t) ]   (nats)
```

When the reference distribution `p` is the empirical data (one-hot on the token that actually occurred), this mean NLL *is* the cross-entropy `H(p, q)` — that is the identity Lecture 02 built. **Perplexity is its exponential:**

```text
    PPL  =  exp(L)  =  exp( H(p, q) )  =  exp( (1/N) Σ_t −log q(x_t | x_<t) )
```

That is the whole definition. Everything else is reading it differently.

**As a base change.** `exp` and `log` are natural (nats) here, but the loss is identical content in any base. If you measure cross-entropy in **bits** (`log2`), perplexity is `2` raised to the bits-per-token:

```text
    PPL  =  2^( bits per token )  =  2^( H_bits(p, q) )
    H_bits  =  H_nats / ln 2          (1 nat = 1/ln2 ≈ 1.4427 bits)
```

A loss of `2.0 nats` is `2.0 / 0.6931 ≈ 2.885 bits`, and `PPL = e^2.0 = 2^2.885 ≈ 7.39`. Use whichever base your tooling reports; just never mix them silently.

**As a geometric mean.** Push the `exp` through the sum and it becomes a product. Perplexity is the geometric mean of the *reciprocal* probabilities the model assigned to the true tokens:

```text
    PPL  =  exp( (1/N) Σ_t −log q(x_t) )
         =  exp( (1/N) Σ_t  log (1 / q(x_t)) )
         =  ( Π_t  1 / q(x_t) )^(1/N)
```

This is the most physical reading. `1/q(x_t)` is "how surprised the model was at token `t`" — large when it assigned the truth a small probability. Perplexity is the typical such surprise, geometric-averaged so one catastrophic token (a probability near zero) blows it up multiplicatively, exactly as it blows up the loss additively. A geometric mean, not arithmetic, because we averaged in log-space.

---

## 2. Interpretation: the effective branching factor

Here is the sentence to memorize:

> A model with perplexity `K` is **as uncertain, on average, as if it were choosing uniformly among `K` equally-likely tokens** at each step.

The anchor is the uniform distribution. If `q` spreads its mass evenly over a vocabulary of size `V`, then `q(x_t) = 1/V` for every token, every `1/q(x_t) = V`, and the geometric mean is exactly `V`:

```text
    uniform over V symbols   →   PPL = V        (uniform-predictor baseline)
    perfect, confident model →   PPL → 1        (q(truth)=1 everywhere → 0 loss)
    confidently wrong model →   PPL can exceed V without bound
```

Perplexity lies in `[1, ∞]`, not `[1, V]`. With two tokens, assigning the observed token probability 0.001 gives PPL 1000. The effective-branching interpretation compares average surprise; it does not count how many tokens the model literally retained.

**Worked micro-example.** Vocabulary `V = 4`: `{a, b, c, d}`. A two-token reference sequence `[a, c]`. The model predicts:

```text
    step 1, true=a:   q = (a:0.7, b:0.1, c:0.1, d:0.1)   →  q(a) = 0.7
    step 2, true=c:   q = (a:0.2, b:0.2, c:0.5, d:0.1)   →  q(c) = 0.5
```

Compute the loss and perplexity:

```text
    NLL_1 = −ln 0.7 = 0.3567
    NLL_2 = −ln 0.5 = 0.6931
    L     = (0.3567 + 0.6931) / 2 = 0.5249 nats
    PPL   = exp(0.5249) = 1.690
    check (geometric mean): ( (1/0.7)(1/0.5) )^(1/2) = (2.857)^(1/2) = 1.690  ✓
```

PPL `1.69` is lower than the uniform-predictor baseline of 4 on these two observations. This tiny example demonstrates arithmetic, not broad model quality. A confidently wrong predictor can perform worse than uniform.

---

## 3. The tokenizer-dependence trap

This is the error that voids most cross-model perplexity comparisons in the wild, and the one a senior engineer catches in review.

Perplexity is **per token**. But "token" is not a property of the text — it is a property of the *tokenizer*. The same English sentence becomes a different number of tokens under GPT-4's BPE, Llama's SentencePiece, and a byte-level fallback. The loss is summed over tokens and divided by the token count, so the *same underlying information* gets sliced into a different number of pieces, and the per-token average shifts — **even if the two models are equally good at predicting the actual bytes.**

```text
    Text: "internationalization"   (20 characters, 1 word)

    Tokenizer A (coarse BPE):   [internation, alization]        → 2 tokens
    Tokenizer B (fine BPE):     [inter, nation, al, ization]    → 4 tokens
    Tokenizer C (byte-level):   [i,n,t,e,r,...,n]               → 20 tokens

    Suppose each model spends the SAME total 12 nats to predict this word.
        A:  PPL = exp(12 / 2)  = exp(6.0) = 403
        B:  PPL = exp(12 / 4)  = exp(3.0) = 20.1
        C:  PPL = exp(12 / 20) = exp(0.6) = 1.82
```

Three wildly different perplexities — `403` vs `20` vs `1.8` — for **identical predictive skill on identical text**. The model that splits text into more, smaller, easier-to-predict tokens looks dramatically "better" by per-token PPL, purely as a tokenization artifact. Comparing PPL across models with different tokenizers or vocabularies is therefore meaningless. It is comparing the average difficulty of *their own* puzzle pieces, not their skill.

**A common denominator: raw bytes.** Each model assigns a probability to a token sequence representing the text. The chain rule factorizes that probability within that model; it does not force different tokenizers or models to assign the same probability. Compare total negative log-likelihood on identical text spans, divided by their original UTF-8 byte count, after specifying normalization, special tokens, boundaries, and context policy. That gives bits-per-byte (BPB):

```text
    BPB  =  total_nats / ( ln 2 · n_bytes )      =  total_bits / n_bytes
```

In the example, the stipulated equal totals of 12 nats on 20 ASCII bytes give `12/(ln(2)*20)=0.866` bits/byte. The shared denominator removes the arbitrary token-count scale; different models will generally have different NLL numerators. BPB is useful evidence, not automatic proof of a fair comparison or better downstream behavior.

---

## 4. Sliding-window perplexity for long documents

The second trap is about *context*, and it bites the moment your evaluation document is longer than the model's context window.

A model with context length `L` cannot condition on an arbitrarily long prefix in one ordinary forward pass. Non-overlapping chunks repeatedly discard left context. This changes the evaluation protocol and often raises measured PPL; a particular learned model is not mathematically guaranteed to improve with every added context token. Use overlapping windows when the intended question is prediction with as much available left context as the window allows.

```text
    Document (10k tokens), context L = 4096:

    NON-OVERLAPPING (shorter context near boundaries):
      chunk 1: [t0 .................. t4095]   t0 has 0 ctx, t4095 has full ctx
      chunk 2: [t4096 ............... t8191]   t4096 has 0 ctx AGAIN  ← inflates PPL
      chunk 3: [t8192 ............... t9999]   t8192 has 0 ctx AGAIN
                ↑ every chunk boundary re-pays the "cold start" penalty

    STRIDED SLIDING WINDOW (HF method), stride s < L:
      window 1: [t0 ........................ t4095]   score ALL (first window)
      window 2: [t_s ....................... t_{s+4095}]
                 └ first (L − s) tokens: CONTEXT ONLY, masked with -100
                                          last  s tokens: SCORED with available left context
      window 3: slide by s again ...
                → later targets see L−s through L−1 context tokens in full windows
```

A width-`L`, stride-`s` window scores newly encountered targets and masks overlapping context. In full later windows, those targets have between `L−s` and `L−1` preceding tokens, not all `L` tokens. Stride one maximizes context per scored token but requires more forward passes. State the chosen protocol and inspect the actual target mask after causal shifting.

Here is a framework adaptation sketch for one unpadded token stream. Pin a compatible model, dataset, tokenizer, and Transformers release before executing it; this model-sized example was not run during the course refresh. The [CPU lab](Lab-01-Masked-Quality-Metrics.md) tests the denominator and window accounting independently:

```python
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

device = "cuda"
model_id = "your/model"
model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float16).to(device)
tok   = AutoTokenizer.from_pretrained(model_id)
model.eval()

# WikiText-2: concatenate the test split into one long token stream.
test = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
enc  = tok("\n\n".join(test["text"]), return_tensors="pt")

max_len = model.config.max_position_embeddings   # the model's context L
stride  = min(512, max_len - 1)                  # require 1 <= stride < L
assert max_len >= 2 and stride >= 1
seq_len = enc.input_ids.size(1)

nll_sum, n_tokens, prev_end = 0.0, 0, 0
for begin in range(0, seq_len, stride):
    end      = min(begin + max_len, seq_len)     # window of width <= L
    trg_len  = end - prev_end                    # the NEW tokens to actually score
    ids      = enc.input_ids[:, begin:end].to(device)

    targets  = ids.clone()
    targets[:, :-trg_len] = -100                 # mask context tokens: no loss, just ctx

    with torch.no_grad():
        out = model(ids, labels=targets)
        # For standard causal loss: count valid targets AFTER its label shift.
        # Later windows can predict their first NEW token using overlapping context.
        num_scored = (targets[:, 1:] != -100).sum().item()
        if num_scored == 0:
            raise ValueError("No scored targets in this window")
        nll_sum   += out.loss.float() * num_scored
        n_tokens  += num_scored

    prev_end = end
    if end == seq_len:
        break

avg_nll = nll_sum / n_tokens
ppl     = torch.exp(avg_nll)
print(f"WikiText-2  PPL = {ppl.item():.3f}   (L={max_len}, stride={stride})")
```

Two checks prevent denominator bugs: aggregate NLL sums weighted by actual scored-token counts, and count labels after shifting and masking. For length 10, window 6, stride 3, the counts are 5, 3, 1; subtracting one from each new-token count incorrectly gives 5, 2, 0. The first label in the first window has no preceding prediction; later windows supply overlap for their first new target. The [executable lab](Lab-01-Masked-Quality-Metrics.md) verifies every intended target is counted once.

---

## 5. Standard practice and reproducibility

Perplexity is only a number; it is comparable only against another number computed *the same way*. The community has converged on a small set of baselines so that "PPL 5.68" means something:

| Corpus | What it is | Typical use |
|---|---|---|
| **WikiText-2** | ~2M tokens, curated Wikipedia (raw + tokenized variants) | the fast de-facto sanity baseline; what `llama-perplexity` defaults near |
| **WikiText-103** | ~103M tokens, same source, larger | longer-context / lower-variance LM evaluation |
| **C4** | Colossal Clean Crawled Corpus, web-scale | pretraining-distribution PPL, less Wikipedia-overfit |
| **The Pile / domain sets** | code, books, papers | domain-shift checks (a quant can regress on code but not prose) |

A perplexity result is **not reproducible** unless you report, alongside the number:

```text
    □ exact corpus + split + version   (wikitext-2-raw-v1 test ≠ wikitext-2-v1)
    □ context length L                 (changes the conditioning available to each prediction)
    □ stride s                         (changes overlap, compute, and conditioning)
    □ dtype / precision                (FP16 vs BF16 vs the quant under test)
    □ tokenizer + whether raw or pre-tokenized text was used
    □ how documents were joined (newline glue) and whether BOS was prepended
```

The single most common reproducibility failure is comparing your PPL to a paper's while using a different `L` or stride: longer context and smaller stride often lower PPL, so a setup difference masquerades as a model difference. Pin all six knobs, or the comparison is folklore.

---

## 6. Perplexity as the canonical quantization metric

This is where perplexity earns its keep for a systems engineer. When you quantize a model — FP16 → INT4 — you need a fast, automatic answer to "did that break it?" Perplexity is the first instrument you reach for, because the recipe is trivial and the signal is real:

```text
    1. Compute PPL_fp16 on a fixed corpus / L / stride  (the reference).
    2. Quantize → compute PPL_int4 on the SAME corpus / L / stride.
    3. Report the DELTA, absolute and relative:
           ΔPPL      = PPL_int4 − PPL_fp16
           PPL ratio = PPL_int4 / PPL_fp16
```

The rule of thumb the community uses: a **good 4-bit quant adds well under 1–2% perplexity** over FP16. A Q4_K_M or AWQ INT4 that lands at +0.5–1.5% is shipping-grade; a quant that adds 5%, 10%, or "blows up" to a PPL of hundreds has a broken scale, a mis-handled outlier channel, or a bad group size — and you caught it in one forward pass instead of in production.

The de-facto tool in the GGUF world is **`llama-perplexity`** (the `llama.cpp` binary, formerly `perplexity`):

```text
    # FP16 reference
    ./llama-perplexity -m model-f16.gguf  -f wiki.test.raw -c 4096
        →  PPL = 5.6789  (final), printed with a running estimate + stderr

    # INT4 candidate, IDENTICAL corpus and context
    ./llama-perplexity -m model-q4_k_m.gguf -f wiki.test.raw -c 4096
        →  PPL = 5.7361
        ⇒  ΔPPL = +0.057  ,  ratio = 1.0101   (+1.01%)  →  acceptable, ship-track
```

> **Hardware lens:** perplexity is the **first-line acceptance test** for a quantized GGUF or any edge deployment, and the reason is cost. It is a single forward pass over a fixed corpus — a few hundred to a few thousand sequences — which is *cheap* relative to running a full downstream benchmark suite (MMLU, GSM8K, HumanEval), each of which is many generations with sampling and scoring harnesses. On a Jetson or a laptop you can score WikiText-2 in minutes and get a hard go/no-go: the **ΔPPL gates whether your INT4 build even enters the benchmark queue.** If the quant added 8% perplexity, you do not waste GPU-hours benchmarking it; you fix the quantization first. PPL is the smoke test that runs before the integration tests.

**But perplexity is a *rough* proxy for output quality, and a senior engineer says so out loud.** Two quants with *equal* perplexity can behave differently in generation: PPL is an average over a corpus of teacher-forced next-token surprise, so it is blind to *where* the errors land. A quant can preserve average perplexity while degrading specifically on the high-stakes low-entropy tokens (the closing brace, the correct digit, the function name) that determine whether code runs or math is right — precisely the tokens a user notices. Perplexity also says nothing about whether the model's *top* token still matches the FP16 model's top token, which is what greedy decoding actually emits. The rigorous instruments — **KL divergence** between the quant and FP16 next-token distributions, and **top-token agreement** — correlate far better with perceived quality, and grading a quant properly with all four numbers (PPL ratio, mean KLD, top-1 agreement, token-probability RMS) is the entire subject of **[Lecture 05](Lecture-05.md)**. Treat PPL here as the necessary first gate, not the verdict.

For where this quantization-grading workflow plugs into a broader model-compression pipeline — pruning, distillation, and quantization as a coordinated campaign — see the compression treatment in [Practical Machine Learning (CS329P) — Lecture 10](../../../Phase%203%20-%20Artificial%20Intelligence/Track%20B%20-%20Agentic%20AI%20and%20ML%20Engineering/4.%20ML%20Engineering%20and%20MLOps/Practical%20Machine%20Learning%20%28CS329P%29/Lecture-10.md), which *uses* these metrics; this lecture is where they are derived.

> **2026 update:** two shifts in practice. First, perplexity is increasingly reported as **bits-per-byte** rather than raw per-token PPL precisely so that models with different tokenizers (the norm now, as vocabularies have diverged to 128k–256k) can be ranked at all — raw cross-tokenizer PPL comparisons are now treated as a red flag in review. Second, the community consensus has hardened that **PPL alone is insufficient for quantization selection**: the January 2026 finding that quantization *format* matters more than nominal bit-width reinforced that intrinsic metrics are necessary but not sufficient, and KL-divergence plus top-token agreement (Lecture 05) are now the preferred discriminators between candidate quants of equal perplexity. PPL gates; KLD decides.

---

## Key takeaways

- **Perplexity is `exp` of the loss** — `PPL = exp(mean NLL) = exp(H(p, q)) = 2^(bits per token)` — and also the **geometric mean of `1/q(x_t)`**. No new content beyond Lecture 02; a new *reading*.
- It reads as an **effective branching factor**: uniform over `V` → `PPL = V`; a confident correct model → `PPL → 1`; the gap between PPL and `V` is exactly what the model learned.
- **Tokenizer trap:** per-token PPL is not comparable across tokenizers; more, smaller tokens flatter the number. Fix by reporting **bits-per-byte**, `BPB = total_nats / (ln2 · n_bytes)`, which cancels tokenization.
- **Long-document accounting:** overlapping windows score newly encountered targets, with context-only labels masked. Count included targets after causal shifting. Later full windows offer between `L−s` and `L−1` left-context tokens; the final partial window must use its actual count.
- **Reproducibility:** a PPL number is meaningless without corpus/split/version, context `L`, stride `s`, dtype, and tokenizer. Pin all of them.
- **Quantization gate:** report **ΔPPL / PPL ratio** of INT4 vs FP16 (good 4-bit quant: <1–2%); `llama-perplexity` is the GGUF tool. But PPL is a **rough** proxy — equal-PPL quants differ — so KL-divergence and top-token agreement (Lecture 05) make the final call.

---

## Current as of

2026-06. The mathematics (`PPL = exp(H(p,q))`, the BPB normalization, the sliding-window correction) is settled and timeless. Tooling pins: `llama.cpp`'s perplexity binary is now `llama-perplexity`; HuggingFace's strided-PPL recipe is the reference long-document method. Practice pins: bits-per-byte is the cross-tokenizer reporting norm as vocabularies diverged to 128k–256k; community consensus (reinforced by the Jan-2026 quantization-format finding) is that PPL is a necessary first gate but **not** sufficient for quant selection — KL-divergence and top-token agreement (Lecture 05) are the preferred discriminators. WikiText-2/103 and C4 remain the de-facto baselines.
