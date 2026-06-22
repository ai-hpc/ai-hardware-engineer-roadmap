# Logprobs, Perplexity & KL Divergence — The Information Theory of LLM Inference

<div class="course-identity mlsys" markdown="1">
<div class="course-identity__icon">−log q</div>
<div markdown="1">
<p class="course-identity__eyebrow">Phase 5 · ML Systems Engineering · Foundations Course</p>
<p class="course-identity__title">The three numbers every LLM systems engineer must read fluently — the log-probability a model assigns, the perplexity that grades it, and the KL divergence that measures how far a quantized, distilled, or aligned model has drifted from the original.</p>
<p class="course-identity__meta">Artifact: a quantization grader that reports PPL delta, mean KLD, and top-token agreement · Measure: bits-per-token, perplexity ratio, D_KL, token-probability RMS</p>
</div>
</div>

> *Logprobs are what the model says. Perplexity is how surprised it was. KL divergence is how far your cheaper model strayed from the real one. They are not three topics — they are three readings of one equation.*

Every serious decision in LLM inference is a decision about a probability distribution you cannot see directly. Did INT4 quantization break the model or just dent it? Is the distilled 1B faithful to the 27B teacher? Is the speculative draft close enough to the target to be worth running? Did RLHF push the policy too far from the base model? You answer all four with the same three instruments — **log-probabilities**, **perplexity**, and **KL divergence** — and all three are bound by a single identity:

```text
    H(p, q)        =        H(p)         +        D_KL(p ‖ q)
  cross-entropy           entropy                  KL divergence
  the loss you             the irreducible          the EXCESS — the part
  actually train on        floor (data's own        quantization, distillation,
  (mean −log q)            uncertainty)             and RLHF spend their lives fighting
        │                                                  │
        └──────────  Perplexity = exp(H(p, q))  ───────────┘
                     the metric you report
```

`p` is the distribution you trust — the data, the teacher, the FP16 reference, the policy you don't want to leave. `q` is the distribution you can afford — your model, the student, the INT4 quant, the tuned policy. Cross-entropy is what you minimize when you train. Entropy is the floor you can never beat. **KL divergence is the gap between them — the avoidable waste — and almost every systems technique in this phase is a campaign to shrink one specific KL.** This course makes that identity second nature, then spends its last lecture showing it driving the quantization, distillation, alignment, and speculative-decoding machinery you already met elsewhere in Phase 5.

**Layer mapping:** the measurement layer that sits *under* the kernel, compiler, and architecture layers — the instrumentation an MLSys engineer reads to know whether any optimization preserved the model.

**Role targets:** ML Systems Engineer · AI Inference Engineer · Model-Compression Engineer · Evaluation/Quality Engineer · Research Engineer (alignment/distillation).

**Prerequisites:**

* High-school-plus probability (a probability distribution, expectation, the log function) and comfort reading Python/PyTorch.
* [Edge LLM Inference Internals — Lecture 01](../../3.%20Edge%20AI/Edge%20LLM%20Inference%20Internals/Lecture-01.md) for the roofline and why decode is memory-bound — the "Hardware lens" callouts here assume it.
* Helpful but not required: [Practical Machine Learning (CS329P) — Lecture 10 (Model Compression)](../../../Phase%203%20-%20Artificial%20Intelligence/Track%20B%20-%20Agentic%20AI%20and%20ML%20Engineering/4.%20ML%20Engineering%20and%20MLOps/Practical%20Machine%20Learning%20%28CS329P%29/Lecture-10.md), which uses these metrics without deriving them. This course is the derivation.

**Pairs with:** [MLSys Deep Dives](../MLSys%20Deep%20Dives/README.md) (where speculative decoding and quantization appear as systems) and the [Gemma 4 Edge Deployment](../../3.%20Edge%20AI/Gemma%204%20Edge%20Deployment/README.md) course (where perplexity and acceptance length are measured on real hardware).

---

## Why this course is structured the way it is

The five lectures walk the master identity from its raw material to its payoff:

1. **Logprobs** — the raw material. Everything downstream is built from `log q(token)`, so you must know exactly what it is, why it lives in log-space, and how to extract it from any runtime.
2. **Entropy & cross-entropy** — the objective. Cross-entropy *is* the training loss; entropy is the floor it's chasing. This lecture earns the left two terms of the identity.
3. **Perplexity** — the metric. `exp` of cross-entropy, the effective branching factor, and the tokenizer trap that voids most cross-model comparisons.
4. **KL divergence** — the gap. The third term, proven equal to cross-entropy minus entropy, with its asymmetry, its forward/reverse personalities, and its role as the universal "distance from the model I trust."
5. **Applications** — the payoff. The identity turned loose on quantization grading, knowledge distillation, RLHF's KL penalty, and the speculative-decoding acceptance rule, ending in a capstone quant grader.

---

## Course Map (5 lectures)

<div class="lecture-map" markdown>

| # | Lecture | The thread |
|---|---------|-----------|
| [01](Lecture-01.md) | **Logprobs — what a language model actually emits** — logits → softmax → log-probabilities, why log-space (additivity, stability, log-sum-exp), sequence logprob, temperature, and extracting logprobs from OpenAI / HF / vLLM / llama.cpp | the raw material |
| [02](Lecture-02.md) | **Entropy, Cross-Entropy & NLL** — Shannon entropy, cross-entropy as the training loss, bits vs nats, teacher forcing, and the first sight of `H(p,q) = H(p) + D_KL` | the objective |
| [03](Lecture-03.md) | **Perplexity — the exponential of confusion** — `PPL = exp(mean NLL)`, effective branching factor, bits-per-byte, the tokenizer-comparison trap, sliding-window PPL, and perplexity as the canonical quantization metric | the metric |
| [04](Lecture-04.md) | **KL Divergence — the gap between two distributions** — `D_KL(p‖q)`, Gibbs' inequality, asymmetry, forward vs reverse KL, Jensen–Shannon, and the proof that `D_KL = H(p,q) − H(p)` | the gap |
| [05](Lecture-05.md) | **Where it all lands** — llama.cpp `--kl-divergence` quant grading, knowledge distillation, the RLHF/PPO/GRPO KL penalty, the speculative-decoding acceptance rule, and a capstone quant grader | the payoff |

</div>

---

## Course Outcomes

By the end you should be able to:

* Extract and interpret **token and sequence logprobs** from any runtime, and explain why log-space (not probability-space) is the right place to do the arithmetic.
* Derive **cross-entropy = entropy + KL divergence** from scratch and explain what each term means physically — and why cross-entropy can never beat entropy.
* Compute **perplexity** correctly with a sliding window, report **bits-per-byte** when tokenizers differ, and explain why a raw PPL comparison across two tokenizers is meaningless.
* Compute **KL divergence** between two next-token distributions, distinguish forward from reverse KL, and say which one a given technique (MLE, distillation, variational inference, RL) actually minimizes.
* Grade a **quantization** the way llama.cpp does — PPL ratio, mean KLD, top-token agreement, token-probability RMS — and explain why PPL alone is "rough" and KL/agreement correlate better with perceived quality.
* Recognize the **same KL term** inside knowledge distillation, the RLHF KL penalty, and the speculative-decoding acceptance rule — and explain why speculative decoding is *lossless*.

---

## Currency / Refresh Discipline

The mathematics here is settled (Shannon 1948; Kullback–Leibler 1951). What moves is the **tooling and practice**:

* The quantization-evaluation section (Lecture 05) tracks current practice — llama.cpp's `--kl-divergence` mode and the **January 2026** finding ([arXiv 2601.14277](https://arxiv.org/abs/2601.14277)) that quantization *format* matters more than nominal bit-width, and that intrinsic metrics (PPL, KLD) are necessary but not sufficient — you still validate on downstream benchmarks.
* RLHF objectives evolve (PPO → DPO → GRPO); the **KL term is the invariant**, and that is what this course teaches.
* Every lecture closes with a **`## Current as of`** note marking what is timeless math versus 2026 tooling.

---

## Exit Criteria

You are done with this course when you can take a **freshly quantized model** and, without looking anything up:

* Compute its perplexity on WikiText-2 with a correct sliding window, and its bits-per-byte.
* Compute mean KL divergence and top-1 token agreement against the FP16 reference.
* Read those three numbers together and give a defensible verdict — *ship it, or the quant broke something* — and say which downstream test would confirm it.
* Point at the `H(p,q) = H(p) + D_KL(p‖q)` diagram and explain exactly which term your quantization moved.

If you can quote the definitions but can't grade a quant with them, you have notation. The point of this course is the verdict.

---

*Related: [MLSys Deep Dives](../MLSys%20Deep%20Dives/README.md) · [Gemma 4 Edge Deployment](../../3.%20Edge%20AI/Gemma%204%20Edge%20Deployment/README.md) · [Practical Machine Learning (CS329P) — Model Compression](../../../Phase%203%20-%20Artificial%20Intelligence/Track%20B%20-%20Agentic%20AI%20and%20ML%20Engineering/4.%20ML%20Engineering%20and%20MLOps/Practical%20Machine%20Learning%20%28CS329P%29/Lecture-10.md) · [Phase 5 — ML Systems Engineering Guide](../Guide.md)*
