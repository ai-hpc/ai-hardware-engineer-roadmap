# Lecture 06 — Speculative decoding without confusing speed and correctness

[Previous: inspect a model](Lecture-05.md) · [Course index](README.md) · [Next: edge systems and capstone](Lecture-07.md)

Autoregressive generation depends on earlier accepted tokens. Speculative
methods propose several tokens cheaply, then use a target model to verify them.
This can produce more accepted progress per expensive target pass, but the
benefit depends on acceptance, draft cost, verification cost and workload.

**Prerequisites:** probabilities that sum to one, conditional probability and
the timing contract from Lecture 01. Your goal is to distinguish distributional
correctness from token-by-token identity and from a measured speedup. Allow one
hour, plus the optional inference lab exercise.

## What “exact” means

Let the target distribution for the next token be p and the draft distribution
be q. The standard rejection-sampling construction accepts a proposed token x
with probability `min(1,p(x)/q(x))`, then samples from the appropriate residual
distribution after rejection. The [speculative-sampling paper](https://arxiv.org/abs/2302.01318)
establishes a distribution-preserving construction under its assumptions.

It does not promise the same sampled token sequence under the same random seed
when random draws are consumed differently. Greedy verification can preserve
target argmax decisions with matching numerical and tie-breaking behavior, but
quantization, changed logits processing, stop handling or a different target
artifact can change those decisions. State the contract instead of using
“lossless” as a substitute for it.

For a two-token vocabulary, let `p=[0.8,0.2]` and `q=[0.5,0.5]`. Proposals of the
first token are accepted with probability one, and proposals of the second with
probability 0.4. Total acceptance is `0.5×1+0.5×0.4=0.7`. Rejection contributes
the missing 0.3 probability mass to the first token through the residual rule,
recovering p. Accepting every proposal would instead return q.

The [CPU inference contracts lab](../AI%20Inference%20Engineer%202026/Part%201%20-%20Fundamentals/labs/README.md)
checks the exact rejection identity on a tiny distribution. This is stronger
than a noisy empirical frequency comparison for that mathematical example;
it does not validate every server's implementation.

## Work through the speed calculation

Suppose K=4 proposals have a constant conditional acceptance probability a=0.8.
If verification can also produce a bonus/correction token, the simplified
expected progress is:

```text
E[tokens per cycle] = 1 + a + a² + a³ + a⁴ = 3.3616
```

Assume ordinary target decode costs one time unit, each draft step costs 0.1,
and the batched target verification costs 1.3. Expected speedup in this
illustrative model is `3.3616/(4×0.1+1.3)≈1.98`. It is not 3.36×: drafting and
verification are not free. EOS truncation, variable acceptance, queueing and
memory pressure require a more complete model.

**Predict:** if verification cost rises to 3.2 with everything else unchanged,
is speculation still faster?

<details markdown="1">
<summary>Answer</summary>

The denominator becomes 3.6, giving about 0.934×: a slowdown. A longer draft block
can increase accepted progress while making the complete cycle worse. Measure
accepted tokens per cycle and cycle duration together.

</details>

## Drafting methods and current artifacts

An MTP head, an autoregressive drafter and a block-diffusion drafter propose work
in different ways. Their existence does not establish that a specific target,
tokenizer, sampling mode and runtime are compatible. A drafter must be matched
to the required target representation and feature interface.

The [pinned DFlash repository](https://github.com/z-lab/dflash/blob/07ebd93db9f472af339b644bb70221ad8428328a/README.md)
now documents both DFlash and DFlash 2, with distinct checkpoints and backend
coverage. Its block drafting exposes more parallel work than sequential drafting,
but a published speedup applies to the reported model, hardware and experiment.
There is no universal “2–6×” guarantee.

Attention kernels affect verification cost, but they are a different component
from the proposal/acceptance algorithm. Use the [FlashAttention course](../../../Phase%204%20-%20Track%20C%20-%20DL%20Inference%20Optimization/02%20-%20Kernel%20Engineering/FlashAttention%20Course/Guide.md)
for online softmax, masks, gradients and current FA3/FA4 boundaries.

## State and quality remain part of the algorithm

Rejected work must leave the target and draft state at the accepted prefix.
Hybrid models may carry recurrent matrices and convolution buffers in addition
to token-indexed caches. The [hybrid-state lab](../GLM-5.3-Flash%20Architecture%20Mastery/CPU-Hybrid-State-Lab.md)
contains partial-rollback counterexamples. Token output on one short prompt
cannot establish correct rollback.

Keep target and drafter quantization experiments separate. Quantizing only the
drafter can reduce acceptance while an exact verifier still samples the same
fixed target distribution. Quantizing the target changes the distribution being
preserved and needs a quality comparison against the original target. A shorter
acceptance length alone does not identify which model changed.

**Exit check:** show the rejection identity, one configuration where speculation
slows down, and a continuation check after rejection. If GPU serving is available,
report baseline and speculative runs with the same target and workload; otherwise
retain the CPU mathematical result and mark serving unexecuted. Reviewed October
18, 2026; source-reported results are not local benchmarks.
