# Module 08 — Behavior Preservation: Quality, Fidelity and Acceptance

**Collection:** [Hardware-Aware LLM Quantization](README.md) | **Previous:** [Previous lesson](Lecture-07.md) | **Next:** [Next lesson](Lecture-09.md)

Three questions need different measurements: does the model solve the task, does
it resemble the reference distribution, and does a drafter agree with its target?
Calling all three “quality” hides important failures.

## 1. Define the reference and the observation

Use a pinned reference model, tokenizer, chat template, logits processing and
context. For teacher-forced fidelity comparisons, both models receive the same
prefix tokens. Free-running generations diverge in prefixes, so their next-token
KL is a different experiment. Reference fidelity itself does not establish task
correctness: the reference can be wrong.

| Question | Useful measures | What the measure does not prove |
|---|---|---|
| Labeled task quality | exact match, task loss, per-slice errors | equivalence to the reference |
| Language-model likelihood | token-weighted NLL/perplexity | every capability or unchanged distribution |
| Reference fidelity | full-vocabulary KL, TV, logit/output error | factual correctness |
| Decisions | top-1 agreement with tie policy | calibrated probabilities |
| Speculative efficiency | accepted draft fraction, emitted tokens/cycle | target task quality |

## 2. Perplexity and KL, calculated correctly

For valid scored tokens, `NLL=-sum(log p(token))/N`, `PPL=exp(NLL)`. Aggregate token
loss sums and counts, not the arithmetic mean of per-batch perplexities. Shift
causal logits/labels consistently, exclude padding, and state whether prompt
positions are scored. Tokenizers must match before comparing token-level PPL.

For one shared prefix, `KL(p||q)=sum_v p_v(log p_v-log q_v)`. Compute log-softmax in
adequate precision. A top-k API response omits tail mass and cannot directly supply
full-vocabulary KL or a normalized perplexity calculation.

A CPU sketch for causal logits `[B,T,V]` and a validity mask `[B,T]` is:

```python
# Both models receive identical input_ids and attention_mask in eval mode.
lp = reference_logits[:, :-1].float().log_softmax(-1)
lq = candidate_logits[:, :-1].float().log_softmax(-1)
valid = attention_mask[:, 1:].bool() & score_mask[:, 1:].bool()
# The caller must also prevent scoring a token across a packed-document boundary.
per_position_kl = (lp.exp() * (lp - lq)).sum(-1)
kl_sum = per_position_kl[valid].double().sum()
count = valid.sum()  # accumulate both across batches, then divide; reject count=0
```

Teacher-forced sequence KL decomposes into expected conditional KL when contexts
follow the reference distribution. A convenient corpus average is a useful
empirical metric, not automatically that sequence-level expectation.

**Why several metrics?** Two models can assign the same probability to each observed
label and redistribute probability among unobserved tokens: PPL matches, KL need
not. Conversely, near a tie, arbitrarily small KL can change top-1. There is no
universal “0.01 nats guarantees 99% agreement” threshold.

## 3. What acceptance actually measures

For a **fixed prefix**, target distribution `p`, drafter distribution `q`, and a
proposal `x~q`, exact speculative sampling accepts with probability
`min(1,p(x)/q(x))`. Its expected acceptance is:

```text
alpha(prefix) = sum_x q(x) min(1,p(x)/q(x))
              = sum_x min(p(x),q(x)) = 1 - TV(p,q)
```

Terms with `q(x)=0` contribute zero; never divide by zero for an unproposed token.
On rejection, sample from normalized `max(p-q,0)`. Accepted mass plus corrected
mass is exactly p. The [original algorithm](https://arxiv.org/abs/2211.17192)
requires correct distributions and verification/state handling.

This identity relates **drafter q to target p**, not a quantized target to its
unquantized reference r. Even with a fixed drafter, a change in `TV(p,q)` does not
identify `TV(p,r)`. If the drafter changes, target-quality inference is weaker still.

**Counterexample:** `r=(0.9,0.1)`, `q=(0.1,0.9)`. Changing the target from r to q
raises acceptance to 1 while moving it far from r. High acceptance is not fidelity.

## 4. Tokens emitted per cycle are not a measured TV distance

With K proposed tokens and a bonus/correction token, ignoring early EOS and caps:

```text
E[emitted] = 1 + sum_{j=1}^K P(first j proposals are accepted)
```

Only under a homogeneous independent acceptance approximation does this become
`1+alpha+...+alpha^K`. Runtime counters differ: some report accepted drafts, others
include the bonus. Record the definition. Do not invert a mean acceptance length
and label the resulting effective alpha as **measured TV**; contexts, correlation,
dynamic K, branches and EOS violate the simplified inversion.

The DSpark case's 2.792→2.546 change is a drafter ablation, not evidence for a
0.082 increase in target-reference TV. [Pinned evidence](Source-Evidence.md).

## 5. Uncertainty and evaluation gates

The familiar `sqrt(alpha(1-alpha)/N)` standard error assumes independent Bernoulli
trials. Draft tokens within requests are correlated and later positions are
conditionally observed. Use paired prompts and bootstrap independent requests or
sessions; report prompt counts, seeds, lengths, EOS rules and confidence intervals.
There is no universal 8,000-token sample size that makes every small delta credible.

Set gates from product requirements and a development set, before looking at test
results: task accuracy/error limits, critical slices, tail probability changes,
context regimes and reproducibility. Keep metrics separate; one scalar cannot
certify behavior. A statistically indistinguishable score is not proof of equivalence;
predeclare an equivalence/noninferiority margin if that is the intended claim.

## Practice and answer guidance

Run the [CPU contract tests](CPU-Quantization-Lab.md). For `p=(0.8,0.2)` and
`q=(0.2,0.8)`, acceptance is 0.4. Accepted mass is `(0.2,0.2)`, rejected mass is 0.6,
and the correction distribution is `(1,0)`, yielding `(0.8,0.2)` exactly.
Returning a fresh p sample after every rejection would yield `(0.68,0.32)` instead:
that tempting shortcut is biased.

**Deliverable:** a report with task quality, reference fidelity and draft efficiency
in separate columns; identify masks, denominators and independent sampling units.
