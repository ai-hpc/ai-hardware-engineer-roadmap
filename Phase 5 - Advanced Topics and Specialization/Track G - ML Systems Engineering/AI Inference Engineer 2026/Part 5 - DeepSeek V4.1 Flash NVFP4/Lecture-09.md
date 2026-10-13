# Lecture 09 — Quality and speculative decoding

**Prerequisites:** Lecture 03's numerical ladder and Lecture 08's workload contract. **Time:** 3–5 hours plus model execution. **Outcome:** assess model quality independently of speed.

## Freeze the evaluation before tuning

Use disjoint development and final evaluation sets. Include your actual task families: text, long context, structured tool arguments, images if supported, and multi-turn state. Record tokenizer, prompt encoder, image preprocessing, sampling, reasoning settings, token budget, and stopping rules.

A lower reasoning budget or shorter output can increase throughput while reducing task success. Hold it fixed during performance comparisons, or report a separate quality/cost frontier.

For a converted checkpoint, distinguish integrity checks, reconstructed tensor fidelity, next-token distribution changes, and end-task quality. The default conversion's publisher does not claim end-to-end evaluation; do not inherit the source model's benchmark scores.

## Paired comparison

Evaluate the same examples under reference and candidate conditions. Report sample counts, task-level scores, paired changes, and uncertainty. If you use token-distribution KL divergence, align token positions and the full probability support; truncated top-k logprobs alone do not establish full-distribution KL.

Small nondeterministic differences do not necessarily imply a bug. Large discrepancies should be localized before broad testing: prompt IDs → tensor layouts/scales → operator outputs → routing → logits → task score.

## Speculation is another implementation

The [official model card](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash) names DSpark as its speculative mechanism. Treat its support and checkpoint components as a separate runtime feature gate. Generic MTP flags do not prove DSpark integration.

For any speculative scheme, measure:

```text
time_per_committed_token =
  (draft_time + verification_time + rollback_and_bookkeeping_time)
  / committed_tokens
```

Count rejected drafts as work, not user-visible throughput. Exact-distribution claims depend on the verification/sampling algorithm and implementation; greedy output equality is useful but does not establish correctness under stochastic sampling.

## Lab

1. Write the quality budget before viewing candidate scores.
2. Run the reference and converted candidate on identical held-out cases.
3. Inspect at least five regressions, even if the aggregate passes.
4. Repeat with speculation disabled and enabled, if supported.
5. Compare accepted draft length, committed tokens, latency, and peak memory.
6. Re-run the mixed-load benchmark: verification batches can change scheduling behavior.

**Negative control:** intentionally alter a prompt template or reconstruction scale in a small isolated test and show that the quality gate catches it.

**Reduced exercise:** implement the evaluation bookkeeping and analyze an explicitly synthetic paired result table. Record full model evaluation as not run.

## Self-check

- Is a high cosine similarity of weight rows sufficient? It does not measure accumulated model error or end-task quality.
- Is a higher draft acceptance rate always faster? Draft and verification costs can outweigh accepted work.
- Why inspect categories separately? An aggregate may hide a multimodal or long-context regression.

**Exit artifact:** an evaluation manifest, paired results, failure examples, and a quality decision. No speed result is accepted before this decision passes.

[Previous](Lecture-08.md) · [Overview](README.md) · [Next: capstone](Lecture-10.md)
