# Lecture 04 — Speculative decoding as a cost experiment

**Input:** a correct target-only runtime. **Output:** a draft/verify cost model and controlled comparison. [Previous](Lecture-03.md) · [Next: multimodal integration](Lecture-05.md).

## 1. A proposal is not a committed token

A drafter proposes a sequence; the target verifies it and commits an accepted prefix, then handles rejection according to the decoding algorithm. A faster small model is useful only when its proposals agree often enough and verification amortizes their cost. Tokenizer/template compatibility and the runtime's exact algorithm matter.

For sampled decoding, the classic exact rejection-sampling algorithm accepts a draft token with probability `min(1, p(x)/q(x))`, where `p` and `q` are target and draft distributions under the same prefix and sampling transformations. Rejection uses the normalized positive residual `max(p-q, 0)`. This preserves the target distribution under the algorithm's assumptions; it does not promise identical sampled strings for the same random seed. See the original [speculative decoding paper](https://arxiv.org/abs/2211.17192).

## 2. Derive the break-even condition

Let `D` be time spent drafting a block, `V` target verification time, `O` orchestration overhead, and `C` the number of tokens actually committed in that iteration, including any correction/bonus token according to the implementation. Then average cost per committed token is approximately:

\[
t_{spec} = rac{\sum_i (D_i+V_i+O_i)}{\sum_i C_i}.
\]

Compare this with measured target-only time per token for the same workload. Do not substitute proposed tokens or a headline acceptance percentage in the denominator.

**Illustrative calculation:** drafting 4 tokens costs 8 ms, verification 14 ms, overhead 2 ms, and the measured mean is 3 committed tokens per iteration. Cost is 8 ms/token. If target-only costs 10 ms/token, the estimated speedup is 1.25×. If only 2 tokens are committed on average, cost rises to 12 ms/token and speculation loses.

Acceptance at position three depends on the earlier prefix being accepted. A single scalar acceptance rate multiplied by draft length generally does not recover expected accepted-prefix length.

## 3. Implement the accounting before the optimization

For each verification iteration log request ID, proposed count, accepted-prefix count, committed count, draft time, verification time, and total wall time. Preserve token IDs for a small debugging run. Distinguish accepted draft tokens from correction/bonus tokens.

Check memory too: drafter weights, state/cache, candidate buffers, and target verification workspace may push a fitting target-only service into OOM or paging. Increased bandwidth contention can erase compute savings on a Jetson.

## 4. Compare three workloads

Use repetitive text, diverse held-out instructions, and multimodal prompts if the implementation supports them. Sweep draft length. Hold target sampling and output limits fixed. A draft that helps templated responses may be poor for unfamiliar visual descriptions. Record queueing/TTFT as well as decode speed.

## Lab and acceptance

First validate greedy target-only versus speculative output under the runtime's deterministic constraints. For sampling, validate the acceptance/correction implementation on a tiny vocabulary where the expected distribution can be computed; do not demand token-for-token identity from differently consumed random streams. Submit accepted-prefix distributions, cost per committed token, quality, peak memory, and a draft length that loses. The next lecture uses the winning target-only or speculative configuration, whichever the evidence supports.
