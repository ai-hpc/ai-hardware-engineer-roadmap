# Module 10 — Speculative Decoding: Preserve the Target, Measure the Cycle

**Collection:** [Hardware-Aware LLM Quantization](README.md) | **Previous:** [Previous lesson](Lecture-09.md) | **Next:** [Next lesson](Lecture-11.md)

A small model proposes tokens. A larger target verifies them together. The drafter
is useful when the cost of proposing and verifying a group is lower than emitting
those tokens with serial target calls. Correct sampling is part of the algorithm.

## 1. Exactness has a precise meaning

At each verified prefix, accept a proposal from q with probability `min(1,p/q)`.
On rejection use normalized `max(p-q,0)`, then discard subsequent unverified draft
state. If all K drafts are accepted, the target can supply a bonus token.
[Algorithm and proof](https://arxiv.org/abs/2211.17192).

This preserves the **selected target distribution p** under the algorithm's
assumptions. It does not restore an original BF16 distribution after target
quantization. Quantizing only the drafter can change acceptance and latency while
leaving the target distribution unchanged. Greedy verification uses matching
target argmax/tie rules; it is a distinct deterministic contract.

Temperature, top-k/top-p and penalties define the distributions being sampled.
Use the actual transformed p and q in the correction, and the right prefix for
each position. Incorrect normalization, stale state, EOS handling, a different
tokenizer or a proposal-only “take it if confidence is high” shortcut breaks the
exactness argument. A fixed seed need not yield the identical text across samplers:
they consume random numbers differently.

## 2. Measure one complete cycle

Let `T1` be serial target step time, `Tverify(K)` verification time,
`Tdraft(K)` drafting time, `Toverhead(K)` rollback/scheduling/communication time,
and `tau(K)` mean emitted tokens per complete cycle under a declared EOS rule.

```text
Tcycle(K) = Tverify(K) + Tdraft(K) + Toverhead(K)
speculative tokens/s = tau(K) / Tcycle(K)
speedup vs serial = tau(K) × T1 / Tcycle(K)
```

A useful **simplified** model sets `Tverify=T1`, `Tdraft=K c T1` and overhead zero:
`S=tau/(1+Kc)`. It is a starting hypothesis, not a hardware law. Verification often
gets more expensive with K; draft tokens share prefixes and can have correlated
acceptance. `tau=1+alpha+...+alpha^K` additionally assumes homogeneous independent
acceptance. [Module 08](Lecture-08.md) gives the general prefix-survival formula.

**Worked synthetic example:** T1=10ms, K=3, draft=3ms, verify=12ms, overhead=1ms,
tau=2.8. Throughput is 175 token/s versus 100 serial: 1.75×. Treating verification
as 10ms and ignoring overhead would overpredict 215.4 token/s.

## 3. Correct interpretation of the published DSpark result

The [pinned community drafter card](Source-Evidence.md) reports:

| Drafter precision choice | Reported emitted tok/s | Reported acceptance length |
|---|---:|---:|
| MLP + O quantized; Q/K/V wide | 147.87 | 2.792 |
| Also quantize Q/K/V | 150.73 | 2.546 |

The observed throughput ratio is **1.01934**, while the acceptance-length ratio is
**0.91189**. If those counters have identical definitions, implied cycle times are
`2.792/147.87≈18.88ms` and `2.546/150.73≈16.89ms`.

This supports a trade between fewer emitted tokens per cycle and shorter cycles.
It does **not** identify a 10.5% reduction in target bytes, a specific TV change,
or an 84% target-quality penalty. Different draft compute, verification lengths
and scheduling can contribute to cycle time. Source-reported throughput can be
analyzed mathematically without pretending to have reproduced its device run.

The card also reports 155.75 tok/s at 2.886 after a separate tuning/evaluation
configuration. Do not combine that number with an unrelated 81.6 tok/s historical
target run to solve for an allegedly measured MTP cost. It is an external drafter,
and baseline/workload alignment must be established first.

## 4. Choose draft depth and precision together

Sweep K over a bounded range supported by your runtime, with a fixed target and
paired prompts. Report mean and tail latency, time to first token, output tokens,
draft/verify/rollback times, rejection location, early EOS and memory. Increasing K
can waste work; the best K depends on the prompt distribution and runtime.

Then change drafter precision and repeat. A cheaper drafter with lower acceptance
can win. A more accurate expensive drafter can lose. Do not forbid drafter FP4
because acceptance fell, and do not ship solely because acceptance rose.

For target quantization, first evaluate target task quality/fidelity independently,
then retune the drafter. Holding q fixed helps control an experiment, but acceptance
against q still does not substitute for comparison with the BF16 reference r.

## 5. Independent CPU proof and hardware extension

The [CPU lab](CPU-Quantization-Lab.md) enumerates categorical accepted and corrected
probability mass. It includes zero-support proposals, identical distributions and
a deliberately wrong rejection rule. These tests establish the sampling identity,
not an LLM engine implementation or a throughput claim.

On supported hardware, add prefix-state checks: a rejected proposal cannot remain
in KV/recurrent state, and rollback must restore the correct request. Compare
output distributions on a small vocabulary and deterministic greedy outputs before
measuring performance. Batching, cache layout and hybrid recurrent state require
their own correctness evidence.

**Completion:** explain the 18.88→16.89ms cycle calculation, distinguish target
quality from draft efficiency, and provide a K sweep or a reproducible CPU proof
plus an explicitly unexecuted device experiment plan.
