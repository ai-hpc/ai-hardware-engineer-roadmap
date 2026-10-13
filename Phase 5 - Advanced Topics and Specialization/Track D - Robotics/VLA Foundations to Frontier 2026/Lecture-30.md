# Lecture 30 — The October 2026 frontier: comparison and critical reading

Turn a fast-moving literature review into an engineering decision whose evidence can be checked and whose limitations are visible.

**Stack:** application requirements, evaluation, model/runtime selection, hardware budgeting. **Roles:** senior research engineer, technical lead, inference engineer. **Prerequisites:** [Lecture 29](Lecture-29.md) and the evaluation modules. **Next:** [Lecture 31](Lecture-31.md). **Evidence cutoff:** **2026-10-13**, not the end of October.

## What “up to date” means in this course

The [frontier register](Frontier-Register.md) records dated primary-source evidence, access type, and reproduction scope. It includes established baselines, 2026 family releases, and recent research. It is a selected evidence map, not a claim that every October paper has been reviewed or that one model is universally best.

A search result's crawl date is not a publication date. A paper first submitted in August and revised on October 6 is October-current evidence but not necessarily an October model launch. A checkpoint modified in October may contain a documentation change rather than new weights. Track first publication, version date, weight release, and date checked separately.

The family studies in Lectures 25–29 establish several trajectories: continuous action generation, more diverse conditioning, compact policies, reusable representations, hierarchical reasoning, and joint world/action modeling. Their practical availability differs. Public code, public weights, public training data, hosted APIs, and partner demonstrations are five separate facts.

## Read a claim through its denominator

For every headline result ask: success on what tasks, from which initial states, over how many trials, with what training exposure, horizon, interventions, and hardware? The answer determines what the claim supports.

An experiment can be zero-shot with respect to a task label while sharing its objects, skills, environment, or embodiment with training. “Unseen” needs a named axis. Write separate columns for unseen objects, layouts, instructions, tasks, embodiments, and dynamics. Do not compress them into a single generalization badge.

Likewise, simulator performance and physical performance have different failure distributions. A clean simulator reset can be free; a real reset can consume most of an operator's day. Deployment studies should count setup, recovery, supervision, and data collection effort alongside model execution.

The RoboCasa365 leaderboard inspected for this course is dated 2026-10-13. Use it as a benchmark-specific snapshot and follow its linked protocol and artifact evidence. A table entry alone does not verify training recipe, weight access, or performance on another benchmark. [Official leaderboard](https://robocasa.ai/leaderboard.html).

## Worked example: uncertainty changes the conclusion

Suppose two **synthetic** pilot experiments report 24/30 and 27/30 successes: 80% and 90%. It is tempting to call the latter a decisive winner. A Wilson interval for a binomial proportion is

\[
\frac{\hat p+z^2/(2n)\ \pm\ z\sqrt{\hat p(1-\hat p)/n+z^2/(4n^2)}}{1+z^2/n},
\]

with \(z\approx1.96\) for a conventional 95% interval. The approximate intervals are 62.7–90.5% and 74.4–96.5%. They overlap substantially. Overlap alone is not a hypothesis test, but these pilots plainly do not support a precise universal 10-point advantage.

If both models use the same initial states, retain paired outcomes. Count cases where only A succeeds and where only B succeeds; the shared successes carry different comparative information. If tasks cluster episodes, use task-level resampling or a hierarchical analysis rather than pretending all rollouts are independent.

A deployment choice may still be clear. Suppose A meets a 100 ms p95 budget and B takes 300 ms. If the application cannot tolerate B's delay, its higher pilot success is insufficient. Define feasibility constraints before computing a weighted score, or a large accuracy weight may hide an unacceptable timing violation.

## Precision is an experimental variable

The September 21, 2026 VLAQuantBench paper studies interactions among quantization scope, numerical format, and calibration across closed-loop experiments. Its findings include recipe-dependent failures and recoveries, rather than a universal ordering of sensitive layers. The paper links code, configurations, and episode records. [VLAQuantBench](https://arxiv.org/abs/2609.25376).

The engineering lesson is to preserve a high-precision baseline and measure the full control loop. Weight-file size, fake-quantization accuracy, and real-kernel speed are different outputs. Quantization changes action errors; those errors change future camera images; later policy inputs therefore diverge. Offline action similarity is useful for diagnosis but cannot replace closed-loop testing.

A format or optimization validated for an LLM checkpoint does not establish compatibility with a VLA's image encoder, continuous action head, normalization pipeline, or execution backend. Specify which tensors change precision and which remain protected. Record whether numerical conversion uses actual device kernels or a floating-point emulation.

Estimate end-to-end benefit with Amdahl's law. If optimized layers account for 40% of request time and become twice as fast, overall speedup is

\[
S=\frac{1}{0.6+0.4/2}=1.25.
\]

This analytical estimate assumes unchanged overhead and no new bottleneck. Measure again when image transfers, kernel launches, or simulator contention become dominant.

## Build it: a review that can reject its own favorite

**Effort:** 1 hour source collection, 3–5 hours experiment/reproduction work, 2 hours writing. **Hardware:** CPU for source and protocol auditing; model-specific resources for optional reruns. Stop after auditing three candidate systems and executing one supported pilot. Do not buy hardware or start a large training job to fill missing evidence.

1. Choose an application with measurable requirements: action space, cameras, memory limit, decision deadline, target task distribution, and acceptable intervention rate. Name which are hard constraints.
2. Choose three candidates from the register: one reproducible baseline, one recent accessible model, and one restricted-access research reference. Explain why each informs the decision.
3. For every source, capture publication date, revision, claim, supporting section/table, access status, and a local evidence note. Separate author-reported results, your measurements, analytical estimates, and unknowns.
4. Audit each reported benchmark. Create a protocol matrix with suite version, task count, seeds, horizons, success detector, camera inputs, training data, adaptation budget, and interventions. Mark noncomparable rows explicitly.
5. Run one baseline through a supported evaluation path. Save code/weight hashes, normalization data, raw episode outcomes, timing traces, and failures. A reduced CPU path can reproduce calculations and manifest validation, but not model quality.
6. Write a decision memo with one conditional recommendation, one rejected alternative, and the next experiment that could change your conclusion. Include integration cost and access limitations.

For the negative case, introduce a fabricated record with a publication date after 2026-10-13 and another with only a crawl date. Your review process must exclude the first from the snapshot and mark the second unverified rather than guessing a date.

**Exercise / pseudocode:**

```python
for claim in candidate_claims:
    assert claim.primary_source_url
    if claim.publication_date is None:
        claim.status = "date_unverified"
    elif claim.publication_date > cutoff:
        claim.status = "outside_snapshot"
    elif not claim.protocol_matches(target_protocol):
        claim.status = "context_only"
    else:
        claim.status = "eligible_for_comparison"
```

An eligible claim still needs quality assessment; passing a date and protocol filter does not make the result independently replicated.

## Guided variation: change the deployment constraint

Repeat the decision with an offboard GPU server instead of an onboard GPU. Add network delay and failure modes; do not simply substitute server inference latency. Then impose intermittent connectivity. Identify which candidate remains technically feasible and what control functionality must remain local.

For independent transfer, select a newly published paper after completing the course. Add one row to your own register, verify primary dates and artifacts, and propose a controlled test that would falsify its central engineering benefit. Do not overwrite this historical cutoff with later evidence.

## Open problems worth a professional investigation

Long-horizon reliability requires better completion detection and recovery, not just higher one-step accuracy. Cross-embodiment transfer requires explicit control semantics and observation contracts. Language grounding needs tests where words causally change the required action. Memory needs freshness and revision. World models need action-sensitive physical evaluation. Inference optimization needs tail latency and closed-loop correctness. Each is a tractable project when narrowed to one task distribution and one falsifiable hypothesis.

## Self-check and completion

1. **Does the newest paper automatically deserve the baseline slot?** No. Use an accessible, stable, task-compatible baseline and evaluate the new contribution against it.
2. **Does public inference code imply reproducible pretraining?** No. Training data, configurations, compute, and checkpoint provenance may still be unavailable.
3. **Can success rates from different horizons be ranked directly?** No. The horizon changes the opportunity to finish and recover.
4. **What is an honest result when access is unavailable?** A source-backed architecture/access analysis and clearly bounded reproduction plan, not an invented benchmark.

**Ship:** a dated evidence register, protocol comparison matrix, one reproducible pilot or explicitly labeled reduced audit, and a decision memo with uncertainty and a falsifiable next experiment. That artifact demonstrates professional judgment rather than paper-count memorization.
