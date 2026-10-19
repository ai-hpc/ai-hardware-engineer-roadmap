# Lecture 25 — Physical Intelligence: from π0 to steerable generalists

Learn to separate the architecture, training recipe, released checkpoint, and deployment system behind a model family.

**Stack:** model workload, training software, inference runtime, accelerator memory. **Roles:** robot learning engineer, VLA inference engineer. **Prerequisites:** action representations, flow matching, adaptation, and [Lecture 24](Lecture-24.md). **Next:** [GR00T](Lecture-26.md). **Evidence cutoff:** 2026-10-13; artifact details are in the [frontier register](Frontier-Register.md).

## Why this lineage matters

A robot foundation model is useful only if semantic knowledge reaches a controller without exceeding its timing budget. The π family makes that tension concrete. Its history is also a lesson in scientific reading: a paper, a product demonstration, and a downloadable policy answer different questions.

π0 first appeared on 2024-10-31. It connects a pretrained vision-language model to continuous action generation through flow matching and studies several robot platforms. Code and weights followed on 2025-02-04. These are different milestones. A paper revision in 2026 does not make π0 a newly released 2026 model. [π0 paper](https://arxiv.org/abs/2410.24164), [release announcement](https://www.pi.website/blog/openpi).

π0.5, published 2025-04-22, examines transfer from heterogeneous supervision, including semantic subtasks and robot actions. Its generalization experiments motivate a crucial distinction: understanding an unseen word, operating in a new room, and transferring a manipulation skill to another embodiment are separate tests. [π0.5 paper](https://arxiv.org/abs/2504.16054).

π*0.6 was announced 2025-11-17; its paper followed November 18 and was revised November 19. RECAP combines demonstrations, autonomous experience, and corrections through advantage-conditioned policy learning. Its evaluation separates task success from successful completions per hour: improving throughput requires both reliability and speed. Public paper access does not establish public checkpoint access. [Official index](https://www.pi.website/blog), [π*0.6 paper](https://arxiv.org/abs/2511.14759v2).

MEM was announced 2026-03-03 and submitted March 4, with a March 8 revision. It combines compressed recent video with longer-horizon textual memory. The official index also dates efficient online RL for precise manipulation to March 19. [MEM paper](https://arxiv.org/abs/2603.03596), [research index](https://www.pi.website/blog).

π0.7 was announced and submitted 2026-04-16; paper v2 is April 24. Its reported contribution is steerability through varied context: language, behavior metadata, control conventions, and visual subgoals. The authors report compositional and cross-embodiment experiments. These remain upstream claims under their experimental conditions; the inspected openpi inventory does not provide a π0.7 checkpoint. [π0.7 announcement](https://www.pi.website/blog/pi07), [paper](https://arxiv.org/abs/2604.15483v2), [openpi inventory](https://github.com/Physical-Intelligence/openpi).

**October 18 access update:** PI announced a managed fine-tuning and hosted-inference beta on October 13. Beta service access is distinct from downloadable policy weights; participation requires applying. No service was executed in this course review. [Official announcement](https://www.pi.website/blog/api-beta), [dated update](Current-Update-2026-10-18.md).

The accessible paper describes roughly 5B parameters, including a Gemma3 4B backbone, a history encoder, and an 860M flow action expert. Its experiments generate 50-step chunks with five denoising steps, executing a shorter prefix, with asynchronous subgoal and action updates. Some novel tasks use human language coaching; others use a learned high-level policy. Preserve that distinction when interpreting autonomy. [Architecture and runtime sections](https://arxiv.org/html/2604.15483v2).

## Mental model: conditioning is part of the controller

Write a generic policy as

\[
A_t \sim p_\theta(A\mid I_t,q_t,\ell,z_t,e),
\]

where \(I_t\) contains timestamped camera views, \(q_t\) is proprioception, \(\ell\) is the task, \(z_t\) is optional context, \(e\) specifies embodiment, and \(A_t\in\mathbb R^{H\times d}\) is an action chunk. This is an explanatory abstraction, not a claim that every version uses identical inputs.

The interface must define what each quantity means. A control-mode token cannot convert a joint-position checkpoint into a torque controller. A speed annotation is a learned condition, not a hard bound. A visual subgoal describes appearance, not necessarily a physically reachable state. Engineering therefore has two layers: learned preference inside the distribution, and enforced contracts around its output.

For a simple flow-matching teaching formulation, sample expert action \(A\), Gaussian noise \(\epsilon\), and interpolation time \(s\). Let

\[
X_s=(1-s)\epsilon+sA,\qquad
L=\mathbb E\|v_\theta(X_s,s,c)-(A-\epsilon)\|_2^2.
\]

Integrating the learned vector field from noise to action costs several network evaluations. Different papers reverse the time convention; derive the corresponding target before comparing implementations. Using one convention's interpolation with the other's velocity target trains motion in the wrong direction.

Separate perception cost from action integration:

\[
T_{\text{policy}}=T_{\text{vision+language}}+K T_{\text{action step}}+T_{\text{transforms}}.
\]

If an **illustrative** system spends 36 ms on perception, 4 ms per flow step, and 4 ms on transforms, ten steps cost 80 ms; five cost 60 ms. Halving action steps gives a 1.33× overall speedup, not 2×. A 20 Hz control tick is 50 ms, so neither configuration supports synchronous fresh inference every tick. Chunk execution or scheduling must close that gap, and changed integration quality needs testing.

## Training knowledge without erasing it

Attaching a randomly initialized action module changes the gradients entering a pretrained representation. The knowledge-insulation paper studies this training interaction and motivates protecting useful semantic representations while learning continuous control. It is dated 2025-05-29 on arXiv. [Primary paper](https://arxiv.org/abs/2505.23705).

The engineering experiment is broader than one implementation: compare frozen, jointly trained, and selectively insulated representations under matched data and optimizer budgets. Track a language-sensitive robot test as well as action loss. A policy can lower imitation loss by memorizing the frequent task while becoming less responsive to instructions.

Build a paired scene in which the image stays unchanged and the requested object changes. Hold the random seed fixed for diagnosis; later evaluate across seeds. If actions barely change, investigate whether the language input is dropped, truncated, overwhelmed by a dataset prior, or genuinely unnecessary for that scene. One insensitive example does not diagnose the whole model.

## Build it: checkpoint and contract audit

**Effort:** 45–90 minutes setup, 2–3 hours execution, 1 hour analysis. **Hardware:** CPU for metadata and replay validation; a supported GPU for full policy execution, selected using the pinned upstream requirements. Stop after 100 inference samples and one bounded evaluation suite. Do not start full pretraining for this lesson.

1. Choose a released π0.5 or π0 checkpoint from openpi. Record code commit, weight hashes, configuration, normalization statistics, tokenizer revision, dependencies, and code/weight/backbone terms separately. The repository includes Apache-2.0 code and a separate Gemma license file; retain both in the manifest.
2. Inspect one observation before transforms and after transforms. Record camera identity, shape, dtype, color order, time, state dimension, prompt, and missing-view mask. Compare these with the selected checkpoint's configuration.
3. Use the pinned upstream no-robot inference example, then replace random observations with a recorded episode matching its embodiment. Random images validate plumbing only. The upstream repository documents base and task-adapted checkpoints and a flow-head implementation for π0.5. [Implementation and examples](https://github.com/Physical-Intelligence/openpi).
4. Instrument vision encoding, action generation, input transfer, output transfer, and total request time. Warm up compilation separately. Synchronize device timing correctly; report cold start and steady-state p50/p95 independently.
5. Introduce a wrong normalization-statistics file as a negative case. The adapter should reject its schema or provenance before execution. If the schema matches but distributions differ, flag the mismatch in the manifest and compare output scale offline.
6. Vary the executed prefix length while holding predicted chunk length fixed. Log observation age at action execution, deadline misses, smoothness, and task success. Do not infer behavior quality from inference frequency.

**Exercise / pseudocode, not an upstream API:**

```python
record = load_timestamped_observation()
assert record.schema_id == policy_manifest.observation_schema
assert normalizer.digest == policy_manifest.normalizer_digest
chunk = adapter.infer(record)
validate_finite_values_units_and_limits(chunk)
save_trace(record, chunk, stage_timings)
```

The CPU reduction validates metadata, transforms, rejection behavior, and the timing model. It cannot establish model latency, learned skill, or physical transfer.

## Guided variation and independent transfer

First vary only language: “put the red cup in the tray” versus “put the blue cup in the tray.” Next vary only scene layout. Finally vary both. Keep those three result groups separate. This small factorial design helps distinguish semantic following from geometric robustness.

For independent transfer, propose a new two-camera arm setup. Produce a compatibility report before collecting demonstrations: identify reachable action coordinates, camera calibration requirements, gripper convention, and missing source signals. Explain which differences require a transform, which require retraining, and which cannot be inferred from public evidence.

## Failure modes and self-check

Common failures include confusing a base model with a robot-ready checkpoint, changing denoising steps without measuring task success, reusing normalizers across embodiments, and describing visually impressive demos as controlled zero-shot evaluation.

1. **Does π0.7's announcement make openpi π0.5 obsolete for this lab?** No. Public access, correct embodiment support, and reproducible evaluation determine the executable baseline.
2. **Why can more conditioning hurt?** Incorrect, stale, contradictory, or out-of-distribution context can steer the policy away from the intended behavior.
3. **Can lower flow loss prove better language following?** No. Use instruction-controlled tests where different instructions require different actions.
4. **What remains after a successful offline replay?** Closed-loop validation, because the model's actions change future observations and error compounds.

**Ship:** a model/normalizer manifest, annotated data-flow diagram, timing trace, paired language test, and one-page transfer decision. Completion means another learner can reproduce your compatibility checks and distinguish measurements from analytical estimates and upstream claims.
