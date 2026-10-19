# Lecture 29 — Reasoning, memory, hierarchy, and world-action models

Choose an intermediate representation for robot decision-making and test whether its extra computation improves action outcomes.

**Stack:** model architecture, inference scheduling, GPU memory, control integration. **Roles:** VLA researcher, embodied inference engineer. **Prerequisites:** [Lecture 28](Lecture-28.md), generative action models, partial observability, and planning. **Next:** [Frontier comparison](Lecture-30.md). **Cutoff:** 2026-10-13.

## Four mechanisms with different jobs

An action policy maps observations to controls. A reasoner proposes decisions or subgoals. A memory mechanism retains information absent from the latest observation. A world model predicts consequences. A system may contain several, but their names should not replace a data-flow diagram.

The central question is causal: does the added mechanism improve the robot's decisions, under a fixed resource budget? A fluent explanation does not establish that the explanation caused a useful action. A plausible video does not establish accurate contact dynamics. A large context window does not establish successful retrieval of an object location seen several minutes earlier.

In partial observability, two identical current images may require different actions because the history differs. Let

\[
h_t=f(h_{t-1},o_t,a_{t-1}),\qquad
a_t\sim\pi(a\mid o_t,h_t,\ell).
\]

The hidden state could be a recurrent tensor, retrieved episodes, a structured task record, or selected past frames. Those choices have different update costs, storage needs, and failure modes. Here `a` must represent the action actually applied, or the history must separately include the requested action, controller override, and execution duration. Updating memory as though a rejected action moved the robot corrupts the next state estimate. Test memory by creating histories that end at the same observation but require different decisions; otherwise a memoryless policy might pass the task.

## What 2026 adds to this discussion

Figure's Helix 02 announcement, dated 2026-01-27, describes a hierarchy spanning goal reasoning, joint targets, and low-level whole-body control. Helix 2.5, announced September 17, reports three behaviors evaluated across 30 unseen homes. Its zero-shot claim concerns homes and objects: behavior-specific adaptation occurred elsewhere. No public weights were verified here. This is a useful example of why hierarchical timing and the exact generalization axis matter even when a system is inaccessible. [Helix 02](https://www.figure.ai/news/helix-02), [Helix 2.5](https://www.figure.ai/news/helix-2-5-zero-shot-30-home-generalization).

DreamZero's paper, first submitted 2026-02-17, studies joint video/action prediction using a pretrained video-diffusion backbone. It reports a 14B model and an optimized closed-loop configuration. Its public repository supplies code and checkpoints, including a distributed inference path. The performance claims belong to the paper's hardware and method configuration. [DreamZero paper](https://arxiv.org/abs/2602.15922), [implementation](https://github.com/dreamzero0/dreamzero).

“Continuous Reasoning for Vision-Language-Action,” submitted 2026-05-29 and revised June 8, investigates reasoning representations aligned with continuous control rather than assuming text is the right intermediate format. Treat this as a research hypothesis with specific experiments, not a universal rejection of language-based planning. [Primary paper](https://arxiv.org/abs/2606.00229).

“In-Context VLA,” submitted 2026-08-06, studies grounded contextual inputs and tool-mediated acquisition of task information. Its reported comparisons challenge unrestricted textual reasoning under its tested conditions. This review verified the paper; executable public weights were not established. [Primary paper](https://arxiv.org/abs/2608.05738).

These studies motivate three distinct intervention tests: remove the future-prediction objective, replace the intermediate representation, or withhold the additional observation acquired by a tool. Merely comparing final model names cannot identify which mechanism helped.

## World models: predictive learning and planning are different uses

A generic latent world model learns

\[
\hat z_{t+1}=F_\psi(z_t,a_t),\qquad
\hat r_t=R_\psi(z_t,a_t).
\]

Model-predictive control can sample action sequences and select

\[
A^*=\arg\max_A\sum_{j=0}^{H-1}\gamma^j\hat r_{t+j},
\]

subject to the application's state/action constraints, then execute only a prefix before observing again. A larger predicted reward does not make an infeasible candidate valid. This simple finite-horizon objective omits a terminal value; disclose that truncation or supply a validated terminal cost/value. Here a model explicitly supports search. A joint world-action policy can instead predict future observations and actions together without running that planning algorithm. An auxiliary predictive loss may improve representations even when no future video is generated at inference. Distinguish all three.

Future prediction can exploit shortcuts. A decoder can produce a visually plausible gripper closure while the imagined object ignores friction. A latent predictor can encode the task label without modeling useful consequences. Evaluate counterfactual actions: do different actions lead to appropriately different predicted outcomes? Check temporal consistency and action sensitivity, not just attractive frames.

Latent prediction saves output bandwidth and potentially compute, but it can hide errors that were visible in pixels. Choose evaluation targets tied to control: object displacement, contact events, grasp state, collision, and task progress. If a representation predicts those reliably, photorealistic reconstruction may be unnecessary for the application.

## Worked example: allocating a 100 ms decision budget

Suppose an **illustrative** controller can tolerate a new decision every 100 ms. Observation processing costs 20 ms and final action decoding 15 ms. That leaves 65 ms for memory retrieval, reasoning, or imagination.

Candidate A evaluates four predicted futures at 12 ms each and spends 5 ms scoring them: total \(20+48+5+15=88\) ms. Candidate B uses a 90 ms text-generation step: total 125 ms, exceeding the budget before jitter. Candidate C predicts one latent future in 8 ms: total 43 ms but may provide less useful look-ahead. These are analytical examples, not measurements of named models.

These timings assume the futures are evaluated serially and all stages belong to one decision. Four candidates may batch on suitable hardware, but their wall time must then be measured; multiplying one-candidate latency by four no longer describes that execution. The 100 ms budget must also state whether it starts at sensor capture or at processing arrival.

The deployment choice depends on the success/latency curve, not only budget feasibility. Compare under both equal wall time and each method's best supported configuration. Also report p95: a method averaging 88 ms can still miss most difficult decisions if its tail exceeds the deadline.

For large generative models, token count and serial work matter. Video patches, temporal frames, action tokens, and denoising steps can each increase inference cost. A 14B BF16 weight-only estimate is 28 GB; actual runtime memory is larger and distributed inference adds communication. The inspected DreamZero instructions describe multi-GPU execution with at least two GPUs, tested on datacenter hardware. A laptop fixture does not validate that runtime. [Deployment prerequisites](https://github.com/dreamzero0/dreamzero).

## Build it: evidence for reasoning value

**Effort:** 1 hour setup, 3–4 hours execution, 2 hours analysis. **Hardware:** CPU for the reduced latent-state fixture; supported GPU/simulator for a full policy study; the DreamZero path requires its pinned distributed setup. **Stopping condition:** complete one predeclared 40-scenario fixture study or one bounded upstream evaluation pilot, not full pretraining.

Build a partially observed sorting environment. An object is briefly visible in one of two boxes, then both boxes close. The final image is identical across conditions. The task is to open the correct box and retrieve the object. Add an optional inspection action that costs time but reveals the location.

1. Implement ground-truth state separately from the observations. Confirm that the final current observation alone does not identify the target box.
2. Compare three controllers: latest-observation only, stored observation memory, and an inspection-using planner. Give each the same task distribution and maximum completion time.
3. Add a lightweight transition model that predicts whether an inspection or opening action reveals the object. Label it a synthetic fixture model; do not call it a reproduction of a learned video model.
4. Measure success, inspections per episode, memory size, decision time, stale-memory errors, and action count. Save failures and the full observation/action history.
5. Corrupt memory in a controlled fraction of episodes, swap the target after the original observation, and delay the inspection result. These negative cases reveal whether the controller can revise its belief.
6. In an optional upstream model study, pin weights and code, use a supported environment, and repeat one matched mechanism ablation where the released implementation permits it. If that ablation requires retraining you cannot afford, report it as untested.

**Exercise / pseudocode:**

```python
belief = update_belief(memory, observation)
if belief.confidence < inspection_threshold:
    candidate = inspect_hidden_region()
else:
    candidate = act_on_most_likely_location(belief)
validate_current_episode_and_action_preconditions(candidate)
execute(candidate)
```

The threshold is a policy parameter to evaluate, not an objective probability of correctness unless confidence has been calibrated. Include acquisition latency in the inspection cost.

## Guided variation and independent transfer

First increase the delay between seeing the object and needing to act. Plot success versus delay while keeping distractor count constant. Then add distractors with delay fixed. This separates retention from retrieval interference.

For transfer, design a mobile manipulation task in which a robot must remember whether it already inspected a drawer. Specify what happens after an episode reset, interrupted task, object movement, or another robot's intervention. The memory should record evidence and freshness, not merely a confident natural-language summary.

## Self-check and completion

1. **Is every world model a robot policy?** No. A model may predict futures without producing executable actions or supplying a planning algorithm.
2. **What demonstrates useful memory?** Better decisions when the current observation is insufficient, plus appropriate revision after new evidence.
3. **Why can reasoning decrease success?** It can add latency, introduce incorrect intermediate assumptions, or consume resources needed by control.
4. **Does visual realism prove physical accuracy?** No. Test action-conditioned effects and task-relevant physical events.
5. **Can one vendor's frequency be compared with another's chunk latency?** Only after aligning measurement boundaries, batch size, hardware, input count, and action semantics.

**Ship:** a mechanism diagram, a partially observed task with hidden-state ground truth, a controlled ablation table, resource measurements, and a failure analysis distinguishing reasoning errors from sensing and control errors.
