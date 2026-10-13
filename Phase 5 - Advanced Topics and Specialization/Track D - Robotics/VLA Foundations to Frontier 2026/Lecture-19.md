# Lecture 19 — Closed-Loop Evaluation, Benchmarks, and Statistical Evidence

[Previous](Lecture-18.md) · [Course overview](README.md) · [Next](Lecture-20.md)

**Prerequisites:** Lectures 09, 14, and 18; means, variance, and confidence intervals.

**Effort:** 75 min teaching, 30 min setup, 2–3 h CPU analysis; GPU rollouts are an optional extension.

**Evidence cutoff:** October 13, 2026. Numerical examples below are illustrative unless explicitly identified as executed lab results.

## The engineering question

A candidate policy has lower held-out action loss. Should it replace the deployed policy? Only if its behavior improves under the deployment distribution and its constraints still hold. An imitation dataset contains actions selected by a demonstrator. A rollout contains states induced by the learner. Those distributions differ, and a small action error can move the gripper into a state with no useful training examples.

Separate four questions: does the input/output contract hold; does the policy reproduce useful actions on recorded observations; does it complete tasks in closed loop; and does it remain acceptable under the intended operating conditions? Each level can fail while the previous one passes. An evaluation engineer designs evidence for all four, with explicit denominators.

**Outcome:** produce a paired evaluation report whose estimate, uncertainty, task scope, and failure exclusions another engineer can audit.

## Specify the experimental unit

An episode is one attempt with a task, initial state, instruction, policy revision, and seed. Frames within that episode are dependent observations. Counting 10,000 video frames as 10,000 independent trials produces unjustifiably tight uncertainty. A seed is a reproducibility aid, not proof that two frameworks generate identical initial states; store initial states when the simulator supports it.

For task \(j\), let \(s_{ji}\) be a binary success outcome. Estimate its rate as

\[
\hat p_j = \frac{1}{n_j}\sum_i s_{ji},\qquad
\hat p_{\mathrm{macro}} = \frac{1}{J}\sum_j\hat p_j.
\]

A micro-average pools all episodes, weighting tasks by trial count. A macro-average weights tasks equally. Neither is universally correct; the target deployment distribution determines the weights. Report task counts and rates so a reader can recompute the aggregate.

Pair the reference and candidate on the same task and initial state. The paired difference \(d_{ji}=s^{C}_{ji}-s^{R}_{ji}\) can be -1, 0, or 1. Analyze differences directly; two separate confidence intervals discard useful pairing. A candidate can win on easy tasks and lose on rare contact tasks while preserving the global mean. Keep the per-task table.

## A worked comparison

Suppose task A has 90 attempts: reference succeeds 45 times and candidate 63. Task B has 10 attempts: reference succeeds 8 and candidate 4. Candidate-minus-reference is +20 percentage points on A and -40 on B. The micro difference is +14 points; the macro difference is -10 points. Calling the candidate simply “better” hides the evaluation objective.

For a deployment parity gate, predeclare an allowed loss \(\delta\). A noninferiority claim requires a defensible lower confidence bound on the chosen difference above \(-\delta\). A point estimate of zero with a wide interval is inconclusive, not evidence of parity. Do not pick \(\delta\) after seeing results. Latency, intervention rate, and constraint violations need separate gates; a success-rate confidence interval does not certify them.

Bootstrap resampling must match the target population. Resampling tasks estimates uncertainty over task selection; resampling episodes within each fixed task addresses within-task trial variability. A hierarchical bootstrap resamples both, preserving policy pairing. With very few tasks, neither repeated bootstrap draws nor a large frame count creates missing evidence.

## Understand what a benchmark tests

[LIBERO's dataset documentation](https://libero-project.github.io/datasets) distinguishes spatial, object, goal, and broader task-transfer suites. These probe different generalization axes. Record the suite, task split, demonstrations, evaluation starts, observation modalities, and exact environment revision. A training set containing paraphrases of held-out task names can compromise a language-generalization claim even when episode IDs differ.

[RoboCasa365's official leaderboard](https://robocasa.ai/leaderboard.html), updated October 13, 2026, reports a 50-task multitask comparison with atomic-seen, composite-seen, and composite-unseen splits. Its note about reevaluating GR00T N1.5 with a longer horizon shows why evaluation configuration matters. A row on this table is evidence for that protocol, not a universal ranking of robotics models. Save the benchmark configuration with a report; a moving leaderboard is not an immutable experimental artifact.

Do not silently convert “task never completed before timeout” into “infrastructure error.” Define termination, timeout, invalid reset, policy crash, human intervention, and missing recording before execution. Retain all attempts in a ledger and report both excluded counts and reasons. Simulator crashes may justify reruns under a prespecified rule; policy crashes are behavioral failures.

## Build the report

Run the [CPU lab guide](Lab-Guide.md), then use `labs/contracts.py::paired_task_bootstrap` on synthetic paired rows. The helper computes task-macro differences and a task-resampled percentile interval; it treats observed within-task rates as fixed. It is deliberately **not** a hierarchical inference package. Inspect this limitation before interpreting its output.

Create four synthetic tasks with both wins and losses. Give one task ten times as many trials. Compute macro and micro differences, then repeat with uniform trial counts. Preserve task/seed pairs through every resample. Add a duplicate pair and show that the helper rejects it.

For an independent extension, implement a hierarchical bootstrap and compare intervals over repeated generated datasets with known success probabilities. Investigate coverage empirically; matching the width of one expected answer is insufficient. The acceptance artifact is a report with estimand, resampling unit, raw paired rows, exclusions, confidence procedure, and an interpretation that respects its scope.

## Failure clinic

A near-perfect action MSE with poor task success suggests either wrong closed-loop state distribution or a contract mismatch. Plot per-axis physical errors and inspect the first divergence. Better mean success with worse deployment behavior may be a weighting problem. An apparent major gain after changing timeout or image resolution is a confounded experiment until those factors are controlled.

A checkpoint chosen using the test suite is part of the training process, even without gradients. Keep a development suite for repeated decisions and one untouched final suite. If access is too expensive for a final suite, say that the result is exploratory rather than implying an independent confirmation.

## Self-check and answer guidance

1. Why are 200 frames from one failed grasp not 200 independent failures? They share initial conditions, policy trajectory, and contact history; the sampling unit is at least the episode.
2. Can matching seeds alone establish matched trials? No. Simulator revision, randomization order, and reset implementation can change the realized states.
3. Candidate delta is +1 point, interval [-8, +9], tolerance 3 points. Does it pass noninferiority? No; the lower bound is below -3.
4. Why report latency beside success? Delays change the state-action relationship and can invalidate a policy that succeeds under accelerated or paused simulation.

---

[Previous](Lecture-18.md) · [Lab guide](Lab-Guide.md) · [Next](Lecture-20.md)
