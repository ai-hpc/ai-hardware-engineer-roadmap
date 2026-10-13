# Lecture 31 — Capstone I: Train, Diagnose, and Integrate a Grounded Policy

[Previous](Lecture-30.md) · [Course overview](README.md) · [Next](Lecture-32.md)

**Prerequisites:** Lectures 01–24; complete the CPU lab guide before the simulator extension.

**Effort:** 1 h design, 2–4 h CPU implementation/analysis; 12–24 h additional simulator/adaptation work, excluding long training jobs.

**Evidence cutoff:** October 13, 2026. Numerical examples below are illustrative unless explicitly identified as executed lab results.

## Establish an executable learning loop

The first capstone has a complete CPU route and a separate GPU extension. The CPU route trains a tiny synthetic vision-language-action grounding model supplied with this course. The extension adapts a released VLA to a validated simulator task. Report them as different achievements: the tiny exercise establishes learning and contract fundamentals, while the extension addresses a real model's integration and closed-loop behavior.

**Outcome:** deliver a reproducible experiment in which vision, language, state, and actions have explicit roles, supported by negative controls and a bounded interpretation.

The supplied program has no pretrained transformer, no learned visual backbone, no inverse kinematics, and no physical dynamics. It is intentionally small enough to inspect every learned parameter and gradient. This transparency lets you test whether your evaluation detects a policy that ignores language before spending GPU time on a much larger model.

## Understand the synthetic task

Each sample contains an 8×8 RGB image with one red, one green, and one blue point at distinct positions. The instruction is one of “reach red,” “reach green,” or “reach blue.” A two-dimensional state gives the current position. The target action is the displacement from that state to the instructed point in normalized image coordinates.

A fixed encoder computes a centroid \(v_c\in\mathbb R^2\) for each color channel. A learned matrix \(W\in\mathbb R^{3\times3}\) supplies one attention-logit row per instruction. Prediction is

\[
\alpha_c = \operatorname{softmax}(W_{\ell,:})_c,\qquad
\hat a = \sum_{c=1}^{3}\alpha_c v_c-s.
\]

There are nine learned parameters. The image encoder and state subtraction are hand-designed. Successful training should associate each instruction with its corresponding visual token. A uniform attention policy moves toward the average of all three points and ignores the instruction. A wrongly conditioned policy selects a different target even though its visual encoding is correct.

For loss \(L=\tfrac12\|\hat a-a\|^2\), let \(\bar v=\sum_c\alpha_c v_c\) and \(e=\hat a-a\). The active-row gradient is

\[
\frac{\partial L}{\partial W_{\ell,c}}
=\alpha_c e^\top(v_c-\bar v).
\]

This follows from the softmax Jacobian. The supplied finite-difference test checks the analytical implementation independently. Since the action has two coordinates, half the squared Euclidean error also equals mean squared error per coordinate here; that equivalence would change with dimensionality.

## Run the baseline and inspect evidence

Follow [Lab-Guide.md](Lab-Guide.md) for commands tested in this repository. The default experiment uses 240 procedural training samples, 120 test samples from another seed, 30 epochs, and SGD. Different procedural seeds do not guarantee disjoint visual support or establish out-of-distribution generalization; both datasets use the same generator and task family.

The executed reference run produced held-out per-coordinate MSE about 0.000385, versus 0.284165 with uniform attention and 0.780895 with deliberately wrong instructions. These are measurements of the synthetic exercise only. They say nothing about LIBERO success, Isaac control, real-world grasping, or foundation-model quality.

Inspect the three attention rows and the learning curve endpoints. Predict the expected pattern before reading the values. Then intervene: swap instruction labels at test time, remove one color, shift the state, and change the image size. Missing colors should trigger the encoder's explicit contract error rather than silently yielding a centroid. A state translation should produce the corresponding opposite change in displacement.

## Make one meaningful extension

Choose a change that tests a new hypothesis, not merely more epochs. Possibilities include distractor points of the same color, partial occlusion, ambiguous language, a learned visual feature extractor, or a multi-step closed-loop point robot. Each change alters what the baseline can represent.

If two same-color objects appear, a color centroid lies between them. The baseline cannot implement “reach the left red object” because it has discarded the information needed to distinguish them. More training on its nine logits cannot recover information removed by the encoder. This is an architectural limitation, not an optimizer bug.

For a closed-loop point robot extension, define dynamics, action clipping, termination, and an observation model. Recompute observations after each action. Compare one-shot target error with completion under disturbed dynamics. Keep this explicitly synthetic; it is an educational control environment, not an Isaac result.

The extension report should include a hypothesis, baseline, changed representation or mechanism, held-out fixture generation, negative control, and failure interpretation. A failed hypothesis with correct evidence earns more credit than an unsupported success claim.

## GPU extension: one released policy, one validated task

Choose an artifact from the [frontier register](Frontier-Register.md) with accessible weights and an upstream adaptation recipe. Favor a supported path whose action and observation contracts you can satisfy. Record code commit, checkpoint revision, base model, normalizer, dataset schema, license files, and runtime. Do not select a partner-only model and substitute another checkpoint without changing the experiment's title.

Validate the Isaac task using Lecture 20 before importing a model. Replay a demonstration through its controller; test two instructions in the same scene; confirm reset isolation. Construct episode/session-level train and validation splits, then reserve a final evaluation set. Select adapters or full fine-tuning based on the model's supported recipe and measured memory budget.

Training commands depend on the chosen repository and pinned revision. This course deliberately links upstream recipes rather than inventing a universal VLA training CLI. In your submitted artifact, include the exact command you actually executed and its log. Set a cost/step limit before starting. A completed small smoke test may validate the pipeline, but it is not evidence of useful adaptation.

Evaluate at least a reference checkpoint and the adapted candidate under paired task starts. Include wrong-language, stale-observation, and action-scale negative cases in offline or simulated evaluation. If the model cannot satisfy the task contract, report that incompatibility rather than coercing dimensions until it runs.

## Completion and oral defense

Submit source, environment record, raw outputs, a data contract, learning diagnostics, and a report that separates CPU and GPU evidence. Another learner should reproduce the CPU baseline without downloading weights. A GPU extension should identify any inaccessible assets or data that prevent full reproduction.

Be prepared to answer:

1. Which parts are learned? In the baseline, only instruction-to-color attention logits; centroids and state subtraction are fixed.
2. What does the wrong-language intervention establish? Performance depends on the instruction on this task distribution; it does not establish arbitrary language understanding.
3. Why does a same-color distractor expose a limitation? The encoder collapses distinct objects into one centroid, losing identity and spatial relations.
4. What changes when you close the loop? Actions alter future observations, so error recovery, dynamics, latency, and covariate shift become part of the evaluation.

---

[Previous](Lecture-30.md) · [Lab guide](Lab-Guide.md) · [Next](Lecture-32.md)
