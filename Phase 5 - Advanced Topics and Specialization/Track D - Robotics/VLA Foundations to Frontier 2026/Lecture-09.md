# Lecture 09 — Behavioral Cloning, Sequence Losses, and Covariate Shift

Learn to distinguish an action predictor that fits recordings from a policy that succeeds when its own actions change the next observation.

**Prerequisites:** Lectures 04–08; gradients, probability, train/validation splits. **Effort:** setup 30 minutes, implementation 3 hours, analysis 90 minutes. **Hardware:** CPU for the state-based exercise; optional GPU for image observations. **Stack/roles:** model training, data pipelines, and robot applications; robot learning engineer and training systems engineer. **Artifact:** a reproducible imitation-learning experiment with held-out trajectories and rollout failure analysis.

## The supervised objective has a hidden distribution

Let \(h_t=(o_{\leq t},\ell,a_{<t})\) be the information available before action \(a_t\), including instruction \(\ell\). A demonstration dataset contains histories visited by a demonstrator. Behavioral cloning (BC) minimizes

$$
\mathcal L_{BC}(\theta)=\mathbb E_{(h,a)\sim D}[-\log\pi_\theta(a\mid h)].
$$

The deployed policy visits a different distribution because its previous actions determine its current state. Lower held-out loss on demonstrator histories does not establish robustness to those new histories. The classic DAgger analysis formalizes this sequential prediction problem and motivates collecting expert labels on states visited by the learner. Its guarantees rely on explicit learning and expert assumptions; they are not a guarantee for arbitrary learned robot controllers. [Ross, Gordon, and Bagnell, 2011](https://proceedings.mlr.press/v15/ross11a.html).

For a fixed-variance Gaussian, negative log likelihood differs from squared action error by a scale and constant. This makes MSE a modeling assumption: it asks for the conditional mean. With two equally valid paths around an obstacle, the mean may point into the obstacle. A mixture density, discretized distribution, diffusion model, or flow model can represent alternatives. A deterministic policy can also work if its inputs disambiguate intent. Expressivity and missing information are different problems.

Use action-specific losses. A gripper category can use cross-entropy; a continuous joint target can use a normalized regression loss. Rotations require care: Euclidean distance between Euler angles treats the wrap boundary incorrectly, while quaternion distance must respect the equivalence of \(q\) and \(-q\). Do not average meters, radians, and a gripper bit without defining scales and weights.

## Sequences, masks, and observable information

For padded action targets \(A\in\mathbb R^{B\times H\times d}\), use a validity mask \(M\) and explicit dimension weights \(w_j\):

$$
\mathcal L=\frac{\sum_{b,k,j}M_{bkj}w_j\,\ell(\hat A_{bkj},A_{bkj})}
{\sum_{b,k,j}M_{bkj}w_j}.
$$

Require finite nonnegative weights and a strictly positive denominator. An entirely padded batch must be rejected or skipped under an explicit distributed-step rule; it is not a zero-loss training example. Masking by multiplication does not repair NaN padded values (`0 × NaN` is still NaN), so sanitize excluded values or select valid coordinates before evaluating the loss.

This is a mean over valid weighted coordinates, not a mean over padded tensor size. State whether long episodes contribute more samples. State whether each episode or each timestep receives equal weight. Both choices can be reasonable, but they optimize different empirical distributions.

Autoregressive action tokens add teacher forcing: training conditions on the correct previous token, whereas inference conditions on a sampled or predicted token. Measure invalid decoded actions and errors across token positions. Scheduled sampling changes the objective and needs its own justification; it does not automatically solve state-distribution shift.

Acausal features are a particularly damaging leak. A video encoder that sees frames after the decision time, a centered smoothing filter, or an annotation generated from future task completion can make the learner appear unusually capable. Future information can legitimately label a target, but it must not enter the observation available to the deployed policy.

## Worked example: good one-step loss, poor recovery

Consider an illustrative one-dimensional system \(x_{t+1}=x_t+a_t+d_t\), with target zero and bounded disturbance \(d_t\). The demonstrator uses \(a_t=-x_t\), so nearly all recorded states lie near zero. A learner trained on those states might learn a nearly constant zero action and achieve tiny validation MSE. At deployment, a disturbance moves the state to \(x=0.4\). The demonstrator returns \(-0.4\); the learner returns nearly zero and remains displaced.

The problem is missing recovery coverage, even though the controller's output range can represent the correct action. Collecting more repeated near-zero examples does not fill that gap. Adding expert-labeled states around the disturbance does. For an illustrative independent per-step failure probability of 0.01, the probability of at least one failure over 100 steps is \(1-0.99^{100}\approx0.634\). Real robot errors are correlated, so this arithmetic illustrates horizon sensitivity rather than predicting a robot success rate.

## Build the experiment

First run the supplied [CPU policy lesson](../../../Phase%203%20-%20Artificial%20Intelligence/0.%20AI%20and%20Robot%20Learning%20Foundations/Lab-01-CPU-Policy.md), which trains a small policy and evaluates closed-loop behavior, or [this course's grounding baseline](Lab-Guide.md), which learns nine logits from RGB/instruction/state samples. These are implemented references with different scopes. Neither implements the DAgger study below.

**Exercise / pseudocode; the learner implements and runs this experiment.** Use a small MLP and the state system above before introducing image encoders.

1. Generate 200 seeded episodes of length 40 with initial states near zero and small disturbances. Split by episode into training, validation, and a locked test set. Save generator parameters and seeds.
2. Fit a linear policy and an MLP using only current state. Log MSE, signed bias, maximum action error, and training time. Compare against the analytic controller.
3. Roll out each policy from both familiar and wider initial-state ranges. Report terminal error, fraction within a chosen target tolerance, and clipped actions. Record the tolerance before viewing results.
4. Apply a controlled disturbance at step 10. Draw a plot of state and action over time. Inspect at least one failure rather than relying only on a mean score.
5. Perform one DAgger-style iteration in this simulator: run the learner, label its visited states with the analytic expert, append those examples, and retrain. Save which data came from which policy version.
6. Repeat with action labels deliberately shifted by one timestep. This negative control should change performance; if it does not, investigate whether the environment is too trivial or the target leaks into the features.

Stop after three seeds and one aggregation iteration. This lab establishes a debugging method, not a scaling law. Its CPU version does not evaluate vision, contact dynamics, human expert consistency, or physical safety.

**Acceptance criteria:** train and test episodes are disjoint; masks exclude all padded targets; the shifted-label case is detectable; both one-step and rollout metrics are reported; the conclusion identifies where aggregation helps or fails. Improvement is not required to pass if the experiment correctly explains a negative result.

## Failure clinic

- **Low loss, wrong direction:** inspect action-frame conversion and sign before changing the network.
- **Zero-action collapse:** compare action histograms and the fraction of idle data; report error on motion and idle subsets separately.
- **Validation improves while rollouts worsen:** evaluate recovery states, instruction interventions, and closed-loop timing.
- **Loss changes with padding length:** the denominator includes invalid coordinates or the mask is broadcast incorrectly.
- **Unexpectedly perfect validation:** search for repeated episodes, future frames, and normalization computed using test data.

## Self-check and answer guidance

1. **Why can more demonstrations fail to improve recovery?** They may add sample count without adding support in the states where the policy fails.
2. **When is MSE statistically appropriate?** When the chosen conditional mean is a useful control decision and the output representation and scaling are meaningful; it is not automatically appropriate for competing action modes.
3. **Does DAgger require executing expert actions at every step?** Its central idea is obtaining expert labels on learner-visited states; collection and mixing schedules vary. Real interventions add operational constraints absent from the toy simulator.
4. **What prepares you for the next lecture?** You can explain why predicting multiple actions changes temporal structure, but does not remove the need for feedback.

[Previous: Lecture 08](Lecture-08.md) · [Course index](README.md) · [Next: action chunking](Lecture-10.md)
