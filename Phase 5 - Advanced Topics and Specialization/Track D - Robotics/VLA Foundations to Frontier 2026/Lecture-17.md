# Lecture 17 — Offline and Online RL Post-Training for VLAs

Learn when reward-driven improvement is justified, how policy parameterization changes the algorithm, and how to detect gains caused by a faulty reward or evaluation protocol.

**Prerequisites:** Lectures 09, 12, and 16; Markov decision processes, discounted returns, and gradients. **Effort:** setup 1 hour, simulation lab 5 hours, analysis 2 hours. **Hardware:** CPU for a small discrete/state environment; optional GPU for a VLA actor and critic. **Stack/roles:** learning algorithms, simulation, training systems, and operational evaluation; robot learning and RL infrastructure engineers. **Artifact:** a reproducible post-training study with reward audit, baseline, and held-out rollout evidence.

## Improvement requires an objective beyond imitation

BC reproduces the demonstrated distribution. RL optimizes a return such as \(J(\pi)=\mathbb E_\pi[\sum_t\gamma^t r_t]\). More efficient task completion, recovery from learner-specific errors, and reduction of observed failures can motivate post-training. They also require a reliable way to distinguish success from apparent progress.

Offline RL uses fixed logged experience. Online RL collects new experience as the policy changes. A hybrid process alternates collection and updates. “Online” does not mean every robot action triggers an immediate gradient update. Save the policy version for every collected episode and distinguish human corrections from autonomous actions.

Rewards should describe the intended task, not a convenient visual proxy. A reward for moving a mug toward a sink might encourage dropping it nearby; a success classifier can be fooled by occlusion; a time penalty can favor early termination. Compare reward against independently labeled outcomes, including deliberately adversarial examples.

## Critics and support

An action-value critic estimates \(Q(s,a)\); a value critic estimates \(V(s)\). A one-step target is \(r+\gamma(1-d)V(s')\), with \(d\) representing an actual terminal condition. Treating an artificial time limit as physical termination can bias targets; define whether bootstrapping is appropriate for the task.

Offline critics can assign optimistic values to actions never represented in the dataset. Keeping policy updates near supported behavior, using conservative objectives, or extracting policies from in-dataset advantages can reduce this problem, but none creates missing evidence about unseen actions. Implicit Q-Learning is a primary example of an offline approach that avoids querying out-of-distribution actions during its value-learning updates and then extracts a policy through advantage weighting. [IQL, 2021](https://arxiv.org/abs/2110.06169).

Chunk execution changes the critic's decision interval. If one decision commits to \(K\) physical steps, a corresponding target is \(\sum_{i=0}^{K-1}\gamma^i r_{t+i}+\gamma^K V(s_{t+K})\), with bootstrapping removed at a true terminal event. If execution is interrupted after fewer steps, use the actual duration and successor state. An overlapping predicted suffix that never executes cannot be credited as though it caused the observed reward. Save the executed prefix and intervention boundary so training can reconstruct what happened.

For rewards `[1, 2]`, per-step discount 0.9 and successor value 10, a two-step nonterminal chunk target is `1 + 0.9×2 + 0.9²×10 = 10.9`. At a true terminal event it is 2.8. Using `γ` instead of `γ²`, or crediting a predicted but unexecuted suffix, solves a different problem. The [Deep RL CPU lab](../Deep%20RL%20for%20Robot%20Learning/CPU-Lab.md) executes this duration-aware target and termination/truncation checks; its optional CPU framework tests also check latent-chain likelihoods and policy gradients. These are numerical checks, not pretrained VLA post-training.

For a teaching weighted imitation objective, define \(A=Q-V\) and \(w=\min(w_{max},e^{A/\beta})\). Optimize \(\mathbb E[w\,\ell_{BC}]\). Smaller \(\beta\) concentrates weight on estimated better actions. Critic errors are then amplified, so record the weight distribution and effective sample size

$$
N_{eff}=\frac{(\sum_i w_i)^2}{\sum_i w_i^2}.
$$

An apparent million-example dataset can behave like a tiny one when nearly all weight falls on a few transitions.

## Why a flow loss is not a PPO log probability

PPO uses probability ratios such as \(\rho=\pi_\theta(a\mid s)/\pi_{old}(a\mid s)\). For an autoregressive token policy, log probabilities are explicitly available, although token/action grouping still matters. [PPO, 2017](https://arxiv.org/abs/1707.06347).

A flow policy maps random noise through an ODE. Its supervised velocity MSE is not the exact negative log density of the resulting action. Setting `logprob = -flow_loss` and inserting it into PPO silently changes the mathematics. Under appropriate smoothness and invertibility assumptions, continuous change of variables can compute density using a divergence integral, but this adds computation and numerical issues. The continuous ODE density formula is not automatically the exact density of a finite Euler implementation, whose step Jacobians differ. Other research methods introduce stochastic transition formulations or explicit surrogate objectives. State exactly which quantity is optimized and which probability, if any, it represents.

Alternatives include advantage-weighted flow regression, conditioning the policy on quality/advantage indicators, value-guided candidate selection, or training a smaller residual controller. These choices differ in support constraints, critic use, compute cost, and whether actor parameters change. Sampling several candidates and choosing one with a critic is inference-time selection, even if that critic was learned with RL.

## Verified development through October 13, 2026

RECAP, described in the π*₀.₆ report submitted November 18, 2025, combines demonstrations, autonomous experience, and optional teleoperated corrections. It learns values and conditions the policy on an advantage-based improvement indicator. This provides a concrete alternative to requiring a tractable action log likelihood. [π*₀.₆ / RECAP, v2](https://arxiv.org/abs/2511.14759v2).

The 2026 frontier includes several distinct research directions. ForesightFlow learns action and success-potential trajectories with different weighting for action and potential coordinates. Z-1 studies GRPO-style post-training of a flow-based VLA in RoboCasa. TEMPO studies different update rates for semantic projection and action components while freezing the backbone. These are primary preprints, not interchangeable algorithms or independently reproduced course results. Read their method sections before attempting reproduction. [ForesightFlow, June 3, 2026](https://arxiv.org/abs/2606.04968), [Z-1, June 30, 2026](https://arxiv.org/abs/2606.31846), [TEMPO, August 7, 2026](https://arxiv.org/abs/2608.07314).

For each, ask: what is the policy distribution, what produces the training signal, which parameters change, where does experience come from, and how are reward and generalization evaluated? A reported benchmark gain is evidence for that setting, not a universal ranking.

## Worked example: apparent improvement through reward leakage

Imagine 100 illustrative episodes with 60 true successes. A visual reward labels all 60 successes plus 20 failures as successful, yielding an apparent 80% score. An RL policy may exploit the misleading visual configuration more often and increase reward while real completion decreases. Evaluating with the same reward model cannot detect that failure; independent task-state or human-reviewed labels can.

For advantage weights \((1,1,1,17)\), \(N_{eff}=400/292\approx1.37\) although four samples exist. Logging only mean weight hides this concentration. Clip or temper weights as an experimental choice, then measure whether the change improves robust outcomes rather than only the weighted loss.

## Build a bounded post-training study

**Exercise / pseudocode.** Use a small simulated reaching or object-placement environment with explicit true success and an intentionally imperfect reward proxy. Start with a BC baseline; do not begin this exercise on physical hardware.

1. Generate mixed-quality logged episodes and retain terminal, truncation, policy-version, and intervention fields. Hold out environment seeds and one nuisance condition.
2. Train a critic and evaluate its calibration/ranking on held-out successes and failures. Compare learned values with empirical returns where repeated states permit it.
3. Compare BC, success-filtered BC, and advantage-weighted regression using identical logged data. Track weight concentration and effective sample size.
4. Optionally collect one new batch from the updated policy and repeat. Preserve a fixed held-out evaluation set and a baseline checkpoint.
5. Introduce the flawed visual reward and show how independent success measurement detects exploitation. Add a truncation-label error as a second negative control.
6. If using a flow actor, implement weighted velocity regression and label it precisely. Do not call it PPO. A PPO extension must define and validate its likelihood or surrogate construction separately.

Stop after one online collection/update iteration and three seeds. **Acceptance criteria:** task success is independently measured; return and success are both reported; offline and online data budgets are explicit; critic errors and weight concentration are visible; the flawed reward is detected. Improved performance is not required if the study correctly identifies why an update fails.

For a later physical extension, use the existing robot application's action limits, human override, bounded workspace, and rollback mechanism. Stage the candidate through recorded-data checks, simulation, and supervised limited deployment. Keep these execution gates separate from reward optimization; optimizing a penalty alone does not enforce a hard actuator constraint.

## Failure clinic and self-check

**Failure clinic:** critic values rising while success falls suggest reward or extrapolation errors; few actions dominating updates suggest concentrated weights; degraded language response suggests representation drift; unstable online learning may arise from policy-version mixing or stale rollouts.

1. **Why is negative flow MSE not an exact log probability?** It measures prediction error along sampled interpolation points, rather than the density transformation of final actions.
2. **Why retain failures?** They train value and outcome estimates and reveal recoverable states; they need not all be positive imitation targets.
3. **Does online RL eliminate distribution shift?** It provides new experience, but data, reward, and evaluation shifts remain.
4. **What establishes a post-training improvement?** Matched evaluation conditions, independent success labels, disclosed data/compute budgets, uncertainty, and no violation of the execution contract.

[Previous: fine-tuning](Lecture-16.md) · [Course index](README.md) · [Next: training infrastructure](Lecture-18.md)
