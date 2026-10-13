# Lecture 12 — Flow Matching and the π₀ Architecture

Derive a continuous action generator, then connect its equations to a VLM backbone, an action expert, and an executable inference schedule.

**Prerequisites:** Lecture 11; derivatives and numerical integration. **Effort:** setup 45 minutes, lab 4 hours, analysis 2 hours. **Hardware:** CPU for the toy flow; optional GPU for architecture inspection and a pinned pretrained checkpoint. **Stack/roles:** model architecture, training, and accelerator inference; VLA research and inference engineering. **Artifact:** a tested flow sampler plus a tensor and cache contract for a chosen implementation.

## Learn a velocity field

Choose a clean action chunk \(A\), independent noise \(Z\sim\mathcal N(0,I)\), and interpolation time \(\tau\in[0,1]\). In this lecture zero means noise and one means data:

$$
X_\tau=(1-\tau)Z+\tau A,\qquad
U_\tau=\frac{dX_\tau}{d\tau}=A-Z.
$$

Train \(v_\theta(X_\tau,\tau,c)\) to predict \(U_\tau\) using squared error, with action and padding masks. The conditional training target is available without running a numerical solver. The population optimum is the conditional mean velocity at a point, \(v^*(x,\tau,c)=\mathbb E[A-Z\mid X_\tau=x,c]\). Individual sample paths can intersect; the learned marginal field need not recover the exact straight line of any individual training pair. Flow matching supplies the general framework behind this construction. [Lipman et al., 2022](https://arxiv.org/abs/2210.02747).

At inference, draw noise and solve

$$
\frac{dX_\tau}{d\tau}=v_\theta(X_\tau,\tau,c),\qquad X_0\sim\mathcal N(0,I).
$$

Euler integration uses \(X_{i+1}=X_i+\Delta\tau_i v_\theta(X_i,\tau_i,c)\). Higher-order methods can reduce discretization error per step but evaluate the network more often. Compare network evaluations, not just solver steps. Some repositories use the opposite time direction and target \(Z-A\). Both are valid when all conventions agree; copying the loss from one and the sampler from the other reverses generation.

## Where π₀ fits

The original π₀ model combines a pretrained PaliGemma VLM with a separately parameterized action expert that processes robotics tokens. It models continuous chunks using flow matching. The paper describes a 3B VLM plus a 300M action expert, a prediction horizon of 50, bidirectional attention among action tokens, and ten Euler steps in its reported inference setup. These are paper-specific choices, not universal requirements or latency guarantees. The verified paper revision is January 8, 2026. [π₀ paper, v4](https://arxiv.org/abs/2410.24164v4).

Think of the computation as three interfaces: encode image/language context, combine context with state/noisy-action/time inputs, and project the resulting action features into a velocity tensor of shape \(B\times H\times d\). A token can be a continuous feature slot; it does not have to be an integer from a text vocabulary. An action expert is not necessarily sparsely routed by a learned mixture-of-experts gate.

Context caching depends on attention structure. If context tokens cannot attend to changing action tokens, their keys and values can be reused through flow integration. If an architecture allows that dependence, caching the same prefix changes the computation. Verify the actual mask and cache policy in the pinned implementation. Do not equate a diagram with a serialized checkpoint contract.

The official openpi repository provides code and model-specific training/inference guidance, but its default branch evolves. Record commit, configuration name, checkpoint revision, normalizer, and preprocessing before reproducing an example. A paper about a later model does not establish that its weights are released in that repository. [Official openpi](https://github.com/Physical-Intelligence/openpi).

## Worked example: direction and covariance

For scalar \(A=2\), \(Z=-1\), and \(\tau=0.25\), the interpolation is \(X_\tau=-0.25\), and target velocity is 3. An oracle constant field takes an Euler step of size 0.25 to \(X_{0.5}=0.5\). Four steps from \(-1\) end at 2 exactly because this teaching field is constant. Real learned fields vary with state and time, so their numerical errors do not vanish this way.

Given fixed \(A\), the random interpolation has mean \(\tau A\) and covariance \((1-\tau)^2 I\). The square matters: a coefficient multiplying Gaussian noise is a standard-deviation multiplier. Reconstructing this relationship from the sampling equation is safer than copying ambiguous covariance notation.

Also separate physical time from flow time. Fifty action slots may describe one second of control at 50 Hz, while ten flow steps are computational refinements of that same proposed chunk. A denoising step is not one robot timestep.

## Build and inspect

**Exercise / pseudocode.** Use the synthetic conditional distribution from Lecture 11 so data and evaluation are held fixed.

1. Implement interpolation and target construction with a declared time direction. Verify endpoints and the empirical conditional covariance on a fixed target.
2. Fit a small field using uniform \(\tau\) sampling. Compare it to an explicitly documented nonuniform distribution as a separate experiment; changing this distribution changes the weighting of training regions.
3. Generate paired samples using Euler with 2, 4, 8, and 16 evaluations. Add one higher-order solver at matched network-evaluation cost. Report distribution error, invalid actions, and measured latency.
4. Make a mock context encoder whose output is independent of the flow state. Compare cached versus recomputed output with identical starting noise. Then modify the mock architecture to depend on flow state and show why the old cache is invalid.
5. Inspect, without necessarily downloading weights, a pinned openpi configuration and attention-mask construction. Write down input image layout, state dimension, action dimension, horizon, padding convention, normalization, and time direction.
6. Inject reversed integration direction and wrong action scale. Show which algebraic or distribution checks catch each error.

Stop toy training after three seeds or two hours. Do not launch full VLA training until the tensor contract, data audit, and resource estimate from later lectures are complete. **Acceptance criteria:** endpoint and covariance checks pass; cached output parity is tested only where structurally valid; speed comparisons use equal conditioning and disclose evaluation count; the implementation map distinguishes the original paper from the inspected code revision.

## Failure clinic

- **Actions become noisier during integration:** inspect the time direction and velocity sign.
- **Reducing steps changes gripper behavior disproportionately:** examine discrete-versus-continuous decoding and per-coordinate scales.
- **Caching is fast but action parity fails:** inspect action-to-context attention dependence and preprocessing nondeterminism.
- **More solver steps do not help:** model error or distribution mismatch can dominate numerical integration error.

## Self-check and answer guidance

1. **Why can flow matching train without integration?** A sampled interpolation supplies both a training point and its analytic conditional velocity.
2. **Does a straight conditional path guarantee a straight sampled path?** No; the network approximates a marginal conditional field over many possible pairs.
3. **Are flow iterations physical control ticks?** No. They refine one action proposal before execution.
4. **What proves compatibility with π₀ weights?** Matching architecture, configuration, tensor semantics, preprocessing, and weight revision; sharing the phrase “flow matching” is insufficient.

[Previous: diffusion policies](Lecture-11.md) · [Course index](README.md) · [Next: robot data collection](Lecture-13.md)
