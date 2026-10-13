# Lecture 11 — Diffusion Action Policies from Noise to Control

Implement conditional action generation and explain how denoising choices affect both the action distribution and the robot's timing budget.

**Prerequisites:** Lectures 09–10; Gaussian distributions and neural network training. **Effort:** setup 45 minutes, lab 4 hours, analysis 90 minutes. **Hardware:** CPU for one-dimensional distributions; a GPU is useful for image-conditioned training. **Stack/roles:** model algorithms, accelerator execution, and robot control; robot learning and GPU inference engineers. **Artifact:** a conditional action sampler with correctness checks and a latency-versus-quality curve.

## Why generate an action distribution?

An observation can admit several successful motions. A sampler can choose one coherent trajectory rather than regress toward an average of incompatible trajectories. Diffusion Policy applied conditional denoising to action sequences with visual conditioning and receding-horizon execution. Its paper provides concrete CNN and transformer variants; those architectural alternatives should not be conflated with the underlying denoising objective. [Diffusion Policy project and paper](https://diffusion-policy.cs.columbia.edu/).

Diffusion is a distribution model, not a collision checker. If the demonstrations contain unsafe motions, or an unseen object changes what is feasible, sampling does not automatically resolve either problem. Treat the generated chunk as a proposed control sequence subject to the same action validation and execution contract as earlier policies.

## Training: a known corruption process

Let \(A_0\) denote a normalized clean action chunk. Define \(\beta_k\in(0,1)\), \(\alpha_k=1-\beta_k\), and \(\bar\alpha_k=\prod_{i=1}^k\alpha_i\). A standard discrete forward process permits direct sampling at any noise level:

$$
A_k=\sqrt{\bar\alpha_k}A_0+\sqrt{1-\bar\alpha_k}\,\epsilon,
\qquad \epsilon\sim\mathcal N(0,I).
$$

An epsilon-prediction network minimizes a masked version of

$$
\mathcal L_\epsilon=\mathbb E_{A_0,c,k,\epsilon}
[\|\epsilon-\epsilon_\theta(A_k,k,c)\|_2^2],
$$

where \(c\) contains observations, language, and proprioception. This foundational formulation comes from DDPM; an action-policy implementation must additionally define sequence masks and conditioning. [Ho, Jain, and Abbeel, 2020](https://arxiv.org/abs/2006.11239).

The model needs the noise level as input because the same numeric vector can mean different things at different corruption strengths. The action normalizer is equally important: a unit Gaussian is meaningful only relative to a known action scale. Apply the loss only to valid coordinates and timesteps. Do not let padded joints become easy dimensions that dominate the loss.

Noise schedules determine the signal-to-noise ratio \(\operatorname{SNR}_k=\bar\alpha_k/(1-\bar\alpha_k)\). The training distribution over \(k\), prediction target, and loss weighting jointly decide which corruption levels receive learning pressure. Changing a schedule at inference without checking the training convention can invalidate the sampler.

## Sampling: repeated work has a price

One DDPM reverse-step mean is

$$
\mu_\theta(A_k,k,c)=\frac{1}{\sqrt{\alpha_k}}
\left(A_k-\frac{\beta_k}{\sqrt{1-\bar\alpha_k}}\epsilon_\theta(A_k,k,c)\right).
$$

Then sample \(A_{k-1}=\mu_\theta+\sigma_k z\) with the variance convention specified by the implementation; the final step normally omits added noise. DDIM uses a different inference construction that can use fewer timesteps and can be deterministic given its starting noise. Merely skipping iterations in a DDPM loop does not implement DDIM. [Song, Meng, and Ermon, 2020](https://arxiv.org/abs/2010.02502).

A useful cost model is \(T=T_{encode}+N T_{denoise}+T_{decode}\). It is an approximation: launch overhead, attention caching, and kernel shapes change actual timing. Cache conditioning only if its computation does not depend on the changing noisy actions. When reducing \(N\), measure both per-query latency and closed-loop success. A fast but mode-collapsed sampler is not an inference improvement.

## Worked example and target checks

Take an illustrative scalar target \(A_0=0.6\), \(\bar\alpha_k=0.25\), and sampled noise \(\epsilon=-1\). Then \(A_k=0.5(0.6)+\sqrt{0.75}(-1)\approx-0.566\). The epsilon target is \(-1\), not \(0.6\). Recovering the clean action from perfect noise prediction gives

$$
\hat A_0=\frac{A_k-\sqrt{1-\bar\alpha_k}\hat\epsilon}{\sqrt{\bar\alpha_k}}=0.6.
$$

As \(\bar\alpha_k\) approaches zero, this expression amplifies epsilon-prediction errors. This is one reason training objectives, numerical precision, and timestep selection matter. It is also a good unit-level check: a sign error or wrong square root causes failure even with an oracle denoiser.

## Build a distribution experiment

**Exercise / pseudocode.** Start with synthetic conditional action data: instruction `left` selects a Gaussian around \(-1\); `right` selects one around \(+1\); `either` uses both. Use a bounded action representation and disclose the distribution parameters as synthetic fixture values.

1. Implement the forward corruption equation and verify empirical mean and variance using many samples. Check the scalar reconstruction above with an oracle epsilon.
2. Train a small conditional MLP using timestep embeddings. Reserve fresh random seeds for validation. Record loss by noise level, not just aggregate loss.
3. Sample each instruction and compare histograms, mode proportions, mean error, and fraction outside an explicitly chosen valid range. An average near zero is correct for the `either` distribution's mean but does not establish correct samples.
4. Implement one documented sampler consistently, then compare a supported faster sampler. Keep starting noise and evaluation conditions paired where possible.
5. Measure warm and cold latency, network evaluations per sample, and maximum memory. For GPU timing, synchronize around the measured interval and disclose batching.
6. Replace the instruction with a permuted label and deliberately swap epsilon prediction with clean-action prediction. Both negative cases should be identifiable from distribution or reconstruction checks.

Stop after three seeds or two hours of toy training, whichever occurs first. If the tiny task fails, inspect conventions before expanding the model. The reduced lab validates sampling mathematics; it does not establish manipulation capability or reproduce Diffusion Policy's published results.

**Acceptance criteria:** oracle reconstruction passes within a declared numeric tolerance; valid target masks are used; conditional modes respond to language; sampler settings and schedules are saved; faster configurations include quality measurements and failure examples.

## Failure clinic and self-check

**Symptoms:** oscillating chunks suggest sampling or execution inconsistency; identical actions for all prompts suggest ignored conditioning; NaNs at the final step suggest invalid schedule arithmetic; low training loss with wrong samples suggests target or normalization mismatch. Inspect one tensor path before adjusting learning rates.

1. **Why not use MSE directly on multimodal demonstrations?** Its conditional mean may lie between valid modes; diffusion can represent the distribution if training succeeds.
2. **Is random sampling itself a safety mechanism?** No. Feasibility still requires appropriate data, constraints, observation coverage, and execution checks.
3. **Can inference timesteps be changed freely?** They must be supported by the chosen sampler and training convention, then validated empirically.
4. **Why profile the encoder separately?** Denoising-step reductions stop helping once fixed encoding cost dominates.

[Previous: action chunking](Lecture-10.md) · [Course index](README.md) · [Next: flow matching and π₀](Lecture-12.md)
