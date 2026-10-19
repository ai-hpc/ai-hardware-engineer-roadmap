# Lecture 08 — Action representations, quantization, and FAST

Choose an action encoding by balancing physical precision, temporal structure, learning difficulty, and inference cost.

**Prerequisites:** [Lecture 07](Lecture-07.md), normalization, probability, elementary signal processing. **Effort:** 30 minutes setup, two hours lab, 90 minutes analysis. **Hardware:** CPU laptop; no model training required. **Stack:** policy outputs, numerical formats, decoding runtime, control. **Roles:** VLA engineer, inference engineer, robot learning researcher.

[Course](README.md) · [Previous](Lecture-07.md) · [Next](Lecture-09.md)

## Representation changes the learning problem

An action encoder maps a physical command or chunk into a target that a neural network can predict. Its decoder maps predictions back into physical commands. These transformations determine what errors are possible and how expensive generation becomes.

The misconception is that every discrete action representation is ordinary text tokenization. A token can denote a scalar bin, a learned trajectory code, or part of a compressed coefficient sequence. Its meaning depends on a particular decoder and associated metadata.

Also separate action quantization from weight quantization. The first changes the policy's output representation; the second changes the numerical representation used to execute the network. An NVFP4 weight checkpoint does not imply four-bit action commands, and a 256-bin action vocabulary says nothing about the weight format.

## Four output families

A continuous regression head predicts real-valued vectors. Squared error favors the conditional mean, which can be undesirable for multimodal behaviors: averaging “go left around the obstacle” and “go right” may point into the obstacle. Mixture distributions or generative action heads address this differently.

Scalar discretization maps each action dimension to a bin. It permits categorical likelihood training but introduces precision limits and may require many sequential tokens. A learned latent codebook can represent larger patterns, with a decoder reconstructing actions; reconstruction quality and codebook coverage become additional dependencies.

Diffusion and flow-based heads produce continuous action samples through iterative generation. Their compute depends on the action expert and sampling procedure rather than simply the number of text-like tokens. Later lectures develop these objectives; here the important comparison is the full physical action and the measured time to obtain it.

## Worked example: uniform scalar bins

For \(B\) equal-width bins over \([a_{\min},a_{\max}]\), let

$$
\Delta=\frac{a_{\max}-a_{\min}}{B},\qquad
k=\operatorname{clip}\left(\left\lfloor\frac{a-a_{\min}}{\Delta}\right\rfloor,0,B-1\right),
\qquad \hat a=a_{\min}+(k+1/2)\Delta.
$$

Inside the representable range, midpoint reconstruction has worst-case error \(\Delta/2\). Endpoint conventions matter: the maximum input falls into the last bin after clipping. Out-of-range clipping error has no such half-bin bound.

For a displacement range of \([-0.05,0.05]\) m and 256 bins, the width is 0.390625 mm and the half-bin error is 0.1953125 mm. This is an encoding bound, not an end-effector accuracy guarantee. Calibration, inference error, controller tracking, and compliance can dominate it.

At seven dimensions and a 20-step chunk, independent scalar encoding produces 140 tokens. If a hypothetical autoregressive decoder spends 2 ms per generated token, token generation alone takes 280 ms. A compressed representation with 25 tokens would require 50 ms under the same deliberately simplified assumption. Prefill, output decoding, synchronization, and different token distributions can change measured results.

## Why time-series compression can help

Smooth motion contains redundancy. A constant sequence of 20 identical velocity targets does not contain 20 independent decisions. Predicting every scalar at every step creates a long target sequence even when the underlying trajectory has few degrees of freedom.

An orthonormal discrete cosine transform rewrites a length-\(H\) signal in a cosine basis:

$$
c_k=\alpha_k\sum_{n=0}^{H-1}a_n
\cos\left[\frac{\pi}{H}(n+1/2)k\right],\qquad
\alpha_0=H^{-1/2},\quad\alpha_{k>0}=\sqrt{2/H}.
$$

For the four-sample sequence \([1,1,1,1]\), the coefficients are \([2,0,0,0]\), up to numerical roundoff. The transform alone is invertible and does not reduce the number of stored scalars. Compression requires an encoding that exploits coefficient structure, often after quantization. Choose matching transform and inverse conventions; library defaults may use a different normalization. [SciPy DCT reference](https://docs.scipy.org/doc/scipy/reference/generated/scipy.fft.dct.html)

High-frequency coefficients can be small for smooth trajectories but important around abrupt contact or gripper transitions. A codec with low global mean squared error can still move a discrete grasp event one step too early. Report event timing and maximum error alongside average reconstruction error.

## FAST and its evidence boundary

FAST, introduced in January 2025, normalizes action chunks, transforms each dimension into frequency space, quantizes coefficients, and applies byte-pair encoding to the serialized coefficients. Its ordering places low-frequency information early. FAST+ is the released broadly trained tokenizer; a tokenizer is distinct from a complete VLA policy. The paper studies training and task outcomes under its evaluated setups. [FAST paper](https://arxiv.org/html/2501.09747v1)

The official tokenizer release provides an implementation and model artifacts. Record the tokenizer revision, normalization statistics, chunk shape, and vocabulary when reproducing it. This lecture's codec exercise omits byte-pair encoding and is therefore a DCT quantization exercise, not a FAST implementation or reproduction. [Official FAST tokenizer](https://huggingface.co/physical-intelligence/fast)

## Metadata and failure modes

Decoding needs the action dimension count, horizon, frequency, normalization statistics, and token vocabulary. Swapping two action dimensions before encoding and failing to swap them back changes the command while preserving reconstruction error in the wrong coordinate system.

Variable-length output also needs valid termination. Truncated coefficient sequences must not silently decode to a shorter uncontrolled motion. Specify whether the system rejects the chunk, requests regeneration, or executes a separately validated prefix. A decoder should reject nonfinite values and invalid shapes before producing a controller message.

Compression must preserve the intended time axis. Resampling a 50 Hz trajectory to 10 Hz can discard short contact events; replaying 50 Hz deltas at 10 Hz changes trajectory speed. Report both physical duration and sample count.

## Lab: compare precision and event preservation

**Exercise / pseudocode; no upstream tokenizer or policy is executed.** Generate three deterministic 20-step, seven-dimensional trajectories: a constant displacement, a smooth ramp, and a trajectory with an abrupt gripper transition at step 12. Use explicitly stated physical ranges per dimension; do not mix radians and meters in an unlabeled aggregate score.

Implement midpoint scalar quantization with 16, 64, and 256 bins. Implement the orthonormal DCT formula and its transpose inverse directly, then quantize coefficients using \(\hat c=\operatorname{round}(\gamma c)/\gamma\) for \(\gamma\in\{2,10,50\}\). Verify the constant-sequence coefficients before evaluating other signals.

Measure per-dimension maximum error, root mean squared error, gripper transition timing, nonzero coefficient count, and encode/decode runtime. Nonzero coefficient count is only a sparsity proxy: do not label it “FAST token count” or assume every coefficient costs one token.

Negative fixtures must include an out-of-range action, a wrong inverse normalization scale, reversed time order, and a missing final sample. Define which errors can be detected from shape/range checks and which require provenance or a known reference trajectory.

For guided variation, double the sampling frequency while preserving physical duration **and action meaning**. Absolute positions can be interpolated. Velocity values can be resampled and integrated at the new timestep. Per-step displacements instead need duration-aware conversion: 20 intervals of 1 mm at 50 Hz describe 20 mm in 0.4 s; at 100 Hz use 40 intervals of 0.5 mm. Simply duplicating the 1 mm values would double the commanded displacement. Preserve binary gripper events as events, not intermediate opening values. Compare redundancy and event representation after checking total physical displacement. The supplied [episode-conversion lab](Dataset-and-Execution-Lab.md) demonstrates this distinction with two source cadences. For independent transfer, design separate treatment for a binary gripper channel and continuous arm channels, then explain the tradeoff in decoder complexity and joint action modeling.

Stop after the nine codec configurations and four negative fixtures. Submit code, raw reconstruction data, a rate-distortion-style plot with accurately labeled proxy axes, and a decoder manifest. This lab cannot demonstrate policy learning speed, real FAST compression ratio, or physical task success.

## Self-check and exit criteria

1. **Does a DCT alone compress an action chunk?** No. It changes basis; storage reduction needs quantization, sparsity coding, or another encoding mechanism.
2. **Why can low reconstruction MSE be insufficient?** A brief but decisive contact or gripper event can be corrupted while most samples remain accurate.
3. **Does fewer output tokens guarantee a proportional speedup?** No. Prefill, decoder implementation, action reconstruction, and synchronization contribute to latency.

Exit when your encoding comparison states physical error bounds, metadata dependencies, timing effects, and the limits of the evidence collected.

[Previous](Lecture-07.md) · [Next: continuous generative action policies](Lecture-09.md)
