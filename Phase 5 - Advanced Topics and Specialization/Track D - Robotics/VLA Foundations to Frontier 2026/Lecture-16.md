# Lecture 16 — Fine-Tuning, Adapters, Normalization, and Calibration

Adapt a pretrained policy to a new task while keeping the checkpoint, data, numerical representation, and robot interface consistent.

**Prerequisites:** Lectures 12–15; optimizers and trainable parameter groups. **Effort:** setup 1 hour, reduced lab 4 hours, evaluation 2 hours. **Hardware:** CPU for adapter mathematics and data-contract tests; GPU capacity for a full VLA depends on checkpoint and recipe. **Stack/roles:** model adaptation, GPU memory, and deployment validation; robot learning and model optimization engineers. **Artifact:** an adaptation package with baseline comparison, resource measurements, and an executable normalization contract.

## Begin with compatibility, then choose a tuning method

Before changing weights, evaluate the base policy with the intended camera preprocessing, instruction template, state layout, action decoder, and normalizer. Save raw and processed examples. A swapped camera or wrong unnormalization key can make a capable checkpoint appear unable to move correctly.

Fine-tuning options include a frozen backbone with a trained action head, low-rank adapters in selected modules, partial unfreezing, and full tuning. They differ in representational freedom, optimizer memory, activation memory, checkpoint size, and risk of forgetting. There is no universally best order; the correct comparison controls data budget, evaluation episodes, and training compute.

OpenVLA's primary materials document efficient adaptation experiments. OpenVLA-OFT separately studies changes to the fine-tuning design, including action generation choices. Treat a change from autoregressive discrete actions to a continuous or parallel action head as an architectural/objective change, not merely a different optimizer setting. [OpenVLA project](https://openvla.github.io/), [OpenVLA-OFT](https://openvla-oft.github.io/).

## What LoRA saves and what it does not

For a frozen matrix \(W\in\mathbb R^{d_{out}\times d_{in}}\), LoRA learns

$$
W'=W+\frac{\alpha}{r}BA,
\quad A\in\mathbb R^{r\times d_{in}},\quad B\in\mathbb R^{d_{out}\times r}.
$$

The trainable parameter count is \(r(d_{in}+d_{out})\), compared with \(d_{in}d_{out}\) for a full update. For an illustrative 4096×4096 matrix with rank 16, this is 131,072 rather than 16,777,216 parameters, a 128-fold reduction for that matrix's trainable update. This does not imply 128-fold lower total GPU memory: base weights and intermediate activations still occupy memory. Gradients through frozen operations may still be needed to train upstream adapters. [LoRA, 2021](https://arxiv.org/abs/2106.09685).

Quantized-base adaptation adds another distinction: storage precision, compute precision, optimizer precision, and export precision can differ. QLoRA is a specific method with a particular quantization and adaptation recipe. NF4 used in that work is not NVIDIA NVFP4. A quantized text-model recipe does not establish VLA action fidelity or hardware-kernel support. [QLoRA, 2023](https://arxiv.org/abs/2305.14314).

Record target modules, rank, scale, initialization, dropout, and merge procedure. After merging adapters, compare outputs with the unmerged model on held-out inputs using identical sampling noise and controlled evaluation mode, with dropout disabled and normalization state fixed. Quantization or casting during export can invalidate exact equality; define an appropriate action-space tolerance and report it.

## Normalization is part of the model

For z-score normalization, \(x'=(x-\mu)/\max(\sigma,\epsilon)\). Its inverse is \(x=x'\max(\sigma,\epsilon)+\mu\). For quantile scaling, an example convention is

$$
x'=2\frac{x-q_{lo}}{q_{hi}-q_{lo}}-1.
$$

Clipping \(x'\) makes this transformation non-invertible outside the quantile range. Specify which dimensions are normalized, which are categorical, how constant dimensions are treated, and whether padded dimensions are excluded. A normalizer computed on one embodiment cannot be assumed valid for another.

Changing normalization while keeping a pretrained head unchanged changes the physical meaning of its outputs. A policy's normalized output of 0.5 can become a different motor command after a dataset update. Save normalization statistics with the checkpoint, checksum them, and fail clearly if they are absent or mismatched.

Calibration has several meanings here. Sensor calibration establishes geometric and timing relationships. Numerical calibration selects scales for a quantized model. Probabilistic calibration asks whether predicted uncertainty matches observed outcomes. None substitutes for the others. A confident action distribution is not a probability that a task is safe or successful.

## Worked example: a quiet scale failure

Suppose a joint-delta output is 0.4 in normalized units. With training scale 0.05 radians and zero mean, that represents 0.02 radians. An accidentally loaded scale of 0.5 radians produces 0.2 radians, ten times larger. The normalized validation loss can remain unchanged because the bug occurs only during deployment decoding.

The [runnable action-parity bridge](../VLA%20Optimization%20and%20Action-Parity%20Harness/CPU-Action-Parity-Lab.md) trains the same small policy used in Phase 3 and compares a correct folded normalizer with an omitted normalizer. It checks both physical actions and paired rollouts; its dequantized-weight variation is a numerical experiment, not a native INT8 speed measurement.

A round-trip test catches algebraic normalizer errors, but it may not catch using the wrong valid normalizer. You also need metadata identity checks and a known physical-action fixture. This is why preprocessing and action transforms must be versioned together with model weights.

## Run a disciplined adaptation experiment

**Exercise / pseudocode.** Use the [shared CPU lab](Lab-Guide.md) for contract checks, then a small trainable policy or a pinned upstream VLA recipe. Upstream installation and training commands must be labeled **upstream documented, not tested here** until actually executed.

1. Freeze a grouped data split and an evaluation manifest. Record the base checkpoint, library revision, camera ordering, action contract, normalizer hash, and available memory.
2. Verify identity transforms, normalization round trips, fixed-noise inference reproducibility, and one known physical action. Inject a wrong normalizer and require rejection.
3. Compare head-only adaptation with one adapter setting and full tuning only if resources permit. Keep target examples and the stopping budget identical; state when compute cannot be matched.
4. Track train/validation loss, task success, instruction sensitivity, per-dimension action error, peak memory, and wall-clock time. Include a retained source-task evaluation to detect forgetting.
5. Export the chosen model. Test adapter merge parity and reloading in a fresh process. Save optimizer state separately from the deployable policy package.
6. Evaluate a shifted camera or object condition without retraining. Explain whether failure suggests perception, action-interface mismatch, or insufficient data coverage.

Stop after a preregistered step count or three validation checks without improvement; retain the best validation-selected checkpoint. Do not select using the final test set. **Acceptance criteria:** all transform fixtures pass; adapter and full-model comparisons disclose parameter and memory counts; the exported package reproduces held-out outputs within declared tolerances; task and source-retention results accompany loss curves. The toy exercise does not establish that a large VLA fits the learner's GPU.

## Failure clinic and self-check

**Failure clinic:** exploding actions after export suggest normalizer or merge mismatch; unchanged training loss suggests frozen intended parameters or detached gradients; source-task collapse suggests forgetting; unexpectedly high adapter memory suggests activation or frozen-backbone storage costs.

1. **Why can LoRA still require substantial memory?** It reduces trainable updates and optimizer states, not all weights, activations, or computation.
2. **Does a smaller training loss establish a better adaptation?** No; validate closed-loop behavior, generalization, and retained capabilities.
3. **Can you update normalization independently of weights?** Only with an explicit compatibility strategy and validation of physical actions.
4. **What must be shipped?** Weights/adapters, base revision, configuration, preprocessing, action transforms, normalization, and evaluation evidence.

**Go deeper:** [Egocentric Data and VLA Fine-Tuning, Lectures 09–12](../Egocentric%20Data%20and%20VLA%20Fine-Tuning/README.md) compare LeRobot, openpi, Isaac-GR00T and OpenVLA-OFT fine-tuning on 8, 24 and 80 GB, and evaluate the result with a paired test.

[Previous: pretraining](Lecture-15.md) · [Course index](README.md) · [Next: RL post-training](Lecture-17.md)
