# Lecture 04 — RT-1, RT-2, and PaLM-E: three distinct architectural steps

Trace what changed when robot policies adopted transformers, multimodal language models, and action tokens.

**Prerequisites:** [Lecture 03](Lecture-03.md), transformer basics, cross-entropy. **Effort:** 45 minutes reading/setup, two hours lab, one hour analysis. **Hardware:** CPU for architecture and token-budget exercises; no claim of reproducing the original models. **Stack:** model architecture, training workload, accelerator memory, control runtime. **Roles:** VLA researcher, ML systems engineer, inference engineer.

[Course](README.md) · [Previous](Lecture-03.md) · [Next](Lecture-05.md)

## Why the distinctions matter

RT-1, RT-2, and PaLM-E are often grouped under “large models for robotics.” That grouping hides different outputs and different uses of pretraining. A useful architecture review starts with input tensors, output tensors, losses, data mixtures, and execution boundaries.

The misconception is that all three are interchangeable checkpoints for a robot arm. Their research contributions differ, and a published architecture does not imply that the complete training data, weights, robot platform, and evaluation environment are available for independent execution.

## RT-1: scaling robot-conditioned sequence prediction

RT-1 was submitted in December 2022. It used a transformer policy with visual and language conditioning, compacted visual tokens, and discretized robot actions. Its study emphasized learning across many demonstrated tasks and evaluating generalization as training scale and diversity changed. [RT-1 paper](https://arxiv.org/abs/2212.06817)

The project describes an image-processing path using an EfficientNet-based encoder, language conditioning, and TokenLearner before transformer processing. The architectural idea to retain is that compressing observations can make sequence modeling practical under robot control latency constraints. Inspect the published implementation before assuming an arbitrary transformer can replace those components unchanged. [RT-1 project](https://robotics-transformer1.github.io/)

This motivates a general engineering question: how much task-relevant information survives token compression? A smaller sequence reduces attention and cache costs, but may remove the tiny visual cue needed to distinguish a closed gripper around an object from an empty gripper.

## PaLM-E: sensory information inside an embodied language model

PaLM-E was submitted in March 2023. It incorporated visual and continuous state information into sequences processed by a pretrained language model and studied embodied reasoning, including manipulation planning. Its output role should be inspected separately from a direct action predictor: generating a grounded plan still depends on downstream execution machinery. [PaLM-E paper](https://arxiv.org/abs/2303.03378)

Its general lesson is that a model's context can contain learned sensory embeddings rather than only textual descriptions. This avoids forcing every observation through a potentially lossy caption, but does not automatically supply calibrated geometry, hard real-time behavior, or physical feasibility.

For example, a plan “pick the red block, then place it in the drawer” can be semantically appropriate while failing because the drawer is closed or the chosen skill cannot open it. Evaluate both plan correctness and executor success, and report which component caused failure.

## RT-2: making actions part of a vision-language output space

RT-2 was submitted in July 2023. It adapted vision-language models to robot control by representing actions as tokens and combining robot data with vision-language training. The paper reports transfer of web-derived semantic knowledge to the evaluated manipulation settings. That is evidence for a particular transfer mechanism and task distribution, not proof of unlimited motor skill acquisition from internet images. [RT-2 paper](https://arxiv.org/abs/2307.15818)

The project presents action-token examples and evaluations separating familiar skills from semantic generalization. A deployment review must still account for token decoding, action normalization, control timing, and hardware execution. [RT-2 project](https://robotics-transformer2.github.io/)

Web knowledge may help choose an appropriate object, while robot demonstrations teach how the embodiment interacts with it. Recognizing that a rock can serve as a tool does not by itself identify safe contact forces or a stable grasp on every rock.

## A common mathematical view

Let \(z_v=f_v(I)\) be visual tokens, \(z_l=f_l(l)\) language tokens, and \(z_s=f_s(s^{\mathrm{measured}})\) state tokens. An architecture can condition a transformer on some combination of these:

$$
h=F_\theta(z_v,z_l,z_s),\qquad y\sim p_\theta(y\mid h).
$$

The meaning of \(y\) changes everything. It may be a skill description, a discretized motor target, a continuous action head's hidden state, or an answer to a visual question. Identical transformer mathematics does not establish equivalent robotics behavior.

For an action-token sequence \(y_1,\ldots,y_M\), autoregressive training minimizes

$$
\mathcal L_{\mathrm{act}}=-\sum_{j=1}^{M}
\log p_\theta(y_j\mid z_v,z_l,z_s,y_{<j}).
$$

A mixed objective might use \(\mathcal L=\alpha\mathcal L_{\mathrm{act}}+\beta\mathcal L_{\mathrm{VL}}\). This equation is a teaching abstraction, not a claim that every model uses these exact weights. Dataset sampling, loss reduction, and token counts determine the effective balance. A nominal 50:50 example mixture can produce a very different token-level contribution if one task has much longer targets.

## Worked example: token reduction and memory are different problems

Suppose six images each produce 256 visual tokens. Add 32 language tokens and eight action tokens. The total sequence length is 1,576. If a hypothetical compressor reduces each image to 16 tokens, the total becomes 136.

For a dense attention score matrix, the element-count ratio is

$$
\frac{1576^2}{136^2}\approx134.3.
$$

This is an analytical ratio for one intermediate matrix, not a 134-fold end-to-end speedup. Image encoding, projections, feed-forward layers, kernel launch overhead, and memory transfers remain. Fused attention implementations may avoid materializing that full matrix.

Separately, a hypothetical seven-billion-parameter model storing every weight in two bytes needs 14 billion bytes, about 13.0 GiB, for weights alone. Visual token compression does not shrink those weights. Activations, cache, temporary buffers, and allocator overhead add further memory. These arithmetic examples are independent of the original papers' measured hardware results.

## Lab: architecture cards and a falsifiable comparison

**Exercise / pseudocode; original-model training and inference are not run here.** Create one architecture card per model with fields for observation modalities, output semantics, pretraining source, robot-data role, tokenization, policy/executor boundary, and publicly inspectable artifacts. Link every historical claim to the source and mark unknown fields as unknown.

Build a CPU spreadsheet or Python calculator with input variables for cameras, history frames, tokens per image, language length, action length, parameter count, and bytes per weight. Compute sequence length, dense score-matrix elements, and weight-only storage. Keep units visible and label all results analytical.

Use the worked example as a positive fixture. Add a negative case that accidentally multiplies camera count twice; your checks should reveal the resulting sequence inflation. Add a missing action-normalization field to an architecture card and require the review to block a claim of deployability.

For guided variation, compare one camera with six history frames against three cameras with two history frames each. They have equal image counts but different information about occlusion and motion. The compute calculator cannot decide which observation structure is better.

For independent transfer, design a comparison that tests semantic generalization without changing motor difficulty: reuse the same reachable object locations, vary only which object the instruction selects, and separate held-out object identities from held-out layouts. Define the task success criterion and number of trials before collecting results.

Stop after the three cards and a complete evaluation proposal. Submit the calculator, provenance table, and two-page comparison. Do not fill success-rate columns with plausible numbers; use “not measured” until a real implementation is evaluated.

## Self-check and exit criteria

1. **Why is PaLM-E's role different from direct action token prediction?** Explain its embodied reasoning/planning use and identify the downstream executor needed for a physical result.
2. **Does action-token syntax make a VLM a working robot policy?** No. The learned action distribution, normalization, embodiment, and execution interface must also match.
3. **What does visual token compression accelerate?** Primarily the downstream sequence workload; it may leave image encoding and weight storage unchanged.

Exit when your comparison describes concrete data flow and reproducibility boundaries rather than ranking model names by parameter count.

[Previous](Lecture-03.md) · [Next: open generalist policies](Lecture-05.md)
