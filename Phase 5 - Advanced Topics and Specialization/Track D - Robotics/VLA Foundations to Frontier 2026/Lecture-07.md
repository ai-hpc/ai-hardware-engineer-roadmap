# Lecture 07 — Visual and language encoders, fusion, and attention masks

Design a multimodal observation path that preserves useful information while fitting the robot's compute and latency budget.

**Prerequisites:** [Lecture 06](Lecture-06.md), matrix multiplication, gradients, attention. **Effort:** 45 minutes setup, two hours lab, one hour analysis. **Hardware:** CPU for shape and mask exercises; optional GPU profiling is separate. **Stack:** model architecture, memory, kernels, sensor integration. **Roles:** multimodal engineer, inference engineer, model optimization engineer.

[Course](README.md) · [Previous](Lecture-06.md) · [Next](Lecture-08.md)

## An encoder is an information bottleneck

The camera produces pixels, while a transformer consumes vectors. The encoder determines which distinctions are easy for the policy to recover. A visual representation may distinguish object categories well yet lose the small gap between a gripper finger and an object. A language representation may capture topic similarity while underemphasizing a negation or spatial relation.

The misconception is that a high score on general visual understanding guarantees a good robot observation representation. A controller needs task-relevant precision, temporal consistency, and an appropriate inference cost. All three must be measured after the representation is connected to an action head.

CLIP supplies image-text-aligned features; SigLIP studies a pairwise sigmoid objective for image-language pretraining; DINOv2 studies visual representations without language supervision. These different objectives motivate combining complementary features, but no objective guarantees physical grounding by itself. [CLIP](https://arxiv.org/abs/2103.00020), [SigLIP](https://arxiv.org/abs/2303.15343), [DINOv2](https://arxiv.org/abs/2304.07193)

Prismatic VLMs systematically investigates design choices in visually conditioned language models. Read such studies as evidence about the evaluated settings, then separately test the robot-specific consequence of a visual feature or fusion choice. [Prismatic paper](https://arxiv.org/abs/2402.07865)

## From images to tokens

For an image with height \(H_i\), width \(W_i\), and patch width \(P\), a simple patch grid contains \(N_v=(H_i/P)(W_i/P)\) tokens when both dimensions divide evenly. Each patch becomes a vector of width \(d_v\). An adapter projects visual features into the width \(d\) expected by the multimodal backbone.

A linear adapter computes \(z=Wf+b\); an MLP adapter adds nonlinear mixing. Concatenating two encoders' channel features increases feature width while preserving token count if their grids align. Concatenating their token sequences increases sequence length. Those choices have different memory and attention costs, even when both are described as “fusing encoders.”

Different resize and crop rules can prevent patch alignment. A feature at index 30 in one encoder need not describe the same image region in another. Store preprocessing alongside the adapter and verify it with an image containing a known marker grid.

Camera identity and time identity are also inputs. If tokens from two cameras are concatenated without identifying their source, a model can confuse wrist and external viewpoints. If historical frames are reordered, a trajectory can appear to run backward. Positional, camera, and temporal embeddings should express the intended structure.

## Three fusion strategies

Early fusion lets visual, language, and state tokens interact inside a shared transformer. It is flexible, but every added token can affect the backbone workload. Cross-attention lets one stream query another, potentially retaining a smaller action-processing stream. Late fusion independently processes modalities before combining compact features; it can be efficient but may make fine-grained word-to-pixel relations harder to represent.

Attention computes

$$
\operatorname{Attention}(Q,K,V)=
\operatorname{softmax}\left(\frac{QK^\top}{\sqrt{d_k}}+M\right)V,
$$

where \(M\) encodes which pairs may interact. The transformer formulation is established in [Attention Is All You Need](https://arxiv.org/abs/1706.03762). For robotics, the mask is part of the causal contract: a predicted action must not attend to future observations unavailable at execution.

An all-masked query needs an explicit rule: ordinary softmax of a row of negative infinities is undefined. Reject the invalid query or use a documented zero-output path, and exclude nonexistent targets from the loss. Test the chosen behavior rather than relying on a kernel-specific accident.

A padding mask and a causal mask solve different problems. Padding marks nonexistent entries. Causality prevents access to future information. A history frame at the start of an episode may be absent; duplicating the current image is a modeling decision, not automatically equivalent to masking a missing frame.

## Worked example: visual context grows quickly

Use two cameras, three history frames, 224-by-224 images, and 14-by-14 patches. Each image produces \(16^2=256\) tokens, giving \(2\times3\times256=1536\) visual tokens. Add 32 language tokens and eight state/action context tokens for \(N=1576\).

At width 2048 with two-byte elements, one hidden-state tensor for this sequence holds

$$
1576\times2048\times2=6{,}455{,}296\ \mathrm{bytes}
\approx6.16\ \mathrm{MiB}.
$$

That is one tensor, not total activation memory. A hypothetical 32-layer decoder with eight key/value heads of width 128 and two-byte cache elements would use

$$
2\times32\times1576\times8\times128\times2
=206{,}569{,}472\ \mathrm{bytes}\approx197\ \mathrm{MiB}
$$

for that context's K/V cache. The first factor of two accounts for keys and values. This formula assumes that architecture's decoder caches all those tokens; cross-attention, grouped-query settings, and recomputation choices can change the accounting.

Reducing each image to 64 tokens changes the context to 424 tokens. It also discards or compresses information. The correct experiment measures both resource use and task-relevant localization or action quality, with identical scene splits.

## Frozen encoders, gradients, and caching

Freezing an encoder avoids storing gradients for its parameters and can avoid retaining its internal training graph when no upstream trainable component needs that gradient. It does not eliminate its forward-pass compute or output storage. If a trainable image transform sits before it, gradients through the encoder may still be required.

Caching encoder features can help repeated offline training on fixed images. It becomes invalid when crops, augmentation, camera calibration, encoder weights, or model mode/state change. Freezing parameters does not disable dropout or freeze running normalization statistics. Use a controlled evaluation-mode encoder for deterministic feature caches; a cached stochastic training realization changes the training procedure. For online control, a cached old feature is an old observation. Reusing it must obey the same age limit as reusing the old image.

Language embeddings can often be reused while the instruction remains unchanged, but only if the architecture's conditioning path permits it. Caching a representation after joint visual-language attention would also cache the old visual context.

## Lab: audit tensors and expose leakage

**Exercise / pseudocode; no pretrained encoder execution is claimed.** Build a shape calculator for camera count, history, image size, patch size, hidden width, and cache parameters. Reject nondivisible image sizes unless an explicit padding rule is supplied. Reproduce both memory values above and distinguish bytes, MB, and MiB.

Construct a tiny attention example with four observation tokens, two padding positions, and three action tokens. Write the allowed attention matrix explicitly. At training time, use shifted action inputs so action \(a_j\) can see \(a_{<j}\) but never its own target or later targets. Add future observation tokens and verify that every action query masks them.

Use distinct marker values for each token. Perturb a masked future token by a large amount; the action output should remain unchanged within numerical tolerance. Perturb an allowed past token; at least one deliberately constructed query should change. This negative/positive pair is stronger than inspecting a triangular matrix visually.

For guided variation, swap image contents while keeping camera identifiers in their original slots: this mislabels the views. Then move each image together with its identifier. The latter preserves the image/identity pairing, but does not guarantee equal model outputs: absolute positions, masks, and a trained slot-order convention may still matter. Require parity only when the complete architecture and preprocessing contract make that reordering equivalent. For independent transfer, design a cache key that includes encoder revision, preprocessing revision, image identity, and augmentation seed for offline data. Show a collision that would occur if one field were omitted.

Collect tensor sizes, mask-violation count, masked perturbation error, and execution time for the tiny attention operation. Stop after the two context budgets and three mask fixtures. Submit the calculator, mask visualization, perturbation results, and an explicit list of assumptions.

The reduced lab validates accounting and information flow. It cannot measure real encoder quality, kernel latency, GPU memory fragmentation, or robot success. Those require the chosen architecture and hardware.

## Self-check and exit criteria

1. **Does doubling feature width equal doubling token count?** No. They alter projection, feed-forward, and attention costs differently.
2. **Why can an apparently excellent offline policy fail online?** Its training mask may expose future images or target actions that deployment cannot provide.
3. **When is feature caching invalid?** Whenever an input or computation affecting that feature changes, or when a cached online observation exceeds its freshness contract.

Exit when you can trace shapes, identities, visibility masks, and cache validity through the full observation path.

[Previous](Lecture-06.md) · [Next: action representations and FAST](Lecture-08.md)
