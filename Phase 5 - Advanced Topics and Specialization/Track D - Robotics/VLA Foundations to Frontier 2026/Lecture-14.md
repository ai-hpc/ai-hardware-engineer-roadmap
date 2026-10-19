# Lecture 14 — Dataset Schemas, Transformations, Splits, and Mixtures

Turn heterogeneous recordings into training examples while preserving action meaning, episode boundaries, and a credible evaluation split.

**Prerequisites:** Lecture 13; arrays, file formats, and basic data loaders. **Effort:** setup 45 minutes, lab 4 hours, analysis 2 hours. **Hardware:** CPU and local storage; optional GPU for loader throughput measurements. **Stack/roles:** storage, data pipelines, training runtime; robotics data and ML infrastructure engineers. **Artifact:** a versioned conversion pipeline, dataset manifest, and adversarial data audit.

## Storage compatibility is not semantic compatibility

Two datasets can contain an `action` tensor of shape seven and still describe incompatible commands: absolute end-effector position versus translation delta, base frame versus tool frame, radians versus degrees, or gripper-open versus gripper-close probability. A model can consume both without crashing and learn a contradiction.

Define a canonical schema with explicit field names and metadata: episode/session/environment/robot IDs, timestamp, image references, proprioception, instruction, action representation, unit, frame, controller period, validity masks, termination type, and provenance. Conversion functions should expose both their source contract and target contract. Keep a copy of the source schema revision.

RLDS represents episodes as sequences of steps, including boundary markers. Its `is_last` and `is_terminal` meanings matter: a final observation is not automatically a valid action target, and truncation differs from a terminal state. Inspect the documented reward/action alignment when converting from another logger. [Official RLDS schema](https://github.com/google-research/rlds).

LeRobotDataset v3 documents a different storage organization using tabular data, video files, and metadata that can span multiple episodes per shard. File boundaries therefore do not define episode boundaries. Its temporal access features depend on dataset metadata and timestamp semantics. Pin the exact library and dataset revisions used. [Official LeRobotDataset v3 documentation](https://huggingface.co/docs/lerobot/en/lerobot-dataset-v3).

## Transform the meaning, not just the shape

For a translation delta recorded in frame A, a frame change to B uses the appropriate rotation: \(\Delta p_B=R_{BA}\Delta p_A\). Absolute positions also require translation. Angular velocity uses vector transformation, while finite orientation deltas require composition on the correct side. A general six-dimensional twist under a change of reference point needs the SE(3) adjoint, including translation-induced coupling. Do not substitute a six-number concatenation into an arbitrary rotation operation.

Resampling also depends on action semantics. A target position can be held or interpolated according to controller behavior. A per-tick displacement must be integrated and then re-differenced at the new timestep; simply repeating it at twice the rate doubles motion. Preserve gripper events and terminal transitions explicitly.

Normalize using training data only, after semantic conversion. Missing joints need masks, not fake examples of always-zero behavior. Preserve masks through batching, loss calculation, and metrics. Save normalizer parameters and their source manifest as part of the model artifact.

Augmentation must preserve the learning problem. Photometric jitter can simulate lighting variation, but aggressive color changes can erase an instruction's target identity. A horizontal flip changes left/right language and camera geometry; it is not automatically label-preserving. Independently cropping sequential frames can create artificial camera motion. Apply geometrically consistent transforms across time and views where the representation requires them, and update associated geometry or labels explicitly. Keep evaluation preprocessing fixed and save augmentation parameters in the training configuration.

## Splits should match the generalization claim

A random frame split is almost never a credible independence test for robot episodes. Neighboring frames share object pose, background, and trajectory. Start with whole-episode separation, then group by the factor your claim holds out: session, environment, object instance, robot, task, or operator.

Holdouts can conflict. If one robot exists in only one environment, an environment holdout also becomes a robot holdout. Report these confounds. Keep a small development split for iteration and a locked final test split. Repeatedly checking the test set to choose a normalizer, mixture, or training duration turns it into a development set.

Deduplication must span source datasets. Two datasets may contain the same episodes after export, crop, or compression. Track source identifiers and hashes; use image similarity only as supporting evidence. A shared scene is not automatically a duplicate, but an identical trajectory under a new filename is.

## Worked example: mixture weights alter the task

Suppose dataset A has 90,000 episodes and B has 10,000. Sampling proportionally to count gives A 90% of draws. Sampling each dataset equally gives B five times its previous probability. A temperature-style rule

$$
p_i=\frac{n_i^\alpha}{\sum_j n_j^\alpha}
$$

interpolates between uniform datasets at \(\alpha=0\) and count-proportional sampling at \(\alpha=1\). At \(\alpha=0.5\), this example gives A 75% and B 25%. These are analytical probabilities, not evidence that one mixture learns better.

The unit of \(n_i\) matters. Counting frames favors long, high-frequency episodes; counting episodes favors neither their duration nor their information content. Record effective samples drawn per task and robot. Importance weighting can estimate another target distribution, but extreme weights increase estimator variance. A mixture is part of the training objective and belongs in the experiment record.

## Build the conversion pipeline

Start with the supplied [dataset-to-execution lab](Dataset-and-Execution-Lab.md): its JSON recordings convert camera-frame millimeter deltas and base-frame velocities, preserve causal camera/state selection, fit training-only statistics, and reject episode/session leakage. Run its worked examples before implementing the broader exercise below.

**Independent exercise.** Create two additional small synthetic sources with intentionally different units, action frames, frequencies, and episode lengths. Use the supplied implementation as a reference, then document the new action semantics rather than silently extending its two-dimensional contract.

1. Write adapters to a canonical schema and record a transformation graph. Include source checksum, converter revision, and schema version in the output manifest.
2. Construct at least one analytically known motion in both source formats. Convert both and compare the recovered physical motion, not merely tensor values.
3. Create chunk windows only within episodes. At the final observations, pad targets and set explicit masks. Check that no chunk crosses a reset.
4. Build splits grouped by session. Add a copied trajectory under a second source ID and require the audit to flag it before training.
5. Estimate training-only normalizers and perform round-trip checks. Test constant dimensions, missing cameras, and an invalid timestamp.
6. Simulate 100,000 mixture draws for several \(\alpha\) values. Compare observed frequencies to analytical probabilities and report expected exposure per source.
7. Benchmark data loading with and without image decode. Record samples/s, host memory, storage bytes read, and waiting time observed by the training loop.

Stop after both formats pass the audit; additional datasets should be added one adapter at a time. **Acceptance criteria:** frame/unit transformations preserve a known motion; zero reset-crossing chunks; no overlap in grouped splits; duplicate injection is detected; normalization excludes held-out data; mixture probabilities sum to one and their empirical frequencies agree within sampling uncertainty. Synthetic fixtures cannot establish the correctness of undocumented real datasets.

## Failure clinic and self-check

**Failure clinic:** mixed data learning worse can indicate conflicting action conventions; suspiciously strong validation can indicate episode leakage; many zero outputs can indicate unmasked padding; GPU starvation can indicate video seek/decode bottlenecks rather than insufficient GPU memory.

1. **Why cannot you split by file?** Shards may contain multiple episodes, and episodes or related sessions can span files.
2. **Is a seven-dimensional action standardized?** Only its length is known; meaning requires units, frame, mode, and timing.
3. **Why save the mixture schedule?** It determines which empirical distribution generated gradients and may change throughout training.
4. **What is a valid dataset handoff?** Data, contracts, converter revision, split manifest, normalization statistics, and audit evidence together.

[Previous: data collection](Lecture-13.md) · [Course index](README.md) · [Next: pretraining and transfer](Lecture-15.md)
