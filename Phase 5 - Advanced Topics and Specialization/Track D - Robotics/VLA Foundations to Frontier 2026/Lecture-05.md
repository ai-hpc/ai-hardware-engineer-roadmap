# Lecture 05 — Open X-Embodiment, RT-X, Octo, and OpenVLA

Learn why reusable robot policies require a data and interface contract as well as downloadable weights.

**Prerequisites:** [Lecture 04](Lecture-04.md), dataset splits, normalization, tensor shapes. **Effort:** 45 minutes setup, two hours lab, one hour analysis. **Hardware:** CPU for required manifest and data-adapter work; optional checkpoint execution needs a separately verified GPU budget. **Stack:** datasets, model architecture, training infrastructure, deployment integration. **Roles:** robotics data engineer, robot learning engineer, platform engineer.

[Course](README.md) · [Previous](Lecture-04.md) · [Next](Lecture-06.md)

## Data diversity creates an interface problem

A robot dataset is more than a collection of images and numbers. Every action has an embodiment, controller, unit, frame, rate, and normalization rule. Pooling datasets can increase behavioral diversity, but combining incompatible actions without conversion teaches contradictory targets.

Open X-Embodiment brought together robot learning datasets and studied cross-embodiment policies through RT-X models. The paper's first arXiv submission was October 13, 2023, and later revisions exist; cite a version when comparing counts or results. The central research question is whether experience from multiple robot setups can improve policies beyond isolated training. [OXE/RT-X paper](https://arxiv.org/abs/2310.08864)

The official repository provides dataset access and supporting material. Availability varies across constituent datasets, so an umbrella project link does not establish identical download procedures, licenses, or observation schemas for every component. [OXE repository](https://github.com/google-deepmind/open_x_embodiment)

The misconception is that standard storage format means standard physical meaning. A tensor named `action` is not an interoperability guarantee.

## Two useful open-policy design points

Octo's May 2024 paper describes a generalist policy trained on 800,000 trajectories, with language or goal-image conditioning and adaptation to new observation and action spaces. It offers a useful example of robot-policy pretraining that does not require a multi-billion-parameter language decoder. [Octo paper](https://arxiv.org/abs/2405.12213)

The official Octo repository describes transformer-based diffusion policies, released model variants, and masks for absent observations and history padding. Inspect the exact version when adopting these APIs: an absent camera and a camera containing a black image are semantically different inputs. [Octo implementation](https://github.com/octo-models/octo)

OpenVLA's June 2024 paper introduces a seven-billion-parameter model trained on 970,000 robot demonstrations, using a Llama 2 backbone and fused DINOv2/SigLIP visual features. Its significance here is an inspectable VLM-to-action training path and adaptation workflow. Reported comparisons belong to the paper's task and evaluation settings. [OpenVLA paper](https://arxiv.org/abs/2406.09246)

The OpenVLA repository exposes action prediction with dataset-specific unnormalization and distinguishes code from inherited model licensing. Its examples identify a particular robot/data configuration. Copying an example normalization key onto another robot without checking its statistics changes the physical action. [OpenVLA implementation](https://github.com/openvla/openvla)

## Cross-embodiment transfer has several meanings

Transfer can mean reusing a visual encoder, initializing a policy before fine-tuning, directly executing a pretrained policy on a matching setup, or deploying zero-shot on an unseen robot. These are progressively different claims. Always state what changed: object identity, scene, task wording, camera placement, controller, or morphology.

An embodiment identifier may help a shared network distinguish action conventions, but it does not mathematically align them. A six-joint arm and a seven-joint arm can share an end-effector action interface while retaining different reachability and singularity behavior. Conversely, identical joint-count vectors can mean different physical motions.

Map each source into a canonical intermediate representation only when the conversion is justified. Some datasets provide position deltas, others velocities or absolute poses. A velocity can be integrated into a delta only when the time interval is known. Absolute joint positions cannot be converted to a Cartesian delta without kinematics and the previous state.

If information is missing, preserve the limitation or exclude that component. Filling absent quantities with zeros can create a false observation that the model treats as real.

## Worked example: the wrong statistics multiply motion

Suppose a normalized scalar \(u\in[-1,1]\) is mapped to a physical displacement by

$$
a=q_{\mathrm{low}}+\frac{u+1}{2}
(q_{\mathrm{high}}-q_{\mathrm{low}}).
$$

Dataset A uses \([-0.02,0.02]\) meters, while dataset B uses \([-0.10,0.10]\) meters. A prediction \(u=0.5\) becomes 0.01 m under A and 0.05 m under B. Choosing the wrong statistics produces a fivefold motion error while every tensor shape remains valid.

Now suppose dataset B's values actually represent m/s at a 0.1 s step. Its 0.05 value should become a 0.005 m displacement. The same normalized prediction can therefore imply 5, 10, or 50 mm depending on the contract. This arithmetic is a synthetic fixture, not a reported model failure.

Mixture weights also matter. If dataset A has 90,000 episodes and B has 10,000, uniform episode sampling gives B 10% of examples. Uniform dataset sampling gives it 50%. Neither is intrinsically correct: the choice changes exposure, repetition, and the effective optimization objective. Report sample probabilities and actual sampled counts.

## What to pin for a usable checkpoint

Treat a deployment package as a connected set: model weights, model configuration, tokenizer or action head, image processor, normalization statistics, code revision, dependency lock, embodiment adapter, and evaluation assets. Hash local artifacts and record immutable source revisions. A date and the word `main` are useful provenance notes but insufficient reproduction pins.

Store image resizing and crop rules with the model. Changing the crop can remove a target object; changing pixel normalization can shift every feature. Store action horizon, output dimension ordering, and gripper conventions alongside the action statistics. Otherwise a correct neural prediction can become a wrong physical command downstream.

For hardware planning, estimate weights, activation workspace, and preprocessing separately. Model parameter count does not describe data-loader bandwidth, camera copies, or controller CPU contention. Keep research training cost separate from deployed inference cost.

## Lab: write an adapter audit before loading weights

**Exercise / pseudocode; no upstream checkpoint is executed.** Create two synthetic dataset manifests corresponding to A and B above. Each should specify action type, units, timestep, frame, dimension order, camera keys, and missing-data masks. Include 20 numerical records per dataset with known expected canonical displacements.

Implement adapters that convert valid records into meters per control step in a named base frame. Add a round-trip check where the transform is invertible. Reject records with a missing timestep when velocity integration is required, a mismatched dimension count, nonfinite values, or unknown frame. Keep the rejection reason in the output table.

The positive fixture must recover the 5 mm displacement for the velocity example. The negative fixture intentionally swaps the normalization dictionaries. Detect the mismatch using the manifest identifier or hash before evaluating task success; a magnitude bound alone may miss a plausible but wrong action.

For guided variation, remove the wrist camera from half the records. Preserve a presence mask and compare it to the incorrect strategy of silently inserting a black image. For independent transfer, propose a mapping for a bimanual joint-position dataset and list the kinematic information needed before conversion is possible.

Create a checkpoint readiness table for Octo and OpenVLA using the official repositories. Distinguish inspected documentation, available artifacts, selected revisions, and locally executed checks. Leave a revision field empty if no actual revision was selected; do not manufacture hashes.

Measure accepted/rejected record counts, conversion error, per-field completeness, and preprocessing time. Stop after 40 records, four negative cases, and the two readiness cards. The deliverable is the adapter code, manifests, raw validation output, and a transfer claim precise enough for another engineer to challenge.

## Self-check and exit criteria

1. **Why can identical action shapes conceal incompatible behavior?** Units, frames, absolute versus relative targets, time intervals, and gripper semantics can differ.
2. **Does fine-tuning on a new robot count as zero-shot cross-embodiment transfer?** No. State the adaptation data and which components changed.
3. **Why keep missing-observation masks?** Missing information is different from a legitimate observation whose numerical values happen to be zero.

Exit when every action can be traced through its source convention into the intended controller convention, and every model claim includes an explicit adaptation and reproducibility boundary.

[Previous](Lecture-04.md) · [Next: geometry and control contracts](Lecture-06.md)
