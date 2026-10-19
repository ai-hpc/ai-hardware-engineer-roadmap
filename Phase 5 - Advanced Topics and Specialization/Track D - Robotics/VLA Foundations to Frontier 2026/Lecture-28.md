# Lecture 28 — Open VLA systems: choose an artifact, not a headline

Compare accessible policy families by their interfaces, reproduction costs, and training assumptions, then run a controlled adaptation study.

**Stack:** model workloads, training systems, memory, deployment. **Roles:** research engineer, robot learning engineer. **Prerequisites:** [Lecture 27](Lecture-27.md), fine-tuning and benchmark design. **Next:** [Reasoning and world models](Lecture-29.md). **Cutoff:** 2026-10-13.

## Three practical reference points

OpenVLA-OFT is a 2025-02-27 paper and adaptation recipe for OpenVLA. Its combination of parallel decoding, chunking, continuous outputs, and an L1 objective illustrates that adapting a pretrained backbone can change the policy's inference path substantially. Calling it “OpenVLA” without specifying OFT loses an important architectural distinction. [OFT paper](https://arxiv.org/abs/2502.19645).

SmolVLA was announced on 2025-06-03 as a compact, roughly 450M model with a flow-matching action expert and community-data training. It is a useful reference for constrained hardware, not a promise that any CPU can meet a robot's deadline. Its released base checkpoint is intended for adaptation and lists Apache-2.0 licensing. [Announcement](https://huggingface.co/blog/smolvla), [base model card](https://huggingface.co/lerobot/smolvla_base).

RDT-1B first appeared on 2024-10-10. It uses a diffusion transformer and a physically meaningful unified action representation for bimanual manipulation. Its repository supplies both the larger model and an RDT-170M alternative, with MIT licensing stated for its code, weights, and released data. Keep separately acquired dataset and encoder terms in your manifest. [Paper](https://arxiv.org/abs/2410.07864), [implementation](https://github.com/thu-ml/RoboticsDiffusionTransformer).

The point is not to crown an old or new winner. These families expose different engineering choices: backbone adaptation, compact inference, multimodal action generation, and cross-embodiment representation. Your first benchmark should make those differences inspectable.

## Substantial 2026 additions

MolmoAct 2 was announced 2026-05-05; the release post records a May 28 code/data update, while the repository records expanded training code on June 13. It pairs an embodied-reasoning backbone with a flow action expert and offers a depth-reasoning variant. Public base and adapted policies, datasets, and evaluation material make it a useful reproduction candidate. The repository states Apache-2.0; retain individual artifact provenance. [Ai2 announcement](https://allenai.org/blog/molmoact2), [author implementation](https://github.com/allenai/molmoact2).

LingBot-VLA 1.0's repository dates its code/weight release to 2026-01-27. LingBot-VLA 2.0 has a paper dated 2026-07-07 and repository release news dated July 8. Version 2 studies expanded robot-body coverage and predictive representation learning. Its public 6B card describes an MoE action expert and Apache-2.0 terms. These are verifiable artifacts, unlike a model name appearing only in a leaderboard row. [1.0 repository](https://github.com/Robbyant/lingbot-vla), [2.0 paper](https://arxiv.org/abs/2607.06403), [2.0 card](https://huggingface.co/robbyant/lingbot-vla-v2-6b).

VLAct's paper was submitted 2026-08-27 and revised 2026-10-06. It focuses on reusable action-aware backbone representations through prior preservation and multiple action-head supervision. Its transfer object is a backbone; a downstream head still needs configuration and adaptation. [Paper](https://arxiv.org/abs/2608.27550v2).

The released VLAct repository also documents discrepancies between paper settings and downloadable artifacts, including auxiliary-loss weight and some downstream head choices. That is valuable provenance, not a detail to hide. Reproducing the downloadable checkpoint and reproducing a paper table can require different configurations. [VLAct implementation](https://github.com/starVLA/VLAct).

Qwen-VLA, submitted 2026-05-28, studies manipulation, navigation, and trajectory generation with a DiT action decoder and embodiment-aware conditioning. This review verified its paper, not a complete downloadable deployment package; classify it accordingly instead of inferring access from Qwen's other open models. [Qwen-VLA paper](https://arxiv.org/abs/2605.30280).

## Mental model: the comparable unit is an experiment

Represent a candidate as

\[
C=(W,P,N,D,E,R),
\]

where \(W\) is weights, \(P\) is processing, \(N\) is normalization/action semantics, \(D\) is adaptation data, \(E\) is evaluation, and \(R\) is runtime. A model-family name supplies none of these completely.

For example, changing the action head from discrete autoregressive outputs to parallel regression changes both compute and the learned conditional distribution. Changing the camera count changes information available to the policy. Changing episode horizon changes opportunity to recover. A comparison that changes all three can still be operationally useful, but it cannot isolate the effect of one architecture.

Use two study designs. A **deployment comparison** gives each system its supported configuration and asks which meets your application requirements. A **controlled ablation** fixes the backbone, data, and training budget and changes one mechanism. State which design you used before interpreting results.

## Worked example: fitting is not running

An **analytical** BF16 weight estimate gives 0.9 GB for 450M parameters, 12 GB for 6B, and 14 GB for 7B. Those numbers omit runtime memory, frozen encoders outside the quoted parameter count, and allocator overhead. A 16 GB device therefore cannot be selected by subtracting the weight file from 16 and declaring success.

For training, a rough unfactored Adam accounting may include parameters, gradients, master copies, and two optimizer moments. Depending on precision and implementation, a planning estimate near 16 bytes per trainable parameter can be useful, before activations. A 6B fully trainable model would then require roughly 96 GB for those states alone. This estimate is neither a measured requirement nor a universal optimizer rule.

LoRA lowers trainable-state cost but does not eliminate the frozen model, activation storage, or data processing. Suppose adaptation uses only 30M trainable parameters: at the same illustrative 16 bytes, their states are about 480 MB, yet the 12 GB BF16 backbone is still present. Measure total peak memory instead of advertising the adapter size as the memory requirement.

For context, the inspected OFT repository documents about 16 GB inference memory for its LIBERO path and more for its ALOHA path; these are upstream configuration-specific requirements. [OFT requirements at the reviewed commit](https://github.com/moojink/openvla-oft/tree/e4287e94541f459edc4feabc4e181f537cd569a8).

## Build it: one baseline and one challenger

**Effort:** 1–2 hours setup, 3–6 hours execution, 2 hours analysis. **Hardware:** a GPU sized for the selected pinned recipes; CPU reduction for metadata, normalization, and experiment accounting. **Stopping condition:** one baseline smoke test, a predeclared bounded adaptation budget, and at most 30 pilot evaluation episodes per configuration. A pilot estimates feasibility, not leaderboard standing.

1. Select a task-supported baseline and one challenger. Choose accessible weights and an evaluator you can actually run. A smaller model is a valid professional choice when the learning objective is measuring a complete workflow.
2. Write a manifest for code, weights, base encoders, processors, dataset revisions, splits, action convention, controller rate, camera streams, and licenses. Record whether training data are public, partially public, or unavailable.
3. Visualize ten demonstrations. Look for clock misalignment, gripper inversion, impossible velocities, and instruction ambiguity. Split by episode or scene, never randomly across adjacent video frames.
4. Build a lossless adapter for supported fields and explicit masks for unavailable dimensions. Verify normalization/inversion on known data. Do not fill absent physical measurements with plausible numbers and call them ground truth.
5. Run the pinned upstream training/evaluation entry point. Label these instructions **upstream documented, not tested here**. Capture the resolved configuration and raw logs; do not depend on an unrecorded command-line default.
6. Compare held-out closed-loop success, time to completion, p95 sensor-to-action latency, peak VRAM, interventions, and failure categories. Also log offline action error, while explaining why it does not replace closed-loop evaluation.

The negative case swaps two camera labels while retaining valid tensor shapes. A provenance-aware mapping check or a known-view fixture can detect the mismatch. Merely declaring valid camera IDs cannot prove the corresponding pixel contents came from those cameras: falsely relabeled images still satisfy the schema. Preserve source identity through acquisition and preprocessing, and test scene markers or geometric consistency where available. Restore the correct mapping before any robot execution.

**Exercise / pseudocode:**

```python
for candidate in candidates:
    validate_contract(candidate.manifest, task.contract)
    for episode_seed in held_out_seeds:
        result = evaluate_closed_loop(candidate, task, episode_seed)
        write_episode_record(candidate.revision, episode_seed, result)
```

The CPU reduction should produce a complete manifest and adapter round-trip report. It cannot demonstrate the full policies' success or latency.

## Guided variation and independent transfer

Repeat the baseline with half as many demonstrations, keeping the split and training-update budget explicit. Distinguish “same number of updates” from “same number of epochs”; they answer different data-efficiency questions. Examine whether failures are concentrated in one object or camera condition.

For the independent transfer task, compare adding a new action head to a reusable backbone against fine-tuning a complete released policy. Budget the data engineering, integration, and measurement effort as well as GPU hours. Explain why a highly ranked checkpoint with incompatible actions may be the expensive option.

## Self-check and completion

1. **Is a pretrained backbone a deployable policy?** No. It still needs a compatible head, preprocessing, normalization, and embodiment-specific evaluation.
2. **Does an MIT code repository determine every dependency's license?** No. Check weights, base models, datasets, and assets separately.
3. **Why can a large model lose on a deployment comparison?** It can miss deadlines, exhaust memory, or require unavailable inputs despite stronger offline capability.
4. **What proves an adaptation gains transfer rather than memorization?** A held-out split that changes the relevant factor, with controlled baselines and complete failure records.

**Ship:** two pinned candidate manifests, an adapter round-trip test, data-audit images, a bounded adaptation log, and a comparison whose claims match its experimental design. An unsupported candidate may be rejected before expensive execution if the incompatibility is documented precisely.
