# Lecture 15 — Pretraining and Cross-Embodiment Generalization

Design a pretraining experiment that separates semantic transfer, motor transfer, and simple overlap with the evaluation data.

**Prerequisites:** Lectures 06–08 and 12–14; multitask learning and dataset mixtures. **Effort:** setup 1 hour, reduced lab 5 hours, analysis 2 hours. **Hardware:** CPU for the tiny shared-model experiment; optional GPU for adaptation of a documented checkpoint. **Stack/roles:** architecture, training objectives, data, and hardware budgeting; foundation-model and robot learning engineers. **Artifact:** a transfer matrix and controlled pretraining study.

## What knowledge is supposed to transfer?

Vision-language pretraining can provide object categories, relations, and instruction representations. It does not by itself define a robot's motor scale or contact dynamics. Robot pretraining can teach sensor-to-action patterns, recovery behavior, and shared manipulation structure. Its benefit depends on the overlap and compatibility between source and target embodiments.

Separate four claims: transfer to new object appearances, new object instances, new tasks, and new robot embodiments. A model evaluated in an unseen room with a familiar robot has not demonstrated a new action interface. A model fine-tuned with target-robot demonstrations has shown adaptation, not zero-shot cross-embodiment control.

Open X-Embodiment assembled data across robots and studied shared policies, establishing a concrete research program for transfer through pooled robot experience. Its data formats and original evaluation scope should be distinguished from later models trained on selected mixtures. [Open X-Embodiment paper](https://arxiv.org/abs/2310.08864).

## Architecture determines where transfer is possible

Common approaches include a shared trunk with embodiment-specific output heads, a shared padded action space with masks, or a canonical action representation plus robot-specific conversion. Each pays a different cost. Separate heads simplify robot-specific semantics but add configuration. Padding simplifies batching but does not create semantic alignment. Canonical end-effector actions require reliable kinematics and a controller that can realize them.

Use embodiment context when identical numeric outputs mean different physical actions. This can be a robot identifier, morphology description, or calibrated state/action interface. Test whether the model relies on background appearance to infer the robot: that shortcut fails when cameras or environments change.

Octo illustrates a generalist policy designed for flexible observation and action interfaces and adaptation, while OpenVLA illustrates a VLM-based policy trained for robot action prediction with open training/adaptation infrastructure. Compare their input contracts and training objectives before comparing any benchmark table. [Octo, 2024](https://arxiv.org/abs/2405.12213), [OpenVLA, 2024](https://arxiv.org/abs/2406.09246).

## Objective composition and gradient routing

A teaching multitask objective is

$$
\mathcal L=\lambda_a\mathcal L_{action}
+\lambda_l\mathcal L_{language}
+\lambda_v\mathcal L_{vision}
+\lambda_g\mathcal L_{goal}.
$$

The loss weights do not by themselves specify the algorithm. For each term, define the data mixture, mask, reduction denominator, trainable modules, and gradient path. A token mean and an example mean have different scales. A high-frequency robot dataset can dominate even with a small nominal coefficient if sample construction yields many more examples.

Gradient interference occurs when two objectives push a shared parameter in opposing directions. Inspect per-objective gradient norms and their cosine similarity on a small diagnostic batch. These diagnostics are expensive at scale and should be sampled, not added blindly to every step. A decreasing auxiliary loss does not prove improved action control.

Knowledge insulation studies how to train a continuous action expert while protecting useful VLM training signals. Its proposed recipe combines discrete action supervision for the backbone with a stop-gradient boundary for the continuous expert's path into that backbone. This is a specific gradient-routing design, distinct from simply freezing the entire VLM. [Driess et al., 2025](https://arxiv.org/abs/2505.23705).

## The 2025–2026 progression

π₀.₅ studies heterogeneous co-training across robotic and non-robotic tasks, including high-level semantic prediction. The main lesson here is to specify exactly what additional supervision enters the system. [π₀.₅, April 2025](https://arxiv.org/abs/2504.16054).

The π₀.₇ preprint extends the study of generalist policies through diverse context conditioning, including performance metadata and subgoal images, and reports cross-embodiment and compositional generalization experiments. It was submitted April 16, 2026 and revised April 24. Treat those as reported experiments with their own datasets and protocols, not a guarantee that arbitrary unseen robots can be controlled. A technical report also does not establish publicly downloadable weights. [π₀.₇, v2](https://arxiv.org/abs/2604.15483v2).

These examples are evidence available by the course cutoff of October 13, 2026. They motivate experiments; the professional skill is identifying which mechanism explains a result and which competing explanation remains untested.

## Worked example: identical tensors, incompatible robots

Suppose robot A interprets action 0.1 as a 0.1-radian joint delta; robot B interprets 0.1 as a 0.1-meter Cartesian translation. Padding both to length seven leaves the conflict intact. A shared encoder with two explicit heads can model both. A canonical Cartesian representation might work too, but only after converting A using kinematics and defining the controller period.

For a transfer matrix, let \(S_{ij}\) be target \(j\)'s evaluation score after pretraining on source \(i\) and using a fixed adaptation budget. Compare \(S_{ij}\) with training the same architecture from scratch on target \(j\). Hold target demonstrations and compute budget constant. Otherwise a larger data budget can masquerade as positive transfer.

## Build a controlled transfer study

**Exercise / pseudocode.** Start from the [shared CPU lab](Lab-Guide.md), or a tiny state-based policy, and construct two synthetic embodiments with different action mappings and visual nuisance variables.

1. Define separate semantic and motor holdouts. Keep target task names, object identities, and physical mapping explicit in the manifest.
2. Compare target-only training, source pretraining plus target adaptation, and mixed training. Use the same architecture, target examples, evaluation seeds, and stopping budget.
3. Compare a shared head with missing embodiment context against explicit context or separate heads. This negative control should reveal a representational conflict.
4. Add an auxiliary language task and compare full gradient sharing with a stop-gradient boundary in the tiny model. Record gradient norms and action outcomes.
5. Evaluate familiar and shifted backgrounds. Swap embodiment labels as another negative control.
6. Report a transfer matrix with uncertainty over seeds, training time, peak memory, and target-data exposure. Include a case where transfer does not help.

Stop after three seeds and the preregistered model variants. The reduced experiment tests causal design and interface compatibility; it does not reproduce the scale or generalization claims of the cited models. **Acceptance criteria:** adaptation and zero-shot results have distinct labels; target-data budgets match; background shortcuts are probed; every auxiliary loss has an explicit gradient route; negative transfer is investigated rather than hidden.

## Failure clinic and self-check

**Failure clinic:** excellent language scores with poor control suggest missing motor grounding; source-task regression suggests gradient interference or forgetting; a new robot failing despite identical tensor shape suggests semantic incompatibility; transfer disappearing after scene holdout suggests a visual shortcut.

1. **Does cross-embodiment mean one universal action vector?** No; shared representations can coexist with explicit robot-specific interfaces.
2. **Is more heterogeneous data always better?** No; compatibility, weighting, quality, and optimization determine whether it helps.
3. **What distinguishes knowledge insulation from freezing?** The backbone can continue receiving selected learning signals while a particular gradient path is blocked.
4. **What makes a strong transfer claim?** Explicit holdouts, matched adaptation budgets, suitable baselines, and measurements tied to the claimed axis of generalization.

[Previous: dataset engineering](Lecture-14.md) · [Course index](README.md) · [Next: fine-tuning](Lecture-16.md)
