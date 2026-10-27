# Lecture 08 — Learning from Video Without Robot Actions

Compare the ways published systems turn human video into a training signal for robot actions, and choose one for a stated task and data budget.

**Prerequisites:** [Lecture 07](Lecture-07.md) (retargeting), [Lecture 01](Lecture-01.md) (the embodiment gap), [VLA Foundations Lecture 15](../VLA%20Foundations%20to%20Frontier%202026/Lecture-15.md) (pretraining and cross-embodiment). **Effort:** reading 2 hours, exercise 2 hours. **Hardware:** CPU track; the exercise is a written comparison plus one tested lab run. **Roles:** robot foundation-model engineer, embodied AI research engineer.

**Misconception this lecture corrects:** "a clever enough method makes robot data unnecessary." Every method below either still uses robot data (to ground latent actions, train an inverse dynamics model, align features or fine-tune), or reports zero-robot-data results on a narrow set of tasks with a carefully engineered bridge across the visual and kinematic gaps.

## 1. Six families

Human video has frames, often language, and sometimes tracked hands and head pose. It has no robot actions. The published methods differ in what they put in the action slot:

| Family | What fills the action slot | Examples |
|---|---|---|
| **Latent actions** | A learned code for "what changed between two frames" | LAPA (arXiv 2410.11758), GR00T N1 (arXiv 2503.14734) |
| **Pseudo-actions** | Actions predicted by an inverse dynamics model (IDM) | GR00T N1, ACE-Ego-0 (arXiv 2606.17200) |
| **Explicit hand-to-action** | Tracked hand motion in a space shared with the robot | EgoMimic, Humanoid Policy ~ Human Policy, π0.5 human-to-robot transfer, GR00T N1.7, EgoScale, VITRA |
| **Visual-gap methods** | Explicit actions, plus images or features made to match across embodiments | Phantom, EgoZero, EgoBridge |
| **Representation pretraining** | Nothing; video trains the vision-language backbone | PhysBrain (arXiv 2512.16793) |
| **Hierarchical latent plans** | A plan learned from human video; robot demos train the low-level controller | MimicPlay (arXiv 2302.12422) |

## 2. What each method needs and what it reports

"Reported" means the paper, blog or README states it under its own tasks and hardware. None of it was reproduced in this course. "Not recorded here" means this course's sources hold no number for it.

| Method | Supervision taken from human data | Robot data still needed | Reported | Code / data terms |
|---|---|---|---|---|
| LAPA | Frame pairs; a VQ-VAE learns discrete latent actions, no action labels | A small robot set to map latents to robot actions | Positive transfer from human manipulation video alone | Code MIT (`46aca51`) |
| GR00T N1 (latent) | VQ-VAE on consecutive frames of all data; the "continuous pre-quantized embedding" is the label, treated "as a distinct 'LAPA' embodiment" | Real robot trajectories with ground-truth actions in the same pretraining | Human video used includes Ego4D 2,144.7 h, HoloAssist 169.6 h, Ego-Exo4D 123.0 h | Isaac-GR00T, code Apache-2.0 |
| GR00T N1 (IDM) | None directly: a separately trained IDM labels generated robot video, not human video ("neural trajectories": 88 h of teleoperation → about 827 h of video) | An IDM can only be trained on frames that have actions | About 105k L40 GPU-hours to generate the video | Same |
| EgoMimic | Aria 3D hand tracking plus SLAM device pose | Co-training with a low-cost bimanual arm | "1 hour of additional hand data is significantly more valuable than 1 hour of additional robot data" | Code MIT; data has no stated license |
| Humanoid Policy ~ Human Policy | PH2D head and hand poses in a unified human/humanoid state–action space with differentiable retargeting | Humanoid demos, fewer of them | Co-training with less robot data | Code MIT; PH2D card `mit` |
| π0.5 human-to-robot | Human video as another embodiment, "with actions given by 3D hand positions" | Broad robot pretraining: transfer "emerges" once pretraining covers enough scenes, tasks and embodiments | About 2× on 4 human-only generalization scenarios | Not recorded here |
| GR00T N1.7 | Relative end-effector (EEF) actions shared by robot and human; "20K hours of EgoScale human video" | Post-training on your robot's embodiment | Not recorded here | Code Apache-2.0; weights: see [Lecture 09](Lecture-09.md) |
| EgoScale | "off-the-shelf SLAM and hand-pose estimation" (labels the paper calls noisy) over 20,854 h; a 22-DoF hand | Aligned human–robot mid-training (Vive trackers, Manus gloves, head and wrist cameras) | A log-linear scaling law | Not released |
| VITRA | 3D hand and camera motion extracted automatically from unannotated video: 1M episodes, 26M frames | Pretraining data; check the paper for its robot set | Not recorded here | Code MIT; VITRA-1M is metadata only, source videos keep their licenses |
| Phantom | Hand pose (a HaMeR fork with MANO), arm inpainting, a rendered robot overlay | None reported | Up to 92% success, zero robot data | Code MIT; MANO is non-commercial |
| EgoZero | Robot-executable actions from Aria demos; morphology-agnostic state representations | None reported | 70% zero-shot over 7 tasks on a Franka, about 20 min of data per task | Code MIT |
| EgoBridge | Human and robot data; optimal transport aligns their policy latents | Yes; there must be robot latents to align to | +44% absolute over human-augmented baselines | Not recorded here |
| PhysBrain | E2E-3M, a dataset built from egocentric video, gives an egocentric-aware VLM initialization | All the action learning, in VLA fine-tuning | Not recorded here | Not recorded here |
| MimicPlay | Latent plans from human play video | A few teleoperated demos for the low-level controller | Evaluated on 14 real long-horizon tasks | Code MIT |

## 3. Latent actions and pseudo-actions

A **latent action** is a code \( z_t \) computed from two frames \( (o_t, o_{t+k}) \). The policy learns to predict \( z_t \) from \( o_t \) and the instruction, so any video becomes trainable. Two costs follow:

* A latent has no units. Nothing drives a motor until the model also learns robot actions from robot data. LAPA does this in fine-tuning; GR00T N1 trains latent and ground-truth actions in one model.
* The code captures whatever changed between frames, and nothing forces that to be the hand. A head-mounted camera moves with the wearer every frame (Lecture 01), so camera motion is part of every change. Lecture 03's SLAM poses tell you how much.

GR00T N1 trains its VQ-VAE on all its heterogeneous data together, so robot, human and generated video share one latent space.

A **pseudo-action** is a robot-unit action predicted by an inverse dynamics model, \( \hat a_t = \mathrm{IDM}(o_t, o_{t+k}) \). It needs action-labeled data to train the IDM, and the IDM's errors become label noise. GR00T N1 applies pseudo-labels to generated robot video; its human video gets latent actions. ACE-Ego-0 reports pretraining on 1.48K h of human data with pseudo-action labels next to 4.53K h of robot and simulation data.

## 4. Explicit hand actions in a shared space

This is the path Lecture 07 built: wrist and fingertip tracks become a 6-D relative EEF step plus a gripper opening, 7-D in all. The published versions differ in the shared space. GR00T N1.7 uses relative EEF actions for robots and humans. π0.5 uses 3D hand positions. Humanoid Policy ~ Human Policy unifies human and humanoid state and action. EgoScale uses a 22-DoF hand. Three cautions:

* **Transfer depends on the base model.** Physical Intelligence frames human-to-robot transfer as something that emerges with broad pretraining. Do not expect the same effect when fine-tuning a small model on a narrow dataset; test it (Lecture 12).
* **Labels are estimates.** EgoScale calls its SLAM and hand-pose labels noisy. Lecture 06's quality report decides which clips are clean enough to enter the mix.
* **Labels carry licenses.** Labels produced with MANO, as in Phantom's HaMeR fork, inherit MANO's non-commercial terms (Lecture 02).

## 5. Closing the visual gap

Explicit actions still leave a human arm in the image. Phantom inpaints the arm and renders the robot over it, so training images look like deployment images. EgoZero uses state representations that do not depend on the body's appearance. EgoBridge leaves the images alone and aligns human and robot policy latents with optimal transport. HumanEgo (arXiv 2605.24934) reports 92.5% average success without robot data from 30 min of human video per task, but its code is under the PolyForm Noncommercial 1.0.0 license.

A zero-robot-data success rate answers "can human video alone solve this task on this setup?" It does not answer "how much does adding human video improve my robot-data fine-tune?" Only a paired ablation answers that (Lecture 12).

## 6. Co-training mixtures: weights, not sizes

When human and robot data train one policy, the sampler decides how much gradient each source gets. Sampling in proportion to size lets a large human set drown a small robot set. `budget_eval.mixture_batches` draws each example's source by weight, and `epochs_seen` reports how often each source repeats.

**Tested here** (from the repository root; `EGO_COURSE` is set as in the [Lab guide](Lab-Guide.md)). Sources are counted in frames: 100 h of human video at 30 fps against Lecture 01's 50 robot episodes of 30 s at 30 fps, over 20,000 batches of 64.

```bash
PYTHONPATH="$EGO_COURSE/labs" python3 - <<'EOF'
from budget_eval import mixture_batches, epochs_seen
sources = {'human': 10_800_000, 'robot': 45_000}
plans = [('size-proportional', dict(sources)), ('robot 0.3', {'human': 0.7, 'robot': 0.3}),
         ('robot 0.5', {'human': 0.5, 'robot': 0.5})]
for name, w in plans:
    c = mixture_batches(sources, w, batch_size=64, num_batches=20_000, seed=0)
    n = sum(c.values())
    print(f"{name:18s} robot share {c['robot']/n:.4f}  robot epochs {epochs_seen(sources['robot'], c['robot']):6.2f}"
          f"  human epochs {epochs_seen(sources['human'], c['human']):.3f}")
EOF
```

```text
size-proportional  robot share 0.0041  robot epochs   0.12  human epochs 0.118
robot 0.3          robot share 0.2996  robot epochs   8.52  human epochs 0.083
robot 0.5          robot share 0.4995  robot epochs  14.21  human epochs 0.059
```

**Negative case:** the size-proportional plan gives the robot 0.4% of examples, and the average robot frame is seen 0.12 times, so most robot frames are never seen at all. The weighted plans keep the robot visible, at the price of repetition: at weight 0.5 each robot frame is seen about 14 times, which risks overfitting. The weight is a hyperparameter to evaluate, not a constant to copy. Lecture 10 turns it into a mixture plan.

## 7. Choosing a method

| Your data | Start with | Why |
|---|---|---|
| Robot demos for the task, plus human video of the same task family with 3D hand tracking and SLAM | Explicit hand-to-action co-training, either on a base pretrained with human data in a shared space (GR00T N1.7) or on a broadly pretrained base, as in π0.5's human-to-robot recipe | Lecture 07's retargeting gives actions in the robot's space |
| Many hours of video without hand tracking, a small robot set | Latent actions (LAPA-style) | Needs no labels; the robot set grounds the latents |
| No robot data, one narrow task | A visual-gap method (Phantom, EgoZero) | Zero-robot-data results are reported, at narrow scope |
| A long-horizon task and human play video | Hierarchical plans (MimicPlay) | Video supplies the plan; robot demos supply the motor skill |
| You only want better perception | Representation pretraining (PhysBrain) | No action machinery at all |

Then check the blockers: the license of every label source, whether the code or data is released (EgoScale is not), and the compute (GR00T N1's generated video cost about 105k L40 GPU-hours).

## 8. Exercise

**Exercise / pseudocode** (CPU only):

1. **Worked example → the course task.** Use the Lecture 01 budget: "put the cup in the bowl" on a 7-DoF arm with a parallel-jaw gripper, 50 robot episodes, 10 h of human clips with Aria-style hand tracking, of which about 4 h is assumed retarget-ready. The table in Section 7 points to explicit hand-to-action co-training on the retarget-ready hours, with latent actions as an option for the rest. Rerun the Section 6 snippet with `'human': 432_000` (4 h at 30 fps) and record the robot share and epochs for two weights.
2. **Guided variation.** The human clips lose their hand tracking and SLAM (plain head-camera video). Which rows of Section 2 can still use them? Which robot-data requirement grows?
3. **Independent transfer.** Take your own task and the budget from your `data_pyramid.md`. Compare at least four methods, choose one, and name the paired ablation that would show whether it helped.

**Deliverable:** `method_comparison.md`, with one row per candidate method: supervision required, whether your data has it (with evidence), robot data still needed, license or release blockers, the reported result with its citation, and the decision. Paste the mixture output for your sizes. **Acceptance:** every number is attributed to a paper or README, or labeled as a lab output; at least one method is rejected for a concrete reason (missing labels, license, not released, compute); the chosen method states its robot-data requirement; the mixture output shows the robot share and epochs for at least two weights.

## Failure clinic and self-check

**Failure clinic:** a latent-action policy is deployed with no robot fine-tuning, and its outputs have no meaning in joint or EEF units. A zero-robot-data success rate is quoted in a proposal as "what human data adds." A size-proportional sampler gives a 25-minute robot set under 1% of the examples next to 100 h of human video. A commercial project adopts Phantom without noticing the MANO dependency. A plan assumes EgoScale data can be downloaded. A team expects π0.5-style transfer from a small model trained on one task.

1. **Why does a latent-action policy still need robot data?** Latent codes have no units. A mapping from latents to robot actions has to be learned from robot data, either in fine-tuning (LAPA) or by training both together (GR00T N1).
2. **In GR00T N1, which data got IDM pseudo-actions, and what limits them in general?** The generated neural trajectories; human video got latent actions. An IDM needs action-labeled training data, and its errors become label noise.
3. **Your human set is 240 times larger than your robot set. What does size-proportional sampling do?** The Section 6 output: the robot gets about 0.4% of examples and each robot frame is seen about 0.12 times on average.
4. **Phantom reports up to 92% success with no robot data. Is that evidence that human data will improve your robot fine-tune?** No. It measures a different question on a different setup. Only a paired with/without-human-data evaluation on your task answers yours.
5. **Why might π0.5's human-to-robot result not carry over to a small model?** The paper reports that the transfer emerges once the VLA is pretrained on enough scenes, tasks and embodiments. A small, narrowly pretrained model may be below that point.

[Previous: retargeting and feasibility](Lecture-07.md) · [Course index](README.md) · [Next: choosing a base model and stack](Lecture-09.md)
