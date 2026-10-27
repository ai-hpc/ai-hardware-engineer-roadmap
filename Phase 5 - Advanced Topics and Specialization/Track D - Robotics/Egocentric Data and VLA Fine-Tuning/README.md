# Egocentric Data and VLA Fine-Tuning — Special Course

<div class="course-identity robotics" markdown="1">
<div class="course-identity__icon">EGO</div>
<div markdown="1">
<p class="course-identity__eyebrow">Phase 5 · Robotics · Special Course</p>
<p class="course-identity__title">Turn first-person human video into training data a robot can use, then fine-tune a vision-language-action policy on it and measure whether it helped.</p>
<p class="course-identity__meta">13 lectures · tested CPU labs · 8 GB and cloud tracks · capstone with a paired human-data ablation</p>
</div>
</div>

**Sources checked: October 26, 2026.** Versions, licenses and hardware requirements in this course are pinned to that date and listed in [Sources](Sources.md). This field changes monthly: re-check a pin before you build on it.

Robot demonstrations are expensive, often a few hours per task, collected by teleoperation on one robot. People wearing head-mounted cameras generate first-person ("egocentric") video of hands doing manipulation tasks at a far larger scale. Recent policies use that video directly. Physical Intelligence reports that human-to-robot transfer emerges once a VLA is pretrained broadly enough. NVIDIA's GR00T N1.7 is pretrained on 20,000 hours of egocentric human video in an action space shared by humans and robots. Research systems such as EgoMimic (co-training with robot data), EgoZero (smart glasses, no robot data) and Phantom (human demonstration video, no robot data) train manipulation policies largely from human video.

Using that data well is an engineering problem with three parts: **data you are allowed to use and can trust**, **a defensible mapping from human hands to robot actions**, and **a fine-tuning run whose effect you can measure**. This course teaches all three.

**Layer mapping:** L3–L8 — data systems and storage, model adaptation, training memory and throughput, and closed-loop evaluation.

**Role targets:** Robot Learning Data Engineer · Robot Foundation-Model (post-training) Engineer · Embodied AI Research Engineer · ML Data Platform Engineer for robotics.

**Prerequisites:**

* Python and basic linear algebra (rotations, rigid transforms), probability, and gradient descent.
* [VLA Foundations to Frontier 2026](../VLA%20Foundations%20to%20Frontier%202026/README.md), Lectures 01, 06, 09 and 12: what a VLA is, frames and action contracts, behavioral cloning, flow matching. Lectures 13, 14 and 16 of that course (robot data collection, dataset schemas and mixtures, fine-tuning and normalization) are the short versions of this course's Lectures 04 and 10–11; this course goes deeper and adds egocentric human data.
* For the evaluation lecture: [Deep RL for Robot Learning, Lecture 13](../Deep%20RL%20for%20Robot%20Learning/Lecture-13.md) (confidence intervals and evaluation rigor).

**Entry diagnostic:** compose two rigid transforms and say which frame the result maps from and to; explain why a random train/test split of video clips can leak; normalize a vector with mean/std and undo it; and estimate the bytes needed to store the AdamW state of a 450M-parameter model. If two of these are uncertain, start with the prerequisites above.

---

## Hardware tracks

| Track | Hardware | What you can do here | Sources |
|---|---|---|---|
| **CPU** | Any laptop | All labs in this course (geometry, auditing, retargeting, dataset contracts, budgets, evaluation statistics) | `labs/`, tested here |
| **8 GB GPU** (e.g. RTX 4060) | Consumer GPU | Train ACT; run SmolVLA inference; attempt SmolVLA fine-tuning at small batch with the frozen vision encoder and expert-only training (a stretch goal to verify, not a documented fit) | LeRobot hardware guide (indicative figures) |
| **24 GB** | RTX 4090 / L4 / A10G | SmolVLA fine-tuning at batch 8 (indicative "~10–16GB"); openpi π0/π0-FAST **LoRA** ("> 22.5 GB") | LeRobot, openpi README |
| **40–80 GB** | A100 / H100 (rented) | GR00T N1.7 fine-tuning ("40 GB+", default head-only tuning peaks under ~35 GB); π0.5 full fine-tuning (openpi "> 70 GB"; LeRobot "sized for a single 80 GB GPU"); OpenVLA-OFT | Isaac-GR00T, openpi, LeRobot, OpenVLA-OFT docs |

Every lecture states which track its exercise needs and offers a reduced CPU exercise when the full one needs a GPU you may not have.

---

## Lecture map

<div class="lecture-map" markdown>

| # | Lecture | Inspectable artifact |
|---|---|---|
| | **Part A — Egocentric human data** | |
| 01 | [Why Egocentric Human Data for Robot Learning](Lecture-01.md) | Data-pyramid budget and embodiment-gap inventory for one task |
| 02 | [The Egocentric Dataset Landscape and Its Licenses](Lecture-02.md) | Dataset selection record with license and access audit |
| 03 | [Capture Geometry: Head-Mounted Cameras, SLAM and Time](Lecture-03.md) | Verified camera → device → world chain (`labs/ego_geometry.py`) |
| 04 | [The Data Pipeline: Decoding, Sharding and the LeRobot Format](Lecture-04.md) | Storage/decode budget and a validated LeRobot-style episode layout |
| 05 | [Auditing a Dataset: Statistics, Leakage and Label Quality](Lecture-05.md) | Data card with leakage, duplicate and label-quality findings (`labs/ego_audit.py`) |
| 06 | [Hands and Objects: 3D Hand Pose, Contact and Failure Modes](Lecture-06.md) | Hand-pose quality report with an occlusion failure taxonomy |
| 07 | [From Human Hands to Robot Actions: Retargeting and Feasibility](Lecture-07.md) | Retargeted 7-D action stream with a feasibility report (`labs/retarget.py`) |
| 08 | [Learning from Video Without Robot Actions](Lecture-08.md) | Comparison of latent actions, pseudo-labels and co-training for one task |
| | **Part B — VLA fine-tuning** | |
| 09 | [Choosing a Base Model and Fine-Tuning Stack](Lecture-09.md) | Stack-selection record: license, format, normalization and hardware |
| 10 | [Building the Fine-Tuning Dataset: Formats, Normalization and Mixtures](Lecture-10.md) | Converted dataset with stats parity and a mixture plan (`labs/dataset_contract.py`) |
| 11 | [Fine-Tuning Mechanics on 8, 24 and 80 GB](Lecture-11.md) | Memory budget, training run record and checkpoint-selection rule (`labs/budget_eval.py`) |
| 12 | [Evaluating Fine-Tuned Policies and What Human Data Bought](Lecture-12.md) | Paired evaluation with confidence intervals and a failure taxonomy |
| 13 | [Capstone: An Egocentric-Augmented Fine-Tune](Lecture-13.md) | Data card, model card and a paired with/without-human-data result |

</div>

Labs and their tests are in [`labs/`](Lab-Guide.md). They are pure Python, run in seconds on a CPU, and run in this roadmap's CI on every change.

---

## Why this is a hardware-first course

* **Video is the expensive part.** Egocentric datasets are measured in thousands of hours. Decoding, storing and shuffling that video bounds a training run long before the model does. Lecture 04 budgets disk, decode throughput and GPU memory before any training starts.
* **Fine-tuning memory is mostly optimizer state.** With AdamW in mixed precision, each *trainable* parameter costs about 16 bytes (bf16 weight and gradient, an fp32 master copy and two fp32 moments), while a frozen one costs only its 2-byte weight. Which parameters you train (LoRA, the action expert only, or everything) decides whether a run fits on 8, 24 or 80 GB. Lecture 11 does the arithmetic and checks it against documented requirements.
* **Normalization statistics are part of the model.** The same checkpoint fed actions on the wrong scale fails silently. Each stack computes and stores them differently (mean/std, min-max or q01/q99 quantiles). Lecture 10 makes stats parity a gate.
* **Deployment latency follows from data choices.** Action-chunk length, control rate and image resolution are set when you build the dataset, and they set the inference budget later. The [VLA Optimization](../VLA%20Optimization%20and%20Action-Parity%20Harness/README.md) course picks up from there.

---

## What you ship

* **A dataset selection record** — sources, licenses (commercial or not), access terms and why each source is in or out.
* **A data card** — statistics, leakage checks, duplicates, label quality and known failure modes, produced by the audit lab on your data.
* **A retargeting report** — human hand tracks converted to a 7-D action stream, with frames, calibration and feasibility checks you can defend.
* **A fine-tuning record** — stack and version pins, dataset version, normalization stats, memory budget, the run record and the checkpoint-selection rule.
* **A paired evaluation** — the same episodes run with and without human data, with confidence intervals and an exact paired test.

---

## Exit criteria

You are done when you can:

* choose egocentric sources for a stated use (research or commercial) and justify each license decision
* trace a 3D hand point from a fisheye pixel to the robot base frame, naming every transform and clock
* show that your train/test split does not leak participants or locations, with the audit output to prove it
* convert human hand tracks into robot actions and say which parts of a clip a given arm cannot execute
* fine-tune an open VLA with a dataset whose schema and normalization statistics are verified against the target stack
* report whether human data helped, with a paired evaluation that another engineer can reproduce

---

## References for the whole course

* [Sources](Sources.md) — every pinned repository, dataset, paper and license used in the course, with the date checked.
* [VLA Foundations to Frontier 2026](../VLA%20Foundations%20to%20Frontier%202026/README.md) — the architecture and training foundations this course builds on.
* [Curriculum Authoring Guide](../../../Curriculum-Authoring-Guide.md) — how commands are labeled (tested here, upstream documented, exercise) and how evidence is reported.

---

## Next

* Start: [Lecture 01 — Why Egocentric Human Data for Robot Learning](Lecture-01.md)
* Back: [Track D — Robotics Guide](../Guide.md)
