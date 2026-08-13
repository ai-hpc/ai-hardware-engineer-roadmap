# Practical Machine Learning — adapted from Stanford CS329P

<div class="course-identity ai-workloads" markdown="1">
<div class="course-identity__icon">PML</div>
<div markdown="1">
<p class="course-identity__eyebrow">Phase 3 · ML Engineering & MLOps · Special Course</p>
<p class="course-identity__title">The full machine-learning lifecycle the way a practitioner actually meets it — data, models, validation, distribution shift, tuning, compression, and responsible deployment — adapted from Stanford CS329P and refreshed for 2026.</p>
<p class="course-identity__meta">Artifact: an end-to-end ML study on one dataset + one model · Measure: validated generalization, robustness to shift, and a compressed model that hits a latency/memory budget</p>
</div>
</div>

> *"Machine learning topics that matter but are often skipped."* — the opening goal of CS329P. The famous courses teach you to fit a model. This one teaches you everything that surrounds the fit: where data comes from, how it lies to you, how the world drifts away from your training set, and what it costs to ship.

This course is an adaptation of **Stanford CS329P — Practical Machine Learning** (2021 Fall) by **Qingqing Huang, Mu Li, and Alex Smola** — the team behind [*Dive into Deep Learning*](https://d2l.ai/). The original is taught as a spine of three concerns: **Data → Model training → Deployment**. We keep that spine, ground every lecture in the original slide content, and add a **2026 refresh layer** — because the data tooling, the tuning practice, and especially the compression stack have moved hard since 2021 (INT4 LLM quantization, GPTQ/AWQ, modern distillation, the collapse of hand-rolled NAS).

This is the course that makes you dangerous in a real ML role. Most engineers can call `model.fit()`. Far fewer can tell you *why their validation score lied*, detect that production traffic has drifted, or shrink a model 4× without losing the accuracy that justified it. That gap is this syllabus.

**Layer mapping:** the ML-engineering layer — the practice that sits between a research model and a served product. It feeds directly into [Module 4B — ML Engineering & MLOps](../Guide.md) and the serving/compression work in [Phase 5 — ML Systems Engineering](../../../../Phase%205%20-%20Advanced%20Topics%20and%20Specialization/Track%20G%20-%20ML%20Systems%20Engineering/Guide.md).

**Role targets:** Machine Learning Engineer · Applied Scientist · MLOps Engineer · Data Scientist (modeling track) · ML Platform Engineer.

**Prerequisites:**

* Python fluency, basic statistics, and basic ML (what a loss function and gradient descent are).
* [Module 2 — Deep Learning Frameworks](../../../2.%20Deep%20Learning%20Frameworks/Guide.md) for the PyTorch used in the model and compression lectures.
* No prior MLOps knowledge needed — this course *is* the on-ramp to it.

**Pairs with:** [Module 4B — ML Engineering & MLOps](../Guide.md) (the infrastructure side of shipping) and, for the hardware payoff of Lecture 10, [Phase 5 — Model Compression and the inference stack](../../../../Phase%205%20-%20Advanced%20Topics%20and%20Specialization/Track%20G%20-%20ML%20Systems%20Engineering/MLSys%20Deep%20Dives/README.md).

---

## Why this course is structured the way it is

A research course optimizes for the model. A *practical* course optimizes for the moment the model meets reality — and reality attacks in a fixed order:

```text
   1. your data is dirty, unlabeled, and not yet features     → Lectures 01–02
   2. you pick and fit a model                                → Lecture 03
   3. your validation score is optimistic and you don't know  → Lectures 04–05
   4. production data drifts away from training                → Lectures 06–07
   5. the model is too slow / too big / under-tuned            → Lectures 08, 10
   6. you need to reuse pretrained knowledge, not start cold   → Lecture 09
   7. the model must be multimodal, fair, and explainable      → Lecture 11
```

Every lecture is one of these collisions. The thread is **generalization under reality**: not "does it fit the training set" but "does it still work when the data is messy, shifted, multimodal, and running on a budget."

---

## Course Map (11 lectures)

<div class="lecture-map" markdown>

| # | Lecture | The thread |
|---|---------|-----------|
| [01](Lecture-01.md) | **Data I — Acquisition, Scraping & Labeling** — where data comes from, web scraping, and the labeling stack (active learning, weak supervision, semi-supervised) | getting data |
| [02](Lecture-02.md) | **Data II — Cleaning, Transformation & Feature Engineering** — dirty-data taxonomy, normalization, and features for tabular / text / image | data → features |
| [03](Lecture-03.md) | **ML Models Recap — Trees, Linear, Neural Nets** — the practitioner's working set: when each model is the right call | the model zoo |
| [04](Lecture-04.md) | **Model Validation & Evaluation** — the metrics that match the problem, under/overfitting, and validation that doesn't lie to you | trusting the score |
| [05](Lecture-05.md) | **Model Combination — Bagging, Boosting, Stacking** — bias–variance, and the three ways to combine models that win Kaggle and production alike | ensembling |
| [06](Lecture-06.md) | **Distribution Shift — Covariate & Label Shift** — why production accuracy decays, detection via two-sample tests, and importance-weighting correction | the world drifts |
| [07](Lecture-07.md) | **Data Beyond IID — Sequences & Graphs** — independence tests, sequence models, and graph/GNN structure when samples aren't independent | structured data |
| [08](Lecture-08.md) | **Model & Hyperparameter Tuning — HPO, NAS, Deep-Net Tuning** — search algorithms, the rise and fall of NAS, and the norm/residual/attention toolkit | tuning |
| [09](Lecture-09.md) | **Transfer Learning — CV, NLP, Prompting** — fine-tuning as the default workflow, and the line from feature-extraction to prompt-based learning | reuse, don't restart |
| [10](Lecture-10.md) | **Model Compression — Pruning, Quantization, Distillation** — the hardware-facing lecture: shrink a model to a latency/memory/energy budget | shipping it small |
| [11](Lecture-11.md) | **Multimodal, Fairness & Explainability** — fusing modalities, fairness criteria and their impossibility, and making a model explain itself | responsible deployment |

</div>

---

## Course Outcomes

By the end you should be able to:

* Stand up a **data pipeline** from acquisition through cleaning, transformation, and feature engineering, and name the failure mode each stage guards against.
* Pick an **evaluation metric and validation scheme** that match the problem (imbalanced classes, time-ordered data, small data) instead of defaulting to accuracy + a random split.
* **Detect and correct distribution shift** — distinguish covariate shift from label shift, run a two-sample test to know it's happening, and apply importance weighting to fix it.
* Run **hyperparameter search and deep-network tuning** deliberately, and explain why NAS faded while transfer learning won.
* **Compress a model** with pruning, quantization, and distillation to hit a target latency/memory/energy budget, and report the accuracy tradeoff with numbers — the skill that connects this course to the rest of the roadmap.
* Reason about **fairness criteria** (and why you can't satisfy all of them at once) and produce a basic **explanation** of a model's prediction.

---

## Attribution & Currency

This course is an **adaptation, not a copy.** Source material — lecture structure, examples, and the slide content each lecture is grounded in — is **Stanford CS329P (2021 Fall)** by Qingqing Huang, Mu Li, and Alex Smola, released under **CC-BY-SA-4.0** (the slides) and **MIT-0** (the notebooks). This adaptation is shared under the same **CC-BY-SA-4.0** terms.

* Original course: **<https://c.d2l.ai/stanford-cs329p>** · companion textbook: **[D2L](https://d2l.ai/)**.
* Each lecture closes with a **`## Current as of`** note marking what was refreshed for 2026 versus what is taught as the original CS329P material — most heavily in **Lecture 08 (NAS)**, **Lecture 09 (prompting → instruction tuning)**, and **Lecture 10 (LLM-era quantization/distillation)**.
* Where 2021 framing is now dated, the original is taught first (it is still the right mental model), then the update is flagged explicitly. We do not silently rewrite history.

---

## Exit Criteria

You are done with this course when you can take **one dataset and one model end to end**:

* Acquire it, clean it, engineer features, and justify each transformation.
* Validate it with a scheme that survives scrutiny — and state the metric *before* you look at the score.
* Probe it for distribution shift and show whether train and production are the same distribution.
* Tune it, then **compress it to a deployment budget** and report tokens/s (or latency) and accuracy at each rung.
* Say one honest sentence about its fairness and one about why it made a given prediction.

If you can fit a model but can't do the ten things around the fit, you have a notebook. The point of this course is the other ten things.

---

*Related: [Module 4B — ML Engineering & MLOps](../Guide.md) · [Phase 5 — ML Systems Engineering](../../../../Phase%205%20-%20Advanced%20Topics%20and%20Specialization/Track%20G%20-%20ML%20Systems%20Engineering/Guide.md) · original [Stanford CS329P](https://c.d2l.ai/stanford-cs329p)*
