# Lecture 11 - Multimodal, Fairness & Explainability

**Collection:** [Practical Machine Learning (CS329P)](README.md) | **Previous:** [← Lecture 10](Lecture-10.md) | **Next:** [Course index](README.md)

---

Every earlier lecture assumed a tidy interface: a tensor goes in, a number comes out, you optimize the number. The last mile of practical ML is where that assumption breaks against reality. Real data does not arrive as one clean modality — a house listing is tables *and* photos *and* free text, a self-driving car fuses camera, lidar, and radar, an Amazon product page is images plus reviews plus a category graph. A real deployment does not act in a vacuum either: its outputs decide who gets a loan, who gets bail, who sees a job ad — and the same model that posts a great AUC can be quietly, systematically unfair to a protected group. And a real stakeholder — a regulator, a denied applicant, a debugging engineer — does not accept "the network said so"; they demand to know *why*.

This lecture consolidates the three CS329P topics that govern **responsible deployment**: multimodal data (consuming the messy real world), fairness (treating people equitably), and explainability (being able to justify a decision). They share a thesis. Each is the part of the pipeline that, if skipped, does not show up as a bad loss curve — it shows up later as a model that misses half its signal, gets your product banned, or gets your company sued. Accuracy is necessary; it is not sufficient. The work below is what stands between a model that scores well offline and a model that survives contact with the world and its laws.

A second thread runs underneath all three: **common sense and understanding the problem**. Fusion strategy, fairness criterion, and feature attribution are all technical machinery, but every one of them encodes a judgment call — what counts as "similar," what counts as "fair," what counts as "the reason." The machinery does not make those calls for you. It only makes them explicit enough to argue about.

---

## Learning objectives

By the end of this lecture you should be able to:

1. **Define multimodal data** and choose between early, late, and intermediate fusion for a given mix of tables, text, and images.
2. **Explain contrastive image-text pretraining** (CLIP-style) and why text supervision yields features that transfer better than ImageNet labels.
3. **Name the legal frameworks** behind fairness (protected attributes; disparate treatment vs. disparate impact) and connect them to real-world harms.
4. **State the formal fairness criteria** — demographic parity, equalized odds, calibration/predictive parity — and explain the **impossibility result** that they cannot all hold at once.
5. **Distinguish explanation strategies** along two axes (intrinsic vs. post-hoc, global vs. local) and recognize spurious-correlation ("Clever Hans") failures.
6. **Compare attribution methods** — axiomatic (SHAP, Integrated Gradients) vs. heuristic (LIME, saliency, Grad-CAM) — and judge when each can and cannot be trusted.

---

## 1. Multimodal data

### 1.1 What "multimodal" means

Data is naturally **multimodal** in industry applications — the raw record contains tables, text, images, audio, graphs, often all at once. CS329P's running examples make the point concretely:

- **House sales** — tabular attributes (beds, baths, zip), free-text description, and listing photos.
- **Amazon product** — images + text + tabular, plus a category/co-purchase **graph**.
- **Self-driving cars** — camera images, lidar point clouds, radar, all timestamped and spatially aligned.

A unimodal model trained on only one of these channels is leaving signal on the table. The engineering question is how to combine modalities so the model sees more than any single channel offers.

### 1.2 Fusion strategies

The core problem of multimodal learning is **how to match different-modal data into the same semantic space**, then how to construct a loss over the combined representation. The strategies differ in *where* the modalities meet.

| Strategy | Where modalities combine | Mechanism | Best when |
|---|---|---|---|
| **Early fusion** | At the input | Concatenate raw features / shallow embeddings, feed one model | Modalities are tightly coupled; a single model can learn cross-modal interactions |
| **Intermediate fusion** | At a hidden layer | Each modality gets its own encoder; merge the mid-level representations, then jointly process | You want modality-specific feature extractors *and* learned interaction — the common deep-learning default |
| **Late fusion** | At the output | Train a separate model ("tower") per modality; combine their predictions (e.g. average, stack) | Modalities are loosely coupled; you want modular, independently-trainable components |

In CS329P's notation, **early fusion** pushes `[text, image]` through one deep network, while **late fusion** runs a `Tower A` and a `Tower B` separately and combines at the end. Intermediate fusion sits between: per-modality encoders feeding a shared head.

Which wins? The empirical answer from the lecture is **it's close, and late fusion has a slight edge** in their experiments. On a tabular + text benchmark (a shallow variant using an ELECTRA text encoder, Shi et al., NeurIPS '21), averaged over 13 datasets:

- **Early fusion: 0.662**
- **Late fusion: 0.667** (larger is better)

The gap is small. The practical lever is often **ensembling on top** rather than the fusion point itself — stacking the multimodal net with other base models pushed the same benchmark from 0.667 to **0.683**, beating single-strategy baselines like AutoGluon's n-gram text path (0.659) or an H2O Word2vec path (0.600).

### 1.3 Aligning modalities and constructing the loss

Two ways to construct the training objective over fused modalities:

- **Joint label learning** — combine the modalities and train to predict a shared label end-to-end (supervised).
- **Contrastive learning** — learn an embedding space in which **similar sample pairs stay close and dissimilar ones are pushed far apart**. This is self-supervised: the "label" is just whether two things belong together. It is the engine behind image-from-text supervision below.

### 1.4 Image + text: contrastive pretraining (CLIP-style)

The headline multimodal result of the era is learning **image representations from text supervision**. **CLIP** (Radford et al., ICML '21) pairs an image encoder (ResNet or ViT) with a text encoder (a GPT-style transformer) and trains on **300M (image, text) pairs** scraped from the web with a contrastive objective: the embedding of an image should be close to the embedding of its caption and far from every other caption in the batch.

The payoff is that the resulting features are **comparable to or better than features trained on ImageNet**, despite never seeing a single hand-drawn class label. Natural-language supervision is richer than a fixed label set — "a photo of a corgi on a skateboard" carries more structure than the class index `263` — and it scales to whatever text the web already attaches to images.

The same contrastive recipe generalizes across modality pairs. **VideoBERT** (Sun et al., ICCV '19) aligns **video + audio**: it takes text from automatic speech recognition (ASR) on the audio track and pairs it with video frames, trained on ~23K hours of cooking/recipe YouTube videos. Text-from-speech becomes free supervision for video understanding.

### 1.5 Challenges

- **Alignment** — getting two modalities into the *same* semantic space is the hard part. A pixel and a token live in totally different spaces; the encoders must learn a shared geometry where "dog" the word and a dog photo land near each other. Misalignment quietly degrades everything downstream.
- **Missing modalities** — at inference a record may lack a photo, a review, or a sensor reading. A model that assumed all channels are present can fail badly; robust multimodal systems must degrade gracefully when a modality drops out (late fusion helps here — a missing tower can be skipped).
- **Scale and noise** — web-scraped pairs are abundant but noisy; the contrastive objective tolerates noise far better than a scheme that demands clean labels, which is precisely why it scaled.

> **The summary CS329P lands on:** real data is often multimodal; project each modality into a common space via early or late fusion; then either jointly learn labels or use contrastive learning for self-supervised training.

---

## 2. Fairness

### 2.1 Real-world harm

Fairness is not abstract — biased models have harmed people, sometimes for a century. CS329P opens with cases:

- **Simpson's paradox (UC Berkeley admissions, 1973)** — aggregate admission rates looked biased against women, yet *per department* the bias reversed or vanished. Women applied disproportionately to more competitive departments. The aggregate lies; you must condition on the right variable.
- **COMPAS (ProPublica, 2016)** — a recidivism risk tool assigned systematically different risk-score distributions to Black vs. White defendants, producing **higher false-positive rates for Black defendants** (31% vs. 15% of those who did *not* reoffend were flagged high-risk in Broward County).
- **Bias in online advertising** (Lambrecht & Tucker) — an ostensibly neutral ad-delivery system showed a STEM job ad to men more than women, for economic reasons unrelated to intent.
- **Bias in lending** (Martinez & Kirchner, 2021) — modern mortgage-approval algorithms showed disparities by race.
- **Redlining** — location-based credit denial in the 1930s discriminated against minorities, and its effects are **still measurable 90 years later**. Bias baked into a system propagates across generations.

The lesson the lecture states bluntly: **bias and lack of fairness can harm people for a century — we owe it to everyone to be mindful.**

### 2.2 Legal frameworks

Fairness has a legal substrate that constrains what a deployed model is allowed to do. **Protected attributes** are the characteristics the law forbids discriminating on. US federal law:

- **Title VII of the Civil Rights Act (1964)** — bars discrimination on **race, color, religion, national origin, or sex** (and protects against retaliation for raising a claim).
- **Pregnancy Discrimination Act** — extends "sex" to pregnancy and childbirth.
- **Equal Pay Act (1963)** — bars sex-based wage discrimination for equal work.
- **Age Discrimination in Employment Act (1967)** — protects workers **40 and older**.
- **Americans with Disabilities Act, Title I (1990)** — bars discrimination against a qualified person with a disability.

Europe's **GDPR** adds data-protection and a right to explanation around automated decisions. And regulators are moving toward a **legal requirement to show that hiring algorithms are unbiased** — which immediately raises the question the rest of this section answers: *what does "unbiased" actually mean?*

Two legal concepts every ML engineer should hold:

- **Disparate treatment** — *intentionally* using a protected attribute (or treating people differently because of it). Roughly the **anti-classification** idea: don't put `race` in the model.
- **Disparate impact** — a facially neutral policy that nonetheless produces *unequal outcomes* across groups, regardless of intent. This is the dangerous one for ML: you can trigger disparate impact with a model that never sees the protected attribute, because **secondary attributes** (zip code, name, dreadlocks vs. an Oxford shirt, "pony vs. motorbike") proxy for it.

### 2.3 Risk distributions across groups

Corbett-Davies & Goel's framing (ICML 2019 tutorial) is the cleanest mental model. For a decision like "search this vehicle for contraband?", each group has a **risk distribution** — the spread of `p(contraband | x)` over its members, from 0 to 1. Two facts about these distributions drive everything:

- **Different average risk** = different base rates of actually carrying contraband.
- **Different variance** = different *ability to tell who* is carrying — lower variance means harder to separate the guilty from the innocent within that group.

This exposes the trap in the naive **"outcome test"** (judge fairness by hit rates). Consider two groups that both carry contraband 30% of the time, under a facially neutral policy: *search if probability > 50%*. The group with **higher-variance** risk has more "obviously guilty" members above the threshold, so its **hit rate comes out higher** — and the outcome test wrongly cries bias against the other group. Conversely, a policy that genuinely discriminates (searching one group at a lower threshold, 45% vs. 50%) can be tuned so the **hit rates come out equal**, and the outcome test wrongly finds *no* discrimination. Equal hit rates are neither necessary nor sufficient for fairness. You have to reason about the distributions, not the surface statistic.

### 2.4 Formal fairness criteria

To evaluate a classifier `f : 𝒳 → ℝ` (where larger `f(x)` means `y = 1` is more likely, though the raw scores need not be calibrated), we build on standard diagnostics — the confusion matrix (TP/FP/FN/TN), the **ROC curve** (true-positive rate vs. false-positive rate as the threshold `z` varies), and the **precision-recall curve**. The fairness question is: **match which of these quality scores across groups** (African American, Asian, Latino, White; men, women; …)? The candidates:

| Criterion | Plain-English definition | Formal condition | Also called |
|---|---|---|---|
| **Demographic / statistical parity** | Same *rate of positive predictions* across groups, regardless of truth | `TP + FP = const` per group → cutpoints vary between groups | Group fairness |
| **Equalized odds / classification parity** | Same **error rates** across groups (e.g. equal false-positive rate, equal true-positive rate) | `FPR_a = FPR_b` and `TPR_a = TPR_b` | Classification parity |
| **Predictive parity / calibration** | Given the same risk score, the **outcome is independent of group** — a score of 0.7 means 70% for everyone | `p(y = 1 | score, group)` independent of group | Calibration within groups |
| **Anti-classification** | The protected attribute is **not used** by the algorithm | `race ∉ features` | Fairness through unawareness |
| **Conditional demographic disparity (CDD)** | Whether a group gets a smaller share of positive vs. negative outcomes *relative to its demographics*, partitioned to avoid Simpson's paradox | `CDD = Σ_c p(c)·[ p(ŷ=1∣c)/p(ŷ=1) − p(ŷ=−1∣c)/p(ŷ=−1) ]` | — |

Two cautions the lecture stresses. **Anti-classification can make outcomes worse** — dropping the protected attribute lets the model discriminate via proxies *and* removes your ability to measure or correct for it; it also costs accuracy. And **calibration can be "hacked"**: by choosing features that simply aren't predictive for one group, you can shift that group's scores below the decision threshold while the scores *remain technically calibrated* — e.g. detain defendants with risk > 0.5, but engineer features so no blue-group defendant ever crosses 0.5, leaving the average reoffending rate unchanged. Satisfying a criterion on paper is not the same as being fair.

### 2.5 The impossibility result

Here is the result that makes fairness genuinely hard rather than merely fiddly. **Kleinberg, Mullainathan & Raghavan (2016)** and **Chouldechova (2017)** independently proved:

> It is **impossible** to simultaneously satisfy (1) calibration within groups, (2) balance for the positive class, and (3) balance for the negative class — **unless** the classifier is perfect, *or* the score distributions are identical across groups.

In plain terms: you generally **cannot have demographic parity, equalized odds, and calibration all at once.** Improving one criterion provably degrades another — a tension noted as far back as Darlington (1971) and surveyed in Hutchinson & Mitchell's "50 Years of Test (Un)fairness" (2018).

CS329P frames the deep reason as a **"Pokémon" theorem**. Let `p` and `q` be the distributions on `𝒳 × {0,1}` for two protected groups. Suppose you've matched a finite set of statistics, `sᵢ[p] = sᵢ[q]` for `i = 1…n`. If `p ≠ q`, there *always* exists another statistic `s′` with `s′[p] ≠ s′[q]`. The proof is Maximum Mean Discrepancy: two distributions are identical only if **all** expectations (over an infinite class of test functions) match — a finite number of matched statistics can never be enough, so there is always one fairness criterion you fail. **Regardless of how many criteria you satisfy, another one breaks.** (Related: Simoiu, Corbett-Davies & Goel 2017, infra-marginality.) The root cause is geometric — different distributions have different ROC/PR curves, and cutting different curves at thresholds produces different outcomes in general.

The takeaway is not nihilism. It is that **"fair" is not a single optimizable target.** You must choose *which* criterion matters for *your* problem and defend that choice — because you cannot get them all.

### 2.6 Fairness in practice

So what do you actually do? The lecture's guidance is pointed:

- **Use (multiple) fairness measures as *indicators*, not training targets.** Compute several; if you find large discrepancies between groups, **debug the model** — they are excellent at *spotting* problems. But **do not just bolt a fairness criterion onto the loss and hope** — naively optimizing one criterion can *increase* discrimination elsewhere, dropping the protected attribute reduces accuracy, and crude "affirmative action" in the objective can backfire (e.g. reduce diversity via stereotyping; Lipton 2019 on admissions).

Mitigations, when warranted, attach at three stages of the pipeline:

| Stage | What it does | Examples / risks |
|---|---|---|
| **Pre-processing** | Fix the **data** before training | Reweight/resample under-represented groups; remove biased features. Check for data-collection bias: population skew (too many white actors in CelebFace), cultural stereotypes in text ("female nurse / male doctor"), temporal bias (a social network's early user base) |
| **In-processing** | Constrain the **training** objective | Add a fairness constraint/penalty. Powerful but dangerous — can reduce accuracy or shift discrimination; never deploy blind |
| **Post-processing** | Adjust the **outputs / thresholds** | Use group-specific cutpoints to equalize a chosen criterion. Note: per-group thresholds may itself be disparate treatment — a legal question, not just a technical one |

And the non-algorithmic safeguards the lecture insists on: a **diverse team** catches more issues; gather **stakeholder feedback**; ask **where the data came from**; look for problems *proactively* ("if things look strange, they probably are"); and **keep testing after deployment** — external red-teamers will find your weaknesses (Microsoft's *Tay* turned racist within hours; *AI Dungeon* on GPT-2 generated abusive content; image classifiers have mislabeled humans). Finally, encode **asymmetric risk** into decisions rather than naively taking the most-likely label: a box of mushrooms 99% safe to eat is not worth eating, because `R[poison | edible]` ≫ `R[edible | poison]`. Use the risk matrix, not the MLE.

> **The fairness summary:** examples, law, algorithmic criteria, impossibility, practice — but above all, **use your common sense and try to understand the problem.** The math constrains; it does not decide.

---

## 3. Explainability

### 3.1 Why explain

The motivating story: you arrive in the US, land an engineer job, apply for a credit card — **denied** for "bad credit history" you don't understand. The model used weird features (the *age* of your credit line keeps your card alive), you have to actively game the score, and once set it's easy to maintain — but only if someone tells you *what it keys on*. That is the user-facing case for explainability. The full set of reasons:

- **Trust** — users and stakeholders accept a decision they can understand.
- **Debugging** — explanations reveal when a model is right for the *wrong reason* (see backdoors below).
- **Compliance** — GDPR's right to explanation, and emerging audit requirements, can make explanation a *legal* obligation.

Formally, given data `X` and a trained model `f`, three distinct explanation questions:

1. **Global** — explain what `f` does *in general*.
2. **Feature attribution** — explain *which/how* features `xᵢ` affect the output `f(x)`.
3. **Local** — explain how `f(x)` behaves *near a specific point* (why is *this* applicant's rating poor?).

### 3.2 Strategies — two axes

Explanation methods organize along two axes. **Intrinsic vs. post-hoc**: is the model interpretable *by construction*, or do you explain a black box after the fact? **Global vs. local**: explain the whole model, or one prediction?

| Family | Axis | Idea | Cost |
|---|---|---|---|
| **Simplicity** (intrinsic, global) | Use an inherently interpretable model | Linear/logistic (weights `wᵢ` = importance, optionally with `ℓ₁` sparsity), small decision **trees**, decision **lists** (if-then-elsif). The "Rule of Nines" for burn triage is the ideal: simple enough to use under stress | Limited capacity; linearity fails on high-dim image/text/time-series; "too tedious to operationalize" |
| **Approximate simplicity** (post-hoc, global) | Train a complex model, then **distill** it into a simple one | Fit `g` to match `f`'s predictions: `minimize_g Σ l(g(xᵢ), f(xᵢ))`. Generate auxiliary data to distill on if the training set is small (Fakoor et al., 2020) | The simple surrogate is only as faithful as the distillation |
| **Local simplicity** (post-hoc, local) | Approximate the black box **near one query** | LIME — a "Taylor expansion by regression": sample points `xⱼ` around `x`, fit a local linear `g` to `(xⱼ, f(xⱼ))` (Ribeiro et al., 2016) | Faithful only locally; unstable (see §3.5) |

The intuition for local methods: a black box may be hopelessly complex *globally* yet **linearizable in a small neighborhood**. You can't approximate the whole surface, but you can fit a simple model to the patch around the point you care about.

### 3.3 Conditioning and backdoors — the Clever Hans problem

The subtle, important part. To attribute influence, you measure how `f` changes when you change a feature, `Δx`, relative to a **reference value `x₀`** (means are a reasonable default for images, text, tabular, audio). But changing `xᵢ` is confounded: `xᵢ` may be correlated with the *other* features `x₋ᵢ`, so naive perturbation conflates **direct** influence (what you want) with **indirect** influence through the correlations.

This is the **backdoor problem**, drawn as a causal graph. A latent variable `z` (say, true creditworthiness) generates two observed features `x₁, x₂`. The observed `x₁, x₂` are dependent — `p(x₁, x₂) ≠ p(x₁)p(x₂)` — *through* `z`. If you explain by perturbing observed attributes while letting them stay correlated, you measure the backdoor path, not the causal effect. **Changing an observed attribute does not change the underlying condition.** The fix, following Pearl's **`do`-operator**, is to draw the left-out features from their **marginal** `p(x₋ᵢ)` rather than the conditional `p(x₋ᵢ | xᵢ)` — sampling from marginals breaks the spurious dependence and is both easier and usually *more correct*:

```text
E[ y | do(X₁ = x₁) ] = ∫ p(x₂, x₃) · E[ y | x₁, x₂, x₃ ] dx₂ dx₃     # average over marginals, not conditionals
```

The practical face of this is the **Clever Hans** effect — a model that appears to solve the task but actually keys on a **spurious correlation** in the data. (Clever Hans was a horse that "did arithmetic" by reading its trainer's body language.) The canonical ML versions: a husky-vs-wolf classifier that really detects **snow in the background**; a pneumonia model that keys on the **hospital's scanner ID**; a tank detector that learned **time of day**. The model is right on the test set and catastrophically wrong in deployment, because the backdoor feature it leaned on doesn't transfer. **Explainability is how you catch this** — a good attribution method points at the snow, and you realize the model never learned the animal at all.

### 3.4 Axiomatic approaches — SHAP and Integrated Gradients

The principled response: don't invent a heuristic and hope it's reasonable — write down the **axioms** an attribution *should* satisfy and derive the unique method that satisfies them.

**Shapley values** come from cooperative game theory. CS329P's parable: the Parliament of Micronesia, parties A (45 seats), B, C, D (15–25 each), must reach 51 votes to pass a \$1M bill. A coalition's "payoff" `v(S)` is whether it wins. The fair way to split credit among players satisfies three axioms — **symmetry** (interchangeable players get equal credit), **dummy player** (a player who adds nothing gets their stand-alone value), and **additivity** (credit over a sum of games is the sum of credits) — and the **Shapley value theorem** says these axioms pin down a *unique* allocation: the average marginal contribution of player `i` over all orderings,

```text
ϕ(i, N) = Σ_{S ⊆ N∖{i}}  [ |S|! (|N|−|S|−1)! / |N|! ] · [ v(S ∪ {i}) − v(S) ]
```

Replace **parties with features** and **payoff with model output**, and you get **SHAP** (Lundberg & Lee, 2017). The SHAP theorem mirrors the game-theory one: the *only* feature-attribution score satisfying **local accuracy** (the explanation matches `f` at the point), **missingness** (a missing feature gets zero attribution, `ϕᵢ = 0`), and **consistency** (if a feature's marginal contribution grows under a new model, its attribution doesn't shrink) is the Shapley value. It recovers the exact weights for linear models and the local weights for LIME as special cases — a unifying result. The **devil is in the details**, though: what does "leaving a feature out" *mean* — set it to zero, or to its mean? (The slide's guidance: don't try to model the conditional distribution; draw **unrelated values** from the marginal — Janzing et al., 2020 — which works for tabular but is trickier for text/images where context carries meaning.) And the sum is `O(2^|N|)` over feature subsets, so it needs approximation: **TreeSHAP**, **DeepSHAP**, **KernelSHAP**, and Shapley **sampling** all trade exactness for tractability.

**Integrated Gradients** (Sundararajan et al.) is the gradient-based axiomatic method. Its axioms — **completeness** (attributions sum to `f(x) − f(x₀)`), **sensitivity** (a feature `f` doesn't depend on gets zero), **implementation invariance** (the score doesn't depend on *how* `f` is coded), **linearity**, and **symmetry** — uniquely determine the integral of the gradient along a straight path from a baseline `x′` to the input `x`:

```text
ϕ(i, x) = (xᵢ − x′ᵢ) · ∫₀¹ ∂_{xᵢ} f( x′ + α(x − x′) ) dα
```

Integrating *along the path* (rather than reading the gradient at a single point) is what gives it completeness and cures the saturation problem that plagues plain gradients — and there's a useful connection: where IG is computable, it's a cheaper route to Shapley-style attributions.

### 3.5 Heuristic approaches — and why to distrust them

The cheap, popular methods — and the warning that comes with them. **Sensitivity analysis / saliency** is just the local gradient `s_f(x) = ∂_x f(x)`, trivially computed by backprop. The problem: **ReLU and other clipping operations zero out gradients**, so saliency "misses out on relevant changes" and **leads to weird, noisy results**. The first patch is **grad × input** (`Δx · ∂_x f(x)`, Bach et al., 2015); a better one is **DeepLIFT**, which replaces derivatives with **finite differences** against a reference, `[f(x′ + Δxᵢ) − f(x′)] / Δxᵢ · Δxᵢ`, with special decomposition rules for ReLU, general activations, and linear ops, all computable by a modified backprop. Other members of the family: **guided backprop**, **Grad-CAM** (gradient-weighted class activation maps that highlight image regions), **KernelSHAP**, and LIME from §3.2.

The honest assessment CS329P delivers: these heuristics are **unreliable**. Raw gradients are noisy and broken by clipping. Applying them to **text and images** is harder still — you must identify *larger components* (you can't drop random pixels or characters and expect meaning), and there's no obvious **reference `x₀`** (what is the "neutral" text?). And the deepest gap: all of them tell you *what* the model attended to, but **what we actually want is causality — why.** A saliency map over the snow tells you the pixels mattered; it does not, on its own, tell you the model failed to learn the wolf. Use heuristics as fast first looks; reach for axiomatic methods (and the marginal-sampling discipline of §3.3) when an attribution has to hold up.

| Method | Family | Local/Global | Reliability |
|---|---|---|---|
| **Linear weights / trees / lists** | Intrinsic simplicity | Global | High — but only if the model is genuinely simple |
| **Distillation** | Approximate simplicity | Global | As faithful as the surrogate |
| **LIME** | Local surrogate | Local | Useful but **unstable** across reruns |
| **SHAP** | Axiomatic (Shapley) | Local (+ global agg.) | Strong axioms; cost/reference choices matter |
| **Integrated Gradients** | Axiomatic (gradient) | Local | Strong axioms; needs a baseline `x₀` |
| **Saliency / grad×input** | Heuristic gradient | Local | **Noisy**, broken by ReLU clipping |
| **Grad-CAM / guided backprop** | Heuristic gradient | Local | Popular for images; can be misleading |

> **The explainability summary:** simplicity, approximate simplicity, local simplicity; conditioning and backdoors; axiomatic methods (SHAP, Integrated Gradients); heuristics. Prefer axioms over heuristics, sample from marginals to dodge backdoors, and never forget the goal is *why*, not just *what*.

---

> **2026 update:** Five shifts since the 2021 slides. **(1) Multimodal went mainstream and generative.** CLIP was the precursor; the era since is dominated by **multimodal LLMs** — **GPT-4V/4o**, **Gemini**, **Claude with vision**, and the open **LLaVA** family — that ingest interleaved image + text (and increasingly audio/video) and *generate*, not just classify. The fusion question is alive and well: most VLMs are **intermediate fusion** (a frozen vision encoder → a projection → the LLM's token stream), the direct descendant of §1.2. **(2) Fairness moved into LLMs.** The harms are the same (stereotypes in text — "female nurse / male doctor"), now measured by **bias evals** (BBQ, BOLD, WinoBias, HELM's fairness suite) and **red-teaming**, and the COMPAS-style impossibility result still binds — you still cannot satisfy every criterion at once. **(3) Mechanistic interpretability** emerged as a new branch beyond the attribution methods here — **sparse autoencoders**, circuit analysis, and feature steering try to explain *what concepts a network represents internally* rather than which inputs mattered, a more ambitious "why." **(4) Regulation got teeth.** The **EU AI Act** (in force 2024, phasing in through 2026–27) classifies hiring, credit, and biometric systems as **high-risk**, mandating documentation, bias assessment, and human oversight — turning the §2.2 "potential legal requirement to show hiring algorithms are unbiased" into binding law with real penalties. The US **NIST AI Risk Management Framework** plays a parallel voluntary role. **(5) Attribution tooling matured** — SHAP and Captum (Integrated Gradients) are standard, and the field's own caution about heuristic-saliency unreliability (Adebayo et al.'s "sanity checks") is now received wisdom. The lecture's core warnings aged well: fusion is still a judgment call, fairness still cannot be reduced to one number, and explanations still must aim at *why*.

> **Hardware lens:** Each topic has a deployment-compute story. **Multimodal inference** is heavier than text alone — a VLM runs a vision encoder *and* an LLM, and the image tokens (often hundreds to thousands per image at high resolution) inflate the **KV cache** and prefill cost; serving them economically drives the same batching, quantization, and caching work as any LLM, plus encoder reuse across a conversation. **Explainability is expensive at scale** — exact SHAP is `O(2^|N|)`, and even Integrated Gradients needs tens of forward/backward passes *per explained instance* along the path integral, so attribution for a high-traffic model is a real compute line item, which is why TreeSHAP/DeepSHAP approximations and batched IG exist. **Fairness auditing is continuous compute** — the lecture's "keep testing after deployment" means standing up monitoring that slices metrics by group on live traffic, an ongoing cost, not a one-time check. The throughput lesson from Lecture 02 recurs: responsible-deployment machinery (multimodal encoders, per-instance attributions, sliced fairness monitors) competes with the model for accelerator cycles, and budgeting for it is part of shipping a system that is fair, explainable, *and* affordable.

---

## Current as of

Written June 2026. The **original CS329P content** — multimodal fusion (early/late/intermediate, the CLIP and VideoBERT examples, the 0.662 vs. 0.667 benchmark), the fairness arc (harm examples, US/EU law, risk distributions and the outcome-test trap, the formal criteria, the Kleinberg–Chouldechova / "Pokémon" impossibility result, and the pre/in/post-processing practice notes), and the explainability arc (intrinsic vs. post-hoc and global vs. local strategies, the backdoor/Clever-Hans problem and the `do`-operator fix, the SHAP and Integrated Gradients axioms, and the heuristic saliency methods with their unreliability) — is taught first because it remains the correct working mental model and maps one-to-one onto the 2021 slides. The **refresh layer** flags what moved since: multimodal LLMs (GPT-4V/Gemini/LLaVA) as the dominant fusion application, LLM bias evals and red-teaming, **mechanistic interpretability** as a new explanation branch, and the **EU AI Act** turning fairness from best practice into binding high-risk-system law. Where 2021 framing is dated, the original is presented before the update; nothing is silently rewritten.

*Adapted from [Stanford CS329P](https://c.d2l.ai/stanford-cs329p) — Huang, Li & Smola, CC-BY-SA-4.0.*
