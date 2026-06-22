# Lecture 06 - Distribution Shift: Covariate & Label Shift

**Collection:** [Practical Machine Learning (CS329P)](README.md) | **Previous:** [← Lecture 05](Lecture-05.md) | **Next:** [Lecture 07](Lecture-07.md)

---

Every supervised model you have ever trained rests on one assumption so quiet you probably never wrote it down: that the data you train on and the data you serve on are drawn from the *same* distribution `p(x, y)`. This is the IID assumption — independent and identically distributed — and it is the silent load-bearing wall of machine learning. The whole edifice of generalization theory, the reason a good validation score is supposed to mean anything at all, is built on it. And production quietly breaks it. Not loudly, not with an exception in the logs — quietly. The model that scored 94% in your notebook ships, runs fine for a quarter, and then accuracy drifts down to 88% with no code change, no bug, no bad deploy. The model did not rot. The world moved.

This is **distribution shift**, and it is the single most common reason deployed models decay. The empirical distribution lies — your training set was only ever a finite sample of a world that keeps changing underneath it. A vending-machine vision model demoed at CES fails on the show floor because the lighting is different from the lab (a true story from CES 2019: the fix was new data, a tablecloth to kill the table reflection, and an all-night retrain). A speech model trained on West-Coast accents stumbles on a Texan drawl. A medical classifier trained when COVID was rare gets deployed into a town right after a super-spreader event where it is suddenly common. None of these is a modeling failure in the usual sense. The function `f` is fine; the input it is now being asked about comes from a distribution it never saw.

This lecture builds the taxonomy and the math to handle it: how to **name** the kind of shift you have (covariate, label, or concept), how to **correct** for the two tractable kinds by reweighting your training data, how to **detect** that a shift happened at all using two-sample tests, and how the worst case of shift — adversarial data — connects to robustness. The throughline, in CS329P's framing: *training ≠ testing*, and the discipline of an ML engineer is knowing precisely *how* they differ and what you are allowed to do about it.

---

## Learning objectives

By the end of this lecture you should be able to:

1. **State the IID assumption** and explain why a strong validation score is *necessary but not sufficient* for good test-time performance.
2. **Classify a shift** as covariate shift, label shift, or concept drift from how the joint `p(x, y)` factorizes, and give a concrete production example of each.
3. **Correct covariate shift** by importance weighting, estimating the density ratio with a domain classifier, and stabilizing it by clipping weights / monitoring effective sample size.
4. **Connect adversarial examples to distribution shift** as the worst-case perturbation, and explain invariances and adversarially-robust loss as defenses.
5. **Detect a shift** with a two-sample test — classifier test, Maximum Mean Discrepancy (MMD), and the Kolmogorov–Smirnov statistic — and know which to reach for.
6. **Correct label shift** with Black-Box Shift Estimation (BBSE): recover `q(y)` from a confusion matrix and the predicted-label distribution, then reweight.

---

## 1. Generalization recap — why a good val score is not enough

Start from what training actually does. You have a data distribution `p(x, y)`, you draw a finite dataset from it, and training minimizes the **empirical risk** plus regularization:

```text
Empirical (training) risk:
  R_emp[f, X, Y] = (1/m) Σ_{i=1..m} l( f(x_i, w), y_i )

What you actually care about — expected risk over the true distribution:
  R[f, p] = E_{(x,y)~p} [ l( f(x, w), y ) ]
```

Training drives down the first. Deployment is judged by the second — the expected loss over *all the other data you could have seen*, not the handful you did. The gap between them is the entire subject of generalization, and the reason the gap is ever bounded is the IID assumption: train and test are the same `p`.

How do we get any guarantee that low empirical risk implies low expected risk? Hold out a **validation set** that was never used for training, and apply a concentration bound. For a loss bounded in `[0, 1]`, the Chernoff/Hoeffding bound says the empirical mean over `m` independent points is close to the true mean with high probability:

```text
Pr[ | (1/m) Σ l(f(x_i), y_i) − E[l(f(x), y)] | > ε ]  ≤  exp(−2 m ε²)

Solving for a confidence 1 − δ gives a sample-size rule of thumb:
  for δ = 0.05 and ε = 0.01  →  m ≈ 15,000 independent validation points.
```

Two caveats are exactly where production breaks. **First**, the bound assumes the validation set was *never* used for training — an assumption "often violated" the moment you tune hyperparameters against it, peek repeatedly, or leak preprocessing statistics across the split (the discipline from Lecture 02: fit normalizers on train only). **Second, and the whole point of this lecture: the bound assumes the validation points are drawn from the same distribution as test.** A pristine, untouched, perfectly-IID-with-training validation set still tells you *nothing* about a deployment whose distribution has moved.

So the takeaway is sharp. Good performance on the training set does not guarantee good test performance unless you regularize capacity (input noise, dropout, weight decay — all forms of smoothing `f`) *or* hold out an independent validation set for honest calibration. And even both together only certify performance **on the distribution the validation set came from.** A good val score is necessary — a model that can't even validate well is hopeless — but it is not sufficient. The rest of this lecture is about the "not sufficient" part.

---

## 2. A taxonomy of shift

The clean way to organize distribution shift is by *which factor of the joint distribution changed.* Write the training distribution as `p(x, y)` and the test distribution as `q(x, y)`. The joint always factors two ways — `p(x)·p(y|x)` or `p(y)·p(x|y)` — and each kind of shift holds one factor fixed while the other moves.

| Shift type | What changes | What stays fixed | Factorization | Concrete example |
|---|---|---|---|---|
| **Covariate shift** | `p(x) → q(x)` (the inputs) | `p(y\|x)` (label given input) | `q(x, y) = q(x)·p(y\|x)` | Train a face/object recognizer on studio lighting; deploy under fluorescent store lighting. The pixels shift; *what a cat looks like given its pixels* does not. (The CES vending machine.) |
| **Label shift** | `p(y) → q(y)` (class priors) | `p(x\|y)` (input given label) | `q(x, y) = q(y)·p(x\|y)` | A COVID classifier: train where the disease is rare, deploy in a town after a super-spreader event where `q(C19) ≫ p(C19)`. The *prevalence* shifts; the symptoms of a sick patient, `p(symptoms\|C19)`, do not. |
| **Concept drift** | `p(y\|x) → q(y\|x)` (the labeling rule) | `p(x)` (often) | the conditional itself moves | What counts as "fashionable", "spam", or a "fraudulent" transaction changes over time. The same input now maps to a different label. |

The distinction is not academic — **it dictates what fix is even possible.**

- **Covariate shift** (`p(x) ≠ q(x)`, label rule fixed) is correctable by reweighting training examples so the training input distribution looks like the test one. The relationship you learned, `p(y|x)`, is still valid; you just sampled the inputs from the wrong density. This is §3.
- **Label shift** (`p(y) ≠ q(y)`, `p(x|y)` fixed) is correctable by reweighting *by class*. The class-conditional appearance is still valid; only the mix of classes changed. This is §6, and it is in some ways the easier fix — *"when you have q(y), fixing models is easy."*
- **Concept drift** is the genuinely hard one. The dependency `p(y|x)` itself has changed, so nothing you learned about the old mapping transfers cleanly. CS329P is blunt: *"much bigger problem if concept shifts between training and test set — no real guarantees possible."* If it drifts slowly over time, you can sometimes track it by training a time-indexed model `p(y|x, t)`; if it shifts abruptly, you need fresh labels.

Two related cases the lecture flags as "things we didn't cover" but worth naming. **Covariate drift** is slow covariate shift over time (language usage, demographics, geographic preferences — Canada vs. USA search behavior); the strategy is to model the covariate density as a time-varying function. And **adversarial data** is its own pathological case — `supp(p) ≠ supp(q)`, where test data lands *off the support* of training data entirely — which §4 treats as the worst case.

A useful sanity rule: covariate shift = "the covariate distribution lies," label shift = "the label distribution lies," concept drift = "the relationship lies." When deciding which you face, ask *which causal direction generated the data.* If `x` causes `y` (image → label), shift in `p(x)` is covariate shift. If `y` causes `x` (disease → symptoms), shift in `p(y)` is label shift. The causal arrow tells you which factor is stable and therefore which fix applies.

---

## 3. Covariate shift correction — the math

Assume covariate shift: `p(y|x)` is unchanged, but `p(x) ≠ q(x)`. Your training risk integrates the loss against the *training* input density `p(x)`; what you actually want is the risk against the *test* density `q(x)`:

```text
What you minimized (training):   ∫ p(x) ∫ p(y|x) l(f(x,w), y) dy dx
What you want (test):            ∫ q(x) ∫ p(y|x) l(f(x,w), y) dy dx
```

The two differ only in the input density out front. The fix is a change of measure — multiply and divide by `p(x)`:

```text
∫ q(x) f(x) dx  =  ∫ p(x) · [ q(x)/p(x) ] · f(x) dx  =  ∫ p(x) · β(x) · f(x) dx
                                  └───────┘
                            importance weight β(x)
```

So if you reweight each training example by the **density ratio** `β(x) = q(x)/p(x)`, the *reweighted* training risk is an unbiased estimate of the test risk. Examples that are over-represented in training relative to test get down-weighted; examples that look like test get up-weighted. Concretely, the corrected objective becomes:

```text
Original:    minimize_w  (1/m) Σ_i           l( y_i, f(x_i, w) )
Reweighted:  minimize_w  (1/m) Σ_i  β(x_i) · l( y_i, f(x_i, w) )
```

**The catch:** you don't have `p` or `q` — you have *samples* from each (your training set and a batch of test/production inputs), and directly estimating two high-dimensional densities just to divide them is hard and unstable (what do you do where a density is near zero?). The elegant move is to never estimate the densities at all. **Estimate the ratio directly by training a classifier to tell training points from test points.**

Label every training point `+1` and every test point `−1`, pool them, and fit a probabilistic binary classifier `r(y | x)` (logistic regression is the canonical choice). At optimum its conditional class probability is:

```text
r(y = +1 | x) =  p(x) / ( p(x) + q(x) )

⇒  β(x) = q(x)/p(x) = r(y = −1 | x) / r(y = +1 | x)
```

That ratio of the classifier's two output probabilities *is* the importance weight — no density estimation required. The full procedure:

```text
COVARIATE SHIFT CORRECTION
  1. Build a pooled dataset: training points labeled +1, test/prod points labeled −1.
  2. Train a binary classifier (e.g. logistic regression) to separate the two.
  3. CHECK FIRST: if it can't beat chance, there is no covariate shift — stop, do nothing.
     (Reuse the generalization-performance estimate from §1 to decide.)
  4. For each training point x_i, set weight  β(x_i) = r(−1 | x_i) / r(+1 | x_i).
  5. Retrain your real model on the training data, reweighting example i by β(x_i).
```

Step 3 is doing double duty: the *same* classifier that produces the weights is itself a two-sample test (§5). If train and test are indistinguishable, the weights are all ≈ 1 and you have done no harm by skipping the correction.

There is a deeper unity here worth seeing. This reweighting-to-be-indistinguishable is **exactly the GAN objective**: a generator reweights training data `{(x_i, y_i)} → {β_i(x_i, y_i)}` so a discriminator can no longer tell it from test data, and the minimax optimum is reached precisely when `β(x) = q(x)/p(x)` and the distributions are no longer distinguishable (the discriminator is forced to entropy mode, `r = 0.5` everywhere). Covariate-shift correction, GANs, and a maximum-entropy / moment-matching view (matching feature means via MMD) are three faces of the same idea.

**Practical instability — the part that bites you.** The weights `β` are *estimates*, so they add bias and variance, and the real danger is when `p` and `q` are very different: a few enormous weights dominate and your effective sample size collapses. Make it quantitative. For a weighted sample mean `x̄ = Σ β_i x_i`, the variance scales as `‖β‖₂² σ²`, which motivates the **effective sample size**:

```text
m* = ‖β‖₁² / ‖β‖₂²        (equals m when all weights are equal; collapses when a few dominate)
```

If three points carry 90% of the weight mass, your "10,000-example" reweighted training set behaves like a handful of points — high variance, garbage. The standard remedy is to **clip the weights**, `β̄_i = min(β_i, C)`:

```text
Unclipped weights:   less bias, but HIGH variance when m* is small.
Clipped weights:     a little bias, but smaller variance (larger effective sample size).
```

Clipping trades a controlled bias for a large variance reduction — almost always worth it in practice. The discipline: compute `m*` before trusting a reweighting; if it has cratered relative to `m`, your covariate shift is too severe to paper over with importance weighting, and you need new data, not new weights.

---

## 4. Adversarial examples & invariants

Distribution shift has a worst case, and it is instructive. Instead of the world drifting by accident, imagine an adversary *choosing* the shift to maximize your loss. This is an **adversarial example**: take a correctly-classified input `x`, and find the smallest perturbation `δ` that flips the prediction.

```text
maximize_δ   l( f(x + δ), y )
subject to   ‖δ‖ ≤ ε        (perturbation imperceptibly small)
```

These exist and are easy to construct — adversarial images that dodge face recognition (even realized physically as 3D-printed glasses frames), audio perturbed inaudibly to transcribe as a different phrase. **Why does it work?** Because real training and "natural" test data live in a small, thin subset of input space, and the function's behavior is essentially *undefined away from where data occurred.* An adversarial point sits slightly off that support — `supp(p) ≠ supp(q)` — in a region the model was never constrained on, so the loss surface there can be pushed anywhere. This is also the abstract structure of an arms race you already know: **spam filtering.** The host retrains, the spammer finds a modification that evades, the host extends the dataset and retrains — a moving distribution driven by an adversary, forever.

The connection to the rest of the lecture: there is a theorem that *you can always find a distribution that makes things worse.* Given a loss with mean `R[p, f]` and variance `σ²[p, f]`, there exists a `q(x)` with `R[q, f] ≥ R[p, f] + σ` — just overweight the inputs where the model already errs (a mean-value-theorem argument: some region has above-average conditional loss; put `q`'s mass there). The lesson is defensive: **always confirm your train/test distributions actually match before drawing conclusions** — otherwise an apparent failure may just be an unlucky (or adversarial) reweighting.

The two defenses are duals of each other, and the difference is *what you know*:

- **Invariances** — transformations you *know* leave the label unchanged. A left-right flip of a cat is still a cat; a cropped, recolored, slightly-distorted image is the same class; speech with background noise added is the same words. So you **augment** the training set with these transforms (this is the data augmentation of Lecture 02, seen now through the shift lens): you are deliberately *widening the support* of `p` to cover variations you expect at test time. Classic roots: tangent distance (Simard 1995) and virtual support vectors (Schölkopf 1997) explored a point's neighborhood; today it's the standard ImageNet augmentation stack (random crop/scale/flip, hue–saturation–brightness jitter via libraries like imgaug / Albumentations).
- **Adversarial robustness** — transformations you do *not* know should preserve the label, and which in fact change the model's output. The defense is to treat them *as if* they were invariances: bake the worst case into the loss. An **adversarially-robust loss** takes the supremum over a family of transformations `Δ` (with a penalty `η(δ)` discouraging extreme distortions):

```text
L(x, y, f) = sup_{δ ∈ Δ}  η(δ) · l( f(x + δ), y )
```

At each training step you find the *worst* perturbation and train against it, so the model is forced to be correct not just on the data point but on a robust neighborhood around it. The clean summary: with **invariances** you *know* the transformation keeps the outcome unchanged and add it to be more robust; with **adversarial data** you *don't* know it should, observe that it changes the outcome, and defend by treating it as an invariance anyway.

---

## 5. Two-sample tests — how to KNOW a shift happened

Correction is moot if you don't know a shift occurred. The detection question is a classical statistics problem: given `X = {x₁, …, x_m}` drawn from `p` and `X' = {x'₁, …, x'_{m'}}` drawn from `q`, **test whether `p = q`.** Three tools, in rough order of "what to actually use."

**(1) Classifier two-sample test — the one to choose.** This is the same trick as §3, repurposed as a hypothesis test: *if you can train a classifier that tells the two samples apart with above-chance accuracy on held-out data, then `p ≠ q`.* The math underneath: the classifier objective `E_p[log π(+1|x)] + E_q[log π(−1|x)]` is minimized at `π(+1|x) = p(x)/(p(x)+q(x))`, and plugging that back in yields `2·H[(p+q)/2] − H[p] − H[q] + 2log 2`, which by convexity of entropy is minimized *exactly when `p = q`*. So inseparability is equivalent to equality of distributions. It's the recommended test because it works in high dimensions, uses tooling you already have, and the *trained classifier doubles as the importance-weight estimator* if the test comes back positive.

**(2) Maximum Mean Discrepancy (MMD).** Find the function with the largest gap in expectation between the two distributions:

```text
MMD(p, q) = sup_{f ∈ F}  ( E_p[f(x)] − E_q[f(x)] )      — if large, p ≠ q
```

For linear functions in a kernel feature space `φ` (a Reproducing Kernel Hilbert Space, `k(x, x') = ⟨φ(x), φ(x')⟩`), the supremum has a *closed form*: it's the distance between the mean embeddings, `‖E_p[φ(x)] − E_q[φ(x)]‖`. The big practical win is that **you don't have to train anything** — the discriminant is `f(x') = E_p[k(x, x')] − E_q[k(x, x')]`, and on finite samples MMD² is a simple sum of kernel evaluations over pairs of points:

```text
MMD² ∝ (1/m(m−1)) Σ_{i≠j} [ k(x_i, x_j) + k(x'_i, x'_j) − k(x_i, x'_j) − k(x'_i, x_j) ]
```

Pick an RBF kernel, evaluate, done — an easy-to-generate discriminator with no training loop.

**(3) Kolmogorov–Smirnov (KS) test — great for 1-D.** Restrict the witness function to bounded total variation `TV[f] ≤ 1`, and the MMD reduces to the largest gap between the two **cumulative distribution functions**:

```text
sup_z | F_p(z) − F_q(z) |  =  ‖F_p − F_q‖_∞ ,    where  F_p(z) = ∫_{−∞}^{z} p(x) dx
```

That sup-distance between CDFs is exactly the KS statistic. It's the natural choice for **monitoring one feature at a time** — run a KS test per scalar feature of your production stream against a training reference, and you get a cheap, interpretable per-feature drift alarm. (For high-dimensional joint shift, go back to the classifier test.)

All three answer the same question — *are `X` and `X'` from the same distribution?* — and CS329P's framing is that this is the **sanity check** you run to confirm distributions match before you trust any conclusion (or trigger any correction).

---

## 6. Label shift correction — BBSE

Now the other tractable case: label shift, `q(x, y) = q(y)·p(x|y)`, where the class-conditional appearance `p(x|y)` is fixed and only the class prior `p(y) → q(y)` moved (disease prevalence jumps; the symptoms of a sick patient don't). Two scenarios.

**The easy case — you already know `q(y)`.** Then correcting a trained model is almost trivial. By Bayes' rule, the test posterior is the training posterior times the prior ratio:

```text
q(y|x) ∝ p(y|x) · [ q(y) / p(y) ]      then renormalize over y
```

You trained `p(y|x)` on the original data; multiply each class probability by `β(y) = q(y)/p(y)`, renormalize, and you have the corrected predictions. Equivalently, for retraining, reweight each example by `β(y_i)`. *"When you have q(y), fixing models is easy."*

**The real case — you do NOT have labels from the test distribution**, so you can't just count up `q(y)`. You have unlabeled production inputs and a trained model. The key insight powering **Black-Box Shift Estimation (BBSE)** (Lipton et al., 2018): because `p(x|y)` is unchanged, the *distribution of your model's predictions per true class is also unchanged* between train and test. So measure the model's behavior, not the unobservable labels. Define the confusion structure and the test prediction distribution:

```text
Confusion (per-class prediction dist. on train):  p(ŷ, y) = ∫ p(ŷ | x) p(x|y) p(y) dx
Predicted-label distribution observed on test:     q(ŷ) = ∫ p(ŷ|x) q(x) dx = Σ_y p(ŷ, y) · β_y
```

That last equation is the engine: a **linear system** `q(ŷ) = C · β` where `C` is the (estimable) confusion matrix and `q(ŷ)` is the (observable) vector of how often each class is *predicted* on the unlabeled test set. Solve it for the per-class weights `β`, which give you `q(y) = β · p(y)`. The algorithm is short:

```text
BBSE — BLACK-BOX SHIFT ESTIMATION
  C = 0 ;  q = 0
  for each training point i:           # build confusion matrix
      C[:, y[i]]  +=  p(· | x[i])      # soft predictions, column = true class
  for each test point i (unlabeled):   # build predicted-label distribution
      q           +=  p(· | x'[i])
  β = C⁻¹ q                            # naive solve
  # Better: constrained least squares —
  minimize_β  ‖ q − C β ‖²   s.t.   β[y] ≥ 0  and  Σ_y β[y] p[y] = 1
  # Then deploy: q(y|x) ∝ p(y|x) · β[y], renormalized.
```

The constrained version (non-negative weights, valid normalized prior) is the one to use — it can't return a nonsensical negative or unnormalized prior. **Why it's trustworthy:** BBSE is *robust under misspecification* — even if the model's predictions `ŷ(x)` are themselves *wrong*, the method still recovers the right weights as long as the model is **calibrated consistently**, i.e. it makes the *same* errors on the hold-out and test sets (its confusion structure is stable). You don't need an accurate model; you need a *consistently-erring* one. The confusion matrix and label vector concentrate (provable via matrix Bernstein), and the algorithm is cheap: cubic in the number of classes, linear in sample size — so it scales fine to thousands of examples and modest class counts. Extensions handle the harder regimes: streaming estimation of the weights via SGD on moment-matching, and feature/GAN moment-matching (MMD) or a train-vs-test score classifier for very large label sets.

The contrast with §3 is the whole point of the taxonomy paying off: **covariate shift reweights by `β(x) = q(x)/p(x)` (per *input*); label shift reweights by `β(y) = q(y)/p(y)` (per *class*).** Same importance-weighting machinery, applied to the factor that actually moved.

---

> **Hardware lens / production:** Distribution shift is where MLOps earns its keep, and the mechanism is concrete: **monitoring a deployed model for drift is just running two-sample tests on its inputs and outputs, continuously.** Snapshot the feature distribution at training time as a reference; then on the live serving stream run a per-feature **KS test** (§5) on each scalar feature and a **classifier two-sample test** on the joint to catch correlated shift no single feature reveals — alarm when either crosses threshold. Watch the *prediction* distribution `q(ŷ)` too: that's the cheap front-half of **BBSE** (§6) and the first sign of label shift, because you can compute it from unlabeled production traffic with **zero new labels** (ground-truth labels usually arrive late or never). This ties directly to **MLOps Module 4B** — drift detection is a monitoring pipeline, not a one-off audit: it has compute cost (those kernel sums and per-feature tests run on every batch), it needs the reference statistics versioned alongside the model, and it should gate an automated response — page a human, trigger importance-reweighted or fresh-data retraining, or in the severe case (effective sample size `m*` collapsed, §3) refuse to auto-correct and demand new labeled data. The engineering parallel to Lecture 02's input pipeline: just as you measure samples/s to catch a starved GPU, you measure *distributional distance over time* to catch a starved model — one whose training distribution has quietly diverged from the world it now serves.

> **2026 update:** Drift monitoring is now productized infrastructure rather than something you hand-roll. **Evidently** (open-source) generates drift dashboards and test suites — per-feature statistical tests (KS, PSI, Wasserstein, chi-square) plus prediction-drift and data-quality reports — and is the common default for tabular pipelines. **NannyML** specializes in the hard, valuable case from §6: *estimating model performance on unlabeled production data* via confidence-based and BBSE-style methods, so you get an accuracy estimate before ground-truth labels land. **WhyLabs** (built on the open-source `whylogs` profiling format) does lightweight statistical profiling at scale for streaming/production telemetry without shipping raw data off-box. Cloud platforms ship it natively now — SageMaker Model Monitor, Vertex AI Model Monitoring, Azure ML data-drift monitors all run scheduled two-sample tests against a training baseline and emit alerts. The newest frontier is **LLM and embedding drift**: the inputs are unstructured text, so monitoring moved to **embedding-space drift** (MMD / classifier tests on embedding distributions — exactly §5 in a learned feature space), plus **eval drift** — tracking an LLM-judge or task-accuracy metric over time as prompts, user behavior, and an upstream model's own updates silently change the distribution under you (the spam-arms-race of §4 reborn as prompt-injection and changing usage). The CS329P mental model — *name the shift, test for it, reweight or retrain* — maps cleanly onto every one of these tools; they are productized versions of this exact lecture.

---

## Current as of

Written June 2026. The **original CS329P content** — the IID assumption and generalization recap, the covariate/label/concept taxonomy keyed to how `p(x, y)` factors, importance weighting via a domain classifier with weight clipping and effective sample size, adversarial examples and invariances/robust loss, the three two-sample tests (classifier / MMD / KS), and BBSE for label shift — is taught first because it is still the correct working framework and maps one-to-one onto the 2021 slides (Lectures 6–7 of the original course). The **refresh layer** flags only what's new in tooling: the productization of drift monitoring (Evidently, WhyLabs, NannyML, and native cloud model monitors), unlabeled-data performance estimation as a standard capability, and the extension of these exact tests into embedding space and LLM-eval drift. The math is unchanged; the world it monitors got bigger.

*Adapted from [Stanford CS329P](https://c.d2l.ai/stanford-cs329p) — Huang, Li & Smola, CC-BY-SA-4.0.*
