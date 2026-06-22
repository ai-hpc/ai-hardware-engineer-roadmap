# Lecture 05 - Model Combination: Bagging, Boosting, Stacking

**Collection:** [Practical Machine Learning (CS329P)](README.md) | **Previous:** [← Lecture 04](Lecture-04.md) | **Next:** [Lecture 06](Lecture-06.md)

---

Lecture 04 taught you to *trust the score* — to validate a single model honestly. This lecture is about what you do once you have several decent-but-imperfect models and want one that is better than any of them. A single model is rarely the strongest entry on a leaderboard or the most reliable thing in production; the standard move is to **combine** models. Ensembling is, at bottom, a way to **spend compute to buy accuracy** — you train and serve more models than you strictly need, and the combination beats the best individual learner.

The reason ensembling works, and the reason there are *three different* ways to do it, comes straight from the **bias–variance decomposition**. Generalization error splits into three additive pieces — bias², variance, and irreducible noise — and each combiner attacks a different piece. **Bagging** drives down *variance* by averaging independent models. **Boosting** drives down *bias* by stacking weak models that each fix the last one's mistakes. **Stacking** learns a meta-model over diverse base learners and can chip at both. So the decomposition is not abstract theory; it is the decision procedure. Diagnose whether your model is bias-limited (underfitting) or variance-limited (overfitting), and the math tells you which combiner to reach for.

The cost is real and worth stating up front: an ensemble of $N$ models costs roughly $N\times$ the inference latency and $N\times$ the memory of one model. That tension — accuracy from combination versus the cost of serving many models — is exactly the pressure that makes the compression and distillation of Lecture 10 necessary. Combine to win the offline metric; compress to afford the online one.

---

## Learning objectives

By the end of this lecture you should be able to:

1. **Derive the bias–variance decomposition** $\mathbb{E}[(y - \hat f)^2] = \text{bias}^2 + \text{variance} + \text{noise}$, and say what raises each term and how model complexity trades them off.
2. **Map each ensemble method to the term it reduces** — bagging→variance, boosting→bias, stacking→both — and pick the right one from a bias/variance diagnosis.
3. **Implement bagging**: bootstrap sampling, independent parallel training, averaging/voting, the out-of-bag estimate, and Random Forest as the canonical case.
4. **Implement boosting**: the sequential residual-fixing loop, AdaBoost reweighting vs. gradient boosting's residual fitting, and why it overfits without regularization.
5. **Build a stacking ensemble** with diverse base learners and a meta-learner trained on *out-of-fold* predictions — and explain the leakage trap that breaks naive stacking.
6. **Reason about the inference cost** of an ensemble and connect it forward to model compression.

---

## 1. Bias–variance decomposition — the decision procedure

Everything in this lecture hangs off one equation, so we derive it. Assume the data is generated as $y = f(x) + \varepsilon$, where $f$ is the true (unknown) function and $\varepsilon$ is irreducible noise with $\mathbb{E}[\varepsilon] = 0$ and variance $\sigma^2$. We sample a dataset $D = \{(x_1, y_1), \dots, (x_n, y_n)\}$ and learn an estimator $\hat f_D$ by minimizing MSE on $D$. We want $\hat f_D$ to **generalize** — to do well on a fresh point $(x, y)$ it never trained on.

The quantity we care about is the expected error on that new point, *averaged over all the training sets $D$ we could have drawn*. Add and subtract the average prediction $\mathbb{E}_D[\hat f_D]$ inside the square:

```text
E_D[(y - f̂_D(x))²]
  = E_D[ ( (f - E_D[f̂_D]) - (f̂_D - E_D[f̂_D]) + ε )² ]
  = (f - E_D[f̂_D])²            ← Bias²    : how far the average model is from truth
  + E_D[(f̂_D - E_D[f̂_D])²]      ← Variance : how much the model wobbles across datasets
  + σ²                          ← Noise    : irreducible, no model can beat it

  = Bias[f̂_D]² + Var[f̂_D] + σ²
```

The cross-terms vanish in expectation because $\varepsilon$ has zero mean and is independent of the data, and because $\mathbb{E}_D[\hat f_D - \mathbb{E}_D[\hat f_D]] = 0$ by definition. What remains are three **additive, non-negative** terms:

| Term | What it measures | What raises it | How you lower it |
|---|---|---|---|
| **Bias²** | Error of the *average* model vs. the truth | Model too simple to express $f$ (underfitting) | Use a more complex model — more layers / hidden units; **boosting**; stacking |
| **Variance** | How much the model changes when the dataset changes | Model too flexible, memorizes the sample (overfitting) | Use a simpler model; regularization; **bagging**; stacking |
| **Noise $\sigma^2$** | Irreducible randomness in $y$ given $x$ | Bad/insufficient features; measurement error | You can't — only better *data* shrinks it |

### The tradeoff

As you increase model complexity, **bias falls** (a richer model can fit $f$ more closely) but **variance rises** (a richer model latches onto the noise in this particular $D$). Total error is U-shaped: too simple is **underfitting** (high bias, low variance), too complex is **overfitting** (low bias, high variance), and the sweet spot — best generalization — sits in between.

```text
 Error
   │ \                                   /   ← total = bias² + var + σ²
   │  \                                 /
   │   \        best generalization    /
   │    \              ↓              /
   │     \____      ___________     /        ← Variance (rises with complexity)
   │          \____/           \___/
   │   Bias²  ___                              ← Bias² (falls with complexity)
   │      \__/   \________________________
   └───────────────────────────────────────►  Model complexity
     underfitting          overfitting
```

### Why this picks your combiner

This is the punchline that organizes the rest of the lecture. **Ensemble learning** means training and combining multiple models to improve predictive performance — and the decomposition tells you *how* to combine:

- If your model **underfits** (high bias, e.g. shallow trees) → you need to **reduce bias** → **boosting** (or stacking).
- If your model **overfits** (high variance, e.g. deep unpruned trees) → you need to **reduce variance** → **bagging** (or stacking).
- **Stacking** can attack either, by learning how to weight diverse models.

Reduce noise only by **improving the data** — more samples, better features, cleaner labels. No combiner touches $\sigma^2$.

---

## 2. Bagging — Bootstrap AGGregatING

**Bagging reduces variance by averaging many independent models.** You train $n$ base learners *in parallel*, each on a slightly different version of the data, then combine their outputs. Because the models are trained independently, this is the variance-killing combiner.

The "different version of the data" is a **bootstrap sample**: given a dataset of $m$ examples, draw $m$ examples **with replacement**. Sampling with replacement means some examples appear several times and others not at all. A classic result: each bootstrap sample contains about $1 - 1/e \approx 63\%$ of the unique examples, leaving ~37% untouched — the **out-of-bag (OOB)** examples, which you get to use as free validation data.

Combine the learners by **averaging** their outputs for regression, or **majority voting** for classification.

```python
from sklearn.base import clone
import numpy as np

class Bagging:
    def __init__(self, base_learner, n_learners):
        self.learners = [clone(base_learner) for _ in range(n_learners)]

    def fit(self, X, y):
        for learner in self.learners:
            # bootstrap: sample len(X) indices WITH replacement
            idx = np.random.choice(len(X), len(X), replace=True)
            learner.fit(X.iloc[idx], y.iloc[idx])

    def predict(self, X):
        preds = [learner.predict(X) for learner in self.learners]
        return np.array(preds).mean(axis=0)   # vote for classification
```

### Why averaging shrinks variance

For regression, the bagged prediction is the average over base learners, $\hat f(x) = \mathbb{E}_D[\hat f_D(x)]$. By Jensen's inequality, $\mathbb{E}[X]^2 \le \mathbb{E}[X^2]$, so:

```text
(f(x) - f̂(x))²  ≤  E_D[(f(x) - f̂_D(x))²]
└── error of the bagged model ──┘   └── average error of a single learner ──┘
```

The ensemble's error is **no worse than**, and in practice better than, the average individual learner's error — the gain comes from cancelling the independent fluctuations (variance) while leaving the shared bias untouched. Bagging therefore **does not reduce bias**: if every base learner systematically underfits the same way, averaging them keeps that bias.

### Unstable learners are where bagging pays off

Bagging helps most for **unstable** (high-variance) learners — models whose predictions swing a lot when the training set changes slightly.

- **Decision trees are unstable** — re-sample the data and a deep tree can split on entirely different features, producing a very different function. Perfect bagging candidate.
- **Linear regression is stable** — its fit barely moves under resampling, so there is little variance to average away, and bagging buys you almost nothing.

### Random Forest — the canonical case

**Random Forest = bagging with decision trees, plus one extra trick.** Beyond bootstrapping the rows, at each split a random forest considers only a **random subset of features**. This decorrelates the trees: without it, a few dominant features would make every tree look alike, and averaging near-identical models barely reduces variance. By forcing trees to use different features, you get genuinely diverse learners and the averaging bites harder. The result is one of the most robust off-the-shelf models in existence — little tuning, strong defaults, and the OOB samples give you a validation estimate for free, no separate split required.

> Bagging is **embarrassingly parallel**: the $n$ learners share nothing during training, so you can train them on $n$ machines/cores at once. This is its operational signature and the cleanest contrast with boosting.

---

## 3. Boosting — sequential bias reduction

**Boosting reduces bias by training weak learners sequentially, each fixing the previous ensemble's mistakes.** Where bagging trains independent models in parallel and averages, boosting trains models *one after another*, and each new learner concentrates on the examples the current ensemble gets wrong. A "weak learner" is a model only slightly better than chance — a shallow tree (a *stump* is depth-1). Stacked the right way, weak learners compound into a strong one.

The general loop, at step $t$:

1. Evaluate the current ensemble's errors $\varepsilon_t$ on the training data.
2. Train a new weak learner $\hat f_t$ that **focuses on the wrongly-predicted examples**.
3. **Additively combine** $\hat f_t$ into the ensemble.

The two famous variants differ only in *how* a learner is made to focus on the errors:

- **AdaBoost** — *reweight/resample by error.* Misclassified examples get higher weight (or are resampled more often) so the next learner pays them more attention; each learner also gets a confidence weight in the final vote.
- **Gradient Boosting** — *fit the residual.* Train the next learner to predict the current errors directly, then add it on.

### Gradient boosting, precisely

Gradient boosting supports any **differentiable loss**. Let $H_t(x)$ be the combined model's output at step $t$, starting from $H_1(x) = 0$. At each step:

- Train a new learner $\hat f_t$ on the **residuals** $\{(x_i,\; y_i - H_t(x_i))\}_{i=1}^{m}$.
- Combine with a **shrinkage** (learning-rate) parameter $\eta$ for regularization: $H_{t+1}(x) = H_t(x) + \eta\, \hat f_t(x)$.

The name comes from the gradient view. For squared-error loss $L = \tfrac12 (H(x) - y)^2$, the residual *is* the negative gradient of the loss w.r.t. the model's output:

```text
L = ½ (H(x) − y)²        ⇒        y − H(x) = − ∂L/∂H
```

So fitting the residual = taking a gradient-descent step **in function space**. For a general loss $L$, the new learner approximates the negative gradient $-\partial L / \partial H_t$ — hence *gradient* boosting. $\eta$ is the step size: small $\eta$ means each tree contributes a little, you need more trees, and you generalize better.

```python
from sklearn.base import clone
import numpy as np

class GradientBoosting:
    def __init__(self, base_learner, n_learners, learning_rate):
        self.learners = [clone(base_learner) for _ in range(n_learners)]
        self.lr = learning_rate

    def fit(self, X, y):
        residual = y.copy()
        for learner in self.learners:
            learner.fit(X, residual)
            residual = residual - self.lr * learner.predict(X)   # chase what's left

    def predict(self, X):
        preds = [learner.predict(X) for learner in self.learners]
        return np.array(preds).sum(axis=0) * self.lr
```

### Why boosting overfits without regularization

Boosting drives **bias** down — by construction it keeps fitting whatever error remains, so given enough rounds it can fit the training set arbitrarily well, *including its noise*. That is exactly the overfitting failure mode: keep boosting and training error goes to zero while test error turns back up. Unlike bagging (where adding more trees is essentially free and monotonically safe), **more boosting rounds is a knob that can hurt.** The standard regularizers:

- **Shrinkage** — small learning rate $\eta$ (e.g. 0.01–0.1); each learner moves the ensemble only a little.
- **Subsampling** — train each learner on a random subset of rows (stochastic gradient boosting) and/or columns.
- **Early stopping** — stop adding trees when validation error stops improving.
- **Weak base learners** — shallow trees (small `max_depth`) so no single learner overfits.

### GBDT in practice — XGBoost & LightGBM

The dominant form is **Gradient Boosting Decision Trees (GBDT)**: the weak learner is a decision tree, regularized by a small `max_depth` and random feature sampling. The catch is that boosting is **inherently sequential** — tree $t$ needs the residuals from tree $t-1$, so you cannot train the trees in parallel the way bagging does. Naive GBDT is slow.

The industry libraries solve this with accelerated algorithms that parallelize *within* each tree's construction:

- **XGBoost** — histogram-based split finding, regularized objective, sparsity-aware splits, the workhorse default.
- **LightGBM** — leaf-wise tree growth and histogram bucketing for large datasets; typically the fastest, very memory-efficient.
- **CatBoost** — ordered boosting and native categorical handling, strong when you have many categorical features.

These are the reason GBDTs, not deep nets, remain the default for **tabular** data.

---

## 4. Stacking — a meta-learner over diverse models

**Stacking trains diverse base learners and then a meta-learner on top of their predictions.** Where bagging gets diversity from *bootstrap samples of the same model type*, stacking gets it from **different model types entirely** — a Random Forest, a GBDT, and an MLP all look at the same inputs but extract different kinds of structure. The meta-learner (often a simple linear model) learns how to **weight and combine** their outputs.

```text
                 ┌─────────────┐
                 │   Dense     │   ← meta-learner: learns the combination weights
                 └──────┬──────┘
                 ┌──────┴──────┐
                 │   Concat    │   ← stack base-learner predictions
                 └──────┬──────┘
        ┌───────────────┼───────────────┐
   ┌────┴────┐     ┌────┴────┐      ┌────┴────┐
   │ Random  │     │  GBDT   │  …   │   MLP   │   ← diverse base learners
   │ Forest  │     │         │      │         │
   └────┬────┘     └────┬────┘      └────┬────┘
        └───────────────┼───────────────┘
                 ┌──────┴──────┐
                 │   Inputs    │
                 └─────────────┘
```

This is the method that **wins competitions**. On the CS329P house-sales benchmark, stacking diverse learners beat each one alone:

| Model | Test error |
|---|---|
| GBDT | 0.259 |
| RandomForest | 0.243 |
| **Stacking (AutoGluon)** | **0.229** |

```python
from autogluon.tabular import TabularPredictor

predictor = TabularPredictor(label=label).fit(train)   # stacks a zoo of models for you
```

### The leakage trap — you MUST use held-out predictions

Here is the mistake that quietly ruins naive stacking. If you train the base learners on the full training set and then feed their predictions *on that same training set* to the meta-learner, the base models have already **seen** those labels — their training predictions are unrealistically good, so the meta-learner learns to trust them far more than it should. It looks brilliant in training and falls apart on new data. This is **label leakage** through the meta-features.

The fix is to give the meta-learner **out-of-fold (OOF) predictions** — predictions on data the base learner did *not* train on:

- **Blending** — split off a small holdout: train base learners on the rest, predict on the holdout, train the meta-learner on those holdout predictions. Simple, but the meta-learner only sees a small slice.
- **Stacking (proper)** — use **k-fold** cross-validation. Train base learners on $k-1$ folds, predict the held-out fold; rotate so every training row gets an out-of-fold prediction. The meta-learner trains on the full set of OOF predictions — uses all the data, no leakage.

### Multi-layer stacking

You can stack in **multiple levels** to also reduce **bias**: the level-2 learners train on the *outputs* of the level-1 learners (often **concatenating the original inputs**, which helps). To keep this from overfitting, train each level on different data — split into A and B, train L1 on A, run inference on B to generate L2's training data. **AutoGluon** generalizes this with **repeated k-fold bagging**: train $k$ models as in k-fold CV, combine each model's out-of-fold predictions, then repeat $n$ times and average — giving the upper level clean, low-variance training data.

The cost is brutal and worth seeing in numbers. On the same benchmark, adding **one** extra stacked level with 5-fold repeated bagging moved error only `0.229 → 0.227` while training time went `39 s → 207 s` (≈5×):

```python
from autogluon.tabular import TabularPredictor

predictor = TabularPredictor(label=label).fit(
    train, num_stack_levels=1, num_bag_folds=5)
```

That is the law of diminishing returns on a slide: a tiny accuracy gain for a multiplicative cost — the same accuracy-vs-compute bargain that runs through this whole lecture.

---

## 5. Putting it together — which combiner, when

| | **Bagging** | **Boosting** | **Stacking** |
|---|---|---|---|
| **Reduces** | Variance | Bias | Variance (both, if multi-layer) |
| **Training order** | Parallel (independent) | Sequential (each fixes the last) | Base parallel, meta on top |
| **Parallelizable?** | Yes — embarrassingly | No — sequential dependency | Base learners yes; levels sequential |
| **Base learners** | Many copies of *one* unstable model (e.g. trees) | Many *weak* learners (shallow trees) | *Diverse* model types (RF + GBDT + MLP) |
| **Diversity from** | Bootstrap samples (+ feature subsets) | Reweighting / residual focus | Different model architectures |
| **Overfit risk** | Low — more learners is safe | **High if unregularized** — more rounds can hurt | Medium — leaks if you skip out-of-fold |
| **Canonical example** | Random Forest | XGBoost / LightGBM (GBDT) | AutoGluon / Kaggle winners |
| **Free validation** | Out-of-bag (OOB) estimate | Early-stopping on a holdout | Out-of-fold predictions |

The CS329P summary table, with $n$ = number of learners, $l$ = levels, $k$ = folds:

```text
                       Reduce Bias   Reduce Var   Compute cost   Parallelization
   Bagging                  -            Y             n               n
   Boosting                 Y            -             n               1
   Stacking                 -            Y             n               n
   K-fold multi-level       Y            Y           n×l×k            n×k
```

Read it as a recipe: **underfitting → boost; overfitting → bag; want the last 1% and have the compute → stack.**

> **Hardware lens:** Every accuracy gain in this lecture is bought with **inference cost**. An ensemble of $N$ models is, at serve time, $N$ forward passes — roughly **$N\times$ the latency and $N\times$ the memory** of a single model (a 1000-tree GBDT evaluates 1000 trees per prediction; a 5-model stack runs 5 full models plus the meta-learner; multi-layer stacking multiplies again by levels and folds). Offline, on a Kaggle box, that is free — you have all night. Online, behind a latency SLA on a fixed accelerator budget, it is often *unaffordable*: you cannot put a 5× latency hit in a request path that must answer in 20 ms, and you may not have 5× the GPU memory to hold the ensemble resident. This is precisely the gap **Lecture 10 (Model Compression — Pruning, Quantization, Distillation)** exists to close. **Distillation** is the most direct answer: train the big ensemble to win the metric, then train *one* small student to mimic the ensemble's outputs — keeping much of the accuracy at $1\times$ inference cost. The workflow that ships is *ensemble to find the ceiling, distill to afford it.* So measure both numbers: the leaderboard score **and** the per-prediction cost, because in production the second one is a hard constraint, not a footnote. [→ Lecture 10](Lecture-10.md)

> **2026 update:** Five years on, the lecture's thesis holds, sharpened. **(1) GBDT ensembles still own tabular.** XGBoost, LightGBM, and CatBoost remain the default winners on tabular Kaggle competitions and most structured industry data — deep nets have *not* displaced them there, despite repeated attempts (TabNet, FT-Transformer, SAINT), and recent in-context tabular models like **TabPFN v2** are promising on small data but have not dethroned boosted trees at scale. **(2) Deep ensembles for uncertainty.** Training a handful of neural nets with different seeds and averaging is now a standard, strong baseline for **predictive uncertainty and calibration** — it reliably beats fancier Bayesian approximations, and is used where knowing *when the model is unsure* matters (safety, active learning, OOD detection). **(3) Weight-averaging instead of output-averaging — "model soups".** A 2022-onward idea that stuck: rather than serve $N$ models, **average their weights** into a single set. **Model soups** (averaging the weights of multiple fine-tunes of the same base model) and **SWA / WiSE-FT** improve accuracy and robustness while paying only **$1\times$ inference cost** — the holy grail the hardware lens asks for, with the catch that it requires the models to live in a compatible region of weight space (typically fine-tunes of a shared initialization, not arbitrary architectures). For LLMs, weight-space *merging* (e.g. linear/SLERP merges, TIES, DARE) is the same instinct applied to combining specialized fine-tunes into one served model. The pattern of the decade: **get the ensemble's accuracy without paying the ensemble's serving cost** — by distilling it (Lecture 10) or by averaging weights instead of outputs.

---

## Current as of

Written June 2026. The **original CS329P content** — the bias–variance decomposition and tradeoff, bagging (bootstrap, OOB, Random Forest, the variance-reduction argument), boosting (AdaBoost vs. gradient boosting, residual = negative gradient, shrinkage/subsampling/early-stopping, GBDT with XGBoost/LightGBM), and stacking (diverse base learners, out-of-fold meta-features, multi-layer stacking with repeated k-fold bagging, AutoGluon, and the summary table) — is taught first because it remains the correct working mental model and maps one-to-one onto the 2021 slides. The **2026 refresh layer** flags what moved: GBDTs' continued dominance over deep nets on tabular data (and TabPFN as the watch-this-space exception), **deep ensembles** as the standard uncertainty/calibration baseline, and **weight-averaging — model soups, SWA, WiSE-FT, and LLM weight-merging** — as the way to capture an ensemble's accuracy at single-model inference cost, which together with **distillation (Lecture 10)** answers the hardware lens's core complaint. Where 2021 framing is dated, the original is presented before the update; nothing is silently rewritten.

*Adapted from [Stanford CS329P](https://c.d2l.ai/stanford-cs329p) — Huang, Li & Smola, CC-BY-SA-4.0.*
