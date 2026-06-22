# Lecture 04 - Model Validation & Evaluation

**Collection:** [Practical Machine Learning (CS329P)](README.md) | **Previous:** [← Lecture 03](Lecture-03.md) | **Next:** [Lecture 05](Lecture-05.md)

---

Every number you report about a model is a promise about data it has never seen. "94% accuracy" is not a statement about your test set; it is a bet that the next thousand examples from the real world will behave like the ones you measured on. Evaluation is the discipline of making that bet honest. The uncomfortable truth of this lecture is that *most of the ways people measure model quality lie to them* — and they lie in the optimistic direction, which is the worst direction, because an inflated score is the one you don't investigate. You ship it.

There are three families of lie, and this lecture takes them in order. The first is **the wrong metric**: 94% accuracy on a problem where 95% of examples are negative means your model is worse than a constant that always says "no." The second is **the wrong place on the complexity curve**: a model that nails the training data and falls apart on new data has memorized, not learned, and the training score told you nothing. The third, and the deadliest because it survives code review, is **a contaminated estimate** — leakage, where information from the future or from the test set sneaks into training and your validation score quietly stops measuring generalization at all. CS329P's blunt rule of thumb: *if your model's performance is too good to be true, it almost certainly is, and a contaminated validation set is the number-one reason.*

The throughline is a single quantity you can never directly observe — **generalization error**, the error on the true distribution — and the entire apparatus of metrics, the bias-variance picture, and held-out splits exists to *estimate* it without fooling yourself. Get the estimate right and everything downstream (model selection, hyperparameter tuning, the go/no-go ship decision) rests on solid ground. Get it wrong and you are optimizing a fiction.

---

## Learning objectives

By the end of this lecture you should be able to:

1. **Choose the right metric** for a classification, regression, or ranking problem, and explain precisely why accuracy fails under class imbalance.
2. **Read a confusion matrix** and derive precision, recall, F1, ROC-AUC, and PR-AUC from it — and pick a decision threshold deliberately rather than defaulting to 0.5.
3. **Diagnose underfitting vs. overfitting** from the training-vs-generalization error gap and the model-complexity curve, and prescribe the right fix for each.
4. **Design a train/validation/test split** with the discipline that the test set is used exactly once and hyperparameters are tuned on validation only.
5. **Apply k-fold cross-validation** and recognize the cases where vanilla CV is *wrong* — time series and grouped data.
6. **Spot the classic leakage bugs** — fitting a scaler on the full dataset, target leakage, and near-duplicate records straddling the split — before they inflate your score.

---

## 1. Evaluation metrics — measuring the right thing

The loss you train on (cross-entropy, MSE) measures how well the model fits, but it is rarely the number anyone cares about. We evaluate models with **multiple metrics**, and CS329P sorts them into two buckets:

- **Model metrics** measure performance on examples: accuracy, precision/recall, F1, AUC. These are what you compute on a held-out set.
- **Business metrics** measure the model's impact on the product: revenue, inference latency, click-through rate. These are what the company actually optimizes.

Selecting a model is like choosing a car — you weigh several metrics at once, and the best validation accuracy is not automatically the model you ship. Hold that idea; we return to it at the end of the section.

### 1.1 Classification metrics

**Accuracy** = (correct predictions) / (total examples). It is the first metric everyone reaches for and the first one that betrays them. The failure mode is **class imbalance**. Consider fraud detection where 99.8% of transactions are legitimate. A model that predicts "not fraud" for *every single transaction* scores 99.8% accuracy and is completely worthless — it catches zero fraud, the only thing the system exists to do. Accuracy is dominated by the majority class, so on any skewed problem (fraud, disease screening, ad clicks, defect detection) it is actively misleading. The fix is to measure the classes separately.

**The confusion matrix** is the foundation everything else is built on. For binary classification it tabulates predictions against truth:

```text
                      Predicted Positive    Predicted Negative
Actual Positive       True Positive  (TP)   False Negative (FN)
Actual Negative       False Positive (FP)   True Negative  (TN)
```

From its four cells come the metrics that *don't* collapse under imbalance:

- **Precision** = TP / (TP + FP) — of everything you flagged positive, what fraction really was? Low precision = crying wolf. *(Watch for division by zero when the model predicts no positives.)*
- **Recall** (a.k.a. sensitivity, TPR) = TP / (TP + FN) = TP / (all actual positives) — of everything that truly was positive, what fraction did you catch? Low recall = missing real cases.

Precision and recall trade off, and *which one matters is a domain decision, not a math one.* Cancer screening prizes recall — a missed tumor (false negative) is catastrophic, a false alarm merely means a follow-up scan. A spam filter prizes precision — a missed spam (false negative) is a minor annoyance, but a real email dumped in the spam folder (false positive) loses you a job offer. You cannot maximize both for free.

**F1** collapses the two into one number — the **harmonic mean** of precision and recall:

```text
F1 = 2 · (precision · recall) / (precision + recall)
```

The harmonic mean (not the arithmetic) is deliberate: it punishes imbalance between the two. A model with precision 1.0 and recall 0.0 has arithmetic mean 0.5 but F1 of 0 — F1 refuses to give credit for being great on one axis while failing the other. When false positives and false negatives carry different costs, weight them with the general **Fβ** (β > 1 favors recall, β < 1 favors precision).

**Threshold choice.** A classifier outputs a *score* (a probability-like number `o`), and you turn it into a label by comparing to a threshold θ: predict positive if `o ≥ θ`, else negative. The default θ = 0.5 is a convention, not a law. Lowering θ catches more positives (recall ↑) at the cost of more false alarms (precision ↓); raising it does the reverse. Every (precision, recall) pair you saw above is *a single point on a curve* that the threshold sweeps out — which is exactly what the next two metrics summarize.

**ROC-AUC** measures how well the model *separates the two classes across all thresholds*, independent of any one choice of θ. Sweep θ from high to low and plot the **ROC curve** — True Positive Rate against False Positive Rate:

```text
TPR = #(true positive predictions)  / #(positive examples)     ← y-axis (= recall)
FPR = #(false positive predictions) / #(negative examples)     ← x-axis
```

The **area under this curve (AUC)** lands in `[0.5, 1]`: 0.5 is a coin flip (the diagonal, no separating power), 1.0 is perfect separation. Intuitively, AUC is the probability that the model scores a random positive higher than a random negative. Its strength — threshold-independence — is also its trap: **ROC-AUC is optimistic on heavily imbalanced data** because FPR has a huge negative-class denominator, so even thousands of false positives barely move the x-axis.

**PR-AUC** (area under the **Precision-Recall** curve) is the fix. It plots precision against recall across thresholds and ignores true negatives entirely, so it stays honest when negatives vastly outnumber positives. **Rule of thumb: balanced classes → ROC-AUC is fine; rare-positive problems (fraud, retrieval, anomaly detection) → trust PR-AUC.**

### 1.2 Regression metrics

When the target is continuous, you measure the size of the residuals `(yᵢ − ŷᵢ)`:

| Metric | Formula | Units | Behavior |
|---|---|---|---|
| **MSE** (mean squared error) | `(1/n) Σ (yᵢ − ŷᵢ)²` | target² | Squares errors → punishes large misses hard; outlier-sensitive |
| **RMSE** (root MSE) | `√MSE` | target | MSE back in the target's units — directly interpretable; still outlier-sensitive |
| **MAE** (mean absolute error) | `(1/n) Σ \|yᵢ − ŷᵢ\|` | target | Linear in error → robust to outliers; treats a \$1M miss as 10× a \$100K miss, not 100× |
| **R²** (coefficient of determination) | `1 − SS_res / SS_tot` | unitless | Fraction of variance explained; 1.0 = perfect, 0 = no better than predicting the mean, <0 = worse than the mean |

The **MSE-vs-MAE** choice mirrors precision-vs-recall: it is about how much you fear large errors. MSE/RMSE square the residual, so a single wildly-wrong prediction dominates the score — choose them when big misses are disproportionately bad (and watch that outliers don't hijack training). MAE weights every dollar of error equally and shrugs off outliers — choose it when the data has a heavy tail you don't want to chase. **R²** is the one to report to non-specialists because it's normalized and unit-free: "the model explains 85% of the variance" travels across audiences in a way "RMSE = 41,000" does not.

### 1.3 Ranking and business metrics

Many real systems don't classify in isolation — they **rank**. Search, recommendation, and ad systems return an ordered list, and what matters is whether the good items land near the top. The metrics shift accordingly: **Precision@k** and **Recall@k** (quality of the top *k*), **MAP** (mean average precision), **NDCG** (normalized discounted cumulative gain, which discounts relevance by rank position so a great result at position 1 beats the same result at position 10). For object detection the analog is **mAP** (mean average precision across IoU thresholds).

But the metric that ultimately decides everything is the **business metric**, and CS329P's ad-display case study is the canonical lesson in why model metrics and business metrics diverge. Displaying ads is, at its core, a **binary classification problem**: estimate the click-through rate (CTR) for each candidate ad, then show the top ads ranked by `CTR × price`. The headline model metric is **AUC**. The business, meanwhile, is governed by:

```text
revenue = #pageviews × ASN × CTR × ACP
          where  ASN = avg #ads shown per page
                 CTR = actual user click-through rate
                 ACP = avg price advertiser pays per click
```

Here is the trap that catches every ML team eventually: **a new model with higher AUC can *hurt* revenue.** Possible reasons, straight from the slides — the better-calibrated model estimates *lower* CTRs, so the system displays fewer ads (ASN ↓); the real-world CTR comes in below the offline number because you trained and evaluated on *past* data that no longer reflects current behavior; or the ranking shift lowers realized prices. The offline metric improved and the thing you actually care about got worse. **The only way to know is an online experiment** — deploy the model to a slice of real traffic (an A/B test) and measure the business metrics directly. Offline AUC proposes; online revenue disposes.

### Metric-selection cheat sheet

| Problem type | Default metric | Reach for instead when… | Avoid |
|---|---|---|---|
| **Balanced classification** | Accuracy, ROC-AUC | — | — |
| **Imbalanced classification** | F1, PR-AUC | recall is non-negotiable (screening) → recall@fixed-precision | **Accuracy** (majority-class trap) |
| **Probability calibration matters** | Log loss, Brier score | downstream uses the probability (CTR × price) | thresholded accuracy |
| **Regression** | RMSE | heavy-tailed targets / outliers present → **MAE**; cross-audience reporting → **R²** | MSE alone (uninterpretable units) |
| **Ranking / retrieval / recsys** | NDCG, MAP | only the very top matters → Precision@k | accuracy |
| **The thing that pays the bills** | **Business metric via online A/B test** | always, before shipping | trusting offline metrics alone |

---

## 2. Underfitting vs. overfitting — landing on the complexity curve

CS329P opens this topic with a parable. A lender asks you to predict who will repay their loans. You have 100 applicants; 5 defaulted. You build a model and discover a "surprising" signal — **all 5 who defaulted wore blue shirts to their interviews**, and your model leans hard on it. It will score beautifully on these 100 people and be worthless on the next applicant, because shirt color is noise that happened to correlate in a tiny sample. That is **overfitting** in one sentence: learning the quirks of the training set instead of the structure of the world.

To make this precise, define two errors:

- **Training error** — the model's error on the data it was trained on.
- **Generalization error** — the model's error on *new, unseen* data. This is the only one that matters; the only one you can't directly see.

The relationship between them diagnoses the model:

| | Training error **low** | Training error **high** |
|---|---|---|
| **Generalization error low** | **Good** — the goal | (impossible — a bug; you can't generalize better than you fit) |
| **Generalization error high** | **Overfitting** — memorized the training set | **Underfitting** — too weak to fit even the training set |

The gap between the two is the tell. **Underfitting**: both errors are high and *close* — the model is too simple to capture the pattern even in data it has seen (think a straight line through a curve). **Overfitting**: training error is low but generalization error is high and the *gap is wide* — the model fit the training data including its noise, and that noise doesn't recur.

### The model-complexity curve

Plot error against **model complexity** — the capacity of a function class to fit data, roughly the number of learnable parameters and the range of values they can take (more rigorously, **VC dimension**: the largest set of points the model can shatter). The picture is the central diagram of the whole topic:

```text
 error
   ^
   |  \                                              /   <- generalization error
   |   \  underfitting                              /        (U-shaped: high when too
   |    \  (both high)                             /          simple AND too complex)
   |     \                                        /
   |      \___                              _____/   <- overfitting
   |          \___                    _____/             (gap opens up)
   |              \___           ____/
   |                  \_________/  <-- training error (falls monotonically:
   |                      ^               more capacity always fits train better)
   |                      |
   |              optimal complexity (minimize GENERALIZATION error, not training)
   +----------------------|------------------------------> model complexity
```

Training error falls monotonically — give a model more capacity and it will always fit the training set better, all the way to memorizing it. Generalization error is **U-shaped**: it falls as the model gains enough capacity to capture real structure, bottoms out at the **optimal complexity**, then *rises* as extra capacity is spent fitting noise. You want the bottom of the U, and you find it by watching generalization (validation) error, never training error. CS329P's concrete demo: a scikit-learn `DecisionTreeRegressor(max_depth=n)` on house-sales data — `max_depth=2` underfits (the tree is too shallow to express the price surface), a large `max_depth` overfits (a leaf for nearly every house), and the right depth sits in between.

### It's not just the model — data complexity matters too

"Optimal complexity" is not a property of the model alone; it depends on the **data**. Data complexity rises with the number of examples, the number of features per example, and the separability of the classes (the rigorous version is **Kolmogorov complexity** — data is simple if a short program can generate it). The two interact:

| | Data complexity **low** | Data complexity **high** |
|---|---|---|
| **Model complexity low** | Normal (matched) | **Underfitting** — model too weak for rich data |
| **Model complexity high** | **Overfitting** — model too rich for thin data | Normal (matched) |

The diagonal is where you want to be: **match model complexity to data complexity.** A deep neural network on 200 rows overfits; a linear model on 200 million rich examples underfits and leaves accuracy on the table. This is also why "complex models need more data" — their generalization error only beats a simple model's *once enough data is available* to constrain all those parameters; below that crossover point the simple model wins.

The theory behind the picture is the **generalization-error bound** (informally): the gap between unseen-data error and training error grows with VC-dimension `D` and shrinks as training examples `N` grow — more capacity widens the potential gap, more data narrows it. Crucially, generalization also depends on the **training algorithm**, not just the model: adding **regularization** penalizes complex models and pulls them back toward the optimum, and models trained with **stochastic gradient methods** tend to generalize better than the bound alone would suggest.

### Diagnosing and fixing

| Symptom | Diagnosis | What to do |
|---|---|---|
| Train error **high**, val error **high** (small gap) | **Underfitting** — model too simple / data too rich | **Increase capacity**: bigger model, more/better features, more feature crosses, train longer, *reduce* regularization |
| Train error **low**, val error **high** (large gap) | **Overfitting** — model memorizing noise | **Constrain it**: more training data, regularization (L2, dropout), early stopping, data augmentation, *reduce* capacity / feature count |
| Both errors low and close | **Good fit** — you're at the bottom of the U | Ship it (after §3's validation discipline) |

The two prescriptions are near-mirror images, which is why diagnosing *which* problem you have — by reading the train-vs-val gap — is the whole game. The most common rookie error is throwing a bigger model at an overfitting problem (making it worse) or piling on regularization when the model is underfitting (making *that* worse). Read the gap first.

---

## 3. Model validation — estimating generalization without fooling yourself

You can't observe generalization error, so you **approximate it with a holdout test set** — data the model has never seen and that *can be used exactly once*. CS329P's analogies are sharp: it's your midterm exam score (you don't get to retake it after seeing the questions), the final price of a pending house sale, the private-leaderboard data in a Kaggle competition. The instant you make a decision based on the test set — tweak a hyperparameter, pick a model — it stops being a test set and becomes part of training. Its one-shot nature is the entire point.

So how do you make decisions during development without burning your test set? You hold out a **validation set** from the training data:

- A subset of the data, **not used for training**, that you *can* use many times — for model selection and hyperparameter tuning.
- It should be drawn to be **close to the test distribution** (and the real world), so that doing well on it predicts doing well on test.
- A terminology landmine worth memorizing: in casual ML usage "**test error**" almost always means error on the *validation* set. The true test set is the untouchable final exam.

### The three-way discipline

```text
┌─────────────────────────┬──────────────┬──────────────┐
│         TRAIN           │  VALIDATION  │     TEST      │
│  fit model parameters   │  tune hyper- │  touch ONCE,  │
│  (weights, splits)      │  params,     │  final number │
│                         │  pick model  │  you report   │
└─────────────────────────┴──────────────┴──────────────┘
   used many times          used many       used exactly
                            times            once
```

The rule that protects the estimate: **fit parameters on train, choose hyperparameters on validation, report on test once.** Splits are often a random `n%` for validation — typical `n` is 10–50 — but *how* you split is where it goes wrong, as we'll see.

### k-fold cross-validation

When you don't have enough data to spare a fat validation set, **k-fold cross-validation** recycles it. The algorithm:

```text
Partition the training data into K equal folds.
For i = 1 … K:
    train on the K−1 folds that aren't fold i
    validate on fold i  →  record validation error_i
Report the average validation error over all K rounds.
```

```text
            ┌──────┬──────┬──────┬──────┬──────┐
   Fold 1:  │ VALID│ train│ train│ train│ train│
   Fold 2:  │ train│ VALID│ train│ train│ train│
   Fold 3:  │ train│ train│ VALID│ train│ train│   →  error = mean(error_1..error_5)
   Fold 4:  │ train│ train│ train│ VALID│ train│
   Fold 5:  │ train│ train│ train│ train│ VALID│
            └──────┴──────┴──────┴──────┴──────┘
```

Every example serves as validation exactly once, so you get a lower-variance estimate of generalization error and an error bar (the spread across folds) for free, at the cost of training K times. **Popular choices are K = 5 or 10.**

### When cross-validation is *wrong*

Vanilla random splitting (and vanilla k-fold) assumes examples are **i.i.d.** — independent and identically distributed. When that assumption breaks, random splitting **underestimates** generalization error, because information leaks across the split. Two cases dominate, plus a sampling fix:

- **Sequential / time-series data** (house sales, stock prices). The validation set must **not overlap with training in time**. If a March sale is in train and a February sale from the same neighborhood is in validation, you're using the future to predict the past — a luxury you won't have in production. The fix is **forward chaining**: always train on the past and validate on the future, growing the window forward. CS329P's house-sales case study makes this concrete — splitting the same data **randomly vs. sequentially** changes the picture: the random split flatters a deeper tree (best `max_depth ≈ 13`) while the honest sequential split prefers a shallower one (best `max_depth ≈ 6`). Random splitting let the model exploit temporal leakage and *looked better than it was.*

```text
 forward chaining (time series): never let train see anything after valid
   fold 1:  [== train ==][valid]
   fold 2:  [===== train =====][valid]
   fold 3:  [======== train ========][valid]
            └──────────── time ────────────►
```

- **Clustered / grouped data** — photos of the same person, clips from the same video, multiple lab results from the same patient. Examples within a cluster are correlated, so if some clips of a video land in train and others in validation, the model recognizes the *video*, not the action. **Split whole clusters, not individual examples** — this is **group k-fold**, where every record sharing a group ID stays on the same side of the split.
- **Highly imbalanced classes** — **sample more from the minority class** when forming splits (stratify) so a rare class isn't entirely absent from validation by chance.

### The leakage hall of fame

CS329P's bluntest slide: **if your model's performance is too good to be true, there is very likely a bug, and a contaminated validation set is the #1 reason.** Leakage is the failure that survives code review because the code is *correct* — it's the data flow that's wrong. The classic bugs:

- **Fitting the scaler (or any preprocessing) on the full dataset.** This is the subtlest and most common. You call `StandardScaler().fit(X)` on all the data, *then* split — and now the training data's normalization statistics were computed using the validation/test data's mean and variance. Test information has leaked into training. The fix: fit every transform on the **training fold only**, then `transform` validation and test. (This is the Lecture-02 normalization warning, now a load-bearing rule.)

```python
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

X_tr, X_val, y_tr, y_val = train_test_split(X, y, test_size=0.2, random_state=0)

scaler = StandardScaler().fit(X_tr)    # fit on TRAIN ONLY
X_tr_s  = scaler.transform(X_tr)
X_val_s = scaler.transform(X_val)      # apply the same stats to val — no leakage

# WRONG, and silently inflates your score:
#   X_all_s = StandardScaler().fit_transform(X)   # learns stats from val+test too
#   then split  →  val statistics have bled into the scaler
```

A `Pipeline` makes this automatic — it re-fits every step inside each CV fold, which is exactly why you should wrap preprocessing in one rather than transforming up front.

- **Target leakage** — a feature that encodes the answer. A `was_charged_late_fee` column predicting "loan defaulted" is leakage if the fee is only assessed *after* default; an `account_closed_date` predicting churn; a feature computed using post-outcome information. It produces a spectacular validation score and total failure in production, because at prediction time the leaking feature doesn't exist yet. Audit: for every feature, ask *"would I actually have this value at the moment I need to predict?"*
- **Duplicate / near-duplicate records straddling the split.** Common when you merge datasets — e.g. you scrape images from a search engine to evaluate a model trained on ImageNet, and some are the very same images. The "test" example is one the model already memorized, so the score is fiction. De-duplicate (including *near*-duplicates — resized, recompressed, lightly cropped copies) **before** splitting.
- **Excessive validation reuse is cheating.** Tune hyperparameters against the same validation set hundreds of times and you slowly overfit *to the validation set* — your "validation error" stops tracking the test set. Nested CV, or a final untouched test set, is the guard.

The summary: the test set is spent once; a validation set held out from training estimates it and may be reused for selection — but only if it's drawn to resemble the test distribution and kept clean. An improper validation set is the most common path to **over-estimating** model performance, and over-estimates are the ones that ship.

---

> **2026 update:** Evaluation is the part of the 2021 lecture that the generative-AI era stress-tested hardest, because the bedrock assumption — *there exists a single ground-truth label to compute a metric against* — often no longer holds. When the output is a paragraph, an essay, or generated code, there's no one right string to match: a translation can be excellent in a dozen surface forms, so exact-match and even BLEU/ROUGE correlate weakly with quality. The field's pragmatic answer is **LLM-as-judge** — prompt a strong model to score or rank outputs (pairwise comparison is more reliable than absolute scoring), validated against human preference. It scales, but it imports its own biases (position bias toward the first option, verbosity bias toward longer answers, self-preference toward its own family's style), so it is a noisy proxy, not ground truth, and should be calibrated against human ratings. (For *how* to wire up an LLM judge and which model to call, defer to the provider's own current docs rather than memory — see the Lecture-05 tooling notes.) Meanwhile the **leakage** theme of §3 reappears, scaled to the whole internet, as **benchmark contamination**: if the test questions (MMLU, GSM8K, HumanEval) leaked into the pretraining corpus, a high score measures memorization, not capability — the exact "duplicate records across the split" bug, except the split is *the internet vs. the benchmark* and you usually can't inspect the training set. The defenses are the same in spirit (held-out, freshly-collected, time-gated evaluation sets released *after* a model's training cutoff; canary strings; private leaderboards), and so is the moral: **the score is a promise about unseen data, and contamination is the lie that makes it too good to be true.**

> **Hardware lens:** Metrics are not just statistics, they are *deployment constraints*, and on the systems side the most consequential ones are the operational metrics this lecture lists almost in passing — **inference latency** and throughput. A model that wins on F1 but misses a p99 latency budget cannot ship in an ad-serving or interactive path; the "best model" is the Pareto frontier of *quality × latency × cost*, not the top of the accuracy column. This reframes evaluation as a hardware problem: you measure tail latency (p50/p95/p99) under realistic batch sizes on the *actual* target accelerator, because a model that's accurate enough but 3× too slow gets quantized (INT8/FP8), distilled, or pruned — each of which trades a sliver of the §1 quality metrics for the latency the §1.3 business metrics demand. And evaluation itself is now a compute budget: running a 50-task LLM benchmark suite, or an LLM-as-judge pass over thousands of generations, is a non-trivial GPU bill, which is why offline eval increasingly gets the same throughput engineering — batching, caching, the input-pipeline discipline from Lecture-02 — as training. Generalization, in the end, is measured on the hardware you'll actually serve from, under the latency the business actually requires.

---

## Current as of

Written June 2026. The **original CS329P content** — the model-vs-business-metric split, the binary-classification metric family (accuracy and its imbalance trap, precision/recall/F1, ROC-AUC), the ad-display case study, the underfitting/overfitting diagnosis via the train-vs-generalization gap and the model-/data-complexity curves, and the validation discipline (holdout, k-fold, non-i.i.d. splitting, and the leakage "common mistakes") — is taught first because it remains the correct working mental model and maps one-to-one onto the 2021 slides. The **refresh layer** adds only what the slides predate: **PR-AUC** as the imbalance-honest companion to ROC-AUC, the explicit ranking metrics (NDCG/MAP) and regression-metric table, **forward chaining** and **group k-fold** as the named fixes for the slides' "sequential" and "clustered" cases, and the 2026 evaluation crisis — **LLM-as-judge** and **benchmark contamination** as the generative-era reincarnation of the lecture's own leakage thesis. Where the 2021 framing is dated, the original is presented before the update; nothing is silently rewritten.

*Adapted from [Stanford CS329P](https://c.d2l.ai/stanford-cs329p) — Huang, Li & Smola, CC-BY-SA-4.0.*
