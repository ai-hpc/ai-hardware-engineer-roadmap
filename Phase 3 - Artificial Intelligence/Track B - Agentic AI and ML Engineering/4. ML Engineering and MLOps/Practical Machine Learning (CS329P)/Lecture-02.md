# Lecture 02 - Data II: Cleaning, Transformation & Feature Engineering

**Collection:** [Practical Machine Learning (CS329P)](README.md) | **Previous:** [← Lecture 01](Lecture-01.md) | **Next:** [Lecture 03](Lecture-03.md)

---

In Lecture 01 you got data into the building — scraped, labeled, however imperfectly. This lecture is what happens before any of it is allowed near a model. It is the least glamorous stage in the entire pipeline and the one that most decides your model's ceiling. *Garbage in, garbage out* is not a slogan here; it is a quantitative claim. A model trained on dirty, mis-scaled, badly-represented data does not fail loudly — it converges, posts a plausible loss curve, and quietly underperforms a model that got clean inputs, and you may not notice until it is in production making bad recommendations that poison the very data you collect next.

The work splits into three movements, run roughly in order: **cleaning** (find and fix the values that are wrong), **transformation** (reshape what's correct into the fixed-length, well-conditioned, nicely-distributed form algorithms actually want), and **feature engineering** (turn raw columns into representations relevant to the target). Then a fourth, threaded through all of them: **data summary** — looking at distributions hard enough to know which of the first three you even need. Deep learning has eroded the manual feature-engineering step for unstructured data (a CNN learns its own features), but for tabular data — still most of industry's ML — every step below is yours to do by hand, and doing it well is the difference between a Kaggle medal and a forgettable submission.

The mental model from CS329P: this is one box in a pipeline — `raw data → labelling & cleaning → data transformation → feature engineering → model training` — and it is the box where you spend most of your wall-clock time and earn most of your accuracy.

---

## Learning objectives

By the end of this lecture you should be able to:

1. **Classify a data error** as an outlier, a rule violation, or a pattern violation, and pick the right detector for each.
2. **Normalize real-valued columns** and choose between min-max, z-score, decimal, and log scaling for a given distribution.
3. **Transform unstructured data** — resize/crop/whiten images, clip and sample video, tokenize text — while trading off storage, quality, and load speed.
4. **Engineer features** for tabular (bucketing, one-hot, hashing, datetime, crosses), text (BoW, TF-IDF, n-grams, embeddings), and image data (hand-crafted → learned).
5. **Read a dataset's distributions** well enough to decide which cleaning and transformation steps are actually warranted.
6. **Reason about preprocessing cost** as a throughput problem, not a free pre-step.

---

## 1. Data cleaning — finding what's wrong

Data can contain missing values, mislabeled examples, invalid measurements, and valid rare events. Model robustness depends on the architecture, loss, data, and noise mechanism; deep networks can memorize noisy labels too. Investigate suspicious records before removing them, and preserve a clean evaluation protocol while developing cleaning rules.

CS329P splits errors into three types, and the split matters because each demands a different detector:

| Error type | Definition | Example | How you catch it |
|---|---|---|---|
| **Outliers** | Values that deviate significantly from other observations | A house priced at \$5 in a real-estate set | Statistical / distributional (boxplot, IQR) |
| **Rule violations** | Values that break integrity constraints | A `NOT NULL` field that is null; a negative age; a duplicate primary key | Rule / constraint checks |
| **Pattern violations** | Values that break syntactic or semantic constraints | `"eng"`, `"en"`, `"english"` for the same language; `"Stanford"` in a `Country` column | Pattern / type / knowledge checks |

### Outliers vs. under-sampled rare events

The hard part of outlier detection is that an outlier and a *legitimate rare event* look identical from the value alone. A \$40M sale is a statistical outlier in a housing dataset and also a real transaction. Deleting it because it's far from the median throws away signal. The judgment call — drop, cap (winsorize), or keep — is domain reasoning, not a threshold. The standard *first-pass* tool is the boxplot / Tukey fence: anything beyond `Q3 + 1.5·IQR` or below `Q1 − 1.5·IQR` is flagged as a candidate, not a verdict.

```python
import numpy as np

def iqr_outlier_mask(x: np.ndarray, k: float = 1.5) -> np.ndarray:
    q1, q3 = np.percentile(x, [25, 75])
    iqr = q3 - q1
    lo, hi = q1 - k * iqr, q3 + k * iqr
    return (x < lo) | (x > hi)   # True = candidate outlier, then USE JUDGMENT
```

### Rule-based detection

Rules encode integrity constraints you know must hold. Two flavors from the lecture:

- **Functional dependencies** — `x → y` means a value of `x` determines a unique `y`. For a deliberately constrained robot log, `sensor_serial → sensor_model` may be a declared dependency. Validate the domain rule first; real postal codes can cross jurisdiction boundaries and company identifiers can span name changes.
- **Denial constraints** — richer first-order-logic conditions: *"phone number must not be empty if the vendor has an EIN"*; *"if two captures share a tag number, the earlier one must be marked original."* These catch cross-column inconsistencies a single-column check never would.

```python
# Example declared dependency: one sensor model per serial in this dataset.
violations = (
    df.groupby("sensor_serial")["sensor_model"].nunique()
      .loc[lambda s: s > 1]          # investigate serials associated with multiple models
)
```

### Pattern-based detection

- **Syntactic patterns** — map a column to its most prominent data type and flag values that don't fit, or canonicalize variants (`eng`, `en`, `english` → `English`). A "date" column where 2% of cells are free text is a syntactic violation.
- **Semantic patterns** — bring in external knowledge, e.g. a knowledge graph that says values in a `Country` column must belong to the declared country vocabulary, so `"Stanford"` is invalid even though it's a perfectly well-formed string.

**Fixing**, not just finding, is the other half. Multiple tools exist along a spectrum from manual to automatic: interactive graphical wranglers (Trifacta Wrangler, OpenRefine) let a human see and fix; automatic systems detect-and-repair against the rules above at scale. In practice you start interactive on a sample to *discover* the rules, then codify them into an automatic check that runs on every future batch — cleaning is a pipeline stage, not a one-time scrub.

---

## 2. Data transformation — reshaping correct data

ML algorithms *prefer well-defined, fixed-length, well-conditioned, nicely-distributed input.* Cleaning made the data correct; transformation makes it digestible. The methods are per-modality.

### 2.1 Normalization for real-valued columns

Normalization makes training more stable — it conditions the optimization so gradients across features are comparable and the loss surface isn't a stretched ravine. Four standard maps:

| Method | Formula | Output range | Use when |
|---|---|---|---|
| **Min-max** | `x' = (x − min)/(max − min)·(b − a) + a` | training extrema map to `[a, b]`; later values may exceed it | You need a bounded range; distribution is roughly uniform; *but* sensitive to outliers (one extreme value squashes everything else) |
| **Z-score** | `x' = (x − mean)/std` | training mean 0/std 1 when nonconstant; unbounded | The default. Roughly Gaussian features; less outlier-sensitive than min-max |
| **Decimal scaling** | `x' = x / 10ʲ`, smallest `j` s.t. `max(|x'|) < 1` | `(−1, 1)` | Quick magnitude normalization |
| **Log scaling** | `x' = log(x)` | compresses tail | Heavy-tailed / multiplicative data (prices, counts, populations) |

Fit all learned transforms on training data. Min-max scaling maps the fitted extrema to the requested endpoints; a new value outside those extrema falls outside the interval unless explicitly clipped. Z-score uses mean and standard deviation and remains sensitive to outliers, as the [StandardScaler documentation](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html) explains. Use robust statistics when justified; handle constant columns and missing values explicitly. `log1p` accepts zero but still requires inputs greater than −1.

```python
# Fit these values on training data only; reuse them for later splits.
lo, hi = x_train.min(), x_train.max()
mean, std = x_train.mean(), x_train.std()
x_minmax = (x - lo) / (hi - lo if hi > lo else 1.0)
x_zscore = (x - mean) / (std if std > 0 else 1.0)
x_log = np.log1p(x)  # require x > -1; decide a domain rule before applying
```

> **Discipline:** fit normalization statistics (`min`, `max`, `mean`, `std`) on the **training split only**, then apply them to validation and test. Computing them over the full dataset leaks test information into training and is one of the most common silent evaluation bugs — a preview of Lecture 04.

### 2.2 Image transformations

Storage is the forcing function. CS329P's running example: scraping ~5M US home sales/year × ~20 images × ~153 KB at ~1041×732 is ~**15 TB/year**. Resize to ~320×224 and it drops to ~**1.4 TB** — a 10× cut — and the storage reduction is an estimate; measure whether lost spatial detail damages the task. The transforms:

- **Cropping, downsampling, compression** — shrink resolution and re-encode to save storage and load faster at training time.
- **Lossy compression awareness** — JPEG is not free. Medium (80–90%) JPEG compression can cost ~1% accuracy on ImageNet. Know your quality knob; don't compress your eval set into a different distribution than reality.
- **Image whitening** — a generalized normalization for vectors. Pixels in a local neighborhood are highly correlated; whitening removes that redundancy via a linear transform. With zero-mean column vectors and covariance `Σ`, choose `W` so `WΣWᵀ=I`. For `Σ=UΛUᵀ`, PCA whitening uses `W=Λ^(−1/2)Uᵀ`; regularize small eigenvalues before inversion. Common choices: the eigensystem of `Σ` (PCA whitening) or `Σ^(−1/2)` (ZCA whitening). Whether whitening improves training must be measured; it can amplify noise in low-variance directions.

### 2.3 Video transformations

Video's problem is input variability: movies run ~2 h, YouTube ~11 min, TikTok ~15 s. ML problems get tractable on **short clips (<10 s)**, ideally each a single coherent event (one human action) — semantic segmentation of long video into such events is extremely hard. The common pipeline trades storage against quality and load speed: **decode a playable clip, sample a sequence of frames, and compute spectrograms for the audio.** Frames-and-spectrograms are trivial to feed a model but cost more storage than the source video; that tradeoff is the whole game.

### 2.4 Text transformations

- **Stemming and lemmatization** — collapse a word to a common base form. `am, are, is → be`; `car, cars, car's, cars' → car`. Useful where surface form is noise, e.g. topic modeling. (Modern subword tokenizers often make this unnecessary — see the 2026 update.)
- **Tokenization** — split a string into the smallest unit the algorithm sees:

| Granularity | Method | Tradeoff |
|---|---|---|
| **By word** | `text.split(' ')` | Interpretable; huge vocab; chokes on OOV / typos |
| **By char** | `list(text)` | Tiny vocab, no OOV; very long sequences, weak units |
| **By subword** | learned vocab (WordPiece, Unigram, BPE) | Best of both — fixed vocab, graceful on rare words |

Subword is the modern default: `"a new gpu!"` → `"a", "new", "gp", "##u", "!"`, where the vocabulary is *learned from the corpus* (WordPiece/Unigram) so frequent words stay whole and rare ones decompose into known pieces.

---

## 3. Feature engineering — raw data → useful representation

A feature is a representation of raw data relevant to the target task. Before deep learning, feature engineering *was* the job: classical CV detected corners and interest points by hand and fed them to an SVM or softmax regression. Deep nets flipped this — a CNN learns the feature extractor end-to-end, so features become more relevant to the task than anything hand-designed, at the cost of being data-hungry and compute-heavy. The dividing line is modality: **for unstructured data (image/video/audio/text), learned features win and you should reach for them; for tabular data, hand-engineered features are still where accuracy comes from.**

### 3.1 Tabular features

- **Int / float** — use directly, or **bin** into `n` discrete buckets. Bucketing lets a linear model express nonlinearity (age 0–18, 18–35, 35–65, 65+ behave differently) and tames outliers by capping them into an edge bin.
- **Categorical → one-hot** — `cat → [0,1,0,0,0]`, `dog → [0,0,0,1,0]`. Map rare categories to a single `"Unknown"` bucket so you don't explode the dimension on values seen twice. When cardinality is huge (millions of user IDs), one-hot is infeasible — use the **hashing trick**: hash the category into a fixed number of buckets, accepting rare collisions for a bounded, fixed-width vector.
- **Datetime** — a single timestamp explodes into a feature list: `[year, month, day, day_of_year, week_of_year, day_of_week]`, plus is-weekend / is-holiday flags. Most temporal signal lives in these cyclic parts, not the raw epoch.
- **Feature crosses (combinations)** — the Cartesian product of two feature groups: `[cat, dog] × [male, female] → [(cat,male), (cat,female), (dog,male), (dog,female)]`. Crosses let a linear model capture interactions it otherwise can't (the effect of "rainy" depends on "weekend"). They blow up dimensionality fast, so cross deliberately and lean on hashing to bound the result.

```python
# Datetime explosion
ts = df["event_time"]
df["year"]        = ts.dt.year
df["month"]       = ts.dt.month
df["day_of_week"] = ts.dt.dayofweek
df["is_weekend"]  = (ts.dt.dayofweek >= 5).astype(int)
```

### 3.2 Text features

Working up from bag-of-counts to learned vectors:

- **Bag of words (BoW)** — represent text as token counts over a vocabulary. `"dog and cat and dinosaur"` → a count vector like `[0, 1, 2, 1, 1]` against vocab `[fish, cat, and, dog, dinosaur]`. Simple and strong baselines, but needs careful vocabulary design and *loses word context* — order is gone.
- **n-grams** — count contiguous sequences (bigrams, trigrams) instead of single tokens, recovering a little local order (`"not good"` becomes its own feature) at the cost of a much larger, sparser vocabulary.
- **TF-IDF** — reweight BoW counts by **term frequency × inverse document frequency** so that words common in *this* document but rare across the corpus score high, and ubiquitous words ("the") are damped. This is the standard upgrade over raw counts for classical text models.
- **Word embeddings (e.g. Word2vec)** — map each word to a dense vector so similar words sit close together; trained by predicting a target word from its context. Dense, low-dimensional, and they encode meaning rather than identity.
- **Pre-trained language models (BERT, GPT, universal sentence encoders)** — giant transformers trained on vast unannotated text. Use them two ways: pull out a text **embedding** as features, or **fine-tune** the whole model on your downstream task. Compare this route with a sparse baseline under the same task and resource budget.

| Text representation | Captures context? | Dimensionality | Where it shines |
|---|---|---|---|
| **BoW** | No | High, sparse | Fast baselines, interpretable |
| **n-grams** | Local only | Higher, sparser | Phrase signals (sentiment) |
| **TF-IDF** | No | High, sparse | Classical text classification / retrieval |
| **Word2vec** | Word-level | Low, dense | Similarity, pre-DL pipelines |
| **Pretrained LM** | Full | Dense, contextual | Context-sensitive tasks; evaluate quality and cost |

### 3.3 Image / video features

Traditionally you extracted images by hand-crafted descriptors like **SIFT** and fed them to a classifier. Now the default is a **pre-trained deep net as a frozen feature extractor**:

- **ResNet** — trained on ImageNet (image classification) — the workhorse image-feature backbone.
- **I3D** — trained on Kinetics (action classification) — for video.

Many off-the-shelf backbones exist; you rarely train a feature extractor from scratch. This is the on-ramp to **transfer learning** (Lecture 09), where reusing pretrained features *is* the workflow.

**Augmentation** — synthetically expand training data with label-preserving transforms: random crop, horizontal flip, color jitter, rotation, cutout/mixup for images; time-shift and noise for audio. Augmentation is half data-prep, half regularization — it teaches invariances (a flipped cat is a cat) and is one of the cheapest ways to buy generalization on limited data.

> **The headline:** feature *engineering* vs. feature *learning*. Prefer learning when it's available — images, video, audio, text. Hand-engineer when it's tabular. That single rule organizes most of this section.

---

## 4. Data summary — understanding distributions

Threaded through everything above is the discipline the lecture opens with: **exploratory data analysis.** Before you choose a normalizer or a bucketing scheme or decide a value is an outlier, you *look* — at per-column distributions, missing-value rates, cardinalities, correlations, class balance. The summary tells you which of the prior three sections you actually need:

- A heavy right tail says *log-scale before z-score*.
- A column that's 60% null says *impute or drop, don't one-hot the nulls*.
- A categorical with 10⁶ unique values says *hash, don't one-hot*.
- A 95/5 class imbalance changes your metric and validation scheme entirely (Lecture 04).

```python
df.describe(include="all")          # ranges, mean/std, top categories, counts
df.isna().mean().sort_values()      # per-column missing-value rate
df.nunique()                        # cardinality → one-hot vs. hashing decision
df["label"].value_counts(normalize=True)   # class balance
```

CS329P frames the whole data part as a flow chart that starts a project: *have enough data?* → if no, discover / augment / generate (Lecture 01); if yes, **preprocess** — EDA → cleaning → transformation → feature engineering — then train, evaluate, and iterate. The iteration loop ("improve label, data, or model?") usually sends you *back into this lecture*, not forward to a fancier model. The data is the lever.

---

> **Practice update, reviewed 2026-10-18:** learned representations and pretrained models are useful candidates, while sparse text features, geometry, and engineered tabular features remain valid baselines. A feature store can centralize definitions and point-in-time retrieval but does not automatically remove train/serve skew. Test timestamps, availability, transformations, and missing-value behavior. Choose preprocessing from measured task quality and resource costs, not a rule that one representation has made all others obsolete.

> **Hardware lens:** Preprocessing is not free setup — it is a throughput stage that competes with training for compute, and on a fast accelerator it is usually the bottleneck. The classic failure: a multi-GPU training run sits at 40% utilization because a handful of CPU cores can't JPEG-decode, resize, and augment images fast enough to feed it — the GPUs starve waiting on the input pipeline. Fixes are an engineering discipline of their own: prefetch and overlap (double-buffer the next batch while the GPU computes the current one), cache the decoded/resized tensors so you pay the cost once not every epoch, store in a sequential format (TFRecord / WebDataset / Parquet) so you read streams not millions of tiny files, and — the big one — **push decode/resize/augment onto the GPU itself** with NVIDIA **DALI**, which can turn a CPU-bound pipeline into a GPU-bound one and recover that idle utilization. The 15 TB → 1.4 TB resize from §2.2 is the same lesson at rest: smaller inputs mean less I/O, less decode, faster epochs. Measure input-pipeline throughput in samples/s alongside model FLOPs; a model that *could* train at 5,000 img/s but is fed at 1,200 img/s is a data-loading problem wearing a model costume.

---

## Current as of

Written June 2026. The **original CS329P content** — the three-way error taxonomy (outliers / rule / pattern violations), the four normalizers, the image/video/text transformation methods, and the tabular/text/image feature-engineering catalog — is taught first because it is still the correct working mental model and maps one-to-one onto the 2021 slides. The **refresh layer** flags what moved: the dominance of learned over hand-crafted features for unstructured data (while affirming feature engineering's continued primacy for tabular), the arrival of **feature stores** (Feast/Tecton) as standard infrastructure, the **data-centric AI** reframing, subword tokenization as universal, and the **DALI / GPU-side preprocessing** throughput story that the original slides — focused on storage cost — only gestured at. Where 2021 framing is dated, the original is presented before the update; nothing is silently rewritten.

*Adapted from [Stanford CS329P](https://c.d2l.ai/stanford-cs329p) — Huang, Li & Smola, CC-BY-SA-4.0.*
