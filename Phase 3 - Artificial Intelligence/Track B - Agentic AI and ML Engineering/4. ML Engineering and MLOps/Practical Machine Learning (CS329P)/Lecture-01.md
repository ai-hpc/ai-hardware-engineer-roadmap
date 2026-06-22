# Lecture 01 - Data I: Acquisition, Scraping & Labeling

**Collection:** [Practical Machine Learning (CS329P)](README.md) | **Previous:** [← Course index](README.md) | **Next:** [Lecture 02](Lecture-02.md)

---

CS329P opens with a deceptively simple promise: teach the machine-learning topics *that matter but are often skipped*. The skipped material is, overwhelmingly, the data. The course is organized as a spine — **Data → Model training → Deployment** — and it spends its first two lectures entirely on the first word, because that is where a working ML engineer spends most of their time. You can call `model.fit()` in one line. You cannot acquire, scrape, integrate, and label a dataset in one line, and no amount of architecture search will rescue a model trained on data you never properly assembled.

The collision this lecture is about is the gap between the academic mental model of ML and the industrial one. In the academic model the dataset is a given — `mnist`, `imagenet`, a CSV from the textbook — and the interesting work is the model. In the industrial model there *is* no dataset until you build one, and "building one" is a project that can involve multiple teams, a storage pipeline, legal review, and privacy controls before a single gradient is computed. This lecture walks the three ways data actually arrives: **find it** (existing datasets, benchmarks, competitions, raw data lakes), **harvest it** (web scraping at scale), and **label it** (semi-supervised learning, active learning, weak supervision, crowdsourcing).

A note on how this course teaches data: the **Exploratory Data Analysis** companion is a hands-on Jupyter notebook, not a lecture. Before you decide *how* to acquire more data, you load what you have, plot the distributions, look for the missing values and the value conflicts, and let the data tell you what it is. Everything below assumes you have already met your data through EDA and concluded you need more of it, cleaner labels, or both.

---

## Learning objectives

1. Decide between **finding** an existing dataset and **building** a new one, and weigh academic vs. competition vs. raw industrial data on the right axes (cleanliness, scale, realism, effort).
2. Plan a **web-scraping** job: choose between crawling and scraping, use a headless browser instead of `curl`, estimate cloud cost, and stay inside the legal and ToS guardrails.
3. Apply **semi-supervised learning** — specifically self-training with confidence thresholding — to exploit a small labeled set plus a large unlabeled pool.
4. Use **active learning** query strategies (uncertainty sampling, query-by-committee) to spend a labeling budget on the most informative examples.
5. Generate noisy labels at scale with **weak supervision / data programming** (labeling functions, à la Snorkel), and combine it with crowdsourcing under proper **quality control**.
6. Frame the whole decision as a flow chart — *have enough data? → external datasets? → generation? → labels? → budget? → weak labels?* — and know which tool each branch points to.

---

## 1. Data acquisition: find it, generate it, or build it

CS329P frames acquisition as a flow chart. You start an ML application and ask: **do I have enough data?** If not: **are there external datasets** I can discover or integrate? If not: **do I have a data-generation method**? Each "no" pushes you one box to the right and one notch up in cost.

### 1.1 Finding existing data

The cheapest data is data someone already collected. The course catalogs the canonical ML datasets and what each one actually *is* — provenance matters because it determines bias:

| Dataset | What it is | Modality |
|---|---|---|
| MNIST | Digits handwritten by US Census Bureau employees | Image |
| ImageNet | Millions of images scraped from image search engines | Image |
| AudioSet | YouTube sound clips for sound classification | Audio |
| LibriSpeech | ~1000 hours of English read from public-domain audiobooks | Speech |
| Kinetics | YouTube clips for human-action classification | Video |
| KITTI | Traffic scenes from car-mounted cameras + LiDAR | Multi-sensor |
| Amazon Review | Customer reviews from Amazon shopping | Text |
| SQuAD | Question–answer pairs derived from Wikipedia | Text |

Notice the pattern: most of the "found" datasets were themselves *scraped or crowdsourced* (ImageNet, Kinetics from search engines and YouTube; SQuAD from Wikipedia + MTurk). The line between "finding" and "building" is thinner than it looks.

Where to look, in roughly increasing order of rawness:

- **Papers With Code Datasets** — academic datasets with a leaderboard attached, so you know the state of the art before you start.
- **Kaggle Datasets** — datasets uploaded by data scientists, often with notebooks.
- **Google Dataset Search** — a search engine over datasets published anywhere on the web.
- **Framework hubs** — TensorFlow Datasets, Hugging Face `datasets` — one-line loaders.
- **Competitions** — Kaggle and company/conference ML competitions.
- **Open Data on AWS** — 100+ large-scale *raw* datasets.
- **Your own organization's data lake** — the most valuable and the messiest.

### 1.2 The build-vs-find tradeoff

The three classes of data trade off cleanliness against realism and effort:

| Source | Pros | Cons |
|---|---|---|
| **Academic datasets** | Clean, properly calibrated difficulty | Limited choices, over-simplified, usually small scale |
| **Competition datasets** | Closer to real ML applications | Still simplified; only exist for hot topics |
| **Raw data** | Total flexibility | Enormous effort to process |

The practical takeaway from the slides: **in industry you almost always deal with raw data**, and curating it is a *big project* — a processing pipeline, storage, legal review, and privacy handling, frequently spanning multiple teams. Academic datasets are training wheels; the job is raw data.

A specific raw-data skill the course calls out is **data integration** — combining multiple sources into one coherent dataset. Product data lives in multiple tables (a table for house attributes, one for sales, one for listing agents). You **join on keys**, which are usually entity IDs, and the recurring pain is identifying the right IDs, missing rows, redundant columns, and *value conflicts* where two sources disagree about the same field.

### 1.3 Generating data when none exists

If there is no dataset to find, the rightmost branch of the flow chart is **generate it**:

- **GANs** — synthesize realistic samples. The slides cite `thispersondoesnotexist.com` (synthetic faces) and synthetic furnished-room imagery (Gadde et al., ICCV'21).
- **Simulation** — render perfectly labeled data from a simulator, the dominant approach for rare events in autonomous driving (more below).
- **Data augmentation** — the everyday workhorse. Cheap label-preserving transforms multiply an existing labeled set: image augmentation (crop, flip, color jitter — e.g. the `imgaug` library) for vision, and **back-translation** (translate to another language and back) to paraphrase text.

The summary the course lands on: finding the right data is hard; raw industrial data is the norm; data integration stitches sources together; augmentation is standard practice; and **synthesizing data is getting popular**. That last line was prescient.

> **2026 update:** "synthesizing data is getting popular" became the **data-centric AI** movement and then the default. The 2021 toolkit was GANs + augmentation; in 2026 the dominant synthetic-data engine is the **LLM**. Teams generate instruction-tuning corpora, classifier training sets, and eval suites by prompting frontier models, then filter for quality — distillation of capability through data rather than weights. The practitioner's risk shifts from *not enough data* to **model collapse** (training on your own model's outputs until diversity decays) and **contamination** (synthetic or scraped text leaking your benchmark into the training set). When you generate data with an LLM, keep a human-verified seed set, measure diversity, and hold out a clean, provably-uncontaminated eval split.

---

## 2. Web scraping: data at scale when there is no API

When there is no dataset and no API, you **scrape**. The goal is to extract data from websites — it is noisy, the labels are weak and sometimes spammy, but it is available *at scale*, and many landmark datasets (ImageNet, Kinetics) were born this way. A price-comparison or price-tracking product is essentially a scraper with a UI.

First, the vocabulary distinction the course insists on:

- **Crawling** — indexing whole pages across the internet (what a search engine does).
- **Scraping** — extracting *particular fields* from the pages of a *specific* site (what you usually want).

### 2.1 Tools: why `curl` fails

The naive approach — `curl` the URL, parse the HTML — *often doesn't work*, because site owners deploy bot defenses. The standard answer is a **headless browser**: a real browser (Chromium) driven without a GUI, so it executes JavaScript and renders the page the way a human's browser would. You locate fields with the browser's **Inspect** tool, find the HTML element for each field (price, beds, square footage), and repeat per field.

Scraping at scale also needs **many IP addresses**, because a single IP hammering a site gets banned fast. The slides note you can rent IP diversity from the public clouds — of all IPv4 addresses, AWS owns ~1.75%, Azure ~0.55%, GCP ~0.25% — and when an instance's IP is banned you restart it to get a new one.

### 2.2 Case study and cost: Zillow houses

The worked example crawls houses sold near Stanford. The pattern generalizes: index pages list house **IDs**, paginated by a number in the URL (`.../sold/2-p/`); you collect IDs from the index, then fetch each **detail page** by ID (`.../homedetails/<zpid>/`) and extract fields by inspecting the HTML.

The economics are the point — scraping is *cheap on the cloud*:

```text
Instance:   AWS EC2 t3.small  (2 GB RAM, 2 vCPU, ~$0.02/hr)
            2 GB is required — the headless browser is memory-hungry;
            CPU and bandwidth are rarely the bottleneck.
Speed:      ~3 seconds per page
Scale:      crawl 1,000,000 houses  →  ~$16.6 in compute
            ~8.3 hours wall-clock with 100 instances in parallel
Extras:     storage + the cost of restarting instances on IP bans

Images:     a listing has ~20 images
            crawling all images: ~$300
            STORING them: ~$300 PER MONTH  ← storage, not compute, dominates
            mitigation: downscale resolution, or stream data back and discard
```

The non-obvious lesson: for image scraping, **storage, not crawling, is the recurring cost**. Compute is a one-time ~$300; holding the images is ~$300 *every month*.

### 2.3 robots.txt, politeness, and the law

Scraping responsibly is both an engineering and a legal discipline.

**Politeness (engineering).** Respect `robots.txt`, the file at a site's root that declares which paths bots may touch and, via `Crawl-delay`, how fast. Rate-limit yourself, identify your bot with a real `User-Agent`, back off on errors, and prefer an official API or data dump if one exists. A scraper that ignores these gets blocked — and deserves to.

**The law (the part that ends careers).** The course is blunt: web scraping *isn't illegal by itself*, **but**:

- Do **not** scrape data with **sensitive information** — credentials (username/password), personal health or medical records.
- Do **not** scrape **copyrighted** data — YouTube videos, Flickr photos, and the like.
- **Follow the Terms of Service.** If the ToS explicitly prohibits scraping, that prohibition is binding on you.
- If you are scraping **for profit, consult a lawyer.**

> **2026 update:** the legal terrain hardened sharply for *generative* AI. Post-2021 litigation (Authors Guild v. OpenAI, Getty v. Stability, NYT v. OpenAI) put scraped-training-data provenance on trial, and the EU AI Act now obliges general-purpose model providers to publish training-data summaries and respect machine-readable opt-outs. The practical rule for 2026: treat `robots.txt` and ToS as the *floor*, not the ceiling; record the **provenance and license** of every scraped item; and assume that "we scraped it, so we can train a commercial model on it" is a claim you may one day have to defend in court. CS329P's "consult a lawyer if you do it for profit" aged into a standing requirement.

---

## 3. Data labeling: turning data into labeled data

Scraping gets you *data*; supervised learning needs *labeled* data. The course frames labeling as another flow chart: **have data? → improve label/representation? → enough labels to start? → enough budget? → enough for weak labels?** Each branch points at a different technique — semi-supervised learning, crowdsourcing, or weak supervision.

### 3.1 Semi-supervised learning and self-training

**Semi-supervised learning (SSL)** targets the common case: a *small* labeled set plus a *large* unlabeled pool. It works by assuming something about the data distribution so the unlabeled points become useful:

- **Continuity assumption** — points with similar features likely share a label.
- **Cluster assumption** — the data has cluster structure; points in a cluster tend to share a label.
- **Manifold assumption** — the data lies on a manifold of far lower dimension than the input space.

The flagship SSL method is **self-training**, which bootstraps labels from the model itself:

```text
Self-training loop
  1. TRAIN   a model on the (small) labeled data.
             You may use expensive models here — deep nets, ensembles/bagging.
  2. PREDICT on the unlabeled data  →  pseudo-labels.
  3. KEEP    only the HIGH-CONFIDENCE predictions.
  4. MERGE   those pseudo-labeled points into the labeled set.
  5. Repeat.
```

The confidence threshold in step 3 is the whole game — keep everything and you amplify your own errors; keep only the confident predictions and each round adds genuine signal.

### 3.2 Active learning: spend the budget where it matters

**Active learning** is the same small-labeled / large-unlabeled scenario, but with a **human in the loop**. The contrast with self-training is exact and worth memorizing:

- **Self-training:** the model propagates labels to the data it is *most confident* about (cheap, automatic, low-risk).
- **Active learning:** the model selects the *most interesting / most uncertain* data and asks a **human** to label it (expensive, manual, high-information).

The two query strategies the course names:

| Strategy | Selects examples where… | Intuition |
|---|---|---|
| **Uncertainty sampling** | the prediction is least confident — the top class score is near random (≈ 1/n) | the model is on the fence, so a label resolves the most doubt |
| **Query-by-committee** | an ensemble of models *disagrees* | disagreement marks the regions the current hypothesis class hasn't pinned down |

In practice **active learning and self-training are combined**: train, predict on the unlabeled pool, auto-accept the *most* confident predictions as pseudo-labels (self-training), and route the *least* confident ones to human labelers (active learning). One pass through the unlabeled data feeds both the model and the annotation queue.

### 3.3 Crowdsourcing and quality control

When you need volume that a small team can't produce, you **crowdsource**. The canonical example is **ImageNet**, labeled across millions of images via **Amazon Mechanical Turk** — it took *years and millions of dollars*. SageMaker Ground Truth's rough MTurk price card shows why task design matters:

| Task | Estimated price |
|---|---|
| Image / text classification | $0.012 per label |
| Bounding box | $0.024 per box |
| Semantic segmentation | $0.84 per image |

Three labeling-team challenges recur:

- **Simplify the interaction** — easy tasks, clear instructions, a simple UI (MIT Places365 is the cited example of a well-designed instruction sheet). Complex jobs (e.g. labeling medical images) need *qualified* workers, not just any worker.
- **Cost** — total cost ≈ *#tasks × time-per-task*; reduce both. Active learning attacks `#tasks`; good UI attacks time-per-task.
- **Quality control** — labelers make mistakes, honest or not, and misread instructions. A bounding box comes back too big, too small, or around the wrong object.

The standard quality-control mechanism is **redundancy + majority voting**: send the same task to multiple labelers and take the majority label. It is the simplest method and the most expensive. The refinements: send *more* copies for the controversial examples, and **prune low-quality labelers** by tracking who disagrees with the consensus.

### 3.4 Weak supervision / data programming

The alternative to paying humans per label is to **generate labels programmatically**. **Weak supervision** (a.k.a. **data programming**, the idea popularized by **Snorkel**) semi-automatically produces labels that are *less accurate than manual ones but good enough to train on*:

- Encode **domain-specific heuristics** as **labeling functions** — keyword search, pattern matching, or calls to a third-party model.
- Example from the slides: rules to decide whether a YouTube comment is **spam** or **ham**.
- Each labeling function is noisy and may abstain; a label model reconciles their votes (and their estimated accuracies) into a single probabilistic training label.

The trade is explicit: you exchange a *little* label accuracy for *enormous* scale and the ability to relabel the entire dataset in seconds when your definition of the target changes.

### 3.5 Where it all goes: self-driving cars

The course closes labeling with the most demanding real example. **Tesla and Waymo both run large in-house labeling teams**, and the label types stack up: 2D and 3D bounding boxes, image semantic segmentation, 3D LiDAR point-cloud annotation, video annotation. The full toolkit appears at once:

- **Active learning** to find the scenarios that need more data and labels.
- **ML auto-labeling** to pre-label and let humans correct.
- **Simulation** to manufacture *perfectly labeled, unlimited* data for the rare and dangerous situations you can't safely collect on a real road.

The summary: the ways to get labels are **self-training** (iteratively label the unlabeled pool), **crowdsourcing** (global labelers, manual), and **data programming** (heuristic programs, noisy). And the meta-point — if labels are too expensive in every form, reconsider whether you need them at all: **unsupervised and self-supervised learning** sidestep the labeling problem entirely.

> **2026 update:** self-supervised pretraining won the framing the slide hints at. The default 2026 pipeline is *pretrain self-supervised on unlabeled data, then label only a small set for the downstream task* — exactly the SSL regime, but the "unlabeled" stage now does most of the work. Programmatic labeling generalized from Snorkel-style functions to **LLM-as-labeler**: prompt a strong model to annotate, treat its output as a (very capable) noisy labeling function, and reconcile it against cheap human spot-checks. The economics flipped — the bottleneck is no longer *getting* labels but *trusting* them, which is why quality control (redundancy, consensus, auditing the auto-labeler) is now the part of this lecture that scales worst and matters most.

---

## Current as of

The spine of this lecture — the acquisition flow chart, the dataset catalog and the academic-vs-competition-vs-raw tradeoff, data integration by key joins, GAN/simulation/augmentation generation, the Zillow scraping case study with its cost arithmetic, the robots.txt/ToS/legal cautions, and the labeling stack (self-training, active learning with uncertainty sampling and query-by-committee, crowdsourcing with majority-vote quality control, and Snorkel-style data programming) — is taught as the **original Stanford CS329P (2021 Fall)** material and tracks the slides directly. The **2026 refresh** layer flags what moved since: LLM-generated synthetic data and the data-centric-AI movement, the hardened legal landscape for generative-model training data (the EU AI Act and the 2023–2025 scraping litigation), and LLM-as-labeler programmatic annotation with its new trust-and-contamination failure modes. The Exploratory Data Analysis companion remains a hands-on notebook, not a lecture. Reviewed June 2026.

*Adapted from [Stanford CS329P](https://c.d2l.ai/stanford-cs329p) — Huang, Li & Smola, CC-BY-SA-4.0.*
