# Lecture 07 - Data Beyond IID: Sequences & Graphs

**Collection:** [Practical Machine Learning (CS329P)](README.md) | **Previous:** [← Lecture 06](Lecture-06.md) | **Next:** [Lecture 08](Lecture-08.md)

---

Almost every introductory ML result quietly assumes your data is **IID** — independent and identically distributed. That assumption is what lets you shuffle rows, carve out a random 20% test set, and trust the resulting accuracy number. The trouble is that most data worth modeling is *not* IID. Stock prices, sensor streams, words in a sentence, clicks in a user session, transactions on a payment network, friendships in a social graph — in every one of these, the value of one observation tells you something about its neighbors. The samples are *dependent*: `p(x, y) ≠ p(x) · p(y)`.

Dependence is not a nuisance to be removed — it is usually the entire signal. The reason we can forecast tomorrow from today, autocomplete a sentence, or flag a fraud ring is precisely *because* observations are correlated. The danger is pretending the dependence isn't there. A random train/test split on a time series leaks the future into the past and reports an accuracy you will never see in production. A random split on a social graph puts a user's friends on both sides of the split and lets the model memorize the answer. The model looks great offline and collapses live.

This lecture covers the two big families of dependent data — **sequences** (a 1-D chain of dependence through time or token position) and **graphs** (arbitrary relational dependence between entities) — plus the tests that tell you whether your data is dependent at all. The recurring theme: once data is non-IID, *both* your evaluation protocol *and* your model architecture have to change. Get the split wrong and no architecture saves you.

---

## Learning objectives

By the end of this lecture you should be able to:

- Explain what IID means, recognize common non-IID data, and state why dependence invalidates a random train/test split.
- Apply practical tests for independence (permutation/classifier test, MMD/HSIC, mutual information) and reason about what each measures.
- Frame a sequence problem autoregressively as `p(x_t | x_<t)` and choose between classical models, RNN/LSTM, and Transformers.
- Prepare sequence data without leaking the future: windowing, time-ordered splits, teacher forcing, and stationarity checks.
- Describe graph data (nodes, edges, features), node/edge/graph-level tasks, and the message-passing intuition behind GCN and GraphSAGE.
- Recognize where sequences and graphs show up in real systems and the hardware cost of serving each.

---

## 1. Independence tests & why IID breaks

### What "independent" buys you

Two random variables are independent when the joint factorizes into the product of marginals:

```text
Independent:   p(x, y) = p(x) · p(y)
Dependent:     p(x, y) ≠ p(x) · p(y)
```

Why do we care about the difference? Because supervised learning *lives* on dependence. Classification and regression both estimate `p(y | x) = p(x, y) / p(x)`. If `x` and `y` were truly independent, that whole expression would collapse to `p(y | x) = p(y)` — the input would tell you nothing, and there would be nothing to learn. Conversely, if observations you *assumed* were independent are actually dependent, your sampling and splitting logic is built on a false premise.

### Examples of non-IID data

| Data | Dependence structure | Correct split |
|------|----------------------|----------------|
| Stock / sensor time series | Each value correlated with its recent past | By time (train on past, test on future) |
| Text / language | Each token depends on prior tokens | By document, then within-doc left-to-right |
| User clickstream / sessions | Events within a session are correlated | By user / session, not by event |
| Medical records | Multiple rows per patient | By patient (group split) |
| Social network | Friends share labels (homophily) | By community / time, not by node |
| Molecules | Atoms bonded into structure | By scaffold / molecule |

### Why a random split lies

A random split assumes every row is an independent draw, so any partition gives an equally reliable estimate. Break that assumption and two failure modes appear:

- **Temporal leakage.** Shuffle a time series and rows from *after* the test point land in the training set. The model effectively peeks at the future. Offline accuracy soars; live accuracy does not, because at inference time the future genuinely does not exist yet.
- **Group/entity leakage.** Put the same patient, user, or social neighborhood on both sides of the split and the model memorizes entity-specific quirks instead of learning a generalizable rule. It is graded on data it has effectively already seen.

The rule of thumb: **split along the axis of dependence.** Time-ordered data splits by time; grouped data splits by group; graph data splits by community or by a time cut. Never let information cross the train/test boundary that wouldn't be available at prediction time.

### Testing for dependence

You often *suspect* dependence but want to confirm it. Several tests, in rough order of sophistication:

- **Permutation / classifier test.** Take your real paired data `Z = {(x_i, y_i)}`. Build a shuffled copy `Z' = {(x_i, y_π(i))}` where `π` randomly permutes the labels — this destroys any real `x`–`y` relationship while preserving each marginal. Train a classifier to tell `Z` from `Z'`. If it can, the pairing carries information and `x`, `y` are dependent. (Bonus: the pairs the classifier is *most* confident about are your most strongly related ones.) The same `f(x, y)` that scores "is this a real pair" can even be turned into a predictor via `ŷ = argmax_y f(x, y)`.
- **MMD (Maximum Mean Discrepancy).** Compare the joint expectation `E_{(x,y)}[φ(x)·φ(y)]` against the independent expectation `E_x E_y[φ(x)·φ(y)]` in a kernel feature space. A large gap means dependence. Useful when feature maps don't factorize.
- **HSIC (Hilbert-Schmidt Independence Criterion).** A kernel covariance operator that vanishes exactly when `x ⊥ y`. It reduces to the clean trace form `tr(H K H L)` with centering matrix `H_ij = δ_ij − 1/m` and kernel matrices `K`, `L` over `x` and `y`. The de-facto modern nonparametric independence test.
- **Mutual information.** From information theory, `I(x, y) = H[x] + H[y] − H[(x, y)]` equals the KL divergence between the joint and the product of marginals — it counts the *extra bits* needed to encode `x` and `y` separately rather than jointly. If the data is independent, `I = 0` and no bits can be saved.

```python
# Permutation (classifier) independence test — the practical workhorse.
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import cross_val_score

def independence_pvalue(X, y, n_perms=200):
    real = np.column_stack([X, y])              # genuine (x, y) pairs
    def auc(Z):                                 # can a classifier spot real vs shuffled?
        Zp = np.column_stack([X, np.random.permutation(y)])
        data = np.vstack([Z, Zp])
        lab  = np.r_[np.ones(len(Z)), np.zeros(len(Zp))]
        return cross_val_score(GradientBoostingClassifier(), data, lab,
                               cv=3, scoring="roc_auc").mean()
    observed = auc(real)
    null = [auc(np.column_stack([X, np.random.permutation(y)]))   # AUC under H0
            for _ in range(n_perms)]
    # p = fraction of null AUCs at least as separable as observed
    return float((np.sum(np.asarray(null) >= observed) + 1) / (n_perms + 1))
# p small  -> reject independence -> data is dependent -> do NOT random-split.
```

---

## 2. Sequence models

### The autoregressive framing

A sequence is observations `x_1, x_2, …, x_T` where order matters. By the chain rule of probability, the joint *always* factorizes:

```text
p(x_1, …, x_T) = p(x_1) · p(x_2 | x_1) · p(x_3 | x_1, x_2) · … · p(x_T | x_1, …, x_{T-1})
```

This is the **autoregressive** view: predict each step from everything before it, `x_t ∼ p(x_t | x_<t)`. Because of causality, decomposing *forward* in time is more accurate than the (mathematically valid) backward decomposition — the past genuinely generates the future, not the reverse.

Conditioning on the *entire* past is expensive and often unnecessary. Two standard simplifications:

- **Windowing (Markov assumption).** Condition only on the last `τ` steps: `p(x_t | x_{t-τ}, …, x_{t-1})`. Takens' theorem says that, under mild regularity, a finite window of the recent past carries enough information. You then train an ordinary regressor with target `x_t` and features `(x_{t-τ}, …, x_{t-1})`. This relies on **stationarity** — the assumption that the conditional distribution depends on the *values* in the window, not on the absolute time `t`.
- **Latent state.** When a long history matters, summarize it into a hidden state instead of carrying the raw past: `h_t = g(x_{t-1}, h_{t-1})`, `x_t = f(x_{t-1}, h_t)`. The state `h_t` is a learned, fixed-size memory of everything that came before.

### The model ladder

| Model | History mechanism | Strengths | Weaknesses |
|-------|-------------------|-----------|------------|
| AR / ARIMA | Fixed linear window | Simple, interpretable, strong baseline | Linear; assumes stationarity |
| Markov chain | Last `k` discrete states | Cheap, transparent | State space explodes with `k` |
| RNN | Recurrent hidden state | Unbounded context in principle | Vanishing gradients; slow (sequential) |
| LSTM / GRU | Gated memory cell | Keeps long-range signal | Still sequential; truncated BPTT |
| Transformer | Self-attention over all positions | Parallel training, long range, SOTA | O(T²) attention; large memory |

The classical end (ARIMA, Markov) is the place to *start* — it gives a baseline you must beat. **RNNs** add a learned state but train slowly because each step depends on the previous one. **LSTM** (Hochreiter & Schmidhuber, '97) mimics a circuit memory cell with input/forget/output gates so gradients survive long chains; **GRU** (Cho et al., '14) is a cheaper, slightly weaker variant. In practice, deep-and-simple beats shallow-and-complex, training is expensive (backprop through a long chain), and frameworks truncate the gradient by default.

```text
LSTM cell (the gated memory device):
  i_t = σ(W_i · [x_t, h_{t-1}] + b_i)              # input gate
  f_t = σ(W_f · [x_t, h_{t-1}] + b_f)              # forget gate
  o_t = σ(W_o · [x_t, h_{t-1}] + b_o)              # output gate
  c_t = f_t ⊙ c_{t-1} + i_t ⊙ tanh(W_c·[x_t,h_{t-1}] + b_c)   # cell state
  h_t = o_t ⊙ tanh(c_t)                            # hidden state
```

**Transformers** replaced the recurrence with attention. Seq2seq (Sutskever et al., '14) first encoded a source sentence into one fixed vector `φ(s)` with an LSTM and decoded it token by token — which broke on long inputs because a single vector isn't rich enough ("The table is round" worked; a long paragraph produced "Error…"). Adding **attention** (Bahdanau et al., '14) let the decoder look back at any source position via query/key/value weights `α_ij ∝ exp(a(h̃_{i-1}, h_j))`, removing the bottleneck. Dropping the recurrence entirely gave the Transformer: parallel training, natural bidirectional context for embeddings, and the foundation of every modern sequence model (covered in depth in Lecture 10++).

### Sequence tasks

- **Forecasting** — predict future values `x_{t+1}, …, x_T` from the past (demand, load, prices).
- **Tagging / labeling** — one output per input position (part-of-speech, named-entity recognition, anomaly flags).
- **Seq2seq** — map an input sequence to an output sequence of different length (translation, summarization, speech-to-text).

### The data-prep gotcha: don't train on the future to predict the past

This is where most sequence projects fail before the model is even chosen.

- **No shuffling across time.** With IID data you random-partition and every fold is equally reliable. With dependent data you must respect order: **train on `x_1, …, x_t`, evaluate on `x_{t+1}, …, x_T`.** Shuffling leaks the future.
- **Windowing without leakage.** Build `(features, target)` pairs from a sliding window, and make sure every feature timestamp is strictly *before* the target. Scalers, encoders, and imputers must be fit on the training window only — fitting on the full series leaks future statistics.
- **Watch stationarity and drift.** Using all past history assumes stationarity. Real data drifts: **concept shift** (COVID shifted spending from dining out to durable goods), **seasonality** (Christmas every year), and one-off **nonstationarity** (a new product launch). Sometimes the drift has an external cause — condition on it (umbrella sales given weather) and the residual becomes closer to independent.
- **Teacher vs. student forcing.** During training, feed the *true* previous token (teacher forcing) so the model is always one step from ground truth. At inference it must consume its *own* predictions (student forcing), and small errors compound — autoregressive rollouts can diverge rapidly. Scheduled sampling, and generative training (GAN/VAE-style objectives), help close this train/serve gap.

```python
# Time-ordered split + windowing — never shuffle a time series.
import numpy as np

def make_windows(series, tau, horizon=1):
    X, y = [], []
    for t in range(tau, len(series) - horizon + 1):
        X.append(series[t - tau:t])          # features: strictly the past
        y.append(series[t + horizon - 1])    # target:   a future step
    return np.array(X), np.array(y)

cut = int(len(series) * 0.8)                 # chronological cut, NOT random
train, test = series[:cut], series[cut:]     # past trains, future tests
mu, sd = train.mean(), train.std()           # fit scaler on TRAIN ONLY
train, test = (train - mu) / sd, (test - mu) / sd   # no future leakage
Xtr, ytr = make_windows(train, tau=24)
Xte, yte = make_windows(test,  tau=24)
```

---

## 3. Graphs

Some dependence simply cannot be flattened into a sequence. A road network, a payment graph, a citation web, a "who-follows-whom" social network — these have arbitrary relational structure. The clean factorizations are graphical models — directed `p(x) = ∏_i p(x_i | x_{π(i)})` or undirected `p(x) = ∏_C ψ_C(x_C)` over cliques — but exact inference on them is often intractable, so modern practice learns vertex representations directly.

### Graph data

A graph is `G(V, E)`:

- **Vertices (nodes)** `i ∈ V`, each optionally with a feature vector `x_i`.
- **Edges** `(i, j) ∈ E`, each optionally with a feature vector `x_ij`.

### Task levels

| Level | Question | Examples |
|-------|----------|----------|
| Node | Label the unknown vertices | Fraud detection, user classification |
| Edge | Predict missing / future edges | Link prediction, recommendation |
| Graph | One label for the whole graph | Molecular property, toxicity |

A classic node task: given labels on *some* vertices, infer the rest (fraud). A classic edge task: given some edge attributes, predict the missing ones (link recommendation).

### Message passing — the GNN intuition

The core idea predates deep learning. The **Weisfeiler-Lehman** algorithm (1976) makes vertices unique by *repeatedly hashing each vertex together with its neighbors* until the labels stabilize — and those hashes turn out to be excellent structural features. **PageRank** (Page & Brin, '90s) does the same shape of computation: each page's score is a function of its neighbors' scores, iterated to a fixed point. Both are **local update** rules: a node's new value is a function of its neighborhood.

**Graph Neural Networks** make that update *learnable*. Each layer, every node:

1. **collects** feature vectors from its neighbors (the *message*),
2. **aggregates** them with a permutation-invariant function — sum, mean, or max (a "function on a set"),
3. **updates** its own representation by combining the aggregate with its current state, then applies a learned transform.

```text
Message passing, one GNN layer:
  m_i      = AGGREGATE_{j ∈ N(i)}  message(h_j, x_ij)     # gather from neighbors
  h_i'     = UPDATE(h_i, m_i)                              # combine + learned transform

After k layers, each node "sees" its k-hop neighborhood.
```

Stack `k` layers and information flows `k` hops: a node's representation absorbs its entire `k`-hop neighborhood. That is exactly how a GNN turns raw node features plus graph structure into embeddings you can classify or use for link prediction.

### GCN and GraphSAGE

- **GCN** (Kipf & Welling, 2016) — Graph *Convolutional* Network. Each layer aggregates neighbor features with a **degree-normalized sum**, applies a linear map and a nonlinearity. It is the canonical message-passing GNN: simple, strong, and the standard baseline.
- **GraphSAGE** (Hamilton et al., 2017) — **SA**mple-and-aggrer**G**at**E**. Instead of using the *whole* neighborhood (impossible on a billion-edge graph), it **samples a fixed number of neighbors** per node and learns an aggregator (mean / pooling / LSTM). Sampling makes it scale and enables **inductive** prediction on nodes never seen at training time — essential in production where the graph keeps growing.
- **GAT** (Veličković et al., 2017) — **G**raph **AT**tention. Weights each neighbor with learned attention `α_ij` instead of a fixed normalization, so the most relevant neighbors dominate.

```python
# A GCN layer in PyTorch Geometric — message passing in a few lines.
import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv

class GCN(torch.nn.Module):
    def __init__(self, in_feats, hidden, num_classes):
        super().__init__()
        self.conv1 = GCNConv(in_feats, hidden)        # aggregate 1-hop neighborhood
        self.conv2 = GCNConv(hidden, num_classes)     # aggregate 2-hop neighborhood

    def forward(self, x, edge_index):
        x = F.relu(self.conv1(x, edge_index))         # gather + transform + nonlinearity
        return self.conv2(x, edge_index)              # logits per node

# Semi-supervised: loss is computed on labeled nodes only; the graph
# propagates that signal to the unlabeled ones via message passing.
```

A practical caveat (Dai et al., 2018): backprop through a whole graph is expensive — "six degrees of separation" means a few hops touch the entire graph, and BPTT-over-graph blows up. Fixes include fixed-point iteration in place of deep unrolling and **sampling a small subset of vertex updates** per step (the GraphSAGE trick).

### Where graphs appear

- **Recommenders** — users, items, and vendors as a bipartite/heterogeneous graph; recommendation is edge prediction.
- **Fraud / anti-abuse** — fraud rings show up as suspicious subgraphs; node classification with strong relational signal.
- **Molecules & materials** — atoms as nodes, bonds as edges; graph-level property prediction.
- **Social & information networks** — meme/fake-news spread, influence, and community detection.

> **Hardware lens:** the two data families stress hardware in opposite ways. **Sequence inference is memory-bandwidth-bound.** An autoregressive decoder generates one token at a time and re-reads the **KV cache** — the stored keys/values for every prior token — at every step, so throughput is gated by how fast you can stream that cache out of HBM, not by raw FLOPs. The cache grows linearly with context length and dominates memory. **GNNs are sparse, irregular-memory workloads.** Gathering neighbor features is scattered, data-dependent indexing into large vertex tables — the antithesis of the dense, regular matmuls GPUs love. Performance is bound by random-access memory bandwidth and poor cache locality; mini-batch neighbor sampling (GraphSAGE) exists partly to tame this. Both motivate the hardware-software co-design — KV-cache paging, quantization, and sparse-gather kernels — covered in [Phase 5 — MLSys Deep Dives](../../../../Phase%205%20-%20Advanced%20Topics%20and%20Specialization/7.%20ML%20Systems%20Engineering/MLSys%20Deep%20Dives/README.md).

> **2026 update:** Transformers have decisively won general sequence modeling — RNNs and LSTMs are now mostly legacy or edge/streaming niches. For *very long* context, **state-space models (Mamba / S4 and successors)** offer linear-time, constant-memory alternatives to quadratic attention and ship in hybrid attention-SSM stacks. Classical **ARIMA/Prophet still win on small, clean, low-frequency forecasts** where a Transformer would overfit. **GNNs remain a specialist tool** — but a strong one — dominating molecular/materials ML, drug discovery, recommender retrieval, and fraud/anti-abuse, with **temporal graph networks (TGN-style)** handling graphs that evolve over time (the natural fusion of this lecture's two halves). The IID-breaking lessons — split by time/entity, never leak the future — are unchanged and remain the single most common cause of "great offline, broken in production."

---

## Current as of

June 2026. Model recommendations (Transformers, SSMs/Mamba for long context, GCN/GraphSAGE/GAT, temporal graph nets) reflect the landscape as of this date; the independence and leakage principles are timeless.

*Adapted from [Stanford CS329P](https://c.d2l.ai/stanford-cs329p) — Huang, Li & Smola, CC-BY-SA-4.0.*
