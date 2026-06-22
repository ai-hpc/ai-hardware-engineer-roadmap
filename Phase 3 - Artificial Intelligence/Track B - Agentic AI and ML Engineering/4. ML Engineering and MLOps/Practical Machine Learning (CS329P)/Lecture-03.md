# Lecture 03 - ML Models Recap: Trees, Linear, and Neural Nets

**Collection:** [Practical Machine Learning (CS329P)](README.md) | **Previous:** [← Lecture 02](Lecture-02.md) | **Next:** [Lecture 04](Lecture-04.md)

---

The last two lectures got data into the building and turned it into clean, well-conditioned features. This lecture is the inventory of models you reach for once that work is done. It is deliberately *not* a from-scratch derivation course — there is no closed-form least-squares proof, no backpropagation chain rule, no convergence analysis. Those live in a theory class. This is the practitioner's working set: the four model families you will actually fit in a real ML role, and — more importantly — the decision of *when to reach for each one*. Knowing how to derive softmax matters far less in practice than knowing that a gradient-boosted tree will probably beat your neural net on the tabular dataset in front of you, and why.

CS329P frames every supervised model as three interchangeable parts: a **model** (a parameterized function from inputs to a prediction), a **loss** (how wrong a prediction is), and an **optimization** procedure (how you move the parameters to reduce the loss). Almost everything below is a different choice of those three parts bolted to the same scaffold. Trees swap out the optimizer for a greedy recursive split; linear methods and neural networks share mini-batch SGD and differ only in how expressive the model function is. Seeing the families this way — as variations on `model + loss + optimization` rather than as unrelated algorithms — is what lets you reason about a new method you have never seen by asking only those three questions.

The payoff is a single chart you can hold in your head: **tabular data → gradient-boosted trees; images → CNN; sequences and text → Transformer; everything small or interpretability-critical → linear.** The rest of this lecture earns that chart, and the closing decision table makes it concrete. Treat this as the map of the territory before Lecture 04 teaches you to *trust the score* a model gives you and Lecture 05 teaches you to *combine* several of them.

---

## Learning objectives

By the end of this lecture you should be able to:

1. **Decompose any supervised model** into its three parts — model function, loss, and optimization — and classify a problem as supervised, semi-/self-supervised, unsupervised, or reinforcement learning.
2. **Explain the iid assumption** and how generalization (test-set performance) is the actual goal, not training-set fit.
3. **Build up the tree family** from a single decision tree to Random Forest (bagging) to gradient-boosted trees, and state the pros (interpretable, native mixed/tabular data, little preprocessing) and cons (no smooth boundaries, unstable).
4. **Connect linear regression, softmax/logistic regression, and the MLP** as a single progression — a linear layer, then a classification head, then stacked linear-plus-nonlinear layers — and say what regularization and the linear decision boundary buy you.
5. **Match a neural architecture to a data structure** — MLP for vectors, CNN for images (locality + parameter sharing), RNN for sequences, Transformer for long-range dependencies.
6. **Pick the right model family for a given dataset** using a defensible decision rule, and reason about the very different hardware each family demands at inference.

---

## 1. The ML model landscape

### 1.1 Types of learning

Before models, the learning *setting* — what kind of supervision you have decides which algorithms are even on the table.

| Type | Trains on | Idea | Example |
|---|---|---|---|
| **Supervised** | Labeled data | Learn a map from inputs to known labels | Listing → sale price |
| **Semi-supervised** | Labeled + unlabeled | Use a model to infer labels for the unlabeled part, then learn from both | Self-training |
| **Unsupervised** | Unlabeled data | Find structure with no targets | Clustering, density estimation |
| **Self-supervised** | Unlabeled data | *Generate* labels from the data itself, so an unsupervised problem becomes a supervised one | word2vec, BERT (mask a word, predict it) |
| **Reinforcement learning** | Interaction | Take actions in an environment to maximize a reward signal | Game-playing, robotics |

Two of these blur on purpose. **Self-supervised learning** is the trick that unlocked modern NLP and much of vision: there are no human labels, but you can *design a supervised task* over raw data — hide a word and predict it (BERT), predict the next token (GPT), or generate fake samples with a trivially-known "fake" label (GANs). The crucial subtlety CS329P stresses is that *the training task can differ from how the model is ultimately evaluated or used*: you pre-train BERT on mask-filling, a task nobody cares about, precisely so the learned representations transfer to the task you do care about. That decoupling of training objective from end use is the engine of the foundation-model era (Lecture 09).

### 1.2 The three components of supervised training

Every supervised model in this lecture is the same scaffold with different parts swapped in:

```text
   Model         a parameterized function  f(x; θ)  mapping input x to a prediction
                   (parameters θ are learned; hyperparameters are set by you)
   Loss          a number measuring how bad one prediction is
                   (squared error, cross-entropy, contrastive, triplet, ranking, ...)
   Objective     the thing to minimize — usually the mean loss over all examples
   Optimization  the algorithm that adjusts θ to minimize the objective (SGD, greedy splits, ...)
```

The distinction between **model parameters** (weights `w`, bias `b` — learned from data) and **hyperparameters** (learning rate, tree depth, number of layers — set by you, tuned in Lecture 08) runs through the whole course. When you "train a model" you are running the optimization to find good parameters; when you "tune a model" you are searching over hyperparameters. Keep them separate in your head and most of ML stops being confusing.

The four supervised families CS329P names, by how the model function is built:

- **Decision trees** — make a prediction by walking a tree of yes/no questions.
- **Linear methods** — predict from a linear combination of input features.
- **Kernel machines** — compute feature *similarities* via a kernel function (the SVM lineage; we touch them only in passing).
- **Neural networks** — *learn* the feature representation rather than hand-craft it.

### 1.3 The iid assumption and generalization

Almost all of supervised learning rests on one quiet premise: training and future data are drawn **independently and identically distributed (iid)** from the same underlying distribution. *Identically* means the test world looks like the training world; *independently* means one example tells you nothing about the next. This is the assumption that makes a training-set average a sensible proxy for future performance — and Lectures 06–07 are entirely about what to do when it breaks (distribution shift; sequences and graphs where samples are *not* independent).

The goal is never to fit the training set — that is trivial; memorize it. The goal is **generalization**: low error on *unseen* data from the same distribution. A model that nails training and fails test has **overfit**; one that fails both has **underfit**. This gap between training fit and true performance is the single most important idea in practical ML, and Lecture 04 is devoted to measuring it honestly. Hold it in mind for every model below: expressiveness is a double-edged sword, because a model powerful enough to fit any pattern is also powerful enough to fit the noise.

---

## 2. Tree methods

Trees are the workhorse of tabular ML and the first family to reach for when your data lives in a spreadsheet.

### 2.1 The decision tree

A decision tree predicts by walking from the root down a series of yes/no questions until it reaches a leaf, which holds the answer. It handles **classification** ("has enough data?" → branch → "improve label?" → leaf) and **regression** ("is in Palo Alto?" → "living sqft > 2k?" → `price = $2.8M`) with the same structure — a leaf simply holds a class for classification or a number for regression.

Its standout virtues, straight from the slides:

- **Explainable.** The decision path *is* the explanation — you can read off exactly why a prediction was made, which matters enormously in regulated settings (credit, healthcare).
- **Handles numerical and categorical features together, with no preprocessing.** No normalization, no one-hot encoding, no scaling. A tree splits `sqft > 2000` and `city == "Palo Alto"` natively. After Lecture 02's preprocessing gauntlet, this is a genuine relief — trees are nearly immune to feature scale and monotone transforms.

Trees are built **top-down and greedily**. Start at the root with all examples and all features. At each node, pick the one feature-and-threshold split that best separates the examples, then recurse on each child. "Best" is measured by a split criterion:

| Criterion | Target type | Maximize |
|---|---|---|
| **Variance reduction** | Continuous | Drop in target variance after the split |
| **Information gain** (`1 − entropy`) | Categorical | Reduction in entropy (disorder) |
| **Gini impurity** (`1 − Σ pᵢ²`) | Categorical | Reduction in impurity |

All examples at a node participate in choosing its split. The tree stops when a node is pure, too small, or hits a depth limit.

### 2.2 Why a single tree is not enough

Two limitations push you past one tree, and they motivate everything that follows:

- **Trees overfit.** An unconstrained tree grows until each leaf is a single training example — perfect training accuracy, terrible generalization. You fight this by *limiting depth* (fewer levels of splitting) and *pruning* branches that don't pay their way on held-out data.
- **Trees are unstable.** Changing a handful of training examples can flip an early split and produce a completely different tree downstream. This high *variance* (Lecture 05's term) is the deep flaw of single trees — and the reason ensembles dominate.

One more practical wrinkle: tree *building* is **hard to parallelize** because each split depends on the one above it. This sequential dependence is exactly why the hardware story for trees looks so different from neural nets (see the Hardware lens).

### 2.3 Random Forest — bagging away the instability

The fix for instability is to train **many trees and average them**. A Random Forest trains a forest of decision trees and combines them — **majority vote** for classification, **average** for regression. The robustness comes entirely from *injected randomness*, from two sources:

- **Bagging (bootstrap aggregating).** Each tree trains on a bootstrap sample — draw `n` examples *with replacement*, so `[1,2,3,4,5]` might become `[1,2,2,3,4]`. Every tree sees a slightly different dataset, so their errors are partly independent and average out.
- **Random feature subsets.** At each split, consider only a random subset of features, which decorrelates the trees further so no single dominant feature makes them all the same.

Because the trees are independent, they **train in parallel** — recovering the parallelism a single tree lacked. Random Forest is the easy, robust default that turns the unstable single tree into something you can trust, and it needs very little tuning.

### 2.4 Gradient Boosting — the tabular champion

Bagging builds trees *in parallel* to reduce variance. **Boosting** builds them *sequentially*, each new tree correcting the errors of the ensemble so far — fitting the residual the current ensemble still gets wrong, then adding it in with a small learning rate. (The bias–variance contrast between bagging and boosting is Lecture 05's territory; here, just register that boosting is the other way to combine trees.)

In practice you almost never hand-roll this — you reach for one of two libraries, and they are among the most important tools in applied ML:

| Library | Edge | Use when |
|---|---|---|
| **XGBoost** | Battle-tested, regularized, huge ecosystem | The safe default for tabular competitions and production |
| **LightGBM** | Histogram-based splits, leaf-wise growth, faster on large data | Big tabular datasets where training speed matters |

**Gradient-boosted trees (GBT) are, as of 2026, still the best general-purpose model for tabular data** — they routinely beat deep nets on structured/spreadsheet problems, which is the single most important practical fact in this lecture. When the data is rows and columns, your first move is XGBoost or LightGBM, not a neural network.

```python
# Gradient-boosted trees: the tabular default
from xgboost import XGBClassifier
clf = XGBClassifier(
    n_estimators=300,      # number of sequential trees
    max_depth=6,           # shallow trees, boosted — controls overfitting
    learning_rate=0.05,    # shrinkage: each tree contributes a little
)
clf.fit(X_train, y_train)  # native mixed types, no scaling needed
```

### 2.5 Tree family — pros and cons

| Pros | Cons |
|---|---|
| Interpretable (single tree); feature-importance for ensembles | **No smooth decision boundaries** — predictions are axis-aligned staircases, bad for inherently smooth/continuous relationships |
| Native handling of mixed numerical + categorical features | A single tree is **unstable** (high variance) — fixed only by ensembling |
| Little to no preprocessing — scale-invariant, no normalization | Boosted trees can overfit if over-grown; need their own tuning |
| Excellent on tabular data; fast to train and tune | Don't natively handle unstructured data (pixels, raw text, audio) |

---

## 3. Linear methods

Linear methods are the simplest model family and the conceptual seed from which neural networks grow. Master this section and the next one is mostly bookkeeping.

### 3.1 Linear regression

Predict a real number as a weighted sum of features plus a bias. For a house with `x₁` beds, `x₂` baths, `x₃` living-sqft:

```text
   ŷ = w₁·x₁ + w₂·x₂ + w₃·x₃ + b
```

and in general, for input `x = [x₁, …, xₚ]`:

```text
   ŷ = ⟨w, x⟩ + b           (inner product of weights and features, plus bias)
```

The weights `w = [w₁, …, wₚ]` and bias `b` are the learned parameters. The loss is **mean squared error (MSE)** — average the squared gap between prediction and truth over `n` training examples:

```text
   w*, b*  =  argmin  (1/n) · Σᵢ (yᵢ − ⟨xᵢ, w⟩ − b)²
              w, b
```

MSE has a closed-form solution (a worthwhile exercise), but you rarely use it — for anything large, and for every model after this one, you optimize iteratively instead.

### 3.2 Mini-batch stochastic gradient descent

The optimizer that powers everything except trees. The recipe:

- Randomly initialize the weights.
- Repeat until convergence: sample a small random **mini-batch** of `b` examples, compute the gradient of the loss on just that batch, and step the parameters downhill: `w ← w − η · ∇w ℓ`.

Here `b` is the batch size and `η` the learning rate. **Pros:** it solves *every* objective in this course except trees — the same algorithm trains linear regression, softmax regression, and deep neural networks. **Cons:** it is sensitive to its hyperparameters; pick `b` or `η` badly and training diverges or crawls (Lecture 08 tunes them).

```python
# Mini-batch SGD for linear regression (the pattern behind all NN training)
w = torch.normal(0, 0.01, size=(p, 1), requires_grad=True)
b = torch.zeros(1, requires_grad=True)

for epoch in range(num_epochs):
    for X, y in data_iter(batch_size, features, labels):   # random mini-batches
        y_hat = X @ w + b
        loss = ((y_hat - y) ** 2 / 2).mean()               # MSE
        loss.backward()                                    # gradients
        for param in (w, b):
            param -= learning_rate * param.grad            # SGD step
            param.grad.zero_()
```

### 3.3 From regression to classification: softmax / logistic regression

To classify into `m` classes, first try the naive route: encode the label as a **one-hot** vector (`y = [0,…,1,…,0]`, a 1 in the true class's slot) and fit a linear model per class, `oᵢ = ⟨x, wᵢ⟩ + bᵢ`, training with MSE and predicting `argmaxᵢ oᵢ`. It works but **wastes model capacity** forcing the off-class outputs toward exactly 0 when all we care about is their *order*.

**Softmax regression** fixes this. Pass the raw scores `o` (the *logits*) through softmax to turn them into a probability distribution — non-negative, summing to 1:

```text
   ŷᵢ = softmax(o)ᵢ = exp(oᵢ) / Σₖ exp(oₖ)
```

Then train with **cross-entropy loss**, which compares the predicted distribution `ŷ` to the true distribution `y` and, for a one-hot label, simplifies to `−log ŷ_(true class)`:

```text
   H(y, ŷ) = − Σᵢ yᵢ · log ŷᵢ = − log ŷ_y
```

The elegance: cross-entropy only penalizes the *true* class's predicted probability, so the model stops wasting effort pushing wrong-class logits to any particular value as long as they stay below the right one. Crucially, **softmax regression is still a linear model** — the decision is made on a linear transformation of the input, since `argmaxᵢ ŷᵢ = argmaxᵢ oᵢ`. The softmax is just a smooth, differentiable stand-in for the non-differentiable argmax. The two-class special case of this is **logistic regression**, the most widely deployed classifier in industry.

```python
# Softmax classification head: a dense layer + cross-entropy
logits = X @ W + b                    # raw scores, shape (batch, num_classes)
loss = F.cross_entropy(logits, y)     # softmax + cross-entropy in one stable op
```

### 3.4 The linear decision boundary, and regularization

A linear classifier carves the input space with a single **straight hyperplane** — on one side it predicts class A, on the other class B. That is its power (simple, fast, interpretable — each weight is the signed importance of a feature) and its ceiling: data that isn't linearly separable (the classic XOR pattern) cannot be perfectly split by any line, no matter the weights. Breaking past that ceiling is exactly what §4 is about.

When features are many relative to examples, an unconstrained linear model overfits — it fits noise. **Regularization** reins it in by penalizing large weights, adding a term to the objective:

- **L2 (ridge):** add `λ‖w‖²` — shrinks weights smoothly toward zero, the standard stabilizer.
- **L1 (lasso):** add `λ‖w‖₁` — drives some weights to *exactly* zero, doing feature selection for free.

The knob `λ` trades training fit against weight size; it is a hyperparameter you tune (Lecture 08), and it is your first and cheapest defense against the overfitting that Lecture 04 teaches you to detect.

### 3.5 Linear methods → MLP

Here is the bridge to neural networks, and it is smaller than it looks. A **dense (fully-connected, linear) layer** with weight matrix `W ∈ ℝ^{m×n}` and bias `b ∈ ℝ^m` computes `y = Wx + b` — a vector of `m` linear combinations of the inputs. In this language:

- **Linear regression** = a dense layer with **1** output.
- **Softmax regression** = a dense layer with **m** outputs, followed by softmax.

Stack two dense layers and you get… still a linear model — the composition of two linear maps is one linear map, so nothing is gained. The missing ingredient is **nonlinearity**, and that single addition is what turns linear methods into neural networks.

---

## 4. Neural networks

The defining move of neural networks: instead of feeding *hand-crafted* features to a linear/softmax model, you let the network **learn the features** end-to-end. The price is more data and more computation; the prize is models that build their own representations and dominate every unstructured modality.

### 4.1 The multilayer perceptron (MLP)

Insert an elementwise **nonlinear activation** between dense layers and the composition stops collapsing. The standard activations:

```text
   sigmoid(x) = 1 / (1 + exp(−x))        # squashes to (0, 1)
   ReLU(x)    = max(x, 0)                 # the modern default — cheap, no saturation
```

An **MLP** stacks *hidden layers*, each a dense layer followed by an activation, then a final output layer. With even one hidden layer, the **universal approximation theorem** says an MLP can approximate any continuous function to arbitrary accuracy given enough hidden units — the formal statement of why "just add a hidden layer" works. Its hyperparameters are the number of hidden layers and the width (number of outputs) of each — your first taste of architecture design.

```python
# MLP with one hidden layer — linear, nonlinear, linear
W1 = nn.Parameter(torch.randn(num_inputs, num_hiddens) * 0.01)
b1 = nn.Parameter(torch.zeros(num_hiddens))
W2 = nn.Parameter(torch.randn(num_hiddens, num_outputs) * 0.01)
b2 = nn.Parameter(torch.zeros(num_outputs))

H = relu(X @ W1 + b1)       # hidden layer: dense + nonlinearity
Y = H @ W2 + b2             # output layer
```

### 4.2 Convolutional neural networks (CNN) — for images

An MLP on images is hopeless. A single hidden layer of 10K units on 300×300 ImageNet images needs **~1 billion parameters**, because "fully connected" means every output is a weighted sum over *every* input pixel. Too big to train, and it ignores everything we know about images. CNNs bake two pieces of prior knowledge into the architecture instead:

- **Translation invariance.** An object is the same object wherever it sits in the frame, so the same detector should apply everywhere.
- **Locality.** A pixel relates most to its near neighbors, not to one across the image.

A **convolution layer** encodes both. Each output is computed from a small `k × k` window of the input (**locality**), and *the same* `k × k` weight matrix — the **kernel** — slides across the whole image (**translation invariance**, via **parameter sharing**). The killer property: a conv layer's parameter count is just the kernel size and **does not depend on the input or output resolution** — the same `3×3` kernel works on any image size. That is how CNNs get the same modeling power as that 1-billion-parameter MLP with a tiny fraction of the parameters. A learned kernel becomes a pattern detector (an edge, a texture, a curve).

```python
# Single-channel 2D convolution: slide kernel K over input X
h, w = K.shape
Y = torch.zeros((X.shape[0] - h + 1, X.shape[1] - w + 1))
for i in range(Y.shape[0]):
    for j in range(Y.shape[1]):
        Y[i, j] = (X[i:i+h, j:j+w] * K).sum()    # same K reused everywhere
```

A **pooling layer** then takes the max or mean over small windows, shrinking the feature map and adding tolerance to small shifts. A real **CNN** stacks convolution → activation → pooling repeatedly to extract a hierarchy of features (edges → textures → parts → objects), ending in dense layers for the prediction. Modern CNNs — **AlexNet, VGG, Inception, ResNet, MobileNet** — are deep stacks of this pattern with various connectivity tricks (ResNet's skip connections are revisited in Lecture 08).

### 4.3 Recurrent neural networks (RNN) — for sequences

Language is a sequence: predict the next word from those before it (`hello` → `world`; `hello world` → `!`). A plain MLP handles sequences badly because it has no memory of what came earlier. An **RNN** adds a **hidden state** that is carried forward and updated at every time step, threading information through the sequence:

```text
   hₜ = ϕ(W_hh · hₜ₋₁ + W_hx · xₜ + b_h)
```

The *only* structural difference from an MLP is that term `W_hh · hₜ₋₁` — the hidden state from the previous step feeds into the current one, giving the network memory. **Gated RNNs (LSTM, GRU)** add learned gates for finer control of information flow — selectively *forgetting the input* or *forgetting the past* — which is what lets them carry signal across long sequences without it vanishing. RNNs also come **bidirectional** (read the sequence both ways, for tasks where future context is available) and **deep** (stack recurrent layers).

```python
# Simple RNN: carry hidden state H across time steps
H = torch.zeros(num_hiddens)
for X in inputs:                                  # inputs: (num_steps, batch, num_inputs)
    H = torch.tanh(X @ W_xh + H @ W_hh + b_h)     # update memory each step
    outputs.append(H)
```

### 4.4 Attention and the Transformer — a pointer forward

The RNN's sequential hidden state is also its weakness: it must be computed step by step (no parallelism across time, like the decision tree), and information from far back in a long sequence degrades as it is repeatedly overwritten. The **attention mechanism** solves both by letting every position look *directly* at every other position — no relaying through a hidden state — and doing it in parallel. The **Transformer** architecture, built entirely on attention, is now the backbone of essentially all of NLP, much of vision, audio, and the large language models this whole roadmap orbits. CS329P (and this course) develops the attention/Transformer toolkit properly in **[Lecture 08](Lecture-08.md)**; for now, register it as the fourth architecture and the one that displaced the RNN for sequences.

### 4.5 Which neural net for which data

| Architecture | Encodes | Use for |
|---|---|---|
| **MLP** | Universal function approximation over a fixed-length vector | Tabular vectors, the output head of bigger nets |
| **CNN** | Locality + translation invariance via parameter sharing | Images, audio spectrograms, video |
| **RNN / LSTM / GRU** | Sequential memory via a hidden state | Sequences when models must be small/streaming |
| **Transformer** | All-pairs attention, fully parallel | Text, long sequences, multimodal — the modern default ([L08](Lecture-08.md)) |

---

## 5. Which model when

The whole lecture compresses to one decision: match the model family to the *structure* of your data.

| Data type | First choice | Why | Fallback |
|---|---|---|---|
| **Tabular** (rows × columns, mixed types) | **Gradient-boosted trees** (XGBoost / LightGBM) | Native mixed types, no preprocessing, beats deep nets on structured data | Random Forest; linear/logistic for a fast interpretable baseline |
| **Images / video** | **CNN** (ResNet, MobileNet) or a vision Transformer | Locality + parameter sharing match image structure | Pretrained backbone as a feature extractor ([L09](Lecture-09.md)) |
| **Text / language** | **Transformer** (pretrained LLM) | All-pairs attention captures long-range dependence | Linear/logistic on TF-IDF for a cheap, strong baseline |
| **Time series / sequences** | **Transformer** or **RNN/LSTM** | Need sequential or long-range structure | GBT on lagged/windowed features is shockingly strong |
| **Audio** | **CNN** on spectrograms or audio **Transformer** | Spectrograms are images; attention for long context | — |
| **Small data / need interpretability** | **Linear / logistic regression** or a single tree | Few parameters resist overfitting; the model *is* its own explanation | Regularized linear; shallow tree |

Two rules of thumb sit on top of the table. First, **always fit a simple baseline before a complex model** — a regularized linear model or a Random Forest is fast, hard to get wrong, and tells you whether your problem is even tractable and whether the fancy model is earning its complexity (Lecture 04 makes this measurable). Second, **let the data's structure pick the family** before you tune anything: structure → family → model → hyperparameters, in that order.

> **2026 update:** The slide-era model selection chart aged remarkably well, with two big shifts. **(1) Transformers ate CV, NLP, and audio.** RNNs and LSTMs are now legacy for most new work — vision Transformers rival CNNs, and attention is the default for any sequence or text task. The chart's "RNNs for text/speech" box should read "Transformers" today. **(2) The foundation-model workflow inverted "pick and train a model."** For unstructured data you rarely train from scratch anymore — you take a pretrained foundation model (a vision backbone, an LLM) and fine-tune or prompt it (Lecture 09), turning model selection into *model selection-and-reuse*. The one box that did **not** change: **gradient-boosted trees are still the champion for tabular data.** Despite years of "deep learning for tabular" papers (TabNet, FT-Transformer, and others), XGBoost and LightGBM remain the first and usually best choice on structured data — the most durable practical result in this entire lecture.

> **Hardware lens:** the two model families demand opposite hardware, and this split echoes through the rest of the roadmap. **Tree inference is branchy and control-flow heavy** — evaluating a tree means a chain of data-dependent `if` comparisons that walk an irregular path down the tree, with poor memory locality and no big matrix to multiply. That is a **CPU** workload: branch predictors and large caches are exactly what it needs, and a GPU's thousands of lanes sit mostly idle on it. **Neural-net inference is the opposite — dense, regular GEMM** (general matrix multiply): every layer is a big matrix multiplication with no branching, the canonical job for the massively parallel ALUs of a **GPU, TPU, or NPU**, and the reason those accelerators exist. So the same accuracy can land on wildly different silicon: a gradient-boosted tree serving fraud decisions runs happily on a CPU at microsecond latency, while a Transformer of similar "size" wants a GPU and a batching system to be economical. This branchy-vs-GEMM divide is the foundation of the entire serving and compression stack in **[Phase 5 — ML Systems Engineering](../../../../Phase%205%20-%20Advanced%20Topics%20and%20Specialization/7.%20ML%20Systems%20Engineering/Guide.md)**, and the reason Lecture 10's compression techniques (quantization, pruning, distillation) target neural nets specifically — there is a dense GEMM to shrink. When you pick a model family, you are also picking a hardware target.

---

## Current as of

Written June 2026. The **original CS329P content** — the learning-type taxonomy, the `model + loss + optimization` decomposition, the tree progression (decision tree → Random Forest → boosting), the linear-to-softmax-to-MLP build-up, and the MLP/CNN/RNN architectures with their data-structure motivations — is taught first because it remains the correct working mental model and maps one-to-one onto the 2021 slides. The **refresh layer** flags what moved since: gradient-boosted trees' continued reign over tabular data, the displacement of RNNs by **Transformers** across CV/NLP/audio (with the proper treatment deferred to Lecture 08), the **foundation-model** workflow that turned "train a model" into "fine-tune or prompt a pretrained one" (Lecture 09), and the **branchy-CPU vs. dense-GEMM-accelerator** hardware split that the original slides did not address but that drives Phase 5. Where 2021 framing is dated, the original is presented before the update; nothing is silently rewritten.

*Adapted from [Stanford CS329P](https://c.d2l.ai/stanford-cs329p) — Huang, Li & Smola, CC-BY-SA-4.0.*
