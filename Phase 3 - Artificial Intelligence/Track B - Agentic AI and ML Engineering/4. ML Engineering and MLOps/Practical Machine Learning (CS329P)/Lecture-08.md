# Lecture 08 - Model & Hyperparameter Tuning: HPO, NAS, Deep-Net Tuning

**Collection:** [Practical Machine Learning (CS329P)](README.md) | **Previous:** [← Lecture 07](Lecture-07.md) | **Next:** [Lecture 09](Lecture-09.md)

---

Tuning is where compute meets diminishing returns. You have a model that fits, a validation scheme that you trust (Lectures 04–05), and now a knob-filled console: learning rate, batch size, weight decay, layer counts, channel widths, optimizer choice, dropout, warmup schedule. Each knob has a plausible range, the ranges multiply into a search space that is exponential in the number of knobs, and every point you evaluate costs a full training run. The discipline of this lecture is *not* "find the best hyperparameters" — that's the goal, not the skill. The skill is **spending a fixed search budget wisely**, and knowing in advance which family of changes — architecture, hyperparameters, or training tricks — is actually going to move the needle for your problem, so you don't burn a thousand GPU-hours discovering that your learning rate was already fine.

CS329P frames this through a cost curve that has only sharpened since 2021: **compute cost per trial falls exponentially, human cost rises.** A data scientist costs >\$500/day; a trial on a typical task is minutes-to-an-hour of CPU/GPU time costing cents to a few dollars. The moment an automated tuner beats a human after ~1000 trials — and a decent one beats roughly 90% of data scientists — the economics say automate the search and spend the human on the parts a search can't do: framing the objective, designing the search space, and reading the results. This is the entire premise of **AutoML**: automate every step of applying ML — data cleaning, feature extraction, model selection — with **Hyperparameter Optimization (HPO)** and **Neural Architecture Search (NAS)** as its two best-developed pillars.

We teach this in three movements and then a fourth. HPO: the search algorithms, from the embarrassingly simple to the genuinely clever. NAS: searching the *architecture itself*, its brief moment of glory and its honest decline. The deep-network tuning toolkit: the three structural inventions — normalization, residuals, attention — that did more for deep nets than any tuner ever could, because they changed what was *trainable* in the first place. And throughout, the 2026 reality check: the field re-sorted itself, and the right default today is not what it was when these slides were written.

---

## Learning objectives

By the end of this lecture you should be able to:

1. **Classify hyperparameters** and decide what to tune manually, what to automate, and what to leave at a well-chosen baseline.
2. **Choose an HPO algorithm** — grid, random, Bayesian optimization, Hyperband/ASHA, BOHB — and justify it from your budget and parallelism, not from habit.
3. **Explain why random search beats grid search**, and why multi-fidelity methods beat both when most configurations are bad.
4. **Describe the NAS pipeline** — search space, search strategy, performance estimation — and explain the cost problem that made naive NAS (~2000 GPU-days) impractical and one-shot methods necessary.
5. **Contrast batch norm and layer norm**, explain why LN won for transformers, and explain how residual connections keep very deep nets trainable.
6. **Reason about hardware-aware tuning** — latency-targeted NAS, the effect of norms/residuals on kernel fusion — and state honestly where NAS sits in 2026.

---

## 1. Model tuning overview

### 1.1 What you are actually tuning

Hyperparameters are the values *you* set before training, as opposed to parameters the data fills in by gradient descent. They split into a few rough families, and the family tells you how to treat each one:

| Family | Examples | Character |
|---|---|---|
| **Optimization** | learning rate, batch size, momentum, weight decay, warmup/schedule | Continuous or log-continuous; usually the highest-leverage knobs |
| **Architecture** | #layers, #channels/width, kernel size, #hidden units, #attention heads | Discrete/categorical; expensive to change (full retrain) |
| **Regularization** | dropout rate, label smoothing, augmentation strength | Continuous; interacts strongly with dataset size |
| **Capacity-vs-cost** | model size, sequence length, resolution | Sets the cost of *every other* trial |

The practical first move from CS329P is **start from a good baseline** — default settings from a high-quality toolkit, or values reported in a paper on a similar task — and tune *relative to it*. Defaults exist because someone already spent the budget; you inherit it for free. Then tune one value, retrain, observe the change, and repeat. The point of the repetition is not just to find a better value; it's to build **insight**: which hyperparameters actually matter, how sensitive the model is to each, and what the good ranges are. That insight is what you carry to the next project, and it's the thing a tuner can't hand you.

### 1.2 Manual vs automated

Manual tuning is a human running that loop by hand. It works, it builds intuition, and for a 2–3 knob problem it's often fastest. Its failure mode is **experiment management**: after fifty runs you cannot remember which learning rate produced which curve. CS329P is blunt about the fix — *save your training logs and hyperparameters so you can compare, share, and reproduce.* The simplest version is logs in text and key metrics in a spreadsheet; better options are TensorBoard and Weights & Biases (and in 2026, MLflow). The discipline matters more than the tool.

Automated tuning hands the loop to an algorithm. You pay for it in compute and in the upfront work of specifying a search space, and you get back coverage, consistency, and the ability to run a hundred trials overnight without a human in the seat. The decision between them is the cost curve from the intro: automate when compute is cheaper than the data-scientist-hours the search would otherwise consume.

### 1.3 Reproducibility is genuinely hard

Reproducing a result is harder than it sounds, because it depends on three things that all drift:

- **Environment** — hardware and library versions. A different GPU, a different cuDNN, a different PyTorch can change numerics enough to move the last point of accuracy.
- **Code** — the exact code path, including the preprocessing you forgot was non-deterministic.
- **Randomness** — the seed. Weight init, shuffling, dropout, and augmentation are all stochastic; without a pinned seed, "the same run" isn't.

For tuning specifically this has a sharp consequence: **fair comparison demands you hold the nuisance variables fixed.** If config A ran on a different seed, a different machine, or a longer schedule than config B, the difference between them is contaminated and you may "tune" your way to a worse model. Pin the seed, pin the environment, give every config the same budget, and compare on the same held-out split — otherwise the search is measuring noise.

### 1.4 The cost of tuning

Every trial is a full (or partial) training run, and the search space is exponential in the number of hyperparameters. This is the **curse of dimensionality** stated as a budget problem: you cannot afford to evaluate the grid, so the whole game is *getting more signal per trial-dollar.* Everything in Section 2 is an answer to that single pressure — either evaluate fewer points (smarter sampling) or evaluate each point more cheaply (lower fidelity).

---

## 2. HPO algorithms

### 2.1 Defining the search space

Before any algorithm runs you must specify a **range for each hyperparameter** — `lr ∈ [1e-5, 1e-1]` on a log scale, `batch_size ∈ {32, 64, 128, 256}`, `optimizer ∈ {sgd, adam, adamw}`. The space can be exponentially large, so **designing it well is itself a tuning decision**: a log scale for learning rate, sane bounds that exclude regimes you know diverge, and dropping knobs you've already learned don't matter. A tight, well-shaped space makes a mediocre algorithm look good; a sloppy one defeats a great algorithm.

### 2.2 Black-box vs multi-fidelity

Two philosophies divide every HPO method:

- **Black-box** treats each training job as an opaque function: you hand it a config, it runs to completion, it returns a score. Grid search, random search, and Bayesian optimization are black-box.
- **Multi-fidelity** *modifies* the training job to get a cheap, noisy estimate of how good a config will be, so you can kill bad ones early. The standard cheapening moves: train on a subsampled dataset, shrink the model (fewer layers/channels), or — most importantly — **stop a bad configuration early** instead of running it to convergence. Successive Halving and Hyperband are multi-fidelity.

The multi-fidelity bet is that **most configurations are bad and reveal it quickly**, so spending full budget on them is waste.

### 2.3 Grid search

Evaluate every combination in the space.

```python
def grid_search(search_space):
    best = None
    for config in search_space:          # every point on the grid
        score = train_and_eval(config)
        best = better_of(best, score, config)
    return best
```

It guarantees you find the best point *in the grid* and it's trivially parallel. It also dies to the curse of dimensionality: a 5-value grid over 6 hyperparameters is 15,625 trials, and most of them vary a knob the model doesn't care about while pinning the one it does.

### 2.4 Random search — and why it beats grid

Sample `n` configurations uniformly at random from the space.

```python
def random_search(search_space, n):
    best = None
    for _ in range(n):
        config = random_select(search_space)   # sample, don't enumerate
        score = train_and_eval(config)
        best = better_of(best, score, config)
    return best
```

The counterintuitive result — established by Bergstra & Bengio, *Random Search for Hyper-Parameter Optimization* — is that random search is **more efficient than grid search, both empirically and in theory.** The reason is the *low effective dimensionality* of most HPO problems: only a couple of hyperparameters actually matter, but you don't know which. A grid wastes its resolution by trying many distinct values of the *unimportant* knobs while testing only `grid_size` distinct values of the important one. Random search, by sampling every knob independently each trial, tries `n` distinct values of the important knob for the same `n` trials. CS329P's practical verdict is unambiguous: **in practice, start with random search.** It's the right default and the baseline every fancier method must beat.

### 2.5 Bayesian optimization (BO)

BO is the "smart" black-box method: it *learns* the shape of the objective as it goes and uses that to pick where to look next, rather than sampling blindly.

It has two parts. A **surrogate model** is a probabilistic regression model — a Gaussian process or a random forest — that estimates how the objective (validation score) depends on the hyperparameters, *and* reports its own uncertainty. An **acquisition function** turns the surrogate into a decision: it scores each candidate config by *both* its predicted objective and the surrogate's uncertainty there, and the next trial is sampled where that score is highest. Maximizing the acquisition function explicitly **trades off exploration** (go where the surrogate is uncertain — you might learn something) **against exploitation** (go where it predicts a high score — you might win now).

```text
loop:
  fit surrogate(observed trials)          # GP / random forest over HP -> score
  next_config = argmax acquisition(surrogate)   # high predicted score OR high uncertainty
  score = train_and_eval(next_config)
  observe (next_config, score)
```

BO has two real limitations, both from the slides. **In its initial stages it behaves like random search** — with no observations the surrogate is flat, so the first handful of trials carry no advantage. And the **optimization is inherently sequential**: each trial's choice depends on all previous results, which makes BO awkward to parallelize compared to random search's embarrassing parallelism. You can batch-BO around this, but the core loop wants to be serial.

### 2.6 Successive Halving and Hyperband (multi-fidelity)

Successive Halving attacks the budget from the other side: **don't sample smarter, evaluate cheaper, and reallocate budget toward survivors.**

```text
Successive Halving:
  randomly pick n configs, train each for m epochs
  repeat until one config remains:
    keep the best n/2, train them another m epochs
    keep the best n/4, train them another 2m epochs
    ...
```

The mechanism: give many configs a tiny budget, throw away the worst half, and pour the saved budget into the survivors. Promising configs earn progressively more epochs; hopeless ones are killed after a few. You pick `n` and `m` from your total budget and how many epochs a full training needs.

The tension is that `n` and `m` pull against each other: **`n` controls exploration** (how many configs you try) and **`m` controls exploitation** (how long before you judge them). Pick `m` too small and you kill a slow-starting config that would have won; pick `n` too small and you never sample the good config at all. **Hyperband** resolves this by *not picking* — it runs **multiple Successive Halving brackets**, each with a different `(n, m)` trade-off, sweeping from "many configs, short budget" to "few configs, long budget." Early brackets explore widely; later brackets exploit; the best result across brackets wins. **ASHA** (Asynchronous Successive Halving) is the production-grade variant: it removes the synchronization barrier so workers don't idle waiting for a rung to fill, which is what makes it scale to hundreds of parallel workers — the version you'll actually find inside Ray Tune.

### 2.7 BOHB — combining the two ideas

BO and Hyperband fix different weaknesses. Hyperband allocates budget brilliantly but **samples configs at random**, so it never gets smarter about *where* to look. BO samples configs intelligently but **spends full budget on each** and starts cold. **BOHB** (Bayesian Optimization + Hyperband) marries them: use Hyperband's bracket structure to decide *how much budget* each config gets, and a BO surrogate to decide *which configs to sample* instead of drawing them uniformly. You get Hyperband's early-stopping efficiency and BO's sample-efficiency in one method.

### 2.8 Comparison

| Algorithm | Type | Sample efficiency | Parallelism | Early stop | When to reach for it |
|---|---|---|---|---|---|
| **Grid search** | Black-box | Worst | Embarrassing | No | Tiny space (1–2 knobs), reproducibility/coverage demands |
| **Random search** | Black-box | Good baseline | Embarrassing | No | **Default first move**; strong, simple, trivially parallel |
| **Bayesian opt (BO)** | Black-box | High | Poor (sequential) | No | Expensive trials, modest budget, few workers |
| **Successive Halving** | Multi-fidelity | High | Good | Yes | Many cheap-to-judge configs; most are bad early |
| **Hyperband / ASHA** | Multi-fidelity | High | Excellent (ASHA async) | Yes | Lots of parallel compute, unknown `(n,m)` trade-off |
| **BOHB** | Both | Highest | Good | Yes | You want smart sampling *and* early stopping |

**Practical takeaway (CS329P):** start with random search to get a baseline and a feel for the space; graduate to Hyperband/ASHA when you have parallel compute, or BO/BOHB when trials are individually expensive. And whatever you do, **mine your own logs** — the top-performing configurations cluster, and you can often find them faster by looking at what worked here before, or what configs the relevant papers and codebases used, than by any blind search.

---

## 3. Neural Architecture Search (NAS)

NAS asks the next question: instead of tuning a fixed network's hyperparameters, can we *search the network itself*? A neural net has architecture-level hyperparameters — the **topological structure** (ResNet-ish vs MobileNet-ish, how many layers) and the **per-layer choices** (kernel size, channel count in a conv layer, hidden width in a dense/recurrent layer). NAS automates choosing them. Every NAS method, following Elsken et al. 2019, decomposes into three components: **search space** (what architectures are reachable), **search strategy** (how you explore them), and **performance estimation** (how you score a candidate without bankrupting yourself).

### 3.1 Search space — cell vs macro

The search space defines what NAS can build, and its design dominates the result.

- **Macro search** chooses the whole network's wiring directly — every layer, every connection. Maximally expressive, brutally large.
- **Cell (micro) search** searches for a small repeatable building block — a "cell" — then stacks copies of it into a network with a fixed macro skeleton (the trick from NASNet). The search space shrinks enormously, the found cell transfers across depths and datasets, and this became the standard because it's the only thing that made the search tractable.

### 3.2 Search strategy

**Reinforcement learning (Zoph & Le, 2017).** The seminal approach: an **RNN controller** emits a *sequence of tokens* describing an architecture (this layer is a 3×3 conv, that many filters, connect here…). The proposed architecture is trained to convergence, and its **validation accuracy is the reward**. The controller is updated with the **REINFORCE** policy-gradient rule to make high-reward architectures more likely. It worked — and it was *staggeringly expensive*: the naive approach is sample-inefficient and burned on the order of **~2000 GPU-days** for a single search, because every reward required training a network from scratch. Two escape routes were proposed immediately: **estimate performance** more cheaply, and **share parameters** across candidate architectures (EAS, ENAS) so you don't retrain from zero every time.

**Evolutionary search.** Maintain a population of architectures, mutate the good ones (swap an operation, add a connection), keep the fit, discard the unfit. Conceptually simple, parallelizes naturally, and competitive with RL — AmoebaNet matched or beat RL-found nets — but it carries the same fundamental cost if each candidate is trained from scratch.

**One-shot / weight-sharing.** The cost breakthrough. Instead of training thousands of separate networks, **combine the learning of architecture and weights into a single over-parameterized "supernet"** that contains every candidate architecture as a sub-path sharing one set of weights. Train the supernet once; then evaluate candidate sub-architectures by inheriting those shared weights and measuring accuracy after only a few epochs — you only need the candidate *ranking*, not absolute accuracy, so a cheap proxy metric suffices. Finally, **re-train the most promising candidate from scratch** for the real result. **ENAS** applied weight-sharing to the RL controller and cut the search from ~2000 GPU-days to roughly *half a GPU-day*.

**DARTS — Differentiable Architecture Search.** The most elegant one-shot method makes the discrete search *continuous and gradient-trainable*. Instead of hard-choosing one operation per layer, **relax the categorical choice into a softmax over all candidate operations.** With candidate operations `oᵢˡ` at layer `l`, the output passed to the next layer is the *weighted mixture*:

```text
output(l) = Σ_i  αᵢˡ · oᵢˡ(input)        with   αˡ = softmax(aˡ)
```

The mixing weights `aˡ` are now ordinary continuous parameters. You **jointly learn `aˡ` and the network weights by gradient descent**, then at the end pick the single operation with the largest `α` at each layer (`argmaxᵢ αᵢ`). Because the whole search is one differentiable optimization, **DARTS reached SOTA and cut search time to ~3 GPU-days** — three orders of magnitude below the RL original.

### 3.3 Performance estimation

The cost of NAS *is* the cost of scoring candidates, so estimation is where budget is won. The toolkit: a **proxy metric** (accuracy after a few epochs instead of to convergence, on the bet that early ranking predicts final ranking); **weight inheritance / sharing** (evaluate without training from scratch, per one-shot above); training on a **subsampled dataset or smaller model**; and **lower-fidelity** proxies generally — the same multi-fidelity logic as Hyperband, applied to architectures.

### 3.4 A different lever: compound scaling (EfficientNet)

Worth separating from search proper, because it largely *replaced* it for CNNs. A CNN can be scaled three ways — **deeper** (more layers), **wider** (more channels), **larger inputs** (higher resolution). EfficientNet's insight is that scaling them *together in a fixed ratio* beats scaling any one alone. **Compound scaling** sets depth `∝ αᵠ`, width `∝ βᵠ`, resolution `∝ γᵠ`, under the constraint `α·β²·γ² ≈ 2`, so a single coefficient `ϕ` cleanly trades one knob — total FLOPs roughly double per unit of `ϕ`. You do a small search once to find good `α, β, γ`, then just dial `ϕ` to get an entire family (EfficientNet-B0…B7). This is dramatically cheaper than searching a new architecture per compute budget, and it's why "scale a known-good architecture" beat "search a new one" in practice.

### 3.5 The honest state of NAS

CS329P (2021) presents NAS as "practical to use now" — and at the time, with compound scaling and differentiable one-shot search, it was. But be honest about the arc. NAS had real research problems even then: **explainability** (you get an architecture with no story for *why* it's good), and the gap between a clean search benchmark and a messy real task. And it had a structural vulnerability — it searches *small networks from scratch*, which is exactly the regime that the field walked away from. The 2026 update below is not a footnote here; it's the headline.

> **2026 update:** Classic NAS has largely faded from mainstream practice. The premise that justified it — *search a small network from scratch for your task* — was overtaken by **transfer learning, scaling laws, and foundation models**: you no longer design a bespoke net, you **take a large pretrained model and adapt it** (Lecture 09), and architecture is set by what scales predictably, not by what a controller discovers. The famous results (NASNet, AmoebaNet, DARTS, EfficientNet) are still worth knowing as ideas, and **hardware-aware NAS lives on at the edge** (MnasNet, FBNet, Once-for-All — see the Hardware lens). AutoML as a whole survives best for **tabular data**, where `AutoGluon`/`auto-sklearn` genuinely win and there's no foundation model to transfer from. For everyday HPO, nobody hand-rolls a tuner: the tools are **Optuna** (define-by-run, TPE/CMA-ES samplers, built-in pruners) and **Ray Tune** (distributed, ASHA/PBT/BOHB). Learn the original algorithms first — they're the mental model — then reach for these.

---

## 4. The deep-network tuning toolkit

The biggest gains in deep learning came not from tuning but from **structural inventions that changed what was trainable.** CS329P's framing: deep learning is a *differentiable programming language* for extracting information from data, with design patterns from the layer level up to the architecture level. Three patterns matter most.

### 4.1 Normalization: batch norm vs layer norm

**Why normalize at all.** Standardizing inputs makes the loss surface *smoother* — formally, a smaller Lipschitz constant `β` in `‖∇f(x) − ∇f(y)‖ ≤ β‖x − y‖`, and a smaller `β` permits a larger learning rate. That trick works for linear models on the input but **doesn't help deep nets**, because the inputs to *internal* layers drift during training. **Batch Normalization (BN)** standardizes the inputs to internal layers, improving smoothness and making training easier. (Why BN works is *still* somewhat controversial — the original "internal covariate shift" story is contested — but that it helps is not.)

CS329P factors *every* normalization layer into the same three steps, which is the right way to hold them in your head:

1. **Reshape** the input into a 2D matrix.
2. **Normalize** (standardize) along the chosen axis: `x̂ ← (x − mean) / std`.
3. **Recover** with learnable scale and shift: `y = γ·x̂ + β`, so the layer can undo the normalization if that's what's best.

The *only* thing that distinguishes the variants is the **reshape** — which axis you normalize over.

- **Batch norm** reshapes `X ∈ ℝ^{n×c×w×h} → ℝ^{nwh×c}` and normalizes **per channel, across the batch**. Its statistics are computed over the batch, so it needs a moving mean/var maintained during training for use at inference.
- **Layer norm** reshapes `X ∈ ℝ^{n×c×w×h} → ℝ^{cwh×n}` and normalizes **per example, across that example's features** — everything else is identical to BN.

```python
def batch_norm(X, gamma, beta, moving_mean, moving_var, eps, momentum):
    if not torch.is_grad_enabled():                 # inference: use running stats
        X_hat = (X - moving_mean) / torch.sqrt(moving_var + eps)
    else:
        if len(X.shape) == 2:                        # (batch, features)
            mean = X.mean(dim=0)
            var = ((X - mean) ** 2).mean(dim=0)
        else:                                        # (batch, channel, H, W)
            mean = X.mean(dim=(0, 2, 3), keepdim=True)
            var = ((X - mean) ** 2).mean(dim=(0, 2, 3), keepdim=True)
        X_hat = (X - mean) / torch.sqrt(var + eps)
        moving_mean = momentum * moving_mean + (1.0 - momentum) * mean
        moving_var = momentum * moving_var + (1.0 - momentum) * var
    return gamma * X_hat + beta, moving_mean, moving_var
```

**Why LN won for transformers.** BN's dependence on the batch is fatal for sequence models. Applied to an RNN, BN must keep **separate moving statistics for every time step** — and for very long sequences at inference, time steps you never saw during training have no statistics at all. LN sidesteps this entirely: it normalizes **within each example, up to the current step**, so it needs no batch statistics and is **consistent between training and inference** regardless of sequence length or batch size. That property — no cross-example coupling, identical behavior train and test — is exactly what a transformer needs, and it's why **LN is the normalization of the transformer block** while BN remains the default for CNNs. (Other variants just pick other reshapes: InstanceNorm normalizes per-channel-per-example, GroupNorm splits channels into groups, and you can also normalize weights or gradients instead of activations.)

| | Batch Norm | Layer Norm |
|---|---|---|
| Normalizes over | Batch dimension (per channel) | Feature dimension (per example) |
| Needs batch statistics | Yes (moving mean/var) | No |
| Train ≠ inference behavior | Yes (uses running stats at test) | No (identical) |
| Batch-size sensitive | Yes (fails at batch size 1) | No |
| Long sequences | Problematic (per-step stats) | Natural |
| De-facto home | CNNs | Transformers / RNNs |

### 4.2 Residual connections

**The problem they solve.** Naively, *adding layers can make accuracy worse* — not just overfit, but degrade on training data too, because the optimizer struggles to drive the extra layers toward even an identity mapping. Adding a layer `f` changes the function class from `g(x)` to `f(g(x))`; if the new class doesn't *contain* the old one, the deeper net can be strictly worse.

**The fix.** A residual connection makes the block compute `f(g(x)) + g(x)` — adding the input back via an identity shortcut. Two things follow. First, the function class is now **nested**: the block can recover `g(x)` exactly by driving `f → 0`, so adding the layer can never hurt the achievable class. Second, and operationally decisive, the **gradient gets an identity path**:

```text
without residual:   ∂/∂x [ f(g(x)) ]        = f'(g(x)) · g'(x)        # product of Jacobians, can vanish
with residual:      ∂/∂x [ f(g(x)) + g(x) ] = f'(g(x)) · g'(x) + g'(x)  # extra +g'(x) survives
```

That extra `+g'(x)` term is the gradient of the *shallower* sub-network; it flows backward undiminished even when the matmul Jacobians multiply down toward zero. This is what makes genuinely deep nets trainable — people have trained CNNs past **1000 layers** on the strength of it. **ResNet** is just stacked residual blocks and became the de-facto CNN architecture:

```python
class ResidualBlock(nn.Module):
    def __init__(self, in_ch, num_ch):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, num_ch, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(num_ch, num_ch, kernel_size=3, padding=1)
        self.bn1, self.bn2 = nn.BatchNorm2d(num_ch), nn.BatchNorm2d(num_ch)

    def forward(self, X):
        Y = F.relu(self.bn1(self.conv1(X)))
        Y = self.bn2(self.conv2(Y))
        return F.relu(Y + X)            # the identity shortcut: + X
```

Conceptually it's **boosting over features** (each block adds a correction to a running representation), and it has variants — add the shortcut at different points, change the output channel count, or concatenate instead of add (DenseNet). The identity path is the load-bearing idea, and it reappears as the residual stream of every transformer.

### 4.3 Attention (brief — detail in Phase 5)

Attention solves an RNN bottleneck: at time `t`, an RNN's only access to the past is the single hidden state `h_{t−1}`; it can't directly reach `h_{t−2}, …, h_1`. **Attention lets the output be a weighted sum over *all* past states**, `Σ αᵢ hᵢ` with `α = softmax(a)`, where each weight `aᵢ` scores how relevant state `i` is to the current query.

The dominant scorer is **scaled dot-product attention**: `aᵢ = ⟨hᵢ, x_t⟩ / √d`, where `d` is the vector length and the `√d` divisor keeps the dot products from growing large enough to saturate the softmax. Generalized to **queries, keys, and values (QKV)** — query `q`, key–value pairs `(kᵢ, vᵢ)` — the output is `Σ αᵢ vᵢ` with `αᵢ = softmax(score(q, kᵢ))`:

```python
def dot_product_attention(queries, keys, values):
    d = queries.shape[-1]
    scores = torch.bmm(queries, keys.transpose(1, 2)) / math.sqrt(d)
    return torch.bmm(F.softmax(scores, dim=-1), values)
```

**Self-attention** uses the *same* sequence for Q, K, and V, so every position attends to every other. Its cost profile is the key trade against CNN/RNN:

| | CNN | RNN | Self-attention |
|---|---|---|---|
| Computation | `O(k·n·d²)` | `O(n·d²)` | `O(n²·d)` |
| Parallelization | `O(n)` | `O(1)` | `O(n)` |
| Max path length | `O(n/k)` | `O(n)` | `O(1)` |

The decisive rows are the last two: self-attention is **fully parallel** (unlike the strictly sequential RNN) and has **`O(1)` maximum path length** — any two positions interact directly, so gradients and information travel in one hop regardless of distance. That `O(n²·d)` compute cost is the price, and taming it is a Phase 5 topic.

A **transformer block** stacks exactly the three tools in this section: **multi-head self-attention** to aggregate inputs by element-relations, a **point-wise FFN** (a per-position MLP with shared weights) to transform each output, and **LayerNorm + residual connections** wrapped around both to make training easy.

```python
def transformer_block(X):                 # X: (batch, seq_len, d)
    Y = nn.LayerNorm(...)(multi_head_attention(X, X, X) + X)   # attention + residual + LN
    return nn.LayerNorm(...)(ffn(Y) + Y)                       # FFN + residual + LN
```

Stack these and you get the architecture behind **BERT** (encoder-only, good at *encoding* text), **GPT** (decoder-only, good at *generating* text), and **ViT** (a transformer fed image patches). Attention has become a fourth fundamental architecture alongside MLP/CNN/RNN — the full mechanics (multi-head, masking, KV-cache, positional encodings, FlashAttention) are developed in **[Phase 5 — ML Systems Engineering](../../../../Phase%205%20-%20Advanced%20Topics%20and%20Specialization/Track%20G%20-%20ML%20Systems%20Engineering/MLSys%20Deep%20Dives/README.md)**.

> **Hardware lens:** Tuning has a hardware twin, and it's the part of NAS that survived. **Hardware-aware NAS** drops accuracy as the sole objective and searches under a *device latency constraint* — measured (or modeled) on the *actual* target, because edge devices span CPU/GPU/DSP/NPU with ~100× performance spread and hard power budgets, so FLOPs are a poor proxy for wall-clock latency. The slide's formulation is to **minimize `loss × log(latency)^β`** so the search trades accuracy against measured speed; **MnasNet** searches on-phone latency directly, and **Once-for-All (OFA)** trains one supernet and extracts a *specialized sub-network per device* with no retraining — exactly the "design once, deploy to many constraints" goal. The structural tools also have direct kernel-level consequences: **residual adds** and **norm layers** are prime **kernel-fusion** targets (fuse conv→BN→ReLU into one kernel; fold BN into the preceding conv's weights at inference so it costs *nothing*; fuse the residual add into the epilogue), and BN-vs-LN changes the memory-access pattern a fused kernel must support. These connect straight to **[Phase 5 — Edge AI / Model Compression](../../../../Phase%205%20-%20Advanced%20Topics%20and%20Specialization/Track%20G%20-%20ML%20Systems%20Engineering/Guide.md)** and the inference-stack work in the MLSys deep dives — where the model you tuned here becomes a model you *serve* under a budget (Lecture 10).

---

## Current as of

**June 2026.** The HPO algorithms (random > grid, Bayesian optimization, Successive Halving/Hyperband/ASHA, BOHB) and the deep-net structural toolkit (BN/LN, residuals, attention) are taught as the durable, original CS329P material — they have not dated and remain exactly the right mental model. The reframing is **NAS**: CS329P (2021) presents it as freshly practical, and the headline 2026 correction is that **classic from-scratch NAS has largely faded** — transfer learning, scaling laws, and foundation models replaced "search a small net from scratch," AutoML survives best for tabular data (AutoGluon), hardware-aware NAS persists at the edge (MnasNet, Once-for-All), and day-to-day HPO runs on **Optuna and Ray Tune**, not hand-rolled search. The original is taught first because it is still how to *think* about the problem; the update is flagged, not silently substituted.

*Adapted from [Stanford CS329P](https://c.d2l.ai/stanford-cs329p) — Huang, Li & Smola, CC-BY-SA-4.0.*
