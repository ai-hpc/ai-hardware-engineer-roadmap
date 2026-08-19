# Module 05 — Calibration: Choosing the Scales

**Collection:** [Hardware-Aware LLM Quantization](README.md) | **Previous:** [← Module 04](Lecture-04.md) | **Next:** [Module 06 →](Lecture-06.md)

---

[Module 02](Lecture-02.md) ended on an unresolved trade: set a block's scale from its absolute maximum and the small values are annihilated; clip the maximum and the outlier is mangled. There is no universally correct answer — only an error metric and a search over scales.

That search is **calibration**, and this module is about running it well. The organizing idea is that the published methods are not competitors on a leaderboard; they are **answers to different error modes**. Pick by diagnosis, not by popularity.

---

## Learning objectives

By the end of this module you should be able to:

1. Choose between PTQ and QAT on cost/benefit grounds for a 27 B model.
2. Design a calibration set — size, domain, sequence length — and explain each choice.
3. Derive the MSE-optimal clipping threshold and explain why absmax is rarely it.
4. Explain the mechanism of **AWQ**, **GPTQ**, and **SmoothQuant** in one sentence each, and state which error mode each addresses.
5. Select a method from a diagnosis rather than from a benchmark table.

---

## 1. PTQ vs QAT — settle this first

```text
   PTQ (post-training quantization)
     calibrate on 128–1024 sequences · minutes to hours · no gradients · no training data needed

   QAT (quantization-aware training)
     fine-tune with fake-quant in the forward pass · GPU-days · needs training data + recipe
```

| | PTQ | QAT |
|---|---|---|
| Cost for a 27 B model | ~1 GPU-hour | ~100s of GPU-hours |
| Data needed | a few hundred unlabeled sequences | real training corpus |
| Typical recovery at 4-bit | good with a strong method | better, sometimes materially |
| Risk | none to the base model | can *degrade* other capabilities if the recipe is wrong |

**For this course's problem, use PTQ.** The reason is not cost — it is the framework from [Module 01](Lecture-01.md). Your total available throughput win from further weight quantization is bounded (Module 04 put `lm_head` at +8.7 %). Spending 100 GPU-hours of QAT to recover the last 0.3 % of behavior on a change worth 8.7 % throughput is a bad allocation, especially when [Module 07](Lecture-07.md) shows that *choosing different tensors* recovers more behavior than *training harder on the same tensors*.

QAT earns its cost in two specific cases: you are going below 4 bits (which [Module 03](Lecture-03.md) already ruled out on this silicon), or you are quantizing activations to 4 bits and hitting the outlier wall ([Module 06](Lecture-06.md)).

---

## 2. Calibration set design

Calibration estimates the distributions your scales must cover. Get the distribution wrong and every downstream scale is wrong.

```text
   SIZE          128–512 sequences is the standard working range.
                 Below ~64: scale estimates are noisy, especially percentile-based ones.
                 Above ~1024: returns flatten. This is a cheap parameter — sweep it once.

   LENGTH        Match your DEPLOYMENT sequence length distribution.
                 Calibrating at 512 tokens and serving at 262,144 is a real mismatch:
                 activation magnitudes and attention-sink behaviour both change with length
                 (Modules 06, 09).

   DOMAIN        Match deployment. Code-heavy serving → include code.
                 A general-web calibration set on a code model measurably misplaces scales.

   CONTAMINATION Never calibrate on your evaluation set. It is the quantization equivalent
                 of training on the test set, and it produces a quant that grades well and
                 serves badly.
```

**The most common calibration bug is length mismatch**, and it is invisible in short-context evaluation. If you serve long context, calibrate on long sequences even though it costs more.

A defensible default:

```python
CALIBRATION = dict(
    n_sequences   = 256,
    seq_length    = 4096,          # or your p50 deployment length, whichever is larger
    domains       = ["web", "code", "math", "chat"],   # weighted to your traffic mix
    seed          = 0,             # calibration is an experiment; it must be reproducible
    exclude       = ["wikitext-2", "your_eval_sets"],
)
```

---

## 3. Choosing a scale: four estimators

Given a block or channel of values, the scale determines the representable window. Four ways to pick it:

### 3.1 Absmax

```text
   s = max(|x|) / q_max
```

Zero clipping error by construction, maximal rounding error. **One outlier sets the scale for the entire group** — this is exactly the failure demonstrated in [Module 02 §4](Lecture-02.md), where `0.9` quantized to `0` because one element was `23.7`.

Fine for weights with well-behaved distributions and small blocks. Poor for activations.

### 3.2 Percentile

```text
   s = percentile(|x|, p) / q_max          typical p ∈ [99.9, 99.99]
```

Clip the tail, keep resolution for the bulk. Cheap and surprisingly effective. The cost is that `p` is a hyperparameter you must sweep, and the right value differs per tensor type.

### 3.3 MSE-optimal

Search the clipping threshold that minimizes reconstruction error:

```text
   c*  =  argmin_c  E[ ( Q_c(x) − x )² ]

   where the error splits exactly into the two mechanisms from Module 02:

     E[e²]  =  ∫       (Q(x)−x)² p(x) dx     ← rounding, decreases as c decreases
               |x|≤c
            +  ∫       (c·sign(x) − x)² p(x) dx   ← clipping, increases as c decreases
               |x|>c
```

```text
   error
     ▲
     │╲                                    ╱
     │ ╲  clipping error                  ╱  rounding error
     │  ╲                                ╱
     │   ╲                              ╱
     │    ╲__                        __╱
     │       ╲──__            __──╱
     │             ╲──_  _──╱
     │                 ╲╱          ← c*, the MSE-optimal threshold
     └──────────────────┴──────────────────────▶  clipping threshold c
                                          absmax (c = max|x|)
```

A 20–50 point grid search over `c ∈ [0.5, 1.0] × max|x|` per group is cheap and is the workhorse of good PTQ pipelines.

```python
def mse_optimal_scale(x, q_grid, n_steps=40):
    """Grid-search the clipping ratio that minimizes reconstruction MSE."""
    amax, best, best_err = x.abs().max(), None, float("inf")
    for r in torch.linspace(0.5, 1.0, n_steps):
        s = (amax * r) / q_grid.max()
        xq = quantize_to_grid(x / s, q_grid) * s     # round-to-nearest + clamp
        err = ((xq - x) ** 2).mean().item()
        if err < best_err:
            best_err, best = err, s
    return best
```

### 3.4 KL / entropy-based

Minimize the divergence between the pre- and post-quantization value distributions rather than the pointwise error. This is TensorRT's classic INT8 activation calibrator. It optimizes distribution shape over element-wise fidelity, which sometimes matters more — but for weights, MSE is usually the better proxy and is far cheaper to reason about.

### Which to use

| Tensor | Recommended | Why |
|---|---|---|
| Weights, small blocks (NVFP4 group 16) | **MSE-optimal**, absmax acceptable | group is small; outlier damage is contained |
| Weights, large groups (≥128) | **MSE-optimal** | one outlier hurts many values |
| Activations | **percentile or MSE**, never absmax | rare token spikes ([Module 06](Lecture-06.md)) |
| KV cache | **percentile** per head | cheap, runs online, must not stall decode |

---

## 4. Beyond scales: the three methods that matter

Scale selection alone treats each tensor in isolation. The three methods below exploit **structure** — and each targets a different error mode.

### 4.1 AWQ — protect the salient weight channels

**Observation:** not all weight channels matter equally. The channels multiplied by large activations dominate the output, and ~1 % of channels account for a disproportionate share of the error.

**Mechanism:** rather than keeping those channels in higher precision (which breaks the uniform layout the kernels need), AWQ **migrates scale** between activations and weights. For a per-channel factor `s`:

```text
   y = (x / s) · (s · W)
       └──────┘   └─────┘
       activation  weight scaled UP before quantization
       scaled DOWN → occupies more of the E2M1 grid → quantizes more accurately
```

The product is mathematically unchanged; the quantization error is not. The scaling is folded into the preceding layernorm at export, so **it costs nothing at runtime**.

**Error mode addressed:** *weight* channels whose quantization error is amplified by large activations.

### 4.2 GPTQ — compensate error layer by layer

**Observation:** quantizing weight `w_i` produces a known error. The remaining un-quantized weights in the same layer can be *adjusted* to absorb it.

**Mechanism:** using the layer's Hessian `H = 2·E[x xᵀ]` (from calibration activations), quantize weights one at a time and update the rest along the direction that minimizes the layer's output error — the Optimal Brain Surgeon update, made tractable by processing in a fixed order with a Cholesky factorization:

```text
   for each column i:
       q_i   = quantize(w_i)
       err   = (w_i − q_i) / H⁻¹_ii
       w_{i+1:}  −=  err · H⁻¹_{i, i+1:}       ← remaining weights absorb the damage
```

**Error mode addressed:** *accumulated* layer-output error. GPTQ does not care about individual weight fidelity; it minimizes the error of the layer's output, which is the thing that actually propagates.

### 4.3 SmoothQuant — move the difficulty from activations to weights

**Observation:** activations have systematic per-channel outliers; weights do not ([Module 06](Lecture-06.md)). Activations are therefore much harder to quantize than weights.

**Mechanism:** migrate the difficulty with a per-channel smoothing factor `s_j = max|X_j|^α / max|W_j|^(1−α)`:

```text
   Y = (X · diag(s)⁻¹) · (diag(s) · W)
        └────────────┘    └──────────┘
        activations        weights become
        become SMOOTHER    slightly harder
                           (they can afford it)
```

`α ≈ 0.5` balances the two; `α → 1` pushes all difficulty onto weights.

**Error mode addressed:** *activation* channel outliers. Essential for W8A8 and W4A4; **irrelevant for W4A16**, because if activations stay in BF16 there is no activation quantization error to smooth.

### Method selection by diagnosis

```text
   What is your dominant error source?
   │
   ├── Weight channels amplified by large activations   ──▶  AWQ
   │
   ├── Accumulated layer-output error at low bit-width  ──▶  GPTQ
   │      (and AWQ + GPTQ compose — they address different things)
   │
   ├── Activation per-channel outliers                  ──▶  SmoothQuant
   │      ONLY if you are quantizing activations at all
   │
   └── Rare token-level activation spikes               ──▶  neither; see Module 06
          (these are not channel-structured; smoothing does not remove them)
```

> For the case study — **W4A16 NVFP4 on `sm_120`, batch 1** — activations stay BF16, so SmoothQuant is **not applicable**. The relevant methods are AWQ and GPTQ. Engineers routinely apply SmoothQuant to W4A16 pipelines and report no benefit, then conclude the method is weak. The method is fine; it was aimed at an error mode that configuration does not have.

---

## 5. Calibration is an experiment, so version it

Every quantized artifact must carry its calibration provenance, or you cannot reproduce or debug it:

```yaml
quantization:
  format: NVFP4              # E2M1 + block16 + E4M3 scale + FP32 global
  scheme: W4A16
  scale_estimator: mse_optimal
  clip_grid: [0.5, 1.0, 40]
  methods: [awq, gptq]
  gptq:
    damping: 0.01
    actorder: true
calibration:
  n_sequences: 256
  seq_length: 4096
  domains: {web: 0.4, code: 0.3, math: 0.15, chat: 0.15}
  seed: 0
  dataset_hash: sha256:...
excluded_layers: [lm_head, layers.0., layers.47.]   # see Module 07
provenance:
  toolkit: llm-compressor 0.x / TensorRT Model Optimizer 0.x
  commit: ...
```

Two rules that will save you real time:

* **`dataset_hash` is not optional.** "Calibrated on 256 web sequences" is not reproducible; a hash is.
* **First and last layers are excluded by default in most good recipes.** [Module 07](Lecture-07.md) explains why, and how to verify it for your model rather than inheriting it as folklore.

---

## Checkpoint

You should now be able to:

1. Justify PTQ over QAT for this problem with a cost/benefit argument, and name the two cases where QAT wins.
2. Name the four calibration-set parameters and the failure mode of getting each wrong.
3. Sketch the clipping/rounding error curves and locate `c*`.
4. State AWQ, GPTQ, and SmoothQuant's mechanisms in one sentence each.
5. Explain why SmoothQuant does nothing for a W4A16 pipeline.
6. List the fields a reproducible quantization manifest must carry.

---

## Ship it

Take one linear layer from a real model and produce a **calibration ablation table**:

| Estimator | Layer output MSE | Fraction annihilated | Wall-clock |
|---|---|---|---|
| absmax | | | |
| percentile 99.9 | | | |
| percentile 99.99 | | | |
| MSE-optimal | | | |
| MSE-optimal + AWQ | | | |
| MSE-optimal + AWQ + GPTQ | | | |

Then answer in writing: **which row would you ship, and what would change your mind?** The second half is the part that matters.

---

## Current as of

* **Timeless:** the clipping/rounding decomposition, the MSE-optimal search, the mechanisms of AWQ/GPTQ/SmoothQuant, error-mode-driven selection.
* **2026 tooling:** `llm-compressor` (vLLM ecosystem) and **NVIDIA TensorRT Model Optimizer** are the two mainstream PTQ toolkits with NVFP4 support. Both implement AWQ and GPTQ; verify NVFP4 + `sm_120` export coverage in the version you install, since it moves between releases.
* **Refresh surface:** which methods are implemented for which formats. The error-mode taxonomy in §4 is stable; the tool that implements each is not.

---

**Next:** [Module 06 — Activation Outliers →](Lecture-06.md)
