# Lecture 10 - Model Compression: Pruning, Quantization, Distillation

**Collection:** [Practical Machine Learning (CS329P)](README.md) | **Previous:** [← Lecture 09](Lecture-09.md) | **Next:** [Lecture 11](Lecture-11.md)

---

Every model you have trained so far in this course has been, almost certainly, *overparameterized*. The universal approximation theorem says even a single-hidden-layer MLP can approximate any continuous function — yet we routinely train networks with billions of parameters to fit problems whose intrinsic complexity is far smaller. We do this on purpose: a larger model is *easier to optimize* and generalizes better, thanks to the interaction of SGD with the model's structure (the theory here is still under active development). The slack is a training-time gift. At deployment it becomes a bill.

CS329P frames deployment as a set of hard budgets, and it is worth quoting them directly because they are the entire reason this lecture exists. In production you face: **Memory** — you often share memory with other processes, so your model does not get the whole machine. **Latency** — some applications require realtime responses (ads ranking, live captioning and translation, self-driving). **Cost** — power machines are more expensive, and you pay for every one you provision. And **Energy** — *both computation and accessing memory* consume significant energy, which is decisive for battery-powered devices. The slide's blunt conclusion: deploying big models is hard, and by far only a little of the time does anyone deploy the full multi-billion-parameter transformer as-is.

**Model compression** is the set of techniques that buy back those budgets: reduce model size and compute cost without (significantly) hurting predictive performance. The three classical levers are **pruning** (set weight elements to 0 so you neither store nor compute them), **quantization** (use fewer bits per weight, e.g. float32 → int8), and **knowledge distillation** (transfer what a big teacher learned into a small student). This is the precise seam where ML engineering hands the model off to hardware: pruning and quantization only pay off if the *silicon* can skip the zeros and run the low-bit math, and that constraint shapes which compression you choose. Treat this lecture as the bridge into Phase 5 — the place where "make the model smaller" becomes "make the chip go faster," and where the [MLSys Deep Dives](../../../../Phase%205%20-%20Advanced%20Topics%20and%20Specialization/7.%20ML%20Systems%20Engineering/MLSys%20Deep%20Dives/README.md) course and the [Gemma 4 Edge Deployment](../../../../Phase%205%20-%20Advanced%20Topics%20and%20Specialization/3.%20Edge%20AI/Gemma%204%20Edge%20Deployment/README.md) material pick up the story at full hardware depth.

---

## Learning objectives

By the end of this lecture you should be able to:

1. **Justify compression from the deployment budget** — explain why overparameterized models must be compressed, naming the memory / latency / cost / energy constraints and why *memory access* energy matters as much as compute.
2. **Run the prune→retrain loop** and distinguish unstructured from structured (channel/filter) pruning, applying a magnitude criterion and reasoning about the sparsity-vs-accuracy trade-off.
3. **Quantize a model** from FP32 to INT8 and below — derive the scale/zero-point mapping, choose symmetric vs asymmetric, and pick PTQ vs QAT given an accuracy budget.
4. **Distill a teacher into a student** using soft targets with temperature, and explain why "dark knowledge" in the softmax tail makes the student trainable.
5. **Map each technique to the hardware that accelerates it** — INT8 tensor cores, 2:4 structured sparsity, memory-bandwidth-bound decode — and explain why unstructured sparsity is hard to accelerate.
6. **Translate the 2021 picture to the 2026 LLM era** — INT4/GPTQ/AWQ, NF4+QLoRA, FP8, SmoothQuant, modern distillation — and know where to go deeper.

---

## 1. Why compress: overparameterization meets the deployment budget

Start from the asymmetry. Training happens once, in a data center, where you are happy to burn parameters because they make optimization easier and generalization better. Inference happens *millions or billions of times*, often on a shared or battery-powered machine, where every one of those parameters is paid for again on each forward pass. The model that was cheap to train is expensive to serve.

CS329P's deployment-challenges slide enumerates the four costs you are actually trading against:

| Constraint | What the slide says | Why it bites |
|---|---|---|
| **Memory** | "often share memory with others" | Your model competes with the OS, other tenants, the KV cache. Weights that don't fit in fast memory get fetched from slow memory, every call. |
| **Latency** | "some requires realtime (ads, live captioning/translation, self-driving)" | A 40 ms SLA is not negotiable; a bigger model that misses it is unusable regardless of accuracy. |
| **Cost** | "power machines are more expensive" | You provision for peak QPS. Halving model cost can halve your fleet. |
| **Energy** | "both computation and accessing memory needs a significant amount energy, especially for devices powered by batteries" | On a phone or a Jetson, the battery is the real budget — and moving bytes costs more than crunching them. |

That last row is the one hardware engineers must internalize, so it gets its own callout.

> **Hardware lens:** The slide explicitly lists *memory access* alongside computation as an energy cost — and on real silicon, memory access dominates. A 32-bit floating-point multiply-add costs on the order of a picojoule; reading the operands for it from off-chip DRAM costs *hundreds* of picojoules. The arithmetic is nearly free next to the data movement. This single fact reorganizes the whole field: the reason quantization works is not mainly that INT8 math is faster (though it is), it is that an INT8 weight is *4× smaller to move* than an FP32 weight. Compression is, first and foremost, a bandwidth and energy play. Keep that in mind for every technique below.

> **Hardware lens — the roofline preview.** Whether a kernel is *compute-bound* or *memory-bound* decides which compression helps. A large batched matmul on a GPU is compute-bound — pruning/quantizing the math wins. Autoregressive LLM *decode* (one token at a time, batch 1) is **memory-bandwidth-bound**: you re-read the entire weight matrix to produce a single token, the tensor cores sit mostly idle, and time-per-token is set by how fast you can stream weights out of HBM. There, shrinking the *weights* (INT8/INT4) is the only thing that moves latency. The roofline model that makes this rigorous lives in the [MLSys Deep Dives](../../../../Phase%205%20-%20Advanced%20Topics%20and%20Specialization/7.%20ML%20Systems%20Engineering/MLSys%20Deep%20Dives/README.md).

---

## 2. Pruning — make the weights sparse

### 2.1 The prune→retrain loop

The high-level algorithm from the slide is short and has stayed essentially unchanged for a decade:

```text
1. Train a network to convergence.
2. Assign each weight element a score (an importance estimate).
3. Set some elements to 0 based on their scores.
4. Fine-tune the model to recover accuracy.
   (optionally repeat 2–4 — iterative pruning)
```

The fine-tune step is not optional polish; zeroing weights perturbs the function, and a few epochs of retraining let the surviving weights compensate. Three knobs from the slide define a pruning method:

- **Score** — what makes a weight "important." Simplest is the **absolute value** (magnitude): small weights contribute little, prune them. Richer scores use a weight's contribution to activations or to gradients. You can compare scores **locally** (rank within a layer, prune the bottom k% of each) or **globally** (one threshold across the whole network, letting some layers stay dense and others go very sparse).
- **Scheduling** — prune **all at once** (one-shot) to a target sparsity, or prune a **fraction iteratively**, retraining between rounds. Iterative is slower but reaches higher sparsity at the same accuracy.
- **Fine-tuning** — after zeroing, do you **re-use the trained weights** (the standard, robust choice) or **randomly re-initialize** the surviving connections? That second option is where the lottery ticket comes in (below).

### 2.2 Unstructured vs structured

This distinction is the crux of the whole topic for a hardware engineer, so read the slide's two definitions carefully:

- **Unstructured pruning** sets *individual weights* to 0. It gives you the highest accuracy at a given sparsity, because you remove exactly the least-useful connections. But it "leads to sparse matrices/tensors that are often less efficient to compute" — the zeros are scattered with no pattern.
- **Structured pruning** sets a *whole unit / channel / block* to 0 — an entire convolutional filter, a whole row of a weight matrix, an attention head.

```text
Unstructured (90% sparse)          Structured (prune whole channels)
weight matrix:                     weight matrix:
  0  .3  0   0  .1                   .3 .2  0  .1  .4    <- kept
 .2  0   0  .4  0                    .1 .5  0  .2  .3    <- kept
  0  0  .5  0   0                     0  0  0   0   0    <- pruned channel
 .1  0   0  0  .2                    .4 .1  0  .3  .2    <- kept
=> 90% of cells are 0 but           => the zero column is removed entirely;
   in random positions                 the matrix literally gets smaller
```

That picture is the entire reason the next callout exists.

> **Hardware lens — why unstructured sparsity is hard to accelerate.** A dense matmul is the best-case workload for a GPU: perfectly regular, every lane busy, operands streamed in contiguous blocks. Punch 90% of the weights to zero *at random* and you have made it worse, not better — you now store index metadata to say where the survivors are, your memory accesses are gather-scattered instead of contiguous, and the tensor cores (which want dense tiles) cannot use the structure. A 90%-sparse unstructured layer routinely runs *slower* than the dense original on a GPU. The zeros save you nothing unless the hardware can *skip* them, and to skip them efficiently the sparsity must be **structured**. Structured pruning, by contrast, deletes whole rows/channels and yields a genuinely smaller *dense* tensor that every accelerator already runs fast — which is why production CNN compression overwhelmingly uses channel pruning.

> **Hardware lens — 2:4 structured sparsity (the hardware-friendly middle ground).** NVIDIA Ampere (A100, 2020) and every datacenter GPU since added **Sparse Tensor Cores** that accept a specific *fine-grained structured* pattern: **2:4 sparsity** — in every contiguous block of 4 weights, exactly 2 are zero. This is regular enough that the hardware stores only the 2 survivors plus a 2-bit index and feeds them through at roughly **2× throughput**, while being fine-grained enough to keep most of the accuracy of unstructured pruning. It is the compromise the slide's two categories were converging toward: structure the sparsity *just enough* that silicon can exploit it. (`torch.sparse` + cuSPARSELt, or TensorRT, will compress a 2:4 model to actually realize the speedup.)

### 2.3 Results, and the lottery ticket

The slide's empirical takeaways (from Blalock et al., MLSys'20, surveying the field) are sobering and worth stating plainly:

- Pruning algorithms **outperform random sparsification** — choosing *which* weights to drop matters.
- **Train-big-then-prune can beat training the small model directly** — the overparameterization helps during training even if you throw it away after.
- But pruning "**does not help as much as switching to a better architecture**." Compression is a finishing move, not a substitute for a good model. If you can use a more efficient backbone, do that first.

The **Lottery Ticket Hypothesis** (Frankle & Carbin) is the research idea lurking behind the fine-tuning knob: inside a large trained network there exists a sparse subnetwork — a "winning ticket" — that, *if reset to its original initialization* and trained in isolation, matches the full network's accuracy. It suggests the dense network's job was partly to find that subnetwork's lucky initialization. It is intellectually important and an active research line; it is *not yet* a reliable production recipe, which is exactly why the slide's default fine-tuning choice is "re-use weights from the trained network," not "randomly re-initialize."

---

## 3. Quantization — use fewer bits

### 3.1 Why low bits, and the hardware that rewards them

Quantization uses low-bit numbers to accelerate inference (and sometimes training): less memory, and access to special hardware paths. It does not hurt accuracy much *except* at extremely low bit-widths (1–2 bit), where it can fall off a cliff. The slide motivates it with raw throughput numbers — modern hardware is simply much faster at low-bit ops. On an NVIDIA A100:

| Data type | A100 throughput | Note |
|---|---|---|
| IEEE float32 (FP32) | 19.5 TFLOPS | the baseline |
| TF32 | 156 TFLOPS | 19-bit internal, FP32-range |
| IEEE float16 (FP16) | 312 TFLOPS | half precision |
| Bfloat16 (BF16) | 312 TFLOPS | FP32 exponent range, fewer mantissa bits |
| **INT8** | **624 TOPS** | 2× FP16 |
| **INT4** | **1248 TOPS** | 2× INT8 |

The pattern: each halving of bit-width roughly doubles arithmetic throughput *and* halves the bytes you move. Two practical rules from the slide: use **floats for training, integers for inference** (gradients need a larger dynamic range than a forward pass does), and **only quantize the heavy layers** — conv and dense — while leaving activations, weight updates, and the like in the default FP32. You quantize where the FLOPs and bytes are, not everywhere.

> **Hardware lens — INT8 tensor cores are the workhorse.** That 624-TOPS INT8 number is not a niche path; it is the bread and butter of production inference accelerators. TensorRT, ONNX Runtime, TFLite, and every edge NPU/DSP center on INT8 dense matmul. The 4× memory shrink vs FP32 is what gets a model into a phone's cache or a Jetson's limited DRAM; the 2× compute over FP16 is the bonus. When someone says "we quantized the model for deployment," the unmarked default they mean is INT8.

### 3.2 Two families: low-bit floats vs integers

**Low-bit floating-point** (FP16, BF16, TF32). Casting FP32 down is "straightforward — trim the fraction/exponent bits." The catch is *exponent* bits, which set the dynamic range. FP16 has *fewer exponent bits* than FP32, so small activations and gradients can underflow to exactly 0. The fix the slide gives is **loss scaling**: train with `λ·loss(ŷ, y)` for a tunable `λ > 0`, which scales activations and gradients up by `λ×` so values near zero survive FP16; you unscale before the optimizer step. BF16 sidesteps this by keeping FP32's exponent range (trading away mantissa precision instead), which is why it became the default training type for large models.

**Integer quantization** is the more aggressive, inference-time path. The simplest scheme the slide gives:

```text
XY  ≈  σx·σy · clip(round(X / σx)) · clip(round(Y / σy))
                └──────────── integer matrix multiply ────────────┘

X, Y    : the FP32 weight / activation matrices
σx, σy  : per-tensor scales, computed from the data in X (and Y)
round   : map the scaled real value to the nearest integer
clip    : saturate into the integer range, e.g. [-128, 127] for INT8
```

Concretely, the affine mapping between a real value `r` and its quantized integer `q` is:

```text
q = round(r / scale) + zero_point        (quantize)
r ≈ scale · (q − zero_point)             (dequantize)

scale       = (r_max − r_min) / (q_max − q_min)     # FP32, the step size
zero_point  = integer that the real value 0 maps to # keeps 0 exactly representable
```

- **Symmetric** quantization fixes `zero_point = 0` and uses a range like `[−127, 127]`. It is cheaper (the zero-point term drops out of the matmul) and is the standard for *weights*, which are roughly zero-centered.
- **Asymmetric** quantization lets `zero_point ≠ 0` to fit a range that isn't centered — the natural choice for **activations after a ReLU**, which are all ≥ 0, so spending half the integer range on negatives would waste resolution.

### 3.3 PTQ vs QAT, calibration, and where accuracy dies

The slide warns: "directly quantizing the trained weights may decrease accuracy." That gives the two regimes:

- **Post-Training Quantization (PTQ)** — take a finished FP32 model and quantize it with no further training. To pick the scales you run a few hundred representative inputs through the model and record the observed ranges — this is **calibration**. Fast (minutes) and data-light, but lossier.
- **Quantization-Aware Training (QAT)** — the slide's phrasing: "performs clip/round during training, but keeps float32." You simulate the rounding/clipping in the forward pass (a "fake-quant" op) while keeping a full-precision master copy of the weights and letting gradients flow through (via a straight-through estimator). The network *learns to be robust* to its own quantization, recovering most of the lost accuracy at the price of a training run.

```python
# Sketch of the QAT idea: round in the forward pass, but keep FP32 weights
# and pass gradients straight through the non-differentiable round().
def fake_quant(x, scale, zero_point, qmin=-128, qmax=127):
    q = torch.clamp(torch.round(x / scale) + zero_point, qmin, qmax)
    x_hat = (q - zero_point) * scale          # de-quantized approximation
    # straight-through estimator: forward uses x_hat, backward uses dL/dx
    return x + (x_hat - x).detach()
```

**Where does accuracy actually get lost?** In the **outliers**. A weight or (more often) activation tensor whose values are mostly small but with a few extreme entries forces `scale` to be large enough to cover the extremes — which crushes the resolution available to the many small values, where most of the signal lives. The few big numbers steal the dynamic range. This single failure mode is what motivates almost every modern advance in the 2026 update below.

> **Hardware lens — what "INT8 inference" really costs you to get.** The throughput is free; the *accuracy* is the engineering. Per-tensor scales are cheapest for the kernel but least accurate; **per-channel** (a scale per output channel of a weight matrix) is the standard production trade-off — far better accuracy, still a clean matmul. The hardware multiply-accumulates in INT8 but **accumulates in INT32** to avoid overflow, then requantizes the result back down. Knowing where the accumulator width and the requantize step sit is the difference between a model that hits its accuracy target and one that silently degrades.

---

## 4. Knowledge distillation — train a small model to imitate a big one

The third lever does not touch the weights of a fixed model; it trains a *different, smaller* model to behave like the big one. The slide's examples span all of ML: Random forest → decision tree; ResNet-152 → ResNet-34; BERT-Base → BERT-mini. The student is "better than training directly" because the teacher tells it *what it learned* in a form easier to fit than the raw data, and effectively augments the data with pseudo-labels.

### 4.1 The function-approximation view (why it can work at all)

CS329P gives the theory cleanly. The teacher `f` was learned by empirical risk minimization on a finite dataset `Dₙ` of `n` points sampled from the true distribution `p`. We then learn a student `g` to be close to `f` under some distance `d`:

```text
ℱ(f, g, Dₙ) = (1/n) Σᵢ d( f(xᵢ), g(xᵢ) )
```

Here is the subtle, important point: training the student only on `Dₙ` means we **pay twice for the statistical error of sampling** — once when `f` learned from `Dₙ`, again when `g` distills from `f` on the *same* `n` points. But the teacher `f` is a *function we can query anywhere*, not just on the `n` labeled points. So **sample a surrogate set `D′ₘ` from a distribution `q` with `m ≫ n`**, label it with the teacher, and distill on that. The generalization bound the slide shows makes the trade-offs explicit:

```text
ℱ(f, g*, p)  ≤  ℱ(f, g*, D′ₘ)  +  √((V − log δ)/m)  +  ‖p − q‖₁
 (generalized       (training        (shrinks as m         (q should be
    error)            error)        grows — use lots         close to p)
                                      of surrogate data)
```

The reading: more surrogate data (`m`) shrinks the middle term, and a surrogate distribution `q` close to the real `p` shrinks the last. This is *why* distillation generalizes better than training the small model directly — you have escaped the `n`-point sample by augmenting with teacher-labeled data. Concretely you generate `q` by **augmentation**: Gibbs-sampling features for tabular (FAST-DAD), the usual image augmentations, and for text, using a pretrained BERT to fill masked tokens, plus back-translation and mixup.

### 4.2 Soft targets and temperature — the "dark knowledge"

The mechanism that makes distillation more than relabeling: **the softmax outputs of the negative classes carry information that hard labels do not.** A teacher shown a photo of a dog might output `{dog: 0.9, wolf: 0.08, cat: 0.015, car: 0.005}`. The one-hot label says only "dog"; the teacher additionally says *"this looks somewhat like a wolf, not at all like a car."* That relational structure over the wrong answers — Hinton's **"dark knowledge"** — is a far richer training signal than a single 1.

To expose it we soften the distribution with a **temperature** `T` in the softmax:

```text
                exp(xᵢ / T)
S_T(x)ᵢ  =  ─────────────────────
              Σⱼ exp(xⱼ / T)
```

A larger `T` flattens the distribution, amplifying the small probabilities on the negative classes so the student can actually see and fit them. The distillation loss the slide gives matches the student's *soft* outputs to the teacher's *and* the student's hard outputs to the true label:

```text
  CE( S_T(g(x)), S_T(f(x)) )   +   λ · CE( S₁(g(x)), y )
  └──── soft term (temperature T) ────┘     └─── normal classification ───┘
```

Two notes the slide adds: as `T → ∞` the soft term approaches `MSE(g(x), f(x))` (matching raw logits), and despite the theory, **`T = 1` often works well** in practice — start simple.

### 4.3 Beyond outputs: intermediate representations, and distilling ensembles

You need not match only the final layer. Hidden activations carry richer information, so you can **match student layers to teacher layers** (add a small dense projection if their widths differ), with an MSE / L2 / even-learned loss. This is exactly how **TinyBERT** is built: the slide reports distilling BERT-base (110M params) into TinyBERT (14M) — an ~8× shrink — keeping most of GLUE (BERT-base 79.6 → TinyBERT 75.1 average), with the ablation showing intermediate-representation matching, layer-matching strategy, output loss, and data augmentation each contributing.

> **Tie-back to Lecture 05.** The most elegant use of distillation closes the loop with **ensembling** (Lecture 05). Ensembles win accuracy but multiply inference cost by the number of members — exactly the deployment cost this lecture is fighting. Distillation lets you **train a strong ensemble (or a heavy AutoGluon stack), then distill it into a single small network** that inherits much of the ensemble's accuracy at one model's cost. The teacher can be a *combination* of models; the student is one cheap net. `predictor.distill()` in AutoGluon does precisely this for tabular. **Self-distillation** — student and teacher share an architecture, the student trained on the teacher's soft labels — is the degenerate, surprisingly effective special case.

---

## 5. Choosing a technique — the three levers side by side

The three are **complementary**, not competing: a deployed model is frequently distilled *and* pruned *and* quantized. But they trade off differently:

| | **Pruning** | **Quantization** | **Distillation** |
|---|---|---|---|
| **What it changes** | Sets weights to 0 (removes connections/channels) | Fewer bits per number (FP32→INT8/4) | Trains a *new, smaller* model to imitate a teacher |
| **What it saves** | Model size; compute *only if* the sparsity is exploitable | Memory bandwidth + storage (≈4× at INT8) **and** compute | Everything — student is a genuinely smaller dense model |
| **Accuracy risk** | Low at moderate sparsity; cliff at high unstructured sparsity | Low at INT8; rises sharply below ~4-bit, driven by outliers | Usually small; student can approach teacher with enough surrogate data |
| **Hardware support needed** | **Structured / 2:4** sparse tensor cores to get real speedup; unstructured rarely helps on GPU | Low-bit tensor cores (INT8/INT4/FP8) — ubiquitous on modern accelerators | **None special** — output is a standard small dense net, runs anywhere |
| **When training is required** | Yes — fine-tune after pruning | PTQ: no (just calibration); QAT: yes | Yes — train the student |
| **Best when** | Channels are redundant (CNNs); you have 2:4-capable HW | You are memory/bandwidth-bound (LLM decode, edge) | You want a smaller *architecture*, or to collapse an ensemble |

The hardware column is the engineer's decision rule: **distillation and INT8 quantization are portable wins** (any device benefits), while **pruning's payoff is gated on the silicon** — only worth it when you can target structured/2:4 sparsity hardware or a sparse-aware runtime.

---

## 6. 2026 update — the LLM era of model compression

> **2026 update — the original ideas, mapped onto today's stack.** CS329P's 2021 framing is *exactly right* and still the foundation — but the center of gravity has moved from CNNs to LLMs, where the model is memory-bandwidth-bound at decode (Section 1) and compression is no longer optional. Learn the originals above first; here is where each one went.
>
> **Quantization went sub-INT8 and got outlier-aware.** The slide's "extremely low-bit hurts accuracy" warning was right *for naive PTQ* — the field beat it by handling the outliers from Section 3.3 directly:
>
> - **GPTQ** and **AWQ** are *weight-only* INT4 PTQ methods that quantize an LLM to 4 bits with near-FP16 quality. GPTQ uses second-order (Hessian) information to compensate for rounding error layer by layer; AWQ ("activation-aware") scales weight channels by their activation magnitude so the *important* channels keep precision. Both make INT4 a production default for serving 7B–70B models.
> - **SmoothQuant** is the direct answer to the activation-outlier problem: it *migrates* the difficulty from activations into weights by a per-channel scaling, so both can be quantized to INT8 — enabling true INT8 *activation* quantization (W8A8), not just weights.
> - **NF4 + QLoRA** quantizes a frozen base model to a 4-bit "NormalFloat" type and trains tiny LoRA adapters on top, fine-tuning a 65B model on a *single GPU*. This is QAT's spirit (train-aware-of-quantization) made practical at LLM scale.
> - **FP8** is the new training/inference type on **Hopper (H100)** and **Blackwell (B100/B200)** — an 8-bit *float* (E4M3 / E5M2) that keeps a usable exponent range, so it tolerates outliers far better than INT8 for *activations*. Blackwell pushes further to **FP4/MXFP4** microscaling formats. The slide's "low-bit floats are straightforward to cast" intuition scales right down to 8 and 4 bits.
> - **GGUF + llama.cpp** is how all of this reaches the edge: a file format packing K-quant INT4/INT5 weights that runs LLMs on CPUs, Apple Silicon, and phones — the practical home of "quantize for a battery-powered device."
>
> **Distillation went from BERT to GPT-4.** **DistilBERT** (40% smaller, 60% faster, ~97% of BERT) is the canonical realization of Section 4. Today the dominant form is **distilling a giant frontier model into a small one**: prompt a strong teacher (e.g. GPT-4-class) to generate high-quality outputs, then fine-tune a small open student on them — the surrogate-data argument of Section 4.1, where `q` is "whatever the teacher will answer." This is how most capable small open models are made. (Note the licensing and policy constraints on distilling from a closed API — that is now an engineering *and* legal decision.)
>
> **Pruning at LLM scale** stayed hard exactly as Section 2.2 predicted. **SparseGPT** and **Wanda** prune LLMs to ~50% in one shot without retraining; the durable lesson holds — to get *speed* you still need **2:4 structured** sparsity on Ampere+/Hopper, because scattered zeros remain unaccelerable. **Mixture-of-Experts (MoE)** is the architectural cousin: route each token to a few of many expert FFNs, so a model with huge *total* parameters activates only a small *fraction* per token — conditional computation as compression, and the reason several frontier models are far cheaper to serve than their parameter count suggests.
>
> Where this goes deep: the [MLSys Deep Dives](../../../../Phase%205%20-%20Advanced%20Topics%20and%20Specialization/7.%20ML%20Systems%20Engineering/MLSys%20Deep%20Dives/README.md) course covers INT4/FP8 kernels, paged-KV, and the roofline math; the [Gemma 4 Edge Deployment](../../../../Phase%205%20-%20Advanced%20Topics%20and%20Specialization/3.%20Edge%20AI/Gemma%204%20Edge%20Deployment/README.md) and [Edge LLM Inference Internals](../../../../Phase%205%20-%20Advanced%20Topics%20and%20Specialization/3.%20Edge%20AI/Edge%20LLM%20Inference%20Internals/Lecture-01.md) material walk GGUF quantization and on-device serving end to end.

---

## Summary

Models are overparameterized — a training-time advantage that becomes a deployment-time bill against four hard budgets: memory, latency, cost, and energy (where *moving* bytes costs more than computing on them). Three classical levers buy the budget back:

- **Pruning** zeros weights; the catch for hardware is that *unstructured* sparsity is rarely accelerable, so real speedups come from *structured* / 2:4 patterns the silicon can skip.
- **Quantization** uses fewer bits — INT8 as the portable workhorse, with accuracy lost mainly to *outliers* — choosing PTQ (fast, calibration-only) vs QAT (a training run that recovers accuracy).
- **Distillation** trains a small student on a teacher's *soft targets*, whose temperature-softened "dark knowledge" over the negative classes makes the student trainable, and which can collapse a whole ensemble into one cheap net.

They compose, and the *hardware support* a technique needs is the engineer's decision rule. This is the handoff point from ML engineering to hardware — Phase 5 is where it goes to the metal.

---

## Current as of

June 2026. The classical pruning/quantization/distillation core from Stanford CS329P (2021) is unchanged and foundational; this lecture refreshes the **quantization** material heavily for the LLM era — INT4 weight-only PTQ (GPTQ/AWQ), NF4+QLoRA, FP8/FP4 on Hopper and Blackwell, SmoothQuant for activation outliers, and GGUF/llama.cpp on the edge — plus modern frontier-model distillation and MoE as conditional-computation compression. INT8 remains the unmarked production default; INT4 weight-quantized LLM serving is now mainstream.

*Adapted from [Stanford CS329P](https://c.d2l.ai/stanford-cs329p) — Huang, Li & Smola, CC-BY-SA-4.0.*
