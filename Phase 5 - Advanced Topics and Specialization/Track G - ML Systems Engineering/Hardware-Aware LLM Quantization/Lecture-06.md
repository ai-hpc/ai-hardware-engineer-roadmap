# Module 06 — Activation Outliers

**Collection:** [Hardware-Aware LLM Quantization](README.md) | **Previous:** [← Module 05](Lecture-05.md) | **Next:** [Module 07 →](Lecture-07.md)

---

Weights are a fixed, inspectable tensor you can spend an hour calibrating. Activations are a **distribution over inputs you have not seen yet**, and they contain structures that are pathological for any fixed-scale quantizer.

This module distinguishes possible outlier structures and the scaling algebra used to handle them. Measure them in your model; they are not universal facts about every transformer or recurrent layer. Both activation error and the actual kernel path matter when comparing W4A16 and W4A4.

---

## Learning objectives

By the end of this module you should be able to:

1. Contrast weight and activation distributions and explain why the latter are harder.
2. Distinguish **systematic channel outliers** from **rare token spikes**, and name the different remedy each requires.
3. Distinguish **massive activations** from **attention sinks**, and test whether changing them affects the task.
4. State which scaling axes are mathematically legal for a GEMM, and why per-channel activation scaling requires a weight correction.
5. Decide whether W4A4 is worth it for a given operating point.

---

## 1. Two different tensors

```text
   WEIGHTS                                ACTIVATIONS
   ────────────────────────────           ──────────────────────────────
   fixed after training                   change with every input
   distribution is model-dependent            heavy-tailed, structured
   range must be measured                     range must be measured
   calibrate once, offline                can use static or dynamic scales
   outliers scattered                     outliers may be channel- or token-structured
   you can spend an hour per tensor       you have microseconds
```

The diagram describes useful diagnostic possibilities, not measured distributions for this case. Weight outliers and difficult recurrent gates also occur. Collect per-layer statistics before choosing a policy.

---

## 2. Structure one: systematic channel outliers

In some trained transformers, a **small, consistent set of hidden dimensions carries values 10–100× larger than the rest**, in the same channels, for essentially every token.

```text
   hidden dimension index (d_model = 8192)
   0        1000      2000      3000      4000      5000      6000      7000
   │         │         │         │         │         │         │         │
   ▁▁▁▂▁▁▁▁▁▁█▁▁▁▁▁▂▁▁▁▁▁▁▁▁▁▁▁▁▁▁█▁▁▁▁▁▁▁▁▁▁▂▁▁▁▁▁▁▁▁▁▁▁▁▁█▁▁▁▁▁▁▁▁▁▁▁▁▁
            ▲                     ▲                          ▲
            └── the SAME channels, on nearly EVERY token ────┘
```

The consequence for a per-tensor scale is immediate. If three channels out of 8192 are 50× larger than the rest, a per-tensor absmax scale is set by those three, and the other 8189 channels are compressed into the bottom 2 % of the representable grid. With E2M1's eight magnitudes, that means **almost everything quantizes to zero** — the [Module 02 §4](Lecture-02.md) annihilation, applied to 99.96 % of the tensor.

**Because this structure is consistent, it is fixable.** SmoothQuant ([Module 05](Lecture-05.md)) migrates the per-channel magnitude into the weights, where it can be absorbed. Per-channel activation scaling would also fix it — if it were legal, which brings us to §4.

---

## 3. Structure two: rare token spikes and their possible role

The second structure is different in kind. A **small number of token positions** — very often the first token — produce activations with magnitudes thousands of times the median, concentrated in a couple of dimensions. These are the **massive activations** described in the recent literature, and they are closely tied to **attention sinks**.

```text
   activation magnitude by token position
   ▲
   │ █
   │ █
   │ █                                                    ← position 0 (often BOS):
   │ █                                                       magnitude 1000×+ the rest
   │ █
   │ █ ▁ ▂ ▁ ▁ ▂ ▁ ▁ ▁ ▂ ▁ ▁ ▁ ▁ ▂ ▁ ▁ ▁ ▁ ▁ ▂ ▁ ▁ ▁ ▁
   └──┴─┴─┴─┴─┴─┴─┴─┴─┴─┴─┴─┴─┴─┴─┴─┴─┴─┴─┴─┴─┴─┴─┴─┴──▶ token position
      0 1 2 3 ...
```

**One proposed explanation.** Softmax distributes total probability mass 1 across the available keys. Some trained models repeatedly allocate substantial mass to early tokens, called attention sinks. Large activations in selected dimensions have been observed alongside sink behavior, but an activation spike alone does not establish the attention mechanism or functional importance. Inspect attention/output behavior and run controlled interventions.

**Two hypotheses to test:**

1. **Clipping can remove useful information.** Compare unclipped and clipped candidates on task, likelihood and long-context tests. A large value is neither automatically harmless noise nor automatically essential. The relevant failure can appear in several metrics.

2. **Percentile calibration can clip rare spikes.** Whether a 99.9th-percentile threshold clips them depends on sample frequency and the statistic’s axes. Include representative rare conditions and measure the effect; the percentile alone does not establish a faulty recipe.

```text
   channel outliers   →  SYSTEMATIC   →  migrate them (SmoothQuant)         ✅
   token spikes       →  POSSIBLY IMPORTANT  →  compare dynamic scaling, clipping
                                             and retained higher precision
```

**This is why the two structures need different remedies, and why treating "activation outliers" as one problem fails.**

---

## 4. Which scaling axes are legal

This is the part most treatments skip, and it explains why activation quantization is constrained in a way weight quantization is not.

For `Y = X · W` with `X : [T, K]` and `W : [K, N]`:

```text
   Y[t, n]  =  Σ_k  X[t, k] · W[k, n]
```

**Per-token scaling of X (per row) — LEGAL.**

```text
   X[t, :] = a_t · X̂[t, :]     ⟹     Y[t, n] = a_t · Σ_k X̂[t,k] W[k,n]
```

`a_t` factors cleanly out of the whole output row. You can dequantize afterwards. The factor can be applied outside the reduction. Computing scales and applying them still costs work; fusion and hardware support determine that cost.

**Per-output-channel scaling of W (per column) — LEGAL.**

```text
   W[:, n] = b_n · Ŵ[:, n]     ⟹     Y[t, n] = b_n · Σ_k X[t,k] Ŵ[k,n]
```

The factor can be applied to the output column; storage and execution overhead remain implementation-dependent.

**Per-input-channel scaling of X (per column, index `k`) — NOT LEGAL alone.**

```text
   X[t, k] = c_k · X̂[t, k]     ⟹     Y[t,n] = Σ_k c_k X̂[t,k] W[k,n]
```

`c_k` sits **inside the summation over `k`**. It does not factor out. You cannot undo it after the GEMM.

```text
   The reduction axis is the one you cannot scale freely.
   ──────────────────────────────────────────────────────────
   And it is exactly the axis where the channel outliers live.
```

One way to support a per-input-channel factor without changing a conventional GEMM reduction is to move it to the other operand:

```text
   Y = (X · diag(c)⁻¹) · (diag(c) · W)
```

which is **precisely SmoothQuant** ([Module 05](Lecture-05.md)). The identity is exact before quantization. An alternative is a kernel that applies scales within partial reductions, as block-scaled hardware does; the algebra does not prohibit such kernels.

So the toolkit is:

| Structure | Legal remedy |
|---|---|
| Channel outliers (reduction axis) | SmoothQuant-style migration into W, or fine-grained blocks |
| Token spikes (row axis) | per-token dynamic scaling, with measured cost and quality |
| Both | per-token scaling **and** migration, plus small blocks |

---

## 5. Why block scaling helps, and how much

NVFP4's 16-element blocks provide a third defence that needs no algebra at all: **an outlier only contaminates its own block.**

```text
   PER-TENSOR scale, one 50× outlier in 8192 values:
   ├──────────────────────── all 8192 values share one window ─────────────────────┤
        → 8191 values crushed into the bottom 2 % of the grid  → mass annihilation

   BLOCK-16 scale, same outlier:
   ├──16──┤├──16──┤├──16──┤ ... ├──16──┤
                    ▲
                    └─ only THESE 16 values are affected;  16/8192 = 0.2 % of the tensor
```

With an unchanged global scale, one changed local block scale directly affects 16 rather than 8192 elements, a **512× smaller group** in this hypothetical matrix. If a two-level format recomputes its global scale, other block scales may also change; scale underflow can affect distant blocks. This localizes direct grid changes; subsequent network operations can propagate their effects. It does not prove that a block format universally replaces per-tensor INT8.

It is also why the NVFP4-vs-MXFP4 block-size difference (16 vs 32) matters more for activations than for weights. Activations are where the outliers are.

---

## 6. Compare actual W4A16 and W4A4 implementations

Activation payload can be small at batch one while activation **precision** still selects a different compute path. Native block-scaled FP4 MMA consumes compatible low-precision operands; weight-only W4A16 commonly dequantizes into wider arithmetic. A weight-only roofline cannot prove W4A4 has zero benefit. Conversely, activation conversion and numerical errors may outweigh its gains.

The pinned community case's quantization metadata declares 4-bit input activations and weights, so it is not evidence for a universal W4A16 recommendation. Compare supported operators at the relevant batch, prompt/context lengths and verification sizes. Report quantize/dequantize costs and quality. Prefill is not compute-bound by definition, and no single batch threshold applies to every layer.

---

## 7. Measure it yourself

Do not take any of this on faith for your model. The diagnostic is thirty lines:

```python
import torch
from collections import defaultdict

stats = defaultdict(list)

def make_hook(name):
    def hook(_mod, inp, _out):
        x = inp[0].detach().float().reshape(-1, inp[0].shape[-1])  # flatten leading dimensions
        stats[name].append({
            "per_channel_absmax": x.abs().amax(dim=0).cpu(),   # [hidden]
            "per_token_absmax":   x.abs().amax(dim=-1).flatten().cpu(),
            "median":             x.abs().median().item(),
        })
    return hook

handles = []
for name, mod in model.named_modules():
    if isinstance(mod, torch.nn.Linear):
        handles.append(mod.register_forward_hook(make_hook(name)))

# Run representative calibration batches in eval/inference mode.
# Then remove hooks so repeated experiments do not accumulate callbacks:
for handle in handles:
    handle.remove()

for name, records in stats.items():
    ch  = torch.stack([r["per_channel_absmax"] for r in records]).amax(0)
    tok = torch.cat([r["per_token_absmax"] for r in records])
    med = sum(r["median"] for r in records) / len(records)

    print(f"{name:50s}  "
          f"channel_ratio={ch.max()/ch.median().clamp_min(1e-12):7.1f}  "   # >10 → channel outliers
          f"token_ratio={tok.max()/tok.median().clamp_min(1e-12):8.1f}  "   # >100 → massive activations
          f"outlier_channels={(ch > 10*ch.median()).sum().item():4d}")
```

Read it like this:

| Signal | Threshold | Diagnosis | Remedy |
|---|---|---|---|
| `channel_ratio` | > 10 | systematic channel outliers | SmoothQuant / smaller blocks |
| `token_ratio` | > 100 | massive activations / sinks | test dynamic scaling and clipping separately |
| `outlier_channels` | small (1–10) | classic transformer structure | expected; not a bug |
| both low | — | this layer is easy | measure output/task sensitivity |

Run this **before** choosing a method. It is the diagnosis that [Module 05 §4](Lecture-05.md)'s selection tree requires, and it takes ten minutes.

---

## Checkpoint

You should now be able to:

1. Name the two activation-outlier structures and give the different remedy each needs.
2. Describe an attention-sink observation and design an experiment to test clipping effects.
3. Prove which GEMM scaling axes are legal, and derive SmoothQuant's form from the illegal one.
4. Quantify the blast-radius advantage of block-16 scaling over per-tensor scaling.
5. Explain why activation payload alone cannot choose W4A16 versus a native W4A4 kernel.
6. Read a channel/token ratio table and pick a method from it.

---

## Ship it

Produce an **activation profile** for one model: the per-layer table from §7, a histogram of per-channel absmax for the worst layer, a plot of per-token absmax versus position (testing whether early-position spikes occur), and a one-paragraph diagnosis naming which structures your model has and which remedy you selected.

If your `token_ratio` exceeds 100, compare clipped and unclipped variants. A percentile choice is not a bug by itself; assess its numerical and task effects.

---

## Current as of

* **Timeless:** the two outlier structures, the legal-axis proof, the blast-radius argument, the distinction between activation bytes and kernel operand precision.
* **Research observations:** [StreamingLLM](https://arxiv.org/abs/2309.17453) studies attention sinks; [Massive Activations](https://arxiv.org/abs/2402.17762) studies large activations. Apply their findings within the tested models and verify a new model rather than adopting an unconditional clipping rule.
* **Refresh surface:** whether runtimes expose per-token dynamic activation scaling for NVFP4 on `sm_120`. If they do not, W4A4 is off the table for you regardless of the analysis here.

---

**Next:** [Module 07 — Layer Sensitivity →](Lecture-07.md)


**Answer guidance:** low channel/token ratios do not establish low task sensitivity. The thresholds above are exploratory, not validated decision boundaries. Attention sinks and massive activations are related observations in particular models, not interchangeable diagnoses; test removal/clipping rather than assuming every large value is essential. See [SmoothQuant](https://arxiv.org/abs/2211.10438) and [the CPU calibration lab](CPU-Quantization-Lab.md).
