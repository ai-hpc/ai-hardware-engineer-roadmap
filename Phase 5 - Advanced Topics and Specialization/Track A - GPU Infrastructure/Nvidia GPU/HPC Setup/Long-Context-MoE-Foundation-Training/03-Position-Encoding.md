# Module 03 — Positional Encoding for Long Context

**Parent:** [Long-Context MoE Foundation Training](README.md)

**One-line purpose:** Choose and configure a positional encoding scheme (RoPE with base scaling, YaRN, position interpolation, ALiBi) that lets a model trained at one context length generalize to a much longer one without retraining from scratch.

**Prerequisites:** Module 02. Familiarity with rotary position embeddings (RoPE).

**Artifact:** A per-position retrieval-accuracy plot comparing at least three position-encoding strategies on a model extended from 4K to 32K (or 32K → 128K), evaluated on a needle-in-haystack-style task.

---

## Why it matters

A model trained at 4K can technically be evaluated at 32K — the matmuls work. But the attention outputs are usually garbage past the training length because the position encoding was not designed to extrapolate. Choosing the right extension scheme determines whether you need a full retrain (expensive) or a short continual pretraining (cheap). This is one of the few areas where the right architectural knob saves weeks of compute.

---

## Mental model

### RoPE in one paragraph

Rotary position embedding rotates the Q and K vectors by a position-dependent matrix `R(pos, θ_k)` where `θ_k = base^(-2k/D)`. The dot product `Q(pos_q)ᵀ K(pos_k)` then depends only on the relative offset `pos_q − pos_k`. The `base` (usually 10000) controls how slowly the high-frequency rotations cycle.

### Why naive RoPE breaks at extension

RoPE encodes each dimension at a different rotational frequency. At positions far beyond the training length, the **high-frequency** dimensions complete many rotations the model has never seen — the rotation pattern is outside the training distribution. The model's attention scores at those positions degenerate.

Concretely: with `base = 10000` and a model trained at `N = 4096`, position `pos = 100_000` produces dot products that look nothing like anything the model saw in pretraining.

### Three extension strategies

#### 1. RoPE base scaling

Increase the `base` from `10000` to something like `500000` or `1M` and continue pretraining briefly. Higher base = slower rotations = the high-frequency dimensions cycle less per position, so the rotational patterns at position 100K still resemble what the model saw during continual pretraining at e.g. 32K.

This is the simplest method. Llama 3.1 used `base = 500000` for its long-context variants. Works well up to ~10× the training length.

#### 2. Position interpolation (PI)

Instead of using position `pos`, use `pos · (N_train / N_extended)`. So a model trained at 4K, evaluated at 32K, uses scaled positions `pos / 8`. The rotational pattern stays within the training distribution, just compressed.

Cheap, works without retraining for modest extensions (2–4×). Loses precision per position (resolution drops).

#### 3. YaRN (Yet another RoPE extensioN)

Hybrid scheme: apply position interpolation only to the **low-frequency** dimensions (which need it) and leave high-frequency dimensions unchanged (where PI would damage resolution). Adds a temperature correction to compensate for distribution shift.

Empirically the best single extension method for going from 32K to 128K+ with minimal continual pretraining. Used by Mistral, DeepSeek, several Qwen long-context variants.

#### 4. NTK-aware / dynamic NTK scaling

Conceptual ancestor of YaRN. Treats RoPE through the lens of the neural tangent kernel and adjusts base or position to preserve high-frequency information. YaRN is generally preferred over plain NTK now.

### ALiBi (Attention with Linear Biases)

Different approach: drop positional encoding entirely, add a per-head linear bias `-m_h · |i − j|` to attention scores. Naturally extrapolates because the bias is well-defined at any position. MPT and BloombergGPT used this. Less popular in modern long-context models because it constrains the attention pattern more than RoPE.

### Why "trained 128K" ≠ "useful at 128K"

Even with a correct position encoding, the model must have **seen training data with informative long-range dependencies**. If your continual pretraining data has random unrelated documents concatenated to fill 32K, the model has no incentive to actually use positions past, say, 1K. The position encoding makes long context possible; the data and curriculum (Modules 06, 07) make it useful.

---

## Build it

### Setup

Take a small public model (e.g. Llama-3-8B base or a Qwen2.5-7B base — anything with RoPE). Set up an evaluation harness that runs needle-in-haystack at multiple positions and multiple total context lengths.

### Sweep

For each extension strategy:

- **Baseline**: model as-is, no extension.
- **RoPE base scaling**: continual-pretrain ~1B tokens at `N = 32K` with `rope_theta = 500_000`. (For a quick course lab, skip continual pretraining and just evaluate with the new base — quality will be worse but the comparison is instructive.)
- **Position interpolation**: at eval time set `pos = pos * (N_train / N_eval)`. No training.
- **YaRN**: same idea but with the YaRN formula. Reference implementation: <https://github.com/jquesnelle/yarn>.

Evaluate each on needle-in-haystack at:

- Insertion positions: `{1K, 4K, 8K, 16K, 24K, 31K}` (for 32K context).
- Total context: `{8K, 16K, 32K}`.
- Reusable templates: <https://github.com/gkamradt/LLMTest_NeedleInAHaystack>.

### Plot

```
y = retrieval accuracy (0..1)
x = needle position
lines = (baseline, base-scaling, PI, YaRN)
panels = per total-context value
```

You should see:

- Baseline collapsing past the training length.
- Position interpolation working at modest extension, degrading at large extension.
- YaRN holding accuracy across the range.

If you can run a small continual-pretraining job (~1B tokens) for RoPE base scaling, that should match or beat YaRN.

---

## Use it in the real stack

The `transformers` library exposes RoPE scaling via the `rope_scaling` field in the config. Concrete configs:

- Llama 3.1 long-context: `rope_theta = 500000.0` baked into the model.
- Qwen2.5 with YaRN: `rope_scaling = {"type": "yarn", "factor": 4.0, "original_max_position_embeddings": 32768}`.
- Mistral 7B 32K: same YaRN config.

When you adopt one of these models for downstream work, the config tells you what extension strategy is already applied. Do not stack a second one on top — you will break the rotation pattern further.

For your own training runs, the corresponding Megatron-LM flags:

```
--position-embedding-type rope
--rotary-base 500000
--rotary-percent 1.0
--rotary-seq-len-interpolation-factor 1.0   # = no PI; raise for PI
```

YaRN in Megatron is supported via a custom `--rotary-base-strategy yarn` in newer forks (check current main).

---

## Measure it

Per (strategy, position, context length) row:

- Retrieval accuracy (binary: needle found / not).
- Aggregate across needles per (strategy, context length) — report mean and worst-position.
- Perplexity on a held-out long document (sanity that nothing is broken globally).

Plot **per-position** accuracy. A single average can hide "lost in the middle" — the failure mode where positions in the middle 50% of the context are systematically worse than the edges.

---

## Ship it

Drop into `lcm-course/`:

1. `position_encoding_sweep.csv` and `position_encoding_sweep.png` — the per-position plot.
2. `rope_extension_notes.md` — one paragraph each on RoPE base scaling, PI, YaRN, ALiBi, with the failure mode each fixes.
3. A short recommendation: which scheme you would pick to go from 4K → 32K, 32K → 128K, and 128K → 1M, with one reason per choice.

---

## Related pages

- [Module 02 — Long-context attention mechanics](02-Long-Context-Attention.md)
- [Module 06 — Adaptive data pipelines](06-Adaptive-Data-Pipelines.md)
- [Module 07 — Long-context evaluation](07-Long-Context-Evaluation.md)
- YaRN paper: <https://arxiv.org/abs/2309.00071>
- "Lost in the Middle" paper: <https://arxiv.org/abs/2307.03172>
- Effective Long-Context Scaling (Meta): <https://arxiv.org/abs/2309.16039>
