# Module 03 — Positional Encoding and Usable Context

**Parent:** [Course](README.md) · **Before this:** [attention mechanics](02-Long-Context-Attention.md)

**Learn:** derive a rotary pair, identify what a scaling method changes, and design a fair extension experiment. **Artifact:** a position-by-length evaluation grid with actual tokenizer counts and an unchanged short-context control.

## Work through one rotary pair

For even rotary width `d`, pair `j` uses angular frequency `ω_j=θ^(−2j/d)`. A position `p` applies a two-dimensional rotation through angle `p*ω_j`:

```text
R(a) [x,y] = [x*cos(a) − y*sin(a), x*sin(a) + y*cos(a)]
```

For fixed content vectors, `(R(pω)q)ᵀ(R(tω)k)=qᵀR((t−p)ω)k`. The rotational factor depends on relative position. The full score still depends on the content vectors; arbitrary tokens at the same relative offset need not have the same score. [RoFormer](https://arxiv.org/abs/2104.09864).

**Hand check:** with `q=k=[1,0]`, the score is `cos((t−p)ω)`. Moving both positions by 100 leaves it unchanged. Rotating Q but forgetting K breaks this property. Apply positions from the actual sequence, including cached prefixes and packing semantics, rather than resetting each chunk to zero.

Increasing `θ` lowers frequencies with `j>0`; the highest-frequency pair `j=0` remains `ω=1`. The old explanation that raising the base slows every high-frequency pair was incorrect. Some fast pairs already complete many rotations inside the original training window. Extrapolation failures concern learned combinations of frequencies, distances, attention scaling and data, not a universal first unseen rotation.

## Four approaches and their limits

| Approach | Change | Question to test |
|---|---|---|
| Base change | Modify the frequency spectrum | Did continued training adapt to the altered representation? |
| Position interpolation | Replace `p` with `p/s` | Does compressed local separation hurt short-range tasks? |
| YaRN | Frequency-dependent interpolation plus attention scaling | Does this model/runtime implement the intended variant and original length? |
| ALiBi | Add a head-dependent distance penalty to attention scores | Does the recency bias fit the task and training architecture? |

For PI with `s=8`, position 24,000 uses phase corresponding to position 3,000 in the unscaled representation. Adjacent positions also become eight times closer. Keeping phases within an earlier range does not guarantee unchanged attention or quality. The original PI work includes fine-tuning; it is not a blanket promise of training-free extension. [PI paper](https://arxiv.org/abs/2306.15595).

YaRN preserves different frequency bands differently and modifies attention scaling. It is a named family of specific formulas and configuration values, not “set a longer maximum.” Read the checkpoint's declared original length, factor and implementation. No method is uniformly best for every model or extension ratio. [YaRN paper](https://arxiv.org/abs/2309.00071).

For causal ALiBi, an allowed key at distance `p−t` receives a bias such as `−m_h*(p−t)`. The formula exists for long distances, but usable extrapolation remains an empirical result. Replacing a pretrained RoPE model with ALiBi at inference changes its learned attention function and is not a compatible configuration tweak. [ALiBi paper](https://arxiv.org/abs/2108.12409).

## Read the exact configuration

As reviewed October 18, 2026, the current Transformers utility documentation describes `rope_parameters`, including per-layer-type settings. Older checkpoints and releases may serialize `rope_scaling`/`rope_theta`; migration and validation depend on the model class. Do not copy a Qwen configuration into a different model or assume one `type` key works across releases. [Transformers rotary utilities](https://huggingface.co/docs/transformers/main/en/internal/rope_utils).

Save model and tokenizer revisions, original configuration, resolved runtime configuration, loader version and the changed fields. A model can use partial rotary dimensions or different attention layer types. The configuration must be interpreted by the actual implementation.

The inspected Megatron source is linked in [source evidence](labs/sources.json). Use flags implemented by that checkout and recipe. A fork-only `--rotary-base-strategy yarn` is not a portable mainline command. Configuration validation and a small numerical test come before an expensive continuation run.

## Build a fair experiment

First run the unchanged checkpoint within its supported context. Test the same documents and questions under each proposed extension; record truncation, generation budget, precision and failed/OOM requests. If one variant receives continued training, report its training budget separately. Do not compare a trained variant against an untrained configuration change and attribute the entire gain to a positional formula.

For each total token length `N`, choose insertion depths such as 10%, 30%, 50%, 70% and 90%, then reserve room for the question and answer. Absolute insertion position 31K is invalid in an 8K test. Tokenize the assembled prompt and confirm the evidence survives preprocessing.

Use multiple independently generated documents and distractors. Add tasks requiring two distant facts, conflicting evidence, multi-file code reasoning and summarization. A one-needle success proves only that narrow retrieval behavior. Keep a frozen final set separate from the development grid used to choose the extension.

Record per-position success, sample count and uncertainty; retain invalid and failed requests in the declared denominator. A heatmap exposes positions that an average conceals. Repeat short-context tests to identify regressions. Accept a negative result rather than drawing the curve a tutorial predicted.

**Self-check answers:** a larger base does not change `ω₀`; PI changes local distances as well as maximum phase; accepting a million tokens does not establish understanding of a million tokens. The completion artifact is a reproducible experiment with these distinctions, not a particular winner.

[Next: MoE routing and training](04-MoE-Fundamentals.md)
