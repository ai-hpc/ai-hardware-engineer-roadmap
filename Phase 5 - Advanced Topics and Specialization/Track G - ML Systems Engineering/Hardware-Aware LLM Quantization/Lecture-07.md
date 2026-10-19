# Module 07 — Layer Sensitivity: Measure the Operator and the Task

**Collection:** [Hardware-Aware LLM Quantization](README.md) | **Previous:** [Previous lesson](Lecture-06.md) | **Next:** [Next lesson](Lecture-08.md)

The question is whether a quantized component changes an important output, not
whether its histogram looks unusual. Q/K, MLP, output heads and recurrent gates
can each be sensitive. There is no model-independent ranking that lets us skip
measurement.

## 1. What a linear layer does to weight error

For a row vector `x`, weights `W`, and quantization error `E`,

```text
y = x W;       y_quant - y = x E
||x E||₂ ≤ ||x||₂ ||E||₂
```

If scalar errors are independent, zero mean, variance `sigma²`, then for one output
`Var(delta_y)=sigma² sum_i x_i²`. A `1/sqrt(K)` **relative** error improvement needs
a signal that grows coherently with K. It is not a general property of dot products.
For `x=(1,1)` and `w=(1,-1)`, the output is zero: tiny error can have unbounded
relative size. Residual additions can attenuate or compound perturbations.

**Worked example:** two equal-size weight errors are `E₁=(0.1,0)` and `E₂=(0,0.1)`.
For deployment input `x=(100,1)`, their output errors are 10 and 0.1. Equal weight
MSE does not imply equal layer-output MSE. This is why calibration activations matter.

## 2. Attention scores and softmax

For `s_i=qᵀk_i/sqrt(d)`, perturbed vectors give the exact score change:

```text
delta_s_i = (delta_qᵀ k_i + qᵀ delta_k_i + delta_qᵀ delta_k_i) / sqrt(d)
```

Dropping the last term is a first-order approximation. Score changes depend on
norms, alignment and correlations, not only on the quantizer bit-width.

Softmax does **not** exponentially amplify every small absolute logit error.
Its Jacobian is `J=diag(p)-ppᵀ`; the absolute row sum is `2p_i(1-p_i)≤1/2`.
Consequently `||softmax(s+e)-softmax(s)||∞ ≤ 0.5 ||e||∞`. Adding the same constant
to all logits changes no probabilities. Relative changes in tiny probabilities
can be large, and a small change near a tie can change argmax; these are different
statements from unbounded absolute amplification.

**Worked example:** logits `(0,0)` give probabilities `(0.5,0.5)`. Changing them to
`(0.01,-0.01)` gives about `(0.505,0.495)`. Argmax changes under a tie rule while
absolute probability error remains small. Value magnitude matters too:
`delta_output = delta_P V + P delta_V + delta_P delta_V`.

## 3. RoPE and context length

An exact RoPE rotation is orthogonal: `||R(position)e||₂=||e||₂`. Quantizing Q/K
weights does not by itself introduce a frequency error that grows linearly with
position. A numerical error in frequency/position computation is a **separate**
mechanism. Do not infer monotonic quantization damage from position alone.

Long-context evaluation is still necessary: it changes keys, attention patterns,
retrieval difficulty, cache precision, distributions and implementation paths.
Compare short/long matched tasks and actual position handling. For hybrid models,
measure recurrent state and gates as well as the full-attention projections.

## 4. Read the case-study ablation correctly

The pinned **DSpark drafter** card compares keeping its Q/K/V wide with quantizing
them. It reports a decrease in mean acceptance from 2.792 to 2.546 and a throughput
increase from 147.87 to 150.73 tok/s for that ablation. These are upstream results,
not locally reproduced measurements. The target remains the target; this is an
ablation of **proposal quality/cost**. [Source and revision](Source-Evidence.md).

Under exact speculative sampling, a worse drafter changes efficiency while the
output distribution remains the selected target distribution. These results do
not show that the target lost language quality, that Q/K are universally worst,
or that its weight traffic dropped 10.5%. The two models must have separate ledgers.

## 5. A sensitivity experiment you can defend

1. Freeze train/calibration/development/test splits, reference revision, tokenizer,
   preprocessing, sampling transform and metrics.
2. Quantize one group at a time; measure layer-output error on calibration data and
   model/task metrics on development data. Keep the untouched reference immutable.
3. Include groups that exist in this model: MLP, full-attention Q/K/V/O, hybrid
   recurrent gates, embeddings and output head. Do not invent missing layers.
4. Test the strongest candidate pairs jointly. Their errors need not add.
5. Choose a policy on development data; evaluate held-out and shifted data once.
6. If optimizing the drafter, freeze the target and measure draft/verify time,
   emitted tokens per cycle, and task outputs using correct target sampling.

The [CPU lab](CPU-Quantization-Lab.md) compares weight reconstruction error with
output-aware clipping on a trained nonlinear classifier. It teaches selection
and failure analysis; its sensitivity ordering is not an LLM benchmark.

## Self-check and answers

- **Does a larger matrix guarantee smaller relative output error?** No; cancellation
  and correlated error defeat the averaging argument.
- **Can two quantizers have equal weight MSE and different output error?** Yes;
  calibration input covariance weights different error directions differently.
- **Does lower drafter acceptance mean worse final answers?** Not for a correct
  exact sampler with the same target; it affects speed.
- **Can a low-outlier layer be sensitive?** Yes; logits near decision boundaries,
  recurrence and downstream operators can make small error consequential.

**Deliverable:** a per-group sensitivity table, one joint-error test, and an explicit
separation between target fidelity, task quality and drafter agreement.
