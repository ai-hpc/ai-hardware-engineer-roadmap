# Module 05 — Calibration: Optimize an Error You Can Explain

**Collection:** [Hardware-Aware LLM Quantization](README.md) | **Previous:** [Previous lesson](Lecture-04.md) | **Next:** [Next lesson](Lecture-06.md)

Calibration chooses scales, clipping or rounding policies using representative
inputs. It is part of fitting the quantized model. Its data must be separated from
the final test, and its objective must match the quantity you care about.

## 1. PTQ and QAT

Post-training quantization starts from trained weights and fits a quantization
policy. Some methods only collect statistics; some optimize rounding/scales with
gradients. “PTQ means no gradients” is therefore too restrictive. Quantization-aware
training places simulated quantization into training/fine-tuning and adapts weights
to it. Either approach can require substantial compute and can damage capabilities.
There is no universal one-hour PTQ or hundred-GPU-hour QAT cost for a 27B model.

Start with a supported PTQ baseline, measure the remaining gap, then justify more
expensive adaptation. QAT may help low bits, activation quantization or difficult
tasks, but hardware compatibility and independent evaluation are still necessary.

## 2. Split the data before calibrating

Distinguish model training, quantizer calibration, policy development and final
test. Group related samples by conversation/document/robot episode so near-duplicates
do not cross splits. Fit preprocessing on the designated training data only.
Representative calibration should cover deployment domains, modalities, sequence
lengths, position handling and important rare conditions.

A starting budget such as 128–512 sequences is a **pilot choice**, not a law.
Sweep a bounded number of calibration samples and stop when the policy and
held-out-development metrics stabilize within a declared tolerance. Match a
length distribution, not only its median. Never repeatedly inspect the final test
while deciding scale ranges or exclusions.

For the CPU lab, training episodes fit a small neural model; separate calibration
episodes select weight clipping; development episodes select the model checkpoint;
fixed test and biased-sensor test episodes are evaluated after these decisions.
This synthetic fixture teaches the protocol, not real collision safety.

## 3. Three different objectives

For a quantized weight matrix Q(W), these losses need not choose the same policy:

```text
weight reconstruction: mean((Q(W)-W)^2)
layer output:          mean((X Q(W)-X W)^2)       X from calibration
end-to-end fidelity:   mean(KL(reference_logits || candidate_logits))
```

The layer objective depends on the input covariance. Large weight error in a
rarely used direction may matter less than small error in a frequent direction.
End-to-end task metrics can disagree with either proxy. Measure them independently.

**Worked example:** x=(100,1). A weight error `(0.01,0)` has output error 1; error
`(0,0.1)` has output error 0.1 despite ten times the weight-error norm. Choosing
only the smallest weight error would select the worse output here.

## 4. Scale estimators and clipping

For a grid with maximum magnitude q_max, absmax chooses `s=max(abs(x))/q_max` before
rounding the stored scale. If the scale itself rounds down, some values can exceed
the reconstructed range; “absmax guarantees zero clipping” needs that qualification.
All-zero blocks need a positive-scale/zero-code convention, not division by zero.

Percentile clipping trades tail damage for resolution. MSE search enumerates
candidate thresholds c and minimizes reconstruction or output error. For floating
grids the error curve need not be smooth, convex, or monotonically improved by
smaller c. A grid search finds the best **searched** candidate, not a continuous
optimum. Include absmax so the searched calibration loss cannot be worse than that
candidate; independent test loss still can be worse.

```python
# Conceptual search; the executable lab includes real E4M3 rounding and padding.
candidates = [0.50, 0.65, 0.80, 0.90, 1.00]
results = []
for ratio in candidates:
    artifact = pack_fp4(weight, clip_ratio=ratio)
    reconstructed = unpack_fp4(artifact)
    error = ((calibration_inputs @ reconstructed.T
              - calibration_inputs @ weight.T) ** 2).mean()
    results.append((float(error), ratio, artifact))
best = min(results, key=lambda row: (row[0], -row[1]))
```

Histogram entropy calibration is a particular proxy with binning/range choices;
it is not output-token KL. TensorRT's older implicit INT8 calibrators are not a
generic NVFP4 recipe. TensorRT 11.4 uses its documented explicit workflows; use
the chosen release's [quantization workflow](https://docs.nvidia.com/deeplearning/tensorrt/latest/inference-library/quantized-types-workflows.html).

## 5. AWQ, GPTQ and SmoothQuant

These methods motivate different ways to improve a quantizer. Their original
formats and implementations are not interchangeable with every NVFP4 exporter.

**AWQ** uses activation information to identify important channels and select
weight/activation rescaling that reduces low-bit weight error. For row-vector
inputs, `XW = (X diag(s)^-1)(diag(s)W)`. Scaling before rounding changes the error.
Folding that scaling into an adjacent operation can remove runtime work **when**
the graph and all consumers permit it; it is not universally free.
[Original AWQ paper](https://arxiv.org/abs/2306.00978).

**GPTQ** approximately minimizes layer-output reconstruction error by quantizing
weights and compensating the remaining ones using calibration covariance/Hessian
information. For one output row, squared error gives a Hessian proportional to
`XᵀX` under a compatible orientation. Damping, dead columns, ordering and blockwise
updates matter. A schematic inverse-Hessian update is not drop-in code for the
implementation's Cholesky factor: its diagonal and update interpretation differ.
[Original GPTQ paper](https://arxiv.org/abs/2210.17323).

**SmoothQuant** moves input-channel dynamic range into weights through the same
algebraic rescaling, targeting weight-and-activation quantization. Its original
W8A8 results do not prove W4A4 success. If activations stay wide, its activation-
error motivation is absent, but rescaling still changes weight quantization, so
“it mathematically does nothing for W4A16” is false.
[Original SmoothQuant paper](https://arxiv.org/abs/2211.10438).

Do not assume toolkits implement every algorithm for every dtype. Record exporter
version, scheme, exclusions, calibration method and the **actual produced tensors**.
The pinned community case advertises 4-bit input activations and weights in its
quantization config; it must not be relabeled W4A16 from weight-file size alone.
[Metadata evidence](Source-Evidence.md).

## 6. Manifest and completion

Keep dataset IDs/hash, grouping rule, seed, preprocessing, candidate thresholds,
objective, clipping/rounding conventions, scale dtype/granularity, block axis,
retained tensors and serialization revision. A data hash identifies bytes but does
not by itself make inaccessible private data reproducible; provide an authorized
source or a licensed small fixture.

Run the [CPU lab](CPU-Quantization-Lab.md) and compare absmax, weight MSE and output
MSE. Submit chosen ratios and calibration losses **before** test results, then the
full task metrics on clean and shifted observations. Explain why lower calibration
loss is a selection result, not a guarantee of better held-out accuracy.
