# Lecture 02 — Quantization that preserves a deployment contract

**Input:** Lecture 01's checkpoint manifest. **Output:** one supported representation plus quality and kernel evidence. [Previous](Lecture-01.md) · [Next: runtime](Lecture-03.md).

## 1. Specify more than a bit width

A complete quantization description includes weight and activation formats, group/block size, scale dtype, zero-point convention, packing, accumulator precision, and exceptions such as embeddings or output heads. Two files called “4-bit” need not share a layout or kernel. Do not infer native NVFP4 execution on Orin from the presence of FP4 weights: storage and executed arithmetic are separate.

Before converting, confirm that the pinned converter understands the exact Gemma architecture and that the target runtime understands its output. Use the [llama.cpp source and conversion tools](https://github.com/ggml-org/llama.cpp) or the chosen runtime's official support documentation; a historical minimum version number is not proof of current model support.

## 2. A complete numerical exercise

This symmetric INT4 teaching example uses values from -7 to 7 and a scale per row. It stores the codes in INT8 containers for inspection; it does not pack two weights per byte or encode GGUF/NVFP4. Run the complete block with Python and NumPy, for example in the [optional Qwen CPU environment](../Qwen%20Inference%20Optimization/Lab-Guide.md#2-run-the-optional-current-library-experiment). The [portable Qwen quantization exercise](../Qwen%20Inference%20Optimization/Lab-Guide.md#1-run-the-portable-experiments) implements the same mechanism with the standard library if NumPy is unavailable.

```python
import numpy as np

w = np.array([[0., -1., 2., 0.1], [0.02, -0.03, 0.04, 0.]], dtype=np.float32)
x = np.array([1., -2., 0.5, 3.], dtype=np.float32)
scale = np.maximum(np.max(np.abs(w), axis=1, keepdims=True) / 7,
                   np.finfo(np.float32).tiny)
q = np.clip(np.rint(w / scale), -7, 7).astype(np.int8)
w_hat = q.astype(np.float32) * scale
reference = w @ x
candidate = w_hat @ x
print("weights_max_abs", float(np.max(np.abs(w - w_hat))))
print("outputs", reference, candidate)
print("output_max_abs", float(np.max(np.abs(reference - candidate))))

# Nearest rounding without clipping has per-weight error at most scale/2.
# A small allowance covers arithmetic rounding in this FP32 example.
weight_bound = scale / 2 + 1e-6
assert np.all(np.abs(w - w_hat) <= weight_bound)
output_bound = (scale[:, 0] / 2) * np.sum(np.abs(x))
assert np.all(np.abs(reference - candidate) <= output_bound + 1e-6)

corrupted = q.astype(np.float32) * scale[::-1]
corruption_detected = not np.all(np.abs(w - corrupted) <= weight_bound)
assert corruption_detected
print("swapped_scales_rejected", corruption_detected)
```

The observed FP32 outputs are approximately `[3.3, 0.1]` and `[2.714286, 0.1]`, with maximum output error `0.585714`. The first row's scale is about `0.285714`; its weight error is at most `0.142858`, but the input can amplify that error. By the triangle inequality, the first output error is bounded by `(scale/2) × sum(abs(x)) = 0.928572`. This is a numerical reconstruction check, not a task-quality acceptance threshold.

There is a useful rounding trap: `-1 / (2/7)` is mathematically `-3.5`. FP32 division here lands just above that midpoint and rounds to `-3`; the portable Python-float exercise rounds to `-4`, giving first output `3.285714`. Both satisfy the rounding-error bound. A format contract must specify scale and arithmetic precision as well as the rounding rule; matching the label “INT4” is insufficient to demand identical integer codes.

Now put a large outlier in one row. Predict which small values disappear before running it. Compare per-row and per-tensor scaling. The supplied scale-swap assertion must still detect the original corrupted reconstruction; check any changed fixture independently. This tests the mechanism rather than relying on a model producing plausible prose.

## 3. Account for metadata and temporary storage

For an illustrative group of 32 four-bit weights plus one FP16 scale, payload is `16 + 2 = 18` bytes, or **4.5 bits/weight**, before padding or zero points. Four times smaller weights do not imply four times lower peak memory or four times faster inference. Include scale loads, dequantization work, cache traffic, and unchanged components.

Use held-out examples that represent the deployment: text instructions, changed lighting, image detail, and language variation where relevant. Calibration inputs and evaluation inputs must be separate. Keep preprocessing and decoding settings identical across representations.

## 4. Conversion and measurement gates

1. Save the source checkpoint/processor hashes and converter commit.
2. Convert with a documented path; save the command, metadata, and output hashes.
3. Load a known input and inspect whether the intended backend and kernels run.
4. Compare reference outputs or logits where accessible, plus the actual task metric.
5. Measure memory and timing on the target only after those checks pass.

A GPU with more RAM can host the float reference if the Jetson cannot. Clearly state that reference quality was measured elsewhere, and do not compare unlike hardware as an optimization speedup.

## Lab and acceptance

Submit two representations of the same checkpoint, the exact format contract, numerical checks, held-out quality, and peak runtime memory. Declare the permitted quality loss before measurement. Retain a rejected setting and explain why it failed. If conversion support is absent, document the exact failing operation/version and use the numerical exercise; do not fabricate a runnable Gemma command. Lecture 03 consumes the accepted artifact and compatibility record.
