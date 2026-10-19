# 04 — Quantization & Low-Precision Inference

<div class="course-identity auto-course" style="--course-accent: #475569; --course-accent-rgb: 71, 85, 105;" markdown="1">
<div class="course-identity__icon">QLPI</div>
<div markdown="1">
<p class="course-identity__eyebrow">Deep Dive · Compiler Track</p>
<p class="course-identity__title">Specialized course identity for 04 — Quantization & Low-Precision Inference.</p>
<p class="course-identity__meta">Artifact: compiler/inference optimization · Measure: op count, memory, latency</p>
</div>
</div>


**Order:** Fourth. After you have kernels and compiler context (01–03), you add low-precision kernels and deployment.

**Role target:** DL Inference Optimization Engineer · **MTS Kernels** (INT8/INT4 kernels, quantization-aware implementations).

---

## Why this comes fourth

Quantization (INT8, INT4, etc.) is a major lever for latency and throughput. It touches graph transforms, kernel implementations (integer matmul, quantized attention), and runtime integration. Doing 01–03 first lets you see where quantization plugs in and how it affects kernel choice and fusion.

---

## 1. Post-training quantization (PTQ)

* **INT8 / INT4** — Calibration (representative data), per-tensor vs per-channel scales, calibration datasets.
* **Tooling** — TensorRT explicit Q/DQ, ONNX Runtime quantization, and [torchao](https://docs.pytorch.org/ao/stable/index.html). PyTorch documents migration away from older `torch.quantization`/`torch.ao.quantization` workflows; match the recipe to installed versions.
* **Kernel impact** — Inspect whether the runtime uses a native low-precision kernel, fused unpack/dequantization, a floating-point fallback, or an unsupported operator. A file containing INT4 weights does not determine the arithmetic instructions.

## Worked example: representation before speed

For affine integer quantization, define the positive scale `s`, integer zero-point `z`, integer range, and rounding rule:

```text
q = clamp(round(x / s) + z, q_min, q_max)
x_reconstructed = s × (q - z)
```

With signed INT4, `q_min=-8`, `q_max=7`, `s=0.1`, `z=0`, and round-to-nearest/ties-to-even:

| Original value | Integer | Reconstructed | Absolute error |
|---|---|---|---|
| −0.16 | −2 | −0.20 | 0.04 |
| 0.04 | 0 | 0.00 | 0.04 |
| 0.27 | 3 | 0.30 | 0.03 |
| 0.90 | 7 | 0.70 | 0.20 |

The final row saturates. Rounding error without clipping is bounded by half a quantization step; clipping can make the error larger. FP4 formats use a floating-point codebook and possibly multiple scales, so this integer formula is not an NVFP4 implementation.

**Memory calculation:** 64 INT4 weights use 32 packed payload bytes. Add a 2-byte scale per 64-value group and the total is 34 bytes before other metadata, padding, or alignment. Against FP16's 128 bytes, that is about `3.76×` smaller, not automatically `4×`. It is a storage estimate, not measured speedup.

**Self-check:** if a zero-point is swapped with a scale, can output shape checks detect it? **Answer:** usually not. Reconstruct known values and compare outputs; a semantic parameter error can preserve every shape.

## Current support is a tuple, not a dtype name

Record `(runtime release, device, operator, weight format, activation format, accumulation dtype, scale layout)`. Verify the actual executed kernel before claiming acceleration. Calibration uses representative development data; held-out evaluation data remains separate.

**Reviewed 2026-10-18 (TensorRT 11.4 `latest` documentation):** the [TensorRT explicit-quantization documentation](https://docs.nvidia.com/deeplearning/tensorrt/latest/inference-library/quantized-types-explicit-quantization.html) uses Q/DQ with strong typing, and [its scheme matrix](https://docs.nvidia.com/deeplearning/tensorrt/latest/inference-library/quantized-types-schemes.html) places specific restrictions on INT4 and block formats. TensorRT 11.4 networks are strongly typed by default. Older TensorRT releases used different precision/calibration workflows. TensorRT 11.4 does not support JetPack: use the installed JetPack-supported 10.x documentation for Jetson, as explained in [NVIDIA's migration boundary](https://docs.nvidia.com/deeplearning/tensorrt/latest/api/migration/tensorrt-10x-to-11x-jetson.html). Use the documentation matching the target's supported release, particularly on Jetson. The [PyTorch migration page](https://docs.pytorch.org/docs/stable/quantization.html) directs new quantization development to torchao.

---

## 2. Quantization-aware training (QAT)

* **Fake quantization** — Simulate quantization in the forward pass; straight-through estimator for gradients.
* **Accuracy recovery** — Fine-tuning to recover accuracy after quantization.
* **When to use QAT vs PTQ** — Accuracy vs effort; production trade-offs.

---

## 3. Advanced formats and methods

* **Precision** — FP16/BF16 on GPUs; INT4/INT8 on CPUs and accelerators.
* **LLM-focused** — SmoothQuant, GPTQ, AWQ for LLM inference (concepts and where they hook into the stack).

---

## Resources

* [TensorRT Explicit Quantization](https://docs.nvidia.com/deeplearning/tensorrt/latest/inference-library/quantized-types-explicit-quantization.html)
* [torchao Workflows](https://docs.pytorch.org/ao/stable/workflows/index.html)
* [tinygrad](https://github.com/tinygrad/tinygrad) — Search for quantization passes and kernel support.

---

## Projects

1. **INT8 with TensorRT** — Quantize a CNN to INT8 using TensorRT. Report accuracy change and latency/speedup vs FP32.
2. **PTQ vs QAT** — Compare PTQ and QAT on the same model and target. Document accuracy, latency, and engineering effort for each.
3. **Kernel path** — For one quantized model (e.g. TensorRT or ONNX Runtime), identify which kernels run for a few key layers (e.g. INT8 conv/GEMM). Document the dispatch path.

---

## Next

→ **[05 — Inference Runtimes and Deployment](../05%20-%20Inference%20Runtimes%20and%20Deployment/Guide.md)** — Production deployment (TensorRT, ONNX Runtime, Triton server) and measurable outcomes.


## Integration lab: Does four-bit storage imply four-bit arithmetic?

Quantize a small matrix with explicit block/group scales, reconstruct it, and execute the supported target path. Measure scale/metadata overhead and detect an intentionally incorrect scale before benchmarking.

**Acceptance evidence:** Compare output and held-out task quality, then identify actual kernel dtypes. Orin does not gain native NVFP4 compute by loading an FP4 file; describe unpack/dequantization work where needed.

**Continue:** take this artifact to [05 - Inference Runtimes and Deployment](../05%20-%20Inference%20Runtimes%20and%20Deployment/Guide.md). Explain which assumption from this course the next engineer must preserve.
