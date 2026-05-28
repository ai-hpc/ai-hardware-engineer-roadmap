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
* **Tooling** — TensorRT INT8, ONNX Runtime QDQ, PyTorch `torch.quantization`. How each represents scales and zero-points.
* **Kernel impact** — Runtimes dispatch to quantized kernels (e.g. INT8 GEMM, INT4 matmul); understanding the kernel path for quantized models.

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

* [TensorRT Quantization](https://docs.nvidia.com/deeplearning/tensorrt/developer-guide/index.html#optimizing_int8_c)
* [PyTorch Quantization](https://pytorch.org/docs/stable/quantization.html)
* [tinygrad](https://github.com/tinygrad/tinygrad) — Search for quantization passes and kernel support.

---

## Projects

1. **INT8 with TensorRT** — Quantize a CNN to INT8 using TensorRT. Report accuracy change and latency/speedup vs FP32.
2. **PTQ vs QAT** — Compare PTQ and QAT on the same model and target. Document accuracy, latency, and engineering effort for each.
3. **Kernel path** — For one quantized model (e.g. TensorRT or ONNX Runtime), identify which kernels run for a few key layers (e.g. INT8 conv/GEMM). Document the dispatch path.

---

## Next

→ **[05 — Inference Runtimes and Deployment](../05%20-%20Inference%20Runtimes%20and%20Deployment/Guide.md)** — Production deployment (TensorRT, ONNX Runtime, Triton server) and measurable outcomes.
