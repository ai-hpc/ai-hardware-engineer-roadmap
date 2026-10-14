# 01 — Graph and Operator Optimization

<div class="course-identity auto-course" style="--course-accent: #0284c7; --course-accent-rgb: 2, 132, 199;" markdown="1">
<div class="course-identity__icon">GAOO</div>
<div markdown="1">
<p class="course-identity__eyebrow">Deep Dive · Compiler Track</p>
<p class="course-identity__title">Specialized course identity for 01 — Graph and Operator Optimization.</p>
<p class="course-identity__meta">Artifact: compiler/inference optimization · Measure: op count, memory, latency</p>
</div>
</div>


**Order:** First (foundation). You need to know *what* you're optimizing before writing kernels.

**Role target:** DL Inference Optimization Engineer · MTS Kernels

**Before this unit:** Read [Basic concepts](../Guide.md#basic-concepts-read-before-part-2) in the track guide (LLM inference, TensorRT-LLM/vLLM, distributed training, KV-cache, and why new hardware changes kernel design).

---

## Why this comes first

Before writing or tuning kernels, you must:

1. **Understand the graph** — which ops run, in what order, and how they connect.
2. **Find bottlenecks** — which ops or layers are compute-bound vs memory-bound.
3. **Know fusion opportunities** — which op chains can become a single kernel (e.g. Conv–BN–ReLU).

This unit gives you the graph/operator view and profiling skills that every kernel engineer uses daily.

---

## 1. Graph-level optimizations

* **Constant folding** — Evaluate constant subgraphs at build time (e.g. shape ops, fixed weights).
* **Dead code elimination** — Remove ops whose outputs are never used.
* **Common subexpression elimination (CSE)** — Reuse computed values instead of recomputing.
* **Operator fusion** — Combine multiple ops into one kernel:
    * Conv–BN–ReLU, Linear–Activation, Attention (Q/K/V + softmax + matmul).
    * Reduces memory traffic and kernel launch overhead.
* **Layout and shape transformations** — NCHW vs NHWC, transpose folding, reshape/expand for hardware-friendly layouts.
* **Framework graph formats** — ONNX, TorchScript, TensorFlow SavedModel; how optimization passes are applied in each.

**Concepts to internalize:** A single "layer" in a model often becomes many ops in the graph; fusion turns them back into fewer, faster kernels.

---

## Worked rewrite: fold fixed batch normalization

For output channel c, let convolution produce `z_c = W_c * x + b_c`. In evaluation mode with fixed running mean `mu_c`, variance `v_c`, scale `gamma_c`, and shift `beta_c`, define:

```text
a_c  = gamma_c / sqrt(v_c + epsilon)
W'_c = a_c * W_c
b'_c = a_c * (b_c - mu_c) + beta_c
```

Then `Conv(x, W', b')` replaces the convolution-plus-BN output over real arithmetic. ReLU stays after that output; moving it before BN can change the result. If the convolution has no bias, start with `b_c=0`. Training-mode BN depends on the current batch statistics, so this fixed rewrite is not valid there. Floating-point rounding and later quantization still need output checks.

**Worked scalar channel:** `W=2`, `b=1`, `mu=3`, `gamma=4`, `beta=5`, and `sqrt(v+epsilon)=2` give `W'=4`, `b'=1`. At `x=2`, both expressions produce 9. Make gamma negative in a second example and show why changing activation order fails. Pass W'/b' and their dtype/layout to the kernel course, together with the graph hash and validation inputs.

## 2. Operator-level optimization

* **Kernel selection and dispatch** — How runtimes choose implementations: cuBLAS, cuDNN, oneDNN, or custom kernels. Algorithm selection (e.g. conv algorithm) and heuristics.
* **Memory planning** — Buffer allocation, in-place ops where safe (same buffer for input/output), reducing peak memory.
* **Batching and dynamic batching** — Batching requests for inference servers; trade-offs between latency and throughput.

---

## 3. Profiling and bottleneck identification

* **Tools:**
    * **Nsight Systems** — Timeline view: kernel launches, memory copies, CPU–GPU overlap.
    * **Nsight Compute** — Per-kernel: occupancy, memory throughput, compute utilization.
    * **PyTorch profiler** — Op-level and kernel-level timing in Python.
    * **ONNX Runtime** — Execution provider timing, operator cost.
* **Roofline-style analysis** — For each major op/layer: compute-bound vs memory-bound; arithmetic intensity and roofline limits.
* **End-to-end latency breakdown** — Data loading → preprocess → inference (per layer) → postprocess. Where does time go?

**Goal:** From a single model run, you should be able to name the top 3–5 bottlenecks and say whether they are compute or memory bound.

---

## Resources

* [TensorRT Developer Guide](https://docs.nvidia.com/deeplearning/tensorrt/developer-guide/) — Graph optimization and layer fusion.
* [ONNX Runtime Performance Tuning](https://onnxruntime.ai/docs/performance/) — Graph and execution provider tuning.
* [PyTorch Profiler](https://pytorch.org/tutorials/recipes/recipes/profiler_recipe.html) — Profiling PyTorch models.
* [NVIDIA Nsight Systems / Compute](https://developer.nvidia.com/nsight-systems) — GPU profiling.

---

## Projects

1. **Fusion and measure** — Take a ResNet-style model (or small transformer). Fuse Conv–BN–ReLU in ONNX or TorchScript (or use a framework that does it). Measure latency before and after; document the speedup.
2. **Profile and report** — Profile a transformer block (attention + FFN) with Nsight Systems and PyTorch profiler. Identify the top 3 bottlenecks; for each, state whether it is compute-bound or memory-bound and why.
3. **End-to-end breakdown** — For one inference pipeline (e.g. image → model → result), break down time into: data load, preprocess, each major graph region, postprocess. Draw a simple timeline and note the largest segment.

---

## Next

→ **[02 — Kernel Engineering](../02%20-%20Kernel%20Engineering/Guide.md)** — Design and implement the high-performance kernels that implement these ops (Triton, CUTLASS, Flash-Attention).


## Integration lab: When is a graph rewrite legal?

Fold evaluation-mode batch normalization into a convolution using fixed running statistics. Compare the reference and folded outputs; then demonstrate why training-mode batch statistics invalidate the same rewrite.

**Acceptance evidence:** Use irregular shapes and a fixed tolerance, report intermediate bytes and kernels removed, and hand the operator shape/dtype contract to kernel engineering.

**Continue:** take this artifact to [02 - Kernel Engineering](../02%20-%20Kernel%20Engineering/Guide.md). Explain which assumption from this course the next engineer must preserve.
