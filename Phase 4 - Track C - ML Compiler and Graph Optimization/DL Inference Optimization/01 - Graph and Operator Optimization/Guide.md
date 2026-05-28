# 01 — Graph and Operator Optimization

**Order:** First (foundation). You need to know *what* you're optimizing before writing kernels.

**Role target:** DL Inference Optimization Engineer · MTS Kernels

**Before this unit:** Read [Basic concepts](../Guide.md#basic-concepts-read-this-first) in the DL Inference Optimization guide (LLM inference, TensorRT-LLM/vLLM, distributed training, KV-cache, and why new hardware changes kernel design).

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
