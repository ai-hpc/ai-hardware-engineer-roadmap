# 06 — tinygrad Deep Dive (Optional)

**Order:** Sixth, optional. Hands-on compiler/kernel interface after you've seen graph, kernels, compiler, quantization, and deployment.

**Role target:** DL Inference Optimization Engineer · **MTS Kernels** (compiler–kernel interface, custom backends).

---

## Why this is optional and last

tinygrad is a minimal stack that exposes IR, scheduling, and backends in one codebase. Doing it *after* 01–05 lets you connect everything: graph → scheduler → kernel selection/codegen → runtime. It's optional if your focus is only Triton/CUTLASS in a big framework; it's valuable if you want to add backends or compiler passes.

---

## 1. IR and ops

* **Linearized representation** — How tinygrad turns a graph into a linear list of ops.
* **Op types** — Unary, binary, reduce, movement, load/store; how they map to memory and compute.
* **Memory buffers** — How tensors and buffers are assigned and reused.

---

## 2. Scheduler

* **How ops are grouped and scheduled** — Which ops fuse; how the scheduler makes fusion decisions.
* **BEAM search** — Exploring fusion choices; trading kernel count vs kernel size and register pressure.

---

## 3. Backends

* **CPU, CUDA, OpenCL** — How tinygrad emits code for each. Where to add or tune a backend.
* **Custom backend** — What a minimal custom backend needs (codegen, launch, memory).

---

## 4. Quantization in tinygrad

* **Passes** — How quantization is represented and applied in the graph.
* **Integration with compiler** — How quantized ops flow through scheduler and backends.

---

## Resources

* [Phase 5 — Autonomous Driving / tinygrad](../../../Phase%205%20-%20Advanced%20Topics%20and%20Specialization/5.%20Autonomous%20Vehicles/3.%20tinygrad%20for%20Inference/Guide.md) — Hands-on tinygrad material in this repo.
* [tinygrad GitHub](https://github.com/tinygrad/tinygrad) — Codebase for study and contribution.

---

## Projects

1. **Trace the pipeline** — Run a small model in tinygrad. Trace from Python to scheduled kernels; document the pipeline (graph → linearized IR → scheduler → backend code).
2. **Add an optimization** — Implement a simple optimization (e.g. constant fold or fuse two ops) in tinygrad. Measure impact on kernel count and runtime.
3. **Backend hook** — Add a minimal "identity" or logging hook in a backend (e.g. log each kernel launch). Use it to verify which kernels run for a given model.

---

## You've completed the track

You've gone through: **Graph & operators → Kernels → Compiler → Quantization → Runtimes → (optional) tinygrad.**

Next steps: deepen the areas that match your role (e.g. more Triton/CUTLASS for MTS Kernels, or more TensorRT/Triton server for inference deployment), and keep building portfolio projects (kernels, benchmarks, portability reports).
