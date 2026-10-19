# 06 — tinygrad Deep Dive (Optional)

<div class="course-identity auto-course" style="--course-accent: #ea580c; --course-accent-rgb: 234, 88, 12;" markdown="1">
<div class="course-identity__icon">TDDO</div>
<div markdown="1">
<p class="course-identity__eyebrow">Deep Dive · Compiler Track</p>
<p class="course-identity__title">Specialized course identity for 06 — tinygrad Deep Dive (Optional).</p>
<p class="course-identity__meta">Artifact: compiler/inference optimization · Measure: op count, memory, latency</p>
</div>
</div>


**Order:** Sixth, optional. Hands-on compiler/kernel interface after you've seen graph, kernels, compiler, quantization, and deployment.

**Role target:** DL Inference Optimization Engineer · **MTS Kernels** (compiler–kernel interface, custom backends).

---

## Why this is optional and last

tinygrad is a minimal stack that exposes IR, scheduling, and backends in one codebase. Doing it *after* 01–05 lets you connect everything: graph → scheduler → kernel selection/codegen → runtime. It's optional if your focus is only Triton/CUTLASS in a big framework; it's valuable if you want to add backends or compiler passes.

---

## 1. IR and ops

* **UOp graphs and lowered programs** — Follow tensor-level UOps through graph scheduling, kernel optimization, linearization/rendering, and execution. `LazyBuffer` is part of older tutorials; use the [layout documentation](https://docs.tinygrad.org/developer/layout/) with the pinned source revision.
* **Op types** — Unary, binary, reduce, movement, load/store; how they map to memory and compute.
* **Memory buffers** — How tensors and buffers are assigned and reused.

---

## 2. Scheduler

* **How ops are grouped and scheduled** — Which ops fuse; how the scheduler makes fusion decisions.
* **BEAM search** — Search equivalent implementations of an individual kernel, such as tiling and local work distribution; graph grouping is the scheduler's responsibility. See [tinygrad speed documentation](https://docs.tinygrad.org/developer/speed/).

---

## 3. Backends

* **CPU, CUDA, OpenCL** — How tinygrad emits code for each. Where to add or tune a backend.
* **Custom backend** — What a minimal custom backend needs (codegen, launch, memory).

---

## 4. Quantization in tinygrad

* **Representation** — Inspect one concrete quantized model path and its scale/packing operations; do not assume every supported checkpoint implies a general quantization pass.
* **Integration with compiler** — Trace unpacking, dequantization, arithmetic dtype, and generated kernels on the named backend. Low-bit storage alone does not establish native low-bit compute.

---

## Resources

* [Phase 5 — Autonomous Driving / tinygrad](../../Phase%205%20-%20Advanced%20Topics%20and%20Specialization/Track%20E%20-%20Autonomous%20Vehicles/3.%20tinygrad%20for%20Inference/Guide.md) — Hands-on tinygrad material in this repo.
* [tinygrad GitHub](https://github.com/tinygrad/tinygrad) — Codebase for study and contribution.

---

## Projects

1. **Trace the pipeline** — Start with the [tested CPU lab](../00%20-%20CPU%20Compiler%20Lab/Guide.md). Document tensor/UOp graph → scheduled kernels → optimized kernel IR → rendered source → runtime execution. Preserve generated-code and launch evidence.
2. **Add an optimization** — Implement a simple optimization (e.g. constant fold or fuse two ops) in tinygrad. Measure impact on kernel count and runtime.
3. **Backend hook** — Add a minimal "identity" or logging hook in a backend (e.g. log each kernel launch). Use it to verify which kernels run for a given model.

---

## You've completed the track

You've gone through: **Graph & operators → Kernels → Compiler → Quantization → Runtimes → (optional) tinygrad.**

Next steps: deepen the areas that match your role (e.g. more Triton/CUTLASS for MTS Kernels, or more TensorRT/Triton server for inference deployment), and keep building portfolio projects (kernels, benchmarks, portability reports).


## Integration lab: Can a minimal compiler backend preserve model semantics?

Pin tinygrad to a commit and trace a reduction or matrix operator through graph grouping, code generation, allocation, and launch. Log one backend boundary without changing numerical behavior.

**Acceptance evidence:** Compare shape/layout variants against a reference; preserve the exact source paths from that commit. Pass an unsupported-op policy and launch/buffer ABI to the custom accelerator project.

**Continue:** take this artifact to [5. Runtime and Driver Development](../../Phase%204%20-%20Track%20A%20-%20Xilinx%20FPGA/5.%20Runtime%20and%20Driver%20Development/Guide.md). Explain which assumption from this course the next engineer must preserve.
