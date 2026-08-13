# 03 — Compiler Stack for Inference (IR, Scheduling, Codegen)

<div class="course-identity auto-course" style="--course-accent: #0891b2; --course-accent-rgb: 8, 145, 178;" markdown="1">
<div class="course-identity__icon">CSFI</div>
<div markdown="1">
<p class="course-identity__eyebrow">Deep Dive · Compiler Track</p>
<p class="course-identity__title">Specialized course identity for 03 — Compiler Stack for Inference (IR, Scheduling, Codegen).</p>
<p class="course-identity__meta">Artifact: compiler/inference optimization · Measure: op count, memory, latency</p>
</div>
</div>


**Order:** Third. After graph/ops (01) and kernel authoring (02), you see how compilers generate and schedule kernels.

**Role target:** DL Inference Optimization Engineer · **MTS Kernels** (Member of Technical Staff, Kernels — roles focused on code generation, compiler–hardware mapping, and owning kernel/backend implementation at scale).

---

## Why this comes third

Kernels (02) are either hand-written or compiler-generated. This unit covers how compilers represent the model (IR), decide fusion and placement (scheduling), and emit code (codegen). You need this to co-design with frameworks and to add or tune backends.

---

## 1. Intermediate representation (IR)

* **Graph IR vs linearized IR** — Graph: nodes = ops, edges = tensors. Linearized: list of ops in execution order (e.g. tinygrad's linearized ops).
* **SSA form** — Single assignment; each value defined once. Enables clear memory and alias analysis.
* **Memory and alias analysis** — Which buffers can overlap; when fusion or in-place is safe.

**Takeaway:** The IR is the contract between "model graph" and "kernel backend." Your kernels are targets for lowering from this IR.

---

## 2. Scheduling and lowering

* **tinygrad** — Scheduler, BEAM search for kernel fusion and placement. One op vs fused op; how BEAM explores fusion choices.
* **TVM** — TIR (Tensor IR), AutoTVM/AutoScheduler for mapping to hardware. Schedule primitives (tile, vectorize, parallel).
* **MLIR** — linalg/tensor dialects; progressive lowering (linalg → loops → vector → gpu). How high-level ops become loops and then GPU kernels.

**Takeaway:** Scheduling decides *which* kernels run (fused or not) and *how* they're tiled/parallelized; codegen then emits CUDA/LLVM/etc.

---

## 3. Code generation

* **Backend codegen** — From IR/schedule to CUDA, OpenCL, LLVM, or custom target. Role of codegen in Triton, TVM, tinygrad.
* **Kernel fusion and tile selection** — How the compiler chooses tile sizes and fusion sets for GPUs and accelerators.

---

## Resources

* [tinygrad](https://github.com/tinygrad/tinygrad) — IR, scheduler, BEAM, backends.
* [TVM Documentation](https://tvm.apache.org/docs/) — TIR, AutoTVM, BYOC.
* [MLIR Tutorial](https://mlir.llvm.org/docs/Tutorials/) — Dialects and lowering.

---

## Projects

1. **BEAM in tinygrad** — Run tinygrad with BEAM on a small model. Compare scheduled kernel count and runtime vs default scheduler; document what BEAM fused.
2. **Fusion pass** — Implement a simple fusion pass (e.g. Conv+ReLU) in a graph you control (ONNX or tinygrad). Measure impact on kernel count and latency.
3. **Trace lowering** — Pick one op (e.g. matmul) in TVM or tinygrad and trace from high-level op to generated code. Document the lowering steps.

---

## Next

→ **[04 — Quantization](../04%20-%20Quantization/Guide.md)** — Low-precision inference (PTQ, QAT) and how it affects kernels and deployment.
