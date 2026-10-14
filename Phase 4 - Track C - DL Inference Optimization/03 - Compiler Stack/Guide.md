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

* **tinygrad** — The scheduler groups graph operations into kernels; BEAM searches implementation choices for an already formed kernel. Keep graph fusion decisions separate from kernel optimization search. See [tinygrad speed documentation](https://docs.tinygrad.org/developer/speed/).
* **TVM** — Separate Relax graph-level work from TensorIR scheduling and code generation. Pin a TVM release: older AutoTVM/AutoScheduler tutorials and newer tuning APIs are not interchangeable. Start with tile, vectorize, and parallel transformations and inspect the generated loops.
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

1. **BEAM in tinygrad** — Run tinygrad with BEAM on a small model. Compare scheduled kernel count and runtime vs default scheduler; distinguish scheduler grouping from BEAM kernel choices.
2. **Fusion pass** — Implement a simple fusion pass (e.g. Conv+ReLU) in a graph you control (ONNX or tinygrad). Measure impact on kernel count and latency.
3. **Trace lowering** — Pick one op (e.g. matmul) in TVM or tinygrad and trace from high-level op to generated code. Document the lowering steps.

---

## Next

→ **[04 — Quantization](../04%20-%20Quantization/Guide.md)** — Low-precision inference (PTQ, QAT) and how it affects kernels and deployment.


## Integration lab: What proof obligation connects a rewrite to a machine instruction?

Trace one matrix operator from graph IR through buffer planning, loop/kernel scheduling, and generated code. Add a legal rewrite with an explicit shape/dtype guard and a counterexample rejected by that guard.

**Acceptance evidence:** Save before/after IR and correctness results. Separate graph fusion from BEAM kernel search, and pass the lowered operator plus memory-layout contract to quantization/runtime work.

**Continue:** take this artifact to [04 - Quantization](../04%20-%20Quantization/Guide.md). Explain which assumption from this course the next engineer must preserve.

**Custom-accelerator handoff:** take the lowered operator, buffer-lifetime plan, and reference vectors to [Compiler, Runtime ABI, and SoC Integration](../../Phase%205%20-%20Advanced%20Topics%20and%20Specialization/Track%20F%20-%20AI%20Chip%20Design/Programmable%20VLA%20Inference%20Accelerator/Lecture-06.md). Specify command fields, alignment, DMA ownership, completion/error reporting, and fallback so the same contract can drive a functional simulator, RTL implementation, and Linux driver.
