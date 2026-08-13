# tinygrad — The Hackable Compiler Framework

<div class="course-identity auto-course" style="--course-accent: #0891b2; --course-accent-rgb: 8, 145, 178;" markdown="1">
<div class="course-identity__icon">TTHC</div>
<div markdown="1">
<p class="course-identity__eyebrow">Deep Dive · AI Workloads</p>
<p class="course-identity__title">Specialized course identity for tinygrad — The Hackable Compiler Framework.</p>
<p class="course-identity__meta">Artifact: model or workload study · Measure: accuracy, latency, memory, throughput</p>
</div>
</div>


**Parent:** [Module 2 — Deep Learning Frameworks](../Guide.md)

> *A minimal DL framework (~10K lines) that exposes the entire compiler pipeline in readable Python. The ideal codebase for understanding what happens between `loss.backward()` and the GPU kernel that actually runs.*

---

## Why tinygrad Is Uniquely Valuable

- It's the inference engine inside **openpilot** (Phase 5E)
- It exposes the IR, scheduler, and code generation that **Phase 4C** teaches you to build
- You can add a **custom backend** (Phase 4C §7) targeting your own accelerator
- It runs on CUDA, OpenCL, Metal, LLVM, and custom targets

## Key Concepts

| Concept | What it teaches | Stack connection |
|---------|----------------|-----------------|
| Lazy evaluation | Nothing runs until `.realize()` | L2: compiler decides when to execute |
| 3 operation types | Elementwise, Reduce, Movement (25 primitives) | L5: what the PE array must support |
| ShapeTracker | Zero-copy reshapes and transposes | L2: memory layout optimization |
| UOp IR | Intermediate representation before codegen | L2: same concept as MLIR/TVM IR |
| BEAM search | Explores fusion choices to minimize runtime | L2: auto-tuning |
| Backends | Same IR generates CUDA, OpenCL, LLVM code | L2: multi-target compilation |

## Projects

1. **Trace a matmul:** `Tensor` → lazy buffer → scheduled ops → generated CUDA kernel. Document every step.
2. **DEBUG=4:** Run a small model, read the generated kernels. Count kernel launches.
3. **BEAM=3 vs BEAM=0:** Compare kernel count and latency on the same model.
4. **Logging backend:** Add a minimal hook that prints each kernel launch.

## Deep Dive

The full tinygrad learning path (11 parts, 7 projects) is in [Phase 5E — Autonomous Vehicles / tinygrad](../../../Phase%205%20-%20Advanced%20Topics%20and%20Specialization/Track%20E%20-%20Autonomous%20Vehicles/3.%20tinygrad%20for%20Inference/Guide.md).

## Resources

- [tinygrad GitHub](https://github.com/tinygrad/tinygrad)
- [tinygrad Discord](https://discord.gg/tinygrad)
