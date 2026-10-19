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

> *A compact DL framework that exposes the entire compiler pipeline in readable Python. The ideal codebase for understanding what happens between `loss.backward()` and the GPU kernel that actually runs.*

---

## Why tinygrad Is Uniquely Valuable

- It's the inference engine inside **openpilot** (Phase 5E)
- It exposes the IR, scheduler, and code generation that **Phase 4C** teaches you to build
- You can add a **custom backend** (Phase 4C §7) targeting your own accelerator
- Supported targets and compiler paths depend on the release; inspect the pinned runtime directory and device availability

## Key Concepts

| Concept | What it teaches | Stack connection |
|---------|----------------|-----------------|
| Lazy evaluation | Builds lazy graphs; realization and host-value access trigger execution | L2: compiler decides when to execute |
| Operation families | Elementwise, reductions, and movement; inspect the pinned IR | L5: what the PE array must support |
| Layouts and views | Determine when a view can avoid a copy | L2: memory layout optimization |
| UOp IR | Intermediate representation before codegen | L2: same concept as MLIR/TVM IR |
| BEAM search | Searches supported kernel optimization choices | L2: auto-tuning |
| Backends | Inspect code generation and runtime separately for the selected target | L2: multi-target compilation |

## First executable route

Run the [tested tinygrad 0.14.0 CPU compiler lab](../../../Phase%204%20-%20Track%20C%20-%20DL%20Inference%20Optimization/00%20-%20CPU%20Compiler%20Lab/Guide.md). Compare materialized, lazy, and folded variants against a scalar oracle before counting kernels. It provides a hashed wheel pin, non-divisible dimensions, changed-input checks, and generated-code tracing. More fusion is not automatically lower end-to-end latency.

Current [developer layout](https://docs.tinygrad.org/developer/layout/) describes UOps across tensor/gradient/compiler stages. `LazyBuffer` belongs to earlier implementations. Set process configuration before importing tinygrad, or use its supported `Context` override; changing an environment variable after import need not update cached settings. Backend code and debug output are version-specific.

## Projects

1. **Trace a matmul:** `Tensor` → UOp graph → scheduled kernels → generated target code. Document every step.
2. **DEBUG=4:** Run a small model, read the generated kernels. Count kernel launches.
3. **BEAM=3 vs BEAM=0:** Compare kernel count and latency on the same model.
4. **Logging backend:** Add a minimal hook that prints each kernel launch.

## Deep Dive

The full tinygrad learning path (11 parts, 7 projects) is in [Phase 5E — Autonomous Vehicles / tinygrad](../../../Phase%205%20-%20Advanced%20Topics%20and%20Specialization/Track%20E%20-%20Autonomous%20Vehicles/3.%20tinygrad%20for%20Inference/Guide.md).

## Resources

- [tinygrad GitHub](https://github.com/tinygrad/tinygrad)
- [tinygrad Discord](https://discord.gg/tinygrad)


## Carry this course into the physical-AI system

**Start from:** A passing PyTorch reference and a pinned tinygrad commit.

**Build and challenge:** Trace one matmul-plus-activation to IR and generated code. Change a dimension so a tile has a tail and verify output before comparing kernel count.

**Pass forward:** A minimal tensor program and correctness vectors for a custom backend. Continue with [the connected course](../../../Phase%205%20-%20Advanced%20Topics%20and%20Specialization/Track%20G%20-%20ML%20Systems%20Engineering/TVM%20Deep%20Dives/README.md) and keep the artifact in the [Physical AI Design Studio](../../../Physical-AI-Design-Studio.md).
