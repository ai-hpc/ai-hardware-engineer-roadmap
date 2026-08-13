# Module 2 — Deep Learning Frameworks

<div class="course-identity dl-frameworks" markdown="1">
<div class="course-identity__icon">FWK</div>
<div markdown="1">
<p class="course-identity__eyebrow">Module 2 · Deep Learning Frameworks</p>
<p class="course-identity__title">Trace how micrograd, PyTorch, and tinygrad turn tensor code into executable workloads.</p>
<p class="course-identity__meta">Artifact: framework trace · Measure: graph shape, op count, memory, runtime</p>
</div>
</div>


**Parent:** [Phase 3 — Artificial Intelligence](../Guide.md)

> *Understand how software generates the workloads your hardware must run — from autograd to GPU kernels.*

**Prerequisites:** Module 1 (Neural Networks — understand what a forward/backward pass computes).

**Layer mapping:** **L1** (Application) — you use frameworks to build models. **L2** (Compiler) — tinygrad exposes the compiler pipeline that Phase 4C teaches you to build.

---

## Why a Dedicated Frameworks Module

Module 1 teaches you *what* neural networks compute. This module teaches you *how* — the software machinery that turns `model(x)` into GPU kernel launches. Understanding this machinery is essential because:

- **L2 (Compiler):** You can't build an ML compiler without understanding what frameworks produce (computational graphs, ops, tensors)
- **L5 (Architecture):** You can't design an accelerator without knowing which ops dominate real workloads
- **L6 (RTL):** You can't build a PE array without understanding the precision and data flow of actual training/inference

---

## Three-Framework Mental Model

Study these three frameworks in order. Each teaches a different level of the stack.

| Framework | What it teaches | Size | Your learning goal |
|-----------|----------------|------|-------------------|
| **[micrograd](micrograd/Guide.md)** | How autograd works — reverse-mode differentiation from scratch | ~100 lines | Build it yourself. Understand backprop at the code level. |
| **[PyTorch](PyTorch/Guide.md)** | Industry-standard API — tensors, modules, optimizers, data loading | Millions of lines | Use it fluently. Train models, export ONNX, profile with torch.profiler. |
| **[tinygrad](tinygrad/Guide.md)** | How a compiler turns tensor ops into GPU kernels — IR, scheduler, backends | ~10,000 lines | Read the source. Trace from `Tensor` to generated CUDA/OpenCL code. |

```
micrograd          PyTorch              tinygrad
(education)        (production)         (hackable production)
    │                  │                     │
    ▼                  ▼                     ▼
 Autograd          Full API             Compiler pipeline
 from scratch      industry standard    IR → scheduler → codegen
    │                  │                     │
    └──────────────────┴─────────────────────┘
              Understanding grows left → right
```

---

## 1. micrograd — Autograd from Scratch

**[micrograd](https://github.com/karpathy/micrograd)** by Andrej Karpathy. ~100 lines of Python. Implements:
- A `Value` class that tracks computation history
- Reverse-mode automatic differentiation (backpropagation)
- A tiny neural network API (`Neuron`, `Layer`, `MLP`)

**What you'll build:**
```python
from micrograd.engine import Value

# Forward pass
x = Value(2.0)
y = Value(3.0)
z = x * y + y ** 2  # z = 2*3 + 9 = 15

# Backward pass (autograd)
z.backward()
print(x.grad)  # dz/dx = y = 3.0
print(y.grad)  # dz/dy = x + 2y = 2 + 6 = 8.0
```

**Why it matters for hardware:** Every training accelerator must implement this backward pass. Understanding the computation graph and gradient flow tells you what memory access patterns and operations the hardware must support.

**Project:** Implement micrograd from scratch (don't copy — type it yourself). Train an MLP on a 2D classification dataset. Visualize the computation graph.

---

## 2. PyTorch — Industry Standard

**[PyTorch](https://pytorch.org/)** is the framework most models are written in. You need fluency here because:
- Models you deploy on hardware are written in PyTorch
- ONNX export comes from PyTorch (`torch.onnx.export`)
- `torch.compile` (Inductor) is a production ML compiler
- Profiling tools (`torch.profiler`, Nsight) show you where time is spent

**Key concepts to master:**

| Concept | Why it matters for hardware |
|---------|---------------------------|
| `torch.Tensor` | The data structure accelerators process. Shape, dtype, layout (contiguous, channels-last). |
| `nn.Module` | How models are structured. Layers → forward pass → computational graph. |
| Autograd (`loss.backward()`) | Generates the backward graph that training hardware executes. |
| Data loading (`DataLoader`) | CPU-GPU pipeline. Bottleneck if not overlapped with compute. |
| `torch.onnx.export()` | How models leave PyTorch and enter the compiler/runtime stack (Phase 4C). |
| `torch.compile()` | PyTorch's built-in compiler (Inductor). Generates Triton kernels. Connection to Phase 4C. |
| `torch.profiler` | Where is time spent? Kernel launches, memory copies, CPU overhead. |
| Mixed precision (`torch.cuda.amp`) | FP16/BF16 training — what tensor cores accelerate. |
| Quantization (`torch.ao.quantization`) | INT8 inference — what L6 PE arrays must support. |

**Projects:**
1. Train a CNN (ResNet-18) on CIFAR-10 from scratch. Profile with `torch.profiler`. Identify the top-3 time-consuming operations.
2. Export the trained model to ONNX. Visualize the graph with Netron. Count the total number of ops and parameters.
3. Apply post-training quantization (PTQ) to INT8. Measure accuracy drop and inference speedup on CPU.
4. Use `torch.compile()` on a transformer block. Compare eager vs compiled execution time.

---

## 3. tinygrad — The Hackable Compiler

**[tinygrad](https://github.com/tinygrad/tinygrad)** is a minimal DL framework (~10K lines) that exposes the entire compiler pipeline in readable Python. It's the ideal codebase for understanding what happens between `loss.backward()` and the GPU kernel that actually runs.

**Why tinygrad is uniquely valuable for this roadmap:**
- It's the inference engine inside openpilot (Phase 5E)
- It exposes the IR, scheduler, and code generation that Phase 4C teaches you to build
- You can add a custom backend (Phase 4C §7) — targeting your own accelerator
- It runs on CUDA, OpenCL, Metal, LLVM, and custom targets

**Key concepts:**

| Concept | What it teaches | Connection to stack |
|---------|----------------|-------------------|
| Lazy evaluation | Nothing runs until `.realize()` | L2: compiler decides when to execute |
| 3 operation types | Elementwise, Reduce, Movement (25 primitives total) | L5: what the PE array must support |
| ShapeTracker | Zero-copy reshapes and transposes | L2: memory layout optimization |
| UOp IR | The intermediate representation before code generation | L2: same concept as MLIR/TVM IR |
| BEAM search | Explores fusion choices to minimize runtime | L2: auto-tuning for kernel optimization |
| Backends | How the same IR generates CUDA, OpenCL, or LLVM code | L2: multi-target compilation |

**Projects:**
1. Trace a matmul through tinygrad: `Tensor` → lazy buffer → scheduled ops → generated CUDA kernel. Document every step.
2. Run a small model with `DEBUG=4` to see the generated kernels. Count the number of kernel launches.
3. Run with `BEAM=3` and compare kernel count and latency vs `BEAM=0`.
4. (Advanced) Add a minimal logging backend that prints each kernel launch — verify which ops fuse.

**Deep dive:** The full tinygrad learning path (11 parts, 7 projects) is in [Phase 5E — Autonomous Vehicles / tinygrad](../../Phase%205%20-%20Advanced%20Topics%20and%20Specialization/Track%20E%20-%20Autonomous%20Vehicles/3.%20tinygrad%20for%20Inference/Guide.md).

---

## How Frameworks Connect to the Rest of the Roadmap

| Framework skill | Where it leads |
|----------------|---------------|
| micrograd autograd | Phase 4C: understand what compiler must differentiate |
| PyTorch model export (ONNX) | Phase 4C §1: graph IR as compiler input |
| PyTorch quantization | Phase 4C Part 2 §4: quantization passes |
| `torch.compile` (Inductor) | Phase 4C §5: production ML compiler pipeline |
| tinygrad IR and scheduler | Phase 4C §5: BEAM search, fusion strategies |
| tinygrad backends | Phase 4C §7: custom backend for your accelerator |
| PyTorch profiling | Phase 4C Part 2 §1: graph/operator optimization |

---

## Resources

| Resource | What it covers |
|----------|---------------|
| [Andrej Karpathy — micrograd video](https://www.youtube.com/watch?v=VMj-3S1tku0) | Build autograd from scratch (2 hours) |
| [PyTorch Tutorials](https://pytorch.org/tutorials/) | Official beginner → advanced tutorials |
| [tinygrad GitHub](https://github.com/tinygrad/tinygrad) | Source code — read it |
| [tinygrad Discord](https://discord.gg/tinygrad) | Community, contributions, help |
| *Deep Learning with PyTorch* (Stevens, Antiga, Viehmann) | Comprehensive PyTorch book |

---

## Next

→ [**Module 3 — Computer Vision**](../Track%20A%20-%20Hardware%20and%20Edge%20AI/3.%20Computer%20Vision/Guide.md) — the perception workloads that drive edge AI and autonomous systems.
