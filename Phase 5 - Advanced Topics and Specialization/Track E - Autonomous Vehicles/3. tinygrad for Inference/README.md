# Tinygrad

Part of the [AI Hardware Engineer Roadmap](../../../README.md). Learn tinygrad at the **hackable level** — understand the compiler, IR, lazy evaluation, and how to extend it. Used by [Openpilot](https://github.com/commaai/openpilot) for neural network inference on Snapdragon.

## Quick Start

```bash
pip install tinygrad numpy
python3 projects/00_intro.py
```

For upstream source code, clone [tinygrad](https://github.com/tinygrad/tinygrad).

## Structure

```
tinygrad/
├── Guide.md          — Full learning guide (11 parts, ~730 lines)
├── notes/
│   ├── overview.md   — Philosophy, features, real-world usage
│   └── internals.md  — Hacking the compiler, IR, schedule, backends
├── ops/
│   ├── README.md     — Operations overview and composition reference
│   ├── elementwise.md — UnaryOps, BinaryOps, TernaryOps (16 primitives)
│   ├── reduce.md     — ReduceOps: SUM, MAX, derived ops
│   └── movement.md   — MovementOps: zero-copy via ShapeTracker
└── projects/
    ├── README.md     — Setup, debug env vars, project descriptions
    ├── 00_intro.py   — Run first: lazy eval, fusion, ShapeTracker demo
    ├── 01_tensor_basics.py    — Tensor API, lazy eval, op fusion
    ├── 02_op_types.py         — The 3 op types, manual matmul
    ├── 03_autograd.py         — Autograd, gradient checking, XOR MLP
    ├── 04_mnist.py            — Full CNN training (target: >98%)
    ├── 05_compiler_pipeline.py — Schedule, UOp IR, algebraic rewrites
    ├── 06_custom_ops.py       — GELU, LayerNorm, attention, loss fns
    └── 07_custom_backend.py   — Allocator + Compiler + Runner from scratch
```

## What Makes Tinygrad Hackable

| Concept | What It Means |
|---------|--------------|
| **Lazy evaluation** | Ops build a graph; nothing executes until `.realize()` |
| **3 op types** | ElementwiseOps, ReduceOps, MovementOps compose everything |
| **No CONV/MATMUL** | Built from primitives: `RESHAPE + EXPAND + MUL + SUM` |
| **UOp IR** | The computation graph is exposed in Python — read and modify it |
| **Kernel fusion** | Consecutive elementwise ops compile to a single GPU kernel |
| **ShapeTracker** | Reshape/permute/expand are zero-copy — metadata only |
| **All in Python** | Compiler, IR, codegen — no hidden C++/CUDA |

## Debug Environment Variables

| Command | What You See |
|---------|-------------|
| `DEBUG=1 python3 script.py` | Kernel count and timing per `.realize()` |
| `DEBUG=2 python3 script.py` | Kernel names and output shapes |
| `DEBUG=3 python3 script.py` | Generated kernel source (C/CUDA/MSL) |
| `DEBUG=4 python3 script.py` | Full UOp IR at every optimization stage |
| `VIZ=1 python3 script.py` | Browser-based computation graph visualizer |
| `BEAM=2 python3 script.py` | BEAM search kernel auto-tuning |
| `NOOPT=1 python3 script.py` | Disable algebraic rewrites (baseline) |
| `CLANG=1 python3 script.py` | Force CPU/Clang backend (readable C) |

## Learning Path

| Step | Resource | Time |
|------|----------|------|
| 0 | Run `projects/00_intro.py` | 15 min |
| 1 | Read `notes/overview.md` | 30 min |
| 2 | Read `ops/README.md` + op guides | 1–2h |
| 3 | Read `Guide.md` (full theory) | 2–3h |
| 4 | Work through `projects/01–07` | ~38h |
| 5 | Read `notes/internals.md` + tinygrad source | ongoing |

## Resources

- [tinygrad GitHub](https://github.com/tinygrad/tinygrad)
- [Discord](https://discord.gg/tinygrad)
- [abstractions.py](https://github.com/tinygrad/tinygrad/blob/master/docs/abstractions.py) — annotated architecture walkthrough
- [Community notes](https://mesozoic-egg.github.io/tinygrad-notes/) — JIT, ShapeTracker, BEAM, pattern matcher

## Practice and assessment

**Entry diagnostic (prerequisite check):** Trace a tensor expression into operations, scheduling, and generated execution. If this is unfamiliar, use the prerequisite material above or the [roadmap diagnostics](../../../Learning-Guide.md#diagnose-before-reading) before starting the lab.

1. **Baseline lab:** Follow one matmul or reduction through the pinned tinygrad source and compare with a reference.
2. **Guided variation:** Change shape/layout and inspect the resulting schedule and kernel choices.
3. **Independent transfer:** repeat the method on a new input or configuration and explain where your original assumptions stop applying.

**Evidence to submit:** Source commit, correctness, kernel count, compile/warmup time, execution latency, and memory. Keep raw results and reproduction steps; distinguish measured results from estimates or work not run.

**Self-check:** Which observation would disprove your explanation of the result? Identify a competing explanation and the smallest experiment that separates them. A passing answer names a concrete predicted difference, not only another tool to try.

**Exit criteria:** reproduce the baseline, demonstrate the negative/variation check, and meet the [shared completion rubric](../../../Learning-Guide.md#completion-rubric). Use the [lesson cycle](../../../Learning-Guide.md#a-repeatable-lesson-cycle) for retrieval practice and spaced review.
