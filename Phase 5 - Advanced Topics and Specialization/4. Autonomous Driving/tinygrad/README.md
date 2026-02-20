# Tinygrad: Hackable Learning Path

Part of the [AI Hardware Engineer Roadmap](../../../README.md). Learn tinygrad at the **hackable level**—understand the compiler, IR, and how to extend it. Used by [Openpilot](https://github.com/commaai/openpilot) (comma.ai ADAS) for neural network inference.

## Hackable Learning Path

| Step | Resource | What You Learn |
|------|----------|----------------|
| 1 | **hands-on-example.py** | Run first. See lazy eval, fusion, UOp in action |
| 2 | **hacking-tinygrad.md** | Inspect graph, IR, schedule, add backends |
| 3 | **tinygrad-notes.md** | Core philosophy: 3 op types, no CONV/MATMUL primitives |
| 4 | **ops/** | Elementwise, Reduce, Movement—how everything composes |
| 5 | **tinygrad-source/** | Read the code. It's small and Python. |

## Quick Start

```bash
pip install tinygrad numpy
python hands-on-example.py
```

Uses `../tinygrad-source/` (git submodule). Clone with `git clone --recurse-submodules` or run `git submodule update --init` after clone.

## What Makes It Hackable

- **UOp** — Graph nodes. `tensor.uop`, `tensor.schedule()` expose the execution plan
- **All in Python** — No hidden C++/CUDA. Compiler, IR, codegen visible
- **3 primitives** — ElementwiseOps, ReduceOps, MovementOps. CONV/MATMUL are composed
- **Lazy + fusion** — Build graph → optimize → compile → run. See each stage with `DEBUG=4`

## Structure

```
tinygrad/
├── README.md              # This file
├── hands-on-example.py    # Run this first
├── hacking-tinygrad.md    # UOp, schedule, IR, backends
├── tinygrad-notes.md      # Philosophy and overview
└── ops/                   # Operation deep dives
    ├── 01-elementwise-ops.md
    ├── 02-reduce-ops.md
    ├── 03-movement-ops.md
    ├── complete-reference.md
    └── elementwise/       # Unary, binary, ternary
```

## Resources

- [tinygrad GitHub](https://github.com/tinygrad/tinygrad)
- [Discord](https://discord.gg/tinygrad)
