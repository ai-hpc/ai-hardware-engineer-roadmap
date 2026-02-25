# Tinygrad Projects

Hands-on projects for the [tinygrad learning guide](../Guide.md). Work through them in order — each builds on the previous.

## Setup

```bash
# Install tinygrad (editable from source — recommended)
git clone https://github.com/tinygrad/tinygrad
cd tinygrad
pip install -e ".[dev]"
cd -

# Or from pip (stable)
pip install tinygrad

# Verify
python3 -c "from tinygrad import Tensor; print(Tensor([1,2,3]).numpy())"
```

## Debug Environment Variables

These unlock tinygrad's internals. Use them constantly while working through the projects.

| Command | Purpose |
|---------|---------|
| `DEBUG=1 python3 script.py` | Show kernel count and timing per `.realize()` |
| `DEBUG=2 python3 script.py` | Show kernel names and output shapes |
| `DEBUG=3 python3 script.py` | Print generated kernel source code (C/CUDA/MSL) |
| `DEBUG=4 python3 script.py` | Dump full UOp IR at every optimization stage |
| `VIZ=1 python3 script.py` | Open browser-based computation graph visualizer |
| `BEAM=2 python3 script.py` | Enable BEAM search kernel auto-tuning |
| `NOOPT=1 python3 script.py` | Disable all algebraic rewrites (baseline comparison) |
| `CLANG=1 python3 script.py` | Force CPU/Clang backend (readable C output) |

---

## Project 1: Tensor Basics and Lazy Evaluation

**File:** `01_tensor_basics.py`

What you learn:
- All tensor factory methods and their shapes/dtypes
- Why tinygrad doesn't compute until `.realize()` or `.numpy()`
- Op fusion — multiple Python ops compiled into a single GPU kernel
- How to inspect the execution schedule before running

```bash
python3 01_tensor_basics.py
DEBUG=1 python3 01_tensor_basics.py     # see that the fused path uses 1 kernel
DEBUG=3 python3 01_tensor_basics.py     # read the generated C for the fused kernel
```

**Key question to answer:** How many kernels does `(a + b).relu().sum()` produce? Why?

---

## Project 2: The Three Op Types

**File:** `02_op_types.py`

What you learn:
- ElementwiseOps: independent per-element — trivially fuse
- ReduceOps: dimension-collapsing — break fusion boundaries
- MovementOps: zero-copy via ShapeTracker
- How matmul decomposes to expand + multiply + sum

```bash
python3 02_op_types.py
DEBUG=1 python3 02_op_types.py
```

**Key insight:** Movement ops alone produce **zero kernels**. Verify this with `DEBUG=1`.

---

## Project 3: Autograd from Scratch

**File:** `03_autograd.py`

What you learn:
- How `.backward()` computes gradients via reverse-mode autodiff
- Verifying gradients analytically (sum of squares, sigmoid, etc.)
- Numerical gradient checking with finite differences
- Training a small MLP to overfit XOR

```bash
python3 03_autograd.py
```

**Key exercise:** Implement the numerical gradient checker yourself. Apply it to any custom op you write in Project 6.

---

## Project 4: Training MNIST

**File:** `04_mnist.py`

What you learn:
- Full training loop: data → forward → loss → backward → optimizer step
- tinygrad's `nn` module: `Conv2d`, `BatchNorm`, `Linear`
- `safe_save` / `safe_load` for model persistence
- Profiling with `DEBUG=1` and BEAM search

```bash
python3 04_mnist.py
BEAM=2 python3 04_mnist.py              # auto-tune kernels
DEBUG=1 python3 04_mnist.py             # profile kernel counts per step
```

**Target:** >98% test accuracy in 5 epochs.

---

## Project 5: Inspecting the Compiler

**File:** `05_compiler_pipeline.py`

What you learn:
- Schedule generation: which ops get fused into which kernels
- UOp tree structure: the IR nodes that represent computation
- Algebraic rewrites: `x*1→x`, `log(exp(x))→x`, constant folding
- Kernel count patterns for common ML ops (attention, layernorm, conv)
- BEAM search: how tinygrad auto-tunes kernel tiling

```bash
python3 05_compiler_pipeline.py
VIZ=1   python3 05_compiler_pipeline.py     # graph browser
DEBUG=4 python3 05_compiler_pipeline.py     # full IR dump
NOOPT=1 python3 05_compiler_pipeline.py     # see unoptimized kernels
BEAM=4  python3 05_compiler_pipeline.py     # enable kernel tuning
```

**Deep dive:** Run `DEBUG=4 CLANG=1` and read the IR stages for a softmax. Identify where the reduction and the normalization division are scheduled.

---

## Project 6: Custom Operations

**File:** `06_custom_ops.py`

What you learn:
- Building GELU, Swish, Mish, RMSNorm, LayerNorm from tinygrad primitives
- Scaled dot-product attention (with causal masking) from primitives
- Gradient verification for custom ops
- Fusion impact: fused vs unfused implementation of the same op

```bash
python3 06_custom_ops.py
DEBUG=1 python3 06_custom_ops.py    # count kernels for each op
DEBUG=3 python3 06_custom_ops.py    # see the generated kernels
```

**Challenge:** Implement Flash Attention's tiled softmax using tinygrad primitives. Verify it matches the naive implementation numerically.

---

## Project 7: Custom Backend

**File:** `07_custom_backend/backend.py`

What you learn:
- The three components every tinygrad backend implements: Allocator, Compiler, Runner
- How memory is managed at the device level
- How generated C source is compiled to a .so and executed
- How to benchmark your backend vs the built-in CLANG backend

```bash
python3 07_custom_backend/backend.py              # run tests
python3 07_custom_backend/backend.py benchmark    # benchmark vs CLANG
```

**Extension:** Add OpenMP parallelism to the element-wise loop in the generated C. Measure speedup on your CPU's core count.

---

## Learning Path

```
01 Basics → 02 Op Types → 03 Autograd → 04 MNIST → 05 Compiler → 06 Custom Ops → 07 Backend
  (3h)         (3h)           (4h)         (4h)        (6h)           (6h)            (12h)
```

Total: ~38 hours of focused work. Return to earlier projects with `DEBUG=4` and `VIZ=1` after completing later ones — you'll see much more.

---

## Further Reading

- [Guide.md](../Guide.md) — full learning guide with theory for each topic
- [tinygrad-notes.md](../tinygrad-notes.md) — overview and core philosophy
- [hacking-tinygrad.md](../hacking-tinygrad.md) — code snippets for internals
- [tinygrad source](https://github.com/tinygrad/tinygrad) — read `tinygrad/tensor.py` and `tinygrad/runtime/ops_clang.py`
- [abstractions.py](https://github.com/tinygrad/tinygrad/blob/master/docs/abstractions.py) — annotated architecture walkthrough
