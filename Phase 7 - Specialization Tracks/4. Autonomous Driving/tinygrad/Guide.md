# Tinygrad Learning Guide

**A structured, hands-on path from first tensor to custom backend**

> Tinygrad is a minimal deep learning framework where **the entire compiler and IR are visible and hackable in Python**. It is the ideal codebase for understanding what happens between `loss.backward()` and the GPU kernel that actually runs.

**Time estimate:** 3–6 months for Parts 1–6 · Ongoing for Part 7

---

## Prerequisites

- Python proficiency (functions, classes, decorators, generators)
- Basic linear algebra (matrix multiply, dot product, broadcasting)
- Familiarity with neural network training (forward pass, loss, backprop, optimizer)
- Optional but helpful: CUDA/OpenCL basics for backend sections

---

## Part 1: Why Tinygrad?

### The Problem with PyTorch for Learning

PyTorch's autograd, JIT, and kernel dispatch live in thousands of lines of C++/CUDA. You can't read them. You can only observe their effects. Tinygrad solves this:

- **The entire framework is ~5,000 lines of Python**
- Every optimization, IR transformation, and kernel is readable
- You can set `DEBUG=4` and watch every stage of compilation

### Tinygrad's Position in the Ecosystem

```
micrograd  →  tinygrad  →  PyTorch
(no GPU)      (hackable)   (production)
```

- **micrograd** (Karpathy): pure autograd, no tensors, no GPU — great for understanding backprop
- **tinygrad**: lazy tensors, real GPU backends, hackable compiler — great for understanding frameworks
- **PyTorch**: full production framework, opaque internals — great for building things

### Why It Matters for AI Hardware Engineers

Tinygrad is the software interface between ML workloads and custom hardware. Understanding it lets you:
- Know exactly what operations your accelerator must support
- Design hardware for the actual compute patterns (matmul, reduction, elementwise)
- Write compiler backends that target your custom chip
- Understand why Openpilot uses tinygrad for production ADAS inference

---

## Part 2: Setup and First Steps

### Installation

```bash
# Option 1: pip (stable)
pip install tinygrad

# Option 2: from source (recommended for learning — you can read/modify it)
git clone https://github.com/tinygrad/tinygrad
cd tinygrad
pip install -e ".[dev]"   # editable install so changes take effect immediately
```

### Verifying Your Setup

```python
from tinygrad import Tensor, Device
print(Device.DEFAULT)           # shows your default backend (CUDA, METAL, CLANG, etc.)
print(Tensor([1,2,3]).numpy())  # [1. 2. 3.]
```

### Debug Environment Variables

These are your most important tools for understanding what tinygrad does:

| Variable | Value | What You See |
|----------|-------|-------------|
| `DEBUG` | `1` | Kernel count and timing |
| `DEBUG` | `2` | Kernel names and shapes |
| `DEBUG` | `3` | Generated kernel source code |
| `DEBUG` | `4` | Full UOp IR at every optimization stage |
| `VIZ` | `1` | Opens a browser graph visualizer for UOps |
| `BEAM` | `2` | Enable BEAM search kernel optimization |
| `NOOPT` | `1` | Disable optimizations (see unoptimized IR) |

---

## Part 3: Tensor API

### Creating Tensors

```python
from tinygrad import Tensor
import numpy as np

# From Python lists
t = Tensor([1, 2, 3, 4])

# From numpy
t = Tensor(np.array([[1.0, 2.0], [3.0, 4.0]]))

# Factory methods
t = Tensor.zeros(3, 4)
t = Tensor.ones(3, 4)
t = Tensor.randn(3, 4)         # normal distribution
t = Tensor.rand(3, 4)          # uniform [0, 1)
t = Tensor.arange(0, 10, 2)    # [0, 2, 4, 6, 8]
t = Tensor.eye(4)              # identity matrix
```

### Operations (PyTorch-Compatible API)

```python
a = Tensor.randn(4, 4)
b = Tensor.randn(4, 4)

# Elementwise
c = a + b
c = a * b
c = a.relu()
c = a.exp()
c = a.log()
c = a.sqrt()

# Reduction
s = a.sum()
s = a.sum(axis=0)              # sum along rows → shape (4,)
m = a.max(axis=1, keepdim=True)

# Matrix operations
c = a @ b                      # matmul → shape (4,4)
c = a.T                        # transpose

# Shape manipulation
c = a.reshape(2, 8)
c = a.permute(1, 0)            # like numpy.transpose
c = a.expand(2, 4, 4)         # broadcast-expand (zero-copy)
c = a.unsqueeze(0)             # add dim at position 0
c = a.squeeze(0)               # remove dim of size 1
```

### Getting Values Back (Materializing)

```python
# .realize() executes the lazy graph and returns the same tensor
t = Tensor.randn(3, 3)
t = t.realize()

# .numpy() realizes AND copies to numpy
arr = t.numpy()    # triggers .realize() if not already done

# .item() for scalar tensors
loss_val = loss.item()
```

---

## Part 4: Lazy Evaluation — The Core Concept

This is the most important concept to internalize. **Nothing runs until you call `.realize()` or `.numpy()`.**

### What "Lazy" Means

```python
import os
os.environ['DEBUG'] = '2'

a = Tensor.randn(4, 4)   # no computation
b = a + 1                 # no computation — records ADD in graph
c = b * 2                 # no computation — records MUL in graph
d = c.relu()              # no computation — records RELU in graph

# Only now does tinygrad compile all 3 ops into ONE kernel
d.realize()               # prints: "1 kernels, X.Xms"
```

Tinygrad fused `a+1`, `*2`, and `relu()` into a single kernel — no intermediate buffers needed. This is **op fusion**, a key optimization.

### The UOp: Tinygrad's IR Node

Every tensor has a `.uop` attribute — a node in the computation graph:

```python
from tinygrad import Tensor

x = Tensor([1.0, 2.0, 3.0])
y = x + 1

print(type(y.uop))    # <class 'tinygrad.uop.UOp'>
print(y.uop.op)       # the operation type
print(y.uop.src)      # input UOps (the graph edges)
```

UOps form a DAG (Directed Acyclic Graph). The compiler walks this graph to:
1. Fuse operations
2. Apply algebraic rewrites (e.g., `x * 1 → x`)
3. Generate kernel code

---

## Part 5: The Three Operation Types

Everything in tinygrad decomposes to three kinds of primitive operations. **No conv2d primitive. No matmul primitive.** They are built from these three:

### 1. ElementwiseOps

Operate on each element independently — trivially parallelizable:

| Category | Operations |
|----------|-----------|
| UnaryOps | `SQRT, LOG2, EXP2, SIN, NEG, RECIP, CAST` |
| BinaryOps | `ADD, MUL, SUB, DIV, MAX, MOD, CMPLT` |
| TernaryOps | `WHERE (if/else), MULACC` |

```python
# All of these lower to ElementwiseOps:
x.relu()          # WHERE(x > 0, x, 0)  →  TernaryOp(WHERE)
x.sigmoid()       # 1 / (1 + exp(-x))   →  chain of UnaryOps + BinaryOps
x.exp()           # EXP2(x * log2(e))   →  BinaryOp(MUL) + UnaryOp(EXP2)
```

### 2. ReduceOps

Collapse a dimension — require communication across elements:

```python
x = Tensor.randn(1024, 1024)

# These are ReduceOps:
x.sum(axis=0)     # SUM reduce along axis 0
x.max(axis=1)     # MAX reduce along axis 1
x.mean()          # SUM / size  →  ReduceOp + ElementwiseOp
```

ReduceOps are the hard part of GPU programming — they require careful parallel reduction patterns.

### 3. MovementOps (via ShapeTracker)

Reshape, permute, expand — **zero-copy** because they only change how indices map to memory:

```python
x = Tensor.randn(4, 8, 16)

# These don't copy data — they update the ShapeTracker:
y = x.reshape(32, 16)       # just relabels dimensions
y = x.permute(2, 0, 1)      # reorders access pattern
y = x.expand(10, 4, 8, 16)  # broadcasts (adds a new dimension)
y = x[1:3, :, ::2]          # slice + stride

# Data only moves when a materialization (realize/numpy) is needed
```

**ShapeTracker** stores the strides and offsets that define how a logical index maps to a physical buffer index. This is how zero-copy views work.

---

## Part 6: Autograd

### How tinygrad Implements Backprop

Tinygrad uses **reverse-mode automatic differentiation** (backprop). Every op that needs a gradient has a corresponding backward function.

```python
from tinygrad import Tensor

# Gradient tracking is enabled with requires_grad=True
# (or automatically when a leaf tensor needs grad)
x = Tensor([2.0, 3.0], requires_grad=True)
y = (x * x).sum()          # y = sum(x^2)

y.backward()               # computes dy/dx = 2x

print(x.grad.numpy())      # [4. 6.]  — correct: d(sum(x^2))/dx = 2x
```

### Training Loop Pattern

```python
from tinygrad import Tensor
from tinygrad.nn.optim import Adam, SGD

# Model (simple 2-layer MLP)
class MLP:
    def __init__(self):
        self.l1 = Tensor.randn(784, 128) * 0.01
        self.l2 = Tensor.randn(128, 10) * 0.01

    def __call__(self, x):
        return x.linear(self.l1).relu().linear(self.l2)

model = MLP()
optim = Adam([model.l1, model.l2], lr=0.001)

for step in range(1000):
    x = Tensor.randn(32, 784)          # fake batch
    y_target = Tensor.zeros(32, 10)    # fake labels

    optim.zero_grad()
    y_pred = model(x)
    loss = (y_pred - y_target).pow(2).mean()   # MSE loss
    loss.backward()
    optim.step()

    if step % 100 == 0:
        print(f"step {step}, loss={loss.item():.4f}")
```

### tinygrad's nn Module

```python
from tinygrad.nn import Linear, BatchNorm, Conv2d
from tinygrad.nn.optim import Adam, SGD, AdamW
from tinygrad.nn.state import get_parameters, get_state_dict, load_state_dict, safe_save, safe_load

# Linear layer
layer = Linear(128, 64, bias=True)

# Conv2d
conv = Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1)

# Get all parameters of a model
params = get_parameters(model)

# Save/load weights
state = get_state_dict(model)
safe_save(state, "model.safetensors")
state = safe_load("model.safetensors")
load_state_dict(model, state)
```

---

## Part 7: The Compiler Pipeline

This is where tinygrad becomes uniquely educational. Set `DEBUG=4` and watch every stage.

### Stage 1: Schedule Generation

After you call `.realize()`, tinygrad first creates a **schedule** — a list of kernels that need to run:

```python
from tinygrad import Tensor

x = Tensor.randn(4, 4)
y = Tensor.randn(4, 4)
z = (x @ y).relu().sum(axis=0)

# Get the schedule without executing
sched = z.schedule()
print(f"{len(sched)} kernels")
for item in sched:
    print(item.ast)    # the abstract syntax tree for each kernel
```

Fused ops appear as a single schedule item. Multiple reductions or ops across buffer boundaries become separate items.

### Stage 2: UOp Lowering and Optimization

Each schedule item's AST is lowered to a **UOp tree** and then optimized through a series of **pattern-matching rewrite rules**:

```
AST  →  UOp tree  →  rewrite rules  →  optimized UOp  →  codegen
```

Key rewrite rules include:
- **Constant folding**: `x + 0 → x`, `x * 1 → x`
- **Algebraic identities**: `log(exp(x)) → x`
- **Loop fusion**: merge adjacent loops with compatible access patterns
- **Linearization**: convert the tree into a flat list of operations for codegen

### Stage 3: Kernel Optimization (BEAM Search)

Set `BEAM=2` to enable BEAM search — tinygrad tries multiple loop orderings and tile sizes, benchmarks them, and picks the fastest:

```bash
BEAM=2 python3 my_script.py    # slower first run, fast after caching
```

BEAM search explores:
- **Loop ordering**: which dimension to parallelize, which to tile
- **Work-group sizes**: how many threads per block
- **Unrolling and vectorization**: SIMD width

### Stage 4: Code Generation

The optimized UOp tree is converted to target-specific code:

```python
import os
os.environ['DEBUG'] = '3'
os.environ['CLANG'] = '1'    # use CPU backend to see readable C

from tinygrad import Tensor
x = Tensor.randn(4, 4)
y = Tensor.randn(4, 4)
(x @ y).realize()
# Prints: actual C function that implements the matmul
```

Backends: `CLANG` (C), `CUDA` (PTX/SASS), `METAL` (MSL), `AMD` (HIP), `OpenCL`

---

## Part 8: Backends

### Existing Backend Structure

In the tinygrad source, backends live in `tinygrad/runtime/`:

```
tinygrad/runtime/
  ops_gpu.py    # OpenCL backend
  ops_cuda.py   # NVIDIA CUDA backend
  ops_metal.py  # Apple Metal backend
  ops_clang.py  # CPU/Clang backend (simplest — read this first)
  ops_hsa.py    # AMD HIP backend
```

Each backend implements three things:
1. **Allocator**: allocate/free device memory, copy to/from host
2. **Compiler**: take generated source code, compile to binary (PTX, SPIR-V, etc.)
3. **Runtime**: load a compiled program and execute it with given buffers

### Reading the Clang Backend

`ops_clang.py` is the simplest — it generates C, compiles with clang, and runs as a shared library. Start here:

```python
# Simplified structure of ops_clang.py:
class ClangAllocator(Allocator):
    def _alloc(self, size): return (ctypes.c_float * size)()
    def _free(self, buf): del buf
    def copyin(self, dst, src): ctypes.memmove(dst, src, src.nbytes)
    def copyout(self, dst, src): ctypes.memmove(dst, src, ctypes.sizeof(src))

class ClangCompiler(Compiler):
    def compile(self, src: str) -> bytes:
        # Write C to temp file, compile with clang, return .so bytes
        ...

class ClangRuntime(Runner):
    def __call__(self, *bufs, global_size, local_size):
        # Call the compiled C function with the given buffers
        ...
```

### Adding a Custom Backend (Skeleton)

```python
# my_backend.py — minimal custom backend skeleton

from tinygrad.device import Compiled, Allocator, Compiler, Runner
from tinygrad.renderer.cstyle import CStyleLanguage

class MyAllocator(Allocator):
    def _alloc(self, size: int, options):
        # Allocate `size` bytes on your device
        # Return a handle (pointer, buffer object, etc.)
        ...

    def _free(self, buf, options):
        # Free the buffer
        ...

    def copyin(self, dst, src: memoryview):
        # Copy from host (numpy array) to device buffer
        ...

    def copyout(self, dst: memoryview, src):
        # Copy from device buffer to host (numpy array)
        ...

class MyCompiler(Compiler):
    def compile(self, src: str) -> bytes:
        # src is the generated C/kernel source as a string
        # Compile it to binary and return bytes
        ...

class MyRunner(Runner):
    def __init__(self, name: str, lib: bytes):
        # Load the compiled binary
        ...

    def __call__(self, *bufs, global_size, local_size, wait=False):
        # Execute the kernel with given buffers and launch dimensions
        ...

class MyDevice(Compiled):
    def __init__(self, device: str):
        super().__init__(
            device,
            MyAllocator(),
            CStyleLanguage(),   # reuse the C code generator
            MyCompiler(),
            MyRunner,
        )

# Register the device:
# from tinygrad.device import Device
# Device._devices["MYDEVICE"] = MyDevice
```

---

## Part 9: Projects

Work through these in order. Each builds on the previous.

### Project 1: Tensor Basics and Lazy Evaluation

**Goal:** Understand that tinygrad doesn't compute until `.realize()`, and see op fusion in action.

**File:** `projects/01_tensor_basics.py`

**Tasks:**
1. Create tensors with all factory methods. Check shapes and dtypes.
2. Set `DEBUG=1`. Run `(a + b).relu().sum()` and count how many kernels run (should be 1 — fused).
3. Set `DEBUG=3`. Read the generated kernel. Find the ADD, RELU, and SUM operations in the C code.
4. Break fusion: call `.realize()` after each op. Count kernels now (should be 3). Compare speed.
5. Use `.schedule()` to print the schedule before and after a `.realize()` call.

**Key insight:** Op fusion is free performance — tinygrad does it automatically when you let the graph grow before realizing.

---

### Project 2: The Three Op Types

**Goal:** Understand ElementwiseOps, ReduceOps, and MovementOps by inspecting what tinygrad generates for each.

**File:** `projects/02_op_types.py`

**Tasks:**
1. ElementwiseOps: Run `x.relu()`, `x * 2`, `x.exp()` with `DEBUG=3`. Find the corresponding operations in the C output.
2. ReduceOps: Run `x.sum()`, `x.max(axis=0)` with `DEBUG=3`. Observe the loop structure in the reduction kernel.
3. MovementOps: Run `x.reshape(...)`, `x.permute(...)`, `x.expand(...)`. Use `DEBUG=2` — notice that movement ops alone produce **0 kernels**. They are zero-copy.
4. Mixed: Build `(x.permute(1,0) @ y.reshape(4,4)).sum(axis=1)`. Count kernels with `DEBUG=1`. Try to predict how many before running.
5. Implement matmul manually without `@`: use `expand` + `*` + `sum`. Verify it matches `@`.

**Key insight:** Movement ops are free because ShapeTracker only changes index math. Fusing movement ops into downstream compute ops is how tinygrad avoids unnecessary memory copies.

---

### Project 3: Autograd from Scratch

**Goal:** Understand how tinygrad implements reverse-mode autodiff.

**File:** `projects/03_autograd.py`

**Tasks:**
1. Manually verify gradients:
   - `y = x.pow(2).sum()` → `dy/dx = 2x` (verify with `.grad`)
   - `y = (x * w).sum()` → `dy/dw = x` (verify)
   - `y = x.sigmoid()` → `dy/dx = sigmoid(x) * (1 - sigmoid(x))` (verify)
2. Build a 2-layer MLP and train it to overfit a 10-sample XOR dataset. Verify loss reaches near zero.
3. Read the tinygrad source: `tinygrad/tensor.py`, search for `def _broadcasted` and `class Function`. Read 3 backward functions (e.g., `Mul`, `Add`, `Sum`). Write a comment explaining each.
4. Implement a custom loss function: Huber loss. Verify its gradient numerically using finite differences `(f(x+ε) - f(x-ε)) / 2ε`.

**Key insight:** Autograd is just a chain of `backward()` functions stored in the graph. Each op records how to propagate the gradient back through it.

---

### Project 4: Training MNIST

**Goal:** Train a real model end-to-end. Understand the full training loop, data loading, and evaluation.

**File:** `projects/04_mnist.py`

**Tasks:**
1. Download MNIST: tinygrad provides `from tinygrad.nn.datasets import mnist` (or download manually).
2. Build a CNN with `Conv2d`, `BatchNorm`, `relu`, and `Linear`.
3. Train for 5 epochs with Adam. Target: >98% test accuracy.
4. Profile the training loop with `DEBUG=1`. How many kernels per step? Which ones take the most time?
5. Enable `BEAM=2`. Does it speed up training? By how much?
6. Save the model with `safe_save` and reload it. Verify accuracy is identical after reload.

**Reference:** `tinygrad/examples/mnist.py` in the tinygrad source — read it before writing your own.

---

### Project 5: Inspecting the Compiler

**Goal:** Understand the full pipeline from Python tensor ops to GPU kernel.

**File:** `projects/05_compiler_pipeline.py`

**Tasks:**
1. **Schedule inspection:** Build a computation graph with 5+ ops. Call `.schedule()`. Print the AST for each kernel item. Explain why some ops are fused and others aren't.
2. **UOp tree:** Set `VIZ=1` and run a matmul. Navigate the browser graph visualizer. Find the elementwise, reduce, and loop nodes.
3. **IR stages with DEBUG=4:** Run a small matmul with `DEBUG=4`. Identify and describe:
   - The initial UOp IR (before optimization)
   - The IR after constant folding
   - The IR after loop analysis
   - The final linearized IR
4. **Algebraic rewrites:** Set `NOOPT=1` and run `x * 1 + 0`. Note the kernel. Remove `NOOPT=1` and run again. Observe the optimized kernel.
5. **BEAM search:** Run a 1024×1024 matmul with `BEAM=0` and `BEAM=4`. Compare runtime. What tile size did BEAM choose?

---

### Project 6: Custom Operations and Backend Extensions

**Goal:** Extend tinygrad with a custom op and understand the backend interface.

**File:** `projects/06_custom_ops.py`

**Tasks:**
1. **Compose custom ops:** Implement these from primitives only (no `torch`-like built-ins):
   - `gelu(x)`: `x * 0.5 * (1 + (x / sqrt(2)).erf())`  — use tinygrad's `erf` or approximate it
   - `rms_norm(x)`: `x / sqrt((x*x).mean() + 1e-6)`
   - `swiglu(x, gate)`: `x * gate.sigmoid()`
2. Verify each custom op matches a reference PyTorch implementation numerically.
3. Write gradients test for `rms_norm`: compute gradient via autograd, verify with finite differences.
4. **Inspect what they compile to:** Use `DEBUG=3` to see the kernel for each. Count operations.
5. **Fuse vs. unfused:** Implement `gelu` step-by-step with `.realize()` between each step vs. all at once. Compare kernel count and runtime.

---

### Project 7: Implement a Custom Backend

**Goal:** Build a functional custom backend that runs tinygrad ops.

**File:** `projects/07_custom_backend/`

This is the capstone project. You'll implement a backend that targets the **CPU via ctypes** (simpler than CUDA, but teaches the full interface).

**Part A — Allocator:**
- Implement `_alloc(size)` using `ctypes.create_string_buffer`
- Implement `copyin` and `copyout` using `ctypes.memmove`
- Test: allocate a buffer, copy a numpy array in, copy it back out. Verify round-trip.

**Part B — Compiler:**
- Implement `compile(src: str) → bytes` by calling `clang` as a subprocess on the generated C source
- Return the compiled `.so` as bytes
- Test: compile a trivial C kernel `void kernel(float* a) { a[0] = 42.0f; }` and verify it builds

**Part C — Runner:**
- Load the compiled `.so` bytes into memory with `ctypes.CDLL`
- Implement `__call__(*bufs, global_size, local_size)` to invoke the compiled function
- Handle the global_size loop in Python (single-threaded — focus on correctness)

**Part D — Integration:**
- Assemble `MyAllocator`, `MyCompiler`, and `MyRunner` into a `MyDevice(Compiled)` class
- Register it: `Device._devices["MY"] = MyDevice`
- Run `Tensor([1,2,3], device="MY") + Tensor([4,5,6], device="MY")` and get `[5,7,9]`

**Part E — Benchmark:**
- Run a 256×256 matmul on `MY`, `CLANG`, and (if available) `CUDA`
- Compare GFLOPS. Document the overhead of your Python runtime vs. the C clang runtime.

---

## Part 10: Reading the Source

After completing the projects, read these files in the tinygrad source in order:

| File | What It Teaches |
|------|----------------|
| `tinygrad/tensor.py` | Tensor API and autograd — the user-facing layer |
| `tinygrad/engine/lazy.py` | LazyBuffer — how lazy evaluation is implemented |
| `tinygrad/engine/schedule.py` | How the execution schedule is built from the graph |
| `tinygrad/codegen/uops.py` | UOp IR definition and all op types |
| `tinygrad/codegen/lowerer.py` | Lowering schedule AST to UOp IR |
| `tinygrad/codegen/linearizer.py` | Linearization — UOp tree to flat kernel instructions |
| `tinygrad/renderer/cstyle.py` | C/CUDA/Metal code generation from linear UOps |
| `tinygrad/runtime/ops_clang.py` | Simplest backend — Allocator, Compiler, Runner |
| `tinygrad/runtime/ops_cuda.py` | CUDA backend — compare with clang backend |
| `tinygrad/nn/__init__.py` | Standard layers (Linear, Conv2d, BatchNorm) |

**Study method:** Pick a function, add `print()` statements, run an example with `DEBUG=4`, and trace the execution path from `Tensor.add()` all the way to the compiled C string.

---

## Part 11: Contributing to Tinygrad

### Where to Start

1. **Read `CONTRIBUTING.md`** in the tinygrad source
2. **Run the test suite:** `python -m pytest test/ -x` — understand what's tested
3. **Find a "good first issue"** on the GitHub issue tracker

### Types of Contributions

- **New ops or nn layers:** Implement a missing activation, loss, or layer
- **Backend improvements:** Optimize kernel generation for a specific GPU architecture
- **New backend:** Add support for a new device (e.g., RISC-V simulator, custom FPGA)
- **Bug fixes:** Fix a correctness issue with a specific op/dtype combination
- **Documentation:** Write examples or clarify existing docs

### Testing Your Changes

```bash
# Run tests for a specific file
python -m pytest test/test_tensor.py -x -v

# Run tests for a specific test
python -m pytest test/test_ops.py::TestOps::test_relu -v

# Test with a specific backend
DEVICE=CLANG python -m pytest test/ -x

# Run the full CI suite (slow)
python -m pytest test/ -x --timeout=300
```

### The tinygrad Standard

The core team is strict about code quality. When contributing:
- Keep changes minimal — tinygrad values simplicity above all
- No new dependencies
- All tests must pass
- New features need tests
- Code style: no type annotations in function bodies, minimal comments

---

## Resources

| Resource | What It's For |
|----------|--------------|
| [tinygrad GitHub](https://github.com/tinygrad/tinygrad) | Source, issues, discussions |
| [tinygrad Docs](https://tinygrad.github.io/tinygrad/) | API reference and quickstart |
| [tinygrad Discord](https://discord.gg/tinygrad) | Community, Q&A, contributor help |
| [tinygrad-notes](https://mesozoic-egg.github.io/tinygrad-notes/) | Community deep-dives on internals |
| [George Hotz streams (Twitch/YouTube)](https://www.youtube.com/@georgehotzarchive) | Live coding tinygrad — watch the compiler evolve |
| [abstractions.py](https://github.com/tinygrad/tinygrad/blob/master/docs/abstractions.py) | Annotated walkthrough of tinygrad's architecture |
| `hacking-tinygrad.md` (this folder) | Code snippets for inspecting internals |

---

*See the `projects/` folder for runnable Python scripts for each project above.*
