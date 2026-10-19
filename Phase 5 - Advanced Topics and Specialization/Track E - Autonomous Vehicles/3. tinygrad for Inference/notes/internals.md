# Read the actual tinygrad 0.14.0 pipeline

**Before:** complete the [compiler project](../projects/README.md#3-inspect-compilation-separately). **Time:** 60–90 minutes. **Artifact:** one source-supported graph-to-execution trace. Paths below refer to [tag v0.14.0](https://github.com/tinygrad/tinygrad/tree/v0.14.0), not a moving checkout.

## 1. Locate the installed source

From the environment in the main guide:

```bash
/tmp/course-tinygrad-venv/bin/python - <<'PY'
from importlib.metadata import version
from pathlib import Path
import tinygrad
print(version('tinygrad'))
print(Path(tinygrad.__file__).parent)
PY
```

The printed directory is the actual code the project imports. Record it and the wheel hash before interpreting a source diagram. The optional project runner rejects versions other than 0.14.0.

## 2. Follow one function through the layers

| Source location | Role in the pinned package |
|---|---|
| `tinygrad/tensor.py` | Tensor data/API, graph integration, realization and backward entry |
| `tinygrad/uop/ops.py` | UOp representation and operation vocabulary |
| `tinygrad/mixin/gradient.py` | Reverse-mode gradient rules and graph traversal |
| `tinygrad/schedule/` | Group graph work into executable kernels and memory decisions |
| `tinygrad/codegen/` and `codegen/opt/` | Lowering and kernel optimization |
| `tinygrad/renderer/` | Target-specific program representation |
| `tinygrad/engine/realize.py` | Compile/execute the scheduled work |
| `tinygrad/runtime/ops_cpu.py` | Tested CPU runtime integration |
| `tinygrad/nn/optim.py` | Optimizer state/update and training-context requirement |

Use search within these files rather than assuming an old private function still exists. `LazyBuffer`, `ShapeTracker`, `ops_clang.py`, `ops_hsa.py` and a fabricated `Device._devices` registry are not this source map. [Official layout overview](https://docs.tinygrad.org/developer/layout/) gives orientation; the tag/wheel resolves discrepancies.

## 3. Do not consume a schedule just to count it

In the pinned Tensor implementation, `linear_with_vars` transforms a sink, applies a tensor mapping and creates executable linear work; `schedule_linear()` exposes that schedule when no symbolic variables remain. This is an internal execution path with state effects, not a universally read-only graph query.

During review, creating a schedule and then asking the original unexecuted tensor for values could expose storage before that scheduled work had run. The course therefore counts **actual executed kernels**, and inspects a separate pre-execution UOp root. If you study the internal schedule API, trace its matching execution function and lifetime rules before reusing those tensors.

**Worked distinction:** three Python operations need not mean three kernels; the generated program can fuse them. Conversely, one displayed high-level UOp tree can require several scheduled operations. Count at the layer that answers your question.

## 4. Trace autodiff and training separately

For `y=sum(x*x)`, follow the multiply/reduce gradient rules and accumulation when `x` appears twice. The expected result is `2x`. Then follow `Optimizer.step → schedule_step` to see why this version requires `Context(TRAINING=1)` and how parameter assignments become realized updates.

The current constructor does not accept the older `requires_grad=True` argument. The package's parameter flag and gradient traversal differ from that older API. Do not infer API compatibility merely because the expression resembles PyTorch.

**Exercise:** explain the gradient of broadcasting a vector into two identical rows before summing. Each original element is used twice, so its gradient is 2. Find where the gradient implementation reduces broadcast axes back to the source shape.

## 5. Separate renderer output from runtime responsibility

A renderer describes executable work; the runtime must allocate/address storage, compile/load a compatible program, pass its ABI arguments, execute it with correct ordering, and keep buffers alive. A correct C string plus an incorrect argument declaration can still corrupt a result.

The [standalone C runtime](../projects/07_custom_backend.py) isolates those obligations with contiguous FP32 buffers and a declared `size_t` count. It is not registered with tinygrad. Extending it into a framework backend requires the actual pinned device/allocator/compiler/runtime interfaces plus dtype, layout, synchronization and error-path tests.

## Completion

Draw the path for the affine/ReLU example with exact source filenames, include a generated-code excerpt from your own debug run, and identify the input/output pointers and loop bounds. Add the changing-input JIT result and one limit of your trace. Keep instrumented compilation evidence separate from performance measurements.
