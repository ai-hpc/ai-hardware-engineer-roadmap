# Module 3 — tinygrad: tensors to a checked inference result

**Before:** Python, matrix multiplication and derivatives; [AI foundations](../../../Phase%203%20-%20Artificial%20Intelligence/0.%20AI%20and%20Robot%20Learning%20Foundations/Guide.md) supplies the background. **Access:** Python 3.11+, Linux x86-64 and Clang for the tested CPU route. **Time:** 6–10 hours across the exercises. No GPU or vehicle is required.

This course uses **tinygrad 0.14.0**, released 24 August 2026, with a hash-pinned wheel. The code was executed on Python 3.12.3 and Clang 18.1.3 on 18 October 2026. It is separate from the tinygrad submodule pinned by openpilot. Current framework APIs are checked against the installed release and [v0.14.0 source](https://github.com/tinygrad/tinygrad/tree/v0.14.0); moving documentation can describe a different revision.

## 1. Begin with a function you can calculate

For `x=[-1,0,2]`, compute `max(2*x+1,0)=[0,1,5]`. A tensor program expresses the same operation on a buffer of a named dtype and shape. Creating the expression, compiling it and executing it are different events.

Run from the repository root:

```bash
python3 -m venv /tmp/course-tinygrad-venv
/tmp/course-tinygrad-venv/bin/python -m pip install --only-binary=:all: --require-hashes -r "Phase 5 - Advanced Topics and Specialization/Track E - Autonomous Vehicles/3. tinygrad for Inference/labs/requirements-tinygrad.txt"
clang --version
/tmp/course-tinygrad-venv/bin/python "Phase 5 - Advanced Topics and Specialization/Track E - Autonomous Vehicles/3. tinygrad for Inference/projects/00_intro.py"
```

The runner selects `DEV=CPU`, `BEAM=0` and a numeric debug setting before importing tinygrad. The wheel does not install Clang or a GPU driver. Use an available supported CPU toolchain; other host platforms were not validated here. [Package release](https://pypi.org/project/tinygrad/0.14.0/), [official runtimes](https://docs.tinygrad.org/runtime/).

## 2. Learn shape, broadcasting and reduction together

For `A` shaped `[2,3]` and `B` shaped `[3,2]`, matrix multiplication produces `[2,2]`. The manual expression expands the multiplication into `[2,3,2]`, then sums over the inner dimension:

```python
# Inside the pinned environment; all values are explicitly FP32.
from tinygrad import Tensor, dtypes
A = Tensor([[1,2,3],[-1,0,2]], device="CPU", dtype=dtypes.float32)
B = Tensor([[1,0],[2,1],[-1,3]], device="CPU", dtype=dtypes.float32)
manual = (A.reshape(2,3,1) * B.reshape(1,3,2)).sum(axis=1)
print(manual.tolist())  # [[2.0,11.0],[-3.0,6.0]]
```

Work entry `[0,1]`: `1×0 + 2×1 + 3×3 = 11`. Then compare with `A@B` and the independent scalar [reference](labs/reference.py). A non-square example exposes axis mistakes that a square matrix can hide.

Elementwise, reduction and movement operations are useful learning categories. They are not the entire compiler instruction set. Current UOps also represent buffers, device operations, calls, control and other semantics. The old `LazyBuffer → ShapeTracker → ops_clang.py` picture is not the layout of this pinned release. Use the [source walkthrough](notes/internals.md).

A reshape/transpose can often be represented through indexing without an immediate copy. That does not make every downstream access free, nor does it guarantee all layouts can reach a consumer without materialization. `.contiguous()` and transfer/output operations may require work. Compare values and inspect actual execution.

## 3. Verify gradients before training

In 0.14.0, use `Tensor([2.,3.])` and `(x*x).sum().backward()` to obtain `[4,6]`. The previous `requires_grad=True` constructor and `Tensor.training` examples are not this release's API. Optimizer steps require `with Context(TRAINING=1)`; the context restores its previous state on exit. [Tensor source](https://github.com/tinygrad/tinygrad/blob/v0.14.0/tinygrad/tensor.py), [optimizer source](https://github.com/tinygrad/tinygrad/blob/v0.14.0/tinygrad/nn/optim.py).

For a scalar objective `L=(w-2)²` and SGD learning rate 0.1:

```text
dL/dw = 2(w-2)
w_next = w - 0.1*2(w-2) = 0.8w + 0.4
w_0 = 0 -> w_t = 2(1-0.8^t)
```

The executed autograd lesson compares the twentieth update to this independent formula, yielding about **1.9769415**. It also compares the gradient of a summed RMS-normalized vector with central finite differences. The reviewed maximum difference was below `1e-7` for that fixture. Finite differences need an appropriate step and smooth test points; one result does not validate every operator/domain.

## 4. Train, separate the test set, then reload

The default classification exercise uses four tiny synthetic training examples and two separate synthetic test points. Its purpose is to check the training loop, cross entropy, optimizer and checkpoint. It is not driving data or a generalization benchmark.

The explicit `--mnist` option downloads MNIST through the pinned dataset loader and uses the first **512 training** and **256 test** images. A linear 784-to-10 classifier trains for 40 bounded minibatch updates. This reviewed run produced:

| Measurement | Observed result |
|---|---:|
| Initial full training loss | 2.302599 |
| Final full training loss | 0.502533 |
| Accuracy on the chosen 256 test images | 0.7890625 (202/256) |
| Maximum output difference after save/reload | 0 |

The small first-N subsets are not representative benchmark sampling, and the test result is not a 98% accuracy promise. The notebook-sized exercise teaches a complete working loop. Record dataset/split identity and training configuration before comparing changes. [Raw run](evidence/mnist-cpu-2026-10-18.json), [dataset loader source](https://github.com/tinygrad/tinygrad/blob/v0.14.0/tinygrad/nn/datasets.py).

**Transfer:** add a hidden layer, preserve the same test split and compare loss and accuracy. Tune using a separate validation split rather than repeatedly selecting settings from the test score. Add output parity after checkpoint reload before measuring inference time.

## 5. Inspect work that actually executes

The compiler lesson compares `(x*2+1).relu()` with and without forced intermediate realization. On the reviewed CPU configuration, it counted **three** model kernels for the materialized expression and **one** for the lazy expression. This is evidence for that expression and release, not a universal promise that an arbitrary chain becomes one kernel.

It also calls `TinyJit` with four different inputs of the same shape. The later output must reflect the new values: a repeated identical input would miss stale-buffer mistakes. Consume/copy output before the next call if the runtime reuses output storage.

A scheduler inspection can change the state of a graph. In this release, `schedule_linear()` produces a schedule for execution; merely calling it and then treating the original tensor as if no scheduling occurred is not a safe general inspection recipe. The beginner runner inspects its pre-execution UOp root and uses actual execution counters. The deeper [source lesson](notes/internals.md) explains how to trace the complete pinned pipeline.

Use `COURSE_TINYGRAD_DEBUG=4` on a separate run for generated-code inspection. Verbose tracing perturbs timing; this course's kernel-count lesson does not publish a latency speedup. For a timed comparison with warmup, synchronization and raw samples, use the [CPU compiler experiment](../../../Phase%204%20-%20Track%20C%20-%20DL%20Inference%20Optimization/00%20-%20CPU%20Compiler%20Lab/Guide.md).

## 6. Preserve operator meaning

The custom-operations lesson checks RMSNorm, SwiGLU and causal attention against independent scalar references:

- RMSNorm divides by the root mean square plus epsilon; it does not subtract the mean. State the normalized axes.
- With two already-projected branches, `SwiGLU(value,gate)=value × gate × sigmoid(gate)`. Omitting the factor `gate` changes the function. At gate zero the correct output is zero.
- Causal attention masks future score entries **before** softmax. Its first row can read only the first value vector. Slicing an output after an unmasked softmax does not repair future-token contamination.

Read the worked [elementwise](ops/elementwise.md), [reduction](ops/reduce.md) and [movement](ops/movement.md) lessons. Compose an operation first; a new private UOp or backend is a different engineering project with a larger contract.

## 7. Connect inference to an action contract

The final CPU lesson uses fixed illustrative weights `[-0.08,-0.4]` to map lateral error in metres and heading error in radians to road-wheel radians. For `[1,0.1]`, the expected output is `-0.12`. It checks shape, finite values, bounded output and the named manifest, including a deliberate wrong-unit rejection.

This small model is not an openpilot network and has no learned driving-quality claim. It isolates the numerical/interface obligations that an optimized model must preserve. The same manifest connects to the [vehicle replay](../2.%20openpilot%20Reference%20Stack/Replay-Lab.md); timestamp/freshness checks still apply after inference passes.

## 8. Finish with an actual runtime boundary

Project 7 compiles a C FP32 affine/ReLU function, loads it with `ctypes`, declares its ABI, allocates guarded buffers and checks 12 zero/tail/changing-input cases. This is an executed **standalone runtime exercise**, not a registered tinygrad backend. It makes allocation, layout, binary loading, lifetime and numerical comparison concrete before a larger integration.

To build a real tinygrad backend, study the pinned runtime interfaces and implement their full allocator/compiler/execution obligations. An ellipsis-filled skeleton or a fabricated `Device._devices[...]` registration is not a functioning backend.

## Run and assess

Use the [project commands](projects/README.md). Submit hand predictions, the JSON report, a failed wrong-unit check, the gradient comparison and a source path tracing one executed kernel. The stdlib references have seven tests; six additional tests execute the pinned optional framework when available. A skipped framework suite is not evidence of compiled inference.

Apply the [shared rubric](../../../Learning-Guide.md#completion-rubric), then carry input/output, memory and timing contracts to [advanced perception](../4.%20Advanced%20Perception%20and%20Prediction/Guide.md).
