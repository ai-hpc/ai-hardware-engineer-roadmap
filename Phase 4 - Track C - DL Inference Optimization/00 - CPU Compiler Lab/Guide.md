# Your first inference compiler experiment

Take a small neural-network graph from a mathematical reference to generated CPU code, prove one rewrite is legal, and measure what actually changed.

**Layer mapping:** L1 model semantics → L2 graph transformation and compiler → L3 execution. **Roles:** inference, compiler, kernel, and robot deployment engineers. **Before:** Python functions, matrix shapes, and [AI foundations](../../Phase%203%20-%20Artificial%20Intelligence/0.%20AI%20and%20Robot%20Learning%20Foundations/Guide.md). **After:** [graph optimization](../01%20-%20Graph%20and%20Operator%20Optimization/Guide.md), then [compiler stack](../03%20-%20Compiler%20Stack/Guide.md).

**Access:** the mathematical reference uses only Python's standard library. The actual compiler experiment uses Python 3.11+, Clang, and a pinned tinygrad package in an isolated environment, with a supported CPU runtime. No model download or GPU is needed. Local execution was tested on Linux x86-64, Python 3.12.3, Clang 18.1.3, tinygrad 0.14.0 on 2026-10-18; other platforms have not been validated here.

**Effort:** setup 10–20 min; reference and compiler lab 60–90 min; interpretation 30 min. Compilation and execution should finish within a minute for these tiny shapes. Stop and inspect a setup failure rather than repeatedly launching a long job.

## 1. The problem: fewer operations must still mean the same function

Imagine the final layers of a perception network on a robot. They compute a linear transform, normalize each output channel using fixed statistics, and apply ReLU:

```text
X [B,K] -> Linear(W [K,N], b [N]) -> BatchNorm(eval) -> ReLU -> Y [B,N]
```

`B` is batch size, `K` input features, and `N` output channels. Our example has no learned task accuracy claim; it isolates the compiler's numerical obligations. Saving a few microseconds in this graph also does not prove the whole robot pipeline meets its deadline.

There are two different optimization decisions:

- **Algebraic rewriting:** replace the linear transform and fixed BatchNorm with new constant weights and biases.
- **Kernel fusion:** a compiler may execute neighboring operations without writing every intermediate array to memory.

A graph with fewer nodes need not have fewer executed kernels. A compiler may already fuse the original graph. The experiment makes this difference visible.

## 2. Work the rewrite by hand

For output channel `j`:

```text
z[j] = sum_k X[k] W[k,j] + b[j]
normalized[j] = gamma[j] × (z[j] - mean[j]) / sqrt(variance[j] + epsilon) + beta[j]
Y[j] = max(0, normalized[j])

a[j] = gamma[j] / sqrt(variance[j] + epsilon)
W_folded[k,j] = W[k,j] × a[j]
b_folded[j] = a[j] × (b[j] - mean[j]) + beta[j]
```

Now `ReLU(X @ W_folded + b_folded)` computes the same real-valued function. Floating-point evaluation can round differently, so define a numerical tolerance before comparing.

**Worked scalar:** `W=2`, `b=1`, `mean=3`, `variance=3`, `epsilon=1`, `gamma=4`, `beta=5`. Then `a=2`, `W_folded=4`, `b_folded=1`. At `X=2`, both graphs produce `9`.

**Legality conditions:** all parameters are finite; shapes and channel axes agree; epsilon is positive; variance is nonnegative; normalization uses fixed statistics. The folded artifact must be regenerated if weights or normalization statistics change. This function's contract excludes NaN/infinity inputs.

**Counterexample:** let `z=-1`, `gamma=-1`, `mean=beta=0`, and `sqrt(variance+epsilon)=1`. Applying normalization then ReLU gives `1`; applying ReLU before normalization gives `0`. Negative scale makes this especially easy to see. Moving the activation is a different rewrite and is not justified by the folding formula.

## 3. Run the independent mathematical reference

From the repository root, run these **tested-here** commands:

```bash
python3 "Phase 4 - Track C - DL Inference Optimization/00 - CPU Compiler Lab/labs/reference.py"
python3 -m unittest discover -s "Phase 4 - Track C - DL Inference Optimization/00 - CPU Compiler Lab/labs" -p "test_*.py" -v
```

Seven tests cover the hand calculation, negative scales, irregular shapes, training-mode rejection, invalid variance/epsilon, malformed/nonfinite inputs, and stale folded parameters. The test suite uses no compiler package. Passing it establishes the reference and rewrite checks; it does not establish compiled execution.

Open [reference.py](labs/reference.py). Follow `linear`, `reference`, then `fold_batchnorm`. Its scalar loops intentionally make indexing visible. They are a correctness oracle, not a competitive CPU kernel.

## 4. Install the isolated optional compiler

The environment used for this review had no ML compiler framework installed. We installed only the 2.8 MB tinygrad wheel in a temporary virtual environment. The package is pinned by version and wheel SHA-256 in [requirements-tinygrad.txt](labs/requirements-tinygrad.txt), matching the [0.14.0 PyPI release](https://pypi.org/project/tinygrad/0.14.0/), released 2026-08-24.

These are **tested-here** setup commands; run from the repository root:

```bash
python3 -m venv /tmp/course-trackc-venv
/tmp/course-trackc-venv/bin/python -m pip install --only-binary=:all: --require-hashes -r "Phase 4 - Track C - DL Inference Optimization/00 - CPU Compiler Lab/labs/requirements-tinygrad.txt"
```

The runner sets `DEV=CPU`, `BEAM=0`, and a numeric `DEBUG` before importing tinygrad. This matters because the host environment may use `DEBUG` for another program. No global Python packages are changed. If your platform lacks a supported CPU runtime or compiler dependency, retain the mathematical-reference result and mark the compiled route **not run**; do not substitute an unmeasured GPU claim.

Check `clang --version` before the compiled route. The tested CPU backend invokes Clang to compile rendered C; tinygrad's Python wheel does not install a system compiler. Use your system's package manager if Clang is missing. The runner records the compiler version/target arguments and CPU model in the report.

## 5. Compare three executed programs

```bash
/tmp/course-trackc-venv/bin/python "Phase 4 - Track C - DL Inference Optimization/00 - CPU Compiler Lab/labs/optional_tinygrad.py" --output /tmp/compiler-cpu-report.json
```

The [optional runner](labs/optional_tinygrad.py) is separate from the portable test suite and checks the pinned package version. It uses the same inputs and parameters for three variants:

| Variant | What it does | Question |
|---|---|---|
| `materialized` | Explicitly realizes the linear output and normalized output before ReLU | What does forcing intermediates cost? |
| `lazy` | Builds the original graph and realizes only the final output | What can the compiler fuse automatically? |
| `folded` | Uses precomputed folded weights/bias and then ReLU | Does constant rewriting improve on ordinary lazy fusion? |

**Data-movement prediction:** two distinct intermediate `[B,N]` FP32 tensors have a combined payload of `2 × B × N × 4` bytes. Writing and later reading both corresponds to `4 × B × N × 4` bytes of intermediate traffic if every value traverses that boundary. This is a data-flow estimate; caches, allocator reuse, and actual generated code determine observed memory behavior. For `(B,N)=(3,7)`, the payload is 168 bytes and estimated write/read traffic is 336 bytes, far too little to infer a bandwidth bottleneck from size alone.

It exercises `(B,K,N) = (1,3,4), (3,5,7), (8,16,9)`. Odd dimensions are intentional. Every variant is checked against the scalar reference; a second input with the same shape checks that JIT replay uses new data instead of a stale result.

The default records 30 warm calls. The first three calls, including execution/capture/replay setup, are reported separately. Timing covers the host `TinyJit` call plus CPU synchronization; it excludes input construction and output copying. This is **host-observed compiled inference latency**, not hardware-only kernel time. The first-call measurement can also reflect a populated disk cache; it is not guaranteed to be a cache-free compilation benchmark.

**Local observation:** all nine shape/variant pairs passed. The largest FP32 error was below `5e-7` in the reviewed run. tinygrad reported three kernels per warm `materialized` call and one for each `lazy`/`folded` call. Some tiny measurements were noisy enough that fewer kernels did not always mean a lower median. The JSON preserves every timing sample; reproduce the result on your machine instead of treating local microseconds as a target.

The reviewed run is preserved as [raw CPU results](evidence/tinygrad-cpu-2026-10-18.json) and a [separate generated-code trace](evidence/tinygrad-cpu-trace-2026-10-18.txt). These are measurements from the named local environment, not universal performance targets.

**Why not compare scalar Python speed with tinygrad speed?** That would mainly demonstrate interpreter and library overhead differences. The interesting comparison here holds framework, device, input, dtype, and timing boundary fixed while changing graph structure.

## 6. Inspect generated CPU code

Run a separate trace experiment; do not use verbose tracing for the latency comparison:

```bash
/tmp/course-trackc-venv/bin/python "Phase 4 - Track C - DL Inference Optimization/00 - CPU Compiler Lab/labs/optional_tinygrad.py" --debug 4 --repeats 5 --output /tmp/compiler-trace-report.json > /tmp/compiler-trace.txt 2>&1
```

This **tested-here** command emitted generated CPU source and execution records. The trace includes runtime support kernels as well as the model kernels. Find a model reduction/matmul kernel, identify its input and output pointers, and locate the reduction and activation. Compare the materialized sequence with the fused sequence. Count measured kernel launches from the JSON; counting every function in the trace would also count setup and helper code.

The current [tinygrad layout documentation](https://docs.tinygrad.org/developer/layout/) describes UOp graphs, scheduling, kernel optimization, rendering, and execution. It is useful orientation, but source paths and private APIs must be checked against the installed release. An old `LazyBuffer` diagram is not a map of every modern checkout.

## 7. Guided variations and answers

Predict each result before changing the code:

1. **Remove the two intermediate `.realize()` calls.** This should make that variant equivalent to the lazy expression; the scheduler can keep intermediates inside a fused computation. Measure rather than assuming an exact kernel count on every release.
2. **Change one running mean after folding.** The stored folded weights/bias no longer describe the updated graph. The stale-parameter test demonstrates the mismatch. A deployment package must version both learned and normalization state.
3. **Pass `training=True` to the rewrite.** It must reject the request: training-mode BatchNorm uses statistics from the current batch. Those cannot be replaced by these fixed constants for arbitrary future batches.
4. **Increase batch size and use a dimension just beyond a multiple of 16.** Recompute reference outputs, then inspect generated loops and timing. More work can change the bottleneck and scheduling strategy. Never extrapolate this fixture's microsecond ordering to a large GPU model.

## 8. Independent transfer and completion

Implement one second rewrite: fold a fixed input normalization into a linear layer. For `x_normalized[k] = (x[k]-m[k])/s[k]`, derive `W_new[k,j] = W[k,j]/s[k]` and `b_new[j] = b[j] - sum_k m[k]W[k,j]/s[k]`. Require finite nonzero scales and check a negative-scale example. Confirm the output before timing the compiled version.

Submit the derivation, failing counterexample, portable tests, pinned dependency file, JSON results, generated-code excerpt, and a paragraph explaining whether graph-node reduction and kernel-count reduction agreed. Use the [shared completion rubric](../../Learning-Guide.md#completion-rubric), with full marks in correctness and reproduction. The reference-only route can pass a mathematics milestone; the compiler milestone also requires actual compiler execution evidence.

**Known unexecuted outcomes:** GPU code generation/performance, PyTorch/TVM export, full-model quality, quantization kernels, and real robot timing. This lab establishes a small CPU compiler experiment and its numerical contract.
