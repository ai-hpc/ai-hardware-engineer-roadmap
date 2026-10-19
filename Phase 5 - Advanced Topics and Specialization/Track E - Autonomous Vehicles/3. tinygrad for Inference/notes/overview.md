# Three questions for every tensor program

Use the [pinned 0.14.0 guide](../Guide.md) and [executed projects](../projects/README.md). The purpose of a small framework is to make semantics and execution easier to inspect, not to make their contracts disappear.

## 1. What mathematical function is intended?

For a driving model, the input contract includes image format, calibration, normalization, history and capture identity. The output contract includes feature order, units, coordinate frame and a time horizon. For the small course expression, it is simply FP32 affine/ReLU on a stated shape.

Work one scalar or matrix result before inspecting a compiler. `[[1,2,3],[-1,0,2]] @ [[1,0],[2,1],[-1,3]]` must produce `[[2,11],[-3,6]]`. A pretty generated kernel cannot repair an incorrect expected axis order.

## 2. What work actually executes?

Tensor expressions can remain lazy until their values are needed. A realization boundary can force a buffer that a larger fused expression would avoid. Index transformations may be incorporated into later computation; a consumer can also require contiguous storage or a transfer.

For this course's affine/add/ReLU expression, the reviewed CPU runtime executed three kernels with forced intermediates and one with a final-only realization. Kernel count is an observation about this program. It is not a proof of lower latency, lower energy or the same behavior on every backend.

The JIT exercise changes values at fixed shape across four calls. This establishes that the observed replay outputs consume new inputs in the tested cases. It does not establish support for arbitrary dynamic shapes, mutation patterns or hidden external state.

## 3. What can the result justify?

| Evidence | Valid conclusion | Additional evidence needed |
|---|---|---|
| Scalar reference match | The tested numerical mapping agrees | More domains, shapes and tolerances |
| Finite-difference gradient check | Derivative agrees at selected smooth points | Nonsmooth conventions, extreme values, other operators |
| Checkpoint output parity | Reload preserves tested inference output | Full deployment preprocessing and artifact identity |
| Fewer executed kernels | Materialization/scheduling changed | Fair measured latency/resource comparison |
| Model output passes schema | Its declared meaning and values fit this interface | Acquisition age, controller feasibility, task performance |

For the steering fixture, `[-.08,-.4]` maps `[lateral_error_m,heading_error_rad]` to road-wheel radians. The fixture's fixed weights are illustrative; compiling them successfully is not a learned policy-quality result.

## Learn by changing one boundary

Run the baseline, predict a deliberate failure, then change one variable. Swap feature order; change declared output units; force an intermediate buffer; remove a causal mask; reload weights under a different normalization. Each should answer a specific question with an observable difference.

The [source walkthrough](internals.md) explains where to inspect the pinned implementation. The [operator lessons](../ops/README.md) supply hand calculations and counterexamples. Use the same known inputs when moving from Python tensors to the standalone C runtime, and preserve an independent expected result.
