# Compiler and accelerator engineering: defend a transformation

Practice connecting a mathematical operation to its memory traffic, numerical
behavior, schedule, and implementation evidence. These are original design
exercises with analytical numbers, not claims about a measured accelerator.

**Prerequisites:** [computer architecture](../../Phase%201%20-%20Foundational%20Knowledge/2.%20Computer%20Architecture%20and%20Hardware/Guide.md)
and [inference/compiler fundamentals](../../Phase%204%20-%20Track%20C%20-%20DL%20Inference%20Optimization/Guide.md).
**Artifact:** a justified optimization or accelerator proposal with reference
outputs and a capacity/traffic calculation. Allow two 60–90 minute sessions.

## 1. A one-TFLOP/s accelerator is slow on a matrix-vector product

Compute `y = W x` with `W` of shape `[1024, 1024]`. All stored values are FP16.
Assume cold weights and input, one output write, no intermediate spills,
50 decimal GB/s sustained memory bandwidth, and one TFLOP/s peak arithmetic.
Count a multiply-add as two operations. What are the lower-bound times?

<details markdown="1"><summary>Answer guidance</summary>

The operation count is `2 × 1024 × 1024 = 2,097,152`. Traffic is
`2 × (1024² + 1024 + 1024) = 2,101,248 bytes`. Arithmetic intensity is about
0.998 FLOP/byte. The bandwidth time is about 42.0 microseconds; the peak-compute
time is about 2.10 microseconds. Even with ideal overlap, the larger constraint
sets a lower bound of about 42.0 microseconds under these assumptions.

Real execution adds launch, scheduling, alignment, reductions, and imperfect
utilization. This estimate explains why increasing peak arithmetic may not help.
It does not predict measured runtime precisely.

**Follow-up:** batch 16 independent vectors and reuse `W`. Recompute input/output
bytes and weight reuse. Then explain whether that batch size is compatible with
the robot application's action deadline.

</details>

## 2. Is a fused graph necessarily correct and faster?

You want to fuse an affine layer, normalization, and activation. What must you
establish before comparing performance?

<details markdown="1"><summary>Answer guidance</summary>

Specify exact operation semantics, shapes, broadcasting, layouts, dtypes,
training/evaluation mode, and tolerances. Folding batch normalization using fixed
inference statistics differs from training-time normalization. Reordering
floating-point arithmetic can change rounding. Aliasing and in-place updates can
change which values later consumers observe.

Use a small scalar or tensor reference, normal cases, and boundary cases such as
odd dimensions, noncontiguous input if supported, and very large/small values.
Compare outputs before timing. Then measure launches, global-memory traffic,
register pressure, occupancy where relevant, and the end-to-end critical path.
Fusion can increase resource pressure or remove useful parallelism.

**Follow-up:** the fused kernel is faster in isolation but causes a larger
workspace or downstream layout conversion. Include those effects in the workload
comparison.

</details>

## 3. Eight MiB of SRAM and a nine-MiB working set

A design has 8 MiB of SRAM. The proposed always-resident weights use 6 MiB, and
live activations and DMA buffers use 3 MiB. Is the design feasible? How would you
revise the schedule?

<details markdown="1"><summary>Answer guidance</summary>

The simultaneous requirement is 9 MiB, so it does not fit as stated. Check whether
the quoted 3 MiB includes double buffering, alignment, scales, descriptors, and
reserved capacity. Use a liveness timeline to identify buffers that can safely
share storage. Otherwise reduce resident weights, tile the workload, spill some
state, or change the hardware budget.

Recompute transfer volume and dependencies after changing residency. A warm
steady-state schedule and a cold first invocation have different traffic. If
weights vary between tasks, a residency assumption may not survive task changes.

**Follow-up:** two buffers have nonoverlapping compiler lifetimes, but DMA from
one has not finished. Physical completion and ownership extend the effective
lifetime beyond the last source-level use.

</details>

## 4. A tiled kernel fails only at width 65

The kernel processes tiles of width 32. Why can widths 32 and 64 pass while 65
fails? What tests would you add?

<details markdown="1"><summary>Answer guidance</summary>

Width 65 needs three tiles, with only one valid element in the last tile. Verify
load/store masks, reduction identities for invalid lanes, strides, and allocated
versus logical size. Padding an allocation is not permission to expose padded
values as real outputs.

Test widths around boundaries: 1, 31, 32, 33, 63, 64, 65, and a nonsquare shape.
Compare with an independent reference. Include a case where adjacent rows have
different sentinel values so an out-of-bounds row access cannot hide behind zeros.

**Follow-up:** a maximum reduction initializes masked lanes to zero. That is
incorrect when all valid elements are negative; the appropriate identity must
match the reduction's domain and numerical contract.

</details>

## 5. A graph passes numerical checks but the device hangs

What belongs in the compiler/runtime interface beyond tensor values?

<details markdown="1"><summary>Answer guidance</summary>

Define buffer layout, alignment, address/length bounds, access permissions,
operation encoding, version compatibility, submission ordering, completion,
timeouts, and error reporting. Specify which side owns a buffer before, during,
and after execution. A cache flush, fence, or synchronization primitive is a
platform-specific mechanism implementing that contract, not an optional detail.

Test malformed descriptors, unsupported operations, early buffer reuse, stale
completions, and device restart. Preserve the first failing submission and trace
the producer/consumer sequence. Numerical equivalence alone cannot verify a
runtime protocol.

**Follow-up:** an old completion arrives after restart with a reused request ID.
Explain how a session/generation identifier prevents it from releasing the wrong
buffer.

</details>

## 6. A prototype reports a large speedup

Which evidence makes the claim useful to a reviewer? What changes between a CPU
cost model, RTL simulation, FPGA, and silicon?

<details markdown="1"><summary>Answer guidance</summary>

Name the exact workload, baseline, precision, quality criterion, device/tool
versions, timing boundaries, and measurement method. Include raw results and
failures. Separate compile/setup time from repeated execution when reporting
steady-state behavior, and report cold behavior when it matters to the product.

A CPU cost model checks arithmetic and assumptions. RTL simulation can validate
cycle/protocol behavior for modeled components. An FPGA demonstrates implemented
logic and interfaces at its achieved constraints. Silicon adds physical effects
that require actual characterization. None automatically supplies the other's
frequency, power, or end-to-end application measurements.

**Follow-up:** the new design is faster but has lower task success. Report useful
completed tasks and quality, not only operations/s. Repeated failed attempts can
change energy per successful task.

</details>

## Final design review

Bring a one-page proposal containing an operator contract, shape/precision table,
traffic calculation, memory-lifetime diagram, correctness evidence, and one
limiting assumption. Have a reviewer halve memory bandwidth or add an odd-shaped
operator. Revise the proposal and identify which measurements must be repeated.

For implementation context, follow the official [MLIR Toy tutorial](https://mlir.llvm.org/docs/Tutorials/Toy/)
and the roadmap's [programmable VLA accelerator course](../../Phase%205%20-%20Advanced%20Topics%20and%20Specialization/Track%20F%20-%20AI%20Chip%20Design/Programmable%20VLA%20Inference%20Accelerator/README.md).
Compiler frameworks evolve; record the source revision used for a reproducible
implementation rather than treating the tutorial's moving website as a build pin.

**Exit criteria:** explain the calculation, show why the transformation is legal,
reproduce a boundary failure, and distinguish measured evidence from a model.

[Interview overview](../README.md) · [Learning rubric](../../Learning-Guide.md)
