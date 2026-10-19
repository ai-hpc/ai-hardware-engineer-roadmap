# Lecture 04 — Fusion, layouts, dynamic shapes, and external code generation

Graph optimization changes the boundaries between operations. Its value depends on memory traffic, launches, recomputation, resource pressure, and available libraries. It is not a collection of free wins. Use the lab's `relu(XW+b)` graph as a small example you can inspect completely.

## Fusion starts with dataflow

If matmul produces an intermediate Y, bias addition produces Z, and ReLU produces O, an unfused implementation may write/read each intermediate. An epilogue fused into matmul can apply bias and ReLU before the final output store. The precise traffic reduction depends on cache behavior and whether intermediate values are otherwise needed.

Suppose another consumer also needs raw Y. Removing Y's materialization is no longer obviously valid: the compiler may retain it, duplicate work, or choose a multi-output fused kernel. Fusion can also increase live values/register pressure, reduce occupancy, or prevent use of a highly optimized library call. Compare the complete graph, including layout conversions and extra consumers.

In the [CPU lab](labs/README.md), inspect the original and lowered modules rather than assuming `zero` fuses every operation. The tests establish output correctness for several odd shapes, but do not claim a particular fusion or speedup. If you add a pass, save before/after IR and identify the operation boundary that changed.

## Shapes and layouts are semantics

A logical `[M,K]` tensor can have different physical strides, blocked layouts, or packed quantized representations. A transpose may be a view or a copy depending on how a consumer accepts it. Layout conversion can cost more than the faster kernel it enables for a small problem.

Dynamic shape support means the compiled program has a defined symbolic-shape contract and supported runtime paths. It does not mean every arbitrary size, rank, dtype, or ragged layout is accepted. A compiler may generate guards, specialize common sizes, pad to buckets, or call a generic fallback. Test the smallest supported shapes, nonmultiples of tiles, and sizes that change the chosen path.

The static lab deliberately rejects X `[2,7]` when compiled for `[3,7]`. That failure is a useful contract check. To extend the exercise, design a symbolic batch dimension and repeat correctness and allocation tests for several values; do not remove shape checks merely to make invalid inputs run.

## Bring Your Own Codegen (BYOC)

BYOC partitions supported graph regions for an external code generator or library. A complete integration needs a pattern/support definition, legality checks, lowering/serialization, runtime invocation, tensor conversion/ownership, error propagation, and fallback behavior. Matching a graph pattern is only the start.

For an accelerator accepting a fixed INT8 matmul, the support predicate must check input/output shapes, transpose conventions, scales/zero points, accumulator/saturation behavior, alignment, memory/device placement, and supported activation epilogues. A floating-point reference may need to model quantization explicitly. Unsupported subgraphs must stay on a valid path, and transitions must be included in timings.

The [current BYOC tutorial](https://tvm.apache.org/docs/how_to/tutorials/bring_your_own_codegen.html) is the source for the moving development API. A CUTLASS/TensorRT integration requires the corresponding build options and versions; installing a generic CPU wheel does not establish that external backend works. MLC dependencies and older Relay BYOC examples have their own snapshot constraints.

## A worked offload decision

Assume a hypothetical external kernel saves 40 microseconds, while two layout/device conversions cost 30 microseconds each. Isolating the kernel suggests a win; the subgraph path loses 20 microseconds before additional synchronization. If several adjacent operations can remain on the accelerator, conversions may amortize. Measure the boundary, not only the fastest internal kernel.

For a branched graph, include the cost of preserving an intermediate needed by both consumers. For dynamic workloads, weight measurements by the observed shape distribution rather than selecting one favorable static shape.

**Exercise.** May a compiler reorder two writes to aliased buffers as though they were independent tensors? **Answer:** only when the alias/effect contract proves the reordering preserves observable behavior. Incorrect non-alias promises can make an apparently legal optimization corrupt data.

**Deliverable.** A diagram of one candidate fused/offloaded region, its input/output contract, every conversion, and a correctness/performance acceptance criterion. Next: [shipping an artifact](Lecture-05.md).
