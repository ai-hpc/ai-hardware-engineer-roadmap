# Lecture 02 — A schedule changes execution, including the tails

A compute definition states the result. A schedule chooses loop structure, parallel placement, and memory use. A legal transformation must preserve the dependence and numerical contract; a fast transformation must also suit the target and shape. The [CPU lab](labs/README.md) makes both concerns visible with 37 elements.

## Begin with a simple loop

```text
for i in 0..36:
    B[i] = 2*A[i] + 1
```

The low-level function in the tested TVM release uses `@T.prim_func(s_tir=True)`, `T.sblock`, and a spatial axis. The function declares non-aliasing buffers; callers must honor that promise. A reduction would need reduction axes and correct initialization, not just another spatial loop.

The schedule obtains the block with `get_sblock('affine')`, obtains its loop, splits it into an outer and inner loop, and vectorizes the inner loop. The complete executable source is [compiler_lab.py](labs/compiler_lab.py); its imports match the release wheel, not a guessed mixture of old and new APIs.

## Why 37 rather than 32?

A tile of eight covers five groups: indices 0–7, 8–15, 16–23, 24–31, and 32–39. Only 32–36 are valid in the last group. The transformed IR needs a predicate equivalent to `8*outer+inner < 37`, or another legal tail strategy.

The test checks every valid result and sentinel values after the logical buffer. It supplies a 37-element DLPack view backed by a larger allocation; a store to positions 37–63 changes a sentinel and fails. Tile sizes 1, 4, 8, 16, and 64 cover exact single-element, partial-vector, and tile-larger-than-domain cases. Checking only a divisible length would miss the main boundary bug.

Vectorization expresses a legal vector operation; it does not guarantee a particular SIMD width or a speedup. LLVM may scalarize, mask, or choose another instruction sequence based on target capabilities and tails. Inspect generated code if instruction selection matters.

## Extend the reasoning to matmul

For `C[i,j] += A[i,k]*B[k,j]`, i and j are output dimensions and k is a reduction. Tiling i/j/k can improve reuse, but initialization must happen once per output and the reduction must accumulate every k exactly as intended. Moving an initialization inside the k tile can silently keep only the final partial sum.

A tile working set includes its A/B fragments, output accumulators, staging buffers, and metadata. Larger tiles can reduce repeated loads but increase register/shared-memory use and reduce parallelism. There is no single tile that is best for every shape or target. Reordering a floating-point reduction can change rounding; establish acceptable error before tuning.

On GPUs, binding loops to blocks/threads determines which workers own outputs. Shared-memory caches require synchronization before consumers read and before producers overwrite. Tensorization replaces a supported subcomputation with an instruction/intrinsic contract: shape, layout, alignment, dtype, accumulator, and synchronization all matter. Matching only the arithmetic expression is insufficient.

## Read a schedule trace as a hypothesis

For each change, write the intended effect: “reuse this tile,” “make this axis contiguous,” or “expose independent outputs.” Then check legality, output correctness, resource use, and measured time. Some changes reduce traffic but add instructions; some improve one shape and regress another. A roofline is a bound, not proof that every scheduling primitive moves a kernel toward it.

The [upstream transformation tutorial](https://tvm.apache.org/docs/deep_dive/tensor_ir/tutorials/tir_transformation.html) gives current development examples. Use its matching API snapshot; old `T.block/get_block` examples are not directly the same as the tested `T.sblock/get_sblock` interface.

**Exercise.** If a 37-element vector is padded to 40, may the kernel write all 40 elements? **Answer:** only if the API explicitly allocates and owns those extra outputs and defines their treatment. An internal allocation strategy cannot silently weaken a caller's 37-element buffer contract.

**Deliverable.** The split/vectorized IR, five tested tile sizes, sentinel evidence, and a short explanation of the tail predicate. Next: [tuning](Lecture-03.md).
