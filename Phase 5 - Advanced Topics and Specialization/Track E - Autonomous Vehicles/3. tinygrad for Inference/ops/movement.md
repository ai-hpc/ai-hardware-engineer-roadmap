# Movement and layout: changing indices is still a contract

A tensor's shape describes logical entries. Its storage, stride, offset and consumer requirements determine how those entries are accessed. A view can avoid a copy while still changing the cost or legality of a later operation.

## 1. Work a transpose from addresses

A contiguous row-major 2×3 array has offset `row*3+column`. Its transpose has shape 3×2 and maps transposed `(i,j)` back to original `(j,i)`, giving offset `j*3+i`. The values of `[[1,2,3],[4,5,6]]` transpose to `[[1,4],[2,5],[3,6]]`.

A consumer that blindly assumes the transpose is contiguous would use offset `i*2+j`, producing a different sequence. Either the consumer supports the actual index mapping or the runtime materializes a compatible buffer. Calling a transformation “zero-copy” does not remove this obligation.

The basics lesson explicitly materializes a transposed tensor with `.contiguous()` and compares its values. Whether a transformation produces a kernel depends on the complete expression, runtime and already-realized storage. Do not promise zero kernels for every movement operation.

## 2. Reshape preserves element order, not arbitrary semantics

A reshape must preserve the number of elements. Reinterpreting 6 entries as 3×2 is valid arithmetically, but it is not the same operation as transposing a 2×3 matrix. For `[[1,2,3],[4,5,6]]`, reshape to 3×2 gives `[[1,2],[3,4],[5,6]]`.

**Checkpoint:** a model expects `[batch,channels,height,width]`, but a buffer is `[batch,height,width,channels]`. Reshaping to the requested dimensions does not perform the axis permutation. On a small uniquely numbered tensor, compare the expected channel values before optimizing anything.

## 3. Broadcasting reuses values

Broadcast `[1,2,3]` to two rows, then sum all six logical entries. The result is 12. The derivative with respect to each original element is 2 because each is used twice. Broadcasting does not create six independent trainable parameters.

The manual matmul lesson broadcasts `[2,3,1]` against `[1,3,2]`. The later reduction implements the shared inner dimension. Fusing this index expansion into the reduction can avoid materializing the entire conceptual product, but the logical shape remains useful for reasoning.

## 4. Connect layout to image preprocessing

An NV12 frame may have aligned row stride and plane heights. The logical pixel count alone cannot identify the UV offset or allocation size. The [camera lesson](../../2.%20openpilot%20Reference%20Stack/camerad/Guide.md#3-work-the-nv12-allocation-instead-of-guessing-it) shows a real source-reviewed allocation calculation and a padding-contamination example.

Image color order, crop, warp and normalization must also match the model. A mathematically correct matmul cannot compensate for swapping RGB/BGR or including row padding as visible pixels. Keep preprocessing versioned with the weights and output decoder.

## 5. Transfer to a runtime boundary

Project 7's C function accepts contiguous FP32 vectors only. To accept a strided view, add an explicit stride argument and update indexing, or materialize a contiguous input before the call. Test a stride-2 slice with guard values in skipped positions, zero length, and an odd tail. State whether stride counts bytes or elements.

The [pinned source walkthrough](../notes/internals.md) uses current UOps and runtime interfaces. Older ShapeTracker-based explanations can be historical context but are not a reliable source map for tinygrad 0.14.0.

**Completion:** explain transpose versus reshape with the six numbered entries, pass the non-square matrix check, and provide one example where a legal tensor shape still violates the consuming model's layout contract.
