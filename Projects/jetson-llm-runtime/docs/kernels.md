# CUDA kernels: specify the operation before tuning it

These are experimental kernel sources, not a validated pretrained inference engine. The review compiled CUDA translation units for `sm_87` but had no GPU execution. Start with [testing](testing.md) and keep compile-only results separate from numerical and performance results.

## 1. Follow one decode step

A simplified transformer block forms Q, K and V projections, applies position encoding, attends to valid cached tokens, applies an output projection and residual, then evaluates a feed-forward block. Normalization and residual placement depend on the exact architecture.

Drawing this dataflow is useful, but it does not establish model support. Tensor encodings, head dimensions, activation functions, RoPE layout/scaling and tokenizer semantics must match a pinned model specification.

## 2. Q4_K matrix-vector multiplication

`gemv_q4.cu` stores a 256-weight block in 144 bytes, including packed scales and minima. The value reconstruction uses both a scale and a minimum term; treating every nibble as a generic signed INT4 number is incorrect.

The current kernel assigns four rows per 128-thread block, one warp per row. Lanes distribute complete Q4_K blocks and reduce partial dot products. K must be divisible by 256; the wrapper rejects incompatible shapes. This is a source-level work assignment, not measured evidence of optimal coalescing or occupancy.

**Reference exercise:** independently decode a block with nonzero scale and minimum values. Compare every reconstructed weight, then multiply by a vector with mixed signs. Test multiple blocks and an output-row count not divisible by four. A zero-filled block cannot expose many layout errors.

Fusing dequantization avoids materializing a complete FP16 weight matrix, but speedup depends on traffic, decoding instructions and reuse. The [performance lesson](performance.md) derives the 3.56x storage ratio without turning it into an invented speedup.

## 3. RMS normalization is not variance normalization

For a vector x of dimension d:

```text
rms_inverse = 1 / sqrt(sum(x[i]^2)/d + epsilon)
y[i] = x[i] * rms_inverse * weight[i]
```

RMSNorm does not subtract the mean. If a residual is fused, state exactly which intermediate is normalized and which residual value is retained for the next operation.

Use a reference with nonconstant x, unequal weights, positive epsilon and signed values. Check small dimensions, dimensions beyond one block and tails. Shared-memory requirements, register use and launch constraints must come from the actual kernel and target, not a table copied from another GPU.

## 4. Attention needs a layout and a stable reduction

For one query head:

```text
score[t] = dot(q, key[t]) / sqrt(head_dimension)
probability = softmax(score over valid tokens)
output = sum_t probability[t] * value[t]
```

Grouped-query attention maps several query heads to one KV head. A token-major cache `[token, KV_head, dimension]` is not interchangeable with a head-major cache `[KV_head, token, dimension]`. Include at least two heads and several tokens in the fixture; a one-head case hides the distinction.

Online softmax processes score tiles while maintaining a running maximum m, denominator l and weighted-value accumulator o. When a new maximum m_new appears, rescale old l and o by `exp(m - m_new)` before adding the new tile. The final result is `o/l`.

Negative scores matter. Reinterpreting a negative float as a signed integer and applying integer maximum does not implement floating-point maximum. Also test an incomplete final tile, context longer than one tile and large score differences. For decode, the query has a score vector of length T; do not claim to avoid an already materialized T-by-T matrix that this single-query operation never required.

The [worked attention lab](../labs/Attention-Layout-Lab.md) now supplies the
corrected token-major reader, floating warp/block reductions, a double-precision
CPU oracle, and a separate actual-kernel parity harness. Thirteen fixture cases
cover negative scores, GQA, partial tiles, changing prefixes and explicit INT8
scales. The CPU mathematical checks ran; the CUDA harness compiled and returned
skip code 77 because no usable GPU was present. The runtime still rejects INT8
cache mode because it has no calibrated append-time scale storage.

`flash_attention_decode()` returns validation/launch status. Its caller checks
that status and later stream completion; launch success alone does not establish
execution success. Output buffers are poisoned separately for each parity case.

## 5. Position and activation checks

RoPE rotates pairs using a position-dependent angle. At position zero it is an identity, so also test nonzero positions, pair layout and the model's scaling convention. Interleaved and split-half layouts can require different indexing while both look plausible.

The experimental GPU mapper requires an even head dimension of at least two,
as well as its existing architecture and encoding restrictions. Generic metadata
inspection remains broader; inspecting a model shape does not admit that model
to the runtime.

SwiGLU combines `silu(gate) * up`, where `silu(x)=x/(1+exp(-x))`. Include large positive and negative inputs and compare with a stable reference. Floating-point overflow or underflow must be interpreted under a stated output contract.

## 6. Optimization requires a baseline

Orin Nano has eight SMs, not sixteen. Compute capability 8.7's shared-memory limits and opt-in requirements differ from a generic 48 KiB default allocation threshold. Query the device, inspect compiler resource reports, and check every launch.

Once reference comparisons pass on the actual GPU, measure the complete workload. Record warmup, synchronization, input shape, clocks and repeat statistics. A smaller kernel count may increase registers or reduce occupancy; it is not automatically faster.

**Finish:** select one operation, write its independent reference, identify a fixture that defeats a simplistic smoke test, and retain the actual pass/fail/skip result. Follow the [runtime milestones](../ROADMAP.md) before claiming complete model support.
