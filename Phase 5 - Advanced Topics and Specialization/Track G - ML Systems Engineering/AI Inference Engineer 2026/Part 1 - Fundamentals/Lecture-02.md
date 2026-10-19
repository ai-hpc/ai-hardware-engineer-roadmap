# Part 1 · Lecture 02 — Follow one token through attention

Work through this lesson with the [CPU attention lab](labs/README.md). It executes a tiny reference using lists, then optionally compares it with PyTorch. No model weights or GPU are required. You should finish able to explain why an incorrect cache mask can return plausible but wrong outputs.

## Prefill and incremental decoding

A causal language model maps token IDs to vectors, applies transformer blocks, projects to vocabulary logits, and selects a next token. During prefill, many prompt positions are evaluated together. During ordinary incremental decoding, a request contributes one new position per step. Batching combines positions from several requests; speculative verification can contribute several positions from one request.

A block commonly contains normalization, attention, a residual addition, another normalization, an MLP, and another residual. Architectures vary: inspect the model implementation before assuming this exact order. RoPE applies a position-dependent rotation to query/key components; it does not make causal attention bidirectional.

For one sequence with `T` positions and hidden width `D`, a linear layer multiplies `[T,D] @ [D,F]`. Under the convention that multiply-add counts as two FLOPs, its work is approximately `2TDF`. With one token, the same weight matrix serves one vector. With many tokens, it can be reused across rows, which often increases arithmetic intensity.

## Queries, keys, and values

Let query heads be `Hq`, KV heads `Hkv`, and attention head dimension `d`. A query head computes scores against earlier keys, scales by `1/sqrt(d)`, applies the causal mask, normalizes with softmax, and uses those weights to sum values.

```text
score[i,j] = dot(Q[i], K[j]) / sqrt(d), allowed only when j <= i
probabilities[i] = softmax(allowed scores[i])
output[i] = sum_j probabilities[i,j] * V[j]
```

For grouped-query attention (GQA), several query heads share one K/V head. With `Hq=4` and `Hkv=2`, query heads 0–1 use KV head 0 and heads 2–3 use KV head 1. The lab checks that grouping. Read an explicit `head_dim` field when present: Qwen3-235B-A22B uses hidden width 4096, 64 query heads, and head dimension 128. Here `4096/64=64` is **not** the attention head dimension. Projection width need not equal residual width.

## What the cache saves

Past K/V vectors can be retained because later positions need them again. The cache avoids recomputing past projections and blocks; it does not remove the attention scan over past keys and values. For unsharded, unshared GQA caches with equal K/V dimensions:

```text
KV bytes = 2 × layers × KV_heads × head_dim × bytes_per_element
           × sum(request_context_lengths)
```

For 80 layers, 8 KV heads, head size 128, and two-byte elements, each token occupies 327,680 bytes = 320 KiB. One 131,072-token sequence occupies exactly 40 GiB before page rounding, metadata, workspace, or weights. This is storage capacity, not automatically HBM traffic per step: traffic depends on the kernel and cache hierarchy.

Paged caches round each request to allocated blocks. With 16-token pages, lengths 1 and 17 allocate 16+32=48 token slots, not 18 and not 32. Prefix sharing can reduce physical pages, but refcounts, copy-on-write, tokenizer/model identity, and isolation must be correct. Sliding windows and MLA use different accounting.

## The cached-mask trap

If a decode query represents absolute position 2 and the cache holds positions 0–2, it may attend to all three. A naive one-row lower-triangular matrix allows only column 0. With equal scores and values `[1,3,8]`, the correct answer is 4; the naive mask returns 1. Both are finite numbers, so a “no NaNs” test misses the bug.

The reference uses `key_position <= query_start + local_query_index`. The optional PyTorch test creates that boolean mask explicitly and sets dropout to zero. PyTorch SDPA documents upper-left causal alignment for a non-square matrix; do not assume every attention API chooses the same alignment. See the [PyTorch 2.14 SDPA contract](https://docs.pytorch.org/docs/2.14/generated/torch.nn.functional.scaled_dot_product_attention.html).

## Where time goes

Projection/MLP work grows roughly linearly with token count; full causal attention has a quadratic prefill component and a context-length-dependent decode component. FlashAttention changes data movement by avoiding a materialized full score/probability matrix; it does not erase the mathematical attention work. The [original FlashAttention paper](https://arxiv.org/abs/2205.14135) explains this IO distinction.

**Try it.** Run all prefix positions together, then in chunks of 1, 2, 3, and 7. Outputs should match the reference. Change a future value: earlier outputs must stay unchanged. Remove the absolute-position offset and retain the failing example in your notes.

**Check your understanding.** Does halving weight precision halve KV storage? **Answer:** only if KV precision is also changed; they are separate tensors and may use separate kernels/scales. Next: [roofline estimates](Lecture-03.md).
