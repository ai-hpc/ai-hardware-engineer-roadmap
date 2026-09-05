# Module 04 — KDA II: Chunked Parallelism & the Full Sublayer

**Collection:** [GLM-5.3-Flash Architecture Mastery](README.md) | **Previous:** [← Module 03](Lecture-03.md) | **Next:** [Module 05 →](Lecture-05.md)

---

[Module 03](Lecture-03.md) derived the recurrence correctly, but a token-by-token Python loop implementing it would be useless on real hardware: prefill hands you hundreds or thousands of tokens at once, and discarding that parallelism to satisfy a serial recurrence throws away nearly all of the GPU's throughput. This module covers the two things a production KDA implementation needs that the recurrence alone does not give you: a way to parallelize it, and everything in the sublayer that surrounds it.

---

## Learning objectives

By the end of this module you should be able to:

1. Derive the affine-composition identity that makes chunked/parallel KDA possible.
2. Explain why decode and prefill need structurally different kernels for the same recurrence.
3. List every component of the KDA sublayer beyond the state update, and explain why benchmarking the recurrence alone understates layer latency.
4. State the two-part correctness invariant a chunked implementation must satisfy against the recurrent reference.

---

## 1. The recurrence has affine form

[Module 03](Lecture-03.md)'s closed form:

```text
   S_t  =  (I − β_t k_t k_tᵀ) D_t · S_{t−1}  +  β_t k_t v_tᵀ
```

is an instance of the general affine update:

```text
   S_t  =  A_t · S_{t−1}  +  B_t

   where   A_t  =  (I − β_t k_t k_tᵀ) D_t          B_t  =  β_t k_t v_tᵀ
```

Affine updates **compose**. Substitute the update for step `t` into the update for step `t+1`:

```text
   S_{t+1}  =  A_{t+1} · S_t  +  B_{t+1}
            =  A_{t+1} · (A_t · S_{t−1} + B_t)  +  B_{t+1}
            =  (A_{t+1} A_t) · S_{t−1}  +  (A_{t+1} B_t + B_{t+1})
```

Two consecutive steps collapse into a single affine update with a **composed** transition matrix `A_{t+1}A_t` and a **composed** offset `A_{t+1}B_t + B_{t+1}`. Nothing stops you from continuing this for an entire chunk of `L` tokens: the whole chunk's effect on the incoming state reduces to one composed `A` and one composed `B`, computable independently of the state you had *before* the chunk started. That independence is exactly what makes chunk-level parallelism possible — you can compute each chunk's composed transition while chunks are processed in parallel across sequence positions, then apply the (cheap, sequential) chunk-to-chunk composition afterward.

```text
   ┌────────────────────────────────────────────────────────────────────┐
   │  This explains WHY parallel formulations of the delta rule exist.  │
   │  It does NOT mean a good kernel should materialize the dense        │
   │  d_k × d_k matrix A_t for every token and multiply them out         │
   │  explicitly — that throws away the diagonal (D_t) and rank-one      │
   │  ((I − β k kᵀ)) structure that makes each A_t cheap to apply in     │
   │  the first place. A real chunked kernel keeps A_t and B_t in their  │
   │  structured (diagonal + rank-one) form throughout the composition. │
   └────────────────────────────────────────────────────────────────────┘
```

---

## 2. Why decode and prefill need different kernels

The affine-composition trick tells you parallelism is *mathematically available*; it does not tell you decode should use it.

```text
   DECODE                                    PREFILL
   ────────────────────────────              ────────────────────────────
   ONE new token arrives at a time            HUNDREDS/THOUSANDS of tokens
                                               arrive at once (the prompt)
   the incoming state S_{t-1} is already      no state exists yet for
   sitting in memory, ready to use            most of the sequence — it
                                               has to be BUILT, in order
   a single fused recurrent step is           a serial token-by-token loop
   the natural, cheap operation                discards nearly all available
                                               parallelism across the L
                                               prompt tokens
   → FUSED RECURRENT KERNEL                   → CHUNKED KERNEL using §1's
                                                 composition to process
                                                 blocks of tokens in parallel,
                                                 then compose block results
                                                 sequentially
```

This is the same prefill/decode asymmetry that governs ordinary transformer attention — prefill is throughput-oriented and can exploit parallelism across positions, decode is latency-oriented and bottlenecked on state that must be produced in order — applied to a recurrent mechanism instead of a quadratic one. A serving system for this model needs **both** kernel families for KDA, selected by request phase, exactly as it needs separate prefill and decode paths for the DSA/MLA layers.

---

## 3. The sublayer is bigger than the recurrence

The recurrence in [Module 03](Lecture-03.md) is the mathematical core, but the executed KDA sublayer wraps it in several additional operations:

```text
   x  ──▶  Q/K/V projections
       ──▶  short causal convolution  (on Q, K, and/or V — a small local mixing step)
       ──▶  Q/K normalization
       ──▶  learned decay gate   →  produces α_t (and hence D_t)
       ──▶  learned write gate   →  produces β_t
       ──▶  THE RECURRENCE  (Module 03)
       ──▶  gated normalization on the output
       ──▶  output projection
       ──▶  back into the residual stream (via mHC — Module 07)
```

```text
   ┌──────────────────────────────────────────────────────────────────┐
   │   Do not benchmark only the state update and call that            │
   │   "KDA layer latency." The projections, convolution, and gating   │
   │   surrounding the recurrence can account for a substantial        │
   │   share of a KDA sublayer's execution time even when the           │
   │   recurrence itself is efficiently implemented.                    │
   └──────────────────────────────────────────────────────────────────┘
```

This has a direct consequence for [Module 10](Lecture-10.md)'s profiling discipline: when you set out to optimize "KDA," first determine *which* of these eight stages is actually consuming the time budget you're trying to reduce. A beautifully fused recurrent kernel delivers nothing if the short convolution or the gating projections are what's actually dominating the sublayer's latency.

---

## 4. The correctness invariant

Because decode and prefill use structurally different kernels for the *same* mathematical recurrence, and because real serving mixes both (a prefix processed once during prefill, continued token-by-token during decode, sometimes re-processed in chunks for retries or speculative rollback), a chunked implementation's correctness bar is higher than "the numbers look close on one example":

```text
   REQUIRED:  recurrent execution  ≡  chunked execution

              on BOTH of:

              (a)  the output sequence  o_1, o_2, ..., o_L
              (b)  the FINAL STATE  S_L

              agreeing within your chosen numerical tolerance.
```

```text
   ┌────────────────────────────────────────────────────────────────────┐
   │   Matching outputs on one short prompt is NOT sufficient.           │
   │                                                                      │
   │   An order-of-operations bug (Module 03 §3) or an off-by-one in     │
   │   chunk boundary handling can produce outputs that look correct     │
   │   over a short window while the carried STATE has already            │
   │   diverged — and that divergence only becomes visible several       │
   │   tokens later, or after the state crosses a chunk boundary, or      │
   │   after it is checkpointed and resumed in a later request turn.     │
   └────────────────────────────────────────────────────────────────────┘
```

The concrete test design, expanding on [Module 03](Lecture-03.md)'s single-step reference:

```python
def test_recurrent_vs_chunked(seq_len, chunk_sizes, tol=1e-4):
    x = random_sequence(seq_len)

    ref_outputs, ref_state = run_recurrent(x)              # token-by-token, Module 03's update

    for chunk_size in chunk_sizes:
        chunked_outputs, chunked_state = run_chunked(x, chunk_size)

        assert allclose(chunked_outputs, ref_outputs, tol), \
            f"OUTPUT mismatch at chunk_size={chunk_size}"
        assert allclose(chunked_state, ref_state, tol), \
            f"FINAL STATE mismatch at chunk_size={chunk_size}"   # ← the check most tests skip

        # also verify the STATE agrees at every chunk boundary, not just at the end —
        # this is what catches a boundary bug that happens to cancel out by seq_len
        for boundary in range(chunk_size, seq_len, chunk_size):
            assert allclose(chunked_state_at(boundary), ref_state_at(boundary), tol)
```

Run this across chunk sizes that do **and do not** evenly divide `seq_len` — an implementation that only handles full chunks correctly and mishandles the trailing partial chunk is a common, easy-to-miss failure mode, and the same boundary discipline you will need again for DSA's incomplete-tail pooling in [Module 06](Lecture-06.md).

---

## Checkpoint

You should now be able to:

1. Derive `S_{t+1} = (A_{t+1}A_t)S_{t-1} + (A_{t+1}B_t + B_{t+1})` from the affine form and explain what it enables.
2. Explain why materializing dense transition matrices would defeat the purpose of the composition trick.
3. Name all eight stages of the KDA sublayer beyond the recurrence itself.
4. State the two-part (output + final state) correctness invariant and design a test that catches a boundary-handling bug.

---

## Ship it

This is **Stage 3 of the [capstone ladder](Lecture-12.md)**: extend Module 03's reference implementation with a chunked/parallel execution path using the affine composition from §1, then run the `test_recurrent_vs_chunked` design above across at least three chunk sizes, including ones that do not evenly divide your test sequence length. Report agreement on outputs, final state, and state at every chunk boundary — not final state alone.

---

## Current as of

* **Timeless:** the affine-composition identity and why it enables chunk-parallel execution, the prefill/decode kernel-family distinction, the two-part correctness invariant.
* **Checkpoint-specific:** the exact stage list in §3 (which normalizations, which gates, convolution kernel width) should be verified against the reference implementation's actual KDA sublayer — the ordering shown here is the conceptual shape, not a guarantee of the precise operation sequence in every revision.

---

**Next:** [Module 05 — MLA: Compressing Per-Token History →](Lecture-05.md)
