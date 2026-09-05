# Module 06 — DSA & KPool: Selective Retrieval

**Collection:** [GLM-5.3-Flash Architecture Mastery](README.md) | **Previous:** [← Module 05](Lecture-05.md) | **Next:** [Module 07 →](Lecture-07.md)

---

[Module 05](Lecture-05.md) answered "how is each historical token represented?" This module answers a different question: **which historical positions does this query actually get to read?** DeepSeek Sparse Attention (DSA) separates a cheap selection mechanism from the expensive attention computation it gates — and GLM-5.3-Flash's specific implementation, pooled indexing, has boundary behavior that is easy to get subtly wrong.

---

## Learning objectives

By the end of this module you should be able to:

1. Distinguish the indexer's dimensions from the main MLA heads' dimensions, and explain why they differ.
2. Derive the indexer's selection budget, including the incomplete-tail case.
3. Explain the specific causal-visibility bug that pooled indexing makes possible, and name the sequence lengths where it clusters.
4. Derive why a fixed top-k selection budget does not imply constant-time decode.
5. Explain why pooling is a selection mechanism, not a cache-size reduction.

---

## 1. Indexer dimensions are not main-attention dimensions

```text
   Indexer property          Value               Main MLA (Module 05, for contrast)
   ────────────────────      ──────              ────────────────────────────────
   Indexer heads              32                  64 main heads
   Indexer head dimension     128                 256-dim main K/Q/V heads
   Pool width                 4 tokens            (no pooling — token-indexed)
   Main selection budget      2,048 positions     (no cap — full causal history)
   Incomplete-tail handling   enabled             n/a
```

These are two separate mechanisms with two separate parameter sets. The indexer's job is cheap approximate scoring over the *entire* causal history to find which positions are worth reading in full; the main MLA path then does the expensive, exact computation only over the positions the indexer selected. Confusing the two — for instance, assuming the indexer's pool width tells you anything about the main attention cache — is the single most common misreading of this mechanism, and §4 names it explicitly.

---

## 2. The selection budget, with its trap

The indexer groups tokens into **pools of 4**, builds a learned, channel-wise-weighted representation of each *complete* pool, scores those pool representations, and selects the highest-scoring pools up to the budget:

```text
   main selection budget  =  2,048 positions
   pool width             =  4 tokens

   complete pools selectable  =  2,048 / 4  =  512 pools
```

Selected pools are then **expanded back to their original constituent token positions** for the main attention computation — the pooling is used only to make scoring cheap, not to reduce what the main attention actually reads.

The current, still-incomplete pool at the query's position — up to 3 tokens that haven't yet formed a complete group of 4 — can be appended separately as a **visible tail**, since a query is always allowed to see itself and its immediate, causally-valid recent context regardless of pool completion:

```text
   max index slots required  =  512 pools × 4 tokens/pool  +  3 tail positions
                              =  2,051 slots
```

```text
   ┌────────────────────────────────────────────────────────────────────┐
   │  A buffer sized to a hard 2,048 slots — the "selection budget"     │
   │  number everyone quotes — will overflow by up to 3 positions on    │
   │  the incomplete-tail case. Size index buffers to 2,051 (or         │
   │  whatever your pool width and budget actually imply), including   │
   │  room for invalid/padded entries, not to the round headline        │
   │  number.                                                            │
   └────────────────────────────────────────────────────────────────────┘
```

---

## 3. Causality is more delicate than a triangular mask

Ordinary causal attention needs one rule: position `i` may attend to position `j` only if `j ≤ i`. Pooled indexing adds a second rule that is easy to miss: **a pool may only become selectable once its final constituent token is causally visible to the query.** If a pool's representation is allowed to influence scoring before its last token exists, that pool representation has effectively leaked information from the future into a selection decision made for an earlier query.

```text
   pool = tokens [4, 5, 6, 7]     (pool width 4, this is the 2nd pool: positions 4-7)

   a query at position 5 must NOT be able to select this pool's
   representation — the pool isn't "complete" until position 7 exists,
   and position 7 is in this query's future.

   a query at position 8 (or later) MAY select this pool — all four
   of its constituent tokens are now in the past.
```

This is exactly the kind of boundary condition that clusters around small sequence lengths and pool-boundary arithmetic, particularly once you add padding or packed multi-document batches to the mix:

```text
   Watch lengths:  3, 4, 5, 7, 8, 9

   3  →  a sequence that never completes its first pool at all
         (does the visible-tail path handle this correctly on its own?)
   4  →  exactly one complete pool, zero tail — an off-by-one-prone boundary
   5  →  one complete pool + a 1-token tail
   7  →  one complete pool + a 3-token tail (the MAXIMUM tail size)
   8  →  exactly two complete pools, zero tail — another exact-boundary case
   9  →  two complete pools + a 1-token tail
```

A correctness suite for this mechanism should construct sequences at every one of these lengths — and repeat the exercise with padded and packed-document inputs, where a document boundary can silently reintroduce the same off-by-one class of bug in a different guise (a pool spanning a document boundary must not let one document's tokens contribute to another document's selection or attention, exactly as ordinary packed-sequence attention must prevent cross-document leakage).

---

## 4. Pooling is not a fourfold cache reduction

This is the trap named in §1, stated as its own rule because it is worth repeating on its own:

```text
   pool width 4   ⇏   main latent cache divided by 4
```

The pool representation exists **only to make indexer scoring cheap**. Selected pools are expanded back to full constituent token positions before the main MLA attention ([Module 05](Lecture-05.md)) reads them — the main attention's per-token latent cache is exactly as large as [Module 05 §2](Lecture-05.md) derived, regardless of the indexer's pool width. Pooling and the MLA latent cache are two different data structures serving two different purposes, and neither one's size follows from the other's configuration.

---

## 5. Fixed top-k does not mean constant decode cost

The main attention reads a roughly-fixed number `K` of positions once selection completes — but selection itself is not free, and its cost scales with context length even when the thing it *selects* does not.

For a context of `T` tokens, decode cost decomposes into (at least) three separately-scaling terms:

```text
   F_decode(T)  ≈  F_projections+MoE+KDA                    ← Modules 02–04, roughly context-independent per step
               +  O( H_I · d_I · T/4 )                       ← INDEXER: scores ~T/4 candidate pools, every step
               +  O( H_MLA · K · r )                          ← MAIN ATTENTION: reads only the selected K positions
```

The middle term is the one a "fixed top-k means flat latency" intuition misses: **the indexer must still score approximately `T/4` candidate pools at every decode step**, even though the expensive main attention that follows reads a bounded `K` positions regardless of `T`. As context grows, indexer scoring cost grows with it — linearly in `T`, even while the main attention stays flat.

```text
   ┌────────────────────────────────────────────────────────────────────┐
   │  A nearly flat decode-latency curve measured across a moderate      │
   │  range of context lengths can be a REAL result — the indexer term   │
   │  may simply be small relative to the other terms at that range.     │
   │  It does NOT prove the whole algorithm is O(1) in context length.   │
   │  Measure the indexer term specifically, at your actual maximum      │
   │  deployed context, before claiming context-independence.            │
   └────────────────────────────────────────────────────────────────────┘
```

For prefill, the situation is sharper still: scoring an *expanding* prefix for every one of many simultaneous queries can retain a genuinely quadratic-in-length indexer component — the same `T/4`-candidates-per-query cost, now multiplied across `T` queries instead of one — even though the main, expensive attention pass remains sparse throughout.

---

## Checkpoint

You should now be able to:

1. State why the indexer's 32 heads × 128 dim are a separate configuration from the 64 main heads × 256 dim, and why conflating them is a category error.
2. Derive the 2,051-slot maximum from a 2,048 budget, pool width 4, and a 3-token tail.
3. State the pool-completion causality rule and construct a test sequence that would expose a violation of it.
4. Name the six boundary lengths worth testing and explain what each one probes.
5. Explain, in one sentence, why pool width does not determine main-cache size.
6. Derive the three-term decode cost decomposition and explain which term breaks a "fixed top-k ⇒ O(1)" assumption.

---

## Ship it

This is **Stage 5 of the [capstone ladder](Lecture-12.md)**: build a **KPool boundary test suite** covering: (1) every sequence length in §3's watch list, with and without padding; (2) a packed-document batch with a pool deliberately spanning a document boundary, verifying no cross-document leakage in either selection or attention; (3) an indexer-cost measurement across a range of context lengths, isolating the `O(H_I d_I T/4)` term from the main attention's cost, to check the linear-growth prediction from §5 against your actual implementation.

---

## Current as of

* **Timeless:** the selection/attention separation that defines DeepSeek Sparse Attention generally, the pool-completion causality argument, the three-term decode cost decomposition.
* **Checkpoint-specific:** 32 indexer heads, dimension 128, pool width 4, budget 2,048 (⇒ 2,051-slot buffers) are this checkpoint's configuration — re-derive the slot count if any of these change in a future revision.

---

**Next:** [Module 07 — mHC: Manifold-Constrained Residual Streams →](Lecture-07.md)
