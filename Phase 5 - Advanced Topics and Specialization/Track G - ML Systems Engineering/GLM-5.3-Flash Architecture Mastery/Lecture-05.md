# Module 05 — MLA: Compressing Per-Token History

**Collection:** [GLM-5.3-Flash Architecture Mastery](README.md) | **Previous:** [← Module 04](Lecture-04.md) | **Next:** [Module 06 →](Lecture-06.md)

---

KDA compresses an entire history into one fixed-size state matrix, discarding token identity entirely. MLA — used in this checkpoint's 11 sparse attention layers — does something different: it keeps history **token-indexed** (you can still ask "what did token 47 look like?"), but shrinks what has to be stored per token by an order of magnitude. This module derives that shrinkage, and the algebraic trick that lets attention operate directly on the compressed form without ever re-expanding it.

---

## Learning objectives

By the end of this module you should be able to:

1. State what MLA caches instead of expanded per-head keys and values, and why.
2. Derive the memory ratio between a fully-expanded cache and the cached latent, from the checkpoint's dimensions.
3. Derive the absorption identity that lets attention read the latent directly, and state exactly what it does and does not change.
4. Explain why this checkpoint's NoPE configuration does not make the model order-blind.
5. Explain why the fastest algebraic form differs between prefill and decode.

---

## 1. The latent representation

Rather than caching every head's expanded key and value vectors, MLA projects each token down to one shared low-rank latent:

```text
   c_t  =  RMSNorm( W_D · x_t )              c_t  ∈  R^r          (the cached quantity)
```

Per-head keys and values are then **reconstructed from the latent, on demand**, by per-head up-projection matrices:

```text
   k_{t,h}  =  U_h^K · c_t
   v_{t,h}  =  U_h^V · c_t
```

The point: `c_t` is what gets cached and persisted across the sequence. `k_{t,h}` and `v_{t,h}` are computed *from* it when needed, rather than stored directly. This checkpoint configures a **KV latent width of 512**, **64 main attention heads**, and **256-dimensional main key/query and value heads**, with the main MLA path configured as **NoPE** (zero rotary dimensions on the main query/key — §4 explains why this doesn't mean what it sounds like).

---

## 2. The memory advantage, derived

Compare what a fully-expanded BF16 cache would need against what the shared latent actually needs, per token, per layer.

**Fully expanded** (hypothetical — this is *not* what gets cached): 64 heads, each storing a 256-dim key and a 256-dim value, at 2 bytes (BF16):

```text
   64 heads × (256 + 256) dims × 2 bytes  =  65,536 bytes / token / layer
```

**Shared latent** (what actually gets cached): 512-dim latent at 2 bytes:

```text
   512 × 2  =  1,024 bytes / token / layer
```

```text
   ┌────────────────────────────────────────────────────────────────┐
   │                     65,536 / 1,024  =  64×                      │
   │                                                                    │
   │   This is a 64× reduction for THESE specific logical              │
   │   representations, on THIS checkpoint's dimensions. It is NOT     │
   │   a claim that total model memory or end-to-end inference speed   │
   │   improves by 64× — weights, the DSA indexer, workspace buffers,   │
   │   and communication overhead are all separate terms that don't    │
   │   shrink because this one term did.                                │
   └────────────────────────────────────────────────────────────────┘
```

[Module 09](Lecture-09.md) uses this exact 512-wide, BF16, per-layer figure to build the full per-request cache budget across all 11 MLA layers and a stated context length — this module only establishes the per-token, per-layer unit.

---

## 3. The algebra that avoids re-expanding history

The memory win in §2 would be worthless if computing attention still required materializing every historical token's full-size `k_{t,h}` and `v_{t,h}` from the cached latents at every step — you'd be trading a smaller cache for a larger recompute. MLA avoids this with an algebraic reshuffling.

**For the score.** A query-key dot product against a cached historical token `j`:

```text
   q_hᵀ · k_{j,h}   =   q_hᵀ · (U_h^K c_j)                      [substitute k_{j,h}]
                    =   (q_hᵀ U_h^K) · c_j                       [associativity]
                    =   ( (U_h^K)ᵀ q_h )ᵀ · c_j                  [transpose identity]
```

Define `q̃_h = (U_h^K)ᵀ q_h` — computed **once per query, per head**, independent of which historical token `j` you're scoring against. Then every score against the cache becomes a direct dot product with the cached latent:

```text
   q_hᵀ · k_{j,h}   =   q̃_hᵀ · c_j            ← operates DIRECTLY on the cached latent
                                                  no k_{j,h} ever materialized
```

**For the output.** The attention-weighted sum over values has the same structure, and linearity lets the fixed matrix factor outside the sum over history:

```text
   Σ_j  p_j · v_{j,h}   =   Σ_j  p_j · (U_h^V c_j)                 [substitute v_{j,h}]
                        =   U_h^V · ( Σ_j  p_j · c_j )              [U_h^V doesn't depend on j — factors out]
```

So the weighted accumulation happens **in latent space** (summing `p_j · c_j` over history), and only the *final* accumulated result gets expanded through `U_h^V` — once, not once per historical token.

```text
   ┌──────────────────────────────────────────────────────────────────┐
   │   Neither derivation requires ever reconstructing a historical    │
   │   token's full-size key or value. Scores read the latent          │
   │   directly via a pre-transformed query; the value sum accumulates │
   │   in latent space and expands exactly once, at the end.            │
   └──────────────────────────────────────────────────────────────────┘
```

### The one thing this algebra must never change

```text
   Rewriting HOW a dot product is computed does not authorize
   rewriting WHAT it's compared against.

   The softmax temperature / attention scale is a property of the
   ORIGINAL q·k formulation. Moving to the absorbed (latent-space)
   form must reproduce IDENTICAL score values — not merely
   proportional ones — or you have changed the model's attention
   distribution while believing you only changed its data layout.
```

This is exactly the class of bug [Module 11](Lecture-11.md)'s correctness matrix names as "expanded-versus-absorbed attention equivalence," and it deserves a dedicated numerical test, not an assumption: compute both forms on identical inputs and require the scores to match to floating-point tolerance, not merely to produce similar-looking downstream text.

---

## 4. NoPE does not mean order-blind

This checkpoint's main MLA path uses **NoPE** — zero rotary position dimensions on the main query/key. It is tempting to read this as "the model can't tell token order in these layers," which is wrong, for two independent reasons:

```text
   1. GLM-5.3-Flash is a HYBRID model. 34 of its 45 layers are KDA, and
      KDA's recurrence is INHERENTLY order-sensitive — S_t depends on the
      exact sequence of updates that produced it. A model with 34
      order-sensitive layers is not rendered order-blind by 11 layers
      that individually lack rotary position encoding.

   2. Within the MLA layers themselves, CAUSAL MASKING already restricts
      which positions a query can attend to (only j ≤ t). That alone
      encodes a coarse notion of order — "before vs. after" — even
      without a positional embedding encoding exact relative distance.
```

What NoPE *does* remove, specifically, is fine-grained relative-distance information *within* what causal masking already permits — the kind of signal RoPE would otherwise inject directly into the dot product. Whether that matters for a given downstream task is an empirical question about *this specific* layer's role in the hybrid stack, not something you can conclude from the presence or absence of rotary encoding alone.

---

## 5. Prefill and decode may want different algebra

Two facts do not automatically settle a third: neither the memory argument (§2) nor the algebraic equivalence (§3) tells you which of the following two computational strategies is fastest on your hardware, at your batch size:

```text
   NAIVE (per-head expansion)     :  expand k_{j,h}, v_{j,h} for every historical
                                      token, run ordinary multi-head attention
   ABSORBED (latent-space, §3)   :  keep everything in the r-dimensional latent
                                      space, expand only the final accumulated result
```

```text
   PREFILL   :  many queries share the same historical keys/values.
                Expanding once and reusing across queries can make the
                naive form's larger per-step matmuls a good FIT for
                tensor cores — this is a COMPUTE-heavy regime
                (see Hardware-Aware LLM Quantization — Module 01
                 for the general compute-vs-bandwidth framing).

   DECODE    :  one query at a time, against a cache that dominates
                memory traffic. The absorbed form's smaller cache
                directly reduces the dominant cost — this is a
                BANDWIDTH-heavy regime, and it is where the 64×
                reduction from §2 pays off most directly.
```

**Selecting a path by counting FLOPs alone is unreliable** — the right choice depends on which resource (compute or bandwidth) is actually binding, exactly the roofline distinction that governs every optimization decision in the [Hardware-Aware LLM Quantization](../Hardware-Aware%20LLM%20Quantization/README.md) course. [Module 10](Lecture-10.md) returns to this as a profiling hypothesis, not a rule to memorize: measure which regime you're in before assuming which algebraic form wins.

---

## Checkpoint

You should now be able to:

1. State what MLA caches (`c_t`) versus what it reconstructs on demand (`k_{t,h}`, `v_{t,h}`).
2. Derive the 64× memory ratio from this checkpoint's dimensions without looking it up.
3. Derive `q̃_h = (U_h^K)ᵀ q_h` and explain why it lets scoring skip per-token key expansion.
4. Derive the latent-space value-accumulation identity and explain why `U_h^V` factors outside the sum.
5. Explain why NoPE in the main MLA path does not make the whole model order-blind.
6. Explain why prefill and decode can legitimately prefer different algebraic forms of the same attention.

---

## Ship it

This is **Stage 4 of the [capstone ladder](Lecture-12.md)**: build an **MLA equivalence laboratory** — implement both the naive (per-head expansion) and absorbed (latent-space) forms of attention over cached latents, verify their scores and outputs match to floating-point tolerance on identical inputs, then measure wall-clock and memory traffic for both forms at a decode-shaped workload (batch 1, growing cache) and a prefill-shaped workload (many queries, shared history). Report which form wins in which regime, and whether that matches the §5 prediction.

---

## Current as of

* **Timeless:** the latent-caching idea and the score/value absorption algebra — this is the joint low-rank compression and "absorb the up-projection into the query/output" technique from the DeepSeek-MLA line of work, applied here to this checkpoint's specific dimensions.
* **Checkpoint-specific:** KV latent width 512, 64 main heads, 256-dim main K/Q/V heads, NoPE on the main path — verify against the actual checkpoint config before reusing these constants.

---

**Next:** [Module 06 — DSA & KPool: Selective Retrieval →](Lecture-06.md)
