# Module 03 — KDA I: The Delta-Rule Recurrence

**Collection:** [GLM-5.3-Flash Architecture Mastery](README.md) | **Previous:** [← Module 02](Lecture-02.md) | **Next:** [Module 04 →](Lecture-04.md)

---

Kimi Delta Attention is the mechanism this course asks you to master most deeply, and the reason is structural: it is the one mechanism where a subtly wrong implementation still produces fluent-looking text, still passes a casual smoke test, and is still a different model. Getting the update rule exactly right — including the order of two operations that do **not** commute — is the entire module.

---

## Learning objectives

By the end of this module you should be able to:

1. State what KDA's state matrix represents, and what it explicitly is not.
2. Derive the five-step update and expand it into the closed-form recurrence, from memory.
3. Explain, using the algebra, why the decay and correction operations cannot be reordered.
4. Reproduce a worked numeric example of one update step by hand.
5. Explain why the state is per-request serving data, not a checkpoint parameter.

---

## 1. What the state matrix stores

For one attention head, define query, key, and value vectors and an evolving state matrix:

```text
   q_t, k_t  ∈  R^{d_k}          key/query space
   v_t       ∈  R^{d_v}          value space
   S_t       ∈  R^{d_k × d_v}    the state — an associative memory, NOT a list of past (k, v) pairs
```

GLM-5.3-Flash configures **64 KDA heads with head dimension 128**, so each head maintains a `128 × 128` state matrix. Throughout this module, treat `q_t` and `k_t` as already carrying whatever normalization and query scaling the model applies before the recurrence — the update below is the recurrence itself, not the full sublayer (that is [Module 04](Lecture-04.md)'s job).

The critical framing: `S_t` is a **compressed, fixed-size summary** of everything seen so far, continuously overwritten — the opposite of a KV cache, which grows by appending and never overwrites. This is the same "recurrent state replaces a growing cache" trade that Mamba/SSM architectures make ([MLSys Deep Dives — Lecture 04](../MLSys%20Deep%20Dives/Lecture-04.md) works the general version of this argument with concrete state-vs-KV-cache numbers); KDA is one specific, delta-rule instance of that family.

---

## 2. The five-step update

Let `D_t = Diag(α_t)` supply a **separate decay factor for each key channel** — not one scalar decay for the whole state, which is what distinguishes KDA from a coarser gated linear attention variant. Then one step reads as five named operations:

```text
   S̃_t  =  D_t · S_{t−1}              (1) FORGET SELECTIVELY — per-channel decay
   v̂_t  =  S̃_tᵀ · k_t                 (2) PREDICT — what the memory currently says v_t should be
   e_t  =  v_t − v̂_t                  (3) CORRECT — the prediction error
   S_t  =  S̃_t + β_t · k_t · e_tᵀ     (4) WRITE — the correction, scaled by a write gate β_t
   o_t  =  S_tᵀ · q_t                 (5) READ — the updated memory, queried
```

Read step (2)–(4) as a sentence, because it is the conceptual heart of the delta rule: **"what does my memory already predict for this key, and what remains to be corrected?"** An ordinary additive linear-attention memory would instead do `S_t = S_{t-1} + k_t v_tᵀ` — blindly accumulating associations forever, with no mechanism to notice or fix a stale one. KDA's write is proportional to the *error*, not to the raw value, which is what lets a single state matrix track a changing world instead of just averaging over its entire history.

---

## 3. Expanding to the closed form — and why order matters

Substitute (1)–(3) into (4) and expand fully:

```text
   S_t  =  S̃_t + β_t · k_t · e_tᵀ
        =  S̃_t + β_t · k_t · (v_t − v̂_t)ᵀ                         [substitute e_t]
        =  S̃_t + β_t · k_t · v_tᵀ  −  β_t · k_t · v̂_tᵀ             [distribute]
```

Now expand `v̂_tᵀ`. Since `v̂_t = S̃_tᵀ k_t`, its transpose is `v̂_tᵀ = k_tᵀ S̃_t` — a standard transpose-of-a-product flip:

```text
   S_t  =  S̃_t + β_t · k_t · v_tᵀ  −  β_t · k_t · (k_tᵀ S̃_t)
        =  S̃_t  −  β_t · k_t · k_tᵀ · S̃_t  +  β_t · k_t · v_tᵀ
        =  (I − β_t · k_t · k_tᵀ) · S̃_t  +  β_t · k_t · v_tᵀ
```

Substituting `S̃_t = D_t · S_{t-1}` gives the closed form:

```text
   ┌──────────────────────────────────────────────────────────────────────┐
   │                                                                        │
   │   S_t  =  (I − β_t · k_t · k_tᵀ) · D_t · S_{t−1}  +  β_t · k_t · v_tᵀ  │
   │                                                                        │
   └──────────────────────────────────────────────────────────────────────┘
```

This is the KDA recurrence in the order an implementation actually applies it. Now the warning that this whole derivation exists to justify:

```text
   IN GENERAL:      (I − β k kᵀ) · D     ≠     D · (I − β k kᵀ)

   because k depends on t and D_t is diagonal but NOT a multiple of the identity —
   the two matrices do not commute except in special cases.
```

Both expressions look almost identical on the page, and a re-implementation that swaps their order will compile, run, and produce a state matrix that is subtly wrong from the very first token with nonzero decay and a nonzero correction — an error that compounds silently across every subsequent step, because each `S_t` feeds the next. This is precisely why [Module 04](Lecture-04.md) requires validating a chunked implementation against **both output and final state**, not output alone: an order-of-operations bug can leave short-sequence outputs looking nearly correct while the carried state has already diverged.

---

## 4. A worked example, by hand

Take decay disabled (`D_t = I`, so `S̃_t = S_{t-1}`), a `2×2` state, and:

```text
   S_{t−1}  =  [ 2  0 ]         k_t  =  [ 1 ]         v_t  =  [ 6 ]         β_t = 1/2
              [ 0  4 ]                  [ 0 ]                 [ 1 ]
```

**Step 2 — predict.** `v̂_t = S̃_tᵀ k_t`. Since `S_{t-1}` is diagonal, its transpose equals itself:

```text
   v̂_t  =  [ 2  0 ] [ 1 ]  =  [ 2 ]
           [ 0  4 ] [ 0 ]     [ 0 ]
```

**Step 3 — correct.**

```text
   e_t  =  v_t − v̂_t  =  [ 6 ] − [ 2 ]  =  [ 4 ]
                          [ 1 ]   [ 0 ]     [ 1 ]
```

**Step 4 — write.** `S_t = S̃_t + β_t · k_t · e_tᵀ`:

```text
   k_t · e_tᵀ  =  [ 1 ] [ 4  1 ]  =  [ 4  1 ]
                  [ 0 ]              [ 0  0 ]

   β_t · k_t · e_tᵀ  =  [ 2   0.5 ]
                        [ 0    0  ]

   S_t  =  [ 2  0 ] + [ 2   0.5 ]  =  [ 4   0.5 ]
           [ 0  4 ]   [ 0    0  ]     [ 0    4  ]
```

```text
   ┌──────────────────────────────────────────────────────────────┐
   │   S_t  =  [ 4   0.5 ]                                        │
   │           [ 0    4  ]                                        │
   │                                                                │
   │   The association touched by k_t moved HALFWAY toward the     │
   │   requested value (4 → nearly the corrected direction, with    │
   │   an off-diagonal term of 0.5 appearing where k_t and e_t      │
   │   interact). The unrelated row (second row) is untouched —     │
   │   k_t = [1, 0] never reads or writes it.                       │
   └──────────────────────────────────────────────────────────────┘
```

Reproduce this exact example in code as your first correctness check before touching a real model — if your five-line implementation doesn't produce this `S_t`, do not proceed to [Module 04](Lecture-04.md).

---

## 5. State vs. checkpoint weights — a distinction that matters for serving

```text
   CHECKPOINT WEIGHTS                        RECURRENT STATE  S_t
   ─────────────────────────                 ──────────────────────────────
   W_q, W_k, W_v, decay/write gate params     one S_t PER (request, head, layer)
   fixed after training                       changes every token, every request
   shared across every request                belongs to a specific sequence prefix
   loaded once                                allocated, cached, evicted per-request
```

`S_t` is **request-scoped serving state**, exactly analogous to a conventional KV cache slot — not a training update to the model. This distinction has an immediate, practical consequence for anyone building a serving system: `S_t` must be tracked, cached across a multi-turn conversation's turns, correctly evicted, and correctly copied or reconstructed on any operation that duplicates or forks a request (speculative rollback, prefix sharing, request migration). Treating it as ephemeral scratch space that "doesn't matter once the forward pass returns" is a serving bug waiting to happen — [Module 08](Lecture-08.md) covers exactly this failure mode for speculative decoding rollback, and [Module 09](Lecture-09.md) sizes the VRAM budget this state requires across a full multi-GPU deployment.

---

## Checkpoint

You should now be able to:

1. State what `S_t` represents and why it is not a list of past key/value pairs.
2. Write the five-step update from memory and expand it into the closed-form recurrence.
3. Explain, using the transpose-of-a-product identity, exactly where the closed form comes from.
4. State the non-commutativity warning and describe a concrete bug it predicts.
5. Reproduce the worked numeric example without referring back to this page.
6. Explain why `S_t` must be treated as per-request serving state rather than transient scratch memory.

---

## Ship it

This is **Stage 2 and the first milestone of the [capstone ladder](Lecture-12.md)**: implement the five-step update as a small FP32 reference (NumPy is sufficient — no GPU, no batching, one head) and reproduce §4's worked example exactly. Then process a short random sequence step by step and print the resulting `S_t` at each step. Do not proceed to [Module 04](Lecture-04.md) until this reference implementation exists and matches the hand-worked example bit for bit (up to floating-point rounding).

---

## Current as of

* **Timeless:** the five-step update, its closed-form expansion, the non-commutativity of the decay and correction operators, and the delta-rule motivation (write the error, not the raw value) — this is the KDA recurrence from the underlying delta-rule/KDA formulation, expressed here in an implementation-friendly operation order.
* **Checkpoint-specific:** 64 heads, head dimension 128 (so a `128×128` state per head) are properties of this checkpoint's configuration.

---

**Next:** [Module 04 — KDA II: Chunked Parallelism & the Full Sublayer →](Lecture-04.md)
