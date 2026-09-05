# Module 08 — Vision, MTP, and Hybrid Serving State

**Collection:** [GLM-5.3-Flash Architecture Mastery](README.md) | **Previous:** [← Module 07](Lecture-07.md) | **Next:** [Module 09 →](Lecture-09.md)

---

Two remaining pieces of the system are easy to under-scope precisely because they look like small additions to a diagram: a vision encoder bolted onto the front, and one auxiliary prediction head bolted onto the back. Both are simple to describe and each introduces a serving-correctness obligation that a text-only, non-speculative mental model does not prepare you for.

---

## Learning objectives

By the end of this module you should be able to:

1. Explain why vision and language processing must be timed as separate stages, and predict a specific profiling error that results from not doing so.
2. State what a single MTP auxiliary layer does and does not establish about serving speedup.
3. Enumerate every piece of per-request state a speculative rollback must restore for this specific hybrid architecture, and explain why "decrease the KV length" is insufficient.

---

## 1. Vision is a separate subsystem

GLM-5.3-Flash is multimodal: image patches are encoded, processed through a vision transformer, and merged/projected into the language model's hidden width before joining the four residual streams described in [Module 01 §3](Lecture-01.md). The important operational fact is that this is architecturally **separate** from the language decoder — a different set of weights, a different computational shape (patches, not tokens), running before the language model's own forward pass begins on the resulting features.

```text
   image ──▶ patchify ──▶ vision transformer ──▶ project to d_model=4096 ──▶ joins
                                                                              the four
   text  ─────────────────────────────────────────────────────────────────▶ residual
                                                                              streams
```

### The profiling consequence

If you collapse "how long did this request take" into a single number, a vision-heavy request and a text-only request become incomparable, and a genuinely effective text-decoder optimization can appear to fail:

```text
   request A:  text-only, 500 tokens                    total = T_prefill + T_decode
   request B:  one image + 500 tokens of text            total = T_preprocess + T_vision + T_prefill + T_decode

   You ship a 30% faster KDA kernel. It improves T_decode specifically.

   Measuring only TOTAL time:
     request A: total drops noticeably     (T_decode was a large share of the total)
     request B: total barely moves          (T_preprocess + T_vision dominate; T_decode
                                              was always a small slice of this request's time)

   Conclusion a careless benchmark draws: "the KDA optimization doesn't help
   multimodal requests." WRONG — it helped exactly as much as it should have;
   the request's bottleneck was simply somewhere else.
```

```text
   ┌────────────────────────────────────────────────────────────────────┐
   │   Keep at least THREE timings distinct in any profiling setup      │
   │   that touches multimodal requests:                                 │
   │                                                                      │
   │        image preprocessing   │   vision encoding   │   LM prefill    │
   │                                                                      │
   │   Collapsing these into one number makes it impossible to tell      │
   │   whether an optimization worked or was simply irrelevant to        │
   │   this request's actual bottleneck.                                 │
   └────────────────────────────────────────────────────────────────────┘
```

This is the same discipline [Module 10](Lecture-10.md) applies to every other region of the model — never optimize, or judge an optimization, against an aggregate number that mixes stages with different bottlenecks.

---

## 2. One MTP layer is not a serving speedup by itself

The checkpoint configures **one next-token-prediction auxiliary layer** — a multi-token-prediction (MTP) head trained alongside the main model. That fact, on its own, establishes only that the *training recipe* included this auxiliary objective. It does **not** by itself establish that serving is faster, because speculative serving needs several additional pieces working together, none of which come for free just because a checkpoint has a draft head:

```text
   a checkpoint HAS an MTP head          ⇏      serving IS faster

   speculative serving additionally needs:

     1.  a PROPOSAL mechanism            (the MTP head generates draft continuations)
     2.  TARGET VERIFICATION              (the full model checks the drafts)
     3.  an ACCEPTANCE RULE                (which drafted tokens get kept)
     4.  CORRECT STATE HANDLING            (§3 — the hard part for THIS architecture)
```

And even with all four correctly implemented, the benefit is not a function of "how many tokens get proposed" — it is a function of **accepted progress relative to the cost of drafting and verifying**, exactly the accept-rate-versus-overhead economics worked out in full in [Hardware-Aware LLM Quantization — Module 10](../Hardware-Aware%20LLM%20Quantization/Lecture-10.md). A drafter that proposes aggressively but gets rejected often can make serving *slower*, not faster — the presence of an MTP head in the checkpoint says nothing about which side of that line a given deployment lands on.

---

## 3. Rollback: the state this architecture actually carries

This is the module's central point, and it follows directly from work already done in this course. A conventional transformer's speculative-decoding rollback is comparatively simple: on a rejected draft token, truncate the KV cache back to the last accepted position, and continue. **That operation is insufficient for this architecture**, because this model carries several other kinds of per-request state that a KV-length truncation does not touch at all.

Recall what's actually accumulating, request-scoped, across the hybrid stack:

```text
   MECHANISM              STATE THAT MUST BE RECONCILED ON ROLLBACK
   ──────────────────      ────────────────────────────────────────────────────
   KDA (Modules 03–04)    the recurrent state matrix S_t, per head, per KDA layer —
                          NOT indexed by token position at all. You cannot "truncate"
                          it the way you truncate a list; a rejected token means the
                          state update it caused must be UNDONE or RECOMPUTED from a
                          checkpoint taken before that update.

   short causal            convolution state (Module 04 §3) — a small sliding window
   convolution              of recent inputs feeding the convolution. Also not simply
                             "shorter" after rollback; it must reflect exactly the
                             accepted prefix, no more.

   MLA (Module 05)         the token-indexed latent cache c_t — THIS one genuinely is
                          truncatable by position, much like an ordinary KV cache.

   DSA/KPool (Module 06)  index/selection metadata built over the (now-rolled-back)
                          history — must be rebuilt or truncated consistently with
                          the new, shorter accepted prefix, respecting the same
                          pool-completion causality rule from Module 06 §3.
```

```text
   ┌────────────────────────────────────────────────────────────────────┐
   │   "Rollback" for this model means: restore or recompute the KDA     │
   │   recurrent state AND the convolution state to the point matching   │
   │   the last accepted token, AND truncate the MLA latent cache to      │
   │   that same point, AND reconcile the DSA indexer's selection         │
   │   metadata against the new prefix length.                            │
   │                                                                        │
   │   Treating rollback as "decrease the KV length" — correct for a       │
   │   conventional transformer — silently leaves the KDA and             │
   │   convolution state pointing at a history that includes REJECTED     │
   │   tokens. Every subsequent token generated from that corrupted        │
   │   state is wrong, and nothing about the symptom will look like an     │
   │   obvious crash — it will look like a model that's slightly, then     │
   │   increasingly, incoherent.                                           │
   └────────────────────────────────────────────────────────────────────┘
```

Two practical designs close this gap, and a real system typically needs some combination of both:

```text
   CHECKPOINT-AND-RESTORE:  snapshot S_t (and convolution state) before each
                            speculative verification step; on rejection, restore
                            the snapshot exactly. Costs memory proportional to
                            how many speculative steps you checkpoint.

   RECOMPUTE:               on rejection, recompute S_t from the last known-good
                            checkpoint forward through the accepted tokens only,
                            using the ordinary recurrent update (Module 03).
                            Costs compute instead of memory.
```

Either way, the correctness bar is the same one [Module 04 §4](Lecture-04.md) already established for chunked-versus-recurrent execution: **state after rollback must agree with state from a request that had simply never generated the rejected tokens in the first place** — not merely "close," and not verified by output plausibility alone.

---

## Checkpoint

You should now be able to:

1. Explain why a vision-heavy and a text-only request are not comparable under one aggregate latency number, with a concrete example of the wrong conclusion that follows from ignoring this.
2. List the three timings a multimodal profiling setup must keep separate.
3. State the four components speculative serving needs beyond "the checkpoint has an MTP head."
4. Enumerate, for this specific architecture, every kind of state a correct rollback must reconcile — not just the KV/latent cache.
5. Explain why "decrease the KV length" is a correct rollback strategy for a conventional transformer but an incomplete one here.

---

## Ship it

Design (and, where feasible, implement against a small test harness) a **rollback correctness test**: generate a sequence with speculative decoding enabled, force a rejection at a chosen position, and verify that the post-rollback KDA state, convolution state, and MLA cache each exactly match the corresponding state from an independent run that generated only the accepted prefix directly — the "never generated the rejected tokens" invariant from §3, made executable.

---

## Current as of

* **Timeless:** the profiling discipline for multi-stage multimodal requests, the general requirements for a working speculative-decoding system, and the state-reconciliation argument for hybrid recurrent/token-indexed architectures under rollback.
* **Checkpoint-specific:** the presence of exactly one MTP auxiliary layer, and the specific multimodal encoder/merge design, are properties of this checkpoint — verify against the actual serving implementation before assuming a different revision carries the same speculative-decoding support.

---

**Next:** [Module 09 — The 8-GPU Memory Model →](Lecture-09.md)
