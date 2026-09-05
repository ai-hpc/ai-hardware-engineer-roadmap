# Module 11 — Correctness as Architecture Mastery

**Collection:** [GLM-5.3-Flash Architecture Mastery](README.md) | **Previous:** [← Module 10](Lecture-10.md) | **Next:** [Module 12 →](Lecture-12.md)

---

Every module in this course has flagged a specific way to implement its mechanism plausibly and wrong — the router's correction bias, the KDA operator ordering, MLA's softmax scale, DSA's pool-completion causality, mHC's approximate doubly-stochastic tolerance, hybrid rollback state. This module collects them into one discipline: **define the contract a change must preserve before you measure whether it's fast.**

---

## Learning objectives

By the end of this module you should be able to:

1. Distinguish a same-checkpoint kernel replacement from a new quantization as two different kinds of experiment with two different required evaluations.
2. Build the full per-mechanism correctness test matrix for this architecture.
3. State the prefix/chunk/continuation invariant and explain what class of bug it catches that per-token output comparison misses.
4. Specify a numerical contract before benchmarking, and explain why both "looks reasonable" and "must be bitwise identical" are usually the wrong bar.

---

## 1. Two different experiments, two different evaluations

```text
   SAME-CHECKPOINT KERNEL REPLACEMENT           NEW QUANTIZATION / NEW PRECISION
   ──────────────────────────────────           ──────────────────────────────────
   same weights, same math, different            different NUMBERS — a deliberate,
   CODE PATH computing it                        bounded change to the computation

   compare against the EXISTING computation       needs everything the left column
   with a numerical-tolerance check                needs, PLUS model-quality evaluation
   (Module 04's output+state agreement,            against the higher-precision reference
   Module 05's expanded-vs-absorbed check)          (Hardware-Aware LLM Quantization —
                                                     Module 08's KL/acceptance-length gate)
```

Treating these as one kind of test is a common shortcut with a specific failure mode: a kernel-replacement test suite that only checks "does this look like reasonable text" will happily pass a change that is algebraically wrong (wrong softmax scale, swapped operator order) as long as the wrongness is subtle enough not to break fluency — exactly the class of bug this course has repeatedly named as the dangerous one, because fluent output is not evidence of a correct implementation.

---

## 2. The full correctness matrix

| Area | Required tests |
|---|---|
| **MoE** | Selected expert IDs match reference; mixture weights use the *raw sigmoid score*, not score-plus-bias ([Module 02 §2](Lecture-02.md)); gate clamp is upper-only, up-projection clamp is two-sided ([Module 02 §3](Lecture-02.md)) |
| **KDA** | Recurrent and chunked execution agree on **both output and final state** ([Module 04 §4](Lecture-04.md)); operator order matches `(I − βkkᵀ)D`, not `D(I − βkkᵀ)` ([Module 03 §3](Lecture-03.md)) |
| **KPool** | Pool-boundary lengths (3, 4, 5, 7, 8, 9 — [Module 06 §3](Lecture-06.md)) and incomplete-tail handling produce correct selections; buffers sized to 2,051 slots, not 2,048 |
| **Padding / packing** | No cross-document state leakage in KDA's recurrence, MLA's cache, or DSA's pooling across packed-sequence boundaries |
| **MLA** | Expanded-form and absorbed-form attention produce identical scores (not merely proportional ones) on identical inputs — the softmax scale must not silently move ([Module 05 §3](Lecture-05.md)) |
| **mHC** | Stream layout, collapse/mix orientation, and normalization order match reference; `R`'s row/column sums fall within the *actual* Sinkhorn-iteration tolerance used, not exact 1.0 ([Module 07 §2](Lecture-07.md)) |
| **Serving state** | Prefix-cache hits, request reordering, branching, cancellation, and speculative rollback all reconcile KDA state, convolution state, MLA cache, and DSA metadata consistently ([Module 08 §3](Lecture-08.md)) |
| **Precision** | Long-context accumulation behavior, extreme-gate values (the ±10 clamps — [Module 02 §3](Lecture-02.md)), and near-tied routing scores are covered, not just typical-case inputs |

Every row of this table is a direct callback to a specific derivation earlier in the course — this matrix is not a generic checklist, it is what falls out of having actually derived each mechanism instead of only reading about it.

---

## 3. The invariant that generalizes across every mechanism

One property should hold regardless of *how* a prefix was processed:

```text
   ┌────────────────────────────────────────────────────────────────────┐
   │   Processing the same prefix in ONE PREFILL PASS, in MULTIPLE       │
   │   CHUNKS, or through a CACHED CONTINUATION (prefix reuse across      │
   │   turns) should lead to EQUIVALENT continuation state and logits,    │
   │   within a stated tolerance.                                         │
   └────────────────────────────────────────────────────────────────────┘
```

This single invariant subsumes several of this course's mechanism-specific checks at once:

```text
   ONE PREFILL vs. CHUNKED PREFILL       →  tests KDA's chunk composition (Module 04)
                                             and DSA's pool-boundary handling (Module 06)

   CACHED CONTINUATION vs. FULL REPLAY   →  tests whether prefix-cache reuse correctly
                                             restored EVERY piece of hybrid state (Module 08),
                                             not just the token-indexed MLA cache

   ANY OF THE ABOVE, POST-ROLLBACK       →  tests the rollback reconciliation itself
                                             (Module 08 §3) against a from-scratch run
```

A test suite built around this one invariant, exercised at the boundary lengths from [Module 06 §3](Lecture-06.md) and across the state inventory from [Module 08 §3](Lecture-08.md), catches a large fraction of the bugs this course has named as individual, mechanism-specific risks — because most of them are, underneath, the same failure: **some piece of state didn't survive a code path that was supposed to be equivalent to the reference path.**

---

## 4. Specify the contract before you benchmark

Neither extreme is the right default:

```text
   "the output looks reasonable"     →  too weak. Catches nothing this course has
                                          flagged — a wrong router weighting, a
                                          swapped operator order, and a mis-scaled
                                          softmax can ALL still look reasonable.

   "must be bitwise identical"       →  too strong, for the wrong reason. Algebraically
                                          EQUIVALENT floating-point reductions (e.g.
                                          summing in a different order, Module 05's
                                          expanded vs. absorbed forms) are not
                                          bitwise identical and should not be demanded
                                          to be — that bar rejects correct code.
```

The right move is to **state the tolerance and the metric before running the comparison**, matched to what kind of change you're testing:

```text
   kernel replacement, same checkpoint   :  numerical tolerance on outputs AND state
                                             (Module 04 §4's exact test design)

   new quantization / precision change   :  KL divergence + top-1 agreement against
                                             the higher-precision reference, exactly as
                                             Hardware-Aware LLM Quantization — Module 08
                                             specifies, PLUS this course's state-agreement
                                             checks — a quantization change to this
                                             architecture needs BOTH evaluations, not
                                             either one alone
```

Committing to the contract *before* seeing results is what makes the subsequent comparison an experiment rather than a rationalization — precisely the discipline [Hardware-Aware LLM Quantization — Module 12](../Hardware-Aware%20LLM%20Quantization/Lecture-12.md) argues for generally, applied here to a model where "does it still work" has five separate mechanisms that each need their own answer to that question.

---

## Checkpoint

You should now be able to:

1. Explain why a kernel-replacement test and a quantization-evaluation test are different experiments requiring different evidence.
2. Recite the correctness matrix's eight areas and, for each, name the specific failure mode it exists to catch.
3. State the prefix/chunk/continuation invariant and explain which three testing scenarios it subsumes.
4. Explain why both "looks reasonable" and "bitwise identical" are usually the wrong bar, and state what belongs between them.

---

## Ship it

Build the **full correctness matrix as an executable test suite** — one test group per row of §2's table — and add the prefix/chunk/continuation invariant from §3 as a cross-cutting test run against every mechanism's state, not just MLA's. For each test, write down the tolerance and metric *before* running it, per §4's discipline. This suite is what [Module 12](Lecture-12.md)'s capstone gates every optimization against — nothing ships past this course's discipline without passing it first.

---

## Current as of

* **Timeless:** the kernel-replacement-vs-quantization distinction, the correctness matrix (each row derived from an earlier module's specific finding), the prefix/chunk/continuation invariant, and the discipline of specifying a numerical contract before benchmarking.
* **Checkpoint-specific:** the exact clamp values, tolerance thresholds, and boundary lengths cited in the matrix trace back to this checkpoint's configuration — re-derive them from the corresponding earlier module if you are applying this matrix to a different revision.

---

**Next:** [Module 12 — Capstone: The KDA-First Mastery Ladder →](Lecture-12.md)
