# Module 07 — mHC: Manifold-Constrained Residual Streams

**Collection:** [GLM-5.3-Flash Architecture Mastery](README.md) | **Previous:** [← Module 06](Lecture-06.md) | **Next:** [Module 08 →](Lecture-08.md)

---

Every mechanism so far changed *what gets computed* — which experts run, what gets remembered, what gets read. mHC (manifold-constrained Hyper-Connections) changes something different: **how information flows between sublayers.** It is the mechanism most often misread as multiplying the model's effective depth or width, and this module exists to make precise why it does neither.

---

## Learning objectives

By the end of this module you should be able to:

1. Write the four-stream residual update in collapse/compute/mix form.
2. State the doubly-stochastic constraint on the mixing matrix and explain what it stabilizes.
3. Reproduce a small numeric Sinkhorn-normalization example and explain why the result is only approximately doubly stochastic in practice.
4. Explain precisely why four residual streams does not mean four attention modules, and why 45 layers does not become 45/4 effective layers.

---

## 1. From one residual stream to four

An ordinary transformer residual sublayer is a single running sum:

```text
   x'  =  x  +  F( Norm(x) )
```

With `n = 4` parallel residual streams, the per-token state at any point in the network is not a vector but a small matrix:

```text
   X  ∈  R^{4 × 4096}          — four parallel copies of the residual, side by side
```

A useful way to describe what happens at each sublayer (attention or feed-forward) is three steps — **collapse, compute, mix**:

```text
   x̂  =  Xᵀ · a                          COLLAPSE: combine the 4 streams into ONE vector
                                           before the expensive sublayer runs

   u  =  F( Norm(x̂) )                    COMPUTE: run the actual sublayer (KDA / MLA / FFN)
                                           ONCE, on the collapsed vector — not four times

   X'  =  R · X  +  b · uᵀ               MIX: distribute u back across the streams (b),
                                           and mix the streams among themselves (R)
```

`a` and `b` are learned combination/distribution vectors; `R` is a learned residual-mixing matrix. All three can depend on the incoming state — this is what makes the connection pattern *learned* rather than fixed, which is the generalization Hyper-Connections-style architectures make over a plain residual sum.

```text
   ┌─────────────────────────────────────────────────────────────────┐
   │   THE EXPENSIVE SUBLAYER (attention or FFN) RUNS ONCE PER LAYER,  │
   │   ON A COLLAPSED VECTOR — regardless of how many residual         │
   │   streams exist. Widening the residual state does not widen       │
   │   how many times the costly computation executes.                 │
   └─────────────────────────────────────────────────────────────────┘
```

---

## 2. The manifold constraint

"Manifold-constrained" refers specifically to a constraint placed on `R`, the residual-mixing matrix: it is pushed toward being **doubly stochastic** — nonnegative entries, every row summing to 1, every column summing to 1:

```text
   R · 1  ≈  1              (every row sums to ~1 — each output stream is a weighted
                              AVERAGE of input streams, not an uncontrolled blend)

   1ᵀ · R  ≈  1ᵀ             (every column sums to ~1 — each input stream's
                              contribution is conserved across outputs, not
                              silently amplified or discarded)
```

A doubly stochastic mixing matrix cannot let the residual stream's overall magnitude explode or collapse purely from the *mixing* step — mass is conserved across the mix regardless of how the specific weights are learned. This is a stability property, not an accuracy claim: **it constrains the residual mixing path specifically. It does not prove every operation elsewhere in the network is non-expansive**, and it says nothing about the attention or feed-forward computation `F` itself, which can still amplify or shrink its input arbitrarily.

### Reaching the constraint: Sinkhorn normalization

The standard way to push an arbitrary matrix toward doubly-stochastic is alternating row and column normalization (Sinkhorn–Knopp). A small, generic illustration — not this checkpoint's actual mixing matrix, just the mechanism, on an arbitrary 3×3 starting point:

```text
   START (arbitrary, nonnegative):        AFTER 20 ROUND-TRIPS (row, then col):
      3.0  1.0  0.5                          0.645  0.252  0.103
      0.5  2.0  1.0                          0.131  0.616  0.252
      1.0  0.5  3.0                          0.224  0.131  0.645

                                           row sums: [1.000, 1.000, 1.000]
                                           col sums: [1.000, 1.000, 1.000]
```

```text
   AFTER ONLY 2 ROUND-TRIPS (what a fixed, small iteration budget looks like):
      0.645  0.251  0.104
      0.133  0.619  0.255
      0.222  0.130  0.641

   row sums: [1.000, 1.007, 0.993]    ←  close, but NOT exactly 1
   col sums: [1.000, 1.000, 1.000]
```

```text
   ┌─────────────────────────────────────────────────────────────────┐
   │   A real implementation runs a FIXED, SMALL number of Sinkhorn   │
   │   iterations (for speed) and adds numerical stabilizers.          │
   │   Treat the result as APPROXIMATELY doubly stochastic — the       │
   │   "≈" in R·1 ≈ 1 is load-bearing, not decorative. Do not          │
   │   implement or test this as an exact symbolic projection.         │
   └─────────────────────────────────────────────────────────────────┘
```

For a correctness test, this means: check row and column sums are *close to* 1 within a stated tolerance tied to the actual iteration count used, not that they equal 1 exactly — an exact-equality test against this mechanism will fail on entirely correct code.

---

## 3. Two things mHC does not do

Both follow directly from the collapse-compute-mix structure in §1, and both are worth stating as standalone corrections because they are the most common way this mechanism gets over-interpreted:

```text
   FOUR RESIDUAL STREAMS  ≠  FOUR ATTENTION/FFN MODULES

     The collapse step (x̂ = Xᵀa) reduces four streams to ONE vector
     BEFORE the sublayer runs. F(Norm(x̂)) executes once. There are not
     four parallel KDA computations or four parallel MoE dispatches
     happening because there are four streams — there is one of each,
     per layer, exactly as in a single-stream model.


   45 DECODER LAYERS  ≠  45/4  EFFECTIVE SERIAL LAYERS

     mHC changes CONNECTIVITY between sublayers — how information from
     the residual streams is combined going in and redistributed coming
     out. It does not shorten the DEPENDENCY CHAIN: layer 12's output
     still depends on layer 11's output, which still depends on layer
     10's, all the way down. Nothing about having four streams lets you
     skip layers or run them out of order. The serial depth is 45,
     full stop.
```

The practical cost model this leaves you with: wider residual **activations** (4× the per-position residual memory, since you're carrying `X ∈ R^{4×4096}` instead of a single 4096-vector) is the entire price of mHC, in exchange for the flexibility of learned collapse/distribute/mix behavior. It buys richer information flow between sublayers; it does not buy free parallelism across the decoder's serial depth, and it does not multiply how many times the expensive sublayers execute.

---

## 4. What this means for profiling and implementation

Given §1–3, the mHC-specific execution surface a real implementation adds around every sublayer is exactly three small operations:

```text
   mHC collapse + normalization   (X → x̂)
   [ the sublayer itself: KDA, MLA, or FFN — Modules 02–06 ]
   mHC residual mixing            (u → X')
```

These collapse and mixing operations are comparatively cheap — small matrix-vector products against `a`, `b`, and `R` — relative to the sublayers they surround. But "comparatively cheap per operation" and "negligible in aggregate across 45 layers × 2 sublayers per layer × 2 mHC operations per sublayer" are different claims, and this is exactly the class of cost [Module 10](Lecture-10.md) flags as a specific profiling hypothesis: **small-kernel launch overhead and redundant activation traffic** from many cheap operations can accumulate into a real cost even when no single one of them shows up as expensive in isolation. Do not assume mHC's overhead is negligible; measure it, on this specific implementation, the same way you would measure anything else.

---

## Checkpoint

You should now be able to:

1. Write the collapse/compute/mix equations for a 4-stream residual update from memory.
2. State the doubly-stochastic constraint on `R` and explain what it stabilizes versus what it does not guarantee.
3. Reproduce a small Sinkhorn-normalization example and explain why a fixed small iteration count yields only approximate row/column sums.
4. Give the two-sentence correction for "four streams" and "45 layers" that this module exists to teach.
5. Name the mHC-specific profiling hypothesis from [Module 10](Lecture-10.md) and explain why it applies even though each individual mHC operation is cheap.

---

## Ship it

This is the other half of **Stage 6 of the [capstone ladder](Lecture-12.md)** (paired with [Module 02](Lecture-02.md)'s router audit). Build an **mHC stream-layout and mixing test**: implement the collapse/compute/mix update on a toy multi-stream residual, verify `R`'s row and column sums land within your implementation's actual Sinkhorn-iteration tolerance (not exact 1.0), and verify that the sublayer function `F` is invoked exactly once per layer regardless of stream count — a direct, executable check against the "four streams ≠ four modules" claim in §3.

---

## Current as of

* **Timeless:** the collapse/compute/mix decomposition of a multi-stream residual, the Sinkhorn-normalization mechanism and its approximate (not exact) convergence under a finite iteration budget, and the depth/width corrections in §3 — these follow from the structure of any doubly-stochastic-constrained multi-stream residual design, independent of a specific checkpoint.
* **Checkpoint-specific:** the stream count (4) and the exact placement of mHC pre/post-processing around each sublayer are properties of this checkpoint's configuration.

---

**Next:** [Module 08 — Vision, MTP, and Hybrid Serving State →](Lecture-08.md)
