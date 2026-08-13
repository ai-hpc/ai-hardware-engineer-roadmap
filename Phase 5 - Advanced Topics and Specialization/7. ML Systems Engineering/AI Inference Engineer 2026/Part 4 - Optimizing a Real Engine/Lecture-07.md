# Part 4 · Lecture 07 — Sharding 896 Experts, and the Amdahl Trap That Followed

## Overview

896 routed experts, top-16, across 8 GPUs. The obvious partition — 112 whole experts per rank — is correct, balanced *on average*, and **leaves ~43% of the MoE layer's critical path on the table**, for a reason that is pure probability rather than engineering.

This lecture covers the four decisions that make a tensor-parallel MoE layer fast:

1. **Where the reduce goes** — determined by algebra, not by a diagram.
2. **What it costs** — measured, and much less than everyone assumes.
3. **How to shard** — and why the critical path is a *maximum*, not a mean.
4. **Which collective runs** — and the 7.2% that sat behind an unset environment variable for weeks.

Plus the case study's cleanest Amdahl demonstration: a **3× MoE speedup that made tensor-parallel scaling worse**, 2.44× → ~1.1×.

By the end you should be able to place a collective from first principles, compute the expected critical path of a top-*k* routing scheme, and decide a sharding geometry from a balls-in-bins argument rather than a sweep.

---

## 1. Where the reduce goes is an algebra question

Start here, because getting it wrong is silent ([Lecture 10 §4.3](Lecture-10.md)) and everything else depends on it.

The routed experts are expert-sharded, so each rank's dispatch accumulator holds a **partial sum** over the token's top-16 — it has whatever subset of those 16 it owns, and zero for the rest. The all-reduce turns eight partial sums into what one GPU holding all 896 experts would have computed.

The only question is *where*. And the answer is forced:

```text
   the next op after the expert dispatch is  ffn_routed_norm,
   an RMS norm.  and:

        rms_norm( Σ partial )   ≠   Σ rms_norm( partial )

   RMS norm is NOT LINEAR, so the cross-rank sum must complete
   BEFORE it.
```

> **A collective's position is determined by the first non-linearity downstream of the partial sum.** Every linear op — scaling, adding a residual, a further matmul — commutes with the sum and can happen on either side. The first non-linearity cannot, and that is where the reduce must land.

Move it two ops later, past `routed_up`, and two things go wrong simultaneously: you skip the reduce the experts needed, **and** — because `routed_norm`, `routed_up` and the shared experts are all replicated, so every rank already holds the complete tensor — you reduce a complete tensor and **multiply the FFN output by `tp_size`**.

Multiplying an FFN output by 8 does not crash. It produces fluent text.

### 1.1 The payload is narrower than you would guess

```text
   the reduce is 3584 wide,  NOT 7168.
```

The routed experts live in a **down-projected latent space** — `expert_latent_length` 3584 against `hidden_size` 7168. Sizing the collective off hidden width predicts twice the bytes that actually move. This is the same trap that makes every expert GEMM wrong by 2× if you size it off hidden ([Lecture 01 §1.1](Lecture-01.md)).

At f32 that is `3584 × 4 = 14 KiB` per collective. Note that `7168 × 2` (bf16) is *also* 14 KiB — the same number for a different reason, which is exactly the sort of coincidence that hides an error. The engine runs f32 deliberately: routing an f32 residual stream through a bf16 all-reduce would truncate to ~8 mantissa bits **at every layer boundary**, undoing the executor's numerics.

### 1.2 Count the collectives from the forward pass

```text
   attention REPLICATED (ExpertsOnly):
       1 MoE reduce × 92 MoE layers                 =  92 per token
       (93 layers, minus the leading dense block, which has no
        expert dispatch to reduce)

   attention HEAD-SHARDED (after PR #63, Lecture 06 §5):
       92 MoE reduces  +  93 attention reduces      = 185 per token
```

The textbook figure from [Part 2 Lecture 04](../Part%202%20-%20Dense%20at%20Hopper/Lecture-04.md) — two all-reduces per layer — is right only for a fully-sharded model. Which number applies to *you* depends on your shard policy, and the case study's own docs record having quoted 186 when the answer was 92.

The count is **asserted**, not eyeballed: *"A missing reduce leaves a partial expert sum; an extra one multiplies a complete tensor by `tp_size`. Neither crashes, so the count is asserted rather than eyeballed."* Every PR in this lecture reports `collectives=185` in its bench output, on every arm. That is how you notice a change that silently altered the communication pattern.

---

## 2. What the collective costs — measure before optimizing it

A 93-layer model on 8 GPUs *sounds* comm-bound. It was not.

```text
   MEASURED, ExpertsOnly, tp_size 8, 8× H200, NCCL:

     92 all-reduces/token × 14 KiB × 58.7 µs/call  =  ~5.4 ms/token
     token time at the time of measurement          =  281.6 ms
     collective share                               ≈  2%
```

Verdict: not the bottleneck. And the regime matters as much as the share:

```text
   512× the data costs 1.45× the time     (14 KiB → 7 MiB)
   ⇒ LATENCY-bound, not bandwidth-bound.
   ⇒ report µs/call, not GB/s.
   ⇒ optimize the CALL COUNT and the BARRIER, not the bytes.
```

This is why the validation tool reports microseconds per call, and why §7's improvements are about *barrier mechanism* rather than bandwidth. For any collective under ~1 MB, GB/s is a meaningless unit — it tells you about your fixed overhead, not your fabric.

---

## 3. The Amdahl trap

The shard policy `ExpertsOnly` bands the 896 routed experts (531 of 553 GiB) and **replicates everything else, including attention**. That was the right first cut: essentially all of the memory win, one collective per layer instead of two, and no per-rank shape threading anywhere in the forward pass.

Then the MoE dispatch got ~3× faster:

| | before the MoE speedup | after |
|---|---:|---:|
| tp=1, 16 layers | 196.65 ms/token | **60.66** |
| tp=8, 16 layers | 80.61 ms/token | **54.33** |
| **tp=8 vs tp=1** | **2.44×** | **~1.1×** |

Nothing regressed — `tp=8` improved from 80.61 to 54.33 ms/token, a real 1.48×. But the replicated attention did not get faster and did not get sharded, so it became the **entire serial term**. Solving Amdahl for the serial fraction:

```text
   speedup(N) = 1 / ( s + (1−s)/N )

   before:  2.44× at N=8  →  s ≈ 0.33
   after:   1.10× at N=8  →  s ≈ 0.90
```

The parallel part shrank 3×; the serial part did not move; its *share* went from a third to nine tenths. Every additional GPU now buys almost nothing.

Two conclusions, and the second is the one teams get wrong:

**Re-derive your serial fraction after every significant win.** A scaling number measured before an optimization is not valid after it. Publishing a stale TP-scaling figure is how an organization buys GPUs that do nothing.

**The next optimization is chosen by the new serial fraction, not by the old plan.** The repo's roadmap said so directly — *"Attention is now the whole serial term"* — and [PR #63](https://github.com/gittensor-ai-lab/sparkinfer-k3/pull/63) head-sharded both attention bands for **+72.6%** ([Lecture 06 §5](Lecture-06.md)). That is what fixing the serial term looks like once you have identified it.

A third, quieter point: `ShardPolicy::Full` — sharding the attention *weights* in the loader, not just the compute — is declared and **deliberately not enabled**, because the loader would shard weights the executor still indexes at full width, reading past the end of a slice. Knowing the right next move and knowing it is not yet safe are different states, and conflating them ships silent corruption.

---

## 4. The critical path is a maximum, not a mean

Now the best piece of reasoning in this lecture, from [**PR #96**](https://github.com/gittensor-ai-lab/sparkinfer-k3/pull/96).

Whole-expert sharding looks perfectly balanced on paper: 896 experts ÷ 8 ranks = 112 each, and top-16 means each rank owns an *expected* 2 of the token's active experts. Even. Fair.

But a synchronous layer does not wait for the average rank. **It waits for the busiest one.**

```text
   top_k = 16 active experts, thrown into 8 ranks
     → balls in bins:  16 balls, 8 bins

   expected count in a given bin      =  2.0
   expected count in the BUSIEST bin  ≈  4.2      ← this is what you wait for

   critical-path expert-rows = 4.20 × 3072 rows = 12,902
```

The gap between 2.0 and 4.2 is a factor of **2.1×**, and it is not a load-balancing bug you can fix by shuffling the assignment. It is the max of a multinomial. With few balls and many bins, the maximum is far above the mean, and the layer's cost is set by the maximum.

> **Steal this.** For any top-*k* routing over *N* ranks, your synchronous critical path is governed by `E[max bin]`, not `k/N`. With small *k*, those differ by a factor of two or more. Compute it — analytically or by simulation — before you conclude your expert partition is balanced.

### 4.1 The fix: trade a badly-balanced axis for a perfectly-balanced one

The insight is that **the FFN width is a fixed dimension.** Every expert has exactly 3072 FFN rows, always, regardless of routing. Splitting on that axis is *perfectly* balanced by construction — there is no randomness in it at all.

So shard on **two** axes: expert groups × FFN band.

```text
   eg = 2:  each rank owns  448 of 896 experts
                       AND  768 of each expert's 3072 FFN rows
            → the SAME weight bytes per rank as before

   fewer, fatter expert groups → worse expert balance:
       16 balls in 2 bins  →  E[busiest] ≈ 9.57   (up from 4.2)

   but each active expert is now only 768 rows, not 3072:

   critical-path expert-rows   before  4.20 × 3072 = 12,902
                               after   9.57 ×  768 =   7,350
                                                     (0.57×)
```

You deliberately make the *random* axis worse — 4.2 → 9.57 expected busiest — because you get the loss back on an axis with **no variance**. Total bytes per rank are unchanged; only the critical path shortens, by 43%.

Measured: **33.42 → 29.81 ms/token at 128k, +12.1% decode.**

> **The general principle.** When a partition's imbalance comes from randomness, look for a second axis that is deterministic, and move parallelism onto it. Trading a stochastic dimension for a fixed one shortens the critical path even when it lengthens the average.

This is the single-node relative of the grouped-GEMM / all-to-all machinery in [Part 3 Lecture 03](../Part%203%20-%20MoE%20at%20Blackwell/Lecture-03.md) — both are answers to "routing is uneven, and synchronous layers pay for the worst case."

### 4.2 Why it needed no kernel changes

The implementation detail that makes this cheap is worth noting, because it says something about how to write MoE kernels in the first place:

> *No `kernels/` file changes. `moe_gate_up_situ_kernel` already indexes `gate_exps + (e·ffn + j)·blocks_per_row` and `moe_down_combine_kernel` `down_exps + (e·latent + o)·blocks_per_row`; **passing the rank's 768-row band as `ffn` addresses its packed buffer exactly as 3072 addressed the whole one.** Only the loader's packing changes.*

Because the kernels took the FFN width as a *parameter* rather than a constant, sharding that axis is a change to what the loader writes and what number gets passed in. A kernel with `#define FFN 3072` would have required a rewrite.

And the correctness argument is by analogy to something already in the codebase:

> *The shared expert already shards this way: gate/up row-shard, `situ` is elementwise and preserves the band, `down` col-shards over the same band leaving a full-width partial that the existing expert all-reduce sums. **No new collective and no change to its width or count** — still 185 collectives/token.*

That is the row-then-col composition from [Part 2 Lecture 04](../Part%202%20-%20Dense%20at%20Hopper/Lecture-04.md): row-shard the up-projection so each rank produces its own band of intermediates, keep the elementwise activation inside the band, then col-shard the down-projection so each rank produces a full-width partial that the *existing* reduce sums. The FFN band rides on a collective that was already there.

---

## 5. The default must be the thing you claim

[**PR #96**](https://github.com/gittensor-ai-lab/sparkinfer-k3/pull/96)'s first revision put 2-D sharding behind an opt-in environment variable. The result:

> *That was a mistake: **the eval builds the tree and runs the bench, it does not set environment variables**, so it measured the whole-expert path and correctly scored the PR `eval:none` at 0.1% over frontier. **A perf change the harness cannot reach is not a perf change.**"*

This is in direct tension with the single-binary A/B technique from [Lecture 04 §6](Lecture-04.md), and the resolution is important: **gate it so you can measure it, then make the new behaviour the default and keep the gate for the old one.** `SPARKINFER_K3_MOE_2D=1` now restores whole-expert sharding — the gate still exists, but it selects the *old* path.

Being a default forced two things opt-in did not need:

* **A shape that cannot take it must still load.** Refusing would turn a tuning default into a portability regression. So: *explicit request → hard error; chosen-for-you → fall back to whole-expert and say so on stderr.* Safe because the dimension validator runs *before* it writes anything, leaving the whole-expert band intact.
* **The default policy is pinned by a test**, *"since a behavioural default that drifts silently re-shards every deployment's experts."*

> **The two rules.** An explicitly requested configuration should **fail loudly** if impossible; a configuration chosen on the user's behalf should **degrade quietly and say so**. And any behavioural default worth having is worth a test that fails when it changes.

### 5.1 The 7.2% that sat behind an unset variable

[**PR #74**](https://github.com/gittensor-ai-lab/sparkinfer-k3/pull/74) is the purest instance of this failure, and one of the highest-value-per-line changes in the repository — about 45 lines, no new code paths.

[PR #59](https://github.com/gittensor-ai-lab/sparkinfer-k3/pull/59) had built a fast collective and measured both arms on the eval node:

| build | ms/token | tok/s |
|---|--:|--:|
| #59, NCCL backend | 58.77 / 57.18 | 17.02–17.49 |
| **#59, peer one-shot** | **54.52 / 54.48** | **18.35** |

7.2% apart. And then:

> *The eval round that followed recorded the frontier at **17.46 tok/s** — the NCCL number. **The fast path #59 built has been sitting behind an environment variable nothing in the measurement chain sets, since the day it merged.**"*

An unset `SPARKINFER_TP_BACKEND` meant NCCL. Nothing in the benchmark scripts or the eval bot set it. So the fast collective was merged, validated, measured — and never once ran in production or in any scored round. #74 makes unset mean **auto**: peer-oneshot when peer access exists across all pairs, NCCL otherwise.

Note also what the author says about the evidence:

> *Nothing about this PR's claim is my measurement — it is #59's own numbers on the pinned node, cross-checked against the public ledger. What this PR changes is only that the default reaches the arm that won.*

A PR whose entire content is changing a default, justified entirely by someone else's already-sealed measurements, cross-checked against the public log. That is only possible because the ledger exists ([Lecture 02](Lecture-02.md)).

> **Audit your defaults against your measurements.** For every performance flag, ask: *does the configuration my benchmark runs match the configuration my users run?* An optimization only your flags reach is an optimization nobody has.

And the restraint worth copying: **multimem is deliberately left out of the auto set**, because it remains unvalidated on hardware while peer-oneshot has a measured before/after and a validation tool behind it. "Auto" should select among options you have *evidence* for, not among options that exist.

---

## 6. Faster collectives: what actually changed

[**PR #59**](https://github.com/gittensor-ai-lab/sparkinfer-k3/pull/59) — **16.06 → 18.35 tok/s, +14.3%** — bundles three things, and the split between them is instructive:

| build | ms/token | tok/s | vs main |
|---|--:|--:|--:|
| main | 62.24 / 62.29 | 16.06 | — |
| this PR, **NCCL** backend | 58.77 / 57.18 | 17.25 | −6.9% |
| this PR, **peer one-shot** | 54.52 / 54.48 | **18.35** | −12.5% |

The middle row is the value of everything *except* the collective algorithm — the shared-expert banding and the staging-copy elimination. The bottom row adds the barrier mechanism. Reporting both isolates the two contributions.

**What "peer one-shot" changes.** Every rank reads all peers directly and sums in f32, with an **in-kernel flag barrier** — one kernel per rank, no host events, no cross-stream graph edges. Against NCCL's ring, at 14 KiB, the win is not bandwidth; it is the *barrier*. Since §2 established this regime is latency-bound, the barrier mechanism is the only thing left to optimize. (This is the same small-message custom-all-reduce story as [Part 2 Lecture 07](../Part%202%20-%20Dense%20at%20Hopper/Lecture-07.md), and the code is attributed to vLLM's `custom_all_reduce` under Apache-2.0.)

**Why f32 had to be built.** The fast backends were bf16-only, so K3's f32 residual stream fell back to NCCL — the fast path existed and was unreachable for this model. And the fallback is negotiated *early*: `make_collective(..., need_f32=true)` downgrades a bf16-only backend **before the 20-minute weight load**, rather than failing at the first collective.

**Mode B, and why staging is not an option.** NCCL reduces the caller's own buffer in place; the fast backends cannot — only multicast-bound or peer-registered allocations can back their loads, so the buffer must belong to the collective. Hiding that behind a copy into and out of a caller buffer would add two 14 KiB device-to-device copies per collective — **372 extra copies per token**. So the in-place API *returns false* on a Mode-B backend rather than silently staging, making a mis-wired forward fail immediately instead of paying that cost 186 times a token.

> **When an optimization requires a different calling convention, expose the convention.** Papering over it with copies can cost more than the optimization saves, and a silent fallback to the slow shape is worse than a hard error.

### 6.1 Reassociation, and which direction it moves

Both #59 and #96 are explicitly **not bit-identical**, and both explain exactly what reassociates:

* **#59:** *"banding the shared expert replaces one 6144-term contraction with eight 768-term partials summed by the collective, and the f32 reduce reassociates. **The measured effect is toward the reference, not away from it.**"* Mean KLD vs llama.cpp **8.075e-03 → 3.867e-03** — a 2× *improvement*, for the pairwise-summation reason from [Lecture 06 §2.1](Lecture-06.md).
* **#96:** three-way comparison rather than two:

```text
                                   mean KLD    top-1     top-5
   whole-expert  vs reference      4.063e-03   100.00 %  100.00 %
   2-D default   vs reference      5.146e-03   100.00 %  100.00 %
   2-D  vs  whole-expert directly  3.255e-03   100.00 %  100.00 %
```

And the reading, which is a model of statistical honesty:

> *The 2-D path sits slightly further from the reference than whole-expert here; **on the previous base it sat slightly closer.** Both differences are smaller than the gap either one has to the reference, which is the honest reading: re-associating the sum moves the logits by about the size of the existing quantisation noise, in whichever direction the prompt happens to fall.*

The author had a result that mildly favoured the *old* path, reported it, noted it had gone the other way on a previous base, and concluded the difference is noise-scale rather than spinning either sign. The **direct A-vs-B comparison** (3.255e-03) is what makes that argument possible: if the two candidates differ from each other by less than either differs from the reference, neither is meaningfully closer.

> **Steal this.** When comparing two approximations, measure all three distances: A-to-reference, B-to-reference, and **A-to-B**. Without the third you cannot tell a real difference from resampling noise.

---

## 7. Verifying a shard: conservation, not counting

[#96](https://github.com/gittensor-ai-lab/sparkinfer-k3/pull/96) ships **7,150 CPU checks** — 5,117 on shard math, 1,740 on the weight plan, 293 on residency — all GPU-free. And one of them is the best test in the case study:

> *The residency test adds a **byte-level 2-D conservation case**: it replays every rank's copy descriptor over a byte map and requires **each byte to be covered exactly once**, since **a byte *count* check would pass a stride that reads the wrong bytes in the right quantity.**"*

That clause is the whole insight. A 2-D strided copy with transposed strides moves exactly the right number of bytes from exactly the wrong places. Any test that verifies *totals* passes it. Only a test that verifies *which bytes* catches it — and [Lecture 10 §2](Lecture-10.md) is a real transposed-stride bug that shipped and was fast.

The test also covers the **refusals**, which is where sharding constraints live:

```text
   · group count must divide BOTH tp_size AND n_experts
   · the FFN shard must be a whole number of quant blocks
        768 = 3 × 256   ✓
   · a refusal must leave the ShardDims UNTOUCHED
        (the fallback path in §5 depends on this)
```

The quant-block constraint is the general form of the FP8-block/TP-alignment footgun from [Part 2 Lecture 03 §5.4](../Part%202%20-%20Dense%20at%20Hopper/Lecture-03.md): **you cannot split a quantization block across ranks**, so every shard boundary must be a block boundary. Any sharding scheme on quantized weights inherits this, and it constrains your group counts before performance does.

And the last line — *a refusal must leave the state untouched* — is a transactional property. §5's quiet-fallback behaviour is only safe if the validator cannot half-write a configuration before rejecting it.

---

## 8. Disclosing a regression you did not fix

[#96](https://github.com/gittensor-ai-lab/sparkinfer-k3/pull/96)'s closing section is a model of how to ship an imperfect change.

**The cost:** model load goes **~22 s → ~31 s**. The `down` slice is a small-pitch `cudaMemcpy2D` — 1.6M rows of ~150 B at IQ1_S — which is less efficient than a contiguous host-to-device transfer despite an identical byte count. *"Outside the timed region and it does not touch the score, but it is a real regression."*

**The attempted fix, measured and rejected:**

> *An earlier revision of this PR promised a host-side repack would recover it. **I implemented that and measured it, and it is worse: 48.3 s against 30.9 s.** The gather is 1.6M single-threaded small `memcpy`s per rank, first-touching mmap'd page-cache pages, plus a second full pass over the bytes — **the driver's own strided walk beats it.**"*

**Where the negative result was recorded:**

> *Reverted, with the numbers and the reason **recorded at the call site** so the next person sizing up that line does not spend the same afternoon on it.*

A comment at the line someone will want to change, containing the measurement that says not to. That is worth more than the same information in a closed PR, because it is where the next engineer will be looking.

**And the boundary of the claim:**

> *I did not re-run the logit comparison under the gather, so "correct" there is **by construction** — it copies the same source ranges into the same layout — **not by measurement.** [...] A threaded gather might still win, but I have not measured it and will not claim it.*

Three habits in one paragraph: distinguishing correct-by-construction from correct-by-measurement, disclosing an unfixed regression rather than hoping nobody looks at load time, and declining to speculate about an optimization not attempted.

Note also that the regression is **outside the scored region**. It would have been easy to omit. The reason to include it is that load time is real to whoever deploys this, and a scoring metric that does not cover something does not make it free.

### 8.1 Two more reporting habits from the same PR

**Statistics, not just means.** Five ABBA-interleaved pairs, first discarded as warm-up:

```text
   whole-expert   33.43 33.44 33.36 33.43 33.43   mean 33.418  sd 0.033
   2-D default    29.82 29.83 29.80 29.80 29.80   mean 29.810  sd 0.014

   delta −3.608 ms/token   (−10.80% latency, +12.10% throughput)
   Welch t = 226;  the arms do not overlap
                   (base min 33.36 > 2-D max 29.83)
```

The non-overlap statement is the useful one for a reader: *the worst run of the fast arm beat the best run of the slow arm.* No statistics degree required.

**Declining a flattering denominator.** The box measured `main` itself at 29.92 against a recorded frontier of 26.09:

> *quoting "+28% over frontier" would be crediting this change with a box/build difference it did not cause. The defensible number is the same-box A/B: **+12.1%**.*

And on percentages versus absolutes: an earlier revision measured the same change at 38.31 → 34.76 on an older base. *"The absolute saving is essentially unchanged at ~3.6 ms because the MoE imbalance is a **fixed cost** that #90 did not touch; the percentage grew because the rest of the token got faster."* **A fixed-cost fix earns a rising percentage as everything else improves** — which is worth understanding before you conclude your change got better.

---

## Lab — shard an MoE layer and defend the geometry

1. **Derive your reduce point.** Write out your MoE layer's ops in order. Find the first non-linearity after the partial sum. Place the collective there and state the algebraic reason. Then state what would happen if you moved it one op later.
2. **Count and assert.** Collectives per token, and payload width — from the forward pass, not a diagram. Add an assertion on the count and print it in your bench output.
3. **Price it.** Microbenchmark at your real payload, and at 8× and 64× it. If time barely moves, you are latency-bound: report µs/call and target the call count and barrier.
4. **Compute `E[max bin]`.** For your top-*k* over *N* ranks, analytically or by simulation. Compare to `k/N`. That ratio is the imbalance you are paying (§4).
5. **Find a deterministic axis.** Is there a fixed dimension — FFN width, head dim, hidden — you could shard instead of or alongside experts? Compute the critical path both ways as `E[max bin] × rows_per_expert` (§4.1).
6. **Check your quant-block constraint.** Does your candidate shard width divide into whole quantization blocks? Enumerate the legal widths before optimizing among them.
7. **Write the conservation test.** Replay every rank's copy descriptor over a byte map; assert each byte is covered **exactly once**. Then deliberately transpose two strides and confirm the test fails — a byte-count test would not (§7).
8. **Audit your defaults.** For every performance flag, check whether your benchmark and your production entry point set it. Report any flag where they differ (§5.1).
9. **Measure your serial fraction, twice.** At N=1 and N=max, before and after your change. Solve Amdahl for `s` each time. If `s` rose, say what the next lever is (§3).
10. **Report the three KLDs.** A-to-reference, B-to-reference, and A-to-B. Conclude only what the third supports (§6.1).

Pass criterion: a committed sharding decision justified by an `E[max bin]` calculation, a byte-conservation test you have watched fail, a defaults audit, and before/after serial fractions.

---

## Self-check

1. Your expert dispatch produces a partial sum, then a scale, then an RMS norm, then a matmul. Where can the all-reduce go, and where must it not? Give the algebra.
2. You move the reduce past the up-projection. Nothing crashes and the text is fluent. What is the output multiplied by, and why does that specific factor appear?
3. Your model has hidden 7168 and expert latent 3584. You size the collective off hidden. By what factor is your bandwidth estimate wrong, and would you notice from the timing?
4. Top-8 routing over 4 ranks. Compute `k/N` and estimate `E[max bin]`. What is the imbalance factor, and what does it cost a synchronous layer?
5. Explain why deliberately worsening expert balance (2 groups instead of 8) can shorten the critical path. Give the arithmetic with a 3072-row FFN.
6. A collective takes 58.7 µs at 14 KiB and 85 µs at 7 MiB. What should you optimize, and what unit should you report?
7. A merged, measured, validated fast collective never ran in any scored round. Give the mechanism and the one-line audit that would have caught it on day one.
8. Your shard test verifies each rank copies the correct number of bytes. Describe a bug it cannot catch, and the test that can.
9. A 3× improvement to a phase makes your 8-GPU scaling worse. Compute the serial fraction before and after, and name what to fix next.
10. Approximation A is 4.063e-03 from the reference, B is 5.146e-03, and A-to-B is 3.255e-03. Which is more accurate, and what is the defensible claim?

---

## References

* **The PRs** — [#59](https://github.com/gittensor-ai-lab/sparkinfer-k3/pull/59) (f32 peer one-shot, Mode B, shared-expert banding), [#74](https://github.com/gittensor-ai-lab/sparkinfer-k3/pull/74) (make unset mean auto), [#96](https://github.com/gittensor-ai-lab/sparkinfer-k3/pull/96) (2-D MoE sharding; the balls-in-bins argument; the byte-conservation test; the disclosed load regression), plus [#107](https://github.com/gittensor-ai-lab/sparkinfer-k3/pull/107) (one-rendezvous all-reduce). [`docs/tensor-parallel.md`](https://github.com/gittensor-ai-lab/sparkinfer-k3/blob/main/docs/tensor-parallel.md) is the source for §1–§3.
* **Megatron-LM tensor parallelism** — [arXiv:1909.08053](https://arxiv.org/abs/1909.08053) — the row-then-col composition §4.2 relies on.
* **vLLM custom all-reduce** — [github.com/vllm-project/vllm](https://github.com/vllm-project/vllm), `csrc/custom_all_reduce.cuh` (Apache-2.0) — the `Signal` layout, dual-counter barrier and packed FP32 reduce that §6's peer one-shot adapts, with attribution retained in the repo's `NOTICE`.
* **NCCL** — [docs.nvidia.com/deeplearning/nccl](https://docs.nvidia.com/deeplearning/nccl/) — ring and tree all-reduce, and the small-message regime where a custom kernel wins.
* **NVLink SHARP / multimem** — [NVIDIA NVLink Switch](https://www.nvidia.com/en-us/data-center/nvlink/) — in-network reduction; §5.1's deliberately-excluded backend.
* **Balls into bins / maximum load** — Raab & Steger, *"Balls into Bins — A Simple and Tight Analysis"* ([RANDOM 1998](https://link.springer.com/chapter/10.1007/3-540-49543-6_13)) — the maximum-load result behind §4.
* **DeepEP** — [github.com/deepseek-ai/DeepEP](https://github.com/deepseek-ai/DeepEP) — expert-parallel communication at cluster scale; the multi-node relative of §4.1.

Cross-references:

* [Part 2 Lecture 04 — Tensor parallelism on 8× Hopper](../Part%202%20-%20Dense%20at%20Hopper/Lecture-04.md) — row/col sharding and the "two all-reduces per layer" figure §1.2 corrects.
* [Part 2 Lecture 07 — Inside the communication layer](../Part%202%20-%20Dense%20at%20Hopper/Lecture-07.md) — the small-message custom all-reduce §6 implements.
* [Part 3 Lecture 03 — Expert parallelism and the gating hot path](../Part%203%20-%20MoE%20at%20Blackwell/Lecture-03.md) — the all-to-all answer to the same imbalance, at cluster scale.
* [Lecture 03 §6 — Diagnosis](Lecture-03.md) — the Amdahl table in its diagnostic context.
* [Lecture 06 §5 — Head-shard the bands](Lecture-06.md) — the fix for §3's serial term.
* [Lecture 10 §4.2–4.3 — Silently wrong](Lecture-10.md) — partition and collective-placement bugs as a failure family.

---

## Current as of 2026-08

8× H200 SXM, `sm_90`, CUDA 12.8+, NCCL, UD-IQ1_S, tp=8, scored context 131,072. 896 experts / top-16 / expert latent 3584 / expert FFN 3072; 2-D sharding default at `tp_size ≥ 4` with `eg=2` (448 experts × 768 FFN rows per rank, 768 = 3 × 256 quant blocks). 185 collectives/token with attention head-sharded; 92 with it replicated. The reduce-placement algebra, the `E[max bin]` argument, the defaults audit, and byte-conservation testing are the durable content.

---

## Next

* Next: [Lecture 08 — Graph-resident decode: killing the launch bill for good](Lecture-08.md)
* Previous: [Lecture 06 — Attention at 128k: split over context, split over heads](Lecture-06.md)
* Up: [Part 4 — Optimizing a Real Engine](README.md)
