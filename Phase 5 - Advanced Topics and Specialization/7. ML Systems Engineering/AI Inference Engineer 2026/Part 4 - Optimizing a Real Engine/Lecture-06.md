# Part 4 · Lecture 06 — Attention at 128k: Split Over Context, Split Over Heads

## Overview

This is where the 18× gap lived.

At context 64 the engine was 1.8× behind the reference. At 131,072 it was **18× behind** — 1.01 tok/s against 18.44 — and the reference's rate was essentially *flat* across that whole range. All of that divergence was one kernel: `attn_mla` consumed **69.2% of decode at 128k**, and it was structured in a way that could not scale with depth.

Three pull requests took it from 1.01 to 15.93 tok/s — **15.8×** — with no change to the mathematics of attention. Each one found a different axis to parallelize:

```text
   #49   split over CONTEXT   →  990.98 → 220.94 ms/token   4.49×
   #57   batch HEADS per block →  220.5  → 110.4  ms/token   2.00×
   #63   shard heads over RANKS → 108.4  →  62.8  ms/token   1.73×
```

By the end you should be able to derive the split-attention combine from the online-softmax recurrence, choose a tile width from register pressure rather than by sweeping, and — the part that distinguishes this case study — *test and measure a code path your correctness gate structurally cannot reach.*

---

## 1. The shape that cannot scale with depth

The original `mla_decode_attn_kernel` gave **each of 96 heads a single thread block that walked all 131,072 tokens serially.**

```text
   grid = n_head = 96 blocks    on a 132-SM H200
                                → 36 SMs receive NO BLOCK AT ALL
   96 blocks × 256 threads      = 24,576 threads
   each block walks 131,072 KV positions, serially

   time per layer ∝ n_ctx,  with 27% of the GPU idle throughout
```

Two independent defects stacked. **Occupancy**: 96 blocks on 132 SMs, and no arrangement of 96 fills 132. **Serial depth**: every block's runtime grows linearly with context, because the sequence dimension is walked rather than parallelized.

The reference engine did not have this problem because it kept a *compressed* MLA cache (`kv_lora` 512, f16) and a kernel that parallelizes over depth. Hence the flat curve. This is precisely the divergence [Lecture 03 §5](Lecture-03.md) says to look for: **your candidate's cost grows with a dimension the reference's does not.**

### 1.1 The number that names the bug

[**PR #57**](https://github.com/gittensor-ai-lab/sparkinfer-k3/pull/57) states the underlying waste in one sentence, and it is the best framing of MLA-at-decode I have seen:

> *MLA **is** MQA — 96 query heads attend over one shared latent KV cache — but decode gave each head its own block, so at 128k **every layer streamed the same 302 MB cache 96 times.**"*

MLA compresses KV into a single shared latent per position (§1 of [Part 3 Lecture 01](../Part%203%20-%20MoE%20at%20Blackwell/Lecture-01.md)). That compression is the architecture's whole selling point — and a kernel with one block per head throws it away, because 96 blocks each read the entire shared cache independently. The architecture provided the reuse; the launch geometry declined it.

And the supporting arithmetic, from `nsys`:

```text
   mla_decode_attn_split   56.8%  = 124 ms   (5.165 ms × 24 MLA layers)
   proj_q8_0_multirow      17.0%
   proj_q8_0_fused4         8.9%

   main at ctx 64 is 98.7 ms/token
   ⇒ ~124 ms of the 221 ms token IS the cache walk.
```

Subtracting the short-context token time from the long-context token time isolates the depth-dependent term. That is a two-measurement diagnosis requiring no profiler, and it should be the first thing you do when a curve slopes.

---

## 2. Split over context

[**PR #49**](https://github.com/gittensor-ai-lab/sparkinfer-k3/pull/49) — *"split MLA decode over context (4.49× at 128k) + fail the forward on a failed launch."*

The move is **Flash-Decoding**: manufacture parallelism from the one dimension that is large at decode — the KV cache itself.

```text
   grid:  n_head   →   n_head × splits
```

Each block runs the same online softmax over a **contiguous** slice of the context and writes a partial `(m, l, latent)`. A combine kernel merges them:

```text
   m   = max_i  m_i
   l   = Σ_i    l_i  · exp(m_i − m)
   acc = Σ_i    acc_i · exp(m_i − m)
```

This is the online-softmax rescaling of FlashAttention, applied *across blocks* rather than across tiles within a block. Each partial carries its own running max `m_i` and normalizer `l_i`; the merge rescales every partial to a common maximum before summing. It is exact up to floating-point reassociation.

Six implementation decisions in that PR, each worth naming because each closes a real failure:

**Contiguous slices, not strided.** *"Contiguous rather than strided so each block's cache walk stays sequential for the prefetcher."* Strided assignment would give perfect load balance and destroy locality. At batch 1 over a 302 MB cache, locality wins.

**The output projection moves into the combine.** `wv_b` needs the *merged* latent, so it cannot run per-slice. Recognizing which downstream op is a function of the combined result — rather than of each partial — is the part of split-K that is easy to get wrong.

**Gate the split below a threshold.** Split only above `kMlaSplitMinCtx` (4096): *"below it the combine pass and extra global round trip cost more than the parallelism buys, and the un-split kernel stays the one the numeric test pins."* A split adds a kernel launch and a round trip through global memory. At short context those exceed the parallelism gained.

**Per-device scratch, not one global allocation.** The bug this prevents is worth quoting in full because it is a *tensor-parallel* trap, not an attention one: *"a single static pointer is allocated by whichever rank arrives first and then dereferenced by the other seven on devices it does not belong to."* Eight ranks, eight devices, one `static` pointer — and seven ranks reading memory that belongs to another device.

**Empty slices must be neutral, not `NaN`.** When `n_ctx < splits`, some slices have no work. They set `l = 0, m = -1e30` so their `exp(m_i − m)` contributes zero. Left uninitialized, `exp(-inf - -inf)` gives `NaN`, and one `NaN` in the merge poisons the token.

**Shared memory is `O(key_length + kv_lora + tile)`, never `O(n_ctx)`.** This matters because [PR #33](https://github.com/gittensor-ai-lab/sparkinfer-k3/pull/33) had just fixed a bug where the kernel *"sized dynamic shared memory by context length"* — which is why *"MLA decode attention silently stops launching past ~11.7k context."* Shared memory per block has a hard cap; any allocation that scales with context has a context at which the launch fails. See §5.3 and [Lecture 10 §4.4](Lecture-10.md).

### 2.1 Splitting can *improve* accuracy

The split path is deliberately **not** bit-identical — summing per-slice partials reassociates the terms. Measured at 128k across all 8 ranks:

```text
   mean KLD         1.333795e-10     (threshold < 1e-5)
   top-1 agreement  100.0000 %
   top-5 overlap    100.0000 %
   RMS dp           0.000001 %

   base argmax 378 @ 10.563942
   PR   argmax 378 @ 10.563975
        "a 5th-decimal difference, the signature of summation
         reassociation, five orders of magnitude inside tolerance"
```

And then the pleasant surprise:

> *It happens to land **~4× more accurate** at n_ctx=20000 (relL2 3.734e-07 vs 1.607e-06), because **per-slice partials are effectively pairwise summation**.*

Summing 131,072 terms sequentially into one f32 accumulator accumulates rounding error roughly as `O(n)`. Summing them in 64 independent partials and then combining is a two-level tree — the error term drops toward `O(log n)`. **Splitting for parallelism gives you pairwise summation for free.**

> **When you split a long reduction, expect accuracy to improve, not degrade.** If it degrades, your combine is wrong. This is a useful sanity check on any split-K implementation, and it inverts the instinct that "not bit-identical" means "worse."

---

## 3. The measurement craft in #49

Three techniques from this PR that are worth more than the optimization.

### 3.1 A control phase

The profile is presented with a phase that the PR *does not touch*:

| phase | before | after | |
|---|--:|--:|---|
| `attn_mla` | 64912.89 ms (69.2%) | 9607.90 ms (28.8%) | **6.76× less** |
| `ffn_moe` | 23378.38 ms (24.9%) | 20993.16 ms (63.0%) | ~10% (from another PR) |
| `attn_kda` | 2655.21 ms (2.8%) | 2657.01 ms (8.0%) | **2655 → 2657: unchanged** |
| total | 93740.70 ms | 33333.80 ms | |

> *"**`attn_kda` is the control.** This PR does not touch it, and across a separate build and run it moved by 0.07%. **One phase falling 6.8× while its neighbour sits still is what distinguishes a real effect from a harness artefact.**"*

If a rebuild, a driver change, or box drift were responsible for the improvement, *everything* would move. A neighbouring phase that reproduces to 0.07% across separate builds and runs establishes that the harness is stable and the effect is localized.

> **Steal this.** Every profile comparison should name a phase you did not touch and report it. It costs nothing and it is the difference between "this phase got faster" and "this run was faster."

Note also what the table shows about percentages: `ffn_moe` went from 24.9% to **63.0%** of decode while getting ~10% *faster* in absolute terms. Percentages are ratios against a shrinking total. **Always report absolute times alongside shares**, or you will read a fixed cost as a regression.

### 3.2 Measure your noise floor with a gated-off path

The context sweep includes rows the PR cannot affect, and the author uses them as an instrument:

```text
   ctx 128    97.58 → 95.46 ms    +2.2%
   ctx 4k    124.07 → 122.68 ms   +1.1%
   ctx 8k    151.72 → 124.02 ms   +22%    ← the split first pays
   ctx 128k  990.98 → 220.94 ms   +349%
```

> *"Read the 128 and 4k rows as neutral, not as wins. Below `kMlaSplitMinCtx` the split path is gated off and **both builds run the same code**, so the 2.2% at ctx 128 is a direct measurement of this node's run-to-run spread. That puts the noise floor near 2%, which makes the 4k row (+1.1%) **not a result.** The claims worth scoring are 8k and 128k, which are 9× and 200× that floor."*

This is a genuinely elegant trick. A gated-off code path gives you an **A/A test embedded inside your A/B test**: same binary, same command, same box, identical code, so any difference is pure measurement noise. You get your noise floor from the same run that produces your result, on the same hardware, at the same moment.

Then the same discipline is applied *against* the author's own change. The launch guard's cost measured 141.35 → 142.54 ms, ~0.8% — below the 2% floor:

> *"so the honest claim is 'no measurable cost', not '0.8% cost'."*

A number smaller than your noise floor is not a small number. It is *no number*. Reporting it as 0.8% would overstate the precision of the measurement in the direction that makes the author look careful — which is still overstating it.

### 3.3 Reasoning when the tool is unavailable

`ncu` was not available on the node (`ERR_NVGPUCTRPERM` — the classic locked-down performance-counter permission). So the author could not directly distinguish two hypotheses for the 691.6 ms:

```text
   H1  bandwidth-bound:  blocks drift apart and re-read DRAM
   H2  latency-bound:    blocks stay in lockstep, L2 absorbs the reuse

   → "I targeted the deficiency BOTH models agree on:
      ~9% occupancy with 36 SMs idle."

   → the 4.49× result settles it:
      "a bandwidth-bound kernel could not respond to pure
       parallelism like this."
```

Two moves: **act on what competing hypotheses agree about**, and **let the outcome discriminate between them.** A bandwidth-limited kernel does not get 4.5× faster from more blocks — the bytes are the bytes. Getting 4.5× from parallelism alone is itself the evidence that the kernel was latency- and occupancy-limited.

> **A missing profiler is not a blocked investigation.** Find the intervention every candidate explanation predicts will help, do it, and use the magnitude of the response to identify which explanation was right.

---

## 4. Batch heads per block, and stop at the register wall

[**PR #57**](https://github.com/gittensor-ai-lab/sparkinfer-k3/pull/57) attacks the *other* half: the 302 MB cache being streamed 96 times per layer. Batching **12 heads per block** means one pass over the cache serves 12 heads.

This is the same reuse-versus-occupancy tension as [Lecture 05 §2.3](Lecture-05.md), and the PR resolves it with the clearest data in the entire case study:

| heads/block | ms/token @ 128k | registers | blocks/SM |
|---|--:|--:|--:|
| 8 | 141.1 | 96 | 2 |
| **12** | **133.9** | 125 | 2 |
| 16 | 149.9 | 148 | **1** |

> *why 12 heads per block and not more: 16 **halves traffic again and loses to occupancy**.*

Sixteen heads per block reads the cache 6 times per layer instead of 8 — strictly less memory traffic — and is **12% slower**, because 148 registers per thread means only **one block resident per SM** instead of two. With one block per SM there is nothing to overlap its memory latency against.

This is the register-pressure cliff, and it is a *step function*, not a gradient. Occupancy is `floor(registers_per_SM / registers_per_block)`; crossing a threshold halves your resident blocks in one step. Sweeping 8 → 12 → 16 and reporting registers alongside the timing is what makes the mechanism visible rather than mysterious.

> **Steal this.** When tuning a tile or batch width, report **register count and blocks-per-SM** next to each timing. The optimum is almost always the largest tile that stays on the current side of an occupancy step — and you can compute where that step is from `-Xptxas -v` instead of discovering it by sweeping.

### 4.1 Attributing a stacked change

#57 bundles several changes and publishes the incremental attribution, measured on one box, interleaved:

```text
   main                                                        221.4 ms
   + MLA heads batched 12/block, 4 tokens per staged-q pass    133.9
   + attn_res_mix device-wide, KDA state staged, slices = SMs   125.6
   + Q8_0 projections ROWS 16/8/4, fused4 2 → 4                 110.4
```

A single "220 → 110, 2×" headline is unreviewable. This table says which change bought what, so a reviewer can question any one of them, and a later engineer can revert one without guessing. When you must bundle — and [PR #49](https://github.com/gittensor-ai-lab/sparkinfer-k3/pull/49) explains that the one-open-PR rule sometimes forces it — publish the ladder.

---

## 5. Shard the heads across ranks

[**PR #63**](https://github.com/gittensor-ai-lab/sparkinfer-k3/pull/63) — *"head-shard both attention bands, budget the split cap, default the Q8 projection path."* **108.4 → 62.8 ms/token at 128k: 9.23 → 15.93 tok/s, +72.6%.**

The observation: attention was running **redundantly on all eight ranks.** The shard policy bands the 896 experts across ranks but replicates everything else, so all 8 GPUs computed all 96 heads and threw away 7/8 of the work. Head-sharding the 24 MLA layers *and* the 69 KDA layers makes each rank compute its own head band, at the cost of an additional all-reduce per layer (185 collectives per token in this configuration, up from 92).

This is the [Lecture 03 §6](Lecture-03.md) Amdahl story getting its answer: replicated attention was the serial term, and head-sharding is what parallelized it. [Lecture 07](Lecture-07.md) covers the collective side.

### 5.1 The shard removes the work *and* the parallelism

The fourth item in #63's list is the most transferable sentence in this lecture:

> *Budget the MLA split cap on `n_head × splits`, because head-sharding **collapses the attention grid to 64 blocks** on a 132-SM part. Worth **−8%** and easy to miss: **the head-shard removes the work *and* the parallelism.**"*

Head-sharding divides `n_head` by `tp_size`. The attention grid is `n_head × splits`. So the same change that cut the work per rank by 8× also cut the grid by 8× — straight back into [Lecture 04 §1](Lecture-04.md)'s starvation. The author anticipated it, budgeted the split cap on the *product* `n_head × splits` rather than on `splits` alone, and recovered the 8%.

```text
   BEFORE shard:  96 heads × S splits   →  plenty of blocks
   AFTER  shard:  12 heads × S splits   →  64 blocks. starved.
   FIX: budget the cap on the PRODUCT, so S grows as n_head shrinks.
```

> **Any change that divides a parallel axis must re-derive every grid computed from it.** Not "check the kernel you changed" — re-derive the *budget*, so the constant adapts instead of needing to be re-tuned. And then, as [Lecture 04 §5](Lecture-04.md) shows, audit the sibling kernels: this same shard collapsed the KDA decode grid to 12 blocks, and that went unnoticed until [PR #77](https://github.com/gittensor-ai-lab/sparkinfer-k3/pull/77) three functions away.

### 5.2 Report conservative rows as conservative

```text
   ctx     128     512     4k     32k     128k
   before  13.58   13.59   9.84   11.82    9.23
   after   16.42   16.48  11.31   14.46   15.93
```

> *"The 128k row is the full stack (both attention bands). The 128/512/4k/32k rows were swept on the MLA-only build and are therefore **conservative** — the KDA band is not in them."*

The shorter rows *understate* the change. Saying so is the opposite of the usual instinct, and it is what makes the rest of the table credible. A reviewer who finds one number quietly flattering discounts all of them; a reviewer who finds one number labelled "this is lower than reality because of how I measured it" believes the rest.

### 5.3 The launch guard, and why it belongs with an attention change

#49 bundled a one-line-per-phase `cudaGetLastError()` poll — 3 per layer, ~279 per token against ~2,300 launches — because **18 of the k3 launchers return `void` and nothing on the forward path polled for errors.** The consequence:

> *a failed launch leaves the previous layer's data in reused scratch and the model keeps emitting **fluent, wrong output**.*

That is [Lecture 10 §4.4](Lecture-10.md)'s family, and the reason it sits in an *attention* PR is that attention kernels are where launch configurations are most likely to exceed a limit: shared memory sized from context, grids sized from head count times splits, and both changing as the engine evolves. [PR #33](https://github.com/gittensor-ai-lab/sparkinfer-k3/pull/33) had just fixed one instance — shared memory sized by context length, failing silently past ~11.7k. #49 fixed the *class*.

And the guard immediately earned its place, catching a bug in the very PR that introduced it — the cross-device scratch pointer from §2:

```text
   [tp] ncclAllReduce(f32) rank 0: unhandled cuda error
   [tp] ncclGroupEnd: unhandled cuda error
   [k3-tp] all-reduce failed at layer 33
```

> *An error inside NCCL's collective, naming a layer with nothing wrong with it, pointing nowhere near the faulting kernel.*

CUDA errors are **sticky**: an error raised by one launch is returned by the *next* API call that checks. Without per-phase polling, the first thing to check is the collective — so a broken attention kernel at layer 5 is reported as an all-reduce failure at layer 33. Per-phase polling converts a misleading error into a located one.

> **In a long chain of unchecked async launches, the error surfaces at the first checkpoint, not the fault.** The value of frequent polling is not detection — it is *localization*.

---

## 6. Testing a path your gate cannot reach

The recurring structural problem in this lecture: the split and batched paths engage only above `kMlaSplitMinCtx` = 4096, and the correctness gate runs at ≤4096. **The gate cannot see the code these PRs changed.**

All three PRs address it, in three different and complementary ways.

**#49 — run your own parity check, at depth, and say why.**

> *"Neither existing gate covers the split path: `kimi_k3_numeric_test` is single-device, and `kimi_k3_eval.sh` scores `bench/refdata/hello.ids`, a short prompt that never reaches `kMlaSplitMinCtx` and therefore exercises the un-split path. **Both would pass this PR while testing none of it.**"*

So the author ran `compare_logits.py` at 128k across all 8 ranks — the numbers in §2.1. Recognizing that your green gates are *irrelevant* to your change, and building the missing check, is the whole discipline.

**#57 — use the blind spot as a targeted bit-identity check.**

> *"The gate runs at n_ctx 4, below `kMlaSplitMinCtx`, so it takes the same per-head kernel main takes — **which is what makes it a bit-identity check on the projection, `attn_res_mix` and KDA changes, all of which do run there.**"*

The same PR touched changes that *do* execute at short context. So the gate reporting `mean_kld` equal to `main` **to the last digit** (`0.0040455027115537685`) is a genuine bit-identity proof — of the subset it reaches. Knowing precisely which of your changes a gate validates, and claiming exactly that much, is better than either ignoring the gate or over-claiming it.

**#57 and #63 — extend the tests to reach the unreachable path.**

```text
   kimi_k3_numeric_test   35 cases, 0 failures.
     adds ctx 20000 and 12289 at 12 heads — the batched path,
     "otherwise unreachable on a device, it needs ctx > kMlaSplitMinCtx"
     and 20000 at 8 heads — the fallback

   cpu_reference_test
     models the batched + context-split SCHEDULE against a float64
     two-pass reference at K3's real dims (576 / 512 / 128)
     PLUS a NEGATIVE CONTROL that the slice merge needs its
          exp(m_i − m) rescale
```

Three things here. **A CPU reference at the real dimensions** lets you test a schedule that needs 20,000 tokens of context without 20,000 tokens of GPU memory — modelling the *schedule*, not the kernel. **Testing at the boundary values** (12289 is just past 12288) catches the off-by-one in the split arithmetic. And the **negative control** is the detail most test suites lack: a check that *deliberately removes* the `exp(m_i − m)` rescale and asserts the result is now wrong. Without it, a test that passes might be passing because the rescale is unnecessary at the tested shape — and you would not know your test has no power.

> **A test that has never failed has not been shown to work.** Add a negative control that breaks the invariant on purpose and assert the test catches it. This is [Lecture 10 §5](Lecture-10.md)'s "assert your assertions ran," at the unit-test level.

#63's fifth item is the same instinct: *"a CPU-reference test for a wide combine, **which the end-to-end accuracy gate structurally cannot reach.**"*

---

## 7. The ladder

```text
   decode @ 131,072, 8× H200, UD-IQ1_S, tp=8

   1.01  ──#49──▶  4.53  ──#57──▶  9.06  ──#63──▶  15.93   tok/s
         split         batch heads      shard heads
         context       12/block         across ranks
         4.49×         2.00×            1.73×

   llama.cpp on the same box: 18.44
   ⇒ three PRs took 5.5% of the reference to 86% of it.
```

Three PRs, 15.8×, and not one line of new attention mathematics. Every one of them found a different dimension to parallelize over — context, heads-within-a-block, heads-across-ranks — because the workload's natural axis (sequence, at decode) is size 1.

That is the lesson of the lecture, and it is the same one as [Lecture 04](Lecture-04.md) in a harder setting: **at batch 1, performance work is the search for an axis.**

---

## Lab — attack your own depth curve

1. **Plot the curve.** Decode tok/s at ≥5 context depths spanning short to your shipping context, for your engine and your reference. If your curve slopes and theirs does not, you have this lecture's problem.
2. **Isolate the depth-dependent term.** `time(long) − time(short)` is the cost that scales with depth. What fraction of your token is it? (§1.1)
3. **Read the attention grid.** `blocks` vs SM count, and whether any block's loop trip count is `O(n_ctx)`. Either alone is a defect; together they are this lecture.
4. **Find your A/A test.** Identify a gated-off path or a context below your thresholds where both arms run identical code. Measure it 10× and report the spread — **that is your noise floor**, and any result under it is not a result. (§3.2)
5. **Name a control phase.** Pick a phase your change cannot touch. Report it before and after. If it moves more than your noise floor, stop and fix the harness. (§3.1)
6. **Implement the context split.** Contiguous slices, partial `(m, l, acc)`, the rescaling combine, a minimum-context gate, per-device scratch, and neutral empty slices. Verify accuracy *improves* — if it degrades, your combine is wrong. (§2)
7. **Sweep your tile width with registers.** For heads- or rows-per-block, tabulate time, register count, and blocks/SM. Find the occupancy step and sit just below it. (§4)
8. **Check what your gate reaches.** For each change, state whether your correctness gate executes the modified path. For every "no," build the missing check — and add a **negative control** proving the check has power. (§6)
9. **Poll for launch errors per phase.** Count your `void` launchers. Add a per-phase `cudaGetLastError()` and confirm a deliberately over-sized launch is now reported at the right phase. (§5.3)

Pass criterion: a committed depth curve for both engines, a measured noise floor from an A/A path, one attention split with accuracy shown to improve, a tile-width table including register counts, and a test that reaches a path your gate cannot.

---

## Self-check

1. A kernel launches 96 blocks, each walking `n_ctx` positions serially, on a 132-SM GPU. Name the two independent defects and say which one explains the *slope* of the depth curve.
2. MLA compresses KV to one shared latent per position. Explain how a one-block-per-head kernel converts that advantage into a 96× penalty.
3. Derive the split-attention combine from the online-softmax recurrence. Why must the output projection run *after* the merge rather than per slice?
4. Your split path is not bit-identical and is *more* accurate than the unsplit version at long context. Explain, and say what it would mean if it were less accurate.
5. You measure +2.2% at a context where your new code path is gated off. What have you measured, and what does it imply about a +1.1% result elsewhere in the same table?
6. `ncu` is unavailable. You have two hypotheses — bandwidth-bound and latency-bound. Describe the intervention that is justified under both, and how its outcome tells you which was right.
7. 16 heads per block reads the cache less than 12 and is 12% slower. Give the mechanism and the two numbers you would report to prove it.
8. Head-sharding cuts per-rank attention work 8× and yields less than 8×. Give two reasons, one of which is not the collective.
9. A CUDA error from a bad attention launch at layer 5 is reported as an all-reduce failure at layer 33. Explain the mechanism and the fix.
10. Your test for a split-merge invariant has passed since it was written. Describe the one addition that would tell you whether it can fail at all.

---

## References

* **The PRs** — [#33](https://github.com/gittensor-ai-lab/sparkinfer-k3/pull/33) (MLA decode silently stops launching past ~11.7k), [#49](https://github.com/gittensor-ai-lab/sparkinfer-k3/pull/49) (split over context + the launch guard), [#57](https://github.com/gittensor-ai-lab/sparkinfer-k3/pull/57) (batch heads per block), [#63](https://github.com/gittensor-ai-lab/sparkinfer-k3/pull/63) (head-shard both bands), [#73](https://github.com/gittensor-ai-lab/sparkinfer-k3/pull/73) / [#77](https://github.com/gittensor-ai-lab/sparkinfer-k3/pull/77) (the grid collapses #63 created). Their bodies are the primary source.
* **Flash-Decoding** — [PyTorch blog, Oct 2023](https://pytorch.org/blog/flash-decoding/) — split the KV dimension across SMs, then combine. The technique #49 implements.
* **FlashAttention** — [arXiv:2205.14135](https://arxiv.org/abs/2205.14135) and **FlashAttention-2** [arXiv:2307.08691](https://arxiv.org/abs/2307.08691) — the online-softmax rescaling that §2's combine generalizes across blocks.
* **Online softmax** — Milakov & Gimelshein, [arXiv:1805.02867](https://arxiv.org/abs/1805.02867) — the `(m, l)` running-max formulation.
* **MLA (Multi-head Latent Attention)** — DeepSeek V2 [arXiv:2405.04434](https://arxiv.org/abs/2405.04434), V3 [arXiv:2412.19437](https://arxiv.org/abs/2412.19437) — the compressed-KV design whose reuse §1.1 is about.
* **FlashInfer** — [arXiv:2501.01005](https://arxiv.org/abs/2501.01005) (MLSys 2025) — the production attention engine with load-balanced split scheduling; what you use instead of writing this yourself, where you can.
* **Pairwise summation error bounds** — Higham, *Accuracy and Stability of Numerical Algorithms*, ch. 4 — the `O(n)` → `O(log n)` result behind §2.1.

Cross-references:

* [Part 2 Lecture 06 — Long context at 128K on Hopper](../Part%202%20-%20Dense%20at%20Hopper/Lecture-06.md) — KV scaling and chunked prefill, the production framing.
* [Part 3 Lecture 01 — Anatomy of a modern MoE](../Part%203%20-%20MoE%20at%20Blackwell/Lecture-01.md) — MLA's KV-compression mechanism.
* [Lecture 04 — Launch geometry](Lecture-04.md) — the grid collapses §5.1 creates, and the occupancy arithmetic.
* [Lecture 07 — Sharding 896 experts](Lecture-07.md) — the collective cost of §5's head-sharding.
* [Lecture 10 — Silently wrong](Lecture-10.md) — §5.3's silent-launch family.

---

## Current as of 2026-08

8× H200 SXM, `sm_90`, 132 SMs, CUDA 12.8+, UD-IQ1_S, tp=8, scored context 131,072. `kMlaSplitMinCtx` = 4096; 12 heads/block at 125 registers, 2 blocks/SM. Numbers from PRs #49 / #57 / #63; all single-binary env-gated A/B where stated. The axis-search framing, the A/A noise floor, the control phase, and the negative-control test are the durable content.

---

## Next

* Next: [Lecture 07 — Sharding 896 experts, and the Amdahl trap that followed](Lecture-07.md)
* Previous: [Lecture 05 — Fusion and the activation-quantization discipline](Lecture-05.md)
* Up: [Part 4 — Optimizing a Real Engine](README.md)
