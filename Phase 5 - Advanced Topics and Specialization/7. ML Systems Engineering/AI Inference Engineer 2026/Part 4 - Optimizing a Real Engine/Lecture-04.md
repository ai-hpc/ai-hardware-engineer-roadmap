# Part 4 · Lecture 04 — Launch Geometry: Grids, Occupancy, and 327 Norms per Token

## Overview

The cheapest wins in this case study were not better algorithms. They were **the same arithmetic, launched in a different shape.**

An H200 has **132 streaming multiprocessors**. A kernel launched with 12 blocks uses 9% of it. A kernel launched with one block uses 0.8%. In batch-1 decode, where every natural parallel axis has collapsed to size 1, launching tiny grids is not an unusual mistake — it is the *default* outcome of writing a kernel that is correct.

Three PRs in this lecture, worth +4.0%, +6.7%, and a 21% step, contain no new mathematics between them. One widened a block. One added a grid axis. One replaced a hardcoded constant with a derivation. What makes them worth a lecture is the *reasoning* that found them and the discipline that proved they changed nothing but speed.

By the end you should be able to look at a kernel's launch configuration and say whether it is starving the machine, distinguish starvation from contention in a profile, and prove a launch-shape change is bit-identical rather than merely passing a tolerance.

---

## 1. The arithmetic nobody does

Two numbers, multiplied. That is the whole diagnostic.

```text
   H200 (sm_90):  132 SMs

   grid    1 block   →   0.8%  of SMs have work
   grid   12 blocks  →   9.1%
   grid   48 blocks  →  36.4%
   grid  128 blocks  →  97.0%   (but a tail: 132 would be one wave)
   grid  256 blocks  →  two waves, each ~full
```

The reason this is not done routinely is that **the symptom does not look like a shape problem.** A starved kernel shows low achieved bandwidth, low FLOPs, and a duration longer than the work justifies — exactly the fingerprint of a memory-bound kernel. Every throughput metric agrees that you are bandwidth-limited. None of them mentions that 91% of the GPU is idle.

The distinguishing measurement is grid dimensions against SM count, which no throughput counter reports. You have to go and look.

### 1.1 Why decode is structurally prone to this

From [Lecture 03 §4.1](Lecture-03.md): batch-1 decode produces tensors whose sequence dimension is **1**. Batch, sequence, and tile — every axis a training kernel parallelizes over — are gone. What remains is heads, hidden width, and experts.

So the recurring fix in this lecture and [Lecture 06](Lecture-06.md) is one move under three names: **manufacture a parallel axis where the workload's natural axes ran out.** Split over context. Split over value tiles. Split over expert groups × FFN bands. Widen the block until the width itself is the parallelism.

---

## 2. The barrier was right; the width was wrong

[**PR #115**](https://github.com/gittensor-ai-lab/sparkinfer-k3/pull/115) — *"widen the last single-block launch — 327 norms/token were running on 128 threads."*

The finding: `rms_norm_f32` ran as **a single block of 128 scalar threads** over widths up to 7168 — and it ran **327 times per token**.

Now the part that makes this a good teaching case. A naive reading is "one block is a bug, parallelize it." That reading is wrong, and the PR says so:

> *One block is correct (the mean must complete before any element is scaled). What was wrong is the width.*

RMS norm needs a full reduction over the row before any element can be scaled. Within a single block that is a `__syncthreads()`; across blocks it would need a global barrier or a two-pass kernel. **One block is the right structure.** The defect was that the block was 128 threads wide when the row was 7168 elements.

The fix works on **bytes in flight**, not on block count:

```text
   BEFORE   128 scalar threads          →  ~512 B in flight
   AFTER    block sized to the width,
            float4 loads where legal    →  1024 threads, ~16 KB in flight
                                           at hidden 7168

   the grid is still 1 block. the barrier is untouched.
   what changed is how much memory the block has outstanding.
```

### 2.1 The half that was deliberately *not* widened

Here is the detail that makes this a great teaching case, and it is easy to miss: **the sum of squares still runs on exactly 128 threads.** Only the elementwise apply after it was widened. The kernel's own comment explains why, and it is worth reading closely:

```c
float acc = 0.0f;
if (threadIdx.x < 128) {                              /* the REDUCTION stays narrow */
    for (int d = (int)threadIdx.x; d < n; d += 128) acc += x[d] * x[d];
}
const float ss  = block_sum<BLOCK>(acc, shm);
const float inv = rsqrtf(ss / (float)n + eps);        /* then the APPLY goes wide */
```

> Changing how many threads contribute to `ss` changes the float32 **association** of the sum of squares, which changes `inv`, which changes **every output element** by ~1e-7 relative — enough to push KL vs `main` over the 2× ratchet while still clearing the absolute KL bar.

And this is not hypothetical: an earlier measurement of this branch drew an **`accuracy-regression` label for exactly that reason** (ctx2048 at 4.63× main's KL). The final version splits the kernel at the line where association stops mattering:

```text
   the REDUCTION  → association-sensitive → must keep the same thread count
   the APPLY      → elementwise, once `inv` is fixed → widen freely, byte-identical
```

Two supporting facts make the split exactly bit-identical. Idle threads above 128 contribute `0` to `block_sum`, and `x + 0` is an IEEE identity — so the longer warp-tree sum over 1024 threads produces the same bits as the 4-warp one. And at `BLOCK == 128` with no vectorized path the code stays on the *original* kernel rather than the specialization, so a reviewer can check the equivalence in one line instead of trusting it.

> **Split a kernel at the boundary between its reduction and its elementwise apply.** The reduction's width is part of its numerics; the apply's is not. Widening the second gives you the bandwidth win with byte-identical output, while widening the first is an accuracy change you would have to defend. This is the same association argument as [Lecture 06 §2.1](Lecture-06.md) and [Lecture 08 §8](Lecture-08.md), used here to decide *where not to optimize*.

Two more details of craftsmanship worth copying:

* **The vector path is conditional and the scalar path is exact.** `float4` is used only when `n % 4 == 0` and all three pointers are 16-byte aligned. The fallback is not a degraded approximation — it is the original exact path. A fast path that is only *sometimes* legal must fail into correctness, not into "close enough."
* **It stays on the existing launch mechanism.** Launches go through `k3_pdl_launch`, keeping the kernel on the programmatic-dependent-launch path the rest of the decode chain uses ([Lecture 08](Lecture-08.md)). An optimization that quietly opts out of the engine's launch infrastructure gives back what it gains.

### 2.2 The same idea scored `none` twice

This change was submitted before as [#71](https://github.com/gittensor-ai-lab/sparkinfer-k3/pull/71) and measured three times:

| round | tok/s | top-1 | KL | % over frontier | tier |
|---|--:|--:|--:|--:|---|
| 1 | 16.25 | 1.0 | 0.0065 | +1.2% | `eval:none` |
| 2 | 17.53 | 1.0 | 0.0035 | +0.4% | `eval:none` |
| 3 | 26.57 | 1.0 | 0.0026 | +1.8% | `eval:none` |

Three real, correct, positive measurements — all inside the 2% significance gate, all scoring zero. The change only earned a tier when it was resubmitted as #115 against a different frontier.

Two lessons, both from [Lecture 02](Lecture-02.md) and worth seeing land on a specific diff: **a win is measured against a state, not for all time**, and **the significance gate is an absolute tok/s bar that moves as you improve.** Notice also what the tok/s column implies: the branch measured 16.25 in round 1 and 26.57 in round 3 while never clearing 2% over the frontier — so `main` itself got roughly 63% faster underneath this PR while it sat open, and the same diff's absolute contribution stayed inside the gate the whole time.

---

## 3. Starvation, and how to tell it from contention

[**PR #77**](https://github.com/gittensor-ai-lab/sparkinfer-k3/pull/77) — *"value-tile the KDA decode step — 12 blocks on 132 SMs (+6.7%, bit-identical)."*

The finding, stated with unusual precision:

```text
   kda_decode_step's grid IS n_head.

   #63 head-sharded KDA, taking n_head from 96 to 12 at tp=8.
   → 12 blocks on a 132-SM H200 = 9% of the device,
     consuming 11.8% of ALL GPU kernel time.
```

### 3.1 The diagnostic that identifies starvation

This is the most transferable paragraph in the PR, and it deserves to be quoted and then unpacked:

> *Profiled at ctx 131072, the kernel moves ~3.1 MB per launch and takes **87 µs against a ~0.9 µs bandwidth floor**, with a **982 ns stddev**. Tight variance and a 95× gap to bandwidth is not contention and not arithmetic — it is starvation of blocks.*

The inference chain:

```text
   1.  bytes moved ÷ device bandwidth  =  0.9 µs   ← what it SHOULD cost
   2.  measured                        =  87 µs    ← 95× off
   3.  stddev                          =  982 ns   ← 1.1% of the mean

   contention would be NOISY   (other work interfering → high variance)
   arithmetic would be VISIBLE (FLOPs would account for the time)
   a 95× gap that is REPEATABLE TO 1% is a structural property
       of the launch, not of the machine's state.

   ⇒ the blocks are not there.
```

**Variance is the discriminator.** A kernel that is slow because it is fighting other work is slow *inconsistently*. A kernel that is slow because it only ever occupies 9% of the device is slow with metronome regularity. Reporting a standard deviation alongside a mean is what makes that distinction available — and most profiling writeups omit it.

> **Steal this.** Always report the spread, not just the mean. `87 µs ± 1 µs` and `87 µs ± 30 µs` are different diagnoses with the same headline.

### 3.2 Manufacturing the axis, and proving it is legal

The fix adds a second grid axis over **value tiles**, taking 12 blocks to 48. The justification for why this is allowed is the part to study, because "add a grid axis" is only safe if the work actually decomposes:

> *Every step is **column-local**: for state column `j`, the decay, `sk[j] = Σᵢ S[i][j]·k[i]`, `d[j]`, the rank-1 update and `o[j]` all touch only column `j` plus the per-head `k`/`q`/`g` vectors, which are read-only and shared. So columns split across blocks with **no extra reduction and no duplicated state traffic** — a block reads exactly its own `BV` rows of the j-major buffer.*

Three properties make a split cheap, and this one has all three:

| Property | Why it matters | Here |
|---|---|---|
| No cross-block reduction | Otherwise you pay a combine pass | Columns are independent |
| No duplicated reads | Otherwise traffic × number of splits | Each block reads its own rows |
| Layout unchanged | Otherwise you pay a transpose | `S[i][j]` at `s[j*D+i]` still holds |

Compare with the *context* split in [Lecture 06](Lecture-06.md), which does require a combine pass — and is still worth it. Knowing which kind of split you have tells you what to expect.

**The tile width was not tuned — it was cited.** `BV = 32` is what the reference implementations use for exactly this kernel: FLA's `fused_recurrent_gated_delta_rule`, which vLLM ships for decode, launches `grid = (cdiv(V, BV) · N · HV)` with `BV = 32`.

> **Look up what the reference implementation does before sweeping.** A citation is cheaper than a tuning run and generalizes better than a number you found on your own hardware. If your value disagrees with the reference's, that is interesting and worth understanding — but start from theirs.

**A second, separable win in the same PR.** Pass 1 stored `S·exp(g)` to global memory and pass 2 re-read it; now pass 2 recomputes the product instead. That removes **a full write of the state per launch — a quarter of the kernel's global traffic** — plus one staging loop and one barrier per chunk. Recomputing a cheap product to avoid a round trip through HBM is the memory-bound trade in its purest form.

Net result: **52.96 → 49.63 ms/token** at the scored 128k context, **18.88 → 20.15 tok/s**, **+6.7%**, output bit-identical.

---

## 4. Derive the constant; do not pin it

[**PR #73**](https://github.com/gittensor-ai-lab/sparkinfer-k3/pull/73) — *"derive the MLA slice floor so head-sharding cannot strand the grid."*

The MLA decode kernel had a **flat 1024-token slice floor** — a hardcoded minimum slice size. Perfectly reasonable when written. Then head-sharding changed the head count, and the interaction stranded the grid below the occupancy target sitting in the adjacent constant.

The fix derives the floor from the shape. The result:

```text
   grid at ctx 131,072:   128 blocks  →  256 blocks   (on a 132-SM part)
   measured against the kMlaBlocksPerSm = 2 target beside it

   main      54.53 ms/token   18.34 tok/s
   this PR   52.44 ms/token   19.07 tok/s      +4.0%
```

Note the target: `kMlaBlocksPerSm = 2`, so 264 blocks is "two per SM" and 256 is essentially there. **Two blocks per SM rather than one** is deliberate — it lets one block's memory latency overlap with another's compute, which a single resident block per SM cannot do.

> **A hardcoded launch constant is a cached derivation, and caches go stale.** Any constant whose correct value depends on `n_heads`, `tp_size`, `head_dim`, or context length should be computed from those, with the occupancy target it is aiming at named next to it. Write the *intent* (`kMlaBlocksPerSm = 2`) and derive the number.

---

## 5. Each optimization plants the next occupancy bug

The most important structural insight in this lecture is the chain between #63, #73, and #77, and the PR author names it plainly:

> *"That regression is mine. **#63 budgeted the MLA split cap (`kMlaSplitBudget = 96 · kMlaMaxSplits`) for precisely this failure** — head-sharding collapsing a grid — and left the KDA kernel three functions away at one block per head. #73 has since refined the MLA side; nobody had looked at this one."*

The sequence:

```text
   #63   head-shard both attention bands       →  n_head 96 → 12 at tp=8
         (an eval:xl win — see Lecture 06)
         author ANTICIPATED grid collapse and budgeted the MLA split cap for it

   #73   MLA slice floor derived from shape    →  the MLA side, fixed properly

   #77   KDA decode grid IS n_head             →  12 blocks. three functions
                                                  away from the fix, unnoticed
```

The author foresaw the exact failure mode, defended one kernel against it, and missed the sibling kernel a few functions away. This is not carelessness; it is what parallelism-reducing changes *do*.

> **Any change that divides a parallel axis across ranks reduces every grid derived from that axis.** After sharding a dimension, audit **every** kernel whose grid mentions it. Not the one you were thinking about — all of them. `grep` for the dimension, not for the kernel.

And a corollary about attribution: #63 was correctly scored `xl`. Its measured end-to-end win was real. It also created a latent −6.7% that took two more PRs to recover. Both facts are true, and a ladder that records only the first is not lying so much as incomplete. This is why the case study's per-PR receipts and its frontier-on-`main` measurements are separate numbers ([Lecture 01 §5.1](Lecture-01.md)).

---

## 6. One binary, two arms

Every PR in this lecture used the same measurement technique, and it is worth adopting wholesale.

Each change is behind a **runtime environment gate**, so "before" and "after" are the *same compiled binary* with a different flag:

```text
   #77:   off  = SPARKINFER_K3_KDA_VT=0    →  main's launch geometry
          vt   = SPARKINFER_K3_KDA_VT=1    →  the new grid

          "Both arms are the same binary [...] so nothing but the grid
           differs and no rebuild separates them."
```

What this eliminates: build-flag drift, compiler version differences, driver state, link-order effects, and the whole class of "we rebuilt and something else changed." Those are not hypothetical — they are why [Lecture 02](Lecture-02.md) records a PR that had to *rebuild every eval from scratch with a pinned compiler.*

[**PR #127**](https://github.com/gittensor-ai-lab/sparkinfer-k3/pull/127) takes it to its conclusion. It bundles a dozen changes, each individually gated, and publishes the **complete restore set** that returns `main`'s exact behaviour on one binary:

```text
   SPARKINFER_K3_MOE_WEPS=0 IQ1S_PACK=0 MLA_PVT=0 B2WIDE=0
   COLL_BLOCK=512 WARP_TARGET=1792 ROUTER_REG=0 HEAD_1BAR=0
   RES_FUSE=0 RMSG=0 RMSU=0 COLL_CAP=36 FUSED4_ROWS=4 MLA_KLEN=0
```

> *"Every change is gated at runtime; the restore set [...] returns `main`'s exact behaviour in place on one binary, **without a revert**."*

That last clause is the operational payoff. If any of those twelve changes turns out to regress something in production, the mitigation is an environment variable — not a revert, a rebuild, and a redeploy.

### 6.1 ABBA interleaving, and reporting per-rep numbers

#127's measurement discipline is the other half:

```text
   A_this   16.88 / 16.87 / 16.88   →  16.88 ms   (59.26 tok/s)
   B_main   20.46 / 20.50 / 20.46   →  20.46 ms   (48.87 tok/s)
```

Interleaved A/B/B/A, three pairs, **individual reps printed**. Publishing the reps rather than the mean lets a reader see the spread themselves and is what makes §3.1's variance argument checkable by someone else.

### 6.2 Two instruments will disagree — pick one per comparison

#77 reports the same change measured two ways, and explains the discrepancy rather than hiding it:

| instrument | before | after | delta |
|---|--:|--:|--:|
| `kimi_k3_eval.sh` (the scoring instrument) | 18.88 | 20.15 | **+6.7%** |
| `kimi_k3_tp_bench`, 136 tokens, 3 reps | 18.86 | 20.03 | **+6.2%** |

> *"My 32-token bench and the harness disagree on absolute speed — the harness times 8- and 136-token runs and takes the marginal differential, ~7 s of sustained load against ~1.6 s — so an A/B is only meaningful when both arms use one instrument."*

**Both deltas are valid; neither absolute number is comparable to the other's.** The rule: a delta is only meaningful within a single instrument, and the instrument that decides the tier is the one that matters. Reporting both, with the reason they differ, is strictly better than picking the flattering one.

The same PR also flags something it *could not* resolve — a model fingerprint mismatch against the eval bot's runs (`8c3b548967ff7583` vs `4128f23a2dcea136`), with the reasoning that the weights are almost certainly identical because `mean_kld` reproduces to all 18 digits. Then: *"I have not proven that, and would rather flag it than leave it out of a PR quoting node numbers."* That is the correct handling of an unexplained observation in a performance claim.

---

## 7. Proving bit-identity as bytes

Two of these three PRs claim **bit-identical**. That is a strong claim and it needs a strong check.

**The check is `cmp`, not a tolerance.** From #77:

> *`cmp` of the logits dumps under `SPARKINFER_K3_KDA_VT=1` and `=0` reports **byte-identical** [...] `kimi_k3_numeric_test` seeds a random non-zero state and checks both `out` and the state against a float64 reference [...] **But it is a tolerance check, so the `cmp` above — not the test — is what the bit-identical claim rests on.**"

A numeric test that passes at `1e-6` tells you the change is *acceptable*. It cannot tell you it is *identical*. If you claim bit-identity, compare bytes.

#127 does the same at two depths, and reports it as a per-depth table:

```text
   ctx128: IDENTICAL   ctx256: IDENTICAL   ctx512: IDENTICAL
   ctx1024: IDENTICAL  ctx2048: IDENTICAL  ctx4096: IDENTICAL

   and at 2x the deepest graded depth, 8192 real ids, full 93 layers:
   BIT-IDENTICAL: default build == all-gates-off, one binary
```

### 7.1 The FMA trap

The finest detail in this lecture, from #77's write-elision. It is worth reading twice.

Pass 1 used to store a product to shared memory; the new version feeds it onward directly. The PR uses `__fmul_rn` rather than `*`:

> *The old pass 1 stored the product to shared, which forced it to be materialised as a **rounded f32**, and feeding an add directly would let `nvcc` (`--fmad=true` by default) **contract it into an `fma` and skip that rounding**. That would move the last bits while passing every tolerance check in the suite.*

Unpacked:

```text
   OLD:  tmp = a * b;        // rounded to f32, because it went to shared
         ...
         out = tmp + c;      // add of a ROUNDED product

   NAIVE NEW:  out = a * b + c;
                     ^^^^^^^^^ nvcc contracts to fma(a, b, c)
                               → the product is NOT rounded first
                               → different last bits

   CORRECT NEW:  out = __fmul_rn(a, b) + c;
                       ^^^^^^^^^^ forces the round-to-nearest f32 product,
                                  reproducing the old rounding exactly
```

Fused multiply-add is *more* accurate — it keeps the full intermediate precision. But "more accurate" is not "identical," and a bit-identity claim is a claim about *reproducing the previous rounding*, including its losses.

> **When you eliminate a round trip through memory, you may also be eliminating a rounding.** Removing a store removes a materialization, and the compiler will then happily contract the arithmetic across the seam. If you want the old bits, force the old rounding — `__fmul_rn`, `__fadd_rn`, or `-fmad=false` scoped to the file.

This is also the clearest possible illustration of why a tolerance check cannot police bit-identity: an FMA contraction moves the last bits and *improves* accuracy. It would pass every numeric test in the suite while making the claim false.

---

## 8. When a launch-shape change is not free

Not every reshape is bit-identical, and #73 is the honest counter-example.

> *Not bit-identical: **raising the split count reassociates the online-softmax combine**, as the split path has always done.*

Splitting attention over more slices changes the *order* in which partial softmax results are combined. Floating-point addition is not associative, so a different split count gives different last bits. That is inherent to split-K/split-context attention, not a defect ([Lecture 06](Lecture-06.md)).

And then the most valuable sentence, because it is about the limits of the gate:

> *"The end-to-end accuracy gate cannot see this change — it scores a short reference prompt, and splitting only engages above `kMlaSplitMinCtx` — so **that gate passing is not evidence either way here.**"*

The parity gate runs at ≤4096 tokens; the split path engages only at long context. So the gate is *silent* on this change — and the author says so rather than presenting a green gate as validation. Compare [Lecture 02 §7](Lecture-02.md)'s stated limit: *"4096 is not the 131,072 you are scored at."* Here is a specific change that lands squarely in that blind spot.

> **A passing gate is only evidence if the gate exercises the code you changed.** Before citing green CI, ask which of your gates actually reached the new path. If none did, say so.

#127 handles a similar situation the opposite way — by *construction* rather than by measurement. Its expert-weight threshold drops routed-expert contributions whose renormalized router weight is below `8e-2`, but only past position 16384:

* Below 16k the gate holds it **off**, so it is byte-inert at every depth the parity suite grades.
* Shallow-depth thresholding **was measured and found harmful** — KL 1.1e-1 at ctx 2048 ungated — and is excluded by construction rather than by convention.
* Past 16k it acts only on the flattened tail of the router's own weight distribution, and the output norm absorbs the scale.

> *"The gate is a quality guard, not an optimisation."*

That is the right shape for an approximation you cannot fully validate: **bound it to the regime where you have an argument, prove it inert everywhere you can measure, and document the measurement that made you bound it.**

---

## Lab — audit your launch geometry

1. **Dump every kernel's grid and block.** Instrument your launches or read them from a profiler. Produce a table: kernel, grid, block, calls per token, total time per token.
2. **Add the occupancy column.** `blocks ÷ SMs`. Flag everything under 50%.
3. **Sort by wasted SM-time** — `time_per_token × (1 − occupancy)`. This ranks by *recoverable* time, not by duration, and the order will differ from your profiler's.
4. **For your top offender, do the #77 calculation.** Bytes moved ÷ device bandwidth = floor. Measured time. Ratio. **And the standard deviation over ≥10 reps.** Tight variance plus a large ratio means starvation; wide variance means contention.
5. **Find an axis.** For that kernel, identify a dimension you can split with no cross-block reduction, no duplicated reads, and no layout change. If all three fail, you have a genuinely harder problem — say which one failed.
6. **Gate it at runtime.** Implement the new geometry behind an environment variable so both arms are one binary. Measure ABBA-interleaved, three pairs, and publish the individual reps.
7. **Prove the correctness claim you are making.** If bit-identical: `cmp` the output bytes, at more than one depth. If not: say what reassociates, and state whether your gate can even see the change.
8. **`grep` for your sharded dimensions.** For every dimension your parallelism divides, find every kernel whose grid mentions it (§5). Report the list, even the ones that are fine.

Pass criterion: a committed table ranking your kernels by recoverable SM-time, one reshaped kernel measured single-binary ABBA, and a correctness claim backed by the right kind of check.

---

## Self-check

1. A kernel takes 87 µs where bandwidth says 0.9 µs, with a 982 ns standard deviation over ten reps. Name the diagnosis and the two alternatives you have ruled out, with the reasoning for each.
2. RMS norm needs a full-row reduction, so one block is structurally required. Give two ways to make it faster that do not add blocks, and say what each raises.
3. In that same kernel, the reduction stays on 128 threads while the apply widens to 1024. Explain why widening the reduction would be an accuracy change, trace the propagation from `ss` to the output, and give the two facts that make the widened apply exactly bit-identical.
4. Your team shards attention heads 8 ways for a large end-to-end win. Write the audit you run immediately afterwards, and say what you `grep` for.
5. You remove a store to shared memory and feed the product straight into an add. Your tolerance test still passes but `cmp` reports differing bytes. What happened, and what is the one-token fix?
6. `kMlaBlocksPerSm = 2` rather than 1. Why would you target two resident blocks per SM instead of one?
7. A PR claims bit-identical and cites a numeric test passing at `1e-6`. Write the two-sentence review comment.
8. Your accuracy gate is green, but the code path you changed only engages above 16k context and the gate runs at 4k. What may you conclude from the green gate?
9. A correct, positive, well-measured optimization scores `none` three times and then `s` with no code change. Explain, and say what you do with the branch in the meantime.

---

## References

* **The three PRs** — [#115](https://github.com/gittensor-ai-lab/sparkinfer-k3/pull/115) (widen the single-block norm), [#77](https://github.com/gittensor-ai-lab/sparkinfer-k3/pull/77) (value-tile the KDA decode step), [#73](https://github.com/gittensor-ai-lab/sparkinfer-k3/pull/73) (derive the MLA slice floor), plus [#127](https://github.com/gittensor-ai-lab/sparkinfer-k3/pull/127) (small-grid widening + the restore set) and [#71](https://github.com/gittensor-ai-lab/sparkinfer-k3/pull/71) (the same idea as #115, scored `none`). Their bodies are the primary source for this lecture.
* **Flash Linear Attention (FLA)** — [github.com/fla-org/flash-linear-attention](https://github.com/fla-org/flash-linear-attention) — `fused_recurrent_gated_delta_rule`, the reference launch shape (`BV = 32`) that #77 cites rather than tunes. Shipped in vLLM for gated-delta decode.
* **Gated DeltaNet** — [arXiv:2412.06464](https://arxiv.org/abs/2412.06464) — the recurrence whose column-locality makes #77's split legal.
* **CUDA occupancy** — [CUDA C++ Best Practices Guide § Occupancy](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/#occupancy); the Occupancy Calculator API for computing achievable blocks per SM.
* **FMA contraction** — [CUDA C++ Programming Guide § Mathematical Functions](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#mathematical-functions-appendix) and the `--fmad` / `__fmul_rn` / `__fadd_rn` intrinsics. The mechanism behind §7.1.
* **"What Every Computer Scientist Should Know About Floating-Point Arithmetic"** — Goldberg, [ACM Computing Surveys 1991](https://dl.acm.org/doi/10.1145/103162.103163) — non-associativity, the root of §8's reassociation.

Cross-references:

* [Lecture 03 — Diagnosis](Lecture-03.md) — the occupancy ceiling in context, and why the binding ceiling moves.
* [Lecture 06 — Attention at 128k](Lecture-06.md) — the context split, which *does* need a combine pass, and #63's head-sharding that set up §5.
* [Lecture 08 — Graph-resident decode](Lecture-08.md) — `k3_pdl_launch` and the launch infrastructure #115 stays on.
* [Lecture 10 — Silently wrong](Lecture-10.md) — why the `cmp`-versus-tolerance distinction in §7 is not pedantry.

---

## Current as of 2026-08

8× H200 SXM, `sm_90`, 132 SMs, CUDA 12.8+. Measurements from PRs #73 / #77 / #115 / #127 at the scored 131,072-token context, UD-IQ1_S, tp=8; all single-binary env-gated A/B. `BV = 32` per FLA's gated-delta decode kernel. The occupancy arithmetic, the starvation-vs-contention discriminator, and the bit-identity discipline are the durable content.

---

## Next

* Next: [Lecture 05 — Fusion and the activation-quantization discipline](Lecture-05.md)
* Previous: [Lecture 03 — Diagnosis: launch-bound, bandwidth-bound, or comm-bound?](Lecture-03.md)
* Up: [Part 4 — Optimizing a Real Engine](README.md)
