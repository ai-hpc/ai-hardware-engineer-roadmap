# Part 4 · Lecture 08 — Graph-Resident Decode: Killing the Launch Bill for Good

## Overview

By this point in the ladder the engine had done the work of Lectures 04 through 07: the grids were sized, the projections fused, attention split three ways, the experts sharded on two axes. And **~30 kernel launches per layer × 93 layers ≈ 2790 launches per token** were still being issued from the host, one at a time, per rank.

At 21 tok/s that is 47 ms per token and ~17 µs of budget per launch — comfortable. At 26 tok/s it is 38 ms and ~14 µs. The launch bill does not shrink as you optimize; it becomes a larger fraction of a smaller number, until it *is* the number.

The answer is to stop launching: capture the whole decode step as **one CUDA graph per rank** and replay it. [PR #89](https://github.com/gittensor-ai-lab/sparkinfer-k3/pull/89) did that for **+22.7%**, and captured a graph of **4,257 nodes per rank**.

But the most important consequence is not the 22.7%. It is that **graph capture changes which other optimizations are worth doing** — and one kernel split that had been correctly rejected before became a win immediately after.

By the end you should be able to make a decode step capturable, test a graph correctly (one replay proves nothing), and recognize when an optimization's value depends on the launch mechanism rather than on the arithmetic.

---

## 1. What capture buys — and the half nobody mentions

A CUDA graph records a sequence of kernel launches, memory operations, and their dependencies once, then replays the whole thing with a single submission. Launch overhead becomes graph-construction cost, paid once.

The measured result, in a five-paired-rep A/B with order alternated per rep, all on **one binary** with the features env-gated:

```text
   everything off  (GRAPH=0, QACT_HOIST=0)        48.60 ms    sem 0.55
   graph on, single-kernel combine                39.68 ms
   graph on, split combine  (the PR)              38.30 ms

   47.1 → 38.37 ms/token   ·   21.24 → 26.06 tok/s   ·   +22.7%
   captured: 4257 nodes/rank, 185 collectives/token, mla splits=32
```

Now look at the `sem` column — the standard error — because it carries the finding that does not appear in the CUDA documentation:

> *The "everything off" arm [...] is the noisy one (**eager mode carries host-side variance that capture removes**) — hence the sem of 0.55 against 0.02 for the captured arms.*

**Capture cut the run-to-run variance by ~27×.** That makes sense once stated: in eager mode every token's timing includes host scheduling, driver work, and CPU contention, all of which vary. A replayed graph is one submission of a fixed plan — the host is barely involved, so there is little left to vary.

Two practical consequences:

* **Graph capture is a measurement improvement, not only a performance one.** Your noise floor drops, which means smaller genuine wins become measurable. Some of the case study's later 3–5% results are only detectable *because* the engine is graph-resident.
* **It changes what "p99 latency" means.** Tail latency in a launch-bound decode loop is substantially host-side jitter. Removing the host from the inner loop compresses the tail more than it moves the mean.

[PR #86](https://github.com/gittensor-ai-lab/sparkinfer-k3/pull/86) reports the same phenomenon from the other side, and much more dramatically:

> *the **eager baseline on this model has been observed to move ~15% across sessions** while the optimised side stays stable, so a delta taken from two separate passes is not comparable. Interleaving pins both arms to the same box state.*

A 15% session-to-session drift on the eager arm is larger than most tiers. It is why every PR in this part interleaves arms *within one session* rather than measuring back to back — and why an un-captured baseline is a bad thing to compare against across time.

---

## 2. What blocks capture: host values that get frozen

A graph records **the values that were live at capture time**. Any address, size, or count computed on the host becomes a constant baked into the graph — and replay uses the captured constant forever.

Four such values blocked capture here. Each one is a template for the class:

**A pointer computed from the token position.** The MLA K-cache row address `cache + position × key_length` was computed on the host and fed to two `cudaMemcpyAsync` calls.

> *Replay would rewrite **one row forever.**"*

The fix: a kernel derives the row from a **device-resident** `d_pos`, and *"it moves the same bytes in the same order."* The general move — **promote the varying value to device memory and compute the address on the device.**

**A size passed by value.** The attention context length was a kernel argument. The fix relies on an observation about how it is used: *"`n_ctx` is only ever a loop bound in the three MLA kernels, so they read `*d_pos + 1`. No arithmetic changed."* A value used only as a loop bound can be read from memory at kernel start; a value used to size a *grid* cannot (see below).

**The state update itself.** The position increment was host-side. It now runs *inside* the captured region:

> *the single line that makes a replay a **different token**.*

That is the crux of graph-resident autoregressive decode. A graph replayed identically produces an identical result — which for a decode step is useless. Exactly one thing must change per replay, and the increment that changes it has to be *inside* the graph, operating on device state.

**A value that decides the grid, which cannot be promoted.** `splits` sizes the grid and selects which kernel runs.

> *A graph can change neither, so it stays a host decision and **the graph is re-captured when the plan moves.** The driver's check and the launcher call the same `k3_mla_decode_plan()`, so a probe cannot drift from what launches.*

This is the honest boundary of graph capture: **grid dimensions and kernel identity are structural.** They cannot be data-dependent within one graph. The answer is a small set of graphs, re-captured when the plan changes — and the discipline that makes it safe is that the code deciding *whether to re-capture* and the code deciding *what to launch* call the same function. Two separate implementations of "which plan applies" is a divergence waiting to happen.

> **The audit.** Before capturing, list every host-computed value that reaches a launch: pointers, sizes, counts, loop bounds, grid dimensions. For each, decide — promote to device memory, or make it a capture key. Anything you miss becomes a constant, and the symptom is a model that is correct on the first token.

---

## 3. Operations that are illegal inside a capture

Separately from frozen values, some CUDA operations cannot occur during capture at all — synchronous copies and allocations among them. Two were hiding in lazy initialization paths:

```text
   ensure_iq1s_tables() / ensure_iq2xs_tables()
       upload lattice tables via SYNCHRONOUS cudaMemcpyToSymbol
   k3_mla_split_scratch()
       cudaMallocs when capacity is short
```

Both are classic lazy-init: do it on first use, cache it forever. Both are invisible in eager mode. And note *where* they fired:

> *both firing on the **first MoE layer** under IQ1_S — so capture recorded the leading dense layer cleanly and died at layer 1, while `cudaGetLastError` blamed the MoE dispatch from the next launch.*

Two lessons, and the second is from [Lecture 06 §5.3](Lecture-06.md): CUDA errors are **sticky**, so the failure was attributed to whatever ran next. The reported location was not the fault.

The fix is pre-warming both at init — and then a piece of engineering judgment worth adopting:

> *A capture failure now **disables capture and re-issues the token eagerly**, rather than failing the run.*

Graph capture is an optimization. A failure to capture should cost you performance, not correctness or availability. This is the same "slow, never wrong" principle as [Lecture 05 §4](Lecture-05.md), applied at the level of an execution strategy.

> **Steal this.** Audit for lazy initialization before capturing: first-use table uploads, on-demand allocations, cached descriptor creation, memoized handles. Pre-warm all of it at startup. Then make capture failure a graceful downgrade to eager, and log it.

---

## 4. Testing a graph: one replay proves nothing

The single most important line in [#89](https://github.com/gittensor-ai-lab/sparkinfer-k3/pull/89) for anyone about to do this work:

> *At short context, where the MLA plan is `splits=1`, the logits are **byte-identical** between `SPARKINFER_K3_GRAPH=1` and `=0` across **capture plus 7 replays** — **one replay proves nothing, because the failure mode here is correct on replay 1 and wrong from replay 2.**"*

Consider the frozen-pointer bug from §2. At capture time the address is correct — it was computed for the current position. The first replay reproduces exactly what capture did, which was right. From the second replay onward it writes the same row again, and the KV cache silently stops advancing.

```text
   capture   → correct  (the value was live)
   replay 1  → correct  (identical to capture)
   replay 2  → WRONG    (the frozen value is now stale)
   replay 3+ → WRONG, and the output is still fluent
```

A test that captures and replays once passes. So does a test that generates two tokens. **You need at least three, and more is better** — #89 used capture plus seven.

And note what makes it a strong check: **byte-identical against the un-captured path**, not "close." Graph replay executes the identical kernels with identical parameters, so:

> *anything but an exact match would be a bug rather than drift.*

Graph capture is one of the few optimizations where bit-identity is not an aspiration but a *definition*. If replay differs from eager, something is stale.

The accuracy figure agrees to full precision: `mean_kld` **0.005910210209380935**, described as *"main's value to every digit."*

---

## 5. The A/A control that prevented a wrong conclusion

This is the best debugging anecdote in the case study, and the author included it specifically so nobody repeats it:

> ***A byte-compare at 131072 is not a valid check, and I want to save the next person the detour.** `--seek` advances the position without filling the KV cache, so the `splits=32` path reads ~131k rows of **uninitialised memory** and two runs of the **same** arm do not match each other. I concluded from that comparison that capture was broken at 128k, **before running the `off`-vs-`off` control that shows the test is meaningless.**"*

The chain of events: run a byte-compare at 128k, see a mismatch, conclude graph capture is broken at long context, start debugging a bug that does not exist. The thing that resolved it was running the *same configuration against itself* — and finding that it also mismatched.

```text
   graph ON  vs  graph OFF   at 128k  →  MISMATCH   "capture is broken!"
   graph OFF vs  graph OFF   at 128k  →  MISMATCH   ...the TEST is broken.
```

This is the A/A control from [Lecture 06 §3.2](Lecture-06.md) doing different work. There it established a noise floor; here it establishes that **the measurement itself has no discriminating power.** Same technique, and it should be the *first* thing you run when a comparison fails unexpectedly, not the last.

The underlying cause is worth remembering on its own: the seek-based benchmark leaves the KV cache **zeroed and unfilled** ([Lecture 02 §7](Lecture-02.md)). That is legitimate for *timing*, because the MLA reduction is dense and data-independent. It is meaningless for *bitwise comparison*, because the kernel reads uninitialized memory whose contents differ between runs. A shortcut that is valid for one kind of measurement and invalid for another — exactly why the repo insists every shortcut document what it makes faithful and what it does not.

> **When a comparison produces an unexpected result, run it against itself before you believe it.** A/A first, then A/B.

---

## 6. Capture changes which optimizations are worth doing

Here is the conceptual payload of the lecture.

`mla_decode_combine_kernel` merged the per-slice attention partials *and* projected the merged latent through `wv_b`, all in `grid = n_head` blocks. With attention head-sharded a rank owns 12 heads — **12 blocks on a 132-SM part, ~9% of the machine**, for a kernel reading 256 KB of `wv_b` per head. Its thread 0 also walked `O(splits)` serially to build the rescale weights while the rest of the block waited at a barrier ([Lecture 05 §7](Lecture-05.md)'s serial prologue).

The fix is to split it in two so both halves fill the grid:

```text
   merge     grid (n_head,  8)   writes the normalised latent to per-device scratch
   project   grid (n_head, 16)   each block owns a slice of v_dim and reads only
                                 its own wv_b rows → no redundant weight traffic

   12 blocks  →  96 and 192.
```

And now the sentence that matters:

> ***The extra kernel is precisely why the single-kernel form was right before.*** *A second launch per MLA layer is 24 more launches per token, and at 93 layers that mattered. **It stops mattering once the decode is captured as one CUDA graph, where the launch is a baked graph node.** This is downstream of the capture in this same branch, not an independent change.*

The single-kernel form was not a mistake. It was **correct under the old launch economics** and became wrong the moment those economics changed. Measured on its own: **39.68 → 38.30 ms, +3.5%**, with `sem 0.02`, `n=5`, `t=69.0`.

This generalizes into a rule that reframes a lot of performance work:

> **Graph capture changes the price of a kernel launch from ~5 µs to ~0.** Every optimization you previously rejected because "it adds a launch" should be reconsidered. Fusion and fission are opposites, and which one wins depends on the launch mechanism — not on the arithmetic.

Concretely, once you are graph-resident: splitting kernels to improve occupancy becomes cheap; specialized variants instead of branchy general kernels become cheap; extra small kernels that improve grid shape become cheap. Conversely, **fusion is worth less than it was**, because part of what fusion was buying was launch elimination — and the graph already bought that.

And note the discipline in the split itself: *"Summation order is unchanged in both stages — the slice sum still runs `i` ascending, the projection dot still strides `r` by 32 across the warp — so this is bit-identical to the kernel it replaces rather than a re-rounding of it."* Splitting a kernel in two does not have to move bits, if you preserve the accumulation order across the seam ([Lecture 04 §7](Lecture-04.md)).

---

## 7. Overlap, PDL, and leave-one-out attribution

[**PR #90**](https://github.com/gittensor-ai-lab/sparkinfer-k3/pull/90) — *"five decode fast paths and programmatic launch"* — **48.69 → 37.75 ms/token, 20.54 → 26.49 tok/s, +29.0%.**

Six changes, *"each in its own translation unit behind its own toggle, measured as one binary."* And rather than a cumulative ladder, the attribution is **leave-one-out**:

| arm | ms/token | tok/s | factor worth |
|---|--:|--:|--:|
| base (every toggle off) | 47.48 | 21.06 | — |
| **all on** | **39.44** | **25.35** | — |
| `SPARKINFER_K3_PDL=0` | 44.06 | 22.70 | **PDL 4.62 ms** |
| `SPARKINFER_K3_KDA_IP=0` | 42.10 | 23.75 | KDA step 2.66 ms |
| `SPARKINFER_K3_PROJ_1BAR=0` | 39.90 | 25.06 | projections 0.46 ms |

Leave-one-out measures each factor's **marginal** value in the presence of all the others, which is what you actually want to know — a cumulative ladder credits whichever change you happened to apply first with all the shared benefit. And the base arm is validated against reality: *"`base` lands within 1% of main measured separately (47.48 vs 46.93), **which is what makes these deltas mean what they say.**"* An all-toggles-off arm that does not reproduce a real `main` build is not a baseline.

**PDL is the largest single factor at 4.62 ms** — roughly 10% of the token. Programmatic Dependent Launch is a Hopper (`sm_90`) feature that lets a dependent kernel begin its prologue — scheduling blocks onto SMs, allocating registers, faulting in its first instructions — while its predecessor is still draining; `cudaGridDependencySynchronize()` inside the successor is where it waits for the predecessor's writes to become visible.

The crucial point is that it is **not** a substitute for graph capture, and the PR says so in capitals. The two attack different halves of the same gap:

```text
   GRAPH CAPTURE  removes the HOST cost of submitting a launch
   PDL            removes the DEVICE-side gap between two kernels
                  that are already submitted
   → they stack.
```

A K3 token issues roughly 4,000 dependent kernels per rank in 47 ms — about **12 µs per kernel for work whose bytes justify one or two.** Graphs take the submission cost out; PDL takes the spin-up cost out. This is why [PR #115](https://github.com/gittensor-ai-lab/sparkinfer-k3/pull/115) was careful to keep its widened norm on the PDL path ([Lecture 04 §2](Lecture-04.md)) — an optimization that opts out of the launch infrastructure gives back what it gains.

One safety note worth carrying into your own code: the sync call *"MUST be executed by every thread before ANY read of memory the predecessor wrote,"* and placing it too late *"is a data race that produces plausible wrong numbers, not a crash."* The discipline that makes it reviewable is placement — every kernel calls it as its first statement, so the ordering claim is a one-line property rather than an argument about which loads alias what. And because the call is a **no-op unless the kernel is launched programmatically**, a kernel carrying it is byte-identical when launched the ordinary way — which is exactly what makes `SPARKINFER_K3_PDL=0` an honest A/B on one binary rather than two different programs.

[PR #114](https://github.com/gittensor-ai-lab/sparkinfer-k3/pull/114) — *"overlap independent work inside each decode layer"*, **41.31 → 47.00 tok/s** — is the third member of this family: within a layer, find operations with no data dependency and let them run concurrently on separate streams. Graph capture, PDL, and intra-layer overlap are three mechanisms aimed at the same waste — **the machine being idle between things it could have been doing.**

### 7.1 One test binary per factor

> *One test binary per factor, so a dropped factor takes its own test with it. `k3_kda_step_cpu_test` needs **no device** — the only coverage that runs on a fork's PR. The GPU tests use K3's **real** dims, which `kimi_k3_numeric_test`'s shrunk dims (`kv_lora` 64, `key_length` 80) never reach.*

Three practices worth copying. **Test-per-factor** means reverting one change removes exactly its own test, with no orphaned assertions. **A device-free test** is the only thing that can run in CI on an untrusted fork's PR — so the most portable coverage should target your most intricate logic. And **shrunk test dimensions do not exercise real code paths**: a numeric test at `kv_lora 64` never reaches the shapes that a 512-wide latent produces, which is why the GPU tests use the real dimensions even though they cost more.

---

## 8. Which reductions are order-independent

[#90](https://github.com/gittensor-ai-lab/sparkinfer-k3/pull/90) improved accuracy, and the explanation is a small masterclass:

> *`mean_kld` **0.687× main's**, `top1` 1.0 on both [...] That is by construction: **the KDA reduction becomes a tree instead of a 128-long sequential chain**, and the CPU model measures the tree at **1.13e-7** against float64 where the sequential schedule measures **2.41e-7**. Everything else is bit-identical, checked by `memcmp` — **including the quantiser, whose scan is a max over magnitudes and therefore order-independent.**"*

Two distinct ideas:

**A tree reduction is more accurate than a chain**, for the same reason splitting attention over context improved accuracy in [Lecture 06 §2.1](Lecture-06.md): error grows as `O(n)` for sequential accumulation and roughly `O(log n)` for pairwise. Restructuring for parallelism improves numerics as a side effect. The claim is backed by a *float64 reference measurement* of both schedules, not asserted.

**Knowing which operations are order-independent tells you which reorderings are free.** The quantizer's scan is a **max over magnitudes** — and `max` is associative and commutative over the reals *and* over floats, exactly. So reordering it is bit-identical by construction, and `memcmp` confirms it. Compare floating-point *addition*, which is neither.

> **Steal this.** Before reordering a reduction, classify the operator. `max`, `min`, and bitwise ops are exactly associative — reorder freely, claim bit-identity. Floating-point `+` and `×` are not — reordering changes bits, and you must say so and check on tolerance. This one distinction resolves most "can I claim bit-identical?" questions.

---

## 9. Being more precise than your reference is a liability

[**PR #86**](https://github.com/gittensor-ai-lab/sparkinfer-k3/pull/86) — five decode-path changes, **33.67 → 35.99 tok/s (+6.9%)** — narrows the MLA latent KV cache from **F32 to F16**. Halving the cache's bytes at 128k is a large bandwidth win, and one of the two constants [PR #107](https://github.com/gittensor-ai-lab/sparkinfer-k3/pull/107) later had to re-tune was invalidated by exactly this change ([Lecture 05 §7](Lecture-05.md)).

The interesting part is the accuracy argument. Three measurements:

| comparison | mean KLD | top-1 |
|---|--:|--:|
| `main` vs reference capture | 5.146e-03 | 100.00 % |
| **this PR** vs reference capture | **7.513e-03** | 100.00 % |
| this PR vs `main` | 4.600e-03 | 100.00 % |

The PR is *further* from the reference than `main`. And the explanation inverts the intuition:

> *The divergence is the F16 cache, which moves **toward** the reference rather than away from it — **llama.cpp's default `type_k` is F16, so `main` was carrying more KV precision than the implementation it is scored against.**"*

The engine was keeping the cache in F32 while the reference kept it in F16. Since correctness here is defined as *agreement with the reference* ([Lecture 02 §2.1](Lecture-02.md)), the extra precision was a source of *disagreement*. Matching the reference's precision reduces the divergence that matters even though it reduces the absolute precision.

> **When your correctness criterion is agreement with a reference implementation, "more accurate than the reference" is a defect in the metric you are graded on.** Decide deliberately whether you are chasing fidelity to the mathematics or fidelity to the reference — they are different targets, and only one of them is your gate.

Two more habits from the same PR:

**Isolate your own contribution.** *"`main`'s own 0.0051 against the same refdata on this node is the baseline to read this against — the weights staged here are not bit-identical to the snapshot `hello.spkl` was captured from, so **both arms carry that offset.** The number that isolates this PR is the 0.0046 in the third row."* When both arms share a systematic offset, the A-to-B comparison is the only clean signal — the same three-way logic as [Lecture 07 §6.1](Lecture-07.md).

**A tool printing FAIL is not necessarily a gate failing.** *"Note `compare_logits.py` prints `FAIL` for all three rows: its own bar is 1e-5, the tolerance for two runs of the **same** implementation. It is not the accuracy gate, and it **fails `main` against refdata too.**"* A tool's built-in threshold was designed for a different question. Knowing which of your tools *is* the gate, and what each one's default bar means, prevents both false alarms and false confidence.

And one more instance of [Lecture 07 §5.1](Lecture-07.md)'s dead-fast-path pattern: change 2 in that PR is *"a one-line routing fix rather than new kernel work. `k3_proj_q8_multirow_1bar` and its `x_pre_q8` parameter both already existed; **nothing reached them on the hoisted path.**"* Another optimization that was merged, correct, and unreachable.

---

## Lab — make your decode step graph-resident

1. **Count your launches.** Kernels per layer × layers × ranks. Divide your step time by it. If the per-launch budget is under ~10 µs, capture is worth doing (§Overview).
2. **Audit host-computed values.** List every pointer, size, count, loop bound, and grid dimension that reaches a launch and is computed on the host. Classify each: *promote to device*, or *capture key*. Do not skip loop bounds (§2).
3. **Find the state update.** Identify the one thing that must differ per replay, and move it inside the captured region, operating on device memory (§2).
4. **Audit lazy initialization.** Grep for `ensure_*`, `get_or_create`, first-use allocations, cached handles. Pre-warm all of it at init (§3).
5. **Capture, and make failure graceful.** On capture failure, disable capture, re-issue eagerly, and log. Test it by deliberately leaving one lazy init un-warmed (§3).
6. **Test with ≥3 replays**, byte-compared against the eager path. Confirm that a test with a single replay would have passed by deliberately freezing a pointer (§4).
7. **Run the A/A control.** Before believing any mismatch, compare a configuration against itself. If that mismatches, your test is broken, not your code (§5).
8. **Measure the variance, not just the mean.** Report the standard error of both arms. Expect the captured arm to be far tighter (§1).
9. **Revisit your rejected optimizations.** List every change you declined because "it adds a launch." Re-evaluate each under graph residency and measure the most promising (§6).
10. **Attribute leave-one-out.** If you land several factors, publish a leave-one-out table and validate your all-off arm against a real baseline build (§7).

Pass criterion: a captured decode step with a node count, a ≥3-replay byte-identity test, an A/A control in your test suite, standard errors for both arms, and one previously-rejected fission re-measured under capture.

---

## Self-check

1. Your captured graph produces a correct first token and fluent-but-wrong output from the third. Name the bug class and the two most likely specific causes.
2. Why does the position increment have to be inside the captured region, and what does that imply about where the position must live?
3. `n_ctx` is used as a loop bound; `splits` sizes the grid. One can be promoted to device memory and one cannot. Explain the difference and give the strategy for the second.
4. Graph capture cut your standard error from 0.55 to 0.02 ms. Give two consequences for how you run your benchmark suite from now on.
5. A lazy table upload fires on the first MoE layer and breaks capture, but the error names the MoE dispatch. Explain the mechanism (§3, and [Lecture 06 §5.3](Lecture-06.md)).
6. You byte-compare two runs of the *same* build at 128k and they differ. Before concluding anything, what do you check, and what is the likely cause given how the benchmark reaches 128k?
7. A single-kernel design was correct; then you captured the decode step and splitting it into two became a win. Explain, and name two other optimization classes whose value changes the same way.
8. Which of these can you reorder and still claim bit-identical: a sum of f32, a max over magnitudes, a product of f32, a bitwise OR? Justify each.
9. Your engine keeps the KV cache in F32; the reference keeps it in F16. Your parity metric gets *worse* the more precision you keep. Explain, and say what you would do.
10. `compare_logits.py` prints FAIL for your build, for `main`, and for the reference against itself. What is happening, and what should you conclude?

---

## References

* **The PRs** — [#89](https://github.com/gittensor-ai-lab/sparkinfer-k3/pull/89) (per-rank CUDA graphs + split MLA combine; the capture blockers; the A/A detour), [#90](https://github.com/gittensor-ai-lab/sparkinfer-k3/pull/90) (PDL and five fast paths; leave-one-out attribution; the tree reduction), [#86](https://github.com/gittensor-ai-lab/sparkinfer-k3/pull/86) (F16 latent cache; the precision-as-liability argument; session drift), [#114](https://github.com/gittensor-ai-lab/sparkinfer-k3/pull/114) (intra-layer overlap). Their bodies are the primary source.
* **CUDA Graphs** — [CUDA C++ Programming Guide § CUDA Graphs](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#cuda-graphs) and [Getting Started with CUDA Graphs](https://developer.nvidia.com/blog/cuda-graphs/) — capture semantics, stream capture, and the list of operations illegal during capture (§3).
* **Programmatic Dependent Launch** — [CUDA C++ Programming Guide § Programmatic Dependent Launch and Synchronization](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#programmatic-dependent-launch-and-synchronization) — the `sm_90` feature worth 4.62 ms in §7.
* **CUDA graphs in production LLM serving** — vLLM's graph capture (`--enforce-eager` disables it) and TensorRT-LLM's CUDA-graph decode path; the production equivalents of §1. See [Part 1 Lecture 05](../Part%201%20-%20Fundamentals/Lecture-05.md).
* **Megakernels / persistent runtimes** — the logical endpoint of this lecture: instead of capturing many kernels, run *one* resident kernel. See [MLSys Deep Dives Lecture 03](../../MLSys%20Deep%20Dives/Lecture-03.md) on TileRT and the megakernel approach.
* **Pairwise vs sequential summation error** — Higham, *Accuracy and Stability of Numerical Algorithms*, ch. 4 — the `O(n)` → `O(log n)` result behind §8.

Cross-references:

* [Lecture 03 §3 — Diagnosis](Lecture-03.md) — the launch-bill arithmetic, and why the binding ceiling moves.
* [Lecture 04 §6 — Launch geometry](Lecture-04.md) — the single-binary env-gated A/B discipline every PR here uses; §7's `k3_pdl_launch`.
* [Lecture 05 §7 — Fusion and activation quantization](Lecture-05.md) — the serial prologue that §6's combine kernel also had, and #107's constants invalidated by §9's F16 cache.
* [Lecture 06 §3.2 — Attention at 128k](Lecture-06.md) — the A/A control in its other role, and the split whose combine §6 improves.
* [Lecture 10 §4.4 — Silently wrong](Lecture-10.md) — sticky CUDA errors and work that silently does not happen.

---

## Current as of 2026-08

8× H200 SXM, `sm_90`, 132 SMs, CUDA 12.8+, UD-IQ1_S, tp=8, scored context 131,072. Captured decode step: 4,257 nodes/rank, 185 collectives/token, `mla splits=32`. PDL worth 4.62 ms of a 47.48 ms token. Graph-arm standard error 0.02 ms against 0.55 ms eager. The capture audit, the ≥3-replay test, the A/A-before-believing rule, and the "capture changes the price of a launch" reframing are the durable content.

---

## Next

* Next: [Lecture 09 — The phase you forgot: batched prefill](Lecture-09.md)
* Previous: [Lecture 07 — Sharding 896 experts, and the Amdahl trap that followed](Lecture-07.md)
* Up: [Part 4 — Optimizing a Real Engine](README.md)
