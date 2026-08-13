# Part 4 · Lecture 05 — Fusion and the Activation-Quantization Discipline

## Overview

The single largest step in the case study's history was **3.54 → 9.94 tok/s** — a 2.77× improvement, bit-identical, from **five independent defects in one kernel**. Not a new algorithm. Five ways the same GEMV was leaving the machine on the table.

This lecture is about that kernel and its neighbours: the quantized projection path that every non-expert matrix multiply in the model goes through, and the discipline of **activation quantization** — deciding once, correctly, where a vector gets encoded to int8 and who is allowed to read it.

It is also where a specific and repeated observation lands. Three separate PRs found a kernel achieving **about 7 GB/s on a 4.8 TB/s part** and concluded, correctly:

> *That is not work, it is launch overhead.*

By the end you should be able to audit a quantized GEMV for the five defects in §2, decide whether an activation should be hoisted or fused, and — most importantly — architect a shared-buffer optimization so that a mistake makes it *slow* rather than *wrong*.

---

## 1. Where decode's time actually goes

In a sparse MoE at batch 1, the expert weights are the bulk of the *bytes*. But the **projections** — Q/K/V/gate, the MLA down/up projections, the router, the shared experts, the LM head — are the bulk of the *kernels*, and they run on every one of 93 layers.

The case study's Q8_0 projection GEMV is described as *"the kernel behind every non-expert K3 projection, and now the dominant decode cost."* That is the shape to expect: after the obvious memory win (quantizing weights), the cost migrates to the many small matrix-vector products that surround the big ones.

Two structural facts about that kernel drive everything below:

```text
   Q8_0 weights + f32 activation:
       the activation must be QUANTIZED before an int8 dot product.
       → somebody has to encode it. once? or once per consumer?

   K3 has FOUR consumers reading the SAME activation on every KDA layer:
       attn_q, attn_k, attn_v, ssm_g   all read s.normed at [qkv, H]
       → four launches over one vector, and four re-quantizations of it.
```

Everything in §2–§6 is a consequence of those two lines.

---

## 2. Five defects in one kernel

[**PR #25**](https://github.com/gittensor-ai-lab/sparkinfer-k3/pull/25) — *"Q8_0 projection path — 3.54 → 9.94 tok/s at ctx 128k, bit-identical."* Each defect is worth studying separately; the ordering below is roughly increasing subtlety.

### 2.1 A 34-byte struct defeats vectorized loads

```text
   BlockQ8_0 is 34 bytes, alignment 2.
   → qs[] starts at 34·b + 2
   → 4-byte aligned only for ODD b.  never uniformly.

   nvcc cannot widen a scalar qs[j] loop, so it emits
   ONE ld.global.s8 PER ELEMENT.  32 loads where 8 would do.
```

The fix is `get_int_b2()`, which rebuilds the same 4 bytes from two **always-aligned `uint16` loads**. And the tell that this is the right fix: *"ggml's CUDA path carries the same helper for the same reason."*

> **Steal this.** A quantization block format with a size that is not a multiple of 4 will silently defeat every vectorized load in every kernel that touches it. You cannot change the on-disk format, so the fix is a helper that reconstructs wide words from the alignment you actually have. Check your block struct's size and alignment before you profile anything.

### 2.2 Quantizing the activation and then not using integer arithmetic

The quantized-activation path was computing its int8 dot product with **32 scalar IMAD** instructions — *"giving up the point of having quantised the activation."*

`__dp4a` performs a 4-element int8 dot-product-accumulate in one instruction. The entire reason to encode activations to int8 is to reach it. A pipeline that quantizes and then does scalar integer multiply-adds has paid the quantization cost and collected none of the benefit.

> **Check that your fast path reaches the instruction it exists for.** Quantization is a means to `dp4a` / IMMA / tensor-core paths. If the profile shows scalar IMAD after a quantize step, the quantization is pure overhead.

### 2.3 The activation is bigger than the weights

The counterintuitive one, and the reason multi-row exists:

```text
   projection with N = 12288 output rows, K = 7168:

   weight traffic       ~94 MB
   ACTIVATION traffic   ~344 MB     ← re-read once per output row

   the activation is 3.7x the weights.
```

At batch 1 with one block per output row, each block loads the whole activation. The fix puts **4 rows in a block** and reuses each activation load across them — cutting activation traffic ~4×.

And the constraint that keeps it honest: dispatched only for **N ≥ 1024**, because multi-row also *divides the grid*. At `ssm_beta`'s N=96, four rows per block would drop the grid to **24 blocks** — straight into [Lecture 04](Lecture-04.md)'s starvation. Two optimizations pulling in opposite directions, resolved by a threshold rather than by picking a side.

> **Reuse and occupancy trade against each other.** Any "process R rows per block" change divides your grid by R. It is a win at large N and a regression at small N, so it needs a dispatch condition — not a global flag.

### 2.4 A fixed block width against variable work

```text
   BLOCK was 128 threads regardless of available work.
   the loop strides  b < blocks_per_row:

     ssm_f_b   (K=128)  →  124 of 128 threads idle — 96.9% —
                           on all 69 KDA layers
     attn_q_b           →  62.5% idle, on all 24 MLA layers
```

This is [Lecture 04](Lecture-04.md)'s occupancy problem inside a block rather than across a grid, and the PR notes it is *the same defect PR #7 fixed in the MoE dispatch, still present here, surviving for the same reason*:

> *Idle threads contribute exact zeros, so output is correct and only occupancy is wrong.*

That sentence is the whole mechanism of why this class of bug persists. Nothing is wrong. A reduction over mostly-zero contributions gives the right answer. There is no test that fails, no assertion to trip, no numerical drift — the only symptom is that you paid for a 128-thread block and used four threads.

### 2.5 Four launches over one activation

`attn_q`, `attn_k`, `attn_v` and `ssm_g` read the same normed hidden state at the same shape on every KDA layer. Fusing them means **one activation load feeds 8 dot products**, and removes **207 launches per token**.

The safety argument is worth noting because it is a *data-flow* argument, not a convention: `ssm_g` can be hoisted to join the group *"safe because `s.normed` is written once at `attn_norm` and never touched again in the block."* The producer and the consumers are in straight-line code with nothing between them. §4 is about what happens when that is not true.

### 2.6 The win holds where it matters

| context | before | after | speedup |
|---|--:|--:|--:|
| 128 | 3.54 | 8.89 | 2.48× |
| 4k | 3.13 | 8.03 | 2.25× |
| 32k | 3.57 | 9.77 | 2.72× |
| **128k** | **3.54** | **9.94** | **2.77×** |

> *"The scored row is the strongest one, which is the point: this is not a short-context win that evaporates where the model is actually used."*

Sweeping context and showing the win *grows* with depth is what distinguishes an optimization from a benchmark artifact. Compare [Lecture 03 §5](Lecture-03.md): a change that helps at 4k and vanishes at 128k has found something about your test, not your engine.

---

## 3. A "free" operation with a launch bill

Now the recurring finding, in three PRs, with three sets of numbers.

[**PR #67**](https://github.com/gittensor-ai-lab/sparkinfer-k3/pull/67):

```text
   quantize_q8_0:  59,696 launches,  8.7% of GPU time
                   4.9 µs each to move ~36 KB
                   ≈ 7 GB/s   on a 4.8 TB/s part

   "That is not work, it is launch overhead."
```

[**PR #81**](https://github.com/gittensor-ai-lab/sparkinfer-k3/pull/81), profiled at ctx 131072:

```text
   quantize_q8_0:  61,848 launches,  7.9% of GPU kernel time
                   4.19 µs each to move ~28 KB
                   ≈ 6.7 GB/s  on a 4.8 TB/s part
```

Two independent measurements agreeing that ~8% of all GPU time went to a kernel running at **0.15% of peak bandwidth.** Quantizing a 28 KB vector is genuinely trivial work — which is exactly why it was invisible. Nobody profiles the cheap op.

> **The diagnostic: divide bytes moved by time taken, and compare to peak.** A kernel achieving a tiny fraction of peak bandwidth on a tiny payload is not doing memory work — it is *being launched*. Its cost is proportional to how many times you call it, and the fix is to call it fewer times, not to make it faster.

This is [Lecture 03](Lecture-03.md)'s launch-bound diagnosis localized to a single kernel, and it generalizes: **any small helper kernel invoked per-consumer rather than per-value is a launch-count bug wearing a bandwidth costume.**

### 3.1 The fix, and its size

[#81](https://github.com/gittensor-ai-lab/sparkinfer-k3/pull/81) quantizes once per *activation* and hands the buffer to each consumer:

```text
   61,848 launches (7.9%)   →   40,104 launches (4.5%)
                                −302 per token per rank

   50.01 → 47.71 ms/token at 128k
   20.00 → 20.96 tok/s        +4.8%,  bit-identical
```

[#67](https://github.com/gittensor-ai-lab/sparkinfer-k3/pull/67) attacks the same waste from the other direction — *fusing* the four consumers so there is only one of them:

```text
   57.33 → 55.13 ms/token at 128k
   17.45 → 18.14 tok/s        +4.0%,  bit-identical
```

Both are correct and they compose. **Hoisting** removes redundant *producer* calls; **fusion** removes redundant *consumer* launches. §4 explains why one of them is architecturally safer.

---

## 4. Pass the buffer; do not cache it

This is the most important section in the lecture, because it is about how to make an optimization that *cannot* be silently wrong.

Sharing a quantized activation between consumers had been attempted in this repository before, as a **cache**:

```text
   THE FAILED VERSION
     cache the quantized activation inside the projection,
     keyed on x, guarded by "was the scratch last written from
     this pointer?"

   RESULT SHIPPED:   top1 0.0     mean_kld 0.937
                     — every token wrong, while running faster.

   THE BUG:  the guard tracked which POINTER the scratch came from,
             never whether the BYTES behind it still matched.
```

A pointer-identity guard is not a validity guard. The same buffer address can hold different contents at different points in a forward pass, and it usually does. And note the signature from [Lecture 10](Lecture-10.md): **wrong, and faster.**

The successful version changes the *architecture*, not the guard. Three properties, and each one closes a specific failure:

| Property | What it prevents |
|---|---|
| **The buffer is a parameter.** `k3_quantize_act_f32` writes it, `k3_proj_q8act_f32` reads it, and the window between them is straight-line code on one stream. | *"There is no invariant for a later change to violate silently."* |
| **`act_q8` is a separate allocation from `proj_q8`.** Every un-hoisted projection still quantizes into `proj_q8`. | Sharing one buffer would let an interleaved call overwrite the hoisted bytes — *"the exact aliasing that broke the cached version."* |
| **`proj_h` falls back** to the full path whenever the hoist did not apply. | **"A missed hoist is slow, never wrong."** |

And one more, which prevents the two paths from diverging over time: `k3_proj_ggml_f32` is reimplemented as `k3_quantize_act_f32` + `k3_proj_q8act_f32`, so *"there is one implementation and the hoisted and per-call paths cannot drift apart."*

[#67](https://github.com/gittensor-ai-lab/sparkinfer-k3/pull/67) makes the same point about fusion, and states it best:

> *Fusing needs no promise from anyone. The activation cannot go stale between its producer and its consumer **because they are the same launch.** Correctness stops depending on a claim about the surrounding code, which is what made the reuse version unsafe to keep.*

> **The design rule.** When two pieces of code must agree about the contents of shared memory, prefer — in this order: (1) **make them the same launch** so no agreement is needed; (2) **pass the buffer explicitly** so the agreement is visible in the signature; (3) *never* **cache with a guard**, because the guard is a proxy for the invariant and proxies drift. Design so that a failure of the optimization is a *slowdown*, not a wrong answer.

That last clause deserves emphasis. "A missed hoist is slow, never wrong" is a property you *engineer*, by making the fast path an opt-in refinement of a correct path rather than a replacement for it. It is the same shape as [Lecture 04](Lecture-04.md)'s conditional `float4` load falling back to an exact scalar path.

---

## 5. The prediction gap as a debugging instrument

[#81](https://github.com/gittensor-ai-lab/sparkinfer-k3/pull/81)'s second commit is the finest piece of methodology in the entire case study, and it takes a moment to appreciate.

The first version of the change measured **+1.8%**, against a *predicted* saving of 462 launches per token per rank. Measured: **−233**.

Most engineers would bank the +1.8% and move on. Instead:

> *That gap named the bug rather than being noise.*

The reasoning:

```text
   proj_f32_kernel runs 11,592 times
                 = 161 per token per rank
                 = 69 × 2  +  24 × 1        ← EXACTLY

   ⇒ ssm_f_a and ssm_beta are f32-WEIGHTED and were never
     quantized at all. So on all 69 KDA layers the hoist
     ADDED a launch and removed none — while
     k3_proj_ggml_f32_x4 re-quantized the same bytes independently.

   FIX: give the fused group an x_pre_q8 parameter.
        removes exactly 4,968 = 69 × 8 × 9 further launches.

   +1.8%  →  +4.1%
```

Two techniques here, both cheap and both underused:

**Predict a countable quantity, not just a speedup.** "This should remove 462 launches per token per rank" is *falsifiable* in a way "this should be faster" is not. When the count came in at 233, there was a specific residual — 229 — to explain.

**Factor the residual against the model's structure.** 161 per token per rank decomposing as exactly `69 × 2 + 24 × 1` is not a coincidence; 69 and 24 are the KDA and MLA layer counts. That factorization *is* the diagnosis: two calls per KDA layer, one per MLA layer, from a path that was never quantized.

> **Steal this.** Instrument a counter, not just a clock. Then check the counter against the arithmetic of your model — layers, heads, experts, ranks. A launch count that does not factor cleanly into your architecture's dimensions is telling you about a path you have not accounted for. This is the technique that converts "it's a bit faster than I hoped" from a shrug into a bug report.

---

## 6. Fusion regressions hide behind boolean guards

[#67](https://github.com/gittensor-ai-lab/sparkinfer-k3/pull/67) opens with a small horror story:

> *Enabling Q8 activations in [#63](https://github.com/gittensor-ai-lab/sparkinfer-k3/pull/63) **silently disabled** the four-way projection fusion on every KDA layer, because the guard read `!ggml_qact_proj && ...`*

A feature flag was turned on. A dispatch guard elsewhere read *"only fuse if we are NOT using quantized activations."* So enabling the new path disabled the old optimization on all 69 KDA layers — no error, no warning, just a silent return to four separate projections and four re-quantizations of the same vector.

This is the [Lecture 04 §5](Lecture-04.md) pattern in a different guise. There, sharding an axis shrank every grid derived from it. Here, enabling a path disabled every optimization whose guard excluded it. Same shape:

> **Every `if (!new_feature)` in a dispatch condition is a fusion or fast path that your new feature turns off.** After enabling anything, `grep` for negations of its flag. And when you write such a guard, leave a note saying what the Q8 counterpart would need to be — because someone will enable it.

The recovery is the counterpart implementation: `k3_proj_ggml_f32_x4`'s Q8-activation twin, plus re-enabling the fusion. **+4.0%**, which is really *recovering* 4.0% that #63 had quietly spent.

And the honest scope statement, which prevents over-claiming:

> *Only the KDA `q/k/v/g` group qualifies: `k3_proj_ggml_f32_x4` requires four identical shapes. `ssm_f_a`, `ssm_beta` and `ssm_f_b` read the same activation but at different `N`, and the MLA and MoE projections keep their own quantisations — so this removes about **a fifth** of the 59,696 launches, not all of them.*

Fusion requires shape agreement. Four consumers of one activation at *different* output widths cannot share a launch without a more general grouped kernel. Knowing that your fix addresses 20% of the instances is what tells you there is more on the table.

---

## 7. The serial prologue

[**PR #107**](https://github.com/gittensor-ai-lab/sparkinfer-k3/pull/107) adds a fourth family to §2's list, and it is one that no occupancy metric shows:

> *Four decode kernels spent their prologue in **a single thread walking an array with dependent global loads** while the rest of the block waited at a barrier.*

The grid is fine. The block is fine. Achieved occupancy looks fine. And for the duration of the prologue, one thread issues a chain of dependent global loads — each one waiting on the previous to compute its address — while 1023 threads sit at `__syncthreads()`.

```text
   thread 0:   load → use → load → use → load → ...   (latency-chained)
   threads 1..1023:   ▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓ waiting at barrier ▓▓▓▓▓▓▓▓▓▓
```

A dependent-load chain cannot be hidden by more warps, because there is only one warp doing anything. Parallelizing the prologue — having the whole block cooperatively walk the array — converts a latency chain into a bandwidth problem, which the hardware is good at.

The PR pairs this with something already familiar from [Lecture 04 §4](Lecture-04.md): *"retuning two MLA constants that **#86's F16 latent cache invalidated**."* A previous PR halved the latent cache's element size, which changed the arithmetic that two tuned constants were derived from. Another cached derivation gone stale, from a different direction.

There is a discrepancy in this PR worth naming rather than papering over: **its title claims +15.9% at 128k while its body measures +4.5%** — 43.30 → 45.25 tok/s, four interleaved loads alternating `main`/branch, with the two `main` passes and the two branch passes agreeing to 0.05%. The bot's own round measured 41.39 tok/s. I quote the body's number here because it is the one with a method attached, and flag the gap because a title and a measurement disagreeing is exactly the kind of thing [Lecture 02](Lecture-02.md) says to notice. Accuracy *improved* alongside it: mean KLD **0.00751 → 0.00669**.

Two further wins in the same PR are worth a line each, because both are the "dead fast path" pattern from §6 in a new guise. The **LM head** — a 163,840 × 7,168 projection, at Q8_0 the largest single weight read in the model at 1.25 GB, roughly 45× the biggest per-layer projection — was running **on rank 0 alone** while the other seven H200s held the identical hidden state and nothing to do with it. Banding it was a pointer offset and a smaller `N`, because every rank already *held* the weight. And the all-reduce was paying **two cross-GPU rendezvous where one suffices**: at 14–42 KB payloads the reduce moves ~3% of NVLink in the time the kernel takes, so *"what the kernel costs is almost entirely the two rendezvous, and there are 185 of them per token."* The second barrier existed only to stop a fast rank overwriting the shared input while a slow peer still read it — a hazard that disappears if you rotate input buffers instead of synchronizing ([Lecture 07 §6](Lecture-07.md)).

---

## 8. Proving a quantization change is bit-identical

A change to a quantized GEMV is exactly where you would expect bits to move, so the verification bar in #25 is high. Note especially that most of it happened **before** the node run.

**On the host, before spending node hours:**

```text
   14.8M checks on the packed-load and __dp4a rewrites
   160k rows at EVERY blocks_per_row K3 produces,
        including the 32 / 64 / 65 dispatch boundaries
   6 real projection shapes for multi-row, with a ragged tail
   the fused path vs four separate single-row projections
   — with the warp-shuffle reduction tree MODELLED EXACTLY,
     "since a reduction reorder is where a change like this
      would actually move bits"
```

Three things to copy. **Test at the dispatch boundaries** — 32/64/65 is where a `blocks_per_row` branch changes behaviour, and off-by-one bugs live there. **Test the ragged tail**, because a multi-row kernel processing 4 rows at a time meets an N that is not a multiple of 4. **Model the reduction order exactly**, because in a bit-identity claim the reduction tree is the only place the bits can move.

**On the node, as bytes:**

```text
   cmp main.spkl proj.spkl        →  identical
   md5  d4f468195f15e941600690838cf808c7   (both)
   argmax id 1379, logit 13.732111         (both)
   top-1 1.0   mean KLD 0.004045515251179501   (identical, all digits)
   ctest 21/21 on 8x H200
```

And the *reason* it is bit-identical, stated as an invariant rather than an observation:

> *Every output row keeps the same thread-to-block striding, the same `i` order, the same four adds per `i`, and the same `block_sum`. **Only load width, launch shape, and which dot products share a CUDA block change.**"

That is the correct way to justify a bit-identity claim: enumerate what is preserved (accumulation order, reduction structure, operand order) and what is allowed to change (how bytes are fetched, how work is grouped). If your change touches anything in the first list, it is not bit-identical, and you should say what reassociates ([Lecture 04 §8](Lecture-04.md)).

Note also the sharpness of the KLD check: *identical to main, all digits*, `0.004045515251179501`. A metric that agrees to 18 significant figures is a much stronger statement than one that agrees to 3 — and it is free.

---

## 9. The contributor who downgraded their own tier

One more thing from [#67](https://github.com/gittensor-ai-lab/sparkinfer-k3/pull/67), under a heading that is hard to improve on: **"The harness reports this as `L`. It is `S`, and the difference is not mine to keep."**

The harness returned `pct_over_frontier 13.1, label "L"`, because `reference.lock` held `frontier_tps 16.06` measured before [#59](https://github.com/gittensor-ai-lab/sparkinfer-k3/pull/59) merged. #59 was already in `main` and worth ~8.3% on its own. So the reported 13.1% was #59's gain plus this PR's.

The contributor measured `main` themselves (62.27 → 57.33 ms/token), found the discrepancy, and argued their own tier *down* from `L` to `S`.

This is [Lecture 02](Lecture-02.md)'s stale-frontier problem viewed from the contributor's side, and it makes a point about incentive design worth stating plainly: **the reason this behaviour is possible is that the harness publishes both the frontier and the raw measurements, so a contributor can check the arithmetic.** A scoring system that emits only a label cannot be corrected by the person best placed to notice it is wrong.

The same PR also demonstrates the discipline from [Lecture 04 §6](Lecture-04.md) — single-binary A/B via `SPARKINFER_K3_FUSE_QKVG=0`, both instruments reported with their disagreement explained, and per-rep numbers printed. And it reports register pressure as evidence the fusion did not cost occupancy:

```text
   proj_q8_0_q8_0_fused4_kernel<128,4>   REG:46  STACK:0  LOCAL:0  SHARED:20
   "no spill, and half the f32 sibling's register count"
```

> **When you fuse, report register pressure.** Fusion's classic failure mode is that combining four kernels' live state spills to local memory, and a spilled fused kernel can easily be slower than four unfused ones. `REG` and `STACK:0` are the evidence that it did not happen.

---

## Lab — audit your quantized projection path

1. **Check your block struct.** Size and alignment of your quantization block. Is it a multiple of 4? If not, find or write the aligned-reconstruction helper and confirm from the SASS or PTX that your loads widened (§2.1).
2. **Confirm you reach the integer instruction.** Disassemble your quantized inner loop. Do you see `DP4A` / `IMMA`, or scalar `IMAD`? If the latter, your quantization is currently pure cost (§2.2).
3. **Compute activation vs weight traffic** for your three widest projections, at batch 1, one output row per block. If activation traffic dominates, prototype multi-row — and find the N below which the grid division hurts more than the reuse helps (§2.3).
4. **Count launches of every "cheap" kernel.** For each, compute bytes ÷ time and compare to peak bandwidth. Anything under ~1% of peak on a small payload is a launch-count bug (§3).
5. **Find your repeated encodings.** Which values are quantized, normalized, or transformed more than once per token? For each, decide: fuse the consumers, or hoist the producer and pass the buffer. Justify which (§4).
6. **Predict a counter, then measure it.** Before implementing, predict the change in launch count per token. Measure it. **If the gap is more than ~10%, factor the residual against your layer/head/expert counts before you accept the speedup** (§5).
7. **`grep` for negations of your feature flags.** Every `if (!feature)` in a dispatch path is an optimization your feature disables (§6).
8. **Architect for "slow, never wrong."** For your shared-buffer change, write down what happens if the fast path is skipped. If the answer is anything other than "it is slower," redesign (§4).
9. **Verify on the host first.** Exhaustive checks at dispatch boundaries and ragged tails, with the reduction order modelled, before you book hardware (§8).

Pass criterion: one quantized projection change, measured single-binary A/B, with a launch-count prediction that was checked, a bit-identity claim backed by `cmp`, and a written argument that a missed fast path cannot produce a wrong answer.

---

## Self-check

1. Your quantization block is 34 bytes with alignment 2. Explain precisely why the compiler emits one byte-load per element, and what the fix reconstructs from what.
2. A kernel moves 28 KB in 4.19 µs and runs 61,848 times, totalling 7.9% of GPU time. Compute its achieved bandwidth as a fraction of a 4.8 TB/s part. What is the diagnosis, and what is *not* the fix?
3. You process 4 output rows per block to reuse the activation load. Give the benefit, the cost, and the dispatch condition — with the arithmetic for a projection at N=96 on a 132-SM GPU.
4. A kernel runs 128 threads where only 4 have work. Why does no test fail, and why has this survived a code review?
5. A previous engineer cached a quantized activation keyed on its pointer, guarded by "was the scratch last written from this pointer?" It shipped `top1 0.0` at a faster ms/token. Explain the bug and give two architectures that make it impossible.
6. You predict −462 launches per token per rank and measure −233. The residual factors as `69 × 2 + 24 × 1`. What does that tell you, and how did you know to try that factorization?
7. Enabling a new quantized path silently disabled a fusion on 69 layers. Write the `grep` you run after enabling any feature flag, and the comment you leave when writing such a guard.
8. You fuse four projections into one kernel and it is *slower*. Name the first thing you check and the two numbers you report.
9. Your bit-identity claim covers load width, launch shape, and block grouping. List three things you must *not* have changed, and say which one is most likely to have moved the bits.

---

## References

* **The PRs** — [#25](https://github.com/gittensor-ai-lab/sparkinfer-k3/pull/25) (the five-defect Q8_0 projection path), [#67](https://github.com/gittensor-ai-lab/sparkinfer-k3/pull/67) (fuse the four KDA projections; the failed cache; the self-downgraded tier), [#81](https://github.com/gittensor-ai-lab/sparkinfer-k3/pull/81) (quantize once; the prediction-gap debugging), [#107](https://github.com/gittensor-ai-lab/sparkinfer-k3/pull/107) (the serial prologue). Their bodies are the primary source.
* **`ggml` / llama.cpp CUDA quantized paths** — [github.com/ggml-org/llama.cpp](https://github.com/ggml-org/llama.cpp), `ggml/src/ggml-cuda/` — `get_int_b2`-style aligned reconstruction and the `Q8_0` MMVQ kernels that §2.1 and §2.2 mirror. The reference implementation for this whole family.
* **`__dp4a`** — [CUDA C++ Programming Guide § Integer Intrinsics](https://docs.nvidia.com/cuda/cuda-c-programming-guide/); 4-way int8 dot-product-accumulate, available since sm_61.
* **GGUF / K-quant and I-quant formats** — [llama.cpp GGUF spec](https://github.com/ggml-org/llama.cpp/blob/master/docs/gguf.md) — the block layouts whose sizes create §2.1.
* **W8A8 quantization** — SmoothQuant [arXiv:2211.10438](https://arxiv.org/abs/2211.10438) — why activations get quantized at all, and the outlier problem that decides where.
* **Register pressure and occupancy** — [CUDA C++ Best Practices Guide § Register Pressure](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/); `-Xptxas -v` for the `REG`/`STACK` numbers §9 reports.

Cross-references:

* [Part 1 Lecture 04 — The precision stack](../Part%201%20-%20Fundamentals/Lecture-04.md) — where INT8/W8A8 sits among FP16/FP8/FP4.
* [Part 2 Lecture 03 — Quantizing 70B-class models](../Part%202%20-%20Dense%20at%20Hopper/Lecture-03.md) — the weight-quantization side; this lecture is the *activation* side.
* [Lecture 04 — Launch geometry](Lecture-04.md) — §2.3's reuse/occupancy tradeoff and §6's flag-negation pattern in their other form.
* [Lecture 10 — Silently wrong](Lecture-10.md) — the failed cache in §4 is a canonical entry in its taxonomy.

---

## Current as of 2026-08

8× H200 SXM, `sm_90`, CUDA 12.8+, UD-IQ1_S, tp=8, scored context 131,072. Numbers from PRs #25 / #67 / #81 / #107; all single-binary env-gated A/B where stated. `BlockQ8_0` = 34 bytes / align 2 per the GGUF Q8_0 layout. The five-defect audit, the launch-bill diagnostic, the pass-don't-cache rule, and the prediction-gap technique are the durable content.

---

## Next

* Next: [Lecture 06 — Attention at 128k: split over context, split over heads](Lecture-06.md)
* Previous: [Lecture 04 — Launch geometry: grids, occupancy, and 327 norms per token](Lecture-04.md)
* Up: [Part 4 — Optimizing a Real Engine](README.md)
