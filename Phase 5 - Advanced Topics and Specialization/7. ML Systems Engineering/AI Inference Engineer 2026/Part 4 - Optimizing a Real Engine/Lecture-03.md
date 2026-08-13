# Part 4 · Lecture 03 — Diagnosis: Launch-Bound, Bandwidth-Bound, or Comm-Bound?

## Overview

You have a scoreboard ([Lecture 02](Lecture-02.md)) and a number that is bad ([Lecture 01](Lecture-01.md)). The next question decides your entire month: **which ceiling is binding?**

There are four candidates in a multi-GPU decode loop, and they demand completely different work:

```text
   BANDWIDTH   you are reading weights as fast as HBM allows.
               → reduce bytes per token. quantize, compress the cache, read less.

   OCCUPANCY   the kernels are running, but on a fraction of the machine.
               → change grid geometry, split work, widen blocks.

   LAUNCH      the GPU is idle between kernels more than it is busy.
               → fuse, capture graphs, make the step device-resident.

   COMM        the collectives dominate the step.
               → change the algorithm, the payload, or where the reduce sits.
```

Guessing costs weeks. The case study's own diagnosis was that the collective — the thing that *looks* most expensive in a 93-layer 8-GPU model — accounted for about **2% of the token**, and the real cost was launch overhead in ~30 unfused kernels per layer. Getting that backwards would have meant optimizing NCCL for a month.

By the end you should be able to take a decode profile and name the binding ceiling with a number attached to each candidate, plus the *one* measurement that would falsify your conclusion.

---

## 1. Start with the ceiling you cannot beat

The bandwidth ceiling is the only one of the four that is a law rather than a symptom, so it goes first. From [Part 1 Lecture 03](../Part%201%20-%20Fundamentals/Lecture-03.md):

```text
   tokens/s  ≤  aggregate HBM bandwidth  ÷  bytes read per token
```

For an MoE at batch 1, "bytes read per token" has three parts, and the mistake is counting only the first:

```text
   bytes/token  =  ACTIVE expert weights          (top-k of N, not all N)
                +  every replicated/dense weight   (attention, norms, LM head,
                                                    shared experts, router)
                +  the attention state you touch   (KV cache at depth,
                                                    or recurrent state)
```

Worked for the case study's shape, to show the method:

```text
   weights          553 GiB for 2.8T params   →  ~0.21 bytes/param  (~1.7 bits)
   active params    ~50B  (top-16 of 896 routed + 2 shared + dense)
   active bytes     50e9 × 0.21             ≈  10.6 GB per token
   8× H200 HBM3e    ~4.8 TB/s per card       →  ~38 TB/s aggregate

   naive ceiling    38e12 / 10.6e9          ≈  3600 tok/s
```

That number is useless as a target and extremely useful as a diagnosis. The engine measured **3.55 tok/s** at the start and **60.17 tok/s** at the end. Three orders of magnitude below a bandwidth ceiling means **bandwidth is not the constraint and no amount of quantization will help.** Something else is throwing the machine away.

> **The most valuable output of a roofline calculation is often "not this one."** A ceiling you are 1000× below is not a ceiling; it is proof you are looking at the wrong resource.

Three reasons that naive ceiling is unreachable at batch 1, worth knowing so you do not chase it:

* **Aggregate bandwidth assumes all 8 cards are reading useful bytes simultaneously.** With expert sharding, the ~2 active experts per rank are a small read; the rank finishes and waits.
* **Batch 1 has no reuse.** Every weight byte is read to serve one token. This is the GEMV regime — arithmetic intensity ≈ 1 — where you cannot amortize a read across a tile of work.
* **A stream of small kernels never reaches peak bandwidth**, because each one spends its opening microseconds ramping and its closing microseconds draining.

The repo's own summary of where it actually sat: **"the profile is launch/occupancy-bound at 13× off roofline."** Two of the four candidates, named together, because they are hard to separate — §4.

---

## 2. Price the collective, because everyone assumes it is the problem

A 93-layer model on 8 GPUs *sounds* comm-bound. Measure it before believing it. The case study's arithmetic, which is a template you can reuse:

```text
   MEASURED, ShardPolicy::ExpertsOnly, K3, tp_size 8:

     1 collective per MoE layer × 92 MoE layers   =  92 all-reduces per token
     expert_latent 3584 × 4 bytes (f32)           =  14 KiB per collective
     58.7 µs per call, 8× H200, NCCL              =  ~5.4 ms/token of collective

     token time at the time of measurement        =  281.6 ms  (3.55 tok/s)
     collective share                             ≈  2%
```

Verdict: **not the bottleneck.** The repo's conclusion — *"Launch overhead in the ~30 unfused kernels per layer is."*

Four things in that calculation are easy to get wrong, and each is a lesson.

**Count the collectives from the code, not the diagram.** The doc originally quoted **186** per token — two per layer across 93 layers, the standard tensor-parallel figure from [Part 2 Lecture 04](../Part%202%20-%20Dense%20at%20Hopper/Lecture-04.md). The real number is **92**, because this shard policy *replicates attention* (so there is no attention reduce) and the leading dense layer has no expert dispatch. A 2× error in your comm estimate, from reading a textbook figure instead of the forward pass.

**Get the payload width right.** The reduce is **3584** wide, not 7168 — the routed experts live in a down-projected latent space. Half the bytes you would predict from `hidden_size`.

**Know your dtype and why.** 14 KiB happens to be the same whether you compute `3584 × 4 B` (f32) or `7168 × 2 B` (bf16) — *the same number for a different reason*, which is exactly the kind of coincidence that hides an error. K3 runs an f32 residual stream deliberately, and routing it through a bf16 all-reduce would truncate to ~8 mantissa bits at every layer boundary, undoing the executor's numerics.

**At these sizes you are latency-bound, so report µs/call, not GB/s.** The measured scaling is the proof: **512× the data costs 1.45× the time** (14 KiB → 7 MiB). A GB/s figure at 14 KiB tells you nothing about the hardware and everything about your fixed overhead. The validation tool reports microseconds per call for this reason.

> **Steal this.** For any collective under ~1 MB, the useful unit is latency per call, and the useful optimization target is *the number of calls and the barrier mechanism* — not bandwidth.

---

## 3. Estimate the launch bill

This is the ceiling most people never quantify, and in this case study it was the answer twice.

The count: **~30 unfused kernel launches per layer × 93 layers ≈ 2790 launches per token**, per rank — and the driver issues all 8 ranks from a single host thread.

Now compare against the token budget at two points in the project's history:

```text
   at 3.55 tok/s   →  281.6 ms/token  ÷  2790  =  ~101 µs per kernel
                      launch overhead (~3–5 µs) is ~4% of that.
                      the KERNELS are slow. fix kernels.

   at 60.17 tok/s  →   16.6 ms/token  ÷  2790  =  ~6 µs per kernel
                      launch overhead is now most of the budget.
                      the LAUNCHES are the work. fuse, or capture a graph.
```

That table is the single most important idea in this lecture:

> **The binding ceiling moves as you optimize.** A diagnosis has a shelf life. The correct answer at 3.55 tok/s ("the kernels are slow") is the wrong answer at 60 tok/s, and a team that diagnoses once and executes for six weeks will spend the back half of those weeks on the wrong ceiling.

This is why [Lecture 08](Lecture-08.md) — CUDA graphs and device-resident decode — is late in this part rather than early. Graph capture buys nothing when each kernel runs for 101 µs. It bought **+22.7%** once each kernel ran for a few microseconds. Same change, same code, different value, entirely because of where the rest of the engine had got to.

### 3.1 How to tell launch-bound from slow-kernel, cheaply

Before reaching for a profiler:

| Signal | Reading |
|---|---|
| Sum of kernel durations ≪ wall-clock step time | **Gaps.** Launch/sync-bound. |
| Sum of kernel durations ≈ wall-clock step time | Kernels are the cost. Look at occupancy and bandwidth. |
| Step time barely changes when you shrink a kernel's work | That kernel is launch- or latency-dominated, not compute-dominated. |
| Step time scales with the *number* of layers, not the work per layer | Per-layer fixed cost — launches, syncs, or collectives. |
| A `cudaDeviceSynchronize()` removal changes the number | You were measuring stalls, not throughput. |

In Nsight Systems terms: **the whitespace between kernels on the timeline is the measurement.** Engineers instinctively look at the widest bar; in a launch-bound decode loop the answer is the gaps between the bars.

---

## 4. Occupancy — the ceiling hidden inside "the kernel is slow"

A kernel can be slow because it is doing a lot, or because it is doing a little on a sliver of the GPU. The second is far more common in decode, and far more embarrassing when you find it.

The arithmetic is elementary and almost never done:

```text
   H200 (sm_90):  132 SMs

   grid of  12 blocks  →   9% of the SMs have work.  91% idle.
   grid of   1 block   →  0.8%.
   grid of 132+ blocks →  every SM has something, and the tail matters
```

Two of the case study's real findings, both worth more than their size suggests:

* A norm kernel with **327 invocations per token running on 128 threads** — a single block. 0.8% of the machine, 327 times a token. ([#115](https://github.com/gittensor-ai-lab/sparkinfer-k3/pull/115), [Lecture 04](Lecture-04.md))
* The KDA decode step running **12 blocks on 132 SMs** — 9% occupancy, fixed by value-tiling to widen the grid. ([#77](https://github.com/gittensor-ai-lab/sparkinfer-k3/pull/77))

Neither is an algorithmic insight. Both are *shape* problems: the work was there, the grid did not expose it. And critically, **both look like bandwidth problems in a naive profile** — low achieved bandwidth, low FLOPs, kernel taking longer than it should. The distinguishing measurement is grid dimensions against SM count, which no throughput metric shows you.

### 4.1 Why decode is structurally prone to this

Batch-1 decode produces tensors with a **sequence dimension of 1**. Every parallelization axis a training kernel relies on — batch, sequence, tile — has collapsed. What is left is heads, hidden dimension, and experts. If a kernel parallelizes over heads and the model has 96, you get 96 blocks and a permanently underfilled GPU on a 132-SM card.

So the recurring fix in [Lectures 04](Lecture-04.md) and [06](Lecture-06.md) is **finding another axis to split**: split over context (Flash-Decoding's idea), split over value tiles, split over expert groups × FFN bands. Every one of those is the same move — *manufacture parallelism where the workload's natural axes ran out.*

---

## 5. The diagnosis changes with context depth

One profile is not enough, because the mix moves with sequence length. The case study's clearest data:

| 8× H200, UD-IQ1_S, same weights | ctx 64 | ctx 131,072 | change |
|---|--:|--:|--:|
| llama.cpp | 18.32 | 18.44 | ~flat |
| SparkInfer-K3, as first measured | 10.34 | **1.00** | **−90%** |

Same engine, same weights, same box. One number is 1.8× behind the reference; the other is 18× behind. **They are diagnoses of different bottlenecks.**

At ctx 64 the KV cache is nearly empty, so per-token weight reads dominate and the engine looks merely unoptimized. At 131,072 the attention reduction over depth dominates — and the reference held flat because it keeps a *compressed* MLA cache (`kv_lora` 512, f16) while the candidate reduced over **576 f32 values per token in a kernel with one block per head.**

Three separate defects in one sentence, and you can only see them at depth: the cache is uncompressed (bandwidth), it is f32 (bandwidth), and the kernel has one block per head (occupancy).

> **Profile at the depth you ship.** A flat curve is a claim, not a default. If your candidate degrades with context and your reference does not, the gap *is* your diagnosis — and it is invisible at short context.

The corollary for your own harness: sweep context and plot both engines. The *shape* of the divergence tells you which resource depth consumes. [Lecture 06](Lecture-06.md) is what fixing this one looked like.

---

## 6. Amdahl's law, with receipts

The last diagnostic trap is the one that catches good engineers *after* a win: **optimizing one phase changes which phase is the bottleneck, and can make a parallelization strategy that used to work stop working.**

The case study's instance is unusually clean. The shard policy `ExpertsOnly` bands the 896 routed experts (531 of 553 GiB) across 8 GPUs and **replicates everything else, including attention.** That was the right first cut — it captures essentially all of the memory win, needs one collective per layer instead of two, and lets the forward run at full dimensions on every rank with no per-rank shape threading.

Then the MoE dispatch got ~3× faster:

| | before the MoE speedup | after |
|---|---:|---:|
| tp=1, 16 layers | 196.65 ms/token | **60.66** |
| tp=8, 16 layers | 80.61 ms/token | **54.33** |
| **tp=8 vs tp=1 speedup** | **2.44×** | **~1.1×** |

The MoE got 3× faster and **tensor-parallel scaling collapsed from 2.44× to 1.1×.** Nothing regressed — `tp=8` went from 80.61 to 54.33 ms/token, a real 1.48× win. But the replicated attention did not get faster and did not get sharded, so it became the entire serial term.

Amdahl's law says exactly this, and the numbers let you read the serial fraction off directly:

```text
   speedup(N) = 1 / ( s + (1-s)/N )      s = serial fraction

   before:  2.44× at N=8   →   s ≈ 0.33   (a third of the token was serial)
   after:   1.10× at N=8   →   s ≈ 0.90   (nine tenths is now serial)
```

The parallel part shrank by 3×; the serial part did not move; so its *share* went from a third to nine tenths. Every subsequent GPU you add buys ~nothing until attention is sharded.

Two transferable conclusions:

1. **After any significant win, re-derive your serial fraction.** A scaling number measured before an optimization is not valid after it. Publishing a stale TP-scaling figure is how a team ends up buying GPUs that do nothing.
2. **The next optimization is chosen by the new serial fraction, not by the old plan.** The repo's own roadmap says so plainly: *"Attention is now the whole serial term — `ShardPolicy::Full` is the main lever."*

And a third, subtler one: `ShardPolicy::Full` is *declared and deliberately not enabled*, because the loader would shard weights the executor still indexes at full width — reading past the end of a slice. Knowing the right next move and knowing it is not yet safe are different states, and conflating them ships silent corruption. [Lecture 07](Lecture-07.md) and [Lecture 10](Lecture-10.md).

---

## 7. The diagnostic sequence

Cheapest first. Stop when you have a number for every candidate and one of them is obviously binding.

```text
   1.  BANDWIDTH CEILING           (arithmetic, no GPU)
       bytes/token → tok/s bound.  How far below are you?
       ≫10× below  → not bandwidth. keep going.

   2.  COLLECTIVE COST             (one microbenchmark)
       calls/token × µs/call.  What % of the step?
       <10%        → not comm. keep going.

   3.  LAUNCH BILL                 (count kernels, divide)
       launches/token vs step time.  µs available per kernel?
       <10 µs      → launch-bound. fuse or capture.

   4.  OCCUPANCY                   (read grid dims)
       blocks per launch vs SM count. Which kernels are <25%?
       any hot kernel with a tiny grid → occupancy. reshape it.

   5.  DEPTH SWEEP                 (repeat 1–4 at your real context)
       does your curve diverge from the reference's with depth?

   6.  SERIAL FRACTION             (after every win, re-derive)
       measure at N=1 and N=max. solve Amdahl for s.
```

Steps 1–4 cost an afternoon and no rented hardware for the first two. In the case study they would have produced, in order: *not bandwidth* (1000× below), *not comm* (~2%), *launch-bound* (~6 µs/kernel at the target rate), *and several hot kernels under 10% occupancy* — which is, in fact, the whole content of Lectures 04 through 08.

### 7.1 The write-it-down rule

Before you change code, commit a prediction: **which ceiling binds, what you will change, and how much you expect.** Then measure.

This is not ceremony. It is the only way to detect that you were wrong about *why* something worked. A change that delivers the predicted 10% for the predicted reason teaches you something about the machine; a change that delivers 10% for a reason you did not anticipate is a coincidence you are about to generalize from. The case study has both kinds, and only the sealed receipts let anyone tell them apart afterwards.

---

## Lab — diagnose before you touch anything

Continues the `SHAPE.md` from [Lecture 01](Lecture-01.md)'s lab, and uses the harness from [Lecture 02](Lecture-02.md).

1. **Grade your prediction.** Compare the bandwidth ceiling you predicted in Lecture 01 to your measured tok/s. State the ratio. If you are within 2×, you are bandwidth-bound and Lectures 05 and 06 are your part; if you are 10× or more below, continue.
2. **Price your collectives.** Count all-reduces per token *from the forward pass*, not from a diagram. Microbenchmark one at your real payload size. Report µs/call and percentage of step time. Also report the latency at 8× and 64× the payload — if time barely moves, you are latency-bound and the target is call count.
3. **Count your launches.** Kernels per layer × layers. Divide your step time by it. Report µs available per kernel.
4. **List every kernel's grid.** Sort by (time × idle SMs). Report the top five with `blocks` vs your SM count.
5. **Sweep context.** At least four depths spanning short to your shipping context, both your engine and your reference. Plot both. Describe the divergence in one sentence.
6. **Derive your serial fraction.** Measure at your lowest and highest parallelism degree; solve Amdahl for `s`. If you cannot run at N=1, say so and bound it.
7. **Commit `DIAGNOSIS.md`** naming the binding ceiling, the number supporting each of the four candidates, your predicted win for the first change, and **the one measurement that would prove you wrong.**

Pass criterion: someone reads `DIAGNOSIS.md` and can restate your binding ceiling and your falsification test without asking a question.

---

## Self-check

1. Your bandwidth ceiling says 3600 tok/s; you measure 3.55. List the four candidate explanations in the order you would spend effort on them, and the cheapest measurement that eliminates each.
2. A collective takes 58.7 µs at 14 KiB and 85 µs at 7 MiB. What regime is it in, what unit should you report, and what is the optimization target?
3. You count 2790 kernel launches per token and a 16.6 ms step. Is this launch-bound? Show the arithmetic and state the assumption you had to make about per-launch overhead.
4. Two kernels each take 400 µs. One runs 132 blocks, the other runs 12. Which do you attack first, why, and what would you measure to confirm before writing code?
5. Your engine is 1.8× behind the reference at context 512 and 18× behind at 128k. What single architectural property of your attention cache most likely explains the divergence?
6. You ship a 3× improvement to the phase that was 70% of your token. Compute your new serial fraction and your new expected speedup at 8-way parallelism. What is now the highest-value next change?
7. Your TP-scaling slide says 2.4× at 8 GPUs; it was measured two months and four merged optimizations ago. Argue that the slide is now a liability rather than merely out of date.

---

## References

* **Roofline model** — Williams, Waterman, Patterson, *"Roofline: An Insightful Visual Performance Model"* ([CACM 2009](https://dl.acm.org/doi/10.1145/1498765.1498785)). The original; §1's method is this applied to bytes-per-token.
* **Amdahl's law** — Amdahl, *"Validity of the single processor approach…"* (AFIPS 1967). §6 is a textbook instance with measured numbers.
* **Flash-Decoding** — [PyTorch blog, Oct 2023](https://pytorch.org/blog/flash-decoding/) — the canonical "manufacture parallelism by splitting over context" move that §4.1 describes and [Lecture 06](Lecture-06.md) applies.
* **NVIDIA Nsight Systems** — [docs.nvidia.com/nsight-systems](https://docs.nvidia.com/nsight-systems/) — the tool for §3.1. The whitespace is the measurement.
* **CUDA occupancy** — [CUDA C++ Best Practices Guide § Occupancy](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/#occupancy) and the Occupancy Calculator API.
* **SparkInfer-K3** — [`docs/tensor-parallel.md`](https://github.com/gittensor-ai-lab/sparkinfer-k3/blob/main/docs/tensor-parallel.md) is the source for §2's collective arithmetic and §6's Amdahl table.

Cross-references:

* [Part 1 Lecture 03 — Roofline, bandwidth, and the memory hierarchy](../Part%201%20-%20Fundamentals/Lecture-03.md) — the ceiling in §1.
* [Part 2 Lecture 04 — Tensor parallelism on 8× Hopper](../Part%202%20-%20Dense%20at%20Hopper/Lecture-04.md) — where the "two all-reduces per layer" figure §2 corrects comes from.
* [Part 2 Lecture 07 — Inside the communication layer](../Part%202%20-%20Dense%20at%20Hopper/Lecture-07.md) — small-message latency, the regime §2 lands in.

---

## Current as of 2026-08

Measurements from SparkInfer-K3 on 8× H200 SXM (`sm_90`, 132 SMs, ~4.8 TB/s/card), UD-IQ1_S, CUDA 12.8+, NCCL. Collective figures from `tp_allreduce_check`; occupancy findings from PRs #77 / #115; Amdahl table from `docs/tensor-parallel.md`. The *sequence* in §7 is the durable content.

---

## Next

* Next: [Lecture 04 — Launch geometry: grids, occupancy, and 327 norms per token](Lecture-04.md)
* Previous: [Lecture 02 — The scoreboard: a benchmark that cannot be gamed](Lecture-02.md)
* Up: [Part 4 — Optimizing a Real Engine](README.md)
