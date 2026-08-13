# Part 4 · Lecture 09 — The Phase You Forgot: Batched Prefill

## Overview

For roughly six weeks, the case-study engine had **no prefill path at all**. Every token of a prompt went through the single-token decode step. Ingesting 32,768 tokens took **812.2 seconds** of wall clock.

Nobody had missed it in the sense of not knowing. It is written down in the project's own roadmap as item 2 of "next, in order." What happened is more interesting and much more common: **the scoreboard measured decode, so the engineering went to decode**, and the phase that was 3.57× behind the reference sat untouched while the phase that ended up 3.26× ahead got seventeen rounds of attention.

Then the scored metric changed, and in **two days** prefill went from 40.35 to 99.68 tok/s — a 2.47× step, and the largest single change in the project's history.

By the end of this lecture you should be able to (1) explain why prefill and decode are different optimization problems that share kernels, (2) derive why batching the prompt is worth an order of magnitude on a dense model and much less on a sparse one, and (3) recognize the organizational failure mode where your metric quietly decides your roadmap.

---

## 1. Two phases, two regimes

From [Part 1 Lecture 02](../Part%201%20-%20Fundamentals/Lecture-02.md), restated in the terms this lecture needs:

```text
   PREFILL   process N prompt tokens.  All N are known up front.
             → parallel over the sequence.  GEMM.  arithmetic intensity ~ N.
             → COMPUTE-bound.  a weight tile is read once, used N times.

   DECODE    produce token t+1 given t.  Inherently serial.
             → sequence dimension is 1.  GEMV.  arithmetic intensity ~ 1.
             → MEMORY-bound.  a weight tile is read once, used once.
```

The consequence that matters here: **prefill's win comes from batching, and it is free in the sense that no approximation is involved.** You are not trading accuracy or changing the math — you are reading each weight tile once instead of *N* times for a computation whose result is identical either way.

An engine that walks the prompt token by token is doing decode's work *N* times to accomplish prefill's job. Its prefill throughput is, by construction, its decode throughput. That is exactly what the case study measured:

```text
   main @ a169ff4, 32,768 prompt tokens:   812.2 s   =   40.35 tok/s
   decode on the same build:                          ~40   tok/s

   "Ingesting is not faster than generating,
    which is the whole opportunity."
```

When your prefill and decode numbers are the same number, you have not measured two things. You have measured one thing twice, and the missing feature is worth roughly the batch width you are not using.

### 1.1 The reference's shape tells you what is possible

| 8× H200, UD-IQ1_S, same weights | llama.cpp | SparkInfer-K3 (before) | |
|---|--:|--:|---|
| decode @ 128k | 18.44 | 56.8 → 60.17 | **3.08–3.26× ahead** |
| prefill @ 32k | **143.88** ± 0.23 | 40.35 | **3.57× behind** |

Read that table as a diagnosis rather than a scoreboard. The reference is **7.8× faster at prefill than at decode** on the same hardware and weights (143.88 vs 18.44). That ratio is the signature of a working batched prefill. The candidate's ratio was ~1.0. The gap between those two ratios *is* the missing feature, and it was quantified before anyone wrote the code.

Note also the reference's error bar: **±0.23 tok/s, ±0.16%**, over 3 reps. Prefill is a much quieter measurement than decode — it is compute-bound and runs long enough to average out scheduling noise. That has a practical consequence for [Lecture 02](Lecture-02.md)'s significance gate: your noise floor is not one number for the whole engine, and a 2% gate that is generous for decode may be loose for prefill.

---

## 2. The transformation: move the layer loop outside the token loop

The entire structural change, in the repo's own description:

> *"The layer loop now sits outside the token loop, so a chunk of tokens goes through each kernel together and a weight tile is read once for the chunk instead of once per token."*

In pseudocode, the before and after:

```text
   BEFORE — per-token walk                 AFTER — chunked / tiled prefill
   ────────────────────────                ──────────────────────────────
   for tok in prompt:                      for chunk in prompt.chunks(C):
       for layer in 0..93:                     for layer in 0..93:
           attn(tok, layer)                        attn(chunk, layer)     # C rows
           moe(tok, layer)                         moe(chunk, layer)      # C rows
                                                   # one all-reduce for the chunk

   weight reads:  93 × K × N               weight reads:  93 × K × N/C
   launches:      93 × ~30 × N             launches:      93 × ~30 × N/C
   collectives:   92 × N                   collectives:   92 × N/C
```

Three costs drop by the same factor `C`, and it is worth separating them because they respond differently:

* **Weight traffic** falls by `C` *only for weights every token in the chunk touches.* §4 is about why that qualifier is the whole story on an MoE.
* **Launch count** falls by `C` unconditionally. Given [Lecture 03](Lecture-03.md)'s finding that this engine was launch-bound, this alone is significant — a chunk of 32 removes 31/32 of the launch bill for the prompt.
* **Collective count** falls by `C` unconditionally, and each collective gets `C` times wider. Since [Lecture 03 §2](Lecture-03.md) established these are latency-bound (512× the payload for 1.45× the time), trading many narrow reduces for few wide ones is close to pure profit.

### 2.1 The lineage, in four PRs

The change did not land in one commit, and the sequence is instructive because each step attacked a different one of those three costs:

In merge order — which is **#133 → #144 → #136**, confirmed by the frontier commits (`59.59 → 66.62` at #144's merge `23949db6`, then `66.62 → 69.02` at #136's `2a6c66f9`):

| PR | Tier | What it did | Attacks |
|---|---|---|---|
| [#133](https://github.com/gittensor-ai-lab/sparkinfer-k3/pull/133) | `l` | MLA slice fill for prefill (**+11.1% @32k**), IQ1_S on the int8 tensor cores | attention grid + arithmetic |
| [#144](https://github.com/gittensor-ai-lab/sparkinfer-k3/pull/144) | `l` | **phase-major** tile prefill — 59.04 → 63.80 tok/s @32k | graph nodes + collectives |
| [#136](https://github.com/gittensor-ai-lab/sparkinfer-k3/pull/136) | `s` | batch the tile's projections and MoE — 63.28 → 69.19 | the arithmetic *inside* a phase |
| [#148](https://github.com/gittensor-ai-lab/sparkinfer-k3/pull/148) | — | batch prompt ingestion — M=1 GEMV becomes M=B GEMM | weight-read amortization |

That order matters, because the four PRs are **three different axes applied in sequence**, not four attempts at one thing:

* **#144 batches the schedule.** *Phase-major* means every token in the tile completes one phase — all the attention, then one reduce, then all the FFN — instead of each token walking the whole layer. Same kernels, same arithmetic, same M=1 shapes; only the issue order changes. And the reason it pays is not latency: collectives cost ~0.01 ms of an 18.8 ms token, so batching them looks worthless. Under CUDA-graph capture the binding cost is **graph size** — capture is worth 1.93× on the per-token path's 3,308-node graph and only 1.56× on a T=16 tile's 61,258-node one. **Nodes are the currency**, and 185 collectives per token are worth deleting whatever they cost in time.
* **#136 batches the arithmetic inside each phase** — projections, router, expert dispatch — so the M=1 GEMVs finally become M=T. This is what #144 was a prerequisite for.
* **#148 amortizes the weight reads**, which is the different and larger effect §4 is about. Its own commit draws the distinction: the tile driver *"batches the collectives but still streams the weights per token."*

The legality argument for #144's reorder is worth keeping, because it is the one you will need for any schedule change: the only cross-token dependencies inside a layer run along the **recurrent** axis — the KDA convolution state and the MLA KV rows — and both are consumed by **attention**, which still runs strictly in token order. Nothing in token *t+1*'s attention reads token *t*'s FFN output; that flows to the next *layer*. **A reorder is legal exactly when every cross-token dependency lives on an axis you keep ordered.**

### 2.2 A layer-major driver is a prerequisite, not a win

The clearest evidence for why the sequence matters is the PR that tried to skip it. [#145](https://github.com/gittensor-ai-lab/sparkinfer-k3/pull/145) — the direct ancestor of #148 — built the layer-major chunked driver first, on its own, and measured:

```text
   128 tokens, 8× H200, same binary and session:

   token loop (main)      55.61 tok/s
   chunked, B = 1         56.78          ← slightly faster
   chunked, B = 8         24.78          ← 2.2× SLOWER
   chunked, B = 32        25.69
```

B=1 beats the loop, because the 1.25 GB LM-head projection runs once per *prompt* instead of once per token. And then batching makes it dramatically worse, for two reasons the PR names as measurements rather than mysteries:

* **The chunked path did not capture a CUDA graph**, and capture alone is worth ~1.8× on prefill. A chunk has to earn that back before it earns anything.
* **The collectives were not actually batched yet.** Sizing them B-fold at init made the *weight load* fail at layer 68, because the collective's owned buffers are peer-mapped and replicated per rank per slot — so the reduce fell back to slicing the payload to existing capacity, which works out to **one token per call.** Same collective count as the token loop, plus two extra device-to-device copies.

> **A restructuring that unlocks a class of optimization is not itself an optimization.** Report it as a prerequisite with its own cost, and keep `B=1` bit-identical to the path it replaces — that equivalence is what makes the driver a usable control rather than an unverifiable rewrite. #148 is what the same driver is worth *after* graph capture and genuinely batched collectives: **+43.6%** instead of −55%.

---

## 3. The same change, three honest ratios

Here is a subtlety that will bite you the first time you report a batching win. The same PR was measured three ways, and all three numbers are correct:

| Comparison | Before | After | Ratio |
|---|--:|--:|--:|
| Same binary, against its own per-token walk | 69.41 | 99.68 | **1.44×** |
| The PR's headline, against its **pre-rebase parent** | 56.53 | 98.80 | **1.75×** |
| Against the last published pre-batching frontier | 40.35 | 99.68 | **2.47×** |

Nothing is being spun. They answer three different questions:

* **1.44×** — *what does chunking buy, holding everything else fixed?* The cleanest attribution of the change itself, because the only variable is the code path. This is the number a kernel engineer should care about.
* **1.75×** — *what did the branch measure when it was written?* Its parent scored 56.53; #136 merged afterwards and moved the pin to 69.02, so this delta is against **a `main` that no longer exists.** The author says exactly that rather than restating it as current: *"I would rather say so than restate a delta I did not measure."*
* **2.47×** — *how much better is prefill than it was before this line of work started?* The right number for a release note, wrong for attributing a single diff.

Worth noting what is *not* in that table: **an eval tier.** #148 merged with no `eval:*` label at all. Its only scored round was a **REJECT** at 169.72 tok/s ([Lecture 10 §3](Lecture-10.md)) — and the honest 99.68 has never been sealed, which is why the pinned frontier deliberately holds at 69.02 ([Lecture 02 §5.2](Lecture-02.md)).

> **State your denominator, and say when it has expired.** "2.47× faster" without "than the 40.35 published frontier of 2026-08-05" is not a claim, it is a mood. And a delta measured against a parent that has since been superseded needs that said out loud — the arithmetic was right when it was taken and is not a statement about today.

---

## 4. Why 2.47× and not 32× — sparsity fights batching

If a chunk of `C` tokens reads each weight tile once instead of `C` times, why is the measured win a small multiple rather than something close to `C`?

For a **dense** model, it very nearly is `C` (until you hit the compute roofline). For a **sparse MoE**, it is not, and the reason is routing divergence:

```text
   dense layer, chunk of C tokens:
       every token needs the SAME weights.
       reads amortize perfectly.  traffic ÷ C.

   MoE layer, top-16 of 896, chunk of C tokens:
       token 1 → experts {a, b, c, ... }      16 of 896
       token 2 → experts {d, e, f, ... }      probably mostly different
       ...
       the chunk touches  UNION of all tokens' top-16
                       ≈ min(16·C, 896)  distinct experts

       at C = 32:  up to 512 distinct experts for 32 tokens.
       each expert's weights read once, used ~1 time.
       expert traffic barely amortizes at all.
```

So on this architecture, chunking amortizes the **dense and replicated** parts — attention projections, norms, router, the 2 shared experts, the LM head — and largely fails to amortize the **896 routed experts**, which are 531 of the 553 GiB. That is precisely why a 2.47× lands where a dense model would show much more.

Two independent confirmations that this is the real constraint rather than a plausible story:

**The project's own next steps target exactly these two places.** The open work at the time of writing is [#154](https://github.com/gittensor-ai-lab/sparkinfer-k3/pull/154) "*chunk-native causal MLA attention for batched prefill*" and [#155](https://github.com/gittensor-ai-lab/sparkinfer-k3/pull/155) "*CSR + grouped IQ1_S MMA on scored chunk MoE*". The second is the fix this analysis predicts: **group the chunk's tokens by expert** so that when you do pay to read an expert's weights, you spend them on every token in the chunk that routed there, and express the routing as a sparse (CSR) structure rather than a per-token loop.

**It is the same problem expert parallelism solves at scale.** [Part 3 Lecture 03](../Part%203%20-%20MoE%20at%20Blackwell/Lecture-03.md)'s all-to-all — gather every token destined for expert *e* onto the rank holding *e*, run one grouped GEMM, scatter back — is grouping-by-expert with a network in the middle. The single-node chunked version has the same shape without the network. **Grouped GEMM is the canonical MoE primitive for exactly this reason**, and it is what turns a chunk of divergently-routed tokens back into dense work.

> **The general lesson.** Batching amortizes a weight read across the tokens that *share* it. Any sparsity that makes tokens need different weights reduces batching's value in direct proportion. On a sparse model, "batch it" is not the end of the design — it is the setup for "now group by what they share."

---

## 5. The organizational failure: your metric is your roadmap

Now the part that generalizes beyond kernels.

Prefill was known-missing, written down, and 3.57× behind the reference. It stayed untouched for six weeks. The mechanism is not negligence — it is that **the scoreboard measured decode at 128k, the reward paid for decode at 128k, and so every contributor's rational move was decode at 128k.** Seventeen frontier advances went into a phase that was already winning.

The fix was to change the metric, and the reasoning recorded in the contribution guide is exactly right:

> Decode was the right thing to score while sparkinfer was 18× behind llama.cpp there. It is now 3.08× ahead, and the untouched gap is ingestion.

[PR #131](https://github.com/gittensor-ai-lab/sparkinfer-k3/pull/131) made prefill @ 32k the tier basis and demoted decode @ 128k to a **regression guard**. Within days, the work moved. Same contributors, same repo, same incentive *structure* — different denominator.

Three transferable points:

**A metric is a resource-allocation policy.** Whatever you score is what you get, at the expense of everything you do not score. If a known gap is not moving, check whether anyone is paid to move it before concluding it is hard.

**Rotate the metric when the marginal return drops.** Not on a schedule, and not because the old metric became wrong — decode at 128k was a perfectly good metric that had simply been mostly harvested. The trigger is *"where is the largest remaining gap between us and the reference,"* which is a question you can answer quarterly with the sweep from [Lecture 03](Lecture-03.md).

**Demote, do not delete.** Decode did not stop being measured; it became a guard with a **1% floor**. This matters because the two phases share kernels:

> *"Decode and prefill share kernels. Batching the prompt will move decode. The 1% guard bounds how far, and it is a refusal rather than a tier — a prefill gain bought by giving decode back has not moved the engine forward, it has moved work around."*

That is the sharpest available statement of why single-metric optimization is dangerous. Without the guard, the first easy prefill win would have been to spend decode, and the ladder would have recorded progress while the engine stood still.

### 5.1 The accident that proves the coupling

The strongest evidence that prefill and decode shared everything came from a mistake, described in [Lecture 02 §5.1](Lecture-02.md) and worth the second visit from this angle.

The prefill frontier was hand-seeded at **40.35** and attributed to the wrong commit. By the time it was written, three decode PRs ([#114](https://github.com/gittensor-ai-lab/sparkinfer-k3/pull/114), [#115](https://github.com/gittensor-ai-lab/sparkinfer-k3/pull/115), [#127](https://github.com/gittensor-ai-lab/sparkinfer-k3/pull/127)) had merged. Because prefill ran the *same single-token path* as decode, those decode wins had lifted prefill too — the first round to actually measure prefill found **53.02**, a **31% understatement**.

So: three decode optimizations delivered a 31% prefill improvement that nobody had asked for or noticed, because nothing was measuring prefill. And then three prefill PRs optimized against the stale 40.35 and reported **+25.4%, +24.5%, +5.3%** — figures that, re-based on the real 53.02, become **−4.6%, −5.3%, −19.8%**. Every one of those claimed wins was, against the actual state of `main`, a regression. The one PR that measured `main` itself landed within **0.23%** of the bot.

Nothing was mis-*scored* — the bot measured the real frontier every round and scored against it, so a PR whose claim was inflated could still be revised and earn a legitimate tier. What the stale pin cost was **three contributors' engineering direction**: they were tuning against a target 31% below reality, and the sign of their own results was hidden from them until a round ran.

Two rules fall out, and they are cheap:

1. **Measure the head you are actually starting from.** Not the published number, not last week's number. The single PR that did this was the only one whose claim survived.
2. **When two phases share a code path, an optimization to either moves both — in whichever direction.** So both need a number every round, even if only one earns a tier.

---

## 6. What batched prefill looks like in production runtimes

The case study built this from scratch for a model nothing else could load. If you are working in a mainstream runtime, the same mechanics appear with different names, and the vocabulary is worth mapping.

| Concept here | Production name | Where |
|---|---|---|
| Chunk of prompt tokens through the kernels together | **batched prefill** | every serving runtime |
| Splitting a long prompt into fixed-size pieces | **chunked prefill** | vLLM `--enable-chunked-prefill`, SGLang default |
| Mixing prompt chunks with decode steps in one batch | **continuous batching / mixed batching** | vLLM, SGLang, TRT-LLM |
| Group tokens by routed expert before the GEMM | **grouped GEMM / expert grouping** | DeepEP, CUTLASS grouped GEMM |
| Run prefill and decode on separate hardware | **P/D disaggregation** | Mooncake, Splitwise, DistServe |

The reason chunked prefill exists in production is not throughput but **latency fairness**: a 32k prompt monopolizing the GPU stalls every concurrent decode, so the prompt is broken into chunks and interleaved with decode steps. That is [Part 2 Lecture 05](../Part%202%20-%20Dense%20at%20Hopper/Lecture-05.md) and [Part 2 Lecture 06](../Part%202%20-%20Dense%20at%20Hopper/Lecture-06.md).

Worth noting what the case study is *not* doing, because it clarifies scope: it is a single-stream engine, so it has no scheduler, no continuous batching, and no request queue. Its "batching" is over the tokens of one prompt, not over concurrent requests. That is the right first target — you cannot schedule a phase you have not implemented — but it means the numbers in this lecture are single-stream numbers and do not speak to serving throughput at all. When [Part 3 Lecture 04](../Part%203%20-%20MoE%20at%20Blackwell/Lecture-04.md) discusses P/D disaggregation, it presumes both phases already work.

---

## 7. Where it landed, and what is left

```text
   prefill @ 32k, 8× H200, UD-IQ1_S

      40.35  ──▶  53.02  ──▶  59.59  ──▶  66.62  ──▶  69.02  ──▶  99.68
      (per-token walk, lifted by decode wins)          │            │
                                                   last sealed   batched,
                                                                 unsealed

      llama.cpp:  143.88                          →  1.44× ahead of us
```

The remaining gap is no longer *"they batch and we do not."* Both engines batch. What is left is **the batching's own efficiency** — chunk-native attention rather than a slice fill, and grouped expert GEMMs rather than per-token dispatch inside the chunk (§4).

That is a much better position to be in, and worth saying explicitly as a matter of engineering judgment: **the difference between "missing a feature" and "our version of the feature is 1.44× off" is the difference between a roadmap item and a tuning problem.** The first is estimated in weeks and has unknown unknowns; the second is estimated in profiles.

---

## Lab — find your own forgotten phase

1. **Measure every phase, not the one you optimize.** For your workload: TTFT, prefill tok/s at a realistic prompt length, decode tok/s at a realistic context, and — if you serve concurrently — throughput at your target batch. Four numbers, same build, same box.
2. **Compute your reference's phase ratio and yours.** `prefill_tok_s / decode_tok_s` for both engines. A reference ratio far above yours means you are missing batching somewhere. Report both ratios.
3. **Find your worst ratio-to-reference.** Which phase are you furthest behind on? That is your candidate, regardless of which one you have been working on.
4. **If you have a per-token path anywhere, estimate the chunked win.** Count weight bytes that *all* tokens in a chunk share versus bytes that only some share. The shared fraction bounds your amortization — this is §4's arithmetic on your model.
5. **Add a guard before you optimize.** Pick the phase you are *not* optimizing and add a regression floor for it (1–2% below current). Then verify the guard fires by deliberately regressing that phase.
6. **Re-derive after the win.** Measure all four numbers again. Did the phase you were not optimizing move? By how much, and in which direction? This is §5.1 on your own code.

Pass criterion: your artifact contains a phase table with both engines' ratios, and a guard that you have *watched fire* — not one you believe would fire.

---

## Self-check

1. Your engine reports prefill 42 tok/s and decode 41 tok/s. What single conclusion follows immediately, and what is the expected size of the fix?
2. A reference engine does 143.88 prefill and 18.44 decode on the same box and weights. What does the 7.8× ratio tell you about its implementation, and what would a ratio near 1.0 tell you?
3. You chunk prefill at C=32 on a dense model and see ~20×; on a top-8-of-256 MoE you see 3×. Explain the difference in one paragraph, then name the optimization that recovers most of the gap.
4. The same PR is honestly described as a 1.44×, a 1.75×, and a 2.47×. Give the question each answers, and say which belongs in a tier and which in a release note.
5. Your team has a known 3.5× gap in a phase nobody is working on. Before concluding it is hard, what do you check?
6. You demote decode from tier basis to a 1% regression guard. A contributor submits a prefill win that costs 0.8% of decode. It passes. Argue both sides, then state what you would actually do.
7. Three decode PRs improved prefill by 31% and nobody noticed for a week. What is the minimum instrumentation that would have caught it, and what would it have cost?

---

## References

* **SparkInfer-K3 prefill work** — [#133](https://github.com/gittensor-ai-lab/sparkinfer-k3/pull/133), [#136](https://github.com/gittensor-ai-lab/sparkinfer-k3/pull/136), [#144](https://github.com/gittensor-ai-lab/sparkinfer-k3/pull/144), [#148](https://github.com/gittensor-ai-lab/sparkinfer-k3/pull/148); the metric change in [#131](https://github.com/gittensor-ai-lab/sparkinfer-k3/pull/131) and [#132](https://github.com/gittensor-ai-lab/sparkinfer-k3/pull/132); the stale-seed post-mortem in [`bench/scripts/reference.lock`](https://github.com/gittensor-ai-lab/sparkinfer-k3/blob/main/bench/scripts/reference.lock) and [`CONTRIBUTING.md`](https://github.com/gittensor-ai-lab/sparkinfer-k3/blob/main/CONTRIBUTING.md).
* **Chunked prefill (Sarathi)** — [arXiv:2308.16369](https://arxiv.org/abs/2308.16369) and **Sarathi-Serve** [arXiv:2403.02310](https://arxiv.org/abs/2403.02310) — the origin of splitting prefill to interleave with decode, and the stall-free-batching argument.
* **Orca (continuous batching)** — [OSDI '22](https://www.usenix.org/conference/osdi22/presentation/yu) — iteration-level scheduling, the mixed-batch idea.
* **vLLM chunked prefill** — [docs.vllm.ai](https://docs.vllm.ai/en/latest/) — the production knob.
* **DistServe** — [arXiv:2401.09670](https://arxiv.org/abs/2401.09670) — the argument that prefill and decode want different resources, taken to its conclusion.
* **CUTLASS grouped GEMM** — [github.com/NVIDIA/cutlass](https://github.com/NVIDIA/cutlass) — the primitive §4 identifies as the fix for divergent routing in a chunk.

Cross-references:

* [Part 1 Lecture 02 — Transformer execution: from tokens to bits](../Part%201%20-%20Fundamentals/Lecture-02.md) — the prefill/decode split.
* [Part 2 Lecture 05 — Modern serving stack](../Part%202%20-%20Dense%20at%20Hopper/Lecture-05.md) and [Lecture 06 — Long context at 128K](../Part%202%20-%20Dense%20at%20Hopper/Lecture-06.md) — chunked prefill as a scheduling technique.
* [Part 3 Lecture 03 — Expert parallelism and the gating hot path](../Part%203%20-%20MoE%20at%20Blackwell/Lecture-03.md) — grouping by expert, at cluster scale.
* [Part 3 Lecture 04 — Disaggregated prefill/decode](../Part%203%20-%20MoE%20at%20Blackwell/Lecture-04.md) — what you do once both phases work.

---

## Current as of 2026-08

SparkInfer-K3 at `7689cc7`. Prefill 99.68 tok/s @ 32k (unsealed; pinned frontier 69.02) against llama.cpp 143.88 ± 0.23. Decode guard floor 56.232. Chunk-native MLA ([#154](https://github.com/gittensor-ai-lab/sparkinfer-k3/pull/154)) and grouped chunk MoE ([#155](https://github.com/gittensor-ai-lab/sparkinfer-k3/pull/155)) open at the time of writing. The phase-ratio diagnostic in §1.1 and the sparsity argument in §4 are the durable content.

---

## Next

* Next: [Lecture 10 — Silently wrong: the failure mode unique to inference engines](Lecture-10.md)
* Previous: [Lecture 08 — Graph-resident decode](Lecture-08.md)
* Up: [Part 4 — Optimizing a Real Engine](README.md)
