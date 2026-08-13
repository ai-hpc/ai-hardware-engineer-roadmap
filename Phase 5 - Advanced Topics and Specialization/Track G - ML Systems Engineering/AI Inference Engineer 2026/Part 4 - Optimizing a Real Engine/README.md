# Part 4 — Optimizing a Real Engine

Parts 1–3 taught the stack. This part is one engine, one model, one node, and **96 pull requests** — every claim a number somebody measured on rented hardware, every number traceable to the diff that produced it.

The anchor is **[SparkInfer-K3](https://github.com/gittensor-ai-lab/sparkinfer-k3)** (MIT): a from-scratch CUDA inference engine for **Kimi K3** — 2.8T total parameters, 93 layers, **896 routed experts**, a hybrid **KDA + MLA** attention stack, 1M-token context — running on a **single 8× H200 node** at `UD-IQ1_S` (553 GiB of weights).

It is a useful case study for one specific reason: **it started badly, in public, and the record was kept.**

| 8× H200 · UD-IQ1_S · same box, same weights | llama.cpp | SparkInfer-K3 | |
|---|--:|--:|--:|
| **decode @ 128k**, first measurement (2026-08-01) | 18.44 tok/s | **1.01 tok/s** | 18× *behind* |
| **decode @ 128k**, current | 18.44 tok/s | **60.17 tok/s** | **3.26× ahead** |
| **prefill @ 32k**, before batching | 143.88 tok/s | 40.35 tok/s | 3.57× behind |
| **prefill @ 32k**, current | 143.88 tok/s | **99.68 tok/s** | 1.44× behind |

A **60×** decode improvement, in about six weeks, on hardware that never changed. No new silicon, no new model, no lowered precision — the weights and the box are the same at both ends of that table. Everything in between is the subject of this part.

## Why a case study, and why this one

Optimization writing usually arrives pre-laundered. The paper reports the win; the blog post reports the win; the release notes report the win. What gets deleted is the part you actually need: which measurement was wrong, which "speedup" was a bug, which six weeks went into the phase that turned out not to matter.

This repository kept all of it, in the places where engineering history normally survives — `reference.lock` comments, revert commits, `eval:none` labels, and a changelog that records retracted numbers alongside real ones. Three examples of what that buys a reader:

* A PR measured **169.72 tok/s** prefill and was rejected. Two live defects made it read the wrong rows, and reading the wrong rows was *faster*. The honest number for the same change was **98.80**. ([Lecture 10](Lecture-10.md))
* The project scored decode at context **64** for its first weeks while its own docs claimed 128k, because a benchmark harness hardcoded `max_ctx=64`. At 64 it was 1.8× behind llama.cpp; at 128k it was **18×** behind. Weeks of optimization were aimed at a context nobody runs. ([Lecture 02](Lecture-02.md))
* A **3× MoE speedup** made tensor-parallel scaling *worse* — 2.44× → ~1.1× — because the replicated attention it left behind became the entire serial term. Amdahl's law, with receipts. ([Lecture 07](Lecture-07.md))

None of those are in a paper. All three are the kind of thing that decides whether your own optimization month was worth anything.

## What is different about this part

Parts 1–3 pin *teaching anchors* — representative numbers for representative hardware, refreshed on a cadence. Part 4 pins *one measured history*. That changes how you should read it:

| | Parts 1–3 | Part 4 |
|---|---|---|
| Numbers | representative, refreshed | one node, one quant, one engine — frozen with the commits |
| Sourcing | model cards, papers, vendor docs | pull requests, sealed receipts, `reference.lock` |
| Failure modes | described | *diagnosed, with the diff that fixed them* |
| Ask of the reader | understand the mechanism | reproduce the reasoning on your own workload |

The techniques transfer. The numbers do not — they belong to Kimi K3 on `sm_90` at IQ1_S, and quoting them for any other workload is exactly the mistake [Lecture 02](Lecture-02.md) is about.

## Lectures

<div class="lecture-map" markdown>

| # | Title | Core question |
|---|-------|---------------|
| 01 | [The workload, the baseline, and the ladder](Lecture-01.md) | What does 2.8T on one node actually cost, and what did 60× look like rung by rung? |
| 02 | [The scoreboard — a benchmark that cannot be gamed](Lecture-02.md) | How do you build the measurement *before* the optimization, so the numbers survive contact with incentives? |
| 03 | [Diagnosis — launch-bound, bandwidth-bound, or comm-bound?](Lecture-03.md) | The engine is 13× off roofline. Which of the four ceilings is binding, and how do you prove it? |
| 04 | [Launch geometry — grids, occupancy, and 327 norms per token](Lecture-04.md) | Why were the cheapest wins on the board about *grid shape*, not arithmetic? |
| 05 | [Fusion and the activation-quantization discipline](Lecture-05.md) | When does quantizing an activation pay, and how do you fuse without breaking bit-identity? |
| 06 | [Attention at 128k — split over context, split over heads](Lecture-06.md) | Why did decode fall 90% with depth while llama.cpp stayed flat? |
| 07 | [Sharding 896 experts — and the Amdahl trap that followed](Lecture-07.md) | Where does the collective go, and why did a 3× MoE win destroy TP scaling? |
| 08 | [Graph-resident decode — killing the launch bill for good](Lecture-08.md) | What is left when ~30 kernel launches per layer × 93 layers stop being launches? |
| 09 | [The phase you forgot — batched prefill](Lecture-09.md) | How do you spend six weeks on decode while prompt ingestion runs one forward per token? |
| 10 | [Silently wrong — the failure mode unique to inference engines](Lecture-10.md) | What do you do about bugs that produce fluent text, and "speedups" that are corruption? |

</div>

The order is the order you would want to *work* in, not the order the project discovered things in. Lectures 01–03 are setup and diagnosis; 04–08 are the four optimization families, roughly in increasing cost-to-implement; 09–10 are the two disciplines that decide whether any of it counted.

## Prerequisites

* **[Part 1 Lecture 03 — Roofline, bandwidth, and the memory hierarchy](../Part%201%20-%20Fundamentals/Lecture-03.md)** — Lecture 03 here is a roofline argument end to end and assumes you can read one.
* **[Part 2 Lecture 04 — Tensor parallelism on 8× Hopper](../Part%202%20-%20Dense%20at%20Hopper/Lecture-04.md)** and **[Lecture 07 — Inside the communication layer](../Part%202%20-%20Dense%20at%20Hopper/Lecture-07.md)** — Lecture 07 here is the MoE variant of both, with measured collective latencies.
* **[Part 3 Lecture 01 — Anatomy of a modern MoE](../Part%203%20-%20MoE%20at%20Blackwell/Lecture-01.md)** and **[Lecture 03 — Expert parallelism](../Part%203%20-%20MoE%20at%20Blackwell/Lecture-03.md)** — for MLA and expert routing.
* **[Logprobs, Perplexity & KL Divergence](../../Logprobs,%20Perplexity%20and%20KL%20Divergence/README.md)** — the accuracy gate in this part is a mean-KL and top-1-agreement bar. That course derives both.
* Comfort reading CUDA C++. You will not write kernels to follow the argument, but the diffs are CUDA and the interesting ones are quoted.

## What you ship from Part 4

Part 4's artifact is deliberately *not* "reimplement SparkInfer-K3." It is the discipline, applied to a workload you own:

1. **A scoreboard first.** For one model + runtime + hardware target of yours: a pinned reference, a same-box interleaved harness, a correctness gate that runs *before* the speed gate, and a frontier file that is raise-only. No optimization until this exists. ([Lecture 02](Lecture-02.md))
2. **A diagnosis, with a number per ceiling.** Bandwidth ceiling, launch-overhead estimate, collective cost as a percentage of step time, achieved-vs-peak occupancy. State which one binds and what you predict changing it will buy. ([Lecture 03](Lecture-03.md))
3. **An optimization ladder.** At least five changes, each its own commit, each with a before → after measured on the same box, each labeled with its own tier — including the ones that measured `none`. The `none` rows are the point; a ladder with no failures is a ladder that was edited.
4. **A correctness log.** Every change's worst-depth KL and top-1 against your reference. Any change claiming bit-identity, proved bit-identical. ([Lecture 10](Lecture-10.md))
5. **A retrospective.** One page: which ceiling you *thought* was binding, which actually was, what the biggest single win was, and what you would have measured differently at the start. Cite your own `none`s and reverts.

Artifact ladder target: **Level 4–5** — a measurement report with raw data plus a reusable harness another engineer can run.

## Exit criteria

You are done with Part 4 when you can:

* Take any optimization claim — a PR, a paper, a vendor slide — and name **three ways the measurement could be flattering itself** before you argue about the technique.
* Look at a decode profile and say whether the binding ceiling is bandwidth, launch overhead, occupancy, or collectives — and name the *one* measurement that would settle it.
* Explain why a 3× improvement to one phase can make end-to-end scaling worse, and compute the crossover from Amdahl's law.
* Defend a reduce point in a tensor-parallel MoE layer from first principles — including why moving it two ops later multiplies your output by `tp_size` and does not crash.
* Describe three inference-engine bugs that produce **fluent, plausible, wrong** output, and the assertion that catches each one.
* State, for your own artifact, which of your measured wins you would still believe if someone hostile audited it.

If you can do all six, the transferable content of this part has landed. If you can only recite the SparkInfer-K3 numbers, it has not — those numbers are 2026-08 history for one model on one box, and their only job is to make the reasoning concrete.
