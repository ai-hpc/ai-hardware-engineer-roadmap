# AI Inference Engineer 2026 — Special Course

<div class="course-identity mlsys" markdown="1">
<div class="course-identity__icon">INF</div>
<div markdown="1">
<p class="course-identity__eyebrow">Phase 5 · ML Systems Engineering · Special Course</p>
<p class="course-identity__title">From transformer-execution fundamentals to dense-70B on Hopper to MoE-672B on Blackwell — the modern inference stack, end to end.</p>
<p class="course-identity__meta">Artifact: reproducible inference benchmark · Measure: TTFT, TPOT, throughput, $/MTok, parity vs reference</p>
</div>
</div>

> *Up-to-date is not a side requirement. It is the discipline.*

The inference layer has moved more in the last twelve months than the previous three years combined. FP4 went from research to native silicon. Disaggregated prefill/decode went from paper to production. MoE went from "interesting" to "the default architecture above 30B." Any course that does not lead with what shipped in 2025–2026 is already mis-training engineers.

This course is structured as **three parts** that can be read independently or as a sequence. Each part stands on its own; together they walk the precision floor down (FP16 → FP8 → FP4), the architecture from dense to sparse (Llama / Qwen → DeepSeek / Qwen3-MoE), and the hardware up (single-GPU → 8× Hopper → GB200 NVL72 Blackwell).

**Layer mapping:** L3–L8. Runtime / scheduler / kernels / collectives / fabric / observability — the full ML Systems stack as it applies to inference.

**Role targets:** AI Inference Engineer · GPU Runtime Engineer · LLM Runtime Optimization Engineer · Production Inference Engineer · MLSys Engineer.

**Prerequisites:**

* Phase 5 — ML Systems Engineering — [Stage 0–3](../Guide.md#stage-0-measurement-discipline) — measurement discipline, runtime foundations, transformer execution internals, GPU kernels.
* Phase 5 — Edge AI — [Edge LLM Inference Internals](../../3.%20Edge%20AI/Edge%20LLM%20Inference%20Internals/Lecture-01.md) — GEMV vs GEMM, roofline basics, the decode bottleneck.
* Phase 3 — [Neural Networks → Transformer Fundamentals](../../../Phase%203%20-%20Artificial%20Intelligence/1.%20Neural%20Networks/Transformer%20Fundamentals/Lecture-01.md) — Q/K/V, attention, multi-head, the full block.
* Comfort reading Rust or C++ for kernel work, Python for runtime and benchmark glue.

**What comes after:** a reproducible inference benchmark repo for a model + runtime + hardware target of your choice, with a parity report against a published reference and a measured $/MTok cost line.

---

## 🧠 Interactive companion: LLM Inference Visualizer

A 3D, hands-on companion to this course: **[LLM Inference Visualizer](https://github.com/ai-hpc/llm-inference-viz)** — walk the forward pass of a dense decoder-only transformer (**Qwen 2.5 7B/72B**, **Llama 3.3 70B**), see each stage land **memory-bound vs compute-bound** on an **NVIDIA H200 roofline**, and slice the model across **TP = 1/2/4/8** GPUs to watch the weights shard and the all-reduce cost grow.

It makes the core lessons of this course tangible — especially:

* **Part 1 · [Lecture 03 — Roofline, bandwidth, and the memory hierarchy](Part%201%20-%20Fundamentals/Lecture-03.md)** — the roofline chart, decode on the memory-bound side.
* **Part 2 · [Lecture 01 — Anatomy of a 70B-class dense model](Part%202%20-%20Dense%20at%20Hopper/Lecture-01.md)** — GQA, RoPE, RMSNorm, SwiGLU rendered to scale on the Llama/Qwen pair.
* **Part 2 · [Lecture 04 — Single-node multi-GPU serving (tensor parallelism)](Part%202%20-%20Dense%20at%20Hopper/Lecture-04.md)** — TP sharding and the all-reduce collectives, visualized.

Run it locally: `git clone https://github.com/ai-hpc/llm-inference-viz && cd llm-inference-viz && npm install && npm run dev` → open `http://localhost:3002/llm`.

---

## Course Map (3 parts, 16 lectures)

### 🧭 Part 1 — Fundamentals of AI Inference / MLSys (5 lectures)

The mental model, the metrics, the math, and the runtime landscape. Anyone who finishes Part 1 can read any model card in 2026 and predict its inference cost shape.

| # | Title |
|---|-------|
| 01 | [The 2026 inference engineer's mental model](Part%201%20-%20Fundamentals/Lecture-01.md) |
| 02 | [Transformer execution — from tokens to bits](Part%201%20-%20Fundamentals/Lecture-02.md) |
| 03 | [Roofline, bandwidth, and the memory hierarchy](Part%201%20-%20Fundamentals/Lecture-03.md) |
| 04 | [The precision stack — FP16 → FP8 → FP4 → INT4](Part%201%20-%20Fundamentals/Lecture-04.md) |
| 05 | [The runtime landscape — vLLM, SGLang, TensorRT-LLM, llama.cpp, MLX](Part%201%20-%20Fundamentals/Lecture-05.md) |

### ⚙️ Part 2 — Dense Decoder-Only Inference at Hopper (6 lectures)

The end-to-end Hopper stack for 70B-class dense models. Anchored on a **Llama 3.3 70B ↔ Qwen 2.5 72B** comparison so every concept lands on two concrete deployable systems.

| # | Title |
|---|-------|
| 01 | [Anatomy of a 70B-class dense model — Llama 3.3 70B vs Qwen 2.5 72B](Part%202%20-%20Dense%20at%20Hopper/Lecture-01.md) |
| 02 | [Hopper hardware story — H100, H200, Transformer Engine, FP8](Part%202%20-%20Dense%20at%20Hopper/Lecture-02.md) |
| 03 | [Quantizing Llama 3.3 70B and Qwen 2.5 72B — AWQ, GPTQ, QuaRot, SpinQuant, FP8](Part%202%20-%20Dense%20at%20Hopper/Lecture-03.md) |
| 04 | [Single-node multi-GPU serving — tensor parallelism on 8× H100/H200](Part%202%20-%20Dense%20at%20Hopper/Lecture-04.md) |
| 05 | [Modern serving stack — continuous batching, paged KV, prefix cache, speculation](Part%202%20-%20Dense%20at%20Hopper/Lecture-05.md) |
| 06 | [Long context at 128K on Hopper — KV scaling, YaRN, chunked prefill, prefix sharing](Part%202%20-%20Dense%20at%20Hopper/Lecture-06.md) |

### 🧬 Part 3 — MoE Inference at Blackwell (5 lectures)

The Blackwell stack for modern MoE — DeepSeek V3.1 (with MLA + MTP) and Qwen3-MoE (235B-A22B) — at FP4 on GB200 NVL72.

| # | Title |
|---|-------|
| 01 | [Anatomy of a modern MoE — DeepSeek V3.1 and Qwen3-MoE 235B-A22B](Part%203%20-%20MoE%20at%20Blackwell/Lecture-01.md) |
| 02 | [Blackwell hardware story — B200, B300, GB200 NVL72, Transformer Engine 2, FP4](Part%203%20-%20MoE%20at%20Blackwell/Lecture-02.md) |
| 03 | [Expert parallelism (EP) and the gating hot path](Part%203%20-%20MoE%20at%20Blackwell/Lecture-03.md) |
| 04 | [Disaggregated prefill / decode — Mooncake, Splitwise, DistServe](Part%203%20-%20MoE%20at%20Blackwell/Lecture-04.md) |
| 05 | [Production MoE serving — MTP speculation, constrained decode, cost model](Part%203%20-%20MoE%20at%20Blackwell/Lecture-05.md) |

---

## Course Outcomes

By the end of all three parts you should be able to:

* Read any 2026-era model card and predict its inference cost shape (KV growth, prefill vs decode dominance, dense vs MoE routing, expected dominant precision).
* Pick a runtime (vLLM / SGLang / TRT-LLM / llama.cpp / MLX) for a workload + hardware + SLO and defend the choice.
* Quantize a 70B-class dense or 200B+ MoE model and validate parity against a reference within a defined budget.
* Stand up 8× Hopper TP serving with continuous batching, paged KV, prefix cache, and speculation — and explain which knobs moved which metric.
* Stand up Blackwell EP serving for MoE with token-level routing, MTP, and (where applicable) disaggregated P/D.
* Ship a reproducible benchmark with TTFT, TPOT, throughput, p99, and a defended $/MTok cost line.

---

## Currency / Refresh Discipline

Up-to-date is the differentiator of this course. The discipline is baked in:

* Every lecture closes with `## Current as of YYYY-MM` stating the date the content was written and the specific model / runtime / hardware versions it pinned.
* [`REFRESH-LOG.md`](REFRESH-LOG.md) tracks every dated update to lectures and benchmark data.
* Versioned benchmark data: when a model or runtime ships a new version, prior benchmark numbers are kept in dated subfiles for archaeology; the latest pinned at the top of each lecture.
* **Primary sources only** — model cards, technical reports, GitHub releases, official benchmark pages. Blog posts (which silently rot) are last-resort and dated.
* **Live benchmark reference:** for current cross-stack numbers (tokens/s, perf/$, tokens/MW, interactivity) across hardware (H100 → B200 → GB200/GB300 NVL72, MI355X) and runtimes (vLLM / SGLang / TRT-LLM), use a continuously-updated public benchmark such as **[SemiAnalysis InferenceX](https://github.com/SemiAnalysisAI/InferenceX)** ([live dashboard](https://inferencex.com/), Apache-2.0) rather than any fixed number printed in a lecture — software-stack gains move these weekly. Treat the lecture numbers as *teaching anchors*; treat the live dashboard as *truth at time of deployment*.
* **Refresh cadence:** six months default; three months if a major model class drops (e.g. DeepSeek V4, Llama 5, Qwen 4) or a hardware generation lands (B300 → Vera Rubin etc.).

---

## What You Should Produce

A single repo that, by the end of Part 2, contains:

* A reproducible benchmark harness (parametric over model / runtime / hardware).
* One full pipeline at three quantization levels (FP16 reference → FP8 → AWQ-INT4) with parity report.
* TP-scaling numbers on 8× H100 or H200 with NCCL timing breakdown.
* Continuous-batching + prefix-cache + speculation numbers with each knob isolated.
* 128K-context bench with FP8 KV vs FP16 KV.
* A cost model: $/MTok across configurations on the chosen hardware.

By the end of Part 3, the same harness extends to MoE on Blackwell with EP, MTP speculation, and (where the cluster allows) disaggregated P/D measurements.

---

## Exit Criteria

You are done with this course when you can:

* Explain, on a whiteboard, why decode is bandwidth-bound and what makes a workload escape that regime.
* Defend a precision floor (FP16 vs FP8 vs INT4 vs FP4) to a roboticist who does not trust quantization.
* Tell, from a profile trace alone, whether a workload is compute-bound, memory-bound, comm-bound, or scheduler-bound — and what to change to verify.
* Walk through your benchmark repo with another engineer and they reproduce your numbers on the same hardware class within ±5%.

If you cannot do all four, you have a notebook of recipes, not a body of inference-engineering work. Re-run the benchmarks.
