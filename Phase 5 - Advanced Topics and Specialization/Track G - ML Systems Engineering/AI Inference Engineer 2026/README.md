# AI Inference Engineer 2026 — Special Course

<div class="course-identity mlsys" markdown="1">
<div class="course-identity__icon">INF</div>
<div markdown="1">
<p class="course-identity__eyebrow">Phase 5 · ML Systems Engineering · Special Course</p>
<p class="course-identity__title">Transformer execution, dense and MoE serving, and DeepSeek V4.1 Flash NVFP4 checkpoint engineering across the inference stack.</p>
<p class="course-identity__meta">Artifact: reproducible inference benchmark · Measure: TTFT, TPOT, throughput, $/MTok, parity vs reference</p>
</div>
</div>

> *Up-to-date is not a side requirement. It is the discipline.*

Inference engineering connects model execution, numerical quality, GPU behavior, and service latency. This course uses dated case studies to teach methods that transfer across model and hardware generations. Recheck runtime support and compare actual workloads before adopting a configuration.

The course has **five parts and 37 lectures**. Parts 1–3 develop execution, precision, dense serving, and MoE fundamentals. Part 4 studies an optimization history and its correctness gates. Part 5 applies the same discipline to **DeepSeek V4.1 Flash NVFP4**, starting with checkpoint identity and loader compatibility, then memory, kernels, parallelism, scheduling, and quality. Historical case-study numbers remain specific to their original setups.

**Layer mapping:** L3–L8. Runtime / scheduler / kernels / collectives / fabric / observability — the full ML Systems stack as it applies to inference.

**Role targets:** AI Inference Engineer · GPU Runtime Engineer · LLM Runtime Optimization Engineer · Production Inference Engineer · MLSys Engineer.

**Prerequisites:**

* Phase 5 — ML Systems Engineering — [Stage 0–3](../Guide.md#stage-0-measurement-discipline) — measurement discipline, runtime foundations, transformer execution internals, GPU kernels.
* Phase 5 — Edge AI — [Edge LLM Inference Internals](../../Track%20C%20-%20Edge%20AI/Edge%20LLM%20Inference%20Internals/Lecture-01.md) — GEMV vs GEMM, roofline basics, the decode bottleneck.
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

## Course Map (5 parts, 37 lectures)

### 🧭 Part 1 — Fundamentals of AI Inference / MLSys (5 lectures)

The mental model, the metrics, the math, and the runtime landscape. Anyone who finishes Part 1 can read any model card in 2026 and predict its inference cost shape.

<div class="lecture-map" markdown>

| # | Title |
|---|-------|
| 01 | [The 2026 inference engineer's mental model](Part%201%20-%20Fundamentals/Lecture-01.md) |
| 02 | [Transformer execution — from tokens to bits](Part%201%20-%20Fundamentals/Lecture-02.md) |
| 03 | [Roofline, bandwidth, and the memory hierarchy](Part%201%20-%20Fundamentals/Lecture-03.md) |
| 04 | [The precision stack — FP16 → FP8 → FP4 → INT4](Part%201%20-%20Fundamentals/Lecture-04.md) |
| 05 | [The runtime landscape — vLLM, SGLang, TensorRT-LLM, llama.cpp, MLX](Part%201%20-%20Fundamentals/Lecture-05.md) |

</div>

### ⚙️ Part 2 — Dense Decoder-Only Inference at Hopper (7 lectures)

The end-to-end Hopper stack for 70B-class dense models. Anchored on a **Llama 3.3 70B ↔ Qwen 2.5 72B** comparison so every concept lands on two concrete deployable systems.

<div class="lecture-map" markdown>

| # | Title |
|---|-------|
| 01 | [Anatomy of a 70B-class dense model — Llama 3.3 70B vs Qwen 2.5 72B](Part%202%20-%20Dense%20at%20Hopper/Lecture-01.md) |
| 02 | [Hopper hardware story — H100, H200, Transformer Engine, FP8](Part%202%20-%20Dense%20at%20Hopper/Lecture-02.md) |
| 03 | [Quantizing Llama 3.3 70B and Qwen 2.5 72B — AWQ, GPTQ, QuaRot, SpinQuant, FP8](Part%202%20-%20Dense%20at%20Hopper/Lecture-03.md) |
| 04 | [Single-node multi-GPU serving — tensor parallelism on 8× H100/H200](Part%202%20-%20Dense%20at%20Hopper/Lecture-04.md) |
| 05 | [Modern serving stack — continuous batching, paged KV, prefix cache, speculation](Part%202%20-%20Dense%20at%20Hopper/Lecture-05.md) |
| 06 | [Long context at 128K on Hopper — KV scaling, YaRN, chunked prefill, prefix sharing](Part%202%20-%20Dense%20at%20Hopper/Lecture-06.md) |
| 07 | [Inside the communication layer — NCCL, custom all-reduce, the vLLM communicator stack](Part%202%20-%20Dense%20at%20Hopper/Lecture-07.md) |

</div>

### 🧬 Part 3 — MoE Inference at Blackwell (5 lectures)

The Blackwell stack for modern MoE — DeepSeek V3.1 (with MLA + MTP) and Qwen3-MoE (235B-A22B) — at FP4 on GB200 NVL72.

<div class="lecture-map" markdown>

| # | Title |
|---|-------|
| 01 | [Anatomy of a modern MoE — DeepSeek V3.1 and Qwen3-MoE 235B-A22B](Part%203%20-%20MoE%20at%20Blackwell/Lecture-01.md) |
| 02 | [Blackwell hardware story — B200, B300, GB200 NVL72, Transformer Engine 2, FP4](Part%203%20-%20MoE%20at%20Blackwell/Lecture-02.md) |
| 03 | [Expert parallelism (EP) and the gating hot path](Part%203%20-%20MoE%20at%20Blackwell/Lecture-03.md) |
| 04 | [Disaggregated prefill / decode — Mooncake, Splitwise, DistServe](Part%203%20-%20MoE%20at%20Blackwell/Lecture-04.md) |
| 05 | [Production MoE serving — MTP speculation, constrained decode, cost model](Part%203%20-%20MoE%20at%20Blackwell/Lecture-05.md) |

</div>

### 🔬 Part 4 — Optimizing a Real Engine (10 lectures)

A measured case study. Parts 1–3 pin *teaching anchors*; this part pins **one engine's history**: [SparkInfer-K3](https://github.com/gittensor-ai-lab/sparkinfer-k3) running **Kimi K3** (2.8T params, 896 experts, hybrid MLA + KDA) on a single **8× H200** node, from **1.01 → 60.17 tok/s** decode at 128k across ~96 pull requests — measured against llama.cpp at 18.44 on the same box and the same weights.

The subject is not the numbers, which belong to one model on one box in 2026-08. It is the **discipline**: build the scoreboard before the optimization, find the binding ceiling before writing kernels, and recognize the failure mode where a bug makes the benchmark *better*.

<div class="lecture-map" markdown>

| # | Title |
|---|-------|
| 01 | [The workload, the baseline, and the ladder](Part%204%20-%20Optimizing%20a%20Real%20Engine/Lecture-01.md) |
| 02 | [The scoreboard — a benchmark that cannot be gamed](Part%204%20-%20Optimizing%20a%20Real%20Engine/Lecture-02.md) |
| 03 | [Diagnosis — launch-bound, bandwidth-bound, or comm-bound?](Part%204%20-%20Optimizing%20a%20Real%20Engine/Lecture-03.md) |
| 04 | [Launch geometry — grids, occupancy, and 327 norms per token](Part%204%20-%20Optimizing%20a%20Real%20Engine/Lecture-04.md) |
| 05 | [Fusion and the activation-quantization discipline](Part%204%20-%20Optimizing%20a%20Real%20Engine/Lecture-05.md) |
| 06 | [Attention at 128k — split over context, split over heads](Part%204%20-%20Optimizing%20a%20Real%20Engine/Lecture-06.md) |
| 07 | [Sharding 896 experts — and the Amdahl trap that followed](Part%204%20-%20Optimizing%20a%20Real%20Engine/Lecture-07.md) |
| 08 | [Graph-resident decode — killing the launch bill for good](Part%204%20-%20Optimizing%20a%20Real%20Engine/Lecture-08.md) |
| 09 | [The phase you forgot — batched prefill](Part%204%20-%20Optimizing%20a%20Real%20Engine/Lecture-09.md) |
| 10 | [Silently wrong — the failure mode unique to inference engines](Part%204%20-%20Optimizing%20a%20Real%20Engine/Lecture-10.md) |

</div>

---

### Part 5 — DeepSeek V4.1 Flash NVFP4 (10 lectures)

[Part overview and lab tools](Part%205%20-%20DeepSeek%20V4.1%20Flash%20NVFP4/README.md) — inspect official and community checkpoint formats, derive capacity and traffic budgets, verify loader/kernel compatibility, profile prefill and decode, and optimize useful throughput only after a quality gate passes. CPU exercises are included; full-model results require the named hardware and runtime.

The default NVFP4 teaching artifact is a community conversion. Its format and evaluation status are stated in the part overview; this course does not claim a validated launch merely from an architecture support listing.

---

## Course Outcomes

By the end of the applicable parts you should be able to:

* Read any 2026-era model card and predict its inference cost shape (KV growth, prefill vs decode dominance, dense vs MoE routing, expected dominant precision).
* Pick a runtime (vLLM / SGLang / TRT-LLM / llama.cpp / MLX) for a workload + hardware + SLO and defend the choice.
* Quantize a 70B-class dense or 200B+ MoE model and validate parity against a reference within a defined budget.
* Stand up 8× Hopper TP serving with continuous batching, paged KV, prefix cache, and speculation — and explain which knobs moved which metric.
* Stand up Blackwell EP serving for MoE with token-level routing, MTP, and (where applicable) disaggregated P/D.
* Ship a reproducible benchmark with TTFT, TPOT, throughput, p99, and a defended $/MTok cost line.
* **Take any optimization claim apart** — name three ways the measurement could be flattering itself before arguing about the technique — and build a gate for your own workload that survives an honest engineer, a hostile contributor, and non-deterministic hardware (Part 4).

---

## Currency / Refresh Discipline

Up-to-date is the differentiator of this course. The discipline is baked in:

* Record a checked date and exact model/runtime/hardware identifiers for version-sensitive work. Part 5 keeps its source status in the part overview and immutable experiment revisions in the learner manifest.
* [`REFRESH-LOG.md`](REFRESH-LOG.md) tracks every dated update to lectures and benchmark data.
* Versioned benchmark data: when a model or runtime ships a new version, prior benchmark numbers are kept in dated subfiles for archaeology; the latest pinned at the top of each lecture.
* **Primary sources only** — model cards, technical reports, GitHub releases, official benchmark pages. Blog posts (which silently rot) are last-resort and dated.
* **Live benchmark reference:** for current cross-stack numbers (tokens/s, perf/$, tokens/MW, interactivity) across hardware (H100 → B200 → GB200/GB300 NVL72, MI355X) and runtimes (vLLM / SGLang / TRT-LLM), use a continuously-updated public benchmark such as **[SemiAnalysis InferenceX](https://github.com/SemiAnalysisAI/InferenceX)** ([live dashboard](https://inferencex.com/), Apache-2.0) rather than any fixed number printed in a lecture — software-stack gains move these weekly. Treat the lecture numbers as *teaching anchors*; check the live dashboard's workload, versions, and methodology before using it as a comparison.
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

Part 4 turns that harness into something auditable: a pinned reference with a drift assertion, a correctness gate ordered *before* the speed gate, a raise-only frontier whose every value traces to a committed measurement, a diagnosis naming the binding ceiling with a number per candidate, an optimization ladder of at least five changes **including the ones that measured nothing**, and a deliberate-corruption suite proving the gate catches a bug that makes the engine *faster*.

---

## Exit Criteria

You are done with this course when you can:

* Explain when decode is bandwidth-bound, and use a trace and workload model to identify exceptions.
* Defend a precision floor (FP16 vs FP8 vs INT4 vs FP4) to a roboticist who does not trust quantization.
* Tell, from a profile trace alone, whether a workload is compute-bound, memory-bound, comm-bound, or scheduler-bound — and what to change to verify.
* Walk through your benchmark repo with another engineer and reproduce results within a tolerance justified by observed run-to-run variation.
* Ask another engineer to challenge your assumptions and document the limits of the evidence.

If an exit criterion is missing, use the corresponding lab to close that gap; mark unavailable hardware experiments as not run.

**Study method:** use the [shared lesson cycle and completion rubric](../../../Learning-Guide.md) with this course's existing labs. Predict first, test one variation, retain a failure case, and revisit the self-check after two days and one week.


## Carry this course into the physical-AI system

**Start from:** A model/task/hardware tuple and the measurement discipline in Track G.

**Build and challenge:** Progress from fundamentals to dense/MoE cases and one real-engine optimization, then audit DeepSeek. Carry the same quality and timing definitions through all five parts.

**Pass forward:** An inference evidence bundle reusable by the physical-AI design studio. Continue with [the connected course](../../Track%20D%20-%20Robotics/VLA%20Optimization%20and%20Action-Parity%20Harness/README.md) and keep the artifact in the [Physical AI Design Studio](../../../Physical-AI-Design-Studio.md).
