# Gemma 4 on Jetson — Edge Deployment, PDL, and Physical AI

<div class="course-identity edge-ai" markdown="1">
<div class="course-identity__icon">G4E</div>
<div markdown="1">
<p class="course-identity__eyebrow">Phase 5 · Edge AI · Special Course</p>
<p class="course-identity__title">Deploy Google's Gemma 4 family on Jetson and edge hardware — from architecture internals to speculative decoding, PDL runtime stack, and multimodal physical-AI pipelines.</p>
<p class="course-identity__meta">Artifact: end-to-end Gemma 4 deployment with measured tok/s, KV-memory, and TTFT on a Jetson target · Measure: tokens/s, tok/s/W, TTFT, KV-cache GB, accuracy vs BF16 baseline</p>
</div>
</div>

> *The question for edge AI in 2025 is no longer "can a frontier-quality model run on device?" — Gemma 4 answers that. The question is "what does it cost in watts, bytes, and milliseconds, and how do you design the pipeline to meet the budget?"*

Gemma 4 (April 2025, Google DeepMind) is the first open model family that is simultaneously **frontier-competitive** at 27B and **edge-deployable** at 1B–4B — not because the small models are compromised, but because the architecture was explicitly co-designed for constrained compute. The key is **interleaved local/global attention**: 6 of every 7 attention layers are sliding-window (O(1) KV growth with context), making 128 K-token context tractable on a Jetson Orin. At Jetson AGX Thor scale (128 GB, 273 GB/s), the 27B model in INT4 runs with a KV footprint an order of magnitude smaller than a pure-attention equivalent.

This course is the engineer's manual for that deployment. It is paired with three concepts you will hear throughout:

**PDL — Portable Deployment and Loading** is the term this course uses for the end-to-end pipeline from a Gemma checkpoint to a running service on Jetson: quantization calibration → format conversion → runtime compilation → loading optimization → serving configuration. It maps to Google's **AI Edge / LiteRT** ecosystem plus the inference optimization layer (paged KV, speculative decoding, CUDA-graph-backed batching). Every lecture is one stage of this pipeline.

**Layer mapping:** L1–L6 — model architecture, quantization, compilation, runtime, serving, and the co-design loop between them.

**Role targets:** Edge AI Engineer · Embedded AI Engineer · Jetson Inference Engineer · Physical-AI Systems Engineer · On-Device ML Engineer.

**Prerequisites:**

* [Edge LLM Inference Internals](../Edge%20LLM%20Inference%20Internals/Lecture-01.md) — GEMV vs GEMM, roofline, bandwidth ceiling. Every number in this course is grounded in that model.
* [Qwen Inference Optimization](../Qwen%20Inference%20Optimization/README.md) — quantization, KV cache, decode optimization on Jetson. Gemma 4 covers the same stack; that series fills any gaps.
* [MLSys Deep Dives → Lecture 04](../../7.%20ML%20Systems%20Engineering/MLSys%20Deep%20Dives/Lecture-04.md) — SSM/hybrid architectures and KV cache theory. Gemma 4's interleaved attention is the same design principle.
* Comfort with Python, `llama.cpp`/`ollama` CLI, and basic CUDA profiling.

**Pairs with:** [TVM Deep Dives](../../7.%20ML%20Systems%20Engineering/TVM%20Deep%20Dives/README.md) (for MLC-LLM compilation) and [AI Inference Engineer 2026](../../7.%20ML%20Systems%20Engineering/AI%20Inference%20Engineer%202026/README.md) (for the datacenter side of the same models).

---

## Why Gemma 4 Changes the Edge Landscape

Three other open-model families were competing for the edge slot in mid-2025: **Qwen3** (Alibaba, strong at 4B–14B), **Phi-4** (Microsoft, 14B dense), and **Llama 3.2** (Meta, 1B/3B/11B). Gemma 4 differentiates on three dimensions:

| Differentiator | Gemma 4 | Qwen3 | Phi-4 | Llama 3.2 |
|---|---|---|---|---|
| Interleaved local/global attention | **Yes (1:6)** | No (pure global) | No | No |
| 128K context with bounded KV | **Yes** | No (KV grows) | No | 128K but full KV |
| Multimodal in same family | **1B–27B all vision** | 7B+ only | No | 11B vision only |
| Quantization stability (QK-norm) | **Yes** | Yes (Qwen3) | Partial | No |
| Google AI Edge / LiteRT first-class | **Yes** | No | No | No |
| Apache-2.0 / Gemma ToS (production) | Gemma ToS | Apache-2.0 | MIT | Llama ToS |

The **interleaved attention** is the architectural moat at the edge. At 128 K context, a pure-attention 4B model needs ~18 GB of KV cache; Gemma 4 4B needs ~3.3 GB — a 5.5× reduction. That is the difference between a model that fits on a Jetson Orin 64 GB and one that doesn't.

---

## Course Map (5 lectures)

<div class="lecture-map" markdown>

| # | Lecture | The thread |
|---|---------|-----------|
| [01](Lecture-01.md) | **Why Gemma 4 at the Edge** — architecture for edge engineers: interleaved attention, GQA, QK-norm, 128K with linear KV, model lineup and competitive positioning | the case for Gemma 4 |
| [02](Lecture-02.md) | **Quantization and Format Conversion** — GPTQ/AWQ/K-quants for Gemma 4, calibration, GGUF/LiteRT/ExecuTorch/.pt2/TRT formats, accuracy vs speed tradeoffs | weights to bits |
| [03](Lecture-03.md) | **The PDL Runtime Stack** — LiteRT (Google AI Edge), llama.cpp, MLC-LLM (TVM Unity), TensorRT-LLM: selection matrix, latency/throughput profiles on Orin and Thor | runtime choice |
| [04](Lecture-04.md) | **Speculative Decoding with Gemma 4** — 1B draft + 4B/12B target, EAGLE-3-style self-draft heads, lookahead decoding, acceptance-length analysis on edge hardware | the algorithm layer |
| [05](Lecture-05.md) | **Physical AI and Multimodal Gemma 4** — SigLIP-400M vision encoder, VLM pipeline on Jetson, robot perception use cases, latency budget, capstone | closing the loop |

</div>

---

## Course Outcomes

By the end you should be able to:

* Explain why Gemma 4's interleaved attention architecture produces a **bounded KV footprint** at long context, and compute the exact KV memory for any (batch, seq, model) configuration.
* Select the right quantization format and runtime for a given Jetson target, and predict throughput from the bandwidth-ceiling formula.
* Deploy Gemma 4 from checkpoint to serving on a Jetson Orin or Thor using at least two different runtimes, with measured tokens/s and TTFT.
* Configure speculative decoding with a Gemma 4 1B draft and a 4B/12B target, report the acceptance length, and verify output parity.
* Describe the multimodal VLM pipeline (SigLIP encoder + Gemma decoder) and fit it within a robot's latency budget.

---

## Exit Criteria

You are done with this course when you can:

* Run `bandwidth_ceiling(HBM_GBs, model_bytes) → max_tok_per_s` on any Gemma 4 configuration on your target and explain which physical bound it's hitting.
* Produce a deployment table with rung-by-rung measurements: baseline BF16 → INT8 → INT4 → INT4 + speculative decode, with tok/s, KV-GB, and TTFT at each rung.
* Look at a new edge model release (any architecture) and immediately ask "what is the KV cache growth rate, and does it fit my memory budget at my target context length?" — the analytical habit this course exists to build.

---

## Currency / Refresh Discipline

Gemma 4 launched April 2025. The deployment ecosystem is moving fast:

* Every lecture closes with a **`## Current as of`** date and the specific versions / benchmark numbers pinned.
* Vendor-reported numbers (Google's benchmark claims, NVIDIA's Jetson Thor throughput sheets) are **explicitly flagged** and treated as teaching anchors, not ground truth.
* The runtime stack (LiteRT versions, llama.cpp GGUF support, MLC-LLM dlight schedules) changes monthly — always re-verify against the tagged release.

---

*Related: [Qwen Inference Optimization](../Qwen%20Inference%20Optimization/README.md) · [Edge LLM Inference Internals](../Edge%20LLM%20Inference%20Internals/Lecture-01.md) · [MLSys Deep Dives](../../7.%20ML%20Systems%20Engineering/MLSys%20Deep%20Dives/README.md)*
