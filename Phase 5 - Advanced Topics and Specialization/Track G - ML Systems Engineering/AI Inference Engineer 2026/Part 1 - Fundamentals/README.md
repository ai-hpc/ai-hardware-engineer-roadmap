# Part 1 — Fundamentals of AI Inference / MLSys

The mental model layer of the course. Five lectures that build, in order, the four things every AI inference engineer needs before touching a specific model or hardware target:

1. **What the role is** — the four inference shapes, the metrics that matter, the diagnostic flow.
2. **What a transformer actually executes** — prefill vs decode, the KV cache as the load-bearing structure, the three regimes (compute / memory / scheduler bound).
3. **What the hardware does** — roofline, bandwidth, the memory hierarchy, what the marketing-shaped specs actually buy.
4. **What precision means in 2026** — FP16 → FP8 → FP4 → INT4 and the parity validation discipline that keeps quantization honest.
5. **What runtimes exist and why** — vLLM, SGLang, TensorRT-LLM, llama.cpp, MLX, and the workload matrix for picking between them.

By the end of Part 1, a reader should be able to open a model card for a model they have never heard of, and predict — within a margin defended by the math — what its KV cache will cost, where its decode bandwidth ceiling sits, which precision floor it will tolerate, and which runtime is the right starting point. The model-specific parts (Part 2 dense / Part 3 MoE) then deepen that mental model with two concrete anchor pairs each.

## Lectures

<div class="lecture-map" markdown>

| # | Title | Core question it answers |
|---|-------|--------------------------|
| 01 | [The 2026 inference engineer's mental model](Lecture-01.md) | What does the role *do*, day to day, and what metrics decide whether the work was good? |
| 02 | [Transformer execution — from tokens to bits](Lecture-02.md) | What actually runs on the GPU when a token is generated, and why decode is the bandwidth-bound problem? |
| 03 | [Roofline, bandwidth, and the memory hierarchy](Lecture-03.md) | Which hardware spec lines move which metric, and which are noise? |
| 04 | [The precision stack — FP16 → FP8 → FP4 → INT4](Lecture-04.md) | What does each precision floor cost, what does each one buy, and how do we *know* parity? |
| 05 | [The runtime landscape — vLLM, SGLang, TensorRT-LLM, llama.cpp, MLX](Lecture-05.md) | Given a workload + hardware + SLO, which runtime do we start with — and why? |

</div>

## What you ship from Part 1

* A worked benchmark template repo that can boot one runtime against one model and emit TTFT / TPOT / throughput / peak memory.
* A short written analysis of one model card (your choice) that derives its KV-cache cost, dominant-cost regime, and a recommended runtime + precision starting point.
* A roofline plot for one target GPU + one specific kernel showing arithmetic intensity vs achieved bandwidth.

You will use this template repo throughout Parts 2 and 3.

## Exit criteria

You are ready for Part 2 when you can:

* Sketch the prefill-vs-decode split of a transformer forward pass on a whiteboard and label which stage is compute-bound and which is bandwidth-bound at batch=1.
* Compute the KV cache bytes-per-token formula for any model from its `config.json`.
* State which precision floor a published model can tolerate without re-validating from scratch — and which it cannot.
* Defend, in two sentences, why you would pick vLLM over TensorRT-LLM (or vice versa) for a given workload.

If any of these is shaky, re-read the matching lecture before moving on. Part 2 assumes you have all four.
