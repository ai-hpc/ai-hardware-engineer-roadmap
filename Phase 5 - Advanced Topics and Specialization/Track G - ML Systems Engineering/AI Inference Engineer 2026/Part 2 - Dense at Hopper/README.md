# Part 2 — Dense Decoder-Only Inference at Hopper

The end-to-end production inference stack for 70B-class dense models on Hopper-class hardware (H100 / H200). Seven lectures, anchored on a side-by-side comparison of two released dense-model architectures:

* **Llama 3.3 70B Instruct** — a dense-model case study, 8192 hidden, 28672 FFN, GQA (64 Q / 8 KV), bias-free attention.
* **Qwen 2.5 72B Instruct** — a second dense-model case study, dimensionally near-identical (8192 hidden, *slightly* wider 29568 FFN, same GQA 64 Q / 8 KV), differing in QKV bias (present) and a larger multilingual tokenizer (152K vocab). *(Beware secondary sources that misquote Qwen as 12288 / 49152 — Lecture 01 §2.1 shows why that fails a parameter-count check.)*

Both models share the same family-shape (80 layers, 128K context, GQA, RoPE, RMSNorm, SwiGLU) **and the same core dimensions**. The highlighted differences are tokenizer/vocabulary, FFN width, and QKV bias; this is not an exhaustive comparison. Inspect RoPE configuration, normalization constants, training, and templates at each pinned revision. The pair is the highest-information teaching anchor in the dense space because *every concept lands on two concrete deployable systems* — and the near-identical geometry isolates the differences that actually matter.

By the end of Part 2 you should be able to ship either model to production on 4–8× H200, defend the precision recipe, defend the runtime choice, and produce reproducible benchmarks for TTFT / TPOT / throughput / $/MTok / parity-vs-reference.

## Access and compatibility route

The full deployment experiment requires appropriate hardware. With a CPU or smaller GPU, complete the config audit, memory/traffic derivation, numerical subgraph checks, and benchmark design; label cluster throughput and cost results **not run**. This is an analysis milestone, not completion of the hardware deployment milestone.

Build a model × runtime × format × parallelism compatibility table first. The parameter lists below describe the desired experiment space, not a promise that every combination is supported. Run only supported cells; explain excluded cells and keep them in the report.

## Lectures

<div class="lecture-map" markdown>

| # | Title | Core question |
|---|-------|---------------|
| 01 | [Anatomy of a 70B-class dense model — Llama 3.3 70B vs Qwen 2.5 72B](Lecture-01.md) | What stays the same between these two and what changes? What does each difference cost or buy? |
| 02 | [Hopper hardware story — H100, H200, Transformer Engine, FP8](Lecture-02.md) | What does Hopper actually provide that Ampere doesn't, and what does H200 add over H100? |
| 03 | [Quantizing Llama 3.3 70B and Qwen 2.5 72B — AWQ, GPTQ, QuaRot, SpinQuant, FP8](Lecture-03.md) | What precision recipe ships for each model, defended by parity numbers? |
| 04 | [Single-node multi-GPU serving — tensor parallelism on 8× H100/H200](Lecture-04.md) | How does TP scale, where do the collectives dominate, and what's the runtime-specific config? |
| 05 | [Modern serving stack — continuous batching, paged KV, prefix cache, speculation](Lecture-05.md) | Which knobs move which metric, on this hardware, on these models? |
| 06 | [Long context at 128K on Hopper — KV scaling, YaRN, chunked prefill, prefix sharing](Lecture-06.md) | What breaks at 128K and what is the precision recipe at that context? |
| 07 | [Inside the communication layer — NCCL, custom all-reduce, the vLLM communicator stack](Lecture-07.md) | How does a runtime actually move bytes between GPUs, and which collective path wins at decode? |

</div>

## What you ship from Part 2

A single benchmark repo, extending the harness from Part 1, that contains:

* A reproducible bench harness parametric over `--model {llama-3.3-70b, qwen-2.5-72b}` × `--runtime {vllm, sglang, trt-llm}` × `--precision {fp16, fp8, awq-int4}` × `--tp {2, 4, 8}` × `--context {4k, 32k, 128k}`.
* A precision recipe per model with parity report (MMLU / BFCL / GSM8K / RULER subset).
* A TP-scaling chart with NCCL all-reduce time annotated.
* A long-context bench showing FP8 KV vs FP16 KV at 128K.
* A `$/MTok` cost model for each (model, runtime, hardware, precision) cell.

## Exit criteria

You can do all of:

* Sketch the inference graphs of Llama 3.3 70B and Qwen 2.5 72B, show they are dimensionally near-identical (same 8192 hidden, same KV cost per token), and name where the real differences live — vocab/tokenizer, a ~3% wider FFN, and QKV bias.
* Choose between supported AWQ/GPTQ or other precision candidates using matched quality and performance measurements; a result on another checkpoint is a hypothesis, not the answer.
* Walk the all-reduce step in tensor parallelism on 8× H100 and explain why ring vs tree NCCL matters.
* Predict the TTFT change from enabling chunked prefill at 32K context on H200, then verify.
* State your $/MTok for one (model, precision, TP) cell on H200, with the formula it came from.

If any of these is shaky, re-read the matching lecture before moving to Part 3.


## Carry this course into the physical-AI system

**Start from:** Pinned dense-model configs and the fundamentals benchmark protocol.

**Build and challenge:** Compare supported quantization and parallelism candidates on matched requests. Do not force AWQ to win; use data to choose. Without large GPUs, complete config/math work and mark throughput unmeasured.

**Pass forward:** A dense-model quality/capacity/latency comparison before studying sparse routing. Continue with [the connected course](../Part%203%20-%20MoE%20at%20Blackwell/README.md) and keep the artifact in the [Physical AI Design Studio](../../../../Physical-AI-Design-Studio.md).
