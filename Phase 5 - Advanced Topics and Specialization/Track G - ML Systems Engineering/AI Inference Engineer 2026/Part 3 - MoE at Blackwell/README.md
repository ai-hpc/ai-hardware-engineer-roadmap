# Part 3 — MoE Inference at Blackwell

The Blackwell-class production inference stack for modern Mixture-of-Experts models. Five lectures, anchored on a side-by-side comparison of the two dominant 2025 open-weights MoE families:

* **DeepSeek V3.1** — 671B total / 37B active params, **256 routed experts + 1 shared** per layer, top-8 routing, **MLA** (Multi-head Latent Attention) with compressed KV, native **multi-token prediction (MTP)** head.
* **Qwen3-MoE 235B-A22B** — 235B total / 22B active params, **128 routed experts + 0 shared** per layer, top-8 routing, standard GQA attention, no native MTP.

The pair is uniquely useful: same era, same year, both well-supported in vLLM/SGLang/TRT-LLM, but architecturally different in three teaching-relevant ways — attention type (MLA vs GQA), shared-expert design, and native speculation. Every concept lands on two concrete deployable systems.

**Hardware target:** B200 (192 GB HBM3e), B300 (288 GB HBM3e), and primarily **GB200 NVL72** — the 72-GPU NVLink domain that is the production target for trillion-parameter MoE serving in 2026.

By the end of Part 3 you should be able to ship either MoE to production on a multi-Blackwell deployment, defend the precision recipe at FP4 + FP8 KV, defend the EP/TP partition, and ship a reproducible benchmark with `$/MTok` numbers measured on real hardware.

## Lectures

<div class="lecture-map" markdown>

| # | Title | Core question |
|---|-------|---------------|
| 01 | [Anatomy of a modern MoE — DeepSeek V3.1 and Qwen3-MoE 235B-A22B](Lecture-01.md) | What's the same, what differs, and how does each difference change inference cost? |
| 02 | [Blackwell hardware story — B200, B300, GB200 NVL72, TE2, FP4](Lecture-02.md) | What does Blackwell silicon provide that Hopper doesn't, and how big is NVL72? |
| 03 | [Expert parallelism (EP) and the gating hot path](Lecture-03.md) | How is an MoE partitioned across many GPUs, and where does the all-to-all cost dominate? |
| 04 | [Disaggregated prefill / decode — Mooncake, Splitwise, DistServe](Lecture-04.md) | When does separating prefill GPUs from decode GPUs pay for itself? |
| 05 | [Production MoE serving — MTP speculation, constrained decode, cost model](Lecture-05.md) | What's the full production recipe, and what's the $/MTok at GB200 NVL72 scale? |

</div>

## What you ship from Part 3

Extending the benchmark repo from Parts 1 and 2:

* A reproducible bench harness parametric over `--model {deepseek-v3.1, qwen3-moe-235b-a22b}` × `--runtime {sglang, vllm, trt-llm}` × `--precision {bf16, fp8, fp4}` × `--ep {2, 4, 8, 16}` × `--p-d-mode {colocated, disaggregated}`.
* Precision-parity reports for each model at FP4 + FP8 KV, validated on MMLU / GSM8K / HumanEval / BFCL / RULER.
* Expert-load-balance measurements showing tokens-per-expert distribution and the gating compute cost.
* All-to-all communication time as a fraction of step time at EP=8 and EP=16.
* For at least one of the models, a measured disaggregated P/D run showing the cost-economics crossover with the colocated baseline.
* Final cost model: `$/MTok` for each (model, runtime, precision, EP, mode) cell on B200 and GB200 NVL72.

## Exit criteria

You can do all of:

* Explain MLA's KV-compression mechanism in three sentences and compute its per-token KV bytes against GQA.
* Sketch the all-to-all communication pattern for MoE EP and explain why it's harder than TP's all-reduce.
* Defend EP=8 vs EP=16 for DeepSeek V3.1 on GB200 NVL72 from a measurement.
* Predict where disaggregated P/D wins for an MoE workload and verify with one measurement.
* State your `$/MTok` for one (model, runtime, precision, EP) cell on GB200 NVL72 and walk the formula.

You have completed the course when these are all defended by numbers in your benchmark repo.
