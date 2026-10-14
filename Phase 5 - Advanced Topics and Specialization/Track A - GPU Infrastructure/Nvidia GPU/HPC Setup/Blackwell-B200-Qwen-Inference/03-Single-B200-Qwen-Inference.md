# Chapter 3: Qwen on a Single B200 — Budget, Run, Measure

**Previous:** [Numerics](02-FP4-Numerics-Transformer-Engine.md) · **Next:** [Multi-GPU placement](04-Multi-B200-NVL72.md)

## Freeze one reproducible deployment

Start with a named Qwen checkpoint and a runtime release that explicitly supports its architecture and quantization format. Use a smaller supported model to establish the process if the desired 72B model or hardware is unavailable. A successful smaller-model run validates the method, not the larger deployment's performance.

This chapter provides a measurement procedure. It does not report repository-run B200 throughput or claim an undocumented `MX-FP4-mixed` engine is available. Commands for quantization and serving must come from the pinned runtime example matching the backend; an engine-build flow and a PyTorch-backend checkpoint flow are not interchangeable. [TensorRT-LLM quantization examples](https://nvidia.github.io/TensorRT-LLM/examples/llm_quantization.html).

## Derive the KV cache before choosing concurrency

For conventional grouped-query attention:

`KV bytes = 2 × layers × KV heads × head dimension × resident tokens × bytes per element`

For Qwen2.5-72B-Instruct, the published configuration has 80 layers, 64 attention heads, 8 KV heads, and hidden size 8,192, giving head dimension 128. At BF16 this is `2 × 80 × 8 × 128 × 2 = 327,680` bytes per resident token. A single 32,768-token sequence therefore needs **10 GiB** of raw KV. FP8 payload would halve this to 5 GiB, before scales, padding, and cache-manager overhead. [Qwen model configuration](https://huggingface.co/Qwen/Qwen2.5-72B-Instruct/blob/main/config.json).

Sixteen such sequences consume 160 GiB of raw BF16 KV or 80 GiB of raw FP8 KV. Add weights, activation peaks, graph pools, scratch, allocator reserves, and runtime overhead. Query actual free bytes; a GPU advertised in decimal GB has fewer GiB. Long-context support additionally depends on the model's position-encoding configuration and evaluation, not just memory capacity.

A useful budget is:

```text
available device bytes
  - resident packed weights (measured after load)
  - maximum activation/workspace (profiled)
  - graphs/runtime/reserve (measured or stated budget)
  = upper bound for KV pool
```

If the result is negative, change precision, context/concurrency, or placement. Do not fix OOM by silently truncating prompts in the evaluation harness.

## Establish correctness and timing separately

1. Save `platform.json` from Chapter 1 and the checkpoint manifest from Chapter 2.
2. Run a fixed short input and compare against the reference with deterministic settings where supported. Record tolerances; floating-point execution can still differ between kernels.
3. Warm up initialization, compilation, and graph capture. Report cold startup separately.
4. Sweep input length, output length, and concurrency independently. Use explicit stop/EOS behavior so two engines generate comparable work.
5. Record TTFT, inter-token latency, completion latency, tokens per second, peak allocated/reserved memory, error rate, and quality. Separate per-request rate from aggregate throughput.
6. Inspect a trace for at least one low-concurrency and one saturated run. Determine whether the bottleneck is launch work, GEMM, KV traffic, sampling, or queueing.

For asynchronous GPU timing, host timing must include the relevant completion/synchronization. Server request latency intentionally includes queueing and communication; device kernel timing does not.

## A results table that cannot hide the workload

| Run | Checkpoint/revision + precision | Input/output tokens | Concurrency | TTFT p50/p99 | Inter-token p50/p99 | Goodput | Peak memory | Quality |
|---|---|---|---|---|---|---|---|---|
| Reference | Fill from manifest | Fixed protocol | Sweep | Measure | Measure | Define SLO first | Measure | Measure |
| Candidate | Change one factor | Same protocol | Same sweep | Measure | Measure | Same SLO | Measure | Same tasks |

Goodput counts only requests or tokens satisfying the predeclared service objective. Include failed and timed-out requests in the denominator of success rate. Do not infer production quality from a model loading or a single plausible response.

## One instance, replicas, and speculation

Two processes on one GPU duplicate some memory and contend for resources. Compare them against one batched server under the same total load. A draft model additionally consumes memory; speculative decoding helps only if accepted tokens amortize draft and verification cost. Record acceptance rate alongside latency.

One B200 remains one CUDA device despite its two dies. There is no ordinary TP=2 placement across those dies. Move to Chapter 4 when actual per-device capacity or latency measurements justify multiple GPUs.

**Handoff:** deliver the completed memory worksheet, raw load-test records, one profiler trace, and a reasoned placement decision. For the robot SoC, repeat this accounting with camera tokens and action-head state; a text-token KV formula alone is insufficient for an arbitrary VLA architecture.
