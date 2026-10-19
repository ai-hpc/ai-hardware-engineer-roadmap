# Chapter 3: Qwen on a Single B200 — Budget, Run, Measure

**Previous:** [Numerics](02-FP4-Numerics-Transformer-Engine.md) · **Next:** [Multi-GPU placement](04-Multi-B200-NVL72.md)

## Freeze one reproducible deployment

Start with a named Qwen checkpoint and a runtime release that explicitly supports its architecture and quantization format. Use a smaller supported model to establish the process if the desired 72B model or hardware is unavailable. A successful smaller-model run validates the method, not the larger deployment's performance.

This chapter provides a measurement procedure. It does not report repository-run B200 throughput or claim an undocumented `MX-FP4-mixed` engine is available. Commands for quantization and serving must come from the pinned runtime example matching the backend; an engine-build flow and a PyTorch-backend checkpoint flow are not interchangeable. [TensorRT-LLM quantization examples](https://nvidia.github.io/TensorRT-LLM/examples/llm_quantization.html).

## Derive the KV cache before choosing concurrency

For conventional grouped-query attention:

`KV bytes = 2 × layers × KV heads × head dimension × resident tokens × bytes per element`

For Qwen2.5-72B-Instruct, the published configuration has 80 layers, 64 attention heads, 8 KV heads, and hidden size 8,192, giving head dimension 128. At BF16 this is `2 × 80 × 8 × 128 × 2 = 327,680` bytes per resident token. A single 32,768-token sequence therefore needs **10 GiB** of raw KV. FP8 payload would halve this to 5 GiB, before scales, padding, and cache-manager overhead. [Qwen model configuration](https://huggingface.co/Qwen/Qwen2.5-72B-Instruct/blob/495f39366efef23836d0cfae4fbe635880d2be31/config.json).

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

## A pinned checkpoint and a CPU memory worksheet

The official snapshot is `Qwen/Qwen2.5-72B-Instruct` revision
`495f39366efef23836d0cfae4fbe635880d2be31`. Its shard index reports
**145,412,407,296 tensor bytes**, not exactly `72e9×2`. This is serialized tensor
payload, not measured runtime residency. The card declares the **Qwen license**
(`license: other`), so do not assume all Qwen sizes have an Apache-2.0 license.
[Metadata and hashes](evidence/qwen-metadata.json).

The unmodified config has `max_position_embeddings=32768` and
`use_sliding_window=false`. A nonzero `sliding_window` field alone does not enable
windowed attention. The card's 128K discussion requires its context-extension
recipe; memory capacity and a larger CLI limit do not establish positional or
quality support. For the baseline, count prompt plus generated tokens within the
selected validated context limit.

Run this **CPU calculation** from the repository root; it reuses the existing
quantization course's arithmetic, with no model downloads:

```bash
python3 - <<'PYCODE'
import json, sys
from pathlib import Path
course = Path("Phase 5 - Advanced Topics and Specialization/Track A - GPU Infrastructure/Nvidia GPU/HPC Setup/Blackwell-B200-Qwen-Inference")
sys.path.insert(0, "Phase 5 - Advanced Topics and Specialization/Track G - ML Systems Engineering/Hardware-Aware LLM Quantization/labs")
from quant_contracts import kv_payload_bytes
m = json.loads((course / "evidence/qwen-metadata.json").read_text())
c = m["configuration"]
per_token = kv_payload_bytes(1, 1, c["num_hidden_layers"], c["num_key_value_heads"], c["hidden_size"] // c["num_attention_heads"], 2)
pool = 170_000_000_000 - m["index_summary"]["total_tensor_bytes"] - 8_000_000_000
print({"KV_bytes_per_token": per_token, "KV_GiB_at_32768": per_token*32768/2**30, "arithmetic_token_capacity": pool//per_token})
assert per_token*32768 <= pool < per_token*32768*2
PYCODE
```

**Observed arithmetic result:** 327,680 bytes/token, 10 GiB at 32,768 tokens and
50,621 aggregate token slots under the hypothetical 170 GB available/8 GB reserve
assumptions. One full 32K sequence fits this estimate; two do not. The actual
allocation and cache manager may differ. The 50,621-slot pool also does not raise
the per-request model context limit. [Retained calculation](evidence/cpu-budget-result.json).

## An explicit optional serving baseline

The following is a **source-checked vLLM command template, not a tested B200 run**.
The flags were checked in vLLM 0.32.0 source at
`5df63a7203cdb9328d824a27b682e77c7b9928be`. Start with a locally staged, verified
snapshot and an installed runtime/container whose CUDA/driver/device support has
been checked. It is intentionally a BF16 short-context baseline; it does not
pretend to convert the checkpoint to NVFP4 by passing a dtype label.

```bash
MODEL_DIR=/absolute/path/to/the/verified/Qwen2.5-72B-Instruct-snapshot
vllm serve "$MODEL_DIR" \
  --served-model-name qwen72b-bf16 \
  --host 127.0.0.1 --port 8000 \
  --dtype bfloat16 --tensor-parallel-size 1 \
  --max-model-len 4096 --max-num-seqs 1 \
  --gpu-memory-utilization 0.90 --enforce-eager
```

`--enforce-eager` establishes an initial graph-free mode, not the recommended
fastest mode. The 0.90 utilization setting is a runtime budget, not a promise of
90% occupancy or bandwidth. Save `vllm --version`, `vllm serve --help=all`, startup
logs, the model manifest and resolved kernel paths. After numerical checks,
compare the supported graph mode separately. Do not install a newer host CUDA
toolkit and assume an existing prebuilt wheel changed its compiled kernels.
[Current CLI docs](https://docs.vllm.ai/en/stable/cli/serve/) and
[pinned argument source](https://github.com/vllm-project/vllm/blob/5df63a7203cdb9328d824a27b682e77c7b9928be/vllm/engine/arg_utils.py).

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

Two processes on one GPU duplicate some memory and contend for resources. Compare them against one batched server under the same total load. A draft model additionally consumes memory; speculative decoding helps only if accepted tokens amortize draft and verification cost. Record accepted draft fraction, emitted tokens per cycle, drafting/verification cost and latency separately. Exact speculative correction preserves the chosen target distribution; lower drafter acceptance alone is not evidence of worse target quality.

One B200 remains one CUDA device despite its two dies. There is no ordinary TP=2 placement across those dies. Move to Chapter 4 when actual per-device capacity or latency measurements justify multiple GPUs.

**Handoff:** deliver the completed memory worksheet, raw load-test records, one profiler trace, and a reasoned placement decision. For the robot SoC, repeat this accounting with camera tokens and action-head state; a text-token KV formula alone is insufficient for an arbitrary VLA architecture.
