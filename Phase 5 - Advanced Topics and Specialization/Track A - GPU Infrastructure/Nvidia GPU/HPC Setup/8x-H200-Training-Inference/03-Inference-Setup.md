# 03 — Set up H200 inference from a model contract

**Outcome:** configure a feasible model, memory budget and runtime, then measure quality and service behavior. Source review: 2026-10-18. These GPU deployment examples are not executed results.

## Context length is a model and runtime property

Use an official model identifier and immutable revision, including tokenizer and quantization metadata. Original Llama 3 8B/70B checkpoints had different context support from Llama 3.1; changing `max_model_len` does not train a model for a longer context. The official Llama 3.1 family includes 8B, 70B and 405B variants. [Meta model collection](https://huggingface.co/meta-llama).

Nominal 1,128 GB across eight GPUs is not one allocation. Compute the largest-rank weights, cache, workspace and reserve. For a conventional 80-layer, 8-KV-head, 128-head-dimension BF16 cache, 32 sequences of 131,072 tokens require about **1,374.39 GB cache alone** across the model. That workload does not fit merely because 70B weights are 140 GB. Use [the cache derivation](04-Memory-Management.md#kv-cache-use-kv-heads-and-actual-retained-tokens), actual retained lengths and runtime sharding.

## Choose a runtime by supported behavior

vLLM, SGLang and TensorRT-LLM expose different scheduling, cache, quantization and model-support choices. Triton Inference Server can host compatible backends and ensembles; it is not itself a universal LLM kernel implementation. Choose one pinned release that supports the exact architecture, format and GPU, then obtain a small correct baseline. No engine is universally fastest.

Install a supported pinned wheel/container according to that release's compatibility matrix. “CUDA 12.x and PyTorch 2.x” is too broad to establish ABI compatibility. Record container digest or package lock, driver and runtime versions.

## A conservative vLLM smoke configuration

After preparing a local immutable model snapshot and confirming its supported context/format, set `MODEL` to that directory. The following starts with a short context and few sequences; use the smallest TP degree that satisfies capacity and task latency, then compare larger groups.

```bash
vllm --version
vllm serve --help
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 vllm serve "$MODEL" \
    --served-model-name course-model --tensor-parallel-size 8 \
    --dtype bfloat16 --max-model-len 4096 --max-num-seqs 8 \
    --gpu-memory-utilization 0.85 --host 127.0.0.1 --port 8000
```

```bash
curl --fail http://127.0.0.1:8000/v1/chat/completions \
    -H 'Content-Type: application/json' \
    -d '{"model":"course-model","messages":[{"role":"user","content":"Explain NVLink in one sentence."}],"max_tokens":64,"temperature":0}'
```

Use the model's correct chat template and inspect output token IDs/text against a reference before increasing concurrency. An OpenAI-compatible local endpoint is a protocol choice, not a call to a hosted OpenAI service. [vLLM serving](https://docs.vllm.ai/en/latest/serving/openai_compatible_server/).

## TensorRT-LLM has changed its execution path

Current TensorRT-LLM uses its **PyTorch backend**. The old TensorRT backend, per-model checkpoint converters and `trtllm-build` examples were removed from current source. The current migration guide starts from `trtllm-serve <model>` or `LLM(model=...)` without a separate engine build. Use the pinned release's supported model/parallel configuration rather than combining removed converter flags with a new package. [Official migration guide](https://nvidia.github.io/TensorRT-LLM/latest/legacy/tensorrt-backend-removal.html).

Standalone TensorRT is a separate product boundary: current 11.4 server documentation does not establish JetPack 10.x compatibility, and older serialized engines are not universally portable across platforms/releases. [TensorRT Jetson migration boundary](https://docs.nvidia.com/deeplearning/tensorrt/latest/api/migration/tensorrt-10x-to-11x-jetson.html).

## Optimize one bottleneck with a quality check

**FP8 weights/activations:** H200 supports FP8 arithmetic, but scales, clipping, accumulation, eligible operators and runtime kernels determine numerical behavior. Casting every tensor to a float8 dtype is not a calibrated quantization pipeline. Doubling a theoretical arithmetic ceiling does not imply doubled tokens/s.

**KV quantization:** changing BF16 cache elements from 2 bytes to 1 byte approximately halves that cache component, plus scale metadata. It does not halve weights/workspace or guarantee twice the concurrency, and it does not expand trained context. Validate long-context retrieval/task quality and cache-scale handling.

**Continuous batching:** admit/remove requests at iteration boundaries according to scheduler constraints. It can improve utilization, but batching, prefill work, cache pressure and arrival patterns affect TTFT and inter-token tails. The GPU is not guaranteed to remain maximally occupied.

**Speculative decoding:** the draft proposes tokens and the target verifies them. Correct rejection/correction logic preserves the intended target distribution; arbitrary draft acceptance does not. Draft cost, tokenizer compatibility, acceptance rate, verification batching and added memory determine net benefit. Current vLLM uses a `speculative_config` interface; old standalone `speculative_model` keyword examples are version-specific. Follow the pinned release's model-specific recipe and compare equal output semantics. [vLLM speculative decoding](https://docs.vllm.ai/en/latest/features/spec_decode/), [original algorithm](https://arxiv.org/abs/2211.17192).

## Measure the service you need

Use [the validation workflow](06-Benchmarks-and-Validation.md). TTFT includes queuing/processing until the first output token; it is not always just prefill kernel time. Record input/output distributions, arrival rate, successes, timeouts, p50/p99 latency and quality-qualified goodput. There are no universal H200 token/s or TTFT targets in this chapter.

**Exercise:** reduce weights from 140 to 70 GB while a hypothetical cache remains 80 GB and other memory 10 GB. Total falls from 230 to 160 GB (30.4%), not 50%. Then separately halve cache: total becomes 120 GB. Explain why runtime support and quality must be checked for each change rather than combining two headline compression ratios.
