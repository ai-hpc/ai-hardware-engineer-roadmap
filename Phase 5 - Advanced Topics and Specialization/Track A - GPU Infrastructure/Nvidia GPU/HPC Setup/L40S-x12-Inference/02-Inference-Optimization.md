# 02 — Optimize L40S inference without losing the task

**Outcome:** establish a feasible baseline, identify a measured bottleneck and test one optimization. Source review: 2026-10-18. No L40S inference results are claimed here.

## Begin with the actual constraints

L40S has 48 GB GDDR6, 864 GB/s advertised local-memory bandwidth and PCIe Gen4 x16, with no NVLink. H200's higher memory-bandwidth ceiling does not translate into a fixed multiple of decode speed; batch, arithmetic intensity, kernels and cache use matter. L40S supports FP8 Tensor Core operations; AWQ/GPTQ are weight-quantization methods, not substitutes for a supposed missing “hardware scaling switch.” [L40S specification](https://www.nvidia.com/en-us/data-center/l40s/).

A 70B BF16 model has roughly 140 GB raw weights, so it cannot fit on two 48 GB cards without another memory strategy. Four-bit raw weights are about 35 GB, but scales, metadata, non-quantized weights, cache and workspace determine whether one card is usable. Do not promise a context/concurrency before measuring the largest rank.

## Current quantization workflow

AutoAWQ is archived/deprecated and directs new work to maintained tooling; AutoGPTQ is also an archived project. Historical scripts remain useful only with their exact supported dependency stack. Current quantization choices should be selected from the runtime's format/hardware compatibility matrix and maintained conversion tools, such as supported LLM Compressor or GPTQModel recipes, for the specific architecture. [AutoAWQ status](https://github.com/casper-hansen/AutoAWQ), [AutoGPTQ status](https://github.com/AutoGPTQ/AutoGPTQ), [LLM Compressor](https://github.com/vllm-project/llm-compressor), [GPTQModel](https://github.com/ModelCloud/GPTQModel).

A valid procedure is:

1. Pin the original checkpoint/tokenizer and a held-out task test set.
2. Choose representative **calibration-only** examples from the intended input distribution. Repeating “calibration text” is not representative calibration, and one example is not 128 samples.
3. Run a supported quantizer with recorded bit width, grouping, scale/zero-point policy, excluded modules and package versions. Budget the higher-precision conversion working set separately from the final artifact.
4. Save the quantization manifest and reload through the intended inference runtime.
5. Compare held-out task quality, numerical/logprob errors where useful, memory and latency with the original checkpoint under identical prompts/decoding.

Passing `torch_dtype=torch.float8_e4m3fn` to a generic model loader does not create scales, a supported quantized graph or a quality-validated FP8 model. AWQ is not universally better than GPTQ; method, calibration and kernel implementation interact.

## A small local serving configuration

Prepare a compatible local model snapshot that fits, and set `MODEL` to its path. This hardware recipe is unexecuted here:

```bash
CUDA_VISIBLE_DEVICES=0 vllm serve "$MODEL" \
    --served-model-name course-model --dtype bfloat16 \
    --max-model-len 4096 --max-num-seqs 8 \
    --gpu-memory-utilization 0.85 --host 127.0.0.1 --port 8000
```

For a quantized artifact use that runtime's supported dtype/quantization configuration, often detected from its manifest. Do not blindly add a format flag to an unrelated checkpoint. Start with a low concurrency and verify output/peak memory before sweeping it. [vLLM quantization](https://docs.vllm.ai/en/latest/features/quantization/).

## Derive cache use rather than guessing a “sweet spot”

For 32 layers, 8 KV heads, head dimension 128 and BF16, each retained token costs `2×32×8×128×2 = 131,072 bytes` (128 KiB), before metadata. At 256 sequences with 256 retained tokens each, that is **8 GiB**. These dimensions describe one example architecture; verify the actual checkpoint. Prefill workspace, new output tokens, fragmentation and other memory still matter.

Halving cache element size approximately halves that component, not total memory. Increasing `max_model_len` only changes runtime acceptance/allocation policy; it does not make an 8K-trained model a validated 32K model. Use a checkpoint with supported context and evaluate its long-context behavior.

## Attention and speculative decoding are separate experiments

PyTorch SDPA dispatches among supported implementations according to device, shape, dtype, masks and build. It does not guarantee FlashAttention-2 for every Ada call. Installing a separate `flash-attn` package is not the same as changing PyTorch's built-in dispatcher. Check the selected kernel and compare outputs under an explicit tolerance. Avoid promising a fixed 2–4× speedup. [PyTorch SDPA](https://docs.pytorch.org/docs/2.14/generated/torch.nn.functional.scaled_dot_product_attention.html).

Speculative decoding adds draft computation/cache and target verification. A different tokenizer or unsupported target/draft pairing can make a naive combination invalid; a smaller draft is not automatically better. Use a supported runtime `speculative_config` recipe, record acceptance and target-equivalent sampling, and compare whole-request latency. [vLLM speculative decoding](https://docs.vllm.ai/en/latest/features/spec_decode/).

## Serving backend integration

Triton's vLLM backend has its own model-repository contract, model configuration and decoupled streaming behavior. Follow the exact backend release's example, including its `model.json` engine configuration where required, and compatible Triton/vLLM versions. A generic `model.py` folder plus guessed `max_batch_size=256` and `parameters` does not establish a functioning backend. [Official backend examples](https://github.com/triton-inference-server/vllm_backend).

**Practice:** run the [benchmark method](05-Benchmarks.md), then change one of concurrency, precision, cache dtype or TP degree. Predict whether memory traffic, compute or communication should change. Submit quality, useful latency/throughput and peak rank memory. For a fleet robot request, count a late response as a deadline failure even if it eventually completes.
