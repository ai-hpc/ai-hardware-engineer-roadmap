# Part 1 · Lecture 05 — Select a runtime by a tested contract

A runtime loads a model, prepares weights, manages state, chooses kernels, schedules requests, and exposes an application interface. Choose one that supports your exact model, format, hardware, and operational needs. A popularity ranking is not a compatibility check, and no runtime wins every shape.

## Compare concrete capabilities

| Candidate | What to investigate |
|---|---|
| vLLM | Supported architecture/quantization/backend, batching/cache behavior, distributed topology, structured output, operational metrics |
| SGLang | Supported model/attention/EP paths, prefix reuse, scheduling, distributed and structured-output integration |
| TensorRT-LLM | Current PyTorch backend, supported NVIDIA hardware/model/format paths, deployment packaging and dependencies |
| llama.cpp | GGUF model/quantization support, CPU and optional accelerator backends, memory split/offload and serving interface |
| MLX-LM | Supported model/conversion path on the MLX stack, Apple platform requirements, quantization and serving behavior |

Use the official [vLLM documentation](https://docs.vllm.ai/en/latest/), [SGLang documentation](https://docs.sglang.ai/), [llama.cpp repository](https://github.com/ggml-org/llama.cpp), and [MLX-LM repository](https://github.com/ml-explore/mlx-lm). Record a release or commit after checking the relevant feature. “Supports FP8” without a model, kernel, scale format, and GPU is incomplete. An API-compatible endpoint can still differ in chat templates, sampling, token counting, or supported schema features.

## A current TensorRT-LLM boundary

The reviewed current development documentation removed the legacy TensorRT execution backend and directs users to the PyTorch backend. The former `convert_checkpoint.py` → `trtllm-build` sequence is an archived workflow, not a default current recipe. See the [backend migration guide](https://nvidia.github.io/TensorRT-LLM/latest/legacy/tensorrt-backend-removal.html). Match the installed release/container to its documentation; a moving `latest` page is not an immutable pin.

This also changes the comparison: requiring a prebuilt legacy engine is not a blanket reason to reject the current runtime. Compare the actual supported execution path and startup behavior. The course does not claim a universal throughput advantage over vLLM or SGLang.

## Build a compatibility matrix first

For each candidate record model/tokenizer revision, weights/activation/KV format, attention/cache layout, target hardware and minimum driver/toolchain, parallelism groups, maximum supported shapes, and application requirements. Mark a cell supported, unsupported, or unverified with a source. Do not silently treat unverified as supported.

For a tool-using service, also check schema features, streaming, cancellation, timeouts, concurrent requests, observability, and error behavior. For a robot, a separate control loop must remain safe when an inference request is late or fails; a language-model server is not a hard-real-time controller.

## Start small and keep the baseline fair

Prove one small supported model executes before scheduling a costly large-model run. Keep preprocessing/template, prompt tokens, decoding policy, output length, and quality evaluation aligned. A different tokenizer or template can change both task quality and measured work. Measure cold startup separately from steady state and disclose prefix-cache hits.

Then sweep offered load, prompt/output lengths, and concurrency. Keep latency, successful throughput, memory, failures, and task quality together. A comparison at batch one may reverse at high concurrency; a default configuration can disadvantage a runtime unfairly. Save exact commands/configuration and raw outputs so the comparison can be reviewed.

A CPU-only learner can complete the [reference contract lab](labs/README.md) and a source-backed compatibility table. That establishes understanding and reference behavior; it does not establish that a large model serves on a GPU cluster. Hardware deployment is an additional milestone when suitable resources are available.

## Diagnose a failed run

If loading fails, check architecture and checkpoint-format support before changing performance knobs. If outputs differ, check tokenizer/template, position/cache semantics, numerical mode, and sampling before assuming quantization is responsible. If memory is exhausted, include workspace, temporary loading copies, KV allocation, and captures. If latency is poor, separate queueing, CPU preparation, device execution, communication, and output delivery.

**Exercise.** One runtime emits 500 tokens/s but uses a different template and fails 10% more tasks. Is it faster for the product? **Answer:** the experiment has not held the workload/quality contract constant. Report the mismatch and rerun a comparable supported configuration.

Continue to [Part 2 — Dense at Hopper](../Part%202%20-%20Dense%20at%20Hopper/README.md) or return to the [course overview](../README.md).
