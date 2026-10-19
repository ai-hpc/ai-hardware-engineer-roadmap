# Part 2 · Lecture 03 — Build a quantization experiment you can defend

Quantizing a 70B-class model is a constrained experiment: reduce storage or latency while preserving the quality needed by the application. Do the [CPU storage exercise](../Part%201%20-%20Fundamentals/labs/README.md) first. Loading and quantizing the actual model requires sufficient host/device memory and authorized access to its weights; no such full-model run is claimed here.

## Define the baseline and the candidate

A useful experiment table has model/tokenizer revision, weight/activation/KV format, group size and scale dtype, calibration-data identity, exporter version, serving runtime, GPU topology, and evaluation settings. Keep the original reference available. A “4-bit” directory name does not establish which tensors stayed in BF16 or which quantized kernel will run.

Start with one candidate supported by your exact runtime/model combination. Weight-only INT4, FP8 weights/activations, and FP8 KV solve different problems. Do not change all three and then attribute the result to one. A conversion that succeeds can still trigger fallback kernels or fail for a larger context shape.

## Calibration is part of the artifact

For methods that use calibration, collect representative prompts with the actual chat template, languages, tool schemas, and length ranges. Keep calibration separate from evaluation. Record dataset hashes and preprocessing. A fixed count of examples is not a universal sufficient calibration budget: coverage and sensitivity matter.

AWQ and GPTQ are algorithm families with different error objectives. The [AWQ paper](https://arxiv.org/abs/2306.00978) and [GPTQ paper](https://arxiv.org/abs/2210.17323) explain their methods; compare a specific exported format and implementation rather than promising that one always wins. Likewise, a W8A8 failure on one older Llama checkpoint does not prove every Llama 3.3 recipe fails.

## Current TensorRT-LLM workflow boundary

The documented current development line removed the legacy TensorRT execution backend and directs users to the PyTorch backend. The old `convert_checkpoint.py` → `trtllm-build` recipe is therefore an archived workflow, not a current generic command sequence. Follow the [backend migration guide](https://nvidia.github.io/TensorRT-LLM/latest/legacy/tensorrt-backend-removal.html) and the [Model Optimizer deployment guidance](https://nvidia.github.io/Model-Optimizer/deployment/1_tensorrt_llm.html) matched to a pinned release. A moving `latest` page is a source to inspect, not a reproducibility pin.

Before a long export, prove a small supported model loads, runs, and selects the intended precision path in that environment. Save help output and configuration with the artifact. This course does not provide invented current flags for a toolchain it has not run.

## Quality before performance

Use paired evaluation: the same inputs through reference and candidate. For deterministic subgraphs, compare outputs with a justified absolute/relative tolerance and include near-zero values. For an application, score tasks: exact schema validity, tool/argument correctness, retrieval usefulness, factual or coding tests as appropriate. For stochastic generation, use fixed decoding policies and enough repeated samples; identical seeds across different kernels do not guarantee identical sequences.

Inspect failures by prompt length and task type. A mean can hide failures on rare tool names or long contexts. Report sample counts and uncertainty; a tiny score difference on a tiny dataset may be noise. Predefine an acceptable degradation before selecting the fastest result.

## Then measure the serving regime

Sweep concurrency and context with cache policy fixed. Count memory for weights, scales, KV pages, activations, workspace, graph captures, runtime allocations, and reserved headroom. Measure both cold and warm behavior where they matter. Record the actual kernel/backend; smaller storage can lose time to unpacking or conversions.

**Worked comparison.** Candidate A lowers weight storage from 140 GB to a hypothetical 38 GB including metadata, but leaves a 40 GiB/request KV cache unchanged. At long context, cache allocation can still dominate capacity. Candidate B halves KV element size but adds scale metadata and quality risk. These changes cannot be summarized by one “model precision” label.

**Exercise.** A candidate is 25% faster on short prompts but fails 8 of 200 long-context tasks that the baseline passes. What is the result? **Answer:** report the speedup for its measured workload and the 4 percentage-point paired failure increase; it has not passed a quality gate that forbids that regression. Do not average away the failure class.

Deliver a supported, pinned recipe with paired evaluation and raw timing data—even if the baseline wins. Next: [tensor parallelism](Lecture-04.md).
