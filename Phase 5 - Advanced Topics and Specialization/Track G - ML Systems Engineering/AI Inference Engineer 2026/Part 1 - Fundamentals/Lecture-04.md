# Part 1 · Lecture 04 — Precision is a storage and arithmetic contract

Quantization represents values with fewer bits, plus enough scaling metadata to interpret them. It can reduce memory and enable faster kernels, but neither speed nor quality follows from the bit count alone. In this lesson, derive storage and a small rounding example before choosing a deployment format.

## Identify three independent choices

Specify weight, activation, and KV-cache precision separately. “INT4 model” often describes packed weights while activation arithmetic remains BF16/FP16. “FP8” may mean tensor-scaled or block-scaled weights/activations, with a higher-precision accumulator. The runtime must support the exact format, scaling granularity, shapes, and hardware architecture.

| Choice | Main concern |
|---|---|
| Weight-only quantization | Payload, group scales/zero points, unpack/dequantization kernels |
| Activation quantization | Runtime ranges, outliers, scale calculation, calibration mismatch |
| KV quantization | Growing context storage, repeated attention reads, context-dependent quality |
| Accumulation precision | Reduction error, overflow, output conversion |

FP16 and BF16 each use 16 storage bits but have different exponent/mantissa tradeoffs. FP8 E4M3 and E5M2 are different numerical formats. FP4 E2M1, MXFP4, NVFP4, and packed INT4 are not interchangeable. NVIDIA's [versioned precision primer](https://docs.nvidia.com/deeplearning/transformer-engine-releases/release-2.19/examples/fp8_primer.html) distinguishes Hopper FP8 from Blackwell MXFP8/NVFP4 support. A Transformer Engine **software 2.x** version is not the hardware generation of the same number.

## A transparent INT4 example

For a symmetric toy quantizer with integer range `[-7,7]`, choose `scale=max(abs(x))/7` for a nonzero group. Store `q=clip(round(x/scale),-7,7)` and reconstruct `x_hat=scale*q`. State the tie-breaking rule for `round`; the actual kernel/format must agree. For an all-zero group, use a defined positive scale and zero codes instead of dividing by zero.

With values `[-1,0,0.3,1]`, scale is `1/7`. The codes are `[-7,0,2,7]`; reconstructed 0.3 becomes approximately 0.285714. The lab's `quantize_int4_group` runs this example, all-zero groups, halfway ties, and held-out clipping. This is one group's reconstruction error. Model quality also depends on how errors interact with activations, layer sensitivity, and decoding. A smaller local mean squared error is not a complete task-quality guarantee.

AWQ uses activation information to protect important weight channels; GPTQ uses approximate second-order information to compensate quantization errors. Read the [AWQ paper](https://arxiv.org/abs/2306.00978) and [GPTQ paper](https://arxiv.org/abs/2210.17323) for their assumptions. A format name does not identify the calibration data, exporter version, or serving kernel. Avoid extrapolating one reported model anomaly to all releases or checkpoints.

## Count the metadata

For one row of 128 INT4 weights plus a two-byte scale, storage is `64+2=66` bytes: 4.125 bits/weight, before alignment or zero points. For a row of 129 weights with groups of 128, the lab's format stores `ceil(129/2)=65` payload bytes plus two scales, or 69 bytes. It does not pad the final group payload. Other formats do; use their actual specification.

The [CPU lab](labs/README.md) tests odd row lengths and partial groups. Its accounting model is not a GGUF/AWQ/NVFP4 encoder. For a real checkpoint, inspect tensor payloads, scale tensors, unquantized layers, and runtime allocations; do not estimate the complete model as parameter count divided by two.

## Evaluate a candidate recipe

Pin a licensed checkpoint/tokenizer, select representative calibration data that is separate from evaluation, export with a documented tool version, and verify that the serving runtime consumes that format. Compare the same prompts, chat template, sampling policy, context lengths, and arrival pattern against the chosen higher-precision reference.

Check numerical outputs for a tiny subgraph, then task metrics for the actual application. Include long-context cases if KV precision changes; include rare tool names and argument values for agents. Report absolute quality changes, uncertainty from sample size, throughput, memory, and latency together. Identical generated strings are often too strict for stochastic sampling and too weak as the only quality test.

**Exercise.** Can a smaller checkpoint be slower? **Answer:** yes—unpacking, scale loads, unsupported shapes, extra conversions, or a poor kernel can outweigh reduced traffic. Can an FP8 weight export make a BF16 KV cache smaller? No; KV precision is a separate choice.

Next: [runtime selection](Lecture-05.md). The practical goal is a defended format/runtime combination, not the lowest bit count.
