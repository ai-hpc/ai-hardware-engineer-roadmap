# Lecture 3: Diagnose and optimize Qwen decode on Jetson

**Reviewed 18 October 2026.** Input: a supported runtime and an evaluated artifact, or the portable numerical model. Output: a controlled before/after experiment with correct arithmetic and separate quality results. This lesson does not report measured Orin inference throughput.

## 1. Establish the baseline before changing clocks

Use the [Jetson compatibility guide](../../../Phase%204%20-%20Track%20B%20-%20Nvidia%20Jetson/Compatibility.md) for the actual module/BSP/runtime combination. Record power mode, clocks, temperature, cooling, memory pressure, backend, and offloaded layers. Query `nvpmodel -q --verbose` on the device. Numeric mode IDs and maximum frequencies differ across modules and software configurations; do not copy a fixed `-m 0` recipe and call it MAXN.

Measure time to first token, time between generated tokens, total generation time, and peak memory. State prompt length, cache depth, output length, batch/concurrency, warmup and repetitions. A periodic `tegrastats` sample showing an idle GPU can miss bursts or waiting periods; backend logs and a CUDA trace establish actual execution placement. No fixed 50× gain follows from changing power mode.

## 2. Correct the byte arithmetic

For one Qwen3-4B layer, exact illustrative payloads are:

| Matrix/format | Calculation | Bytes |
|---|---|---:|
| Q, Q4_K | `4096*2560*144/256` | 5,898,240 |
| K, Q4_K | `1024*2560*144/256` | 1,474,560 |
| V, Q6_K | `1024*2560*210/256` | 2,150,400 |
| O, Q4_K | `2560*4096*144/256` | 5,898,240 |
| Gate or up, Q4_K | `9728*2560*144/256` | 14,008,320 each |
| Down, Q6_K | `2560*9728*210/256` | 20,428,800 |

This explicitly chosen mixture is about **63.87 MB per layer**, not 21.6 MB. Other quantization recipes have different mixtures. Calculate with the actual tensor inventory rather than inferring byte traffic from kernel names. At 4,096 tokens, this model's FP16 cache payload is 576 **MiB**, or 603.98 MB. A storage payload is not automatically the exact number of DRAM bytes every kernel transfers.

Run the [inference budget](../Edge%20LLM%20Inference%20Internals/Lecture-01.md) for a separately defined all-projections-Q4_K mixture. Changing the mixture changes the calculation; label both assumptions.

## 3. Fuse only operations whose contract you preserve

Q/K/V packing gives output width `4096+1024+1024=6144`. Gate/up packing gives `9728+9728=19456`, split at index 9728, followed by `SiLU(gate)*up`. The portable tests demonstrate concatenation equivalence and scalar FFN-related operations.

Fusion may reduce host launches, intermediate traffic, or repeated input reads. It does not automatically remove the much larger weight reads. Mixed quantized layouts require a compatible kernel or repacking; blindly concatenating Q4_K and Q6_K bytes does not create a valid homogeneous matrix.

Two ordinary launches on the **same CUDA stream are ordered**; submitting them there does not make them overlap or eliminate launch overhead. Separate streams permit concurrency only when dependencies and resources allow it. Measure timeline behavior before attributing a speedup to overlap.

## 4. CUDA Graphs reduce submission overhead

Graphs capture a dependency graph and replay it with lower host submission cost. They do not turn all device work into one kernel. Stable shapes/pointers are common implementation strategies; graph update APIs and runtime bucket strategies permit specific controlled changes. A decode query may have length one while the attended KV length changes every step.

A valid experiment warms up the runtime, records whether the graph path actually executes, and changes only graph enablement. Compare kernel work, CPU submission gaps, total latency and memory. Model loading, graph capture and steady-state generation are distinct measurements. Use the [CUDA graph documentation](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/cuda-graphs.html) for the installed toolkit; no GPU graph experiment was executed here.

## 5. Decode attention has a different score shape

For one query at context length `T`, each head's score vector has shape **`1×T`**, not `T×T`. A dense prefill of `T` queries has a `T×T` logical score matrix. Fused attention can stream tiles and maintain an online softmax rather than materializing all scores; decode kernels may also parallelize the long KV reduction. See [Flash-Decoding](https://crfm.stanford.edu/2023/10/12/flashdecoding.html).

For head `h`, use KV head `h//4` in this 4B case. Preserve the scale `1/sqrt(128)`, causal positions and cache validity. The optional [tiny Transformers CPU run](Lab-Guide.md) proves cached versus full inference parity under one actual implementation and detects a wrong-position injection. A hardware kernel needs equivalent numerical tests plus profiling.

KV quantization saves payload but adds metadata/dequantization and may affect task quality. FP8, Q8_0 and signed INT8 are different formats. Runtime support varies by hardware, K/V choice and attention backend; there is no universal “INT8 KV is quality-free” rule. Paging alone does not change element precision.

## 6. Speculation needs a measured accepted prefix

A draft proposes tokens; the target verifies them. Greedy matching and exact stochastic rejection sampling are different algorithms. For sampling, acceptance involves target/draft probabilities and a corrected rejection distribution. The [original algorithm](https://arxiv.org/abs/2211.17192) preserves the target distribution under its assumptions; approximate verification can change it.

A draft/target pair must satisfy the selected runtime's token-mapping and architecture requirements. The original dense Qwen3 family includes **0.6B**, not a release named “Qwen3-0.5B”; see the [official family release](https://qwenlm.github.io/blog/qwen3/). Same family does not guarantee a measured acceptance rate. Current llama.cpp has speculative facilities; inspect its [server documentation](https://github.com/ggml-org/llama.cpp/blob/master/tools/server/README.md) for supported methods rather than assuming the feature is absent.

Lecture 5 derives the prefix formula and includes draft/verification overhead. A larger accepted count can still lose wall-clock performance when drafting, extra weights/KV, or synchronization are expensive.

## Experiment and answers

Choose one change, preserve all other settings, and retain raw traces/results. For each run report quality, prompt processing, generation at the same context depth, request latency, peak memory, and thermal state. Do not add independent percentage gains as if they were guaranteed to compose.

- **Why can fusion have a small speedup despite fewer launches?** Most time may remain in weight/cache traffic or attention.
- **Why test long context separately?** KV size, attention work and cache allocation change with `T`.
- **Why does a wrong position matter even if cache length is correct?** Stored keys and the new query would represent inconsistent rotary coordinates.
- **Why can fast default clocks be a valid result?** Power policy is an experimental condition; correctness does not require locked clocks.

Next: [multi-GPU execution](Lecture-04.md), which uses the same tensor contracts with additional communication.
