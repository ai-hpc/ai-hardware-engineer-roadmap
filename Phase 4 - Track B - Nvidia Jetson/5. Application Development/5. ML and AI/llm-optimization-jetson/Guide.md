# LLM optimization on Jetson: budget memory and verify the runtime

<div class="course-identity auto-course" style="--course-accent: #7c3aed; --course-accent-rgb: 124, 58, 237;" markdown="1">
<div class="course-identity__icon">LOOJ</div>
<div markdown="1">
<p class="course-identity__eyebrow">Deep Dive · Jetson Track</p>
<p class="course-identity__title">Specialized course identity for LLM Optimization on Jetson — From Cloud Techniques to Edge Reality.</p>
<p class="course-identity__meta">Artifact: Jetson integration demo · Measure: latency, memory, power, logs</p>
</div>
</div>

**Source check: 18 October 2026.** Learn to predict memory use, choose a supported model/runtime pair, and measure prefill and decoding separately. This is a deployment experiment, not a catalog of promised tokens per second.

Start with [ML deployment fundamentals](../Guide.md), the [Jetson compatibility record](../../../Compatibility.md) and the [memory architecture lesson](../../../1.%20Nvidia%20Jetson%20Platform/Orin-Nano-Memory-Architecture/Guide.md). Complete the [CPU export/calibration lab](../cpu-export-quantization/Guide.md) to see how an apparently reasonable precision change can alter behavior.

## 1. Resolve the software family before installing

| Runtime | What to verify |
|---|---|
| llama.cpp | Exact commit, supported model architecture and GGUF quantization, CUDA toolkit and SM target |
| Ollama | Exact release, model manifest/digest, supported JetPack route and resolved backend |
| TensorRT Edge-LLM | Its own platform, model and precision matrices; it is a distinct edge project |
| TensorRT-LLM | Its own datacenter/model support; a CUDA-capable Jetson is not an automatic supported target |

The current [TensorRT Edge-LLM matrix](https://nvidia.github.io/TensorRT-Edge-LLM/user_guide/getting_started/support-matrix.html) lists Orin with **JetPack 7.2, CUDA 13.2, platform TensorRT**, restricted to **FP16, INT8 and INT4** model engines. It lists Thor with selected 7.x stacks and model-dependent precision. It does not establish every patch-level combination; the course's JetPack 7.2.1 record must still be reconciled with the runtime's exact support statement.

The current datacenter TensorRT-LLM line removed its TensorRT engine backend: old `convert_checkpoint.py` → `trtllm-build` recipes are historical, while the current execution backend loads supported checkpoints through PyTorch. Do not use that old recipe as a Jetson installation procedure. [Official removal/migration guide](https://nvidia.github.io/TensorRT-LLM/latest/legacy/tensorrt-backend-removal.html).

Generic TensorRT 11.4 is not supported on JetPack; keep the platform-supported 10.x package. For Ollama, use its [hardware support documentation](https://docs.ollama.com/gpu) and record the exact model artifact. Pulling an arbitrary model tag does not guarantee a particular quantization such as Q4_K_M.

## 2. Count total resident weights, then everything else

A first lower bound for dense weight storage is

\[
M_W=P\times b/8,
\]

where `P` is the total parameter count and `b` the bits per parameter. A nominal 3-billion-parameter model needs 6 GB at 16 bits or 1.5 GB at 4 bits **before** scales, mixed-precision tensors, metadata and alignment. Measure the actual file and resident memory.

A mixture-of-experts model can activate only a few experts per token while retaining many more parameters. Active parameter count describes some computation; total resident experts determine memory unless an explicit offload scheme moves them. That movement has a latency and bandwidth cost.

On an 8 GB unified-memory board, CPU, GPU, camera buffers, services, workspace and KV cache share DRAM. Offloading weights to the CPU does not create a second physical 8 GB pool. Memory-mapped storage, page cache, pinned host memory and managed allocation have distinct accounting and performance behavior. Do not add mutually overlapping counters as if they were independent allocations.

Create a budget with actual weights, peak workspace, maximum KV allocation, application buffers, OS/services and headroom. Check cold-load peak too: conversion and temporary buffers can exceed steady-state memory.

## 3. Derive the KV cache from the configuration

For a conventional full-attention decoder with grouped-query attention:

\[
M_{KV}=2 L H_{KV} D T B s.
\]

`L` = layers, `H_KV` = key/value heads, `D` = head dimension, `T` = allocated tokens per sequence, `B` = concurrent sequences, and `s` = bytes per element. The factor 2 counts keys and values. Use KV heads, not query heads.

**Metadata-checked teaching model:** `HuggingFaceTB/SmolLM2-360M-Instruct`, revision `a10cc1512eabd3dde888204e902eca88bddb4951`, has 32 layers, 15 query heads, 5 KV heads, hidden size 960, and maximum-position configuration 8192. Thus `D=960/15=64`. With FP16 KV, 4096 tokens and one sequence:

```text
2 × 32 × 5 × 64 × 4096 × 1 × 2 = 167,772,160 bytes = 160 MiB
```

At 8192 tokens it is 320 MiB; two independently allocated sequences double that. This is an older, small teaching baseline, not a claim about the strongest current model. Its [official model card](https://huggingface.co/HuggingFaceTB/SmolLM2-360M-Instruct) and [immutable config](https://huggingface.co/HuggingFaceTB/SmolLM2-360M-Instruct/blob/a10cc1512eabd3dde888204e902eca88bddb4951/config.json) establish identity and license. Metadata was retrieved; its weights and Jetson inference were not run for this revision of the course.

This formula changes for sliding-window attention, recurrent/state-space layers, latent/compressed attention, shared prefixes, paging and runtime padding. Paged allocation reduces some waste; it does not erase the per-token tensors. Quantized KV adds scales and requires matching attention-kernel support.

## 4. Estimate a bound, then measure

If one decode step truly reads `W` bytes from DRAM and achieved bandwidth is `BW`, then `time >= W/BW`. For a hypothetical 2 GB read at a theoretical 102 GB/s, the bound is about 19.6 ms, or at most 51 such steps/s before other work. This is a mathematical bound, not an Orin measurement. Caching, batching, sparse expert activation, dequantization, KV reads and effective bandwidth alter the actual work.

Prefill processes many input tokens together and often uses matrix-matrix operations. Batch-one decode repeatedly applies weights to small vectors. A setting that improves prompt throughput can hurt decode latency or memory. The model's context length and concurrent requests must be fixed in a comparison.

## 5. A reproducible llama.cpp hardware experiment

The following commands were checked against source at commit `ff5a23376725e5a889fa0bfb987b7541dc304da0`; **CUDA compilation and inference were not executed here**. Use a CUDA toolkit supported by your installed BSP. The build shown targets Orin SM87; Thor requires its supported toolchain/SM110 configuration, not this architecture flag.

```bash
git clone https://github.com/ggml-org/llama.cpp.git /tmp/course-llama
cd /tmp/course-llama
git checkout ff5a23376725e5a889fa0bfb987b7541dc304da0
cmake -S . -B build -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=87
cmake --build build --config Release -j 2
./build/bin/llama-cli --help
./build/bin/llama-bench --help
```

Use a local GGUF whose original model revision, conversion revision, quantization, license and SHA256 are recorded. The commands below expect you to set its actual path; they do not download a large model or choose a community conversion silently.

```bash
export COURSE_GGUF=/absolute/path/to/your-verified-model.gguf
sha256sum "$COURSE_GGUF"
./build/bin/llama-cli -m "$COURSE_GGUF" -ngl 99 -c 2048 -n 64 \
  --flash-attn auto -p 'Explain why a robot should timestamp its camera frames.'
./build/bin/llama-bench -m "$COURSE_GGUF" -ngl 99 -p 512 -n 128 -r 5 -o json \
  > /tmp/course-llama-bench.json
```

`-ngl 99` requests extensive GPU placement; inspect the startup log for actual placement and memory. `llama-bench` reports synthetic token-processing measurements, not task quality or user-perceived streaming latency. Its prefill and generation tests are distinct. [Pinned build documentation](https://github.com/ggml-org/llama.cpp/blob/ff5a23376725e5a889fa0bfb987b7541dc304da0/docs/build.md), [benchmark definitions](https://github.com/ggml-org/llama.cpp/blob/ff5a23376725e5a889fa0bfb987b7541dc304da0/tools/llama-bench/README.md).

For a local service, retain one request slot while establishing the baseline:

```bash
./build/bin/llama-server -m "$COURSE_GGUF" -ngl 99 -c 2048 -np 1 \
  --flash-attn auto --host 127.0.0.1 --port 8080
```

Record the selected chat template and token IDs. A valid-looking HTTP response does not validate model-specific tokenization. Inspect the pinned [server contract](https://github.com/ggml-org/llama.cpp/blob/ff5a23376725e5a889fa0bfb987b7541dc304da0/tools/server/README.md) before changing context sharing, concurrency or speculative settings.

## 6. Change one bottleneck at a time

| Change | Why it may help | What can get worse |
|---|---|---|
| Lower weight precision | Reduces weight traffic/residency | Dequantization cost, unsupported kernels, task quality |
| Shorter context | Reduces KV storage and attention work | Missing task evidence/history |
| Quantized KV | Reduces some cache storage/traffic | Scale overhead, fallback, long-context quality |
| Fused attention | Avoids materializing intermediate attention matrices | Unsupported shapes/dtypes or fallback |
| CUDA Graphs | Reuses launch structure | Capture restrictions, graph variants and memory |
| Batching | Shares weight reads across requests | Tail latency and cache capacity |
| Speculative decoding | Verifies multiple draft tokens together | Extra model/state, rejected work and memory contention |

Speculation is not free speed. If the draft proposes `k` tokens, compare accepted tokens per verification with draft+verification time and include both models' memory. Exact speculative sampling also needs a correct acceptance/correction procedure; taking a draft's output without target verification changes the model distribution.

Orin's Ampere GPU does not acquire native FP8/FP4 Tensor Core support because a file has that dtype. Follow the runtime's supported representations and compute kernels. Current llama.cpp's `GGML_CUDA_FA_QUANTS` selects compiled K/V combinations; the old `GGML_CUDA_FA_ALL_QUANTS` spelling is a deprecated alias at the pinned revision. Do not force every kernel option before reading the trace.

## 7. Report user-visible outcomes

Measure cold start, time to first token, inter-token latency, end-to-end request duration, peak memory and joules per completed request under a declared load. Distinguish model-generated token counts from network/SSE chunks, which can combine or split text arbitrarily. Record prompt length, generated token count, cache state, sampling settings and cancellations.

Use held-out task prompts and exact evaluation criteria. For perplexity, compute total negative log likelihood over valid predicted tokens, divide by their count, then exponentiate. Mask padding and correctly shift targets. Different tokenizers make raw per-token perplexities incomparable; use a common task or normalized text measure when needed.

The [experimental native runtime lesson](../jetson-llm-runtime/Guide.md) is an implementation audit exercise. It does not establish an improvement over llama.cpp merely because it has custom kernels or a memory-budget class.

## 8. Independent exercise and answers

Build a worksheet for two local models with exact revisions. Calculate weight lower bounds and the architecture-specific KV budget at 1024/4096 tokens. On hardware, measure the same prompt/output lengths and one cancellation; without hardware, complete the arithmetic checks in the [CPU lab](../cpu-export-quantization/Guide.md) and mark the target run pending.

1. **A model activates 3B of 30B parameters. Can weight memory be calculated from 3B?** Only if the actual residency/offload design demonstrates that bound; ordinarily resident experts are counted too.
2. **Does 8 GB unified memory plus CPU offload mean 16 GB?** No. The physical pool is shared.
3. **Will halving KV element size halve total memory?** It halves that payload before metadata, not weights, workspace or the OS.
4. **Can the Edge-LLM matrix validate datacenter TensorRT-LLM?** No. They are different projects with separate contracts.
