# Study an LLM runtime one contract at a time

This directory is an **experimental teaching project**, with a working CPU course for sampling, GGUF inspection, and a deliberately small tokenizer. The CUDA engine remains a source-audit exercise; this checkout has no validated pretrained-model generation or Jetson performance result.

Start with the [CPU lab](labs/README.md). It builds on an ordinary Linux computer without CUDA or a model download:

```bash
cmake -S Projects/jetson-llm-runtime -B /tmp/jllm-course -DCMAKE_BUILD_TYPE=Release
cmake --build /tmp/jllm-course -j2
ctest --test-dir /tmp/jllm-course --output-on-failure
```

The default build includes the actual C++ sampler, bounded GGUF parser, configuration reader, toy tokenizer, and native tests. The [testing guide](TESTING.md) records the evidence and optional sanitizer workflow. The [roadmap](ROADMAP.md) defines the remaining work needed for a pretrained inference engine.

After the CPU lab, follow the short lessons on [memory and ownership](docs/memory.md), [kernel contracts](docs/kernels.md), [independent testing](docs/testing.md), [performance measurements](docs/performance.md), and [serving boundaries](docs/server.md). Each includes a worked example or a concrete review exercise.

| Component | Current teaching contract |
| --- | --- |
| Sampling | Explicit penalty/temperature/top-k/top-p order, inspectable probabilities, validated inputs, seeded replay |
| GGUF | Bounded little-endian v3 profile; typed metadata; supported tensor layouts, alignment, dimensions, nonoverlap and payload bounds |
| Model configuration | Exact metadata keys, declared head dimension, vocabulary/embedding consistency and actual tensor-byte sum |
| Tokenizer | `course.greedy.bytes` only; unsupported BPE/SentencePiece/Qwen profiles rejected |
| CUDA kernels and engine | Optional experimental sources; require numerical, model-parity, lifetime and target testing |
| Server and hardware management | Experimental source examples; no deployment or reliability claim |

The original repository link, [GeniePod/genie-ai-runtime](https://github.com/GeniePod/genie-ai-runtime), points to a separate implementation. Its README reviewed on 2026-10-18 identifies v1.3.1 and reports its own hardware results. Those results do not validate the code in this folder. Select and review a specific upstream revision when studying it; benchmark it under your own workload before making comparisons.

Orin Nano 8 GB has eight SMs, 1,024 CUDA cores, 32 Tensor Cores, and no DLA; Super mode does not add SMs. Query device properties and the active power mode instead of substituting the defaults in a header for measurements. The [course platform guide](../../Phase%204%20-%20Track%20B%20-%20Nvidia%20Jetson/1.%20Nvidia%20Jetson%20Platform/Guide.md) provides the current software/hardware context.

The optional Jetson source build uses `-DJLLM_BUILD_JETSON=ON` with a compatible aarch64 CUDA toolchain. It does not enable reference tokenizer support. INT8 KV and graph replay are rejected by the engine until their missing scale/state contracts are implemented and tested. The default CPU path is the complete exercise offered here.
