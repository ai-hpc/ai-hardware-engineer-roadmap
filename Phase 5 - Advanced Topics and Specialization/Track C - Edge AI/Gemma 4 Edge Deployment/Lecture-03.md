# Lecture 03 — A reproducible runtime on the actual edge target

**Input:** verified weights and format from Lecture 02. **Output:** a service manifest and measurement protocol. [Previous](Lecture-02.md) · [Next: speculation](Lecture-04.md).

## 1. Prove the entire compatibility chain

A runtime supporting “Gemma” or “CUDA” is insufficient. Check exact architecture, quantization, modality/processor, CPU architecture, GPU capability, CUDA driver/toolkit, and host BSP. Text support may precede vision/audio support. A package import proves less than one correct inference.

| Candidate | Evidence required before using it |
|---|---|
| llama.cpp | Pinned converter/loader support, CUDA backend, actual multimodal path if used |
| LiteRT-LM | Documented target platform/backend, model package, tokenizer/processor, API for that release |
| MLC-LLM | Matching architecture, conversion/compile path, target backend and memory plan |
| TensorRT-LLM | Exact Jetson platform and release support plus the specific model implementation |

Consult each project's maintained source: [llama.cpp](https://github.com/ggml-org/llama.cpp), [LiteRT-LM](https://github.com/google-ai-edge/LiteRT-LM), [MLC-LLM](https://github.com/mlc-ai/mlc-llm), [TensorRT-LLM](https://github.com/NVIDIA/TensorRT-LLM). Do not install every runtime merely to fill a comparison table; one fully validated path is the baseline.

## 2. Build an auditable baseline

After checking out an immutable llama.cpp commit with support for your model, run the following **from that llama.cpp checkout**, where its `CMakeLists.txt` lives. A CUDA toolkit, C/C++ compiler, CMake and a supported build target are prerequisites. These commands follow the upstream [CMake build instructions inspected on October 18](https://github.com/ggml-org/llama.cpp/blob/ff5a23376725e5a889fa0bfb987b7541dc304da0/docs/build.md); this is an optional target-build exercise, not a build or Gemma inference executed by the supplied CPU lab:

```bash
cmake -S . -B build -DGGML_CUDA=ON -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release -j2
./build/bin/llama-cli --help
./build/bin/llama-bench --help
```

Choose build parallelism for available RAM. These commands build the runtime; they do not download or prove compatibility with a checkpoint. Query the target's compute capability rather than assuming Thor is `sm_90` because it is newer than Orin. Save CMake output and the runtime commit. Use the selected release's CLI options for loading and benchmarking; keep the exact command in the manifest. For images/audio, inspect the separate [multimodal implementation and model assets](https://github.com/ggml-org/llama.cpp/blob/ff5a23376725e5a889fa0bfb987b7541dc304da0/tools/mtmd/README.md); a successful text CLI invocation does not exercise that path.

Without the target/toolkit, complete the [supplied CPU metadata exercise](README.md#start-with-the-supplied-cpu-exercises) and retain its config hashes as the first manifest entries. Its meta-device parameter count allocates shapes, not pretrained tensors. Mark build, model quality, runtime cache, and target timing as **not run** until those separate experiments succeed.

For deployment, constrain supported input/context sizes and reject excessive requests before allocating cache. Include model and processor readiness in the health check. Cancellation must release queued work and buffer ownership without corrupting the next request.

## 3. Define the metrics before timing

- **Cold startup:** process start to ready-for-correct-inference, including load and initialization.
- **Service TTFT:** request arrival at the service to first emitted token; this includes its queue and tokenization time. A metric starting at model execution is a different, explicitly named subinterval. Client-visible TTFT also includes the relevant network transit.
- **Inter-token latency:** distribution of time between emitted tokens after the first.
- **Decode throughput:** committed generated tokens divided by the matching decode interval.
- **Aggregate throughput:** completed work across concurrent requests over wall time.

For physical AI, also measure observation-capture to usable result. A server can increase aggregate throughput while making every robot observation older.

**Worked queue example:** requests arrive every 50 ms but a serial service requires 60 ms each. Its maximum service rate is about 16.7 requests/s versus 20 arrivals/s; an unbounded queue grows. Batching or a faster kernel may help, but bounded admission and expiry are still required.

## 4. Thermal and memory controls

Record power mode, clocks, temperatures, and background services. Warm the system before steady-state measurements; report cold start separately. Keep prompt length, generation length, sampling policy, and image token budget fixed for comparisons. A runtime-reserved cache can occupy memory even with short prompts; distinguish reserved capacity from live tokens.

## Lab and acceptance

Produce a known-input smoke result, a manifest, and at least three context/concurrency cases with raw timestamps and memory. Include an oversized request, cancellation, and restart. Passing means correct bounded behavior and reproducibility, not reaching an unsourced token-rate target. Pass these measurements into Lecture 04 before adding speculative execution.
