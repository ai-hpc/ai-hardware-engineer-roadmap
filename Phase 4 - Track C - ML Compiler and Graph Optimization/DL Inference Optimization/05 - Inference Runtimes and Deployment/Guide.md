# 05 — Inference Runtimes & Deployment Targets

**Order:** Fifth. After graph, kernels, compiler, and quantization (01–04), you deploy and measure in production-like settings.

**Role target:** DL Inference Optimization Engineer · **MTS Kernels** (deployment, production reliability, measurable outcomes).

---

## Why this comes fifth

Your kernels and optimizations only matter if they run correctly in a runtime and meet latency/throughput goals. This unit covers the main inference runtimes and how to measure and compare them.

---

## 1. Runtimes

* **TensorRT** — Engine build, plugins, dynamic shapes, DLA. How your kernels and graph optimizations show up in the engine.
* **ONNX Runtime** — Execution providers (CUDA, TensorRT, OpenVINO). Graph optimizations and provider selection.
* **Triton Inference Server** — Batching, model concurrency, metrics. Serving multiple models and dynamic batching.

---

## 2. TensorRT-LLM — Production LLM Inference

[**TensorRT-LLM**](https://github.com/NVIDIA/TensorRT-LLM) is NVIDIA's open-source library purpose-built for LLM inference on NVIDIA GPUs. It extends TensorRT with LLM-specific optimizations that general-purpose runtimes cannot match. If you deploy large language models on NVIDIA hardware, this is the production path.

### Why TensorRT-LLM exists

Standard TensorRT handles vision and small models well, but LLMs have unique challenges:
- **KV-cache management** — grows linearly with context length, must be paged and reused across requests.
- **Autoregressive decoding** — each token depends on all previous tokens; batching is complex.
- **Multi-GPU serving** — models that don't fit on one GPU need tensor/pipeline parallelism.
- **Mixed workloads** — prefill (compute-bound) and decode (memory-bound) phases have opposite bottlenecks.

TensorRT-LLM solves all of these with a Python API that compiles LLMs into optimized TensorRT engines with LLM-specific runtime features.

### Core features

* **In-flight batching (continuous batching):**
    * Batch requests as they arrive — don't wait for a full batch. New requests join while others are mid-generation.
    * Maximizes GPU utilization by mixing prefill and decode phases across requests.

* **Paged KV-cache:**
    * Inspired by vLLM's PagedAttention — allocates KV-cache in fixed-size blocks, not contiguous per-sequence.
    * Eliminates memory fragmentation; enables serving more concurrent sequences.

* **Quantization:**
    * FP8 (Hopper+), INT8 (SmoothQuant), INT4 (AWQ, GPTQ) — all with fused dequantize in GEMM kernels.
    * FP4 on Blackwell for maximum throughput.

* **Tensor parallelism and pipeline parallelism:**
    * Split model across GPUs: tensor parallel (split within layers) or pipeline parallel (split across layers).
    * NCCL-based communication, overlapped with compute.

* **Speculative decoding:**
    * Draft model generates candidate tokens; main model verifies in one forward pass.
    * Reduces time-to-first-token and overall latency.

* **Custom Hopper/Blackwell kernels:**
    * CUTLASS-based GEMM kernels optimized for each GPU generation.
    * Warp specialization, persistent kernels, Transformer Engine integration.

* **CUDA Graphs:**
    * Captures the decode loop as a CUDA graph — eliminates per-token kernel launch overhead.

### Build and deploy workflow

```bash
# Install
pip install tensorrt-llm

# Step 1: Convert model checkpoint to TRT-LLM format
python convert_checkpoint.py \
    --model_dir ./llama-3-8b \
    --output_dir ./trt_ckpt \
    --dtype float16 \
    --tp_size 2           # tensor parallel across 2 GPUs

# Step 2: Build TRT-LLM engine
trtllm-build \
    --checkpoint_dir ./trt_ckpt \
    --output_dir ./trt_engine \
    --gemm_plugin float16 \
    --max_batch_size 64 \
    --max_input_len 2048 \
    --max_seq_len 4096 \
    --paged_kv_cache enable \
    --use_fused_mlp enable

# Step 3: Run inference
python run.py \
    --engine_dir ./trt_engine \
    --tokenizer_dir ./llama-3-8b \
    --input_text "Explain how a systolic array works"

# Step 4: Serve with Triton Inference Server
# TRT-LLM integrates with Triton via the TRT-LLM backend
# → in-flight batching, streaming, multi-model serving
```

### TensorRT-LLM vs vLLM

| | TensorRT-LLM | vLLM |
|---|---|---|
| **Approach** | Compile model to optimized engine (ahead-of-time) | JIT with PyTorch + custom CUDA kernels |
| **Performance** | Highest throughput on NVIDIA GPUs (custom Hopper/Blackwell kernels) | Very good; slightly lower peak but faster iteration |
| **Quantization** | FP8, INT8, INT4, FP4 with fused kernels | GPTQ, AWQ, FP8 via external libraries |
| **Multi-GPU** | TP + PP via NCCL | TP via NCCL |
| **Setup complexity** | Higher — build step required | Lower — load and serve |
| **Model support** | Major LLMs (Llama, Mistral, GPT, Falcon, etc.) | Broader model support via HuggingFace |
| **Hardware** | NVIDIA only | NVIDIA + AMD (ROCm) |
| **Best for** | Maximum throughput in production on NVIDIA | Rapid prototyping, AMD support, flexibility |

### Key concepts to internalize

* **Prefill vs decode phases** — Prefill processes the entire prompt in one pass (compute-bound, high arithmetic intensity). Decode generates one token at a time (memory-bound, low arithmetic intensity). TRT-LLM optimizes both with different kernel strategies.
* **KV-cache sizing** — For a 7B model at FP16 with 4096 context: ~2 GB KV-cache per sequence. With paged KV-cache, 80 GB H100 can serve ~30 concurrent sequences. Understanding this math is essential.
* **Engine build trade-offs** — `max_batch_size`, `max_input_len`, `max_seq_len` are baked into the engine. Larger = more memory reserved, fewer concurrent engines. Size for your actual workload.

---

## 3. Measurable outcomes

* **Latency** — p50, p99; what to measure (single request, batch). For LLMs: time-to-first-token (TTFT) and inter-token latency (ITL).
* **Throughput** — QPS, tokens/s; how batch size and concurrency affect it. For LLMs: output tokens/s across all concurrent requests.
* **Memory footprint** — Peak GPU/system memory; impact of batching and precision. For LLMs: model weights + KV-cache + activation memory.
* **Methodology** — Reproducible benchmarks; A/B comparison (e.g. before/after kernel change, or TensorRT-LLM vs vLLM).

---

## Resources

* [TensorRT Best Practices](https://docs.nvidia.com/deeplearning/tensorrt/best-practices/)
* [TensorRT-LLM GitHub](https://github.com/NVIDIA/TensorRT-LLM) — Source, examples, model support matrix.
* [TensorRT-LLM Documentation](https://nvidia.github.io/TensorRT-LLM/) — Build, deploy, and optimize guides.
* [vLLM](https://github.com/vllm-project/vllm) — Alternative LLM serving engine for comparison.
* [Triton Inference Server](https://github.com/triton-inference-server/server) — Production serving with TRT-LLM backend.
* [MLPerf Inference](https://mlcommons.org/benchmarks/inference/) — Reference benchmarks and methodology.

---

## Projects

1. **Runtime comparison** — Deploy the same model with ONNX Runtime and TensorRT (same hardware). Compare latency and throughput; document configuration and measurement method.
2. **Triton server** — Set up a minimal Triton server with dynamic batching. Measure QPS vs batch size and document how batching affects latency and throughput.
3. **Benchmark report** — For one model and one runtime, produce a one-page benchmark report: latency (p50/p99), throughput, memory, and exact environment (GPU, driver, runtime version).
4. **TensorRT-LLM engine build** — Build a TensorRT-LLM engine for Llama-3-8B with FP16 and INT8 quantization. Measure tokens/s, TTFT, and memory usage. Compare with vLLM on the same hardware.
5. **Multi-GPU LLM serving** — Deploy a 70B model across 2+ GPUs with tensor parallelism using TensorRT-LLM. Measure scaling efficiency (tokens/s per GPU) vs single-GPU with a smaller model.
6. **TRT-LLM + Triton** — Deploy TensorRT-LLM engine behind Triton Inference Server with in-flight batching enabled. Load test with concurrent clients and measure p50/p99 latency under load.

---

## Next

→ **[06 — tinygrad Deep Dive](../06%20-%20tinygrad%20Deep%20Dive/Guide.md)** (optional) — Hands-on compiler/kernel interface: IR, scheduler, backends, and adding a simple optimization.
