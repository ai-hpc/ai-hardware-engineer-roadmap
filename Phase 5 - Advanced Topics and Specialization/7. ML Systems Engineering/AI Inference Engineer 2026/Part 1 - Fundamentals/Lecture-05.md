# Part 1 · Lecture 05 — The Runtime Landscape (2026)

## Overview

The **runtime** is the layer between the model file and the GPU. It owns:

* The scheduler that turns requests into kernel launches.
* The kernel selection that decides whether attention is FlashAttention 4 or something older.
* The memory manager that owns the KV cache.
* The quantization integration that uses (or fails to use) the FP8 / FP4 / INT4 path on the hardware.
* The API surface that the rest of the stack talks to.

Picking the right runtime is the **single largest engineering decision** after picking the model. A **wrong choice costs 2–5× in throughput**, blocks features (paged KV, prefix cache, FP8), or limits the hardware you can target. A right choice means most of the hard kernel and scheduler work is already done.

This lecture covers the 2026 landscape:

1. **vLLM** — the open-source workhorse, V1 engine in 2026.
2. **SGLang** — RadixAttention + EP, the structured-output + MoE serving leader.
3. **TensorRT-LLM** — NVIDIA's max-throughput path, FP8 and FP4 native.
4. **llama.cpp** — the edge / desktop / CPU+GPU universal runtime.
5. **MLX** — Apple Silicon native.
6. **LMDeploy and others** — TurboMind kernels, secondary players worth knowing.

For each: what it does well, what it does badly, when to pick it.

By the end you should have a runtime decision matrix for any workload × hardware combination, and a defended starting point you can prototype on within an hour of receiving a new project.

---

## 1. vLLM — the open-source workhorse

**Project:** [github.com/vllm-project/vllm](https://github.com/vllm-project/vllm)
**Pinned version (this lecture):** 0.22.x with the **V1 engine** as the default path.

### 1.1 What it does

* **Continuous batching** — the headline feature; arbitrary mixing of prefill and decode in one batch, recomposed every step.
* **PagedAttention v2** — block-level KV memory management, eliminates the memory fragmentation that previously capped batch sizes.
* **Prefix cache** — shared system prompts and conversation prefixes hit the cache automatically.
* **Multi-LoRA serving** — many adapters can share one base model in a single replica.
* **Multi-GPU** — tensor parallelism (NCCL) and (in V1) better pipeline + EP support.
* **OpenAI-compatible API** — drop-in for clients that target OpenAI's REST shape.
* **Quantization** — AWQ, GPTQ, FP8 (Marlin kernels), GGUF (via experimental kernels).
* **Speculative decoding** — draft model, Medusa, EAGLE families supported.

### 1.2 V0 → V1 engine

vLLM's V0 engine (2023 → early 2025) accumulated complexity. The **V1 engine** (2025-onwards) is a rewrite that:

* Cleans up the scheduler-kernel boundary.
* Improves CUDA Graph integration on decode.
* Better multi-modal support (vision encoders inline).
* Lowers Python overhead — meaningful for short decodes.

**Always run V1 unless you have a specific reason** (a third-party plugin that hasn't migrated). V0 references should be treated as archaeology.

### 1.3 What it does badly

* Disaggregated prefill / decode (Mooncake-style) is still maturing as of mid-2026; not the V1 strong suit.
* Some bleeding-edge model architectures land later than in SGLang (the SGLang team often gets DeepSeek / Qwen3 support first).
* TensorRT-LLM still outperforms it on raw single-replica throughput when both can use FP8 on Hopper.

### 1.4 When to pick vLLM

* OpenAI-API-compatible serving with multiple models or LoRAs.
* Hopper or earlier hardware with mixed workloads (chat + agent + occasional batch).
* You want the broadest community support and the largest ecosystem of integrations.
* You need multi-modal (vision) inline serving.

**Default starting point for most production deployments.**

---

## 2. SGLang — RadixAttention + EP

**Project:** [github.com/sgl-project/sglang](https://github.com/sgl-project/sglang)
**Pinned version:** 0.5.x.

### 2.1 What it does

* **RadixAttention** — a radix tree over prefix tokens that lets *any* shared prefix (not just system prompts) hit the cache. This is the most aggressive prefix-cache architecture in any open runtime.
* **Expert parallelism (EP)** — first-class support for MoE models. SGLang is the open-source reference for serving DeepSeek V3 / Qwen3-MoE.
* **Structured output** — XGrammar and Outlines integrations; tool-call grammar enforcement is fast.
* **Speculative decoding** — EAGLE-2, EAGLE-3, MTP for DeepSeek.
* **Multi-GPU** — TP, PP, EP combinations including disaggregated P/D in current releases.
* **OpenAI-compatible API** with extended endpoints for structured output.

### 2.2 What it does badly

* Smaller community than vLLM; ecosystem of integrations is thinner.
* Documentation lags features; relies on the project's example notebooks.
* Multi-modal support exists but is less mature than vLLM's.

### 2.3 When to pick SGLang

* **MoE serving** — DeepSeek V3.1 / Qwen3-MoE at scale → SGLang first.
* **Agent / tool-use workloads** — XGrammar integration is fast; RadixAttention helps with repeated tool catalogs in system prompts.
* **RAG workloads** — heavy prefix sharing across requests; RadixAttention is the structural win.
* **Long-prompt batch processing** — prefix sharing on system instructions across many requests.

**Default starting point for MoE in Part 3.**

---

## 3. TensorRT-LLM — NVIDIA's max-throughput path

**Project:** [github.com/NVIDIA/TensorRT-LLM](https://github.com/NVIDIA/TensorRT-LLM)
**Pinned version:** 1.3.x.

### 3.1 What it does

* **TensorRT engine compilation** — model graph compiled into a hardware-specific engine plan; usually 1.5–2× faster than vLLM at peak single-replica throughput on the same Hopper GPU.
* **FP8 native** — first-class FP8 support (E4M3 weights, E5M2 KV), the most mature FP8 path among the runtimes.
* **FP4 / Blackwell** — TE2 FP4 path landed in the 1.x releases; the canonical Blackwell inference path will probably be TRT-LLM.
* **In-flight batching** — NVIDIA's continuous-batching equivalent.
* **Triton Inference Server integration** — production serving orchestration.
* **All NVIDIA kernels** — FlashAttention 4, Hopper TMA + WGMMA, Blackwell tcgen05 (UMMA), all hand-tuned.

### 3.2 What it does badly

* **Engine compilation friction** — each model + precision + batch-size-bucket needs a compiled engine plan. Iteration is slow.
* **Vendor lock-in** — NVIDIA hardware only. No AMD, no Apple, no edge ARM.
* **Configuration complexity** — many knobs; expert-level documentation; non-trivial to deploy without an NVIDIA solution architect's help.
* **Slower to support new model architectures** — TRT-LLM's release cycle often trails the open-source models by weeks.

### 3.3 When to pick TensorRT-LLM

* **Maximum single-replica throughput** on Hopper / Blackwell, model fixed.
* **Production deployment** where the engine plan can be compiled once and served for months.
* **FP8 or FP4 native** is required and the model is stable.
* **NVIDIA-only stack** is acceptable.

**Default for the cost-sensitive batch / high-throughput chat workloads on Hopper / Blackwell once the model has stabilized.**

---

## 4. llama.cpp — universal edge / desktop

**Project:** [github.com/ggml-org/llama.cpp](https://github.com/ggml-org/llama.cpp)
**Pinned version:** post-2026-04 builds.

### 4.1 What it does

* **Runs everywhere** — CPU (AVX2/AVX512/NEON), CUDA, ROCm, Metal (Apple), Vulkan, OpenCL. Single binary across desktop, server, and edge.
* **GGUF format** — single-file model bundles with embedded tokenizer + config + quantization.
* **K-quants and IQ-quants** — bespoke quantization formats (Q4_K_M, IQ4_XS, IQ3_S, etc.) — most diverse precision options.
* **Speculative decoding** — draft-model speculation supported.
* **Server mode** — `llama-server` exposes a HTTP/JSON API (OpenAI-compatible subset).
* **Mature for small models** — Qwen3-4B, Llama 3.2-1B, Phi-4-mini etc.

### 4.2 What it does badly

* **Throughput at scale** is far below vLLM/SGLang on Hopper — llama.cpp is designed for "one user, one machine," not "100 concurrent users, one cluster."
* **No paged KV in the vLLM sense** — its memory model is per-session.
* **No tensor parallelism worth using on Hopper-class GPUs** — TP-1 is the practical mode.

### 4.3 When to pick llama.cpp

* **Edge** — Jetson Orin Nano, Raspberry Pi 5, Mac mini, single-GPU desktop deployments.
* **Mac development** — the CPU path on Apple Silicon is excellent; for Apple deployment use MLX (§5) instead.
* **CPU-only deployments** — `llama.cpp` is the only mature option.
* **Bundled model + tokenizer** workflow where GGUF distribution simplifies operations.

**Default for any per-user, per-machine, edge or desktop scenario.**

---

## 5. MLX — Apple Silicon native

**Project:** [github.com/ml-explore/mlx](https://github.com/ml-explore/mlx)
**Pinned version:** 0.31.x.

### 5.1 What it does

* **Apple Silicon native** — uses unified memory + Apple GPU + Neural Engine where appropriate.
* **NumPy-like API** — familiar to PyTorch users.
* **MLX-LM** — separate package with LLM inference utilities, quantization (4-bit / 8-bit), generation loop.
* **Performance on M-series** — on M4 Max / M3 Ultra, MLX outperforms llama.cpp Metal backend at most precisions because it uses the Apple GPU more aggressively.

### 5.2 What it does badly

* **Apple-only** — useless on NVIDIA / AMD.
* **Smaller community than llama.cpp** — fewer model conversions available.
* **Server mode is minimal** — geared at single-user inference.

### 5.3 When to pick MLX

* **Apple Silicon deployment** — Mac Studio, MacBook Pro, deployed AI appliance on Mac mini.
* **Local dev on Mac** with high-fidelity inference (vs llama.cpp's Metal path).
* **Single-user, high-end Mac** scenario where you want to use 64 GB of unified memory for a 70B model.

---

## 6. The secondary tier

Worth knowing, rarely the first pick:

* **LMDeploy** ([github.com/InternLM/lmdeploy](https://github.com/InternLM/lmdeploy)) — InternLM's runtime with TurboMind kernels. Strong on Chinese-language workloads, integrates well with the InternLM model family. Sometimes faster than vLLM on a single A100 / H100.
* **Hugging Face TGI** ([github.com/huggingface/text-generation-inference](https://github.com/huggingface/text-generation-inference)) — production-quality but feature pace has slowed; vLLM has eaten most of TGI's lunch.
* **Triton Inference Server** ([github.com/triton-inference-server/server](https://github.com/triton-inference-server/server)) — NVIDIA's orchestration layer. Hosts TensorRT-LLM, vLLM, or anything else. Not an LLM runtime itself; the *deployment* layer.
* **Ray Serve / Ray LLM** — Anyscale's distributed serving layer. Used for routing across replicas, multi-region deployments.

---

## 7. The decision matrix

| Workload | Hardware | First pick | Second pick | Notes |
|----------|----------|------------|-------------|-------|
| Chat, dense 70B, OpenAI-API serving | H100/H200, 8× | **vLLM** | TensorRT-LLM | vLLM if mixed workloads; TRT-LLM if throughput-only |
| Chat, MoE 200B+ | H200 / B200, 8×+ | **SGLang** | vLLM | SGLang's EP support is mature |
| Agent / tool-use with structured output | any | **SGLang** | vLLM | XGrammar integration is fast |
| Batch / offline embedding | H100 | TensorRT-LLM | vLLM | TRT-LLM throughput wins |
| Batch / offline LLM | H200 / B200 | **TensorRT-LLM** | vLLM | engine compile pays off |
| RAG with heavy prefix sharing | H100 / H200 | **SGLang** | vLLM | RadixAttention wins |
| Edge / single user | Jetson, Mac, RPi | **llama.cpp** | MLX (Mac) | llama.cpp is universal |
| Apple Silicon | M-series | **MLX** | llama.cpp | MLX is faster on Apple GPU |
| Multi-modal vision | H100 / H200 | **vLLM** | SGLang | vLLM's multi-modal support is most mature |
| MoE on Blackwell | B200 / GB200 | **SGLang** | TensorRT-LLM | SGLang EP + FP4 path |

Starting point — not eternal truth. Re-bench when your hardware or model changes.

---

## 8. The shape of the runtime layer

All five runtimes have the same architectural skeleton:

```text
HTTP / gRPC API
       │
       ▼
   scheduler  ──►  pending request queue
       │              │
       ▼              ▼
   model graph   ──►  batched forward pass
       │              │
       ▼              ▼
    kernels      ──►  attention, FFN, sampling
       │              │
       ▼              ▼
  KV manager    ──►  paged KV / radix tree / per-session
       │
       ▼
    HBM / fabric
```

Where they differ:

* **Scheduler** — vLLM and SGLang have continuous batching; llama.cpp is per-session; TRT-LLM has in-flight batching.
* **KV manager** — vLLM PagedAttention v2 vs SGLang RadixAttention vs llama.cpp per-session.
* **Kernels** — vLLM uses Triton + C++; SGLang uses its own kernels; TRT-LLM uses NVIDIA kernels; llama.cpp uses GGML kernels.
* **Quantization** — each has its own integration of AWQ / GPTQ / FP8 / GGUF.

A senior engineer can read any runtime's source and locate each box. **Junior engineers learn one runtime; seniors learn *the boxes*** and apply them to whichever runtime they're handed.

---

## Lab — bench three runtimes on one model

Goal: produce a same-machine, same-model comparison across three runtimes.

1. **Pick one model** — Qwen3-4B Instruct (AWQ-INT4 or original BF16, both available).
2. **Pick three runtimes** — vLLM (V1), llama.cpp (post-2026-04), and one of: SGLang or TensorRT-LLM.
3. **Same hardware**, same warmup, same prompt set.
4. **Bench**:
   * Chat shape: prompt length 512, output 128, concurrency 1, 8, 32.
   * Agent shape: prompt 256, output 16, concurrency 16 (low-latency).
5. **Report** TTFT, TPOT, throughput, peak HBM, with profiler traces for one run.
6. **Pick a winner** per shape and defend in one paragraph.

Pass criterion: the report is in your benchmark repo. Anyone reading it can see *why* one runtime won at one shape.

---

## Self-check

1. You receive a project: serve Qwen3-MoE 235B-A22B for an agent product on 8× H200. What runtime do you start with, and what do you check before committing?
2. A teammate insists on TensorRT-LLM for a chat product with a model that ships a new fine-tune every week. Defend or reject the choice in two sentences.
3. You are shipping an LLM appliance on Jetson Orin Nano Super 8 GB. Which runtime, which model size, which quantization? Why?
4. SGLang's RadixAttention wins on RAG. Why specifically does RAG benefit more than chat?
5. Your benchmark shows vLLM at 920 tok/s and TRT-LLM at 1430 tok/s on Llama 3.3 70B at FP8 on H200, single replica. You ship vLLM anyway. Give one engineering reason where this is the right call.

---

## References

* vLLM — [docs.vllm.ai](https://docs.vllm.ai) · GitHub [vllm-project/vllm](https://github.com/vllm-project/vllm)
* SGLang — [sgl-project.github.io](https://sgl-project.github.io/) · GitHub [sgl-project/sglang](https://github.com/sgl-project/sglang)
* TensorRT-LLM — [nvidia.github.io/TensorRT-LLM/](https://nvidia.github.io/TensorRT-LLM/) · GitHub [NVIDIA/TensorRT-LLM](https://github.com/NVIDIA/TensorRT-LLM)
* llama.cpp — GitHub [ggml-org/llama.cpp](https://github.com/ggml-org/llama.cpp)
* MLX — [ml-explore.github.io/mlx/](https://ml-explore.github.io/mlx/build/html/index.html) · GitHub [ml-explore/mlx](https://github.com/ml-explore/mlx)
* LMDeploy — GitHub [InternLM/lmdeploy](https://github.com/InternLM/lmdeploy)
* HF TGI — GitHub [huggingface/text-generation-inference](https://github.com/huggingface/text-generation-inference)
* Triton Inference Server — [docs.nvidia.com/deeplearning/triton-inference-server/](https://docs.nvidia.com/deeplearning/triton-inference-server/)

Cross-references:

* [Phase 5 → Edge AI → Qwen Inference Optimization](../../../3.%20Edge%20AI/Qwen%20Inference%20Optimization/README.md) — runtime + model walkthroughs on edge
* [Phase 5 → ML Systems Engineering Guide → Stage 4 Inference Serving Systems](../../Guide.md#stage-4-inference-serving-systems)

---

## Current as of 2026-06

Versions pinned: vLLM 0.22.x V1, SGLang 0.5.x, TensorRT-LLM 1.3.x, llama.cpp post-2026-04, MLX 0.31.x. Update when vLLM V1 ships as the unconditional default, or when a runtime ships a breaking API change, or when TRT-LLM's Blackwell FP4 path lands a stable release.

---

## Next

* Next: [Part 2 — Dense at Hopper](../Part%202%20-%20Dense%20at%20Hopper/README.md) — concrete Llama 3.3 70B ↔ Qwen 2.5 72B walkthroughs
* Previous: [Lecture 04 — The precision stack](Lecture-04.md)
* Up: [Part 1 — Fundamentals](README.md)
