# DL Inference Optimization (Part 2 of Phase 4 Track C)

<div class="course-identity dl-inference" markdown="1">
<div class="course-identity__icon">INF</div>
<div markdown="1">
<p class="course-identity__eyebrow">Track C · DL Inference Optimization</p>
<p class="course-identity__title">Profile, fuse, quantize, kernel-optimize, and deploy neural networks for real inference targets.</p>
<p class="course-identity__meta">Artifact: optimized inference pipeline · Measure: latency, throughput, memory, accuracy</p>
</div>
</div>


**Role target:** DL Inference Optimization Engineer

**Also aligns with:** Kernel-focused roles such as **Member of Technical Staff, Kernels** (e.g. AGI/LLM companies): designing and implementing high-performance kernels for training and inference, long-context optimization, and production deployment on NVIDIA GPUs and alternative accelerators (TPU, etc.).

**Prerequisite:** Track C Part 1 (at least §1–§2 and §5), Phase 4 Track B (Jetson, TensorRT, CUDA).

**Parent guide:** [Phase 4 Track C — ML Compiler & Graph Optimization](../Guide.md)

---

## Basic concepts (read this first)

Before diving into graph optimization, kernels, and compilers, you need the **vocabulary and mental model** of modern LLM inference and why kernel engineers are critical. This section sets the stage for the rest of the track.

### LLM inference: TensorRT-LLM, vLLM, and core optimizations

Production LLM serving relies on:

* **In-flight batching (dynamic request batching)** — Batch requests as they arrive; don’t wait for a full batch. Improves throughput without killing latency.
* **Paged KV-cache** — Attention needs key/value cache per token; long context = huge memory. Paging and reuse make it memory-efficient.
* **Speculative decoding** — Draft multiple tokens with a small model, verify with the big model; fewer forward passes for the same output.
* **EAGLE decoding & multi-token prediction** — Predict several tokens per step to cut latency.
* **Throughput vs latency** — Batch more → higher throughput, worse latency. You tune batching and scheduling to the SLA.

Frameworks you’ll work with: **TensorRT-LLM**, **vLLM**, Hugging Face integration, NVIDIA NGC containers. Models: Llama 3/4, DeepSeek R1, Qwen 3, Gemma 3, Phi 4, T5/BART. Optimization techniques: quantization (INT8, FP8, FP4), LoRA integration, kernel fusion.

### Advanced attention & memory

* **KV-cache sharding, paging, reuse** — Spread or page the cache across devices and reuse memory across requests.
* **Long-context optimization** — 100K–1M+ tokens; memory bandwidth and layout dominate. Efficient attention kernel design is the lever.
* **Memory bandwidth vs compute** — Many inference workloads are memory-bound. You optimize data movement and reuse.

### Distributed inference & training

When the model or batch doesn’t fit on one GPU:

* **Data parallelism** — Same model, different data; sync gradients (e.g. AllReduce).
* **Model / tensor parallelism** — Split layers or tensors across GPUs.
* **Pipeline parallelism** — Different layers on different GPUs; keep the pipeline full.
* **Expert parallelism (MoE)** — Scale mixture-of-experts by sharding experts.

At scale, **communication and synchronization** dominate. **NCCL** (and alternatives) become the bottleneck. Kernel engineers overlap compute with communication and reduce memory movement; that alone can yield 20–40% speedup (e.g. compute + async sync instead of compute → sync → compute → sync).

### Production inference systems

* **Disaggregated serving** — Split context encoding vs token generation across GPUs or nodes.
* **Continuous batching** — Add and remove requests from the batch without full flush.
* **High-throughput serving** — Architecture and scheduling for millions of requests and 100B+ parameter models.
* **GPU resource scheduling** — Utilization, fairness, multi-tenant.

### CuTe DSL (CUDA Template Engine) — why it shows up in kernel work

When you read CUTLASS, cuBLASLt, or kernel talks, you’ll see **CuTe** (CUDA Template Engine). It’s a C++ header library and DSL that defines **layouts** (how tensor dimensions map to memory: shape + stride, possibly tiled) and **copy** operations (vectorized, async, composable loads/stores). Kernel authors use CuTe to describe tiling (e.g. block tile, warp tile, thread tile) and data movement between global memory, shared memory, and registers without hand-written indexing. That makes it easier to get peak performance and to retarget when hardware changes (e.g. new tile sizes on Blackwell). In this track you’ll meet it in **02 — Kernel Engineering** (CUTLASS/CuTe) and when studying production GEMM/attention kernels.

### New architecture = new kernel challenges

Every new GPU generation (e.g. **NVIDIA Blackwell**, Hopper, Ada Lovelace) changes:

* **Execution model** — Warp scheduling, occupancy, how many warps hide latency.
* **Memory hierarchy** — Registers, shared memory, L2, HBM sizes and bandwidth.
* **Instruction throughput** — New ops (e.g. Transformer Engine, FP4), different optimal tile sizes.

Old kernels and tile sizes can become suboptimal or wrong. **Tiling** (blocks that fit in shared memory/registers) must be retuned: older GPUs → smaller tiles; newer GPUs → larger shared memory → bigger tiles. Wrong tile size → low occupancy, memory stalls. **Warp scheduling** and **memory latency patterns** also change; you measure and adapt instead of reusing old tricks.

### Why hardware-specific optimization matters (e.g. Blackwell)

* **Built for next-gen LLMs** — Huge transformers, long context, high-throughput inference.
* **Memory is the bottleneck** — LLMs are often memory-bound. Blackwell improves HBM and data movement; your job is to exploit it in attention and layout.
* **Transformer Engine / low precision** — FP4 and mixed-precision pipelines; you write kernels that use them and stay numerically stable.
* **Multi-GPU scaling** — Better NVLink/interconnects; critical for distributed training and large inference clusters.

Companies that ask for “Blackwell experience” mean: *can you get the most out of the latest hardware before everyone else?* That implies profiling (Nsight Compute, Nsight Systems), first-principles reasoning (bandwidth vs compute, latency vs occupancy), and throwing away old assumptions when they don’t hold.

### Engineering focus

* **End-to-end pipeline optimization** — From graph to deployed kernel.
* **Profiling and bottleneck analysis** — Where is time spent? Why is the SM idle? Where are the stalls?
* **Scalable deployment** — Single GPU → multi-GPU → clusters.
* **Production-grade reliability and performance tuning** — Measurable latency/throughput, not just benchmarks.

**In one sentence:** This role is about turning new GPU hardware into real-world AI performance gains before anyone else knows how.

---

## How to use this track

Study the topics **in order**. Each folder is one unit; do them one by one. The order is chosen so the most important and foundational topics come first, and each unit builds on the previous.

| Order | Folder | What you learn |
|:-----:|--------|----------------|
| **1** | [01 — Graph and Operator Optimization](01%20-%20Graph%20and%20Operator%20Optimization/Guide.md) | What we're optimizing: graph, ops, fusion, profiling, bottlenecks. **Do this first.** |
| **2** | [02 — Kernel Engineering](02%20-%20Kernel%20Engineering/Guide.md) | How to write and own kernels: Triton, CUTLASS, Flash-Attention, long-context, NCCL, production. **Core of MTS Kernels.** |
| **3** | [03 — Compiler Stack](03%20-%20Compiler%20Stack/Guide.md) | How compilers produce kernels: IR, scheduling (BEAM), codegen, TVM, MLIR. |
| **4** | [04 — Quantization](04%20-%20Quantization/Guide.md) | Low-precision inference: PTQ, QAT, INT8/INT4, kernel and runtime integration. |
| **5** | [05 — Inference Runtimes and Deployment](05%20-%20Inference%20Runtimes%20and%20Deployment/Guide.md) | Production: TensorRT, ONNX Runtime, Triton server; latency, throughput, methodology. |
| **6** | [06 — tinygrad Deep Dive](06%20-%20tinygrad%20Deep%20Dive/Guide.md) | Optional: hands-on IR, scheduler, backends; compiler–kernel interface. |

---

## Learning path (one by one)

1. **Start:** [01 — Graph and Operator Optimization](01%20-%20Graph%20and%20Operator%20Optimization/Guide.md) — Understand the graph and find bottlenecks before writing kernels.
2. **Then:** [02 — Kernel Engineering](02%20-%20Kernel%20Engineering/Guide.md) — Design and implement high-performance kernels (Triton, CUTLASS, attention, long-context).
3. **Then:** [03 — Compiler Stack](03%20-%20Compiler%20Stack/Guide.md) — See how IR, scheduling, and codegen feed kernel selection and fusion.
4. **Then:** [04 — Quantization](04%20-%20Quantization/Guide.md) — Add low-precision kernels and deployment (PTQ, QAT).
5. **Then:** [05 — Inference Runtimes and Deployment](05%20-%20Inference%20Runtimes%20and%20Deployment/Guide.md) — Deploy and measure in production-like settings.
6. **Optional:** [06 — tinygrad Deep Dive](06%20-%20tinygrad%20Deep%20Dive/Guide.md) — Hands-on compiler and backend work.

---

## Summary

| Area | Key skills |
|------|------------|
| Graph & operators | Fusion, constant folding, profiling, bottleneck analysis |
| **Kernel engineering** | Triton, CUTLASS, CuTe; Flash-Attention, long-context; NCCL/MSCCLPP; TPU/Pallas/Mojo; testing, correctness, porting |
| Compiler | IR, scheduling (e.g. BEAM), codegen, TVM/MLIR concepts |
| Quantization | PTQ, QAT, INT8/INT4, tooling (TensorRT, ONNX Runtime) |
| Runtimes | TensorRT, ONNX Runtime, Triton; latency/throughput methodology |

This track supports the **DL Inference Optimization Engineer** role and **kernel-focused roles** (e.g. Member of Technical Staff, Kernels): from graph-level optimization and custom kernel design to deployment with measurable latency and throughput, including porting to alternative hardware and production reliability.
