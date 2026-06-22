# 7. ML Systems Engineering (Phase 5)

<div class="course-identity mlsys" markdown="1">
<div class="course-identity__icon">SYS</div>
<div markdown="1">
<p class="course-identity__eyebrow">Track 5G · ML Systems Engineering</p>
<p class="course-identity__title">Build the runtimes, schedulers, kernels, training systems, and serving infrastructure behind AI at scale.</p>
<p class="course-identity__meta">Artifact: MLSys benchmark/runtime · Measure: latency, memory, communication, utilization</p>
</div>
</div>


> Build the runtimes, distributed systems, kernels, schedulers, and infrastructure that make modern AI workloads train and serve reliably.

**Layer mapping:** L3-L8. This track connects model math, CUDA kernels, runtime scheduling, distributed communication, cluster orchestration, compiler/runtime work, serving systems, and observability.

**Role targets:** ML Systems Engineer · AI Infrastructure Engineer · Inference Systems Engineer · Training Systems Engineer · GPU Runtime Engineer · Edge AI Runtime Engineer

**Prerequisites:** [Operating Systems](../../Phase%201%20-%20Foundational%20Knowledge/3.%20Operating%20Systems/Guide.md), [C++ and Parallel Computing](../../Phase%201%20-%20Foundational%20Knowledge/4.%20C%2B%2B%20and%20Parallel%20Computing/Guide.md), [Neural Networks](../../Phase%203%20-%20Artificial%20Intelligence/1.%20Neural%20Networks/Guide.md), [Deep Learning Frameworks](../../Phase%203%20-%20Artificial%20Intelligence/2.%20Deep%20Learning%20Frameworks/Guide.md), one Phase 4 deployment path, and enough [GPU Infrastructure](../1.%20GPU%20Infrastructure/Guide.md) to run and debug GPU benchmarks.

**What comes after:** a public systems artifact: inference runtime, distributed training runbook, CUDA/Triton kernel benchmark suite, scheduler prototype, compiler/runtime demo, or edge MLSys case study with reproducible measurements.

---

## Course Contract

This is not a course about using ML frameworks as black boxes. It is a course about the systems below those frameworks.

By the end, you should be able to:

- trace a transformer through tensor shapes, memory traffic, kernel launches, and communication calls
- separate prefill, decode, training forward pass, backward pass, optimizer state, KV cache, and scheduler overhead
- build a small inference runtime with batching, streaming, cancellation, and backpressure
- run and profile DDP/FSDP/ZeRO training experiments instead of only reading about them
- write and benchmark at least one CUDA or Triton kernel used by a transformer path
- explain when a workload is compute-bound, memory-bound, communication-bound, or scheduler-bound
- use profiler traces, counters, and logs as the default source of truth
- ship a reproducible benchmark that another engineer can run

The point is not to become a generic model trainer. The point is to become the engineer who knows why a model is slow, expensive, unstable, underutilized, or hard to deploy.

---

## Why This Belongs In An AI Hardware Roadmap

Hardware is only useful when the software stack can feed it.

MLSys is where workload reality meets hardware reality:

```text
model architecture
  -> framework graph
  -> runtime scheduler
  -> memory planner
  -> kernels
  -> collectives
  -> cluster fabric
  -> observability
  -> production behavior
```

If you want to design accelerators, edge AI platforms, or AI infrastructure, you need to understand what the workload actually asks the machine to do:

- how many bytes the KV cache consumes
- how attention changes with sequence length
- why decode often becomes memory-bandwidth limited
- why training stalls on all-reduce or activation memory
- why one extra synchronization point can destroy scaling
- why a scheduler can make a fast kernel look slow
- why a topology mismatch can break distributed inference

MLSys is the discipline of finding those constraints, measuring them, and changing the system so the hardware does useful work more often.

---

## The Core Mental Model

Most MLSys work reduces one of five costs:

```text
T_total = T_compute + T_memory + T_communication + T_synchronization + T_scheduling
```

For inference:

```text
T_latency = T_prefill + T_decode + T_queueing + T_communication + T_streaming
```

For training:

```text
T_step = T_forward + T_backward + T_optimizer + T_allreduce + T_checkpoint + T_sync
```

For memory:

```text
M_training = M_weights + M_activations + M_gradients + M_optimizer + M_workspace
M_inference = M_weights + M_kv_cache + M_workspace + M_batch_state
```

For distributed scaling:

```text
efficiency = useful_compute_time / wall_clock_time
```

Good MLSys engineering starts with a hypothesis and ends with measurement:

```text
observe -> isolate -> change one thing -> benchmark -> profile -> explain -> repeat
```

Avoid vague claims like "this is faster." Use concrete claims:

- p95 TTFT dropped from 420 ms to 260 ms at 32 concurrent requests
- decode throughput increased from 38 to 51 tok/s after KV-cache compaction
- all-reduce time fell from 31 percent to 18 percent of step time after bucket tuning
- peak training memory fell by 42 percent after activation checkpointing
- GPU utilization improved because queue starvation was fixed, not because kernels changed

---

## Course Map

<div class="lecture-map" markdown>

| Stage | Focus | Core artifact |
|-------|-------|---------------|
| 0 | Measurement discipline | benchmark harness and profiling template |
| 1 | Systems runtime foundations | async inference-like server with backpressure |
| 2 | Transformer execution internals | transformer shape, memory, and throughput report |
| 3 | GPU kernels and CUDA performance | CUDA/Triton kernel benchmark with roofline notes |
| 4 | Inference serving systems | continuous batching or paged KV-cache prototype |
| 5 | Distributed training systems | DDP/FSDP/ZeRO benchmark and bottleneck report |
| 6 | AI infrastructure and orchestration | reproducible cluster/runbook with failure recovery |
| 7 | Compiler and runtime layer | graph lowering, fusion, or memory-planning demo |
| 8 | Research and source-code loop | paper reproduction or source-code deep dive |

</div>

Recommended order:

1. If you already build edge inference runtimes: Stage 0 -> 1 -> 2 -> 3 -> 5.
2. If you want AI infrastructure roles: Stage 0 -> 1 -> 4 -> 5 -> 6.
3. If you want compiler/runtime roles: Stage 0 -> 2 -> 3 -> 7 -> 8.
4. If you want edge MLSys: Stage 0 -> 1 -> 2 -> 3 -> 4, then add Stage 5 selectively.

---

## Stage 0: Measurement Discipline

### Why It Matters

MLSys work without measurement turns into tool tourism. Before changing runtimes or kernels, build the habit of collecting comparable numbers. Performance numbers (latency, throughput) tell you how *fast* the system runs; **quality** numbers — logprobs, perplexity, KL divergence — tell you whether your optimization *preserved the model*. A 2× speedup that quietly raises perplexity 15% is a regression, not a win.

> **★ Companion course — quality measurement:** [**Logprobs, Perplexity & KL Divergence**](Logprobs,%20Perplexity%20and%20KL%20Divergence/README.md) is the foundations course for this stage. It derives `cross-entropy = entropy + KL`, `perplexity = exp(cross-entropy)`, and shows how llama.cpp grades a quantization with [mean KLD and top-token agreement](Logprobs,%20Perplexity%20and%20KL%20Divergence/Lecture-05.md) — exactly the "did the optimization break the model?" gate Stage 0 is about. 5 lectures.

### Learn

- latency percentiles: p50, p95, p99
- throughput: requests/sec, tokens/sec, samples/sec, tokens/sec/GPU
- GPU counters: occupancy, memory bandwidth, tensor-core utilization, SM activity
- memory metrics: peak allocated, fragmentation, KV-cache blocks, activation memory
- **quality metrics: perplexity / bits-per-byte, KL divergence vs. the FP16 reference, top-1 token agreement** (see companion course above)
- distributed metrics: collective time, bandwidth, GPU idle time, rank skew
- reliability metrics: error rate, retry rate, checkpoint restore time, failed-node recovery

### Build It

Create a small benchmark harness that can run the same workload repeatedly and emit:

- command line used
- hardware and driver summary
- git commit hash
- model or synthetic workload config
- concurrency level or batch size
- warmup count and measured iteration count
- CSV or JSON result file

### Use It In The Real Stack

Use the harness for every later stage. Do not hand-copy benchmark numbers into notes. Generate them from scripts.

### Measure It

Your harness should report:

- mean and percentile latency
- throughput
- peak memory
- CPU and GPU utilization if available
- profiler trace path

### Ship It

Ship `bench/`, `results/`, and `reports/` directories with one reproducible baseline. The baseline can be synthetic, but it must be rerunnable.

---

## Stage 1: Systems Runtime Foundations

### Why It Matters

An inference or training service is still a distributed Linux program. It queues work, moves bytes, schedules threads, manages memory, handles failure, and emits telemetry.

### Learn

- Linux processes, threads, signals, cgroups, namespaces, and filesystems
- memory mapping, page faults, huge pages, pinned memory, NUMA, and zero-copy paths
- concurrency with threads, locks, atomics, queues, work stealing, cancellation, and backpressure
- networking with TCP, HTTP streaming, gRPC, `epoll`, `io_uring`, and RDMA concepts
- CPU performance: cache locality, SIMD basics, `perf`, flamegraphs, and tracing
- production runtime basics: structured logs, metrics, traces, health checks, and graceful shutdown

Languages:

- Python for ML ecosystem integration
- C++ for runtime and CUDA integration
- Rust for infrastructure and systems components when it fits the project

### Build It

Build an inference-shaped server without a real model first:

1. Accept requests with prompt length, max tokens, priority, and deadline.
2. Put requests into a scheduler queue.
3. Batch compatible requests every few milliseconds.
4. Stream fake tokens back to clients.
5. Support cancellation.
6. Apply backpressure when queues or memory budgets are exceeded.

Then add a second component:

- tokenizer runtime with zero-copy request parsing, or
- mini tensor runtime with explicit allocation and shape tracking, or
- memory arena for request state and KV-cache-like blocks.

### Use It In The Real Stack

Compare your design to the control-plane responsibilities in vLLM, SGLang, Ray Serve, and Triton Inference Server. Focus on what the runtime schedules and what the GPU kernels actually execute.

### Measure It

- p50/p95/p99 latency under increasing concurrency
- queue wait time versus execution time
- throughput under different batching windows
- allocation count and peak RSS
- CPU utilization, lock contention, and context switches
- cancellation latency

### Ship It

Ship an async runtime with a load generator, dashboard or metrics endpoint, and a short report explaining when batching helps and when it hurts tail latency.

---

## Stage 2: Transformer Execution Internals

### Why It Matters

MLSys engineers do not need to invent every model architecture, but they must understand the compute and memory flow of the models they serve or train.

Transformer inference flow:

```text
tokens -> embeddings -> attention(Q, K, V) -> MLP -> residual -> logits -> sampler
```

Transformer training flow:

```text
forward -> loss -> backward -> gradients -> optimizer step -> updated weights
```

Attention:

```text
Attention(Q, K, V) = softmax((Q * K^T) / sqrt(d_k)) * V
```

### Learn

- embeddings, attention, MLP, residual paths, RMSNorm/layer norm, logits, and sampling
- QKV projection, grouped-query attention, RoPE, ALiBi-style position handling, and KV cache
- prefill versus decode
- activation memory and gradient flow
- optimizer state memory
- mixed precision, loss scaling, and gradient accumulation
- batching, speculative decoding, quantization, and long-context behavior

### Build It

Build a small transformer path that can run both training and inference:

1. Print tensor shapes at every major operation.
2. Track activation memory during training.
3. Track KV-cache memory during inference.
4. Separate prefill and decode timing.
5. Add a minimal sampler.
6. Add gradient accumulation and mixed precision.

### Use It In The Real Stack

Map your toy implementation to PyTorch modules and to production runtime concepts:

| Concept | Training system concern | Inference system concern |
|---------|-------------------------|--------------------------|
| activations | memory and recomputation | usually not retained |
| gradients | all-reduce/reduce-scatter | not present |
| optimizer state | often larger than weights | not present |
| KV cache | not central in normal training | primary serving memory pressure |
| batch size | throughput and convergence | throughput and latency |
| sequence length | activation and attention cost | KV cache and prefill cost |

### Measure It

- tokens/sec for prefill and decode
- samples/sec or tokens/sec during training
- activation memory by layer
- KV-cache bytes per token
- effect of batch size and sequence length
- numerical drift across precision modes

### Ship It

Ship a transformer execution report with diagrams or tables for shape flow, memory flow, and throughput. Include at least one surprising bottleneck you found from measurement.

---

## Stage 3: GPU Kernels And CUDA Performance

### Why It Matters

This is where MLSys becomes hardware-shaped. You need enough GPU knowledge to know whether a bottleneck is bandwidth, launch overhead, occupancy, synchronization, or tensor-core utilization.

Memory hierarchy:

```text
HBM -> L2 cache -> SM shared memory -> registers
```

### Learn

- CUDA grids, blocks, warps, streams, events, and synchronization
- warp execution, divergence, occupancy, memory coalescing, and bank conflicts
- shared memory, registers, L2 behavior, and HBM bandwidth
- tensor cores and matrix tiling
- reductions, scans, softmax, normalization, and matmul kernels
- kernel launch overhead, CUDA graphs, persistent kernels, and kernel fusion
- profiler workflow with Nsight Systems, Nsight Compute, and simple CUDA events

### Build It

Build a small kernel ladder:

1. vector add baseline
2. reduction kernel
3. layer norm or RMSNorm
4. tiled matrix multiply
5. fused RMSNorm + residual or fused bias + activation
6. Triton version of one kernel

Then add a transformer-shaped benchmark:

- compare naive attention, tiled attention, and library attention where possible
- compare single kernel versus fused path
- compare launch-by-launch execution versus CUDA graph capture when applicable

### Use It In The Real Stack

Read source with one question in mind: what memory traffic did this code remove?

Study:

- CUTLASS for tiled GEMM and template-based kernel structure
- FlashAttention for attention memory traffic reduction
- TensorRT-LLM kernels for production LLM inference paths
- vLLM or SGLang for scheduler/runtime interaction with kernels
- llama.cpp CUDA paths for smaller, readable inference kernels

### Measure It

- achieved bandwidth
- achieved FLOP/s
- occupancy
- global memory transactions
- shared-memory bank conflicts
- tensor-core utilization
- kernel launch count
- numerical error versus reference implementation

### Ship It

Ship a kernel benchmark suite with before/after numbers, profiler screenshots or exported reports, and a short roofline-style explanation for each kernel.

---

## Stage 4: Inference Serving Systems

### Why It Matters

Serving is where kernels become product infrastructure. The runtime must control queueing, memory, streaming, fairness, overload, placement, and observability.

Serving latency decomposes roughly into:

```text
T_latency = T_queue + T_prefill + T_decode + T_stream + T_network
```

Throughput is often constrained by:

```text
min(compute capacity, memory bandwidth, KV-cache capacity, scheduler efficiency)
```

### Learn

- prefill versus decode scheduling
- batching and continuous batching
- request admission and overload control
- streaming responses and cancellation
- paged KV cache and block allocation
- prefix caching and prompt sharing
- speculative decoding
- tensor-parallel and pipeline-parallel inference
- autoscaling, placement, health checks, and drain logic
- observability: TTFT, inter-token latency, queue depth, active sequences, KV blocks, tokens/sec/GPU

### Build It

Build a serving prototype in layers:

1. request queue with deadlines and priorities
2. continuous batching scheduler
3. KV-cache block allocator
4. streaming token output
5. cancellation and eviction path
6. admission control based on memory budget
7. metrics endpoint

Optional advanced additions:

- speculative decoding path with draft and target model
- distributed router that selects replicas by queue depth and health
- tensor-parallel toy runtime with explicit communication calls

### Use It In The Real Stack

Study:

- vLLM for PagedAttention, continuous batching, and serving abstractions
- SGLang for structured generation runtime ideas
- TensorRT-LLM for optimized NVIDIA inference paths
- Triton Inference Server for production model serving patterns
- Ray Serve for distributed service orchestration
- llama.cpp for local and edge inference constraints

### Measure It

- p50/p95/p99 time-to-first-token
- p50/p95/p99 inter-token latency
- requests/sec and tokens/sec/GPU
- prefill throughput versus decode throughput
- KV-cache utilization and fragmentation
- active sequence count
- scheduler overhead
- tail latency under overload

### Ship It

Ship a serving report that can answer:

- What is the bottleneck at low concurrency?
- What is the bottleneck at high concurrency?
- How much memory does the KV cache consume per active request?
- When does batching improve throughput but hurt latency?
- What does the system do when overloaded?

---

## Stage 5: Distributed Training Systems

### Why It Matters

Training systems teach the part of MLSys that inference alone does not: gradient synchronization, activation memory, optimizer-state partitioning, checkpointing, data loading, and failure recovery.

Basic update:

```text
theta_next = theta - learning_rate * gradient(loss, theta)
```

Step time:

```text
T_step = T_forward + T_backward + T_optimizer + T_communication + T_sync
```

Training memory:

```text
M_total = M_weights + M_activations + M_gradients + M_optimizer + M_workspace
```

### Learn

- autograd, activation memory, and recomputation
- mixed precision, gradient scaling, and gradient accumulation
- optimizer state memory and sharding
- data parallelism, tensor parallelism, pipeline parallelism, sequence parallelism, and expert parallelism
- DDP, FSDP, ZeRO, and DTensor/DeviceMesh concepts
- all-reduce, reduce-scatter, all-gather, broadcast, and point-to-point send/recv
- NCCL topology, NVLink/NVSwitch, PCIe, InfiniBand, RoCE, and RDMA concepts
- checkpointing, elastic recovery, rank failure, and restart behavior

### Build It

Build the training progression in this order:

1. single-GPU transformer training loop
2. memory profile with activations and optimizer states
3. DDP run on 2 or more GPUs
4. FSDP or ZeRO run on the same model
5. activation checkpointing experiment
6. distributed checkpoint save and restore
7. NCCL debug/profile run

If you only have one GPU locally, use rented GPUs for the distributed step. The artifact matters more than owning the cluster.

### Use It In The Real Stack

Study:

- PyTorch Distributed for process groups, DDP, FSDP, DTensor, and DeviceMesh
- DeepSpeed for ZeRO and optimizer/memory partitioning
- Megatron-LM for tensor, pipeline, and sequence parallelism patterns
- Ray Train for job orchestration and distributed training ergonomics
- Slurm or Kubernetes for scheduling real GPU jobs

### Measure It

- samples/sec or tokens/sec per GPU
- step time breakdown
- scaling efficiency
- all-reduce/reduce-scatter time
- GPU idle time
- data-loader stall time
- activation memory
- optimizer-state memory
- checkpoint write and restore time

### Ship It

Ship a training-systems report comparing single GPU, DDP, and FSDP/ZeRO. Include profiler traces and a clear explanation of which cost dominated each run.

---

## Stage 6: AI Infrastructure And Orchestration

### Why It Matters

MLSys does not stop at kernels and frameworks. Real systems need scheduling, deployment, isolation, storage, monitoring, rollout, and recovery.

### Learn

- GPU scheduling with Slurm, Kubernetes, Ray, or a smaller custom scheduler
- placement constraints: GPU type, memory, topology, MIG, NUMA, and network reachability
- container images, driver compatibility, CUDA runtime compatibility, and reproducible environments
- data pipeline throughput and storage locality
- checkpoint storage, artifact versioning, and restore paths
- autoscaling and admission control
- multi-tenant isolation and quota policies
- metrics, tracing, logging, alerts, and SLOs
- incident debugging: hangs, OOM, bad nodes, slow links, clock throttling, and version skew

### Build It

Build a small but realistic runbook:

1. define a container image for training and serving
2. run a benchmark job locally or on one node
3. run the same benchmark through a scheduler
4. collect logs, metrics, and profiler traces
5. simulate one failure: OOM, killed process, lost worker, failed checkpoint, or bad config
6. document the recovery path

### Use It In The Real Stack

Tie the infrastructure to the workload:

- training jobs need checkpoint/restart and efficient data loading
- inference services need health checks, draining, and overload behavior
- multi-GPU jobs need topology-aware placement
- edge systems need thermal, power, and storage constraints in the deployment plan

### Measure It

- job startup time
- image size and cold-start time
- GPU allocation efficiency
- failed-job recovery time
- checkpoint restore time
- data-loader throughput
- service availability during rollout

### Ship It

Ship an operations-grade runbook with exact commands, configs, expected metrics, and a failure-mode table.

---

## Stage 7: Compiler And Runtime Layer

### Why It Matters

Compiler/runtime work connects model graphs to hardware execution. It is where high-level operations become fused kernels, memory plans, and backend-specific code.

Compiler path:

```text
framework graph -> IR -> graph rewrite -> fused operators -> lowered kernels -> runtime execution
```

### Learn

- PyTorch graph capture, export, and compile paths
- graph optimization: constant folding, dead-code elimination, layout changes, and operator fusion
- memory planning and buffer reuse
- MLIR dialects and passes
- TVM schedules and auto-tuning concepts
- XLA graph compilation
- TensorRT graph optimization
- Triton language for custom kernels
- correctness testing across rewritten graphs

### Build It

Pick one:

- fuse two simple tensor ops and measure launch-count reduction
- write a Triton kernel for RMSNorm, softmax, or a small matmul
- lower a toy tensor op through MLIR
- compare TensorRT output against eager PyTorch for a small model fragment
- build a static memory planner for a fixed graph

### Use It In The Real Stack

Connect this stage back to hardware:

- fusion reduces memory traffic and launch overhead
- layout changes can make kernels faster or slower
- dynamic shapes increase runtime complexity
- quantization changes both graph structure and kernel selection
- compiler wins are only real if numerical quality and deployment constraints survive

### Measure It

- operator count before/after
- kernel launch count
- memory traffic
- latency and throughput
- compile time
- peak memory
- numerical error versus reference

### Ship It

Ship a compiler/runtime artifact with a before/after benchmark and correctness tests.

---

## Stage 8: Research And Source-Code Loop

### Why It Matters

At senior MLSys level, papers and production code become inputs to engineering decisions. The goal is not to collect papers. The goal is to turn papers into measurements and design choices.

Research loop:

```text
read -> implement or reproduce -> benchmark -> profile -> compare -> write findings
```

### Read

Prioritize systems venues and systems-heavy ML work:

- MLSys
- OSDI
- NSDI
- ASPLOS
- SOSP
- NeurIPS systems, efficiency, and infrastructure papers

### Study Source Code

Use this reading pattern:

1. Identify the hot path.
2. Find the scheduler or runtime boundary.
3. Find memory allocation and cache policy.
4. Find communication calls.
5. Find the kernel launch path.
6. Reproduce a small benchmark.
7. Change one setting and measure the effect.

Good source-code targets:

| Area | Systems |
|------|---------|
| Inference | vLLM, SGLang, llama.cpp, TensorRT-LLM, Triton Inference Server |
| Distributed training | PyTorch Distributed, DeepSpeed, Megatron-LM, Horovod |
| Infrastructure | Ray, Ray Serve, Ray Train, Kubernetes, Slurm, Kubeflow |
| Kernels | FlashAttention, CUTLASS, xFormers, FlashInfer |
| Compiler/runtime | Triton language, TVM, MLIR, XLA, TensorRT |
| Edge | Jetson Linux, TensorRT, Holoscan, ONNX Runtime, llama.cpp |

### Ship It

Every paper or source-code study should produce one artifact:

- reproduction
- benchmark
- implementation note
- diagram
- profiler trace
- bug report
- small patch
- clear negative result

---

## The 6-Month Training Systems Milestone

If your current strength is inference/runtime work, this is the best next milestone. Keep it systems-heavy.

| Month | Focus | Artifact |
|-------|-------|----------|
| 1 | transformer training internals, autograd, activation memory | single-GPU training loop with memory profile |
| 2 | mixed precision, gradient accumulation, optimizer states | throughput and memory report across precision modes |
| 3 | DDP and NCCL basics | 2-8 GPU DDP benchmark with step-time breakdown |
| 4 | FSDP/ZeRO and checkpointing | memory scaling comparison and restore test |
| 5 | DeepSpeed or Megatron-LM internals | annotated runbook for one realistic model config |
| 6 | custom optimization | fused kernel, scheduler improvement, checkpoint improvement, or communication tuning |

Minimum acceptable output:

- one repo
- one model config
- three training modes
- profiler traces
- memory tables
- scaling chart
- written bottleneck analysis

Do not stop at "it runs." The milestone is complete when you can explain why it scales or fails to scale.

---

## Featured Special Course

- [**AI Inference Engineer 2026**](AI%20Inference%20Engineer%202026/README.md) — three-part deep dive on the modern production inference stack:
  - **Part 1 — Fundamentals** (5 lectures): the mental model, transformer execution, roofline, the FP16→FP8→FP4→INT4 precision stack, the runtime landscape
  - **Part 2 — Dense at Hopper** (7 lectures): Llama 3.3 70B ↔ Qwen 2.5 72B side-by-side on H100/H200, quantization recipes including the LLaMA-3-70B W8A8 anomaly, tensor parallelism, the modern serving stack, 128K context, and the distributed communication layer
  - **Part 3 — MoE at Blackwell** (5 lectures): DeepSeek V3.1 + Qwen3-MoE 235B-A22B on B200/GB200 NVL72, MLA + MTP, EP all-to-all, disaggregated prefill/decode, full `$/MTok` cost model

Anchored on the most recent open-weights models and current runtime versions, with a [refresh log](AI%20Inference%20Engineer%202026/REFRESH-LOG.md) that tracks every version pin.

- [**MLSys Deep Dives**](MLSys%20Deep%20Dives/README.md) — a 7-lecture senior-engineer tour of the **2026 machine-learning-systems landscape** as one connected, cost-driven system: the economics of inference (tokens/s, TOK/$, perf/watt); the kernel-language explosion (Triton, CUTLASS/cuTile, ThunderKittens, TileLang); compilers and runtimes (TVM, Mojo/MAX, TensorRT-LLM, the TileRT megakernel runtime); post-transformer architectures (Mamba/SSMs and the hybrid wave — Nemotron-H, Jamba, Falcon-H1, MiniMax); the 2026 frontier models read as systems artifacts (Qwen3, Llama Nemotron Ultra 253B, Xiaomi MiMo, DeepSeek V3/R1, MoE + MLA + MTP); inference acceleration (speculative decoding — EAGLE-3, DFlash — and Flash kernels, with Together AI's research line); and the edge / physical-AI frontier (1000-TOPS hardware vs the 1000-tok/s milestone, plus a measured optimization-ladder capstone).

- [**TVM Deep Dives**](TVM%20Deep%20Dives/README.md) — a 5-lecture senior-engineer course on the Apache TVM compiler stack: the compilation flow (Relax, TensorIR, the unified `IRModule`), hand-scheduling kernels, auto-tuning (AutoTVM → Ansor → MetaSchedule), Relax graph optimization (dynamic shapes, fusion, BYOC for custom accelerators), and shipping (runtime, microTVM, MLC-LLM). Compile a model to the metal, tune it, and beat the framework baseline — with the numbers to prove it.

---

## Edge MLSys Specialization

For this roadmap, the strongest niche is **Edge MLSys + Inference Runtime Engineering**.

This combines:

- Jetson and embedded Linux
- local AI and robotics inference
- low-power deployment
- memory-efficient serving
- multimodal runtime work
- scheduler design
- CUDA/TensorRT optimization
- Rust/C++ runtime engineering
- MLIR/Triton compiler paths
- observability on constrained devices

Good edge MLSys projects:

- Jetson LLM serving runtime with continuous batching and KV-cache accounting
- low-memory LoRA or adapter-training experiment
- edge model adaptation pipeline with checkpoint recovery
- multimodal scheduler for camera, audio, and text workloads
- local/private AI appliance runtime with observability and overload control
- CUDA kernel optimization report on Orin versus desktop GPU
- thermal-aware inference scheduler that changes batch/concurrency under power limits

The edge niche is valuable because it joins skills that are usually split across different engineers: embedded Linux, GPU optimization, AI inference, runtime engineering, and production deployment.

---

## Capstone Options

Choose one. A good capstone is narrow enough to finish and deep enough to prove systems ability.

### Option A: Edge Inference Runtime

Build a Jetson-focused runtime with:

- tokenizer path
- request scheduler
- continuous batching
- KV-cache accounting
- streaming output
- metrics endpoint
- Nsight profile
- power and thermal notes

Success criteria:

- reproducible benchmark
- p50/p95 TTFT and inter-token latency
- tokens/sec under at least three concurrency levels
- memory report for weights, KV cache, and workspace
- overload behavior documented

### Option B: Distributed Training Systems Report

Build a training benchmark suite with:

- single-GPU baseline
- DDP run
- FSDP or ZeRO run
- activation checkpointing experiment
- checkpoint restore test
- profiler traces

Success criteria:

- tokens/sec or samples/sec per GPU
- step-time breakdown
- scaling efficiency
- memory comparison
- communication-cost analysis
- one concrete tuning recommendation

### Option C: Compiler/Kernel Runtime Demo

Build a model-fragment optimization with:

- reference PyTorch implementation
- custom CUDA or Triton kernel
- graph fusion or lowering path
- correctness tests
- benchmark harness

Success criteria:

- before/after latency
- kernel launch count
- memory traffic estimate
- numerical error table
- explanation of when the optimization stops helping

---

## Portfolio Standard

A strong MLSys portfolio artifact includes:

- architecture diagram
- exact hardware and software versions
- reproducible setup commands
- benchmark harness
- profiler traces
- raw result files
- summary charts
- bottleneck analysis
- failure modes
- next optimization hypothesis

Weak artifact:

```text
I used vLLM and it was faster.
```

Strong artifact:

```text
Continuous batching improved throughput from 410 to 690 tok/s at 64 concurrent
requests, but p95 TTFT increased from 380 ms to 610 ms. The profiler shows
prefill bursts starving decode, so the next experiment limits prefill tokens per
scheduling iteration.
```

---

## Career Positioning

This track supports titles like:

- ML Systems Engineer
- AI Infrastructure Engineer
- Inference Systems Engineer
- Training Systems Engineer
- GPU Runtime Engineer
- Edge AI Runtime Engineer
- LLM Runtime Optimization Engineer

Strong positioning:

```text
ML Systems Engineer | GPU Runtime Optimization | CUDA | TensorRT-LLM |
Distributed Inference | Edge AI Infrastructure | Jetson | MLIR | C++ | Rust
```

or:

```text
Inference Systems Engineer | LLM Runtime Optimization | CUDA Kernels |
Tensor Parallelism | Edge AI | Jetson | TensorRT-LLM | ML Systems
```

The title matters less than public proof. Publish benchmark graphs, latency profiles, memory reports, architecture diagrams, profiler traces, and small runtime components.

---

## Official References

Use official docs and primary sources first:

- [PyTorch Distributed Overview](https://docs.pytorch.org/tutorials/beginner/dist_overview.html)
- [vLLM Documentation](https://docs.vllm.ai/)
- [NVIDIA TensorRT-LLM Documentation](https://docs.nvidia.com/tensorrt-llm/)
- [NVIDIA CUDA C++ Best Practices Guide](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/)
- [NVIDIA NCCL Documentation](https://docs.nvidia.com/deeplearning/nccl/)
- [DeepSpeed ZeRO Documentation](https://deepspeed.readthedocs.io/en/stable/zero3.html)
- [Ray Train Documentation](https://docs.ray.io/en/latest/train/overview.html)
- [NVIDIA Megatron-LM](https://github.com/NVIDIA/Megatron-LM)
- [Triton Language Documentation](https://triton-lang.org/main/index.html)
- [MLIR Documentation](https://mlir.llvm.org/docs/)
- [Apache TVM Documentation](https://tvm.apache.org/docs/)
- [Triton Inference Server Documentation](https://docs.nvidia.com/deeplearning/triton-inference-server/)

---

## Exit Criteria

You are ready to claim MLSys competency when you can:

- explain transformer training and inference as shape, memory, kernel, and communication flows
- profile a workload before proposing an optimization
- write and benchmark at least one custom GPU kernel
- debug a distributed training run limited by memory, communication, or synchronization
- explain how continuous batching and paged KV cache affect serving throughput and tail latency
- connect runtime decisions to hardware constraints
- operate a small training or inference system with logs, metrics, and recovery steps
- ship a reproducible benchmark artifact

The outcome is not a certificate. It is a body of systems work that proves you can make AI workloads run faster, cheaper, and more reliably.
