# 7. ML Systems Engineering (Phase 5)

> Build the infrastructure, runtimes, distributed systems, and serving engines that make AI models train and run at scale.

**Layer mapping:** L3-L8. This track connects model math, kernels, runtime scheduling, distributed communication, cluster orchestration, serving systems, and production observability.

**Role targets:** ML Systems Engineer · AI Infrastructure Engineer · Inference Systems Engineer · Training Systems Engineer · GPU Runtime Engineer · Edge AI Runtime Engineer

**Prerequisites:** [Operating Systems](../../Phase%201%20-%20Foundational%20Knowledge/3.%20Operating%20Systems/Guide.md), [C++ and Parallel Computing](../../Phase%201%20-%20Foundational%20Knowledge/4.%20C%2B%2B%20and%20Parallel%20Computing/Guide.md), [Neural Networks](../../Phase%203%20-%20Artificial%20Intelligence/1.%20Neural%20Networks/Guide.md), [Deep Learning Frameworks](../../Phase%203%20-%20Artificial%20Intelligence/2.%20Deep%20Learning%20Frameworks/Guide.md), one deployment path from Phase 4, and enough [GPU Infrastructure](../1.%20GPU%20Infrastructure/Guide.md) to run real benchmarks.

**What comes after:** a public systems artifact: an inference runtime, distributed training runbook, CUDA kernel benchmark suite, scheduler prototype, MLIR/Triton lowering demo, or edge MLSys case study with reproducible measurements.

---

## Why This Track Exists

MLSys is not "ML engineering with more frameworks." The MLSys engineer builds the machinery underneath the model:

- training systems
- inference engines
- GPU scheduling
- memory optimization
- distributed communication
- kernels
- model serving
- AI infrastructure
- compiler and runtime layers
- observability
- large-scale orchestration

The core tools are systems tools: PyTorch internals, vLLM, SGLang, Ray, DeepSpeed, Megatron-LM, TensorRT-LLM, Kubernetes, NCCL, Triton Inference Server, CUDA, CUTLASS, FlashAttention, MLIR, TVM, XLA, and Triton language.

The daily question is not "Can I train a model?" It is:

> Where are compute, memory, communication, synchronization, and scheduling wasting time or capacity?

That is why this track sits after inference, GPU infrastructure, and compiler/runtime work. It turns those pieces into one operating discipline.

---

## Mental Model

MLSys work is the constant reduction of five bottlenecks:

```text
T_total = T_compute + T_memory + T_communication + T_synchronization + T_scheduling
```

For training:

```text
T_step = T_forward + T_backward + T_allreduce + T_checkpoint + T_sync
```

For inference:

```text
T_latency = T_prefill + T_decode + T_memory + T_communication + T_queueing
```

For memory:

```text
M_training = M_weights + M_activations + M_gradients + M_optimizer
M_inference = M_weights + M_kv_cache + M_workspace + M_batch_state
```

Most meaningful MLSys improvements reduce one of these terms without breaking model quality, reliability, or operational simplicity.

---

## Course Map

| Stage | Focus | Production systems to study | Artifact |
|-------|-------|-----------------------------|----------|
| 1 | Systems programming for runtimes | Linux, networking, async runtimes, profilers | async inference server or scheduler |
| 2 | Deep learning internals | PyTorch, JAX, transformer implementations | small transformer with profiled training and inference |
| 3 | GPU architecture and CUDA | CUDA, CUTLASS, FlashAttention, Nsight | custom kernel and roofline report |
| 4 | Distributed training | PyTorch DDP/FSDP, DeepSpeed, Megatron-LM, NCCL | multi-GPU training benchmark |
| 5 | Serving systems | vLLM, SGLang, TensorRT-LLM, Triton Inference Server | continuous batching or KV-cache prototype |
| 6 | Compiler and runtime layer | MLIR, TVM, XLA, TensorRT, Triton language | operator fusion or lowering demo |
| 7 | Research-level MLSys | MLSys, OSDI, NSDI, ASPLOS, NeurIPS systems papers | paper replication or benchmark reproduction |

---

## Stage 1: Systems Programming For MLSys

This is the foundation. The runtime is still software running on Linux, moving bytes through memory, sockets, drivers, and queues.

### Learn

- Linux internals: processes, threads, signals, cgroups, namespaces, filesystems
- memory: `mmap`, page faults, huge pages, pinned memory, NUMA, zero-copy paths
- concurrency: threads, locks, atomics, work stealing, backpressure
- networking: TCP, gRPC, HTTP streaming, RDMA concepts, `epoll`, `io_uring`
- performance: CPU cache locality, SIMD basics, perf, flamegraphs, tracing
- languages: Python for ML ecosystem work, C++ for kernels/runtime integration, Rust for infrastructure and runtime systems

### Build

- async inference server with request queueing and streaming responses
- tokenizer runtime with zero-copy request parsing
- scheduler that supports priorities, cancellation, batching windows, and backpressure
- mini tensor runtime with explicit allocation and shape tracking

### Measure

- p50/p95/p99 latency
- throughput under different queue depths
- allocation count and peak RSS
- CPU utilization, lock contention, and context switches

### Ship

A small runtime that accepts inference-like requests, batches them, streams partial outputs, and produces a benchmark report under load.

---

## Stage 2: Deep Learning Internals

The goal is not to train MNIST. The goal is to understand the exact computation and memory flow of transformers so runtime decisions are grounded in model structure.

### Learn

- transformer architecture: embeddings, attention, MLP, residuals, layer norm/RMSNorm, logits, sampler
- attention: QKV projection, grouped-query attention, RoPE, KV cache
- decoding: prefill vs decode, sampling, batching, speculative decoding
- optimization: quantization, FlashAttention, paged attention, fused kernels, CUDA graph capture
- parallelism: tensor parallelism, pipeline parallelism, expert parallelism

The basic attention form:

```text
Attention(Q, K, V) = softmax((Q * K^T) / sqrt(d_k)) * V
```

The transformer inference flow:

```text
token -> embedding -> attention(QKV) -> MLP -> residual -> logits -> sampler
```

The transformer training flow:

```text
forward pass -> loss -> backward pass -> gradients -> optimizer step -> updated weights
```

### Build

- small transformer from scratch or from a tiny framework
- custom training loop with mixed precision and gradient accumulation
- KV-cache implementation with explicit memory accounting
- tokenizer-to-logits inference path

### Measure

- tokens/sec for prefill and decode
- activation memory during training
- KV-cache growth with sequence length and batch size
- effect of batch size on throughput and latency

### Ship

A transformer notebook or repo that reports shape flow, memory usage, and throughput for both training and inference.

---

## Stage 3: GPU Architecture And CUDA

This is where MLSys becomes hardware-shaped. You need enough GPU knowledge to know whether a bottleneck is memory bandwidth, launch overhead, occupancy, synchronization, or tensor-core utilization.

### Learn

- CUDA programming model: grids, blocks, warps, streams, events
- warp execution, divergence, occupancy, memory coalescing
- memory hierarchy: HBM, L2, shared memory, registers
- tensor cores and matrix-multiply tiling
- kernel launch overhead, CUDA graphs, persistent kernels
- kernel fusion and memory traffic reduction
- NCCL collectives at the GPU boundary

Memory hierarchy to keep in mind:

```text
HBM -> L2 cache -> SM shared memory -> registers
```

### Study

- CUTLASS
- FlashAttention
- vLLM scheduler and paged attention
- TensorRT-LLM kernels
- llama.cpp CUDA paths

### Build

- custom CUDA vector and matrix kernels
- fused RMSNorm kernel
- benchmark comparing naive attention, tiled attention, and FlashAttention-style memory reduction
- Jetson inference path optimization with Nsight traces

### Measure

- achieved memory bandwidth
- achieved FLOP/s
- occupancy
- global memory transactions
- tensor-core utilization
- kernel launch count

### Ship

A kernel benchmark suite with before/after numbers, profiler screenshots, and a short explanation of whether each kernel is compute-bound or memory-bound.

---

## Stage 4: Distributed Training Systems

This is the right next milestone after inference-runtime work. Do not spend this phase on generic fine-tuning tutorials. Focus on the systems mechanics of training.

### Learn

- autograd and activation memory
- mixed precision and loss scaling
- gradient accumulation
- optimizer state memory
- DDP, FSDP, ZeRO
- tensor, pipeline, sequence, and expert parallelism
- all-reduce, reduce-scatter, all-gather, broadcast
- NCCL topology, NVLink/NVSwitch, InfiniBand, RDMA
- checkpointing, elastic recovery, and failure handling

Weight update:

```text
theta_(t+1) = theta_t - eta * grad_theta L(theta_t)
```

Distributed step time:

```text
T_step = T_forward + T_backward + T_allreduce + T_sync
```

### Build

- single-GPU transformer training loop with memory profiling
- multi-GPU DDP experiment
- FSDP or ZeRO comparison with the same model and batch target
- NCCL profiling run that explains communication cost
- distributed checkpointing experiment

### Measure

- samples/sec or tokens/sec per GPU
- scaling efficiency
- all-reduce time
- GPU idle time
- optimizer-state memory
- checkpoint save/restore time

### Ship

A training-systems report that compares single GPU, DDP, and FSDP/ZeRO runs with profiler traces and clear bottleneck analysis.

---

## Stage 5: Serving Systems And Distributed Inference

Serving is where inference systems become product infrastructure. The runtime must schedule requests, control memory, stream tokens, isolate failures, and expose operational signals.

### Learn

- batching and continuous batching
- request scheduling and fairness
- streaming inference
- async serving and cancellation
- backpressure and overload control
- paged KV cache
- speculative decoding
- tensor-parallel and pipeline-parallel inference
- inference sharding
- autoscaling and placement
- observability: traces, metrics, logs, token accounting

Inference scaling:

```text
T_latency = T_compute + T_memory + T_communication + T_queueing
```

### Study

- vLLM
- SGLang
- TensorRT-LLM
- Triton Inference Server
- llama.cpp
- Ray Serve

### Build

- continuous batching scheduler
- paged KV-cache prototype
- speculative decoding prototype
- tensor-parallel LLM serving experiment
- distributed inference router with health checks and backpressure

### Measure

- p50/p95/p99 time-to-first-token
- inter-token latency
- requests/sec
- tokens/sec/GPU
- KV-cache utilization
- batch occupancy
- tail latency under overload

### Ship

A serving system that can explain its own behavior through metrics: queue depth, active sequences, KV-cache blocks, token latency, and GPU utilization.

---

## Stage 6: Compiler And Runtime Layer

Compiler/runtime work is where MLSys becomes a full stack: framework graph -> IR -> optimized operators -> kernels -> hardware execution.

### Learn

- PyTorch graph capture and export paths
- graph optimization
- operator fusion
- kernel lowering
- MLIR dialects and passes
- TVM schedules
- XLA and TensorRT graph optimization
- Triton language kernels
- runtime memory planning

Compiler path:

```text
PyTorch graph -> IR -> fused operators -> lowered kernels -> GPU execution
```

### Build

- small operator fusion pass
- Triton kernel for a transformer primitive
- MLIR lowering demo for a toy tensor op
- TensorRT graph optimization comparison
- runtime memory planner for a fixed graph

### Measure

- operator count before/after fusion
- kernel launch count
- memory traffic
- latency and throughput
- numerical differences

### Ship

A compiler/runtime artifact that takes a small model fragment and shows the performance effect of lowering or fusion with reproducible commands.

---

## Stage 7: Research-Level MLSys

At this level, papers become engineering inputs. The loop is:

```text
read paper -> implement or reproduce -> benchmark -> profile -> optimize -> write findings
```

### Read

- MLSys
- OSDI
- NSDI
- ASPLOS
- NeurIPS systems and efficiency papers

### Focus Areas

- serving systems
- distributed training
- memory optimization
- scheduling
- compiler/runtime optimization
- efficient attention
- GPU kernels
- low-power and edge inference

### Ship

Every paper should produce one artifact: a reproduction, benchmark, diagram, implementation note, profiler trace, or clear negative result.

---

## The Practical 6-Month Training Milestone

If your current strength is inference/runtime work, the next milestone should be training systems. Make it systems-heavy from day one.

| Month | Focus | Artifact |
|-------|-------|----------|
| 1 | transformer training internals, autograd, activation memory | small transformer training loop with memory profile |
| 2 | mixed precision, gradient accumulation, optimizer states | throughput and memory report across precision modes |
| 3 | DDP and NCCL basics | 2-8 GPU DDP benchmark, even if rented |
| 4 | FSDP/ZeRO and checkpointing | memory scaling comparison and restore test |
| 5 | DeepSpeed or Megatron-LM internals | annotated runbook for one realistic model config |
| 6 | custom optimization | fused kernel, scheduler improvement, or distributed checkpoint improvement |

The point is not to become a generic trainer of models. The point is to understand why training infrastructure stalls, runs out of memory, fails, or scales poorly.

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

Good edge MLSys projects:

- Jetson LLM serving runtime with continuous batching and KV-cache accounting
- low-memory LoRA or adapter training experiment
- edge model adaptation pipeline with checkpoint recovery
- multimodal inference scheduler for camera, audio, and text workloads
- local/private AI appliance runtime with observability and overload control
- CUDA kernel optimization report on Orin vs desktop GPU

This niche is valuable because it combines skills that are usually split across different engineers: embedded systems, GPU optimization, AI inference, runtime engineering, and production deployment.

---

## Open Source Systems To Study

| Area | Systems |
|------|---------|
| Inference | vLLM, SGLang, llama.cpp, TensorRT-LLM, Triton Inference Server |
| Distributed training | PyTorch Distributed, DeepSpeed, Megatron-LM, Horovod |
| Infrastructure | Ray, Ray Serve, Ray Train, Kubernetes, Slurm, Kubeflow |
| Kernels | FlashAttention, CUTLASS, xFormers, FlashInfer |
| Compiler/runtime | Triton language, TVM, MLIR, XLA, TensorRT |
| Edge | Jetson Linux, TensorRT, Holoscan, llama.cpp, ONNX Runtime |

Study source code with a profiler open. Reading without measurement is too easy to fool yourself.

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

Strong positioning looks like:

```text
ML Systems Engineer | GPU Runtime Optimization | CUDA | TensorRT-LLM |
Distributed Inference | Edge AI Infrastructure | Jetson | MLIR | C++ | Rust
```

or:

```text
Inference Systems Engineer | LLM Runtime Optimization | CUDA Kernels |
Tensor Parallelism | Edge AI | Jetson | TensorRT-LLM | ML Systems
```

The public proof matters more than the title. Publish benchmark graphs, latency profiles, memory reports, architecture diagrams, and small but real runtime components.

---

## Capstone

Build one MLSys artifact that covers both inference and training mechanics:

1. A small transformer training stack with DDP/FSDP benchmarks.
2. An inference runtime with continuous batching, KV-cache accounting, and streaming output.
3. One custom CUDA or Triton kernel used in the runtime or training path.
4. A deployment target: Jetson, single workstation GPU, or rented multi-GPU node.
5. A report with latency, throughput, memory, communication, and failure-mode analysis.

The capstone is complete when another engineer can clone it, run the benchmarks, reproduce the charts, and understand which bottleneck you attacked.

---

## Exit Criteria

You are ready to claim MLSys competency when you can:

- explain transformer training and inference as shape, memory, and communication flows
- write and profile at least one custom GPU kernel
- debug a distributed training run that is limited by communication or memory
- explain how continuous batching and paged KV cache affect serving throughput
- use profiler traces instead of guesses
- connect runtime decisions to hardware constraints
- ship a reproducible benchmark artifact

The outcome is not a certificate. It is a body of systems work that proves you can make AI workloads run faster, cheaper, and more reliably.
