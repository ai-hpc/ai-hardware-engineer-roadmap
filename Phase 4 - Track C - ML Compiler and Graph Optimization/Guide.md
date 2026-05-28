# Phase 4 — Track C: ML Compiler & Graph Optimization (6–12 months)

<div class="course-identity ml-compiler" markdown="1">
<div class="course-identity__icon">IR</div>
<div markdown="1">
<p class="course-identity__eyebrow">Track C · ML Compiler & Graph Optimization</p>
<p class="course-identity__title">Lower model graphs into optimized kernels, memory plans, quantized paths, and hardware-specific runtimes.</p>
<p class="course-identity__meta">Artifact: compiler/runtime demo · Measure: op count, memory, latency, correctness</p>
</div>
</div>


> *The bridge between AI models and hardware — learn how compilers lower neural-network graphs to efficient, hardware-specific code, then apply that knowledge to build and optimize real inference pipelines.*

**Prerequisites:** Phase 1 §4 (C++ and Parallel Computing), Phase 3 (Neural Networks). Recommended: Phase 1 §3 (Operating Systems — memory, processes).

**Layer mapping:** Primarily **Layer 2** (Compiler & Graph Optimization) of the AI chip stack, with connections into Layer 1 (framework graphs) and Layer 3 (runtime that executes compiled artifacts).

**Role targets:** AI Compiler Engineer · DL Graph Optimization Engineer · ML Compiler Backend Engineer · DL Inference Optimization Engineer · MTS Kernels

---

## Why a dedicated compiler track

Tracks A (FPGA) and B (Jetson) teach you to deploy on specific hardware. This track teaches you *how models become hardware instructions* — the compiler stack that sits between a PyTorch/ONNX graph and the kernel code running on any accelerator. Whether you target GPU, FPGA, or a custom NPU, you need to understand IR design, graph optimization, scheduling, and code generation. This is also the fastest-growing hiring area in AI infrastructure.

---

## Track structure

This track has two parts. **Part 1** covers compiler fundamentals (IR, graph optimization, LLVM, MLIR, compilation pipelines, fusion, custom backends). **Part 2** applies these concepts to real DL inference workloads (profiling, kernel engineering, quantization, runtimes, tinygrad deep dive). Together they take you from theory to production.

| Part | Focus | Sections |
|------|-------|----------|
| **Part 1 — Compiler Fundamentals** | How compilers work, from IR to hardware code | §1–§7 below |
| **Part 2 — DL Inference Optimization** | Applying compiler + kernel skills to real inference | [DL Inference Optimization →](DL%20Inference%20Optimization/Guide.md) (6 units) |

**Recommended order:** Work through Part 1 first (or at least §1–§2 and §5), then Part 2 in order. Part 2 units 01–03 reinforce and deepen Part 1 concepts with GPU-specific practice; units 04–06 add quantization, deployment, and hands-on tinygrad.

---

## Part 1 — Compiler Fundamentals

### 1. Graph Representation & Intermediate Representation (IR)

* **Computational graphs:**
    * How PyTorch, TensorFlow, and ONNX represent models as directed acyclic graphs (DAGs).
    * Tracing vs scripting vs export: `torch.export`, `torch.compile`, ONNX export.
    * Graph-level metadata: shapes, dtypes, memory layout (NCHW vs NHWC).

* **Intermediate representations:**
    * **Graph IR vs linearized IR** — Graph: nodes = ops, edges = tensors. Linearized: list of ops in execution order (tinygrad's approach).
    * **SSA (Single Static Assignment) form** — Each value defined once; enables clean alias and memory analysis.
    * **ONNX as interchange IR** — Opsets, shape inference, version converters.

* **tinygrad IR study:**
    * Trace a model from `Tensor` ops to `LazyBuffer` to linearized ops to generated code.
    * Understand how tinygrad's IR differs from graph-based IRs (TorchFX, ONNX).

**Projects:**
* Export a CNN (ResNet-18) to ONNX. Visualize the graph with Netron. Identify redundant ops.
* Trace a matmul through tinygrad: `Tensor` → `LazyBuffer` → scheduled ops → generated CUDA kernel. Document every IR boundary.

---

### 2. Graph Optimization Passes

* **Algebraic simplifications:**
    * Constant folding, dead code elimination, common sub-expression elimination (CSE).
    * Strength reduction (e.g., replace `x / 2` with `x * 0.5`, replace `pow(x, 2)` with `x * x`).

* **Operator fusion:**
    * Why fusion matters: reduces memory traffic and kernel launch overhead.
    * **Vertical fusion:** Conv → BatchNorm → ReLU merged into a single kernel.
    * **Horizontal fusion:** Independent ops run in one kernel to saturate compute.
    * Fusion in practice: TensorRT's layer fusion, tinygrad's BEAM-based fusion, XLA's fusion heuristics.

* **Layout and memory optimizations:**
    * Data layout transformations (NCHW ↔ NHWC) to match hardware preference.
    * In-place operation detection via alias analysis.
    * Memory planning: operator-level liveness analysis → buffer reuse → reduced peak memory.

* **Quantization as a graph pass:**
    * Inserting quantize/dequantize nodes (PTQ).
    * Folding BN into Conv weights before quantization.
    * Calibration: collecting activation ranges to set scale/zero-point.

**Projects:**
* Implement a Conv+BN+ReLU fusion pass on an ONNX graph using `onnx` + `onnxruntime` Python APIs. Measure kernel count reduction.
* Write a memory planning pass: given an op schedule, compute minimum buffer allocation using liveness intervals.

---

### 3. LLVM Fundamentals for AI Compilers

* **Three-phase compiler design:**
    * Frontend → Optimizer → Backend. Why this modularity matters for AI targets.
    * LLVM IR: types, instructions, SSA, basic blocks, functions.

* **LLVM IR in depth:**
    * Address spaces (important for GPU/accelerator memory models).
    * Intrinsics: how hardware-specific operations are exposed in IR.
    * Metadata and debug info.

* **Optimization passes:**
    * Canonicalization, loop unrolling, vectorization, dead store elimination.
    * Writing a custom LLVM pass (C++): register a pass, walk the IR, transform.

* **Backend and code generation:**
    * Instruction selection via SelectionDAG / GlobalISel.
    * TableGen: declaratively describing target instructions.
    * Register allocation, instruction scheduling.
    * How GPU targets (NVPTX, AMDGPU) use LLVM.

**Resources:**
* [LLVM Language Reference](https://llvm.org/docs/LangRef.html)
* [LLVM Programmer's Manual](https://llvm.org/docs/ProgrammersManual.html)
* [Writing an LLVM Pass](https://llvm.org/docs/WritingAnLLVMPass.html)

**Projects:**
* Write and compile a small function to LLVM IR (`clang -emit-llvm`). Read the IR, identify SSA values, trace through `opt` passes.
* Write a custom LLVM pass that counts floating-point multiply-accumulate (FMA) operations in a function — a proxy for FLOPS estimation.

---

### 4. MLIR: Multi-Level Intermediate Representation

* **Why MLIR:**
    * LLVM has one IR level; AI compilers need many. MLIR provides a framework for building *dialects* at every abstraction level.
    * Progressive lowering: high-level tensor ops → loop nests → vector instructions → hardware code.

* **Core MLIR concepts:**
    * **Dialects:** `tensor`, `linalg`, `memref`, `affine`, `vector`, `scf`, `gpu`, `llvm`.
    * **Operations, regions, blocks** — the MLIR data model.
    * **Types and attributes** — extensible type system.

* **Key dialects for AI:**
    * **`linalg`** — Named ops (conv, matmul, pooling) and generic ops on tensors. Tiling, fusion, promotion.
    * **`affine`** — Polyhedral loop analysis. Dependence analysis, loop interchange, tiling.
    * **`tensor` / `memref`** — Bufferization: converting value-semantics tensors to in-memory buffers.
    * **`vector`** — Target-independent SIMD representation.
    * **`gpu`** — GPU kernel launch, thread/block mapping.

* **Progressive lowering walkthrough:**
    * `tosa` (from ONNX/TF) → `linalg` → `affine`/`scf` → `vector` → `gpu` or `llvm`.
    * Bufferization pass: when and how tensors become memrefs.

* **Custom dialect development:**
    * Defining a custom NPU dialect with ODS (Operation Definition Specification).
    * Writing lowering passes from `linalg` to your custom dialect.

**Resources:**
* [MLIR Documentation](https://mlir.llvm.org/)
* [MLIR Tutorial: Creating a Dialect](https://mlir.llvm.org/docs/Tutorials/CreatingADialect/)
* [Toy Tutorial](https://mlir.llvm.org/docs/Tutorials/Toy/) — end-to-end custom language in MLIR.

**Projects:**
* Complete the MLIR Toy tutorial (chapters 1–7). Understand how to define ops, write lowering passes, and generate LLVM IR.
* Write a minimal NPU dialect with a single `npu.matmul` op. Lower `linalg.matmul` to `npu.matmul` with a tiling strategy.

---

### 5. ML-to-Hardware Compilation Pipelines

* **TVM:**
    * **Relay** (graph-level IR) → **TIR** (Tensor IR, loop-level) → target code.
    * Schedule primitives: `tile`, `vectorize`, `parallel`, `unroll`, `reorder`.
    * **AutoTVM / AutoScheduler (Ansor):** Search-based tuning for operator schedules.
    * **BYOC (Bring Your Own Codegen):** Offloading subgraphs to custom accelerators.

* **tinygrad compiler:**
    * Scheduler: how ops are grouped into kernels.
    * **BEAM search:** exploring fusion choices to minimize runtime.
    * Backends: CUDA, OpenCL, Metal, LLVM, custom.
    * Adding a new backend to tinygrad.

* **Production compilers:**
    * **torch.compile + Inductor:** TorchFX → Triton kernels. How `torch._inductor` schedules and generates code.
    * **XLA (Accelerated Linear Algebra):** Used by JAX and TF. HLO IR → target code (TPU, GPU, CPU).
    * **IREE:** MLIR-based compiler for heterogeneous deployment (CPU, GPU, DSP).
    * **Triton-MLIR:** Python-level kernel writing compiled via MLIR pipeline.

* **End-to-end flow comparison:**

| Pipeline | Input | IR(s) | Scheduling | Target |
|----------|-------|-------|------------|--------|
| TVM | ONNX/Relay | Relay → TIR | AutoTVM / Ansor | CUDA, LLVM, custom |
| tinygrad | tinygrad graph | Linearized ops | BEAM search | CUDA, OpenCL, Metal, LLVM |
| torch.compile | TorchFX | FX → Inductor IR | Triton codegen | CUDA (Triton), CPU |
| XLA | HLO | HLO → LLO | XLA scheduler | TPU, GPU, CPU |
| IREE | MLIR (tosa/linalg) | linalg → vector → spirv/llvm | Tile + distribute | CPU, GPU, DSP |

**Projects:**
* Compile a ResNet-18 with TVM for your GPU. Use AutoTVM to tune 3 key ops (conv2d, dense, batch_matmul). Compare latency before/after tuning.
* Study tinygrad BEAM: run with `BEAM=3` on a small model. Compare kernel count and latency vs `BEAM=0`.
* Use `torch.compile` on a transformer block. Read the generated Triton kernel for the attention op. Annotate the tiling strategy.

---

### 6. Kernel Fusion & Tiling Strategies

* **Tiling for memory hierarchy:**
    * Why tile: make working sets fit in L1/L2/shared memory/SRAM scratchpad.
    * Tile size selection: auto-tuning vs analytical models (roofline-guided).
    * Multi-level tiling: thread-block tiles → warp tiles → register tiles (GPU); array tiles → PE tiles (accelerator).

* **Fusion strategies:**
    * **Producer-consumer fusion:** Fuse ops where one's output is the other's input (Conv → ReLU).
    * **Parallel fusion:** Fuse independent ops sharing inputs to reduce reads.
    * **Reduction fusion:** Fuse element-wise ops with reductions (softmax components).
    * Fusion legality: when aliasing or data dependencies prevent fusion.

* **Dataflow scheduling:**
    * Weight-stationary, output-stationary, row-stationary, no-local-reuse.
    * How the compiler maps these strategies to hardware tiling parameters.
    * Connection to Layer 5 (hardware architecture): the compiler must know the hardware's dataflow.

**Projects:**
* Implement a 2-level tiled matmul in C++ (outer tiles for L2, inner tiles for L1). Benchmark vs naive and compare with vendor BLAS.
* In tinygrad or TVM, modify a fusion heuristic. Measure the impact on a real model (kernel count, total latency, memory traffic).

---

### 7. Custom Backend Development

* **What is a custom backend:**
    * The code generator that takes compiler IR and produces instructions for your specific hardware (FPGA, NPU, custom ASIC).

* **TVM BYOC path:**
    * Annotate subgraph → partition → custom codegen function → emit code or runtime calls.
    * Build a minimal BYOC backend for a simulated accelerator.

* **tinygrad custom backend:**
    * Implement `Runtime` and `Compiler` classes for a new target.
    * Map linearized ops to hardware instructions.

* **MLIR custom lowering:**
    * Define target dialect → write lowering pass from `linalg` → emit target-specific code.
    * Integration with LLVM backend or standalone code emitter.

* **Connection to other tracks:**
    * Track A (FPGA): your compiler backend generates HLS directives or RTL control sequences.
    * Track B (Jetson): your compiler backend targets CUDA, DLA, or TensorRT.

**Projects:**
* Build a minimal TVM BYOC backend that offloads `nn.dense` to a Python-simulated accelerator. Verify correctness.
* Add a simple tinygrad backend for a simulated 4×4 systolic array (compute matmul tiles, verify output).
* (Advanced) Write an MLIR lowering pass from `linalg.matmul` to a custom dialect that emits tiled DMA + compute commands.

---

## Part 2 — DL Inference Optimization

> *Apply compiler and kernel skills to real inference workloads: profile, write kernels, quantize, and deploy.*

This part was previously a Phase 5 specialization track. It now lives here because the skills (profiling, kernel authoring, quantization, runtime integration) are the direct application of Part 1's compiler fundamentals — and they're needed before the advanced Phase 5 specializations (HPC infrastructure, AI chip design).

**Role target:** DL Inference Optimization Engineer · **MTS Kernels** (Member of Technical Staff, Kernels)

**Prerequisites for Part 2:** Part 1 (at least §1–§2 and §5), Phase 4 Track B (Jetson, TensorRT, CUDA).

Study the units **in order**. Each folder is one unit; do them one by one.

| Order | Unit | What you learn | Guide |
|:-----:|------|----------------|-------|
| **1** | Graph & Operator Optimization | What we're optimizing: graph, ops, fusion, profiling, bottlenecks. Reinforces Part 1 §2 with GPU-specific profiling. | [01 →](DL%20Inference%20Optimization/01%20-%20Graph%20and%20Operator%20Optimization/Guide.md) |
| **2** | Kernel Engineering | How to write and own kernels: Triton, CUTLASS/CuTe, Flash-Attention, long-context, NCCL. Core of MTS Kernels. | [02 →](DL%20Inference%20Optimization/02%20-%20Kernel%20Engineering/Guide.md) |
| **3** | Compiler Stack | How compilers produce kernels: IR, scheduling (BEAM), codegen, TVM, MLIR. Reinforces Part 1 §3–§5 with hands-on projects. | [03 →](DL%20Inference%20Optimization/03%20-%20Compiler%20Stack/Guide.md) |
| **4** | Quantization | Low-precision inference: PTQ, QAT, INT8/INT4, kernel and runtime integration. | [04 →](DL%20Inference%20Optimization/04%20-%20Quantization/Guide.md) |
| **5** | Inference Runtimes & Deployment | Production: TensorRT, ONNX Runtime, Triton server; latency, throughput, methodology. | [05 →](DL%20Inference%20Optimization/05%20-%20Inference%20Runtimes%20and%20Deployment/Guide.md) |
| **6** | tinygrad Deep Dive | Optional: hands-on IR, scheduler, backends; compiler-kernel interface. | [06 →](DL%20Inference%20Optimization/06%20-%20tinygrad%20Deep%20Dive/Guide.md) |

### Basic concepts (read before Part 2)

Before diving into the units, you need the **vocabulary and mental model** of modern LLM inference. Key concepts:

* **In-flight batching** — Batch requests as they arrive; improves throughput without killing latency.
* **Paged KV-cache** — Attention needs key/value cache per token; paging and reuse make it memory-efficient.
* **Speculative decoding** — Draft multiple tokens with a small model, verify with the big model; fewer forward passes.
* **Throughput vs latency** — Batch more → higher throughput, worse latency. Tune batching and scheduling to the SLA.
* **CuTe DSL** — C++ header library inside CUTLASS that defines layouts and copy operations for tiling, used in production GEMM/attention kernels.
* **New architectures** — Every GPU generation (Hopper, Blackwell) changes execution model, memory hierarchy, and instruction throughput. Old tile sizes become suboptimal.

See the [DL Inference Optimization overview](DL%20Inference%20Optimization/Guide.md) for the full vocabulary (LLM inference, distributed inference, production systems).

---

## Relationship to Other Tracks

| This track (C) provides | Track A (FPGA) uses it for | Track B (Jetson) uses it for |
|--------------------------|---------------------------|------------------------------|
| Graph optimization passes | Mapping ONNX → HLS-friendly subgraphs | TensorRT graph optimization understanding |
| MLIR / TVM compilation | Vitis AI / FINN compilation flow | torch.compile + Inductor on GPU |
| Custom backend development | FPGA backend in TVM or tinygrad | DLA/TensorRT backend integration |
| Tiling & dataflow scheduling | HLS pragma-driven tiling | CUDA kernel tiling strategies |
| IR & SSA fundamentals | Understanding Vivado synthesis IR | Understanding NVPTX code generation |
| Kernel engineering (Part 2) | — | Triton/CUTLASS kernels for GPU inference |
| Quantization (Part 2) | Vitis AI quantizer understanding | TensorRT INT8/INT4 deployment |
| Inference runtimes (Part 2) | Vitis AI / FINN runtime | TensorRT, Triton server, DeepStream |

---

## Build Summary

### Part 1 — Compiler Fundamentals

| Module | Hands-on deliverable |
|--------|---------------------|
| §1 IR | ONNX graph analysis + tinygrad IR trace |
| §2 Graph opts | Conv+BN+ReLU fusion pass, memory planner |
| §3 LLVM | Custom LLVM pass (FMA counter) |
| §4 MLIR | Toy tutorial + minimal NPU dialect |
| §5 Pipelines | TVM AutoTVM tuning, BEAM comparison, torch.compile analysis |
| §6 Fusion/tiling | Tiled matmul, fusion heuristic modification |
| §7 Custom backend | TVM BYOC or tinygrad backend for simulated accelerator |

### Part 2 — DL Inference Optimization

| Unit | Hands-on deliverable |
|------|---------------------|
| 01 Graph & ops | Fusion + measure, profiling report, end-to-end breakdown |
| 02 Kernel engineering | Triton fused kernel, long-context attention, NCCL at scale |
| 03 Compiler stack | BEAM in tinygrad, fusion pass, lowering trace |
| 04 Quantization | INT8 with TensorRT, PTQ vs QAT comparison, kernel path trace |
| 05 Runtimes | Runtime comparison, Triton server setup, benchmark report |
| 06 tinygrad (optional) | Pipeline trace, add optimization, backend hook |
