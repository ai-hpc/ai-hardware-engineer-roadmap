# C++ and Parallel Computing (Phase 1 §4)

<div class="course-identity parallel-computing" markdown="1">
<div class="course-identity__icon">CUDA</div>
<div markdown="1">
<p class="course-identity__eyebrow">Module 4 · C++ & Parallel Computing</p>
<p class="course-identity__title">Map work onto CPU SIMD, thread pools, CUDA, HIP, OpenCL, and SYCL execution models.</p>
<p class="course-identity__meta">Artifact: parallel benchmark · Measure: speedup, occupancy, bandwidth, synchronization</p>
</div>
</div>


> *From a single instruction to a thousand GPU threads — how computing became parallel, and why every AI hardware engineer must think in parallel.*

**Layer mapping:** **L1** (application — you write the code that runs on hardware), **L3** (runtime — CUDA runtime, OpenCL runtime are the bridge to the GPU driver).

**Prerequisites:** Phase 1 §2 (Computer Architecture — memory hierarchy, pipelining, caches), Phase 1 §3 (Operating Systems — threads, processes, scheduling).

**What comes after:** Phase 3 (Neural Networks — tensors map directly to these execution models), Phase 4 Track B (CUDA on Jetson with power limits), Phase 4 Track A (HLS/RTL — you design the hardware these kernels run on).

---

## Why This Module Exists

AI workloads expose large amounts of parallel work, while power, memory movement, and response time constrain how efficiently it can run. Understanding *which work is independent*, *how hardware executes it*, and *how to measure the complete application* is the foundation for this module. A robot's small control loop and a server's large training batch may need different execution choices.

If you can't think in parallel, you can't design hardware that runs parallel workloads.

---

## Learning Outcomes

By the end of this module, you should be able to:

- explain why modern AI systems are throughput-oriented rather than clock-speed-oriented
- write and debug parallel code across SIMD, CPU multicore, and GPU execution models
- reason about memory locality, synchronization, occupancy, and bandwidth bottlenecks
- measure the difference between a correct implementation and a performant one
- produce at least one portfolio artifact that demonstrates real low-level performance work

---

## How To Work This Module

Do not treat this as five isolated reading sections. Work it as a sequence:

1. Learn the execution model
2. Implement a small kernel or parallel workload
3. Measure speedup and identify the limiting bottleneck
4. Write down what changed and why

For every sub-track, finish with a visible artifact. Good outputs include:

- vectorization benchmark tables
- OpenMP scaling plots
- Nsight screenshots
- occupancy calculations
- HIP porting notes
- SYCL portability comparison notes

---

## Part 1 — How Computing Became Parallel

Before diving into code, understand the historical pressure that created the hardware you'll design.

### Step 1: Sequential → Clock Speed Wall

Clock frequency used to provide a large share of performance growth. Power and heat made that route increasingly costly. A simplified dynamic-power model is `P_dynamic ≈ activity × capacitance × V² × f`; leakage and other system power also matter. Frequency still varies across products and workloads, so a fixed 3–4 GHz ceiling is not a useful rule.

Parallelism and specialization help perform more useful work within power and memory limits. They still need enough independent work to offset scheduling, communication, and synchronization costs.

### Step 2: Instruction-Level and Data Parallelism

Multiple things per clock cycle inside a single core:

```
Pipelining:   overlap fetch/decode/execute of different instructions
Superscalar:  issue multiple independent instructions per cycle
SIMD:         one instruction processes 4/8/16/32 data elements

Without SIMD:       With SIMD (8-wide AVX2):
a[0] = b[0] + c[0]  a[0..7] = b[0..7] + c[0..7]   ← ONE instruction
a[1] = b[1] + c[1]
...                  8 FP32 elements per vector instruction
a[7] = b[7] + c[7]
```

Pipelining and superscalar issue exploit instruction-level parallelism. SIMD expresses **data parallelism**. Eight lanes do not guarantee an eightfold speedup: instruction throughput, dependencies, memory bandwidth, and the scalar baseline all matter.

### Step 3: Multi-Core / Shared Memory

Multiple cores provide another way to execute independent work. Your application, libraries, or runtime must expose that work; adding cores does not automatically accelerate a serial dependency chain. Other processes may also be using the machine.

```
Core 0: Process chunk [0..N/4]
Core 1: Process chunk [N/4..N/2]     ← all running simultaneously
Core 2: Process chunk [N/2..3N/4]
Core 3: Process chunk [3N/4..N]
```

The challenge: shared memory means **race conditions**, **deadlocks**, and **cache coherence**.

### Step 4: Heterogeneous Computing (CPU + GPU)

**Specialized hardware for specific workload types.**

| | CPU | GPU |
|---|---|---|
| Execution units | CPU cores, often with vector units | Multiprocessors containing arithmetic lanes and specialized units |
| Optimized for | Latency (one task fast) | Throughput (many tasks at once) |
| Control strategy | Complex speculation and latency-oriented cores | Many throughput-oriented execution lanes; scheduling and cache design vary |
| Storage | Per-core and shared caches; capacities vary | Registers, shared memory, and caches; capacities vary |
| Best for | Sequential code, branching | Data-parallel: matrix math, convolution |

Many neural networks spend much of their arithmetic in matrix multiplication and convolution. End-to-end inference also includes memory movement, normalization, attention, sampling or action generation, and host scheduling. GPU speedup depends on shapes, precision, utilization, and power limits; measure it against an optimized CPU baseline at the same quality target.

---

## Part 2 — The Five Sub-Tracks

Start with C++/SIMD, then CPU threads. Choose one GPU or portable-compute branch when you have a compatible device. Learning every vendor API is optional; understanding ownership, synchronization, and measurement transfers between branches.

**CPU-only starting point:** finish the [numbers, shapes, and bytes lab](../Labs/Numbers-Shapes-and-Bytes.md), implement one correct vector loop, and compare it with a threaded version. A serial result, explicit input bounds, and tests for short/uneven lengths come before a speedup plot.

| # | Sub-track | What you learn | Guide |
|:-:|-----------|---------------|-------|
| 1 | **C++ and SIMD** | Modern C++17 for parallel code, SIMD intrinsics (SSE/AVX/NEON), auto-vectorization, AoS vs SoA, roofline | [Guide →](C%2B%2B%20and%20SIMD/Guide.md) |
| 2 | **OpenMP and oneTBB** | Multi-core CPU parallelism, fork-join, reductions, tasks, work-stealing, flow graphs | [Guide →](OpenMP%20and%20OneTBB/Guide.md) |
| 3 | **CUDA and SIMT** | NVIDIA GPU programming — thread hierarchy, memory spaces, tiling, streams, Tensor Cores, profiling | [Guide →](CUDA%20and%20SIMT/Guide.md) |
| 4 | **ROCm and HIP** | AMD GPU programming — CDNA architecture, HIP API, HIPIFY porting, Matrix Cores, RCCL multi-GPU | [Guide →](ROCm%20and%20HIP/Guide.md) |
| 5 | **OpenCL and SYCL** | Vendor-neutral compute — OpenCL platform model, SYCL modern C++, sub-groups, FPGA pipes | [Guide →](OpenCL%20and%20SYCL/Guide.md) |

---

## Build, Measure, Ship

Use this completion rubric instead of stopping at "I read the guide."

| Sub-track | Build | Measure | Ship |
|-----------|-------|---------|------|
| C++ and SIMD | vectorized math kernel | scalar vs SIMD speedup, cache behavior | short benchmark report |
| OpenMP / oneTBB | multicore loop or pipeline | scaling across core counts, load imbalance | plot + notes on scheduling choice |
| CUDA | at least one tiled GPU kernel | occupancy, memory throughput, kernel timeline | Nsight profile and tuned kernel |
| ROCm / HIP | HIP port of a CUDA kernel | correctness + performance delta vs CUDA | portability notes |
| OpenCL / SYCL | one portable parallel kernel | backend/device comparison | portability tradeoff summary |

Choose the artifact that matches your available hardware and next track. A reproducible CPU experiment is a complete starting artifact; CUDA is a useful next step for NVIDIA GPU work.

---

## Exit Criteria

You are ready to move on when you can do all of the following without guessing:

- choose between SIMD, CPU threads, and GPU execution for a simple workload
- explain the bottleneck of a kernel in terms of compute, memory, or synchronization
- read a profiler output and identify one actionable optimization
- connect kernel-level behavior to downstream AI workloads such as matmul, attention, or convolution

Carry one measured workload into Phase 3 or Phase 4. You can start those phases while gradually deepening the hardware-specific branches here.

---

### Sub-Track 1: C++ and SIMD

> *The gateway to GPU thinking — same operation, multiple data.*

CPU vector instructions that process 4–16+ data elements per instruction. The same concept GPUs take to 32-wide warps.

| ISA | Extension | Width | FP32 elements/instruction |
|-----|-----------|-------|---------------------------|
| x86 | SSE4.2 | 128 bits | 4 |
| x86 | AVX2 | 256 bits | 8 |
| x86 | AVX-512 | 512 bits | 16 |
| ARM | NEON | 128 bits | 4 |
| ARM | SVE2 | 128–2048 bits | 4–64 (scalable) |

**What the guide covers:**
- Modern C++17 for parallel computing (lambdas, move semantics, smart pointers, parallel STL, templates, constexpr)
- Auto-vectorization and manual intrinsics (`_mm256_fmadd_ps`, `_mm256_load_ps`, etc.)
- Top 10 most-used intrinsics with isolated examples
- AoS vs SoA data layout for vectorization
- Cache alignment, cross-lane limitations, debugging SIMD registers
- Roofline model and the SIMD → GPU mental model bridge
- 9 hands-on projects

---

### Sub-Track 2: OpenMP and oneTBB

> *Scale from one core to many — the CPU parallelism layer.*

**OpenMP** — `#pragma omp parallel for` can distribute loop iterations among a thread team. You must establish independent iterations or use appropriate reductions/synchronization; team size and affinity are runtime choices.

**oneTBB** — task-based parallelism with a work-stealing scheduler. It is useful for irregular tasks, nested parallelism, and pipelines. OpenMP also supports tasks; benchmark the implementation and workload before choosing between them.

| | OpenMP | oneTBB |
|---|---|---|
| Model | Pragma annotations | C++ template library |
| Ease | One-line changes | Moderate (lambdas + ranges) |
| Load balancing | Static/dynamic (loop-level) | Work-stealing (task-level) |
| Best for | Regular loops | Irregular parallelism, pipelines, flow graphs |

**What the guide covers:**
- OpenMP fork-join, scheduling (static/dynamic/guided), reductions, custom reductions, barriers, sections, tasks
- Fibonacci benchmark: serial vs OpenMP vs oneTBB with real measurements
- oneTBB parallel_for, parallel_reduce, parallel_scan (exclusive scan, stream compaction, prefix max, CDF)
- parallel_pipeline with 4-stage batch inference example
- parallel_invoke, task_group, enumerable_thread_specific (ETS)
- Flow graphs: topology diagrams, ML feature extraction, complete video inference pipeline template
- Work-stealing scheduler deep dive

---

### Sub-Track 3: CUDA and SIMT (Main Focus)

> *Thousands of threads, one program — the execution model behind every modern AI workload.*

Prioritize this sub-track when your goal uses NVIDIA GPUs. For CPU inference, AMD systems, or FPGA work, select the branch that matches your target and retain the same correctness and measurement habits.

```
Grid (the entire job)
├── Block 0
│   ├── Warp 0 (threads 0–31)     ← 32 threads execute same instruction (SIMT)
│   ├── Warp 1 (threads 32–63)
│   └── ...
├── Block 1
└── ...
```

**What the guide covers (21 sections):**
- GPU transistor budget (why GPU exists), heterogeneous programming model
- Thread hierarchy (grid → block → warp → thread), indexing, bounds checking
- SM architecture (Ampere/Hopper/Vera): CUDA cores, Tensor Cores, warp schedulers, register file
- Memory hierarchy: registers → shared memory → L1 → L2 → HBM (with latency/bandwidth table)
- Coalesced memory access, bank conflicts, shared memory padding
- Synchronization (`__syncthreads`, atomics, cooperative groups)
- Warp-level primitives (`__shfl_sync`, `__ballot_sync`, `__reduce_sync`)
- Tiled matrix multiply with shared memory
- Streams, events, async copies, multi-stream overlap
- cuBLAS / cuDNN / Tensor Core WMMA
- Nsight Systems and Nsight Compute profiling
- Occupancy analysis, register pressure, launch configuration
- 8 progressive projects (vector add → multi-stream pipeline)

---

### Sub-Track 4: ROCm and HIP

> *AMD's answer to CUDA — write GPU code that runs on both NVIDIA and AMD hardware.*

HIP deliberately resembles CUDA, but a port still needs API, library, synchronization, and performance review. AMD wavefront width depends on the architecture and execution mode; do not hardcode 64 for every AMD GPU. Query the device's `warpSize` and follow the target's documentation. See the [HIP porting guide](https://rocm.docs.amd.com/projects/HIP/en/latest/how-to/hip_porting_guide.html).

| CUDA | HIP | Notes |
|------|-----|-------|
| `cudaMalloc()` | `hipMalloc()` | Same signature |
| `__syncthreads()` | `__syncthreads()` | Identical |
| Warp (32 threads) | Wavefront (target-dependent width) | Audit masks, reductions, and subgroup assumptions |
| Tensor Cores | **Matrix Cores** | rocWMMA API |
| cuBLAS / cuDNN | rocBLAS / MIOpen | Different API for DNN |
| NCCL | **RCCL** | Multi-GPU collectives |

**What the guide covers (11 sections):**
- CDNA (data center) vs RDNA (gaming) architectures
- MI300X chiplet architecture (8 XCDs, 304 CUs, 192 GB HBM3)
- HIP API with code examples, wavefront width implications
- HIPIFY workflow (hipify-clang, what auto-converts, what needs manual work)
- AMD CU deep dive with NVIDIA SM comparison table
- ROCm software stack and library mapping
- HIP streams, events, pinned memory, async execution
- Memory management (explicit, managed/HMM, coherent)
- Matrix Core programming with rocWMMA (FP16/BF16/FP8/INT8)
- Multi-GPU with RCCL and Infinity Fabric topology
- PyTorch on AMD (`torch.cuda` maps to HIP transparently)

---

### Sub-Track 5: OpenCL and SYCL

> *Write once, run anywhere — portable compute across GPU, FPGA, and CPU.*

CUDA targets NVIDIA GPUs. HIP, OpenCL, and SYCL offer different portability routes, but each requires a supported compiler, runtime, device, and feature set. A source port does not guarantee equal performance or support for a new custom accelerator.

```
OpenCL:  Explicit C API — query installed platforms, devices, and extensions
SYCL:    C++ kernels — verify implementation backends and device aspects

OpenCL setup:   ~50 lines of boilerplate (platform, device, context, queue, ...)
SYCL equivalent: ~5 lines (queue + parallel_for lambda)
```

| | CUDA | OpenCL | SYCL |
|---|---|---|---|
| Vendor | NVIDIA only | Any (Khronos) | Any (Khronos) |
| Language | CUDA C++ | OpenCL C (separate) | Standard C++ |
| Kernel style | `__global__` | `.cl` source string | Lambda in host code |
| Targets | NVIDIA GPUs | CPU, GPU, FPGA | CPU, GPU, FPGA |

**What the guide covers:**
- **Part 1 — OpenCL:** platform model, execution model (CUDA mapping table), memory model, full host code walkthrough (8-step setup), tiled matmul with local memory, event profiling, device query
- **Part 2 — SYCL:** buffer/accessor model, USM (CUDA-like pointers), local_accessor tiling, sub-group portable reductions (warp/wavefront-agnostic), FPGA pipes with Intel oneAPI, implementations comparison (DPC++, AdaptiveCpp)
- **Part 3:** decision guide (when to use OpenCL vs SYCL vs CUDA vs HIP)
- 10 progressive projects

---

## Part 3 — The Parallelism Spectrum

Everything in this module sits on a spectrum from narrow (1 core, SIMD) to wide (10,000 GPU threads):

```
Parallelism level    Mechanism           Hardware            Threads    Sub-track
─────────────────────────────────────────────────────────────────────────────────
Data-level           SIMD intrinsics     CPU vector unit     lanes      1
Thread/task-level    OpenMP / oneTBB     CPU cores           team       2
SIMT                 CUDA kernels       NVIDIA GPU SMs      grid       3
GPU parallelism      HIP kernels        supported GPUs      grid       4
Portable compute     OpenCL / SYCL      supported backends  ND-range   5
```

The **same optimization principle** applies at every level: **minimize memory traffic, maximize compute reuse** (tiling). This is equally true for AVX2 intrinsics on a CPU, shared memory in a CUDA kernel, BRAM tiling on an FPGA, and scratchpad design in a custom AI chip.

---

## Part 4 — How This Connects to the Roadmap

| What you learn here | Where it leads |
|--------------------|---------------|
| SIMD / vectorization | Phase 4C: MLIR `vector` dialect, compiler auto-vectorization |
| OpenMP / multi-core | Phase 2: concurrent Linux services; compare their scheduling with MCU/RTOS tasks |
| CUDA kernels | Phase 4B: Jetson inference, Phase 4C: Triton/CUTLASS kernel engineering |
| ROCm / HIP | Phase 5A: AMD GPU infrastructure (MI300X), portable kernel engineering |
| OpenCL | Phase 4A: Xilinx Vitis FPGA host API, embedded GPU compute |
| SYCL | Portable source across supported backends; a custom NPU still needs compiler/runtime support |
| Memory hierarchy thinking | Phase 4A: FPGA BRAM/URAM tiling, Phase 5F: scratchpad design for AI chip |
| Tiled matmul | Phase 4A: HLS matmul accelerator, Phase 5F: systolic array architecture |
| GPU architecture model | Phase 5B: CUDA-X libraries, Phase 5F: design something better |

**The big picture:**
- Phase 1 §4 teaches you to **program** parallel hardware
- Phase 4 teaches you to **optimize and deploy** on real parallel hardware
- Phase 5F teaches you to **design** new parallel hardware

You're learning the workload first. Then you'll build the machine that runs it.

---

## Key Takeaways

1. **Parallelism exists at multiple levels** — instruction (SIMD), thread (OpenMP), massive (CUDA/HIP)
2. **CPU vs GPU is latency vs throughput** — different tools for different jobs
3. **Memory is the real bottleneck** — not compute. This is true for CUDA kernels, FPGA accelerators, and custom AI chips.
4. **CUDA is the industry standard** for GPU compute and AI — learn it first
5. **HIP makes your CUDA skills portable** — same code runs on AMD and NVIDIA
6. **Tiling is the universal optimization** — from shared memory in CUDA to systolic arrays in silicon
7. **Future = heterogeneous + portable** — SYCL targets CPU, GPU, FPGA, and custom accelerators from one codebase

---

## Next

→ [**Phase 3 — Neural Networks**](../../Phase%203%20-%20Artificial%20Intelligence/1.%20Neural%20Networks/Guide.md) — the workloads that make all this parallelism necessary.

## Keep one workload while changing execution models

Use vector add, a reduction, and a small matrix product with identical inputs and an explicit tolerance across the subtracks. Start with scalar C++, then SIMD, CPU tasks, and one GPU backend. HIP and SYCL are transfer electives once you can explain one implementation; they are not prerequisites for every learner.

Report kernel time and end-to-end time separately, including allocation, transfer, and synchronization policy. Record shape, dtype, bytes, compiler flags, and CPU/GPU clocks or power mode. A port is successful when it preserves the contract; it is an optimization only after matched measurements show improvement.

**Carry forward:** Shared benchmark inputs and a backend comparison with identical correctness checks. Continue with [the next course](C%2B%2B%20and%20SIMD/Guide.md) and reuse that evidence.
