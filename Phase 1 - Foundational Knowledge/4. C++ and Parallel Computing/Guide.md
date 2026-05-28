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

Every AI inference chip — from NVIDIA's H100 to your future custom NPU — exists because **sequential computing hit a wall**. Understanding *why* parallelism is necessary, *how* it evolved, and *how to program it* is the foundation for everything in this roadmap.

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

For decades, performance came from clock speed scaling (4.77 MHz in 1980 → 3+ GHz by 2004). Then the **power wall** hit: power scales as `P ∝ V² × f`, and chips couldn't dissipate the heat. Clock speeds plateaued at 3–4 GHz.

**This is why AI chips exist.** If clock speed still scaled freely, you'd just run everything on a faster CPU. The power wall forced the industry into parallelism and specialization.

### Step 2: Instruction-Level Parallelism (ILP / SIMD)

Multiple things per clock cycle inside a single core:

```
Pipelining:   overlap fetch/decode/execute of different instructions
Superscalar:  issue 4–6 independent instructions per cycle
SIMD:         one instruction processes 4/8/16/32 data elements

Without SIMD:       With SIMD (8-wide AVX2):
a[0] = b[0] + c[0]  a[0..7] = b[0..7] + c[0..7]   ← ONE instruction
a[1] = b[1] + c[1]
...                  8× throughput improvement
a[7] = b[7] + c[7]
```

This is the first taste of **data parallelism** — the same concept that GPUs take to the extreme.

### Step 3: Multi-Core / Shared Memory

When clock speed stopped scaling, chip designers added cores (2005: 2 cores → 2024: 96 cores). But software must explicitly use multiple cores — a single-threaded program leaves 95 cores idle.

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
| Cores | 4–96 (complex) | 1,000–16,000 (simple) |
| Optimized for | Latency (one task fast) | Throughput (many tasks at once) |
| Control logic | ~50% of die area | ~5% of die area |
| Cache per core | Large (MB) | Small (KB) |
| Best for | Sequential code, branching | Data-parallel: matrix math, convolution |

Neural network inference is almost entirely matrix multiplication — perfectly data-parallel. A GPU processes 1,000× more multiply-accumulate operations per second than a CPU at the same power.

---

## Part 2 — The Five Sub-Tracks

Study these **in order**. Each builds on the previous.

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

If you only complete one artifact in this whole module, make it the CUDA one.

---

## Exit Criteria

You are ready to move on when you can do all of the following without guessing:

- choose between SIMD, CPU threads, and GPU execution for a simple workload
- explain the bottleneck of a kernel in terms of compute, memory, or synchronization
- read a profiler output and identify one actionable optimization
- connect kernel-level behavior to downstream AI workloads such as matmul, attention, or convolution

That is the minimum bar for Phase 3 and Phase 4 to be useful.

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

**OpenMP** — add `#pragma omp parallel for` to a loop and it distributes across all cores. Zero boilerplate.

**oneTBB** — task-based parallelism with work-stealing scheduler. Better for irregular workloads (tree traversals, graph algorithms, nested parallelism).

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

This is the **most important sub-track** in Phase 1. CUDA programs NVIDIA GPUs, which run the vast majority of AI training and inference.

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

HIP is 95% identical to CUDA. The critical difference: AMD wavefronts are **64 threads** wide (vs CUDA warps = 32). AMD CDNA GPUs (MI300X) have 192 GB HBM3 at 5.3 TB/s — competitive with H100.

| CUDA | HIP | Notes |
|------|-----|-------|
| `cudaMalloc()` | `hipMalloc()` | Same signature |
| `__syncthreads()` | `__syncthreads()` | Identical |
| Warp (32 threads) | **Wavefront (64 threads)** | Key architectural difference |
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

CUDA locks you to NVIDIA. HIP gets you AMD. OpenCL and SYCL get you **everything**.

```
OpenCL:  Explicit C API — verbose but runs on any device
SYCL:    Modern C++ lambdas — clean but same portability

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
Instruction-level    SIMD intrinsics     CPU vector unit     4–16       1
Thread-level         OpenMP / oneTBB     CPU cores           4–96       2
Massive (NVIDIA)     CUDA kernels        NVIDIA GPU SMs      10K+       3
Massive (AMD)        HIP kernels         AMD GPU CUs         10K+       4
Portable massive     OpenCL / SYCL       Any GPU/FPGA/CPU    10K+       5
```

The **same optimization principle** applies at every level: **minimize memory traffic, maximize compute reuse** (tiling). This is equally true for AVX2 intrinsics on a CPU, shared memory in a CUDA kernel, BRAM tiling on an FPGA, and scratchpad design in a custom AI chip.

---

## Part 4 — How This Connects to the Roadmap

| What you learn here | Where it leads |
|--------------------|---------------|
| SIMD / vectorization | Phase 4C: MLIR `vector` dialect, compiler auto-vectorization |
| OpenMP / multi-core | Phase 2: FreeRTOS multi-core on Jetson SPE, Zynq PS |
| CUDA kernels | Phase 4B: Jetson inference, Phase 4C: Triton/CUTLASS kernel engineering |
| ROCm / HIP | Phase 5A: AMD GPU infrastructure (MI300X), portable kernel engineering |
| OpenCL | Phase 4A: Xilinx Vitis FPGA host API, embedded GPU compute |
| SYCL | Future portable compute — CPU, GPU, FPGA, custom NPU from one source |
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
