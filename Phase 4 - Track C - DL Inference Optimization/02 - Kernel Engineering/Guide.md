# 02 — Kernel Engineering for Training & Inference

<div class="course-identity auto-course" style="--course-accent: #dc2626; --course-accent-rgb: 220, 38, 38;" markdown="1">
<div class="course-identity__icon">KEFT</div>
<div markdown="1">
<p class="course-identity__eyebrow">Deep Dive · Compiler Track</p>
<p class="course-identity__title">Specialized course identity for 02 — Kernel Engineering for Training & Inference.</p>
<p class="course-identity__meta">Artifact: compiler/inference optimization · Measure: op count, memory, latency</p>
</div>
</div>


**Order:** Second. After you know the graph and bottlenecks (01), you implement and own the kernels.

**Role target:** Core of **MTS Kernels** (Member of Technical Staff, Kernels) and **DL Inference Optimization Engineer** — design, implement, deploy, and maintain high-performance kernels; production reliability; co-design with training, inference, and reinforcement-learning (RL) teams.

---

## Why this is second

Unit 01 tells you *what* to optimize (which ops and where time is spent). This unit is *how*: writing and tuning the actual **kernels** — the small, hardware-specific programs that run on GPU threads and execute the heavy operations (matrix multiplies, attention, etc.). This is the heart of kernel-engineer roles (e.g. NVIDIA, AGI/LLM kernel teams).

---

## 1. Custom kernel authoring frameworks

* **Triton**
    * A language and compiler for GPU kernels expressed through Python. You describe logical tiles; compiler layouts map values across threads, registers, and shared memory as needed. A logical tile is not automatically a shared-memory allocation. Used for matmul, attention, and custom operators.
    * **Automatic tuning** benchmarks a supplied configuration search space for selected keys. It has a compilation/measurement cost and does not guarantee that the search space contains the best kernel; record tuning budget separately from steady-state execution.
    * Integrates with PyTorch via `torch.compile` and custom ops. Study official tutorials and Flash-Attention–style patterns to see how production attention kernels are expressed.
* **CUTLASS / CuTe (CuTe DSL)**
    * **CUTLASS** — NVIDIA's **CUDA template library** for **GEMM** (General Matrix Multiply: the core math behind linear layers and matmuls), **conv** (convolution, used in CNNs), and custom ops. It is the reference implementation for high-performance matrix math on NVIDIA GPUs; many frameworks and libraries build on or mimic its design.
    * **CuTe C++ library** — Layout and tensor abstractions shipped inside CUTLASS. It gives you:
        * **Layout abstractions:** A way to describe *how* data is arranged in memory — **shape** (e.g. 128×64), **stride** (how many elements to skip along each dimension), and **composition** (e.g. a big matrix as a grid of smaller tiles). You define logical **tiles** (rectangular sub-blocks) and map them to **global memory** (GPU DRAM), **shared memory** (on-chip, per thread block), and **registers** (per-thread), without writing error-prone index math by hand.
        * **Copy primitives:** Building blocks for vectorized, asynchronous, and predicated loads/stores on supported targets. Layout abstractions reduce manual indexing, but the author must still validate bounds, alignment, synchronization, lifetimes, and tail handling.
        * **Tiling and partitioning:** You compose layouts to describe **thread block tile** (work per block of threads), **warp tile** (work per warp — 32 threads), and **MMA** (Matrix Multiply-Accumulate: the tensor-core operation that does many multiply-adds in one instruction). Getting these sizes right is essential to match **tensor cores** (hardware units that do dense matrix math very fast), **shared memory size**, and **register pressure** (how many registers each thread needs; too many can limit occupancy).
    * CuTe provides layout and tensor abstractions used by modern CUTLASS. The C++ CuTe library and the Python-facing CuTe DSL are related but distinct interfaces; follow the [official CuTe documentation](https://docs.nvidia.com/cutlass/latest/media/docs/cpp/cute/index.html) and [CuTe DSL documentation](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/overview.html) for the pinned release. Study cuBLASLt as a separate library API rather than assuming its internal implementation.
* **Flash-Attention (v2/v3), Quack**
    * **Fused attention** kernels usually consume already projected **Q/K/V** tensors, then combine attention-score, softmax, and weighted-value calculations without materializing the full score matrix. Q/K/V projection is usually a separate operation; verify the particular implementation's fusion boundary.
    * **Online softmax** means computing the softmax in a single pass over the data (with a recurrence), so you don't need to store all scores in memory — critical for **long-context** (very long sequences) where the full attention matrix would not fit.
    * **Memory-efficient attention** avoids materializing the full N×N attention matrix; you stream and reduce in tiles. Study **tiling** (how the workload is split into blocks), **memory utilization** (how much of bandwidth you use), and **data movement** (what gets read/written and where).
    * **Deep dive:** [FlashAttention — A Systems / Kernel Course](FlashAttention%20Course/Guide.md) — 10-lecture course covering roofline, online softmax, FA1/FA2/FA3 algorithms, repo anatomy, backward correctness harness, inference / paged-KV path, Hopper specifics, and a capstone patch.
* **Mojo, Pallas/Mosaic (JAX)**
    * **Mojo** — A systems language aimed at performance portability across CPUs, GPUs, and other accelerators; useful for writing or porting kernels when you need to target non-NVIDIA hardware.
    * **Pallas / Mosaic** — JAX's way to write custom **GPU and TPU kernels** (TPU = Google's Tensor Processing Unit). Useful when you need to run or port kernels on TPU or compare behavior across backends.

---

## 2. Long-context and attention kernels

* **Challenges**
    * **Memory utilization** — For long sequences, the attention mechanism would naively need O(N²) memory (N = sequence length). You must design kernels that use memory sparingly (tiling, streaming, online softmax) so models can scale to **1M+ context** (millions of tokens).
    * **KV-cache layout** — During autoregressive generation, **key** and **value** tensors from previous tokens are cached and reused. How you lay them out in memory (contiguous, paged, sharded) affects bandwidth and kernel efficiency; poor layout can dominate runtime.
    * **Data movement and bandwidth** — Attention is often **memory-bound**: the GPU spends more time moving data than computing. Kernels that reduce redundant reads/writes and maximize useful **bandwidth** (bytes per second from memory) are critical.
* **Patterns** — Study pinned [FlashAttention](https://github.com/Dao-AILab/flash-attention) and [FlashInfer](https://github.com/flashinfer-ai/flashinfer) implementations. Compare supported variable-length inputs, cache layouts, hardware, dtypes, and test coverage. A project name alone does not establish support for your exact workload.
* **Analysis**
    * **Roofline** — A simple model that relates performance to **arithmetic intensity** (ops per byte). It shows a "roof" (compute-bound limit) and a "ridge" (memory-bound limit). For attention, you often sit on the memory-bound side; the goal is to reduce bytes moved or increase reuse.
    * **Occupancy** — How many **warps** (groups of 32 threads) can run concurrently on an **SM** (Streaming Multiprocessor — the GPU's compute unit). Higher occupancy can hide **latency** (e.g. memory access delay), but too many registers per thread can lower it; you balance register use and parallelism.
    * **Memory-bound bottlenecks** — When the GPU is waiting on memory rather than doing math. You avoid them by reducing data movement, improving locality, and using the right tile sizes so that data in shared memory/registers is reused.
    * **Sustained throughput** — Actual achieved GFLOPS or tokens/sec in real workloads, not just peak theoretical; the metric that matters for production.

---

## 3. Collective communication

When the model or batch is spread across **multiple GPUs** or **multiple nodes** (machines), kernels must exchange data. **Collective communication** is the set of patterns for this: every rank (GPU) participates in the same operation.

* **NCCL** (NVIDIA Collective Communications Library)
    * **All-reduce** — Every GPU has a tensor; after the call, every GPU has the *sum* (or another reduction) of all tensors. Used e.g. to sum gradients in data-parallel training or to synchronize state.
    * **All-gather** — Each GPU has a chunk; after the call, every GPU has the full concatenated tensor. Used when you need the whole tensor on every rank.
    * **Reduce-scatter** — First reduce (e.g. sum) across ranks, then scatter the result so each rank gets a distinct slice. Often used in combination with all-gather for efficient gradient reduction.
    * **Tuning for multi-node, multi-GPU** — Different topologies (NVLink, InfiniBand, Ethernet) and sizes require different algorithms and buffer sizes; NCCL is tuned for NVIDIA hardware.
    * **Overlap of communication with compute** — While one part of the model is computing, you can be sending/receiving data for the next step in the background, so communication does not fully block progress; critical for scaling.
* **MSCCLPP** — Microsoft's collective library; an alternative to NCCL. Compare with NCCL when you need portability (e.g. AMD/other GPUs) or alternative backends.

---

## 4. Production and portability

* **Robustness and testing**
    * **Functional correctness** — Kernels produce the right outputs (or within acceptable numerical tolerance) for all supported inputs and configurations.
    * **Numerical stability** — Especially for custom **attention** and **softmax**: order of operations, scaling, and precision can affect overflow/underflow and accuracy; you must validate on edge cases.
    * **Reproducible benchmarks and CI** — Benchmarks that run the same way everywhere, and **CI** (Continuous Integration) that runs tests and sometimes benchmarks on every change, so regressions are caught early.
* **Porting to alternative hardware** — Evaluate or port kernels to **TPU** (Pallas/Mosaic), AMD GPUs, or other accelerators. This often involves **abstraction layers** (e.g. one kernel description, multiple backends) and **backend-specific trade-offs** (different tile sizes, instruction sets, memory models).
* **Co-design** — Clear **contracts** and **APIs** with training, inference, and RL teams: who owns which kernel, what interfaces they expose, how versioning and deployment work so that production stays reliable as models and frameworks evolve.

---

## 5. Computer architecture and code generation

* **Low-level expertise** — Understanding how your kernels map to hardware:
    * **Memory hierarchy** — Registers (fastest, per thread) → **shared memory** (on-chip, per thread block) → L1/L2 cache → **global memory** (GPU DRAM). Kernels are tuned by keeping hot data in faster levels and minimizing traffic to global memory.
    * **Warp and SM behavior** — A **warp** is 32 threads that execute in lockstep; **SM** (Streaming Multiprocessor) is the unit that runs warps. You reason about **divergence** (threads in a warp taking different paths, which can serialize execution), **occupancy** (how many warps per SM), and **instruction throughput** (how many ops per cycle the hardware can do) to explain and improve performance.
* **Code generation** — How compilers turn high-level descriptions into GPU/TPU code:
    * **IR** (Intermediate Representation) — An internal form of the program (e.g. **MLIR**, **Triton IR**) that the compiler optimizes and then lowers to machine code. Kernel engineers often need to understand or emit IR so that the right kernels are selected and generated.
    * **Mapping ops to kernels** — A single high-level op (e.g. "matmul") can be implemented by many possible kernels (different tile sizes, precisions, backends). Compilers choose or generate kernels based on shape, device, and heuristics; you influence this by writing custom kernels or improving the compiler's selection.

---

## Resources

* [Triton Documentation](https://triton-lang.org/) — Language and GPU kernel patterns.
* [CUTLASS](https://github.com/NVIDIA/cutlass) — CUDA templates for GEMM and more.
* [CuTe](https://docs.nvidia.com/cutlass/latest/media/docs/cpp/cute/index.html) — Layout and tensor abstractions; [CuTe DSL](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/overview.html) for the Python interface.
* [Flash-Attention](https://github.com/Dao-AILab/flash-attention) — Memory-efficient attention.
* [NCCL](https://developer.nvidia.com/nccl) — NVIDIA Collective Communications Library.
* [Pallas / Mosaic (JAX)](https://github.com/google/jax/tree/main/jax/experimental/pallas) — GPU/TPU kernel authoring.
* [FlashInfer](https://github.com/flashinfer-ai/flashinfer) — Inference kernel library and source tests; record a release/commit for every experiment.

---

## Projects

1. **Triton fused kernel** — Implement a **fused operator** (e.g. **layer norm** + **residual**: normalize activations and add the skip connection in one kernel instead of two) or a custom attention variant in Triton. Benchmark vs PyTorch and **profile** with **Nsight Compute** (NVIDIA's profiler for GPU kernels: instruction-level timing, memory throughput, occupancy).
2. **Long-context attention** — Study Flash-Attention's tiling and online softmax. Implement a simplified long-context attention kernel; measure memory vs throughput trade-offs.
3. **NCCL at scale** — Run NCCL all-reduce at scale (multi-GPU or multi-node if available). Tune and document overlap opportunities with compute.
4. **Portability report** — One-page write-up: what it would take to port one of your kernels to TPU via Pallas or to another backend (e.g. Mojo).

---

## Next

→ **[03 — Compiler Stack](../03%20-%20Compiler%20Stack/Guide.md)** — How **IR** (intermediate representation), **scheduling** (when and where ops run), and **codegen** (code generation) produce and select these kernels (e.g. in tinygrad, TVM, MLIR).


## Integration lab: Can the kernel be correct beyond its preferred tile?

Implement a matrix or normalization operator for dimensions divisible by and just beyond the tile size. Validate tail masks and accumulation precision before tuning occupancy or vector width.

**Acceptance evidence:** Submit per-shape error and latency plus resource limits. Keep compile/warmup time outside steady-state timing; pass supported layouts and alignment restrictions to the compiler.

**Continue:** take this artifact to [03 - Compiler Stack](../03%20-%20Compiler%20Stack/Guide.md). Explain which assumption from this course the next engineer must preserve.
