# AMD GPU for HPC and AI

<div class="course-identity auto-course" style="--course-accent: #ca8a04; --course-accent-rgb: 202, 138, 4;" markdown="1">
<div class="course-identity__icon">AGFH</div>
<div markdown="1">
<p class="course-identity__eyebrow">Deep Dive · Specialization</p>
<p class="course-identity__title">Specialized course identity for AMD GPU for HPC and AI.</p>
<p class="course-identity__meta">Artifact: specialization case study · Measure: performance, reliability, role fit</p>
</div>
</div>


**Timeline:** 6–12 months.

**Prerequisites:** Phase 1 §4 (C++ and Parallel Computing — CUDA/OpenCL), Phase 4 Track C (ML compiler fundamentals), Nvidia GPU sub-track recommended for context.

---

## Why AMD GPU

AMD's Instinct GPUs (MI300X, MI300A, MI350) are the primary alternative to Nvidia for AI training and inference at scale. Understanding both ecosystems makes you more versatile and valuable — especially as cloud providers (Azure, Oracle, Meta) deploy AMD hardware alongside Nvidia.

---

## 1. AMD GPU Architecture

* **CDNA vs RDNA:**
    * CDNA (Compute DNA): data-center optimized — MI300X, MI250X. Matrix cores, large HBM, high bandwidth.
    * RDNA (Radeon DNA): consumer/gaming GPUs. Relevant for understanding the architecture family.
* **MI300X architecture:**
    * Chiplet design: 8 XCDs (Accelerator Complex Dies) + 4 IODs.
    * 192 GB HBM3 with 5.3 TB/s bandwidth.
    * Matrix cores: FP16, BF16, FP8, INT8.
    * Infinity Fabric for inter-chiplet and inter-GPU communication.
* **Compute units:**
    * CDNA wavefronts have 64 threads; RDNA targets can use 32-thread waves. Query the target and avoid a universal AMD=64 assumption.
    * SIMD units, LDS (Local Data Share) vs CUDA shared memory.
    * Occupancy model differences from Nvidia.

---

## 2. ROCm Software Stack

* **ROCm (Radeon Open Compute):**
    * Open-source GPU compute platform. ROCm 6+ for MI300X.
    * Kernel driver (`amdgpu`), runtime (`hip-runtime`), compiler (`amd-clang`).
* **HIP (Heterogeneous-computing Interface for Portability):**
    * CUDA-like API for AMD GPUs. `hipMalloc`, `hipMemcpy`, `hipLaunchKernelGGL`.
    * **HIPIFY:** Tool to convert CUDA source to HIP (`hipify-perl`, `hipify-clang`).
    * Source portability is a starting point; backend support and performance still require separate validation.
* **Libraries:**
    * **rocBLAS** (GEMM), **MIOpen** (cuDNN equivalent), **rocFFT**, **rocSPARSE**.
    * **RCCL** (ROCm Communication Collectives Library) — AMD's NCCL equivalent for multi-GPU.
    * **Composable Kernel (CK)** — AMD's equivalent to CUTLASS for custom GEMM/attention kernels.
* **Profiling:**
    * **rocProfiler** / **rocTracer** — kernel-level profiling.
    * **Omniperf** — roofline analysis, occupancy, memory throughput (like Nsight Compute).
    * **Omnitrace** — timeline profiling (like Nsight Systems).

---

## 3. Porting CUDA to AMD

* **HIPIFY workflow:**
    * Automated conversion: `hipify-clang` for source-to-source translation.
    * Manual work: CUDA-specific intrinsics, inline PTX, warp-level primitives.
* **Key differences:**
    * CUDA warps have 32 lanes; AMD wave size depends on the target. Use `warpSize` and a sufficiently wide lane mask for reduction, ballot, and shuffle code.
    * LDS bank count and access phases depend on architecture and instruction width. MI300/CDNA3 uses 32 banks, so “AMD has 64 banks” is not a portable rule. [AMD LDS guidance](https://rocm.docs.amd.com/projects/HIP/en/latest/understand/performance_optimization.html).
    * Memory coalescing rules differ slightly.
    * No direct equivalent to Nvidia Tensor Cores — use Matrix Cores via `rocWMMA` or CK.
* **Framework support:**
    * PyTorch: native ROCm support (`torch.cuda` works on AMD via HIP).
    * TensorFlow, JAX: ROCm backends available.
    * TVM, ONNX Runtime: ROCm execution providers.
    * vLLM, TensorRT-LLM alternatives for AMD: vLLM + ROCm, AMD Inference Server.

---

## 4. Multi-GPU and Cluster Operations

* **Infinity Fabric:**
    * Intra-node GPU-to-GPU interconnect. Bandwidth and topology compared to NVLink.
    * MI300X: 896 GB/s aggregate fabric bandwidth per GPU.
* **RCCL:**
    * All-reduce, all-gather, reduce-scatter on AMD GPUs.
    * Tuning for MI300X topology and Infinity Fabric.
* **Multi-node:**
    * InfiniBand and RoCE (RDMA over Converged Ethernet) support.
    * GPUDirect RDMA equivalent on AMD.
    * Slurm and Kubernetes with ROCm container support.

---

## 5. Kernel Development on AMD

* **HIP kernel writing:**
    * Thread hierarchy: grid → block → thread (same as CUDA).
    * Shared memory (`__shared__`), synchronization (`__syncthreads()`).
    * Wavefront-aware programming: `__ballot`, `__shfl`, warp intrinsics via wavefront equivalents.
* **Composable Kernel (CK):**
    * Templated library for high-performance GEMM, attention, and custom ops.
    * Tile-based programming model (similar concept to CuTe/CUTLASS).
    * Writing custom kernels with CK building blocks.
* **Triton on AMD:**
    * Triton supports AMD GPUs via ROCm backend.
    * Same Python kernel code, different backend code generation.
    * Performance tuning differences (tile sizes, occupancy targets).

---

## Resources

* [ROCm Documentation](https://rocm.docs.amd.com/)
* [HIP Programming Guide](https://rocm.docs.amd.com/projects/HIP/en/latest/)
* [Composable Kernel](https://github.com/ROCm/composable_kernel)
* [HIPIFY](https://rocm.docs.amd.com/projects/HIPIFY/en/latest/)
* [Omniperf](https://rocm.docs.amd.com/projects/omniperf/en/latest/)
* [AMD Instinct MI300X](https://www.amd.com/en/products/accelerators/instinct/mi300/mi300x.html)

---

## Projects

1. **HIPIFY a CUDA kernel** — Take your CUDA matmul or vector-add kernel from Phase 1 §4. Convert to HIP using `hipify-clang`. Run on a supported AMD GPU; a ROCm container packages userspace and does not emulate an AMD GPU. Compare output and performance.
2. **Profile with Omniperf** — Profile a PyTorch model on ROCm. Generate a roofline plot. Identify compute-bound vs memory-bound layers.
3. **RCCL benchmark** — Run RCCL all-reduce across multiple AMD GPUs. Compare bandwidth and latency with NCCL on equivalent Nvidia hardware.
4. **Triton on AMD** — Write a Triton fused kernel (e.g., layer norm + residual). Run on both Nvidia (CUDA) and AMD (ROCm). Compare generated code and performance.
5. **CK custom GEMM** — Use Composable Kernel to implement a custom GEMM with epilogue fusion (bias + activation). Benchmark against rocBLAS.

## Port one reduction before a full model

Bring a CUDA reduction from Phase 1 and test input lengths 31, 32, 33, 63, 64, and 65 plus a large nonmultiple. These cases expose hard-coded warp masks, lane counts, and incomplete tail handling. Verify the same numeric contract on CPU and the selected AMD target before profiling.

Separate “HIPIFY produced source,” “the kernel passed,” and “the port is efficient.” Record architecture, wave size, workgroup size, registers/LDS, achieved bandwidth, and backend/library revisions. Pass that result to [platform evaluation](../Accelerator%20Platform%20Evaluation/Guide.md); a container image or successful compilation alone does not establish portability.

## Practice and assessment

**Entry diagnostic (prerequisite check):** Draw GPU/CPU/NIC topology and explain bandwidth versus message latency. If this is unfamiliar, use the prerequisite material above or the [roadmap diagnostics](../../../Learning-Guide.md#diagnose-before-reading) before starting the lab.

1. **Baseline lab:** Establish compute, memory-transfer, and collective baselines on available hardware.
2. **Guided variation:** Change placement or message size and test a recoverable worker failure.
3. **Independent transfer:** repeat the method on a new input or configuration and explain where your original assumptions stop applying.

**Evidence to submit:** Topology, achieved bandwidth/latency, scaling efficiency, error logs, and configuration. Keep raw results and reproduction steps; distinguish measured results from estimates or work not run.

**Self-check:** Which observation would disprove your explanation of the result? Identify a competing explanation and the smallest experiment that separates them. A passing answer names a concrete predicted difference, not only another tool to try.

**Exit criteria:** reproduce the baseline, demonstrate the negative/variation check, and meet the [shared completion rubric](../../../Learning-Guide.md#completion-rubric). Use the [lesson cycle](../../../Learning-Guide.md#a-repeatable-lesson-cycle) for retrieval practice and spaced review.
