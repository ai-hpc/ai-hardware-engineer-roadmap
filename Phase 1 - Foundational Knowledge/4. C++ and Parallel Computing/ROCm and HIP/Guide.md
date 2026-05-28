# ROCm and HIP (Phase 1 §4 — Sub-Track 4)

<div class="course-identity auto-course" style="--course-accent: #7c3aed; --course-accent-rgb: 124, 58, 237;" markdown="1">
<div class="course-identity__icon">RAHP</div>
<div markdown="1">
<p class="course-identity__eyebrow">Deep Dive · Digital Foundations</p>
<p class="course-identity__title">Specialized course identity for ROCm and HIP (Phase 1 §4 — Sub-Track 4).</p>
<p class="course-identity__meta">Artifact: working low-level demo · Measure: timing, memory, correctness</p>
</div>
</div>


**Parent:** [C++ and Parallel Computing](../Guide.md)

> *AMD's answer to CUDA — write GPU code that runs on both NVIDIA and AMD hardware.*

**Prerequisites:** Sub-Track 3 (CUDA and SIMT). You need working CUDA knowledge first — HIP is learned by comparison.

**Layer mapping:** **L1** (application — you write HIP kernels), **L3** (runtime — HIP runtime, ROCm driver stack).

---

## What ROCm Is

**ROCm** (Radeon Open Compute) is AMD's open-source GPU compute platform. It includes the HIP programming language, kernel driver (`amdgpu`), runtime, compiler (`amd-clang`), math libraries (rocBLAS, MIOpen, rocFFT), and profiling tools (Omniperf, Omnitrace). ROCm is to AMD what CUDA is to NVIDIA.

## What HIP Is

**HIP** (Heterogeneous-computing Interface for Portability) is a C++ API almost identical to CUDA. HIP code compiles to both AMD GPUs (via ROCm) and NVIDIA GPUs (via CUDA backend). This means you can write one kernel and run it on both vendors.

**Why learn this now (not just in Phase 5A):**
- AMD Instinct GPUs (MI300X, MI350) are deployed at scale by Microsoft Azure, Meta, Oracle
- Understanding both ecosystems makes you more valuable for any GPU role
- HIP is the fastest path from CUDA to portable GPU code
- Phase 5A (GPU Infrastructure) goes deeper; this sub-track gives you the programming foundation

---

## 1. CDNA vs RDNA — AMD's Two GPU Architectures

AMD makes two completely different GPU architectures for different markets. Understanding the difference is essential before writing any AMD GPU code.

### RDNA (Radeon DNA) — Gaming and Consumer

RDNA powers Radeon RX consumer GPUs. Optimized for graphics workloads: high clock speeds, rasterization, ray tracing, display output.

- **Products:** Radeon RX 7900 XTX, RX 7800 XT, Steam Deck APU, PlayStation 5, Xbox Series X
- **Design focus:** High single-thread performance, graphics pipeline, low power
- **Compute units:** Dual-issue SIMD, 32-wide wavefronts (RDNA 3+ supports wave32 *and* wave64)
- **Memory:** GDDR6 (up to 24 GB), no HBM
- **Use case:** Gaming, content creation, lightweight ML inference

### CDNA (Compute DNA) — Data Center and AI

CDNA powers AMD Instinct data-center GPUs. Optimized for matrix math and HPC — no graphics pipeline at all.

- **Products:** MI300X, MI300A, MI250X, MI210
- **Design focus:** Maximum compute throughput, matrix operations, multi-GPU scaling
- **Compute units:** 64-wide wavefronts, optimized for FP64/FP32/FP16/INT8 matrix operations
- **Matrix Cores:** Hardware matrix multiply-accumulate (equivalent to NVIDIA Tensor Cores)
- **Memory:** HBM3 (up to 192 GB on MI300X with 5.3 TB/s bandwidth)
- **Interconnect:** Infinity Fabric for GPU-to-GPU (like NVLink)
- **Use case:** AI training, AI inference, HPC, scientific computing

### CDNA vs RDNA Comparison

| | RDNA (Consumer) | CDNA (Data Center) |
|---|---|---|
| **Target** | Gaming, desktop | AI training/inference, HPC |
| **Products** | Radeon RX 7900 XTX | Instinct MI300X |
| **Graphics pipeline** | Yes (rasterization, ray tracing) | **No** — compute only |
| **Wavefront size** | 32 (native) + 64 (compatibility) | **64** (native) |
| **Matrix Cores** | Limited | Full matrix core array (FP16, BF16, FP8, INT8) |
| **Memory** | GDDR6 (up to 24 GB) | **HBM3** (up to 192 GB, 5.3 TB/s) |
| **Multi-GPU** | CrossFire (consumer) | **Infinity Fabric** (data center) |
| **ECC** | No | Yes |
| **FP64** | 1/16 rate | **Full rate** (important for scientific computing) |
| **ROCm support** | Limited | **Full** |
| **Price** | $500–1,000 | $10,000–25,000 |

**Why this matters for AI hardware engineers:**
- When you read "AMD GPU for AI," it means **CDNA / Instinct**, not RDNA / Radeon
- CDNA's matrix cores are AMD's answer to NVIDIA tensor cores — the hardware you'd study (or compete with) in Phase 5F (AI Chip Design)
- MI300X's chiplet design (8 XCDs on one package) is a reference for advanced packaging (L8)

### MI300X Architecture (Current Flagship)

```
┌─────────────────────────────────────────────────┐
│              MI300X Package                      │
│                                                  │
│  ┌─────┐ ┌─────┐ ┌─────┐ ┌─────┐              │
│  │ XCD │ │ XCD │ │ XCD │ │ XCD │  ← 8 Compute │
│  │  0  │ │  1  │ │  2  │ │  3  │    Chiplets   │
│  └──┬──┘ └──┬──┘ └──┬──┘ └──┬──┘    (CDNA 3)  │
│     │       │       │       │                    │
│  ┌──┴───────┴───────┴───────┴──┐                │
│  │      Infinity Fabric         │                │
│  │    (inter-chiplet network)   │                │
│  └──┬───────┬───────┬───────┬──┘                │
│  ┌──┴──┐ ┌──┴──┐ ┌──┴──┐ ┌──┴──┐              │
│  │ XCD │ │ XCD │ │ XCD │ │ XCD │               │
│  │  4  │ │  5  │ │  6  │ │  7  │               │
│  └─────┘ └─────┘ └─────┘ └─────┘               │
│                                                  │
│  ┌──────┐ ┌──────┐ ┌──────┐ ┌──────┐          │
│  │ HBM3 │ │ HBM3 │ │ HBM3 │ │ HBM3 │  192 GB │
│  │stack │ │stack │ │stack │ │stack │  5.3TB/s │
│  └──────┘ └──────┘ └──────┘ └──────┘          │
└─────────────────────────────────────────────────┘
```

Each XCD contains 38 Compute Units (CUs), each CU has 64 stream processors + matrix cores. Total: 304 CUs, 19,456 stream processors.

---

## 2. CUDA vs HIP — Almost Identical API

| CUDA | HIP | Notes |
|------|-----|-------|
| `cudaMalloc()` | `hipMalloc()` | Same signature |
| `cudaMemcpy()` | `hipMemcpy()` | Same signature |
| `cudaStream_t` | `hipStream_t` | Same concept |
| `__shared__` | `__shared__` | Identical |
| `__syncthreads()` | `__syncthreads()` | Identical |
| `threadIdx.x` | `threadIdx.x` | Identical |
| `cudaDeviceSynchronize()` | `hipDeviceSynchronize()` | Same |
| `cudaLaunchKernel()` | `hipLaunchKernelGGL()` | Slightly different |
| Warp size: 32 | **Wavefront size: 64** | **Key architectural difference** |

### HIP Kernel Example

```cpp
#include <hip/hip_runtime.h>

__global__ void vector_add(float* a, float* b, float* c, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        c[i] = a[i] + b[i];
    }
}

int main() {
    float *d_a, *d_b, *d_c;
    hipMalloc(&d_a, n * sizeof(float));
    hipMalloc(&d_b, n * sizeof(float));
    hipMalloc(&d_c, n * sizeof(float));

    hipMemcpy(d_a, h_a, n * sizeof(float), hipMemcpyHostToDevice);
    hipMemcpy(d_b, h_b, n * sizeof(float), hipMemcpyHostToDevice);

    vector_add<<<(n+255)/256, 256>>>(d_a, d_b, d_c, n);

    hipMemcpy(h_c, d_c, n * sizeof(float), hipMemcpyDeviceToHost);
    hipFree(d_a); hipFree(d_b); hipFree(d_c);
}
```

If you know CUDA, you already know 95% of HIP. The critical differences:

**1. Wavefront = 64 threads (vs CUDA warp = 32):**
This affects any code that uses warp-level primitives:
```cpp
// CUDA: assumes warp = 32
unsigned mask = __ballot_sync(0xFFFFFFFF, predicate);  // 32-bit mask

// HIP on AMD: wavefront = 64
unsigned long long mask = __ballot(predicate);          // 64-bit mask
```
Reductions, shuffles, and vote operations all need adjustment for the wider wavefront.

**2. Shared memory (LDS) banks:** 64 banks on AMD (vs 32 on NVIDIA). Different bank conflict patterns — code that avoids conflicts on NVIDIA may still conflict on AMD, and vice versa.

**3. Matrix Cores (not Tensor Cores):** AMD uses `rocWMMA` (Wavefront Matrix Multiply-Accumulate) or the Composable Kernel (CK) library instead of NVIDIA's `mma.sync` / CUTLASS.

---

## 3. HIPIFY — Automatic CUDA → HIP Conversion

HIPIFY is AMD's tool suite for converting CUDA code to HIP. It's the fastest way to port existing CUDA projects to run on AMD GPUs.

### Two Conversion Tools

| Tool | How it works | Accuracy | Speed |
|------|-------------|----------|-------|
| **hipify-clang** | Uses Clang's AST parser to understand CUDA code semantically | ~95% accurate | Slower (full compilation) |
| **hipify-perl** | Regex-based find-and-replace (`cuda` → `hip`) | ~85% accurate | Very fast |

### hipify-clang (Recommended)

```bash
# Install (comes with ROCm)
sudo apt install hipify-clang

# Convert a single CUDA file
hipify-clang my_kernel.cu -o my_kernel.hip.cpp

# Convert an entire project (recursive)
hipify-clang --project-dir ./cuda_project --output-dir ./hip_project

# Show what would change without writing (dry run)
hipify-clang my_kernel.cu --print-stats
```

### hipify-perl (Quick and Dirty)

```bash
# Simple text replacement — fast but misses context-dependent conversions
hipify-perl my_kernel.cu > my_kernel.hip.cpp
```

### What HIPIFY Converts Automatically

| CUDA | Converted to HIP | Status |
|------|------------------|--------|
| `cuda*.h` headers | `hip/hip_runtime.h` | Automatic |
| `cudaMalloc/Free/Memcpy` | `hipMalloc/Free/Memcpy` | Automatic |
| `cudaStream_t`, `cudaEvent_t` | `hipStream_t`, `hipEvent_t` | Automatic |
| `__syncthreads()` | `__syncthreads()` | No change needed |
| `atomicAdd()` | `atomicAdd()` | No change needed |
| `cuBLAS` calls | `rocBLAS` calls | **Partial** — API differs slightly |
| `cuDNN` calls | `MIOpen` calls | **Manual** — different API design |
| `cuFFT` calls | `rocFFT` calls | **Partial** |

### What HIPIFY Cannot Convert (Manual Work Required)

| CUDA feature | Why it can't auto-convert | Manual fix |
|-------------|--------------------------|-----------|
| **Inline PTX assembly** | PTX is NVIDIA-specific ISA | Rewrite using HIP intrinsics or GCN assembly |
| **`__ballot_sync(0xFFFFFFFF, ...)`** | Assumes 32-bit warp mask | Use `__ballot()` (64-bit on AMD) |
| **`__shfl_sync(mask, val, lane)`** | Warp size dependent | Use `__shfl(val, lane)` with wavefront awareness |
| **Cooperative groups** | NVIDIA-specific extension | Use HIP cooperative groups (subset supported) |
| **cuDNN → MIOpen** | Completely different API | Rewrite using MIOpen API |
| **Thrust** | NVIDIA template library | Use rocThrust (mostly compatible) |
| **CUTLASS** | NVIDIA template library | Use AMD Composable Kernel (CK) |

### Typical HIPIFY Workflow for a Real Project

```bash
# Step 1: Run hipify-clang on the entire project
hipify-clang --project-dir ./my_cuda_project --output-dir ./my_hip_project

# Step 2: Try to build
cd my_hip_project
mkdir build && cd build
cmake .. -DCMAKE_CXX_COMPILER=hipcc
make -j$(nproc)

# Step 3: Fix compilation errors (usually 5-15% of files need manual fixes)
# Common issues:
#   - warp size assumptions (32 → 64)
#   - library API differences (cuBLAS → rocBLAS)
#   - missing HIP equivalents for NVIDIA extensions

# Step 4: Run and validate
./my_program
# Compare output with CUDA version for correctness

# Step 5: Profile and optimize
rocprof --stats ./my_program
omniperf analyze -p ./profile_output
```

---

## 4. AMD GPU Architecture — CDNA Compute Unit Deep Dive

Understanding the CU (Compute Unit) is essential for writing fast HIP kernels — it's AMD's equivalent of NVIDIA's SM.

```
┌───────────────────────────────────────────┐
│           CDNA 3 Compute Unit (CU)        │
│                                           │
│  ┌─────────────────────────────────────┐  │
│  │  4x SIMD Units (16-wide each)      │  │
│  │  = 64 stream processors total      │  │
│  │  Execute one wavefront (64 threads) │  │
│  └─────────────────────────────────────┘  │
│                                           │
│  ┌─────────────────────────────────────┐  │
│  │  Matrix Cores                       │  │
│  │  FP16, BF16, FP8, INT8 MMA         │  │
│  │  (like NVIDIA Tensor Cores)         │  │
│  └─────────────────────────────────────┘  │
│                                           │
│  ┌──────────────┐  ┌──────────────────┐  │
│  │  Scalar Unit │  │  LDS (64 KB)     │  │
│  │  (control)   │  │  (shared memory) │  │
│  └──────────────┘  └──────────────────┘  │
│                                           │
│  ┌─────────────────────────────────────┐  │
│  │  Vector Register File               │  │
│  │  256 KB (vs ~256 KB per SM)         │  │
│  └─────────────────────────────────────┘  │
│                                           │
│  ┌──────────────┐  ┌──────────────────┐  │
│  │  L1 Cache    │  │  Scheduler       │  │
│  │  (16 KB)     │  │  (wavefront mgr) │  │
│  └──────────────┘  └──────────────────┘  │
└───────────────────────────────────────────┘
```

### NVIDIA SM vs AMD CU Comparison

| | NVIDIA SM (Hopper) | AMD CU (CDNA 3) |
|---|---|---|
| **ALUs per unit** | 128 CUDA cores | 64 stream processors |
| **Thread group** | Warp (32 threads) | Wavefront (64 threads) |
| **Shared memory** | 0–228 KB (configurable with L1) | 64 KB LDS (fixed) |
| **Register file** | 256 KB | 256 KB |
| **Matrix unit** | Tensor Cores (4th gen) | Matrix Cores |
| **L1 cache** | Shared with SMEM (configurable) | 16 KB (separate from LDS) |
| **Max wavefronts/warps** | 64 warps per SM | 32 wavefronts per CU |
| **Occupancy model** | Warps hide latency | Wavefronts hide latency (fewer but wider) |

### Key Optimization Differences When Porting from CUDA

- **Block size:** On NVIDIA, 256 threads = 8 warps. On AMD, 256 threads = 4 wavefronts. Fewer wavefronts means less latency hiding — consider using 512 or 1024 threads per block on AMD.
- **LDS vs shared memory:** AMD's LDS is fixed at 64 KB per CU (not configurable). On NVIDIA you can trade shared memory for L1 cache. Plan your tiling accordingly.
- **Bank conflicts:** 64 LDS banks (vs 32 on NVIDIA). A stride of 2 that's conflict-free on NVIDIA causes 2-way conflicts on AMD.

---

## 5. ROCm Software Stack

```
┌──────────────────────────────────────┐
│  Your HIP Application                │
├──────────────────────────────────────┤
│  Libraries: rocBLAS, MIOpen, rocFFT  │
│             Composable Kernel (CK)   │
├──────────────────────────────────────┤
│  HIP Runtime (hiprt)                 │
├──────────────────────────────────────┤
│  ROCm Compiler (amd-clang / hipcc)   │
│  LLVM AMDGPU backend                 │
├──────────────────────────────────────┤
│  ROCr (Runtime) + ROCt (Thunk)       │
├──────────────────────────────────────┤
│  amdgpu kernel driver (Linux)        │
├──────────────────────────────────────┤
│  AMD GPU Hardware (CDNA / RDNA)      │
└──────────────────────────────────────┘
```

### ROCm Libraries (Equivalent to CUDA-X)

| CUDA-X Library | ROCm Equivalent | Notes |
|---------------|-----------------|-------|
| cuBLAS | **rocBLAS** | GEMM, BLAS routines |
| cuDNN | **MIOpen** | Different API — not a drop-in replacement |
| cuFFT | **rocFFT** | FFT routines |
| cuSPARSE | **rocSPARSE** | Sparse matrix operations |
| cuRAND | **rocRAND** | Random number generation |
| NCCL | **RCCL** | Multi-GPU collectives |
| CUTLASS | **Composable Kernel (CK)** | Custom GEMM/attention kernels |
| Thrust | **rocThrust** | Parallel algorithms (mostly compatible) |
| CUB | **hipCUB** | Block/warp primitives |
| Nsight Systems | **Omnitrace** | Timeline profiling |
| Nsight Compute | **Omniperf** | Kernel-level analysis, roofline |

### Profiling Tools

```bash
# Timeline profiling (like nsys)
omnitrace-run -- ./my_hip_program

# Kernel-level roofline analysis (like ncu)
omniperf profile -n my_run -- ./my_hip_program
omniperf analyze -p workloads/my_run
```

---

## 6. HIP Streams and Asynchronous Execution

Just like CUDA streams, HIP streams let you overlap compute, memory transfers, and host work.

```cpp
hipStream_t s1, s2;
hipStreamCreate(&s1);
hipStreamCreate(&s2);

// Overlap two independent kernels on different streams
kernel_A<<<grid, block, 0, s1>>>(d_a);
kernel_B<<<grid, block, 0, s2>>>(d_b);

// Overlap H2D copy with compute
hipMemcpyAsync(d_in, h_in, bytes, hipMemcpyHostToDevice, s1);
kernel<<<grid, block, 0, s1>>>(d_in, d_out);
hipMemcpyAsync(h_out, d_out, bytes, hipMemcpyDeviceToHost, s1);

// Wait for both streams
hipStreamSynchronize(s1);
hipStreamSynchronize(s2);
hipStreamDestroy(s1);
hipStreamDestroy(s2);
```

**Events for timing:**

```cpp
hipEvent_t start, stop;
hipEventCreate(&start);
hipEventCreate(&stop);

hipEventRecord(start, stream);
kernel<<<grid, block, 0, stream>>>(args);
hipEventRecord(stop, stream);
hipEventSynchronize(stop);

float ms;
hipEventElapsedTime(&ms, start, stop);
printf("Kernel: %.3f ms\n", ms);
```

**Pinned (page-locked) memory** — required for async transfers to overlap with compute:

```cpp
float *h_pinned;
hipHostMalloc(&h_pinned, bytes, hipHostMallocDefault);  // pinned on host
// hipMemcpyAsync now truly async (DMA engine, no CPU involvement)
hipHostFree(h_pinned);
```

---

## 7. Memory Management

### 7.1 Explicit Allocation (Standard)

```cpp
float *d_ptr;
hipMalloc(&d_ptr, bytes);                               // device memory
hipMemcpy(d_ptr, h_ptr, bytes, hipMemcpyHostToDevice);  // explicit copy
hipFree(d_ptr);
```

### 7.2 Managed Memory (HMM)

AMD's Heterogeneous Memory Management (HMM) provides CUDA-like managed memory — the runtime migrates pages between host and device automatically:

```cpp
float *managed;
hipMallocManaged(&managed, bytes);    // accessible from both host and device

// Host code: just use it
for (int i = 0; i < n; i++) managed[i] = i;

// Device code: pages migrate on demand
kernel<<<grid, block>>>(managed, n);
hipDeviceSynchronize();

// Host reads back: pages migrate back
printf("%f\n", managed[0]);
hipFree(managed);
```

**When to use managed memory:**
- Prototyping (avoids manual copy bookkeeping)
- Irregular access patterns where you can't predict which data the GPU needs
- **NOT** for performance-critical paths — explicit async copies with pinned memory are always faster

### 7.3 HIP Memory Comparison

| Method | Performance | Ease of use | When to use |
|--------|-----------|-------------|-------------|
| `hipMalloc` + `hipMemcpy` | Best | Manual | Production kernels |
| `hipMalloc` + `hipMemcpyAsync` + pinned | Best (overlapped) | Manual | Pipeline overlapping |
| `hipMallocManaged` | Good (page faults) | Automatic | Prototyping, irregular access |
| `hipHostMalloc` (coherent) | Moderate | Direct access | Small data, host+device shared |

---

## 8. Matrix Core Programming (rocWMMA)

Matrix Cores are AMD's equivalent of NVIDIA Tensor Cores. The `rocWMMA` library provides a WMMA-style interface similar to NVIDIA's `nvcuda::wmma`.

```cpp
#include <rocwmma/rocwmma.hpp>

using namespace rocwmma;

// Matrix dimensions: 16×16 tiles, FP16 input, FP32 accumulate
constexpr int M = 16, N = 16, K = 16;

__global__ void gemm_wmma(half* A, half* B, float* C, int lda, int ldb, int ldc) {
    // Declare matrix fragments (live in registers)
    fragment<matrix_a, M, N, K, half, row_major> frag_a;
    fragment<matrix_b, M, N, K, half, col_major> frag_b;
    fragment<accumulator, M, N, K, float>         frag_c;

    // Initialize accumulator to zero
    fill_fragment(frag_c, 0.0f);

    // Load tiles from global memory into fragments
    load_matrix_sync(frag_a, A + blockRow * M * lda, lda);
    load_matrix_sync(frag_b, B + blockCol * N, ldb);

    // Matrix multiply-accumulate: C += A × B
    mma_sync(frag_c, frag_a, frag_b, frag_c);

    // Store result back to global memory
    store_matrix_sync(C + blockRow * M * ldc + blockCol * N, frag_c, ldc, mem_row_major);
}
```

**Supported data types on MI300X:**

| Input type | Accumulate type | Tile size | Throughput |
|-----------|----------------|-----------|------------|
| FP16      | FP32           | 16×16×16  | Full rate  |
| BF16      | FP32           | 16×16×16  | Full rate  |
| FP8 (E4M3/E5M2) | FP32    | 16×16×32  | 2× FP16   |
| INT8      | INT32          | 16×16×32  | 2× FP16   |
| FP64      | FP64           | 16×16×4   | Full rate  |

For production-quality GEMM kernels, use **Composable Kernel (CK)** — AMD's equivalent of CUTLASS. CK provides template-based, auto-tuned matrix multiply and attention kernels that target Matrix Cores.

---

## 9. Multi-GPU Programming with RCCL

RCCL (ROCm Communication Collectives Library) is AMD's equivalent of NCCL. It implements ring-allreduce, all-gather, broadcast, and other collectives across multiple AMD GPUs connected via Infinity Fabric or PCIe.

```cpp
#include <rccl/rccl.h>

// Initialize one communicator per GPU
int nGPUs = 8;
ncclComm_t comms[8];
int devs[8] = {0, 1, 2, 3, 4, 5, 6, 7};
ncclCommInitAll(comms, nGPUs, devs);

// All-reduce: sum gradients across all GPUs
for (int i = 0; i < nGPUs; i++) {
    hipSetDevice(i);
    ncclAllReduce(send_buf[i], recv_buf[i], count,
                  ncclFloat, ncclSum, comms[i], streams[i]);
}

// Synchronize all streams
for (int i = 0; i < nGPUs; i++) {
    hipSetDevice(i);
    hipStreamSynchronize(streams[i]);
}
```

**Multi-GPU topology on MI300X systems:**

```
8× MI300X in OAM form factor:

  GPU 0 ←─ Infinity Fabric ──→ GPU 1
    │                             │
    │         ┌───────────┐       │
    ├─────────┤ xGMI/IF  ├───────┤
    │         │  switch   │       │
    ├─────────┤  fabric   ├───────┤
    │         └───────────┘       │
  GPU 2 ←─────────────────────→ GPU 3
    ⋮                             ⋮
  GPU 6 ←─────────────────────→ GPU 7

Each link: ~400 GB/s bidirectional (comparable to NVLink 4.0)
All-to-all bandwidth: sufficient for 8-way tensor parallelism
```

**PyTorch on AMD — it just works:**

```bash
# Install ROCm-enabled PyTorch
pip install torch --index-url https://download.pytorch.org/whl/rocm6.3

# Same code, different backend
import torch
x = torch.randn(4096, 4096, device='cuda')  # 'cuda' maps to HIP on AMD
y = torch.mm(x, x)  # calls rocBLAS under the hood
```

PyTorch uses HIP as the backend — the `torch.cuda.*` API works identically on AMD GPUs. This is why HIP's CUDA-compatible API design was so important.

---

## 10. Resources

| Resource | URL |
|----------|-----|
| ROCm Documentation | https://rocm.docs.amd.com/ |
| HIP Programming Guide | https://rocm.docs.amd.com/projects/HIP/en/latest/ |
| HIPIFY | https://rocm.docs.amd.com/projects/HIPIFY/en/latest/ |
| Composable Kernel (CK) | https://github.com/ROCm/composable_kernel |
| Omniperf | https://rocm.docs.amd.com/projects/omniperf/en/latest/ |
| AMD Instinct MI300X | https://www.amd.com/en/products/accelerators/instinct/mi300/mi300x.html |

---

## 11. Projects

1. **HIPIFY your CUDA kernel** — Take your CUDA vector add and matmul from Sub-Track 3. Convert to HIP using `hipify-clang`. Build with `hipcc`. Run on AMD GPU (or ROCm Docker on NVIDIA). Verify identical output.
2. **Wavefront vs warp** — Write a parallel reduction kernel. Run on both AMD (wavefront=64) and NVIDIA (warp=32). Measure how the wavefront width difference affects performance and code structure.
3. **Profile with Omniperf** — Profile your HIP tiled matmul with `omniperf`. Generate a roofline plot. Compare with NVIDIA `ncu` roofline for the same kernel.
4. **rocBLAS vs cuBLAS** — Call rocBLAS `rocblas_sgemm` for matrix multiply. Compare performance and API with cuBLAS `cublasSgemm` for the same matrix sizes.
5. **HIPIFY a real project** — Pick a small open-source CUDA project (e.g., a convolution kernel or sorting algorithm). Run the full HIPIFY workflow: convert → build → fix errors → validate → profile.
6. **Stream overlap** — Implement a pipeline that overlaps H2D copy, kernel execution, and D2H copy across 3 streams. Measure throughput improvement over synchronous execution.
7. **rocWMMA GEMM** — Write a tiled FP16 matrix multiply using rocWMMA fragments. Compare throughput against rocBLAS `rocblas_hgemm` for M=N=K=4096.
8. **Multi-GPU allreduce** — Use RCCL to sum a 1 GB float array across 2+ GPUs. Measure bandwidth and compare to theoretical Infinity Fabric limit.

---

## Next

→ [**Sub-Track 5 — OpenCL and SYCL**](../OpenCL%20and%20SYCL/Guide.md) — portable compute across GPU, FPGA, and CPU.
