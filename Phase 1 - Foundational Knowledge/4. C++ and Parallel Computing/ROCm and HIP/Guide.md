# ROCm and HIP (Phase 1 §4 — Sub-Track 4)

<div class="course-identity auto-course" style="--course-accent: #7c3aed; --course-accent-rgb: 124, 58, 237;" markdown="1">
<div class="course-identity__icon">RAHP</div>
<div markdown="1">
<p class="course-identity__eyebrow">Deep Dive · Digital Foundations</p>
<p class="course-identity__title">Port the assumptions as well as the API calls.</p>
<p class="course-identity__meta">Artifact: working low-level demo · Measure: timing, memory, correctness</p>
</div>
</div>


**Parent:** [C++ and Parallel Computing](../Guide.md)

> *AMD's answer to CUDA — write GPU code that runs on both NVIDIA and AMD hardware.*

**Prerequisites:** Sub-Track 3 (CUDA and SIMT). You need working CUDA knowledge first — HIP is learned by comparison.

**Layer mapping:** **L1** (application — you write HIP kernels), **L3** (runtime — HIP runtime, ROCm driver stack).

---

## What ROCm and HIP provide

ROCm is AMD's GPU software platform: compiler, runtime, libraries, profiling tools, and integration with the kernel driver. HIP is its C++ programming interface. CUDA experience transfers because many concepts and calls correspond, but portable source still needs a supported compiler, target GPU, libraries, and operating system for each backend.

**Compatibility checked 2026-10-18:** AMD's current [ROCm 10.1.0 compatibility matrix](https://rocm.docs.amd.com/en/latest/compatibility/compatibility-matrix.html) lists Instinct, Radeon, and Ryzen configurations. Select your exact model and OS there before installing. The top-level ROCm version and individual component versions need not match. A container supplies user-space software; it still needs a compatible host driver and actual supported GPU.

Start with a CPU reference and a small ragged vector, then validate a HIP implementation. Only after it matches should you investigate occupancy, matrix instructions, or multiple GPUs. The [OpenCL CPU lab](../OpenCL%20and%20SYCL/labs/README.md) provides an executable heterogeneous-compute exercise if you have no supported GPU.

## 1. CDNA and RDNA: identify the actual target

RDNA is AMD's graphics-oriented family, used in Radeon and some integrated GPUs. CDNA is its data-center compute family, used in recent Instinct accelerators. Both can run AI workloads when the selected device and software combination supports them. Neither family name specifies every memory size, precision, wavefront mode, or supported library.

| Question | Why it changes your program |
|---|---|
| What GPU model and `gfx` target am I compiling for? | Selects legal instructions and compiler code generation. |
| What wavefront width does this target use? | Determines shuffle/reduction participation and mask width. |
| Which matrix operand/accumulator types are supported? | Determines whether a chosen GEMM path is legal. |
| How much local memory and how many registers can a workgroup use? | Limits resident workgroups and legal tile sizes. |
| Is this GPU/OS/framework combination supported? | A generic HIP example does not establish framework availability. |

For a concrete historical case, MI300X is a CDNA 3 accelerator with eight compute chiplets (XCDs). AMD documents 38 active CUs per XCD, or 304 CUs total. This explains why work must be distributed across many compute units; it does not predict the latency of one kernel. Use [AMD's MI300 microarchitecture description](https://rocm.docs.amd.com/en/latest/reference/gpu-arch/mi300.html) for that device and the matching architecture document for a different generation.

**Worked portability check:** a 256-thread workgroup contains eight 32-lane waves or four 64-lane waves. The thread count is unchanged. This arithmetic alone says nothing about which GPU hides latency better: register usage, resident workgroups, instruction throughput, and independent work also matter. Measure several legal workgroup sizes rather than automatically doubling the block size on AMD.

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
| NVIDIA warp: 32 | **Query `warpSize`: commonly 32 or 64 by target** | Cross-lane algorithms must match the target |

### HIP Kernel Example

```cpp
#include <hip/hip_runtime.h>
#include <cstdlib>
#include <iostream>
#include <vector>

__global__ void vector_add(const float* a, const float* b, float* c, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) c[i] = a[i] + b[i];
}

int main() {
    auto check = [](hipError_t status) {
        if (status != hipSuccess) {
            std::cerr << hipGetErrorString(status) << '\n';
            std::exit(EXIT_FAILURE);
        }
    };
    constexpr int n = 1025; // Deliberately not divisible by the block size.
    const size_t bytes = n * sizeof(float);
    std::vector<float> a(n, 1.0f), b(n, 2.0f), c(n);
    float *d_a, *d_b, *d_c;
    check(hipMalloc(&d_a, bytes));
    check(hipMalloc(&d_b, bytes));
    check(hipMalloc(&d_c, bytes));
    check(hipMemcpy(d_a, a.data(), bytes, hipMemcpyHostToDevice));
    check(hipMemcpy(d_b, b.data(), bytes, hipMemcpyHostToDevice));
    vector_add<<<(n + 255) / 256, 256>>>(d_a, d_b, d_c, n);
    check(hipGetLastError());
    check(hipDeviceSynchronize());
    check(hipMemcpy(c.data(), d_c, bytes, hipMemcpyDeviceToHost));
    check(hipFree(d_a));
    check(hipFree(d_b));
    check(hipFree(d_c));
    for (float value : c) if (value != 3.0f) return EXIT_FAILURE;
    std::cout << "Validated " << n << " elements\n";
}
```

Build with the selected HIP toolchain and run on a supported device. This course review did not execute HIP hardware; preserve the compile/runtime log when reproducing it.


CUDA experience transfers to HIP, but API coverage, execution width, and library support must be checked for the selected release. The critical differences:

**1. Wavefront width depends on the target:**
This affects any code that uses warp-level primitives:
```cpp
// CUDA: assumes warp = 32
unsigned mask = __ballot_sync(0xFFFFFFFF, predicate);  // 32-bit mask

// HIP on AMD: wavefront = 64
unsigned long long mask = __ballot(predicate);          // 64-bit mask
```
Reductions, shuffles, and vote masks must handle the actual wavefront width; a hard-coded 64 is no more portable than a hard-coded 32.

**2. Shared memory (LDS) banks:** AMD bank counts vary by architecture. Re-derive conflicts for the target rather than assuming 64 banks for every GPU. [AMD’s LDS guide](https://rocm.docs.amd.com/projects/composable_kernel/en/7.13.0-preview/conceptual/ck_tile/hardware/lds_bank_conflicts.html) explains the bank mapping.

**3. Matrix Cores (not Tensor Cores):** AMD uses `rocWMMA` (Wavefront Matrix Multiply-Accumulate) or the Composable Kernel (CK) library instead of NVIDIA's `mma.sync` / CUTLASS.

---

## 3. HIPIFY — Automatic CUDA → HIP Conversion

HIPIFY is AMD's tool suite for converting CUDA code to HIP. It's the fastest way to port existing CUDA projects to run on AMD GPUs.

HIPIFY converts supported source constructs; it neither runs the program nor verifies correctness. Consult [AMD’s supported API tables](https://rocmdocs.amd.com/projects/HIPIFY/en/latest/reference/hip_roc_supported_apis.html) and [portability workflow](https://rocm.blogs.amd.com/software-tools-optimization/hipify/README.html).

### Two Conversion Tools

| Tool | How it works | Review requirement | Cost |
|------|-------------|----------|-------|
| **hipify-clang** | Parses CUDA with Clang | Inspect unsupported APIs and semantic assumptions | Requires source/build context |
| **hipify-perl** | Textual substitutions | Review context-sensitive changes manually | Lightweight initial pass |

### hipify-clang (Recommended)

```bash
# Install (comes with ROCm)
sudo apt install hipify-clang

# Convert a single CUDA file
hipify-clang my_kernel.cu -o my_kernel.hip.cpp

# For a project, inspect hipify-clang --help and supply its compilation database.
# Convert a reviewed list of translation units with the flags for that release.

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
# Step 1: in a separate source checkout, convert each selected translation unit
hipify-clang my_kernel.cu -o my_kernel.hip.cpp

# Step 2: Try to build
cd my_hip_project
mkdir build && cd build
cmake .. -DCMAKE_CXX_COMPILER=hipcc
make -j$(nproc)

# Step 3: Fix compilation errors and inspect unconverted features
# Common issues:
#   - warp size assumptions (32 → 64)
#   - library API differences (cuBLAS → rocBLAS)
#   - missing HIP equivalents for NVIDIA extensions

# Step 4: Run and validate
./my_program
# Compare output with CUDA version for correctness

# Step 5: Profile and optimize
rocprofv3 --kernel-trace --output-format csv -- ./my_program
```

---

## 4. Compute units, local memory, and occupancy

A compute unit schedules wavefronts, executes scalar/vector/matrix instructions, and provides registers and local data share (LDS). These resources constrain the number of simultaneously resident workgroups. Think of occupancy as the available pool of work that can hide stalls, not as a score to maximize regardless of useful computation.

Suppose a teaching device has 64 KiB of usable LDS per CU and a kernel reserves 24 KiB per workgroup. LDS alone permits at most `floor(64/24) = 2` resident workgroups. Registers, thread slots, or architectural limits can lower that count. Reducing the tile to 16 KiB permits four by this *one* limit, but may increase global-memory traffic. These are illustrative capacities, not a specification for every AMD GPU.

A bank conflict is also a mapping calculation, not a vendor slogan. In a simplified model with 32 banks and 4-byte elements, consecutive lanes accessing `x[2*lane]` touch only 16 banks, so distinct addresses repeat banks. Broadcasts, vector instruction phases, swizzles, and the actual bank organization change the result. Consult the target's LDS documentation before applying padding learned on a different architecture.

For each candidate tile, record workgroup size, registers, LDS bytes, measured time, and validated output. Compare those observations with the profiler's resource and stall counters. A larger block can reduce occupancy if it consumes enough resources to prevent a second block from fitting.

---

## 5. ROCm Software Stack

```
┌──────────────────────────────────────┐
│  Your HIP Application                │
├──────────────────────────────────────┤
│  Libraries: rocBLAS, MIOpen, rocFFT  │
│             Composable Kernel (CK)   │
├──────────────────────────────────────┤
│  HIP Runtime                 │
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
| Nsight Systems | **ROCm Systems Profiler** | CPU/GPU timeline analysis |
| Nsight Compute | **ROCm Compute Profiler** | Kernel counters and roofline analysis |

### Profiling tools

Use the profiler packaged for your ROCm release. Older material calls ROCm Compute Profiler “Omniperf” and ROCm Systems Profiler “Omnitrace.” Their current names and interfaces are documented separately.

```bash
# Documented rocprofv3 command; requires a supported GPU and installed SDK.
rocprofv3 --kernel-trace --output-format csv -- ./my_hip_program
```

This produces dispatch timing records; it does not by itself explain a memory bottleneck. Start with [rocprofv3 tracing](https://rocm.docs.amd.com/projects/rocprofiler-sdk/en/latest/how-to/using-rocprofv3.html), then use [ROCm Compute Profiler](https://rocm.docs.amd.com/projects/rocprofiler-compute/en/latest/) for hardware counters. Counter collection can replay kernels or perturb timing. Keep correctness checks and an unprofiled timing run alongside the profile. These GPU commands were source-checked, not executed in the CPU course review.

---

## 6. HIP Streams and Asynchronous Execution

HIP streams express ordering within a stream and allow independent work across streams. Concurrent execution depends on device resources and dependencies. The following sketch omits error handling for readability; use the checked-call pattern from the vector example in a complete program. Host and device buffers must remain valid until the relevant stream finishes.

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

**Pinned (page-locked) host memory** enables eligible asynchronous transfer paths; it does not guarantee overlap. This allocation sketch frees only after all operations using it have completed:

```cpp
float *h_pinned;
hipHostMalloc(&h_pinned, bytes, hipHostMallocDefault);  // pinned on host
// Enqueue the copy, then wait for its completion before reusing or freeing this buffer.
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

`hipMallocManaged` offers a shared virtual allocation. Migration, host/device accessibility, and coherence behavior depend on GPU, OS, driver, and managed-memory capabilities; HMM support is not a universal consequence of using HIP. Check the target before assuming demand paging or concurrent access. This sketch establishes a device-completion boundary before the host reads the result:

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
- Performance-sensitive code: compare explicit async copies with pinned memory against managed memory on the actual platform

### 7.3 HIP Memory Comparison

| Method | Performance | Ease of use | When to use |
|--------|-----------|-------------|-------------|
| `hipMalloc` + `hipMemcpy` | Measure copy and execution costs | Manual | Explicit ownership and movement |
| `hipMalloc` + `hipMemcpyAsync` + pinned | Overlap depends on resources/dependencies | Manual | Pipelined independent work |
| `hipMallocManaged` | Depends on migration and access pattern | Shared allocation | Prototyping after checking capabilities |
| `hipHostMalloc` | Depends on placement and interconnect | Pinned host allocation | Transfers; supported mapped access with explicit ordering |

---

## 8. Matrix instructions and rocWMMA

rocWMMA provides matrix fragments and collective load, multiply-accumulate, and store operations. A fragment distributes a small matrix across participating lanes; it is not a normal row-major array owned by one thread. Start from a supported sample in the [rocWMMA documentation](https://rocm.docs.amd.com/projects/rocWMMA/en/latest/) for your GPU and release.

The mathematical contract for a single tile is `C[m,n] += sum_k A[m,k] * B[k,n]`. To implement it, specify input and accumulator types, tile dimensions, A/B storage layouts, leading dimensions, and the exact participating wave. For row-major B, element `(k,n)` is at `B[k*ldb+n]`; for column-major B it is at `B[k+n*ldb]`. Changing only the fragment's layout tag while keeping the old pointer arithmetic changes the matrix being read.

A complete GEMM must also initialize accumulators, iterate across all K tiles, handle ragged M/N/K edges, and store each output once. All required lanes must reach collective fragment operations consistently. A kernel containing undefined `blockRow`/`blockCol`, an omitted K loop, or an unsupported fragment specialization is not a runnable general GEMM.

**Exercise:** A is 2×3 and B is 3×2, both row-major. A contains rows `[1,2,3]`, `[4,5,6]`; B contains `[7,8]`, `[9,10]`, `[11,12]`. Compute C before using any matrix library. **Answer:** `[[58,64],[139,154]]`. If a fixed 16×16 hardware tile is used for this example, fill out-of-range operands with zero and store only valid output coordinates. Padding establishes correctness but may waste most of the hardware work.

For a larger implementation, compare a supported rocBLAS/hipBLASLt or Composable Kernel path before writing your own fragment kernel. Precision support and dense/sparse peak rates belong to a specific device specification; they are not a universal throughput table for every fragment type.

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

// Group calls when one host thread drives multiple devices.
ncclGroupStart();
// All-reduce: sum gradients across all GPUs
for (int i = 0; i < nGPUs; i++) {
    hipSetDevice(i);
    ncclAllReduce(send_buf[i], recv_buf[i], count,
                  ncclFloat, ncclSum, comms[i], streams[i]);
}
ncclGroupEnd();

// Synchronize all streams
for (int i = 0; i < nGPUs; i++) {
    hipSetDevice(i);
    hipStreamSynchronize(streams[i]);
}
```

This is an API sketch: allocate buffers and streams on each device, check every HIP/RCCL return code, and destroy communicators and streams after completion. Test two devices and a small known sum before scaling the buffer size.

Collective performance depends on the actual GPU-to-GPU links, PCIe switches, NUMA placement, message size, algorithm, and concurrent traffic. Aggregate fabric bandwidth is not the bandwidth of one link or the end-to-end all-reduce rate. Record topology and measure application-visible bytes/time with a clearly stated convention.

**PyTorch on AMD:** choose a supported framework build from [AMD's PyTorch installation guide](https://rocm.docs.amd.com/projects/ai-ecosystem/en/latest/frameworks/pytorch/install.html). Avoid copying a wheel URL from an older ROCm release into a different stack. After installation, this Python smoke test checks which build is running and a known result:

```python
import torch
print("PyTorch:", torch.__version__, "HIP:", torch.version.hip)
assert torch.version.hip is not None, "This is not a ROCm PyTorch build"
assert torch.cuda.is_available(), "No accessible supported GPU"
a = torch.tensor([[1., 2., 3.], [4., 5., 6.]], device="cuda")
b = torch.tensor([[7., 8.], [9., 10.], [11., 12.]], device="cuda")
expected = torch.tensor([[58., 64.], [139., 154.]])
torch.testing.assert_close((a @ b).cpu(), expected)
print(torch.cuda.get_device_name(0), "matmul passed")
```

PyTorch retains the `cuda` device spelling for its HIP backend. That API convention does not imply identical availability for every extension, dtype, operator, or profiler. This smoke test has not been run on AMD hardware in this course review.

---

## 10. Resources

| Resource | URL |
|----------|-----|
| ROCm Documentation | https://rocm.docs.amd.com/ |
| HIP Programming Guide | https://rocm.docs.amd.com/projects/HIP/en/latest/ |
| HIPIFY | https://rocm.docs.amd.com/projects/HIPIFY/en/latest/ |
| Composable Kernel (CK) | https://github.com/ROCm/composable_kernel |
| ROCm Compute Profiler | https://rocm.docs.amd.com/projects/rocprofiler-compute/en/latest/ |
| AMD Instinct MI300X | https://www.amd.com/en/products/accelerators/instinct/mi300/mi300x.html |

---

## 11. Projects

1. **HIPIFY your CUDA kernel** — Take your CUDA vector add and matmul from Sub-Track 3. Convert to HIP using `hipify-clang`. Build with `hipcc`. Run on a supported AMD GPU, or use a separately supported HIP-on-CUDA toolchain for NVIDIA. A ROCm container does not emulate AMD hardware. Compare output with a stated numerical tolerance.
2. **Wavefront vs warp** — Write a parallel reduction kernel. Run on supported AMD and NVIDIA targets, querying and recording each execution width. Measure how the wavefront width difference affects performance and code structure.
3. **Profile with ROCm Compute Profiler** — Profile your HIP tiled matmul with the profiler supplied for your ROCm release. Generate a roofline plot. Compare with NVIDIA `ncu` roofline for the same kernel.
4. **rocBLAS vs cuBLAS** — Call rocBLAS `rocblas_sgemm` for matrix multiply. Compare performance and API with cuBLAS `cublasSgemm` for the same matrix sizes.
5. **HIPIFY a real project** — Pick a small open-source CUDA project (e.g., a convolution kernel or sorting algorithm). Run the full HIPIFY workflow: convert → build → fix errors → validate → profile.
6. **Stream overlap** — Implement a pipeline that overlaps H2D copy, kernel execution, and D2H copy across 3 streams. Measure throughput improvement over synchronous execution.
7. **rocWMMA GEMM** — Write a tiled FP16 matrix multiply using rocWMMA fragments. Compare throughput against rocBLAS `rocblas_hgemm` for M=N=K=4096.
8. **Multi-GPU allreduce** — Use RCCL to sum a 1 GB float array across 2+ GPUs. Measure bandwidth and compare to theoretical Infinity Fabric limit.

---

## Next

→ [**Sub-Track 5 — OpenCL and SYCL**](../OpenCL%20and%20SYCL/Guide.md) — portable compute across GPU, FPGA, and CPU.

## Port the assumptions as well as the API calls

Port the same ragged reduction and matrix product used in CUDA. List every NVIDIA-specific assumption: warp mask width, shuffle participation, inline assembly, shared-bank mapping, matrix instruction, and library call. Replace assumptions using documented target capabilities; compile success is only the first checkpoint.

Compare output tolerance first, then kernel and transfer time separately. Preserve an unsupported-feature list and a target manifest with GPU architecture, compiler/runtime, and chosen matrix library. This becomes a portability test for the future compiler: which decisions belong in a shared IR, and which belong in target lowering?

**Carry forward:** Portability matrix and matched CUDA/HIP correctness/performance results. Continue with [the next course](../OpenCL%20and%20SYCL/Guide.md) and reuse that evidence.
