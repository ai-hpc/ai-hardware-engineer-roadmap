# OpenCL and SYCL (Phase 1 §4 — Sub-Track 5)

<div class="course-identity auto-course" style="--course-accent: #0284c7; --course-accent-rgb: 2, 132, 199;" markdown="1">
<div class="course-identity__icon">OASP</div>
<div markdown="1">
<p class="course-identity__eyebrow">Deep Dive · Digital Foundations</p>
<p class="course-identity__title">Make dependencies visible before measuring portability.</p>
<p class="course-identity__meta">Artifact: working low-level demo · Measure: timing, memory, correctness</p>
</div>
</div>


**Parent:** [C++ and Parallel Computing](../Guide.md)

> *Vendor-neutral parallel compute — one programming model for CPU, GPU, and FPGA.*

**Prerequisites:** Sub-Track 3 (CUDA and SIMT). OpenCL maps directly onto CUDA concepts; SYCL then adds modern C++ on top.

**Layer mapping:** **L1** (application — you write kernels), **L3** (runtime — OpenCL/SYCL runtime, device drivers).

---

## Why Learn This

OpenCL and SYCL provide portable programming interfaces across supported implementations. They do not automatically support every accelerator: the compiler, runtime backend, driver, device, and required features must all match. A custom accelerator still needs a backend and runtime integration. Start with one working implementation, then test a second target with the same numerical contract.

| | CUDA | HIP | OpenCL | SYCL |
|---|---|---|---|---|
| **Vendor** | NVIDIA only | AMD + NVIDIA | Any (Khronos standard) | Any (Khronos standard) |
| **Language** | CUDA C++ | HIP C++ | OpenCL C (separate) | Standard C++ |
| **Kernel style** | Inline `__global__` | Inline `__global__` | Separate `.cl` source | Lambda in host code |
| **Maturity** | 18+ years | 8+ years | 15+ years | 5+ years |
| **AI ecosystem** | Dominant | Growing | Limited | Growing (Intel) |
| **Best for** | NVIDIA GPUs | AMD/NVIDIA GPUs | Portable legacy, FPGAs | Modern portable C++ |

---

## Part 1: OpenCL

### 1.1 Platform Model

OpenCL organises hardware into a hierarchy:

```
Platform (vendor driver: Intel, AMD, NVIDIA, Xilinx)
  └── Device (GPU, CPU, FPGA, accelerator)
        └── Compute Unit (CU) — SM on NVIDIA, CU on AMD, PE on FPGA
              └── Processing Element (PE) — individual ALU / thread
```

**Context** binds one or more devices together. **Command queues** send work to a specific device within a context.

```
Host (CPU)
  │
  ├── Context ─────────── Device 0 (GPU)
  │     │                     └── Command Queue 0
  │     │
  │     └──────────────── Device 1 (FPGA)
  │                           └── Command Queue 1
  │
  └── Buffers, Programs, Kernels (shared within context)
```

### 1.2 Execution Model

OpenCL's execution model maps directly onto CUDA concepts:

| OpenCL | CUDA equivalent | Meaning |
|--------|----------------|---------|
| Work-item | Thread | Single execution instance |
| Work-group | Block | Group of work-items with shared memory |
| NDRange | Grid | Global problem space (1D/2D/3D) |
| `get_global_id(0)` | `blockIdx.x * blockDim.x + threadIdx.x` | Global index |
| `get_local_id(0)` | `threadIdx.x` | Local index within work-group |
| `get_group_id(0)` | `blockIdx.x` | Work-group index |
| `barrier(CLK_LOCAL_MEM_FENCE)` | `__syncthreads()` | Work-group barrier |

### 1.3 Memory Model

```
OpenCL memory spaces:

┌─────────────────────────────────────────────┐
│ Global Memory (device DRAM / HBM)           │  ← CUDA: global memory
│   Accessible by all work-items              │
│   Slow (~300 cycles)                        │
├─────────────────────────────────────────────┤
│ Constant Memory                              │  ← CUDA: constant memory
│   Read-only, cached                          │
├─────────────────────────────────────────────┤
│ Local Memory (per work-group)               │  ← CUDA: shared memory (__shared__)
│   Fast (~4 cycles), programmer-managed      │
│   Shared within work-group                  │
├─────────────────────────────────────────────┤
│ Private Memory (per work-item)              │  ← CUDA: registers / local
│   Fastest, compiler-allocated               │
└─────────────────────────────────────────────┘
```

### 1.4 OpenCL C Kernel

Kernels are written in **OpenCL C** (a C99 dialect) and compiled at runtime from source strings.

```c
// kernel.cl — vector addition
__kernel void vector_add(
    __global const float* a,
    __global const float* b,
    __global float* c,
    const int n)
{
    int i = get_global_id(0);
    if (i < n) {
        c[i] = a[i] + b[i];
    }
}
```

**Memory qualifiers:**
- `__global` — pointer to device global memory
- `__local` — pointer to work-group shared memory
- `__constant` — pointer to constant memory
- `__private` — (default) per-work-item

### 1.5 Run a checked host-and-kernel example

Use the [CPU OpenCL lab](labs/README.md) and its complete [vector_add_opencl.cpp](labs/vector_add_opencl.cpp). It enumerates platforms/devices, prefers a CPU device, checks every API operation, prints compiler errors, selects a legal work-group size, pads the launch, validates every output, and releases resources. An OpenCL **ICD loader** dispatches API calls; it needs an installed device implementation such as PoCL to execute kernels.

```bash
# From this sub-track directory, with OpenCL headers/loader and a device runtime installed:
g++ -O2 -std=c++17 labs/vector_add_opencl.cpp -lOpenCL -o /tmp/vector_add_opencl
/tmp/vector_add_opencl --self-test
/tmp/vector_add_opencl --inject-build-error
```

The last command should fail and display `BUILD_LOG`; treating a failed build as a valid kernel is a common cause of confusing downstream errors. The normal program uses an OpenCL 1.2 API/kernel baseline and does not enable relaxed floating-point math. It was compiled and executed on PoCL 5.0's CPU implementation on 2026-10-18; GPU and SYCL execution have not been validated by these checks.

Follow one request through eight operations: device selection → context/queue creation → buffer allocation → input copies → program build → kernel argument setup → dispatch → checked output read. The queue is in order, and the output read blocks for completion. Kernel event timing excludes compilation, host allocation, transfers, and end-to-end application work.

### 1.6 Tiled Matrix Multiply with Local Memory

This is the OpenCL equivalent of CUDA shared-memory tiled GEMM:

```c
// matmul.cl — tiled matrix multiply using local memory
#define TILE 16

__kernel void matmul(
    __global const float* A,
    __global const float* B,
    __global float* C,
    const int N)
{
    int row = get_local_id(1);
    int col = get_local_id(0);
    int gRow = get_global_id(1);
    int gCol = get_global_id(0);

    __local float tileA[TILE][TILE];
    __local float tileB[TILE][TILE];

    float sum = 0.0f;

    for (int t = 0; t < N / TILE; t++) {
        // Load tiles into local memory
        tileA[row][col] = A[gRow * N + t * TILE + col];
        tileB[row][col] = B[(t * TILE + row) * N + gCol];
        barrier(CLK_LOCAL_MEM_FENCE);

        // Compute partial dot product from local memory
        for (int k = 0; k < TILE; k++)
            sum += tileA[row][k] * tileB[k][col];
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    C[gRow * N + gCol] = sum;
}
```

Launch with 2D NDRange:

```cpp
size_t global[2] = {N, N};
size_t local[2]  = {TILE, TILE};   // 16×16 = 256 work-items per group
clEnqueueNDRangeKernel(queue, kernel, 2, NULL, global, local, 0, NULL, NULL);
```

### 1.7 Event-Based Profiling

```cpp
// Create queue with profiling enabled
cl_queue_properties props[] = {CL_QUEUE_PROPERTIES, CL_QUEUE_PROFILING_ENABLE, 0};
cl_command_queue queue = clCreateCommandQueueWithProperties(ctx, device, props, NULL);

cl_event event;
clEnqueueNDRangeKernel(queue, kernel, 1, NULL,
                       &global_size, &local_size, 0, NULL, &event);
clWaitForEvents(1, &event);

cl_ulong start, end;
clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_START, sizeof(start), &start, NULL);
clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_END,   sizeof(end),   &end,   NULL);

printf("Kernel time: %.3f ms\n", (end - start) / 1e6);
```

### 1.8 Device Query

Always query what the device supports before assuming capabilities:

```cpp
char name[128];
cl_uint cu_count;
cl_ulong global_mem, local_mem;
size_t max_wg;

clGetDeviceInfo(device, CL_DEVICE_NAME,                128, name,       NULL);
clGetDeviceInfo(device, CL_DEVICE_MAX_COMPUTE_UNITS,     4, &cu_count,  NULL);
clGetDeviceInfo(device, CL_DEVICE_GLOBAL_MEM_SIZE,       8, &global_mem,NULL);
clGetDeviceInfo(device, CL_DEVICE_LOCAL_MEM_SIZE,        8, &local_mem, NULL);
clGetDeviceInfo(device, CL_DEVICE_MAX_WORK_GROUP_SIZE,sizeof(max_wg), &max_wg, NULL);

printf("Device: %s\n", name);
printf("CUs: %u, Global mem: %lu MB, Local mem: %lu KB, Max WG: %zu\n",
       cu_count, global_mem / (1024*1024), local_mem / 1024, max_wg);
```

---

## Part 2: SYCL

### 2.1 What SYCL Is

SYCL is a **modern C++ abstraction** built on top of OpenCL (and now other backends). It eliminates OpenCL's boilerplate by expressing kernels as C++ lambdas and managing buffers with RAII accessors.

```
OpenCL:
  Platform → Device → Context → Queue → Buffer → Program → Kernel → SetArgs → Launch
  (~50 lines of setup)

SYCL:
  queue q;
  q.submit([&](handler& h) { h.parallel_for(..., [=](id<1> i) { ... }); });
  (~5 lines)
```

**SYCL implementations:**

| Implementation | Vendor | Backend targets |
|---------------|--------|-----------------|
| **Intel oneAPI DPC++** | Intel | Supported Intel targets; other GPU targets need compatible backend plugins |
| **AdaptiveCpp** (formerly hipSYCL) | Open-source | AMD GPUs, NVIDIA GPUs, CPUs |
| **Altera FPGA flow** | Altera | Separate FPGA toolchain/add-on and supported board package; see section 2.6 |

### 2.2 SYCL Hello World — Vector Addition

```cpp
#include <sycl/sycl.hpp>
#include <vector>
#include <iostream>

int main() {
    const int N = 1024;
    std::vector<float> a(N), b(N), c(N);

    for (int i = 0; i < N; i++) { a[i] = i; b[i] = i * 2; }

    {
        // Create buffers that wrap host vectors
        sycl::buffer<float> buf_a(a.data(), N);
        sycl::buffer<float> buf_b(b.data(), N);
        sycl::buffer<float> buf_c(c.data(), N);

        // Select an available device using the implementation's default policy.
        // gpu_selector_v specifically requires a GPU; it does not fall back to CPU.
        sycl::queue q(sycl::default_selector_v);
        std::cout << "Device: "
                  << q.get_device().get_info<sycl::info::device::name>()
                  << "\n";

        q.submit([&](sycl::handler& h) {
            // Accessors: read from a,b; write to c
            auto A = buf_a.get_access<sycl::access::mode::read>(h);
            auto B = buf_b.get_access<sycl::access::mode::read>(h);
            auto C = buf_c.get_access<sycl::access::mode::write>(h);

            h.parallel_for(sycl::range<1>(N), [=](sycl::id<1> i) {
                C[i] = A[i] + B[i];
            });
        });
        q.wait_and_throw();  // surface asynchronous execution errors
    }  // buffer destructor → implicit copy back to host vectors

    for (int i = 0; i < N; ++i) if (c[i] != a[i] + b[i]) return 1;
    std::cout << "c[0] = " << c[0] << ", c[" << N-1 << "] = " << c[N-1] << "\n";
}
```

```bash
# Compile with Intel DPC++
icpx -fsycl -o vecadd vecadd.cpp

# Or with AdaptiveCpp (targeting NVIDIA)
acpp -o vecadd vecadd.cpp --acpp-targets="cuda:sm_80"

# Run
./vecadd
```

**Compare to OpenCL:** SYCL's buffer/accessor model handles data movement automatically. No explicit `clCreateBuffer`, `clEnqueueWriteBuffer`, `clSetKernelArg` — the SYCL runtime infers dependencies and schedules transfers.

### 2.3 USM (Unified Shared Memory) — CUDA-Like Pointers in SYCL

The buffer/accessor model expresses data dependencies; correctness still depends on bounds and lifetimes. USM provides raw pointers, so you must express dependencies explicitly. The following ordering example requires device-USM support; a full application must check allocation failures and release already allocated memory on exceptions:

```cpp
#include <sycl/sycl.hpp>
#include <vector>
#include <stdexcept>

int main() {
    sycl::queue q(sycl::default_selector_v);
    if (!q.get_device().has(sycl::aspect::usm_device_allocations))
        throw std::runtime_error("device USM is unsupported");
    const int N = 1024;

    // Allocate device memory (like cudaMalloc)
    float* d_a = sycl::malloc_device<float>(N, q);
    float* d_b = sycl::malloc_device<float>(N, q);
    float* d_c = sycl::malloc_device<float>(N, q);

    // Host data
    std::vector<float> h_a(N, 1.0f), h_b(N, 2.0f), h_c(N);

    // Copy to device (like cudaMemcpy)
    q.memcpy(d_a, h_a.data(), N * sizeof(float));
    q.memcpy(d_b, h_b.data(), N * sizeof(float));
    q.wait_and_throw();

    // Launch kernel
    q.parallel_for(sycl::range<1>(N), [=](sycl::id<1> i) {
        d_c[i] = d_a[i] + d_b[i];
    }).wait_and_throw();

    // Copy back
    q.memcpy(h_c.data(), d_c, N * sizeof(float)).wait_and_throw();

    // Free
    sycl::free(d_a, q);
    sycl::free(d_b, q);
    sycl::free(d_c, q);
}
```

**USM allocation types:**

| Type | Host access | Device access | Data movement |
|------|-----------|--------------|---------------|
| `malloc_device` | No | Yes | Explicit `memcpy` |
| `malloc_host` | Yes | If the device supports host USM | Access path depends on shared/discrete memory architecture |
| `malloc_shared` | Yes | If the device supports shared USM | Runtime/platform managed; do not assume free concurrent access |

Check `sycl::aspect::usm_device_allocations`, `usm_host_allocations`, or `usm_shared_allocations` before selecting a memory model. Check allocation failure and surface asynchronous errors with `wait_and_throw()` in a complete application. USM pointers do not make an out-of-order queue infer dependencies: retain events or explicitly order dependent operations. See [Khronos USM concepts](https://github.khronos.org/SYCL_Reference/iface/usm_basic_concept.html).

### 2.4 Local Memory and Work-Group Tiling

SYCL exposes local memory through `local_accessor`:

```cpp
q.submit([&](sycl::handler& h) {
    auto A = buf_a.get_access<sycl::access::mode::read>(h);
    auto B = buf_b.get_access<sycl::access::mode::read>(h);
    auto C = buf_c.get_access<sycl::access::mode::write>(h);

    constexpr int TILE = 16;
    sycl::local_accessor<float, 1> tileA(sycl::range<1>(TILE * TILE), h);
    sycl::local_accessor<float, 1> tileB(sycl::range<1>(TILE * TILE), h);

    h.parallel_for(
        sycl::nd_range<2>({N, N}, {TILE, TILE}),
        [=](sycl::nd_item<2> item) {
            int row = item.get_local_id(1);
            int col = item.get_local_id(0);
            int gRow = item.get_global_id(1);
            int gCol = item.get_global_id(0);

            float sum = 0.0f;
            for (int t = 0; t < N / TILE; t++) {
                tileA[row * TILE + col] = A[gRow * N + t * TILE + col];
                tileB[row * TILE + col] = B[(t * TILE + row) * N + gCol];
                item.barrier(sycl::access::fence_space::local_space);

                for (int k = 0; k < TILE; k++)
                    sum += tileA[row * TILE + k] * tileB[k * TILE + col];
                item.barrier(sycl::access::fence_space::local_space);
            }
            C[gRow * N + gCol] = sum;
        }
    );
});
```

### 2.5 Sub-groups: use the actual execution group

A SYCL subgroup is implementation- and kernel-dependent; do not infer its size solely from the GPU vendor. Use group algorithms rather than assuming a power-of-two width and reading an out-of-range shuffle source. The [Khronos group-algorithm reference](https://github.khronos.org/SYCL_Reference/iface/group-algorithms-library.html) specifies the collective participation requirements.

```cpp
// Sketch preconditions: N > 0; device supports local size 128;
// data points to N accessible floats; partial points to global accessible floats
// initialized to zero. Both allocations remain alive until completion.
constexpr size_t local = 128;
const size_t global = ((N + local - 1) / local) * local;
q.parallel_for(sycl::nd_range<1>(global, local), [=](sycl::nd_item<1> item) {
    auto sg = item.get_sub_group();
    size_t i = item.get_global_linear_id();
    float value = i < N ? data[i] : 0.0f;  // no early return before the collective
    float sum = sycl::reduce_over_group(sg, value, sycl::plus<float>{});
    if (sg.leader()) {
        // Reserve local slots per work-group: subgroup IDs are local to a group.
        size_t slot = item.get_group_linear_id() * local + sg.get_group_linear_id();
        partial[slot] = sum;
    }
}).wait_and_throw();
```

Each subgroup writes a unique partial result; untouched slots remain zero. A second reduction combines the partials. The deliberately roomy output avoids collisions between different work-groups without fixing subgroup width. Production code can use a denser layout or a work-group/library reduction. Floating-point addition can regroup, so compare with a justified tolerance. This sketch has not been executed in the CPU OpenCL checks.

### 2.6 SYCL for FPGA: select the maintained FPGA toolchain

**Reviewed 2026-10-18:** Intel removed FPGA support from its DPC++ compiler starting with release 2025.1; the old 2025.0 flow is a historical baseline. Use the separately documented [Altera FPGA add-on/toolchain](https://www.altera.com/products/development-tools/oneAPI-base-toolkit) and its supported board package for new FPGA work. See [Intel's transition notice](https://www.intel.com/content/www/us/en/developer/tools/oneapi/fpga.html). Do not assume that installing a current generic `icpx` supplies FPGA compilation.

The following pipe example describes the programming idea and needs the selected FPGA toolchain's headers, device selector, and interface configuration. GPU source portability does not imply a suitable FPGA architecture.

```cpp
// FPGA-optimised: use pipes for streaming
using PipeA = sycl::ext::intel::pipe<class pA, float, 16>;  // depth 16

// Producer kernel
q.submit([&](sycl::handler& h) {
    h.single_task([=]() {
        for (int i = 0; i < N; i++)
            PipeA::write(input[i]);
    });
});

// Consumer kernel (runs concurrently on FPGA fabric)
q.submit([&](sycl::handler& h) {
    h.single_task([=]() {
        for (int i = 0; i < N; i++)
            output[i] = PipeA::read() * 2.0f;
    });
});
```

Hardware compilation includes synthesis and physical implementation and can be expensive. Use emulation for functional iteration, then measure the implemented FPGA separately. These are **historical 2025.0-style command examples**, not commands validated with the current Altera flow:

```bash
# Emulation (fast, runs on CPU)
icpx -fsycl -fintelfpga -DFPGA_EMULATOR -o emu emu.cpp

# FPGA hardware compile (duration depends on design, machine, and tool version)
icpx -fsycl -fintelfpga -Xshardware -o hw hw.cpp
```

---

## Part 3: OpenCL vs SYCL Decision Guide

| Question | Evidence to collect |
|---|---|
| Which devices must run the program? | Exact vendor/device/runtime or compiler-backend support |
| Which language features are required? | Device capabilities, supported extensions, compiler revision |
| Is FPGA hardware compilation required? | Vendor toolchain, board package, supported API and license |
| How is the program deployed? | Ahead-of-time or runtime compilation, driver and library requirements |
| Does it meet the workload need? | Numerical validation and end-to-end timing on each intended target |

OpenCL is not simply in “maintenance mode”: the Khronos registry lists **OpenCL3.1**, published 2026-09-17, as checked 2026-10-18. The CPU lab intentionally uses an OpenCL1.2 baseline on a tested PoCL5.0 runtime; a newer specification does not make that runtime expose every newer capability. SYCL2020 is a separate C++ programming model with implementation-specific backend support. [OpenCL registry](https://registry.khronos.org/OpenCL/), [SYCL2020 specification](https://registry.khronos.org/SYCL/specs/sycl-2020/html/sycl-2020.html).

**Choose from evidence:** list target devices, required features, supported toolchains, deployment constraints, and available libraries. Run one correctness test and one end-to-end measurement on each candidate backend. A portable API does not guarantee portable performance; a vendor-specific API is not automatically faster for every workload.

---

## Resources

| Resource | Type | Focus |
|----------|------|-------|
| Khronos OpenCL registry (current and archived specifications) | Standard | Authoritative API reference |
| Khronos SYCL 2020 specification | Standard | Authoritative SYCL reference |
| Intel oneAPI DPC++ documentation | Docs | SYCL implementation; FPGA uses the separate supported Altera flow |
| AdaptiveCpp (github.com/AdaptiveCpp) | Open-source | SYCL on AMD/NVIDIA/CPU |
| *OpenCL Programming Guide* — Munshi et al. | Textbook | Comprehensive OpenCL |
| *Data Parallel C++* — Reinders, Ashbaugh, Brodman | Textbook | SYCL/DPC++ (free PDF from Intel) |
| Codeplay developer blog | Blog | SYCL tutorials, GPU porting guides |

---

## Projects

| # | Project | Concepts practiced | Complexity |
|---|---------|-------------------|------------|
| 1 | **OpenCL vector add** | Platform/device/context/queue setup, kernel compile | Beginner |
| 2 | **SYCL vector add (buffer + USM)** | Buffer/accessor vs USM, device selection | Beginner |
| 3 | **OpenCL tiled matmul** | Local memory, barriers, 2D NDRange | Intermediate |
| 4 | **SYCL tiled matmul** | local_accessor, nd_range, nd_item | Intermediate |
| 5 | **Cross-device benchmark** | Run same kernel on CPU + GPU, compare throughput | Intermediate |
| 6 | **SYCL sub-group reduction** | Warp/wavefront-portable primitives | Intermediate |
| 7 | **Port CUDA kernel to SYCL** | Compare CUDA vs SYCL for same algorithm | Intermediate |
| 8 | **OpenCL device query tool** | Enumerate all platforms/devices, print capabilities | Beginner |
| 9 | **SYCL FPGA pipe pipeline** | Intel FPGA, pipe, emulation, hardware compile | Advanced |
| 10 | **Multi-device task graph** | SYCL event dependencies, split work across GPU+CPU | Advanced |

---

## Next

→ Back to [C++ and Parallel Computing hub](../Guide.md)

→ [Phase 3 — Neural Networks](../../../Phase%203%20-%20Artificial%20Intelligence/1.%20Neural%20Networks/Guide.md)

## Make dependencies visible before measuring portability

Implement a two-stage pipeline with explicit USM events, then a buffer/accessor version. Draw the copy → kernel → copy dependency graph. Test an out-of-order queue where supported; omitting a dependency should be identified as a correctness bug, not treated as a performance improvement. Query required aspects and reject unsupported configurations clearly.

Compare identical shapes and tolerance on available devices. Distinguish source portability, functionality, and performance portability in the report. For a future custom NPU, specify the compiler/runtime work needed to implement its backend instead of assuming a standards-based source language creates hardware support automatically.

**Carry forward:** Dependency graph, capability checks, and a portable operator contract. Continue with [the next course](../../../Phase%204%20-%20Track%20C%20-%20DL%20Inference%20Optimization/03%20-%20Compiler%20Stack/Guide.md) and reuse that evidence.
