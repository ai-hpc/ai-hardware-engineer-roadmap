# Orin Nano 8GB -- CUDA Programming Deep Dive

<div class="course-identity auto-course" style="--course-accent: #4f46e5; --course-accent-rgb: 79, 70, 229;" markdown="1">
<div class="course-identity__icon">ON8C</div>
<div markdown="1">
<p class="course-identity__eyebrow">Deep Dive · Jetson Track</p>
<p class="course-identity__title">Specialized course identity for Orin Nano 8GB -- CUDA Programming Deep Dive.</p>
<p class="course-identity__meta">Artifact: Jetson integration demo · Measure: latency, memory, power, logs</p>
</div>
</div>


> **Scope:** Production-level CUDA programming on Jetson Orin Nano 8GB (T234 SoC) -- toolchain, kernel authoring, memory optimization, profiling, camera zero-copy, TensorRT integration, and production patterns.
>
> **Prerequisites:** Familiarity with [Orin Nano memory architecture](../Orin-Nano-Memory-Architecture/Guide.md) (unified memory, CMA, SMMU) and C/C++. JetPack 6.x installed.

---


## 1. CUDA on Jetson vs Discrete GPU

The critical mental-model shift: **there is no PCIe bus between CPU and GPU**. They share the same physical LPDDR5.

| Property | Discrete GPU (e.g. A100) | Jetson Orin Nano 8GB |
|---|---|---|
| Memory | Separate VRAM (HBM/GDDR) | Shared 8GB LPDDR5 |
| CPU-GPU transfer | PCIe/NVLink DMA | No transfer -- same DRAM |
| `cudaMemcpy` H2D/D2H | Real DMA copy | memcpy within DRAM (no bus crossing) |
| Memory bandwidth | 2 TB/s (A100 HBM) | 68 GB/s shared CPU + GPU + DLA + ISP |
| `cudaMallocManaged` | Page migration over PCIe | Pages already in unified pool |
| Power envelope | 300-700 W system | 7-15 W module |
| SM count | 108 (A100) | 8 SMs (1024 CUDA cores) |

Practical consequences: prefer `cudaMallocManaged` or zero-copy over explicit `cudaMemcpy`. Bandwidth is the bottleneck -- 68 GB/s is shared across all engines. Kernel launch overhead is proportionally larger; fuse aggressively.

---

## 2. Jetson CUDA Toolchain

### 2.1 Verification and Compilation

```bash
nvcc --version                    # JetPack 6.x ships CUDA 12.2+
nvcc -arch=sm_87 -O2 -o my_kernel my_kernel.cu   # sm_87 = Orin Nano

# Key flags:
#   -arch=sm_87          Ampere GA10B compute capability 8.7
#   --ptxas-options=-v   Show register/shared memory usage per kernel
#   -use_fast_math       Fast but less precise (use with care)
```

### 2.2 Cross-Compilation from x86 Host

```bash
/usr/local/cuda-12.2/bin/nvcc -arch=sm_87 \
    --compiler-bindir=/usr/bin/aarch64-linux-gnu-g++ \
    -O2 -o my_kernel my_kernel.cu
```

### 2.3 CMakeLists.txt Template

```cmake
cmake_minimum_required(VERSION 3.18)
project(jetson_cuda LANGUAGES CXX CUDA)
set(CMAKE_CUDA_ARCHITECTURES 87)
set(CMAKE_CUDA_STANDARD 17)
add_executable(my_kernel src/my_kernel.cu)
target_link_libraries(my_kernel PRIVATE cuda cudart)
```

---

## 3. Ampere Architecture on Orin Nano

Cut-down GA10B (Ampere), compute capability **sm_87**.

| Resource | Value | Resource | Value |
|---|---|---|---|
| SMs | 8 | Tensor Cores/SM | 4 (3rd-gen) |
| CUDA cores/SM | 128 (4x32 FP32) | Max threads/SM | 1536 |
| Total CUDA cores | 1024 | Warp size | 32 |
| Registers/SM | 65536 (32-bit) | Shared mem/SM | Up to 164 KB |
| L2 cache | 512 KB | Max clock | ~625 MHz |

Thread hierarchy: Grid -> Block (up to 1024 threads) -> Warp (32 threads) -> Thread. Each SM has 4 sub-partitions, each with its own warp scheduler, 32 FP32 ALUs, and 1 Tensor Core. Shared resources: L1/shared memory (164 KB configurable) and register file (256 KB).

**Register pressure:** 65536 regs / 1536 max threads = 42 regs/thread before occupancy drops. Check with `nvcc --ptxas-options=-v -arch=sm_87 kernel.cu`.

---

## 4. Tensor Cores

Fixed-function **D = A * B + C** units. FP16: 16x8x16 tile, 256 FMAs/TC/cycle. INT8: 16x8x32, 512 MACs. TF32: 16x8x8, 128 FMAs.

### 4.1 WMMA Example (FP16 GEMM)

```cuda
#include <mma.h>
using namespace nvcuda::wmma;

__global__ void tc_gemm(half *A, half *B, float *C, int M, int N, int K) {
    fragment<matrix_a, 16, 16, 16, half, row_major> a_frag;
    fragment<matrix_b, 16, 16, 16, half, col_major> b_frag;
    fragment<accumulator, 16, 16, 16, float>         c_frag;
    fill_fragment(c_frag, 0.0f);
    int warpM = (blockIdx.x * blockDim.x + threadIdx.x) / 32 * 16;
    int warpN = blockIdx.y * 16;
    for (int k = 0; k < K; k += 16) {
        load_matrix_sync(a_frag, A + warpM*K + k, K);
        load_matrix_sync(b_frag, B + k*N + warpN, N);
        mma_sync(c_frag, a_frag, b_frag, c_frag);
    }
    store_matrix_sync(C + warpM*N + warpN, c_frag, N, mem_row_major);
}
```

When to use: dense FP16/INT8 matrix ops not covered by cuBLAS/TensorRT (which route to TCs automatically via `cublasGemmEx` with `CUBLAS_COMPUTE_16F`).

---

## 5. Unified Memory Deep Dive

| API | Best Use on Jetson |
|---|---|
| `cudaMalloc` | GPU-only buffers, CUDA graphs |
| `cudaMallocHost` | CPU+GPU concurrent access, DMA targets |
| `cudaMallocManaged` | Default -- pages mapped in both CPU/GPU page tables, no copy |
| `NvBufSurface` | Camera/video zero-copy (DMA-BUF backed) |

On Jetson, `cudaMallocManaged` pages never physically migrate -- same LPDDR5 page in both address spaces:

```cuda
float *data;
cudaMallocManaged(&data, N * sizeof(float));
for (int i = 0; i < N; i++) data[i] = i * 0.1f;  // CPU writes
kernel<<<blocks, threads>>>(data, N);              // GPU reads -- same DRAM, no copy
cudaDeviceSynchronize();
printf("result[0] = %f\n", data[0]);               // CPU reads -- no copy back
cudaFree(data);
```

---

## 6. Memory Access Patterns

### 6.1 Coalesced Global Memory Access

A warp accessing consecutive 4-byte elements issues one 128-byte transaction. Strided access causes 10-30x slowdown -- critical when 68 GB/s is shared.

```cuda
// GOOD: coalesced -- threads read consecutive addresses
int i = blockIdx.x * blockDim.x + threadIdx.x;
out[i] = in[i] * 2.0f;
// BAD: strided -- each thread skips 'stride' elements
int i = (blockIdx.x * blockDim.x + threadIdx.x) * stride;
```

### 6.2 Shared Memory Bank Conflicts

32 banks, 4 bytes each. Two threads in a warp accessing the same bank (different addresses) serialize. Same address broadcasts for free.

### 6.3 L2 Cache Residency Control

Orin Nano L2 is 512 KB. Keep working sets under this. Use `cudaAccessPropertyPersisting` hints:

```cuda
cudaStreamAttrValue attr;
attr.accessPolicyWindow.base_ptr = (void *)data;
attr.accessPolicyWindow.num_bytes = 256 * 1024;
attr.accessPolicyWindow.hitRatio = 1.0f;
attr.accessPolicyWindow.hitProp = cudaAccessPropertyPersisting;
attr.accessPolicyWindow.missProp = cudaAccessPropertyStreaming;
cudaStreamSetAttribute(stream, cudaStreamAttributeAccessPolicyWindow, &attr);
```

---

## 7. Writing Your First CUDA Kernel on Jetson

### 7.1 Vector Addition (Complete Example)

```cuda
// vecadd.cu -- compile: nvcc -arch=sm_87 -O2 -o vecadd vecadd.cu
#include <cstdio>
#include <cuda_runtime.h>

__global__ void vec_add(const float *a, const float *b, float *c, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) c[i] = a[i] + b[i];
}
int main() {
    const int N = 1 << 20;
    float *a, *b, *c;
    cudaMallocManaged(&a, N*sizeof(float));
    cudaMallocManaged(&b, N*sizeof(float));
    cudaMallocManaged(&c, N*sizeof(float));
    for (int i = 0; i < N; i++) { a[i] = 1.0f; b[i] = 2.0f; }
    vec_add<<<(N+255)/256, 256>>>(a, b, c, N);
    cudaDeviceSynchronize();
    printf("c[0] = %f (expected 3.0)\n", c[0]);
    cudaFree(a); cudaFree(b); cudaFree(c);
}
```

### 7.2 Naive Matrix Multiply

```cuda
__global__ void matmul(const float *A, const float *B, float *C, int M, int N, int K) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < M && col < N) {
        float sum = 0.0f;
        for (int k = 0; k < K; k++) sum += A[row*K + k] * B[k*N + col];
        C[row*N + col] = sum;
    }
}  // Launch: dim3 block(16,16); dim3 grid((N+15)/16, (M+15)/16);
```

---

## 8. Convolution Kernel

### 8.1 Naive 2D Convolution

```cuda
__global__ void conv2d_naive(const float *in, const float *kern,
                             float *out, int H, int W, int KH, int KW) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int hKH = KH / 2, hKW = KW / 2;
    if (row < H && col < W) {
        float sum = 0.0f;
        for (int kh = 0; kh < KH; kh++)
            for (int kw = 0; kw < KW; kw++) {
                int r = row + kh - hKH, c = col + kw - hKW;
                if (r >= 0 && r < H && c >= 0 && c < W)
                    sum += in[r * W + c] * kern[kh * KW + kw];
            }
        out[row * W + col] = sum;
    }
}
```

### 8.2 Tiled with Shared Memory (3x3)

```cuda
#define TILE 16
#define RAD  1

__global__ void conv2d_tiled(const float *in, const float *kern, float *out, int H, int W) {
    __shared__ float tile[TILE + 2*RAD][TILE + 2*RAD];
    int tx = threadIdx.x, ty = threadIdx.y;
    int col = blockIdx.x * TILE + tx, row = blockIdx.y * TILE + ty;
    int sc = tx + RAD, sr = ty + RAD;

    tile[sr][sc] = (row < H && col < W) ? in[row * W + col] : 0.0f;
    // Load halo edges (left, right, top, bottom -- boundary threads only)
    if (tx < RAD)
        tile[sr][tx] = (row < H && col-RAD >= 0) ? in[row*W + col-RAD] : 0.0f;
    if (tx >= TILE - RAD)
        tile[sr][sc+RAD] = (row < H && col+RAD < W) ? in[row*W + col+RAD] : 0.0f;
    // (top/bottom halo similar)
    __syncthreads();

    if (row < H && col < W) {
        float sum = 0.0f;
        for (int kr = -RAD; kr <= RAD; kr++)
            for (int kc = -RAD; kc <= RAD; kc++)
                sum += tile[sr+kr][sc+kc] * kern[(kr+RAD)*3 + (kc+RAD)];
        out[row * W + col] = sum;
    }
}
```

Tiled version reduces global reads from H\*W\*K\*K to ~H\*W\*(1 + halo) -- 5-9x improvement for 3x3-7x7.

---

## 9. CUDA Streams and Concurrency

Streams enable concurrent kernel execution. On Orin Nano (8 SMs), concurrency helps when individual kernels are small.

```cuda
cudaStream_t s1, s2;
cudaStreamCreate(&s1); cudaStreamCreate(&s2);
kernelA<<<grid, block, 0, s1>>>(dataA);  // may run concurrently
kernelB<<<grid, block, 0, s2>>>(dataB);
cudaStreamSynchronize(s1); cudaStreamSynchronize(s2);
cudaStreamDestroy(s1); cudaStreamDestroy(s2);
```

### 9.1 Multi-Stream Pipeline

Process frame N on GPU while CPU prepares N+1. Double-buffer with 2 streams:

```cuda
cudaStream_t st[2];
for (int i = 0; i < 2; i++) cudaStreamCreate(&st[i]);
for (int f = 0; f < total; f++) {
    int s = f % 2;
    preprocess<<<g, b, 0, st[s]>>>(in[s], pre[s]);
    inference<<<g, b, 0, st[s]>>>(pre[s], res[s]);
    postprocess<<<g, b, 0, st[s]>>>(res[s], out[s]);
}
for (int i = 0; i < 2; i++) { cudaStreamSynchronize(st[i]); cudaStreamDestroy(st[i]); }
```

---

## 10. Synchronization Primitives

### 10.1 Events (Timing)

```cuda
cudaEvent_t start, stop;
cudaEventCreate(&start); cudaEventCreate(&stop);
cudaEventRecord(start, stream);
my_kernel<<<grid, block, 0, stream>>>(data);
cudaEventRecord(stop, stream);
cudaEventSynchronize(stop);
float ms; cudaEventElapsedTime(&ms, start, stop);
```

### 10.2 Inter-Stream Dependencies

```cuda
cudaEventRecord(ev, streamA);          // record after kernelA
cudaStreamWaitEvent(streamB, ev, 0);   // streamB blocks until ev fires
kernelB<<<g, b, 0, streamB>>>(dataB);  // guaranteed after kernelA
```

### 10.3 Sync API

`cudaDeviceSynchronize()` blocks on all streams. `cudaStreamSynchronize(s)` blocks on one. `cudaEventQuery(e)` / `cudaStreamQuery(s)` poll without blocking.

---

## 11. CUDA + Camera Zero-Copy

Camera frames flow ISP -> NvBufSurface (DMA-BUF). Import directly into CUDA -- no copy.

```
Sensor -> VI -> ISP -> NvBufSurface (DMA-BUF fd)
                            |
                 EGLImageKHR (EGL interop)
                            |
                 cudaGraphicsResource (cuGraphicsEGLRegisterImage)
                            |
                 cudaArray / cudaSurfaceObject -> CUDA Kernel
```

```cuda
#include <cudaEGL.h>
void process_frame(EGLImageKHR egl_image) {
    cudaGraphicsResource_t res;
    cudaGraphicsEGLRegisterImage(&res, egl_image, cudaGraphicsRegisterFlagsReadOnly);
    cudaArray_t arr;
    cudaGraphicsSubResourceGetMappedArray(&arr, res, 0, 0);
    cudaResourceDesc rd = {}; rd.resType = cudaResourceTypeArray; rd.res.array.array = arr;
    cudaSurfaceObject_t surf; cudaCreateSurfaceObject(&surf, &rd);
    process_kernel<<<grid, block>>>(surf, width, height);
    cudaDeviceSynchronize();
    cudaDestroySurfaceObject(surf);
    cudaGraphicsUnregisterResource(res);
}
```

Higher-level: `NvBufSurfaceMapEglImage(surface, 0)` then `surface->surfaceList[0].mappedAddr.eglImage`. True zero-copy -- ISP-written LPDDR5 pages read directly by CUDA kernel.

---

## 12. CUDA + TensorRT Integration

### 12.1 Custom Plugin with CUDA Kernel

```cpp
class NormalizePlugin : public nvinfer1::IPluginV2DynamicExt {
    int enqueue(const nvinfer1::PluginTensorDesc *inDesc, /*...*/) override {
        int n = inDesc[0].dims.d[0] * inDesc[0].dims.d[1] * inDesc[0].dims.d[2] * inDesc[0].dims.d[3];
        normalize_kernel<<<(n+255)/256, 256, 0, stream>>>(
            (const half*)inputs[0], (half*)outputs[0], n, mean_, std_);
        return 0;
    }
};
```

### 12.2 Fused Pre-Processing Kernel

Resize + normalize + HWC-to-CHW in a single kernel avoids 3 separate passes:

```cuda
__global__ void preprocess_fused(const uint8_t *src, float *dst,
                                  int srcH, int srcW, int dstH, int dstW, int C) {
    int dx = blockIdx.x * blockDim.x + threadIdx.x;
    int dy = blockIdx.y * blockDim.y + threadIdx.y;
    if (dx >= dstW || dy >= dstH) return;
    // bilinear sample -> normalize -> write CHW
    for (int c = 0; c < C; c++)
        dst[c*dstH*dstW + dy*dstW + dx] = (pixel[c]/255.0f - mean[c]) / std_dev[c];
}
```

Full pipeline: Camera (zero-copy) -> CUDA preprocess (fused) -> TensorRT `engine.execute` -> CUDA postprocess (NMS/decode) -> Output.

---

## 13. Nsight Systems Profiling

```bash
nsys profile -o trace --trace=cuda,nvtx,osrt --gpu-metrics-device=0 ./my_app
# Remote: nsys profile --target=ssh://user@jetson_ip --trace=cuda,nvtx -o trace /path/to/app
```

Transfer `.nsys-rep` to host for GUI analysis. Look for: gaps between kernels (CPU bottleneck), missing stream overlap, unnecessary `cudaMemcpy`. Annotate with NVTX:

```cuda
#include <nvtx3/nvToolsExt.h>
nvtxRangePush("Preprocessing");
preprocess<<<g, b, 0, stream>>>(data);
nvtxRangePop();
```

---

## 14. Nsight Compute

```bash
ncu --set full --kernel-name "my_kernel" --launch-skip 2 --launch-count 3 ./my_app
```

| Metric | Target | Red Flag |
|---|---|---|
| Achieved Occupancy | >50% | <25% -- register/smem pressure |
| Memory Throughput | Near 68 GB/s | Well below -- check coalescing |
| SM Throughput | >60% | <30% -- kernel too small or memory-bound |

Find optimal block size programmatically: `cudaOccupancyMaxPotentialBlockSize(&minGrid, &blockSz, my_kernel, 0, 0);`

---

## 15. Common Optimization Techniques

**Occupancy tuning:** Use `__launch_bounds__(256, 4)` to hint max threads and min blocks/SM. Reduces register allocation, increases concurrent warps.

**Warp divergence:** Avoid branching within a warp (`threadIdx.x % 2`). Restructure so divergence aligns to warp boundaries (multiples of 32).

**ILP (Instruction-Level Parallelism):** Each thread processes multiple elements so independent operations pipeline:

```cuda
__global__ void ilp4(float *out, const float *in, int N) {
    int base = (blockIdx.x * blockDim.x + threadIdx.x) * 4;
    float a=in[base], b=in[base+1], c=in[base+2], d=in[base+3];
    out[base]=a*a; out[base+1]=b*b; out[base+2]=c*c; out[base+3]=d*d;
}
```

**Kernel fusion:** Launch overhead (~5-10 us) is proportionally expensive on Jetson. Fuse sequential kernels:

```cuda
// BEFORE: normalize<<<g,b>>>(d); activate<<<g,b>>>(d); scale<<<g,b>>>(d);
// AFTER:
__global__ void fused(float *data, float mean, float std, float s) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    data[i] = fmaxf((data[i] - mean) / std, 0.0f) * s;
}
```

---

## 16. Power-Aware CUDA Programming

Power modes: **15W** (~625 MHz GPU), **7W** (~306 MHz GPU). Both keep all 8 SMs active.

```bash
sudo nvpmodel -m 0 && sudo jetson_clocks     # max performance
tegrastats --interval 1000                    # monitor power
# Lock GPU clock for latency-sensitive work:
sudo sh -c 'echo 625500000 > /sys/devices/17000000.gpu/devfreq/17000000.gpu/min_freq'
```

Maximizing TOPS/watt: use INT8/FP16 (Tensor Core sweet spot), minimize idle GPU time, batch small kernels, offload conv layers to DLA (2-5x better TOPS/watt -- see [DLA Deep Dive](../Orin-Nano-DLA-Deep-Dive/Guide.md)). The GPU governor (`nvhost_podgov`) scales clocks on load; bursty workloads may run at lower clocks during ramp-up.

---

## 17. Debugging CUDA on Jetson

### 17.1 compute-sanitizer

```bash
compute-sanitizer --tool memcheck  ./my_app   # OOB/misaligned
compute-sanitizer --tool racecheck ./my_app   # race conditions (slow)
compute-sanitizer --tool initcheck ./my_app   # uninitialized reads
```

### 17.2 Common Pitfalls

| Pitfall | Symptom | Fix |
|---|---|---|
| Missing `cudaDeviceSynchronize()` | Stale GPU results on CPU | Sync before CPU reads |
| Silent launch failure | Wrong results, no error | `cudaGetLastError()` after every launch |
| Shared memory overflow | `cudaErrorLaunchFailure` | Reduce smem or block size |
| OOM on 8GB shared pool | `cudaErrorMemoryAllocation` | Monitor with `tegrastats`; reduce batch |
| Wrong arch (sm_50 on sm_87) | `cudaErrorInvalidDeviceFunction` | Compile with `-arch=sm_87` |

### 17.3 Error-Checking Macro

```cuda
#define CUDA_CHECK(call) do {                                       \
    cudaError_t err = call;                                         \
    if (err != cudaSuccess) {                                       \
        fprintf(stderr, "CUDA error %s:%d: %s\n",                  \
                __FILE__, __LINE__, cudaGetErrorString(err));        \
        exit(EXIT_FAILURE); }                                       \
} while (0)

// Usage: check alloc, launch, and sync
CUDA_CHECK(cudaMallocManaged(&data, N * sizeof(float)));
my_kernel<<<grid, block>>>(data);
CUDA_CHECK(cudaGetLastError());          // launch config errors
CUDA_CHECK(cudaDeviceSynchronize());     // execution errors
```

---

## 18. Production CUDA Patterns

### 18.1 RAII Resource Management

```cpp
struct CudaBuffer {
    void *ptr = nullptr; size_t size = 0;
    CudaBuffer(size_t bytes) : size(bytes) { CUDA_CHECK(cudaMallocManaged(&ptr, bytes)); }
    ~CudaBuffer() { if (ptr) cudaFree(ptr); }
    CudaBuffer(const CudaBuffer&) = delete;
    CudaBuffer& operator=(const CudaBuffer&) = delete;
    CudaBuffer(CudaBuffer&& o) noexcept : ptr(o.ptr), size(o.size) { o.ptr = nullptr; }
};
```

### 18.2 Async Pipeline with Error Propagation

```cpp
class CudaPipeline {
    cudaStream_t stream_; cudaEvent_t done_;
public:
    CudaPipeline() {
        CUDA_CHECK(cudaStreamCreateWithFlags(&stream_, cudaStreamNonBlocking));
        CUDA_CHECK(cudaEventCreateWithFlags(&done_, cudaEventDisableTiming));
    }
    ~CudaPipeline() { cudaStreamDestroy(stream_); cudaEventDestroy(done_); }
    void submit(const float *in, float *out, int N) {
        my_kernel<<<(N+255)/256, 256, 0, stream_>>>(in, out, N);
        CUDA_CHECK(cudaGetLastError());
        cudaEventRecord(done_, stream_);
    }
    bool is_complete() { return cudaEventQuery(done_) == cudaSuccess; }
    void wait() { CUDA_CHECK(cudaEventSynchronize(done_)); }
};
```

### 18.3 Python (CuPy on Jetson)

```python
import cupy as cp
a = cp.random.randn(1024, 1024, dtype=cp.float32)
c = cp.matmul(a, a)  # runs on Orin Nano GPU

# Custom kernel via RawKernel
kern = cp.RawKernel(r'''
extern "C" __global__ void relu(float *d, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) d[i] = fmaxf(d[i], 0.0f);
}''', 'relu')
data = cp.random.randn(1 << 20, dtype=cp.float32)
kern(((data.size + 255) // 256,), (256,), (data, data.size))
```

---

## 19. References

### Internal Guides

* [Orin Nano Platform Guide](../Guide.md) -- boot chain, JetPack, ROS2, inference optimization
* [Orin Nano Memory Architecture](../Orin-Nano-Memory-Architecture/Guide.md) -- SMMU, CMA, zero-copy, GPU memory
* [Orin Nano DLA Deep Dive](../Orin-Nano-DLA-Deep-Dive/Guide.md) -- DLA vs GPU, TensorRT DLA, multi-engine scheduling

### NVIDIA Documentation

* [CUDA C++ Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/)
* [CUDA C++ Best Practices Guide](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/)
* [Jetson Orin Nano Developer Kit](https://developer.nvidia.com/embedded/jetson-orin-nano-developer-kit)
* [Nsight Systems User Guide](https://docs.nvidia.com/nsight-systems/)
* [Nsight Compute User Guide](https://docs.nvidia.com/nsight-compute/)
* [TensorRT Developer Guide](https://docs.nvidia.com/deeplearning/tensorrt/developer-guide/)
* [CUDA WMMA API](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#wmma)
* [Jetson Linux Multimedia API](https://docs.nvidia.com/jetson/l4t-multimedia/)
