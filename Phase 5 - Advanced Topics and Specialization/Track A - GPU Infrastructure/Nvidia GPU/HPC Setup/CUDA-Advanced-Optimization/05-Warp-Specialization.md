# 05 — Warp Specialization

## 1. The Problem: Compute and Memory Latency Cannot Both Be Hidden Simultaneously

Traditional CUDA kernels use **all warps for all tasks** — both data loading and computation:

```
Standard kernel (128 threads = 4 warps):

Warp 0: [Load from HBM....wait....][Compute GEMM][Load from HBM....wait....][Compute]
Warp 1: [Load from HBM....wait....][Compute GEMM][Load from HBM....wait....][Compute]
Warp 2: [Load from HBM....wait....][Compute GEMM][Load from HBM....wait....][Compute]
Warp 3: [Load from HBM....wait....][Compute GEMM][Load from HBM....wait....][Compute]

Problem:
  During [Load...wait] phases: Tensor Cores are IDLE
  During [Compute GEMM] phases: HBM DMA is IDLE
  → Neither unit is fully utilized
```

This is called the **memory wall** — the latency to load data from HBM (hundreds of nanoseconds) prevents the GPU's compute units from staying busy.

**Warp specialization** solves this by assigning different roles to different warps:

```
Warp-specialized kernel:

Warp 0 (Producer): [Load tile A from HBM][Load tile B from HBM][Load tile C][...]
Warp 1 (Consumer): [wait for tile A][Compute GEMM with tile A][Compute GEMM with B][...]
Warp 2 (Consumer): [                    ][Compute GEMM with tile B][Compute with C][...]
Warp 3 (Consumer): [                         ][Compute GEMM with C][...           ][...]

Result:
  Producer warps: always loading (hide compute latency)
  Consumer warps: always computing (hide memory latency)
  Both HBM DMA and Tensor Cores run simultaneously → near-peak utilization
```

---

## 2. Hardware Foundation: Hopper Warpgroup MMA (WGMMA)

On Hopper (H100/H200), NVIDIA introduced **Warpgroup Matrix Multiply-Accumulate (WGMMA)** — a new instruction that operates on **4 warps (128 threads) simultaneously** called a **warpgroup**.

```
H100/H200 SM capabilities:
  - 4 Tensor Core units per SM
  - WGMMA operates all 4 simultaneously on a 64×64×16 tile
  - WGMMA latency: ~23 cycles
  - A new WGMMA can be issued every 8 cycles (pipelined)
  - Optimal: keep WGMMA and TMA (Tensor Memory Accelerator) running in parallel
```

Hopper also added **TMA (Tensor Memory Accelerator)** — a hardware unit that performs asynchronous memory copies from HBM to shared memory **without using CUDA cores or warp slots**. This is what makes warp specialization practical:

```
TMA: dedicated hardware for HBM→SRAM copies
     Uses 1 thread to issue, hardware does the rest
     Producer warp issues TMA, then frees all other warps for compute

Without TMA (Ampere A100):
  All threads participate in data loading (ldgsts = async copy instruction)
  Threads are busy during loading

With TMA (H100/H200):
  1 thread issues TMA load
  Remaining 127 threads immediately compute
  → Warp specialization becomes near-zero overhead
```

---

## 3. Warp Specialization Pattern

### Basic Structure

```cpp
#include <cuda.h>
#include <cuda_runtime.h>
#include <cute/tensor.hpp>   // CUTLASS CuTe library

__global__ void warp_specialized_gemm(
    const float* A, const float* B, float* C,
    int M, int N, int K
) {
    // Identify warpgroup role
    int warp_id = threadIdx.x / 32;
    int warpgroup_id = threadIdx.x / 128;   // 4 warps per warpgroup

    // Producer warpgroup: handles data loading
    // Consumer warpgroup: handles computation
    bool is_producer = (warpgroup_id == 0);
    bool is_consumer = (warpgroup_id != 0);

    // Shared memory pipeline buffers (double/triple buffering)
    __shared__ float smem_A[2][TILE_M][TILE_K];   // 2 = double buffer
    __shared__ float smem_B[2][TILE_K][TILE_N];

    // Pipeline synchronization barriers
    // (Hopper: cuda::barrier or __shared__ cuda::pipeline)
    __shared__ cuda::barrier<cuda::thread_scope_block> barriers[2];

    int buf = 0;   // current buffer index (ping-pong)

    if (is_producer) {
        // === PRODUCER ROLE ===
        // Issue async loads to fill pipeline
        for (int k_tile = 0; k_tile < K / TILE_K; k_tile++) {

            // Asynchronously copy tile from HBM to SRAM
            cuda::memcpy_async(
                smem_A[buf], &A[...], sizeof(smem_A[0]),
                barriers[buf]
            );
            cuda::memcpy_async(
                smem_B[buf], &B[...], sizeof(smem_B[0]),
                barriers[buf]
            );

            // Signal consumer: data is in flight (not yet arrived)
            barriers[buf].arrive();
            buf ^= 1;   // switch buffer
        }

    } else {
        // === CONSUMER ROLE ===
        float acc[TILE_M][TILE_N] = {0};   // accumulator in registers

        for (int k_tile = 0; k_tile < K / TILE_K; k_tile++) {

            // Wait for producer to fill this buffer
            barriers[buf].wait(/*phase*/);

            // Compute GEMM on SRAM tile (Tensor Cores)
            wgmma_gemm(smem_A[buf], smem_B[buf], acc, TILE_M, TILE_N, TILE_K);

            buf ^= 1;
        }

        // Write accumulator to HBM
        store_result(acc, C, ...);
    }
}
```

---

## 4. FlashAttention-3: Warp Specialization in Practice

FlashAttention-3 (for H100/H200) uses warp specialization to overlap **Q@K^T GEMM**, **Softmax**, and **P@V GEMM** simultaneously:

```
FlashAttention-3 warpgroup layout (per SM, 128 threads per warpgroup):

Warpgroup 0 (Q-tiles producer):
  Load Q tile from HBM using TMA → write to SRAM_Q
  Signal SRAM_Q ready to consumer

Warpgroup 1 (QK GEMM consumer / PV GEMM producer):
  Wait for SRAM_Q
  Compute S_tile = Q_tile @ K_tile^T  (WGMMA)
  Compute Softmax(S_tile) → P_tile   (in registers)
  Write P_tile to SRAM_P

Warpgroup 2 (PV GEMM consumer):
  Wait for SRAM_P
  Compute O_tile = P_tile @ V_tile    (WGMMA)
  Accumulate into O register

All three run concurrently via async barriers
```

```
Timeline (without warp specialization):
  [Load Q][QK GEMM][Softmax][Load P][PV GEMM][Load Q][QK GEMM]...
           SM busy ←→ SM stalls on memory

Timeline (with warp specialization):
  WG0: [Load Q0][Load Q1][Load Q2][Load Q3]...
  WG1:          [QK0+Softmax][QK1+Softmax][QK2+Softmax]...
  WG2:                   [PV0][PV1][PV2]...
  → All stages run in parallel → ~2× throughput vs FA2
```

---

## 5. Software Pipelining with Double Buffering

Warp specialization requires **software pipelining** — pre-loading the next tile while computing the current one:

```
Without pipelining (producer-consumer without lookahead):
  Load tile 0 → [stall until loaded] → Compute tile 0 → Load tile 1 → [stall] → Compute tile 1

With pipelining (double buffering, 2 SRAM buffers):
  Load tile 0 (buf 0) → Compute tile 0 (buf 0)
                         Load tile 1 (buf 1)
                                              Compute tile 1 (buf 1)
                                              Load tile 2 (buf 0)
                                                                     Compute tile 2 (buf 0)
  → Load and compute fully overlap after initial startup
```

```cpp
// Double-buffered software pipeline with Hopper async barriers
#include <cuda/pipeline>

__shared__ float A_smem[2][TILE_M * TILE_K];  // 2 buffers
__shared__ float B_smem[2][TILE_K * TILE_N];
__shared__ cuda::pipeline_shared_state<cuda::thread_scope_block, 2> pipeline_state;

auto pipeline = cuda::make_pipeline(cg::this_thread_block(), &pipeline_state);

// === PROLOGUE: Pre-load first tile ===
pipeline.producer_acquire();
cuda::memcpy_async(A_smem[0], A_ptr, tile_bytes, pipeline);
cuda::memcpy_async(B_smem[0], B_ptr, tile_bytes, pipeline);
pipeline.producer_commit();

// === MAIN LOOP ===
for (int stage = 0; stage < num_stages - 1; stage++) {
    int buf_load    = (stage + 1) % 2;
    int buf_compute = stage % 2;

    // Pre-load next tile (producer role)
    pipeline.producer_acquire();
    cuda::memcpy_async(A_smem[buf_load], &A_ptr[next_tile], tile_bytes, pipeline);
    cuda::memcpy_async(B_smem[buf_load], &B_ptr[next_tile], tile_bytes, pipeline);
    pipeline.producer_commit();

    // Wait for current tile and compute (consumer role)
    pipeline.consumer_wait();
    compute_gemm(A_smem[buf_compute], B_smem[buf_compute], acc);
    pipeline.consumer_release();
}

// === EPILOGUE: Drain last tile ===
pipeline.consumer_wait();
compute_gemm(A_smem[(num_stages-1) % 2], B_smem[(num_stages-1) % 2], acc);
pipeline.consumer_release();
```

---

## 6. CUTLASS 3.x — Warp Specialization Library

NVIDIA's CUTLASS 3.x implements warp specialization with TMA and WGMMA via CuTe:

```cpp
#include <cutlass/gemm/device/gemm_universal_adapter.h>
#include <cutlass/gemm/kernel/gemm_universal.hpp>
#include <cutlass/gemm/collective/collective_builder.hpp>

// Define a Hopper warp-specialized GEMM kernel
using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder<
    cutlass::arch::Sm90,                    // H100/H200
    cutlass::arch::OpClassTensorOp,         // Tensor Cores
    cutlass::bfloat16_t,                    // A dtype
    cutlass::layout::RowMajor,
    8,                                      // alignment
    cutlass::bfloat16_t,                    // B dtype
    cutlass::layout::ColumnMajor,
    8,
    float,                                  // accumulator dtype
    cutlass::Shape<128, 128, 64>,           // tile shape
    cutlass::Shape<1, 1, 1>,               // cluster shape
    cutlass::gemm::collective::StageCountAutoCarveout<sizeof(float[128][128])>,
    cutlass::gemm::KernelTmaWarpSpecialized // ← warp-specialized schedule
>::CollectiveOp;

// CollectiveMainloop now handles:
// - Producer warps issuing TMA loads
// - Consumer warps issuing WGMMA instructions
// - Double/triple buffering in shared memory
// - Async barrier synchronization
```

---

## 7. When Warp Specialization Applies

| Workload | Benefit | Notes |
|---|---|---|
| Large GEMM (attention QK, PV, MLP) | Very high | Industry standard in FA3, TRT-LLM |
| Convolutions (CNNs) | High | cuDNN uses this internally |
| Sparse attention | High | Irregular access → specialization helps |
| Small GEMMs (< 256×256) | Low | Overhead not amortized over small tiles |
| Element-wise ops | None | No compute to overlap with loading |
| Memory-bound reductions | Low | Already bandwidth-limited |

**Warp specialization is only worthwhile when:**
1. Kernel is on or near the compute roofline
2. Tile size is large enough to amortize setup
3. Running on Hopper (H100/H200) — TMA makes it practical

---

## 8. Practical Results

### FlashAttention-3 vs FA2 Performance (H200, BF16)

| Sequence Length | FA2 (A/B warps) | FA3 (warp-specialized) | Speedup |
|---|---|---|---|
| 512 | 280 TFLOPS | 350 TFLOPS | 1.25× |
| 2048 | 510 TFLOPS | 740 TFLOPS | 1.45× |
| 8192 | 620 TFLOPS | 950 TFLOPS | 1.53× |
| 16384 | 650 TFLOPS | 1050 TFLOPS | 1.62× |

H200 BF16 peak: 1979 TFLOPS
FA3 at seq=16K achieves **53% of peak** vs FA2's **33%** — both are compute-bound, but FA3's pipeline hides WGMMA and TMA latency simultaneously.

---

## 9. Interview-Level Summary

```
Warp Specialization:
  Divide warps into producers (load data) and consumers (compute)
  Producers use TMA to issue async HBM→SRAM copies with minimal thread usage
  Consumers use WGMMA to execute Tensor Core matrix multiply
  Software pipeline (double buffer) ensures both are always busy
  Result: Tensor Cores and HBM DMA run in parallel → near-peak utilization

Requires:
  Hopper (H100/H200) for TMA + WGMMA
  CUTLASS 3.x or hand-written CuTe kernel
  Large tile sizes to amortize producer-consumer coordination overhead

Used in:
  FlashAttention-3
  TensorRT-LLM GEMM kernels
  cuBLAS Hopper-native GEMM
  NVIDIA NCCL ring kernels
```

---

## References

- [FlashAttention-3 Paper](https://arxiv.org/abs/2407.08608)
- [CUTLASS 3.x Warp Specialization](https://github.com/NVIDIA/cutlass/blob/main/docs/cute/0x_warp_specialization.md)
- [Hopper Architecture: TMA and WGMMA](https://developer.nvidia.com/blog/nvidia-hopper-architecture-in-depth/)
- [CuTe Tutorial](https://github.com/NVIDIA/cutlass/blob/main/media/docs/cute/00_quickstart.md)
- [Dissecting the Ampere GPU Architecture](https://arxiv.org/abs/2108.11458)
- [PTX ISA: WGMMA Instructions](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#warp-level-matrix-instructions-for-mma)
