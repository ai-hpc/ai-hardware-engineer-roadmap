# Chapter 5: Blackwell Kernel Engineering for Qwen Inference

## Overview

The kernels that make Qwen-class inference fast on Blackwell are not the same kernels that ran on Hopper. The new instruction set (`tcgen05.mma`), the redesigned async copy engine (TMA-2), the broader 5th-gen WGMMA tile shapes, and the warp specialization patterns that exploit FP4/FP6 — all of these are Blackwell-specific. A vLLM or TRT-LLM build compiled for Hopper will run on Blackwell, but will leave a large factor of throughput on the table.

This chapter is for engineers who need to read, write, or tune the kernels that production runtimes use. You don't have to write WGMMA assembly to deploy Qwen on B200, but you do have to understand what the kernels are doing if you want to debug a 30% performance gap.

By the end you should be able to:

* Distinguish 5th-gen WGMMA from Hopper's WGMMA and identify which one a kernel uses.
* Read TMA-2 descriptors and explain what async copy patterns they enable.
* Understand persistent-kernel architecture and why it dominates Blackwell-era LLM serving.
* Recognize FlashAttention-3's Blackwell adaptations vs FA-2.
* Use CUTLASS 4 for custom kernels that target FP4 mixed precision.

---

## 1. 5th-Generation Tensor Cores at the Instruction Level

Tensor cores on Hopper exposed `wgmma.mma_async` — an async warp-group MMA that operated on warp groups of 4 warps (128 threads). Blackwell adds `tcgen05.mma`, which:

* Supports MX-FP4/FP6 as native operand types (Hopper's WGMMA didn't have these).
* Has new tile shapes optimized for the new formats (`m64n16k64` for FP4 vs `m64n16k16` for FP16).
* Reduces register pressure for the MX paths — important because FP4 GEMMs use small accumulator tiles and the compiler often spills on Hopper.
* Adds support for **shared scale factors** as a third operand, enabling true MX block-format consumption without per-block dequant.

A simplified PTX snippet for an MX-FP4 GEMM tile on Blackwell:

```
.reg .b32 a_desc, b_desc;            // matrix descriptors
.reg .b64 scale_desc;                // shared scale factor descriptor
.reg .b32 accum<8>;                  // FP32 accumulator tile

// New Blackwell instruction:
tcgen05.mma.async.aligned.m64n16k64.row.col.f32.e2m1.e2m1.ue8m0
    {accum0, accum1, ..., accum7},
    a_desc, b_desc, scale_desc;
```

The `e2m1.e2m1.ue8m0` triplet says: operand A is MX-FP4 (E2M1), operand B is MX-FP4, scale factors are UE8M0. The accumulator is FP32 (the standard MMA accumulator type).

You won't write this by hand. CUTLASS 4 templates and Triton 3.x emit it. But seeing `tcgen05.mma` in your `cuobjdump` output is how you confirm the Blackwell path is active.

### 1.1 If you see `wgmma.mma_async` on Blackwell

It runs — Hopper instructions are forward-compatible — but you're leaving ~30–40% of peak throughput on the table for MX-FP4 workloads. The compiler chose the safer/older instruction. Common causes:

* CUDA toolkit < 13.
* CUTLASS < 4.0.
* TRT-LLM < 0.20.
* Custom kernel built with `-arch=sm_90` instead of `sm_100`.

Rebuild with Blackwell-targeted toolchain and re-disassemble.

---

## 2. TMA-2: The Async Copy Engine

Hopper introduced the **Tensor Memory Accelerator (TMA)** — a copy engine that moves multi-dimensional tile chunks from HBM to shared memory asynchronously, freeing warps to do compute while the load happens. Blackwell extends it to **TMA-2** with:

* **Multicast** — one TMA load can deliver the same tile to multiple SMs' shared memory simultaneously. Crucial for attention kernels where K and V tiles are read by many SMs.
* **Reduction-on-store** — TMA can accumulate stores into HBM with atomic-like behavior, useful for cross-SM partial reductions.
* **Larger descriptors** — supports up to 5D tile descriptors (vs Hopper's 4D), making it easier to express complex paged-attention KV layouts.
* **Lower-latency completion signals** — the SM gets the completion barrier in ~half the cycles.

For Qwen inference specifically, TMA-2 multicast changes attention kernels significantly:

```
FlashAttention on Hopper:
  Each SM loads its own K and V tiles independently.
  Aggregate HBM read traffic = N_SMs × (K + V tile size).

FlashAttention-3 on Blackwell with TMA-2 multicast:
  One TMA load broadcasts K (or V) to all SMs that need it.
  Aggregate HBM read traffic = 1 × (K + V tile size).
```

For Qwen2.5-72B at long context, this can roughly **halve** the KV cache bandwidth pressure during attention — the second-biggest bandwidth load in decode (after weights).

---

## 3. The Persistent-Kernel Pattern

The dominant inference-kernel architecture on Blackwell is the **persistent kernel**. Instead of launching one kernel per matmul (the per-launch overhead is real even with CUDA Graphs), a persistent kernel:

* Launches once at startup, with grid size matching SM count.
* Each SM runs an infinite loop reading work items from a queue.
* Work items describe "do this matmul on these inputs and write output here."
* When the host wants to compute layer N's QKV, it pushes a work item to the queue; the persistent kernel picks it up.

Benefits on Blackwell:

* Zero per-op kernel launch overhead.
* SMs stay warm; instruction caches don't get flushed between ops.
* Better overlap of compute and memory across operations — the kernel can prefetch the next op's TMA load while computing the current.
* Compatible with CUDA Graphs for the host-side dispatch.

For the Qwen decode hot path, persistent kernels eliminate the ~150 µs of launch overhead per token (across ~470 unfused kernels, even tiny) that would otherwise cap tok/s on Blackwell where everything else is fast.

### 3.1 Skeleton of a persistent-kernel design

```c++
// Pseudocode for a persistent GEMM kernel on Blackwell
__launch_bounds__(256, 1)
__global__ void persistent_gemm(WorkQueue* queue) {
    while (true) {
        WorkItem item = queue->try_pop();
        if (item.type == WorkType::SHUTDOWN) break;
        if (item.type == WorkType::IDLE) {
            __nanosleep(100);
            continue;
        }

        // Execute the matmul described by item
        if (item.dtype == DType::MX_FP4) {
            gemm_tile_mx_fp4(item.A, item.B, item.scale, item.C,
                             item.M, item.N, item.K);
        } else if (item.dtype == DType::MX_FP8) {
            gemm_tile_mx_fp8(item.A, item.B, item.scale, item.C, ...);
        }

        // Signal completion
        atomicAdd(&item.completion_counter, 1);
    }
}
```

In practice TRT-LLM, CUTLASS 4 PerformanceKernel, and vLLM's Blackwell backend all use variants of this pattern. The persistent-kernel architecture is also how TRT-LLM achieves CUDA-Graph-comparable steady-state efficiency with more flexibility (no captured-shape lock-in).

---

## 4. FlashAttention-3 on Blackwell

FlashAttention-2 (Hopper) was the canonical attention kernel for Llama/Qwen/etc. inference. FlashAttention-3 is its Blackwell evolution, with several substantive changes:

| Aspect | FA-2 (Hopper) | FA-3 (Blackwell) |
|---|---|---|
| Tensor core instruction | `wgmma.mma_async` | `tcgen05.mma.async` |
| TMA features used | TMA (load/store) | TMA-2 (multicast, reduce-on-store) |
| Warp specialization | Producer/consumer split | Producer/consumer + reduction warp |
| Accumulator precision | FP32 | FP32 (unchanged) |
| Max tile size | 128 × 64 | 256 × 64 (more registers per SM) |
| Q precision | FP16/BF16 | FP16/BF16/FP8 |
| K, V precision | FP16/BF16 | FP16/BF16/FP8/FP4 |
| Async pipeline depth | 2-3 | 4-6 stages |
| Speedup vs FA-2 same arch | baseline | 1.6–2.2× on MX-FP4/FP8 paths |

FA-3 is what TRT-LLM uses by default on Blackwell. If your kernel trace shows `flash_attention_v2_decode_kernel` rather than `fa3_decode_*_sm100`, you're on the older path — rebuild with FA-3 enabled.

### 4.1 Decode-specific variants

For autoregressive decode (seq_len=1 per query), there's a specialized kernel: **FlashDecoding-3**. It:

* Parallelizes across the KV cache dimension (not the Q dimension, which is 1).
* Uses TMA-2 multicast to broadcast the single Q to all SMs.
* Reduces partial results across SMs via atomic-accumulating TMA stores.

The result: a Qwen2.5-72B decode step's attention kernel runs in ~0.5 ms on a single B200 at ctx=4k, ~3 ms at ctx=32k. That's the kernel-level cost behind the per-token latencies in Chapter 3 §3.

---

## 5. CUTLASS 4 for Custom Kernels

If a runtime's stock kernels don't cover your case (custom fusion, novel quantization, special attention pattern), you'll go to CUTLASS. CUTLASS 4 is the Blackwell-aware iteration.

The mental model: CUTLASS is a C++ template library that lets you compose tile-level matmul kernels by picking:

* Tile sizes (`m`, `n`, `k`).
* Operand dtypes (FP4 / FP6 / FP8 / FP16).
* Accumulator dtype (FP32 or FP16).
* Epilogue (activation, scaling, store pattern).
* Schedule (kernel architecture — persistent, cooperative, etc.).

A minimal CUTLASS 4 GEMM template for MX-FP4 mixed Qwen FFN:

```cpp
using ElementA = cutlass::float_e2m1_t;      // MX-FP4
using ElementB = cutlass::float_e2m1_t;
using ElementAccumulator = float;            // FP32
using ElementC = cutlass::bfloat16_t;        // BF16 output

using GemmKernel = cutlass::gemm::collective::CollectiveBuilder<
    cutlass::arch::Sm100,                     // Blackwell
    cutlass::arch::OpClassTensorOp,
    ElementA, cutlass::layout::RowMajor, 16,
    ElementB, cutlass::layout::ColumnMajor, 16,
    ElementAccumulator,
    Shape<_128, _128, _128>,                  // tile size
    Shape<_2, _1, _1>,                        // cluster shape
    cutlass::gemm::collective::StageCountAutoCarveout<
        sizeof(typename CollectiveMainloop::SharedStorage)>,
    cutlass::gemm::KernelTmaWarpSpecializedFP4Pingpong
>::CollectiveOp;

// ... epilogue ...

using GemmOperator = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;
```

What this gives you:

* A Blackwell-targeted, TMA-2-using, MX-FP4 ping-pong-scheduled GEMM kernel.
* Tile size 128×128×128, warp-specialized producer/consumer.
* Persistent-kernel architecture via the `Pingpong` schedule.

This is the level you operate at when production runtimes don't have the exact dtype/shape combo your model wants.

---

## 6. Warp Specialization on Blackwell

Hopper introduced warp specialization — different warps within a CTA execute different programs (producer warps do TMA loads, consumer warps do MMA). Blackwell extends this with:

* **Reduction warps** — dedicated warps that accumulate partial sums across the SM, useful for attention's softmax stage.
* **Tighter producer/consumer balance** — TMA-2's lower-latency completion means producer warps spend less time waiting, more time prefetching.
* **More warp-group MMA pipeline stages** — 4-6 stages of in-flight MMA per warp group (vs 2-3 on Hopper), reducing register pressure per stage.

A simplified persistent decode-kernel layout for Qwen2.5-72B on Blackwell:

```
Per SM (Blackwell has ~160 SMs per B200):
  Warp group 0 (warps 0-3):  TMA-2 producer
    - Issues async loads of weight tiles and KV tiles
    - Waits on completion barriers
  Warp group 1 (warps 4-7):  MMA consumer
    - Runs tcgen05.mma on incoming tiles
    - Accumulates into FP32 register tile
  Warp group 2 (warps 8-11): Reduction warp group
    - Accumulates across SM via shared memory
    - Issues TMA-2 reduce-on-store to global memory
```

This three-warp-group pattern is now standard for LLM kernels on Blackwell. Older two-warp-group designs (Hopper-era) leave the reduction work to one of the consumer groups and underutilize the SM.

---

## 7. Triton 3.x on Blackwell

For higher-level kernel work, Triton 3.x added Blackwell support:

```python
import triton
import triton.language as tl

@triton.jit
def qwen_ffn_mx_fp4_kernel(
    X_ptr, W_gate_ptr, W_up_ptr, W_down_ptr,
    Out_ptr, scale_ptr,
    M, N, K,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    # Triton 3.x exposes MX-FP4 via tl.dot with format hints
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    # Compute SwiGLU FFN as one fused kernel
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    gate = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    up   = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k in range(0, K, BLOCK_K):
        offs_k = k + tl.arange(0, BLOCK_K)
        x  = tl.load(X_ptr + offs_m[:, None]*K + offs_k[None, :])
        wg = tl.load(W_gate_ptr + offs_n[:, None]*K + offs_k[None, :],
                     dtype=tl.float_e2m1)         # MX-FP4 hint
        wu = tl.load(W_up_ptr   + offs_n[:, None]*K + offs_k[None, :],
                     dtype=tl.float_e2m1)
        gate += tl.dot(x, wg.T)                   # uses tcgen05.mma
        up   += tl.dot(x, wu.T)

    swiglu = tl.sigmoid(gate) * gate * up         # SwiGLU
    # Then a second matmul into W_down...
```

Triton emits Blackwell SASS via the MLIR backend. It's how a lot of "small custom kernel work" gets done in 2026 — full WGMMA-level control without writing PTX. The compile times can be slow (auto-tuning sweeps blow up at FP4 because the tile space is larger), but the runtime is competitive with hand-tuned CUTLASS for many shapes.

---

## 8. Performance Diagnostics — Is Your Kernel on the Blackwell Path?

```bash
# 1. Disassemble the kernel and look for tcgen05.mma
cuobjdump --dump-sass your_kernel.cubin | grep -E "tcgen05|wgmma" | head
# Expect tcgen05.mma; if only wgmma, you're on the Hopper path.

# 2. Confirm TMA-2 multicast in attention kernels
cuobjdump --dump-sass fa3_kernel.cubin | grep "TMA.LDGSTS.MULTICAST"
# Expect at least one MULTICAST instruction per K/V load.

# 3. Profile with Nsight Compute
ncu --set full --kernel-name "gemm_mx_fp4_sm100" your_app
# Look at:
#   - Achieved Occupancy (target: > 60%)
#   - TC Throughput (target: > 50% of peak FP4)
#   - DRAM Bandwidth (target: > 70% of peak)
#   - L2 Cache Hit Rate (target: > 30% for repeated weight loads)

# 4. Confirm persistent-kernel pattern
ncu --metrics launch__waves_per_multiprocessor your_app
# Persistent: ~1 wave per SM (kernel runs continuously)
# Non-persistent: many waves (kernel launches per op)
```

If `tcgen05.mma` is absent, fix the build. If TMA-2 MULTICAST is absent, you're on FA-2 not FA-3. If occupancy is low, your tile size or register pressure is suboptimal — tune via Triton's autotune or CUTLASS template parameters.

---

## 9. The Real-World Stack Map

Where each layer lives in mid-2026:

| Layer | Tool | Purpose |
|---|---|---|
| User-facing API | OpenAI-compatible REST | `/v1/chat/completions` |
| Serving frontend | TRT-LLM Triton frontend / vLLM | Request batching, scheduling |
| Engine | TensorRT engine (compiled) | The compiled-once graph |
| Kernel library | TRT-LLM kernels, FA-3, custom CUTLASS | The per-op implementations |
| Tile-level GEMM | CUTLASS 4 (templated) | Reusable tile kernels |
| Custom GPU code | Triton 3.x | Quick fused kernels, exploration |
| PTX assembly | Manual / cuobjdump output | Performance debugging only |
| SASS | NVCC backend / SM SASS | What actually runs on the SM |

You rarely touch the bottom three layers in normal LLM deployment. You touch CUTLASS 4 when stock kernels are missing your dtype/shape. You touch Triton when prototyping a new fusion. Most of the time you're at the top three layers and the bottom is taking care of itself — *if* you've validated with the diagnostics above that the right kernel paths are active.

---

## Key Takeaways

| Takeaway | Why it matters |
|---|---|
| `tcgen05.mma` is the new instruction; `wgmma` is the fallback | Look for it in disassembly to confirm Blackwell path |
| TMA-2 multicast halves KV bandwidth in attention | Crucial for long-context decode |
| Persistent kernels dominate Blackwell LLM serving | Zero launch overhead, warm caches, work-queue dispatch |
| FlashAttention-3 is the canonical attention; FA-2 is legacy | 1.6–2.2× speedup on the MX-FP4 path |
| CUTLASS 4 is the layer for custom Blackwell kernels | Templated tile-level composition with FP4 first-class |
| Triton 3.x handles MX-FP4 with `tl.dot` and dtype hints | Quick exploration without PTX |
| Diagnostics: disasm shows `tcgen05`, ncu shows >60% occupancy | Validate every deployment, not just first one |

---

## Resources

* **[NVIDIA CUDA 13 Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/):** `tcgen05.mma`, TMA-2 reference.
* **[CUTLASS 4 GitHub](https://github.com/NVIDIA/cutlass):** Blackwell-aware tile kernel templates.
* **[FlashAttention-3 paper](https://arxiv.org/abs/2407.08608):** The attention kernel design.
* **[Triton 3.x documentation](https://triton-lang.org/):** Higher-level kernel DSL.
* **[Nsight Compute on Blackwell](https://docs.nvidia.com/nsight-compute/):** Profiling.
* **[Phase 5 — CUDA Advanced Optimization](../CUDA-Advanced-Optimization/README.md):** Persistent kernels, warp specialization (Hopper foundation).
* **[Qwen Inference — Lecture 6 (Batched GEMM)](../../../../Track%20C%20-%20Edge%20AI/Qwen%20Inference%20Optimization/Lecture-06.md):** cuBLAS counterpart to this lecture.
* **[Chapter 6 — Production Serving on Blackwell](06-Production-Serving-on-Blackwell.md):** Putting the kernels into production.
