# Chapter 5: Blackwell Kernel Engineering for Qwen Inference

## Overview

The kernels that make Qwen-class inference fast on Blackwell are not the same kernels that ran on Hopper. Blackwell's 5th-generation tensor cores are driven by a new instruction family (`tcgen05.*`) that accumulates into a new per-SM **Tensor Memory (TMEM)** instead of registers. TMA, Hopper's async copy engine, gains a few Blackwell-only modes. And the warp-specialized kernel structure is reorganized around both. Hopper's `wgmma.mma_async` is an `sm_90a`-only feature, so it does not run on Blackwell at all. A build without Blackwell (`sm_100a` / `sm_100f`) kernels either falls back to slower generic code paths or fails to load, and in either case leaves a large factor of throughput on the table.

This chapter is for engineers who need to read, write, or tune the kernels that production runtimes use. You don't have to write `tcgen05` PTX to deploy Qwen on B200, but you do have to understand what the kernels are doing if you want to debug a 30% performance gap.

By the end you should be able to:

* Distinguish Blackwell's `tcgen05.mma` (accumulator in Tensor Memory) from Hopper's `wgmma.mma_async` (accumulator in registers) and identify which one a kernel uses.
* Explain which TMA features Hopper already had (5-D tensor copies, cluster multicast, bulk reductions) and which ones Blackwell adds.
* Understand persistent-kernel architecture and why it dominates Blackwell-era LLM serving.
* Tell which attention kernels actually target Blackwell (FlashAttention-3 is Hopper-only).
* Use CUTLASS 4 for custom kernels that target FP4 mixed precision.

---

## 1. 5th-Generation Tensor Cores at the Instruction Level

Tensor cores on Hopper exposed `wgmma.mma_async`: an async MMA issued by a warp group of 4 warps (128 threads), with the accumulator held in that warp group's registers. Blackwell replaces it with the `tcgen05` family:

* **The accumulator lives in Tensor Memory (TMEM),** a dedicated per-SM memory allocated with `tcgen05.alloc`. The MMA's D operand is a TMEM address. Results reach registers only through `tcgen05.ld`, usually in the epilogue. The registers Hopper spent on accumulators are free.
* **One thread issues the MMA,** not a warp group, and it completes asynchronously. Completion is tracked by `tcgen05.commit` arriving on an mbarrier. With `.cta_group::2`, one thread drives the tensor cores of a pair of SMs.
* **Low-precision and block-scaled formats are native:** `.kind::f8f6f4` for FP8/FP6/FP4, and with `.block_scale`, `.kind::mxf8f6f4`, `.kind::mxf4` and `.kind::mxf4nvf4` (NVFP4). The scale factors are read from TMEM too.
* **The shape and operand types are not spelled out in the mnemonic.** They are encoded in a 32-bit instruction descriptor (`idesc`).

The PTX for one block-scaled MX-FP4 MMA, following the PTX ISA's `tcgen05.mma` syntax:

```
// D (FP32, in TMEM) += A (MX-FP4) x B (MX-FP4), with E8M0 block scales held in TMEM
tcgen05.mma.cta_group::1.kind::mxf4.block_scale
    [d_tmem],                       // TMEM address of the accumulator tile
    a_desc, b_desc,                 // 64-bit shared-memory matrix descriptors
    idesc,                          // 32-bit instruction descriptor: M, N, A/B formats, scale format
    [scale_a_tmem], [scale_b_tmem], // TMEM addresses of the block scales
    enable_input_d;                 // predicate: accumulate (D += A*B) or overwrite (D = A*B)

// Epilogue: move the finished tile from TMEM to registers
tcgen05.ld.sync.aligned.32x32b.x8.b32 {r0, r1, r2, r3, r4, r5, r6, r7}, [d_tmem];
tcgen05.wait::ld.sync.aligned;
```

You won't write this by hand. cuBLAS/cuBLASLt, CUTLASS (CuTe's SM100 MMA atoms) and Triton emit it. To confirm the Blackwell path is active, dump the PTX (`cuobjdump --dump-ptx`) and look for `tcgen05.mma`. In SASS, Hopper's `wgmma` appears as `HGMMA`, and Blackwell kernels should not contain it.

### 1.1 If your Blackwell profile shows no `tcgen05` kernels

`wgmma.mma_async` requires `sm_90a`, and architecture-specific (`a`) features do not carry forward to later architectures. So a Hopper-tuned kernel never runs on a B200. Either the binary also carries generic code (Ampere-style `mma.sync`, or PTX the driver JIT-compiles) and runs that slower path, or the kernel fails to load. Common causes:

* CUDA toolkit older than 12.8, the first release with Blackwell (`sm_100`) targets.
* A CUTLASS release without SM100 kernels (before 3.8), or a runtime build compiled without Blackwell kernels.
* A custom kernel built with `-arch=sm_90a` (or `sm_90`) and no `sm_100a` / `sm_100f` target.

Rebuild with a Blackwell-targeted toolchain and re-disassemble.

---

## 2. TMA on Blackwell: What Is Actually New

Hopper introduced the **Tensor Memory Accelerator (TMA)**, a copy engine that moves multi-dimensional tiles between global and shared memory asynchronously, so warps can compute while loads are in flight. There is no "TMA-2" in NVIDIA's documentation, and most features attributed to it were already in Hopper (`sm_90`):

* **1-D to 5-D tensor copies** (`cp.async.bulk.tensor.1d` … `.5d`).
* **Multicast** to the CTAs of a thread-block cluster (`.multicast::cluster`).
* **Bulk tensor reductions** into global memory (`cp.reduce.async.bulk.tensor`).

What Blackwell adds (PTX targets `sm_100a` / `sm_100f`):

* **`.tile::gather4` and `.tile::scatter4`** — load or store four rows selected by index in one operation. Useful for gathering scattered rows, such as MoE tokens or paged KV blocks.
* **`.im2col::w` and `.im2col::w::128` modes** for convolution-style access.
* **`.cta_group`** — a TMA load can signal the mbarrier of either CTA in a pair, matching `tcgen05.mma.cta_group::2`.

What multicast buys, on Hopper and Blackwell alike:

```
Without multicast:
  Each CTA in a cluster loads its own copy of a shared tile.
  L2 -> SMEM traffic for that tile = cluster_size x tile_size.

With multicast:
  One TMA load delivers the tile to every CTA in the cluster that needs it.
  One source tile load can be distributed to several destination CTAs.
  Destination shared memories still each receive their own tile.
```

The saving is bounded by the cluster size (8 CTAs portable, up to 16 non-portable), not by the number of SMs, and it only applies to tiles that several CTAs in a cluster actually share. In decode attention each KV tile is usually read once per (sequence, KV-head) pair, so how much multicast saves depends on the kernel's work split. Measure L2 and DRAM traffic in Nsight Compute rather than assuming a fixed reduction.

---

## 3. Persistent tile scheduling versus a device service loop

In a persistent GEMM, a limited set of resident CTAs repeatedly takes tiles from the **current operation** until its work is finished. The kernel still has a launch and a finite lifetime. This can improve tile scheduling and load balance; it does not imply that a serving engine launches one immortal kernel at startup for the entire model.

A separate design is a device-resident service loop that consumes host-submitted operations. That design requires a carefully specified queue, memory-ordering protocol, cancellation, completion, and shutdown. It can reserve resources that other kernels need, and watchdog or platform restrictions may matter. Do not attribute such a loop to a runtime without inspecting its implementation.

For a teaching persistent tile scheduler:

```text
launch a bounded number of CTAs
for each CTA:
    obtain a valid tile index for this GEMM
    while a tile remains:
        wait for operands, compute tile, write output
        obtain next tile
finish; signal completion through the normal launch/runtime path
```

This is algorithmic pseudocode, not a runnable CUDA synchronization implementation. Compare it with the actual scheduler in the selected [CUTLASS SM100 example](https://docs.nvidia.com/cutlass/latest/media/docs/cpp/blackwell_functionality.html). Measure launch cost, tiles per CTA, idle tails, and occupancy. A small or irregular GEMM may favor a different schedule.

---

## 4. Attention Kernels on Blackwell

The FlashAttention generations target different hardware, and only the newest targets Blackwell:

| Kernel | Target | Tensor-core path |
|---|---|---|
| FlashAttention-2 | Ampere, Ada, Hopper (generic) | `mma.sync`; runs on Blackwell, but far from its peak |
| FlashAttention-3 | **Hopper only** ("optimized for Hopper GPUs", per the FlashAttention repository) | `wgmma` + TMA + warp specialization; cannot run on Blackwell |
| FlashAttention-4 | Hopper and Blackwell (H100, B200); written in CuTe DSL | On Blackwell: `tcgen05.mma` with TMEM accumulators |

Production runtimes on B200 (TensorRT-LLM, FlashInfer, cuDNN's SDPA, CUTLASS's Blackwell FMHA examples) ship their own SM100 attention kernels, and kernel names vary by runtime and version. To check which one ran, read the kernel names in an Nsight Systems trace and confirm `tcgen05` in their PTX, as in §8.

### 4.1 Decode-specific variants

Autoregressive decode has one query token per sequence, so there is little work along Q. **Flash-Decoding** (split-KV) parallelizes along the KV sequence instead. Each CTA processes a slice of the KV cache and writes partial outputs plus softmax statistics, and a second step combines the partials. Most production decode kernels on Blackwell use this split-KV structure. Measure the attention kernel's share of a decode step on your own B200; use the measured decode budget from Chapter 3 rather than assuming a fixed token rate.

---

## 5. CUTLASS 4 for Custom Kernels

If a runtime's stock kernels don't cover your case (custom fusion, novel quantization, special attention pattern), you'll go to CUTLASS. CUTLASS 4 is the Blackwell-aware iteration.

The mental model: CUTLASS is a C++ template library that lets you compose tile-level matmul kernels by picking:

* Tile sizes (`m`, `n`, `k`).
* Operand dtypes (FP4 / FP6 / FP8 / FP16).
* Accumulator dtype (FP32 or FP16).
* Epilogue (activation, scaling, store pattern).
* Schedule (kernel architecture — persistent, cooperative, etc.).

The following is a structural CUTLASS sketch, not a complete buildable kernel or a verified Qwen checkpoint recipe. Start a runnable lab from a pinned upstream SM100 block-scaled example:

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

* A Blackwell-targeted, TMA-fed, MX-FP4 GEMM kernel. The template above is illustrative: take the exact SM100 kernel-schedule and builder type names from CUTLASS's Blackwell examples for your CUTLASS version, because they changed across 3.8-4.x.
* Tile size 128×128×128, warp-specialized producer/consumer.
* Persistent-kernel architecture via the `Pingpong` schedule.

This is the level you operate at when production runtimes don't have the exact dtype/shape combo your model wants.

---

## 6. Warp Specialization on Blackwell

Hopper introduced warp specialization: different warps within a CTA run different programs. Producer warps issue TMA loads, and consumer warp groups run `wgmma` while holding the accumulators in their registers. Blackwell keeps the idea but changes who does what, because the accumulator moved into Tensor Memory:

* **A dedicated MMA warp** — one elected thread issues `tcgen05.mma`. It needs no accumulator registers.
* **Separate epilogue warps** — they `tcgen05.ld` a finished tile out of TMEM, apply scaling and activation, and store it. Meanwhile the MMA warp is already accumulating the next tile into a second TMEM buffer.
* **Deeper pipelines are cheaper** — shared-memory stages hold operands, and TMEM holds accumulators, so adding stages no longer costs register pressure the way it did on Hopper.

A simplified warp-specialized GEMM/decode-kernel layout on Blackwell:

```
Per SM (B200 has 148 SMs):
  Producer warp:   issues TMA loads of weight and KV tiles into shared memory
                   (multicast within the cluster where tiles are shared), waits on mbarriers
  MMA warp:        one elected thread issues tcgen05.mma; the accumulator lives in Tensor Memory;
                   tcgen05.commit signals the epilogue through an mbarrier
  Epilogue warps:  tcgen05.ld the finished tile from TMEM into registers, apply scale/activation,
                   store via TMA (or reduce split-K partials with cp.reduce.async.bulk)
```

This producer / MMA / epilogue split is the structure of CUTLASS's SM100 warp-specialized kernels. On Hopper the consumer warp groups owned the accumulators, so the epilogue ran in those same warps. On Blackwell, a separate set of warps can drain TMEM while the tensor cores keep working.

---

## 7. Triton 3.x on Blackwell

For higher-level kernel work, select a Triton release with support for the exact target. The following incomplete sketch illustrates gate/up projection structure; it omits output stores, tail masks, synchronization/layout details, and the down projection. Do not run it as a complete SwiGLU kernel:

```python
import triton
import triton.language as tl

@triton.jit
def qwen_ffn_mx_fp4_kernel(
    X_ptr, W_gate_ptr, W_up_ptr, W_down_ptr,
    Out_ptr,
    Sg_ptr, Su_ptr,          # E8M0 block scales for the FP4 weights, one per 32 values
    M, N, K,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    # Triton exposes OCP-MX matmuls through tl.dot_scaled (formats "e2m1", "e4m3", "e5m2", "bf16", "fp16")
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    # Compute SwiGLU FFN as one fused kernel
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    gate = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    up   = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k in range(0, K, BLOCK_K):
        offs_k  = k + tl.arange(0, BLOCK_K)
        offs_kp = k // 2 + tl.arange(0, BLOCK_K // 2)      # FP4: two values packed per uint8 along K
        offs_ks = k // 32 + tl.arange(0, BLOCK_K // 32)    # one E8M0 scale per 32 values
        x   = tl.load(X_ptr + offs_m[:, None]*K + offs_k[None, :])                 # bf16 [BM, BK]
        wg  = tl.load(W_gate_ptr + offs_kp[:, None]*N + offs_n[None, :])          # uint8 [BK/2, BN]
        wu  = tl.load(W_up_ptr   + offs_kp[:, None]*N + offs_n[None, :])
        sg  = tl.load(Sg_ptr + offs_n[:, None]*(K // 32) + offs_ks[None, :])       # uint8 [BN, BK/32]
        su  = tl.load(Su_ptr + offs_n[:, None]*(K // 32) + offs_ks[None, :])
        # Native block-scaled MMA on Blackwell; emulated (upcast to bf16) on GPUs without MX support
        gate = tl.dot_scaled(x, None, "bf16", wg, sg, "e2m1", acc=gate)
        up   = tl.dot_scaled(x, None, "bf16", wu, su, "e2m1", acc=up)

    swiglu = tl.sigmoid(gate) * gate * up         # SwiGLU
    # Then a second matmul into W_down...
```

Triton emits Blackwell SASS via the MLIR backend. It's how a lot of "small custom kernel work" gets done in 2026 — tensor-core-level control without writing PTX. The compile times can be slow (auto-tuning sweeps blow up at FP4 because the tile space is larger), but the runtime is competitive with hand-tuned CUTLASS for many shapes.

---

## 8. Performance Diagnostics — Is Your Kernel on the Blackwell Path?

```bash
# 1. Look for tcgen05.mma in the embedded PTX
cuobjdump --dump-ptx your_binary | grep -E "tcgen05\.mma|wgmma" | head
# Embedded PTX may be absent in a cubin-only build; correlate with executed kernels and SASS.
# mma.sync alone is not evidence of a bug: dtype/shape can justify a legacy path.

# 2. Identify the attention kernel that actually ran
nsys profile -o decode_trace your_app && nsys stats --report cuda_gpu_kern_sum decode_trace.nsys-rep
# Check the attention kernel's name and that its PTX contains tcgen05 (step 1).

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

If `tcgen05.mma` is absent, fix the build. If the attention kernel is FlashAttention-2 or another generic path, switch to a Blackwell attention backend (§4). If occupancy is low, your tile size or register pressure is suboptimal — tune via Triton's autotune or CUTLASS template parameters.

---

## 9. The Real-World Stack Map

Where each layer lives in mid-2026:

| Layer | Tool | Purpose |
|---|---|---|
| User-facing API | OpenAI-compatible REST | `/v1/chat/completions` |
| Serving frontend | TRT-LLM Triton frontend / vLLM | Request batching, scheduling |
| Engine | TensorRT engine (compiled) | The compiled-once graph |
| Kernel library | TRT-LLM kernels, FlashInfer / cuDNN / FlashAttention-4 attention, custom CUTLASS | The per-op implementations |
| Tile-level GEMM | CUTLASS 4 (templated) | Reusable tile kernels |
| Custom GPU code | Triton 3.x | Quick fused kernels, exploration |
| PTX assembly | Manual / cuobjdump output | Performance debugging only |
| SASS | NVCC backend / SM SASS | What actually runs on the SM |

You rarely touch the bottom three layers in normal LLM deployment. You touch CUTLASS 4 when stock kernels are missing your dtype/shape. You touch Triton when prototyping a new fusion. Most of the time you're at the top three layers and the bottom is taking care of itself — *if* you've validated with the diagnostics above that the right kernel paths are active.

---

## Key Takeaways

| Takeaway | Why it matters |
|---|---|
| `tcgen05.mma` (accumulator in TMEM) is the Blackwell MMA; `wgmma` is `sm_90a`-only | Look for `tcgen05` in the PTX to confirm the Blackwell path |
| TMA multicast (a Hopper feature) cuts L2 traffic for tiles shared within a cluster | Savings are bounded by cluster size; measure them |
| Persistent kernels dominate Blackwell LLM serving | Zero launch overhead, warm caches, work-queue dispatch |
| FlashAttention-3 is Hopper-only; Blackwell needs SM100 attention kernels | FA-4, FlashInfer, cuDNN or runtime-specific kernels |
| CUTLASS 4 is the layer for custom Blackwell kernels | Templated tile-level composition with FP4 first-class |
| Triton 3.x handles MX-FP4 with `tl.dot` and dtype hints | Quick exploration without PTX |
| Diagnostics: disasm shows `tcgen05`, ncu shows >60% occupancy | Validate every deployment, not just first one |

---

## Resources

* **[PTX ISA](https://docs.nvidia.com/cuda/parallel-thread-execution/):** `tcgen05.*` instructions, Tensor Memory, and `cp.async.bulk.tensor` (TMA), with per-target requirements.
* **[Blackwell Tuning Guide](https://docs.nvidia.com/cuda/blackwell-tuning-guide/):** compute capability 10.0 specifics, L2 capacity.
* **[CUTLASS 4 GitHub](https://github.com/NVIDIA/cutlass):** Blackwell-aware tile kernel templates.
* **[FlashAttention-3 paper](https://arxiv.org/abs/2407.08608):** The Hopper attention design (wgmma, TMA, warp specialization).
* **[FlashAttention repository](https://github.com/Dao-AILab/flash-attention):** FlashAttention-4 (CuTe DSL) for Hopper and Blackwell.
* **[Triton 3.x documentation](https://triton-lang.org/):** Higher-level kernel DSL.
* **[Nsight Compute on Blackwell](https://docs.nvidia.com/nsight-compute/):** Profiling.
* **[Phase 5 — CUDA Advanced Optimization](../CUDA-Advanced-Optimization/README.md):** Persistent kernels, warp specialization (Hopper foundation).
* **[Qwen Inference — Lecture 6 (Batched GEMM)](../../../../Track%20C%20-%20Edge%20AI/Qwen%20Inference%20Optimization/Lecture-06.md):** cuBLAS counterpart to this lecture.
* **[Chapter 6 — Production Serving on Blackwell](06-Production-Serving-on-Blackwell.md):** Putting the kernels into production.
