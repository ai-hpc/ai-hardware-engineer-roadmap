# Lecture 4 — GPU Kernel Performance Basics

**Parent:** [FlashAttention Course](Guide.md)

**One-line purpose:** Cover the strict minimum of GPU performance hardware (memory hierarchy, warps, tensor cores, occupancy, coalescing) needed to read FlashAttention source code without getting lost.

**Prerequisites:** Lectures 1–3. Basic CUDA from Phase 4 Track B.

**Artifact:** Nsight Compute reports for a coalesced vs uncoalesced copy kernel and a warp-shuffle reduction microbenchmark.

---

## Why it matters

The FlashAttention codebase is dense with CUDA primitives — `__shfl_xor_sync`, shared-memory bank patterns, register tile sizes, `cp.async`, WGMMA descriptors. If those words are noise to you, you cannot read the kernel; you are reading a foreign language. This lecture is the dictionary.

You do not need to write GEMM from scratch. You do need to know what each layer of memory costs, how warps and tensor cores execute, and what the common failure modes are.

---

## Mental model

### Memory hierarchy on a modern NVIDIA GPU

| Level | Size (Hopper SXM) | Latency | Bandwidth | Who owns it |
|------|--------------------|---------|-----------|-------------|
| HBM (global) | 80 GB | ~400-500 ns | ~3.35 TB/s | device-wide |
| L2 | 50 MB | ~150-200 ns | ~12 TB/s | device-wide, partitioned |
| Shared / L1 | 228 KB / SM | ~30 ns | ~30 TB/s per SM | per thread block |
| Registers | 64 K × 32-bit / SM | 0 (compiler-managed) | unbounded | per thread |

The whole point of a high-performance kernel is to stage data so that the inner loop only touches registers and shared memory, and HBM is touched once per element. FlashAttention is built exactly around this principle.

### Warps and the SM

- A **warp** is 32 threads that execute the same instruction in lockstep (SIMT). All performance reasoning is "what does the warp do this cycle."
- An **SM** (Streaming Multiprocessor) runs many warps concurrently. Hopper SM has 128 KB of register file and 228 KB shared/L1.
- **Occupancy** = how many warps are resident on an SM. Higher occupancy hides memory latency but costs registers per thread. FlashAttention deliberately keeps occupancy modest (e.g. 2–4 warps per block, blocks per SM = 1–2) because per-thread register tiles are large.

### Coalescing

When the 32 threads of a warp issue loads that hit a contiguous 128-byte region of HBM, the hardware combines them into one transaction. If the addresses scatter, each thread triggers its own transaction and bandwidth collapses. Rule of thumb: lay out tensors in `[batch, head, seq, dim]` order (NHWC-like) so the fastest-changing axis matches the warp's thread index.

### Shared-memory bank conflicts

Shared memory is organised into 32 banks of 4-byte words. If two threads in the same warp read different addresses in the same bank, the access serialises. FlashAttention avoids this by carefully padding shared-memory tiles (`+8` or `+1` on the inner dimension) and by using the `ldmatrix` / `stmatrix` instructions on Hopper.

### Tensor cores

Tensor cores execute small fixed-shape matmuls per warp / warpgroup in one instruction:

- **HMMA** (Volta/Turing/Ampere): one warp issues `m16n8k16` fp16/bf16 MMA → 256 FMA in ~8 cycles.
- **WGMMA** (Hopper): one **warpgroup** (4 warps = 128 threads) issues an asynchronous `m64n{N}k16` MMA against shared memory descriptors.
- **TCGen5** (Blackwell): tensor cores now operate at warpgroup-cluster scale with FP4/FP6 support.

FlashAttention is a tensor-core engine wrapped in softmax glue. Most of the runtime is in MMA instructions. The rest is the rescale + exponentials around them.

### Warp shuffles

`__shfl_sync`, `__shfl_xor_sync`, etc. let threads in a warp read each other's registers in 1 cycle — no shared memory needed. Used for warp-level reductions (`m`, `ℓ` in the softmax recurrence) and for swizzling MMA inputs.

### `cp.async` and TMA

- `cp.async` (Ampere+) issues a DMA from HBM to shared memory that the warp can wait on later, overlapping with compute.
- **TMA** (Hopper) is the same idea at coarser granularity: a single instruction copies a multi-dimensional tile from HBM to shared memory, the whole warp / warpgroup gets it for free, and you commit/wait on a barrier.

FA2 leans on `cp.async`. FA3 leans on TMA. The reason FA3 is faster is largely about overlapping these copies with the MMA so the math units never stall.

---

## Build it

Two microbenchmarks. Both ship as standalone CUDA files in your course working directory.

### 1. Coalesced vs uncoalesced copy

```cpp
// coalesce.cu
__global__ void copy_coalesced(const float* __restrict__ x, float* __restrict__ y, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) y[idx] = x[idx];
}

__global__ void copy_strided(const float* __restrict__ x, float* __restrict__ y, int n, int stride) {
    int idx = (blockIdx.x * blockDim.x + threadIdx.x) * stride;
    if (idx < n) y[idx] = x[idx];
}
```

Run both at `n = 2^24`. The coalesced one should hit ~85–90% of HBM peak. The strided one (e.g. `stride = 32`) should drop by 10–20×. Compare under `ncu --metrics gpu__time_duration.sum,dram__bytes.sum`.

### 2. Warp-shuffle reduction

```cpp
// warp_reduce.cu
__inline__ __device__ float warp_sum(float v) {
    for (int off = 16; off > 0; off >>= 1)
        v += __shfl_down_sync(0xffffffff, v, off);
    return v;
}

__global__ void rowsum_via_shuffle(const float* x, float* out, int n) {
    int row = blockIdx.x;
    int tid = threadIdx.x;
    float acc = 0.f;
    for (int j = tid; j < n; j += blockDim.x) acc += x[row * n + j];
    acc = warp_sum(acc);
    if ((tid & 31) == 0) atomicAdd(&out[row], acc);
}
```

This is structurally the same code FlashAttention uses to reduce `m` and `ℓ` across threads of a warp before broadcasting back. Time it against a naive shared-memory reduction; on Hopper the shuffle version should be ~2× faster for warp-local accumulators.

---

## Use it in the real stack

In `flash-attention/csrc/flash_attn/src/`:

- `softmax.h`: search for `quad_shfl_xor_sync` / `Allreduce` — these are the warp shuffles that compute `m` and `ℓ` reductions inside a warp.
- `block_info.h` and `kernel_traits.h`: `kBlockM`, `kBlockN`, `kNWarps`, `kStages` — the tile sizes, warps per block, and `cp.async` stage count.
- `mask.h` and `softmax.h` again: the rescale-and-add update of `tOrO` after each `tOrP` — this is the FP register-level realisation of the recurrence from Lecture 2.

Pick one CUDA file and annotate, in your own notes, where each of these is happening. Do not move on to Lecture 5 until you can point at the MMA call, the shuffle reduce, and the `cp.async` issue / commit / wait in real source.

---

## Measure it

Standard Nsight Compute metrics worth memorising:

| Metric | Meaning |
|--------|---------|
| `sm__throughput.avg.pct_of_peak_sustained_elapsed` | Overall SM utilisation |
| `dram__bytes.sum` | HBM traffic |
| `lts__t_sectors.sum` | L2 traffic |
| `sm__warps_active.avg.pct_of_peak_sustained_active` | Achieved occupancy |
| `smsp__inst_executed_pipe_tensor_op_hmma.sum` | Tensor-core HMMA instructions |
| `smsp__sass_average_data_bytes_per_sector_mem_global.pct` | Coalescing efficiency |

For your two microbenchmarks, report at minimum: kernel time, achieved HBM bandwidth, achieved occupancy. For the warp-shuffle reduction, also report tensor-core instruction count (should be zero — it is a pure CUDA-core kernel).

---

## Ship it

Drop into `flash-attn-course/`:

1. `coalesce.cu` and a `coalesce_bench.txt` with the two kernel times and bandwidths.
2. `warp_reduce.cu` and a `reduce_bench.txt` comparing shuffle vs shared-memory reduce.
3. A one-page Markdown cheat sheet mapping each term in the FlashAttention kernel (`tOrO`, `tOrP`, `tOrS`, `cp.async.commit_group`, `__shfl_xor_sync`, `wgmma.mma_async`) to a one-sentence explanation in your own words.

If you have those three, you are ready for Lecture 5.

---

## Related pages

- [Lecture 3 — FlashAttention-1 algorithm](Lecture%2003%20-%20FlashAttention-1%20Algorithm.md)
- [Lecture 5 — Repo anatomy and Python / CUDA API](Lecture%2005%20-%20Repo%20Anatomy%20and%20Python%20CUDA%20API.md)
- NVIDIA Nsight Compute docs: <https://docs.nvidia.com/nsight-compute/>
- PTX ISA: <https://docs.nvidia.com/cuda/parallel-thread-execution/>
