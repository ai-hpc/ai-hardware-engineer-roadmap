# Lecture 6 — FlashAttention-2

**Parent:** [FlashAttention Course](Guide.md)

**One-line purpose:** Explain why FA2 is ~2× faster than FA1 at the same algorithm, by understanding the work-partitioning changes — sequence parallelism, per-head parallelism, warp partitioning, and reduced shared-memory traffic.

**Prerequisites:** Lectures 1–5. You should have read `flash_fwd_kernel.h` end to end.

**Artifact:** A benchmark table comparing PyTorch SDPA, FA2, and a "FA1-style" reference (you can fake this with a Triton implementation) across a `(seq_len × head_dim)` sweep on your GPU.

---

## Why it matters

FA1 had the algorithmic breakthrough; FA2 had the engineering breakthrough. On a typical attention shape, FA2 sits at **50–70% of the theoretical max GEMM throughput** the GPU can deliver — the gap to GEMM is where the few remaining non-matmul FLOPs (softmax exponentials, rescales) live. If you understand the four changes FA2 made you also understand why FA3 went the way it did on Hopper.

---

## Mental model

### What FA1 left on the table

FA1's inner loop was: **for each query tile, loop over all KV tiles, update `(m, ℓ, O)`**. That has three inefficiencies on real GPUs:

1. **Few independent thread blocks.** Grid was `(num_q_tiles, num_heads, batch)`. For small batch × short seq × few heads, you cannot fill the GPU.
2. **Per-block reductions hit shared memory.** Each tile's `m_ij` and `ℓ_ij` reductions wrote to shared memory and synced threads.
3. **Non-matmul FLOPs ran on the same warps that did MMAs.** The exponential / rescale work stalled the matmul pipeline.

FA2 fixes all three.

### FA2 change 1: parallelise over the sequence dimension

In FA1, work over `seq_len` was serialised inside one block (the outer-`i` loop). FA2 makes that loop a **grid dimension**. Grid is now `(num_q_tiles, num_heads, batch)` *all of which contribute independent thread blocks*, plus the inner loop over KV tiles is the only remaining serial loop.

For small batch × few heads × long sequence (typical of fine-tuning / inference prefill), this is the single biggest win. You go from 8 blocks to 128 blocks; the GPU fills.

### FA2 change 2: parallelise warps within a block over heads/KV

FA1 had all warps in a block working on the same query tile. FA2 splits warps so different warps within a block handle different KV columns of the same query tile. The advantage: each warp's `MMA` writes go to different rows of the output accumulator, **no inter-warp sync** is needed for the GEMM portion. The block-level reduction only happens at the end to combine partial `(m, ℓ)` across warps.

### FA2 change 3: keep softmax stats in registers, not shared memory

FA1 spilled `m_ij`, `ℓ_ij` to shared memory between the rowmax and the rescale. FA2 keeps both in registers and uses warp shuffles (`__shfl_xor_sync`) to do the row-wise reductions. Shared memory is reserved for the matmul tiles.

The practical effect: shared memory pressure drops, you can fit larger `B_r × d` and `B_c × d` tiles, and the warps never stall on `__syncthreads()` waiting for the softmax math.

### FA2 change 4: rescale `O` less often

FA1 rescaled the running `O_i` accumulator every inner iteration with `rescale_old = exp(m_old - m_new)`. FA2 observes that you can defer the rescale: just keep `O` "in the wrong scale" until the end of the inner loop and rescale once at the epilogue. This drops one multiply across the whole `O` accumulator per inner iteration — meaningful at large `d`.

The math still works because:

```
O_final = exp(m_last - m_global) · O_accumulated_unrescaled
ℓ_final = … (same trick)
O = O_final / ℓ_final
```

You only need one final divide and one final scalar multiply per output row.

### Net effect

These four changes together get FA2 to ~70% of a pure GEMM at the same `(M, N, K)` shape on A100/H100 — close to the practical ceiling, because the softmax exponentials cost a non-trivial fraction of `O(N·d)` non-matmul FLOPs.

---

## Build it

You will not reimplement FA2; you will read it and benchmark it.

### Reading task

Open `csrc/flash_attn/src/flash_fwd_kernel.h`. Locate:

1. The grid launch dimensions (look in `flash_fwd_launch_template.h` for `dim3 grid(...)`). Confirm it is `(num_m_blocks, num_heads, batch)`. **This is FA2 change 1.**
2. The warp partitioning of the `S = QKᵀ` accumulator (`tSrS`). The `Tiled_mma` type tells you how warps are split — search for `TiledMma` and the `using` aliases in `kernel_traits.h`. **This is FA2 change 2.**
3. `softmax_rescale_o_` and `softmax_template` in `softmax.h` — the warp-shuffle reductions, not shared-memory reductions. **This is FA2 change 3.**
4. The deferred final divide-by-`ℓ` at the end of the kernel — search for the epilogue block that writes `tOrO / lse`. **This is FA2 change 4.**

Write a short note linking each FA2 paper change to a concrete code location. Without that note, you cannot debug FA2 patches.

### Benchmark task

Use the repo's `benchmarks/benchmark_flash_attention.py`. Run a sweep at fp16:

```
for seqlen in 512 1024 2048 4096 8192 16384; do
  for headdim in 64 128; do
    python benchmarks/benchmark_flash_attention.py \
      --mode fwd --batch_size 2 --nheads 16 \
      --seqlen $seqlen --headdim $headdim \
      --dtype fp16 --causal
  done
done | tee fa2_sweep.txt
```

Then write `plot_fa2_sweep.py` to produce a CSV with `(seqlen, headdim, sdpa_ms, fa2_ms, fa2_tflops, sdpa_tflops)` columns and a single-figure plot of TFLOPs vs seqlen for each backend.

You should see:

- SDPA's `math` backend slowing down quadratically.
- FA2 staying flat or improving slightly as seqlen grows (because per-launch overhead dilutes).
- FA2 reaching 40–60% of your GPU's peak fp16 TFLOPS at `(seqlen ≥ 4096, headdim = 128)`.

---

## Use it in the real stack

PyTorch's `scaled_dot_product_attention` will pick FA2 under these conditions: `dtype ∈ {fp16, bf16}`, head dim a supported value (`{64, 128, 256}` depending on version), `dropout_p == 0` (during inference), no custom mask other than causal or none, and on a supported architecture (sm80+). When any of those fail, SDPA silently falls back to the math kernel. Use:

```python
from torch.nn.attention import sdpa_kernel, SDPBackend
with sdpa_kernel([SDPBackend.FLASH_ATTENTION]):
    ...
```

to force-fail loudly if the conditions are not met. Pair this with a benchmark and you can see the gap to `math` for yourself.

---

## Measure it

Per shape, report:

- Kernel time (median over ≥ 20 runs after 10 warmup runs).
- Achieved TFLOPs (`4 · B · H · N² · d / time`).
- Achieved HBM bandwidth (`bytes / time`).
- Ratio to GPU peak fp16 TFLOPs.

For at least one shape, dump the Nsight Compute report and verify:

- Tensor-core instructions dominate (`smsp__inst_executed_pipe_tensor_op_hmma.sum` is large).
- DRAM traffic is far below `O(N²)` bytes.
- Occupancy is ~1–2 blocks per SM — FA2 deliberately uses big per-thread register tiles, so high occupancy is not a target.

---

## Ship it

Drop into `flash-attn-course/`:

1. `fa2_code_notes.md` — the four FA2 changes mapped to file:line in the FA repo.
2. `fa2_sweep.csv` and `fa2_sweep.png` — the benchmark output across `(seqlen, headdim)`.
3. One Nsight Compute report at a single FA2 shape (`fa2_ncu_4096_d128.ncu-rep` or similar) showing tensor-core utilisation and HBM traffic.

These artifacts are what you will diff against FA3 in Lecture 9.

---

## Related pages

- [Lecture 5 — Repo anatomy](Lecture%2005%20-%20Repo%20Anatomy%20and%20Python%20CUDA%20API.md)
- [Lecture 7 — Backward pass and validation](Lecture%2007%20-%20Backward%20Pass%20and%20Validation.md)
- [Lecture 9 — Hopper / FA3 / FA4](Lecture%2009%20-%20Hopper%20FA3%20FA4.md)
- FA2 paper: <https://tridao.me/publications/flash2/flash2.pdf>
