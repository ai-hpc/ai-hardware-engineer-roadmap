# Lecture 9 — Hopper / FlashAttention-3 / FlashAttention-4

**Parent:** [FlashAttention Course](Guide.md)

**One-line purpose:** Understand the Hopper-specific (and Blackwell-targeted) features that FA3 / FA4 exploit — WGMMA, TMA, warp specialisation, ping-pong scheduling, FP8 — and where to find each one in the source.

**Prerequisites:** Lectures 1–8. Strong familiarity with FA2 source.

**Artifact:** FA3 vs FA2 benchmark on H100 / H200, plus an annotated source map identifying the WGMMA, TMA, and warp-specialised pipeline stages in `csrc/flash_attn_hopper/`.

---

## Why it matters

FA3 is the first attention kernel that gets within 5–10% of theoretical peak on H100/H200 for the shapes that matter (long-context training and prefill). The reason is not algorithmic — it is the FA2 algorithm scheduled on top of Hopper's new instructions and new memory primitives. FA4 (currently a blog post and prototype) pushes the same ideas onto Blackwell with FP4/FP6. If you understand the FA3 changes, FA4 is "the same play on better hardware."

You will also need this understanding to read TransformerEngine's attention path and the Hopper / Blackwell variants in cuDNN.

---

## Mental model

### Hopper hardware that FA3 actually uses

| Feature | What it does | Why FA3 cares |
|---------|--------------|---------------|
| **WGMMA** (`wgmma.mma_async`) | Warpgroup (4 warps = 128 threads) issues an async MMA reading shared-memory operands via descriptors. | Bigger matmul per instruction; async means math overlaps with the next load. |
| **TMA** (Tensor Memory Accelerator) | One instruction copies a multi-dim tile from HBM → shared memory with built-in swizzle. | Replaces complex `cp.async` boilerplate, frees registers, and the wait is a barrier rather than per-thread. |
| **Distributed shared memory** | Threads in one CTA can read another CTA's shared memory in the same cluster. | Enables warp-specialised pipelines that exchange tiles without round-tripping through HBM. |
| **Async PTX barriers** (`mbarrier`) | Lightweight barriers for producer / consumer warps. | Coordinates WGMMA producers and softmax consumers without `__syncthreads()`. |
| **FP8 tensor cores** | E4M3 / E5M2 MMA at 2× the rate of fp16. | Lets FA3 burn fewer cycles per MMA on the prefill GEMM, leaving headroom for softmax. |

### Warp specialisation in FA3

Where FA2 had homogeneous warps all doing the same MMA + softmax sequence, FA3 splits warps in a block into roles:

- **Producer warps**: issue TMA loads of the next `K, V` tile. Wait on `mbarrier`. Hand off to consumer.
- **Consumer warps (math)**: do `WGMMA(Q, Kᵀ)`, the softmax (rescale, exp, rowmax/rowsum), and `WGMMA(P, V)`. Wait on producer barrier for the next tile.

This is exactly the producer/consumer pipeline pattern that CUTLASS uses for GEMM. It hides memory latency *inside the block*, not just across blocks.

### Ping-pong scheduling

Two consumer warpgroups alternate: while warpgroup A does its softmax + epilogue for tile `j`, warpgroup B does the WGMMA for tile `j+1`. Because softmax / WGMMA use different pipelines (CUDA cores vs tensor cores), they can run concurrently on the same SM. The result: tensor cores stay busy ~95% of the time instead of ~60% for FA2.

### Overlapping softmax with GEMM

The non-matmul FLOPs of softmax (the exponentials, the rescales) used to stall the warp. On Hopper, with ping-pong + warp specialisation, those FLOPs run on the CUDA cores in parallel with the WGMMAs running on tensor cores. The cost of softmax effectively disappears for long-enough KV tiles.

### FP8

FA3 supports FP8 for the prefill `Q · Kᵀ` and `P · V` matmuls. Accumulation is in fp32 to preserve numerics for the softmax. The cost is calibration (per-tensor scaling factors `s_q`, `s_k`, `s_v`); the benefit is ~1.8× speedup over bf16 for the same shape on H100.

### CuTeDSL

FA3 is written using CUTLASS's CuTe DSL, not plain CUDA C++. CuTe gives you composable layout algebra (`Layout`, `Stride`, `Shape`), TMA descriptor builders, and MMA atom selectors that compile down to the right WGMMA instructions. Reading FA3 source = reading CuTe code. There is no avoiding it.

The headline CuTe primitives in FA3:

- `cute::Tensor` — a "logical tensor" with a layout that may live in registers / SMEM / GMEM.
- `cute::copy(...)` — fires the right load/store instruction (TMA, `ldmatrix`, `cp.async`) for the layout.
- `cute::gemm(...)` — fires the right WGMMA/MMA atom.
- `cute::TiledMma` — describes how warps within a warpgroup partition the MMA result.

### FA4 sketch (Blackwell)

FA4 is currently a blog post; the public repo is the FA3 codebase with new dispatch paths in progress. The main bets:

- **TCGen5** tensor cores with FP4 / FP6 support → another 1.5–2× over FP8 if you can stomach the numerics.
- **Larger cluster shared memory** → larger blocks per CTA cluster, more reuse.
- **More aggressive warp specialisation** with three-stage pipelines (producer, math, epilogue).

Until FA4 is public-stable, treat this as a preview. The mental model carries over.

---

## Build it

### Reading task

Open `csrc/flash_attn_hopper/`. Locate:

1. **TMA descriptor construction** — search for `Sm90_TMA_LOAD` or `make_tma_copy(...)`. This is where Q, K, V loads are wired up.
2. **WGMMA instantiation** — search for `wgmma` or `SM90_64x*x16_F32BF16BF16_SS` (the SM90 MMA atom for bf16 → fp32). The `cute::gemm` call inside the consumer warps fires these.
3. **Warp specialisation** — search for `warpgroup_idx` or `cooperative_warp_specialize_blockwise`. The `if (warpgroup_idx == ProducerWarpGroup) { ... } else { ... }` block is the producer/consumer split.
4. **Barriers** — `mbarrier` usage in `producer_acquire` / `consumer_wait`. This is the FA3 ping-pong machinery.
5. **FP8 paths** — search `e4m3` or `fp8`. There is a separate fwd kernel for FP8 with its own dispatch table.

Write your source map as `fa3_code_notes.md`. Compare it line-for-line to your FA2 map from Lecture 6 — that diff is the FA3 contribution.

### Benchmark task

```python
# fa3_vs_fa2.py
import torch, time
from flash_attn import flash_attn_func                  # FA2
from flash_attn.flash_attn_interface import flash_attn_func_v3  # FA3 (Hopper only)

shapes = [(2, 16, N, 128) for N in [1024, 4096, 8192, 16384, 32768]]
def time_fn(fn, *args, **kw):
    for _ in range(5): fn(*args, **kw)
    torch.cuda.synchronize()
    s = torch.cuda.Event(enable_timing=True); e = torch.cuda.Event(enable_timing=True)
    s.record()
    for _ in range(20): fn(*args, **kw)
    e.record(); e.synchronize()
    return s.elapsed_time(e) / 20

for (B, H, N, D) in shapes:
    q = torch.randn(B, N, H, D, device="cuda", dtype=torch.bfloat16)
    k = torch.randn_like(q); v = torch.randn_like(q)
    t2 = time_fn(flash_attn_func, q, k, v, causal=True)
    t3 = time_fn(flash_attn_func_v3, q, k, v, causal=True)
    print(f"N={N:>5}  FA2={t2:7.3f} ms  FA3={t3:7.3f} ms  speedup={t2/t3:.2f}x")
```

On H100/H200 you should see ~1.5–2.0× speedup at long sequence lengths. The speedup grows with `N` because FA3's ping-pong scheduling matters more when the inner KV loop is long. If you have access to FP8 (sm89+), repeat with the FP8 path.

---

## Use it in the real stack

- **TransformerEngine** (NVIDIA): wraps FA3 for Hopper training. Read `transformer_engine/pytorch/attention.py` to see the dispatch.
- **vLLM with `--attention-backend FLASH_ATTN_VLLM_V1`** picks FA3 on Hopper automatically.
- **cuDNN's SDPA** has a Hopper-tuned attention path that uses many of the same primitives independently of FA3.

For inference, the FA3 prefill kernel is the right choice on Hopper; the FA decode (`with_kvcache`) is still using the FA2 codebase but is being ported to FA3 incrementally as of mid-2026.

---

## Measure it

For each (FA2, FA3, FP8) row:

- Kernel time (ms).
- Achieved TFLOPS (`4 · B · H · N² · D / time`).
- Achieved % of GPU peak (bf16 ~989 TFLOPs on H100/H200, FP8 ~1978 TFLOPs).
- Achieved HBM bandwidth (should be far below peak — FA3 is compute-bound at large N).

Run Nsight Compute on FA3 and look for:

- WGMMA instruction count (`smsp__inst_executed_pipe_tensor_op_wgmma.sum`) — should be dominant.
- TMA load count (`tma_load` metrics) — non-zero only on Hopper.
- Tensor pipe utilization → 90%+ at long N.

---

## Ship it

Drop into `flash-attn-course/`:

1. `fa3_code_notes.md` — source map for TMA / WGMMA / warp-specialisation / FP8 paths.
2. `fa3_vs_fa2.csv` and `fa3_vs_fa2.png` showing the speedup curve.
3. One Nsight Compute report at the best FA3 shape (`fa3_ncu_32k_d128.ncu-rep`) with WGMMA + TMA metrics highlighted.

These are also the inputs to the capstone.

---

## Related pages

- [Lecture 6 — FlashAttention-2](Lecture%2006%20-%20FlashAttention-2.md)
- [Lecture 8 — Inference path](Lecture%2008%20-%20Inference%20Path%20KV%20Cache%20and%20Decode.md)
- [Lecture 10 — Capstone](Lecture%2010%20-%20Capstone.md)
- FA3 blog: <https://tridao.me/blog/2024/flash3/>
- CUTLASS / CuTe DSL: <https://github.com/NVIDIA/cutlass>
- PTX ISA: <https://docs.nvidia.com/cuda/parallel-thread-execution/>
