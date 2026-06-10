# Lecture 02 - The Kernel-Language Explosion: Tiles as the New ISA

**Collection:** [MLSys Deep Dives](README.md) | **Previous:** [← Lecture 01](Lecture-01.md) | **Next:** [Lecture 03](Lecture-03.md)

---

Lecture 1 said every token's cost is set by tokens-per-second, and tokens-per-second starts at the kernel. So: how does a fast GPU kernel actually get *written* in 2026? The answer changed more in the last three years than in the prior decade, and it changed in one direction — **away from scalar threads and toward tiles**.

This lecture is the map of that change: why the tile became the unit of GPU programming, and a working tour of the languages fighting to own it — **Triton, CUTLASS/CuTe/cuTile, ThunderKittens, TileLang** — arranged on the one axis that actually distinguishes them: how much control they hand you, versus how much they decide for you.

---

## Learning objectives

By the end of this lecture, you should be able to:

1. Explain why the scalar SIMT (thread-centric) model broke down, and why the **tile** is the natural unit on Tensor-Core hardware.
2. Place the major kernel tools on the **ease ↔ control** spectrum and name what each decides for you vs. exposes.
3. Write and read a **Triton** tile kernel, and explain its role under `torch.compile`/Inductor.
4. Distinguish NVIDIA's three on-ramps — **CUTLASS C++ + CuTe**, **CuTe DSL**, **cuTile** — and what "Tile IR" is.
5. Say what **ThunderKittens** and **TileLang** each optimize for, and when you'd reach for them over Triton.
6. Connect a kernel choice back to tokens/s and `$/Mtok` from Lecture 1.

---

## 1. Why the thread model broke

For a decade, CUDA's mental model was the **scalar thread**: you wrote what one thread does, launched millions, and reasoned about warps, shared memory, and synchronization by hand. That model matched the hardware — lots of simple scalar lanes.

Then the hardware stopped being scalar lanes. Modern accelerators are dominated by:

```text
   Tensor Cores       do a whole MATRIX-TILE multiply-accumulate per instruction (e.g. 16×16×16)
   TMA                (Tensor Memory Accelerator) moves whole TILES async between HBM and SRAM
   warp specialization different warps run producer (load) vs consumer (compute) pipelines
   async pipelines    overlap tile-load and tile-compute across many stages
```

Against that hardware, the scalar thread is the *wrong abstraction*. The natural unit of work is no longer "what one thread computes" — it is **"what happens to this tile"**: load a tile, matmul-accumulate it, store a tile. Writing that as thousands of coordinated scalar threads is error-prone boilerplate the compiler should handle. So the field converged, independently and explicitly, on the **tile** as the programming primitive.

```text
   tile abstraction:  you describe TILES + a GRID of them.
                      the compiler handles thread partitioning, shared memory,
                      data movement (TMA), and (often) the pipeline.

   lineage:  CUDA C++ (2012) → Triton (2019) → CuTe (2023)
                → ThunderKittens (2024) → TileLang (Jan 2025) → cuTile (2025)
```

This is the single most important trend in GPU programming right now, and it is *why* there are suddenly five competing kernel languages: they are all racing to own the tile.

---

## 2. The spectrum that organizes everything

Do not memorize five tools as a flat list. Arrange them on **one axis — ease/productivity vs. control/peak-performance** — and the whole landscape snaps into place.

```text
  EASE / PRODUCTIVITY  ◄──────────────────────────────────────────►  CONTROL / PEAK PERF
  ┌──────────┬───────────────────────────┬───────────────────────────────┬─────────────────┐
  │ TensorRT │ cuTile · Triton · Pallas   │ TileLang · CuTe DSL · TKittens │ CUTLASS C++/CUDA│
  │ (closed, │ (compiler decides thread   │ (you annotate scheduling &      │ (you write every│
  │  builds  │  & memory layout for you)  │  layout; near hand-tuned perf)  │  thread/barrier)│
  │ engine)  │                            │                                 │                 │
  └──────────┴───────────────────────────┴───────────────────────────────┴─────────────────┘
        ▲
        └── and ALONGSIDE the DSLs: autoscheduling compilers (TVM, IREE) that SEARCH the
            schedule space instead of asking you to choose — covered in Lecture 03.
```

The trade is always the same: the more the compiler decides, the faster you ship and the more portable you are — but the closer you sit to a performance ceiling someone else set. The more you control, the closer to peak you can get — at the cost of effort and target lock-in. A senior engineer picks a *point on this axis per kernel*, not one tool for life. The 95%-of-cases kernel goes in Triton; the one attention kernel that dominates your cost budget might go in ThunderKittens or CuTe DSL.

---

## 3. Triton — the mindshare leader

**Triton** (OpenAI, public since 2021, MLIR-based since 2.0) is the tile language most people mean when they say "I wrote a kernel." You write a Python function decorated with `@triton.jit` that operates on **block-level tiles**; the compiler handles coalescing, shared memory, and intra-block scheduling.

```python
import triton
import triton.language as tl

@triton.jit
def fused_add_relu(x_ptr, y_ptr, out_ptr, n, BLOCK: tl.constexpr):
    pid  = tl.program_id(0)                    # which tile this program instance owns
    offs = pid * BLOCK + tl.arange(0, BLOCK)   # the TILE of indices, not one element
    mask = offs < n
    x = tl.load(x_ptr + offs, mask=mask)       # load a whole tile
    y = tl.load(y_ptr + offs, mask=mask)
    out = tl.maximum(x + y, 0.0)               # fused add + ReLU, on the tile
    tl.store(out_ptr + offs, out, mask=mask)
```

Note what you did *not* write: no thread indices, no `__shared__`, no `__syncthreads()`. You described a tile; Triton mapped it to threads. For matmul you'd use `tl.dot(a_tile, b_tile)`, which lowers to Tensor Cores. `@triton.autotune` sweeps block sizes, `num_warps`, and `num_stages` to pick a config.

Why Triton leads mindshare, concretely:

* **It is the codegen target of `torch.compile`.** PyTorch's TorchInductor (the default backend) **generates Triton** for GPU. Every time someone runs `torch.compile`, Triton kernels are produced underneath. That makes it the de-facto kernel IR of mainstream PyTorch.
* **It is genuinely cross-vendor.** NVIDIA, AMD (ROCm), Intel (XPU), and an experimental CPU backend — one kernel, many targets. NVIDIA even shipped a **Tile-IR backend for Triton** so it can lower to NVIDIA's new tile virtual-ISA (more on that next).

The honest critique (made loudly by Chris Lattner / Modular): Triton is a *compiler-decides* model, and independent measurements have shown a meaningful gap — on the order of **~20% on H100** — versus hand-optimized CUDA for the hardest kernels, with weaker portability across GPU generations and lagging support for the newest features (FP8, TMA) until the compiler catches up. Other studies are kinder (62–101% of cuBLAS across platforms with zero arch-specific tuning). The truth is in between, and it is exactly why the control-end tools in §5–6 exist.

---

## 4. NVIDIA's answer: CUTLASS, CuTe, CuTe DSL, and cuTile

NVIDIA's response to "everyone is writing tiles in not-CUDA" was to provide *official* tile on-ramps — and there are confusingly several, which you must keep straight.

* **CUTLASS** — the long-standing open C++ template library for peak GEMM/attention on Tensor Cores. This is how NVIDIA itself writes fast kernels. Max control, max effort.
* **CuTe** — the **layout algebra** at the heart of CUTLASS 3.x: composable `Layout`/`Tensor`/"atom" objects describing the thread-and-data hierarchy as shapes + strides. It is the formal vocabulary the newer DSLs reuse. When you hear "CuTe," think *the math of who-owns-which-element*.
* **CuTe DSL** (new with CUTLASS 4.0, GTC 2025) — NVIDIA's first **Python** kernel DSL, **low-level** and fully consistent with CuTe C++. You get full thread/data/layout control from Python, JIT-compiled through MLIR + ptxas, with claimed **C++-parity performance** and **~100× faster compilation** than C++ templates. This is the *control end*, in Python.
* **cuTile** (a.k.a. CUDA Tile, GTC 2025, shipping with CUDA 13.1) — a **separate, higher-level** Python tile DSL. You write tile kernels; the compiler abstracts block parallelism, memory movement, and thread partitioning. This is the *productivity end* — and it is widely read as a **direct response to Triton**. (One ex-CUDA architect: "it's hard not to suspect cuTile was developed directly to counter Triton.")

Underneath cuTile is **Tile IR** — a new **virtual ISA for tile programming**, effectively "PTX for tiles." Crucially, NVIDIA built a Tile-IR backend *for Triton too*, so Triton can lower through it. The strategic read: NVIDIA is trying to **reclaim the tile abstraction inside the CUDA platform**, offering both a control on-ramp (CuTe DSL) and a productivity on-ramp (cuTile), both first-class on Blackwell.

The catch is the obvious one: all of it is **NVIDIA-only**. You trade Triton's portability for deeper hardware integration and (at the CuTe DSL end) near-C++ peak.

---

## 5. ThunderKittens — minimal tiles, maximal attention

**ThunderKittens** (Stanford Hazy Research, 2024) takes a different bet: stay **inside CUDA C++** as a small embedded DSL (a header/template library), and ask *how small can the tile abstraction be and still hit SOTA?*

Its primitive is literally a **tile sized to the Tensor Core**. The library abstracts the repetitive plumbing — tile layouts, shared-memory allocation, register fragments, TMA tensor maps, Tensor Core descriptors — while keeping you close enough to reason about data movement and scheduling yourself.

```text
   ThunderKittens mental model:
   ┌──────────────────────────────────────────────────────────┐
   │  declare register/shared TILES (16×16-ish, TC-shaped)      │
   │  TMA-load input tiles  →  mma(acc, a_tile, b_tile)  →  store│
   │  you still schedule the pipeline; TK removes the boilerplate│
   └──────────────────────────────────────────────────────────┘
```

It is **known for fast attention** — its origin motivation was "FlashAttention is ~1200 lines; how compact can this be?" — and it is used in production by **Together AI, Jump Trading, and Cursor**. **ThunderKittens 2.0** (early 2026) brought full Blackwell support and low-precision **MXFP8 / NVFP4**. There is even a Metal/MLX port ("ThunderMittens").

When to reach for it: the *one* kernel that dominates your cost (usually attention), where you want near-hand-tuned performance and full scheduling control but refuse to write 1200 lines of raw CUTLASS. It is research-led with a smaller ecosystem than Triton — a scalpel, not a default.

---

## 6. TileLang — decoupling schedule from dataflow

**TileLang** (open-sourced Jan 2025; from the `tile-ai` group with Microsoft Research and Peking University; built **on Apache TVM's TIR infrastructure**) makes the control-vs-productivity trade *explicit and adjustable* inside one language.

Its core idea: **separate the dataflow (what is computed) from the scheduling space (thread binding, memory layout, `tensorize`, software pipelining), and expose the scheduling as overridable annotations** — with automatic layout inference filling in what you don't specify.

```text
   Triton:    you write dataflow; the compiler picks ALL scheduling.        (less control)
   CUTLASS:   you write dataflow AND every scheduling/layout detail.         (all control, all effort)
   TileLang:  you write dataflow; you OVERRIDE the scheduling you care        (control where it pays,
              about and let inference handle the rest.                         automation where it doesn't)
```

That positions it deliberately between Triton (productivity) and CUTLASS (control), and it targets a wide backend set — **CUDA, ROCm/HIP, Metal, WebGPU, and CPU** — with a CuTe DSL backend added late 2025. It is part of a broader stack from the same group: **TileLang** (author kernels) + **TileScale** + the **TileRT** runtime (which you'll meet in Lecture 3 and again in Lecture 6 as the engine behind a notable throughput milestone).

When to reach for it: you need **more scheduling control than Triton and more portability than CUTLASS**, e.g. shipping the same hand-tuned kernel across NVIDIA *and* AMD *and* Apple. The cost is a smaller ecosystem and a heavier (TVM-based) dependency than Triton.

*(Honorable mention: **JAX Pallas** — the same tile + `grid` + `BlockSpec` idea, JAX-native, lowering to Triton on GPU and Mosaic on TPU. If you live in JAX/TPU land, Pallas is your tile DSL.)*

---

## 7. Choosing — a senior engineer's table

| Tool | Owner | Model | Lives where | Targets | Reach for it when |
|---|---|---|---|---|---|
| **Triton** | OpenAI | block/tile, `@jit` Python | under `torch.compile` | NV / AMD / Intel / CPU | default — portable, productive, 95% of kernels |
| **cuTile** | NVIDIA | Python tile, Tile IR | CUDA 13.1 | NVIDIA | NVIDIA-only, want Triton-like ease + deep CUDA integration |
| **CuTe DSL** | NVIDIA | Python, CuTe-consistent | CUTLASS 4.x | NVIDIA | want C++-parity peak from Python, fast iterate |
| **CUTLASS C++** | NVIDIA | C++ templates + CuTe | hand-written | NVIDIA | building a library kernel, need absolute peak |
| **ThunderKittens** | Stanford | tile = TC-shaped, in CUDA | C++ header lib | NV (Blackwell), Metal | the one attention kernel that dominates cost |
| **TileLang** | tile-ai / MSR / PKU | tiled, schedule⊥dataflow | on TVM | NV / AMD / Metal / WebGPU / CPU | hand-tuned *and* multi-vendor portable |
| **Pallas** | Google | tile + `BlockSpec` | JAX | GPU (Triton) / TPU (Mosaic) | you're in JAX / on TPU |

And the meta-point, which is contested and worth knowing both sides of: the convergence on tiles is **not** consolidation. NVIDIA's cuTile/Tile-IR is an attempt to pull the abstraction back into CUDA; Modular's Lattner argues the result is *fragmentation* — "1,001 ways to write CUDA kernels in Python," Python-eDSLs that "look like Python but aren't" — which is the pitch for **Mojo as a real language** instead of an eDSL (next lecture). Both readings are defensible. As an engineer, your job is not to pick the winning ideology; it is to put each kernel at the right point on the §2 axis.

---

## 8. Measure it — tie the kernel back to the token

A kernel is not done when it runs; it is done when you know its effect on tokens/s. The loop:

```text
   write tile kernel  →  benchmark GFLOP/s and % of roofline peak
                      →  swap it into the model's hot path
                      →  measure end-to-end tokens/s delta
                      →  recompute $/Mtok  (Lecture 1, §6)
```

A 1.5× faster attention kernel is interesting; a 1.5× faster attention kernel that lifts end-to-end decode tokens/s by 1.2× and drops `$/Mtok` from $0.56 to $0.47 is **shippable, defensible work**. Always carry the number to the last step. The kernel is the means; the token is the end.

---

## 9. Mini-lab: one kernel, three points on the axis

Pick a single op that matters (fused `add+RMSNorm`, or a small matmul).

1. **Productivity point:** write it in **Triton** with `@triton.autotune`. Record GFLOP/s and % of roofline.
2. **Control point:** if you have the hardware, reimplement in **CuTe DSL** or **ThunderKittens** (or hand-CUDA). Record the same.
3. **Compiler point (preview of Lec 3):** let `torch.compile` generate a Triton kernel for the same op and compare.
4. **Connect:** put the best kernel in a model's hot path and measure the **end-to-end tokens/s and `$/Mtok`** delta versus the framework default.

Deliverable: a table of `{Triton, control-tool, torch.compile}` × `{GFLOP/s, % peak, end-to-end tokens/s, $/Mtok}`, plus one paragraph on where each sat on the ease↔control axis and whether the control was worth the effort. That last judgment — *was the control worth it* — is the entire skill of this lecture.

---

## Key takeaways

- The **scalar thread model broke** because Tensor Cores, TMA, and warp-specialized pipelines made the **tile** the natural unit of work. Every modern kernel language is a race to own the tile.
- Organize the tools on **one axis: ease/productivity ↔ control/peak**. Pick a point per kernel, not a tool for life.
- **Triton** leads mindshare: tile-in-Python, the codegen target of `torch.compile`, genuinely cross-vendor — at a ~20%-on-the-hardest-kernels gap vs hand-tuned CUDA.
- **NVIDIA's on-ramps**: CUTLASS C++/CuTe (peak), **CuTe DSL** (C++-parity from Python), **cuTile** (Triton-like productivity) — all NVIDIA-only, over the new **Tile IR** ("PTX for tiles").
- **ThunderKittens** = minimal tiles in CUDA, the attention scalpel (used by Together, Cursor). **TileLang** = decouples scheduling from dataflow, hand-tuned *and* multi-vendor, built on TVM.
- A kernel is finished when you've measured its effect on **end-to-end tokens/s and `$/Mtok`** — not at "it runs."

---

## References

- OpenAI Triton: [https://openai.com/index/triton/](https://openai.com/index/triton/) · repo [https://github.com/triton-lang/triton](https://github.com/triton-lang/triton)
- NVIDIA, "Achieve CUTLASS C++ performance with Python — CuTe DSL": [https://developer.nvidia.com/blog/achieve-cutlass-c-performance-with-python-apis-using-cute-dsl/](https://developer.nvidia.com/blog/achieve-cutlass-c-performance-with-python-apis-using-cute-dsl/)
- NVIDIA, "Simplify GPU programming with CUDA Tile (cuTile) in Python": [https://developer.nvidia.com/blog/simplify-gpu-programming-with-nvidia-cuda-tile-in-python/](https://developer.nvidia.com/blog/simplify-gpu-programming-with-nvidia-cuda-tile-in-python/)
- ThunderKittens (Stanford Hazy Research): [https://github.com/HazyResearch/ThunderKittens](https://github.com/HazyResearch/ThunderKittens) · blog [https://hazyresearch.stanford.edu/blog/2024-05-12-tk](https://hazyresearch.stanford.edu/blog/2024-05-12-tk)
- TileLang: [https://github.com/tile-ai/tilelang](https://github.com/tile-ai/tilelang) · paper arXiv 2504.17577 [https://arxiv.org/abs/2504.17577](https://arxiv.org/abs/2504.17577)
- Modular, "Democratizing AI Compute, Part 7 — Triton and Python eDSLs": [https://www.modular.com/blog/democratizing-ai-compute-part-7-what-about-triton-and-python-edsls](https://www.modular.com/blog/democratizing-ai-compute-part-7-what-about-triton-and-python-edsls)
- *TVM Deep Dives* — [Lecture 02 — TensorIR & the schedule space](../TVM%20Deep%20Dives/Lecture-02.md), for the scheduling primitives TileLang exposes.

---

## Current as of

2026-06. Pins: Triton 3.x (MLIR, Tile-IR backend), CUTLASS 4.x (CuTe DSL, GTC 2025), cuTile / Tile IR with CUDA 13.1, ThunderKittens 2.0 (Blackwell + MXFP8/NVFP4), TileLang open-sourced Jan 2025 on TVM. The ~20% Triton-vs-CUDA gap is a contested, kernel-and-generation-dependent figure — treat as illustrative, re-measure on your hardware.

---

*Next: [Lecture 03 — Compilers & runtimes](Lecture-03.md)*
