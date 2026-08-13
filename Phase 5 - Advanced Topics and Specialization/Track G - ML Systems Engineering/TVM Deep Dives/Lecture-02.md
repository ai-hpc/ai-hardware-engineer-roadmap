# Lecture 02 - TensorIR and the Schedule Space: Turning a Compute Definition into a Hardware-Mapped Kernel

**Collection:** [TVM Deep Dives](README.md) | **Previous:** [← Lecture 01](Lecture-01.md) | **Next:** [Lecture 03](Lecture-03.md)

---

In Lecture 1 we said the central invariant of the stack: **scheduling changes loop order and memory mapping, never the result.** This lecture is what that sentence means in practice.

A compute definition — `C[i,j] = sum_k A[i,k] * B[k,j]` — is a statement of *what* to compute. It says nothing about *how*: which loop is outermost, what is tiled to fit in L1, what lives in registers, which axis is bound to which GPU thread, whether the inner kernel is a scalar multiply-add or a Tensor Core instruction. The **schedule** is the *how*. TensorIR makes the *how* a first-class, transformable program, and the space of all legal hows is the **schedule space** you will spend your career navigating.

This is the heart of the compiler. Get it, and Lecture 3 (letting a machine search this space) and Lecture 5 (dlight's pre-built schedules for LLMs) are just automation of what you do here by hand.

---

## Learning objectives

By the end of this lecture, you should be able to:

1. Generate a TIR `PrimFunc` from a Tensor Expression (TE), and read its block / iter-var structure.
2. Explain a TIR **block**: spatial vs reduction axes (`T.axis.remap("SSR", ...)`), the `init` region, and read/write regions.
3. Drive the `tir.Schedule` API: `get_block`, `get_loops`, `split`, `reorder`, `bind`, `cache_read`, `cache_write`, `compute_at`, `vectorize`, `unroll`, `parallel`, `decompose_reduction`.
4. Hand-schedule a matmul for a **CPU** (tile → vectorize → parallel) and a **GPU** (block/thread bind → shared-memory cache → cooperative fetch).
5. Map each primitive to its **roofline** effect — why it moves you toward (or to a higher) performance roof.
6. `tensorize` an inner block onto a hardware intrinsic (Tensor Core MMA), and measure GFLOP/s before and after.

---

## 1. TE and TIR: two ways to reach the same loop nest

You get a TIR `PrimFunc` two ways. You can **write it directly** in TVMScript (as in Lecture 1), or you can **describe the math** with a Tensor Expression and let TVM generate the loop nest. TE is the gentle on-ramp: you state the computation declaratively, TVM materializes the loops.

```python
import tvm
from tvm import te, tir

def matmul_te(M, N, K, dtype="float32"):
    A = te.placeholder((M, K), name="A", dtype=dtype)
    B = te.placeholder((K, N), name="B", dtype=dtype)
    k = te.reduce_axis((0, K), name="k")              # the reduction axis
    C = te.compute(
        (M, N),
        lambda i, j: te.sum(A[i, k] * B[k, j], axis=k),
        name="C",
    )
    return te.create_prim_func([A, B, C])             # TE  →  TIR PrimFunc

PrimFunc = matmul_te(1024, 1024, 1024)
PrimFunc.show()
```

`te.create_prim_func` is the bridge: declarative TE in, schedulable TensorIR out. What it prints is the same kind of block-structured loop nest you saw in Lecture 1 — and *that* is the thing we schedule. From here on, TE has done its job; we work on the TIR.

> Why have both? TE is convenient for expressing standard ops; direct TIR (TVMScript) is what you write when you need exact control, what auto-tuning manipulates, and what the importer emits. Senior engineers read and write both, and treat TE as a generator for TIR — never as a separate world.

---

## 2. Anatomy of a TIR block

Everything schedulable in TensorIR lives inside a **block**. The block is the unit the scheduler reasons about, and its annotations are a contract that keeps every transformation legal.

```python
for i, j, k in T.grid(1024, 1024, 1024):
    with T.block("C"):
        vi, vj, vk = T.axis.remap("SSR", [i, j, k])   # ← the contract
        with T.init():
            C[vi, vj] = T.float32(0)                  # reduction init
        C[vi, vj] = C[vi, vj] + A[vi, vk] * B[vk, vj]  # reduction update
```

Read the contract:

* **`T.axis.remap("SSR", [i, j, k])`** declares three block iteration variables. The string `"SSR"` types them: `vi`, `vj` are **S**patial (independent output coordinates — safe to parallelize, reorder freely), `vk` is a **R**eduction (accumulates into one output — reordering it changes nothing numerically *only because* it is marked associative).
* **`T.init()`** is the reduction's initialization region: the `C = 0` that must happen once per `(vi, vj)` *before* any `k`. Marking it separately is what lets the scheduler hoist or parallelize the reduction safely.
* The block also carries **read/write regions** (which elements of `A`, `B`, `C` it touches), inferred from the body. The scheduler uses these to know what is safe to cache, move, or fuse.

This is why TIR can be aggressively transformed where raw C cannot: the block tells the compiler *exactly* what is spatial, what is reduction, and what memory is touched. The schedule primitives are just legal rewrites of this structure.

---

## 3. The schedule object and the primitives

You schedule by constructing a `tir.Schedule` over the module and issuing primitives. Each primitive mutates the IR in place; `sch.mod.show()` shows the result.

```python
sch = tir.Schedule(PrimFunc)         # a mutable schedule over the TIR
block_C = sch.get_block("C")         # grab the block by name
i, j, k = sch.get_loops(block_C)     # its loops, outer→inner
```

The core primitives, grouped by what they physically do:

| Primitive | Call | Physically |
|---|---|---|
| **split** | `i0, i1 = sch.split(i, [None, 32])` | cut one loop into a nest of two (tiling) |
| **reorder** | `sch.reorder(i0, j0, i1, j1)` | change loop nesting order |
| **fuse** | `sch.fuse(i, j)` | merge two loops into one |
| **bind** | `sch.bind(i0, "blockIdx.x")` | map a loop onto a GPU grid/block axis |
| **cache_read** | `sch.cache_read(blk, 0, "shared")` | stage an input buffer in faster memory |
| **cache_write** | `sch.cache_write(blk, 0, "local")` | accumulate the output in registers, write back once |
| **compute_at** | `sch.compute_at(prod, loop)` | move a producer block inside a consumer loop (locality) |
| **vectorize** | `sch.vectorize(i1)` | emit SIMD over this loop |
| **unroll** | `sch.unroll(i1)` | unroll for ILP / less loop overhead |
| **parallel** | `sch.parallel(i0)` | run this loop across CPU cores |
| **decompose_reduction** | `sch.decompose_reduction(blk, k0)` | split `init` out above the reduction loop |
| **tensorize** | `sch.tensorize(i1, INTRIN)` | replace an inner region with a hardware intrinsic |

These compose. A real schedule is a *program* of twenty to a hundred of these calls. The art is knowing which sequence maps the loop nest onto the target's memory hierarchy and execution units — which is exactly what the roofline tells you, and exactly what Lecture 3 automates.

---

## 4. Worked example: scheduling matmul for a CPU

Start from the naive nest and transform it into a cache-blocked, vectorized, multi-threaded kernel. The naive version is correct but leaves >90% of the machine on the floor: it streams `B` from DRAM with no reuse and uses one core and one lane.

```python
sch = tir.Schedule(matmul_te(1024, 1024, 1024))
C = sch.get_block("C")
i, j, k = sch.get_loops(C)

# 1) Tile i and j so a block of C stays hot in cache; split k for an inner accumulate loop
i0, i1 = sch.split(i, factors=[None, 32])      # 32×32 output tile
j0, j1 = sch.split(j, factors=[None, 32])
k0, k1 = sch.split(k, factors=[None, 4])

# 2) Reorder: tiles outermost, reduction in the middle, the hot 32×32×4 micro-kernel innermost
sch.reorder(i0, j0, k0, i1, k1, j1)

# 3) Accumulate the output tile in registers, not back to DRAM every k-step
C_local = sch.cache_write(C, 0, "local")
sch.reverse_compute_at(C_local, j0)

# 4) SIMD over the contiguous j1, multi-thread over the outer tile
sch.vectorize(j1)
sch.parallel(i0)

sch.mod.show()                                  # read the transformed nest — this is the kernel
```

Walk the intent against the memory hierarchy:

* **split + reorder** create a `32×32` output tile that fits in L1/registers, so each loaded element of `A` and `B` is **reused** across the tile instead of re-fetched. This is the single biggest lever — it raises **arithmetic intensity**.
* **cache_write to "local"** keeps the partial sums in registers across the `k` loop; DRAM is written once per tile, not once per multiply-add.
* **vectorize(j1)** turns the inner 32-wide loop into AVX/NEON SIMD.
* **parallel(i0)** spreads tiles across cores.

Now compile and **measure** (the only thing that counts):

```python
import numpy as np
target, dev = "llvm -mcpu=native", tvm.cpu(0)
func = tvm.build(sch.mod, target=target)

M = N = K = 1024
a = tvm.nd.array(np.random.rand(M, K).astype("float32"), dev)
b = tvm.nd.array(np.random.rand(K, N).astype("float32"), dev)
c = tvm.nd.array(np.zeros((M, N), "float32"), dev)

ev = func.time_evaluator(func.entry_name, dev, number=50)
t = ev(a, b, c).mean
print(f"{t*1e3:.2f} ms   {2*M*N*K/t/1e9:.1f} GFLOP/s")
```

On a typical desktop core you will see the naive nest land in the low tens of GFLOP/s and the scheduled version several **times** higher — the gap is the schedule, not the math. Compare your number to the machine's peak (cores × SIMD width × FMA × clock) to see how much roof is left.

---

## 5. Worked example: scheduling matmul for a GPU

The GPU schedule answers different questions: which loop is the **grid**, which is the **thread block**, what gets **cooperatively loaded** into shared memory. The primitives are the same; the targets of `bind` and `cache_read` change.

```python
sch = tir.Schedule(matmul_te(1024, 1024, 1024))
C = sch.get_block("C")
i, j, k = sch.get_loops(C)

# 1) Two-level tile on the spatial axes: outer → thread blocks, inner → threads
bi, ti = sch.split(i, factors=[None, 16])      # 16×16 threads per block
bj, tj = sch.split(j, factors=[None, 16])
sch.reorder(bi, bj, ti, tj, k)

# 2) Map loops onto the CUDA grid
sch.bind(bi, "blockIdx.y")
sch.bind(bj, "blockIdx.x")
sch.bind(ti, "threadIdx.y")
sch.bind(tj, "threadIdx.x")

# 3) Stage A and B tiles in shared memory; the threads in a block fetch them cooperatively
k0, k1 = sch.split(k, factors=[None, 8])       # K-tile of 8
A_sh = sch.cache_read(C, 0, "shared")
B_sh = sch.cache_read(C, 1, "shared")
sch.compute_at(A_sh, k0)                        # load the A tile once per K-step, shared by the block
sch.compute_at(B_sh, k0)

target, dev = "cuda", tvm.cuda(0)
func = tvm.build(sch.mod, target=target)
print(func.imported_modules[0].get_source())   # ← read the generated CUDA C!
```

The mental model:

```text
   grid of thread blocks            each block owns a 16×16 tile of C
   ┌──────┬──────┬──────┐
   │ blk  │ blk  │ blk  │           inside a block, 256 threads each own one C element
   ├──────┼──────┼──────┤           A-tile and B-tile staged in SHARED memory,
   │ blk  │ blk  │ blk  │           loaded cooperatively, reused by all 256 threads
   └──────┴──────┴──────┘           → DRAM traffic cut by the tile width
```

`func.imported_modules[0].get_source()` prints the **generated CUDA C**. Reading it is non-negotiable: you will see the `__shared__` arrays, the `__syncthreads()`, the thread-indexed loads. That generated source is the proof the schedule did what you intended — and the place you debug when it didn't.

The remaining work to reach a vendor-competitive kernel — register tiling, double-buffering the shared loads, vectorized `float4` global loads, avoiding bank conflicts — is *more of the same primitives*. This is precisely the long, target-specific tail that you do **not** want to hand-tune for every shape. Which is the entire motivation for Lecture 3.

---

## 6. Every primitive is a roofline move

The reason a senior engineer can schedule without flailing: each primitive has a known effect on the roofline. You are not guessing — you are moving a point toward a roof, or jumping to a higher roof.

| Primitive | What it changes physically | Roofline effect |
|---|---|---|
| split / reorder / tile | blocking, loop order | ↑ arithmetic intensity via cache/register **reuse** → move right toward the compute roof |
| cache_read / cache_write | stage data in shared / registers | ↓ DRAM traffic → ↑ AI, and feed the compute units without stalling |
| bind (block/thread) | expose GPU parallelism | lets you **reach** the roofs at all (no parallelism, no throughput) |
| vectorize | SIMD lanes | ↑ achieved compute throughput per instruction |
| unroll | ILP, less loop overhead | hide latency → get closer to the compute roof |
| parallel | multicore | scale the **CPU** compute roof with core count |
| **tensorize** | map inner block to an MMA intrinsic | **jump to a higher compute roof entirely** (Tensor Cores) |

So the diagnostic loop is: profile → "am I memory-bound or compute-bound?" → pick the primitive that moves *that* bound. Memory-bound? Tile harder, cache more, coalesce. Compute-bound on FP32 cores but there are Tensor Cores idle? `tensorize`.

This is also the language you use to *read* a tuned schedule (Lecture 3) or a dlight schedule (Lecture 5): you recognize the tile sizes, the shared-memory stages, the tensorize call, and you know what roof each one is chasing.

---

## 7. `tensorize`: jumping to the Tensor Core roof

FP32 SIMD on the CUDA cores has one compute roof. The **Tensor Cores** have a far higher one — but only for the specific shape and dtype of their MMA (matrix-multiply-accumulate) instruction (e.g. a `16×16×16` fp16-in / fp32-accumulate tile). `tensorize` is the primitive that replaces a matching inner block with that hardware intrinsic.

The pattern:

1. Schedule so the innermost block's shape **matches the intrinsic's** (e.g. tile `i,j,k` down to `16,16,16`, load fragments to the right memory scopes).
2. `tensorize` that inner block against a registered **tensor intrinsic**.

```python
# Conceptually (Tensor Core path — requires fp16 inputs, fp32 accumulate, the right scopes):
from tvm.tir.tensor_intrin import cuda as cuda_intrin   # predefined wmma/mma intrinsics

# ... schedule down to a 16×16×16 inner block `mma`, with A/B in wmma.matrix_a/b scope ...
sch.tensorize(mma, cuda_intrin.WMMA_SYNC_16x16x16_f16f16f32_INTRIN)
```

TVM ships a library of registered intrinsics (`tvm.tir.tensor_intrin`) for NVIDIA wmma/mma, and you can **register your own** (`tvm.tir.TensorIntrin.register`) — which is exactly how you'd target a *custom* accelerator's GEMM instruction or TVM's open VTA accelerator. That extensibility is the bridge from "compiler" to "compiler for hardware that does not exist yet," and it leads directly into BYOC in Lecture 4.

The payoff is large and worth measuring directly: a correctly tensorized fp16 matmul moves from the CUDA-core FP32 roof to the Tensor Core roof — typically a multiple, not a few percent. Print the GFLOP/s before and after the `tensorize` call; the jump is the lesson.

---

## 8. Measure it

A schedule is a hypothesis. The `time_evaluator` is how you test it. Build the discipline now, because every later lecture grades you by the number.

```python
def gflops(func, shapes, dev, number=50):
    M, N, K = shapes
    a = tvm.nd.array(np.random.rand(M, K).astype("float32"), dev)
    b = tvm.nd.array(np.random.rand(K, N).astype("float32"), dev)
    c = tvm.nd.array(np.zeros((M, N), "float32"), dev)
    ev = func.time_evaluator(func.entry_name, dev, number=number)
    t = ev(a, b, c).mean
    return 2 * M * N * K / t / 1e9, t

# Always verify correctness before celebrating speed:
ref = a.numpy() @ b.numpy()
np.testing.assert_allclose(c.numpy(), ref, rtol=1e-3)
```

Report three numbers, always: **correctness** (parity vs a reference — a fast wrong kernel is worthless), **GFLOP/s**, and **% of roofline peak**. The third is what tells you whether to keep scheduling or stop. At 85% of peak, go home. At 15%, you have a memory-bound kernel and the table in §6 tells you which primitive to reach for.

---

## 9. Mini-lab: schedule a kernel three ways

Take matmul (or a 2D convolution if you want a harder reduction structure).

1. **Baseline:** build the naive nest, record GFLOP/s and % of CPU peak.
2. **CPU schedule:** tile → `cache_write` local → `vectorize` → `parallel`. Sweep the tile size over `{8, 16, 32, 64}` and plot GFLOP/s vs tile. Find the cache cliff.
3. **GPU schedule:** block/thread bind → shared-memory `cache_read` with `compute_at`. Print the generated CUDA, find the `__shared__` arrays and `__syncthreads()`. Record GFLOP/s and % of GPU FP32 peak.
4. **(Stretch) tensorize:** fp16 inputs, schedule to a `16×16×16` inner block, `tensorize` to a wmma intrinsic. Record the jump to the Tensor Core roof.

Deliverable: one table — `{baseline, cpu, gpu, tensorized}` × `{GFLOP/s, % peak, parity}` — and a two-sentence note per row naming the roofline bound each schedule was chasing. That table is a Level-4 artifact: numbers, with raw data and interpretation.

---

## Key takeaways

- A compute definition is *what*; the **schedule** is *how*. TensorIR makes the *how* a first-class, legal-by-construction program.
- A **TIR block** declares spatial vs reduction axes (`"SSR"`), an `init` region, and read/write regions. Those annotations are the contract that keeps every primitive a numerically-identical rewrite.
- The primitives — `split`, `reorder`, `bind`, `cache_read/write`, `compute_at`, `vectorize`, `unroll`, `parallel`, `decompose_reduction`, `tensorize` — compose into a schedule that maps loops onto the memory hierarchy and execution units.
- CPU scheduling is tile → cache in registers → vectorize → parallelize. GPU scheduling is block/thread bind → cooperative shared-memory staging. Same primitives, different `bind`/scope targets.
- **Every primitive is a roofline move.** Profile, name the bound, pick the primitive that moves it. `tensorize` jumps to a *higher* roof (Tensor Cores / custom MMA).
- Read the generated source (`get_source()`) and always measure correctness + GFLOP/s + % of peak. A schedule is a hypothesis until the evaluator confirms it.
- Hand-scheduling for every shape/target is exactly the toil Lecture 3 automates — but you must understand it by hand first to trust (and debug) the machine.

---

## References

- TensorIR deep dive — blocks, axes, schedule primitives: [https://tvm.apache.org/docs/deep_dive/tensor_ir/index.html](https://tvm.apache.org/docs/deep_dive/tensor_ir/index.html)
- "Blitz Course to TensorIR" tutorial: [https://tvm.apache.org/docs/tutorial/tensor_ir_blitz_course.html](https://tvm.apache.org/docs/tutorial/tensor_ir_blitz_course.html)
- `tvm.tir.schedule` API reference: [https://tvm.apache.org/docs/reference/api/python/tir.html](https://tvm.apache.org/docs/reference/api/python/tir.html)
- "Use Tensorize to Leverage Hardware Intrinsics": [https://tvm.apache.org/docs/how_to/work_with_schedules/tensorize.html](https://tvm.apache.org/docs/how_to/work_with_schedules/tensorize.html)
- Feng et al., "TensorIR: An Abstraction for Automatic Tensorized Program Optimization," ASPLOS 2023: [https://arxiv.org/abs/2207.04296](https://arxiv.org/abs/2207.04296)
- *TVM Deep Dives* — [Lecture 03 — Auto-tuning](Lecture-03.md), which searches this exact schedule space automatically.

---

*Next: [Lecture 03 — Auto-tuning: AutoTVM → Ansor → MetaSchedule](Lecture-03.md)*
