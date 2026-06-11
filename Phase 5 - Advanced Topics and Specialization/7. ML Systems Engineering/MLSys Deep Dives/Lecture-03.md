# Lecture 03 - Compilers and Runtimes: From Autoscheduling to Megakernels

**Collection:** [MLSys Deep Dives](README.md) | **Previous:** [← Lecture 02](Lecture-02.md) | **Next:** [Lecture 04](Lecture-04.md)

---

Lecture 2 was about *writing* a kernel — you, choosing tiles. This lecture is about the layer that decides what happens when you *don't* want to choose, or when there are ten thousand kernels and a whole graph to schedule, fuse, and keep resident on the device. That is the job of **compilers** (which pick or search for schedules) and **runtimes** (which decide how the resulting kernels actually execute at scale).

We tour four compilers that sit at different points — **TVM** (search), **Mojo/MAX** (a whole language), **TensorRT-LLM** (closed vendor), **IREE** (portable MLIR) — and then the runtime axis that is quietly where a lot of 2026's throughput gains are coming from: the move from per-op launches to **persistent megakernels**, exemplified by **TileRT**.

---

## Learning objectives

By the end of this lecture, you should be able to:

1. Distinguish *hand-written DSL* (Lec 2) from *compiler-chosen* schedules, and the three ways compilers choose: heuristic, search, whole-language.
2. Summarize **TVM** (Relax/TensorIR/MetaSchedule/BYOC/MLC-LLM) as the autoscheduling pole, and know where to go for depth.
3. Explain **Mojo/MAX** as "a real language, not an eDSL," and the fragmentation argument behind it.
4. Place **TensorRT-LLM** as the closed-vendor pole and describe its 2025 shift toward PyTorch-native authoring + Dynamo.
5. Describe the **runtime axis** — eager → CUDA graph → **persistent megakernel** — and why **TileRT**'s megakernel design wins on overhead.
6. Decide, for a given workload, whether to hand-write, compile/search, or buy the closed engine.

---

## 1. Who chooses the schedule?

A kernel's performance is mostly its **schedule** — tile sizes, loop order, what's cached in shared memory, how the pipeline overlaps. Lecture 2's DSLs put that choice in *your* hands (Triton hides some; ThunderKittens/CUTLASS expose all). Compilers take the choice *away* from you, in one of three ways:

```text
   ① HEURISTIC   compiler applies expert default schedules, no search
                 (XLA fusion, TVM's dlight, torch.compile's Inductor)        → instant, ~good
   ② SEARCH      compiler measures many schedules on real hardware, keeps best
                 (TVM MetaSchedule, Ansor)                                    → slow, ~peak
   ③ WHOLE LANG  you write ONE language spanning kernel→graph→serving
                 (Mojo/MAX)                                                   → unified, new ecosystem
```

This is the *other half* of Lecture 2's ease↔control axis. The DSLs ask "how do you want this scheduled?"; the compilers answer "let me figure it out" — by heuristic (fast, portable, ~80–90% of peak) or by search (slow, peak, target-specific). Neither is strictly better; they're different points on the same effort-vs-peak trade, now automated.

---

## 2. TVM — the autoscheduling pole

**Apache TVM** is the most complete embodiment of the *search* approach, and the broadest multi-target open compiler. The one-paragraph shape (this course has a whole companion course on it — [TVM Deep Dives](../TVM%20Deep%20Dives/README.md) — so we stay at altitude here):

```text
   import model → Relax (graph IR, dynamic shapes) → lower to TensorIR (loop IR)
        → MetaSchedule SEARCHES schedules, measured on the real device
        → codegen (CUDA/LLVM/Metal/Vulkan/WebGPU/C) → runtime
        + BYOC: offload subgraphs to external codegen (CUTLASS/TensorRT/a custom NPU)
        + MLC-LLM: the whole stack pointed at LLMs, cross-platform
```

What makes TVM matter for this course's thesis:

* **It searches instead of asking you.** MetaSchedule generates candidate TensorIR schedules, ranks them with a learned cost model, measures the promising ones on the actual GPU (over RPC if it's an edge board), and keeps the winner. You trade tuning *time* for a kernel tuned to *your* exact shape and silicon — which is how TVM can beat vendor libraries on shapes the vendor didn't anticipate.
* **It is the foundation under newer tools.** **TileLang** (Lec 2) and **TileRT** (§6) are built on TVM infrastructure; **MLC-LLM** is TVM Unity pointed at LLMs across CUDA/Metal/Vulkan/WebGPU. When you learn TVM you learn the substrate of a chunk of the 2026 stack.
* **dlight** is its *heuristic* mode: default GPU schedules with **no** tuning, used by MLC-LLM precisely because you cannot run a search on every user's phone GPU. TVM contains both poles — heuristic (dlight) and search (MetaSchedule) — and the senior move is knowing which to use where.

Reach for TVM when you need **portability across many targets** (including edge and web) and are willing to invest tuning time for peak — or when you're bringing up a backend for **new hardware** (its BYOC + autoschedule path is the on-ramp). For the build-it details, take the [TVM Deep Dives](../TVM%20Deep%20Dives/README.md) course.

---

## 3. Mojo / MAX — a real language, not an eDSL

**Modular's** bet (founded by Chris Lattner — LLVM, Swift, MLIR) is that the whole "1,001 Python eDSLs" situation from Lecture 2 is a symptom of a missing piece: **a real systems language for accelerators**. That language is **Mojo** — a Python-superset built directly on **MLIR**, designed to be "Python-easy, C++/Rust-fast," spanning CPU/GPU/accelerators in one syntax.

The argument, stated fairly: a Python eDSL (Triton, cuTile) "looks like Python but isn't" — it's a restricted subset that traces to a kernel, with its own rules and failure modes. Mojo is instead a *complete* language, so the same code can express a Tensor-Core kernel, a model, and the glue around it without crossing a language boundary. Since mid-2025 Mojo's **standard library** provides portable GPU programming directly (the `gpu` module), and Modular claims one of the largest open repos of CPU+GPU kernels.

**MAX** is the product on top: Modular's **inference engine + graph compiler + serving** stack, with its own FlashAttention/MLA kernels written *in Mojo*, targeting NVIDIA Hopper/Blackwell and AMD MI-series, claiming competitive Blackwell performance.

The open-source timeline matters because it's the main adoption question:

```text
   2024-03   Mojo stdlib open-sourced (Apache 2.0)
   2025-11   all MAX Python API modules open-sourced (v25.7)
   2026-fall Modular's public commitment to open-source the Mojo LANGUAGE/compiler
```

Until the compiler opens, Mojo/MAX is the **most Modular-locked** option here — that is the live risk. But the thesis ("stop writing eDSLs, write a language") is the cleanest counter-argument to the tile-DSL fragmentation, and worth understanding even if you don't adopt it. Reach for it when you want **one language from kernel to serving** and are comfortable betting on Modular's stack and timeline.

---

## 4. TensorRT-LLM — the closed-vendor pole

At the opposite end from TVM's open search sits **TensorRT / TensorRT-LLM** (NVIDIA). TensorRT is a closed **builder**: feed it a model, it emits a hardware-specific *engine* using NVIDIA's proprietary kernels (fusion, precision calibration, kernel auto-selection). **TensorRT-LLM** specializes that for LLM serving: custom attention kernels, **in-flight (continuous) batching**, **paged KV cache**, quantization (**FP8/FP4/INT4-AWQ/INT8 SmoothQuant**), speculative decoding, multi-GPU/multi-node.

Two 2025 shifts you should know, because they change how it's used:

* **PyTorch-native authoring.** TRT-LLM moved away from the old "build an opaque engine" flow toward **PyTorch-native model definitions** + a modular Python runtime + a stable production API, and added **AutoDeploy** to deploy PyTorch models with less manual conversion. The closed kernels stay closed; the *authoring* got far more open.
* **Dynamo + FlashInfer upstreaming.** It integrates with **NVIDIA Dynamo** (datacenter-scale distributed inference) and — notably — NVIDIA is now **publishing its top LLM kernels into the open FlashInfer library** (Lec 6) for reuse by vLLM/SGLang. The "only way to go fast on NVIDIA is TRT-LLM" moat is eroding as open stacks (vLLM, SGLang) + FlashInfer close the gap.

Reach for TRT-LLM when the deployment is **NVIDIA-only and you want best-in-class latency with least kernel effort**, and you accept closed, non-portable, non-inspectable kernels. It is the productivity-and-peak corner *if* you never leave NVIDIA — the opposite trade from TVM.

---

## 5. IREE and the MLIR substrate

One layer under almost everything in Lectures 2–3 is **MLIR** — the LLVM-project multi-level IR that Triton, cuTile's Tile IR, Mojo, TVM-adjacent tooling, and IREE all build on. It's the substrate, not a kernel language.

**IREE** ("Intermediate Representation Execution Environment") is the MLIR-ecosystem's **retargetable compiler *and* runtime** (part of OpenXLA): AOT + JIT, scaling from datacenter GPUs down to mobile/edge, CPU/GPU/accelerator backends. It overlaps TVM's "portable compiler + runtime" goal but leans on the MLIR/linalg dialect stack rather than TVM's autoschedule-centric design. If your organization is MLIR-committed (or JAX/OpenXLA-centric), IREE is the natural compiler; if you want mature search-based autotuning and the LLM path, TVM/MLC is more developed. (And JAX's **Pallas/Mosaic**, from Lec 2, is the TPU-side tile path in the same ecosystem.)

The practical takeaway: you rarely "choose MLIR" directly — you choose a compiler (TVM, IREE, Mojo, Triton) and it chooses MLIR underneath. Knowing it's the shared substrate explains why these tools interoperate and why a Tile-IR-for-Triton backend was even possible.

---

## 6. The runtime axis — and why megakernels are winning

Compilers produce kernels; the **runtime** decides how they execute. This is where a surprising amount of 2026 throughput is hiding, because the naive execution model wastes the GPU between kernels.

```text
   ① EAGER (per-op launch)     [launch][op][launch][op][launch][op]...     ← gap before every op
                                                                              kernel-launch overhead
                                                                              + HBM round-trip per op
   ② CUDA GRAPH / GRAPH EXEC   [──── replay a captured graph, far fewer launches ────]
                                                                              amortizes launch cost
   ③ PERSISTENT MEGAKERNEL     [──────── one resident kernel; warps specialized ───────]
                                load / compute / comm OVERLAPPED inside; µs-scale overhead,
                                pipeline stays GPU-resident across the whole forward pass
```

The trend is rightward. At small batch and short decode steps — exactly the memory-bound regime where LLM decode lives — **kernel-launch overhead and inter-op HBM round-trips become a large fraction of the step time**. Put numbers on it, because the claim sounds abstract until you do:

```text
   batch-1 decode step, 70B-class transformer (illustrative):
   ~80 layers × ~8–10 unfused kernels/layer  ≈  600–800 launches per token
   × ~2–5 µs launch + teardown each          ≈  1.5–4 ms of pure gap per step
   vs a TPOT budget of ~10–20 ms             →  10–30% of the step is nobody-computing time
   …and at every kernel boundary the activations spill to HBM and come back,
   paying bandwidth the roofline never required.
```

Capturing the graph (CUDA Graphs) reclaims most of the *launch* cost; fusing the *entire* model into a **single persistent "megakernel"** reclaims the *round-trips* too, because the data never leaves on-chip memory between ops and the launch gaps vanish. Note the regime-dependence: at batch 128 prefill those same gaps are noise behind big GEMMs — which is why megakernels are a *decode/latency* story, not a universal one.

**TileRT** (from the `tile-ai` group, same lineage as TileLang) is the 2026 exemplar: a **persistent-megakernel runtime** that decomposes LLM operators into **fine-grained tile-level tasks** and dynamically **overlaps compute, I/O, and communication** across GPUs, with **warp specialization** keeping the whole pipeline resident and overhead at microsecond scale. It is the runtime behind a headline throughput result you'll meet in Lecture 6 (a 1-trillion-parameter model pushed past **1000 tokens/s** on a single 8-GPU commodity node). For now, hold the principle: **once kernels are fast, the next bottleneck is the gaps between them, and megakernels close the gaps.**

---

## 7. Choosing — the decision table

| You want… | Reach for | Why |
|---|---|---|
| Portability across many targets + peak via tuning | **TVM** (MetaSchedule) | searches schedules per shape/device; edge→web→LLM |
| Instant good schedules, no tuning, on any backend | **TVM dlight** / `torch.compile` | heuristic schedules; ship now |
| One language from kernel → model → serving | **Mojo / MAX** | a real language, not an eDSL (Modular-locked for now) |
| Best NVIDIA latency, least kernel effort, NV-only | **TensorRT-LLM** | closed vendor kernels + serving features |
| MLIR/OpenXLA-aligned portable compiler+runtime | **IREE** | retargetable, server→edge, MLIR-native |
| Kill the gaps between fast kernels | **megakernel runtime (TileRT)** | persistent, overlapped, µs overhead |

And the meta-decision — **hand-write (Lec 2) vs. compile/search vs. buy-the-closed-engine** — comes down to three questions: *How many shapes/targets?* (many → compile), *How much of my cost is in one kernel?* (concentrated → hand-write that one), *Am I NVIDIA-only and latency-critical?* (yes → TRT-LLM is hard to beat). A mature stack mixes all three: TRT-LLM or vLLM for serving, a hand-written ThunderKittens attention kernel for the hot path, TVM/MLC for the odd target, a megakernel runtime to close the gaps.

---

## 8. Measure it

Same discipline as Lecture 2, now at the model level. Compile one model two ways and carry the number to dollars:

```text
   path A: torch.compile (Inductor → Triton)   → tokens/s, $/Mtok
   path B: TVM + MetaSchedule (tuned)           → tokens/s, $/Mtok, + tuning time
   path C: TensorRT-LLM (built engine)          → tokens/s, $/Mtok, TTFT
   (optional) wrap the winner in a megakernel/graph runtime → tokens/s delta from killing gaps
```

Report tokens/s, TTFT, TPOT, and `$/Mtok` for each, plus the **one-time cost** each path charged you (TVM's tuning hours, TRT-LLM's build + NVIDIA lock-in, Mojo's ecosystem bet). The right answer is rarely "one compiler" — it's "this path for this part of the workload," justified by the table.

---

## 9. Mini-lab: three compilers, one model

Take a small model you can run end to end.

1. **Baseline:** eager PyTorch. Record tokens/s, TTFT, TPOT, `$/Mtok`.
2. **Heuristic compile:** `torch.compile`. Record the deltas; note that the kernels underneath are Triton (Lec 2).
3. **Search compile:** tune the model with **TVM MetaSchedule** (or apply dlight for the instant path). Record deltas *and* the tuning time you spent.
4. **Closed engine (if on NVIDIA):** build a **TensorRT-LLM** engine. Record deltas and TTFT.
5. **Reason about gaps:** profile the fastest path; estimate how much time is *between* kernels (launch + idle). That gap is the megakernel opportunity.

Deliverable: a `{eager, torch.compile, TVM, TRT-LLM}` × `{tokens/s, TTFT, TPOT, $/Mtok, one-time cost}` table, plus a paragraph on which path you'd ship and why, and how much headroom the inter-kernel gaps still hold. That synthesis is the lecture.

---

## Key takeaways

- DSLs (Lec 2) put the schedule in *your* hands; **compilers choose it for you** — by **heuristic** (instant, ~good), **search** (slow, ~peak), or by giving you a **whole language** (Mojo).
- **TVM** is the autoscheduling pole: Relax + TensorIR + **MetaSchedule** (search) and **dlight** (heuristic), broad targets, the substrate under TileLang/TileRT/MLC-LLM. Depth lives in the companion [TVM Deep Dives](../TVM%20Deep%20Dives/README.md) course.
- **Mojo/MAX** answers eDSL fragmentation with "a real language" spanning kernel→model→serving — powerful, MLIR-based, but Modular-locked until the compiler opens (committed fall 2026).
- **TensorRT-LLM** is the closed-vendor pole: peak NVIDIA latency, least kernel effort, NV-only; 2025 made authoring PyTorch-native and began upstreaming kernels into open FlashInfer.
- The **runtime axis** moves eager → CUDA graph → **persistent megakernel**. Once kernels are fast, the gaps between them dominate; **TileRT**'s megakernel design closes them (the engine behind a 1000-tok/s milestone, Lec 6).
- There's no single winner — mix hand-written kernels, a compiler, a closed engine, and a megakernel runtime, each justified by tokens/s and `$/Mtok`.

---

## References

- Apache TVM (Relax/Unity, MetaSchedule, MLC-LLM): [https://tvm.apache.org/docs/](https://tvm.apache.org/docs/) — and the companion [TVM Deep Dives](../TVM%20Deep%20Dives/README.md) course.
- Modular Mojo & MAX: [https://www.modular.com/open-source/mojo](https://www.modular.com/open-source/mojo) · MAX changelog [https://docs.modular.com/max/changelog/](https://docs.modular.com/max/changelog/)
- NVIDIA TensorRT-LLM: [https://github.com/NVIDIA/TensorRT-LLM](https://github.com/NVIDIA/TensorRT-LLM) · overview [https://nvidia.github.io/TensorRT-LLM/overview.html](https://nvidia.github.io/TensorRT-LLM/overview.html)
- IREE (OpenXLA, MLIR compiler + runtime): [https://github.com/iree-org/iree](https://github.com/iree-org/iree)
- TileRT (tile-ai, persistent-megakernel runtime): [https://github.com/tile-ai/TileRT](https://github.com/tile-ai/TileRT)
- NVIDIA, "Run high-performance LLM inference kernels from NVIDIA using FlashInfer": [https://developer.nvidia.com/blog/run-high-performance-llm-inference-kernels-from-nvidia-using-flashinfer/](https://developer.nvidia.com/blog/run-high-performance-llm-inference-kernels-from-nvidia-using-flashinfer/)

---

## Current as of

2026-06. Pins: TVM Unity/Relax mainline (MetaSchedule + dlight); Mojo stdlib open (2024), MAX Python modules open (v25.7, Nov 2025), Mojo language open-source committed "fall 2026"; TensorRT-LLM PyTorch-native + AutoDeploy + Dynamo; TileRT preview (tile-ai). Megakernel throughput claims are workload-specific — see Lecture 6 for the pinned MiMo/TileRT figures and their vendor-reported caveat.

---

*Next: [Lecture 04 — Beyond the dense transformer](Lecture-04.md)*
