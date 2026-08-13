# TVM Deep Dives — Apache TVM for MLSys Engineers

<div class="course-identity mlsys" markdown="1">
<div class="course-identity__icon">TVM</div>
<div markdown="1">
<p class="course-identity__eyebrow">Phase 5 · ML Systems Engineering · Special Course</p>
<p class="course-identity__title">Compile a model to the metal: the Apache TVM stack — Relax, TensorIR, MetaSchedule, BYOC, and MLC-LLM — as a senior MLSys engineer actually uses it.</p>
<p class="course-identity__meta">Artifact: a tuned, deployed compiled model with a measured speedup report · Measure: latency, GFLOP/s vs roofline, tuning trials, binary size, $/inference</p>
</div>
</div>

> *A framework runs the graph someone else optimized. A compiler lets you become that someone.*

Most ML engineers stop at the framework boundary. They call `model(x)`, the framework dispatches to cuDNN or oneDNN, and the hardware does what the vendor library decided it should do. That is fine until you hit a shape, a dtype, a fused pattern, or an accelerator that **no vendor library covers** — a custom NPU, an edge MCU, a WebGPU target, a quantization scheme nobody shipped a kernel for. At that boundary you stop being a user of kernels and start being an author of them. **Apache TVM is the open compiler stack for doing that at scale**, across the whole target zoo, without hand-writing a kernel per shape per device.

This course is how an MLSys engineer reads, drives, extends, and ships that stack. It is **build-first**: every lecture lowers a real computation, schedules it, tunes it, or deploys it, and reads a number off the other side.

**Layer mapping:** L4 (compiler / graph optimization) is the spine, reaching down into L3 (kernels / codegen) and up into L5–L6 (runtime / deployment). This is the core L4 compiler-engineer skill set.

**Role targets:** ML Compiler Engineer · MLSys Engineer · AI Inference Engineer · Edge AI / TinyML Engineer · Accelerator Software Engineer (the person who writes the BYOC backend for a new chip).

**Prerequisites:**

* Phase 4 — Track C — [ML Compiler and Graph Optimization](../../../Phase%204%20-%20Track%20C%20-%20DL%20Inference%20Optimization/Guide.md) — what a graph IR, an operator, and a lowering pass are.
* Phase 5 — Edge AI — [Edge LLM Inference Internals](../../Track%20C%20-%20Edge%20AI/Edge%20LLM%20Inference%20Internals/Lecture-01.md) — GEMV vs GEMM, the roofline, the decode bottleneck. You will need the roofline to read every "Measure it" section here.
* Comfort with Python (the whole TVM frontend is Python) and the ability to read CUDA C and LLVM-flavored loop code. No prior compiler-internals knowledge assumed — that is what this course builds.

**What comes after:** a compiled-model repo — one network, lowered and tuned with MetaSchedule for at least two targets (e.g. x86 + CUDA, or CUDA + an edge board), with a benchmark table comparing the framework baseline, the un-tuned TVM build, and the tuned TVM build against the hardware roofline, plus one BYOC offload experiment.

---

## Why TVM, and why now

TVM is the oldest and most general of the open ML compilers (born at UW in 2017), and the one whose ideas — **a schedulable tensor IR, learning-based auto-tuning, bring-your-own-codegen** — propagated into everything that came after. Knowing TVM is the fastest way to understand the *whole* category, because the others are largely re-expressions of pieces of it.

| Compiler | Graph IR | Tensor IR / scheduling | Auto-tuning | Strength | Where TVM differs |
|---|---|---|---|---|---|
| **Apache TVM** | Relax (Unity); Relay (legacy) | TensorIR (TIR) + schedules | MetaSchedule / Ansor / AutoTVM | Breadth of targets, open accelerator path (BYOC, VTA), edge → web → LLM | The reference point for this table |
| **XLA** (JAX/TF) | HLO | implicit (fusion + libnvjit) | none (heuristics) | TPU, whole-program fusion | TVM exposes the schedule; XLA hides it |
| **TensorRT** | builder graph | closed kernels | builder tactics | NVIDIA peak perf | Closed, single-vendor; TVM is open + multi-target |
| **torch.compile / Inductor** | FX graph | Triton + C++ | autotune (Triton configs) | PyTorch-native, fast iteration | TVM ships standalone artifacts to non-Python, non-GPU targets |
| **IREE / MLIR** | linalg-on-tensors | nested MLIR dialects | limited | mobile/edge, MLIR ecosystem | Overlapping goals; TVM's tuning + LLM path (MLC) are more mature |

The 2026-relevant reason to learn it specifically: **MLC-LLM** — the project that runs Llama / Qwen / Phi-class models on Metal, Vulkan, WebGPU, ROCm, and phones — is *built on TVM Unity*. The same Relax + TIR + dlight stack you learn in Lectures 4–5 is what compiles those models. TVM is the bridge between "I understand transformer execution" and "I can ship that transformer to a device nobody wrote a kernel for."

---

## Course Map (5 lectures)

The arc is the compiler itself, top to bottom and back up: read the stack → schedule a kernel → let the machine schedule it → optimize the graph and reach external codegen → ship the artifact.

<div class="lecture-map" markdown>

| # | Lecture | What you build / measure |
|---|---------|--------------------------|
| [01](Lecture-01.md) | **The TVM stack and the compilation flow** — Relax, TensorIR, and the unified `IRModule` | Import a model, print every IR level, build and run; map the import → optimize → lower → codegen → runtime pipeline |
| [02](Lecture-02.md) | **TensorIR and the schedule space** — turning a compute definition into a hardware-mapped kernel | Schedule a matmul for CPU (tile + vectorize) and GPU (block/thread bind + shared-memory cache); `tensorize` to a Tensor Core intrinsic; GFLOP/s before/after |
| [03](Lecture-03.md) | **Auto-tuning** — AutoTVM → Ansor → MetaSchedule, the learning-based compiler | `ms.tune_tir` a kernel and `ms.tune_relax` a model; read the tuning curve; tune a remote edge board over an RPC tracker |
| [04](Lecture-04.md) | **Relax in depth** — dynamic shapes, operator fusion, and Bring Your Own Codegen (BYOC) | Symbolic-shape a model; fuse with `FuseOps`/`FuseTIR`; offload a subgraph to CUTLASS/TensorRT and measure fused-vs-unfused and offloaded-vs-native |
| [05](Lecture-05.md) | **Shipping it** — runtime, microTVM, and LLMs with MLC-LLM | Package a module (GraphExecutor / Relax VM / AOT); deploy a tiny model to a bare-metal MCU with microTVM; compile + run an LLM with MLC-LLM across backends |

</div>

---

## Course Outcomes

By the end you should be able to:

* Read a TVM `IRModule` at every level — Relax graph, TIR `PrimFunc`, generated CUDA/LLVM — and explain what each pass changed and why.
* Hand-schedule a kernel from a compute definition to a hardware-mapped implementation, and explain every primitive (`split`, `bind`, `cache_read`, `compute_at`, `tensorize`) in roofline terms.
* Run MetaSchedule on a kernel and a whole model, distribute the measurement to real devices over RPC, and read the cost-model-vs-hardware tradeoff in the tuning curve.
* Optimize a Relax graph (fusion, layout, memory planning, dynamic shape) and offload a subgraph to an external codegen path via BYOC — the exact skill of bringing up a new accelerator backend.
* Ship a compiled artifact to a non-Python target: a server `.so`, an MCU C library, or an LLM across CUDA/Metal/Vulkan/WebGPU via MLC-LLM — and defend the numbers.

---

## Exit Criteria

You are done with this course when you can:

* Take a model the framework already runs, compile it with TVM, **tune it, and beat the framework baseline** on at least one target — and show the roofline gap you closed.
* Look at a profiler trace of your tuned kernel and say whether the remaining gap is compute-bound, memory-bound, or launch-bound, and which schedule primitive would move it.
* Explain to an accelerator team **what a BYOC backend would have to implement** to offload their op set, and roughly how much of the graph it would capture.
* Walk another engineer through your compiled-model repo and have them reproduce your tuned numbers on the same hardware class within ±10%.

If you can only make TVM *run* a model, you have a transpiler. The point of this course is to make it *optimize* one — and to know, by the number, that it did.

---

*Related: [MLSys Deep Dives](../MLSys%20Deep%20Dives/README.md) · [AI Inference Engineer 2026](../AI%20Inference%20Engineer%202026/README.md) · [Phase 4 — Track C — ML Compiler and Graph Optimization](../../../Phase%204%20-%20Track%20C%20-%20DL%20Inference%20Optimization/Guide.md)*
