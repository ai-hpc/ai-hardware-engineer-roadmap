# Lecture 01 - The TVM Stack and the Compilation Flow: Relax, TensorIR, and the Unified IRModule

**Collection:** [TVM Deep Dives](README.md) | **Previous:** [← TVM Deep Dives index](README.md) | **Next:** [Lecture 02](Lecture-02.md)

---

A framework **dispatches** a graph. A compiler **rewrites** it.

When you call `model(x)` in PyTorch, the framework walks the graph op by op and, for each op, calls into a precompiled kernel — cuDNN, cuBLAS, oneDNN, a hand-written CUDA file. The hardware does what the **vendor library author** decided, for the shapes the vendor library author anticipated.

Apache TVM does something different. It takes the *whole* graph as data, runs analysis and rewriting passes over it, lowers it to an explicit loop representation, mechanically searches for a good way to map those loops onto the memory hierarchy and execution units of **your** target, and emits a standalone artifact.

This lecture is the map of that machine. Before you can schedule a kernel (Lecture 2) or tune one (Lecture 3), you need to know **what the levels are, what artifact carries them, and what each stage of the flow is allowed to change**.

---

## Learning objectives

By the end of this lecture, you should be able to:

1. Explain why a tensor compiler exists — the shape × dtype × target × fusion explosion that vendor libraries cannot cover.
2. Name the levels of the TVM stack — **Relax** (graph IR), **TensorIR / TIR** (loop IR), target codegen — and what each one represents.
3. Describe the **`IRModule`** as the single artifact that holds *all* levels at once, and why that "unity" matters.
4. Walk the compilation flow: import → graph optimization → lower → schedule → codegen → runtime.
5. Place Relay vs Relax historically and say why Relax (TVM Unity) is the forward direction.
6. Import a real model, print every IR level, build it, and run it — and read the output of each step.

---

## 1. Why a tensor compiler exists

Consider one operator: a matrix multiply, `C[M,N] = A[M,K] @ B[K,N]`.

The *math* is four lines. The *fast implementation* is not, and there is not **one** fast implementation. The right code depends on:

```text
shape      M, N, K — tall-skinny vs square vs batched-tiny behave totally differently
dtype      fp32 / fp16 / bf16 / int8 / fp8 / int4 — different vector widths, different cores
target     AVX-512 vs NEON vs SM80 Tensor Core vs SM90 vs Apple AMX vs a custom NPU
fusion     is a bias-add, a ReLU, a dequant fused onto the epilogue?
layout     row-major, column-major, NCHW, NHWC, blocked NC/16c
```

The cross product is enormous. A vendor library ships a few thousand hand-tuned kernels covering the **common** points of that space. Step off the grid — an odd shape, a new dtype, a fused pattern nobody anticipated, an accelerator nobody wrote a library for — and you fall back to a slow generic kernel, or to nothing.

A compiler's bet is the opposite: **describe the computation once, describe the hardware mapping as a transformable program, and generate the kernel for the exact point you are standing on.** That is the entire reason TVM exists. Everything else — Relax, TIR, MetaSchedule, BYOC — is machinery in service of that bet.

The hardware-first way to say it: a framework gives you the hardware the vendor exposed. A compiler lets you target the hardware **you actually have**, at the shape **you actually run**.

---

## 2. The levels of the stack

TVM is a **multi-level** compiler. Two levels matter most, and you will live in both.

```text
        high level  │  Relax     — the dataflow graph: ops, tensors, control flow, shapes
                    │              "what computation, in what order"
   ─────────────────┼──────────────────────────────────────────────────────────
        low level   │  TensorIR  — the loop nest: for-loops, buffers, compute, reductions
            (TIR)   │              "exactly which element, in which order, into which memory"
   ─────────────────┼──────────────────────────────────────────────────────────
        target      │  CUDA C / LLVM IR / Metal / Vulkan SPIR-V / C   — emitted source
```

**Relax** (short for "Relax", the relaxation of Relay) is the **graph IR**. A Relax function looks like a sequence of tensor operations with a dataflow structure — much like an FX graph or an ONNX graph, but with first-class **symbolic shapes** and the ability to call down into TIR directly.

**TensorIR (TIR)** is the **loop-level IR**. A TIR `PrimFunc` is an explicit nest of `for` loops over buffers, with the computation written out element-by-element, plus **block** annotations that mark the schedulable units. This is the level you *schedule* — split loops, bind them to GPU threads, cache into shared memory, vectorize.

The key TVM design decision, and the thing that distinguishes it from XLA (which hides the loop level) or TensorRT (which hides everything): **the loop level is first-class, inspectable, and programmatically transformable.** You can print it, rewrite it, and the rewrite is the optimization.

---

## 3. The IRModule: one artifact, all levels

Here is the idea that names "TVM Unity": **a single container, the `IRModule`, holds functions at every level simultaneously.** A graph-level Relax function and the loop-level TIR functions it calls live in the *same* module and can be optimized *together*.

```python
import tvm
from tvm.script import ir as I, relax as R, tir as T

@I.ir_module
class MyModule:
    # ---- TIR level: an explicit loop nest, schedulable ----
    @T.prim_func
    def matmul(A: T.Buffer((128, 128), "float32"),
               B: T.Buffer((128, 128), "float32"),
               C: T.Buffer((128, 128), "float32")):
        for i, j, k in T.grid(128, 128, 128):
            with T.block("C"):
                vi, vj, vk = T.axis.remap("SSR", [i, j, k])   # S=spatial, R=reduction
                with T.init():
                    C[vi, vj] = T.float32(0)
                C[vi, vj] = C[vi, vj] + A[vi, vk] * B[vk, vj]

    # ---- Relax level: the dataflow graph, calls the TIR func above ----
    @R.function
    def main(x: R.Tensor((128, 128), "float32"),
             w: R.Tensor((128, 128), "float32")) -> R.Tensor((128, 128), "float32"):
        cls = MyModule
        with R.dataflow():
            # call_tir: invoke a TIR PrimFunc from the graph, declaring the output shape
            lv = R.call_tir(cls.matmul, (x, w), out_sinfo=R.Tensor((128, 128), "float32"))
            R.output(lv)
        return lv
```

Read this carefully, because it is the whole stack in twenty lines:

* `@T.prim_func matmul` is **TIR** — the loop nest. The `T.block("C")` with `T.axis.remap("SSR", ...)` declares a schedulable block whose `i,j` axes are *spatial* and whose `k` axis is a *reduction*. (Lecture 2 lives entirely inside this kind of block.)
* `@R.function main` is **Relax** — the graph. `R.call_tir` is the bridge: the graph reaches *down* and calls the TIR kernel, declaring the output shape with `out_sinfo`.
* `R.dataflow()` marks a **dataflow block** — a side-effect-free region the optimizer is free to reorder, fuse, and rewrite. (Lecture 4 lives here.)

That co-residence is the point. The graph-level fuser can fold two ops together and *generate a new TIR function* for the fused result. The tuner can rewrite a TIR function and the graph still points at it. There is no lossy hand-off between a "graph compiler" and a separate "kernel compiler" — it is one module the whole way down.

`mod.show()` pretty-prints any `IRModule` as this **TVMScript** — Python-syntax that round-trips to and from the actual IR. It is your primary debugging tool. Print early, print often.

---

## 4. The compilation flow

Here is what happens between `import model` and `output = compiled(x)`.

```text
   PyTorch / ONNX / TF / JAX model
            │   ① import  (frontend → Relax)
            ▼
   ┌────────────────────────────────────┐
   │ IRModule                           │
   │   @R.function main      (Relax)    │
   │   @T.prim_func ...      (TIR)      │
   └────────────────────────────────────┘
            │   ② graph-level passes        [operates on Relax]
            │        fuse ops, transform layout, fold constants, plan memory
            │   ③ legalize + lower           [Relax → TIR]
            │        turn each high-level op into a TIR PrimFunc
            │   ④ schedule / tune            [operates on TIR]
            │        map loops → memory hierarchy & execution units  (Lec 2 & 3)
            │   ⑤ codegen                    [TIR → target source]
            ▼
   target module:  .ptx/.cubin (CUDA) | .o (LLVM) | .metallib | SPIR-V | .c
            │   ⑥ runtime
            ▼
   loaded module you call like a function   (GraphExecutor | Relax VM | AOT)
```

Each stage has a strict contract about **what it may change**:

| Stage | Input → Output | What it is allowed to change | What it must preserve |
|---|---|---|---|
| ① Import | framework graph → Relax | representation only | numerics |
| ② Graph passes | Relax → Relax | op structure: fuse, re-layout, fold, plan memory | the function's input→output mapping |
| ③ Legalize/lower | Relax → Relax+TIR | replace abstract ops with concrete TIR loop nests | semantics of each op |
| ④ Schedule/tune | TIR → TIR | *loop order, tiling, memory placement, threading* | the computed result, exactly |
| ⑤ Codegen | TIR → target src | nothing semantic — pure translation | everything |
| ⑥ Runtime | module → callable | execution policy (static graph vs VM vs AOT) | results |

Internalize the one in bold: **scheduling changes only the order and the memory mapping, never the result.** That invariant is what makes auto-tuning safe — the machine can try ten thousand schedules and every one of them is, by construction, numerically the same kernel. Lecture 3 is built entirely on that guarantee.

---

## 5. Relay vs Relax: why the stack has a history

You will see two graph IRs in TVM documentation and you should know which is which.

| | **Relay** (legacy) | **Relax** (TVM Unity, current) |
|---|---|---|
| Era | ~2018, first-gen graph IR | ~2023+, the "Unity" direction |
| Shapes | mostly static; dynamic shape bolted on | **symbolic / dynamic shapes first-class** |
| Relationship to TIR | separate compile step, lossy hand-off | **same IRModule**, `call_tir` cross-level |
| Optimization | graph-level only | graph + tensor level, **jointly** |
| LLM / dynamic models | awkward | designed for them (MLC-LLM is built on it) |
| Status | still present, maintained, but legacy | the forward direction; new work targets Relax |

The short version: **Relay** proved the graph-IR + auto-tuning idea but kept the graph and tensor levels in separate worlds with a one-way hand-off between them. **Relax** ("Relax" = a more relaxed, dynamic, cross-level IR) collapses that wall — graph and TIR share one module, dynamic shapes are native, and the whole thing was designed around the workloads that broke Relay: LLMs with KV-caches, variable sequence lengths, and control flow.

For this course: **we write Relax.** Relay appears only when you read an older tutorial. If a doc says `relay.build`, the modern equivalent is `relax.build` / `tvm.compile`.

This is not academic trivia. The reason MLC-LLM (Lecture 5) can run a Qwen model on a phone GPU is that Relax made dynamic-shape, cross-level compilation a first-class thing. The history *is* the capability.

---

## 6. Hands-on: import, inspect, build, run

Let's take a real model end to end. The pattern is identical whether the source is PyTorch or ONNX; the frontend differs, everything after is the same.

**From PyTorch (via `torch.export`):**

```python
import torch, torch.nn as nn
import tvm
from tvm import relax
from tvm.relax.frontend.torch import from_exported_program

class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 256)
        self.fc2 = nn.Linear(256, 10)
    def forward(self, x):
        return self.fc2(torch.relu(self.fc1(x)))

model = MLP().eval()
example = (torch.randn(1, 784),)

# torch.export gives a clean, functional graph; TVM imports it into Relax
exported = torch.export.export(model, example)
mod = from_exported_program(exported, keep_params_as_input=False)
```

**From ONNX:**

```python
import onnx
from tvm.relax.frontend.onnx import from_onnx

onnx_model = onnx.load("mlp.onnx")
mod = from_onnx(onnx_model, keep_params_in_input=False)
```

Either way you now hold a Relax `IRModule`. **Print it** — this is the habit that separates people who use TVM from people who debug it:

```python
mod.show()        # pretty-printed TVMScript: the Relax graph, ops still high-level
```

Now run the **default optimization pipeline** (fusion, legalization to TIR, the works) and look again:

```python
mod = relax.get_pipeline("zero")(mod)   # the "zero" pipeline: a sane default opt sequence
mod.show()                              # now you'll see TIR PrimFuncs appear — ops got lowered
```

You just watched stages ② and ③ happen: the high-level `R.matmul` / `R.nn.relu` calls became `R.call_tir` into concrete `@T.prim_func` loop nests, and adjacent elementwise ops got fused.

**Build and run.** `relax.build` compiles the module for a target; the result is an `Executable` you load onto the **Relax Virtual Machine**:

```python
target = tvm.target.Target("llvm")        # or "cuda", "metal", "vulkan", ...
dev = tvm.cpu(0)                          # or tvm.cuda(0)

ex = relax.build(mod, target)            # ⑤ codegen → a loadable Executable
vm = relax.VirtualMachine(ex, dev)       # ⑥ runtime

x = tvm.nd.array(torch.randn(1, 784).numpy(), dev)
out = vm["main"](x)                      # call it like a function
print(out.numpy().shape)                 # (1, 10)
```

> **API note.** Recent TVM exposes a unified front door, `tvm.compile(mod, target)`, that dispatches to the Relax build path — the two are equivalent for our purposes. Older tutorials use `relax.build`; both return something you run on `relax.VirtualMachine`. If you are reading a *Relay*-era tutorial you'll see `relay.build(...)` returning a `GraphModule` instead — same idea, legacy path.

That is the entire flow, executed. Import, inspect, optimize, inspect again, build, run. Every later lecture zooms into one stage of exactly this sequence.

---

## 7. Targets, codegen, and runtimes (the preview)

Two choices at the bottom of the flow shape everything downstream; we name them now and detail them in Lecture 5.

**Target** = the device + codegen backend. One `IRModule`, many targets:

```text
"llvm"                         CPU via LLVM (x86, ARM, RISC-V — set -mtriple/-mcpu)
"cuda"                         NVIDIA GPU (PTX/cubin)
"rocm"                         AMD GPU
"metal"                        Apple GPU
"vulkan" / "opencl"            cross-vendor GPU
"webgpu"                       browser GPU  (this is how MLC-LLM runs in a tab)
"c"                            portable C source — the microTVM / bare-metal path
```

**Runtime** = how the compiled graph is executed:

| Runtime | Model | Use it for |
|---|---|---|
| **GraphExecutor** | static graph, ahead-of-time memory plan | fixed-shape, classic inference; lowest overhead |
| **Relax VM** | bytecode VM, supports control flow & dynamic shape | LLMs, dynamic models, anything with `if`/loops |
| **AOT** | compiled-ahead, no interpreter, C callable | microcontrollers, embedded, no-Python deploy |

For a fixed-shape ResNet you want GraphExecutor. For a Qwen model with a growing KV-cache you want the Relax VM. For a Cortex-M MCU you want AOT. Same compiler, three runtimes — Lecture 5 ships all three.

---

## 8. Where TVM sits in the compiler landscape

You will be asked, in any MLSys interview, "why TVM and not XLA / TensorRT / torch.compile?" The honest answer is about **what each one exposes and what it hides**.

```text
                 exposes loop level?   open / multi-vendor?   auto-tuning?   edge+web+LLM path?
  TVM                    yes                   yes               yes (MS)            yes
  XLA                    no                    TPU+GPU           no                  partial
  TensorRT               no                    NVIDIA only       tactic search       no
  torch.compile          via Triton            GPU-centric       Triton autotune     no
  IREE/MLIR              yes (MLIR)            yes               limited             yes
```

TVM's distinctive position: it is the one that (a) makes the **schedule** a first-class, tunable object, (b) is **open and multi-target** down to MCUs and browsers, and (c) carries a mature **learning-based tuner** (Lecture 3) and a **new-accelerator on-ramp** (BYOC + VTA, Lecture 4). The cost is that it asks more of you than `torch.compile` — you are closer to the metal, which is exactly the point of this course.

It is not a competition you "win." A production team often runs TensorRT for NVIDIA-only serving *and* TVM for the edge/odd-target tail. Knowing TVM teaches you the category; the others are subsets of its ideas.

---

## 9. Mini-lab: read the whole stack on one model

Pick any small model — an MLP, a small CNN, or a single transformer block.

1. Import it into Relax (`from_exported_program` or `from_onnx`).
2. `mod.show()` **before** any pass. Identify the high-level ops.
3. Apply `relax.get_pipeline("zero")`. `mod.show()` **after**. Find one place where two ops got **fused**, and one high-level op that became a `call_tir` into a `@T.prim_func`.
4. Build for `"llvm"`, run on the Relax VM, and check the output against the PyTorch reference (`np.testing.assert_allclose`, `rtol=1e-3`).
5. Change the target to `"cuda"` (if you have a GPU) and run again. **Same module, different codegen** — note that you changed one string and nothing else.

Deliverable: a short note pasting the *before* and *after* TVMScript with the fused op and the `call_tir` circled, plus the parity check passing on two targets. That note is proof you can read the stack — the prerequisite for scheduling it.

---

## Key takeaways

- A tensor compiler exists because the fast implementation of an op depends on shape × dtype × target × fusion × layout — a space too large for vendor libraries to cover. TVM generates the kernel for the exact point you stand on.
- The stack has two main levels: **Relax** (graph IR — what computation) and **TensorIR/TIR** (loop IR — exactly which element, which memory, which order). Target codegen sits below.
- The **`IRModule`** holds *all* levels at once. Graph (Relax) and kernel (TIR) functions co-reside and are optimized together — this is "TVM Unity," and `R.call_tir` is the bridge.
- The flow is import → graph passes → legalize/lower → schedule/tune → codegen → runtime. Crucially, **scheduling changes only loop order and memory mapping, never the result** — the invariant that makes auto-tuning safe.
- **Relax** superseded **Relay**: symbolic shapes first-class, cross-level optimization, built for LLMs. Write Relax; treat `relay.build` as legacy.
- `mod.show()` is your microscope. Print the module before and after every pass.

---

## References

- Apache TVM documentation — Overview & "Quick Start": [https://tvm.apache.org/docs/](https://tvm.apache.org/docs/)
- TVM Unity vision post: [https://tvm.apache.org/2021/12/15/tvm-unity](https://tvm.apache.org/2021/12/15/tvm-unity)
- Relax / TVM Unity docs (frontends, `relax.build`, VirtualMachine): [https://tvm.apache.org/docs/reference/api/python/relax/relax.html](https://tvm.apache.org/docs/reference/api/python/relax/relax.html)
- TensorIR deep dive (blocks, `T.axis`, `call_tir`): [https://tvm.apache.org/docs/deep_dive/tensor_ir/index.html](https://tvm.apache.org/docs/deep_dive/tensor_ir/index.html)
- Chen et al., "TVM: An Automated End-to-End Optimizing Compiler for Deep Learning," OSDI 2018: [https://www.usenix.org/conference/osdi18/presentation/chen](https://www.usenix.org/conference/osdi18/presentation/chen)
- *AI Inference Engineer 2026* — runtime landscape lecture, for where TVM sits among serving stacks.

---

*Next: [Lecture 02 — TensorIR and the schedule space](Lecture-02.md)*
