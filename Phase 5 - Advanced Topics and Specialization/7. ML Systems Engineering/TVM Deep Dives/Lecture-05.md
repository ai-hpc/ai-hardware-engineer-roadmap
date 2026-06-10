# Lecture 05 - Shipping It: Runtime, microTVM, and LLMs with MLC-LLM

**Collection:** [TVM Deep Dives](README.md) | **Previous:** [← Lecture 04](Lecture-04.md) | **Next:** [TVM Deep Dives index](README.md)

---

A compiled kernel that only runs inside a Python tuning script has shipped nothing. The whole point of a compiler — versus a framework — is that the output is a **standalone artifact** you deploy where Python, CUDA, and a 4 GB runtime cannot go: a C++ inference server, a Cortex-M microcontroller with 256 KB of SRAM, a browser tab, a phone.

This lecture closes the loop: how TVM's output becomes a deployable thing, across the full range from datacenter to bare metal, and how the most important application of the entire stack — **MLC-LLM**, universal LLM deployment — assembles everything from Lectures 1–4 into something that runs a Qwen model on hardware no vendor wrote an LLM kernel for.

It is also the course capstone. By the end you should be able to take a model, compile it, tune it, and ship it with numbers.

---

## Learning objectives

By the end of this lecture, you should be able to:

1. Choose among the three runtimes — **GraphExecutor**, **Relax VM**, **AOT** — and export a loadable module + params.
2. Load and run a compiled TVM module from a **non-Python** environment (the C++ runtime).
3. Explain the **microTVM** path: AOT + C codegen, the Model Library Format, deployment to a bare-metal MCU, and CMSIS-NN / Ethos-U offload via BYOC.
4. Describe the **MLC-LLM** pipeline end to end: `relax.frontend.nn` model → quantization → Relax/TIR opt → **dlight** schedules → per-backend codegen → `MLCEngine`.
5. Reason about **dlight vs MetaSchedule** — instant portable schedules vs tuned peak — and choose correctly.
6. Compile and run an LLM across backends, and a tiny model on an MCU, and report latency / memory / size.

---

## 1. The three runtimes, and the artifact

Lecture 1 named them; now we ship with them. The runtime is *how the compiled graph executes*, and the choice is dictated by the deployment target.

| Runtime | Execution model | Overhead | Dynamic shapes / control flow | Ship it to |
|---|---|---|---|---|
| **GraphExecutor** | static graph, pre-planned memory | low | no | fixed-shape server/edge inference |
| **Relax VM** | bytecode VM | small per-op | **yes** | LLMs, dynamic models, anything with `if`/loops |
| **AOT** | compiled ahead, no interpreter, C entry | **near-zero** | limited | microcontrollers, bare metal, no-OS |

The **artifact** is the same idea in all three: a compiled module plus the parameters. Export it, and you no longer need the compiler — only the lightweight runtime.

```python
import tvm
from tvm import relax

ex = relax.build(mod, target="cuda")          # the Executable from Lecture 1
ex.export_library("model_cuda.so")            # ← the shippable artifact: a shared library

# later, anywhere, with only the TVM runtime present:
loaded = tvm.runtime.load_module("model_cuda.so")
vm = relax.VirtualMachine(loaded, tvm.cuda(0))
out = vm["main"](x)
```

For the classic static path it is `GraphModule` over an exported `.so` + a params blob; for embedded it is a `.tar` in **Model Library Format** (next sections). The principle is invariant: **compile once, produce a self-contained artifact, deploy it without the toolchain.**

---

## 2. Deploying without Python

The TVM runtime is a small C++ library (the "minimal runtime" can be a few hundred KB). That is what makes the artifact portable: you load the `.so` from C++, Rust, Java, or JavaScript, with no Python and no compiler on the deployment box.

```cpp
// minimal C++ deployment — the shape of every production TVM integration
#include <tvm/runtime/module.h>
#include <tvm/runtime/registry.h>

tvm::runtime::Module mod = tvm::runtime::Module::LoadFromFile("model_cuda.so");
auto vm_load = tvm::runtime::Registry::Get("relax.VirtualMachine");
// set device, set inputs as NDArrays, call the "main" function, read outputs
```

This is the boundary that frameworks cannot cross cleanly and compilers are built for. Your inference server is C++; it loads the artifact and calls it like a function. The same artifact, cross-compiled, runs on an ARM SBC. The runtime is the only dependency.

> **RPC, again.** The same RPC system you used for *tuning* in Lecture 3 is also a *deployment and profiling* tool: push the cross-compiled artifact to a remote board, run it, and pull timings back — without installing a toolchain on the device. Tune over RPC, then deploy and profile over RPC. One mechanism, the whole edge lifecycle.

---

## 3. microTVM: down to bare metal

Now go all the way down — to a Cortex-M microcontroller with no operating system, no `malloc`, and SRAM measured in **kilobytes**. This is **microTVM**, and it is where the compiler approach pays off most dramatically: there is no PyTorch here, no CUDA, often no Linux. The model *is* C code, statically planned.

The path swaps two things from the server flow:

```text
   server:   target "cuda"   + Relax VM/GraphExecutor + .so + dynamic alloc
   micro:    target "c"      + AOT executor           + MLF + STATIC memory plan
```

* **`c` codegen** emits portable C source for the kernels — no LLVM backend needed for the device, just the vendor's C compiler.
* **AOT executor** compiles the *graph itself* to C (no interpreter, no runtime graph walk) — a single `tvmgen_default_run()` entry point.
* **Static memory planning** is mandatory: every buffer is laid out at compile time into a fixed workspace arena, because there is no heap. The compiler must fit the whole model's activations into the SRAM budget — and report the number so you know if it fits.
* The output is the **Model Library Format (MLF)** — a `.tar` containing the generated C, the params, and the metadata a **Project API** integration uses to drop the model into a Zephyr, Arduino, or CMSIS build.

```python
# conceptual microTVM build: AOT + C runtime, exported as Model Library Format
import tvm
from tvm import relax
from tvm.micro import export_model_library_format

target = tvm.target.Target("c -keys=arm_cpu -mcpu=cortex-m55")
# build with the embedded C runtime + AOT executor (no OS, static workspace),
# then package for project generation:
export_model_library_format(built_module, "model_cortex_m.tar")
# → feed the .tar to the Project API to generate a Zephyr/Arduino firmware project, flash, run
```

And BYOC (Lecture 4) reaches down here too: on Arm MCUs you offload to **CMSIS-NN** (hand-optimized DSP/SIMD kernels) and, on parts with the **Ethos-U** micro-NPU, to the Ethos-U codegen — TVM partitions the graph, sends conv/matmul to the NPU, and keeps the rest as generated C. The TinyML deployment story — keyword spotting, anomaly detection, vision wake-words on a coin-cell budget — is exactly this pipeline.

The metric that matters on an MCU is not GFLOP/s; it is **does it fit and does it hit the deadline**: flash size, SRAM peak (the workspace arena), and inference latency under the power budget. microTVM reports all three at compile time, which is why it beats hand-porting: you know before you flash.

---

## 4. MLC-LLM: the whole stack, pointed at LLMs

Everything so far converges here. **MLC-LLM** (Machine Learning Compilation for LLMs) is the project that compiles Llama / Qwen / Phi / Gemma-class models to run on **CUDA, ROCm, Metal, Vulkan, OpenCL, and WebGPU** — server GPUs, Macs, iPhones, Android, and browser tabs — and it is **built on TVM Unity**. When you understand MLC-LLM, you understand what the entire course was building toward: a real, shipping, universal-deployment system whose every stage is a lecture you've already done.

```text
   HF model (Llama / Qwen / Phi / Gemma ...)
        │  ① define architecture with relax.frontend.nn   (PyTorch-like module API)
        ▼
   Relax IRModule  —  SYMBOLIC shapes: seq_len, batch, kv-cache length   [Lec 4]
        │  ② quantize weights:  q4f16_1 (4-bit group) / q4f16_awq / q0f16 ...
        ▼
   Relax GRAPH optimization  —  fusion incl. FuseDequantizeMatmulEwise    [Lec 4]
        │  ③ legalize → lower to TIR                                       [Lec 1]
        ▼
   TIR optimization                                                       [Lec 2]
        │  ④ dlight default schedules: GEMV, matmul, attention, RMSNorm  — NO tuning
        ▼
   ⑤ codegen per backend:  CUDA | ROCm | Metal | Vulkan | WebGPU | OpenCL [Lec 1]
        ▼
   model library (.so / .dylib / .wasm)  +  MLCEngine runtime (OpenAI-compatible API)
        ▼
   runs on: server GPU · Mac · iPhone · Android · browser tab
```

Map each numbered stage to where you learned it: **①** is `relax.frontend.nn` building a Relax graph; **②** quantization feeds the dequant-fusion of Lecture 4; **③/⑤** are the import/lower/codegen flow of Lecture 1; **④** is TensorIR scheduling from Lecture 2 — except done by **dlight** instead of by hand. MLC-LLM is not a different system. It is *this* system, productized.

Defining a model uses the `relax.frontend.nn` API — deliberately PyTorch-shaped so model authors feel at home, but it builds a Relax `IRModule`:

```python
from tvm.relax.frontend import nn

class MLP(nn.Module):
    def __init__(self):
        self.fc1 = nn.Linear(784, 256)
        self.fc2 = nn.Linear(256, 10)
    def forward(self, x):
        return self.fc2(nn.op.relu(self.fc1(x)))

# export to a Relax IRModule + params — the same module type from Lectures 1–4
mod, params = MLP().export_tvm(
    spec={"forward": {"x": nn.spec.Tensor((1, 784), "float32")}}
)
```

A real LLM definition is the same idea at scale: attention with a KV-cache, RoPE, RMSNorm, a SwiGLU FFN — all built from `nn` ops, all producing a symbolic-shape Relax module the rest of the pipeline optimizes.

---

## 5. dlight: good schedules with zero tuning — and why MLC needs it

Here is the design tension MLC-LLM resolves, and it is a genuinely important systems lesson. MetaSchedule (Lecture 3) produces *peak* kernels — but it tunes by **measuring on the target device**, for minutes to hours. You cannot run a 2000-trial tuning job **on every user's phone GPU**, or **in a browser tab**, or on an iPhone you don't physically have. Portability and per-device tuning are in direct conflict.

**dlight** is the resolution: a library of **hand-written default schedule rules**, one per important op class (GEMV for decode, matmul for prefill, reductions, attention, RMSNorm), that produce *good* GPU schedules **instantly, analytically, with no measurement**. Not peak — but 80–90% of it, on any backend, immediately.

```python
import tvm.dlight as dl

with tvm.target.Target("vulkan"):              # or metal, webgpu, cuda, rocm ...
    mod = dl.ApplyDefaultSchedule(             # apply default rules, NO tuning loop
        dl.gpu.Matmul(),
        dl.gpu.GEMV(),                         # the decode-phase workhorse
        dl.gpu.Reduction(),
        dl.gpu.GeneralReduction(),
        dl.gpu.Fallback(),                     # a safe default for anything unmatched
    )(mod)
```

The tradeoff in one table — and knowing which to reach for is a senior-engineer judgment call:

| | **dlight** | **MetaSchedule** |
|---|---|---|
| How | analytical default rules per op class | search + on-device measurement |
| Time to a kernel | instant | minutes–hours |
| Quality | good (~80–90% of peak) | peak |
| Needs the target device? | **no** | **yes** (measures on it) |
| Portability across backends | immediate | re-tune per target |
| Right for | **MLC-LLM**: ship to phones/web/Macs you can't tune on | server kernels you control and tune once |

So the rule: **dlight when you cannot tune the deployment target** (consumer devices, the long tail of GPUs, the browser) — which is exactly MLC-LLM's situation. **MetaSchedule when you own the silicon** and can amortize a one-time tuning job into peak throughput (your datacenter serving fleet). Mature deployments use both — dlight for portability, MetaSchedule for the kernels on hardware they control.

---

## 6. Hands-on: compile and run an LLM across backends

The MLC-LLM CLI walks the pipeline of §4 as three commands, then `MLCEngine` serves the result with an OpenAI-compatible API.

```bash
# ① convert + quantize weights (4-bit group quant, fp16 activations)
mlc_llm convert_weight ./models/Qwen2.5-7B-Instruct/ \
    --quantization q4f16_1 -o ./dist/Qwen2.5-7B-q4f16_1-MLC

# ② generate runtime config (chat template, context window, KV settings)
mlc_llm gen_config ./models/Qwen2.5-7B-Instruct/ \
    --quantization q4f16_1 --conv-template qwen2 -o ./dist/Qwen2.5-7B-q4f16_1-MLC

# ③ compile to a model library for a target backend  (swap --device to retarget)
mlc_llm compile ./dist/Qwen2.5-7B-q4f16_1-MLC/mlc-chat-config.json \
    --device cuda -o ./dist/libs/Qwen2.5-7B-q4f16_1-cuda.so
#   --device metal | vulkan | webgpu | rocm | android | iphone   ← same model, different silicon
```

Run it:

```python
from mlc_llm import MLCEngine

engine = MLCEngine(model="./dist/Qwen2.5-7B-q4f16_1-MLC",
                   model_lib="./dist/libs/Qwen2.5-7B-q4f16_1-cuda.so")
for chunk in engine.chat.completions.create(
        messages=[{"role": "user", "content": "Explain operator fusion in one paragraph."}],
        stream=True):
    print(chunk.choices[0].delta.content, end="", flush=True)
```

The quantization mode is a deployment lever you now understand from the inside (Lecture 4's dequant fusion is what makes it fast):

| Mode | Bits / format | Use |
|---|---|---|
| `q0f16` | fp16, no quant | quality reference / big-VRAM server |
| `q4f16_1` | 4-bit group weights, fp16 act | the workhorse — laptops, phones, most GPUs |
| `q4f16_awq` | 4-bit AWQ weights, fp16 act | better accuracy at 4-bit (activation-aware) |
| `q3f16_1` | 3-bit group | last resort for the tightest memory |

---

## 7. Measure it

Ship with numbers or you didn't ship. The deliverables differ by target but the discipline is identical: name the metric the target actually cares about.

| Target | Metrics that matter |
|---|---|
| Server GPU (Relax VM / MetaSchedule) | latency, throughput, GFLOP/s vs roofline, $/inference |
| LLM via MLC-LLM | **tokens/s** (prefill + decode), peak VRAM, model size on disk, time-to-first-token |
| Edge board (RPC-tuned) | latency under power cap, memory, device-tuned vs desktop-tuned delta |
| MCU (microTVM) | **flash size, SRAM peak (workspace arena), latency, energy/inference** — does it fit, does it hit the deadline |

For the LLM specifically, report tokens/s on at least two backends (e.g. CUDA and Metal, or CUDA and Vulkan) at the same quantization, plus the VRAM and disk footprint. The cross-backend spread is the lesson: **same Relax model, same dlight schedules, different silicon** — and you can read which backend's codegen is leaving performance on the table.

---

## 8. Course capstone

This is the artifact that proves the whole course. Pick one model you care about and produce a **compiled-model repo**:

1. **Import & build** (Lec 1): bring the model into Relax; build and run on the Relax VM; parity-check vs the framework reference.
2. **Schedule & tune** (Lec 2–3): MetaSchedule-tune it for **two targets** (e.g. x86 + CUDA, or CUDA + an edge board over RPC). Keep the tuning database in the repo.
3. **Optimize the graph** (Lec 4): show fusion (kernel-count before/after) and run **one BYOC experiment** (offload a subgraph to CUTLASS/TensorRT, or to CMSIS-NN if your target is an MCU), reporting the fraction of the graph offloaded.
4. **Ship** (Lec 5): export the artifact and load it from outside Python (C++ runtime, or MLC-LLM `MLCEngine` if it's an LLM, or microTVM MLF if it's an MCU).
5. **Benchmark table**: framework baseline vs un-tuned TVM vs tuned TVM vs (tuned + BYOC), with latency, GFLOP/s, **% of roofline peak**, and parity at every row.

That repo is a Level-5 artifact: another engineer clones it, runs your script, and reproduces your tuned numbers on the same hardware class. It is also a portfolio piece that says, unambiguously, *I can take a model from framework to metal and prove the speedup* — which is the entire job.

---

## 9. Course exit criteria

You have completed **TVM Deep Dives** when you can:

* Read an `IRModule` at every level — Relax graph, TIR `PrimFunc`, generated CUDA/C — and say what each pass changed (Lec 1).
* Hand-schedule a kernel and explain every primitive in **roofline** terms, including `tensorize` to a Tensor Core roof (Lec 2).
* Tune a kernel and a model with MetaSchedule, read the tuning curve to the knee, and measure on the **target device** over RPC (Lec 3).
* Fuse a graph and offload a subgraph via **BYOC**, and explain what a vendor implements to give a new accelerator a software stack (Lec 4).
* Ship the result to a non-Python target — server `.so`, MCU firmware, or an LLM across backends with MLC-LLM — and **defend the numbers against the roofline** (Lec 5).

If you can only make TVM *run* a model, you built a transpiler. If you can make it *beat the baseline and prove it on two targets*, you are an ML compiler engineer. That was the whole point.

---

## Key takeaways

- The compiler's value is the **standalone artifact**: `export_library` → a `.so`/`.tar` you load from C++/Rust/JS with only the small TVM runtime — no Python, no toolchain on the device.
- Three runtimes, three deployments: **GraphExecutor** (fixed-shape), **Relax VM** (dynamic/LLM), **AOT** (bare metal).
- **microTVM** compiles the model to **C** with an **AOT** executor and a **static memory plan**, packaged as **Model Library Format** for MCUs; BYOC reaches down to **CMSIS-NN** and **Ethos-U**. The metric is *fit + deadline*, not GFLOP/s.
- **MLC-LLM** is the whole stack pointed at LLMs, built on TVM Unity: `relax.frontend.nn` model → quantize → Relax/TIR opt (incl. dequant fusion) → **dlight** schedules → per-backend codegen → `MLCEngine`. Every stage is a prior lecture.
- **dlight vs MetaSchedule**: dlight gives instant, portable, ~80–90% schedules with no on-device tuning (essential when you ship to phones/web you can't tune on); MetaSchedule gives peak when you own the silicon. Mature stacks use both.
- Ship with numbers matched to the target — tokens/s + VRAM for an LLM, flash + SRAM + latency for an MCU — and always defend them against the roofline.

---

## References

- TVM deploy & runtime docs (`export_library`, GraphExecutor, Relax VM, C++ deploy): [https://tvm.apache.org/docs/how_to/deploy/](https://tvm.apache.org/docs/how_to/deploy/)
- microTVM (AOT, Model Library Format, Project API, Zephyr/Arduino): [https://tvm.apache.org/docs/topic/microtvm/index.html](https://tvm.apache.org/docs/topic/microtvm/index.html)
- Arm Ethos-U & CMSIS-NN via TVM BYOC: [https://tvm.apache.org/docs/](https://tvm.apache.org/docs/)
- MLC-LLM documentation (compile models, `MLCEngine`, quantization modes): [https://llm.mlc.ai/docs/](https://llm.mlc.ai/docs/)
- MLC-LLM source & supported model/backend matrix: [https://github.com/mlc-ai/mlc-llm](https://github.com/mlc-ai/mlc-llm)
- `tvm.dlight` (default GPU schedules): [https://tvm.apache.org/docs/](https://tvm.apache.org/docs/)
- *AI Inference Engineer 2026* — quantization and serving lectures, for the production-LLM context this feeds.

---

*Back to: [TVM Deep Dives index](README.md)*
