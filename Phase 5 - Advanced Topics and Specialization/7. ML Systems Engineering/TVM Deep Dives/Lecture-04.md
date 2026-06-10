# Lecture 04 - Relax in Depth: Dynamic Shapes, Operator Fusion, and Bring Your Own Codegen (BYOC)

**Collection:** [TVM Deep Dives](README.md) | **Previous:** [← Lecture 03](Lecture-03.md) | **Next:** [Lecture 05](Lecture-05.md)

---

Lectures 2 and 3 lived *inside a kernel* — one operator, scheduled and tuned. This lecture zooms back out to the **graph**, because two of the largest wins in a real deployment are not inside any single kernel:

* **Operator fusion** — collapsing a chain of ops into one kernel kills the DRAM round-trips and launch overhead *between* them. On memory-bound models this is often a bigger win than tuning any individual op.
* **Bring Your Own Codegen (BYOC)** — handing parts of the graph to an external library or a *custom accelerator's* compiler, while TVM handles the rest. This is the mechanism by which a new chip gets a software stack.

Both live at the **Relax** level, and both depend on Relax's defining capability: **first-class symbolic shapes**, the thing that lets one compiled artifact serve a variable batch size or a growing LLM sequence. We take all three in turn, then put them together — because BYOC on a dynamic-shape graph with fusion is, concretely, what bringing up an accelerator for modern models looks like.

---

## Learning objectives

By the end of this lecture, you should be able to:

1. Read a Relax function in depth: dataflow blocks, bindings, `call_tir`, struct info, pure vs impure.
2. Write and reason about **symbolic / dynamic shapes** (`R.Tensor(("n", 4096), ...)`), and explain why LLMs require them.
3. Run the graph-level optimization passes — **`FuseOps` → `FuseTIR`**, legalize, layout transform, memory planning — and explain what each buys.
4. Explain why fusion matters in **roofline / DRAM-traffic** terms, including the dequant-matmul-epilogue fusion that makes quantized LLMs fast.
5. Execute the **BYOC** flow: `partition_for_<backend>` → `MergeCompositeFunctions` → `RunCodegen`, offloading a subgraph to CUTLASS/TensorRT.
6. Describe what a chip vendor must implement to give their accelerator a BYOC backend — the L4 accelerator-software job.

---

## 1. Relax up close

A Relax function is a typed dataflow program. Here is one with the features that matter named:

```python
from tvm.script import relax as R, tir as T

@R.function
def main(x: R.Tensor(("n", 4096), "float16"),          # ← symbolic dim "n"
         w1: R.Tensor((4096, 11008), "float16"),
         w2: R.Tensor((11008, 4096), "float16")) -> R.Tensor(("n", 4096), "float16"):
    n = T.int64()                                       # the symbolic shape variable
    with R.dataflow():                                  # a pure, optimizer-owned region
        lv0  = R.matmul(x, w1)                          # (n, 11008)
        lv1  = R.nn.silu(lv0)                           # elementwise
        lv2  = R.matmul(lv1, w2)                        # (n, 4096)
        out  = R.add(x, lv2)                            # residual
        R.output(out)
    return out
```

Read the structure:

* **Struct info** is Relax's type system: `R.Tensor(shape, dtype)` carries *both* shape and dtype as part of the type. The compiler reasons about shapes statically wherever it can — that is what makes fusion and memory planning possible.
* **Dataflow block** (`R.dataflow()`) marks a **pure, side-effect-free** region. Inside it, the optimizer may reorder, fuse, eliminate, and rewrite bindings freely. The `R.output(...)` line names which values escape the block. Operations with side effects (mutating a KV-cache, I/O) live *outside* dataflow blocks, which is how Relax cleanly separates "optimizable math" from "stateful effects."
* **Bindings** (`lv0 = ...`) are single-assignment. The graph is a DAG of these.
* **`call_tir`** (from Lecture 1) appears once high-level ops are lowered — the graph reaching down into scheduled TIR.

That clean separation — typed shapes, a pure dataflow region, explicit effects — is what makes Relax aggressively optimizable where a raw imperative trace is not.

---

## 2. Symbolic shapes: the LLM-shaped problem

A vision model often has one fixed input shape; you can compile for `(1, 3, 224, 224)` and be done. An **LLM cannot**. The sequence length grows token by token during decode; the batch size varies per request; the KV-cache length is different every step. Compiling a separate binary per sequence length is absurd.

Relax's answer is the **symbolic shape variable**. The `"n"` in `R.Tensor(("n", 4096), ...)` above is not a constant — it is a variable resolved at runtime. The Relax VM tracks the concrete value of `n` as the program runs and threads it through every shape computation.

```text
   static-shape compiler            symbolic-shape compiler (Relax)
   ───────────────────              ───────────────────────────────
   compile for n = 1                compile ONCE for n = "n"
   compile for n = 2                VM binds n = 7 this call,
   compile for n = 4                          n = 8 next call,
   ... (impossible for LLMs)                  n = 512 for a prefill
```

Two mechanisms support it:

* **Shape expressions** — shapes can be arithmetic over symbolic vars (`(n, n + 1, 4096)`), so the compiler reasons about, e.g., a KV-cache of length `n` becoming `n + 1`.
* **`R.match_cast`** — at a boundary where a shape is dynamically known (a data-dependent op like `unique`, or an external call), `match_cast` refines an opaque shape into one with named symbolic vars so downstream code can be optimized again.

This is *the* reason MLC-LLM (Lecture 5) is built on Relax and not Relay: variable-length sequences, KV-caches, and ragged batches are dynamic-shape problems, and Relax made dynamic shapes a first-class citizen of the IR rather than an afterthought. When you hear "TVM Unity made LLMs work," this is the concrete capability they mean.

---

## 3. Graph-level passes: where the free wins live

Before any kernel runs, a sequence of **graph passes** rewrites the Relax module. The default `relax.get_pipeline("zero")` bundles a sane order; in production you assemble your own. The ones that move the needle:

| Pass | What it does | Why it matters |
|---|---|---|
| `LegalizeOps` | high-level op → `call_tir` into a TIR `PrimFunc` | bridges graph to kernel level (Lecture 1) |
| `AnnotateTIROpPattern` | tags each TIR func: injective / reduction / etc. | tells the fuser what is safe to merge |
| **`FuseOps`** | groups op chains into fused Relax sub-functions | the fusion *decision* |
| **`FuseTIR`** | merges a fused group's TIR into **one** `PrimFunc` | the fusion *realization* — one kernel |
| `ConvertLayout` / layout transform | rewrite NCHW↔NHWC↔blocked | match the layout the target wants |
| `FoldConstant` | evaluate constant subgraphs at compile time | fold scales, fold reshapes of weights |
| `StaticPlanBlockMemory` | plan & reuse intermediate buffers | shrink peak memory, reuse allocations |
| `DeadCodeElimination` / canonicalize | prune, normalize | clean graph for later passes |

You apply them as a pipeline:

```python
from tvm import relax
seq = tvm.transform.Sequential([
    relax.transform.LegalizeOps(),
    relax.transform.AnnotateTIROpPattern(),
    relax.transform.FuseOps(),
    relax.transform.FuseTIR(),
    relax.transform.StaticPlanBlockMemory(),
    relax.transform.FoldConstant(),
])
mod = seq(mod)
mod.show()        # count the PrimFuncs before vs after — fusion collapsed several into one
```

---

## 4. Why fusion is often the biggest single win

Take the SiLU-gated FFN from §1: `matmul → silu → matmul → add`. Unfused, that is four kernels, and between each one the intermediate tensor is **written to DRAM and read back**:

```text
   UNFUSED                                     FUSED (epilogue into the matmul)
   matmul ─► [write 11008·n to DRAM]           matmul ─┐
   silu   ─► [read it, write it back]                  ├─ silu + add done in the
   matmul ─► [read it, write 4096·n]                   │  matmul's epilogue, in registers
   add    ─► [read, write]                      matmul ─┘  → intermediates never touch DRAM
   = 4 launches, several DRAM round-trips      = fewer launches, DRAM traffic slashed
```

For a **memory-bound** model (which decode-phase LLMs emphatically are — see the Edge LLM and AI Inference Engineer courses), the elementwise ops (`silu`, `add`, norms, activations) cost almost nothing in FLOPs but everything in DRAM traffic if they are separate kernels. Fusing them into a matmul's epilogue means the intermediate **never leaves registers**. The roofline reading: fusion raises arithmetic intensity by *deleting* the memory traffic of the intermediates, moving the fused op rightward toward the compute roof.

The most important fused pattern for 2026 LLMs is **dequantize-matmul-epilogue**. A 4-bit weight has to be dequantized to fp16 before the matmul; doing that as a separate pass would write the full-precision weights to DRAM — defeating the entire point of 4-bit storage. So MLC-LLM's pipeline (Lecture 5) fuses **dequant + matmul + bias/activation** into a single kernel (`FuseDequantizeMatmulEwise`): the weights are dequantized *in registers, on the fly*, consumed immediately by the matmul, and never materialized in full precision. That one fusion is a large part of why a 7B model in 4-bit runs fast on a laptop GPU.

This is why a senior engineer profiles for kernel *count* and *DRAM traffic*, not just per-kernel GFLOP/s. Ten perfectly-tuned kernels that should have been one fused kernel is a slower model than one decently-tuned fused kernel.

---

## 5. BYOC: handing the graph to someone else's compiler

Sometimes the best kernel is not one TVM should generate. NVIDIA's CUTLASS has battle-hardened fused-attention and GEMM kernels; TensorRT has years of tactic tuning; a custom NPU has an instruction TVM knows nothing about. **Bring Your Own Codegen** lets TVM *partition* the graph, hand the matching parts to an external codegen, and keep generating code for the rest — then stitch them into one runnable module.

The flow has three moves:

```text
   full Relax graph
   ┌──────────────────────────────────────────────┐
   │ conv → bn → relu → matmul → softmax → add ... │
   └──────────────────────────────────────────────┘
        │  ① partition_for_<backend>   (pattern-match subgraphs the backend supports)
        ▼
   ┌────────────────────────┬─────────────────────┐
   │ composite fn → CUTLASS │  rest → TVM codegen  │
   │  matmul (+epilogue)    │  softmax, odd ops    │
   └────────────────────────┴─────────────────────┘
        │  ② MergeCompositeFunctions   (group adjacent offloadable regions)
        │  ③ RunCodegen                (invoke the external compiler per region)
        ▼
   ONE runtime module: external kernels + TVM kernels, glued by the Relax VM
```

In code, the high-level convenience path (CUTLASS shown; TensorRT, cuDNN, DNNL/oneDNN are analogous):

```python
from tvm import relax
from tvm.relax.backend.contrib.cutlass import partition_for_cutlass

# ① tag every subgraph that matches a CUTLASS pattern as an offloadable composite function
mod = partition_for_cutlass(mod)

# ②③ merge adjacent offload regions, then run the external codegen on each
mod = relax.transform.RunCodegen()(mod)

# build as usual — TVM generates the rest, the VM links the external kernels in
ex = relax.build(mod, target="cuda")
vm = relax.VirtualMachine(ex, tvm.cuda(0))
```

Under the convenience function is the **general** mechanism, which is what you use for a backend TVM doesn't already ship:

```python
from tvm.relax.dpl.pattern import is_op, wildcard

# define the patterns your backend can handle (e.g. matmul + bias + relu)
pat = is_op("relax.matmul")(wildcard(), wildcard())
mod = relax.transform.FuseOpsByPattern(
    [("my_npu.matmul_bias_relu", pat)],
    annotate_codegen=True,            # mark the fused fn for an external codegen named "my_npu"
)(mod)
mod = relax.transform.MergeCompositeFunctions()(mod)
mod = relax.transform.RunCodegen()(mod)   # dispatches "my_npu" regions to your registered codegen
```

Anything the backend can't match stays as ordinary TVM-generated kernels. **The graph is split, not surrendered.** That is the crucial property: you never have to support the *whole* op set to be useful — you offload what you do well and let TVM cover the long tail (and tune it with MetaSchedule).

---

## 6. BYOC is how a new accelerator gets a software stack

This is the part that makes Lecture 4 a *hardware* lecture, and it is the day job of an accelerator-software engineer.

A chip company builds a fast NPU. It does not want to write — and cannot keep up with — a frontend for PyTorch, ONNX, JAX, and every model architecture. With BYOC it doesn't have to. It implements, **once**:

1. A **pattern table**: the op patterns the NPU's kernels cover (its GEMM, its conv, its attention, its activation set).
2. A **codegen**: given an offloaded subgraph, emit calls into the NPU's runtime / driver (or its own compiler).
3. A **runtime module**: load and execute those kernels on the device, interoperating with TVM's runtime for the rest.

In return it gets, **for free**, every model any TVM frontend can import. TVM handles import, dynamic shapes, fusion of the parts that stay on host/GPU, memory planning, and the glue; the NPU handles the ops it's good at. The same mechanism is how TVM's own open accelerator, **VTA** (Versatile Tensor Accelerator), plugs in, and how `tensorize` (Lecture 2) extends downward to custom instructions.

```text
   vendor implements ONCE:                  gets for free:
   ┌─────────────────────┐                  ┌──────────────────────────┐
   │ pattern table       │                  │ PyTorch / ONNX / JAX      │
   │ codegen → driver    │ ── BYOC ──►       │ import + dynamic shape    │
   │ runtime module      │                  │ fusion + memory planning  │
   └─────────────────────┘                  │ MetaSchedule for the rest │
                                            └──────────────────────────┘
```

When an interviewer asks "how would you bring up a software stack for a new accelerator without writing a compiler from scratch," **BYOC on TVM is the answer**, and §5 is the concrete plan: pattern table, codegen, runtime module, partition the graph, offload what you cover, let TVM carry the tail.

---

## 7. Hands-on + Measure it

Put it together on a real model (a small CNN for CUTLASS/TensorRT, or a transformer block for the fusion story).

**Fusion:**

```python
mod_unfused = relax.transform.LegalizeOps()(mod)          # lowered, NOT fused
mod_fused   = relax.get_pipeline("zero")(mod)             # includes FuseOps + FuseTIR

def count_prim_funcs(m):
    return sum(1 for gv in m.functions if isinstance(m[gv], tvm.tir.PrimFunc))

print("kernels unfused:", count_prim_funcs(mod_unfused))
print("kernels fused:  ", count_prim_funcs(mod_fused))    # fewer — chains collapsed
```

**BYOC offload:**

```python
mod_byoc = partition_for_cutlass(mod)           # or partition_for_tensorrt
mod_byoc = relax.transform.RunCodegen()(mod_byoc)
# build all three, time end-to-end on the Relax VM, verify parity vs the framework reference
```

Report a table:

| Build | End-to-end latency | # kernels | % graph offloaded | Parity |
|---|---|---|---|---|
| TVM, unfused | — | many | 0% | ref |
| TVM, fused (`zero`) | — | fewer | 0% | ✓ |
| TVM + tuned (MetaSchedule) | — | fewer | 0% | ✓ |
| TVM + BYOC (CUTLASS/TRT) | — | — | __% | ✓ |

The three numbers that tell the story: **fusion** should cut kernel count and latency on the memory-bound parts; **% of graph offloaded** tells you how much of the model the BYOC backend captured (and what TVM still had to generate); **parity** must hold at every row — a fused or offloaded graph that changed the numbers is a bug, not an optimization.

---

## 8. Mini-lab: fuse, offload, and reason about coverage

1. Import a model; build it three ways — unfused, fused (`zero` pipeline), and fused+tuned (MetaSchedule from Lecture 3). Record latency and kernel count. Identify one fused group in the TVMScript and name the DRAM round-trips it eliminated.
2. Add a symbolic batch/sequence dimension to the input and recompile. Confirm one binary serves multiple input sizes on the Relax VM. (This is the LLM capability in miniature.)
3. Offload with `partition_for_cutlass` or `partition_for_tensorrt`. Measure latency and compute the **fraction of the graph** that went to the external backend (offloaded kernels / total). Write two sentences on which ops the backend *didn't* take and why TVM had to keep them.

Deliverable: the §7 table, the named fused group, the dynamic-shape proof, and the BYOC coverage analysis. That last analysis — "the backend captured 70% of the FLOPs but left these three ops to TVM" — is exactly the report an accelerator-software engineer writes when scoping a new backend.

---

## Key takeaways

- The largest deployment wins often live at the **graph** level, not inside any kernel: **fusion** and **BYOC**.
- Relax is a typed dataflow IR: struct info carries shape+dtype, **dataflow blocks** mark pure optimizable regions, effects live outside them. This cleanliness is what enables aggressive graph rewriting.
- **Symbolic shapes** (`R.Tensor(("n", ...))`) let one binary serve variable batch/sequence sizes — the capability LLMs require and the reason MLC-LLM is built on Relax.
- **`FuseOps` → `FuseTIR`** collapse op chains into single kernels, deleting the DRAM round-trips between them. On memory-bound models this beats per-kernel tuning. **Dequant-matmul-epilogue fusion** is why 4-bit LLMs run fast.
- **BYOC** partitions the graph and offloads matching subgraphs to an external codegen (CUTLASS / TensorRT / cuDNN / a custom NPU) while TVM generates and tunes the rest: `partition_for_<backend>` → `MergeCompositeFunctions` → `RunCodegen`.
- BYOC is **how a new accelerator gets a software stack**: implement a pattern table, a codegen, and a runtime module once; receive every TVM-importable model for free. This is the L4 accelerator-software engineer's job.

---

## References

- Relax / TVM Unity documentation (struct info, dataflow, dynamic shape): [https://tvm.apache.org/docs/reference/api/python/relax/relax.html](https://tvm.apache.org/docs/reference/api/python/relax/relax.html)
- Lai et al., "Relax: Composable Abstractions for End-to-End Dynamic Machine Learning," 2023: [https://arxiv.org/abs/2311.02103](https://arxiv.org/abs/2311.02103)
- TVM "Bring Your Own Codegen to TVM" developer guide: [https://tvm.apache.org/docs/dev/how_to/relay_bring_your_own_codegen.html](https://tvm.apache.org/docs/dev/how_to/relay_bring_your_own_codegen.html)
- Relax BYOC backends (`partition_for_cutlass`, `partition_for_tensorrt`) in `tvm.relax.backend.contrib`: [https://tvm.apache.org/docs/](https://tvm.apache.org/docs/)
- VTA (Versatile Tensor Accelerator) — TVM's open accelerator + BYOC/tensorize example: [https://tvm.apache.org/docs/topic/vta/index.html](https://tvm.apache.org/docs/topic/vta/index.html)
- MLC-LLM compiler pass pipeline (`FuseDequantizeMatmulEwise` and the dlight path): [https://llm.mlc.ai/docs/](https://llm.mlc.ai/docs/)

---

*Next: [Lecture 05 — Shipping it: runtime, microTVM, and LLMs with MLC-LLM](Lecture-05.md)*
