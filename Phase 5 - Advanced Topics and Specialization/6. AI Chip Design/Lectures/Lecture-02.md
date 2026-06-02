# Lecture 2: LLVM Passes & Code Generation for Hardware Backends

## Overview

Lecture 1 established that LLVM IR is the universal language between frontends and backends. This lecture addresses the next question: what happens to that IR between input and output? The answer is **passes** — modular transformations that analyze, optimize, and ultimately lower the IR into machine code. The core challenge is understanding how a matrix multiply expressed as simple loops in LLVM IR becomes a sequence of vector FMA instructions with optimal register usage and no unnecessary memory traffic. The mental model is a **pipeline of transformations**: each pass reads the IR, makes a specific improvement, and hands the result to the next pass. For AI hardware engineers, this is directly relevant because (1) when you write a backend for a custom accelerator, you need to implement target-specific passes, and (2) understanding what the optimizer can and cannot do tells you what your hardware must handle in silicon vs. what the compiler can fix in software.

---

## The Pass Pipeline

LLVM organizes passes into a pipeline that runs in a specific order. The `opt` tool runs middle-end passes on LLVM IR; `llc` runs the backend (code generation) passes.

```
LLVM IR (.ll / .bc)
    │
    ▼
┌────────────────────────────────────────────┐
│          Middle-End Passes (opt)           │
│                                            │
│  Analysis:  dominator tree, alias analysis │
│  Transform: SROA, GVN, LICM, vectorize    │
│  Cleanup:   DCE, SimplifyCFG              │
└────────────────────┬───────────────────────┘
                     │
                     ▼  (optimized LLVM IR)
┌────────────────────────────────────────────┐
│          Backend Passes (llc)              │
│                                            │
│  Instruction Selection (DAG → MachineInstr)│
│  Register Allocation                       │
│  Instruction Scheduling                    │
│  Machine-specific optimizations            │
│  Code Emission (assembly / object)         │
└────────────────────────────────────────────┘
                     │
                     ▼
              Native Code (.o / .ptx / .s)
```

---

## Middle-End Optimization Passes

These passes work on LLVM IR and are **target-independent** — they improve the code regardless of which chip will run it.

### Pass Categories

| Category | Purpose | Key Passes |
|---|---|---|
| **Scalar** | Optimize individual values and expressions | SROA, GVN, SCCP, InstCombine |
| **Loop** | Optimize loops (the critical path for AI kernels) | LICM, IndVarSimplify, LoopUnroll, LoopVectorize |
| **Interprocedural** | Optimize across function boundaries | Inlining, DeadArgElim, GlobalOpt |
| **Vectorization** | Convert scalar code to vector operations | LoopVectorize, SLPVectorize |
| **Cleanup** | Remove redundant code after other passes | DCE, SimplifyCFG, MergedLoadStoreMotion |

### Passes Critical for AI Workloads

**1. Loop Vectorization (LoopVectorize)**

This is the **single most important pass** for AI kernel performance on CPUs. It transforms scalar loops into **vector operations** that use SIMD hardware.

```llvm
; BEFORE vectorization — processes one element at a time
loop:
  %i = phi i64 [ 0, %entry ], [ %i.next, %loop ]
  %a_ptr = getelementptr float, ptr %A, i64 %i
  %b_ptr = getelementptr float, ptr %B, i64 %i
  %a = load float, ptr %a_ptr
  %b = load float, ptr %b_ptr
  %mul = fmul float %a, %b
  %c_ptr = getelementptr float, ptr %C, i64 %i
  store float %mul, ptr %c_ptr
  %i.next = add i64 %i, 1
  %cmp = icmp slt i64 %i.next, %n
  br i1 %cmp, label %loop, label %exit

; AFTER vectorization (VF=8 on AVX-256) — processes 8 elements at a time
loop.vec:
  %i = phi i64 [ 0, %entry ], [ %i.next, %loop.vec ]
  %a_ptr = getelementptr float, ptr %A, i64 %i
  %b_ptr = getelementptr float, ptr %B, i64 %i
  %va = load <8 x float>, ptr %a_ptr, align 32
  %vb = load <8 x float>, ptr %b_ptr, align 32
  %vmul = fmul <8 x float> %va, %vb
  %c_ptr = getelementptr float, ptr %C, i64 %i
  store <8 x float> %vmul, ptr %c_ptr, align 32
  %i.next = add i64 %i, 8
  %cmp = icmp slt i64 %i.next, %n
  br i1 %cmp, label %loop.vec, label %exit
```

The vectorizer needs to prove that iterations are **independent** (no loop-carried dependencies) and that memory accesses **don't alias**. This is why `restrict` pointers and alignment annotations matter so much for AI kernel performance.

**2. Loop Unrolling (LoopUnroll)**

Reduces loop overhead and exposes more instruction-level parallelism (ILP) to the hardware scheduler.

```llvm
; BEFORE: tight loop with branch every iteration
; AFTER (unroll factor 4): 4 iterations inlined, branch every 4th
loop.unrolled:
  %v0 = load <8 x float>, ptr %p0     ; iteration 0
  %v1 = load <8 x float>, ptr %p1     ; iteration 1
  %v2 = load <8 x float>, ptr %p2     ; iteration 2
  %v3 = load <8 x float>, ptr %p3     ; iteration 3
  ; ... compute on all four ...
  ; single branch back to loop header
```

For AI kernels, unrolling is essential to keep the pipeline full — especially on GPUs where each warp needs enough independent instructions to hide memory latency.

**3. Inlining**

Replaces a function call with the function body. Critical for AI compilers because small helper functions (activation functions, quantization scales) become zero-cost when inlined.

```llvm
; BEFORE: call overhead for every element
define float @relu(float %x) {
  %cmp = fcmp ogt float %x, 0.0
  %r = select i1 %cmp, float %x, float 0.0
  ret float %r
}

; AFTER inlining: relu body is directly in the loop
; No call/ret overhead, enables further vectorization
```

**4. Global Value Numbering (GVN)**

Eliminates redundant computations. If two expressions compute the same value, GVN keeps one and replaces the other with a reference.

```llvm
; BEFORE: redundant address computation
%ptr1 = getelementptr float, ptr %base, i64 %idx
%val1 = load float, ptr %ptr1
; ... some code that doesn't modify *ptr1 ...
%ptr2 = getelementptr float, ptr %base, i64 %idx   ; same as ptr1!
%val2 = load float, ptr %ptr2                        ; same as val1!

; AFTER GVN: second load eliminated
%ptr1 = getelementptr float, ptr %base, i64 %idx
%val1 = load float, ptr %ptr1
; ... uses %val1 instead of %val2 ...
```

**5. Loop-Invariant Code Motion (LICM)**

Moves computations that don't change across loop iterations out of the loop.

```llvm
; BEFORE: scale factor recomputed every iteration (expensive division)
loop:
  %scale = fdiv float 1.0, %max_val    ; loop-invariant!
  %val = load float, ptr %p
  %scaled = fmul float %val, %scale
  ; ...

; AFTER LICM: division moved before the loop
preheader:
  %scale = fdiv float 1.0, %max_val    ; computed once
  br label %loop
loop:
  %val = load float, ptr %p
  %scaled = fmul float %val, %scale    ; uses precomputed scale
```

---

## Alias Analysis: The Gatekeeper of Optimization

Many critical optimizations (vectorization, load elimination, code motion) require the compiler to prove that **two pointers don't refer to the same memory**. This is **alias analysis**.

```llvm
; Can the compiler vectorize this loop?
define void @add(ptr %A, ptr %B, ptr %C, i64 %n) {
  ; If A == C (output aliases input), vectorization changes semantics!
  ; The compiler must prove A != C or insert runtime checks
}
```

**Why this matters for AI compilers:** TVM and Triton annotate their buffers with `noalias` metadata because they know tensor buffers don't overlap. Without this, LLVM would conservatively refuse to vectorize most tensor operations.

```llvm
; TVM-generated IR includes noalias:
define void @fused_relu(ptr noalias %input, ptr noalias %output, i64 %n) {
  ; Compiler can freely vectorize — buffers guaranteed not to overlap
}
```

> **Key Insight:** When building an AI compiler, the most impactful thing you can do for code quality is provide accurate alias information. A single missing `noalias` annotation can prevent vectorization of an entire kernel, causing a 4–16× performance regression on CPUs.

---

## The Backend: From LLVM IR to Machine Code

The backend (code generator) transforms LLVM IR into **native instructions** for a specific target. This is where AI hardware engineers spend **most of their time** when building a compiler for a custom chip.

### Backend Pipeline

```
LLVM IR
    │
    ▼
┌─────────────────────────────────┐
│ 1. Instruction Selection        │  IR → SelectionDAG → MachineDAG
│    (pattern matching)           │  "Which hardware instruction
│                                 │   implements this IR operation?"
├─────────────────────────────────┤
│ 2. Instruction Scheduling       │  Reorder to minimize pipeline
│    (pre-register-allocation)    │  stalls and maximize ILP
├─────────────────────────────────┤
│ 3. Register Allocation          │  Map virtual registers to
│    (graph coloring / linear scan)│  physical registers
├─────────────────────────────────┤
│ 4. Instruction Scheduling       │  Post-RA scheduling: account for
│    (post-register-allocation)   │  actual register constraints
├─────────────────────────────────┤
│ 5. Machine-Specific Passes      │  Peephole optimization, branch
│                                 │  relaxation, constant pooling
├─────────────────────────────────┤
│ 6. Code Emission                │  Encode to assembly (.s) or
│    (MCInst → bytes)             │  machine code (.o)
└─────────────────────────────────┘
```

### Stage 1: Instruction Selection (ISel)

ISel maps LLVM IR patterns to target-specific machine instructions using **TableGen** pattern matching.

**TableGen** is LLVM's domain-specific language for describing target architectures. You write `.td` files that declare registers, instructions, and patterns:

```tablegen
// Define a register class for 32 general-purpose registers
def GPR : RegisterClass<"MyAccel", [i32, f32], 32,
                         (sequence "R%u", 0, 31)>;

// Define an instruction: MAC r1, r2, r3 → r1 = r1 + r2 * r3
def MAC : Instruction {
  let OutOperandList = (outs GPR:$dst);
  let InOperandList = (ins GPR:$acc, GPR:$src1, GPR:$src2);
  let AsmString = "mac $dst, $acc, $src1, $src2";

  // Pattern: match fmuladd intrinsic → emit MAC instruction
  let Pattern = [(set GPR:$dst,
                   (fma GPR:$src1, GPR:$src2, GPR:$acc))];
}
```

When LLVM sees an `fma` node in the SelectionDAG, it matches the pattern and emits a `MAC` instruction. This is how your custom accelerator's hardware operations get connected to LLVM IR.

### Stage 2: Instruction Scheduling

The scheduler reorders instructions to maximize utilization of execution units and hide latency. It uses a **scheduling model** that describes your hardware's pipeline.

```tablegen
// Define pipeline stages for a simple accelerator
def MyPipeline : SchedMachineModel {
  let IssueWidth = 2;           // 2 instructions per cycle
  let LoadLatency = 3;          // loads take 3 cycles
  let MispredictPenalty = 10;   // branch mispredict cost
}

// MAC instruction uses the multiply unit for 2 cycles
def : WriteRes<WriteFMul, [MulUnit]> { let Latency = 2; }
```

The scheduler uses this model to interleave loads with computes — while one MAC is executing, the next iteration's load can be in-flight. This is the compiler equivalent of **software pipelining**, essential for keeping accelerator datapaths busy.

### Stage 3: Register Allocation

Maps unlimited virtual registers to the finite physical register file. For AI accelerators with large register files (GPUs have 65536 registers per SM), **register pressure is often the bottleneck** — running out of registers forces values to **spill to memory**, which is catastrophic for performance.

| Algorithm | Speed | Code Quality | Used When |
|---|---|---|---|
| Linear scan | Fast | Good | JIT compilation, -O1 |
| Graph coloring (Greedy) | Moderate | Best | Ahead-of-time, -O2/-O3 |
| PBQP | Slow | Best for irregular architectures | Special targets |

> **Key Insight:** For AI accelerators, register allocation quality directly determines kernel performance. A matrix multiply tile that fits in registers runs at compute speed; one that spills runs at memory speed. This is why GPU programming cares so much about "occupancy" — it's really about register pressure. When designing your accelerator's register file size, simulate the register allocation on representative AI kernels to find the right trade-off between area and spill frequency.

---

## Writing a Custom Backend: The Minimal Steps

To make LLVM target your custom AI accelerator, you need to provide these components:

```
llvm/lib/Target/MyAccel/
├── MyAccelTargetMachine.cpp      # Entry point: creates the target
├── MyAccelInstrInfo.td           # TableGen: instruction definitions
├── MyAccelRegisterInfo.td        # TableGen: register file description
├── MyAccelInstrInfo.cpp          # Instruction semantics, legalization
├── MyAccelISelDAGToDAG.cpp       # Custom instruction selection patterns
├── MyAccelFrameLowering.cpp      # Stack frame layout
├── MyAccelAsmPrinter.cpp         # Emit assembly text
└── MyAccelSubtarget.td           # TableGen: CPU variants, features
```

**Minimum viable backend workflow:**

1. **Register the target:** `TargetRegistry::RegisterTarget(TheMyAccelTarget, "myaccel", ...)`
2. **Define registers:** List your register file in TableGen
3. **Define instructions:** Map LLVM IR operations to your hardware instructions
4. **Handle legalization:** Tell LLVM which types and operations your hardware supports natively vs. which must be expanded (e.g., "my chip has no 64-bit divide — expand to a library call")
5. **Implement AsmPrinter:** Emit the text assembly for your chip's assembler

```cpp
// Legalization example: "my accelerator only supports i8 and i32 multiply"
void MyAccelTargetLowering::setOperationAction() {
  // i8 multiply: native hardware instruction
  setOperationAction(ISD::MUL, MVT::i8, Legal);

  // i32 multiply: native
  setOperationAction(ISD::MUL, MVT::i32, Legal);

  // i16 multiply: promote to i32, then multiply
  setOperationAction(ISD::MUL, MVT::i16, Promote);

  // i64 multiply: expand to two i32 multiplies + adds
  setOperationAction(ISD::MUL, MVT::i64, Expand);

  // f32 fused multiply-add: native (our MAC unit)
  setOperationAction(ISD::FMA, MVT::f32, Legal);
}
```

---

## Case Study: The NVPTX Backend (CUDA)

The NVPTX backend is how LLVM compiles to NVIDIA GPUs — and the model for how Triton generates GPU code.

**Key design decisions:**
- Emits **PTX** (assembly) rather than binary SASS — NVIDIA's `ptxas` handles final instruction scheduling and register allocation
- Maps LLVM address spaces to GPU memory: 0→generic, 1→global, 3→shared, 4→constant
- Defines intrinsics for thread indexing (`threadIdx.x`), barriers (`__syncthreads`), warp shuffles, tensor core operations
- No register allocation needed in the backend — PTX uses virtual registers; `ptxas` does physical register allocation

```
LLVM IR → NVPTX Backend → PTX assembly → ptxas → SASS binary (cubin)
```

**What Triton does:**
```
Triton Python → Triton IR → LLVM IR (NVPTX) → PTX → ptxas → cubin
                                                  ↑
                                    LLVM handles this step
```

This is why understanding LLVM's backend pipeline is essential for understanding how AI compilers like Triton actually work.

---

## Viewing the Pass Pipeline

```bash
# See all passes that -O2 runs
opt -O2 -print-pipeline-passes input.ll -o /dev/null

# Run specific passes and inspect the result
opt -passes="loop-vectorize,loop-unroll" input.ll -S -o output.ll

# See the backend pipeline for a specific target
llc -O2 -debug-pass=Structure input.ll -o /dev/null 2>&1 | head -50

# View SelectionDAG (instruction selection) for debugging
llc -view-isel-dags input.ll    # generates a .dot graph

# View the final machine instructions before emission
llc -print-after-all input.ll -o /dev/null 2>&1
```

---

## Hands-On Exercises

1. **Observe vectorization:** Write a simple element-wise ReLU loop in C. Compile with `clang -S -emit-llvm -O2` and examine the generated LLVM IR. Then compile with `clang -S -O2 -march=skylake` and examine the x86 assembly — find the `vmaxps` or `vblendvps` instructions. Repeat with `-march=armv8-a+simd` for NEON.

2. **Kill vectorization with aliasing:** Take the ReLU kernel and remove the `restrict` qualifier from the pointer parameters. Recompile and observe that the vectorizer either gives up or inserts expensive runtime alias checks. Add `noalias` metadata in the `.ll` file and verify vectorization returns.

3. **Explore the NVPTX backend:** Write a simple LLVM IR function that loads from address space 1 (global), multiplies by a constant, and stores back. Compile with `llc -march=nvptx64 -mcpu=sm_80`. Read the PTX output — identify `ld.global`, `mul.f32`, and `st.global` instructions.

4. **Backend skeleton:** Using LLVM's documentation, sketch a TableGen file for a hypothetical accelerator with: 16 general-purpose registers (R0–R15), `LOAD`, `STORE`, `ADD`, `MUL`, and `MAC` (fused multiply-add) instructions. Define the pattern for `MAC` to match `fadd(fmul(a, b), c)`.

---

## Key Takeaways

| Concept | Why It Matters for AI Hardware |
|---|---|
| Loop vectorization | Converts scalar AI kernels to SIMD — 4–16× speedup on CPUs |
| Alias analysis | Without `noalias`, vectorization fails — the single biggest code quality factor |
| Instruction selection | How your custom instructions get used by LLVM |
| TableGen | The declarative language for describing your chip to LLVM |
| Register allocation | Determines whether your kernel runs at compute or memory speed |
| Scheduling model | Tells LLVM how to interleave instructions for your pipeline |
| NVPTX backend | The reference for how an AI-relevant backend works |

---

## Resources

* **[Writing an LLVM Backend](https://llvm.org/docs/WritingAnLLVMBackend.html):** Official guide for implementing a new target.
* **[LLVM TableGen Reference](https://llvm.org/docs/TableGen/):** Syntax and semantics of `.td` files.
* **[NVPTX Backend Documentation](https://llvm.org/docs/NVPTXUsage.html):** How LLVM targets NVIDIA GPUs.
* **[LLVM's Analysis and Transform Passes](https://llvm.org/docs/Passes.html):** Reference for all built-in passes.
* **"Engineering a Compiler" by Cooper & Torczon:** Textbook covering instruction selection, register allocation, and scheduling in depth.
* **Triton source: `python/triton/backends/nvidia/compiler.py`:** How Triton invokes the LLVM NVPTX backend.
