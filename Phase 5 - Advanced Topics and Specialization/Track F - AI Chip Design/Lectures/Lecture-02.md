# Lecture 2: LLVM Passes & Code Generation for Hardware Backends

## Overview

Lecture 1 established that LLVM IR is the universal language between frontends and backends. This lecture addresses the next question: what happens to that IR between input and output? The answer is **passes** — modular transformations that analyze, optimize, and ultimately lower the IR into machine code. The core challenge is understanding how a matrix multiply expressed as simple loops in LLVM IR becomes a sequence of vector FMA instructions while respecting register limits, dependencies, and memory behavior. The mental model is a **pipeline of transformations**: each pass reads the IR, makes a specific improvement, and hands the result to the next pass. For AI hardware engineers, this is directly relevant because (1) when you write a backend for a custom accelerator, you need to implement target-specific passes, and (2) understanding what the optimizer can and cannot do tells you what your hardware must handle in silicon vs. what the compiler can fix in software.

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

These passes work on LLVM IR and are shared across targets. Many consult target cost information; a legal transformation is not necessarily profitable on every chip.

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

This is an important pass for suitable CPU loops. Data movement, algorithm choice, library calls, fusion, and parallelism can matter more for a particular workload. It transforms scalar loops into **vector operations** that use SIMD hardware.

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

; AFTER vectorization: illustrative vector loop for n>0 and n divisible by 8.
; A real generic function also needs a zero-trip guard, a scalar/masked tail,
; and any required alias/alignment checks. This is not a complete function.
loop.vec:
  %i = phi i64 [ 0, %entry ], [ %i.next, %loop.vec ]
  %a_ptr = getelementptr float, ptr %A, i64 %i
  %b_ptr = getelementptr float, ptr %B, i64 %i
  %va = load <8 x float>, ptr %a_ptr, align 4
  %vb = load <8 x float>, ptr %b_ptr, align 4
  %vmul = fmul <8 x float> %va, %vb
  %c_ptr = getelementptr float, ptr %C, i64 %i
  store <8 x float> %vmul, ptr %c_ptr, align 4
  %i.next = add i64 %i, 8
  %cmp = icmp slt i64 %i.next, %n
  br i1 %cmp, label %loop.vec, label %exit
```

The vectorizer must establish that its chosen transformation preserves dependencies. It may use alias analysis, runtime checks, recognized reductions, or other legality rules; complete non-aliasing is not required for every loop. This is why `restrict` pointers and alignment annotations matter so much for AI kernel performance.

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

Unrolling can expose independent work, but also increases code size and live registers. Measure the effect; excessive unrolling can cause spills or lower GPU occupancy.

**3. Inlining**

Replaces a function call with the function body. Inlining removes the call boundary and can expose further optimization. The arithmetic itself still costs work, and larger code can hurt instruction-cache behavior.

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
  ; Exact in-place C == A can be legal for an elementwise loop.
  ; Shifted overlap, such as C == A + 1, can create loop-carried dependence.
  ; Analyze actual access ranges and the transformation, not only pointer equality.
}
```

**Why this matters for AI compilers:** views and in-place operations can overlap. Attach `noalias` attributes only when the compiler or caller contract proves the required independence. Otherwise use a supported aliasing path or runtime checks. A false `noalias` promise can silently miscompile a program.

```llvm
; TVM-generated IR includes noalias:
define void @fused_relu(ptr noalias %input, ptr noalias %output, i64 %n) {
  ; Compiler can freely vectorize — buffers guaranteed not to overlap
}
```

**Worked example:** in `C[i] = A[i] + B[i]`, exact `C == A` is elementwise in-place work. With `C == A + 1`, iteration zero writes the A element that iteration one would read. A scalar forward loop and an independently evaluated vector batch can therefore produce different results. Describe which alias patterns your public function permits before selecting an optimization.

---

## The Backend: From LLVM IR to Machine Code

The backend (code generator) transforms LLVM IR into **native instructions** for a specific target. A custom-chip compiler also needs high-level partitioning, memory planning, runtime support, and numerical validation; the backend is one part of that system.

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

Instruction selection maps legal IR operations to target instructions. LLVM targets can use SelectionDAG or GlobalISel, with generated TableGen patterns and target-specific code.

**TableGen** describes registers, instructions, and patterns. The following fragments omit required target scaffolding, encoding fields, and register definitions; they illustrate design choices, not a buildable backend:

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

// Result latency of 2 cycles; resource occupancy is a separate scheduling fact
def : WriteRes<WriteFMul, [MulUnit]> { let Latency = 2; }
```

The scheduler uses this model to interleave loads with computes — while one MAC is executing, the next iteration's load can be in-flight. This is the compiler equivalent of **software pipelining**, essential for keeping accelerator datapaths busy.

### Stage 3: Register Allocation

Maps unlimited virtual registers to the finite physical register file. Register-file capacity is target-specific. Excess live values may spill to memory, add instructions, or reduce the number of resident GPU workgroups. Inspect the generated code and measurements rather than assigning a fixed performance penalty.

LLVM provides several register allocators; selection and behavior depend on the target and pipeline. The default greedy allocator uses heuristics including live-range splitting and eviction; it is not simply a synonym for textbook graph coloring. Faster and more specialized allocators have different tradeoffs.

Occupancy is constrained by registers **and** local/shared memory, thread limits, workgroup limits, and allocation granularity. A tile that fits in registers can still be bandwidth-bound. Compare representative kernels when choosing register-file capacity, and include the area and power cost of that capacity.

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

  // i64 multiply: expand using a legal sequence or helper call; not necessarily two multiplies
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

1. **Observe vectorization:** write an elementwise ReLU loop in C and compile with `clang -S -emit-llvm -O2 -Rpass=loop-vectorize -Rpass-missed=loop-vectorize`. Preserve the diagnostic and inspect the IR. For assembly comparison use an explicit target, such as `--target=x86_64-linux-gnu -march=skylake` or `--target=aarch64-linux-gnu -march=armv8-a+simd`; changing `-march` alone does not switch the target architecture. Cross-compilation may need an appropriate sysroot. Do not execute a binary requiring unsupported host instructions.

2. **Investigate aliasing:** compare a function whose contract permits overlapping buffers with a `restrict` version whose callers provide disjoint buffers. Inspect vectorization reports and any runtime range checks. The optimizer may vectorize both. Do not add `noalias` to an overlapping call merely to make a diagnostic disappear. Explain exact in-place versus shifted overlap using a four-element example.

3. **Explore the NVPTX backend:** Write a simple LLVM IR function that loads from address space 1 (global), multiplies by a constant, and stores back. Compile with `llc -march=nvptx64 -mcpu=sm_80`. Read the PTX output — identify `ld.global`, `mul.f32`, and `st.global` instructions.

4. **Backend skeleton:** Using LLVM's documentation, sketch a TableGen file for a hypothetical accelerator with: 16 general-purpose registers (R0–R15), `LOAD`, `STORE`, `ADD`, `MUL`, and `MAC` (fused multiply-add) instructions. Define a pattern for a fused operation and explain why contracting an ordinary `fadd(fmul(a,b),c)` is not always numerically legal.

---

## Key Takeaways

| Concept | Why It Matters for AI Hardware |
|---|---|
| Loop vectorization | Exposes SIMD when legal and profitable; measure benefit |
| Alias analysis | Establishes safe memory transformations; false promises can miscompile |
| Instruction selection | How your custom instructions get used by LLVM |
| TableGen | The declarative language for describing your chip to LLVM |
| Register allocation | Controls placement, spills, and one resource limit on occupancy |
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

## Connect scheduling to a measured artifact

In the [tiling and dataflow lab](../Labs/Tiling-and-Dataflow.md), compare TK=1 and TK=3 while retaining the same output tile. The numerical result and modeled logical traffic stay the same, while storage and array restart costs change. A cost model must represent these competing effects; a transformation name alone does not prove an improvement. Then inspect the real MLIR-generated LLVM IR and distinguish a legal lowering from a demonstrated speedup.

Primary references: [LLVM vectorizers](https://llvm.org/docs/Vectorizers.html), [register allocation infrastructure](https://llvm.org/docs/CodeGenerator.html#register-allocation), and [GlobalISel](https://llvm.org/docs/GlobalISel/).
