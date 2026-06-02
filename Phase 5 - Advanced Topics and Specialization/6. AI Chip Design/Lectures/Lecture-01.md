# Lecture 1: LLVM IR & Architecture

## Overview

Every AI compiler — TVM, Triton, tinygrad, XLA, torch.compile — eventually needs to turn high-level tensor operations into **machine instructions** that run on a specific chip. LLVM is the infrastructure that makes this possible **without writing a full compiler from scratch** for every new hardware target. The core challenge this lecture addresses is: how does LLVM represent programs in a way that is both hardware-independent (so optimizations apply everywhere) and hardware-aware (so the final code exploits specific chip features)? The mental model to carry forward is that LLVM is a **universal translation layer**: frontends (C, Rust, CUDA, ML compilers) lower their programs into LLVM IR, LLVM optimizes the IR, and then a target-specific backend generates native code. For an AI chip designer, understanding LLVM IR means you can write a backend for your custom accelerator and immediately inherit decades of compiler optimizations — without reimplementing constant folding, dead code elimination, or loop unrolling.

---

## Why LLVM Matters for AI Hardware

| Without LLVM | With LLVM |
|---|---|
| Every new chip needs a full compiler from scratch | Write only the backend; inherit the optimization pipeline |
| Optimizations reimplemented per target | ~70 optimization passes shared across all targets |
| ML frameworks must generate assembly directly | Frameworks emit LLVM IR; LLVM handles instruction selection, register allocation, scheduling |
| No ecosystem tooling | Get debuggers (LLDB), profilers, sanitizers, LTO for free |

**Concrete examples in AI:**
- **TVM** uses LLVM to generate CPU kernels (x86 AVX-512, ARM NEON/SVE) and GPU kernels (NVPTX for CUDA, AMDGPU for ROCm)
- **Triton** (used by torch.compile) lowers to LLVM IR → NVPTX backend → PTX → cubin
- **Julia** (used in Flux.jl ML framework) compiles to LLVM IR
- **MLIR** (next lectures) is built on top of LLVM's infrastructure
- **Custom AI accelerator compilers** (Cerebras, SambaNova, Graphcore) use LLVM backends

---

## LLVM Architecture: The Three-Phase Design

```
┌──────────────┐     ┌──────────────────┐     ┌──────────────────┐
│   Frontend    │     │   Middle-End     │     │    Backend       │
│              │     │   (Optimizer)     │     │   (CodeGen)      │
│  C → Clang   │     │                  │     │                  │
│  Rust → rustc│────▶│   LLVM IR        │────▶│  x86-64          │
│  CUDA → nvcc │     │   Optimization   │     │  AArch64         │
│  TVM → codegen│    │   Passes         │     │  NVPTX (GPU)     │
│  Triton      │     │                  │     │  AMDGPU          │
│              │     │                  │     │  RISC-V          │
│              │     │                  │     │  Your custom chip │
└──────────────┘     └──────────────────┘     └──────────────────┘
```

The key insight is **separation of concerns**:
- Frontends only need to emit valid LLVM IR — they don't know about registers, instruction encoding, or pipeline hazards
- The optimizer works on LLVM IR and is target-independent (mostly) — it doesn't know whether the code came from C or a neural network compiler
- Backends only need to translate LLVM IR to machine code — they don't know whether the program is a web server or a matrix multiply kernel

> **Key Insight:** This three-phase design is why LLVM dominates compiler infrastructure. Adding a new source language means writing one frontend. Adding a new hardware target means writing one backend. The N×M problem (N languages × M targets) becomes N+M. For AI hardware, this means your custom chip gets a compiler for free once you write the backend — and every ML framework that emits LLVM IR can target your chip.

---

## LLVM IR: The Language Between Languages

LLVM IR is a **typed, SSA-based, low-level virtual instruction set**. Each of these properties matters:

### Static Single Assignment (SSA)

Every variable is **assigned exactly once**. Instead of mutating variables, you create new ones.

```llvm
; NOT SSA (imperative style — LLVM doesn't allow this):
;   x = 5
;   x = x + 3    ← x is assigned twice

; SSA form (what LLVM requires):
%x1 = add i32 5, 3       ; %x1 = 8, assigned once
%x2 = mul i32 %x1, 2     ; %x2 = 16, assigned once
```

**Why SSA?** It makes optimization trivially correct. If `%x1` is defined exactly once, every use of `%x1` sees the same value — no need to track which assignment reaches which use. This enables **constant propagation, dead code elimination, and common subexpression elimination** to be simple graph transformations.

When control flow merges (e.g., after an if-else), SSA uses **phi nodes** to select which value to use:

```llvm
; if (cond) { a = 1; } else { a = 2; }
; result = a;

entry:
  br i1 %cond, label %then, label %else

then:
  br label %merge

else:
  br label %merge

merge:
  %a = phi i32 [ 1, %then ], [ 2, %else ]
  ; %a is 1 if we came from %then, 2 if from %else
```

### Type System

LLVM IR is strongly typed. Every value has an explicit type. This catches errors early and enables type-based optimizations.

| Type | Syntax | AI Relevance |
|---|---|---|
| Integer | `i1`, `i8`, `i16`, `i32`, `i64` | INT8/INT4 quantized weights, boolean masks |
| Float | `half` (f16), `bfloat` (bf16), `float` (f32), `double` (f64) | Model precision: FP16/BF16 training, FP32 accumulation |
| Vector | `<4 x float>`, `<16 x i8>` | SIMD: AVX-512 processes 16 floats, NEON processes 16 bytes |
| Pointer | `ptr` (opaque since LLVM 15) | Memory addresses for tensor buffers |
| Array | `[1024 x float]` | Fixed-size tensor dimensions |
| Struct | `{ i32, float, ptr }` | Tensor metadata (shape, stride, data pointer) |

### Key Instructions

**Arithmetic:**
```llvm
%sum  = add i32 %a, %b              ; integer add
%prod = fmul float %x, %y           ; floating-point multiply
%mac  = call float @llvm.fmuladd.f32(float %a, float %b, float %c)
                                      ; fused multiply-add: a*b + c
                                      ; maps to FMA instruction on hardware that supports it
```

**Memory:**
```llvm
%ptr  = alloca [1024 x float]        ; allocate 1024 floats on stack
%val  = load float, ptr %ptr          ; load from memory
store float %result, ptr %out_ptr     ; store to memory
%elem = getelementptr [1024 x float], ptr %ptr, i64 0, i64 %idx
                                      ; pointer arithmetic for array indexing
```

**Control flow:**
```llvm
br i1 %cond, label %true_bb, label %false_bb   ; conditional branch
br label %loop_header                            ; unconditional branch
ret float %result                                ; return
```

**Vector operations (critical for AI):**
```llvm
; Load 8 floats from memory as a vector
%vec = load <8 x float>, ptr %tensor_ptr

; Element-wise multiply two 8-float vectors
%mul = fmul <8 x float> %vec_a, %vec_b

; Horizontal reduction (sum all 8 elements)
%sum = call float @llvm.vector.reduce.fadd.v8f32(float 0.0, <8 x float> %mul)

; Shuffle: rearrange elements (used in transpose, gather operations)
%shuf = shufflevector <4 x float> %a, <4 x float> %b, <4 x i32> <i32 0, i32 4, i32 1, i32 5>
```

---

## LLVM IR in Practice: A Dot Product

Consider computing the dot product of two 4-element vectors — the fundamental operation inside every matrix multiply, which is the core of neural network computation.

**C source:**
```c
float dot4(float *a, float *b) {
    float sum = 0.0f;
    for (int i = 0; i < 4; i++)
        sum += a[i] * b[i];
    return sum;
}
```

**LLVM IR (simplified, after optimization):**
```llvm
define float @dot4(ptr %a, ptr %b) {
entry:
  ; Load all 4 elements as vectors
  %va = load <4 x float>, ptr %a, align 16
  %vb = load <4 x float>, ptr %b, align 16

  ; Element-wise multiply
  %vmul = fmul <4 x float> %va, %vb

  ; Horizontal sum (reduce)
  %sum = call float @llvm.vector.reduce.fadd.v4f32(float 0.0, <4 x float> %vmul)

  ret float %sum
}
```

**What the backends produce:**

| Target | Generated instructions |
|---|---|
| x86-64 (AVX) | `vmovaps` → `vmulps` → `vhaddps` → `vhaddps` |
| AArch64 (NEON) | `ld1` → `fmul` → `faddp` → `faddp` |
| NVPTX (CUDA) | `ld.global.v4.f32` → `fma.rn.f32` (×4) |

The **same LLVM IR** produces optimal code for three completely different architectures. This is the power of the **three-phase design**.

---

## LLVM IR Representations: Three Forms

LLVM IR exists in three equivalent forms:

| Form | Extension | Use Case |
|---|---|---|
| **Human-readable text** | `.ll` | Debugging, learning, manual inspection |
| **Bitcode (binary)** | `.bc` | On-disk storage, LTO (Link-Time Optimization) |
| **In-memory C++ objects** | — | Programmatic construction by frontends and passes |

```bash
# Compile C to LLVM IR (text)
clang -S -emit-llvm -O2 dot4.c -o dot4.ll

# Compile to bitcode
clang -c -emit-llvm -O2 dot4.c -o dot4.bc

# Convert between forms
llvm-as dot4.ll -o dot4.bc    # text → bitcode
llvm-dis dot4.bc -o dot4.ll   # bitcode → text

# Compile bitcode to native object
llc dot4.bc -o dot4.o -filetype=obj

# Run optimization passes on bitcode
opt -O2 dot4.bc -o dot4_opt.bc
```

> **Key Insight:** ML compilers like TVM construct LLVM IR programmatically using the C++ API (or the C bindings). They never write `.ll` files — they build `llvm::Module`, `llvm::Function`, `llvm::BasicBlock`, and `llvm::Instruction` objects in memory, then call the backend to emit machine code directly. Understanding the text form is essential for debugging, but the programmatic API is what you'll use in practice.

---

## Module Structure

An LLVM IR module is a complete compilation unit. For an AI kernel, a module typically contains one or a few functions (the kernel entry points) plus metadata.

```llvm
; Module-level declarations
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

; Global constant (e.g., lookup table for activation function)
@relu_lut = internal constant [256 x i8] [ ... ]

; Function declaration (external — will be linked later)
declare void @llvm.memcpy.p0.p0.i64(ptr, ptr, i64, i1)

; Function definition (the actual kernel)
define void @matmul_4x4(ptr %A, ptr %B, ptr %C) #0 {
entry:
  ; ... kernel body ...
  ret void
}

; Function attributes
attributes #0 = { nounwind "target-cpu"="skylake" "target-features"="+avx2,+fma" }

; Metadata (debug info, optimization hints)
!llvm.module.flags = !{!0}
!0 = !{i32 1, !"wchar_size", i32 4}
```

**Key fields for AI compilers:**
- `target datalayout` — endianness, pointer sizes, alignment. Your custom chip defines its own.
- `target triple` — `arch-vendor-os`. For NVIDIA GPUs: `nvptx64-nvidia-cuda`. For your chip: `myaccel-unknown-unknown`.
- `attributes` — tell the optimizer which hardware features are available (AVX-512, FMA, etc.)

---

## Address Spaces

LLVM supports multiple address spaces via the pointer type, which is critical for GPU and accelerator programming where different memory regions have different performance characteristics.

| Address Space | GPU Meaning | AI Relevance |
|---|---|---|
| 0 (default) | Generic | CPU memory, or generic GPU memory |
| 1 | Global | Tensor data in VRAM (weights, activations) |
| 3 | Shared | Tile buffers in on-chip SRAM (shared memory on GPU) |
| 4 | Constant | Read-only data (quantization scale factors, LUTs) |
| 5 | Local (private) | Per-thread registers/stack |

```llvm
; Load from global memory (GPU VRAM)
%val = load float, ptr addrspace(1) %global_ptr

; Store to shared memory (on-chip SRAM)
store float %val, ptr addrspace(3) %shared_ptr
```

For a custom AI accelerator, you define your own address space mapping to describe your memory hierarchy (e.g., address space 1 = weight SRAM, 2 = activation SRAM, 3 = accumulator registers).

---

## Intrinsics: Hardware-Specific Operations

Intrinsics are LLVM's mechanism for exposing **hardware-specific operations** that have no equivalent in generic IR. They look like function calls but compile to **specific instructions**.

```llvm
; Fused multiply-add (maps to FMA instruction)
%fma = call float @llvm.fmuladd.f32(float %a, float %b, float %c)

; Vector reduction (maps to horizontal add on x86, FADDP on ARM)
%sum = call float @llvm.vector.reduce.fadd.v8f32(float 0.0, <8 x float> %v)

; Matrix multiply intrinsic (AMX on Sapphire Rapids)
call void @llvm.x86.tamx.tdpbf16ps(i8 %dst, i8 %src1, i8 %src2)

; NVIDIA-specific: warp shuffle (requires NVPTX backend)
%shfl = call i32 @llvm.nvvm.shfl.sync.bfly.i32(i32 -1, i32 %val, i32 %lane, i32 31)
```

**For custom accelerator design:** You define your own intrinsics. If your chip has a dedicated MAC (multiply-accumulate) unit that operates on 8×8 INT8 tiles, you'd define:
```llvm
declare <64 x i32> @llvm.myaccel.mac.i8.8x8(<64 x i8> %a, <64 x i8> %b, <64 x i32> %acc)
```

The backend pattern-matches this intrinsic to your hardware instruction.

---

## Building LLVM IR Programmatically

This is how ML compilers (TVM, Triton, etc.) actually construct LLVM IR — not by writing text files, but by using the C++ API.

```cpp
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"

// Create context and module
LLVMContext ctx;
Module mod("ai_kernel", ctx);
IRBuilder<> builder(ctx);

// Define function: void relu(float* input, float* output, int n)
Type *floatPtrTy = PointerType::get(Type::getFloatTy(ctx), 0);
Type *i32Ty = Type::getInt32Ty(ctx);
FunctionType *fnTy = FunctionType::get(
    Type::getVoidTy(ctx),
    {floatPtrTy, floatPtrTy, i32Ty},
    false
);
Function *relu = Function::Create(fnTy, Function::ExternalLinkage, "relu", mod);

// Create basic blocks
BasicBlock *entry = BasicBlock::Create(ctx, "entry", relu);
BasicBlock *loop  = BasicBlock::Create(ctx, "loop", relu);
BasicBlock *exit  = BasicBlock::Create(ctx, "exit", relu);

// Entry block: initialize loop counter
builder.SetInsertPoint(entry);
builder.CreateBr(loop);

// Loop block: load, compare, select (ReLU), store
builder.SetInsertPoint(loop);
PHINode *i = builder.CreatePHI(i32Ty, 2, "i");
i->addIncoming(ConstantInt::get(i32Ty, 0), entry);

// ... load input[i], compute max(0, val), store to output[i] ...
// ... increment i, branch back or exit ...
```

> **Key Insight:** TVM's `codegen_llvm.cc` does exactly this — it walks the TVM IR (TIR) and emits LLVM `IRBuilder` calls for each node. Understanding the IRBuilder API is essential if you're building a compiler for a custom AI accelerator. You don't need to be a compiler expert — the API is straightforward once you understand the IR structure.

---

## Hands-On Exercises

1. **Inspect ML compiler output:** Install TVM, compile a simple model (e.g., element-wise ReLU on a 1024-element tensor) for the `llvm` target. Extract the LLVM IR with `mod.get_source("ll")`. Identify the loop structure, vectorization, and any intrinsics used.

2. **Write a dot product in LLVM IR:** Write a `.ll` file that computes the dot product of two 8-element float vectors using vector instructions. Compile with `llc` for both x86-64 and AArch64 targets. Compare the generated assembly.

3. **Explore address spaces:** Write LLVM IR that uses address space 1 (global) and address space 3 (shared). Compile for the NVPTX target with `llc -march=nvptx64`. Examine the PTX output — you should see `ld.global` and `ld.shared` instructions.

4. **Custom intrinsic sketch:** Design the intrinsic signature for a hypothetical accelerator that has a fused Conv2D+ReLU instruction operating on 3×3 INT8 patches. What are the input types, output types, and side effects?

---

## Key Takeaways

| Concept | Why It Matters for AI Hardware |
|---|---|
| SSA form | Enables the optimization passes that make generated code fast |
| Type system | Maps directly to hardware data types (INT8, FP16, BF16) |
| Vector types | The IR representation of SIMD — how AI compilers express parallelism within a thread |
| Address spaces | Model the memory hierarchy of GPUs and custom accelerators |
| Intrinsics | The escape hatch for hardware-specific operations (tensor cores, MAC units) |
| Three-phase design | Write one backend → get all ML frameworks for free |

---

## Resources

* **[LLVM Language Reference Manual](https://llvm.org/docs/LangRef.html):** The authoritative specification of LLVM IR — every type, instruction, and intrinsic.
* **[LLVM Programmer's Manual](https://llvm.org/docs/ProgrammersManual.html):** How to use the C++ API (IRBuilder, PassManager, etc.).
* **"Getting Started with LLVM Core Libraries" by Bruno Cardoso Lopes and Rafael Auler:** Practical book for building tools on LLVM.
* **[Mapping High-Level Constructs to LLVM IR](https://mapping-high-level-constructs-to-llvm-ir.readthedocs.io/):** Community guide for translating programming patterns to IR.
* **TVM source: `src/target/llvm/codegen_llvm.cc`:** Real-world example of an ML compiler generating LLVM IR programmatically.
