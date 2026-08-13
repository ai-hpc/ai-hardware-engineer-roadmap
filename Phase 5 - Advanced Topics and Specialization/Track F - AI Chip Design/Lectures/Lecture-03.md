# Lecture 3: MLIR Fundamentals — Multi-Level IR & Dialects

## Overview

LLVM IR is powerful but it has a fundamental limitation: it operates at **one level of abstraction** — roughly "C with vectors." When an ML compiler needs to reason about tensor operations, loop tiling, data layout, or hardware-specific memory hierarchies, LLVM IR is **too low-level**. You've already lowered away the information the optimizer needs. MLIR (Multi-Level Intermediate Representation) solves this by allowing **multiple levels of abstraction to coexist in the same IR**, connected by progressive lowering. The mental model is a **stack of languages**: at the top, you talk about "matrix multiply two tensors"; in the middle, you talk about "tile this loop nest and map tiles to processing elements"; at the bottom, you emit LLVM IR or hardware instructions. Each level is a **dialect** with its own operations, types, and optimization rules. For AI hardware engineers, MLIR is the framework that connects ML models to custom accelerator backends — it is the compiler infrastructure that TVM, Triton (via TTIR/TTGIR), IREE, and hardware vendor compilers are all converging on.

---

## Why LLVM IR Is Not Enough for AI Compilers

| Problem | LLVM IR Limitation | MLIR Solution |
|---|---|---|
| Tensor semantics | No tensor type — only flat arrays and pointers | `tensor<128x64xf32>` as a first-class type |
| Loop tiling decisions | Loops are already lowered to branches and phi nodes | `affine.for` preserves loop structure for polyhedral analysis |
| Hardware mapping | One flat address space model (with numbered spaces) | Dialects define custom memory hierarchy (scratchpad, accumulator, etc.) |
| Multi-level optimization | Must lower everything to one level before optimizing | Each dialect optimizes at its own abstraction level |
| Custom operations | Must use intrinsics (opaque function calls) | Define operations with full semantics, verification, and canonicalization |
| Extensibility | Adding a new concept requires modifying LLVM core | Dialects are modular — add new ones without touching existing code |

**The key insight:** When you lower `matmul(A, B)` to LLVM IR loops, you lose the information that this is a matrix multiply. The LLVM vectorizer can vectorize the innermost loop, but it cannot **tile the loop nest** for cache locality or **map it to a systolic array**. MLIR keeps the high-level semantics alive long enough for **hardware-aware optimizations** to act on them.

---

## MLIR Architecture

```
┌──────────────────────────────────────────────────────────────────┐
│                        ML Framework                              │
│                   (PyTorch, TensorFlow, tinygrad)                │
└──────────────────────────┬───────────────────────────────────────┘
                           │
                           ▼
┌──────────────────────────────────────────────────────────────────┐
│                     High-Level Dialects                          │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌────────────────┐  │
│  │  tosa     │  │ stablehlo│  │  torch   │  │  custom.myop   │  │
│  │(Tensor Op │  │(StableHLO│  │ (PyTorch │  │  (your own     │  │
│  │ Set Arch.)│  │ from XLA)│  │  ops)    │  │   dialect)     │  │
│  └────┬─────┘  └────┬─────┘  └────┬─────┘  └──────┬─────────┘  │
│       │              │             │               │             │
│       └──────────────┼─────────────┼───────────────┘             │
│                      ▼                                           │
│  ┌─────────────────────────────────────────────────────────────┐ │
│  │              Mid-Level Dialects                             │ │
│  │  ┌────────┐  ┌────────┐  ┌─────────┐  ┌─────────────────┐ │ │
│  │  │ linalg │  │ tensor │  │  arith  │  │    memref       │ │ │
│  │  │(Linear │  │(Tensor │  │(Arith-  │  │(Memory-backed   │ │ │
│  │  │Algebra)│  │ ops)   │  │ metic)  │  │ tensors)        │ │ │
│  │  └───┬────┘  └───┬────┘  └────┬────┘  └──────┬──────────┘ │ │
│  └──────┼───────────┼────────────┼───────────────┼────────────┘ │
│         │           │            │               │              │
│         └───────────┼────────────┼───────────────┘              │
│                     ▼                                           │
│  ┌─────────────────────────────────────────────────────────────┐ │
│  │              Low-Level Dialects                             │ │
│  │  ┌────────┐  ┌────────┐  ┌──────────┐  ┌───────────────┐  │ │
│  │  │ affine │  │  scf   │  │   gpu    │  │   vector      │  │ │
│  │  │(Affine │  │(Struct.│  │(GPU      │  │(Hardware      │  │ │
│  │  │ loops) │  │Control │  │ mapping) │  │ vectors)      │  │ │
│  │  │        │  │ Flow)  │  │          │  │               │  │ │
│  │  └───┬────┘  └───┬────┘  └────┬─────┘  └──────┬────────┘  │ │
│  └──────┼───────────┼────────────┼───────────────┼────────────┘ │
│         └───────────┼────────────┼───────────────┘              │
│                     ▼                                           │
│  ┌─────────────────────────────────────────────────────────────┐ │
│  │              Target Dialects                                │ │
│  │  ┌────────┐  ┌────────────┐  ┌───────────────────────────┐ │ │
│  │  │ llvm   │  │  nvvm /    │  │   your_accel              │ │ │
│  │  │(LLVM IR│  │  rocdl /   │  │   (custom accelerator     │ │ │
│  │  │ in MLIR│  │  spirv     │  │    instructions)          │ │ │
│  │  │ form)  │  │            │  │                           │ │ │
│  │  └───┬────┘  └────┬───────┘  └──────┬────────────────────┘ │ │
│  └──────┼────────────┼─────────────────┼──────────────────────┘ │
└─────────┼────────────┼─────────────────┼────────────────────────┘
          ▼            ▼                 ▼
    LLVM Backend    PTX / AMDGPU     Custom Assembly
```

---

## Core Concepts

### 1. Operations

An **operation** is the fundamental unit of computation in MLIR. Every node in the IR is an operation. Operations are **fully extensible** — any dialect can define new ones.

```mlir
// An operation has:
//   - a name (dialect.operation_name)
//   - operands (SSA values)
//   - results (SSA values)
//   - attributes (compile-time constants)
//   - regions (nested IR, for control flow)
//   - types

%result = arith.addf %a, %b : f32
//         ^^^^^^^^^^^         ^^^
//         operation name      type
//                    ^^  ^^
//                    operands

%out = linalg.matmul ins(%A, %B : tensor<128x64xf32>, tensor<64x256xf32>)
                     outs(%C : tensor<128x256xf32>) -> tensor<128x256xf32>
//     ^^^^^^^^^^^^^^
//     high-level operation: "matrix multiply"
//     carries full semantic information
```

### 2. Dialects

A **dialect** is a namespace of operations, types, and attributes. Think of it as a mini-language for a specific abstraction level or domain.

| Dialect | Abstraction Level | Purpose |
|---|---|---|
| `tosa` | Highest | Standard tensor operations (conv2d, matmul, relu) — hardware-agnostic |
| `stablehlo` | Highest | StableHLO ops from XLA/JAX ecosystem |
| `linalg` | High-mid | Named and generic linear algebra on tensors (matmul, conv, pooling) |
| `tensor` | Mid | Tensor manipulation (extract_slice, insert_slice, reshape) |
| `memref` | Mid | Memory-backed tensors with explicit buffers and layouts |
| `affine` | Mid-low | Affine loop nests for polyhedral optimization |
| `scf` | Low | Structured control flow (for, while, if) |
| `vector` | Low | Fixed-size vector operations (maps to SIMD/vector units) |
| `gpu` | Low | GPU-specific: thread/block indexing, barriers, shared memory |
| `arith` | Low | Arithmetic operations (add, mul, cmp) on scalars and vectors |
| `math` | Low | Math operations (exp, log, sqrt, tanh) |
| `llvm` | Lowest | LLVM IR operations expressed in MLIR syntax |
| `nvvm` | Target | NVIDIA-specific: warp shuffles, tensor cores, TMA |

### 3. Regions and Blocks

Operations can contain **regions**, which contain **blocks** of operations. This is how MLIR represents nested structure — a loop body, a function body, or a GPU kernel body.

```mlir
// A function is an operation with a region containing blocks
func.func @relu(%input: tensor<1024xf32>) -> tensor<1024xf32> {
  // This is a block inside the function's region
  %zero = arith.constant 0.0 : f32
  %result = linalg.generic {
    indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>],
    iterator_types = ["parallel"]
  } ins(%input : tensor<1024xf32>)
    outs(%output : tensor<1024xf32>) {
    // This is a nested region inside linalg.generic
    ^bb0(%in: f32, %out: f32):
      %cmp = arith.cmpf ogt, %in, %zero : f32
      %relu = arith.select %cmp, %in, %zero : f32
      linalg.yield %relu : f32
  } -> tensor<1024xf32>
  return %result : tensor<1024xf32>
}
```

> **Key Insight:** The nested structure of regions is what enables MLIR to represent multi-level abstractions in a single IR. A `linalg.matmul` operation encapsulates the computation; its region defines the element-wise body. A pass can choose to tile the matmul, map tiles to GPU threads, or lower the whole thing to a hardware instruction — all by transforming the IR at the appropriate level.

### 4. Types

MLIR's type system is extensible. Each dialect can define its own types.

```mlir
// Built-in types
f16, bf16, f32, f64                           // floating point
i1, i8, i16, i32, i64                         // integer
index                                          // loop indices, sizes

// Tensor types (value semantics — no memory associated)
tensor<128x64xf32>                            // static shape
tensor<?x64xf32>                              // dynamic first dimension
tensor<*xf32>                                 // unranked (any shape)

// MemRef types (reference semantics — backed by memory)
memref<128x64xf32>                            // default layout (row-major)
memref<128x64xf32, affine_map<(d0,d1) -> (d1, d0)>>  // column-major
memref<128x64xf32, #gpu.address_space<workgroup>>     // GPU shared memory

// Vector types (fixed-size, for SIMD)
vector<8xf32>                                 // 8-element vector
vector<4x4xf32>                               // 2D vector (for matrix tiles)
```

**Tensor vs. MemRef:** This distinction is fundamental.
- `tensor` has value semantics — like a mathematical matrix. No side effects. Enables functional-style optimizations (fusion, CSE).
- `memref` has reference semantics — it points to actual memory. Has side effects (loads/stores). Required for code generation.
- **Bufferization** is the pass that converts `tensor` → `memref`, deciding where to allocate buffers and when to reuse them.

---

## Progressive Lowering

The core principle of MLIR: **don't lower everything at once**. Lower one level at a time, **optimizing at each level**.

```
tosa.conv2d                           ← "convolution on tensors"
       │
       ▼  (tosa-to-linalg)
linalg.conv_2d_nhwc_hwcf              ← "conv as loop nest over tensors"
       │
       ▼  (linalg tiling)
linalg.conv (tiled to 4x4)            ← "tiled conv with explicit tile sizes"
       │
       ▼  (linalg-to-loops)
scf.for / affine.for                  ← "explicit loop nest"
       │
       ▼  (loop vectorization)
vector.contract / vector.fma          ← "vector operations on tiles"
       │
       ▼  (bufferization)
memref.load / memref.store            ← "explicit memory operations"
       │
       ▼  (convert-to-llvm)
llvm.load / llvm.store / llvm.call    ← "LLVM IR operations"
       │
       ▼  (mlir-translate)
LLVM IR                               ← "standard LLVM IR"
       │
       ▼  (llc)
Native code                           ← "machine instructions"
```

Each arrow is a **lowering pass** that converts operations from one dialect to another. At each level, optimization passes specific to that dialect can run:

| Level | Optimizations Available |
|---|---|
| `linalg` | Tiling, fusion of adjacent ops, interchange (loop reordering) |
| `affine` | Polyhedral optimization, dependence analysis, loop skewing |
| `scf` | Loop unrolling, pipelining, peeling |
| `vector` | Vector distribution, transfer read/write optimization |
| `gpu` | Thread/block mapping, shared memory promotion |

> **Key Insight:** The reason MLIR outperforms "lower everything to LLVM IR and optimize there" is that each level retains information that lower levels lose. At the `linalg` level, you know it's a matrix multiply — you can tile it for a systolic array. At the `affine` level, you know the loop bounds are affine functions of outer indices — you can apply polyhedral optimization. Once it's in LLVM IR, it's just loops and loads — the optimizer can vectorize and unroll but cannot re-tile or re-map to a spatial architecture.

---

## Passes and Transformations

MLIR passes work similarly to LLVM passes but operate on MLIR operations.

### Pass Types

```cpp
// Operation pass: runs on a specific operation type (e.g., FuncOp)
struct MyTilingPass : public PassWrapper<MyTilingPass, OperationPass<func::FuncOp>> {
  void runOnOperation() override {
    func::FuncOp func = getOperation();
    // Walk all linalg operations and tile them
    func.walk([](linalg::MatmulOp op) {
      // Tile with tile sizes [32, 32, 16]
      linalg::tileUsingForOp(op, {32, 32, 16});
    });
  }
};

// Module pass: runs on the entire module
struct MyBufferizationPass : public PassWrapper<MyBufferizationPass, OperationPass<ModuleOp>> {
  // ...
};
```

### Canonicalization

Every dialect can register **canonicalization patterns** — simplification rules that are always correct to apply:

```mlir
// Before canonicalization:
%x = arith.addf %a, %zero : f32     // adding zero

// After canonicalization:
// %x is replaced with %a (the add is eliminated)
```

```mlir
// Before: redundant tensor.cast
%t1 = tensor.cast %input : tensor<128xf32> to tensor<?xf32>
%t2 = tensor.cast %t1 : tensor<?xf32> to tensor<128xf32>
// After: both casts eliminated, %t2 → %input
```

### Dialect Conversion Framework

When lowering from one dialect to another, MLIR provides a structured framework:

```cpp
// Define which operations to convert
struct MatmulToLoopsPattern : public OpConversionPattern<linalg::MatmulOp> {
  LogicalResult matchAndRewrite(
      linalg::MatmulOp op,
      OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    // Replace linalg.matmul with nested scf.for loops
    auto loc = op.getLoc();
    auto zero = rewriter.create<arith.ConstantIndexOp>(loc, 0);
    // ... build loop nest ...
    rewriter.replaceOp(op, result);
    return success();
  }
};
```

---

## Defining a Custom Dialect

For a custom AI accelerator, you define your own dialect with operations that map directly to your **hardware instructions**.

```tablegen
// MyAccel.td — TableGen dialect definition

def MyAccel_Dialect : Dialect {
  let name = "myaccel";
  let summary = "Dialect for MyAccel AI accelerator";
  let cppNamespace = "::mlir::myaccel";
}

// Define a matrix multiply operation for the accelerator
def MyAccel_MatMulOp : Op<MyAccel_Dialect, "matmul", [Pure]> {
  let summary = "Accelerator matrix multiply (8x8 INT8 tile)";
  let arguments = (ins
    MemRefOf<[I8]>:$lhs,       // left input tile in accelerator SRAM
    MemRefOf<[I8]>:$rhs,       // right input tile in accelerator SRAM
    MemRefOf<[I32]>:$acc       // accumulator in register file
  );
  let results = (outs MemRefOf<[I32]>:$result);

  let assemblyFormat = [{
    `(` $lhs `,` $rhs `,` $acc `)` attr-dict `:` type($result)
  }];
}

// Define a DMA transfer operation
def MyAccel_DMAOp : Op<MyAccel_Dialect, "dma_transfer", []> {
  let summary = "Transfer data between main memory and accelerator SRAM";
  let arguments = (ins
    MemRefOf<[AnyType]>:$src,
    MemRefOf<[AnyType]>:$dst,
    Index:$size
  );
}
```

The lowering pipeline for your accelerator would be:
```
linalg.matmul → tiling to 8×8 tiles → myaccel.dma_transfer + myaccel.matmul
```

---

## MLIR Tools

```bash
# Parse and verify MLIR
mlir-opt input.mlir

# Run specific passes
mlir-opt --linalg-tile="tile-sizes=32,32,16" input.mlir
mlir-opt --convert-linalg-to-loops input.mlir
mlir-opt --convert-scf-to-cf --convert-to-llvm input.mlir

# Full lowering pipeline
mlir-opt input.mlir \
  --linalg-tile="tile-sizes=32,32,16" \
  --convert-linalg-to-loops \
  --lower-affine \
  --convert-scf-to-cf \
  --convert-to-llvm \
  -o lowered.mlir

# Translate to LLVM IR
mlir-translate --mlir-to-llvmir lowered.mlir -o output.ll

# Then compile with LLVM
llc output.ll -o output.o -filetype=obj
```

---

## Hands-On Exercises

1. **Read MLIR output:** Install MLIR (comes with LLVM build). Write a simple `linalg.matmul` in MLIR text format. Run `mlir-opt --convert-linalg-to-loops` and observe the generated `scf.for` loop nest. Then run `--convert-scf-to-cf --convert-to-llvm` and observe the LLVM dialect output.

2. **Progressive lowering:** Start with a `tosa.conv2d` operation. Lower it through `tosa-to-linalg` → `linalg-tile` → `linalg-to-loops` → `convert-to-llvm`. At each stage, print the IR and observe how information is preserved then consumed.

3. **Tensor vs MemRef:** Write a function that takes `tensor<16x16xf32>` inputs, performs an element-wise add, and returns a tensor. Run `--one-shot-bufferize` and observe how tensors become memrefs with explicit `memref.alloc` and `memref.dealloc`.

4. **Dialect design exercise:** Design (on paper) an MLIR dialect for a hypothetical NPU with: a 16×16 INT8 MAC array, 64KB weight SRAM, 32KB activation SRAM, and DMA for host↔SRAM transfers. Define the operations, types, and memory spaces. Sketch the lowering from `linalg.matmul` to your dialect.

---

## Key Takeaways

| Concept | Why It Matters for AI Hardware |
|---|---|
| Multi-level IR | Preserve high-level semantics for hardware-aware optimization |
| Dialects | Modular, extensible — add your accelerator's ops without forking MLIR |
| Progressive lowering | Optimize at each level; don't prematurely discard information |
| Tensor vs MemRef | Separate algorithm (tensor) from memory management (memref) |
| Regions | Enable nested structure: kernels, loop bodies, pipeline stages |
| Custom dialects | The mechanism for connecting your hardware to ML frameworks |

---

## Resources

* **[MLIR Language Reference](https://mlir.llvm.org/docs/LangRef/):** The authoritative specification of MLIR syntax and semantics.
* **[MLIR Dialects Documentation](https://mlir.llvm.org/docs/Dialects/):** Reference for all built-in dialects (linalg, affine, scf, gpu, vector, etc.).
* **"MLIR: Scaling Compiler Infrastructure for Domain-Specific Computation" (CGO 2021):** The foundational paper by Lattner et al.
* **[MLIR Tutorial](https://mlir.llvm.org/docs/Tutorials/Toy/):** The official Toy language tutorial — builds a full compiler using MLIR from scratch.
* **[MLIR Open Design Meetings (YouTube)](https://www.youtube.com/channel/UCMQl4dniSlBiEFPXl9n5ueg):** Recordings of MLIR design discussions covering real-world use cases.
