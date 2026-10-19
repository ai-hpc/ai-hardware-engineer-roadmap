# Lecture 3: MLIR Fundamentals — Multi-Level IR & Dialects

## Overview

LLVM IR is powerful but it has a fundamental limitation: it operates at **one level of abstraction** — roughly "C with vectors." When an ML compiler needs to reason about tensor operations, loop tiling, data layout, or hardware-specific memory hierarchies, LLVM IR is **too low-level**. You've already lowered away the information the optimizer needs. MLIR (Multi-Level Intermediate Representation) solves this by allowing **multiple levels of abstraction to coexist in the same IR**, connected by progressive lowering. The mental model is a **stack of languages**: at the top, you talk about "matrix multiply two tensors"; in the middle, you talk about "tile this loop nest and map tiles to processing elements"; at the bottom, you emit LLVM IR or hardware instructions. Each level is a **dialect** with its own operations, types, and optimization rules. For AI hardware engineers, MLIR is the framework that connects ML models to custom accelerator backends — Triton, IREE, and several vendor compilers use MLIR, while TVM also maintains its own compiler IRs and scheduling infrastructure. These are distinct designs, not one universal pipeline.

---

## Why LLVM IR Is Not Enough for AI Compilers

| Problem | LLVM IR Limitation | MLIR Solution |
|---|---|---|
| Tensor semantics | No tensor type — only flat arrays and pointers | `tensor<128x64xf32>` as a first-class type |
| Loop tiling decisions | Loops are already lowered to branches and phi nodes | `affine.for` preserves loop structure for polyhedral analysis |
| Hardware mapping | One flat address space model (with numbered spaces) | Dialects define custom memory hierarchy (scratchpad, accumulator, etc.) |
| Multi-level optimization | Must lower everything to one level before optimizing | Each dialect optimizes at its own abstraction level |
| Custom operations | Must use intrinsics (opaque function calls) | Define operations with full semantics, verification, and canonicalization |
| Extensibility | New semantics may use intrinsics, calls, metadata, or compiler extensions | Dialects provide first-class extensible operations, types, and interfaces |

**The key insight:** lowering a matmul to loads, arithmetic, and branches makes its high-level structure harder to recognize. LLVM does have loop transformations, and projects such as Polly analyze affine loops; the issue is preserving useful information and expressing target-specific schedules reliably, not that all loop tiling is impossible below MLIR. MLIR lets a compiler choose when to consume that structure.

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
| `vector` | Low | Fixed-size and scalable vector operations; target-supported lowering required |
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
  %output = tensor.empty() : tensor<1024xf32>
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

// Vector types (these examples are fixed-size; scalable forms also exist)
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

Retaining structure makes relevant transformations easier to express and verify. It does not guarantee that an MLIR-based compiler is faster: legal schedules, target support, cost models, and runtime overhead still determine the result. Recovering a high-level operation from low-level IR may be difficult, so choose lowering boundaries deliberately.

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

Every dialect can register **canonicalization patterns** — simplification rules whose preconditions must establish semantic equivalence:

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

## Run a real progressive lowering

Use the complete [matrix fixture and lab](../Labs/Tiling-and-Dataflow.md), tested with MLIR20.1.2. Most operation examples above are fragments with surrounding values omitted; `matmul.mlir` is a complete module.

```bash
# From this Lectures directory, with a matching MLIR toolchain on PATH:
mlir-opt ../Labs/matmul.mlir --convert-linalg-to-loops -o /tmp/course-loops.mlir
python3 ../Labs/run_mlir_lab.py --output-dir /tmp/course-mlir-stages
```

The helper parses/verifies, lowers to loops, converts the remaining dialects to LLVM, translates to LLVM IR, and actually executes the native CPU result. Expected result: **415**, after checking each matrix element. See the lab for version-suffixed executable names and environment overrides.

Do not assume old tutorial flags such as `--linalg-tile` or `--linalg-tile-and-fuse-tensor-ops` exist in your build. Tiling can be driven through the Transform dialect or compiler APIs. Inspect `mlir-opt --help` and use the [Transform tutorial](https://mlir.llvm.org/docs/Tutorials/transform/Ch0/) matching your version. A successful parse, a successful lowering, and a correct executed result are three different checkpoints.

---

## Hands-On Exercises

1. **Read MLIR output:** run the complete matrix lab. Locate the three nested reduction loops in `02-loops.mlir`, then the branches and pointer operations in `03-llvm-dialect.mlir`. Explain where M, N, and K are represented at each stage.

2. **Catch a real error:** change B's reduction dimension from three to four in a small `linalg.matmul` module while keeping A at 2×3. Run the verifier. Then change an input value without changing shape and run the lab's element checks. **Answer:** the first is a static shape error; the second is valid IR whose result violates the expected numerical test. A verifier cannot prove an arbitrary model computes the desired application behavior.

3. **Tensor vs MemRef:** Write a function that takes `tensor<16x16xf32>` inputs, performs an element-wise add, and returns a tensor. Run `--one-shot-bufferize` and observe where tensors become memrefs and allocations appear. One-Shot Bufferize does not insert deallocations; study the ownership-based buffer deallocation pipeline separately, then check that no live alias outlasts its buffer. [Bufferization reference](https://mlir.llvm.org/docs/Bufferization/).

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

## Tensor initialization is part of the operation

The ReLU example uses `tensor.empty` only as an output shape because its scalar body overwrites every output element without reading `%out`. Its comparison/select rule maps NaN to zero and negative zero to positive zero; document that choice if matching a framework operator. For matmul, the output operand is an accumulator, so an uninitialized `tensor.empty` is insufficient: fill it with zero before computing `C = A @ B`, or supply a defined initial C for accumulation. The executable lab makes this initialization visible with `linalg.fill`.
