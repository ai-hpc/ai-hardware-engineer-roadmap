# Lecture 4: MLIR for ML Compilers — Linalg, Tensor, Affine & Vector Dialects

## Overview

Lecture 3 introduced MLIR's architecture: dialects, operations, progressive lowering. This lecture dives into the **specific dialects that matter for AI compilation** — the ones that sit between "this is a matrix multiply" and "emit LLVM IR." These dialects form the **compilation backbone** of projects like IREE (Google's ML compiler), Triton-MLIR, and hardware vendor compilers for NPUs and AI accelerators. The core challenge is: how do you take a high-level tensor operation (like conv2d or attention) and systematically transform it into tiled, vectorized, hardware-mapped code — while making optimal decisions about data layout, memory placement, and parallelism? Each dialect in this lecture provides the vocabulary and transformation rules for one stage of that journey.

---

## The Dialects and Their Roles

```
                    ML Model Graph
                         │
        ┌────────────────┼────────────────┐
        ▼                ▼                ▼
   ┌─────────┐    ┌───────────┐    ┌──────────┐
   │  TOSA    │    │ StableHLO │    │  Torch   │    Entry dialects
   │          │    │           │    │  MLIR    │    (from frameworks)
   └────┬─────┘    └─────┬─────┘    └────┬─────┘
        │                │               │
        └────────────────┼───────────────┘
                         ▼
              ┌─────────────────────┐
              │      linalg         │    ← THIS LECTURE
              │  (structured ops)   │      Tiling, fusion, interchange
              └──────────┬──────────┘
                         │
              ┌──────────┼──────────┐
              ▼          ▼          ▼
        ┌──────────┐ ┌────────┐ ┌────────┐
        │  tensor  │ │ memref │ │ arith  │  ← THIS LECTURE
        │          │ │        │ │  math  │    Bufferization, computation
        └────┬─────┘ └───┬────┘ └───┬────┘
             │           │          │
             └───────────┼──────────┘
                         ▼
              ┌─────────────────────┐
              │  affine / scf       │    ← THIS LECTURE
              │  (loop nests)       │      Loop optimization, tiling
              └──────────┬──────────┘
                         │
                         ▼
              ┌─────────────────────┐
              │      vector         │    ← THIS LECTURE
              │  (SIMD / tiles)     │      Hardware vector mapping
              └──────────┬──────────┘
                         │
                    ┌────┼────┐
                    ▼    ▼    ▼
                  llvm  nvvm  spirv     Target dialects
```

---

## 1. The Linalg Dialect: Structured Linear Algebra

`linalg` is the most important dialect for AI compilation. It represents tensor computations as **structured operations** — operations whose access patterns are fully described by **affine maps**, enabling systematic tiling, fusion, and hardware mapping.

### Named Operations

Linalg provides named ops for common AI workloads:

```mlir
// Matrix multiply: C = A × B
%C = linalg.matmul ins(%A, %B : tensor<128x64xf32>, tensor<64x256xf32>)
                   outs(%C_init : tensor<128x256xf32>) -> tensor<128x256xf32>

// Batch matrix multiply (attention: Q @ K^T)
%out = linalg.batch_matmul ins(%Q, %K : tensor<8x128x64xf32>, tensor<8x64x128xf32>)
                           outs(%init : tensor<8x128x128xf32>) -> tensor<8x128x128xf32>

// 2D Convolution (NHWC format)
%out = linalg.conv_2d_nhwc_hwcf
    ins(%input, %filter : tensor<1x28x28x3xf32>, tensor<3x3x3x16xf32>)
    outs(%output : tensor<1x26x26x16xf32>) -> tensor<1x26x26x16xf32>

// Pooling
%out = linalg.pooling_nhwc_max
    ins(%input, %window : tensor<1x28x28x16xf32>, tensor<2x2xf32>)
    outs(%output : tensor<1x27x27x16xf32>) -> tensor<1x27x27x16xf32>

// Element-wise operations via linalg.generic (see below)
```

### linalg.generic: The Universal Operation

Any operation expressible as a **loop nest with affine access patterns** can be written as `linalg.generic`. This includes matmul, conv, attention, normalization — everything.

```mlir
// ReLU: output[i] = max(input[i], 0)
#map = affine_map<(d0) -> (d0)>

%result = linalg.generic {
    indexing_maps = [#map, #map],        // both input and output indexed by d0
    iterator_types = ["parallel"]         // d0 is parallelizable
  } ins(%input : tensor<1024xf32>)
    outs(%output : tensor<1024xf32>) {
  ^bb0(%in: f32, %out: f32):
    %zero = arith.constant 0.0 : f32
    %relu = arith.maximumf %in, %zero : f32
    linalg.yield %relu : f32
  } -> tensor<1024xf32>
```

```mlir
// Matrix multiply as linalg.generic
// C[i,j] += A[i,k] * B[k,j]
#mapA = affine_map<(i, j, k) -> (i, k)>
#mapB = affine_map<(i, j, k) -> (k, j)>
#mapC = affine_map<(i, j, k) -> (i, j)>

%C = linalg.generic {
    indexing_maps = [#mapA, #mapB, #mapC],
    iterator_types = ["parallel", "parallel", "reduction"]
    //                 i: parallel  j: parallel  k: reduction (summed over)
  } ins(%A, %B : tensor<128x64xf32>, tensor<64x256xf32>)
    outs(%C_init : tensor<128x256xf32>) {
  ^bb0(%a: f32, %b: f32, %c: f32):
    %mul = arith.mulf %a, %b : f32
    %add = arith.addf %c, %mul : f32
    linalg.yield %add : f32
  } -> tensor<128x256xf32>
```

**Why this representation matters:**
- `indexing_maps` describe exactly which elements each iteration accesses → enables dependence analysis
- `iterator_types` declare which dimensions are parallel vs. reduction → guides parallelization
- The body is a simple scalar computation → decoupled from tiling/scheduling decisions

> **Key Insight:** The power of `linalg.generic` is that tiling, fusion, vectorization, and hardware mapping are all **transformations on the indexing maps and iterator types** — not on the body. A tiling pass splits a dimension by changing the loop bounds and adjusting index expressions; the body remains identical. This separation of concerns is what makes MLIR-based compilers systematically composable, unlike hand-written kernels where tiling decisions are baked into the code.

### Linalg Transformations

These are the core transformations that convert high-level linalg ops into hardware-optimized code:

**Tiling:**
```mlir
// BEFORE: full 128x256 matmul
%C = linalg.matmul ins(%A, %B) outs(%C_init)

// AFTER tiling with tile sizes [32, 64, 16]:
// Outer loops iterate over tiles; inner op is a 32x64 matmul with k-reduction of 16
scf.for %i = 0 to 128 step 32 {
  scf.for %j = 0 to 256 step 64 {
    scf.for %k = 0 to 64 step 16 {
      %a_tile = tensor.extract_slice %A[%i, %k] [32, 16] [1, 1]
      %b_tile = tensor.extract_slice %B[%k, %j] [16, 64] [1, 1]
      %c_tile = tensor.extract_slice %C[%i, %j] [32, 64] [1, 1]
      %result = linalg.matmul ins(%a_tile, %b_tile) outs(%c_tile)
      // ... insert_slice result back ...
    }
  }
}
```

**Fusion:**
```mlir
// BEFORE: matmul followed by ReLU — two separate kernels, intermediate buffer
%mm = linalg.matmul ins(%A, %B) outs(%C_init)
%relu = linalg.generic {/*relu*/} ins(%mm) outs(%out_init)

// AFTER fusion: ReLU is fused into the matmul's output tile loop
// No intermediate buffer needed — the relu is applied to each tile as it's computed
scf.for %i = ... {
  scf.for %j = ... {
    %mm_tile = linalg.matmul ins(%a_tile, %b_tile) outs(%c_tile)
    %relu_tile = linalg.generic {/*relu*/} ins(%mm_tile) outs(%out_tile)
  }
}
```

**Interchange (loop reordering):**
```mlir
// Change iteration order from (i, j, k) to (j, i, k)
// This changes the memory access pattern — critical for cache behavior
// e.g., making the innermost loop access contiguous memory
```

**Promotion (buffer to faster memory):**
```mlir
// Promote a tile's memory from main memory to shared memory / scratchpad
// BEFORE: tiles read from global memory each iteration
// AFTER: tiles loaded into memref in a faster address space
%promoted = memref.alloc() : memref<32x16xf32, #gpu.address_space<workgroup>>
// ... copy tile to shared memory ...
// ... compute on shared memory tile ...
```

---

## 2. The Tensor and MemRef Dialects: Value vs. Reference

### Tensor Dialect

Operates on **immutable tensor values**. No memory allocation, no side effects. Enables functional optimization.

```mlir
// Extract a subtensor (like NumPy slicing)
%slice = tensor.extract_slice %input[0, 0] [32, 64] [1, 1]
    : tensor<128x256xf32> to tensor<32x64xf32>

// Insert a subtensor back (functional update — creates a new tensor)
%updated = tensor.insert_slice %computed_tile into %output[%i, %j] [32, 64] [1, 1]
    : tensor<32x64xf32> into tensor<128x256xf32>

// Reshape
%reshaped = tensor.collapse_shape %input [[0, 1], [2]]
    : tensor<8x16x64xf32> into tensor<128x64xf32>

%expanded = tensor.expand_shape %flat [[0, 1]]
    : tensor<1024xf32> into tensor<32x32xf32>

// Pad (for convolution boundary handling)
%padded = tensor.pad %input low[0, 1] high[0, 1] {
  ^bb0(%arg0: index, %arg1: index):
    tensor.yield %zero : f32
} : tensor<28x28xf32> to tensor<28x30xf32>
```

### MemRef Dialect

**Memory-backed references.** Explicit allocation, loads, stores. Required for actual code generation.

```mlir
// Allocate memory
%buf = memref.alloc() : memref<128x256xf32>

// Load and store
%val = memref.load %buf[%i, %j] : memref<128x256xf32>
memref.store %result, %buf[%i, %j] : memref<128x256xf32>

// Subview (like extract_slice but for memrefs — zero-copy, just pointer arithmetic)
%tile = memref.subview %buf[%i, %j] [32, 64] [1, 1]
    : memref<128x256xf32> to memref<32x64xf32, strided<[256, 1], offset: ?>>

// Cast to different memory space
%shared = memref.alloc() : memref<32x64xf32, #gpu.address_space<workgroup>>

// Dealloc
memref.dealloc %buf : memref<128x256xf32>
```

### Bufferization: Tensor → MemRef

The **one-shot bufferization** pass converts the entire program from tensor semantics to memref semantics in a single analysis, deciding buffer allocation and reuse.

```mlir
// BEFORE (tensor world — functional, no memory):
%0 = linalg.matmul ins(%A, %B : tensor<M×K>, tensor<K×N>)
                   outs(%C : tensor<M×N>) -> tensor<M×N>
%1 = linalg.generic {relu} ins(%0 : tensor<M×N>)
                           outs(%out : tensor<M×N>) -> tensor<M×N>

// AFTER one-shot bufferization (memref world — imperative, explicit memory):
%buf_A = bufferization.to_memref %A : memref<M×K×f32>
%buf_B = bufferization.to_memref %B : memref<K×N×f32>
%buf_C = memref.alloc() : memref<M×N×f32>          // buffer for matmul output
linalg.matmul ins(%buf_A, %buf_B) outs(%buf_C)     // writes into %buf_C
linalg.generic {relu} ins(%buf_C) outs(%buf_C)     // IN-PLACE: reuses %buf_C
// The bufferizer determined that %0 is only consumed by the relu, so it can
// reuse the same buffer — no extra allocation needed
```

> **Key Insight:** Bufferization is where the compiler makes critical memory allocation decisions. For AI workloads, this determines whether intermediate tensors (between matmul and activation, between attention layers) are allocated as separate buffers or share memory. Good bufferization can halve memory usage — bad bufferization can OOM on edge devices. This is why keeping tensor semantics as long as possible (and bufferizing late) produces better code.

---

## 3. The Affine Dialect: Polyhedral Optimization

The `affine` dialect represents loop nests where bounds and access patterns are **affine expressions** (linear combinations of loop variables and constants). This enables powerful **polyhedral analysis** — the compiler can reason mathematically about **data dependencies**, loop transformations, and optimal tiling.

```mlir
// Simple affine loop nest for matrix multiply
affine.for %i = 0 to 128 {
  affine.for %j = 0 to 256 {
    affine.for %k = 0 to 64 {
      %a = affine.load %A[%i, %k] : memref<128x64xf32>
      %b = affine.load %B[%k, %j] : memref<64x256xf32>
      %c = affine.load %C[%i, %j] : memref<128x256xf32>
      %prod = arith.mulf %a, %b : f32
      %sum = arith.addf %c, %prod : f32
      affine.store %sum, %C[%i, %j] : memref<128x256xf32>
    }
  }
}
```

### Affine Maps

Affine maps describe the relationship between loop indices and memory access positions:

```mlir
// Row-major access: A[i][k]
#row_major = affine_map<(i, k) -> (i, k)>

// Column-major access: B[k][j] stored as B_col[j][k]
#col_major = affine_map<(k, j) -> (j, k)>

// Strided access: every other element
#strided = affine_map<(i) -> (2 * i)>

// Tiled access: element within a 32x32 tile
#tiled = affine_map<(tile_i, tile_j, i, j) -> (tile_i * 32 + i, tile_j * 32 + j)>
```

### Affine Transformations

| Transformation | Effect | AI Use Case |
|---|---|---|
| **Loop tiling** | Split loop into outer (tile) and inner (element) | Cache blocking for matmul; tile for systolic array |
| **Loop interchange** | Reorder loop dimensions | Match memory layout for contiguous access |
| **Loop fusion** | Merge adjacent loop nests | Fuse matmul + activation to avoid intermediate buffer |
| **Loop distribution** | Split one loop into separate loops | Separate compute-bound from memory-bound phases |
| **Loop skewing** | Shift iteration space for parallelism | Enable wavefront parallelism on systolic arrays |
| **Unroll-and-jam** | Unroll outer loop, fuse with inner | Increase register reuse in GEMM micro-kernels |

```bash
# Apply affine loop tiling
mlir-opt --affine-loop-tile="tile-size=32" input.mlir

# Apply loop interchange (reorder dimensions)
mlir-opt --affine-loop-interchange="permutation=1,0,2" input.mlir

# Apply loop fusion
mlir-opt --affine-loop-fusion input.mlir

# Run full affine optimization pipeline
mlir-opt --affine-loop-tile --affine-loop-interchange --affine-loop-unroll input.mlir
```

---

## 4. The Vector Dialect: Hardware Vector Mapping

The `vector` dialect bridges the gap between abstract loop operations and **hardware SIMD/vector instructions**. It represents **fixed-size, multi-dimensional vector operations** — the abstraction that maps to AVX-512, NEON/SVE, GPU warp operations, or accelerator matrix units.

```mlir
// Transfer data from memref to vector (tile load)
%tile = vector.transfer_read %A[%i, %k], %pad
    : memref<128x64xf32>, vector<4x8xf32>
// Reads a 4×8 tile from memory into a vector register tile

// Vector contraction (generalized matmul on vector registers)
// C_vec[i,j] += A_vec[i,k] * B_vec[k,j]
%result = vector.contract {
    indexing_maps = [
      affine_map<(i, j, k) -> (i, k)>,  // A access
      affine_map<(i, j, k) -> (k, j)>,  // B access
      affine_map<(i, j, k) -> (i, j)>   // C access
    ],
    iterator_types = ["parallel", "parallel", "reduction"]
  } %a_vec, %b_vec, %c_vec
    : vector<4x8xf32>, vector<8x4xf32> into vector<4x4xf32>

// Transfer result back to memory
vector.transfer_write %result, %C[%i, %j]
    : vector<4x4xf32>, memref<128x256xf32>

// Element-wise vector operations
%relu = arith.maximumf %vec, %zero_vec : vector<8xf32>

// Reduction
%sum = vector.reduction <add>, %vec : vector<8xf32> into f32

// Broadcast (replicate scalar to vector)
%bcast = vector.broadcast %scalar : f32 to vector<8xf32>

// Shuffle and transpose
%transposed = vector.transpose %mat, [1, 0] : vector<4x8xf32> to vector<8x4xf32>
```

### Vector to Hardware Mapping

The vector dialect is designed to lower naturally to different hardware:

| Vector Operation | x86 AVX-512 | ARM SVE | GPU (CUDA) | Custom Accelerator |
|---|---|---|---|---|
| `vector.transfer_read` | `vmovups` | `ld1` | `ld.global` | DMA load |
| `vector.contract` (matmul) | FMA loop | outer product | `mma.sync` (tensor core) | Systolic MAC |
| `vector.reduction <add>` | `vhaddps` chain | `faddv` | warp shuffle reduce | Accumulator read |
| `arith.maximumf` (ReLU) | `vmaxps` | `fmax` | `max.f32` | Activation unit |

> **Key Insight:** The `vector.contract` operation is the bridge between `linalg.matmul` (abstract) and hardware matrix instructions (concrete). When targeting a GPU, `vector.contract` with tile sizes matching tensor core dimensions (16×16×16) lowers to `mma.sync`. When targeting your custom accelerator, the same operation lowers to your systolic array instruction. The compiler only needs to choose the right tile sizes — the lowering is mechanical.

---

## 5. The GPU Dialect: Mapping to Parallel Hardware

The `gpu` dialect provides operations for expressing GPU execution — **thread/block hierarchy, shared memory, barriers** — in a target-independent way.

```mlir
// Launch a GPU kernel
gpu.launch blocks(%bx, %by, %bz) in (%gbx = %grid_x, %gby = %grid_y, %gbz = %grid_z)
           threads(%tx, %ty, %tz) in (%tbx = %block_x, %tby = %block_y, %tbz = %block_z) {

  // Compute global indices
  %gid_x = arith.addi %bx_offset, %tx : index

  // Allocate shared memory
  %shared = memref.alloc() : memref<32x32xf32, #gpu.address_space<workgroup>>

  // Load tile to shared memory
  %val = memref.load %global[%gid_x, %k] : memref<M×K×f32>
  memref.store %val, %shared[%tx, %ty] : memref<32x32xf32, ...>

  // Barrier: wait for all threads to finish loading
  gpu.barrier

  // Compute on shared memory tile
  // ...

  gpu.terminator
}
```

### Mapping Linalg to GPU

The typical pipeline: `linalg.matmul` → tile → map tiles to GPU blocks/threads → promote tiles to shared memory → vectorize inner computation:

```
linalg.matmul
    │
    ▼  tile [128, 128, 32] — one tile per thread block
    │
    ▼  map outer loops to gpu.block_id
    │
    ▼  tile [32, 32, 32] — one sub-tile per warp
    │
    ▼  map to gpu.thread_id
    │
    ▼  promote A_tile, B_tile to shared memory
    │
    ▼  gpu.barrier
    │
    ▼  vectorize inner 32×32 compute
    │
    ▼  lower vector.contract to mma.sync (tensor core)
```

---

## Putting It All Together: A Full Lowering Example

Tracing a `linalg.matmul` from high-level to near-hardware:

```mlir
// Stage 1: High-level (linalg)
%C = linalg.matmul ins(%A, %B : tensor<512x256xf32>, tensor<256x512xf32>)
                   outs(%C_init : tensor<512x512xf32>) -> tensor<512x512xf32>
```

```mlir
// Stage 2: After tiling (linalg + scf)
scf.for %i = 0 to 512 step 64 {
  scf.for %j = 0 to 512 step 64 {
    scf.for %k = 0 to 256 step 32 {
      %a_tile = tensor.extract_slice %A[%i,%k][64,32][1,1]
      %b_tile = tensor.extract_slice %B[%k,%j][32,64][1,1]
      %c_tile = tensor.extract_slice %C[%i,%j][64,64][1,1]
      %res = linalg.matmul ins(%a_tile, %b_tile) outs(%c_tile)
      %C = tensor.insert_slice %res into %C[%i,%j][64,64][1,1]
    }
  }
}
```

```mlir
// Stage 3: After bufferization (memref)
%A_buf = memref.alloc() : memref<512x256xf32>
%B_buf = memref.alloc() : memref<256x512xf32>
%C_buf = memref.alloc() : memref<512x512xf32>
scf.for %i = 0 to 512 step 64 {
  scf.for %j = 0 to 512 step 64 {
    %a_sub = memref.subview %A_buf[%i,0][64,256][1,1]
    %b_sub = memref.subview %B_buf[0,%j][256,64][1,1]
    %c_sub = memref.subview %C_buf[%i,%j][64,64][1,1]
    scf.for %k = 0 to 256 step 32 {
      // ... load, compute, accumulate ...
    }
  }
}
```

```mlir
// Stage 4: After vectorization (vector)
// Inner loop body becomes vector operations
%a_vec = vector.transfer_read %a_sub[%ii, %kk], %zero : memref<64x32xf32>, vector<8x4xf32>
%b_vec = vector.transfer_read %b_sub[%kk, %jj], %zero : memref<32x64xf32>, vector<4x8xf32>
%c_vec = vector.contract ... %a_vec, %b_vec, %c_acc : ... into vector<8x8xf32>
```

```mlir
// Stage 5: After lowering to LLVM dialect
%ptr = llvm.getelementptr %base[%offset] : (!llvm.ptr, i64) -> !llvm.ptr
%vec = llvm.load %ptr : !llvm.ptr -> vector<8xf32>
%fma = llvm.intr.fmuladd(%a, %b, %c) : (vector<8xf32>, vector<8xf32>, vector<8xf32>) -> vector<8xf32>
```

---

## Hands-On Exercises

1. **Write a linalg.generic:** Express a softmax operation (`exp(x - max(x)) / sum(exp(x - max(x)))`) as a sequence of `linalg.generic` operations. Identify the `indexing_maps` and `iterator_types` for each stage (max reduction, subtract, exp, sum reduction, divide).

2. **Tile and fuse:** Write a `linalg.matmul` followed by a `linalg.generic` (ReLU). Use `mlir-opt --linalg-tile-and-fuse-tensor-ops` to fuse them. Examine the output — verify that the ReLU is inside the tile loops.

3. **Bufferization analysis:** Take the fused matmul+ReLU from exercise 2 and run `--one-shot-bufferize`. Count the number of `memref.alloc` in the output. Is the intermediate buffer between matmul and ReLU eliminated?

4. **Vector lowering:** Write a simple `vector.contract` that computes a 4×4 matrix multiply. Lower it with `--convert-vector-to-llvm` and examine the generated LLVM dialect — identify the FMA instructions.

---

## Key Takeaways

| Dialect | Role in AI Compilation |
|---|---|
| `linalg` | Structured representation of tensor ops; enables tiling, fusion, interchange |
| `tensor` | Immutable tensor values; enables functional optimization, deferred allocation |
| `memref` | Memory-backed buffers; required for code generation |
| `affine` | Polyhedral loop analysis; optimal tiling and scheduling |
| `scf` | General structured control flow; loop nests after tiling |
| `vector` | Fixed-size vector tiles; maps to SIMD, tensor cores, MAC arrays |
| `gpu` | Thread/block hierarchy; shared memory promotion; barriers |

The compilation pipeline for any AI workload follows the pattern:
**linalg → tile → fuse → bufferize → vectorize → map to hardware → emit**

---

## Resources

* **[Linalg Dialect Documentation](https://mlir.llvm.org/docs/Dialects/Linalg/):** Reference for all linalg operations and transformations.
* **[Linalg on Tensors (MLIR RFC)](https://discourse.llvm.org/t/rfc-linalg-on-tensors/):** Design rationale for tensor-based linalg.
* **[One-Shot Bufferize](https://mlir.llvm.org/docs/Bufferization/):** How MLIR converts tensors to memory references.
* **[Structured Code Generation in MLIR](https://www.youtube.com/watch?v=5gXsRJuN5Hg):** Talk by Nicolas Vasilache on the linalg→vector pipeline.
* **[IREE Compiler](https://iree.dev/):** Google's production ML compiler built on MLIR — study its pipeline for real-world patterns.
* **"Compiler Design: Virtual Machines" by Reinhard Wilhelm et al.:** Background on IR design and optimization frameworks.
