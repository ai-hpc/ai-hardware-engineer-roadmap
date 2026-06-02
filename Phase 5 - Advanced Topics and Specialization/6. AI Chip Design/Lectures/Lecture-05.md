# Lecture 5: ML-to-Hardware Compilation Pipelines — TVM, MLIR-Based Compilers & tinygrad

## Overview

The previous four lectures built up the foundation: LLVM IR as the universal low-level representation (Lectures 1–2), and MLIR as the multi-level framework for progressive lowering (Lectures 3–4). This lecture connects everything to the real world: how do actual ML compilers take a PyTorch or TensorFlow model and produce optimized code for GPUs, NPUs, FPGAs, and custom AI accelerators? The core challenge is understanding the **end-to-end pipeline** — from a `model.forward()` call in Python to fused, tiled, vectorized machine code running on hardware. We examine three compiler families that represent different design philosophies: **Apache TVM** (schedule-based, LLVM backend), **MLIR-based compilers** (IREE, Triton-MLIR, torch-mlir), and **tinygrad** (minimal lazy-evaluation compiler). For an AI hardware engineer, understanding these pipelines tells you exactly where your custom backend plugs in — and what the compiler needs from your hardware description.

---

## The Big Picture: Three Compiler Families

```
                        ML Model (PyTorch, ONNX, etc.)
                                    │
              ┌─────────────────────┼─────────────────────┐
              ▼                     ▼                     ▼
     ┌─────────────────┐  ┌──────────────────┐  ┌─────────────────┐
     │   Apache TVM    │  │  MLIR-Based      │  │    tinygrad      │
     │                 │  │                  │  │                 │
     │  Relay/Relax IR │  │  torch-mlir /    │  │  LazyBuffer DAG │
     │       │         │  │  StableHLO       │  │       │         │
     │       ▼         │  │       │          │  │       ▼         │
     │  TIR (Tensor IR)│  │       ▼          │  │  UOp Graph      │
     │       │         │  │  linalg/tensor   │  │  (linearized IR)│
     │       ▼         │  │       │          │  │       │         │
     │  Schedule +     │  │       ▼          │  │       ▼         │
     │  Auto-tuning    │  │  tiling/fusion   │  │  BEAM search    │
     │       │         │  │       │          │  │  (auto-tuning)  │
     │       ▼         │  │       ▼          │  │       │         │
     │  LLVM codegen   │  │  vector/gpu/llvm │  │       ▼         │
     │  (or microTVM)  │  │  dialect lowering│  │  Backend codegen│
     │       │         │  │       │          │  │  (CUDA/OpenCL/  │
     │       ▼         │  │       ▼          │  │   Metal/custom) │
     │  x86/ARM/CUDA/  │  │  LLVM / NVVM /  │  │       │         │
     │  FPGA / custom  │  │  SPIRV           │  │       ▼         │
     └─────────────────┘  └──────────────────┘  └─────────────────┘
```

---

## 1. Apache TVM

TVM is the **most mature open-source ML compiler**. Its key innovation is **separating computation from schedule** — the same algorithm can be optimized differently for different hardware via schedule transformations.

### Pipeline

```
PyTorch / ONNX / TensorFlow
         │
         ▼
┌──────────────────────┐
│  Frontend Import     │  torch.export → Relay/Relax graph
│  (Relay or Relax IR) │  ONNX → Relay graph
└──────────┬───────────┘
           │
           ▼
┌──────────────────────┐
│  Graph-Level Passes  │  Operator fusion (conv+bn+relu → single op)
│                      │  Constant folding, layout optimization (NCHW→NHWC)
│                      │  Quantization-aware rewrites
└──────────┬───────────┘
           │
           ▼
┌──────────────────────┐
│  TIR (Tensor IR)     │  Low-level loop representation
│  + Schedule          │  Each op becomes a loop nest with schedule primitives
└──────────┬───────────┘
           │
           ▼
┌──────────────────────┐
│  Auto-Tuning         │  AutoTVM / MetaSchedule / ANSOR
│                      │  Search tile sizes, unroll factors, vectorization widths
│                      │  for the specific target hardware
└──────────┬───────────┘
           │
           ▼
┌──────────────────────┐
│  Code Generation     │  TIR → LLVM IR (CPU targets)
│                      │  TIR → CUDA source (GPU targets)
│                      │  TIR → C code (microTVM for MCUs)
│                      │  TIR → Verilog (FPGA, experimental)
└──────────────────────┘
```

### TIR: Tensor IR

TIR is TVM's low-level IR — roughly equivalent to MLIR's `affine` + `scf` dialects but with TVM-specific scheduling primitives.

```python
# TVM TIR for a matrix multiply (before scheduling)
import tvm
from tvm import te

M, N, K = 1024, 1024, 1024
A = te.placeholder((M, K), name="A", dtype="float32")
B = te.placeholder((K, N), name="B", dtype="float32")

# Define computation
k = te.reduce_axis((0, K), name="k")
C = te.compute((M, N), lambda i, j: te.sum(A[i, k] * B[k, j], axis=k), name="C")

# Create schedule
s = te.create_schedule(C.op)

# Apply schedule transformations (this is where hardware knowledge enters)
# Split loops for tiling
xo, xi = s[C].split(s[C].op.axis[0], factor=32)   # tile M by 32
yo, yi = s[C].split(s[C].op.axis[1], factor=32)   # tile N by 32
ko, ki = s[C].split(k, factor=4)                   # tile K by 4

# Reorder for cache locality: tile-level outer loops, element-level inner
s[C].reorder(xo, yo, ko, xi, yi, ki)

# Vectorize innermost loop
s[C].vectorize(yi)

# Unroll the k-reduction inner loop
s[C].unroll(ki)
```

### TVM → LLVM Code Generation

TVM's LLVM codegen (`src/target/llvm/codegen_llvm.cc`) translates TIR directly to LLVM IR using the IRBuilder API:

```python
# Generate LLVM code for x86 with AVX2
target = tvm.target.Target("llvm -mcpu=skylake -mattr=+avx2,+fma")
func = tvm.build(s, [A, B, C], target=target, name="matmul")

# Inspect the generated LLVM IR
print(func.get_source("ll"))
# Shows vectorized loops with <8 x float> operations, FMA intrinsics
```

### Auto-Tuning: MetaSchedule

The key insight of TVM's auto-tuning: instead of hand-writing schedules for every hardware target, **search the space of valid schedules automatically**.

```python
from tvm import meta_schedule as ms

# Define the search space
database = ms.tune_tir(
    mod=tir_module,
    target="nvidia/nvidia-a100",
    max_trials_global=2000,
    work_dir="./tune_results",
)

# The tuner explores tile sizes, loop orders, vectorization widths,
# shared memory usage, and thread binding — evaluating each candidate
# by compiling and running it on the actual hardware
```

| Auto-Tuner | Approach | Search Space |
|---|---|---|
| **AutoTVM** | Template-based: human writes schedule template with knobs | Bounded by template design |
| **ANSOR** | Task-level: generates sketch + random annotation | Larger; discovers novel schedules |
| **MetaSchedule** | Unified: trace-based with modular search rules | Most flexible; production-ready |

### microTVM: Targeting Microcontrollers

For edge AI, TVM can compile models to bare-metal C code that runs on Cortex-M and RISC-V microcontrollers:

```python
target = tvm.target.Target("c -mcpu=cortex-m7")
# Generates C code with CMSIS-NN integration
# Runs on devices with as little as 256KB SRAM
```

> **Key Insight:** TVM's power is that the same model description (Relay graph) compiles to AVX-512 code on a server, CUDA kernels on a GPU, and CMSIS-NN calls on a Cortex-M — by changing only the target and schedule. For an AI hardware engineer, adding a new target to TVM means implementing a code generator (often just LLVM backend + scheduling rules) and providing auto-tuning parameters.

---

## 2. MLIR-Based Compilers

### torch-mlir: PyTorch → MLIR

`torch-mlir` converts PyTorch models into MLIR, serving as the entry point for MLIR-based compilation pipelines.

```
PyTorch Model
      │
      ▼  (torch.export / torch.jit.trace)
TorchScript / FX Graph
      │
      ▼  (torch-mlir)
┌─────────────────────────────────────┐
│  Torch Dialect (MLIR)               │
│  torch.aten.mm, torch.aten.relu,    │
│  torch.aten.conv2d, ...             │
└──────────────┬──────────────────────┘
               │
      ┌────────┼────────┐
      ▼        ▼        ▼
   TOSA    StableHLO   Linalg        Multiple lowering targets
```

```python
# Using torch-mlir
import torch
import torch_mlir

class MyModel(torch.nn.Module):
    def forward(self, x, y):
        return torch.mm(x, y).relu()

model = MyModel()
example_input = (torch.randn(128, 64), torch.randn(64, 256))

# Export to MLIR (linalg dialect)
module = torch_mlir.compile(model, example_input,
                            output_type="linalg-on-tensors")
print(module.operation.get_asm())
```

Output:
```mlir
func.func @forward(%arg0: tensor<128x64xf32>, %arg1: tensor<64x256xf32>) -> tensor<128x256xf32> {
  %cst = arith.constant 0.0 : f32
  %0 = tensor.empty() : tensor<128x256xf32>
  %1 = linalg.fill ins(%cst) outs(%0) -> tensor<128x256xf32>
  %2 = linalg.matmul ins(%arg0, %arg1) outs(%1) -> tensor<128x256xf32>
  %3 = linalg.generic {/* relu */} ins(%2) outs(%0) -> tensor<128x256xf32>
  return %3 : tensor<128x256xf32>
}
```

### IREE: Google's Production ML Compiler

IREE (Intermediate Representation Execution Environment) is the most complete MLIR-based ML compiler, targeting CPUs, GPUs, and custom accelerators.

```
StableHLO / TOSA / Linalg
         │
         ▼
┌────────────────────────────┐
│  IREE Flow Dialect         │  Graph-level: dispatch region formation
│  (workload partitioning)   │  Decides which ops run together as one kernel
└────────────┬───────────────┘
             │
             ▼
┌────────────────────────────┐
│  IREE Stream Dialect       │  Execution scheduling: async dispatch,
│  (resource management)     │  buffer allocation, synchronization
└────────────┬───────────────┘
             │
             ▼
┌────────────────────────────┐
│  IREE HAL Dialect          │  Hardware Abstraction Layer:
│  (hardware abstraction)    │  command buffers, executables, buffers
└────────────┬───────────────┘
             │
        ┌────┼────┐
        ▼    ▼    ▼
     LLVM  SPIR-V  VMVX       Backend targets
     (CPU) (GPU)   (portable VM)
```

**IREE's key innovation:** It treats ML compilation as a **systems problem**, not just a kernel optimization problem. It handles:
- Multi-kernel scheduling and pipelining
- Async execution across heterogeneous devices
- Memory allocation and lifetime management
- Executable packaging and deployment

### Triton's MLIR Pipeline

Triton (used by `torch.compile`) has transitioned to an MLIR-based pipeline:

```
Triton Python (user-written kernel)
         │
         ▼
┌────────────────────────┐
│  Triton IR (TTIR)      │  High-level: block-level operations
│  tt.dot, tt.load,      │  on tensor<128x128xf16> blocks
│  tt.store, tt.reduce   │
└────────────┬───────────┘
             │
             ▼
┌────────────────────────┐
│  Triton GPU IR (TTGIR) │  GPU-specific: thread/warp layout,
│  shared memory alloc,  │  data movement planning
│  pipeline stages       │
└────────────┬───────────┘
             │
             ▼
┌────────────────────────┐
│  LLVM Dialect (MLIR)   │  Low-level: near-LLVM-IR operations
│  + NVVM intrinsics     │  with GPU-specific intrinsics
└────────────┬───────────┘
             │
             ▼
         LLVM IR → PTX → cubin
```

```python
# Triton kernel — compiles through MLIR
import triton
import triton.language as tl

@triton.jit
def matmul_kernel(A, B, C, M, N, K,
                  BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    # Compute block pointers
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    # Accumulator
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k in range(0, K, BLOCK_K):
        offs_k = k + tl.arange(0, BLOCK_K)
        a = tl.load(A + offs_m[:, None] * K + offs_k[None, :])
        b = tl.load(B + offs_k[:, None] * N + offs_n[None, :])
        acc += tl.dot(a, b)    # This becomes tt.dot → WGMMA on Hopper

    tl.store(C + offs_m[:, None] * N + offs_n[None, :], acc)
```

Triton's `tl.dot` is lowered through TTIR → TTGIR → LLVM+NVVM, and on Hopper it ultimately emits `wgmma.mma_async` instructions. The MLIR infrastructure makes each transformation stage composable and debuggable.

---

## 3. tinygrad: The Minimal Compiler

tinygrad takes a **radically different approach**: instead of LLVM/MLIR infrastructure, it implements a **self-contained compiler in ~10K lines of Python**.

### Pipeline

```
Python tensor operations (tinygrad API)
         │
         ▼
┌────────────────────────┐
│  LazyBuffer DAG        │  Deferred execution: operations are recorded,
│                        │  not executed. Creates a computation graph.
└────────────┬───────────┘
             │
             ▼
┌────────────────────────┐
│  Scheduler             │  Decides kernel boundaries: which ops fuse
│  (kernel partitioning) │  into one kernel launch.
└────────────┬───────────┘
             │
             ▼
┌────────────────────────┐
│  UOp Graph             │  Linearized IR: ~12 primitive operations
│  (linearized IR)       │  (LOAD, STORE, ALU, REDUCE, CONST, etc.)
└────────────┬───────────┘
             │
             ▼
┌────────────────────────┐
│  Optimization          │  Pattern-matching rewrite rules
│  + BEAM Search         │  Auto-tune: tile sizes, local sizes,
│                        │  unroll factors, upcast widths
└────────────┬───────────┘
             │
             ▼
┌────────────────────────┐
│  Backend Codegen       │  Emit source code for target:
│                        │  CUDA, OpenCL, Metal, HIP, LLVM, WebGPU
│                        │  Each backend: ~200-500 lines of Python
└────────────────────────┘
```

### UOp IR: The 12 Primitives

tinygrad reduces all tensor computation to ~12 primitive operations:

| UOp | Meaning | Example |
|---|---|---|
| `LOAD` | Read from buffer | Load weight tensor element |
| `STORE` | Write to buffer | Store output activation |
| `CONST` | Constant value | Zero for bias init, scale factor |
| `ALU` | Arithmetic (add, mul, max, etc.) | Element-wise operations |
| `REDUCE` | Reduction (sum, max) over axis | Sum over k-dimension in matmul |
| `DEFINE_GLOBAL` | Declare a buffer argument | Input/output tensor pointers |
| `DEFINE_LOCAL` | Declare local/shared memory | Shared memory tile |
| `DEFINE_ACC` | Declare accumulator | Register for partial sums |
| `RANGE` | Loop range | Iteration over tiles |
| `SPECIAL` | Thread/block index | `threadIdx.x`, `blockIdx.y` |
| `BARRIER` | Synchronization | `__syncthreads()` |
| `CAST` | Type conversion | FP32→FP16 for mixed precision |

### tinygrad Codegen Example

```python
import tinygrad
from tinygrad import Tensor

# This lazy expression builds a UOp graph — nothing executes yet
a = Tensor.rand(1024, 512)
b = Tensor.rand(512, 1024)
c = (a @ b).relu()  # matmul + relu — the scheduler will fuse these

# Force execution — triggers compilation and kernel launch
c.realize()
```

What happens inside:
1. `a @ b` creates a `LazyBuffer` with op=`MATMUL(a, b)`
2. `.relu()` creates a `LazyBuffer` with op=`MAX(matmul_result, 0)`
3. `.realize()` triggers the scheduler
4. Scheduler sees matmul→relu chain and fuses into **one kernel**
5. UOp graph is built: nested RANGE loops, LOAD from a/b, ALU mul+add (reduction), ALU max (relu), STORE
6. BEAM search finds optimal tile sizes for the target GPU
7. Backend emits CUDA/OpenCL/Metal source code
8. Code is compiled (nvcc/clang) and launched

### tinygrad vs. LLVM/MLIR

| Aspect | tinygrad | TVM / MLIR |
|---|---|---|
| **IR complexity** | ~12 UOps | Hundreds of ops across multiple dialects |
| **Codebase** | ~10K Python | Millions of lines of C++ |
| **Backend effort** | ~300 lines per target | Thousands of lines per target |
| **Optimization** | BEAM search (runtime) | Compile-time analysis + optional tuning |
| **Maturity** | Active development, limited hardware | Production-proven on many targets |
| **Hackability** | One person can understand the whole compiler | Team effort to understand fully |
| **Peak performance** | 70–90% of vendor libraries (improving) | 90–100% with vendor backends |
| **Best for** | Non-NVIDIA hardware, rapid prototyping | Production deployment, NVIDIA GPU, custom ASIC |

> **Key Insight:** tinygrad proves that a functional ML compiler can be built in remarkably few lines — the core compilation logic is simpler than most engineers assume. Its limitation is that peak performance on NVIDIA hardware requires architecture-specific templates (FlashAttention, WGMMA) that can't be derived from 12 primitive operations. The lesson for hardware designers: if your accelerator's programming model is simple enough, a tinygrad-style compiler may be all you need. If your hardware has complex, non-orthogonal features (like tensor cores with specific layout requirements), you'll need MLIR-level infrastructure to express them.

---

## 4. Comparison: Where Each Pipeline Excels

### Choosing a Compiler for Your Hardware

| Your Hardware | Recommended Compiler | Why |
|---|---|---|
| **Standard CPU (x86, ARM)** | TVM with LLVM backend | Mature auto-tuning, LLVM codegen handles SIMD well |
| **NVIDIA GPU** | Triton (MLIR) or direct CUDA | Triton for custom kernels; cuDNN/cuBLAS for standard ops |
| **AMD GPU** | TVM or IREE (MLIR) | ROCm LLVM backend is production-ready |
| **Mobile NPU (Qualcomm, MediaTek)** | tinygrad or TVM | tinygrad already has Qualcomm backend; TVM has broad mobile support |
| **Edge TPU (Coral)** | TensorFlow Lite + EdgeTPU compiler | Proprietary, no open-source alternative |
| **Custom FPGA accelerator** | MLIR custom dialect → HLS/RTL | MLIR progressive lowering maps naturally to FPGA design |
| **Custom ASIC** | MLIR custom dialect → your toolchain | Define ops matching your hardware; lower through MLIR stack |
| **MCU (Cortex-M, RISC-V)** | microTVM or TFLM | microTVM for auto-tuned C; TFLM for hand-optimized CMSIS-NN |

### End-to-End Example: Custom Accelerator with MLIR

If you're designing a custom AI accelerator, here's how the compiler pipeline would look:

```
PyTorch Model
      │
      ▼  torch-mlir
Linalg on Tensors (MLIR)
      │
      ▼  linalg tiling (tile to your MAC array size, e.g., 16×16)
Tiled Linalg + SCF loops
      │
      ▼  custom lowering pass
Your Accelerator Dialect (MLIR)
      │  myaccel.dma_load %weight_sram, %global_weights, [tile_i, tile_k]
      │  myaccel.dma_load %act_sram, %global_acts, [tile_k, tile_j]
      │  myaccel.matmul %acc_regs, %weight_sram, %act_sram
      │  myaccel.activate %acc_regs, "relu"
      │  myaccel.dma_store %global_output, %acc_regs, [tile_i, tile_j]
      │
      ▼  your backend (MLIR → assembly or binary)
Custom Assembly / Binary
      │
      ▼  your assembler / runtime
Executable on your chip
```

The critical decisions are:
1. **Tile sizes** — must match your hardware's MAC array dimensions and SRAM capacity
2. **Data layout** — your SRAM banks may require specific data arrangements
3. **Double buffering** — overlap DMA transfers with computation
4. **Fusion** — which ops can run on the accelerator vs. fall back to CPU

---

## 5. The Convergence: Where the Industry Is Heading

The ML compiler landscape is converging around a few key ideas:

**1. MLIR as the common infrastructure.** TVM is integrating MLIR (TVM Unity). Triton has moved to MLIR. IREE is built on MLIR. Hardware vendors (Intel, AMD, Qualcomm) are building MLIR backends. Even tinygrad's UOp graph is conceptually similar to an MLIR dialect.

**2. Separation of algorithm from schedule.** Whether via TVM schedules, MLIR transformations, or tinygrad's BEAM search — the principle is the same: express the computation once, optimize the execution plan separately.

**3. Auto-tuning over hand-written kernels.** The search-based approach (TVM MetaSchedule, tinygrad BEAM, Triton auto-tuning) is replacing hand-optimized library kernels for an increasing fraction of workloads. The exception: critical-path kernels like attention, where hand-written implementations (FlashAttention) still dominate.

**4. Hardware-software co-design.** The compiler's capabilities **constrain the hardware design space**, and vice versa. Designing an accelerator without understanding the compiler pipeline is like designing an ISA without understanding the software — you'll build features that the compiler can't use.

---

## Hands-On Exercises

1. **TVM end-to-end:** Install TVM. Import a ResNet-18 from PyTorch via `torch.export` + `from_exported_program`. Compile for `llvm -mcpu=skylake`. Extract the LLVM IR (`mod.get_source("ll")`). Find the vectorized loop and identify AVX instructions. Then compile for `cuda` and compare the generated CUDA source.

2. **torch-mlir exploration:** Install torch-mlir. Convert a simple model (linear + relu) to MLIR linalg. Then manually run the lowering pipeline: `--linalg-tile --convert-linalg-to-loops --lower-affine --convert-to-llvm`. Examine the output at each stage.

3. **tinygrad kernel inspection:** Install tinygrad. Set `DEBUG=4` environment variable. Run a simple matmul (`Tensor.rand(512,512) @ Tensor.rand(512,512)`). Examine the printed UOp graph and generated kernel source. Change the backend (`CLANG=1` for CPU, `CUDA=1` for GPU) and compare the generated code.

4. **Custom backend design (paper exercise):** Design a compilation pipeline for a hypothetical accelerator with:
   - 32×32 INT8 systolic array
   - 128KB weight SRAM, 64KB activation SRAM
   - DMA engine for host↔SRAM transfers
   - Fused ReLU/ReLU6 in the output pipeline

   Write the MLIR dialect operations, sketch the lowering from `linalg.matmul` to your dialect, and describe the auto-tuning parameters (tile sizes, double-buffer depth, DMA scheduling).

5. **Compiler comparison benchmark:** Take a single model (e.g., MobileNetV2) and compile it with TVM (auto-tuned), ONNX Runtime, and tinygrad for the same target (e.g., x86 CPU or CUDA GPU). Compare inference latency, compile time, and generated code size. Document where each compiler makes different tiling/fusion decisions.

---

## Key Takeaways

| Concept | Why It Matters for AI Hardware |
|---|---|
| TVM's schedule separation | Express computation once, optimize for each hardware target separately |
| MLIR progressive lowering | Each dialect level retains information the next level needs |
| tinygrad's minimalism | A complete ML compiler can be ~10K lines — complexity is a choice, not a requirement |
| Auto-tuning | Searching tile sizes and schedules often beats hand-written optimization |
| Custom MLIR dialects | The mechanism for connecting your accelerator to the ML ecosystem |
| Convergence on MLIR | Industry is standardizing — learn MLIR once, apply to any hardware target |

---

## Resources

* **[Apache TVM Documentation](https://tvm.apache.org/docs/):** Official docs, tutorials, and API reference.
* **[TVM: An Automated End-to-End Optimizing Compiler for Deep Learning (OSDI 2018)](https://www.usenix.org/conference/osdi18/presentation/chen):** The foundational TVM paper.
* **[ANSOR: Generating High-Performance Tensor Programs for Deep Learning (OSDI 2020)](https://www.usenix.org/conference/osdi20/presentation/zheng):** Auto-scheduling for TVM.
* **[torch-mlir GitHub](https://github.com/llvm/torch-mlir):** PyTorch to MLIR bridge.
* **[IREE Documentation](https://iree.dev/):** Google's production MLIR-based ML compiler.
* **[Triton MLIR Pipeline (OpenAI)](https://triton-lang.org/):** How Triton uses MLIR for GPU kernel compilation.
* **[tinygrad GitHub](https://github.com/tinygrad/tinygrad):** The codebase — readable in a weekend.
* **[Compiler Explorer (godbolt.org)](https://godbolt.org/):** Interactive tool for examining compiler output — supports LLVM IR, multiple architectures.
