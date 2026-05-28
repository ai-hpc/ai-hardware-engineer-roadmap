# Orin Nano — Tensor Core Architecture and How It Works

<div class="course-identity auto-course" style="--course-accent: #4f46e5; --course-accent-rgb: 79, 70, 229;" markdown="1">
<div class="course-identity__icon">ONTC</div>
<div markdown="1">
<p class="course-identity__eyebrow">Deep Dive · Jetson Track</p>
<p class="course-identity__title">Specialized course identity for Orin Nano — Tensor Core Architecture and How It Works.</p>
<p class="course-identity__meta">Artifact: Jetson integration demo · Measure: latency, memory, power, logs</p>
</div>
</div>


> **Context:** This deep dive explains **tensor core** architecture on the Jetson Orin Nano (Ampere GPU) and how it enables fast, power-efficient AI inference. Understanding this helps you choose precisions (FP16/INT8), interpret benchmarks, and reason about why TensorRT and cuDNN achieve high TOPS on Orin.

---


## 1. What Are Tensor Cores?

**Tensor Cores** are dedicated hardware units inside NVIDIA GPUs that perform **matrix multiply-accumulate (MMA)** operations in a single instruction. They are optimized for the dense matrix math that dominates deep learning: linear layers, convolutions, and attention are all built from matrix multiplies.

* **Introduced:** Volta (2017); evolved in Turing, Ampere, Ada, Hopper, Blackwell.
* **Orin Nano GPU:** Based on **Ampere** architecture — the same family as datacenter A100, but scaled down for edge (fewer SMs, lower power).

In one cycle, a tensor core can do many more multiply-adds than a CUDA core on the same matrix operation. That is why **FP16** and **INT8** inference on Orin Nano can reach **40 AI TOPS** (trillion operations per second) while staying within a few watts: most of the work is done by tensor cores, not by the general-purpose CUDA cores.

---

## 2. Tensor Cores vs CUDA Cores

| Aspect | CUDA Cores | Tensor Cores |
|--------|------------|--------------|
| **Function** | General-purpose: scalar/vector math, logic, memory ops | Specialized: matrix multiply-accumulate (D = A×B + C) |
| **Granularity** | Per-thread: one or a few ops per instruction | Per-warp: one instruction does a full small matrix (e.g. 16×16×16) |
| **Precision** | FP32, INT32, etc. | FP16, BF16, TF32, INT8, INT4 (architecture-dependent) |
| **Used for** | Non-matmul work: activations, reductions, element-wise, control flow | Matmul: linear layers, conv, attention (when expressed as matmul) |
| **Throughput** | Lower ops/cycle for matrix math | Much higher ops/cycle for matrix math |

On Orin Nano (Ampere):

* **1024 CUDA cores** — handle everything that is not a dense matmul: activations (ReLU, GELU), softmax, normalization, data movement, custom kernels.
* **32 tensor cores** — handle the heavy matmul work. When TensorRT or cuDNN runs a layer, they schedule **WMMA** (Warp Matrix Multiply-Accumulate) or library-built kernels that target these 32 tensor cores.

So: **CUDA cores** = general compute; **tensor cores** = matrix engines. Inference is fast when most time is spent in tensor-core matmul and the rest is minimal (fused ops, good memory access).

---

## 3. Ampere Tensor Core Architecture (Orin Nano)

Orin Nano’s GPU is a **Tegra234** (T234) SoC with an **Ampere**-class GPU. The exact layout is proprietary, but the public model is:

* **Streaming Multiprocessors (SMs):** The GPU is divided into SMs. Each SM has:
  * **CUDA cores** (integer and floating-point)
  * **Tensor cores** (one or more per SM)
  * **Shared memory**, **L1 cache**, **warp schedulers**

* **Orin Nano 8GB** has **32 tensor cores** total (often quoted as “32 tensor cores” in the key specs). These are shared across all SMs; each SM can issue tensor-core instructions from its warps.

* **Memory path:** Tensor cores read **A** and **B** matrices (and optionally **C** for accumulate) from **registers** and/or **shared memory**. Data is brought from **global memory** (LPDDR5) by CUDA cores or load instructions, then staged in shared memory and registers so that tensor cores operate on tiles. So **memory bandwidth** (LPDDR5 speed) and **tiling** (how well you reuse data in shared memory) still limit peak tensor-core utilization.

Conceptually:

```
                    Orin Nano GPU (Ampere)
┌─────────────────────────────────────────────────────────────┐
│  SMs (Streaming Multiprocessors)                             │
│  ┌─────────────┐  ┌─────────────┐       ┌─────────────┐     │
│  │ CUDA cores  │  │ Tensor cores│  ...  │ Tensor cores│     │
│  │ Shared mem  │  │ (MMA units) │       │ (MMA units) │     │
│  └─────────────┘  └─────────────┘       └─────────────┘     │
│         ↕                  ↕                      ↕            │
│  L1 / Shared memory ←→ Register file ←→ Tensor Core arrays    │
└─────────────────────────────────────────────────────────────┘
                              ↕
                    L2 cache / Unified memory (LPDDR5)
```

---

## 4. How Tensor Cores Work: Matrix Multiply-Accumulate

### 4.1 The Operation

Tensor cores compute:

**D = A × B + C**

* **A**, **B**: input matrices (e.g. activations and weights).
* **C**: accumulator (often the previous partial result).
* **D**: output (accumulated result).

This is exactly the pattern of a linear layer or a convolution (when flattened to matmul). One **tensor-core instruction** completes a small block of this (e.g. a 16×16×16 MMA in FP16). Many such blocks are scheduled by the compiler/runtime to cover the full matrix.

### 4.2 Warp-Level Operation

Tensor cores are **warp-level**: one **warp** (32 threads) cooperates to feed one MMA. The threads in the warp hold different parts of the matrices (e.g. different rows/columns of the 16×16 tile). The hardware executes the full small matmul in one go. So:

* You don’t write a loop of scalar multiplies; you **load tiles** into registers (and shared memory), then **issue one WMMA instruction** per tile.
* **Tiling** (how you break the big matrix into 16×16 or 8×8 blocks) is chosen so that tiles fit in registers and shared memory and so that tensor cores stay busy.

### 4.3 Data Flow (Simplified)

1. **Load:** CUDA loads tiles of **A** and **B** from global/shared memory into registers (or shared memory for the tensor core to read).
2. **Compute:** Tensor core instruction: **D = A×B + C** on that tile.
3. **Store:** Result **D** is written back to shared memory or registers; then written to global memory or reused for the next layer.

Efficient kernels **fuse** steps (e.g. add bias, ReLU) in the same kernel so that result **D** is not written to global memory and read back — that saves bandwidth and improves performance.

---

## 5. Precision Support: FP16, BF16, INT8, TF32

Tensor cores support different **numeric formats**; each trades precision for throughput and power.

| Format | Bit width | Typical use | Throughput (vs FP32) | Orin Nano / Ampere |
|--------|-----------|-------------|------------------------|---------------------|
| **FP32** | 32-bit float | Training, reference | 1× (no tensor core) | CUDA cores only |
| **TF32** | 19-bit (mantissa truncated) | Training on Ampere+ | High | Ampere supports |
| **FP16** | 16-bit float | Inference, mixed precision | ~2× (tensor core) | ✅ Native |
| **BF16** | 16-bit (same exponent range as FP32) | Training, some inference | ~2× (tensor core) | ✅ Native |
| **INT8** | 8-bit integer | Quantized inference | ~4× (tensor core) | ✅ Native |
| **INT4** | 4-bit | Very low bit inference | Higher still | Architecture-dependent |

On **Orin Nano (Ampere)** for inference you care most about:

* **FP16** — Default for TensorRT and many models; tensor cores are used; good accuracy.
* **INT8** — Quantized models; 2× or more speedup over FP16 when calibrated correctly; slight accuracy loss.
* **FP32** — No tensor cores; runs on CUDA cores; slower, used for debugging or when precision is required.

TensorRT on Jetson will choose kernels that use tensor cores when you build an engine with `--fp16` or `--int8`. So “how tensor cores work” directly explains why FP16/INT8 engines are so much faster than FP32 on the same hardware.

---

## 6. How Software Uses Tensor Cores (TensorRT, cuDNN)

### 6.1 TensorRT

When you build a TensorRT engine (e.g. `trtexec --onnx=model.onnx --fp16`):

1. The **ONNX** (or other) graph is parsed and optimized.
2. Layers (e.g. `Gemm`, `Conv`) are **mapped to GPU kernels**. For linear/conv layers, TensorRT selects **tensor-core kernels** (FP16 or INT8) when available.
3. The engine is a **sequence of kernel launches**. Many of those kernels are **WMMA-based** or use NVIDIA’s internal tensor-core APIs so that the 32 tensor cores on Orin Nano are used for the bulk of the math.
4. **Fusion** (e.g. conv + bias + ReLU in one kernel) keeps data in registers/shared memory and reduces round-trips to global memory.

You don’t write tensor-core code by hand for inference; TensorRT (and cuDNN under the hood) do it. Understanding that they are using tensor cores explains the **40 TOPS** and the big gain from `--fp16` / `--int8`.

### 6.2 cuDNN

cuDNN provides **routines** for convolutions, matmuls, and other ops. On Ampere, these routines use **tensor-core implementations** when the problem size and precision match. PyTorch and other frameworks call cuDNN; TensorRT also uses cuDNN or its own tensor-core kernels. So “software uses tensor cores” means: **TensorRT and cuDNN (and thus most frameworks) automatically schedule tensor-core MMA instructions** when you use FP16/INT8.

### 6.3 Writing Your Own Kernels (WMMA, CUTLASS)

If you write custom CUDA:

* **WMMA (Warp Matrix Multiply-Accumulate)** — PTX/CUDA APIs let a warp perform a small matrix multiply (e.g. 16×16×16) in one instruction. The compiler lowers this to tensor-core instructions.
* **CUTLASS / CuTe** — NVIDIA’s template library for GEMM and related ops; it explicitly tiles and schedules tensor-core MMAs. Used when you need maximum control (e.g. custom shapes, fusions).

On Orin Nano, most users rely on TensorRT/cuDNN; the deep dive here is so you know **what** is running on the 32 tensor cores when you enable FP16/INT8.

---

## 7. Why This Matters for Edge Inference

* **Throughput:** Tensor cores deliver most of the **40 AI TOPS** on Orin Nano 8GB. Without them, the same chip would be much slower on neural networks.
* **Power:** Doing more ops per cycle means the same workload finishes sooner and the GPU can sleep sooner — important for battery and thermal limits.
* **Precision choice:** FP16 and INT8 are the “tensor-core paths”; FP32 is the slow path. Picking FP16 (or INT8 with calibration) is how you get both speed and acceptable accuracy.
* **Debugging:** If a model is slow, check that the engine is built with FP16/INT8 and that layers are not falling back to FP32 or to non–tensor-core kernels (e.g. small or odd shapes).
* **DLA vs GPU:** Orin Nano also has a **DLA** (Deep Learning Accelerator). The DLA is a separate fixed-function block; the **tensor cores** are inside the **GPU**. TensorRT can run some layers on DLA and others on GPU (tensor cores + CUDA cores). So “tensor cores” = GPU matmul acceleration; “DLA” = separate accelerator for supported ops.

---

## 8. Resources

* **NVIDIA Ampere Architecture (white paper)** — Tensor core description and block diagrams.
* **NVIDIA CUDA Programming Guide** — WMMA API and warp-level matrix ops.
* **TensorRT Developer Guide** — How TensorRT selects kernels and uses FP16/INT8.
* **Jetson Orin Nano datasheet / technical brief** — Official SM and tensor core counts and TOPS.
* **cuDNN Developer Guide** — Convolution and matmul algorithms and tensor-core usage.

---

*Back to [Nvidia Jetson Platform — Practical Complete Guide](../Guide.md).*
