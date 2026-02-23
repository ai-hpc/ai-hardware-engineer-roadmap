# Tinygrad: A Minimalist Deep Learning Framework

## Overview

Tinygrad is a lightweight neural network framework created by George Hotz (geohot) and maintained by tiny corp. It positions itself between PyTorch and micrograd, offering simplicity without sacrificing functionality.

**Links:**
- Homepage: https://tinygrad.org
- GitHub: https://github.com/tinygrad/tinygrad
- Documentation: https://tinygrad.github.io/tinygrad/quickstart/
- Discord: https://discord.gg/tinygrad

## Core Philosophy

Tinygrad breaks down complex neural networks into just 3 operation types:

### 1. ElementwiseOps
UnaryOps, BinaryOps, and TernaryOps that operate on 1-3 tensors elementwise
- **UnaryOps** (1 input): SQRT, LOG2, EXP2, SIN, NEG, RECIP, CAST
- **BinaryOps** (2 inputs): ADD, MUL, SUB, DIV, MAX, MOD, CMPLT
- **TernaryOps** (3 inputs): WHERE, MULACC

### 2. ReduceOps
Operate on one tensor and return a smaller tensor
- Examples: SUM, MAX

### 3. MovementOps
Virtual ops that move data around, copy-free with ShapeTracker
- Examples: RESHAPE, PERMUTE, EXPAND, etc.

**Note:** No primitive operators for CONV or MATMUL — these are built from basic operations!

## Key Features

- **Extreme simplicity** — Easiest framework to add new accelerators to
- **Lazy evaluation** — All tensors are lazy, enabling aggressive operation fusion
- **Custom kernel compilation** — Compiles a custom kernel for every operation
- **Full training support** — Forward and backward passes with autodiff
- **Hackable** — Entire compiler and IR are visible and modifiable
- **Multi-backend** — Supports NVIDIA, AMD, and other accelerators

## Performance

Tinygrad aims to be 2x faster than PyTorch for common ML papers on 1 NVIDIA GPU.

Speed advantages:
1. Custom kernel compilation for each operation
2. Aggressive operation fusion through lazy tensors
3. 10x+ simpler backend makes optimizations more impactful

## Installation

```bash
pip install tinygrad
```

## Basic Usage

```python
from tinygrad import Tensor

# Create tensors
t1 = Tensor([1, 2, 3, 4, 5])
t2 = Tensor([2, 3, 4, 5, 6])

# Operations (similar to PyTorch)
result = t1 + t2
result = t1 * t2

# Lazy evaluation — computation happens when .realize() is called
result.realize()
```

## Real-World Usage

Tinygrad is used in [Openpilot](https://github.com/commaai/openpilot) (comma.ai ADAS) to run the driving model on Snapdragon 845 GPU, replacing SNPE with:
- Better performance
- ONNX file loading support
- Training support
- Attention mechanism support

## tinygrad vs Vendor SDKs: The SNPE Case Study

Qualcomm isn't asleep — their incentives are just fundamentally different from tinygrad's. The result is a conservative, closed, production-oriented SDK (SNPE) instead of a hacker-friendly, maximum-performance open stack.

### The 2× Performance Gap on Snapdragon 845

tinygrad achieved roughly **2× speedup vs SNPE** (Qualcomm's own library) for openpilot's driving model on the Snapdragon 845. How:

- tinygrad is optimized by people who only care about one thing: wringing maximum performance out of a few target models and GPUs — even if that means relying on undocumented tricks or brittle assumptions (e.g., how Adreno handles image textures, tiling, cache behavior).
- SNPE has to support many customers, models, quantization schemes, and product cycles with a single binary SDK. More abstraction, more safety checks, less aggressively specialized kernels for "weird but high-performing" shapes. Good enough for OEMs, not optimized for one open-source ADAS project.

### Why Qualcomm Doesn't Make SNPE Like tinygrad

| Constraint | Qualcomm (SNPE) | tinygrad |
|-----------|----------------|----------|
| **Risk profile** | Sells into phones/cars with SLAs — a regression in face unlock or camera pipeline is a business problem | Can break main and fix it later |
| **Support matrix** | Must run well on dozens of SoCs, OS versions, model types | "Fast on this GPU and these models — everything else is best-effort" |
| **Hardware docs** | Low-level Hexagon/HTP and Adreno details are under NDA; even internal teams are boxed in by API stability and OEM legal constraints | Reverse-engineer via OpenCL/GL/Vulkan, experiment aggressively |
| **Business model** | Priority is selling silicon + providing a stable SDK for big customers | Priority is performance; no OEM contracts to protect |
| **Strategic interest** | Shipping an open "sharp-edges" framework that bypasses SNPE undercuts their SDK story and creates support expectations they don't want | Freely publish everything |

From Qualcomm's perspective, SNPE being slower than tinygrad in some setups is **acceptable** as long as:
- It's fast enough for OEMs' use cases
- It's stable, supported, and doesn't break every quarter
- It helps sell more Snapdragon-based devices

### What This Means for Open-Source ML Stacks

- The performance gap is proof that **open-source, hardware-aware stacks can beat vendor SDKs on the vendor's own hardware** when allowed to specialize and iterate quickly.
- Qualcomm not competing aggressively on this front leaves space for independent projects to define "best-in-class" performance on Snapdragon — exactly what openpilot + tinygrad did.
- Long-term, this pressure pushes vendors toward either:
  - Exposing more low-level knobs (better Vulkan/CL, perf counters, scheduling hints), or
  - Shipping their own high-performance experimental stacks while keeping SNPE as the conservative default.

### The Systems Architecture Lesson

This is the classic **"vendor SDK for mass market vs. hand-tuned stack for a narrow domain"** story:

```
Vendor SDK (SNPE):
  Optimize for: stability, broad support, OEM contracts
  Accept tradeoff: 2× slower on specific workloads
  Target: millions of devices, dozens of use cases

tinygrad on Adreno:
  Optimize for: one GPU, one model, maximum FLOP/s
  Accept tradeoff: brittle, undocumented, may break
  Target: openpilot's driving model on 845
```

The QCOM backend in tinygrad (`DEVICE=QCOM`) is direct evidence of this: tinygrad ships a first-class Qualcomm GPU backend targeting Adreno — something Qualcomm won't do for you via SNPE because there's no NDA-safe way to expose the same low-level tuning.

## The Same Rule on Jetson Orin Nano 8GB

The "vendor SDK vs hacker stack" rule applies equally to Orin Nano — but NVIDIA is already much closer to tinygrad's philosophy than Qualcomm is, which changes the dynamics significantly.

### Official Path vs Open Stack on Orin

NVIDIA's official path is **TensorRT + CUDA/cuDNN**, which — like SNPE — is designed for stability across models and customers, not for one project's absolute maximum performance. The critical difference: **NVIDIA already exposes very low-level, well-documented CUDA and tensor core APIs**. An open stack (tinygrad, PyTorch custom kernels, Triton) can get very close to or even beat TensorRT on specific workloads by hand-tuning kernels, fusion, and memory layout.

### Why Orin Feels Better Than Snapdragon

| Aspect | Snapdragon 845 (Adreno) | Jetson Orin Nano 8GB (Ampere) |
|--------|------------------------|-------------------------------|
| **Tooling** | Opaque CL/Vulkan/HTP stack, missing docs | Full CUDA toolchain, Nsight profilers, stable ISA view |
| **Hardware docs** | Most useful details under NDA | Tensor core layout, warp scheduling publicly documented |
| **Optimization path** | Reverse-engineer tiling/cache behavior | Hand-tune matmuls, fused convs, tensor-core paths directly |
| **Profiler** | Limited, vendor-gated | Nsight Systems + Nsight Compute expose cycle-level detail |
| **Ceiling** | Hit hardware limits quickly without inside docs | Much higher — you're not fighting the platform |
| **tinygrad backend** | `DEVICE=QCOM` — reverse-engineered | `DEVICE=NV` — first-class CUDA path |

On Snapdragon you fight opaque stacks; on Orin you fight the actual math — which is a much better place to be.

### Practical Takeaway for ADAS Development

```
Qualcomm 845 with tinygrad:
  2× faster than SNPE
  Achieved by: undocumented texture/tiling tricks, reverse-engineered cache behavior
  Cost: brittle, may break on SDK updates

Orin Nano 8GB with tinygrad (DEVICE=NV):
  Can beat generic TensorRT on your exact ADAS models
  Achieved by: custom kernel fusion, tensor-core paths, graph-specific scheduling
  Cost: kernel writing effort — but no reverse-engineering needed
```

If you're willing to write or tune kernels, **Orin Nano 8GB is an excellent tinygrad target**. The same principle applies — a small, ruthless open stack can beat the generic TensorRT path for your exact models — but NVIDIA gives you the tools and visibility to exploit it, so you spend more time optimizing and less time fighting the platform.

### Where the Performance Wins Come From on Orin

| Technique | Generic TensorRT | tinygrad / custom | Win |
|-----------|-----------------|-------------------|-----|
| Kernel fusion | Layer-by-layer (conservative) | Cross-op fusion via lazy scheduler | Less memory bandwidth |
| Tensor core layout | Auto (may not match your shape) | Hand-pick `m×n×k` tile sizes | Better utilization |
| Memory layout | NCHW/NHWC auto-selection | Choose per-layer for cache locality | Fewer stalls |
| Graph scheduling | Fixed TRT build-time plan | Dynamic lazy graph, reorder at runtime | Better batching |
| DLA offload | Manual, coarse-grained | Can slice ops more finely | Better power/perf |

---

## Supported Devices

Tinygrad supports multiple backends:
- **NV/CUDA**: NVIDIA GPUs
- **AMD**: RDNA2+ GPUs
- **METAL**: Apple M1+ devices
- **QCOM**: Qualcomm 6xx series GPUs
- **OpenCL**: Any OpenCL 2.0 device
- **CPU**: Fallback using clang/LLVM
- **WEBGPU**: Browser-based via Dawn

## How Tinygrad Compares to PyTorch

### Similar
- Eager Tensor API
- Autograd (automatic differentiation)
- Optimizers (SGD, Adam, etc.)
- Basic datasets and layers
- You can write familiar training loops

### Unlike PyTorch
- **The entire compiler and IR are visible and hackable**
- Everything is in Python (no hidden C++/CUDA)
- Lazy evaluation by default
- Simpler, more transparent architecture
- Easier to add custom backends

## Community Tutorials (tinygrad-notes)

Prerequisite knowledge before contributing. [GitHub](https://github.com/mesozoic-egg/tinygrad-notes) · [Website](https://mesozoic-egg.github.io/tinygrad-notes/)

| Topic | Description |
|-------|-------------|
| Introduction | Read first |
| JIT explained | Just-in-time compilation |
| Shapetracker explained | Shape and stride tracking |
| Convolution and arange | The trick in conv/arange |
| BEAM search | Kernel optimization |
| Matrix multiplication | The trick in matmul |
| VIZ=1 | Visualizing graph rewrite |
| Pattern matcher | Rewrite rules |
| Memoryview | Buffer views |
| Operator fusion | Fusing ops |
| UOp is singleton | IR design |
| LOP3 (PTX/SASS) | GPU instruction |

## The Tinybox

Tiny corp sells high-performance AI workstations:
- **Red v2**: 4x AMD 9070XT, $12,000
- **Green v2**: 4x RTX PRO 6000, $60,000
- **Pro v2**: 8x RTX 5090, $60,000

## Status

Currently in alpha. Will leave alpha when it can reproduce common papers 2x faster than PyTorch on 1 NVIDIA GPU.

## Learning Resources

- [Quickstart Guide](https://tinygrad.github.io/tinygrad/quickstart/)
- [MNIST Tutorial](https://docs.tinygrad.org/mnist/)
- [GitHub Examples](https://github.com/tinygrad/tinygrad/tree/main/examples)
- [Runtime Documentation](https://docs.tinygrad.org/runtime/)
- See [internals.md](internals.md) for hacking the compiler, IR, and scheduler
