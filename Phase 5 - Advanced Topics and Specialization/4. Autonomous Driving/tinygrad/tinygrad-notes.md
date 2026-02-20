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

📖 See detailed guide: `ops/elementwise/`

### 2. ReduceOps
Operate on one tensor and return a smaller tensor
- Examples: SUM, MAX

### 3. MovementOps
Virtual ops that move data around, copy-free with ShapeTracker
- Examples: RESHAPE, PERMUTE, EXPAND, etc.

**Note:** No primitive operators for CONV or MATMUL - these are built from basic operations!

## Key Features

- **Extreme simplicity** - Easiest framework to add new accelerators to
- **Lazy evaluation** - All tensors are lazy, enabling aggressive operation fusion
- **Custom kernel compilation** - Compiles a custom kernel for every operation
- **Full training support** - Forward and backward passes with autodiff
- **Hackable** - Entire compiler and IR are visible and modifiable
- **Multi-backend** - Supports NVIDIA, AMD, and other accelerators

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

# Lazy evaluation - computation happens when .realize() is called
result.realize()
```

## Real-World Usage

Tinygrad is used in [Openpilot](https://github.com/commaai/openpilot) (comma.ai ADAS) to run the driving model on Snapdragon 845 GPU, replacing SNPE with:
- Better performance
- ONNX file loading support
- Training support
- Attention mechanism support

## Supported Devices

Tinygrad supports multiple backends:
- **NV/CUDA**: NVIDIA GPUs
- **AMD**: RDNA2+ GPUs
- **METAL**: Apple M1+ devices
- **QCOM**: Qualcomm 6xx series GPUs
- **OpenCL**: Any OpenCL 2.0 device
- **CPU**: Fallback using clang/LLVM
- **WEBGPU**: Browser-based via Dawn

## Learning Resources

- [Quickstart Guide](https://tinygrad.github.io/tinygrad/quickstart/)
- [MNIST Tutorial](https://docs.tinygrad.org/mnist/)
- [GitHub Examples](https://github.com/tinygrad/tinygrad/tree/main/examples)
- [Runtime Documentation](https://docs.tinygrad.org/runtime/)

### Community Tutorials (tinygrad-notes)

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

Excellent performance per dollar for deep learning workloads.

## How Tinygrad Compares to PyTorch

### ✅ Similar
- Eager Tensor API
- Autograd (automatic differentiation)
- Optimizers (SGD, Adam, etc.)
- Basic datasets and layers
- You can write familiar training loops

### 🔁 Unlike PyTorch
- **The entire compiler and IR are visible and hackable**
- Everything is in Python (no hidden C++/CUDA)
- Lazy evaluation by default
- Simpler, more transparent architecture
- Easier to add custom backends

## Status

Currently in alpha. Will leave alpha when it can reproduce common papers 2x faster than PyTorch on 1 NVIDIA GPU.

## Hacking Tinygrad

See `hacking-tinygrad.md` for detailed examples of:
- Inspecting the computation graph (`tensor.uop`, `tensor.shape`)
- Viewing the IR and optimization stages (`DEBUG=4`)
- Creating custom operations (compose primitives)
- Exploring the scheduler (`tensor.schedule()`)
- Adding custom backends

Run `hands-on-example.py` to see tinygrad's internals in action!

## Notes

- API similar to PyTorch but simpler and more refined
- Less stable during alpha phase
- Great for learning how deep learning frameworks work
- Ideal for adding custom accelerator support
- Perfect for understanding what happens "under the hood"
