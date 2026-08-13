# CUDA Advanced Optimization — Deep Dive

Five techniques used by GPU engineers at NVIDIA, OpenAI, and Meta to push inference and HPC kernels to hardware limits. These go far beyond basic CUDA programming — they are what separates a "working" GPU kernel from a "production" one.

## Why These Techniques Matter

```
Naive CUDA kernel:         ~30% of hardware peak  (common)
With kernel fusion:        ~50% of hardware peak
With CUDA Graphs:          +10-30% latency reduction
With cooperative groups:   enables algorithms impossible without them
With persistent kernels:   near-zero kernel launch overhead
With warp specialization:  ~70-85% of hardware peak  (elite)
```

Every LLM inference engine (TensorRT-LLM, vLLM, FasterTransformer, FlashAttention) uses all five.

## Topic Index

| # | Topic | What It Solves |
|---|---|---|
| 01 | [CUDA Graphs](./01-CUDA-Graphs.md) | CPU launch overhead kills latency at small batch sizes |
| 02 | [Cooperative Groups](./02-Cooperative-Groups.md) | Thread block boundary limits synchronization flexibility |
| 03 | [Persistent Kernels](./03-Persistent-Kernels.md) | Repeated kernel launches waste SM setup time |
| 04 | [Kernel Fusion](./04-Kernel-Fusion.md) | Separate kernels waste HBM bandwidth on intermediate results |
| 05 | [Warp Specialization](./05-Warp-Specialization.md) | Compute and memory latency are not overlapped inside a kernel |

## How They Relate

```
CUDA Graphs          → reduces CPU↔GPU interface overhead
Cooperative Groups   → enables flexible intra-kernel synchronization
Persistent Kernels   → eliminates kernel launch overhead entirely
Kernel Fusion        → reduces HBM round-trips between operations
Warp Specialization  → overlaps compute and memory within a single kernel

Combined (e.g. FlashAttention-3):
  Persistent kernel + warp specialization + cooperative groups
  → 90%+ of H200 BF16 peak on attention kernels
```

## Quick Navigation

- **LLM inference latency too high?** → [01-CUDA-Graphs](./01-CUDA-Graphs.md)
- **Writing a custom reduction/scan?** → [02-Cooperative-Groups](./02-Cooperative-Groups.md)
- **Kernel launch overhead visible in profile?** → [03-Persistent-Kernels](./03-Persistent-Kernels.md)
- **GPU memory bandwidth bottleneck?** → [04-Kernel-Fusion](./04-Kernel-Fusion.md)
- **Want to write FlashAttention-style kernels?** → [05-Warp-Specialization](./05-Warp-Specialization.md)
