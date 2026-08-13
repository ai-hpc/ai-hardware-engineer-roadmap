# NCCL Deep Dive — NVIDIA Collective Communications Library

NCCL (pronounced "Nickel") is the **core communication engine** that makes multi-GPU AI training possible. Every time PyTorch runs `dist.all_reduce()`, every time DeepSpeed syncs gradients, every time Megatron-LM does tensor parallelism — NCCL is executing the actual GPU-to-GPU data movement.

Understanding NCCL at depth means understanding **why your training runs fast or slow**, and how to fix it when it isn't.

## What NCCL Solves

```
Naive multi-GPU synchronization (without NCCL):
  GPU0 copies gradient → CPU RAM
  CPU reduces all gradients
  CPU copies result back to each GPU

  Bottleneck: PCIe bandwidth (32 GB/s) × 2 transfers × 8 GPUs
  Time for 1 GB gradient sync: ~500 ms

NCCL approach:
  GPU-to-GPU direct via NVLink (900 GB/s bidirectional)
  No CPU involvement, no PCIe crossing
  Time for 1 GB gradient sync: ~2 ms

  → 250× faster
```

## Topic Index

| # | Topic | Key Questions Answered |
|---|---|---|
| 01 | [Fundamentals](./01-Fundamentals.md) | What are collectives? What does each operation do? |
| 02 | [Algorithms & Bandwidth](./02-Algorithms-and-Bandwidth.md) | How does Ring AllReduce work? How does NCCL hit 900 GB/s? |
| 03 | [Framework Integration](./03-Framework-Integration.md) | How do PyTorch/DeepSpeed/Megatron use NCCL internally? |
| 04 | [Configuration & Tuning](./04-Configuration-and-Tuning.md) | Which env vars matter? How to tune for H200 vs PCIe? |
| 05 | [Multi-Node Clusters](./05-Multi-Node-Clusters.md) | InfiniBand, SHARP offload, hierarchical AllReduce |
| 06 | [Debugging](./06-Debugging.md) | Hangs, timeouts, topology mismatches — how to fix them |
| 07 | [Trillion-Parameter Scale](./07-Trillion-Parameter-Scale.md) | How NCCL + tensor/pipeline parallelism trains 1T+ models |

## Quick Reference

- **Training slow, GPUs idle?** → [04-Configuration-and-Tuning](./04-Configuration-and-Tuning.md)
- **NCCL hang / timeout?** → [06-Debugging](./06-Debugging.md)
- **Understanding Ring AllReduce math?** → [02-Algorithms-and-Bandwidth](./02-Algorithms-and-Bandwidth.md)
- **Building multi-node cluster?** → [05-Multi-Node-Clusters](./05-Multi-Node-Clusters.md)
- **Training 70B+ models?** → [07-Trillion-Parameter-Scale](./07-Trillion-Parameter-Scale.md)
