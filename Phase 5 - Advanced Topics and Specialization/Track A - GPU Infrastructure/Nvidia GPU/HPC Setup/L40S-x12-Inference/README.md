# L40S x12 — Inference Deep Dive

The NVIDIA L40S is a PCIe-based GPU designed for AI inference, graphics, and enterprise workloads. Unlike H100/H200 SXM, it uses GDDR6 memory and connects via PCIe — making it more cost-effective for inference-heavy deployments where the extreme bandwidth of HBM is not the bottleneck.

## System Snapshot

| Property | Value |
|---|---|
| GPU | NVIDIA L40S |
| Count | 12 |
| Memory per GPU | 48 GB GDDR6 |
| Total GPU Memory | 576 GB |
| Memory Bandwidth | 864 GB/s per GPU |
| FP8 Tensor Core TFLOPS | ~733 TFLOPS per GPU |
| BF16 / FP16 TFLOPS | ~366 TFLOPS per GPU (sparse) |
| TF32 TFLOPS | ~183 TFLOPS per GPU |
| GPU Interconnect | PCIe 4.0 x16 (no NVLink) |
| Form Factor | PCIe full-height, dual-slot |
| TDP | 350 W per GPU |

## L40S vs H200: When to Choose L40S

| Factor | L40S x12 | H200 x8 |
|---|---|---|
| Total memory | 576 GB GDDR6 | 1,128 GB HBM3e |
| Memory bandwidth | 10.4 TB/s total | 38.4 TB/s total |
| GPU interconnect | PCIe (no NVLink) | NVLink 4.0 900 GB/s |
| Cost (approx) | ~$60K | ~$400K+ |
| Best for | Cost-efficient inference | Training + large model inference |
| Power/rack | ~4,200 W (12 GPUs) | ~5,600 W (8 GPUs) |
| Max single model | ~70B (multi-GPU) | ~405B (multi-GPU) |

**Choose L40S when:** inference throughput matters more than model size, cost is constrained, or you're running multiple smaller models simultaneously.

## Topic Index

| # | Topic | Description |
|---|---|---|
| 01 | [Hardware Architecture](./01-Hardware-Architecture.md) | Ada Lovelace die, GDDR6, PCIe topology, NVLink absence |
| 02 | [Inference Optimization](./02-Inference-Optimization.md) | vLLM, TRT-LLM, quantization, batching strategies |
| 03 | [Multi-GPU Strategy](./03-Multi-GPU-Strategy.md) | PCIe-constrained parallelism, pipeline vs tensor parallel |
| 04 | [Deployment Guide](./04-Deployment-Guide.md) | Multi-instance deployment, model sharding, production setup |
| 05 | [Benchmarks](./05-Benchmarks.md) | Throughput targets, latency baselines, cost/perf comparison |

## Quick Navigation

- **Serving a 7B model?** → [04-Deployment-Guide](./04-Deployment-Guide.md) — single GPU inference
- **Serving a 70B model?** → [03-Multi-GPU-Strategy](./03-Multi-GPU-Strategy.md) — 2 GPUs with TP=2
- **Maximizing throughput?** → [02-Inference-Optimization](./02-Inference-Optimization.md)
- **Performance bottleneck?** → [05-Benchmarks](./05-Benchmarks.md)
