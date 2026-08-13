# 8x H200 GPU — Training & Inference Deep Dive

The NVIDIA H200 SXM5 is the current flagship GPU for AI workloads, featuring 141 GB HBM3e memory at 4.8 TB/s bandwidth. An 8-GPU SXM node with NVLink 4.0/NVSwitch is the industry-standard building block for large model training and high-throughput inference.

## System Snapshot

| Property | Value |
|---|---|
| GPU | NVIDIA H200 SXM5 |
| Count | 8 |
| Memory per GPU | 141 GB HBM3e |
| Total GPU Memory | 1,128 GB (1.1 TB) |
| Memory Bandwidth | 4.8 TB/s per GPU |
| FP8 Tensor Core TFLOPS | ~3,958 TFLOPS per GPU |
| BF16 Tensor Core TFLOPS | ~1,979 TFLOPS per GPU |
| GPU Interconnect | NVLink 4.0 (900 GB/s bidirectional) |
| NVSwitch | 3rd Gen (full mesh) |
| Host Interconnect | PCIe 5.0 / CXL |
| Form Factor | SXM5 baseboard |

## Topic Index

| # | Topic | Description |
|---|---|---|
| 01 | [Hardware Architecture](./01-Hardware-Architecture.md) | Chip design, HBM3e, NVLink 4.0, NVSwitch topology |
| 02 | [Training Setup](./02-Training-Setup.md) | Distributed training, 3D parallelism, FSDP, DeepSpeed |
| 03 | [Inference Setup](./03-Inference-Setup.md) | Tensor parallel inference, vLLM, TensorRT-LLM |
| 04 | [Memory Management](./04-Memory-Management.md) | KV cache, paged attention, memory pooling |
| 05 | [Performance Optimization](./05-Performance-Optimization.md) | Profiling, roofline, kernel tuning, CUDA Graphs |
| 06 | [Benchmarks & Validation](./06-Benchmarks-and-Validation.md) | MFU, MBU, latency, throughput targets |

## Quick Navigation

- **Training a 70B model?** → Start with [02-Training-Setup](./02-Training-Setup.md)
- **Inference serving?** → Start with [03-Inference-Setup](./03-Inference-Setup.md)
- **GPU memory OOM?** → See [04-Memory-Management](./04-Memory-Management.md)
- **Low GPU utilization?** → See [05-Performance-Optimization](./05-Performance-Optimization.md)
