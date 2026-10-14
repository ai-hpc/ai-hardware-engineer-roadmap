# Chapter 1: Blackwell Architecture for Inference

**Previous:** [Course guide](README.md) · **Next:** [Low-precision numerics](02-FP4-Numerics-Transformer-Engine.md)

## Start with three different system boundaries

A Blackwell GPU, a Grace Blackwell Superchip, and an NVL72 system are different units of deployment. A learner must distinguish them before estimating memory capacity or choosing tensor parallelism.

| Boundary | What software must account for | What it does not imply |
|---|---|---|
| One B200 GPU | One CUDA device built from two connected dies; query available memory and SM properties | Two independently assignable CUDA GPUs inside the package |
| HGX/DGX B200 node | Multiple CUDA GPUs connected through an NVLink switching fabric | One flat HBM allocation pool |
| GB200 Superchip | A Grace CPU coupled to two Blackwell GPUs through NVLink-C2C | HBM-speed access to every CPU memory address |
| GB200 NVL72 system | 72 GPUs and 36 Grace CPUs in an NVLink domain | One CUDA device or uniform rack-wide memory latency |

The published DGX B200 configuration has **1,440 GB across eight GPUs** (180 GB per GPU), whereas early architecture presentations used 192 GB figures. GB200 has its own platform specifications. Use the actual SKU and runtime-reported free memory in a deployment budget. Do not convert a decimal GB number directly to the same number of GiB. [DGX B200 user guide](https://docs.nvidia.com/dgx/dgxb200-user-guide/introduction-to-dgxb200.html), [GB200 NVL72 specifications](https://www.nvidia.com/en-us/data-center/gb200-nvl72/).

## What changes at the kernel level

Blackwell SM100 introduces `tcgen05.mma` operations and Tensor Memory (TMEM) for accumulator storage. This differs from copying a Hopper WGMMA kernel and merely changing its target name. Operand layout, descriptor encoding, synchronization, supported types, and TMEM lifetime all matter. [CUTLASS Blackwell functionality](https://docs.nvidia.com/cutlass/latest/media/docs/cpp/blackwell_functionality.html).

Study one supported CUTLASS GEMM first. Draw its path from global memory through shared memory to matrix computation and output. Identify the producer, consumer, barrier, and buffer owner. Then inspect one measured VLA projection shape: a peak-throughput square matrix may not represent batch-one action inference.

Hardware format support is only one requirement. A serving runtime also needs a kernel for the model operation, shape, packing, scaling scheme, and target architecture. “Blackwell” includes different compute-capability targets; a kernel built for SM100 is not automatically an optimal kernel for another Blackwell device.

## Memory bandwidth is a bound, not a token-rate prediction

For a weight-streaming decode step, a first lower-bound estimate is:

`step time >= bytes read from HBM / sustained HBM bandwidth`

Suppose an illustrative packed model reads 40 GB per token and a hypothetical sustained bandwidth is 6 TB/s. The weight read alone takes at least about 6.7 ms. KV reads, attention arithmetic, launches, synchronization, sampling, and queueing add cost; cache reuse can change the bytes read. The reciprocal is an optimistic upper bound under these assumptions, not a benchmark result.

Batching changes the calculation because multiple tokens can reuse weights. Prefill has different matrix dimensions and can become compute limited. Compare achieved bandwidth and arithmetic intensity for each phase rather than predicting a fixed B200-to-H100 speedup from a datasheet ratio.

## Lab: inventory before inference

On an available NVIDIA host, collect:

```bash
nvidia-smi --query-gpu=name,memory.total,driver_version,pci.bus_id --format=csv
nvidia-smi topo -m
```

Use the runtime/device-query tools installed with the pinned CUDA stack to record compute capability, free memory, power limit, and supported peer access. Save the node's CPU/NIC topology, container digest, and runtime version as well. A course document cannot infer these from the product family name.

**Deliverable:** `platform.json` plus a topology diagram labeling HBM, CPU DRAM, NVLink, NVLink-C2C where present, PCIe, and NIC links. For each number state directionality and whether it is a vendor limit or measurement.

**Reasoning check:** why can 8 × 180 GB of HBM fail to fit a model whose total allocated bytes are below 1,440 GB? Explain uneven rank placement, replicated tensors, and per-device workspace. Carry the largest-rank budget into Chapter 3.

## Connection to the physical-AI chip

The transferable idea is **explicit placement and data movement**. A custom SoC also has distinct SRAM, external memory, CPU, and DMA domains. Use the GPU baseline to find which traffic matters before choosing SRAM size or adding a matrix unit. The architecture figure in the [AI-chip guide](../../../../Track%20F%20-%20AI%20Chip%20Design/Guide.md#4-compiler-and-accelerator-co-design-one-executable-vertical-slice) shows where the compiler and Linux driver must agree.
