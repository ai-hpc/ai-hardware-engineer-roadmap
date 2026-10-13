# GPUDirect Storage (GDS) — Deep Dive

GPUDirect Storage (GDS) eliminates the CPU from the GPU-to-storage data path. Instead of data traveling GPU → PCIe → CPU RAM → PCIe → NVMe, GDS creates a **direct DMA path** between GPU HBM and NVMe/network storage — the CPU complex is bypassed entirely.

## Why GDS Exists

```
Without GDS (traditional path):
  NVMe → PCIe → CPU DRAM (bounce buffer) → PCIe → GPU HBM
  CPU must be awake and involved for every I/O
  CPU DRAM becomes the bottleneck (~50 GB/s DRAM bandwidth shared)

With GDS (direct path):
  NVMe → PCIe → GPU HBM  (direct DMA)
  No CPU bounce buffer
  No CPU involvement in the data path
  Limited only by PCIe bandwidth and NVMe throughput
```

## Reference Configuration (Western Digital Technical Brief)

This section is based on the WD OpenFlex Data24 + NVIDIA GDS validation setup, which represents a real production GDS deployment:

| Component | Specification |
|---|---|
| CPU | Dual Intel Xeon Gold 6348, 26C @ 2.60 GHz |
| RAM | 512 GiB |
| GPU | 4× NVIDIA A100 80 GB PCIe |
| NIC | 6× ConnectX-7 (CX-6 also tested) |
| CUDA | 12.2.1 |
| GDS | 2.17.3 |
| libcufile | 1.7.1.12 |
| OS | RHEL 9: 5.14.0-70.70.1.el9_0 |
| OFED | Mellanox OFED 5.8-3.0.7.0 |
| Nvidia Driver | 535.86.10 |
| Ethernet Switch | NVIDIA SN3700, 32-port 200 Gb (Spectrum 2) |
| Storage | WD OpenFlex Data24 3200 Series |
| Storage Bandwidth | 75 GB/s theoretical (6 × 100 Gb/s frontend) |

## Topic Index

| # | Topic | Description |
|---|---|---|
| 01 | [Architecture & Data Path](./01-Architecture-and-Data-Path.md) | How GDS works, PCIe topology, NUMA pinning |
| 02 | [Hardware Setup & Configuration](./02-Hardware-Setup.md) | Reference config from WD brief, PCIe layout, NIC placement |
| 03 | [Software Stack & Installation](./03-Software-Stack.md) | GDS install, libcufile, verification, version requirements |
| 04 | [libcufile Programming API](./04-libcufile-API.md) | cuFile API — read, write, register buffers, async I/O |
| 05 | [Performance Tuning](./05-Performance-Tuning.md) | Alignment, buffer registration, multi-stream, benchmarking |
| 06 | [Disaggregated Storage (OpenFlex + RapidFlex)](./06-Disaggregated-Storage.md) | NVMe-oF over RDMA/RoCE, WD OpenFlex, linear scale-out |

## Quick Navigation

- **First time setting up GDS?** → [03-Software-Stack](./03-Software-Stack.md)
- **Writing GDS-enabled code?** → [04-libcufile-API](./04-libcufile-API.md)
- **Storage bandwidth too low?** → [05-Performance-Tuning](./05-Performance-Tuning.md)
- **Disaggregated NVMe-oF?** → [06-Disaggregated-Storage](./06-Disaggregated-Storage.md)

## Practice and assessment

**Entry diagnostic (prerequisite check):** Trace storage, host memory, and GPU buffers for both direct and staged I/O. If this is unfamiliar, use the prerequisite material above or the [roadmap diagnostics](../../../../../Learning-Guide.md#diagnose-before-reading) before starting the lab.

1. **Baseline lab:** Compare supported direct and staged reads with identical files and cache conditions.
2. **Guided variation:** Vary transfer size and separate cold from warm reads.
3. **Independent transfer:** repeat the method on a new input or configuration and explain where your original assumptions stop applying.

**Evidence to submit:** Effective bandwidth, CPU use, transfer latency, alignment, cache state, and data integrity. Keep raw results and reproduction steps; distinguish measured results from estimates or work not run.

**Self-check:** Which observation would disprove your explanation of the result? Identify a competing explanation and the smallest experiment that separates them. A passing answer names a concrete predicted difference, not only another tool to try.

**Exit criteria:** reproduce the baseline, demonstrate the negative/variation check, and meet the [shared completion rubric](../../../../../Learning-Guide.md#completion-rubric). Use the [lesson cycle](../../../../../Learning-Guide.md#a-repeatable-lesson-cycle) for retrieval practice and spaced review.
