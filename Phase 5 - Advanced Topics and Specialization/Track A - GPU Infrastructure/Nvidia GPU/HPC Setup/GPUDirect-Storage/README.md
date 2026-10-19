# GPUDirect Storage (GDS) — Deep Dive

GPUDirect Storage can move file payloads between storage and GPU memory without staging those payloads through CPU DRAM. The CPU still opens files, submits cuFile operations, handles control work, and may participate in fallback paths. Successful `cuFileRead` is not proof that direct DMA was used. [NVIDIA GDS overview](https://docs.nvidia.com/gpudirect-storage/overview-guide/).

## Follow the payload and the control path

```text
Traditional payload: storage -> CPU bounce buffer -> GPU memory
Direct payload:      storage -> supported PCIe/NIC path -> GPU memory
Control in both:     host application -> file/API/driver setup and completion
Fallback:            cuFile may stage through CPU memory on unsupported paths
```

Before tuning, record filesystem, mount, kernel, GPU, NIC, cuFile version, and selected I/O mode. Compare equal-size aligned reads with direct and compatibility paths, including CPU usage and end-to-end loader time. The next stage may still require CPU decoding or augmentation, which can erase the I/O gain.

## Start with one correct record

Follow the lessons in order: payload path → topology → supported software → API
contracts → measurement → shared-storage design. The first three need only basic
Linux and bandwidth arithmetic. The portable lab works with Python 3.12 and no GPU:

```bash
python3 'Phase 5 - Advanced Topics and Specialization/Track A - GPU Infrastructure/Nvidia GPU/HPC Setup/GPUDirect-Storage/labs/io_contract.py'
```

Run from the repository root. Inspect the chunk offsets, exact payload comparison,
truncation rejection, and ownership failure before porting to cuFile. The lab uses
CPU file I/O and a synthetic ownership model; it does not measure GDS performance.

**Reviewed October 18, 2026:** removed invented daemon/API examples, unsupported
throughput targets, universal alignment rules, and obsolete hardware assumptions.
The hardware path remains an upstream-documented exercise requiring the selected
release's [support matrix](https://docs.nvidia.com/gpudirect-storage/overview-guide/index.html).
Save the exact driver/CUDA/cuFile/filesystem configuration and observed transfer
mode. Historical product examples are not a current installation recipe.

## Topic Index

| # | Topic | Description |
|---|---|---|
| 01 | [Architecture & Data Path](./01-Architecture-and-Data-Path.md) | How GDS works, PCIe topology, NUMA pinning |
| 02 | [Hardware Setup & Configuration](./02-Hardware-Setup.md) | Inspect devices, shared links, negotiated topology, and support |
| 03 | [Software Stack & Installation](./03-Software-Stack.md) | Choose a matched stack and distinguish capability from observed path |
| 04 | [libcufile Programming API](./04-libcufile-API.md) | cuFile API — read, write, register buffers, async I/O |
| 05 | [Performance Tuning](./05-Performance-Tuning.md) | Alignment, buffer registration, multi-stream, benchmarking |
| 06 | [Disaggregated Storage and Sizing](./06-Disaggregated-Storage.md) | Shared bandwidth, capacity sizing, contention, and failure domains |

## Quick Navigation

- **First time setting up GDS?** → [03-Software-Stack](./03-Software-Stack.md)
- **Writing GDS-enabled code?** → [04-libcufile-API](./04-libcufile-API.md)
- **Storage bandwidth too low?** → [05-Performance-Tuning](./05-Performance-Tuning.md)
- **Disaggregated NVMe-oF?** → [06-Disaggregated-Storage](./06-Disaggregated-Storage.md)

## Practice and assessment

**Entry diagnostic (prerequisite check):** Trace storage, host memory, and GPU buffers for both direct and staged I/O. If this is unfamiliar, use the prerequisite material above or the [roadmap diagnostics](../../../../../Learning-Guide.md#diagnose-before-reading) before starting the lab.

1. **Required CPU baseline:** run the portable contract lab, explain all four chunk
   ranges, and reproduce truncation and premature-buffer-reuse failures.
2. **Guided CPU variation:** change file length and chunk size. Predict the final
   chunk before running and verify every payload byte after reconstruction.
3. **Independent CPU transfer:** add a record header and two tensors as described
   in [the API lesson](04-libcufile-API.md#practice-and-answer-guidance). Check each
   tensor's shape, byte range, and first/last value independently.
4. **Optional supported-GPU extension:** compare direct and staged reads with
   identical files and declared cache conditions, then vary transfer size. Use
   observed path evidence before interpreting performance.

**CPU evidence to submit:** exact command, Python version, raw JSON, chunk-range
prediction, payload comparison, and one caught failure. State that the ownership
events are modeled and cuFile has not been executed. For the optional GPU work,
add the exact stack, I/O path evidence, effective bandwidth, CPU use, transfer
latency, alignment, and cache state. CPU completion does not establish GPU support
or performance; hardware access is not required to complete the CPU route.

**Self-check:** Which observation would disprove your explanation of the result? Identify a competing explanation and the smallest experiment that separates them. A passing answer names a concrete predicted difference, not only another tool to try.

**Exit criteria:** reproduce the baseline, demonstrate the negative/variation check, and meet the [shared completion rubric](../../../../../Learning-Guide.md#completion-rubric). Use the [lesson cycle](../../../../../Learning-Guide.md#a-repeatable-lesson-cycle) for retrieval practice and spaced review.
