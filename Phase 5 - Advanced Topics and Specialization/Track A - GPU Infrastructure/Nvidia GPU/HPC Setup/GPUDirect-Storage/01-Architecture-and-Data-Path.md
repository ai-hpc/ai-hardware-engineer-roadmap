# 01 — Follow the payload and the control work

GPUDirect Storage (GDS) can avoid staging file payloads in CPU memory before they
reach a GPU. Learn to draw the actual path before predicting a speedup.

**Prerequisites:** files, memory, and bandwidth units. **Outcome:** a path diagram
and a bounded bandwidth estimate for a robot-data or model-loading workload.
Allow 45–90 minutes. Hardware execution is not required for the first exercise.

## Two paths for the same bytes

```text
Staged payload: storage -> host-memory buffer -> GPU buffer
Direct payload: supported storage/transport path -> GPU buffer
Control: host code opens files, submits requests, and handles completion
```

The [NVIDIA overview](https://docs.nvidia.com/gpudirect-storage/overview-guide/index.html)
explains the distinction. “Direct” describes the payload path; it does not mean
that the CPU disappears. In a staged path, DMA can perform transfers involving
host memory. It is incorrect to assume every byte must pass through CPU
instructions or the L3 cache.

Avoid counting PCIe traffic by the number of boxes in the drawing. Each physical
link carries the traffic that actually traverses it. A direct path may remove
host-memory traffic without halving traffic on every link.

## A small bandwidth calculation

Consider these **hypothetical capacities**, chosen for an arithmetic exercise:

- a storage device sustains 6 GB/s on the selected access pattern;
- a shared upstream path can carry 10 GB/s;
- the destination path can carry 20 GB/s.

One reader cannot exceed `min(6, 10, 20) = 6 GB/s`. Four storage devices behind the
same 10 GB/s upstream path do not create 24 GB/s of usable throughput through it.
The shared link limits their aggregate. At 6 GB/s, moving a decimal 1 GB of payload
has a transfer-only lower bound of about 0.167 seconds, before setup and other
work. This is a calculation, not a benchmark.

**Guided variation:** add 100 ms of required CPU image decoding per GB. Can GDS
remove that time? Not by changing storage transfer alone. You need a different
decode pipeline or an experiment showing overlap. The image still has to be
interpreted before the model can use it.

## Local storage, remote storage, and GPU communication

| Mechanism | What it connects | What still needs verification |
|---|---|---|
| Local file I/O with GDS | A supported local storage/filesystem path and GPU memory | Mount, driver, topology, API path, payload correctness |
| Remote file/block I/O with GDS | A supported network storage path and GPU memory | Transport, NIC, filesystem, network, and selected release support |
| GPU-to-GPU peer access / GPUDirect RDMA | GPU memory and peer devices | Communication topology and peer/transport support |

GPU-to-GPU communication is related infrastructure, not a third file-storage
API mode. An NCCL all-reduce is not a GDS read. Likewise, a remote namespace that
appears under `/dev/nvme...` still has network latency and failure dependencies.

## Topology changes the hypothesis

Draw GPUs, storage, NICs, PCIe switches/root complexes, CPUs, and inter-socket
links. Mark shared resources and distinguish physical reachability from supported
peer transfer. Sharing a NUMA node does not by itself prove a direct path or an
optimal configuration. Firmware, address translation, access controls, switches,
and software mode can change support.

Use the official [GDS design guide](https://docs.nvidia.com/gpudirect-storage/design-guide/index.html)
for the selected architecture. Do not assume a fixed cross-socket penalty or
apply one historical system's measured bandwidth to a different topology.

## Fallback is part of the experiment

Depending on configuration and support, cuFile can use a compatibility path that
stages data through host memory. A successful `cuFileRead` establishes neither a
speedup nor direct DMA. Inspect path diagnostics and compare the same integrity-
checked workload under declared modes. A disabled fallback can cause a visible
failure on an unsupported path; an enabled fallback can preserve functionality.

The software lesson explains the current release-dependent mode distinction:
[Establish a supported path](03-Software-Stack.md).

## Practice: a robot-data loader

Draw the complete path for recorded camera data:

```text
file -> read -> decode -> resize/normalize -> batch -> model -> evaluation
```

For each stage, label CPU or GPU work, input/output format, allocated bytes, and
completion dependency. Choose a 1 GB hypothetical input and estimate each stage's
bound from explicitly stated capacities. Then identify the measurements needed
to replace those estimates.

**Self-check with answer:** if storage already finishes while a longer computation
runs, a faster isolated read may not reduce the steady-state step interval. The
critical path and available overlap determine the benefit. First draw the timeline,
then test it.

**Exit artifact:** path diagram, shared-link budget, and one falsifiable bottleneck
hypothesis. Continue to [hardware/topology inspection](02-Hardware-Setup.md).
