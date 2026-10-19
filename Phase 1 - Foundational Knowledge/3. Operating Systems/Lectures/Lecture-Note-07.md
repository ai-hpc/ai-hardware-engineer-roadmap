# Lecture Note 07: DMA Ownership and NUMA Placement

Consolidates [Lecture 15](Lecture-15.md) and [Lecture 16](Lecture-16.md).

## Coherence, ordering, completion, and lifetime

These are four different questions:

| Question | Example failure |
|---|---|
| Are the latest bytes visible? | CPU reads a stale cache line after device output |
| Were writes published in the required order? | Device sees descriptor-valid before its address/length |
| Has the producer finished? | GPU reads while the camera is still writing |
| Does the storage remain valid? | Timeout handler frees memory before late DMA |

A coherent allocation removes the corresponding streaming cache-maintenance requirement; it does not remove ordering or lifetime rules. A streaming mapping uses the DMA API's map/unmap or explicit synchronization transitions, including on coherent hardware. Select the direction from the mapped device's point of view.

**Question:** Six scatterlist entries map to four DMA segments. Which count programs hardware, and which count unmaps?

**Answer:** Use the four mapped segments for the device and the original six entries for unmapping, according to the DMA API. A zero mapping result is a failure, not permission to submit.

## Shared buffers and device isolation

DMA-BUF exporters own storage; importers attach/map it for their own devices. An fd is not a pointer or universal DMA address. Transfer it using `SCM_RIGHTS` across processes. Compatibility, format/planes/strides, producer completion, cache synchronization, and reuse must all be defined.

An IOMMU group represents the minimum isolation unit for assignment under the platform's capabilities. It does not simply mean devices share one translation engine. An IOMMU does not prevent arbitrary CPU writes by kernel code. VFIO is a kernel-mediated device-access framework, not the absence of a kernel driver.

Jetson's integrated CPU/GPU share physical DRAM; that does not make every CPU buffer accessible to CUDA or immediately coherent with a camera. Discrete GPUs have different movement paths. Check the supported allocator/runtime APIs instead of treating `cudaMalloc` as an exportable camera buffer.

## NUMA means topology-dependent cost

A machine can have multiple NUMA nodes per socket, memory-only nodes, or heterogeneous memory tiers. Remote access often has different latency/bandwidth, but “always twice the latency and half the bandwidth” is not a law. The reported distance matrix is relative policy/topology information, not measured nanoseconds.

Inspect `numactl --hardware`, `/sys/devices/system/node/`, and process placement where available. Containers may restrict or virtualize the view. `set_mempolicy()` affects the calling thread's policy; `mbind()` applies policy to a range. Existing pages do not necessarily move just because a policy changes.

Under the relevant default/local policy, first allocation faults influence placement, subject to allowed nodes, fallback, shared mappings, and other rules. Initializing a buffer on one node before workers on another use it can create expensive remote traffic; measure rather than assuming every access is remote or equally costly.

**Question:** Does `numa_alloc_onnode()` return prove all pages reside on that node?

**Answer:** No. Check errors/warnings and effective placement; policy enforcement, actual faults, permissions, and migration matter.

## A controlled placement experiment

Hold workload, allocation size, thread count, and access pattern constant. Compare a supported local placement against another explicit placement, recording actual pages/CPUs and latency/bandwidth. Change automatic NUMA balancing only in a controlled experiment with permission to modify that host policy; disabling it everywhere is not a universal inference optimization.

For GPU jobs, include CPU/GPU/interconnect topology and the memory-access pattern. Binding everything to one node can help locality or overload that node; multi-GPU communication can favor another arrangement.

**Question:** Does a better NUMA placement prove a real-time deadline?

**Answer:** No. It improves one measured cost under specified conditions. Blocking, scheduling, interrupts, device work, and other interference remain.
