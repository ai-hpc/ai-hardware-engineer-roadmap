# Lecture 15: DMA, IOMMU & GPU Memory Management

## Overview

Modern AI systems depend on moving large amounts of data — camera frames, model weights, activations — between CPU, GPU, camera ISP, and storage at high speed. CPU copies may consume significant cycles and bandwidth, but their cost must be measured for the actual size and rate. Avoiding a copy is useful only if the replacement path preserves ownership and completion semantics. **Direct Memory Access (DMA)** lets hardware engines transfer payload data without a CPU executing the bulk copy. The core challenge is ensuring the CPU's caches and the device always see consistent data. The mental model for this lecture is a production pipeline: raw sensor data enters memory, gets processed by multiple hardware engines in sequence, and reaches the inference engine — ideally without a single software copy. For an AI hardware engineer, mastering DMA and the IOMMU is essential for designing zero-copy camera-to-inference pipelines, writing correct PCIe device drivers, and understanding why Jetson's unified memory architecture behaves differently from a discrete GPU.

---

## DMA: Direct Memory Access

**DMA** allows a device (NIC, NVMe controller, GPU PCIe DMA engine, camera ISP) to transfer data directly to or from system RAM without the CPU executing the bulk data copy. The CPU programs a **descriptor** specifying source address, destination address, transfer length, and flags; the device executes the transfer autonomously; completion is signaled via an interrupt or a status flag the driver polls.

DMA can overlap transfers with CPU work; actual overlap and throughput depend on the engine, buffers, interconnect, scheduling, and software overhead.

```
Without DMA (CPU-copy path):
Device → [device buffer] → CPU reads → CPU writes → [system RAM]
           (device generates data)    (CPU memcpy)   (destination)
CPU is fully occupied during the entire transfer.

With DMA:
Device → DMA engine → [system RAM]
CPU submits one descriptor, then is free to do other work.
Completion interrupt notifies CPU when transfer is done.

┌──────────────┐   DMA descriptor    ┌─────────────┐
│   Device /   │ ─────────────────>  │  DMA Engine │
│  Controller  │   (src, dst, len)   │             │
│              │ <─────────────────  │             │
│              │   IRQ on complete   └──────┬──────┘
└──────────────┘                            │ bus master
                                            ▼
                                     ┌─────────────┐
                                     │  System RAM │
                                     └─────────────┘
```

---

## Cache Coherency Problem

Modern CPUs cache data in L1/L2/L3. When a device writes to RAM (DMA write), the CPU may hold a **stale cached copy**. When a device reads from RAM (DMA read), the device may read stale data that the CPU has modified in cache but not yet written back. **Two programming models** address this:

```
Cache Coherency Hazard (DMA write, device→CPU):

[Device DMA writes new data to PA 0x1000]
    ↓
[RAM at 0x1000] = new value
    ↓ (but)
[CPU L1/L2 cache for VA→0x1000] = old stale value  ← BUG if CPU reads here

Solution A (Coherent DMA): hardware coherency fabric or non-cached mapping
Solution B (Streaming DMA): driver uses the required map/sync/unmap ownership API
```

### Coherent (Consistent) DMA

`dma_alloc_coherent(dev, size, &dma_handle, GFP_KERNEL)`

- Returns a CPU virtual pointer and a device-visible `dma_handle`; do not substitute a CPU physical address for the DMA address.
- CPU and device accesses are coherent under the platform DMA API. Implementations may use hardware-coherent cached memory or uncached mappings; "coherent" does not universally mean "uncached".
- Coherence does not order descriptor writes. Publish descriptor fields before a valid/ownership bit using the ordering required by the device and DMA API.
- Check allocation failure and configure an appropriate DMA mask before allocation. The buffer cannot be freed while the device still references it.

These are the portable guarantees in the [Linux DMA API HOWTO](https://docs.kernel.org/core-api/dma-api-howto.html); physical contiguity and cacheability must not be inferred from a virtual pointer.

### Streaming DMA and explicit ownership

Streaming DMA is an API ownership/mapping model, including on hardware-coherent platforms. Drivers use the required map/unmap or sync calls; the platform decides whether cache maintenance, address translation, or bouncing is needed:

```c
/* CPU writes or updates buffer contents, preparing data for device */
/* The DMA API performs the platform-required mapping/synchronization. */
dma_addr_t dma = dma_map_single(dev, cpu_ptr, size, DMA_TO_DEVICE);
if (dma_mapping_error(dev, dma))
    return -EIO; /* Reading exercise: enclosing function/error cleanup omitted. */
/* Do not access the buffer while the device owns this transfer. */
/* submit descriptor to device hardware to start DMA */
/* ... device performs DMA read from system RAM into its memory ... */
/* When DMA completes, tell the kernel DMA transfer is finished and the CPU can safely reuse or modify cpu_ptr */
dma_unmap_single(dev, dma, size, DMA_TO_DEVICE);
/* After dma_unmap_single, ownership of cpu_ptr returns to CPU and it is safe for CPU to access or modify the memory */
```

`dma_map_single()` establishes a device mapping and performs platform-required synchronization in the given direction. Check `dma_mapping_error()` before submitting work; the returned address may differ from the CPU physical address.

For a map → submit → complete → unmap transfer, the CPU leaves the buffer untouched while the device owns it. A persistent mapping can alternate ownership with `dma_sync_single_for_cpu()` and `dma_sync_single_for_device()`; wait for completion before returning ownership to the CPU. Cache maintenance operations are architecture-dependent, so application drivers should use the DMA API rather than hard-code flush/invalidate sequences.

Use `DMA_TO_DEVICE` when the device reads, `DMA_FROM_DEVICE` when it writes, and `DMA_BIDIRECTIONAL` only when both are required. With scatter-gather mappings, distinguish the original entry count from the mapped DMA segment count. Review the [DMA API guide](https://docs.kernel.org/core-api/dma-api-howto.html) alongside the target driver.

---

## IOMMU (Input-Output Memory Management Unit)

The **IOMMU** sits between PCIe (or AXI) devices and the system memory bus. It translates device-issued **IOVAs** (I/O Virtual Addresses) to physical addresses using device-specific page tables. Think of the IOMMU as a **second MMU dedicated to devices** — just as the CPU's MMU gives each process its own virtual address space, the IOMMU gives each device its own I/O virtual address space.

```
Without IOMMU:
Device issues DMA to physical address 0x8000_0000
                ↓
          [System RAM]     ← Device can DMA anywhere in physical RAM
                              (any malicious or buggy driver = security hole)

With IOMMU:
Device issues DMA to IOVA 0x0001_0000
                ↓
         ┌──────────────┐
         │    IOMMU     │   IOVA → PA translation table (per device/group)
         │  page tables │
         └──────┬───────┘
                │ maps to PA 0x8000_0000 (if mapped)
                │ or IOMMU fault (if not mapped)
                ▼
          [System RAM]     ← Device can only reach explicitly mapped regions
```

| IOMMU implementation | Platform |
|----------------------|----------|
| Intel VT-d | x86 Intel SoCs and Xeon |
| AMD-Vi (IOMMU) | x86 AMD Ryzen / EPYC |
| ARM SMMU v2/v3 | ARM SoCs, Jetson Orin, server ARM |

### Security and Isolation

Without IOMMU: a device (or compromised driver) can issue DMA to any physical address, reading or corrupting arbitrary memory. With IOMMU: device can only DMA to explicitly mapped IOVAs; any access outside mapped windows causes an IOMMU fault (logged, device stalled or reset).

### IOMMU Groups

An **IOMMU group** is the kernel’s minimum isolation unit for device assignment. Grouping can reflect hardware topology, aliasing, peer-to-peer paths, and isolation capabilities; it does not simply mean the devices use one physical translation engine. Assignment must preserve isolation for the whole group, but that does not mean every device in it must be exposed as a userspace handle. Follow the group/device binding rules of the actual VFIO implementation. [Linux VFIO groups](https://docs.kernel.org/driver-api/vfio.html#groups-devices-and-iommus).

### Kernel API

```c
/* Concept only: direct IOMMU API signatures change with kernel version. */
/* Map an IOVA range to physical pages with chosen permissions. */
iommu_unmap(domain, iova, size);
```

These low-level calls are typically used by framework code (DMA subsystem, VFIO). Driver authors use the higher-level `dma_map_*()` API, which handles IOMMU mapping automatically.

### VFIO

**VFIO** (Virtual Function I/O) exposes IOMMU-backed device access to **userspace through kernel VFIO drivers and IOMMU-mediated access**. Used by: DPDK (user-space NIC drivers), SPDK (user-space NVMe), FPGA userspace drivers, KVM GPU passthrough. The VFIO container maps groups to IOMMU domains and allows userspace DMA via `/dev/vfio/N`.

---

## DMA-BUF Framework

With individual DMA APIs understood, the next challenge is sharing a single DMA buffer between multiple hardware engines — for example, a camera ISP, a GPU inference engine, and a display engine all processing the same frame. Copies can add memory traffic and latency; whether they dominate depends on the workload. **DMA-BUF** solves this.

DMA-BUF is a kernel abstraction for sharing DMA buffers between independent subsystems (CPU, GPU, camera ISP, video encoder, display engine) without copying.

### Roles

- **Exporter**: owns the buffer; allocates backing pages or device memory; implements `struct dma_buf_ops` (`attach`, `map_attachment`, `unmap_attachment`, `mmap`, `release`, `vmap`)
- **Importer**: attaches to an exported buffer; maps it for its device's DMA engine

### Lifecycle

**Protocol pseudocode; exact kernel signatures/locking depend on the target version:**

```text
exporter creates a dma_buf; check errors and establish its release callback
exporter creates an fd; handle failure and reference ownership correctly
transfer descriptor with SCM_RIGHTS when crossing a process boundary
importer acquires a reference and attaches its device; check each failure
map the attachment for the direction in which THAT device accesses it
wait for producer completion and establish required visibility
submit device work; retain mappings and references while access is outstanding
after completion/quiescence: unmap, detach, and release importer reference
release exporter storage only when all required references/accesses are gone
```

A GPU reading a captured frame normally uses a device-read direction, not an arbitrary `DMA_FROM_DEVICE` copied from a producer example. Each mapping is for its own device. Buffer CPU access may need a separate cache-synchronization protocol, and fences do not automatically implement mutual exclusion. See [DMA-BUF synchronization](https://docs.kernel.org/driver-api/dma-buf.html).

### Zero-Copy Pipeline

```
┌──────────────┐    DMA-BUF fd    ┌─────────────────┐   DMA-BUF fd   ┌──────────────────┐
│  V4L2 camera │ ───────────────> │  CUDA importer  │ ────────────>  │  Display engine  │
│  driver      │  (kernel buffer  │  (same physical │                │  (same physical  │
│  (exporter)  │   export as fd)  │   pages mapped  │                │   pages mapped   │
└──────────────┘                  │   into GPU VA)  │                │   for display)   │
                                  └─────────────────┘                └──────────────────┘
                    No CPU copy at any stage — same physical pages throughout
```

```
V4L2 camera driver → DMABUF fd → CUDA importer → inference kernel → display
```

This is a possible compatible-buffer path, not a guarantee for every camera/CUDA/display combination. The fd can be passed using `SCM_RIGHTS`. Establish the supported producer/consumer completion and cache-ownership protocol; a generic fence call does not make arbitrary imports compatible.

### V4L2 Integration

V4L2 supports DMA-BUF as a memory type (`V4L2_MEMORY_DMABUF`). The single-planar API carries an fd in `struct v4l2_buffer.m.fd`; the multi-planar API uses the corresponding plane descriptors. The ISP driver maps the buffer for its DMA engine automatically on `VIDIOC_QBUF`.

> **Key Insight:** DMA-BUF file descriptors are the "passport" of shared hardware buffers. A single fd can be passed across process boundaries, imported by multiple device drivers, and mapped by each device's DMA engine into the device's address space — all without any data movement in system RAM.

---

## GPU memory and platform capabilities

A discrete GPU commonly has separate VRAM, while Jetson's integrated GPU shares physical DRAM with the CPU. Shared physical DRAM does not imply that every pointer is GPU-accessible or that accesses are automatically synchronized. Allocators, mappings, cacheability, and ownership remain distinct.

| Allocation kind | Common CUDA API | Contract to investigate |
|---|---|---|
| Device allocation | `cudaMalloc()` | Device access, explicit transfers where needed, lifetime until outstanding uses finish |
| Pinned host allocation | `cudaMallocHost()` / registration | Host memory resource cost, supported asynchronous transfer/mapping paths |
| Managed allocation | `cudaMallocManaged()` | Platform-specific managed-memory capabilities, residency and concurrent-access rules |
| Mapped host allocation | Supported host allocation/registration plus device pointer query | Access over the actual interconnect and required synchronization |
| External shared buffer | Supported external/graphics/vendor interop | Handle type, allocator, format, import compatibility and fences |

Memory migration granularity is not universally 64 KiB. Some platforms migrate; some support hardware-coherent access; some impose additional restrictions. Check device attributes and the applicable [CUDA for Tegra guidance](https://docs.nvidia.com/cuda/cuda-for-tegra-appnote/index.html).

Prefetching can influence residency and move work outside a measured interval where supported. It is not mandatory or useful for every platform, and its return value and stream dependencies matter. API signatures also evolve: CUDA 13.x documentation uses a `cudaMemLocation` destination and flags in `cudaMemPrefetchAsync`, while older lessons often show an integer-device overload. Compile against the selected toolkit and follow its [runtime memory API](https://docs.nvidia.com/cuda/cuda-runtime-api/cuda_runtime_api/group__CUDART__MEMORY.html). This lesson supplies no untested cross-version CUDA wrapper.

For a discrete GPU, a Gen5 x16 link has an encoding ceiling of about 63.02 GB/s in one direction before packet overhead. Summing both directions gives about 126.03 GB/s; that sum cannot be used as a one-way transfer rate. BAR aperture size, Above-4G decoding, and resizable BAR are related platform concepts, not interchangeable terms or a fixed one-eighth-of-VRAM rule. [Lecture 20](Lecture-20.md) develops the route calculation.

On Jetson, do not reduce all allocators to `nvmap`, or assert that an ISP frame is immediately readable by CUDA at full DRAM bandwidth. Use the current supported allocator/import API and [camera buffer protocol](Lecture-18.md). A format conversion or repacking step may still be required even when no CPU copy is used.

## Worked scatter-gather contract

Suppose the original scatterlist has 6 entries and `dma_map_sg()` returns 4 mapped DMA segments after merging. Program the device using the **4 mapped segments**, subject to its descriptor limits. Unmap using the **original 6-entry count**, as the DMA API requires. If mapping returns zero, do not submit the transfer. Keep the source pages and mapping valid through device completion.

Now suppose an application times out and drops a request ID. That does not stop DMA. A generation counter can reject a stale notification, but only completion or the device's abort/reset/quiescence protocol makes the storage reusable. This is the same ownership principle exercised by the [native SPSC lab](../labs/SPSC-Queue.md), with additional device/cache rules.

## Answered checks

**Does coherent allocation remove memory barriers?** No. Coherence and ordering differ. Descriptor fields must become visible before the device observes a valid/ownership flag, according to the platform/device contract.

**Does `dma_unmap_single()` wait for hardware?** Do not assume so. The driver must establish transfer completion before unmapping and reclaiming the storage.

**Does an IOMMU contain a buggy driver's CPU writes?** No. It restricts device DMA under its configuration, not arbitrary privileged CPU memory accesses. Bypass mappings and shared isolation groups also affect the boundary.

**How many payload bytes are in one tightly packed 1080p RGBA frame?** `1920×1080×4 = 8,294,400` bytes. At 60 fps that is about 0.498 GB/s payload. A CPU copy adds both reads and writes; padded strides or another pixel format change the calculation.

**Does using one physical DRAM eliminate all movement?** No. Copies, cache maintenance, format conversions, and memory traffic may still occur. Measure the actual allocation and access path.
