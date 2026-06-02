# Lecture 15: DMA, IOMMU & GPU Memory Management

## Overview

Modern AI systems depend on moving large amounts of data — camera frames, model weights, activations — between CPU, GPU, camera ISP, and storage at high speed. Doing this through the CPU (copying data in software) would be impossibly slow and waste CPU cycles needed for inference. **Direct Memory Access (DMA)** lets hardware engines transfer data autonomously without CPU involvement. The core challenge is ensuring the CPU's caches and the device always see consistent data. The mental model for this lecture is a production pipeline: raw sensor data enters memory, gets processed by multiple hardware engines in sequence, and reaches the inference engine — ideally without a single software copy. For an AI hardware engineer, mastering DMA and the IOMMU is essential for designing zero-copy camera-to-inference pipelines, writing correct PCIe device drivers, and understanding why Jetson's unified memory architecture behaves differently from a discrete GPU.

---

## DMA: Direct Memory Access

**DMA** allows a device (NIC, NVMe controller, GPU PCIe DMA engine, camera ISP) to transfer data directly to or from system RAM **without CPU participation** in the data path. The CPU programs a **descriptor** specifying source address, destination address, transfer length, and flags; the device executes the transfer autonomously; completion is signaled via an interrupt or a status flag the driver polls.

Benefits: CPU is freed during bulk transfers; transfer latency overlaps with CPU computation; system throughput increases.

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
Solution B (Streaming DMA): driver explicitly invalidates cache before CPU reads
```

### Coherent (Consistent) DMA

`dma_alloc_coherent(dev, size, &dma_handle, GFP_KERNEL)`

- Returns a physically contiguous CPU virtual address and a device-visible `dma_handle`
- The memory allocated by `dma_alloc_coherent` is **non-cacheable** *only for that specific DMA buffer*, to ensure CPU and device always see the same data. When the CPU accesses this buffer, its data is not cached.
- Importantly, the CPU *can still cache data for all other normal memory regions*—the non-cacheable setting applies only to the buffer returned by the coherent DMA allocation. All other (non-DMA) memory uses normal CPU caching and is unaffected.
- Since CPU access to the coherent DMA buffer is uncached, reads and writes are slower; use this kind of memory for things like small control rings, descriptor tables, or status blocks where correctness is more important than speed.

### Streaming (Non-Coherent) DMA

CPU uses cached memory; driver **explicitly synchronizes**. This is more complex but allows faster CPU access to the data when the CPU is not sharing the buffer with a device:

```c
/* CPU writes or updates buffer contents, preparing data for device */
/* dma_map_single flushes (writes back) any cached data covering cpu_ptr, so device sees the latest data in RAM */
dma_addr_t dma = dma_map_single(dev, cpu_ptr, size, DMA_TO_DEVICE);
/* At this point, the buffer must not be touched by the CPU until dma_unmap_single is called */
/* submit descriptor to device hardware to start DMA */
/* ... device performs DMA read from system RAM into its memory ... */
/* When DMA completes, tell the kernel DMA transfer is finished and the CPU can safely reuse or modify cpu_ptr */
dma_unmap_single(dev, dma, size, DMA_TO_DEVICE);
/* After dma_unmap_single, ownership of cpu_ptr returns to CPU and it is safe for CPU to access or modify the memory */
```

`dma_map_single()` performs the required cache maintenance for DMA in the specified direction: it flushes (writes back and invalidates) CPU cache lines covering the buffer for `DMA_TO_DEVICE` (so the device sees all CPU updates), or invalidates cache for `DMA_FROM_DEVICE` (so the CPU doesn't see stale data after device writes). `dma_unmap_single()` restores CPU ownership and, depending on direction, may invalidate the cache again so the CPU will read any data written by the device.

The mapping and unmapping calls perform the **cache synchronization**. Between `dma_map_single()` and `dma_unmap_single()`, the buffer "belongs" to the device — **the CPU must not touch it**.

For fragmented memory, `dma_map_sg()` / `dma_unmap_sg()` operate on scatter-gather lists.

DMA directions:
- `DMA_TO_DEVICE`: CPU→device; flush cache before map
- `DMA_FROM_DEVICE`: device→CPU; invalidate cache before device write, invalidate again after
- `DMA_BIDIRECTIONAL`: both; most conservative (flush + invalidate)

> **Key Insight:** The direction flag is not just documentation — it controls which cache operation is performed. `DMA_TO_DEVICE` flushes dirty cache lines so the device reads correct data. `DMA_FROM_DEVICE` invalidates cache lines so the CPU cannot read stale data after the device writes. Getting the direction wrong causes silent data corruption that is extremely hard to debug.

> **Common Pitfall:** Accessing a buffer between `dma_map_single()` and `dma_unmap_single()` from the CPU is undefined behavior. The kernel documentation calls this "ownership": the device owns the buffer between map and unmap. Some architectures will silently return stale cached data; others will see both CPU and device writes collide. Always obey the ownership rule.

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

An **IOMMU group** is a set of PCIe devices that share the same IOMMU translation hardware, meaning their memory accesses are managed together and **cannot be separated** at the IOMMU level. As a result, all devices in the same group must be treated as a unit when configuring access control—for example, when assigning devices to userspace (such as for passthrough to another software component or user process). It is not possible to give access to only one device in the group while restricting the others; access is always granted to the entire group together.

### Kernel API

```c
iommu_map(domain, iova, paddr, size, IOMMU_READ | IOMMU_WRITE);
iommu_unmap(domain, iova, size);
```

These low-level calls are typically used by framework code (DMA subsystem, VFIO). Driver authors use the higher-level `dma_map_*()` API, which handles IOMMU mapping automatically.

### VFIO

**VFIO** (Virtual Function I/O) exposes IOMMU-backed device access to **userspace without a kernel driver**. Used by: DPDK (user-space NIC drivers), SPDK (user-space NVMe), FPGA userspace drivers, KVM GPU passthrough. The VFIO container maps groups to IOMMU domains and allows userspace DMA via `/dev/vfio/N`.

---

## DMA-BUF Framework

With individual DMA APIs understood, the next challenge is sharing a single DMA buffer between multiple hardware engines — for example, a camera ISP, a GPU inference engine, and a display engine all processing the same frame. Copying the data between each stage would eliminate the bandwidth benefit of DMA entirely. **DMA-BUF** solves this.

DMA-BUF is a kernel abstraction for sharing DMA buffers between independent subsystems (CPU, GPU, camera ISP, video encoder, display engine) without copying.

### Roles

- **Exporter**: owns the buffer; allocates backing pages or device memory; implements `struct dma_buf_ops` (`attach`, `map_attachment`, `unmap_attachment`, `mmap`, `release`, `vmap`)
- **Importer**: attaches to an exported buffer; maps it for its device's DMA engine

### Lifecycle

```c
/* Exporter */
struct dma_buf *buf = dma_buf_export(&exp_info);  /* create exportable buffer */
int fd = dma_buf_fd(buf, O_CLOEXEC);              /* get a file descriptor handle */
/* pass fd to importer process via Unix socket */

/* Importer */
struct dma_buf *buf = dma_buf_get(fd);            /* get reference from fd */
struct dma_buf_attachment *att = dma_buf_attach(buf, importer_dev);  /* attach device */
struct sg_table *sgt = dma_buf_map_attachment(att, DMA_FROM_DEVICE); /* get sg list */
/* sgt contains scatter-gather list with IOVAs for importer device */
dma_buf_unmap_attachment(att, sgt, DMA_FROM_DEVICE);
dma_buf_detach(buf, att);
dma_buf_put(buf);
```

The fd is a **cross-process, cross-subsystem handle** to the same physical buffer. Passing an fd via a Unix socket lets two separate processes share the same DMA buffer **without any kernel copy**.

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

No CPU copy at any stage. The fd is passable between processes via `SCM_RIGHTS` on a Unix socket, enabling cross-process zero-copy. Synchronization between producers and consumers uses the DMA fence framework (`dma_fence_wait()`).

### V4L2 Integration

V4L2 supports DMA-BUF as a memory type (`V4L2_MEMORY_DMABUF`). Userspace passes the DMA-BUF fd in `struct v4l2_buffer.m.fd`. The ISP driver maps the buffer for its DMA engine automatically on `VIDIOC_QBUF`.

> **Key Insight:** DMA-BUF file descriptors are the "passport" of shared hardware buffers. A single fd can be passed across process boundaries, imported by multiple device drivers, and mapped by each device's DMA engine into the device's address space — all without any data movement in system RAM.

---

## GPU Memory Architecture

With the general DMA framework established, GPU memory management adds one more layer of complexity: GPU VRAM is separate from system DRAM and connected via PCIe, creating a bandwidth bottleneck that must be carefully managed.

### VRAM and PCIe BAR

GPU VRAM (GDDR6X on GeForce, HBM2e/HBM3 on datacenter GPUs) is accessed by the CPU through PCIe **Base Address Registers (BAR)**. The GPU exposes a **BAR aperture**; the CPU maps it as uncached MMIO.

```
CPU (System DRAM)                           GPU (VRAM)
┌──────────────┐         PCIe Bus          ┌──────────────┐
│   DRAM       │ <─────────────────────>   │   VRAM       │
│  (CPU DDR5)  │  ~128 GB/s (PCIe 5.0 x16) │  (HBM3)      │
└──────────────┘                           └──────────────┘
       │                                         │
  CPU accesses                           GPU accesses
  GPU VRAM via                           VRAM directly
  PCIe BAR                               at 3.35 TB/s
  (uncached MMIO)                        (H100 HBM3)
```

- PCIe 5.0 x16 peak bandwidth: ~128 GB/s bidirectional
- Discrete GPU typically exposes 1/8 of VRAM via the 256 MB or 16 GB BAR (resizable BAR / Above-4G Decoding)
- NVIDIA NVLink bypasses PCIe for GPU-GPU transfers (600 GB/s on H100 NVLink 4.0)

### CUDA Memory Types

| Type | API | Coherent CPU↔GPU? | Notes |
|------|-----|-------------------|-------|
| Device memory | `cudaMalloc()` | No | VRAM; fastest for GPU kernels |
| Pinned host memory | `cudaMallocHost()` | No | DMA-able; faster H2D/D2H transfers |
| Unified Memory | `cudaMallocManaged()` | Yes (demand migration) | Single pointer, both CPU and GPU |
| Mapped host memory | `cudaHostGetDevicePointer()` | No | Zero-copy over PCIe; low bandwidth |

### CUDA Unified Memory

A **single pointer is valid on both CPU and GPU**. The CUDA driver and OS collaborate to migrate 64KB-granule pages on fault:
- GPU page fault → migrate from CPU to VRAM
- CPU page fault → migrate from VRAM to CPU DRAM

```
cudaMallocManaged() allocation lifecycle:
                    ┌─────────────────────┐
                    │  Single virtual ptr │
                    │  (valid on CPU+GPU) │
                    └─────────┬───────────┘
                              │
              ┌───────────────┴───────────────┐
              │                               │
    CPU accesses                      GPU accesses
    (page fault)                      (page fault)
              │                               │
    ┌─────────▼──────┐              ┌─────────▼──────┐
    │ Migrate to     │              │ Migrate to     │
    │ CPU DRAM       │              │ GPU VRAM       │
    └────────────────┘              └────────────────┘
```

`cudaMemPrefetchAsync(ptr, size, device, stream)`: explicit prefetch before access; hides migration latency.
`cudaMemAdvise()` hints: `cudaMemAdviseSetReadMostly` (replicate across devices), `cudaMemAdviseSetPreferredLocation` (preferred residency).

Production inference code typically uses explicit `cudaMemcpy()` with pinned buffers for **deterministic latency**. Unified Memory is most useful for **rapid prototyping** and memory-constrained models that exceed VRAM.

> **Common Pitfall:** Using `cudaMallocManaged()` in production inference without `cudaMemPrefetchAsync()` causes unpredictable latency spikes at first access. The first kernel invocation triggers page migrations that can stall for milliseconds. Always prefetch Unified Memory regions before the timed inference path begins.

### nvmap on Jetson

Jetson uses a **unified memory architecture** (CPU and GPU share DRAM). **nvmap** manages carveout (physically contiguous) and IOMMU-mapped allocations for IOMMU-less peripherals:

```
Jetson Unified Memory Architecture
┌────────────────────────────────────────────────┐
│                  Shared DRAM                   │
│  ┌──────────┐  ┌──────────┐  ┌──────────────┐ │
│  │ CPU code │  │ GPU code │  │  NVDLA/VIC   │ │
│  │  & data  │  │  & data  │  │  DMA buffers │ │
│  └────┬─────┘  └────┬─────┘  └──────┬───────┘ │
│       │             │               │          │
│       └─────────────┴───────────────┘          │
│              All point to same physical         │
│              pages — no PCIe transfer!          │
└────────────────────────────────────────────────┘
```

- NVDLA, VIC, camera ISP, display engine use nvmap buffers
- `/dev/nvmap` userspace interface; `NVMAP_IOC_ALLOC`, `NVMAP_IOC_SHARE` ioctls
- CPU and GPU access the same physical pages; no PCIe transfer needed

> **Key Insight:** On Jetson, the distinction between "CPU memory" and "GPU memory" collapses — it is all the same DRAM. This eliminates PCIe transfer cost entirely. A camera frame captured by the ISP is immediately visible to both the CPU and the GPU CUDA kernel at full DRAM bandwidth, without any copy or PCIe transaction.

---

## Summary

| DMA type | Cache sync? | IOMMU needed? | API | Use case |
|----------|------------|--------------|-----|---------|
| Coherent | No (hardware) | No | `dma_alloc_coherent()` | Descriptor rings, control registers |
| Streaming single | Yes (explicit) | Optional | `dma_map_single()` | Single-buffer bulk transfers |
| Streaming SG | Yes (explicit) | Optional | `dma_map_sg()` | NVMe, network scatter-gather |
| DMA-BUF | Fence-based | Optional | `dma_buf_*` | Cross-device zero-copy pipelines |
| CUDA Unified | Automatic (driver) | N/A | `cudaMallocManaged()` | Prototype multi-stage inference |

### Conceptual Review

- **Why does DMA require cache synchronization on non-coherent systems?** The CPU caches data in L1/L2/L3. If the CPU writes to a buffer and the device then reads the same physical addresses via DMA, the device may read stale data from DRAM that has not yet been written back from CPU cache. `dma_map_single(DMA_TO_DEVICE)` flushes the cache first.

- **What is the difference between coherent and streaming DMA?** Coherent DMA memory is always in sync between CPU and device — typically because it is mapped uncached or because a hardware coherency fabric keeps it consistent. Streaming DMA uses cached memory for speed but requires explicit `map`/`unmap` calls for synchronization.

- **What does the IOMMU protect against?** A buggy or malicious device that issues DMA to arbitrary physical addresses. The IOMMU limits each device to only the IOVAs explicitly mapped for it. Any access outside those windows causes a fault, logged and the device is stalled.

- **Why can DMA-BUF file descriptors be passed between processes?** A DMA-BUF fd is a kernel file descriptor backed by a reference-counted buffer object. Like any fd, it can be sent via `SCM_RIGHTS` on a Unix socket. The receiving process gets its own fd pointing to the same physical buffer — no copy occurs.

- **When should you prefer `cudaMemcpy` with pinned buffers over Unified Memory?** In production inference where deterministic latency is required. Unified Memory migrations happen lazily on first access, causing unpredictable latency. Pinned buffers with explicit `cudaMemcpyAsync` give deterministic transfer timing.

- **Why does Jetson not need PCIe transfers between CPU and GPU?** Jetson's integrated SoC architecture puts CPU and GPU on the same die sharing the same DRAM. There is no discrete GPU across a PCIe link. CPU and GPU both have direct DRAM access, so "transfer" is simply a virtual address remapping — zero data movement.

---

## AI Hardware Connection

- DMA-BUF with `V4L2_MEMORY_DMABUF` enables camera frame→CUDA zero-copy on Jetson; eliminates a full 1080p60 frame copy (~12 MB/frame × 60 = 720 MB/s CPU copy avoided)
- `dma_alloc_coherent` is the correct API for Zynq PL↔PS AXI DMA command rings and status descriptors where correctness and simplicity outweigh throughput
- IOMMU (ARM SMMU on Jetson Orin) partitions DRAM access per hardware engine; a malfunctioning NVDLA kernel cannot corrupt VIC or camera ISP buffers outside its mapped IOVA window
- CUDA Unified Memory with `cudaMemPrefetchAsync` allows prototype inference code to exceed VRAM size by streaming weights through CPU DRAM, at the cost of migration latency
- Streaming DMA with `DMA_FROM_DEVICE` is the access pattern for NVMe-to-GPU GPUDirect Storage, where NVMe data bypasses CPU cache and lands directly in GPU-accessible pinned memory
- VFIO enables userspace FPGA DMA drivers that bypass the kernel driver model entirely, useful for low-latency AI accelerator cards requiring sub-microsecond command submission
