# Lecture Note 07 (L15, L16): DMA, IOMMU & GPU Memory; NUMA & HPC Optimization

**Combines:** Lecture L15 (DMA, IOMMU & GPU Memory Management) and Lecture L16 (NUMA Topology & HPC Memory Optimization).

---

## How This Note Is Organized

1. **Part 1 — DMA & IOMMU:** Direct Memory Access; cache coherency (coherent vs streaming); DMA-BUF; IOMMU and security; VFIO; GPU memory and zero-copy.
2. **Part 2 — NUMA:** Non-uniform memory access; first-touch and placement; memory policies (bind, preferred, interleave); numactl and libnuma; AutoNUMA; multi-GPU and CPU–GPU affinity.

---

# Part 1: DMA, IOMMU & GPU Memory Management

**Context:** Devices move data to/from RAM **without CPU copies**. The CPU must not see **stale data** (cache coherency). The **IOMMU** translates device addresses and restricts access; **DMA-BUF** shares one buffer across CPU, GPU, camera, display.

---

## DMA (Direct Memory Access)

- Device (NIC, NVMe, GPU, camera ISP) transfers data to/from system RAM **autonomously**. CPU programs **descriptor** (src, dst, length); device runs transfer; completion via IRQ or poll.
- **Without DMA:** Device → CPU copy → RAM. **With DMA:** Device → DMA engine → RAM; CPU is free after submitting descriptor.

---

## Cache Coherency: Coherent vs Streaming

- **Coherent DMA:** `dma_alloc_coherent()` — uncached or hardware-coherent; CPU and device always see same data; no explicit sync. Use for small control/descriptor regions.
- **Streaming DMA:** CPU uses cached memory; driver **synchronizes** explicitly:
  - **DMA_TO_DEVICE:** CPU wrote data → `dma_map_single()` flushes cache → device reads.
  - **DMA_FROM_DEVICE:** Device will write → after transfer, `dma_unmap_single()` invalidates cache → CPU reads. Between map and unmap the buffer is "owned" by the device — CPU must not touch it.
- **dma_map_sg** / **dma_unmap_sg** for scatter-gather (fragmented buffers). Direction controls which cache op is used; wrong direction = silent corruption.

---

## IOMMU (Input-Output MMU)

- Sits between devices and memory bus. Translates **IOVA** (I/O Virtual Address) to physical address using per-device/group page tables.
- **Without IOMMU:** Device can DMA to any physical address (security risk). **With IOMMU:** Only mapped IOVAs are allowed; unmapped access → fault.
- **Implementations:** Intel VT-d, AMD-Vi, ARM SMMU (Jetson Orin). **IOMMU groups:** Devices behind same translation unit; for VFIO passthrough, whole group is assigned together.
- Drivers usually use **dma_map_*** which uses IOMMU automatically; low-level `iommu_map`/`iommu_unmap` in framework code.

---

## DMA-BUF & Zero-Copy Pipeline

- **DMA-BUF:** Kernel abstraction to share one DMA buffer across subsystems (CPU, GPU, camera ISP, display). **Exporter** allocates and exports; **importer** attaches and maps for its device.
- **Lifecycle:** Exporter creates buffer, gets fd; fd passed (e.g. Unix socket); importer `dma_buf_get(fd)`, `dma_buf_attach`, `dma_buf_map_attachment` → sg_table with IOVAs for that device.
- **Zero-copy:** V4L2 camera → DMA-BUF fd → CUDA importer → same physical pages for inference → display. No CPU copy; sync via DMA fence (`dma_fence_wait`). V4L2 supports `V4L2_MEMORY_DMABUF`; userspace passes fd in buffer.

---

## GPU Memory & Unified Memory

- Discrete GPU: dedicated VRAM; driver manages allocations and CPU↔GPU copies. **Unified memory (e.g. Jetson):** CPU and GPU share physical RAM; one address space; no explicit copy for shared buffers, but bandwidth and latency depend on placement.
- **Resizable BAR (SAM):** GPU BAR1 can cover full VRAM; CPU can address all GPU memory (important for GPUDirect Storage, zero-copy).

---

# Part 2: NUMA Topology & HPC Memory Optimization

**Context:** On multi-socket machines, memory attached to socket 0 is **"local"** to CPUs on socket 0 and **"remote"** to socket 1. Remote access has **higher latency and lower bandwidth**; placement of data and threads matters.

---

## NUMA Architecture

- Each socket (NUMA node) has **local DRAM** (lower latency, full bandwidth). Access to another node's DRAM goes over **interconnect** (QPI/UPI, Infinity Fabric) — **~2× latency, ~half bandwidth**.
- **Discovery:** `numactl --hardware`, `lstopo`, `numastat`, `/sys/devices/system/node/nodeN/distance`. Distance matrix: local = 10; remote often 20–40.

---

## First-Touch and Placement

- **Default (first-touch):** Page is allocated on the node of the CPU that first **faults** it. If main thread on node 0 initializes a buffer later used only by workers on node 1, every access is remote — silent slowdown.
- **Fix:** Allocate (and fault) on the same node that will use the data: pin thread to node, set mempolicy, then malloc + memset; or use `numa_alloc_onnode()` / `mbind(..., MPOL_MF_MOVE)` to move pages.

---

## Memory Policies & numactl

- **MPOL_DEFAULT:** First-touch. **MPOL_BIND:** Only listed nodes; fail if full. **MPOL_PREFERRED:** Prefer one node; fallback. **MPOL_INTERLEAVE:** Round-robin across nodes (bandwidth-bound).
- **set_mempolicy** (process); **mbind** (VMA; MPOL_MF_MOVE migrates existing pages).
- **numactl:** `--cpunodebind=0 --membind=0 ./app` (bind CPU and memory to node 0); `--interleave=all` (interleave); `--preferred=1` (prefer node 1).

---

## libnuma & AutoNUMA

- **libnuma:** `numa_node_of_cpu(sched_getcpu())`, `numa_alloc_onnode()`, `numa_bind()`, `numa_set_membind()`, `numa_set_interleave_mask()`.
- **AutoNUMA:** Kernel migrates "hot" pages to the node of the accessing CPU. Overhead (scan, TLB shootdown, migration); can cause tail latency spikes. **Recommendation:** Disable (`numa_balancing=0`) on RT and latency-sensitive inference; use explicit numactl/mbind.

---

## Multi-GPU & CPU–GPU Affinity

- GPUs are attached to **one socket's PCIe root**. CPU↔GPU traffic from the other socket **crosses interconnect**. Use `nvidia-smi topo -m` to see topology; **bind process and memory to the node that owns the GPU(s) used**.

---

## Summary Tables

**DMA:** Coherent = uncached/coherent, no sync. Streaming = map (flush/invalidate) → device uses → unmap (invalidate/flush); CPU must not touch between map/unmap.

**IOMMU:** IOVA→PA; per-device/group; restricts DMA; VFIO exposes to userspace for passthrough.

**NUMA:** Local vs remote latency/bandwidth; first-touch; bind/preferred/interleave; numactl; disable AutoNUMA for RT.

---

## AI Hardware Connection

- **DMA-BUF + V4L2:** Zero-copy camera → GPU; fd over IPC. **Streaming DMA** direction must be correct (TO_DEVICE / FROM_DEVICE) to avoid corruption.
- **IOMMU:** Security and isolation for GPU/VFIO; required for safe passthrough.
- **NUMA:** Pin inference and weights to same node as GPU; first-touch on wrong node halves effective bandwidth. **numastat -p** to verify; disable AutoNUMA on inference servers.
- **Multi-GPU:** Place processes and allocations on the node that owns the target GPU; use topology tools to avoid cross-socket PCIe.

---

*Combines Lectures L15, L16 (DMA, IOMMU, GPU Memory; NUMA, HPC Optimization).*
