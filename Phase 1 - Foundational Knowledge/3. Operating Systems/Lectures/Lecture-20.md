# Lecture 20: PCIe, NVMe & GPU Driver Architecture

## Overview

Every GPU, NVMe SSD, FPGA, and high-speed NIC in an AI system is connected through the same backbone: PCIe. Understanding PCIe is understanding the physical limits of data movement in your system. If you wonder why copying data from an NVMe SSD to the GPU takes a certain amount of time, or why two GPUs in the same server can communicate faster than expected, the answer lies in the PCIe topology.

The mental model to carry through this lecture is the **hierarchy of interconnects**: data flows from storage (NVMe) through the PCIe fabric to compute (GPU), and the bandwidth and latency at each hop are determined by the PCIe generation and the number of lanes. The GPU driver architecture then sits on top of this physical layer, translating CUDA API calls into DMA transactions and register writes.

AI hardware engineers need to understand PCIe topology and GPU driver internals because GPUDirect Storage, peer-to-peer GPU communication, and FPGA inference accelerators all depend on knowing which devices share a PCIe switch, what BAR space is available, and how the IOMMU interacts with DMA mappings. Diagnosing performance anomalies in training and inference pipelines frequently requires reading PCIe bandwidth counters and understanding the driver's ioctl paths.

---

## PCIe Topology

PCIe (Peripheral Component Interconnect Express) is the standard **high-speed interconnect** for GPUs, NVMe SSDs, FPGAs, and NICs in AI hardware systems.

**Hierarchy**: Root Complex (CPU/SoC) → Root Ports → PCIe Switches → Endpoints (GPU, NVMe, NIC, FPGA)

**Lane bandwidth** by generation (per lane, each direction):

| Generation | Transfer rate | x16 bandwidth (bidirectional) |
|---|---|---|
| Gen3 | 8 GT/s (~1 GB/s/lane) | ~32 GB/s |
| Gen4 | 16 GT/s (~2 GB/s/lane) | ~64 GB/s |
| Gen5 | 32 GT/s (~4 GB/s/lane) | ~128 GB/s |

The topology of a typical AI server looks like this. The **position of a device in the tree** directly determines the bandwidth available for peer-to-peer transfers.

```
CPU (Root Complex)
├── Root Port 0
│   └── PCIe Switch
│       ├── GPU 0 (x16)     ← A100/H100
│       ├── GPU 1 (x16)     ← A100/H100
│       └── NVMe SSD (x4)   ← GPUDirect Storage target
├── Root Port 1
│   └── PCIe Switch
│       ├── GPU 2 (x16)
│       ├── GPU 3 (x16)
│       └── NIC (x16)       ← 100 GbE for distributed training
└── Root Port 2
    └── FPGA (x8)           ← Pre-processing accelerator
```

> **Key Insight:** Two GPUs connected to the same PCIe switch can communicate peer-to-peer without data transiting system RAM. GPU 0 and GPU 1 in the diagram above can DMA directly to each other's memory via the switch. GPU 0 and GPU 2 must route through the Root Complex and across two Root Ports — this is slower and may traverse system RAM depending on the IOMMU configuration.

### PCIe Device Discovery

During boot, BIOS/firmware walks the bus hierarchy via configuration cycles. Each device exposes a **config space**:

- Standard (256 B): Vendor ID, Device ID, Command, Status, Revision, Class Code
- Extended (4 KB): PCIe capabilities linked list (MSI-X, AER, SR-IOV, etc.)
- Kernel reads config space via `/sys/bus/pci/devices/0000:01:00.0/config`

The boot-time discovery sequence works as follows:

1. **BIOS issues Configuration Read transactions** to each possible bus/device/function combination to probe for device presence.
2. **Each device responds with its Vendor ID and Device ID**, identifying itself. If nothing is present, the read returns 0xFFFF.
3. **BIOS reads the Base Address Registers (BARs)** by writing all-ones and reading back the size mask. This tells the BIOS how much address space each device needs.
4. **BIOS programs BAR addresses** by writing physical addresses into each BAR register. From this point, the device responds to memory-mapped I/O at those addresses.
5. **Linux kernel re-enumerates at boot** via `pci_scan_root_bus()`, reading the same config space and building the `struct pci_dev` tree.

### Base Address Registers (BARs)

BARs declare what **memory and I/O regions** a device needs. The kernel allocates physical addresses and programs them during PCI enumeration.

- Kernel maps BAR into virtual address space via `ioremap()`; accessed with `readl()`/`writel()`
- **GPU BAR0**: device registers and control (16 MB–256 MB)
- **GPU BAR1**: VRAM aperture (CPU-accessible window into GPU framebuffer; up to full VRAM on recent GPUs with Resizable BAR / SAM)

> **Key Insight:** Resizable BAR (called Smart Access Memory or SAM by AMD) expands GPU BAR1 to cover the entire VRAM — up to 80 GB on an H100. Without it, only a small window (typically 256 MB) of VRAM is CPU-accessible at any time, requiring the driver to move the window for large allocations. With full BAR, the CPU can directly address any GPU memory location, which is critical for GPUDirect Storage zero-copy.

### PCIe DMA

Devices perform DMA to system RAM as **bus masters**. The kernel provides `dma_map_sg()` for scatter-gather mappings. The IOMMU translates bus addresses to physical addresses, providing **isolation and protection**.

> **Common Pitfall:** When enabling PCIe peer-to-peer DMA, the IOMMU must either be configured to permit the P2P transaction or bypassed via ACS (Access Control Services) disable. By default, many systems route all DMA through the IOMMU, which may serialize P2P traffic through the Root Complex even when a direct switch-level path exists. Check `lspci -vvv` for ACS capability and the current ACS enable bit.

### PCIe Peer-to-Peer (P2P)

Devices on the same PCIe switch can **DMA directly to each other** without data transiting system RAM:

- Requires `pci_p2pmem_alloc_sgl()` and IOMMU bypass or ACS (Access Control Services) disabled
- GPUDirect Storage uses P2P: NVMe SSD → GPU memory without CPU involvement
- FPGA → GPU inference pipeline: FPGA posts results directly to GPU buffers

```
Without P2P (data path through RAM):
NVMe SSD → PCIe Switch → Root Complex → System RAM → Root Complex → PCIe Switch → GPU
  ~100 µs latency, double bandwidth consumption, CPU involvement

With P2P (direct switch path):
NVMe SSD → PCIe Switch → GPU
  ~50 µs latency, no system RAM bandwidth consumed, zero CPU involvement
```

---

## NVMe

With PCIe topology established, NVMe is the **storage protocol that runs on top of PCIe** for flash SSDs. It was designed from the ground up for flash characteristics — unlike SATA, which was designed for spinning disks.

NVMe (Non-Volatile Memory Express) is the **PCIe-native storage protocol** designed for flash SSDs.

- **Latency**: ~100 µs (vs ~5 ms for SATA SSD, ~5–10 ms for HDD)
- **Queue depth**: up to 64K queues × 64K commands per queue
- **MSI-X**: one interrupt vector per queue; queues mapped to CPU cores for NUMA efficiency
- **Namespace**: logical drive abstraction; supports multiple namespaces per controller
- **ZNS (Zoned Namespaces)**: divide SSD into sequential-write zones; reduces write amplification for log workloads

> **Key Insight:** The 64K×64K queue structure is not theoretical headroom — it reflects the reality that modern NVMe SSDs can process thousands of I/O requests in parallel across internal NAND channels. A SATA SSD has one queue of depth 32. An NVMe SSD with 64K queues means one queue per CPU core on a large server, with no lock contention between cores submitting I/O.

### NVMe Linux Driver Stack

- `nvme_core.ko` + transport module (`nvme.ko` for PCIe, `nvme-rdma.ko` for NVMe-oF)
- `blk-mq`: multi-queue block layer; per-CPU software queues map to NVMe hardware queues
- `io_uring` or `libaio` submit directly to blk-mq; bypass page cache with `O_DIRECT`

The relationship between NVMe and io_uring (from Lecture 19) is direct: io_uring SQEs are translated into **blk-mq requests**, which are dispatched to NVMe hardware queues. Each NVMe hardware queue maps to one MSI-X interrupt vector and one CPU core, so completions interrupt **exactly the core that submitted the I/O**.

---

## GPU Driver Architecture (NVIDIA)

The GPU driver is split into **two layers**: a **kernel-mode driver** that manages hardware, and a **user-mode library** that manages CUDA contexts and streams. Understanding this split explains why CUDA API calls sometimes involve ioctls and sometimes do not.

### Kernel-Mode Driver (KMD)

`nvidia.ko` is the kernel-mode driver responsible for:

- PCIe device initialization, BAR mapping, firmware load
- GPU memory management (VRAM allocation, page table setup)
- UVM (Unified Virtual Memory): migrate pages between CPU and GPU on demand
- Context scheduling: time-share GPU across multiple processes
- MSI-X interrupt handling for completion events
- GSP-RM (GPU System Processor Resource Manager): since Ampere/Ada, firmware runs on the GPU's internal ARM core; `nvidia.ko` communicates with GSP via RPC

> **Key Insight:** Since Ampere, NVIDIA moved a large portion of resource management into the GSP — a dedicated ARM Cortex-A9 core inside the GPU die. This means that many operations that previously required `nvidia.ko` to do complex register sequences now just involve sending an RPC message to the GSP. The benefit is faster and more reliable driver updates (firmware updates the GSP; the kernel module stays the same). The cost is one more indirection layer to debug.

### User-Mode Driver (UMD)

`libcuda.so` is the CUDA user-mode driver:

- Communicates with `nvidia.ko` via `ioctl()` on `/dev/nvidia0`, `/dev/nvidiactl`, `/dev/nvidia-uvm`
- Manages CUDA contexts, streams, and events in userspace
- NVCC compiles CUDA C++ → PTX (portable assembly) → SASS (GPU ISA) via `ptxas`

The full call path from a CUDA API call to hardware looks like this:

1. Application calls `cudaMemcpyAsync(dst, src, size, cudaMemcpyHostToDevice, stream)`.
2. `libcuda.so` packages this as an ioctl payload and calls `ioctl(/dev/nvidia0, NV_ESC_RM_DMA_COPY, ...)`.
3. `nvidia.ko` receives the ioctl, validates the addresses against the process's page tables, and programs the GPU's copy engine DMA descriptor rings.
4. The copy engine DMA engine reads from system RAM and writes to VRAM over PCIe.
5. When the DMA completes, the GPU raises an MSI-X interrupt.
6. `nvidia.ko` interrupt handler writes a completion record and optionally wakes a waiting `cudaStreamSynchronize()` call.

### Device Nodes

| Node | Purpose |
|---|---|
| `/dev/nvidia0` | Per-GPU: context creation, memory allocation |
| `/dev/nvidiactl` | Global: device enumeration, capability query |
| `/dev/nvidia-uvm` | Unified Virtual Memory: managed memory, prefetch |
| `/dev/nvidia-modeset` | Display output (DRM/KMS via `nvidia-drm.ko`) |

### CUDA Execution Model

- **Context**: per-process GPU state; one context per device per process by default
- **Stream**: in-order command queue within a context; multiple streams enable overlap of compute and memory transfer
- **Event**: synchronization primitive; `cudaEventRecord()` / `cudaStreamWaitEvent()`

```
Process A                          GPU
  Context 0
    Stream 0: ──kernel──copy──────────────────▶  SM partition
    Stream 1: ──copy──kernel──────────────────▶  Copy engine (overlapped)
  Context 1 (another process)
    Stream 0: ──kernel────────────────────────▶  SM partition (time-shared)
```

> **Common Pitfall:** Creating multiple CUDA contexts in the same process (using the driver API directly) is allowed but each context has independent VRAM allocations and cannot share memory without explicit export/import. Most applications should use the runtime API which creates one implicit context per device. If two threads each call `cuCtxCreate()`, they get separate VRAM address spaces and cannot directly pass pointers between them.

---

## Open-Source GPU Drivers

The broader GPU driver ecosystem extends beyond NVIDIA's proprietary stack.

| Driver | GPU | Status | Userspace |
|---|---|---|---|
| `amdgpu` | AMD RDNA/CDNA | Fully open (driver + firmware) | `mesa` (ROCm, OpenGL, Vulkan) |
| `i915` | Intel | Fully open | `mesa` |
| `nouveau` | NVIDIA | Reverse-engineered; limited perf | `mesa` (no CUDA) |
| `nvidia-open` | NVIDIA (Turing+) | Open KMD; proprietary firmware | CUDA (same as proprietary) |

All open drivers use the Linux **DRM (Direct Rendering Manager)** subsystem. DRM provides the kernel framework for **GPU command submission**, GEM buffer objects, and KMS (Kernel Mode Setting) for display.

> **Key Insight:** `nvidia-open` (open since 2022 for Turing and later) uses an open-source kernel module that performs the same hardware interface as the proprietary `nvidia.ko`, but the GSP firmware blob remains proprietary. This allows Linux distributions to build the kernel module from source (improving security auditing and Secure Boot compatibility) while NVIDIA retains control of the GPU's internal firmware.

The transition to open-source infrastructure also means that `nvidia-open` participates in the DRM subsystem, enabling better integration with display managers, hibernation/resume, and GSP-based reconfiguration that previously required proprietary kernel hooks.

---

## FPGA as PCIe Endpoint

FPGAs appear in AI hardware pipelines as **pre-processing accelerators** (radar signal processing, sensor fusion) or as **inference accelerators** for custom neural network architectures.

Xilinx/AMD XDMA IP core exposes FPGA as a **PCIe DMA device**:

- H2C (Host-to-Card) and C2H (Card-to-Host) DMA channels
- Device nodes: `/dev/xdma0_h2c_0`, `/dev/xdma0_c2h_0`
- Write bitstream via PCIe during runtime reconfiguration (partial reconfiguration)
- PCIe P2P: FPGA can DMA directly to GPU buffer for post-processing pipelines

The FPGA data flow in a typical AI pipeline:

```
Radar ADC input
      ↓
FPGA (XDMA endpoint):
  - Signal processing (FFT, CFAR)
  - Feature extraction
  - Packs feature tensor into C2H DMA buffer
      ↓ PCIe P2P (no CPU, no system RAM)
GPU memory:
  - Feature tensor arrives as CUDA device pointer
  - Neural network inference kernel runs on tensor
      ↓
Detection output
```

> **Common Pitfall:** PCIe P2P from FPGA to GPU requires that both devices be in the same PCIe domain (behind the same Root Complex) and that P2P capability be enabled. On multi-socket servers, an FPGA connected to CPU 0's PCIe root and a GPU connected to CPU 1's PCIe root cannot do direct P2P — the data must cross the QPI/UPI inter-socket link and pass through system RAM. Always verify topology with `lstopo` or `nvidia-smi topo -m` before designing a P2P pipeline.

---

## Summary

| Component | Interface | Latency | Bandwidth | Kernel driver |
|---|---|---|---|---|
| GPU (A100) | PCIe Gen4 x16 | ~1 µs (DMA) | ~64 GB/s | `nvidia.ko` |
| NVMe SSD | PCIe Gen4 x4 | ~100 µs | ~7 GB/s | `nvme.ko` |
| FPGA (XDMA) | PCIe Gen3/4 x8 | ~5 µs (DMA) | ~16–32 GB/s | `xdma.ko` |
| NIC (100 GbE) | PCIe Gen4 x16 | ~1 µs | ~12 GB/s | `mlx5_core.ko` |
| GPU (P2P to NVMe) | PCIe switch | ~100 µs | PCIe switch BW | GPUDirect Storage |

### Conceptual Review

- **What is the difference between PCIe bandwidth per lane and total slot bandwidth?** Each PCIe lane provides bandwidth in each direction (it is full duplex). A Gen4 x16 slot provides approximately 2 GB/s per lane × 16 lanes = 32 GB/s in each direction, totalling ~64 GB/s bidirectional. The generation doubles bandwidth per lane each step (Gen3 → Gen4 → Gen5).

- **Why does BAR size matter for GPUDirect Storage?** GPUDirect Storage requires the CPU to set up DMA mappings that span GPU VRAM. With a small BAR (e.g., 256 MB), only a portion of VRAM is directly addressable by the CPU at one time. With Resizable BAR (full VRAM exposed), the entire GPU address space is always visible and DMA can target any location without BAR window management.

- **What is the role of `nvidia.ko` vs `libcuda.so`?** `nvidia.ko` owns the hardware: it initializes the GPU, manages VRAM page tables, handles MSI-X interrupts, and arbitrates between multiple processes. `libcuda.so` is the userspace face of CUDA: it provides the developer API, compiles PTX to SASS, and manages streams and events — all by sending ioctls to `nvidia.ko`.

- **How does MIG (covered in Lecture 23) relate to PCIe?** MIG partitions the GPU's internal compute and memory resources into independent instances, but all MIG instances still share the same PCIe connection to the host. The PCIe bandwidth is shared between MIG instances; only compute SMs and HBM slices are dedicated. This means MIG isolation is about preventing noisy-neighbor GPU utilization, not PCIe bandwidth isolation.

- **What is the ACS (Access Control Services) requirement for P2P DMA?** ACS controls whether a PCIe switch forwards peer-to-peer transactions directly between downstream ports or routes them through the Root Complex. For true P2P (bypassing system RAM), ACS must be disabled or configured to allow P2P forwarding. The Linux kernel exposes ACS state via `lspci -vvv | grep ACS`.

- **Why does NVMe use per-queue MSI-X interrupts instead of a single interrupt line?** With one interrupt line, all I/O completions would arrive at one CPU core, creating a bottleneck and requiring locking to dispatch completions to the correct requesting thread. With per-queue MSI-X, each completion interrupts exactly the CPU core that owns that queue, eliminating cross-core wakeups and lock contention entirely.

---

## AI Hardware Connection

- GPU BAR0 register mapping via `ioremap()` is the foundation of every CUDA context; `nvidia.ko` programs GPU page tables through this interface
- GPUDirect Storage uses PCIe P2P to stream dataset batches from NVMe directly into GPU HBM without CPU involvement, eliminating a critical bottleneck in training pipelines
- PCIe P2P between FPGA and GPU enables real-time pre-processing (e.g., radar signal processing on FPGA → feature tensors in GPU memory) with no CPU in the data path
- The XDMA driver exposes FPGA H2C/C2H channels as character devices; inference accelerator prototypes commonly use this interface for tensor transfer
- `nvidia.ko` ioctl paths are how every CUDA API call ultimately reaches the GPU; understanding them is necessary for profiling and debugging anomalous CUDA launch latency
- MSI-X per-queue interrupts on NVMe (and per-stream on GPU) allow pinning completion interrupts to specific CPU cores, eliminating cross-NUMA interrupt overhead in multi-GPU servers
