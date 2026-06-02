# Lecture Note 08 (L17, L18, L19, L20): Driver Model & Device Tree; Char Drivers & V4L2; io_uring & Zero-Copy; PCIe, NVMe & GPU Drivers

**Combines:** Lecture L17 (Device Driver Model & Device Tree), L18 (Character Drivers, Interrupt-Driven I/O & V4L2), L19 (io_uring, DMA-BUF & Zero-Copy), L20 (PCIe, NVMe & GPU Driver Architecture).

---

## How This Note Is Organized

1. **Part 1 — Driver model & Device Tree:** Bus, device, driver; match/probe; platform devices; DTS/DTB; compatible, reg, interrupts; of_* API.
2. **Part 2 — Character drivers & V4L2:** chrdev registration; file_operations (read, write, ioctl, mmap, poll); ioctl safety; interrupt-driven I/O; V4L2 pipeline and DMA-BUF.
3. **Part 3 — Modern I/O & zero-copy:** io_uring (SQ/CQ rings, batching, SQPOLL); DMA-BUF sharing; sendfile, zero-copy pipelines.
4. **Part 4 — PCIe, NVMe & GPU:** PCIe topology and BARs; NVMe queues and driver stack; GPU driver layout; peer-to-peer DMA.

---

# Part 1: Linux Driver Model & Device Tree

**Context:** Hardware is diverse; the **driver model** gives a single framework: buses enumerate devices, **match them to drivers**, call probe/remove. On embedded SoCs, hardware is described in the **Device Tree** (DTS → DTB) passed at boot; platform devices are created from it.

---

## Bus, Device, Driver

- **Bus** (`struct bus_type`): Enumerates devices and matches them to drivers (PCIe, USB, I2C, SPI, **platform**).
- **Device** (`struct device`): One hardware instance. **Driver** (`struct device_driver`): Code that manages a device type. Bus holds device and driver lists; **match** runs; on match it calls **driver->probe(device)**; on unbind/remove, **driver->remove(device)**.
- Driver gets resources (MMIO, IRQ, clocks) from the **device object, not hardcoded** — **same driver binary for different boards** via different Device Tree.

---

## Platform Devices & Device Tree

- SoC peripherals (UART, I2C, CSI, accelerator) are not self-describing like PCIe. They are **platform devices**; description comes from **Device Tree** or ACPI.
- **DTS** (source) → **dtc** → **DTB** (binary). Bootloader passes DTB address to kernel (e.g. ARM64 x0). Kernel parses DTB; `of_platform_populate()` creates **platform_device** for nodes with `status = "okay"`.
- **Matching:** Driver has `of_match_table` with `compatible` strings. Node’s `compatible` must match. Example: `compatible = "vendor,mydev-v2"`; driver has `{ .compatible = "vendor,mydev-v2" }, {}`.
- **Resources in DTS:** `reg` (address, size), `interrupts`, `clocks`, `status`. Driver gets them in probe via **of_*** API: `of_iomap`, `of_irq_get`, `of_get_property`, etc.
- **MODULE_DEVICE_TABLE(of, ...)** embeds table so udev can load the module when a matching node appears.

---

# Part 2: Character Drivers, Interrupt-Driven I/O & V4L2

**Context:** A **character driver** exposes a file in `/dev/`; userspace uses read, write, ioctl, mmap, poll. **Interrupt-driven I/O** lets hardware signal when data is ready instead of polling. **V4L2** is the standard subsystem for cameras and video.

---

## Character Device Registration

- **alloc_chrdev_region** (or static); **cdev_init** with **file_operations**; **cdev_add**; **class_create**; **device_create** → `/dev/mydev0`.
- **file_operations:** open, release, read, write, **unlocked_ioctl**, mmap, poll. Each syscall dispatches to the corresponding function.

---

## ioctl & Safety

- **ioctl(fd, cmd, arg)** for device-specific commands. Macros: `_IO`, `_IOR`, `_IOW` (magic, number, direction, type). In kernel: **never** dereference user pointer; use **copy_from_user** / **copy_to_user** and check return. Casting `(struct foo *)arg` without copy is a security bug (TOCTOU, invalid pointer).

---

## mmap in Drivers

- **mmap** maps kernel or device memory into user VA. Enables zero-copy access to DMA buffers (one map, then read/write in userspace). Use **remap_pfn_range** for contiguous physical (DMA, MMIO); **pgprot_noncached** for MMIO/DMA output so CPU does not cache. **vm_insert_page** / **vm_ops->fault** for non-contiguous or demand-mapped regions.

---

## Interrupt-Driven I/O

- Instead of polling, device raises **IRQ** when data is ready. Driver: top half (ISR) acknowledges hardware, queues work or signals wait queue; bottom half or process context drains data. **wait_queue_head_t**; `wake_up_interruptible()` to unblock **read()**. Avoids busy-wait and reduces latency.

---

## V4L2 (Video4Linux2)

- Subsystem for capture/display: devices under `/dev/video*`. **Video device** → **buffer queue**; userspace enqueues buffers (e.g. **V4L2_MEMORY_MMAP** or **V4L2_MEMORY_DMABUF**), starts streaming; driver fills buffers (capture) or consumes them (output); **ioctl** for format, crop, buffer management. **DMA-BUF** path: userspace passes fd; driver uses that buffer for capture/output — zero-copy with GPU/other subsystems.

---

# Part 3: Modern I/O — io_uring, DMA-BUF & Zero-Copy

**Context:** Traditional read/write pays **syscall and copy cost per operation**. At high IOPS or high bandwidth, **io_uring reduces syscalls**; **DMA-BUF and zero-copy** techniques avoid copies between kernel and devices.

---

## io_uring (Linux 5.1+)

- **Two shared rings:** **SQ (Submission Queue)** — application writes SQE descriptors; kernel reads. **CQ (Completion Queue)** — kernel writes CQE; application reads. Both mmap’d; can submit and reap without a syscall per op (batch submit; poll CQ).
- **Flow:** App prepares SQE(s) (e.g. io_uring_prep_read), optionally batches; **io_uring_submit()** (one syscall for many ops); kernel processes async; kernel pushes CQE; app polls or waits for CQE, then **io_uring_cqe_seen()**.
- **IORING_SETUP_SQPOLL:** Kernel thread drains SQ; submit path can be **zero-syscall**. **IORING_SETUP_IOPOLL:** Kernel polls for completion (low latency). **Fixed buffers** avoid get_user_pages per op.
- **Operations:** read, write, send, recv, accept, connect, fsync, splice, openat, statx, etc. **liburing** simplifies setup and use. At high IOPS, io_uring greatly reduces CPU and syscall count vs read/write.

---

## DMA-BUF & Zero-Copy Pipelines

- **DMA-BUF** (see Lecture-Note-04): One buffer shared across driver, GPU, camera, display via fd. **Zero-copy pipeline:** Camera driver exports DMA-BUF fd → userspace passes to CUDA/display; same physical pages throughout; sync with fences.
- **sendfile():** Kernel copies from file to socket (or between fds) without bouncing to userspace — reduces copies in file serving. **VisionIPC-style:** Shared memory (e.g. mmap of shared region or DMA-BUF) between processes; producer writes frames, consumer reads; no copy.

---

# Part 4: PCIe, NVMe & GPU Driver Architecture

**Context:** **PCIe** is the standard interconnect for GPUs, NVMe, NICs, FPGAs. **Topology** (root complex → root ports → switches → endpoints) and **lane count determine bandwidth**. NVMe runs on PCIe; GPU drivers sit on top of PCIe and expose CUDA/OpenCL.

---

## PCIe Topology & Discovery

- **Hierarchy:** Root Complex → Root Ports → PCIe Switches → Endpoints (GPU, NVMe, NIC, FPGA). **Lane bandwidth:** Gen3 ~1 GB/s/lane; Gen4 ~2; Gen5 ~4. x16 ≈ 32/64/128 GB/s bidirectional.
- **Discovery:** BIOS/kernel walks bus; reads **config space** (Vendor ID, Device ID, BARs, capabilities). **BARs (Base Address Registers):** Device declares MMIO size; BIOS/kernel assigns physical base; kernel **ioremap**’s BAR for register access. **GPU BAR0:** control/registers; **BAR1:** VRAM aperture (Resizable BAR can expose full VRAM to CPU).

---

## PCIe DMA & Peer-to-Peer

- Devices are bus masters; DMA to system RAM (and possibly each other). Kernel uses **dma_map_sg** etc.; **IOMMU** translates IOVA→PA. **Peer-to-peer (P2P):** Two devices on the **same** PCIe switch can DMA to each other without going through system RAM — lower latency, no RAM bandwidth. Requires P2P mapping support and often IOMMU/ACS configuration. **GPUDirect Storage:** NVMe → GPU memory via P2P when on same switch.

---

## NVMe

- **NVMe** = PCIe-native protocol for SSDs. Low latency (~100 µs vs ms for SATA); many queues (e.g. 64K queues × 64K commands). **MSI-X:** one vector per queue; map queues to CPUs for NUMA. **Linux stack:** nvme_core + transport (nvme.ko); **blk-mq** (multi-queue block layer); io_uring or libaio submits to blk-mq; **O_DIRECT** bypasses page cache for raw throughput.

---

## GPU Driver Architecture

- **Userspace:** CUDA/OpenCL runtime; ioctl to kernel driver. **Kernel driver:** Manages GPU VM (address space), submissions (command buffers), DMA (allocations, mapping), interrupts (completion). **Memory:** Allocate VRAM; map for CPU (BAR or GMMU); share with other devices via DMA-BUF. **Resizable BAR:** Full VRAM visible to CPU; important for zero-copy and GPUDirect.

---

## Summary Tables

**Driver model:** Bus matches device to driver → probe; resources from DT/ACPI; platform_driver + of_match_table for SoC.

**Char driver:** fops (open, read, write, ioctl, mmap, poll); copy_from_user/copy_to_user for ioctl; remap_pfn_range for mmap; wait_queue for interrupt-driven read.

**io_uring:** SQ/CQ rings; batch submit; poll CQ; SQPOLL for zero-syscall submit; IOPOLL for low-latency completion.

**PCIe:** Topology and lanes set bandwidth; BARs for MMIO/VRAM; P2P on same switch; NVMe = PCIe + blk-mq; GPU = PCIe + kernel driver + userspace ioctl.

---

## AI Hardware Connection

- **Device Tree** describes camera, NPU, and accelerator nodes on Jetson/custom SoC; one driver supports multiple boards via compatible strings. **V4L2 + DMA-BUF** for camera → GPU zero-copy; **ioctl** with copy_from_user for control.
- **io_uring** for high-throughput storage and network in training/data pipelines; **DMA-BUF** for camera–inference–display pipeline without copies.
- **PCIe topology** and **nvidia-smi topo** for placing jobs and memory on the right socket; **P2P** for NVMe→GPU and GPU↔GPU when on same switch. **Resizable BAR** for full VRAM access and GPUDirect Storage.

---

*Combines Lectures L17, L18, L19, L20 (Driver Model & Device Tree; Char Drivers & V4L2; io_uring & Zero-Copy; PCIe, NVMe & GPU Drivers).*
