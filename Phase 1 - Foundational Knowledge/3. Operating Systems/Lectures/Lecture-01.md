# Lecture 1: Modern OS Architecture & the Linux Kernel

## Overview

Every piece of AI hardware — a Jetson inference node, a camera pipeline server, or an autonomous vehicle compute platform — runs on top of an operating system that determines how code reaches hardware. The core challenge this lecture addresses is: how does the OS safely mediate between many competing software components and the physical hardware beneath them? The mental model to carry forward is that the OS is a **trusted referee**: it owns the hardware, enforces rules about who can touch what, and presents clean abstractions so application code never has to know register addresses. For an AI hardware engineer, understanding this architecture means you can trace a slow inference pipeline to a kernel subsystem, debug a crashing camera driver without corrupting the whole system, and make deliberate choices about which kernel version to target for a production deployment.

---

## OS Definition: Three Roles

| Role | Meaning |
|---|---|
| Resource manager | Arbitrates CPU time, memory pages, I/O bandwidth, and network across competing processes |
| Abstraction layer | Presents uniform interfaces (files, sockets, virtual memory) over heterogeneous hardware |
| Protection boundary | Isolates processes from each other and from kernel data structures; enforced in hardware |

Without **protection**, a buggy camera driver corrupts kernel memory. Without **abstraction**, each application must know specific hardware register layouts.

> **Key Insight:** These three roles are inseparable. Abstraction without protection would let any process break another's view of hardware. Protection without abstraction would force every developer to write device-specific code. All three together are what make Linux viable as a platform for complex AI systems with multiple concurrent workloads.

---

## Privilege Levels

When code runs on a CPU, the hardware enforces who is allowed to do what. This is the foundation of the OS **protection boundary**. Think of it as the CPU acting as a security guard: code in **user space** asks for services, the OS grants or denies them, and the hardware makes it impossible to bypass the check.

### x86 Rings

| Ring | Mode | Access | Example occupants |
|---|---|---|---|
| Ring 0 | Kernel | All instructions, I/O ports, MSRs, CR0/CR3 | Linux kernel, device drivers |
| Ring 1–2 | Unused | — | Historical OS/2; unused by Linux |
| Ring 3 | User | Restricted; privileged instruction → GP Fault | Applications, runtimes, Python |

Mode switch: `SYSCALL` instruction → Ring 0; `SYSRET` → Ring 3.

### ARM Exception Levels (AArch64)

| EL | Name | Purpose |
|---|---|---|
| EL0 | User | Applications, TensorRT, ONNX Runtime, ROS2 nodes |
| EL1 | Kernel | Linux kernel, exception handlers, MMU configuration |
| EL2 | Hypervisor | KVM, Xen; controls VM-to-VM memory partitioning |
| EL3 | Secure Monitor | ARM TrustZone, PSCI power management, BCT/BL31 signing |

The two-level EL0/EL1 split is sufficient for most deployments. EL2 adds **virtualization overhead**; EL3 handles **secure world** and is always present on Cortex-A platforms. On Jetson Orin (Cortex-A78AE), the Linux kernel runs at EL1 and NVIDIA MB1/SC7 firmware occupies EL3.

```
AArch64 Exception Level Stack
┌─────────────────────────────────────┐
│  EL3 — Secure Monitor (TrustZone)   │  ← NVIDIA MB1/SC7 firmware
│  ARM TrustZone, PSCI, OTP key mgmt  │
├─────────────────────────────────────┤
│  EL2 — Hypervisor (optional)        │  ← KVM, Xen
│  VM-to-VM memory partitioning       │
├─────────────────────────────────────┤
│  EL1 — Kernel                       │  ← Linux kernel
│  MMU, exception handlers, drivers   │
├─────────────────────────────────────┤
│  EL0 — User                         │  ← TensorRT, ROS2, Python
│  Restricted; must SVC to reach EL1  │
└─────────────────────────────────────┘
```

> **Key Insight:** On a Jetson, your TensorRT inference code runs at EL0 and cannot directly touch the NVDLA hardware registers. It must pass through EL1 (the Linux kernel) via system calls and ioctls. This indirection is what makes it safe to run multiple inference workloads concurrently — the kernel serializes access and prevents one process from corrupting another's DMA buffers.

---

## Linux Kernel Architecture

Linux is a **monolithic kernel with loadable modules**: all core subsystems share a single address space at Ring 0 / EL1 with no IPC overhead between them. Drivers and filesystems compile as `.ko` modules inserted at runtime without rebuilding the kernel.

The monolithic design delivers fast in-kernel calls at the cost of **shared fate** — a crashing driver can panic the whole system. Contrast with **microkernels** (QNX, seL4) where drivers are separate processes; slower IPC, but **crash isolation**.

```
Linux Kernel Internal Architecture (Monolithic)
┌──────────────────────────────────────────────────────────┐
│                    Ring 0 / EL1 (Kernel Space)           │
│                                                          │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌─────────┐  │
│  │Scheduler │  │  Memory  │  │   VFS    │  │  Net    │  │
│  │ (CFS/    │  │ Manager  │  │ (ext4,   │  │ (TCP/IP,│  │
│  │  EEVDF)  │  │ (mm/)    │  │  procfs) │  │  XDP)   │  │
│  └──────────┘  └──────────┘  └──────────┘  └─────────┘  │
│                                                          │
│  ┌──────────────────────────────────────────────────┐    │
│  │  Device Drivers (drivers/)  ~60% of kernel LOC   │    │
│  │  GPU / DRM │ V4L2 Camera │ NVMe │ PCIe │ GPIO    │    │
│  └──────────────────────────────────────────────────┘    │
│                                                          │
│  ┌──────────────────────────────────────────────────┐    │
│  │  arch/ (arm64, x86): MMU, entry.S, IRQ setup     │    │
│  └──────────────────────────────────────────────────┘    │
└──────────────────────────────────────────────────────────┘
              ↑ syscall / interrupt boundary ↑
┌──────────────────────────────────────────────────────────┐
│             Ring 3 / EL0 (User Space)                    │
│   TensorRT  │  camerad  │  modeld  │  Python  │  ROS2    │
└──────────────────────────────────────────────────────────┘
```

### Major Subsystems

| Directory | Subsystem | Responsibility |
|---|---|---|
| `kernel/` | Scheduler, signals, timers | CFS/EEVDF, workqueues, kthreads |
| `mm/` | Memory manager | Page allocator, slab/slub, OOM killer, mmap |
| `drivers/gpu/drm/` | DRM / GPU | Display, render, GEM/PRIME buffer management |
| `drivers/media/` | V4L2 / Media | Camera ISP, video capture pipeline |
| `drivers/char/`, `drivers/block/` | Char / Block | TTY, GPIO, NVMe, MMC |
| `fs/` | VFS | ext4, tmpfs, overlayfs, procfs, sysfs |
| `net/` | Networking | TCP/IP, netfilter, XDP, RDMA |
| `arch/` | CPU-specific | x86, arm64 — exception tables, syscall entry, MMU |
| `include/` | Headers | Shared kernel-wide type definitions |
| `Documentation/` | Docs | ABI contracts, Device Tree bindings, admin guides |

> **Common Pitfall:** When a camera driver crashes on Jetson, it can kernel-panic the entire system because all drivers share Ring 0 space. The fix is to test drivers with IOMMU protection enabled (`iommu=on` in the kernel command line) so that a misbehaving DMA operation hits a fault rather than scribbling over kernel memory.

Now that we understand how the kernel is organized as a monolithic system, let's look at how the kernel manages **versioning** — and why AI platforms are conservative about which version they run.

---

## Kernel Versioning

Format: `major.minor.patch` — e.g., `6.1.57`

| Track | Lifespan | Purpose | Example |
|---|---|---|---|
| Mainline | ~9 weeks per release | New features merge here | 6.13, 6.14 |
| Stable | ~3 months | Fixes only after release | 6.13.y |
| LTS | 2–6 years | Security and critical fixes only | 5.10, 5.15, 6.1, 6.6 |

### Why AI Platforms Pin to LTS

**Board Support Packages** couple tightly to a specific kernel **ABI**. Upgrading to mainline breaks downstream **out-of-tree modules** and vendor driver patches.

| Platform | Kernel base | Notable additions |
|---|---|---|
| Jetson L4T 35.x (JetPack 5) | 5.10 LTS | NVDLA, Argus ISP/VI/CSI, NvMedia, Tegra PCIe |
| Jetson L4T 36.x (JetPack 6) | 6.1 LTS | Orin NvDLA v2, Tegra ISP v5, PCIe Gen4 |
| openpilot Agnos (comma 3X) | Ubuntu LTS-based | Snapdragon camera ISPs, CAN-over-SPI, openpilot services |
| Yocto Kirkstone embedded AI | 5.15 LTS | Stripped BSP; FPGA/NPU out-of-tree modules |
| Yocto Scarthgap embedded AI | 6.6 LTS | EEVDF scheduler; latest nftables, XDP |

> **Key Insight:** The reason NVIDIA ships L4T 36.x based on Linux 6.1 rather than a newer kernel is that the NVDLA v2 driver, Tegra ISP v5, and PCIe Gen4 support all live in out-of-tree patches. Rebasing to a newer mainline kernel would require re-porting and re-validating hundreds of thousands of lines of vendor driver code. LTS stability is a practical engineering constraint, not laziness.

With the kernel version context established, let's look at the two most important runtime inspection interfaces the kernel exposes: `/proc` and `/sys`.

---

## /proc Virtual Filesystem

`/proc` is a **kernel interface** that looks like a filesystem. Files have **no disk representation**; reads invoke kernel functions that format data on demand. Think of it as the kernel's "dashboard" — you read from it the same way you read a file, but what you get back is a **live snapshot** of kernel state.

| Path | Content |
|---|---|
| `/proc/cpuinfo` | Per-CPU: model, frequency, flags (avx512, neon, crypto) |
| `/proc/meminfo` | MemTotal, MemFree, Buffers, Cached, HugePages |
| `/proc/interrupts` | Per-CPU interrupt counts per IRQ line and name |
| `/proc/cmdline` | Kernel boot parameters passed by bootloader |
| `/proc/[pid]/maps` | VMA layout: address range, permissions, backing file |
| `/proc/[pid]/status` | State, VmRSS, threads, capability sets |
| `/proc/[pid]/fd/` | Symlinks to all open file descriptors |
| `/proc/[pid]/wchan` | Kernel function where process is currently sleeping |

> **Common Pitfall:** `/proc/[pid]/maps` shows virtual address ranges, not physical ones. Two processes can both show a mapping at `0x7fff0000` and those are completely independent — different pages in physical RAM. Confusing virtual and physical addresses is a frequent source of debugging errors in DMA and GPU buffer management.

---

## /sys (sysfs)

**sysfs** exports kernel objects — devices, drivers, buses — as a directory tree. Structure mirrors the **kernel object model** rather than process hierarchy. While `/proc` is about processes and kernel state, `/sys` is about **devices and the hardware model**.

| Path | Purpose |
|---|---|
| `/sys/class/net/eth0/` | Interface attributes: speed, mtu, carrier state |
| `/sys/bus/pci/devices/` | PCI devices; vendor/device IDs, resource (BAR) files |
| `/sys/class/thermal/thermal_zone*/temp` | Zone temperature in millidegrees Celsius |
| `/sys/class/gpio/` | GPIO pin export, direction, and value control |
| `/sys/fs/cgroup/` | cgroup v2 hierarchy; resource controllers |
| `/sys/kernel/debug/` | Debugfs: DVFS state, tracing, GPU activity monitors |
| `/sys/firmware/devicetree/base/` | Live Device Tree from running kernel |

On Jetson, `/sys/class/thermal/` exposes CPU, GPU, and SoC thermal zones. Polling during inference detects throttling before it causes latency spikes.

> **Key Insight:** `/sys/class/thermal/thermal_zoneN/temp` is not just a monitoring curiosity — it is a direct signal of whether your inference workload is being throttled by the kernel's thermal governor. If a GPU zone exceeds its trip point, the kernel reduces clock frequency without warning your application. An inference benchmark that ignores thermal readings is not measuring real-world performance.

---

## Kernel Modules

**Modules** run at Ring 0 and have full kernel access. They provide the standard integration path for **device drivers** without rebuilding the kernel.

```bash
insmod my_driver.ko          # load from file; no dependency resolution
rmmod my_driver              # unload by name
modprobe nvidia              # load with dependency resolution (reads modules.dep)
lsmod                        # list loaded modules and usage count
modinfo nvme                 # show parameters, license, firmware requirements
```

### Module Lifecycle in C

```c
static int __init my_init(void) {
    pr_info("my_driver: loaded\n");
    return 0;  /* non-zero = load failure; kernel will not insert the module */
}
static void __exit my_exit(void) {
    pr_info("my_driver: unloaded\n");
    /* release all resources here: IRQs, DMA buffers, device nodes */
}
module_init(my_init);
module_exit(my_exit);
MODULE_LICENSE("GPL");
MODULE_DEVICE_TABLE(of, my_of_match);  /* enables udev auto-load on DT match */
```

`__init` lets the kernel discard initialization code after boot, reclaiming memory. `MODULE_DEVICE_TABLE` generates a `modules.alias` entry so `modprobe` can auto-load when the Device Tree node appears.

The module lifecycle follows a strict sequence when a device is discovered:

1. **DT match**: kernel finds a Device Tree node whose `compatible` string matches `my_of_match`.
2. **udev triggers**: `modules.alias` maps the `compatible` string to the module name; udev calls `modprobe`.
3. **Module loads**: kernel verifies signature (if `CONFIG_MODULE_SIG_FORCE`), maps `.ko` into Ring 0 address space.
4. **`probe()` runs**: the driver's probe function claims hardware resources, registers device nodes.
5. **Device ready**: application code can now open `/dev/my_accel0` and issue ioctls.

**DKMS** (Dynamic Kernel Module Support) rebuilds out-of-tree modules when the kernel is updated — used by the **NVIDIA GPU driver** on development hosts and custom FPGA PCIe drivers.

> **Common Pitfall:** Loading a module compiled against a different kernel version fails with `version magic mismatch`. Always compile modules against the exact kernel headers of the running kernel (`uname -r`). DKMS automates this rebuild, but manual module builds break silently after a `apt upgrade` that bumps the kernel version.

---

## Kernel Source Layout

| Directory | Contents |
|---|---|
| `kernel/` | Core scheduler, signals, timers, printk, kprobes |
| `mm/` | Buddy allocator, slab/slub, vmalloc, OOM killer |
| `drivers/` | All device drivers — approximately 60% of kernel source lines |
| `arch/arm64/`, `arch/x86/` | Platform entry (head.S, entry.S), IRQ setup, NUMA |
| `fs/` | Filesystems: ext4, xfs, tmpfs, procfs, overlayfs |
| `include/` | Kernel headers; `include/linux/` for cross-arch types |
| `net/` | TCP, UDP, netfilter, socket layer, XDP |
| `Documentation/` | `Documentation/ABI/` defines stable sysfs interfaces |

---

## Linux on AI Platforms

Understanding kernel architecture pays off when you navigate vendor-specific kernel trees for real hardware. Each platform below has custom subsystems that only exist because of the module and subsystem architecture described above.

### Jetson L4T

NVIDIA's **Linux for Tegra** is a **downstream LTS fork** with patches for NVDLA, VIC (Video Image Compositor), ISP (Image Signal Processor), NvMedia, and Tegra PCIe IOMMU. L4T 35.x is based on 5.10; L4T 36.x rebased to 6.1. These patches are **absent from mainline**; they live in `drivers/gpu/`, `drivers/media/`, and `arch/arm64/` of the L4T tree.

### openpilot Agnos

Comma's **AGNOS** is a **forked and custom-modified Linux** that runs on comma 3X and comma four (Snapdragon-based). It is built for one practical use case: **running openpilot on the road**. The kernel ([agnos-kernel-sdm845](https://github.com/commaai/agnos-kernel-sdm845)) is forked from Linux and **custom developed** with patches for road-facing and driver-monitoring camera ISPs, CAN-over-SPI, and power management for always-on vehicle operation. The OS image is produced by [agnos-builder](https://github.com/commaai/agnos-builder) (boot chain, device tree, userspace). `modeld`, `camerad`, and `controlsd` depend on stable V4L2 and SocketCAN ABIs; comma **pins the kernel** so upstream changes do not break camera ISP register maps or driver interfaces. To see how each OS lecture topic maps to **what was changed in the AGNOS fork** for openpilot on the road, see the [AGNOS Guide — OS Lectures ↔ Development Changes](../../../Phase%205%20-%20Advanced%20Topics%20and%20Specialization/Track%20E%20-%20Autonomous%20Vehicles/2.%20openpilot%20Reference%20Stack/agnos/Guide.md#os-lectures--agnos-development-changes-fork--custom-mods-for-openpilot) in Phase 5 (Autonomous Driving specialization).

### Yocto for Custom Boards

Production inference nodes use **Yocto** to build minimal, reproducible kernel + rootfs images. Only required drivers are compiled; **attack surface** and boot time are minimized for safety-critical deployment.

---

## Summary

| Component | Location in kernel tree | Purpose |
|---|---|---|
| Process scheduler | `kernel/sched/` | CPU time allocation across tasks |
| Memory manager | `mm/` | Virtual memory, page and slab allocators |
| VFS | `fs/` | Uniform file API over all filesystem types |
| Device drivers | `drivers/` | Hardware abstraction and resource management |
| Architecture code | `arch/arm64/`, `arch/x86/` | Platform entry, MMU, exception tables |
| Network stack | `net/` | Protocols, socket layer, XDP |
| IPC | `kernel/` (futex, signal), `ipc/` | Inter-process communication primitives |

### Conceptual Review

- **Why is Linux monolithic rather than microkernel?** In-kernel calls between subsystems (scheduler → memory manager → driver) have zero IPC cost. A monolithic design delivers throughput required for high-bandwidth GPU and camera workloads; the tradeoff is that a driver bug can crash the whole system.
- **Why do AI platforms pin to LTS kernels?** Board Support Packages include hundreds of out-of-tree driver patches (NVDLA, ISP, PCIe IOMMU). These patches cannot be instantly rebased to a newer mainline kernel. LTS gives 2–6 years of security fixes without requiring a full BSP re-port.
- **What is `/proc` and why does it look like a filesystem?** `/proc` is a virtual filesystem backed entirely by kernel functions, not disk storage. Reads trigger kernel code that formats live data. The filesystem metaphor means existing tools (`cat`, `grep`, shell scripts) can query kernel state without special APIs.
- **What is a kernel module, and why would a driver be one?** A module is a `.ko` file that loads into Ring 0 at runtime without rebuilding the kernel. Device drivers are modules so new hardware can be supported without a full kernel rebuild and reboot.
- **What does `MODULE_DEVICE_TABLE` do?** It generates an entry in `modules.alias` that maps a Device Tree `compatible` string (or PCI vendor/device ID) to a module name, enabling udev to automatically `modprobe` the correct driver when hardware is detected.
- **What is sysfs (`/sys`) and how does it differ from `/proc`?** sysfs organizes kernel objects (devices, buses, drivers) in a hierarchy mirroring the kernel object model. `/proc` is process-oriented and kernel-state-oriented. Both are virtual filesystems with no disk backing.

---

## AI Hardware Connection

- L4T downstream patches add NVDLA, VIC, and ISP drivers absent from mainline; production Jetson deployments rely on these for hardware-accelerated inference and camera pipelines — understanding the kernel source layout locates them immediately when debugging initialization failures.
- `/sys/class/thermal/thermal_zoneN/temp` is the primary interface for detecting GPU and CPU throttling on Jetson; inference benchmarks should poll this to correlate latency spikes with temperature events.
- Custom FPGA PCIe accelerators require out-of-tree `.ko` modules compiled against the target LTS kernel; DKMS manages rebuilds across point releases automatically.
- openpilot Agnos pins its kernel to maintain camera ISP register map compatibility with comma 3X hardware; upstream kernel changes would break the V4L2 subdevice interface.
- `/proc/interrupts` is the first diagnostic for interrupt storm conditions in high-framerate camera pipelines — per-CPU counts per IRQ line reveal unbalanced distribution before it becomes a throughput problem.
- The monolithic architecture means an FPGA DMA driver crash kernel-panics the entire system; production AI hardware deployments invest in IOMMU protection and driver fault injection testing to contain faults before deployment.
