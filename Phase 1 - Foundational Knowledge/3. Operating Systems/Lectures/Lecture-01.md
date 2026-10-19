# Lecture 1: Modern OS Architecture & the Linux Kernel

## Overview

Most deployed AI systems — a Jetson inference node, a camera pipeline server, or an autonomous vehicle compute platform — runs on top of an operating system that determines how code reaches hardware. The core challenge this lecture addresses is: how does the OS safely mediate between many competing software components and the physical hardware beneath them? The mental model to carry forward is that the OS is a **trusted referee**: it owns the hardware, enforces rules about who can touch what, and presents clean abstractions so application code never has to know register addresses. For an AI hardware engineer, understanding this architecture means you can trace a slow inference pipeline to a kernel subsystem, understand why a crashing kernel driver can affect the whole system, and make deliberate choices about which kernel version to target for a production deployment.

---

## OS Definition: Three Roles

| Role | Meaning |
|---|---|
| Resource manager | Arbitrates CPU time, memory pages, I/O bandwidth, and network across competing processes |
| Abstraction layer | Presents uniform interfaces (files, sockets, virtual memory) over heterogeneous hardware |
| Protection boundary | Isolates processes from each other and from kernel data structures; enforced in hardware |

User/kernel protection isolates ordinary applications; a buggy in-kernel driver can still corrupt kernel memory. Without **abstraction**, each application must know specific hardware register layouts.

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
| EL3 | Secure Monitor | Secure monitor and platform firmware services; signing is a separate build/provisioning operation |

EL0 and EL1 describe the common application/kernel distinction. EL2 and EL3 availability and use depend on the architecture and firmware configuration; virtualization can also change where a host kernel executes. Jetson boot components such as MB1 must not be treated as names for ARM exception levels.

```
AArch64 Exception Level Stack
┌─────────────────────────────────────┐
│  EL3 — Secure Monitor (TrustZone)   │  ← secure monitor / TF-A-style firmware
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

> **Key Insight:** On a Jetson, your TensorRT inference code runs at EL0 and cannot directly touch the NVDLA hardware registers. It must pass through EL1 (the Linux kernel) via system calls and ioctls. This indirection is what makes it safe to run multiple inference workloads concurrently — the kernel and driver mediate access and establish mappings; their correctness and device isolation determine the actual boundary.

---

## Linux Kernel Architecture

Linux is a **monolithic kernel with loadable modules**: all core subsystems share a single address space at Ring 0 / EL1 with no IPC overhead between them. Drivers and filesystems compile as `.ko` modules inserted at runtime without rebuilding the kernel.

The monolithic design delivers fast in-kernel calls at the cost of **shared fate** — a crashing driver can panic the whole system. Contrast with **microkernels** (QNX, seL4) where drivers are separate processes; explicit IPC and **process-level isolation**, with costs depending on the design.

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
│  │  Device Drivers (drivers/)                       │    │
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
| `drivers/tty/`, `drivers/gpio/`, `drivers/nvme/`, `drivers/mmc/` | Device subsystems | TTY, GPIO, NVMe, MMC implementations |
| `fs/` | VFS | ext4, tmpfs, overlayfs, procfs, sysfs |
| `net/` | Networking | TCP/IP, netfilter, XDP, RDMA |
| `arch/` | CPU-specific | x86, arm64 — exception tables, syscall entry, MMU |
| `include/` | Headers | Shared kernel-wide type definitions |
| `Documentation/` | Docs | ABI contracts, Device Tree bindings, admin guides |

> **Driver isolation:** An IOMMU can restrict device DMA when mappings and the platform support it; it cannot contain arbitrary CPU writes by a buggy kernel driver. There is no universal `iommu=on` switch for every SoC. Use the target DMA API, validate lifetimes and bounds, and test failure paths.

Now that we understand how the kernel is organized as a monolithic system, let's look at how the kernel manages **versioning** — and why AI platforms are conservative about which version they run.

---

## Kernel Versioning

`uname -r` identifies the running kernel release. Upstream mainline releases, stable branches, long-term branches, distribution kernels, and vendor BSP kernels have different maintenance policies. Check the [kernel.org release schedule](https://www.kernel.org/category/releases.html) and the BSP supplier's support window; an LTS label is not a universal promise of six years of fixes.

A BSP couples kernel source/configuration, firmware, DTBs, modules, and userspace interfaces. Linux aims to preserve userspace ABI; internal driver APIs and module compatibility can change. A kernel upgrade therefore requires rebuilding and validating the relevant driver stack, not assuming that every ordinary userspace application breaks.

| Pinned family | Kernel baseline | Scope |
|---|---|---|
| Jetson Linux R35 / JetPack 5 | 5.10 | Legacy supported configurations depend on the exact release |
| Jetson Linux R36 / JetPack 6 | 5.15 | Do not relabel it as Linux 6.1 |
| Newer Jetson releases | Check that exact release's notes | Board, firmware, module and userspace compatibility all matter |
| Yocto machine | Selected by its kernel recipe/BSP | A Yocto release does not force one kernel on every board |
| AGNOS | Depends on the pinned hardware/image | Do not generalize one Snapdragon kernel to every generation |

Use the [Jetson release archive](https://developer.nvidia.com/embedded/jetson-linux-archive) to identify a real BSP. The course's current Jetson platform chapters distinguish newer releases from the historical R35/R36 exercises.

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

> **Common Pitfall:** `/proc/[pid]/maps` shows virtual address ranges, not physical ones. Two processes can both show a mapping at `0x7fff0000` without revealing whether the underlying pages differ or are intentionally shared. Confusing virtual and physical addresses is a frequent source of debugging errors in DMA and GPU buffer management.

---

## /sys (sysfs)

**sysfs** exports kernel objects — devices, drivers, buses — as a directory tree. Structure mirrors the **kernel object model** rather than process hierarchy. While `/proc` is about processes and kernel state, `/sys` is about **devices and the hardware model**.

| Path | Purpose |
|---|---|
| `/sys/class/net/eth0/` | Interface attributes: speed, mtu, carrier state |
| `/sys/bus/pci/devices/` | PCI devices; vendor/device IDs, resource (BAR) files |
| `/sys/class/thermal/thermal_zone*/temp` | Zone temperature in millidegrees Celsius |
| `/sys/class/gpio/` | Legacy GPIO sysfs API; new applications normally use the GPIO character-device API |
| `/sys/fs/cgroup/` | Separate cgroup filesystem commonly mounted below `/sys` |
| `/sys/kernel/debug/` | Separate optional debugfs mount; unstable diagnostic interfaces |
| `/sys/firmware/devicetree/base/` | Live Device Tree from running kernel |

On Jetson, thermal zones can expose temperatures. Temperature alone does not prove throttling: correlate named zones, trip/cooling state, clocks, power mode, and the application timeline. Available files depend on the BSP.

---

## Kernel Modules

**Modules** run at Ring 0 and have full kernel access. They provide the standard integration path for **device drivers** without rebuilding the kernel.

Administrative command examples for a deliberately prepared module experiment; the foundation lab only builds a module and does not load it.

```bash
insmod my_driver.ko          # load from file; no dependency resolution
rmmod my_driver              # unload by name
modprobe nvidia              # load with dependency resolution (reads modules.dep)
lsmod                        # list loaded modules and usage count
modinfo nvme                 # show parameters, license, firmware requirements
```

### Module Lifecycle in C

Reading fragment: include the target kernel headers and define the matching table before using this in a real module.

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

`__init` lets the kernel discard initialization code after successful built-in initialization or module initialization, reclaiming memory. `MODULE_DEVICE_TABLE` generates a `modules.alias` entry so `modprobe` can auto-load when the Device Tree node appears.

The module lifecycle can follow this common sequence when a device is discovered:

1. **DT match**: kernel finds a Device Tree node whose `compatible` string matches `my_of_match`.
2. **udev triggers**: `modules.alias` maps the `compatible` string to the module name; udev calls `modprobe`.
3. **Module loads**: kernel verifies signature (if `CONFIG_MODULE_SIG_FORCE`), maps `.ko` into Ring 0 address space.
4. **`probe()` runs**: the driver's probe function claims hardware resources, registers device nodes.
5. **Device ready**: application code can now open `/dev/my_accel0` and issue ioctls.

**DKMS** (Dynamic Kernel Module Support) rebuilds out-of-tree modules when the kernel is updated — used by the **NVIDIA GPU driver** on development hosts and custom FPGA PCIe drivers.

> **Common Pitfall:** Loading a module compiled against a different kernel version fails with `version magic mismatch`. Always compile modules against the exact kernel headers of the running kernel (`uname -r`). DKMS automates this rebuild, but manual module builds may fail to load after a `apt upgrade` that bumps the kernel version.

---

## Kernel Source Layout

| Directory | Contents |
|---|---|
| `kernel/` | Core scheduler, signals, timers, printk, kprobes |
| `mm/` | Buddy allocator, slab/slub, vmalloc, OOM killer |
| `drivers/` | Device driver implementations; subsystem layout varies |
| `arch/arm64/`, `arch/x86/` | Platform entry (head.S, entry.S), IRQ setup, NUMA |
| `fs/` | Filesystems: ext4, xfs, tmpfs, procfs, overlayfs |
| `include/` | Kernel headers; `include/linux/` for cross-arch types |
| `net/` | TCP, UDP, netfilter, socket layer, XDP |
| `Documentation/` | `Documentation/ABI/` defines stable sysfs interfaces |

---

## Linux on AI Platforms

Understanding kernel architecture pays off when you navigate vendor-specific kernel trees for real hardware. Each platform below has custom subsystems that only exist because of the module and subsystem architecture described above.

### Jetson L4T

Jetson Linux combines a kernel, board firmware, modules, and userspace components. Some Tegra support is upstream and some is downstream; inspect the selected release instead of asserting that every GPU, ISP, or PCIe function is absent from mainline. R35 uses a 5.10 baseline and R36 a 5.15 baseline.

### openpilot AGNOS

AGNOS is a target-specific OS image built by [agnos-builder](https://github.com/commaai/agnos-builder). Use the course's [pinned AGNOS study](../../../Phase%205%20-%20Advanced%20Topics%20and%20Specialization/Track%20E%20-%20Autonomous%20Vehicles/2.%20openpilot%20Reference%20Stack/agnos/Guide.md) to associate an image and kernel with the actual device. The historical `sdm845` kernel is not evidence about every newer hardware generation, and openpilot's vehicle interface is not universally a SocketCAN or CAN-over-SPI path.

### Yocto for custom boards

Yocto can build a selected kernel and rootfs from versioned metadata. Image size, reproducibility, security, and boot behavior must be tested; they do not follow automatically from using Yocto. [Lecture 25](Lecture-25.md) connects this course to an actual native package and pinned metadata exercise.

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

### Conceptual review and practice

**Why pin a BSP?** Its kernel, modules, firmware, DTB, and userspace components must form a tested combination. Pinning identifies that combination; updates still need a maintenance and validation plan.

**Can one driver's bug crash Linux?** Yes. In-kernel drivers share privileged memory. An IOMMU constrains device accesses under its mapping policy; it does not isolate all kernel code.

**Does the same virtual address in two processes mean shared data?** No. It identifies an address in each process's mapping. Shared backing requires additional evidence.

On a Linux host, run `uname -r`, read `/proc/self/maps`, and inspect `/proc/self/status`. Identify the running kernel, one mapped library, and the allowed CPU list. These are read-only observations. They do not reveal physical buffer addresses or certify a device's DMA safety.
