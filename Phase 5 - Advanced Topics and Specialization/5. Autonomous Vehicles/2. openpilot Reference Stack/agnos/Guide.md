# AGNOS: Learn with the Operating System Course

<div class="course-identity auto-course" style="--course-accent: #0f766e; --course-accent-rgb: 15, 118, 110;" markdown="1">
<div class="course-identity__icon">ALWT</div>
<div markdown="1">
<p class="course-identity__eyebrow">Deep Dive · Specialization</p>
<p class="course-identity__title">Specialized course identity for AGNOS: Learn with the Operating System Course.</p>
<p class="course-identity__meta">Artifact: specialization case study · Measure: performance, reliability, role fit</p>
</div>
</div>


**Goal:** Use the [Operating Systems](../../../../Phase%201%20-%20Foundational%20Knowledge/3.%20Operating%20Systems/Guide.md) course (Phase 1) to understand AGNOS—the **forked and custom-modified Linux** that runs **openpilot on the road** on comma 3X and comma four. The lectures map to both where code lives and to the **development changes** comma made in the fork for this practical use case.

---

## What AGNOS Is: Forked Linux + Custom Development for openpilot on the Road

AGNOS is **not** a generic Linux distro. It is:

1. **A fork of the Linux kernel** — [agnos-kernel-sdm845](https://github.com/commaai/agnos-kernel-sdm845) is based on the Linux kernel (Android/common baseline for SDM845), then **custom modified and developed** by comma for their hardware and for openpilot.
2. **A custom-built OS image** — [agnos-builder](https://github.com/commaai/agnos-builder) produces the full stack: that kernel, boot chain (XBL, ABL, etc.), device tree, initramfs, and Ubuntu-based userspace.
3. **Built for one practical use case** — **running openpilot in the car, on the road.** Camera pipelines, CAN bus, real-time control loops, and inference all depend on this OS. The development changes in the fork (driver patches, device tree, boot config, scheduler behavior) are there so that openpilot can meet latency and reliability requirements in production.

So when you study the OS lectures and then look at AGNOS, you are looking at **how a real team forked Linux and changed it** to support a specific product (openpilot on comma devices). The table below ties each lecture to both **where** that topic appears in the tree and **what kind of development change** in AGNOS relates to it.

---

## Git History by Function (from Fork Date) → OS Lecture

The [agnos-kernel-sdm845](https://github.com/commaai/agnos-kernel-sdm845) repo was created **2019-08-26** and is based on Qualcomm/Android msm-4.9 (SDM845). Comma’s changes are layered on top. Below, commit history is grouped by **function / area** (folder-based), not by individual commit. Each area maps to the relevant OS lecture.

| Function / area | Typical paths | Example changes (from commit messages) | OS lecture |
|------------------|---------------|----------------------------------------|------------|
| **Camera (msm/camera)** | `drivers/media/platform/msm/`, `techpack/` | Expose IFE PHY_NUM_SEL, workqueues on sysfs, high-priority WQ, unset WQ_UNBOUND; mclk drive strength; ICP enable; Thundercomm camerad updates; Bantian tuning | [18](../../../../Phase%201%20-%20Foundational%20Knowledge/3.%20Operating%20Systems/Lectures/Lecture-18.md) V4L2, [6](../../../../Phase%201%20-%20Foundational%20Knowledge/3.%20Operating%20Systems/Lectures/Lecture-06.md) workqueue priority |
| **MIPI / display (dsi)** | `drivers/gpu/drm/msm/dsi-staging/`, `drivers/video/` | MIPI DCS debug, TE line init, 60Hz jitter, panel init in heat; brightness sysfs; lcd3v3 regulator; Mate 10 lite / Tizi / Bantian display bringup | [18](../../../../Phase%201%20-%20Foundational%20Knowledge/3.%20Operating%20Systems/Lectures/Lecture-18.md) character drivers |
| **Touch** | `drivers/input/touchscreen/` | Hynitron, Samsung clones, touch count, IRQ retry, firmware flasher; Mici bringup | [18](../../../../Phase%201%20-%20Foundational%20Knowledge/3.%20Operating%20Systems/Lectures/Lecture-18.md) interrupt-driven I/O, [3](../../../../Phase%201%20-%20Foundational%20Knowledge/3.%20Operating%20Systems/Lectures/Lecture-03.md) IRQ |
| **Device Tree (DTS)** | `arch/arm64/boot/dts/` | Remove comma_tici.dts, dts cleanup, move to device tree; support sdm845, sdm v2; Mici dtsi; dts merge for SDM845 MTP | [5](../../../../Phase%201%20-%20Foundational%20Knowledge/3.%20Operating%20Systems/Lectures/Lecture-05.md), [17](../../../../Phase%201%20-%20Foundational%20Knowledge/3.%20Operating%20Systems/Lectures/Lecture-17.md) |
| **Boot / defconfig** | `arch/arm64/configs/`, `init/` | Slim tici_defconfig for boot time; fix XBL support; fix reboot | [5](../../../../Phase%201%20-%20Foundational%20Knowledge/3.%20Operating%20Systems/Lectures/Lecture-05.md) |
| **SPI (CAN-over-SPI)** | `drivers/spi/` | spidev bufsiz 8192; spi-geni-qcom delay_usecs | [18](../../../../Phase%201%20-%20Foundational%20Knowledge/3.%20Operating%20Systems/Lectures/Lecture-18.md), openpilot CAN |
| **Storage / block** | `drivers/scsi/`, `block/`, `fs/` | NVMe APST revert, NVMe regulator; sdcard; jbd2 upstream fixes; squashfs, cramfs | [20](../../../../Phase%201%20-%20Foundational%20Knowledge/3.%20Operating%20Systems/Lectures/Lecture-20.md), [21](../../../../Phase%201%20-%20Foundational%20Knowledge/3.%20Operating%20Systems/Lectures/Lecture-21.md), [22](../../../../Phase%201%20-%20Foundational%20Knowledge/3.%20Operating%20Systems/Lectures/Lecture-22.md) |
| **Scheduler / workqueues** | `kernel/sched/`, workqueue usage | Camera high-priority WQ, unset WQ_UNBOUND; expose WQ on sysfs | [6](../../../../Phase%201%20-%20Foundational%20Knowledge/3.%20Operating%20Systems/Lectures/Lecture-06.md), [8](../../../../Phase%201%20-%20Foundational%20Knowledge/3.%20Operating%20Systems/Lectures/Lecture-08.md) |
| **Power / thermal** | `drivers/power/`, `drivers/thermal/` | Thermal probes on C4; QPNP_FG_GEN3; CPU freq governor cap; mici thermal sensors | [15](../../../../Phase%201%20-%20Foundational%20Knowledge/3.%20Operating%20Systems/Lectures/Lecture-15.md) DMA/power |
| **Network / WiFi** | `net/`, `drivers/net/` | WiFi log level; build wifi in main kernel; RNDIS; CONFIG_IFB, NETEM, TTL; MAC from SOC serial | [4](../../../../Phase%201%20-%20Foundational%20Knowledge/3.%20Operating%20Systems/Lectures/Lecture-04.md) networking |
| **Kernel config / cgroups** | `Kconfig`, `kernel/cgroup/` | enable memory control for cgroups; audio, uart, logitech modules | [5](../../../../Phase%201%20-%20Foundational%20Knowledge/3.%20Operating%20Systems/Lectures/Lecture-05.md), [23](../../../../Phase%201%20-%20Foundational%20Knowledge/3.%20Operating%20Systems/Lectures/Lecture-23.md) |
| **Platform / misc drivers** | `drivers/`, `arch/arm64/` | SOM id pins driver; USB serial PID; hostname (comma/tici); nfs | [17](../../../../Phase%201%20-%20Foundational%20Knowledge/3.%20Operating%20Systems/Lectures/Lecture-17.md), [2](../../../../Phase%201%20-%20Foundational%20Knowledge/3.%20Operating%20Systems/Lectures/Lecture-02.md) |
| **Upstream / Qualcomm merges** | various | msm-4.9 merges, dts overlay merge, camera/mdss/ipa fixes | Base; many lectures |

*To inspect history yourself: `git log --oneline -- drivers/media/` (camera), `git log --oneline -- arch/arm64/boot/dts/` (DT), etc. Repo: [agnos-kernel-sdm845](https://github.com/commaai/agnos-kernel-sdm845).*

---

## Repositories (Clone and Study)

| Repo | Purpose | Link |
|------|--------|------|
| **agnos-kernel-sdm845** | Linux kernel for SDM845 (Snapdragon 845) modules. Camera ISP, CAN-over-SPI, power management, scheduler, drivers. | [commaai/agnos-kernel-sdm845](https://github.com/commaai/agnos-kernel-sdm845) |
| **agnos-builder** | Builds AGNOS: kernel + system image, initramfs, boot artifacts, userspace. Uses kernel as submodule. | [commaai/agnos-builder](https://github.com/commaai/agnos-builder) |

**Local paths in this roadmap:**

- `../agnos-kernel-sdm845/` — kernel source
- `../agnos-builder/` — build system, scripts, userspace, firmware

### One-time clone (if not already present)

```bash
cd "Phase 5 - Advanced Topics and Specialization/5. Autonomous Vehicles"

# Kernel (large history; shallow clone recommended)
git clone --depth 1 https://github.com/commaai/agnos-kernel-sdm845.git

# Builder (smaller)
git clone https://github.com/commaai/agnos-builder.git
cd agnos-builder
git submodule update --init agnos-kernel-sdm845   # if building
./tools/extract_tools.sh                          # if building
```

---

## How to Learn: OS Lectures → AGNOS Repos

Study the **Operating Systems** lectures first, then map each topic to AGNOS code and config. **Every lecture (1–26)** is linked below to concrete paths in **agnos-kernel-sdm845** and **agnos-builder**.

---

<a id="all-os-lectures--agnos-overview"></a>
## All OS Lectures → AGNOS (Overview)

| # | Lecture | Kernel (agnos-kernel-sdm845) | Builder (agnos-builder) |
|:-:|--------|-------------------------------|--------------------------|
| 1 | [Modern OS Architecture & the Linux Kernel](../../../../Phase%201%20-%20Foundational%20Knowledge/3.%20Operating%20Systems/Lectures/Lecture-01.md) | `arch/`, `kernel/`, `mm/`, `drivers/` — monolithic layout | — |
| 2 | [Processes, task_struct & the Linux Process Model](../../../../Phase%201%20-%20Foundational%20Knowledge/3.%20Operating%20Systems/Lectures/Lecture-02.md) | `kernel/fork.c`, `include/linux/sched.h` (task_struct) | `userspace/` — processes started by systemd |
| 3 | [Interrupts, Exceptions & Bottom Halves](../../../../Phase%201%20-%20Foundational%20Knowledge/3.%20Operating%20Systems/Lectures/Lecture-03.md) | `arch/arm64/kernel/entry.S`, `kernel/irq/`, driver IRQ handlers | — |
| 4 | [System Calls, vDSO & eBPF](../../../../Phase%201%20-%20Foundational%20Knowledge/3.%20Operating%20Systems/Lectures/Lecture-04.md) | `arch/arm64/kernel/syscall.c`, `arch/arm64/kernel/vdso/` | — |
| 5 | [Kernel Modules, Boot Process & Device Tree](../../../../Phase%201%20-%20Foundational%20Knowledge/3.%20Operating%20Systems/Lectures/Lecture-05.md) | `arch/arm64/boot/dts/`, `drivers/` (of_match_table), `init/` | `firmware/`, `build_kernel.sh`, `build_system.sh`, initramfs |
| 6 | [CPU Scheduling: CFS, EEVDF & Real-Time Classes](../../../../Phase%201%20-%20Foundational%20Knowledge/3.%20Operating%20Systems/Lectures/Lecture-06.md) | `kernel/sched/core.c`, `fair.c`, `rt.c` | Boot cmdline (isolcpus, etc.) |
| 7 | [Real-Time Linux: PREEMPT_RT & Determinism](../../../../Phase%201%20-%20Foundational%20Knowledge/3.%20Operating%20Systems/Lectures/Lecture-07.md) | `kernel/sched/`, `kernel/locking/`, preempt config in `Kconfig` | — |
| 8 | [Multi-Core Scheduling, CPU Affinity & isolcpus](../../../../Phase%201%20-%20Foundational%20Knowledge/3.%20Operating%20Systems/Lectures/Lecture-08.md) | `kernel/sched/core.c` (affinity), `arch/arm64/` (CPU topology) | Boot config for isolcpus; openpilot `set_core_affinity()` |
| 9 | [Synchronization: Spinlocks, Mutexes, RW Locks](../../../../Phase%201%20-%20Foundational%20Knowledge/3.%20Operating%20Systems/Lectures/Lecture-09.md) | `kernel/locking/`, `include/linux/spinlock.h`, `mutex.c` | — |
| 10 | [Lock-Free Programming: RCU, Atomics](../../../../Phase%201%20-%20Foundational%20Knowledge/3.%20Operating%20Systems/Lectures/Lecture-10.md) | `kernel/rcu/`, `include/linux/atomic.h` | — |
| 11 | [Deadlock, Priority Inversion & PI Mutexes](../../../../Phase%201%20-%20Foundational%20Knowledge/3.%20Operating%20Systems/Lectures/Lecture-11.md) | `kernel/locking/rtmutex.c`, PI in scheduler | — |
| 12 | [Virtual Memory & the Linux Memory Model](../../../../Phase%201%20-%20Foundational%20Knowledge/3.%20Operating%20Systems/Lectures/Lecture-12.md) | `mm/` (vm_area, page tables), `arch/arm64/mm/` | — |
| 13 | [Page Tables, TLBs & Huge Pages](../../../../Phase%201%20-%20Foundational%20Knowledge/3.%20Operating%20Systems/Lectures/Lecture-13.md) | `arch/arm64/mm/`, `mm/memory.c`, huge page support | — |
| 14 | [Memory Allocation: SLUB, kmalloc & CMA](../../../../Phase%201%20-%20Foundational%20Knowledge/3.%20Operating%20Systems/Lectures/Lecture-14.md) | `mm/slub.c`, `mm/cma.c`, `include/linux/slab.h` | — |
| 15 | [DMA, IOMMU & GPU Memory Management](../../../../Phase%201%20-%20Foundational%20Knowledge/3.%20Operating%20Systems/Lectures/Lecture-15.md) | `drivers/iommu/`, `drivers/base/dma-mapping.c`, GPU in `drivers/` | — |
| 16 | [NUMA Topology & HPC Memory Optimization](../../../../Phase%201%20-%20Foundational%20Knowledge/3.%20Operating%20Systems/Lectures/Lecture-16.md) | `arch/arm64/mm/`, NUMA if enabled; SDM845 is UMA | — |
| 17 | [Linux Device Driver Model & Device Tree](../../../../Phase%201%20-%20Foundational%20Knowledge/3.%20Operating%20Systems/Lectures/Lecture-17.md) | `drivers/base/`, `arch/arm64/boot/dts/`, platform_driver, of_* | — |
| 18 | [Character Drivers, Interrupt-Driven I/O & V4L2](../../../../Phase%201%20-%20Foundational%20Knowledge/3.%20Operating%20Systems/Lectures/Lecture-18.md) | `drivers/media/`, V4L2 camera pipeline for comma 3X | — |
| 19 | [Modern I/O: io_uring, DMA-BUF & Zero-Copy](../../../../Phase%201%20-%20Foundational%20Knowledge/3.%20Operating%20Systems/Lectures/Lecture-19.md) | `io_uring/`, DMA-BUF in `drivers/dma-buf/` | VisionIpc / zero-copy in openpilot on top |
| 20 | [PCIe, NVMe & GPU Driver Architecture](../../../../Phase%201%20-%20Foundational%20Knowledge/3.%20Operating%20Systems/Lectures/Lecture-20.md) | `drivers/pci/`, `drivers/nvme/`, GPU under `drivers/` (Adreno on SDM845) | — |
| 21 | [Filesystems: ext4, btrfs, F2FS & overlayfs](../../../../Phase%201%20-%20Foundational%20Knowledge/3.%20Operating%20Systems/Lectures/Lecture-21.md) | `fs/ext4/`, `fs/overlayfs/` (often in Android/AGNOS rootfs) | `userspace/` rootfs layout; build_system packs fs |
| 22 | [Embedded Storage: eMMC, UFS, NVMe & OTA](../../../../Phase%201%20-%20Foundational%20Knowledge/3.%20Operating%20Systems/Lectures/Lecture-22.md) | `drivers/scsi/`, UFS for SDM845 storage, block layer | `firmware/`, partition layout, A/B slots in builder |
| 23 | [Containers, cgroups v2 & NVIDIA Container](../../../../Phase%201%20-%20Foundational%20Knowledge/3.%20Operating%20Systems/Lectures/Lecture-23.md) | `kernel/cgroup/` | Optional: AGNOS userspace can use cgroups for isolation |
| 24 | [OS for AI Systems: L4T, openpilot OS & RT Tuning](../../../../Phase%201%20-%20Foundational%20Knowledge/3.%20Operating%20Systems/Lectures/Lecture-24.md) | Entire kernel as “openpilot OS” (Agnos); sched, drivers, RT | agnos-builder = build for “openpilot OS”; RT tuning in config |
| 25 | [Capstone: Custom Linux Images with Yocto](../../../../Phase%201%20-%20Foundational%20Knowledge/3.%20Operating%20Systems/Lectures/Lecture-25.md) | — (Agnos uses its own builder, not Yocto) | **agnos-builder** is the capstone: custom image build (kernel + rootfs) |
| 26 | [eBPF: Programmable Kernel Observability](../../../../Phase%201%20-%20Foundational%20Knowledge/3.%20Operating%20Systems/Lectures/Lecture-26.md) | `kernel/bpf/`, `net/bpf/` (if enabled in config) | Observability of openpilot on AGNOS via eBPF tools |

---

<a id="os-lectures--agnos-development-changes-fork--custom-mods-for-openpilot"></a>
## OS Lectures ↔ AGNOS Development Changes (Fork + Custom Mods for openpilot)

The kernel in AGNOS is **forked from Linux and custom modified**. The table below connects each OS lecture to the **kind of development change** comma made in that fork for the **on-the-road openpilot** use case. Use it to see why each OS topic matters in practice.

| # | Lecture topic | Development change in AGNOS (fork / custom work) | Why it matters for openpilot on the road |
|:-:|----------------|--------------------------------------------------|------------------------------------------|
| 1 | OS architecture, monolithic kernel | Fork keeps Linux monolithic layout; **platform-specific code** in `arch/arm64/` and **vendor drivers** (camera ISP, CAN, GPU) added or patched for SDM845 and comma 3X hardware. | One kernel image drives cameras, CAN, and GPU; no extra IPC cost. Upstream changes could break ABI — comma **pins the kernel** to keep V4L2/ISP and SocketCAN stable for camerad and controlsd. |
| 2 | Processes, task_struct, fork/exec | Kernel process model unchanged; **agnos-builder** defines **which processes run** (systemd, openpilot’s camerad, modeld, controlsd) and how they are started. | openpilot’s multi-process design (camerad → modeld → plannerd → controlsd) relies on this; crash isolation and COW fork come from the same process model. |
| 3 | Interrupts, exceptions, bottom halves | **Driver changes**: camera and CAN drivers register IRQs and bottom-half paths; **latency of these paths** directly affects frame capture and CAN write timing. Comma’s patches keep interrupt and softirq handling predictable. | High-framerate camera and 100 Hz CAN need bounded interrupt latency; `/proc/interrupts` and IRQ affinity matter for tuning. |
| 4 | System calls, vDSO | Syscall and vDSO code is largely upstream; **no major fork-specific change**. vDSO `clock_gettime` is used by openpilot for sensor and model timestamps. | Stable, low-overhead timestamps for fusion and logging without syscall cost. |
| 5 | Boot, Device Tree, kernel modules | **Heavy development**: (1) **Boot chain** — agnos-builder supplies firmware (XBL, ABL), kernel, DTB, initrd; (2) **Device Tree** — DTS for comma 3X/four (cameras, SPI, CAN, GPIO); (3) **Driver probe** — camera ISP, CAN-over-SPI, power management drivers added/patched and matched via `compatible` in DT. | Correct boot and DT are required for cameras and CAN to appear; wrong DT or missing driver = no openpilot on the road. |
| 6 | CPU scheduling (CFS, SCHED_FIFO) | **Config and usage**: kernel has CFS/RT classes; **boot cmdline** (e.g. isolcpus) and **openpilot** use of `set_realtime_priority()` and `set_core_affinity()` (in openpilot repo). Comma may tune default scheduler or cmdline for RT workloads. | controlsd and modeld must hit deadlines; SCHED_FIFO and affinity avoid CFS jitter that would cause missed CAN frames or dropped frames. |
| 7 | Real-time (PREEMPT_RT, determinism) | **Kernel config**: preemption and locking options chosen for low latency; PREEMPT_RT or related patches may be applied in the fork for deterministic response. | controlsd CAN writes and inference loops need bounded latency; RT config is part of “openpilot on the road” reliability. |
| 8 | Multi-core, affinity, isolcpus | **Boot and userspace**: **agnos-builder** or device config sets **isolcpus** (and related cmdline); openpilot **set_core_affinity()** pins camerad/modeld/controlsd to chosen cores. Fork may carry scheduler/affinity fixes. | Isolating cores and pinning RT processes avoids interference from other work and reduces tail latency. |
| 9 | Synchronization (spinlocks, mutexes) | **Driver and core code**: any **new or modified driver** (camera, CAN, SPI) uses kernel locking primitives; lock design in the fork affects contention and latency. | Bad locking in a driver can cause stalls or priority inversion; PI mutex (Lecture 11) is relevant for openpilot’s high-priority control vs lower-priority readers. |
| 10 | RCU, lock-free, atomics | **Core kernel** uses RCU and atomics; fork may backport or tune for SDM845. No openpilot-specific RCU change, but **rcu_nocbs** on isolated cores (cmdline) moves RCU off RT cores. | Less jitter on cores running control and inference. |
| 11 | Deadlock, priority inversion, PI mutex | **Kernel** has rtmutex and PI; **openpilot** has high- and low-priority processes sharing state (e.g. cereal). Fork keeps PI mutex support so high-priority control is not blocked by low-priority readers. | Avoids “Mars Pathfinder”–style priority inversion in production. |
| 12 | Virtual memory, COW | **Core mm/** and **arch/arm64/mm**; fork keeps standard VM and COW. **openpilot** benefits: process isolation and COW fork for modeld/camerad without doubling RAM. | Stable VM model for multi-process openpilot and large model mappings. |
| 13–14 | Page tables, TLBs, SLUB, CMA | **mm/** and **arch/arm64/mm**; **CMA** is important for **camera and DMA buffers** on embedded SoCs — fork may enable or tune CMA for SDM845. | Camera pipeline and zero-copy buffers depend on contiguous and slab allocation. |
| 15 | DMA, IOMMU, GPU memory | **Driver and SoC support**: DMA and GPU (Adreno) drivers in the fork; IOMMU if enabled. **Buffer sharing** (camera ↔ GPU ↔ display) is platform-specific. | Needed for camera → GPU inference and display without extra copies. |
| 16 | NUMA | SDM845 is UMA; **little fork-specific NUMA work**. Affinity (Lecture 8) matters more than NUMA here. | — |
| 17 | Device driver model, Device Tree | **Central to the fork**: **new DTS files** and **of_match_table** in drivers for comma 3X/four; platform and bus code tie DT to driver probe. | Every comma-specific device (camera, CAN-over-SPI, etc.) is brought up via this model. |
| 18 | Character drivers, V4L2 | **Major development**: **camera ISP and V4L2** drivers are added or heavily patched for **road-facing and driver-monitoring cameras** on comma hardware. Register maps and subdevice layout are device-specific. | camerad depends on V4L2; “pins its kernel to maintain camera ISP register map compatibility” (Lecture-01) — these are the drivers that get pinned. |
| 19 | io_uring, DMA-BUF, zero-copy | **DMA-BUF and zero-copy** in kernel; **openpilot VisionIpc** uses shared memory and DMA-BUF-style sharing on top. Fork may enable or patch DMA-BUF for camera/GPU. | Low-latency, zero-copy path from camera to modeld and encoder. |
| 20 | PCIe, NVMe, GPU | **GPU (Adreno)** and storage (UFS) drivers in the fork for SDM845; PCIe/NVMe if present on carrier. | GPU runs inference; storage holds OS and logs. |
| 21 | Filesystems (ext4, overlayfs) | **agnos-builder** chooses rootfs layout (ext4 or similar) and overlay for updates; kernel has matching fs support. | Read-only root + writable overlay is common for reliable in-field updates. |
| 22 | eMMC/UFS, OTA partitioning | **agnos-builder** defines **partition layout and A/B slots**; kernel block and UFS drivers support the device. **OTA** flow (e.g. updateinstallerd) relies on this. | Safe, resettable updates for openpilot in the field without bricking the device. |
| 23 | cgroups | **Optional** in AGNOS userspace for isolation or resource limits; kernel cgroup support is standard. | Can limit non-critical services so they don’t starve openpilot. |
| 24 | OS for AI (L4T vs openpilot OS, RT tuning) | **AGNOS = “openpilot OS”** — the **entire fork and builder** are this. All previous rows (boot, DT, drivers, scheduler, memory, storage) are the “development changes” for AI/RT on the road. L4T is the Jetson counterpart; Agnos is the comma counterpart. | Direct mapping: this lecture describes why Agnos exists and how it’s tuned for openpilot. |
| 25 | Capstone (custom Linux image) | **agnos-builder** is the **custom image build** for comma devices (not Yocto, but same idea): reproducible kernel + rootfs + boot artifacts. | Delivers the single “AGNOS” image that goes on the device and runs openpilot. |
| 26 | eBPF observability | **Kernel** may enable CONFIG_BPF; **observability** of openpilot (camerad, modeld, controlsd) with bpftrace/perf runs on this kernel. | Debugging and profiling openpilot in development and in the field. |

---

## Detailed Mapping by Lecture

### Lecture 1: Modern OS Architecture & the Linux Kernel  
**OS Lecture:** [Lecture-01](../../../../Phase%201%20-%20Foundational%20Knowledge/3.%20Operating%20Systems/Lectures/Lecture-01.md)

| Concept | Where in AGNOS |
|--------|-----------------|
| Monolithic kernel layout | **agnos-kernel-sdm845:** Top-level `arch/`, `kernel/`, `mm/`, `drivers/`, `fs/`, `net/` — same as any Linux tree. |
| Platform-specific code | `arch/arm64/` — ARM64 entry, MMU, exceptions, SoC setup for SDM845. |
| Agnos / “openpilot OS” | This kernel is the one running on comma 3X/four; Lecture-01’s “openpilot Agnos” section refers to this repo. |

---

### Lecture 2: Processes, task_struct & the Linux Process Model  
**OS Lecture:** [Lecture-02](../../../../Phase%201%20-%20Foundational%20Knowledge/3.%20Operating%20Systems/Lectures/Lecture-02.md)

| Concept | Where in AGNOS |
|--------|-----------------|
| task_struct, PCB | **agnos-kernel-sdm845:** `include/linux/sched.h`, `kernel/fork.c` (copy_process, task_struct layout). |
| fork/exec/wait | `kernel/fork.c`, `fs/exec.c` — used by every userspace process; openpilot’s camerad, modeld, controlsd are such processes. |
| Userspace processes | **agnos-builder:** `userspace/` — systemd units, init scripts; processes that run on AGNOS (including openpilot) are started from this userspace. |

---

### Lecture 3: Interrupts, Exceptions & Bottom Halves  
**OS Lecture:** [Lecture-03](../../../../Phase%201%20-%20Foundational%20Knowledge/3.%20Operating%20Systems/Lectures/Lecture-03.md)

| Concept | Where in AGNOS |
|--------|-----------------|
| Exception entry, IRQ handling | **agnos-kernel-sdm845:** `arch/arm64/kernel/entry.S`, `arch/arm64/kernel/irq.c`, `kernel/irq/` (generic IRQ, chip drivers). |
| Bottom halves, softirq, tasklets | `kernel/softirq.c`, `kernel/time/` (timers). |
| Driver IRQs | Any driver in `drivers/` that does `request_irq()` — e.g. camera, SPI, GPIO; critical for camerad latency. |

---

### Lecture 4: System Calls, vDSO & eBPF  
**OS Lecture:** [Lecture-04](../../../../Phase%201%20-%20Foundational%20Knowledge/3.%20Operating%20Systems/Lectures/Lecture-04.md)

| Concept | Where in AGNOS |
|--------|-----------------|
| Syscall table and dispatch | **agnos-kernel-sdm845:** `arch/arm64/kernel/syscall.c`, `include/uapi/asm-generic/unistd.h`. |
| vDSO | `arch/arm64/kernel/vdso/` — e.g. clock_gettime used by openpilot for timestamps. |
| eBPF (if enabled) | `kernel/bpf/` — config-dependent; used for observability (Lecture 26). |

---

### Lecture 5: Kernel Modules, Boot Process & Device Tree  
**OS Lecture:** [Lecture-05](../../../../Phase%201%20-%20Foundational%20Knowledge/3.%20Operating%20Systems/Lectures/Lecture-05.md)

| Concept | Where in AGNOS |
|--------|-----------------|
| **Boot sequence** (ROM → bootloader → kernel → init) | **agnos-builder:** `firmware/` (XBL, ABL, etc.), `build_system.sh`, `build_kernel.sh`. Boot chain is platform-specific; builder produces kernel + initrd + rootfs. |
| **Device Tree** (`.dts`/`.dtb`, `compatible`, driver probe) | **agnos-kernel-sdm845:** `arch/arm64/boot/dts/`, `*.dts` / `*.dtsi`. Search for `compatible` and driver `of_match_table` in `drivers/`. |
| **Kernel command line** (`isolcpus`, `nohz_full`, `rcu_nocbs`) | **agnos-builder:** Boot config; **agnos-kernel-sdm845:** `kernel/sched/` consumes cmdline for isolcpus etc. |
| **initramfs** | **agnos-builder:** `build_system.sh` and `userspace/` pack early rootfs; inspect for init and switch_root. |
| **Kernel modules** | **agnos-kernel-sdm845:** `drivers/`, `Kconfig`, `Makefile`; builder builds kernel (and modules) as part of AGNOS image. |

**Tasks:** `grep -r "compatible" agnos-kernel-sdm845/arch/arm64/boot/dts/ | head -30`; trace artifacts in `agnos-builder/build_kernel.sh` and `build_system.sh`.

---

### Lecture 6: CPU Scheduling (CFS, EEVDF, SCHED_FIFO)  
**OS Lecture:** [Lecture-06](../../../../Phase%201%20-%20Foundational%20Knowledge/3.%20Operating%20Systems/Lectures/Lecture-06.md)

| Concept | Where in AGNOS |
|--------|-----------------|
| **Fair scheduler (CFS/EEVDF)** | **agnos-kernel-sdm845:** `kernel/sched/core.c`, `fair.c` (CFS), `rt.c` (RT class). 4.x kernel uses CFS. |
| **SCHED_FIFO** | Kernel implements `sched_setscheduler(SCHED_FIFO)`; openpilot’s `set_realtime_priority()` in `openpilot/common/util.cc` calls it (see [Lecture-06 Real example](../../../../Phase%201%20-%20Foundational%20Knowledge/3.%20Operating%20Systems/Lectures/Lecture-06.md#real-example-in-openpilot-this-repo)). |
| **CPU isolation / affinity** | Boot cmdline (from builder) + openpilot `set_core_affinity()`; scheduler respects isolcpus and affinity. |

**Tasks:** Inspect `kernel/sched/fair.c` (vruntime, pick-next) and `rt.c` (FIFO/RR).

---

### Lecture 7: Real-Time Linux (PREEMPT_RT & Determinism)  
**OS Lecture:** [Lecture-07](../../../../Phase%201%20-%20Foundational%20Knowledge/3.%20Operating%20Systems/Lectures/Lecture-07.md)

| Concept | Where in AGNOS |
|--------|-----------------|
| Preemption config | **agnos-kernel-sdm845:** `Kconfig` (CONFIG_PREEMPT_*), `kernel/sched/`, `kernel/locking/` (rtmutex if PREEMPT_RT). |
| Latency, determinism | Same scheduler and locking code; openpilot’s controlsd/modeld depend on low latency — see Lecture 6 and 8. |

---

### Lecture 8: Multi-Core Scheduling, CPU Affinity & isolcpus  
**OS Lecture:** [Lecture-08](../../../../Phase%201%20-%20Foundational%20Knowledge/3.%20Operating%20Systems/Lectures/Lecture-08.md)

| Concept | Where in AGNOS |
|--------|-----------------|
| Affinity, CPU mask | **agnos-kernel-sdm845:** `kernel/sched/core.c` (set_cpus_allowed_ptr, load balance), `arch/arm64/` (topology). |
| isolcpus | Set via kernel cmdline; **agnos-builder** or device boot config provides it; openpilot then pins with `set_core_affinity()` (util.cc). |

---

### Lecture 9: Synchronization (Spinlocks, Mutexes, RW Locks)  
**OS Lecture:** [Lecture-09](../../../../Phase%201%20-%20Foundational%20Knowledge/3.%20Operating%20Systems/Lectures/Lecture-09.md)

| Concept | Where in AGNOS |
|--------|-----------------|
| Spinlocks, mutexes, rwlock | **agnos-kernel-sdm845:** `kernel/locking/spinlock.c`, `mutex.c`, `rwlock.c`, `include/linux/spinlock.h`, `mutex.h`. |
| Usage in drivers | Any `drivers/` code that protects shared state; e.g. V4L2, SPI, block layer. |

---

### Lecture 10: Lock-Free Programming (RCU, Atomics)  
**OS Lecture:** [Lecture-10](../../../../Phase%201%20-%20Foundational%20Knowledge/3.%20Operating%20Systems/Lectures/Lecture-10.md)

| Concept | Where in AGNOS |
|--------|-----------------|
| RCU | **agnos-kernel-sdm845:** `kernel/rcu/` — used widely (scheduler, networking, VFS). |
| Atomics, memory ordering | `include/linux/atomic.h`, arch-specific in `arch/arm64/include/asm/atomic.h`. |

---

### Lecture 11: Deadlock, Priority Inversion & PI Mutexes  
**OS Lecture:** [Lecture-11](../../../../Phase%201%20-%20Foundational%20Knowledge/3.%20Operating%20Systems/Lectures/Lecture-11.md)

| Concept | Where in AGNOS |
|--------|-----------------|
| rtmutex, priority inheritance | **agnos-kernel-sdm845:** `kernel/locking/rtmutex.c`; scheduler integrates PI (see Lecture 11’s openpilot cereal/controlsd note). |
| Deadlock avoidance | Lock ordering and design in `kernel/locking/` and drivers. |

---

### Lecture 12: Virtual Memory & the Linux Memory Model  
**OS Lecture:** [Lecture-12](../../../../Phase%201%20-%20Foundational%20Knowledge/3.%20Operating%20Systems/Lectures/Lecture-12.md)

| Concept | Where in AGNOS |
|--------|-----------------|
| VMAs, page tables | **agnos-kernel-sdm845:** `mm/mmap.c`, `mm/memory.c`, `arch/arm64/mm/` (page table walk, TLB). |
| COW, fork | `kernel/fork.c` (copy-on-write for private mappings); openpilot multi-process (camerad, modeld, controlsd) uses this. |

---

### Lecture 13: Page Tables, TLBs & Huge Pages  
**OS Lecture:** [Lecture-13](../../../../Phase%201%20-%20Foundational%20Knowledge/3.%20Operating%20Systems/Lectures/Lecture-13.md)

| Concept | Where in AGNOS |
|--------|-----------------|
| PTE manipulation, TLB flush | **agnos-kernel-sdm845:** `arch/arm64/mm/` (pgtable, tlbflush), `mm/memory.c`. |
| Huge pages | `arch/arm64/mm/`, hugetlbfs in `fs/hugetlbfs/` (if enabled). |

---

### Lecture 14: Memory Allocation (SLUB, kmalloc, CMA)  
**OS Lecture:** [Lecture-14](../../../../Phase%201%20-%20Foundational%20Knowledge/3.%20Operating%20Systems/Lectures/Lecture-14.md)

| Concept | Where in AGNOS |
|--------|-----------------|
| SLUB, kmalloc | **agnos-kernel-sdm845:** `mm/slub.c`, `include/linux/slab.h`; `kmalloc`/`kfree` used by all drivers. |
| CMA (Contiguous Memory Allocator) | `mm/cma.c` — often used for camera buffers, DMA on embedded SoCs. |

---

### Lecture 15: DMA, IOMMU & GPU Memory Management  
**OS Lecture:** [Lecture-15](../../../../Phase%201%20-%20Foundational%20Knowledge/3.%20Operating%20Systems/Lectures/Lecture-15.md)

| Concept | Where in AGNOS |
|--------|-----------------|
| DMA API, IOMMU | **agnos-kernel-sdm845:** `drivers/base/dma-mapping.c`, `drivers/iommu/` (if enabled for SDM845). |
| GPU (Adreno on SDM845) | GPU driver under `drivers/` (vendor/Qualcomm); shares buffers with camera and display. |

---

### Lecture 16: NUMA Topology & HPC Memory Optimization  
**OS Lecture:** [Lecture-16](../../../../Phase%201%20-%20Foundational%20Knowledge/3.%20Operating%20Systems/Lectures/Lecture-16.md)

| Concept | Where in AGNOS |
|--------|-----------------|
| NUMA (if present) | **agnos-kernel-sdm845:** `arch/arm64/mm/`, NUMA config; SDM845 is typically UMA — single node. |
| Memory topology | CPU topology in `arch/arm64/` / kernel; relevant for affinity (Lecture 8). |

---

### Lecture 17: Linux Device Driver Model & Device Tree  
**OS Lecture:** [Lecture-17](../../../../Phase%201%20-%20Foundational%20Knowledge/3.%20Operating%20Systems/Lectures/Lecture-17.md)

| Concept | Where in AGNOS |
|--------|-----------------|
| Platform driver, of_* (OF = Device Tree) | **agnos-kernel-sdm845:** `drivers/base/platform.c`, `drivers/base/of_*.c`, `arch/arm64/boot/dts/`. |
| Bus, device, driver model | `drivers/base/`, `include/linux/device.h`; every camera, SPI, I2C driver plugs in here. |

---

### Lecture 18: Character Drivers, Interrupt-Driven I/O & V4L2  
**OS Lecture:** [Lecture-18](../../../../Phase%201%20-%20Foundational%20Knowledge/3.%20Operating%20Systems/Lectures/Lecture-18.md)

| Concept | Where in AGNOS |
|--------|-----------------|
| V4L2, camera pipeline | **agnos-kernel-sdm845:** `drivers/media/` — V4L2 subdevs, video capture; this is what openpilot camerad talks to. |
| Character drivers, chardev | `drivers/` (e.g. SPI, I2C expose chardev or are used by V4L2); interrupt-driven I/O in driver handlers. |

---

### Lecture 19: Modern I/O (io_uring, DMA-BUF & Zero-Copy)  
**OS Lecture:** [Lecture-19](../../../../Phase%201%20-%20Foundational%20Knowledge/3.%20Operating%20Systems/Lectures/Lecture-19.md)

| Concept | Where in AGNOS |
|--------|-----------------|
| io_uring | **agnos-kernel-sdm845:** `io_uring/` (if enabled in config). |
| DMA-BUF, zero-copy | `drivers/dma-buf/`; used by camera/GPU pipeline; openpilot VisionIpc builds on shared memory / zero-copy. |

---

### Lecture 20: PCIe, NVMe & GPU Driver Architecture  
**OS Lecture:** [Lecture-20](../../../../Phase%201%20-%20Foundational%20Knowledge/3.%20Operating%20Systems/Lectures/Lecture-20.md)

| Concept | Where in AGNOS |
|--------|-----------------|
| PCIe, NVMe | **agnos-kernel-sdm845:** `drivers/pci/`, `drivers/nvme/` (if used on platform). |
| GPU (Adreno) | GPU driver in `drivers/` (Qualcomm); used for openpilot inference (e.g. tinygrad/OpenCL/Vulkan). |

---

### Lecture 21: Filesystems (ext4, btrfs, F2FS, overlayfs)  
**OS Lecture:** [Lecture-21](../../../../Phase%201%20-%20Foundational%20Knowledge/3.%20Operating%20Systems/Lectures/Lecture-21.md)

| Concept | Where in AGNOS |
|--------|-----------------|
| VFS, ext4, overlayfs | **agnos-kernel-sdm845:** `fs/ext4/`, `fs/overlayfs/`, `fs/` (VFS layer). |
| Rootfs layout | **agnos-builder:** `userspace/` and `build_system.sh` define what goes on rootfs; often ext4 or similar. |

---

### Lecture 22: Embedded Storage (eMMC, UFS, OTA Partitioning)  
**OS Lecture:** [Lecture-22](../../../../Phase%201%20-%20Foundational%20Knowledge/3.%20Operating%20Systems/Lectures/Lecture-22.md)

| Concept | Where in AGNOS |
|--------|-----------------|
| Block layer, UFS (SDM845 storage) | **agnos-kernel-sdm845:** `drivers/scsi/` (UFS), `block/`. |
| Partitions, A/B, OTA | **agnos-builder:** `firmware/`, partition scripts; A/B slots for system updates (similar to Lecture 22’s openpilot Agnos note). |

---

### Lecture 23: Containers, cgroups v2  
**OS Lecture:** [Lecture-23](../../../../Phase%201%20-%20Foundational%20Knowledge/3.%20Operating%20Systems/Lectures/Lecture-23.md)

| Concept | Where in AGNOS |
|--------|-----------------|
| cgroups | **agnos-kernel-sdm845:** `kernel/cgroup/` — cgroup v2 if enabled. |
| Userspace | **agnos-builder:** Optional use of cgroups in userspace for process isolation or resource limits. |

---

### Lecture 24: OS for AI Systems (L4T, openpilot OS & RT Tuning)  
**OS Lecture:** [Lecture-24](../../../../Phase%201%20-%20Foundational%20Knowledge/3.%20Operating%20Systems/Lectures/Lecture-24.md)

| Concept | Where in AGNOS |
|--------|-----------------|
| “openpilot OS” | **AGNOS = openpilot’s OS** on comma 3X/four: **agnos-kernel-sdm845** + **agnos-builder**-built userspace. |
| RT tuning (modeld, controlsd, camerad) | Scheduler (Lectures 6, 8), affinity, isolcpus; openpilot `util.cc` (set_realtime_priority, set_core_affinity) on this kernel. |
| L4T comparison | Lecture 24 compares L4T (Jetson) vs Agnos (comma); this repo is the Agnos side. |

---

### Lecture 25: Capstone — Custom Linux Images (Yocto)  
**OS Lecture:** [Lecture-25](../../../../Phase%201%20-%20Foundational%20Knowledge/3.%20Operating%20Systems/Lectures/Lecture-25.md)

| Concept | Where in AGNOS |
|--------|-----------------|
| Custom OS image | **agnos-builder** is the capstone for “custom Linux image” for comma devices: it builds kernel (from agnos-kernel-sdm845) + rootfs + boot artifacts, not Yocto but same idea. |
| Reproducible build | Docker-based build in agnos-builder; versioned kernel submodule. |

---

### Lecture 26: eBPF — Programmable Kernel Observability  
**OS Lecture:** [Lecture-26](../../../../Phase%201%20-%20Foundational%20Knowledge/3.%20Operating%20Systems/Lectures/Lecture-26.md)

| Concept | Where in AGNOS |
|--------|-----------------|
| eBPF core | **agnos-kernel-sdm845:** `kernel/bpf/`, `net/bpf/` (if CONFIG_BPF enabled). |
| Observability of openpilot | Tracing openpilot (camerad, modeld, controlsd) on AGNOS with bpftrace/perf; Lecture 26’s openpilot pipeline examples apply to this kernel. |

---

## Suggested Study Path

1. **Phase 1 — OS course (all 26 lectures)**  
   Use the [All OS Lectures → AGNOS (Overview)](#all-os-lectures--agnos-overview) table above: every lecture links to the OS slide and to concrete kernel/builder paths. Start with Lectures 1–6 (architecture, processes, interrupts, syscalls, boot/DT, scheduling), then follow the rest in order or by topic.

2. **Clone and open the two repos**  
   Keep [Phase 1 — Operating Systems — Guide](../../../../Phase%201%20-%20Foundational%20Knowledge/3.%20Operating%20Systems/Guide.md) and lecture list handy.

3. **agnos-kernel-sdm845**  
   - Browse `arch/arm64/`, `kernel/sched/`, `drivers/`.  
   - Search for `compatible`, `of_match_table`, `probe`, `module_init` to see Device Tree and module flow (Lecture 5).  
   - Look at `kernel/sched/fair.c` and `rt.c` for CFS and RT (Lecture 6).

4. **agnos-builder**  
   - Read `README.md`, run (if you have Docker) `./build_kernel.sh` and/or `./build_system.sh`.  
   - Trace boot artifacts: kernel image, DTB, initrd, rootfs in scripts and `firmware/`.

5. **Cross-link with openpilot**  
   openpilot runs on AGNOS. [Lecture-05](../../../../Phase%201%20-%20Foundational%20Knowledge/3.%20Operating%20Systems/Lectures/Lecture-05.md) and [Lecture-06](../../../../Phase%201%20-%20Foundational%20Knowledge/3.%20Operating%20Systems/Lectures/Lecture-06.md) “Real example in openpilot” sections point to `openpilot/common/util.cc` (`set_realtime_priority`, `set_core_affinity`) — those system calls target the AGNOS kernel.

---

## Quick Reference: Key Paths in Cloned Repos

| What | Path (relative to repo root) |
|------|-------------------------------|
| ARM64 boot / Device Tree | `agnos-kernel-sdm845/arch/arm64/` |
| Device Tree sources | `agnos-kernel-sdm845/arch/arm64/boot/dts/` |
| Scheduler (CFS, RT) | `agnos-kernel-sdm845/kernel/sched/` |
| Drivers (camera, SPI, etc.) | `agnos-kernel-sdm845/drivers/` |
| Build kernel | `agnos-builder/build_kernel.sh` |
| Build system image | `agnos-builder/build_system.sh` |
| Userspace / initramfs content | `agnos-builder/userspace/` |
| Boot/firmware related | `agnos-builder/firmware/` |

---

## Resources

- **agnos-kernel-sdm845:** [GitHub — commaai/agnos-kernel-sdm845](https://github.com/commaai/agnos-kernel-sdm845) — “Kernel for the SDM845 modules”.  
- **agnos-builder:** [GitHub — commaai/agnos-builder](https://github.com/commaai/agnos-builder) — “Build AGNOS, the operating system for the comma three, 3X, and four.”  
- **OS course:** [Phase 1 — Operating Systems — Guide](../../../../Phase%201%20-%20Foundational%20Knowledge/3.%20Operating%20Systems/Guide.md) and [Lecture index](../../../../Phase%201%20-%20Foundational%20Knowledge/3.%20Operating%20Systems/Guide.md#lecture-index).  
- **openpilot:** Runs on AGNOS; see [Autonomous Driving Guide](../Guide.md) and [flow diagram](../flow-diagram.md).
