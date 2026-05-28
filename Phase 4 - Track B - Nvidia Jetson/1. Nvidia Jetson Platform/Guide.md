# Nvidia Jetson Platform

<div class="course-identity jetson-platform" markdown="1">
<div class="course-identity__icon">JET</div>
<div markdown="1">
<p class="course-identity__eyebrow">Track B1 · NVIDIA Jetson Platform</p>
<p class="course-identity__title">Bring up Orin hardware, boot flow, CUDA, sensors, TensorRT, power, and production constraints.</p>
<p class="course-identity__meta">Artifact: Jetson platform validation report · Measure: boot, memory, power, inference</p>
</div>
</div>


**Phase 4 — Track B — Nvidia Jetson** · Module 1 of 7

> **Focus:** Go from unboxed **Jetson Orin Nano 8GB** hardware to a production-quality AI pipeline with ROS 2 integration, sensor fusion, optimized inference, OTA updates, and hardened security.
>
> **Primary hardware:** Jetson Orin Nano 8GB Developer Kit

**Next:** [2. Custom Carrier Board Design](../2.%20Custom%20Carrier%20Board%20Design%20and%20Bring-Up/Guide.md)

---


<a id="1-orin-nano-8gb--hardware--boot-chain-internals"></a>
## 1. Orin Nano 8GB — Hardware & Boot Chain Internals

This section walks through the **real boot chain**, firmware layout, memory usage, A/B slots, and what happens specifically on Orin Nano 8GB. Understanding this gives you the foundation to debug boot failures, customize firmware, optimize memory, and work confidently with the Jetson platform at the hardware level.

> **Deep dive:** For production-level memory architecture details (SMMU translation, CMA internals, camera zero-copy pipeline, DLA memory path, multi-camera planning, and production debugging), see [**Orin Nano Memory Architecture Deep Dive**](Orin-Nano-Memory-Architecture/Guide.md).

> **Deep dive:** For production-scale Yocto/OpenEmbedded BSP development — meta-tegra layer, custom layers, rootfs optimization, cross-compilation, secure boot integration, CI/CD pipelines, OTA at scale (25,000+ devices), system bring-up, boot performance, licensing compliance, and release engineering — see [**Orin Nano Yocto BSP & Production Deployment**](Orin-Nano-Yocto-BSP-Production/Guide.md).

> **Deep dive:** For **tensor core architecture** and how it works on Orin Nano (Ampere) — what tensor cores are, how they differ from CUDA cores, matrix multiply-accumulate (MMA), precision (FP16/INT8), and how TensorRT/cuDNN use them for high TOPS — see [**Orin Nano — Tensor Core Architecture and How It Works**](Orin-Nano-Tensor/Guide.md).

### 1.1 Hardware Context — What Orin Nano 8GB Actually Is

Orin Nano 8GB uses:

* **SoC:** Tegra234 (T234)
* **CPU:** ARM Cortex-A78AE cluster (6 cores)
* **GPU:** Ampere architecture (1024 CUDA cores)
* **Memory:** 8GB LPDDR5 (unified — shared between CPU and GPU)
* **Storage (Dev Kit):** NVMe (usually)
* **Boot storage:** QSPI NOR flash (for bootloader + firmware)

Important distinctions:

* Boot components are **not all stored on NVMe** — critical boot stages live in **QSPI NOR flash**
* Jetson is closer to **smartphone architecture** than PC architecture

### 1.2 Secure Boot Chain (Hardware Root of Trust)

Boot starts inside SoC silicon. Each stage verifies the next before handing off control.

#### Step 1: BootROM (BR)

* Hard-coded in silicon — cannot be modified
* Executes immediately at power-on
* Reads fuses for secure boot configuration
* Verifies next stage (MB1)

If secure boot is enabled:

* Uses PKC (Public Key Cryptography)
* Verifies digital signatures on all subsequent stages

#### Step 2: MB1 (Microboot1)

Loaded from QSPI NOR flash.

Responsibilities:

* DRAM training (LPDDR5 initialization)
* Power rails configuration
* Clock setup
* Security configuration
* Initializes BPMP (Boot and Power Management Processor)
* Verifies MB2

Without correct DRAM training, the system will not boot.

#### Step 3: MB2

Still loaded from QSPI NOR flash.

MB2 handles:

* Additional hardware initialization
* Memory carveout preparation
* Loading UEFI, OP-TEE (Trusted OS), and firmware blobs

At this stage the CPU is still not running Linux — this is still the NVIDIA boot world.

### 1.3 UEFI Stage (Firmware Layer)

Jetson Orin Nano uses an EDK2-based UEFI implementation.

UEFI responsibilities:

* Enumerates devices (PCIe, NVMe, USB)
* Selects boot device
* Handles A/B slot logic
* Loads `Image` (kernel), `kernel-dtb` (device tree), and `initrd`

Boot variables are stored in QSPI and EFI variable storage.

### 1.4 A/B Partition Layout (Critical for OTA)

Orin Nano uses a redundancy system for safe over-the-air updates:

```
APP        → rootfs A
APP_b      → rootfs B

kernel      → A
kernel_b    → B
```

UEFI checks:

* Which slot is active
* Boot success flag
* Retry counter

If boot fails too many times, the system automatically switches to the other slot. This is handled by the Boot Control Block (BCB) and EFI variables — this is how OTA updates work safely.

### 1.5 Linux Kernel Stage (On Orin Nano)

UEFI jumps to kernel entry.

Kernel binary:

```
/boot/Image
```

Device Tree:

```
tegra234-p3767-0000-p3768-0000-a0.dtb
```

#### CPU Setup

* Switch to EL1 (Exception Level 1)
* Setup page tables
* Enable MMU
* Initialize SMP cores

#### Memory Layout

8GB RAM is split into:

* **Linux usable RAM** — available to userspace and kernel
* **CMA region** — for GPU and V4L2 allocations
* **Carveouts** for:
  * BPMP (Boot and Power Management Processor)
  * RCE (camera real-time engine)
  * SPE (Safety Processor Engine)
  * OP-TEE (Trusted Execution Environment)
  * Display firmware

Inspect the memory layout with:

```bash
cat /proc/iomem
```

#### NVIDIA Driver Stack

Unlike normal PC Linux, Orin loads Jetson-specific drivers:

* `nvgpu` — GPU driver
* `nvhost` — host1x multimedia subsystem
* `tegra-camrtc` — camera real-time controller
* `vi` — Video Input driver
* `isp` — Image Signal Processor driver
* `nvcsi` — NVIDIA CSI driver

Camera pipeline path:

```
Sensor → NVCSI → VI → ISP → Memory (NVMM) → CUDA / V4L2
```

### 1.6 Camera Pipeline (Jetson Specific)

On Orin Nano, the camera data flow is:

1. Sensor connected via CSI (Camera Serial Interface)
2. NVCSI block receives raw data
3. VI (Video Input) captures frames
4. ISP processes (debayer, denoise, tone-map)
5. Memory allocated via CMA
6. Exposed as `/dev/video0`

Zero-copy path:

* **NVMM memory** — NVIDIA multimedia memory
* **DMA-BUF** — kernel buffer sharing
* **CUDA interop** — direct GPU access without copies

This is why CMA size is critical for camera + AI workloads.

### 1.7 initramfs Stage

Before mounting the full rootfs:

* Loads kernel modules
* Checks root partition
* Handles encryption (if enabled)
* Switches root

Then executes:

```bash
exec /sbin/init
```

### 1.8 systemd Stage

On Jetson Ubuntu, systemd starts:

* `nvargus-daemon` — camera service
* `nvpmodel` service — power mode management
* `networkd` — networking
* `display manager` — GUI
* `docker` — container runtime (if enabled)

NVIDIA-specific services:

* `nvpmodel` controls power modes (e.g., 15W vs 7W)
* `jetson_clocks` tool adjusts clocks for maximum performance

### 1.9 Graphics Stack

Display flow:

```
Kernel DRM driver
 → NVIDIA display controller
  → Wayland (default on recent Ubuntu)
   → GNOME
    → Login screen
```

GPU firmware is loaded during driver probe.

### 1.10 Full Boot Chain Summary

```
Power On
 ↓
BootROM (silicon — immutable)
 ↓
MB1 (QSPI — DRAM training + security)
 ↓
MB2 (QSPI — loads UEFI + OP-TEE + firmware)
 ↓
UEFI (device enumeration + A/B slot selection)
 ↓
Load Kernel + DTB + initrd
 ↓
Linux Kernel (memory + drivers + GPU + camera)
 ↓
Mount APP or APP_b
 ↓
initramfs → switch root
 ↓
systemd (NVIDIA services + desktop)
```

### 1.11 Advanced Engineering Details

#### Where Are Bootloaders Stored?

QSPI NOR flash contains:

* MB1, MB2, UEFI
* Firmware blobs
* Boot configuration tables

Rootfs (APP partition) lives on:

* NVMe (Dev Kit)
* Or eMMC (production module variants)

#### Firmware Running Outside Linux

Even when Linux is running, these microcontrollers remain active:

* **BPMP** — Boot and Power Management Processor
* **SPE** — Safety Processor Engine
* **RCE** — Camera Real-Time Engine

They run their own firmware loaded during boot. Linux communicates with them via mailbox + IPC.

### 1.12 What Makes Orin Nano Different From PC?

| PC Boot            | Jetson Orin Boot              |
|--------------------|-------------------------------|
| BIOS/UEFI          | NVIDIA MB1 + MB2 + UEFI      |
| Simple DRAM init   | Complex LPDDR5 training       |
| Standard GPU       | Firmware-heavy embedded GPU   |
| No carveouts       | Multiple memory carveouts     |
| No BPMP            | Dedicated power MCU (BPMP)    |

Jetson is closer to smartphone architecture than PC — understanding this distinction is key to working effectively with the platform.

### 1.13 Reading a Real First-Boot Log

Below is an annotated walkthrough of an actual Orin Nano 8GB first boot (JetPack 36.4.3, SD card, 2025-01-08 firmware). Every line you will see here is normal — understanding what each section means prevents hours of unnecessary debugging.

#### UEFI prompt (before kernel)

```
Jetson System firmware version 36.4.3-gcid-38968081 date 2025-01-08
ESC   to enter Setup.
F11   to enter Boot Manager Menu.
Enter to continue boot.
L4TLauncher: Attempting Direct Boot
EFI stub: Booting Linux Kernel...
EFI stub: Using DTB from configuration table
EFI stub: Loaded initrd from LINUX_EFI_INITRD_MEDIA_GUID device path
EFI stub: Exiting boot services...
```

- **36.4.3** = L4T (Linux for Tegra) version. Maps to JetPack 6.1.
- **L4TLauncher** = NVIDIA's EFI application that selects A/B slot and hands off to the kernel.
- **EFI stub** = the kernel's built-in EFI loader. DTB and initrd are loaded here, before Linux takes over.

#### CPU and memory

```
[    0.000000] Machine model: NVIDIA Jetson Orin Nano Engineering Reference Developer Kit Super
[    0.000000] Linux version 5.15.148-tegra
[    0.000000] Memory: 7517560K/8133248K available (... 353544K reserved, 262144K cma-reserved)
[    0.000000] Reserved memory: created CMA memory pool at 0x000000024a000000, size 256 MiB
[    0.000000] NUMA: Faking a node at [mem 0x0000000080000000-0x0000000277ffffff]
```

- **7.17 GB available of 8 GB** — ~615 MB consumed by reserved carveouts (BPMP, OP-TEE, firmware blobs, SMMU tables).
- **256 MB CMA** — used for GPU and V4L2 zero-copy camera buffers. Configurable in device tree.
- **NUMA faking** — Orin is a single-node SoC; Linux creates a synthetic NUMA node to satisfy the kernel's NUMA APIs.

```
[    0.005418] CPU1: Booted secondary processor 0x0000000100
[    0.005935] CPU2: Booted secondary processor 0x0000000200
[    0.006368] CPU3: Booted secondary processor 0x0000000300
[    0.008843] CPU4: Booted secondary processor 0x0000010200
[    0.009374] CPU5: Booted secondary processor 0x0000010300
[    0.009461] smp: Brought up 1 node, 6 CPUs
```

6 cores confirmed. CPUs 0–3 are in cluster 0 (MPIDR `0x0000_00XX`); CPUs 4–5 are in cluster 1 (MPIDR `0x0001_02XX`). This matters for CPU affinity — isolating latency-critical threads to one cluster avoids cross-cluster coherence traffic.

#### Security state

```
[    0.000000] secureboot: Secure boot disabled
[    0.000000] CPU features: detected: Spectre-v4
[    0.000000] CPU features: detected: Spectre-BHB
[    0.000000] CPU features: kernel page table isolation forced ON by KASLR
[    0.000000] CPU features: detected: Kernel page table isolation (KPTI)
```

- **Secure boot disabled** — expected on the developer kit out of the box. Enabling it requires fuse programming (covered in the Security section).
- **Spectre mitigations active** — KPTI and SSBS are on. Small performance cost (~2–5% on syscall-heavy workloads); do not disable in production.

```
[    3.321122] optee: probing for conduit method.
[    3.321179] optee: revision 4.2 (d78bc5fa)
[    3.380504] optee: dynamic shared memory is enabled
[    3.380768] optee: initialized driver
```

OP-TEE (Trusted Execution Environment) runs even with secure boot disabled. It provides the secure-world side for key storage, secure storage, and cryptographic operations.

```
I/TC: WARNING: Failed to get monotonic counter for REE FS, using 0
I/TC: WARNING: Failed to commit dirh counter 2
```

> These two OP-TEE warnings appear on every developer kit with secure boot disabled. They mean the secure storage anti-rollback counter cannot be committed to a hardware one-time counter — which requires secure boot fuses to be blown. **Not a bug. Ignore in development.**

#### SMMU (IOMMU)

```
arm-smmu 8000000.iommu: SMMUv2 with:
  stage 1 translation, stage 2 translation, nested translation
  stream matching with 128 register groups
  128 context banks
  Stage-1: 48-bit VA -> 48-bit IPA
  Stage-2: 48-bit IPA -> 48-bit PA
```

Three SMMUv2 instances serve different peripheral groups. The SMMU enforces DMA isolation — a misbehaving peripheral cannot read arbitrary physical memory. Relevant when writing custom kernel drivers or DMA engines.

#### RTCPU IVC warnings

```
tegra-ivc-bus bc00000.rtcpu:ivc-bus:echo@0: ivc channel driver missing
tegra-ivc-bus bc00000.rtcpu:ivc-bus:ivccapture@4: ivc channel driver missing
```

> These appear when no CSI camera is attached and the `tegra-camrtc` camera real-time engine has nothing to control. **Normal with no camera connected.** They disappear once you attach a camera and load the full V4L2 camera stack.

#### Storage and partitions

```
[    3.909550] mmc0: new ultra high speed SDR104 SDXC card at address 0001
[    3.910216] mmcblk0: mmc0:0001 SD16G 117 GiB
[    3.918828] GPT:Primary header thinks Alt. header is not at the end of the disk.
[    3.918831] GPT:47044607 != 245759999
[    3.918837] GPT: Use GNU Parted to correct GPT errors.
[    3.918855]  mmcblk0: p1 p2 p3 p4 p5 p6 p7 p8 p9 p10 p11 p12 p13 p14 p15
```

The GPT warning is expected after flashing: the flash tool writes a partition table sized for the minimum image, but the physical card is much larger. Fix it to use the full card capacity:

```bash
# On the Jetson (after first boot)
sudo parted /dev/mmcblk0 ---pretend-input-tty resizepart 1 Yes 100%
sudo resize2fs /dev/mmcblk0p1
```

15 partitions is normal for Orin Nano — they include A/B kernel, DTB, firmware blobs, and the rootfs APP partition (p1).

#### nvmap page pool

```
nvmap_page_pool_init: nvmap page pool size: 243839 pages (952 MB)
```

nvmap pre-allocates 952 MB of zeroed pages for GPU and multimedia allocations. This comes out of available RAM. On an 8 GB device under memory pressure, reduce it via the `nvmap_pagepoolsize` kernel parameter.

#### First-boot USB serial prompt

```
[   24.092716] Please complete system configuration setup on the serial port
               provided by Jetson's USB device mode connection. e.g. /dev/ttyACMx
```

The Jetson exposes a USB gadget serial port (ACM) during first-boot setup. Connect from the host:

```bash
sudo screen /dev/ttyACM0 115200
# or
sudo minicom -D /dev/ttyACM0 -b 115200
```

Walk through the Ubuntu initial-setup wizard (locale, user creation, etc.). This only appears once.

#### Boot timing summary (from the log)

| Milestone | Timestamp |
|-----------|-----------|
| UEFI hands off to kernel | 0.000 s |
| All 6 CPUs online | 0.009 s |
| Kernel drivers loaded | ~3.5 s |
| Rootfs mounted (mmcblk0p1) | 7.9 s |
| systemd started | 8.2 s |
| USB serial config prompt | 24.1 s |

Total cold boot to usable shell: **~25 seconds** from SD card. NVMe reduces this to ~15 seconds.

---

## 2. Jetson Hardware Overview

### Orin Nano 8GB vs the Orin Family

| Module              | CPU           | GPU         | RAM   | AI TOPS | Power  | Use Case              |
|---------------------|---------------|-------------|-------|---------|--------|-----------------------|
| Orin Nano 4GB       | 6-core A78AE  | 512-core A  | 4GB   | 20      | 5–10W  | Light inference       |
| **Orin Nano 8GB**   | **6-core A78AE** | **1024-core A** | **8GB** | **40** | **5–15W** | **This guide** |
| Orin NX 8GB         | 6-core A78AE  | 1024-core A | 8GB   | 70      | 10–20W | Mid-range robotics    |
| Orin NX 16GB        | 8-core A78AE  | 1024-core A | 16GB  | 100     | 10–25W | Heavy inference       |
| AGX Orin 32GB       | 12-core A78AE | 2048-core A | 32GB  | 200     | 15–60W | Autonomous vehicles   |
| AGX Orin 64GB       | 12-core A78AE | 2048-core A | 64GB  | 275     | 15–60W | Full AV systems       |

### Orin Nano 8GB Key Hardware

```
CPU:  6× Arm Cortex-A78AE @ up to 1.5 GHz
GPU:  1024× CUDA cores (Ampere architecture)
      32× Tensor Cores
DLA:  1× Deep Learning Accelerator (up to 10 TOPS)
RAM:  8GB LPDDR5 (shared between CPU + GPU)
Storage: microSD + M.2 NVMe slot (2280)
I/O:
  1× USB 3.2 Gen2 Type-A
  1× USB 3.2 Gen2 Type-C (DisplayPort alt mode)
  1× Gigabit Ethernet
  40-pin GPIO header (I2C, SPI, UART, PWM, I2S)
  M.2 Key M (NVMe SSD)
  M.2 Key E (WiFi/BT)
  Camera connector (CSI-2, 2-lane)
```

> **Tensor Cores** are the hardware units that deliver most of the **40 AI TOPS** on Orin Nano: they perform matrix multiply-accumulate (MMA) in one instruction for FP16/INT8, which is why FP16 and INT8 inference are so much faster than FP32. For a detailed explanation of tensor core architecture and how they work, see [**Orin Nano — Tensor Core Architecture and How It Works**](Orin-Nano-Tensor/Guide.md).

### Memory Architecture (Critical for AI)

The Orin Nano uses **unified memory** — CPU and GPU share the same physical LPDDR5 pool:

```
CPU processes ←──────────────── 8GB LPDDR5 ────────────────→ GPU processes
                          No PCIe transfer needed!
                    Tensors live in unified address space
```

This means zero-copy GPU inference: camera frames captured by CPU stay in place and the GPU reads them directly without copying. This is a major advantage over discrete GPU systems.

---

## 3. Installing JetPack — Step by Step

### What is JetPack?

JetPack is NVIDIA's full SDK stack for Jetson. It bundles:
- **L4T (Linux for Tegra)**: Ubuntu-based OS with Jetson kernel + drivers
- **CUDA Toolkit**: GPU programming runtime
- **cuDNN**: GPU-accelerated deep learning primitives
- **TensorRT**: optimized inference engine
- **VPI** (Vision Programming Interface): hardware-accelerated CV
- **DeepStream**: video analytics pipeline SDK
- **Multimedia API**: camera/video capture

### JetPack Version → Software Stack

| JetPack | Ubuntu | CUDA  | cuDNN | TensorRT | L4T     |
|---------|--------|-------|-------|----------|---------|
| 5.1.4   | 20.04  | 11.4  | 8.6   | 8.6      | 35.6.x  |
| 6.0     | 22.04  | 12.2  | 9.0   | 10.0     | 36.3.x  |
| 6.1     | 22.04  | 12.6  | 9.3   | 10.3     | 36.4.x  |
| 6.2     | 22.04  | 12.8  | 9.7   | 10.7     | 36.5.x  |

### Method 1: SD Card Image (Easiest — Dev Kit Only)

The Orin Nano Developer Kit can boot from microSD. This is the fastest way to get started.

```bash
# Step 1: Download the SD card image from NVIDIA
# Go to: developer.nvidia.com/embedded/jetpack
# Select: Jetson Orin Nano Developer Kit → JetPack 6.x → SD Card Image
# File: jp6x-orin-nano-sd-card-image.zip  (~15GB)

# Step 2: Flash with Balena Etcher (GUI) or dd (CLI)
# Using dd (Linux):
unzip jp6x-orin-nano-sd-card-image.zip
sudo dd if=jp6x-orin-nano-sd-card-image.img of=/dev/sdX bs=1M status=progress
# Replace /dev/sdX with your SD card device (check with lsblk)

# Step 3: Insert SD card, connect HDMI + keyboard + power
# System boots into Ubuntu 22.04 setup wizard
```

### Method 2: SDK Manager (Full Control, Host PC Required)

SDK Manager gives you precise control over which JetPack components to install and enables NVMe boot configuration.

**Requirements:**
- Host PC running Ubuntu 20.04 or 22.04 (native, not VM recommended)
- USB-C cable (data, not charge-only) between host PC and Jetson USB-C port
- Jetson Orin Nano Dev Kit in **recovery mode**

```bash
# ── HOST PC SETUP ──────────────────────────────────────────

# Step 1: Download SDK Manager
# developer.nvidia.com/sdk-manager → download .deb
sudo dpkg -i sdkmanager_*.deb
sudo apt-get install -f   # fix any dependency issues

# Step 2: Launch SDK Manager
sdkmanager

# Step 3: Log in with NVIDIA Developer account

# ── JETSON: ENTER RECOVERY MODE ────────────────────────────
# On Orin Nano Dev Kit:
#   1. Power OFF the board
#   2. Hold the RECOVERY button (3-pin header, middle pin to GND)
#   3. While holding, press POWER button
#   4. Release RECOVERY button after 2 seconds
#   5. Connect USB-C from host PC to Jetson USB-C port

# Verify Jetson is detected in recovery mode:
lsusb | grep NVIDIA
# Should show: NVIDIA Corp. APX
```

#### JetPack 6.2.2 Component Breakdown (SDK Manager)

Below is the exact component list for **JetPack 6.2.2** on **Jetson Orin Nano 8GB Developer Kit** as shown in SDK Manager. Understand what each component does — this maps directly to the 8-layer stack.

**Jetson Linux (L4T) — Flash to device**

| Component | Version | Size | Layer | What it does |
|-----------|---------|------|:-----:|-------------|
| Jetson Linux Image | 36.5 | 2455 MB | L4 | Full L4T root filesystem image (Ubuntu 22.04 base + NVIDIA drivers) |
| Drivers for Jetson | 36.5 | 715 MB | L3/L4 | Kernel modules: nvgpu, nvhost, camera drivers, display, PCIe, DMA |
| File System and OS | 36.5 | 1749 MB | L4 | Ubuntu 22.04 rootfs with NVIDIA-specific packages and config |
| **Flash Jetson Linux** | **36.5** | **9520 MB** | — | Writes all above to eMMC/NVMe via USB recovery mode |

**Jetson Runtime Components — Install on target**

| Component | Version | Size | Layer | What it does |
|-----------|---------|------|:-----:|-------------|
| Additional Setups | — | 4.1 MB | L4 | Post-flash configuration scripts |
| DateTime Target Setup | — | 1.0 MB | L4 | NTP / hardware clock sync |
| GStreamer | 36.5 | 1.4 MB | L3 | Multimedia pipeline framework (feeds DeepStream, camera pipelines) |
| DLA Compiler | 36.5 | 2.7 MB | L2 | Compiles TensorRT layers for Deep Learning Accelerator hardware |

**CUDA & AI Runtime — The inference stack**

| Component | Version | Size | Layer | What it does |
|-----------|---------|------|:-----:|-------------|
| CUDA Runtime | 12.6 | 2199 MB | L3 | GPU runtime: cudart, cudaMemcpy, streams, events, kernel launch |
| CUDA X-AI Runtime | — | 1197 MB | L1/L3 | cuBLAS, cuFFT, cuSPARSE, cuSOLVER — math libraries for AI workloads |
| cuDNN Runtime | 9.3 | 778 MB | L1/L2 | Optimized conv, attention, normalization kernels for neural networks |
| TensorRT Runtime | 10.3 | 419 MB | L2/L3 | Graph optimization, layer fusion, INT8/FP16 engine build + execution |

**Computer Vision Runtime**

| Component | Version | Size | Layer | What it does |
|-----------|---------|------|:-----:|-------------|
| OpenCV Runtime | 4.8 | 12.1 MB | L1 | Image processing, feature detection, camera calibration |
| cuPVA Runtime | 2.5 | 0.3 MB | L3 | Programmable Vision Accelerator — hardware CV engine on Orin |
| VPI Runtime | 3.2 | 33.0 MB | L1/L3 | Vision Programming Interface — unified API for CPU/GPU/PVA/DLA vision ops |

**Container & Multimedia**

| Component | Version | Size | Layer | What it does |
|-----------|---------|------|:-----:|-------------|
| NVIDIA Container Runtime | — | 4.9 MB | L3/L4 | Run NGC containers with GPU access (docker + nvidia-container-toolkit) |
| Multimedia API | 36.5 | 71.9 MB | L3 | NVDEC/NVENC hardware video decode/encode, V4L2 camera, ISP access |

**Jetson SDK Components — Developer tools (host + target)**

| Component | Version | Size | Layer | What it does |
|-----------|---------|------|:-----:|-------------|
| CUDA Toolkit for L4T | 12.6 | 2196 MB | L3 | nvcc compiler, cuda headers, samples — build CUDA kernels on-device |
| Nsight Systems | 2024.5 | 299 MB | L1–L3 | System-wide profiler: timeline view of CPU, GPU, DLA, memory, streams |
| Nsight Graphics | 2024.2 | 196 MB | L1 | Graphics debugger and frame profiler |
| DeepStream | 7.1 | 603 MB | L1/L3 | GStreamer-based multi-stream video analytics pipeline (nvinfer + tracker) |
| GXF Runtime | 4.1 | 466 MB | L3 | Graph eXecution Framework — dataflow runtime for Holoscan/sensor pipelines |
| Jetson Platform Services | 2.0 | 0.1 MB | L4 | System services for fleet management, monitoring, diagnostics |

> **Total download:** ~22 GB. After flash + install, the Orin Nano rootfs uses ~14 GB.
>
> **What to select:** For AI inference work, select everything. For minimal edge deployment, you can skip Nsight Graphics, DeepStream, and GXF — add them later via `apt`.

```bash
# ── SDK MANAGER STEPS ──────────────────────────────────────
# 1. Select: Jetson Orin Nano [8GB Developer Kit]
# 2. JetPack version: 6.2.2
# 3. Select target components (see tables above)
# 4. Click Continue, accept licenses
# 5. Flashing starts — takes 10–20 minutes
```

### Method 3: NVMe Boot (Recommended for Production)

microSD is slow (max ~100 MB/s read) and has limited write endurance. NVMe SSD gives:
- Read: ~3500 MB/s (35× faster than SD)
- Much higher write endurance
- Faster model loading and dataset access

```bash
# After initial boot from SD card:

# Step 1: Install NVMe SSD in M.2 Key M slot (2280 form factor)
# (power off first, insert SSD, power on)

# Step 2: On Jetson, clone SD card to NVMe
sudo apt-get install pv
sudo dd if=/dev/mmcblk0 | pv | sudo dd of=/dev/nvme0n1 bs=4M status=progress

# Step 3: Expand NVMe partition
sudo parted /dev/nvme0n1 resizepart 1 100%
sudo resize2fs /dev/nvme0n1p1

# Step 4: Update UEFI boot order to prefer NVMe
# On Orin, use: sudo nvbootctrl set-active-boot-slot 0
# Or use NVIDIA's provided extlinux script:
sudo /opt/nvidia/jetson-io/config-by-hardware.py

# Step 5: Reboot and verify
df -h   # root should show NVMe size, not SD size
lsblk   # verify boot partition
```

### Post-Install Verification

```bash
# Check JetPack version
cat /etc/nv_tegra_release

# Check CUDA
nvcc --version
# Expected: Cuda compilation tools, release 12.x

# Check GPU
nvidia-smi
# Shows: Orin GPU, memory, driver version

# Check TensorRT
dpkg -l | grep tensorrt
python3 -c "import tensorrt; print(tensorrt.__version__)"

# Run NVIDIA system profiler
sudo tegrastats
# Shows: CPU%, GPU%, RAM, power, temperature — live
```

---

## 4. Upgrading JetPack on Orin Nano 8GB

### Can You Upgrade In-Place?

**Short answer:** Minor version upgrades (e.g., 6.0 → 6.1) can sometimes be done via `apt`. Major version upgrades (e.g., 5.x → 6.x) **always require reflashing**.

### Minor Version Upgrade via apt (Same Major Version)

```bash
# Add NVIDIA Jetson apt repository
sudo apt-get update
sudo apt-get upgrade   # upgrades L4T packages if available

# Check what version you'll get before upgrading:
apt-cache policy nvidia-l4t-core

# Upgrade specific JetPack components
sudo apt-get install --only-upgrade \
  cuda-toolkit-12-x \
  libcudnn9 \
  tensorrt

# After upgrade, reboot
sudo reboot

# Verify new version
cat /etc/nv_tegra_release
```

**Warning:** In-place apt upgrades can occasionally break dependencies. Always snapshot/backup before upgrading production systems.

### Major Version Upgrade (5.x → 6.x) — Reflash Required

```
JetPack 5.x (Ubuntu 20.04, CUDA 11.x)
             ↓  Cannot apt-upgrade across major versions
JetPack 6.x (Ubuntu 22.04, CUDA 12.x)

Why reflash?
  - Different Ubuntu base (20.04 → 22.04)
  - Different L4T kernel series (35.x → 36.x)
  - Different bootchain
  - Different partition layout
```

```bash
# Process for Orin Nano 8GB: 5.x → 6.x
# 1. Backup your application data and custom configs
rsync -av /home/user/ /media/backup/

# 2. Note all installed packages
dpkg --get-selections > /media/backup/installed_packages.txt

# 3. Reflash using SDK Manager (Method 2 above)
#    Select JetPack 6.x on host PC

# 4. After flash, reinstall your packages and restore data
# 5. Reinstall Python packages (new Python version on Ubuntu 22.04)
pip3 install -r /media/backup/requirements.txt
```

### Upgrade Decision Matrix

| Current → Target        | Method       | Time    | Risk | Data Loss |
|-------------------------|--------------|---------|------|-----------|
| JP6.0 → JP6.1           | apt upgrade  | 10 min  | Low  | No        |
| JP6.0 → JP6.2           | apt or reflash | 15 min | Med | No (apt) |
| JP5.1.x → JP6.x         | Reflash only | 30 min  | Med  | Yes*      |
| JP4.x → JP5.x/6.x       | Reflash only | 30 min  | High | Yes*      |

*Back up data to external storage before reflash.

---

## 5. Benefits of Higher JetPack Versions

### JetPack 6.x vs 5.x (Orin Nano Context)

```
JetPack 5.1.4                    JetPack 6.2
─────────────────                ─────────────────────────
Ubuntu 20.04                  →  Ubuntu 22.04 (LTS until 2027)
Python 3.8                    →  Python 3.10
CUDA 11.4                     →  CUDA 12.8
cuDNN 8.6                     →  cuDNN 9.7
TensorRT 8.6                  →  TensorRT 10.7
GCC 9                         →  GCC 11
OpenCV 4.5                    →  OpenCV 4.8
ROS2 Foxy (EOL)               →  ROS2 Humble/Jazzy (supported)
VPI 2.x                       →  VPI 3.x (CUDA graphs, better CPU)
DeepStream 6.x                →  DeepStream 7.x
```

### Concrete Benefits

**CUDA 12.x improvements:**
```
- CUDA Graph launch overhead: -40% vs CUDA 11.x
- INT8/FP8 Tensor Core utilization: improved Ampere support
- cudaMemcpyAsync improvements: better overlap with computation
- Cooperative Groups improvements
```

**TensorRT 10.x improvements:**
```
- Strongly Typed mode: explicit tensor types throughout the network
- Faster engine build times
- Better INT8 calibration
- New plugins for attention layers (Transformers on edge)
- Improved BF16 support
- Better quantization-aware training (QAT) support
```

**cuDNN 9.x improvements:**
```
- New graph-based API (replaces legacy API)
- Better memory reuse across operations
- Fused attention (FlashAttention-style) on Ampere
```

**Security improvements in JetPack 6:**
```
- Secure Boot with anti-rollback protection
- UEFI Secure Boot support
- Disk encryption (DM-Crypt) officially supported
- Measured Boot support
- OTA (Over-the-Air) update infrastructure built-in
```

**ROS2 compatibility:**
```
JetPack 5.x → Ubuntu 20.04 → ROS2 Foxy (EOL 2023) or Galactic (EOL 2022)
JetPack 6.x → Ubuntu 22.04 → ROS2 Humble (LTS until 2027) ← Use this
```

**Practical rule:** Always use the latest JetPack unless you have a specific reason not to (e.g., a dependency that hasn't been ported yet). Newer JetPack = better performance, better security, longer support.

---

## 6. Porting AI Models to Jetson

### The Porting Pipeline

```
Training Environment          Jetson Deployment
(Desktop/Cloud)               (Orin Nano 8GB)

PyTorch / TensorFlow          TensorRT Engine
     model.pt          →      model.engine
     model.pb          →      (compiled, optimized,
     model.onnx        →       quantized)

Steps:
  1. Train model (any framework)
  2. Export to ONNX (universal format)
  3. Convert ONNX → TensorRT engine
  4. Run inference with TensorRT runtime
```

### Step 1: Export PyTorch Model to ONNX

```python
# On training machine (or Jetson if RAM allows)
import torch
import torch.onnx

model = MyModel()
model.load_state_dict(torch.load("model.pt"))
model.eval()

# Define dummy input matching your model's expected input
dummy_input = torch.randn(1, 3, 640, 640)   # e.g., YOLO input

torch.onnx.export(
    model,
    dummy_input,
    "model.onnx",
    opset_version=17,              # use latest stable
    input_names=["images"],
    output_names=["output"],
    dynamic_axes={
        "images": {0: "batch"},    # dynamic batch size
        "output": {0: "batch"}
    }
)

# Verify ONNX model
import onnx
model_onnx = onnx.load("model.onnx")
onnx.checker.check_model(model_onnx)
print("ONNX export successful")
```

### Step 2: Build TensorRT Engine on Jetson

**Always build TensorRT engines ON the target Jetson** — engines are hardware-specific.

```bash
# Method A: trtexec (command-line tool, easiest)

# FP32 (no quantization)
trtexec --onnx=model.onnx \
        --saveEngine=model_fp32.engine \
        --verbose

# FP16 (2× speedup, <1% accuracy loss typically)
trtexec --onnx=model.onnx \
        --saveEngine=model_fp16.engine \
        --fp16 \
        --verbose

# INT8 (4× speedup, needs calibration data)
trtexec --onnx=model.onnx \
        --saveEngine=model_int8.engine \
        --int8 \
        --calib=calib_data.cache \
        --verbose

# Dynamic batch size (1–16)
trtexec --onnx=model.onnx \
        --saveEngine=model_dynamic.engine \
        --fp16 \
        --minShapes=images:1x3x640x640 \
        --optShapes=images:4x3x640x640 \
        --maxShapes=images:16x3x640x640
```

```python
# Method B: Python TensorRT API (more control)
import tensorrt as trt

TRT_LOGGER = trt.Logger(trt.Logger.WARNING)

def build_engine(onnx_path, engine_path, fp16=True):
    with trt.Builder(TRT_LOGGER) as builder, \
         builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)) as network, \
         trt.OnnxParser(network, TRT_LOGGER) as parser, \
         builder.create_builder_config() as config:

        config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 2 << 30)  # 2GB

        if fp16:
            config.set_flag(trt.BuilderFlag.FP16)

        with open(onnx_path, 'rb') as f:
            if not parser.parse(f.read()):
                for error in range(parser.num_errors):
                    print(parser.get_error(error))
                raise RuntimeError("ONNX parse failed")

        serialized = builder.build_serialized_network(network, config)
        with open(engine_path, 'wb') as f:
            f.write(serialized)
        print(f"Engine saved to {engine_path}")

build_engine("model.onnx", "model_fp16.engine", fp16=True)
```

### Step 3: Run TensorRT Inference

```python
import tensorrt as trt
import numpy as np
import pycuda.driver as cuda
import pycuda.autoinit

class TRTInferencer:
    def __init__(self, engine_path):
        with open(engine_path, 'rb') as f:
            runtime = trt.Runtime(trt.Logger(trt.Logger.WARNING))
            self.engine = runtime.deserialize_cuda_engine(f.read())
        self.context = self.engine.create_execution_context()

        # Allocate device buffers
        self.inputs, self.outputs, self.bindings = [], [], []
        for binding in self.engine:
            size = trt.volume(self.engine.get_binding_shape(binding))
            dtype = trt.nptype(self.engine.get_binding_dtype(binding))
            host_mem = cuda.pagelocked_empty(size, dtype)
            device_mem = cuda.mem_alloc(host_mem.nbytes)
            self.bindings.append(int(device_mem))
            if self.engine.binding_is_input(binding):
                self.inputs.append({'host': host_mem, 'device': device_mem})
            else:
                self.outputs.append({'host': host_mem, 'device': device_mem})
        self.stream = cuda.Stream()

    def infer(self, input_data):
        np.copyto(self.inputs[0]['host'], input_data.ravel())
        cuda.memcpy_htod_async(self.inputs[0]['device'], self.inputs[0]['host'], self.stream)
        self.context.execute_async_v2(bindings=self.bindings, stream_handle=self.stream.handle)
        cuda.memcpy_dtoh_async(self.outputs[0]['host'], self.outputs[0]['device'], self.stream)
        self.stream.synchronize()
        return self.outputs[0]['host']

# Usage
inferencer = TRTInferencer("model_fp16.engine")
frame = np.random.randn(1, 3, 640, 640).astype(np.float32)
result = inferencer.infer(frame)
```

### DLA (Deep Learning Accelerator) — Extra Efficiency

The Orin Nano has 1 DLA engine. For supported layers, DLA runs at ~10 TOPS while freeing the GPU for other tasks.

```python
# Enable DLA in TensorRT engine build
config.default_device_type = trt.DeviceType.DLA
config.DLA_core = 0           # Orin Nano has DLA 0 only
config.set_flag(trt.BuilderFlag.GPU_FALLBACK)  # GPU fallback for unsupported layers
config.set_flag(trt.BuilderFlag.FP16)           # DLA requires FP16 or INT8

# Check which layers run on DLA vs GPU after building
# Use: trtexec --onnx=model.onnx --useDLACore=0 --verbose 2>&1 | grep "DLA"
```

---

## 7. ROS2 Integration

### Installing ROS2 Humble on JetPack 6 (Ubuntu 22.04)

```bash
# Add ROS2 apt repository
sudo apt install -y software-properties-common curl
sudo curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key \
    -o /usr/share/keyrings/ros-archive-keyring.gpg
echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] \
    http://packages.ros.org/ros2/ubuntu $(. /etc/os-release && echo $UBUNTU_CODENAME) main" \
    | sudo tee /etc/apt/sources.list.d/ros2.list

sudo apt update
sudo apt install -y ros-humble-desktop          # full install with RViz
# Or for minimal footprint:
sudo apt install -y ros-humble-ros-base         # no GUI tools

# Install build tools
sudo apt install -y python3-colcon-common-extensions python3-rosdep

# Initialize rosdep
sudo rosdep init
rosdep update

# Add to .bashrc
echo "source /opt/ros/humble/setup.bash" >> ~/.bashrc
source ~/.bashrc
```

### ROS2 + TensorRT: AI Node Architecture

```
Camera (CSI/USB) ─→ image_raw ─→ [preprocessing node] ─→ [TensorRT inference node]
                                                                ↓
LiDAR ──────────→ scan ─────────→ [pointcloud node] ──→ [sensor fusion node]
                                                                ↓
IMU ────────────→ imu ──────────→ [EKF node] ───────→ [object tracker]
                                                                ↓
                                                     [control output node]
                                                                ↓
                                                     cmd_vel / actuator commands
```

### Writing a TensorRT Inference ROS2 Node

```python
#!/usr/bin/env python3
# ros2_trt_node.py

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from vision_msgs.msg import Detection2DArray, Detection2D, BoundingBox2D
from cv_bridge import CvBridge
import numpy as np
import cv2
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit

class TRTDetectionNode(Node):
    def __init__(self):
        super().__init__('trt_detection_node')

        # Parameters
        self.declare_parameter('engine_path', 'model_fp16.engine')
        self.declare_parameter('confidence_threshold', 0.5)
        self.declare_parameter('input_size', [640, 640])

        engine_path = self.get_parameter('engine_path').value
        self.conf_thresh = self.get_parameter('confidence_threshold').value

        # Load TensorRT engine
        self.engine = self.load_engine(engine_path)
        self.context = self.engine.create_execution_context()
        self.allocate_buffers()

        self.bridge = CvBridge()

        # Subscribe to camera image
        self.sub = self.create_subscription(
            Image,
            '/camera/image_raw',
            self.image_callback,
            10
        )

        # Publish detections
        self.pub = self.create_publisher(Detection2DArray, '/detections', 10)

        self.get_logger().info(f'TRT inference node started: {engine_path}')

    def load_engine(self, path):
        with open(path, 'rb') as f:
            runtime = trt.Runtime(trt.Logger(trt.Logger.WARNING))
            return runtime.deserialize_cuda_engine(f.read())

    def allocate_buffers(self):
        self.inputs, self.outputs, self.bindings = [], [], []
        self.stream = cuda.Stream()
        for binding in self.engine:
            size = trt.volume(self.engine.get_binding_shape(binding))
            host = cuda.pagelocked_empty(size, np.float32)
            device = cuda.mem_alloc(host.nbytes)
            self.bindings.append(int(device))
            if self.engine.binding_is_input(binding):
                self.inputs.append({'host': host, 'device': device})
            else:
                self.outputs.append({'host': host, 'device': device})

    def preprocess(self, img):
        img = cv2.resize(img, (640, 640))
        img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR→RGB, HWC→CHW
        img = img.astype(np.float32) / 255.0
        img = np.expand_dims(img, 0)               # add batch dim
        return np.ascontiguousarray(img)

    def infer(self, data):
        np.copyto(self.inputs[0]['host'], data.ravel())
        cuda.memcpy_htod_async(self.inputs[0]['device'], self.inputs[0]['host'], self.stream)
        self.context.execute_async_v2(self.bindings, self.stream.handle)
        cuda.memcpy_dtoh_async(self.outputs[0]['host'], self.outputs[0]['device'], self.stream)
        self.stream.synchronize()
        return self.outputs[0]['host'].copy()

    def image_callback(self, msg):
        # Convert ROS Image to OpenCV
        frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')

        # Preprocess
        input_data = self.preprocess(frame)

        # Run inference
        output = self.infer(input_data)

        # Parse output and publish
        detections = self.parse_detections(output, frame.shape, msg.header)
        self.pub.publish(detections)

    def parse_detections(self, output, img_shape, header):
        # Implement based on your model's output format
        # Example for YOLO-style output
        detections = Detection2DArray()
        detections.header = header
        # ... parse boxes, scores, classes from output ...
        return detections

def main(args=None):
    rclpy.init(args=args)
    node = TRTDetectionNode()
    rclpy.spin(node)
    rclpy.shutdown()

if __name__ == '__main__':
    main()
```

### ROS2 Node Performance: Executor Strategy

```python
# Single-threaded executor (default): simple, no race conditions
rclpy.spin(node)

# Multi-threaded executor: callbacks run in parallel
from rclpy.executors import MultiThreadedExecutor
from rclpy.callback_groups import ReentrantCallbackGroup, MutuallyExclusiveCallbackGroup

class MyNode(Node):
    def __init__(self):
        super().__init__('my_node')
        # Camera callback: can overlap with LiDAR callback
        camera_group = MutuallyExclusiveCallbackGroup()
        lidar_group = MutuallyExclusiveCallbackGroup()

        self.camera_sub = self.create_subscription(
            Image, '/camera/image_raw',
            self.camera_cb, 10,
            callback_group=camera_group
        )
        self.lidar_sub = self.create_subscription(
            PointCloud2, '/lidar/points',
            self.lidar_cb, 10,
            callback_group=lidar_group
        )

executor = MultiThreadedExecutor(num_threads=4)
executor.add_node(node)
executor.spin()
```

### ROS2 QoS for Sensor Data

```python
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy, QoSDurabilityPolicy

# Sensor data (real-time, drop old frames, don't retry)
sensor_qos = QoSProfile(
    reliability=QoSReliabilityPolicy.BEST_EFFORT,  # drop if can't deliver
    history=QoSHistoryPolicy.KEEP_LAST,
    depth=1,                                        # only care about latest frame
    durability=QoSDurabilityPolicy.VOLATILE
)

# Control commands (must be delivered, no drops)
control_qos = QoSProfile(
    reliability=QoSReliabilityPolicy.RELIABLE,
    history=QoSHistoryPolicy.KEEP_LAST,
    depth=10
)

self.camera_sub = self.create_subscription(
    Image, '/camera/image_raw',
    self.camera_cb, sensor_qos   # Use sensor QoS
)
self.cmd_pub = self.create_publisher(
    Twist, '/cmd_vel', control_qos  # Use reliable QoS
)
```

---

## 8. Optimizing AI Inference

> **Deep dive:** For DLA-specific optimization — hardware architecture, TensorRT DLA integration, supported layers, multi-engine scheduling (DLA + GPU), profiling, and production deployment patterns — see [**Orin Nano DLA Deep Dive**](Orin-Nano-DLA-Deep-Dive/Guide.md).

> **Deep dive:** For CUDA programming on Jetson — unified memory, kernel optimization, streams, shared memory, GStreamer/TensorRT integration, and power-aware CUDA patterns — see [**Orin Nano CUDA Programming**](Orin-Nano-CUDA-Programming/Guide.md).

> **Deep dive:** For real-time and deterministic inference — TensorRT engine optimization, DLA latency consistency, CUDA graphs, CPU isolation, multi-model scheduling, Triton serving, and p99 latency profiling — see [**Orin Nano Real-Time & Deterministic Inference**](Orin-Nano-Real-Time-Inference/Guide.md).
>
> For kernel-level RT Linux internals — PREEMPT_RT patch architecture, interrupt threading, lock primitives, priority inversion/PI mutexes, SCHED_DEADLINE, rt-tests suite, ARM Cortex-A78AE specifics, GICv3 tuning, WCET analysis, RT-safe kernel modules, and ftrace debugging — see [**Orin Nano RT Linux Deep Dive**](Orin-Nano-RT-Linux-Deep-Dive/Guide.md).

### Precision Tradeoffs

```
Precision    Memory   Speed      Accuracy   Use Case
FP32         100%     1×         Baseline   Training, debug
FP16         50%      2–3×       ~0.1% drop General inference
INT8         25%      3–5×       ~1% drop   Production
INT4         12.5%    4–8×       ~3% drop   Very edge, LLMs

On Orin Nano (Ampere GPU):
  FP16 Tensor Cores: native support, excellent
  INT8 Tensor Cores: native support, fastest path
  FP32: no Tensor Core acceleration
```

### CUDA Streams: Overlapping Work

```python
import pycuda.driver as cuda

stream_inference = cuda.Stream()
stream_preprocess = cuda.Stream()

# Pipeline: while GPU is running inference on frame N,
#           CPU is preprocessing frame N+1

with cuda.Stream() as s1, cuda.Stream() as s2:
    # Frame N: copy to GPU on s1
    cuda.memcpy_htod_async(gpu_input_n, cpu_frame_n, s1)

    # Frame N: run inference on s1
    context.execute_async_v2(bindings, s1.handle)

    # Frame N+1: preprocessing on CPU happens concurrently
    cpu_frame_n1 = preprocess(next_raw_frame)

    # Sync s1, get result
    s1.synchronize()
    cuda.memcpy_dtoh_async(cpu_output_n, gpu_output_n, s1)
```

### CUDA Graphs: Reduce Launch Overhead

For a fixed inference workload (same input shape every call), CUDA Graphs eliminate kernel launch overhead:

```python
import torch

model = model.cuda().half()

# Warm up
dummy = torch.randn(1, 3, 640, 640, device='cuda', dtype=torch.float16)
for _ in range(3):
    _ = model(dummy)

# Capture CUDA graph
g = torch.cuda.CUDAGraph()
with torch.cuda.graph(g):
    output = model(dummy)

# Replay graph (ultra-low overhead)
def fast_infer(x):
    dummy.copy_(x)    # update input in-place
    g.replay()        # replay captured graph
    return output.clone()
```

### Batch Inference Strategy

```
Single-frame inference (typical naive approach):
  Frame → [wait] → Model → Result → Frame → [wait] → ...
  Throughput: 1 frame / inference_time

Batched inference (correct approach):
  Accumulate frames → [batch=4] → Model → 4 results
  Throughput: 4 frames / inference_time (same GPU time!)
  Latency: slightly higher, but throughput multiplied

Dynamic batching: accept 1–N frames, fill timeout or max batch
```

### Profiling the Inference Pipeline

```bash
# 1. Quick throughput benchmark
trtexec --loadEngine=model_fp16.engine \
        --batch=1 \
        --iterations=100 \
        --warmUp=500 \
        --avgRuns=100

# 2. System-wide profiling with Nsight Systems
nsys profile \
    --trace=cuda,cudnn,tensorrt,osrt \
    --output=profile \
    python3 inference_script.py

# View report:
nsys-ui profile.qdrep

# 3. GPU kernel profiling with Nsight Compute
ncu --set full \
    --target-processes all \
    python3 inference_script.py

# 4. Jetson-specific: tegrastats
sudo tegrastats --interval 100 | tee tegrastats.log
```

### Latency Measurement

```python
import time
import numpy as np

def benchmark_inference(inferencer, input_data, n_runs=200, warmup=50):
    # Warmup (important: first runs include JIT compilation)
    for _ in range(warmup):
        inferencer.infer(input_data)

    # Measure
    times = []
    for _ in range(n_runs):
        t0 = time.perf_counter()
        inferencer.infer(input_data)
        times.append((time.perf_counter() - t0) * 1000)  # ms

    times = np.array(times)
    print(f"Latency: mean={times.mean():.2f}ms  "
          f"p50={np.percentile(times,50):.2f}ms  "
          f"p95={np.percentile(times,95):.2f}ms  "
          f"p99={np.percentile(times,99):.2f}ms")
    print(f"Throughput: {1000/times.mean():.1f} FPS")

benchmark_inference(inferencer, dummy_input)
```

---

## 9. End-to-End AI Pipeline: Sensors → Inference → Control

> **Deep dive:** For hardware video codecs, GStreamer accelerated pipelines, and DeepStream SDK — NVDEC/NVENC specs, transcoding, multi-stream inference, analytics, zero-copy video pipeline, and RTSP serving — see [**Orin Nano Video Codec & DeepStream**](Orin-Nano-Video-Codec-DeepStream/Guide.md).

### Full Pipeline Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│  SENSOR LAYER                                                        │
│  Camera (30fps) ──► CSI/USB → V4L2 → CUDA buffer                   │
│  LiDAR (10Hz)  ──► Ethernet/USB → PointCloud buffer                 │
│  IMU (100Hz)   ──► I2C/SPI → ring buffer                            │
└──────────────────────────┬──────────────────────────────────────────┘
                           │
┌──────────────────────────▼──────────────────────────────────────────┐
│  PREPROCESSING LAYER (GPU — CUDA/VPI)                                │
│  Image: resize, normalize, color convert (CUDA)                      │
│  PointCloud: voxelization, ground removal (CUDA)                     │
│  IMU: integration, bias correction (CPU)                             │
└──────────────────────────┬──────────────────────────────────────────┘
                           │
┌──────────────────────────▼──────────────────────────────────────────┐
│  INFERENCE LAYER (GPU — TensorRT)                                    │
│  Object Detection: YOLO / SSD on camera frames                       │
│  PointCloud Detection: PointPillars on LiDAR scan                    │
│  Fusion: BEVFusion / simple box fusion                               │
└──────────────────────────┬──────────────────────────────────────────┘
                           │
┌──────────────────────────▼──────────────────────────────────────────┐
│  FUSION & TRACKING LAYER (CPU + GPU)                                 │
│  EKF/UKF: fuse IMU + camera + LiDAR detections                      │
│  Object tracker: Hungarian algorithm + Kalman filter                 │
└──────────────────────────┬──────────────────────────────────────────┘
                           │
┌──────────────────────────▼──────────────────────────────────────────┐
│  CONTROL LAYER (CPU — real-time)                                     │
│  Path planner (A*, DWA, MPC)                                         │
│  Velocity/steering commands → actuators                              │
└─────────────────────────────────────────────────────────────────────┘
```

### Zero-Copy Camera Pipeline (Optimized)

```python
# Use GStreamer + nvarguscamerasrc for zero-copy CSI camera pipeline
import cv2

# CSI camera (native, zero-copy, hardware-accelerated)
gst_pipeline = (
    "nvarguscamerasrc sensor-id=0 ! "
    "video/x-raw(memory:NVMM), width=1920, height=1080, framerate=30/1 ! "
    "nvvidconv ! "                                    # stays in GPU memory
    "video/x-raw, format=BGRx ! "
    "videoconvert ! "
    "video/x-raw, format=BGR ! "
    "appsink drop=1"
)

cap = cv2.VideoCapture(gst_pipeline, cv2.CAP_GSTREAMER)
```

```python
# Better: use pycuda + nvargus for true zero-copy (frame stays in GPU)
# or use NVIDIA's Jetson.Utils library:
import jetson.utils as ju

camera = ju.videoSource("csi://0", argv=["--width=1920", "--height=1080", "--framerate=30"])
display = ju.videoOutput("display://0")

while True:
    img = camera.Capture()         # CUDA image, stays on GPU
    # img is a jetson.utils.cudaImage — pointer into shared GPU memory
    # pass directly to TensorRT without any CPU roundtrip
    result = detect(img)
    display.Render(img)
```

### Pipeline Timing Budget (Example: 30 FPS = 33ms per frame)

```
Budget: 33ms total for one pipeline cycle at 30 FPS

Camera capture:     2ms   (DMA from ISP to memory)
Preprocessing:      3ms   (GPU: resize, normalize)
TensorRT inference: 10ms  (GPU: FP16 detection model)
LiDAR processing:   5ms   (GPU: voxelization, PointPillars)
Sensor fusion:      4ms   (CPU: EKF update)
Object tracking:    2ms   (CPU: Hungarian + Kalman)
Path planning:      4ms   (CPU: DWA or A*)
Control output:     1ms   (CAN/UART command send)
Margin:             2ms
─────────────────────────
Total:             33ms = 30 FPS ✓
```

### Pipelining with Threads

```python
import threading
import queue
import time

# Thread-safe queues between pipeline stages
raw_frame_q = queue.Queue(maxsize=2)     # camera → preprocess
gpu_frame_q  = queue.Queue(maxsize=2)    # preprocess → inference
detection_q  = queue.Queue(maxsize=2)    # inference → fusion
control_q    = queue.Queue(maxsize=2)    # fusion → control

def camera_thread():
    """Runs at camera frequency (30 Hz)"""
    while True:
        ret, frame = cap.read()
        if not raw_frame_q.full():
            raw_frame_q.put_nowait(frame)

def preprocess_thread():
    """GPU preprocessing"""
    while True:
        frame = raw_frame_q.get()
        processed = gpu_preprocess(frame)  # CUDA kernel
        gpu_frame_q.put(processed)

def inference_thread():
    """TensorRT inference"""
    while True:
        processed = gpu_frame_q.get()
        detections = trt_infer(processed)
        detection_q.put(detections)

def control_thread():
    """Control loop — must run at fixed rate"""
    while True:
        if not detection_q.empty():
            detections = detection_q.get_nowait()
            cmd = compute_control(detections)
            send_command(cmd)
        time.sleep(0.01)  # 100 Hz control loop

# Start all threads
threads = [
    threading.Thread(target=camera_thread, daemon=True),
    threading.Thread(target=preprocess_thread, daemon=True),
    threading.Thread(target=inference_thread, daemon=True),
    threading.Thread(target=control_thread, daemon=True),
]
for t in threads:
    t.start()
```

---

## 10. LiDAR, Camera, IMU — Practical Integration

> **Deep dive:** For camera subsystem internals — NVCSI/VI/ISP hardware, sensor driver development, device tree configuration, ISP tuning, Libargus API, multi-camera sync, and camera-to-CUDA zero-copy — see [**Orin Nano Camera ISP & Sensor Bringup**](Orin-Nano-Camera-ISP-Sensor-Bringup/Guide.md).

> **Deep dive:** For 40-pin header peripherals — GPIO, I2C, SPI, UART, PWM, CAN bus, pin multiplexing, sensor integration, motor control, and custom device tree overlays — see [**Orin Nano GPIO/SPI/I2C/CAN**](Orin-Nano-GPIO-SPI-I2C-CAN/Guide.md).

### Camera Integration

#### CSI Camera (Recommended for performance)

```bash
# Check if CSI camera is detected
sudo dmesg | grep imx          # IMX219, IMX477 modules
v4l2-ctl --list-devices

# Test: capture a frame
nvgstcapture-1.0 --sensor-id=0

# List available formats
v4l2-ctl --device=/dev/video0 --list-formats-ext
```

```python
# Camera calibration with OpenCV (critical for accurate 3D projection)
import cv2
import numpy as np

# After capturing calibration images with checkerboard:
objpoints = []  # 3D world points
imgpoints = []  # 2D image points

# ... fill objpoints and imgpoints from checkerboard detection ...

ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(
    objpoints, imgpoints, gray.shape[::-1], None, None
)

print(f"Camera matrix:\n{camera_matrix}")
print(f"Distortion coeffs: {dist_coeffs}")

# Save for use in ROS2 camera_info topic
np.save('camera_matrix.npy', camera_matrix)
np.save('dist_coeffs.npy', dist_coeffs)
```

#### USB Camera Setup

```bash
# Identify camera
ls /dev/video*
v4l2-ctl --device=/dev/video0 --list-formats-ext

# Check USB speed (should be USB 3.x for high-res)
lsusb -t

# Set optimal format
v4l2-ctl --device=/dev/video0 \
    --set-fmt-video=width=1280,height=720,pixelformat=MJPG
```

### LiDAR Integration

#### RPLIDAR (Common, low-cost)

```bash
# Install RPLIDAR ROS2 driver
sudo apt-get install ros-humble-rplidar-ros

# Connect via USB, find port
ls /dev/ttyUSB*
sudo chmod 666 /dev/ttyUSB0

# Launch
ros2 launch rplidar_ros rplidar_a2_launch.py \
    serial_port:=/dev/ttyUSB0 \
    serial_baudrate:=115200 \
    frame_id:=laser
```

#### Velodyne LiDAR (High-end)

```bash
sudo apt-get install ros-humble-velodyne

# Velodyne connects via Ethernet (192.168.1.201 by default)
sudo ip addr add 192.168.1.100/24 dev eth0

ros2 launch velodyne_driver velodyne_driver_node-VLP16-launch.py
```

#### PointCloud Processing

```python
# ROS2 PointCloud2 → numpy
from sensor_msgs.msg import PointCloud2
import sensor_msgs_py.point_cloud2 as pc2
import numpy as np

def lidar_callback(self, msg):
    # Extract XYZ points
    points = np.array(list(pc2.read_points(
        msg, field_names=('x', 'y', 'z', 'intensity'),
        skip_nans=True
    )))

    if len(points) == 0:
        return

    xyz = points[:, :3]                    # shape [N, 3]
    intensity = points[:, 3:4]             # shape [N, 1]

    # Ground removal (simple height filter)
    mask = xyz[:, 2] > -0.3               # remove points below 30cm
    xyz_filtered = xyz[mask]

    # Pass to CUDA for voxelization
    self.process_pointcloud_gpu(xyz_filtered)
```

### IMU Integration

#### Reading IMU via I2C (MPU6050 / ICM42688)

```bash
# Enable I2C on Jetson GPIO header
sudo i2cdetect -y -r 1    # scan I2C bus 1
# Should show device address (0x68 for MPU6050)
```

```python
import smbus2
import struct
import time

class MPU6050:
    ADDR = 0x68
    PWR_MGMT_1 = 0x6B
    ACCEL_XOUT_H = 0x3B
    GYRO_XOUT_H = 0x43

    def __init__(self, bus_num=1):
        self.bus = smbus2.SMBus(bus_num)
        self.bus.write_byte_data(self.ADDR, self.PWR_MGMT_1, 0)  # wake up
        time.sleep(0.1)

    def _read_word(self, reg):
        high = self.bus.read_byte_data(self.ADDR, reg)
        low  = self.bus.read_byte_data(self.ADDR, reg + 1)
        val  = (high << 8) + low
        return val - 65536 if val >= 0x8000 else val

    def get_accel(self):
        ax = self._read_word(self.ACCEL_XOUT_H) / 16384.0     # g
        ay = self._read_word(self.ACCEL_XOUT_H + 2) / 16384.0
        az = self._read_word(self.ACCEL_XOUT_H + 4) / 16384.0
        return ax, ay, az

    def get_gyro(self):
        gx = self._read_word(self.GYRO_XOUT_H) / 131.0        # °/s
        gy = self._read_word(self.GYRO_XOUT_H + 2) / 131.0
        gz = self._read_word(self.GYRO_XOUT_H + 4) / 131.0
        return gx, gy, gz

# Publish as ROS2 Imu message
from sensor_msgs.msg import Imu

imu = Imu()
imu.header.stamp = self.get_clock().now().to_msg()
imu.header.frame_id = 'imu_link'
ax, ay, az = mpu.get_accel()
imu.linear_acceleration.x = ax * 9.81
imu.linear_acceleration.y = ay * 9.81
imu.linear_acceleration.z = az * 9.81
self.imu_pub.publish(imu)
```

### Extrinsic Calibration: Camera ↔ LiDAR

Extrinsic calibration finds the rigid transformation (rotation + translation) between sensor coordinate frames. This is critical for fusion.

```bash
# Install calibration tools
sudo apt-get install ros-humble-camera-calibration
pip3 install kalibr

# Kalibr calibration (recommended):
# 1. Print an AprilGrid target
# 2. Record a ROS2 bag with camera + IMU moving together
ros2 bag record -o calib_bag /camera/image_raw /imu/data

# 3. Run Kalibr
kalibr_calibrate_cameras \
    --bag calib_bag.bag \
    --topics /camera/image_raw \
    --models pinhole-equi \
    --target aprilgrid.yaml

# Camera-LiDAR extrinsic:
# Use: github.com/PJLab-ADLab/SensorsCalibration
```

```python
# Apply extrinsic transform: project LiDAR points into camera image
import numpy as np

# Extrinsic: 4×4 transform matrix (LiDAR frame → camera frame)
T_cam_lidar = np.array([
    [ 0.9998, -0.0052,  0.0191,  0.1500],   # example values
    [ 0.0049,  0.9999,  0.0104, -0.0050],
    [-0.0192, -0.0103,  0.9997,  0.0300],
    [ 0.0000,  0.0000,  0.0000,  1.0000]
])

# Camera intrinsic matrix
K = np.array([[fx, 0, cx],
              [0, fy, cy],
              [0,  0,  1]])

def lidar_to_image(points_lidar, T_cam_lidar, K):
    """Project LiDAR points onto camera image plane"""
    N = points_lidar.shape[0]
    pts_h = np.hstack([points_lidar[:, :3], np.ones((N, 1))])   # homogeneous
    pts_cam = (T_cam_lidar @ pts_h.T).T                          # camera frame

    # Keep only points in front of camera
    mask = pts_cam[:, 2] > 0
    pts_cam = pts_cam[mask]

    # Project to image
    pts_img = (K @ pts_cam[:, :3].T).T
    u = pts_img[:, 0] / pts_img[:, 2]
    v = pts_img[:, 1] / pts_img[:, 2]
    depth = pts_cam[:, 2]

    return u, v, depth, mask
```

### Time Synchronization Between Sensors

```python
# Use message_filters for approximate time synchronization
import message_filters
from sensor_msgs.msg import Image, PointCloud2

class FusionNode(Node):
    def __init__(self):
        super().__init__('fusion_node')

        self.camera_sub = message_filters.Subscriber(self, Image, '/camera/image_raw')
        self.lidar_sub  = message_filters.Subscriber(self, PointCloud2, '/lidar/points')

        # Synchronize: accept messages within 50ms of each other
        self.ts = message_filters.ApproximateTimeSynchronizer(
            [self.camera_sub, self.lidar_sub],
            queue_size=10,
            slop=0.05    # 50ms tolerance
        )
        self.ts.registerCallback(self.fused_callback)

    def fused_callback(self, camera_msg, lidar_msg):
        # Both messages are time-synchronized
        # Process together for accurate fusion
        pass
```

---

## 11. Device Tree Configuration

> **Deep dive:** For full kernel internals — source tree structure, build system, device tree architecture, driver model, camera kernel stack, nvgpu driver, boot time optimization, custom module development, and production hardening — see [**Orin Nano Kernel Internals & Customization**](Orin-Nano-Kernel-Internals/Guide.md).

Device tree configuration on NVIDIA Jetson platforms is critical for enabling I2C buses, configuring GPIO pins on the 40-pin header, and integrating camera sensors (CSI). Jetson systems utilize **Device Tree Overlays** (`.dtbo`) to modify base hardware configurations during boot, allowing for peripheral customization without re-flashing the entire board.

### What is Pinmux?

**Pinmux** (pin multiplexing) is the process of assigning each physical I/O pin on the SoC to a specific function. Jetson SoCs have **Multi-Purpose I/O (MPIO)** pins that can be configured as either:

- **GPIO** — General-Purpose I/O for custom logic, LEDs, buttons, etc.
- **SFIO** — Special Function I/O for dedicated interfaces (I2C, SPI, UART, PWM, etc.)

Signals on the 40-pin header pass through multiplexers; the device tree (or Pinmux Excel sheet for carrier-board design) configures which function each pin serves. For custom carrier boards, NVIDIA provides a Pinmux Excel template that generates `config-pinmux.dtsi`, `padvoltage-default.dtsi`, and `gpio-default.dtsi` for integration into the boot process.

**Reference:** [Jetson AGX Series Module Pinmux Application Note (DA-12015-001)](https://developer.download.nvidia.com/assets/embedded/secure/jetson/agx_orin/Jetson-AGX-Series-Module-Pinmux-Application-Note_DA-12015-001v1.0.pdf) — official NVIDIA pinmux process, Excel sheet usage, and mandatory interface pin direction configuration. *(Also available from [Jetson Download Center](https://developer.nvidia.com/embedded/jetson) — search for "Orin Pinmux" or "Pinmux Application Note".)*

### 10.1 Jetson-IO Tool: The Preferred Method

The easiest way to manage device trees on Jetson (Orin Nano/NX, Xavier NX) is the **jetson-io** tool, which generates overlays for the 40-pin header.

```bash
# Run the tool
sudo /opt/nvidia/jetson-io/jetson-io.py
```

**Purpose:** Configure pinmux for I2C, SPI, PWM, and UART, and save them as a custom `.dtbo`.

**Method:** Select "Configure Jetson 40pin Header", enable the desired interface (e.g., I2C1), and save to a new overlay.

### 10.2 I2C and GPIO Configuration

**Pinmuxing:** Signals on the 40-pin header travel from the Tegra SoC through multiplexers. The device tree configures these multiplexers for specific roles.

**I2C:** To change I2C speeds (e.g., 400 kHz to 100 kHz) or enable a bus, modify the `clock-frequency` and `status` properties in the DTSI file corresponding to the I2C controller.

**GPIO:** Pins can be configured as GPIO or SFIO (Special Function I/O). A minimal overlay defines the pin's Linux name (e.g., `PIO09` for pin 7) and sets its function to `gpio`.

**Voltage:** Jetson GPIO pins are **3.3V rated**; 5V signals can damage the board.

### 10.3 Camera/Sensor Integration

Camera integration requires modifying the device tree to define the sensor, I2C bus, CSI lanes, and power management.

- **tegra-camera-platform:** Define the `tegra-camera-platform` node in the device tree to describe the camera module, including position (e.g., `"rear"`) and orientation.
- **V4L2 Sensor Driver:** Use V4L2 framework 2.0 for new driver development, ensuring the `devname` matches the I2C address (e.g., `imx185 30-001a`).
- **Resources:** Specify regulators (`vana-supply`, `vdig-supply`) and GPIOs for reset/power-down (e.g., `H3-gpio`, `H6-gpio`) in the sensor DTSI file.

### 10.4 Creating and Applying Custom Overlays

For complex peripherals not covered by jetson-io, manual overlay creation is necessary.

1. **Write the DTS:** Create a `.dts` file, ensuring the `compatible` string matches your Jetson board (e.g., `"nvidia,p3509-0000+p3668-0001"` for Orin Nano).
2. **Compile:** Use the Device Tree Compiler (`dtc`):

   ```bash
   dtc -O dtb -o my-overlay.dtbo -@ my-overlay.dts
   ```

3. **Apply:** Move the `.dtbo` file to `/boot/` and update `/boot/extlinux/extlinux.conf` to load it using the `FDT` or `OVERLAY_DTB_FILE` entry.

### 10.5 Essential Tips

- **Verification:** Inspect the active device tree at `/proc/device-tree` on a running system.
- **Camera Nodes:** Camera sensor properties (clock, I2C address, reset GPIOs) are typically defined in a `<sensor>.dtsi` file included in the main platform DTS.
- **Safety:** Always have a backup configuration in `extlinux.conf` to prevent boot loops if an overlay is misconfigured.

**Further reading:** [NVIDIA Jetson Developer Docs](https://docs.nvidia.com/jetson/) (device tree, camera), [RidgeRun Developer Wiki](https://developer.ridgerun.com/) (custom overlays, extlinux).

---

## 12. Power and Thermal Management in Practice

> **Deep dive:** For power and thermal internals — PMIC architecture, nvpmodel deep dive, DVFS mechanics, INA3221 power measurement, thermal zones, fan control, battery operation, per-workload power profiling, enclosure thermal design, and production optimization — see [**Orin Nano Power Optimization & Thermal Design**](Orin-Nano-Power-Thermal/Guide.md).

### Power Modes

The Orin Nano 8GB has configurable TDP levels:

```bash
# Check current power mode
sudo nvpmodel -q

# Available modes for Orin Nano 8GB
# Mode 0: MAXN — full performance (15W TDP)
# Mode 1: 7W   — balanced (7W TDP)

# Set max performance mode
sudo nvpmodel -m 0

# Set max CPU and GPU clocks (within current power mode)
sudo jetson_clocks

# Verify
cat /sys/kernel/debug/bpmp/debug/clk/cpu0/rate   # CPU frequency
cat /sys/kernel/debug/bpmp/debug/clk/gpu/rate     # GPU frequency
```

### Real-Time Power and Temperature Monitoring

```bash
# tegrastats: comprehensive live view
sudo tegrastats --interval 100

# Example output:
# RAM 3045/7772MB (lfb 2x2MB) SWAP 0/3886MB
# CPU [35%@1510,28%@1510,12%@1510,9%@1510,15%@1510,8%@1510]
# EMC_FREQ 38% GR3D_FREQ 89%
# CPU@42C SOC0@40C SOC1@38C SOC2@38C GPU@44C tj@44C
# VDD_IN 6234mW VDD_CPU_GPU_CV 2901mW VDD_SOC 1158mW

# Parse tegrastats programmatically:
import subprocess
import re

def get_tegrastats():
    result = subprocess.run(['sudo', 'tegrastats', '--once'],
                            capture_output=True, text=True)
    line = result.stdout.strip()

    gpu_temp  = float(re.search(r'GPU@(\d+\.?\d*)C', line).group(1))
    cpu_temp  = float(re.search(r'CPU@(\d+\.?\d*)C', line).group(1))
    power_mw  = float(re.search(r'VDD_IN (\d+)mW', line).group(1))
    gpu_freq  = int(re.search(r'GR3D_FREQ (\d+)%', line).group(1))

    return {
        'gpu_temp_c': gpu_temp,
        'cpu_temp_c': cpu_temp,
        'power_mw': power_mw,
        'gpu_util_pct': gpu_freq
    }
```

### Thermal Throttling — Prevention

Thermal throttling kills performance silently. The CPU/GPU clocks drop without warning when the temperature exceeds the thermal budget.

```bash
# Check throttle temperature thresholds
cat /sys/devices/virtual/thermal/thermal_zone*/trip_point_*_temp

# Monitor for throttle events
dmesg | grep -i throttle
journalctl -f | grep -i thermal

# Practical thresholds for Orin Nano:
# CPU/GPU @ 85°C: begin throttling
# CPU/GPU @ 95°C: emergency shutdown
# Operating target: keep below 75°C
```

### Practical Cooling Solutions

```
Dev Kit enclosure: passive heatsink only → OK for 7W mode, marginal at 15W
Production fixes:
  1. Active cooling (5V fan on GPIO header)
  2. Thermal paste quality check (factory paste is mediocre)
  3. Heatsink + copper shim for better contact
  4. Enclosure with forced air flow (cut vents top and bottom)
  5. Avoid direct sunlight on enclosure
  6. Derate: run at 10W instead of 15W for reliable 24/7 operation
```

```python
# Automatic fan control via PWM (GPIO pin 33 = PWM0)
import Jetson.GPIO as GPIO
import time

GPIO.setmode(GPIO.BOARD)
GPIO.setup(33, GPIO.OUT)

fan_pwm = GPIO.PWM(33, 25000)   # 25kHz PWM frequency
fan_pwm.start(0)                 # 0% duty cycle = off

def set_fan_speed(pct):
    """pct: 0-100"""
    fan_pwm.ChangeDutyCycle(pct)

def thermal_control():
    while True:
        stats = get_tegrastats()
        temp = max(stats['gpu_temp_c'], stats['cpu_temp_c'])

        if temp < 50:
            set_fan_speed(0)    # off
        elif temp < 60:
            set_fan_speed(30)   # 30%
        elif temp < 70:
            set_fan_speed(60)   # 60%
        elif temp < 80:
            set_fan_speed(80)   # 80%
        else:
            set_fan_speed(100)  # full blast

        time.sleep(2)
```

### Power Optimization for Battery-Powered Systems

```bash
# Disable unused hardware
sudo systemctl disable bluetooth    # if not needed
sudo systemctl disable cups         # printer service, never needed

# CPU frequency governor
echo powersave | sudo tee /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor
# or for max performance:
echo performance | sudo tee /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor

# Disable HDMI output when not needed (saves ~0.5W)
sudo systemctl disable gdm3          # disable GUI
sudo systemctl set-default multi-user.target

# Measure actual power draw per component
sudo cat /sys/bus/i2c/drivers/ina3221/*/iio:device*/in_power*_input
# Shows VDD_IN, VDD_CPU_GPU_CV, VDD_SOC individually
```

### Power Budget for Mobile Robot (Example: 5Ah @ 12V = 60Wh)

```
Component               Typical Draw    Peak
Jetson Orin Nano        8W              15W
Camera (USB)            2W              2.5W
LiDAR (RPLIDAR A2)      1.5W            2W
IMU                     0.1W            0.1W
Ethernet/WiFi           0.5W            1W
Drive motors            5–30W           50W
──────────────────────────────────────────
AI compute (no motors): ~12W typical
Runtime on 5Ah@12V: 60Wh / 12W = 5 hours
```

---

## 13. OTA Update Best Practices

> **Deep dive:** For complete OTA internals — nv_update_engine, BUP generation, Tegra bootloader update chain, QSPI flash operations, payload signing, delta updates, SWUpdate/Mender/RAUC integration, fleet-scale deployment, rollback mechanisms, power-fail safety, and field failure case studies — see [**Orin Nano OTA Deep Dive**](Orin-Nano-OTA-Deep-Dive/Guide.md).
>
> For production-level rootfs architecture, A/B redundancy internals, flash XML layout, safe field update design, and bootloop debugging, see [**Orin Nano Rootfs & A/B Redundancy Deep Dive**](Orin-Nano-Rootfs-and-AB-Redundancy/Guide.md).

### OTA Update Strategies on Jetson

#### Strategy 1: apt-based updates (Simple, for application updates)

```bash
# Unattended security updates (OS packages only)
sudo apt-get install unattended-upgrades
sudo dpkg-reconfigure unattended-upgrades

# Only enable security updates, not all upgrades
# Edit /etc/apt/apt.conf.d/50unattended-upgrades:
Unattended-Upgrade::Allowed-Origins {
    "Ubuntu:${distro_codename}-security";
    "NVIDIA:${distro_codename}";   // if NVIDIA updates are needed
};

# Test update process:
sudo unattended-upgrades --dry-run --debug
```

#### Strategy 2: NVIDIA Jetson OTA (Full JetPack updates)

NVIDIA provides official OTA infrastructure for L4T updates in JetPack 6.x:

```bash
# Check available OTA updates
sudo apt-get update
apt list --upgradable 2>/dev/null | grep nvidia-l4t

# Apply L4T OTA update (minor version, e.g., 36.3 → 36.4)
sudo apt-get upgrade nvidia-l4t-core nvidia-l4t-cuda

# Reboot to activate new kernel
sudo reboot
```

#### Strategy 3: A/B Partition Scheme (Production — Zero Downtime)

The Orin supports dual-boot partition (A/B slots). This is the industry standard for reliable OTA:

```
Slot A (active):  JetPack 6.1  ← currently running
Slot B (standby): JetPack 6.2  ← being written / updated

OTA process:
  1. Download new image
  2. Write to Slot B while Slot A keeps running
  3. Verify Slot B image integrity (SHA256)
  4. Atomically switch boot to Slot B
  5. Reboot into Slot B
  6. If boot fails: automatically fall back to Slot A
  7. If boot succeeds for N minutes: mark Slot B as permanent
```

```bash
# Check current boot slot
sudo nvbootctrl dump-slots-info

# Mark current slot as successful (call from application after health check)
sudo nvbootctrl mark-boot-successful

# Manually switch slot (for testing)
sudo nvbootctrl set-active-boot-slot 1   # switch to slot B
sudo reboot

# Check rollback info
sudo nvbootctrl get-current-slot
sudo nvbootctrl get-active-boot-slot
```

#### Strategy 4: Custom OTA with Mender.io or Balena

For fleet management (10+ devices):

```bash
# Install Mender client on Jetson
# Reference: docs.mender.io/get-started/preparation/prepare-a-raspberrypi-device
# (Jetson Orin support available in Mender 3.x+)

# Mender provides:
#  - Web dashboard for fleet management
#  - Staged rollouts (deploy to 10% first, then 100%)
#  - Rollback on failure
#  - Delta updates (only send changed files, saves bandwidth)
#  - Device authentication and authorization
```

### OTA Best Practices

```
1. ALWAYS use A/B partitioning for OS updates
   Never update the running system in place

2. Verify before applying:
   - Check SHA256/GPG signature of update package
   - Verify the update is from a trusted source
   - Test on a staging device before fleet rollout

3. Staged rollout:
   Fleet of 100 devices → deploy to 5 → monitor 24h → deploy to 100
   Never push to 100% simultaneously

4. Rollback conditions (auto-trigger):
   - Boot fails to complete in X seconds
   - Application fails health check after boot
   - Critical service fails to start

5. Bandwidth management:
   - Use delta updates (only changed files)
   - Schedule updates during off-hours (low activity)
   - Implement rate limiting to not saturate network

6. Application update vs OS update:
   - Application: can update via docker pull / Python package
   - OS/drivers/kernel: requires full OTA with A/B
   - Treat these separately with different cadences
```

### Application-Level Update (Docker)

```bash
# Package your AI application in a container
# docker-compose.yml on Jetson:

version: '3'
services:
  ai_pipeline:
    image: your-registry/jetson-ai:latest
    runtime: nvidia
    environment:
      - NVIDIA_VISIBLE_DEVICES=all
    volumes:
      - /dev:/dev
      - ./models:/models
    devices:
      - /dev/video0
    restart: unless-stopped

# Update application only (no reflash needed):
docker compose pull && docker compose up -d

# Rollback application:
docker compose down
docker tag your-registry/jetson-ai:previous your-registry/jetson-ai:latest
docker compose up -d
```

---

## 14. Security Hardening

> **Deep dive:** For security architecture internals — T234 Security Engine, secure boot chain, OP-TEE trusted applications, disk encryption with hardware AES, fuse programming, secure storage (RPMB), firmware update security, runtime hardening (seccomp/AppArmor), model protection, debug lockdown, and supply chain security — see [**Orin Nano Security Deep Dive**](Orin-Nano-Security/Guide.md).

### Threat Model for Jetson Edge Devices

```
Attack Vectors:
  Physical access:     device stolen, SD card extracted, JTAG debug
  Network:             SSH brute force, unencrypted MQTT, open ports
  Supply chain:        malicious model weights, tampered update packages
  Side channel:        power analysis, timing attacks on crypto
  Application:         input injection via camera/LiDAR data, model poisoning

Assets to Protect:
  Model weights (IP)
  Sensor data (privacy)
  Control authority (safety-critical)
  Device identity/credentials
```

### Secure Boot

```bash
# Jetson Orin supports UEFI Secure Boot
# Configure during flash with SDK Manager:
# "Secure Boot" option → requires signing key pair

# Or manually:
# 1. Generate RSA-2048 key pair (on air-gapped machine, store offline)
openssl genrsa -out secure_boot_key.pem 2048
openssl req -new -x509 -key secure_boot_key.pem -out secure_boot_cert.pem -days 3650

# 2. Enroll in UEFI (via SDK Manager or L4T flash scripts)
# 3. After enrollment: only signed bootloaders will run
# 4. This prevents booting modified L4T from attacker's SD card

# Enable anti-rollback (prevent downgrade attacks):
# Set fuse JTAG_DISABLE + BOOTROM_PRODUCTION_MODE
# WARNING: IRREVERSIBLE — test thoroughly before fusing production units
```

### Disk Encryption

```bash
# Enable LUKS encryption on data partition (not rootfs)
sudo cryptsetup luksFormat /dev/nvme0n1p3          # data partition
sudo cryptsetup luksOpen /dev/nvme0n1p3 data_enc
sudo mkfs.ext4 /dev/mapper/data_enc
sudo mount /dev/mapper/data_enc /data

# Store LUKS key in TPM or secure enclave (not on disk)
# Or derive from device-unique hardware ID:
sudo tpm2_createprimary -C e -c primary.ctx
sudo tpm2_create -C primary.ctx -u key.pub -r key.priv -a "fixedtpm|fixedparent|sensitivedataorigin|userwithauth|decrypt|sign"
```

### Network Hardening

```bash
# 1. Firewall (nftables/ufw)
sudo apt-get install ufw
sudo ufw default deny incoming
sudo ufw default allow outgoing
sudo ufw allow from 192.168.1.0/24 to any port 22  # SSH: local network only
sudo ufw allow from 192.168.1.0/24 to any port 7400  # ROS2 DDS: local only
sudo ufw enable

# 2. Disable unnecessary services
sudo systemctl disable avahi-daemon     # mDNS (if not needed)
sudo systemctl disable cups             # printer
sudo systemctl disable ModemManager    # cellular (if no modem)
sudo systemctl list-units --type=service --state=active  # audit all

# 3. SSH hardening (/etc/ssh/sshd_config)
PermitRootLogin no
PasswordAuthentication no          # key-based auth only
PubkeyAuthentication yes
AllowUsers your_user               # whitelist specific user
Port 2222                          # non-standard port (minor obscurity)
MaxAuthTries 3
ClientAliveInterval 300
ClientAliveCountMax 2

# 4. Change default NVIDIA credentials immediately after flash!
# Default: user=nvidia, password=nvidia  ← ALWAYS change this
passwd  # change current user password
sudo passwd -l root  # lock root account
```

### ROS2 Security (DDS Security)

```bash
# ROS2 uses DDS (Data Distribution Service) which by default is unencrypted
# Enable ROS2 Security (SROS2):

# 1. Install security tools
sudo apt-get install ros-humble-sros2

# 2. Create key infrastructure
ros2 security create_keystore ~/ros2_keystore
ros2 security create_key ~/ros2_keystore /trt_detection_node
ros2 security create_key ~/ros2_keystore /control_node

# 3. Set permissions policy (define who can publish/subscribe to what)
# Create a permissions file defining topic access per node

# 4. Launch with security enabled
export ROS_SECURITY_KEYSTORE=~/ros2_keystore
export ROS_SECURITY_ENABLE=true
export ROS_SECURITY_STRATEGY=Enforce

ros2 run your_package trt_detection_node
```

### Model Weight Protection

```python
# Encrypt TensorRT engine weights at rest
from cryptography.fernet import Fernet

# Generate key (store in hardware security module or TPM in production)
key = Fernet.generate_key()

def encrypt_engine(engine_path, encrypted_path, key):
    f = Fernet(key)
    with open(engine_path, 'rb') as fp:
        data = fp.read()
    with open(encrypted_path, 'wb') as fp:
        fp.write(f.encrypt(data))

def load_encrypted_engine(encrypted_path, key):
    f = Fernet(key)
    with open(encrypted_path, 'rb') as fp:
        data = f.decrypt(fp.read())
    runtime = trt.Runtime(trt.Logger(trt.Logger.WARNING))
    return runtime.deserialize_cuda_engine(data)
    # Engine is only in memory, never decrypted to disk
```

### Security Audit Checklist

```
Before deployment, verify:
□ Default passwords changed
□ SSH key-based auth only, no password auth
□ Firewall enabled with minimal open ports
□ Secure boot enabled and verified
□ Anti-rollback fuses set (if production)
□ Data partition encrypted (if sensitive data)
□ All services audited, unused ones disabled
□ Kernel patched to latest L4T version
□ ROS2 DDS security enabled (if network-connected)
□ Model weights encrypted at rest
□ OTA update channel uses TLS + signature verification
□ Physical ports secured (disable USB in production if not needed)
□ Log aggregation set up (centralized syslog for anomaly detection)
```

---

## 15. Projects

### Project 1: Fresh JetPack 6 Install + NVMe Boot
Flash Orin Nano 8GB to JetPack 6.x with SDK Manager, configure NVMe boot, and benchmark I/O speed vs SD card.

### Project 2: YOLO on Jetson with TensorRT
Train YOLOv8 on custom dataset, export to ONNX, convert to TensorRT FP16 engine, achieve >25 FPS on Orin Nano.

### Project 3: Camera + LiDAR Fusion Pipeline
Build a ROS2 node that fuses camera detection boxes with LiDAR point cloud depth, publish 3D bounding boxes.

### Project 4: Power vs Performance Profiling
Using `tegrastats`, plot FPS vs power consumption for: FP32 / FP16 / INT8 / DLA modes. Find the optimal operating point.

### Project 5: Thermal Stress Test + Cooling Comparison
Run inference at 100% load for 30 minutes. Compare: bare heatsink vs active cooling. Measure FPS drop due to throttling.

### Project 6: A/B OTA Update
Implement a simple OTA update service that fetches a new Docker image, tests it in a container, then switches production traffic to it with rollback capability.

### Project 7: Security Hardening
Start from a fresh Jetson install. Apply all items in the Security Audit Checklist. Verify each item. Run `nmap` from another machine to confirm attack surface.

### Project 8: End-to-end product integration (self-directed capstone)
Tie together **Track B** modules **2–7**: a Jetson Orin Nano–class device (dev kit or your custom carrier), **L4T** image, **application** stack (networking, optional GUI, inference), **security / OTA**, and a **compliance / manufacturing** checklist suitable for a pilot build. Optional: use [OpenClaw](https://github.com/openclaw/openclaw) or another orchestrator for voice, automation, or browser tooling — scope and document your own acceptance tests.

---

<a id="16-jetson-containers--cloud-native-ml-deployment"></a>
## 16. Jetson Containers — Cloud-Native ML Deployment

> **Deep dive:** For container and fleet deployment — nvidia-container-runtime, L4T base images, jetson-containers project, cross-compilation, GPU/camera access in containers, Docker Compose, K3s on Jetson, fleet management (Balena/AWS IoT/Azure IoT Edge), CI/CD pipelines, monitoring, and container security — see [**Orin Nano Container & Fleet Deployment**](Orin-Nano-Container-Fleet/Guide.md).

[jetson-containers](https://github.com/dusty-nv/jetson-containers) is a modular container build system that provides the latest AI/ML packages for NVIDIA Jetson and JetPack-L4T. It enables **cloud-native deployment** on edge devices: reproducible environments, version-pinned dependencies, and OTA-friendly updates via `docker pull`.

### Why Containers on Jetson?

| Bare-metal install | Containerized (jetson-containers) |
|--------------------|------------------------------------|
| `pip install` conflicts, broken venvs | Isolated per-container dependencies |
| JetPack upgrade breaks PyTorch | Pin L4T/CUDA in image tag |
| Manual CUDA/cuDNN/TensorRT matching | Pre-built wheels for your JetPack |
| Hard to reproduce across devices | Same image → same behavior |
| OTA = full reflash or risky apt | OTA = `docker compose pull && up -d` |

### Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────────┐
│  jetson-containers (Python CLI + package definitions)                   │
│  - autotag: finds compatible image for your JetPack (r36.x, cu12.x)     │
│  - build: composes packages (pytorch + transformers + ros)             │
│  - run: docker run with --runtime nvidia, /data mount, devices          │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│  Package Stack (modular)                                                │
│  ML: pytorch, tensorflow, onnxruntime, deepstream, jax                   │
│  LLM: ollama, vllm, sglang, llama.cpp, transformers, exllama            │
│  VLM: llava, vila, nanoowl, nanosam                                     │
│  Robotics: ros:humble-desktop, lerobot, openvla, zed                    │
│  Speech: whisper, whisper_trt, faster-whisper, piper                    │
│  RAG: llama-index, langchain, nanodb, faiss                             │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│  Base: L4T (Ubuntu 22.04/24.04) + CUDA + cuDNN + TensorRT               │
│  Registry: dustynv/* on Docker Hub, pypi.jetson-ai-lab.io for wheels     │
└─────────────────────────────────────────────────────────────────────────┘
```

### Installation and Quick Start

```bash
# On Jetson (after JetPack 6.2 or 7.x installed)
git clone https://github.com/dusty-nv/jetson-containers
bash jetson-containers/install.sh

# Pull and run a compatible PyTorch container (no build needed)
jetson-containers run $(autotag l4t-pytorch)

# Or run a specific pre-built image
sudo docker run --runtime nvidia -it --rm --network=host dustynv/l4t-pytorch:r36.2.0
```

### Key Concepts

**`autotag`** — Resolves the correct image tag for your JetPack/L4T version. If no pre-built image exists, it can trigger a build. Example: `autotag l4t-pytorch` → `dustynv/l4t-pytorch:r36.4.0` on JetPack 6.2.

**`jetson-containers run`** — Wraps `docker run` with:
- `--runtime nvidia` (GPU access)
- `-v /data:/data` (persistent cache for models)
- `--network=host` (for ROS2, streaming)
- Device passthrough for cameras, serial ports

**Package composition** — Combine packages into a single image:

```bash
# Build custom image: PyTorch + Transformers + ROS2 Humble
jetson-containers build --name=my_ai_robot pytorch transformers ros:humble-desktop

# Run it
jetson-containers run my_ai_robot
```

### Version and CUDA Customization

```bash
# Rebuild for specific CUDA version
CUDA_VERSION=12.6 jetson-containers build transformers

# Ubuntu 24.04 base (JetPack 6/7)
LSB_RELEASE=24.04 jetson-containers build pytorch:2.8

# Request specific PyTorch / cuDNN / TensorRT via env vars (see docs/build.md)
```

### Supported JetPack Versions

- **JetPack 6.2** (CUDA 12.6, L4T r36.5) — primary target
- **JetPack 7** (CUDA 13.x) — supported
- **Ubuntu 24.04** — available for newer stacks

### Integration with OTA (Section 12)

jetson-containers fits directly into the Docker-based OTA flow:

```bash
# Your docker-compose.yml can use dustynv images
services:
  inference:
    image: dustynv/l4t-pytorch:r36.4.0
    # or your custom-built image from a private registry
    runtime: nvidia
    volumes:
      - /data:/data
      - ./models:/models

# OTA update: pull new image, restart
docker compose pull && docker compose up -d
```

### Practical Use Cases

| Use case | Package(s) | Command |
|----------|------------|---------|
| PyTorch inference | `l4t-pytorch` | `jetson-containers run $(autotag l4t-pytorch)` |
| Local LLM (Ollama) | `ollama` | `jetson-containers run $(autotag ollama)` |
| ROS2 + PyTorch | `pytorch`, `ros:humble-desktop` | Build combined image |
| Whisper speech-to-text | `whisper` or `whisper_trt` | `jetson-containers run $(autotag whisper)` |
| Object detection (ViT) | `nanoowl` | `jetson-containers run $(autotag nanoowl)` |
| RAG with vector DB | `nanodb`, `llama-index` | Build or run `nanodb` |

### Documentation and Community

- **Package list**: [github.com/dusty-nv/jetson-containers/packages](https://github.com/dusty-nv/jetson-containers/tree/master/packages)
- **System setup**: Docker daemon config, memory/storage tuning
- **Jetson AI Lab**: [jetson-ai-lab.com](https://www.jetson-ai-lab.com) — tutorials, SLM/VLM demos

---

## 17. Resources

### Official Documentation
- **Jetson Linux Developer Guide** (L4T r36.4): [docs.nvidia.com/jetson/archives/r36.4/DeveloperGuide](https://docs.nvidia.com/jetson/archives/r36.4/DeveloperGuide/index.html) — primary reference: boot, kernel, device tree, pinmux, camera, flashing, OTA, security
- **JetPack SDK**: developer.nvidia.com/embedded/jetpack
- **NVIDIA Device Tree / Camera**: docs.nvidia.com/jetson — camera sensor DTSI, tegra-camera-platform
- **NVIDIA SDK Manager**: developer.nvidia.com/sdk-manager
- **L4T Developer Guide** (archives): docs.nvidia.com/jetson/archives/
- **TensorRT Developer Guide**: docs.nvidia.com/deeplearning/tensorrt/developer-guide/
- **VPI Documentation**: docs.nvidia.com/vpi/

### Community and Tools
- **JetsonHacks** (https://jetsonhacks.com/): Practical Jetson tutorials, GPIO pinouts (Orin Nano, AGX Orin, Xavier, Nano), JetPack updates, hardware setup — essential reference for Jetson mastering
- **NVIDIA NGC**: ngc.nvidia.com — pre-built containers optimized for Jetson
- **jetson-containers** (https://github.com/dusty-nv/jetson-containers): See [Section 16](#16-jetson-containers--cloud-native-ml-deployment) for deep dive. Quick start: `jetson-containers run $(autotag l4t-pytorch)`.
- **DeepStream Getting Started**: docs.nvidia.com/metropolis/deepstream/

### Performance and Profiling
- `tegrastats` — always running in a side terminal during development
- Nsight Systems — system-level GPU+CPU trace
- `trtexec --verbose` — TensorRT engine build + benchmark

### Security
- **NVIDIA Jetson Security Guide**: docs.nvidia.com/jetson/archives/l4t-archived/ (Jetson Security section)
- **SROS2 Wiki**: wiki.ros.org/sros2

---

*Next: [2. Custom Carrier Board](../2.%20Custom%20Carrier%20Board%20Design%20and%20Bring-Up/Guide.md) → [3. L4T Customization](../3.%20L4T%20Customization/Guide.md) → [4. FSP Customization](../4.%20FSP%20%28Firmware%20Support%20Package%29%20Customization/Guide.md) → [5. Application Development](../5.%20Application%20Development/Guide.md) → [6. Security and OTA](../6.%20Security%20and%20OTA/Guide.md) → [7. Compliance and Manufacturing](../7.%20Compliance%20and%20Manufacturing/Guide.md)*
