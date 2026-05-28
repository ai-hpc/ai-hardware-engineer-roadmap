# Orin Nano 8GB — Kernel Internals & Customization

<div class="course-identity auto-course" style="--course-accent: #ea580c; --course-accent-rgb: 234, 88, 12;" markdown="1">
<div class="course-identity__icon">ON8K</div>
<div markdown="1">
<p class="course-identity__eyebrow">Deep Dive · Jetson Track</p>
<p class="course-identity__title">Specialized course identity for Orin Nano 8GB — Kernel Internals & Customization.</p>
<p class="course-identity__meta">Artifact: Jetson integration demo · Measure: latency, memory, power, logs</p>
</div>
</div>


> **Scope:** Production-level understanding of the Jetson Linux (L4T) kernel on Orin Nano 8GB — from source tree structure and build system through device tree architecture, driver model, kernel configuration, boot time optimization, custom module development, and production kernel hardening.
>
> **Prerequisites:** Familiarity with the [Orin Nano boot chain](../Guide.md#1-orin-nano-8gb--hardware--boot-chain-internals), [memory architecture](../Orin-Nano-Memory-Architecture/Guide.md), and basic Linux kernel concepts (processes, syscalls, modules).

---


## 1. Jetson Kernel Overview

The Jetson Linux (L4T) kernel is **not mainline Linux**. It is:

* Based on Linux 5.10 (JetPack 5.x) or Linux 5.15 (JetPack 6.x)
* Patched by NVIDIA with Tegra-specific drivers, device tree bindings, and subsystem modifications
* Includes out-of-tree NVIDIA modules (nvgpu, camera stack, multimedia)
* Compiled for `aarch64` (ARM 64-bit)

### What NVIDIA Changes From Mainline

| Area              | Mainline Linux                    | Jetson L4T Kernel                          |
|-------------------|-----------------------------------|--------------------------------------------|
| GPU driver        | nouveau (open-source, limited)    | nvgpu (NVIDIA proprietary, full features)  |
| Camera            | Standard V4L2                     | V4L2 + NVIDIA VI/ISP/CSI extensions        |
| Power management  | Generic DVFS                      | BPMP-based, nvpmodel integration           |
| Display           | DRM/KMS                           | DRM/KMS + NVIDIA display controller        |
| Device tree       | Mainline ARM DT bindings          | NVIDIA-specific bindings + overlays        |
| Thermal           | Generic thermal framework         | Tegra-specific thermal zones + governors   |

You **cannot** use a mainline kernel on Jetson — NVIDIA drivers and firmware require the L4T kernel.

---

## 2. Kernel Source Tree Structure

After downloading the L4T kernel sources:

```
Linux_for_Tegra/source/
├── kernel/
│   └── kernel-5.10/              ← Main kernel source
│       ├── arch/arm64/           ← ARM64 architecture code
│       │   ├── boot/dts/         ← Mainline device trees
│       │   └── configs/          ← defconfig files
│       ├── drivers/              ← All kernel drivers
│       │   ├── gpu/nvgpu/        ← NVIDIA GPU driver
│       │   ├── media/            ← V4L2 + NVIDIA camera
│       │   ├── platform/tegra/   ← Tegra platform drivers
│       │   └── ...
│       ├── include/              ← Kernel headers
│       ├── net/                  ← Networking stack
│       └── Makefile
├── hardware/
│   └── nvidia/
│       ├── platform/t23x/        ← T234 platform device trees
│       │   └── concord/          ← Orin module DTS files
│       └── soc/t23x/             ← T234 SoC-level DTS
└── nvidia-oot/                   ← Out-of-tree NVIDIA modules
    ├── drivers/
    │   ├── video/tegra/host/     ← nvhost (multimedia)
    │   ├── gpu/nvgpu/            ← nvgpu module
    │   └── media/platform/tegra/ ← Camera drivers
    └── Makefile
```

### Key Directories

| Path                              | Content                                    |
|-----------------------------------|--------------------------------------------|
| `kernel/kernel-5.10/`             | Base Linux kernel                          |
| `hardware/nvidia/platform/t23x/`  | T234 device tree source files              |
| `hardware/nvidia/soc/t23x/`       | SoC-level device tree includes             |
| `nvidia-oot/`                      | NVIDIA out-of-tree modules (nvgpu, camera) |

---

## 3. Building the Kernel From Source

### Prerequisites

```bash
# Install cross-compilation toolchain
sudo apt install gcc-aarch64-linux-gnu build-essential bc flex bison libssl-dev

# Set environment variables
export CROSS_COMPILE=aarch64-linux-gnu-
export ARCH=arm64
export LOCALVERSION=-tegra
```

### Build Steps

```bash
cd Linux_for_Tegra/source/kernel/kernel-5.10/

# Step 1: Configure kernel (use Jetson defconfig)
make tegra_defconfig

# Step 2: (Optional) Customize configuration
make menuconfig

# Step 3: Build kernel Image
make -j$(nproc) Image

# Step 4: Build device tree blobs
make -j$(nproc) dtbs

# Step 5: Build modules
make -j$(nproc) modules

# Step 6: Install modules to staging directory
make modules_install INSTALL_MOD_PATH=<staging_dir>
```

### Build NVIDIA Out-of-Tree Modules

```bash
cd Linux_for_Tegra/source/nvidia-oot/

# Build against the kernel you just compiled
make -j$(nproc) \
    KERNEL_SRC=../kernel/kernel-5.10 \
    M=$(pwd) \
    modules
```

### Deploy to Jetson

```bash
# Copy kernel Image
cp arch/arm64/boot/Image <Linux_for_Tegra>/kernel/Image

# Copy DTB
cp arch/arm64/boot/dts/nvidia/*.dtb <Linux_for_Tegra>/kernel/dtb/

# Copy modules
cp -r <staging_dir>/lib/modules/* <Linux_for_Tegra>/rootfs/lib/modules/

# Flash (or copy to device via SCP for development)
sudo ./flash.sh jetson-orin-nano-devkit internal
```

For iterative development, copy Image and modules directly to the Jetson via SCP instead of reflashing:

```bash
scp arch/arm64/boot/Image jetson:/boot/Image
scp -r <staging_dir>/lib/modules/<version> jetson:/lib/modules/
ssh jetson "sudo reboot"
```

---

## 4. Kernel Configuration (defconfig)

### Default Configuration

The Jetson default config enables everything — all supported hardware, all filesystems, all debugging. This provides maximum compatibility but increases boot time and kernel size.

```bash
# View current config on running Jetson
zcat /proc/config.gz | grep CONFIG_NVGPU
# CONFIG_NVGPU=m

# Or check the defconfig file
cat arch/arm64/configs/tegra_defconfig
```

### Configuration Categories

| Category           | Default State    | Production Recommendation              |
|--------------------|------------------|----------------------------------------|
| All NVIDIA drivers | Enabled          | Keep enabled (required for GPU/camera) |
| Filesystems        | Many built-in    | Modularize unused (NTFS, FUSE, VFAT)  |
| Audio codecs       | Enabled          | Disable if no audio needed             |
| USB gadget         | Enabled          | Disable if not used                    |
| Debug/trace        | Enabled          | Disable (FTRACE, KMEMLEAK, etc.)       |
| Network protocols  | Many enabled     | Keep only needed (TCP/IP, CAN if used) |
| HID drivers        | Enabled          | Modularize (not needed at boot)        |

### Creating a Custom defconfig

```bash
# Start from Jetson default
make tegra_defconfig

# Customize
make menuconfig

# Save as custom defconfig
make savedefconfig
cp defconfig arch/arm64/configs/my_product_defconfig

# Future builds use your config
make my_product_defconfig
```

### Critical Configs for AI Edge Systems

```
# Must be enabled
CONFIG_NVGPU=m                    # GPU driver
CONFIG_VIDEO_TEGRA=m              # Camera pipeline
CONFIG_TEGRA_BPMP=y               # Power management
CONFIG_ARM_SMMU=y                 # IOMMU (required for DMA)
CONFIG_CMA=y                      # Contiguous memory allocation
CONFIG_DMA_CMA=y                  # CMA for DMA
CONFIG_CMA_SIZE_MBYTES=768        # CMA size (adjust per workload)
CONFIG_IOMMU_SUPPORT=y            # SMMU support
CONFIG_VFIO=n                     # Usually not needed on edge

# Should be modularized for boot speed
CONFIG_FUSE_FS=m
CONFIG_VFAT_FS=m
CONFIG_NTFS_FS=m
CONFIG_USB_HID=m
CONFIG_SND_SOC_TEGRA_ALT=n        # Disable if no audio

# Should be disabled in production
# CONFIG_FTRACE is not set
# CONFIG_KMEMLEAK is not set
# CONFIG_DEBUG_INFO is not set
# CONFIG_DYNAMIC_DEBUG is not set
```

---

## 5. Device Tree Architecture on T234

The device tree (DT) describes the hardware to the kernel. On Jetson, it is a multi-layered system.

### DTS File Hierarchy

```
tegra234-p3767-0000-p3768-0000-a0.dts     ← Top-level (what gets compiled)
 └── includes:
     ├── tegra234-p3767-0000.dtsi          ← Orin Nano module
     │   └── tegra234-soc.dtsi             ← T234 SoC peripherals
     │       └── tegra234-soc-base.dtsi    ← Base SoC definitions
     ├── tegra234-p3768-0000.dtsi          ← Dev Kit carrier board
     └── tegra234-power-tree.dtsi          ← Power domains
```

### Key Device Tree Directories

```
hardware/nvidia/platform/t23x/concord/     ← Orin module + carrier board DTS
hardware/nvidia/soc/t23x/                   ← SoC-level device tree includes
```

The DTB that gets flashed:

```
hardware/nvidia/platform/t23x/concord/kernel-dts/tegra234-p3767-0000-p3768-0000-a0.dts
```

### Device Tree Node Anatomy

Every hardware block on the SoC has a device tree node:

```dts
/* Example: VI (Video Input) controller */
vi@15c10000 {
    compatible = "nvidia,tegra234-vi";
    reg = <0x0 0x15c10000 0x0 0x10000>;     /* MMIO registers */
    interrupts = <GIC_SPI 200 IRQ_TYPE_LEVEL_HIGH>;
    clocks = <&bpmp_clks TEGRA234_CLK_VI>;
    clock-names = "vi";
    resets = <&bpmp_resets TEGRA234_RESET_VI>;
    reset-names = "vi";
    power-domains = <&bpmp TEGRA234_POWER_DOMAIN_VE>;
    iommus = <&smmu TEGRA234_SID_VI>;       /* SMMU stream ID */
    status = "okay";                         /* "okay" = enabled */
};
```

### Disabling Unused Hardware

To disable a peripheral you are not using (reduces boot time and power):

```dts
/* In an overlay or board-level DTS */
&spi0 {
    status = "disabled";    /* Kernel skips probing this device */
};

&sound {
    status = "disabled";    /* No audio codec initialization */
};
```

Every disabled node saves probe time during boot.

---

## 6. Device Tree Overlays

Overlays modify the base device tree without editing the original DTS files. This is how you add sensors, change pin muxing, or configure peripherals for your product.

### Overlay File Structure

```dts
/* my-camera-overlay.dts */
/dts-v1/;
/plugin/;

/ {
    overlay-name = "My Camera Overlay";
    compatible = "nvidia,p3768-0000+p3767-0000";

    fragment@0 {
        target = <&vi>;
        __overlay__ {
            num-channels = <1>;
            /* camera channel configuration */
        };
    };

    fragment@1 {
        target = <&i2c2>;
        __overlay__ {
            imx219@10 {
                compatible = "sony,imx219";
                reg = <0x10>;
                /* sensor properties */
            };
        };
    };
};
```

### Compiling and Applying Overlays

```bash
# Compile overlay
dtc -@ -I dts -O dtb -o my-camera-overlay.dtbo my-camera-overlay.dts

# Copy to Jetson
scp my-camera-overlay.dtbo jetson:/boot/

# Apply via extlinux.conf
# Add to APPEND line:
# FDT /boot/tegra234-p3767-0000-p3768-0000-a0.dtb
# FDTOVERLAYS /boot/my-camera-overlay.dtbo
```

### Jetson-IO Tool

NVIDIA provides `jetson-io.py` for common overlay tasks:

```bash
sudo /opt/nvidia/jetson-io/jetson-io.py
```

This GUI/CLI tool configures:

* Pin muxing (GPIO, SPI, I2C, UART)
* CSI camera lanes
* Fan control
* SPI flash

---

## 7. NVIDIA Driver Model — How Jetson Drivers Work

Jetson drivers differ from standard Linux drivers because many interact with dedicated firmware processors.

### Driver Categories

| Category         | Examples                    | How They Work                              |
|------------------|-----------------------------|--------------------------------------------|
| **Platform**     | VI, ISP, NVCSI              | Direct MMIO register access, DMA           |
| **BPMP-backed**  | Clocks, resets, power gates | Sends IPC messages to BPMP firmware        |
| **Firmware IPC** | nvgpu, camera RTC           | Communicates with firmware via mailbox     |
| **Standard**     | USB, Ethernet, I2C          | Standard Linux driver model                |

### BPMP Driver Communication

Many hardware controls on T234 go through BPMP (Boot and Power Management Processor):

```
Linux driver
   ↓ (IPC message via tegra-bpmp driver)
BPMP firmware
   ↓ (controls hardware registers)
Actual hardware (clocks, resets, power domains)
```

Example — setting a clock rate:

```c
/* In kernel driver code */
clk = devm_clk_get(&pdev->dev, "vi");
clk_set_rate(clk, 400000000);  /* 400 MHz */

/* This doesn't directly program clock registers.
   Instead, the clock framework sends an IPC message to BPMP,
   which programs the actual PLL/divider registers. */
```

This means clock and power changes require BPMP firmware to be running and responsive. If BPMP hangs, all clock/power operations stall.

### Module Loading Order

On Jetson boot, modules load in dependency order:

```
tegra-bpmp.ko          ← First: BPMP IPC (needed by everything)
 ↓
tegra-fuse.ko          ← Fuse reading (chip identification)
 ↓
arm-smmu.ko            ← IOMMU (needed before any DMA device)
 ↓
nvgpu.ko               ← GPU driver
 ↓
tegra-vi.ko            ← Video Input
tegra-isp.ko           ← Image Signal Processor
nvcsi.ko               ← CSI controller
 ↓
(camera sensor drivers) ← I2C sensor drivers (imx219, etc.)
```

If a module fails to load, all dependent modules also fail. This is why kernel/module version mismatch causes cascading failures.

---

## 8. Key Kernel Subsystems on Orin Nano

### Memory Management

See [Memory Architecture Deep Dive](../Orin-Nano-Memory-Architecture/Guide.md) for full details. Kernel-relevant highlights:

* **CMA** — configured via kernel command line or DTB; critical for camera and DMA
* **SMMU** — ARM SMMU v2 driver (`arm-smmu.ko`); maps IOVA for all DMA devices
* **DMA-BUF** — kernel buffer sharing framework; enables zero-copy between engines

### Interrupt Handling

T234 uses ARM GICv3 (Generic Interrupt Controller):

```bash
# View interrupt distribution across CPUs
cat /proc/interrupts

# Key interrupts on Orin Nano
# nvgpu           — GPU interrupts
# tegra-vi        — camera frame complete
# host1x          — multimedia engine sync
# tegra-pmc       — power management controller
```

For real-time workloads, you can pin specific interrupts to specific CPU cores using `irqbalance` or manual `/proc/irq/<N>/smp_affinity`.

### Scheduling

Default scheduler is CFS (Completely Fair Scheduler). For latency-sensitive inference:

```bash
# Set real-time priority for inference process
sudo chrt -f 50 ./my_inference_app

# Or use SCHED_DEADLINE for guaranteed timing
# (requires kernel CONFIG_SCHED_DEADLINE=y)
```

### Filesystem

Default rootfs filesystem is ext4. For production:

* **ext4** — reliable, well-tested, supports journaling
* **squashfs** — read-only compressed rootfs (saves storage, fast mount)
* **overlayfs** — writable layer on top of read-only squashfs

---

## 9. Camera Kernel Stack (V4L2 + NVIDIA)

The camera subsystem on Jetson is one of the most complex kernel stacks.

### Architecture

```
Userspace:
  libargus / nvargus-daemon
  GStreamer (nvv4l2camerasrc)
  V4L2 ioctl interface
       ↓
Kernel:
  ┌─────────────────────────────────────────┐
  │ V4L2 subsystem (media controller)       │
  │   ↓                                     │
  │ tegra-video (nvidia,tegra234-vi)         │
  │   ↓                                     │
  │ tegra-isp (nvidia,tegra234-isp)          │
  │   ↓                                     │
  │ nvcsi (nvidia,tegra234-nvcsi)            │
  │   ↓                                     │
  │ I2C sensor driver (e.g., imx219)         │
  └─────────────────────────────────────────┘
       ↓
Hardware:
  Sensor → CSI lanes → NVCSI → VI → ISP → DRAM (via SMMU)
```

### Media Controller Graph

The Linux media controller exposes the pipeline as a graph:

```bash
# View media graph
media-ctl -p -d /dev/media0

# Output shows connected entities:
# entity: nvcsi-0 → vi-output-0 → tegra-isp
```

### Sensor Driver Anatomy

Every camera sensor needs a kernel driver:

```c
static const struct of_device_id imx219_of_match[] = {
    { .compatible = "sony,imx219" },
    { },
};

static struct i2c_driver imx219_driver = {
    .driver = {
        .name = "imx219",
        .of_match_table = imx219_of_match,
    },
    .probe = imx219_probe,
    .remove = imx219_remove,
};
```

The probe function:

1. Reads sensor ID via I2C
2. Registers V4L2 subdevice
3. Configures format, resolution, frame rate
4. Links to NVCSI in the media graph

### Adding a New Camera Sensor

1. Write or port the sensor driver (I2C, V4L2 subdev)
2. Add device tree node under the I2C bus with correct address
3. Add device tree node for CSI channel configuration
4. Create device tree overlay linking sensor → NVCSI → VI
5. Build and install the driver module
6. Verify with `v4l2-ctl --list-devices`

---

## 10. GPU Kernel Driver (nvgpu)

### Architecture

The nvgpu driver is NVIDIA's Jetson GPU kernel module:

```
CUDA runtime (userspace)
   ↓ (ioctl)
/dev/nvhost-gpu
   ↓
nvgpu.ko (kernel)
   ↓
GPU hardware (Ampere cores, memory controller, tensor cores)
   ↓
SMMU (memory isolation)
   ↓
DRAM
```

### Key nvgpu Responsibilities

* GPU power management (power gating, clock scaling)
* Channel management (multiple CUDA contexts)
* Memory management (GPU page tables, buffer mapping)
* Fence/sync point management (synchronization between engines)
* Firmware loading (GPU microcode)

### GPU Firmware

The GPU requires firmware loaded during driver probe:

```
/lib/firmware/nvidia/gv11b/
├── acr_ucode.bin          ← ACR (Advanced Code Region) loader
├── fecs.bin               ← Front End Context Switching
├── gpccs.bin              ← GPC Context Switching
└── ...
```

If firmware files are missing or corrupted:

```
nvgpu: firmware "nvidia/gv11b/acr_ucode.bin" not found
```

GPU will fail to initialize — no CUDA, no inference, no display acceleration.

### nvgpu Module Parameters

```bash
# View current parameters
cat /sys/module/nvgpu/parameters/*

# Key parameters:
# enable_elcg   — Enable clock gating (saves power)
# enable_elpg   — Enable power gating (deeper sleep)
# gpu_freq      — Current GPU frequency
```

---

## 11. Power and Clock Management in Kernel

### nvpmodel — Power Mode Framework

nvpmodel is a userspace tool that configures the kernel's power and clock settings:

```bash
# View current power mode
sudo nvpmodel -q

# Set to maximum performance (15W on Orin Nano)
sudo nvpmodel -m 0

# Set to power-saving mode (7W)
sudo nvpmodel -m 1
```

Internally, nvpmodel writes to:

* `/sys/devices/system/cpu/cpu*/cpufreq/scaling_max_freq` — CPU max frequency
* `/sys/devices/gpu.0/devfreq/*/max_freq` — GPU max frequency
* `/sys/kernel/nvpmodel_emc_cap/emc_iso_cap` — memory bandwidth cap

### Clock Tree

Clocks on T234 are managed by BPMP firmware. The kernel clock framework sends requests via IPC:

```bash
# View clock tree
cat /sys/kernel/debug/clk/clk_summary

# Key clocks:
# vi_clk      — Video Input (camera capture rate)
# isp_clk     — ISP processing rate
# nvdec_clk   — Video decoder
# gpu_clk     — GPU core clock
# emc_clk     — Memory controller clock (DRAM bandwidth)
```

### Dynamic Voltage and Frequency Scaling (DVFS)

The kernel adjusts GPU and CPU frequencies based on load:

```bash
# GPU DVFS (devfreq governor)
cat /sys/devices/gpu.0/devfreq/*/governor
# "nvhost_podgov" — NVIDIA's load-based governor

# CPU DVFS (cpufreq governor)
cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor
# "schedutil" — scheduler-integrated frequency scaling
```

### jetson_clocks

Locks all clocks to maximum (bypasses DVFS):

```bash
sudo jetson_clocks

# Useful for benchmarking (deterministic performance)
# NOT recommended for production (high power, high temperature)
```

---

## 12. Kernel Boot Sequence on Orin Nano

After UEFI loads the kernel Image:

### Early Boot (Architecture-Specific)

```
head.S (arch/arm64/kernel/head.S)
 ↓
__primary_switched
 ↓
start_kernel()          ← First C function
 ↓
setup_arch()            ← ARM64 architecture setup
  ├── parse DTB
  ├── setup page tables + MMU
  ├── configure memory zones
  └── reserve CMA
 ↓
mm_init()               ← Memory management init
  ├── buddy allocator
  ├── slab allocator (kmalloc)
  └── CMA init
```

### Driver Initialization

```
 ↓
driver_init()
  ├── buses_init()      ← Register bus types (platform, I2C, SPI, PCI)
  ├── devices_init()    ← Create /sys/devices
  └── platform_bus_init()
 ↓
do_initcalls()          ← Execute all __init functions (level 0–7)
  Level 0: pure_initcall    ← Very early (irqchip, clocksource)
  Level 1: core_initcall     ← Core subsystems
  Level 2: postcore_initcall ← Bus drivers
  Level 3: arch_initcall     ← Architecture-specific
  Level 4: subsys_initcall   ← Subsystem init (SMMU, DMA)
  Level 5: fs_initcall       ← Filesystem init
  Level 6: device_initcall   ← Device drivers (nvgpu, tegra-vi)
  Level 7: late_initcall     ← Late initialization
```

### Post-Init

```
 ↓
prepare_namespace()     ← Find and mount rootfs
  ├── initramfs or direct mount
  └── root= from kernel command line
 ↓
kernel_init()
 ↓
run_init_process("/sbin/init")  ← Hand off to systemd
```

### Measuring Boot Time

```bash
# Kernel boot timeline
dmesg | head -50
# [    0.000000] Booting Linux on physical CPU 0x0000000000
# [    0.123456] ... each line shows timestamp

# systemd-analyze for full boot breakdown
systemd-analyze
systemd-analyze blame    # Show slowest services
systemd-analyze critical-chain   # Show critical path
```

---

## 13. Kernel Boot Time Optimization

The default L4T kernel is configured for maximum compatibility, not minimum boot time. Production systems should optimize.

### Optimization Categories

#### 1. Disable Unused Device Tree Nodes

Every enabled device tree node triggers driver probing. Disable what you do not use:

```dts
/* Disable SPI if not used */
&spi0 { status = "disabled"; };
&spi1 { status = "disabled"; };

/* Disable audio if not used */
&sound { status = "disabled"; };
&tegra_sound { status = "disabled"; };

/* Disable USB if not used */
&xusb { status = "disabled"; };
```

Device tree directories:

```
<top>/hardware/nvidia/platform/t23x/
<top>/hardware/nvidia/soc/t23x/
```

#### 2. Disable Console Printing Over UART

Console printing is a major boot time bottleneck. Each `printk` waits for UART transmission.

For Orin series, edit the platform configuration file and remove:

```
console=ttyTCU0,115200
```

You can still review logs via framebuffer console or `dmesg` after boot.

Alternatively, reduce console verbosity:

```bash
# In kernel command line (extlinux.conf)
APPEND ... loglevel=1 quiet
```

`loglevel=1` shows only KERN_EMERG. `quiet` suppresses most boot messages.

#### 3. Modularize Non-Essential Drivers

Move drivers not needed at boot time from built-in (`=y`) to module (`=m`):

```
# Filesystems (loaded on demand)
CONFIG_FUSE_FS=m
CONFIG_VFAT_FS=m
CONFIG_NTFS_FS=m

# HID (USB keyboard/mouse — not needed at boot for headless)
CONFIG_USB_HID=m
CONFIG_HID_GENERIC=m

# Network drivers not used at boot
CONFIG_NET_VENDOR_INTEL=m
CONFIG_NET_VENDOR_REALTEK=m

# QSPI (not needed after boot)
CONFIG_SPI_TEGRA210_QUAD=m
```

This reduces kernel Image size and defers initialization.

#### 4. Use Asynchronous Probe

Drivers can probe asynchronously instead of blocking the boot sequence:

```c
static struct platform_driver my_driver = {
    .driver = {
        .name = "my-driver",
        .of_match_table = my_of_match,
        .probe_type = PROBE_PREFER_ASYNCHRONOUS,  /* Non-blocking probe */
    },
    .probe = my_probe,
    .remove = my_remove,
};
```

NVIDIA already uses this for some drivers. You can add it to custom drivers or patch additional Jetson drivers.

#### 5. Disable Audio Configurations

If your product has no audio (common for vision-only AI edge devices):

```
# CONFIG_SND_SOC_TEGRA_ALT is not set
# CONFIG_SND_SOC_TEGRA_ALT_FORCE_CARD_REG is not set
# CONFIG_SND_SOC_TEGRA_T186REF_ALT is not set
# CONFIG_SND_SOC_TEGRA_T186REF_MOBILE_ALT is not set
```

Audio codec initialization adds hundreds of milliseconds.

#### 6. Disable Kernel Debugging

Production kernels should not include debug infrastructure:

```
# CONFIG_FTRACE is not set
# CONFIG_FUNCTION_TRACER is not set
# CONFIG_KMEMLEAK is not set
# CONFIG_DEBUG_INFO is not set
# CONFIG_DYNAMIC_DEBUG is not set
# CONFIG_DEBUG_FS is not set           # Removes /sys/kernel/debug
# CONFIG_KPROBES is not set
# CONFIG_PROFILING is not set
```

This reduces kernel size, boot time, and memory usage.

#### 7. Optimize initramfs

* Remove unnecessary modules from initramfs
* Use `lz4` compression (faster decompression than gzip)
* If rootfs is on NVMe and always present, consider booting without initramfs

### Boot Time Measurement

```bash
# Total kernel boot time (from first message to init)
dmesg | grep "Freeing unused kernel"
# Time between first dmesg line and this line = kernel boot time

# Driver probe times
dmesg | grep "probe" | sort -t'[' -k2 -n

# Detailed boot chart
systemd-analyze plot > boot.svg
```

### Typical Results

| Configuration        | Kernel Boot Time |
|----------------------|-----------------|
| Default L4T kernel   | 8–15 seconds    |
| Optimized (above)    | 3–6 seconds     |
| Aggressive (minimal) | 1–3 seconds     |

---

## 14. Writing Custom Kernel Modules

### Module Skeleton

```c
/* my_module.c */
#include <linux/module.h>
#include <linux/platform_device.h>
#include <linux/of.h>

static int my_probe(struct platform_device *pdev)
{
    dev_info(&pdev->dev, "my_module probed\n");
    /* Initialize hardware, register interfaces */
    return 0;
}

static int my_remove(struct platform_device *pdev)
{
    dev_info(&pdev->dev, "my_module removed\n");
    return 0;
}

static const struct of_device_id my_of_match[] = {
    { .compatible = "my-company,my-device" },
    { },
};
MODULE_DEVICE_TABLE(of, my_of_match);

static struct platform_driver my_driver = {
    .driver = {
        .name = "my-module",
        .of_match_table = my_of_match,
    },
    .probe = my_probe,
    .remove = my_remove,
};
module_platform_driver(my_driver);

MODULE_LICENSE("GPL");
MODULE_AUTHOR("Your Name");
MODULE_DESCRIPTION("My custom Jetson module");
```

### Makefile (Out-of-Tree Build)

```makefile
# Makefile
KERNEL_SRC ?= /lib/modules/$(shell uname -r)/build

obj-m := my_module.o

all:
	make -C $(KERNEL_SRC) M=$(PWD) modules

clean:
	make -C $(KERNEL_SRC) M=$(PWD) clean
```

### Build and Load

```bash
# Build (on Jetson or cross-compile)
make KERNEL_SRC=/path/to/kernel-5.10

# Load
sudo insmod my_module.ko

# Verify
dmesg | tail -5
lsmod | grep my_module

# Auto-load on boot
sudo cp my_module.ko /lib/modules/$(uname -r)/extra/
sudo depmod -a
echo "my_module" | sudo tee /etc/modules-load.d/my_module.conf
```

### Accessing Hardware From a Module

Common patterns for Jetson modules:

```c
/* Memory-mapped I/O */
void __iomem *base = devm_ioremap_resource(&pdev->dev, res);
writel(0x1, base + CONTROL_REG);

/* Clocks (via BPMP) */
struct clk *clk = devm_clk_get(&pdev->dev, "my-clk");
clk_prepare_enable(clk);

/* DMA with SMMU */
dma_addr_t dma_handle;
void *buf = dma_alloc_coherent(&pdev->dev, size, &dma_handle, GFP_KERNEL);

/* GPIO */
struct gpio_desc *gpio = devm_gpiod_get(&pdev->dev, "reset", GPIOD_OUT_HIGH);
```

---

## 15. Kernel Debugging on Jetson

### Serial Console

The primary debugging tool. Connect to the debug UART header:

```
Baud: 115200
Data: 8N1
```

All kernel messages (`printk`) appear here. Essential for boot failures where SSH is not available.

### Dynamic Debug

Enable per-file or per-function debug messages at runtime:

```bash
# Enable debug for nvgpu driver
echo "module nvgpu +p" | sudo tee /sys/kernel/debug/dynamic_debug/control

# Enable debug for a specific file
echo "file tegra-vi.c +p" | sudo tee /sys/kernel/debug/dynamic_debug/control

# View all available debug points
cat /sys/kernel/debug/dynamic_debug/control | grep tegra
```

Requires `CONFIG_DYNAMIC_DEBUG=y` (enabled by default, disable for production).

### ftrace

Kernel function tracer for profiling and debugging:

```bash
# Trace all function calls
echo function | sudo tee /sys/kernel/debug/tracing/current_tracer
echo 1 | sudo tee /sys/kernel/debug/tracing/tracing_on

# Read trace
cat /sys/kernel/debug/tracing/trace

# Trace specific functions
echo nvgpu_* | sudo tee /sys/kernel/debug/tracing/set_ftrace_filter
```

### Kernel Crash Analysis

If kernel panics:

1. Capture serial console output (has the panic trace)
2. Decode the stack trace:

```bash
# On host, with matching vmlinux
scripts/decode_stacktrace.sh vmlinux < panic_log.txt
```

3. Common Jetson panic causes:

| Panic Message                     | Likely Cause                          |
|-----------------------------------|---------------------------------------|
| `Unable to handle kernel NULL pointer` | Driver bug (uninitialized pointer) |
| `arm-smmu: Unhandled context fault`    | DMA to unmapped address            |
| `nvgpu: gpu init failed`              | GPU firmware missing or corrupt     |
| `kernel BUG at mm/page_alloc.c`       | Memory corruption or CMA issue      |

---

## 16. Production Kernel Hardening

### Lockdown

```
CONFIG_SECURITY_LOCKDOWN_LSM=y
CONFIG_LOCK_DOWN_KERNEL_FORCE_INTEGRITY=y
```

Prevents unsigned module loading and restricts access to sensitive kernel interfaces.

### Module Signing

```
CONFIG_MODULE_SIG=y
CONFIG_MODULE_SIG_FORCE=y
CONFIG_MODULE_SIG_SHA256=y
```

Only modules signed with your private key can be loaded. Prevents unauthorized code execution in kernel space.

### Remove Debug Interfaces

```
# CONFIG_DEBUG_FS is not set       # No /sys/kernel/debug
# CONFIG_PROC_KCORE is not set     # No /proc/kcore (memory dump)
# CONFIG_KEXEC is not set          # No kexec (reboot into arbitrary kernel)
# CONFIG_KALLSYMS is not set       # No symbol table (harder to exploit)
```

### Address Space Randomization

```
CONFIG_RANDOMIZE_BASE=y          # KASLR
CONFIG_RANDOMIZE_MODULE_REGION_FULL=y
```

Makes kernel exploitation harder by randomizing memory layout.

### Watchdog

Enable hardware watchdog to recover from kernel hangs:

```bash
# Enable watchdog
sudo systemctl enable watchdog
sudo systemctl start watchdog

# Configure timeout (e.g., 60 seconds)
echo 60 | sudo tee /sys/class/watchdog/watchdog0/timeout
```

If the kernel hangs and the watchdog is not fed, the system reboots automatically. Combined with A/B redundancy, this provides automatic recovery.

---

## 17. Kernel Update Strategy With A/B

When updating the kernel in an A/B system, the kernel, DTB, and modules must be updated atomically.

### What Gets Updated Per Slot

| Component    | Partition        | Must Match           |
|--------------|------------------|----------------------|
| Kernel Image | `kernel` / `kernel_b` | Module version   |
| DTB          | `kernel-dtb` / `kernel-dtb_b` | Kernel version |
| Modules      | In rootfs `/lib/modules/` | Kernel version |
| GPU firmware | In rootfs `/lib/firmware/` | nvgpu version  |

### Safe Update Flow

```
1. Currently running Slot A (kernel 5.10.104-tegra)
2. Prepare Slot B:
   a. Write new kernel Image to kernel_b partition
   b. Write new DTB to kernel-dtb_b partition
   c. Write new rootfs (with matching modules + firmware) to APP_b
3. Set Slot B active
4. Reboot
5. Validate (GPU loads, camera works, inference runs)
6. Mark Slot B successful
```

See [Rootfs & A/B Redundancy Guide](../Orin-Nano-Rootfs-and-AB-Redundancy/Guide.md) for full OTA details.

### Never Do This

* Update only the kernel without updating modules → `modprobe` failures
* Update modules without updating the kernel → version mismatch panics
* Use `apt upgrade` to update kernel packages → breaks A/B invariant
* Mix JetPack versions across slots → driver/firmware incompatibility

---

## 18. Common Kernel Issues and Solutions

### GPU Driver Fails to Load

```
nvgpu: probe of gpu.0 failed with error -110
```

**Causes:**
* GPU firmware missing from `/lib/firmware/nvidia/`
* Power domain not enabled in DTB
* BPMP firmware version mismatch

**Solution:** Verify firmware files exist, check `dmesg | grep bpmp`, ensure DTB matches your JetPack version.

### Camera Not Detected

```
tegra-vi: no channels found
```

**Causes:**
* Camera device tree node missing or `status = "disabled"`
* I2C address wrong in DTB
* CSI lane configuration mismatch
* Sensor driver not loaded

**Solution:** Check `media-ctl -p`, verify DTB camera nodes, check `i2cdetect` for sensor presence.

### Module Version Mismatch

```
modprobe: FATAL: Module nvgpu not found in directory /lib/modules/5.10.104-tegra
```

**Cause:** Kernel version does not match installed modules.

**Solution:** Rebuild and install modules matching the running kernel, or reflash with matching kernel + rootfs.

### Kernel OOM During Inference

```
Out of memory: Killed process 1234 (python3)
```

**Cause:** Combined CUDA + CPU + camera memory exceeds available RAM.

**Solution:** Quantize models (INT8), reduce camera buffer count, use DLA offload, monitor with `tegrastats`. See [Memory Architecture Guide](../Orin-Nano-Memory-Architecture/Guide.md#memory-planning-for-multi-model-pipelines).

### Slow Boot

**Cause:** Default kernel config with all drivers and debug enabled.

**Solution:** Follow Section 13 optimizations — disable unused DT nodes, modularize drivers, disable debug, suppress UART console.

---

## 19. References

* [NVIDIA Jetson Linux — Kernel](https://docs.nvidia.com/jetson/archives/r36.4/DeveloperGuide/SD/Kernel/Kernel.html) — official kernel documentation
* [NVIDIA Jetson Linux — Kernel Customization](https://docs.nvidia.com/jetson/archives/r36.4/DeveloperGuide/SD/Kernel/KernelCustomization.html) — customization guide
* [NVIDIA Jetson Linux — Kernel Boot Time Optimization](https://docs.nvidia.com/jetson/archives/r36.4/DeveloperGuide/SD/Kernel/KernelBootTimeOptimization.html) — boot optimization reference
* [NVIDIA Jetson Linux — Device Tree](https://docs.nvidia.com/jetson/archives/r36.4/DeveloperGuide/SD/Kernel/KernelAdaptation.html) — device tree adaptation
* [Linux Kernel Documentation](https://www.kernel.org/doc/html/latest/) — upstream kernel docs
* [ARM64 Booting](https://www.kernel.org/doc/html/latest/arm64/booting.html) — ARM64 boot protocol
* Main guide: [Nvidia Jetson Platform Guide](../Guide.md)
* Memory deep dive: [Orin Nano Memory Architecture](../Orin-Nano-Memory-Architecture/Guide.md)
* Rootfs & OTA: [Orin Nano Rootfs & A/B Redundancy](../Orin-Nano-Rootfs-and-AB-Redundancy/Guide.md)
