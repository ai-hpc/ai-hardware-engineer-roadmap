# Lecture 25: Capstone Project — Custom Linux Images with Yocto

## Overview

The previous 24 lectures covered how Linux works: processes, scheduling, memory management, drivers, filesystems, containers, and real-time tuning. This capstone lecture asks: how do you *build* the Linux image that runs on your AI hardware target? The answer is Yocto — the **industry-standard build system for embedded Linux**. Every platform covered in this course (Jetson, openpilot Agnos, custom FPGA boards, automotive ECUs) uses either Yocto or a Yocto-derived toolchain to produce **reproducible, minimal, production-grade** Linux images.

The mental model to carry here is that Yocto is a **factory** for Linux distributions. You describe what you want (machine hardware, packages, filesystem layout), and Yocto assembles, cross-compiles, and packages it into a flashable image. Unlike installing Ubuntu and removing packages, Yocto **starts from nothing** — only what you explicitly include ends up on the target. This **minimizes attack surface**, reduces OTA update size, and guarantees reproducibility: the same inputs always produce bit-identical outputs.

This lecture is structured as two projects that build on each other:
- **Project 1**: Build a minimal general-purpose image on QEMU (x86-64) — no hardware required
- **Project 2**: Build a full AI-ready custom image for the **NVIDIA Jetson Orin Nano 8GB** using meta-tegra

---

## Yocto Core Concepts

Before building, understand the vocabulary. Yocto has a steep initial learning curve that flattens once these five terms click.

```
Yocto Conceptual Map

  poky/                          ← Reference distribution (Yocto Project)
  ├── meta/                      ← Core recipes (Linux, glibc, busybox, systemd)
  ├── meta-poky/                 ← Poky distro configuration
  ├── meta-yocto-bsp/            ← Reference BSP machines (qemux86, qemuarm64)
  │
  meta-tegra/                    ← NVIDIA Tegra BSP layer (external)
  meta-openembedded/             ← Additional packages (Python, networking tools)
  meta-my-ai-image/             ← YOUR custom layer
  │
  build/
  ├── conf/
  │   ├── local.conf             ← Your machine/distro/build settings
  │   └── bblayers.conf          ← Which layers are active
  └── tmp/                       ← Build artifacts, rootfs, kernel
```

| Term | Meaning |
|------|---------|
| **Layer** | A directory of recipes grouped by purpose (BSP, distro, application). Stacked like Git branches — higher layers override lower ones. |
| **Recipe** (`.bb`) | A build script for one package: source URL, patches, compile flags, install paths. Analogous to a Debian `.spec` file. |
| **Machine** | Hardware target definition: CPU arch, kernel config, bootloader, Device Tree. Examples: `qemux86-64`, `jetson-orin-nano-devkit`. |
| **Distro** | Global policy: libc (glibc/musl), init system (systemd/SysV), package format (rpm/deb/ipk), debug flags. |
| **Image** | A recipe that collects packages into a rootfs and disk image. `core-image-minimal` ≈ 20 MB; `core-image-full-cmdline` ≈ 80 MB. |
| **BitBake** | The build engine. Parses all recipes, resolves dependencies, runs fetch/compile/install/package tasks in parallel. |

> **Key Insight:** Yocto does not download pre-built binaries — it cross-compiles everything from source for your exact target. This means you control every compiler flag, every enabled feature, and every kernel config option. The trade-off is that the first build takes 2–6 hours (it is building a complete cross-toolchain plus hundreds of packages); incremental rebuilds take minutes.

---

## Project 1: Minimal x86 Image on QEMU

### Goal

Build a bootable minimal Linux image for `qemux86-64` and run it in QEMU. **No physical hardware needed**. This teaches the **full Yocto workflow** in a safe environment before touching real hardware.

### Step 1: Install Prerequisites

```bash
# Ubuntu 22.04 / 24.04 host (recommended)
sudo apt-get update
sudo apt-get install -y \
    gawk wget git diffstat unzip texinfo gcc build-essential \
    chrpath socat cpio python3 python3-pip python3-pexpect \
    xz-utils debianutils iputils-ping python3-git python3-jinja2 \
    libegl1-mesa libsdl1.2-dev python3-subunit mesa-common-dev \
    zstd liblz4-tool file locales libacl1
# Ensure UTF-8 locale (BitBake requires it)
sudo locale-gen en_US.UTF-8
```

The build requires approximately **50–100 GB of free disk space** and 8 GB of RAM (16 GB recommended). BitBake parallelizes across all available CPU cores by default.

### Step 2: Clone Poky (Yocto Reference Distribution)

```bash
git clone git://git.yoctoproject.org/poky --branch scarthgap --depth 1
# scarthgap = Yocto 5.0 LTS, released April 2024, kernel 6.6 LTS
# Alternative: kirkstone (Yocto 4.0 LTS, kernel 5.15 LTS)
cd poky
```

> **Key Insight:** Always pin to a named LTS release (`scarthgap`, `kirkstone`). The `master` branch receives breaking changes continuously. Production systems stay on the same LTS release for the entire product lifetime — just like Linux kernel LTS branches.

### Step 3: Initialize the Build Environment

```bash
source oe-init-build-env build-qemu
# This script:
# 1. Creates the build/ directory
# 2. Creates build/conf/local.conf (your settings file)
# 3. Creates build/conf/bblayers.conf (active layers list)
# 4. Sets PATH and environment variables for BitBake
# 5. Changes your working directory to build/
```

After sourcing, you are inside `build-qemu/`. All `bitbake` commands run from here.

### Step 4: Configure local.conf

Open `conf/local.conf` and review/set these key variables:

```bash
# The hardware target
MACHINE = "qemux86-64"

# Number of parallel compile threads (set to number of CPU cores)
BB_NUMBER_THREADS = "8"
PARALLEL_MAKE = "-j 8"

# Download cache — reused across builds (set to a persistent directory)
DL_DIR = "/opt/yocto/downloads"

# Shared state cache — speeds up rebuilds dramatically
SSTATE_DIR = "/opt/yocto/sstate-cache"

# Image features: add ssh server for remote access
EXTRA_IMAGE_FEATURES += "ssh-server-openssh"

# Disk image format for QEMU
IMAGE_FSTYPES = "ext4 wic.qcow2"
```

> **Common Pitfall:** If you set `DL_DIR` and `SSTATE_DIR` on a separate disk, ensure it supports POSIX extended attributes (`xattr`). Some filesystems (FAT32, exFAT, some network shares) do not — BitBake will fail with cryptic errors about file locking.

### Step 5: Build the Minimal Image

```bash
bitbake core-image-minimal
```

This single command triggers BitBake to:
1. Parse all recipe files in all active layers
2. Resolve the full dependency graph (kernel → glibc → busybox → init → image)
3. Fetch source archives (Linux kernel, BusyBox, systemd, etc.)
4. Build the cross-compilation toolchain (sysroot for `qemux86-64`)
5. Cross-compile every package
6. Assemble the root filesystem
7. Create the disk image (`*.ext4`, `*.wic.qcow2`)

**First build time:** 1–3 hours depending on hardware. Watch the progress:

```bash
# In a separate terminal, monitor build status
bitbake -u taskexp core-image-minimal   # graphical task explorer
# Or watch the log:
tail -f tmp/log/cooker/qemux86-64/console-latest.log
```

### Step 6: Run in QEMU

```bash
# Boot the image in QEMU (no external QEMU installation needed — Yocto builds its own)
runqemu qemux86-64 core-image-minimal nographic

# You should see kernel boot messages, then a login prompt
# Default login: root (no password)
```

Inside the QEMU VM, verify the system:

```bash
uname -r              # should show 6.6.x kernel
cat /proc/cpuinfo     # QEMU virtual CPU
free -m               # memory
df -h                 # filesystem (tiny — this is a minimal image)
ps aux                # running processes (very few — truly minimal)
```

### Step 7: Understand the Build Artifacts

```
build-qemu/
├── conf/
│   ├── local.conf                    ← your settings
│   └── bblayers.conf                 ← active layers
└── tmp/
    ├── deploy/images/qemux86-64/
    │   ├── core-image-minimal-qemux86-64.ext4    ← root filesystem
    │   ├── core-image-minimal-qemux86-64.wic     ← complete disk image
    │   ├── bzImage                               ← compressed kernel
    │   └── modules-qemux86-64.tgz                ← kernel modules
    ├── work/x86_64-linux/            ← host tools (cross-compiler)
    ├── work/qemux86_64-poky-linux/   ← target packages
    │   └── linux-yocto-6.6.*/        ← kernel build directory
    └── sysroots/qemux86-64/          ← target sysroot for SDK
```

The `work/` directory contains the unpacked, patched, compiled, and installed files for every package. **When debugging a package build failure, this is where you look.**

### Step 8: Customize — Add a Package

To add Python 3 to the image, add to `local.conf`:

```bash
IMAGE_INSTALL:append = " python3 python3-pip"
```

Then rebuild (incremental — only adds the new packages):

```bash
bitbake core-image-minimal
```

> **Key Insight:** Yocto's shared state (sstate) cache records every task's output hash. Adding a package does not rebuild the kernel — it pulls the cached kernel artifact and only builds the new packages. Incremental builds are fast (seconds to minutes), which is why the 2-hour initial build is worth the investment.

---

## Project 2: NVIDIA Jetson Orin Nano 8GB Custom AI Image

### Goal

Build a production-quality AI image for the Jetson Orin Nano 8GB developer kit that includes:
- L4T 36.x kernel (6.1 LTS with NVIDIA Tegra patches)
- NVIDIA GPU drivers and CUDA userspace
- TensorRT and cuDNN
- OpenCV with CUDA support
- systemd, SSH, Python 3

This image can replace the default NVIDIA JetPack install with a **reproducible, customizable, minimal** Yocto-based alternative. Used in production for edge AI appliances.

### Architecture: meta-tegra

```
Layer Stack for Jetson Orin Nano 8GB:

meta-tegra/              ← NVIDIA Tegra BSP: kernel, U-Boot, flash tools
                            Machine: jetson-orin-nano-devkit
meta-openembedded/       ← Extra packages: OpenCV, Python libs, networking
meta-tegra-community/    ← Optional: TensorRT, CUDA recipes from community
meta-my-jetson-image/    ← YOUR layer: custom recipes, image definition
poky/meta/               ← Core (gcc, glibc, busybox, systemd)
```

**meta-tegra** is maintained by the Open Embedded for Tegra (OE4T) community. It **tracks NVIDIA's L4T releases** and provides the kernel, U-Boot, Device Tree, and board-specific drivers.

### Step 1: Prepare the Workspace

```bash
mkdir jetson-orin-build && cd jetson-orin-build

# Clone all required layers
git clone git://git.yoctoproject.org/poky             --branch scarthgap
git clone https://github.com/OE4T/meta-tegra          --branch scarthgap
git clone https://github.com/openembedded/meta-openembedded --branch scarthgap

# Confirm meta-tegra supports Orin Nano
ls meta-tegra/conf/machine/
# Should list: jetson-orin-nano-devkit.conf, jetson-agx-orin-devkit.conf, etc.
```

> **Key Insight:** meta-tegra branch names track Yocto release names (`scarthgap`, `kirkstone`), not L4T versions. The L4T version is set inside meta-tegra's recipes. Scarthgap branch of meta-tegra corresponds to L4T 36.x (JetPack 6.x).

### Step 2: Initialize and Configure the Build

```bash
source poky/oe-init-build-env build-jetson

# Edit bblayers.conf to include all layers:
cat > conf/bblayers.conf << 'EOF'
POKY_BBLAYERS_CONF_VERSION = "2"
BBPATH = "${TOPDIR}"
BBFILES ?= ""

BBLAYERS ?= " \
  ${TOPDIR}/../poky/meta \
  ${TOPDIR}/../poky/meta-poky \
  ${TOPDIR}/../poky/meta-yocto-bsp \
  ${TOPDIR}/../meta-tegra \
  ${TOPDIR}/../meta-openembedded/meta-oe \
  ${TOPDIR}/../meta-openembedded/meta-python \
  ${TOPDIR}/../meta-openembedded/meta-networking \
  "
EOF
```

### Step 3: Configure local.conf for Jetson Orin Nano 8GB

```bash
cat > conf/local.conf << 'EOF'
# ── Machine ───────────────────────────────────────────────────────────────────
MACHINE = "jetson-orin-nano-devkit"
# jetson-orin-nano-devkit covers both Orin Nano 4GB and 8GB developer kits.
# The Orin Nano 8GB module uses a different SOM; the MACHINE is the same
# because the devkit carrier board is identical. The SOM variant is selected
# at flash time via the appropriate Device Tree Blob (DTB).

# ── Distribution ─────────────────────────────────────────────────────────────
DISTRO = "poky"
# Use systemd instead of SysV init (required for most modern AI stacks)
DISTRO_FEATURES:append = " systemd"
VIRTUAL-RUNTIME_init_manager = "systemd"
DISTRO_FEATURES_BACKFILL_CONSIDERED += "sysvinit"

# ── NVIDIA GPU / CUDA ─────────────────────────────────────────────────────────
# Include NVIDIA binary drivers (requires accepting NVIDIA's EULA)
LICENSE_FLAGS_ACCEPTED = "commercial_nvidia-l4t-core nvidia-eula"
# Enable CUDA support in packages that support it (OpenCV, etc.)
CUDA_NVCC_EXTRA_FLAGS = "--gpu-architecture=sm_87"
# Jetson Orin Nano has Ampere GPU, SM 8.7

# ── Packages to include ────────────────────────────────────────────────────────
# Core AI stack
IMAGE_INSTALL:append = " \
    cuda-toolkit \
    tensorrt \
    libcudnn \
    opencv \
    python3 \
    python3-numpy \
    python3-pip \
"
# System utilities
IMAGE_INSTALL:append = " \
    openssh \
    git \
    htop \
    i2c-tools \
    can-utils \
    iproute2 \
"
# Real-time tuning tools (from previous OS lectures)
IMAGE_INSTALL:append = " \
    rt-tests \
    trace-cmd \
    perf \
"

# ── Image Features ────────────────────────────────────────────────────────────
EXTRA_IMAGE_FEATURES += "ssh-server-openssh debug-tweaks"
# debug-tweaks: enables root login without password (remove for production)

# ── Build parallelism ─────────────────────────────────────────────────────────
BB_NUMBER_THREADS = "12"
PARALLEL_MAKE = "-j 12"

# ── Shared caches ─────────────────────────────────────────────────────────────
DL_DIR = "/opt/yocto/downloads"
SSTATE_DIR = "/opt/yocto/sstate-cache"

# ── Output image format ───────────────────────────────────────────────────────
# Tegra images use tegraflash format — includes CBoot, DTB, rootfs partition
IMAGE_FSTYPES = "tegraflash"
EOF
```

> **Common Pitfall:** The `LICENSE_FLAGS_ACCEPTED` variable must include `nvidia-eula` exactly as shown. If you forget it, BitBake will silently skip all NVIDIA binary recipes and build without CUDA — you will not see an error, just a missing package at runtime. Always run `bitbake -g core-image-my-ai` and check `task-depends.dot` if CUDA packages seem absent.

### Step 4: Create a Custom Image Recipe

Create your own layer and image recipe to keep customizations **clean and version-controlled**:

```bash
mkdir -p ../meta-my-jetson-image/recipes-core/images
cat > ../meta-my-jetson-image/recipes-core/images/jetson-orin-ai.bb << 'EOF'
# Custom AI image for Jetson Orin Nano 8GB
SUMMARY = "AI inference image for Jetson Orin Nano 8GB"
LICENSE = "MIT"

# Inherit from the standard Tegra image class
require recipes-core/images/tegra-image.inc

# Add the base Tegra runtime (kernel modules, L4T libs)
IMAGE_INSTALL += "tegra-libraries-core"

# ── AI inference stack ────────────────────────────────────────────────────────
IMAGE_INSTALL += " \
    cuda-toolkit \
    tensorrt \
    libcudnn \
    opencv \
    python3 \
    python3-numpy \
    python3-onnxruntime \
"

# ── Real-time and diagnostic tools ───────────────────────────────────────────
IMAGE_INSTALL += " \
    rt-tests \
    trace-cmd \
    bpftrace \
    can-utils \
    i2c-tools \
"

# ── System configuration ──────────────────────────────────────────────────────
IMAGE_INSTALL += " \
    openssh \
    systemd \
    util-linux \
    procps \
    htop \
"

# Enable root filesystem expansion on first boot (fills the eMMC partition)
IMAGE_FEATURES += "read-only-rootfs-delayed-postinsts"
EOF
```

Add the new layer to `bblayers.conf`:

```bash
# Add to the BBLAYERS list in conf/bblayers.conf:
#   ${TOPDIR}/../meta-my-jetson-image \
```

### Step 5: Build the Image

```bash
bitbake jetson-orin-ai
```

This build is larger than Project 1 — it includes CUDA and TensorRT. Expected time on a modern workstation:
- First build: 4–8 hours (downloads ~15 GB of sources and binary blobs)
- Incremental rebuild after config change: 10–30 minutes
- Incremental rebuild after adding one package: 2–5 minutes

Monitor progress:
```bash
# Real-time task log
tail -f tmp/log/cooker/jetson-orin-nano-devkit/console-latest.log

# Show currently running tasks
bitbake -u taskexp jetson-orin-ai
```

### Step 6: Inspect the Output

```bash
ls tmp/deploy/images/jetson-orin-nano-devkit/
# Key files:
# jetson-orin-ai-jetson-orin-nano-devkit.tegraflash.tar.gz  ← flash bundle
# Image                                                      ← compressed kernel
# tegra234-p3768-0000+p3767-0005-nv.dtb                    ← Orin Nano 8GB DTB
# bootloader/                                               ← CBoot, MB1, TOS
```

The `tegraflash.tar.gz` bundle contains **everything `tegraflash.py` needs to flash the board**. The DTB `p3767-0005` is the Orin Nano 8GB module identifier.

```
tegraflash bundle contents:
├── flash.xml              ← partition layout (eMMC map)
├── Image                  ← kernel
├── tegra234-*.dtb         ← Device Tree for your specific module
├── boot.img               ← kernel + initramfs
├── system.img             ← rootfs (ext4, ~2–4 GB)
├── bootloader/
│   ├── mb1_t234_prod.bin  ← MB1 bootloader (NVIDIA signed)
│   ├── cboot.bin          ← CBoot (U-Boot replacement)
│   └── tos-a.img          ← Trusted OS (TrustZone)
└── tegraflash.py          ← flash script
```

### Step 7: Flash to Jetson Orin Nano 8GB

**Hardware setup:**

```
1. Connect Jetson Orin Nano devkit to host PC via USB-C (J15 port — Recovery USB)
2. Insert jumper on J14 (Force Recovery header) pins 1-2
3. Power on the board
4. Confirm the board appears on host: lsusb | grep NVIDIA
   → "ID 0955:7323 NVIDIA Corp. APX" indicates recovery mode
5. Remove the recovery jumper
```

**Flash:**

```bash
# Extract the tegraflash bundle
tar xf tmp/deploy/images/jetson-orin-nano-devkit/jetson-orin-ai-jetson-orin-nano-devkit.tegraflash.tar.gz
cd jetson-orin-ai-*tegraflash/

# Flash all partitions
sudo ./tegraflash.py --flash all
# This takes 5–15 minutes
# Progress: writes bootloader, DTB, kernel, and rootfs to eMMC
```

> **Common Pitfall:** Flashing fails with "device not found" if the USB cable is plugged into the wrong port. The Orin Nano devkit has two USB-C ports: J14 (display/data) and J15 (recovery/debug). Recovery mode only works on J15. If your cable is marginal (high-resistance), the APX device appears briefly then disappears — use a short, known-good USB 3.0 cable.

### Step 8: First Boot and Validation

After flashing, remove the recovery jumper (if still in), power cycle, and connect via serial console or SSH:

```bash
# Serial console (115200 baud) on J14 micro-USB
screen /dev/ttyUSB0 115200

# Or wait for DHCP and SSH in
ssh root@<jetson-ip>
```

Validate the key AI components:

```bash
# Verify NVIDIA GPU driver loaded
nvidia-smi
# Should show: "Orin (nvgpu)", CUDA Version, memory

# Verify CUDA installation
nvcc --version
python3 -c "import ctypes; ctypes.cdll.LoadLibrary('libcuda.so'); print('CUDA OK')"

# Verify TensorRT
python3 -c "import tensorrt as trt; print('TRT', trt.__version__)"

# Check kernel version (should be L4T 6.1.x)
uname -r

# Check thermal zones (from Lecture 1 — thermal monitoring)
cat /sys/class/thermal/thermal_zone*/temp
# CPU, GPU, SoC, CV zones in millidegrees Celsius

# RT scheduling tools available?
cyclictest --help
chrt -p 1
```

### Step 9: Customization Patterns

Once the base image works, these are the **common customizations for AI deployment**:

```bash
# ── Disable unnecessary services (reduce boot time and attack surface) ────────
# In a bbappend or local.conf:
SYSTEMD_AUTO_ENABLE:pn-avahi-daemon = "disable"
SYSTEMD_AUTO_ENABLE:pn-bluetooth = "disable"

# ── Pre-install a TensorRT model at build time ────────────────────────────────
# In your image recipe:
IMAGE_INSTALL += "my-model-package"
# Where my-model-package.bb installs the .engine file to /opt/models/

# ── Apply RT tuning at boot (from Lecture 7) ─────────────────────────────────
# Add a systemd service that runs the RT tuning script:
# isolcpus=4-11 nohz_full=4-11 rcu_nocbs=4-11
# in /boot/extlinux/extlinux.conf APPEND line (modify via bbappend)

# ── Set root password for production (remove debug-tweaks) ───────────────────
# Remove "debug-tweaks" from IMAGE_FEATURES
# Add to local.conf:
INHERIT += "extrausers"
EXTRA_USERS_PARAMS = "usermod -p '\$6\$...' root;"

# ── OTA A/B update support (from Lecture 22) ─────────────────────────────────
# meta-tegra supports OTA updates via NVIDIA's Over-the-Air (OTA) framework
# Enable with:
TEGRA_REDUNDANT_BOOT = "1"
```

---

## Connecting to the OS Curriculum

| OS Lecture | Yocto Manifestation |
|-----------|---------------------|
| Lecture 1: Linux kernel architecture | Yocto builds the kernel from source; `KCONFIG_MODE` controls which drivers compile in |
| Lecture 5: Boot process & Device Tree | `meta-tegra` provides DTB files; `SRC_URI` in kernel recipe patches the DTS |
| Lecture 7: PREEMPT_RT | Add `PREFERRED_PROVIDER_virtual/kernel = "linux-tegra"` and patch `CONFIG_PREEMPT_RT=y` |
| Lecture 17: Linux driver model | Recipes in `meta-tegra/recipes-kernel/` add out-of-tree modules (`nvgpu.ko`, `nvcsi.ko`) |
| Lecture 21: Filesystems | `IMAGE_FSTYPES` controls ext4/btrfs/F2FS; `EXTRA_IMAGECMD:ext4` sets block size |
| Lecture 22: OTA partitioning | `tegraflash.xml` defines A/B partition layout; `TEGRA_REDUNDANT_BOOT=1` enables it |
| Lecture 23: Containers | Add `docker` to `IMAGE_INSTALL`; NVIDIA Container Runtime is a separate meta-tegra recipe |
| Lecture 24: L4T, Yocto for AI | This project IS the production workflow for AI edge devices |

---

## Summary

| Project | Target | Image | Key Tool | Flash Method |
|---------|--------|-------|----------|--------------|
| Project 1 | `qemux86-64` (virtual) | `core-image-minimal` | `runqemu` | No flashing needed |
| Project 2 | Jetson Orin Nano 8GB | `jetson-orin-ai` | `tegraflash.py` | USB-C Recovery Mode |

### Conceptual Review

- **Why use Yocto instead of flashing the stock JetPack image?** Stock JetPack is a fixed Ubuntu-based distribution you cannot easily customize or reproduce bit-for-bit. Yocto builds a minimal image from source that contains only what you explicitly include, with reproducible builds guaranteed by the sstate cache. For production devices, this reduces attack surface, controls binary composition, and enables automated OTA updates.
- **What does `source oe-init-build-env` actually do?** It sets `BBPATH`, `PATH`, and `BUILDDIR` environment variables, creates `build/conf/local.conf` and `bblayers.conf` from templates, and cd's into the build directory. BitBake relies on these environment variables to locate recipes and configuration.
- **Why does the Jetson image use `tegraflash` format instead of a raw ext4?** The Tegra platform has a complex partition layout: separate partitions for MB1 (first-stage bootloader), CBoot, Device Tree, A/B kernel, A/B rootfs, and RPMB (secure storage). `tegraflash.py` knows how to write each partition to the correct eMMC offset using the `flash.xml` partition map. A raw ext4 would only contain the rootfs and boot on nothing.
- **What is the relationship between `meta-tegra` and L4T?** `meta-tegra` is the Yocto layer that packages L4T. It pulls the L4T kernel source from NVIDIA's GitHub, applies Tegra-specific patches, and builds it with the configuration needed for the target board. The layer pins to specific L4T versions in its recipe files.
- **How do you add a custom Python application to the Yocto image?** Write a recipe (`my-app.bb`) that sets `SRC_URI` to your application's source, defines `do_install` to copy files to `${D}/opt/my-app/`, and adds `my-app` to `IMAGE_INSTALL`. BitBake handles cross-compilation, dependencies, and packaging automatically.
- **Why does the first build take 4–8 hours but incremental builds take minutes?** BitBake hashes the inputs of every task (source code, recipe contents, environment variables). If the hash is unchanged, it restores the output from the sstate cache without rerunning the task. Adding one package to `IMAGE_INSTALL` only triggers that package's fetch/compile/install tasks, then reassembles the rootfs — the kernel, glibc, and all other packages are served from cache instantly.

---

## AI Hardware Connection

- Yocto is the standard build system for every production AI edge device in this roadmap: openpilot Agnos, Jetson commercial deployments, automotive ECU Linux partitions, and custom FPGA SoC boards all use Yocto or Yocto-derived build systems (meta-tegra, OpenWRT).
- The `PREEMPT_RT` kernel config from Lecture 7 is added to a Yocto Jetson build by creating a `linux-tegra_%.bbappend` file that sets `CONFIG_PREEMPT_RT=y` in a kernel fragment — the same kernel you studied is now yours to configure at build time.
- OTA A/B updates (Lecture 22) on Jetson are enabled by `TEGRA_REDUNDANT_BOOT = "1"` in Yocto and managed by NVIDIA's `nv-update-engine` — building this into the image from the start is far cheaper than retrofitting it after deployment.
- The TensorRT and CUDA packages in the Yocto image (`cuda-toolkit`, `tensorrt`) are the same userspace libraries studied in Phase 4 Track B (Jetson), but now you control which version is on the device — critical for ensuring model compatibility across a fleet of devices that may have been manufactured months apart.
- Reproducibility matters for safety: ISO 26262 ASIL certification requires the ability to rebuild the exact binary that was certified. Yocto's sstate cache and `SSTATE_MIRRORS` infrastructure provide this guarantee — a certified build can be reproduced years later from the same recipe inputs.
- Custom layers (`meta-my-jetson-image`) are the production engineering artifact: when you join an AI hardware team, you will be working in a meta layer that defines the company's product image, adding sensors, optimizing services, and managing the OS across firmware updates.
