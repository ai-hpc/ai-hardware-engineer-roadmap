# Lecture 5: Kernel Modules, Boot Process & Device Tree

## Overview

Before any AI inference can happen, the hardware must be initialized, the kernel must be loaded, and every device — camera sensor, NPU, GPIO expander — must be discovered and driven. The core challenge this lecture addresses is: how does a kernel that cannot know about all possible hardware in advance still manage to drive it correctly? The mental model has two parts: the **boot sequence** as a chain of trust (each stage verifies and hands off to the next), and the **Device Tree** as a hardware description contract (a data file that tells the kernel what hardware exists without hardcoding it into kernel source). For an AI hardware engineer, this matters every time you bring up a new camera sensor, integrate an FPGA accelerator, or debug why an inference accelerator node never gets a `probe()` call at boot.

---

## Linux Boot Sequence on ARM SoC

Modern embedded AI boards (Jetson Orin, Zynq UltraScale+, i.MX8) follow this sequence. Each stage is a **separate program** that runs, initializes a layer of hardware, and then launches the next stage.

| Stage | Responsible | Artifact loaded | Notes |
|---|---|---|---|
| 1. Power-on | SoC ROM (BootROM) | BootROM code from mask ROM | Checks fuses, loads signed next stage from flash |
| 2. Primary bootloader | U-Boot SPL or UEFI stub | SPL / TF-A BL2 | Initializes DRAM, minimal clocks |
| 3. Firmware | TF-A BL31 / PSCI | Secure monitor (EL3) | Sets up ATF, optionally loads OP-TEE |
| 4. Second bootloader | U-Boot proper or UEFI | Kernel + DTB + initramfs | Reads storage, sets boot args |
| 5. Kernel decompression | Kernel `head.S` | Decompresses `Image.gz` | Verifies DTB, sets up early paging |
| 6. Kernel init | `start_kernel()` | Initializes all subsystems | Memory, scheduler, drivers, VFS |
| 7. PID 1 | systemd (or `init`) | Mounts rootfs, starts services | udev creates `/dev` nodes |

On x86: UEFI (or legacy BIOS) replaces U-Boot. On Jetson: NVIDIA's MB1 (Miniboot) and MB2 correspond to SPL and U-Boot; CBoot is NVIDIA's U-Boot replacement in JetPack 5.

```
ARM SoC Boot Chain — Jetson Example
                    Power applied
                         │
                         ▼
               ┌─────────────────┐
               │  BootROM (EL3)  │  ← burned into SoC silicon
               │  checks OTP     │
               │  verifies MB1   │
               └────────┬────────┘
                         │ loads signed MB1 from eMMC
                         ▼
               ┌─────────────────┐
               │  MB1 (EL3)      │  ← NVIDIA Miniboot
               │  init DRAM      │
               │  clock setup    │
               └────────┬────────┘
                         │ loads MB2 + TF-A BL31
                         ▼
               ┌─────────────────┐
               │  TF-A BL31(EL3) │  ← ARM Trusted Firmware
               │  PSCI handler   │  stays resident in EL3
               │  switches to EL1│
               └────────┬────────┘
                         │ loads CBoot / UEFI
                         ▼
               ┌─────────────────┐
               │  CBoot / UEFI   │  ← like U-Boot for Jetson
               │  (EL2 / EL1)    │
               │  loads kernel   │
               │  + DTB +initrd  │
               └────────┬────────┘
                         │ jumps to kernel entry (EL1)
                         ▼
               ┌─────────────────┐
               │  Linux Kernel   │  ← start_kernel()
               │  (EL1)          │
               │  subsystem init │
               │  device probing │
               └────────┬────────┘
                         │ executes /sbin/init
                         ▼
               ┌─────────────────┐
               │  systemd (EL0)  │  ← PID 1
               │  mounts rootfs  │
               │  starts services│
               └─────────────────┘
```

> **Key Insight:** The boot chain is not just a sequence of programs — it is a **chain of trust**. Each stage verifies the digital signature of the next before executing it. If any link in the chain is broken (wrong key, tampered binary), the boot halts. For production AV platforms, this chain is the security foundation: it guarantees that the running kernel and inference binaries have not been tampered with since the manufacturer signed them.

---

## Secure Boot

**Trust chain** from ROM to running kernel: each stage **verifies the signature** of the next before executing it.

- BootROM verifies SPL/BL2 signature with a key burned into fuses
- BL2 verifies BL31 and U-Boot/UEFI
- U-Boot verifies kernel image signature
- Kernel verifies rootfs (dm-verity) or module signatures

**Jetson Secure Boot**: BCT (Boot Configuration Table) and BL31 are signed with NVIDIA's key by default; production deployment uses a customer RSA-2048/4096 or ECDSA key pair fused into OTP. `tegraflash` handles signing and flashing.

**Secure boot is mandatory** for production AV platforms — prevents substitution of a malicious kernel image or modified inference binary at the bootloader stage.

> **Common Pitfall:** During development, it is tempting to disable secure boot for convenience. The danger is forgetting to re-enable it before shipping a product. A development image with secure boot disabled will accept any unsigned kernel — including one an attacker installs via physical access to the eMMC. Always develop with secure boot enabled using developer keys, and switch to production keys for release builds.

---

## UEFI vs U-Boot

| | UEFI | U-Boot |
|---|---|---|
| Primary domain | x86 servers, workstations, modern ARM servers | Embedded: Zynq, Jetson, Raspberry Pi, i.MX |
| Standard | UEFI specification (Tianocore/EDK2) | Open-source; board-specific configuration |
| Boot protocol | EFI stub in kernel; DTB from EFI config | Passes kernel + DTB address in registers |
| Script / config | EFI variables, NVRAM | U-Boot environment variables, `boot.cmd` |
| Network boot | PXE via UEFI network stack | TFTP + NFS; common for embedded development |

Both load the kernel image, a DTB, and optionally an initramfs into memory, then jump to the kernel entry point.

---

## initramfs

- Compressed cpio archive (gzip, lz4, zstd) embedded in the kernel or loaded separately
- Contains early userspace: busybox, udev, cryptsetup, `fsck`, custom `init` script
- Kernel mounts it as the initial rootfs in a tmpfs; runs `/init`; mounts the real rootfs; `switch_root` to the permanent root
- On Jetson, NVIDIA uses initramfs for early TNSPEC (board identification) and extlinux-based boot selection

```bash
mkdir /tmp/initrd && cd /tmp/initrd
zcat /boot/initrd.img | cpio -idm    # decompress and extract the initramfs cpio archive
ls -la                               # busybox, lib, udev, etc.
```

The initramfs extraction above unpacks the entire early userspace, revealing what tools and scripts the kernel uses before the real root filesystem is mounted.

> **Key Insight:** The initramfs exists because the kernel cannot always mount the real root filesystem without drivers that haven't been loaded yet. For example, if root is on a LUKS-encrypted NVMe drive, the kernel needs `cryptsetup` from initramfs to unlock it before it can mount root. On Jetson, initramfs handles board identification (TNSPEC) so the same kernel image can boot on multiple Jetson variants with different carrier boards.

Now that we understand how the kernel gets loaded, let's look at how it discovers what hardware it needs to drive — the Device Tree.

---

## Device Tree

### Purpose

**Hardware description** for SoCs without self-describing buses. PCIe is self-describing (devices report vendor/device IDs); I2C, SPI, UART, AXI, and MMIO peripherals are not — the kernel must be told they exist.

The Device Tree replaces per-board `#ifdef` hacks in kernel source with a **data file** the bootloader passes to the kernel at runtime. U-Boot places the **DTB address** in a register before jumping to the kernel entry point.

Think of the Device Tree as a wiring diagram: it tells the kernel "there is a Sony IMX477 camera sensor connected to I2C bus 0 at address 0x1a, reset by GPIO 42, and its CSI output connects to CSI port 0."

```
Device Tree → Driver Binding Flow
┌──────────────────────────────────────────────────────────┐
│  Device Tree (DTB file, loaded by bootloader)            │
│                                                          │
│  i2c0 {                                                  │
│    camera0: imx477@1a {                                  │
│      compatible = "sony,imx477";  ← binding key         │
│      reg = <0x1a>;                                       │
│    };                                                    │
│  };                                                      │
└───────────────────────┬──────────────────────────────────┘
                        │ kernel parses DTB at boot
                        ▼
┌──────────────────────────────────────────────────────────┐
│  Kernel OF (Open Firmware) matching                      │
│  → scans all registered drivers                         │
│  → finds imx477 driver with of_match_table:             │
│    { .compatible = "sony,imx477" }                      │
└───────────────────────┬──────────────────────────────────┘
                        │ match found
                        ▼
┌──────────────────────────────────────────────────────────┐
│  Driver probe() called                                   │
│  → reads reg property (I2C addr 0x1a)                   │
│  → requests GPIO 42 for reset                           │
│  → registers V4L2 subdevice                             │
│  → camera is now accessible at /dev/videoN              │
└──────────────────────────────────────────────────────────┘
```

### Node Structure

```dts
/* Example: IMX477 camera sensor on I2C bus */
&i2c0 {
    camera0: imx477@1a {
        compatible = "sony,imx477";      /* driver match string */
        reg = <0x1a>;                    /* I2C address 0x1a */
        clocks = <&clk IMX477_CLK>;      /* clock provider reference */
        clock-names = "xclk";
        reset-gpios = <&gpio 42 GPIO_ACTIVE_LOW>; /* GPIO 42, active-low reset */
        port {
            cam0_ep: endpoint {
                remote-endpoint = <&csi0_ep>;  /* connects to CSI port 0 */
                data-lanes = <1 2>;            /* uses MIPI D-PHY lanes 1 and 2 */
            };
        };
    };
};
```

| Property | Meaning |
|---|---|
| `compatible` | String list; kernel matches against `of_match_table` in driver |
| `reg` | MMIO base address and size, or bus address (I2C, SPI) |
| `interrupts` | IRQ specifier: GIC SPI number, trigger type |
| `clocks` | Clock provider handle and clock ID |
| `dma-names` | Named DMA channels assigned to the device |
| `status` | `"okay"` to enable; `"disabled"` to suppress driver probe |

### Compilation and Inspection

```bash
dtc -I dts -O dtb -o my_board.dtb my_board.dts    # compile DTS → DTB binary
dtc -I dtb -O dts -o decoded.dts my_board.dtb      # decompile DTB back to readable DTS
ls /sys/firmware/devicetree/base/                   # live DT from running kernel
cat /sys/firmware/devicetree/base/model             # board model string
```

### Device Tree Overlays (DTBO)

**Overlays** patch the base DT at runtime without rebuilding the kernel or base DTB. This is the mechanism for adding support for a new sensor or peripheral to a platform that ships with a fixed base DTB.

The overlay sequence:

1. **Write a `.dts` overlay file** that adds or modifies nodes in the base tree.
2. **Compile to `.dtbo`** using `dtc`.
3. **Apply at runtime** (Raspberry Pi) or configure in the bootloader (Jetson extlinux.conf).
4. **Kernel merges the overlay** into the live device tree in memory.
5. **Driver probe() fires** for any newly enabled node.

- **Jetson pin mux overlays**: select UART vs SPI vs I2C function for carrier board expansion pins
- **Raspberry Pi HAT overlays**: enable I2S audio, SPI ADC, camera sensor nodes
- **Zynq PL overlays**: load partial bitstream and add DT nodes for FPGA-connected AXI peripherals

```bash
dtoverlay imx477                        # apply overlay (Raspberry Pi)
cat /boot/extlinux/extlinux.conf        # FDT_OVERLAYS= line on Jetson
ls /sys/firmware/devicetree/base/       # verify overlay nodes appeared
```

> **Key Insight:** Device Tree overlays are the correct mechanism for adding carrier board peripherals to a base Jetson or Raspberry Pi image without modifying the vendor-provided base DTB. A custom carrier board with an IMX477 camera on a non-standard I2C bus can be supported with a 20-line overlay file, rather than requiring a full DTB modification that would break on every L4T update.

> **Common Pitfall:** If a driver's `probe()` never runs, the first thing to check is whether the DT node has `status = "okay"`. Nodes default to `"disabled"` if the property is absent or set to any other value. This is a frequent mistake when porting overlays between platforms — the `status` property must be explicitly set to `"okay"` for the kernel to attempt to bind a driver to the node.

---

## Kernel Modules

### Module Entry, Exit, and Device Matching

```c
static const struct of_device_id my_of_match[] = {
    { .compatible = "vendor,my-accel" },  /* match DT node with this compatible string */
    { }                                    /* sentinel; marks end of match table */
};
MODULE_DEVICE_TABLE(of, my_of_match);    /* writes alias → modules.alias → udev auto-load */

static struct platform_driver my_driver = {
    .probe  = my_probe,    /* called when a matching DT node is found */
    .remove = my_remove,   /* called when the device is removed or module unloaded */
    .driver = {
        .name           = "my-accel",
        .of_match_table = my_of_match,  /* binds this driver to DT nodes */
    },
};
module_platform_driver(my_driver);       /* wraps module_init / module_exit */
MODULE_LICENSE("GPL");
```

When a DT node with `compatible = "vendor,my-accel"` appears (at boot or via overlay), udev reads `modules.alias`, calls `modprobe`, and the driver's `probe()` function runs.

### modprobe vs insmod

| Command | Behavior |
|---|---|
| `insmod my.ko` | Load from file path; no dependency resolution |
| `modprobe mymodule` | Resolve and load dependencies from `/lib/modules/$(uname -r)/modules.dep` |
| `modprobe -r mymodule` | Unload module and any unused dependencies |
| `depmod -a` | Rebuild `modules.dep` after installing new modules |
| `lsmod` | List loaded modules, usage count, dependents |
| `modinfo nvme` | Show parameters, license, firmware version fields |

> **Key Insight:** `modprobe` is almost always the right command to use. `insmod` loads a single `.ko` file without checking dependencies. If your FPGA PCIe driver depends on the `dma-buf` subsystem module, `insmod` will fail with a cryptic symbol resolution error, while `modprobe` automatically loads `dma-buf` first. Always use `modprobe` in scripts and systemd units; use `insmod` only when you are testing a single module during development.

### Module Signing

- `CONFIG_MODULE_SIG_FORCE` **rejects unsigned modules**; production Jetson and automotive ECU kernels enforce this
- Sign during build: `scripts/sign-file sha256 signing_key.pem signing_cert.pem my_driver.ko`
- Custom FPGA PCIe driver must be signed before deployment on a secure-boot platform

### DKMS

**Dynamic Kernel Module Support** rebuilds out-of-tree modules automatically when the kernel is updated.

```bash
dkms add -m my-fpga-driver -v 1.0       # register the module source with DKMS
dkms build -m my-fpga-driver -v 1.0    # compile against current kernel
dkms install -m my-fpga-driver -v 1.0  # install the compiled .ko
```

DKMS is standard for the NVIDIA proprietary GPU driver on development hosts and for custom FPGA PCIe drivers that must survive kernel point-release updates without manual rebuilds.

> **Common Pitfall:** DKMS rebuilds fail silently if the kernel headers are not installed for the new kernel version. After a `apt upgrade` that upgrades the kernel, run `apt install linux-headers-$(uname -r)` before rebooting, or the DKMS post-install hook will fail to rebuild. On systems where CUDA/TensorRT is critical, include this in the upgrade procedure.

---

## Kernel Command Line

Passed by U-Boot (`bootargs` env variable) or UEFI. **Readable at `/proc/cmdline`**.

| Parameter | Effect |
|---|---|
| `console=ttyS0,115200n8` | Kernel log to serial at boot |
| `root=/dev/mmcblk0p1` | Root filesystem device |
| `rdinit=/sbin/init` | First userspace process in initramfs |
| `isolcpus=4-7` | Exclude cores 4–7 from the general scheduler |
| `nohz_full=4-7` | Eliminate scheduler tick interrupts on isolated cores |
| `rcu_nocbs=4-7` | Move RCU callbacks off isolated cores |
| `systemd.unit=inference.target` | Boot to custom systemd target |
| `nvidia-l4t-bootloader.secure-boot=1` | Jetson: enable verified boot chain |

```bash
cat /proc/cmdline                         # inspect active boot parameters
cat /sys/devices/system/cpu/isolated      # verify isolcpus took effect
```

The CPU isolation triple — `isolcpus`, `nohz_full`, `rcu_nocbs` — works together as a unit:

1. **`isolcpus=4-7`**: removes cores 4–7 from the general scheduler pool. No normal process will be scheduled there unless explicitly pinned.
2. **`nohz_full=4-7`**: stops the 250 Hz scheduler tick interrupt on those cores. Without this, the tick fires 250 times per second even on isolated cores, causing jitter.
3. **`rcu_nocbs=4-7`**: moves RCU (Read-Copy-Update) grace-period callbacks off isolated cores. Without this, occasional RCU work fires on the isolated core.

All three together give inference threads essentially interrupt-free CPU time.

> **Key Insight:** `isolcpus` alone is not sufficient for real-time isolation. Without `nohz_full`, the scheduler tick fires 250 Hz on the isolated core, adding 4 ms periodic jitter. Without `rcu_nocbs`, RCU callbacks fire occasionally — typically a few microseconds, but at unpredictable intervals. The combination of all three eliminates the kernel's involuntary intrusions into the isolated core's execution.

> **Common Pitfall:** Changes to the kernel command line on Jetson require modifying `extlinux.conf` (for CBoot/extlinux) or the UEFI boot variables (for JetPack 6 UEFI). Editing `/proc/cmdline` has no effect — it is read-only. After changing `extlinux.conf`, verify with `cat /proc/cmdline` after reboot that the parameters actually took effect.

---

## Summary

| Boot stage | Responsible | Artifact loaded | Jetson equivalent |
|---|---|---|---|
| BootROM | SoC mask ROM | Signed first-stage loader | Jetson BootROM → MB1 |
| SPL / BL2 | U-Boot SPL / TF-A | DRAM init; TF-A BL31 | MB1 → MB2 |
| Bootloader | U-Boot / CBoot | Kernel + DTB + initramfs | CBoot (JetPack 5), UEFI (JetPack 6) |
| Kernel entry | `head.S` → `start_kernel()` | Subsystem init | L4T kernel image |
| Early userspace | initramfs `/init` | Mount real rootfs | NVIDIA TNSPEC + `switch_root` |
| PID 1 | systemd | All services, udev | Jetson systemd inference target |

### Conceptual Review

- **Why does the boot sequence have so many stages instead of jumping directly from ROM to kernel?** Each stage initializes hardware (DRAM, clocks, secure monitor) that the next stage depends on. ROM code is tiny and cannot initialize DRAM; U-Boot SPL initializes DRAM and then loads the full bootloader; the full bootloader has enough memory to read the kernel and DTB from storage. Each stage does the minimum needed to enable the next.
- **What is the Device Tree and why was it needed?** The Device Tree is a data file describing what hardware is present on a board — memory addresses, bus connections, IRQ numbers, clock sources. Before Device Trees, this information was hardcoded as `#ifdef BOARD_X` in kernel source. A single kernel binary can now support thousands of different boards because the hardware description is external data, not compiled-in code.
- **What is the `compatible` property in a Device Tree node?** It is a string (or list of strings) that the kernel uses to match a device node to a driver. The kernel compares the DT node's `compatible` value against every registered driver's `of_match_table`. When a match is found, the driver's `probe()` function is called with a pointer to the device node.
- **What does DKMS do and when is it necessary?** DKMS rebuilds out-of-tree kernel modules when the running kernel is updated. It is necessary for any driver that is not part of the upstream kernel source — NVIDIA's proprietary GPU driver, custom FPGA PCIe drivers, vendor-specific sensor drivers. Without DKMS, a `apt upgrade` that updates the kernel breaks these drivers until manually rebuilt.
- **What is the effect of `isolcpus` in the kernel command line?** It removes specified CPU cores from the general scheduler pool. Normal processes will not be scheduled on these cores. Inference threads pinned to the isolated cores run without interference from other processes. This is the foundation of hard real-time isolation on multi-core embedded AI platforms.
- **Why is initramfs needed before mounting the real root filesystem?** The real root filesystem may be encrypted (LUKS), on a RAID array, or on a device whose driver is not built into the kernel. initramfs provides a minimal environment with the tools needed to set up the real root (cryptsetup, mdadm, fsck) before `switch_root` transfers execution to the permanent root.

---

## AI Hardware Connection

- U-Boot for Zynq/MPSoC loads the FPGA bitstream (BOOT.BIN) before the kernel starts — the PL (Programmable Logic) is configured and ready when the kernel's Device Tree describes the AXI inference accelerator nodes, eliminating a firmware-load delay from the boot path.
- Device Tree `compatible` strings are the binding contract between Jetson camera sensor drivers and hardware; changing `sony,imx477` to `sony,imx219` in the DTBO selects a different V4L2 subdevice configuration, affecting every frame captured by `camerad`.
- DTBO overlays for IMX477 on Jetson AGX Orin allow runtime CSI lane and pin mux configuration without reflashing the base DTB — essential for carrier board bring-up and multi-sensor robot arm payloads.
- Secure boot chain verification (BootROM → MB1 → CBoot → kernel) is mandatory for production AV deployment; an unverified kernel image breaks the safety argument for ISO 26262 ASIL compliance and opens the platform to persistent rootkit attacks.
- DKMS manages the NVIDIA proprietary GPU driver on development hosts across kernel point-release updates — without it, every kernel update would break CUDA initialization and TensorRT engine builds.
- `isolcpus=4-7 nohz_full=4-7 rcu_nocbs=4-7` in the Jetson kernel command line reserves the big-cluster Cortex-A78AE cores exclusively for `modeld`, `camerad`, and `controlsd` before any userspace process starts, forming the foundation of real-time inference CPU isolation.

### Real example in openpilot (this repo)

Lecture-05 topics (boot, Device Tree, kernel modules, kernel cmdline) live in the **platform** (Agnos on comma devices, L4T on Jetson), not in the openpilot application source. On comma devices, that platform is **AGNOS** — **forked and custom-modified Linux** built for openpilot on the road; the **development changes** for boot chain, device tree, and kernel modules are in [agnos-kernel-sdm845](https://github.com/commaai/agnos-kernel-sdm845) and [agnos-builder](https://github.com/commaai/agnos-builder) (see Phase 5 — [AGNOS Guide — Development Changes](../../../Phase%205%20-%20Advanced%20Topics%20and%20Specialization/5.%20Autonomous%20Vehicles/2.%20openpilot%20Reference%20Stack/agnos/Guide.md#os-lectures--agnos-development-changes-fork--custom-mods-for-openpilot)). The openpilot repo in this roadmap still ties in as follows:

| Lecture-05 concept | How it connects to openpilot |
|--------------------|-----------------------------|
| **Kernel command line** (`isolcpus`, `nohz_full`, `rcu_nocbs`) | Set in the platform’s boot config (e.g. `extlinux.conf` on Jetson). Those parameters reserve cores so that when openpilot runs, only the intended processes use them. |
| **CPU isolation → userspace pinning** | Once the kernel has isolated cores (cmdline), openpilot pins threads to those cores. **`openpilot/common/util.cc`**: `set_core_affinity(std::vector<int> cores)` uses `sched_setaffinity(tid, ...)` so `modeld`/`camerad`/`controlsd` run only on the isolated cores — the **userspace half** of the isolation described in this lecture. |
| **Device Tree / DTBO** | Camera sensor nodes (e.g. IMX477) and CSI are described in the platform DTB/overlays. openpilot’s `camerad` talks to the V4L2 devices that those DT nodes bring up; no DTB sources are in the openpilot app repo. |
| **Boot chain / secure boot** | On comma hardware, Agnos implements the verified boot chain; on Jetson, L4T/CBoot do. openpilot assumes a correctly booted kernel and rootfs. |
| **Kernel modules** | Camera, CAN, and GPU drivers are loaded by the platform (udev/modprobe from the OS image). openpilot does not ship kernel modules. |

For concrete examples of DTB, `extlinux.conf`, and `isolcpus` on a Jetson-style stack, see the Phase 4 Track B (Jetson/Orin) guides (e.g. Orin-Nano-Real-Time-Inference, Orin-Nano-Security, Orin-Nano-Yocto-BSP-Production). For the matching userspace scheduling and affinity code, see **Lecture-06** and **`openpilot/common/util.cc`** (`set_realtime_priority`, `set_core_affinity`).
