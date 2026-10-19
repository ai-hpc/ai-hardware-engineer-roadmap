# Lecture 5: Boot, Device Tree, and Kernel Modules

Before a perception process can open a camera, firmware must initialize the board, Linux must start, and a matching driver must bind successfully. This lesson teaches the **handoff contracts** between those steps. The CPU-only [Device Tree and module lab](../../../Phase%204%20-%20Track%20B%20-%20Nvidia%20Jetson/1.%20Nvidia%20Jetson%20Platform/Orin-Nano-Kernel-Internals/labs/README.md) lets you inspect real compiled artifacts without loading a module or touching hardware.

## There is no single ARM boot chain

```text
SoC reset / BootROM
    → board-specific initialization and firmware stages
    → boot manager / bootloader
    → selected kernel + hardware description + optional initramfs
    → kernel initialization
    → early userspace or final rootfs init
    → services and application
```

Some systems use U-Boot SPL, TF-A, and U-Boot proper. Jetson has NVIDIA-specific MB1/MB2 and firmware components. These are not interchangeable filenames or privilege levels. JetPack 5 and later Jetson releases use UEFI in the relevant bootloader role; describing JetPack 5 as CBoot is obsolete. NVIDIA documents the transition in its [UEFI adaptation guide](https://docs.nvidia.com/jetson/archives/r36.4.4/DeveloperGuide/SD/Bootloader/UEFI.html).

An ARM64 `Image` is normally uncompressed. If a compressed image is used, a supported earlier stage must decompress it; do not assume ARM64 `head.S` universally inflates `Image.gz`. The boot protocol constrains register state, memory placement, and the Device Tree handoff. ACPI-based platforms have a different hardware-description path.

The kernel's initialization code sets up memory management, scheduling, buses, and configured drivers. Built-in drivers can probe before userspace; loadable modules may appear later. Deferred probe permits a driver to wait for dependencies such as clocks or regulators. “The compatible string matches” is necessary in many DT paths but does not establish a working device.

## Secure boot is a configured trust policy

Verified boot requires an actual chain: a trusted stage validates an authorized next-stage artifact before handing control to it. Which stages, keys, algorithms, rollback data, and rootfs mechanisms are involved depends on the SoC/BSP and product configuration. Do not infer that an unfused development board authenticates every image or that one kernel command-line flag enables the hardware trust chain.

Kernel/module signing, rootfs integrity, confidential data encryption, and rollback prevention are separate mechanisms. `dm-verity` verifies blocks against an authenticated root digest; it does not encrypt them. A valid signature says who authorized particular bytes, not that those bytes implement safe control behavior. Functional-safety compliance does not follow automatically from secure boot.

Key provisioning can be irreversible on real devices. This foundation exercise therefore inspects artifacts; board-specific provisioning belongs in the platform's explicitly planned deployment procedure. Preserve release and recovery identities when changing firmware.

## initramfs: userspace before the final root

An initramfs supplies an early filesystem archive, often containing an `/init` program, drivers, and tools to discover or unlock the final rootfs. If that `/init` is executed, it normally mounts the final root and transitions to it. Some systems can boot without an external initramfs because required support is built in.

Compression and archive layout vary; an installed image may contain concatenated archives or an uncompressed early part. Do not assume `zcat /boot/initrd.img | cpio -idm` works for every distribution. Use the distribution's listing tool on an explicitly chosen file, or inspect the supplied build artifacts in a disposable directory. Reading archive contents does not verify that those scripts boot a target board.

## Device Tree is data with a binding

A Device Tree describes devices and their connections. The interpretation comes from the binding and driver. `reg` under an I2C controller is commonly a bus address; `reg` under an MMIO bus is encoded using the parent's `#address-cells` and `#size-cells`. `ranges` can translate child addresses into parent space. An interrupt specifier belongs to its interrupt-controller binding, not a universal list of CPU interrupt numbers.

| Property | What to check |
|---|---|
| `compatible` | Exact binding and driver's match table |
| `reg` | Parent cell widths, address translation, resource size |
| `interrupts` | Interrupt parent, cells, source and trigger type |
| `clocks` / `clock-names` | Existing provider references and names the driver requests |
| `resets`, supplies, power domains | Dependencies and required startup order |
| `status` | Missing or `"okay"`/`"ok"` is normally available; `"disabled"` is unavailable |
| Graph endpoints | Correct reciprocal links and actual bus/format constraints |

The previous common misconception—“missing status means disabled”—is false for the normal OF availability helper. An enabled node still needs a supported parent/bus, matching driver, dependencies, and successful probe.

**Worked encoding:** with two address cells and two size cells, `reg = <0x0 0x12340000 0x0 0x1000>` describes an address value `0x12340000` and a 4096-byte resource in that bus's address space. It does not prove that this address exists on your board. With one address cell and one size cell, the same four-cell property would have a different interpretation.

A camera sensor node may register a V4L2 subdevice; it does not necessarily create a complete `/dev/videoN` capture path by itself. CSI receivers, media graph endpoints, formats, clocks, pinmux, regulators, and ISP/application paths must agree. A synthetic 20-line overlay is not a universal carrier-board camera integration.

## Compile and inspect before deployment

The native lab contains an actual base tree, an overlay, and assertions using `dtc`, `fdtoverlay`, and `fdtget`. It checks address/size interpretation, overlay application, malformed input, and missing-target behavior. Use that fixture before substituting a real board's addresses.

```text
DTS source + includes/binding macros
    → preprocessing when required
    → dtc → DTB
base DTB + compatible overlay → fdtoverlay → merged DTB
merged DTB → inspect properties and validate applicable schemas
```

Parsing successfully does not prove schema validity, electrical correctness, or driver compatibility. A DTS source using C-style binding headers needs the kernel's preprocessing/build flow; invoking standalone `dtc` on arbitrary kernel sources can fail legitimately.

Runtime overlay support and bootloader overlay support are separate. A configfs overlay directory is not universally supplied by mainline Linux. Removing an overlay while driver callbacks still hold pointers into it can cause lifetime bugs; see the [kernel overlay notes](https://docs.kernel.org/devicetree/overlay-notes.html). Use the platform-supported boot flow for real hardware.

## Kernel modules and a build-only milestone

A `.ko` is kernel code built for a target kernel/configuration. `insmod` loads a specified object; `modprobe` also resolves installed module aliases/dependencies. Built-in drivers do not appear as loadable modules. `MODULE_DEVICE_TABLE` supplies alias metadata used by module tools; it does not itself initialize a device.

A `platform_driver` has match information and lifecycle callbacks. Their exact signatures change across kernel versions. Compile against the target headers rather than copying a `.remove` callback from an unrelated BSP. Module signing enforcement depends on the kernel's configuration and boot policy; not every production or Jetson kernel enforces the same settings.

The linked lab actually built a simple module against recorded Linux 6.8 headers **without loading it**. That validates source/header compilation, not a probe, IRQ, DMA, or removal path. Before a real driver can be unloaded, its work and hardware access must be stopped and all live references handled; [Lecture 17](Lecture-17.md) develops that protocol.

DKMS can automate rebuilds when appropriately packaged and configured. Inspect rebuild logs and the **new target kernel's** headers before booting it. Running `uname -r` before a reboot names the currently running kernel, which may be the old one; it is not necessarily the header version that a pending update needs.

## Boot parameters and timing

`cat /proc/cmdline` reads the active command line. Editing that file does not change future boot configuration. The selected boot manager's configuration and platform policy determine what is passed.

`isolcpus`, `nohz_full`, and `rcu_nocbs` address particular scheduling/tick/RCU behaviors, with version- and configuration-specific rules. They do not produce interrupt-free execution or remove shared-resource contention. A 250 Hz tick has a 4 ms period; that does **not** mean each tick stalls the CPU for 4 ms. Start with the [tested RT demo](demos/rt-demo/README.md) and [isolation lesson](Lecture-08.md) before tuning a board.

**Check:** A DTB compiles and a module builds. Is the device validated? **Answer:** No. Matching, resource setup, actual hardware behavior, failure recovery, and teardown still need target tests.

**Check:** Why inspect a merged DTB rather than only an overlay source? **Answer:** Parent cells, labels, prior values, and applied changes determine the final resource interpretation.
