# Deploy a matched BSP and prove what booted

**Scope:** Orin NX/Nano on a release-matched Jetson Linux BSP; current references are R39.2.1. Hardware build/flash commands below are **documented procedures, not executed board results**. The linked CPU lab has separate native execution evidence.

A deployment is a set of mutually compatible artifacts. Treating `Image`, a DTB, and a rootfs as unrelated files makes failures difficult to reproduce. This lesson follows a change from source through installation and boot, with a concrete release record.

## 1. Define the target before the command

Write down module SKU, carrier revision, storage devices, Jetson Linux release, boot-security state, rootfs redundancy policy, and recovery connection. The commercial Nano 8 GB module and developer-kit Nano 8 GB module differ; “8 GB Orin” does not identify the flash target. Board names select BSP configuration files, and storage arguments refer to the selected workflow's target devices.

Do not infer a Jetson device's NVMe name from the host's NVMe name. A host-direct flashing option changes the namespace again. Match capacity, physical connector, model/serial and the documented command before any write.

For a learning project, use a spare device/media with serial recovery. Keep destructive flash commands in a reviewed, target-specific script rather than an unqualified command copied into generic course notes.

## 2. Capture a release manifest

A small manifest can start as:

```text
product: lab-carrier-a
module: P3767-0003
carrier_revision: A
jetson_linux: 39.2.1
jetpack: 7.2.1
rootfs_release: Ubuntu 24.04
source_tag: <exact release tag from release notes>
source_patch_commit: <your commit>
board_configuration: <your board .conf>
partition_configuration: <exact XML and hash>
kernel_config_sha256: <hash>
Image_sha256: <hash>
dtb_sha256: <hash>
rootfs_package_manifest_sha256: <hash>
boot_artifact_manifest_sha256: <hash>
flash_log: <retained log>
```

This is a template, not a qualified board configuration. Record generated pinmux/GPIO/pad-voltage inputs, overlays, firmware, toolchain version and build environment too. Store signing-key identifiers, not private keys, in the record. A hash detects a changed artifact; authenticity additionally requires a trusted signature or authenticated distribution path.

Package pinning requires an update process. A permanently frozen vulnerable package is not the goal. Produce a new qualified manifest when changing dependencies and test the migration from the previously deployed state.

## 3. Build the kernel, modules, DTBs and initramfs together

R35's `sources/` trees and `nvbuild.sh` examples do not describe every current tree. For R39.2.1, NVIDIA documents `Linux_for_Tegra/source`, `kernel/kernel-noble`, separate out-of-tree module builds and a DTB output under `build/nvidia-public/devicetree/generic-dtbs/`. Even R39.2 and R39.2.1 documentation differ in the stated DTB output location. Follow the exact release and inspect actual build outputs. [R39.2.1 kernel customization](https://docs.nvidia.com/jetson/archives/r39.2.1/DeveloperGuide/SD/Kernel/KernelCustomization.html).

The conceptual build sequence is:

```text
matched source tag + patch set + toolchain
    -> configured kernel and in-tree modules
    -> NVIDIA out-of-tree modules against that kernel
    -> device trees against that source/build context
    -> install into the staged rootfs and BSP
    -> regenerate initramfs
    -> create/sign the selected boot and flash artifacts
```

A module must match the target kernel's ABI/configuration and relevant symbol versions. Host `uname -r` describes the host; it must not choose an arm64 target module directory during a cross-build. An initramfs may contain an older driver even when the rootfs copy is new. Keep both in the release audit.

**Worked failure:** the kernel starts, then cannot mount NVMe root. Check whether the storage/PCIe/filesystem drivers needed before root mount are built in or present in the selected initramfs, and whether the command line selects the expected PARTUUID. Adding a module only to the inaccessible rootfs cannot fix an early mount dependency.

## 4. Prepare the rootfs deliberately

Use one exact sample-rootfs archive or a pinned generated rootfs. Apply the matching NVIDIA integration and product package changes in a reproducible staging flow. Preserve permissions, ownership, symlinks and capabilities as required by the chosen packaging method. A cross-architecture `chroot` needs its documented emulation/binfmt setup and correct mounted virtual filesystems; changing directories alone does not make arm64 binaries execute on x86.

For archiving, match compression and suffix: `tar -cJf rootfs.tar.xz` creates xz, whereas `tar -cjf rootfs.tar.bz2` creates bzip2. Renaming an xz file to `.tbz2` does not convert it. A rootfs archive still needs the BSP's image-generation and partition workflow.

Choose unique device identity at provisioning or first boot. Cloning one live system can duplicate host keys, machine identity, network configuration or application credentials. Validate the resulting service/package inventory rather than assuming a smaller image remains functional.

## 5. Generate before writing

NVIDIA's `--no-flash` modes prepare artifacts without the final physical flash operation. They are **not pure syntax checkers**: image generation writes host files and can require a recovery-connected board or accurate board/chip metadata. Do not invent EEPROM/RAM-code values to make a command run. The specific help text and `tools/kernel_flash/README_initrd_flash.txt` in the selected BSP are part of the procedure. [Flashing support](https://docs.nvidia.com/jetson/archives/r39.2.1/DeveloperGuide/SD/FlashingSupport.html).

Before the approved hardware write, inspect the generated flash log and partition inputs. Confirm that your custom source produced the selected binaries. A stale prebuilt DTB can survive in a staging directory and override the file you just compiled.

For initrd flashing, also prove the recovery kernel can access the custom carrier's storage and USB/network path. A product kernel that works after boot cannot compensate for a recovery environment missing the required controller configuration.

## 6. Understand partition capacity

Use integers and the units declared by the partition configuration. A teaching example with 512-byte sectors and 62,500,000 sectors has:

```text
capacity = 512 × 62,500,000 = 32,000,000,000 bytes
         ≈ 29.8023 GiB
```

A 28 GiB rootfs partition occupies 30,064,771,072 bytes, leaving about 1.80 GiB for **all** remaining space. It is not valid to add a second 28 GiB slot to this layout. GPT metadata, alignment, boot partitions and reserved areas consume space too. These numbers are arithmetic examples, not a supplied NVIDIA partition XML.

A sparse image's host file size is not the expanded target requirement. A file that occupies 3 GiB on the host may describe a larger filesystem. Compare declared partition size, expanded filesystem/image geometry and target capacity. Preserve adequate growth space for the update policy.

## 7. Inspect the booted system

These are read-only **target** observations, to run after a successful board boot:

```bash
cat /etc/nv_tegra_release
uname -a
cat /etc/os-release
cat /proc/cmdline
findmnt -no SOURCE,FSTYPE,OPTIONS /
lsblk -o NAME,SIZE,MODEL,SERIAL,FSTYPE,PARTUUID,MOUNTPOINTS
tr '\0' '\n' < /sys/firmware/devicetree/base/compatible
```

Run commands individually and retain failures as evidence. A command absent from a minimal rootfs should not be silently ignored. Add driver-specific observations: loaded module identity, expected probe messages and a known peripheral transaction. The live device tree can differ from the build input because firmware applies fixups and overlays.

An application health check should include its required sensor/model/configuration contract. “Process exists” or “GPU enumerates” does not establish correct inference. Keep deterministic input/output checks separate from throughput and thermal tests.

## 8. Diagnose by the last stage that worked

| Last evidence | Next investigation |
| --- | --- |
| No recovery USB enumeration | Power, cable, recovery straps/buttons and physical USB path |
| Early firmware log then stop | Module identity, BCT/DRAM/firmware compatibility, security/signing state |
| UEFI runs but wrong boot option | Boot order, selected storage, partition identities and installed artifacts |
| Kernel starts but no root mount | Command line, storage drivers, initramfs, filesystem and partition identity |
| Rootfs mounts but driver missing | DT selection, compatible/binding, module availability and dependencies |
| Driver probes but data wrong | Pinmux, rails, reset, clock, transaction format and physical signal |
| Works cold but fails under load | Power/thermal behavior, concurrency, buffers, timing and actual workload |

Change one hypothesis at a time and retain the original failure trace. A new random collection of DT and register edits hides which assumption was wrong.

## 9. Finish the release exercise

Use the [native device-tree lab](../1.%20Nvidia%20Jetson%20Platform/Orin-Nano-Kernel-Internals/labs/README.md) to submit source and output identities plus a valid and invalid overlay. For a physical board, add the exact flash invocation, observed boot configuration, one peripheral contract test, recovery procedure and update/rollback evidence.

**Answers:** a rootfs-only change need not rebuild a kernel unless its dependencies changed; a kernel change can require modules and initramfs regeneration; a BCT change belongs in the boot artifact flow; an edited file without selection/boot evidence proves only that the file was edited. Continue with the [engineering flow](Orin-Nano-8GB-Custom-Board-L4T-Engineering-Flow.md).
