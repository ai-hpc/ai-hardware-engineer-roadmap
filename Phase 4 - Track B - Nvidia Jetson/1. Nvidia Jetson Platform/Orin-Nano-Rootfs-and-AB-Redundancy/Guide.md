# Root filesystems, storage layout, and A/B recovery

The root filesystem supplies the operating system's starting directory tree. A/B redundancy provides two eligible system copies so an update can be tried without immediately destroying the previous one. To use that redundancy well, you must know what belongs to a slot, what is shared, and who decides that a new boot succeeded.

**Outcomes:** inspect a mounted system; distinguish module, boot storage, and rootfs; budget two slots and update workspace; explain boot validation; and preserve data compatibility across rollback. Plan two 60–90-minute sessions. Prerequisites: basic Linux paths and the [Jetson platform introduction](../Guide.md).

**Reviewed 2026-10-18.** Choose the complete release combination in the [compatibility matrix](../../Compatibility.md). R39.2's samplefs examples use Ubuntu Noble and offer desktop, minimal, and basic flavors; minimal/basic require appropriate headless access and initial-user setup. Their installed sizes depend on packages and customization. [NVIDIA rootfs guide](https://docs.nvidia.com/jetson/archives/r39.2/DeveloperGuide/SD/RootFileSystem.html).

## 1. The visible directory tree can contain many filesystems

When Linux mounts a root filesystem at `/`, other filesystems can be mounted inside it. `/proc` and `/sys` expose kernel interfaces; `/run` commonly uses volatile storage; a separate partition may hold `/data`. Consequently, copying files visible under `/` is not automatically a consistent rootfs image or backup.

Use read-only inspection on a Linux machine:

```bash
findmnt /
findmnt /boot
findmnt /data
lsblk -o NAME,SIZE,FSTYPE,LABEL,UUID,PARTUUID,MOUNTPOINTS
df -hT
```

A nonexistent `/data` mount may produce no match; that is information, not a reason to invent a partition. These commands report the machine on which you run them. A desktop's output is not a Jetson layout.

Distinguish a filesystem **UUID** from a partition **PARTUUID**. Record which identifier the boot configuration actually uses. Copying an image can also copy its filesystem UUID, so cloning requires an explicit identity plan. A label alone may be ambiguous if two devices share it.

**Exercise:** identify the filesystem mounted at `/`, the device or virtual backing it uses, and one separate mount. Explain why measuring free space on `/` might not tell you how much space remains on `/data`.

## 2. Know where the platform pieces live

The Jetson module, carrier, QSPI firmware storage, optional developer-module microSD, and external NVMe are different pieces. Orin Nano production modules do not universally contain an eMMC rootfs. The exact module and carrier determine the available boot/storage path. [Orin Nano hardware layout](https://docs.nvidia.com/jetson/orin-nano-devkit/user-guide/latest/hardware_layout.html).

A conceptual deployment can include:

| Artifact | Role | Question to answer |
| --- | --- | --- |
| Platform boot firmware | Initialize and select later boot stages | Which physical storage and update chain? |
| Kernel and device tree | Describe/operate the hardware | Loaded from which partition, file, or boot entry? |
| Kernel modules | Drivers built for the installed kernel | Which rootfs and exact kernel build? |
| Rootfs A and B | Alternative OS/application environments | Which bytes are protected by the slot selection? |
| Persistent data | Identity, calibration, configuration, application state | Shared or versioned, and readable after rollback? |
| Recovery environment | Diagnose or restore an unusable system | What still boots if neither normal slot works? |

Do not assume fixed partition numbers such as `nvme0n1p2` for a kernel. Match the actual partition XML, flashed layout, and running mount/boot configuration. A filename in `/boot` is not sufficient proof that firmware loaded it.

## 3. Build a coherent image

The BSP workspace typically separates bootloader assets, kernel assets, rootfs staging, and preparation/flashing tools. A sample rootfs provides a base userspace; BSP application and package installation supply additional platform components. Do not treat `apply_binaries.sh` as an installer for every CUDA, TensorRT, or application package you might need.

Create an image manifest containing the BSP/tool hashes, Ubuntu release, package versions, kernel build/configuration, module set, device tree, service configuration, and your application/model identities. Check the artifact licenses and retain the corresponding source/build information required by your distribution.

The release's `nv_build_samplefs.sh` and flashing workflow are device operations, not a generic CPU lab. Resolve the flavor, release, board, storage and layout before constructing a command. In particular, an old `--version focal` example does not become a current R39 recipe by keeping the rest of the command unchanged.

The image must also have a usable first-boot path: users, credentials/provisioning, networking, console access, time initialization, and device identity. Avoid baking one deployed device's private keys or unique identity into a fleet-wide image.

For reproducible embedded builds, continue with [Yocto BSP production](../Orin-Nano-Yocto-BSP-Production/Guide.md). “Reproducible” needs a declared level: pinned inputs, repeatable functionality, or byte-identical output. Those are different claims.

## 4. Budget storage with explicit units and headroom

Consider a hypothetical usable storage budget of **64 GiB**:

| Allocation | Size |
| --- | --- |
| Rootfs A | 16 GiB |
| Rootfs B | 16 GiB |
| Persistent data | 12 GiB |
| Download/reconstruction workspace | 8 GiB |
| Firmware, boot, recovery and other reserved allocation | 4 GiB |
| Unallocated planning reserve | 8 GiB |
| Total | 64 GiB |

This is arithmetic, not a Jetson partition template. Actual layouts have their own alignment, filesystem overhead, reserved space, and boot requirements. A marketed 64 GB drive contains about `64 × 10^9 / 2^30 = 59.60 GiB` before layout overhead, so it cannot satisfy this 64 GiB example.

Check both the compressed download and its uncompressed/reconstructed size. Account for retained images, container layers, model caches, logs, temporary files, and data growth. Peak installation space can be larger than the final installed size.

A rootfs image fitting into an empty slot does not prove that the updater has enough workspace. A package update can also need temporary space on a different mount than the final application files.

**Exercise:** if each rootfs grows by 2 GiB and all other allocations remain fixed, reserve falls from 8 to 4 GiB. Explain which two partitions grew and why counting only one new image would understate the persistent requirement.

## 5. Understand what A/B does and does not cover

An A/B design installs into an inactive copy, selects a trial, and retains a fallback until acceptance. The actual boot platform determines how firmware, kernel, rootfs, and retries are paired. Rootfs slot state and bootloader slot state must be inspected at the correct target.

On a configured Jetson, inspect rather than change the state first:

```bash
sudo nvbootctrl dump-slots-info
sudo nvbootctrl -t rootfs dump-slots-info
```

For R39.2, the rootfs guide documents a default of three failed boots and a configurable maximum retry value from zero to three. Do not copy the old seven-retry rootfs examples. The installed tool and matched documentation determine available commands; this course does not assume a `mark-boot-successful` subcommand. [Rootfs slot management](https://docs.nvidia.com/jetson/archives/r39.2/DeveloperGuide/SD/RootFileSystem.html).

A/B does not protect every shared partition, data migration, signing-key change, or security-floor advancement. Nor does it repair an incorrectly provisioned recovery path. State the failure you expect it to handle, then demonstrate that failure and recovery.

## 6. Connect boot validation to application validation

The documented flow orders customer rootfs validation before boot-status verification. The customer validation hook is absent by default, so it supplies no application check until integrated. The verification service calls `nvbootctrl verify`. Inspect the installed unit/script names and behavior for your release. [Validation flow](https://docs.nvidia.com/jetson/archives/r39.2/DeveloperGuide/SD/RootFileSystem.html).

A practical investigation begins with unit discovery and read-only inspection:

```bash
systemctl list-unit-files '*rootfs*' '*bootloader*'
# Use the exact discovered unit name below:
# systemctl cat UNIT_NAME
# systemctl status UNIT_NAME
# journalctl -b -u UNIT_NAME
```

The last three lines are inspection templates; replace `UNIT_NAME` with the actual result. Explain start ordering, timeout, exit status, retry behavior, and what happens if validation itself hangs. A background service must have a defined completion point before a test deliberately reboots the machine.

For a robot, boot acceptance may require fresh sensor data and a valid bounded inference, not merely an import succeeding. Tie results to the exact boot and deployment. The [OTA course](../Orin-Nano-OTA-Deep-Dive/Guide.md) develops that contract and its negative tests.

## 7. Choose writable and persistent storage deliberately

A read-only rootfs can reduce unintended system changes. It does not mean every application becomes stateless. Logs, caches, machine identity, calibration, databases, and runtime files still need suitable locations and lifecycle rules.

Overlayfs presents a combined view of lower and upper layers. Changes can cause copy-up and deletions can be represented by whiteouts; the lower image remains unchanged. If the upper layer persists across a new lower image, old changes can hide new content. Decide whether an upper layer is transient, versioned, or migrated, and test that policy. [Linux Overlayfs semantics](https://docs.kernel.org/filesystems/overlayfs.html).

If an upper layer is memory-backed, its writes consume memory and disappear with the appropriate reset. If it is disk-backed, budget space and endurance. Neither choice automatically creates a compatible rollback of shared application data.

Create a file-policy table with path, owner, confidentiality, integrity requirement, lifetime, size bound, backup rule, and rollback compatibility. A robot's calibration and device credentials deserve different policies from a disposable model cache.

## 8. Practice recovery without flashing a board

Run the [interrupted-update CPU lab](../Orin-Nano-OTA-Deep-Dive/labs/README.md). It preserves a confirmed copy while installing another, tests five process-exit points, exhausts failed trials, rejects stale health evidence, and checks when a version floor would remove fallback.

Map its concepts to a device worksheet:

| Lab concept | Device evidence required |
| --- | --- |
| `good` image | Actual eligible boot/rootfs state and its version/hash |
| `pending` trial | Supported platform mechanism and bounded attempt policy |
| Published image | Exact partition/files, verification and write ordering |
| Confirmation token | Application health associated with one boot/deployment |
| Fallback | Compatible firmware, rootfs, security policy, and persistent data |

The lab's file-backed algorithm is not the NVIDIA algorithm. Its counters, JSON fields, and synchronization calls cannot be assumed to describe firmware internals. It is a smaller system in which learners can inspect the recovery logic before tackling a real one.

## 9. Produce the deployment artifact

Finish with a storage diagram, exact release manifest, per-mount capacity budget, persistent-file policy, and update/recovery state table. Include one normal boot path and one failed-candidate path. State how kernel modules match the booted kernel and how the recovery application reads data left by the candidate.

For hardware completion, retain actual partition and mount output, boot logs, update logs, accepted-health evidence, and a controlled rollback result. Label an unexecuted flash/OTA recipe as a plan. CPU lab success establishes its local teaching protocol; it does not establish a bootable Jetson image.

**Self-check:** why can two intact rootfs slots still leave a robot unavailable? Shared firmware can be incompatible, a database can become unreadable to the old app, a security floor can exclude the fallback, or the recovery path can be missing. A complete design accounts for each relevant dependency.
