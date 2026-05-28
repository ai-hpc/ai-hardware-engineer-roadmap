# Orin Nano 8GB — Root File System & A/B Redundancy

<div class="course-identity auto-course" style="--course-accent: #16a34a; --course-accent-rgb: 22, 163, 74;" markdown="1">
<div class="course-identity__icon">ON8R</div>
<div markdown="1">
<p class="course-identity__eyebrow">Deep Dive · Jetson Track</p>
<p class="course-identity__title">Specialized course identity for Orin Nano 8GB — Root File System & A/B Redundancy.</p>
<p class="course-identity__meta">Artifact: Jetson integration demo · Measure: latency, memory, power, logs</p>
</div>
</div>


> **Scope:** Production-level understanding of Jetson rootfs architecture, the three rootfs flavors, A/B slot redundancy, OTA update strategy, boot validation, partition layout, and how rootfs connects to the memory and boot chain architecture.
>
> **Prerequisites:** Familiarity with the [Orin Nano boot chain](../Guide.md#1-orin-nano-8gb--hardware--boot-chain-internals) and [memory architecture](../Orin-Nano-Memory-Architecture/Guide.md).

---


## 1. L4T Root File System Overview

On Jetson, the root file system (rootfs) is the entire Linux filesystem tree:

```
/
├── bin        ← core binaries
├── lib        ← shared libraries (including NVIDIA drivers)
├── etc        ← system configuration
├── usr        ← user programs, CUDA toolkit, TensorRT
├── opt        ← NVIDIA-specific tools
├── home       ← user data
└── var        ← logs, runtime data
```

Jetson Linux (L4T — Linux for Tegra) is:

* Based on Ubuntu (20.04 focal for JetPack 5.x, 22.04 jammy for JetPack 6.x)
* Customized with NVIDIA BSP (Board Support Package)
* Includes CUDA, cuDNN, TensorRT, multimedia stack, and Jetson-specific drivers
* Kernel is NVIDIA-patched (not mainline)

The rootfs is what gets written to the APP partition (NVMe on Dev Kit, eMMC on production modules).

---

## 2. Rootfs Generation and BSP Structure

### BSP Directory Layout

After extracting the L4T BSP, the directory structure is:

```
Linux_for_Tegra/
├── bootloader/          ← MB1, MB2, UEFI, firmware blobs
├── kernel/              ← Image, DTB, modules
├── rootfs/              ← Ubuntu root filesystem (initially empty)
├── tools/
│   └── samplefs/
│       └── nv_build_samplefs.sh   ← rootfs generator
├── nv_tegra/            ← NVIDIA drivers, configs
├── flash.sh             ← flashing script
└── apply_binaries.sh    ← applies NVIDIA binaries to rootfs
```

### Building a Rootfs

```bash
# Step 1: Generate base Ubuntu rootfs
cd Linux_for_Tegra/tools/samplefs/
sudo ./nv_build_samplefs.sh --abi aarch64 --distro ubuntu --version focal

# Step 2: Copy to rootfs directory
sudo cp <generated>.tbz2 ../../rootfs/
cd ../../rootfs/
sudo tar xpf <generated>.tbz2

# Step 3: Apply NVIDIA binaries (drivers, CUDA, etc.)
cd ..
sudo ./apply_binaries.sh

# Step 4: Flash
sudo ./flash.sh jetson-orin-nano-devkit internal
```

`apply_binaries.sh` installs:

* `nvgpu` kernel module
* CUDA runtime libraries
* Multimedia libraries (nvbufsurface, nvargus, etc.)
* Jetson-specific systemd services
* Device tree overlays

---

## 3. The Three Rootfs Flavors

NVIDIA provides three rootfs configurations. Choosing the right one is a critical production decision.

### Desktop

* Full Ubuntu GUI (GNOME/GDM3)
* OEM setup wizard on first boot
* Includes: desktop apps, browser, text editor, file manager
* Size: ~4–6GB
* Use case: development boards, prototyping, demos

### Minimal

* No GUI — SSH / UART access only
* Core system utilities
* Smaller footprint (~1.5–2GB)
* Use case: robotics, industrial vision, production edge systems

### Basic

* Smallest footprint (~800MB–1.2GB)
* Includes Docker/container runtime dependencies
* Designed for container-based deployment (all apps run in containers)
* Use case: cloud-native edge AI, fleet-managed devices

### Which to Use in Production

| Scenario                    | Recommended    | Why                                      |
|-----------------------------|----------------|------------------------------------------|
| Development / prototyping   | Desktop        | Easy setup, GUI debugging tools          |
| Single-purpose AI device    | Minimal        | Small, predictable, low attack surface   |
| Fleet of managed devices    | Basic + Docker | Containerized updates, reproducible      |
| Autonomous robot/vehicle    | Minimal        | Tight control, custom services only      |

Production AI edge systems almost always use **minimal or basic**. Desktop rootfs wastes storage, memory, and increases attack surface.

---

## 4. Rootfs Customization for Production

### Removing Unnecessary Packages

```bash
# After generating rootfs, chroot into it
sudo mount --bind /dev rootfs/dev
sudo mount --bind /proc rootfs/proc
sudo mount --bind /sys rootfs/sys
sudo chroot rootfs

# Remove packages
apt remove --purge snapd thunderbird libreoffice-*
apt autoremove

# Exit chroot
exit
sudo umount rootfs/{dev,proc,sys}
```

### Adding Production Packages

Common additions for edge AI systems:

```bash
# Inside chroot
apt install -y \
    openssh-server \
    python3-pip \
    docker.io \
    chrony \           # time sync (critical for sensor fusion)
    watchdog \         # hardware watchdog
    logrotate          # prevent log storage exhaustion
```

### Read-Only Rootfs

For maximum reliability, make rootfs read-only with an overlay for writable data:

```
/dev/nvme0n1p1 (APP)  → mount read-only at /
tmpfs                 → overlay for /tmp, /var/run
/dev/nvme0n1p2 (data) → mount read-write at /data
```

Benefits:

* Survives unexpected power loss without filesystem corruption
* Prevents log accumulation from filling storage
* Forces clean separation of system vs. application data

---

## 5. Partition Layout on Orin Nano

### Default Layout (No A/B)

```
QSPI NOR Flash:
  mb1, mb2, uefi, firmware blobs, BCT, boot config

NVMe / eMMC:
  APP          → rootfs (ext4)
  [optional]   → data partition
```

### Layout With A/B Enabled

```
QSPI NOR Flash:
  mb1       mb1_b
  mb2       mb2_b
  uefi      uefi_b
  BCT       BCT_b
  (each firmware blob duplicated)

NVMe / eMMC:
  APP       → rootfs A (ext4)
  APP_b     → rootfs B (ext4)
  kernel    → kernel A
  kernel_b  → kernel B
  kernel-dtb   → DTB A
  kernel-dtb_b → DTB B
```

Every component is duplicated — bootloader, kernel, DTB, and rootfs. This ensures a complete fallback if any single component is corrupted.

### Viewing Current Partition Layout

```bash
# On a running Jetson
lsblk
sudo fdisk -l /dev/nvme0n1

# Or check flash layout XML before flashing
cat Linux_for_Tegra/bootloader/generic/cfg/flash_t234_qspi.xml
```

---

## 6. A/B Rootfs Redundancy

### Why It Exists

Imagine deploying hundreds of AI edge devices in the field — traffic cameras, industrial inspection systems, autonomous robots. If an OTA update breaks the system and there is no redundancy:

* Device is bricked
* Physical access required to reflash
* Downtime, truck rolls, customer impact

With A/B redundancy:

* Update is written to the **inactive slot**
* System reboots into the updated slot
* If boot succeeds, the slot is marked good
* If boot fails, **automatic rollback** to the previous slot

This is the same approach used in automotive ECUs (ISO 26262), Android devices, and industrial PLCs.

### How Slots Are Paired

Each slot is a complete bootable system:

```
Slot A:                           Slot B:
  Bootloader A (MB1, MB2, UEFI)    Bootloader B
  Kernel A                          Kernel B
  DTB A                             DTB B
  Rootfs A (APP)                    Rootfs B (APP_b)
```

The pairing is critical — you cannot mix Slot A bootloader with Slot B rootfs. This would cause driver mismatches, module load failures, and potentially a non-booting system.

### Enabling A/B

Set in the flash configuration before flashing:

```bash
# In flash environment
export ROOTFS_AB=1
sudo ./flash.sh jetson-orin-nano-devkit internal
```

This cannot be enabled after flashing — the partition table must be created with dual slots from the start.

---

## 7. Boot Flow With A/B Enabled

### Normal Boot (Slot A Active)

```
Power On
 ↓
BootROM (reads fuses, selects boot media)
 ↓
MB1 (Slot A — DRAM training, power rails)
 ↓
MB2 (Slot A — loads UEFI A)
 ↓
UEFI (Slot A — checks slot state, loads kernel A)
 ↓
Kernel A (from kernel partition A)
 ↓
Mounts rootfs A (APP partition)
 ↓
systemd → validation services → mark slot A successful
```

### Boot Failure Recovery

If Slot A fails to boot (kernel panic, hang, validation failure):

```
Boot attempt 1 → Slot A → FAIL
Boot attempt 2 → Slot A → FAIL
Boot attempt 3 → Slot A → FAIL (retry count exhausted)
 ↓
cpu-bootloader marks Slot A invalid
 ↓
Switch to Slot B
 ↓
MB1 B → MB2 B → UEFI B → Kernel B → Rootfs B
 ↓
System boots from Slot B (rollback complete)
```

### If Both Slots Fail

If both Slot A and Slot B fail to boot:

```
Both slots exhausted
 ↓
Recovery kernel (if configured)
 ↓
Minimal recovery environment
 ↓
Requires manual intervention (reflash via USB)
```

This is why production systems must have robust validation — you want to catch failures in Slot A and roll back to a known-good Slot B, not exhaust both slots.

---

## 8. Slot States and nvbootctrl

### Slot Attributes

Each slot has four attributes:

| Attribute     | Meaning                                           |
|---------------|---------------------------------------------------|
| **Active**    | The slot UEFI will boot on next reboot            |
| **Current**   | The slot currently running                        |
| **Bootable**  | The slot contains a valid OS image                |
| **Retry count** | Remaining boot attempts before marking failed  |

### nvbootctrl Commands

```bash
# Dump current slot information
sudo nvbootctrl -t rootfs dump-slots-info

# Example output:
# Current slot: A
# Slot A:
#   Priority: 15
#   Suffix: _a
#   Retry count: 7
#   Boot successful: 1
# Slot B:
#   Priority: 14
#   Suffix: _b
#   Retry count: 7
#   Boot successful: 1

# Check which slot is currently active
sudo nvbootctrl -t rootfs get-current-slot

# Set Slot B as the next boot target
sudo nvbootctrl -t rootfs set-active-boot-slot 1

# Mark current slot as successfully booted
sudo nvbootctrl -t rootfs mark-boot-successful
```

The `-t rootfs` flag targets the rootfs partition table. Without it, `nvbootctrl` operates on the bootloader partition table.

---

## 9. Boot Validation Services

Two critical systemd services run after boot to validate the slot. Understanding these prevents a very common production mistake.

### nv-l4tbootloader-config.service

* Runs early in the boot process
* Calls `nvbootctrl verify` to validate bootloader integrity
* Marks the slot as successfully booted if verification passes

### l4t-rootfs-validation-config.service

* Custom validation hook — you add your own health checks here
* Default: minimal checks (filesystem mounted, basic services running)
* Production: add application-specific checks (inference engine starts, camera opens, network connects)

### The Common Production Mistake

If you reboot before validation services complete:

1. System boots into Slot A
2. Validation service starts but hasn't finished
3. You reboot (manual or watchdog)
4. Retry count decrements
5. After 3 reboots, system marks Slot A as failed
6. Switches to Slot B (unintended rollback)

Prevention:

* Ensure validation services complete before any reboot
* Set retry count high enough for your boot time
* Add explicit health checks before calling `mark-boot-successful`

### Custom Validation Example

Create a production validation script:

```bash
#!/bin/bash
# /opt/nvidia/validation/validate.sh

# Check CUDA is functional
nvidia-smi > /dev/null 2>&1 || exit 1

# Check camera opens
v4l2-ctl --device=/dev/video0 --all > /dev/null 2>&1 || exit 1

# Check inference engine
python3 -c "import tensorrt" > /dev/null 2>&1 || exit 1

# Check network connectivity
ping -c 1 -W 5 8.8.8.8 > /dev/null 2>&1 || exit 1

# All checks passed — mark boot successful
nvbootctrl -t rootfs mark-boot-successful
```

Wire this into `l4t-rootfs-validation-config.service` so the slot is only marked good when your application stack is confirmed working.

---

## 10. Partition Size With A/B Enabled

When A/B is enabled:

```
ROOTFS_AB=1
```

The rootfs partition is split in half:

```
ROOTFSSIZE / 2
```

Example:

| ROOTFSSIZE | APP (Slot A) | APP_b (Slot B) |
|------------|--------------|----------------|
| 28 GiB     | 14 GiB       | 14 GiB         |
| 56 GiB     | 28 GiB       | 28 GiB         |
| 128 GiB    | 64 GiB       | 64 GiB         |

### Storage Planning Mistakes

If you forget that A/B halves your rootfs:

* Rootfs fills up during operation
* Docker images fail to pull
* OTA updates fail (no space for new image)
* Log files exhaust remaining space
* System becomes unstable

### Production Storage Budget (14 GiB Slot Example)

| Component                     | Size      |
|-------------------------------|-----------|
| Minimal rootfs (L4T + NVIDIA) | ~2 GB     |
| CUDA + TensorRT libraries     | ~2 GB     |
| Application code              | ~500 MB   |
| TensorRT engines              | ~500 MB   |
| Docker images (if used)       | ~2–4 GB   |
| Logs + temp (with rotation)   | ~1 GB     |
| **Free space buffer**         | **~4–6 GB** |

Use **log rotation**, **tmpfs for /tmp**, and **minimal rootfs** to maximize available space.

---

## 11. UUID-Based Partition Mounting

### Why UUID Is Mandatory With A/B

When A/B is enabled, device names are unreliable:

```
/dev/nvme0n1p1   ← Could be Slot A or Slot B depending on layout
```

The kernel command line and fstab must use **UUID** to identify the correct rootfs:

```
root=UUID=xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx
```

### Where UUIDs Are Stored

Jetson stores partition UUIDs in the bootloader directory:

```
Linux_for_Tegra/bootloader/l4t-rootfs-uuid.txt      ← Slot A UUID
Linux_for_Tegra/bootloader/l4t-rootfs-uuid.txt_b     ← Slot B UUID
```

UEFI reads these during boot to pass the correct `root=UUID=` parameter to the kernel.

### Checking Current UUID

```bash
# On a running system
blkid
lsblk -f

# Find the root partition UUID
findmnt / -o UUID
```

### What Happens If UUID Is Wrong

* Kernel fails to mount rootfs
* Boot drops to initramfs emergency shell
* If retry count exhausts, system switches to other slot
* If both UUIDs wrong, system is unbootable

---

## 12. OTA Update Strategy With A/B

### Correct OTA Flow

```
1. Device running on Slot A (current, verified)
2. Download new system image
3. Write new image to Slot B (inactive)
4. Update Slot B bootloader, kernel, DTB
5. Set Slot B as active: nvbootctrl -t rootfs set-active-boot-slot 1
6. Reboot
7. System boots into Slot B
8. Validation services run health checks
9. Mark Slot B successful: nvbootctrl -t rootfs mark-boot-successful
10. Slot B is now the running, verified slot
```

If boot into Slot B fails:

```
Retry count exhausts → automatic rollback to Slot A
Device continues running on previous known-good image
```

### What NOT to Do

**`apt upgrade` is NOT supported with A/B.**

Debian package upgrades modify the running rootfs in place. This:

* Only updates the active slot
* Leaves the inactive slot stale
* Breaks the A/B invariant (both slots should be complete, independent images)
* Can leave the system in an inconsistent state

Use **image-based OTA** — write a complete rootfs image to the inactive slot.

### OTA Methods

| Method                    | Pros                          | Cons                         |
|---------------------------|-------------------------------|------------------------------|
| NVIDIA OTA tools          | Official, tested              | NVIDIA ecosystem lock-in     |
| Custom image-based        | Full control                  | More engineering effort      |
| Mender / SWUpdate / RAUC  | Open-source, fleet management | Integration effort with L4T  |
| Container-based updates   | Fast, no rootfs change needed | Base OS updates still needed |

### Container-Based OTA (Recommended for Fleet)

For fleet-managed devices, a hybrid approach works well:

1. **Base rootfs**: Minimal or Basic, rarely updated (quarterly)
2. **Application**: Runs in Docker containers, updated frequently
3. **A/B**: Protects the base rootfs during rare OS updates
4. **Container registry**: Pulls new inference containers on schedule

This minimizes A/B partition switches while keeping application updates fast and safe.

---

## 13. Flash XML Layout Files

The partition layout is defined in XML files that `flash.sh` reads during flashing.

### Key Layout Files

```
Linux_for_Tegra/bootloader/generic/cfg/
├── flash_t234_qspi.xml        ← QSPI NOR (bootloader partitions)
└── flash_t234_nvme.xml        ← NVMe (rootfs, kernel, DTB)
```

### Example Partition Entry (Simplified)

```xml
<partition name="APP" type="data">
    <allocation_policy> sequential </allocation_policy>
    <filesystem_type> basic </filesystem_type>
    <size> 30064771072 </size>    <!-- 28 GiB in bytes -->
    <file_system_attribute> 0 </file_system_attribute>
    <allocation_attribute> 0x8 </allocation_attribute>
    <filename> system.img </filename>
</partition>
```

With `ROOTFS_AB=1`, the layout file is modified to include:

```xml
<partition name="APP" ...>
    <size> 15032385536 </size>    <!-- 14 GiB -->
    <filename> system.img </filename>
</partition>
<partition name="APP_b" ...>
    <size> 15032385536 </size>    <!-- 14 GiB -->
    <filename> system.img </filename>
</partition>
```

### Customizing Partition Layout

To add a persistent data partition:

```xml
<partition name="UDA" type="data">
    <allocation_policy> sequential </allocation_policy>
    <filesystem_type> basic </filesystem_type>
    <size> 10737418240 </size>    <!-- 10 GiB -->
    <filename> uda.img </filename>
</partition>
```

This creates a data partition that survives A/B slot switches and OTA updates — use it for models, configuration, and application data.

### Applying Custom Layout

```bash
# Flash with custom layout
sudo ROOTFS_AB=1 ./flash.sh -c my_custom_layout.xml jetson-orin-nano-devkit internal
```

---

## 14. Rootfs and Memory Architecture Connection

Rootfs and A/B slots interact with memory architecture in important ways.

### Each Slot Has Its Own Kernel and DTB

With A/B enabled, each slot loads:

* Its own kernel (`Image` from kernel or kernel_b partition)
* Its own device tree (`kernel-dtb` or `kernel-dtb_b`)
* Its own kernel modules (from rootfs `/lib/modules/`)

If these are mismatched between slots, you get:

| Mismatch                        | Symptom                              |
|---------------------------------|--------------------------------------|
| Kernel version != module version | `modprobe: FATAL: Module not found`  |
| DTB != kernel                    | Wrong memory carveouts, boot crash   |
| Old NVIDIA drivers with new kernel | GPU driver load failure, no CUDA   |
| Mismatched CMA config           | Camera buffer allocation failures    |

Always update kernel + DTB + rootfs + modules as a complete atomic unit per slot.

### Memory Carveouts Depend on DTB

Memory carveouts (BPMP, SPE, RCE, OP-TEE) are defined in the DTB. If Slot A and Slot B have different DTBs with different carveout configurations:

* Switching slots changes the memory map
* Firmware processors may malfunction if carveouts are wrong
* CMA size may differ, affecting camera buffer allocation

Production systems should use identical DTBs across both slots unless intentionally migrating to a new memory configuration.

### Kernel Modules and Driver Stack

The rootfs contains NVIDIA kernel modules under `/lib/modules/<version>/`:

```
nvidia.ko
nvgpu.ko
nvhost-*.ko
tegra-*.ko
```

These must match the running kernel exactly. A rootfs with modules from JetPack 5.1 will not work with a kernel from JetPack 6.0.

---

## 15. Designing Safe Field Updates for AI Edge Devices

### Update Architecture

```
Cloud / Update Server
        ↓ (image + manifest + signature)
Device Agent (runs on Jetson)
        ↓ (verify signature)
Write to inactive slot
        ↓
Set inactive slot as active
        ↓
Reboot
        ↓
Validation services
        ↓ (pass)                    ↓ (fail)
Mark successful              Automatic rollback
        ↓
Report success to server     Report failure to server
```

### Design Principles

1. **Atomic updates** — entire slot (bootloader + kernel + DTB + rootfs) is updated together
2. **Signature verification** — every image must be cryptographically signed; device verifies before writing
3. **Bandwidth efficiency** — use delta updates (binary diff) when possible; full images for major versions
4. **Rollback is automatic** — never require manual intervention for failed updates
5. **Health checks are application-specific** — default validation is insufficient; add camera, CUDA, inference, and network checks
6. **Data partition survives updates** — models, configuration, and logs live on a separate partition (UDA)
7. **Staged rollout** — update 1% of fleet first, monitor, then expand

### Update Failure Modes and Mitigations

| Failure Mode                   | Mitigation                                    |
|--------------------------------|-----------------------------------------------|
| Power loss during write        | Inactive slot is written; active slot intact   |
| Corrupted download             | Checksum verification before writing           |
| New image doesn't boot         | A/B rollback (automatic)                      |
| New image boots but app fails  | Custom validation rejects slot                 |
| Both slots corrupted           | Recovery kernel + USB reflash capability       |
| Storage full                   | Pre-check free space before download           |

---

## 16. Debugging Bootloops in A/B Systems

### Symptoms

* Device reboots repeatedly
* Alternates between Slot A and Slot B
* Eventually stops booting (both slots exhausted)

### Step 1 — Capture Serial Console

Connect UART (115200 baud) to the Jetson debug header. Serial output shows:

```
[UEFI] Slot A: retry_count=2, boot_successful=0
[UEFI] Booting Slot A...
[kernel] ... panic ...
[UEFI] Slot A: retry_count=1
```

Serial console is essential — without it, you're debugging blind.

### Step 2 — Check Slot State

If you can get to a shell (recovery or brief boot window):

```bash
sudo nvbootctrl -t rootfs dump-slots-info
```

Look for:

* `retry_count: 0` — slot has exhausted retries
* `boot_successful: 0` — slot was never validated

### Step 3 — Common Causes

| Cause                                | Diagnosis                                  |
|--------------------------------------|--------------------------------------------|
| Kernel panic                         | Serial log shows panic trace               |
| GPU driver fails to load             | `dmesg | grep nvgpu` shows errors          |
| Rootfs corrupted (fsck needed)       | initramfs drops to emergency shell         |
| Validation service marks slot failed | Service logs: `journalctl -u nv-l4t*`      |
| Watchdog reboot before validation    | Increase watchdog timeout or defer start   |
| DTB mismatch                         | Wrong carveouts, devices not probed        |

### Step 4 — Recovery

If both slots are exhausted:

1. Enter recovery mode (hold recovery button during power-on)
2. Connect USB-C to host machine
3. Reflash using `flash.sh`:

```bash
sudo ./flash.sh jetson-orin-nano-devkit internal
```

### Prevention

* Always test OTA images in a staging environment before fleet deployment
* Set retry count to a reasonable value (3–7, not 1)
* Ensure validation services complete before any watchdog-triggered reboot
* Keep serial console accessible on production hardware (debug header or test points)

---

## 17. Production Hardening Checklist

Before deploying Jetson Orin Nano in the field:

### Storage

- [ ] A/B enabled with correct ROOTFSSIZE
- [ ] Separate UDA data partition for persistent storage
- [ ] Log rotation configured (prevent storage exhaustion)
- [ ] Read-only rootfs with overlay (if applicable)

### Boot and Recovery

- [ ] Custom validation service with application-specific health checks
- [ ] Retry count set appropriately (3–7)
- [ ] Recovery kernel configured
- [ ] Serial console accessible for debugging

### OTA

- [ ] Image-based OTA (not apt upgrade)
- [ ] Cryptographic signature verification on images
- [ ] Staged rollout process defined
- [ ] Rollback tested and verified

### Security

- [ ] Secure boot enabled (fuses burned)
- [ ] SSH keys deployed (password auth disabled)
- [ ] Unnecessary services removed
- [ ] Firewall configured

### Memory

- [ ] CMA sized for workload (see [Memory Architecture Guide](../Orin-Nano-Memory-Architecture/Guide.md#8-how-to-resize-cma))
- [ ] Memory budget calculated (cameras + model + runtime + OS)
- [ ] OOM behavior tested under load

### Monitoring

- [ ] tegrastats or equivalent running
- [ ] CMA/buddyinfo logged periodically
- [ ] Thermal monitoring with alerts
- [ ] Remote health reporting to fleet management

---

## 18. References

* [NVIDIA Jetson Linux Developer Guide — Bootloader](https://docs.nvidia.com/jetson/archives/r36.4/DeveloperGuide/SD/Bootloader/JetsonModuleBootProcess.html) — boot flow and A/B documentation
* [NVIDIA Jetson Linux Developer Guide — Root File System](https://docs.nvidia.com/jetson/archives/r36.4/DeveloperGuide/SD/RootFileSystem.html) — rootfs generation and customization
* [NVIDIA Jetson Linux Developer Guide — OTA](https://docs.nvidia.com/jetson/archives/r36.4/DeveloperGuide/SD/SoftwarePackagesAndTheUpdateMechanism.html) — OTA update mechanisms
* [Mender for Jetson](https://docs.mender.io/devices/nvidia-jetson) — open-source OTA for Jetson
* [SWUpdate](https://sbabic.github.io/swupdate/) — software update framework for embedded Linux
* Main guide: [Nvidia Jetson Platform Guide](../Guide.md)
* Memory deep dive: [Orin Nano Memory Architecture](../Orin-Nano-Memory-Architecture/Guide.md)
