# Lecture 22: Embedded Storage: eMMC, UFS, NVMe & OTA Partitioning

## Overview

Embedded AI devices — dashcams, autonomous driving computers, edge inference boxes — must store operating systems, neural network weights, and continuous sensor recordings. The storage technology chosen for each device determines write throughput, random I/O latency, physical resilience, and how long the device lasts before the flash wears out. Unlike a laptop or server where storage can be upgraded, embedded storage is soldered to the board and must last for years of continuous operation.

The mental model to carry through this lecture is **storage as a system component with a finite lifespan**: flash memory wears out after a defined number of Program/Erase cycles. Every design decision — filesystem choice, partition layout, OTA strategy, TRIM configuration — either extends or shortens that lifespan. OTA partition layout determines whether a failed software update bricks a device or recovers gracefully, which in a deployed fleet of autonomous vehicles is the difference between a software recall and a seamless update.

AI hardware engineers need to understand embedded storage because the wrong partition layout makes an OTA update irreversible. The wrong filesystem choice on a dashcam eMMC leads to premature flash death. The wrong storage type (eMMC vs NVMe) on a new platform limits camera recording throughput. These are not theoretical concerns — they are recurring failure modes in real deployed AI edge hardware.

---

## Storage Hierarchy for Embedded AI Devices

Three storage technologies appear in AI edge hardware, chosen by **cost, form factor, and throughput** requirements:

| Technology | Interface | Seq Read | Random IOPS | Form Factor |
|---|---|---|---|---|
| eMMC 5.1 | Parallel (8-bit HS400) | ~400 MB/s | ~15K | Soldered BGA |
| UFS 3.1 | Serial MIPI M-PHY | ~2100 MB/s | ~70K | Soldered / socket |
| UFS 4.0 | Serial MIPI M-PHY | ~4200 MB/s | ~130K | Soldered / socket |
| NVMe Gen4 x4 | PCIe | ~7000 MB/s | ~1M | M.2 / BGA |

The choice follows a **cost-performance curve**: eMMC is cheapest and slowest; NVMe is fastest and most expensive (and requires PCIe lanes from the SoC). UFS occupies the **middle ground** and is the standard for high-end mobile AI platforms.

```
Cost / Integration                     Performance
  ◀────────────────────────────────────────────────▶
  eMMC 5.1          UFS 3.1/4.0           NVMe Gen4
  (8-bit parallel)  (MIPI serial)         (PCIe x4)
  ~400 MB/s         ~2100-4200 MB/s       ~7000 MB/s
  ~15K IOPS         ~70K-130K IOPS        ~1M IOPS
  Jetson Nano       comma 3X              Jetson Orin
  Low-cost edge     Snapdragon 845        M.2 slot
```

---

## eMMC (embedded MultiMediaCard)

eMMC implements the **JEDEC JESD84 standard**. The flash controller, wear leveling, ECC, and bad block management are all **integrated inside the package**. From the host's perspective, eMMC presents a simple block device with a fixed capacity.

### Internal Partition Structure

Every eMMC device exposes fixed partitions that are separate from the user data area and cannot be repartitioned by the OS:

```
┌─────────────────────────────────────────────────────┐
│              eMMC Physical Package                   │
│                                                     │
│  ┌──────────┐  ┌──────────┐  ┌────────┐            │
│  │  BOOT0   │  │  BOOT1   │  │  RPMB  │            │
│  │  (4 MB)  │  │  (4 MB)  │  │ (4 MB) │            │
│  │ U-Boot   │  │ Backup   │  │ Secure │            │
│  │   SPL    │  │ bootldr  │  │  keys  │            │
│  └──────────┘  └──────────┘  └────────┘            │
│                                                     │
│  ┌─────────────────────────────────────────────┐   │
│  │           User Data Area (UDA)               │   │
│  │  GPT partitioned by the OS                  │   │
│  │  /dev/mmcblk0 → mmcblk0p1, p2, p3, ...      │   │
│  └─────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────┘
```

- **BOOT0 / BOOT1**: two independent boot areas, each up to 4 MB; write-protected after provisioning; hold bootloader (U-Boot SPL, UEFI)
- **RPMB (Replay Protected Memory Block)**: 4 MB authenticated storage; read/write protected by HMAC-SHA256 using a device-unique key provisioned at manufacturing
- **User Data Area (UDA)**: the main storage region; GPT-partitioned by the OS

### RPMB Security Properties

RPMB prevents **rollback attacks on firmware**. The secure world (OP-TEE, ARM TrustZone) increments a write counter on each write; the eMMC controller rejects any write that does not carry a **valid HMAC** and a counter value ≥ current. Used in Jetson secure boot to store:

- Firmware version counter (anti-rollback)
- Encryption key material for full-disk encryption

> **Key Insight:** RPMB is a hardware-enforced write counter that the host cannot fake. An attacker who gains root access on the host OS cannot decrement the firmware version counter in RPMB because every write must include a valid HMAC computed with a key that exists only inside the TrustZone secure world. This makes firmware downgrade attacks (to a version with known exploits) impossible even with full OS compromise.

### eMMC Reliability

- Consumer grade: 3K–10K P/E cycles per cell (MLC); 1K–3K (TLC)
- Enterprise grade (JEDEC JESD84-B51): 16 PBW (petabytes written) rated
- Over-provisioning: 10–28% of raw capacity reserved for wear leveling and bad block replacement

> **Common Pitfall:** Consumer-grade eMMC used in a high-write application (continuous camera recording) without F2FS or TRIM can exhaust P/E cycles in months rather than years. A 64 GB TLC eMMC at 1K P/E cycles has a total byte write budget of roughly 64 TB. A dashcam writing 50 MB/s continuously could theoretically exhaust this in ~15 days if write amplification is 1x — which it never is without F2FS. With F2FS and proper TRIM configuration, the effective write amplification drops, extending device life significantly.

---

## UFS (Universal Flash Storage)

UFS uses a **serial, full-duplex MIPI M-PHY** physical layer with a SCSI-based command set (UFS Transport Protocol). Key advantages over eMMC:

- **Command queuing**: up to 32 commands in flight (vs eMMC single-command); critical for random I/O latency
- **Full duplex**: simultaneous read and write; improves mixed workload throughput
- **Lower CPU overhead**: command processing offloaded to UFS device controller

UFS uses the same logical partition structure as eMMC (BOOT, RPMB, UDA). It is the **standard on Qualcomm Snapdragon platforms**. The comma 3X (openpilot primary hardware) uses a Snapdragon 845 with UFS storage for camera recording and OS.

> **Key Insight:** The most important practical difference between eMMC and UFS for an autonomous driving platform is command queuing. eMMC processes one command at a time. While the camera driver issues a write for frame N, any other I/O (OS, modeld reading model weights, logging) must wait. UFS can process 32 commands simultaneously, meaning camera writes, model reads, and log writes all proceed in parallel with no head-of-line blocking.

The improvement in random IOPS (~70K for UFS 3.1 vs ~15K for eMMC 5.1) directly reflects this: random I/O throughput scales with the number of outstanding commands the device can handle.

---

## NVMe on Embedded Platforms

For platforms where **maximum storage throughput** is required — large-scale dataset logging, high-frequency sensor recording, or training data capture — **NVMe via PCIe** is the correct choice.

Jetson Orin exposes an M.2 Key M slot connected to PCIe Gen4 x4 (~7 GB/s):

- Supports standard NVMe 2280 and 2230 form-factor SSDs
- Used when large-scale dataset logging, replay, or high-frequency sensor recording is needed
- `blk-mq` with per-CPU NVMe queues; combine with `io_uring` + `O_DIRECT` for maximum throughput
- `nvme list` to enumerate devices; `nvme smart-log /dev/nvme0` for health and wear indicators

With the io_uring techniques from Lecture 19 and the PCIe architecture from Lecture 20, the full path from CUDA inference to NVMe logging is: CUDA output → GPU memory → DMA-BUF → GPUDirect Storage → NVMe PCIe DMA → SSD — with no CPU involvement in the data path.

---

## OTA Partition Layouts

With the storage technologies understood, the next question is how to structure partitions for safe over-the-air updates. The partition layout determines what happens when an update fails.

Over-the-air update strategy determines **recovery behavior and downtime** on failure.

### A/B Seamless Update

Two full system partition sets (**slot A and slot B**). The **inactive slot receives the update** while the active slot runs normally.

```
/dev/mmcblk0p1  boot_a      (active)
/dev/mmcblk0p2  boot_b      (inactive → receives new image)
/dev/mmcblk0p3  system_a    (active rootfs)
/dev/mmcblk0p4  system_b    (inactive → receives new rootfs)
/dev/mmcblk0p5  userdata    (persistent, not updated)
```

The A/B update and rollback sequence:

```
Normal operation:          Slot A active, Slot B empty/old
      ↓
OTA download begins:       New image written to Slot B
      ↓                    (device continues running on Slot A)
Download complete:         Bootloader marks Slot B as "try next boot"
      ↓
Reboot:                    Bootloader activates Slot B, boots count = 1
      ↓
Boot success check:        If userspace confirms success → mark B permanent
Boot failure check:        If boot count exceeds threshold (3) → revert to A
      ↓
Result:
  Success path: Slot B is now active, Slot A becomes the new inactive slot
  Failure path: Slot A is active again, Slot B contains the failed image
```

On reboot: bootloader marks slot B as active, tries to boot. If boot count exceeds threshold (typically 3) without a successful boot marker, bootloader reverts to slot A. Used in: Android A/B, openpilot Agnos, Jetson UEFI capsule update.

> **Key Insight:** The critical property of A/B OTA is that the running system is never interrupted during the update download and write process. The new image is written to the inactive slot while the device continues operating normally on the active slot. This eliminates the downtime window during which the device would be in an unbootable state. For an autonomous driving computer, this means an OTA update can be delivered and prepared while the vehicle is parked, completing the final atomic boot-slot-switch in seconds on the next restart.

### Recovery Partition Update

Single active partition + a small recovery image. Update process: download new image → reboot into recovery → flash system partition → reboot into new system. **No rollback without a backup image**. Used in older Android devices and simple embedded systems.

> **Common Pitfall:** Recovery-partition updates have no automatic rollback. If the new image fails to boot, the device is stuck. For production autonomous driving deployments, this is unacceptable — a failed update must never result in a permanently bricked device in the field. A/B partitioning is the minimum requirement for any OTA-updatable AI edge device.

### OSTree: Git-Like Atomic Updates

OSTree maintains a **content-addressed object store** of filesystem trees, analogous to git. Deployed updates are hard-linked from the object store; **atomic switchover via a symlink update**. Used in Automotive Grade Linux (AGL) and automotive ECU Linux platforms. Compatible with ext4 (hard links) and btrfs.

The OSTree model is conceptually elegant: each deployed OS version is a git commit hash. Rolling back to a previous version is as simple as updating a symlink. Differential updates (only changed files) are as compact as possible because unchanged files are already in the object store.

---

## Partition Tools and Kernel Interfaces

```bash
parted /dev/mmcblk0 print       # display current GPT partition table
sgdisk -p /dev/mmcblk0          # GPT partition table details
partprobe /dev/mmcblk0          # re-read partition table without reboot
cat /proc/partitions             # kernel's view of all block devices
cat /sys/block/mmcblk0/mmcblk0p1/size  # partition size in 512-byte sectors
ls /dev/mmcblk0*                 # list all partition device nodes
```

- `parted` / `sgdisk`: create and modify GPT partition tables
- `partprobe /dev/mmcblk0`: re-read partition table without reboot
- `/proc/partitions`: kernel's view of all block devices and partitions
- `/sys/block/mmcblk0/mmcblk0p1/size`: partition size in 512-byte sectors
- `/dev/mmcblk0p1`, `/dev/nvme0n1p1`: device nodes for partition access

---

## Wear Leveling and Write Amplification

Flash storage degrades with **Program/Erase (P/E) cycles**. The Flash Translation Layer (FTL) manages longevity:

- **Dynamic wear leveling**: preferentially writes to least-worn blocks; effective for frequently updated data
- **Static wear leveling**: periodically migrates cold data from worn blocks to fresh ones; prevents hot/cold imbalance
- **Write Amplification Factor (WAF)**: physical writes / logical writes; WAF of 1.0 is ideal; random small writes to consumer SSDs can produce WAF of 10–50 without filesystem cooperation
- Minimize WAF with: sequential writes, large block sizes, F2FS or btrfs, and proper TRIM configuration

The Write Amplification Factor cascade:

```
Application writes 4 KB (one page)
       ↓ filesystem (F2FS: log-structured, ~1x WAF)
       ↓ filesystem (ext4 random: ~3-5x WAF)
       ↓ FTL garbage collection (without TRIM: +5-10x)
Physical NAND writes: 4 KB × WAF_fs × WAF_ftl
  Best case (F2FS + TRIM): ~4 KB physical
  Worst case (ext4 random, no TRIM): ~160-200 KB physical
```

> **Key Insight:** WAF is multiplicative: filesystem WAF multiplies with FTL WAF. A WAF of 3 at the filesystem layer combined with a WAF of 10 at the FTL layer means 30x more physical writes than logical writes. On a TLC eMMC rated for 1K P/E cycles, this reduces effective device lifetime by 30x. The difference between a 6-month flash lifespan and a 15-year flash lifespan can come down entirely to filesystem and TRIM configuration.

---

## Storage Diagnostics

```bash
# Real-time I/O statistics: util%, await (ms), r/s, w/s, throughput
iostat -x 1

# Block-level I/O tracing for a specific device
blktrace -d /dev/mmcblk0 -o trace    # capture block I/O events

# Parse and display timing breakdown from trace
blkparse trace.blktrace.0            # shows per-request latency breakdown

# NVMe health and wear indicators
nvme smart-log /dev/nvme0            # temperature, available spare, wear indicator

# eMMC health
cat /sys/class/mmc_host/mmc0/mmc0:0001/life_time  # eMMC wear level (0x01-0x0A)
```

`await` in `iostat` output is the **average I/O service time** including queue wait. A rising `await` under camera recording load indicates the **storage is saturated**. `blktrace` identifies which process is responsible for latency spikes.

> **Common Pitfall:** Diagnosing storage saturation only with `iostat -x`. The `%util` field in iostat shows queue utilization, not device saturation — a modern NVMe SSD can be at 100% queue utilization while still having headroom if the queue depth is low. Always check `await` (average service time) and `r_await`/`w_await` separately. For eMMC specifically, check the life_time sysfs attribute monthly during development to validate that your write patterns are within the expected wear budget.

---

## Summary

| Storage type | Interface | Sequential read | Random IOPS | Typical use in AI hardware |
|---|---|---|---|---|
| eMMC 5.1 | 8-bit parallel HS400 | ~400 MB/s | ~15K | Jetson Nano, low-cost edge devices |
| UFS 3.1 | MIPI M-PHY serial | ~2100 MB/s | ~70K | comma 3X (Snapdragon), mid-range SoC |
| UFS 4.0 | MIPI M-PHY serial | ~4200 MB/s | ~130K | Flagship mobile AI SoC |
| NVMe Gen4 x4 | PCIe Gen4 | ~7000 MB/s | ~1M | Jetson Orin (M.2 slot), dataset logging |
| NVMe Gen3 x4 | PCIe Gen3 | ~3500 MB/s | ~500K | Workstation AI dev, cloud training node |

### Conceptual Review

- **Why does RPMB prevent firmware rollback attacks even when the attacker has root access on the host OS?** RPMB write authentication requires an HMAC computed with a key that lives exclusively in the TrustZone secure world (OP-TEE). Root access on the normal world Linux OS does not grant access to this key. The eMMC hardware controller itself validates the HMAC and monotonically increasing write counter — it rejects any write that does not pass this check, regardless of who is requesting it.

- **What practical advantage does UFS command queuing provide over eMMC for an autonomous driving platform?** eMMC accepts one command at a time (depth-1 queue). On a system simultaneously recording camera frames (sequential writes), running neural net inference (random reads of model weights), and logging sensor data (mixed I/O), these operations serialize behind each other. UFS with 32 commands in flight allows all three to proceed in parallel, reducing the average latency of each operation.

- **What is the A/B OTA rollback trigger, and who controls it?** The bootloader maintains a boot attempt counter for the inactive slot being tried. If the userspace software (update daemon, openpilot's updateinstallerd) does not write a "boot successful" marker within a timeout after boot, the counter continues to increment. When the counter exceeds the threshold (typically 3), the bootloader marks the inactive slot as failed and reverts to the previously active slot. The explicit success confirmation from userspace is required — silence is treated as failure.

- **How does Write Amplification Factor (WAF) interact with eMMC lifespan, and what is the best way to minimize it?** Physical P/E cycles consumed = logical writes × WAF_filesystem × WAF_FTL. Minimizing WAF requires: (1) a log-structured filesystem (F2FS) that produces sequential writes matching flash erase block boundaries; (2) TRIM enabled (via `systemd-fstrim.timer`) so the FTL knows which blocks are free and can avoid garbage collection overhead; (3) adequate over-provisioning reserved in the partition layout.

- **What does OSTree have in common with git, and why is this useful for OTA updates?** OSTree stores filesystem trees as content-addressed objects (like git blobs and trees). Each OS version is a commit pointing to its root tree. A differential update downloads only the objects not already present in the local store — exactly like `git fetch`. Rollback is a symlink update to point to the previous commit's tree. The delta between versions can be arbitrarily small, making network-efficient incremental updates practical even over limited bandwidth.

- **What does `blktrace` reveal that `iostat` cannot?** `iostat` reports aggregate device statistics: total throughput, average latency, queue depth. `blktrace` captures every individual I/O request with timestamps at each stage of the block layer pipeline: plug (batched), unplug (dispatched), issue (sent to device), complete (DMA done). This allows you to identify which specific process is generating the highest-latency requests, where in the pipeline latency is being added, and whether the issue is queue contention vs. actual device slowness.

---

## AI Hardware Connection

- A/B OTA partition layout is the foundation of openpilot Agnos update strategy; the active slot continues running while the new firmware is written to the inactive slot, with automatic rollback on failed boot
- RPMB authenticated storage is how Jetson secure boot prevents firmware downgrade attacks; the TrustZone secure world increments the anti-rollback counter on each successful firmware update
- UFS on the Snapdragon 845 (comma 3X) provides the random IOPS needed to sustain simultaneous recording from three cameras while running inference
- NVMe on Jetson Orin enables large-scale on-device dataset logging for continuous learning pipelines; PCIe P2P allows the logged data to be pre-processed directly in GPU memory
- F2FS on eMMC is the correct filesystem choice for long-running edge AI dashcam devices; it minimizes write amplification and extends eMMC lifespan compared to ext4
- `iostat -x 1` is the first diagnostic tool to run when a camera recording pipeline drops frames; `await` rising above the camera frame period indicates a storage bottleneck
