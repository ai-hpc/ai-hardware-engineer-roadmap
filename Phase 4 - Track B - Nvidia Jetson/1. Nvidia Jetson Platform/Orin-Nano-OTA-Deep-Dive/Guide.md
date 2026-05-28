# Orin Nano 8GB — OTA (Over-The-Air) Update Deep Dive

<div class="course-identity auto-course" style="--course-accent: #65a30d; --course-accent-rgb: 101, 163, 13;" markdown="1">
<div class="course-identity__icon">ON8O</div>
<div markdown="1">
<p class="course-identity__eyebrow">Deep Dive · Jetson Track</p>
<p class="course-identity__title">Specialized course identity for Orin Nano 8GB — OTA (Over-The-Air) Update Deep Dive.</p>
<p class="course-identity__meta">Artifact: Jetson integration demo · Measure: latency, memory, power, logs</p>
</div>
</div>


> **Scope:** Complete production-level understanding of OTA update mechanisms on Jetson Orin Nano — from NVIDIA's native update engine internals, through payload generation and cryptographic signing, A/B slot orchestration, Tegra-specific bootloader update chains, to fleet-scale deployment with SWUpdate/Mender/RAUC, delta update engineering, rollback guarantees, and field failure analysis.
>
> **Prerequisites:** Familiarity with [Orin Nano boot chain and A/B redundancy](../Orin-Nano-Rootfs-and-AB-Redundancy/Guide.md), [security architecture](../Orin-Nano-Security/Guide.md), and [Yocto BSP production](../Orin-Nano-Yocto-BSP-Production/Guide.md).

---


## 1. Why OTA Is Hard on Jetson

OTA on Jetson is fundamentally more complex than on a standard Linux server or even a Raspberry Pi. Understanding why is critical before designing your update system.

### 1.1 The Jetson Update Problem Space

```
Standard Linux server OTA:
  - Single boot path (BIOS/UEFI → GRUB → kernel → rootfs)
  - apt/yum upgrade, reboot
  - Rollback: snapshot (ZFS/btrfs) or backup

Jetson Orin OTA — what actually needs updating:
  ┌─────────────────────────────────────────────────┐
  │ QSPI NOR Flash (32 MB)                         │
  │  ├── BCT (Boot Configuration Table)             │
  │  ├── MB1 (Microboot 1 — DRAM init, power rails) │
  │  ├── MB2 (Microboot 2 — loads UEFI)             │
  │  ├── UEFI (replaces traditional bootloader)     │
  │  ├── SPE firmware (Safety Processor Engine)     │
  │  ├── RCE firmware (Real-time Camera Engine)     │
  │  ├── SCE firmware (Safety/Crypto Engine)        │
  │  ├── APE firmware (Audio Processing Engine)     │
  │  ├── BPMP firmware (Boot and Power Mgmt)        │
  │  ├── TOS (Trusted OS — OP-TEE)                  │
  │  ├── EKS (Encryption Key Store)                 │
  │  └── DCE firmware (Display Controller Engine)   │
  ├─────────────────────────────────────────────────┤
  │ NVMe / eMMC                                     │
  │  ├── Kernel Image                               │
  │  ├── Kernel DTB (Device Tree Blob)              │
  │  ├── Kernel modules (in rootfs)                 │
  │  └── Root filesystem (APP partition)            │
  └─────────────────────────────────────────────────┘
```

### 1.2 What Makes Jetson OTA Different

| Challenge | Why It's Hard |
|---|---|
| Multi-component atomicity | 15+ firmware blobs must be consistent with kernel and rootfs |
| QSPI flash is not a block device | Cannot simply `dd` — requires Tegra-specific flash protocols |
| Bootloader is not GRUB | MB1/MB2/UEFI chain has NVIDIA-proprietary update path |
| Firmware processor coupling | BPMP/RCE/SPE firmware must match kernel DTB carveouts |
| Fuse-locked secure boot | Signed images only — wrong key = bricked device |
| Power-fail during QSPI write | QSPI has no A/B on some partitions — corruption = brick |
| 8 GB RAM constraint | Cannot hold full rootfs image in memory during update |
| NVMe wear | Repeated full-image writes wear flash cells |
| Field devices are unattended | No physical access for recovery if update fails |

### 1.3 The Update Surface Matrix

Every component has different update characteristics:

```
Component        Storage    A/B?   Update Method           Risk Level
─────────────────────────────────────────────────────────────────────
MB1              QSPI       Yes    nv_update_engine        CRITICAL
MB2              QSPI       Yes    nv_update_engine        CRITICAL
UEFI             QSPI       Yes    nv_update_engine        CRITICAL
BPMP-FW          QSPI       Yes    nv_update_engine        CRITICAL
SPE-FW           QSPI       Yes    nv_update_engine        HIGH
RCE-FW           QSPI       Yes    nv_update_engine        HIGH
TOS (OP-TEE)     QSPI       Yes    nv_update_engine        CRITICAL
BCT              QSPI       Yes    nv_update_engine        CRITICAL
EKS              QSPI       No*    nv_update_engine        CRITICAL
Kernel           NVMe       Yes    dd / image write        MEDIUM
Kernel DTB       NVMe       Yes    dd / image write        MEDIUM
Rootfs           NVMe       Yes    dd / image write        LOW-MEDIUM
Kernel modules   In rootfs  Yes    Part of rootfs update   LOW
Application      Container  N/A    docker pull             LOW

* EKS is shared — special handling required
```

---

## 2. OTA Architecture Overview

### 2.1 End-to-End OTA System Architecture

```
┌──────────────────────────────────────────────────────────────────┐
│                        BUILD SYSTEM                              │
│  ┌──────────────┐  ┌──────────────┐  ┌───────────────────────┐  │
│  │ Yocto/L4T    │  │ Payload      │  │ Signing Service       │  │
│  │ Build        │→ │ Generator    │→ │ (HSM / KMS)           │  │
│  │              │  │              │  │                       │  │
│  │ rootfs.ext4  │  │ BUP payload  │  │ RSA-4096 / ECDSA-P256│  │
│  │ kernel Image │  │ SWU archive  │  │                       │  │
│  │ DTBs         │  │ Mender artif.│  │ Signed manifests      │  │
│  │ QSPI blobs   │  │              │  │ Signed payloads       │  │
│  └──────────────┘  └──────────────┘  └───────────────────────┘  │
└──────────────────────────────┬───────────────────────────────────┘
                               │ Upload signed artifacts
                               ▼
┌──────────────────────────────────────────────────────────────────┐
│                     OTA UPDATE SERVER                            │
│  ┌──────────────┐  ┌──────────────┐  ┌───────────────────────┐  │
│  │ Artifact     │  │ Rollout      │  │ Device Registry       │  │
│  │ Repository   │  │ Controller   │  │                       │  │
│  │              │  │              │  │ - Device ID           │  │
│  │ Versioned    │  │ Staged %     │  │ - Current version     │  │
│  │ Full + Delta │  │ Canary logic │  │ - Hardware revision   │  │
│  │ artifacts    │  │ Auto-pause   │  │ - Update status       │  │
│  └──────────────┘  └──────────────┘  └───────────────────────┘  │
└──────────────────────────────┬───────────────────────────────────┘
                               │ HTTPS + mTLS
                               ▼
┌──────────────────────────────────────────────────────────────────┐
│                     DEVICE (Jetson Orin Nano)                    │
│                                                                  │
│  ┌──────────────┐  ┌──────────────┐  ┌───────────────────────┐  │
│  │ Update Agent │  │ Payload      │  │ Boot Validation       │  │
│  │ (daemon)     │  │ Applier      │  │ Service               │  │
│  │              │  │              │  │                       │  │
│  │ Poll/push    │  │ Verify sig   │  │ Health checks         │  │
│  │ Download     │  │ Write slot B │  │ mark-boot-successful  │  │
│  │ Pre-flight   │  │ Update QSPI  │  │ Rollback trigger      │  │
│  │ checks       │  │ Switch slot  │  │                       │  │
│  └──────────────┘  └──────────────┘  └───────────────────────┘  │
└──────────────────────────────────────────────────────────────────┘
```

### 2.2 Update Layers

Production OTA systems on Jetson operate on three distinct layers with different update cadences:

```
Layer 1: Platform (Quarterly — high risk, full A/B)
  ├── QSPI firmware (MB1, MB2, UEFI, BPMP, SPE, RCE, TOS)
  ├── Kernel + DTB
  └── Rootfs (base OS)
  → Requires: Full A/B slot switch, device reboot
  → Rollback: Automatic via nvbootctrl

Layer 2: Runtime (Monthly — medium risk, container restart)
  ├── CUDA/TensorRT libraries (if not in rootfs)
  ├── Inference engine
  └── AI models / TensorRT engines
  → Requires: Container restart or service restart
  → Rollback: Previous container image tag

Layer 3: Configuration (Weekly/Daily — low risk, no restart)
  ├── Model weights / parameters
  ├── Application configuration
  └── Feature flags
  → Requires: Config reload signal (SIGHUP)
  → Rollback: Previous config version in data partition
```

### 2.3 Update Decision Matrix

```
What Changed?                  → Update Method
──────────────────────────────────────────────────────────
App code only                  → Container pull (Layer 2)
Model weights only             → File sync to /data (Layer 3)
Kernel security patch          → Full A/B OTA (Layer 1)
NVIDIA driver update           → Full A/B OTA (Layer 1)
New JetPack minor version      → Full A/B OTA (Layer 1)
New JetPack major version      → Factory reflash recommended
UEFI vulnerability fix         → QSPI + rootfs OTA (Layer 1)
Custom DTB change              → Kernel+DTB OTA (Layer 1)
```

---

## 3. NVIDIA nv_update_engine Internals

### 3.1 What Is nv_update_engine

`nv_update_engine` is NVIDIA's native tool for updating QSPI firmware and boot partitions on Jetson. It is the **only supported method** for updating bootloader components in the field (without USB recovery mode).

```
Location:     /usr/sbin/nv_update_engine
Package:      nvidia-l4t-tools
Source:       Closed-source (NVIDIA proprietary)
Input:        BUP (Bootloader Update Payload)
Targets:      QSPI NOR flash partitions
A/B aware:    Yes — writes to inactive slot only
```

### 3.2 BUP (Bootloader Update Payload)

The BUP is a structured binary that contains all firmware blobs needed for a bootloader update:

```
BUP File Structure:
  ┌─────────────────────────────────────┐
  │ BUP Header                          │
  │   Magic: "NVIDIA__BLOB__V2"         │
  │   Version                           │
  │   Number of entries                 │
  │   Total payload size                │
  └─────────────────────────────────────┘
  ┌─────────────────────────────────────┐
  │ Entry Table                         │
  │   Entry 0: mb1 → offset, size, hash│
  │   Entry 1: mb2 → offset, size, hash│
  │   Entry 2: uefi → offset, size, hash│
  │   Entry 3: bpmp-fw → ...           │
  │   Entry 4: spe-fw → ...            │
  │   Entry 5: rce-fw → ...            │
  │   Entry 6: sce-fw → ...            │
  │   Entry 7: tos → ...               │
  │   Entry 8: eks → ...               │
  │   Entry 9: dce-fw → ...            │
  │   Entry N: ...                      │
  └─────────────────────────────────────┘
  ┌─────────────────────────────────────┐
  │ Payload Data                        │
  │   [MB1 binary blob]                 │
  │   [MB2 binary blob]                 │
  │   [UEFI binary blob]               │
  │   [BPMP firmware blob]             │
  │   ...                              │
  └─────────────────────────────────────┘
```

### 3.3 Generating a BUP

```bash
# On the host machine (x86_64), inside Linux_for_Tegra directory

# Step 1: Set up the environment
export BOARDID=3767           # Orin Nano module
export BOARDSKU=0005          # 8GB variant
export FAB=TS4                # Fabrication revision
export BOARDREV=              # Board revision
export FUSELEVEL=fuselevel_production
export CHIPREV=0

# Step 2: Generate the BUP payload
# This packages all QSPI firmware blobs into a single update file
sudo ./l4t_generate_soc_bup.sh t234

# Output: payloads_t234/bl_update_payload
# This is the BUP file to deploy to devices

# Step 3: Verify the generated BUP
ls -la payloads_t234/bl_update_payload
# Typical size: 30-60 MB depending on configuration
```

### 3.4 Generating BUP for Multi-Spec (Multiple Board Variants)

```bash
# For fleets with multiple carrier board designs:
# Generate a multi-spec BUP that handles all variants

# Create a spec file listing all supported configurations
cat > multi_spec.conf << 'EOF'
# BOARDID  BOARDSKU  FAB  BOARDREV  FUSELEVEL  CHIPREV
3767       0005      TS4            fuselevel_production  0
3767       0004      TS4            fuselevel_production  0
3768       0000      TS1            fuselevel_production  0
EOF

# Generate multi-spec BUP
sudo ./l4t_generate_soc_bup.sh -e multi_spec.conf t234

# The resulting BUP contains firmware for all listed board variants
# nv_update_engine on the device selects the correct variant automatically
```

### 3.5 Applying BUP on the Device

```bash
# On the Jetson device:

# Step 1: Copy BUP to device
scp bl_update_payload jetson-device:/tmp/

# Step 2: Check current bootloader version
sudo nv_update_engine -e --check

# Step 3: Apply the update (writes to INACTIVE slot)
sudo nv_update_engine -i /tmp/bl_update_payload -e

# What happens internally:
#   1. Parses BUP header, validates structure
#   2. Identifies current active slot (e.g., Slot A)
#   3. For each entry in BUP:
#      a. Reads corresponding partition from QSPI
#      b. Compares hash — skips if identical
#      c. Erases target partition on inactive slot (Slot B)
#      d. Writes new firmware blob
#      e. Verifies write (read-back and hash compare)
#   4. Updates slot metadata (retry count, priority)
#   5. Does NOT switch active slot — that's a separate step

# Step 4: Verify the update was written correctly
sudo nv_update_engine -e --verify

# Step 5: Switch to the updated slot
sudo nvbootctrl set-active-boot-slot 1   # if currently on slot 0
sudo reboot
```

### 3.6 nv_update_engine Internals — What Happens During Update

```
nv_update_engine execution flow:

  1. Parse BUP file
     ├── Validate magic number ("NVIDIA__BLOB__V2")
     ├── Read entry table
     └── Verify overall checksum

  2. Detect current hardware
     ├── Read BOARDID from EEPROM (via /proc/device-tree)
     ├── Read module SKU, FAB, CHIPREV
     └── Select matching spec from multi-spec BUP

  3. Determine target slot
     ├── Query nvbootctrl for current slot
     └── Target = inactive slot (opposite of current)

  4. For each firmware entry:
     ├── Open QSPI MTD device for target partition
     │     /dev/mtd0 (mb1_b), /dev/mtd1 (mb2_b), etc.
     ├── Read existing content
     ├── Compare SHA-256 hash with BUP entry
     ├── If different:
     │     ├── Erase MTD partition (block-by-block)
     │     ├── Write new firmware (with ECC if supported)
     │     └── Read back and verify
     └── If identical: skip (saves time and flash wear)

  5. Update BCT (Boot Configuration Table) on target slot
     ├── BCT contains boot parameters, DRAM config
     └── Critical: wrong BCT = device won't boot

  6. Finalize
     ├── Set target slot retry count (default: 7)
     ├── Set target slot as bootable
     └── Return success/failure code
```

### 3.7 Critical nv_update_engine Caveats

```
1. NEVER interrupt nv_update_engine during execution
   - QSPI erase+write is NOT atomic per-partition
   - Interruption mid-write corrupts the target slot
   - The INACTIVE slot is written, so active slot is safe
   - But if both slots are corrupt: device needs USB recovery

2. Power-fail safety
   - If power fails during nv_update_engine:
     Active slot: SAFE (untouched)
     Inactive slot: POTENTIALLY CORRUPT
   - Recovery: re-run nv_update_engine (it will detect and rewrite)

3. EKS (Encryption Key Store) is SHARED
   - Some T234 configurations share EKS between slots
   - Updating EKS affects BOTH slots
   - If EKS update fails: BOTH slots may fail to boot
   - Mitigation: only update EKS when absolutely necessary

4. Secure boot constraints
   - If fuses are burned with a signing key:
     BUP MUST be signed with the same key
   - Wrong key → nv_update_engine rejects the payload
   - Lost key → device cannot be updated (permanently)

5. Space requirement
   - BUP file: 30-60 MB
   - Temporary space during apply: ~100 MB
   - Ensure /tmp or target directory has sufficient space
```

---

## 4. Tegra Bootloader Update Chain

### 4.1 Boot Chain Components and Update Order

The Tegra T234 boot chain has a strict ordering. Understanding this is essential for safe updates.

```
Boot Chain Order (power-on sequence):

  BootROM (mask ROM — cannot be updated)
     ↓
  BCT (Boot Configuration Table)
     ↓  DRAM timing, boot device config
  MB1 (Microboot 1)
     ↓  PMIC sequencing, DRAM training, initial security
  MB1-BCT (MB1 Boot Config)
     ↓
  MB2 (Microboot 2)
     ↓  Loads remaining firmware processors
  ┌────┬────┬────┬────┬────┐
  │BPMP│ SCE│ RCE│ SPE│ APE│  (Firmware processors — loaded by MB2)
  └────┴────┴────┴────┴────┘
     ↓
  TOS / OP-TEE (Trusted OS)
     ↓
  UEFI (replaces U-Boot on Orin)
     ↓  UEFI boot manager, A/B slot selection
  Kernel
     ↓
  Rootfs
```

### 4.2 Partition Map on QSPI NOR

```bash
# View QSPI partitions on a running Jetson
cat /proc/mtd

# Typical output for T234 with A/B:
# dev:    size   erasesize  name
# mtd0: 00080000 00010000 "BCT"
# mtd1: 00080000 00010000 "BCT_b"
# mtd2: 00080000 00010000 "mb1"
# mtd3: 00080000 00010000 "mb1_b"
# mtd4: 00100000 00010000 "mb2"
# mtd5: 00100000 00010000 "mb2_b"
# mtd6: 00400000 00010000 "uefi"
# mtd7: 00400000 00010000 "uefi_b"
# mtd8: 00200000 00010000 "bpmp-fw"
# mtd9: 00200000 00010000 "bpmp-fw_b"
# mtd10: 00100000 00010000 "spe-fw"
# mtd11: 00100000 00010000 "spe-fw_b"
# mtd12: 00100000 00010000 "rce-fw"
# mtd13: 00100000 00010000 "rce-fw_b"
# mtd14: 00100000 00010000 "tos"
# mtd15: 00100000 00010000 "tos_b"
# mtd16: 00040000 00010000 "eks"

# Note: eks has NO _b variant — it's shared between slots
```

### 4.3 QSPI NOR Flash Characteristics

```
QSPI NOR on Orin Nano:
  Chip:           Typically Macronix MX25U25635F or similar
  Size:           32 MB (256 Mbit)
  Erase block:    64 KB (0x10000)
  Page size:      256 bytes
  Write endurance: 100,000 erase cycles per block
  Data retention:  20 years at 85°C

Why this matters for OTA:
  - Erase is per-block (64 KB) — cannot erase a single byte
  - Write is per-page (256 bytes)
  - Erase-before-write is mandatory
  - Each OTA consumes 1 erase cycle per modified block
  - At quarterly updates: 4 cycles/year × 25 years = 100 cycles
    (well within 100,000 cycle limit)
  - At daily updates: 365/year × 10 years = 3,650 cycles
    (still within limit, but monitor wear)
```

### 4.4 MTD Operations for Manual QSPI Access

```bash
# CAUTION: Direct MTD operations bypass nv_update_engine safety checks
# Only use for debugging or recovery — never in production OTA flow

# Read a partition
sudo dd if=/dev/mtd2 of=/tmp/mb1_backup.bin bs=64k

# Erase a partition
sudo flash_erase /dev/mtd3 0 0

# Write a partition (after erase)
sudo dd if=mb1_new.bin of=/dev/mtd3 bs=64k

# Verify
sudo dd if=/dev/mtd3 of=/tmp/mb1_verify.bin bs=64k
md5sum /tmp/mb1_verify.bin mb1_new.bin
```

---

## 5. A/B Slot Orchestration for OTA

### 5.1 Slot State Machine

```
                    ┌──────────────┐
                    │  SLOT EMPTY  │
                    │ (factory)    │
                    └──────┬───────┘
                           │ Flash initial image
                           ▼
                    ┌──────────────┐
                    │  BOOTABLE    │
                    │ retry_count=7│
                    │ successful=0 │
                    └──────┬───────┘
                           │ Set as active, reboot
                           ▼
                    ┌──────────────┐
              ┌────►│  BOOTING     │◄────────────────────────┐
              │     │ retry_count--│                          │
              │     └──────┬───────┘                          │
              │            │                                  │
              │     ┌──────┴──────┐                           │
              │     ▼             ▼                           │
              │  Boot OK     Boot FAIL                        │
              │     │         retry_count > 0?                │
              │     │            │          │                  │
              │     │           Yes         No                │
              │     │            │          │                  │
              │     │            └──────────┘                  │
              │     │                       │                  │
              │     │                       ▼                  │
              │     │              ┌──────────────┐           │
              │     │              │  UNBOOTABLE  │           │
              │     │              │ retry_count=0│           │
              │     │              │ Switch slot  │           │
              │     │              └──────────────┘           │
              │     ▼                                         │
              │  Validation                                   │
              │  service runs                                 │
              │     │                                         │
              │  ┌──┴──┐                                      │
              │  ▼     ▼                                      │
              │ PASS  FAIL ──────────────────────────────────►│
              │  │     (mark-boot-unsuccessful)    reboot     │
              │  │                                            │
              │  ▼                                            │
              │ mark-boot-successful                          │
              │  │                                            │
              │  ▼                                            │
              │ ┌──────────────┐                              │
              │ │  SUCCESSFUL  │                              │
              │ │ successful=1 │                              │
              │ │ (stable)     │                              │
              └─┴──────────────┘
                  OTA writes to OTHER slot
```

### 5.2 nvbootctrl Deep Dive

```bash
# nvbootctrl operates on TWO partition tables:
#   1. Bootloader partitions (QSPI): nvbootctrl -t bootloader
#   2. Rootfs partitions (NVMe):     nvbootctrl -t rootfs

# Full slot dump — ALWAYS run this before and after OTA
sudo nvbootctrl -t bootloader dump-slots-info
sudo nvbootctrl -t rootfs dump-slots-info

# Example output:
# magic: 0x43424e00
# version: 3.1
# Features: 2 Slots, max retries: 7, A/B enabled
# Slot 0:
#   Priority:        15
#   Suffix:          _a
#   Retry count:     7
#   Boot successful: 1
# Slot 1:
#   Priority:        14
#   Suffix:          _b
#   Retry count:     7
#   Boot successful: 1
# Active slot: 0

# --- Key commands for OTA orchestration ---

# 1. Get current running slot
CURRENT=$(sudo nvbootctrl -t rootfs get-current-slot)
echo "Currently running slot: ${CURRENT}"

# 2. Determine target (inactive) slot
if [ "${CURRENT}" = "0" ]; then
    TARGET=1
else
    TARGET=0
fi
echo "OTA target slot: ${TARGET}"

# 3. After writing update to target slot:
sudo nvbootctrl -t rootfs set-active-boot-slot ${TARGET}
sudo nvbootctrl -t bootloader set-active-boot-slot ${TARGET}

# 4. Reboot into updated slot
sudo reboot

# 5. After successful boot on new slot:
sudo nvbootctrl -t rootfs mark-boot-successful
sudo nvbootctrl -t bootloader mark-boot-successful
```

### 5.3 Bootloader vs Rootfs Slot Synchronization

```
CRITICAL CONCEPT:
  Bootloader slots and rootfs slots are managed INDEPENDENTLY.
  They MUST be synchronized during OTA.

  If you update rootfs Slot B but bootloader remains on Slot A:
    → Slot A bootloader loads Slot B kernel
    → Driver version mismatch
    → GPU fails, CUDA broken, inference down

Correct OTA sequence:
  1. Write new rootfs to inactive rootfs slot
  2. Write new BUP to inactive bootloader slot (nv_update_engine)
  3. Set BOTH rootfs AND bootloader to inactive slot
  4. Reboot
  5. Validate
  6. Mark BOTH slots successful

Wrong OTA sequence (common mistake):
  1. Update rootfs only
  2. Forget bootloader
  3. Boot with mismatched bootloader+rootfs
  4. Subtle failures (GPU driver version mismatch, DTB wrong)
```

```bash
# Safe OTA slot switch script
#!/bin/bash
set -euo pipefail

CURRENT=$(sudo nvbootctrl -t rootfs get-current-slot)
TARGET=$((1 - CURRENT))

echo "Switching from slot ${CURRENT} to slot ${TARGET}"

# Set both partition tables to target slot
sudo nvbootctrl -t rootfs set-active-boot-slot ${TARGET}
sudo nvbootctrl -t bootloader set-active-boot-slot ${TARGET}

# Verify
echo "Rootfs active slot: $(sudo nvbootctrl -t rootfs get-active-boot-slot)"
echo "Bootloader active slot: $(sudo nvbootctrl -t bootloader get-active-boot-slot)"

# Reboot
sudo reboot
```

### 5.4 Retry Count Tuning

```
Default retry count: 7
  → System will attempt to boot the new slot 7 times before fallback

Why 7 may be wrong for your system:

  Too high (retry=7):
    If boot takes 60 seconds and fails at kernel panic:
    7 × 60s = 7 minutes of repeated failure before rollback
    → 7 minutes of downtime for safety-critical systems

  Too low (retry=1):
    Transient failures (NVMe init delay, race condition) cause
    immediate rollback even though slot is actually good
    → False rollback, stuck on old version

Recommendations:
  General edge AI:         retry=3 (balance of speed and tolerance)
  Safety-critical:         retry=2 (fast rollback, minimize downtime)
  Devices with slow boot:  retry=5 (allow for NVMe/network init delays)
  Development/testing:     retry=7 (default, more forgiving)
```

```bash
# Set retry count for the inactive slot before OTA
# (retry count is reset when set-active-boot-slot is called)
# The retry count is configured in the bootloader config

# Check current retry count
sudo nvbootctrl -t rootfs dump-slots-info | grep -i retry
```

---

## 6. Payload Generation and Packaging

### 6.1 Full OTA Payload Structure

A complete OTA payload for Jetson contains multiple sub-payloads:

```
OTA Payload (complete)
├── manifest.json          ← Version, hardware compat, checksums
├── manifest.json.sig      ← RSA/ECDSA signature of manifest
├── bl_update_payload      ← BUP (all QSPI firmware)
├── rootfs.ext4.zst        ← Compressed rootfs image
├── kernel_Image           ← Kernel binary
├── kernel_dtb.dtb         ← Device Tree Blob
└── post-update.sh         ← Post-install script (optional)
```

### 6.2 Building the Rootfs Image

```bash
# Method 1: From L4T BSP (stock)
cd Linux_for_Tegra
sudo ./apply_binaries.sh
sudo ./tools/l4t_create_default_user.sh -u myuser -p mypassword -a
# Create ext4 image from rootfs directory
sudo mke2fs -d rootfs -t ext4 -b 4096 rootfs.ext4 $((2 * 1024 * 1024))  # 2 GB

# Method 2: From Yocto build
bitbake myproject-image
# Output: tmp/deploy/images/jetson-orin-nano-devkit/myproject-image-*.ext4

# Method 3: Incremental from running device (development only)
# On the device, snapshot the current rootfs
sudo dd if=/dev/nvme0n1p1 bs=4M status=progress | zstd -T0 > rootfs-snapshot.ext4.zst
```

### 6.3 Manifest File Format

```json
{
    "version": "2024.03.15-r1",
    "hardware_compatibility": [
        "3767-0005-TS4",
        "3768-0000-TS1"
    ],
    "minimum_version": "2024.01.01-r1",
    "components": [
        {
            "name": "bootloader",
            "filename": "bl_update_payload",
            "sha256": "a1b2c3d4...",
            "size": 45678912,
            "type": "tegra-bup"
        },
        {
            "name": "rootfs",
            "filename": "rootfs.ext4.zst",
            "sha256": "e5f6a7b8...",
            "size": 524288000,
            "type": "raw-image",
            "target_partition": "APP",
            "compressed": "zstd"
        },
        {
            "name": "kernel",
            "filename": "kernel_Image",
            "sha256": "c9d0e1f2...",
            "size": 38912000,
            "type": "raw-image",
            "target_partition": "kernel"
        },
        {
            "name": "dtb",
            "filename": "kernel_dtb.dtb",
            "sha256": "a3b4c5d6...",
            "size": 262144,
            "type": "raw-image",
            "target_partition": "kernel-dtb"
        }
    ],
    "pre_checks": {
        "min_battery_pct": 50,
        "min_free_space_mb": 1024,
        "required_services": ["networkmanager"]
    }
}
```

### 6.4 Packaging Script

```bash
#!/bin/bash
# build_ota_payload.sh — Packages a complete OTA payload
set -euo pipefail

VERSION="${1:?Usage: $0 <version>}"
OUTPUT_DIR="ota-payload-${VERSION}"
SIGN_KEY="${OTA_SIGN_KEY:-keys/ota_priv.pem}"

mkdir -p "${OUTPUT_DIR}"

echo "=== Building OTA payload v${VERSION} ==="

# 1. Copy artifacts
cp payloads_t234/bl_update_payload "${OUTPUT_DIR}/"
cp rootfs.ext4.zst "${OUTPUT_DIR}/"
cp kernel/Image "${OUTPUT_DIR}/kernel_Image"
cp kernel/dtb/tegra234-p3767-0005-p3768-0000-a0.dtb "${OUTPUT_DIR}/kernel_dtb.dtb"

# 2. Generate checksums
cd "${OUTPUT_DIR}"
for f in bl_update_payload rootfs.ext4.zst kernel_Image kernel_dtb.dtb; do
    SHA=$(sha256sum "$f" | awk '{print $1}')
    SIZE=$(stat -c%s "$f")
    echo "  ${f}: sha256=${SHA}, size=${SIZE}"
done

# 3. Generate manifest (with computed checksums)
python3 ../scripts/generate_manifest.py \
    --version "${VERSION}" \
    --dir . \
    --output manifest.json

# 4. Sign the manifest
openssl dgst -sha256 -sign "${SIGN_KEY}" \
    -out manifest.json.sig manifest.json

# 5. Create final archive
cd ..
tar cf "${OUTPUT_DIR}.tar" "${OUTPUT_DIR}/"
echo "=== OTA payload ready: ${OUTPUT_DIR}.tar ==="
```

---

## 7. Cryptographic Signing and Verification

### 7.1 OTA Signing Architecture

```
Build Server (HSM-backed signing)
  │
  │  Private key: NEVER leaves HSM
  │  Signs: manifest, BUP, rootfs hash
  │
  ▼
OTA Server
  │
  │  Stores: signed artifacts
  │  Transport: HTTPS + certificate pinning
  │
  ▼
Device
  │
  │  Public key: burned into rootfs (/etc/ota/signing.pub.pem)
  │  Or: burned into fuses (for secure boot chain)
  │  Verifies: manifest signature, payload checksums
  │
  ▼
Apply update (only if all signatures valid)
```

### 7.2 Key Management

```bash
# Generate OTA signing key pair (do this ONCE, store private key in HSM)

# Option 1: RSA-4096 (widely supported, larger signatures)
openssl genrsa -out ota_priv.pem 4096
openssl rsa -in ota_priv.pem -pubout -out ota_pub.pem

# Option 2: ECDSA P-256 (smaller signatures, faster verification)
openssl ecparam -genkey -name prime256v1 -noout -out ota_priv.pem
openssl ec -in ota_priv.pem -pubout -out ota_pub.pem

# Install public key on device during manufacturing
# /etc/ota/signing.pub.pem — included in rootfs image

# Key rotation strategy:
#   - Primary key: current signing key
#   - Secondary key: next rotation key (pre-installed on devices)
#   - Rotate annually or on compromise
#   - Include BOTH keys on device for transition period
```

### 7.3 Signing the BUP for Secure Boot

```bash
# If secure boot fuses are burned, the BUP MUST be signed
# with the secure boot key (different from OTA signing key)

# Generate signed BUP during build
cd Linux_for_Tegra

# Sign with PKCS#11 (HSM) or PEM key
sudo ./l4t_generate_soc_bup.sh \
    --sign \
    --key rsa_priv.pem \
    --encrypt_key sym_key.key \
    t234

# The signed BUP will only be accepted by devices with matching
# public key in their fuses
```

### 7.4 Device-Side Verification Flow

```bash
#!/bin/bash
# /usr/bin/ota-verify.sh — Device-side payload verification
set -euo pipefail

PAYLOAD_DIR="$1"
PUB_KEY="/etc/ota/signing.pub.pem"

echo "Verifying OTA payload..."

# Step 1: Verify manifest signature
openssl dgst -sha256 -verify "${PUB_KEY}" \
    -signature "${PAYLOAD_DIR}/manifest.json.sig" \
    "${PAYLOAD_DIR}/manifest.json"
echo "✓ Manifest signature valid"

# Step 2: Parse manifest and verify each component checksum
python3 -c "
import json, hashlib, sys

with open('${PAYLOAD_DIR}/manifest.json') as f:
    manifest = json.load(f)

for comp in manifest['components']:
    path = '${PAYLOAD_DIR}/' + comp['filename']
    expected = comp['sha256']
    actual = hashlib.sha256(open(path, 'rb').read()).hexdigest()
    if actual != expected:
        print(f'FAIL: {comp[\"name\"]} checksum mismatch')
        sys.exit(1)
    print(f'✓ {comp[\"name\"]}: checksum valid')
"

# Step 3: Verify hardware compatibility
python3 -c "
import json, subprocess

with open('${PAYLOAD_DIR}/manifest.json') as f:
    manifest = json.load(f)

# Read device hardware ID
boardid = open('/proc/device-tree/nvidia,boardids', 'rb').read().decode().strip('\x00')
compat = manifest.get('hardware_compatibility', [])
# Simplified check — production would be more thorough
print(f'Device board: {boardid}')
print(f'Compatible boards: {compat}')
"

echo "=== All verification checks passed ==="
```

---

## 8. Rootfs Image-Based OTA

### 8.1 Writing Rootfs to Inactive Slot

```bash
#!/bin/bash
# ota-apply-rootfs.sh — Write rootfs image to inactive slot
set -euo pipefail

IMAGE="$1"  # e.g., rootfs.ext4.zst

# Determine inactive partition
CURRENT_SLOT=$(sudo nvbootctrl -t rootfs get-current-slot)
if [ "${CURRENT_SLOT}" = "0" ]; then
    TARGET_DEV="/dev/nvme0n1p2"    # APP_b
    TARGET_SLOT=1
else
    TARGET_DEV="/dev/nvme0n1p1"    # APP
    TARGET_SLOT=0
fi

echo "Current slot: ${CURRENT_SLOT}"
echo "Target device: ${TARGET_DEV}"

# Pre-flight checks
FREE_SPACE_MB=$(df /tmp --output=avail -B1M | tail -1 | tr -d ' ')
if [ "${FREE_SPACE_MB}" -lt 512 ]; then
    echo "ERROR: Insufficient temp space (${FREE_SPACE_MB} MB < 512 MB)"
    exit 1
fi

# Ensure target is not mounted
if mountpoint -q "${TARGET_DEV}" 2>/dev/null; then
    echo "ERROR: Target partition is mounted"
    exit 1
fi

# Write compressed image directly to target partition
echo "Writing rootfs to ${TARGET_DEV}..."
zstd -d "${IMAGE}" --stdout | sudo dd of="${TARGET_DEV}" bs=4M status=progress conv=fsync

# Verify write integrity
echo "Verifying write..."
EXPECTED_HASH=$(zstd -d "${IMAGE}" --stdout | sha256sum | awk '{print $1}')
ACTUAL_HASH=$(sudo dd if="${TARGET_DEV}" bs=4M count=$(( $(stat -c%s "${IMAGE%.zst}") / 4194304 + 1 )) \
    status=none | sha256sum | awk '{print $1}')

if [ "${EXPECTED_HASH}" != "${ACTUAL_HASH}" ]; then
    echo "ERROR: Write verification failed!"
    echo "Expected: ${EXPECTED_HASH}"
    echo "Actual:   ${ACTUAL_HASH}"
    exit 1
fi

echo "✓ Rootfs written and verified on slot ${TARGET_SLOT}"
```

### 8.2 Streaming OTA (No Local Storage Required)

```bash
# For devices with limited storage, stream directly from server to partition
# No need to download the entire image first

CURRENT_SLOT=$(sudo nvbootctrl -t rootfs get-current-slot)
TARGET_DEV="/dev/nvme0n1p$((CURRENT_SLOT == 0 ? 2 : 1))"

# Stream HTTPS → decompress → write to partition
curl -fsSL "https://ota.example.com/v2024.03/rootfs.ext4.zst" \
    | zstd -d \
    | sudo dd of="${TARGET_DEV}" bs=4M conv=fsync

# Advantages:
#   - No temporary storage needed
#   - Works even when rootfs partition is nearly full
#   - Single-pass operation

# Disadvantages:
#   - Cannot retry partial download (must restart from beginning)
#   - Cannot verify checksum before writing
#   - Network interruption wastes all progress

# Hybrid approach: download with resume, then write
curl -fsSL -C - -o /data/ota/rootfs.ext4.zst \
    "https://ota.example.com/v2024.03/rootfs.ext4.zst"
# Verify checksum
echo "${EXPECTED_SHA256}  /data/ota/rootfs.ext4.zst" | sha256sum -c -
# Then write
zstd -d /data/ota/rootfs.ext4.zst --stdout | sudo dd of="${TARGET_DEV}" bs=4M conv=fsync
```

### 8.3 Rootfs Size Optimization for OTA

```
Full JetPack rootfs:          12-16 GB  → OTA impractical over cellular
Minimal L4T rootfs:           1.5-2 GB  → OTA feasible over WiFi
Yocto minimal + NVIDIA stack: 800 MB    → OTA feasible over LTE
Compressed (zstd):            400-600 MB → OTA practical over most networks

Optimization techniques:
  1. Remove desktop packages (saves 3-4 GB)
  2. Remove documentation and man pages (saves 200 MB)
  3. Remove unused locales (saves 100 MB)
  4. Strip debug symbols from libraries (saves 500 MB)
  5. Remove sample code and NVIDIA examples (saves 300 MB)
  6. Use Yocto minimal image instead of stock L4T

OTA transfer time estimates (compressed 500 MB image):
  Network          Bandwidth    Transfer Time
  ─────────────────────────────────────────────
  Gigabit Ethernet  100 MB/s     5 seconds
  WiFi 5 (AC)       50 MB/s      10 seconds
  WiFi 4 (N)        10 MB/s      50 seconds
  LTE Cat 6         5 MB/s       100 seconds
  LTE Cat 4         1.5 MB/s     5.5 minutes
  LTE Cat M1        0.05 MB/s    2.7 hours
```

---

## 9. Bootloader + Firmware OTA

### 9.1 Complete Bootloader Update Procedure

```bash
#!/bin/bash
# ota-apply-bootloader.sh — Apply BUP to inactive bootloader slot
set -euo pipefail

BUP_FILE="$1"

echo "=== Bootloader OTA Update ==="

# Step 1: Verify BUP integrity
echo "Verifying BUP payload..."
# nv_update_engine does internal verification, but pre-check hash
echo "${EXPECTED_BUP_SHA256}  ${BUP_FILE}" | sha256sum -c -

# Step 2: Check current slot state
echo "Current bootloader slot info:"
sudo nvbootctrl -t bootloader dump-slots-info

# Step 3: Apply BUP
echo "Applying BUP to inactive slot..."
sudo nv_update_engine -i "${BUP_FILE}" -e

# Step 4: Verify
echo "Verifying bootloader update..."
sudo nv_update_engine -e --verify

echo "=== Bootloader update complete ==="
echo "DO NOT REBOOT YET — complete rootfs update first"
```

### 9.2 UEFI Capsule Updates (JetPack 6.x)

JetPack 6.x introduces UEFI Capsule Update support, aligning Jetson with the UEFI specification:

```
UEFI Capsule Update flow:

  1. Place capsule file in EFI System Partition:
     /boot/efi/EFI/UpdateCapsule/TEGRA_BL.Cap

  2. Set UEFI variable to trigger update:
     EFI_OS_INDICATIONS → EFI_OS_INDICATIONS_FILE_CAPSULE_DELIVERY_SUPPORTED

  3. Reboot

  4. UEFI firmware processes capsule before booting OS:
     - Reads capsule from ESP
     - Validates signature
     - Applies to inactive slot
     - Boots into updated slot

  Advantages:
    - Standard UEFI mechanism (not NVIDIA-proprietary)
    - Processed before OS boot (safer)
    - Integrated with UEFI Secure Boot validation

  Current status (JetPack 6.x):
    - Supported but not widely adopted
    - nv_update_engine remains the primary method
    - Capsule support may become the default in future JetPack
```

### 9.3 Firmware Version Tracking

```bash
# Track firmware versions for each component
# Essential for fleet management — know exactly what's deployed

# UEFI version
sudo cat /sys/firmware/efi/efivars/TegraPlatformSpec-* 2>/dev/null || \
    echo "Check UEFI log in serial console"

# L4T version (kernel + BSP)
cat /etc/nv_tegra_release
# Example: # R36 (release), REVISION: 4.0, ...

# Kernel version
uname -r
# Example: 5.15.136-tegra

# NVIDIA driver version
cat /proc/driver/nvidia/version 2>/dev/null || \
    dpkg -l nvidia-l4t-core 2>/dev/null | grep ^ii

# JetPack version
apt list --installed 2>/dev/null | grep nvidia-jetpack
# Or: cat /etc/nv_tegra_release

# Bootloader slot versions
sudo nvbootctrl -t bootloader dump-slots-info

# Comprehensive version report
cat << 'EOF' > /usr/local/bin/version-report.sh
#!/bin/bash
echo "=== Jetson Version Report ==="
echo "Hostname: $(hostname)"
echo "Date: $(date -u)"
echo "L4T: $(head -1 /etc/nv_tegra_release)"
echo "Kernel: $(uname -r)"
echo "Boot slot (rootfs): $(sudo nvbootctrl -t rootfs get-current-slot)"
echo "Boot slot (bl): $(sudo nvbootctrl -t bootloader get-current-slot)"
echo "Rootfs UUID: $(findmnt / -o UUID -n)"
echo "NVMe model: $(cat /sys/block/nvme0n1/device/model 2>/dev/null)"
echo "Temperature: $(cat /sys/class/thermal/thermal_zone0/temp)m°C"
echo "Uptime: $(uptime -p)"
EOF
chmod +x /usr/local/bin/version-report.sh
```

---

## 10. Kernel and DTB OTA

### 10.1 Kernel Partition Update

```bash
# Kernel is stored in a separate partition (not inside rootfs)
# on NVMe/eMMC — this allows kernel update independent of rootfs

# Find kernel partitions
sudo fdisk -l /dev/nvme0n1 | grep -i kernel
# Or check by partition label:
ls -la /dev/disk/by-partlabel/ | grep kernel

# Typical layout:
#   kernel    → /dev/nvme0n1p2 (Slot A kernel)
#   kernel_b  → /dev/nvme0n1p5 (Slot B kernel)
#   kernel-dtb   → /dev/nvme0n1p3 (Slot A DTB)
#   kernel-dtb_b → /dev/nvme0n1p6 (Slot B DTB)

# Update kernel on inactive slot
CURRENT_SLOT=$(sudo nvbootctrl -t rootfs get-current-slot)
if [ "${CURRENT_SLOT}" = "0" ]; then
    KERNEL_DEV="/dev/disk/by-partlabel/kernel_b"
    DTB_DEV="/dev/disk/by-partlabel/kernel-dtb_b"
else
    KERNEL_DEV="/dev/disk/by-partlabel/kernel"
    DTB_DEV="/dev/disk/by-partlabel/kernel-dtb"
fi

# Write new kernel
sudo dd if=Image of="${KERNEL_DEV}" bs=64k conv=fsync
# Write new DTB
sudo dd if=tegra234-p3767-0005.dtb of="${DTB_DEV}" bs=64k conv=fsync
```

### 10.2 Kernel Module Consistency

```
CRITICAL: Kernel modules MUST match the kernel binary exactly.

The kernel binary is on the kernel partition.
Kernel modules are inside the rootfs at /lib/modules/<version>/.

If you update the kernel partition but not the rootfs:
  → modprobe fails for ALL NVIDIA drivers
  → No GPU, no CUDA, no cameras, no display
  → System boots but is non-functional

If you update the rootfs but not the kernel partition:
  → Old kernel loads new modules → symbol mismatch
  → Kernel oops/panic on module load
  → System may not boot at all

Rule: ALWAYS update kernel + rootfs + DTB as an atomic unit.
```

### 10.3 Device Tree Overlay OTA

```bash
# For minor hardware configuration changes, you can update
# device tree overlays without a full rootfs update

# DTB overlays are stored in:
#   /boot/tegra234-p3767-0005-p3768-0000-a0.dtbo (example)

# Apply overlay at runtime (JetPack 6.x with UEFI):
# 1. Place overlay in /boot/
# 2. Update extlinux.conf or UEFI boot config:
#    FDT /boot/tegra234-p3767-0005.dtb
#    FDTOVERLAYS /boot/my-custom-overlay.dtbo

# CAUTION: DTB overlay changes are NOT covered by A/B
# unless the overlay is part of the rootfs image
# Best practice: include overlays in rootfs image for A/B protection
```

---

## 11. Container-Based Application OTA

### 11.1 Layered Update Architecture

```
┌─────────────────────────────────────────────────┐
│ Layer 3: Application Containers (daily/weekly)  │
│  ┌─────────┐ ┌─────────┐ ┌─────────┐           │
│  │ Inference│ │ Sensor  │ │ Comms   │           │
│  │ Engine  │ │ Pipeline│ │ Agent   │           │
│  └─────────┘ └─────────┘ └─────────┘           │
├─────────────────────────────────────────────────┤
│ Layer 2: Base Container (monthly)               │
│  ┌──────────────────────────────────────┐       │
│  │ l4t-tensorrt:r36.4.0                │       │
│  │ CUDA 12.x + cuDNN + TensorRT       │       │
│  └──────────────────────────────────────┘       │
├─────────────────────────────────────────────────┤
│ Layer 1: Host OS — Rootfs (quarterly)           │
│  Minimal L4T + Docker + nvidia-container-toolkit│
│  Updated via A/B rootfs OTA                     │
├─────────────────────────────────────────────────┤
│ Layer 0: Bootloader / Firmware (rare)           │
│  MB1, MB2, UEFI, BPMP, SPE, RCE                │
│  Updated via nv_update_engine + A/B             │
└─────────────────────────────────────────────────┘
```

### 11.2 Container Update Flow

```bash
#!/bin/bash
# container-ota.sh — Update application containers
set -euo pipefail

REGISTRY="registry.mycompany.com"
APP_IMAGE="${REGISTRY}/edge-ai/inference:latest"
COMPOSE_FILE="/opt/myapp/docker-compose.yml"

# Step 1: Pre-pull new image (while old container still running)
echo "Pulling new container image..."
docker pull "${APP_IMAGE}"

# Step 2: Verify image signature (using Docker Content Trust)
export DOCKER_CONTENT_TRUST=1
docker trust inspect "${APP_IMAGE}" || {
    echo "ERROR: Image signature verification failed"
    exit 1
}

# Step 3: Run pre-update health check
docker exec myapp-inference /usr/local/bin/healthcheck.sh || {
    echo "WARNING: Current container unhealthy — proceeding anyway"
}

# Step 4: Rolling update (zero-downtime if configured)
cd /opt/myapp
docker compose pull
docker compose up -d --remove-orphans

# Step 5: Post-update health check
sleep 10  # Wait for container startup
MAX_RETRIES=6
for i in $(seq 1 ${MAX_RETRIES}); do
    if docker exec myapp-inference /usr/local/bin/healthcheck.sh; then
        echo "✓ New container is healthy"
        break
    fi
    if [ "$i" = "${MAX_RETRIES}" ]; then
        echo "ERROR: New container failed health check — rolling back"
        docker compose down
        # Restore previous image
        docker tag "${APP_IMAGE}-previous" "${APP_IMAGE}"
        docker compose up -d
        exit 1
    fi
    echo "Waiting for container health... (attempt ${i}/${MAX_RETRIES})"
    sleep 10
done

# Step 6: Clean up old images
docker image prune -f --filter "until=72h"
echo "=== Container update complete ==="
```

### 11.3 Model Update (Hot-Swap)

```bash
# AI models can be updated independently of the container
# Store models on the persistent DATA partition

# Model directory structure:
# /data/models/
#   ├── current -> v2024.03.15/
#   ├── v2024.03.15/
#   │   ├── model.engine          (TensorRT engine)
#   │   ├── model.onnx            (source model)
#   │   ├── calibration.cache     (INT8 calibration)
#   │   └── manifest.json         (version, input/output spec)
#   └── v2024.02.01/
#       └── ...                   (previous version, kept for rollback)

# Download new model
MODEL_VERSION="v2024.03.15"
MODEL_URL="https://models.mycompany.com/${MODEL_VERSION}/model.engine"

mkdir -p "/data/models/${MODEL_VERSION}"
curl -fsSL "${MODEL_URL}" -o "/data/models/${MODEL_VERSION}/model.engine"

# Verify checksum
echo "${EXPECTED_SHA}  /data/models/${MODEL_VERSION}/model.engine" | sha256sum -c -

# Atomic symlink switch
ln -sfn "${MODEL_VERSION}" /data/models/current.tmp
mv -T /data/models/current.tmp /data/models/current

# Signal application to reload model
docker exec myapp-inference kill -SIGHUP 1
# Or: application watches /data/models/current symlink via inotify

echo "Model updated to ${MODEL_VERSION}"
```

---

## 12. Delta Updates — Engineering for Bandwidth

### 12.1 Why Delta Updates Matter

```
Scenario: Fleet of 5,000 Jetson devices on LTE Cat 4 (1.5 MB/s)

Full image OTA (500 MB compressed):
  Per device:     500 MB / 1.5 MB/s = 5.5 minutes
  Fleet total:    5,000 × 500 MB = 2.44 TB bandwidth
  Cost (at $10/GB): $24,400
  Time (staged):  ~48 hours at 50 concurrent updates

Delta OTA (25 MB for minor patch):
  Per device:     25 MB / 1.5 MB/s = 17 seconds
  Fleet total:    5,000 × 25 MB = 122 GB bandwidth
  Cost (at $10/GB): $1,220
  Time (staged):  ~4 hours at 50 concurrent updates

  Savings: 95% bandwidth, 95% cost, 92% time
```

### 12.2 Delta Update Methods

```
Method 1: Binary diff (bsdiff/bspatch)
  ────────────────────────────────────
  Input:   old rootfs image + new rootfs image
  Output:  binary delta patch
  Apply:   Read old partition → apply patch → write to inactive partition

  Pros:  Smallest delta size, well-understood
  Cons:  Requires old image on host to generate delta
         Device needs both slots accessible
         Memory-intensive on device (holds both images)
         O(n) memory where n = image size

  Generation:
    bsdiff old-rootfs.ext4 new-rootfs.ext4 rootfs.patch
    # Typical: 15-50 MB for minor updates

  Application (on device):
    bspatch /dev/nvme0n1p1 /dev/nvme0n1p2 rootfs.patch
    # Reads slot A, applies delta, writes slot B


Method 2: Block-level diff (casync / content-addressable)
  ─────────────────────────────────────────────────────
  Input:   new rootfs image → split into content-addressed chunks
  Output:  chunk store (only new/changed chunks transferred)
  Apply:   Download missing chunks → reassemble image

  Pros:  Efficient for shared content across versions
         Resumable downloads
         Server-side deduplication
  Cons:  Requires chunk store infrastructure
         Higher complexity

  Generation:
    casync make --store=chunk-store/ rootfs.caidx rootfs.ext4
    # Only new chunks are uploaded to CDN

  Application (on device):
    casync extract --store=https://cdn.example.com/chunks/ \
        rootfs.caidx /dev/nvme0n1p2


Method 3: File-level diff (mender-artifact delta)
  ────────────────────────────────────────────────
  Input:   old and new rootfs images
  Output:  Mender delta artifact
  Apply:   Mender client applies delta

  Pros:  Integrated with Mender fleet management
  Cons:  Commercial feature (requires Mender Professional)


Method 4: Package-level diff (NOT recommended for A/B)
  ──────────────────────────────────────────────────
  apt/dpkg upgrade — only changed packages transferred
  Breaks A/B invariant, not recommended for production
```

### 12.3 Implementing casync-Based Delta Updates

```bash
# === On the build server ===

# Generate chunk index and store for new rootfs
casync make \
    --store=casync-store/ \
    --chunk-size-min=32768 \
    --chunk-size-avg=65536 \
    --chunk-size-max=131072 \
    rootfs-v2024.03.caidx \
    rootfs-v2024.03.ext4

# Upload store to CDN (only new chunks)
aws s3 sync casync-store/ s3://ota-chunks/ --size-only

# Upload chunk index
aws s3 cp rootfs-v2024.03.caidx s3://ota-manifests/

# === On the Jetson device ===

# Install casync
apt install casync  # or include in Yocto image

CURRENT_SLOT=$(sudo nvbootctrl -t rootfs get-current-slot)
TARGET_DEV="/dev/nvme0n1p$((CURRENT_SLOT == 0 ? 2 : 1))"
SOURCE_DEV="/dev/nvme0n1p$((CURRENT_SLOT + 1))"

# Apply delta — casync reads existing chunks from current slot,
# downloads only missing/changed chunks from CDN
sudo casync extract \
    --store=https://cdn.example.com/ota-chunks/ \
    --seed="${SOURCE_DEV}" \
    https://cdn.example.com/ota-manifests/rootfs-v2024.03.caidx \
    "${TARGET_DEV}"

# casync automatically:
#   1. Reads chunk index from server
#   2. Checks which chunks exist locally (in current rootfs)
#   3. Downloads only missing chunks
#   4. Assembles complete image on target partition
```

---

## 13. OTA Frameworks — SWUpdate, Mender, RAUC

### 13.1 Framework Comparison for Jetson

```
                    SWUpdate          Mender             RAUC
─────────────────────────────────────────────────────────────────────
License             GPL-2.0           Apache/Commercial   LGPL-2.1
Yocto layer         meta-swupdate     meta-mender         meta-rauc
Server              hawkBit/custom    Mender Server       Custom
Delta updates       Yes (zchunk)      Yes (commercial)    Yes (casync)
Tegra BUP support   Custom handler    Custom module       Custom handler
Signed updates      Yes (RSA/CMS)     Yes (RSA)           Yes (CMS)
Encrypted updates   Yes               No                  No
Web UI on device    Yes (mongoose)    No                  No
Fleet management    Via hawkBit       Built-in            Custom
A/B slot control    Custom script     Built-in            Built-in
Production scale    25,000+ proven    25,000+ proven      10,000+ proven
Community size      Large             Large               Medium
Jetson adoption     High              Medium              Low
```

### 13.2 SWUpdate for Jetson — Complete Integration

```bash
# === SWUpdate with custom Tegra handler ===

# 1. Install SWUpdate on device (in Yocto image)
IMAGE_INSTALL:append = " swupdate swupdate-www"

# 2. Create custom Tegra handler for BUP updates
# meta-myproject/recipes-support/swupdate/files/tegra_handler.c

# The handler interfaces between SWUpdate and nv_update_engine:
#   - Receives BUP from SWUpdate pipeline
#   - Writes to temp file
#   - Calls nv_update_engine -i <bup_file> -e
#   - Reports success/failure back to SWUpdate
```

```c
/* tegra_handler.c — SWUpdate handler for NVIDIA BUP updates */
#include <swupdate/handler.h>
#include <swupdate/util.h>

static int tegra_bup_handler(struct img_type *img, void *data)
{
    char cmd[512];
    int ret;

    /* Write BUP to temporary file */
    char bup_path[] = "/tmp/bl_update_payload_XXXXXX";
    int fd = mkstemp(bup_path);
    if (fd < 0) {
        ERROR("Failed to create temp file for BUP");
        return -1;
    }

    /* Copy image data to temp file */
    ret = copyimage(&fd, img, NULL);
    close(fd);
    if (ret < 0) {
        ERROR("Failed to write BUP to temp file");
        unlink(bup_path);
        return -1;
    }

    /* Apply BUP via nv_update_engine */
    snprintf(cmd, sizeof(cmd),
             "nv_update_engine -i %s -e 2>&1", bup_path);

    ret = system(cmd);
    unlink(bup_path);

    if (ret != 0) {
        ERROR("nv_update_engine failed with code %d", ret);
        return -1;
    }

    TRACE("BUP applied successfully");
    return 0;
}

__attribute__((constructor))
void tegra_handler_init(void)
{
    register_handler("tegra-bup", tegra_bup_handler,
                     PARTITION_HANDLER, NULL);
}
```

```json
// sw-description for complete Jetson OTA (rootfs + bootloader)
{
    "software": {
        "version": "2024.03.15",
        "hardware-compatibility": ["orin-nano-8gb-v2"],
        "orin-nano": {
            "images": [
                {
                    "filename": "rootfs.ext4.zst",
                    "type": "raw",
                    "device": "/dev/nvme0n1p2",
                    "compressed": "zstd",
                    "installed-directly": true,
                    "sha256": "@rootfs.ext4.zst"
                },
                {
                    "filename": "bl_update_payload",
                    "type": "tegra-bup",
                    "sha256": "@bl_update_payload"
                }
            ],
            "scripts": [
                {
                    "filename": "post-update.sh",
                    "type": "shellscript",
                    "sha256": "@post-update.sh"
                }
            ]
        }
    }
}
```

### 13.3 Mender for Jetson

```bash
# === Mender integration with Jetson Orin ===

# 1. Add meta-mender to Yocto build
# bblayers.conf:
BBLAYERS += "meta-mender/meta-mender-core"
BBLAYERS += "meta-mender/meta-mender-demo"  # remove for production

# 2. Configure in local.conf:
INHERIT += "mender-full"
MENDER_DEVICE_TYPE = "jetson-orin-nano"
MENDER_SERVER_URL = "https://mender.mycompany.com"
MENDER_TENANT_TOKEN = "your-token-here"

# Partition configuration for A/B
MENDER_STORAGE_DEVICE = "/dev/nvme0n1"
MENDER_ROOTFS_PART_A = "/dev/nvme0n1p2"
MENDER_ROOTFS_PART_B = "/dev/nvme0n1p3"
MENDER_DATA_PART = "/dev/nvme0n1p4"

# 3. Create Mender artifact from Yocto build output
mender-artifact write rootfs-image \
    --device-type jetson-orin-nano \
    --artifact-name "release-2024.03" \
    --file tmp/deploy/images/*/myproject-image-*.ext4

# 4. Upload to Mender server
mender-artifact upload release-2024.03.mender \
    --server https://mender.mycompany.com

# 5. Create deployment (staged rollout)
# Via Mender web UI or API:
curl -X POST https://mender.mycompany.com/api/management/v1/deployments \
    -H "Authorization: Bearer ${TOKEN}" \
    -d '{
        "name": "Release 2024.03",
        "artifact_name": "release-2024.03",
        "devices": ["device-id-1", "device-id-2"],
        "phases": [
            {"batch_size": 5, "delay": 86400},
            {"batch_size": 20},
            {"batch_size": 100}
        ]
    }'
```

### 13.4 RAUC for Jetson

```ini
# /etc/rauc/system.conf — RAUC system configuration for Jetson

[system]
compatible=mycompany-orin-nano-v2
bootloader=custom
mountprefix=/mnt/rauc

[keyring]
path=/etc/rauc/keyring.pem

[handlers]
bootloader-custom-backend=/usr/lib/rauc/tegra-boot-handler.sh

[slot.rootfs.0]
device=/dev/nvme0n1p1
type=ext4
bootname=A

[slot.rootfs.1]
device=/dev/nvme0n1p2
type=ext4
bootname=B

[slot.bootloader.0]
device=/dev/mtd0
type=raw
parent=rootfs.0

[slot.bootloader.1]
device=/dev/mtd1
type=raw
parent=rootfs.1
```

```bash
#!/bin/bash
# /usr/lib/rauc/tegra-boot-handler.sh
# Custom boot handler for RAUC to interface with nvbootctrl

case "$1" in
    get-primary)
        SLOT=$(sudo nvbootctrl -t rootfs get-current-slot)
        echo "rootfs.${SLOT}"
        ;;
    set-primary)
        SLOT_NAME="$2"  # e.g., "rootfs.1"
        SLOT_NUM="${SLOT_NAME##*.}"
        sudo nvbootctrl -t rootfs set-active-boot-slot "${SLOT_NUM}"
        sudo nvbootctrl -t bootloader set-active-boot-slot "${SLOT_NUM}"
        ;;
    get-state)
        # Check if slot is marked successful
        INFO=$(sudo nvbootctrl -t rootfs dump-slots-info)
        # Parse and return "good" or "bad"
        ;;
esac
```

---

## 14. NVIDIA L4T OTA via Debian Packages

### 14.1 When apt-Based OTA Is Appropriate

```
apt-based OTA (nvidia-l4t-* packages) is ONLY appropriate when:
  ✓ A/B partitioning is NOT enabled
  ✓ Device count is small (< 50)
  ✓ Physical access is available for recovery
  ✓ Downtime is acceptable
  ✓ Update is a minor L4T patch (e.g., 36.3 → 36.4)

apt-based OTA is NOT appropriate when:
  ✗ A/B is enabled (breaks A/B invariant)
  ✗ Fleet devices (no physical recovery access)
  ✗ Safety-critical systems
  ✗ Major version upgrades (e.g., JetPack 5 → 6)
  ✗ Production systems that cannot tolerate downtime
```

### 14.2 L4T apt OTA Process

```bash
# Step 1: Add NVIDIA L4T repository (usually pre-configured)
cat /etc/apt/sources.list.d/nvidia-l4t-apt-source.list
# deb https://repo.download.nvidia.com/jetson/common r36.4 main
# deb https://repo.download.nvidia.com/jetson/t234 r36.4 main

# Step 2: Check available updates
sudo apt-get update
apt list --upgradable 2>/dev/null | grep nvidia-l4t

# Step 3: Apply minor L4T update
# IMPORTANT: This updates the RUNNING rootfs in place
sudo apt-get upgrade nvidia-l4t-core nvidia-l4t-cuda \
    nvidia-l4t-multimedia nvidia-l4t-camera

# Step 4: If bootloader update is included
# NVIDIA packages may trigger nv_update_engine automatically
# Check: dpkg -L nvidia-l4t-bootloader | grep postinst
sudo apt-get upgrade nvidia-l4t-bootloader

# Step 5: Reboot
sudo reboot

# What can go wrong:
#   - apt upgrade interrupted (power fail, SSH disconnect)
#     → Partially applied packages → inconsistent system
#   - Kernel updated but modules not → boot failure
#   - Bootloader updated but rootfs not → mismatch
#   - No automatic rollback mechanism
```

### 14.3 L4T OTA Package Dependencies

```
nvidia-l4t-bootloader
  ├── Contains: MB1, MB2, UEFI, BPMP-FW, SPE-FW, etc.
  ├── postinst: runs nv_update_engine to flash QSPI
  └── Triggers: reboot required

nvidia-l4t-core
  ├── Contains: core NVIDIA drivers, nvgpu.ko
  └── Depends on: matching kernel version

nvidia-l4t-kernel
  ├── Contains: Linux kernel Image
  └── Depends on: nvidia-l4t-kernel-dtbs

nvidia-l4t-kernel-dtbs
  ├── Contains: Device Tree Blobs
  └── Must match: kernel version + bootloader version

nvidia-l4t-cuda
  ├── Contains: CUDA runtime libraries
  └── Depends on: nvidia-l4t-core (driver version)

nvidia-l4t-multimedia
  ├── Contains: nvbufsurface, nvargus, multimedia stack
  └── Depends on: nvidia-l4t-core

# The dependency chain means: updating one package
# often requires updating all packages to maintain consistency
```

---

## 15. Fleet-Scale OTA Deployment

### 15.1 Fleet Architecture

```
┌────────────────────────────────────────────────────────┐
│                   OTA Control Plane                    │
│                                                        │
│  ┌──────────┐  ┌──────────┐  ┌──────────────────────┐ │
│  │ Artifact │  │ Rollout  │  │ Device Registry      │ │
│  │ Storage  │  │ Engine   │  │                      │ │
│  │          │  │          │  │ 25,000 devices       │ │
│  │ S3/GCS   │  │ Staged   │  │ Per-device state     │ │
│  │ + CDN    │  │ rollouts │  │ Group membership     │ │
│  │          │  │ Auto-halt│  │ Hardware variants    │ │
│  └──────────┘  └──────────┘  └──────────────────────┘ │
│                                                        │
│  ┌──────────┐  ┌──────────┐  ┌──────────────────────┐ │
│  │ Metrics  │  │ Alert    │  │ Dashboard            │ │
│  │ Ingestion│  │ Manager  │  │                      │ │
│  │          │  │          │  │ Fleet health         │ │
│  │ Per-device│  │ Rollout  │  │ Update progress     │ │
│  │ telemetry│  │ failures │  │ Rollback events      │ │
│  └──────────┘  └──────────┘  └──────────────────────┘ │
└─────────────────────────┬──────────────────────────────┘
                          │ HTTPS + mTLS + cert pinning
                          │
        ┌─────────────────┼──────────────────┐
        │                 │                  │
        ▼                 ▼                  ▼
  ┌───────────┐    ┌───────────┐     ┌───────────┐
  │ Region A  │    │ Region B  │     │ Region C  │
  │ 8,000 dev │    │ 12,000 dev│     │ 5,000 dev │
  │           │    │           │     │           │
  │ CDN edge  │    │ CDN edge  │     │ CDN edge  │
  │ node      │    │ node      │     │ node      │
  └───────────┘    └───────────┘     └───────────┘
```

### 15.2 Staged Rollout Strategy

```
Stage 0: Lab Validation (pre-rollout)
  ───────────────────────────────
  Target:  5 internal test devices
  Duration: 48 hours
  Gate:    100% success, no anomalies in logs
  Auto-halt: Any failure → stop rollout

Phase 1: Canary (0.1%)
  ───────────────────────────────
  Target:  25 devices (diverse hardware variants)
  Duration: 72 hours
  Monitoring: Crash rate, inference latency, temperature
  Gate:    < 0.5% failure rate
  Auto-halt: > 2 rollbacks in this phase

Phase 2: Early Adopters (5%)
  ───────────────────────────────
  Target:  1,250 devices
  Duration: 48 hours
  Monitoring: All Phase 1 metrics + bandwidth usage
  Gate:    < 0.2% failure rate
  Auto-halt: > 10 rollbacks total

Phase 3: Wide Release (50%)
  ───────────────────────────────
  Target:  12,500 devices
  Duration: 72 hours
  Rate limit: 200 concurrent updates (avoid CDN spike)
  Gate:    < 0.1% failure rate
  Auto-halt: > 25 rollbacks total

Phase 4: Full Fleet (100%)
  ───────────────────────────────
  Target:  remaining 11,225 devices
  Duration: until complete
  Rate limit: 500 concurrent updates
  Stragglers: retry offline devices for 14 days
```

### 15.3 Device Grouping

```bash
# Devices are grouped by multiple dimensions:

# 1. Hardware variant
#    - orin-nano-8gb-devkit
#    - orin-nano-8gb-carrier-v2
#    - orin-nano-4gb-carrier-v2

# 2. Deployment region
#    - us-east, us-west, eu-central, apac

# 3. Risk tier
#    - canary (always gets updates first)
#    - standard (normal rollout)
#    - conservative (gets updates last, after full validation)

# 4. Connectivity
#    - ethernet (no bandwidth constraints)
#    - wifi (moderate constraints)
#    - lte (strict bandwidth, use delta updates)

# 5. Operational window
#    - 24x7 (can update anytime)
#    - business-hours-only (update outside 8am-6pm)
#    - maintenance-window (Sunday 2am-6am only)

# Group assignment is stored in device registry
# and sent to device during provisioning
```

### 15.4 Bandwidth Management at Scale

```
Problem: 25,000 devices × 500 MB = 12.2 TB of bandwidth per full OTA

Mitigation strategies:

  1. CDN with edge caching
     - CloudFront / Cloudflare / Akamai
     - Artifacts cached at edge PoPs
     - Devices download from nearest edge
     - Cost: $0.02-0.08/GB vs $0.09/GB direct

  2. Rate limiting
     - Max 200-500 concurrent downloads
     - Prevents CDN/server overload
     - Prevents network saturation at customer sites

  3. Delta updates (see Section 12)
     - 90-95% bandwidth savings for minor updates
     - Full image only for major version changes

  4. P2P update distribution (advanced)
     - Devices on same LAN share chunks
     - One device downloads, others get from LAN peer
     - Frameworks: apt-p2p concept, custom implementation
     - Saves WAN bandwidth at sites with many devices

  5. Scheduled downloads
     - Download during off-hours (2am-5am)
     - Apply during maintenance window
     - Separate download and apply phases

  6. Resume support
     - HTTP Range requests for interrupted downloads
     - Never re-download completed chunks
     - Critical for unreliable cellular connections
```

---

## 16. Rollback Mechanisms and Failure Recovery

### 16.1 Automatic Rollback via Boot Counter

```bash
# The boot counter mechanism — how automatic rollback works

# BEFORE OTA:
#   Slot A: active, successful=1, retry=7
#   Slot B: inactive, successful=1, retry=7

# AFTER OTA (slot B updated, set as active):
#   Slot A: inactive, successful=1, retry=7   ← fallback target
#   Slot B: active, successful=0, retry=7     ← booting this

# First boot attempt into Slot B:
#   retry count: 7 → 6
#   If boot succeeds and mark-boot-successful runs:
#     Slot B: active, successful=1, retry=7   ← stable
#   If boot fails (panic, hang, validation fail):
#     retry count decrements each attempt: 6, 5, 4, 3, 2, 1, 0
#     After retry=0: slot marked UNBOOTABLE
#     System falls back to Slot A
#     Slot A boots, system is on known-good image

# Entire rollback flow:
# Boot 1: Slot B → fail → retry=6
# Boot 2: Slot B → fail → retry=5
# Boot 3: Slot B → fail → retry=4
# ...
# Boot 7: Slot B → fail → retry=0 → UNBOOTABLE
# Boot 8: Slot A → success → system recovered
```

### 16.2 Application-Level Rollback

```bash
#!/bin/bash
# /etc/systemd/system/ota-validation.service
# [Unit]
# Description=Post-OTA validation and rollback trigger
# After=multi-user.target
# Wants=network-online.target
#
# [Service]
# Type=oneshot
# ExecStart=/usr/local/bin/ota-validate.sh
# TimeoutStartSec=120
# RemainAfterExit=yes
#
# [Install]
# WantedBy=multi-user.target
```

```bash
#!/bin/bash
# /usr/local/bin/ota-validate.sh
set -euo pipefail

LOG="/var/log/ota-validation.log"
exec &> >(tee -a "${LOG}")

echo "=== OTA Validation Started $(date -u) ==="

CHECKS_PASSED=0
CHECKS_TOTAL=0

check() {
    local name="$1"
    shift
    CHECKS_TOTAL=$((CHECKS_TOTAL + 1))
    if "$@" > /dev/null 2>&1; then
        echo "✓ ${name}"
        CHECKS_PASSED=$((CHECKS_PASSED + 1))
    else
        echo "✗ ${name} FAILED"
    fi
}

# Critical checks — any failure triggers rollback
check "GPU accessible"         nvidia-smi
check "CUDA functional"        python3 -c "import ctypes; ctypes.CDLL('libcudart.so')"
check "TensorRT loadable"      python3 -c "import tensorrt"
check "Camera device exists"   test -e /dev/video0
check "NVMe healthy"           nvme smart-log /dev/nvme0 2>/dev/null
check "DNS resolution"         host google.com
check "NTP synchronized"       timedatectl show -p NTPSynchronized --value | grep -q yes
check "Inference service"      systemctl is-active myapp-inference
check "Watchdog running"       systemctl is-active watchdog

echo "Checks passed: ${CHECKS_PASSED}/${CHECKS_TOTAL}"

if [ "${CHECKS_PASSED}" -lt "${CHECKS_TOTAL}" ]; then
    echo "VALIDATION FAILED — triggering rollback"
    # Do NOT mark boot successful — retry count will decrement
    # System will automatically roll back after retries exhaust
    exit 1
fi

echo "VALIDATION PASSED — marking boot successful"
sudo nvbootctrl -t rootfs mark-boot-successful
sudo nvbootctrl -t bootloader mark-boot-successful

# Report success to OTA server
curl -sf -X POST "https://ota.mycompany.com/api/v1/devices/$(cat /etc/machine-id)/status" \
    -H "Content-Type: application/json" \
    -d '{"status": "update_confirmed", "version": "'"$(cat /etc/ota-version)"'"}'

echo "=== OTA Validation Complete ==="
```

### 16.3 Manual Rollback

```bash
# Force rollback to previous slot (without reboot-cycling)

# Check current state
sudo nvbootctrl -t rootfs dump-slots-info
sudo nvbootctrl -t bootloader dump-slots-info

# Switch to other slot
CURRENT=$(sudo nvbootctrl -t rootfs get-current-slot)
TARGET=$((1 - CURRENT))

echo "Rolling back from slot ${CURRENT} to slot ${TARGET}"

# Ensure target slot is bootable
sudo nvbootctrl -t rootfs set-active-boot-slot ${TARGET}
sudo nvbootctrl -t bootloader set-active-boot-slot ${TARGET}

# Reboot into previous version
sudo reboot
```

### 16.4 Recovery When Both Slots Fail

```
If both Slot A and Slot B are unbootable:

  Option 1: USB Recovery Mode
  ─────────────────────────────
  1. Power off device
  2. Hold RECOVERY button + power on (or jumper REC pin)
  3. Connect USB-C to host machine
  4. Host detects device in APX mode:
       lsusb | grep NVIDIA  # should show 0955:7523 or similar
  5. Reflash:
       cd Linux_for_Tegra
       sudo ./flash.sh jetson-orin-nano-devkit internal

  Option 2: Recovery Partition (if configured)
  ─────────────────────────────────────────────
  - A minimal recovery kernel/rootfs on a third partition
  - Boots automatically when both A/B slots fail
  - Provides SSH access for manual repair
  - Can pull and apply a new OTA image

  Option 3: External Boot Media
  ─────────────────────────────
  - Boot from SD card or USB drive
  - Mount NVMe partitions
  - Repair or reflash rootfs partitions
  - Requires physical access

  Prevention:
  - ALWAYS test OTA images on staging devices first
  - Never update both slots simultaneously
  - Keep serial console accessible on production hardware
  - Monitor fleet for devices stuck in boot loops
```

---

## 17. OTA Security Threat Model

### 17.1 Threat Matrix

```
Threat                         Attack Vector               Mitigation
──────────────────────────────────────────────────────────────────────
Malicious firmware             Compromised OTA server      Code signing + HSM
Man-in-the-middle              Network interception        mTLS + cert pinning
Replay attack                  Re-send old (vulnerable)    Version monotonicity
                               firmware                    + anti-rollback fuse
Downgrade attack               Force older version         Minimum version check
Denial of service              Flood with invalid updates  Rate limiting + auth
Supply chain compromise        Malicious build artifact    Reproducible builds
Key compromise                 Stolen signing key          HSM + key rotation
Local privilege escalation     Attacker on device writes   Signed BUP verification
                               to QSPI directly            in nv_update_engine
Partial update corruption      Power fail mid-write        A/B atomicity
```

### 17.2 Anti-Rollback Protection

```bash
# Anti-rollback prevents downgrade attacks by enforcing
# monotonically increasing version numbers

# Fuse-based anti-rollback (HARDWARE — irreversible):
# T234 has dedicated anti-rollback fuses
# Each fuse burn increments the minimum acceptable bootloader version
# Bootloader version < fuse count → boot blocked by BootROM

# Software-based anti-rollback:
# Store minimum version in persistent storage (DATA partition)
# Check during OTA before applying

CURRENT_VERSION=$(cat /etc/ota-version)    # e.g., "2024.03.15"
MIN_VERSION=$(cat /data/ota/min-version)   # e.g., "2024.01.01"
NEW_VERSION="${1}"                         # from OTA manifest

# Version comparison (date-based versioning)
if [[ "${NEW_VERSION}" < "${MIN_VERSION}" ]]; then
    echo "REJECTED: version ${NEW_VERSION} < minimum ${MIN_VERSION}"
    echo "This may be a downgrade attack"
    exit 1
fi

# After successful update, advance minimum version
echo "${NEW_VERSION}" > /data/ota/min-version
```

### 17.3 Secure Transport

```bash
# Device-to-server communication for OTA

# 1. Certificate pinning
# Pin the OTA server's TLS certificate (or CA) in the update agent
# Prevents MITM even if device CA store is compromised

# curl with certificate pinning:
curl --cacert /etc/ota/ca-bundle.pem \
     --pinnedpubkey "sha256//YhAMBk2DMHO2kzE2bMra..." \
     https://ota.mycompany.com/api/v1/updates

# 2. Mutual TLS (mTLS)
# Device presents its client certificate to the server
# Server verifies device identity before serving updates

curl --cert /etc/ota/device.crt \
     --key /etc/ota/device.key \
     --cacert /etc/ota/ca-bundle.pem \
     https://ota.mycompany.com/api/v1/updates

# 3. Device identity
# Each device has a unique identity provisioned during manufacturing:
#   - X.509 client certificate (from device PKI)
#   - Hardware-bound key (from Jetson Security Engine / OP-TEE)
#   - Used for: mTLS, update authorization, telemetry authentication
```

---

## 18. Power-Fail Safe OTA Design

### 18.1 Power Failure Analysis

```
Where can power fail during OTA, and what happens?

Phase                    Power Fail Impact           Recovery
──────────────────────────────────────────────────────────────────
Downloading payload      No damage                   Resume download
Verifying signature      No damage                   Re-verify
Writing rootfs to        Inactive slot corrupt       Re-run OTA
  inactive slot          Active slot SAFE             (active slot OK)
Writing BUP via          Inactive BL slot corrupt    Re-run nv_update_engine
  nv_update_engine       Active BL slot SAFE          (active slot OK)
Switching active slot    Atomic operation —           Boot into whichever
  (nvbootctrl)           either old or new active     slot is marked active
Rebooting                Normal reboot               Boot normally
Validation running       Retry count decrements      May trigger rollback
                                                      if retry exhausts
Marking boot successful  Retry count not cleared     Next reboot: retry
                         — auto-rollback possible     or rollback

CRITICAL WINDOW:
  The only dangerous moment is if both slots have been written
  to and neither is verified. This should NEVER happen because:
  - OTA writes to INACTIVE slot only
  - Active slot is NEVER modified during OTA
  - At worst: inactive slot is corrupt → system stays on active slot
```

### 18.2 UPS and Power Management for OTA

```bash
# For battery-powered or UPS-backed devices:

# Check power before starting OTA
check_power_for_ota() {
    # Check battery level (if applicable)
    BATTERY_PCT=$(cat /sys/class/power_supply/battery/capacity 2>/dev/null || echo 100)
    if [ "${BATTERY_PCT}" -lt 50 ]; then
        echo "Battery too low (${BATTERY_PCT}%) — deferring OTA"
        return 1
    fi

    # Check AC power
    AC_ONLINE=$(cat /sys/class/power_supply/*/online 2>/dev/null | head -1 || echo 1)
    if [ "${AC_ONLINE}" != "1" ]; then
        echo "AC power not connected — deferring OTA"
        return 1
    fi

    return 0
}

# Inhibit system sleep/suspend during OTA
systemd-inhibit --what=sleep:shutdown:idle \
    --who="OTA Update" \
    --why="System update in progress" \
    /usr/local/bin/ota-apply.sh
```

---

## 19. OTA Testing and Validation

### 19.1 OTA Test Matrix

```
Every OTA release must pass these tests before fleet deployment:

Test Category          Test Cases                              Pass Criteria
──────────────────────────────────────────────────────────────────────────────
Basic functionality    Fresh update from v(N-1) to vN          Boots, runs app
                       Fresh update from v(N-2) to vN          Boots, runs app
                       Update from v(N-1) + rollback           Returns to v(N-1)

Power failure          Power cut during rootfs write            Active slot OK
                       Power cut during BUP write               Active slot OK
                       Power cut during reboot                  Boots into one slot
                       Power cut during validation              Rollback works

Network failure        Disconnect during download               Resume works
                       Disconnect during streaming OTA          Active slot OK
                       Slow network (throttle to 100 kbps)     Completes eventually
                       DNS failure during update check          Graceful retry

Slot management        A→B update + verify                     Slot B active
                       B→A update + verify                     Slot A active
                       A→B update, B fails, rollback to A      Slot A active
                       Both slots used, update again            Correct slot target

Security               Unsigned payload                        Rejected
                       Wrong-key signed payload                 Rejected
                       Tampered payload (bit flip)              Rejected
                       Downgrade attempt                        Rejected
                       Expired certificate                      Rejected

Edge cases             Full disk during download                Cleanup + retry
                       Concurrent update requests               Serialized
                       Update during active inference           No inference drop
                       Watchdog fires during validation         Rollback works
                       Clock skew (NTP not synced)              Update succeeds
```

### 19.2 Automated OTA Testing

```bash
#!/bin/bash
# ota-test-harness.sh — Automated OTA test on hardware-in-the-loop
set -euo pipefail

DEVICE_IP="${1}"
OTA_PAYLOAD="${2}"
SSH_KEY="~/.ssh/jetson_test"

ssh_cmd() { ssh -i "${SSH_KEY}" "user@${DEVICE_IP}" "$@"; }
scp_cmd() { scp -i "${SSH_KEY}" "$@"; }

echo "=== OTA Test Harness ==="

# Record pre-update state
PRE_VERSION=$(ssh_cmd "cat /etc/ota-version")
PRE_SLOT=$(ssh_cmd "sudo nvbootctrl -t rootfs get-current-slot")
echo "Pre-update: version=${PRE_VERSION}, slot=${PRE_SLOT}"

# Deploy OTA payload
scp_cmd "${OTA_PAYLOAD}" "user@${DEVICE_IP}:/tmp/ota-payload.tar"

# Trigger update
ssh_cmd "cd /tmp && tar xf ota-payload.tar && sudo /usr/local/bin/ota-apply.sh ota-payload/"

# Wait for reboot
echo "Waiting for device to reboot..."
sleep 30
for i in $(seq 1 60); do
    if ssh_cmd "echo online" 2>/dev/null; then
        break
    fi
    sleep 5
done

# Verify post-update state
POST_VERSION=$(ssh_cmd "cat /etc/ota-version")
POST_SLOT=$(ssh_cmd "sudo nvbootctrl -t rootfs get-current-slot")
BOOT_SUCCESS=$(ssh_cmd "sudo nvbootctrl -t rootfs dump-slots-info" | grep -A2 "Slot ${POST_SLOT}" | grep successful)

echo "Post-update: version=${POST_VERSION}, slot=${POST_SLOT}"
echo "Boot status: ${BOOT_SUCCESS}"

# Validate
if [ "${POST_VERSION}" != "${PRE_VERSION}" ] && [ "${POST_SLOT}" != "${PRE_SLOT}" ]; then
    echo "✓ OTA update successful"
else
    echo "✗ OTA update FAILED"
    exit 1
fi

# Run application health checks
ssh_cmd "/usr/local/bin/ota-validate.sh"
echo "✓ Post-update validation passed"
```

### 19.3 Chaos Testing for OTA

```bash
# Simulate failure conditions to validate OTA robustness

# 1. Power-fail simulation (requires smart power switch)
# Use a network-controlled PDU (Power Distribution Unit)
# or relay-controlled power supply

test_power_fail_during_write() {
    # Start OTA update
    ssh_cmd "sudo /usr/local/bin/ota-apply.sh /tmp/payload/ &"
    # Wait random time during write phase (5-30 seconds)
    sleep $((RANDOM % 25 + 5))
    # Cut power
    pdu_control off ${DEVICE_OUTLET}
    sleep 5
    # Restore power
    pdu_control on ${DEVICE_OUTLET}
    # Wait for boot
    sleep 60
    # Verify device boots on active (previous) slot
    verify_device_boots "${DEVICE_IP}"
}

# 2. Network interruption during download
test_network_fail_during_download() {
    # Start download
    ssh_cmd "curl -o /tmp/rootfs.ext4.zst https://ota.example.com/rootfs.ext4.zst &"
    sleep 10
    # Drop network
    ssh_cmd "sudo iptables -A OUTPUT -p tcp --dport 443 -j DROP"
    sleep 30
    # Restore network
    ssh_cmd "sudo iptables -D OUTPUT -p tcp --dport 443 -j DROP"
    # Verify download resumes
    ssh_cmd "curl -C - -o /tmp/rootfs.ext4.zst https://ota.example.com/rootfs.ext4.zst"
}

# 3. Corrupt inactive slot to test rollback
test_rollback_on_corrupt_rootfs() {
    INACTIVE=$((1 - $(ssh_cmd "sudo nvbootctrl -t rootfs get-current-slot")))
    INACTIVE_DEV="/dev/nvme0n1p$((INACTIVE + 1))"
    # Write garbage to inactive slot
    ssh_cmd "sudo dd if=/dev/urandom of=${INACTIVE_DEV} bs=1M count=10"
    # Set inactive as active
    ssh_cmd "sudo nvbootctrl -t rootfs set-active-boot-slot ${INACTIVE}"
    ssh_cmd "sudo reboot"
    # Wait and verify rollback to original slot
    sleep 120
    CURRENT=$(ssh_cmd "sudo nvbootctrl -t rootfs get-current-slot")
    if [ "${CURRENT}" != "${INACTIVE}" ]; then
        echo "✓ Rollback successful"
    else
        echo "✗ Rollback failed — device booted corrupt slot"
    fi
}
```

---

## 20. Production OTA Monitoring and Telemetry

### 20.1 Device-Side Telemetry

```bash
#!/bin/bash
# /usr/local/bin/ota-telemetry.sh
# Reports OTA status and device health to fleet server

FLEET_URL="https://fleet.mycompany.com/api/v1/telemetry"
DEVICE_ID=$(cat /etc/machine-id)

report_telemetry() {
    local payload
    payload=$(cat << EOF
{
    "device_id": "${DEVICE_ID}",
    "timestamp": "$(date -u +%Y-%m-%dT%H:%M:%SZ)",
    "ota_version": "$(cat /etc/ota-version 2>/dev/null || echo unknown)",
    "boot_slot": "$(sudo nvbootctrl -t rootfs get-current-slot)",
    "boot_successful": $(sudo nvbootctrl -t rootfs dump-slots-info | grep -c "Boot successful: 1"),
    "uptime_seconds": $(cut -d. -f1 /proc/uptime),
    "rootfs_usage_pct": $(df / --output=pcent | tail -1 | tr -d ' %'),
    "temperature_mc": $(cat /sys/class/thermal/thermal_zone0/temp),
    "memory_available_kb": $(grep MemAvailable /proc/meminfo | awk '{print $2}'),
    "gpu_utilization_pct": $(cat /sys/devices/gpu.0/load 2>/dev/null || echo 0),
    "last_ota_status": "$(cat /data/ota/last-status 2>/dev/null || echo none)",
    "last_ota_timestamp": "$(cat /data/ota/last-timestamp 2>/dev/null || echo none)",
    "rollback_count": $(cat /data/ota/rollback-count 2>/dev/null || echo 0)
}
EOF
)

    curl -sf -X POST "${FLEET_URL}" \
        -H "Content-Type: application/json" \
        -H "X-Device-ID: ${DEVICE_ID}" \
        --cert /etc/ota/device.crt \
        --key /etc/ota/device.key \
        -d "${payload}" || true
}

report_telemetry
```

### 20.2 Fleet Dashboard Metrics

```
Key metrics to track during rollouts:

  Real-time:
  ─────────
  - Update progress: % of fleet on target version
  - Active downloads: concurrent download count
  - Rollback events: count in last 1h/24h
  - Failed updates: devices stuck in update state
  - Device unreachable: devices not reporting telemetry

  Per-rollout:
  ────────────
  - Success rate: successful updates / total attempts
  - Average update time: download + apply + reboot + validation
  - Bandwidth consumed: total GB transferred
  - Rollback rate: devices that rolled back / total updated
  - Time to full fleet: hours from rollout start to 100%

  Historical:
  ───────────
  - Updates per device per quarter
  - Rollback frequency by version
  - Mean time to recover (after failed update)
  - Devices requiring manual intervention (truck rolls)

  Alert thresholds:
  ─────────────────
  - Rollback rate > 1% in any phase: PAUSE rollout
  - > 10 devices unreachable after update: INVESTIGATE
  - Any device in boot loop > 30 min: ALERT on-call
  - Bandwidth spike > 2× expected: CHECK for update loop
```

---

## 21. Field Failure Case Studies

### 21.1 Case Study: QSPI Corruption During Power Cycle

```
Symptom:
  300 devices deployed at traffic intersections
  After OTA update, 3 devices (1%) stopped booting

Root cause:
  Power supply at some intersections had brief brownouts
  Brownout occurred during nv_update_engine QSPI write
  Inactive slot partially written (corrupted)
  Active slot still good — but nvbootctrl already set
  inactive as active before the write completed

  The bug: OTA script called set-active-boot-slot BEFORE
  nv_update_engine finished writing

Timeline:
  1. OTA script starts nv_update_engine (background)
  2. Script immediately calls set-active-boot-slot (wrong!)
  3. Power brownout during nv_update_engine write
  4. Device reboots into partially written (corrupt) slot
  5. Boot fails, retry count exhausts
  6. Falls back to old slot — but old slot's bootloader
     was also being updated (both BL slots shared EKS)
  7. EKS corrupted → both slots fail → brick

Fix:
  1. NEVER set-active-boot-slot before write completes
  2. Wait for nv_update_engine exit code
  3. Verify write with --verify flag
  4. THEN switch slot
  5. Add UPS or supercapacitor hold-up for QSPI write window

Corrected script:
  sudo nv_update_engine -i payload -e && \       # wait for completion
  sudo nv_update_engine -e --verify && \          # verify write
  sudo nvbootctrl set-active-boot-slot ${TARGET}  # THEN switch
```

### 21.2 Case Study: Bootloop from Watchdog vs Validation Race

```
Symptom:
  After OTA, 15% of devices rolled back to previous version
  Devices were functionally correct on new version
  Rollback was unnecessary (false negative)

Root cause:
  Hardware watchdog timeout: 60 seconds
  Boot time after OTA: 45 seconds (kernel + systemd)
  Validation service start: boot + 40 seconds = 85 seconds
  mark-boot-successful: boot + 50 seconds = 95 seconds

  Timeline:
    T=0:   Power on, kernel starts
    T=45:  systemd reaches multi-user.target
    T=60:  WATCHDOG FIRES — device reboots
    T=85:  (would have been) validation service starts
    T=95:  (would have been) mark-boot-successful

  Watchdog rebooted the device BEFORE validation could run.
  After 3 reboots: retry count exhausted → rollback

Fix:
  1. Increase watchdog timeout to 180 seconds (3 minutes)
  2. Start validation service earlier (After=basic.target)
  3. Pet the watchdog from an early-boot service while
     waiting for full validation to complete
  4. Profile boot time BEFORE deploying OTA

  # Watchdog timeout adjustment:
  # /etc/watchdog.conf
  watchdog-timeout = 180
  # Or in systemd:
  # RuntimeWatchdogSec=180
```

### 21.3 Case Study: Delta Update Applied to Wrong Base Version

```
Symptom:
  50 devices received delta update, 12 failed to boot

Root cause:
  Delta update was generated against rootfs v2024.02
  12 devices were still on v2024.01 (missed previous update)
  bspatch applied delta to wrong base → corrupted rootfs

  v2024.01 + (v2024.02 → v2024.03 delta) = garbage
  Expected: v2024.02 + (v2024.02 → v2024.03 delta) = v2024.03

Fix:
  1. ALWAYS verify base version before applying delta
  2. Include base version hash in delta manifest
  3. Check hash of current rootfs partition matches expected base
  4. Fallback: if base mismatch, download full image instead

  # Pre-delta verification:
  CURRENT_HASH=$(dd if=/dev/nvme0n1p1 bs=4M | sha256sum | cut -d' ' -f1)
  EXPECTED_BASE="${MANIFEST_BASE_HASH}"
  if [ "${CURRENT_HASH}" != "${EXPECTED_BASE}" ]; then
      echo "Base version mismatch — falling back to full image"
      download_full_image
  fi
```

### 21.4 Case Study: Certificate Expiration Bricked Fleet

```
Symptom:
  On January 15, 2025, all 8,000 devices stopped checking for updates
  Devices functioned normally but could not receive security patches

Root cause:
  OTA server TLS certificate expired
  Certificate was issued January 15, 2024 (1-year validity)
  No monitoring on certificate expiration
  Device certificate pinning rejected the new (auto-renewed) cert

Fix:
  1. Monitor certificate expiration (alert at 30, 14, 7 days)
  2. Use longer-validity CA certificates (5-10 years)
  3. Include backup CA in device trust store
  4. Implement certificate rotation OTA (update trust store)
  5. Never pin leaf certificate — pin intermediate CA instead

  Prevention:
    - Automated cert renewal (Let's Encrypt / internal PKI)
    - Pin CA certificate, not server leaf certificate
    - Include certificate rotation in OTA update testing
    - Calendar alerts for all certificates in the OTA chain
```

---

## 22. References

* [NVIDIA Jetson Linux Developer Guide — Over-the-Air Update](https://docs.nvidia.com/jetson/archives/r36.4/DeveloperGuide/SD/SoftwarePackagesAndTheUpdateMechanism.html) — official OTA documentation
* [NVIDIA Jetson Linux Developer Guide — Bootloader Update](https://docs.nvidia.com/jetson/archives/r36.4/DeveloperGuide/SD/Bootloader/UpdateAndRedundancy.html) — nv_update_engine and BUP
* [NVIDIA Jetson Linux Developer Guide — Disk Encryption and Security](https://docs.nvidia.com/jetson/archives/r36.4/DeveloperGuide/SD/Security.html) — secure boot and signing
* [SWUpdate Documentation](https://sbabic.github.io/swupdate/) — SWUpdate framework
* [Mender Documentation](https://docs.mender.io/) — Mender OTA platform
* [RAUC Documentation](https://rauc.readthedocs.io/) — RAUC update framework
* [meta-tegra](https://github.com/OE4T/meta-tegra) — Yocto/OE layer for Tegra
* [casync](https://github.com/systemd/casync) — content-addressable data synchronization
* Main guide: [Nvidia Jetson Platform Guide](../Guide.md)
* A/B deep dive: [Orin Nano Rootfs & A/B Redundancy](../Orin-Nano-Rootfs-and-AB-Redundancy/Guide.md)
* Security deep dive: [Orin Nano Security](../Orin-Nano-Security/Guide.md)
* Yocto BSP: [Orin Nano Yocto BSP Production](../Orin-Nano-Yocto-BSP-Production/Guide.md)
