# Lecture 22: Embedded Storage and OTA Partitioning

Choose storage by workload and recovery requirements, not by a single headline bandwidth. A robot may load a model occasionally, record large camera streams continuously, and update configuration in small durable writes. These stress different parts of a device.

Read [filesystem durability](Lecture-21.md) and [PCIe/NVMe](Lecture-20.md) first. All partition examples here are conceptual; no command reformats a host device.

## Interfaces do not determine a device benchmark

| Family | Common interface | What must be verified on the actual product |
|---|---|---|
| eMMC | Managed NAND over an MMC interface | Supported mode, controller, queueing, capacity, endurance, power-loss behavior |
| UFS | Managed storage using MIPI transport and SCSI-derived commands | Version, lanes/gears, logical units, queue resources, thermal behavior |
| NVMe | Storage controller, commonly over PCIe | Negotiated link, queue limits, namespace, firmware and workload behavior |

HS400 eMMC's interface ceiling is not a promise that every device writes 400 MB/s. eMMC 5.1 includes command-queueing support; “all eMMC accepts only one command” is wrong. UFS permits multiple outstanding commands, but queue depth does not equal physically simultaneous NAND operations or eliminate head-of-line delays. Device/host support must be checked.

On a Jetson, module storage and carrier-board connectors differ. An Orin Nano development kit should not be described as if it necessarily contains an eMMC module with the same partition layout as every other Jetson. M.2 connector size, keying, supported lanes, power, and boot support are board-specific.

## eMMC hardware regions and ordinary partitions

Managed eMMC can expose a user area, boot regions, RPMB, and optional configured regions. Their sizes and configuration come from device capabilities and fields such as EXT_CSD. Do not hard-code BOOT0/BOOT1 or RPMB as 4 MiB on every device. Linux may expose boot areas as separate block devices; GPT partitions inside the user area are a different level.

UFS uses logical-unit configuration and its own descriptors. Reusing the exact eMMC BOOT0/BOOT1/UDA diagram for every UFS device hides important differences. Inspect the selected hardware documentation and Linux driver interface.

## What RPMB authenticates

Replay Protected Memory Block storage provides authenticated operations and a device-managed write counter. An authenticated write supplies the expected **current** counter; successful processing advances it. A generic rule “accept any counter greater than or equal to current” would undermine the protocol. Key provisioning and counter exhaustion behavior belong to the specific standard/device.

RPMB authentication is not encryption: it does not make stored bytes confidential automatically. An OS can use authenticated storage as one ingredient in rollback protection, but firmware authorization, monotonic version policy, secure key handling, and the boot chain must all use it correctly. RPMB does not by itself protect every partition or make every downgrade impossible.

The [OP-TEE secure-storage documentation](https://optee.readthedocs.io/en/latest/architecture/secure_storage.html) explains how a trusted OS can use RPMB storage. This lesson does not provision keys or fuse a device. A key's location and threat model are deployment facts, not consequences of the word “RPMB.”

## Workload arithmetic before a purchase or layout

Three 10 MB/s camera streams plus 2 MB/s of telemetry produce 32 MB/s of logical application payload. At continuous duty, that is `32 × 86400 = 2,764,800 MB/day`, or about 2.765 TB/day in decimal units, before filesystem/device overhead. At 25% recording duty it is about 0.691 TB/day.

A 128 GB usable log region holds only `128000 / 32 = 4000 s`, about 66.7 minutes, if overhead and reserved space are ignored. This retention calculation does not tell you write endurance. Bounded rotation, event selection, compression cost, and upload policy change both requirements.

Use the vendor's rated endurance and its workload assumptions when available. P/E-cycle estimates are a model, not a substitute for a product TBW/DWPD specification. There is no universal JEDEC “enterprise eMMC = 16 PBW” rating.

## Write amplification has explicit boundaries

Define `A_fs = host block bytes / application bytes` and `A_ftl = physical NAND bytes / host block bytes`, using matching intervals and counters. Then total physical/application ratio is `A_fs × A_ftl`, when those counters are available and consistently defined.

**Illustrative model:** at 0.691 TB/day application traffic, `A_fs=1.2` and `A_ftl=1.5` imply about `0.691×1.2×1.5=1.244 TB/day` of physical writes. These factors are assumptions, not measurements for ext4 or F2FS. Compression, caching, metadata, garbage collection, device fullness, and counter definitions can change the interpretation.

A hypothetical raw physical budget of 64 TB divided by 50 MB/s gives about 14.8 days at amplification one and continuous duty. That arithmetic does not establish a real device's endurance: capacity, spare area, NAND properties, rated workload, retention, and firmware matter. F2FS and discard cannot promise a particular multiplier or lifetime.

## A/B updates as a state machine

```text
active slot A is confirmed healthy
    → receive and authenticate candidate B
    → write/verify B without damaging the known-good boot path
    → persist a bounded trial-boot selection
    → boot B and evaluate explicit health criteria
    → confirm B, or revert according to the tested boot policy
```

Counted boot attempts, firmware/rootfs pairing, and health confirmation need a single coherent protocol. The count “three” is an example policy, not a universal BSP constant. Shared writable data can break rollback even when the old rootfs still boots: a new database schema may be unreadable by old software.

Recovery images, A/B slots, content-addressed deployment systems, and filesystem snapshots are different building blocks. They do not automatically solve authentication, rollback policy, power-loss consistency, or recovery from corrupted metadata. A bootloader must know how to select the actual artifacts the update system produced.

Study the existing [OTA store exercise](../../../Phase%204%20-%20Track%20B%20-%20Nvidia%20Jetson/1.%20Nvidia%20Jetson%20Platform/Orin-Nano-OTA-Deep-Dive/Guide.md), which tests a file-based update protocol and process-interruption points. It explicitly does not simulate NVIDIA firmware or physical power cuts. Use the appropriate BSP's documented layout and flashing process for a board milestone.

## Read-only inspection and useful evidence

```bash
lsblk -o NAME,TYPE,SIZE,FSTYPE,MOUNTPOINTS
findmnt /
cat /proc/partitions
```

These identify visible block devices and mounts without changing partitions. Device names such as `/dev/mmcblk0` or `/dev/nvme0n1` are not stable proof of identity. Record model, serial where appropriate, capacity, firmware, and the exact board storage role before any later deployment procedure.

`iostat -x` can report queue and timing statistics. `await` includes time waiting and being serviced; it is not pure device service time. `%util` is based on observed busy time and does not universally measure remaining throughput headroom for parallel devices. Correlate with queue depth, workload latency, errors, and thermal state.

Health information varies: eMMC EXT_CSD lifetime estimates, UFS health descriptors, and NVMe health logs have different units and granularity. A fixed sysfs path may not exist. Preserve raw values and the device-specific interpretation; coarse lifetime bins cannot reveal exact remaining days.

**Check:** Can a rootfs rollback undo an irreversible shared-data migration? **Answer:** Not automatically. Data compatibility needs its own migration and rollback design.

**Check:** Does a 4 GB/s sequential-read result prove a 50 MB/s logger meets durable-write deadlines? **Answer:** No. Write behavior, fsync, sustained fullness, thermal state, and tail latency are different measurements.

**Check:** Does authenticated RPMB data imply encrypted RPMB data? **Answer:** No. Authentication and confidentiality are separate properties.
