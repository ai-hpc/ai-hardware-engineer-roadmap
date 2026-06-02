# Lecture 21: Filesystems: ext4, btrfs, F2FS & overlayfs

## Overview

A filesystem is the layer between your application and raw storage. It is easy to treat filesystems as invisible infrastructure — you open a file, write data, close it, and assume the data persists. But the choice of filesystem has direct, measurable consequences for AI hardware systems: it determines how fast camera frames land on storage, how long a reboot takes after a crash, whether a failed OTA update bricks a device or safely rolls back, and how long an eMMC chip lasts before wearing out.

The mental model to carry through this lecture is the **layered storage stack**: applications talk to the VFS (Virtual File System) layer, which provides a uniform API. Below VFS sits the actual filesystem implementation (ext4, btrfs, F2FS). Below that is the block layer, which routes I/O to the physical device. Each layer makes decisions that affect the layers above it.

AI hardware engineers need to understand filesystems because edge AI devices (dashcams, robotics controllers) write continuously to flash storage, which wears out. OTA update reliability depends on filesystem snapshot and rollback capabilities. Containerized inference deployments use overlayfs for immutable base images. And the IPC mechanism that ties together all of openpilot's processes runs on a RAM-backed filesystem (tmpfs).

---

## Filesystem Role and VFS

A filesystem organizes files into directories, provides **persistence across power cycles**, manages free space, and ensures **metadata consistency after crashes**.

### VFS (Virtual File System)

Linux VFS is an **abstraction layer** that presents a **uniform API** regardless of the underlying filesystem implementation.

The layered architecture looks like this:

```
Application
    │  open() read() write() mmap() fsync()
    ▼
┌─────────────────────────────────────────┐
│          VFS (Virtual File System)       │
│  struct file_operations dispatch table  │
│  dentry cache (name → inode)            │
│  inode cache (per-file metadata)        │
└──────┬──────────┬──────────┬────────────┘
       │          │          │
  ┌────▼───┐ ┌───▼───┐ ┌───▼────┐
  │  ext4  │ │ btrfs │ │  F2FS  │  ... (any registered filesystem)
  └────┬───┘ └───┬───┘ └───┬────┘
       └─────────┴──────────┘
                 │
    ┌────────────▼────────────┐
    │      Block Layer        │
    │  blk-mq, I/O scheduler  │
    └────────────┬────────────┘
                 │
    ┌────────────▼────────────┐
    │    Storage Device       │
    │  NVMe / eMMC / UFS      │
    └─────────────────────────┘
```

Key data structures:

- `struct super_block`: per-mounted-filesystem state; points to root inode
- `struct inode`: per-file metadata (uid, gid, size, timestamps, block pointers); no filename
- `struct dentry`: name-to-inode cache entry; forms the directory tree in memory
- `struct file`: per-open-file state; current offset, flags; points to inode

Key operations table: `struct file_operations` — `open`, `read`, `write`, `mmap`, `ioctl`, `fsync`, `llseek`. Every filesystem registers its **own implementation** of these callbacks. `mmap` is critical for AI workloads: **memory-mapped dataset files** avoid read() syscalls entirely.

> **Key Insight:** When you call `open("/data/frames/frame0001.raw", O_RDONLY)` on an ext4 filesystem and then again on a btrfs filesystem, you get the same file descriptor, the same `read()` API, the same behavior from the application's perspective. VFS is the reason this works. What differs between filesystems is what happens underneath — how blocks are allocated, how crashes are handled, and how fast metadata operations complete.

---

## ext4

ext4 is the **default Linux filesystem**; backward-compatible with ext2/ext3.

ext4 is a mature, well-tested filesystem that prioritizes **reliability and broad hardware compatibility**. It is the right choice for general-purpose Linux root partitions — including Jetson rootfs — where predictable behavior matters more than advanced features.

### Key Features

- **Extent-based allocation**: replaces indirect block maps; one extent = (logical block, physical block, length); reduces metadata for large contiguous files
- **64-bit block addresses**: supports volumes up to 1 EB
- **Delayed allocation** (`delalloc`): batch block allocation at writeback time; improves spatial locality; `nodelalloc` disables it
- **dir_index** (htree): large directories stored as hash B-tree; O(log n) lookup vs O(n) linear scan

> **Key Insight:** The ext4 extent format matters for AI dataset performance. A large file described by a single extent (logical block 0 → physical block 1000, length 100K blocks) can be read in one contiguous DMA operation. A file described by hundreds of indirect block pointers requires many separate metadata lookups. For large training dataset files, ext4 with extent allocation approaches sequential read performance.

### Journal Modes

| Mode | What is journaled | Data safety | Performance |
|---|---|---|---|
| `data=ordered` (default) | Metadata only; data written before commit | Good | Fast |
| `data=journal` | Metadata + data | Best | Slowest |
| `data=writeback` | Metadata only; data order not guaranteed | Weakest | Fastest |

Mount with `mount -o data=ordered /dev/mmcblk0p1 /mnt`. For Jetson rootfs, `data=ordered` is the **standard**.

Default journal size: 128 MB (tunable via `tune2fs -J size=N`). Journal is a **circular log**; on crash, uncommitted transactions are discarded; committed but not checkpointed are replayed.

The crash recovery sequence for `data=ordered` mode:

1. **File data is written to disk** before any metadata journal entry referencing it is committed. This ensures that if the system crashes after the journal commit, the data blocks it points to are already on disk.
2. **Journal commit** writes a commit block to the circular journal log, making the metadata change durable.
3. **Checkpoint**: at some later time, the journaled metadata is written to its final location in the filesystem. Only then can the journal space be reused.
4. **On crash recovery**: `e2fsck` reads the journal, replays committed but uncheckpointed entries, discards uncommitted entries. A clean journal means no fsck needed.

> **Common Pitfall:** `data=writeback` is sometimes used for maximum write throughput (e.g., on a temporary data partition). The danger is that after a crash, a committed metadata entry (file size, block pointer) may reference data blocks that were not yet flushed, resulting in garbage data in files that appeared successfully written. Never use `data=writeback` for application data that must survive a crash.

---

## btrfs

btrfs is a **Copy-on-Write B-tree filesystem** with features designed for modern storage and system management. Where ext4 prioritizes simplicity and compatibility, btrfs prioritizes **advanced features** — particularly snapshots and checksums — that are critical for OTA update strategies.

### Core Features

- **Copy-on-Write (CoW)**: writes never overwrite existing data in place; new data is written to free space, then the B-tree root pointer is atomically updated
- **Snapshots**: O(1) creation; a snapshot is simply a new B-tree root pointing to the same leaf nodes; used for A/B rootfs in OTA update strategies
- **Subvolumes**: independent filesystem trees within the same btrfs pool; can be mounted separately or snapshotted individually
- **Checksums**: CRC32c (default), xxHash, SHA256, or Blake2 on both data and metadata; detects silent bit rot
- **RAID modes**: RAID 0, 1, 10, 5, 6 across multiple devices
- **Transparent compression**: zstd (best ratio), lzo (fastest), zlib; per-file or per-subvolume
- **send/receive**: compute a delta between two snapshots and stream it to another system; enables incremental OTA updates

> **Key Insight:** btrfs snapshot creation is O(1) because a snapshot is just a new B-tree root that shares the same leaf nodes as the original. No data is copied. The original and the snapshot both reference the same physical blocks. When either is modified, CoW creates new blocks for the modified data — the other snapshot continues to reference the original blocks undisturbed. This is what makes instant A/B rootfs switching possible.

### A/B Rootfs with btrfs Snapshots

```bash
# Create read-only snapshot before OTA
btrfs subvolume snapshot -r /rootfs /.snapshots/$(date +%Y%m%d)

# After update, create new snapshot from updated rootfs
btrfs subvolume snapshot /rootfs_new /.snapshots/new

# Rollback: delete new rootfs, restore from snapshot
```

The A/B OTA flow with btrfs snapshots works as follows:

1. **Before the update begins**: create a read-only snapshot of the current rootfs. This takes milliseconds and uses no additional disk space.
2. **Apply the update**: write new files, packages, or a full rootfs image to a new subvolume. The snapshot of the old rootfs is untouched.
3. **Reboot into the new rootfs**: bootloader mounts the new subvolume as the active rootfs.
4. **If the new rootfs fails to boot**: bootloader detects the failure (boot counter exceeds threshold) and mounts the old snapshot as the active rootfs. Full rollback with zero data movement.
5. **After a successful boot**: the old snapshot can be deleted to reclaim space.

`btrfs scrub start /`: verify all data blocks against stored checksums; background operation; critical for long-running edge AI devices where **silent corruption** is a concern.

> **Common Pitfall:** btrfs CoW creates extra write amplification for database-style workloads (many small, random updates to the same file regions). SQLite databases and RocksDB on btrfs can perform worse than on ext4 because each small write copies an entire B-tree leaf node. Use `chattr +C` (disable CoW for specific files or directories) for database files on btrfs, or place database files on a separate ext4 partition.

---

## F2FS (Flash-Friendly File System)

With ext4 and btrfs understood, we turn to the filesystem designed specifically for the storage hardware used in AI edge devices: NAND flash in eMMC and UFS packages.

F2FS is designed for **NAND flash storage**: eMMC, UFS, and NVMe SSDs. Developed by Samsung; merged into Linux 3.8.

### Design Principles

- **Log-structured with node/data separation**: node area (inodes and index) and data area are logged separately; reduces fragmentation from mixed updates
- **Adaptive logging**: switches between normal logging (high utilization) and threaded logging (low utilization) to balance write amplification and fragmentation
- **Reduced write amplification**: flash-aware allocation avoids partial block updates; aligns writes to flash erase block size
- **Optimized `fsync()` latency**: important for database workloads (SQLite on Android); uses a roll-forward recovery mechanism to avoid full checkpoint on every fsync

> **Key Insight:** Flash memory has a fundamental asymmetry: you can write small chunks but must erase large chunks (typically 256 KB–2 MB). A filesystem that writes small random updates forces the flash controller to read-modify-write large erase blocks (write amplification). F2FS's log-structured design collects writes sequentially, aligning them to erase block boundaries and dramatically reducing this amplification — extending the life of the flash chip.

Use cases: Android (since Android 9 default on eMMC/UFS), Chromebooks, embedded AI devices with eMMC storage.

---

## overlayfs

overlayfs is the mechanism that makes **containers, OTA overlays, and read-only rootfs** configurations work. Understanding it is required for working with Docker containers and Jetson OTA strategies.

overlayfs overlays **two directory trees** into a unified view:

- **lower**: read-only base layer (one or more stacked layers)
- **upper**: writable layer; receives all modifications
- **workdir**: temporary directory on the same filesystem as `upper`; used for atomic rename operations

On first write to a file from `lower`, the file is **copied up to `upper`** (copy-on-write). Subsequent writes go directly to `upper`.

```
Application sees /container/merged (unified view):
  /bin/bash       ← from lower layer (image)
  /lib/libc.so    ← from lower layer (image)
  /etc/config     ← from upper layer (modified by container)
  /tmp/cache      ← from upper layer (created by container)

Physical layout:
┌────────────────────────────────────────────────────────┐
│  upper (writable, per-container)                        │
│  /etc/config   /tmp/cache                               │
├────────────────────────────────────────────────────────┤
│  lower layer 2 (image layer, read-only)                 │
│  /etc/config.orig   (shadowed by upper /etc/config)     │
├────────────────────────────────────────────────────────┤
│  lower layer 1 (base image, read-only)                  │
│  /bin/bash   /lib/libc.so                               │
└────────────────────────────────────────────────────────┘
overlayfs merges these: upper takes precedence, lower fills in the rest
```

### Mount Syntax

```bash
mount -t overlay overlay \
  -o lowerdir=/image/layer2:/image/layer1,\   # colon-separated, top-to-bottom
     upperdir=/container/rw,\                 # writable layer (must be empty)
     workdir=/container/work \                # must be on same fs as upperdir
  /container/merged                           # unified view mount point
```

### Use Cases

- **Docker/OCI containers**: image layers form the `lower` stack; the container's writable layer is `upper`; the container sees `/merged` as its rootfs
- **Jetson OTA**: new rootfs image as `lower`; persistent user data in `upper`; avoids a full rootfs copy for configuration
- **Read-only rootfs with writable overlay**: boot from read-only squashfs; mount overlayfs with tmpfs as `upper`; writable rootfs in RAM without modifying flash

> **Common Pitfall:** The `upper` and `workdir` directories must reside on the same filesystem (same mount point). This is because overlayfs uses atomic renames within workdir for safe copy-up operations. Placing workdir on a different filesystem than upper causes the mount to fail with `EINVAL`. A common mistake is placing workdir on tmpfs and upper on ext4 (or vice versa).

---

## tmpfs: Filesystem in RAM

overlayfs often uses tmpfs as its writable `upper` layer for ephemeral containers. But tmpfs has its own important standalone role: it is the foundation of all shared memory IPC in Linux.

`tmpfs` uses anonymous pages (RAM + swap) for storage; **no disk I/O** for reads or writes.

```bash
mount -t tmpfs -o size=1G tmpfs /dev/shm  # explicit tmpfs mount, 1 GB limit
```

- `/dev/shm` and `/run` are tmpfs by default on systemd systems
- `shm_open()` / `mmap(MAP_SHARED)` use tmpfs for POSIX shared memory
- openpilot `cereal` msgq allocates shared memory segments via `shm_open()` on `/dev/shm`; all IPC between processes (modeld, plannerd, controlsd) traverses these RAM-backed segments

> **Key Insight:** When openpilot's `modeld` publishes a model output (trajectory prediction, detected objects) and `plannerd` reads it, the data never touches the disk. The "file" created by `shm_open("/cereal_modeld", ...)` exists only in tmpfs — in RAM. This is how a multi-process architecture achieves the same low-latency IPC as a single-process design with direct memory writes.

---

## TRIM and Flash Longevity

Every time the OS deletes a file on a flash-backed filesystem, it knows those blocks are now free — but the **flash hardware does not automatically learn this**. TRIM bridges this gap.

`fstrim` notifies the SSD of **freed block ranges** so the controller can reclaim them during idle time.

- `discard` mount option: issue TRIM inline on every `unlink()` (higher latency per delete)
- `systemd-fstrim.timer`: weekly `fstrim` sweep (preferred; batches TRIMs)
- Without TRIM, the FTL does not know which logical blocks are free; write amplification increases as the drive ages
- Critical for embedded AI devices with eMMC (dashcams, edge inference boxes) that write continuously and cannot be taken offline for maintenance

> **Common Pitfall:** Deploying an eMMC-based AI edge device with `discard` disabled and `systemd-fstrim.timer` not enabled. Over weeks of continuous camera recording, the FTL fills its mapping table with "live" blocks that the OS knows are free but the SSD does not. Write amplification grows, throughput drops, and eventually write latency spikes cause frame drops in the recording pipeline. The fix is simple — enable the timer — but diagnosing the root cause after the fact requires correlating `iostat` throughput degradation over time with device age.

---

## Summary

| Filesystem | Journaling | CoW | Best for | Key limitation |
|---|---|---|---|---|
| ext4 | Yes (ordered/journal/writeback) | No | General-purpose Linux rootfs | No snapshot support |
| btrfs | No (CoW replaces journal) | Yes | A/B OTA, snapshot-based rollback | Higher CPU overhead |
| F2FS | Checkpointing | No | eMMC/UFS embedded devices | Not ideal for HDDs |
| overlayfs | Inherits from upper layer | On first write | Container rootfs, OTA overlay | upper/lower must differ |
| tmpfs | None (RAM) | No | IPC shared memory, /tmp | Lost on reboot/OOM |
| ext4 + F2FS | Yes | No | Mixed: rootfs (ext4) + data (F2FS) | Separate partitions needed |

### Conceptual Review

- **Why does ext4 use `data=ordered` mode by default instead of `data=journal`?** Full data journaling (`data=journal`) writes every byte of data twice: once to the journal, once to its final location. For a device recording camera frames at high throughput, this doubles the write load. `data=ordered` provides strong crash consistency (data is on disk before metadata commits to the journal) at the cost of only journaling metadata — a much smaller amount of data.

- **What makes btrfs snapshots O(1) in time and initially O(0) in space?** A btrfs snapshot is a new B-tree root node that points to the same leaf nodes as the original subvolume. Creating it requires writing only one new root node — regardless of how many files are in the filesystem. No data blocks are duplicated at snapshot time. Space is only consumed when one version (original or snapshot) diverges from the other through writes, which triggers CoW.

- **Why is F2FS preferred over ext4 on eMMC for continuous recording workloads?** eMMC NAND flash requires erase-before-write at the granularity of large erase blocks. ext4's random-write patterns force the eMMC FTL to perform read-modify-write cycles frequently, multiplying the physical write count (write amplification). F2FS's log-structured design aggregates writes sequentially, matching the natural write granularity of flash and reducing write amplification, extending the device's P/E cycle budget.

- **How does overlayfs enable immutable container images?** The base image layers are mounted as `lower` (read-only). Any modification the container makes — installing a package, creating a log file — goes to the `upper` (writable) layer. The base image on disk is never modified. When the container stops, the `upper` layer can be discarded, leaving the base image exactly as it was. Multiple containers can share the same base image layers simultaneously with no conflicts.

- **What role does tmpfs play in openpilot's IPC architecture?** openpilot processes (modeld, plannerd, controlsd, camerad) communicate via cereal msgq, which uses POSIX shared memory (`shm_open()`) backed by `/dev/shm` — a tmpfs mount. Messages between processes are written to RAM, not disk. This gives inter-process communication the same latency as direct memory writes, with no filesystem overhead.

- **What happens to write amplification on an eMMC without TRIM over time?** The Flash Translation Layer (FTL) maintains a logical-to-physical block mapping. When a file is deleted, the OS frees the logical blocks. Without TRIM, the FTL still considers those physical blocks "live" and cannot use them for new writes without first erasing them. As the device fills with these zombie blocks, the FTL must increasingly perform garbage collection (erase + copy) on partially-used erase blocks before any new write can land, multiplying the effective write amplification by 10–50x in the worst case.

---

## AI Hardware Connection

- btrfs snapshots provide O(1) A/B rootfs switching in openpilot Agnos and Jetson OTA update pipelines; on failed boot, the bootloader activates the previous snapshot without data movement
- F2FS on eMMC significantly reduces write amplification in embedded AI edge devices (dashcams, robotics controllers) that perform continuous camera recording
- overlayfs enables containerized TensorRT inference deployments where the base image layer is read-only and immutable while the container adds only runtime state in the upper layer
- tmpfs and `shm_open()` underpin openpilot cereal msgq IPC; all model outputs, trajectory plans, and control commands between processes traverse RAM-backed shared memory segments
- ext4 `data=ordered` mode is the standard for Jetson rootfs partitions; it provides the strongest crash consistency guarantee without the overhead of full data journaling
- `systemd-fstrim.timer` is a mandatory configuration item for any long-running eMMC-based AI edge device; without it, write amplification degrades throughput and reduces flash lifespan
