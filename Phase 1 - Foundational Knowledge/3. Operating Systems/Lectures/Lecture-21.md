# Lecture 21: Filesystems, Durability, and Temporary Storage

A successful `write()` means the kernel accepted bytes according to that operation's contract. It does not necessarily mean those bytes will survive sudden power loss. A filesystem choice also does not implement a complete OTA protocol. This lesson separates **names**, **cached data**, **durability**, and **recovery**.

## VFS: a common interface, different behavior

Linux's virtual filesystem layer connects operations such as open/read/write/mmap to particular filesystems. A pathname resolves through mounts and directories to a file object; a descriptor refers to an open file description. Removing a name does not immediately destroy storage still referenced by open files or mappings.

A regular file, tmpfs object, device node, procfs entry, and socket can all be reached through file-related interfaces while having very different semantics. Do not infer persistence or seekability merely because `open()` succeeded.

## ext4 and the application durability boundary

Ext4 journals filesystem metadata. Its data modes affect ordering and journaling of file contents, but journaling is not an application transaction and not a backup. The common ordered mode aims to order related data writes before metadata commitment; it does not promise that every recent application update survives a crash.

A common replace-a-file protocol is:

```text
create a temporary file in the destination directory
write the complete new contents, handling short writes and errors
fsync the temporary file; check success
rename it over the destination within the same filesystem
fsync the containing directory; check success
```

This is a protocol sketch, not a guarantee for every storage stack. Permissions, ownership, concurrent writers, filesystem behavior, and device flush/power-loss characteristics matter. Same-filesystem rename gives atomic namespace replacement; it is not by itself a durability barrier. The [fsync manual](https://man7.org/linux/man-pages/man2/fsync.2.html) explains why the containing directory may need its own sync.

**Worked failure question:** the process crashes after rename but before directory fsync. Other processes may already have seen the new name, yet a later power failure can leave persistence uncertain. Application visibility and durable commitment are different observations.

## btrfs: snapshots share storage

Btrfs uses copy-on-write metadata and supports subvolume snapshots, checksums, and other features. A snapshot initially shares data extents with its source and consumes metadata; it is not literally zero-space storage or a universal constant-time operation independent of transaction state. Later changes retain older referenced extents, increasing space use.

Checksums detect some corruption; repairing it requires another valid copy or other recovery source. A snapshot on the same device is not an independent backup against device failure. Btrfs has tree-log/recovery behavior; “no ext4-style journal” should not be taught as “there is no logging.” Consult the [btrfs documentation](https://btrfs.readthedocs.io/en/latest/Introduction.html).

Snapshots can participate in an update design, but boot selection, kernel/firmware compatibility, health confirmation, rollback rules, and persistent application data still need their own protocol. Do not delete or replace a running root subvolume to demonstrate a snapshot. Use a disposable filesystem image and its documented tools if you later study snapshots hands-on.

## F2FS and the flash translation layer

F2FS is designed around flash-friendly placement and cleaning. Managed flash devices such as eMMC, UFS, and SSDs also contain a flash translation layer (FTL), which performs internal mapping, wear leveling, and garbage collection. The filesystem does not generally know or control every physical NAND erase-block decision.

Sequential placement can help some workloads; it does not prove that F2FS always outperforms ext4 or always gives a write-amplification factor near one. Measure the actual device, fullness, discard policy, write pattern, fsync frequency, steady-state latency, and health counters. Cleaning and device garbage collection can affect tail latency even when average throughput looks good. See the [kernel F2FS documentation](https://docs.kernel.org/filesystems/f2fs.html).

## OverlayFS: a combined view

OverlayFS combines one or more lower trees with an upper writable tree. A file may be read from a lower layer until a write requires copy-up into the upper layer. Whiteouts/opaque directories represent changes to the combined view. Copy-up can make a seemingly small first write more expensive than later writes.

For the conventional writable overlay setup, upper and work directories must be on the same suitable filesystem, the work directory must be empty, and paths must be configured according to OverlayFS constraints. The upper directory need not always be empty—it commonly stores existing changes. Do not edit a lower tree behind a mounted overlay and expect defined behavior.

**Documented mount syntax for a prepared disposable environment; not executed by this lesson:**

```bash
mount -t overlay overlay -o lowerdir=/lab/lower,upperdir=/lab/upper,workdir=/lab/work /lab/merged
```

This requires suitable privileges and existing directories; `/lab` is a placeholder. An inline comment after a shell line-continuation backslash breaks continuation, so keep comments outside the command. See the [OverlayFS documentation](https://docs.kernel.org/filesystems/overlayfs.html) for supported lower/upper options and limitations.

Container image layers often use an overlay-style storage driver, but the exact driver and backing filesystem are runtime choices. A read-only application root plus explicitly writable data directories can make state easier to reason about; it does not make that state automatically durable.

## tmpfs and shared memory

Tmpfs stores file data in memory and may use swap when permitted. It is not persistent across reboot, yet “tmpfs never causes storage I/O” is false when swapping occurs. A `size=` limit bounds usage; it does not preallocate or reserve that amount of RAM. See [tmpfs semantics](https://docs.kernel.org/filesystems/tmpfs.html).

POSIX shared memory on Linux commonly uses tmpfs mounted at `/dev/shm`. `shm_open()` objects need sizing, cleanup, and synchronization. Other IPC implementations can use memfd, sockets, mapped files, or device buffers. Do not label every openpilot message as a named `/cereal_modeld` POSIX-shm object without inspecting the pinned implementation.

Do not remount the host's `/dev/shm` as a beginner experiment: that can affect other processes. The existing [SPSC lab](../labs/SPSC-Queue.md) teaches bounded ownership using ordinary process memory and real threads. Its result says nothing about filesystem durability or portable interprocess atomics.

## Discard, capacity, and diagnosis

Discard/TRIM informs supported storage that logical blocks are no longer in use. It is distinct from secure erasure and from guaranteeing that a particular write is durable. Periodic trimming and filesystem discard policies have workload/device tradeoffs; inspect support and distribution policy instead of enabling every option by default.

Useful read-only starting points are `findmnt`, `df -h`, `lsblk -f`, and `/proc/mounts`. These report mounts, capacity, or filesystem identities. A free-space report does not expose all internal flash overprovisioning, and filesystem free space is not the same as available application memory.

## A small design exercise

A robot stores a model, logs, and a mutable configuration file. Put each on a row with: required durability, maximum size, write rate, who may update it, and recovery behavior. The model can be an immutable verified artifact; logs may use bounded rotation; configuration needs a defined durable replacement protocol. They need not share one policy simply because they share one storage device.

**Check:** Does a journal guarantee the last logged sensor frame survives power loss? **Answer:** No. The application must define and execute its persistence boundary, and the full storage stack must honor it.

**Check:** Does a snapshot constitute an A/B updater? **Answer:** No. It is one storage primitive; boot and health state must still be coordinated.

**Check:** Can tmpfs pressure delay inference? **Answer:** Yes. Memory pressure, allocation/reclaim, and possible swap compete with the application's resources.
