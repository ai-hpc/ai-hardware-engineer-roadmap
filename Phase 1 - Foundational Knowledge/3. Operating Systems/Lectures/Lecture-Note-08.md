# Lecture Note 08: Drivers, Buffers, and I/O Paths

Consolidates [Lecture 17](Lecture-17.md), [Lecture 18](Lecture-18.md), [Lecture 19](Lecture-19.md), and [Lecture 20](Lecture-20.md).

## Match, acquire, initialize, publish

The driver core matches devices to drivers using bus-specific rules. Device Tree, ACPI, and self-describing buses supply different discovery paths. A successful match does not guarantee successful probe: clocks, reset, regulators, IRQs, DMA constraints, and other dependencies matter.

Initialize handler-visible state before registering an IRQ and initialize all user-visible state before publishing an interface. `devm_*` automates matched resource release; it does not quiesce arbitrary DMA, callbacks, open files, or mappings. A device can disappear while old file descriptors remain.

**Question:** Why is a configfs overlay command not portable to every Linux board?

**Answer:** Mainline runtime overlay support does not automatically supply that vendor-specific userspace interface. Bootloader and runtime overlay paths are separate.

## Character-device contracts

`cdev_add()` publishes operations; `class_create("name")` is the modern signature, while older BSPs may require the module argument. Check target headers and every return. Device-node creation does not revoke old users during teardown.

Dispatch ioctl commands before copying arguments. RESET may have no argument; GET_STATUS copies initialized output; SET_CONFIG copies and validates input. Fixed-width UAPI types and explicit compatibility rules matter. `_IO*` macros encode a convention; they do not validate arbitrary driver code automatically.

Map storage with the helper appropriate to its allocation. `dma_mmap_coherent()` and MMIO mapping have different contracts. A universal `virt_to_phys` plus uncached `remap_pfn_range` recipe is wrong for arbitrary DMA allocations.

Wait queues notify consumers to re-check a synchronized predicate. A Boolean can lose multiple completions; multiple readers can race to claim one item. Handle interrupted waits and device removal explicitly.

## V4L2 buffer ownership

```text
QBUF → camera owns → DQBUF → application owns
     → GPU reads asynchronously → completion proves last read ended → QBUF
```

Returning a buffer immediately after a CUDA launch can let the camera overwrite it during GPU access. MMAP, USERPTR, and DMA-BUF use different fields; multi-planar queues need plane descriptors. Use the format/stride/size returned by negotiation. Compatible import and synchronization are prerequisites for zero-copy.

**Question:** Does STREAMOFF prove unrelated GPU work is complete?

**Answer:** No. Stop and synchronize each participating subsystem according to its own contract before reclaiming storage.

## io_uring and zero-copy

SQEs describe requests; CQEs describe outcomes. Give requests identities, account for negative results and short I/O, and keep referenced buffers alive until terminal completion. Completion order need not match submission order. Cancellation and multishot operations require their specific completion rules.

SQPOLL spends CPU on submission polling and can require a wakeup syscall. IOPOLL needs supported operations/devices; it is not a universal socket accelerator. Registered buffers have resource/lifetime costs. Measure CPU use and tail latency alongside IOPS.

DMA-BUF shares an allocation, while `sendfile`/`splice` avoid particular userspace payload bounces. None means zero CPU work or unrestricted buffer reuse. The [SPSC lab](../labs/SPSC-Queue.md) supplies the simpler same-process ownership analogue.

## PCIe and NVMe arithmetic

A Gen4 x16 link has an encoding ceiling of about 31.51 GB/s **per direction**, before protocol overhead. Adding both directions is not a valid divisor for a one-way transfer. Negotiated width/rate, shared uplinks, mappings, and device support matter.

An NVMe controller advertises its actual queue limits. Queues need not map one-to-one to vectors or CPUs. A completion need not interrupt the submitter's CPU. A same-switch route can enable P2P, but does not establish compatibility or guarantee that every GPU/storage API uses it.

**Question:** Does large BAR space increase PCIe bandwidth?

**Answer:** No. It changes an address aperture; it does not add lanes or increase their transfer rate.

## Integrate the evidence

Draw one request from application submission through kernel/device work to completion and release. Label all address domains and owners. Then distinguish a source example, compiled artifact, observed host run, and physical-device test. The native DT/module, SPSC, RT, and BPF build labs validate different boundaries; none should be relabeled a camera hardware test.
