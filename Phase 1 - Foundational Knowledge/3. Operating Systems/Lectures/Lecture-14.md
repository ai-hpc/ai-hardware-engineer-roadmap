# Lecture 14: Allocation, contiguity, and memory pressure

An allocation is useful only when its address, size, lifetime, and access rules match its consumers. A driver may need ordinary CPU storage, a device DMA mapping, or both. This lesson explains why “free RAM exists” is not enough to select an allocator.

**Prerequisites:** virtual memory and page tables. Plan 60–90 minutes including the small userspace experiment. Kernel snippets are documented reading exercises; they were not loaded as a driver. Continue to [DMA ownership](Lecture-15.md).

## 1. Count pages before choosing an API

A buddy allocator manages page blocks in power-of-two orders. With an **assumed 4 KiB base page**, order 0 covers 4 KiB, order 1 covers 8 KiB, and order 9 covers 2 MiB. A kernel configured with a different base page size changes those byte counts; a maximum allocation order is also configuration-dependent.

Physical fragmentation can leave many free pages without one suitable contiguous block. Conversely, a virtual mapping can present a contiguous address range backed by separate physical pages. `/proc/buddyinfo` helps inspect free blocks by node/zone/order; it is not a promise that a future allocation will succeed.

**Check:** 512 free 4 KiB pages total 2 MiB. They do not necessarily form one physically contiguous 2 MiB region. A CPU algorithm may work with a virtual mapping of those pages; a specific device mapping may have different requirements.

## 2. Choose storage according to its contract

| API family | Useful contract | Lifetime/release |
|---|---|---|
| `kmalloc`/`kzalloc`/`kcalloc` | Kernel CPU storage with physically contiguous backing; check size/context constraints | `kfree` |
| `kmem_cache_*` | Repeated objects with a selected cache layout/constructor contract | Matching cache free/destroy lifecycle |
| `alloc_pages` | A power-of-two order of contiguous physical pages | Matching page release API |
| `vmalloc` | Virtually contiguous storage; backing need not be physically contiguous | `vfree` |
| `kvmalloc` | Allocation that may use either kmalloc or vmalloc backing | `kvfree`; caller must tolerate either backing |
| DMA allocation/mapping APIs | CPU/device addresses and device-specific access contract | Matching DMA release/unmap after final device use |

There is no universal 8 KiB boundary above which `kmalloc` stops being valid. Cache classes, large-allocation thresholds, limits, and implementation details depend on the kernel and architecture. SLUB uses object caches and CPU-local mechanisms, but its exact locking/fast path depends on configuration. [Kernel allocation guide](https://docs.kernel.org/core-api/memory-allocation.html).

`kvmalloc` is useful when either backing is acceptable; it is not a universal substitute for DMA-capable storage or atomic-context allocation. A virtually contiguous pointer cannot simply be passed to a device as a DMA address.

## 3. Validate size and cleanup paths

For an array, use an API that checks multiplication overflow instead of silently multiplying an untrusted count into a byte size. This kernel reading example also imposes an application limit:

```c
/* Exercise excerpt: declarations, MAX_DESCRIPTORS, and caller omitted. */
if (count == 0 || count > MAX_DESCRIPTORS)
    return -EINVAL;
entries = kcalloc(count, sizeof(*entries), GFP_KERNEL);
if (!entries)
    return -ENOMEM;
/* Initialize, validate, and use entries under the subsystem's rules. */
kfree(entries);
```

Zeroed storage does not automatically construct every synchronization object or establish valid device descriptors. Initialize the required fields and APIs explicitly. A dedicated object cache can help repeated allocations, but it is not automatically faster; measure its size, locality, fragmentation, and allocation path.

`devm_*` resources are associated with device lifecycle. They can simplify probe failure/detach cleanup, but are not a replacement for per-request ownership, cancellation, or DMA quiescing. A device-managed allocation can remain allocated far longer than one frame; a driver must stop its users before release.

**Worked lifetime table:** a camera request owns a descriptor from successful allocation through queueing and final completion/cancellation. Its data buffer may have a separate owner. Every error path must either preserve ownership for retry or release exactly once. A completion interrupt cannot safely reference a descriptor already freed on a timeout path.

## 4. Match allocation context and failure handling

`GFP_KERNEL` permits sleeping/reclaim where the context allows it. Non-sleeping paths require a suitable nonblocking allocation strategy, often preallocated storage; `GFP_NOWAIT` and `GFP_ATOMIC` have different reserve/reclaim semantics. `GFP_ATOMIC` is not a success guarantee. Threaded interrupt handlers and hard IRQ handlers also have different context rules. [Allocation API guidance](https://docs.kernel.org/core-api/memory-allocation.html).

A practical sequence is: determine whether the caller may sleep, establish the object’s size/lifetime, choose the allocator/context flags, and handle failure. Do not choose `GFP_DMA32` merely because a device advertises a 32-bit DMA engine. Configure the device DMA mask and use the DMA API; translation or bounce buffering may be involved.

Memory zones reflect architecture and configuration. Some x86 systems use familiar low-address DMA zones, but `ZONE_NORMAL` is not universally “all memory above 4 GiB,” and CMA is not simply another name for `ZONE_MOVABLE`. Inspect the actual target. [Physical memory organization](https://docs.kernel.org/mm/physical_memory.html).

## 5. Understand CMA without promising availability

CMA reserves suitable physical regions so movable allocations can use them until a contiguous allocation needs the space. Reclaiming that region can require migration. Pins, placement constraints, pressure, or migration failure can still prevent a request from succeeding. Reserve sizing and allocation timing are design choices, not a proof that a camera allocation will always work.

A successful `dma_alloc_coherent` returns a CPU pointer plus a device-visible DMA address. Its implementation may use CMA, an IOMMU, or other platform mechanisms; the caller should use the documented contract instead of assuming every Jetson buffer is uncached, physically contiguous in one particular way, or IOMMU-less. Coherence does not eliminate descriptor ordering or device lifetime requirements. [DMA API](https://docs.kernel.org/core-api/dma-api-howto.html).

Read-only Linux inventory:

```bash
getconf PAGESIZE
cat /proc/buddyinfo
rg 'MemAvailable|CmaTotal|CmaFree|Slab|SReclaimable|SUnreclaim' /proc/meminfo
```

CMA fields may be absent when not configured. There is no universal `/proc/cma` interface. Debugfs CMA views require the relevant kernel configuration/mount/permissions. Do not clear caches or reserve hundreds of MiB merely to make an inventory command produce a value.

## 6. Observe first-touch faults in a userspace model

This bounded Python experiment was executed on the course’s Linux host. It allocates 8 MiB, writes one byte per base-page interval, and closes the mapping. It does not allocate kernel DMA buffers or establish physical placement.

```python
import mmap
import resource
page = mmap.PAGESIZE
size = 8 * 1024 * 1024
with mmap.mmap(-1, size) as data:
    before = resource.getrusage(resource.RUSAGE_SELF).ru_minflt
    for offset in range(0, size, page):
        data[offset] = 1
    after_first = resource.getrusage(resource.RUSAGE_SELF).ru_minflt
    for offset in range(0, size, page):
        data[offset] = 2
    after_second = resource.getrusage(resource.RUSAGE_SELF).ru_minflt
print({'page_bytes': page, 'touched_offsets': len(range(0, size, page)),
       'first_pass_minor_faults': after_first - before,
       'second_pass_minor_faults': after_second - after_first})
```

Expect the first pass to expose allocation/first-touch work that the second often avoids. Do not assert an exact fault count: huge-page behavior, mapping type, interpreter activity, and the host affect the result. A minor fault does not imply disk I/O. A successful prefault pass does not mean every future access by every thread can never fault.

**Transfer:** explain why touching the main thread’s stack cannot prepare a new worker’s stack. The [native API demo](demos/rt-demo/README.md) touches a small buffer on the actual worker and labels that limited scope.

## 7. Plan for pressure instead of protecting one process blindly

Memory use includes model weights, activations/workspaces, camera pools, queues, allocator caches, shared libraries, and simultaneous old/new deployments. Budget peak initialization and failure paths as well as steady state.

An OOM adjustment changes victim selection; it does not create memory. Protecting one process can move failure to a dependency or make recovery harder. Likewise, changing reserve/reclaim parameters does not guarantee nonblocking allocations always succeed. Record cgroup limits, system pressure, allocation failures, and the chosen recovery policy. [Linux OOM handling](https://docs.kernel.org/admin-guide/sysctl/vm.html#oom-kill-allocating-task).

Use bounded pools and queues where they fit the application. Define which operation is rejected, which resource is reclaimed, and how the system returns to service. Avoid an unbounded retry that holds a lock needed by recovery.

**Completion:** submit a page-count calculation, an allocation/lifetime table for one sensor buffer and descriptor, the first-touch observation, and one allocation-failure path. Explain CPU virtual, physical, and device-visible addresses separately before starting [Lecture 15](Lecture-15.md).
