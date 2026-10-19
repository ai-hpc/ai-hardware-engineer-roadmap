# Lecture Note 06: Virtual Memory, Translation, and Allocation

Consolidates [Lecture 12](Lecture-12.md), [Lecture 13](Lecture-13.md), and [Lecture 14](Lecture-14.md).

## Three address domains

A CPU virtual address identifies a location in an address space. Page tables translate it to physical storage or describe an access that faults. A device DMA address belongs to a device's mapping domain and need not equal either CPU address. Never put a userspace pointer directly into an arbitrary device descriptor.

Virtual mappings describe permissions, backing, and lifetime. Allocating virtual address space does not prove physical pages are populated, locked, contiguous, or DMA-accessible. Shared mappings need a synchronization protocol; identical virtual addresses in two processes say nothing by themselves about shared backing.

## Faults are events, not fixed timing constants

A minor fault can resolve without storage I/O, such as demand-zero or CoW work. A major fault involves the relevant I/O/accounting path. Neither has a universal 1 µs or 1–10 ms duration. Allocation, reclaim, contention, storage, and scheduling affect the observed interval.

Memory locking can reduce some paging risk, but limits and failures must be handled. It does not pre-resolve every future mapping, GPU migration, permission fault, or device wait. The [RT demo](demos/rt-demo/README.md) checks configuration failures instead of pretending that a failed lock request succeeded.

## A page-table calculation

For a simplified four-level x86-64 layout with 4 KiB pages, each 9-bit index selects one of 512 entries and the 12-bit page offset selects a byte. A TLB miss may require translation work; page-walk caches, huge leaves, and memory caches mean it is not always four uncached DRAM reads. Other paging modes and architectures have other shapes.

| Mapping size for an aligned 1 GiB region | Number of leaf mappings |
|---|---:|
| 4 KiB | 262,144 |
| 2 MiB | 512 |
| 1 GiB | 1 |

These are coverage counts, not entries simultaneously resident in a TLB. Supported sizes, alignment, availability, allocation cost, and access pattern determine whether large pages help.

**Question:** Is `1 GB / 4 KB` the same precise unit calculation as `1 GiB / 4 KiB`?

**Answer:** No. Binary and decimal units differ. State units before calculating translation counts or bandwidth.

## HugeTLB, THP, and placement

Explicit HugeTLB pages use their configured pools and mapping rules. Transparent huge pages have policy and kernel-version-dependent allocation/collapse behavior; modern implementations can support multiple sizes. `MADV_HUGEPAGE` is a policy hint, not proof that a region is huge or that latency improved. Inspect the resulting mapping and measure.

ASID/PCID tags can preserve translations across switches, but invalidations, rollover, mapping changes, and shootdowns still occur. Thread switches sharing an address space can differ from process switches. No universal TLB-hit or context-switch latency should be inferred from a table.

## Allocation and physical contiguity

`kmalloc`, `vmalloc`, page allocators, DMA allocators, and userspace malloc solve different problems. A virtually contiguous `vmalloc` region is not generally physically contiguous or suitable for `dma_map_single`. A physically contiguous allocation is also not proof of the correct device DMA address or ownership state.

CMA can reserve regions whose movable pages may be migrated when contiguous memory is needed. Pins, pressure, device constraints, and migration can still prevent allocation. Use the target DMA API; it chooses the appropriate platform mechanism rather than promising that every request comes from CMA.

**Question:** A 64 MiB region needs how many 4 KiB or 2 MiB pages?

**Answer:** 16,384 or 32. This does not establish actual TLB misses, latency, or suitability for a device.

## Completion exercise

For a model's weights and one camera frame, record allocation API, virtual size, residency evidence, address domain, current owner, final consumer, and release condition. Then explain which entries change when using a discrete GPU versus integrated shared DRAM.
