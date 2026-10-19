# Lecture 13: Page Tables, TLBs, and Huge Pages

## Follow one address through translation

Assume 4 KiB pages and a virtual address `0x12345`. Dividing by 4096 gives virtual page number `0x12` and offset `0x345`. If the page table maps that virtual page to physical frame `0xABC`, the physical byte address is `0xABC000 + 0x345 = 0xABC345`. Permissions must also allow the access.

```python
page_size = 4096
address = 0x12345
vpn, offset = divmod(address, page_size)
physical = 0xABC * page_size + offset
assert (vpn, offset, physical) == (0x12, 0x345, 0xABC345)
```

This arithmetic example is not a way to obtain real physical addresses from ordinary user-space pointers.

## Page tables and the TLB answer different questions

Page tables describe mappings in memory. The TLB caches translations and relevant permissions for a limited working set. A TLB hit avoids a new walk; a miss can walk valid tables without an OS page fault. Page-walk caches and data caches can reduce the actual memory traffic.

Linux's generic hierarchy is `PGD → P4D → PUD → PMD → PTE`. Architectures fold levels they do not use; large-page entries can end a walk at an upper level. For a four-level x86 example with 512 eight-byte entries per table, one table is `512 × 8 = 4096` bytes. One leaf page-table page describes `512 × 4 KiB = 2 MiB` of base-page mappings. This does not mean that the whole range is backed by one 2 MiB physical page. See [Linux page tables](https://docs.kernel.org/mm/page_tables.html).

The number of virtual pages in a region is not the number of TLB entries physically present in the CPU. It is the number of translations needed to cover the whole region at that page size. Capacity, associativity, supported page sizes, access patterns, and other address spaces determine misses.

## Work the units before claiming a benefit

Assume an aligned 1 GiB region, fully mapped with one page size:

| Page size | Pages covering the region |
|---|---:|
| 4 KiB | 262,144 |
| 2 MiB | 512 |
| 1 GiB | 1 |

If base-page leaf entries are 8 bytes each, their payload is `262144 × 8 = 2 MiB`, plus upper-level tables. This is table-storage arithmetic, not total process memory overhead.

**AI example:** exactly 7,000,000,000 parameters at two bytes each require 14,000,000,000 bytes for the parameter payload. With an aligned contiguous virtual region, rounding up gives 3,417,969 base pages of 4 KiB or 6,676 pages of 2 MiB. It does not give 27 PMD entries. Optimizer state, activations, metadata, padding, and device copies are excluded. A fragmented allocation layout can require more mappings.

```python
size = 7_000_000_000 * 2
assert (size + 4095) // 4096 == 3_417_969
assert (size + (2 << 20) - 1) // (2 << 20) == 6_676
```

## Translation invalidation and address-space tags

Changing a mapping or permission can require invalidating cached translations on CPUs that could use the old mapping. The mechanism may involve targeted operations and cross-CPU coordination. Not every update flushes every CPU's entire TLB, and no single fixed shootdown cost applies to every machine.

Address-space tags such as x86 PCID and Arm ASID can retain translations for multiple contexts. They reduce some flushes; they do not eliminate invalidation, tag reuse, or security-related operations. A context switch therefore does not imply either “all translations survive” or “everything is flushed.” Measure the actual workload and kernel. See the [x86 TLB documentation](https://docs.kernel.org/arch/x86/tlb.html).

## Transparent huge pages and explicit HugeTLB

THP and HugeTLB are different allocation/mapping mechanisms. THP can obtain larger backing dynamically and may later split it. Current kernels support more than a single universal THP size; available sizes and policies depend on architecture/kernel. HugeTLB uses explicit supported huge-page sizes and a managed pool/reservation model with its own limits and failure behavior. Read [THP](https://docs.kernel.org/admin-guide/mm/transhuge.html) and [HugeTLB](https://docs.kernel.org/admin-guide/mm/hugetlbpage.html).

Larger pages can reduce translation pressure but increase allocation, zeroing, fragmentation, or copying cost. They do not guarantee a zero TLB-miss rate, eliminate demand faults, or speed up every inference workload. A pool containing huge pages does not establish that a particular model allocation uses them.

Start with read-only inspection:

```bash
getconf PAGESIZE
cat /sys/kernel/mm/transparent_hugepage/enabled
cat /sys/kernel/mm/transparent_hugepage/defrag
cat /proc/meminfo
```

Where present, inspect per-size THP settings and the target mapping in `/proc/PID/smaps`. Do not change a shared host's global policy for this exercise. Compare default behavior and a deliberately configured application-level experiment on a development system, recording actual mappings and page-fault/latency measurements.

`MADV_HUGEPAGE` is a policy request, not proof that huge backing was allocated. `MADV_WILLNEED` is advice and can initiate useful readahead; it does not guarantee that every access is resident or that all page faults are eliminated. Do not substitute an advice call for checking population and the measured critical path. The [madvise manual](https://man7.org/linux/man-pages/man2/madvise.2.html) documents each advice separately.

## Device mappings are not ordinary cacheable RAM

MMIO registers may have side effects and ordering requirements. Do not implement a driver by hand-setting one x86 PCD bit: memory types involve architecture-specific rules, and Arm uses different attributes. Use the supported mapping/access APIs such as `ioremap`-family helpers and `readl`/`writel`, with the device's required ordering/readback behavior. `volatile` does not replace the kernel I/O API or DMA synchronization. Follow [bus-independent device access](https://docs.kernel.org/driver-api/device-io.html).

CPU page tables and device IOMMU mappings are separate. A huge CPU mapping does not establish a huge IOMMU mapping or coherent device access. [Lecture 15](Lecture-15.md) follows the DMA ownership boundary.

## Completion exercise

For a hypothetical aligned 64 MiB matrix buffer, calculate its base-page count at 4 KiB and its large-page count at 2 MiB. **Answer:** 16,384 and 32. Then explain why these numbers alone cannot predict speedup: the access pattern may already fit the TLB working set, memory bandwidth or arithmetic may dominate, and the requested mapping may never be obtained.

Run the tiny arithmetic examples, then use [Lecture 14's first-touch experiment](Lecture-14.md) to observe real fault counters. Keep those observations separate from assumptions about page size, storage latency, and device memory.
