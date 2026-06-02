# Lecture Note 06 (L12, L13, L14): Virtual Memory, Page Tables & TLBs, Kernel Memory Allocation

**Combines:** Lecture L12 (Virtual Memory & Linux Memory Model), L13 (Page Tables, TLBs & Huge Pages), L14 (Memory Allocation: SLUB, kmalloc & CMA).

---

## How This Note Is Organized

1. **Part 1 — Virtual memory:** Address space; layout (x86-64/ARM64); mm_struct, VMA; mmap; demand paging; COW; mlock; RSS vs PSS.
2. **Part 2 — Page tables & TLBs:** Page table walk; PTE bits; TLB and TLB shootdown; PCID/ASID; huge pages (2MB, 1GB); THP vs HugeTLBFS; madvise.
3. **Part 3 — Kernel allocation:** Buddy allocator; SLUB/kmalloc; vmalloc; zones; GFP flags; CMA.

---

# Part 1: Virtual Memory & the Linux Memory Model

**Context:** Each process has a **private virtual address space**. The **MMU translates VA→PA** per page. Isolation, overcommit, and demand paging are central; **page faults and swap are critical for RT** (avoid in hot path).

---

## Virtual Address Space

- **Isolation:** One process cannot access another's memory without kernel-mediated sharing.
- **Overcommit:** Total virtual across processes can exceed physical RAM; pages allocated on demand.
- **Layout (x86-64):** User: text, data/BSS, heap (↑), mmap (↓), stack (↓). Kernel: direct map, vmalloc, kernel text, modules, vDSO. ARM64: TTBR0 = user, TTBR1 = kernel.

---

## mm_struct & VMA

- **mm_struct:** One per process; `pgd`, `mmap` list, `mm_rb` tree, `start_code`/`end_code`, `start_brk`/`brk`, `total_vm`, `locked_vm`.
- **VMA (vm_area_struct):** Contiguous region with same permissions and backing. Fields: `vm_start`/`vm_end`, `vm_flags` (READ, WRITE, EXEC, SHARED, LOCKED), `vm_file`, `vm_pgoff`, `vm_ops` (fault, open, close).

**VMA types:** Text (file, R-X); data/BSS (file/zero); heap (anonymous, zero-fill); stack (anonymous, grow-down); file mmap shared (e.g. model weights); anonymous shared (e.g. VisionIPC); vDSO (kernel-provided).

---

## mmap() & Demand Paging

- **mmap()** creates a VMA; it does **not** allocate physical pages. Pages are allocated on first access (demand paging).
- **Fault types:** **Minor** — page in RAM but PTE absent (e.g. COW, demand-zero); ~1 µs. **Major** — page from disk (file/swap); 1–10 ms; fatal for RT.
- **mlockall(MCL_CURRENT|MCL_FUTURE)** prevents eviction; does not fault in untouched pages — **pre-touch** (e.g. memset) to avoid minor faults in the RT loop.

---

## Copy-on-Write (COW)

After `fork()`, parent and child **share pages read-only**. On first write, kernel allocates new page, copies content, installs writable PTE for writer. Makes `fork()` O(1); **model weights shared read-only consume one physical copy**.

---

## Inspecting Memory: maps, smaps, RSS vs PSS

- `/proc/<pid>/maps`: VMAs (range, perms, backing). `/proc/<pid>/smaps`: per-VMA RSS, PSS, swap, AnonHugePages.
- **RSS:** Resident set; double-counts shared pages. **PSS:** Proportional; shared pages split by number of sharers — correct for per-process accounting.

---

# Part 2: Page Tables, TLBs & Huge Pages

**Context:** Every access needs VA→PA translation. The **TLB caches it**; a miss triggers a **multi-level page table walk** (e.g. 4 reads, 300–400 ns). **Larger pages reduce TLB pressure**.

---

## Page Table Walk (x86-64)

- 48-bit VA split: PGD index (9), PUD (9), PMD (9), PTE (9), offset (12). CR3 → PGD; each level one 4 KB table (512 entries). Four memory reads on TLB miss.
- **PTE bits:** Present, R/W, U/S, PWT, PCD (cache disable — use for MMIO), Accessed, Dirty, Global, NX, PFN. **PCD=1** for device registers (`ioremap` sets it).

---

## TLB & TLB Shootdown

- **TLB:** Hardware cache of VA→PA; hit in 1–6 cycles; miss → hardware walk.
- **TLB shootdown:** When one CPU changes a PTE, others may have stale TLB entries. Kernel sends IPI; each CPU flushes affected entries. Expensive on many CPUs; hot in mmap/munmap-heavy workloads.

---

## PCID / ASID

- **PCID (x86):** Tag TLB entries by process; context switch can keep TLB (new CR3 with same PCID or NOFLUSH). Reduces switch cost.
- **ASID (ARM64):** Same idea in TTBR0; avoids full TLB flush on switch.

---

## Huge Pages

- **4 KB:** 1 GB = 262K PTEs; exceeds typical L2 TLB.
- **2 MB (PMD):** One PMD entry; 512× fewer entries for same range. **HugeTLBFS:** Pre-allocated, pinned; `MAP_HUGETLB`. **THP:** Kernel promotes 4 KB→2 MB via khugepaged; use `madvise` mode and `MADV_HUGEPAGE` on chosen regions to avoid unpredictable latency.
- **1 GB (PUD):** Boot-time only; `hugepagesz=1G hugepages=N`.
- **madvise:** `MADV_HUGEPAGE`, `MADV_NOHUGEPAGE`, `MADV_WILLNEED` (prefetch), `MADV_DONTNEED`, `MADV_SEQUENTIAL`/`MADV_RANDOM`.

---

# Part 3: Kernel Memory — Buddy, SLUB, vmalloc, CMA

**Context:** **Buddy** manages physical pages; **SLUB** carves objects from pages; **vmalloc** gives virtual continuity without physical continuity; GFP and zones control where and how; **CMA** reserves contiguous regions for DMA.

---

## Buddy Allocator

- Free lists by **order** 0..10 (2^order pages). `alloc_pages(gfp, order)` returns 2^order contiguous pages. Splits higher-order blocks when needed; merges buddies on free.
- **Fragmentation:** High-order allocation can fail despite total free memory. **CMA** reserves contiguous region at boot before fragmentation.

---

## SLUB & kmalloc

- **SLUB:** Per-CPU slabs; fast path lock-free. Partial slab list shared when CPU slab is full.
- **kmem_cache:** Dedicated cache for one object type (e.g. DMA descriptors); `kmem_cache_alloc`/`kmem_cache_free`.
- **kmalloc(size, gfp):** Size-based caches (8, 16, … 8K); above 8K uses buddy. `kzalloc` = zeroed. Use **GFP_KERNEL** in process context; **GFP_ATOMIC** in IRQ/spinlock (never sleep).

---

## vmalloc & kvmalloc

- **vmalloc(size):** Virtually contiguous, physically scattered; page tables built in vmalloc region. Slower than kmalloc; use for large buffers that do not need DMA contiguity.
- **kvmalloc** / **kvfree:** Prefer kmalloc; fall back to vmalloc for large sizes.

---

## Zones & GFP Flags

- **Zones:** ZONE_DMA (0–16 MB), ZONE_DMA32 (0–4 GB), ZONE_NORMAL (4 GB+), ZONE_MOVABLE (migration/CMA). GFP selects zone and behavior.
- **GFP_KERNEL:** May sleep, reclaim; process context. **GFP_ATOMIC:** No sleep/reclaim; IRQ/spinlock; can return NULL. **GFP_DMA**/ **GFP_DMA32** for device DMA range. **__GFP_ZERO**, **__GFP_NOFAIL**, **__GFP_NOWARN**.

---

## CMA (Contiguous Memory Allocator)

- Reserves contiguous physical region at boot (in ZONE_MOVABLE). Movable pages can use it until a device requests it; then kernel migrates them out. Driver gets contiguous DMA memory via `dma_alloc_*` or alloc_pages from CMA. Avoids runtime fragmentation failure for large DMA buffers.

---

## Summary Tables

**VMA / fault:** Text/data/heap/stack/file mmap/anon shared — fault = file read, zero-fill, or swap-in. Minor ~1 µs; major 1–10 ms.

**Page size / TLB:** 4 KB → 262K entries/GB; 2 MB → 512/GB; 1 GB → 1/GB. THP = dynamic; HugeTLBFS = explicit, pinned.

**Allocator:** Buddy = pages; SLUB = objects; kmalloc = general; vmalloc = virtual contiguity; CMA = contiguous for DMA.

---

## AI Hardware Connection

- **mmap(MAP_SHARED)** on DMA-BUF fd: zero-copy GPU/CPU buffer sharing. **mlockall** + pre-touch in RT inference to avoid faults. **COW** after fork shares model weights without doubling RAM.
- **THP / MADV_HUGEPAGE** on weight tensors reduces TLB misses. **PSS** in smaps for correct per-process memory accounting. **PCID/ASID** reduce context-switch cost.
- **GFP_ATOMIC** in ISR; **GFP_KERNEL** in process context. **CMA** or early allocation for large DMA buffers; avoid high-order alloc_pages on fragmented systems. **slabtop** for slab leaks.

---

*Combines Lectures L12, L13, L14 (Virtual Memory; Page Tables, TLBs, Huge Pages; SLUB, kmalloc, CMA).*
