# Lecture 13: Page Tables, TLBs & Huge Pages

## Overview

Every time a program accesses memory, the CPU must translate a virtual address — the address the program sees — into a physical address — the actual DRAM location. This translation is the job of the **Memory Management Unit (MMU)**, guided by a multi-level structure called the **page table**. The core challenge is that this translation must happen on every single memory access, so its speed determines overall system performance. The mental model to carry through this lecture is a multi-level directory lookup: think of it like a postal system where a full address is broken into country → state → city → street → house number, and each level is stored as a table in memory. For an AI hardware engineer, this matters enormously: a 7-billion-parameter model loaded for inference touches gigabytes of memory across millions of virtual addresses, and poor page table design means the CPU spends more time translating addresses than executing inference computations. Huge pages and TLB-awareness are first-class performance tools in any production AI system.

---

## x86-64 Four-Level Page Table Walk

### Virtual Address Decomposition

The translation process begins by slicing a virtual address into **index fields** that guide a **hierarchical lookup**. A 48-bit canonical virtual address is split into **five fields**:

```
┌─────────────────────────────────────────────────────────┐
│          48-bit Virtual Address Layout                  │
├──────────┬──────────┬──────────┬──────────┬────────────┤
│ [47:39]  │ [38:30]  │ [29:21]  │ [20:12]  │  [11:0]   │
│  9 bits  │  9 bits  │  9 bits  │  9 bits  │  12 bits  │
│  PGD idx │  PUD idx │  PMD idx │  PTE idx │  offset   │
│ (512 ent)│ (512 ent)│ (512 ent)│ (512 ent)│ (4KB page)│
└──────────┴──────────┴──────────┴──────────┴────────────┘
```

```
Bits [47:39] → PGD index   (9 bits, 512 entries)
Bits [38:30] → PUD index   (9 bits, 512 entries)
Bits [29:21] → PMD index   (9 bits, 512 entries)
Bits [20:12] → PTE index   (9 bits, 512 entries)
Bits [11:0]  → page offset (12 bits, 4KB page)
```

Walk: `CR3` holds the **physical address of the PGD**. Each table is exactly one 4KB page containing 512 × 8-byte entries. The hardware MMU performs the walk automatically **on a TLB miss**, issuing up to **four sequential memory reads** before the final data access.

The four-level walk in action — each arrow represents one memory read:

```
CR3 register
    │ (holds PGD physical address)
    ▼
┌─────────┐   [47:39]   ┌─────────┐   [38:30]   ┌─────────┐
│   PGD   │ ──────────> │   PUD   │ ──────────> │   PMD   │
│(4KB tbl)│             │(4KB tbl)│             │(4KB tbl)│
└─────────┘             └─────────┘             └─────────┘
                                                      │ [29:21]
                                                      ▼
                                                ┌─────────┐   [20:12]   ┌──────────┐
                                                │   PTE   │ ──────────> │ Physical │
                                                │(4KB tbl)│             │  Page    │
                                                └─────────┘             └──────────┘
                                                                              │ + [11:0] offset
                                                                              ▼
                                                                        Final data
```

> **Key Insight:** Four memory reads happen before the actual data access on a TLB miss. This is why the TLB cache exists — without it, every memory access would cost 5× memory latency. A cold page table walk can take 300–400 ns on modern DDR5 DRAM.

### Page Table Entry (PTE) Fields

Each 8-byte PTE entry encodes not just the **physical frame number** but also **permission bits** and **cache policy hints**:

| Bit | Name | Function |
|-----|------|----------|
| 0 | Present (P) | Entry valid; page fault if clear |
| 1 | R/W | 0 = read-only, 1 = read/write |
| 2 | U/S | 0 = supervisor only, 1 = user accessible |
| 3 | PWT | Page Write-Through cache policy |
| 4 | PCD | Page Cache Disable (uncached MMIO) |
| 5 | Accessed (A) | Set by hardware on any read; used by page reclaim |
| 6 | Dirty (D) | Set by hardware on write; used by swapper |
| 7 | PAT | Page Attribute Table index extension |
| 8 | Global (G) | Skip TLB flush on CR3 reload (kernel pages) |
| 63 | NX/XD | No-Execute; prevents instruction fetch |
| 51:12 | PFN | Physical Frame Number |

The **PCD** bit is especially important for AI hardware engineers: MMIO-mapped device registers (PCIe BAR windows, GPU registers) must be mapped with PCD=1 (cache disabled) to ensure reads always go to the hardware, not a stale CPU cache line.

> **Common Pitfall:** Mapping MMIO regions as cached (PCD=0) is a subtle but serious bug. Reads may return stale values from CPU cache instead of the hardware register. The symptom is non-deterministic driver behavior. Always use `ioremap()` (which sets PCD) rather than manually constructing PTEs for device register regions.

---

## ARM64 Page Table Walk

ARM64 uses **two separate base registers**: `TTBR0_EL1` for user-space (low VA, `0x0000...`) and `TTBR1_EL1` for kernel-space (high VA, `0xFFFF...`). This avoids the need to **swap a single CR3** on kernel entry.

This is a key architectural difference from x86. On x86, CR3 holds both kernel and user-space page table roots, so a kernel entry may require TLB management. On ARM64, the two TTBRs allow the kernel's TLB entries to remain in place while user-space entries are replaced — reducing kernel entry/exit overhead.

```
User VA (0x0000...)          Kernel VA (0xFFFF...)
       │                              │
       ▼                              ▼
  TTBR0_EL1                     TTBR1_EL1
  (User PGD)                  (Kernel PGD)
       │                              │
       └──────────┬───────────────────┘
                  ▼
           MMU walk (same
           table structure,
           separate roots)
```

| Configuration | Walk depth | Notes |
|---------------|-----------|-------|
| 4KB pages, 48-bit VA | 4 levels (PGD→PUD→PMD→PTE) | Standard Linux ARM64 |
| 4KB pages, 52-bit VA | 5 levels (ARMv8.2-LPA) | Large Physical Address extension |
| 16KB pages, 48-bit VA | 3 levels | Reduced depth, coarser granularity |
| 64KB pages, 48-bit VA | 2–3 levels | Low TLB entry count per GB |

> **Key Insight:** ARM64's 64KB page granule option is valuable on Jetson-class SoCs. It reduces the page table walk depth to 2–3 levels, shrinking translation overhead for large contiguous buffers like DMA input tensors and feature maps.

---

## Translation Lookaside Buffer

The **TLB** is a hardware cache of recent virtual→physical translations. On a **hit**, the CPU obtains the physical address in **one cycle**. On a **miss**, the hardware page table walker executes the full multi-level walk.

Think of the TLB as the CPU's address book. When you frequently call the same person, you remember their number and don't need to look it up again. When the address book is too small to hold all entries, lookups become expensive.

```
Virtual Address
      │
      ▼
 ┌──────────┐  HIT (1 cycle)   ┌──────────────┐
 │  L1 TLB  │ ───────────────> │ Physical Addr│
 │ (64 ent) │                  └──────────────┘
 └──────────┘
      │ MISS
      ▼
 ┌──────────┐  HIT (6-8 cycles) ┌──────────────┐
 │  L2 TLB  │ ────────────────> │ Physical Addr│
 │(1536 ent)│                   └──────────────┘
 └──────────┘
      │ MISS
      ▼
 Hardware Page Table Walker
 (4 sequential DRAM reads, 300-400 ns)
      │
      ▼
 Physical Address + TLB fill
```

### Typical TLB Hierarchy (Intel Skylake / AMD Zen)

| Level | Type | Entries | Latency |
|-------|------|---------|---------|
| L1 iTLB | Instruction | 128 (4KB), 8 (2MB) | 1 cycle |
| L1 dTLB | Data | 64 (4KB), 32 (2MB) | 1 cycle |
| L2 TLB | Unified | 1024–1536 (4KB) | 6–8 cycles |
| Hardware walker on DRAM miss | — | — | 100–200 cycles |

A cold walk touching all four page table levels in uncached DRAM costs four sequential DRAM round-trips (~300–400 ns on DDR5 at 80 ns latency), dominating memory access time for sparse access patterns.

### TLB Shootdown on SMP

When a CPU modifies a PTE that other CPUs may have cached in their TLBs, all stale copies must be invalidated. This process is called a **TLB shootdown** and is one of the more expensive synchronization operations in the kernel:

1. **Modify the PTE** in page table memory — update the physical mapping.
2. **Issue `INVLPG vaddr` locally** — flush the local CPU's stale entry immediately.
3. **Send IPI to all CPUs** that may hold the mapping — cross-processor interrupt.
4. **Target CPUs execute `INVLPG` or `TLBI` (ARM64)** — each remote CPU flushes its copy.
5. **Originating CPU waits for all acknowledgements** — must not proceed until every CPU has confirmed the flush.

`flush_tlb_range()` implements this. Shootdown is **expensive at high CPU counts** (N×IPI round-trip latency) and is a **hot path** in `mmap()`/`munmap()`-heavy workloads such as Python memory allocators and JVM garbage collectors.

> **Common Pitfall:** In multi-threaded inference servers that frequently call `mmap()`/`munmap()` (e.g., repeatedly loading model shards), TLB shootdowns become a bottleneck at high CPU counts. Profile with `perf stat -e tlb:tlb_flush` to detect this. Consider pre-allocating fixed memory pools with `mmap(MAP_FIXED)` rather than remapping per request.

---

## PCID and ASID: Context-Switch Optimization

Without process tagging, every context switch requires **flushing the entire TLB** to prevent one process from using another's cached translations. **PCID** (x86) and **ASID** (ARM64) solve this by **tagging each TLB entry** with the process that owns it.

### PCID (x86, Process Context Identifier)

**PCID** is a 12-bit tag stored in `CR3[11:0]`. TLB entries carry the PCID of the process that installed them. On a context switch, the new CR3 value carries the new process PCID. With the `NOFLUSH` bit set in CR3, TLB entries from other PCIDs are retained but invisible to the new process. Linux enables PCID on kernel ≥ 4.14, reducing context-switch overhead by 10–20% in syscall-heavy workloads.

### ASID (ARM64, Address Space Identifier)

**ASID** is an 8-bit or 16-bit tag (configurable at kernel build time) written into `TTBR0`. Each `mm_struct` is assigned a unique ASID. When the ASID space is exhausted, a rollover event triggers a global TLB flush and ASID reassignment across all cores.

> **Key Insight:** In a real-time inference daemon that frequently switches between inference threads and kernel I/O threads (e.g., handling GPU completion interrupts), PCID/ASID prevents the costly full TLB flush at each context switch. This is a "free" 10–20% improvement enabled simply by running a modern Linux kernel.

---

## Standard vs. Huge Pages

Now that we understand what TLB entries are and why they're limited, we can see why page size matters so much: larger pages cover more address space per TLB entry.

### 4KB Standard Pages

- Fine-grained physical memory protection and allocation
- 1 GB allocation → 262,144 PTEs → 512 page tables → 2 MB of page table memory
- 262,144 TLB entries required to cover 1 GB with zero misses; far exceeds L2 TLB capacity
- A 7B parameter model at fp16 (14 GB) requires ~3.6 million active PTEs

With 4KB pages and an L2 TLB of 1536 entries, a 7B model causes **constant TLB misses**. Every miss triggers a hardware walk costing hundreds of nanoseconds. This **adds up quickly** during a forward pass that touches every weight tensor.

### 2MB Huge Pages (PMD-level, x86)

A single **PMD entry** marked as a huge page covers **2MB directly**; the PMD[21] `PS` bit indicates a leaf entry. TLB entry count **reduced 512×** for the same address range.

```
4KB Pages (1 GB of model weights)         2MB Huge Pages (same 1 GB)
┌─────────────────────────────┐           ┌─────────────────────────────┐
│ PTE[0]   → 4KB page         │           │ PMD[0]  → 2MB huge page     │
│ PTE[1]   → 4KB page         │           │ PMD[1]  → 2MB huge page     │
│ ...                          │           │ ...                          │
│ PTE[262143] → 4KB page      │           │ PMD[511] → 2MB huge page    │
│                              │           │                              │
│ 262,144 TLB entries needed  │           │ 512 TLB entries needed      │
│ (exceeds L2 TLB capacity)   │           │ (fits in L2 TLB)            │
└─────────────────────────────┘           └─────────────────────────────┘
```

**HugeTLBFS** (explicit pre-allocation):
```bash
echo 512 > /sys/kernel/mm/hugepages/hugepages-2048kB/nr_hugepages
mmap(NULL, size, PROT_READ|PROT_WRITE,
     MAP_HUGETLB|MAP_ANONYMOUS|MAP_PRIVATE, -1, 0)
```
Pages are pinned; cannot be swapped or migrated. Used by databases (PostgreSQL shared_buffers), DPDK packet pools, and GPU driver host-pinned buffers.

**Transparent Huge Pages (THP)** (automatic kernel promotion):
```bash
echo always  > /sys/kernel/mm/transparent_hugepage/enabled
echo madvise > /sys/kernel/mm/transparent_hugepage/enabled
echo never   > /sys/kernel/mm/transparent_hugepage/enabled

# Per-VMA opt-in (effective under madvise mode)
madvise(addr, len, MADV_HUGEPAGE);
```
The `khugepaged` kernel thread scans anonymous VMAs and collapses 512 contiguous 4KB pages into one PMD entry when alignment and availability allow. `/proc/vmstat` fields `thp_collapse_alloc` and `thp_split_page` track promotion and demotion events.

THP uses **`madvise` mode** to give the application control: only regions explicitly marked with `MADV_HUGEPAGE` are promoted. This is the **recommended mode for inference workloads** — `always` mode can cause **unexpected latency spikes** from background khugepaged activity.

> **Common Pitfall:** Setting THP to `always` can cause unexpected latency in latency-sensitive inference servers. The `khugepaged` background thread collapses pages at unpredictable times, causing brief stalls. Use `madvise` mode instead and call `madvise(MADV_HUGEPAGE)` only on the specific weight tensor allocations where TLB pressure is measured.

### 1GB Huge Pages (PUD-level, x86)

Single PUD leaf entry covers 1GB. Static-only allocation via kernel boot parameter:
```
hugepagesz=1G hugepages=4
```
Cannot be freed at runtime. Maximally reduces TLB footprint for full-model weight buffers.

---

## madvise() Hints

The `madvise()` system call is the main interface for communicating memory access patterns to the kernel VM. Think of it as hints you give the kernel so it can optimize its behavior for your workload. `madvise(addr, len, advice)` communicates access pattern hints to the kernel VM:

| Hint | Effect |
|------|--------|
| `MADV_HUGEPAGE` | Enable THP for this VMA |
| `MADV_NOHUGEPAGE` | Disable THP; force 4KB pages |
| `MADV_WILLNEED` | Prefetch pages into RAM (synchronous readahead) |
| `MADV_DONTNEED` | Release physical pages; VMA mapping retained |
| `MADV_SEQUENTIAL` | Increase readahead window for linear scan |
| `MADV_RANDOM` | Disable readahead; random access pattern |
| `MADV_FREE` | Pages may be reclaimed lazily; data lost on reclaim |

For AI workloads, `MADV_WILLNEED` is **particularly valuable**: calling it on model weight regions before the first inference call forces the kernel to read those pages into RAM immediately, **eliminating page-fault latency** from the critical inference path.

---

## Monitoring TLB and Huge Page Behavior

With the conceptual framework in place, these monitoring tools show exactly what the kernel is doing with pages and translations at runtime:

`/proc/[pid]/smaps` per-VMA fields:
- `AnonHugePages`: bytes in this VMA backed by 2MB THP
- `THPeligible: 1`: VMA meets size and alignment criteria for promotion

`/proc/vmstat` system-wide counters:
- `thp_fault_alloc`: THP allocated directly on page fault
- `thp_collapse_alloc`: THP created by khugepaged background scan
- `thp_split_page`: THP split back to 4KB (alignment failure, copy-on-write, etc.)

`perf stat -e dTLB-load-misses,iTLB-load-misses`: hardware PMU TLB miss rate per run.

A high `thp_split_page` rate indicates huge pages are being allocated and then immediately split — often due to copy-on-write forks or memory mappings that cross alignment boundaries. Investigate VMA alignment when this occurs.

---

## Summary

| Page size | Arch level | TLB entries for 1GB | Allocation method | Use case |
|-----------|-----------|---------------------|-------------------|----------|
| 4KB | PTE | 262,144 | Default `mmap` / page fault | General purpose, fine protection granularity |
| 2MB | PMD | 512 | THP (auto) or HugeTLBFS (explicit) | Model weights, DPDK buffers, GPU pinned memory |
| 1GB | PUD | 1 | HugeTLBFS static boot parameter | Full-model single-mapping, NUMA servers |
| 64KB (ARM64) | PTE | 16,384 | Kernel build config (`PAGE_SIZE=64K`) | ARM SoC with 64KB granule, fewer walk levels |

### Conceptual Review

- **What is the TLB and why does it exist?** The TLB is a hardware cache of virtual-to-physical address translations. It exists because a full 4-level page table walk requires up to 4 DRAM reads, which would make every memory access 5× slower without caching.

- **Why do huge pages reduce TLB miss rate?** Each TLB entry covers the entire page size. A 2MB page needs 512× fewer TLB entries than 4KB pages for the same address range. Fewer entries means a higher hit rate in the fixed-size TLB.

- **What is a TLB shootdown and when does it happen?** A TLB shootdown is a cross-CPU invalidation sequence triggered when a PTE is modified. It is needed because each CPU caches translations independently. It happens during `munmap()`, `mprotect()`, and copy-on-write.

- **What is the difference between HugeTLBFS and THP?** HugeTLBFS pages are pre-allocated at boot or via sysfs, are pinned, and cannot be swapped. THP pages are created dynamically by `khugepaged` by collapsing 512 adjacent 4KB pages. HugeTLBFS is more predictable; THP is more flexible.

- **When should you use `MADV_WILLNEED`?** Before the first access to a large data region (model weights, embedding tables) when latency on first access is critical. It triggers asynchronous kernel readahead to pre-populate pages into RAM.

- **Why does `PCD=1` matter for device driver authors?** MMIO registers must not be cached by the CPU. Setting PCD=1 ensures every register read goes directly to the hardware. Using `ioremap()` (which sets PCD automatically) is the correct driver pattern.

---

## AI Hardware Connection

- THP with `MADV_HUGEPAGE` reduces TLB miss rate for PyTorch and TensorRT model weight tensors; a 7B parameter fp16 model needs 27 PMD entries at 2MB granularity versus 3.6 million PTEs at 4KB
- HugeTLBFS pre-allocates 2MB pages for DPDK packet buffer pools on network-attached AI inference servers, guaranteeing zero TLB pressure during line-rate packet processing
- PCID (x86) and ASID (ARM64) eliminate full TLB flushes on context switches between the inference daemon and kernel I/O threads, critical when GPU completion interrupts and inference share the same CPU cores
- `madvise(MADV_WILLNEED)` on model weight files pre-faults pages into RAM before the first inference call, removing page-fault latency from the critical path of cold-start inference
- NVIDIA GPU drivers map large VRAM BAR apertures using 2MB PMD entries on the host side, reducing host TLB pressure when CPU accesses GPU memory through the PCIe BAR window
- On Jetson with the ARM64 kernel built for 64KB page granule, NVDLA DMA allocations require fewer page table levels and fewer TLB entries to cover large contiguous input feature map buffers
