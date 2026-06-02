# Lecture 14: Memory Allocation: SLUB, kmalloc & CMA

## Overview

The kernel needs to manage every byte of physical RAM: deciding which process or subsystem owns which pages, how to efficiently carve large pages into smaller objects, and how to guarantee that DMA-capable devices get physically contiguous memory. The core challenge is **fragmentation** — over time, repeated allocations and frees leave RAM as a patchwork of gaps that cannot satisfy a large contiguous request even when total free memory is ample. The mental model for this lecture is a hierarchy of allocators: the **buddy allocator** manages physical pages, **SLUB** carves those pages into objects, **vmalloc** provides virtually-contiguous fallback, and **CMA** reserves a contiguous region at boot for DMA. For an AI hardware engineer, these allocators directly determine whether a camera ISP driver can get its DMA buffers, whether a custom kernel driver leaks memory, and whether the inference process survives a memory pressure event on an embedded system.

---

## Physical Memory: The Buddy Allocator

The **buddy allocator** is the Linux kernel's primary physical page manager. It is the foundation on which all other kernel allocators are built. Free memory is tracked in 11 **free lists** indexed by order (2^0 through 2^10 pages). `alloc_pages(gfp_flags, order)` returns 2^order physically contiguous pages; `__get_free_pages()` returns the kernel virtual address directly.

```
Buddy Allocator Free Lists
Order 0  (4KB):   [page] [page] [page] ...
Order 1  (8KB):   [page-pair] [page-pair] ...
Order 4  (64KB):  [block] [block] ...
Order 9  (2MB):   [huge-block] ...
Order 10 (4MB):   [max-block] ...

alloc_pages(GFP_KERNEL, 9)
  → finds a free order-9 block (2MB contiguous pages)
  → if none, splits an order-10 block into two order-9 blocks
  → returns one; puts the other ("buddy") back in the order-9 list
```

| Order | Size | Typical use |
|-------|------|-------------|
| 0 | 4 KB | Single page, page table |
| 1 | 8 KB | Small DMA descriptor ring |
| 4 | 64 KB | Medium DMA buffer |
| 9 | 2 MB | Huge page backing |
| 10 | 4 MB | Large contiguous allocation |

### Fragmentation

Over time, free pages become **non-contiguous**. High-order allocations fail even when total free memory is sufficient because no single 2^N contiguous block exists. `/proc/buddyinfo` shows the count of free blocks at each order per zone per NUMA node. `echo 3 > /proc/sys/vm/drop_caches` reclaims clean page cache and slab objects (use with care in production; does not solve fragmentation).

> **Key Insight:** Fragmentation is why CMA exists. By boot time, before any process has had a chance to fragment memory, CMA carves out a guaranteed-contiguous region. If you wait until an AI accelerator driver needs DMA buffers, the contiguous pages may no longer be available.

> **Common Pitfall:** A driver that calls `alloc_pages(GFP_KERNEL, 9)` at runtime (for a 2MB DMA buffer) may fail with `-ENOMEM` even when hundreds of megabytes of free memory exist, because fragmentation has eliminated all contiguous 2MB blocks. Always use CMA or pre-allocate large DMA buffers at driver probe time, not on demand.

---

## SLUB Allocator

The buddy allocator works in units of pages (4KB minimum). Most kernel data structures — network socket descriptors, inode objects, driver private data — are far smaller than a page. The **SLUB allocator** (default since kernel 2.6.23) bridges this gap by managing **sub-page allocations**. Objects of the same type and size are grouped into **slabs**. Each slab is one or more contiguous pages. SLUB provides:

```
SLUB Allocator Structure
┌─────────────────────────────────────────┐
│            kmem_cache "myobj"           │
│  ┌─────────────────────────────────┐   │
│  │  Per-CPU slab (CPU 0)           │   │
│  │  [obj][obj][obj][FREE][FREE]..  │   │  ← Fast path: no lock
│  └─────────────────────────────────┘   │
│  ┌─────────────────────────────────┐   │
│  │  Per-CPU slab (CPU 1)           │   │
│  │  [obj][obj][FREE][FREE][FREE].. │   │
│  └─────────────────────────────────┘   │
│  ┌─────────────────────────────────┐   │
│  │  Partial slab list (node 0)     │   │  ← Shared; used when CPU slab full
│  │  [slab1] → [slab2] → [slab3]   │   │
│  └─────────────────────────────────┘   │
└─────────────────────────────────────────┘
```

- **Per-CPU slabs**: each CPU maintains a local active slab; allocation and free are lock-free on the fast path
- **Partial slab list**: per-node list of partially filled slabs; shared between CPUs when the local slab is exhausted
- **Object reuse**: freed objects are returned to the cache without zeroing (unless `__GFP_ZERO`); avoids repeated initialization overhead

### kmem_cache Interface

When a driver allocates the same structure type repeatedly (e.g., per-frame DMA descriptors), creating a dedicated `kmem_cache` is more efficient than using generic `kmalloc`:

```c
struct kmem_cache *cache = kmem_cache_create(
    "myobj",                        /* cache name visible in /proc/slabinfo */
    sizeof(struct myobj),           /* object size */
    __alignof__(struct myobj),      /* minimum alignment */
    SLAB_HWCACHE_ALIGN | SLAB_PANIC, /* align to cache line; panic if create fails */
    NULL);                          /* optional constructor */

struct myobj *p = kmem_cache_alloc(cache, GFP_KERNEL);
kmem_cache_free(cache, p);
kmem_cache_destroy(cache);  /* call only when module unloads */
```

`SLAB_HWCACHE_ALIGN` pads objects to a CPU cache line boundary, preventing false sharing between objects on different CPU cores — important for per-frame state in high-frequency ISR paths.

### kmalloc

`kmalloc(size, gfp_flags)` is the **general-purpose kernel allocator**. It selects among size-based SLUB caches covering power-of-2 sizes: 8, 16, 32, 64, 128, 256, 512, 1K, 2K, 4K, 8K bytes. `kzalloc(size, gfp)` zero-fills. `kfree(ptr)` returns to the owning slab cache.

Think of `kmalloc` as the kernel equivalent of `malloc()` — you don't need to manage a pool manually, but you accept some overhead from the generality. For sizes above 8KB, `kmalloc` delegates to the buddy allocator and returns a physically contiguous allocation.

### Diagnostics

- `/proc/slabinfo`: cache name, active objects, total objects, object size, pages-per-slab
- `slabtop`: live sorted view by memory consumption; first tool to check for slab leaks in drivers
- `ksize(ptr)`: returns the actual allocated size (may exceed requested size due to cache rounding)
- KASAN/KFENCE: kernel sanitizers; detect use-after-free and out-of-bounds in slab objects

> **Common Pitfall:** A driver that allocates a per-frame `kmalloc` object in the V4L2 `VIDIOC_QBUF` path but never calls `kfree()` on error paths will silently leak slab memory. `slabtop` will show the cache's object count growing unboundedly. Always pair allocations with frees in all error paths, or use `devm_kzalloc()` for probe-time allocations.

---

## vmalloc

With the buddy and SLUB allocators covered, there is one more scenario: a large kernel buffer that does not need to be physically contiguous for DMA, but must be virtually contiguous for easy CPU access. `vmalloc(size)` allocates **virtually contiguous but physically non-contiguous** memory. The kernel sets up page tables in the `vmalloc` VA region, mapping individual pages. It is slower than `kmalloc` due to page table construction and TLB flushing on setup and teardown.

```
vmalloc virtual address space
┌─────────────────────────────────────────────────────┐
│  VA: 0xffff_c000_0000_0000 ... (vmalloc region)     │
│                                                     │
│  vaddr[0..4KB]   → physical page A (any location)  │
│  vaddr[4K..8KB]  → physical page B (any location)  │
│  vaddr[8K..12KB] → physical page C (any location)  │
│  ...                                                │
│  (pages are NOT contiguous in physical RAM)         │
└─────────────────────────────────────────────────────┘
CPU sees one contiguous buffer; DMA engine cannot use it directly.
```

- `vfree(ptr)`: release; flushes vmalloc VA page tables
- `kvmalloc(size, gfp)` / `kvfree(ptr)`: preferred hybrid; attempts `kmalloc` first (contiguous, fast); falls back to `vmalloc` for large sizes
- Use cases: large kernel buffers that do not require physically contiguous DMA (module BSS, FPGA bitstream staging, large firmware images)
- Maximum size: limited by `VMALLOC_SIZE`; typically hundreds of GB on 64-bit kernels

> **Key Insight:** `kvmalloc` is the best default for variable-size allocations of unknown size. It transparently gives you the fastest allocation path (physically contiguous `kmalloc`) for small sizes and falls back to `vmalloc` when that fails — without requiring the caller to know which path was taken.

---

## Memory Zones

The physical address space is partitioned into **zones** based on **accessibility constraints** imposed by hardware. Not all physical RAM is equally usable for all purposes — legacy ISA devices, 32-bit PCIe devices, and modern 64-bit devices each have different DMA address range limits.

```
Physical Address Space → Memory Zones
┌─────────────────────────────────────────────┐
│ 0x000_0000_0000 (0GB)                       │
│   ┌─────────────────────────────────────┐   │
│   │ ZONE_DMA    (0–16 MB)               │   │  GFP_DMA    — legacy ISA
│   └─────────────────────────────────────┘   │
│   ┌─────────────────────────────────────┐   │
│   │ ZONE_DMA32  (0–4 GB)                │   │  GFP_DMA32  — 32-bit PCIe
│   └─────────────────────────────────────┘   │
│ 0x000_1000_0000 (4GB)                       │
│   ┌─────────────────────────────────────┐   │
│   │ ZONE_NORMAL (4 GB+)                 │   │  GFP_KERNEL — standard kernel
│   └─────────────────────────────────────┘   │
│   ┌─────────────────────────────────────┐   │
│   │ ZONE_MOVABLE (configurable)         │   │  — CMA, page migration
│   └─────────────────────────────────────┘   │
└─────────────────────────────────────────────┘
```

| Zone | Physical range | GFP flag | Notes |
|------|---------------|----------|-------|
| ZONE_DMA | 0–16 MB | `GFP_DMA` | Legacy ISA 24-bit DMA; rarely needed today |
| ZONE_DMA32 | 0–4 GB | `GFP_DMA32` | 32-bit DMA-capable devices (PCIe default) |
| ZONE_NORMAL | 4 GB+ | `GFP_KERNEL` | Standard kernel allocations |
| ZONE_MOVABLE | Configurable | — | Pages eligible for migration/CMA |

---

## GFP Flags (Get Free Pages)

**GFP flags** (Get Free Pages) are passed to every allocation function and control two things: which **memory zone** to allocate from, and what the allocator is allowed to do when memory is tight (sleep, reclaim, swap).

GFP flags control the allocation context and zone selection:

| Flag | Behavior |
|------|----------|
| `GFP_KERNEL` | May sleep, may reclaim; use in process context |
| `GFP_ATOMIC` | No sleep, no reclaim; use in interrupt or spinlock context; may return NULL |
| `GFP_DMA` | Must allocate from ZONE_DMA (0–16 MB) |
| `GFP_DMA32` | Must allocate from ZONE_DMA32 (0–4 GB) |
| `__GFP_ZERO` | Zero-fill the returned memory |
| `__GFP_NOFAIL` | Retry indefinitely until success; use sparingly |
| `__GFP_NOWARN` | Suppress OOM warning on allocation failure |
| `__GFP_COMP` | Return compound page (required for huge page backing) |

Key constraint: **never use `GFP_KERNEL` in interrupt context** or while holding a spinlock; it may sleep waiting for reclaim. Use `GFP_ATOMIC` in these paths and handle `NULL` return.

The GFP flag decision tree for driver authors:
1. Am I in interrupt context or holding a spinlock? → `GFP_ATOMIC`
2. Does the device's DMA engine address only 32-bit addresses? → `GFP_DMA32`
3. Is this a legacy ISA device needing sub-16MB memory? → `GFP_DMA`
4. Otherwise → `GFP_KERNEL`

> **Common Pitfall:** Using `GFP_KERNEL` in an interrupt service routine causes a kernel BUG splat ("sleeping function called from invalid context"). The ISR runs with interrupts disabled and cannot block waiting for memory reclaim. Always use `GFP_ATOMIC` in ISR context and pre-allocate buffers before the interrupt path is entered.

---

## CMA (Contiguous Memory Allocator)

With fragmentation in mind, CMA provides a clean solution: **reserve a physically contiguous region at boot time**, before any fragmentation occurs, and make it available to DMA devices on demand. CMA reserves physically contiguous regions at boot time within `ZONE_MOVABLE`. Normal movable pages can use this memory until a device requests it; at that point, the kernel migrates the movable pages out of the reserved region.

```
System Boot
┌────────────────────────────────────────────────────┐
│ Physical RAM                                       │
│ ┌──────────────┬──────────────────┬─────────────┐ │
│ │ Kernel image │   CMA reserved   │ General RAM │ │
│ │  (static)    │  (256MB @ 4GB)   │  (movable)  │ │
│ └──────────────┴──────────────────┴─────────────┘ │
└────────────────────────────────────────────────────┘

At runtime (before NVDLA needs it):
CMA region: [movable page][movable page][movable page]...
           (general OS pages using the reserved space)

When NVDLA driver calls dma_alloc_coherent():
  1. Kernel migrates movable pages out of CMA region
  2. Returns the now-free contiguous physical region to NVDLA
  3. NVDLA DMA engine has guaranteed contiguous PA
```

### Configuration

Kernel command line:
```
cma=256M
cma=128M@0x100000000   # reserve 128 MB starting at 4 GB PA
```

Device Tree (for platform-level assignment):
```dts
reserved-memory {
    linux,cma {
        compatible = "shared-dma-pool";
        reusable;
        size = <0 0x10000000>;
        linux,cma-default;
    };
};
```

### API

`dma_alloc_coherent(dev, size, &dma_handle, GFP_KERNEL)` uses CMA on platforms that support it. Returns a CPU virtual address and a device-visible `dma_handle` (physical address or IOVA). The region is non-cacheable or hardware-coherent.

This single API call abstracts the difference between platforms: on x86 with IOMMU, it returns pinned pages with an IOVA; on Jetson without IOMMU, it returns a physically contiguous PA. Driver code is portable across both.

CMA users on Jetson:
- NVDLA: input/output feature map DMA buffers
- VIC (Video Image Compositor): frame buffer transfers
- Camera ISP: raw frame DMA
- Display engine: framebuffer
- NVIDIA IOMMU-less DMA requires physically contiguous PA

### Monitoring

`/proc/cma`: total, used, maximum allocation. `dmesg | grep cma` at boot confirms reservation.

> **Key Insight:** CMA is invisible to drivers — they just call `dma_alloc_coherent()`. CMA's job happens transparently in the background, migrating ordinary pages out of the reserved region when a device needs it. The reservation at boot is what makes this reliable: no fragmentation has occurred yet.

---

## OOM Killer

When all reclaim paths (page cache eviction, swap, slab shrinkers) fail, the **OOM killer** selects and terminates a process. Understanding the OOM killer is critical for **embedded AI systems** where there is no swap and multiple processes compete for limited RAM.

Selection: `badness(task)` scores each process proportional to its RSS and swap usage. Adjustable per-process:
```bash
echo -1000 > /proc/$(pidof inferenced)/oom_score_adj   # protect
echo +500  > /proc/$(pidof bloated_loader)/oom_score_adj  # prefer killing
```
Range: -1000 (never kill) to +1000 (kill first). Score -1000 disables OOM kill entirely for that process.

Setting `oom_score_adj = -1000` on the inference daemon is **standard practice on embedded systems**. The inference daemon represents the primary system function — killing it during memory pressure is worse than killing a background data loader or log aggregator.

---

## Memory Pressure Tuning

These kernel parameters shape how aggressively the VM reclaims memory under pressure:

- `vm.swappiness` (0–200): tendency to swap anonymous pages vs. reclaiming page cache; 0 avoids swap, 100 is balanced, 200 prefers swap
- `vm.min_free_kbytes`: minimum free memory before `kswapd` reclaim activates; increase on embedded systems with no swap to avoid GFP_ATOMIC failures
- `vm.overcommit_memory`: 0 = heuristic, 1 = always allow overcommit, 2 = strict (committed ≤ RAM + swap × ratio)

On embedded inference systems with no swap, set `vm.swappiness=0` (avoid swap activity on a device that has no swap) and increase `vm.min_free_kbytes` to ensure that `GFP_ATOMIC` allocations in interrupt handlers always succeed.

---

## Summary

| Allocator | Contiguous PA? | Can sleep? | Max size | Primary use |
|-----------|---------------|-----------|---------|-------------|
| kmalloc / SLUB | Yes (up to ~8KB) | Depends on GFP | 8 KB typical | Kernel objects, driver structs |
| alloc_pages | Yes | Depends on GFP | 4 MB (order 10) | Huge page backing, large contiguous |
| vmalloc | No (virtual only) | Yes | vmalloc VA range (100s GB) | Firmware images, large non-DMA buffers |
| dma_alloc_coherent / CMA | Yes | Yes | CMA reservation size | DMA-capable device buffers |

### Conceptual Review

- **Why does fragmentation cause high-order buddy allocations to fail?** The buddy allocator requires 2^N physically contiguous pages. After runtime allocation and free activity, free pages are scattered non-contiguously. Even with 500MB free, there may be no single 2MB contiguous block.

- **What is the difference between `GFP_KERNEL` and `GFP_ATOMIC`?** `GFP_KERNEL` may sleep to wait for memory reclaim, so it can only be called in process context. `GFP_ATOMIC` never sleeps and is safe in interrupt context, but may return NULL if memory is tight. `GFP_ATOMIC` allocations consume a small emergency reserve.

- **Why does CMA reserve memory at boot rather than at driver probe time?** By the time a driver probes, the system has been running long enough for memory to fragment. Boot-time reservation guarantees a contiguous region before fragmentation begins.

- **What is the purpose of `SLAB_HWCACHE_ALIGN`?** It pads each slab object to a CPU cache line boundary. This prevents false sharing — two objects that happen to share a cache line being modified by different CPU cores, which would cause expensive cache coherence traffic.

- **When should you use `kmem_cache_create` instead of `kmalloc`?** When you allocate many objects of the same type repeatedly (e.g., per-frame descriptors in a camera driver). A dedicated cache has lower overhead, better memory locality, and shows clearly in `slabtop` for leak detection.

- **How does the OOM killer decide which process to kill?** It computes a `badness` score proportional to each process's RSS and swap usage, modified by `oom_score_adj`. The process with the highest score is killed first. Setting `oom_score_adj = -1000` prevents a process from ever being chosen.

---

## AI Hardware Connection

- CMA reserves physically contiguous memory at boot for Jetson NVDLA and VIC DMA buffers, ensuring availability before any user-space process has fragmented memory
- `GFP_DMA32` is required when writing PCIe device drivers for AI accelerator cards whose DMA engines cannot address physical memory above 4 GB
- `GFP_ATOMIC` is the correct flag for camera ISR buffer allocation in V4L2 drivers, where the interrupt handler cannot sleep waiting for reclaim
- OOM score protection (`oom_score_adj = -1000`) prevents the inference daemon (modeld, controlsd) from being killed under memory pressure in openpilot's embedded environment
- `slabtop` is the first diagnostic tool for identifying memory leaks in custom sensor drivers where per-frame slab allocations accumulate without matching frees
- `kvmalloc` is the preferred pattern in FPGA driver code that loads variable-size bitstreams, transparently selecting contiguous or non-contiguous backing based on availability
