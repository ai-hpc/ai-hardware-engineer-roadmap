# Lecture 12: Virtual Memory & the Linux Memory Model

## Overview

The core problem this lecture addresses is: how does every process get its own private, seemingly infinite memory space, even though the hardware has a fixed amount of physical RAM? Virtual memory is the OS's most fundamental abstraction — it makes each process believe it owns the entire address space, hides the details of physical memory layout, and lets the OS swap pages to disk when RAM runs low. The mental model to carry here is that of a hotel: each guest believes they have their own room (virtual address), but the hotel maps rooms to actual beds (physical memory) on demand, and can put luggage in storage (swap) when rooms are full. For an AI hardware engineer, virtual memory directly affects inference latency (page faults can add 10ms), GPU buffer sharing (zero-copy requires shared mappings), and memory accounting (RSS vs PSS determine whether your system will OOM).

---

## Virtual Address Space

Each process has a private **virtual address space** isolated from all other processes. The **MMU** translates virtual addresses to physical addresses per-page using the **page table tree**. No process can directly access another's physical memory without explicit kernel-mediated sharing.

Key properties:
- **Isolation**: a bad pointer in one process cannot corrupt another's memory
- **Overcommit**: total virtual memory across all processes can exceed physical RAM; pages allocated on demand
- **Abstraction**: uniform flat address space regardless of physical DRAM layout or fragmentation

> **Key Insight:** Virtual memory is not just a performance feature — it is a security and stability guarantee. Without it, a single runaway pointer in any process could corrupt the kernel or another process's memory. The isolation guarantee means that a crashing inference process cannot corrupt the control loop running in a separate process, which is why openpilot runs its components as separate processes.

---

## x86-64 Virtual Address Layout

```
┌─────────────────────────────────────────────────────────┐
│         x86-64 Virtual Address Space (per process)      │
├─────────────────────────────────────────────────────────┤
│  0xFFFFFFFFFFFFFFFF                                      │
│  ┌───────────────────────────────────┐                  │
│  │  Kernel Space (~128 TB)           │  (TTBR1 / CR3)   │
│  │  - Direct physical mapping        │                  │
│  │  - vmalloc region                 │                  │
│  │  - Kernel text + data             │                  │
│  │  - Module text                    │                  │
│  │  - vDSO / vsyscall                │                  │
│  └───────────────────────────────────┘                  │
│  0xFFFF800000000000  ◄── kernel base                    │
│                                                         │
│  [non-canonical hole — access triggers #GP fault]       │
│                                                         │
│  0x00007FFFFFFFFFFF  ◄── user space ceiling             │
│  ┌───────────────────────────────────┐                  │
│  │  Stack (grows downward ↓)         │                  │
│  │  ...                              │                  │
│  │  mmap region (grows downward ↓)   │  (.so, anon mmap)│
│  │  ...                              │                  │
│  │  Heap (grows upward ↑)            │  (malloc, brk)   │
│  │  BSS / Data segment               │  (global vars)   │
│  │  Text segment (code)              │  (read+exec)     │
│  └───────────────────────────────────┘                  │
│  0x0000000000000000                                      │
└─────────────────────────────────────────────────────────┘
```

```
0x0000000000000000 – 0x00007FFFFFFFFFFF   User space (~128 TB)
  [ text | data/BSS | heap → | ← mmap region | ← stack ]

0xFFFF800000000000 – 0xFFFFFFFFFFFFFFFF   Kernel space (~128 TB)
  [ direct physical mapping | vmalloc | kernel text | modules | vDSO ]
```

- **ASLR**: randomizes stack, heap, and mmap base addresses at process start; controlled via `/proc/sys/kernel/randomize_va_space`; disable on latency-sensitive deterministic systems
- **Canonical addresses**: bits 48–63 must sign-extend bit 47; non-canonical address triggers #GP fault; 57-bit VA (5-level paging) extends this to bit 56

---

## ARM64 Virtual Address Layout

| Register | Maps | Notes |
|---|---|---|
| `TTBR0_EL1` | User space (VA starting at 0x0) | Per-process; reloaded on every context switch |
| `TTBR1_EL1` | Kernel space (VA with high bits set) | Constant; always present for all processes |

- 4KB pages × 4 levels of page tables = 48-bit VA by default
- 64KB pages = 3 levels of page tables
- ARMv8.2 LPA extension: 52-bit VA; used on Cortex-A78, A710, Jetson Orin (Cortex-A78AE)

---

## Page Table Walk

The MMU translates a virtual address to a physical address by **walking a tree of page tables** in memory. On x86-64 with 4KB pages and **4-level paging**:

```
  4-Level Page Table Walk (x86-64, 48-bit VA):

  Virtual Address (48 bits):
  ┌───────┬───────┬───────┬───────┬────────────────┐
  │PGD idx│PUD idx│PMD idx│PTE idx│  Page Offset   │
  │  9b   │  9b   │  9b   │  9b   │     12b        │
  └───────┴───────┴───────┴───────┴────────────────┘

  CR3 register
  (physical addr of PGD)
       │
       ▼
  ┌──────────┐         ┌──────────┐         ┌──────────┐         ┌──────────┐
  │  PGD     │ ──────► │  PUD     │ ──────► │  PMD     │ ──────► │  PTE     │
  │(Page     │         │(Page     │         │(Page     │         │(Page     │
  │ Global   │         │ Upper    │         │ Middle   │         │ Table    │
  │ Dir)     │         │ Dir)     │         │ Dir)     │         │ Entry)   │
  └──────────┘         └──────────┘         └──────────┘         └────┬─────┘
                                                                       │
                                                                       ▼
                                                              Physical Page Base
                                                              + Page Offset
                                                              = Physical Address

  Each level lookup: physical_addr = table[index].pfn << PAGE_SHIFT
  TLB caches the final PGD→physical mapping to avoid repeated walks
```

The **TLB** (Translation Lookaside Buffer) caches recent virtual→physical translations. A **TLB miss** requires a full 4-level walk through RAM — 4 memory accesses. **TLB shootdowns** (flushing remote CPUs' TLBs after a mapping change) require inter-processor interrupts and add ~10–30µs on large SMP systems.

---

## mm_struct — Process Memory Descriptor

`struct mm_struct` describes a process's **complete virtual address space**. **One instance per process**; all threads of a process share the same `mm_struct`.

| Field | Meaning |
|---|---|
| `pgd` | Physical address of Page Global Directory (top-level page table) |
| `mmap` | Linked list of all `vm_area_struct` entries |
| `mm_rb` | Red-black tree of VMAs for O(log n) address lookup |
| `start_code`, `end_code` | Text segment bounds |
| `start_data`, `end_data` | Initialized data segment bounds |
| `start_brk`, `brk` | Heap bounds (`brk` grows upward via `sbrk()`/`brk()` syscall) |
| `start_stack` | Initial stack pointer value |
| `total_vm` | Total virtual pages mapped |
| `locked_vm` | Pages pinned by `mlock`/`mlockall` |

---

## vm_area_struct (VMA)

A **VMA** represents **one contiguous, uniformly-mapped region** of virtual address space. Think of it as a "chapter" in the address space book — each chapter has a start and end, a set of **permissions**, and an optional **backing file**.

| Field | Meaning |
|---|---|
| `vm_start`, `vm_end` | Address range (page-aligned; exclusive end) |
| `vm_flags` | `VM_READ`, `VM_WRITE`, `VM_EXEC`, `VM_SHARED`, `VM_LOCKED`, `VM_GROWSDOWN` |
| `vm_file` | Pointer to backing `struct file` (NULL for anonymous mappings) |
| `vm_pgoff` | Offset within the backing file (in pages) |
| `vm_ops` | VMA operations: `fault()`, `open()`, `close()`, `mmap()` |

---

## VMA Types

| VMA Type | `vm_flags` | Backed By | Fault Action | Example |
|---|---|---|---|---|
| Text (code) | `VM_READ\|VM_EXEC` | Executable / `.so` file | Read from file | `main()` code |
| Data/BSS | `VM_READ\|VM_WRITE` | Executable file / zero-fill | Read from file or zero | Global variables |
| Heap | `VM_READ\|VM_WRITE` | Anonymous | Zero-fill on demand | `malloc()` regions |
| Stack | `VM_READ\|VM_WRITE\|VM_GROWSDOWN` | Anonymous | Zero-fill; expand on fault | Thread stack |
| File mmap (shared) | `VM_READ\|VM_WRITE\|VM_SHARED` | File pages | Read from file | Model weights mmap |
| Anonymous shared | `VM_READ\|VM_WRITE\|VM_SHARED` | `tmpfs` / swap | Zero-fill | POSIX shm, VisionIPC |
| vDSO | `VM_READ\|VM_EXEC` | Kernel-provided | Pre-mapped | `gettimeofday()` |

---

## mmap() — Creating Mappings

`mmap()` is the system call that **creates VMAs**. It is used for **file-backed mappings**, **anonymous memory**, and **shared memory** between processes.

```c
// File-backed, shared — changes visible to all processes mapping same file
// Ideal for model weights: load once, share read-only across worker processes
void *p = mmap(NULL, length, PROT_READ|PROT_WRITE, MAP_SHARED, fd, offset);

// Anonymous, private — zero-filled; not backed by file; used for heap/stack extensions
void *p = mmap(NULL, length, PROT_READ|PROT_WRITE, MAP_PRIVATE|MAP_ANONYMOUS, -1, 0);

// Anonymous, shared — IPC between related processes; backed by tmpfs
// Used in VisionIPC for camera frame ring buffers shared between processes
void *p = mmap(NULL, length, PROT_READ|PROT_WRITE, MAP_SHARED|MAP_ANONYMOUS, -1, 0);

// HUGE pages — 2MB pages; reduce TLB pressure for large model buffers
// A 1GB model with 4KB pages requires 262144 TLB entries; with 2MB pages only 512
void *p = mmap(NULL, length, PROT_READ|PROT_WRITE,
               MAP_PRIVATE|MAP_ANONYMOUS|MAP_HUGETLB, -1, 0);
```

`mmap()` creates the VMA descriptor but does **not** allocate physical pages. Pages are allocated **on first access** via **demand paging**.

> **Key Insight:** The fact that `mmap()` returns immediately without allocating physical pages is what enables overcommit. The kernel makes a virtual promise — "here is address space" — and only fulfills the physical backing when each page is first touched. This is efficient but dangerous for real-time: the first access to any unmapped page causes a page fault.

---

## Demand Paging

Page table entry (PTE) is initially absent. On first access, the CPU raises a **page fault**. The kernel's `do_page_fault()` handler:

1. Finds the VMA covering the faulting virtual address — O(log n) lookup in the red-black tree
2. Validates permissions (`vm_flags` vs. fault type — read/write/execute)
3. Allocates a physical page frame from the free list
4. Fills the frame: zero-fill (anonymous), read from file (file-backed), swap-in (previously evicted)
5. Installs the PTE in the process's page table and flushes the local TLB entry
6. Returns to user space; the faulting instruction retries transparently

| Fault Type | Cause | Typical Cost |
|---|---|---|
| Minor (soft) | Page in memory but PTE absent (COW, demand-zero) | ~1µs |
| Major (hard) | Page must be read from disk (file-backed or swap) | 1–10ms |

**Major faults are fatal** to real-time latency guarantees. `mlockall()` prevents them entirely.

> **Common Pitfall:** Many developers call `mlockall()` and assume all page faults are eliminated. But `mlockall(MCL_FUTURE)` only prevents *future* pages from being evicted — pages that have never been faulted in are still demand-paged on first access. You must pre-touch all buffers after `mlockall()` to eliminate minor faults as well. A missed pre-touch is a latency spike waiting to happen.

---

## Copy-on-Write (COW)

After `fork()`, parent and child **share all pages mapped read-only**. On **first write** to a shared page:

1. Hardware page fault triggered (write to read-only PTE)
2. Kernel allocates a new page frame
3. Copies original page content to new frame
4. Installs a writable PTE in the writing process's page table
5. Original page remains shared and unchanged

```
  Copy-on-Write after fork():

  BEFORE first write:                AFTER child writes to page X:
  Parent PTE ──► Physical Page X     Parent PTE ──► Physical Page X (unchanged)
  Child PTE  ──► Physical Page X     Child PTE  ──► Physical Page X' (private copy)
  (read-only shared)                 (writable, new allocation)
```

COW makes `fork()` **O(1)** — the address space size does not matter; **only written pages** are physically duplicated. Model weights shared read-only via COW after `fork()` consume **only one physical copy**.

> **Key Insight:** COW is why forking a process with a 1GB model loaded in memory does not require 1GB of additional RAM. Both parent and child share the same physical pages for the model weights. Only pages that one process actually modifies (like stack frames and heap allocations) get duplicated. This is the mechanism that lets openpilot restart `modeld` without doubling physical RAM usage.

---

## mlock() / mlockall() — Pinning Pages in RAM

```c
mlockall(MCL_CURRENT | MCL_FUTURE);   // pin all current + future mappings
// MCL_CURRENT: locks all pages that are currently mapped and present
// MCL_FUTURE: any future mmap() or heap growth is automatically locked on fault-in

mlock(addr, length);                   // pin specific address range only
munlockall();                          // release all locks; allows eviction again
```

- `MCL_CURRENT`: locks all pages currently mapped in the address space
- `MCL_FUTURE`: any future `mmap()` or heap growth is automatically locked
- Requires `CAP_IPC_LOCK` or `RLIMIT_MEMLOCK` raised sufficiently

After `mlockall()`, pre-touch all pages to eliminate minor faults as well:

```c
// Force demand-zero pages to be physically allocated immediately
// Without this, the first access to each page still causes a minor page fault (~1µs)
memset(buffer, 0, buffer_size);
// After memset, buffer pages are faulted in, PTEs installed, and mlocked —
// subsequent accesses have zero fault overhead
```

---

## Inspecting Virtual Memory

Understanding what is actually in your process's virtual address space is essential for debugging memory issues, OOM failures, and unexpected page faults.

```bash
cat /proc/<pid>/maps           # all VMAs: address range, permissions, backing file
cat /proc/<pid>/smaps          # per-VMA detail: RSS, PSS, dirty, swap, AnonHugePages
cat /proc/<pid>/smaps_rollup   # per-process totals from smaps
pmap -x <pid>                  # formatted smaps output
```

| Metric | Definition |
|---|---|
| RSS | Resident Set Size: total physical pages mapped (double-counts shared pages) |
| PSS | Proportional Set Size: RSS but shared pages divided proportionally; correct per-process metric |
| Swap | Pages evicted to swap currently |
| AnonHugePages | Anonymous pages backed by 2MB transparent huge pages |

**PSS is the correct metric** for per-process memory accounting in multi-process systems where shared libraries would otherwise be **double-counted by RSS**.

```
  RSS vs PSS for shared libraries:

  libcuda.so (100MB physical):
  ├── Process A: RSS counts 100MB
  ├── Process B: RSS counts 100MB
  └── Total RSS: 200MB ← WRONG (only 100MB physical)

  PSS: split proportionally (3 processes sharing):
  ├── Process A: PSS counts 33MB
  ├── Process B: PSS counts 33MB
  ├── Process C: PSS counts 33MB
  └── Total PSS: 100MB ← CORRECT
```

---

## Summary

| VMA Type | Flags | Backed By | Fault Action | Example |
|---|---|---|---|---|
| Text segment | `R-X` | ELF / `.so` file | Read from file | Code pages |
| Data/BSS | `RW-` | ELF file / zero | File read or zero-fill | Global vars |
| Heap | `RW-` | Anonymous | Zero-fill on demand | `malloc` |
| Stack | `RW-` (growsdown) | Anonymous | Zero-fill, auto-extend | Thread stack |
| File mmap (shared) | `RWS` | File | Read from file | Model weight file |
| Anonymous shared | `RWS` | `tmpfs` | Zero-fill | VisionIPC ring buffer |
| vDSO | `R-X` | Kernel-provided | Pre-mapped at boot | `clock_gettime` |

### Conceptual Review

- **Why does `mmap()` return immediately even for a 1GB mapping?** Because `mmap()` only creates the VMA descriptor — it makes no physical memory allocation. Physical pages are allocated lazily on first access via demand paging. This is what makes overcommit possible and `fork()` fast.
- **What is the difference between a minor and major page fault?** A minor fault means the physical page is already in RAM but the PTE is absent (e.g., after `fork()` COW or demand-zero). A major fault means the page must be read from disk (file-backed or swap). Minor faults cost ~1µs; major faults cost 1–10ms and are fatal to real-time guarantees.
- **Why does `mlockall()` require a subsequent memset to fully eliminate faults?** `mlockall(MCL_FUTURE)` pins pages once they are faulted in and prevents eviction. But pages that have never been accessed are not yet in RAM — they are still demand-paged on first touch, causing minor faults. Memset forces all pages to be faulted in and locked simultaneously.
- **Why is PSS the correct metric and not RSS?** RSS double-counts shared pages. A shared library like `libcuda.so` might appear as 100MB in 10 different process's RSS figures, giving a total of 1GB — but only 100MB of physical memory is actually used. PSS divides shared pages proportionally and gives an accurate sum.
- **How does COW enable `fork()` to be O(1) regardless of address space size?** After `fork()`, all pages are marked read-only and shared. No copying occurs. The child starts executing immediately. Only when either the parent or child *writes* to a shared page does a private copy get made — and only for that one page. For a process with 1GB of model weights that are never written, zero copying ever occurs.
- **What does the page table walk do, and when is it avoided?** The TLB caches recent VA→PA translations. If the TLB contains the mapping, the MMU uses it directly. On a TLB miss, the MMU walks the 4-level page table in RAM (4 memory reads). TLB shootdowns (during munmap, fork, or process exit) require IPIs to remote CPUs and can add 10–30µs to unrelated threads.

---

## AI Hardware Connection

- `mmap(MAP_SHARED)` on a DMA-BUF file descriptor maps GPU-allocated DRAM into a CPU process's virtual address space for zero-copy buffer sharing between V4L2 camera capture and CUDA inference; no data copy crosses the PCIe bus
- `mlockall(MCL_CURRENT|MCL_FUTURE)` is called at startup in real-time inference daemons on Jetson Orin to eliminate major page faults from the inference loop; a single fault to swap during a forward pass can add 10ms of latency
- COW `fork()` allows openpilot to spawn child processes sharing the parent's model weight pages in physical memory without doubling RAM; on a 512MB weight tensor, this avoids a 512MB physical copy on every process restart
- Shared anonymous `mmap` (`MAP_SHARED|MAP_ANONYMOUS`) backed by `memfd_create` provides the inter-process camera frame ring buffer in VisionIPC; `camerad` writes frames directly into the shared region and `modeld` reads without any copy
- `/proc/<pid>/smaps` PSS is the correct memory metric for per-process accounting in multi-process edge AI systems; RSS would count each shared library (libcuda, libtorch) multiple times across `camerad`, `modeld`, and `controlsd`
- CUDA Unified Virtual Memory (UVM) uses the same demand-paging mechanism as Linux VM: GPU page faults trigger CPU→GPU or GPU→CPU page migrations via the CUDA driver's fault handler, transparently mapping GPU and CPU virtual address spaces to the same physical memory
