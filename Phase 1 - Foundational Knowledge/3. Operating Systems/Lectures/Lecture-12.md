# Lecture 12: Virtual Memory — Mappings, Faults, and Ownership

## An address is meaningful in an address space

Two processes can both use virtual address `0x400000` while mapping different physical storage there. Conversely, two different virtual addresses can refer to the same shared page. A pointer from one process is therefore not automatically a valid pointer in another, and neither is automatically a DMA address.

The MMU applies translation and permissions for CPU memory access. This provides process isolation when configured correctly, but explicit shared mappings, authorized debugging, drivers, and kernel vulnerabilities are separate considerations. It is not a guarantee that a faulty process cannot affect another through shared state, resource exhaustion, or device access. Keep a robot's control interface bounded even when perception runs in a separate process.

## Distinguish three layers

| Layer | Example question | Example observation |
|---|---|---|
| Virtual region | Is this address readable, writable, shared, or file-backed? | `/proc/PID/maps` |
| Translation/backing | Does this range currently map resident pages? | Relevant `/proc/PID/smaps` fields and faults |
| Physical/device ownership | May CPU or device access the storage now? | Allocator/DMA/runtime contract |

A Linux VMA describes a virtual range with compatible attributes; its end address is exclusive. Current kernels track VMAs in a **Maple Tree** in `mm_struct`, rather than the historical `mm_rb` red-black tree model. Threads sharing an address space share its memory context. Kernel-internal fields and locking rules are not stable user-space APIs. See [process address management](https://docs.kernel.org/mm/process_addrs.html).

## Architecture examples have assumptions

For a conventional x86-64 four-level, 4 KiB-page example, 48 address bits split as `9 + 9 + 9 + 9 + 12`: four table indices plus a byte offset. Linux's generic five-level naming folds unused levels. Five-level x86 paging changes the virtual address width; an Arm configuration can use different granules and table depths.

`CR3` belongs to x86; `TTBR0_EL1` and `TTBR1_EL1` belong to Arm. Do not combine their names in one register diagram. The vDSO is a kernel-provided mapping accessible in user space, not ordinary kernel-only text. Kernel page-table isolation and architecture configuration also affect which mappings are active. Consult the [x86 layout](https://docs.kernel.org/arch/x86/x86_64/mm.html) and [AArch64 layout](https://docs.kernel.org/arch/arm64/memory.html) for the actual kernel. Leave ASLR enabled for this course; deterministic test inputs do not require disabling address randomization.

A TLB miss is not a page fault: a valid translation may be found by a hardware page-table walk. Page-walk caches and normal caches can serve intermediate entries, so a four-level walk does not necessarily issue four DRAM accesses. [Lecture 13](Lecture-13.md) develops this distinction.

## Mapping memory is not the same as touching it

An ordinary anonymous mapping often obtains physical backing lazily. However, mapping flags, memory locking, existing file cache, device mappings, and explicit population change behavior. “`mmap` never allocates pages” is too strong. Always check for `MAP_FAILED` in C, and retain the length for `munmap`.

A file-backed read-only model mapping can share cached file pages across workers. `MAP_PRIVATE` prevents writes through the mapping from updating the underlying file; a write can create a private copy. `MAP_SHARED` enables shared updates subject to permissions, but does not supply a lock, a message protocol, or a transaction. Sharing bytes between processes still needs synchronization and lifetime rules.

This small Python example uses a temporary file and shows the distinction without modifying any user file:

```python
import mmap
import tempfile

with tempfile.TemporaryFile() as file:
    file.write(b"ABCD")
    file.flush()
    with mmap.mmap(file.fileno(), 4, access=mmap.ACCESS_COPY) as private:
        private[0] = ord("Z")
        assert private[:] == b"ZBCD"
    file.seek(0)
    assert file.read() == b"ABCD"
    with mmap.mmap(file.fileno(), 4, access=mmap.ACCESS_WRITE) as shared:
        shared[1] = ord("Y")
        shared.flush()
    file.seek(0)
    assert file.read() == b"AYCD"
print("private copy and shared file update verified")
```

This example was executed on the Linux review host with Python 3.12.3. It checks mapping behavior, not multi-process ordering or crash durability.

## Page faults and copy-on-write

A fault transfers control to the kernel because access cannot proceed with the current translation or permissions. A valid access may need demand-zero allocation, copy-on-write, a file page, or swap-in. An invalid access may instead cause a signal. “Minor” means no required storage I/O for servicing that fault; it does not mean zero cost. Major faults involve I/O. Costs depend on memory pressure, mapping type, storage, and scheduling; do not assign universal 1 µs or 10 ms numbers.

Copy-on-write can avoid immediately copying all private user data when `fork()` creates a child. It still requires process metadata and page-table work, so `fork()` is **not O(1) independent of address-space size**. Shared mappings and special mapping flags also have their own semantics. The [fork manual](https://man7.org/linux/man-pages/man2/fork.2.html) describes Linux's implementation and inheritance rules.

**Worked count:** assume a 16 MiB private region backed by 4 KiB pages. It contains 4096 pages. If the child writes one byte in each of 10 distinct pages that require copying, the copied page payload is 40 KiB, not 10 bytes and not automatically the full 16 MiB. Metadata and allocator overhead are additional; huge pages and kernel optimizations can change the actual work.

## Memory locking has a precise scope

`mlock`/`mlockall` prevent covered pages from being paged out under their documented conditions. Without on-fault behavior, locking can populate pages; `MCL_ONFAULT` explicitly changes when population happens. The simplistic claim that every successful `mlockall` leaves all pages demand-paged is wrong.

Check the call, limits, and flags; later heap growth or stack expansion may fail under `MCL_FUTURE` if locking limits are exceeded. Fork/copy-on-write and newly touched thread stacks require care. Locking is not the same as DMA pinning, device mapping, or a promise of no future faults from arbitrary code. See the [memory-locking manual](https://man7.org/linux/man-pages/man2/mlock.2.html).

For a bounded critical path, allocate and initialize its actual buffers and thread stacks, warm required paths, inspect fault counters, and avoid uncontrolled allocation during the measured interval. The checked [rt-demo](demos/rt-demo/README.md) makes denied locking visible and explains its limited prefault coverage.

## Account for the question you are asking

RSS counts resident mapped pages for a process; summing RSS across processes can count shared storage repeatedly. PSS apportions shared pages among mappings/processes according to kernel accounting. Neither directly equals all system memory use, GPU allocations, or a service's cgroup charge.

**Example:** A and B each have 8 MiB private resident memory and both map the same 12 MiB resident shared region. Each has RSS 20 MiB; total RSS is 40 MiB. With exactly those two sharers, each PSS is 14 MiB and their PSS sum is 28 MiB. Physical payload in this simplified example is `8 + 8 + 12 = 28 MiB`. Page tables and other overhead are excluded.

Inspect `/proc/PID/smaps_rollup` and per-region `smaps` while the process is alive. Permissions may restrict another process's data. The [proc documentation](https://docs.kernel.org/filesystems/proc.html) defines fields; do not infer a CUDA allocation's full storage from a host VMA alone.

## Retrieval questions

1. Does a successful 1 GiB mapping prove 1 GiB is resident? **No; inspect backing and the mapping's population contract.**
2. Does a TLB miss require file I/O? **No; a valid page-table mapping may suffice.**
3. Is a shared mapping a safe concurrent queue? **No; it supplies shared bytes, not ownership or ordering.**
4. Does `fork()` immediately duplicate every model-weight byte? **Usually copy-on-write avoids that for eligible private mappings, but metadata work and future writes still cost memory.**
5. Does process RSS equal accelerator memory use? **No; use the allocator/runtime and system accounting relevant to the device.**
