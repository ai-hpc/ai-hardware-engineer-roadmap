# Lecture 3: Interrupts, Exceptions & Bottom Halves

## Overview

Hardware does not wait for software to ask it questions — it signals the CPU asynchronously when something needs attention. The core challenge this lecture addresses is: how does the kernel respond to hardware events (a camera frame arriving, a GPU job completing, a network packet landing) quickly enough that nothing is dropped, while also not monopolizing the CPU for bookkeeping? The mental model is a **two-stage pipeline**: a fast top half that acknowledges the hardware in microseconds, and a flexible bottom half that does the real work later without blocking normal execution. For an AI hardware engineer, interrupts are the foundation of every camera pipeline, GPU completion event, and CAN bus message — misunderstanding them leads to dropped frames, latency spikes, and driver hangs that are very hard to debug after the fact.

---

## Interrupts vs Exceptions

Both **interrupts** and **exceptions** cause the CPU to stop what it is doing and run kernel code, but they differ in **origin and timing**.

| Type | Origin | Synchronous? | Example |
|---|---|---|---|
| Hardware interrupt | External device asserts IRQ line | No (async) | NIC packet arrives, GPU job done, camera frame end |
| Software interrupt | CPU executes INT/SVC instruction | Yes (sync trap) | System call, debug breakpoint |
| Exception (fault) | CPU detects error during instruction | Yes (sync) | Page fault, divide-by-zero, GP fault |
| Exception (abort) | Unrecoverable hardware error | Yes (sync) | Machine check, double fault |

**Faults** re-execute the faulting instruction after the handler resolves the error (e.g., page fault installs a PTE). **Traps** advance to the next instruction after the handler returns (e.g., syscall). **Aborts** do not return.

> **Key Insight:** The distinction between "fault" and "trap" has a critical practical consequence. A page fault is a fault — the hardware re-executes the faulting memory access after the kernel installs a valid page table entry, so the user program never knows it happened. A system call is a trap — the kernel executes the service and returns to the instruction *after* the `SYSCALL` instruction. Getting this wrong in a custom exception handler means either re-executing an instruction that shouldn't be repeated, or skipping one that should.

---

## Interrupt Controllers

### ARM GIC (Generic Interrupt Controller v3)

Used on Jetson Orin, Qualcomm SoCs, NXP i.MX.

| Interrupt type | ID range | Description |
|---|---|---|
| SGI (Software Generated) | 0–15 | IPI — one CPU signals another; used by scheduler migration and TLB shootdowns |
| PPI (Per-CPU Private) | 16–31 | Per-core timers, PMU (performance monitoring) |
| SPI (Shared Peripheral) | 32–1019 | All external device interrupts: cameras, GPUs, NVMe, CAN |

**Priority levels**: 0 (highest) to 255 (lowest). CPU interface register `PMR` (Priority Mask Register) masks interrupts below a threshold — used during spinlock-held sections.

```c
gic_send_sgi(target_cpu, sgi_id);  /* trigger IPI from kernel/smp.c */
```

### x86 APIC

- Local APIC per CPU: receives IPIs and local timer interrupts
- I/O APIC: routes external device IRQs to specific CPUs
- IDT (Interrupt Descriptor Table): 256 entries; 0–31 reserved for CPU exceptions; 32–255 for external IRQs and software traps
- TPR (Task Priority Register): masks interrupts at or below a priority level

---

## MSI and MSI-X (PCIe)

Legacy wire-based IRQs share physical lines, limiting parallelism. PCIe **MSI** (Message Signaled Interrupts) replaces line assertion with a **memory write to a special MMIO address** — the CPU's APIC/GIC interprets the write as an interrupt.

Think of legacy IRQ lines as a single **shared phone line** — only one caller at a time. **MSI-X** gives each device queue its own **dedicated phone line** with a dedicated recipient CPU.

| Feature | MSI | MSI-X |
|---|---|---|
| Max vectors per device | 32 | 2048 |
| Per-vector CPU affinity | No | Yes (each vector independently affinable) |
| Vector table location | Capability register | Separate BAR region |

### Why MSI-X Matters for AI Hardware

- **NVMe**: 32+ queues each get a dedicated MSI-X vector, pinned to the core running that queue's I/O — eliminates contention during large model weight loading via GPUDirect Storage.
- **NVIDIA GPU**: separate MSI-X vectors for CUDA compute engine completion, copy engine completion, and fault notification — CUDA's event system is built on per-engine MSI-X delivery.
- **NICs (100GbE)**: per-TX/RX-queue MSI-X enables multi-queue RSS without lock contention.

```bash
cat /proc/interrupts | grep nvidia   # per-CPU counts for each GPU MSI-X vector
cat /proc/interrupts | grep nvme     # per-queue NVMe completion counts
```

> **Key Insight:** When a CUDA kernel finishes on a GPU, the GPU writes a completion value to a memory-mapped register. The PCIe bridge converts this into an MSI-X write to the CPU's APIC. The CPU fires the interrupt, wakes the CUDA runtime thread. The entire chain from GPU completion to userspace wakeup passes through MSI-X. If the MSI-X vector is routed to the wrong CPU (one that is busy with other work), the wakeup latency increases by 10–50 µs. Pinning the MSI-X vector to the CUDA stream management core eliminates this.

---

## Interrupt Handling Flow

The full path from hardware event to kernel handler is **deterministic** and takes the same path every time. Understanding this path helps you instrument the right points for **latency measurement**.

```
Hardware Interrupt Flow — Full Path
┌────────────────────────────────────────────────────────┐
│  HARDWARE                                              │
│  Camera sensor → frame-end pulse → CSI controller     │
│  → GIC SPI assertion                                  │
└────────────────────┬───────────────────────────────────┘
                     │  IRQ line asserted
                     ▼
┌────────────────────────────────────────────────────────┐
│  INTERRUPT CONTROLLER (GIC / APIC)                     │
│  1. Assigns IRQ number                                 │
│  2. Selects target CPU (affinity mask)                 │
│  3. Signals selected CPU                               │
└────────────────────┬───────────────────────────────────┘
                     │
                     ▼
┌────────────────────────────────────────────────────────┐
│  CPU — TOP HALF (hardirq context)                      │
│  4. Completes current instruction                      │
│  5. Saves minimal state to kernel stack (PC, SP, regs) │
│  6. Looks up handler:                                  │
│     x86: IDT[irq_number] → ISR function pointer       │
│     ARM64: VBAR_EL1 vector table entry                 │
│  7. ISR runs: acknowledge HW, save data ptr,           │
│     schedule bottom half (raise_softirq / queue_work) │
│  8. EOI (End Of Interrupt) written to GIC/APIC         │
└────────────────────┬───────────────────────────────────┘
                     │
                     ▼
┌────────────────────────────────────────────────────────┐
│  BOTTOM HALF (softirq / workqueue / threaded IRQ)      │
│  9. Process captured data                              │
│  10. Wake waiting user-space processes (camerad)       │
│  11. Return to previously preempted context            │
└────────────────────────────────────────────────────────┘
```

The numbered sequence in detail:

1. **Device asserts IRQ**: the camera sensor's frame-end pin goes high; the GIC sees it.
2. **GIC selects target CPU**: using the affinity configuration from `/proc/irq/N/smp_affinity`.
3. **CPU finishes current instruction**: the CPU does not drop mid-instruction; it completes the current operation first.
4. **State saved to kernel stack**: the CPU hardware automatically pushes the minimal register state (PC, SP, PSR/RFLAGS) to the current process's kernel stack.
5. **Vector table lookup**: x86 reads the IDT; ARM64 jumps to the VBAR_EL1 vector entry for the appropriate exception class.
6. **ISR (top half) runs**: acknowledges the hardware register (clears the interrupt flag in the device), saves a pointer to the received data, schedules deferred work.
7. **EOI written**: signals the interrupt controller that this CPU has finished handling the interrupt; allows the controller to deliver the next interrupt.
8. **Softirqs/workqueues run**: the actual data processing happens here, with interrupts re-enabled.

---

## Top Half vs Bottom Half

| Half | Context | Can sleep? | Goal |
|---|---|---|---|
| Top half (hardirq / ISR) | IRQs disabled on local CPU | No | Acknowledge hardware; schedule deferred work; must complete in < 1 µs ideal |
| Bottom half | IRQs re-enabled | Depends on mechanism | Process data; wake waiters; do actual work |

The **ISR must be minimal**: acknowledge the interrupt controller, save a pointer to received data, and schedule a bottom half. All I/O processing happens in **bottom halves**.

> **Key Insight:** The reason the top half must be short is that while it runs, the local CPU cannot receive any other interrupts (they are disabled). If a camera frame ISR takes 100 µs to process a frame, no other interrupts on that CPU can be acknowledged during that window — including the scheduler tick, which means other RT tasks cannot be woken. Long ISRs create latency bubbles that appear as mysterious delays in completely unrelated processes.

---

## Bottom-Half Mechanisms

There are three main **bottom-half mechanisms** in Linux, with different tradeoffs between latency, flexibility, and compatibility. Choosing the right one matters for **driver correctness and performance**.

### Softirqs

- 10 statically defined types: `NET_TX`, `NET_RX`, `BLOCK`, `TASKLET`, `SCHED`, `HRTIMER`, `RCU`, and others
- Run in interrupt context immediately after hardirq completion; can run concurrently on multiple CPUs simultaneously (no per-softirq lock)
- Cannot sleep; cannot be dynamically added without patching the kernel
- When backlog grows too large, `ksoftirqd/N` kernel threads drain the queue to bound latency

```c
raise_softirq(NET_RX_SOFTIRQ);   /* schedule NET_RX softirq from ISR */
```

### Tasklets

- Dynamically allocated; built on `TASKLET_SOFTIRQ`
- Serialized per instance — same tasklet cannot run on two CPUs simultaneously
- **Deprecated** in new driver code since Linux 5.x; migrate to workqueues or threaded IRQs

### Workqueues

Deferred work executed in **kernel threads** (`kworker/N:M`). **Process context** — can sleep, allocate memory, take mutexes.

```c
INIT_WORK(&work, handler_fn);         /* initialize work item; binds handler function */
schedule_work(&work);                  /* queue onto system_wq; runs in next available kworker */
schedule_work_on(cpu, &work);         /* queue to specific CPU's kworker; avoids migration */

/* Dedicated high-priority workqueue for camera ISP completion */
wq = alloc_workqueue("isp_done", WQ_HIGHPRI | WQ_UNBOUND, 1);
/* WQ_HIGHPRI: kworker runs at nice -20; WQ_UNBOUND: not tied to a specific CPU */
queue_work(wq, &work);
```

This code creates a dedicated high-priority workqueue for camera ISP completions. The `WQ_HIGHPRI` flag ensures the kernel worker thread processes ISP completions before normal-priority work, reducing the delay between frame capture and when `camerad` can dequeue the buffer.

| Workqueue | Threads | Use |
|---|---|---|
| `system_wq` | Shared `kworker` pool | General deferred work |
| `system_highpri_wq` | High-priority kworkers | Latency-sensitive paths (camera, CAN) |
| `system_unbound_wq` | CPU-unbound | Work that should migrate freely |
| Custom via `alloc_workqueue()` | Dedicated | Exclusive queue for one driver |

### Threaded IRQs

```c
request_threaded_irq(irq, hard_handler, thread_fn,
                     IRQF_SHARED, "cam-frame-done", dev);
/* hard_handler: runs in hardirq context — must be minimal */
/* thread_fn: runs in kernel thread "irq/N-cam-frame-done" — can sleep */
```

- `hard_handler`: minimal hardirq context — acknowledges hardware, returns `IRQ_WAKE_THREAD`
- `thread_fn`: runs in dedicated kernel thread (`irq/N-cam-frame-done`) — can sleep, take locks, call `v4l2_buffer_done()`
- **Preferred for all new drivers**; required under `PREEMPT_RT` (mainlined in Linux 6.12)
- Thread priority is settable via `chrt`, enabling bounded latency via the RT scheduler
- `IRQF_NO_THREAD`: forces true hardirq context even under PREEMPT_RT — use only for interrupt controllers and `hrtimer`

```
Bottom-Half Mechanism Comparison
┌──────────────┬──────────────┬───────────────┬──────────────────┐
│  Mechanism   │   Context    │  Can Sleep?   │  Latency Target  │
├──────────────┼──────────────┼───────────────┼──────────────────┤
│  Softirq     │ IRQ context  │      No       │    < 10 µs       │
│  Tasklet     │ Via softirq  │      No       │    < 10 µs       │
│  (deprecated)│              │               │                  │
│  Workqueue   │  kworker     │     Yes       │   10–100s µs     │
│  Threaded IRQ│  irq/N thd   │     Yes       │  RT-schedulable  │
└──────────────┴──────────────┴───────────────┴──────────────────┘
```

> **Key Insight:** Threaded IRQs are the modern answer to a fundamental tension: hardirq context needs to be fast and cannot sleep, but real work (processing a camera frame, completing a DMA transfer) often needs to allocate memory, take a mutex, or call sleeping APIs. By moving the work to a kernel thread, threaded IRQs get the best of both worlds — they still respond quickly (the hard handler acknowledges the hardware immediately), but the actual processing runs in a schedulable thread with bounded priority.

> **Common Pitfall:** Using `request_irq()` instead of `request_threaded_irq()` for a driver that calls sleeping functions in its ISR will cause `BUG: scheduling while atomic` kernel warnings and potential crashes. Any function that can block (mutex_lock, msleep, copy_to_user, GFP_KERNEL allocation) is forbidden in hardirq context. Under `PREEMPT_RT`, this becomes even stricter — spinlocks become sleeping locks, so nearly everything in a hardirq ISR can accidentally sleep.

---

## Interrupt Affinity

```bash
cat /proc/irq/42/smp_affinity         # hex bitmask (e.g., 0x8 = CPU 3)
cat /proc/irq/42/smp_affinity_list    # human-readable (e.g., "3")
echo 8 > /proc/irq/42/smp_affinity   # pin IRQ 42 to CPU 3
systemctl stop irqbalance             # prevent irqbalance from overriding manual settings
```

Pinning a GPU MSI-X completion vector to the same core as the CUDA stream management thread eliminates **cross-core wakeup latency** of 10–50 µs. Camera frame-done IRQs should be **pinned away from RT inference cores** to prevent ISR execution interfering with deadline tasks.

> **Common Pitfall:** `irqbalance` is a daemon that automatically redistributes IRQs across CPUs to balance load. It will override any manual `smp_affinity` settings periodically. Always stop `irqbalance` before manually pinning IRQs on a production inference system. Use `irqbalance --banirq=N` to exclude specific IRQs from balancing if you need partial manual control.

Now that we understand how IRQs are delivered and handled, let's look at a specialized optimization for high-bandwidth devices that would drown the CPU in individual interrupts.

---

## Interrupt Coalescing: NAPI

For network and high-bandwidth devices, firing one interrupt per packet is unsustainable at high rates. **NAPI** (New API) uses interrupt coalescing:

1. **First packet arrives** → one interrupt fires.
2. **ISR disables further interrupts** for this queue, schedules a `poll()` callback via softirq.
3. **`poll()` runs in softirq context**, drains the queue in a loop (up to budget packets).
4. **Queue drained** → re-enable interrupts for this queue.

This **amortizes the interrupt overhead** across many packets. The tradeoff is **latency**: the first packet after re-enabling interrupts may wait until the next interrupt.

Tradeoff: coalescing adds latency (up to `ethtool -C eth0 rx-usecs`) but dramatically reduces interrupt overhead at high packet rates. Relevant for multi-camera Ethernet (GigE Vision) and RDMA NIC configurations in AI server racks.

---

## Exception Handling: Page Faults

**Page fault** is the most frequent exception in normal operation. Handler: `do_page_fault()` → `handle_mm_fault()`.

Fault reasons (ARM64 `ESR_EL1` or x86 CR2 + error code):
- **Anonymous mapping not yet allocated**: allocate physical page, update PTE, return — the program never sees this.
- **File-backed mapping not in page cache**: read from storage into cache, map PTE — this is where model file loading latency comes from.
- **CoW write fault**: allocate private copy, update PTE, return — this is the `fork()` copy-on-write mechanism in action.
- **Permission fault on unmapped address**: send SIGSEGV to process — a real bug in the application.

The page fault handling sequence:

1. **MMU raises fault**: the hardware detects that a virtual address has no valid PTE (or wrong permissions) and raises a fault exception.
2. **CPU saves state**: the faulting instruction's PC and the fault address are saved (in CR2 on x86, in FAR_EL1 on ARM64).
3. **`do_page_fault()` called**: the kernel's fault handler looks up the VMA (Virtual Memory Area) containing the faulting address.
4. **Fault type determined**: anonymous/file-backed/CoW/invalid based on VMA flags.
5. **Handler runs**: allocates page, reads from disk, or copies — depending on fault type.
6. **PTE installed**: the new physical page's address is written into the page table.
7. **Instruction re-executed**: the faulting instruction is retried — it succeeds this time because the PTE is now valid.

Under `PREEMPT_RT`, page faults in RT threads introduce unbounded latency spikes. `mlockall(MCL_CURRENT | MCL_FUTURE)` prevents this by locking all pages in RAM before the RT phase begins.

```c
mlockall(MCL_CURRENT | MCL_FUTURE);
// Pin ALL currently-mapped and future-mapped pages in RAM.
// MCL_CURRENT: lock all pages mapped right now (stack, heap, libraries).
// MCL_FUTURE:  lock all pages mapped after this call (new mmap, malloc, stack growth).
// Without this, a page fault during inference adds 1–10 ms of latency
// as the kernel reads from NVMe into page cache before resuming the RT thread.
// Requires CAP_IPC_LOCK capability.
```

This code prevents any page that is part of the inference process from being evicted to swap or unmapped while the RT thread is running.

> **Common Pitfall:** Calling `mlockall()` without pre-faulting pages can still result in the first access to each page causing a fault, even though the page will stay locked afterward. The correct pattern is: `mlockall(MCL_CURRENT | MCL_FUTURE)`, then touch every page of your working set (write a byte to each page), then enter the RT loop. The `MCL_FUTURE` flag ensures stack growth and new `mmap` calls also lock their pages, but the pages are still faulted in on first access — just locked immediately after.

---

## Summary

| Mechanism | Context | Can sleep? | Latency target | Example use |
|---|---|---|---|---|
| Top half (ISR/hardirq) | Interrupts disabled | No | < 1 µs | Acknowledge HW, start DMA |
| Softirq | Post-hardirq interrupt context | No | < 10 µs | Network RX/TX, block completion |
| Tasklet | Via softirq (serialized) | No | < 10 µs | Legacy; deprecated |
| Workqueue | Process context (kworker) | Yes | 10s–100s µs | General deferred work, memory allocation |
| Threaded IRQ | Process context (irq/N thread) | Yes | Bounded by RT scheduler | Modern drivers, PREEMPT_RT compatible |

### Conceptual Review

- **Why must the top half (ISR) be as short as possible?** While a hardirq ISR runs on a CPU, that CPU cannot receive any other interrupts — they are masked. A long ISR creates a window during which all other devices on that CPU are unserviced. This causes dropped frames in camera pipelines and missed deadlines in RT tasks.
- **What is the difference between a softirq and a workqueue?** A softirq runs in interrupt context immediately after the hardirq completes — it cannot sleep, allocate memory with `GFP_KERNEL`, or take mutexes. A workqueue runs in a kernel thread (process context) — it can do all of those things. Choose workqueues when deferred work needs to sleep or take locks.
- **Why are threaded IRQs preferred for new drivers?** They are compatible with `PREEMPT_RT` (where spinlocks become sleeping locks, making traditional hardirq ISRs unsafe), their priority is tunable via `chrt`, and they allow the actual driver work to use the full kernel API including sleeping functions.
- **What is a page fault and when is it expensive?** A page fault occurs when a virtual address has no valid physical mapping. Minor faults (page not yet allocated) complete in microseconds. Major faults (data must be read from disk) take milliseconds. For RT inference threads, even minor faults are unacceptable — `mlockall()` prevents both.
- **What does MSI-X provide that legacy IRQs cannot?** Per-vector CPU affinity: each MSI-X vector can be independently pinned to a specific CPU. This means a 32-queue NVMe drive can have each queue's completion interrupt wake the exact CPU thread that submitted the I/O, eliminating cross-core lock contention.
- **Why does NAPI coalesce interrupts instead of using one interrupt per packet?** At 100 Gbit/s, a NIC could generate ~148 million interrupts per second for 64-byte packets. Each interrupt forces a mode switch and cache effects. NAPI batches packet processing into a single `poll()` call per interrupt, reducing overhead to a manageable level at the cost of slightly increased latency for the first packet.

---

## AI Hardware Connection

- GPU MSI-X per-engine completion vectors enable async CUDA stream operation; pinning each vector to the CUDA stream management core eliminates 10–50 µs cross-core wakeup overhead in inference pipelines.
- Camera frame-done ISR → threaded IRQ path (or workqueue) sets the end-to-end latency floor for `camerad`; measuring it with `bpftrace tracepoint:irq:irq_handler_entry` directly quantifies camera-to-model input delay in openpilot.
- NVMe per-queue MSI-X is prerequisite for GPUDirect Storage: each NVMe completion must wake the DMA engine on the correct NUMA node without cross-node memory traffic — requires both MSI-X and CPU affinity to be configured.
- Threaded IRQs are mandatory under PREEMPT_RT (Linux 6.12 mainline): camera and GPU completion handlers become schedulable threads, bounding their latency via the RT scheduler rather than interrupt-disable windows.
- `/proc/irq/N/smp_affinity` is the primary control point for IRQ isolation on AV platforms — GPU, NVMe, and camera IRQs are moved off RT cores to prevent top-half execution interfering with `modeld` and `controlsd` deadlines.
- FPGA AXI-stream DMA completion interrupts on Zynq/MPSoC follow the same top-half/workqueue pattern: the ISR acknowledges the CDMA, queues a work item that processes the result buffer and signals the inference thread via a wait queue.
