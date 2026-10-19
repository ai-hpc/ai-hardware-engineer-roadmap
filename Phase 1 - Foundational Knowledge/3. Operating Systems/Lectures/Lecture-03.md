# Lecture 3: Interrupts, Exceptions, and Deferred Work

A camera finishes writing a frame while your inference thread is asleep. How does that thread learn the frame is ready? Follow **device completion → interrupt handling → runnable thread → scheduled thread**. These are different events. A fast interrupt handler does not guarantee a fast application response.

After this lesson you should be able to choose a valid execution context, distinguish a wakeup from execution, and explain where to measure a late frame. Read [Lecture 2](Lecture-02.md) first; continue to [scheduling](Lecture-06.md) afterward.

## Interrupts and exceptions

A hardware interrupt normally arrives asynchronously to the currently executing program. A synchronous exception is caused by an instruction: an invalid access, a system-call instruction, or a breakpoint. A page fault can be resolved and the instruction retried, or lead to a signal if the access is invalid. Machine-check recovery is architecture- and error-dependent; “abort means every machine check is synchronous and fatal” is not a useful portable rule.

On AArch64, hardware records exception state in registers such as ELR and SPSR; software entry code saves the required general registers on a kernel stack. Do not transfer the x86 hardware-stack description unchanged to ARM. The entry path also depends on whether execution came from userspace, the kernel, or a guest.

| Layer | Identifier or mechanism | What to inspect |
|---|---|---|
| Device | Completion bit, queue descriptor, interrupt mask | Device documentation and driver |
| ARM interrupt controller | GIC hardware interrupt ID | Device Tree interrupt specifier and IRQ domain |
| Linux IRQ core | Linux IRQ number | `/proc/interrupts`, driver registration |
| Scheduler | Thread ID and allowed CPUs | `/proc/PID/task/TID/status`, scheduler trace |

The Linux IRQ number need not equal the GIC hardware ID. GICv3 has SGIs for interprocessor interrupts, PPIs for per-CPU sources, SPIs for shared sources, and LPIs used with interrupt translation facilities. Implemented priority bits and routing capabilities depend on the controller; an 8-bit field does not promise 256 distinct implemented priorities.

PCIe MSI and MSI-X deliver interrupts as transactions instead of a shared physical pin. MSI-X gives devices independently configurable table entries and commonly supports queue-specific routing. A device may expose fewer vectors than queues, and Linux may share vectors or change affinity. A GPU operation need not cause one CPU interrupt: driver queues, memory polling, and completion batching also exist.

## The ownership and scheduling path

```text
camera DMA writes buffer 3
    ↓ device-defined completion ordering
IRQ handler confirms that this device caused the interrupt
    ↓ acknowledges/masks as the device specification requires
completed-buffer state published under the driver's synchronization protocol
    ↓ wakeup
application becomes runnable
    ↓ scheduler chooses it on an allowed CPU
VIDIOC_DQBUF returns buffer 3 to userspace
```

Acknowledging too early or unmasking before draining a level-triggered source can lose work or cause an interrupt storm. Reading a status register does not universally clear it: some registers are read-to-clear, some are write-one-to-clear, and others require another sequence. A shared handler returns `IRQ_NONE` if the device has no relevant pending event.

For a conventional hard-IRQ handler, keep work bounded and do not sleep. `kmalloc(..., GFP_KERNEL)`, blocking mutex acquisition, and userspace copies are inappropriate there. PREEMPT_RT moves many interrupt handlers into threads, but does not turn every low-level interrupt region into sleepable application context. The [generic IRQ documentation](https://docs.kernel.org/core-api/genericirq.html) describes registration and handler contracts.

## Choosing where deferred work runs

| Mechanism | May ordinary callback code sleep? | Useful distinction |
|---|---|---|
| Hard IRQ | No | Urgent device acknowledgment and bounded state update |
| Softirq / tasklet | No | Atomic deferred work; do not start new designs assuming tasklets are the preferred interface |
| Ordinary threaded workqueue | Yes | Shared worker pools execute work in process context |
| `WQ_BH` workqueue | No | Uses softirq context, despite the workqueue name |
| Threaded IRQ | Usually process context | Device interrupt work tied to the IRQ lifecycle |
| Userspace worker | Yes | Application processing after ownership transfer |

`WQ_HIGHPRI` selects higher-priority worker pools; it is not a real-time deadline guarantee. A private workqueue object does not necessarily allocate private worker threads. If strict FIFO execution of one work item at a time is required, use `alloc_ordered_workqueue()` rather than assuming an unbound queue with `max_active=1` provides the same ordering. Always check allocation failure and arrange cancellation/draining before freeing callback state. See [workqueue semantics](https://docs.kernel.org/core-api/workqueue.html).

A threaded-IRQ registration is a **driver-reading fragment**, not a complete module:

```c
/* priv and all handler-visible state must already be initialized. */
ret = devm_request_threaded_irq(dev, irq, my_top_half, my_thread,
                                IRQF_ONESHOT, "my-sensor", priv);
if (ret)
    return ret;
```

The top half may return `IRQ_WAKE_THREAD` for relevant work. `IRQF_ONESHOT` keeps the interrupt masked while its thread runs, subject to IRQ-core and device behavior. Shared IRQs need an appropriate shared-source test. On teardown, stop new device work, synchronize handlers, cancel deferred work, then release the memory they could access. Managed allocation does not implement that protocol for you.

## Networking: NAPI is a hybrid

NAPI commonly begins with an interrupt, masks further notifications, and polls receive/transmit work within a budget. When the driver completes polling and re-enables notifications, it must honor the API's budget and completion rules. This avoids an interrupt per packet under load. Hardware interrupt coalescing (`ethtool -c` to inspect) is a separate mechanism. Some configurations use busy polling or threaded NAPI; CPU cost and tail latency must be measured together. See the [NAPI documentation](https://docs.kernel.org/networking/napi.html).

## A worked latency budget

Suppose a recorded trace shows:

| Event | Time relative to DMA completion |
|---|---:|
| IRQ begins | 18 µs |
| IRQ wakes application | 25 µs |
| Application is scheduled | 145 µs |
| Dequeue completes | 158 µs |

The IRQ ran for 7 µs before wakeup. The runnable application then waited 120 µs. Optimizing that handler by 3 µs cannot remove the 120 µs scheduling delay. Investigate competing work, affinity, throttling, interrupts, and the actual scheduling policy. These numbers are an arithmetic example, not a measurement from this course.

A related trap: a wakeup means “the condition may now be true,” not “this thread owns a completed buffer.” The consumer must re-check the protected predicate. See the tested [SPSC queue](../labs/SPSC-Queue.md) for a userspace ownership-transfer example; its C++ atomics are not a replacement for kernel IRQ locking rules.

## Read-only inspection and practice

On a Linux host, inspect what the machine actually exposes:

```bash
cat /proc/interrupts
cat /proc/softirqs
cat /proc/self/status
```

Virtual machines and containers may hide these files or expose a limited view. For a specific IRQ, inspect its `smp_affinity_list` and effective affinity only after identifying the device. Managed IRQs and offline CPUs constrain writable affinity. Moving an IRQ does not reserve the destination CPU, and stopping an IRQ-balancing service globally is not a prerequisite for learning this lesson.

Run the existing [real-time demo](demos/rt-demo/README.md) on its unprivileged path. Record the maximum observed wakeup lateness and sample count. Then explain why that finite observation cannot certify a worst-case deadline.

**Check:** Does a threaded handler make a camera buffer safe to free when a timeout expires? **Answer:** No. Timeout is an application decision; hardware and queued callbacks must be quiesced or proven complete before their storage is reclaimed.

**Check:** Can a `WQ_BH` callback block on a mutex? **Answer:** No. It executes in softirq context. Select a sleepable mechanism or redesign the synchronization.
