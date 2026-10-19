# Lecture Note 02: Interrupts, System Calls, and Measurement

Consolidates [Lecture 3](Lecture-03.md) and [Lecture 4](Lecture-04.md).

## Four timestamps, four different events

```text
device completion → IRQ handler → task wakeup → task actually scheduled
```

An IRQ number is a Linux identifier, not necessarily a GIC hardware ID or x86 vector. MSI-X can provide multiple independently routable entries, but queues and vectors are not guaranteed one-to-one. A GPU completion need not trigger one interrupt per operation.

Hard-IRQ code cannot sleep. Ordinary threaded workqueues can sleep; `WQ_BH` workqueues run in softirq context and cannot. High-priority worker pools do not provide real-time deadlines. Select `alloc_ordered_workqueue()` when its documented one-at-a-time ordering is required; a private queue need not mean private threads.

**Question:** An IRQ wakes a thread at 25 µs; the thread runs at 145 µs. What interval is directly visible?

**Answer:** A 120 µs runnable-to-running delay. A fast handler does not remove it. Investigate scheduling, affinity, throttling, interrupts, and other evidence before attributing a cause.

## System-call duration is not entry cost

A syscall can block for a device or lock. `strace -T` reports an observed interval while adding its own tracing effects. Two 200 fps cameras with four calls/frame generate 1,600 calls/s; one camera would generate 800. A hypothetical 300 ns transition cost gives 0.48 ms/s for the two-camera case, not the full camera latency.

Libc can use vDSO helpers for supported time queries. `CLOCK_MONOTONIC` avoids wall-clock steps but can be rate-adjusted. RAW uses a different local clock contract. None automatically synchronizes sensor or remote-host clocks.

Timer slack is inherited and can be configured for applicable waits. It is not applied under real-time scheduling policies. Reducing it does not eliminate scheduler, interrupt, or device delay.

## Shared memory and readiness

A new `shm_open()` object has length zero. The owner must size it before mapping/access, initialize its protocol before publication, and define unlink/lifetime behavior. `MAP_SHARED` shares bytes but does not make a data race legal. A memfd still needs sizing and synchronization; pass descriptors with `SCM_RIGHTS` across processes.

With edge-triggered epoll, use nonblocking operations and drain until `EAGAIN` before waiting for another edge, while preventing one busy source from starving others. Level-triggered operation is also valid. Readiness is not ownership of a full message.

**Question:** Why can an mmap succeed but touching its shared-memory pages later cause SIGBUS?

**Answer:** The requested virtual range can extend beyond the backing object's current length. Check and establish the file size before use; also define who can change that size.

## Evidence from a real build

The [BPF build lab](../labs/BPF-Build-Lab.md) compiles an ioctl tracer with a shared event layout, checked entry/exit pairing, and loss counters. A compiler pass is not kernel verifier acceptance. The recorded run does not attach hooks or measure latency.

BPF verification checks particular execution/memory constraints; it does not prove an inference deadline or make every program read-only. Seccomp filters syscall metadata and reduces accessible operations; it is not a complete sandbox.

**Question:** Zero ring-reservation failures means every ioctl was measured—true or false?

**Answer:** False. Missing entry pairs, attachment windows, filtering, failed map insertion, and shutdown can all omit calls independently of ring pressure.
