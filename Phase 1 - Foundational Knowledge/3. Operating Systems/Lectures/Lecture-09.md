# Lecture 9: Synchronization — Ownership Before Primitive Selection

## Start with the shared invariant

Suppose a sensor sample contains `(sequence, timestamp, value)`. A consumer must see all three from the same update. Making only `sequence` atomic does not automatically protect ordinary reads and writes to the other fields. Concurrent conflicting non-atomic C/C++ accesses without the required synchronization can be a **data race**, with undefined behavior; a wider race condition can also occur in entirely atomic code when a multi-step protocol is wrong.

Write down who owns the object, who can change it, and when it may be destroyed. Then choose a primitive. Mutual exclusion does not itself guarantee fairness, bounded waiting, priority inheritance, or freedom from deadlock.

## User space and kernel space have different contracts

A `pthread_mutex_t`, Linux `struct mutex`, `spinlock_t`, and a C++ atomic flag are different APIs. Do not copy a kernel algorithm into portable C++ and assume the memory model is identical.

| Need | Starting point | Question to answer |
|---|---|---|
| Protect an ordinary user-space object | Mutex plus scoped ownership | Does every access use the same lock? |
| Wait for a condition | Condition variable plus predicate under a mutex | Can notification occur before the waiter sleeps? |
| Publish fixed-size samples SPSC | Reviewed atomic ownership protocol | Can a producer overwrite a slot before consumption? |
| Protect kernel data in atomic context | Appropriate raw/spin lock variant | Can this context sleep, and can an interrupt take the same lock? |
| Notify kernel request completion | Completion with request lifetime | What happens after timeout or cancellation? |
| Read a kernel snapshot with retries | Appropriate seqcount/seqlock API | Can the writer be preempted by a reader that spins? |

## Mutex: protect the whole operation

In user space, keep the predicate and its updates under the same mutex:

```cpp
// Conceptual C++ pattern; state and mutex must outlive both threads.
{
    std::lock_guard<std::mutex> lock(mutex);
    ready = true;
}
condition.notify_one();

// The waiter checks before sleeping and rechecks after each wakeup.
std::unique_lock<std::mutex> lock(mutex);
condition.wait(lock, [&] { return ready; });
```

The predicate handles early notification and spurious wakeups. A notification is not a durable replacement for the predicate. If `ready` is used to publish a sample, access that sample under the same ownership rule too. Use a bounded queue or sequence number when multiple events must be preserved; one Boolean cannot count ten arrivals.

Kernel mutexes can use optimistic spinning before sleeping, so “mutex never spins” is inaccurate. Interruptible acquisition returns an error that must be handled before entering the protected section. Hard interrupt context cannot wait on a sleeping mutex. A threaded IRQ handler is task context, but its particular API and surrounding locks still impose constraints. See [kernel mutex design](https://docs.kernel.org/locking/mutex-design.html).

## Spinlocks and PREEMPT_RT

On a non-RT kernel, acquiring `spinlock_t` disables preemption; IRQ-disabling variants also manage local interrupt state. If an interrupt on the same CPU can take the same lock, allowing it to interrupt a lock holder can deadlock. IRQ-save forms restore the prior local interrupt state instead of blindly enabling interrupts afterward.

PREEMPT_RT changes `spinlock_t` and `rwlock_t` semantics to RT-mutex-based implementations; they must not be treated as raw hard-IRQ locks. `raw_spinlock_t` retains raw spinning semantics. A raw locked region cannot call a sleeping operation. No universal “under 1 µs means safe” threshold replaces context analysis and a bounded critical section. Do not depend on a particular warning, crash, or silent failure when violating the rule.

Use the kernel's [lock type rules](https://docs.kernel.org/locking/locktypes.html) for the configured kernel. Reducing a lock's hold time helps, but first verify correctness of every access and interrupt path.

## Reader/writer locks

A read lock allows concurrent readers, while a write lock excludes readers and other writers. This can help large read-dominated critical sections, but reader accounting and cache-line contention can cost more than a simple mutex for tiny reads. Fairness and priority-inheritance behavior depend on the primitive and configuration; do not assume every reader/writer lock provides writer preference or bounded writer delay.

**Worked lifetime problem:** an inference thread holds a read lock throughout a 20 ms forward pass. An updater waiting for the write lock may wait for that pass to finish, even if its eventual pointer swap takes only 1 µs. Releasing the read lock immediately after reading a pointer would require a separate mechanism that keeps the model alive throughout inference. A reference-counted immutable model is one possible design; synchronization of publication and destruction remains necessary.

## Sequence counters: retry does not erase invalid memory access

In a kernel seqcount pattern, the writer marks an update in progress, changes the fields, then marks completion. A reader copies the fields and accepts the copy only if the sequence observations show no intervening write. Writers must be serialized by the appropriate lock or ownership rule. `seqlock_t` includes a writer lock: **writers can wait for other writers**. Lockless readers normally do not block writers, but readers can retry repeatedly.

Consider a writer preempted halfway through an update. A higher-priority reader spinning on the odd sequence on that same CPU can prevent the writer from finishing. Kernel seqcount variants address serialization and preemption requirements; choose the variant for the actual writer lock and RT configuration. Read [sequence counters and sequential locks](https://docs.kernel.org/locking/seqlock.html).

If the snapshot contains a pointer that the writer can free, a reader can dereference freed memory before its final retry check. Retrying afterward cannot repair that invalid access. Sequence protection therefore does not replace lifetime protection. Similarly, a naive portable C++ seqlock that reads ordinary payload fields concurrently with writes still has a data race; use a properly justified implementation or a mutex.

## Completions and timeout ownership

A Linux completion records an event so a later waiter can observe it. A successful `complete()` does not inherently identify which hardware request completed. Waiters need an associated request object and a defined producer lifetime.

**Worked timeout:** request 17 is submitted; the wait times out; the driver frees its completion object; a late IRQ calls `complete()` through the old pointer. The timeout did not cancel the IRQ or DMA, so this can become use-after-free. Before freeing memory, stop/quiesce the producer using the device's supported sequence and ensure callbacks cannot access the object. Alternatively retain ownership until an explicit terminal state. Do not reuse a completion for request 18 while request 17 can still signal it.

`complete_all()` and `reinit_completion()` require coordinated reuse; resetting while waiters or producers still use the object can lose an event. The [completion documentation](https://docs.kernel.org/scheduler/completion.html) explains waiting contexts and timeout lifetime constraints. Kernel completion examples are distinct from C++ condition-variable semantics.

## Lock dependency analysis

Document a lock order, such as `device configuration → submission queue → request state`. If one path takes queue then request while another takes request then queue, draw the cycle and remove it. Trylock only helps if failure releases the required resources and has a progress policy; repeatedly retrying can livelock.

Lockdep tracks observed relationships between lock classes and can diagnose potential dependency cycles without waiting for an actual deadlock. It is a development tool, not a proof that every path has been exercised or that timing requirements hold. Class assignment and nested locking need correct annotations; classes are not simply one numerical rank per object. Overhead depends on configuration and workload. See [lockdep design](https://docs.kernel.org/locking/lockdep-design.html).

## Run and explain

Run the checked C [thread/API demo](demos/rt-demo/README.md), then the C++ [SPSC queue lab](../labs/SPSC-Queue.md). The first checks mutex/condition/rwlock lifecycle; the second exposes the producer-to-consumer and consumer-to-producer ownership transfers. Its deliberate overwrite mutation must fail.

**Self-check answers:**

1. Is a mutex unnecessary on a one-core machine? No; preemption can interleave accesses, and the language synchronization contract still applies.
2. Does a notification prove a predicate is true? No; inspect the predicate under its lock.
3. Does a completion timeout prove DMA stopped? No; timeout is an observation by the waiter.
4. Does a seqlock guarantee writer execution immediately? No; writers serialize, scheduling still applies, and reader/writer contexts must obey the API.
5. Does passing a stress test prove absence of all races? No; combine API/lifetime reasoning with appropriate tests and diagnostics.
