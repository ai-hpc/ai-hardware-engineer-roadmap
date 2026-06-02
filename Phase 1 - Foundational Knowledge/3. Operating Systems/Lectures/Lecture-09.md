# Lecture 9: Synchronization: Spinlocks, Mutexes, RW Locks & Seqlocks

## Overview

The core problem this lecture addresses is: how do multiple CPUs or threads safely access the same data without corrupting it? When two threads write to the same memory location simultaneously, the result is undefined — this is a race condition. Synchronization primitives are the contracts that prevent races by ensuring only one writer (or multiple readers) can access shared state at a time. The mental model to carry here is that of a shared whiteboard in an office: a spinlock is someone standing in front of it spinning in place waiting for their turn; a mutex is someone going to sit down and do other work until notified; a seqlock is a reader who copies the board and then checks whether anyone erased it while they were copying. For an AI hardware engineer, choosing the wrong synchronization primitive can be the difference between a camera driver that works at 100Hz and one that introduces 50µs of latency on every frame handoff.

---

## Mutual Exclusion Problem

A **race condition** occurs when multiple CPUs or threads access shared state concurrently without coordination, and the outcome depends on scheduling order.

Three requirements for correct mutual exclusion:

1. **Mutual exclusion**: at most one thread in the critical section at a time
2. **Progress**: if no thread is in the critical section, a waiting thread must eventually enter
3. **Bounded waiting**: no thread waits indefinitely (no starvation)

> **Key Insight:** Choosing the right synchronization primitive is not just a correctness decision — it is a performance decision. The wrong choice can add latency, reduce throughput, or introduce priority inversion. Each primitive in this lecture has a specific use case; using a mutex where a seqlock is appropriate wastes CPU cycles on writers that never contend.

---

## Spinlock (`spinlock_t`)

**Busy-waits (spins)** on an atomic test-and-set instruction until the lock is available. **Disables preemption** on the local CPU while held.

```c
spin_lock(&lock);                        // acquire; disables preemption on local CPU
/* critical section */
spin_unlock(&lock);                      // release; re-enables preemption

// When data is shared with an interrupt service routine:
spin_lock_irqsave(&lock, flags);         // disables preemption + local IRQs; saves IRQ state
/* critical section (safe against ISR) */
spin_unlock_irqrestore(&lock, flags);    // restores IRQ state + re-enables preemption
// flags saves the interrupt enable/disable state before the lock — not a priority number
// This is necessary because the caller may have already disabled IRQs before acquiring the lock
```

Key constraints:
- Only for **very short critical sections** (<1µs); holding while sleeping is a kernel bug
- On NUMA systems, contended spinlocks cause **cache-line bouncing** across sockets
- Under `PREEMPT_RT`: `spinlock_t` becomes a sleeping `rtmutex`; `raw_spinlock_t` retains true spin behavior

```
  Spinlock Acquisition Flow:

  Thread A              Thread B (attempting lock)
  ────────────          ────────────────────────────
  spin_lock(&L)         spin_lock(&L)
  [lock acquired]       [lock busy → spin in tight loop]
  ... critical ...      test-and-set ... test-and-set ...
  spin_unlock(&L)  ──►  [lock acquired]
                        ... critical ...
                        spin_unlock(&L)

  Note: Thread B burns CPU cycles while waiting.
  This is intentional for very short waits where a context switch
  would cost more than the spin itself.
```

> **Common Pitfall:** If the critical section under a spinlock calls any function that can sleep — `kmalloc(GFP_KERNEL)`, `copy_from_user()`, `msleep()` — the kernel will BUG_ON on a PREEMPT_RT kernel and silently corrupt state on a standard kernel. Audit every line inside a spinlock-held section for potential sleep points.

---

## Mutex (`struct mutex`)

Task **blocks** (`TASK_UNINTERRUPTIBLE`) if lock is unavailable; scheduler runs other tasks. **No CPU wasted spinning**.

```c
mutex_lock(&mutex);                    // blocks until acquired; process context only
/* critical section */
mutex_unlock(&mutex);

mutex_trylock(&mutex);                 // non-blocking; returns 1 if acquired, 0 if not
// Use trylock when you can do useful work while waiting (e.g., process other requests)

mutex_lock_interruptible(&mutex);      // returns -EINTR if signal received while waiting
// Preferred in syscall handlers so that kill/ctrl-C can interrupt a blocked task
```

- Cannot use in interrupt context — interrupts cannot sleep
- Userspace equivalent: `pthread_mutex_t` backed by `futex` (fast path avoids syscall when uncontended)
- Kernel `mutex` is not the same as `pthread_mutex_t`; different implementation and rules

The transition from spinlock to mutex mirrors the **CPU vs. scheduler trade-off**: spinlocks waste CPU cycles but avoid context switch overhead. For critical sections **longer than a few microseconds**, putting the task to sleep is almost always cheaper.

### rtmutex — Priority Inheritance Mutex

`rtmutex` is a mutex with **priority inheritance (PI)**:
- When high-priority task `H` blocks on an `rtmutex` held by low-priority task `L`, the kernel temporarily boosts `L`'s priority to `H`'s level
- `L` runs sooner, releases the lock sooner, `H` unblocks; `L` returns to original priority on release
- Under `PREEMPT_RT`, most kernel `spinlock_t` instances are replaced with `rtmutex`
- Prevents the Mars Pathfinder scenario (see Lecture 11) in kernel code

> **Key Insight:** Priority inheritance is what separates `rtmutex` from a regular mutex. Without PI, a high-priority task waiting for a lock held by a low-priority task can be stalled indefinitely by medium-priority tasks that preempt the lock holder. With PI, the lock holder's priority is raised to match the waiter's, eliminating that indefinite stall.

---

## Read/Write Semaphore (`struct rw_semaphore`)

Allows **multiple concurrent readers** **or** **one exclusive writer**; not both simultaneously. This is ideal when **reads are far more frequent than writes** — a common pattern for configuration tables and model weight structures.

```c
down_read(&rw_semaphore);          // shared read lock; multiple readers can hold concurrently
/* read shared data */
up_read(&rw_semaphore);

down_write(&rw_semaphore);         // exclusive write lock; blocks all readers + other writers
/* modify shared data */
up_write(&rw_semaphore);

// Downgrades write lock to read lock without releasing (atomic, no window of unlocked state)
downgrade_write(&rw_semaphore);
```

- Sleeping lock; process context only; cannot use in interrupt context
- Writer-bias: once a writer is waiting, new readers are blocked to prevent writer starvation
- Used for: `mmap_lock` (in `mm_struct`), filesystem inode locks, device driver config tables
- Spinlock variant `rwlock_t` exists for interrupt context (busy-waits; not sleeping)

```
  RW Semaphore Access Patterns:

  Timeline ──────────────────────────────────────────────────►

  Reader A: ──[read]──────────────────────────────
  Reader B:    ──[read]──────────────────────       (concurrent with A)
  Reader C:         ──[read]──────────────────      (concurrent with A and B)
  Writer D:                        ──[WAIT]──[write]──
                                    ▲
                              blocks new readers after this point
  Reader E:                        ──[WAIT]──────[read]──
                                               ▲
                                         resumes after writer finishes
```

---

## Seqlock (`seqlock_t`)

**Writers never block**. Readers detect concurrent writes via a **sequence counter** and retry if needed. The seqlock trades **occasional reader retries** for **zero writer blocking** — the right choice when writers need guaranteed forward progress.

```c
// Writer — always proceeds immediately, never blocks on readers
write_seqlock(&seqlock);
/* sequence counter incremented to ODD (signals: write in progress) */
/* modify shared data */
write_sequnlock(&seqlock);
/* sequence counter incremented to EVEN (signals: write complete) */

// Reader — retries if a concurrent write was detected
unsigned seq;
do {
    seq = read_seqbegin(&seqlock);   // sample counter; if ODD, a write is in progress (spin/retry)
    /* read shared data into local variables */
} while (read_seqretry(&seqlock, seq));  // if counter changed, our read may be torn — retry
// After the loop, local variables hold a consistent snapshot
```

The seqlock mechanism works because the counter is always odd during a write and even otherwise. A reader that starts when the counter is odd knows to retry immediately. A reader that finishes and finds the counter changed knows its data is inconsistent.

Properties:
- Writer overhead: two atomic counter increments; no blocking
- Reader overhead on write-free path: two reads of the counter (extremely cheap)
- Reader overhead on write path: retry loop (rare if writes are infrequent)
- Not suitable for pointer-containing data (reader may dereference freed pointer before retry detects the change)

Linux kernel uses seqlocks for: `jiffies_64`, `timespec64` for clock reads, kernel `xtime` timekeeping, `vDSO` time variables.

> **Key Insight:** Seqlocks invert the usual lock trade-off. Most locks block writers to protect readers. Seqlocks let writers proceed freely and ask readers to retry if they were unlucky. This is the right choice when the writer is a high-priority real-time task (e.g., a sensor ISR updating a timestamp) and readers are lower-priority consumers.

> **Common Pitfall:** Never use a seqlock to protect data containing pointers. If a reader reads a pointer, then a writer replaces the object and frees the old one, the reader may dereference the freed pointer before `read_seqretry` detects the change. Use RCU (Lecture 10) for pointer-containing data structures.

---

## Completion (`struct completion`)

**One-shot synchronization**: thread A waits for an event signaled by thread B. Cleaner than a semaphore for one-time events because its intent is **explicit** and it has **no spurious-wakeup risk**.

```c
DECLARE_COMPLETION(dma_done);         // static declaration; starts in "not complete" state

// Waiter thread
wait_for_completion(&dma_done);               // blocks in TASK_UNINTERRUPTIBLE until signaled
wait_for_completion_timeout(&dma_done, HZ);   // with timeout; returns 0 on timeout, remaining jiffies on success
wait_for_completion_interruptible(&dma_done); // can be interrupted by a signal (returns -ERESTARTSYS)

// Signaler (e.g., DMA completion ISR)
complete(&dma_done);      // wake one waiter (FIFO order)
complete_all(&dma_done);  // wake all waiters simultaneously

reinit_completion(&dma_done);  // reset for reuse after complete_all
```

Use cases: DMA transfer complete notification, kthread startup synchronization, driver probe sequencing, FPGA firmware load completion.

With completions, the synchronization flow is unambiguous: the ISR signals "DMA done" and the waiting thread resumes. Compare this to a mutex (which guards access) or a semaphore (which counts available resources) — a completion is specifically for "wait until event X happens."

---

## lockdep — Lock Dependency Validator

All the synchronization primitives described above can be **used incorrectly**. `lockdep` is the kernel's **runtime tool to catch those mistakes** at development time, before they manifest as hard-to-reproduce **deadlocks**.

`CONFIG_PROVE_LOCKING` enables lockdep, the kernel's runtime lock dependency validator:

- Assigns each lock a **lock class** based on its static address in the binary
- Records lock acquisition chains: "lock A held when lock B was acquired"
- Reports **AB-BA cycles** (potential deadlocks) at first occurrence — before they manifest as actual hangs
- Reports **lock-from-interrupt violations**: sleeping lock acquired in interrupt context
- Output: `WARNING: possible circular locking dependency` with full stack traces in `dmesg`

```bash
# Enable in kernel config
CONFIG_PROVE_LOCKING=y
CONFIG_LOCK_STAT=y    # adds lock contention statistics per lock class

# View lock stats after a workload run
cat /proc/lock_stat | head -40
# Output shows: lock name, acquisition count, contention count, wait time histogram
```

**Always enable during driver development** and regression testing. `lockdep` adds ~10% overhead; **disable in production**.

> **Key Insight:** `lockdep` detects potential deadlocks the *first time* a new lock ordering is observed — even if that specific execution order has never actually deadlocked. This is more powerful than waiting for an actual hang, which may only occur under rare timing conditions in production. Think of lockdep as a static analyzer that runs at runtime.

---

## Summary

| Primitive | Context Usable | Blocks? | PI Support | Best For |
|---|---|---|---|---|
| `spinlock_t` | Process + IRQ | No (busy-wait) | No | Short CS (<1µs), ISR-shared data |
| `raw_spinlock_t` | Process + IRQ | No (busy-wait) | No | Hardware-critical, cannot be sleeping lock |
| `mutex` | Process only | Yes | No | Longer CS in process context |
| `rtmutex` | Process only | Yes | Yes | RT tasks, PREEMPT_RT kernel |
| `rwlock_t` | Process + IRQ | No (spin) | No | Read-heavy, short CS, IRQ context |
| `rw_semaphore` | Process only | Yes | No | Read-heavy, longer CS |
| `seqlock_t` | Process + IRQ (writer); retry (reader) | Writer: No; Reader: retry | No | Rarely-written, frequently-read data |
| `struct completion` | Process only | Yes | No | One-shot event signaling |

### Conceptual Review

- **When should you use a spinlock instead of a mutex?** When the critical section is very short (<1µs), when you are in interrupt context (cannot sleep), or when the cost of a context switch exceeds the spin wait time. For anything longer or in process context, a mutex is almost always better.
- **What makes `rtmutex` different from `mutex`?** Priority inheritance. When a high-priority task blocks on an rtmutex held by a low-priority task, the low-priority task's priority is temporarily raised to avoid priority inversion. Regular `mutex` has no PI mechanism.
- **Why can't you use a mutex in an interrupt handler?** Interrupt handlers cannot sleep. A mutex blocks the caller when the lock is unavailable, which means putting the current execution context to sleep. An interrupt handler has no "current task" to sleep — it runs on top of whatever was interrupted.
- **What is the seqlock's fundamental assumption?** That writes are rare and reads are frequent. If writes are frequent, readers spin-retry constantly and the seqlock becomes worse than a mutex. It also assumes the protected data contains no pointers that could be dereferenced during a torn read.
- **What class of bugs does lockdep catch?** AB-BA deadlock cycles (lock A then B on CPU 0, lock B then A on CPU 1) and sleeping-lock-in-interrupt-context violations. It catches these the first time a new ordering is observed, before an actual deadlock occurs.
- **Why use `spin_lock_irqsave` instead of `spin_lock` when sharing data with an ISR?** An ISR can preempt the thread holding a spinlock. If the ISR then tries to acquire the same spinlock, it spins forever — deadlock. `spin_lock_irqsave` disables local interrupts first, preventing the ISR from running while the lock is held.

---

## AI Hardware Connection

- `spin_lock_irqsave` in a camera DMA completion ISR protects the current frame buffer index shared with the inference thread; must not sleep in interrupt context; critical section is a single index write
- `rw_semaphore` for model weight hot-reload: many concurrent inference threads hold the read lock during forward passes; the weight updater holds the write lock only during the pointer swap, so inference is never stalled longer than the swap itself
- Seqlock passes hardware timestamps from the sensor acquisition thread to the fusion pipeline without blocking the sensor thread on slow readers; reader retries are negligible since sensor writes occur at 100Hz against reads at 1kHz
- `completion` for "DMA transfer done" synchronization in FPGA accelerator drivers: the host CPU submits a command buffer, blocks on `wait_for_completion`, and the FPGA interrupt handler calls `complete()` when processing finishes
- `rtmutex` with priority inheritance is essential on PREEMPT_RT AV compute platforms; all `spinlock_t` locks become `rtmutex` instances system-wide, providing PI in every kernel subsystem without modifying driver code
- `lockdep` during Jetson embedded driver development catches AB-BA lock ordering violations in camera, ISP, and DMA subsystems before the hardware is deployed in a vehicle or robot
