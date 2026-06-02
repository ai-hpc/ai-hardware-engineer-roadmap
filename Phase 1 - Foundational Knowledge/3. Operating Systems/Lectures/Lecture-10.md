# Lecture 10: Lock-Free Programming: RCU, Atomics & Memory Ordering

## Overview

The core problem this lecture addresses is: can we share data between threads without using locks at all? Locks are correct but expensive — they cause cache-line bouncing, context switches, and priority inversion. For the hottest data paths in a system (a 30Hz camera frame pipeline, a 1kHz sensor fusion loop, a live model configuration update), lock overhead is measurable and unacceptable. The mental model to carry here is that of a library with a special rule: readers can always pick up a book without asking anyone, but when the librarian needs to replace a book with a new edition, they do it by leaving the old book in place until every reader who was already reading it has put it down. For an AI hardware engineer, lock-free techniques are the foundation of zero-copy camera pipelines, live model hot-reload, and sensor data aggregation at kilohertz rates.

---

## Motivation for Lock-Free

Locks introduce unavoidable costs even when not contended:

- **Cache-line bouncing**: the lock variable ping-pongs between CPU L1 caches during contention; each acquire costs ~100–300 cycles on NUMA
- **Context switches**: blocked threads incur scheduler overhead (~1–10µs per switch)
- **Priority inversion**: low-priority lock holder delays high-priority waiter (see Lecture 11)
- **Convoying**: multiple threads queue behind one slow holder, serializing a hot path

Lock-free algorithms use atomic hardware primitives to achieve safe concurrent access without mutual exclusion. They provide better scalability and eliminate priority inversion on performance-critical paths.

> **Key Insight:** "Lock-free" does not mean "without coordination." It means coordination is done through atomic hardware instructions rather than mutual exclusion. These atomics still have costs — cache-line ownership, memory barriers — but those costs are bounded and predictable in a way that lock contention is not.

---

## C++11/C11 Memory Model

CPUs and compilers **reorder instructions** for performance. Without explicit ordering rules, a write in thread A **may not be visible** to thread B for an arbitrary time. The **C11/C++11 memory model** defines ordering guarantees via `std::atomic<T>` / `_Atomic T`:

| Memory Order | Guarantee |
|---|---|
| `memory_order_relaxed` | Atomicity only; no ordering relative to other operations; use for counters |
| `memory_order_acquire` | No load/store after this point may be reordered before it; pairs with release |
| `memory_order_release` | No load/store before this point may be reordered after it; pairs with acquire |
| `memory_order_acq_rel` | Both acquire + release; for atomic read-modify-write operations |
| `memory_order_seq_cst` | Total global order across all threads; full fence; default for `std::atomic<>` |

**Acquire-release pairing**: a `release` store to variable X "happens-before" a subsequent `acquire` load from X in another thread. All writes before the release are visible after the acquire.

```
  Acquire-Release Synchronization:

  Thread A (producer)           Thread B (consumer)
  ─────────────────────         ─────────────────────
  data = 42;                    // reading data before load is wrong
  flag.store(1,                 int f = flag.load(
    memory_order_release);        memory_order_acquire);
                                if (f == 1) {
  ─────────────────────────────► assert(data == 42); // guaranteed
  All writes before release      All loads after acquire
  are visible after acquire      see those writes
```

> **Common Pitfall:** Using `memory_order_relaxed` for a flag that signals "data is ready" is a common error. Relaxed order only guarantees atomicity of the flag itself — it makes no promise that the data written before the flag store is visible to the thread that reads the flag. Always use `release`/`acquire` for producer-consumer signaling.

---

## Hardware Memory Models

Different CPU architectures have **different default ordering guarantees**. Understanding this matters when reading Linux kernel code that uses **bare memory barriers** instead of C11 atomics.

| Architecture | Memory Model | Barrier Requirement |
|---|---|---|
| x86-64 | TSO (Total Store Order) — relatively strong | Few explicit barriers needed; LOCK prefix for atomics |
| ARM64 | Weakly ordered | Explicit `DMB`/`DSB` barriers required; `LDADD`/`CAS` for atomics |
| RISC-V | Weakly ordered (RVWMO) | `FENCE` instruction; `LR`/`SC` for LL/SC atomics |

Linux kernel barrier macros:
- `smp_mb()`: full memory barrier (load + store in both directions)
- `smp_rmb()`: read (load) barrier only
- `smp_wmb()`: write (store) barrier only

On ARM64 (Jetson Orin, Cortex-A78AE), every acquire/release operation translates to explicit `LDAR`/`STLR` instructions. Code that runs correctly on x86-64 TSO without barriers **may silently fail on ARM64** without them. **Always use C11 atomics or kernel barrier macros** rather than plain loads/stores for shared variables.

---

## Atomic Operations

Map to **single indivisible hardware instructions**: `LOCK ADD`/`XADD`/`CMPXCHG` on x86, `LDADD`/`CAS` on ARMv8.1 LSE.

```c
atomic_t counter = ATOMIC_INIT(0);      // initialize to zero
atomic_inc(&counter);                   // LOCK XADD on x86; LDADD on ARMv8.1; increment atomically
atomic_dec_and_test(&counter);          // decrement; return true if result is zero (useful for refcounts)
int old = atomic_cmpxchg(&counter, 5, 10);  // CAS: if counter==5, set to 10; return old value either way
atomic_add_return(n, &counter);         // add n, return new value; useful for rate limiting
```

- `atomic_t` is 32-bit; `atomic64_t` for 64-bit values
- `ATOMIC_INIT(n)` for static initialization
- Use for: reference counts, flags, per-CPU statistics accumulators

---

## Compare-and-Swap (CAS)

**Foundation of most lock-free data structures.** CAS atomically tests a value and **replaces it only if it matches** — providing a "conditional write" without a lock.

```cpp
std::atomic<int> val{0};
int expected = 0;
// Atomically: if val == expected, set val = 1; return true (success)
// If val != expected: load actual value into expected; return false (failure, retry)
bool ok = val.compare_exchange_strong(expected, 1,
              std::memory_order_acq_rel,  // success: both acquire (for reads after) and release (for writes before)
              std::memory_order_acquire); // failure: acquire only (we read the current value)
// On failure, 'expected' now holds the actual value — retry loop uses this updated value
```

`compare_exchange_weak`: may spuriously fail on LL/SC architectures (ARM, RISC-V); preferred inside retry loops because it avoids a separate load instruction.

```
  CAS Lock-Free Update Pattern:

  Thread A                      Thread B
  ──────────────────────        ──────────────────────
  read val (= 0)                read val (= 0)
  CAS(val, 0, 1) ──► SUCCESS    CAS(val, 0, 1) ──► FAIL (val is now 1)
  val is now 1                  expected = 1 (updated)
                                CAS(val, 1, 2) ──► SUCCESS
                                val is now 2
```

### ABA Problem

CAS sees the **same value but the object was replaced**: pointer A → B → A between load and CAS; **CAS succeeds incorrectly**.

```
  ABA Problem:

  Thread A: reads ptr = 0xABC0 (points to Node A)
  Thread B: removes Node A, inserts Node C at 0xABC0 (same address, new object!)
  Thread A: CAS(ptr, 0xABC0, new_node) — SUCCEEDS incorrectly
            Node A may be freed; dereference = use-after-free
```

Solutions:
- **Version tag**: pack a monotonically increasing counter alongside the pointer in a 128-bit CAS (`CMPXCHG16B` on x86-64); pointer change is always detectable
- **Hazard pointers**: reader publishes pointer before dereferencing; reclaimer scans all hazard pointer slots before freeing
- **RCU**: grace period mechanism ensures old objects are not freed until all readers are done

> **Common Pitfall:** ABA is subtle because the buggy case requires exact memory address reuse — easy to miss in testing, dangerous in production allocators that reuse freed memory. Always consider ABA when writing CAS-based list or stack manipulations. Use tagged pointers or RCU instead.

---

## SPSC Ring Buffer (Lock-Free)

**Single-producer / single-consumer** ring buffer requires only `acquire`/`release` atomics on head and tail indices. **No locks, no cache-line bouncing** on payload data.

```c
#define N 256  // must be power of 2 for efficient modulo via bitmask
T buf[N];
atomic_size_t head = 0, tail = 0;  // head: consumer read position, tail: producer write position

// Producer (single thread only — SPSC means exactly one producer)
buf[tail % N] = item;             // write item to current tail slot
// Release store: all writes to buf[tail%N] are visible before this store is seen
atomic_store_explicit(&tail, tail + 1, memory_order_release);

// Consumer (single thread only)
size_t t = atomic_load_explicit(&tail, memory_order_acquire);
// Acquire load: reads tail, and guarantees we see all buf writes that preceded the tail store
if (head != t) {                  // check if there is an item to consume
    item = buf[head % N];         // read item from current head slot
    atomic_store_explicit(&head, head + 1, memory_order_release);
}
```

The acquire-release pair on tail is the synchronization point: the producer releases ownership of the data by updating tail; the consumer acquires ownership by reading tail. No lock needed because there is exactly one writer and one reader.

**Throughput**: limited only by memory bandwidth; **no lock overhead**. Used in openpilot VisionIPC for **zero-copy camera frame passing** from capture process to inference process at 30Hz.

> **Key Insight:** The SPSC ring buffer is the canonical example of why `memory_order_acquire`/`release` exist. A `relaxed` store of tail would be atomic, but the consumer might see the updated tail *before* seeing the data written to `buf[tail%N]`. The release/acquire pair creates an explicit ordering guarantee: "the data in buf is ready by the time you see the tail increment."

---

## RCU (Read-Copy-Update)

Linux kernel's primary mechanism for **read-mostly shared data structures**. Achieves **O(1) read-side overhead**: no locks, no atomics, no cache-line writes. RCU is conceptually the most powerful mechanism in this lecture.

The librarian analogy from the overview maps directly to RCU: readers pick up the book (pointer) without asking; the librarian replaces it by placing the new edition on the shelf, then waiting until everyone reading the old edition finishes before removing it.

### Reader Side

```c
rcu_read_lock();                    // disable preemption only; O(1); no memory barrier emitted on most architectures
ptr = rcu_dereference(gp);          // loads pointer with READ_ONCE + compiler barrier
                                    // prevents compiler from caching or reordering the pointer load
/* safely use *ptr — guaranteed to remain valid until rcu_read_unlock */
rcu_read_unlock();                  // re-enable preemption
```

Only cost: disable/enable preemption — a single per-CPU flag write.

### Writer Side

The write side follows a strict sequence: copy, modify, publish, wait, free.

1. Allocate and populate a new version of the object
2. Atomically publish the new pointer (so new readers see the new version)
3. Wait for a **grace period** (all pre-existing readers have finished)
4. Free the old object (safe: no reader can still hold a reference)

```c
new_obj = kmalloc(sizeof(*new_obj), GFP_KERNEL);  // allocate new version
*new_obj = *old_obj;               // copy: start from the current state
new_obj->field = updated_value;    // modify the copy — old_obj is still visible to current readers
rcu_assign_pointer(gp, new_obj);   // atomic publish: smp_wmb() + pointer store; new readers see new_obj
synchronize_rcu();                 // BLOCK until all CPUs have exited any RCU read section that started
                                   // before rcu_assign_pointer — the "grace period"
kfree(old_obj);                    // safe: no pre-existing reader holds old_obj anymore
```

`call_rcu(&old->rcu_head, free_fn)`: asynchronous callback form; avoids blocking `synchronize_rcu()`; used in interrupt context or when writer must not block.

### Grace Period

A **grace period** is the interval after `rcu_assign_pointer()` until every CPU has passed through a **quiescent state**: a context switch, entry to idle, or return to user space. After the grace period, no pre-existing reader can hold a reference to the old pointer.

```
  RCU Grace Period Timeline:

  CPU 0: [RCU read A]──────────────────────[done]
  CPU 1: [RCU read A]──────────[done]
  CPU 2:                 [context switch]           ← quiescent state
  CPU 3: [idle]──────[idle]                         ← quiescent state

  Writer: rcu_assign_pointer(gp, new) ──► ... wait ... ──► kfree(old)
                                          ◄── grace period ──►
                                    (ends when all CPUs have passed
                                     through at least one quiescent state)
```

### RCU Variants

| Variant | Read Section Can Sleep? | Use Case |
|---|---|---|
| Classic RCU | No | Routing tables, `task_struct` lookup, module parameters |
| SRCU (Sleepable RCU) | Yes | Notifier chains, subsystem registrations |
| PREEMPT_RCU | No (but preemptible) | `CONFIG_PREEMPT` kernels |

### rcu_nocbs= Kernel Parameter

`rcu_nocbs=<cpulist>` **offloads `call_rcu` callbacks** to `rcuoc` kthreads running on non-isolated CPUs. Eliminates RCU callback invocations from isolated RT or inference cores, removing a source of **unpredictable latency jitter**.

> **Key Insight:** RCU's power comes from moving all overhead to the writer side. Readers are completely free: no lock, no atomic, no memory barrier. This is why RCU scales perfectly with reader count — adding more readers adds exactly zero overhead. The writer pays a grace period, but writers are rare on read-mostly structures like routing tables, process lists, and model configurations.

---

## kfifo — Kernel Lock-Free SPSC FIFO

```c
DECLARE_KFIFO(my_fifo, int, 64);   // statically declare a SPSC FIFO of 64 ints; size must be power of 2
INIT_KFIFO(my_fifo);               // initialize head and tail indices to 0

kfifo_put(&my_fifo, value);        // producer: enqueue; safe without lock for SPSC
kfifo_get(&my_fifo, &value);       // consumer: dequeue; safe without lock for SPSC
kfifo_len(&my_fifo);               // number of elements currently available
```

Used in kernel drivers for RX data buffers between ISR (producer) and process context (consumer). `kfifo` is the kernel's standard-library version of the SPSC ring buffer pattern described above.

---

## Hazard Pointers

**Userspace alternative to RCU** for lock-free memory reclamation:
- Reader **publishes** the pointer it is about to dereference into a per-thread hazard pointer slot
- Before freeing, the reclaimer scans all hazard pointer slots for the pointer
- If found, deferral is required; if not found, free is safe

```
  Hazard Pointer Reclamation:

  Thread A reading ptr P:    HP[A] = P         ← "I am using this pointer"
  Thread B freeing ptr P:    scan all HP slots
                             HP[A] == P? YES → defer free
  Thread A done:             HP[A] = NULL
  Thread B retries:          scan all HP slots
                             HP[A] == NULL? YES → kfree(P) safe
```

Used in: Folly (Meta), Java `java.util.concurrent`. `std::hazard_pointer` standardized in C++26.

---

## Summary

| Technique | Reader Overhead | Writer Overhead | Safe for ISR? | Main Limitation |
|---|---|---|---|---|
| Spinlock | Cache-line write + spin | Cache-line write | Yes | Wasted CPU; no sleep |
| CAS loop | Atomic RMW + retry | Atomic RMW + retry | Yes | ABA problem; retry cost |
| SPSC ring buffer | Acquire load | Release store | Yes (producer or consumer) | Single producer AND single consumer only |
| RCU | Preempt disable/enable | Copy + grace period | No (synchronize_rcu blocks) | Writer pays grace period; read-mostly |
| Hazard pointers | Publish + load | Scan all HP slots | No | Higher reclamation overhead than RCU |
| `kfifo` | Acquire load | Release store | Yes | Kernel-only; SPSC only |

### Conceptual Review

- **Why is `memory_order_relaxed` wrong for producer-consumer signaling?** Relaxed only guarantees atomicity of the atomic variable itself. It does not guarantee that data written before a `relaxed` store is visible to a thread that reads the `relaxed` variable. Use `release`/`acquire` to establish a happens-before relationship.
- **What is the fundamental constraint of SPSC ring buffers?** Exactly one producer thread and exactly one consumer thread. With multiple producers or consumers, the head/tail updates become races that require additional synchronization (MPMC queues are significantly more complex).
- **Why does RCU work without any reader-side locks?** RCU leverages the CPU's quiescent state detection: every context switch, idle entry, or user-space return is evidence that no active RCU read section is in progress on that CPU. The kernel tracks these events and declares a grace period complete only after all CPUs have quiesced.
- **What problem does `rcu_nocbs=` solve on isolated inference cores?** Normally, `call_rcu` callbacks execute on the CPU that queued them. On an isolated inference core, this means RCU callbacks fire inside the inference loop at unpredictable times. `rcu_nocbs=` offloads these callbacks to helper threads on non-isolated CPUs, eliminating that latency source.
- **When does CAS fail, and what should the retry loop do?** CAS fails when another thread changed the value between the load and the CAS instruction. On failure, CAS writes the current value into the `expected` variable. The retry loop should re-read dependent state, recompute the desired new value, and try again — not blindly retry with stale computation.
- **How does RCU compare to a reader-writer lock for a model configuration?** An rwlock blocks new readers while a writer holds the write lock. RCU never blocks readers — the writer works on a private copy and publishes it atomically. For a 100Hz inference loop that reads configuration every cycle, even occasional brief write-lock blocking is unacceptable. RCU is the correct choice.

---

## AI Hardware Connection

- RCU enables live model configuration updates (LoRA adapter swaps, quantization config changes) without pausing inference threads; the writer publishes the new config atomically and the old config is freed only after all in-progress forward passes exit their read sections
- SPSC ring buffer with `acquire`/`release` atomics provides the zero-copy camera frame pipeline in openpilot VisionIPC, eliminating mutex overhead on the 30Hz video path between `camerad` and `modeld`
- `rcu_nocbs=` on Jetson inference-dedicated cores removes RCU callback execution jitter, enabling sub-millisecond worst-case latency bounds on isolated real-time inference cores
- CAS-based lock-free queue is used in openpilot VisionIPC for multi-producer sensor data aggregation: each sensor driver enqueues readings without a mutex; the inference thread drains the queue once per inference cycle
- Atomic reference counts (`kref`, `std::atomic<int>`) manage DMA-BUF buffer lifetimes across multiple GPU consumers in the V4L2 + CUDA pipeline; the buffer is freed only when the last consumer decrements the count to zero
- SPSC ring buffer is the correct data structure for ISR-to-inference-thread pipelines; using a mutex in an ISR would require `spin_lock_irqsave`, adding latency; the atomic ring buffer avoids this entirely
