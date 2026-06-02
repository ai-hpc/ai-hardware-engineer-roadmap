# Lecture Note 05 (L10, L11): Lock-Free Programming (RCU, Atomics) & Deadlock / Priority Inversion

**Combines:** Lecture L10 (Lock-Free: RCU, Atomics, Memory Ordering) and Lecture L11 (Deadlock, Priority Inversion, PI Mutexes).

---

## How This Note Is Organized

1. **Part 1 — Lock-free:** Why avoid locks; C11 memory model; atomics and CAS; RCU; SPSC ring buffer; hazard pointers.
2. **Part 2 — Deadlock & priority inversion:** Coffman conditions; prevention (lock ordering, trylock); priority inversion and Mars Pathfinder; PI and priority ceiling; lockdep; watchdog.

---

# Part 1: Lock-Free Programming — RCU, Atomics & Memory Ordering

**Context:** Locks add cost even when uncontended: **cache-line bouncing**, context switches, priority inversion. On hot paths (camera pipeline, sensor fusion, model config), **lock-free techniques use atomics and RCU** to coordinate without mutual exclusion.

---

## Why Lock-Free?

- **Cache-line bouncing:** When multiple CPUs repeatedly acquire and release the same lock, the lock's memory location is transferred back and forth between their caches. This transfer (the "bounce") causes delays of ~100–300 cycles on NUMA systems, as each CPU must fetch the latest copy from another CPU's cache before proceeding.
- **Context switches:** Blocked threads pay scheduler overhead (~1–10 µs).
- **Priority inversion:** Low-priority holder delays high-priority waiter (see Part 2).
- **Convoying:** Many threads queue behind one slow holder.

**Lock-free programming** means coordinating between threads without using traditional locks (like mutexes). Instead, it relies on **atomic** hardware instructions (such as compare-and-swap, or CAS) and mechanisms like RCU. "Lock-free" doesn't mean there's no coordination—rather, the coordination is done using atomics, which generally provides **more predictable and bounded costs** compared to locks.

---

## C11/C++11 Memory Model

Modern CPUs and compilers often **rearrange (reorder) memory operations** to improve performance, which can lead to one thread's updates not being visible to another thread right away, unless explicitly controlled. The **C11/C++11 memory model** describes how and when such reordering is allowed and gives tools (`std::atomic<T>` in C++ / `_Atomic T` in C) to specify the kind of visibility and ordering guarantees required between threads.


| Memory Order           | What It Guarantees                                                                                                                         |
| ---------------------- | ------------------------------------------------------------------------------------------------------------------------------------------ |
| `memory_order_relaxed` | Only guarantees that this operation is atomic (can't be interrupted), but makes no promises at all about the order of any memory accesses. |
| `memory_order_acquire` | Guarantees that all loads and stores written *after* this operation in your code will really happen *after* it (on this CPU).              |
| `memory_order_release` | Guarantees that all loads and stores written *before* this operation in your code will really happen *before* it (on this CPU).            |
| `memory_order_acq_rel` | Combines both acquire and release: useful for read-modify-write (like CAS), guarantees before+after ordering on this operation.            |
| `memory_order_seq_cst` | Strongest: behaves as if there is a single total order for all such operations amongst all threads; the usual default for atomics.         |


### When and why to use `memory_order_relaxed` effectively?

`memory_order_relaxed` is used when you need atomicity (no races on a variable), but you do **not** need to synchronize or order memory between threads. It gives *no* guarantees about visibility or ordering of other memory—only that each operation is atomic (indivisible).

#### Effective use cases:

- **Counters, statistics, ID generators:** When multiple threads increment a shared counter, but the *exact sequence* or *timing* of increments doesn't matter, only that no increments are lost.
- **Flags or progress markers that do NOT control publication of other data:** For example, reporting heartbeats, progress, or status updates, where missing the very latest value is acceptable and other data does not need to be synchronized with the flag.
- **Random number generators (atomic seed update):** You just want atomic update, consistency doesn't depend on timing with other memory.

#### When NOT to use:

- Producer–consumer or hand-off, where one thread must see *all* previous writes from another before proceeding. (See earlier example — `memory_order_relaxed` *cannot* guarantee this and can break correctness.)

#### Example: Safe with `memory_order_relaxed` (atomic counter)

```cpp
std::atomic<uint64_t> requests_handled{0};

void worker_thread() {
    // ... handle a request ...
    requests_handled.fetch_add(1, std::memory_order_relaxed);
    // No need to synchronize with any other memory.
}
```

Here, `memory_order_relaxed` is correct and most efficient: it prevents lost updates (atomicity), but doesn't incur expensive memory fences since ordering is irrelevant.

#### Key takeaway:

- Use `memory_order_relaxed` only when you want atomicity, **not** cross-thread visibility or ordering of other data.  
- For publishing data or signaling between threads, use `memory_order_release` (producer) and `memory_order_acquire` (consumer).

> In summary: `memory_order_relaxed` is fastest, but safe only for cases where you do not care about the order or visibility of anything except the atomic variable itself.

**Acquire-release pairing (Full Example):**  
A **release** store to variable X establishes a "happens-before" relationship with a subsequent **acquire** load from X in another thread. This means that all memory writes performed before the release store in one thread are guaranteed to be visible to the thread that does the acquire load after observing the value.

### Example: Producer–Consumer with C++11 Atomics

```cpp
#include <atomic>
#include <cassert>
#include <thread>
#include <iostream>

std::atomic<int> flag{0};
int data = 0;

void producer() {
    data = 42;  // (1) Store data first
    flag.store(1, std::memory_order_release);  // (2) Signal with release store
}

void consumer() {
    while (flag.load(std::memory_order_acquire) != 1) {
        // spin/wait until producer signals
    }
    // (3) After acquire load observes "1" in flag, all writes before the release are visible
    assert(data == 42);  // Always succeeds!
    std::cout << "Consumer sees data: " << data << std::endl;
}

int main() {
    std::thread t1(producer);
    std::thread t2(consumer);
    t1.join();
    t2.join();
    return 0;
}
```

**What this demonstrates:**  

- The producer writes to `data`, then does a `flag.store(..., memory_order_release)`
- The consumer spins until it sees `flag == 1` via `flag.load(..., memory_order_acquire)`
- After seeing the flag set, the consumer is *guaranteed* to see `data == 42`, as all writes before the release store are visible after the acquire load across threads.

> **Common pitfall:** Using `memory_order_relaxed` for a "data ready" flag only guarantees atomicity of the flag — it does *not* guarantee that the payload written before the flag store is visible to the reader. Always use **release** (producer) / **acquire** (consumer) for producer-consumer signaling.

### Example: `memory_order_acq_rel` with compare-exchange

Read-modify-write operations (e.g. CAS) need to both **publish** the new value to other threads and **observe** prior writes. `memory_order_acq_rel` does both on the same operation: on success, the CAS acts as acquire (sees the latest state) and release (makes the update visible). Use it when the atomic is the single synchronization point for a shared structure (e.g. lock-free counter or stack pointer).

```cpp
#include <atomic>
#include <iostream>
#include <thread>
#include <vector>

std::atomic<int> counter{0};

void increment() {
    int expected = counter.load(std::memory_order_relaxed);
    while (!counter.compare_exchange_strong(
        expected, expected + 1,
        std::memory_order_acq_rel,   // success: publish our write + see others'
        std::memory_order_acquire))   // failure: only acquire (expected gets current)
    {
        // expected was updated; retry with new value
    }
}

int main() {
    const int num_threads = 4;
    const int increments_per_thread = 100000;
    std::vector<std::thread> threads;

    for (int i = 0; i < num_threads; ++i) {
        threads.emplace_back([&]() {
            for (int j = 0; j < increments_per_thread; ++j) {
                increment();
            }
        });
    }
    for (auto& t : threads)
        t.join();

    std::cout << "counter = " << counter.load() << " (expected "
              << num_threads * increments_per_thread << ")\n";
    return 0;
}
```

**How the CAS loop works:**

- `**compare_exchange_strong(expected, desired, success_order, failure_order)`** does one atomic step:
  - **If** `counter`’s current value equals `expected`: store `desired` (i.e. `expected + 1`) into `counter` and return `true`.
  - **Else**: leave `counter` unchanged, **write the current value of `counter` into `expected`**, and return `false`.
- `**while (! ...)**` means: keep retrying until the exchange succeeds. On each failure, another thread changed `counter` before we did, and the CPU already put that new value into `expected`, so the next iteration uses an up-to-date `expected` (we don’t need an extra load).
- **Success order `memory_order_acq_rel`:** When we successfully store `expected + 1`, we both **release** (so our write is visible to others) and **acquire** (so we see all prior writes by other threads). That keeps the counter consistent with any other shared state tied to it.
- **Failure order `memory_order_acquire`:** When the compare fails, we don’t modify `counter`; we only **read** it. Acquire is enough so we see the latest value (and `expected` is updated). We don’t need release on failure because we didn’t write.

Without the loop, a single CAS could fail (e.g. another thread incremented between our load and our CAS), and we would have skipped an increment. The loop retries until our increment is the one that “wins.”

### Example: `memory_order_seq_cst` — single total order

`memory_order_seq_cst` gives a **single total order** of all seq_cst operations across all threads. Every thread agrees on the order of seq_cst loads and stores. This is useful when you want a simple, globally consistent view of a small coordination point. It is the default for `std::atomic` operations when you don’t specify an order.

```cpp
#include <atomic>
#include <iostream>
#include <thread>

std::atomic<bool> stop{false};
std::atomic<int> next_id{0};

void worker() {
    while (!stop.load(std::memory_order_seq_cst)) {
        // do work
    }
}

void request_stop() {
    stop.store(true, std::memory_order_seq_cst);
}

int allocate_id() {
    return next_id.fetch_add(1, std::memory_order_seq_cst);
}

int main() {
    std::thread t1(worker);
    std::thread t2(worker);

    std::cout << "Allocated id: " << allocate_id() << "\n";

    request_stop();

    t1.join();
    t2.join();
    return 0;
}
```

**What this shows:**

- `stop` is a simple shutdown flag that many worker threads can check.
- `request_stop()` sets the flag once, and every worker eventually sees it and exits.
- `allocate_id()` uses `fetch_add(1)` to hand out a strictly increasing ID across threads.

**Real-world example:** imagine a camera or web server processing many tasks at once. Each incoming request or frame needs a unique ID so logs, traces, and results can be matched later. `allocate_id()` acts like a ticket dispenser:

- thread A gets ticket 0
- thread B gets ticket 1
- thread C gets ticket 2

Even if several threads call it at the same time, no two of them get the same number. That is much safer than doing `id = next_id; next_id = next_id + 1;`, which can race and produce duplicates.

**Real-world shutdown example:** in a blockchain miner, the stop flag tells all mining threads to stop cleanly when there is no more PoW work to do:

```cpp
std::atomic<bool> stop{false};

void worker() {
    while (!stop.load(std::memory_order_seq_cst)) {
        // try a nonce, test PoW, submit a share if found
    }
}

void miner_manager() {
    // wait for new block template / work
    while (!stop.load(std::memory_order_seq_cst)) {
        // dispatch work to miners

        // if no new work is available and mining should pause/stop:
        // stop.store(true, std::memory_order_seq_cst);
    }
}

void request_stop() {
    stop.store(true, std::memory_order_seq_cst);
}
```

- `stop == false` means miners should keep searching nonces.
- `request_stop()` sets the flag once when the manager decides to stop, pause, or exit.
- Each worker checks the flag in its loop, finishes the current attempt, and exits cleanly.

This avoids wasting CPU on useless hashes, leaving threads running after the pool is empty, or hard-killing workers in the middle of shared-state updates. So yes, one manager thread can set a shared atomic flag, and all mining threads will stop on the next check.

Why `seq_cst` fits here:

- It is easy to reason about.
- Every thread sees one global order of the atomic operations.
- It avoids subtle bugs when the flag is part of a larger concurrent protocol.

In real systems, `seq_cst` is often chosen for:

- shutdown flags
- “ready” signals
- global counters
- debug builds and correctness-first code
- small coordination points where performance is not the main concern

If you need more performance and can tolerate a weaker guarantee, acquire/release is often enough for a shutdown flag. But for simple control paths, `seq_cst` is the safest and clearest choice.

---

## Hardware Memory Models & Kernel Barriers

**What does this mean?**  
Different CPUs can observe memory operations in different orders *unless* the programmer uses explicit controls to enforce order. This is called the **hardware memory model**. On some CPUs (like x86-64), memory accesses appear mostly in the order written in the program. On others (like ARM64 and RISC-V), they can be freely reordered for performance unless you add explicit synchronization.

Because of these differences, in kernel (or low-level) code you often need to insert **memory barriers**: special instructions that tell the CPU, "do not reorder memory accesses across this point." These can be explicit (special instructions) or implicit (using C11 atomics with specific orderings).


| Architecture | Memory model                     | What this means / Barrier requirements                                                                                                                                             |
| ------------ | -------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| x86-64       | TSO (Total Store Order) — strong | Almost always keeps stores/loads in order; few explicit barriers needed. Atomic operations use `LOCK` prefix.                                                                      |
| ARM64        | Weakly ordered                   | Loads/stores can reorder; you *must* use barriers: `DMB`/`DSB` (memory barrier instructions), and `LDAR`/`STLR` (acquire/release forms of load/store) for correct synchronization. |
| RISC-V       | Weak (RVWMO)                     | Also can reorder; synchronize with `FENCE` instructions and use `LR`/`SC` for LL/SC-style atomics.                                                                                 |


**Linux kernel barrier macros:** (portable APIs that expand to the right instructions for the architecture)

- `smp_mb()` — **Memory Barrier:** Ensures both loads and stores before the barrier are globally observed before any after it.
- `smp_rmb()` — **Read (load) Memory Barrier:** Prevent reordering of loads across the barrier.
- `smp_wmb()` — **Write (store) Memory Barrier:** Prevent reordering of stores across the barrier.

**Practical example/meaning:**  
On ARM64 (e.g. Jetson Orin), even "simple" multithreaded code that works on x86 (because of its strong memory ordering) can break unless you use atomics/barriers — the CPU might reorder updates and readers could see stale or inconsistent data. That's why kernel (and C11/C++11) code needs explicit atomic operations or memory barriers for shared variables: to *guarantee* all threads see updates in the intended order and prevent bugs that only show up on weakly ordered chips.

**Example: Why memory barriers are needed**

Suppose you have a simple producer/consumer flag protocol:

```c
// Producer
data = 123;
flag = 1;

// Consumer
if (flag == 1) {
    assert(data == 123);  // Could fail!
}
```

On x86, this almost always works as expected. But on ARM64 or RISC-V, the CPU may reorder the store to `flag` before `data`, so the consumer could see `flag == 1` but `data` still having an old or garbage value.

**Fix with barriers:**

```c
// Producer (ARM64-style)
data = 123;
smp_wmb();      // Ensure data is visible before flag
flag = 1;

// Consumer
if (flag == 1) {
    smp_rmb();  // Ensure load of data happens after seeing the flag
    assert(data == 123);  // Now guaranteed
}
```

By adding `smp_wmb()` before setting the flag (producer) and `smp_rmb()` after checking the flag (consumer), you prevent reordering and ensure correct visibility of `data` when `flag` indicates it is ready.

---

## Atomic Operations (Kernel)

Map to single indivisible hardware instructions: `LOCK ADD`/`XADD`/`CMPXCHG` on x86, `LDADD`/`CAS` on ARMv8.1 LSE.

```c
atomic_t counter = ATOMIC_INIT(0);
atomic_inc(&counter);                   // LOCK XADD / LDADD
atomic_dec_and_test(&counter);          // decrement; return true if zero (refcounts)
int old = atomic_cmpxchg(&counter, 5, 10);  // if counter==5 set to 10; return old value
atomic_add_return(n, &counter);         // add n, return new value (rate limiting)
```

- `atomic_t` is 32-bit; `atomic64_t` for 64-bit. Use for: reference counts, flags, per-CPU statistics.

## Compare-and-Swap (CAS): Meaning

**Meaning:**  
Compare-and-Swap (CAS) is a hardware-supported atomic instruction that checks if a memory location contains an expected value, and if so, updates it to a new value—**all in one indivisible step**. This is the basic building block for implementing data structures and algorithms that allow multiple threads to coordinate **without using locks**. With CAS, threads can safely update shared data despite running at the same time, because the operation guarantees that no two threads can both succeed at the same update.

**Practical use:**  
You use CAS to implement things like lock-free queues, stacks, reference counters, and other algorithms where multiple threads might try to update the same value at once. If two threads race to update a value, only one will succeed. The other will see that it lost the race and can try again.

**Code example:**

```cpp
std::atomic<int> val{0};
int expected = 0;
bool ok = val.compare_exchange_strong(expected, 1,
    std::memory_order_acq_rel,   // On success: this thread sees all prior writes/updates, and its own write is visible to others.
    std::memory_order_acquire);  // On failure: just read latest, no write.
```

- If `val` currently equals `expected` (0), set it to 1, and `ok` is true.
- If not, `val` is unchanged, `expected` is updated to whatever the value was, `ok` is false, and you typically retry.
- **Inside a loop**, you usually use `compare_exchange_weak` (can give false negatives on some hardware but may be faster).

**Pitfall — the ABA problem (meaning):**  
Suppose a pointer's value is A, then it gets changed to B, then *back* to A (but now A points to a different object). CAS only checks for "A", and if it sees A, it assumes nothing changed—but in reality, the object at A may have been deleted and reused, causing a use-after-free bug.

**Meaning of ABA Solutions:**  

1. **Version tags:** Attach a counter to the pointer, so any change increments the count (need double-wide atomic compare for this, e.g. 128-bit CAS).
2. **Hazard pointers:** Readers advertise "I'm using this pointer"; writers only free memory after no reader is using it.
3. **RCU:** Updates ensure old values are not freed until all prior readers have finished, preventing the bug.

**Example — how version tags fix ABA:**  
Store both the pointer and a version number together. If the pointer changes away from `A` and later comes back to `A`, the version will still be different, so CAS will fail instead of being fooled by the same address.

```cpp
struct TaggedPtr {
    Node* ptr;
    uint64_t version;
};

std::atomic<TaggedPtr> head;

void update_head(Node* old_ptr, Node* new_ptr) {
    TaggedPtr expected = {old_ptr, old_ptr ? old_ptr->version : 0};
    TaggedPtr desired  = {new_ptr, expected.version + 1};

    // Fails if either the pointer OR the version changed.
    head.compare_exchange_strong(expected, desired,
                                 std::memory_order_acq_rel,
                                 std::memory_order_acquire);
}
```

If another thread changes `head` from `A` to `B` and then back to `A`, the pointer value may look the same, but the version will be higher. That tells us the object was modified in between, so the stale CAS does not succeed.

In summary:  
CAS is "atomic if equal, else update and retry", forming the foundation of safe, efficient lock-free programming by providing a way for threads to agree on shared memory changes *without* locks—just using a built-in check-and-swap step that never halfway updates a value.

---

## SPSC Ring Buffer (Lock-Free): In-Depth with MengRao's SPSC_Queue

The lock-free **SPSC** ("Single Producer, Single Consumer") queue is an efficient queue for communication between **exactly one producer thread and one consumer thread**. Let's break down how it works, *inspired by and referencing the widely respected [MengRao/SPSC_Queue](https://github.com/MengRao/SPSC_Queue) implementation*, which is a gold standard for high-performance SPSC queues.

### Key Properties

- **Lock-Free:** No thread ever blocks or waits; all queue operations are non-blocking and progress independently.
- **No False Sharing:** Producer and consumer only touch separate synchronization variables, reducing cache line bouncing.
- **Cache Efficiency:** Payload buffer is never "locked" by both sides at once, so high throughput is possible (critical for high-rate pipelines like camera frames or audio).

---

### Basic Structure

MengRao's queue, and most fast SPSC queues, use the following design:

- **Buffer:** A fixed-size circular buffer (array), where all slots are preallocated and accessed modulo the buffer size (which is a power of 2).
- **Separate Indices:** Two indices, one for the producer (`tail` or `write_idx`) and one for the consumer (`head` or `read_idx`), each owned and only updated by one thread.
- **Fences:** Use of proper memory orderings (`release`, `acquire`) ensures that data written by the producer is visible to the consumer **only after** the consumer observes the index update, and vice versa.

### Reference Structure — MengRao SPSC_Queue

Here is an annotated version, blending the original style with C/C++ conventions (for full detail see [MengRao/SPSC_Queue/queue_spsc.h](https://github.com/MengRao/SPSC_Queue/blob/master/queue_spsc.h)):

```cpp
template<typename T, size_t N = 256>
class SPSCQueue {
    alignas(64) T buf[N]; // Circular buffer for data; cache-line aligned to avoid false sharing
    alignas(64) std::atomic<size_t> head = 0; // Read index — owned only by consumer
    alignas(64) std::atomic<size_t> tail = 0; // Write index — owned only by producer

public:
    // Called only from the producer thread
    bool enqueue(const T &item) {
        size_t tail_cache = tail.load(std::memory_order_relaxed);
        size_t head_cache = head.load(std::memory_order_acquire); // see what consumer has consumed
        if (((tail_cache + 1) & (N - 1)) == (head_cache & (N - 1))) {
            // Buffer is full (one slot left empty to avoid overwrite ambiguity)
            return false;
        }
        buf[tail_cache & (N - 1)] = item;
        tail.store(tail_cache + 1, std::memory_order_release); // publish
        return true;
    }

    // Called only from the consumer thread
    bool dequeue(T &item) {
        size_t head_cache = head.load(std::memory_order_relaxed);
        size_t tail_cache = tail.load(std::memory_order_acquire); // see what producer has added
        if ((head_cache & (N - 1)) == (tail_cache & (N - 1))) {
            // Buffer is empty
            return false;
        }
        item = buf[head_cache & (N - 1)];
        head.store(head_cache + 1, std::memory_order_release);
        return true;
    }
};
```

#### Key Points:

- **Single Ownership Principle:** Only the consumer thread changes `head`, and only the producer changes `tail`. Each thread reads (but does not write) the other's index.
- **Wrap-around:** Buffer size is a power of two — wrapping is done with a bitmask (`& (N - 1)`) for speed.
- **Full vs. Empty:** We keep one slot empty intentionally, so `tail + 1 == head` means full, `tail == head` means empty (this avoids ambiguity).
- **Memory Barriers:**
  - `store(..., memory_order_release)` ensures that all prior stores to the buffer are visible before publishing the index.
  - `load(..., memory_order_acquire)` ensures that the subsequent buffer reads see the latest data.

### Why Is This Better Than Locks or Naive Atomics?

- **Zero contention on data:** Only the relevant thread ever touches its own index; the buffer slot is always written *then* the index updated.
- **No locks, no blocking:** No mutex, semaphore, or heavy synchronization.
- **Scalable for high-throughput:** On modern CPUs, with careful alignment and use of explicit atomics, this design can achieve tens or hundreds of millions of messages per second with minimal cache traffic.

### Practical Usage (like OpenPilot VisionIPC, ML data streaming, etc.)

- Used in high-throughput situations like camera frame streaming, audio pipelines, or neural model pipes, where classic locks or blocking sync would be a performance bottleneck.
- **Constraint:** You **must** have exactly one producer and one consumer. If you need multiple, see MPMC (multi-producer multi-consumer) queues, which are much more complex.

---

### Summary Table


| Feature           | SPSC MengRao Queue     |
| ----------------- | ---------------------- |
| Threads supported | 1 producer, 1 consumer |
| Lock required?    | No                     |
| Blocking/Waiting? | Never                  |
| Throughput        | Extremely high         |
| Caveat            | Exactly 1 prod/1 cons  |
| Used in           | Networking, vision, ML |


---

**See more:**  

- [MengRao/SPSC_Queue - queue_spsc.h](https://github.com/MengRao/SPSC_Queue/blob/master/queue_spsc.h) — See also the README for benchmarking and usage.

---

## RCU (Read-Copy-Update)

Linux kernel’s **primary mechanism for read-mostly shared data**. **O(1) read-side cost:** no locks, no atomics, no cache-line writes — only preempt disable/enable.

### Reader side

```c
rcu_read_lock();
ptr = rcu_dereference(gp);   // READ_ONCE + compiler barrier
/* use *ptr — guaranteed valid until rcu_read_unlock */
rcu_read_unlock();
```

### Writer side (copy → modify → publish → wait → free)

1. Allocate and populate a new version of the object.
2. Publish: `rcu_assign_pointer(gp, new_obj)` (smp_wmb + pointer store).
3. Wait for a **grace period**: all pre-existing readers have finished.
4. Free the old object (no reader can still hold it).

```c
new_obj = kmalloc(sizeof(*new_obj), GFP_KERNEL);
*new_obj = *old_obj;
new_obj->field = updated_value;
rcu_assign_pointer(gp, new_obj);
synchronize_rcu();   // blocks until grace period
kfree(old_obj);
```

**call_rcu(&old->rcu_head, free_fn):** asynchronous form; writer does not block; used in interrupt context or when writer must not block.

### Grace period

A **grace period** is the interval after `rcu_assign_pointer()` until every CPU has passed a **quiescent state** (context switch, entry to idle, or return to user space). The kernel tracks these events; when all CPUs have quiesced, no pre-existing reader can still hold the old pointer.

### RCU variants


| Variant              | Read section can sleep? | Use case                                              |
| -------------------- | ----------------------- | ----------------------------------------------------- |
| Classic RCU          | No                      | Routing tables, task_struct lookup, module parameters |
| SRCU (Sleepable RCU) | Yes                     | Notifier chains, subsystem registrations              |
| PREEMPT_RCU          | No (but preemptible)    | CONFIG_PREEMPT kernels                                |


**rcu_nocbs=`<cpulist>`:** Offloads `call_rcu` callbacks to `rcuoc` kthreads on non-isolated CPUs. On an isolated inference core, normal RCU would run callbacks there at unpredictable times; `rcu_nocbs=` removes that latency source.

---

## kfifo (Kernel SPSC FIFO)

```c
DECLARE_KFIFO(my_fifo, int, 64);   // size power of 2
INIT_KFIFO(my_fifo);
kfifo_put(&my_fifo, value);        // producer
kfifo_get(&my_fifo, &value);       // consumer
kfifo_len(&my_fifo);
```

Used in kernel drivers for RX buffers between ISR (producer) and process context (consumer). Kernel’s standard lock-free SPSC FIFO.

## Hazard Pointers

Userspace alternative to RCU for lock-free reclamation:

- Reader **publishes** the pointer it is about to dereference into a per-thread hazard pointer slot.
- Before freeing, the reclaimer scans all hazard pointer slots for that pointer.
- If found → defer free; if not found → free is safe.

Used in Folly (Meta), Java `java.util.concurrent`; `std::hazard_pointer` in C++26.

---

## Part 1 Summary Table


| Technique       | Reader cost        | Writer cost         | Safe in ISR?                | Limitation                        |
| --------------- | ------------------ | ------------------- | --------------------------- | --------------------------------- |
| Spinlock        | Cache write + spin | Cache write         | Yes                         | Wasted CPU; no sleep              |
| CAS loop        | Atomic RMW + retry | Atomic RMW + retry  | Yes                         | ABA; retry cost                   |
| SPSC ring       | Acquire load       | Release store       | Yes (producer or consumer)  | Single producer AND consumer only |
| RCU             | Preempt disable    | Copy + grace period | No (synchronize_rcu blocks) | Read-mostly; writer pays          |
| Hazard pointers | Publish + load     | Scan HP slots       | No                          | Reclamation overhead              |
| kfifo           | Acquire load       | Release store       | Yes                         | Kernel-only; SPSC only            |


### Part 1 — Conceptual review

- **Why is relaxed wrong for producer-consumer signaling?** Relaxed guarantees only atomicity of the atomic variable; it does not guarantee that data written before the store is visible to the reader. Use release/acquire for “data ready” flags.
- **Why does RCU work without reader-side locks?** The kernel detects **quiescent states** (context switch, idle, return to user). A grace period ends only after every CPU has quiesced, so no pre-existing reader can still hold the old pointer.
- **When does CAS fail and what should the retry loop do?** CAS fails when another thread changed the value. On failure, the `expected` variable is updated with the current value; the loop should re-read dependent state, recompute the new value, and retry — not retry with stale computation.
- **RCU vs rwlock for model config?** With rwlock, writers block new readers. With RCU, readers are never blocked; the writer works on a copy and publishes atomically. For a 100 Hz inference loop reading config every cycle, RCU is the right choice.

---

# Part 2: Deadlock, Priority Inversion & PI Mutexes

**Context:** Deadlock = tasks waiting for each other forever. Priority inversion = high-priority task blocked by a low-priority one (via a shared resource). Both require design-time and runtime measures (ordering, PI, lockdep).

---

## Deadlock: Definition & Coffman Conditions

**Deadlock:** A set of processes are each waiting for a resource held by another in the set; **no process can ever make progress**.

```
  Process A holds L1, waiting for L2  ──►  Process B holds L2, waiting for L1
  Neither can proceed. Both wait forever.
```

**Coffman conditions** (all four must hold for deadlock to be possible; break any one to prevent it):


| Condition            | Definition                                                                     |
| -------------------- | ------------------------------------------------------------------------------ |
| **Mutual exclusion** | At least one resource is non-sharable — only one process may hold it at a time |
| **Hold-and-wait**    | A process holds at least one resource while waiting to acquire another         |
| **No preemption**    | Resources cannot be forcibly taken; only voluntary release                     |
| **Circular wait**    | A cycle exists in the resource-allocation graph: P1→R1→P2→R2→P1                |


---

## Deadlock Prevention

Attack one of the four conditions at design time:


| Condition to break | Technique                                                             | Trade-off                                      |
| ------------------ | --------------------------------------------------------------------- | ---------------------------------------------- |
| Hold-and-wait      | Acquire all locks at once; release all if any acquisition fails       | Less concurrency; retry logic                  |
| Circular wait      | **Global lock ordering:** always acquire in a fixed, documented order | Discipline at all call sites; lockdep enforces |
| No preemption      | `mutex_trylock()` with randomized exponential backoff                 | Retry overhead; livelock risk                  |
| Mutual exclusion   | Lock-free data structures                                             | Higher implementation complexity               |


**Global lock ordering (step-by-step):**

1. Enumerate all mutexes/locks in the subsystem.
2. Assign each a numeric rank (e.g. Lock A = 1, Lock B = 2).
3. Mandate: always acquire in ascending rank order.
4. Document the ordering at the lock declaration site.
5. Use `lockdep_set_class` so lockdep can verify ordering automatically.
6. In code review, reject any patch that acquires a lower-ranked lock while holding a higher-ranked one.

> **Pitfall:** Ordering breaks when new locks are added without updating the global order — two developers can introduce an A→C→D and A→D cycle. Use lockdep in CI to catch cycles.

## Deadlock Detection & Recovery

- **Resource allocation graph:** Nodes = processes (P) and resources (R); edge P→R = P waits for R; edge R→P = R held by P. A cycle = deadlock.
- **Wait-for graph:** Simplified; edge P→Q means P waits for something held by Q; cycle = deadlock.

**Recovery options after detection:** (1) Abort one process and free its resources (choose victim by priority, runtime, resources held). (2) Preempt a resource from one process (requires rollback support). (3) Roll back to a safe checkpoint (requires checkpointing). **lockdep** in Linux detects *potential* deadlock at runtime when a new lock ordering is first seen — before an actual hang.

---

## Priority Inversion

**Mechanism (four steps):**

1. **L** (low priority) acquires mutex M.
2. **H** (high priority) wakes and tries to acquire M — blocks waiting for L.
3. **M** (medium priority) wakes and **preempts L** (M > L).
4. L never runs → cannot release M → H waits indefinitely despite being highest priority.

H is effectively running at L’s priority — **inverted**. Duration of inversion is **unbounded** without PI.

```
  Time ─────────────────────────────────────────────────────────►
  H (high):  [woken]──[BLOCKED on mutex held by L]────────────────►
  M (med):   ─────────────────────[RUNNING]────────────────────────►
  L (low):   [holds mutex]──[PREEMPTED by M]──────[never runs]
```

> **Insight:** Inversion can occur even when every task is correct. It’s an emergent effect of interactions; code review alone cannot catch it.

### Mars Pathfinder (1997)

Rover had **periodic resets** ~18 hours after landing. **Root cause:** unbounded priority inversion.


| Task     | Priority | Role                                    |
| -------- | -------- | --------------------------------------- |
| bc_dist  | High     | Data distribution bus; needed the mutex |
| bc_sched | Low      | Bus scheduler; held the mutex           |
| ASI_MET  | Medium   | Meteorological data; CPU-intensive      |


**Sequence:** bc_sched (low) held mutex → bc_dist (high) blocked → ASI_MET (medium) preempted bc_sched and ran → bc_sched never ran → bc_dist missed its watchdog → VxWorks watchdog fired → full system reset. **Fix:** Enable **priority inheritance** on the shared mutex (feature existed in VxWorks but was off by default). Fix was uploaded to the rover via uplink.

**Lesson:** PI must be the default for any mutex shared between tasks of different priorities in safety-critical or inaccessible systems.

---

## Priority Inheritance (PI)

When H blocks on a mutex held by L, the kernel **temporarily boosts L** to H’s priority until L releases the mutex. **Transitive PI:** if L is also blocked on another mutex held by X, the boost **propagates to X** so the whole chain can run and release.

```
  BEFORE PI:                          AFTER PI:
  H (90): BLOCKED                     H (90): BLOCKED
  M (50): RUNNING                     M (50): BLOCKED (cannot preempt L)
  L (10): PREEMPTED                   L (10→90): RUNNING → releases mutex → H runs
```

- **Linux:** `rtmutex` implements PI; PREEMPT_RT converts most `spinlock_t` to `rtmutex` → PI system-wide (e.g. CAN, V4L2, GPU drivers) without driver changes.
- **Userspace:**

```c
pthread_mutexattr_t attr;
pthread_mutexattr_init(&attr);
pthread_mutexattr_setprotocol(&attr, PTHREAD_PRIO_INHERIT);
pthread_mutex_init(&mutex, &attr);
```

> **Pitfall:** PI is most effective with SCHED_FIFO/SCHED_RR. On SCHED_OTHER (CFS), “priority” is relative and the boost may not have the desired effect.

---

## Priority Ceiling Protocol

Each mutex has a **ceiling priority** = highest priority of any task that will ever acquire it. Any task that acquires it **immediately** runs at the ceiling for the duration — so a **high-priority waiter never has to block** (L is already at ceiling). Prevents inversion without runtime discovery; requires **static analysis** (all acquirers and their priorities known at design time).


|                    | PI                            | Priority ceiling                    |
| ------------------ | ----------------------------- | ----------------------------------- |
| When boost happens | When H blocks on L            | On every acquisition                |
| Overhead           | Only when contention          | Every acquisition                   |
| Suited to          | Dynamic task sets, general RT | AUTOSAR, ARINC 653 (fixed task set) |


POSIX: `PTHREAD_PRIO_PROTECT`. **When mandatory:** AUTOSAR OS and ARINC 653 require formal bounded-inversion proofs; priority ceiling provides that statically.

---

## lockdep — Live Deadlock Detector

`CONFIG_PROVE_LOCKING=y`, `CONFIG_LOCK_STAT=y` (optional, per-lock contention stats).

- Assigns each lock a **lock class** (from static address in kernel binary).
- Records every acquisition chain: “lock A was held when lock B was acquired.”
- Reports **AB-BA cycles** the first time a new ordering is seen — before an actual hang.
- Reports **invalid context:** e.g. `mutex_lock()` from hardirq (sleeping lock in IRQ).

Example report:

```
WARNING: possible circular locking dependency detected
task/1234 is trying to acquire lock: (&lockB){...}, at: function_b+0x30
but task holds lock: (&lockA){...}, taken at: function_a+0x20
which lock already depends on the new lock.
```

- **lockdep_assert_held(&lock):** Documents and verifies that a lock is held at a given point.
- Enable in development/CI; disable in production (~10% overhead).

## Watchdog Timers

If a process does not **kick the watchdog** within the timeout, the system resets or enters a safe state — **defense-in-depth** against deadlocks that slip through. On Mars Pathfinder the watchdog **did** detect bc_dist missing its deadline; the real fix was enabling PI so the deadline was not missed. **Pitfall:** Treating watchdog reset as an acceptable “recovery” for priority inversion is wrong — e.g. on an ADAS controller it forces disengagement; the correct fix is to eliminate the inversion.

---

## Part 2 Summary Table


| Problem            | Symptom                     | Detection               | Solution                            |
| ------------------ | --------------------------- | ----------------------- | ----------------------------------- |
| Deadlock           | Hang; tasks blocked forever | Resource graph; lockdep | Lock ordering; acquire-all; trylock |
| Priority inversion | High-priority task stalls   | Latency; watchdog       | PI mutex; priority ceiling          |
| Livelock           | CPU busy, no progress       | Profiling               | Backoff; arbitration                |
| Starvation         | Low-priority never runs     | Wait monitoring         | Aging; FIFO; boost                  |


### Part 2 — Conceptual review

- **Why can priority inversion occur with “correct” code?** It’s emergent: L holds the lock correctly, H waits correctly, M preempts correctly. The failure is in the system design (shared mutex across priority levels without PI).
- **Mars Pathfinder lesson:** PI existed in VxWorks but was off by default. Safety-critical systems must audit every mutex shared across priorities and enable PI.
- **PI vs priority ceiling?** PI boosts the holder only when a higher-priority waiter blocks (dynamic). Ceiling boosts the acquirer to ceiling on every acquisition (static). Ceiling has stronger predictability for fixed task sets (AUTOSAR, ARINC 653).
- **What does lockdep *not* report?** It does not report priority inversion — that needs latency analysis (e.g. cyclictest) or scheduling analysis.
- **Why is transitive PI important?** If H blocks on L and L blocks on X, X must also be boosted; otherwise L cannot run and cannot release the mutex for H.

---

## AI Hardware Connection

- **RCU:** Live model config and LoRA swaps without pausing inference; writer publishes new config, old freed after grace period. Readers (inference threads) never block.
- **SPSC ring + acquire/release:** Zero-copy camera path (e.g. openpilot VisionIPC between camerad and modeld); no mutex on the 30 Hz frame path; throughput limited by memory bandwidth.
- **rcu_nocbs=:** On Jetson/inference-dedicated cores, offloads RCU callbacks to helper CPUs so isolated RT cores don’t see callback jitter.
- **CAS-based queues:** Used in VisionIPC-style pipelines for multi-producer sensor aggregation; each sensor enqueues without a mutex; inference thread drains once per cycle.
- **Atomic refcounts (kref, std::atomic):** Manage DMA-BUF buffer lifetime across V4L2 and GPU consumers; buffer freed when last consumer decrements to zero.
- **PI mutex (rtmutex / PTHREAD_PRIO_INHERIT):** Required for openpilot controlsd/plannerd and cereal shared state with plannerd (medium) and loggers (low); Mars Pathfinder is a standard reference in ASIL-B safety reviews.
- **ROS2:** rclcpp real-time executor docs explicitly require PTHREAD_PRIO_INHERIT so control callbacks are not starved by data-logging threads.
- **lockdep:** Standard in camera, ISP, and DMA driver bring-up on Jetson/i.MX before deployment; AB-BA cycles must be eliminated.
- **Priority ceiling:** Used in AUTOSAR ECUs with fixed task sets; stronger for ASIL-D certified functions.
- **PREEMPT_RT:** System-wide spinlock→rtmutex gives PI automatically on CAN, V4L2, and GPU drivers without driver code changes.

---

*Lecture Note 05 — Combines L10 (Lock-Free: RCU, Atomics, Memory Ordering) and L11 (Deadlock, Priority Inversion, PI Mutexes).*