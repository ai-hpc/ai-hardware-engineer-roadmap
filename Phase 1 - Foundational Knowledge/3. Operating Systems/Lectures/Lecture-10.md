# Lecture 10: Atomics, bounded queues, and memory reclamation

A camera producer and an inference consumer share two responsibilities: publish complete data and avoid reusing storage while the consumer still needs it. An atomic pointer or index solves only part of that problem. This lesson develops the ordering and lifetime contract before discussing Linux RCU.

**Prerequisites:** [mutexes and synchronization](Lecture-09.md), basic C/C++, and object lifetime. Plan 60–90 minutes including the [native SPSC queue lab](../labs/SPSC-Queue.md). The lab runs on an ordinary CPU; kernel RCU snippets below are reading exercises, not an executed driver.

## 1. Choose synchronization by the contract

A mutex can be the clearest correct solution. Measure contention and blocked time before replacing it. Uncontended locks need not cause a context switch, and atomics can still require cache-line ownership transfers or retries.

Distinguish these terms:

| Property | Meaning |
|---|---|
| Data-race freedom | Conflicting accesses have the required synchronization or are atomic |
| Lock-free progress | System-wide progress is guaranteed under the algorithm’s execution assumptions; one thread may starve |
| Wait-free progress | Each operation completes within a bounded number of its own steps |
| Bounded wall-clock latency | Completion meets a time bound under stated scheduling, hardware, and workload assumptions |

These are different properties. A lock-free operation can be delayed by scheduling or cache misses. `std::atomic<T>` is not guaranteed to be implemented without a library lock for every T and platform; inspect the supported type and `is_lock_free()` where relevant. A queue that returns `false` when full can finish its call promptly while the application that retries forever has no bounded completion guarantee.

## 2. Publish data with an ordering relationship

For ordinary non-atomic objects, unsynchronized conflicting accesses from different threads can constitute a C++ data race and undefined behavior. `volatile` does not establish inter-thread synchronization.

In this one-publication example, initialize `data` and `ready` before starting either thread:

```cpp
int data = 0;
std::atomic<bool> ready{false};
// Producer, exactly once:
data = 42;
ready.store(true, std::memory_order_release);
// Consumer:
if (ready.load(std::memory_order_acquire)) {
    assert(data == 42);
}
```

The acquire must observe the relevant release (or its applicable release sequence) on the same atomic object. An acquire load that still reads `false` does not mean the payload is ready. The producer must also avoid modifying `data` again while the consumer reads it. Publication and permission to reuse are separate relationships. [C++ memory ordering](https://eel.is/c++draft/atomics.order), [data races and happens-before](https://eel.is/c++draft/intro.races).

| Order | Useful role |
|---|---|
| Relaxed | Atomic access without publishing unrelated ordinary data |
| Release | Publish prior work through an atomic store/update |
| Acquire | Observe a matching publication before using its data |
| Acquire-release | Both roles on a suitable read-modify-write operation |
| Sequentially consistent | Adds an ordering requirement across participating sequentially consistent operations |

Sequential consistency is not a global stop-the-world operation or a promise that every access compiles to a full hardware fence. C++ specifies observable behavior; the compiler maps it to the target. Do not infer correctness from an x86 run or copy one architecture’s assembly into portable code.

## 3. Give a bounded ring both synchronization directions

The [complete C++17 lab](../labs/SPSC-Queue.md) stores small value objects and uses **exactly one producer and one consumer**. It reserves one array slot so equal indices mean empty. A storage array of five slots has usable capacity four.

Producer steps:

1. Read its own write index.
2. Acquire the consumer’s read index; if advancing would make the queue full, return `false`.
3. Copy the item into the owned slot.
4. Release-store the new write index to publish the item.

Consumer steps:

1. Read its own read index.
2. Acquire the producer’s write index; if equal, return `false` for empty.
3. Copy the item out of the published slot.
4. Release-store the new read index to permit reuse.

The second direction is essential. A producer that publishes correctly but never checks fullness can overwrite an unread sample. The regression suite deliberately removes that check and verifies that the reference catches it. A power-of-two capacity is not required by this implementation; it explicitly wraps bounded indices.

The queue’s non-atomic payload is safe because ownership and happens-before separate conflicting accesses. Its indices still involve cache-coherence traffic. Capacity, scheduling, copying, and memory layout affect throughput; “no mutex” does not mean “limited only by memory bandwidth.”

**Worked capacity check:** four slots can hold four pending items; a fifth push fails without changing them. After one pop, exactly one new push can succeed. A queue containing buffer pointers additionally needs a pool/lifetime protocol; the sample-value lab does not prove zero-copy DMA or cross-process synchronization.

## 4. Understand CAS and the lifetime problem

Compare-and-exchange changes a value only if it matches the supplied expectation:

```cpp
std::atomic<int> state{0};
int expected = 0;
bool changed = state.compare_exchange_strong(
    expected, 1, std::memory_order_acq_rel, std::memory_order_acquire);
```

On a mismatch, `expected` receives the observed value. Recompute a desired update from that current state when retrying. Weak compare-and-exchange can fail spuriously and is commonly used in a retry loop; choosing weak does not eliminate all other loads or guarantee faster execution. The failure order must be valid for the operation. [C++ compare-and-exchange](https://eel.is/c++draft/atomics.types.generic).

For pointers, a successful comparison does not prove that the object is still the same live object. An address can change from A to B and back to A, including through allocator reuse: the ABA problem. A generation tag can help if its width, wrap policy, atomic representation, and algorithm are correct. Tags are not a general proof of safe reclamation.

A hazard-pointer protocol publishes protection and revalidates that the protected pointer still matches the source before dereferencing it. Merely publishing once and then dereferencing leaves a race with reclamation. Use a reviewed implementation with its full ordering/reclamation protocol instead of copying a three-line sketch. Availability of a standard-library facility depends on the selected language/library implementation; this lab deliberately requires only C++17.

## 5. Read Linux RCU as a lifetime protocol

RCU is useful for read-mostly kernel structures. An updater removes an old version from publication and defers freeing it until the relevant pre-existing readers have completed. It still needs appropriate updater synchronization and object initialization.

```c
/* Kernel-reading sketch; definitions/error handling/updater locking omitted. */
rcu_read_lock();
obj = rcu_dereference(shared);
if (obj)
    consume_immutable_fields(obj);
rcu_read_unlock();
/* Do not retain obj here without a separate lifetime mechanism. */
```

Publish with the proper RCU API; reclaim through an appropriate grace-period/callback mechanism. `synchronize_rcu()` can block. Read-side cost and preemption behavior depend on the RCU flavor/configuration: ordinary RCU read sections can be preemptible, so a context switch is not universally proof that a reader has finished. SRCU supports a different sleepable-reader contract. RCU does not serialize writers or make arbitrary in-place updates coherent. [Linux RCU explanation and examples](https://docs.kernel.org/RCU/whatisRCU.html).

Replacing a model pointer also does not complete outstanding GPU operations. The device’s last use, buffer ownership, and destruction rules need separate evidence. A CPU read-side critical section is not a CUDA completion event.

## 6. Keep kernel and userspace rules separate

Linux `atomic_t`, `refcount_t`, spinlocks, `kfifo`, and RCU belong to the kernel’s APIs and memory model. Kernel refcounts should use the subsystem’s prescribed lifetime API rather than assuming any atomic counter is interchangeable. IRQ suitability depends on the full operation: allocation, retries, blocking, and the lock type all matter. [Kernel atomic types](https://docs.kernel.org/core-api/wrappers/atomic_t.html), [reference counting](https://docs.kernel.org/core-api/refcount-vs-atomic.html).

A kernel `kfifo` supports documented single-reader/single-writer use without an added lock, but callers must check full/empty return values and obey reset/concurrency restrictions. Additional readers or writers need the appropriate synchronization. [Kernel FIFO API](https://docs.kernel.org/core-api/kernel-api.html#fifo-buffer).

Offloading RCU callbacks can reduce interference on selected CPUs, but helper-thread affinity and other kernel work still matter. It does not establish a sub-millisecond worst-case bound. Use [CPU isolation](Lecture-08.md) and measured traces rather than treating a kernel command-line string as a guarantee.

## 7. Check the complete argument

The native lab runs 300,000 operations against a separate FIFO reference and transfers 400,000 items between real threads, checking order, payload, and checksum. It also runs AddressSanitizer/UndefinedBehaviorSanitizer and catches an intentional overwrite defect. Those checks do **not** prove the memory model or cover every interleaving; the ownership/order reasoning is essential.

**Self-checks:**

- Why does the producer acquire the read index? To observe that the consumer finished copying before reusing that slot.
- Does `volatile bool ready` publish a C++ object? No; it is not the required inter-thread synchronization.
- Does an atomic pointer keep its pointee alive? No; publication and reclamation are separate.
- Can a lock-free algorithm miss a deadline? Yes; progress properties are not wall-clock scheduling guarantees.

**Completion:** annotate both queue ownership transfers, demonstrate full/empty/wrap behavior, retain native test output, and explain how a camera-buffer pool would add lifetime/device-completion requirements. Continue to [deadlock and priority inversion](Lecture-11.md).
