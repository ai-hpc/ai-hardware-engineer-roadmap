# Consolidation 5: Synchronization is ordering plus lifetime plus progress

Use this note after [Lecture 10](Lecture-10.md) and [Lecture 11](Lecture-11.md). It is a retrieval exercise: predict the answer before opening the explanation. The complete executable reference is the [native SPSC queue lab](../labs/SPSC-Queue.md).

## 1. Name what the atomic protects

A camera thread writes a descriptor, then stores `ready=true`. The inference thread sees true and reads the descriptor.

**Question:** is an atomic ready flag enough?

**Answer:** only with the required ordering and lifetime protocol. In a one-publication example, a release store observed by an acquire load orders prior payload writes before subsequent reads. If the producer rewrites the payload while the consumer still reads it, publication alone is insufficient. A relaxed flag does not publish unrelated ordinary data; `volatile` is not C++ thread synchronization.

**Transfer:** add a permission-to-reuse flag or a bounded queue. Mark both ownership directions and explain how the second direction prevents overwrite. The lab uses release/acquire on both read and write indices.

## 2. Distinguish full from empty

A capacity-three ring uses four storage slots. Begin with read=write=0, push three values, reject the fourth, pop one, then push another.

**Answer:** write reaches 3; its next value 0 equals read, so the extra push must fail. After read advances to 1, writing slot 3 and wrapping write to 0 is permitted. This implementation does not require a power-of-two capacity. Removing the full check is a real overwrite defect caught by the mutation test.

**Transfer:** a second producer invalidates the single-writer assumption even if each index remains atomic. Select a reviewed multi-producer algorithm or introduce appropriate synchronization; do not merely change the label to MPSC.

## 3. Keep progress distinct from timing

**Question:** can a lock-free queue miss a 10 ms deadline?

**Answer:** yes. Lock-free is a progress property under algorithmic execution assumptions, not a bound on scheduling delay, cache misses, retries by a particular thread, or downstream device work. A failed `push` can return quickly while the caller retries indefinitely. Even an atomic type’s implementation may not be lock-free on every target.

**Transfer:** choose an overload policy: bounded waiting, rejected work, or a separately designed latest-value channel. Define which observation sequences are lost and how a consumer detects stale data.

## 4. A pointer is not a lifetime guarantee

**Question:** a thread atomically loads pointer P; another thread replaces P and frees the old object. May the first thread dereference its copy?

**Answer:** not without a mechanism preserving the object’s lifetime. Atomic pointer access does not keep the pointee alive. A reference count, reviewed hazard-pointer protocol, suitable RCU flavor, or lock may provide the required contract. Device operations retaining a pointer require their own completion/lifetime handling.

Hazard-pointer protection includes publishing and revalidating the pointer. RCU defers reclamation until relevant pre-existing readers finish; implementation/preemption rules vary by flavor. Neither mechanism automatically serializes competing updaters or repairs arbitrary in-place mutation. Read the primary references in Lecture 10 instead of implementing a reclamation algorithm from this short note.

## 5. Separate deadlock from priority inversion

L holds a mutex needed by H. M preempts L. H cannot proceed.

**Answer:** this is priority inversion if execution of unrelated M extends H’s wait through the low-priority holder. Priority inheritance can raise L’s effective priority while H blocks. M remains runnable; it is not itself blocked on the mutex. PI does not eliminate L’s remaining critical-section work.

Now let L also wait for a lock held by H. That circular dependency is a deadlock. Priority boosting cannot make either owner release a lock while both are waiting. Establish a consistent lock ordering and include every nested path.

**Transfer:** a failed trylock followed by retry while retaining another required lock is not a universal deadlock solution. An acquire/release/retry protocol can avoid one condition but still livelock or starve without a suitable policy.

## 6. Account for bounded blocking

H needs 1.2 ms of CPU work and may wait for 0.6 ms of a lower-priority critical section. Its deadline is 2 ms.

**Answer:** 1.8 ms is already committed before interrupt, scheduling, higher-priority, and other interference. Priority ceiling can control blocking under its assumptions, but does not mean H never waits. An additional 1 ms of interference makes this instance exceed its deadline.

**Completion:** retain the three native queue test outcomes, draw both ownership transfers, and explain one example each of data race, lifetime failure, deadlock, and deadline miss. A passing stress test does not replace the memory-order proof; a lockdep-clean run does not establish timing bounds. Continue with [memory allocation](Lecture-14.md) and [DMA ownership](Lecture-15.md).
