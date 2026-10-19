# Publish a sample; reclaim its slot

**Time:** 45–60 minutes. **Requirements:** C++17 compiler, pthread-capable host, Python 3.10+ for the test driver. No GPU, elevated scheduling policy, or hardware driver is involved. This lab was compiled and executed on Linux with GCC 13.3.0.

Download [spsc_queue.h](spsc_queue.h), [spsc_demo.cpp](spsc_demo.cpp), and [test_spsc_queue.py](test_spsc_queue.py) into the same directory. From that directory:

```bash
g++ -std=c++17 -O2 -Wall -Wextra -Werror -pthread spsc_demo.cpp -o /tmp/course-spsc-demo
/tmp/course-spsc-demo
python3 -m unittest discover -s . -p 'test_spsc_queue.py' -v
```

The complete program checks 300,000 sequential operations against `std::deque`, then transfers 400,000 samples between real threads. Each sample contains sequence, payload, and checksum values. The JSON output includes the operation counts and whether the two index atomics report lock-free implementation on your host. There is no throughput or latency claim.

## 1. Start with ownership

Exactly one thread calls `push`, and exactly one calls `pop`. Their roles remain fixed. The queue itself must outlive both; the demo joins the producer before destroying the stack-allocated queue. It copies value objects, so no external payload pointer can dangle.

A queue with capacity three uses four storage slots. Initially `read=write=0`. Publishing three samples advances write to 3 while read remains 0. The next index would be 0, so a fourth push returns false without overwriting anything. After one pop advances read to 1, a push can use slot 3 and wrap write to 0.

**Try:** follow that sequence on paper and identify which slots are owned by each side. Equal indices mean empty, not full; reserving one slot removes the ambiguity. The tests use capacities 1, 3, and 17 to exercise frequent wrapping and avoid a hidden power-of-two assumption.

## 2. Follow both release/acquire pairs

The producer writes a sample, then release-stores the next write index. The consumer’s acquire load of that published index orders its subsequent sample copy after those writes.

The consumer copies a sample, then release-stores the next read index. The producer’s acquire load orders later reuse after that copy finishes. Omitting this second ownership transfer risks overwriting a slot while it is being read.

Each side reads its own index with relaxed ordering because it is the only writer of that index. The other side’s index needs the stated ordering. This reasoning applies to the exact SPSC algorithm; it is not a general rule that any two atomic indices make a thread-safe queue. See [Lecture 10](../Lectures/Lecture-10.md) for the memory-model relationship.

The implementation returns promptly on full or empty. The demonstration retries with `std::this_thread::yield()` and a bounded test deadline. Yielding is not a real-time scheduling policy, and retries are not an application-level loss policy. A robot must decide whether a full queue means backpressure, dropping an observation, replacing an old observation through a suitable different algorithm, or reporting overload.

## 3. Inspect the evidence

Three Python tests compile and execute three variants:

| Variant | Evidence |
|---|---|
| Optimized normal build | Reference FIFO agrees; all concurrent sequences/payloads/checksums agree |
| AddressSanitizer + UndefinedBehaviorSanitizer | Same exercise with those runtime checks enabled |
| Deliberately removed full check | Must fail with `full queue accepted overwrite` |

The sanitizer run checks the behavior those sanitizers instrument; it is **not** ThreadSanitizer, a proof of data-race freedom, or exhaustive scheduling exploration. A successful x86 run alone cannot validate release/acquire reasoning on all architectures. [Validation record](validation.json).

**Check:** changing both index stores to relaxed might still appear to work on one host. Do not use “the stress test passed” as evidence that such a change is valid. The correctness argument would have lost the required publication/reuse ordering even if that particular run did not expose it.

## 4. Add a camera buffer carefully

Passing a pointer instead of a `Sample` saves copying only the small descriptor. It does not solve ownership of the pointed-to allocation, DMA completion, a consumer retaining the pointer, reuse by the camera, or cache synchronization. A cross-process queue additionally needs a supported shared-memory synchronization contract; this ordinary C++ thread object is not automatically portable interprocess IPC.

For an extension, draw four states for each pool buffer: available, producer-owned, queued, consumer-owned. Add the exact transition that returns it to the pool. Mark the device completion event separately from CPU queue publication. Do not free the pool until all users and device operations are finished.

**Completion:** explain full/empty/wrap states, annotate both ordering pairs, retain the three test outcomes, and propose an explicit overload/lifetime policy for one robot sensor. Continue with [deadlocks and priority inversion](../Lectures/Lecture-11.md).
