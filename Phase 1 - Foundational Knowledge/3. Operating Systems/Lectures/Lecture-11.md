# Lecture 11: Deadlock, Priority Inversion & PI Mutexes

## Overview

The core problem this lecture addresses is: what happens when the synchronization primitives from the previous lectures are used incorrectly, or when correct usage leads to emergent timing failures? Two failure modes dominate: deadlock (tasks waiting for each other forever) and priority inversion (a high-priority task stalled by a low-priority one). The mental model to carry here is that of a traffic deadlock at a four-way intersection where each car is blocking the path of the next — no one can move. Priority inversion is subtler: imagine the most important person in a building being unable to get to their meeting because a janitor locked the conference room and then got stuck behind a slow colleague. For an AI hardware engineer, these are not theoretical problems: the Mars Pathfinder spacecraft suffered resets involving priority inversion in 1997, and AV systems share the same architectural patterns that caused it.

---

## Deadlock: Definition

A **deadlock** is a state where a set of processes are each waiting for a resource held by another process in the set, and no process can ever make progress.

```
  Deadlock: Circular Wait on Two Resources

  Process A                   Process B
  ─────────────               ─────────────
  holds Lock L1               holds Lock L2
  waiting for Lock L2 ──────► (held by B)
  (held by A) ◄────────────── waiting for Lock L1

  Neither can proceed. Both wait forever.
```

---

## Coffman Conditions

**All four conditions must hold simultaneously** for a deadlock to be possible. **Breaking any single one prevents deadlock.**

| Condition | Definition |
|---|---|
| Mutual exclusion | At least one resource is non-sharable — only one process may hold it at a time |
| Hold-and-wait | A process holds at least one resource while waiting to acquire additional resources |
| No preemption of resources | Resources cannot be forcibly taken from a holder; only voluntary release |
| Circular wait | A cycle exists in the resource-allocation graph: P1 → R1 → P2 → R2 → P1 |

> **Key Insight:** The Coffman conditions are a checklist for designing deadlock-free systems. Before writing code that acquires multiple locks, check each condition: Can you make resources shareable? Can you acquire all locks upfront? Can you use trylock? Can you define a global lock ordering? The complete protocol must actually break a condition. Merely calling trylock does not release resources already held and can produce livelock if retries are poorly coordinated.

---

## Deadlock Prevention

Attack one of the four conditions at design time:

| Condition to Break | Technique | Trade-off |
|---|---|---|
| Hold-and-wait | Acquire all locks simultaneously before starting; release all if any acquisition fails | Reduces concurrency; may require retry |
| Circular wait | Acquire all nested locks in a consistent order | Cover every call path; lockdep can diagnose observed dependency cycles |
| Hold-and-wait | On a failed trylock, release acquired locks and retry under an explicit policy | No forced ownership transfer; possible livelock/starvation |
| Mutual exclusion where unnecessary | Share immutable snapshots or redesign resource ownership | Requires a valid lifetime/update protocol |

**Global lock ordering** is the most practical approach in kernel and embedded code. Linux documents lock ordering and uses lockdep to validate observed dependency paths. Lockdep diagnoses violations; it does not reorder lock acquisitions or make an incorrect program correct.

Step-by-step: how to implement global lock ordering:

1. Enumerate all mutexes/locks in your subsystem.
2. Assign each a numeric rank (e.g., Lock A = rank 1, Lock B = rank 2).
3. Mandate that locks are always acquired in ascending rank order.
4. Document this ordering in a comment at the lock declaration site.
5. Use appropriate lock classes/subclasses and assertions where needed. A lock class is not a numeric rank; lockdep learns dependencies from instrumented execution.
6. In code review, reject any patch that acquires a lower-ranked lock while holding a higher-ranked one.

> **Common Pitfall:** Global lock ordering breaks silently when two independent developers add new locks without consulting the existing ordering. One developer adds Lock C acquired while holding Lock A; another adds Lock D acquired while holding Lock C. Neither notices a third path acquiring A while holding D, creating A→C→D→A. Exercise the relevant paths under a lockdep-enabled test kernel; unexecuted dependencies may remain unseen.

---

## Deadlock Detection

When prevention is too costly, the OS can detect deadlocks at runtime:

- **Resource allocation graph**: nodes for processes (P) and resources (R); edge P→R means P waits for R; edge R→P means R is held by P; for single-instance exclusive resources, a cycle establishes deadlock; multiple instances need a fuller availability analysis
- **Wait-for graph**: simplified form; edge P→Q means P waits for something held by Q; a cycle can establish deadlock under the single-instance exclusive-wait model

```
  Resource Allocation Graph:

  P1 ──► R1 ──► P2
  ▲             │
  └──── R2 ◄────┘

  P1 holds R2, waits for R1
  P2 holds R1, waits for R2
  Cycle P1→R1→P2→R2→P1 = DEADLOCK
```

Recovery options after detection:
1. Abort one process; free its resources; select victim by cost (priority, runtime, resources held)
2. Preempt a resource from one process and give to another (requires rollback support)
3. Roll back a process to a safe checkpoint (requires checkpointing infrastructure)

`lockdep` in Linux performs static-analysis-style deadlock detection at runtime without waiting for the deadlock to actually occur.

---

## Priority Inversion

**Priority inversion** occurs when a high-priority task `H` is **effectively blocked by a medium-priority task** `M` through an indirect chain involving a **shared resource**.

### Mechanism

The inversion unfolds in four steps:

1. `L` (low priority) acquires mutex `M`
2. `H` (high priority) wakes and tries to acquire mutex `M` — blocks waiting for `L`
3. `M` (medium priority) wakes and preempts `L` (`M` > `L` in priority)
4. `L` cannot run to release mutex `M`; `H` waits indefinitely despite being highest priority

```
  Priority Inversion Timeline:

  Time ──────────────────────────────────────────────────────────►

  Task H (high):  [woken]──[BLOCKED on mutex held by L]──────────────────────────►
  Task M (med):           ──────────────────────[RUNNING freely]──────────────────►
  Task L (low):   [holds mutex]──[PREEMPTED by M]────────────────[never runs]

  H is effectively running at L's priority — INVERTED
  Duration of inversion: unbounded (as long as M runs)
```

H retains its configured priority but cannot run past the lock acquisition. Unrelated medium-priority execution can extend its wait without a bound derived from L's critical section alone.

> **Key Insight:** Priority inversion does not require any bug in the individual tasks. L, M, and H are all behaving correctly according to their own logic. The failure is systemic: the interaction of correct behaviors produces an emergent incorrect outcome. This is why it is so dangerous — code review alone cannot catch it.

---

## Mars Pathfinder: use the first-person account

Flight-software lead Glenn Reeves described a low-priority ASI/MET task holding a mutex within the select/pipe IPC implementation. Higher-priority `bc_dist` blocked on that resource while medium-priority work delayed the holder. The still-higher-priority `bc_sched` detected that distribution had missed its cycle and initiated recovery. The team reproduced the failure in the laboratory and changed the semaphore creation configuration to enable priority inheritance. See the [preserved first-person account by Glenn Reeves](https://huichen-cs.github.io/course/CISC7310X/26FA/lecture/MarsPathfinder.htm).

Do not reverse those priorities or describe the fix as setting a POSIX `PTHREAD_PRIO_INHERIT` flag on VxWorks. The useful lesson is to include library-internal locks in the timing model and preserve diagnostic/recovery capabilities. This historical example is not a mandatory named requirement in every automotive safety standard.

## Priority Inheritance (PI)

When high-priority task `H` blocks on a mutex held by low-priority task `L`, the kernel **temporarily boosts** `L`'s scheduling priority to `H`'s level. The boost is removed when `L` releases the mutex.

- **Transitive PI**: if `L` is also blocked on another mutex held by `X`, the boost propagates to `X` along the entire blocking chain
- Linux `rtmutex` implements PI for kernel code and userspace (via `FUTEX_LOCK_PI`)
- PREEMPT_RT changes `spinlock_t` semantics to an RT-mutex-based implementation; `raw_spinlock_t` remains a spinning primitive. Driver context and nesting rules still require review. [Kernel lock types](https://docs.kernel.org/locking/locktypes.html).

```
  Priority Inheritance in Action:

  BEFORE PI:                          AFTER PI ENABLED:
  H (pri 90): BLOCKED                 H (pri 90): BLOCKED
  M (pri 50): RUNNING                 M (pri 50): READY (does not preempt boosted L)
  L (pri 10): PREEMPTED by M          L (pri 10 → boosted to 90): RUNNING
                                                                   releases mutex
                                                                   L returns to pri 10
                                      H (pri 90): UNBLOCKED, runs
```

Reading fragment showing the setup API names; a complete program must branch on each returned error number.

```c
// POSIX userspace PI mutex
pthread_mutexattr_t attr;
pthread_mutexattr_init(&attr);
pthread_mutexattr_setprotocol(&attr, PTHREAD_PRIO_INHERIT);
// PTHREAD_PRIO_INHERIT: boost holder to waiter's priority when blocked
pthread_mutex_init(&mutex, &attr);
// Check every return code and destroy the attributes when no longer needed.
// This is a setup excerpt; it does not demonstrate a measured PI experiment.
```

PI behavior and useful priority donation depend on the actual scheduling policies, implementation, and protocol support. Check initialization errors and measure the intended mixed-priority workload. Enabling PI does not itself choose task priorities or establish a response-time bound.

---

## Priority ceiling and bounded blocking

An immediate priority-ceiling protocol raises a task to the configured resource ceiling while it owns that resource. The ceiling must account for every task permitted to use it. POSIX exposes a priority-protection protocol, but platform support, valid priorities, return codes, scheduling policy, and the exact protocol must be checked. Do not assume that every Linux mutex implementation supports every POSIX option. [POSIX mutex protocol](https://pubs.opengroup.org/onlinepubs/9799919799/functions/pthread_mutexattr_getprotocol.html).

This does not eliminate blocking. Consider a single-CPU fictional system: L acquires a resource, then H becomes runnable while L has 0.6 ms of protected work left. H cannot use that resource until L releases it. Under an applicable protocol, unrelated medium-priority work should not extend that interval arbitrarily; H still experiences the remaining critical section and relevant scheduler/interference costs.

**Worked response-time accounting:** H needs 1.2 ms of its own CPU work, may block for 0.6 ms on L, and has a 2 ms deadline. The sum is already 1.8 ms before higher-priority interference, interrupts, release jitter, and overhead. A measured 1 ms extra interference would make this instance miss its deadline. PI or a ceiling is not permission to omit blocking from the analysis.

Priority inheritance propagates inherited urgency along applicable blocking chains, but it does not break a deadlock, shorten the protected algorithm, complete a device operation, or cure a holder that never releases. The timing policy needs bounded critical sections, valid lock ordering, and a defined recovery path.

---

## lockdep — Live Deadlock Detector

`CONFIG_PROVE_LOCKING` enables lockdep, the kernel's runtime lock dependency graph validator:

```bash
CONFIG_PROVE_LOCKING=y
CONFIG_LOCK_STAT=y      # adds per-lock contention statistics
```

- Tracks lock classes and their dependency/context relationships; initialization and explicit class/subclass annotations affect that classification
- Records every lock acquisition chain: "lock A was held when lock B was acquired"
- Reports AB-BA cycles at first detection — before they manifest as actual hangs
- Reports invalid lock-from-interrupt context: e.g., `mutex_lock()` called from hardirq context

```
WARNING: possible circular locking dependency detected
task/1234 is trying to acquire lock:
  (&lockB){...}, at: function_b+0x30
but task holds lock:
  (&lockA){...}, taken at: function_a+0x20
which lock already depends on the new lock.
```

Enable the relevant debug configuration in development/test kernels when practical. Its runtime/memory overhead depends on the workload and configuration; measure it rather than assuming a universal percentage.

`lockdep_assert_held(&lock)`: assertion that documents and verifies that a lock is held at a given point; useful for complex drivers with multi-step lock protocols.

> **Key Insight:** lockdep's power is that it detects a *potential* deadlock the first time a new lock ordering path is taken — even if that exact thread interleaving has never caused an actual hang. It builds a global dependency graph and reports cycles in that graph immediately. This is far more reliable than waiting for the rare timing conditions that cause actual deadlocks in production.

---

## Watchdog Timers as Last Resort

Hardware or software **watchdog**: if a process does not **kick the watchdog** within the timeout period, the system resets or enters a safe state. Provides **defense-in-depth** against deadlocks that escape lockdep in deployed hardware.

In the Pathfinder account, cycle-completion checking detected the missed distribution work and invoked recovery. A watchdog/reset mechanism can help recover availability, but the resulting physical behavior depends on the surrounding system.

A reset does not automatically produce a safe physical state or a controlled disengagement. Define the independent actuator/output behavior, fault notification, restart state, and recovery conditions. Investigate the root cause rather than counting successful reboots as proof of correct timing.

---

## Summary

| Problem | Symptom | Detection Tool | Solution |
|---|---|---|---|
| Deadlock | System hangs; tasks blocked forever | Resource allocation graph; lockdep | Lock ordering; acquire-all-at-once; `trylock` + backoff |
| Priority inversion | High-priority task stalls unexpectedly | Latency monitoring; watchdog timeout | PI mutex (`rtmutex`, `PTHREAD_PRIO_INHERIT`); priority ceiling |
| Livelock | Tasks run but make no progress | CPU profiling (100% util, no throughput) | Randomized backoff; central arbitration |
| Starvation | Low-priority task never runs | Long wait monitoring; `perf sched` | Aging; FIFO wait queues; priority boosting |

### Conceptual Review

- **Why can priority inversion occur even when every individual task is behaving correctly?** Inversion is an emergent property of task interactions, not a bug in any one task. L holds a mutex correctly. H waits for it correctly. M preempts L correctly. The problem is that no one designed the system to prevent the indirect blocking chain L→H via the mutex.
- **What is the single most important lesson of Mars Pathfinder?** Priority inheritance was available in VxWorks but disabled by default. Audit the actual shared-resource protocol and bounded blocking, including library-internal synchronization. PI is one applicable mechanism, not a universal substitute for ownership design. "The feature exists" is not the same as "the feature is enabled."
- **How does priority inheritance differ from immediate priority ceiling?** PI raises effective priority in response to a blocked higher-priority waiter; an immediate-ceiling protocol raises it when the resource is acquired. Both still require bounded critical sections and the protocol’s scheduling/nesting assumptions.
- **What does lockdep report and what does it not report?** lockdep reports potential deadlock cycles (AB-BA orderings) and lock-context violations (sleeping lock in IRQ context). It does not report priority inversion — that requires separate tools like `cyclictest` or real-time scheduling analysis.
- **Why is transitive PI important?** If H blocks on a mutex held by L, and L blocks on a different mutex held by X, then X also needs to be boosted — otherwise L cannot run, cannot release its mutex to unblock H. Transitive PI propagates the boost along the entire blocking chain.
- **Does a ceiling mean the high-priority task never waits?** No. A low-priority holder may already own the resource and run at the ceiling before the high-priority task is released. That task still waits for the remaining critical section. The purpose is a controlled blocking bound under the protocol’s assumptions.

---

## Apply the analysis to a robot

List the resources shared by the sensor, inference, logging, and control tasks. For each, record the owner, maximum critical-section work, possible blocking/device waits, task priorities, and ordering relative to other locks. Do not assume a particular openpilot or ROS 2 process uses a PI mutex without examining its pinned source.

PREEMPT_RT changes many lock semantics, but raw locks, device waits, callback lifetime, and driver-specific critical sections still require analysis. A lock-free queue avoids one mutex but introduces its own capacity, ownership, and progress assumptions; compare the tested [SPSC queue](../labs/SPSC-Queue.md).

**Check:** H needs 1.2 ms CPU, can block 0.6 ms, and suffers 0.4 ms additional interference. Does a 2 ms deadline hold? **Answer:** No: response demand is already 2.2 ms under these additive, non-overlapping assumptions. Turning on PI does not erase the 0.6 ms protected work.
