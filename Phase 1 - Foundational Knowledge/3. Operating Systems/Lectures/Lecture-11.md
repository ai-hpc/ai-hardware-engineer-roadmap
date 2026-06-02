# Lecture 11: Deadlock, Priority Inversion & PI Mutexes

## Overview

The core problem this lecture addresses is: what happens when the synchronization primitives from the previous lectures are used incorrectly, or when correct usage leads to emergent timing failures? Two failure modes dominate: deadlock (tasks waiting for each other forever) and priority inversion (a high-priority task stalled by a low-priority one). The mental model to carry here is that of a traffic deadlock at a four-way intersection where each car is blocking the path of the next — no one can move. Priority inversion is subtler: imagine the most important person in a building being unable to get to their meeting because a janitor locked the conference room and then got stuck behind a slow colleague. For an AI hardware engineer, these are not theoretical problems: the Mars Pathfinder rover was reset by priority inversion in 1997, and AV systems share the same architectural patterns that caused it.

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

> **Key Insight:** The Coffman conditions are a checklist for designing deadlock-free systems. Before writing code that acquires multiple locks, check each condition: Can you make resources shareable? Can you acquire all locks upfront? Can you use trylock? Can you define a global lock ordering? Each "yes" removes one condition and eliminates deadlock.

---

## Deadlock Prevention

Attack one of the four conditions at design time:

| Condition to Break | Technique | Trade-off |
|---|---|---|
| Hold-and-wait | Acquire all locks simultaneously before starting; release all if any acquisition fails | Reduces concurrency; may require retry |
| Circular wait | Global lock ordering: always acquire locks in a fixed, documented order | Requires discipline across all call sites; lockdep enforces this |
| No preemption | `mutex_trylock()` with randomized exponential backoff | Retry overhead; potential livelock |
| Mutual exclusion | Use lock-free data structures | Higher implementation complexity |

**Global lock ordering** is the most practical approach in kernel and embedded code. Linux documents lock ordering in code comments and enforces it at runtime with lockdep annotations (`lockdep_assert_held`, `lockdep_set_class`).

Step-by-step: how to implement global lock ordering:

1. Enumerate all mutexes/locks in your subsystem.
2. Assign each a numeric rank (e.g., Lock A = rank 1, Lock B = rank 2).
3. Mandate that locks are always acquired in ascending rank order.
4. Document this ordering in a comment at the lock declaration site.
5. Use `lockdep_set_class` to assign each lock to a class so lockdep can verify the ordering automatically.
6. In code review, reject any patch that acquires a lower-ranked lock while holding a higher-ranked one.

> **Common Pitfall:** Global lock ordering breaks silently when two independent developers add new locks without consulting the existing ordering. One developer adds Lock C acquired while holding Lock A; another adds Lock D acquired while holding Lock C. Neither knows that Lock D is also acquired while holding Lock A elsewhere, creating an A→C→D cycle. Use lockdep in CI to catch this automatically.

---

## Deadlock Detection

When prevention is too costly, the OS can detect deadlocks at runtime:

- **Resource allocation graph**: nodes for processes (P) and resources (R); edge P→R means P waits for R; edge R→P means R is held by P; a cycle indicates deadlock
- **Wait-for graph**: simplified form; edge P→Q means P waits for something held by Q; cycle = deadlock

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

Effective priority of `H` is **inverted to below `M`'s level** for the entire duration of `M`'s execution. Duration of inversion is **unbounded** without PI.

> **Key Insight:** Priority inversion does not require any bug in the individual tasks. L, M, and H are all behaving correctly according to their own logic. The failure is systemic: the interaction of correct behaviors produces an emergent incorrect outcome. This is why it is so dangerous — code review alone cannot catch it.

---

## Mars Pathfinder Case Study (1997)

The Mars Pathfinder rover experienced **periodic system resets** approximately 18 hours after landing on the Martian surface. This case study is **required reading** in every safety-critical system design course because the bug was discovered 140 million km away.

**Root cause: unbounded priority inversion**

| Task | Priority | Role |
|---|---|---|
| `bc_dist` | High | Data distribution bus — held the critical mutex |
| `bc_sched` | Low | Bus scheduler — also needed the mutex to release it |
| `ASI_MET` | Medium | Meteorological data acquisition — CPU-intensive |

Sequence: `bc_sched` (low) held the mutex → `bc_dist` (high) blocked waiting → `ASI_MET` (medium) preempted `bc_sched` and ran continuously → `bc_sched` never ran → `bc_dist` missed its watchdog deadline → VxWorks watchdog timer fired → full system reset.

```
  Mars Pathfinder Priority Inversion:

  bc_dist (HIGH):   [woken]──[BLOCKED waiting for mutex]────────────[WATCHDOG RESET]
  ASI_MET (MED):    ──────────────────────[RUNNING]────────────────►
  bc_sched (LOW):   [holds mutex]──[PREEMPTED]─────────────────────[never resumes]
                                   ▲
                              preempted by ASI_MET here
```

**Fix**: enable the `PTHREAD_PRIO_INHERIT` flag on the shared VxWorks mutex. The PI feature existed in VxWorks but was disabled by default. The fix was uploaded to the already-landed rover via uplink command.

**Lesson for AV/robotics**: priority inversion can cause safety-critical resets in deployed, inaccessible hardware. PI mutexes must be the default for any resource shared between tasks of different priorities in safety-critical systems.

---

## Priority Inheritance (PI)

When high-priority task `H` blocks on a mutex held by low-priority task `L`, the kernel **temporarily boosts** `L`'s scheduling priority to `H`'s level. The boost is removed when `L` releases the mutex.

- **Transitive PI**: if `L` is also blocked on another mutex held by `X`, the boost propagates to `X` along the entire blocking chain
- Linux `rtmutex` implements PI for kernel code and userspace (via `FUTEX_LOCK_PI`)
- PREEMPT_RT converts most kernel `spinlock_t` to `rtmutex` → system-wide PI without changing driver code

```
  Priority Inheritance in Action:

  BEFORE PI:                          AFTER PI ENABLED:
  H (pri 90): BLOCKED                 H (pri 90): BLOCKED
  M (pri 50): RUNNING                 M (pri 50): BLOCKED (cannot preempt L anymore)
  L (pri 10): PREEMPTED by M          L (pri 10 → boosted to 90): RUNNING
                                                                   releases mutex
                                                                   L returns to pri 10
                                      H (pri 90): UNBLOCKED, runs
```

```c
// POSIX userspace PI mutex
pthread_mutexattr_t attr;
pthread_mutexattr_init(&attr);
pthread_mutexattr_setprotocol(&attr, PTHREAD_PRIO_INHERIT);
// PTHREAD_PRIO_INHERIT: boost holder to waiter's priority when blocked
pthread_mutex_init(&mutex, &attr);
// Requires SCHED_FIFO/SCHED_RR threads for PI to have meaningful effect
// On SCHED_OTHER (CFS), priorities are not strictly enforced and PI has limited benefit
```

> **Common Pitfall:** Enabling PI mutexes without using `SCHED_FIFO` or `SCHED_RR` threads is a common mistake. PI boosts the holder's priority, but if all threads use `SCHED_OTHER` (CFS), the concept of "priority" is relative and the boost may not have the desired effect. PI is most effective in real-time scheduling contexts.

---

## Priority Ceiling Protocol

Each mutex is assigned a **ceiling priority** = highest priority of any task that will ever acquire it. Any task acquiring the mutex **immediately runs at the ceiling priority** for the duration of ownership.

- Prevents priority inversion entirely without needing runtime priority discovery
- Requires static analysis: all potential lock acquirers and their priorities must be known at design time
- POSIX: `PTHREAD_PRIO_PROTECT` protocol (`pthread_mutexattr_setprotocol`)
- Mandated by: ARINC 653 (avionics partitioned RTOS), AUTOSAR OS (automotive ECUs)
- Stronger formal guarantees than PI for safety-certified systems where the task model is fixed at configuration time

```
  Priority Ceiling vs Priority Inheritance:

  PI: ceiling is discovered dynamically at runtime when H blocks
      Overhead: runtime priority boost when contention occurs
      Suitable for: dynamic task models, general POSIX RT

  Priority Ceiling: ceiling is set statically at mutex creation time
      Overhead: every acquisition raises priority to ceiling (even without contention)
      Suitable for: AUTOSAR ECUs, ARINC 653 where task set is fixed
      Stronger guarantee: L's priority is raised BEFORE H blocks, so H never blocks at all
```

The transition from understanding deadlock prevention to understanding priority inversion is natural: both problems arise from incorrect interactions between multiple locks and multiple tasks. The difference is that deadlock is a liveness failure (no progress) while priority inversion is a timing failure (correct result, wrong time).

---

## lockdep — Live Deadlock Detector

`CONFIG_PROVE_LOCKING` enables lockdep, the kernel's runtime lock dependency graph validator:

```bash
CONFIG_PROVE_LOCKING=y
CONFIG_LOCK_STAT=y      # adds per-lock contention statistics
```

- Assigns each lock a **lock class** based on its static address in the kernel binary
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

Enable during all development, CI, and regression testing. `lockdep` adds ~10% runtime overhead; disable in production or performance-sensitive contexts.

`lockdep_assert_held(&lock)`: assertion that documents and verifies that a lock is held at a given point; useful for complex drivers with multi-step lock protocols.

> **Key Insight:** lockdep's power is that it detects a *potential* deadlock the first time a new lock ordering path is taken — even if that exact thread interleaving has never caused an actual hang. It builds a global dependency graph and reports cycles in that graph immediately. This is far more reliable than waiting for the rare timing conditions that cause actual deadlocks in production.

---

## Watchdog Timers as Last Resort

Hardware or software **watchdog**: if a process does not **kick the watchdog** within the timeout period, the system resets or enters a safe state. Provides **defense-in-depth** against deadlocks that escape lockdep in deployed hardware.

The Mars Pathfinder watchdog worked correctly; it correctly detected `bc_dist` missing its deadline. The root problem was the missing PI configuration that caused `bc_dist` to miss the deadline in the first place.

> **Common Pitfall:** Treating watchdog resets as an acceptable recovery mechanism for priority inversion is wrong. A watchdog reset on a vehicle's ADAS controller causes a controlled disengagement — the driver must take over. In the worst case, the reset occurs at a moment when the driver has no time to react. The correct fix is to eliminate the inversion, not to rely on the watchdog.

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
- **What is the single most important lesson of Mars Pathfinder?** Priority inheritance was available in VxWorks but disabled by default. Safety-critical systems must audit every mutex that is shared between tasks of different priorities and ensure PI is enabled. "The feature exists" is not the same as "the feature is enabled."
- **How does priority inheritance differ from priority ceiling?** PI boosts the lock holder's priority dynamically when a higher-priority waiter blocks. Priority ceiling boosts the acquirer to the ceiling immediately, even if no high-priority task is waiting. PI has lower average overhead; priority ceiling has stronger predictability guarantees.
- **What does lockdep report and what does it not report?** lockdep reports potential deadlock cycles (AB-BA orderings) and lock-context violations (sleeping lock in IRQ context). It does not report priority inversion — that requires separate tools like `cyclictest` or real-time scheduling analysis.
- **Why is transitive PI important?** If H blocks on a mutex held by L, and L blocks on a different mutex held by X, then X also needs to be boosted — otherwise L cannot run, cannot release its mutex to unblock H. Transitive PI propagates the boost along the entire blocking chain.
- **When is priority ceiling mandatory instead of just recommended?** In AUTOSAR OS and ARINC 653 environments where the complete task set and their resource usage is fixed at configuration time and formally verified. These standards require formal proof of bounded priority inversion, which priority ceiling provides statically.

---

## AI Hardware Connection

- PI mutex (`rtmutex` / `PTHREAD_PRIO_INHERIT`) is required for openpilot `controlsd` (highest priority) sharing cereal shared-memory state with `plannerd` (medium) and log-writing processes (low); without PI, the Mars Pathfinder scenario can recur in a production AV system
- The Mars Pathfinder root cause analysis is a mandatory reference document in ASIL-B safety reviews; it demonstrates that priority inversion can cause safety-critical resets in deployed, inaccessible hardware with zero possibility of manual intervention
- `PTHREAD_PRIO_INHERIT` is used in ROS2 real-time executor configurations to prevent control callbacks from being starved by data-logging threads; the rclcpp real-time executor documentation explicitly references this requirement
- `lockdep` is standard practice during embedded Linux driver development on Jetson and i.MX platforms; AB-BA cycles in camera, ISP, and DMA subsystem locks must be eliminated before hardware deployment
- Priority ceiling protocol is used in AUTOSAR-compliant ECU software where all task priorities and shared resource sets are fixed at configuration time; this is stronger than PI for ASIL-D certified safety functions
- PREEMPT_RT's system-wide conversion of `spinlock_t` to `rtmutex` means that PI is automatically applied to all kernel synchronization on the CAN bus driver, V4L2 camera driver, and GPU driver without any driver code changes
