# RT demo — user-space lab (Lectures 1–9)

Small **Linux** program that illustrates ideas from the first nine OS lectures: processes and threads, syscalls, scheduling (`SCHED_FIFO`), CPU affinity, memory locking / pre-faulting, and pthread synchronization.

Kernel-side topics (interrupts, boot, modules, device tree) are only described in comments or below—not implemented here.

## Build and run

From the repository root, enter this directory:

```bash
cd "Phase 1 - Foundational Knowledge/3. Operating Systems/Lectures/demos/rt-demo"
make
./rt_demo
```

Without `make`: `gcc -std=c17 -Wall -Wextra -Werror -O2 -pthread -o rt_demo rt_demo.c`

**Options:** `./rt_demo --help`. Choose `--cpu` values from your actual allowed mask (`taskset -pc $$`), not an assumed CPU number. Invalid syntax, a CPU outside that mask, and failed setup requests exit nonzero. `--rt` requests FIFO priority 80 and `--lock-memory` requests memory locking; permissions and resource limits determine success. `--no-rt` clears earlier RT/locking options, so option order is explicit. The default requests neither.

**Optional documented trace:** `strace -e sched_setscheduler,sched_setaffinity,mlockall,gettid,clock_gettime ./rt_demo`. `clock_gettime` can use the vDSO without a traced syscall. Trace overhead changes the experiment; this is not a latency measurement.

## Lecture mapping

| Lecture | Topic | In this demo |
|--------|--------|----------------|
| **L1** | OS as resource manager; user vs kernel | Runs in user space; uses syscalls to request kernel services |
| **L2** | Process / threads | `main()` + worker thread; `getpid()`, `gettid()`, affinity |
| **L3** | Interrupts, top/bottom half | Event-style wakeup only; real IRQs are kernel/driver |
| **L4** | System calls | `sched_setscheduler`, `sched_setaffinity`, `mlockall`, `clock_gettime`, … |
| **L5** | Boot, modules, device tree | Not in code; see **L5 on a real machine** below |
| **L6** | Scheduling | `sched_setscheduler(SCHED_FIFO)` |
| **L7** | RT Linux, latency | `mlockall()`, pre-faulting, no alloc in hot loop |
| **L8** | Affinity & isolation | `sched_setaffinity()` |
| **L9** | Synchronization | `pthread_mutex_t`, `pthread_rwlock_t`, `pthread_cond_t` |

### L5 on a real machine

- **Boot:** `dmesg` at boot; `systemd-analyze`
- **Modules:** `lsmod`, `modinfo` for read-only inventory; loading a module is a separate target-specific operation
- **Device tree (ARM):** `/sys/firmware/devicetree/base/` or machine model in boot logs

## Requirements

Linux or WSL with pthreads and `sched.h`. `--rt` / `--lock-memory` depend on configured resource limits and capabilities; neither is required for the default exercise.

## Files

| File | Purpose |
|------|---------|
| `rt_demo.c` | Source |
| `Makefile` | Build |
| `README.md` | This file |

Back to the section hub: **[Operating Systems Guide](../../../Guide.md)**.

## Distinguish wakeup latency from real-time guarantees

Run the demo first with its default non-real-time policy on a CPU allowed by your environment. Record the affinity mask, kernel, requested policy, policy actually obtained, and whether memory locking succeeded. Compare an idle run with a bounded background load; a denied scheduling request is an experimental outcome, not a successful RT configuration.

**Executed validation (2026-10-18):** GCC 13.3.0 compiled the C17 source with warnings as errors. Three tests ran the default and obtained-affinity paths, rejected malformed CPU/options, and forced a child-process memory-lock failure with a zero lock limit. No privileged FIFO scheduling was executed. Run them with `python3 -m unittest discover -s ../../../labs -p test_rt_demo.py -v` from this directory. See the [validation record](../../../labs/validation.json).

The worker touches its own small stack buffer before announcing readiness, and the main thread touches an explicit static buffer. Volatile accesses retain those particular writes; they do not synchronize threads or prefault every future call, mapping, or stack page. Memory locking can also make a later thread creation/allocation fail, and that failure is checked.

The supplied demo discards its `clock_gettime` result and has no periodic wakeup schedule; it is an API/synchronization demonstration, not a latency benchmark. As an extension, add timestamp logging outside the hot path and distinguish waiting from work. Changing priority does not remove IRQ, device, or memory contention. This userspace demo cannot validate a robot controller’s worst-case execution time; carry its trace methodology into the DMA and driver lectures.

**Carry forward:** Baseline/load comparison including requested versus obtained policy. Continue with [the next course](../../Lecture-15.md) and reuse that evidence.
