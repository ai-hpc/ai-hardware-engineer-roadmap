# RT demo — user-space lab (Lectures 1–9)

Small **Linux** program that illustrates ideas from the first nine OS lectures: processes and threads, syscalls, scheduling (`SCHED_FIFO`), CPU affinity, memory locking / pre-faulting, and pthread synchronization.

Kernel-side topics (interrupts, boot, modules, device tree) are only described in comments or below—not implemented here.

## Build and run

From this directory:

```bash
cd "Phase 1 - Foundational Knowledge/3. Operating Systems/Lectures/demos/rt-demo"
make
./rt_demo [options]
```

Without `make`: `gcc -Wall -O2 -pthread -o rt_demo rt_demo.c`

**Options:** `./rt_demo --help` — e.g. `--cpu 2,3`, `--rt` (needs root / `cap_sys_nice`), `--lock-memory` (needs root / `cap_ipc_lock`), `--no-rt`.

**Trace syscalls:** `strace -e sched_setscheduler,sched_setaffinity,mlockall,gettid,clock_gettime ./rt_demo --rt --cpu 1 2>&1 | head -50`

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
- **Modules:** `lsmod`, `modinfo`, `modprobe` (e.g. `loop`)
- **Device tree (ARM):** `/sys/firmware/devicetree/base/` or machine model in boot logs

## Requirements

Linux or WSL with pthreads and `sched.h`. `--rt` / `--lock-memory` usually require root or the matching capabilities.

## Files

| File | Purpose |
|------|---------|
| `rt_demo.c` | Source |
| `Makefile` | Build |
| `README.md` | This file |

Back to the section hub: **[Operating Systems Guide](../../../Guide.md)**.

## Distinguish wakeup latency from real-time guarantees

Run the demo first with its default non-real-time policy on a CPU allowed by your environment. Record the affinity mask, kernel, requested policy, policy actually obtained, and whether memory locking succeeded. Compare an idle run with a bounded background load; a denied scheduling request is an experimental outcome, not a successful RT configuration.

The supplied demo discards its `clock_gettime` result and has no periodic wakeup schedule; it is an API/synchronization demonstration, not a latency benchmark. As an extension, add timestamp logging outside the hot path and distinguish waiting from work. Changing priority does not remove IRQ, device, or memory contention. This userspace demo cannot validate a robot controller’s worst-case execution time; carry its trace methodology into the DMA and driver lectures.

**Carry forward:** Baseline/load comparison including requested versus obtained policy. Continue with [the next course](../../Lecture-15.md) and reuse that evidence.
