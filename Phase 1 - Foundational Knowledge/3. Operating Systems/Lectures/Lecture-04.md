# Lecture 4: System Calls, vDSO, and Observability

A system call asks the kernel to perform an operation under a defined ABI. Its elapsed time includes much more than the privilege transition: validation, locks, page faults, device waiting, and scheduling can all contribute. Learn to distinguish **entry cost**, **CPU work**, and **wall-clock duration** before trying to optimize a camera pipeline.

Read [processes](Lecture-02.md) first. The worked examples below are either explicitly marked fragments or connected to an executable lab.

## The boundary and its return value

On common x86-64 Linux ABIs, a libc wrapper places the syscall number and arguments in registers and executes `SYSCALL`. On AArch64 the usual interface uses `SVC` with the syscall number in `x8`. Entry/dispatch/return implementation details depend on architecture and kernel version. A libc function is not necessarily a syscall: it may use a cached result, userspace synchronization, or vDSO code.

Syscalls establish or change access to kernel-managed resources. Once the kernel permits a mapping, an application may access a mapped device or queue without a new syscall for every operation. This is why “every GPU command is one ioctl” is not a valid universal model.

For libc syscall wrappers, failure often appears as `-1` with `errno` set. Other APIs differ: pthread functions commonly return an error number directly; `clock_nanosleep()` returns an error number; io_uring completions encode negative errors in `res`. Learn each function's contract instead of applying `perror()` indiscriminately.

## Measure an explicit operation

```bash
strace -c ./your_program
strace -T -e trace=mmap,munmap,ioctl ./your_program
```

These are documented inspection commands after replacing `your_program` with a real workload. `strace` itself changes execution timing. Count calls and inspect errors first; use an appropriate lower-overhead method for a serious timing experiment.

**Arithmetic example:** two 200-frame/s cameras issuing four calls per frame produce 1,600 calls/s. If an independently measured minimal-call cost were 300 ns, their transition-only estimate would be 0.48 ms of CPU time per second. That does not explain an 8 ms wait in `VIDIOC_DQBUF`. The hypothetical 300 ns is not a benchmark result, and x86/ARM mitigation costs cannot be ranked without specifying hardware and configuration.

## vDSO and clock domains

The kernel can map a small ELF helper object and shared data into a process. libc may use it to answer supported time queries without entering the kernel. The mapping can span multiple pages; it is not a single “current time” value updated on every call. Helpers combine kernel-maintained state with a clock source and retry when necessary; some requests fall back to syscalls.

Read `/proc/self/maps` and locate `[vdso]`/`[vvar]` where present. They are special mappings, not necessarily files that `nm /proc/PID/map_files/...` can open normally.

| Clock | Suitable use | Important boundary |
|---|---|---|
| `CLOCK_MONOTONIC` | Intervals/deadlines on one host | Does not jump backward with wall-clock steps, but can be rate-adjusted |
| `CLOCK_MONOTONIC_RAW` | Raw local interval measurement | Excludes normal clock discipline; not automatically aligned to another sensor |
| `CLOCK_REALTIME` | Calendar timestamps | Can be stepped; not a monotonic deadline clock |
| Device timestamp | Sensor acquisition timing | Requires a documented relationship to the host clock |

Using MONOTONIC does not synchronize separate computers, cameras, or accelerators. Record the clock domain and any conversion uncertainty. The [native RT demo](demos/rt-demo/README.md) uses absolute monotonic releases and measures wakeup lateness; it does not infer acquisition time from dequeue time.

## mmap and correctly sized shared memory

A new POSIX shared-memory object has length zero. Mapping a positive length without first sizing the object can lead to `SIGBUS` when accessed. `shm_open()` creates or opens a **named** object; it is not anonymous. The [shm_open manual](https://man7.org/linux/man-pages/man3/shm_open.3.html) describes sizing and unlinking.

**Lifecycle pseudocode; process coordination and error cleanup are part of the design:**

```text
owner creates a new unique name with O_CREAT | O_EXCL | O_RDWR
check fd; ftruncate to the validated length; check result
mmap that length; reject MAP_FAILED
initialize data and synchronization before advertising readiness
other process opens the agreed object, validates size/protocol, then maps it
use a supported process-shared synchronization/ownership mechanism
unlink the name when future opens are no longer needed
each process unmaps and closes its references after use
```

`MAP_SHARED` shares backing storage; it does not make conflicting unsynchronized C/C++ accesses legal. The [SPSC lab](../labs/SPSC-Queue.md) tests a same-process thread protocol. Do not assume its `std::atomic` objects are portable process-shared synchronization just because they can be placed in a mapping.

`memfd_create()` provides an anonymous fd-based alternative. It still needs checked `ftruncate()`/`mmap()` and an ownership protocol. Transfer its descriptor with `SCM_RIGHTS`, not its integer value. Mappings can keep storage alive after an fd closes; cleanup is reference-based.

HugeTLB mappings require supported sizes, suitable pool/resources and alignment. `MAP_HUGETLB` alone selects the configured default huge-page size, not universally 2 MiB. Base pages are not universally 4 KiB. CUDA pinned, device, and managed allocations have their own APIs; avoid teaching private driver node names as stable implementation contracts.

## ioctl and epoll

A V4L2 dequeue returns a completed-buffer identity using the selected memory method. For MMAP it is an index into queried mappings; USERPTR and DMA-BUF use different union fields. A blocking call can wait, and nonblocking mode can return `EAGAIN`. Do not label every dequeue as an uninterruptible DMA wait. [Lecture 18](Lecture-18.md) explains the complete buffer lifetime.

Epoll tracks an interest set and a ready set. Level-triggered mode remains ready while an operation can proceed. Edge-triggered mode requires careful draining with nonblocking operations until `EAGAIN`, plus fairness when a source remains busy. A notification does not guarantee a full message, or that another consumer has not consumed it. Neither mode is universally better for latency. See the [epoll manual](https://man7.org/linux/man-pages/man7/epoll.7.html).

## Timing and performance APIs

`PR_SET_TIMERSLACK` controls permitted coalescing for certain timed waits. Zero restores the inherited default; setting 1 ns is not a guarantee of 1 ns wakeup accuracy. Timer slack is not applied to threads under real-time scheduling policies, as the [timer-slack manual](https://man7.org/linux/man-pages/man2/PR_SET_TIMERSLACK.2const.html) explicitly states. Scheduling and interrupt delay can still make a wakeup late.

For `SCHED_DEADLINE`, runtime/deadline/period define a reservation subject to admission rules. They do not launch the application periodically or certify that a GPU job completes within its deadline. Use [Lecture 6](Lecture-06.md) and the RT demo rather than an incomplete `sched_setattr()` fragment without the proper UAPI structure, syscall wrapper, and error checks.

`perf_event_open()` needs a correctly initialized `perf_event_attr`, including `.size`, and checked open/control/read results. Counter availability, permissions, scheduling/multiplexing, and event semantics depend on the PMU. A generic cache-miss event is not a universal exact LLC-miss counter. Report the actual event and running/enabled time when interpreting measurements.

## eBPF, seccomp, and capabilities

The [BPF build lab](../labs/BPF-Build-Lab.md) supplies a real entry/exit tracer. It checks compilation and CLI failures; kernel verification and attachment are explicitly separate. eBPF is not inherently read-only: networking and policy programs can change behavior. A syscall tracepoint may be unavailable, and a probe cannot attach to every arbitrary inline function.

Seccomp uses classic-BPF-style filters on syscall metadata. Check architecture as well as syscall number, and account for ABI variants; numbers differ. A filter cannot safely inspect arbitrary pointed-to strings as if they were immutable values. `no_new_privs` or appropriate privilege and thread-synchronization behavior matter. The [kernel seccomp documentation](https://docs.kernel.org/userspace-api/seccomp_filter.html) explains its scope: it reduces exposed operations but is not a complete sandbox. There is no universal “TensorRT needs fewer than 50 syscalls” allowlist.

Capabilities divide privileges, but limits and policy still apply. `CAP_SYS_NICE` relates to scheduling privileges; `CAP_IPC_LOCK` does not eliminate every possible page fault; raw sockets generally involve `CAP_NET_RAW`, not simply `CAP_NET_ADMIN`. File capabilities can interact with namespaces, mount policy, `no_new_privs`, and packaging. Inspect the deployed result rather than adding `setcap` to an arbitrary binary as a generic lesson command.

**Check:** Does a 10 ms ioctl mean the syscall transition took 10 ms? **Answer:** No. Its interval may include device and scheduler waits.

**Check:** Does one edge-triggered event permit one read and then sleep? **Answer:** Not generally; unread data can remain without a new edge. Drain according to the nonblocking protocol and preserve fairness.
