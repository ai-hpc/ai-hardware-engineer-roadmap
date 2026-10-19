# Lecture 26: eBPF Observability with a Buildable ioctl Tracer

A latency number needs a defined start and end. This lesson measures **the interval between one thread entering and leaving `ioctl`**, then asks what that interval can explain. It includes time blocked or descheduled inside that call; it is not automatically device execution time or scheduler wakeup latency.

Prerequisites: [system calls](Lecture-04.md), [interrupts](Lecture-03.md), and [scheduling](Lecture-06.md). The [BPF build lab](../labs/BPF-Build-Lab.md) compiles an actual BPF object and userspace reader. The recorded validation stops before kernel loading or attachment: no runtime event, verifier-acceptance, or overhead result is claimed.

## What eBPF adds

A BPF program executes at a permitted hook with a constrained interface. The kernel checks the program before loading and may JIT-compile it, depending on configuration. The verifier reasons about safety properties such as allowed memory accesses and bounded execution under its model. It does not prove your measurement is meaningful, your filter identifies the right workload, or your application meets a deadline. Bugs, overhead, missing events, and configuration restrictions remain possible.

| Attachment family | Useful question | Limitation to account for |
|---|---|---|
| Tracepoint | Which defined kernel event occurred? | Availability and fields depend on the kernel |
| kprobe/kretprobe | Which function path ran? | Functions can change, inline, or be unavailable |
| fentry/fexit | Function entry/return with supported BTF tracing | Requires matching kernel capabilities |
| uprobe/uretprobe | Which userspace function ran? | Need a real binary/symbol and valid lifetime |
| XDP/TC | How are packets processed? | Programs can alter/drop traffic, not only observe it |
| sched_ext | How are eligible tasks scheduled? | Separate evolving scheduler interface; not an RT deadline proof |

Do not promise universal 50–200 ns overhead or a fixed advantage for one hook type. Hook frequency, map operations, stack collection, ring pressure, CPU state, and userspace output can dominate. Compare instrumented and uninstrumented workloads using the same run conditions.

## CO-RE and the skeleton

Compile Once–Run Everywhere relocations use BTF type information to adapt supported accesses to target layouts. They do not supply a missing helper, program type, tracepoint, permission, or semantic compatibility. An object compiled with a host's BTF is evidence for that build context; loading it elsewhere remains a separate check.

The [libbpf overview](https://docs.kernel.org/bpf/libbpf/libbpf_overview.html) describes the object/skeleton workflow:

```text
kernel BTF → vmlinux.h
BPF C source → clang BPF target → ELF object
ELF object → bpftool gen skeleton → generated C interface
userspace C + shared event header + skeleton → reader executable
```

The skeleton gives access to maps, programs, and configuration. It does not automatically declare an arbitrary ring-buffer event type for your userspace source. The supplied [shared header](../labs/ioctl_latency.h) defines a fixed 48-byte event used by both sides, with a userspace static size assertion.

## Follow the supplied tracer

Read [ioctl_latency.bpf.c](../labs/ioctl_latency.bpf.c) beside [ioctl_latency.c](../labs/ioctl_latency.c).

At syscall entry:

1. Obtain the 64-bit identity from `bpf_get_current_pid_tgid()`: high 32 bits are TGID, low 32 bits are TID.
2. Filter by the explicitly configured host TGID. A process name alone is truncated and can be reused or changed.
3. Store start time, command, and fd in a bounded hash map keyed by the complete identity.
4. Count failed map updates rather than silently claiming complete coverage.

At syscall exit:

1. Find the matching start, copy it locally, and delete the map entry.
2. Reserve a ring event; if reservation fails, increment the loss counter.
3. Initialize all event bytes and fill elapsed time, result, identity, command, fd, and current thread name.
4. Submit the event. A process-exit hook also removes stale per-thread start state.

The entry/exit pairing uses thread identity, because two threads in one process can call `ioctl` concurrently. Keying only by TGID lets one thread overwrite another's start. Filtering only on exit by a mutable name can leave an entry behind when the name changes.

The reader validates the TGID argument before attempting to load anything. It checks object load, attachment, ring creation, and polling failures. Signal handling uses `volatile sig_atomic_t`; shutdown detaches before draining available events and reporting per-CPU loss counters. Printed output can still slow the consumer enough to lose ring events.

## What its counters mean

| Counter | What it detects | What it does not prove |
|---|---|---|
| Failed start insertion | Bounded map could not store an entry | That every syscall entry hook was observed |
| Failed ring reservation | A matched exit could not allocate an event | That all userspace output reached persistent storage |
| Collected events | Successful matched entry/exit records | Full coverage during attachment/detachment |

An ioctl already running when tracing attaches has no observed entry. One still running when tracing detaches has no observed exit. Interrupted/restarted syscalls can produce kernel-level return behavior different from a single application call's final result. A long duration may reflect device waiting, locks, scheduling, or reclaim. Correlate with other evidence before naming a cause.

**Worked example:** one record reports 8 ms elapsed, while the device's own completion trace spans 2 ms. The 6 ms difference is not automatically “CPU scheduling delay.” The intervals may include setup, waits, lock contention, or different clocks/boundaries. Add scheduler and driver events with a shared time reference and match the request identity.

## A sensible investigation sequence

First run a workload with explicit input/output and count its completed requests. Then inventory available tracepoints on the target; permissions and tracefs location vary. Choose a narrow TGID/filter and a short run. Record kernel, BPF/libbpf versions, attachment types, event counts, loss counters, and workload change with tracing enabled.

Do not probe `dma_map_page` and call the result “DMA transfer latency”: mapping is setup, not hardware completion, and the name may be an inline wrapper rather than a probeable symbol. Do not attach a uprobe to an invented `run_model` function or historical openpilot binary path. Inspect the actual pinned program, symbols, and available instrumentation first.

For scheduling delay, match an appropriate wakeup/runnable event to the task's next scheduled-in event, accounting for already-runnable tasks and migration. For allocation sizes, use the target page size; `order × 4 KiB` is wrong even on a 4 KiB-page machine—the allocation is `PAGE_SIZE × 2**order`.

## Keep observability separate from policy

An XDP program that drops everything except one UDP port can discard ARP, IPv6 neighbor discovery, DHCP, fragmented traffic, and VLAN-encapsulated packets. A parser must check every header bound, EtherType, IPv4 header length, fragmentation, and transport size before using fields. A tracing exercise should not silently install a traffic policy on a robot.

Likewise sched_ext changes scheduling policy; it is not a drop-in inference deadline guarantee. Use the examples and helper names from the selected kernel's `tools/sched_ext` and documentation, since APIs evolve. Watchdog fallback and fair-class scheduling scope are part of the design. Use the existing [RT demo](demos/rt-demo/README.md) to distinguish measured wakeup behavior from a proven response-time bound.

## Completion checks

Run the [build lab](../labs/BPF-Build-Lab.md), retain its tool versions, and explain the exact validation boundary.

**Why not claim the verifier accepted the program after clang succeeds?** Clang checks compilation; the target kernel's verifier runs at load time, with its own features and configuration.

**Why initialize an entire ring event?** Fields not assigned explicitly, including padding, must not expose unrelated bytes or create ambiguous output. The event type is a cross-boundary ABI.

**Why do zero ring losses not prove a complete trace?** Attachment windows, missing entry pairs, failed start insertion, filtering, and shutdown can exclude events independently of ring pressure.
