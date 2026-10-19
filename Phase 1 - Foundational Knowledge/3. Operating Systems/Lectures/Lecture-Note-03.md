# Lecture Note 03: Boot Contracts and CPU Scheduling

Consolidates [Lecture 5](Lecture-05.md) and [Lecture 6](Lecture-06.md).

## Boot is a sequence of explicit handoffs

A generic path is BootROM → board initialization/firmware → boot manager → kernel and hardware description → early/final userspace → application. Actual stage names differ. Jetson's MB1/MB2 are not generic synonyms for U-Boot SPL or ARM exception levels; JetPack 5 and later use UEFI in the relevant bootloader role.

An ARM64 `Image` is normally uncompressed. A compressed image needs a supported decompression stage. An initramfs can provide early drivers/tools and `/init` before switching to the final root. Its compression/archive layout is not universally a single gzip stream.

Secure boot, module signing, rootfs integrity, confidentiality, and rollback protection are separate configured mechanisms. A signed artifact is not proof of correct behavior. Use the actual platform/release's trust chain and key policy.

## Device Tree is a typed hardware description

`compatible` selects a binding; `reg` depends on parent address/size cells and translation; `interrupts` uses its controller's binding. Missing `status` normally means available, as do `"okay"`/`"ok"`; it does not mean disabled. Available nodes still require a suitable driver and dependencies.

**Question:** Does a valid DTB prove an IMX477 camera works?

**Answer:** No. Syntax, schema, bindings, pinmux, clocks, media graph, physical wiring, probe, and actual capture are different milestones. A sensor subdevice is not automatically a completed capture node.

Use the [native Device Tree/module exercise](../../../Phase%204%20-%20Track%20B%20-%20Nvidia%20Jetson/1.%20Nvidia%20Jetson%20Platform/Orin-Nano-Kernel-Internals/labs/README.md) to inspect compiled artifacts. Its module was built, not loaded; its synthetic addresses describe no tested board.

## Fair service and real-time reservations

Linux fair scheduling uses EEVDF in modern kernels; its virtual deadlines are fairness-accounting values, not sensor-to-actuator deadlines. Earlier CFS explanations remain historical context. Kernel versions and vendor backports determine available behavior and tunables.

`SCHED_FIFO` schedules runnable threads by real-time priority subject to eligibility and kernel policy; it has no round-robin quantum. Blocking, yielding, higher-priority execution, and resource enforcement still matter. `SCHED_RR` adds rotation among equal-priority runnable threads. `SCHED_DEADLINE` combines runtime/deadline/period reservations with admission and budget rules; it does not itself periodically call your function or guarantee a GPU result.

**Worked example:** 3 ms CPU execution + 2 ms runnable waiting + 4 ms device waiting + 1 ms CPU execution gives 4 ms CPU demand and 10 ms response time. An 8 ms end-to-end deadline is missed even though CPU demand is below 8 ms.

## Practice without hiding the limits

Run the [RT demo](demos/rt-demo/README.md) first without privileged scheduling changes. Explain its clock, release period, sample count, maximum observed lateness, and failure handling. Then identify which terms a timer-only benchmark excludes from a robot's deadline.

**Question:** Does PREEMPT_RT plus affinity guarantee a deadline?

**Answer:** No. They change mechanisms and interference, but the workload needs bounded computation, blocking/device behavior, resource assumptions, and a complete timing analysis. A finite observed maximum is not a worst-case proof.
