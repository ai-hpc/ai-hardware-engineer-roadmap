# Operating Systems — Final Practice (14 problems)

Short-answer practice aligned with [Operating Systems — Guide](Guide.md) and the Phase 1 lecture notes below. Difficulty: **moderate**. **Each numbered problem is a single question** (write a short paragraph unless noted).

**Suggested time:** 75–110 minutes total. Problems 6–14 check the ownership, numerical, and measurement skills needed for AI/robotics systems.

---

## Problem 1 — Building a custom kernel

**Outline how you build a custom Linux kernel** for a target (embedded AI board or generic tree), and **name typical tools or workflows** used along the way — e.g. where **configuration** happens, how **cross-compilation** fits in, and at least one **distribution-style** approach (not only “run `make` on your laptop for x86”).

*Maps to: [Lecture 25 — Capstone: custom Linux images with Yocto](Lectures/Lecture-25.md) (factory workflow); [Lecture 5](Lectures/Lecture-05.md) / vendor BSP context for **defconfig**, **DTB**, and flashing.*

---

## Problem 2 — PREEMPT_RT (real-time Linux)

**What is `PREEMPT_RT`, what problem on “normal” Linux does it address, and what does it change in the kernel at a high level** (enough to show you understand *predictability* vs raw speed)?

*Maps to: [Lecture 7 — Real-Time Linux: PREEMPT_RT & Determinism](Lectures/Lecture-07.md).*

---

## Problem 3 — ext4

**What is the ext4 filesystem, and what are its main strengths** for typical Linux roots / embedded boards (name several concrete pros from the lecture — allocation, directories, journaling/crash behavior, maturity, etc.)?

*Maps to: [Lecture 21 — Filesystems: ext4, btrfs, F2FS & overlayfs](Lectures/Lecture-21.md).*

---

## Problem 4 — Boot order

**Choose one explicitly identified ARM SoC boot flow or a clearly labeled generic example, and list its handoffs from power-on to PID 1.** Distinguish board-specific initialization/firmware, boot manager, kernel entry, and early/final userspace. Do not imply all boards use the same stage names.

*Maps to: [Lecture 5 — Kernel Modules, Boot Process & Device Tree](Lectures/Lecture-05.md).*

---

## Problem 5 — Kernel modules and device drivers

**What is a loadable kernel module, and how does driver bring-up typically work** for a **Device Tree–matched** driver — including **`compatible` matching**, **`probe()`**, and why **`modprobe`** is usually used instead of **`insmod`**?

*Maps to: [Lecture 5 — Kernel Modules, Boot Process & Device Tree](Lectures/Lecture-05.md).*

---

## Problem 6 — Response time versus CPU time

A job runs for 3 ms, waits runnable for 2 ms, waits for a device for 4 ms, then runs for 1 ms. Assuming no overlap or unlisted intervals, calculate CPU demand and response time. Does it meet an 8 ms response deadline?

*Maps to: [Lecture 6](Lectures/Lecture-06.md).*

## Problem 7 — Late completion after timeout

Request 17 times out and its buffer/completion object is immediately reused for request 18. Explain two ways a late interrupt or DMA operation can violate correctness, and the ownership condition required before reuse.

*Maps to: [Lecture 9](Lectures/Lecture-09.md).*

## Problem 8 — Queue publication

In the [SPSC queue](labs/SPSC-Queue.md), explain why both producer publication and consumer slot release use a release/acquire relationship. What can break if the producer sees a free slot before the previous consumer's reads are complete?

## Problem 9 — Memory arithmetic

An aligned 64 MiB region uses 4 KiB pages, or alternatively 2 MiB pages. Calculate both page counts and explain why neither number proves a TLB miss rate or a latency improvement.

*Maps to: [Lecture 13](Lectures/Lecture-13.md).*

## Problem 10 — Evidence and claims

A 60-second timer benchmark reports a maximum 40 µs wakeup delay on a pinned thread. Explain why this neither proves a 50 µs worst-case bound nor proves a robot's 2 ms sensor-to-command deadline.

*Maps to: [Lecture 7](Lectures/Lecture-07.md) and [Lecture 8](Lectures/Lecture-08.md).*

## Problem 11 — Camera buffer ownership

A capture buffer is dequeued, passed to an asynchronous GPU operation, and immediately requeued. Explain the race and give a correct reuse condition. Does stopping V4L2 capture alone prove GPU completion?

*Maps to: [Lecture 18](Lectures/Lecture-18.md) and [Lecture 19](Lectures/Lecture-19.md).*

## Problem 12 — Storage durability

A program writes a new configuration to a temporary file and renames it over the old name. Explain why visible atomic replacement and durable commitment are different, and identify the usual file/directory synchronization steps.

*Maps to: [Lecture 21](Lectures/Lecture-21.md).*

## Problem 13 — CPU quota and GPU claims

Four threads share a 50 ms CPU quota per 100 ms period. If they all execute simultaneously, how soon can they exhaust it? Does a CPU mask make the CPUs exclusive, and can Jetson Orin use MIG for hardware GPU partitioning?

*Maps to: [Lecture 23](Lectures/Lecture-23.md).*

## Problem 14 — Build evidence

The BPF tracer compiles, its ELF sections exist, and malformed CLI arguments fail correctly. Which claims are established, and which still require kernel or workload execution?

*Maps to: [BPF build lab](labs/BPF-Build-Lab.md).*

## Answer key (for self-check)

<details>
<summary>Click to expand</summary>

**1.** **Typical flow:** obtain **kernel source** (mainline, vendor/L4T BSP, or via a build system recipe) → choose **config** (**`defconfig`**, **`make menuconfig` / nconfig**, or **fragments** / `CONFIG_*` in Yocto **`bbappend`**) → build with a **cross-toolchain** matching the target (**`aarch64-linux-gnu-*`** etc.) or let **Yocto/BitBake** invoke **`make`** inside the recipe → produce **`Image`/`zImage`**, **modules**, and match **DTB** / boot artifacts → deploy (flash, OTA, or replace `/boot`). **Tools / stacks (credit any coherent set):** **Yocto** (**Poky**, **BitBake**, **meta-tegra** / machine layers), **`make` + `LLVM=` optional**, **Buildroot** as an alternative embedded integrator, **DKMS** for **out-of-tree** modules on an existing kernel, **`installkernel` / `modules_install`** for packaging.

**2.** **`PREEMPT_RT`** targets **bounded worst-case latency** (hard/soft RT **predictability**), not higher average throughput. It makes almost the whole kernel **preemptible** by turning many **spinlocks into sleeping rtmutexes**, moving **IRQ handling** into **schedulable threads**, and making **softirq** work **preemptible** — shrinking important non-preemptible sections that delay high-priority tasks. Raw critical sections and some IRQ paths remain; the configuration alone does not prove any numerical deadline.

**3.** **ext4** is the default **ext family** (ext2/3/4) **journaled** Linux disk FS under **VFS**, above the **block layer**. **Pros** (examples): **extent-based** allocation and **64-bit** volumes; **delayed allocation** for better locality; **dir_index** (htree) for large directories; **journaling** and modes like **`data=ordered`** for sensible crash recovery; mature and widely used on **rootfs** (e.g. Jetson). Journaling does not guarantee that every application update is durable; that requires an explicit persistence protocol.

**4.** Generic handoffs: **BootROM** → **board-specific initialization/firmware** → **boot manager** (e.g. a supported U-Boot or UEFI flow) → **selected kernel plus DTB/other hardware description and optional initramfs** → **kernel initialization** → **early `/init` or final PID 1** → application services. Names/order inside firmware differ by SoC. ARM64 `Image` is normally uncompressed; do not invent a decompression stage in every kernel entry. JetPack 5 and later use UEFI in the relevant Jetson bootloader role.

**5.** A **module** is a **`.ko`** linked into the **running** kernel (extra drivers/code) **without** reboot. For **DT** devices, **`MODULE_DEVICE_TABLE` / `compatible`** matches a node; **`modprobe`** loads the module (and **dependencies**); the kernel calls **`probe()`** when the device is **bound** (boot or hotplug). **`modprobe`** is preferred over **`insmod`** because it **resolves dependency order** from **`modules.dep`**.

**6.** CPU demand is `3 + 1 = 4 ms`; response time is `3 + 2 + 4 + 1 = 10 ms`, so the job misses the 8 ms response deadline.

**7.** A stale notification can complete the wrong request, and stale DMA can overwrite reused storage. A generation ID can reject the notification but cannot stop a write. Keep request/buffer lifetime until the supported completion or abort/reset/quiescence protocol proves the device and callbacks can no longer access it.

**8.** The producer publishes initialized payload before the consumer reads it. The consumer publishes completed reading before the producer overwrites the slot. Without the second ordering relation, ordinary payload access can overlap and become a data race/corruption. The fixed SPSC owner counts and object lifetime are also part of the proof.

**9.** `64 MiB / 4 KiB = 16,384`; `64 MiB / 2 MiB = 32`. These count translations needed to cover the region, not physical TLB capacity or actual misses. Page availability, mapping alignment, access pattern, cache/translation hierarchy, allocation cost, and the real bottleneck determine performance.

**10.** It is an observed maximum under finite test conditions. Pinning does not exclude all interference. Wakeup latency excludes CPU execution, queue waits, device transfers/computation, and actuation. Record the full path, clocks, workload, failure model, and analytical assumptions before making a deadline claim.

**11.** The camera may overwrite bytes while the GPU still reads them. Keep the buffer out of the capture queue until the last consumer's completion/fence and cache-visibility contract permit reuse. `STREAMOFF` stops the relevant capture path, not unrelated GPU work.

**12.** Rename changes the namespace atomically under its same-filesystem contract, but it is not a persistence barrier. Write all bytes and check errors, fsync the temporary file, rename, then fsync the containing directory. The filesystem/device stack and concurrent-update protocol still matter.

**13.** About `50 / 4 = 12.5 ms` wall time under the stated full parallel execution assumption. The quota is aggregate CPU time, not a reserved half-core. A CPU eligibility mask alone is not exclusivity. Jetson Orin does not support MIG.

**14.** The recorded source compiles into a BPF object and native reader with a matching event ABI, the named sections exist, and the tested malformed arguments fail before loading. Kernel verifier acceptance, successful attachment, actual event pairing, loss behavior under load, tracing overhead, and timing conclusions remain untested by this build-only exercise.

</details>
