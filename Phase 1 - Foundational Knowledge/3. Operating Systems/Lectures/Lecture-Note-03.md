# Lecture Note 03 (L5, L6): Kernel Modules, Boot & Device Tree; CPU Scheduling (CFS, EEVDF & RT)

**Combines:** Lecture L5 (Kernel Modules, Boot Process & Device Tree) and Lecture L6 (CPU Scheduling: CFS, EEVDF & Real-Time Classes).

---

## How This Note Is Organized

1. **Part 1 (L5) — Boot & Device Tree:** ARM SoC boot chain; secure boot; UEFI vs U-Boot; initramfs; Device Tree (DTS/DTB, compatible, reg, interrupts); driver binding.
2. **Part 2 (L6) — CPU scheduling:** Scheduler class hierarchy (stop, dl, rt, fair, idle); CFS vruntime and nice; EEVDF (eligibility, virtual deadline); SCHED_FIFO/SCHED_RR/SCHED_DEADLINE; when to use RT classes.

---

# Part 1 (L5): Kernel Modules, Boot Process & Device Tree

**Context:** Before inference runs, bootloader loads kernel + DTB; kernel parses **Device Tree** to discover SoC peripherals; drivers bind via **compatible strings**. Same driver binary supports **multiple boards via different DTs**.

---

## Linux Boot Sequence (ARM SoC)

Power-on → BootROM → SPL (DRAM, clocks) → TF-A BL31 (EL3) → U-Boot/CBoot (kernel + DTB + initramfs) → kernel decompress → start_kernel() → init (systemd). Jetson: MB1, MB2, CBoot in place of SPL/U-Boot. **Chain of trust:** each stage verifies next signature.

---

## Secure Boot, UEFI vs U-Boot

Secure boot: ROM verifies SPL; SPL verifies kernel; kernel can verify rootfs/modules. UEFI: x86 and some ARM servers; EFI stub, DTB from config. U-Boot: embedded (Zynq, Jetson, i.MX); passes kernel + DTB in registers; env vars, boot.cmd.

---

## initramfs

Compressed cpio; **early userspace** (busybox, udev, cryptsetup, init). Kernel mounts as root; runs /init; then real root and switch_root. Needed when **root needs drivers or tools not in kernel** (e.g. LUKS). Jetson: TNSPEC and extlinux in initramfs.

---

## Device Tree (DTS → DTB)

**Purpose:** Describe SoC hardware (addresses, IRQs, clocks) without hardcoding in kernel. DTS source → dtc → DTB; bootloader passes DTB physical address (e.g. ARM64 x0). Kernel parses and creates platform_device nodes; **compatible** string binds to driver’s of_match_table.

**Node:** compatible, reg (address/size), interrupts, clocks, status ("okay"/"disabled"). Driver gets resources in probe via of_iomap, of_irq_get, of_get_property. MODULE_DEVICE_TABLE(of, ...) for udev load on match.

---

# Part 2 (L6): CPU Scheduling — CFS, EEVDF & Real-Time Classes

**Context:** Scheduler decides which task runs next. **Real-time classes (dl, rt) are checked before the fair class** (CFS/EEVDF). One SCHED_FIFO task at priority 1 preempts all SCHED_NORMAL tasks. For deadlines (e.g. inference, CAN), use **SCHED_FIFO or SCHED_DEADLINE**.

---

## Scheduler Class Hierarchy

(High to low) **stop_sched_class** (internal) → **dl_sched_class** (SCHED_DEADLINE, CBS/EDF) → **rt_sched_class** (SCHED_FIFO, SCHED_RR, priority 1–99) → **fair_sched_class** (SCHED_NORMAL, CFS/EEVDF) → **idle_sched_class**. Higher class always preempts lower; no override.

---

## CFS (Completely Fair Scheduler, &lt; 6.6)

Ideal CPU at 1/N speed; **vruntime** = weighted runtime; scheduler picks smallest vruntime (red-black tree). **Nice** changes weight (e.g. nice -5 ≈ 3× share of nice 0). **Weakness:** Newly woken task can wait up to sched_latency_ns (e.g. 6 ms) if others have lower vruntime — bad for wakeup latency.

---

## EEVDF (Linux 6.6+)

Replaces CFS. **lag** = CPU time owed vs ideal; **eligible** = not ahead of fair share; **virtual deadline** = when next slice is due. Among **eligible** tasks, run **earliest virtual deadline**. Woken task with positive lag gets scheduled sooner than under CFS — better tail latency for inference.

---

## Real-Time Classes

**SCHED_FIFO:** Priority 1–99; runs until yield or preempted by higher priority; no time slice. **SCHED_RR:** Same but with round-robin within priority. **SCHED_DEADLINE:** Runtime, deadline, period (CBS/EDF). Use for controlsd, modeld, sensor loops. CFS/EEVDF for general work; set inference/control to SCHED_FIFO or DEADLINE to avoid 6 ms wakeup delay.

---

## Tuning and Inspection

sched_latency_ns, sched_min_granularity_ns (CFS). /proc/[pid]/sched (vruntime, slice). **chrt** for policy/priority. For **deterministic latency**, use RT class + isolcpus/PREEMPT_RT (see Lecture-Note 04).

---

## Summary

| L5 | Boot chain; secure boot; UEFI vs U-Boot; initramfs; Device Tree (compatible, reg, interrupts); driver binding. |
| L6 | Scheduler classes (dl, rt, fair, idle); CFS vruntime/nice; EEVDF eligibility/deadline; SCHED_FIFO/RR/DEADLINE for RT. |

---

## AI Hardware Connection

- L5: DTB defines camera, NPU, accelerators on Jetson/custom SoC; one driver supports multiple boards; udev loads module on compatible match.
- L6: modeld/controlsd on SCHED_FIFO or DEADLINE to meet frame and CAN deadlines; EEVDF improves fair-class tail latency on 6.6+; CFS wakeup latency reason to use RT class for inference.

---

*Combines Lectures L5, L6 (Kernel Modules, Boot & Device Tree; CPU Scheduling: CFS, EEVDF & Real-Time Classes).*
