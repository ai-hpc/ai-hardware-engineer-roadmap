# Orin Nano 8GB — Real-Time Linux (PREEMPT_RT) Deep Dive

<div class="course-identity auto-course" style="--course-accent: #0284c7; --course-accent-rgb: 2, 132, 199;" markdown="1">
<div class="course-identity__icon">ON8R</div>
<div markdown="1">
<p class="course-identity__eyebrow">Deep Dive · Jetson Track</p>
<p class="course-identity__title">Specialized course identity for Orin Nano 8GB — Real-Time Linux (PREEMPT_RT) Deep Dive.</p>
<p class="course-identity__meta">Artifact: Jetson integration demo · Measure: latency, memory, power, logs</p>
</div>
</div>


> **Scope:** Kernel-level understanding of real-time Linux on Jetson Orin Nano — PREEMPT_RT patch internals, preemption models, interrupt threading, lock primitives under RT, priority inversion and PI mutexes, rt-tests suite, latency source analysis, WCET methodology, ARM Cortex-A78AE RT specifics, GICv3 tuning, RT-safe kernel module development, and production RT system validation.
>
> **Prerequisites:** Familiarity with [Orin Nano kernel internals](../Orin-Nano-Kernel-Internals/Guide.md) and [real-time inference](../Orin-Nano-Real-Time-Inference/Guide.md). Basic understanding of Linux scheduling and kernel concepts.
>
> **Relationship to Real-Time Inference guide:** The [Real-Time Inference guide](../Orin-Nano-Real-Time-Inference/Guide.md) covers application-level RT — TensorRT optimization, CUDA patterns, DLA determinism, and inference pipeline tuning. This guide covers the **kernel and hardware layers underneath** — what makes RT scheduling actually work, why latencies occur, and how to eliminate them at the source.

---


## 1. Linux Preemption Models

### 1.1 The Four Preemption Levels

Linux supports four preemption configurations. Each trades throughput for determinism:

```
CONFIG_PREEMPT_NONE (No Forced Preemption)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  Kernel code runs to completion (until it voluntarily yields)
  Preemption points: only at syscall return and interrupt return
  Use case: servers maximizing throughput
  Worst-case latency: 10-100 ms (unbounded in pathological cases)

  Task A (RT)     ████████████████░░░░░░░░░░████████
  Task B (kernel) ░░░░░░░░░░░░░░░░████████████░░░░░░
                                   ↑ B runs kernel code
                                     A cannot preempt until B returns


CONFIG_PREEMPT_VOLUNTARY (Voluntary Preemption)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  Adds explicit preemption points (might_sleep()) in long code paths
  Preemption points: ~1000 voluntary check points in kernel
  Use case: desktop Linux (stock Ubuntu/Fedora default)
  Worst-case latency: 1-10 ms

  Task A (RT)     ████████████░░░░░████████████████
  Task B (kernel) ░░░░░░░░░░░░████░░░░░░░░░░░░░░░░
                               ↑ B hits might_sleep(), A preempts


CONFIG_PREEMPT (Preemptible Kernel)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  All kernel code is preemptible EXCEPT:
    - Inside spinlock critical sections
    - When interrupts are disabled
    - In hard IRQ context
  Use case: low-latency desktop, standard JetPack default
  Worst-case latency: 100 µs - 1 ms

  Task A (RT)     ████████████░██████████████████
  Task B (kernel) ░░░░░░░░░░░░█░░░░░░░░░░░░░░░░
                               ↑ B preempted mid-kernel-code
                                 (unless holding spinlock)


CONFIG_PREEMPT_RT (Full Real-Time)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  EVERYTHING is preemptible:
    - Spinlocks → converted to RT mutexes (sleepable, preemptible)
    - Hard IRQs → converted to kernel threads (schedulable)
    - Only raw_spinlocks remain non-preemptible (very few)
  Use case: hard/firm real-time systems, robotics, industrial control
  Worst-case latency: 5-50 µs (on tuned system)

  Task A (RT p80) ████████████████████████████████
  Task B (kernel) ░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░
                  ↑ A preempts B at ANY point, even mid-spinlock
```

### 1.2 Checking Current Preemption Model on Orin Nano

```bash
# Method 1: /sys/kernel/realtime (only exists with PREEMPT_RT)
cat /sys/kernel/realtime 2>/dev/null
# "1" = PREEMPT_RT active
# File missing = not PREEMPT_RT

# Method 2: kernel config
zcat /proc/config.gz | grep -E "^CONFIG_PREEMPT"
# CONFIG_PREEMPT_RT=y           ← full RT
# CONFIG_PREEMPT=y              ← preemptible (stock JetPack)
# CONFIG_PREEMPT_VOLUNTARY=y    ← voluntary only
# CONFIG_PREEMPT_NONE=y         ← server mode

# Method 3: uname
uname -v
# Look for "PREEMPT RT" in version string
# e.g., "#1 SMP PREEMPT RT Tue Jan 15 12:00:00 UTC 2024"

# Method 4: /proc/version
cat /proc/version
# Contains "PREEMPT_RT" if RT patched
```

### 1.3 Latency Comparison on Orin Nano (Measured)

```
cyclictest -t1 -p 90 -n -i 1000 -l 100000 (100k samples)

Preemption Model       Avg (µs)    Max (µs)    p99 (µs)    Jitter (µs)
─────────────────────────────────────────────────────────────────────────
PREEMPT_NONE           ~150        ~12,000     ~800        High
PREEMPT_VOLUNTARY      ~50         ~3,000      ~250        Medium
PREEMPT                ~15         ~500        ~80         Low
PREEMPT_RT             ~5          ~45         ~20         Very Low
PREEMPT_RT + tuned*    ~4          ~18         ~10         Minimal

* tuned = isolcpus + nohz_full + rcu_nocbs + IRQ affinity +
          performance governor + THP disabled + mlockall
```

---

## 2. PREEMPT_RT Patch Internals

### 2.1 What the RT Patch Actually Changes

The PREEMPT_RT patch transforms the Linux kernel from a general-purpose OS into a real-time OS. Here is exactly what it modifies:

```
Transformation 1: Spinlocks → RT Mutexes
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  Standard kernel:
    spinlock_t → busy-wait, interrupts disabled, non-preemptible
    Holding a spinlock blocks ALL higher-priority tasks on that CPU

  PREEMPT_RT:
    spinlock_t → rt_mutex (sleeping lock with priority inheritance)
    Lock holder CAN be preempted by higher-priority task
    If higher-priority task needs same lock → priority inheritance kicks in

    EXCEPTION: raw_spinlock_t remains a true spinlock
    Used only in scheduler core, interrupt descriptor tables,
    and a few critical paths (~50 locations vs ~10,000 spinlocks)


Transformation 2: Hard IRQs → Threaded IRQs
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  Standard kernel:
    Hardware interrupt → hard IRQ handler runs immediately
    ALL tasks suspended during hard IRQ
    Long ISR = unbounded latency for RT tasks

  PREEMPT_RT:
    Hardware interrupt → minimal top-half (acknowledge HW, wake thread)
    Actual work done in irq/<N>-<name> kernel thread
    IRQ threads have scheduling priority → can be preempted by RT tasks

    Example:
    $ ps aux | grep irq/
    root  [irq/35-nvgpu]       ← GPU interrupt handler thread
    root  [irq/42-tegra-i2c]   ← I2C interrupt handler thread
    root  [irq/68-xhci_hcd]    ← USB interrupt handler thread

    Each IRQ thread can be assigned a specific RT priority:
    $ chrt -p $(pgrep -f "irq/35-nvgpu")
    # Shows scheduling policy and priority


Transformation 3: Softirqs → Kernel Threads
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  Standard kernel:
    Softirqs (NET_RX, BLOCK, SCHED, TIMER, etc.) run in
    interrupt context — non-preemptible, can delay RT tasks

  PREEMPT_RT:
    Softirqs run in per-CPU ksoftirqd threads
    Schedulable and preemptible
    RT tasks always preempt softirq processing

    $ ps aux | grep ksoftirqd
    root  [ksoftirqd/0]
    root  [ksoftirqd/1]
    ...


Transformation 4: Sleeping Spinlocks Enable Preemption
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  Since spinlocks become sleeping locks under PREEMPT_RT,
  the critical sections they protect become preemptible.

  This means a high-priority task can preempt a low-priority
  task that holds a lock — something impossible with standard
  spinlocks (which disable preemption entirely).

  The tradeoff: sleeping locks have higher overhead than
  busy-wait spinlocks (~200ns vs ~50ns per lock/unlock).
  But determinism is gained: no unbounded priority blocking.
```

### 2.2 What PREEMPT_RT Does NOT Fix

```
PREEMPT_RT eliminates most kernel-induced latency. It does NOT fix:

1. Hardware latency
   - Cache misses (L1: 1ns, L2: 5ns, L3: 15ns, DRAM: 50-80ns)
   - TLB misses (page table walk: 10-100ns)
   - PCIe transaction latency
   - NVMe I/O latency
   → These are physical constraints, not software

2. NVIDIA GPU scheduling
   - GPU kernel dispatch is NOT real-time
   - CUDA kernels run on GPU's own scheduler (not Linux scheduler)
   - GPU preemption is coarse-grained (compute preemption, not instruction-level)
   → Use DLA for more deterministic execution

3. Thermal throttling
   - When SoC hits thermal limit, DVFS reduces clock speed
   - This increases execution time unpredictably
   → Lock clocks, ensure adequate cooling

4. SMI (System Management Interrupt) equivalent
   - ARM TrustZone / OP-TEE can steal CPU cycles
   - BPMP firmware operations are non-preemptible
   → Cannot be eliminated, but typically < 10 µs

5. Memory allocation
   - kmalloc/vmalloc can cause page reclaim, compaction
   - Even with PREEMPT_RT, memory allocation is not bounded-time
   → Pre-allocate all memory before entering RT loop
```

### 2.3 RT Patch Version Compatibility With Jetson

```
JetPack   Kernel     PREEMPT_RT Patch    Status
─────────────────────────────────────────────────────
5.1.x     5.10.104   rt63               Supported (manual apply)
5.1.2     5.10.120   rt70               Supported (manual apply)
6.0       5.15.122   rt65               Official NVIDIA support
6.1       5.15.136   rt79               Official NVIDIA support
6.2       5.15.148   rt80+              Official NVIDIA support

JetPack 6.x is the first version where NVIDIA officially
provides and tests the PREEMPT_RT configuration.

For JetPack 5.x: patches exist but are community-supported.
Some NVIDIA drivers may have spinlock assumptions that conflict
with PREEMPT_RT — test thoroughly.
```

---

## 3. Building PREEMPT_RT Kernel for Orin Nano

### 3.1 Complete Build Procedure

```bash
# === Host machine (x86_64 Ubuntu 20.04/22.04) ===

# Step 0: Install build dependencies
sudo apt-get install -y build-essential bc flex bison libssl-dev \
    libncurses-dev lz4 gcc-aarch64-linux-gnu

# Step 1: Download L4T kernel source (JetPack 6.x)
# From: https://developer.nvidia.com/embedded/jetson-linux
mkdir -p ~/jetson-rt && cd ~/jetson-rt
wget https://developer.nvidia.com/downloads/embedded/l4t/r36_release_v4.0/sources/public_sources.tbz2
tar xf public_sources.tbz2
cd Linux_for_Tegra/source
tar xf kernel_src.tbz2

# Step 2: Apply PREEMPT_RT patches
# NVIDIA includes the RT patches in the source tarball
./generic_rt_build.sh apply

# Verify patches applied:
grep -r "PREEMPT_RT" kernel/kernel-5.15/Makefile
# Should show RT version string

# Step 3: Configure for RT
export CROSS_COMPILE=aarch64-linux-gnu-
export ARCH=arm64
export LOCALVERSION="-tegra-rt"

cd kernel/kernel-5.15
make O=build tegra_defconfig

# Step 4: Enable RT and critical RT options
cd build
scripts/config --enable  CONFIG_PREEMPT_RT
scripts/config --disable CONFIG_PREEMPT_VOLUNTARY
scripts/config --disable CONFIG_PREEMPT_NONE
scripts/config --disable CONFIG_PREEMPT__LL
scripts/config --disable CONFIG_PREEMPT_DYNAMIC
scripts/config --enable  CONFIG_HIGH_RES_TIMERS
scripts/config --set-val CONFIG_HZ 1000
scripts/config --enable  CONFIG_NO_HZ_FULL
scripts/config --enable  CONFIG_CPU_ISOLATION
scripts/config --enable  CONFIG_IRQ_FORCED_THREADING

# RT debugging (enable during development, disable for production)
scripts/config --enable  CONFIG_DEBUG_PREEMPT
scripts/config --enable  CONFIG_DEBUG_RT_MUTEXES
scripts/config --enable  CONFIG_PREEMPT_TRACER
scripts/config --enable  CONFIG_IRQSOFF_TRACER
scripts/config --enable  CONFIG_SCHED_TRACER
scripts/config --enable  CONFIG_FUNCTION_TRACER

cd ..

# Step 5: Build
make O=build -j$(nproc) Image modules dtbs

# Step 6: Package modules
make O=build INSTALL_MOD_PATH=../modules_out modules_install

# Step 7: Copy to Jetson
scp build/arch/arm64/boot/Image jetson:/tmp/Image-rt
scp -r modules_out/lib/modules/5.15.*-tegra-rt jetson:/tmp/

# === On the Jetson device ===
# Step 8: Install RT kernel
sudo cp /boot/Image /boot/Image.bak
sudo cp /tmp/Image-rt /boot/Image
sudo cp -r /tmp/5.15.*-tegra-rt /lib/modules/
sudo depmod -a 5.15.*-tegra-rt

# Step 9: Update extlinux.conf
sudo cp /boot/extlinux/extlinux.conf /boot/extlinux/extlinux.conf.bak
# Edit LABEL entry to point to new kernel:
# LINUX /boot/Image-rt

# Step 10: Reboot and verify
sudo reboot
# After reboot:
uname -a            # Should show "PREEMPT RT"
cat /sys/kernel/realtime  # Should show "1"
```

### 3.2 RT Kernel Config Fragments

```bash
# configs/rt-core.cfg — Core RT settings
CONFIG_PREEMPT_RT=y
CONFIG_HIGH_RES_TIMERS=y
CONFIG_HZ_1000=y
CONFIG_HZ=1000
CONFIG_NO_HZ_FULL=y
CONFIG_CPU_ISOLATION=y
CONFIG_IRQ_FORCED_THREADING=y
CONFIG_RCU_NOCB_CPU=y
CONFIG_RCU_BOOST=y
CONFIG_RCU_BOOST_DELAY=500

# configs/rt-debug.cfg — RT debugging (development only)
CONFIG_DEBUG_PREEMPT=y
CONFIG_DEBUG_RT_MUTEXES=y
CONFIG_DEBUG_SPINLOCK=y
CONFIG_DEBUG_LOCK_ALLOC=y
CONFIG_PROVE_LOCKING=y
CONFIG_PREEMPT_TRACER=y
CONFIG_IRQSOFF_TRACER=y
CONFIG_SCHED_TRACER=y
CONFIG_FUNCTION_TRACER=y
CONFIG_FTRACE=y
CONFIG_LATENCY_TOP=y
CONFIG_LOCK_STAT=y
CONFIG_STACKTRACE=y

# configs/rt-production.cfg — Production RT (no debug overhead)
# CONFIG_DEBUG_PREEMPT is not set
# CONFIG_DEBUG_RT_MUTEXES is not set
# CONFIG_DEBUG_SPINLOCK is not set
# CONFIG_DEBUG_LOCK_ALLOC is not set
# CONFIG_PROVE_LOCKING is not set
# CONFIG_PREEMPT_TRACER is not set
# CONFIG_LATENCY_TOP is not set
# CONFIG_LOCK_STAT is not set
# CONFIG_FTRACE is not set
# CONFIG_DEBUG_INFO is not set
```

### 3.3 Yocto Integration for RT Kernel

```bash
# meta-myproject/recipes-kernel/linux/linux-tegra_%.bbappend

FILESEXTRAPATHS:prepend := "${THISDIR}/files:"

SRC_URI += " \
    file://rt-core.cfg \
    file://rt-production.cfg \
"

# Apply RT patches (if not already in NVIDIA source)
# The generic_rt_build.sh apply step may need to be done
# in do_patch or via SRC_URI patches

KERNEL_FEATURES:append = " rt-core.cfg rt-production.cfg"

# Or use KERNEL_CONFIG_FRAGMENTS
KERNEL_CONFIG_FRAGMENTS += " \
    ${WORKDIR}/rt-core.cfg \
    ${WORKDIR}/rt-production.cfg \
"
```

---

## 4. Interrupt Threading Under PREEMPT_RT

### 4.1 Hard IRQ vs Threaded IRQ

```
Standard Kernel — Hard IRQ Execution:

  CPU executing RT task
       │
       ▼ ←── hardware interrupt fires
  ┌────────────────────────────────────┐
  │ HARD IRQ CONTEXT                   │
  │ - Interrupts disabled on this CPU  │
  │ - Not preemptible                  │
  │ - RT task BLOCKED for entire ISR   │
  │ - Duration: 1 µs to 500+ µs       │
  └────────────────────────────────────┘
       │
       ▼ RT task resumes
  (latency = ISR duration — unbounded)


PREEMPT_RT — Threaded IRQ Execution:

  CPU executing RT task (priority 80)
       │
       ▼ ←── hardware interrupt fires
  ┌──────────────────────────────┐
  │ MINIMAL TOP-HALF             │
  │ - Acknowledge hardware       │
  │ - Wake IRQ thread            │
  │ - Duration: < 1 µs           │
  └──────────────────────────────┘
       │
       ▼ RT task resumes immediately (if higher priority)

  IRQ thread (priority 50) runs when scheduled:
  ┌──────────────────────────────┐
  │ IRQ THREAD CONTEXT            │
  │ - Runs as SCHED_FIFO thread   │
  │ - Preemptible by higher prio  │
  │ - Can sleep (acquire mutexes) │
  │ - RT task NOT blocked         │
  └──────────────────────────────┘
```

### 4.2 Viewing and Tuning IRQ Threads on Orin Nano

```bash
# List all IRQ threads and their priorities
ps -eo pid,cls,rtprio,ni,comm | grep "irq/"
# PID  CLS RTPRIO  NI  COMM
#  45   FF     50   -  irq/35-nvgpu
#  52   FF     50   -  irq/42-tegra-i2c
#  58   FF     50   -  irq/55-mmc0
#  61   FF     50   -  irq/68-xhci_hcd

# Default: all IRQ threads get SCHED_FIFO priority 50

# View IRQ thread to interrupt mapping
cat /proc/interrupts | head -20
#            CPU0  CPU1  CPU2  CPU3  CPU4  CPU5
# 35:       12458     0     0     0     0     0  GICv3  35  nvgpu
# 42:         803     0     0     0     0     0  GICv3  42  tegra-i2c
# 55:        1205     0     0     0     0     0  GICv3  55  mmc0

# Change IRQ thread priority
# Make GPU IRQ lower priority than your RT inference thread
sudo chrt -f -p 40 $(pgrep -f "irq/35-nvgpu")

# Make a critical sensor IRQ higher priority
sudo chrt -f -p 85 $(pgrep -f "irq/42-tegra-i2c")

# Pin IRQ thread to specific CPU
sudo taskset -p 0x01 $(pgrep -f "irq/35-nvgpu")  # CPU 0 only
```

### 4.3 IRQ Priority Design for Jetson RT System

```
Priority Assignment Strategy for Orin Nano:

  Priority   Assignment                     Rationale
  ─────────────────────────────────────────────────────────────
  99         Watchdog/safety monitor         Must never be blocked
  95         Sensor data acquisition ISR     Time-critical sampling
  90         Control loop thread             Safety-critical actuation
  85         Camera capture IRQ thread       Frame timing
  80         Inference thread                Primary workload
  75         Pre/post-processing threads     Supporting RT workload
  50         Default IRQ threads             System interrupts
  40         GPU IRQ thread (nvgpu)          Can tolerate some delay
  30         Network IRQ threads             Non-RT traffic
  20         Storage IRQ threads             Background I/O
  1-10       Housekeeping threads            Logging, monitoring
  0          SCHED_OTHER (CFS)               Non-RT processes
```

```bash
#!/bin/bash
# irq-priority-setup.sh — Configure IRQ thread priorities on Orin Nano
set -e

echo "=== Configuring IRQ thread priorities ==="

# Helper: set priority of IRQ thread by name pattern
set_irq_prio() {
    local pattern="$1"
    local prio="$2"
    local pid
    pid=$(pgrep -f "irq/.*${pattern}" 2>/dev/null || true)
    if [ -n "${pid}" ]; then
        for p in ${pid}; do
            chrt -f -p "${prio}" "${p}"
            echo "  irq/${pattern} (PID ${p}) → SCHED_FIFO ${prio}"
        done
    fi
}

# Sensor and camera IRQs — high priority
set_irq_prio "tegra-i2c" 85
set_irq_prio "tegra-vi" 85
set_irq_prio "tegra-capture" 85

# GPU — below inference threads
set_irq_prio "nvgpu" 40

# Network — low priority
set_irq_prio "eth0" 30
set_irq_prio "xhci" 30

# Storage — lowest RT priority
set_irq_prio "nvme" 20
set_irq_prio "mmc" 20

echo "=== IRQ priorities configured ==="
```

### 4.4 Forced Threading vs Voluntary Threading

```bash
# PREEMPT_RT forces ALL IRQs into threads by default
# But some drivers register with IRQF_NO_THREAD flag

# Check which IRQs are still hardirq (not threaded):
cat /proc/interrupts  # Look at IRQ counts
# IRQs handled in hardirq context won't have irq/ threads

# Force ALL IRQs to be threaded (even IRQF_NO_THREAD):
# Boot parameter: threadirqs
# Add to extlinux.conf APPEND line:
#   threadirqs

# Verify threading:
ls /proc/irq/*/threaded 2>/dev/null
# Each IRQ directory has a 'threaded' file if threading is supported

# CAUTION: Some IRQs CANNOT be threaded:
#   - Timer interrupt (arch timer)
#   - IPI (Inter-Processor Interrupts)
#   - NMI-like interrupts
# These use raw_spinlock and remain in hardirq context
# This is by design — they are the minimum non-preemptible set
```

---

## 5. Lock Primitives Under RT

### 5.1 Lock Transformation Table

```
Standard Kernel Lock        PREEMPT_RT Equivalent     Behavior Change
─────────────────────────────────────────────────────────────────────────
spinlock_t                  rt_mutex                  Sleeping, preemptible,
  spin_lock()                                         priority inheritance
  spin_lock_irqsave()

rwlock_t                    rt_mutex (readers may     Sleeping, PI,
  read_lock()               run concurrently but      writer priority
  write_lock()              writers are exclusive)     inheritance

raw_spinlock_t              raw_spinlock_t            UNCHANGED — true
  raw_spin_lock()           (same as standard          busy-wait spinlock,
                             spinlock)                 non-preemptible

mutex                       mutex                    UNCHANGED — already
                                                      sleeping lock

semaphore                   semaphore                UNCHANGED

rw_semaphore                rw_semaphore             UNCHANGED

local_irq_disable()         preempt_disable()        Does NOT actually
local_irq_save()            + local_lock             disable interrupts!
                                                      Just disables
                                                      preemption locally

per-CPU data with           local_lock_t             New locking primitive
local_irq_disable()                                   for per-CPU data
```

### 5.2 Why Spinlock → RT Mutex Matters

```c
/* Standard kernel: spinlock disables preemption */
spinlock_t my_lock;

void some_driver_function(void)
{
    spin_lock(&my_lock);
    /* === CRITICAL SECTION ===
     * In standard kernel:
     *   - Preemption disabled
     *   - NO task can run on this CPU
     *   - If another CPU needs this lock → busy-wait
     *   - Duration of this section = DIRECTLY ADDED to
     *     worst-case latency of ALL RT tasks on this CPU
     *
     * In PREEMPT_RT:
     *   - This becomes rt_mutex_lock()
     *   - The task CAN be preempted (by higher priority)
     *   - If another task needs this lock → it sleeps
     *   - Lock holder gets priority boost if needed (PI)
     *   - NO contribution to worst-case latency
     *     (unless this task IS the highest priority)
     */
    do_some_work();
    spin_unlock(&my_lock);
}
```

### 5.3 raw_spinlock — The Last Non-Preemptible Primitive

```c
/* raw_spinlock: remains a true spinlock even under PREEMPT_RT
 * Use ONLY when:
 *   1. Code runs in NMI/hardirq context that cannot sleep
 *   2. Code is in the scheduler itself (can't sleep while scheduling)
 *   3. Code touches hardware registers that must be atomic
 *   4. Critical section is EXTREMELY short (< 1 µs)
 */

#include <linux/spinlock.h>

static DEFINE_RAW_SPINLOCK(hw_reg_lock);

void access_hardware_register(void)
{
    unsigned long flags;
    raw_spin_lock_irqsave(&hw_reg_lock, flags);
    /* MUST be very short — this is non-preemptible */
    writel(value, hw_register_addr);
    readl(hw_register_addr);  /* read-back for posted write */
    raw_spin_unlock_irqrestore(&hw_reg_lock, flags);
}

/* Rule of thumb for Orin Nano kernel modules:
 *   Default: use spinlock_t (becomes RT mutex under PREEMPT_RT)
 *   Exception: use raw_spinlock_t ONLY for hardware register access
 *              with critical sections < 1 µs
 *
 * Count in kernel 5.15: ~10,000 spinlock_t vs ~50 raw_spinlock_t
 */
```

### 5.4 local_lock — Per-CPU Data Under RT

```c
/* Problem: per-CPU data was traditionally protected by
 * local_irq_disable(). Under PREEMPT_RT, that doesn't
 * actually disable interrupts — it just disables preemption.
 * This is insufficient for per-CPU data if IRQ threads
 * also access the data.
 *
 * Solution: local_lock_t — a per-CPU sleeping lock
 */

#include <linux/local_lock.h>

static DEFINE_PER_CPU(struct my_percpu_data, pcpu_data);
static DEFINE_PER_CPU(local_lock_t, pcpu_lock);

void update_percpu_data(void)
{
    local_lock(&pcpu_lock);
    /* Safe to access this_cpu_ptr(&pcpu_data) */
    /* Under PREEMPT_RT: sleepable, preemptible by higher priority */
    /* Under standard kernel: maps to preempt_disable() */
    struct my_percpu_data *data = this_cpu_ptr(&pcpu_data);
    data->counter++;
    local_unlock(&pcpu_lock);
}
```

---

## 6. Priority Inversion and Priority Inheritance

### 6.1 The Priority Inversion Problem

```
Classic priority inversion scenario:

  Three tasks: H (high priority), M (medium), L (low)
  L holds Lock X

  Time →
  ──────────────────────────────────────────────────

  L:  ████ [holds X] ░░░░░░░░░░░░░░░░░░░████ [releases X]
  M:  ░░░░░░░░░░░░░░░████████████████████░░░░
  H:  ░░░░░░░░████ [blocks on X] ░░░░░░░░░░░░████

  What happened:
  1. L acquires Lock X, starts running
  2. H wakes up, preempts L, tries to acquire Lock X → BLOCKED
  3. M wakes up, preempts L (M > L priority)
  4. M runs to completion
  5. L finally resumes, releases Lock X
  6. H finally runs

  Result: H was blocked by M, even though H > M in priority
  H's effective priority was reduced to L's level
  This is PRIORITY INVERSION — unbounded if many M-level tasks exist

  Real-world consequence: Mars Pathfinder reset bug (1997)
  The Sojourner rover experienced system resets due to priority
  inversion between the bus management task and meteorological task.
```

### 6.2 Priority Inheritance (PI) Protocol

```
With Priority Inheritance (enabled by default under PREEMPT_RT):

  L holds Lock X. H blocks on Lock X.
  → L's priority is BOOSTED to H's priority
  → M cannot preempt L (L is now running at H's priority)

  Time →
  ──────────────────────────────────────────────────

  L:  ████ [holds X, boosted to H] ████ [releases X, back to L]
  M:  ░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░████████████
  H:  ░░░░░░░░████ [blocks on X] ░████████████████

  What happened:
  1. L acquires Lock X
  2. H blocks on Lock X → L inherits H's priority
  3. M wakes up but CANNOT preempt L (L now has H's priority)
  4. L completes critical section, releases Lock X
  5. L's priority drops back to normal
  6. H runs immediately (highest priority)
  7. M runs after H finishes

  Result: H's blocking time = L's critical section length (bounded)
  No unbounded priority inversion
```

### 6.3 Priority Inheritance in User Space

```c
#include <pthread.h>

/* Create a mutex with priority inheritance */
pthread_mutex_t pi_mutex;
pthread_mutexattr_t attr;

void init_pi_mutex(void)
{
    pthread_mutexattr_init(&attr);

    /* Enable priority inheritance protocol */
    pthread_mutexattr_setprotocol(&attr, PTHREAD_PRIO_INHERIT);

    /* PTHREAD_PRIO_INHERIT: holder inherits blocker's priority
     * PTHREAD_PRIO_PROTECT: holder runs at mutex ceiling priority
     * PTHREAD_PRIO_NONE:    no priority adjustment (default, DANGEROUS for RT)
     */

    pthread_mutex_init(&pi_mutex, &attr);
    pthread_mutexattr_destroy(&attr);
}

/* Usage in RT application */
void* high_priority_thread(void* arg)
{
    struct sched_param param = { .sched_priority = 80 };
    pthread_setschedparam(pthread_self(), SCHED_FIFO, &param);

    while (running) {
        pthread_mutex_lock(&pi_mutex);
        /* If a lower-priority thread holds this mutex,
         * that thread will be boosted to priority 80
         * until it releases the mutex */
        access_shared_resource();
        pthread_mutex_unlock(&pi_mutex);
    }
    return NULL;
}

void* low_priority_thread(void* arg)
{
    struct sched_param param = { .sched_priority = 20 };
    pthread_setschedparam(pthread_self(), SCHED_FIFO, &param);

    while (running) {
        pthread_mutex_lock(&pi_mutex);
        /* If high_priority_thread blocks on this mutex,
         * THIS thread gets boosted to priority 80 */
        update_shared_resource();
        pthread_mutex_unlock(&pi_mutex);
    }
    return NULL;
}
```

### 6.4 Detecting Priority Inversion on Orin Nano

```bash
# Enable PI debugging in kernel
# CONFIG_DEBUG_RT_MUTEXES=y (already set in rt-debug.cfg)

# Check for PI boosting events in kernel log
dmesg | grep -i "priority\|PI\|boost"

# Trace PI events with ftrace
echo 1 > /sys/kernel/debug/tracing/events/lock/contention_begin/enable
echo 1 > /sys/kernel/debug/tracing/events/lock/contention_end/enable
cat /sys/kernel/debug/tracing/trace_pipe | grep -i "contention"

# Check lock statistics (if CONFIG_LOCK_STAT=y)
cat /proc/lock_stat | head -50
# Shows: lock contention counts, wait times, hold times
# High contention = potential PI issue

# View PI chain of a specific mutex
# (requires CONFIG_DEBUG_RT_MUTEXES)
cat /proc/<pid>/status | grep -i "pi\|prio"
```

### 6.5 Priority Ceiling Protocol (Alternative to PI)

```c
/* Priority ceiling: mutex has a fixed ceiling priority
 * When ANY thread acquires the mutex, its priority is
 * immediately raised to the ceiling — no PI chain needed
 *
 * Advantage: no runtime PI computation overhead
 * Disadvantage: must know maximum priority a priori
 */

pthread_mutex_t ceiling_mutex;
pthread_mutexattr_t ceil_attr;

void init_ceiling_mutex(int ceiling_prio)
{
    pthread_mutexattr_init(&ceil_attr);
    pthread_mutexattr_setprotocol(&ceil_attr, PTHREAD_PRIO_PROTECT);
    pthread_mutexattr_setprioceiling(&ceil_attr, ceiling_prio);
    pthread_mutex_init(&ceiling_mutex, &ceil_attr);
    pthread_mutexattr_destroy(&ceil_attr);
}

/* When to use ceiling vs inheritance:
 *
 * Use PTHREAD_PRIO_INHERIT when:
 *   - Multiple mutexes with different priority levels
 *   - Ceiling priority is not known in advance
 *   - Dynamic task set
 *
 * Use PTHREAD_PRIO_PROTECT when:
 *   - Fixed, known set of tasks and mutexes
 *   - Want zero runtime overhead for PI chain computation
 *   - Simpler to analyze for WCET
 */
```

---

## 7. RT Scheduling Deep Dive

### 7.1 SCHED_FIFO Internals

```
SCHED_FIFO (First-In, First-Out Real-Time Scheduling):

  Per-priority run queue:
  ┌─────────────────────────────┐
  │ Priority 99: [Task A]       │  ← highest priority
  │ Priority 98: []             │
  │ Priority 90: [Task B, C]    │  ← FIFO order within priority
  │ ...                         │
  │ Priority 80: [Task D]       │
  │ Priority 50: [IRQ threads]  │
  │ Priority  1: [Task E]       │
  └─────────────────────────────┘

  Rules:
  1. Highest priority runnable task ALWAYS runs
  2. Equal priority: FIFO order (first to become runnable runs first)
  3. A running task is NEVER preempted by equal-priority task
  4. A running task runs until:
     a. It blocks (I/O, mutex, sleep)
     b. It yields (sched_yield)
     c. A higher-priority task becomes runnable
  5. No time slicing — a priority 80 task runs FOREVER
     if nothing higher-priority exists

  Danger: A SCHED_FIFO bug (infinite loop at priority 99)
  will hang the system. Even the scheduler can't preempt it.
  Only a higher-priority task or NMI can interrupt it.
```

### 7.2 SCHED_RR vs SCHED_FIFO

```
SCHED_RR (Round-Robin Real-Time):

  Same as SCHED_FIFO EXCEPT:
  - Tasks at SAME priority level get time slices
  - Default quantum: 100 ms (configurable)

  Priority 80: [Task A ←100ms→ Task B ←100ms→ Task A ...]

  Check quantum:
  $ cat /proc/sys/kernel/sched_rr_timeslice_ms
  100

  When to use:
  SCHED_FIFO:  When each RT task has a UNIQUE priority
  SCHED_RR:    When multiple RT tasks share a priority level
               and must share CPU time fairly

  For Orin Nano inference: SCHED_FIFO is almost always correct
  (each pipeline stage should have a distinct priority)
```

### 7.3 RT Bandwidth Throttling

```bash
# Linux enforces RT bandwidth limits to prevent RT tasks
# from starving the system (starvation protection)

# Default: RT tasks can use 95% of CPU time per 1-second period
cat /proc/sys/kernel/sched_rt_period_us
# 1000000 (1 second)

cat /proc/sys/kernel/sched_rt_runtime_us
# 950000 (950 ms out of 1000 ms)

# This means: RT tasks are THROTTLED for 50ms every second
# The throttled time allows SCHED_OTHER tasks to run
# (prevents total system starvation)

# For a dedicated RT system, disable throttling:
echo -1 > /proc/sys/kernel/sched_rt_runtime_us
# WARNING: A SCHED_FIFO infinite loop will now lock the CPU forever
# Only do this on systems with proper watchdog and controlled workload

# Symptoms of RT throttling:
#   - cyclictest shows periodic latency spikes every ~1 second
#   - dmesg: "sched: RT throttling activated"
#   - RT task pauses for exactly 50ms periodically

# Verify throttling is happening:
cat /proc/sched_debug | grep -A5 "rt_throttled"

# For Orin Nano inference:
# If your inference threads use > 95% CPU continuously,
# you WILL hit throttling. Solutions:
#   1. Disable throttling (echo -1 above)
#   2. Reduce RT thread CPU usage to < 95%
#   3. Use SCHED_DEADLINE (has its own bandwidth reservation)
```

---

## 8. SCHED_DEADLINE — Earliest Deadline First

### 8.1 SCHED_DEADLINE Fundamentals

```
SCHED_DEADLINE implements EDF (Earliest Deadline First) scheduling.
Instead of fixed priorities, each task declares:

  runtime:   How much CPU time the task needs per period
  deadline:  When the task must complete (relative to period start)
  period:    How often the task repeats

  SCHED_DEADLINE has HIGHER priority than SCHED_FIFO.
  A SCHED_DEADLINE task preempts ANY SCHED_FIFO task.

Example: 30 FPS inference pipeline
  period   = 33.33 ms (1/30 Hz)
  runtime  = 15 ms (actual execution time)
  deadline = 30 ms (must finish within 30 ms of period start)

  ┌──── period = 33.33ms ────┐┌──── period ────┐
  │ runtime │      idle      ││ runtime │ idle  │
  │◄──15ms─►│                ││◄──15ms─►│       │
  │  ▲ start         ▲ deadline        ▲       │
  └──────────────────────────┘└─────────────────┘

  EDF guarantee: if total utilization < 100%, all deadlines are met
  U = Σ(runtime_i / period_i) < 1.0
```

### 8.2 Using SCHED_DEADLINE on Orin Nano

```c
#define _GNU_SOURCE
#include <sched.h>
#include <sys/syscall.h>
#include <linux/sched.h>
#include <linux/types.h>
#include <unistd.h>
#include <stdint.h>
#include <string.h>
#include <stdio.h>

struct sched_attr {
    uint32_t size;
    uint32_t sched_policy;
    uint64_t sched_flags;
    int32_t  sched_nice;
    uint32_t sched_priority;
    uint64_t sched_runtime;    /* nanoseconds */
    uint64_t sched_deadline;   /* nanoseconds */
    uint64_t sched_period;     /* nanoseconds */
};

static int sched_setattr(pid_t pid, const struct sched_attr *attr,
                         unsigned int flags)
{
    return syscall(SYS_sched_setattr, pid, attr, flags);
}

/* Set SCHED_DEADLINE for 30 FPS inference loop */
int set_deadline_30fps(void)
{
    struct sched_attr attr;
    memset(&attr, 0, sizeof(attr));
    attr.size = sizeof(attr);
    attr.sched_policy = SCHED_DEADLINE;
    attr.sched_runtime  = 15000000;   /* 15 ms in ns */
    attr.sched_deadline = 30000000;   /* 30 ms in ns */
    attr.sched_period   = 33333333;   /* 33.33 ms in ns (30 Hz) */

    int ret = sched_setattr(0, &attr, 0);
    if (ret < 0) {
        perror("sched_setattr SCHED_DEADLINE");
        /* Common failures:
         * EPERM:  need CAP_SYS_NICE or root
         * EBUSY:  utilization would exceed 100%
         * EINVAL: runtime > deadline or deadline > period
         */
        return -1;
    }
    return 0;
}

/* Inference loop with SCHED_DEADLINE */
void* deadline_inference_loop(void* arg)
{
    set_deadline_30fps();

    while (running) {
        /* Do inference work — must complete within runtime budget */
        preprocess_frame();
        run_inference();
        postprocess_results();

        /* Yield remaining budget — tell scheduler we're done early */
        sched_yield();
        /* sched_yield() in SCHED_DEADLINE:
         * "I'm done for this period" — task sleeps until next period
         * This is different from SCHED_FIFO where yield just
         * moves task to back of same-priority queue
         */
    }
    return NULL;
}
```

### 8.3 SCHED_DEADLINE vs SCHED_FIFO for Inference

```
Scenario: Camera capture (30 Hz) → Inference → Actuation

SCHED_FIFO approach:
  + Simple: assign priority, done
  + Well-understood
  - Must manually choose priorities
  - No overrun detection
  - Priority assignment becomes complex with many tasks
  - Priority inversion possible (even with PI)

SCHED_DEADLINE approach:
  + Automatic: kernel guarantees deadline if utilization fits
  + Built-in overrun detection (CBS — Constant Bandwidth Server)
  + No priority inversion (EDF is optimal for uniprocessor)
  + System tells you if workload is schedulable (EBUSY on admission)
  - More complex to configure (runtime, deadline, period)
  - Harder to reason about multi-core behavior
  - Cannot use pthread mutexes between DEADLINE tasks easily
  - Some libraries/frameworks assume SCHED_FIFO

Recommendation for Orin Nano:
  Use SCHED_FIFO for simple pipelines (1-3 RT threads)
  Use SCHED_DEADLINE for complex multi-rate systems
    (e.g., 30Hz camera + 100Hz IMU + 10Hz LiDAR + 50Hz control)
```

### 8.4 SCHED_DEADLINE Admission Control

```bash
# SCHED_DEADLINE has built-in admission control
# The kernel rejects tasks if total utilization would exceed limit

# Check current DEADLINE bandwidth reservation
cat /proc/sys/kernel/sched_dl_runtime_us
# 950000

cat /proc/sys/kernel/sched_dl_period_us
# 1000000

# This means: DEADLINE tasks can use 95% of each CPU

# Example admission calculation for Orin Nano (6 cores):
# Task 1: runtime=15ms, period=33ms → U=0.45 (one core)
# Task 2: runtime=5ms,  period=10ms → U=0.50 (one core)
# Task 3: runtime=2ms,  period=20ms → U=0.10 (one core)
# Total U = 1.05 → ERROR: exceeds single-core capacity

# Solutions:
# 1. Pin tasks to different cores (each core has independent U limit)
# 2. Reduce runtime budgets
# 3. Increase periods (lower rate)

# Check current DEADLINE task utilization
cat /proc/sched_debug | grep -A20 "dl_"
```

---

## 9. CPU Isolation Architecture

### 9.1 Complete CPU Isolation on Orin Nano

```
Orin Nano CPU topology:
  Core 0 ─┐
  Core 1 ─┤ Cluster 0 (shared L2 cache)
  Core 2 ─┤
  Core 3 ─┘
  Core 4 ─┐ Cluster 1 (shared L2 cache)
  Core 5 ─┘

Recommended isolation strategy:
  Cores 0-3: System (OS, drivers, IRQs, network, storage)
  Core 4:    Primary RT task (inference)
  Core 5:    Secondary RT task (sensor processing)

Why cores 4-5:
  - Separate L2 cache from system cores → no cache contention
  - Two cores in same cluster → fast inter-core communication
  - System activity on cores 0-3 doesn't pollute L2
```

### 9.2 Boot Parameters for Full Isolation

```bash
# /boot/extlinux/extlinux.conf — APPEND line additions:

isolcpus=managed_irq,domain,4,5
nohz_full=4,5
rcu_nocbs=4,5
rcu_nocb_poll
irqaffinity=0-3
nosoftlockup
tsc=reliable
nowatchdog

# Parameter explanation:
#
# isolcpus=managed_irq,domain,4,5
#   managed_irq: kernel won't assign managed IRQs to cores 4,5
#   domain:      cores 4,5 excluded from scheduler load balancing
#   Result:      NO tasks migrate to cores 4,5 unless explicitly pinned
#
# nohz_full=4,5
#   Tickless mode on cores 4,5
#   When only ONE task runs on core 4: timer tick is STOPPED
#   No periodic interruption from scheduler tick (250Hz or 1000Hz)
#   Eliminates 1-4 µs jitter every 1-4 ms
#
# rcu_nocbs=4,5
#   RCU callbacks offloaded from cores 4,5 to other cores
#   RCU callbacks can take 10-100 µs — removed from RT cores
#
# rcu_nocb_poll
#   RCU callback threads use polling instead of waking on cores 0-3
#   Prevents cross-core IPI wakeups that could affect timing
#
# irqaffinity=0-3
#   Default IRQ affinity set to cores 0-3
#   New device interrupts will be assigned to system cores only
#
# nosoftlockup
#   Disable soft lockup detector on isolated cores
#   The detector fires if a core doesn't schedule for 20 seconds
#   An isolated core running one RT task will trigger this falsely
#
# nowatchdog
#   Disable NMI watchdog (perf-based)
#   NMI watchdog causes ~1 µs periodic interruption
```

### 9.3 Verifying Isolation

```bash
# Verify cores are isolated
cat /sys/devices/system/cpu/isolated
# Expected: 4,5

# Verify nohz_full is active
cat /sys/devices/system/cpu/nohz_full
# Expected: 4,5

# Verify RCU offloading
ps aux | grep "rcu" | grep -v grep
# rcuop/4 and rcuop/5 should exist (offloaded callback threads)
# rcuog/* threads handle offloaded callbacks

# Verify no timer tick on isolated core
# Run a single task on core 4 and check /proc/interrupts
taskset -c 4 stress-ng --cpu 1 --timeout 10s &
sleep 2
cat /proc/interrupts | grep "arch_timer" | awk '{print "CPU4:", $6}'
# Timer count should be nearly static (not incrementing)
# Compare with non-isolated core:
cat /proc/interrupts | grep "arch_timer" | awk '{print "CPU0:", $2}'
# CPU0 timer count increments rapidly

# Verify no workqueues on isolated cores
cat /sys/bus/workqueue/devices/*/cpumask
# None should include bits for cores 4,5

# Check what's actually running on core 4
ps -eo pid,psr,comm | awk '$2==4'
# Should be empty or only your pinned RT task
```

### 9.4 Residual OS Activity on "Isolated" Cores

```
Even with full isolation, some kernel activity CANNOT be removed
from isolated cores. Understanding these residuals is critical
for worst-case latency analysis.

Residual Activity          Duration     Frequency         Removable?
──────────────────────────────────────────────────────────────────────
IPI (Inter-Proc Interrupt)  < 1 µs      On TLB flush      No
Arch timer (if nohz fails)  < 1 µs      1000 Hz           Mostly*
Kernel thread migration     < 5 µs      Rare               Mostly
RCU quiescent state         < 1 µs      Periodic           Mostly**
Page fault (first touch)    5-50 µs     Once per page      Pre-fault
Hardware PMU interrupt       < 1 µs      If perf enabled    Disable perf
TrustZone / OP-TEE call    1-10 µs     Rare               No
BPMP IPC                    < 5 µs      On power events    No

* nohz_full stops tick when exactly 1 runnable task on core
  If 0 or 2+ tasks: tick resumes
** rcu_nocbs offloads callbacks but core must still report
   quiescent states periodically

Practical worst case on tuned Orin Nano:
  ~10-20 µs from non-removable residual activity
  This sets the FLOOR for achievable worst-case latency
```

---

## 10. ARM Cortex-A78AE RT Specifics

### 10.1 Cortex-A78AE Architecture for RT

```
The Cortex-A78AE in Orin Nano is specifically designed for
automotive/edge applications with safety features:

Key Features:
  - Split-Lock mode: cores can run in lock-step for redundancy
  - Dual-core lock-step: two cores execute same instruction stream
  - ECC on all RAM (L1, L2, TLB, branch predictor)
  - ARMv8.2-A architecture with RAS (Reliability, Availability, Serviceability)

Memory Hierarchy (per core):
  L1I cache:    64 KB, 4-way set associative
  L1D cache:    64 KB, 4-way set associative
  L2 cache:     256 KB per core (shared within cluster on Orin)
  L3/SLC:       Shared system-level cache (if present)

Cache Latency Impact on RT:
  L1 hit:    ~1 ns (1 cycle at 1.5 GHz)
  L2 hit:    ~5 ns (5-7 cycles)
  L3/SLC:    ~15 ns
  DRAM:      ~50-80 ns (LPDDR5)

  A single L1 miss → L2 hit adds 4 ns
  A DRAM access adds 50-80 ns
  Cache thrashing from system activity on adjacent cores
  can add microseconds to RT task execution
```

### 10.2 Cache Partitioning (MPAM)

```
ARM Memory Partitioning and Monitoring (MPAM) allows
partitioning cache between RT and non-RT workloads.

T234 MPAM Support:
  - Hardware: Cortex-A78AE supports MPAM (ARMv8.4 feature)
  - Software: Linux MPAM support is evolving (patches available)
  - JetPack: Not officially enabled yet (as of JetPack 6.x)

Without MPAM, use cluster isolation instead:
  Cores 0-3 (Cluster 0): system workload → uses Cluster 0 L2
  Cores 4-5 (Cluster 1): RT workload → uses Cluster 1 L2
  NO cross-cluster L2 contention

With MPAM (when available):
  Partition L3/SLC: 75% for RT, 25% for system
  Partition memory bandwidth: reserve for RT
  Monitor: track cache occupancy per partition
```

### 10.3 ARM Architectural Timer

```
The ARM Generic Timer provides the primary clock source for RT:

  Counter frequency:  Typically 31.25 MHz on T234 (varies)
  Resolution:         32 ns per tick
  Access:             Direct register read (CNTVCT_EL0)
  Overhead:           < 10 ns for a timestamp read
  Monotonic:          Yes (never goes backward)
  Invariant:          Frequency does not change with DVFS

Reading the timer directly (fastest possible timestamp):

  static inline uint64_t arm_timer_read(void)
  {
      uint64_t val;
      asm volatile("mrs %0, cntvct_el0" : "=r" (val));
      return val;
  }

  static inline uint64_t arm_timer_freq(void)
  {
      uint64_t val;
      asm volatile("mrs %0, cntfrq_el0" : "=r" (val));
      return val;
  }

  /* Convert ticks to nanoseconds */
  static inline uint64_t ticks_to_ns(uint64_t ticks, uint64_t freq)
  {
      return (ticks * 1000000000ULL) / freq;
  }

This is the lowest-overhead timing available on Orin Nano.
clock_gettime(CLOCK_MONOTONIC_RAW) ultimately reads this register
but adds ~50 ns of syscall overhead (vDSO optimization reduces this).
```

### 10.4 Branch Prediction and Speculative Execution Impact

```
Speculative execution can cause RT timing variability:

Sources of speculation-induced jitter:
  - Branch misprediction: ~15 cycle penalty (~10 ns)
  - Speculative memory loads that miss cache
  - Speculative TLB walks
  - Store buffer drain on barrier instructions

Mitigation (for hard RT paths):
  1. Avoid complex branching in hot loops
  2. Use branchless programming where possible:
     result = condition ? a : b;  // may branch
     result = a + (b - a) * condition;  // branchless

  3. Use dmb/dsb/isb barriers carefully:
     asm volatile("dsb sy" ::: "memory");  // full barrier
     // dsb drains store buffer — adds 10-50 ns

  4. For truly deterministic paths, consider DLA:
     DLA has no speculative execution → deterministic timing
     (See Real-Time Inference guide Section 5)
```

---

## 11. GICv3 Interrupt Controller Tuning

### 11.1 GICv3 Architecture on T234

```
The GIC (Generic Interrupt Controller) v3 on T234:

  GIC Distributor (GICD):
    - Receives all interrupt sources
    - Routes interrupts to CPU interfaces
    - Manages interrupt priority and grouping
    - One per SoC

  GIC Redistributor (GICR):
    - One per CPU core
    - Handles SGIs and PPIs (per-core interrupts)
    - Manages interrupt enable/disable per core

  GIC CPU Interface:
    - System registers (ICC_*_EL1)
    - Acknowledges, signals EOI (End of Interrupt)
    - Priority masking per core

  Interrupt Types:
    SPI: Shared Peripheral Interrupt (device → any core)
    PPI: Private Peripheral Interrupt (per-core, e.g., timer)
    SGI: Software Generated Interrupt (IPI, cross-core signal)
    LPI: Locality-specific Peripheral Interrupt (MSI/MSI-X for PCIe)
```

### 11.2 Interrupt Routing for RT

```bash
# Route ALL SPIs away from RT cores (4,5)

# Method 1: Boot parameter (recommended)
# irqaffinity=0-3
# Sets default SMP affinity for all new interrupts

# Method 2: Runtime per-IRQ affinity
# Move specific interrupts to specific cores

# List current interrupt routing
for i in $(ls /proc/irq/ | grep -E '^[0-9]+$'); do
    name=$(cat /proc/irq/$i/spurious 2>/dev/null | head -1 || echo "?")
    affinity=$(cat /proc/irq/$i/smp_affinity_list 2>/dev/null || echo "?")
    echo "IRQ $i: affinity=$affinity"
done

# Critical: NVMe interrupt on Orin Nano
# NVMe uses MSI-X → generates LPIs → routed by ITS
# Check NVMe IRQ affinity:
cat /proc/irq/$(cat /proc/interrupts | grep nvme | awk '{print $1}' | tr -d :)/smp_affinity_list

# Route NVMe to cores 0-3 only
echo 0-3 > /proc/irq/$(cat /proc/interrupts | grep nvme | head -1 | awk '{print $1}' | tr -d :)/smp_affinity_list

# Route GPU (nvgpu) to core 0
echo 0 > /proc/irq/$(cat /proc/interrupts | grep nvgpu | awk '{print $1}' | tr -d :)/smp_affinity_list
```

### 11.3 Interrupt Latency Measurement

```bash
# Measure interrupt-to-handler latency

# Method 1: /proc/irq statistics
cat /proc/irq/<N>/spurious
# Shows: count, unhandled count, last unhandled timestamp

# Method 2: ftrace interrupt tracing
echo 1 > /sys/kernel/debug/tracing/events/irq/irq_handler_entry/enable
echo 1 > /sys/kernel/debug/tracing/events/irq/irq_handler_exit/enable
cat /sys/kernel/debug/tracing/trace_pipe
# Shows entry/exit timestamps for each IRQ → compute handler duration

# Method 3: GPIO loopback for external measurement
# Connect GPIO output to GPIO input via wire
# Toggle output from RT thread, measure response time on input
# This measures: thread→GPIO→wire→GPIO→interrupt→thread latency
# Expect: 5-20 µs on tuned PREEMPT_RT system

# Method 4: oscilloscope measurement
# Pulse GPIO at start of inference, pulse at completion
# Measure on oscilloscope for accurate external timing
```

---

## 12. Latency Sources — Complete Taxonomy

### 12.1 Every Source of Latency on Orin Nano

```
Category 1: Hardware Latency (cannot be eliminated)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Source                      Typical          Worst Case
Cache miss (L1→L2)          4 ns             4 ns
Cache miss (L2→DRAM)        50 ns            80 ns
TLB miss                    10 ns            100 ns (page walk)
DRAM refresh                ~100 ns          ~300 ns (LPDDR5)
PCIe round-trip (NVMe)      1-2 µs           5 µs
Branch misprediction        10 ns            10 ns
Memory barrier (dsb)        10 ns            50 ns


Category 2: Kernel Latency (PREEMPT_RT reduces dramatically)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Source                      Without RT       With RT (tuned)
Interrupt handling          1-500 µs         < 1 µs (threaded)
Spinlock contention         1-100 µs         N/A (sleeping lock)
Softirq processing          10-500 µs        0 (threaded)
Scheduler tick              1-4 µs/tick      0 (nohz_full)
RCU callback                10-100 µs        0 (rcu_nocbs)
Workqueue processing        10-200 µs        0 (isolated CPU)
Timer tick processing       1-5 µs           0 (nohz_full)


Category 3: Memory Management Latency
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Source                      Typical          Avoidable?
Page fault (minor)          1-5 µs           Yes (prefault + mlock)
Page fault (major/disk)     1-10 ms          Yes (mlock)
THP compaction              100-5000 µs      Yes (disable THP)
SLUB allocation             0.5-5 µs         Yes (pre-allocate)
Page reclaim (kswapd)       100-10000 µs     Yes (mlock + no swap)
OOM killer                  1-100 ms         Yes (size memory)


Category 4: DVFS and Power Management
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Source                      Typical          Avoidable?
CPU frequency change        50-200 µs        Yes (performance gov)
GPU clock scaling            50-500 µs        Yes (jetson_clocks)
Thermal throttling          Ongoing           Partially (cooling)
Power state transition      100-1000 µs      Yes (disable idle)
EMC (memory) clock change   10-50 µs         Yes (lock clocks)


Category 5: NVIDIA-Specific Latency
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Source                      Typical          Avoidable?
GPU kernel launch           5-20 µs          Partially (CUDA Graphs)
GPU context switch          10-50 µs         Yes (exclusive context)
nvgpu driver lock           1-10 µs          No (driver internal)
CUDA memory copy (H2D)      5-100 µs         Yes (pinned memory)
TensorRT enqueue            5-15 µs          No (framework overhead)
DMA transfer setup          2-10 µs          No
Display vsync blocking      0-16.67 ms       Yes (no display in RT)
```

### 12.2 Latency Budget Example

```
Target: 30 FPS inference with 20 ms end-to-end budget

Component               Budget      Measured p99    Status
──────────────────────────────────────────────────────────
Camera capture (V4L2)    3.0 ms      2.8 ms         ✓
DMA: sensor → memory     0.5 ms      0.3 ms         ✓
Pre-processing (CPU)     2.0 ms      1.5 ms         ✓
H2D memory copy          0.5 ms      0.2 ms         ✓
TensorRT inference       8.0 ms      7.2 ms         ✓
D2H memory copy          0.3 ms      0.1 ms         ✓
Post-processing (CPU)    1.0 ms      0.8 ms         ✓
Actuator command (CAN)   0.5 ms      0.3 ms         ✓
OS/scheduling overhead   0.5 ms      0.1 ms (RT)    ✓
Margin                   3.7 ms      —              ✓
──────────────────────────────────────────────────────────
Total                   20.0 ms     13.3 ms         ✓

Without PREEMPT_RT, "OS/scheduling overhead" can spike
to 1-5 ms, eating into margin and occasionally missing 20 ms.
```

---

## 13. The rt-tests Suite

### 13.1 Overview

```
rt-tests is the standard benchmark suite for Linux RT performance.
It provides multiple tools that test different aspects of RT behavior.

Installation:
  # From package manager
  sudo apt install rt-tests

  # Or build from source (latest version)
  git clone git://git.kernel.org/pub/scm/utils/rt-tests/rt-tests.git
  cd rt-tests
  make
  sudo make install
```

### 13.2 cyclictest — The Gold Standard

```bash
# cyclictest: measures timer wakeup latency
# It creates high-priority threads that sleep for a specified interval,
# then measures the actual wakeup time vs expected wakeup time.
# The difference = scheduling + interrupt + kernel latency.

# Basic test (single thread, 1ms interval, 100k samples)
sudo cyclictest -t1 -p 90 -n -i 1000 -l 100000

# Comprehensive test (one thread per CPU, 100µs interval)
sudo cyclictest -t6 -p 90 -n -i 100 -l 1000000 -a 0-5

# Parameters:
#   -t6         6 threads (one per core)
#   -p 90       SCHED_FIFO priority 90
#   -n          use clock_nanosleep (not busy-wait)
#   -i 100      100 µs interval
#   -l 1000000  1 million iterations
#   -a 0-5      pin threads to cores 0-5
#   -m          lock memory (mlockall)
#   -h 100      histogram with 100 µs max
#   --duration=1h  run for 1 hour

# Production RT validation test (run overnight):
sudo cyclictest \
    -t1 \
    -p 95 \
    -n \
    -i 1000 \
    -l 0 \
    --duration=12h \
    -a 4 \
    -m \
    -h 200 \
    --histfile=cyclictest_results.txt

# Output interpretation:
# T: 0 ( 1234) P:95 I:1000 C: 100000 Min:   2 Act:   4 Avg:   5 Max:  18
#
# T: 0         Thread 0
# (1234)       Thread PID
# P:95         SCHED_FIFO priority 95
# I:1000       Interval 1000 µs (1 ms)
# C:100000     Completed 100000 cycles
# Min: 2       Minimum latency 2 µs
# Act: 4       Last measured latency 4 µs
# Avg: 5       Average latency 5 µs
# Max: 18      Maximum latency 18 µs ← THIS IS THE KEY METRIC

# Acceptable values for Orin Nano (PREEMPT_RT + tuned):
#   Avg: 3-8 µs
#   Max: < 50 µs (good), < 100 µs (acceptable), > 200 µs (investigate)
```

### 13.3 cyclictest Under Load (Stress Test)

```bash
# The real test: cyclictest while system is under stress
# Latency under idle conditions is meaningless for production

# Terminal 1: Generate system stress
stress-ng --cpu 4 --io 2 --vm 2 --vm-bytes 1G \
    --timeout 3600 --metrics &

# Terminal 2: Generate network stress
iperf3 -c <remote> -t 3600 -P 4 &

# Terminal 3: Generate disk stress
fio --name=randwrite --rw=randwrite --bs=4k --size=1G \
    --numjobs=4 --runtime=3600 --time_based &

# Terminal 4: Run cyclictest on isolated core
sudo cyclictest -t1 -p 95 -n -i 1000 -l 0 --duration=1h -a 4 -m

# Expected results on properly tuned Orin Nano:
#   Without stress:  Max ~15-25 µs
#   With stress:     Max ~25-50 µs (good isolation)
#   With stress, no isolation: Max ~200-2000 µs (poor)
```

### 13.4 hwlatdetect — Hardware Latency Detection

```bash
# hwlatdetect: detects latency caused by HARDWARE, not software
# It runs a tight loop on a CPU and measures any gaps
# that cannot be explained by software (SMI, TrustZone, firmware)

sudo hwlatdetect --duration=60 --threshold=10
# --threshold=10: report any gap > 10 µs

# Output:
# hwlatdetect:  test duration 60 seconds
#    parameters:
#         Latency threshold: 10us
#         Sample window:     1000000us
#         Sample width:      500000us
#         Non-sampling period: 500000us
#         Output File:       None
#
# Starting test
# test finished
# Max Latency: 8us
# Samples recorded: 0
# Samples exceeding threshold: 0

# If you see samples > 10 µs:
# Possible causes on Orin Nano:
#   1. BPMP firmware communication (power management)
#   2. OP-TEE / TrustZone operations
#   3. DRAM refresh interference (rare on LPDDR5)
#   4. Thermal throttle interrupt
#   5. PMU (Performance Monitor) NMI

# Run on isolated core specifically:
sudo hwlatdetect --duration=300 --threshold=5 --cpu=4
```

### 13.5 signaltest — Signal Delivery Latency

```bash
# signaltest: measures signal delivery latency between threads
# Important for applications using POSIX signals for synchronization

sudo signaltest -t1 -p 90 -l 100000
# Measures time from signal send to signal handler invocation

# On PREEMPT_RT: expect 3-10 µs average
# On standard kernel: expect 10-50 µs average
```

### 13.6 pip_stress — Priority Inheritance Stress Test

```bash
# pip_stress: tests priority inheritance under load
# Creates multiple threads at different priorities contending on locks

sudo pip_stress -t 5 -d 60
# -t 5:  5 threads at different priorities
# -d 60: run for 60 seconds

# Successful output means PI is working correctly
# Failure (hang or priority inversion detected) = kernel PI bug
```

### 13.7 pmqtest — POSIX Message Queue Latency

```bash
# pmqtest: measures latency of POSIX message queue communication
# Relevant for multi-process RT architectures

sudo pmqtest -t1 -p 90 -i 1000 -l 100000
# Measures send → receive latency via mq_send/mq_receive

# On PREEMPT_RT: expect 5-15 µs average
# Useful for validating IPC latency in multi-process inference pipelines
```

---

## 14. hwlatdetect — Hardware Latency Detection

### 14.1 Understanding Hardware-Induced Latency

```
Hardware latency = time stolen from the CPU by hardware/firmware
that the OS CANNOT control or prevent.

On Orin Nano, hardware latency sources:

1. BPMP (Boot and Power Management Processor)
   - Handles DVFS, power gating, clock management
   - CPU communicates with BPMP via IPC (mailbox)
   - During power state changes, BPMP may hold CPU
   - Typical: 1-5 µs, rare: 10-20 µs

2. OP-TEE / TrustZone (Secure World)
   - Secure monitor calls (SMC) switch to Secure World
   - CPU is unavailable during SMC handling
   - Triggered by: cryptographic operations, secure storage
   - Typical: 1-10 µs

3. DRAM Controller
   - LPDDR5 refresh cycles: ~100 ns per bank
   - All-bank refresh: ~300 ns (access blocked)
   - Frequency: every 3.9 µs (tREFI for LPDDR5)
   - Usually overlapped with other bank access
   - Worst case: multiple pending refreshes = 1-2 µs

4. Cache Maintenance Operations
   - Cache flush/invalidate initiated by other cores or DMA
   - Cross-core snooping: 5-20 ns per cache line
   - Full L2 flush: 50-200 µs (rare, on power transition)

5. Errata Workarounds
   - Some ARM errata require barriers or serialization
   - Adds 1-10 ns per occurrence
```

### 14.2 Measuring and Profiling Hardware Latency

```bash
# Comprehensive hardware latency test script for Orin Nano
#!/bin/bash
set -e

echo "=== Hardware Latency Analysis for Orin Nano ==="

# Test 1: hwlatdetect on each core
for cpu in 0 1 2 3 4 5; do
    echo "Testing CPU ${cpu}..."
    sudo hwlatdetect --duration=30 --threshold=5 --cpu=${cpu} \
        2>/dev/null | tail -3
done

# Test 2: Monitor BPMP activity during test
# BPMP communications show in tegrastats
sudo tegrastats --interval 100 &
TEGRA_PID=$!

# Test 3: Lock clocks to eliminate DVFS-induced latency
sudo jetson_clocks

# Test 4: Re-run hwlatdetect with clocks locked
echo ""
echo "With clocks locked:"
sudo hwlatdetect --duration=60 --threshold=3 --cpu=4

kill ${TEGRA_PID} 2>/dev/null

echo "=== Analysis complete ==="
```

---

## 15. Worst-Case Execution Time (WCET) Analysis

### 15.1 WCET Fundamentals

```
WCET = the MAXIMUM time a code section can take to execute,
considering ALL possible:
  - Input data combinations
  - Cache states (cold/warm/hot)
  - Pipeline states (branch prediction, speculation)
  - Memory access patterns (hits, misses, contention)
  - Interrupt interference
  - OS scheduling overhead

Why WCET matters:
  If your inference deadline is 20 ms and WCET is 25 ms,
  you WILL miss deadlines — guaranteed. No amount of
  "average case" measurement helps.

WCET approaches:
  1. Static analysis (formal, tool-based)
     - Analyzes binary code + hardware model
     - Provides proven upper bound
     - Tools: aiT (AbsInt), Bound-T, OTAWA
     - Very conservative (overestimates by 2-10×)
     - Difficult on modern OoO CPUs like A78AE

  2. Measurement-based (empirical)
     - Run code many times under worst-case conditions
     - Record maximum observed execution time (MOET)
     - Add safety margin: WCET_est = MOET × (1 + margin)
     - Typical margin: 20-50% depending on criticality
     - Cannot prove absence of worse case (unknown unknowns)

  3. Hybrid (measurement + static analysis)
     - Use measurements for most code
     - Use static analysis for critical code sections
     - Most practical for Jetson workloads
```

### 15.2 Measurement-Based WCET on Orin Nano

```c
#include <time.h>
#include <stdio.h>
#include <float.h>

/* WCET measurement framework */
typedef struct {
    double min_ms;
    double max_ms;
    double sum_ms;
    uint64_t count;
    double histogram[1000];  /* 1 µs bins, up to 1 ms */
} wcet_stats_t;

static inline double timespec_diff_ms(struct timespec *start,
                                       struct timespec *end)
{
    return (end->tv_sec - start->tv_sec) * 1000.0 +
           (end->tv_nsec - start->tv_nsec) / 1e6;
}

void wcet_init(wcet_stats_t *stats)
{
    stats->min_ms = DBL_MAX;
    stats->max_ms = 0.0;
    stats->sum_ms = 0.0;
    stats->count = 0;
    memset(stats->histogram, 0, sizeof(stats->histogram));
}

void wcet_record(wcet_stats_t *stats, double elapsed_ms)
{
    if (elapsed_ms < stats->min_ms) stats->min_ms = elapsed_ms;
    if (elapsed_ms > stats->max_ms) stats->max_ms = elapsed_ms;
    stats->sum_ms += elapsed_ms;
    stats->count++;

    /* Record in histogram (1 µs bins) */
    int bin = (int)(elapsed_ms * 1000.0);  /* ms → µs */
    if (bin >= 0 && bin < 1000) {
        stats->histogram[bin]++;
    }
}

void wcet_report(wcet_stats_t *stats, const char *name)
{
    printf("=== WCET Report: %s ===\n", name);
    printf("Samples:    %lu\n", stats->count);
    printf("Min:        %.3f ms\n", stats->min_ms);
    printf("Max (MOET): %.3f ms\n", stats->max_ms);
    printf("Average:    %.3f ms\n", stats->sum_ms / stats->count);
    printf("Est WCET:   %.3f ms (MOET × 1.3)\n", stats->max_ms * 1.3);

    /* Find p99, p99.9 from histogram */
    uint64_t p99_threshold = stats->count * 99 / 100;
    uint64_t p999_threshold = stats->count * 999 / 1000;
    uint64_t cumulative = 0;
    for (int i = 0; i < 1000; i++) {
        cumulative += (uint64_t)stats->histogram[i];
        if (cumulative >= p99_threshold) {
            printf("p99:        %.3f ms\n", i / 1000.0);
            break;
        }
    }
}

/* Usage in inference loop */
void run_wcet_analysis(int iterations)
{
    wcet_stats_t preprocess_stats, inference_stats, postprocess_stats;
    wcet_init(&preprocess_stats);
    wcet_init(&inference_stats);
    wcet_init(&postprocess_stats);

    struct timespec t0, t1, t2, t3;

    for (int i = 0; i < iterations; i++) {
        clock_gettime(CLOCK_MONOTONIC_RAW, &t0);
        preprocess();
        clock_gettime(CLOCK_MONOTONIC_RAW, &t1);
        inference();
        clock_gettime(CLOCK_MONOTONIC_RAW, &t2);
        postprocess();
        clock_gettime(CLOCK_MONOTONIC_RAW, &t3);

        wcet_record(&preprocess_stats, timespec_diff_ms(&t0, &t1));
        wcet_record(&inference_stats, timespec_diff_ms(&t1, &t2));
        wcet_record(&postprocess_stats, timespec_diff_ms(&t2, &t3));
    }

    wcet_report(&preprocess_stats, "Preprocess");
    wcet_report(&inference_stats, "Inference");
    wcet_report(&postprocess_stats, "Postprocess");
}
```

### 15.3 WCET Testing Conditions

```
For valid WCET measurement, test under WORST-CASE conditions:

1. Cache state: COLD
   - Flush caches before each measurement
   - Or: interleave with other workloads to pollute caches
   echo 3 > /proc/sys/vm/drop_caches  # flush page cache

2. Memory pressure
   - Allocate most of available RAM
   - Force the system into active page reclaim
   stress-ng --vm 1 --vm-bytes 6G &

3. I/O load
   - Saturate NVMe with random I/O
   fio --name=stress --rw=randwrite --bs=4k --size=2G --numjobs=4 &

4. CPU load on non-isolated cores
   stress-ng --cpu 4 --cpu-method=all &

5. Network traffic
   iperf3 -c <remote> -P 4 -t 3600 &

6. Thermal stress
   - Run GPU workload to raise temperature
   - Let thermal throttling engage
   - Measure WCET during thermal throttle

7. Power transitions
   - Vary input voltage if possible
   - Test during brownout recovery

Run for extended duration (8-24 hours) under combined stress.
The longest observed execution time across all test runs,
with a 30% margin, is your estimated WCET.
```

---

## 16. RT-Safe Kernel Module Development

### 16.1 Rules for RT-Safe Kernel Modules

```
Writing kernel modules for a PREEMPT_RT system requires
strict adherence to these rules:

Rule 1: Never disable preemption longer than 1 µs
  ✗ preempt_disable(); long_operation(); preempt_enable();
  ✓ Use sleeping locks (spinlock_t becomes rt_mutex)

Rule 2: Never call allocation in atomic/RT context
  ✗ kmalloc(size, GFP_ATOMIC) in spin_lock section
  ✓ Pre-allocate buffers in probe/init
  ✓ Use GFP_KERNEL outside critical sections

Rule 3: Use spinlock_t for general locking (becomes RT mutex)
  ✗ raw_spinlock_t for long critical sections
  ✓ raw_spinlock_t ONLY for < 1 µs hardware register access

Rule 4: IRQ handlers must be threaded
  ✓ request_threaded_irq(irq, NULL, handler, IRQF_ONESHOT, ...)
  ✗ request_irq(irq, handler, 0, ...) with long handler

Rule 5: No busy-wait loops
  ✗ while (!ready) cpu_relax();
  ✓ wait_event_interruptible(wq, ready);
  ✓ Or use completion: wait_for_completion(&done);

Rule 6: Use lock-free structures where possible
  ✓ kfifo (lock-free FIFO for single producer/consumer)
  ✓ atomic operations (atomic_t, atomic64_t)
  ✓ RCU for read-heavy data structures

Rule 7: Annotate sleeping context
  ✓ Use might_sleep() in functions that may sleep
  ✓ Helps lockdep detect incorrect usage
```

### 16.2 RT-Safe Driver Template

```c
/* rt_sensor_driver.c — Example RT-safe Jetson kernel module */

#include <linux/module.h>
#include <linux/platform_device.h>
#include <linux/interrupt.h>
#include <linux/kfifo.h>
#include <linux/completion.h>
#include <linux/spinlock.h>
#include <linux/ktime.h>
#include <linux/hrtimer.h>

#define FIFO_SIZE 256
#define SAMPLE_PERIOD_NS 1000000  /* 1 ms = 1 kHz sampling */

struct rt_sensor_dev {
    void __iomem *regs;
    int irq;

    /* Lock-free FIFO: ISR writes, userspace reads */
    DECLARE_KFIFO(sample_fifo, u32, FIFO_SIZE);

    /* For register access: raw_spinlock (very short, HW access) */
    raw_spinlock_t hw_lock;

    /* For data structure access: regular spinlock (becomes RT mutex) */
    spinlock_t data_lock;

    /* High-resolution timer for periodic sampling */
    struct hrtimer sample_timer;

    /* Pre-allocated DMA buffer (allocated at probe, not in RT path) */
    void *dma_buf;
    dma_addr_t dma_addr;

    struct completion sample_ready;
    bool running;
};

/* Threaded IRQ handler — runs as schedulable kernel thread */
static irqreturn_t rt_sensor_irq_thread(int irq, void *dev_id)
{
    struct rt_sensor_dev *dev = dev_id;
    u32 sample;

    /* Read hardware register (very short critical section) */
    raw_spin_lock(&dev->hw_lock);
    sample = readl(dev->regs + SENSOR_DATA_REG);
    writel(SENSOR_IRQ_ACK, dev->regs + SENSOR_STATUS_REG);
    raw_spin_unlock(&dev->hw_lock);

    /* Store in lock-free FIFO (no lock needed for single producer) */
    kfifo_put(&dev->sample_fifo, sample);

    /* Wake up any waiting readers */
    complete(&dev->sample_ready);

    return IRQ_HANDLED;
}

/* High-resolution timer callback for periodic sampling */
static enum hrtimer_restart rt_sensor_timer(struct hrtimer *timer)
{
    struct rt_sensor_dev *dev =
        container_of(timer, struct rt_sensor_dev, sample_timer);
    u32 sample;

    /* Read sensor (raw_spinlock for HW access) */
    raw_spin_lock(&dev->hw_lock);
    sample = readl(dev->regs + SENSOR_DATA_REG);
    raw_spin_unlock(&dev->hw_lock);

    /* Lock-free FIFO insert */
    kfifo_put(&dev->sample_fifo, sample);

    /* Re-arm timer */
    hrtimer_forward_now(timer, ns_to_ktime(SAMPLE_PERIOD_NS));
    return HRTIMER_RESTART;
}

/* Probe: all allocation happens here (NOT in RT path) */
static int rt_sensor_probe(struct platform_device *pdev)
{
    struct rt_sensor_dev *dev;

    /* Allocate device structure (GFP_KERNEL is fine in probe) */
    dev = devm_kzalloc(&pdev->dev, sizeof(*dev), GFP_KERNEL);
    if (!dev)
        return -ENOMEM;

    /* Pre-allocate DMA buffer */
    dev->dma_buf = dma_alloc_coherent(&pdev->dev, PAGE_SIZE,
                                       &dev->dma_addr, GFP_KERNEL);
    if (!dev->dma_buf)
        return -ENOMEM;

    /* Initialize locks */
    raw_spin_lock_init(&dev->hw_lock);
    spin_lock_init(&dev->data_lock);
    init_completion(&dev->sample_ready);
    INIT_KFIFO(dev->sample_fifo);

    /* Map registers */
    dev->regs = devm_platform_ioremap_resource(pdev, 0);
    if (IS_ERR(dev->regs))
        return PTR_ERR(dev->regs);

    /* Request THREADED IRQ (RT-safe: runs as kernel thread) */
    dev->irq = platform_get_irq(pdev, 0);
    return devm_request_threaded_irq(&pdev->dev, dev->irq,
                                      NULL,  /* no hardirq handler */
                                      rt_sensor_irq_thread,
                                      IRQF_ONESHOT,
                                      "rt-sensor", dev);
}

static struct platform_driver rt_sensor_driver = {
    .probe = rt_sensor_probe,
    .driver = {
        .name = "rt-sensor",
    },
};
module_platform_driver(rt_sensor_driver);
MODULE_LICENSE("GPL");
```

### 16.3 Verifying RT Safety With lockdep

```bash
# Enable lock debugging to catch RT-unsafe patterns
# CONFIG_PROVE_LOCKING=y
# CONFIG_DEBUG_LOCK_ALLOC=y
# CONFIG_DEBUG_RT_MUTEXES=y

# lockdep will warn about:
#   - spinlock held while sleeping (invalid under standard kernel,
#     OK under PREEMPT_RT due to conversion, but lockdep still warns
#     about raw_spinlock misuse)
#   - Lock ordering violations (potential deadlock)
#   - IRQ-safe lock taken in non-IRQ-safe context

# Check for warnings:
dmesg | grep -i "lockdep\|BUG\|WARNING.*lock"

# View lock dependency graph:
cat /proc/lockdep_stats
cat /proc/lockdep_chains | head -50
```

---

## 17. Ftrace for RT Latency Debugging

### 17.1 Ftrace RT Tracers

```bash
# Ftrace provides specialized tracers for RT latency analysis

# List available tracers
cat /sys/kernel/debug/tracing/available_tracers
# function function_graph preemptirqsoff preemptoff irqsoff wakeup
# wakeup_rt wakeup_dl nop

# === Tracer 1: irqsoff ===
# Traces the longest period with interrupts disabled
echo irqsoff > /sys/kernel/debug/tracing/current_tracer
echo 1 > /sys/kernel/debug/tracing/tracing_on
sleep 10
echo 0 > /sys/kernel/debug/tracing/tracing_on
cat /sys/kernel/debug/tracing/tracing_max_latency
# Shows maximum IRQ-off duration in microseconds
cat /sys/kernel/debug/tracing/trace
# Shows the call stack during the longest IRQ-off period

# === Tracer 2: preemptoff ===
# Traces the longest period with preemption disabled
echo preemptoff > /sys/kernel/debug/tracing/current_tracer
echo 1 > /sys/kernel/debug/tracing/tracing_on
sleep 10
echo 0 > /sys/kernel/debug/tracing/tracing_on
cat /sys/kernel/debug/tracing/tracing_max_latency
cat /sys/kernel/debug/tracing/trace

# === Tracer 3: preemptirqsoff ===
# Combines both — traces longest period with either disabled
echo preemptirqsoff > /sys/kernel/debug/tracing/current_tracer
echo 1 > /sys/kernel/debug/tracing/tracing_on
sleep 60
echo 0 > /sys/kernel/debug/tracing/tracing_on
cat /sys/kernel/debug/tracing/tracing_max_latency
cat /sys/kernel/debug/tracing/trace

# === Tracer 4: wakeup_rt ===
# Traces wakeup latency for the highest-priority RT task
echo wakeup_rt > /sys/kernel/debug/tracing/current_tracer
echo 1 > /sys/kernel/debug/tracing/tracing_on
# Run your RT application
sleep 30
echo 0 > /sys/kernel/debug/tracing/tracing_on
cat /sys/kernel/debug/tracing/tracing_max_latency
# Shows: time from task wakeup to actual execution start
cat /sys/kernel/debug/tracing/trace
# Shows: the call chain that caused the delay

# === Tracer 5: wakeup_dl ===
# Same as wakeup_rt but for SCHED_DEADLINE tasks
echo wakeup_dl > /sys/kernel/debug/tracing/current_tracer
```

### 17.2 Finding the Root Cause of Latency Spikes

```bash
# Step-by-step process to debug a latency spike

# Step 1: Confirm the spike exists
sudo cyclictest -t1 -p 95 -n -i 1000 -l 100000 -a 4 -m
# If Max > 100 µs, proceed to investigate

# Step 2: Is it hardware or software?
sudo hwlatdetect --duration=60 --threshold=10 --cpu=4
# If hardware spikes found: firmware/BPMP/TrustZone issue
# If clean: software issue

# Step 3: Use preemptirqsoff tracer
echo preemptirqsoff > /sys/kernel/debug/tracing/current_tracer
echo 0 > /sys/kernel/debug/tracing/tracing_max_latency
echo 1 > /sys/kernel/debug/tracing/tracing_on

# Generate the workload that causes spikes
sleep 60

echo 0 > /sys/kernel/debug/tracing/tracing_on
cat /sys/kernel/debug/tracing/trace > /tmp/latency_trace.txt
echo "Max latency: $(cat /sys/kernel/debug/tracing/tracing_max_latency) µs"

# Step 4: Analyze the trace
head -50 /tmp/latency_trace.txt
# Look for the function call that holds preemption/IRQs disabled longest

# Step 5: Common culprits on Orin Nano
grep -E "nvgpu|tegra|nv_" /tmp/latency_trace.txt
# NVIDIA drivers sometimes hold locks longer than expected

grep "raw_spin_lock" /tmp/latency_trace.txt
# raw_spinlocks are non-preemptible — long holds = latency

grep "console\|printk\|uart" /tmp/latency_trace.txt
# Console output (printk to serial) can hold locks for 100+ µs
# FIX: disable console on RT system or use printk_deferred
```

### 17.3 Function Graph Tracer for Specific Functions

```bash
# Trace specific functions to measure their execution time

# Trace TensorRT-related kernel paths
echo function_graph > /sys/kernel/debug/tracing/current_tracer
echo nvgpu_* > /sys/kernel/debug/tracing/set_ftrace_filter
echo 1 > /sys/kernel/debug/tracing/tracing_on
# Run inference
sleep 5
echo 0 > /sys/kernel/debug/tracing/tracing_on
cat /sys/kernel/debug/tracing/trace

# Output shows call tree with execution times:
#  1)               |  nvgpu_channel_submit() {
#  1)   0.452 us    |    nvgpu_channel_lock();
#  1)   2.123 us    |    nvgpu_submit_append_gpfifo();
#  1)   0.321 us    |    nvgpu_channel_unlock();
#  1)   3.891 us    |  }
```

---

## 18. Memory Management for RT

### 18.1 Why Memory Management Breaks RT

```
Memory-related latency sources:

1. Page faults
   - First access to a mapped-but-not-faulted page
   - Minor fault: 1-5 µs (page table update only)
   - Major fault: 1-10 ms (need to read from disk/swap)
   - Solution: mlockall(MCL_CURRENT | MCL_FUTURE) + pre-fault

2. THP (Transparent Huge Pages) compaction
   - Kernel tries to create 2MB pages from 4KB pages
   - Requires memory compaction: moving pages around
   - Can take 100-5000 µs
   - Solution: echo never > /sys/kernel/mm/transparent_hugepage/enabled

3. SLUB allocator
   - kmalloc/kfree in kernel, malloc/free in userspace
   - Fast path: ~100 ns (take from per-CPU slab)
   - Slow path: ~1-10 µs (refill slab from buddy allocator)
   - Worst case: ~100 µs (buddy allocator compaction)
   - Solution: pre-allocate all buffers before RT loop

4. Page reclaim (kswapd)
   - When memory is low, kernel reclaims pages
   - Can block allocating tasks for milliseconds
   - Solution: size memory appropriately, disable swap

5. mmap/munmap
   - Modifies page tables, may require TLB shootdown
   - TLB shootdown sends IPI to ALL cores → latency spike
   - Solution: never mmap/munmap in RT path
```

### 18.2 Memory Locking and Pre-Faulting

```c
#include <sys/mman.h>
#include <stdlib.h>
#include <string.h>

/* Complete memory preparation for RT thread */
void prepare_memory_for_rt(void)
{
    /* Step 1: Lock ALL current and future pages in RAM */
    if (mlockall(MCL_CURRENT | MCL_FUTURE) != 0) {
        perror("mlockall");
        /* If this fails: need CAP_IPC_LOCK or root */
    }

    /* Step 2: Pre-fault stack
     * Touch every page of a large stack allocation
     * to ensure all stack pages are physically mapped
     */
    {
        volatile char stack_prefault[1024 * 1024]; /* 1 MB */
        memset((void*)stack_prefault, 0, sizeof(stack_prefault));
    }

    /* Step 3: Pre-fault heap allocations
     * Allocate and touch all buffers BEFORE entering RT loop
     */
    void *input_buf = malloc(INPUT_SIZE);
    void *output_buf = malloc(OUTPUT_SIZE);
    memset(input_buf, 0, INPUT_SIZE);   /* force page fault now */
    memset(output_buf, 0, OUTPUT_SIZE);

    /* Step 4: Disable swap (system-wide, requires root) */
    /* system("swapoff -a"); */

    /* Step 5: Set RT thread stack size and pre-fault */
    pthread_attr_t attr;
    pthread_attr_init(&attr);
    pthread_attr_setstacksize(&attr, 2 * 1024 * 1024); /* 2 MB */
    /* Stack will be pre-faulted by mlockall(MCL_FUTURE) */
}

/* The RT loop — NO memory allocation allowed */
void rt_inference_loop(void)
{
    prepare_memory_for_rt();

    while (running) {
        /* ✓ Use pre-allocated buffers */
        /* ✓ No malloc/free */
        /* ✓ No mmap/munmap */
        /* ✓ No file I/O (would allocate page cache pages) */

        read_sensor(pre_allocated_sensor_buf);
        run_inference(pre_allocated_input, pre_allocated_output);
        send_result(pre_allocated_output);

        /* ✓ sched_yield or clock_nanosleep for periodic scheduling */
        clock_nanosleep(CLOCK_MONOTONIC, TIMER_ABSTIME, &next_wake, NULL);
    }
}
```

### 18.3 CUDA Memory and RT

```
NVIDIA CUDA memory operations and their RT impact:

Operation              Latency        RT-Safe?  Alternative
────────────────────────────────────────────────────────────
cudaMalloc             100-500 µs     ✗         Pre-allocate at init
cudaFree               50-200 µs      ✗         Free at shutdown
cudaMemcpy H2D         5-100 µs       △         Use pinned memory
cudaMemcpy D2H         5-100 µs       △         Use pinned memory
cudaMallocHost         200-1000 µs    ✗         Pre-allocate pinned
cudaHostAlloc          200-1000 µs    ✗         Pre-allocate pinned
cudaStreamCreate       10-50 µs       ✗         Pre-create at init
cudaStreamSync         Varies         △         Use events + polling
cudaLaunchKernel       5-20 µs        △         Use CUDA Graphs

RT-safe CUDA pattern:
  INIT:    cudaMalloc, cudaMallocHost, cudaStreamCreate, CUDA Graphs
  RT LOOP: cudaGraphLaunch (pre-compiled, minimal overhead)
  CLEANUP: cudaFree, cudaStreamDestroy (after RT loop ends)
```

---

## 19. NVIDIA Driver Interactions With RT

### 19.1 nvgpu Driver and RT

```
The nvgpu kernel driver is the primary source of NVIDIA-specific
RT interference on Orin Nano. Understanding its behavior is critical.

nvgpu lock contention points:
  1. Channel submission lock (per-channel)
     - Held during GPU command buffer submission
     - Duration: 1-10 µs typically
     - Under PREEMPT_RT: becomes rt_mutex (sleepable, PI-enabled)

  2. Power management lock
     - Held during GPU power state transitions
     - Duration: 10-100 µs during DVFS change
     - Mitigation: lock GPU clocks (jetson_clocks)

  3. Memory management lock
     - Held during GPU memory allocation/free
     - Duration: 5-50 µs
     - Mitigation: pre-allocate all GPU memory at init

  4. Fault handling lock
     - Held during GPU page fault handling
     - Duration: 50-500 µs (rare)
     - Mitigation: pin GPU memory, avoid oversubscription

Interaction with CPU RT tasks:
  - nvgpu driver runs in kernel context on the CPU that initiated the call
  - If CPU RT thread calls CUDA API → enters nvgpu → holds locks
  - Other tasks needing nvgpu locks will block (with PI under PREEMPT_RT)
  - GPU interrupt handler (threaded under PREEMPT_RT) needs locks too
```

### 19.2 Reducing nvgpu Interference

```bash
# Strategy 1: Pin nvgpu IRQ thread to non-RT core
NVGPU_IRQ=$(grep nvgpu /proc/interrupts | awk '{print $1}' | tr -d :)
echo 0 > /proc/irq/${NVGPU_IRQ}/smp_affinity_list

# Set nvgpu IRQ thread priority below your RT thread
NVGPU_PID=$(pgrep -f "irq/.*nvgpu")
chrt -f -p 40 ${NVGPU_PID}

# Strategy 2: Lock GPU clocks to prevent DVFS locks
sudo jetson_clocks
# This eliminates power management lock contention

# Strategy 3: Use dedicated CUDA stream per RT thread
# Avoids channel submission lock contention between threads

# Strategy 4: Use DLA instead of GPU for deterministic inference
# DLA has its own scheduler independent of nvgpu
# (See Real-Time Inference guide Section 5)

# Strategy 5: Monitor nvgpu lock contention
echo 1 > /sys/kernel/debug/tracing/events/lock/contention_begin/enable
cat /sys/kernel/debug/tracing/trace_pipe | grep nvgpu
# Look for high hold times
```

### 19.3 Console/printk Latency (Common Pitfall)

```bash
# Serial console (UART) output is a MAJOR latency source on Jetson.
# printk() with serial console can hold locks for 100-500 µs.
# This affects ALL cores, not just the core printing.

# Problem:
#   1. Any kernel code calls printk()
#   2. printk() acquires console_sem (sleeping lock under RT)
#   3. Console driver (tegra UART) sends bytes at 115200 baud
#   4. Each character takes ~87 µs at 115200 baud
#   5. A 100-character message = 8.7 ms of lock hold time
#   6. Any other printk caller (including RT path) waits

# Solutions:

# 1. Disable console at boot (best for production)
# Add to extlinux.conf APPEND line:
#   console=none
# Or: quiet loglevel=0

# 2. Use high-speed console (reduces hold time)
# Change baud rate to 3000000 (3 Mbaud) if hardware supports it:
#   console=ttyTCU0,3000000n8

# 3. Defer printk in RT code
# In kernel modules: use printk_deferred() in RT-critical paths
# printk_deferred() queues the message for later output

# 4. Redirect console to RAM buffer (dmesg only, no UART)
# Add to extlinux.conf:
#   console=ttyTCU0,115200 loglevel=4

# Verify console latency impact:
echo preemptirqsoff > /sys/kernel/debug/tracing/current_tracer
echo 1 > /sys/kernel/debug/tracing/tracing_on
dmesg  # triggers console output
echo 0 > /sys/kernel/debug/tracing/tracing_on
grep "console\|uart\|serial" /sys/kernel/debug/tracing/trace
```

---

## 20. Production RT Validation and Certification

### 20.1 RT Validation Checklist

```
Before deploying PREEMPT_RT Orin Nano in production:

Kernel Configuration:
  □ CONFIG_PREEMPT_RT=y verified
  □ CONFIG_HIGH_RES_TIMERS=y
  □ CONFIG_HZ=1000
  □ CONFIG_NO_HZ_FULL=y
  □ CONFIG_CPU_ISOLATION=y
  □ Debug options DISABLED for production
  □ RT kernel built and tested with all required NVIDIA drivers

Boot Parameters:
  □ isolcpus=managed_irq,domain,<cores>
  □ nohz_full=<cores>
  □ rcu_nocbs=<cores>
  □ irqaffinity=<system-cores>
  □ nosoftlockup (for isolated cores)
  □ Console minimized or disabled

System Tuning:
  □ CPU governor: performance (locked frequency)
  □ jetson_clocks applied (GPU, memory clocks locked)
  □ THP disabled
  □ Swap disabled
  □ IRQ affinity configured
  □ IRQ thread priorities assigned
  □ RT bandwidth throttling disabled or configured

Application:
  □ mlockall(MCL_CURRENT | MCL_FUTURE)
  □ All memory pre-allocated before RT loop
  □ Stack pre-faulted
  □ SCHED_FIFO or SCHED_DEADLINE set
  □ CPU affinity pinned to isolated cores
  □ PI mutexes used for shared resources
  □ No malloc/free in RT path
  □ No file I/O in RT path
  □ No printk/printf in RT path (use deferred logging)
```

### 20.2 Long-Duration Validation

```bash
#!/bin/bash
# rt-validation-72h.sh — 72-hour RT validation test
set -euo pipefail

DURATION="72h"
OUTPUT_DIR="/data/rt-validation/$(date +%Y%m%d_%H%M)"
mkdir -p "${OUTPUT_DIR}"

echo "=== Starting ${DURATION} RT Validation ==="

# Start system stress (background)
stress-ng --cpu 4 --io 2 --vm 1 --vm-bytes 2G \
    --timeout 0 --metrics-brief &
STRESS_PID=$!

# Start network load (background)
iperf3 -c <remote> -t 0 -P 2 &> /dev/null &
NET_PID=$!

# Start cyclictest on isolated core
sudo cyclictest \
    -t1 \
    -p 95 \
    -n \
    -i 1000 \
    -l 0 \
    --duration=${DURATION} \
    -a 4 \
    -m \
    -h 500 \
    --histfile="${OUTPUT_DIR}/histogram.txt" \
    > "${OUTPUT_DIR}/cyclictest.log" 2>&1 &
CYCLIC_PID=$!

# Start hwlatdetect
sudo hwlatdetect --duration=$((72*3600)) --threshold=5 --cpu=4 \
    > "${OUTPUT_DIR}/hwlatdetect.log" 2>&1 &
HWLAT_PID=$!

# Monitor temperatures
while true; do
    TEMP=$(cat /sys/class/thermal/thermal_zone0/temp)
    echo "$(date +%s) ${TEMP}" >> "${OUTPUT_DIR}/temperature.log"
    sleep 60
done &
TEMP_PID=$!

echo "PIDs: stress=${STRESS_PID} cyclic=${CYCLIC_PID} hwlat=${HWLAT_PID}"
echo "Output: ${OUTPUT_DIR}"
echo "Waiting for ${DURATION}..."

wait ${CYCLIC_PID}

# Cleanup
kill ${STRESS_PID} ${NET_PID} ${HWLAT_PID} ${TEMP_PID} 2>/dev/null

# Generate report
echo "=== RT Validation Report ===" > "${OUTPUT_DIR}/report.txt"
echo "Duration: ${DURATION}" >> "${OUTPUT_DIR}/report.txt"
echo "" >> "${OUTPUT_DIR}/report.txt"
echo "cyclictest results:" >> "${OUTPUT_DIR}/report.txt"
tail -5 "${OUTPUT_DIR}/cyclictest.log" >> "${OUTPUT_DIR}/report.txt"
echo "" >> "${OUTPUT_DIR}/report.txt"
echo "hwlatdetect results:" >> "${OUTPUT_DIR}/report.txt"
tail -10 "${OUTPUT_DIR}/hwlatdetect.log" >> "${OUTPUT_DIR}/report.txt"
echo "" >> "${OUTPUT_DIR}/report.txt"
echo "Temperature range:" >> "${OUTPUT_DIR}/report.txt"
awk '{print $2}' "${OUTPUT_DIR}/temperature.log" | sort -n | \
    awk 'NR==1{min=$1} END{print "Min: " min/1000 "°C, Max: " $1/1000 "°C"}' \
    >> "${OUTPUT_DIR}/report.txt"

cat "${OUTPUT_DIR}/report.txt"
echo "=== Validation complete ==="
```

### 20.3 Pass/Fail Criteria

```
Criteria for production RT deployment on Orin Nano:

MUST PASS (hard requirements):
  ✓ cyclictest Max latency < 100 µs over 72 hours under stress
  ✓ cyclictest p99 latency < 50 µs
  ✓ hwlatdetect: no hardware spikes > 20 µs
  ✓ Zero kernel oops/panics during 72-hour test
  ✓ No RT throttling events
  ✓ Application deadline miss rate < 0.01%
  ✓ No priority inversion detected (lockdep clean)

SHOULD PASS (soft requirements):
  ○ cyclictest Max latency < 50 µs
  ○ cyclictest Avg latency < 10 µs
  ○ Temperature stays below 80°C sustained
  ○ No kswapd activity on isolated cores
  ○ No printk on isolated cores

DOCUMENT (for compliance):
  ◆ Exact kernel version and config
  ◆ Boot parameters used
  ◆ All tuning parameters and scripts
  ◆ cyclictest histogram data
  ◆ hwlatdetect full log
  ◆ Temperature profile over test duration
  ◆ Any anomalies observed and root cause analysis
```

### 20.4 Certification Considerations

```
If targeting safety certification (ISO 26262, IEC 61508, DO-178C):

Linux + PREEMPT_RT is NOT certified out of the box.
However, it is used in certified systems with additional work:

Approach 1: Qualified tool chain
  - PREEMPT_RT kernel is treated as a "software tool"
  - Its output (scheduling behavior) is validated empirically
  - Extensive testing replaces formal verification
  - Used in automotive Tier 1 suppliers for ASIL-B and below

Approach 2: Safety hypervisor + Linux
  - Run a certified RTOS (FreeRTOS, Zephyr, QNX) for safety-critical tasks
  - Run Linux for non-safety AI/ML workloads
  - Hypervisor provides isolation guarantees
  - NVIDIA Jetson supports this via Jetson Safety Extension (NVIDIA Drive)

Approach 3: Formal analysis
  - Use static WCET analysis tools on critical code paths
  - Document all non-deterministic paths and their bounds
  - Provide statistical analysis with confidence intervals
  - Most rigorous but most expensive

For most Jetson edge AI deployments:
  Approach 1 with thorough validation testing is the practical path.
  Document everything, test extensively, maintain traceability.
```

---

## 21. References

* [PREEMPT_RT Wiki](https://wiki.linuxfoundation.org/realtime/start) — official PREEMPT_RT project documentation
* [RT-Tests Git Repository](https://git.kernel.org/pub/scm/utils/rt-tests/rt-tests.git/) — cyclictest, hwlatdetect, and more
* [NVIDIA Jetson Linux Developer Guide — Kernel Customization](https://docs.nvidia.com/jetson/archives/r36.4/DeveloperGuide/SD/Kernel/KernelCustomization.html)
* [Linux Kernel Documentation — RT Mutex](https://www.kernel.org/doc/html/latest/locking/rt-mutex.html) — PI mutex implementation
* [Linux Kernel Documentation — SCHED_DEADLINE](https://www.kernel.org/doc/html/latest/scheduler/sched-deadline.html)
* [Linux Kernel Documentation — CPU Isolation](https://www.kernel.org/doc/html/latest/admin-guide/kernel-parameters.html) — isolcpus, nohz_full
* [ARM Cortex-A78AE Technical Reference Manual](https://developer.arm.com/documentation/101779/latest/) — AE variant specifics
* [ARM GICv3 Architecture Specification](https://developer.arm.com/documentation/ihi0069/latest/) — interrupt controller
* [Ftrace Documentation](https://www.kernel.org/doc/html/latest/trace/ftrace.html) — kernel tracing
* Real-time inference: [Orin Nano Real-Time Inference](../Orin-Nano-Real-Time-Inference/Guide.md)
* Kernel internals: [Orin Nano Kernel Internals](../Orin-Nano-Kernel-Internals/Guide.md)
* Main guide: [Nvidia Jetson Platform Guide](../Guide.md)
