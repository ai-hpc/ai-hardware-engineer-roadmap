# Lecture 24: Put the operating system into an AI system contract

A perception application depends on sensors, buffer ownership, scheduling, inference, message delivery, and a consumer. This lesson connects those mechanisms without assuming that a particular priority number or “zero-copy” label makes the complete system deterministic.

**Prerequisites:** lectures 15–23 and the [native queue lab](../labs/SPSC-Queue.md). Plan 60–90 minutes for a CPU exercise and a separate session for a device investigation. Use the maintained platform chapters for exact release combinations; this lecture does not flash hardware or change a running machine’s scheduler.

## 1. Draw the complete path

For a fictional camera-based robot:

```text
exposure/acquisition → capture completion → preprocessing → inference
        → result validation → command publication → controller acceptance
```

For every arrow, record the payload, timestamp clock, ownership transfer, queue bound, and failure behavior. A frame sitting in a buffer is not necessarily a fresh frame; a model returning is not necessarily a completed device computation until its API’s completion condition is satisfied.

**Worked budget:** suppose the application permits 80 ms from acquisition to accepted result. Allocate hypothetical budgets of 20 ms acquisition/readout, 8 ms preprocessing, 25 ms inference, 7 ms validation/communication, and 10 ms queue/scheduling delay. The sum is 70 ms, leaving 10 ms planning margin. If queuing rises by 15 ms, the new 85 ms path misses the deadline even if inference itself still takes 25 ms. Budgets need measurement and bounding assumptions; this is arithmetic, not a Jetson benchmark.

**Check:** if a throughput report says 60 results/s, can you infer an 80 ms age bound? No. A deep pipeline can produce results frequently while each result is old.

## 2. Treat a Jetson release as a compatible platform

Jetson Linux supplies platform-specific kernel, firmware, boot, and userspace integration. The available engines and driver paths depend on the module and BSP. For example, Orin Nano has no DLA or hardware NVENC encoder; a guide for another Jetson SKU is not evidence those engines exist on Nano. Use the [current compatibility matrix](../../../Phase%204%20-%20Track%20B%20-%20Nvidia%20Jetson/Compatibility.md) and its primary vendor sources.

For camera input, distinguish direct V4L2 capture, a supported Argus/ISP path, USB camera processing, and decoded network video. A DMA-BUF file descriptor can share an allocation between compatible components; it does not itself establish pixel layout, cache visibility, device completion, or absence of every conversion/copy. Complete the [CPU RAW/video exercises](../../../Phase%204%20-%20Track%20B%20-%20Nvidia%20Jetson/1.%20Nvidia%20Jetson%20Platform/Orin-Nano-Camera-ISP-Sensor-Bringup/Guide.md) before adding a device-specific path.

Power modes, clocks, and temperature alter the available operating point. Inspect the actual mode table rather than assuming mode 0 enables every resource at its maximum. The [power course](../../../Phase%204%20-%20Track%20B%20-%20Nvidia%20Jetson/1.%20Nvidia%20Jetson%20Platform/Orin-Nano-Power-Thermal/Guide.md) connects measured energy and sustained timing; a clock-locking command does not remove thermal or electrical limits.

## 3. Read openpilot as source evidence

The course’s openpilot reference pins commit `b1e88c2357f07dfc4e0872370b64d97ad606391b`, with source version 0.11.3. Inspect the [real-time helper](https://github.com/commaai/openpilot/blob/b1e88c2357f07dfc4e0872370b64d97ad606391b/openpilot/common/realtime.py), then the call sites in `modeld`, `card`, and `controlsd`. A helper’s definition does not prove which arguments a particular process uses on a particular platform.

AGNOS is operating-system build/distribution work for supported comma devices; one historical kernel tree is not the OS for every hardware generation. Its [pinned update manifest](https://github.com/commaai/openpilot/blob/b1e88c2357f07dfc4e0872370b64d97ad606391b/openpilot/common/hardware/comma/agnos.json) identifies image components, sizes, and hashes. The [AGNOS reading guide](../../../Phase%205%20-%20Advanced%20Topics%20and%20Specialization/Track%20E%20-%20Autonomous%20Vehicles/2.%20openpilot%20Reference%20Stack/agnos/Guide.md) supplies the maintained route and recorded source evidence.

Separate ordinary message IPC from image-buffer sharing. A shared image pool avoids some pixel copying only when every participant obeys ownership, format, and lifetime rules. It is not evidence that an entire video pipeline contains no copies or that it uses the particular SPSC algorithm in Lecture 10.

Panda has its own STM32 firmware and vehicle-specific safety logic associated with opendbc. Do not label it a Zephyr application or assume its host transport is a generic USB CDC serial port. Follow the selected hardware and pinned submodule source. [Panda’s upstream architecture and firmware](https://github.com/commaai/panda).

This course’s [replay exercise](../../../Phase%205%20-%20Advanced%20Topics%20and%20Specialization/Track%20E%20-%20Autonomous%20Vehicles/2.%20openpilot%20Reference%20Stack/Replay-Lab.md) demonstrates software timing/failure contracts. It does not establish a vehicle safety result or validate a deployed openpilot deadline.

## 4. Compare an RTOS without conflating products

An RTOS such as Zephyr can provide threads, interrupts, synchronization, timers, and device integration on supported microcontrollers. Its suitability depends on the application, configuration, drivers, timing analysis, and validation. A separate MCU reduces some dependencies on the host OS but does not automatically make the complete system safe.

Zephyr and Linux both use devicetree concepts, but their build and device-instantiation mechanisms differ. Zephyr processes devicetree during the build and provides generated definitions used by its device model; it is not simply Linux’s runtime driver probing with identical source. [Zephyr devicetree structure](https://docs.zephyrproject.org/latest/build/dts/intro-syntax-structure.html).

For a host/MCU boundary, specify message schema, sequence, timestamp/timeout, allowed state transitions, and the MCU’s response to missing or invalid input. The [native framed-telemetry lab](../../../Phase%202%20-%20Embedded%20Systems/2.%20Embedded%20Software/labs/Framed-Telemetry.md) provides a CPU-tested parser and freshness exercise. It is not a substitute for electrical, bus, firmware, or actuator testing.

## 5. Tune one cause of delay at a time

| Mechanism | Question it addresses | Limit to preserve |
|---|---|---|
| CPU affinity/cpusets | Where can this task run? | Placement is not exclusive ownership or memory-bandwidth isolation |
| IRQ placement | Which CPUs service device interrupts? | Some interrupt/work paths are not freely movable |
| Tick/RCU configuration | Which periodic/callback work can be reduced or offloaded? | Conditions and helper-thread affinity matter; not “all interrupts disappear” |
| Memory locking/prefaulting | Which residency/first-touch delays can be reduced? | Check limits/results and the actual working set; not “all future faults are impossible” |
| Scheduling policy | How is runnable CPU work selected/budgeted? | Does not bound device, lock, I/O, or end-to-end delays by itself |
| Buffer/queue limits | How much pending work can accumulate? | Define overflow and freshness behavior |

Begin with read-only observations and the default unprivileged lab. Record kernel/configuration, CPU topology, actual affinity, page faults, runnable delay, blocked time, queues, and device completions. Change one mechanism, retain the old configuration, and repeat the same workload and failure case. [Linux CPU isolation](https://docs.kernel.org/admin-guide/kernel-per-CPU-kthreads.html), [NO_HZ conditions](https://docs.kernel.org/timers/no_hz.html).

Memory locking can fail because of privilege/resource limits and can make future allocation or stack growth fail. Prefaulting the main thread’s stack does not touch another thread’s stack. Avoid creating new mappings or copy-on-write obligations after declaring the working set prepared; verify the path you actually execute. [Linux mlock semantics](https://man7.org/linux/man-pages/man2/mlock.2.html).

## 6. Budget CPU service without claiming a complete deadline proof

`SCHED_FIFO` uses fixed-priority scheduling with its applicable preemption/throttling rules. A runnable high-priority worker can interfere with required lower-priority services. A FIFO thread still blocks on locks/I/O and can be delayed by other applicable execution contexts.

`SCHED_DEADLINE` uses runtime, deadline, and period parameters with scheduler admission/bandwidth rules. An assumed 2 ms runtime per 10 ms period corresponds to 20% CPU utilization for that server. It does not prove that an inference plus sensor, GPU, communication, and control path finishes in 10 ms. Blocking, jitter, deadlines shorter than periods, multiprocessor scheduling, dependencies, and execution-time assumptions need the appropriate analysis. Utilization below 100% alone is not a universal schedulability theorem. [Linux deadline scheduler](https://docs.kernel.org/scheduler/sched-deadline.html).

A wakeup-latency test reports the delayed wakeups it observed under its conditions. It does not establish all application interference or a hard worst-case bound. Report duration, sample count, workload, distribution, maximum observed value, and missed application deadlines separately. Do not convert “maximum observed” into “maximum possible.”

## 7. Make deployment part of the runtime contract

An application update changes more than executable bytes. It can change input shape, preprocessing, calibration, persistent schema, memory use, scheduling demand, and failure behavior. A compatible old rootfs may still fail if shared data has become unreadable.

Use the [file-backed interrupted-update lab](../../../Phase%204%20-%20Track%20B%20-%20Nvidia%20Jetson/1.%20Nvidia%20Jetson%20Platform/Orin-Nano-OTA-Deep-Dive/labs/README.md) to practice candidate installation, bounded trials, exact-boot acceptance, and fallback. Use the [local service lab](../../../Phase%204%20-%20Track%20B%20-%20Nvidia%20Jetson/1.%20Nvidia%20Jetson%20Platform/Orin-Nano-Container-Fleet/labs/README.md) to observe a running process that is live but unready. Both label their CPU scope explicitly.

**Completion:** submit one complete observation-to-result diagram, a numerical budget, the native queue evidence, a reproduced stale/overload case, and an update-health criterion. Mark synthetic stages, source-only claims, and actual device measurements separately. Continue to [Yocto integration](Lecture-25.md) and [observability](Lecture-26.md).
