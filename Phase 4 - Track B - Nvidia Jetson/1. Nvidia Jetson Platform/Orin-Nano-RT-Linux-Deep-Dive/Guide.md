# Real-time Linux for a Jetson robot

Real-time engineering starts with a deadline and the consequence of missing it. A fast average is useful, but a robot also needs to know whether an observation is still fresh when an action reaches its actuator. This guide connects Linux scheduling to that complete path.

Start with the [portable timing lab](labs/README.md). Then study the six lessons below. Allow three study sessions: timers and scheduling, blocking and interrupts, then an end-to-end timing report. Basic C, processes, threads, and mutexes are the prerequisites. The [robotics timing lessons](../../../Phase%205%20-%20Advanced%20Topics%20and%20Specialization/Track%20D%20-%20Robotics/Robotics%20Foundations/README.md) provide the robot context.

**Version boundary, checked 2026-10-18:** kernel behavior depends on its version, configuration, drivers, and running preemption mode. Read the [Jetson compatibility matrix](../../Compatibility.md) before selecting packages. The linked R36.4.4 RT installation instructions describe an older teaching baseline and explicitly label that RT support Developer Preview. They are not an instruction to mix an R36 repository with an R39 system. [NVIDIA's release-specific RT documentation](https://docs.nvidia.com/jetson/archives/r36.4.4/DeveloperGuide/SD/Kernel/RealTimeKernel.html) is the authority for that release.

## 1. Define the timing contract before tuning Linux

A periodic task has period T, execution demand C, and relative deadline D. Those three need not be equal. A camera may produce a new frame every 33.3 ms while a controller runs every 10 ms using the most recent acceptable state estimate.

For one action, record these events in a compatible clock domain:

```text
sensor exposure → driver receives data → callback starts → inference finishes
                → controller accepts result → actuator applies command
```

The age of the action's evidence is `actuator_time - exposure_time`. Kernel wake-up latency measures only part of this path. A 50 µs wake-up delay does not compensate for a camera buffer that is 100 ms old. Hardware timestamps, ROS simulation time, wall-clock time, and monotonic process time cannot simply be subtracted without a defined conversion.

Here is a **hypothetical budget**, not a Jetson measurement:

| Stage | Allocated bound |
|---|---:|
| Acquisition and transport | 4 ms |
| Queue wait and dispatch | 2 ms |
| Preprocessing and inference | 8 ms |
| Control and command transport | 2 ms |
| Remaining margin in a 20 ms age limit | 4 ms |

Adding budgets is useful only when the interfaces and overlap assumptions are explicit. Measured stage p99 values do not generally add to the pipeline p99: rare events can coincide, and correlations matter. Trace whole requests with an identifier.

**Check:** inference takes 6 ms, but its input waited 18 ms before it began. Is it fresh under a 20 ms age limit? **Answer:** no; it is at least 24 ms old before subsequent stages.

## 2. Understand what PREEMPT_RT changes

A preemption model controls where a runnable higher-priority task can replace the current task in kernel execution. It does not assign an application a deadline or eliminate contention.

| Configuration family | Main idea | Remaining limitation |
|---|---|---|
| Non-preemptible kernel | Kernel execution reaches a scheduling/blocking point before another task takes over | Long kernel paths can delay runnable tasks |
| Voluntary preemption | Additional explicit rescheduling opportunities | Delays remain between opportunities |
| Preemptible kernel | More kernel execution can be preempted | Protected and interrupt-disabled regions still exist |
| PREEMPT_RT | Moves much interrupt/locking work into schedulable, preemptible contexts | Raw critical sections, hardware, firmware, drivers, and shared resources still matter |

There is no universal “5–50 µs worst case” attached to the PREEMPT_RT configuration. Dynamic preemption options also mean a build configuration alone may not describe the active mode. `might_sleep()` is a debugging/context check, not a synonym for an unconditional voluntary scheduling point.

The kernel's [PREEMPT_RT explanation](https://docs.kernel.org/core-api/real-time/differences.html) describes forced interrupt threading and the remaining special cases. Some interrupt handling stays in hard interrupt context. Firmware activity and hardware response are outside the scheduler's direct control.

Begin with read-only inspection:

```bash
uname -a
cat /sys/kernel/realtime 2>/dev/null
cat /proc/cmdline
ps -eLo pid,tid,cls,rtprio,pri,psr,comm
```

A missing `/sys/kernel/realtime` file is an observation, not sufficient proof of every configuration detail. Where available, inspect `/proc/config.gz` or the matching `/boot/config-*` file and release documentation. Containers share the host kernel; installing a userspace package inside a container does not turn the host into PREEMPT_RT.

## 3. Choose a scheduling policy and bound the work

`SCHED_OTHER` provides ordinary time sharing. `SCHED_FIFO` uses fixed priorities without a round-robin time quantum among equal-priority runnable threads. A FIFO thread that never blocks can prevent lower-priority work from progressing. `SCHED_RR` adds a quantum among peers at the same priority. These policies schedule CPU threads, not GPU kernels or remote actuators. See the [Linux scheduling API](https://man7.org/linux/man-pages/man7/sched.7.html).

A complete design assigns priorities to dependencies too. Giving a controller priority 90 can hurt it if the driver or communication thread it depends on cannot run. Increasing every thread to a high priority is not a scheduling analysis.

For a simple fixed-priority, single-core task model with independent periodic tasks, bounded blocking B, and no release jitter, response-time analysis iterates:

```text
R_i(next) = C_i + B_i + sum over higher-priority tasks j of ceil(R_i / T_j) * C_j
```

Stop when the value converges or exceeds the task's deadline. This model needs defensible execution and blocking bounds; a sample average is not an acceptable substitute.

Worked example: a high-priority task has C=1 ms, T=5 ms. A lower-priority task has C=2 ms, T=D=10 ms and B=0.5 ms. Start at R=2.5 ms; the next value is `2 + 0.5 + ceil(2.5/5)*1 = 3.5 ms`. Repeating gives 3.5 ms, within D. This conclusion is conditional on the model. GPU suspension, multicore migration, lock chains, and release jitter require additional analysis.

`SCHED_DEADLINE` expresses runtime, deadline, and period and uses reservation/admission mechanisms. It is not a way to declare any arbitrary workload feasible. Study the kernel's [deadline scheduling documentation](https://docs.kernel.org/scheduler/sched-deadline.html) before an experiment; its assumptions and bandwidth restrictions matter.

For this course's first lab, keep the ordinary scheduler. Only move to privileged scheduling experiments on a prepared development system with a bounded workload, a recovery route, and a reason supported by measurements.

## 4. Bound blocking, memory faults, and interrupt work

Priority inversion occurs when a high-priority thread waits for a resource owned by a lower-priority thread while unrelated medium-priority work delays that owner. A priority-inheritance mutex can temporarily boost the owner. It does not remove the critical section, repair deadlock, or bound a device transaction inside it.

Kernel locks differ from user-space mutexes. Under PREEMPT_RT, ordinary `spinlock_t` has different semantics from `raw_spinlock_t`; raw spinning locks retain non-preemptible behavior. Plain non-RT `spin_lock()` disables preemption, while the IRQ variants also control local interrupts. Do not generalize that every spin lock always disables interrupts. The [kernel lock-type reference](https://docs.kernel.org/locking/locktypes.html) documents the exact distinctions and nesting rules.

At application level, make a buffer's ownership explicit:

```text
FREE → producer writes → READY → consumer owns → completion → FREE
```

The completion event must cover the last asynchronous reader. Returning a CUDA input buffer to a pool immediately after enqueue can let the camera overwrite data that the GPU still reads. A mutex protecting the queue does not automatically protect an in-flight buffer's lifetime.

Memory faults and allocator work can add latency. Allocate predictable buffers before the timed loop, touch the intended working set, and check failures. `mlock`/`mlockall` can prevent swapping for successfully locked memory within resource limits; they do not make arbitrary future code, allocations, or device access deterministic. `MCL_ONFAULT` deliberately leaves population until access. See the [memory-locking API](https://man7.org/linux/man-pages/man2/mlock.2.html).

For driver work, keep hard IRQ handlers small and move suitable work to a context that may sleep. Check APIs against the actual kernel configuration: code that allocates while holding a raw lock can be invalid even if a superficially similar non-RT path appeared to work. Use lockdep and the matching driver development workflow; a generic tutorial module is not production validation.

## 5. Diagnose a tail before changing system settings

Run the [native timing experiment](labs/README.md), then trace the actual robot application. Distinguish a late release from excessive execution and a blocked dependency.

| Symptom | Evidence to collect | Likely next experiment |
|---|---|---|
| Thread runnable but not scheduled | Scheduler trace with runnable/switch events | Compare competing tasks and priorities |
| Thread waiting on a lock | Lock ownership and duration | Shorten/split the critical section |
| First invocation much slower | Faults, initialization, allocation trace | Separate warm-up from steady state |
| Inference completion delayed | GPU timeline, queue depth, other contexts | Reduce competing work or adjust admission |
| Delays grow with temperature | Thermal and clock telemetry over time | Repeat at sustained operating conditions |
| Fresh kernel timer, stale action | Sensor stamps and queue occupancy | Change queue/expiry policy |

`cyclictest` and timer-latency tracing are useful kernel diagnostics, but they do not execute your full perception/control pipeline. Report their command line, priority, affinity, load, duration, histogram, and kernel. Do not label a fabricated comparison table “measured.” The portable lab includes real raw traces and explicitly states the host on which they were collected.

CPU affinity, IRQ placement, cpusets, tick configuration, and power settings are possible experimental controls. They have costs: moving an interrupt can separate a consumer from its data; pinning too many tasks to one CPU can create a bottleneck; maximizing clocks can cause later thermal throttling. Change one justified variable at a time and preserve a baseline. A Jetson development image's default settings are not a timing guarantee.

Keep an overrun policy outside the performance optimization loop. For example, the controller may reject an expired observation and request its defined stop behavior. A process heartbeat proves that something is alive, not that every command is fresh or that an actuator applied it.

## 6. Build a defensible timing report

The deliverable is a report a teammate can reproduce:

1. **Contract:** period, deadline, maximum observation age, accepted jitter, overload and stale-data behavior.
2. **Configuration:** module, carrier, JetPack/L4T, kernel/configuration, firmware, drivers, scheduler policy, priorities, CPU placement, power mode, clocks, cooling, and application revision.
3. **Workload:** sensors and rates, input sizes, concurrent models, network/storage activity, warm-up and steady-state duration.
4. **Evidence:** raw request traces, sample count, quantile definition, maximum observed value, missed deadlines, rejected stale actions, and relevant resource telemetry.
5. **Limits:** scenarios not exercised and assumptions required for any analytical bound.

A long run with no misses increases empirical confidence under its tested conditions. It does not turn the largest observed sample into WCET. If the system needs a hard bound, use an appropriate hardware/software architecture and a justified analysis of all allowed execution paths. A separate bounded controller may be a better place for a tight actuator loop than a complex perception process.

**Self-check:** explain why each claim is false.

- “PREEMPT_RT makes every instruction preemptible.” Some contexts remain non-preemptible, and hardware/firmware effects remain.
- “A high-priority CUDA stream guarantees the robot's deadline.” Stream priority does not establish CPU-to-actuator response bounds.
- “CPU utilization is below 100%, so all tasks meet deadlines.” Priority, blocking, bursts, and deadlines matter as well as average demand.
- “A 100-sample p99 proves the one-in-a-million tail.” It cannot resolve that tail.
- “Locking memory removes all jitter.” It addresses one source; scheduling, I/O, locks, caches, and accelerators remain.

Next combine these ideas with [memory ownership](../Orin-Nano-Memory-Architecture/Guide.md) and the [inference pipeline guide](../Orin-Nano-Real-Time-Inference/Guide.md).
