# Lecture 6: CPU scheduling, fair service, and real-time budgets

A thread can be ready to run and still wait for a CPU. It can also be blocked on data, a mutex, or a device; raising its scheduling priority does not make that dependency finish. This lecture separates those delays before introducing Linux policies.

**Prerequisites:** processes, threads, interrupts, and system calls. Plan 60 minutes. Use an ordinary unprivileged process for the inspection exercise; changing a system’s real-time policy is an optional target-specific experiment, not a prerequisite.

## 1. Distinguish runnable, running, and blocked

For a fictional worker released at 0 ms:

| Interval | State | Duration |
|---|---|---|
| 0–2 ms | Runnable, waiting for a CPU | 2 ms |
| 2–5 ms | Running CPU computation | 3 ms |
| 5–9 ms | Blocked for a device result | 4 ms |
| 9–10 ms | Runnable again | 1 ms |
| 10–11 ms | Running validation | 1 ms |

The response time is 11 ms; CPU execution is 4 ms, runnable delay is 3 ms, and blocking is 4 ms. A 10 ms application deadline is missed. Reducing runnable delay can help, but neither a priority change nor a faster CPU directly proves the device wait will shrink.

A process can have several threads with different policies and affinity. State whether an observation identifies a process ID, thread ID, CPU, or cgroup.

## 2. Understand policy without an unconditional global hierarchy

Linux has scheduling classes and per-thread policies. In the usual configuration, runnable deadline and fixed-priority real-time work take precedence over ordinary fair-class work on eligible CPUs, subject to budgets, throttling, affinity, and applicable preemption constraints. A task on one CPU does not instantly preempt every normal task on every other CPU.

| Policy | Main idea | Important limit |
|---|---|---|
| `SCHED_OTHER`/normal | Weighted fair service | No application completion deadline is promised |
| `SCHED_BATCH` | Fair-class policy suited to batch work | Still shares CPU with other eligible work |
| `SCHED_IDLE` | Very low-priority userspace policy | Not the kernel’s per-CPU idle thread/class |
| `SCHED_FIFO` | Fixed real-time priority, no ordinary time slice at that priority | Can delay lower-priority dependencies; blocking/throttling still matters |
| `SCHED_RR` | Fixed real-time priorities with a quantum among equal-priority peers | Quantum is not a task deadline |
| `SCHED_DEADLINE` | Runtime/deadline/period with deadline ordering and bandwidth control | Admission and execution assumptions are not a whole-application proof |

Kernel version/configuration can add other mechanisms such as extensible scheduling. Inspect the installed system and matched documentation rather than treating a course diagram as a complete dispatch implementation. [Linux scheduling policies](https://man7.org/linux/man-pages/man7/sched.7.html).

## 3. Learn the CFS idea, then the EEVDF selection rule

Classic CFS accounts for weighted CPU service using virtual runtime. With the same execution time, a higher-weight entity accumulates less virtual runtime; the scheduler aims to distribute service according to weights. Nice values influence weights, but group scheduling, runnable populations, and other constraints affect the observed share. A tuning parameter is not a hard upper bound on wakeup delay. [CFS design](https://docs.kernel.org/scheduler/sched-design-CFS.html).

Linux began transitioning fair scheduling to EEVDF in 6.6. EEVDF uses service lag to determine eligibility, then selects an eligible entity with an early virtual deadline. It retains fair-scheduling machinery; saying all CFS code disappeared in 6.6 is incorrect. A **virtual deadline** is an accounting quantity, not the robot’s wall-clock result deadline. [Linux EEVDF](https://docs.kernel.org/scheduler/sched-eevdf.html).

**Selection exercise — simplified candidate set:**

| Entity | Eligible? | Virtual deadline |
|---|---|---|
| A | yes | 30 |
| B | yes | 24 |
| C | no | 18 |

B wins among these candidates. C’s earlier virtual deadline does not overcome its ineligibility. Real scheduler code also handles hierarchy, wakeups, preemption, and other details. Sleeping does not automatically create an unlimited credit or guarantee immediate execution on waking.

To compare kernels, use the same workload and measure wakeup-to-run delay, throughput, and fairness. Do not promise that EEVDF universally improves every AI workload’s p99. A Yocto release name also does not uniquely identify a BSP’s kernel or backports.

## 4. Reason about FIFO and round-robin dependencies

A FIFO worker normally continues until it blocks, yields, is preempted by applicable higher-priority work, or is constrained by the system’s policy/budget mechanisms. If it spins while waiting for a lower-priority helper on the same CPU, it can prevent the helper from producing the result it needs.

**Worked failure:** H submits work to L and repeatedly polls a flag without blocking. If H always outranks L on their only available CPU, L may receive no execution time. Raising H’s priority makes this dependency problem worse. Use a suitable blocking completion mechanism or redesign the scheduling/ownership contract; see [priority inversion](Lecture-11.md).

Round-robin adds time sharing among equal-priority real-time peers. It does not ensure a complete multi-stage job meets its deadline. Requested privileges/resource limits and actual policy must be checked; a failed scheduler call is not a successful configuration.

## 5. Interpret deadline budgets correctly

A deadline server specifies runtime Q, relative deadline D, and period P, with the API’s validity constraints. An example `Q=2 ms, D=10 ms, P=10 ms` requests a 20% CPU bandwidth budget. It is a fictional example, not a measured openpilot configuration.

Consider three such servers with budgets 2/10, 1/5, and 3/20 ms. Their nominal utilization is `0.2 + 0.2 + 0.15 = 0.55`. That arithmetic is useful input to analysis. It does not account for every blocking dependency, release jitter, interrupt, runtime overrun, shorter deadline, multiprocessor effect, or device stage.

Linux applies admission control and bandwidth rules for its deadline scheduling domains. An `EBUSY` rejection may indicate admission failure; successful admission does not prove an arbitrary program has bounded execution or completes all external work on time. Read the documented CBS, throttling, affinity, and scheduling assumptions. [Deadline scheduler](https://docs.kernel.org/scheduler/sched-deadline.html).

**Check:** a GPU kernel takes 8 ms while its host thread consumes only 0.2 ms of CPU. Reserving 2 ms of CPU per 10 ms does not reserve 8 ms of GPU service. Budget and measure those resources separately.

## 6. Inspect before changing policy

These read-only Linux commands inspect the shell and its process context; availability varies by distribution:

```bash
uname -r
chrt -p $$
taskset -pc $$
ps -o pid,tid,cls,rtprio,ni,psr,comm -p $$
cat /proc/$$/sched
```

`$$` is the current shell’s PID, not the inference worker’s. For the actual program, identify the intended thread and its allowed CPU set. Some `/proc` fields depend on kernel configuration. Record permission errors or missing tools instead of replacing failed output with an assumed policy.

A high involuntary-context-switch count does not by itself prove a scheduling defect or justify elevating priority. Correlate task release, actual execution, blocking, queue age, and deadlines. Preemption can be expected and useful.

The [native API demo](demos/rt-demo/README.md) exposes scheduling and affinity calls, while the [SPSC queue lab](../labs/SPSC-Queue.md) exercises a real threaded producer/consumer. Neither is a worst-case latency benchmark.

## 7. Produce evidence

Submit the 11 ms response-time breakdown, the EEVDF selection answer, the 55% nominal budget calculation, and one read-only policy/affinity record. Explain which fields describe the actual worker and which do not.

For a device extension, instrument a bounded workload under representative background load, report the maximum **observed** delay and distribution, and retain a deadline-miss trace. Any policy change needs its previous value and a restoration procedure. Continue to [PREEMPT_RT and latency](Lecture-07.md), then [CPU placement](Lecture-08.md).
