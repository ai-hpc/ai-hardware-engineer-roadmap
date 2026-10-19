# Lecture 23: Containers, cgroups, and GPU Sharing

A container packages a process and its dependencies while the host kernel still schedules its threads and controls devices. Three questions must be answered separately: **what can the process see, what can it access, and what resources may it consume?** Namespaces, credentials/security policy, and cgroups answer different parts.

Prerequisites: [processes](Lecture-02.md), [memory](Lecture-12.md), and [GPU drivers](Lecture-20.md). This lesson uses read-only inspection and the existing container lab; it does not reconfigure a host's GPU partitions.

## Containers and virtual machines

Containers normally share the host kernel. A VM has a guest kernel behind a virtualization boundary. Neither label alone specifies security, startup time, or performance: image loading, storage, services, device access, and runtime configuration matter. Hardware passthrough also changes the boundary.

Linux namespaces can isolate PID numbering, mount trees, network stacks, hostnames, IPC objects, user IDs, cgroup views, and some clock offsets. A mount namespace is not automatically a private root filesystem. A user namespace changes UID/GID interpretation but does not grant unrestricted host administration. A PID namespace does not by itself make a previously mounted `/proc` show only the new namespace's processes.

Inspect the current environment without creating privileged namespaces:

```bash
ls -l /proc/self/ns
cat /proc/self/cgroup
cat /proc/self/status
```

A process inside a container can have different host and namespace PIDs. Use the correct identity when tracing or inspecting host cgroups.

## cgroup v2: limits, placement, and actual accounting

| File | Meaning | Example interpretation |
|---|---|---|
| `cpu.max` | CPU-time quota and period in µs | `50000 100000`: aggregate 0.5 CPU worth of budget |
| `cpu.weight` | Relative fair-class CPU weight | Competing siblings share available CPU proportionally |
| `cpuset.cpus` | Requested eligible CPU list | `2-3`: tasks can run on those allowed logical CPUs |
| `cpuset.cpus.effective` | Granted CPU list | May differ because of ancestors and online CPUs |
| `cpuset.cpus.exclusive` | CPU **list**, where supported | `1` means CPU 1, not “exclusivity enabled” |
| `cpuset.cpus.partition` | Partition state, where supported | Root/isolated partition rules need valid ancestor setup |
| `memory.max` | Hard memory limit | Reclaim is attempted; unresolved pressure can trigger cgroup OOM |
| `memory.swap.max` | Swap-accounting limit | `0` restricts swap for this cgroup; it does not remove all I/O |
| `pids.max` | Task-count limit | Can make task creation fail |

The [cgroup v2 documentation](https://docs.kernel.org/admin-guide/cgroup-v2.html) defines hierarchy, delegation, and controller rules. Creating a directory alone does not make every controller available. Controllers must be enabled appropriately by the parent, and system managers may own the subtree. Use your service/container manager's supported resource settings rather than editing an arbitrary production pod's cgroup behind its back.

**Worked quota:** four runnable threads collectively consume 50 ms of CPU time during a 100 ms period. They can exhaust that quota in roughly 12.5 ms of wall time if all four run simultaneously, then be throttled. The quota does not reserve a dedicated half-core and does not guarantee a thread wakes within 50 ms.

Compare deltas in `cpu.stat` (`nr_throttled`, `throttled_usec`) with application traces. Increasing throttling is evidence of a limit being reached, not proof that it is the only source of a particular latency spike. Likewise a CPU mask controls eligibility; it does not move every IRQ, exclude every other task, or remove shared-cache/DRAM contention.

## Images, runtime, and lifecycle

An OCI image describes filesystem layers and metadata. An OCI runtime creates and runs the process with configured namespaces, mounts, credentials, and resource controls. Higher-level tools such as Docker or Kubernetes manage images, networking, and lifecycle around that runtime. `runc` and `crun` implementations can differ, but a universal “2–5× faster” claim is not useful without a measured startup workload.

Treat readiness and liveness as separate contracts. A process can be running and answering health requests while rejecting inference because its model/configuration is unavailable. A SIGTERM handler should stop accepting work, drain or cancel it within the service's grace period, and exit with a clear status. A container's stop grace period does not make arbitrary GPU work cancellable.

Use the [real container lifecycle lab](../../../Phase%204%20-%20Track%20B%20-%20Nvidia%20Jetson/1.%20Nvidia%20Jetson%20Platform/Orin-Nano-Container-Fleet/labs/README.md) for a CPU service with actual HTTP, readiness, and SIGTERM tests. Its Docker evidence is for the recorded host architecture; it does not prove that an image contains compatible Jetson CUDA libraries.

## NVIDIA GPU access in a container

The host owns the GPU driver. NVIDIA Container Toolkit arranges selected device access and appropriate host driver components through configured mechanisms, including CDI and runtime integration. The container still needs the application/runtime libraries it uses: exposing a GPU does not install TensorRT, PyTorch, or every CUDA toolkit library into an arbitrary image.

Keep a compatibility record containing host GPU, driver/BSP, image digest and architecture, CUDA/framework versions, model artifact, and a real inference smoke test. `nvidia-smi` can help inspect supported management paths on discrete systems; its success does not prove a CUDA kernel or TensorRT engine runs, and it is not a universal Jetson acceptance test. See the [official toolkit documentation](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/index.html).

## MIG: check the actual device

MIG partitions supported GPUs into hardware instances with defined compute and memory resources. **Jetson Orin does not support MIG.** A100/H100 and other supported products have their own profile tables; do not apply an A100's count, memory size, or profile names to every GPU. NVIDIA's [supported-GPU list](https://docs.nvidia.com/datacenter/tesla/mig-user-guide/supported-gpus.html) is the reference for current products.

A configured instance can reduce particular forms of interference. It does not independently guarantee an end-to-end SLA: CPU scheduling, preprocessing, network, storage, shared host links, power, and application queues still matter. Changing MIG configuration can affect active workloads, so this foundation lesson leaves partition creation to an explicitly planned device exercise.

## MPS: concurrency with version-specific isolation

CUDA MPS can improve utilization when multiple cooperating processes submit GPU work. On Volta and later, MPS clients have separate GPU virtual address spaces; the old blanket statement that every MPS client can directly access every other client's allocation is incorrect. Fault containment and resource controls differ by architecture and configuration. Separate address spaces also do not make MPS equivalent to MIG's resource partitioning or a complete multi-tenant security boundary. Consult the [MPS documentation](https://docs.nvidia.com/deploy/mps/architecture.html).

Do not assume the same sharing mode is available on every edge device or under every driver. Configure pipe/log directories before starting an MPS daemon when that workflow is supported, and account for who owns the daemon and how it shuts down. This course does not start a system-wide daemon as part of a CPU lesson.

## Kubernetes connects these mechanisms

Kubernetes schedules requested resources onto nodes; device plugins or newer device-allocation integrations expose supported GPU resources. The selected plugin, operator, and versions determine discovery and assignment behavior. CPU requests/limits, QoS, CPU Manager policies, NUMA alignment, and device allocation are separate settings. A request of “one GPU” is not automatically a dedicated CPU or a latency reservation.

**Check:** A pod is allowed CPUs 2–3. Are those cores exclusive? **Answer:** No; eligibility and exclusivity are different. Inspect the manager's policy, effective masks/partitions, and other work using those CPUs.

**Check:** Does `memory.swap.max=0` prevent a page fault from delaying inference? **Answer:** No. File-backed faults, allocation/reclaim, device mappings, and other stalls remain.

**Check:** A wrong model path leaves the service process alive but readiness returns 503. Is that contradictory? **Answer:** No. Liveness reports a functioning process; readiness reports whether it can serve the requested work.
