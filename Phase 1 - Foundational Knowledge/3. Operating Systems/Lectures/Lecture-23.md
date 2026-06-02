# Lecture 23: Containers, cgroups v2 & NVIDIA Container Runtime

## Overview

Containers are the standard packaging format for deploying AI inference software in both cloud and edge environments. They solve a fundamental problem in AI deployment: a TensorRT model that runs correctly on one machine — with specific CUDA, cuDNN, and TensorRT library versions — may silently produce different results or fail entirely on another. Containers bundle all dependencies into an immutable image, making deployment reproducible.

The mental model to carry through this lecture is the **layered isolation stack**: Linux namespaces control what a container can see (processes, network, filesystem); cgroups control what a container can consume (CPU, memory, GPU); and the NVIDIA Container Runtime controls what GPU hardware and libraries the container can access. Together these three mechanisms give a container near-bare-metal performance while protecting the host from resource exhaustion and interference.

AI hardware engineers need to understand this stack because containerized inference is how TensorRT models are deployed to production at scale. Understanding cgroups is required for CPU pinning of latency-sensitive inference threads. Understanding the NVIDIA Container Runtime explains why GPU access works inside a container without installing CUDA in every image. And MIG (Multi-Instance GPU) enables multiple independent inference workloads to share an A100 or H100 with guaranteed performance isolation.

---

## Containers vs Virtual Machines

Containers **share the host kernel**. Isolation is provided by Linux namespaces (visibility) and cgroups (resource limits). There is **no hypervisor**, no guest kernel, and no hardware emulation.

| Property | Container | VM |
|---|---|---|
| Kernel | Shared with host | Separate guest kernel |
| Startup time | 5–50 ms | 1–10 s |
| Overhead | ~1% CPU | 5–15% CPU (hypervisor) |
| Isolation | Namespace + cgroups | Full hardware virtualization |
| GPU access | Direct (passthrough) | Requires GPU virtualization (vGPU) |

For inference deployment, containers provide **near-bare-metal GPU performance** with **reproducible software environments**.

> **Key Insight:** The reason containers achieve near-zero overhead compared to VMs is that there is no hypervisor between the container and the hardware. A GPU kernel inside a container is executed by the same physical GPU SMs that execute a native GPU kernel — there is no translation or emulation layer. The container's GPU access is just the host's `/dev/nvidia0` device node mounted into the container's filesystem namespace.

---

## Linux Namespaces

**Seven namespace types** isolate different aspects of the system environment. Each namespace is a **kernel object**; processes that share a namespace see the same view of that resource.

| Namespace | Isolates | Key detail |
|---|---|---|
| PID | Process tree | Container init = PID 1; cannot see host PIDs |
| Network | Network stack | Private interfaces, iptables, routing, ports |
| Mount | Filesystem view | `pivot_root` after overlayfs setup |
| UTS | Hostname, domain name | `uname -n` returns container name |
| IPC | System V IPC + POSIX shm | Prevents cross-container shared memory |
| User | UID/GID mapping | Container UID 0 → host UID 1000 (rootless) |
| Time (5.6+) | CLOCK_BOOTTIME/MONOTONIC offset | CRIU checkpoint/restore |

The relationship between namespaces:

```
Host namespace view:          Container namespace view:
  PID 1: systemd                PID 1: python inference_server.py
  PID 847: containerd           PID 2: /bin/sh (entrypoint)
  PID 1023: inference_server ←──── same process, different PID
  PID 1024: /bin/sh         ←────  same process, different PID
  eth0: 192.168.1.100           eth0: 172.17.0.2 (virtual veth)
  /: host rootfs                /: container overlayfs rootfs
                                    (host rootfs not visible)
```

### Key Operations

```bash
unshare --pid --fork --mount-proc bash   # new PID namespace with fresh /proc
ip netns add myns                        # create network namespace
ip netns exec myns ip addr               # run command inside namespace
nsenter -t <PID> --net --pid bash        # enter existing process's namespaces
```

These commands are what container runtimes do programmatically. `unshare` **creates new namespaces**; `nsenter` **joins existing ones**. When you `docker exec` into a running container, Docker calls `nsenter` to run your command in the container's existing namespace set.

> **Key Insight:** Namespaces do not provide security boundaries by themselves — they only control visibility. A process in a PID namespace cannot see host PIDs, but if it escapes its mount namespace (via a vulnerability), it can read host files. Real container security requires combining namespaces with seccomp filters, AppArmor profiles, and proper user namespace configuration.

---

## cgroups v2 (Unified Hierarchy)

Namespaces control **visibility**. cgroups control **resource consumption**. Together they define the complete container isolation model.

cgroups v2 replaced the per-subsystem hierarchy of v1 with a **single unified hierarchy** at `/sys/fs/cgroup/`. Controllers are enabled per-cgroup and inherited by children.

### Controller Reference

| Controller | Key file | Example value | Effect |
|---|---|---|---|
| `cpu` | `cpu.max` | `500000 1000000` | Limit to 50% of one CPU |
| `memory` | `memory.max` | `4G` | Hard memory limit; triggers OOM kill |
| `memory` | `memory.swap.max` | `0` | Disable swap for this cgroup |
| `cpuset` | `cpuset.cpus` | `4-7` | Restrict to cores 4–7 |
| `cpuset` | `cpuset.cpus.exclusive` | `1` | Exclusive assignment (Kubernetes static CPU) |
| `io` | `io.max` | `8:0 rbps=1073741824` | Limit to 1 GB/s read on device 8:0 |
| `pids` | `pids.max` | `256` | Maximum number of processes in cgroup |

### Creating and Managing cgroups

```bash
# Create a new cgroup for an inference process
mkdir /sys/fs/cgroup/inference

# Pin to cores 4-7 (isolated from OS and other workloads)
echo "4-7" > /sys/fs/cgroup/inference/cpuset.cpus

# Restrict to NUMA node 0 memory
echo "0"   > /sys/fs/cgroup/inference/cpuset.mems

# Hard memory limit: OOM kill if exceeded
echo "8G"  > /sys/fs/cgroup/inference/memory.max

# Disable swap: no latency spikes from swap I/O
echo "0"   > /sys/fs/cgroup/inference/memory.swap.max

# Move the current shell (and any processes it starts) into the cgroup
echo $$    > /sys/fs/cgroup/inference/cgroup.procs
```

After these commands, any process started from this shell runs with the cgroup constraints applied. The kernel **enforces the limits transparently** — the process does not need to be modified.

Kubernetes uses cgroups v2 for **all resource enforcement**. The kubelet creates a cgroup hierarchy per-Pod and per-container; the container runtime writes resource limits into the appropriate cgroup files.

> **Common Pitfall:** Setting `memory.max` without setting `memory.swap.max`. If swap is available and memory.max is hit, the kernel moves pages to swap instead of triggering an OOM kill. For an inference container, this causes catastrophic latency spikes (swap I/O is orders of magnitude slower than RAM). Always set both `memory.max` and `memory.swap.max` to the same value (or set `memory.swap.max` to 0 to disable swap entirely for the cgroup).

---

## Container Runtimes

The cgroup and namespace mechanics are orchestrated by the container runtime.

| Runtime | Role | Notes |
|---|---|---|
| `containerd` | High-level; manages images and snapshots | Default for Kubernetes (CRI) |
| `runc` | Low-level OCI runtime; creates namespaces and cgroups | Used by containerd |
| `crun` | C reimplementation of runc | 2–5× faster startup; lower memory |
| `kata-containers` | Lightweight VM per container | Stronger isolation; used for multi-tenant GPU |

The full container launch sequence:

1. **containerd** receives the OCI container spec from Kubernetes (or Docker).
2. **containerd** prepares the overlayfs rootfs from the image layers (using the snapshotter).
3. **containerd** forks `runc` with the OCI spec as input.
4. **runc** calls `clone()` with `CLONE_NEWPID | CLONE_NEWNET | CLONE_NEWNS` to create the new namespaces.
5. **runc** sets up the overlayfs mount and calls `pivot_root()` to switch the filesystem root.
6. **runc** writes resource limits to the cgroup files (memory.max, cpu.max, cpuset.cpus).
7. **runc** executes the container entrypoint (e.g., `python inference_server.py`).

`containerd` → `runc` → `clone()` with `CLONE_NEWPID | CLONE_NEWNET | CLONE_NEWNS` → set up overlayfs rootfs → `pivot_root` → write cgroup limits → exec container entrypoint.

> **Key Insight:** The container's process has been running in its new namespace since step 4, but it can only see its filesystem after `pivot_root` in step 5. The cgroup limits are written in step 6 before the application code starts in step 7. This ordering guarantees that by the time the application runs, all isolation and resource limits are already in place.

---

## NVIDIA Container Runtime

GPU access from inside a container requires special handling because **CUDA libraries and device nodes must be version-matched** to the host driver. The NVIDIA Container Runtime solves this elegantly.

`nvidia-container-toolkit` is an **OCI runtime hook** that runs before the container process starts. It:

1. **Reads `NVIDIA_VISIBLE_DEVICES` env var** (or `--gpus` flag): determines which physical GPUs or MIG instances to expose to this container.
2. **Mounts the selected `/dev/nvidia*` device nodes** into the container's mount namespace: `/dev/nvidia0`, `/dev/nvidiactl`, `/dev/nvidia-uvm`. The container now sees these devices as if they were local.
3. **Injects CUDA libraries** (`libcuda.so.X`, `libnvrtc.so`, `libnvidia-ml.so`) from the host into the container at well-known paths (`/usr/local/lib/...` or via `ldconfig` entries). The injected libraries match the host driver version exactly.
4. **Sets up `/proc/driver/nvidia` and capability files**: these are required by `libcuda.so` to query GPU capabilities.

The container image **does not need to contain CUDA**; only the CUDA application code is packaged. Host driver version is exposed as-is. This allows a single container image to run on hosts with **different driver versions**, as long as the driver is compatible with the required CUDA toolkit version.

```bash
# Run any CUDA application without CUDA installed in the image
docker run --gpus all --rm nvidia/cuda:12.2-base nvidia-smi

# Run TensorRT inference server with specific GPU and capabilities
docker run --gpus "device=0" -e NVIDIA_DRIVER_CAPABILITIES=compute,utility \
    my-trt-inference-server
```

The `NVIDIA_DRIVER_CAPABILITIES` variable controls which library sets are injected. `compute` injects CUDA compute libraries; `utility` injects `nvidia-smi` tools; `video` injects the NVENC/NVDEC video codec libraries.

> **Common Pitfall:** Forgetting to set `NVIDIA_DRIVER_CAPABILITIES=video` when running a container that uses NVENC or NVDEC for hardware video encoding/decoding. The container will launch successfully and CUDA compute will work, but any call to the video codec API will fail with a "driver not loaded" error. The video codec libraries are a separate injection set and are not included in the default `compute,utility` capabilities.

---

## Multi-Instance GPU (MIG)

MIG is available on A100, H100, and Jetson Orin. It partitions one physical GPU into up to **7 independent GPU Instances (GIs)**, each with dedicated:

- Compute (SM partition)
- Memory (HBM slice with dedicated bandwidth)
- On-chip caches and decoders

Each GI appears as an **independent GPU** to CUDA. There is **no time-sharing**; GIs run truly in parallel.

```
A100 80GB with MIG enabled:
┌─────────────────────────────────────────────────────────┐
│                     A100 Physical GPU                    │
│  ┌────────────┐  ┌────────────┐  ┌────────────────────┐ │
│  │   GI 0     │  │   GI 1     │  │       GI 2         │ │
│  │ 3g.40gb    │  │ 1g.10gb    │  │    1g.10gb         │ │
│  │ 42 SMs     │  │ 14 SMs     │  │    14 SMs          │ │
│  │ 40 GB HBM  │  │ 10 GB HBM  │  │    10 GB HBM       │ │
│  │            │  │            │  │                    │ │
│  │ TRT Model  │  │ TRT Model  │  │    TRT Model       │ │
│  │ (large)    │  │ (small)    │  │    (small)         │ │
│  └────────────┘  └────────────┘  └────────────────────┘ │
│     Truly parallel — no time sharing between GIs         │
└─────────────────────────────────────────────────────────┘
```

```bash
nvidia-smi mig -lgip                        # list GPU instance profiles
nvidia-smi mig -cgi 3g.40gb,1g.10gb -C     # create instances
# Container gets one GI via: --gpus "MIG-GPU-<uuid>"
```

On Jetson Orin, MIG enables running multiple inference models (perception, occupancy, prediction) on isolated GPU partitions with guaranteed memory bandwidth.

> **Key Insight:** MIG's critical property for production inference serving is **bandwidth isolation**, not just compute isolation. HBM memory bandwidth is dedicated per GI — a noisy model in GI 1 cannot steal memory bandwidth from GI 0. This is what makes it possible to guarantee SLA latency targets for individual models in a multi-model serving system. Without MIG, all models compete for the same HBM bus, and one large batch can cause latency spikes for all other models.

---

## GPU Sharing Without MIG

When MIG hardware is not available (older GPUs, edge devices), CUDA MPS provides a software-level sharing mechanism.

CUDA MPS (Multi-Process Service) allows **multiple CUDA processes to share a GPU** through a single MPS server process:

- Processes submit work via the MPS server; it serializes and batches submissions
- Reduces context switch overhead compared to time-sharing without MPS
- No memory isolation between clients (contrast with MIG)
- Use case: many lightweight inference processes sharing one GPU in a serving cluster

```bash
nvidia-cuda-mps-control -d    # start MPS daemon
export CUDA_MPS_PIPE_DIRECTORY=/tmp/nvidia-mps
# All CUDA processes launched after this will share via MPS
```

> **Common Pitfall:** Using CUDA MPS in a multi-tenant environment where containers from different users (or different trust levels) share the same GPU. MPS does not provide memory isolation — a bug in one client process can read or corrupt another client's GPU memory. MPS is appropriate for same-trust processes (e.g., multiple worker threads of the same inference service). For multi-tenant isolation, MIG is the correct mechanism.

---

## Kubernetes for AI Inference

Kubernetes orchestrates containerized inference across a cluster of GPU nodes.

### NVIDIA Device Plugin

The device plugin runs as a DaemonSet on each GPU node. It:

- Calls `nvidia-smi` to enumerate GPUs and MIG instances
- Registers `nvidia.com/gpu` as an Extended Resource with the kubelet
- Allocates specific device nodes to pods scheduled on the node

### Pod Specification

```yaml
resources:
  limits:
    nvidia.com/gpu: "1"     # request 1 GPU (or 1 MIG instance)
    cpu: "4"                # 4 CPU cores
    memory: "16Gi"          # 16 GB RAM
```

### Key Components

- **Node Feature Discovery (NFD)**: labels nodes with GPU model, CUDA version, driver version; used by scheduler for GPU-type-aware placement
- **Triton Inference Server**: containerized; dynamic batching; backends: TensorRT, ONNX Runtime, PyTorch; exposes gRPC + HTTP endpoints
- **DCGM Exporter**: exports GPU metrics (utilization, memory, temperature, NVLink bandwidth) to Prometheus

The complete production inference stack in Kubernetes:

```
Kubernetes Scheduler
        ↓ schedules pod to GPU node
Node (DaemonSet: NVIDIA Device Plugin)
        ↓ allocates /dev/nvidia0 to pod
containerd + NVIDIA Container Runtime
        ↓ sets up namespaces, cgroups, injects CUDA libs
Pod: Triton Inference Server container
        ↓ loads TensorRT engine
CUDA / TensorRT → /dev/nvidia0 → A100 GPU
        ↓
Model output → gRPC response to client
```

> **Key Insight:** The DCGM Exporter feeds GPU metrics into Prometheus, which feeds Grafana dashboards and autoscaler rules. In a production inference cluster, autoscaling decisions (spin up more replicas) are triggered by GPU utilization and request queue depth metrics exported by DCGM. This closes the loop: Kubernetes provides resource isolation (cgroups, namespaces), the device plugin provides GPU access, Triton provides efficient batching, and DCGM provides observability — all four are required for a production-grade inference service.

---

## Summary

| Isolation mechanism | Kernel feature | Resource type | Primary tools |
|---|---|---|---|
| Process visibility | PID namespace | Process tree | `unshare`, `nsenter` |
| Network isolation | Network namespace | Interfaces, ports | `ip netns`, `veth` |
| Filesystem isolation | Mount namespace + overlayfs | Rootfs | `pivot_root`, `containerd` |
| CPU limits | cgroups v2 `cpu` controller | CPU time | `cpu.max`, kubelet |
| Memory limits | cgroups v2 `memory` controller | RAM + swap | `memory.max` |
| CPU pinning | cgroups v2 `cpuset` controller | CPU cores | `cpuset.cpus` |
| GPU access | NVIDIA Container Toolkit | Device nodes + libraries | `--gpus`, device plugin |
| GPU partitioning | MIG (hardware) | SM + HBM slices | `nvidia-smi mig` |

### Conceptual Review

- **Why do containers achieve near-bare-metal GPU performance when VMs require GPU virtualization?** A container's GPU access is the host `/dev/nvidia*` device node mounted into the container's filesystem namespace. The CUDA calls inside the container execute through the same `nvidia.ko` kernel driver as a native application. There is no hypervisor layer, no VGPU translation, no para-virtualization. The GPU hardware executes the kernels directly. VMs require a vGPU driver that translates guest GPU commands through a hypervisor layer, adding overhead.

- **What is the difference between namespace isolation and cgroup resource control?** Namespaces control what a process can see: with a PID namespace, the container cannot see host processes; with a network namespace, it cannot see host network interfaces. But nothing stops the container from consuming all available CPU cycles or RAM — namespaces are about visibility, not resource consumption. cgroups add the resource dimension: `cpu.max` caps CPU time, `memory.max` caps RAM usage. Together, namespaces + cgroups = complete container isolation.

- **Why does the NVIDIA Container Runtime inject libraries from the host rather than packaging them in the container image?** CUDA libraries must exactly match the kernel driver version (`libcuda.so.X` links against `nvidia.ko` internal ABI). If the container bundled its own CUDA libraries, they would need to be rebuilt every time the host driver is updated. By injecting from the host at runtime, a single container image works across any host with a compatible driver version. This also dramatically reduces image size (CUDA libraries are several GB).

- **What is the key operational difference between MIG and CUDA MPS for multi-model inference?** MIG is hardware partitioning: each GPU Instance has dedicated SMs, dedicated HBM bandwidth, and dedicated caches. Interference between GIs is physically impossible. MPS is software multiplexing: the MPS server serializes CUDA commands from multiple processes, reducing context-switch overhead but providing no memory isolation — processes can interfere with each other's GPU memory. MIG provides guaranteed performance isolation; MPS provides only scheduling efficiency.

- **How does the Kubernetes device plugin coordinate GPU allocation with the NVIDIA Container Runtime?** The device plugin advertises available GPU resources (`nvidia.com/gpu`) to the kubelet. When a pod requesting a GPU is scheduled to the node, the kubelet queries the device plugin for a specific device (e.g., `/dev/nvidia0`). The device plugin returns the device path and any required environment variables (e.g., `NVIDIA_VISIBLE_DEVICES=0`). The kubelet passes these to the container runtime, which (via the NVIDIA Container Runtime hook) uses `NVIDIA_VISIBLE_DEVICES` to determine which device nodes and libraries to inject.

- **What does `cpuset.cpus.exclusive` provide that `cpuset.cpus` alone does not?** `cpuset.cpus` restricts a cgroup to running on specific cores, but other cgroups can also run on those same cores. `cpuset.cpus.exclusive` claims those cores exclusively for this cgroup — no other cgroup can be scheduled on them. This is what Kubernetes Static CPU Manager uses to fully dedicate cores to latency-sensitive inference containers, eliminating OS scheduling jitter from other workloads sharing the same cores.

---

## AI Hardware Connection

- NVIDIA Container Runtime enables TensorRT and Triton inference containers to access GPU hardware at native performance; the host driver is injected at container start, eliminating the need to rebuild images per driver version
- MIG on A100/H100/Orin partitions the GPU for multi-tenant inference serving; each model gets a guaranteed memory bandwidth slice, preventing one workload from starving another
- cgroups v2 `cpuset.cpus.exclusive` in Kubernetes static CPU manager pins inference worker threads to dedicated cores, eliminating OS scheduler interference and reducing tail latency
- Rootless containers (user namespace UID remapping) are the correct security posture for edge AI deployments where the container user must not map to a privileged host account
- CUDA MPS enables high-concurrency serving scenarios where many lightweight inference processes (e.g., per-sensor model instances) share one GPU without the overhead of full context switches
- The full stack — cgroups v2 resource isolation + NVIDIA device plugin + Triton dynamic batching — is the production reference architecture for scalable GPU inference in Kubernetes
