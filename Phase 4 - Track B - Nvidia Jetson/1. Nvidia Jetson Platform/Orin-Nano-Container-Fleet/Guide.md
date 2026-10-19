# Container deployment and fleet operation on Jetson

A container packages an application's userspace. A working deployment also depends on the host, hardware access, model and configuration, persistent data, and lifecycle. This chapter starts with an actual small CPU container, then extends the same evidence to a Jetson application.

**Outcomes:** identify an immutable image; distinguish liveness from readiness; constrain a service; test GPU/device access; preserve rollback; and report fleet health without losing failed devices from the denominator. Plan three 60–90-minute sessions. Read [memory architecture](../Orin-Nano-Memory-Architecture/Guide.md) and [OTA recovery](../Orin-Nano-OTA-Deep-Dive/Guide.md) alongside this chapter.

**Reviewed 2026-10-18.** Select the exact host and runtime using the [Jetson compatibility matrix](../../Compatibility.md). Orin Nano has no DLA. Historical R36.3 container tags are not a current universal baseline, and TensorRT 11.4 does not establish JetPack support. Container/runtime compatibility needs the publisher's supported combination and an executed workload, not a matching major number alone.

## 1. Define the deployment identity

Record these inputs together:

| Input | Why it matters |
| --- | --- |
| Module and carrier | GPU capability, sensors, device nodes and electrical interfaces |
| Host BSP/kernel/driver | Kernel interfaces and platform runtime support |
| Container image digest and platform | Exact userspace bytes and CPU architecture |
| Container engine/toolkit configuration | How devices and libraries reach the process |
| Model/configuration hashes | Input contract, computation and output behavior |
| Volumes and credentials | State and access outside the image |
| Runtime limits and power mode | Resource availability and sustained timing |

A tag can be moved to new content. A digest identifies content; for a multi-platform image, also record the selected architecture and resolved image identity. A digest does not establish who authorized that content or whether it works with your host.

Keep a tested host/image matrix. Include one intentionally incompatible case and its diagnostic. The [local container lab](labs/README.md) supplies a parameter mismatch that remains live while failing readiness; a Jetson extension should add a real runtime/host mismatch under a recoverable setup.

## 2. Run the complete CPU service

Complete [Run, reject, and stop an edge service](labs/README.md). It includes the service, parameters, tests, a digest-pinned Dockerfile, and a Docker harness. No learned model or GPU is required for the first steps.

The service evaluates a deterministic stop rule. It validates input shape/range/finiteness, limits request size, reports the loaded parameter digest, and performs known-input checks before declaring readiness. The optional Docker harness actually builds and starts two containers, checks the constrained runtime, sends requests, observes health transitions, and stops them.

The retained amd64 run demonstrates a useful distinction: a container with incorrect parameters can remain running and respond to liveness while readiness and inference fail. An unconditional `{"status":"healthy","gpu":"available"}` endpoint would hide that failure.

For a real learned model, replace the rule only after retaining the lifecycle tests. Add a known tensor, expected outputs/tolerances, output shape checks, invalid-input cases, and representative quality evaluation. Use the [CPU robot-stop training lab](../../../Phase%203%20-%20Artificial%20Intelligence/2.%20Deep%20Learning%20Frameworks/PyTorch/CPU-Robot-Stop-Lab/Guide.md) for actual training evidence; the container example itself does not train a model.

## 3. Understand isolation and resource limits

Containers share the host kernel. Their processes can still compete for CPU time, memory bandwidth, storage I/O, and physical DRAM. Separate model processes may allocate separate weights, workspaces, and CUDA contexts even on a shared-memory SoC. A common physical memory pool does not deduplicate those allocations automatically.

Set explicit limits suited to the workload and verify their effective values. Docker exposes CPU and memory controls, including separate memory/swap behavior. A CPU quota constrains available execution time; it is not a real-time reservation. Memory accounting and GPU allocations require observation on the specific platform. [Docker resource constraints](https://docs.docker.com/engine/containers/resource_constraints/).

The lab's 64 MiB memory limit and half-CPU quota are for its tiny CPU service. They are not recommended values for a TensorRT application. Budget peak initialization, models, activation/workspace buffers, camera queues, runtime libraries, caches, and simultaneous candidate/previous versions during an update.

Inspect a service using its exact name or ID:

```bash
docker inspect CONTAINER_NAME
docker stats --no-stream CONTAINER_NAME
docker logs --tail 100 CONTAINER_NAME
```

These are templates. An OOM-killed process, CPU throttling, and an application exception are distinct failures. Record the exit state and relevant host/application evidence before deciding on a fix.

## 4. Configure NVIDIA access from the supported host

The NVIDIA Container Toolkit integrates device/library access with supported container engines. Its installation guide documents `nvidia-ctk runtime configure` for Docker and containerd; the configuration and any required daemon restart affect the host. Use the Jetson release's supported packages and platform guidance rather than installing a generic discrete-GPU driver onto the board. [Toolkit installation](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html), [platform support](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/supported-platforms.html).

The toolkit also supports Container Device Interface (CDI). Inspect the configured mechanism and available devices instead of assuming every release uses an identical set of legacy mounts. Avoid combining competing injection mechanisms without understanding the documented configuration. [CDI support](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/cdi-support.html).

A useful validation ladder is:

1. Record the host BSP, kernel, toolkit, engine, module, and image identity.
2. Confirm the intended devices and libraries are available to the selected user.
3. Run a tiny CUDA calculation with a numerical reference and explicit synchronization/error checks.
4. Load the real model and run a known-input inference.
5. Test representative sustained input, resource pressure, and restart behavior.

A device listing or successful `nvidia-smi` invocation, when available, does not establish model execution. An image importing CUDA/TensorRT on its own does not establish a successful GPU operation either. The [CUDA course](../Orin-Nano-CUDA-Programming/Guide.md) and [real-time inference course](../Orin-Nano-Real-Time-Inference/Guide.md) supply the next computational and timing questions.

## 5. Give devices and files the access they need

Start from an explicit device/interface inventory: USB V4L2 camera, CSI/Argus camera path, serial controller, CAN socket, model directory, output directory, and required services. Different paths need different host integration. A visible `/dev/video0` does not prove that a CSI/ISP pipeline is configured or accessible.

Test camera capture on the host first, then reproduce the required access inside the container. Record device ownership/group IDs, sockets, plugin versions, and buffer formats. A group name inside the image may map to a different numeric group ID than on the host.

Use a non-root application user, read-only model inputs when practical, and narrowly scoped writable data. Avoid treating a privileged container or a host-wide writable mount as the permanent answer to an unexplained permission failure. The local lab proves its own limited permissions; hardware services need their own inventory and tests.

A container can isolate userspace packages while still having substantial host authority through a mounted Docker socket or device. Include those interfaces in the [security design](../Orin-Nano-Security/Guide.md).

## 6. Separate liveness, readiness, and task health

Liveness asks whether a process can respond. Readiness asks whether it can accept a defined request. Task health asks whether the full application continues to meet its requirements, including data freshness and output quality.

Docker `HEALTHCHECK` reports starting, healthy, or unhealthy based on its command and thresholds. It does not itself define your application's readiness semantics. A restart policy concerns container exit/restart behavior; an unhealthy status alone is not automatic rollback of a running process. [Docker health checks](https://docs.docker.com/reference/dockerfile/#healthcheck), [restart policies](https://docs.docker.com/engine/containers/start-containers-automatically/).

For robotics, add sensor freshness, frame/calibration identity, inference validity, command age, and control-process acknowledgment where required. Decide which failures withdraw readiness, which stop admission, which request bounded restart, and which require a broader recovery action.

A health probe can also create load or fail because of an unrelated network dependency. Keep its work bounded and its meaning explicit. Tie acceptance to the current boot and deployment identity so a delayed success from the old process cannot accept a new candidate.

## 7. Design handover and shutdown

Use exec-form entrypoints or an appropriate init so signal behavior is understood. On shutdown, stop admitting new work, mark the service unavailable to dispatchers, account for in-flight operations, finish or abandon them according to a bounded policy, release resources, and report the result.

Two processes sharing one port is not a complete zero-downtime deployment design. A robot additionally needs exclusive command authority: both versions must not send competing actuator commands during a handover. Define who owns that authority and how stale commands are rejected.

A candidate model can be loaded and checked before activation if memory permits. Preserve the previous compatible model and configuration until acceptance. A changed symlink does not replace an engine already loaded into memory, nor does it synchronize outstanding GPU work.

Use the [OTA lab](../Orin-Nano-OTA-Deep-Dive/labs/README.md) to practice bounded trial/acceptance and stale-result rejection. Container rollout still needs compatible persistent data and a recovery copy.

## 8. Make networking and storage part of the experiment

Bridge networking, published ports, host networking, and orchestrated networking provide different connectivity and isolation. Measure the required communication path instead of asserting that host networking always improves application latency. The CPU lab publishes only on host loopback because its teaching HTTP service has no authentication or TLS.

For ROS 2, discovery and data transport depend on the selected middleware, network interfaces, multicast/unicast configuration, domain and QoS settings. Test communication across the actual host/container boundary with the expected message size and rate. Use the [ROS application course](../../5.%20Application%20Development/6.%20ROS2/Guide.md) to develop the message/freshness contract.

Separate writable application data from immutable image content. Bound logs and caches and monitor available space on the relevant mounts. Do not remove every unused image as a routine update step: a stopped previous image may be the intended rollback artifact. Use a retention policy based on deployment identity and recovery needs.

Overlay layers, bind mounts, and named volumes have different lifetimes. Record which files survive container recreation and how old and new applications read shared data. See the [rootfs/storage guide](../Orin-Nano-Rootfs-and-AB-Redundancy/Guide.md).

## 9. Verify supply-chain claims, then test behavior

Pin the base image and dependency inputs, build for the intended architecture, retain source/build metadata, and scan the resulting artifacts under a defined policy. A clean scanner result does not establish the absence of vulnerabilities or the application's correctness.

A signature check must bind the intended artifact digest to an accepted signer policy. With Sigstore keyless verification, identity and OIDC issuer constraints are explicit verification inputs; accepting any identity defeats that policy. The actual trust model and offline requirements determine the verification procedure. [Cosign verification](https://docs.sigstore.dev/cosign/verifying/verify/).

Docker Content Trust is a specific client-side workflow, not a generic daemon setting that makes every container authorized. Do not copy an invented `content-trust` field into `daemon.json` and assume image enforcement exists. [Docker Content Trust](https://docs.docker.com/engine/security/trust/).

Keep secrets out of image layers and build output. Build-time secret handling and runtime credential provisioning are separate paths. A cross-compiled or emulated arm64 build still needs a native hardware run for GPU, camera, and timing behavior.

## 10. Operate a fleet with measurable acceptance

For each device, retain intended and observed deployment IDs, host identity, last contact, update phase, health result, rollback count, and task-quality/timing metrics. Avoid silently excluding unreachable devices from an update-success calculation.

A staged rollout should specify the cohort, observation duration, acceptance criteria, halt condition, and recovery mechanism. Include representative hardware and environmental variants. A high success rate across reachable devices does not prove the unreachable devices are healthy.

Track request arrivals, accepted inputs, queue drops, successful outputs, deadline misses, and errors separately. For an edge robot, useful work is often a sufficiently fresh, correct result—not merely an HTTP 200 response. Compare distributions and failure cases under sustained thermal conditions.

**Completion:** the local service tests and optional actual Docker report; a host/image/model compatibility manifest; one negative readiness test; a bounded restart/handover plan; and a device experiment covering known-input GPU inference, sensor access, fresh output, and recovery. Hardware-specific container results remain unexecuted until that experiment is run on the intended Jetson.
