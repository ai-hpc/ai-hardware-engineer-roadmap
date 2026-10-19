# Choose a working Jetson and ROS software stack

**Source check: 18 October 2026.** Read this before installing packages or following a board-flashing example. The course contains useful JetPack 6 examples, but JetPack 6 is no longer the newest Jetson software generation.

A working stack is a set of compatible parts: **module + carrier + firmware + board support package (BSP) + operating system + CUDA libraries + application**. Matching one version number does not prove that the complete set works.

> **TensorRT package boundary:** as checked on 2026-10-18, NVIDIA states that TensorRT 11.4 does not support JetPack. Use the TensorRT 10.x release supported by the installed JetPack. DLA also requires a supported module and runtime combination; installing the newest generic package is not an upgrade path. See the [official Jetson migration status](https://docs.nvidia.com/deeplearning/tensorrt/latest/api/migration/tensorrt-10x-to-11x-jetson.html).

## 1. Choose the hardware first

Record the exact module and carrier separately. An Orin Nano developer kit, an Orin Nano production module on a custom carrier, and an Orin NX module can share software concepts while requiring different flash configurations, pin assignments, and power limits.

Orin Nano has no DLA or hardware video encoder. A family diagram, installed DLA compiler package, or device-tree entry does not add these engines. Use GPU inference and an explicitly supported encoding path; DLA exercises require another supported module. Check the [module specifications](https://www.nvidia.com/en-us/autonomous-machines/embedded-systems/jetson-orin/) and the [Orin Nano software encoding guide](https://docs.nvidia.com/jetson/archives/r36.4.4/DeveloperGuide/SD/Multimedia/SoftwareEncodeInOrinNano.html).

## 2. Distinguish current releases from teaching baselines

| Stack | BSP / base OS | Role in this course |
|---|---|---|
| **JetPack 7.2.1** | Jetson Linux **39.2.1**, Ubuntu **24.04**, kernel **6.8** | Current release checked for Orin and Thor; CUDA **13.2.2**, TensorRT **10.16.2** |
| **JetPack 6.2.3** | Jetson Linux **36.5.2**, Ubuntu **22.04**, kernel **5.15** | Current checked maintenance release in the 6.x line for Orin |
| JetPack 6.2.1 | Jetson Linux 36.4.4, Ubuntu 22.04, kernel 5.15 | Older reference baseline; CUDA 12.6 and TensorRT 10.3 |
| Other releases in a lesson | Read that release's own manifest | Historical examples or explicitly pinned experiments |

Sources: [7.2.1 archived release manifest](https://developer.nvidia.com/embedded/jetpack/downloads/archive-7.2.1), [36.5.2 release](https://developer.nvidia.com/embedded/jetson-linux-r3652), [6.2.1 release](https://developer.nvidia.com/embedded/jetpack-sdk-621).

The installation route changed too: the 7.2.1 release page uses a unified ISO route and says it no longer provides the Orin Nano SD-card image used by older tutorials. Select instructions for your release and board; do not apply the old SD-image procedure to a new release by changing its download URL.

Upgrading the BSP, replacing the Ubuntu root filesystem, and updating the CUDA user-space stack are different operations. A container shares the host kernel and uses host driver interfaces. A new container base image cannot make an incompatible BSP or camera driver compatible.

## 3. Match ROS and Isaac ROS separately

**ROS 2** supplies robot middleware and libraries. **Isaac ROS** is a versioned collection of NVIDIA accelerated packages. Installing ROS 2 does not install Isaac ROS, and installing the newest JetPack does not prove support for every Isaac ROS release.

| Learning or deployment route | Explicit pairing | What this establishes |
|---|---|---|
| Course ROS fundamentals lab | Ubuntu 24.04 + ROS 2 Jazzy | Actual local middleware/TF test recorded in the [ROS lab](../Phase%205%20-%20Advanced%20Topics%20and%20Specialization/Track%20D%20-%20Robotics/Robotics%20Foundations/ROS2-Lab.md); no Jetson acceleration claim |
| Existing JetPack 6 application | Ubuntu 22.04 + ROS 2 Humble | Standard Ubuntu package pairing; accelerated packages still need their own matrix |
| Isaac ROS **4.6** | Its matrix names JetPack **7.2**, ROS 2 **Jazzy** | Use the versioned 4.6 setup, package repositories, and requirements |
| Isaac ROS **5.0** | Its matrix names JetPack **7.2**, ROS 2 **Lyrical** | Follow NVIDIA's version-specific buildfarm instructions; this is not a drop-in Jazzy package upgrade |

Sources: [ROS platform policy](https://www.ros.org/reps/rep-2000.html), [Humble Ubuntu installation](https://docs.ros.org/en/humble/Installation/Ubuntu-Install-Debs.html), [Isaac ROS 4.6 requirements](https://nvidia-isaac-ros.github.io/v/release-4.6/getting_started/index.html), [Isaac ROS 5.0 requirements](https://nvidia-isaac-ros.github.io/v/release-5.0/getting_started/index.html).

**Read exact versions.** The 7.2.1 JetPack manifest still says Isaac ROS is “Coming soon”, while the versioned Isaac ROS pages name JetPack 7.2. These pages do not explicitly establish every 7.2.1 combination. Record that discrepancy and require a version-specific support statement or an integration test for your selected combination. Do not silently promote 7.2 to 7.2.1 in a compatibility claim.

## 4. Make a small environment record

Run these read-only commands on the target. A missing command or package is a result to record, not an instruction to install arbitrary replacements.

```bash
uname -m
uname -r
cat /etc/os-release
cat /etc/nv_tegra_release
tr '\0' '\n' < /proc/device-tree/model
dpkg-query -W nvidia-l4t-core nvidia-jetpack
dpkg-query -W 'libnvinfer*' 'cuda-cudart*'
```

The JetPack metapackage may be absent even when individual components are installed. `/etc/nv_tegra_release` and `nvidia-l4t-core` identify the BSP; package queries identify installed user-space components. `nvcc --version`, when available, identifies a compiler toolkit. It is not a complete inventory of the driver or libraries used by a running program.

In a terminal where the intended ROS setup has been sourced:

```bash
printenv ROS_DISTRO
printenv RMW_IMPLEMENTATION
ros2 doctor --report
```

An unset `RMW_IMPLEMENTATION` means the default selection is being used, not that ROS has no middleware. Save the resolved middleware from the diagnostic report. Record a container's image digest and its environment separately from the host record.

Use this small table in every deployment report:

| Field | Your observed value |
|---|---|
| Module / carrier revision | |
| BSP and boot medium | |
| Host OS / kernel | |
| CUDA runtime / TensorRT / framework | |
| ROS distro / RMW / Isaac ROS release if used | |
| Container digest, model revision and model hash | |
| Power mode / cooling / ambient conditions | |
| Matching official release page | |
| Checks actually run; checks still pending | |

## 5. First deployment exercise

1. Fill in the table before changing the device.
2. Choose one supported inference example for that stack. Preserve its model, preprocessing, input shape and numerical reference.
3. Compare outputs on a fixed input set before timing anything. Explain the tolerance for your dtype and task.
4. Run an end-to-end trace: acquisition, preprocessing, inference, postprocessing, publication, command validation. Keep clock domains explicit.
5. Repeat after sustained load and record errors, queue depth, memory, temperature, power mode, median and tail latency.
6. Change one component at a time and repeat the same correctness and timing checks.

**No Jetson available?** Complete the [CPU compiler lab](../Phase%204%20-%20Track%20C%20-%20DL%20Inference%20Optimization/00%20-%20CPU%20Compiler%20Lab/Guide.md) and the [ROS integration lab](../Phase%205%20-%20Advanced%20Topics%20and%20Specialization/Track%20D%20-%20Robotics/Robotics%20Foundations/ROS2-Lab.md). Mark the environment record “host-only”. These exercises teach interfaces and validation without claiming GPU, camera, power, or physical-robot results.

## Check your understanding

1. **A Jazzy node runs on your Ubuntu 24.04 laptop. Is its TensorRT engine ready for Jetson?** No. Establish target runtime, platform and engine compatibility, then check numerical outputs on the target. Rebuild the engine when the chosen TensorRT portability rules require it.
2. **`nvidia-jetpack` is absent. Is CUDA definitely missing?** No. Query actual installed components and the libraries loaded by the application.
3. **A new container starts successfully. Has camera support been validated?** No. Camera capture depends on host BSP, sensor driver, configuration and device access, as well as the application.
4. **Two documentation pages disagree. Which wins automatically?** Neither. Record both exact pages and dates, narrow the claim, and resolve the specific combination before deployment.

Continue with the [platform guide](1.%20Nvidia%20Jetson%20Platform/Guide.md) and the [ROS application course](5.%20Application%20Development/6.%20ROS2/Guide.md).

## DeepStream selection

Checked 2026-10-18, DeepStream 9.1 lists Thor and Orin with JetPack 7.2 GA/L4T 39.2; 8.0 and 9.0 Jetson releases list Thor, while 7.1 remains an older Orin/JetPack 6.1 baseline. The matrix names 7.2, so verify support for a specific 7.2.1 maintenance combination instead of assuming it. Current traditional Python `pyds` bindings are deprecated in favor of Service Maker. See the [official platform matrix](https://docs.nvidia.com/metropolis/deepstream/dev-guide/text/DS_Installation.html), [binding status](https://docs.nvidia.com/metropolis/deepstream/dev-guide/text/DS_Python_Sample_Apps.html), and the [video course with an executed CPU lab](1.%20Nvidia%20Jetson%20Platform/Orin-Nano-Video-Codec-DeepStream/Guide.md).
