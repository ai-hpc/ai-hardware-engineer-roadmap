# Lecture 20: PCIe, NVMe, and GPU Driver Architecture

When an inference pipeline is slow, ask **which bytes cross which connection**. Discrete GPUs, many NVMe controllers, NICs, and FPGA endpoints use PCIe. Integrated GPUs such as Jetson Orin's GPU are not discrete PCIe endpoints attached to their host CPU. Begin with the actual topology, not a generic server drawing.

## A topology is a route and a set of shared limits

```text
CPU / root complex
 ├─ root port → PCIe switch → discrete GPU 0
 │                        ├→ discrete GPU 1
 │                        └→ NVMe controller
 └─ root port → NIC
```

A switch uplink can be shared by several endpoints. PCIe lane width and generation are negotiated; a physical x16 slot does not prove an x16 link. On a suitable host, `lspci -t` shows topology and `lspci -vv -s ADDRESS` shows capabilities and negotiated link state. Replace ADDRESS with an enumerated device address. These are inspection commands, not measurements of payload bandwidth.

For PCIe Gen3–Gen5, the 128b/130b encoding calculation gives the following ceilings **per direction**, before packet/protocol overhead:

| Generation | Rate per lane | x4 ceiling | x16 ceiling |
|---|---:|---:|---:|
| Gen3 | 8 GT/s | 3.94 GB/s | 15.75 GB/s |
| Gen4 | 16 GT/s | 7.88 GB/s | 31.51 GB/s |
| Gen5 | 32 GT/s | 15.75 GB/s | 63.02 GB/s |

For example, Gen4 x16 gives `16e9 × (128/130) × 16 / 8` bytes/s. Adding both directions produces about 63.02 GB/s, but a host-to-device transfer does not get both directions. A 1 GiB transfer divided by 31.51 GB/s gives about 34.1 ms as an idealized link-only lower bound; real transfers include overhead and other bottlenecks.

## Configuration space and BARs

Firmware and the OS enumerate endpoints, discover capabilities, and assign address windows. PCIe extends conventional configuration space to 4 KiB. BARs describe device memory or I/O resources. They are not ordinary RAM allocations. Drivers use the PCI resource/mapping APIs and device-defined accessors rather than casting an arbitrary integer to a C pointer.

GPU BAR labels and sizes are device/driver-specific. A framebuffer aperture may expose a portion of VRAM to peers or the CPU. Resizable BAR can increase an aperture when the platform and device support it. It does not increase PCIe bandwidth, make all CPU accesses coherent with every GPU operation, or constitute a universal prerequisite that “all VRAM must be CPU-addressable” for GPUDirect Storage.

A device performs DMA using addresses supplied through its mapping contract. An IOMMU may translate those addresses and limit access. Device DMA addresses, CPU virtual addresses, and CPU physical addresses are distinct concepts. Review [Lecture 15](Lecture-15.md) before writing a DMA descriptor.

## Peer-to-peer transfers are conditional

Two endpoints behind a switch may be able to exchange data without staging the payload in host RAM. That requires compatible hardware, routing, mappings, and drivers. Sharing a switch is useful evidence, not a proof. ACS can redirect traffic, and IOMMU behavior varies by platform and driver. Do not disable isolation mechanisms as a generic performance recipe; use the supported topology and mappings for the intended deployment.

The [Linux PCI peer-to-peer documentation](https://docs.kernel.org/driver-api/pci/p2pdma.html) describes the kernel framework. Vendor GPU APIs may use additional mechanisms; `pci_p2pmem_alloc_sgl()` is not an application-level prerequisite for every CUDA peer transfer. A route through a root complex does not necessarily mean staging through system RAM.

GPUDirect Storage uses supported storage/GPU paths through `cuFile`; it can have compatibility paths and platform restrictions. CPU work is still needed for control, submission, and application logic even when payload DMA avoids a CPU copy. Check the [GDS design guide](https://docs.nvidia.com/gpudirect-storage/design-guide/index.html) and release support matrix before promising a direct path. A Jetson system does not inherit every discrete datacenter GPU feature.

## NVMe queues and the Linux block layer

NVMe defines submission and completion queues and controller-advertised limits. The protocol's large maximum queue counts do not mean that a consumer SSD implements thousands of independent NAND operations or that Linux provisions the maximum. Actual queue depth, controller resources, flash channels, and workload determine concurrency.

Linux's `blk-mq` maps software submissions onto hardware queues. MSI-X vectors may be shared, affinitized, or unavailable for some queues; polling queues can avoid completion interrupts. A completion is not guaranteed to interrupt exactly the CPU that submitted the request. `io_uring` can submit buffered or direct I/O through filesystem/block paths; it does not universally bypass the page cache or dispatch directly to a controller.

Do not confuse SATA SSDs with hard disks: a blanket “SATA SSD read = 5 ms” is incorrect. Compare measured devices under the same I/O size, queue depth, cache state, and durability requirements. Sequential bandwidth says little about the tail of synchronous small writes.

**Worked queue model:** at 100,000 completions/s and an average measured request latency of 200 µs, Little's law gives average outstanding requests `L = λW = 20`, under a stable steady workload and consistent request boundary. This does not say that setting queue depth to 20 guarantees that rate or its tail latency.

## The GPU software stack

```text
application / framework
    ↓ CUDA runtime or driver API
user-mode driver and supporting libraries
    ↓ setup/control calls plus mapped submission mechanisms
kernel modules, device firmware, GPU queues
    ↓ execution and completion synchronization
```

`libcuda.so` provides the CUDA driver API; the CUDA runtime normally supplies `libcudart` or a static equivalent. Kernel modules manage privileged resources and integrate with the OS. Unified-memory handling is associated with `nvidia-uvm`; display paths involve modeset/DRM components. The exact submission path is implementation-specific: do not invent an ioctl named `NV_ESC_RM_DMA_COPY` or assume every `cudaMemcpyAsync()` generates one syscall and one interrupt.

Supported NVIDIA GPUs can use open kernel modules while firmware and user-mode components retain their own licenses. The [NVIDIA kernel-module guide](https://docs.nvidia.com/datacenter/tesla/driver-installation-guide/kernel-modules.html) documents architecture and release restrictions. GSP is a firmware-managed resource-management component; it is not accurately described as an ARM Cortex-A9 universally present since Ampere.

Other Linux GPU stacks include `amdgpu`, Intel `i915`/`xe`, and Nouveau with Mesa userspace drivers. An open kernel driver does not imply that every firmware binary is open source. ROCm is a compute stack, not simply a Mesa subcomponent. Check the exact GPU and software support matrix for compute use.

## Contexts, streams, and lifetime

A CUDA stream orders its commands according to CUDA semantics. Multiple streams permit overlap when dependencies, resources, and hardware allow it; they do not force overlap. Events establish completion/order relationships, and host-visible timing needs appropriate synchronization.

The runtime typically uses a device's primary context. Explicit driver contexts require deliberate current-context and allocation ownership management. Passing a pointer between unrelated contexts or processes does not establish a valid shared mapping; use the supported interop/sharing API. Discrete VRAM and integrated shared DRAM have different movement costs, but both require synchronization.

## FPGA endpoints and a practical review

An FPGA endpoint can expose DMA channels through an IP core such as AMD XDMA. User nodes and transfer APIs depend on the selected driver. Programming a bitstream or partial region requires a supported management/configuration path; ordinary data DMA does not automatically reconfigure the FPGA. FPGA-to-GPU P2P additionally needs supported GPU-memory registration and completion semantics.

Before optimizing a path, produce a table of buffer owner, allocator, address type, source/destination devices, transfer bytes, and completion signal. Then record the negotiated links and the measured path. No PCIe latency table can replace those facts.

**Check:** Does a Gen4 x16 GPU receive 64 GB/s from the CPU? **Answer:** No. Its encoding ceiling is about 31.51 GB/s in one direction before protocol overhead.

**Check:** Does zero host-memory staging imply zero CPU usage? **Answer:** No. Control, submission, synchronization, and application processing can still consume CPU.

**Check:** Does a device with 64 queues provide 64 interrupt vectors? **Answer:** Not necessarily; controller resources and OS allocation determine the mapping.
