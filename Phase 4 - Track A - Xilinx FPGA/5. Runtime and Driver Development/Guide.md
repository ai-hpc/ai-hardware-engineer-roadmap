# Track A §5 — Runtime & Driver Development for FPGA Accelerators

<div class="course-identity fpga-runtime" markdown="1">
<div class="course-identity__icon">DRV</div>
<div markdown="1">
<p class="course-identity__eyebrow">Track A5 · FPGA Runtime & Drivers</p>
<p class="course-identity__title">Expose FPGA accelerators through drivers, userspace runtimes, and production control paths.</p>
<p class="course-identity__meta">Artifact: driver/runtime interface · Measure: latency, throughput, errors, recovery</p>
</div>
</div>


> *Connect your FPGA accelerator to the host — write the runtime, drivers, and memory management that make hardware usable from software.*

**Prerequisites:** Track A §1–2 (Vivado, Zynq PS/PL, AXI), Phase 1 §3 (Operating Systems — drivers, memory, system calls), Phase 1 §4 (C++).

**Layer mapping:** **Layer 3** (Runtime & Driver) of the AI chip stack. This module bridges Layer 6 (your RTL/HLS accelerator) with Layer 1 (the AI framework or application calling into it).

**Role targets:** FPGA Runtime Engineer · Accelerator Driver Developer · SoC Platform Engineer · Embedded Linux BSP Engineer

---

## Start with a request, not a register write

An application asks an accelerator to transform a tensor. The runtime must establish what bytes those tensor elements occupy, where the device can access them, who owns the storage, how completion is recognized, and what happens if it never arrives. Correct arithmetic inside the FPGA does not repair a host that frees a live DMA buffer.

Start with the [checked CPU matrix-vector contract](../4.%20High-Level%20Synthesis%20%28HLS%29/labs/README.md) and [simulated ready/valid stage](../1.%20Xilinx%20FPGA%20Development/labs/README.md). They provide numerical and streaming contracts. This module extends them to host/device ownership; the following board experiments have not been executed by the course review.

## 1. Choose one supported platform stack

Record the exact board/part, platform shell or exported hardware, boot image, kernel, runtime, tool release, and accelerator binary. An Alveo PCIe card and a custom Zynq platform do not have interchangeable deployment commands. Driver families, memory allocation, programming permissions, and supported APIs vary.

XRT supplies native objects such as `xrt::device`, `xrt::bo`, `xrt::kernel`, and `xrt::run`. An `xclbin` carries the platform's accelerator configuration and metadata; its contents and compatibility depend on the build flow. Do not assume every XRT device uses the same `xocl`/`zocl` driver pair or supports every buffer flag. Follow [XRT native APIs](https://xilinx.github.io/XRT/master/html/xrt_native_apis.html) for the installed release.

Current XRT documentation includes [`xrt-smi` for Alveo](https://xilinx.github.io/XRT/master/html/xrt-smi-alveo.html). Older releases use `xbutil`; inspect installed `--help` and the matching documentation. An `examine` report inventories health/topology/status. It is not, by itself, a timeline of DMA and kernel execution. Collect profiling through the supported trace configuration and analysis tooling for that release.

**First artifact:** a compatibility manifest with actual versions, binary hashes, memory topology, kernel argument order, and a small known-answer test. A screenshot of device detection is insufficient.

## 2. Follow one buffer through XRT

For a simple vector-add kernel with two input pointers, one output pointer, and a scalar length, the conceptual order is:

1. Open the device and load the matching accelerator binary.
2. Obtain the kernel and use its argument connectivity to select valid memory groups for buffer allocation.
3. Allocate sufficiently large input/output buffer objects; map or write the inputs through the supported host API.
4. Synchronize inputs to the device where required by the allocation/platform contract.
5. Submit the run with arguments in the compiled kernel's exact order and types.
6. Wait using the installed API, inspect terminal status, and handle failure.
7. Synchronize output back where required, then compare every active element against the CPU reference.
8. Release buffers only after no run or callback can access them.

This is a lifecycle specification, not an executable example for an unidentified platform. On shared-memory targets a synchronization operation can have different mechanics from PCIe transfer. “Host mapped” does not mean “device coherent,” and the absence of an explicit copy in application code does not prove that no copy occurred.

**Worked size:** 1024 float32 elements occupy 4096 bytes per array; two inputs and one output represent 12,288 bytes of payload if each crosses the host/device boundary once. Report scalar argument units: passing `4096` when the kernel expects an element count of `1024` can cause out-of-bounds device access.

## 3. DMA address, cache state, and packet boundaries

Keep CPU virtual addresses, CPU physical addresses, and device DMA/IOVA addresses distinct. A kernel driver obtains a DMA address through the DMA API, checks mapping errors, and uses the correct direction and lifetime. Coherent allocation removes some cache-maintenance requirements; it does not remove ordering or completion requirements. Scatter-gather mappings can merge entries; use the returned mapped segment count for device descriptors and the documented original count for unmapping. Follow the [Linux DMA API HOWTO](https://docs.kernel.org/core-api/dma-api-howto.html).

A typical noncoherent streaming ownership sequence is:

```text
CPU owns input -> prepare bytes -> map/sync for device
DEVICE owns input/output -> submit -> execute -> confirmed completion
CPU owns output -> sync/unmap as required -> inspect/reuse
```

A `volatile` pointer does not perform the cache maintenance, ordering, or ownership transfer. A page being locked in RAM is not sufficient to make it a valid device mapping.

AXI DMA MM2S reads memory and produces a stream; S2MM consumes a stream and writes memory. A complete stream contract includes accepted beats, valid bytes, packet boundaries, and maximum receive capacity. For a 32-bit stream carrying ten bytes, a conventional packed transfer uses three accepted beats with `TKEEP=1111,1111,0011` and `TLAST` on the final beat. The exact packing/endian convention must match the design. Hold data and relevant sidebands stable while valid is asserted and ready is low.

A missing or misplaced `TLAST`, mismatched length, unsupported alignment, or incorrect DMA mode can cause truncation, an error, or a wait that never completes. Consult the selected IP's [AXI DMA product guide](https://www.amd.com/content/dam/xilinx/support/documents/ip_documentation/axi_dma/v7_1/pg021_axi_dma.pdf); do not infer all DMA engines' behavior from one example. The existing byte-stream lab intentionally lacks a complete DMA packet interface.

## 4. A timeout does not return buffer ownership

Use a state machine with explicit terminal states:

| State | Owner and permitted action |
|---|---|
| Prepared | CPU owns buffers; validate lengths, alignment, access, and request ID |
| Submitted/running | Device may access buffers; caller cannot free or overwrite them |
| Completed | Validate hardware status and generation, synchronize output, then return ownership |
| Timed out/aborting | Waiter stopped waiting; device access may still continue |
| Quiesced/failed | Supported abort/reset path establishes no future access; release or recover |

Suppose request 17 times out and request 18 reuses its slot. A late completion for 17 must not complete 18. A generation number helps distinguish stale notifications, but **does not stop stale DMA writes**. Quiesce the engine and drain callbacks before recycling device-owned storage. Reset behavior, outstanding bus transactions, and required recovery depend on the actual IP and driver.

RAII is useful for normal cleanup; asynchronous destructors still need a policy for outstanding work. A `std::future` is a result transport, not proof that the device has stopped. Keep request objects, mapped buffers, interrupts, and callbacks alive until the protocol allows destruction.

**Fault exercise:** inject invalid lengths before submit, device error status, delayed completion, duplicate completion, timeout followed by late completion, and reset with a request in flight. Save state transitions and demonstrate that every submitted request reaches exactly one application-visible terminal result without unsafe reuse.

## 5. Kernel interface design

For a custom platform device, match the device-tree binding, validate resource sizes, map registers with supported helpers, and use `readl`/`writel` with device-specific ordering. `probe()` establishes resources; removal/error unwinding must stop new requests, quiesce the device, synchronize callbacks, and release ownership in reverse order. Driver callbacks and signatures change across kernel versions; compile against the exact target tree.

A userspace ABI should use fixed-width fields, versioned structure sizes, validated bounds, explicit units, and documented completion semantics. Never trust a userspace pointer as a device address. An `ioctl` argument holding an arbitrary length must not become an unchecked allocation or descriptor count. Prefer existing subsystem APIs when the device fits them.

Use a threaded IRQ or workqueue when the work can sleep and the architecture permits it; do not add new tasklet-based designs merely because an old example did. `wait_for_completion_timeout()` returning zero does not cancel an IRQ producer. Review the [OS synchronization/lifetime lesson](../../Phase%201%20-%20Foundational%20Knowledge/3.%20Operating%20Systems/Lectures/Lecture-09.md).

UIO can expose certain register/interrupt interfaces for controlled prototypes. It does not automatically provide safe userspace DMA memory management or IOMMU isolation. The appropriate choice depends on trusted users, device capabilities, and the required lifecycle; consult [UIO documentation](https://docs.kernel.org/driver-api/uio-howto.html).

## 6. Inference integration is platform and release specific

| Route | What to study | Compatibility boundary |
|---|---|---|
| Legacy Vitis AI DPU flows | Quantizer/compiler, `xmodel`, matching VART/DPU | Keep the exact DPU architecture and release together |
| Current Vitis AI NPU flows | Supported NPU IP, compiled deployment artifacts, matching VART APIs | Do not assume legacy Zynq/Alveo examples describe every Versal NPU release |
| FINN | Generated streaming design, tensor packing, generated host driver | Keep the generated shape/datatype/bit-packing metadata with the bitstream |
| PYNQ | Overlay metadata, device-accessible buffers, synchronization and lifetime | Allocation/coherency/device address depend on the target backend |
| Compiler integration | Explicit external-codegen/runtime ABI for your chosen compiler release | An old VTA tutorial is not proof of current TVM backend support |

The [Vitis AI release documentation](https://vitisai.docs.amd.com/en/latest/) distinguishes releases; its [5.1 guide](https://vitisai.docs.amd.com/en/5.1/) describes the NPU transition. Choose a documented target/release combination instead of treating every version as a DPU-to-`xmodel` pipeline.

FINN generates deployment artifacts and drivers; study its [hardware build/deployment flow](https://finn.readthedocs.io/en/latest/hw_build.html) and the [FINN examples driver](https://github.com/Xilinx/finn-examples/blob/main/finn_examples/driver.py). Do not depend on an unexplained package named `finn-rtlib`. Tensor packing can change both byte count and bit order even when the logical tensor shape is unchanged.

PYNQ's [`allocate`](https://pynq.readthedocs.io/en/latest/pynq_libraries/allocate.html) returns a device-accessible buffer with a device address and synchronization facilities. Use `sync_to_device`/`sync_from_device` or the documented equivalents as appropriate, and keep the allocation alive until hardware is done. A plain NumPy array is not automatically a DMA allocation, and CMA is not a universal description of every PYNQ backend.

## 7. Measure end-to-end benefit

Use timestamps around validated phases: preparation, submission, transfer, execution, completion, output synchronization, and postprocessing. Device counters need a clock-domain/unit conversion before comparison with host timestamps. An accelerator execution time excludes work elsewhere unless the measurement explicitly includes it.

**Example assumptions:** CPU reference takes 300 µs. FPGA execution is 40 µs, but input preparation/transfer takes 120 µs, submission/completion costs 50 µs, and output handling takes 110 µs. Serial end-to-end time is 320 µs: `300/320 = 0.9375×`, so the application is slower despite a 7.5× kernel-only ratio. Overlap may improve steady-state throughput, but first-request latency and buffer ownership still need separate measurements.

Compare identical arithmetic, input shapes, batch size, warmup, and correctness tolerances. Report cold start, sustained throughput, latency distribution, queue age, failures, and power if actually measured. Do not invent an FPGA speedup from a CPU analytical model.

## Capstone and exit evidence

Deliver one supported platform with a numerical reference, compatibility manifest, register/argument map, buffer ownership diagram, complete request-state trace, and known-answer application. Include normal completion and one recovery path. Save actual tool/build/test commands and identify which checks were host-only, simulated, synthesized, implemented, or run on a board.

You are ready to continue when you can explain who owns every buffer after success, timeout, reset, and process exit; demonstrate that stale events cannot complete a new request; and account for transfers in an end-to-end result.

**Continue:** take the ABI and measured boundaries to [Track C's compiler stack](../../Phase%204%20-%20Track%20C%20-%20DL%20Inference%20Optimization/03%20-%20Compiler%20Stack/Guide.md). The compiler must preserve the same shape, datatype, memory, and completion contract.
