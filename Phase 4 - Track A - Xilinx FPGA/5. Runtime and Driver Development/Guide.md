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

## Why runtime & driver matters for FPGA

You can design the fastest accelerator in RTL or HLS — but without a runtime and driver, no software can use it. This module teaches you to:
- Move data between host CPU and FPGA accelerator via DMA
- Manage device memory (DDR attached to PL, BRAM, on-chip buffers)
- Build a user-space API that applications call to submit inference jobs
- Write or extend Linux kernel drivers for your custom IP

---

## 1. Xilinx Runtime (XRT) Architecture

* **XRT overview:**
    * User-space library (`libxrt_core`) + kernel driver (`xocl` / `zocl`).
    * Supported platforms: Alveo (PCIe), Zynq/MPSoC (embedded), Versal.
    * `xclbin` container format: bitstream + metadata + kernel signatures.

* **Host programming model:**
    * `xrt::device`, `xrt::bo` (buffer object), `xrt::kernel`, `xrt::run`.
    * Synchronous and asynchronous execution models.
    * Buffer allocation: device memory, host-only, host-device shared.
    * Memory bank assignment and topology awareness.

* **OpenCL on FPGA (Vitis flow):**
    * How Vitis wraps HLS kernels as OpenCL kernels.
    * `cl::Buffer`, `cl::Kernel`, `cl::CommandQueue` — mapping to XRT underneath.
    * When to use XRT native API vs OpenCL API.

**Projects:**
* Write an XRT host application that loads an `xclbin`, allocates input/output buffers, runs a vector-add kernel, and verifies results.
* Profile the same kernel with `xbutil examine` and `vitis_analyzer`. Identify DMA transfer time vs compute time.

---

## 2. DMA and Memory Management

* **AXI DMA engine:**
    * Memory-mapped (MM2S) and stream (S2MM) channels.
    * Scatter-gather DMA: descriptor chains for non-contiguous transfers.
    * Simple DMA vs SG DMA: trade-offs (latency, flexibility, CPU overhead).

* **Buffer management patterns:**
    * Double-buffering / ping-pong: overlap compute and transfer.
    * Zero-copy: mapping device memory into user-space (`mmap` via driver).
    * CMA (Contiguous Memory Allocator) for large physically-contiguous buffers on Zynq.

* **Memory hierarchy on Zynq/Alveo:**
    * PS DDR ↔ PL DDR ↔ BRAM ↔ UltraRAM.
    * AXI SmartConnect / interconnect bandwidth and arbitration.
    * Cache coherency between PS (ARM) and PL: HPC ports, ACE, snoop control unit.

* **IOMMU/SMMU:**
    * Virtual addresses for DMA on Zynq UltraScale+ (ARM SMMU).
    * Why this matters: memory isolation, security, multi-tenant FPGA.

**Projects:**
* Implement a DMA transfer between PS and PL on Zynq using AXI DMA IP. Measure throughput for different buffer sizes (1KB–16MB).
* Implement double-buffering: while the accelerator processes buffer A, DMA fills buffer B. Measure throughput improvement.

---

## 3. Linux Kernel Driver Development for FPGA

* **Platform device driver (Zynq):**
    * Device tree binding: `compatible`, `reg`, `interrupts`.
    * `probe()` / `remove()` lifecycle.
    * Memory-mapped I/O: `ioremap()`, `readl()` / `writel()` for control registers.

* **Interrupt handling:**
    * Registering interrupt handlers (`request_irq`).
    * Top-half / bottom-half (tasklet, workqueue) for DMA completion.
    * Waiting for accelerator completion: `wait_for_completion()` / poll.

* **Character device interface:**
    * Exposing accelerator to user-space via `/dev/myaccel`.
    * `file_operations`: `open`, `release`, `ioctl`, `mmap`, `poll`.
    * `ioctl` command design: submit job, query status, configure parameters.

* **UIO (Userspace I/O) driver:**
    * Lightweight alternative: map registers + interrupt to user-space.
    * When UIO is sufficient vs when a full kernel driver is needed.
    * `generic-uio` device tree overlay for quick prototyping.

* **PCIe driver (Alveo):**
    * BAR (Base Address Register) mapping.
    * MSI-X interrupts.
    * XRT's `xocl` driver as reference implementation.

**Projects:**
* Write a minimal platform driver for a custom AXI-Lite IP on Zynq. Expose register read/write via `/dev` + `ioctl`.
* Add interrupt-driven DMA completion to your driver. Compare latency vs polling.
* Use UIO to control an HLS accelerator from user-space Python. Benchmark vs kernel driver path.

---

## 4. User-Space Runtime Library

* **Runtime API design:**
    * Model the API after familiar patterns: `create_context()`, `allocate_buffer()`, `submit()`, `sync()`.
    * Asynchronous execution: command queues, events, callbacks.
    * Error handling: hardware timeouts, DMA errors, ECC faults.

* **Building a C++ runtime:**
    * RAII wrappers for device handles, buffers, and kernel objects.
    * Thread-safe job submission with `std::future` / completion callbacks.
    * Memory pool: pre-allocate device buffers to avoid allocation overhead per inference.

* **Python bindings:**
    * `pybind11` wrapper over C++ runtime.
    * NumPy ↔ device buffer zero-copy transfer.
    * Integration point: PyTorch custom operator or ONNX Runtime execution provider.

* **Profiling and observability:**
    * Timestamps at DMA start, DMA end, compute start, compute end.
    * Hardware performance counters (if designed into RTL).
    * Exposing metrics via sysfs or runtime API.

**Projects:**
* Build a C++ runtime library for your HLS matmul accelerator: `init()`, `malloc_device()`, `submit_matmul()`, `sync()`, `free()`.
* Add Python bindings. Run a PyTorch model where the matmul is offloaded to FPGA via your runtime.
* Add profiling: measure per-layer latency breakdown (host overhead, DMA, compute).

---

## 5. Inference Runtime Integration

* **Vitis AI runtime:**
    * DPU (Deep Processing Unit) integration on Zynq/Alveo.
    * Model compilation: Vitis AI quantizer → compiler → `xmodel`.
    * `vart` (Vitis AI Runtime): C++/Python APIs for running compiled models.

* **FINN runtime:**
    * FINN-generated accelerators: stitched IP with AXI-Stream interfaces.
    * `finn-rtlib`: Python runtime for FINN accelerators.
    * Throughput vs latency trade-offs with FINN's streaming architecture.

* **PYNQ overlay model:**
    * Load bitstream + driver from Python.
    * Allocate buffers with `pynq.allocate()` (CMA-backed).
    * Quick prototyping path for ML accelerators.

* **TVM on FPGA:**
    * TVM's VTA (Versatile Tensor Accelerator) as reference FPGA target.
    * BYOC for offloading subgraphs to FPGA accelerator (connection to Track C).

**Projects:**
* Deploy a quantized CNN on Vitis AI DPU (Zynq or Alveo). Measure FPS and compare with CPU baseline.
* Run a FINN-generated binary CNN accelerator. Profile streaming throughput.
* Use PYNQ to prototype: load a custom HLS overlay, run inference from Jupyter, visualize latency.

---

## Relationship to Other Modules

| This module (A §5) | Connects to |
|---------------------|-------------|
| DMA and memory management | A §2 (Zynq PS/PL, AXI interconnect) |
| Kernel driver development | Phase 1 §3 (OS — drivers, interrupts, memory) |
| User-space runtime | A §4 (HLS — the accelerator you're driving) |
| Inference runtime (Vitis AI, FINN) | Phase 3 (Neural Networks — model to deploy) |
| TVM/BYOC on FPGA | Track C (ML Compiler — custom backend) |

---

## Build Summary

| Section | Hands-on deliverable |
|---------|---------------------|
| §1 XRT | XRT host app + profiling |
| §2 DMA | AXI DMA throughput benchmark, double-buffering |
| §3 Kernel driver | Platform driver with interrupt-driven DMA |
| §4 User-space runtime | C++ runtime + Python bindings for HLS accelerator |
| §5 Inference runtime | Vitis AI DPU deployment, FINN streaming, PYNQ prototype |
