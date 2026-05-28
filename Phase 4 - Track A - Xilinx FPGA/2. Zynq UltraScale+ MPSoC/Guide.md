# 2. Zynq UltraScale+ MPSoC

<div class="course-identity zynq-mpsoc" markdown="1">
<div class="course-identity__icon">ZYNQ</div>
<div markdown="1">
<p class="course-identity__eyebrow">Track A2 · Zynq UltraScale+ MPSoC</p>
<p class="course-identity__title">Split work between ARM software, programmable logic, AXI, DMA, interrupts, and Linux.</p>
<p class="course-identity__meta">Artifact: PS/PL accelerator demo · Measure: DMA throughput, latency, CPU overhead</p>
</div>
</div>


> Build systems that split work cleanly between ARM processing cores, programmable logic, memory, DMA, interrupts, and embedded Linux.

**Layer mapping:** L3-L6. This module connects processing system software, programmable logic, AXI interconnects, DMA, boot flow, device tree, Linux drivers, and hardware/software co-design.

**Role targets:** FPGA Systems Engineer · Embedded Linux Engineer · BSP Engineer · Hardware/Software Co-design Engineer · Edge Acceleration Engineer

**Prerequisites:** [Xilinx FPGA Development](../1.%20Xilinx%20FPGA%20Development/Guide.md), [Embedded Linux](../../Phase%202%20-%20Embedded%20Systems/3.%20Embedded%20Linux/Guide.md), and basic C/C++.

**What comes after:** [Advanced FPGA Design](../3.%20Advanced%20FPGA%20Design/Guide.md), [High-Level Synthesis](../4.%20High-Level%20Synthesis%20%28HLS%29/Guide.md), and [Runtime and Driver Development](../5.%20Runtime%20and%20Driver%20Development/Guide.md).

---

## Why This Module Exists

Zynq is not just an FPGA with a processor attached. It is a heterogeneous system where software and hardware share memory, interrupts, clocks, and failure modes.

The core design question is:

```text
What runs on the processing system, what runs in programmable logic, and how do they exchange data safely and fast enough?
```

This module teaches that boundary.

---

## Course Outcomes

By the end, you should be able to:

- explain the Zynq UltraScale+ processing system and programmable logic boundary
- design an AXI-connected PL peripheral
- move data through AXI DMA
- expose PL hardware to Linux through device tree and userspace or kernel drivers
- reason about cache coherency, interrupts, and memory mapping
- build and boot a custom Linux image for a Zynq board
- document a PS/PL performance bottleneck with measurements

---

## Unit Map

| Unit | Focus | Artifact |
|------|-------|----------|
| 1 | MPSoC architecture | PS/PL architecture map |
| 2 | AXI and memory maps | block design with address map |
| 3 | DMA and streaming | measured PS/PL transfer path |
| 4 | Boot and Linux | bootable image and device tree patch |
| 5 | Driver interface | userspace or kernel access path |
| 6 | Co-design optimization | bottleneck report and tuned design |

---

## Unit 1: MPSoC Architecture

### Learn

- Cortex-A application cores
- Cortex-R real-time cores
- programmable logic resources
- on-chip memory, DDR, and cache hierarchy
- AXI high-performance and coherent ports
- interrupt routing
- clock and reset domains
- boot chain at a high level

### Build It

Create a board-specific architecture note:

- processor cores
- memory regions
- PL resources
- available interfaces
- boot media
- debug ports
- target workloads

### Measure It

- DDR bandwidth baseline
- CPU memory-copy bandwidth
- PL clock targets
- board boot time

### Ship It

A Zynq system map that explains where compute, memory, and control live.

---

## Unit 2: AXI And Memory Maps

### Learn

- AXI4-Lite for control registers
- AXI4 memory-mapped interfaces for bulk access
- AXI4-Stream for datapaths
- address assignment
- register maps
- clock/reset crossing concerns
- AXI protocol debugging

### Build It

Create a block design with:

- PS block
- AXI interconnect
- custom AXI-Lite control peripheral
- optional AXI-Stream datapath block

Expose control registers to software.

### Measure It

- register read/write latency
- address map correctness
- utilization and timing
- error behavior for invalid access if observable

### Ship It

Block design diagram, address map, register definition, and software proof of register access.

---

## Unit 3: DMA And Streaming

### Learn

- AXI DMA
- scatter-gather versus simple mode
- cache coherency and buffer flushing
- physically contiguous memory
- streaming backpressure
- throughput versus latency

### Build It

Implement a PS -> DMA -> PL -> DMA -> PS loop.

Start with a pass-through stream block. Then replace it with a simple transform:

- add constant
- threshold
- fixed-point multiply
- checksum

### Measure It

- transfer latency
- sustained throughput
- CPU overhead
- effect of buffer size
- effect of cache maintenance

### Ship It

A DMA benchmark with raw results and a short explanation of the bottleneck.

---

## Unit 4: Boot And Embedded Linux

### Learn

- boot ROM, FSBL, PMU firmware, U-Boot, kernel, device tree, rootfs
- PetaLinux or Yocto-based image flow
- device tree nodes for PL peripherals
- kernel configuration
- systemd service setup
- boot logs and failure triage

### Build It

Build and boot a custom Linux image that includes:

- device tree entry for your PL peripheral
- userspace test program
- startup service or test script
- captured boot log

### Measure It

- build reproducibility
- boot time
- kernel log cleanliness
- peripheral probe success

### Ship It

Image build notes, boot log, device tree patch, and validation commands.

---

## Unit 5: Driver Interface

### Learn

- `/dev/mem` and UIO tradeoffs
- platform drivers
- character devices
- interrupt handling
- DMA buffers
- synchronization and error handling
- when userspace access is acceptable and when a kernel driver is required

### Build It

Expose your PL peripheral through one of:

- userspace memory-mapped access for a simple control block
- UIO driver
- minimal platform driver
- character device with interrupt notification

### Measure It

- control path latency
- interrupt latency
- CPU utilization
- failure behavior on bad input or missing hardware

### Ship It

Driver or userspace access path with tests, logs, and documented limitations.

---

## Unit 6: Hardware/Software Co-design Optimization

### Learn

- task partitioning
- CPU versus PL cost model
- memory movement as the first bottleneck
- batching and buffering
- fixed-point versus floating-point tradeoffs
- real-time constraints
- profiling PS and PL together

### Build It

Choose one workload:

- sensor preprocessing
- image filter
- audio DSP block
- packet parser
- matrix-vector operation

Implement a CPU baseline and a PL-accelerated path.

### Measure It

- latency
- throughput
- CPU utilization
- PL resource utilization
- power if available
- end-to-end speedup including transfer overhead

### Ship It

A co-design report that explains whether acceleration was worth it after data movement and software overhead.

---

## Capstone

Build a Zynq PS/PL accelerator demo:

- custom PL block
- AXI control path
- DMA data path
- Linux image or boot notes
- device tree patch
- driver or userspace interface
- benchmark script
- report with timing, utilization, and throughput

The capstone is complete when the hardware/software boundary is clear, measured, and reproducible.

---

## Exit Criteria

You are ready to move on when you can:

- explain PS/PL partitioning for a workload
- build and validate an AXI-connected peripheral
- move data through DMA and explain cache coherency issues
- boot Linux with a device tree entry for PL hardware
- expose a PL block to software through a defensible interface
- measure end-to-end acceleration rather than only kernel speed
