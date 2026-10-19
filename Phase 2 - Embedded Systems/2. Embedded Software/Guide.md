# ARM MCU, FreeRTOS, and Communication Protocols

<div class="course-identity embedded-software" markdown="1">
<div class="course-identity__icon">MCU</div>
<div markdown="1">
<p class="course-identity__eyebrow">Module 2 · Embedded Software</p>
<p class="course-identity__title">Build firmware around interrupts, DMA, buses, RTOS tasks, and peripheral control.</p>
<p class="course-identity__meta">Artifact: MCU/RTOS project · Measure: latency, jitter, stack, ISR cost</p>
</div>
</div>


*Follows [**Phase 2 section 1 — Schematic Capture and PCB Design**](../1.%20Schematic%20Capture%20and%20PCB%20Design/Guide.md) when you are bringing up **custom hardware**; dev-kit learners can start here in parallel. Builds on Phase 1 (digital, HDL, architecture, OS) — focuses on ARM Cortex-M microcontrollers, RTOS practice, and buses (SPI/UART/I2C/CAN) that connect sensors and peripherals.*

Focused mini-courses that sit inside this module:

- [**IoT**](IoT/Guide.md) for networking stacks like OpenThread and Zigbee
- [**Espressif**](Espressif/Guide.md) for `arduino-esp32`, ESP32-family workflow, the official Espressif education path, and the bridge toward ESP-IDF

**First, run the native C [framed telemetry lab](labs/Framed-Telemetry.md).** You will compile the actual parser, corrupt packets, split messages at every byte boundary, and test counter wrap with no MCU required. Then carry the same framing contract to UART and radio links.

**Before the board lab:** run [Keep an observation fresh](../Labs/Observation-Freshness.md). Predict what happens when a 100 Hz sensor feeds a worker that can process only 40 observations per second. Its tested queue and fault contracts transfer directly to the producer-consumer project below; the simulation does not measure ISR or RTOS timing.

---

## 1. ARM Cortex-M Architecture

### Cortex-M Core Variants

* **Cortex-M0/M0+, M3, M4, M7, M33/M55:** Understand differences across the family — instruction sets (Thumb, Thumb-2), optional floating-point, DSP/vector features, and the Armv8-M security extension. A core family name does not guarantee that an MCU implements every optional feature. Verify the specific part, compiler target, and enabled coprocessors.
* **CMSIS (Cortex Microcontroller Software Interface Standard):** Master CMSIS-Core for hardware abstraction, CMSIS-DSP for signal processing, CMSIS-NN for neural network inference on Cortex-M. These libraries are how production embedded and TinyML code targets ARM hardware.
* **Arm TrustZone for Cortex-M:** On implementations that include the security extension, partition secure and non-secure resources. TrustZone supplies isolation mechanisms; authenticated boot, key handling, and the secure application still need implementation and review.

### Memory System and MPU

* **Memory architecture:** Cortex-M implementations differ: do not assume every core has separate instruction/data buses, caches, an MPU, or DMA. Record the exact MCU core and SoC memory/bus topology from its reference manual before designing buffer ownership.
* **Memory-Mapped Peripherals:** All peripherals (GPIO, timers, UART, SPI, I2C) are accessed via memory addresses — registers are just addresses. Master this model to write low-level drivers.
* **MPU (Memory Protection Unit):** Configure memory regions with access permissions (read/write/execute) to isolate tasks, detect stack overflows, and catch errant pointer writes. Essential for robust production firmware.
* **Stack overflow detection:** On supported cores, configure MPU guards or stack-limit registers and record which exception they raise. A guard detects accesses into its protected region, not every possible stack corruption; retain stack high-water measurements and bounded call-depth analysis.

### CPU Internals

* **Instruction set:** Select the compiler target for the exact core. Cortex-M0/M0+ implement the smaller Armv6-M Thumb instruction set, including selected 32-bit instructions; they do not provide the full Armv7-M instruction set of M3/M4. Prefer CMSIS intrinsics for operations such as `__disable_irq()` and `__DSB()`.
* **Pipeline and interrupts:** Measure interrupt entry and service time on the selected MCU. Flash wait states, bus contention, interrupt masking, nesting, and optional floating-point context handling affect latency. `NVIC` priorities determine which pending exception can preempt another; a clock frequency alone cannot establish a response-time bound.
* **Exception model:** Learn which fault classes the selected core implements. Capture a bounded record of status registers and the exception frame into retained RAM when those addresses are valid. A corrupted stack can make unwinding unsafe, and flash erase/programming inside a fault handler may block or fault again. On the next boot, validate and export the record.

### RISC-V for Embedded (Comparison Track)

* **RISC-V ISA and Extensions:** Modular ISA — RV32I base + M (multiply), A (atomics), F/D (float), C (compressed). Learn how to evaluate which extensions to enable for your application.
* **Embedded Platforms:** SiFive FE310, ESP32-C3, GD32VF103 or FPGA-based soft cores (PicoRV32, VexRiscv on Zynq/Arty). Running FreeRTOS on a RISC-V soft core is a strong Phase 3 bridge project.

**Resources:**
* *"The Definitive Guide to ARM Cortex-M3 and Cortex-M4 Processors"* — Joseph Yiu. The reference text for Cortex-M architecture, NVIC, and memory system.
* *"Embedded Systems: Introduction to Arm Cortex-M Microcontrollers"* — Jonathan Valvano. Practical C programming against Cortex-M hardware.
* CMSIS Documentation (Arm/Keil) — CMSIS-Core, CMSIS-DSP, CMSIS-NN.
* *"RISC-V Reader: An Open Architecture Atlas"* — Patterson & Waterman.

**Projects:**
* **CMSIS-NN keyword spotting:** Deploy a quantized keyword spotting model using CMSIS-NN on a Cortex-M7 board. Benchmark inference cycles and RAM usage.
* **MPU stack guard:** Configure an MPU region as a guard page below the stack on a Cortex-M4. Trigger a controlled overflow and verify the MemManage fault handler fires instead of silent corruption.
* **FreeRTOS on RISC-V soft core:** Port FreeRTOS to a VexRiscv or PicoRV32 core synthesized on an FPGA. Validate task switching and interrupt latency.

---

Primary references for the architecture and allocation distinctions: [Arm Cortex-M0 instruction set](https://support.arm.com/documentation/ddi0432/c/programmers-model/instruction-set-summary?lang=en), [FreeRTOS memory management](https://www.freertos.org/Documentation/02-Kernel/02-Kernel-features/09-Memory-management/01-Memory-management), and [MCUboot design](https://docs.mcuboot.com/design.html).

## 2. C Programming for ARM Embedded Systems

### Memory Management

* **Allocation Strategies:** Static allocation (globals, BSS), stack allocation, and when (and when not) to use heap (`malloc`/`free`) in embedded. Understand fragmentation and why many embedded systems avoid dynamic allocation entirely.
* **Memory Sections:** `.text`, `.data`, `.bss`, `.rodata` — know where code and data live in flash vs. RAM and how the linker script controls this. Critical for Cortex-M startup code and `__attribute__((section(...)))`.
* **DMA (Direct Memory Access):** Hardware moves a configured sequence of transfers; the CPU still sets up ownership, handles completion/errors, and consumes results. On non-coherent cached MCUs, follow the vendor's buffer alignment and cache-maintenance rules. A barrier orders accesses but does not clean dirty cache lines. Avoid sharing a cache line between a DMA buffer and unrelated writable data.

### Bit Manipulation and Register Access

* **CMSIS-style register access:** Use vendor register definitions, `volatile` accesses, and documented masks. Avoid inventing C bitfield layouts for MMIO: allocation order and access width depend on the implementation. Read-modify-write is also incorrect for some write-one-to-clear or side-effect registers.
* **Bit masking patterns:** `SET_BIT()`, `CLEAR_BIT()`, `READ_BIT()` macros, bit-banding only where the selected core/SoC implements it. Master these to write ISRs and driver code that is both correct and fast.

### Interrupt Service Routines (ISRs)

* **NVIC priority configuration:** `NVIC_SetPriority()`, priority grouping, and preemption. Understand how to set priorities to ensure time-critical ISRs preempt lower-priority handlers.
* **Minimal ISRs:** Keep ISRs short — set a flag or post to a queue, do work in task context. This is the bridge from bare-metal to RTOS design.
* **Volatile and synchronization:** `volatile` expresses observable accesses such as device registers; it does not make read-modify-write atomic or synchronize tasks. For ISR/task sharing use the port’s supported atomics, bounded critical sections, or ISR-safe RTOS APIs. CPU barriers and DMA cache maintenance solve separate ordering and visibility problems.

**Resources:**
* *"C Programming for Embedded Systems"* — Kirk Zurell.
* STM32 HAL source code — read the HAL implementation to understand how vendor drivers map to CMSIS register access.
* [CMSIS-Core documentation](https://arm-software.github.io/CMSIS_6/latest/Core/index.html) plus the selected MCU reference manual and vendor cache/DMA application notes.

**Projects:**
* **Circular buffer + DMA UART:** Start with one producer and one consumer. Record the DMA write position, detect missed wraps/overrun, and release completed spans to a task. A span cannot be reused until processing finishes. First copy into a bounded queue; attempt zero-copy only after proving this ownership rule under delayed consumers.
* **Custom peripheral driver:** Write a bare-metal driver (no HAL) for a SPI sensor (e.g., IMU) from the datasheet, using CMSIS register access and interrupt-driven completion.

---

## 3. FreeRTOS

### Core RTOS Concepts

* **Tasks and Scheduling:** Create tasks with `xTaskCreate()`, assign priorities, and understand preemptive scheduling. Know when the scheduler runs (tick ISR, `taskYIELD()`, blocking calls).
* **Scheduling Algorithms:** Preemptive fixed-priority, time-slicing for equal-priority tasks, and co-operative mode. Understand how mutex priority inheritance reduces some priority inversion. It does not prevent deadlock or prove a latency bound; keep lock hold times bounded and avoid inconsistent lock ordering.
* **Task States:** Running, Ready, Blocked, Suspended. Trace task state transitions to diagnose scheduling issues.

### Inter-Task Communication

* **Queues:** `xQueueSend()` / `xQueueReceive()` — the primary IPC mechanism. Use from both task and ISR context (`xQueueSendFromISR()`). Understand blocking with timeouts.
* **Semaphores and Mutexes:** Binary semaphore for signaling (ISR → task), counting semaphore for resource counting, mutex for mutual exclusion with priority inheritance.
* **Event groups:** Use bits for conditions, not counted events: setting an already-set bit does not count another occurrence. Use the documented `FromISR` API in an ISR; event-group updates from an ISR are deferred through the timer service task and can fail if its command queue is full.
* **Stream and message buffers:** Their normal concurrency contract is one writer and one reader, including ISR participants. Multiple producers need explicit serialization or a different primitive. A stream buffer preserves bytes, not application frame boundaries; use the framing lab to recover complete packets.

### Memory Management

* **Heap schemes (heap_1 through heap_5):** Choose the right allocator — heap_1 (no free), heap_4 (first-fit with adjacent-free-block coalescing), heap_5 (non-contiguous regions). Monitor `xPortGetFreeHeapSize()` and `uxTaskGetStackHighWaterMark()`.
* **Static allocation:** Use `xTaskCreateStatic()` / `xQueueCreateStatic()` to place TCBs and stacks in statically-defined arrays — avoids dynamic allocation for those RTOS objects. Application libraries may still allocate; choose an allocation policy from the product requirements and verify the whole program.

### Worked timing and ownership example

A sensor produces one 32-byte record every 10 ms. A UART configured for 115200 baud, 8-N-1 sends ten wire bits per byte. With the framing lab's eight bytes of overhead, one record takes `40 × 10 / 115200 = 3.47 ms` on the wire, before scheduler and driver delays. The nominal bandwidth fits; that does not prove the consumer meets its deadline. A 50 ms logging pause produces five new records. Decide whether to buffer, backpressure, or discard stale telemetry, and count every discard.

Use this ownership chain: ISR acknowledges the peripheral and reports a completed buffer index; the receive task validates framing/CRC; a processing task consumes a copied record or an explicitly loaned buffer; only its owner returns that buffer to the free pool. A queue containing pointers copies the pointers, not the payload they reference.

For a single-core fixed-priority example, suppose sampling needs 1 ms every 10 ms, control needs 2 ms every 10 ms, and reporting needs 3 ms every 20 ms. Their utilization is `0.1 + 0.2 + 0.15 = 0.45`. This leaves nominal CPU time, but priority, release jitter, ISR interference, and blocking still matter. Do not infer guaranteed deadlines from utilization alone. First collect execution-time and blocking bounds; then apply an appropriate schedulability analysis.

**Two port traps:** Cortex-M interrupt priority numbers run in the opposite direction to FreeRTOS task priorities. An ISR calling kernel APIs must satisfy the port's `configMAX_SYSCALL_INTERRUPT_PRIORITY` rules and use `FromISR` operations. Also, base FreeRTOS task stack depth is in `StackType_t` elements, whereas ESP-IDF specifies task stack size in bytes. Read the selected port's API before copying a stack constant.

References: [FreeRTOS Cortex-M interrupt rules](https://www.freertos.org/Documentation/02-Kernel/03-Supported-devices/04-Demos/ARM-Cortex/RTOS-Cortex-M3-M4), [stream-buffer API contract](https://github.com/FreeRTOS/FreeRTOS-Kernel/blob/main/include/stream_buffer.h), and [ESP-IDF FreeRTOS differences](https://docs.espressif.com/projects/esp-idf/en/stable/esp32/api-reference/system/freertos_idf.html).

### Configuration and Debugging

* **FreeRTOSConfig.h:** Tune `configTICK_RATE_HZ`, `configMAX_PRIORITIES`, `configTOTAL_HEAP_SIZE`, `configUSE_TRACE_FACILITY`. Every option has a cost — understand the trade-offs.
* **Runtime stats:** Enable `configGENERATE_RUN_TIME_STATS` to measure per-task CPU time. Use `vTaskList()` / `vTaskGetRunTimeStats()` for a text-format system snapshot.
* **FreeRTOS+Trace (Tracealyzer):** Instrument your application with Percepio Tracealyzer or Segger SystemView to visualize task switching, queue operations, and interrupt latency in a timeline.
* **Stack overflow hooks:** Enable `configCHECK_FOR_STACK_OVERFLOW` (method 1 or 2) and implement `vApplicationStackOverflowHook()` as runtime diagnostics. These checks are not a proof that every stack corruption will be detected before damage.

**Resources:**
* *"Using the FreeRTOS Real Time Kernel: A Practical Guide"* — Richard Barry. The canonical reference.
* *"Mastering the FreeRTOS Real Time Kernel"* — Dr. Richard Barry (FreeRTOS official book, free PDF).
* FreeRTOS official documentation: freertos.org — kernel reference, porting guide, API docs.
* Percepio Tracealyzer / Segger SystemView — check the supported target, trace transport, and current license terms.

**Projects:**
* **Producer-consumer pipeline:** Two tasks communicating via queue — one reads a sensor over SPI, another processes and formats output over UART. Validate with Tracealyzer that no deadlock or starvation occurs.
* **Real-time data acquisition:** Acquire ADC samples at a fixed rate (timer ISR → queue), process in a FreeRTOS task, and log via UART. Verify timing jitter with a logic analyzer.
* **Multi-device concurrent control:** Control an LED PWM, a servo motor, and a display simultaneously from separate FreeRTOS tasks synchronized via semaphores and event groups.

---

## 4. SPI (Serial Peripheral Interface)

### Protocol Specifications

* **Timing and Modes (CPOL/CPHA):** Understand the 4 SPI modes in detail — clock polarity (CPOL) and phase (CPHA) — and trace timing diagrams for each. Know which mode a specific sensor uses before writing a driver.
* **Data framing:** Separate bit order (MSB-first/LSB-first within a word) from byte order (endianness of a multi-byte value). Match CPOL/CPHA, word width, byte layout, and chip-select timing to the device datasheet. A correct bit order does not imply a correct 16-bit sensor value.
* **SPI Device Addressing:** CS line per device vs. daisy-chaining. Understand glitch-free CS assertion and timing constraints from device datasheets.

### Driver Implementation

* **Bare-metal SPI driver:** Configure SPI peripheral registers (CR1/CR2, DR, SR on STM32), poll or use interrupts for TX/RX completion. Write a generic `spi_transfer(uint8_t *tx, uint8_t *rx, size_t len)` API.
* **DMA-driven SPI:** Arrange TX and RX together for full-duplex clocks, including dummy TX bytes when receiving. DMA completion can mean the last word entered a peripheral FIFO, not that its final bit left the pin. Wait for the documented peripheral end-of-transfer/busy condition before releasing CS or reusing a buffer.
* **Linux SPI driver (spidev):** Write a userspace driver using `/dev/spidev*` for development and prototyping. Understand when to move to a kernel driver for production.

### FPGA Implementation

* **SPI controller in Verilog:** Design a state-machine-based SPI master/slave in RTL. Include FIFO buffers, DMA interface, and mode selection registers. Useful for Phase 3 FPGA work.

**Projects:**
* **High-speed IMU data acquisition:** Interface with an IMU (e.g., ICM-42688-P) over SPI in DMA mode. Achieve maximum ODR and demonstrate interrupt-based data-ready handling.
* **SPI OLED display driver:** Write a complete driver for an SSD1306 OLED — init sequence, framebuffer DMA transfer, and partial update.

---

## 5. UART (Universal Asynchronous Receiver/Transmitter)

### Protocol Specifications

* **Baud Rate and Framing:** Relationship between baud rate, bit time, and clock error tolerance. Data framing: start bit, data bits (5–9), parity (even/odd/none), stop bits (1/1.5/2). Know how to calculate baud rate register values.
* **Flow Control:** Hardware RTS/CTS and software XON/XOFF — when to use each, how to configure, and common pitfalls (e.g., missing RTS pull-up).
* **RS-232 vs. TTL vs. RS-485:** Voltage levels, drivers, and when to use a line driver IC. RS-485 for differential multi-drop buses (industrial sensors, motor drives).

### Driver Implementation

* **Interrupt-driven UART with circular buffer:** TX and RX paths each backed by a circular buffer. ISR fills/drains the buffer; application reads/writes without blocking.
* **DMA UART receive:** Use DMA in circular mode for RX to capture incoming bytes without per-byte interrupts. Use half-complete and complete DMA callbacks plus idle-line detection to find newly received spans. An idle event is not a universal application-packet boundary; preserve protocol framing and detect overwritten spans.
* **UART with FreeRTOS:** Protect the TX buffer with a mutex, signal received frames via queue or stream buffer to a processing task. Demonstrate zero-copy receive with direct-to-task DMA.

**Projects:**
* **UART bootloader:** Implement a bootloader that receives a firmware binary over UART (e.g., XMODEM protocol), writes it to flash, and jumps to the new application after verification.
* **GPS NMEA parser:** Receive NMEA sentences from a GPS module via DMA UART, parse latitude/longitude/time, and post structured data to a FreeRTOS queue.
* **RS-485 Modbus RTU node:** Implement a Modbus RTU slave on RS-485 — half-duplex direction control, CRC16 checking, register map.

---

## 6. I2C (Inter-Integrated Circuit)

### Protocol Specifications

* **Bus mechanics:** START/STOP conditions, ACK/NACK, 7-bit and 10-bit addressing, clock stretching. Understand how clock stretching can cause timeout issues with strict masters.
* **Multiple controllers/targets:** Handle arbitration loss and bus errors. If SDA is stuck low while SCL can be released, the I2C specification describes up to nine recovery clocks. SCL stuck low or a target that does not release SDA may require a hardware reset or power cycle. Bound recovery and avoid interfering with another active controller.
* **Speed grades:** Standard (100 kHz), Fast (400 kHz), Fast-Plus (1 MHz), High-Speed (3.4 MHz). Understand pull-up resistor sizing vs. capacitance vs. speed.

### Driver Implementation

* **I2C driver:** Begin with a bounded polling transfer to simplify bring-up. Move to interrupts or DMA when latency/CPU measurements justify it. Every wait has a timeout; the completion result distinguishes success, address NACK, arbitration loss, and bus error.
* **I2C probing:** Start with the expected device address and a documented identity-register read. A blind scan is not harmless for every device: address-only transfers, reserved addresses, and register writes may have side effects. Confirm whether an API expects a 7-bit address or an already-shifted address byte.
* **Error handling and recovery:** Log the transfer phase and status. An address NACK may mean absent, unpowered, or temporarily busy; a data NACK is device/protocol specific, not necessarily overrun. The controller normally NACKs the last byte of a read to finish it.

**Projects:**
* **Multi-sensor I2C hub:** Interface with 3+ sensors (e.g., BME280 environment, MPU-6050 IMU, VL53L0X ToF distance) on the same I2C bus. Implement a FreeRTOS task per sensor with priority-based polling.
* **I2C EEPROM driver:** Write a byte/page write/read driver for an AT24C256 EEPROM. Handle page boundary wrapping and write cycle timing.

---

## 7. CAN (Controller Area Network)

### Protocol Specifications

* **Frame types:** Data frame, remote frame, error frame, overload frame. Understand the CAN frame fields: SOF, arbitration ID (11-bit standard / 29-bit extended), DLC, data (0–8 bytes), CRC, ACK.
* **Bus arbitration:** Non-destructive bitwise arbitration — lower ID wins. Understand dominant/recessive bit levels and fault confinement. A stuck physical line or transceiver can still disrupt the bus; protocol error handling does not eliminate every hardware fault.
* **Error handling:** CAN error counters (TEC/REC), error states (error-active, error-passive, bus-off), and automatic retransmission. Know when and how to recover from bus-off.
* **CAN FD:** Up to 64 data bytes and an optional faster data phase. The usable bit rates depend on controller/transceiver limits, cable length, topology, and timing configuration; 8 Mbit/s is not a universal protocol guarantee. CAN FD has no remote frames. Check the exact MCU part, since not every member of a product family has FDCAN.

### Higher-Layer Protocols

* **CANopen:** Standard application layer for embedded control — Object Dictionary, PDOs (Process Data Objects), SDOs (Service Data Objects), NMT state machine. Widely used in industrial robotics and motion control.
* **SAE J1939:** Heavy vehicle and automotive standard — PGN-based addressing, transport protocol for large payloads, DM1/DM2 diagnostic messages. Relevant for ADAS and autonomous vehicle work.
* **AUTOSAR COM / SOME/IP (context):** Understand how CAN fits in an AUTOSAR stack and how it bridges to Ethernet-based protocols in modern vehicles.

### Driver Implementation

* **CAN filter configuration:** Configure acceptance filters (mask/list mode) on the hardware CAN controller to receive only relevant message IDs and reduce CPU load.
* **TX/RX with FreeRTOS:** Use queues for TX mailbox management and RX FIFO dequeuing. Handle TX abort, RX overrun, and error interrupts properly.
* **ISO-TP (ISO 15765-2):** Multi-frame transport protocol over CAN for messages longer than 8 bytes (used by UDS/OBD-II diagnostics). Implement segmentation, flow control, and reassembly.

**Resources:**
* *"Controller Area Network Projects"* — Wilfried Voss. Practical CAN protocol guide.
* *"A Comprehensible Guide to J1939"* — Wilfried Voss.
* CANopen specifications and application notes from CAN in Automation; check the access terms for the required document.
* STM32 CAN/FDCAN reference manual + application notes (AN5348 FDCAN).

**Projects:**
* **CAN network with two nodes:** Two microcontrollers on a CAN bus exchanging sensor data (e.g., IMU + temperature). Implement proper termination, filter configuration, and error-frame detection.
* **OBD-II reader (J1979):** Send OBD-II PIDs over CAN to a vehicle (or simulator) and parse responses for engine RPM, vehicle speed, coolant temperature.
* **CANopen slave node:** Implement a minimal CANopen slave — heartbeat, NMT state machine, and at least one TPDO mapping sensor data. Use an open-source stack (e.g., CANopenNode).
* **ISO-TP layer:** Begin with the classic 12-bit length format (up to 4095 bytes) and bounded reassembly. Later ISO-TP formats support larger lengths, so negotiate/document the supported format rather than treating 4095 as a universal maximum. Validate sequence, flow-control limits, timeout, and buffer exhaustion against a simulator.

---

## 8. Power Management and OTA Updates

### Low-Power Design

* **Sleep modes:** Master Cortex-M sleep modes — sleep, deep sleep, stop, standby — and their wake-up latency / peripheral retention trade-offs.
* **Event-driven architecture:** Move from polling to interrupt-driven, event-driven designs. Sleep between events using `WFI`/`WFE` instructions and FreeRTOS tickless idle.
* **Power profiling:** Use Nordic PPK2, Otii Arc, or a µCurrent + oscilloscope to measure average current. Profile per-state current for IoT duty-cycle calculations.

### OTA Firmware Updates

* **Bootloader design:** Understand MCUboot — image slots (primary, secondary), signature verification (RSA/ECDSA), and the swap/overwrite update mechanism. Know how to chain from ROM bootloader → MCUboot → application.
* **A/B or slot-based updates:** Write the inactive slot, authenticate the image, request a test boot, and confirm only after health checks. Rollback and power-loss behavior depend on the bootloader’s configured strategy; two slots alone do not make the entire update atomic.
* **OTA transport:** BLE DFU (for example, nRF Connect SDK / Zephyr), MQTT/HTTP over Wi-Fi, cellular (LwM2M/CoAP). Implement delta updates (BSDiff) to reduce payload size.

**Resources:**
* [MCUboot documentation](https://docs.mcuboot.com/) — start with the integration for your supported RTOS and MCU.
* *"Embedded Software Primer"* — David Simon.
* Mender — an OTA platform whose supported client, device integration, licensing, and update-module requirements must be checked for the chosen target; do not assume a Linux client runs on a bare MCU.

**Projects:**
* **Secure bootloader:** MCUboot-compatible bootloader for Cortex-M4 with RSA-2048 signature verification and A/B image slots.
* **Low-power IoT sensor node:** Measure sleep, sample, transmit, and retry charge separately. Derive average current from the chosen reporting period and compare it to the battery budget; do not assume a universal microamp target across radios and sensors.
* **End-to-end OTA system:** First implement full-image signed updates, test-boot confirmation, and rollback. Inject interrupted downloads and failed health checks in a recoverable test setup. Add delta transfer only after the full-image recovery path works.

---

## 9. IoT Networking and OpenThread

### Why This Belongs in Embedded Software

Once a firmware engineer moves beyond a single board and a single peripheral, the next real problem is not just "read a sensor" but "keep a secure, low-power network alive for months." IoT protocols sit directly on top of your timers, UART/SPI links, radio drivers, storage, and power states, which is why they belong in the embedded-software track instead of being treated as a separate cloud topic.

OpenThread is a strong first protocol here because it combines **embedded constraints** with **real networking structure**. You have to understand 802.15.4 radios, low-power timing, packet formats, IPv6, commissioning, host-controller links, and border routers as one coherent system.

### OpenThread as the First IoT Protocol

Thread is an IPv6-based, low-power mesh network built on IEEE 802.15.4. OpenThread is the open-source implementation that makes this concrete on real hardware, including SoC-style MCU designs and Linux-host-plus-radio-co-processor designs.

This makes OpenThread especially relevant to the rest of the roadmap. The same concepts show up later when you build a Thread RCP on an ESP32-C6, attach it to a Jetson over UART, and let Linux run the higher-level OpenThread stack while the small radio chip handles 802.15.4.

Start here:

* [**IoT Networking and Device Connectivity**](IoT/Guide.md)
* [**OpenThread**](IoT/OpenThread/Guide.md)

---

## AI Hardware Connection

| Topic | Connection to AI Hardware Engineering |
|-------|--------------------------------------|
| ARM Cortex-M + CMSIS-NN | TinyML inference on MCUs — deploy quantized models (keyword spotting, anomaly detection) without an OS |
| FreeRTOS | Task scheduling and bounded communication around sensors and control; verify the actual firmware architecture before attributing an RTOS to another project |
| SPI / I2C / CAN | Sensor interfaces for cameras, IMUs, LiDAR, radar — the input pipeline for AI perception systems |
| CAN / J1939 | Vehicle bus protocol — how openpilot reads and writes actuator commands on real cars |
| Power management | Battery-powered edge AI devices: every µA counts when running inference at the edge |
| OTA updates | Production edge AI deployment — pushing new model weights and firmware to deployed devices |
| OpenThread / IoT networking | Connect low-power sensor nodes, gateways, and Linux hosts in real products instead of isolated lab setups |

---

## Projects Summary

| Project | Key Skills |
|---------|-----------|
| CMSIS-NN keyword spotting on Cortex-M7 | ARM architecture, CMSIS-NN, TinyML deployment |
| FreeRTOS producer-consumer sensor pipeline | Task design, queues, ISR-to-task hand-off |
| OpenThread RCP + Linux host | IoT networking, UART/SPI host links, border-router architecture |
| DMA circular buffer UART receiver | DMA, circular buffers, idle-line detection |
| SPI IMU at maximum ODR | SPI DMA, interrupt-driven data-ready |
| I2C multi-sensor hub | I2C bus management, multi-task polling |
| CAN two-node network with error handling | CAN framing, filters, error states |
| OBD-II CAN reader | J1979 protocol, CAN frame parsing |
| MCUboot secure bootloader | Secure boot, A/B slots, signature verification |
| Ultra-low-power IoT node | Deep sleep, event-driven, power profiling |

## Separate acquisition, policy inference, and control

Build a sensor producer with a bounded queue. Each sample carries a sequence ID, timestamp, units, and error flags. Let a task consume it at a slower rate, then inject missed reads and queue pressure. Choose and justify a drop policy; no sample should silently acquire a new timestamp during retry.

For a robot or drone, the MCU/control loop and higher-level VLA policy have different timing and failure responsibilities. Use a simulated actuator to test freshness checks, bounded commands, and watchdog behavior before hardware motion. Export the message definition, clock relationship, queue policy, and measured timing to Linux integration.

**Carry forward:** Sensor/control message schema and fault-injection trace. Continue with [the next course](../3.%20Embedded%20Linux/Guide.md) and reuse that evidence.
