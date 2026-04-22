# ARM MCU, FreeRTOS, and Communication Protocols

*Follows [**Phase 2 section 1 — Schematic Capture and PCB Design**](../1.%20Schematic%20Capture%20and%20PCB%20Design/Guide.md) when you are bringing up **custom hardware**; dev-kit learners can start here in parallel. Builds on Phase 1 (digital, HDL, architecture, OS) — focuses on ARM Cortex-M microcontrollers, RTOS practice, and buses (SPI/UART/I2C/CAN) that connect sensors and peripherals.*

Focused mini-courses that sit inside this module:

- [**IoT**](IoT/Guide.md) for networking stacks like OpenThread and Zigbee
- [**Espressif**](Espressif/Guide.md) for `arduino-esp32`, ESP32-family workflow, and the bridge toward ESP-IDF

---

## 1. ARM Cortex-M Architecture

### Cortex-M Core Variants

* **Cortex-M0/M0+, M3, M4, M7, M33/M55:** Understand differences across the family — instruction sets (Thumb, Thumb-2), presence of FPU (M4/M7), DSP extensions (M4/M7), and security extensions (TrustZone-M on M23/M33/M55).
* **CMSIS (Cortex Microcontroller Software Interface Standard):** Master CMSIS-Core for hardware abstraction, CMSIS-DSP for signal processing, CMSIS-NN for neural network inference on Cortex-M. These libraries are how production embedded and TinyML code targets ARM hardware.
* **ARM TrustZone-M (Cortex-M33/M55):** Hardware-enforced security partitioning — secure and non-secure worlds, secure boot, trusted execution environments (TEE). Directly relevant to IoT security and edge AI deployment.

### Memory System and MPU

* **Harvard vs. Von Neumann:** Cortex-M uses a modified Harvard architecture with separate instruction and data buses — understand implications for cache behavior and DMA.
* **Memory-Mapped Peripherals:** All peripherals (GPIO, timers, UART, SPI, I2C) are accessed via memory addresses — registers are just addresses. Master this model to write low-level drivers.
* **MPU (Memory Protection Unit):** Configure memory regions with access permissions (read/write/execute) to isolate tasks, detect stack overflows, and catch errant pointer writes. Essential for robust production firmware.
* **Stack Overflow Detection:** Use MPU guard regions and hardware stack protection to trigger controlled HardFault on overflow rather than undefined behavior.

### CPU Internals

* **Instruction Set (Thumb-2):** Cortex-M executes Thumb-2 — a mix of 16- and 32-bit instructions. Understand how the compiler targets this ISA and where inline assembly is useful (e.g., `__disable_irq()`, `__DSB()`).
* **Pipeline and Interrupts:** 3-stage pipeline (M0/M3) or 6-stage with branch prediction (M7). Understand how interrupt latency is determined and how `NVIC` (Nested Vectored Interrupt Controller) manages priority and preemption.
* **Exception Model:** HardFault, MemManage, BusFault, UsageFault — implement fault handlers that capture register state and stack trace to flash for post-mortem debugging.

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

## 2. C Programming for ARM Embedded Systems

### Memory Management

* **Allocation Strategies:** Static allocation (globals, BSS), stack allocation, and when (and when not) to use heap (`malloc`/`free`) in embedded. Understand fragmentation and why many embedded systems avoid dynamic allocation entirely.
* **Memory Sections:** `.text`, `.data`, `.bss`, `.rodata` — know where code and data live in flash vs. RAM and how the linker script controls this. Critical for Cortex-M startup code and `__attribute__((section(...)))`.
* **DMA (Direct Memory Access):** Transfer data between peripherals and memory without CPU involvement. Understanding DMA descriptors, cache coherency issues (`__DSB()`, `SCB_CleanDCache()`), and interrupt-based completion is essential for high-throughput I/O.

### Bit Manipulation and Register Access

* **CMSIS-style register access:** Use `typedef struct` + `volatile` + bitfields or `uint32_t` masks for register access. Write portable drivers that work at the register level without HAL overhead.
* **Bit masking patterns:** `SET_BIT()`, `CLEAR_BIT()`, `READ_BIT()` macros, bit-banding on Cortex-M3/M4. Master these to write ISRs and driver code that is both correct and fast.

### Interrupt Service Routines (ISRs)

* **NVIC priority configuration:** `NVIC_SetPriority()`, priority grouping, and preemption. Understand how to set priorities to ensure time-critical ISRs preempt lower-priority handlers.
* **Minimal ISRs:** Keep ISRs short — set a flag or post to a queue, do work in task context. This is the bridge from bare-metal to RTOS design.
* **Volatile and memory barriers:** Use `volatile` correctly for shared variables between ISR and main context. Use `__DSB()` / `__ISB()` where needed.

**Resources:**
* *"C Programming for Embedded Systems"* — Kirk Zurell.
* STM32 HAL source code — read the HAL implementation to understand how vendor drivers map to CMSIS register access.
* ARM Application Notes: AN321 (CMSIS), AN298 (Cortex-M memory system).

**Projects:**
* **Circular buffer + DMA UART:** Implement a lockless circular buffer fed by DMA in UART receive mode. Handle overflow and demonstrate zero-copy receive in an ISR.
* **Custom peripheral driver:** Write a bare-metal driver (no HAL) for a SPI sensor (e.g., IMU) from the datasheet, using CMSIS register access and interrupt-driven completion.

---

## 3. FreeRTOS

### Core RTOS Concepts

* **Tasks and Scheduling:** Create tasks with `xTaskCreate()`, assign priorities, and understand preemptive scheduling. Know when the scheduler runs (tick ISR, `taskYIELD()`, blocking calls).
* **Scheduling Algorithms:** Preemptive fixed-priority, time-slicing for equal-priority tasks, and co-operative mode. Understand priority inversion and how to prevent it with priority inheritance mutexes.
* **Task States:** Running, Ready, Blocked, Suspended. Trace task state transitions to diagnose scheduling issues.

### Inter-Task Communication

* **Queues:** `xQueueSend()` / `xQueueReceive()` — the primary IPC mechanism. Use from both task and ISR context (`xQueueSendFromISR()`). Understand blocking with timeouts.
* **Semaphores and Mutexes:** Binary semaphore for signaling (ISR → task), counting semaphore for resource counting, mutex for mutual exclusion with priority inheritance.
* **Event Groups:** Synchronize multiple tasks or ISRs with bit-flags. `xEventGroupSetBits()` / `xEventGroupWaitBits()` — useful for state machines and multi-source synchronization.
* **Stream and Message Buffers:** Efficient byte-stream or framed-message passing between tasks and ISRs, lower overhead than queues for bulk data.

### Memory Management

* **Heap schemes (heap_1 through heap_5):** Choose the right allocator — heap_1 (no free), heap_4 (best-fit, most common), heap_5 (non-contiguous regions). Monitor `xPortGetFreeHeapSize()` and `uxTaskGetStackHighWaterMark()`.
* **Static allocation:** Use `xTaskCreateStatic()` / `xQueueCreateStatic()` to place TCBs and stacks in statically-defined arrays — eliminates heap use entirely, required in safety-critical designs.

### Configuration and Debugging

* **FreeRTOSConfig.h:** Tune `configTICK_RATE_HZ`, `configMAX_PRIORITIES`, `configTOTAL_HEAP_SIZE`, `configUSE_TRACE_FACILITY`. Every option has a cost — understand the trade-offs.
* **Runtime stats:** Enable `configGENERATE_RUN_TIME_STATS` to measure per-task CPU time. Use `vTaskList()` / `vTaskGetRunTimeStats()` for a text-format system snapshot.
* **FreeRTOS+Trace (Tracealyzer):** Instrument your application with Percepio Tracealyzer or Segger SystemView to visualize task switching, queue operations, and interrupt latency in a timeline.
* **Stack overflow hooks:** Enable `configCHECK_FOR_STACK_OVERFLOW` (method 1 or 2) and implement `vApplicationStackOverflowHook()` to catch overflows at runtime.

**Resources:**
* *"Using the FreeRTOS Real Time Kernel: A Practical Guide"* — Richard Barry. The canonical reference.
* *"Mastering the FreeRTOS Real Time Kernel"* — Dr. Richard Barry (FreeRTOS official book, free PDF).
* FreeRTOS official documentation: freertos.org — kernel reference, porting guide, API docs.
* Percepio Tracealyzer / Segger SystemView — free tiers available for tracing.

**Projects:**
* **Producer-consumer pipeline:** Two tasks communicating via queue — one reads a sensor over SPI, another processes and formats output over UART. Validate with Tracealyzer that no deadlock or starvation occurs.
* **Real-time data acquisition:** Acquire ADC samples at a fixed rate (timer ISR → queue), process in a FreeRTOS task, and log via UART. Verify timing jitter with a logic analyzer.
* **Multi-device concurrent control:** Control an LED PWM, a servo motor, and a display simultaneously from separate FreeRTOS tasks synchronized via semaphores and event groups.

---

## 4. SPI (Serial Peripheral Interface)

### Protocol Specifications

* **Timing and Modes (CPOL/CPHA):** Understand the 4 SPI modes in detail — clock polarity (CPOL) and phase (CPHA) — and trace timing diagrams for each. Know which mode a specific sensor uses before writing a driver.
* **Data Framing:** Byte-oriented, word-oriented, and variable-length transfers. Endianness (MSB-first vs. LSB-first), chip select (CS) management, and bus sharing in multi-device configurations.
* **SPI Device Addressing:** CS line per device vs. daisy-chaining. Understand glitch-free CS assertion and timing constraints from device datasheets.

### Driver Implementation

* **Bare-metal SPI driver:** Configure SPI peripheral registers (CR1/CR2, DR, SR on STM32), poll or use interrupts for TX/RX completion. Write a generic `spi_transfer(uint8_t *tx, uint8_t *rx, size_t len)` API.
* **DMA-driven SPI:** Use DMA for both TX and RX to achieve maximum throughput without blocking the CPU. Handle cache coherency and ensure DMA complete callback wakes the waiting task.
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
* **DMA UART receive:** Use DMA in circular mode for RX to capture incoming bytes without per-byte interrupts. Use half-complete and complete DMA callbacks plus idle-line detection to handle variable-length frames.
* **UART with FreeRTOS:** Protect the TX buffer with a mutex, signal received frames via queue or stream buffer to a processing task. Demonstrate zero-copy receive with direct-to-task DMA.

**Projects:**
* **UART bootloader:** Implement a bootloader that receives a firmware binary over UART (e.g., XMODEM protocol), writes it to flash, and jumps to the new application after verification.
* **GPS NMEA parser:** Receive NMEA sentences from a GPS module via DMA UART, parse latitude/longitude/time, and post structured data to a FreeRTOS queue.
* **RS-485 Modbus RTU node:** Implement a Modbus RTU slave on RS-485 — half-duplex direction control, CRC16 checking, register map.

---

## 6. I2C (Inter-Integrated Circuit)

### Protocol Specifications

* **Bus mechanics:** START/STOP conditions, ACK/NACK, 7-bit and 10-bit addressing, clock stretching. Understand how clock stretching can cause timeout issues with strict masters.
* **Multi-master and multi-slave:** Arbitration loss detection, bus error handling. Know how to avoid bus lockup (stuck SDA) and implement recovery (9-clock pulse procedure).
* **Speed grades:** Standard (100 kHz), Fast (400 kHz), Fast-Plus (1 MHz), High-Speed (3.4 MHz). Understand pull-up resistor sizing vs. capacitance vs. speed.

### Driver Implementation

* **Interrupt + DMA I2C driver:** Avoid polling — use interrupt-based or DMA-based transfers with a FreeRTOS semaphore to block the calling task until completion.
* **I2C device scanning and probing:** Implement a bus scan to enumerate device addresses. Use for board bring-up and diagnostics.
* **Error handling and recovery:** Implement bus timeout detection and software reset. Handle NACK on address (device not present) and NACK on data (overrun) gracefully.

**Projects:**
* **Multi-sensor I2C hub:** Interface with 3+ sensors (e.g., BME280 environment, MPU-6050 IMU, VL53L0X ToF distance) on the same I2C bus. Implement a FreeRTOS task per sensor with priority-based polling.
* **I2C EEPROM driver:** Write a byte/page write/read driver for an AT24C256 EEPROM. Handle page boundary wrapping and write cycle timing.

---

## 7. CAN (Controller Area Network)

### Protocol Specifications

* **Frame types:** Data frame, remote frame, error frame, overload frame. Understand the CAN frame fields: SOF, arbitration ID (11-bit standard / 29-bit extended), DLC, data (0–8 bytes), CRC, ACK.
* **Bus arbitration:** Non-destructive bitwise arbitration — lower ID wins. Understand dominant/recessive bit levels and why CAN is robust to node failures.
* **Error handling:** CAN error counters (TEC/REC), error states (error-active, error-passive, bus-off), and automatic retransmission. Know when and how to recover from bus-off.
* **CAN FD:** Extended payload (up to 64 bytes), higher bit rate in the data phase (up to 8 Mbps). Understand the flexible data-rate frame format and controller requirements (FDCAN peripheral on STM32G0/H7).

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
* CANopen specification (CiA 301) — free download from CAN in Automation.
* STM32 CAN/FDCAN reference manual + application notes (AN5348 FDCAN).

**Projects:**
* **CAN network with two nodes:** Two microcontrollers on a CAN bus exchanging sensor data (e.g., IMU + temperature). Implement proper termination, filter configuration, and error-frame detection.
* **OBD-II reader (J1979):** Send OBD-II PIDs over CAN to a vehicle (or simulator) and parse responses for engine RPM, vehicle speed, coolant temperature.
* **CANopen slave node:** Implement a minimal CANopen slave — heartbeat, NMT state machine, and at least one TPDO mapping sensor data. Use an open-source stack (e.g., CANopenNode).
* **ISO-TP layer:** Implement ISO 15765-2 segmented transfer to send/receive payloads up to 4095 bytes over CAN. Validate with a UDS diagnostic request (SID 0x22 ReadDataByIdentifier).

---

## 8. Power Management and OTA Updates

### Low-Power Design

* **Sleep modes:** Master Cortex-M sleep modes — sleep, deep sleep, stop, standby — and their wake-up latency / peripheral retention trade-offs.
* **Event-driven architecture:** Move from polling to interrupt-driven, event-driven designs. Sleep between events using `WFI`/`WFE` instructions and FreeRTOS tickless idle.
* **Power profiling:** Use Nordic PPK2, Otii Arc, or a µCurrent + oscilloscope to measure average current. Profile per-state current for IoT duty-cycle calculations.

### OTA Firmware Updates

* **Bootloader design:** Understand MCUboot — image slots (primary, secondary), signature verification (RSA/ECDSA), and the swap/overwrite update mechanism. Know how to chain from ROM bootloader → MCUboot → application.
* **Dual-bank A/B updates:** Atomic update — write new firmware to the inactive bank, verify integrity, mark for swap, reboot. Rollback on failed verification.
* **OTA transport:** BLE DFU (Nordic NRF5 SDK / Zephyr), MQTT/HTTP over Wi-Fi, cellular (LwM2M/CoAP). Implement delta updates (BSDiff) to reduce payload size.

**Resources:**
* MCUboot documentation (mcuboot.com) — supports Zephyr, nRF Connect SDK, Mbed OS.
* *"Embedded Software Primer"* — David Simon.
* Mender.io — open-source OTA framework for both Linux and MCU targets.

**Projects:**
* **Secure bootloader:** MCUboot-compatible bootloader for Cortex-M4 with RSA-2048 signature verification and A/B image slots.
* **Ultra-low-power IoT sensor node:** Temperature + accelerometer node with deep-sleep between measurements, targeting <10 µA average at 1-second reporting interval.
* **End-to-end OTA system:** BLE or Wi-Fi OTA with delta compression and rollback on hash verification failure.

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
| FreeRTOS | Real-time scheduling for sensor pipelines feeding AI inference tasks; used in openpilot's comma 3X panda microcontroller |
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
