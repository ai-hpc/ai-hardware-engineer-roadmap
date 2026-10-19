# FSP (Firmware Support Package) and SPE firmware

<div class="course-identity fsp" markdown="1">
<div class="course-identity__icon">FSP</div>
<div markdown="1">
<p class="course-identity__eyebrow">Track B4 · Firmware Support Package</p>
<p class="course-identity__title">Understand low-level firmware, boot configuration, power behavior, and board-specific adaptation.</p>
<p class="course-identity__meta">Artifact: firmware adaptation note · Measure: boot stages, power rails, config deltas</p>
</div>
</div>


**Phase 4 — Track B — Nvidia Jetson** · Module 4 of 7

> **Focus:** Customize firmware that runs on the Jetson **Sensor Processing Engine (SPE)**—the **Cortex-R5** in the **always-on (AON)** domain—using NVIDIA’s **Firmware Support Package (FSP)** on **FreeRTOS**. This is the path for **low-level I/O**, **wake** scenarios, and **time-critical** tasks that should not live on the main Linux **CCPLEX**.
>
> **Scope:** The **step-by-step demo sections** below target **Jetson Orin Nano (T234)** using **r35.6** SPE guide paths (**p3767** MB1 BCT, **p3768** kernel DTS where NVIDIA names them). Other Jetson models use different files—see the same SPE guide for **AGX Xavier** / **AGX Orin** variants.

**Previous:** [3. L4T Customization](../3.%20L4T%20Customization/Guide.md) · **Next:** [5. Application Development](../5.%20Application%20Development/Guide.md) · **Companion:** [3. L4T customization](../3.%20L4T%20Customization/Guide.md) (flash layout, `Linux_for_Tegra`, BCT/pinmux)

---

## Start here: firmware owns a different execution context

**Learning route:** first run the [native C request/reply lab](labs/README.md) on a CPU (45–60 minutes), then read the ownership and timing sections below, then study one release-matched hardware demo. No SPE firmware build, flash, GPIO wiring or Jetson execution was performed in this review; the native lab has its own actual execution record.

**Release check, 2026-10-18:** the current Jetson Linux release is R39.2.1/JetPack 7.2.1. NVIDIA's documentation index links the **R36.4.3 SPE guide** alongside its R36 branch, and the **R35.6.0 SPE guide** alongside its R35 branch. The detailed recipes retained here are deliberately **R35.6.0 examples**. The absence of an R39 SPE guide in that index does not establish either support or impossibility: confirm the vendor-supported SPE package/BSP combination before porting these recipes. Do not flash an R35 `spe_t234.bin` into R39 merely because both use T234. [NVIDIA documentation index](https://docs.nvidia.com/jetson/index.html), [R36.4.3 SPE feature matrix](https://docs.nvidia.com/jetson/archives/r36.4.3/spe/index.html).

An always-on domain does not make every task or peripheral run in every power state. Verify the clock, reset, retention, wake and firmware behavior for the exact board. Moving work to a Cortex-R5 also does not by itself establish a deadline or a safety certification.

## An ownership contract before the demo

For each peripheral, name one register-level owner and list its pinmux, clocks, reset, interrupt route and firewall permissions. Linux may request a measurement through a protocol while SPE owns the controller; Linux and SPE must not independently program that controller at the same time.

A permission bit answers **who may access**, not **who is currently using** the device. Disabling a Linux DT node removes that Linux driver's claim but does not automatically release every firmware owner. A semaphore in one CPU's private RTOS does not synchronize an unrelated Linux driver.

Use a bounded path:

```text
sensor event -> short ISR -> bounded queue -> firmware task -> reply/telemetry
                                              |
Linux request with epoch/sequence -> validated command
```

Define queue capacity, overflow policy, message version/length/byte order, timeout, restart behavior and stale-message rejection. A host pointer is not a portable message field. For shared memory, the transport must also define buffer lifetime, ownership transfer, barriers/cache maintenance and address translation; the CPU lab deliberately does not implement those hardware mechanisms.

## Timing: capture, delivery and reaction

Suppose a signal changes at 1000 microseconds, hardware captures that edge at 1000, the ISR runs at 1030, and the control task reacts at 1080. Capture time, interrupt latency (30 microseconds) and reaction latency (80 microseconds) answer different questions. A precise hardware timestamp does not make the task react immediately.

At a 1 kHz tick rate, a tick is 1 ms; a requested 250 microsecond interval cannot be represented as one quarter of a tick by an integer delay API. Choose a suitable timer/capture mechanism or revise the requirement. Include quantization, interrupt masking, higher-priority work, queue delay and bus transactions in the budget.

ISRs must use the port's supported interrupt-context APIs and obey its interrupt-priority rules. Defer formatting, blocking bus operations and other long work to an appropriate task. A FreeRTOS **software timer callback** runs in the timer service task, not the peripheral ISR, and must not block that shared service. [FreeRTOS software timer semantics](https://www.freertos.org/Documentation/02-Kernel/02-Kernel-features/05-Software-timers/01-Software-timers).

A timeout says the caller stopped waiting. It does not cancel a peripheral transfer, release a DMA buffer, or prove the remote side did nothing. The [native lab](labs/README.md) exercises late replies and restart epochs so this distinction is testable before adding hardware.

## Why this sits next to L4T

SPE firmware is a **separate binary** (`spe_t194.bin` / `spe_t234.bin`) flashed via the same **Jetson Linux** toolkit as the rest of the board. Changing SPE behavior almost always touches **production concerns** you already practice in L4T work: **pinmux**, **GPIO init**, **firewall (SCR)**, **device tree** (including **BPMP** DT for clocks), and **partition-only** flashes. Treat SPE images as **versioned artifacts** alongside kernel and DTB.

---

## Official documentation (pin to your JetPack / L4T line)

The material below is aligned with the **Jetson SPE Developer Guide** packaged with the **r35.6** documentation set. For other releases, open the matching archive under [Jetson documentation](https://docs.nvidia.com/jetson/).

| Topic | Link |
|--------|------|
| **SPE guide (welcome, BSP layout, feature matrix)** | [Jetson SPE Developer Guide — r35.6](https://docs.nvidia.com/jetson/archives/r35.6.0/spe/index.html) |
| **FSP architecture (OSA / CPL / drivers)** | [FSP (Firmware Support Package)](https://docs.nvidia.com/jetson/archives/r35.6.0/spe/md__home_jenkins_workspace_Utilities_rt_aux_cpu_demo_fsp_docs_work_rt_aux_cpu_demo_fsp_doc_fsp.html) |
| **Build, artifacts, flash** | [Compiling and Flashing](https://docs.nvidia.com/jetson/archives/r35.6.0/spe/rt-compiling.html) |
| **AODMIC demo (DMIC5, wake)** | [AODMIC Application](https://docs.nvidia.com/jetson/archives/r35.6.0/spe/md__home_jenkins_workspace_Utilities_rt_aux_cpu_demo_fsp_docs_work_rt_aux_cpu_demo_fsp_doc_aodmic_app.html) |
| **Inter-processor channels** | [IVC](https://docs.nvidia.com/jetson/archives/r35.6.0/spe/md__home_jenkins_workspace_Utilities_rt_aux_cpu_demo_fsp_docs_work_rt_aux_cpu_demo_fsp_doc_ivc.html) |
| **GTE (timestamped GPIO events)** | [GTE Application](https://docs.nvidia.com/jetson/archives/r35.6.0/spe/md__home_jenkins_workspace_Utilities_rt_aux_cpu_demo_fsp_docs_work_rt_aux_cpu_demo_fsp_doc_gte_app.html) |
| **AON GPIO from SPE** | [GPIO Application](https://docs.nvidia.com/jetson/archives/r35.6.0/spe/md__home_jenkins_workspace_Utilities_rt_aux_cpu_demo_fsp_docs_work_rt_aux_cpu_demo_fsp_doc_gpio.html) |
| **AON I2C from SPE** | [I2C application](https://docs.nvidia.com/jetson/archives/r35.6.0/spe/md__home_jenkins_workspace_Utilities_rt_aux_cpu_demo_fsp_docs_work_rt_aux_cpu_demo_fsp_doc_i2c_app.html) |
| **AON SPI from SPE** | [SPI application](https://docs.nvidia.com/jetson/archives/r35.6.0/spe/md__home_jenkins_workspace_Utilities_rt_aux_cpu_demo_fsp_docs_work_rt_aux_cpu_demo_fsp_doc_spi_app.html) |
| **Timer driver demo** | [Timer application](https://docs.nvidia.com/jetson/archives/r35.6.0/spe/md__home_jenkins_workspace_Utilities_rt_aux_cpu_demo_fsp_docs_work_rt_aux_cpu_demo_fsp_doc_timer_app.html) |

---

## SPE / BSP layout (mental model)

From the SPE BSP package:

- **`fsp/source`** — Common **drivers**, **OSA** (OS abstraction), **CPL** (CPU abstraction), and **`soc/<soc>/...`** port/ID data.
- **`rt-aux-cpu-demo-fsp`** — Demo apps (`app/`), build system (`Makefile`, `soc/t19x` / `soc/t23x` **target_specific.mk**), platform code, and **FreeRTOS** integration.
- **`FreeRTOSV10.4.3/FreeRTOS/Source`** — FreeRTOS sources (version as shipped in that BSP drop).

NVIDIA’s welcome page lists **which demos are supported on which platforms** (e.g. **IVC**, **GTE**, **GPIO**, **I2C**, **SPI**, **Timer** on **Orin Nano** in that matrix; **AODMIC** is **not** listed for Orin Nano there). Always confirm against your **SoC** and **guide revision**.

**Orin Nano builds:** In `soc/t23x/target_specific.mk`, set **`ENABLE_SPE_FOR_ORIN_NANO := 1`** for demos that target the Nano module (as in the **GTE**, **GPIO**, and **SPI** recipes below). Use the same file for per-app flags such as **`ENABLE_I2C_APP`**, **`ENABLE_SPI_APP`**, **`ENABLE_TIMER_APP`**.

---

## FSP architecture (short)

### OSA (Operating System Abstraction)

OSA groups operating-system primitives behind an interface. It reduces the number of places that depend directly on FreeRTOS, but a port still needs semantic review: task versus ISR context, priority rules, blocking behavior, allocation, timeout units and scheduling behavior must agree. Replacing only wrapper names cannot establish equivalent timing or correctness.

Practical example: All the typical RTOS primitives (such as semaphores, mutexes, event groups, queues, and tasks) are wrapped by OSA APIs. The relevant headers in `fsp/source/include/osa/freertosv10/osa/` include:

| Functionality      | OSA API Header             |
|--------------------|---------------------------|
| Semaphore          | `osa-semaphore.h`         |
| Mutex              | `osa-mutex.h`             |
| Event group        | `osa-event-group.h`       |
| Queue              | `osa-queue.h`             |
| Tasks / scheduling | `osa-task.h`              |
| Software timer     | `osa-timer.h`             |

When reading a wrapper, follow it to the selected RTOS implementation and document whether it can block and whether an interrupt-context variant exists.

### CPL (CPU / platform abstraction): Why CPL?

CPL groups CPU/platform operations such as cache handling, barriers, interrupt control and register access. It localizes hardware dependencies, but it does not make a peripheral driver independent of register maps, DMA addressing, cache coherence or interrupt architecture. Moving from Cortex-R5 to Cortex-A may require changes above this layer as well. Verify the selected memory and ordering contract instead of assuming a function with the same name has identical effects.

**Examples of CPL-provided headers (paths as in NVIDIA’s FSP):**

| Concern                  | Header                                                      |
|--------------------------|-------------------------------------------------------------|
| Cache operations         | `fsp/source/include/cpu/arm/common/cpu/cache.h`             |
| Register access (Cortex-R5) | `fsp/source/include/cpu/arm/armv7/cortex-r5/reg-access/reg-access.h` |
| Barriers (memory/order)  | `fsp/source/include/cpu/arm/common/cpu/barriers.h`          |
| VIC (interrupts)         | `fsp/source/include/cpu/arm/common/cpu/arm-vic.h`           |
| Chip ID / SKU            | `fsp/source/include/chipid/chip-id.h`                       |

### Drivers

Peripheral drivers sit above OSA/CPL. NVIDIA points to headers such as **`fsp/source/include/gpio/tegra-gpio.h`** for GPIO. **SoC-specific** pieces use `fsp/source/soc/<soc>/port/aon` and `fsp/source/soc/<soc>/ids/aon` for **instance IDs**, **base addresses**, and **IRQ** data.

---

## Toolchain, build, and flash

### Toolchain

The archived R35.6 build procedure names **`gcc-arm-none-eabi-7-2018-q2-update`**. That is a pinned firmware compiler requirement, not a recommendation for all new Arm software. Record the exact compiler and BSP identity; a newer compiler is a porting change requiring validation. WSL2 compilation and USB recovery flashing have different requirements, so do not assume a Linux shell alone provides a supported flashing host.

### Environment and build

Documented archived build template; replace the paths and build only the SoC target you need. These firmware commands were not executed in this course review.

```bash
export SPE_FREERTOS_BSP="/path/to/matched-spe-bsp"
export CROSS_COMPILE="/path/to/pinned-toolchain/bin/arm-none-eabi-"

cd "${SPE_FREERTOS_BSP}/rt-aux-cpu-demo-fsp"
make -j"$(nproc)" bin_t19x    # T194-class
make -j"$(nproc)" bin_t23x    # T234-class
```

Other useful targets (see [Compiling and Flashing](https://docs.nvidia.com/jetson/archives/r35.6.0/spe/rt-compiling.html)):

- **`make docs`** — Doxygen output under `out/docs/index.html`
- **`make` / `make all`** — All SOC binaries + docs
- **`make clean`**, **`make clean_t19x`**, **`make clean_t23x`**, **`make clean_docs`**

**Artifact:** `out/<soc>/spe.bin`

### Install into `Linux_for_Tegra` and flash

1. **Back up** the stock files in `Linux_for_Tegra/bootloader/`:
   - **`spe_t194.bin`** (T194)
   - **`spe_t234.bin`** (T234)
2. Copy your build:
   - T194 → `Linux_for_Tegra/bootloader/spe_t194.bin`
   - T234 → `Linux_for_Tegra/bootloader/spe_t234.bin`
3. When the matching pinmux, firewall and device-tree changes are already deployed, the archived procedure supports a partition-only firmware iteration. Check the actual partition layout, slot/update policy, signing and board/storage arguments first. Updating an A-slot partition does not update every redundant copy. The following is an **R35.6 procedure template**, not a current universal Orin flash command:

```bash
# T194 (partition name spe-fw)
sudo ./flash.sh -k spe-fw <board-name> mmcblk0p1

# T234 (partition name A_spe-fw)
sudo ./flash.sh -k A_spe-fw <board-name> mmcblk0p1
```

Use the same **`<board-name>`** conventions as in the Jetson Linux Developer Guide for your carrier/module.

---

## Orin Nano — IVC echo channel (CCPLEX ↔ AON)

**IVC** uses a mailbox-backed **memory channel**. The stock SPE distribution documents an **echo** channel: Linux sends a string, SPE echoes it back.

NVIDIA’s r35.6 **IVC** page labels the detailed device-tree recipe under **AGX Orin**, while the release feature matrix lists IVC on Nano. Treat the shared T234 recipe as a reference to inspect in the matched Nano source, not proof that a channel number or DT path is identical in every BSP. Confirm the Linux and firmware channel descriptions agree.

1. Run **`source_sync.sh`** (per Jetson Linux documentation) so kernel DTS is available under `Linux_for_Tegra/sources/`.
2. Edit **`Linux_for_Tegra/sources/hardware/nvidia/soc/t23x/kernel-dts/tegra234-soc/tegra234-aon.dtsi`**. Either add an **`aon_echo`** node or, if it already exists disabled, set **`status = "okay"`** and match mailboxes to the guide (**`mboxes = <&aon 1>;`** for the AGX Orin example):

```dts
aon_echo {
    compatible = "nvidia,tegra186-aon-ivc-echo";
    mboxes = <&aon 1>;
    status = "okay";
};
```

3. **Compile** the device tree, install the **DTB** into **`Linux_for_Tegra/kernel/dtb/`** or **`Linux_for_Tegra/dtb/`** (whichever your Jetson Linux layout uses), and **reflash** using the **Building the kernel / device tree** flow from the Jetson Linux Developer Guide.
4. Build the **SPE** firmware that includes the **IVC echo** task, then copy **`out/t23x/spe.bin`** → **`${L4T}/bootloader/spe_t234.bin`** and flash (**`-k A_spe-fw`** when you are only updating SPE).

**Firmware sources (reference):**

- `rt-aux-cpu-demo-fsp/app/ivc-echo-task.c`
- `rt-aux-cpu-demo-fsp/platform/ivc-channel-ids.c`

**Test from Linux** (after the echo channel is enabled):

```bash
sudo su -c 'echo tegra > /sys/devices/platform/aon_echo/data_channel'
cat /sys/devices/platform/aon_echo/data_channel
# expect: tegra
```

Full detail: [IVC](https://docs.nvidia.com/jetson/archives/r35.6.0/spe/md__home_jenkins_workspace_Utilities_rt_aux_cpu_demo_fsp_docs_work_rt_aux_cpu_demo_fsp_doc_ivc.html).

---

## Orin Nano — GTE application (`app/gte-app.c`)

The **Generic Timestamping Engine (GTE)** captures selected signal events into a FIFO for software to drain. The r35.6 guide describes three signal slices, each **32 bits wide**. That width describes the monitored signal bitmap, **not a 32-bit timestamp counter**. Use the actual driver/TRM timestamp representation and conversion; do not infer counter width or wrap interval from the slice width.

The GPIO demo supplies an event source. Its signal-to-slice mapping comes from the matched pinmux information and `gte-tegra-hw.h`. A hardware capture timestamp is distinct from ISR entry or a printed log timestamp. Queueing, FIFO threshold and overflow determine when software sees the captured event. [NVIDIA GTE application](https://docs.nvidia.com/jetson/archives/r35.6.0/spe/md__home_jenkins_workspace_Utilities_rt_aux_cpu_demo_fsp_docs_work_rt_aux_cpu_demo_fsp_doc_gte_app.html).

### How GTE and GPIO Work Together in the Demo

- A specific GPIO line is connected to a GTE input, as defined in your carrier board’s pinmux (refer to the spreadsheet—often the “GTE” column is hidden by default).
- When the GPIO state changes (such as a rising or falling edge), the GTE hardware captures this as an event, tags it with a timestamp, and queues it in the slice’s FIFO.
- The GTE application (`gte-app.c`) running on the SPE reads these FIFO events, parses out the slice, event ID, and timestamp, and can print/log/react to this data.

### Software Setup for GTE Application (Orin Nano)

1. Enable the GTE demo application in the SPE firmware build system. In **`soc/t23x/target_specific.mk`**, set:
    - `ENABLE_GTE_APP := 1`
    - `ENABLE_SPE_FOR_ORIN_NANO := 1`
   Then rebuild the FSP firmware (`bin_t23x`), and copy the resulting `spe.bin` to your deployment location:
   - `${L4T}/bootloader/spe_t234.bin`

2. Disable the default Linux GTE driver in the device tree so that the Linux kernel does not claim exclusive access to the GTE hardware block (the SPE will control it instead). In the following file:
   - `Linux_for_Tegra/sources/hardware/nvidia/platform/t23x/p3768/kernel-dts/cvb/tegra234-p3768-0000-a0.dtsi`
   add or modify:
   ```dts
   gte@c1e0000 {
       status = "disabled";
   };
   ```

3. Rebuild the kernel device tree and copy the updated DTBs to `Linux_for_Tegra/kernel/dtb/`. Next, adjust the system’s hardware firewall to allow the SPE CPU to read and write GTE registers. For Orin Nano, patch this entry in:
   - `${L4T}/bootloader/tegra234-firewall-config-base.dtsi`
   `reg@1359` identifies entry **1359 in the predefined SCR/firewall table**. It is not the address `0x1359` and not an MMIO offset. The matched configuration associates this entry with `GTE_GPIO_SCR_TESCR_0`. Follow that table and the correct TRM for permission-bit meanings. [T234 security configuration](https://docs.nvidia.com/jetson/archives/r35.6.0/DeveloperGuide/SD/Bootloader/SecurityConfig.html).

   ```dts
   reg@1359 { /* GTE_GPIO_SCR_TESCR_0 */
       exclusion-info = <3>;
       value = <0x38001232>;
   };
   ```
   This change grants the access described by the matched permission fields. It does not implement exclusive ownership or stop another permitted processor from using the block. The archived change from `0x38000222` to `0x38001232` flips bits 12 and 4 (`old XOR new = 0x1010`); the arithmetic alone does not tell you what those bits mean.

4. Perform a **full device flash** (not just the SPE partition) so that the new device tree and firewall settings are applied to the device. This is required the **first** time you modify these aspects; for later SPE-only updates you can flash just the firmware.

### What to Expect: Logs and Usage

- When the GTE detects a qualifying input signal change and the FIFO reaches its configured threshold (`GTE_FIFO_OCCUPANCY`), the GTE issues an interrupt to the SPE.
- The SPE’s GTE interrupt service routine (ISR) runs, reads out queued entries, and decodes them.
- Output logs typically contain lines showing the application’s action (for GPIO pairing), recorded event details (“Slice Id”, “Event Id”, “Timestamp in nanosec”), and ISR execution logs such as `gpio_app_task - Setting GPIO_APP_OUT to 1` or `can_gpio_irq_handler` toggling the output GPIO.
- To test the setup, you can wire two GPIO pins as in the GPIO app: toggling an input should produce a captured event once the documented setup is correct. Measure capture and delivery separately; a delayed or dropped log line is not a direct measurement of the edge time.

**Further reading and examples:** NVIDIA’s [GTE Application](https://docs.nvidia.com/jetson/archives/r35.6.0/spe/md__home_jenkins_workspace_Utilities_rt_aux_cpu_demo_fsp_docs_work_rt_aux_cpu_demo_fsp_doc_gte_app.html) page provides deeper architectural details, event format, and a walk-through of how to pair signal, pinmux, GTE configuration, and SPE firmware application together for advanced timestamping use cases.


---

## Orin Nano — GPIO application (`app/gpio-app.c`)

Demo: drive one **AON GPIO** from SPE and receive an **interrupt** on another. **MB1** pinmux and **GPIO int map** must match the Orin Nano carrier (**p3767** BCT in the r35.6 guide).

### Hardware (Orin Nano dev kit)

For the documented **P3768 Orin Nano carrier**, the archived example uses **J12 pin 5 (PDD1 output)** and **J12 pin 3 (PDD2 input)**. These pins normally serve I2C functions; verify the exact board, voltage and configured directions, and remove conflicting attached devices before the documented loopback. Do not short two active outputs or generalize this connector name to another carrier.

### Software steps

1. **`soc/t23x/target_specific.mk`**: **`ENABLE_GPIO_APP := 1`**, **`ENABLE_SPE_FOR_ORIN_NANO := 1`**. Rebuild, copy **`spe.bin`** → **`${L4T}/bootloader/spe_t234.bin`**.
2. **GPIO interrupt map** — **`${L4T}/bootloader/t186ref/BCT/tegra234-mb1-bct-gpioint-p3767-0000.dts`**: under **`port@DD`**, route **DD2** to interrupt line **2**:

```dts
port@DD {
    pin-0-int-line = <4>; // GPIO DD0 to route 4
    pin-1-int-line = <4>; // GPIO DD1 to route 4
    pin-2-int-line = <2>; // GPIO DD2 to INT2
};
```

3. **AON GPIO ownership (MB1 GPIO DTSI)** — **`${L4T}/bootloader/tegra234-mb1-bct-gpio-p3767-dp-a03.dtsi`**: add **DD2** as input and **DD1** as output-low (example from NVIDIA):

```dts
gpio-input = <
    TEGRA234_AON_GPIO(EE, 2)
    TEGRA234_AON_GPIO(EE, 4)
    TEGRA234_AON_GPIO(DD, 2)
>;
gpio-output-low = <
    TEGRA234_AON_GPIO(DD, 1)
    TEGRA234_AON_GPIO(CC, 0)
>;
```

4. **Pinmux** — **`${L4T}/bootloader/t186ref/BCT/tegra234-mb1-bct-pinmux-p3767-dp-a03.dtsi`**: repurpose **gen8_i2c_scl_pdd1** / **gen8_i2c_sda_pdd2** from **I2C8** to **`rsvd1`** with the pull / tristate / input enables NVIDIA specifies (so the pins behave as GPIO for the demo). See the [GPIO Application](https://docs.nvidia.com/jetson/archives/r35.6.0/spe/md__home_jenkins_workspace_Utilities_rt_aux_cpu_demo_fsp_docs_work_rt_aux_cpu_demo_fsp_doc_gpio.html) page for the exact property values.
5. **Full flash** so **MB1 BCT** and **GPIO** settings take effect.

### Expected serial output

```text
gpio_app_task - Setting GPIO_APP_OUT to 1 - IRQ should trigger
can_gpio_irq_handler - gpio irq triggered - setting GPIO_APP_OUT to 0
```

Full detail: [GPIO Application](https://docs.nvidia.com/jetson/archives/r35.6.0/spe/md__home_jenkins_workspace_Utilities_rt_aux_cpu_demo_fsp_docs_work_rt_aux_cpu_demo_fsp_doc_gpio.html).

---

## Orin Nano — I2C application (`app/i2c-app.c`)

NVIDIA groups **Jetson AGX Orin** and **Orin Nano** under one **software** recipe: remove the Linux controller claim, configure the documented SPE permission, then run the matched firmware. Disabling the controller also removes its child devices from Linux; plan that consequence explicitly.

AON has multiple **I2C** instances (**2**, **8**, **10**); which ones are available depends on the platform. This demo uses **I2C bus 8** and reads a **BMI160** (6-axis) **WHO_AM_I**-style ID over the bus.

### Hardware

The upstream r35.6 page describes a **J30** 40-pin connection and a BMI160 at 7-bit address **0x68**, with its address-select input tied low. It groups AGX Orin and Nano software steps, but that does **not** make the physical connector name universal: the Nano reference carrier uses **J12** for its expansion header. Map SDA, SCL, supply, ground and address-select through the exact carrier schematic and sensor module documentation before wiring. Check bus voltage and pull-ups; a breakout board's supply label does not alone establish I/O-level compatibility.

The demo reads a known sensor ID. That proves a small transaction, not a validated IMU sampling rate, calibration, axis convention or timestamp contract. [Archived I2C demo](https://docs.nvidia.com/jetson/archives/r35.6.0/spe/md__home_jenkins_workspace_Utilities_rt_aux_cpu_demo_fsp_docs_work_rt_aux_cpu_demo_fsp_doc_i2c_app.html).

### Software steps

1. Run **`source_sync.sh`** so kernel DTS is under **`Linux_for_Tegra/sources/`**.
2. In **`Linux_for_Tegra/sources/hardware/nvidia/soc/t23x/kernel-dts/tegra234-soc/tegra234-soc-cvm.dtsi`**, disable the **I2C8** controller so the kernel does not claim it:

```dts
i2c@c250000 {
    status = "disabled";
};
```

3. **Compile** DTBs, copy into **`Linux_for_Tegra/kernel/dtb/`** or **`Linux_for_Tegra/dtb/`**, per your Jetson Linux layout.
4. In **`${L4T}/bootloader/tegra234-firewall-config-base.dtsi`**, allow SPE to access **I2C8** clocks/resets (**`CLK_RST_CONTROLLER_AON_SCR_I2C8_0`**):

```dts
reg@2130 { /* CLK_RST_CONTROLLER_AON_SCR_I2C8_0 */
    exclusion-info = <3>;
    value = <0x30001610>;
};
```

5. **`soc/t23x/target_specific.mk`**: **`ENABLE_I2C_APP := 1`** (and **`ENABLE_SPE_FOR_ORIN_NANO := 1`** if you build the Nano SPE image). Rebuild **`bin_t23x`**, copy **`spe.bin`** → **`${L4T}/bootloader/spe_t234.bin`**.
6. **Full flash** so **kernel DTB + firewall + SPE** stay consistent.

### Success output

```text
I2C test successful
```

Full detail: [I2C application](https://docs.nvidia.com/jetson/archives/r35.6.0/spe/md__home_jenkins_workspace_Utilities_rt_aux_cpu_demo_fsp_docs_work_rt_aux_cpu_demo_fsp_doc_i2c_app.html).

---

## Orin Nano — SPI application (`app/spi-app.c`)

**SPI2** lives in the **AON** domain. The demo is a **loopback**: **MISO** and **MOSI** must be shorted; firmware sends fixed bytes and checks the readback.

> **BCT conflict:** The SPI demo’s **MB1 GPIO** edits **remove** several **`TEGRA234_AON_GPIO(CC, …)`** entries from output lists in **`tegra234-mb1-bct-gpio-p3767-dp-a03.dtsi`**. If you merged **GPIO** or **GTE** recipes that rely on those lines, reconcile one coherent BCT before flashing.

### Hardware (Orin Nano dev kit)

The archived Nano source labels the following signals at **J2**. These are high-numbered connector contacts, **not pins 126–130 on a 40-pin expansion header**. Trace the actual carrier routing and accessible test points before attempting the documented MISO/MOSI loopback; disconnect any peripheral that could drive MISO. The table is a source mapping, not a universal wiring instruction:

| J2 pin | SPI2 signal |
|--------|-------------|
| 126 | CLK |
| 127 | MISO |
| 128 | MOSI |
| 130 | CS0 |

### Software steps

1. **Pinmux** — **`${L4T}/bootloader/t186ref/BCT/tegra234-mb1-bct-pinmux-p3767-dp-a03.dtsi`**: set **`spi2_sck_pcc0`**, **`spi2_miso_pcc1`**, **`spi2_mosi_pcc2`**, **`spi2_cs0_pcc3`** to **`nvidia,function = "spi2"`** with pull/tristate/input enables per [SPI application — Orin Nano](https://docs.nvidia.com/jetson/archives/r35.6.0/spe/md__home_jenkins_workspace_Utilities_rt_aux_cpu_demo_fsp_docs_work_rt_aux_cpu_demo_fsp_doc_spi_app.html) (e.g. **MISO** keeps **tristate/input enabled** for sampling).
2. **MB1 GPIO** — **`${L4T}/bootloader/tegra234-mb1-bct-gpio-p3767-dp-a03.dtsi`**: drop **CC0–CC3** from **`gpio-output-low`** / **`gpio-output-high`** as in NVIDIA’s diff so **SPI2** balls are not driven as GPIO.
3. **Firewall** — **`${L4T}/bootloader/tegra234-mb2-bct-scr-p3767-0000.dts`**: add **`reg@2135`** for **`CLK_RST_CONTROLLER_AON_SCR_SPI2_0`**:

```dts
reg@2135 { /* CLK_RST_CONTROLLER_AON_SCR_SPI2_0 */
    exclusion-info = <3>;
    value = <0x30001410>;
};
```

4. **`soc/t23x/target_specific.mk`**: **`ENABLE_SPI_APP := 1`**, **`ENABLE_SPE_FOR_ORIN_NANO := 1`**. Rebuild, install **`spe_t234.bin`**, **full flash**.

### Success output

```text
SPI test successful
```

Full detail: [SPI application](https://docs.nvidia.com/jetson/archives/r35.6.0/spe/md__home_jenkins_workspace_Utilities_rt_aux_cpu_demo_fsp_docs_work_rt_aux_cpu_demo_fsp_doc_spi_app.html).

---

## Orin Nano — Timer application (`app/timer-app.c`)

Uses **timer2** in **periodic** mode (NVIDIA’s example: **5 s** period, **`TIMER2_PTV`**). The demo prints on each IRQ and stops after **`STOP_TIMER`** counts.

1. **`soc/t23x/target_specific.mk`**: **`ENABLE_TIMER_APP := 1`** (set other **`ENABLE_*_APP`** flags to **0** unless you intentionally combine demos). For **Orin Nano** module images, keep **`ENABLE_SPE_FOR_ORIN_NANO := 1`** if your BSP requires it for **`bin_t23x`** on Nano.
2. Rebuild **`bin_t23x`**, copy **`spe.bin`** → **`spe_t234.bin`**, flash (**`-k A_spe-fw`** is enough if you did not change kernel or BCT).

### Expected serial output

```text
Timer2 irq triggered
```

(repeats each period until the stop count is reached.)

Full detail: [Timer application](https://docs.nvidia.com/jetson/archives/r35.6.0/spe/md__home_jenkins_workspace_Utilities_rt_aux_cpu_demo_fsp_docs_work_rt_aux_cpu_demo_fsp_doc_timer_app.html).

---

## Example: AODMIC application (AGX-oriented in r35.6 matrix)

The r35.6 **supported features** table in the SPE welcome page lists **AODMIC** for **AGX Xavier** and **AGX Orin**, not **Orin Nano**. The following is still useful if you work on those carriers or a **custom** T234 board that exposes **DMIC5** to AON.

The **AODMIC** demo shows **DMIC5** capture from SPE, **GPCDMA**, optional **system wake** from volume threshold, and the interaction with **suspend** and **BPMP** clocks. NVIDIA documents:

- **Sample rates** — `TEGRA_AODMIC_RATE_8KHZ`, `_16KHZ` (default in `app/aodmic-app.c`), `_44KHZ`, `_48KHZ`.
- **Channels** — `TEGRA_AODMIC_CHANNEL_STEREO` (default), `MONO_LEFT`, `MONO_RIGHT`.
- **`num_periods`** — typically **2** (double-buffered); max defined in `fsp/source/include/aodmic/tegra-aodmic.h` as `AODMIC_MAX_NUM_PERIODS` (extend if needed).
- **Oversampling** — AODMIC uses **64×** oversampling; effective PDM clock is **64 × sample_rate**—stay within your microphone’s allowed bit clock range.

**Hardware:** PDM microphone on the platform’s **AODMIC** pins (NVIDIA gives **40-pin** mappings for **AGX Xavier** and **AGX Orin**—DAT/CLK pins and power/GND).

**Software integration** (high level—exact edits change by release and carrier):

- **Pinmux** — Xavier: MB1 CFG edits (e.g. `tegra19x-mb1-pinmux-...cfg`). Orin: MB1 **pinmux DTSI** (e.g. `tegra234-mb1-bct-pinmux-...dtsi`) to route **CAN1**/GPIO balls to **dmic5**.
- **GPIO** — Remove conflicting **AON GPIO** claims in the MB1 GPIO DTSI where pins become DMIC.
- **Firewall** — Orin example: **SCR** override DTS allows SPE to touch **AODMIC clock** (`CLK_RST_CONTROLLER_AON_SCR_DMIC5_0`).
- **Enable the app** — In `soc/t19x/target_specific.mk` or `soc/t23x/target_specific.mk`, set **`ENABLE_AODMIC_APP := 1`**, rebuild, copy `spe.bin` to the correct **`spe_t194.bin` / `spe_t234.bin`**, then flash (full flash when MB1/SCR change; **SPE-only** once the matching configuration is stable and the selected partition/update policy permits it).

**Suspend caveat:** After resume, **BPMP** may gate **dmic5** clock and break capture. Mitigations in the guide: runtime `debugfs` clock force-on, or patch **BPMP DTB** `dmic5` **lateinit** clock tuple (third argument = **sample_rate × 64**), then flash **`bpmp-fw-dtb`** or full image.

**Wake path:** For **voice wake** demos, **wake83** must appear in BPMP UART logs in both **wake mask** and **Tier2** routing (NVIDIA gives example mask snippets).

---

## Practice checklist

- [ ] Verify the **exact supported SPE package/BSP combination**, target SKU and compiler. A shared major version or SoC name alone is insufficient.
- [ ] Store **original** `spe_t194.bin` / `spe_t234.bin` and every **custom** build in **Git** or artifact storage with **board + L4T** metadata.
- [ ] When changing **pinmux / SCR / BPMP DT**, plan a **full flash** once, then use **`-k spe-fw`** / **`-k A_spe-fw`** for firmware iteration.
- [ ] Read the **SoC TRM** for **SPE/AON** peripheral instances and **firewall** rules before relying on demos in production.

---

*Primary reference: [Jetson Sensor Processing Engine (SPE) Developer Guide — r35.6](https://docs.nvidia.com/jetson/archives/r35.6.0/spe/index.html) (NVIDIA). Align commands and file names with the archive that matches your Jetson Linux release.*


## Integration lab: Can Linux and the real-time firmware disagree about peripheral ownership?

Start with the [native C protocol lab](labs/README.md): execute the codec and state machine, detect two intentional bugs, and explain timeout versus cancellation. Then use the documented IVC echo example only on a verified matching SPE/BSP combination. Map pinmux, firewall, clocks and ownership before adding sensor access.

**Acceptance evidence:** Prove an echoed message and controlled missing-response behavior. Record the original firmware and recovery procedure; carry the message schema and timing contract into application development.

**Continue:** take this artifact to [5. Application Development](../5.%20Application%20Development/Guide.md). Explain which assumption from this course the next engineer must preserve.
