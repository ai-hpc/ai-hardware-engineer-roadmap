# Custom Carrier Board Design and Bring-Up

<div class="course-identity carrier-board" markdown="1">
<div class="course-identity__icon">BRD</div>
<div markdown="1">
<p class="course-identity__eyebrow">Track B2 · Carrier Board Bring-Up</p>
<p class="course-identity__title">Design and validate the board around a Jetson module: power, signals, connectors, and bring-up risk.</p>
<p class="course-identity__meta">Artifact: carrier bring-up checklist · Measure: rails, interfaces, signal integrity, faults</p>
</div>
</div>


**Phase 4 — Track B — Nvidia Jetson** · Module 2 of 7

> **Focus:** Design a **custom carrier board** for the **Jetson Orin Nano 8GB** SoM starting from the NVIDIA **P3768** reference design, through schematic capture, PCB layout, thermal and power-tree design, and first-article board bring-up validation.
>
> **Primary hardware:** Jetson Orin Nano 8GB (P3767 module) on custom carrier

**Previous:** [1. Nvidia Jetson Platform](../1.%20Nvidia%20Jetson%20Platform/Guide.md) · **Next:** [3. L4T Customization](../3.%20L4T%20Customization/Guide.md) · **Companion:** [3. L4T Customization](../3.%20L4T%20Customization/Guide.md) (BSP side of custom carriers) · **Runtime companion:** [5.1 Peripheral Access](../5.%20Application%20Development/1.%20Peripheral%20Access/Guide.md)

**Scope boundary:** this module owns **connector choice**, **schematics**, **routing**, **power**, and **pinmux planning** for a custom carrier. For runtime Linux access to GPIO, SPI, I2C, UART, CAN, USB, or storage on a running Jetson, use [5.1 Peripheral Access](../5.%20Application%20Development/1.%20Peripheral%20Access/Guide.md).

---


## 1. Why a custom carrier board

The Orin Nano 8GB Developer Kit (P3768 carrier) is designed for evaluation, not for shipping products. A custom carrier lets you:

- **Remove** unused interfaces (DisplayPort, extra USB, SD slot) to reduce cost and board area
- **Add** product-specific I/O (CAN bus, industrial Ethernet, PoE PD, custom audio codec, additional CSI cameras)
- **Optimize** the form factor, mounting, thermal path, and connector placement for your enclosure
- **Control** the BOM, second-sourcing, and long-term supply

**When the dev kit is sufficient:** Prototyping, small internal deployments (<50 units), applications that match the dev kit's connector set exactly.

**When custom is required:** Volume products, harsh-environment deployments, regulatory certification (custom shielding/filtering), cost optimization, or any design where the dev kit form factor doesn't fit.

---

## 2. NVIDIA reference design (P3768) study

NVIDIA publishes the **Orin Nano Developer Kit carrier** design files for reference. Start here before drawing a single line in your EDA tool.

### Key documents

| Document | What to extract |
|----------|-----------------|
| **Jetson Orin Nano OEM Product Design Guide** | Module pinout, power sequencing requirements, keep-out zones, thermal interface spec |
| **P3768 carrier schematic** (Altium/OrCAD) | Reference circuits for every interface: USB, PCIe, CSI, HDMI, Ethernet, power |
| **P3768 carrier layout** (Altium/OrCAD) | Stackup, impedance targets, high-speed trace routing, via strategy |
| **Jetson Orin Nano Design Guide** | Electrical and mechanical specifications, absolute maximum ratings |
| **Pinmux spreadsheet** (`Jetson_Orin_NX_and_Orin_Nano_series_Pinmux_Config_Template`) | Pin function assignment, drive strength, pull configuration |

### Module-to-carrier connector

The Orin Nano SoM connects via a **260-pin Molex board-to-board connector** (0.5 mm pitch). Study the pinout carefully:

- Power pins (VIN, GND, module power rails)
- High-speed differential pairs (USB 3.2, PCIe Gen3/4, CSI-2, HDMI/DP)
- Low-speed I/O (UART, SPI, I2C, GPIO, CAN, I2S)
- Configuration and control pins (FORCE_RECOVERY, POWER_BTN, SYS_RESET)
- JTAG debug pins

### Reference schematic walkthrough

Walk through each functional block in the P3768 schematic:

1. **Power input and regulation** — barrel jack, USB-C PD, main buck regulator
2. **USB hub and ports** — USB 3.2 Gen 1 hub, Type-A connectors
3. **PCIe M.2 slot** — M.2 Key M for NVMe SSD
4. **CSI camera connectors** — 2x 22-pin MIPI CSI-2 FPC
5. **Display output** — HDMI and/or DisplayPort
6. **Ethernet** — Gigabit Ethernet PHY and RJ45 with magnetics
7. **GPIO header** — 40-pin Raspberry Pi-compatible header
8. **Debug** — micro-USB for UART console, recovery mode button
9. **Fan header** — 4-pin PWM fan connector with tachometer

---

## 3. Schematic capture — from reference to custom

### Strategy

1. **Import** the P3768 reference schematic into your EDA tool (KiCad, Altium, OrCAD)
2. **Delete** blocks you don't need (e.g., DisplayPort for a headless product)
3. **Modify** blocks that need changes (e.g., swap USB hub for a different part, add CAN transceiver)
4. **Add** new blocks for product-specific interfaces
5. **Verify** every modified pin against the Pinmux spreadsheet and OEM Design Guide

### Common modifications

| Product type | Remove | Add |
|-------------|--------|-----|
| **Headless edge AI box** | Display, SD card, GPIO header | PoE PD, industrial Ethernet, additional NVMe |
| **Robotics platform** | Display, some USB ports | CAN bus (×2), IMU (SPI), additional CSI cameras |
| **Smart camera** | Most I/O, NVMe slot | CSI FPC (×4), ISP trigger GPIO, PoE PD, audio codec |
| **Vehicle gateway** | Display, USB hub | CAN-FD (×3), automotive Ethernet, LTE/5G modem (USB or PCIe) |

### EDA tool workflow

```
Reference schematic (Altium project)
  │
  ├─ Export to KiCad (if preferred)
  │     └─ Use Altium-to-KiCad converter or redraw key blocks
  │
  ├─ Annotate and organize into hierarchical sheets:
  │     ├─ Power
  │     ├─ SoM connector + decoupling
  │     ├─ USB
  │     ├─ PCIe / NVMe
  │     ├─ CSI / camera
  │     ├─ Ethernet
  │     ├─ CAN / serial
  │     ├─ Audio (if applicable)
  │     └─ Debug / test
  │
  └─ Run ERC (Electrical Rules Check)
```

---

## 4. Connector and interface selection

Choose connectors based on your product's mechanical constraints, environmental requirements, and target cost.

### Interface decisions

| Interface | Dev kit | Typical custom options |
|-----------|---------|----------------------|
| **USB 3.2** | Type-A × 4 (via hub) | Type-C × 1 (no hub, lower cost), or internal header |
| **USB 2.0** | Via hub | Direct to module pins, micro-B for debug |
| **CSI-2** | 22-pin FPC × 2 | 15-pin RPi FPC, Fakra (automotive), board-to-board (module camera) |
| **PCIe** | M.2 Key M | M.2 Key E (WiFi), direct edge connector, mini-PCIe, custom baseboard |
| **Ethernet** | RJ45 with magnetics | RJ45 (standard), M12 (industrial), SFP (fiber), or PHY-only (magnetics on carrier) |
| **Display** | HDMI + DP | HDMI micro (space), eDP (embedded panel), LVDS (via bridge), or none |
| **CAN** | Not on dev kit | MCP2515 (SPI) or MCP2562 (transceiver for native CAN if pinmuxed) |
| **Debug UART** | Micro-USB FTDI | Pin header (0.1"), TagConnect (pogo, no-footprint), or test pad |
| **Power** | Barrel jack + USB-C | Barrel jack, terminal block (industrial), PoE PD (802.3at), automotive 12V/24V with protection |

### Environmental and mechanical considerations

- **Vibration:** Locking connectors (M12, Molex Micro-Fit) instead of standard RJ45/USB-A
- **IP rating:** Panel-mount connectors with O-ring seals
- **Temperature:** Industrial-grade connectors rated to -40 to +85 C
- **Mating cycles:** Choose connectors rated for your expected service life

---

## 5. Power tree design

### Module power requirements

The Orin Nano 8GB module requires:

| Rail | Voltage | Max current | Notes |
|------|---------|-------------|-------|
| **VIN_PWR_BAT** | 5V nominal (4.75–5.25V) | 3A (15W mode) | Main power input to module |
| | | 1.4A (7W mode) | |

### Power sequencing

The OEM Design Guide specifies a **strict power-on sequence**. Violating it can damage the module or prevent boot:

1. **VIN_PWR_BAT** must be stable before **POWER_BTN** is asserted
2. **SYS_RESET_N** must be held low during initial power ramp
3. **CARRIER_PWR_ON** output from module indicates carrier can enable its own rails

### Carrier power tree design

```
DC Input (12V/24V/PoE)
  │
  ├─ Input protection (TVS, reverse polarity, fuse)
  │
  ├─ Main buck regulator → 5V rail (for module VIN_PWR_BAT)
  │     └─ Current monitoring (INA3221 or INA226)
  │
  ├─ 3.3V rail (for carrier peripherals: Ethernet PHY, CAN, GPIO)
  │     └─ LDO or secondary buck from 5V
  │
  ├─ 1.8V rail (if needed for level shifters or specific I/O)
  │
  └─ Fan power (5V or 12V, switched by CARRIER_PWR_ON)
```

### Key design considerations

- **Inrush current:** Use soft-start on the main regulator; the module's bypass capacitors draw significant inrush
- **Load-switch sequencing:** Use load switches with enable pins tied to CARRIER_PWR_ON for carrier peripherals
- **Current monitoring:** Place current sense resistors on VIN_PWR_BAT for power profiling (essential during bring-up)
- **Power budget spreadsheet:** Sum all rail currents at worst-case (max GPU clock + all peripherals active) and size regulators with 20% margin

---

## 6. PCB layout and stackup

### Stackup

A **6-layer** stackup is the minimum recommended for a carrier with high-speed interfaces:

```
Layer 1  — Signal (top, components, high-speed traces)
Layer 2  — GND plane (continuous, no splits under high-speed traces)
Layer 3  — Signal (inner, low-speed routing)
Layer 4  — Power plane(s)
Layer 5  — GND plane
Layer 6  — Signal (bottom, components, routing)
```

An **8-layer** stackup provides better signal integrity for USB 3.2 Gen 2, PCIe Gen 4, and dense designs.

### Critical layout rules

| Rule | Target |
|------|--------|
| **Module keepout** | Respect NVIDIA-specified keepout zones under and around the SoM connector |
| **SoM mounting** | Follow standoff placement and screw torque from the Design Guide |
| **Decoupling** | Place 100 nF + 10 uF at VIN_PWR_BAT pins, as close as possible to connector |
| **Ground plane** | No splits or voids under the SoM connector or high-speed differential pairs |
| **Via stitching** | Stitch GND vias around the SoM footprint and along board edges |
| **Thermal vias** | Under the heatsink mounting area, array of thermal vias to inner GND planes |

### Impedance-controlled traces

| Interface | Impedance | Pair type |
|-----------|-----------|-----------|
| USB 3.2 Gen 1/2 | 90 ohm differential | Coupled differential pair |
| PCIe Gen 3/4 | 85 ohm differential | Coupled differential pair |
| CSI-2 | 100 ohm differential | Coupled differential pair |
| HDMI/DP | 100 ohm differential | Coupled differential pair |
| Ethernet (RGMII) | 50 ohm single-ended, 100 ohm differential | Mixed |

Use your PCB fab's impedance calculator or a tool like Saturn PCB Toolkit to compute trace widths and spacing for your stackup.

---

## 7. High-speed signal integrity

### USB 3.2 Gen 2 (10 Gbps)

- Route as tightly coupled differential pairs on the surface layer
- **Length match:** ±5 mil within a pair, ±100 mil between pairs (if multi-lane)
- **Max trace length:** keep under 150 mm (6 inches) from module pin to connector
- **Avoid vias** in the differential path; if unavoidable, use back-drilled or via-in-pad with fill
- **Series AC coupling caps:** 100 nF, 0402, placed close to the connector end

### PCIe Gen 3/4

- Same differential pair rules as USB but tighter length matching within a lane
- **Reference clock routing:** 100 MHz REFCLK must be length-matched to data pairs
- **TX and RX on same layer** if possible; minimize layer transitions

### CSI-2 (MIPI)

- Short trace lengths (CSI is typically on-board or via short FPC)
- **Clock and data lane skew:** ±0.5 mm within a CSI port
- Terminate per MIPI D-PHY specification if trace length exceeds 100 mm

### Common signal integrity failures

| Symptom | Likely cause | Fix |
|---------|-------------|-----|
| USB 3.0 drops to 2.0 | Impedance mismatch, via stubs | Check stackup impedance, back-drill or shorten stubs |
| PCIe link trains at Gen 1 | Excessive loss at Gen 3/4 speeds | Shorten traces, improve impedance control, check connector |
| CSI camera no image | Clock/data skew | Re-route with tighter length matching |
| Ethernet CRC errors | Missing magnetics termination, layout coupling | Check PHY reference design, isolate from noisy traces |

---

## 8. Thermal design for custom enclosures

### Thermal budget

| Power mode | Module TDP | Typical GPU-heavy workload |
|------------|-----------|---------------------------|
| 7W (MAXN disabled) | 7W | ~6W sustained |
| 15W (MAXN) | 15W | ~12–14W sustained |

### Thermal path

```
Junction (die)
  │  Rjc ≈ 1.5 °C/W (module internal)
  │
Module top surface (thermal interface)
  │  Rtim ≈ 0.5–2 °C/W (thermal pad/paste)
  │
Heatsink
  │  Rhs ≈ 2–8 °C/W (depends on design)
  │
Ambient air
```

**Target:** Junction temperature < 95 C with ambient at 40 C (industrial) or 25 C (consumer).

### Heatsink options

| Solution | Rhs (approx) | Application |
|----------|-------------|-------------|
| **Passive finned heatsink** (40×40×20 mm) | 5–8 C/W | Consumer, quiet, 7W mode |
| **Passive + enclosure as heatsink** | 3–5 C/W | Industrial sealed box |
| **Active fan + small heatsink** | 1.5–3 C/W | 15W mode, continuous AI workload |
| **Heat pipe + fin stack** | 1–2 C/W | High-performance, compact |

### Fan control

The module provides a **PWM fan output** pin. Connect to a 4-pin fan header:

| Pin | Function |
|-----|----------|
| 1 | GND |
| 2 | +5V or +12V |
| 3 | Tachometer (sense) |
| 4 | PWM control |

Configure fan curves via `jetson_clocks` or custom `pwm-fan` device tree entries.

### Thermal validation

- Run `tegrastats` while executing a sustained AI workload
- Monitor **TJ (junction temperature)**, **GPU %**, and **throttling events**
- If throttling occurs, the thermal solution is insufficient — increase heatsink size or add forced airflow

---

## 9. Design-for-test and debug headers

Good test access saves weeks during bring-up and simplifies factory testing.

### Essential test points

| Test point | Purpose |
|------------|---------|
| **VIN_PWR_BAT** | Verify module input voltage and ripple |
| **5V, 3.3V, 1.8V rails** | Verify all carrier power rails |
| **CARRIER_PWR_ON** | Confirm module has signaled carrier power-on |
| **SYS_RESET_N** | Monitor or manually assert reset |
| **POWER_BTN** | Manual power button for debug |

### Debug headers

| Header | Purpose |
|--------|---------|
| **UART debug** (3-pin: TX, RX, GND) | Console output from bootloader and kernel; use a TagConnect or 0.1" header |
| **JTAG** (10-pin ARM Cortex or 20-pin) | Low-level debug if boot fails before UART is available |
| **I2C scan header** | Expose I2C buses for probing during bring-up |
| **SPI test header** | Expose SPI buses for loopback testing |

### LED indicators

| LED | Connected to | Purpose |
|-----|-------------|---------|
| Power LED | After main regulator | Board has input power |
| Module power LED | CARRIER_PWR_ON | Module is powered and running |
| Boot progress LED | GPIO (toggled by bootloader or kernel) | Visual boot progress indicator |
| Network LED | Ethernet PHY | Link and activity |

---

## 10. BOM management and component selection

### Critical component decisions

| Component | Key criteria | Second-source strategy |
|-----------|-------------|----------------------|
| **SoM connector** (260-pin) | Must match NVIDIA spec exactly | Single source (Molex) — no substitution |
| **Main buck regulator** | 5V, 3A+, efficiency >90%, industrial temp | Select from TI/MPS/Analog — pick parts with pin-compatible alternates |
| **Ethernet PHY** | Gigabit, RGMII, industrial temp if needed | Realtek RTL8211F or Microchip LAN8720 (different footprint) |
| **USB hub** (if used) | USB 3.2 Gen 1, industrial temp | Microchip USB5744 or similar |
| **CAN transceiver** | CAN 2.0B or CAN-FD, 3.3V, ESD protected | TI TCAN1042, NXP TJA1042, Microchip MCP2562 — mostly pin-compatible |
| **Passive components** | Use standard values (0402/0603), ±1% resistors | Always specify two manufacturer options in BOM |

### BOM best practices

- **Avoid single-source components** except where forced (SoM connector, specific ICs)
- **Specify manufacturer part number + distributor part number** for every line
- **Include DNP (Do Not Place) options** for debug components stripped in production
- **Track lead times** — the SoM itself has 12–16 week lead times through NVIDIA distribution
- **Use an AVL (Approved Vendor List)** and lock it per hardware revision

---

## 11. Board bring-up procedure

### Pre-power checklist

Before inserting the SoM module:

1. **Visual inspection** — check for solder bridges, missing components, correct orientation of polarized parts
2. **Continuity check** — verify no shorts between VIN and GND, 3.3V and GND, etc.
3. **Power rail test** — apply input power, measure all carrier rails with a multimeter (module not inserted):
   - 5V rail within spec (4.75–5.25V)
   - 3.3V rail within spec
   - 1.8V rail (if present)
4. **Ripple check** — use oscilloscope to verify ripple on 5V rail is <50 mV

### First boot

1. **Insert module** — verify alignment and secure with screws at correct torque
2. **Connect UART debug cable** — open terminal at 115200 baud
3. **Apply power** — observe UART output
4. **Expected sequence:**
   ```
   [MB1] DRAM training...
   [MB2] Loading UEFI...
   [UEFI] Booting Linux...
   [kernel] ...
   ```
5. If no UART output — check POWER_BTN, SYS_RESET, VIN voltage under load

### First flash

```bash
# On host (Ubuntu 22.04)
cd Linux_for_Tegra
sudo ./flash.sh <board_config> mmcblk0p1
# or for NVMe:
sudo ./tools/kernel_flash/l4t_initrd_flash.sh <board_config> external
```

Use the board config from [Module 3 — L4T Customization](../3.%20L4T%20Customization/Guide.md) adapted for your custom carrier.

---

## 12. Pinmux configuration and validation

### Using the NVIDIA Pinmux spreadsheet

1. Open `Jetson_Orin_NX_and_Orin_Nano_series_Pinmux_Config_Template_v1.2.xlsm`
2. For each pin your carrier uses, select the correct function (GPIO, SPI, I2C, UART, etc.)
3. Set drive strength, pull-up/pull-down, and input/output direction
4. **Export** the generated `.dtsi` fragments

### Generating pinmux DT fragments

The spreadsheet generates device tree source include (`.dtsi`) files:

- `tegra234-mb1-bct-pinmux-<board>.dtsi` — pinmux for MB1 boot stage
- `tegra234-mb1-bct-gpio-<board>.dtsi` — GPIO configuration for MB1
- `tegra234-mb1-bct-padvoltage-<board>.dtsi` — pad voltage levels

Place these in `Linux_for_Tegra/bootloader/generic/BCT/` and reference them in your board `.conf` file. See [T23x BCT reference](../3.%20L4T%20Customization/T23x-Deployment.md) for details.

### Validation

```bash
# On target — verify the board came up with the expected pin and bus behavior
sudo gpioinfo gpiochip0
sudo gpioget gpiochip0 43

# Verify I2C bus detection
sudo i2cdetect -y -r 1

# Verify SPI bus
sudo spidev_test -D /dev/spidev0.0 -v

# Verify UART
sudo cat /dev/ttyTHS0  # (check for expected data)
```

Use real line offsets and bus numbers from your target. For the runtime meaning of `gpiochip`, line offsets, and safe `libgpiod` usage, see [5.1 Peripheral Access](../5.%20Application%20Development/1.%20Peripheral%20Access/Guide.md).

---

## 13. Peripheral validation checklist

After a successful first boot with your custom pinmux, systematically validate every interface:

| Interface | Test | Expected result | Command / tool |
|-----------|------|-----------------|----------------|
| **USB 3.2** | Plug USB device | Enumeration at super-speed | `lsusb -t` |
| **USB 2.0** | Plug USB device | Enumeration at high-speed | `lsusb -t` |
| **Ethernet** | Connect cable, run iperf | Link up, ~940 Mbps | `ethtool eth0`, `iperf3 -c <host>` |
| **PCIe / NVMe** | Insert NVMe SSD | Device detected | `lspci`, `nvme list` |
| **CSI camera** | Connect camera module | Frames captured | `v4l2-ctl --stream-mmap` |
| **I2C** | Scan bus | Expected device addresses respond | `i2cdetect -y -r <bus>` |
| **SPI** | Loopback (MOSI→MISO) | Data matches | `spidev_test -D /dev/spidevX.Y` |
| **CAN** | Send/receive frames | Frames on bus | `cansend can0 123#DEADBEEF`, `candump can0` |
| **UART** | Loopback or paired device | Characters echo | `minicom` or `picocom` |
| **GPIO** | Toggle output, read input | State changes | `gpioset`/`gpioget` |
| **Audio** (if present) | Play/record | Sound output / waveform | `aplay`, `arecord` |
| **Fan** | Set PWM duty | Fan speed changes | Write to PWM sysfs or use `jetson_clocks` |
| **Display** (if present) | Connect monitor | Desktop or framebuffer output | `xrandr` or check `/dev/fb0` |

---

## 14. Common bring-up failures and debug

| Symptom | Likely cause | Debug approach |
|---------|-------------|----------------|
| **No UART output at all** | Power sequencing wrong, RESET held low, UART TX/RX swapped | Check VIN under load, verify RESET timing, swap TX/RX |
| **DRAM training fail** | VIN voltage sag under load | Measure VIN with scope during boot, check regulator capacity |
| **Kernel boots but USB doesn't enumerate** | Impedance mismatch, bad solder on connector | Check USB eye diagram, re-inspect SoM connector solder joints |
| **PCIe device not detected** | PERST# not asserted correctly, clock not routed | Verify PERST# toggling with scope, check REFCLK |
| **CSI camera: no frames** | Lane mapping wrong in DT, clock/data skew | Verify DT camera node vs pinmux, check CSI trace lengths |
| **Ethernet link but no traffic** | Missing or wrong magnetics, MDIO misconfigured | Check PHY register access via MDIO, verify magnetics wiring |
| **Module thermal shutdown during AI** | Insufficient thermal solution | Improve heatsink, add fan, or reduce to 7W power mode |
| **Board ID EEPROM error in flash.sh** | EEPROM not populated or wrong I2C address | Use `flash.sh` env override flags or populate carrier EEPROM |

---

## 15. Projects

- **Minimal headless carrier:** Design a carrier for a headless edge node with USB-C power, GbE, NVMe M.2, debug UART, and fan header — no display, no USB hub. Target 4-layer PCB.
- **Robotics carrier:** Design a carrier with dual CAN-FD, 4x CSI-2, PoE PD, IMU (SPI), and GPS (UART). Target 6-layer PCB with industrial-temp components.
- **Bring-up validation suite:** Write a shell script that runs the full peripheral validation checklist from Section 13 and generates a pass/fail report.

---

## 16. Resources

| Resource | Description |
|----------|-------------|
| **NVIDIA Jetson Download Center** | Reference design files, Design Guides, pinmux spreadsheets |
| **Jetson Orin Nano OEM Product Design Guide** | Module pinout, power sequencing, thermal spec, keepout zones |
| **P3768 carrier reference design** | Altium/OrCAD schematic and layout files |
| **Jetson Orin Nano Design Guide** | Electrical and mechanical specifications |
| **KiCad** (kicad.org) | Open-source EDA for schematic and PCB layout |
| **Saturn PCB Design Toolkit** | Impedance calculator, via current, trace width calculator |
| **IPC-2221** | Generic standard on printed board design (trace width, spacing, via) |
| [2. L4T Customization](../3.%20L4T%20Customization/Guide.md) | BSP integration for custom carriers (flash config, DT porting) |
| [Jetson Module Adaptation and Bring-Up](../3.%20L4T%20Customization/Jetson-Module-Adaptation-Bring-Up-Orin-NX-Nano.md) | NVIDIA's guide for moving from dev kit to custom carrier |
| [Orin Nano Custom Board L4T Engineering Flow](../3.%20L4T%20Customization/Orin-Nano-8GB-Custom-Board-L4T-Engineering-Flow.md) | End-to-end engineering flowchart for custom carrier + L4T |
