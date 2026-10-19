# Custom carrier boards: connect requirements, schematics and bring-up evidence

**Outcome:** create a reviewable carrier design brief, resource map, power/thermal budget and staged bring-up plan for a specified Jetson module. **Pace:** six 60–90-minute sessions before a hardware design project. Prerequisites are [schematic/PCB fundamentals](../../Phase%202%20-%20Embedded%20Systems/1.%20Schematic%20Capture%20and%20PCB%20Design/Guide.md), basic power electronics and the [Jetson platform introduction](../1.%20Nvidia%20Jetson%20Platform/Guide.md).

The carrier connects a system-on-module to power, storage, sensors and other devices. A successful design must agree with both the module's electrical requirements and the software configuration that describes those connections. A connector being present on a drawing does not create another hardware controller or high-speed lane.

**Reviewed 18 October 2026.** Download the exact Orin NX/Nano product design guide, module datasheet, P3768 carrier specification/design files, pinmux template and mechanical information from the [NVIDIA download center](https://developer.nvidia.com/embedded/downloads). Record document revisions and module/carrier SKUs. This lesson teaches the engineering method; it does not replace those electrical specifications or claim that a fabricated carrier was validated in this review.

## 1. Decide what the product needs

Write an interface requirement for each device: bandwidth, latency/freshness, distance, connector, power, environment and failure behavior. A mobile robot might need two cameras, one IMU, a motor-controller link, NVMe, Ethernet and service/debug access. “Four cameras” is incomplete until you state sensor modes, lane requirements, synchronization and the processing path.

A custom carrier can improve fit, connector retention, power distribution and production control. It is not universally required above an arbitrary unit count, nor does a developer kit automatically satisfy a finished product's reliability or conformity requirements. Evaluate development cost, qualification, lifecycle and available supported commercial carriers against the product requirements.

Separate the module's capabilities from what this carrier exposes. Pin compatibility across modules does not prove every power, interface and software configuration is interchangeable. Retain the exact configuration identity throughout design, BSP and test records.

## 2. Read the reference board accurately

The Orin Nano developer kit combines a P3767 module with a P3768 reference carrier. Its developer module has a microSD socket; production-module storage arrangements differ. The official layout identifies a DC power jack, a **data-only USB-C** port, four USB Type-A ports, DisplayPort, two CSI connectors and M.2 slots. The Key-E slot is populated by the included wireless module. DisplayPort is the kit's display output; it does not offer native HDMI or DisplayPort over USB-C. [NVIDIA hardware layout](https://docs.nvidia.com/jetson/orin-nano-devkit/user-guide/latest/hardware_layout.html).

These details prevent several design mistakes:

- Removing an imagined carrier microSD slot does not describe the developer module's socket.
- A USB-C connector is not automatically a USB-PD power input or a display output.
- A 40-pin header's mechanical similarity to another board does not establish identical voltage, pinmux or drive limits.
- AGX Orin connector/pin assignments are not Nano carrier assignments.
- A CSI connector cannot become a long-distance automotive camera link merely by changing it to a FAKRA connector; serializers, deserializers and the complete link design matter.

Use the specified 260-contact module connector and the exact compatible footprint/part/mechanical constraints from the design package. Verify orientation, seating height, retention and keep-outs. Do not replace a specified module edge-connector footprint with a generic “board-to-board” symbol because the contact count matches.

Walk the schematic by function: input power/protection, module interface, storage, external I/O, camera/control buses, clock/reset, debug/recovery and thermal/fan support. Follow each external signal back to the module and its supporting components. Retain the required termination, biasing and protection when removing an apparently unused reference block.

## 3. Allocate controllers, lanes and pins together

Create one resource table before drawing connector variants:

| Resource | Record for the chosen design |
| --- | --- |
| PCIe/UPHY | Controller, lane grouping, supported mode, endpoint, reset and reference clock |
| CSI | Sensor mode, lane count/mapping, connector, clock/control bus and synchronization |
| USB | Controller/port role, hub topology, power switch and protection |
| GPIO/low-speed I/O | SoC pad, voltage domain, pinmux, pull, ownership and connector pin |
| Ethernet/CAN | Actual module interface, required external physical-layer circuitry and connector |
| Boot/debug | Recovery path, console, reset/power controls and test access |

Check the complete legal lane/pin configuration, not each desired interface in isolation. Adding two CAN connectors does not create two independent CAN controllers. An external controller and its bus/driver may be needed for an additional independent channel. A passive split also changes the physical bus topology rather than creating isolation.

Pinmux, MB1/MB2 configuration, device tree, ODMDATA and flash configuration must match the new carrier. NVIDIA explicitly lists those adaptation areas for Orin NX/Nano. Keep the board name and revision traceable instead of editing the reference board under its original identity. [NVIDIA module-adaptation guide](https://docs.nvidia.com/jetson/archives/r39.2/DeveloperGuide/HR/JetsonModuleAdaptationAndBringUp/JetsonOrinNxNanoSeries.html).

**Exercise:** your IMU data-ready pad is claimed by Linux GPIO while firmware on another processor expects to own it. Identify the electrical connection, boot ownership configuration, Linux consumer and interprocessor protocol that need to agree. Changing only an application pin number is insufficient.

## 4. Calculate a power budget with clear boundaries

Distinguish the external supply, carrier conversion rails and module input. Use the exact module voltage range, sequencing and transient requirements from the selected datasheet; do not treat the developer kit's barrel-input voltage as a direct module-pin specification. Do not derive a universal current limit from an old 7 W/15 W operating-mode table.

**Illustrative system budget:** allocate 25 W to the module, 6 W to storage, 4 W to cameras and 3 W to the remaining carrier. The loads sum to 38 W. With an assumed 90% aggregate conversion efficiency, the external input estimate is `38 / 0.90 ≈ 42.2 W`. At an illustrative 12 V input, that is about 3.52 A average under this load assumption.

Those numbers are a calculation exercise, not approved module voltages or a final regulator selection. Add startup/inrush, transient response, cable drop, temperature derating and the appropriate margin using actual component specifications and measured load profiles. A supply's headline wattage does not prove that every downstream rail stays in range during a GPU/storage/camera load step.

For each rail, record nominal/range, steady and transient current, source, enable dependency, discharge behavior and measurement point. Derive the sequence from the module specification and reference circuit; do not invent a three-step reset sequence from signal names alone. Check back-powering paths from USB, GPIO and externally powered sensors during off/reset states.

**Voltage-drop exercise:** 3 A through an assumed 80 mΩ round-trip cable/contact path drops 0.24 V and dissipates 0.72 W. Measure at the relevant load pins as well as at the supply. A multimeter's steady reading can miss a short brownout; capture relevant transients with suitable probing.

## 5. Design the thermal path and mechanical assembly together

Define ambient range, enclosure airflow, mounting orientation, dust exposure, fan behavior and sustained workload. Power mode is an operating constraint, not an automatic statement of every component's heat or total system input power.

For a simple steady-state exercise, a 25 W heat load, 40°C ambient and an illustrative 75°C design target allow `(75−40)/25 = 1.4 K/W` from the modeled heat source to ambient. That target is not a Jetson absolute temperature limit. Real systems have several heat sources, interfaces, transient thermal capacitances and sensor locations. Confirm the target and model against the actual thermal specification and measurements.

Inspect thermal-interface thickness/compression, mechanical tolerances, mounting loads and heatsink retention. A fan spinning does not establish contact quality or adequate airflow. Record temperature and clocks under a representative sustained load and at the stated ambient; an idle reading is weak evidence for a robot running perception continuously.

The [power/thermal CPU lab](../1.%20Nvidia%20Jetson%20Platform/Orin-Nano-Power-Thermal/labs/README.md) teaches energy integration and a thermal RC model. Complete it before interpreting a tegrastats plot or claiming energy per useful result.

## 6. Route from an agreed stackup and interface specification

Coordinate the PCB stackup, copper thickness and controlled impedance with the fabricator. Obtain trace geometry from that stackup and the relevant interface requirements. Avoid a universal trace-width, pair-skew or “four layers is enough” recipe for every carrier.

Preserve return-current paths, connector transitions and reference planes. Consider vias/stubs, coupling, layer transitions and the entire channel, including cable and connector. Pair matching alone does not prove acceptable loss or signal integrity. Protect low-voltage sensor interfaces from inappropriate external levels and transients with a circuit suited to their speed and electrical domain.

Separate the tasks of schematic review, ERC, layout DRC, signal/power-integrity analysis and physical validation. Each catches different faults. A clean DRC cannot identify a legal-looking net connected to the wrong module function.

Review manufacturability and assembly access before release: connector seating, fasteners, test pads, cooling installation and the ability to reach recovery/debug signals in the enclosure. Reference-design reuse preserves useful starting points, but every deliberate change needs its own rationale and verification.

## 7. Prepare the BSP before the first board arrives

Keep hardware and software artifacts aligned: schematic/net names, pinmux source, generated boot configuration, device tree, kernel/module configuration, flash layout and board revision. Record what is generated and from which tool/document revision; do not hand-edit generated output without retaining a reproducible source change.

Run the [device-tree CPU lab](../1.%20Nvidia%20Jetson%20Platform/Orin-Nano-Kernel-Internals/labs/README.md). It actually compiles a fictional base tree and overlay, applies the overlay and inspects the resulting properties. It teaches phandles, paths and property encoding, while explicitly making no claim that its fictional nodes configure an Orin carrier.

Then follow [L4T customization](../3.%20L4T%20Customization/Guide.md) for the target-specific artifacts. A DT node existing in a file does not prove that firmware selected that file, a driver bound successfully or the physical interface works. Retain the observed boot/device-tree identity as evidence.

## 8. Bring up one dependency at a time

Use staged gates and record measured values, settings and exact hardware revision:

| Gate | Evidence before continuing |
| --- | --- |
| Unpowered inspection | Assembly/orientation, shorts and connector condition reviewed |
| Carrier power qualification | Required rails and enable/reset sequence observed under the specified conditions |
| Controlled module startup | Correct insertion procedure, supply behavior and console/recovery path |
| Boot identity | Board/BSP/DT/kernel/storage identity matches the intended build |
| One interface at a time | Driver binding plus a test that exercises the actual connector/device path |
| Combined workload | Power, thermal, bandwidth and latency behavior with concurrent required devices |
| Failure/recovery | Controlled disconnect/reset/update failures produce the specified recoverable behavior |

Do not hot-insert the module as a casual test step. Use the manufacturer's handling/power procedure and suitable current limiting/measurement equipment for the setup. Current limits must account for expected startup behavior; a limit set too low can itself create a misleading boot failure.

For a missing interface, separate hypotheses: electrical connection, power/reset/clock, boot configuration, driver binding, application access and protocol correctness. “Kernel booted but USB did not enumerate” is not enough to conclude impedance mismatch; disabled configuration, reset, power and assembly errors are also candidates.

Test communication and data meaning separately. A sensor may answer while reporting the wrong range, endian order, scale or calibration. The [peripheral course](../5.%20Application%20Development/1.%20Peripheral%20Access/Guide.md) provides a real C/Python protocol exercise and freshness checks.

## 9. Turn findings into a release record

For each issue, retain reproduction conditions, observations, hypothesis, controlled change and result. Keep temporary rework separate from the released schematic/BOM/BSP so it is not lost between prototypes and manufacturing.

Choose product-specific pass criteria from the requirements. Do not use a universal NVMe throughput, boot time or temperature threshold independent of the chosen configuration. Where a test is pending, identify the missing measurement/equipment instead of filling the cell with a reference-board expectation.

Connect the qualified configuration to the [factory-record lab](../7.%20Compliance%20and%20Manufacturing/labs/README.md) and [secure update plan](../6.%20Security%20and%20OTA/Guide.md). A later module, carrier, sensor or model revision can change the system that was originally accepted.

## 10. Completion exercise and answers

Produce a design brief for a headless robot carrier with the required cameras, controller link, storage and recovery access. Submit the resource table, power/thermal calculations, interface document references, proposed stackup constraints, BSP artifact map and staged test plan. A learner without EDA tools or hardware can complete this design-review exercise and the linked CPU labs.

1. **Why is USB-C insufficient to specify power input?** Connector shape does not establish power role, negotiation, protection or the board's supported circuit.
2. **Why can an extra camera connector fail to add camera capacity?** Lanes, receiver resources, sensor modes and software support may already be constrained.
3. **Why measure at load pins?** Cable, connector and PCB impedance create drops/transients not visible at the supply output alone.
4. **Why preserve recovery access?** A board/BSP error can prevent normal boot and network access.
5. **What does a passing DT overlay test establish?** The tested tree manipulation works; physical board correctness still needs target evidence.
6. **Why retain the failed factory attempt after rework?** It preserves defect and first-pass information needed to improve the design/process.
