# ODMDATA, pinmux and GPIO: keep the namespaces separate

A schematic names a connector pin. A pinmux spreadsheet names a SoC ball. Device tree uses a controller-specific GPIO specifier. A userspace GPIO request uses a chip and line offset. Those identifiers may describe the same electrical signal while having different numeric values.

This lesson is for **T234 Orin NX/Nano**. Current BSP references are **R39.2.1**, reviewed 2026-10-18; older R35/R36 examples must retain their matching source and tool versions. Read [BCT contracts](T23x-Deployment.md) first.

## 1. ODMDATA selects a supported hardware arrangement

`ODMDATA` is part of the flash/boot configuration. On Orin NX/Nano, named presets select UPHY arrangements. The choice must agree with carrier routing, controller role, pinmux and Linux device tree. Changing a string cannot physically reroute differential pairs.

The reference arrangement uses `gbe-uphy-config-8` with C7 occupying two lanes; selecting `gbe-uphy-config-9` splits those resources between C7 and C9 as one-lane controllers. This is a topology choice, not two ways to get the same x2 bandwidth. Keep a carrier-specific override after the common board configuration so the default does not overwrite it. [NVIDIA UPHY lane configuration](https://docs.nvidia.com/jetson/archives/r39.2.1/DeveloperGuide/HR/JetsonModuleAdaptationAndBringUp/JetsonOrinNxNanoSeries.html#uphy-lane-configuration).

**Worked allocation:** a fictional two-lane pool can support one x2 link or two x1 links if the controller supports that arrangement. It cannot provide two x2 links simultaneously. A disabled DT node may prevent a driver from claiming a controller, but that alone does not change the early lane assignment.

The current documented NX/Nano endpoint procedure uses **C4** and `hsio-uphy-config-41`; it is distinct from AGX Orin's endpoint arrangement. An endpoint requires the matching controller mode, reset direction, clock relationship and endpoint function. It is not enabled merely by exposing a character device. Use the release procedure and actual BSP script location; do not transplant an AGX ODMDATA string. [NVIDIA endpoint procedure](https://docs.nvidia.com/jetson/archives/r39.2.1/DeveloperGuide/SD/Communications/PcieEndpointMode.html).

**Check:** changing endpoint metadata back on the build host affects a later generated image. It does not remotely change a board already flashed in endpoint mode.

## 2. Three layers of GPIO configuration

| Layer | Question answered | Persistence and ownership |
| --- | --- | --- |
| Pad/pinmux and boot GPIO configuration | Which function is connected, at what electrical configuration, and in what early state? | Generated BSP inputs must be selected and deployed |
| Kernel pinctrl/GPIO/controller driver | Which device owns the signal and exposes a supported interface? | Controlled by the running kernel and device description |
| Userspace GPIO line request | Which logical value/direction/events does this process request? | Valid while the request is held; release does not promise a retained output state |

A line already owned by a driver should not be forcibly repurposed by a user process. Likewise, handing an AON controller to SPE requires removing the conflicting Linux owner and satisfying clocks, pinmux and access permissions. See the [SPE ownership lesson](../4.%20FSP%20%28Firmware%20Support%20Package%29%20Customization/Guide.md).

## 3. Work through one identifier without touching hardware

Consider **AON port DD, pin 2**, as a mapping exercise rather than a wiring instruction.

| Name space | Illustrative value | Meaning |
| --- | --- | --- |
| SoC port/pin | DD2 | Named signal inside the GPIO controller |
| DT binding macro | `TEGRA234_AON_GPIO(DD, 2)` → 26 | Binding encoding uses port index 3 times eight plus pin 2 |
| Packed gpiochip line offset | 22 | The preceding implemented ports have 8 + 4 + 8 lines; then add 2 |
| Legacy global GPIO number | Historical base + 22 | Dynamic global numbering; unsuitable as a portable identifier |
| Carrier connector pin | Board-dependent | Requires the carrier schematic; cannot be derived from 22 or 26 |

The upstream [T234 GPIO binding header](https://github.com/torvalds/linux/blob/master/include/dt-bindings/gpio/tegra234-gpio.h) defines the DT encoding. The [Tegra GPIO driver's translation](https://github.com/torvalds/linux/blob/master/drivers/gpio/gpio-tegra186.c) maps the specifier into its packed line representation. Confirm the **matched BSP driver and actual `gpioinfo` output** before using any value; the table teaches why blindly copying the macro's numeric value into a userspace request can select the wrong line.

A line name is useful but can be missing or duplicated. Record the verified GPIO chip identity, line name if present, offset, carrier revision and electrical function. `gpiochip0` is a discovery result, not a permanent name for every board and boot configuration.

**Exercise:** a historical log reports global base 316 for the AON controller. The old global number for this example is 316 + 22 = 338. Does `gpioset` on a modern character-device API take 338 as the line offset? **Answer:** no. The request identifies the chip and its local line; 338 belongs to a different namespace.

## 4. Inspect the installed GPIO API

On a target with the tools installed, these observations do not intentionally change line values:

```bash
gpiodetect
gpioinfo --version
gpioinfo --help
gpioset --help
```

Then use the matching version's `gpioinfo` syntax to inspect the chosen chip. libgpiod 1.x and 2.x command formats differ. For example, the older `--mode=wait` idiom is not a generic option for every release. In current 2.x tools, requests are held until the tool exits unless another supported mode is selected; after release the line state is not guaranteed. [Official GPIO tools](https://libgpiod.readthedocs.io/en/master/gpio_tools.html).

Reading with `gpioget` can request/reconfigure a line; it is not interchangeable with inspecting line information. Do not run it against an output or bus signal merely to “see what happens.” For a known safe test signal, select the proper bias, active-low and drive mode from the circuit's contract.

The old `/sys/class/gpio` interface is deprecated for new work. The character-device API improves request ownership, but it cannot prove that a voltage is safe or that a signal should be driven. Continue with the [current GPIO/SPI/I2C/CAN guide](../1.%20Nvidia%20Jetson%20Platform/Orin-Nano-GPIO-SPI-I2C-CAN/Guide.md) for actual application interfaces and CPU protocol exercises.

## 5. Active-low describes logic, not an inverter you can ignore

Suppose `RESET_N` is active-low. A logical **active** request means asserting reset and therefore driving the electrical signal low, subject to the configured drive type. A logical **inactive** request deasserts it. Writing raw `1` without considering the API's active-low flag can reverse your intent.

A signal with an external pull-up and open-drain output is released by ceasing to pull it low. It is not equivalent to connecting an additional push-pull high driver. Internal bias is also not a substitute for a required external resistor, particularly across reset and unpowered states.

## 6. Direct MMIO is a diagnostic technique with a narrower contract

`devmem` bypasses the normal driver's ownership and sequencing. Even a read may have side effects for some registers. Before any board experiment, identify the exact TRM block, offset, width, legal access semantics, clock/power state and current owner. A borrowed hexadecimal write is not a portable pinmux setup procedure.

For reasoning practice, do the masked-write exercise in [BCT contracts](T23x-Deployment.md#3-a-masked-write-preserves-unrelated-bits) entirely in Python. If a target experiment confirms a pinmux hypothesis, express the intended setting in the supported source configuration and validate it across cold boot and suspend/resume. A successful runtime poke is not evidence that the production boot image has changed.

Never interpret an SCR table index such as `reg@1359` as an MMIO address. The index selects a predefined firmware entry; it does not identify where userspace should map memory.

## 7. Diagnose a failed line or link

| Observation | Plausible next checks |
| --- | --- |
| GPIO request returns busy | Inspect the existing consumer and intended ownership |
| Output API succeeds but no measured transition | Wrong line namespace, mux/SFIO state, electrical routing, driver mode, measurement reference |
| Line works until the tool exits | The request was released; design a persistent owner or hardware default |
| Works after a register poke, fails after reboot | The boot inputs or selected artifacts were not updated |
| PCIe link does not train | Routing, clock/reset/power sequence, UPHY selection, controller role and DT |
| Correct GPIO edge but late reaction | Separate capture time, interrupt delivery, scheduling, queueing and application work |

Submit a complete mapping for one schematic signal and explain at least two numeric identifiers that must not be substituted for one another. No physical access is required to demonstrate the mapping arithmetic; target behavior remains a separate acceptance step.
