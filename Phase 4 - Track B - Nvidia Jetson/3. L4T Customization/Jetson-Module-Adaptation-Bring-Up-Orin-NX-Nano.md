# Adapt Jetson Orin NX/Nano to a carrier board

**Outcome:** connect each schematic change to its boot, device-tree, driver and validation consequences. This lesson targets **T234 Orin NX/Nano** and current **Jetson Linux R39.2.1** documentation, reviewed 2026-10-18. It is an engineering procedure, not a claim that the course has flashed or qualified a custom carrier.

Start with [BCT contracts](T23x-Deployment.md) and the [CPU device-tree lab](../1.%20Nvidia%20Jetson%20Platform/Orin-Nano-Kernel-Internals/labs/README.md). Keep the manufacturer's release-matched [adaptation reference](https://docs.nvidia.com/jetson/archives/r39.2.1/DeveloperGuide/HR/JetsonModuleAdaptationAndBringUp/JetsonOrinNxNanoSeries.html) alongside the schematic while working.

## 1. Describe the differences, not just the new board name

Use a lowercase board/vendor name in the supported character set and keep the reference board intact. The new name should identify your configuration, but renaming a file does not adapt hardware.

Create a delta table before editing:

| Reference feature | Product difference | Consequences to investigate |
| --- | --- | --- |
| Carrier EEPROM present | No carrier EEPROM | MB2 carrier-identification policy; do not disable module identification |
| Default expansion-header I2C | IMU on another controller | Pinmux, pull-ups, voltage, DT bus/address, driver and reset |
| Reference USB connector | Fixed downstream hub | USB2/USB3 lane relationship, VBUS/regulator policy, reset and hub behavior |
| Reference NVMe connector | Different physical routing | PCIe controller/lane preset, reset/clock, power, recovery-flash access |
| Linux owns AON peripheral | SPE owns it | Disable competing Linux device, firmware permission, pinmux/clocks and protocol |

For each row, name the source of truth: schematic revision, module design guide, release source, or measured waveform. Keep assumptions visible until verified.

## 2. Preserve board identity

P3767 identifies a module family with several SKUs. A P3767-0005 developer-kit module is not interchangeable in a manifest with a P3767-0003 commercial Nano 8 GB module. Identify the carrier revision separately.

The documented EEPROM-less carrier change adjusts the **carrier** EEPROM read policy in MB2 configuration. It does not mean that all EEPROM information is optional, and it does not justify copying arbitrary `BOARDID`, `BOARDSKU`, `FAB` or RAM-code overrides. Derive such values from the matching hardware and release procedure.

**Failure exercise:** a flash tool reads the module correctly but fails probing the absent carrier EEPROM. Which file category should you investigate? **Answer:** the selected carrier/MB2 identification configuration. A random Linux I2C DT change cannot repair an earlier firmware probe.

## 3. Generate pinmux from the electrical contract

Use the exact module's spreadsheet and preserve the version and generated pinmux, GPIO and pad-voltage files. Review selected functions, direction, pull configuration, reset state and voltage assumptions against the schematic. Store generated outputs next to the input revision so a reviewer can reproduce them.

A product may need a reset line held low before the driver starts, an input without a strong internal pull, or a bus released for an external pull-up. Those requirements belong in both the startup behavior and runtime driver policy where applicable.

Do not use a late userspace service to hide an unsafe early pin state. If the circuit depends on a state before Linux, verify that state from reset with suitable instruments. The [GPIO namespace lesson](ODMDATA-and-GPIO-Jetson-Linux.md) explains why connector pins, DT specifiers and gpiochip offsets are different.

## 4. Edit the right device-tree layer

The R39.2.1 adaptation material describes source under `source/hardware/nvidia/t23x/nv-public`, with an upstream-aligned base and NVIDIA platform additions. Older `sources/hardware/nvidia/platform/t23x/.../kernel-dts` snippets belong to older trees. Inspect the release's actual includes and build rules rather than moving an old patch to a vaguely similar filename.

A practical source change should identify the controller, its children, power/reset/clock dependencies and binding. Use existing labels or the documented path within the matched source tree. Prefer a small board-specific delta over copying a large SoC description: a copied tree can miss later fixes.

**Worked sensor description:** an I2C sensor at address `0x48` is a child of the intended bus. Its `reg` describes the bus address, while the controller's `reg` describes MMIO ranges. The parent bus's address/size-cell rules determine the encoding. An interrupt specifier and an active-low reset GPIO have their own bindings; their numbers cannot be copied from a connector pinout.

Compile with the matched build system and run available binding checks. Decompiling a DTB is useful for inspection, but it loses source-level structure such as original includes and comments. It is not a replacement for maintaining the actual source patch.

## 5. Understand overlay selection

A built overlay matters only if the relevant boot stage selects and applies it. R39.2.1 distinguishes overlays included through `OVERLAY_DTB_FILE`, which affect UEFI and the kernel tree, from extra `OVERLAYS` in `extlinux.conf`, which affect the kernel tree. Keep that selection in the release manifest. [Overlay selection](https://docs.nvidia.com/jetson/archives/r39.2.1/DeveloperGuide/HR/JetsonModuleAdaptationAndBringUp/JetsonOrinNxNanoSeries.html#device-tree-overlays).

An overlay cannot retroactively change a pin state already consumed by MB1. A kernel-only overlay also cannot fix a UEFI-stage dependency. Predict the earliest consumer before choosing the mechanism.

In the [native overlay lab](../1.%20Nvidia%20Jetson%20Platform/Orin-Nano-Kernel-Internals/labs/README.md), a plugin overlay with an unresolved external label compiles but fails to merge. Another overlay compiles and merges yet violates the sensor's rate contract. These are different failure stages and should produce different diagnostics.

## UPHY lane configuration

Draw the physical lane allocation before selecting ODMDATA. Mark each lane's controller, role, width, connector and shared resources. The [ODMDATA lesson](ODMDATA-and-GPIO-Jetson-Linux.md) works through the C7 x2 versus C7/C9 x1 split and the current C4 endpoint procedure.

For PCIe, require agreement among lane assignment, enabled DT controller, PHY references, reset direction, reference clock and power sequencing. Begin by establishing a stable link at the intended topology, then examine negotiated width/speed and errors. A link training at x1 is not proof of an intended x2 connection merely because enumeration succeeds.

For USB, keep USB2 ports, SuperSpeed lanes, companion relationships and host/device/OTG roles consistent. An external hub also needs its power and reset design. A VBUS switch cannot be represented by an arbitrary always-on regulator unless that describes the actual hardware and required policy.

## 6. Build a complete installation set

The kernel image, in-tree and NVIDIA out-of-tree modules, DTBs and initramfs belong to the same build context. Use the release's documented toolchain and build targets. Do not select target module paths with the x86 host's `uname -r`. The [deployment lesson](Deployment.md) records the current layout and dependency order.

For a minimal rootfs, explicitly keep the NVIDIA integration and runtime dependencies needed by the product. CPU inference passing does not show that the GPU driver, multimedia stack or camera path survived a package-removal experiment.

The default rootfs, a reproducibly customized rootfs, and the recovery initrd can each require different device support. A custom carrier must support the selected recovery/flashing path as well as normal operation.

## 7. Bring up one dependency chain at a time

| Stage | Evidence before continuing |
| --- | --- |
| Power/reset | Correct rails and sequence, no abnormal current, documented reset/strap state |
| Recovery and serial | Identified module in recovery, retained serial log, known recovery path |
| Boot/storage | Correct firmware/kernel/root partition and repeatable cold boots |
| One bus/device | Correct driver, one known register/transaction, waveform if needed |
| Application | Bounded error handling and deterministic input/output check |
| Stress/update | Thermal/load tests, restart behavior, interrupted-update recovery |

A broad I2C scan is not a universally safe discovery step: probing can issue transactions some devices do not tolerate. Prefer the known bus/address and device-specific identification transaction. Likewise, a SPI loopback requires a deliberately prepared, electrically safe setup; it is not a command to run on an unknown live peripheral bus.

Retain logs before changing a configuration. Separate a missing driver, a failed probe, and incorrect data after a successful probe. Their likely causes differ.

## 8. Review exercise

A camera reset works after Linux boots, but the camera occasionally disappears after a cold power cycle. List three distinct hypotheses and observations.

**Example answer:** the pin's early state may be wrong (scope from reset); a rail/clock sequence may violate the sensor requirement (measure the relevant signals); or the deployed DT/driver may differ from the intended build (inspect selected artifact and live properties/probe log). “Add a sleep” is not evidence for any of those hypotheses.

Deliver the schematic delta table, source patch, generated input hashes, build/flash identity, observed running configuration, one controlled failure and a recovery record. Continue with the [end-to-end engineering flow](Orin-Nano-8GB-Custom-Board-L4T-Engineering-Flow.md).
