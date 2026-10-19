# T234 boot configuration: from intent to BCT

**Scope:** Orin NX/Nano and the T234 boot contract. References were checked on **2026-10-18** against Jetson Linux R39.2.1 and explicitly identified archived T234 material. This lesson replaces a duplicated June 2022 PDF transcription with worked explanations. The manufacturer's tables remain authoritative for real register values and supported configurations.

By the end, you should be able to identify a BCT consumer, explain a masked register update, distinguish a configuration index from a hardware address, and trace a generated input into a selected boot artifact. No physical register writes or flashes are required for the arithmetic exercises.

## 1. Why a Linux device tree is not enough

Linux starts after memory, essential power, boot storage and several processors have already been initialized. Configuration required for that earlier work cannot depend on a Linux driver loading later.

A **Boot Configuration Table**, or BCT, carries platform data to an early boot component. NVIDIA's tooling turns release-specific sources into the corresponding binary representation. A file ending in `.dtsi` can be an input to this process without being a Linux driver description. Its **consumer and binding** determine its meaning. [BCT introduction](https://docs.nvidia.com/jetson/archives/r39.2.1/DeveloperGuide/SD/Bootloader/T23xBCTLoaderIntro.html).

| Artifact | Main purpose |
| --- | --- |
| BR-BCT | BootROM-facing boot configuration |
| MB1-BCT | Early platform initialization, including pinmux and power-related settings |
| Mem-BCT | Memory-controller/DRAM initialization parameters |
| MB2-BCT | Later boot configuration, including associated security/interrupt/miscellaneous inputs |
| Kernel DTB | Hardware description consumed by Linux drivers |
| BPMP DTB | Configuration consumed by BPMP firmware |

This is a responsibility map, not a claim that every field is consumed by only one component. Follow the selected release's actual build graph and boot log. The Orin boot path and Thor boot path are different; sharing a current documentation site does not make their BCT fields interchangeable. [Orin boot flow](https://docs.nvidia.com/jetson/archives/r39.2.1/DeveloperGuide/AR/BootArchitecture/JetsonOrinSeriesBootFlow.html).

## Orin Nano: BCT path cheat sheet

The P3767 module family includes NX and Nano SKUs; P3768 names the reference carrier and P3766 the Nano developer kit. Preserve the exact module SKU, not only the family number.

In the current Orin adaptation workflow, generated pinmux and pad-voltage files go under `Linux_for_Tegra/bootloader/generic/BCT/`, while the generated GPIO file goes under `Linux_for_Tegra/bootloader/`. Older R35 SPE examples use `t186ref/BCT/` for some files. Those are release-specific locations, not two folders into which every file should be copied. [Current adaptation guide](https://docs.nvidia.com/jetson/archives/r39.2.1/DeveloperGuide/HR/JetsonModuleAdaptationAndBringUp/JetsonOrinNxNanoSeries.html).

Use the selected board `.conf` and its included files to discover the actual input filenames. Do not guess `PINMUX_CONFIG`, `DTB_FILE` or other variable spellings from an unrelated release. A shell `.conf` can execute code; inspect trusted source before sourcing it. A text search is useful for review but is not a complete shell evaluator:

```bash
# Host, inside a trusted extracted BSP; read-only inspection.
rg -n 'PINMUX_CONFIG|PMC_CONFIG|GPIOINT_CONFIG|MB2_BCT|SCR_CONFIG|DTB_FILE|ODMDATA' \
  p3767.conf.common jetson-orin-nano-devkit.conf
```

Generated file names, inclusion order and later overrides matter. Save both the spreadsheet input and its generated output. A spreadsheet change that never regenerates the selected `.dtsi` cannot alter the image.

## 2. Pinmux, direction and pad voltage are separate decisions

For one physical signal, record: connector pin, SoC ball name, function, voltage domain, input enable, output/tristate behavior, pulls, and reset/boot state. A UART TX, I2C open-drain signal and active-low reset line have different electrical contracts.

A GPIO line set high by Linux says little about its state during reset or MB1 execution. If a peripheral must remain reset until power is stable, verify that early state electrically and that no later driver briefly drives the opposite level.

Pad tolerance is not a level shifter. Enabling a named tolerance setting does not authorize arbitrary voltages or override the module's electrical limits. Use the exact module design guide and generated pinmux contract. “Bidirectional” in a spreadsheet is also not proof that an attached device tolerates simultaneous push-pull drive.

**Exercise:** an active-low reset is pulled low externally, and software should release it after initialization. Is `output-low` equivalent to “deasserted”? **Answer:** no; active-low means low asserts reset. Whether release is a driven high or high-impedance/open-drain action depends on the actual circuit and binding.

## 3. A masked write preserves unrelated bits

For an ordinary readable/writable register, a characterized update often follows:

```text
new = (old & ~mask) | (data & mask)
```

NVIDIA documents this pattern for common pad-level production settings. “Prod” means characterized interface settings, not a performance knob to maximize blindly. [Common prod configuration](https://docs.nvidia.com/jetson/archives/r39.2.1/DeveloperGuide/SD/Bootloader/CommonProdConfig.html).

Work through this invented 8-bit example:

```text
old  = 10100101 = 0xA5
mask = 00001100 = 0x0C
 data= 00001000 = 0x08
new  = 10101001 = 0xA9
```

Only bits 3:2 change, from `01` to `10`. Verify without hardware:

```bash
python3 - <<'PYCODE'
old, mask, data = 0xA5, 0x0C, 0x08
new = (old & (~mask & 0xFF)) | (data & mask)
assert new == 0xA9
assert ((old ^ new) & (~mask & 0xFF)) == 0
print(hex(new))
PYCODE
```

The explicit width is useful because Python integers have arbitrary precision. In C, use the intended unsigned register width and avoid shifting beyond it.

**Boundary:** do not apply this formula indiscriminately to write-one-to-clear, read-to-clear, write-only or side-effect registers. A read-modify-write can acknowledge events accidentally or race another owner. The register specification determines the legal access sequence.

## 4. Address arithmetic and table indexing are different

An MMIO address can be computed as a documented block base plus register offset. For a fictional block at `0x10000000` and offset `0x84`, the result is `0x10000084`. Width, alignment, mapping and access permissions still need checking before an access.

An SCR entry such as `reg@1359` is **an index in a predefined configuration table**, not the physical address `0x1359`. NVIDIA's archived T234 security documentation defines the index interpretation. Never pass it to `devmem` or derive a bitfield from the node name. [T234 security configuration](https://docs.nvidia.com/jetson/archives/r35.6.0/DeveloperGuide/SD/Bootloader/SecurityConfig.html).

The R39.2.1 security page currently discusses **T264** fields as well. Do not carry its expanded `exclusion-info` interpretation into a T234/R35 firmware recipe. A current page can contain SoC-specific content that does not apply to the board in front of you. [Current security configuration](https://docs.nvidia.com/jetson/archives/r39.2.1/DeveloperGuide/SD/Bootloader/SecurityConfig.html).

**Exercise:** a recipe changes `0x38000222` to `0x38001232`. What bits differ? XOR gives `0x1010`, so bits 12 and 4 differ. This arithmetic does **not** identify their permission meanings. The matching firmware table/TRM must provide those meanings; opening every permission to remove a fault is not a valid ownership design.

## 5. Interrupt routing is not a Linux IRQ number

A GPIO interrupt-map value selects a hardware route. It is not a GPIO chip offset, header pin number, or the final Linux virtual IRQ number. For example, `pin-2-int-line = <2>` associates pin 2 in the named port with route 2. `pin-0-int-line = <4>` selects route **4**; a nearby comment saying “INT0” is wrong even if copied from an older example. [GPIO interrupt mapping](https://docs.nvidia.com/jetson/archives/r39.2.1/DeveloperGuide/SD/Bootloader/GpioInterruptMapConfig.html).

Several pins sharing a route may require the handler to inspect pending status. Route selection alone does not specify the event's edge polarity, clear sequence, interrupt priority or maximum latency. Treat a timestamp taken in a handler as a delivery observation unless hardware captured the edge earlier.

## 6. UPHY, power and memory need their own evidence

UPHY lane assignment must match the physical differential-pair routing and the enabled controllers. A desired PCIe x2 link cannot use lanes already assigned to another controller. See the [worked lane-allocation lesson](ODMDATA-and-GPIO-Jetson-Linux.md).

PMIC sequencing describes dependencies and delays, not just a list of voltages. If rail B requires rail A to be stable first, a correct final voltage reading does not prove the startup order. Measure the waveform and reset relationship. Use the exact reference design and platform configuration; the old PDF's example I2C addresses are not a custom carrier power recipe.

Mem-BCT includes DRAM initialization information matched to the module's memory configuration. A custom carrier does not automatically justify changing module DRAM timing. Preserve the supported module data unless the vendor's engineering procedure specifically requires a change.

Firmware rollback/ratchet settings belong to the security lifecycle. A monotonic version floor can intentionally make older images unusable; it is not a reversible tuning setting. Study the [security chapter](../6.%20Security%20and%20OTA/Guide.md) before provisioning such policy.

## 7. Prove selection and behavior

For each change, make a five-column record: requirement, source input, generated artifact, selected flash input, observed behavior. Example:

| Requirement | Source | Build/flash proof | Target proof |
| --- | --- | --- | --- |
| Reset remains asserted until driver ready | Spreadsheet and GPIO initialization | Generated pinmux/GPIO hashes and selected BCT inputs | Scope trace from reset through driver startup |
| Linux owns the IMU controller | Kernel DTS and binding | Compiled/merged controller properties | Bound driver and known sensor read |
| SPE owns an AON controller | Matched firmware, DT disable, permissions | Consistent SPE + BCT + DT artifact set | SPE transaction, no competing Linux owner |

The first two proofs are not interchangeable. A successful binary build establishes that tooling accepted the configuration; only target evidence shows what the hardware did.

Continue with [module adaptation](Jetson-Module-Adaptation-Bring-Up-Orin-NX-Nano.md). Use [Deployment](Deployment.md) for packaging and boot evidence, and the [SPE lesson](../4.%20FSP%20%28Firmware%20Support%20Package%29%20Customization/Guide.md) for ownership across processors.
