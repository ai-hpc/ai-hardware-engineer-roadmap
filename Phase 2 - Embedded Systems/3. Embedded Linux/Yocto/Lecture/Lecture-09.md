# Lecture 9 — Module 7: connect boot artifacts to hardware

**Course:** [Yocto guide](../Guide.md) · **Previous:** [Lecture 08](Lecture-08.md) · **Next:** [Lecture 10](Lecture-10.md)

A root filesystem is only one part of a bootable system. An integrator must connect firmware, bootloader, kernel, hardware description, rootfs, and startup behavior for the chosen machine.

## Trace the selected target

For this course's QEMU x86-64 baseline, the launch tool supplies a virtual platform and compatible boot path. Do not invent a device-tree blob or a separately flashed U-Boot stage just because an Arm board uses them. A physical board's boot chain comes from its SoC/board documentation and BSP.

| Stage | Question to answer |
|---|---|
| Firmware / bootloader | Who selects and loads the kernel, and from where? |
| Kernel | Which provider, version, configuration, and patches are selected? |
| Hardware description | Device tree, ACPI, or another platform mechanism? |
| Root filesystem | Which file/partition is mounted, and with what command line? |
| PID 1 / services | What starts the application, and how are failures visible? |

**Upstream-documented inspection commands:**

```bash
bitbake-getvar PREFERRED_PROVIDER_virtual/kernel
bitbake-getvar -r virtual/kernel PN
bitbake-getvar -r virtual/kernel PV
bitbake-getvar -r virtual/kernel KERNEL_DEVICETREE
ls tmp/deploy/images/qemux86-64
```

An unset preferred-provider variable is not itself an error: other selection rules can choose a provider. Use the resolved recipe identity and build output. A variable may be empty because it does not apply to the target. The [kernel development manual](https://docs.yoctoproject.org/5.0.20/kernel-dev/index.html) explains kernel integration paths.

## Worked failure diagnosis

Suppose the boot log mounts the root filesystem and displays a login prompt, but the age-check program is absent. The bootloader and kernel already achieved some useful work; inspect the image/package path from Lecture 8. Suppose the kernel instead reports that it cannot mount root. Inspect the root device, filesystem support, command line, and image layout before changing the application's recipe.

For a real robot camera or CAN controller, draw the path from schematic/bus resources to the hardware description, driver binding, device node, and userspace. A device-tree entry describes resources; it cannot supply a missing driver or repair incorrect wiring.

## Integration exercise

Make a boot-artifact table containing producer recipe, exact output filename, checksum, consumer stage, and observation proving the stage ran. In a disposable QEMU launch configuration, choose an incorrect root filesystem path and observe the failure, then restore the correct path. Do not alter physical-board boot media for this exercise.

**Self-check:** is a successful `bitbake virtual/kernel` enough to prove a new driver works? **Answer:** no. The matching kernel/configuration must boot on the target, bind the intended device, and pass a functional and failure test.

**Artifact:** annotated boot log and artifact map. For a board-specific extension, add a verified peripheral trace with recovery from a bounded missing-device condition.

**Previous:** [Lecture 08](Lecture-08.md) · **Next:** [Lecture 10](Lecture-10.md)
