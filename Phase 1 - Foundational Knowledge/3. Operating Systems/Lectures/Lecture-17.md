# Lecture 17: Linux Device Driver Model & Device Tree

## Overview

Every piece of hardware connected to a Linux system — whether a USB camera, a PCIe GPU, or a custom AI accelerator on an SoC — needs a kernel driver to manage it. The challenge is that hardware is enormously diverse: different buses, different configuration mechanisms, and different resource layouts. The Linux **driver model** provides a unified framework so that a driver written once works regardless of how the hardware was discovered. The mental model is a matchmaking service: devices announce their identity, drivers announce what they support, and the kernel bus core pairs them. For embedded SoC hardware (Jetson, Zynq, custom FPGA boards), the **Device Tree** is the mechanism by which hardware identity is described to the kernel at boot. For an AI hardware engineer, understanding the driver model and Device Tree is essential for bringing up custom sensors, configuring camera pipelines, and writing drivers for AI accelerator peripherals.

---

## Linux Driver Model: Core Abstractions

The Linux device model provides a **unified framework** across all buses. Three primary structures, defined in `include/linux/device.h`, represent the hardware and software entities:

```
Linux Driver Model: Match → Probe → Manage

Bus Core (e.g., platform_bus)
    │
    ├── Device Registry (linked list of registered devices)
    │       device_A: compatible="vendor,mydevice-v2"
    │       device_B: compatible="arm,pl011"
    │       device_C: VID=0x8086, PID=0x1592
    │
    ├── Driver Registry (linked list of registered drivers)
    │       driver_X: of_match_table="vendor,mydevice-v2"
    │       driver_Y: of_match_table="arm,pl011"
    │
    └── Match Loop
            for each (device, driver) pair:
                if bus_match(device, driver):
                    driver->probe(device)   ← sets up resources
                    (later) driver->remove(device) ← releases resources
```

| Object | Structure | Role |
|--------|-----------|------|
| Bus | `struct bus_type` | Enumeration and matching logic (PCIe, USB, I2C, SPI, platform) |
| Device | `struct device` | One instance of physical or virtual hardware |
| Driver | `struct device_driver` | Code that manages a specific device type |

The bus core compares each driver's `id_table` (PCI/USB) or `of_match_table` (Device Tree `compatible`) against each registered device. **On a match**, the bus calls `driver->probe(device)`. On removal or driver unbind, it calls `driver->remove(device)`.

> **Key Insight:** The probe/remove lifecycle means a driver never hardcodes its device's resource addresses. The bus framework passes the `platform_device` to `probe()`, and the driver queries its resources (MMIO, IRQ, clocks) from that object. This makes the same driver binary usable across multiple hardware variants that differ only in their Device Tree descriptions.

---

## Platform Devices

PCIe and USB devices are **self-describing**: they carry vendor IDs, device IDs, and capability registers that the bus can read automatically. SoC peripherals have **no such mechanism** — there is no way to dynamically discover a UART controller at address 0xFE200000.

SoC peripherals (UART, I2C controller, camera CSI, AI accelerator, FPGA AXI slave) are not self-describing; the bus cannot enumerate them. They are registered as **platform devices** using descriptions from Device Tree or ACPI.

```
Platform Driver Registration

Device Tree (.dts)          Kernel Driver (C code)
┌──────────────────┐        ┌─────────────────────────────┐
│ mydev@40000000   │        │ static const struct         │
│ compatible =     │        │   of_device_id mydev_ids[] =│
│  "vendor,mydev-v2"│─match─│ { .compatible =             │
│ reg = <0x40000000 │       │   "vendor,mydev-v2" },      │
│       0x10000>   │        │ { .compatible =             │
│ interrupts = ... │        │   "vendor,mydev-v1" }, {}   │
│ status = "okay"  │        │ };                          │
└──────────────────┘        │                             │
                            │ .probe  = mydev_probe,      │
                            │ .remove = mydev_remove,     │
                            └─────────────────────────────┘
```

Reading fragment; callbacks, headers, target-version signatures, and error/lifetime handling are omitted.

```c
static const struct of_device_id mydev_of_match[] = {
    { .compatible = "vendor,mydevice-v2" },
    { .compatible = "vendor,mydevice-v1" },   /* older silicon */
    { }   /* sentinel — marks end of the table */
};
MODULE_DEVICE_TABLE(of, mydev_of_match);

static struct platform_driver mydev_driver = {
    .probe  = mydev_probe,
    .remove = mydev_remove,
    .driver = {
        .name           = "mydevice",
        .of_match_table = mydev_of_match,
    },
};
module_platform_driver(mydev_driver);
```

`module_platform_driver()` expands to `module_init` / `module_exit` wrappers that call `platform_driver_register()` and `platform_driver_unregister()`.

`MODULE_DEVICE_TABLE(of, mydev_of_match)` embeds the match table in the module binary. `udevd` uses this to **automatically load the module** when a matching Device Tree node is discovered — without requiring manual `modprobe`.

---

## Device Tree (DTS)

**Device Tree** is a **hardware description language** for embedded SoCs. The source format (`.dts`) is compiled by `dtc` into a binary blob (`.dtb`). The bootloader (U-Boot, EDK II) passes the DTB physical address to the kernel at boot (in a CPU register: `x0` on ARM64, `r2` on ARM32). The kernel parses it to discover all platform devices.

```
DTS → DTB → Kernel boot flow:

camera_isp.dts
    │
    │ dtc -I dts -O dtb
    ▼
camera_isp.dtb   (binary blob, ~10–500 KB)
    │
    │ bootloader loads DTB, passes PA in x0 register
    ▼
Linux kernel
    │
    │ of_platform_populate() scans DTB
    │ populates appropriate available nodes on supported buses
    ▼
platform_device "camera_isp@fe100000"
    │
    │ platform_bus match loop
    │ finds driver with compatible="vendor,cam-isp-v3"
    ▼
mydriver.probe(pdev) called
    │
    │ driver maps MMIO, registers IRQ, enables clocks
    ▼
Device ready for use
```

### Node Structure

Illustrative DTS fragment, not a standalone board description: the parent bus must supply compatible address/size cells and ranges; clock/reset/domain providers and binding macros must exist.

```dts
soc {
    camera_isp@fe100000 {
        compatible = "vendor,cam-isp-v3", "vendor,cam-isp";  /* primary + fallback */
        reg = <0 0xfe100000 0 0x10000>;      /* 64KB MMIO at PA 0xfe100000 */
        interrupts = <GIC_SPI 25 IRQ_TYPE_LEVEL_HIGH>;
        clocks = <&cru SCLK_ISP_CLK>, <&cru ACLK_ISP_CLK>;
        clock-names = "isp", "aclk";
        resets = <&cru SRST_ISP>;
        reset-names = "isp";
        dmas = <&dmac1 5>;
        dma-names = "dma";
        power-domains = <&power RK3588_PD_ISP>;
        status = "okay";
    };
};
```

The `compatible` property lists IDs from **most-specific to least-specific**. The driver's `of_match_table` is checked against all entries; the matching code selects a compatible match according to its scoring and bus rules; do not assume textual table order alone defines every result. This allows one driver to support multiple hardware revisions.

### Key DTS Properties

| Property | Parsed by | Kernel API |
|----------|-----------|-----------|
| `compatible` | Bus match logic | `of_match_table` lookup |
| `reg` | Resource subsystem | `platform_get_resource()`, `devm_ioremap_resource()` |
| `interrupts` | IRQ subsystem | `platform_get_irq()`, `devm_request_irq()` |
| `clocks` / `clock-names` | Clock framework | `devm_clk_get(dev, "isp")`, `clk_prepare_enable()` |
| `resets` / `reset-names` | Reset controller | `devm_reset_control_get(dev, "isp")` |
| `dmas` / `dma-names` | DMA engine | `dma_request_chan(dev, "dma")` |
| `power-domains` | genpd and runtime PM | Driver-specific PM setup; prefer checked `pm_runtime_resume_and_get()` where appropriate |
| `status` | Boot scan | `"okay"` enables; `"disabled"` skips probe |

> **Key Insight:** The `status = "okay"` / `"disabled"` property is how the DTS controls which hardware blocks are active without recompiling the kernel. To disable a peripheral, set `status = "disabled"` in a DTS overlay — the kernel skips its probe entirely. This is how Jetson carrier board customization works.

---

## Device Tree overlays

An overlay applies changes to a base tree. Bootloader support and a kernel's runtime overlay support are separate features. A configfs directory such as `/sys/kernel/config/device-tree/overlays` is **not a universal mainline userspace interface** merely because CONFIG_CONFIGFS_FS is enabled; some vendor kernels add their own interfaces. Do not paste a runtime removal command into an active camera pipeline without a device lifetime design.

Use the [native Device Tree lab](../../../Phase%204%20-%20Track%20B%20-%20Nvidia%20Jetson/1.%20Nvidia%20Jetson%20Platform/Orin-Nano-Kernel-Internals/labs/README.md) to compile a real base/overlay and inspect results with `dtc`, `fdtoverlay`, and `fdtget`. Its source addresses describe a synthetic device; passing the tools does not verify wiring, clocks, or a physical device driver.

## probe(): acquire, initialize, then publish

A successful probe is a transaction. Validate each resource and keep an unwind path for every operation that is not automatically reversed. An IRQ can fire as soon as it is registered, so its handler-visible state must already be initialized. Acquiring a clock/reset handle does not automatically enable the clock or deassert the reset.

**Protocol pseudocode, not a complete driver:**

```text
allocate private state; initialize locks, wait queues, removal flag
map the documented MMIO resource; check error pointer
get IRQ number; if negative, return that error (including deferred probe)
get named clocks and resets; check every error pointer
establish power/clock/reset state in device-required order
mask and clear stale device interrupt sources as specified
register IRQ with fully initialized handler state; check result
initialize queues and device-visible descriptor ordering
publish subsystem interface only after it can accept operations
unmask/start hardware when the device protocol permits
```

Use names matching the binding: a DTS with `clock-names = "isp", "aclk"` does not satisfy a driver requesting `"core"` by magic. `dev_err_probe()` helps report and propagate resource errors such as `-EPROBE_DEFER`; deferred dependencies are not necessarily permanent missing hardware.

On failure or removal, prevent new submissions, stop/abort hardware using its specified protocol, synchronize IRQs and callbacks, drain/cancel work, retire user-visible interfaces, and release resources only when no live reference can use them. Exact ordering is device-specific. `devm_clk_get_enabled()` can pair clock enable/disable on kernels that provide it, but does not replace reset or DMA quiescence.

## devm_* Managed Resources

The `devres` framework is what makes the `devm_*` pattern work. Each `devm_*` call **registers a cleanup action** with the device's resource list. When the device is unbound from its driver or when `probe()` returns an error, the framework walks the resource list **in reverse order** and calls each cleanup function.

`devres` framework automatically releases resources when the device is unbound from its driver or when `probe()` returns an error:

| Function | Resource released on detach |
|----------|-----------------------------|
| `devm_kzalloc()` | `kfree()` |
| `devm_ioremap_resource()` | `iounmap()` + `release_mem_region()` |
| `devm_request_irq()` | `free_irq()` |
| `devm_clk_get()` | `clk_put()` |
| `devm_reset_control_get()` | `reset_control_put()` |
| `devm_regulator_get()` | `regulator_put()` |
| `devm_gpiod_get()` | `gpiod_put()` |

Custom managed resources: `devm_add_action(dev, fn, data)` registers an arbitrary cleanup function.

Managed allocation reduces repetitive unwind code when its lifetime matches the device. Check cleanup-registration failures too: `devm_add_action_or_reset()` is useful when a failed action registration must immediately undo a preceding operation. Managed cleanup still depends on correct acquisition order and hardware/callback lifetime.

---


Managed cleanup is reverse acquisition order, not proof of safe asynchronous teardown. Open files, mappings, IRQ handlers, work items, and DMA may outlive a callback unless the driver explicitly stops or references them. Check target-kernel signatures: platform-driver remove callbacks changed across kernel versions.

## sysfs Attributes

Once a driver is probed, it can expose **hardware state to userspace** through the **sysfs** virtual filesystem. sysfs attributes appear as files under `/sys/bus/platform/devices/` and can be read or written with standard file operations.

Reading fragment: register the attribute group with the device, ensure the device is powered for MMIO, and synchronize removal/concurrent state as the hardware requires. This RO attribute is not writable.

```c
static ssize_t utilization_show(struct device *dev,
                                struct device_attribute *attr, char *buf)
{
    struct mydev_priv *priv = dev_get_drvdata(dev);
    return sysfs_emit(buf, "%u\n", readl(priv->base + REG_UTIL));
}
static DEVICE_ATTR_RO(utilization);

static struct attribute *mydev_attrs[] = {
    &dev_attr_utilization.attr,
    NULL,
};
ATTRIBUTE_GROUPS(mydev);
```

sysfs attributes appear at `/sys/bus/platform/devices/mydevice.0/utilization`. Userspace can read this RO attribute with `cat`; writable attributes require a separately implemented and validated store callback. Callbacks must handle concurrent access and removal under the device's locking and power-management protocol.

sysfs attributes are **ABI** — once exported to userspace, removing or renaming them **breaks userspace consumers**. Export only stable, meaningful values. Use `debugfs` (below) for internal debug state that may change between kernel versions.

With sysfs covered, the next piece is how the kernel notifies userspace when devices appear or disappear.

---

## udev: Userspace Device Management

The kernel emits **uevent** netlink messages on **device add/remove**. `udevd` receives them, matches `/etc/udev/rules.d/` rules, and acts:

- Creates `/dev/` device nodes with correct owner and permissions
- May participate in configured firmware fallback paths; the kernel firmware loader commonly reads firmware directly
- Runs `modprobe` for modules with matching `MODULE_DEVICE_TABLE` entries
- Executes custom scripts for device initialization

```
SUBSYSTEM=="video4linux", ATTR{name}=="EXACT_CAPTURE_NODE_NAME", SYMLINK+="camera0", GROUP="video", MODE="0660"
```

This is a rule template, not a tested sensor identity. Replace the name after inspecting the actual capture node with `udevadm info`; choose stable parent properties when names are ambiguous, and ensure the chosen group exists. A sensor subdevice and a capture node are different objects. Multiple matching devices must not compete for one ambiguous alias.

---

## debugfs

Drivers expose internal state for development diagnostics via **debugfs**:

```c
debugfs_create_u32("frame_count", 0444, priv->dbgfs_dir, &priv->frame_count);
debugfs_create_file("regs", 0444, priv->dbgfs_dir, priv, &mydev_regs_fops);
```

Accessible at `/sys/kernel/debug/mydevice/`. Not ABI-stable; may change between kernel versions. `media-ctl` and `v4l2-ctl` use the media controller and V4L2 ioctl interfaces (which do have stable ABI) for pipeline topology configuration.

debugfs is meant purely for development and debugging. Unlike sysfs, it carries no ABI stability guarantees. Production monitoring should read from sysfs; development register dumps and internal counters belong in debugfs.

---

## Summary

| Bus type | Self-describing? | DT needed? | Match mechanism | Example |
|----------|-----------------|-----------|----------------|---------|
| PCIe | Yes (config space) | No | `pci_device_id` vendor:device | NVIDIA GPU, Intel E810 NIC |
| USB | Yes (descriptor) | No | `usb_device_id` vid:pid | UVC camera, USB GNSS |
| Platform | Usually no generic self-enumeration | DT, ACPI, or another supported description | `of_match_table` compatible | UART, FPGA AXI peripheral |
| I2C | Address probing is not reliable identification | DT/ACPI/board description commonly | `of_match_table` + I2C address | IMX477, ICM-42688 IMU |
| SPI | Chip select is not device identification | DT/ACPI/board description commonly | `of_match_table` + SPI chip select | ADC, display controller |

### Conceptual Review

- **Why do platform devices need Device Tree when PCIe devices do not?** PCIe devices carry self-identification (vendor ID, device ID) in on-chip configuration space that the bus controller reads automatically. SoC peripherals are hardwired at fixed addresses with no self-identification capability. Device Tree provides the description that the hardware itself cannot.

- **What happens when a driver module is loaded and a matching DTS node exists?** The module's `module_init` calls `platform_driver_register()`. The platform bus immediately checks if any already-registered devices match the driver's `of_match_table`. If so, it calls `probe()` right away. Conversely, if the device is registered first, the bus calls `probe()` when the matching driver registers.

- **What is the purpose of the `compatible` property having multiple strings?** It creates a priority list from most-specific to least-specific. Compatible matching prefers more-specific entries according to the bus/OF matching rules. A new driver can match on `"vendor,mydevice-v2"` while an older, fallback driver matches on `"vendor,mydevice"`. This allows hardware revisions to use specialized drivers while maintaining backward compatibility.

- **Why is `devm_*` preferred over manual resource management?** Manual resource management requires a correct cleanup for every possible error path in `probe()`. Adding one new resource requires updating every error label. `devm_*` automates the cleanup: the framework unwinds all managed resources in reverse order whenever the device detaches, regardless of where in `probe()` the failure occurred.

- **What is a Device Tree overlay and when is it used?** An overlay is a compiled patch (`.dtbo`) that adds, modifies, or changes the tree according to the supported overlay semantics at boot time or runtime. It is used for modular hardware configurations like camera carrier boards on Jetson, where the same base board can have different sensor modules installed.

- **What is the difference between sysfs and debugfs for driver state?** sysfs attributes are ABI-stable: once exported, removing them breaks userspace tools and monitoring dashboards. They are for stable, meaningful values like device state and hardware counters. debugfs has no stability guarantee — it is for developer diagnostics, register dumps, and transient debugging data.

---

## Bring-up exercise

Use the native [Device Tree/module lab](../../../Phase%204%20-%20Track%20B%20-%20Nvidia%20Jetson/1.%20Nvidia%20Jetson%20Platform/Orin-Nano-Kernel-Internals/labs/README.md). Explain the difference between: DT syntax accepted, overlay applied, binding validated, module compiled, driver matched, probe succeeded, and hardware operation tested. These are seven distinct evidence boundaries.

For an actual board, build a resource table from the final tree and the driver's named requests. Include MMIO size, interrupt source, clocks, reset, power, pinmux, DMA addressing limits, and removal ownership. Overlays do not make a physically non-hotpluggable camera hot-swappable.

**Check:** Why initialize a wait queue before registering an IRQ? **Answer:** The handler may run immediately after registration and touch the queue. Registering first creates a race with uninitialized state.

**Check:** Does managed cleanup retire a mapped userspace buffer safely? **Answer:** Only if the lifetime protocol already prevents remaining users and device access. `devm_*` reverses allocations; it does not invent that protocol.
