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
    │ creates platform_device for each node with status="okay"
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

The `compatible` property lists IDs from **most-specific to least-specific**. The driver's `of_match_table` is checked against all entries; the **first match wins**. This allows one driver to support multiple hardware revisions.

### Key DTS Properties

| Property | Parsed by | Kernel API |
|----------|-----------|-----------|
| `compatible` | Bus match logic | `of_match_table` lookup |
| `reg` | Resource subsystem | `platform_get_resource()`, `devm_ioremap_resource()` |
| `interrupts` | IRQ subsystem | `platform_get_irq()`, `devm_request_irq()` |
| `clocks` / `clock-names` | Clock framework | `devm_clk_get(dev, "isp")`, `clk_prepare_enable()` |
| `resets` / `reset-names` | Reset controller | `devm_reset_control_get(dev, "isp")` |
| `dmas` / `dma-names` | DMA engine | `dma_request_chan(dev, "dma")` |
| `power-domains` | genpd | Automatic on `pm_runtime_get()` |
| `status` | Boot scan | `"okay"` enables; `"disabled"` skips probe |

> **Key Insight:** The `status = "okay"` / `"disabled"` property is how the DTS controls which hardware blocks are active without recompiling the kernel. To disable a peripheral, set `status = "disabled"` in a DTS overlay — the kernel skips its probe entirely. This is how Jetson carrier board customization works.

---

## Device Tree Overlays (DTBO)

The base DTB describes the SoC itself. For boards that support multiple hardware configurations — different camera sensors, optional peripherals, add-on modules — rebuilding the full DTB for each variant is impractical. **Overlays** solve this: they are incremental patches applied on top of the base DTB.

Overlays are **incremental additions** to the base DTB, applied at runtime (via `/sys/kernel/config/device-tree/overlays/`) or by the bootloader. They add, modify, or delete nodes **without recompiling the full DTB**.

```bash
# Compile overlay source to binary
dtc -I dts -O dtb -o camera_imx477.dtbo camera_imx477.dts

# Jetson: reference in extlinux.conf
FDT_OVERLAYS /boot/camera_imx477.dtbo

# Runtime apply (if ConfigFS enabled)
mkdir /sys/kernel/config/device-tree/overlays/camera
cp camera_imx477.dtbo /sys/kernel/config/device-tree/overlays/camera/dtbo
echo 1 > /sys/kernel/config/device-tree/overlays/camera/status
```

Used in Jetson Xavier/Orin for camera carrier board (sensor module) configuration and in Raspberry Pi for HAT descriptor overlays.

---

## probe() Function: Resource Setup Pattern

The `probe()` function is the driver's **initialization entry point**. Its job is to claim all hardware resources, initialize the device, and register it with any higher-level subsystems (V4L2, IIO, etc.). The key pattern is using `devm_*` (**managed resource**) functions for every allocation:

```c
static int mydev_probe(struct platform_device *pdev)
{
    struct device *dev = &pdev->dev;
    struct mydev_priv *priv;
    struct resource *res;
    int irq, ret;

    /* Allocate driver private state — freed automatically on probe failure or driver unbind */
    priv = devm_kzalloc(dev, sizeof(*priv), GFP_KERNEL);
    if (!priv)
        return -ENOMEM;

    /* Map MMIO registers from DTS 'reg' property — unmapped automatically on cleanup */
    res = platform_get_resource(pdev, IORESOURCE_MEM, 0);
    priv->base = devm_ioremap_resource(dev, res);
    if (IS_ERR(priv->base))
        return PTR_ERR(priv->base);   /* devm unwinds previous allocations */

    /* Register interrupt handler from DTS 'interrupts' property */
    irq = platform_get_irq(pdev, 0);
    ret = devm_request_irq(dev, irq, mydev_isr,
                           IRQF_SHARED, "mydevice", priv);
    if (ret)
        return ret;

    /* Acquire clock and reset from DTS 'clocks' and 'resets' properties */
    priv->clk = devm_clk_get(dev, "core");
    priv->rst = devm_reset_control_get(dev, "rst");

    platform_set_drvdata(pdev, priv);   /* store priv pointer for later retrieval */
    return 0;
}
```

If any `devm_*` call fails and the function returns a negative error code, all previously acquired managed resources are **automatically released in reverse order**. No `goto err_cleanup` labels needed.

The step-by-step sequence of a successful probe:

1. **Allocate private state** (`devm_kzalloc`) — driver-specific context structure that persists for the device lifetime.
2. **Map MMIO** (`devm_ioremap_resource`) — makes hardware registers accessible via `readl`/`writel`.
3. **Register IRQ** (`devm_request_irq`) — connects the hardware interrupt line to the driver's ISR function.
4. **Acquire clocks** (`devm_clk_get`) — gets a handle to the clock that drives the peripheral.
5. **Acquire reset** (`devm_reset_control_get`) — gets control of the hardware reset line for initialization.
6. **Store private data** (`platform_set_drvdata`) — makes the private structure retrievable in other callbacks (ISR, fops, sysfs).
7. **Return 0** — signals success to the bus; device is now active.

> **Common Pitfall:** Calling `clk_prepare_enable()` in `probe()` but forgetting to call `clk_disable_unprepare()` in `remove()` leaves the clock running after the driver unloads. On embedded systems, this wastes power and may prevent the SoC from entering low-power sleep states. Use `devm_clk_get_enabled()` (kernel 5.19+) which disables the clock automatically on device detach.

---

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

> **Key Insight:** The `devres` framework transforms the driver error handling problem from "undo everything correctly on every possible failure path" to "just return the error code." The framework handles the unwinding automatically. This eliminates an entire class of resource leak bugs that plagued pre-devres drivers, especially on iterative FPGA bring-up where `modprobe`/`rmmod` cycles happen dozens of times.

---

## sysfs Attributes

Once a driver is probed, it can expose **hardware state to userspace** through the **sysfs** virtual filesystem. sysfs attributes appear as files under `/sys/bus/platform/devices/` and can be read or written with standard file operations.

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

sysfs attributes appear at `/sys/bus/platform/devices/mydevice.0/utilization`. Userspace reads with `cat`; write with `echo`. Must be re-entrant; protect shared state with a mutex or spinlock.

sysfs attributes are **ABI** — once exported to userspace, removing or renaming them **breaks userspace consumers**. Export only stable, meaningful values. Use `debugfs` (below) for internal debug state that may change between kernel versions.

With sysfs covered, the next piece is how the kernel notifies userspace when devices appear or disappear.

---

## udev: Userspace Device Management

The kernel emits **uevent** netlink messages on **device add/remove**. `udevd` receives them, matches `/etc/udev/rules.d/` rules, and acts:

- Creates `/dev/` device nodes with correct owner and permissions
- Loads firmware via `request_firmware()` mechanism
- Runs `modprobe` for modules with matching `MODULE_DEVICE_TABLE` entries
- Executes custom scripts for device initialization

```
SUBSYSTEM=="video4linux", ATTRS{name}=="IMX477*", SYMLINK+="camera0", MODE="0666"
```

This rule creates a stable symlink `/dev/camera0` pointing to the actual `/dev/video0` node whenever an IMX477 sensor is detected. This insulates application code from device number changes when multiple cameras are present.

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
| Platform | No | Yes (DTS/ACPI) | `of_match_table` compatible | UART, FPGA AXI peripheral |
| I2C | Partial (address) | Yes | `of_match_table` + I2C address | IMX477, ICM-42688 IMU |
| SPI | Partial (CS) | Yes | `of_match_table` + SPI chip select | ADC, display controller |

### Conceptual Review

- **Why do platform devices need Device Tree when PCIe devices do not?** PCIe devices carry self-identification (vendor ID, device ID) in on-chip configuration space that the bus controller reads automatically. SoC peripherals are hardwired at fixed addresses with no self-identification capability. Device Tree provides the description that the hardware itself cannot.

- **What happens when a driver module is loaded and a matching DTS node exists?** The module's `module_init` calls `platform_driver_register()`. The platform bus immediately checks if any already-registered devices match the driver's `of_match_table`. If so, it calls `probe()` right away. Conversely, if the device is registered first, the bus calls `probe()` when the matching driver registers.

- **What is the purpose of the `compatible` property having multiple strings?** It creates a priority list from most-specific to least-specific. The kernel tries each string in order. A new driver can match on `"vendor,mydevice-v2"` while an older, fallback driver matches on `"vendor,mydevice"`. This allows hardware revisions to use specialized drivers while maintaining backward compatibility.

- **Why is `devm_*` preferred over manual resource management?** Manual resource management requires a correct cleanup for every possible error path in `probe()`. Adding one new resource requires updating every error label. `devm_*` automates the cleanup: the framework unwinds all managed resources in reverse order whenever the device detaches, regardless of where in `probe()` the failure occurred.

- **What is a Device Tree overlay and when is it used?** An overlay is a compiled patch (`.dtbo`) that adds, modifies, or removes nodes from the base DTB at boot time or runtime. It is used for modular hardware configurations like camera carrier boards on Jetson, where the same base board can have different sensor modules installed.

- **What is the difference between sysfs and debugfs for driver state?** sysfs attributes are ABI-stable: once exported, removing them breaks userspace tools and monitoring dashboards. They are for stable, meaningful values like device state and hardware counters. debugfs has no stability guarantee — it is for developer diagnostics, register dumps, and transient debugging data.

---

## AI Hardware Connection

- DTS nodes with correct `compatible` strings link Jetson IMX477 and AR0234 camera sensors to NVIDIA's V4L2 sensor drivers; `reg` specifies the I2C address and `clocks` provisions the NVCSI clock tree
- NVDLA on Jetson is a platform device with DTS entries for MMIO base, IRQ lines, clock domains, and DMA channels; the kernel driver maps all resources via `devm_ioremap_resource` and `dma_request_chan`
- Custom Xilinx Zynq AXI peripherals for sensor fusion or pre-processing are described as platform devices in the DTS; `reg` specifies AXI slave base address and size; the driver accesses registers with `ioread32` / `iowrite32`
- DTBO overlays enable hot-swappable camera module configuration on Jetson development kits without reflashing the base system image, accelerating sensor bring-up iteration
- sysfs `DEVICE_ATTR_RO` attributes expose AI accelerator utilization counters and error registers to Prometheus node exporters and health monitoring daemons without requiring privileged ioctls
- `devm_*` managed resources prevent resource leaks in FPGA driver `probe()` error paths, which are exercised at every module reload during iterative hardware bring-up cycles
