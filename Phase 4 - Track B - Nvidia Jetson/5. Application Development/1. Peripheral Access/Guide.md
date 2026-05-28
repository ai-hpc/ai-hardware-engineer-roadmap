# Peripheral Access

<div class="course-identity auto-course" style="--course-accent: #7e22ce; --course-accent-rgb: 126, 34, 206;" markdown="1">
<div class="course-identity__icon">PA</div>
<div markdown="1">
<p class="course-identity__eyebrow">Deep Dive · Jetson Track</p>
<p class="course-identity__title">Specialized course identity for Peripheral Access.</p>
<p class="course-identity__meta">Artifact: Jetson integration demo · Measure: latency, memory, power, logs</p>
</div>
</div>


**Phase 4 — Track B — Module 5.1** · Application Development

> **Focus:** Access and control hardware peripherals on the **Jetson Orin Nano 8GB** from Linux userspace — GPIO, PWM, UART, SPI, I2C, CAN, USB, storage, and backlight. Every interface is shown with both sysfs/chardev and library approaches.

**Hub:** [5. Application Development](../Guide.md)

**Scope boundary:** this module is about **runtime access from Linux**. If you need to change **pinmux**, **device tree**, or **custom carrier board wiring**, use:

- [2. Custom Carrier Board Design and Bring-Up](../../2.%20Custom%20Carrier%20Board%20Design%20and%20Bring-Up/Guide.md)
- [3. L4T Customization](../../3.%20L4T%20Customization/Guide.md)

---


## 1. GPIO (Linux)

Jetson Orin Nano exposes GPIOs through the **gpiochip** character device interface (`/dev/gpiochipN`). The legacy sysfs interface (`/sys/class/gpio/`) is deprecated — use `libgpiod` instead.

### libgpiod tools

```bash
# List all GPIO chips
gpiodetect

# Show all lines on a chip
gpioinfo gpiochip0

# Read a GPIO input from a real line offset
gpioget gpiochip0 43

# Set a GPIO output high on a real line offset
gpioset gpiochip0 43=1

# Monitor GPIO events (rising/falling edge) on a real line offset
gpiomon gpiochip0 43
```

### What `gpiodetect` means on Jetson

On a Jetson Orin Nano dev kit you will typically see something like:

```text
gpiochip0 [tegra234-gpio] (164 lines)
gpiochip1 [tegra234-gpio-aon] (32 lines)
```

- `gpiochip0` is the main Tegra234 GPIO controller
- `gpiochip1` is the **always-on** GPIO controller used for low-power and wake-related signals
- the number in parentheses is how many line offsets exist on that controller

For most 40-pin header work, you will usually spend your time on `gpiochip0`.

### How to read `gpioinfo`

`gpioinfo gpiochip0` prints one row per line offset. Example:

```text
line  35:      "PG.00" "Force Recovery" input active-low [used]
line  43:      "PH.00"       unused   input  active-high
line 131:      "PZ.01"  "interrupt"   input  active-high [used]
```

Read each field like this:

| Field | Meaning |
|-------|---------|
| `line 43` | The **chip-relative line offset**. This is the number you pass to `gpioget`, `gpioset`, and `gpiomon`. |
| `"PH.00"` | The SoC pad or line name exposed by the driver/device tree. |
| `unused` / `"Force Recovery"` / `"interrupt"` | The current consumer. `unused` means no driver or userspace process owns it right now. |
| `input` / `output` | Current direction. |
| `active-high` / `active-low` | Logical polarity. |
| `[used]` | The line is already requested by the kernel or another process. Do not reuse it casually. |

### Runtime names you actually care about

For this guide, keep these names separate:

| Name | Example | Use it for |
|------|---------|------------|
| Header pin number | `Pin 7` | Physical wiring on the dev kit |
| Board signal name | `GPIO09` | Board pinout tables and schematics |
| SoC pad name | `PH.00` | Low-level identity of the signal |
| `gpiochip` line offset | `43` | `gpioget`, `gpioset`, `gpiomon` |

Short version:

- use **header pin numbers** when wiring
- use **board signal names** when reading the pinout
- use **SoC pad names** when reading low-level GPIO or BSP data
- use **line offsets** when talking to Linux through `libgpiod`

### What names like `PH.00` mean

Names like `PH.00`, `PG.00`, or `PZ.01` are **SoC pad / GPIO bank names**.

Read `PH.00` as:

- `P` = GPIO naming prefix
- `H` = bank or port
- `00` = bit number inside that bank

So `PH.00` is the **SoC-side identity** of the signal. It is not a header pin number and not a Linux line offset.

### Why `gpioget gpiochip0 <line>` failed in your shell

`<line>` in the docs is a **placeholder**, not something you type literally.

In `bash`, angle brackets mean shell redirection. So:

```bash
gpioget gpiochip0 <line>
```

is interpreted as "read stdin from a file named `line`", which is why it fails.

Replace the placeholder with a real line offset from `gpioinfo`, for example:

```bash
# Read line 43 on gpiochip0
gpioget gpiochip0 43

# Drive line 43 high
gpioset gpiochip0 43=1

# Monitor line 43 for edges
gpiomon gpiochip0 43
```

Also note:

- `gpioget` is for **reading**
- `gpioset` is for **driving outputs**
- `gpioget gpiochip0 <line>=1` is invalid because `gpioget` does not set values

### Practical workflow on Jetson

Use this sequence when you are trying to identify a GPIO safely:

```bash
# 1. See which controllers exist
gpiodetect

# 2. Inspect the main controller
gpioinfo gpiochip0

# 3. Pick an UNUSED line only after checking pinmux and board function
gpioget gpiochip0 43
```

Before driving a line with `gpioset`, verify all of these:

- the pin is really muxed as GPIO
- it is not marked `[used]`
- it is not a boot, recovery, regulator, wake, or other board-critical signal
- you know what external hardware is attached to it

### Finding a line by name

If the line names are populated, `gpiofind` is often easier than scrolling through the full list:

```bash
gpiofind "PH.00"
```

That returns the owning chip and line offset for that named line.

### Where pinmux and carrier-board work belongs

This guide stops at **runtime Linux access**.

If you need to answer questions like:

- "Which SoC pad should I route on a custom carrier?"
- "How do I edit the NVIDIA pinmux spreadsheet?"
- "How do I change device tree or BSP defaults?"
- "Why is this pin SPI on my board instead of GPIO?"

use these earlier modules instead:

- [2. Custom Carrier Board Design and Bring-Up](../../2.%20Custom%20Carrier%20Board%20Design%20and%20Bring-Up/Guide.md)
- [3. L4T Customization](../../3.%20L4T%20Customization/Guide.md)

### libgpiod in C

```c
#include <gpiod.h>

struct gpiod_chip *chip = gpiod_chip_open("/dev/gpiochip0");
struct gpiod_line *line = gpiod_chip_get_line(chip, 42);

/* Output */
gpiod_line_request_output(line, "my-app", 0);
gpiod_line_set_value(line, 1);

/* Input */
gpiod_line_request_input(line, "my-app");
int val = gpiod_line_get_value(line);

gpiod_chip_close(chip);
```

### libgpiod in Python

```python
import gpiod

chip = gpiod.Chip('gpiochip0')
line = chip.get_line(42)

# Output
line.request(consumer="my-app", type=gpiod.LINE_REQ_DIR_OUT)
line.set_value(1)

# Input
line.request(consumer="my-app", type=gpiod.LINE_REQ_DIR_IN)
val = line.get_value()
```

### Device tree GPIO configuration

This module assumes the pin is **already configured correctly** for GPIO use.

- On the developer kit, that often means using **Jetson-IO**
- On a custom carrier, that means the right **pinmux** and **device tree** are already in your BSP

References:

- [2. Custom Carrier Board Design and Bring-Up](../../2.%20Custom%20Carrier%20Board%20Design%20and%20Bring-Up/Guide.md)
- [3. L4T Customization](../../3.%20L4T%20Customization/Guide.md)

---

## 2. GPIO naming — alphanumeric to numeric assignment

Jetson uses several GPIO naming schemes at once. They are easy to confuse, and they are **not interchangeable**.

### Finding the mapping

```bash
# Show all lines on the main chip
gpioinfo gpiochip0

# Show all lines on the always-on chip
gpioinfo gpiochip1

# Find a line by SoC pad name, if the line has a name
gpiofind "PH.00"

# Kernel debug view
sudo cat /sys/kernel/debug/gpio
```

`gpioinfo | grep -i "GPIO"` is often not very useful on Jetson, because many lines are named like `PA.00`, `PH.00`, `PQ.05`, or have board-specific labels such as `Force Recovery`, not literal strings like `GPIO09`.

### The four names you will see

| Name style | Example | Where you see it | What it means |
|------------|---------|------------------|---------------|
| Board/header label | `GPIO09`, `GPIO15`, `GPIO22` | Carrier schematics, 40-pin header tables, Jetson pinout diagrams | A board-facing label for a header pin or connector signal |
| SoC pad / port.pin | `PH.00`, `PQ.05`, `PZ.01` | `gpioinfo`, pinmux docs, low-level bring-up notes | The Tegra GPIO bank and bit name |
| `gpiochip` line offset | `43`, `105`, `131` | `gpioinfo`, `gpioget`, `gpioset`, `gpiomon` | The number libgpiod uses inside one chip |
| Legacy sysfs/global GPIO number | `348`, `391` | Older Jetson docs, `/sys/class/gpio` workflows | Deprecated numbering model; avoid for new code |

### Short mental model

At runtime, the one you usually act on is:

```text
gpiochipN + line offset
```

Example:

```bash
gpioget gpiochip0 43
```

That means:

- controller: `gpiochip0`
- line offset on that controller: `43`

It does **not** mean:

- 40-pin header pin 43
- sysfs GPIO 43
- board label `GPIO43`

### What to trust when numbers disagree

For this runtime guide, prefer this order:

1. `gpioinfo` and `gpiofind` for the current Linux-visible line offsets
2. 40-pin header tables for physical pin location on the dev kit
3. board schematic for signal identity
4. old sysfs numbers only when reading legacy documentation

If you need to go deeper into **pinmux spreadsheet**, **carrier routing**, or **BSP integration**, jump to:

- [2. Custom Carrier Board Design and Bring-Up](../../2.%20Custom%20Carrier%20Board%20Design%20and%20Bring-Up/Guide.md)
- [3. L4T Customization](../../3.%20L4T%20Customization/Guide.md)

---

## 3. PWM (Linux)

Orin Nano exposes PWM channels via the Linux PWM subsystem.

### sysfs interface

```bash
# Export PWM channel (chip 0, channel 0)
echo 0 > /sys/class/pwm/pwmchip0/export

# Configure: 1 kHz, 50% duty cycle
echo 1000000 > /sys/class/pwm/pwmchip0/pwm0/period      # ns
echo 500000  > /sys/class/pwm/pwmchip0/pwm0/duty_cycle   # ns
echo 1       > /sys/class/pwm/pwmchip0/pwm0/enable
```

### Common uses on Jetson

| Use case | PWM channel | Notes |
|----------|------------|-------|
| **Fan control** | Dedicated fan PWM | Controlled by `jetson_clocks` or device tree `pwm-fan` node |
| **LED brightness** | GPIO-capable PWM pin | Requires pinmux set to PWM function |
| **Servo motor** | GPIO-capable PWM pin | 50 Hz, 1–2 ms pulse width |
| **Backlight** | Display backlight PWM | See [Backlight](#13-backlight-linux) |

---

## 4. ADC (Linux)

The Jetson Orin Nano SoM has **limited ADC** capability — no general-purpose ADC pins are exposed on the carrier connector. For analog sensing:

### External ADC options

| ADC | Interface | Resolution | Channels | Common use |
|-----|-----------|-----------|----------|-----------|
| **ADS1115** | I2C | 16-bit | 4 | Voltage, current, temperature sensors |
| **MCP3008** | SPI | 10-bit | 8 | General-purpose analog input |
| **INA226** | I2C | 16-bit | 2 (V + I) | Power monitoring |

### Reading external ADC via IIO subsystem

If your external ADC has a Linux IIO driver:

```bash
# List IIO devices
ls /sys/bus/iio/devices/

# Read raw value
cat /sys/bus/iio/devices/iio:device0/in_voltage0_raw

# Read scale
cat /sys/bus/iio/devices/iio:device0/in_voltage0_scale
```

Voltage = raw * scale (in millivolts).

---

## 5. UART (Linux)

Orin Nano exposes multiple UART ports. The debug console uses `ttyTCU0` (Tegra Combined UART). Application UARTs appear as `/dev/ttyTHS*`.

### Available UARTs

| Device | Use | Baud rate |
|--------|-----|-----------|
| `/dev/ttyTCU0` | Debug console (bootloader + kernel) | 115200 |
| `/dev/ttyTHS0` | Application UART 0 | Configurable |
| `/dev/ttyTHS1` | Application UART 1 | Configurable |

### Using UART from userspace

```bash
# Quick test with minicom
minicom -D /dev/ttyTHS0 -b 115200

# Or with stty + cat
stty -F /dev/ttyTHS0 115200 cs8 -cstopb -parenb
echo "hello" > /dev/ttyTHS0
cat /dev/ttyTHS0
```

### Python (pyserial)

```python
import serial

ser = serial.Serial('/dev/ttyTHS0', 115200, timeout=1)
ser.write(b'hello\n')
data = ser.readline()
ser.close()
```

---

## 6. SPI (Linux)

SPI buses appear as `/dev/spidevX.Y` (bus X, chip-select Y).

### Enable SPI in device tree

This module assumes SPI is already enabled.

- On the developer kit, use **Jetson-IO**
- On a custom carrier, use your **pinmux + device tree** configuration in the BSP

References:

- [2. Custom Carrier Board Design and Bring-Up](../../2.%20Custom%20Carrier%20Board%20Design%20and%20Bring-Up/Guide.md)
- [3. L4T Customization](../../3.%20L4T%20Customization/Guide.md)

### spidev test

```bash
# Loopback test (connect MOSI to MISO)
spidev_test -D /dev/spidev0.0 -s 1000000 -v
```

### Python (spidev)

```python
import spidev

spi = spidev.SpiDev()
spi.open(0, 0)           # bus 0, CS 0
spi.max_speed_hz = 1000000
response = spi.xfer2([0x01, 0x02, 0x03])
spi.close()
```

---

## 7. I2C (Linux)

I2C buses appear as `/dev/i2c-N`.

### Scanning for devices

```bash
# Scan all addresses on bus 1
sudo i2cdetect -y -r 1
```

### Reading/writing registers

```bash
# Read register 0x00 from device 0x50 on bus 1
sudo i2cget -y 1 0x50 0x00

# Write 0xFF to register 0x01
sudo i2cset -y 1 0x50 0x01 0xFF
```

### Python (smbus2)

```python
from smbus2 import SMBus

with SMBus(1) as bus:
    data = bus.read_byte_data(0x50, 0x00)
    bus.write_byte_data(0x50, 0x01, 0xFF)
```

### Common I2C devices on Jetson carriers

| Device | Address | Purpose |
|--------|---------|---------|
| **EEPROM** (carrier ID) | 0x50–0x57 | Board identification |
| **INA3221** | 0x40–0x43 | Power monitoring (VIN, 3.3V, 5V) |
| **IMU** (BMI088, ICM-42688) | 0x68–0x69 | Inertial measurement (robotics) |
| **RTC** (DS3231) | 0x68 | Real-time clock (battery-backed) |

---

## 8. CAN (Linux)

CAN bus on Jetson requires an external CAN transceiver on the carrier board (the SoC has CAN controllers but no integrated PHY).

### Setup

```bash
# Configure CAN interface (500 kbps)
sudo ip link set can0 type can bitrate 500000
sudo ip link set can0 up

# For CAN-FD
sudo ip link set can0 type can bitrate 500000 dbitrate 2000000 fd on
sudo ip link set can0 up
```

### Send and receive

```bash
# Send a frame
cansend can0 123#DEADBEEF

# Receive (dump all frames)
candump can0

# Generate test traffic
cangen can0
```

### Python (python-can)

```python
import can

bus = can.interface.Bus(channel='can0', interface='socketcan')

# Send
msg = can.Message(arbitration_id=0x123, data=[0xDE, 0xAD, 0xBE, 0xEF])
bus.send(msg)

# Receive
msg = bus.recv(timeout=1.0)
print(f"ID: {msg.arbitration_id:#x}, Data: {msg.data.hex()}")
```

---

## 9. USB host mode (Linux)

Jetson Orin Nano supports USB 3.2 Gen 2 (10 Gbps) and USB 2.0 host ports.

### Enumeration

```bash
# List connected USB devices
lsusb

# Show USB tree with speeds
lsusb -t

# Detailed device info
lsusb -v -d <vendor>:<product>
```

### Common USB peripherals

| Device type | Driver | Notes |
|------------|--------|-------|
| **USB camera** | UVC (`uvcvideo`) | Works out of the box, see [Multimedia](../4.%20Multimedia/Guide.md) |
| **USB serial** (FTDI, CP210x) | `ftdi_sio`, `cp210x` | Appears as `/dev/ttyUSB*` |
| **USB Ethernet** | `cdc_ether`, `r8152` | Appears as `eth*` or `usb*` |
| **USB storage** | `usb-storage` | Appears as `/dev/sd*` |
| **USB audio** | `snd-usb-audio` | ALSA device |

---

## 10. USB device mode (Linux)

The Orin Nano dev kit's USB-C port supports device (gadget) mode for flashing and development.

### USB gadget framework

```bash
# Load the USB gadget configfs
modprobe libcomposite

# Common gadgets:
# - g_ether: USB Ethernet gadget (device appears as network adapter on host)
# - g_serial: USB serial gadget (device appears as ttyACM on host)
# - g_mass_storage: USB mass storage gadget
```

### Typical use cases

| Gadget | Use case |
|--------|----------|
| **Ethernet (RNDIS/ECM)** | SSH into Jetson over USB without network cable |
| **Serial (ACM)** | Debug console over USB |
| **Mass storage** | Expose a partition or file as USB drive |

---

## 11. NVMe / PCIe storage (Linux)

Most Jetson Orin Nano deployments use NVMe SSD for the root filesystem (faster and more reliable than SD card).

### NVMe operations

```bash
# List NVMe devices
nvme list

# Check health
sudo nvme smart-log /dev/nvme0

# Benchmark
sudo fio --name=test --rw=randread --bs=4k --numjobs=4 \
    --size=1G --runtime=30 --time_based --filename=/dev/nvme0n1

# Check PCIe link speed
sudo lspci -vv | grep -A 20 "Non-Volatile"
```

### PCIe link verification

```bash
# Should show Gen3 x4 or Gen4 x4 depending on module
sudo lspci -vv | grep "LnkSta:"
```

If the link trains at a lower speed than expected, check trace routing (see [Module 2 — Carrier Board](../../2.%20Custom%20Carrier%20Board%20Design%20and%20Bring-Up/Guide.md)).

---

## 12. SD/MMC card (Linux)

SD card slot (if present on carrier) appears as `/dev/mmcblk*`.

```bash
# Check SD card info
sudo fdisk -l /dev/mmcblk1

# Mount
sudo mount /dev/mmcblk1p1 /mnt

# Check speed class
cat /sys/block/mmcblk1/device/speed_class
```

**Production note:** SD cards have limited write endurance. Use NVMe for root filesystem; reserve SD for optional data logging or field updates only.

---

## 13. Backlight (Linux)

If your carrier has an LCD with PWM-controlled backlight:

```bash
# List backlight devices
ls /sys/class/backlight/

# Get current brightness
cat /sys/class/backlight/<device>/brightness

# Get max brightness
cat /sys/class/backlight/<device>/max_brightness

# Set brightness (0 = off, max = full)
echo 128 > /sys/class/backlight/<device>/brightness
```

Configure the backlight PWM channel in the device tree for your display panel.

For custom panel timing, PWM routing, and BSP integration, see:

- [2. Custom Carrier Board Design and Bring-Up](../../2.%20Custom%20Carrier%20Board%20Design%20and%20Bring-Up/Guide.md)
- [3. L4T Customization](../../3.%20L4T%20Customization/Guide.md)

---

## 14. Projects

- **Sensor dashboard:** Read an I2C temperature sensor (e.g., TMP102) and a SPI ADC (e.g., MCP3008 for analog input), display values on a UART terminal at 1 Hz.
- **CAN bus monitor:** Build a CAN bus sniffer that logs all frames to a file with timestamps. Add filtering by arbitration ID.
- **GPIO interrupt counter:** Use `gpiomon` or libgpiod event monitoring to count rising edges on an input pin. Measure maximum event rate.
- **USB gadget network:** Configure the Jetson as a USB Ethernet gadget so a host laptop can SSH into it over a single USB-C cable.

---

## 15. Resources

| Resource | Description |
|----------|-------------|
| **Jetson Orin Nano Developer Kit User Guide** | Pin header mapping, UART/SPI/I2C bus assignments |
| **libgpiod documentation** | Character device GPIO API (replaces sysfs) |
| **Linux kernel SPI/I2C docs** | `Documentation/spi/` and `Documentation/i2c/` in kernel source |
| **SocketCAN documentation** | Linux CAN subsystem, `ip link`, `candump`, `cansend` |
| **NVMe CLI** | `nvme-cli` package for NVMe management and diagnostics |
| [2. Custom Carrier Board](../../2.%20Custom%20Carrier%20Board%20Design%20and%20Bring-Up/Guide.md) | Connector routing, electrical design, pinmux planning |
| [3. L4T Customization](../../3.%20L4T%20Customization/Guide.md) | Device tree, BSP integration, flash and production image workflows |
