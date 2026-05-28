# Jetson Orin Nano 8GB -- GPIO, SPI, I2C, CAN, and Peripheral I/O Deep-Dive Guide

<div class="course-identity auto-course" style="--course-accent: #475569; --course-accent-rgb: 71, 85, 105;" markdown="1">
<div class="course-identity__icon">JON8</div>
<div markdown="1">
<p class="course-identity__eyebrow">Deep Dive · Jetson Track</p>
<p class="course-identity__title">Specialized course identity for Jetson Orin Nano 8GB -- GPIO, SPI, I2C, CAN, and Peripheral I/O Deep-Dive Guide.</p>
<p class="course-identity__meta">Artifact: Jetson integration demo · Measure: latency, memory, power, logs</p>
</div>
</div>


> **Target hardware:** Jetson Orin Nano 8GB Developer Kit (T234 SoC, JetPack 6.x, L4T 36.x)
> **40-pin header:** Raspberry-Pi-compatible expansion header, 3.3 V logic levels

---

## 1. Introduction

### 1.1 Why Peripheral I/O Matters for Edge AI

A Jetson Orin Nano sitting in isolation is just a powerful SBC. The moment you wire it to
real-world sensors, actuators, and communication buses, it becomes an edge AI system.
Peripheral interfaces are the nervous system of any embedded AI deployment:

- **Sensor fusion** -- IMUs, LiDAR triggers, environmental sensors, and cameras must
  feed data to inference pipelines through I2C, SPI, UART, and GPIO.
- **Actuator control** -- Inference results must drive motors, relays, and indicators
  through PWM and GPIO.
- **Vehicle / industrial networking** -- CAN bus connects the Jetson to ECUs, PLCs,
  and other nodes on a shared bus.
- **Low-latency triggers** -- GPIO interrupts can gate when inference starts or stops,
  enabling event-driven architectures that save power and reduce latency.

Without mastering these interfaces you will hit a wall the moment you move from a
demo running on a recorded dataset to a live, closed-loop system.

### 1.2 The 40-Pin Expansion Header

The Orin Nano Developer Kit exposes a 40-pin header that is pin-compatible with the
Raspberry Pi header layout. This header is directly connected to the T234 SoC's I/O
pads (through level shifters where needed) and provides:

- Up to 26 GPIO lines (depending on pinmux configuration)
- 2 x SPI buses (SPI0, SPI1) with 2 chip-selects each
- 2 x I2C buses (I2C bus 1 and bus 7 on the 40-pin header)
- 2 x UART interfaces (UART1, UART3)
- 2 x PWM outputs
- 1 x CAN bus (CAN0, directly from T234 mttcan controller, but requires external transceiver)
- 3.3 V and 5 V power rails
- Ground pins

```
Physical header layout (top view, board oriented with the header on the
right side, pin 1 at top-left, Ethernet jack facing away from you):

        +3.3V [ 1][ 2] +5V
   I2C1_SDA   [ 3][ 4] +5V
   I2C1_SCL   [ 5][ 6] GND
   GPIO09     [ 7][ 8] UART1_TXD
        GND   [ 9][10] UART1_RXD
   GPIO11     [11][12] I2S0_SCLK
   GPIO13     [13][14] GND
   GPIO15     [15][16] GPIO16
       +3.3V  [17][18] GPIO18
  SPI1_MOSI   [19][20] GND
  SPI1_MISO   [21][22] GPIO22
  SPI1_SCLK   [23][24] SPI1_CS0
        GND   [25][26] SPI1_CS1
  I2C7_SDA    [27][28] I2C7_SCL
   GPIO_CAN0  [29][30] GND
   GPIO_CAN1  [31][32] GPIO32
   GPIO33     [33][34] GND
  I2S0_LRCK   [35][36] UART3_CTS
   GPIO37     [37][38] I2S0_DIN
        GND   [39][40] I2S0_DOUT
```

### 1.3 Electrical Characteristics

| Parameter              | Value                                   |
|------------------------|-----------------------------------------|
| Logic level            | 3.3 V LVCMOS                            |
| Max GPIO source/sink   | 1 mA per pin (typical T234 limit)       |
| 3.3 V rail max current | 1.0 A (shared across all 3.3 V pins)    |
| 5 V rail max current   | 2.0 A (derived from main power supply)  |
| Pull-up/pull-down      | Configurable per pin via pinmux DT      |
| ESD protection         | On-SoC clamp diodes                     |

**Warning:** The T234 GPIO pins can only source/sink very small currents. Always use
a transistor, MOSFET, or buffer (e.g., 74LVC245) when driving LEDs or any load above
1 mA. Driving loads directly will damage the SoC.

### 1.4 Prerequisites

Before working through this guide, ensure you have:

```bash
# Verify JetPack / L4T version
cat /etc/nv_tegra_release
# Example output: # R36 (release), REVISION: 4.0, ...

# Verify kernel version
uname -r
# Example: 5.15.136-tegra

# Install essential tools
sudo apt-get update
sudo apt-get install -y \
    python3-pip i2c-tools spi-tools can-utils \
    device-tree-compiler minicom picocom \
    libgpiod-dev gpiod python3-libgpiod

# Install Python libraries
pip3 install Jetson.GPIO smbus2 spidev pyserial
```

---

## 2. 40-Pin Header Pinout

### 2.1 Complete Pin Mapping Table

The following table maps every physical pin on the 40-pin header to its default
function, alternate functions, T234 SoC pad name, and Linux GPIO number. GPIO
numbers listed are the `sysfs` numbers used by the kernel (these shifted in L4T 36.x
compared to earlier releases; always verify with `cat /sys/kernel/debug/gpio`).

| Pin | Default Function | Alt Functions                  | T234 Pad Name       | Sysfs GPIO | Voltage |
|-----|------------------|--------------------------------|----------------------|------------|---------|
|  1  | +3.3V            | Power                          | --                   | --         | 3.3V    |
|  2  | +5V              | Power                          | --                   | --         | 5V      |
|  3  | I2C1_SDA         | GPIO, I2C Gen1 SDA             | GP60_I2C1_SDA        | --         | 3.3V    |
|  4  | +5V              | Power                          | --                   | --         | 5V      |
|  5  | I2C1_SCL         | GPIO, I2C Gen1 SCL             | GP59_I2C1_SCL        | --         | 3.3V    |
|  6  | GND              | Ground                         | --                   | --         | --      |
|  7  | GPIO09           | AUD_MCLK, GPIO                 | GP167                | 348        | 3.3V    |
|  8  | UART1_TXD        | GPIO, UART1 TX                 | GP112_UART1_TX       | --         | 3.3V    |
|  9  | GND              | Ground                         | --                   | --         | --      |
| 10  | UART1_RXD        | GPIO, UART1 RX                 | GP113_UART1_RX       | --         | 3.3V    |
| 11  | GPIO11           | UART1_RTS, GPIO                | GP130                | 316        | 3.3V    |
| 12  | I2S0_SCLK        | GPIO, I2S0 Bit Clock           | GP72_I2S0_SCLK       | 464        | 3.3V    |
| 13  | GPIO13           | SPI1_SCK (alt), GPIO           | GP131                | 317        | 3.3V    |
| 14  | GND              | Ground                         | --                   | --         | --      |
| 15  | GPIO15           | PWM3 (alt), GPIO               | GP68                 | 350        | 3.3V    |
| 16  | GPIO16           | SPI0_CS1 (alt), GPIO           | GP161                | 389        | 3.3V    |
| 17  | +3.3V            | Power                          | --                   | --         | 3.3V    |
| 18  | GPIO18           | SPI0_CS0 (alt), GPIO           | GP162                | 390        | 3.3V    |
| 19  | SPI1_MOSI        | GPIO, SPI1 MOSI                | GP136_SPI1_MOSI      | --         | 3.3V    |
| 20  | GND              | Ground                         | --                   | --         | --      |
| 21  | SPI1_MISO        | GPIO, SPI1 MISO                | GP137_SPI1_MISO      | --         | 3.3V    |
| 22  | GPIO22           | GPIO                           | GP163                | 391        | 3.3V    |
| 23  | SPI1_SCLK        | GPIO, SPI1 SCK                 | GP138_SPI1_SCK       | --         | 3.3V    |
| 24  | SPI1_CS0         | GPIO, SPI1 Chip Select 0       | GP139_SPI1_CS0       | --         | 3.3V    |
| 25  | GND              | Ground                         | --                   | --         | --      |
| 26  | SPI1_CS1         | GPIO, SPI1 Chip Select 1       | GP140_SPI1_CS1       | --         | 3.3V    |
| 27  | I2C7_SDA         | GPIO, I2C Gen7 SDA             | GP185_I2C7_SDA       | --         | 3.3V    |
| 28  | I2C7_SCL         | GPIO, I2C Gen7 SCL             | GP186_I2C7_SCL       | --         | 3.3V    |
| 29  | CAN0_DIN         | GPIO, CAN0 RX                  | GP169_CAN0_DIN       | 320        | 3.3V    |
| 30  | GND              | Ground                         | --                   | --         | --      |
| 31  | CAN0_DOUT        | GPIO, CAN0 TX                  | GP170_CAN0_DOUT      | 321        | 3.3V    |
| 32  | GPIO32           | PWM1 (alt), GPIO               | GP164                | 392        | 3.3V    |
| 33  | GPIO33           | PWM5 (alt), GPIO               | GP165                | 393        | 3.3V    |
| 34  | GND              | Ground                         | --                   | --         | --      |
| 35  | I2S0_LRCK        | GPIO, I2S0 Word Select (LRCK)  | GP73_I2S0_LRCK       | 465        | 3.3V    |
| 36  | UART3_CTS        | GPIO, UART3 CTS                | GP171_UART3_CTS      | --         | 3.3V    |
| 37  | GPIO37           | GPIO                           | GP166                | 394        | 3.3V    |
| 38  | I2S0_DIN         | GPIO, I2S0 Data In             | GP74_I2S0_DIN        | 466        | 3.3V    |
| 39  | GND              | Ground                         | --                   | --         | --      |
| 40  | I2S0_DOUT        | GPIO, I2S0 Data Out            | GP75_I2S0_DOUT       | 467        | 3.3V    |

### 2.2 Quick-Reference Pinout Diagram

```
                      Jetson Orin Nano 40-Pin Header
                      (Top view, Pin 1 = top-left)

              ODD (left)                    EVEN (right)
          +-----------+                  +-----------+
  Pin  1  | +3.3V     |                  |  +5V      |  Pin  2
  Pin  3  | I2C1_SDA  |                  |  +5V      |  Pin  4
  Pin  5  | I2C1_SCL  |                  |  GND      |  Pin  6
  Pin  7  | GPIO09    |                  |  UART1_TX |  Pin  8
  Pin  9  | GND       |                  |  UART1_RX |  Pin 10
  Pin 11  | GPIO11    |                  |  I2S_SCLK |  Pin 12
  Pin 13  | GPIO13    |                  |  GND      |  Pin 14
  Pin 15  | GPIO15    |                  |  GPIO16   |  Pin 16
  Pin 17  | +3.3V     |                  |  GPIO18   |  Pin 18
  Pin 19  | SPI1_MOSI |                  |  GND      |  Pin 20
  Pin 21  | SPI1_MISO |                  |  GPIO22   |  Pin 22
  Pin 23  | SPI1_SCLK |                  |  SPI1_CS0 |  Pin 24
  Pin 25  | GND       |                  |  SPI1_CS1 |  Pin 26
  Pin 27  | I2C7_SDA  |                  |  I2C7_SCL |  Pin 28
  Pin 29  | CAN0_DIN  |                  |  GND      |  Pin 30
  Pin 31  | CAN0_DOUT |                  |  GPIO32   |  Pin 32
  Pin 33  | GPIO33    |                  |  GND      |  Pin 34
  Pin 35  | I2S_LRCK  |                  |  UART3_CT |  Pin 36
  Pin 37  | GPIO37    |                  |  I2S_DIN  |  Pin 38
  Pin 39  | GND       |                  |  I2S_DOUT |  Pin 40
          +-----------+                  +-----------+
```

### 2.3 Power Pins

| Pin(s) | Rail  | Notes                                                      |
|--------|-------|------------------------------------------------------------|
| 1, 17  | 3.3V  | Regulated from main supply. Max combined draw: ~1A.        |
| 2, 4   | 5V    | Directly from the barrel jack / USB-C supply. Max ~2A.     |
| 6,9,14,20,25,30,34,39 | GND | All internally connected. Use the nearest GND pin. |

---

## 3. Pin Multiplexing (Pinmux)

### 3.1 What is Pinmux?

Each physical pin on the 40-pin header connects to a T234 SoC pad that can serve
multiple functions. The pin multiplexer (pinmux) determines which function is active.
At boot time, the bootloader (UEFI/CBoot) configures each pad according to the
device tree. You can reconfigure pins in two ways:

1. **Jetson-IO tool** (interactive, recommended for beginners)
2. **Custom device tree overlays** (full control, recommended for production)

### 3.2 Using the Jetson-IO Tool

Jetson-IO is NVIDIA's interactive tool for configuring the 40-pin header. It generates
device tree overlay files and installs them so they take effect on next boot.

```bash
# Launch Jetson-IO (requires a display or SSH with X-forwarding)
sudo /opt/nvidia/jetson-io/jetson-io.py
```

The tool presents a text-based menu:

```
 ==================== Jetson Expansion Header Tool ====================

   Available Configurations:

   1. Configure Jetson 40-pin Header
   2. Configure Jetson CSI Connector
   3. Exit

 ======================================================================
```

Selecting option 1 shows all available pin functions:

```
 Configure for compatible hardware:
   1. Adafruit SparkFun Qwiic Add-on
   2. CAN Bus
   3. PWM (2 channels)
   4. SPI1 (1 device)
   5. SPI1 (2 devices)

 Or configure individual pins:
   6. Configure header pins manually
   7. Back
```

Example: Enable SPI1 with one chip-select and CAN bus:

```bash
# Step-by-step in Jetson-IO:
# 1. Select "Configure Jetson 40-pin Header"
# 2. Select "SPI1 (1 device)"
# 3. Back, then also select "CAN Bus"
# 4. Select "Save and Exit"
# 5. Reboot

sudo reboot
```

After reboot, verify the configuration:

```bash
# Check that spidev device appeared
ls /dev/spidev*
# Expected: /dev/spidev0.0

# Check that CAN interface appeared
ip link show can0
# Or: ifconfig -a | grep can
```

### 3.3 Pinmux Spreadsheet

NVIDIA provides an Excel-based Pinmux Spreadsheet for each Jetson platform. For
the Orin Nano:

1. Download from: https://developer.nvidia.com/jetson-orin-nano-pinmux
2. Open in Excel or LibreOffice Calc.
3. For each pin, select the desired function from the dropdown.
4. The spreadsheet generates a `.dtsi` file you can compile into an overlay.

```bash
# The spreadsheet generates a file like:
# tegra234-mb1-bct-pinmux-p3768-0000-a0.dtsi

# Compile it (if needed, alongside a wrapper overlay):
dtc -I dts -O dtb -o custom-pinmux.dtbo custom-pinmux-overlay.dts
```

### 3.4 Verifying Current Pinmux State

```bash
# View all pin configurations from the kernel's perspective
sudo cat /sys/kernel/debug/tegra_pinctrl_reg

# Or use the GPIO debug file
sudo cat /sys/kernel/debug/gpio

# Example output (truncated):
# gpiochip0: GPIOs 300-511, parent: platform/2200000.gpio, tegra234-gpio:
#  gpio-316 (                    |gpio11             ) in  lo
#  gpio-317 (                    |gpio13             ) in  lo
#  gpio-348 (                    |gpio09             ) out lo
```

### 3.5 Pinmux Device Tree Overlay Structure

```dts
// Example: enable SPI1 on the 40-pin header
/dts-v1/;
/plugin/;

/ {
    overlay-name = "Jetson 40-pin Header SPI1";
    compatible = "nvidia,p3768-0000+p3767-0003";

    fragment@0 {
        target = <&pinmux>;
        __overlay__ {
            pinctrl-names = "default";
            pinctrl-0 = <&jetson_io_pinmux>;

            jetson_io_pinmux: exp-header-pinmux {
                /* Pin 19: SPI1_MOSI */
                hdr40-pin19 {
                    nvidia,pins = "spi1_mosi_pz4";
                    nvidia,function = "spi1";
                    nvidia,pull = <TEGRA_PIN_PULL_DOWN>;
                    nvidia,tristate = <TEGRA_PIN_DISABLE>;
                    nvidia,enable-input = <TEGRA_PIN_DISABLE>;
                };
                /* Pin 21: SPI1_MISO */
                hdr40-pin21 {
                    nvidia,pins = "spi1_miso_pz5";
                    nvidia,function = "spi1";
                    nvidia,pull = <TEGRA_PIN_PULL_DOWN>;
                    nvidia,tristate = <TEGRA_PIN_DISABLE>;
                    nvidia,enable-input = <TEGRA_PIN_ENABLE>;
                };
                /* Pin 23: SPI1_SCLK */
                hdr40-pin23 {
                    nvidia,pins = "spi1_sck_pz3";
                    nvidia,function = "spi1";
                    nvidia,pull = <TEGRA_PIN_PULL_DOWN>;
                    nvidia,tristate = <TEGRA_PIN_DISABLE>;
                    nvidia,enable-input = <TEGRA_PIN_ENABLE>;
                };
                /* Pin 24: SPI1_CS0 */
                hdr40-pin24 {
                    nvidia,pins = "spi1_cs0_pz6";
                    nvidia,function = "spi1";
                    nvidia,pull = <TEGRA_PIN_PULL_UP>;
                    nvidia,tristate = <TEGRA_PIN_DISABLE>;
                    nvidia,enable-input = <TEGRA_PIN_ENABLE>;
                };
            };
        };
    };

    fragment@1 {
        target = <&spi1>;
        __overlay__ {
            status = "okay";
            spi@0 {
                compatible = "spidev";
                reg = <0>;  /* CS0 */
                spi-max-frequency = <10000000>;
            };
        };
    };
};
```

Compile and install:

```bash
dtc -I dts -O dtb -o /boot/spi1-overlay.dtbo spi1-overlay.dts
# Then reference it in extlinux.conf or via Jetson-IO's overlay directory
sudo cp /boot/spi1-overlay.dtbo /boot/
# Add to /boot/extlinux/extlinux.conf:
#   OVERLAYS /boot/spi1-overlay.dtbo
sudo reboot
```

---

## 4. GPIO Programming

### 4.1 GPIO Numbering on the Orin Nano

The T234 SoC uses a single GPIO controller (`tegra234-gpio`) that exposes GPIO
lines starting at a base offset. On L4T 36.x the base is typically 300, but you should
always verify:

```bash
# Find the GPIO base
cat /sys/class/gpio/gpiochip300/base
# Output: 300

# List all GPIOs with their current state
sudo cat /sys/kernel/debug/gpio | head -40
```

The relationship between 40-pin physical pin and Linux sysfs GPIO number:

| 40-Pin Physical | Function  | Sysfs GPIO |
|-----------------|-----------|------------|
| 7               | GPIO09    | 348        |
| 11              | GPIO11    | 316        |
| 13              | GPIO13    | 317        |
| 15              | GPIO15    | 350        |
| 16              | GPIO16    | 389        |
| 18              | GPIO18    | 390        |
| 22              | GPIO22    | 391        |
| 32              | GPIO32    | 392        |
| 33              | GPIO33    | 393        |
| 37              | GPIO37    | 394        |

### 4.2 Jetson.GPIO Python Library

The `Jetson.GPIO` library is NVIDIA's official Python library for GPIO access. It
provides an API compatible with `RPi.GPIO`, making it easy to port Raspberry Pi
projects.

```bash
# Install (if not already)
pip3 install Jetson.GPIO

# Add user to the gpio group (required for non-root access)
sudo groupadd -f -r gpio
sudo usermod -a -G gpio $USER
# Copy the udev rules
sudo cp /opt/nvidia/jetson-gpio/etc/99-gpio.rules /etc/udev/rules.d/
sudo udevadm control --reload-rules && sudo udevadm trigger
# Log out and back in for group membership to take effect
```

#### 4.2.1 Basic Output -- Blink an LED

```
Wiring diagram:

  Pin 7 (GPIO09) ---[330 Ohm]---|>|--- Pin 6 (GND)
                                 LED
                              (anode)  (cathode)

Note: Even with the resistor, a transistor driver is recommended
for production. For bench testing with a small LED, this works.
```

```python
#!/usr/bin/env python3
"""Blink an LED on pin 7 of the Jetson Orin Nano 40-pin header."""

import Jetson.GPIO as GPIO
import time

LED_PIN = 7  # Physical pin 7 (BOARD numbering)

def main():
    GPIO.setmode(GPIO.BOARD)         # Use physical pin numbers
    GPIO.setup(LED_PIN, GPIO.OUT, initial=GPIO.LOW)

    try:
        while True:
            GPIO.output(LED_PIN, GPIO.HIGH)
            time.sleep(0.5)
            GPIO.output(LED_PIN, GPIO.LOW)
            time.sleep(0.5)
    except KeyboardInterrupt:
        pass
    finally:
        GPIO.cleanup()

if __name__ == "__main__":
    main()
```

#### 4.2.2 Basic Input -- Read a Button

```
Wiring diagram:

  +3.3V (Pin 1) ---[10K Ohm]---+--- Pin 11 (GPIO11)
                                |
                            [Button]
                                |
                              GND (Pin 9)

Pull-up resistor ensures GPIO reads HIGH when button is not pressed.
When button is pressed, GPIO is pulled LOW.
```

```python
#!/usr/bin/env python3
"""Read a button on pin 11."""

import Jetson.GPIO as GPIO
import time

BUTTON_PIN = 11  # Physical pin 11

def main():
    GPIO.setmode(GPIO.BOARD)
    GPIO.setup(BUTTON_PIN, GPIO.IN)  # External pull-up on the wire

    try:
        while True:
            state = GPIO.input(BUTTON_PIN)
            if state == GPIO.LOW:
                print("Button PRESSED")
            else:
                print("Button released")
            time.sleep(0.1)
    except KeyboardInterrupt:
        pass
    finally:
        GPIO.cleanup()

if __name__ == "__main__":
    main()
```

#### 4.2.3 Interrupt-Driven GPIO (Edge Detection)

Polling wastes CPU. Use edge detection for efficient, event-driven GPIO:

```python
#!/usr/bin/env python3
"""Interrupt-driven button detection with debouncing."""

import Jetson.GPIO as GPIO
import time

BUTTON_PIN = 11

def button_callback(channel):
    print(f"Button event on channel {channel} at {time.time():.3f}")

def main():
    GPIO.setmode(GPIO.BOARD)
    GPIO.setup(BUTTON_PIN, GPIO.IN)

    # Detect falling edge (button press) with 200ms debounce
    GPIO.add_event_detect(
        BUTTON_PIN,
        GPIO.FALLING,
        callback=button_callback,
        bouncetime=200   # milliseconds -- hardware debounce
    )

    print("Waiting for button press (Ctrl+C to exit)...")
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        pass
    finally:
        GPIO.cleanup()

if __name__ == "__main__":
    main()
```

### 4.3 Sysfs GPIO Interface (Legacy)

The sysfs interface is deprecated in modern kernels but still functional on L4T 36.x.
It is useful for quick shell scripting:

```bash
# Export GPIO 348 (pin 7)
echo 348 | sudo tee /sys/class/gpio/export

# Set direction to output
echo out | sudo tee /sys/class/gpio/gpio348/direction

# Set HIGH
echo 1 | sudo tee /sys/class/gpio/gpio348/value

# Set LOW
echo 0 | sudo tee /sys/class/gpio/gpio348/value

# Read the value
cat /sys/class/gpio/gpio348/value

# Set direction to input
echo in | sudo tee /sys/class/gpio/gpio348/direction

# Configure edge detection for interrupts
echo rising | sudo tee /sys/class/gpio/gpio348/edge
# Options: none, rising, falling, both

# Unexport when done
echo 348 | sudo tee /sys/class/gpio/unexport
```

### 4.4 libgpiod (Modern Character-Device Interface)

`libgpiod` is the modern replacement for sysfs GPIO. It uses the `/dev/gpiochipN`
character device and provides proper atomic line requests with consumer labels.

```bash
# List GPIO chips
gpiodetect
# Output:
# gpiochip0 [tegra234-gpio] (164 lines)
# gpiochip1 [tegra234-gpio-aon] (32 lines)

# List all lines on the main GPIO chip
gpioinfo gpiochip0 | head -20

# Read a specific line (GPIO line 48 = sysfs 348 = pin 7)
# Note: libgpiod uses chip-relative offsets, not sysfs numbers!
gpioget gpiochip0 48

# Set a line high
gpioset gpiochip0 48=1

# Monitor a line for events (interrupt-driven)
gpiomon --falling-edge gpiochip0 16
# (Prints timestamps when falling edges are detected)
```

#### 4.4.1 libgpiod in Python

```python
#!/usr/bin/env python3
"""Using libgpiod v2 Python bindings for GPIO access."""

import gpiod
import time

CHIP_NAME = "/dev/gpiochip0"
LINE_OFFSET = 48  # Chip-relative offset for pin 7

def blink_led():
    """Blink LED using libgpiod character device."""
    request = gpiod.request_lines(
        CHIP_NAME,
        consumer="blink-example",
        config={LINE_OFFSET: gpiod.LineSettings(
            direction=gpiod.line.Direction.OUTPUT,
            output_value=gpiod.line.Value.INACTIVE,
        )},
    )

    try:
        while True:
            request.set_value(LINE_OFFSET, gpiod.line.Value.ACTIVE)
            time.sleep(0.5)
            request.set_value(LINE_OFFSET, gpiod.line.Value.INACTIVE)
            time.sleep(0.5)
    except KeyboardInterrupt:
        pass
    finally:
        request.release()

if __name__ == "__main__":
    blink_led()
```

#### 4.4.2 libgpiod in C

```c
/* gpio_blink.c -- Blink an LED using libgpiod v2 */
#include <gpiod.h>
#include <stdio.h>
#include <unistd.h>

#define CHIP_PATH "/dev/gpiochip0"
#define LINE_OFFSET 48

int main(void)
{
    struct gpiod_chip *chip;
    struct gpiod_line_settings *settings;
    struct gpiod_line_config *line_cfg;
    struct gpiod_request_config *req_cfg;
    struct gpiod_line_request *request;
    unsigned int offsets[] = {LINE_OFFSET};
    enum gpiod_line_value values[] = {GPIOD_LINE_VALUE_INACTIVE};

    chip = gpiod_chip_open(CHIP_PATH);
    if (!chip) { perror("gpiod_chip_open"); return 1; }

    settings = gpiod_line_settings_new();
    gpiod_line_settings_set_direction(settings,
                                      GPIOD_LINE_DIRECTION_OUTPUT);
    line_cfg = gpiod_line_config_new();
    gpiod_line_config_add_line_settings(line_cfg, offsets, 1, settings);

    req_cfg = gpiod_request_config_new();
    gpiod_request_config_set_consumer(req_cfg, "blink-c-example");

    request = gpiod_chip_request_lines(chip, req_cfg, line_cfg);
    if (!request) { perror("request_lines"); return 1; }

    for (int i = 0; i < 20; i++) {
        values[0] = (i % 2) ? GPIOD_LINE_VALUE_ACTIVE
                             : GPIOD_LINE_VALUE_INACTIVE;
        gpiod_line_request_set_values(request, values);
        usleep(500000);
    }

    gpiod_line_request_release(request);
    gpiod_request_config_free(req_cfg);
    gpiod_line_config_free(line_cfg);
    gpiod_line_settings_free(settings);
    gpiod_chip_close(chip);
    return 0;
}
```

```bash
# Compile
gcc -o gpio_blink gpio_blink.c -lgpiod
# Run
sudo ./gpio_blink
```

### 4.5 GPIO Performance and Timing

User-space GPIO toggle rates on the Orin Nano:

| Method            | Approximate Toggle Rate | Notes                       |
|-------------------|------------------------|-----------------------------|
| Jetson.GPIO (Py)  | ~5-10 kHz              | Python overhead              |
| sysfs (shell)     | ~1 kHz                 | Shell fork overhead          |
| libgpiod (Python) | ~10-20 kHz             | Faster than Jetson.GPIO      |
| libgpiod (C)      | ~100-200 kHz           | Recommended for bit-banging  |
| Kernel driver      | ~1 MHz+               | In-kernel GPIO access        |

For anything requiring sub-microsecond timing, use hardware peripherals (SPI, PWM)
instead of bit-banging GPIO.

---

## 5. I2C Interface

### 5.1 Available I2C Buses on the 40-Pin Header

The Orin Nano exposes two I2C buses on the 40-pin header:

| Bus  | 40-Pin SDA | 40-Pin SCL | Linux Device  | T234 Controller | Speed     |
|------|-----------|-----------|---------------|-----------------|-----------|
| I2C1 | Pin 3     | Pin 5     | /dev/i2c-1    | i2c@3160000     | 100/400 kHz |
| I2C7 | Pin 27    | Pin 28    | /dev/i2c-7    | i2c@31e0000     | 100/400 kHz |

Note: The actual Linux bus number may vary depending on L4T version and device
tree configuration. Always verify with `i2cdetect -l`.

```bash
# List all I2C buses
i2cdetect -l
# Example output:
# i2c-0   i2c       Tegra I2C adapter               I2C adapter
# i2c-1   i2c       Tegra I2C adapter               I2C adapter
# ...
# i2c-7   i2c       Tegra I2C adapter               I2C adapter
```

### 5.2 I2C Tools -- Scanning and Basic Access

```bash
# Scan I2C bus 1 for connected devices
sudo i2cdetect -y -r 1
# Output example (MPU6050 at 0x68):
#      0  1  2  3  4  5  6  7  8  9  a  b  c  d  e  f
# 00:                         -- -- -- -- -- -- -- --
# 10: -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --
# 20: -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --
# 30: -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --
# 40: -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --
# 50: -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --
# 60: -- -- -- -- -- -- -- -- 68 -- -- -- -- -- -- --
# 70: -- -- -- -- -- -- -- --

# Read a single byte from register 0x75 (WHO_AM_I) of device 0x68
sudo i2cget -y 1 0x68 0x75
# Output: 0x68  (MPU6050 returns its own address)

# Write 0x00 to register 0x6B (PWR_MGMT_1) to wake up MPU6050
sudo i2cset -y 1 0x68 0x6B 0x00

# Read a block of bytes (register 0x3B, 6 bytes = accel XYZ)
sudo i2cdump -y -r 0x3B-0x40 1 0x68

# Scan I2C bus 7
sudo i2cdetect -y -r 7
```

### 5.3 I2C Wiring

```
Wiring an I2C device to Bus 1:

  Jetson Orin Nano                 I2C Device (e.g., MPU6050)
  +---+                            +---+
  |   | Pin 3 (I2C1_SDA) -------> | SDA |
  |   | Pin 5 (I2C1_SCL) -------> | SCL |
  |   | Pin 1 (+3.3V)    -------> | VCC |
  |   | Pin 6 (GND)      -------> | GND |
  +---+                            +---+

  External pull-up resistors (4.7K to 3.3V) are recommended
  if the device breakout board does not include them.

  +3.3V ----+--------+
            |        |
           [4.7K]   [4.7K]
            |        |
  SDA ------+        +------ SCL
```

### 5.4 Python I2C with smbus2

```python
#!/usr/bin/env python3
"""Read accelerometer data from MPU6050 over I2C bus 1."""

from smbus2 import SMBus
import time
import struct

MPU6050_ADDR   = 0x68
PWR_MGMT_1     = 0x6B
ACCEL_XOUT_H   = 0x3B

def read_word(bus, addr, reg):
    """Read a signed 16-bit value from two consecutive registers."""
    high = bus.read_byte_data(addr, reg)
    low  = bus.read_byte_data(addr, reg + 1)
    value = (high << 8) | low
    if value >= 0x8000:
        value -= 0x10000
    return value

def main():
    with SMBus(1) as bus:  # /dev/i2c-1
        # Wake up the MPU6050 (clear sleep bit)
        bus.write_byte_data(MPU6050_ADDR, PWR_MGMT_1, 0x00)
        time.sleep(0.1)

        print("Reading accelerometer (g) -- Ctrl+C to stop")
        while True:
            ax = read_word(bus, MPU6050_ADDR, ACCEL_XOUT_H) / 16384.0
            ay = read_word(bus, MPU6050_ADDR, ACCEL_XOUT_H + 2) / 16384.0
            az = read_word(bus, MPU6050_ADDR, ACCEL_XOUT_H + 4) / 16384.0
            print(f"  ax={ax:+.3f}g  ay={ay:+.3f}g  az={az:+.3f}g", end='\r')
            time.sleep(0.05)

if __name__ == "__main__":
    main()
```

### 5.5 Block Read / Write (Bulk Transfers)

For higher throughput, use block reads instead of individual register reads:

```python
#!/usr/bin/env python3
"""Efficient block read from MPU6050."""

from smbus2 import SMBus, i2c_msg
import struct

MPU6050_ADDR = 0x68
ACCEL_XOUT_H = 0x3B

def read_accel_block(bus):
    """Read 6 bytes (accel X,Y,Z) in a single I2C transaction."""
    data = bus.read_i2c_block_data(MPU6050_ADDR, ACCEL_XOUT_H, 6)
    ax, ay, az = struct.unpack('>hhh', bytes(data))
    return ax / 16384.0, ay / 16384.0, az / 16384.0

def read_accel_msg(bus):
    """Read using i2c_msg for maximum control."""
    write = i2c_msg.write(MPU6050_ADDR, [ACCEL_XOUT_H])
    read  = i2c_msg.read(MPU6050_ADDR, 6)
    bus.i2c_rdwr(write, read)
    data = list(read)
    ax, ay, az = struct.unpack('>hhh', bytes(data))
    return ax / 16384.0, ay / 16384.0, az / 16384.0

with SMBus(1) as bus:
    bus.write_byte_data(MPU6050_ADDR, 0x6B, 0x00)
    print("Block read:", read_accel_block(bus))
    print("Msg read:  ", read_accel_msg(bus))
```

### 5.6 I2C Speed Configuration

The default I2C speed is 100 kHz (standard mode). To use 400 kHz (fast mode), modify
the device tree:

```dts
// In the device tree or overlay:
&i2c1 {
    clock-frequency = <400000>;  /* 400 kHz fast mode */
    status = "okay";
};
```

You can check the current bus speed:

```bash
# Check dmesg for I2C initialization
dmesg | grep -i i2c
```

### 5.7 I2C Device Tree Node Example

Adding a sensor in the device tree ensures the kernel automatically probes the
correct driver at boot:

```dts
// Overlay to add an MPU6050 on I2C bus 1 at address 0x68
/dts-v1/;
/plugin/;

/ {
    overlay-name = "MPU6050 on I2C1";

    fragment@0 {
        target = <&i2c1>;
        __overlay__ {
            #address-cells = <1>;
            #size-cells = <0>;
            status = "okay";

            mpu6050@68 {
                compatible = "invensense,mpu6050";
                reg = <0x68>;
                interrupt-parent = <&gpio>;
                interrupts = <348 1>;  /* GPIO 348, rising edge */
                mount-matrix = "1", "0", "0",
                               "0", "1", "0",
                               "0", "0", "1";
            };
        };
    };
};
```

### 5.8 Multiple Devices on One Bus

I2C supports multiple devices on the same bus (each with a unique address):

```
  +3.3V ---+--------+--------+
           |        |        |
          [4.7K]   [4.7K]   |
           |        |        |
  SDA -----+--+-----+---+   |
               |         |   |
  SCL ---------+---------+   |
               |         |   |
         +-----+---+ +---+---+---+
         | MPU6050  | | BMP280    |
         | Addr:0x68| | Addr:0x76 |
         +----------+ +-----------+
```

```bash
# Both devices appear on the same bus scan
sudo i2cdetect -y -r 1
#      0  1  2  3  4  5  6  7  8  9  a  b  c  d  e  f
# 60: -- -- -- -- -- -- -- -- 68 -- -- -- -- -- -- --
# 70: -- -- -- -- -- -- 76 --
```

---

## 6. SPI Interface

### 6.1 Available SPI Buses on the 40-Pin Header

The Orin Nano 40-pin header exposes one SPI bus by default (SPI1). A second bus
(SPI0) can be enabled by reconfiguring some GPIO pins via pinmux.

| Bus  | MOSI   | MISO   | SCLK   | CS0    | CS1    | Linux Device        |
|------|--------|--------|--------|--------|--------|---------------------|
| SPI1 | Pin 19 | Pin 21 | Pin 23 | Pin 24 | Pin 26 | /dev/spidev0.0 (CS0)|
| SPI0 | (alt)  | (alt)  | (alt)  | Pin 18 | Pin 16 | /dev/spidev1.0      |

SPI1 is the primary bus for user peripherals. SPI0 requires pinmux changes to
repurpose GPIO pins 16/18 and additional pins.

### 6.2 Enabling SPI via Jetson-IO

```bash
# Use Jetson-IO to enable SPI1
sudo /opt/nvidia/jetson-io/jetson-io.py
# Select: Configure Jetson 40-pin Header
# Select: SPI1 (1 device) -- or SPI1 (2 devices) for both CS0 and CS1
# Save and reboot

# After reboot, verify:
ls -la /dev/spidev*
# Expected: /dev/spidev0.0  (and /dev/spidev0.1 if 2-device mode)
```

### 6.3 SPI Modes and Clock

SPI has four modes defined by CPOL (clock polarity) and CPHA (clock phase):

| Mode | CPOL | CPHA | Clock Idle | Data Sampled On |
|------|------|------|------------|-----------------|
| 0    | 0    | 0    | Low        | Rising edge     |
| 1    | 0    | 1    | Low        | Falling edge    |
| 2    | 1    | 0    | High       | Falling edge    |
| 3    | 1    | 1    | High       | Rising edge     |

Most peripherals use Mode 0. Always check the datasheet.

The T234 SPI controller supports clock rates from ~400 kHz to 65 MHz. Practical
limits depend on wire length, capacitance, and the peripheral's specs.

### 6.4 SPI Wiring

```
  Jetson Orin Nano                    SPI Device (e.g., MCP3008 ADC)
  +---+                                +---+
  |   | Pin 19 (SPI1_MOSI) ---------> | DIN  (MOSI) |
  |   | Pin 21 (SPI1_MISO) <--------- | DOUT (MISO) |
  |   | Pin 23 (SPI1_SCLK) ---------> | CLK  (SCLK) |
  |   | Pin 24 (SPI1_CS0)  ---------> | CS   (SS)   |
  |   | Pin 1  (+3.3V)     ---------> | VCC          |
  |   | Pin 6  (GND)       ---------> | GND          |
  +---+                                +--------------+

  Keep wires short (<15 cm) for reliable high-speed SPI.
```

### 6.5 Command-Line SPI Testing

```bash
# Install spi-tools
sudo apt-get install -y spi-tools

# Send and receive bytes on SPI1 CS0 at 1 MHz, Mode 0
# This performs a full-duplex transfer: sends 0xDE 0xAD, receives 2 bytes
spi-pipe -d /dev/spidev0.0 -s 1000000 -m 0 -b 2 < <(printf '\xDE\xAD') | xxd

# Loopback test: short MOSI (pin 19) to MISO (pin 21) with a jumper wire
# Whatever you send should come back identical
echo -ne '\x01\x02\x03\x04' | spi-pipe -d /dev/spidev0.0 -s 1000000 -b 4 | xxd
# Expected output: 00000000: 0102 0304
```

### 6.6 Python SPI with spidev

```python
#!/usr/bin/env python3
"""SPI loopback test -- short MOSI to MISO for testing."""

import spidev
import time

def main():
    spi = spidev.SpiDev()
    spi.open(0, 0)       # Bus 0, Device 0 = /dev/spidev0.0

    spi.max_speed_hz = 1000000   # 1 MHz
    spi.mode = 0                  # SPI Mode 0 (CPOL=0, CPHA=0)
    spi.bits_per_word = 8

    # Full-duplex transfer: send 4 bytes, receive 4 bytes simultaneously
    tx_data = [0xDE, 0xAD, 0xBE, 0xEF]
    rx_data = spi.xfer2(tx_data)

    print(f"TX: {[hex(b) for b in tx_data]}")
    print(f"RX: {[hex(b) for b in rx_data]}")

    if tx_data == rx_data:
        print("Loopback test PASSED")
    else:
        print("Loopback test FAILED (is MOSI shorted to MISO?)")

    spi.close()

if __name__ == "__main__":
    main()
```

### 6.7 Reading an MCP3008 ADC via SPI

```python
#!/usr/bin/env python3
"""Read analog values from MCP3008 8-channel ADC via SPI."""

import spidev
import time

spi = spidev.SpiDev()
spi.open(0, 0)
spi.max_speed_hz = 1350000  # MCP3008 max is 3.6 MHz at 5V, 1.35 MHz at 2.7V
spi.mode = 0

def read_adc(channel):
    """Read a single channel (0-7) from MCP3008."""
    if channel < 0 or channel > 7:
        raise ValueError("Channel must be 0-7")
    # MCP3008 protocol:
    # Byte 0: Start bit (0x01)
    # Byte 1: Single-ended, channel select (upper nibble)
    # Byte 2: Don't care
    cmd = [0x01, (0x80 | (channel << 4)) & 0xFF, 0x00]
    resp = spi.xfer2(cmd)
    # Result is in the lower 10 bits of bytes 1-2
    value = ((resp[1] & 0x03) << 8) | resp[2]
    return value

try:
    while True:
        for ch in range(8):
            val = read_adc(ch)
            voltage = val * 3.3 / 1023.0
            print(f"  CH{ch}: {val:4d} ({voltage:.2f}V)", end="")
        print()
        time.sleep(0.5)
except KeyboardInterrupt:
    spi.close()
```

### 6.8 SPI Device Tree Overlay

```dts
/* Overlay to enable SPI1 with a custom device on CS0 */
/dts-v1/;
/plugin/;

/ {
    overlay-name = "SPI1 with ADS1256 ADC";

    fragment@0 {
        target = <&spi1>;
        __overlay__ {
            status = "okay";
            #address-cells = <1>;
            #size-cells = <0>;

            spi-max-frequency = <10000000>;
            cs-gpios = <&gpio GP139_SPI1_CS0 GPIO_ACTIVE_LOW>;

            ads1256@0 {
                compatible = "ti,ads1256";
                reg = <0>;   /* CS0 */
                spi-max-frequency = <1920000>;
                spi-cpol;    /* CPOL = 1 */
                spi-cpha;    /* CPHA = 1 -- Mode 3 for ADS1256 */
                vref-supply = <&vdd_3v3>;
                interrupt-parent = <&gpio>;
                interrupts = <391 2>;  /* DRDY on GPIO22, falling */
            };
        };
    };
};
```

### 6.9 SPI Performance Tips

- **Clock speed:** Start at 1 MHz and increase until errors appear, then back off 20%.
- **Wire length:** Keep all SPI wires under 15 cm for >10 MHz operation.
- **CS assertion:** The `spidev` driver handles CS automatically. For manual CS control,
  set `spi.no_cs = True` and toggle a GPIO line.
- **DMA:** The T234 SPI controller supports DMA for large transfers. This is handled
  transparently by the kernel driver for transfers larger than the FIFO depth (64 bytes).
- **Chunked transfers:** For transfers larger than 4096 bytes, split into chunks since
  `spidev` has a default max transfer size (`/sys/module/spidev/parameters/bufsiz`).

```bash
# Check and increase max SPI transfer size
cat /sys/module/spidev/parameters/bufsiz
# Default: 4096

# To increase temporarily:
echo 65536 | sudo tee /sys/module/spidev/parameters/bufsiz
```

---

## 7. UART Interface

### 7.1 Available UARTs on the 40-Pin Header

| UART   | TX Pin | RX Pin | RTS     | CTS     | Linux Device     | T234 Controller |
|--------|--------|--------|---------|---------|------------------|-----------------|
| UART1  | Pin 8  | Pin 10 | Pin 11* | --      | /dev/ttyTHS0     | serial@3100000  |
| UART3  | --     | --     | --      | Pin 36* | /dev/ttyTHS2     | serial@3140000  |

*Pin 11 can be configured as UART1_RTS via pinmux. Pin 36 is UART3_CTS. Full
UART3 TX/RX are available on other board connectors but not the standard 40-pin header
on the devkit. UART1 (pins 8 + 10) is the primary user UART on the 40-pin header.

**Important:** `/dev/ttyTHS0` is the NVIDIA high-speed UART driver. The older
`/dev/ttyS0` refers to the 8250 serial driver and should not be used -- they conflict.

### 7.2 UART Wiring

```
  Jetson Orin Nano                  USB-to-Serial Adapter / Other MCU
  +---+                              +---+
  |   | Pin 8  (UART1_TX) -------->  | RX  |
  |   | Pin 10 (UART1_RX) <--------  | TX  |
  |   | Pin 6  (GND)      -------->  | GND |
  +---+                              +---+

  CRITICAL: Cross TX/RX -- Jetson TX goes to partner RX, and vice versa.
  CRITICAL: Both sides must share a common GND.
  CRITICAL: The Jetson UART is 3.3V logic. Do NOT connect to RS-232
  (+/-12V) or 5V UART directly -- use a level shifter or 3.3V adapter.
```

### 7.3 Basic UART Configuration and Testing

```bash
# Check that ttyTHS0 exists
ls -la /dev/ttyTHS*

# Set permissions (or add user to dialout group)
sudo usermod -a -G dialout $USER
# Re-login for effect

# Disable any getty or console service on this UART (if active)
sudo systemctl stop nvgetty
sudo systemctl disable nvgetty

# Configure UART parameters with stty
sudo stty -F /dev/ttyTHS0 115200 cs8 -cstopb -parenb raw -echo
# 115200 baud, 8 data bits, 1 stop bit, no parity, raw mode

# Send data
echo "Hello from Jetson Orin Nano" | sudo tee /dev/ttyTHS0

# Receive data (blocks until data arrives, Ctrl+C to exit)
sudo cat /dev/ttyTHS0
```

### 7.4 Loopback Test

Short pin 8 (TX) to pin 10 (RX) with a jumper wire:

```bash
# Terminal 1: listen
sudo cat /dev/ttyTHS0 &

# Terminal 2: send
echo "LOOPBACK_TEST_OK" | sudo tee /dev/ttyTHS0

# You should see "LOOPBACK_TEST_OK" printed by the cat process
```

### 7.5 Using minicom / picocom

```bash
# Install
sudo apt-get install -y minicom picocom

# picocom -- simpler, recommended for quick testing
sudo picocom -b 115200 /dev/ttyTHS0
# Exit: Ctrl-A, Ctrl-X

# minicom -- more features
sudo minicom -D /dev/ttyTHS0 -b 115200
# Exit: Ctrl-A, X
# Configure: Ctrl-A, O -> Serial port setup
```

### 7.6 Python UART with pyserial

```python
#!/usr/bin/env python3
"""UART communication using pyserial on Jetson Orin Nano."""

import serial
import time

def uart_loopback_test():
    """Test UART loopback (short TX to RX)."""
    ser = serial.Serial(
        port='/dev/ttyTHS0',
        baudrate=115200,
        bytesize=serial.EIGHTBITS,
        parity=serial.PARITY_NONE,
        stopbits=serial.STOPBITS_ONE,
        timeout=1.0,         # Read timeout in seconds
        write_timeout=1.0,
    )

    test_msg = b"Hello UART Loopback!\n"
    ser.write(test_msg)
    time.sleep(0.1)

    received = ser.read(len(test_msg))
    print(f"Sent:     {test_msg}")
    print(f"Received: {received}")

    if received == test_msg:
        print("Loopback test PASSED")
    else:
        print("Loopback test FAILED")

    ser.close()

def uart_continuous_read():
    """Continuously read lines from UART (e.g., from a GPS module)."""
    ser = serial.Serial('/dev/ttyTHS0', 9600, timeout=1.0)

    print("Reading UART data (Ctrl+C to stop)...")
    try:
        while True:
            line = ser.readline()
            if line:
                print(f"[UART] {line.decode('utf-8', errors='replace').strip()}")
    except KeyboardInterrupt:
        pass
    finally:
        ser.close()

if __name__ == "__main__":
    uart_loopback_test()
```

### 7.7 UART with Flow Control (RTS/CTS)

If you enable UART1_RTS on pin 11 via pinmux, you can use hardware flow control:

```python
ser = serial.Serial(
    port='/dev/ttyTHS0',
    baudrate=921600,
    rtscts=True,    # Enable hardware flow control
    timeout=1.0,
)
```

### 7.8 High Baud Rates and DMA

The Tegra THS UART driver supports baud rates up to 12 Mbaud and uses DMA for
transfers. Supported standard baud rates:

| Baud Rate | Use Case                                    |
|-----------|---------------------------------------------|
| 9600      | GPS modules, slow sensors                   |
| 115200    | General-purpose, debug consoles             |
| 460800    | Fast sensor data                            |
| 921600    | High-speed MCU communication                |
| 3000000   | High-throughput LIDAR, camera triggers      |

```bash
# Set high baud rate
sudo stty -F /dev/ttyTHS0 3000000 raw -echo

# The THS driver automatically uses DMA for large transfers.
# To verify DMA is active, check:
dmesg | grep -i "ttyTHS0" | grep -i dma
```

### 7.9 UART in C

```c
/* uart_example.c -- Basic UART read/write on Jetson Orin Nano */
#include <stdio.h>
#include <string.h>
#include <fcntl.h>
#include <termios.h>
#include <unistd.h>
#include <errno.h>

int main(void)
{
    int fd = open("/dev/ttyTHS0", O_RDWR | O_NOCTTY | O_NDELAY);
    if (fd < 0) { perror("open"); return 1; }

    struct termios tty;
    tcgetattr(fd, &tty);

    cfsetospeed(&tty, B115200);
    cfsetispeed(&tty, B115200);

    tty.c_cflag &= ~PARENB;        /* No parity */
    tty.c_cflag &= ~CSTOPB;        /* 1 stop bit */
    tty.c_cflag &= ~CSIZE;
    tty.c_cflag |= CS8;            /* 8 data bits */
    tty.c_cflag |= CLOCAL | CREAD; /* Enable receiver, local mode */

    tty.c_iflag &= ~(IXON | IXOFF | IXANY); /* No SW flow control */
    tty.c_lflag &= ~(ICANON | ECHO | ECHOE | ISIG); /* Raw mode */
    tty.c_oflag &= ~OPOST;         /* Raw output */

    tty.c_cc[VMIN]  = 0;           /* Non-blocking read */
    tty.c_cc[VTIME] = 10;          /* 1 second timeout */

    tcsetattr(fd, TCSANOW, &tty);

    /* Write */
    const char *msg = "Hello from C UART!\n";
    write(fd, msg, strlen(msg));

    /* Read */
    char buf[256];
    usleep(100000);  /* Wait for loopback */
    int n = read(fd, buf, sizeof(buf) - 1);
    if (n > 0) {
        buf[n] = '\0';
        printf("Received: %s", buf);
    }

    close(fd);
    return 0;
}
```

```bash
gcc -o uart_example uart_example.c
sudo ./uart_example
```

---

## 8. PWM Output

### 8.1 PWM Channels on the 40-Pin Header

The T234 SoC includes multiple PWM controllers. Two PWM outputs are accessible via
the 40-pin header after pinmux configuration:

| PWM Channel | 40-Pin | Default Function | T234 PWM Controller | Sysfs Path            |
|-------------|--------|------------------|---------------------|-----------------------|
| PWM1        | Pin 32 | GPIO32           | pwm@3280000 (ch 0)  | /sys/class/pwm/pwmchip0 |
| PWM5        | Pin 33 | GPIO33           | pwm@32a0000 (ch 0)  | /sys/class/pwm/pwmchip2 |

These pins default to GPIO and must be reconfigured to PWM via Jetson-IO or a
device tree overlay.

### 8.2 Enabling PWM via Jetson-IO

```bash
sudo /opt/nvidia/jetson-io/jetson-io.py
# Select: Configure Jetson 40-pin Header
# Select: PWM (2 channels)
# Save and reboot
sudo reboot

# After reboot, verify PWM chips exist
ls /sys/class/pwm/
# Expected: pwmchip0  pwmchip1  pwmchip2 ...
```

### 8.3 Sysfs PWM Interface

```bash
# --- PWM on pwmchip0, channel 0 (Pin 32) ---

# Export channel 0
echo 0 | sudo tee /sys/class/pwm/pwmchip0/export

# Set period (in nanoseconds) -- 20ms = 50 Hz (servo frequency)
echo 20000000 | sudo tee /sys/class/pwm/pwmchip0/pwm0/period

# Set duty cycle (in nanoseconds) -- 1.5ms = servo center position
echo 1500000 | sudo tee /sys/class/pwm/pwmchip0/pwm0/duty_cycle

# Enable PWM output
echo 1 | sudo tee /sys/class/pwm/pwmchip0/pwm0/enable

# Change duty cycle to move servo (1ms = 0 deg, 2ms = 180 deg)
echo 1000000 | sudo tee /sys/class/pwm/pwmchip0/pwm0/duty_cycle  # 0 degrees
echo 2000000 | sudo tee /sys/class/pwm/pwmchip0/pwm0/duty_cycle  # 180 degrees

# Disable PWM
echo 0 | sudo tee /sys/class/pwm/pwmchip0/pwm0/enable

# Unexport
echo 0 | sudo tee /sys/class/pwm/pwmchip0/unexport
```

### 8.4 Common PWM Frequencies

| Application         | Frequency | Period (ns)    | Notes                   |
|---------------------|-----------|----------------|-------------------------|
| Servo motor         | 50 Hz     | 20,000,000     | 1-2 ms duty cycle       |
| DC motor (H-bridge) | 25 kHz   | 40,000         | Above audible range     |
| LED dimming         | 1 kHz     | 1,000,000      | Flicker-free            |
| Buzzer / tone       | 440 Hz    | 2,272,727      | A4 note                 |
| Fan control         | 25 kHz    | 40,000         | Standard 4-pin PC fan   |

### 8.5 Python PWM Control

```python
#!/usr/bin/env python3
"""PWM servo control on Jetson Orin Nano using sysfs."""

import time
import os

class PWM:
    """Sysfs PWM wrapper for Jetson."""

    def __init__(self, chip=0, channel=0):
        self.base = f"/sys/class/pwm/pwmchip{chip}"
        self.channel = channel
        self.path = f"{self.base}/pwm{channel}"

    def export(self):
        if not os.path.exists(self.path):
            with open(f"{self.base}/export", 'w') as f:
                f.write(str(self.channel))
            time.sleep(0.1)  # Wait for sysfs node creation

    def unexport(self):
        if os.path.exists(self.path):
            with open(f"{self.base}/unexport", 'w') as f:
                f.write(str(self.channel))

    def set_period_ns(self, period_ns):
        with open(f"{self.path}/period", 'w') as f:
            f.write(str(int(period_ns)))

    def set_duty_ns(self, duty_ns):
        with open(f"{self.path}/duty_cycle", 'w') as f:
            f.write(str(int(duty_ns)))

    def set_frequency_hz(self, freq_hz):
        self.set_period_ns(1_000_000_000 / freq_hz)

    def set_duty_percent(self, percent):
        """Set duty cycle as a percentage (0-100)."""
        with open(f"{self.path}/period", 'r') as f:
            period = int(f.read().strip())
        duty = int(period * percent / 100.0)
        self.set_duty_ns(duty)

    def enable(self):
        with open(f"{self.path}/enable", 'w') as f:
            f.write("1")

    def disable(self):
        with open(f"{self.path}/enable", 'w') as f:
            f.write("0")


def sweep_servo():
    """Sweep a servo from 0 to 180 degrees and back."""
    pwm = PWM(chip=0, channel=0)  # Pin 32
    pwm.export()
    pwm.set_period_ns(20_000_000)   # 50 Hz
    pwm.set_duty_ns(1_500_000)      # Center
    pwm.enable()

    try:
        while True:
            # Sweep 0 -> 180 degrees
            for us in range(1000, 2001, 50):
                pwm.set_duty_ns(us * 1000)
                time.sleep(0.03)
            # Sweep 180 -> 0 degrees
            for us in range(2000, 999, -50):
                pwm.set_duty_ns(us * 1000)
                time.sleep(0.03)
    except KeyboardInterrupt:
        pass
    finally:
        pwm.disable()
        pwm.unexport()

if __name__ == "__main__":
    sweep_servo()
```

### 8.6 Jetson.GPIO PWM (Software PWM)

`Jetson.GPIO` provides software-based PWM on any GPIO pin. This is less precise
than hardware PWM but works on any available GPIO:

```python
#!/usr/bin/env python3
"""Software PWM for LED brightness control."""

import Jetson.GPIO as GPIO
import time

LED_PIN = 7  # Any GPIO pin

def main():
    GPIO.setmode(GPIO.BOARD)
    GPIO.setup(LED_PIN, GPIO.OUT)

    # Create a software PWM instance at 1 kHz
    pwm = GPIO.PWM(LED_PIN, 1000)
    pwm.start(0)  # Start with 0% duty cycle

    try:
        while True:
            # Fade in
            for dc in range(0, 101, 5):
                pwm.ChangeDutyCycle(dc)
                time.sleep(0.05)
            # Fade out
            for dc in range(100, -1, -5):
                pwm.ChangeDutyCycle(dc)
                time.sleep(0.05)
    except KeyboardInterrupt:
        pass
    finally:
        pwm.stop()
        GPIO.cleanup()

if __name__ == "__main__":
    main()
```

### 8.7 PWM Wiring for a Servo

```
  Jetson Orin Nano                    Standard Servo
  +---+                               +---+
  |   | Pin 32 (PWM1) ------------->  | Signal (Yellow/White) |
  |   |                               |                       |
  |   | Pin 2  (+5V)  ------------->  | VCC (Red)             |
  |   | Pin 6  (GND)  ------------->  | GND (Brown/Black)     |
  +---+                               +-----------------------+

  IMPORTANT: Servos draw significant current (500mA - 2A).
  Use an external 5V power supply for the servo VCC, NOT pin 2.
  Only share GND between the Jetson and the external supply.

  External 5V PSU -------> Servo VCC (Red)
  External PSU GND --+--> Servo GND (Brown)
                     |
  Jetson Pin 6 (GND) +    (common ground)
  Jetson Pin 32 ----------> Servo Signal
```

---

## 9. CAN Bus

### 9.1 CAN Controller on the Orin Nano

The T234 SoC includes an MTTCAN (Message Triggered CAN) controller that supports:

- CAN 2.0A (11-bit identifiers)
- CAN 2.0B (29-bit extended identifiers)
- CAN FD (Flexible Data Rate) -- up to 8 Mbps data phase

The CAN0 controller signals are exposed on the 40-pin header:

| Signal    | Pin | Description                        |
|-----------|-----|------------------------------------|
| CAN0_DIN  | 29  | CAN RX (from transceiver to SoC)   |
| CAN0_DOUT | 31  | CAN TX (from SoC to transceiver)   |

### 9.2 CAN Transceiver Hardware

The T234 provides CAN controller signals at 3.3V LVCMOS logic. You **must** use an
external CAN transceiver (PHY) to convert to CAN bus differential signaling (CANH/CANL).

Common CAN transceivers:

| Transceiver  | Interface | CAN FD Support | Notes                       |
|--------------|-----------|----------------|-----------------------------|
| MCP2551      | 5V        | No             | Classic CAN 2.0 only        |
| SN65HVD230   | 3.3V      | No             | 3.3V, direct connection     |
| MCP2562FD    | 3.3-5V    | Yes            | CAN FD, recommended         |
| TJA1051T/3   | 3.3V      | Yes            | CAN FD, 3.3V supply         |

```
  Jetson Orin Nano                CAN Transceiver           CAN Bus
  +---+                         (e.g., MCP2562FD)
  |   | Pin 31 (CAN0_DOUT) --> | TXD         CANH | ----+---- CANH
  |   | Pin 29 (CAN0_DIN)  <-- | RXD         CANL | ----+---- CANL
  |   | Pin 1  (+3.3V)     --> | VDD (3.3V)       |     |
  |   | Pin 2  (+5V)       --> | VIO / VSUP (5V)  |   [120R]  (termination)
  |   | Pin 30 (GND)       --> | GND              |     |
  +---+                         +---------+-------+  ---+--- GND
                                          |
                                        [120R]  (bus termination at each end)
                                          |
                                         GND

  120 Ohm termination resistors are required at each end of the CAN bus.
  Many transceiver modules include a switchable termination resistor.
```

### 9.3 Enabling CAN via Pinmux

```bash
# Method 1: Use Jetson-IO
sudo /opt/nvidia/jetson-io/jetson-io.py
# Select: Configure Jetson 40-pin Header
# Select: CAN Bus
# Save and reboot

# Method 2: Manual device tree overlay
# Create and load a CAN overlay (see Section 15)
```

After reboot, verify the CAN interface exists:

```bash
# Check for CAN network interface
ip link show type can
# Expected output:
# 5: can0: <NOARP,ECHO> mtu 16 qdisc noop state DOWN mode DEFAULT group default qlen 10
#     link/can

# If no can0 appears, check:
dmesg | grep -i mttcan
dmesg | grep -i can
```

### 9.4 Configuring the CAN Interface with SocketCAN

```bash
# Install can-utils
sudo apt-get install -y can-utils

# Configure CAN0 for Classic CAN at 500 kbit/s
sudo ip link set can0 type can bitrate 500000
sudo ip link set can0 up

# Configure CAN0 for CAN FD with 500 kbit/s arbitration, 2 Mbit/s data
sudo ip link set can0 type can bitrate 500000 dbitrate 2000000 fd on
sudo ip link set can0 up

# Verify configuration
ip -details link show can0
# Shows bitrate, sample-point, tq, sjw, etc.

# View CAN statistics
ip -statistics link show can0

# Bring interface down
sudo ip link set can0 down
```

### 9.5 Sending and Receiving CAN Messages

```bash
# --- Terminal 1: Receive (candump) ---
candump can0
# Listens for all CAN frames and prints them

# With timestamps:
candump -ta can0

# Filter for specific IDs (only ID 0x123):
candump can0,123:7FF

# --- Terminal 2: Send (cansend) ---

# Send a standard CAN frame: ID=0x123, 8 data bytes
cansend can0 123#DEADBEEF01020304

# Send extended frame (29-bit ID):
cansend can0 18FEF100#0102030405060708

# Send CAN FD frame (up to 64 data bytes):
cansend can0 123##1.DEADBEEFCAFEBABE0102030405060708

# Generate random CAN traffic for testing:
cangen can0 -I i -L 8 -D r -g 10
# -I i: incrementing IDs, -L 8: 8 bytes, -D r: random data, -g 10: 10ms gap
```

### 9.6 CAN Loopback Test (No Transceiver Needed)

For testing without hardware, use loopback mode:

```bash
# Enable loopback mode
sudo ip link set can0 type can bitrate 500000 loopback on
sudo ip link set can0 up

# In one terminal:
candump can0 &

# In same or another terminal:
cansend can0 123#AABBCCDD

# You should see the message echoed back
# Kill the candump:
kill %1
```

### 9.7 Python SocketCAN

```python
#!/usr/bin/env python3
"""CAN bus communication using Python SocketCAN on Jetson Orin Nano."""

import socket
import struct
import time

CAN_INTERFACE = 'can0'

def setup_can():
    """Create a raw CAN socket."""
    sock = socket.socket(socket.AF_CAN, socket.SOCK_RAW, socket.CAN_RAW)
    sock.bind((CAN_INTERFACE,))
    return sock

def send_can_frame(sock, can_id, data):
    """Send a CAN frame.

    Args:
        sock: CAN socket
        can_id: 11-bit or 29-bit CAN identifier
        data: bytes, up to 8 bytes for Classic CAN
    """
    # CAN frame format: ID (4 bytes) + DLC (1 byte) + padding (3 bytes) + data (8 bytes)
    can_dlc = len(data)
    data_padded = data.ljust(8, b'\x00')
    frame = struct.pack("=IB3x8s", can_id, can_dlc, data_padded)
    sock.send(frame)
    print(f"Sent: ID=0x{can_id:03X} Data={data.hex()}")

def receive_can_frame(sock, timeout=1.0):
    """Receive a CAN frame.

    Returns:
        (can_id, data) or None on timeout
    """
    sock.settimeout(timeout)
    try:
        frame = sock.recv(16)
        can_id, can_dlc = struct.unpack("=IB", frame[:5])
        can_id &= 0x1FFFFFFF  # Mask off flags
        data = frame[8:8 + can_dlc]
        print(f"Recv: ID=0x{can_id:03X} DLC={can_dlc} Data={data.hex()}")
        return can_id, data
    except socket.timeout:
        return None

def main():
    sock = setup_can()

    # Send a message
    send_can_frame(sock, 0x123, b'\xDE\xAD\xBE\xEF')

    # Receive messages in a loop
    print("\nListening for CAN messages (Ctrl+C to stop)...")
    try:
        while True:
            receive_can_frame(sock, timeout=1.0)
    except KeyboardInterrupt:
        pass
    finally:
        sock.close()

if __name__ == "__main__":
    main()
```

### 9.8 CAN Bus with python-can Library

```python
#!/usr/bin/env python3
"""Higher-level CAN bus access using the python-can library."""

import can
import time

def main():
    # Create a CAN bus instance using SocketCAN
    bus = can.interface.Bus(channel='can0', interface='socketcan')

    # Send a message
    msg = can.Message(
        arbitration_id=0x7E0,   # OBD-II request ID
        data=[0x02, 0x01, 0x0C, 0x00, 0x00, 0x00, 0x00, 0x00],  # RPM request
        is_extended_id=False,
    )
    bus.send(msg)
    print(f"Sent: {msg}")

    # Receive with timeout
    response = bus.recv(timeout=2.0)
    if response:
        print(f"Recv: {response}")
    else:
        print("No response received")

    # Notifier pattern for asynchronous reception
    def on_message(msg):
        print(f"  Async recv: ID=0x{msg.arbitration_id:03X} data={msg.data.hex()}")

    notifier = can.Notifier(bus, [on_message])
    print("Listening for 10 seconds...")
    time.sleep(10)
    notifier.stop()
    bus.shutdown()

if __name__ == "__main__":
    main()
```

```bash
pip3 install python-can
```

### 9.9 CAN Bus Auto-Start at Boot

```bash
# Create a systemd network configuration for CAN
sudo tee /etc/systemd/network/80-can.network << 'EOF'
[Match]
Name=can0

[CAN]
BitRate=500K
RestartSec=100ms
EOF

# Enable systemd-networkd
sudo systemctl enable systemd-networkd
sudo systemctl restart systemd-networkd

# Alternatively, use an rc.local or systemd service:
sudo tee /etc/systemd/system/can0.service << 'EOF'
[Unit]
Description=CAN0 Interface
After=network.target

[Service]
Type=oneshot
RemainAfterExit=yes
ExecStart=/sbin/ip link set can0 type can bitrate 500000
ExecStart=/sbin/ip link set can0 up
ExecStop=/sbin/ip link set can0 down

[Install]
WantedBy=multi-user.target
EOF

sudo systemctl enable can0.service
sudo systemctl start can0.service
```

---

## 10. ADC (Analog Input)

### 10.1 No On-Header ADC

The Jetson Orin Nano 40-pin header does **not** expose any native analog-to-digital
converter (ADC) pins. All 40-pin signals are digital. To read analog sensors, you must
use one of these approaches:

1. **On-board INA3221** -- For voltage/current monitoring of the Jetson's own power rails.
2. **External ADC via I2C** -- e.g., ADS1115 (16-bit, 4-channel).
3. **External ADC via SPI** -- e.g., MCP3008 (10-bit, 8-channel) or ADS1256 (24-bit).

### 10.2 On-Board INA3221 Power Monitor

The Orin Nano carrier board includes an INA3221 triple-channel current/voltage
monitor that measures the main power rails. It is accessible via I2C.

```bash
# Find the INA3221 on the I2C bus
sudo i2cdetect -y -r 1
# The INA3221 is typically at address 0x40 or 0x41

# Read via the Linux hwmon interface (preferred)
cat /sys/bus/i2c/drivers/ina3221/*/hwmon/hwmon*/in*_input
cat /sys/bus/i2c/drivers/ina3221/*/hwmon/hwmon*/curr*_input

# Or find under /sys/class/hwmon
for d in /sys/class/hwmon/hwmon*/; do
    name=$(cat "${d}name" 2>/dev/null)
    if [ "$name" = "ina3221" ]; then
        echo "INA3221 found at $d"
        cat "${d}in1_input"    # Channel 1 bus voltage (mV)
        cat "${d}curr1_input"  # Channel 1 current (mA)
    fi
done
```

```python
#!/usr/bin/env python3
"""Read Jetson Orin Nano power rail voltages and currents via INA3221."""

import glob
import os

def find_ina3221():
    """Find the INA3221 hwmon directory."""
    for hwmon_dir in glob.glob("/sys/class/hwmon/hwmon*/"):
        name_file = os.path.join(hwmon_dir, "name")
        if os.path.exists(name_file):
            with open(name_file) as f:
                if "ina3221" in f.read():
                    return hwmon_dir
    return None

def read_power():
    hwmon = find_ina3221()
    if not hwmon:
        print("INA3221 not found")
        return

    for ch in range(1, 4):
        v_file = os.path.join(hwmon, f"in{ch}_input")
        i_file = os.path.join(hwmon, f"curr{ch}_input")
        if os.path.exists(v_file) and os.path.exists(i_file):
            with open(v_file) as f:
                voltage_mv = int(f.read().strip())
            with open(i_file) as f:
                current_ma = int(f.read().strip())
            power_mw = voltage_mv * current_ma / 1000.0
            print(f"Channel {ch}: {voltage_mv}mV, {current_ma}mA, {power_mw:.1f}mW")

if __name__ == "__main__":
    read_power()
```

### 10.3 External ADC via I2C -- ADS1115

The ADS1115 is a popular 16-bit, 4-channel delta-sigma ADC with a programmable
gain amplifier (PGA). It communicates via I2C and is ideal for precision analog
measurements.

```
  Jetson Orin Nano                    ADS1115 Breakout
  +---+                               +---+
  |   | Pin 3  (I2C1_SDA) --------->  | SDA   |
  |   | Pin 5  (I2C1_SCL) --------->  | SCL   |
  |   | Pin 1  (+3.3V)    --------->  | VDD   |
  |   | Pin 6  (GND)      --------->  | GND   |
  +---+                               |       |
                                      | A0 <--- Analog Input 0 (0-3.3V)
                                      | A1 <--- Analog Input 1
                                      | A2 <--- Analog Input 2
                                      | A3 <--- Analog Input 3
                                      | ADDR -> GND (address 0x48)
                                      +-------+
```

```python
#!/usr/bin/env python3
"""Read analog values from ADS1115 16-bit ADC via I2C."""

from smbus2 import SMBus
import time

ADS1115_ADDR = 0x48
REG_CONVERSION = 0x00
REG_CONFIG     = 0x01

# Configuration register values
CONFIG_MUX = {
    0: 0x4000,  # AIN0 vs GND (single-ended)
    1: 0x5000,  # AIN1 vs GND
    2: 0x6000,  # AIN2 vs GND
    3: 0x7000,  # AIN3 vs GND
}

# PGA gain settings (full-scale range)
PGA_6_144V = 0x0000   # +/- 6.144V (LSB = 187.5 uV)
PGA_4_096V = 0x0200   # +/- 4.096V (LSB = 125 uV)
PGA_2_048V = 0x0400   # +/- 2.048V (LSB = 62.5 uV) -- default
PGA_1_024V = 0x0600   # +/- 1.024V (LSB = 31.25 uV)

# Sample rate
SPS_128 = 0x0080   # 128 samples/sec
SPS_860 = 0x00E0   # 860 samples/sec (maximum)

# Operating mode
SINGLE_SHOT = 0x0100
OS_START     = 0x8000  # Start a single conversion

def read_channel(bus, channel, gain=PGA_4_096V):
    """Read a single-ended analog value from the specified channel."""
    config = (OS_START | CONFIG_MUX[channel] | gain |
              SINGLE_SHOT | SPS_128 | 0x0003)  # Disable comparator

    # Write config (big-endian 16-bit)
    bus.write_i2c_block_data(ADS1115_ADDR, REG_CONFIG,
                             [(config >> 8) & 0xFF, config & 0xFF])

    # Wait for conversion (at 128 SPS, conversion takes ~7.8ms)
    time.sleep(0.01)

    # Read result (big-endian signed 16-bit)
    data = bus.read_i2c_block_data(ADS1115_ADDR, REG_CONVERSION, 2)
    raw = (data[0] << 8) | data[1]
    if raw >= 0x8000:
        raw -= 0x10000

    # Convert to voltage (PGA_4_096V: LSB = 125 uV)
    voltage = raw * 0.000125
    return voltage

def main():
    with SMBus(1) as bus:
        print("ADS1115 ADC Readings (Ctrl+C to stop)")
        while True:
            for ch in range(4):
                v = read_channel(bus, ch)
                print(f"  CH{ch}: {v:.4f}V", end="")
            print()
            time.sleep(0.5)

if __name__ == "__main__":
    main()
```

### 10.4 External ADC via SPI -- MCP3008

For faster sampling with lower resolution, the MCP3008 (10-bit, 8-channel) is a
good choice. See Section 6.7 for the complete MCP3008 SPI code example.

### 10.5 Analog Sensor Interfacing Summary

| Sensor Type         | Typical Output | ADC Needed | Recommended ADC |
|---------------------|----------------|------------|-----------------|
| Potentiometer       | 0-3.3V         | Yes        | MCP3008 (SPI)   |
| Force-sensitive res.| 0-3.3V divider | Yes        | ADS1115 (I2C)   |
| pH sensor           | 0-3V           | Yes        | ADS1115 (I2C)   |
| Thermocouple        | uV-mV range    | Yes        | MAX31855 (SPI)  |
| Light sensor (LDR)  | Voltage divider| Yes        | MCP3008 (SPI)   |
| Current sensor      | 0-3.3V         | Yes        | ADS1115 (I2C)   |

---

## 11. Sensor Integration Examples

### 11.1 IMU -- MPU6050 via I2C

The MPU6050 provides 3-axis accelerometer and 3-axis gyroscope data. See
Section 5.4 for the basic I2C read code. Here is a more complete example with
gyroscope and temperature:

```python
#!/usr/bin/env python3
"""Complete MPU6050 6-DOF IMU reader with calibration."""

from smbus2 import SMBus
import time
import math

MPU6050_ADDR   = 0x68
PWR_MGMT_1     = 0x6B
SMPLRT_DIV     = 0x19
CONFIG_REG     = 0x1A
GYRO_CONFIG    = 0x1B
ACCEL_CONFIG   = 0x1C
ACCEL_XOUT_H   = 0x3B
TEMP_OUT_H     = 0x41
GYRO_XOUT_H    = 0x43

class MPU6050:
    def __init__(self, bus_num=1, addr=MPU6050_ADDR):
        self.bus = SMBus(bus_num)
        self.addr = addr
        self._init_sensor()
        self.gyro_offset = [0.0, 0.0, 0.0]

    def _init_sensor(self):
        # Wake up (clear sleep bit)
        self.bus.write_byte_data(self.addr, PWR_MGMT_1, 0x00)
        time.sleep(0.1)
        # Sample rate: 1kHz / (1 + 9) = 100 Hz
        self.bus.write_byte_data(self.addr, SMPLRT_DIV, 9)
        # DLPF bandwidth: 44 Hz
        self.bus.write_byte_data(self.addr, CONFIG_REG, 0x03)
        # Gyro: +/- 250 deg/s
        self.bus.write_byte_data(self.addr, GYRO_CONFIG, 0x00)
        # Accel: +/- 2g
        self.bus.write_byte_data(self.addr, ACCEL_CONFIG, 0x00)

    def _read_raw(self, reg):
        data = self.bus.read_i2c_block_data(self.addr, reg, 2)
        value = (data[0] << 8) | data[1]
        if value >= 0x8000:
            value -= 0x10000
        return value

    def read_all(self):
        """Read all 14 bytes (accel + temp + gyro) in one transaction."""
        data = self.bus.read_i2c_block_data(self.addr, ACCEL_XOUT_H, 14)
        ax = ((data[0] << 8) | data[1])
        ay = ((data[2] << 8) | data[3])
        az = ((data[4] << 8) | data[5])
        temp = ((data[6] << 8) | data[7])
        gx = ((data[8] << 8) | data[9])
        gy = ((data[10] << 8) | data[11])
        gz = ((data[12] << 8) | data[13])

        for v in [ax, ay, az, temp, gx, gy, gz]:
            pass  # Sign extension handled below

        def signed(val):
            return val - 0x10000 if val >= 0x8000 else val

        ax, ay, az = signed(ax)/16384.0, signed(ay)/16384.0, signed(az)/16384.0
        gx = signed(gx)/131.0 - self.gyro_offset[0]
        gy = signed(gy)/131.0 - self.gyro_offset[1]
        gz = signed(gz)/131.0 - self.gyro_offset[2]
        temp_c = signed(temp) / 340.0 + 36.53

        return {
            'accel': (ax, ay, az),     # in g
            'gyro': (gx, gy, gz),      # in deg/s
            'temp': temp_c,            # in Celsius
        }

    def calibrate_gyro(self, samples=200):
        """Calibrate gyroscope by averaging readings while stationary."""
        print("Calibrating gyro (keep sensor still)...")
        sums = [0.0, 0.0, 0.0]
        for _ in range(samples):
            data = self.read_all()
            sums[0] += data['gyro'][0]
            sums[1] += data['gyro'][1]
            sums[2] += data['gyro'][2]
            time.sleep(0.005)
        self.gyro_offset = [s / samples for s in sums]
        print(f"Gyro offsets: {self.gyro_offset}")

def main():
    imu = MPU6050()
    imu.calibrate_gyro()

    print("\nIMU Data (Ctrl+C to stop):")
    print(f"{'ax':>8} {'ay':>8} {'az':>8} | {'gx':>8} {'gy':>8} {'gz':>8} | {'temp':>6}")
    try:
        while True:
            d = imu.read_all()
            a = d['accel']
            g = d['gyro']
            print(f"{a[0]:+8.3f} {a[1]:+8.3f} {a[2]:+8.3f} | "
                  f"{g[0]:+8.2f} {g[1]:+8.2f} {g[2]:+8.2f} | "
                  f"{d['temp']:6.1f}C", end='\r')
            time.sleep(0.02)
    except KeyboardInterrupt:
        print()

if __name__ == "__main__":
    main()
```

### 11.2 Temperature / Humidity -- BME280 via I2C

```python
#!/usr/bin/env python3
"""Read temperature, humidity, and pressure from BME280."""

# pip3 install bme280 smbus2
from smbus2 import SMBus
import bme280

BME280_ADDR = 0x76  # or 0x77 depending on SDO pin

with SMBus(1) as bus:
    # Load calibration parameters
    calibration_params = bme280.load_calibration_params(bus, BME280_ADDR)

    while True:
        data = bme280.sample(bus, BME280_ADDR, calibration_params)
        print(f"Temperature: {data.temperature:.2f} C")
        print(f"Humidity:    {data.humidity:.2f} %")
        print(f"Pressure:    {data.pressure:.2f} hPa")
        print()
        import time; time.sleep(1)
```

### 11.3 Distance Sensor -- HC-SR04 Ultrasonic via GPIO

```
  Jetson Orin Nano                    HC-SR04
  +---+                               +---+
  |   | Pin 1  (+3.3V)   --------->  | VCC (5V preferred, 3.3V may work) |
  |   | Pin 7  (GPIO09)  --------->  | TRIG |
  |   |                              |      |
  |   | Pin 11 (GPIO11)  <---[R]---  | ECHO |  (ECHO is 5V! Use divider)
  |   | Pin 6  (GND)     --------->  | GND  |
  +---+                               +------+

  CRITICAL: The HC-SR04 ECHO pin outputs 5V. Use a voltage divider:
  ECHO ---[1K Ohm]---+--- Pin 11 (GPIO11)
                      |
                    [2K Ohm]
                      |
                     GND
  This divides 5V down to ~3.3V.
```

```python
#!/usr/bin/env python3
"""HC-SR04 ultrasonic distance measurement on Jetson Orin Nano."""

import Jetson.GPIO as GPIO
import time

TRIG_PIN = 7    # Physical pin 7
ECHO_PIN = 11   # Physical pin 11

def measure_distance():
    """Measure distance in centimeters using HC-SR04."""
    # Send 10us trigger pulse
    GPIO.output(TRIG_PIN, GPIO.HIGH)
    time.sleep(0.00001)
    GPIO.output(TRIG_PIN, GPIO.LOW)

    # Wait for echo to go HIGH
    timeout = time.time() + 0.04  # 40ms max
    while GPIO.input(ECHO_PIN) == GPIO.LOW:
        pulse_start = time.time()
        if pulse_start > timeout:
            return -1  # Timeout

    # Measure how long echo stays HIGH
    while GPIO.input(ECHO_PIN) == GPIO.HIGH:
        pulse_end = time.time()
        if pulse_end > timeout:
            return -1  # Timeout

    pulse_duration = pulse_end - pulse_start
    distance_cm = pulse_duration * 34300 / 2.0  # Speed of sound: 343 m/s
    return distance_cm

def main():
    GPIO.setmode(GPIO.BOARD)
    GPIO.setup(TRIG_PIN, GPIO.OUT, initial=GPIO.LOW)
    GPIO.setup(ECHO_PIN, GPIO.IN)

    time.sleep(0.5)  # Let sensor settle

    try:
        while True:
            dist = measure_distance()
            if dist > 0:
                print(f"Distance: {dist:.1f} cm")
            else:
                print("Distance: timeout")
            time.sleep(0.2)
    except KeyboardInterrupt:
        pass
    finally:
        GPIO.cleanup()

if __name__ == "__main__":
    main()
```

### 11.4 Time-of-Flight Sensor -- VL53L0X via I2C

```python
#!/usr/bin/env python3
"""VL53L0X Time-of-Flight distance sensor via I2C."""

# pip3 install vl53l0x-python
# Or use the adafruit-circuitpython-vl53l0x library:
# pip3 install adafruit-circuitpython-vl53l0x

from smbus2 import SMBus
import time

VL53L0X_ADDR = 0x29  # Default address

# Simplified register-level access (for full implementation,
# use the adafruit library or vl53l0x-python)
def read_vl53l0x():
    """Basic VL53L0X reading using adafruit library."""
    import board
    import busio
    import adafruit_vl53l0x

    i2c = busio.I2C(board.SCL, board.SDA)
    sensor = adafruit_vl53l0x.VL53L0X(i2c)

    # Set measurement timing budget (microseconds)
    sensor.measurement_timing_budget = 33000  # 33ms for good accuracy

    print("VL53L0X Distance (Ctrl+C to stop):")
    while True:
        distance_mm = sensor.range
        print(f"  Distance: {distance_mm} mm ({distance_mm/10:.1f} cm)")
        time.sleep(0.1)

if __name__ == "__main__":
    read_vl53l0x()
```

### 11.5 Sensor Wiring Summary Table

| Sensor      | Interface | Pins Used         | I2C Address | Notes                    |
|-------------|-----------|-------------------|-------------|--------------------------|
| MPU6050     | I2C       | 3 (SDA), 5 (SCL)  | 0x68/0x69   | AD0 pin selects addr     |
| ICM20948    | I2C/SPI   | 3,5 or 19,21,23,24| 0x68/0x69   | 9-DOF IMU                |
| BME280      | I2C       | 3 (SDA), 5 (SCL)  | 0x76/0x77   | Temp/Humidity/Pressure   |
| BMP390      | I2C/SPI   | 3,5 or SPI pins    | 0x76/0x77   | Barometric pressure      |
| VL53L0X     | I2C       | 3 (SDA), 5 (SCL)  | 0x29        | ToF distance, up to 2m   |
| HC-SR04     | GPIO      | 7 (Trig), 11 (Echo)| --          | Ultrasonic, 2-400 cm     |
| ADS1115     | I2C       | 3 (SDA), 5 (SCL)  | 0x48-0x4B   | 16-bit ADC, 4 channels   |
| MCP3008     | SPI       | 19,21,23,24        | --          | 10-bit ADC, 8 channels   |
| GPS (NEO-6M)| UART     | 8 (TX), 10 (RX)   | --          | NMEA sentences at 9600   |
| MAX31855    | SPI       | 19,21,23,24        | --          | Thermocouple amplifier   |

---

## 12. Display Interfaces

### 12.1 I2C OLED Display -- SSD1306

The SSD1306 is a ubiquitous 128x64 (or 128x32) monochrome OLED controller
commonly used for status displays. It connects via I2C and is ideal for showing
inference results, sensor readings, or system status.

```
  Jetson Orin Nano                   SSD1306 OLED (I2C)
  +---+                               +---+
  |   | Pin 3  (I2C1_SDA) --------->  | SDA  |
  |   | Pin 5  (I2C1_SCL) --------->  | SCL  |
  |   | Pin 1  (+3.3V)    --------->  | VCC  |
  |   | Pin 6  (GND)      --------->  | GND  |
  +---+                               +------+

  I2C Address: 0x3C (or 0x3D if SA0 is high)
```

```python
#!/usr/bin/env python3
"""Display text and inference results on SSD1306 OLED via I2C."""

# pip3 install luma.oled luma.core Pillow
from luma.core.interface.serial import i2c
from luma.oled.device import ssd1306
from luma.core.render import canvas
from PIL import ImageFont
import time

def main():
    # Initialize display on I2C bus 1
    serial = i2c(port=1, address=0x3C)
    device = ssd1306(serial, width=128, height=64, rotate=0)

    # Load a font (use default if TrueType not available)
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 12)
        font_small = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 10)
    except IOError:
        font = ImageFont.load_default()
        font_small = font

    # Display system info
    import subprocess
    while True:
        # Get GPU temperature
        try:
            with open("/sys/devices/virtual/thermal/thermal_zone0/temp") as f:
                temp = int(f.read().strip()) / 1000.0
        except:
            temp = 0.0

        with canvas(device) as draw:
            draw.rectangle(device.bounding_box, outline="white", fill="black")
            draw.text((4, 2),  "Jetson Orin Nano", font=font, fill="white")
            draw.text((4, 18), f"Temp: {temp:.1f} C", font=font_small, fill="white")
            draw.text((4, 30), f"Status: Running", font=font_small, fill="white")
            draw.text((4, 42), f"FPS: 30.2", font=font_small, fill="white")
            draw.text((4, 52), time.strftime("%H:%M:%S"), font=font_small, fill="white")

        time.sleep(1)

if __name__ == "__main__":
    main()
```

### 12.2 SPI TFT Display -- ST7789 / ILI9341

SPI-based TFT displays offer color output (240x240 or 320x240) at higher refresh
rates than I2C OLEDs.

```
  Jetson Orin Nano                   ST7789 TFT (SPI)
  +---+                               +---+
  |   | Pin 19 (SPI1_MOSI) -------->  | SDA/DIN |  (Data In)
  |   | Pin 23 (SPI1_SCLK) -------->  | SCL/CLK |
  |   | Pin 24 (SPI1_CS0)  -------->  | CS      |
  |   | Pin 18 (GPIO18)    -------->  | DC      |  (Data/Command)
  |   | Pin 22 (GPIO22)    -------->  | RST     |  (Reset)
  |   | Pin 16 (GPIO16)    -------->  | BLK     |  (Backlight, optional)
  |   | Pin 1  (+3.3V)     -------->  | VCC     |
  |   | Pin 6  (GND)       -------->  | GND     |
  +---+                               +---------+
```

```python
#!/usr/bin/env python3
"""Drive an ST7789 SPI TFT display from Jetson Orin Nano."""

# pip3 install st7789 Pillow spidev
# Or use luma.lcd:
# pip3 install luma.lcd

import spidev
import Jetson.GPIO as GPIO
import time
from PIL import Image, ImageDraw, ImageFont

# GPIO pins (BOARD numbering)
DC_PIN  = 18
RST_PIN = 22
BL_PIN  = 16

WIDTH  = 240
HEIGHT = 240

class ST7789Simple:
    """Minimal ST7789 driver for Jetson (illustrative, not production)."""

    def __init__(self):
        GPIO.setmode(GPIO.BOARD)
        GPIO.setup(DC_PIN, GPIO.OUT)
        GPIO.setup(RST_PIN, GPIO.OUT)
        GPIO.setup(BL_PIN, GPIO.OUT)

        self.spi = spidev.SpiDev()
        self.spi.open(0, 0)
        self.spi.max_speed_hz = 40000000  # 40 MHz
        self.spi.mode = 0

        self._reset()
        self._init_display()
        GPIO.output(BL_PIN, GPIO.HIGH)  # Backlight on

    def _reset(self):
        GPIO.output(RST_PIN, GPIO.HIGH)
        time.sleep(0.05)
        GPIO.output(RST_PIN, GPIO.LOW)
        time.sleep(0.05)
        GPIO.output(RST_PIN, GPIO.HIGH)
        time.sleep(0.15)

    def _send_command(self, cmd):
        GPIO.output(DC_PIN, GPIO.LOW)  # Command mode
        self.spi.writebytes([cmd])

    def _send_data(self, data):
        GPIO.output(DC_PIN, GPIO.HIGH)  # Data mode
        # Send in chunks (spidev has transfer size limits)
        for i in range(0, len(data), 4096):
            self.spi.writebytes(data[i:i+4096])

    def _init_display(self):
        """Send initialization sequence for ST7789."""
        self._send_command(0x01)  # Software reset
        time.sleep(0.15)
        self._send_command(0x11)  # Sleep out
        time.sleep(0.1)
        self._send_command(0x3A)  # Pixel format: 16-bit RGB565
        self._send_data([0x55])
        self._send_command(0x36)  # Memory access control
        self._send_data([0x00])
        self._send_command(0x29)  # Display on

    def display_image(self, img):
        """Display a PIL Image (must be 240x240 RGB)."""
        rgb565 = []
        pixels = img.load()
        for y in range(HEIGHT):
            for x in range(WIDTH):
                r, g, b = pixels[x, y][:3]
                color = ((r & 0xF8) << 8) | ((g & 0xFC) << 3) | (b >> 3)
                rgb565.append((color >> 8) & 0xFF)
                rgb565.append(color & 0xFF)

        # Set window
        self._send_command(0x2A)  # Column address
        self._send_data([0, 0, 0, WIDTH - 1])
        self._send_command(0x2B)  # Row address
        self._send_data([0, 0, 0, HEIGHT - 1])
        self._send_command(0x2C)  # Memory write
        self._send_data(rgb565)

def main():
    display = ST7789Simple()

    img = Image.new('RGB', (WIDTH, HEIGHT), color=(0, 0, 0))
    draw = ImageDraw.Draw(img)
    draw.rectangle([10, 10, 230, 230], outline=(255, 255, 255))
    draw.text((30, 100), "Jetson Orin Nano", fill=(0, 255, 0))
    draw.text((50, 120), "SPI Display", fill=(255, 255, 0))

    display.display_image(img)
    print("Image displayed. Press Ctrl+C to exit.")

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        GPIO.cleanup()

if __name__ == "__main__":
    main()
```

### 12.3 DSI Display Connection

The Orin Nano Developer Kit carrier board includes a DSI (Display Serial Interface)
connector for direct panel connection. DSI is not on the 40-pin header but is
mentioned here for completeness.

```bash
# Check connected DSI display
cat /sys/class/drm/card*/status

# The DSI interface is configured in the device tree:
# tegra234-p3768-camera-p3768-0000-a0.dtsi or similar
# Panel drivers: panel-simple, panel-lvds, etc.

# For custom DSI panels, you need:
# 1. Panel timing parameters (from datasheet)
# 2. A device tree node under the dsi controller
# 3. Possibly a panel driver module
```

### 12.4 Displaying Inference Results on Attached Displays

```python
#!/usr/bin/env python3
"""Display TensorRT inference results on an SSD1306 OLED."""

from luma.core.interface.serial import i2c
from luma.oled.device import ssd1306
from luma.core.render import canvas
from PIL import ImageFont
import time

# Simulated inference output (replace with actual TensorRT inference)
def run_inference():
    """Placeholder for actual inference -- returns class and confidence."""
    import random
    classes = ["person", "car", "bicycle", "dog", "cat", "truck"]
    return random.choice(classes), random.uniform(0.7, 0.99)

def main():
    serial = i2c(port=1, address=0x3C)
    device = ssd1306(serial, width=128, height=64)

    try:
        font = ImageFont.truetype(
            "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf", 11)
    except IOError:
        font = ImageFont.load_default()

    frame_count = 0
    t_start = time.time()

    while True:
        cls, conf = run_inference()
        frame_count += 1
        fps = frame_count / (time.time() - t_start)

        with canvas(device) as draw:
            draw.text((0, 0),  "-- AI Detector --", font=font, fill="white")
            draw.text((0, 14), f"Class: {cls}", font=font, fill="white")
            draw.text((0, 28), f"Conf:  {conf:.1%}", font=font, fill="white")
            draw.text((0, 42), f"FPS:   {fps:.1f}", font=font, fill="white")
            # Draw confidence bar
            bar_w = int(100 * conf)
            draw.rectangle([0, 56, bar_w, 63], fill="white")

        time.sleep(0.033)  # ~30 FPS

if __name__ == "__main__":
    main()
```

---

## 13. Motor and Actuator Control

### 13.1 DC Motor via PWM + H-Bridge

DC motors require an H-bridge driver (e.g., L298N, DRV8833, TB6612FNG) because
GPIO pins cannot source enough current and motors need bidirectional control.

```
  Jetson Orin Nano                  L298N H-Bridge              DC Motor
  +---+                             +---+                       +---+
  |   | Pin 32 (PWM1)  ---------> | ENA  (Speed/Enable)  |     |   |
  |   | Pin 11 (GPIO11) --------> | IN1  (Direction A)   |-->--| M |
  |   | Pin 13 (GPIO13) --------> | IN2  (Direction B)   |-->--| M |
  |   | Pin 6  (GND)    --------> | GND                  |     +---+
  +---+                             |                      |
                                    | +12V <--- Motor PSU  |
                                    | GND  <--- Motor PSU  |
                                    +----------------------+

  IMPORTANT: Never power motors from the Jetson 5V/3.3V pins.
  Use a separate power supply for the motor driver.
  Connect all GNDs together (Jetson GND + Motor PSU GND + L298N GND).
```

```python
#!/usr/bin/env python3
"""DC motor control via PWM (speed) and GPIO (direction)."""

import Jetson.GPIO as GPIO
import time
import os

# Pin assignments (BOARD numbering)
DIR_A_PIN = 11    # IN1 on H-bridge
DIR_B_PIN = 13    # IN2 on H-bridge

# PWM on hardware PWM pin 32 (via sysfs)
PWM_CHIP = 0
PWM_CHANNEL = 0
PWM_BASE = f"/sys/class/pwm/pwmchip{PWM_CHIP}"
PWM_PATH = f"{PWM_BASE}/pwm{PWM_CHANNEL}"

def pwm_init(freq_hz=25000):
    """Initialize hardware PWM."""
    if not os.path.exists(PWM_PATH):
        with open(f"{PWM_BASE}/export", 'w') as f:
            f.write(str(PWM_CHANNEL))
        time.sleep(0.1)
    period_ns = int(1e9 / freq_hz)
    with open(f"{PWM_PATH}/period", 'w') as f:
        f.write(str(period_ns))
    with open(f"{PWM_PATH}/duty_cycle", 'w') as f:
        f.write("0")
    with open(f"{PWM_PATH}/enable", 'w') as f:
        f.write("1")

def pwm_set_duty(percent):
    """Set PWM duty cycle (0-100)."""
    with open(f"{PWM_PATH}/period", 'r') as f:
        period = int(f.read().strip())
    duty = int(period * max(0, min(100, percent)) / 100)
    with open(f"{PWM_PATH}/duty_cycle", 'w') as f:
        f.write(str(duty))

def motor_forward(speed):
    GPIO.output(DIR_A_PIN, GPIO.HIGH)
    GPIO.output(DIR_B_PIN, GPIO.LOW)
    pwm_set_duty(speed)

def motor_reverse(speed):
    GPIO.output(DIR_A_PIN, GPIO.LOW)
    GPIO.output(DIR_B_PIN, GPIO.HIGH)
    pwm_set_duty(speed)

def motor_stop():
    GPIO.output(DIR_A_PIN, GPIO.LOW)
    GPIO.output(DIR_B_PIN, GPIO.LOW)
    pwm_set_duty(0)

def main():
    GPIO.setmode(GPIO.BOARD)
    GPIO.setup(DIR_A_PIN, GPIO.OUT, initial=GPIO.LOW)
    GPIO.setup(DIR_B_PIN, GPIO.OUT, initial=GPIO.LOW)
    pwm_init(25000)  # 25 kHz PWM

    try:
        print("Forward 50%")
        motor_forward(50)
        time.sleep(2)

        print("Forward 100%")
        motor_forward(100)
        time.sleep(2)

        print("Stop")
        motor_stop()
        time.sleep(1)

        print("Reverse 75%")
        motor_reverse(75)
        time.sleep(2)

        print("Stop")
        motor_stop()

    except KeyboardInterrupt:
        pass
    finally:
        motor_stop()
        GPIO.cleanup()

if __name__ == "__main__":
    main()
```

### 13.2 Servo Motor Control

See Section 8.5 for the complete hardware PWM servo control code. Key parameters:

| Servo Type    | Period | Min Duty | Max Duty | Angle Range |
|---------------|--------|----------|----------|-------------|
| Standard      | 20 ms  | 1.0 ms   | 2.0 ms   | 0-180 deg   |
| Continuous     | 20 ms  | 1.0 ms   | 2.0 ms   | Speed ctrl  |
| Digital (high-res) | 20 ms | 0.5 ms | 2.5 ms | 0-270 deg  |

### 13.3 Stepper Motor via GPIO

```
  Jetson Orin Nano                   ULN2003 Driver        28BYJ-48 Stepper
  +---+                              +---+                  +---+
  |   | Pin 7  (GPIO09)  --------->  | IN1 |  ============  |   |
  |   | Pin 11 (GPIO11)  --------->  | IN2 |  ============  |   |
  |   | Pin 13 (GPIO13)  --------->  | IN3 |  ============  |   |
  |   | Pin 15 (GPIO15)  --------->  | IN4 |  ============  |   |
  |   | Pin 6  (GND)     --------->  | GND |                +---+
  +---+                              | +5V | <-- External 5V
                                     +-----+
```

```python
#!/usr/bin/env python3
"""Stepper motor control (28BYJ-48 + ULN2003) via GPIO."""

import Jetson.GPIO as GPIO
import time

# Stepper motor pins (BOARD numbering)
PINS = [7, 11, 13, 15]

# Half-step sequence (8 steps for smoother movement)
HALF_STEP_SEQ = [
    [1, 0, 0, 0],
    [1, 1, 0, 0],
    [0, 1, 0, 0],
    [0, 1, 1, 0],
    [0, 0, 1, 0],
    [0, 0, 1, 1],
    [0, 0, 0, 1],
    [1, 0, 0, 1],
]

def setup():
    GPIO.setmode(GPIO.BOARD)
    for pin in PINS:
        GPIO.setup(pin, GPIO.OUT, initial=GPIO.LOW)

def step(steps, delay=0.001, direction=1):
    """Move the stepper motor a given number of steps.

    Args:
        steps: Number of steps to take
        delay: Delay between steps (seconds) -- controls speed
        direction: 1 for clockwise, -1 for counter-clockwise
    """
    seq = HALF_STEP_SEQ if direction == 1 else HALF_STEP_SEQ[::-1]

    for i in range(steps):
        pattern = seq[i % len(seq)]
        for pin_idx, pin in enumerate(PINS):
            GPIO.output(pin, pattern[pin_idx])
        time.sleep(delay)

    # De-energize coils to save power (motor will not hold position)
    for pin in PINS:
        GPIO.output(pin, GPIO.LOW)

def main():
    setup()
    try:
        print("Rotating clockwise (512 steps = ~90 degrees)")
        step(512, delay=0.001, direction=1)
        time.sleep(0.5)

        print("Rotating counter-clockwise (512 steps)")
        step(512, delay=0.001, direction=-1)

    except KeyboardInterrupt:
        pass
    finally:
        GPIO.cleanup()

if __name__ == "__main__":
    main()
```

### 13.4 PCA9685 16-Channel PWM Driver via I2C

When you need more PWM channels than the Jetson provides (e.g., for multiple
servos or LEDs), the PCA9685 I2C PWM driver provides 16 independent channels.

```
  Jetson Orin Nano                    PCA9685
  +---+                               +---+
  |   | Pin 3  (I2C1_SDA) --------->  | SDA     |
  |   | Pin 5  (I2C1_SCL) --------->  | SCL     |
  |   | Pin 1  (+3.3V)    --------->  | VCC     |
  |   | Pin 6  (GND)      --------->  | GND     |
  +---+                               |         |
                                      | V+  <--- External 5-6V for servos
                                      | CH0..15  --> Servo/LED signals
                                      +---------+
  I2C Address: 0x40 (default), up to 0x7F with address jumpers
```

```python
#!/usr/bin/env python3
"""Control servos via PCA9685 16-channel PWM driver."""

from smbus2 import SMBus
import time
import math

PCA9685_ADDR   = 0x40
MODE1          = 0x00
PRESCALE       = 0xFE
LED0_ON_L      = 0x06

class PCA9685:
    def __init__(self, bus_num=1, addr=PCA9685_ADDR):
        self.bus = SMBus(bus_num)
        self.addr = addr
        self._reset()

    def _reset(self):
        self.bus.write_byte_data(self.addr, MODE1, 0x00)
        time.sleep(0.005)

    def set_pwm_freq(self, freq_hz):
        """Set PWM frequency for all channels (typically 50 Hz for servos)."""
        prescaleval = 25000000.0 / 4096.0 / freq_hz - 1.0
        prescale = int(math.floor(prescaleval + 0.5))

        oldmode = self.bus.read_byte_data(self.addr, MODE1)
        newmode = (oldmode & 0x7F) | 0x10   # Sleep mode
        self.bus.write_byte_data(self.addr, MODE1, newmode)
        self.bus.write_byte_data(self.addr, PRESCALE, prescale)
        self.bus.write_byte_data(self.addr, MODE1, oldmode)
        time.sleep(0.005)
        self.bus.write_byte_data(self.addr, MODE1, oldmode | 0x80)  # Restart

    def set_pwm(self, channel, on, off):
        """Set PWM on/off values for a channel (0-4095)."""
        reg_base = LED0_ON_L + 4 * channel
        self.bus.write_byte_data(self.addr, reg_base + 0, on & 0xFF)
        self.bus.write_byte_data(self.addr, reg_base + 1, on >> 8)
        self.bus.write_byte_data(self.addr, reg_base + 2, off & 0xFF)
        self.bus.write_byte_data(self.addr, reg_base + 3, off >> 8)

    def set_servo_angle(self, channel, angle):
        """Set servo angle (0-180 degrees) on a given channel."""
        # Map 0-180 to pulse width 1ms-2ms at 50Hz (period=20ms)
        # 1ms = 205/4096, 2ms = 410/4096
        pulse = int(205 + (angle / 180.0) * 205)
        self.set_pwm(channel, 0, pulse)

def main():
    pca = PCA9685()
    pca.set_pwm_freq(50)  # 50 Hz for servos

    try:
        # Sweep servo on channel 0
        while True:
            for angle in range(0, 181, 5):
                pca.set_servo_angle(0, angle)
                time.sleep(0.03)
            for angle in range(180, -1, -5):
                pca.set_servo_angle(0, angle)
                time.sleep(0.03)
    except KeyboardInterrupt:
        pass

if __name__ == "__main__":
    main()
```

### 13.5 Actuator Safety Considerations

- **Emergency stop:** Always implement a hardware kill switch (e.g., relay on a GPIO)
  that cuts motor power independently of software.
- **Current limiting:** Use motor drivers with current sense and feedback.
- **Watchdog:** If the Jetson crashes, motors should default to OFF. Consider a
  hardware watchdog or a separate safety MCU (e.g., Arduino) that monitors a
  heartbeat signal from the Jetson.
- **Voltage isolation:** Use opto-isolators or isolated gate drivers when controlling
  high-voltage loads (>24V).

---

## 14. Real-Time Peripheral Access

### 14.1 Latency Considerations

When integrating peripherals with AI inference pipelines, latency matters. The
following table shows typical round-trip latencies for different peripheral access
methods on the Orin Nano running a standard L4T kernel:

| Operation                        | Typical Latency   | Jitter (Std Dev) |
|----------------------------------|-------------------|------------------|
| GPIO toggle (libgpiod, C)        | 5-10 us           | 2-5 us           |
| GPIO toggle (Python Jetson.GPIO) | 100-200 us        | 50-100 us        |
| I2C read (single byte, 400 kHz)  | 30-50 us          | 10-20 us         |
| I2C block read (14 bytes, 400k)  | 100-150 us        | 20-50 us         |
| SPI transfer (4 bytes, 10 MHz)   | 10-20 us          | 5-10 us          |
| SPI transfer (4096 bytes, 10M)   | 400-500 us        | 20-50 us         |
| CAN send (SocketCAN)             | 50-100 us         | 30-80 us         |
| UART write (10 bytes, 115200)    | ~870 us           | <10 us (HW)      |

Jitter comes from kernel scheduling, preemption, and interrupt handling. Under
heavy CPU load (e.g., during inference), jitter can increase 2-5x.

### 14.2 DMA for SPI and I2C

The T234's SPI and I2C controllers support DMA transfers, which reduce CPU
involvement and improve throughput for large transfers.

**SPI DMA:** The Tegra SPI driver (`spi-tegra210-quad`) automatically uses DMA for
transfers larger than the FIFO depth. You do not need to configure DMA manually.

```bash
# Verify DMA is in use for SPI
dmesg | grep -i "spi.*dma"
# Example: tegra210-quad-spi 3210000.spi: DMA channels allocated

# Check the DMA buffer size
cat /sys/module/spidev/parameters/bufsiz
```

**I2C DMA:** The Tegra I2C driver uses DMA for transfers larger than the threshold
(typically 32 bytes). For smaller transfers, PIO mode is used.

```bash
# Verify I2C DMA
dmesg | grep -i "i2c.*dma"
```

### 14.3 Kernel-Space vs. User-Space Drivers

| Aspect              | User-Space (sysfs/libgpiod/spidev) | Kernel-Space (driver module) |
|---------------------|------------------------------------|------------------------------|
| Development speed   | Fast                               | Slow                         |
| Debugging           | Easy (printf, gdb)                 | Harder (dmesg, ftrace)       |
| Latency             | Higher (context switches)          | Lower (direct HW access)     |
| Jitter              | Higher (scheduler dependent)       | Lower (interrupt context)    |
| Crash impact        | Process dies, system OK            | Can panic the kernel         |
| DMA access          | Limited                            | Full                         |
| Interrupt handling  | poll()/epoll() or blocking read    | Direct IRQ handlers          |

**When to use kernel-space:** When you need deterministic sub-100us response times
or direct DMA control for custom peripherals.

**When to use user-space:** For all prototyping, most production applications, and
whenever latency requirements are >1ms.

### 14.4 PREEMPT_RT Patch for Reduced Jitter

The standard L4T kernel is `PREEMPT` (voluntary preemption). For hard real-time
peripheral access, apply the `PREEMPT_RT` patch:

```bash
# Check current kernel preemption model
cat /sys/kernel/realtime
# 0 = not RT, 1 = PREEMPT_RT

uname -v
# Look for "PREEMPT_RT" in the version string

# If not RT, you need to rebuild the kernel with PREEMPT_RT:
# 1. Download L4T kernel source from NVIDIA
# 2. Apply the PREEMPT_RT patch matching your kernel version
# 3. Configure: make menuconfig
#    -> General Setup -> Preemption Model -> Fully Preemptible (RT)
# 4. Build and install

# After RT kernel is running:
# Set real-time scheduling for your process
sudo chrt -f 80 ./my_realtime_app

# Lock memory to prevent page faults
# In C: mlockall(MCL_CURRENT | MCL_FUTURE);
# In Python: not directly available, use ctypes
```

### 14.5 Isolating CPU Cores for Peripheral Handling

Reserve a CPU core exclusively for peripheral I/O to eliminate scheduling jitter:

```bash
# Isolate CPU core 5 from the scheduler (add to kernel command line)
# Edit /boot/extlinux/extlinux.conf:
#   APPEND ... isolcpus=5 nohz_full=5 rcu_nocbs=5

# After reboot, pin your I/O process to the isolated core:
sudo taskset -c 5 ./my_io_handler

# In Python:
import os
os.sched_setaffinity(0, {5})
```

### 14.6 Real-Time GPIO Interrupt Handler (C, Kernel Module)

```c
/* rt_gpio_irq.c -- Kernel module for low-latency GPIO interrupt handling */
#include <linux/module.h>
#include <linux/gpio.h>
#include <linux/interrupt.h>
#include <linux/time.h>

#define GPIO_NUM   316   /* GPIO11, pin 11 on 40-pin header */
#define GPIO_LABEL "rt_trigger"

static int irq_number;
static ktime_t last_event;

static irqreturn_t gpio_irq_handler(int irq, void *dev_id)
{
    ktime_t now = ktime_get();
    s64 delta_ns = ktime_to_ns(ktime_sub(now, last_event));
    last_event = now;

    pr_info("rt_gpio: IRQ! delta=%lld ns\n", delta_ns);

    /* Toggle an output GPIO for latency measurement with oscilloscope */
    /* gpio_set_value(OUTPUT_GPIO, !gpio_get_value(OUTPUT_GPIO)); */

    return IRQ_HANDLED;
}

static int __init rt_gpio_init(void)
{
    int ret;

    ret = gpio_request(GPIO_NUM, GPIO_LABEL);
    if (ret) { pr_err("gpio_request failed\n"); return ret; }

    gpio_direction_input(GPIO_NUM);
    irq_number = gpio_to_irq(GPIO_NUM);

    ret = request_irq(irq_number, gpio_irq_handler,
                      IRQF_TRIGGER_FALLING, GPIO_LABEL, NULL);
    if (ret) { pr_err("request_irq failed\n"); gpio_free(GPIO_NUM); return ret; }

    last_event = ktime_get();
    pr_info("rt_gpio: registered IRQ %d for GPIO %d\n", irq_number, GPIO_NUM);
    return 0;
}

static void __exit rt_gpio_exit(void)
{
    free_irq(irq_number, NULL);
    gpio_free(GPIO_NUM);
    pr_info("rt_gpio: unloaded\n");
}

module_init(rt_gpio_init);
module_exit(rt_gpio_exit);
MODULE_LICENSE("GPL");
MODULE_DESCRIPTION("Real-time GPIO interrupt handler for Jetson Orin Nano");
```

```makefile
# Makefile for the kernel module
obj-m += rt_gpio_irq.o

KDIR := /lib/modules/$(shell uname -r)/build

all:
	make -C $(KDIR) M=$(PWD) modules

clean:
	make -C $(KDIR) M=$(PWD) clean
```

```bash
make
sudo insmod rt_gpio_irq.ko
dmesg | tail -5
# Trigger the GPIO to see interrupt response
sudo rmmod rt_gpio_irq
```

---

## 15. Custom Device Tree Overlays

### 15.1 Device Tree Overlay Basics

Device Tree Overlays (DTOs) allow you to modify the hardware description at boot
without recompiling the entire device tree. On the Orin Nano, overlays are the
standard way to:

- Enable/disable peripherals (SPI, I2C, CAN, UART)
- Configure pin multiplexing
- Add sensor/device nodes
- Set clock rates and interrupt mappings

### 15.2 Overlay File Structure

```dts
/* template-overlay.dts */
/dts-v1/;
/plugin/;

/ {
    /* Metadata */
    overlay-name = "My Custom Overlay";
    jetson-header-name = "Jetson 40-pin Header";
    compatible = "nvidia,p3768-0000+p3767-0003";  /* Orin Nano devkit */

    /* Fragment 0: Modify an existing node */
    fragment@0 {
        target-path = "/";           /* Or: target = <&some_phandle>; */
        __overlay__ {
            /* Properties and child nodes to add/modify */
        };
    };

    /* Fragment 1: Another modification */
    fragment@1 {
        target = <&spi1>;
        __overlay__ {
            status = "okay";
            /* ... */
        };
    };
};
```

### 15.3 Writing an Overlay -- Example: Enable I2C7 with a Custom Device

```dts
/* i2c7-aht20-overlay.dts -- Enable I2C7 and add AHT20 temp/humidity sensor */
/dts-v1/;
/plugin/;

/ {
    overlay-name = "I2C7 with AHT20 Sensor";
    compatible = "nvidia,p3768-0000+p3767-0003";

    /* Fragment 0: Configure pinmux for I2C7 */
    fragment@0 {
        target = <&pinmux>;
        __overlay__ {
            pinctrl-names = "default";
            pinctrl-0 = <&i2c7_pinmux>;

            i2c7_pinmux: i2c7-pins {
                hdr40-pin27 {
                    nvidia,pins = "dp_aux_ch3_p_pn7";
                    nvidia,function = "i2c7";
                    nvidia,pull = <TEGRA_PIN_PULL_NONE>;
                    nvidia,tristate = <TEGRA_PIN_DISABLE>;
                    nvidia,enable-input = <TEGRA_PIN_ENABLE>;
                };
                hdr40-pin28 {
                    nvidia,pins = "dp_aux_ch3_n_pn0";
                    nvidia,function = "i2c7";
                    nvidia,pull = <TEGRA_PIN_PULL_NONE>;
                    nvidia,tristate = <TEGRA_PIN_DISABLE>;
                    nvidia,enable-input = <TEGRA_PIN_ENABLE>;
                };
            };
        };
    };

    /* Fragment 1: Enable I2C7 controller and add sensor */
    fragment@1 {
        target = <&i2c7>;
        __overlay__ {
            #address-cells = <1>;
            #size-cells = <0>;
            status = "okay";
            clock-frequency = <400000>;

            aht20@38 {
                compatible = "aosong,aht20";
                reg = <0x38>;
            };
        };
    };
};
```

### 15.4 Compiling and Installing Overlays

```bash
# Compile the overlay
dtc -I dts -O dtb -@ -o i2c7-aht20.dtbo i2c7-aht20-overlay.dts
# -@ enables symbol generation (needed for references like <&i2c7>)

# If you get errors about undefined symbols, you may need to
# preprocess with the kernel's device tree includes:
cpp -nostdinc -I /path/to/kernel/include -undef -x assembler-with-cpp \
    i2c7-aht20-overlay.dts | dtc -I dts -O dtb -@ -o i2c7-aht20.dtbo -

# Copy to the boot partition
sudo cp i2c7-aht20.dtbo /boot/

# Method 1: Add to extlinux.conf
sudo nano /boot/extlinux/extlinux.conf
# Add this line under the LABEL section:
#   OVERLAYS /boot/i2c7-aht20.dtbo

# Method 2: Place in Jetson-IO's overlay directory
sudo mkdir -p /boot/device-tree/overlays
sudo cp i2c7-aht20.dtbo /boot/device-tree/overlays/

# Method 3: Load at runtime (if supported by kernel)
sudo dtoverlay i2c7-aht20.dtbo
# (Note: runtime overlay loading may not work on all L4T versions)

# Reboot to apply
sudo reboot
```

### 15.5 Verifying Overlay Application

```bash
# Check if overlay was loaded
cat /proc/device-tree/overlay-name
# Or check the specific node
ls /proc/device-tree/i2c@31e0000/aht20@38/
# Should show: compatible  name  reg

# Check the full device tree as seen by the kernel
dtc -I fs /proc/device-tree 2>/dev/null | less
# Search for your node

# Check dmesg for driver binding
dmesg | grep -i aht20
```

### 15.6 Debugging Device Tree Issues

```bash
# Problem: "overlay apply failed"
# Cause: Usually a phandle mismatch or incompatible target
dmesg | grep -i "overlay\|dtb\|device.tree"

# Decompile the base device tree to inspect it
dtc -I dtb -O dts /boot/dtb/kernel_tegra234-p3768-0000+p3767-0003-nv.dtb \
    -o base-dt.dts
# Now search base-dt.dts for the node you are targeting

# Common issues:
# 1. Wrong compatible string -- must match the base DT
# 2. Missing -@ flag during compilation (no __symbols__ node)
# 3. Target phandle does not exist in base DT
# 4. Syntax errors in DTS file

# Validate DTS syntax
dtc -I dts -O dtb /dev/null -o /dev/null your-overlay.dts
# Will print errors if any

# Check kernel config for overlay support
zcat /proc/config.gz | grep CONFIG_OF_OVERLAY
# Should show: CONFIG_OF_OVERLAY=y
```

### 15.7 Overlay for Multiple Peripherals

```dts
/* multi-peripheral-overlay.dts -- Enable SPI1 + CAN0 + PWM */
/dts-v1/;
/plugin/;

/ {
    overlay-name = "SPI1 + CAN0 + PWM";
    compatible = "nvidia,p3768-0000+p3767-0003";

    /* Pinmux fragment */
    fragment@0 {
        target = <&pinmux>;
        __overlay__ {
            pinctrl-names = "default";
            pinctrl-0 = <&multi_pinmux>;

            multi_pinmux: multi-peripheral-pins {
                /* SPI1 pins (19,21,23,24) */
                spi1-mosi { nvidia,pins = "spi1_mosi_pz4"; nvidia,function = "spi1"; };
                spi1-miso { nvidia,pins = "spi1_miso_pz5"; nvidia,function = "spi1"; };
                spi1-sck  { nvidia,pins = "spi1_sck_pz3";  nvidia,function = "spi1"; };
                spi1-cs0  { nvidia,pins = "spi1_cs0_pz6";  nvidia,function = "spi1"; };

                /* CAN0 pins (29,31) */
                can0-din  { nvidia,pins = "can0_din_paa1";  nvidia,function = "can0"; };
                can0-dout { nvidia,pins = "can0_dout_paa0"; nvidia,function = "can0"; };

                /* PWM on pin 32 */
                pwm-pin32 { nvidia,pins = "soc_gpio33_pq6"; nvidia,function = "gp_pwm1"; };
            };
        };
    };

    /* SPI1 */
    fragment@1 {
        target = <&spi1>;
        __overlay__ {
            status = "okay";
            spi@0 { compatible = "spidev"; reg = <0>; spi-max-frequency = <10000000>; };
        };
    };

    /* CAN0 */
    fragment@2 {
        target = <&mttcan0>;
        __overlay__ {
            status = "okay";
        };
    };

    /* PWM */
    fragment@3 {
        target = <&pwm1>;
        __overlay__ {
            status = "okay";
        };
    };
};
```

---

## 16. Integration with AI Pipeline

### 16.1 Architecture Overview

A typical edge AI system on the Orin Nano connects sensors, inference, and actuators
through peripheral I/O. The data flow looks like this:

```
  +----------+      +----------+      +-----------+      +----------+
  | Sensors  | ---> | Preproc  | ---> | Inference | ---> | Actuators|
  | (I2C/SPI |      | (CPU/    |      | (GPU /    |      | (GPIO/   |
  |  UART/   |      |  NumPy)  |      |  TensorRT |      |  PWM/    |
  |  GPIO)   |      |          |      |  / DLA)   |      |  CAN)    |
  +----------+      +----------+      +-----------+      +----------+
       |                                    |                  |
       +-------- Feedback Loop -------------+------------------+

  Trigger Path (low latency):
  GPIO Interrupt --> Start Inference --> GPIO/PWM Output

  Streaming Path:
  I2C/SPI Sensor --> Ring Buffer --> Batch Inference --> CAN/UART Output
```

### 16.2 Sensor Data to Inference Pipeline

```python
#!/usr/bin/env python3
"""
Complete pipeline: Read IMU data via I2C, run TensorRT inference to classify
motion (e.g., idle/walking/running), output result via GPIO LED and CAN bus.
"""

import threading
import queue
import time
import numpy as np
from smbus2 import SMBus
import Jetson.GPIO as GPIO

# ---- Configuration ----
IMU_BUS       = 1
IMU_ADDR      = 0x68
LED_PIN       = 7       # Green LED: inference running
ALERT_PIN     = 11      # Red LED: anomaly detected
SAMPLE_RATE   = 100     # Hz
WINDOW_SIZE   = 128     # Samples per inference window
NUM_FEATURES  = 6       # accel_x/y/z + gyro_x/y/z

# ---- Shared data ----
sensor_queue = queue.Queue(maxsize=10)

# ---- Sensor Thread ----
def sensor_thread():
    """Read IMU data at fixed rate, push windows to the queue."""
    bus = SMBus(IMU_BUS)
    bus.write_byte_data(IMU_ADDR, 0x6B, 0x00)  # Wake up MPU6050
    time.sleep(0.1)

    buffer = np.zeros((WINDOW_SIZE, NUM_FEATURES), dtype=np.float32)
    idx = 0

    while True:
        # Read 14 bytes: accel(6) + temp(2) + gyro(6)
        data = bus.read_i2c_block_data(IMU_ADDR, 0x3B, 14)
        values = []
        for i in range(0, 12, 2):  # Skip temp bytes at index 6-7
            if i == 6:
                continue
            raw = (data[i] << 8) | data[i+1]
            if raw >= 0x8000:
                raw -= 0x10000
            values.append(raw)

        # Also read gyro
        for i in range(8, 14, 2):
            raw = (data[i] << 8) | data[i+1]
            if raw >= 0x8000:
                raw -= 0x10000
            values.append(raw)

        # Normalize
        accel = [v / 16384.0 for v in values[:3]]
        gyro  = [v / 131.0 for v in values[3:]]

        buffer[idx] = accel + gyro
        idx += 1

        if idx >= WINDOW_SIZE:
            # Push a copy to the queue
            try:
                sensor_queue.put_nowait(buffer.copy())
            except queue.Full:
                pass  # Drop oldest window
            idx = 0

        time.sleep(1.0 / SAMPLE_RATE)


# ---- Inference Thread ----
def inference_thread():
    """Run TensorRT inference on sensor windows."""
    # In production, load a TensorRT engine:
    # import tensorrt as trt
    # engine = load_engine("motion_classifier.engine")
    # context = engine.create_execution_context()

    LABELS = ["idle", "walking", "running", "falling"]

    while True:
        window = sensor_queue.get()
        GPIO.output(LED_PIN, GPIO.HIGH)  # Signal: inference active

        # Preprocess: normalize, reshape for model input
        input_data = window.reshape(1, WINDOW_SIZE, NUM_FEATURES)

        # ---- Placeholder: Replace with actual TensorRT inference ----
        # Simulate inference delay and result
        time.sleep(0.01)
        predicted_class = np.random.randint(0, len(LABELS))
        confidence = np.random.uniform(0.7, 0.99)
        # ---- End placeholder ----

        label = LABELS[predicted_class]
        GPIO.output(LED_PIN, GPIO.LOW)

        print(f"[Inference] {label} ({confidence:.1%})")

        # Trigger alert on anomaly
        if label == "falling" and confidence > 0.85:
            GPIO.output(ALERT_PIN, GPIO.HIGH)
            time.sleep(0.5)
            GPIO.output(ALERT_PIN, GPIO.LOW)


# ---- Main ----
def main():
    GPIO.setmode(GPIO.BOARD)
    GPIO.setup(LED_PIN, GPIO.OUT, initial=GPIO.LOW)
    GPIO.setup(ALERT_PIN, GPIO.OUT, initial=GPIO.LOW)

    t_sensor = threading.Thread(target=sensor_thread, daemon=True)
    t_infer  = threading.Thread(target=inference_thread, daemon=True)

    t_sensor.start()
    t_infer.start()

    print("Pipeline running (Ctrl+C to stop)...")
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        pass
    finally:
        GPIO.cleanup()

if __name__ == "__main__":
    main()
```

### 16.3 Triggering Inference from GPIO

Use a GPIO interrupt to start inference only when an event occurs (e.g., a proximity
sensor trigger or a camera frame-ready signal):

```python
#!/usr/bin/env python3
"""GPIO-triggered inference: run detection only when motion is detected."""

import Jetson.GPIO as GPIO
import time
import subprocess

TRIGGER_PIN = 11   # PIR motion sensor output connected here
LED_PIN     = 7    # Status LED

inference_running = False

def trigger_callback(channel):
    """Called on falling edge of the trigger pin."""
    global inference_running
    if inference_running:
        return  # Skip if already running

    inference_running = True
    GPIO.output(LED_PIN, GPIO.HIGH)
    print(f"[{time.strftime('%H:%M:%S')}] Motion detected -- running inference")

    # Option 1: Run inference in-process (TensorRT, etc.)
    # result = run_tensorrt_inference(capture_frame())

    # Option 2: Trigger an external inference script
    # subprocess.run(["python3", "run_detection.py"])

    # Simulate inference time
    time.sleep(0.5)

    GPIO.output(LED_PIN, GPIO.LOW)
    inference_running = False
    print(f"[{time.strftime('%H:%M:%S')}] Inference complete")

def main():
    GPIO.setmode(GPIO.BOARD)
    GPIO.setup(TRIGGER_PIN, GPIO.IN)
    GPIO.setup(LED_PIN, GPIO.OUT, initial=GPIO.LOW)

    GPIO.add_event_detect(
        TRIGGER_PIN, GPIO.RISING,
        callback=trigger_callback,
        bouncetime=1000   # Minimum 1 second between triggers
    )

    print("Waiting for trigger on pin 11...")
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        pass
    finally:
        GPIO.cleanup()

if __name__ == "__main__":
    main()
```

### 16.4 Controlling Actuators from Inference Results

```python
#!/usr/bin/env python3
"""
Inference-to-actuator pipeline:
Camera -> Object Detection (TensorRT) -> Servo tracks detected object
"""

import time
import os

# Assuming inference returns bounding box center_x (0.0 to 1.0)
# and we map that to servo angle (0 to 180 degrees)

PWM_PATH = "/sys/class/pwm/pwmchip0/pwm0"

def init_servo():
    """Initialize PWM for servo on pin 32."""
    if not os.path.exists(PWM_PATH):
        with open("/sys/class/pwm/pwmchip0/export", 'w') as f:
            f.write("0")
        time.sleep(0.1)
    with open(f"{PWM_PATH}/period", 'w') as f:
        f.write("20000000")  # 50 Hz
    with open(f"{PWM_PATH}/duty_cycle", 'w') as f:
        f.write("1500000")   # Center
    with open(f"{PWM_PATH}/enable", 'w') as f:
        f.write("1")

def set_servo_angle(angle_deg):
    """Set servo angle (0-180)."""
    # Map angle to duty cycle: 0deg=1ms, 180deg=2ms
    duty_us = 1000 + (angle_deg / 180.0) * 1000
    with open(f"{PWM_PATH}/duty_cycle", 'w') as f:
        f.write(str(int(duty_us * 1000)))  # Convert to ns

def inference_to_servo(detection_center_x):
    """Map detection center_x (0.0-1.0) to servo angle."""
    angle = detection_center_x * 180.0
    set_servo_angle(angle)

def main():
    init_servo()

    # Simulated inference loop
    print("Tracking object with servo (simulated)...")
    try:
        while True:
            # Replace with actual inference result
            import math
            t = time.time()
            simulated_x = 0.5 + 0.4 * math.sin(t * 0.5)

            inference_to_servo(simulated_x)
            angle = simulated_x * 180
            print(f"  Object at x={simulated_x:.2f} -> Servo angle={angle:.0f} deg",
                  end='\r')
            time.sleep(0.033)  # 30 FPS
    except KeyboardInterrupt:
        pass
    finally:
        with open(f"{PWM_PATH}/enable", 'w') as f:
            f.write("0")

if __name__ == "__main__":
    main()
```

### 16.5 Sending Inference Results over CAN Bus

```python
#!/usr/bin/env python3
"""Send object detection results over CAN bus for vehicle integration."""

import socket
import struct
import time

CAN_INTERFACE = 'can0'

# CAN message IDs (application-defined)
CAN_ID_DETECTION   = 0x100  # Detection report
CAN_ID_HEARTBEAT   = 0x700  # Heartbeat / alive signal

def create_can_socket():
    sock = socket.socket(socket.AF_CAN, socket.SOCK_RAW, socket.CAN_RAW)
    sock.bind((CAN_INTERFACE,))
    return sock

def send_detection(sock, class_id, confidence, bbox_x, bbox_y):
    """
    Pack detection result into a CAN frame:
    Byte 0:   class_id (0-255)
    Byte 1:   confidence (0-100, percentage)
    Byte 2-3: bbox center X (uint16, pixels)
    Byte 4-5: bbox center Y (uint16, pixels)
    Byte 6-7: reserved
    """
    data = struct.pack('>BBHHxx',
                       int(class_id),
                       int(confidence * 100),
                       int(bbox_x),
                       int(bbox_y))
    frame = struct.pack('=IB3x8s', CAN_ID_DETECTION, 8, data)
    sock.send(frame)

def send_heartbeat(sock, uptime_sec):
    """Send periodic heartbeat with uptime."""
    data = struct.pack('>I4x', int(uptime_sec))
    frame = struct.pack('=IB3x8s', CAN_ID_HEARTBEAT, 8, data)
    sock.send(frame)

def main():
    sock = create_can_socket()
    start_time = time.time()

    print("Sending inference results over CAN bus...")
    try:
        while True:
            # Simulated detection
            send_detection(sock,
                           class_id=2,        # e.g., "person"
                           confidence=0.92,
                           bbox_x=320,
                           bbox_y=240)

            # Heartbeat every iteration
            uptime = time.time() - start_time
            send_heartbeat(sock, uptime)

            time.sleep(0.1)  # 10 Hz
    except KeyboardInterrupt:
        pass
    finally:
        sock.close()

if __name__ == "__main__":
    main()
```

---

## 17. Common Issues and Debugging

### 17.1 Permission Errors

**Symptom:** `Permission denied` when accessing `/dev/spidev*`, `/dev/i2c-*`,
`/dev/ttyTHS*`, or GPIO files.

```bash
# Fix: Add user to the required groups
sudo usermod -a -G gpio,i2c,spi,dialout $USER

# Apply Jetson GPIO udev rules
sudo cp /opt/nvidia/jetson-gpio/etc/99-gpio.rules /etc/udev/rules.d/
sudo udevadm control --reload-rules && sudo udevadm trigger

# Create udev rules for SPI and I2C (if not present)
sudo tee /etc/udev/rules.d/99-peripherals.rules << 'EOF'
# SPI
SUBSYSTEM=="spidev", GROUP="spi", MODE="0660"
# I2C
SUBSYSTEM=="i2c-dev", GROUP="i2c", MODE="0660"
# UART
KERNEL=="ttyTHS*", GROUP="dialout", MODE="0660"
EOF

sudo udevadm control --reload-rules && sudo udevadm trigger

# IMPORTANT: Log out and log back in for group changes to take effect
```

### 17.2 Pinmux Conflicts

**Symptom:** A peripheral does not appear (no `/dev/spidev*`, no CAN interface)
even after Jetson-IO configuration.

```bash
# Check what Jetson-IO configured
ls /boot/device-tree/overlays/
cat /boot/extlinux/extlinux.conf | grep -i overlay

# Verify the pinmux state
sudo cat /sys/kernel/debug/tegra_pinctrl_reg | grep spi
sudo cat /sys/kernel/debug/tegra_pinctrl_reg | grep can

# Check for conflicting overlays or device tree entries
dmesg | grep -i "pinctrl\|pinmux\|conflict"

# Common cause: Two overlays trying to configure the same pin
# Solution: Use a single combined overlay (see Section 15.7)

# Check if the pin is already claimed by another driver
sudo cat /sys/kernel/debug/gpio | grep -E "gpio-(316|317|348)"
```

### 17.3 I2C Bus Hangs

**Symptom:** `i2cdetect` hangs, I2C reads return errors, or the bus becomes
unresponsive.

```bash
# Check for I2C errors in dmesg
dmesg | grep -i "i2c.*error\|i2c.*timeout\|i2c.*nak"

# Reset the I2C bus (bit-bang a clock to clear stuck SDA)
# Method 1: Unbind and rebind the controller
echo "3160000.i2c" | sudo tee /sys/bus/platform/drivers/tegra-i2c/unbind
sleep 0.5
echo "3160000.i2c" | sudo tee /sys/bus/platform/drivers/tegra-i2c/bind

# Method 2: Toggle SCL manually (if SDA is stuck low)
# This requires temporarily configuring SCL as GPIO and sending 9 clock pulses

# Prevention:
# - Use external pull-up resistors (4.7K for 400kHz, 10K for 100kHz)
# - Keep I2C wires short (<30 cm for 400 kHz)
# - Add 100nF decoupling capacitors near each I2C device
# - Avoid hot-plugging I2C devices while the bus is active
```

### 17.4 SPI Clock Issues

**Symptom:** SPI transfers return all 0xFF or all 0x00, or data is corrupted.

```bash
# Check SPI device permissions
ls -la /dev/spidev*

# Verify SPI clock configuration
# The actual clock may differ from the requested frequency due to
# the T234's clock divider. Verify with an oscilloscope if possible.

# Debug steps:
# 1. Loopback test (short MOSI to MISO)
echo -ne '\x55\xAA\x01\x02' | spi-pipe -d /dev/spidev0.0 -s 1000000 -b 4 | xxd
# Expected: 55 aa 01 02

# 2. If loopback works but device does not respond:
#    - Check SPI mode (CPOL/CPHA) matches the device datasheet
#    - Try lower clock speed (some devices fail above 1 MHz)
#    - Verify CS polarity (most devices use active-low)
#    - Check wiring: MOSI goes to device DIN, MISO to device DOUT

# 3. Check that the correct chip-select is used
ls /dev/spidev0.*
# spidev0.0 = CS0 (pin 24), spidev0.1 = CS1 (pin 26)
```

```python
# Python SPI debugging
import spidev
spi = spidev.SpiDev()
spi.open(0, 0)

# Try different modes if data is garbled
for mode in range(4):
    spi.mode = mode
    spi.max_speed_hz = 100000  # Start slow
    result = spi.xfer2([0x9F, 0x00, 0x00, 0x00])  # JEDEC ID read
    print(f"Mode {mode}: {[hex(b) for b in result]}")

spi.close()
```

### 17.5 CAN Bus Errors

**Symptom:** `cansend` fails, `candump` shows error frames, or CAN interface goes
to bus-off state.

```bash
# Check CAN interface state
ip -details link show can0
# Look for "state ERROR-ACTIVE", "ERROR-WARNING", "ERROR-PASSIVE", or "BUS-OFF"

# Check error counters
cat /sys/class/net/can0/statistics/rx_errors
cat /sys/class/net/can0/statistics/tx_errors

# View CAN error frames
candump can0,0:0,#FFFFFFFF
# Error frames have the 0x20000000 flag set

# Common issues and fixes:

# 1. "No buffer space available" -- interface not up
sudo ip link set can0 type can bitrate 500000
sudo ip link set can0 up

# 2. Bus-off recovery
sudo ip link set can0 type can restart-ms 100
# Or manual restart:
sudo ip link set can0 type can restart

# 3. Bitrate mismatch -- all nodes must use the same bitrate
# Verify with oscilloscope or CAN analyzer

# 4. Missing termination resistor
# The bus MUST have exactly two 120-Ohm termination resistors
# Measure resistance between CANH and CANL with bus unpowered:
# Expected: ~60 Ohms (two 120-Ohm in parallel)

# 5. Transceiver not powered
# Check that the CAN transceiver has both VCC and VIO connected
# Some transceivers need 5V on VIO even with 3.3V logic input

# 6. TX/RX swapped
# CAN0_DOUT (pin 31) -> Transceiver TXD
# CAN0_DIN  (pin 29) -> Transceiver RXD
```

### 17.6 Voltage Level Problems

**Symptom:** Devices do not respond, random data corruption, or SoC damage.

```
  Common voltage mistakes and fixes:

  WRONG: Connecting 5V I2C device directly to 3.3V Jetson I2C
  RIGHT: Use a bidirectional level shifter (e.g., TXB0102)

  WRONG: Connecting 5V UART (RS-232) to Jetson UART
  RIGHT: Use a 3.3V USB-to-serial adapter or MAX3232 level converter

  WRONG: Driving LEDs directly from GPIO pins
  RIGHT: Use a transistor (2N2222, MOSFET) or buffer IC (74LVC245)

  WRONG: Powering motors from the Jetson 5V pin
  RIGHT: Use a separate power supply with common GND

  Level Shifter Wiring:

  Jetson 3.3V Side                Level Shifter             5V Device Side
  +---+                           +-----------+              +---+
  |   | Pin 3 (SDA) <----------> | LV1   HV1 | <---------> | SDA |
  |   | Pin 5 (SCL) <----------> | LV2   HV2 | <---------> | SCL |
  |   | Pin 1 (3.3V) ----------> | LV    HV  | <--- 5V     |     |
  |   | Pin 6 (GND)  ----------> | GND   GND | <--- GND    | GND |
  +---+                           +-----------+              +---+
```

### 17.7 General Debugging Checklist

Use this checklist when a peripheral is not working:

```
[ ] 1. POWER: Is the device getting the correct voltage?
       Measure with multimeter at the device pins, not just at the Jetson.

[ ] 2. GROUND: Is GND shared between the Jetson and the device?
       Floating ground is the #1 cause of mysterious failures.

[ ] 3. WIRING: Are connections correct? (TX->RX, MOSI->DIN, etc.)
       Photograph your wiring and compare to the schematic.

[ ] 4. PINMUX: Is the pin configured for the correct function?
       sudo cat /sys/kernel/debug/tegra_pinctrl_reg | grep <pin_name>

[ ] 5. DEVICE PRESENCE: Can the kernel see the device?
       I2C: sudo i2cdetect -y -r <bus>
       SPI: ls /dev/spidev*
       UART: ls /dev/ttyTHS*
       CAN: ip link show can0

[ ] 6. PERMISSIONS: Does your user have access?
       ls -la /dev/<device>
       groups  # Check current user's groups

[ ] 7. DRIVER: Is the correct driver loaded?
       lsmod | grep <driver_name>
       dmesg | grep <driver_name>

[ ] 8. KERNEL LOG: What does dmesg say?
       dmesg | tail -50
       dmesg | grep -i error

[ ] 9. SIGNAL INTEGRITY: Are signals clean?
       Use an oscilloscope or logic analyzer to verify:
       - Clock frequency and duty cycle
       - Data transitions are clean (no ringing/overshoot)
       - Proper voltage levels (3.3V high, <0.8V low)

[ ] 10. SOFTWARE: Is the user-space tool/library compatible?
        Check library version matches the L4T / JetPack version.
```

### 17.8 Useful Debugging Commands Reference

```bash
# ---- System Info ----
cat /etc/nv_tegra_release          # L4T version
uname -r                           # Kernel version
cat /proc/device-tree/model        # Board model
jetson_release                     # JetPack version (if jetson-stats installed)

# ---- GPIO ----
sudo cat /sys/kernel/debug/gpio    # All GPIO states
gpiodetect                         # List GPIO chips
gpioinfo gpiochip0                 # List all lines on chip 0

# ---- I2C ----
i2cdetect -l                       # List I2C buses
sudo i2cdetect -y -r 1             # Scan bus 1
sudo i2cget -y 1 0x68 0x75         # Read register

# ---- SPI ----
ls -la /dev/spidev*                # List SPI devices
cat /sys/module/spidev/parameters/bufsiz  # Max transfer size

# ---- UART ----
ls -la /dev/ttyTHS*                # List Tegra UART devices
sudo stty -F /dev/ttyTHS0 -a      # Show UART config

# ---- CAN ----
ip -details link show can0         # CAN config and state
candump -ta can0                   # Live CAN traffic
cangen can0 -I i -L 8 -D r -g 100 # Generate test traffic

# ---- PWM ----
ls /sys/class/pwm/                 # List PWM chips
cat /sys/class/pwm/pwmchip0/npwm   # Number of channels

# ---- Device Tree ----
ls /proc/device-tree/              # Browse live device tree
dtc -I fs /proc/device-tree 2>/dev/null | head -100  # Decompile

# ---- Pinmux ----
sudo cat /sys/kernel/debug/tegra_pinctrl_reg | head -20

# ---- Kernel Modules ----
lsmod                              # Loaded modules
modinfo <module_name>              # Module details

# ---- Power ----
cat /sys/bus/i2c/drivers/ina3221/*/hwmon/hwmon*/in1_input  # Rail voltage
sudo tegrastats                    # Real-time power/thermal/CPU stats
```

### 17.9 Performance Measurement Script

```python
#!/usr/bin/env python3
"""Measure I/O latency and throughput for all peripheral interfaces."""

import time
import os
import statistics

def measure_gpio_latency(iterations=1000):
    """Measure GPIO toggle latency using sysfs."""
    gpio_path = "/sys/class/gpio/gpio348"
    if not os.path.exists(gpio_path):
        os.system("echo 348 > /sys/class/gpio/export 2>/dev/null")
        time.sleep(0.1)
        os.system(f"echo out > {gpio_path}/direction")

    value_file = f"{gpio_path}/value"
    times = []

    for _ in range(iterations):
        t0 = time.perf_counter_ns()
        with open(value_file, 'w') as f:
            f.write("1")
        with open(value_file, 'w') as f:
            f.write("0")
        t1 = time.perf_counter_ns()
        times.append((t1 - t0) / 1000.0)  # Convert to microseconds

    avg = statistics.mean(times)
    std = statistics.stdev(times)
    mn  = min(times)
    mx  = max(times)
    print(f"GPIO toggle: avg={avg:.1f}us  std={std:.1f}us  "
          f"min={mn:.1f}us  max={mx:.1f}us")

def measure_i2c_latency(bus_num=1, addr=0x68, reg=0x75, iterations=1000):
    """Measure I2C single-byte read latency."""
    from smbus2 import SMBus
    times = []

    with SMBus(bus_num) as bus:
        for _ in range(iterations):
            t0 = time.perf_counter_ns()
            bus.read_byte_data(addr, reg)
            t1 = time.perf_counter_ns()
            times.append((t1 - t0) / 1000.0)

    avg = statistics.mean(times)
    std = statistics.stdev(times)
    print(f"I2C read:    avg={avg:.1f}us  std={std:.1f}us  "
          f"min={min(times):.1f}us  max={max(times):.1f}us")

def measure_spi_latency(iterations=1000):
    """Measure SPI 4-byte transfer latency."""
    import spidev
    spi = spidev.SpiDev()
    spi.open(0, 0)
    spi.max_speed_hz = 10000000
    spi.mode = 0

    times = []
    for _ in range(iterations):
        t0 = time.perf_counter_ns()
        spi.xfer2([0x00, 0x00, 0x00, 0x00])
        t1 = time.perf_counter_ns()
        times.append((t1 - t0) / 1000.0)

    spi.close()
    avg = statistics.mean(times)
    std = statistics.stdev(times)
    print(f"SPI xfer:    avg={avg:.1f}us  std={std:.1f}us  "
          f"min={min(times):.1f}us  max={max(times):.1f}us")

if __name__ == "__main__":
    print("=== Peripheral I/O Latency Measurement ===")
    print(f"Running on: {os.uname().nodename}")
    print()

    try:
        measure_gpio_latency()
    except Exception as e:
        print(f"GPIO test skipped: {e}")

    try:
        measure_i2c_latency()
    except Exception as e:
        print(f"I2C test skipped: {e}")

    try:
        measure_spi_latency()
    except Exception as e:
        print(f"SPI test skipped: {e}")
```

---

*End of Guide*

**Summary of Key Takeaways:**

1. The 40-pin header is your primary interface for connecting sensors and actuators
   to the Orin Nano. All signals are 3.3V logic -- never exceed this.

2. Use Jetson-IO for quick pinmux configuration. Use custom device tree overlays
   for production deployments.

3. For GPIO, prefer `libgpiod` (C) for performance and `Jetson.GPIO` (Python) for
   rapid prototyping. Always use interrupt-driven GPIO over polling.

4. I2C is best for slow, multi-device sensor buses. SPI is best for high-speed,
   point-to-point connections. UART is best for GPS, debug consoles, and MCU links.

5. CAN bus requires an external transceiver and proper termination. Use SocketCAN
   for seamless Linux integration.

6. Always use external power supplies for motors and high-current loads. Never draw
   more than 1 mA from a GPIO pin.

7. For real-time applications, consider `PREEMPT_RT`, CPU core isolation, and
   kernel-space drivers.

8. When debugging, start with the checklist in Section 17.7 and work through it
   systematically.
