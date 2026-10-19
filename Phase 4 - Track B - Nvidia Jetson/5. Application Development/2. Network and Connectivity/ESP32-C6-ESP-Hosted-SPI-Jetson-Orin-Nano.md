# ESP32-C6 Wi-Fi over SPI: current Linux driver and legacy migration

**Source review: 18 October 2026.** This lesson explains how to integrate a wireless coprocessor with Jetson. No ESP firmware, Jetson kernel module, RF test or physical wiring was executed during this course revision. Complete the [CPU transport lab](Localhost-Network-Lab.md) first, then use this as a hardware integration plan.

**Prerequisites:** [peripheral access](../1.%20Peripheral%20Access/Guide.md), Linux driver binding, and the exact carrier schematic. **Outcome:** explain the separate firmware, kernel, pinmux and network contracts; produce a repeatable bring-up record if hardware is available.

## 1. Choose the correct repository generation

The old `esp-hosted/esp_hosted_ng/...` paths no longer exist on current `master`. The official umbrella repository sends new Linux FullMAC work to **esp-hosted-linux**; NG/FG remain on `legacy/fg-ng`. [Official migration directory](https://github.com/espressif/esp-hosted/blob/a0422923d01fa5cd5f4987885fe743656398b7c5/README.md).

| Path | Source checked | Intended use |
|---|---|---|
| Current ESP-Hosted-Linux | `9b56ff96a1bc95acaa68d8023794b8353701c5e3` (16 Oct 2026) | Current Linux Wi-Fi integration; device-tree GPIO descriptors |
| AI-HPC Jetson NG fork | `ce36dec9294988e598f04e84c9d28092ca9e9786` (20 Apr 2026) | Reproduce/analyze an attributed legacy Jetson port |
| ESP-Hosted-MCU | Separate active upstream | RPC/ESP-IDF API architecture; not an interchangeable NG kernel driver |

Use matching host and ESP firmware from one generation. The current driver's GPIO/reset behavior differs from the legacy fork. A source revision is more precise than the generic product name.

## 2. Draw the data and control paths

```text
Linux application → TCP/IP → wlanX → ESP-Hosted Linux kernel driver
                                      |
                           SPI + handshake/data-ready
                                      |
                          ESP firmware → Wi-Fi MAC/PHY → AP
```

The host still runs its IP stack. The ESP supplies Wi-Fi functions and transfers packets/control messages across SPI. A 40 MHz SPI clock is not 40 Mbyte/s: single-bit SPI has a raw 40 Mbit/s wire rate in each active direction, before transport overhead, idle periods and radio work.

The ESP32-C6 supports the current driver's SPI and SDIO paths. Its USB Serial/JTAG connector is not the native ESP-Hosted USB bulk transport listed for ESP32-S31. A USB flashing port does not imply the same device offers every USB network mode. [Supported hardware](https://github.com/espressif/esp-hosted-linux/blob/9b56ff96a1bc95acaa68d8023794b8353701c5e3/docs/reference/supported-hardware.md).

## 3. Prepare an explicit wiring contract

The current firmware defaults for C6 are:

| Signal | ESP32-C6 default | Host requirement |
|---|---:|---|
| SCLK | GPIO6 | SPI clock output |
| MOSI | GPIO7 | Host output to ESP input |
| MISO | GPIO2 | ESP output to host input |
| CS | GPIO10 | Correct chip-select and inactive pull-up |
| Handshake | GPIO3 | Interrupt-capable input on host |
| Data ready | GPIO4 | Interrupt-capable input on host |
| Reset/enable | EN/RST | Optional controlled reset, matched polarity |
| Ground | GND | Common reference |

These are firmware defaults, not a guarantee about your board routing. Inspect the active `sdkconfig` and both schematics. Use compatible 3.3 V logic and a supported board power input. Do not connect two power sources together merely because both boards expose a 5 V pin. [Official wiring reference](https://github.com/espressif/esp-hosted-linux/blob/9b56ff96a1bc95acaa68d8023794b8353701c5e3/docs/getting-started/hardware-setup.md).

For the reference Orin Nano carrier, the header's SPI signals are associated with physical pins 19/21/23/24, but confirm the exact carrier/BSP mapping before wiring. Physical header numbers, SoC pad names, Linux SPI bus numbers, legacy global GPIO numbers and `gpiochip` offsets are different namespaces. `471` from an old fork is not a portable GPIO descriptor.

Use [Jetson peripheral discovery](../1.%20Peripheral%20Access/Guide.md), inspect the live device tree and driver binding, and record the selected controller/chip-select. A `/dev/spidev*` node proves a generic device was instantiated; it does not prove waveform timing or successful ESP communication.

## 4. Build a matched current firmware pair

The following sequence is source-reviewed and **not executed here**. Use a new checkout and allow for ESP-IDF/toolchain disk space:

```bash
git clone https://github.com/espressif/esp-hosted-linux.git
cd esp-hosted-linux
git checkout --detach 9b56ff96a1bc95acaa68d8023794b8353701c5e3
cd esp/esp_driver
./setup.sh
. ./esp-idf/export.sh
cd network_adapter
idf.py set-target esp32c6
idf.py menuconfig
idf.py build
```

Select SPI in the example transport configuration, verify C6 GPIO assignments, and save `sdkconfig` plus the dependency lock. At this checked revision, `.env` pins ESP-IDF `v6.1` commit `fff9895c82d744c7237be8847347bdd1b07c6643`. Do not substitute Zigbee's recommended IDF 5.5.4 merely because both projects use ESP32-C6. [Setup and build source](https://github.com/espressif/esp-hosted-linux/blob/9b56ff96a1bc95acaa68d8023794b8353701c5e3/docs/getting-started/build-and-flash.md), [exact IDF pin](https://github.com/espressif/esp-hosted-linux/blob/9b56ff96a1bc95acaa68d8023794b8353701c5e3/esp/esp_driver/.env).

Identify the actual flashing endpoint with `ls -l /dev/serial/by-id/` and USB enumeration. Either a USB-UART bridge or the chip's Serial/JTAG port may be usable depending on the board. Neither `ttyUSB0` nor `ttyACM0` is universally preferred. Use that path with `idf.py -p PORT flash`; close other owners before flashing or binary transport use.

## 5. Integrate the Jetson kernel driver

Build `host/` against the **running target kernel's prepared build tree**. The checked Makefile accepts `make target=spi KERNEL=/path/to/kernel/build` and cross-build variables `ARCH=arm64` and `CROSS_COMPILE`. Matching aarch64 architecture alone is insufficient: kernel configuration, generated headers, symbol versions and signing policy can matter.

The current SPI node uses `compatible = "espressif,esp32-spi"`, `reg`, `spi-max-frequency`, `spi-cpol`, required `handshake-gpios`/`data-ready-gpios`, and optional `reset-gpios`. The GPIO properties reference the actual controller and offsets. They are not legacy global integers. Do not paste the upstream sample `&pio` node into a Tegra tree; adapt it using the carrier's pinctrl and GPIO definitions. [Current porting contract](https://github.com/espressif/esp-hosted-linux/blob/9b56ff96a1bc95acaa68d8023794b8353701c5e3/docs/porting.md).

The ESP driver and generic `spidev` cannot own the same chip-select simultaneously. Configure a proper device-tree node and check `/sys/bus/spi/devices/.../driver`; an ad hoc unbind during a debug session is not persistent board configuration.

The current driver does **not** accept the old `resetpin=`, `spi_handshake_gpio=` or `spi_dataready_gpio=` module parameters. The optional reset descriptor or acknowledged in-band reset protocol owns recovery. Inspect `modinfo` for the built module before loading it.

## 6. Keep historical evidence attributed

The [AI-HPC fork README at the April revision](https://github.com/ai-hpc/jetson-esp-hosted/blob/ce36dec9294988e598f04e84c9d28092ca9e9786/esp_hosted_ng/host/README.md) reports a particular Orin Nano integration using legacy GPIOs 471/433, `resetpin=-1`, SPI mode 2 and a 10 MHz cap. Those are the fork author's recorded settings, not measurements repeated by this course or values validated for current JetPack.

Use its source to understand the port, including kernel-driver changes and the helper's unbind/load actions. Preserve its exact version if reproducing it. Do not combine its module parameters with the current driver or assume its manual-reset recovery is the current protocol.

## 7. Validate in layers and test recovery

1. **Electrical/bus:** verify power, signal direction, mode, clock, interrupts and a successful firmware capability exchange.
2. **Driver:** inspect actual `iw dev`, `ip -br link` and kernel errors. `wlanX` numbering is assigned at runtime.
3. **Network:** associate with your AP, acquire an address, resolve a name, then exchange application messages.
4. **Recovery:** reset the ESP during traffic, observe interface/session loss, reconnect and check that stale commands expire.
5. **Coexistence:** test the actual simultaneous services and traffic rates, including scans and restart behavior.

The current driver supports `radio_service=none`, `bt`, `154` or `bt+154` when the matching firmware advertises those services. C6 RCP is not enabled by its defaults; inspect the firmware options. An active hosted RCP appears as `/dev/esp_rcp0`, and one process owns the stream. Adding services and removing them have different restart behavior. [Current RCP guide](https://github.com/espressif/esp-hosted-linux/blob/9b56ff96a1bc95acaa68d8023794b8353701c5e3/docs/guides/thread-rcp.md).

The chip has one shared RF path. Software support for several services does not mean they transmit or receive independently at the same instant. A second C6/H2 can isolate radio scheduling, at the cost of more hardware. [C6 coexistence constraints](https://docs.espressif.com/projects/esp-idf/en/v5.5.4/esp32c6/api-guides/coexist.html).

## 8. Completion artifact

Without hardware, submit an annotated migration table and the [passing CPU lab](Localhost-Network-Lab.md). With hardware, add exact commits, BSP/kernel, module build log, `sdkconfig`, device tree, wiring diagram and reconnect measurements. Leave unexecuted stages marked unexecuted.

**Diagnostic answer:** a SPI device node, a loaded module, a registered Wi-Fi interface and a successful application exchange establish four different things. Report each explicitly; none alone proves the other three.
