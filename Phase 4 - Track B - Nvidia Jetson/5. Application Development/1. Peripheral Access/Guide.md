# Peripheral access: turn device bytes into trustworthy observations

**Goal:** connect a Linux application to one peripheral, preserve its units and timing, and handle missing or invalid data. Plan four 60–90-minute sessions. Begin with the CPU protocol lab; use physical I/O only after identifying your module, carrier, wiring and installed software.

A robot application should receive something like “distance 1.2 m, captured at time T, sequence 127, valid,” rather than an unexplained integer from a device file. The interface code must establish what the value means, when it was measured, and what happens when communication fails.

This module covers runtime access. Use [carrier bring-up](../../2.%20Custom%20Carrier%20Board%20Design%20and%20Bring-Up/Guide.md) and [L4T customization](../../3.%20L4T%20Customization/Guide.md) for pinmux, boot configuration and driver binding. The [GPIO/SPI/I2C/CAN deep dive](../../1.%20Nvidia%20Jetson%20Platform/Orin-Nano-GPIO-SPI-I2C-CAN/Guide.md) provides the electrical and transaction details.

## 1. Begin with a working CPU observation

From the course repository root, run:

```bash
python3 -m unittest discover -s "Phase 4 - Track B - Nvidia Jetson/1. Nvidia Jetson Platform/Orin-Nano-GPIO-SPI-I2C-CAN/labs" -p "test_*.py" -v
```

The [sensor protocol lab](../../1.%20Nvidia%20Jetson%20Platform/Orin-Nano-GPIO-SPI-I2C-CAN/labs/README.md) compiles actual C against Linux's CAN headers and exchanges binary records with Python. Its 11 tests include signed scaling, message length/flags, wrapping sequence numbers and a real local datagram socket. It requires a C compiler but no CAN adapter.

The local socket establishes application encoding/transport behavior. It does not simulate CAN arbitration, transceiver voltages or bus errors. Keep that distinction when you later replace the local transport with SocketCAN.

**Hand calculation:** the payload `01 7f 06 ff b0 04 0f 00` contains version 1, sequence 127, signed speed −250 mm/s, distance 1,200 mm and sender age 15 ms. Speed is −0.25 m/s; distance is 1.2 m. Interpreting `06 ff` as an unsigned value gives 65,286 and an implausible result. A packet can arrive successfully and still be decoded incorrectly.

## 2. Discover the device before addressing it

Record the board revision, BSP, kernel, interface driver and library/tool version. Useful read-only discovery commands include:

```bash
uname -r
gpiodetect
gpioinfo --help
i2cdetect -l
lsusb
ip -details link show
```

Some tools may need installation through the target's supported package environment. `i2cdetect -l` lists adapters; it does not probe every attached device. Do not replace it with a bus scan as a universal identification method: some devices react badly to arbitrary probe transactions.

Keep the following identities separate:

| Identity | Meaning |
| --- | --- |
| Connector and physical pin | Where the wire connects on this carrier revision |
| Board signal/net name | The schematic's connection and electrical domain |
| SoC pad/pinmux function | The hardware signal/function selected for that pad |
| Linux controller and relative offset | The runtime GPIO address exposed by that driver |
| Peripheral protocol address | An I2C address, SPI chip-select, CAN identifier, or device-specific address |

There is no universal rule that “header pin 7 means gpiochip0 line 43.” Chip numbering and line information can change with the kernel/device tree. A line labeled unused is only unclaimed by a current consumer; it does not prove that driving it is electrically appropriate. Match the actual carrier schematic, pinmux and controller mapping. [NVIDIA Orin NX/Nano adaptation](https://docs.nvidia.com/jetson/archives/r39.2/DeveloperGuide/HR/JetsonModuleAdaptationAndBringUp/JetsonOrinNxNanoSeries.html).

Record a binding table for your selected sensor: device path, driver, bus parameters, voltage, protocol revision, units and expected sample cadence. Keep this configuration separate from the decoding algorithm so a board revision does not silently change measurement meaning.

## 3. Read a GPIO with the matching libgpiod API

The Linux GPIO character-device API is the current userspace interface. Prefer an existing subsystem driver when it already expresses the device, such as an input button, LED, regulator or IIO sensor. GPIO is not a replacement for every kernel subsystem. [Linux GPIO guidance](https://docs.kernel.org/driver-api/gpio/using-gpio.html).

libgpiod 1.x and 2.x have different command and Python APIs. Check `gpiodetect --version` and each installed tool's help. These are **2.x command templates**, after you have assigned `GPIO_CHIP` to the verified chip and `GPIO_LINE` to its verified relative offset:

```bash
: "${GPIO_CHIP:?Set the verified gpiochip path or identifier}"
: "${GPIO_LINE:?Set the verified chip-relative line offset}"
gpioinfo --chip "$GPIO_CHIP" "$GPIO_LINE"
gpioget --chip "$GPIO_CHIP" "$GPIO_LINE"
```

The shell checks stop if a required value is missing; literal angle-bracket placeholders are not valid shell arguments. In 2.x, `--chip` selects the controller. Old examples using positional `gpioget gpiochip0 43` describe a different CLI generation.

For an explicitly wired output exercise, the 2.x form is `gpioset --chip "$GPIO_CHIP" "$GPIO_LINE=1"`. It holds the request while the process runs. Closing the request releases ownership; do not assume the line remains at that value afterward. Logical active/inactive and electrical high/low differ when active-low is configured. See [libgpiod 2.3 command examples](https://libgpiod.readthedocs.io/en/v2.3/gpio_tools.html).

This complete **hardware-dependent** Python 2.x example reads a verified input:

```python
import os
import gpiod
from gpiod.line import Direction

chip = os.environ["GPIO_CHIP"]
line = int(os.environ["GPIO_LINE"])
with gpiod.request_lines(
    chip,
    consumer="course-input",
    config={line: gpiod.LineSettings(direction=Direction.INPUT)},
) as request:
    print(request.get_value(line))
```

It retains the request through the read and closes it afterward. Permission errors, an already-owned line or an incorrect offset should be reported, rather than converted to a plausible zero. This example was source-checked against [the 2.3 request API](https://libgpiod.readthedocs.io/en/v2.3/python_misc.html); physical GPIO execution was not available in this review.

For edge events, record the configured timestamp clock and sequence information. A default monotonic timestamp is not a UTC wall-clock timestamp. Mechanical switch debounce, GPIO edge time and the time a sensor finished a measurement are separate concerns.

## 4. Choose the bus transaction that the sensor requires

**I2C:** a register read often needs a write of the register address followed by a read under a repeated START. Separate `write()` and `read()` calls can insert a STOP, changing the transaction. Use the Linux combined-transfer interface when the device requires it. Preserve address width, byte order, signedness and conversion scale. A successful register read does not mean the conversion is fresh; read the sensor's data-ready/status contract.

**SPI:** state clock polarity/phase, bit order, maximum clock, word size and chip-select behavior. A command and response may need one transfer message with chip-select held. SPI can transmit and receive simultaneously; do not add TX and RX byte counts as if both necessarily consume separate clock intervals. Dummy response bytes and multi-transfer commands depend on the device.

**UART:** it is a byte stream. One read can contain half a message or several messages. Use explicit framing, maximum length, version, integrity checks where appropriate, a receive deadline and resynchronization. The [native framed-telemetry lab](../../../Phase%202%20-%20Embedded%20Systems/2.%20Embedded%20Software/labs/Framed-Telemetry.md) demonstrates fragmented input, malformed lengths and a C parser. It is a better starting point than assuming a complete JSON line arrives in every read.

**CAN:** Linux SocketCAN exposes a network interface, while a physical controller still needs the appropriate transceiver, wiring, termination and configuration. CAN identifiers describe arbitration/message identity, not automatically a destination device. Classic CAN and CAN FD have different frame capacities and flags. The lab intentionally uses one strict Classic CAN teaching format.

For the Orin Nano developer carrier, use its own J17 CAN connector documentation. Do not copy AGX Orin's J30 pin numbers into a Nano wiring diagram. The deep dive links the board-specific source and explains the distinction.

**Timing example:** 100 UART bytes with 8N1 framing at 115,200 baud take `100 × 10 / 115200 ≈ 8.68 ms` on the wire. The same message at 20 Hz occupies about 17.4% of that direction's nominal line time. Processing, buffering, retries and turnaround add latency. A high CPU clock cannot shorten the wire's bit period.

## 5. Preserve freshness across the application boundary

An observation should carry a schema version, sensor identity, sequence/epoch, measurement units, validity and timing information. A robot may also need coordinate frame, calibration identity and uncertainty.

Sequence 255 followed by 0 can be normal wraparound for an eight-bit counter. The lab uses a half-range rule to distinguish forward motion from old/reordered values under a stated assumption about maximum gaps. A reboot must create a new session/epoch or reset the receiver explicitly; otherwise a restarted counter can be confused with an old packet.

A sender's “age 15 ms” field only measures the age the sender knows about. Unknown network/queue delay afterward still exists. One-way age across devices requires a clock relationship or a bound justified by the communication protocol. Use local receipt and processing timestamps for intervals that actually share a clock. For multi-sensor alignment, continue to [camera-time fusion](../../../Phase%203%20-%20Artificial%20Intelligence/Track%20A%20-%20Hardware%20and%20Edge%20AI/4.%20Sensor%20Fusion/Camera-Time-Fusion-Lab/Guide.md).

Keep acquisition callbacks short. Put validated observations in a bounded queue with an explicit policy: keep all for a lossless logger, or replace older pending work for a latest-observation controller when that is the intended design. Record drops. An unbounded queue can preserve every message while making decisions increasingly stale.

For actuation, define command units/ranges, expiry, acknowledgement and a fallback owned by the receiving controller. A Jetson process sending PWM-like commands is not itself a guarantee of deterministic motor timing. Use a suitable controller for that requirement, and test the loss-of-command path.

## 6. Handle USB, storage and permissions deliberately

USB device discovery should use stable identities where available, not assume `/dev/ttyUSB0` always names the same sensor after reconnection. Match the intended vendor/product/serial/interface and confirm the selected device. Keep reconnects as state changes: discard incomplete frames, establish the new session and require fresh data.

For storage logging, distinguish “write accepted by the process,” “data flushed to the filesystem,” and the durability promised by the full hardware stack. The [OTA crash lab](../../1.%20Nvidia%20Jetson%20Platform/Orin-Nano-OTA-Deep-Dive/labs/README.md) demonstrates real process-crash boundaries with fsync/rename, while explicitly separating them from physical power-loss testing.

Grant device access to the application through the target's appropriate group/udev/service configuration. Running every application as root conceals missing permissions and expands the consequences of a bug. Record the exact access required; a permission error is a configuration observation, not a reason to silently substitute synthetic sensor data.

## 7. Build the peripheral adapter in four sessions

| Session | Work | Evidence |
| --- | --- | --- |
| 1 | Run the native C/Python protocol exercise and decode the golden payload by hand | Native test log and unit conversion |
| 2 | Identify a target interface or document a hardware-pending binding table | Board/BSP identity and mapping source |
| 3 | Add bounded reads, session/sequence checks and explicit invalid/stale handling | Valid, truncated, duplicate, stale and restart cases |
| 4 | Integrate one observation into a logger or controller-facing queue | Receipt/measurement clocks, queue policy and failure behavior |

The CPU route is complete without purchasing hardware. A physical extension adds a real sensor capture and reconnect/failure evidence; it does not replace the parser checks.

## 8. Check your understanding

1. **A GPIO is unused. May you drive it?** Only after checking its electrical function, pinmux, board wiring and ownership requirements.
2. **A sensor read returned zero. Is zero invalid?** Not necessarily. The protocol needs a separate status/validity contract; communication errors must not be disguised as a valid zero.
3. **Why can two I2C calls differ from one combined transaction?** A STOP between calls may change the device's register-selection behavior.
4. **Why can received CAN data still be stale?** It may have waited in a sender or receiver queue, and receipt time is not necessarily measurement time.
5. **Why does a working local socket test not prove CAN works?** It bypasses CAN arbitration, device drivers, wiring and transceivers.
6. **What should happen after reconnect?** Reset transport/parser state, establish session identity, and accept only a fresh valid observation under the new session.

Continue with [network connectivity](../2.%20Network%20and%20Connectivity/Guide.md), then connect the adapter to the [ROS 2 course](../6.%20ROS2/Guide.md).
