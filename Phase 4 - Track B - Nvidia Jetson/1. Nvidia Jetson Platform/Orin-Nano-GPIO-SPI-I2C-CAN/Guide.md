# Connect sensors and controllers to Jetson

A robot's AI model depends on measurements that arrive with the right units, identity, and timing. A successful bus read establishes only one part of that contract. This course takes you from electrical and pin identification through GPIO, I²C, SPI, UART, CAN, and PWM to a testable sensor-to-decision boundary.

**Outcomes:** choose the appropriate Linux interface; distinguish a header pin from a GPIO offset; explain a combined bus transaction; decode a versioned sensor message; and design failure handling without confusing communication success with fresh data. Plan five 60–90-minute sessions. Start with the [CPU protocol lab](labs/README.md); physical-device exercises are optional extensions.

## 1. Identify the exact hardware before connecting it

Record the module, carrier-board revision, Jetson Linux release, sensor part number, and power arrangement. A developer kit's connector numbers are properties of its carrier, not universal properties of every module. The [platform guide](../Guide.md) explains the supported software combinations.

The Orin Nano developer kit's J12 expansion signals use 3.3 V levels. The carrier specification distinguishes directly connected open-drain signals from signals behind weak bidirectional level translators. Do not infer a general LED-drive current, 5 V input tolerance, or a safe pull resistor from “3.3 V GPIO.” Check the actual signal path and peripheral's input requirements. [NVIDIA carrier specification, connector and electrical tables](https://developer.nvidia.com/downloads/assets/embedded/secure/jetson/orin_nano/docs/jetson_orin_nano_devkit_carrier_board_specification_sp.pdf).

**Important board distinction:** Orin Nano/NX developer-kit CAN uses J17 (`CAN_RX` pin 1, `CAN_TX` pin 2). The AGX Orin J30 pin 29/31 mapping is a different carrier. The Nano controller signals require an appropriate external CAN transceiver; they are not CANH/CANL. [NVIDIA CAN platform table](https://docs.nvidia.com/jetson/archives/r39.2/DeveloperGuide/HR/ControllerAreaNetworkCan.html#jetson-platform-details).

Before powering a circuit, create a connection table containing connector pin, signal name, direction, voltage domain, destination, and power-on state. Confirm pin-1 orientation from the drawing. Use a common reference ground where required by the interface; account for isolation when the system requires it. Motor and servo supply current belongs in an appropriately designed power path, not a GPIO output.

**Check:** a module datasheet says “CAN supported.” What remains unknown? The carrier routing, transceiver, connector, pinmux, driver, physical bus, protocol, and application behavior.

## 2. Keep pinmux, driver ownership, and numbering separate

A pad may support several alternate functions. The selected function, electrical configuration, device-tree description, driver binding, and application permission must agree. An existing `/dev` node alone does not prove the signal reaches the intended header.

On supported developer kits, Jetson-IO configures expansion-header functions and boot-time device-tree changes. It has a terminal interface; X forwarding is not a prerequisite. Use its choices for the actual installed release and carrier. Custom products may need the platform pinmux/BCT workflow and a board-specific device tree. [Jetson expansion-header configuration](https://docs.nvidia.com/jetson/archives/r39.2/DeveloperGuide/HR/ConfiguringTheJetsonExpansionHeaders.html).

Do not interchange these identifiers:

| Identifier | Meaning |
| --- | --- |
| Physical connector pin | Position on a particular carrier connector |
| SoC pad/signal name | Hardware signal selected by pinmux |
| GPIO chip and line offset | Kernel character-device identity |
| Legacy global GPIO number | Old sysfs numbering; not a portable identity |
| `/dev/i2c-N`, `/dev/spidevB.C` | Discovered bus/device instance |

Inspect devices and ownership first: `gpiodetect`, the installed version's `gpioinfo --help`, `i2cdetect -l`, and sysfs driver links are useful starting points. Line names can be absent or repeated; retain the verified chip identity and offset. Do not hard-code a global base such as 300 and assume it survives a kernel change.

Use the [device-tree exercise](../Orin-Nano-Kernel-Internals/labs/README.md) to learn compilation and overlay resolution before modifying a board. A syntactically valid overlay can still describe the wrong hardware.

## 3. Use GPIO for the right job

Prefer the existing subsystem driver for a button, LED, PWM output, sensor, regulator, or serial bus when one matches the device. For appropriate userspace GPIO work, use the character-device interface through a maintained library. The current v2 ABI identifies lines by chip-relative offsets and gives a line request a lifetime of its own. Legacy sysfs GPIO examples are not the starting point for this course. [Linux GPIO userspace interface](https://www.kernel.org/doc/html/latest/userspace-api/gpio/chardev.html).

libgpiod 1.x and 2.x Python APIs and command syntax differ. Inspect the installed version before copying an example. In the documented 2.x API, a request uses `gpiod.request_lines(...)` with `LineSettings`; values are logical `ACTIVE`/`INACTIVE`, and active-low configuration changes their electrical meaning. Retain the request while using the line. Do not assume the output remains driven after the requesting process closes it. [libgpiod 2.3 request API](https://libgpiod.readthedocs.io/en/v2.3/python_misc.html).

A button exercise should predict the idle state, press state, debounce behavior, and process-exit behavior before running. Event timestamps and application wake-up timestamps measure different events. In the v2 ABI the default event clock is monotonic; do not subtract it from wall-clock time. Sequence gaps can reveal lost events. [GPIO event fields and clocks](https://www.kernel.org/doc/html/latest/userspace-api/gpio/chardev.html#struct-gpio-v2-line-event).

A Python loop calling `sleep()` does not produce a guaranteed pulse train. Use hardware timers/PWM or an MCU when the peripheral requires bounded timing. An application reacting to an edge can still be delayed after the edge has been timestamped.

## 4. Read I²C devices as transactions

I²C uses an addressed shared bus. Pull-ups, bus capacitance, voltage compatibility, clock stretching, supported speed, and device address configuration all matter. There is no universal “4.7 kΩ at 400 kHz” resistor choice independent of the circuit.

Discover the adapter with `i2cdetect -l` and sysfs. A full address scan is a series of real transactions; consult the device documentation before probing an unfamiliar bus. Prefer reading a documented identification register once the address and transaction format are known. Do not force access to a device already owned by a kernel driver.

Many register reads require a write of the register address followed by a repeated START and a read, with no intervening STOP. Separate `write()` and `read()` calls are not equivalent. The Linux `I2C_RDWR` ioctl supports a combined message set when the adapter supports `I2C_FUNC_I2C`. An SMBus helper is appropriate only when its protocol matches the device. [Linux I²C userspace transactions](https://docs.kernel.org/i2c/dev-interface.html).

**Worked example:** a hypothetical sensor returns bytes `FF 9C` as a signed 16-bit big-endian integer. The raw value is −100. At 0.01 °C/count, the result is −1.00 °C. Reversing the bytes before applying the sign gives a different value; reading two independent byte registers may also mix samples if the device does not latch them. Check byte order, signedness, scaling, and snapshot semantics in the sensor datasheet.

At 400 kbit/s, a simplified transaction with write address, register byte, read address, and six data bytes contains nine nine-clock units: about 202.5 μs, before extra timing and stretching. That is a bus-time estimate, not a promise of 202.5 μs application latency. Bus arbitration, scheduling, retries, and sensor conversion time remain.

## 5. Treat SPI configuration as part of the protocol

SPI requires agreement on clock polarity/phase, bit order, word length, chip-select behavior, maximum frequency, and command/response format. A peripheral may return dummy bytes before valid data. A loopback jumper tests one local path; it does not validate a sensor command or conversion.

Linux `spidev` offers synchronous userspace transfers. `SPI_IOC_MESSAGE(N)` groups transfers and can preserve chip select across a command/response sequence. Separate reads/writes may deassert chip select between operations. A requested speed can be rounded by the controller. [SPI userspace API](https://docs.kernel.org/spi/spidev.html).

**Worked example:** a 3-byte full-duplex transfer at 1 MHz requires 24 μs of clocking, assuming eight bits per byte and no gaps. It does not require 48 μs merely because both transmit and receive buffers are present: both directions shift concurrently. Controller overhead and actual waveform gaps add to elapsed time.

Do not put a generic `compatible = "spidev"` into a new device tree and assume it is a valid device binding; current upstream documentation requires an appropriate device identity or a deliberate driver-override workflow. Prefer a real device driver when available. [SPI binding requirements](https://docs.kernel.org/spi/spidev.html#device-creation-driver-binding).

For debugging, record the requested mode/speed, transmitted bytes, received bytes, and measured waveform. An all-ones read can result from an undriven MISO line. SPI has no universal address acknowledgement, so a successful syscall does not prove a slave answered.

## 6. Give UART bytes a frame format

Match voltage standard before baud rate: a TTL UART header is not an RS-232 connector. Match baud, data bits, parity, stop bits, and flow control. Determine whether the port is already a console or service endpoint before changing it.

At 115,200 baud with 8N1, each data byte uses ten serial bits. The ideal payload-byte ceiling is 11,520 bytes/s before application framing and idle gaps. Sending 100 bytes therefore needs at least about 8.68 ms on that link. A configured baud rate is not an application throughput measurement.

A serial stream can split or merge application messages across reads. Add framing, length limits, integrity checks, and an explicit resynchronization rule. A single `read(64)` is not a promise to return one complete 64-byte message. Handle partial writes, timeouts, disconnects, and rebooted senders.

Complete the [native framed-telemetry lab](../../../Phase%202%20-%20Embedded%20Systems/2.%20Embedded%20Software/labs/Framed-Telemetry.md) for COBS framing, CRC validation, and fragmented input. Then carry the same decoded message contract over a real UART and compare the receive log with the source log.

## 7. Use CAN with physical and application contracts

CAN separates controller logic from the transceiver and bus wiring. The Nano/NX platform table exposes one controller. Match the transceiver's logic voltage and physical-layer capability to the chosen mode; a Classic CAN transceiver is not automatically suitable for a CAN FD data rate. Configure actual bus termination and topology from the network design.

SocketCAN exposes CAN interfaces through Linux sockets. Classic CAN data frames carry up to eight payload bytes; CAN FD permits larger payloads and has a different ABI record size. Standard and extended identifiers, remote frames, error frames, and payload length must be interpreted explicitly. [SocketCAN interface](https://docs.kernel.org/networking/can.html).

The [CPU protocol lab](labs/README.md) exchanges an eight-byte versioned sensor payload between actual C and Python code using Linux's Classic CAN record layout. It rejects wrong identifiers, flags, lengths, and versions. It also demonstrates why sequence 255 → 0 is forward progress while a repeated sequence is a duplicate.

There are several different loopback claims:

| Test | What it can establish |
| --- | --- |
| Application codec fixture | Message layout and validation |
| Linux `vcan` | SocketCAN software path without a physical controller |
| Controller internal loopback | A controller/driver path without proving external wiring |
| Two physical nodes | Transceiver, wiring, bit timing and delivery under that setup |
| Loaded robot network | Application behavior under its actual traffic/failure conditions |

An ACK establishes neither the identity of an intended application nor that its command was accepted. CAN identifiers arbitrate traffic; they do not authenticate a sender. Define command acknowledgements, expected state transitions, freshness limits, timeout behavior, and session reset rules at the application layer.

Inspect `ip -details -statistics link show` for the chosen interface during a bench experiment. Bus-off recovery is a policy decision: automatically restarting a controller does not establish that resuming an actuator command is appropriate. Retain error counters and timestamps instead of diagnosing every send failure as “interface down.”

## 8. Use PWM and ADC through suitable hardware

PWM frequency and duty cycle are separate quantities. A 1.5 ms pulse every 20 ms is 50 Hz and 7.5% duty cycle. A servo's position mapping and valid pulse range come from its specification and calibration; 1–2 ms does not universally mean 0–180°.

Use the appropriate kernel driver or PWM interface and verify the actual chip/channel routing. GPIO and PWM sysfs interfaces have different status: deprecation of GPIO sysfs does not imply the documented PWM sysfs interface vanished. Set a valid period/duty combination and define the output state while disabled and during boot. [Linux PWM interface](https://docs.kernel.org/driver-api/pwm.html).

The expansion header is not a general analog-input ADC. An external ADC requires a reference, input range, front end, sampling plan, and driver. For an illustrative ideal 12-bit converter with a 3.3 V reference, one LSB is 3.3/4096 ≈ 0.806 mV. Noise, offset, gain error, source impedance, and clipping still affect accuracy. Resolution is not accuracy.

Prefer the Industrial I/O subsystem for supported sensors and converters, including its channel, scale/offset, buffering, and timestamp facilities. Read the driver's documented units. A power-monitor rail reading is not interchangeable with an arbitrary sensor voltage. [Linux IIO documentation](https://docs.kernel.org/iio/index.html).

## 9. Connect measurements to AI without losing meaning

Use a message contract containing sensor identity, units, frame, sequence, validity, capture timestamp, receive timestamp, and clock domain where relevant. Preserve these fields through preprocessing and inference. A model result generated now may still depend on an old measurement.

Separate three ages: age at the sensor, time spent in communication/queues, and time spent computing. A sender-declared age plus a receiver's local elapsed time still omits unknown transport delay unless the protocol supplies a justified bound or clock mapping. This distinction is explicit in the CPU lab.

Bound queues and record drops. A latest-sample policy is useful for some control observations; a synchronized multi-sensor estimator may require a different buffering policy. Do not replace one sensor's timestamp with another's merely to make tensors align. Continue with the [camera/time/fusion exercise](../../../Phase%203%20-%20Artificial%20Intelligence/Track%20A%20-%20Hardware%20and%20Edge%20AI/4.%20Sensor%20Fusion/Camera-Time-Fusion-Lab/Guide.md) after verifying its prerequisites.

Use a hardware controller or MCU for timing-critical motor control where the system design requires it. The Jetson can send bounded, versioned requests while the controller independently handles limits, watchdog expiry, and a defined fallback. A perception confidence threshold alone is not a complete actuator policy.

## 10. Complete a sensor integration report

Choose one measurement, such as wheel speed or distance. First run the CPU protocol exercise. Then, if hardware is available, connect a supported sensor using the verified schematic and driver path. Record both normal operation and deliberate disconnection without attaching an active motor to the first experiment.

Your report should include:

- Hardware/software identities, connection table, and power-on states.
- Register/message schema with byte order, units, range, and golden example.
- Measured sample cadence and capture-to-consumer timing assumptions.
- Logs for malformed, stale, missing, duplicated, and restarted data.
- The consumer's explicit response and the evidence supporting it.

**Completion answers:** a successful SPI syscall does not prove the peripheral responded; a correct CAN CRC does not prove the message is fresh; a GPIO timestamp does not bound application response time; and a valid device tree does not prove the correct connector was wired. You should be able to explain the missing evidence in each case and design the next test.
