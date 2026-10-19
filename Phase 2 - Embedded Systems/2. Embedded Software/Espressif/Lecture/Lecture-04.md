# Lecture 4 - Peripherals, libraries, connectivity, and system design

**Course:** [Espressif guide](../Guide.md) | **Previous:** [Task and buffer ownership](Lecture-03.md) | **Next:** [Component integration](Lecture-05.md)

Turn the heartbeat into a sensor report by adding one bounded peripheral transaction. Keep the driver separate from conversion and network code so that a communication failure cannot masquerade as a valid new measurement.

## Read a register deliberately

Identify the sensor's power domain, 7-bit I2C address, identity register, supported clock rate, register-address width, and byte order from its datasheet. Verify SDA/SCL pins against the board schematic, then call `Wire.begin(sda_pin, scl_pin, frequency)` using those pins. Check the return value and configure a finite timeout using the current [I2C API](https://docs.espressif.com/projects/arduino-esp32/en/latest/api/i2c.html).

A common register read writes a register address without ending the bus transaction, then issues a repeated-start read. This helper illustrates a **one-byte register address and a two-byte big-endian value**; those are assumptions to confirm for your sensor, not universal I2C rules. It is a board exercise fragment, not a hardware-tested driver.

```cpp
bool read_u16_be(uint8_t device, uint8_t reg, uint16_t& result) {
    Wire.beginTransmission(device);
    if (Wire.write(reg) != 1) {
        Wire.endTransmission(true);
        return false;
    }
    if (Wire.endTransmission(false) != 0) return false;
    const size_t received = Wire.requestFrom(device, size_t{2}, true);
    if (received != 2) {
        while (Wire.available()) (void)Wire.read();
        return false;
    }
    const int high = Wire.read();
    const int low = Wire.read();
    if (high < 0 || low < 0) return false;
    result = static_cast<uint16_t>((static_cast<unsigned>(high) << 8) |
                                   static_cast<unsigned>(low));
    return true;
}
```

The output changes only on success. Do not refresh the sample timestamp when a read fails and you reuse the previous value. If the sensor supplies signed 16-bit data, explicitly interpret values above 32767 as `raw - 65536`, then apply its scale and offset. Some sensors require status checks, multi-byte register addresses, delays, or different transaction boundaries.

## Budget bytes and events

For UART, use the [framed telemetry lab](../../labs/Framed-Telemetry.md): a single read may contain half a packet or multiple packets. For UDP, each datagram already has a boundary; still validate its length, schema, session, sequence, and freshness. An ACK from a transport does not prove that a physical actuator completed a command.

For network reconnection, use explicit states such as `DISCONNECTED → CONNECTING → READY → BACKOFF`. Keep sensing and local status reporting useful when the network is unavailable. Bound connection attempts and queued data. On reconnect, decide whether old records should be discarded or uploaded with their original timestamps.

ESP32-C6's radio protocols share RF resources. The [official coexistence guide](https://docs.espressif.com/projects/esp-idf/en/stable/esp32c6/api-guides/coexist.html) describes supported combinations and warns that Wi-Fi/BLE activity can increase Thread/Zigbee packet loss for router roles. “The chip supports all three radios” does not imply independent simultaneous throughput or arbitrary role combinations.

## Keep local control independent of model and network delays

An AI service can suggest an action, but the receiving controller must check command type, range, session, expiry, and current operating state. Keep an explicit local stop/fault path. For this introductory project, represent commands with an LED or logged state transition before attaching a motor.

A useful separation is:

```text
sensor driver → validated observation → bounded network sender
remote request → schema/range/age checks → local state machine → output
```

Do not treat generated text as a peripheral register write. Convert an allowed request into a small typed command, then validate it with deterministic code. This makes the same interface usable by a human UI, a test harness, or a model.

## Failure lab and expected results

1. Disconnect the sensor in a recoverable bench setup. Expect an error counter and unchanged last-valid timestamp, not a fabricated zero reading.
2. Make the receiver pause. Expect the documented queue-full behavior and a visible drop/backpressure count.
3. Restart the sender. Expect a new session, so old sequence values are not compared across unrelated boots.
4. Remove the network. Expect sensing/status to continue and connection retries to remain bounded.

Record which checks ran on hardware and which ran in the CPU lab. A useful final record contains `{session, sequence, sampled_at, value, status}` with units and clock assumptions. The host must distinguish a valid zero, a missing value, and an old value.
