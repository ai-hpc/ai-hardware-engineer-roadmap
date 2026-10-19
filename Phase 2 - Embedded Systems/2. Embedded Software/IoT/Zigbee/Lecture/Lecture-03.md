# Lecture 3 - Decode a Zigbee transaction: ZDO, APS, endpoints, and clusters

**Course:** [Zigbee guide](../Guide.md) | **Previous:** [Roles and formation](Lecture-02.md) | **Next:** [Security and sleepy devices](Lecture-04.md)

A Zigbee network must deliver a message to the right device, select the right application inside that device, and interpret the command or value consistently. Learn the stack by tracing one temperature report rather than memorizing acronyms in isolation.

## The layers answer different questions

| Layer or concept | Responsibility in this example |
|---|---|
| IEEE 802.15.4 PHY/MAC | Radio transmission and link-level framing/access |
| NWK | Zigbee addressing and multi-hop delivery |
| APS | Application delivery context: endpoints, profile, cluster, binding and applicable security/acknowledgment |
| ZDO | Device/service discovery and management, using endpoint 0 |
| ZCL | Meaning and encoding of attributes and commands within a cluster |

Application endpoints are logical functions within a node. One physical device can expose several endpoints; changing endpoint does not change which physical radio sent the packet. A network short address can change after rejoining, so a gateway must not treat it as a permanent product identity. See [Silicon Labs' stack overview](https://docs.silabs.com/zigbee/latest/zigbee-fundamentals/05-the-zigbee-stack) and [cluster-library explanation](https://docs.silabs.com/zigbee/latest/zigbee-fundamentals/06-zigbee-cluster-library).

## Trace the addressing context

For a teaching temperature sensor, assume source endpoint 10, destination endpoint 1, Home Automation profile `0x0104`, and Temperature Measurement cluster `0x0402`. The sensor implements the server side because it holds the measured attribute; the reporting consumer handles the corresponding client behavior. Server/client here describes cluster roles, not which device formed the network.

A decoded record needs all this context. Attribute `0x0000` alone is ambiguous because many clusters define an attribute with that number. A packet analyzer may show an APS acknowledgment before your application validates the value. Neither a radio ACK nor an APS ACK proves that the reading is physically plausible or fresh.

Discovery typically asks which endpoints exist and obtains their simple descriptors. Binding associates a source endpoint and cluster with a destination address/endpoint or group; it does not create a network route or guarantee that a receiver supports every command. Check response status and the actual destination's descriptor.

## Work through a ZCL report

Consider these **synthetic decoded ZCL bytes**, after lower-layer delivery and security processing:

```text
18 2a 0a 00 00 29 c4 09
```

They are an application payload exercise, not a complete radio frame or a captured device measurement.

| Bytes | Meaning under this example's context |
|---|---|
| `18` | Global ZCL command, not manufacturer-specific, server-to-client, default response disabled |
| `2a` | Transaction sequence number 42 |
| `0a` | Report Attributes command |
| `00 00` | Little-endian attribute ID `0x0000` (MeasuredValue in this cluster) |
| `29` | ZCL signed 16-bit integer data type |
| `c4 09` | Little-endian `0x09C4 = 2500`, interpreted here as 25.00 °C |

The temperature cluster defines the unit and invalid sentinel, not the `int16` type by itself. In this example `00 80` represents the special invalid value `0x8000`; it must not become a displayed −327.68 °C. Consult the selected stack's [temperature measurement definitions](https://github.com/espressif/esp-zigbee-sdk/blob/main/components/esp-zigbee-lib/include/ezbee/zcl/cluster/temperature_measurement_desc.h) and the applicable ZCL specification when implementing a complete decoder.

You can check the arithmetic on any Python 3 interpreter:

```python
payload = bytes.fromhex("18 2a 0a 00 00 29 c4 09")
assert len(payload) == 8
assert payload[:1] == b"\x18" and payload[2] == 0x0a
assert int.from_bytes(payload[3:5], "little") == 0
assert payload[5] == 0x29
raw = int.from_bytes(payload[6:8], "little", signed=True)
value_c = None if raw == -32768 else raw / 100.0
assert value_c == 25.0
print(value_c)
```

This small fixture was executed during the course review. It is intentionally not a general parser: real reports can contain multiple attributes, manufacturer-specific headers, other data types, and variable-length values. A real decoder must check remaining length before every field, reject unsupported encodings explicitly, and avoid reading beyond the received buffer.

## Commands and observed state

For the On/Off cluster `0x0006`, an `On` command asks the server to change its state. A read/report of the OnOff attribute describes its logical state. A physical lamp can still fail despite logical state being on; application confirmation should match the product claim.

For a robot gateway, distinguish “request accepted,” “controller output enabled,” and “physical motion observed.” A model-generated command should pass a typed allowlist, range, current-state and expiry check before an actuator adapter sees it. An application cluster provides interoperability semantics; it does not replace local control constraints.

## Practice with answers

1. **Replace `c4 09` with `f3 fd`. What is the value?** Signed little-endian `0xFDF3 = −525`, giving −5.25 °C.
2. **Replace it with `00 80`. What should the UI show?** Invalid/unavailable, preserving diagnostic context rather than a numeric temperature.
3. **Can transaction sequence 42 prove this is a fresh sensor observation?** No. ZCL transaction sequence is limited protocol bookkeeping, not a source sampling timestamp or globally unique observation identifier.
4. **Can a short address identify a sensor permanently?** No. Preserve its appropriate stable identity and endpoint mapping across rejoin/address changes.

Deliver one annotated transaction with addressing context, decoded fields, unit, validity, and application timestamp assumptions. If using a real packet capture, record the device/stack revisions and redact deployment credentials.
