# Lecture 5 - ESP32-C6 devices, NCP, gateways, and Linux-host integration

**Course:** [Zigbee guide](../Guide.md) | **Previous:** [Security and sleepy devices](Lecture-04.md)

ESP32-C6 has an IEEE 802.15.4 radio and can participate in Zigbee designs using an appropriate stack and configuration. Decide where the Zigbee stack runs before selecting host software. The words “ESP32-C6” and “serial port” do not define a compatible control protocol.

## Current SDK lines are materially different

Reviewed 2026-10-18: Espressif's [SDK README](https://github.com/espressif/esp-zigbee-sdk) identifies v1.x as the maintained ZBOSS-based LTS line and v2.x as its newer stack recommended for new designs. [Release notes](https://github.com/espressif/esp-zigbee-sdk/blob/main/RELEASE_NOTES.md) record v2.0.5 on 2026-10-16, based on ESP-IDF 5.5.4. The [migration guide](https://docs.espressif.com/projects/esp-zigbee-sdk/en/latest/esp32/migration-guide/v2.x/index.html) covers changes; old `esp_zb_*` examples and new `ezb_*` APIs are not interchangeable by renaming the package.

Keep the Zigbee SDK revision, library component version, IDF version, target, and dependency lockfile together. Standalone IDF 6.1 or Arduino 3.3.12 compatibility does not establish compatibility for this SDK combination.

## Three implementations to distinguish

| Design | Where the Zigbee stack runs | What the other processor must understand |
|---|---|---|
| Standalone MCU device | On the radio MCU with its application | No host protocol required for ordinary operation |
| Network coprocessor (NCP) | On the radio MCU | The selected NCP's commands, framing, events and version contract |
| Host stack with radio coprocessor (RCP) | On the host processor | Radio-control protocol, reset/update behavior, timing and driver integration |

The v1.x [NCP guide](https://docs.espressif.com/projects/esp-zigbee-sdk/en/release-v1.0/esp32c6/user-guide/ncp.html) remains a reference for that pinned architecture. Its [official NCP example](https://github.com/espressif/esp-zigbee-sdk/blob/release/v1.0/examples/esp_zigbee_ncp/README.md) expects a matching host. This is not a promise that an arbitrary Linux Zigbee application supports that protocol.

The current [v2 gateway example](https://github.com/espressif/esp-zigbee-sdk/blob/main/examples/zigbee_gateway/README.md) takes a different approach: a Wi-Fi SoC runs the Zigbee gateway stack, while an H2/C6 radio SoC runs the `ot_rcp` example to provide 802.15.4 access. An RCP firmware name containing “OpenThread” does not mean the application network is necessarily Thread. The actual host stack determines the network protocol.

This distinction also means an ESP-IDF gateway example is not automatically a Linux/Jetson program. Check host platform support, licensing, available stack binaries/source, transport, and version negotiation before planning a port.

## Start with a standalone example

Use the [current development guide](https://docs.espressif.com/projects/esp-zigbee-sdk/en/latest/esp32c6/developing.html) and a pinned SDK example such as a temperature sensor or on/off light. Activate the compatible IDF environment and select the actual target before building:

```bash
idf.py --version
idf.py set-target esp32c6
idf.py build
idf.py size
```

These are documented board-development commands, not a build/flash performed by this course revision. Inspect the selected example's role and commissioning behavior: a light example may be configured as coordinator for the demonstration, which is not a requirement that all lights be coordinators. Preserve configuration and saved network state intentionally. Erasing flash removes credentials and other device data; it is a deliberate fresh-lab operation, not a routine step required on every rebuild.

On a two-board bench, capture formation/join status, endpoint/cluster discovery, a command or report, and the resulting application state. Then disable permit-join and verify that an uncommissioned device is not admitted through the same path. Record exact security settings rather than asserting security from network membership alone.

## Connect to a Linux host without losing the data contract

The existing [Jetson NCP exercise](../../../../../Phase%204%20-%20Track%20B%20-%20Nvidia%20Jetson/5.%20Application%20Development/2.%20Network%20and%20Connectivity/ESP32-C6-Zigbee-NCP-Jetson-Orin-Nano.md) is a separate, version-specific integration. First prove UART framing and reset recovery, then stack/host compatibility, then discovery and application traffic. A readable serial banner is not evidence of a functioning Zigbee network.

Map each report into stable source identity, endpoint, cluster/attribute, value with units, validity, and timing context. A ZCL transaction sequence is not a source sampling timestamp. Do not refresh an observation merely because the host successfully queried a cached value.

If you have no radios, complete the [decoded ZCL transaction exercise](Lecture-03.md) and the [real OpenThread CPU lab](../../OpenThread/labs/CPU-Simulation.md). The latter teaches attachment and application evidence using Thread; it does not validate the Zigbee SDK or an ESP32 radio.

## Exercises with answers

- **Can an NCP be replaced by an RCP without changing the host?** Generally no. The stack split and host protocol are different.
- **Does an `ot_rcp` image force a Thread network?** No. In the cited Zigbee gateway it supplies radio services to a Zigbee stack on the other SoC.
- **What is the first migration artifact?** A compatibility table and a reproducible old baseline, followed by the v1-to-v2 migration steps and repeated discovery/command/recovery tests.

Deliver an architecture diagram marking the stack boundary, pinned versions, and the evidence for one complete application transaction. Keep compile success, network attachment, and physical action as distinct observations.
