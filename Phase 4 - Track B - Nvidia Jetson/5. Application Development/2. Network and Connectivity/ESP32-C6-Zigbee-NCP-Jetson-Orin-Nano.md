# Zigbee on Jetson: versioned NCP and RCP architectures

**Reviewed 18 October 2026.** ESP Zigbee SDK **2.0.5**, released 16 October, is current at commit `e2fdf4ad8558e5beb306b0d7040072d76244c413`. Its main tree does **not** contain the old `examples/esp_zigbee_ncp` directory. Cloning current main and changing into that path fails. This lesson gives a current architecture choice and a separately pinned legacy NCP exercise. [Official release notes](https://github.com/espressif/esp-zigbee-sdk/blob/e2fdf4ad8558e5beb306b0d7040072d76244c413/RELEASE_NOTES.md).

**Prerequisites:** [networking](Guide.md), byte order and [framing lab](Localhost-Network-Lab.md). **Deliverable:** a version/stack-placement map, passing SLIP tests, and a staged integration contract. No Zigbee radio, coordinator network or Jetson host stack was executed during this review.

## 1. Separate three independent choices

| Choice | Examples | What it determines |
|---|---|---|
| Network protocol | Zigbee, Thread | Addressing, network formation and application conventions |
| Stack location | NCP, RCP | Which processor runs the network stack |
| Physical host link | UART, SPI, SDIO | How host and coprocessor exchange bytes |

Zigbee and Thread both use IEEE 802.15.4, but a Thread node does not join a Zigbee network. An NCP runs the network stack and presents higher-level commands; an RCP presents radio/minimal-MAC operations to a host stack. UART is not inherently NCP, and SPI is not inherently RCP.

In a Zigbee NCP design, Linux normally runs gateway/application logic through the coprocessor protocol. The coprocessor does not automatically register a Linux IPv6 interface. In an RCP design, the chosen host must actually implement a compatible Zigbee stack; “a serial device exists” is not that implementation.

## 2. Current v2 and legacy v1 are different software families

| SDK family | Zigbee core | Relevant scope |
|---|---|---|
| v2.x / 2.0.5 | Espressif implementation; `ezb_` core and `esp_zigbee_` SDK APIs | Current designs; on-device apps and ESP-host gateway examples |
| `release/v1.0` maintenance line | ZBOSS; legacy APIs | Existing products and explicitly versioned NCP/host examples |

The v2 migration changes headers, API organization and persistent-storage handling. It is not just a function-name search/replace. Moving old network state into the new NVS representation requires the documented migration path; casually erasing flash discards commissioning state and keys. The 2.0.4 release changed the default NVS partition name, so inspect the exact 2.0.5 example/configuration rather than blindly copying an older partition table. [SDK support policy](https://github.com/espressif/esp-zigbee-sdk/blob/e2fdf4ad8558e5beb306b0d7040072d76244c413/README.md), [migration guide](https://docs.espressif.com/projects/esp-zigbee-sdk/en/latest/esp32c6/migration-guide/v2.x/overview.html).

The current SDK recommends ESP-IDF 5.5.4. The pinned legacy branch README recommends 5.3.2 and its example manifest constrains the Zigbee/ZBOSS libraries to the 1.6 family. The SDK2 recommendation does not validate an old NCP dependency set. Keep the resolved component lock, not only a broad `>=5.0` IDF constraint.

## 3. What the current gateway example actually demonstrates

The v2 `zigbee_gateway` example runs the Zigbee stack/application on an **ESP Wi-Fi-capable host** and uses another IEEE 802.15.4 ESP chip running `ot_rcp` as its radio. It is not a Linux executable for Jetson. Its checked README also leaves higher-level Wi-Fi gateway interaction for future work; do not describe it as a finished cloud gateway. [Current gateway example](https://github.com/espressif/esp-zigbee-sdk/blob/e2fdf4ad8558e5beb306b0d7040072d76244c413/examples/zigbee_gateway/README.md).

Current [ESP-Hosted-Linux RCP support](https://github.com/espressif/esp-hosted-linux/blob/9b56ff96a1bc95acaa68d8023794b8353701c5e3/docs/guides/thread-rcp.md) can expose a compatible radio stream to Linux, but explicitly does not bundle or qualify a Linux Zigbee userspace stack. Upstream Thread RF tests cannot be relabeled Zigbee application tests.

For a product, select a documented host/adapter combination with the required Zigbee role and devices. For learning, the local exercises below establish framing and host-state reasoning without claiming that a missing Linux stack has been implemented.

## 4. Reproduce the legacy source intentionally

The maintenance branch resolved to `c9e2c3e12642c704096dfefe25b62212b47229ba` on review. It contains **both** `esp_zigbee_ncp` and `esp_zigbee_host`; the latter targets ESP chips, not Linux. The NCP README's old statement promising a future host example is stale compared with the actual tree. [NCP source](https://github.com/espressif/esp-zigbee-sdk/tree/c9e2c3e12642c704096dfefe25b62212b47229ba/examples/esp_zigbee_ncp), [ESP host source](https://github.com/espressif/esp-zigbee-sdk/tree/c9e2c3e12642c704096dfefe25b62212b47229ba/examples/esp_zigbee_host).

Source-reviewed, unexecuted hardware sequence:

```bash
git clone --branch release/v1.0 https://github.com/espressif/esp-zigbee-sdk.git esp-zigbee-legacy
cd esp-zigbee-legacy
git checkout --detach c9e2c3e12642c704096dfefe25b62212b47229ba
cd examples/esp_zigbee_ncp
# Activate the selected legacy-compatible ESP-IDF environment first.
idf.py set-target esp32c6
idf.py menuconfig
idf.py build
```

Choose the actual serial transport pins and baud, resolve and retain `dependencies.lock`, then flash the discovered programming port. The example's GPIO4 RX/GPIO5 TX wiring is an example setting, not a Jetson UART mapping. Confirm the board's pinmux and connect compatible logic with TX/RX crossed. Close the serial monitor before the binary host owns the data port.

For a Linux port, study the legacy ESP host's transport, framing and command code, then replace ESP-IDF UART/task/timing dependencies with Linux equivalents. That port is an additional implementation project; this course does not provide an untested one-command daemon.

## 5. Framing is necessary but not sufficient

Legacy ESP ZNSP uses SLIP framing plus a defined payload and a 16-bit CRC. Its fields have specified byte order. A valid SLIP frame can still contain the wrong command ID, length, CRC, transaction reference or payload type. [Pinned protocol source](https://github.com/espressif/esp-zigbee-sdk/blob/c9e2c3e12642c704096dfefe25b62212b47229ba/docs/en/user-guide/ncp.rst).

Run the course's [bounded SLIP implementation](labs/slip.py):

```bash
python3 -m unittest discover -s \
  'Phase 4 - Track B - Nvidia Jetson/5. Application Development/2. Network and Connectivity/labs' \
  -p 'test_slip.py' -v
```

Three tests cover every byte value across fragmented input, independent expected wire bytes, multiple frames, invalid escapes, oversized frames and resynchronization. The code deliberately does not implement ZNSP CRC or commands. Its 4,096-byte default is a teaching bound, not a claim about the protocol's maximum.

**Worked UART budget:** a 100-byte payload at 115200 baud with 8N1 takes at least `100 × 10 / 115200 = 8.68 ms`. If every payload byte needs SLIP escaping, payload transmission approximately doubles before delimiters, CRC and response. Choosing a high-level protocol does not remove transport costs.

## 6. Define host behavior before sending commands

A host implementation needs explicit states: disconnected, version-negotiated, configured, joining/forming, running and recovering. Distinguish a command acknowledgement from an asynchronous event that the network formed or an end device joined. Bound retries and preserve transaction identity.

A coordinator starts a Zigbee network; a router joins and forwards; an end device joins through a parent and may sleep. A mains-powered gateway often uses coordinator functionality, but it cannot become the coordinator of an already existing network merely by changing a role field. Preserve network identity and device state during recovery.

Test transport independently, then version/status queries, network formation, a controlled join window, endpoint/cluster discovery, an attribute read, a state-changing command, and reconnect. Each stage needs an observed response/event and a failure case. Do not treat a successful serial write as successful device control.

The [UDP command exercise](Localhost-Network-Lab.md) shows why losing an acknowledgement after an action makes a blind retry ambiguous. Apply the protocol's actual transaction/response rules rather than inventing exactly-once delivery.

## 7. Coexistence and architecture decisions

C6's Wi-Fi, BLE and IEEE 802.15.4 share a radio. Separate SoCs can reduce contention, while adding power, wiring and software coordination. Current Espressif guidance identifies restricted/unstable combinations, especially continuous router reception while other radios are busy. [RF coexistence guidance](https://docs.espressif.com/projects/esp-idf/en/v5.5.4/esp32c6/api-guides/coexist.html).

Zigbee NCP does not use OTBR's IPv6 border-routing machinery, so an OTBR-specific multicast-routing failure is not the same failure path. This architectural distinction is not proof that a Linux Zigbee host works, nor evidence that the learner's kernel lacks any particular option.

## 8. Assessment

Submit the version map, your escaped-byte example, passing local tests and a host-state diagram with two failure transitions. For hardware work, add source/configuration hashes and actual transport/network/device results.

**Diagnostic answers:** an RCP is not a Zigbee application; an ESP-host example is not a Linux host binary; same UART pins do not imply compatible protocols; a coordinator log does not prove a particular device's cluster commands work. Continue with the [Thread comparison](ESP32-C6-OpenThread-RCP-Jetson-Orin-Nano.md) and explain where each stack runs.
