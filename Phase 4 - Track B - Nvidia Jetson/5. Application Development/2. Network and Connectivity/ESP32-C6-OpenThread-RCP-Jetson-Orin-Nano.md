# ESP32-C6 OpenThread RCP: radio transport before border routing

**Reviewed 18 October 2026.** Learn where the Thread stack runs, configure a matched radio link, and distinguish a functioning Thread node from a border router. This revision inspected upstream source; it did not flash a C6 or run OTBR on Jetson. The old text's claims about “your validated UART” and missing kernel flags were not supported by retained bench evidence and are not prerequisites here.

**Start with:** [networking](Guide.md) and the [actual CPU transport lab](Localhost-Network-Lab.md). **Time:** 60–90 minutes for source/architecture work; hardware bring-up is a separate session.

## 1. RCP and NCP describe stack placement

In an RCP design, the Linux host runs the OpenThread core and the device supplies the radio and minimal MAC functions. In an NCP design, the coprocessor runs the network stack and exposes higher-level operations to the application host. UART versus SPI does not determine which design you have. Spinel is used in OpenThread coprocessor designs; framing and capabilities must match. [OpenThread architecture](https://openthread.io/platforms/co-processor).

```text
Application / OpenThread / Linux         ESP32-C6
              |                             |
       Spinel host protocol ← UART/HDLC → radio/minimal MAC
```

Thread carries IPv6 over an IEEE 802.15.4 mesh. A border router connects that network to an infrastructure IP network. A Thread leader manages a partition; it is not a central Wi-Fi access point and does not by itself establish external reachability.

## 2. Choose one of two current transports

| Transport | Firmware | Linux endpoint |
|---|---|---|
| Dedicated UART or supported USB Serial/JTAG RCP | ESP-IDF `examples/openthread/ot_rcp` | The serial/USB device selected by firmware and board |
| Hosted SPI/SDIO RCP on C6 | Current ESP-Hosted-Linux with RCP enabled | `/dev/esp_rcp0` from the hosted driver |

The second route is now supported upstream and can share the hosted link with Wi-Fi/Bluetooth. It is not the old NG firmware by another name. The dedicated second-radio route remains useful for isolating RF contention and resets. [Current hosted RCP contract](https://github.com/espressif/esp-hosted-linux/blob/9b56ff96a1bc95acaa68d8023794b8353701c5e3/docs/guides/thread-rcp.md).

Use one owner per endpoint. A serial monitor, `ot-daemon` and `otbr-agent` cannot all consume the same binary stream. A board's programming port need not be its configured RCP data port.

## 3. Inspect a reproducible standalone firmware configuration

For a source-reviewed baseline, ESP-IDF v5.5.4 resolves to commit `735507283d5b2f9fb363a1901172dbd9e847945d`. Its [RCP example configuration](https://github.com/espressif/esp-idf/blob/735507283d5b2f9fb363a1901172dbd9e847945d/examples/openthread/ot_rcp/main/esp_ot_config.h) defines UART0 at **460800 baud, 8N1, no hardware flow control** when the UART backend is selected. Manual RX/TX pin configuration is separate; unchanged defaults do not imply GPIO4/5 on every board.

On a build machine with that IDF installed and exported:

```bash
cd "$IDF_PATH/examples/openthread/ot_rcp"
idf.py set-target esp32c6
idf.py menuconfig
idf.py build
```

Select the intended RCP backend and UART pins, then save the resulting configuration and binary hash. Flash with `idf.py -p PORT flash` using the discovered flashing port. These commands have not been executed for this course revision. The [example README](https://github.com/espressif/esp-idf/blob/735507283d5b2f9fb363a1901172dbd9e847945d/examples/openthread/ot_rcp/README.md) and your actual header/source take precedence over a copied baud assumption.

For direct UART, cross TX→RX and RX←TX, share ground and use compatible logic voltage. Confirm the carrier's UART pinmux and Linux device mapping. Do not assume `/dev/ttyTHS1`, disable `nvgetty` indiscriminately, or attach a 5 V serial adapter. Identify an actual conflicting console/service before changing it. [Serial connection guidance](https://docs.espressif.com/projects/esp-idf/en/v5.5.4/esp32c6/get-started/establish-serial-connection.html).

## 4. Validate an OpenThread host before full routing

At reviewed OpenThread commit `227561fe2446fa57e8a1aa192c81e31396db9ac9`, the documented POSIX daemon build is:

```bash
./script/bootstrap
./script/cmake-build posix -DOT_DAEMON=ON
```

Run those inside an intentionally pinned OpenThread checkout; they install/build dependencies and were not run here. The daemon connects to an RCP URL. For a UART configured as above, the shape is:

```text
spinel+hdlc+uart:///dev/YOUR_ACTUAL_PORT?uart-baudrate=460800
```

For the hosted character device, upstream documents `spinel+hdlc+uart:///dev/esp_rcp0`; it is a driver-provided byte stream, not proof of a physical UART at that path. [OpenThread daemon instructions](https://openthread.io/platforms/co-processor/ot-daemon).

First inspect protocol version, RCP capabilities and clean reset/response exchange. Baud mismatch, unsupported Spinel API versions and contention can all produce timeouts. Do not interpret every timeout as missing Linux IPv6 routing.

For a new isolated lab network, use the matching `ot-ctl`:

```bash
sudo ot-ctl dataset init new
sudo ot-ctl dataset commit active
sudo ot-ctl ifconfig up
sudo ot-ctl thread start
sudo ot-ctl state
sudo ot-ctl ipaddr
```

A new dataset changes network identity and credentials. Preserve the intended active dataset for existing devices; do not recreate it on every reboot. Record network state without publishing network keys.

## 5. Add border routing as a separate stage

OTBR adds infrastructure routing, service discovery and related border functions. Choose the actual infrastructure interface from `ip route`/`ip -br address`; `l4tbr0` is not universally an Internet uplink. Bind the radio URL to the endpoint already validated above.

Current official instructions distinguish [native installation](https://openthread.io/guides/border-router/build-native) from [Docker installation](https://openthread.io/guides/border-router/build-docker). The Docker route exposes the radio device and `/dev/net/tun`, uses the host network and network-administration capability, and configures forwarding on the host. A container does not replace the host kernel or add missing compiled kernel features.

If a particular run fails to create a multicast-routing socket, inspect that host's kernel configuration and error context:

```bash
uname -r
zcat /proc/config.gz | rg 'CONFIG_(TUN|IP_MROUTE|IPV6_MROUTE)'
```

If `/proc/config.gz` is unavailable, inspect the installed matching kernel config, often `/boot/config-$(uname -r)`, or the build configuration. The absence of the config file is not proof that a feature is disabled. Requirements also depend on enabled OTBR features. Do not claim every Jetson image lacks multicast routing, or rebuild a kernel before identifying the failing operation.

The checked OTBR source (`6e50b731e86457f7f42e69743cf794796b48ccf4`) explicitly rejects the old `OTBR_MDNS=avahi` selection; use the supported mDNS implementation for the chosen build. This is another reason to preserve build flags and submodule revisions. [Current CMake options](https://github.com/openthread/ot-br-posix/blob/6e50b731e86457f7f42e69743cf794796b48ccf4/etc/cmake/options.cmake).

## 6. Test an end-to-end claim

Test progressively: host↔RCP protocol, two Thread nodes exchanging data, routed IPv6 from an infrastructure host to a Thread device, then service discovery if required. Record addresses and routes appropriate to each test. A process marked `active`, an interface named `wpan0`, or state `leader` is not sufficient evidence for all four.

Disconnect/reconnect the RCP and check recovery. For hosted RCP, a firmware generation change invalidates the old stream and can interrupt Wi-Fi/Bluetooth too. The service must reopen/reinitialize deliberately; it must not splice bytes from the old session into a new frame.

**Worked budget:** 100 serialized bytes on 460800-baud 8N1 UART occupy at least `100 × 10 / 460800 = 2.17 ms`. HDLC escaping, queueing, radio access and a response add more time. This is a serialization calculation, not a measured Thread latency.

## 7. Exercises and evidence

- Explain why a successful leader election does not verify external routing. **Answer:** the first occurs inside a Thread partition; the second requires working infrastructure routes and border functions.
- Explain why an `ot_rcp` radio can appear in a Zigbee gateway example. **Answer:** the radio protocol supplies IEEE 802.15.4 primitives; the compatible host stack determines the network protocol. This does not make Thread and Zigbee interchangeable over the air.
- Submit a two-column host/RCP responsibility diagram, exact version/configuration records, and a fresh-versus-stale message policy from the [CPU lab](Localhost-Network-Lab.md). Add hardware logs only for stages actually run.

Continue to [Zigbee stack-placement options](ESP32-C6-Zigbee-NCP-Jetson-Orin-Nano.md) if your target devices speak Zigbee rather than Thread.
