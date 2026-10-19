# Networking for a robot: connectivity, freshness and recovery

<div class="course-identity auto-course" style="--course-accent: #4f46e5; --course-accent-rgb: 79, 70, 229;" markdown="1">
<div class="course-identity__icon">NAC</div>
<div markdown="1">
<p class="course-identity__eyebrow">Deep Dive · Jetson Track</p>
<p class="course-identity__title">Specialized course identity for Network and Connectivity.</p>
<p class="course-identity__meta">Artifact: Jetson integration demo · Measure: latency, memory, power, logs</p>
</div>
</div>

**Reviewed 18 October 2026.** Learn to discover the actual hardware, configure one network manager, and measure application behavior when messages arrive late or are repeated. Begin with the [localhost transport lab](Localhost-Network-Lab.md); it runs on a normal computer with Python alone. Hardware instructions below are source-reviewed, not reported Jetson bench results.

**Prerequisites:** Linux processes, files and basic Python. Allow two 60-minute sessions plus optional hardware bring-up. **Deliverable:** an interface map, a transport report, and evidence that expired commands are not applied.

## 1. Identify the board and the interfaces

The official Orin Nano developer kit has Gigabit Ethernet and an M.2 Key E slot populated with an included wireless module. A production module, third-party carrier, replacement radio, or regional kit can differ. Do not buy an adapter merely because an old guide says Wi-Fi is absent. Check the actual kit and driver. [NVIDIA hardware layout](https://docs.nvidia.com/jetson/orin-nano-devkit/user-guide/latest/hardware_layout.html).

```bash
uname -r
ip -br link
ip -br address
ip route
nmcli device status
nmcli -f NAME,UUID,TYPE,DEVICE connection show --active
lspci -nnk
lsusb
iw dev
rfkill list
```

`eth0`, `wlan0`, `hci0` and `Wired connection 1` are possible names, not universal identifiers. Linux interface names identify devices; NetworkManager connection names/UUIDs identify saved profiles. Record both. A radio visible on PCIe/USB still needs compatible firmware and a kernel driver before a usable interface appears.

Use the [Jetson compatibility record](../../Compatibility.md) for the installed BSP. Installing arbitrary Ubuntu headers does not ensure they match NVIDIA's running kernel ABI.

## 2. Understand which layer failed

| Observation | What to inspect next |
|---|---|
| No device on PCIe/USB/SPI | Power, wiring, bus, pinmux or device tree |
| Device exists, no network interface | Driver binding, firmware and kernel log |
| Interface exists, no association/link | Cable, AP credentials, band/channel, RF block |
| Link up, no address | DHCP or static profile |
| Address works, names fail | DNS configuration |
| Ping works, application times out | Port, listener, policy, framing, server load |
| High throughput, stale commands | Queueing, deadlines and application scheduling |

A ping response does not prove an HTTP endpoint is listening. A successful `send()` does not prove that an application accepted or acted on the data.

## 3. Configure an existing NetworkManager connection

Use the actual interface and connection UUID from discovery. The examples change connectivity, so run them from a console or retain another working management path.

```bash
read -r -p 'Ethernet interface: ' NET_IF
read -r -p 'Connection UUID: ' NET_UUID
ethtool "$NET_IF"
sudo nmcli connection modify uuid "$NET_UUID" \
  ipv4.method auto ipv4.addresses '' ipv4.gateway '' ipv4.dns '' \
  ipv4.ignore-auto-dns no ipv4.ignore-auto-routes no
sudo nmcli connection up uuid "$NET_UUID" ifname "$NET_IF"
```

For a static network, obtain an unused address, prefix, gateway and DNS from that network's administrator. A private address copied from another lab is not a configuration plan. Clear stale manual settings when returning to DHCP. IPv6 is configured separately; changing IPv4 does not disable IPv6. See [NetworkManager's CLI manual](https://networkmanager.dev/docs/api/latest/nmcli.html).

For Wi-Fi, use a password prompt instead of putting a secret in shell history:

```bash
read -r -p 'Wi-Fi interface: ' WIFI_IF
read -r -p 'Network name: ' WIFI_SSID
nmcli device wifi list ifname "$WIFI_IF"
sudo nmcli --ask device wifi connect "$WIFI_SSID" ifname "$WIFI_IF"
ip address show dev "$WIFI_IF"
```

Choose one owner for the interface. Do not start a second standalone `wpa_supplicant`, DHCP client or `hostapd` over a NetworkManager-managed interface and then diagnose their conflict as radio failure.

An access point also requires adapter/driver AP capability, allowed channels, addressing and DHCP. Inspect `iw list`, including supported interface combinations, before assuming simultaneous client+AP operation. A NetworkManager hotspot profile can provide sharing; a bare `hostapd` file alone does not configure the entire IP path. [NetworkManager examples](https://networkmanager.dev/docs/api/latest/nmcli-examples.html).

## 4. Measure throughput and application delay separately

On a second machine you control, run `iperf3 -s -1`; it handles one test and exits. On Jetson, replace the server address with that machine's address:

```bash
read -r -p 'Test server address: ' TEST_SERVER
iperf3 -c "$TEST_SERVER" -t 10 -J > forward.json
```

Restart the one-shot server before measuring the reverse direction with `iperf3 -c "$TEST_SERVER" -t 10 -R -J`. Record link speed, MTU, direction, radio signal, power mode, versions and background traffic. Roughly 940 Mbit/s can occur on a well-configured Gigabit link; it is not a universal acceptance value or a result measured by this course. [Official iperf3 options](https://software.es.net/iperf/invoking.html).

**Worked units:** sending 1 MiB at a sustained 10 Mbit/s needs at least `1,048,576 × 8 / 10,000,000 = 0.839 s`, before application waiting. A 100-byte command can still be late if it sits behind that data in a queue. Bound queue sizes and decide which telemetry may be replaced by newer samples.

The [actual localhost lab](Localhost-Network-Lab.md) separates round-trip time from injected server work. Its measurements describe loopback and the host scheduler, not Wi-Fi range, RF loss or Jetson performance.

## 5. Bluetooth: distinguish discovery, services and characteristics

Use `bluetoothctl list`, `show`, `power on` and `scan on` to discover the actual controller and devices. Pair only with the intended test device, then inspect its services. Classic Bluetooth SPP/RFCOMM and BLE GATT are different protocols; ESP32-C6 supplies Bluetooth LE, not Classic Bluetooth audio or SPP.

A GATT **service** groups **characteristics**; reads and writes target characteristics. The UUID ending `180a` is the Device Information service, not a readable characteristic. Do not pass a service UUID to `read_gatt_char` and expect a measurement. Discover readable characteristics and their formats. [Bluetooth assigned numbers](https://www.bluetooth.com/specifications/assigned-numbers/).

This optional Bleak 3.0 API example was source-reviewed but not run against hardware:

```python
import asyncio
from bleak import BleakScanner, BleakClient

async def inspect_device(address):
    device = await BleakScanner.find_device_by_address(address, timeout=10.0)
    if device is None:
        raise RuntimeError('Requested device was not discovered')
    async with BleakClient(device) as client:
        for service in client.services:
            for characteristic in service.characteristics:
                print(service.uuid, characteristic.uuid, characteristic.properties)
                # Read only the intended documented characteristic after discovery.

# Supply the address of your own test device:
# asyncio.run(inspect_device('...'))
```

Pass the discovered device object, use one event loop, and handle disconnects. Notification payloads need documented units, byte order, sequence and timing. [Bleak client API](https://bleak.readthedocs.io/en/latest/api/client.html).

## 6. Choose the coprocessor architecture deliberately

| Requirement | Lesson | Where the network stack runs |
|---|---|---|
| Normal Linux Wi-Fi interface over SPI | [Current ESP-Hosted-Linux](ESP32-C6-ESP-Hosted-SPI-Jetson-Orin-Nano.md) | Linux TCP/IP; ESP Wi-Fi MAC/PHY |
| Thread radio attached to Linux | [OpenThread RCP](ESP32-C6-OpenThread-RCP-Jetson-Orin-Nano.md) | OpenThread core on Linux; radio/minimal MAC on ESP |
| Legacy Zigbee NCP experiment | [Versioned Zigbee options](ESP32-C6-Zigbee-NCP-Jetson-Orin-Nano.md) | Zigbee stack on NCP; gateway application on host |

A second radio is a useful isolation option, not a protocol requirement. Current ESP-Hosted-Linux can multiplex a supported RCP and Bluetooth with Wi-Fi, but a shared RF path and shared firmware restart still couple them. Validate coexistence with the exact firmware and traffic pattern.

## 7. Remote access and services

A VPN provides a protected network path; it does not define application authorization or command freshness. For WireGuard, use the [official setup](https://www.wireguard.com/quickstart/), create private keys with restrictive permissions (`umask 077`), and choose allowed routes deliberately. A keepalive interval is for maintaining a NAT mapping; it is not a robot watchdog. Keep private keys out of reports.

Start a management service on loopback while developing. An all-interface listener makes it reachable through every applicable interface and routing path. Before remote operation, add an explicit identity and permission model, bounded request sizes, deadlines, logs and a deployment server. A development Flask server with undefined temperature functions is not a complete device-management product.

For file transfer, first use SSH/SFTP through the established access path. If a product requires SMB, configure actual filesystem ownership and authenticated users; appending a share stanza does not create a directory or grant appropriate access. Bulk transfer should not indefinitely delay time-sensitive control traffic.

## 8. Completion and diagnostic answers

Submit the localhost report and explain these cases:

1. **A command timed out after the server acted. Is retrying harmless?** Only if the command is intrinsically idempotent or a stable request ID is deduplicated across the applicable failure window.
2. **Can two machines compare raw `monotonic_ns()` values?** No. Their origins are independent. Use a clock synchronization contract with an error bound, receiver-issued leases, or another explicitly defined freshness protocol.
3. **Does a new `wlanX` prove reliable connectivity?** It proves interface registration. Association, address assignment, traffic, reconnect and coexistence need separate evidence.
4. **Does Thread `leader` prove border routing?** No. It describes the Thread partition role; test routed IPv6 between Thread and the infrastructure network separately.

Continue to [ROS 2](../6.%20ROS2/Guide.md) with a measured message-size/rate budget and a clear stale-data policy.

The [source evidence record](labs/source-evidence-2026-10-18.json) retains immutable revisions, fetched source hashes and obsolete-path failures found during review. An upstream source check does not replace a firmware build or RF test.
