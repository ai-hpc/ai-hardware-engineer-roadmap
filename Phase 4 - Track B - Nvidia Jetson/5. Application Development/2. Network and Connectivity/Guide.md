# Network and Connectivity

<div class="course-identity auto-course" style="--course-accent: #4f46e5; --course-accent-rgb: 79, 70, 229;" markdown="1">
<div class="course-identity__icon">NAC</div>
<div markdown="1">
<p class="course-identity__eyebrow">Deep Dive · Jetson Track</p>
<p class="course-identity__title">Specialized course identity for Network and Connectivity.</p>
<p class="course-identity__meta">Artifact: Jetson integration demo · Measure: latency, memory, power, logs</p>
</div>
</div>


**Phase 4 — Track B — Module 5.2** · Application Development

> **Focus:** Configure wired and wireless networking on the **Jetson Orin Nano 8GB** — Ethernet, Wi-Fi (client and AP), Bluetooth, VPN tunnels, and lightweight web servers for device management.

**Hub:** [5. Application Development](../Guide.md)

---


## 1. Ethernet (Linux)

Gigabit Ethernet is the primary network interface on most Jetson carriers.

### Configuration

```bash
# Check link status
ip link show eth0
ethtool eth0

# Static IP
sudo nmcli con mod "Wired connection 1" \
    ipv4.addresses 192.168.1.100/24 \
    ipv4.gateway 192.168.1.1 \
    ipv4.dns "8.8.8.8" \
    ipv4.method manual
sudo nmcli con up "Wired connection 1"

# DHCP (default)
sudo nmcli con mod "Wired connection 1" ipv4.method auto
```

### Performance validation

```bash
# Install iperf3
sudo apt install iperf3

# Server (on another machine)
iperf3 -s

# Client (on Jetson)
iperf3 -c <server-ip>
# Expected: ~940 Mbps for GbE
```

---

## 2. Wi-Fi client (Linux)

Jetson Orin Nano dev kit does not include Wi-Fi — add it via USB dongle or M.2 Key E module (Intel AX200/AX210, Realtek RTL8852).

If you want a more embedded integration path, you can also attach an **ESP32-C6** as a wireless coprocessor over SPI with **ESP-Hosted-NG**. See [ESP32-C6 ESP-Hosted over SPI on Jetson Orin Nano](ESP32-C6-ESP-Hosted-SPI-Jetson-Orin-Nano.md).

If you also want **Thread / 802.15.4** on the same Jetson, the cleaner architecture is to keep that first ESP32-C6 for Wi-Fi/BLE and add a **second ESP32-C6** as an **OpenThread RCP**. See [ESP32-C6 OpenThread RCP on Jetson Orin Nano](ESP32-C6-OpenThread-RCP-Jetson-Orin-Nano.md).

If your current Jetson image is missing the kernel support needed for full OTBR routing, a second ESP32-C6 can also be used as a **Zigbee coprocessor** instead. That path stays in the same 802.15.4 family but shifts from OTBR's IPv6 border-router model to a host-controlled Zigbee coprocessor model. See [ESP32-C6 Zigbee NCP on Jetson Orin Nano](ESP32-C6-Zigbee-NCP-Jetson-Orin-Nano.md).

### Connect with NetworkManager

```bash
# Scan for networks
nmcli dev wifi list

# Connect
sudo nmcli dev wifi connect "SSID" password "password"

# Verify
nmcli con show --active
ip addr show wlan0
```

### Connect with wpa_supplicant (headless)

```bash
# /etc/wpa_supplicant/wpa_supplicant.conf
cat <<EOF | sudo tee /etc/wpa_supplicant/wpa_supplicant.conf
network={
    ssid="SSID"
    psk="password"
}
EOF

sudo wpa_supplicant -B -i wlan0 -c /etc/wpa_supplicant/wpa_supplicant.conf
sudo dhclient wlan0
```

---

## 3. Wi-Fi access point mode (Linux)

Run the Jetson as a Wi-Fi hotspot for direct device connection (useful for field configuration or standalone operation).

### Using NetworkManager

```bash
sudo nmcli dev wifi hotspot ifname wlan0 \
    ssid "Jetson-Setup-AP" password "securepass"
```

### Using hostapd (more control)

```bash
sudo apt install hostapd dnsmasq

# /etc/hostapd/hostapd.conf
cat <<EOF | sudo tee /etc/hostapd/hostapd.conf
interface=wlan0
driver=nl80211
ssid=Jetson-Setup-AP
hw_mode=g
channel=7
wmm_enabled=0
auth_algs=1
wpa=2
wpa_passphrase=securepass
wpa_key_mgmt=WPA-PSK
rsn_pairwise=CCMP
EOF

# /etc/dnsmasq.conf (DHCP for AP clients)
cat <<EOF | sudo tee /etc/dnsmasq.d/ap.conf
interface=wlan0
dhcp-range=192.168.4.2,192.168.4.50,255.255.255.0,24h
EOF

sudo systemctl start hostapd
sudo systemctl start dnsmasq
```

---

## 4. Bluetooth (Linux)

Bluetooth is available via USB dongle or M.2 combo Wi-Fi/BT module.

### Basic operations

```bash
# Check adapter
bluetoothctl
  power on
  agent on
  scan on
  # Wait for devices...
  pair <MAC>
  connect <MAC>
```

### Bluetooth serial (SPP / RFCOMM)

```bash
# Listen for incoming serial connections
sudo rfcomm listen /dev/rfcomm0 1

# Connect to a device
sudo rfcomm connect /dev/rfcomm0 <MAC> 1
```

### Bluetooth Low Energy (BLE)

```bash
# Scan for BLE devices
sudo hcitool lescan

# Use bluetoothctl for GATT operations
bluetoothctl
  menu gatt
  list-attributes <MAC>
  select-attribute <UUID>
  read
  write <value>
```

### Python (bleak for BLE)

```python
import asyncio
from bleak import BleakScanner, BleakClient

async def main():
    devices = await BleakScanner.discover()
    for d in devices:
        print(d)

    async with BleakClient("AA:BB:CC:DD:EE:FF") as client:
        value = await client.read_gatt_char("0000180a-0000-1000-8000-00805f9b34fb")
        print(value)

asyncio.run(main())
```

---

## 5. VPN (WireGuard / OpenVPN)

### WireGuard (recommended — faster, simpler)

```bash
sudo apt install wireguard

# Generate keys
wg genkey | tee privatekey | wg pubkey > publickey

# /etc/wireguard/wg0.conf
cat <<EOF | sudo tee /etc/wireguard/wg0.conf
[Interface]
PrivateKey = $(cat privatekey)
Address = 10.0.0.2/24

[Peer]
PublicKey = <server-public-key>
Endpoint = <server-ip>:51820
AllowedIPs = 10.0.0.0/24
PersistentKeepalive = 25
EOF

sudo wg-quick up wg0
```

### OpenVPN

```bash
sudo apt install openvpn
sudo openvpn --config client.ovpn
```

### Use case for deployed Jetson devices

- **Remote access:** SSH into field devices through VPN tunnel
- **Fleet management:** All devices on same VPN for OTA and telemetry
- **Security:** Encrypted communication even on untrusted networks

---

## 6. SMB / network file sharing

Share files between Jetson and other machines on the local network.

```bash
# Install Samba
sudo apt install samba

# Create a share
cat <<EOF | sudo tee -a /etc/samba/smb.conf
[jetson-data]
   path = /data
   browseable = yes
   read only = no
   guest ok = no
EOF

sudo smbpasswd -a $(whoami)
sudo systemctl restart smbd
```

Access from other machines: `\\<jetson-ip>\jetson-data`

---

## 7. Web server (Linux)

Run a lightweight web server on the Jetson for device management, configuration UI, or API endpoints.

### nginx (reverse proxy / static files)

```bash
sudo apt install nginx
sudo systemctl start nginx
# Access at http://<jetson-ip>/
```

### Python Flask (REST API)

```python
from flask import Flask, jsonify
app = Flask(__name__)

@app.route('/api/status')
def status():
    return jsonify({
        'device': 'JetsonEdge',
        'temperature': read_temperature(),
        'uptime': read_uptime()
    })

app.run(host='0.0.0.0', port=8080)
```

### Use cases

| Use case | Technology |
|----------|-----------|
| **Device config portal** | Flask/FastAPI + HTML form |
| **REST API for sensors** | Flask/FastAPI returning JSON |
| **OTA trigger endpoint** | nginx + webhook |
| **Camera stream** | GStreamer RTSP or MJPEG over HTTP |

---

## 8. Projects

- **Headless Wi-Fi setup:** Build a provisioning flow where the Jetson starts as a Wi-Fi AP, serves a web page for entering Wi-Fi credentials, then switches to client mode.
- **[ESP32-C6 ESP-Hosted over SPI on Jetson Orin Nano](ESP32-C6-ESP-Hosted-SPI-Jetson-Orin-Nano.md):** Bring up an external Wi-Fi coprocessor on SPI1 with handshake, data-ready, and reset GPIOs.
- **[ESP32-C6 Zigbee NCP on Jetson Orin Nano](ESP32-C6-Zigbee-NCP-Jetson-Orin-Nano.md):** Use a second ESP32-C6 as a Zigbee coprocessor and treat Jetson as the higher-level Zigbee host or gateway controller.
- **BLE sensor gateway:** Read BLE sensor data (temperature, humidity) and publish to MQTT over Ethernet.
- **VPN fleet:** Set up WireGuard between 3 Jetson devices and a cloud server. Verify SSH access to all devices from the server.
- **Device management API:** Build a Flask REST API that exposes device status (temperature, disk, uptime, firmware version) and accepts OTA trigger commands.

---

## 9. Resources

| Resource | Description |
|----------|-------------|
| **NetworkManager CLI** | `nmcli` documentation for connection management |
| **WireGuard** (wireguard.com) | Modern VPN protocol, kernel-integrated |
| **hostapd** | Wi-Fi access point daemon documentation |
| **BlueZ** | Official Linux Bluetooth stack |
| **bleak** | Cross-platform BLE library for Python |
| **nginx** | Lightweight web server / reverse proxy |
| **ESP-Hosted-NG** | Espressif Linux-hosted Wi-Fi/Bluetooth transport for ESP peripherals over SPI/SDIO/UART |
