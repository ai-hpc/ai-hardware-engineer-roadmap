# Lecture 1 — Linux mental model for this host stack

**Course:** [Jetson ESP-Hosted Host Code guide](../Guide.md) · **Phase 2 — Embedded Linux**

**Next:** [Lecture 02 — Build, load, and board policy](Lecture-02.md)

---

## 1. What Linux thinks is happening

From Linux userspace, the validated Jetson flow eventually looks ordinary:

- `wlan0` exists
- `hci0` exists
- `nmcli` can scan Wi-Fi
- `bluetoothctl` can scan BLE devices

But Linux is not talking to a native Wi-Fi or Bluetooth chip over PCIe or USB. It is talking to an **ESP32-C6** over a custom **SPI transport**.

So the real question is:

How do ordinary Linux abstractions come out of a non-ordinary transport?

The answer is the host stack in `esp_hosted_ng/host/`.

## Related Linux kernel concepts

Use these earlier OS lectures as the conceptual base under this case study:

- [OS Lecture 1 — Modern OS Architecture & the Linux Kernel](../../../../Phase%201%20-%20Foundational%20Knowledge/3.%20Operating%20Systems/Lectures/Lecture-01.md)
- [OS Lecture 5 — Kernel Modules, Boot Process & Device Tree](../../../../Phase%201%20-%20Foundational%20Knowledge/3.%20Operating%20Systems/Lectures/Lecture-05.md)

Lecture 1 from the OS course explains the kernel as the layer that **owns hardware and exposes clean abstractions** to userspace. That is exactly what this host stack is doing: it hides a custom ESP-over-SPI link and instead gives Linux normal objects like `wlan0` and `hci0`.

Lecture 5 matters here because this code only makes sense after boot has already described the board and loaded the right driver path. The ESP-Hosted host code is not inventing hardware from scratch; it is sitting on top of a kernel that already knows about SPI devices, modules, and board description.

Another way to say it is: userspace does not get to “talk to GPIO 471” or “talk to SPI mode 2” directly as a networking model. The kernel driver absorbs those low-level details and publishes higher-level subsystem objects, so userspace can work with Wi‑Fi and Bluetooth concepts instead of transport wiring.

This is why `wlan0` and `hci0` are such important **milestones in the bring-up story**. They prove not just that bytes moved over SPI, but that the kernel accepted the driver’s integration with **real Linux subsystems** and exposed the expected interfaces upward.

The Linux-facing interfaces to keep in mind in this lecture are:

- `cfg80211`
- HCI / BlueZ
- `wireless_dev`
- `wiphy`
- `net_device`

---

## 2. The layered picture

Read this from top to bottom:

1. userspace tools
2. Linux subsystem APIs
3. host driver integration code
4. transport code
5. ESP side firmware

In this codebase, that becomes:

- userspace:
  - `nmcli`
  - `iw`
  - `bluetoothctl`
  - `hciconfig`
- Linux subsystems:
  - `cfg80211` for Wi-Fi
  - HCI / BlueZ for Bluetooth/BLE
- host integration:
  - `esp_cfg80211.c`
  - `esp_bt.c`
  - `main.c`
- transport:
  - `spi/esp_spi.c`
- board bring-up policy:
  - `jetson_orin_nano_init.sh`

That is the first **Embedded Linux lesson**:

- the Linux-facing code is not the same thing as
- the hardware-facing transport code

If you blur those together, driver reading becomes confusing fast.

### Code anchor: one boot event creates two Linux personalities

The cleanest place to see the split is in `main.c`. A single ESP boot-up event eventually fans out into:

- Wi-Fi registration through `esp_add_card(...)`
- Bluetooth registration through `init_bt(...)`

```c
static void init_bt(struct esp_adapter *adapter)
{
	if ((adapter->capabilities & ESP_BT_SPI_SUPPORT) ||
		(adapter->capabilities & ESP_BT_SDIO_SUPPORT)) {
		msleep(200);
		esp_info("ESP Bluetooth init\n");
		esp_init_bt(adapter);
	}
}

static int process_event_esp_bootup(struct esp_adapter *adapter, u8 *evt_buf, u8 len)
{
	...
	if (esp_add_card(adapter)) {
		esp_err("network interface init failed\n");
		return -1;
	}
	init_bt(adapter);
	...
	print_capabilities(adapter->capabilities);
}
```

And the Wi-Fi side is explicitly created as a normal Linux station interface:

```c
static int esp_add_network_ifaces(struct esp_adapter *adapter)
{
	struct wireless_dev *wdev = NULL;

	rtnl_lock();
	wdev = esp_cfg80211_add_iface(adapter->wiphy, "wlan%d", 1,
				      NL80211_IFTYPE_STATION, NULL);
	rtnl_unlock();

	if (wdev)
		return 0;

	return -1;
}
```

That is the **whole course in miniature**:

- one remote chip
- one transport
- two different Linux subsystem identities

---

## 3. The file-reading order that actually works

Use this order:

1. `jetson_orin_nano_init.sh`
2. `Makefile`
3. `main.c`
4. `spi/esp_spi.c`
5. `esp_cfg80211.c`
6. `esp_bt.c`

Why this order works:

- the shell script tells you the board assumptions
- the `Makefile` tells you the kernel module shape
- `main.c` shows the lifecycle
- `esp_spi.c` shows how the bytes actually move
- `esp_cfg80211.c` shows how Linux Wi-Fi appears
- `esp_bt.c` shows how Linux Bluetooth appears

Do not start in the middle of `esp_spi.c` unless you already know what the host is trying to produce.

---

## 4. What the host code is really producing

The host code is not just “a kernel module.”

It is producing two Linux personalities:

- a Wi-Fi device through `cfg80211`
- an HCI controller through the Bluetooth stack

That means this repo is really a study in **Linux subsystem integration**.

The transport could change in theory:

- SPI
- SDIO
- UART

But the way Linux sees Wi-Fi and Bluetooth would still need to map into the same upper subsystem expectations.

That is why the repo splits:

- common logic
- transport-specific logic
- Linux integration logic

---

## 5. The validated runtime story

On the validated Jetson path, the important runtime stages were:

1. module inserted
2. SPI device claimed
3. handshake/data-ready IRQs attached
4. ESP boot-up event received
5. chipset identified over SPI
6. Wi-Fi interface created
7. Bluetooth controller created

That sequence matters.

For Embedded Linux debugging, **“module inserted successfully” is not enough**.

You want to know:

- did the protocol event arrive?
- did the upper subsystem register?
- did Linux gain a real interface?

This is how you avoid false success during bring-up.

---

## 6. The exact files to have open while reading

Keep these open side by side:

- `esp_hosted_ng/host/jetson_orin_nano_init.sh`
- `esp_hosted_ng/host/Makefile`
- `esp_hosted_ng/host/main.c`
- `esp_hosted_ng/host/spi/esp_spi.c`
- `esp_hosted_ng/host/esp_cfg80211.c`
- `esp_hosted_ng/host/esp_bt.c`

As you read, ask:

- what Linux subsystem is this file talking to?
- what hardware assumption does it encode?
- what log line would prove this stage worked?

That habit is the real course objective.

---

## Lab

Write a one-page note that answers:

1. Which files are mostly **board policy**?
2. Which files are mostly **transport logic**?
3. Which files are mostly **Linux subsystem integration**?
4. Why does `wlan0` mean more than `/dev/spidev0.0`?

---

**Previous:** [Course hub](../Guide.md) · **Next:** [Lecture 02 — Build, load, and board policy](Lecture-02.md)
