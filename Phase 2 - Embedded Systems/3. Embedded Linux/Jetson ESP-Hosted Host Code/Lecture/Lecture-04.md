# Lecture 4 — How Wi-Fi becomes `wlan0`

**Course:** [Jetson ESP-Hosted Host Code guide](../Guide.md) · **Phase 2 — Embedded Linux**

**Previous:** [Lecture 03](Lecture-03.md) · **Next:** [Lecture 05 — How BLE becomes `hci0`](Lecture-05.md)

---

## 1. Linux does not want “an SPI Wi-Fi gadget”

Linux userspace tools do not know or care that your radio sits behind a custom SPI transport.

They want:

- a registered wireless device
- a `wiphy`
- a `wireless_dev`
- a net device that userspace can manage

That mapping happens in:

- `esp_hosted_ng/host/esp_cfg80211.c`
- plus the orchestration in `esp_hosted_ng/host/main.c`

## Related Linux kernel concepts

This lecture is best read alongside:

- [OS Lecture 17 — Linux Device Driver Model & Device Tree](../../../../Phase%201%20-%20Foundational%20Knowledge/3.%20Operating%20Systems/Lectures/Lecture-17.md)
- [OS Lecture 1 — Modern OS Architecture & the Linux Kernel](../../../../Phase%201%20-%20Foundational%20Knowledge/3.%20Operating%20Systems/Lectures/Lecture-01.md)

Lecture 17 is the right background because `wiphy`, `wireless_dev`, and `net_device` are all kernel objects that must be allocated and registered in the right order. `esp_cfg80211.c` is a concrete example of the driver model creating Linux-visible objects that represent a remote piece of hardware.

Lecture 1 matters because userspace never sees the transport details directly. `nmcli` and `iw` work only because the driver turns the ESP into standard Linux wireless abstractions instead of exposing “some SPI radio” as a custom API.

The useful mental model is that `cfg80211` is a **contract between your driver and the Linux wireless stack**. The driver has to describe what the radio can do, create the objects Linux expects, and implement the operations Linux will call for scan, connect, disconnect, and regulatory handling.

That is why `wlan0` appearing is a strong signal. It means the code has progressed far beyond transport bring-up and into successful subsystem registration, where the kernel now believes there is a manageable wireless interface that userspace tools can operate on normally.

The kernel interfaces to keep in your head are:

- `wiphy_new(...)`
- `wiphy_register(...)`
- `esp_cfg80211_add_iface(...)`
- `register_netdevice(...)`
- `rtnl_lock()`

---

## 2. The main control flow

In `main.c`, study:

- `process_esp_bootup_event(...)`
- `process_internal_event(...)`
- `process_rx_packet(...)`
- `esp_add_network_ifaces(...)`

The logic is:

1. transport receives ESP boot-up event
2. host validates the chipset
3. host learns capabilities
4. host initializes the Linux-facing interface layer
5. the Wi-Fi side gets registered

This is what turns **“working protocol” into “working Linux network interface.”**

Here is the exact orchestration in `main.c`:

```c
static int process_event_esp_bootup(struct esp_adapter *adapter, u8 *evt_buf, u8 len)
{
	...
	while (len_left > 0) {
		switch (*pos) {
		case ESP_BOOTUP_CAPABILITY:
			adapter->capabilities = *(pos + 2);
			break;
		case ESP_BOOTUP_FIRMWARE_CHIP_ID:
			ret = esp_validate_chipset(adapter, *(pos + 2));
			break;
		case ESP_BOOTUP_FW_DATA:
			fw_p = (struct fw_data *)(pos + 2);
			ret = process_fw_data(fw_p, tag_len);
			break;
		case ESP_BOOTUP_SPI_CLK_MHZ:
			ret = esp_adjust_spi_clock(adapter, *(pos + 2));
			break;
		}
		...
	}

	if (esp_add_card(adapter)) {
		esp_err("network interface init failed\n");
		return -1;
	}
	init_bt(adapter);
	...
}
```

This is the major control-plane jump in the codebase:

- raw boot event comes in over SPI
- driver learns what the remote chip is
- Linux-facing interfaces get built on top of that knowledge

---

## 3. The exact place where `wlan0` is created

In `main.c`, `esp_add_network_ifaces(...)` calls:

- `esp_cfg80211_add_iface(adapter->wiphy, "wlan%d", 1, NL80211_IFTYPE_STATION, NULL)`

That is the moment where the code requests a normal Linux wireless interface.

That one call is a great Embedded Linux lesson:

- the transport does not expose raw packets to userspace directly
- the driver **translates the remote radio into a standard Linux interface model**

That is why `nmcli`, `iw`, and other tools can work.

The call site is small enough to memorize:

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

---

## 4. What `esp_cfg80211.c` contributes

Read these areas in `esp_cfg80211.c`:

- `esp_add_wiphy(...)`
- `esp_cfg80211_add_iface(...)`
- `esp_cfg80211_scan(...)`
- `esp_cfg80211_disconnect(...)`
- `esp_cfg80211_connect(...)`
- the `esp_cfg80211_ops` table

This file is your map from:

- Linux wireless subsystem expectations

to:

- command messages sent to the ESP

The major ideas are:

- create and register a `wiphy`
- declare supported bands, rates, and capabilities
- provide operation handlers for scan/connect/disconnect/etc.
- allocate and register the interface objects Linux expects

That is standard subsystem integration work. The ESP is remote, but Linux still wants the usual `cfg80211` contract.

Two short excerpts show the division of labor well.

First, `esp_add_wiphy(...)` creates and advertises the wireless capabilities Linux expects:

```c
int esp_add_wiphy(struct esp_adapter *adapter)
{
	struct wiphy *wiphy;
	...
	wiphy = wiphy_new(&esp_cfg80211_ops, sizeof(struct esp_device));
	...
	wiphy->interface_modes = BIT(NL80211_IFTYPE_STATION);
	wiphy->bands[NL80211_BAND_2GHZ] = &esp_wifi_bands_2ghz;
	...
	wiphy->max_scan_ssids = 10;
	wiphy->signal_type = CFG80211_SIGNAL_TYPE_MBM;
	wiphy->reg_notifier = esp_reg_notifier;
	...
	ret = wiphy_register(wiphy);
	return ret;
}
```

Then `esp_cfg80211_add_iface(...)` allocates the netdev and asks the ESP firmware to initialize the remote interface:

```c
struct wireless_dev *esp_cfg80211_add_iface(struct wiphy *wiphy,
		const char *name, unsigned char name_assign_type,
		enum nl80211_iftype type, struct vif_params *params)
{
	...
	ndev = ALLOC_NETDEV(sizeof(struct esp_wifi_device), name,
			    name_assign_type, ether_setup);
	...
	esp_wdev->wdev.iftype = type;
	...
	if (cmd_init_interface(esp_wdev))
		goto free_and_return;

	if (cmd_get_mac(esp_wdev))
		goto free_and_return;

	ETH_HW_ADDR_SET(ndev, esp_wdev->mac_address);
	...
	if (register_netdevice(ndev))
		goto free_and_return;
}
```

That is the **Embedded Linux pattern worth learning**:

- build a standard subsystem object locally
- synchronize it with the remote device through command messages
- only then register it with the kernel networking stack

---

## 5. Why `wlan0` is such a strong signal

`wlan0` appearing means much more than “the SPI bus works.”

It implies:

- the host transport is alive
- boot-up event handling completed far enough to initialize upper layers
- `cfg80211` registration succeeded
- interface allocation and registration succeeded

That is why the validated Jetson bring-up used `wlan0` as a major milestone.

It is the difference between:

- low-level electrical or protocol success

and:

- actual Linux subsystem success

---

## 6. What userspace is really exercising

When you run:

- `nmcli dev wifi list`
- `nmcli dev wifi connect ...`

you are indirectly testing:

- `cfg80211` hooks
- command message flow to the ESP
- event and response handling back into Linux

So a Wi-Fi scan is not “just a user command.” It is a **full end-to-end test** of:

- Linux subsystem registration
- transport reliability
- ESP firmware command handling

That is what makes this repo valuable as an Embedded Linux teaching example.

---

## Lab

Answer these:

1. Where is the `wiphy` created?
2. Where is the station interface requested?
3. Why is `wlan0` stronger proof than a successful `insmod`?
4. Which user-space command best tests the full Wi-Fi path after bring-up?

Optional:

- map `nmcli dev wifi list` to the likely `cfg80211` call path at a high level

---

**Previous:** [Lecture 03](Lecture-03.md) · **Next:** [Lecture 05 — How BLE becomes `hci0`](Lecture-05.md)
