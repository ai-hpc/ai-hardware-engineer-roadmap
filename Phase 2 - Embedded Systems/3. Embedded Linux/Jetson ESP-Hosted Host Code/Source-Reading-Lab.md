# Source Lab — Trace Success, Failure and Ownership

**Time:** 60–90 minutes. **Prerequisites:** C pointers, return codes, a terminal and the [course guide](Guide.md). **Outcome:** inspect a real driver revision and explain both its useful architecture and concrete error-path defects. This is a source-reading exercise; no module is loaded and no hardware performance is measured.

## 1. Freeze the example before reading it

This course studies the historical Jetson NG fork at **`ce36dec9294988e598f04e84c9d28092ca9e9786`**, dated April 20, 2026. Its `esp_hosted_ng/host` layout, global GPIO parameters and manual reset observations belong to that revision. They are not the current ESP-Hosted-Linux interface.

For current firmware/driver setup, use the [October 18 Linux migration lesson](../../../Phase%204%20-%20Track%20B%20-%20Nvidia%20Jetson/5.%20Application%20Development/2.%20Network%20and%20Connectivity/ESP32-C6-ESP-Hosted-SPI-Jetson-Orin-Nano.md). It uses a different repository generation and device-tree GPIO descriptors. This source-reading course keeps the older case so you can practice examining real engineering history rather than assuming a newer API was always present.

The [source manifest](evidence/sources.json) pins seven small fork files and a Linux v5.15 HCI reference. From **this course directory**, download and verify those exact bytes:

```bash
python3 - <<'PY'
import hashlib
import json
from pathlib import Path
from urllib.request import urlopen

root = Path('/tmp/course-esp-source-reading')
for row in json.loads(Path('evidence/sources.json').read_text()):
    data = urlopen(row['url'], timeout=30).read()
    if hashlib.sha256(data).hexdigest() != row['sha256']:
        raise SystemExit('Source hash mismatch: ' + row['path'])
    destination = root / row['path']
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_bytes(data)
    print(row['path'], len(data), 'bytes verified')
PY
```

The download needs network access. These files are enough for the reading tasks, **not a complete kernel build tree**. The course recorded all eight hashes and ran the helper's syntax/help paths:

```bash
bash -n /tmp/course-esp-source-reading/esp_hosted_ng/host/jetson_orin_nano_init.sh
bash /tmp/course-esp-source-reading/esp_hosted_ng/host/jetson_orin_nano_init.sh --help
```

Read the script before invoking any other arguments. Its ordinary path builds, unloads/loads modules and may unbind a device; that is a separate hardware bring-up exercise. `--help` exits during argument parsing before that path.

## 2. Draw the success path

Use a text editor or `rg -n` on the downloaded files. Trace:

```text
helper parameters → SPI device/IRQs → received boot event
                                      ├─ Wi-Fi object registration
                                      └─ HCI controller registration
```

Locate `process_esp_bootup_event`, which wraps the event, and `process_event_esp_bootup`, which parses its contents. These similarly named functions are distinct. Then locate `esp_add_network_ifaces`, `esp_cfg80211_add_iface` and `esp_init_bt`.

For each edge, write the input, success return, error return and resource owner. `wlan%d` requests an available interface number; `hci_register_dev` does not promise controller zero. Use actual enumeration on hardware. A registered interface can remain visible after the peer stops responding; registration alone does not establish current bus health.

## 3. Read the failure path with equal care

These are **source findings at the pinned revision**, not reproduced kernel faults or a complete patch. Use the linked functions to verify them yourself.

| Location | Observation | What a correction must preserve |
|---|---|---|
| `esp_cfg80211_add_iface` and its direct caller | Netdev allocation failure returns `ERR_PTR(-ENOMEM)`, while `esp_add_network_ifaces` tests only `if (wdev)` | Encoded error pointers are non-null; distinguish success, null and error consistently across every caller |
| Wi-Fi `free_and_return` | `free_netdev(ndev)` precedes assignments through `esp_wdev`, obtained from `netdev_priv(ndev)` | Stop using the private storage once its owner is released; clear external references in a valid order |
| HCI receive in `process_rx_packet` | The code reads `*type` and `skb->len` after passing `skb` to `hci_recv_frame` | Save required scalar metadata before handing off ownership; do not read the packet afterward |
| HCI registration failure | `hci_register_dev` failure frees `hdev`, but the earlier `adapter->hcidev` assignment is left intact | Clear the adapter's reference on failure and propagate a meaningful error |
| Boot-event TLV loop | The loop reads a two-byte header and payload values without first checking that the complete tag fits locally | Validate header length, remaining payload and tag-specific minimum/exact sizes before dereferencing |

[Legacy main.c](https://github.com/ai-hpc/jetson-esp-hosted/blob/ce36dec9294988e598f04e84c9d28092ca9e9786/esp_hosted_ng/host/main.c), [Wi-Fi implementation](https://github.com/ai-hpc/jetson-esp-hosted/blob/ce36dec9294988e598f04e84c9d28092ca9e9786/esp_hosted_ng/host/esp_cfg80211.c), [HCI implementation](https://github.com/ai-hpc/jetson-esp-hosted/blob/ce36dec9294988e598f04e84c9d28092ca9e9786/esp_hosted_ng/host/esp_bt.c).

The Linux v5.15 `hci_recv_frame` reference frees rejected packets or queues accepted packets for asynchronous processing. Therefore a successful return does not grant the caller continued ownership of `skb`. A pointer into its data has the same lifetime problem as the packet itself. Inspect the exact target kernel when adapting this reasoning. [Linux HCI source](https://github.com/torvalds/linux/blob/v5.15/net/bluetooth/hci_core.c), [packet buffer structure](https://docs.kernel.org/networking/skbuff.html).

**Worked answer:** for a one-byte TLV remainder, testing `len_left>0` permits the loop to read `pos[1]` beyond the local remainder. Before reading the length byte, require at least two bytes; before reading the payload, require `tag_len <= len_left-2`; then check the selected tag's format. A checksum does not replace these bounds checks. This is a reasoning exercise about lengths, not a claim that this exact malformed input reaches every deployed driver.

## 4. Separate scan evidence from a working application

A NetworkManager list can contain cached results. For a hardware exercise, request a fresh scan on the selected interface, observe completion and a timestamped result, then test association, address assignment and application traffic separately. Similarly, a BLE advertisement scan tests discovery, not pairing, GATT reads/writes, throughput or restart recovery. ESP32-C6 does not provide classic Bluetooth merely because Linux exposes an HCI controller.

Treat the fork README's reported board observations as attributed historical evidence. A helper's final “success” message only shows that its own control path reached that print; it does not establish every radio/application milestone.

## 5. Submit a review another learner can follow

Submit the verified manifest, a success-path diagram, an ownership table, and two error-path explanations. For each proposed correction, describe a regression test and what environment it would need. An allocation-failure path needs fault injection or a suitable test seam; a source-level observation is not a passing runtime test.

Finish with the [localhost transport lab](../../../Phase%204%20-%20Track%20B%20-%20Nvidia%20Jetson/5.%20Application%20Development/2.%20Network%20and%20Connectivity/Localhost-Network-Lab.md) to execute framing, freshness and reconnect behavior on a CPU. Keep that application-level result separate from SPI/RF claims. The [review record](evidence/review-2026-10-18.json) states exactly what was inspected and executed here.
