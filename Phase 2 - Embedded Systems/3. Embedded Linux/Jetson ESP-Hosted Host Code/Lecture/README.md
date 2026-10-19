# Jetson ESP-Hosted Host Code — lectures

**Source boundary — reviewed October 18, 2026:** this is the historical NG fork at `ce36dec9294988e598f04e84c9d28092ca9e9786`. Board observations are attributed to that fork, not rerun here. Complete the [source-reading lab](../Source-Reading-Lab.md); it explains the current migration path and defects in these abbreviated source excerpts. Snippets containing `...` are reading excerpts, not standalone compilable programs.

Work through these in order. Each file is one session built around the real `esp_hosted_ng/host/` Linux host code.

**Hub:** [Jetson ESP-Hosted Host Code guide](../Guide.md)

| Lecture | Topic | Main kernel interfaces | Main concepts |
|---------|-------|------------------------|---------------|
| [Lecture-01](Lecture-01.md) | Linux mental model for this host stack | `cfg80211`, HCI, netdev, SPI transport split | subsystem boundaries |
| [Lecture-02](Lecture-02.md) | Build, load, and board policy on Jetson | kernel modules, `module_param`, `spidev` ownership | board policy vs driver code |
| [Lecture-03](Lecture-03.md) | SPI transport, GPIOs, and IRQ-driven bring-up | `gpio_to_irq`, `request_irq`, SPI device reuse | interrupt-driven transport |
| [Lecture-04](Lecture-04.md) | How Wi-Fi becomes `wlan0` | `wiphy_new`, `wiphy_register`, `register_netdevice` | Linux wireless subsystem integration |
| [Lecture-05](Lecture-05.md) | How BLE becomes `hci0` and how to validate the full path | `hci_alloc_dev`, `hci_register_dev`, `hci_recv_frame` | HCI and BLE host/controller split |

## Use one evidence chain across the five lectures

L1 produces a subsystem map; L2 produces a build/board manifest; L3 produces an IRQ-to-transfer trace; L4 verifies Wi-Fi registration and traffic separately; L5 verifies Bluetooth controller registration, an actual exchange, and recovery. Each result names the host kernel, fork commit, peer firmware, carrier/pinmux, and connection used.

At the final review, restart the peer while a bounded test is running and explain whether pending buffers are drained, cancelled, or retried. Do not mix Wi-Fi data-plane success with Bluetooth success just because both share a transport.

**Carry forward:** A board-to-subsystem evidence chain with separate Wi-Fi/Bluetooth results. Continue with [the next course](../../../4.%20Product%20Design/Guide.md) and reuse that evidence.
