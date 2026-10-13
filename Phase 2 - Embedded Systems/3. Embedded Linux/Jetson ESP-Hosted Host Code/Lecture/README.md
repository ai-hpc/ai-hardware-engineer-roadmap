# Jetson ESP-Hosted Host Code — lectures

Work through these in order. Each file is one session built around the real `esp_hosted_ng/host/` Linux host code.

**Hub:** [Jetson ESP-Hosted Host Code guide](../Guide.md)

| Lecture | Topic | Main kernel interfaces | Main concepts |
|---------|-------|------------------------|---------------|
| [Lecture-01](Lecture-01.md) | Linux mental model for this host stack | `cfg80211`, HCI, netdev, SPI transport split | subsystem boundaries |
| [Lecture-02](Lecture-02.md) | Build, load, and board policy on Jetson | kernel modules, `module_param`, `spidev` ownership | board policy vs driver code |
| [Lecture-03](Lecture-03.md) | SPI transport, GPIOs, and IRQ-driven bring-up | `gpio_to_irq`, `request_irq`, SPI device reuse | interrupt-driven transport |
| [Lecture-04](Lecture-04.md) | How Wi-Fi becomes `wlan0` | `wiphy_new`, `wiphy_register`, `register_netdevice` | Linux wireless subsystem integration |
| [Lecture-05](Lecture-05.md) | How BLE becomes `hci0` and how to validate the full path | `hci_alloc_dev`, `hci_register_dev`, `hci_recv_frame` | HCI and BLE host/controller split |

## Practice and assessment

**Entry diagnostic (prerequisite check):** Trace a packet from a Linux network interface to SPI transport and the ESP peer. If this is unfamiliar, use the prerequisite material above or the [roadmap diagnostics](../../../../Learning-Guide.md#diagnose-before-reading) before starting the lab.

1. **Baseline lab:** Annotate one transmit/receive path in the driver and capture a baseline host/peer exchange.
2. **Guided variation:** Test a controlled peer restart and inspect IRQ, queue, and interface recovery.
3. **Independent transfer:** repeat the method on a new input or configuration and explain where your original assumptions stop applying.

**Evidence to submit:** Source revisions, transport errors, throughput/latency, queue behavior, and recovery duration. Keep raw results and reproduction steps; distinguish measured results from estimates or work not run.

**Self-check:** Which observation would disprove your explanation of the result? Identify a competing explanation and the smallest experiment that separates them. A passing answer names a concrete predicted difference, not only another tool to try.

**Exit criteria:** reproduce the baseline, demonstrate the negative/variation check, and meet the [shared completion rubric](../../../../Learning-Guide.md#completion-rubric). Use the [lesson cycle](../../../../Learning-Guide.md#a-repeatable-lesson-cycle) for retrieval practice and spaced review.
