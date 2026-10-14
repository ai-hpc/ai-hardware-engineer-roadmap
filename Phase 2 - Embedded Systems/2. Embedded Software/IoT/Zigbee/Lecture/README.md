# Zigbee - Step-by-Step Lectures

Work through these in order. Each file is one session: read it, do the lab at the end, then move on.

**Hub:** [Zigbee course guide](../Guide.md)

| Lecture | Topic |
|---------|-------|
| [Lecture-01](Lecture-01.md) | What Zigbee is and where it fits |
| [Lecture-02](Lecture-02.md) | Roles, topology, and network formation |
| [Lecture-03](Lecture-03.md) | The Zigbee stack: ZDO, APS, endpoints, and clusters |
| [Lecture-04](Lecture-04.md) | Security, commissioning, sleepy devices, and OTA |
| [Lecture-05](Lecture-05.md) | ESP32-C6 practical path: devices, NCP, gateway, and Jetson context |

## Replace the analogy with a decoded transaction

The L3 analogy introduces vocabulary; finish by tracing a real or published SDK report. Annotate source device, endpoint, cluster, attribute type/value, and the receiver’s interpretation. Mark which fields belong to network addressing and which belong to the application model.

Use L4 to explain commissioning and sleepy-device timing for that transaction, then L5 to place the same contract on a standalone device or host/NCP system. Submit the annotated transaction and one rejoin/restart sequence. If using a captured example instead of hardware, identify the source and do not invent timing measurements.

**Carry forward:** Decoded transaction and architecture placement for the same device. Continue with [the next course](../../../../3.%20Embedded%20Linux/Jetson%20ESP-Hosted%20Host%20Code/Guide.md) and reuse that evidence.
