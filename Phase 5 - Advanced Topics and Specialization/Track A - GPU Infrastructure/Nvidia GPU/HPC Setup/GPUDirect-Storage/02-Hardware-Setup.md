# 02 — Inspect hardware and topology without guessing

Build a record of the machine you actually have. A GPU generation, filesystem
name, or count of NIC ports is insufficient to establish a supported GDS path.

**Prerequisites:** [data paths](01-Architecture-and-Data-Path.md) and Linux basics.
**Artifact:** a labeled topology diagram with a support checklist grounded in the
selected release's documentation. Allow 1–2 hours on an available Linux system.
These hardware commands are documented examples, not locally executed results.

## Start with identities

Record GPU model/form factor, storage controller/device, NIC, CPU sockets, PCIe
switches, kernel, driver, CUDA/cuFile versions, and filesystem mount. Hardware
specifications describe capabilities; the software matrix determines which
combinations are supported.

Use the current [GDS overview/support matrix](https://docs.nvidia.com/gpudirect-storage/overview-guide/index.html)
and [release notes](https://docs.nvidia.com/gpudirect-storage/release-notes/index.html).
For example, older Volta-era guidance should not be carried forward after support
was removed in GDS v1.15. Do not infer support from compute capability alone.

## Read the topology

These are read-only discovery examples. Package availability varies by Linux
installation; inspect your tools' help before using additional flags:

```bash
nvidia-smi --query-gpu=name,pci.bus_id,driver_version --format=csv
nvidia-smi topo -m
lspci -t
numactl --hardware
lsblk -o NAME,TYPE,SIZE,FSTYPE,MOUNTPOINTS
```

Select a real device's PCI address from `lspci`, then examine its negotiated link
and NUMA information. Do not paste placeholder addresses into a configuration:

```bash
lspci -s 0000:65:00.0 -vv
cat /sys/bus/pci/devices/0000:65:00.0/numa_node
```

In the example above, `0000:65:00.0` is illustrative. Link *capability* and current
negotiated link *status* are different. A device capable of a wide link can run
with fewer lanes. NUMA node `-1` can mean the location is not known; it is not a
new physical socket.

## Worked topology exercise

Suppose GPUs A and B share a switch upstream link with 20 GB/s useful capacity.
Two storage readers each request 14 GB/s through that link. The demand is 28 GB/s,
so both cannot simultaneously receive 14 GB/s. Equal sharing would suggest 10
GB/s each, but scheduling and other traffic can make their measured shares differ.

Now move one reader's storage to the other socket. The new path may add a shared
inter-socket dependency or change peer-transfer support. You cannot calculate its
penalty from the word “NUMA.” Draw the revised path and measure it under the same
access pattern.

**Guided variation:** one reader runs alone. Compare its bandwidth with the two-
reader run. A large drop only under concurrency supports a shared-resource
hypothesis; it does not identify which resource without counters or another
controlled test.

## Treat storage as data, not just a device name

A namespace can contain valuable data or be used by another workload. This course
uses an existing dedicated test directory; it does not require formatting a disk.
Record the directory's actual mount with `findmnt -T`, then verify support for
that filesystem and mount configuration.

Remote storage adds transport, permissions, server state, congestion, and failover
considerations. Connecting a namespace does not imply that a normal local
filesystem may be mounted read-write from several hosts. Concurrent access needs
an appropriate coordination/filesystem design.

## Choose a tier based on the workload

| Tier | Possible use | Question to measure |
|---|---|---|
| Local storage | Repeated working set or local checkpoint | Does reuse avoid the shared network bottleneck? |
| Shared storage | Dataset distribution and checkpoint exchange | How do aggregate load and access pattern affect each reader? |
| GPU/host memory cache | Reuse across batches or epochs | Is its capacity and invalidation policy compatible with correctness? |

Prefetching can reduce stalls when data is predictable and resources overlap.
It does not guarantee the GPU never waits. Record the bytes prefetched but unused
as well as latency and useful throughput.

## Finish with a support-and-evidence table

For every link in the proposed path, attach its identity, supported-mode evidence,
negotiated status, and any measurement. Leave unknown values marked unknown rather
than filling them with a vendor peak or a different system's result.

**Self-check:** a local NVMe test works. Does that establish remote NVMe-oF support?
No: the remote transport, NIC, mount, and selected mode create a different path.
Repeat capability, integrity, and performance checks for it.

Continue to [software setup](03-Software-Stack.md), then the
[API contract](04-libcufile-API.md).
