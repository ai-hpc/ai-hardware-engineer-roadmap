# 02 — Hardware Setup & Reference Configuration

## 1. Reference System: WD OpenFlex + NVIDIA GDS Validation

This configuration is from the Western Digital / NVIDIA joint technical brief. It represents a **validated, production-grade GDS deployment** and serves as the baseline for understanding real-world GDS performance.

---

## 2. Full Hardware Specification

### GPU Server

| Component | Specification | Notes |
|---|---|---|
| OS | RHEL 9: `5.14.0-70.70.1.el9_0.x86_64` | GDS requires kernel ≥ 5.4 |
| CPU | 2× Intel Xeon Gold 6348, 26C @ 2.60 GHz | Dual socket, 52 cores total |
| RAM | 512 GiB | DDR4 ECC |
| GPU | 4× NVIDIA A100 80 GB PCIe | PCIe form factor (not SXM) |
| NIC | 6× ConnectX-7 (CX-6 also validated) | 200 Gb/s each |
| NVMe | 8-Bay + 16-Bay configuration | Details below |
| BIOS/FW | FW 01.02.61 / Redfish 1.9.0 / BIOS 1.4B / CPLD F1.0C.08 | |

### Software Stack (Exact Versions)

```
OFED (Mellanox):   5.8-3.0.7.0
Nvidia Driver:     535.86.10
CUDA:              12.2.1
GDS:               2.17.3
libcufile:         1.7.1.12
```

> **Version pinning is critical.** GDS compatibility is tight — mismatching OFED, driver, and GDS versions is the most common cause of GDS failures. Always use the [GDS compatibility matrix](https://docs.nvidia.com/gpudirect-storage/release-notes/index.html) before upgrading any component.

---

## 3. PCIe Topology — The Most Important Hardware Detail

```
Full PCIe layout (reference system):

CPU Socket 0 (NUMA Node 0)          CPU Socket 1 (NUMA Node 1)
        │                                    │
   PCIe Switch 0                        PCIe Switch 1
   ┌────────────────────┐             ┌────────────────────┐
   │ A100 GPU 0  (x16)  │             │ A100 GPU 2  (x16)  │
   │ A100 GPU 1  (x16)  │             │ A100 GPU 3  (x16)  │
   │ ConnectX-7  (x16)  │             │ ConnectX-7  (x16)  │
   │ ConnectX-7  (x16)  │             │ ConnectX-7  (x16)  │
   │ ConnectX-7  (x16)  │             │ ConnectX-7  (x16)  │
   │ NVMe (4 bays, x4)  │             │ NVMe (4 bays, x4)  │
   └────────────────────┘             └────────────────────┘
        │                                    │
   Root Complex 0                      Root Complex 1
   ┌────────────────────┐             ┌────────────────────┐
   │ NVMe (2 bays, x4)  │             │ NVMe (2 bays, x4)  │
   └────────────────────┘             └────────────────────┘

Total PCIe slots: 12 × x16 Gen4
  10 PCIe Switch-connected (5 per switch)
  2 Root Complex-connected (1 per socket)

NVMe configuration:
  8-Bay: Root Complex Connected (direct, lowest latency)
  16-Bay: Switch Connected (via PCIe switch fabric)
```

### Why This Topology Was Chosen

**Co-location of GPU and NIC on the same PCIe switch** is essential for NVMe-oF GDS:

```
Data path for NVMe-oF read to GPU 0:
  OpenFlex (external) → NIC 0 (NUMA 0, Switch 0) → GPU 0 (NUMA 0, Switch 0)
  → All traffic stays on Switch 0 fabric
  → No QPI cross-socket hop
  → No Root Complex bandwidth consumption

If NIC was on Switch 1 and GPU on Switch 0:
  OpenFlex → NIC (Switch 1) → QPI → CPU → Switch 0 → GPU 0
  → QPI adds ~100 ns latency
  → QPI bandwidth shared: ~40 GB/s for 52-core Xeon Gold
  → Significant performance degradation
```

---

## 4. NIC Configuration (ConnectX-7)

```
6 × ConnectX-7, each:
  - Interface: 200 Gb/s (25 GB/s per port)
  - RDMA: RoCE v2 (RDMA over Converged Ethernet)
  - GPUDirect RDMA: enabled (GPU memory directly accessible via RDMA)

3 NICs on NUMA 0 (Switch 0): serve GPU 0 and GPU 1
3 NICs on NUMA 1 (Switch 1): serve GPU 2 and GPU 3

Per-GPU effective bandwidth (NVMe-oF):
  1.5 NICs per GPU (shared among 2 GPUs)
  1.5 × 25 GB/s = 37.5 GB/s theoretical per GPU
  Practical: ~25-30 GB/s per GPU (PCIe switch overhead)
```

### Ethernet Switch

```
NVIDIA SN3700 Spectrum-2:
  32 ports × 200 Gb/s = 6.4 Tb/s total
  Non-blocking fabric for the connected 24 DAC cables
  24 × 200 Gb/s connections (6 server NICs × 4 servers, or 6 NICs + storage)
  RoCEv2 with lossless settings:
    PFC (Priority Flow Control): enabled
    ECN (Explicit Congestion Notification): enabled
    DCBX: enabled for automatic QoS negotiation
```

---

## 5. Storage: WD OpenFlex Data24 3200 Series

```
OpenFlex Data24 3200:
  Architecture:   PCIe Gen3 internal (NVMe SSDs inside chassis)
  Frontend ports: 6 × 100 Gb/s Ethernet (via RapidFlex AIC adapters)
  Total bandwidth: 6 × 12.5 GB/s = 75 GB/s (theoretical max)
  Protocol:       NVMe-oF (NVMe over Fabrics via RDMA/RoCEv2)
  Capacity:       24 NVMe U.2 drives

RapidFlex adapters:
  Make disaggregated OpenFlex storage appear as local NVMe namespaces
  GPU sees the storage as /dev/nvme0n1, /dev/nvme1n1, etc.
  GDS code is identical whether reading local NVMe or OpenFlex NVMe-oF
```

### SSD Caching Tier Topology

```
OpenFlex provides two performance layers:

Layer 1 (Hot tier — local NVMe):
  High-speed NVMe SSDs local to the compute node
  Used for active training dataset (current epoch)
  Bandwidth: ~40-50 GB/s (8 local drives with GDS)

Layer 2 (Warm tier — OpenFlex NVMe-oF):
  Disaggregated NVMe over RoCEv2
  Used for pre-staged data, model checkpoints, next epoch
  Bandwidth: up to 75 GB/s aggregate from storage
  GPU-visible bandwidth: ~25-30 GB/s per GPU (NIC-limited)

Combined strategy:
  Prefetch next epoch's data to OpenFlex while training current epoch from local NVMe
  GPU never stalls waiting for data
```

---

## 6. Verifying Hardware Compatibility

```bash
# Step 1: Check GPU supports GDS (Volta+ architecture required)
nvidia-smi --query-gpu=name,compute_cap --format=csv
# A100: compute_cap=8.0 ✓ (Ampere)
# H100/H200: compute_cap=9.0 ✓ (Hopper)
# V100: compute_cap=7.0 ✓ (Volta, minimum)
# P100 and below: NOT supported

# Step 2: Check NUMA placement of GPU and NVMe
nvidia-smi topo -m         # GPU topology
lspci -vvv | grep -A5 "Non-Volatile"  # NVMe PCIe info
numactl --hardware         # NUMA node layout

# Step 3: Check PCIe generation for each device
lspci -vvv | grep -A2 "LnkSta"
# Should show: Gen4, Width x4 (NVMe) or Width x16 (GPU)

# Step 4: Check NIC-to-GPU NUMA alignment
ibstat | grep "Port 1" -A20 | grep "State"   # NIC state
# Check NIC and GPU are on same NUMA node:
cat /sys/class/infiniband/mlx5_0/device/numa_node  # NIC NUMA node
cat /sys/bus/pci/devices/0000:XX:XX.0/numa_node    # GPU NUMA node
# Should match

# Step 5: Verify nvidia-fs kernel module loaded
lsmod | grep nvidia_fs
# Expected: nvidia_fs 1234567 0
# Not present = GDS kernel module not installed
```

---

## 7. Hardware Checklist for New GDS Deployments

```
Storage:
[ ] NVMe drives: PCIe Gen3 or Gen4 (Gen2 may not support GDS)
[ ] File system: ext4, xfs, or btrfs (NOT ZFS — GDS requires kernel FS)
[ ] File system mounted with correct options (see 03-Software-Stack.md)

GPU:
[ ] Volta architecture or newer (V100, A100, H100, H200)
[ ] PCIe form factor (SXM also works but check baseboard support)
[ ] ECC enabled (recommended for production data integrity)

NIC (for NVMe-oF):
[ ] ConnectX-5 or newer for RoCEv2 GPUDirect RDMA
[ ] ConnectX-7 for maximum bandwidth (200 Gb/s)
[ ] Firmware: latest stable MLNX_OFED

PCIe Topology:
[ ] GPU and NVMe on same PCIe switch (NUMA co-location)
[ ] GPU and NIC on same PCIe switch (for NVMe-oF path)
[ ] PCIe switch non-blocking bandwidth > sum of connected device rates

Network Switch (NVMe-oF):
[ ] PFC (Priority Flow Control) enabled
[ ] ECN enabled for congestion management
[ ] Lossless Ethernet configuration (RoCEv2 requires zero packet loss)
```

---

## References

- [WD OpenFlex Data24 Product Page](https://www.westerndigital.com/products/data-center-platforms/openflex-data24)
- [NVIDIA ConnectX-7 Datasheet](https://www.nvidia.com/en-us/networking/ethernet-adapters/)
- [NVIDIA SN3700 Spectrum-2 Switch](https://www.nvidia.com/en-us/networking/ethernet-switching/)
- [GDS Hardware Requirements](https://docs.nvidia.com/gpudirect-storage/troubleshooting-guide/index.html#hardware-requirements)
- [NVMe-oF Specification](https://nvmexpress.org/developers/nvme-of-specification/)
