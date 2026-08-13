# 01 — GDS Architecture & Data Path

## 1. The Problem GDS Solves

Modern AI training reads enormous datasets — a single GPT-3 training run processes ~300 billion tokens, requiring continuous high-bandwidth streaming from storage to GPU. Before GDS, this path was:

```
Traditional I/O Path (CPU-mediated):

NVMe SSD
    │ PCIe Gen4 (7 GB/s per lane)
    ▼
CPU DRAM (bounce buffer)          ← CPU must allocate, manage, free
    │ Memory controller (~50 GB/s shared)
    ▼
CPU Cache (L3)
    │ PCIe Gen4
    ▼
GPU HBM

Bottlenecks:
  1. CPU DRAM bandwidth shared with all other traffic
  2. CPU must be scheduled to run memcpy (OS context switch overhead)
  3. Data copied twice: NVMe→DRAM, DRAM→GPU (double bandwidth cost)
  4. CPU L3 cache pollution (loading large files thrashes the cache)
```

```
GPUDirect Storage Path (direct DMA):

NVMe SSD (or NVMe-oF target)
    │ PCIe Gen4
    ▼
PCIe Root Complex / Switch
    │ Direct DMA (bypasses CPU)
    ▼
GPU HBM

Advantages:
  1. No CPU bounce buffer → no DRAM bandwidth consumed
  2. No CPU involvement → no context switch, no scheduling latency
  3. Data copied once → half the PCIe traffic
  4. DMA runs in parallel with GPU compute → overlap I/O and processing
```

---

## 2. Hardware Requirements for GDS

### PCIe Topology is Critical

GDS works best when GPU and NVMe share a **PCIe path without crossing NUMA boundaries**. The WD Technical Brief reference system illustrates this:

```
Reference System PCIe Layout:
┌───────────────────────────────────────────────────────────┐
│  NUMA Node 0                    NUMA Node 1               │
│                                                           │
│  PCIe Switch 0                  PCIe Switch 1             │
│  ┌──────────────────┐           ┌──────────────────┐      │
│  │ GPU 0 (A100 80GB)│           │ GPU 2 (A100 80GB)│      │
│  │ GPU 1 (A100 80GB)│           │ GPU 3 (A100 80GB)│      │
│  │ CX-7 NIC 0       │           │ CX-7 NIC 3       │      │
│  │ CX-7 NIC 1       │           │ CX-7 NIC 4       │      │
│  │ CX-7 NIC 2       │           │ CX-7 NIC 5       │      │
│  │ NVMe drives 0-7  │           │ NVMe drives 8-15 │      │
│  └──────────────────┘           └──────────────────┘      │
│         │                               │                  │
│  Intel Xeon Gold 6348           Intel Xeon Gold 6348       │
└───────────────────────────────────────────────────────────┘

Key: GPU and NVMe on the same PCIe switch = same NUMA node = optimal GDS path
     8-Bay NVMe: Root Complex Connected (2 of the bays)
     16-Bay drives: 10 on switches, 2 on root complex
```

### Why NUMA Placement Matters for GDS

```
GPU 0 (NUMA 0) reads NVMe on PCIe Switch 0 (NUMA 0):
  DMA path: NVMe → Switch 0 → GPU 0
  No QPI/UPI inter-socket hop
  Effective bandwidth: near full PCIe Gen4 x4 per NVMe (~7 GB/s)

GPU 0 (NUMA 0) reads NVMe on PCIe Switch 1 (NUMA 1):
  DMA path: NVMe → Switch 1 → QPI → Switch 0 → GPU 0
  QPI bandwidth: ~40 GB/s total, shared
  Effective bandwidth: reduced by ~30-50%
  Also: higher latency (QPI hop = ~100 ns extra)
```

---

## 3. The Three GDS Data Paths

GDS supports three distinct transport mechanisms:

### Path 1: Local NVMe (Most Common)

```
GPU HBM ←──────────────────────── DMA ──────────────────────── NVMe SSD
          PCIe Gen4 (up to 7 GB/s per x4 drive)
```

Best for: single-node training with local NVMe RAID or individual drives.

```
Reference config bandwidth:
  8-Bay NVMe, PCIe Gen4 x4 each:
  8 × 7 GB/s = 56 GB/s aggregate read bandwidth
  (practical: ~40-50 GB/s with GDS, considering overhead)
```

### Path 2: NVMe-oF over RDMA/RoCE (Network-Attached GDS)

```
GPU HBM ←── RDMA DMA ──── NIC (ConnectX-7) ──── RoCE v2 ──── NVMe-oF Target
                                                               (OpenFlex Data24)
```

The WD OpenFlex Data24 makes **remote NVMe look local** via RapidFlex adapters:
- Disaggregated storage appears as local NVMe namespace
- GPU reads from it using the same GDS API as local NVMe
- No CPU on either end of the transfer

```
Reference system:
  6 × ConnectX-7 @ 200 Gb/s (25 GB/s each)
  Total GPU-facing bandwidth: 150 GB/s (6 × 25)
  Storage side: 6 × 100 Gb/s (12.5 GB/s) = 75 GB/s
  Bottleneck: storage side at 75 GB/s
```

### Path 3: GPU Memory P2P (GPUDirect RDMA)

```
GPU 0 HBM ←── NVLink/PCIe ──── GPU 1 HBM

Or across network:
GPU 0 (Server A) ←── NIC ──── RDMA ──── NIC ──── GPU 1 (Server B)
```

This is GPUDirect RDMA — GPU memory is directly readable/writable by remote NICs. Used in NCCL for multi-node training.

---

## 4. GDS Internal Architecture

### libcufile: The GDS User-Space Library

```
Application
     │ cuFileRead() / cuFileWrite()
     ▼
libcufile (user-space daemon: cufile daemon)
     │ ioctl()
     ▼
nvidia-fs kernel module (NVFS)
     │ DMA programming
     ▼
PCIe BAR (GPU memory aperture)
     │ Direct DMA
     ▼
NVMe driver / RDMA driver
```

The **nvidia-fs** kernel module is the core — it programs DMA engines to transfer between GPU HBM physical addresses and NVMe LBA addresses, bypassing the CPU data path.

### Bounce Buffer Fallback

When GDS is unavailable (wrong kernel, wrong driver, wrong PCIe path), libcufile **silently falls back** to the traditional CPU-mediated path:

```
GDS available:  cuFileRead() → direct DMA → GPU HBM
GDS unavailable: cuFileRead() → CPU bounce buffer → GPU HBM (slower but works)
```

Always verify GDS is actually active — see [03-Software-Stack](./03-Software-Stack.md) for verification.

---

## 5. Bandwidth Budget: What GDS Can and Cannot Do

```
PCIe Gen4 x16 total: 32 GB/s per direction
PCIe Gen4 x4 per NVMe: ~7 GB/s sequential read

GPU HBM bandwidth (A100): 2 TB/s
GPU HBM bandwidth (H200): 4.8 TB/s

GDS is limited by STORAGE and PCIe, not GPU HBM:
  4 × A100 on 2 PCIe switches:
    Each switch: 2 GPUs + 3 NICs + 4 NVMe drives
    NVMe aggregate: 4 × 7 = 28 GB/s per switch
    NIC aggregate: 3 × 25 = 75 GB/s per switch
    PCIe switch bandwidth: typically 64 GB/s (non-blocking)

  Practical GDS bandwidth per GPU:
    Local NVMe: 14 GB/s (2 NVMe drives on same switch)
    NVMe-oF: up to 25 GB/s per NIC (ConnectX-7)
    Both simultaneously: limited by PCIe switch total bandwidth
```

---

## 6. GDS vs Traditional I/O: Latency and CPU Utilization

```
Operation: Read 1 GB from NVMe to GPU

Traditional:
  CPU load: ~100% on 1 core (memcpy + copy_to_user + copy_from_user)
  Time: ~500 ms (DDR4 bandwidth limited)
  CPU % freed: 0 (CPU fully busy)

GDS:
  CPU load: < 1% (only submits ioctl, DMA does the rest)
  Time: ~150 ms (PCIe Gen4 limited, 7 GB/s per drive)
  CPU freed: 99% (CPU can run training code while I/O happens)

Key insight: GDS lets the GPU do I/O AND compute simultaneously
             because neither requires the CPU during the transfer.
```

---

## References

- [NVIDIA GPUDirect Storage Overview](https://docs.nvidia.com/gpudirect-storage/overview-guide/index.html)
- [Western Digital OpenFlex + GDS Technical Brief](https://www.westerndigital.com/content/dam/doc-library/en_us/assets/public/western-digital/collateral/technical-brief/technical-brief-openflex-gpudirect-storage.pdf)
- [NVIDIA GDS Design Guide](https://docs.nvidia.com/gpudirect-storage/design-guide/index.html)
- [libcufile API Reference](https://docs.nvidia.com/gpudirect-storage/api-reference-guide/index.html)
