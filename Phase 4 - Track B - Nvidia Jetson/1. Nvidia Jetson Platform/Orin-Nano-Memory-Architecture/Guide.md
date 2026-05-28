# Orin Nano 8GB — Memory Architecture Deep Dive

<div class="course-identity auto-course" style="--course-accent: #be123c; --course-accent-rgb: 190, 18, 60;" markdown="1">
<div class="course-identity__icon">ON8M</div>
<div markdown="1">
<p class="course-identity__eyebrow">Deep Dive · Jetson Track</p>
<p class="course-identity__title">Specialized course identity for Orin Nano 8GB — Memory Architecture Deep Dive.</p>
<p class="course-identity__meta">Artifact: Jetson integration demo · Measure: latency, memory, power, logs</p>
</div>
</div>


> **Scope:** Production-level understanding of how memory works on Jetson Orin Nano 8GB (T234 SoC) — from DRAM initialization through SMMU translation, CMA internals, camera zero-copy pipelines, secure world isolation, and real production debugging.
>
> **Prerequisites:** You should be familiar with the [Orin Nano boot chain](../Guide.md#1-orin-nano-8gb--hardware--boot-chain-internals) and basic Linux memory concepts.

---


## 0. Jetson vs Discrete GPU — The Fundamental Difference

Before any detail: understand **why Jetson memory is fundamentally different** from a desktop/server GPU. This changes everything about how you write CUDA code for edge AI.

### Discrete GPU (Desktop/Server: RTX 4090, H100)

```
┌─────────────────────────────────┐    PCIe Gen5 x16     ┌──────────────────────────┐
│          CPU (Host)             │◄════════════════════►│       GPU (Device)        │
│                                 │     64 GB/s          │                          │
│  DDR5 System RAM                │                      │  HBM3 / GDDR6X           │
│  64–512 GB                      │                      │  24–192 GB               │
│  ~100 GB/s                      │                      │  ~1–3.35 TB/s            │
│                                 │                      │                          │
│  CPU can NOT access GPU memory  │                      │  GPU can NOT access RAM   │
│  directly                       │                      │  directly                │
└─────────────────────────────────┘                      └──────────────────────────┘

Problem: every byte must cross PCIe (64 GB/s bottleneck)
  cudaMemcpy(d_ptr, h_ptr, size, cudaMemcpyHostToDevice);  ← mandatory, slow
  cudaMemcpy(h_ptr, d_ptr, size, cudaMemcpyDeviceToHost);  ← mandatory, slow
```

### Jetson (Orin Nano Super / Orin NX / AGX Orin)

```
┌─────────────────────────────────────────────────────────────────────────┐
│                     T234 SoC (Orin Nano Super)                          │
│                                                                         │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌────────┐  │
│  │  CPU     │  │  GPU     │  │  DLA     │  │  ISP/VI  │  │ NVENC  │  │
│  │  A78AE   │  │  Ampere  │  │          │  │  Camera  │  │ NVDEC  │  │
│  │  6 cores │  │  1024    │  │ ~10 TOPS │  │          │  │        │  │
│  └────┬─────┘  └────┬─────┘  └────┬─────┘  └────┬─────┘  └───┬────┘  │
│       │             │             │             │             │        │
│       └─────────────┴─────────────┴─────────────┴─────────────┘        │
│                              │                                          │
│                    ┌─────────┴─────────┐                                │
│                    │ Memory Controller │                                │
│                    │  (MC) + SMMU      │                                │
│                    └─────────┬─────────┘                                │
└──────────────────────────────┼──────────────────────────────────────────┘
                               │
                    ┌──────────┴──────────┐
                    │  8 GB LPDDR5        │
                    │  ~102 GB/s bandwidth │
                    │  SHARED by ALL      │
                    └─────────────────────┘

No PCIe. No copy. CPU, GPU, DLA, camera ALL access the SAME physical memory.
```

### What This Means for Your Code

| Operation | Discrete GPU | Jetson |
|-----------|-------------|--------|
| **Allocate GPU memory** | `cudaMalloc` (separate VRAM) | `cudaMalloc` (same DRAM pool) |
| **Copy host→device** | `cudaMemcpy` **(mandatory, slow)** | **Often unnecessary** — use zero-copy |
| **Copy device→host** | `cudaMemcpy` **(mandatory, slow)** | **Often unnecessary** — use zero-copy |
| **Managed memory** | Page migration over PCIe (very slow) | Page migration in same DRAM (fast) |
| **Camera → GPU** | Camera→RAM→PCIe→VRAM (3 copies) | Camera→DRAM→GPU reads same DRAM (**0 copies**) |
| **Memory capacity** | CPU: 512 GB + GPU: 192 GB (separate) | **8 GB total** (shared by everything) |
| **Bandwidth** | CPU: 100 GB/s, GPU: 3,350 GB/s (separate) | **~102 GB/s shared** (everyone competes) |

### The Three Programming Implications

**1. Zero-copy is your biggest advantage.**
On discrete GPU, `cudaMemcpy` often dominates execution time. On Jetson, skip it:

```cpp
// ── Discrete GPU: mandatory copy ──────────────────────────────
float *h_data = (float*)malloc(size);          // host RAM
float *d_data;
cudaMalloc(&d_data, size);                     // GPU VRAM
cudaMemcpy(d_data, h_data, size, cudaMemcpyHostToDevice);  // PCIe copy (slow!)
kernel<<<grid, block>>>(d_data);
cudaMemcpy(h_data, d_data, size, cudaMemcpyDeviceToHost);  // PCIe copy (slow!)

// ── Jetson: zero-copy with pinned host memory ─────────────────
float *shared;
cudaMallocHost(&shared, size);                 // pinned, in same DRAM
// Both CPU and GPU access 'shared' directly — NO copy needed
fill_data_on_cpu(shared);                      // CPU writes
kernel<<<grid, block>>>(shared);               // GPU reads same memory
cudaDeviceSynchronize();
read_results_on_cpu(shared);                   // CPU reads GPU output
```

**2. Memory bandwidth is your bottleneck — not compute.**
Discrete H100: 3,350 GB/s HBM → 989 TFLOPS. Jetson Orin Nano Super: 102 GB/s LPDDR5 → 67 TOPS (GPU) + ~10 TOPS (DLA) ≈ 77 TOPS total. The compute-to-bandwidth ratio is dramatically different:

```
H100 ridge point:         989 TFLOPS / 3,350 GB/s ≈ 295 FLOP/byte
Orin Nano Super ridge:    67 TOPS / 102 GB/s ≈ 0.66 OP/byte

→ Almost EVERYTHING is memory-bound on Jetson.
→ Tiling, data reuse, and INT8 quantization are not optional — they're mandatory.
```

**3. Memory capacity is precious — 8 GB total.**
On a server, a 7B LLM model takes 14 GB (FP16) in GPU VRAM, leaving 500+ GB system RAM for everything else. On Jetson 8 GB, that same model won't even fit. You must:
- Quantize aggressively (INT8 → 7 GB, INT4 → 3.5 GB)
- Account for OS + camera + CUDA runtime (~2–3 GB)
- Choose models that fit the budget (YOLOv8-N at 3.2M params, not YOLOv8-X at 68M)

### Memory Budget Comparison

```
Discrete Server (H100 80GB + 512 GB DDR5):
  GPU VRAM: 80 GB  │ System RAM: 512 GB
  ─────────────────│────────────────────
  Model: 40 GB     │ OS: 4 GB
  Activations: 20 GB│ App: 2 GB
  Scratch: 15 GB   │ Datasets: 200 GB
  Free: 5 GB       │ Free: 306 GB

Jetson Orin Nano 8GB (shared):
  ┌──────────────────────────────────┐
  │ 8 GB LPDDR5 (total)             │
  │                                  │
  │ Firmware carveouts:   ~0.4 GB   │
  │ OS + kernel:          ~0.5 GB   │
  │ CMA (camera buffers): ~0.75 GB  │
  │ CUDA runtime:         ~0.3 GB   │
  │ Model (INT8 YOLO):    ~0.1 GB   │
  │ Inference scratch:     ~0.2 GB   │
  │ ─────────────────────────────── │
  │ Free for userspace:    ~5.75 GB │
  └──────────────────────────────────┘
```

### CUDA Memory API Decision Tree for Jetson

```
What are you allocating?
│
├── Camera frames / video decoder output
│   └── Use DMA-BUF / NvBufSurface (zero-copy from hardware engine)
│
├── Model weights (loaded once, read by GPU)
│   └── cudaMalloc (pinned device memory, fastest GPU access)
│
├── Pre/post-processing buffers (CPU writes, GPU reads, or vice versa)
│   └── cudaMallocHost (pinned, both CPU and GPU access, no copy)
│
├── Prototyping / irregular access pattern
│   └── cudaMallocManaged (automatic migration, but unpredictable latency)
│
└── Temporary GPU-only scratch
    └── cudaMalloc (standard, fastest)
```

> **Key mindset shift:** On discrete GPU, you think "minimize PCIe transfers." On Jetson, you think "minimize total DRAM bandwidth consumption" — because CPU, GPU, camera, and display all share the same ~102 GB/s pipe.

---

## 1. T234 SoC Memory Architecture Overview

The Tegra234 SoC in Orin Nano 8GB uses a **unified memory architecture** — CPU, GPU, DLA, and all accelerators share a single 8GB LPDDR5 pool. There is no discrete GPU memory.

```
                        8GB LPDDR5
                    ┌───────────────┐
                    │               │
       ┌────────────┤  Memory       ├────────────┐
       │            │  Controllers  │            │
       │            │  (MC)         │            │
       │            └───────┬───────┘            │
       │                    │                    │
  ┌────┴────┐         ┌────┴────┐         ┌─────┴────┐
  │ CPU     │         │ GPU     │         │ DLA/ISP  │
  │ A78AE   │         │ Ampere  │         │ NVENC    │
  │ cluster │         │ 1024    │         │ NVDEC    │
  │         │         │ cores   │         │ CSI/VI   │
  └─────────┘         └─────────┘         └──────────┘
```

### Memory Controllers

T234 has multiple memory controller (MC) channels connected to the LPDDR5 interface. The MC handles:

* Arbitration between CPU, GPU, and all accelerators
* Bandwidth partitioning (configurable via BPMP firmware)
* Quality-of-Service (QoS) priorities for latency-sensitive engines (e.g., display, camera)

### Unified Virtual Memory (UVM)

CUDA on Jetson supports unified virtual addressing (UVA). A pointer returned by `cudaMallocManaged()` is valid on both CPU and GPU — the runtime handles coherence via page faults and migration. However, for latency-critical paths (camera, inference), explicit allocation with `cudaMalloc()` or NVMM buffers avoids migration overhead.

---

## 2. Cache Hierarchy

Understanding caches is critical for optimizing memory-bound inference workloads.

```
CPU Core (A78AE)
  L1I: 64KB per core (instruction)
  L1D: 64KB per core (data)
  L2:  256KB per core
  L3:  2MB shared (cluster-level)

GPU (Ampere)
  L1 / Shared Memory: per-SM (configurable)
  L2: shared across all SMs
```

### Key Implications

* **CPU L3 is small (2MB)** — large working sets spill to DRAM quickly. This matters for pre/post-processing on CPU.
* **GPU L2** is shared — inference kernels compete for cache. Kernel tiling and occupancy directly affect L2 hit rates.
* **No hardware cache coherence between CPU and GPU** — this is why explicit synchronization (e.g., `cudaStreamSynchronize`) is required, and why zero-copy via NVMM/DMA-BUF is preferred over CPU-GPU memcpy.

---

## 3. Boot-Time Memory Initialization

On Orin Nano, memory is set up in stages before Linux ever runs.

### MB1 — DRAM Training

MB1 (loaded from QSPI NOR) performs LPDDR5 training:

* Calibrates timing parameters for each DRAM channel
* Stores training data for fast-boot on subsequent boots
* Configures ECC if enabled

If DRAM training fails, the board does not boot — no display, no serial output.

### MB2 — Carveout Setup

MB2 reads the device tree and reserves memory regions (carveouts) for firmware processors:

* **BPMP** — power management firmware
* **SPE** — safety processor
* **RCE** — camera real-time engine
* **OP-TEE** — secure world
* **VPR** — Video Protected Region (DRM)

These carveouts are removed from the Linux-visible memory map before the kernel starts.

### UEFI — Memory Map Handoff

UEFI constructs the EFI memory map describing which regions are:

* Usable by Linux (conventional memory)
* Reserved (firmware, carveouts)
* ACPI/runtime services

Linux receives this map via `efi_memmap` and initializes its memory subsystem accordingly.

### Device Tree Memory Nodes

The DTB loaded by UEFI defines:

```dts
memory@80000000 {
    device_type = "memory";
    reg = <0x0 0x80000000 0x0 0x70000000>,   /* Region 1 */
          <0x0 0xf0200000 0x0 0x0fe00000>;   /* Region 2 */
};
```

Carveouts are defined separately:

```dts
reserved-memory {
    #address-cells = <2>;
    #size-cells = <2>;
    ranges;

    bpmp_carveout: bpmp {
        compatible = "nvidia,bpmp-shmem";
        reg = <0x0 0x40000000 0x0 0x200000>;
        no-map;
    };
};
```

The `no-map` property means Linux cannot touch this memory.

If DTB carveouts are wrong, the system may crash before `start_kernel()` or firmware processors will malfunction.

---

## 4. What the 8GB Really Looks Like

On Orin Nano, DRAM is initialized by MB1 → MB2 → UEFI → Linux. Very early boot reserves memory **before Linux even starts**.

### Typical `/proc/iomem` Layout (Conceptual)

```
00000000-000fffff : Reserved (Boot ROM, vectors)
00100000-3fffffff : System RAM
40000000-4fffffff : Reserved (VPR, carveouts)
50000000-57ffffff : CMA
58000000-ffffffff : System RAM
```

### Reserved Regions

| Region                       | Purpose                         |
|------------------------------|---------------------------------|
| VPR (Video Protected Region) | DRM / secure video playback     |
| SPE carveout                 | Safety processor firmware       |
| BPMP carveout                | Power management firmware       |
| OP-TEE secure RAM            | Trusted execution environment   |
| CMA                          | Contiguous DMA for camera / GPU |

These are defined in the **Device Tree (DTB)** loaded by UEFI.

### Inspecting the Real Memory Map

```bash
# Full memory map
cat /proc/iomem

# Linux-visible memory summary
cat /proc/meminfo

# CMA-specific stats
grep Cma /proc/meminfo
```

Example `/proc/meminfo` output on Orin Nano 8GB:

```
MemTotal:        7633536 kB    ← Not 8GB! Carveouts took the rest
CmaTotal:         786432 kB    ← 768MB reserved for CMA
CmaFree:          524288 kB    ← Currently unused CMA
```

The ~400MB difference between 8GB and `MemTotal` is consumed by firmware carveouts, kernel code, and reserved regions.

---

## 5. Linux Memory Zones

Linux divides physical memory into zones:

| Zone        | Address Range    | Purpose                           |
|-------------|------------------|-----------------------------------|
| ZONE_DMA    | 0 – 16MB         | Legacy ISA DMA (mostly unused)    |
| ZONE_DMA32  | 0 – 4GB          | 32-bit addressable DMA            |
| ZONE_NORMAL | Above 4GB        | General-purpose allocations        |

Orin Nano mostly uses **ZONE_NORMAL** since LPDDR5 is mapped above the 4GB boundary for most of the address space.

You can inspect zone state:

```bash
cat /proc/zoneinfo
```

Key fields:

* `free` — pages currently free in this zone
* `min` / `low` / `high` — watermarks that trigger reclaim
* `nr_free_pages` — total free across all orders

When `free` drops below `min`, the kernel starts aggressive reclaim (kswapd, direct reclaim). On Jetson with large camera buffers, this can cause latency spikes.

---

## 6. Buddy Allocator Internals

The Linux buddy allocator manages free pages in powers of two:

```
Order 0 =   4KB  (1 page)
Order 1 =   8KB  (2 pages)
Order 2 =  16KB  (4 pages)
Order 3 =  32KB  (8 pages)
...
Order 9 =   2MB  (512 pages)
Order 10 =  4MB  (1024 pages)
```

### How Allocation Works

When a driver requests 64KB (order 4):

1. Check the order-4 free list
2. If empty, split a higher-order block (e.g., order 5 → two order 4 blocks)
3. Return one block, keep the other as free

### How Freeing Works

When a block is freed:

1. Check if the adjacent "buddy" block is also free
2. If yes, merge into a higher-order block
3. Repeat up the chain

### Inspecting Fragmentation

```bash
cat /proc/buddyinfo
```

Example output:

```
Node 0, zone   Normal  1024  512  256  128  64  32  16  8  4  2  1
```

Each number is the count of free blocks at that order (0 through 10). If high-order columns show `0`, the system is fragmented — large contiguous allocations will fail even if total free memory is sufficient.

### Why This Matters on Jetson

Camera and GPU drivers need large contiguous allocations. A fragmented system with plenty of free memory but no high-order blocks will fail to allocate buffers — this is why CMA exists.

---

<a id="7-cma--contiguous-memory-allocator"></a>
## 7. CMA — Contiguous Memory Allocator

CMA is **not** separate memory. It is:

> A reserved movable region inside normal RAM where Linux can guarantee physically contiguous allocations.

Linux marks CMA pages as **MIGRATE_CMA**. When not used for DMA, these pages hold movable data (e.g., userspace pages). When a contiguous allocation is needed, the kernel migrates those pages elsewhere.

### Allocation Flow

When a driver calls `dma_alloc_contiguous()`:

1. Find free pages inside the CMA region
2. Use the buddy allocator to locate a contiguous block
3. Compact memory if needed (migrate movable pages out of the way)
4. Map the allocation into the SMMU for the requesting device
5. Return the DMA address (IOVA) and kernel virtual address

### Why Long-Running Systems Fail

Over time, memory fragmentation builds up:

```
CMA region:
[used][free][used][free][used][free][used]
```

Even if total free CMA is sufficient, no single contiguous chunk is large enough. The camera fails with:

```
Failed to allocate buffer
```

This is a classic embedded production issue. Mitigations:

* **Right-size CMA** for your workload (see next section)
* **Pre-allocate buffers at boot** and reuse them
* **Monitor CMA fragmentation** in production (see Section 16)
* **Use NVMM buffer pools** rather than allocating/freeing per-frame

---

## 8. How to Resize CMA

### Method 1: Kernel Command Line

Edit `/boot/extlinux/extlinux.conf`:

```
APPEND ... cma=1024M
```

This sets CMA to 1GB.

### Method 2: Device Tree

```dts
reserved-memory {
    linux,cma {
        compatible = "shared-dma-pool";
        reusable;
        size = <0x0 0x40000000>;   /* 1GB */
        linux,cma-default;
    };
};
```

Reflash DTB after modification.

### Sizing Guidelines

| Workload                      | Recommended CMA |
|-------------------------------|-----------------|
| Single camera, light inference | 256–512MB       |
| Single 4K camera + TensorRT   | 512–768MB       |
| Multi-camera AI pipeline       | 768MB–1GB       |
| Multi-camera + large models    | 1–1.5GB         |

Trade-offs:

* Too large CMA = less memory for userspace, model weights, and CUDA
* Too small CMA = camera buffer allocation failures under load
* 768MB–1GB is common for production multi-camera AI workloads

### Verify After Change

```bash
grep Cma /proc/meminfo
```

Confirm `CmaTotal` matches your configured size.

---

## 9. SMMU (IOMMU) — Real Translation Path

Orin Nano uses **ARM SMMU v2** (System Memory Management Unit). Every device that performs DMA goes through the SMMU.

### Translation Flow

```
Device (GPU, CSI, ISP, etc.)
   ↓
IOVA (I/O Virtual Address — what the device sees)
   ↓
SMMU page tables (owned by Linux kernel)
   ↓
Physical DRAM (actual memory location)
```

Devices **never see physical RAM directly**. The kernel:

1. Allocates memory (from CMA or normal pages)
2. Maps those physical pages into the SMMU page tables for the target device
3. Gives the device an IOVA
4. The device performs DMA using the IOVA

### Why This Architecture Is Powerful

* **Device isolation** — a misbehaving device cannot corrupt another device's memory
* **No rogue DMA** — unmapped accesses cause SMMU faults (caught and logged)
* **Shared buffers** — multiple devices can map the same physical pages at different IOVAs
* **Scatter-gather as contiguous** — physically scattered pages appear contiguous to the device

This is why GPU and camera can share buffers safely without copies.

### SMMU Page Table Structure

ARM SMMU uses multi-level page tables similar to CPU MMU:

```
Stream Table Entry (per device/stream ID)
   ↓
Context Descriptor
   ↓
Level 1 Page Table (covers large VA range)
   ↓
Level 2 Page Table
   ↓
Level 3 Page Table (4KB granule)
   ↓
Physical Page
```

Each device has its own stream ID and its own set of page tables, providing full isolation.

### SMMU Fault Debugging

When a device accesses an unmapped IOVA:

```bash
dmesg | grep smmu
```

Example fault:

```
arm-smmu 12000000.iommu: Unhandled context fault: iova=0x1234000, fsynr=0x11
```

This means the device tried to access IOVA `0x1234000` but no mapping existed. Common causes:

* Buffer freed while device still referencing it
* Incorrect DMA-BUF import
* Driver bug in IOVA mapping

---

## 10. Camera → ISP → CUDA Zero-Copy Path

This is the critical data path for real-time AI vision on Orin Nano.

### Step 1 — Sensor Capture

```
MIPI CSI-2 camera sensor
    ↓ (serial lanes)
NVCSI controller (deserializes)
    ↓
VI (Video Input — captures frames)
    ↓
ISP (Image Signal Processor — debayer, denoise, tone-map)
```

### Step 2 — ISP Output to CMA

ISP writes processed frames into **CMA memory**. CMA is required because ISP needs physically contiguous buffers for DMA. The ISP hardware has no scatter-gather capability.

### Step 3 — DMA-BUF Export

The V4L2 driver exports the buffer as a **DMA-BUF** file descriptor. Once exported, the same physical memory can be imported by any DMA-BUF-aware consumer:

* CUDA (via `cudaExternalMemory`)
* NvBufSurface (NVIDIA multimedia buffer API)
* GStreamer (via `nvv4l2camerasrc`)
* DeepStream (via source bin)

### Step 4 — GPU Access via SMMU

The GPU imports the DMA-BUF and maps it through its own SMMU context:

```
GPU IOVA → SMMU → Same physical CMA pages that ISP wrote to
```

**No memcpy.** This is true zero-copy. The GPU reads the exact same physical memory the ISP wrote, just mapped at a different virtual address through a different SMMU stream.

### Full Zero-Copy Pipeline

```
Sensor → NVCSI → VI → ISP → CMA buffer
                                ↓ (DMA-BUF export)
                          ┌─────┴─────┐
                          │           │
                     GPU (CUDA)   Display
                     via SMMU     via SMMU
                          │
                     TensorRT
                     inference
```

This is how real-time 4K vision runs efficiently on Orin Nano without saturating memory bandwidth.

### Memory Flow Summary

| Stage       | Memory Type        | Allocated By        |
|-------------|--------------------|---------------------|
| CSI/VI      | CMA (contiguous)   | V4L2 / nvargus      |
| ISP output  | CMA (same buffer)  | Reused from VI      |
| GPU input   | DMA-BUF import     | CUDA / NvBufSurface |
| GPU output  | CUDA device memory | cudaMalloc          |

---

## 11. GPU Memory Management

### NVMM (NVIDIA Multimedia Memory)

NVMM is NVIDIA's buffer management layer for multimedia pipelines (camera, encode, decode). Key properties:

* Buffers are allocated from CMA or carved-out memory
* Accessible by all NVIDIA engines (ISP, GPU, VIC, NVENC, NVDEC)
* Zero-copy between engines via DMA-BUF
* Managed by `libnvbufsurface`

NVMM is the foundation for GStreamer and DeepStream zero-copy pipelines.

### CUDA Memory on Jetson

On Jetson (unified memory architecture), CUDA memory types behave differently than on discrete GPUs:

| API                    | Where Memory Lives  | CPU Accessible? | GPU Accessible? | Notes                          |
|------------------------|---------------------|-----------------|-----------------|--------------------------------|
| `cudaMalloc`           | DRAM (pinned)       | No              | Yes             | Fastest for GPU-only access    |
| `cudaMallocManaged`    | DRAM (migrating)    | Yes             | Yes             | Page-fault based, higher latency |
| `cudaMallocHost`       | DRAM (pinned)       | Yes             | Yes             | Good for CPU↔GPU shared data   |
| DMA-BUF import         | CMA / NVMM          | Yes             | Yes             | Zero-copy from camera/decoder  |

For inference pipelines, prefer:

* **DMA-BUF import** for camera frames (zero-copy)
* **`cudaMalloc`** for model weights and scratch memory
* **`cudaMallocHost`** for CPU pre/post-processing buffers shared with GPU

Avoid `cudaMallocManaged` in latency-critical paths — page faults add unpredictable latency.

### GPU Memory Pressure

Monitor GPU memory usage:

```bash
# Total GPU memory (same as system memory on Jetson)
nvidia-smi  # or tegrastats

# Detailed CUDA memory
python3 -c "import torch; print(torch.cuda.memory_summary())"
```

On Orin Nano 8GB, GPU and CPU compete for the same 8GB. A large model loaded in CUDA reduces memory available for camera buffers, CMA, and OS. Memory planning is essential.

---

## 12. DLA Memory Path

> **Deep dive:** For full DLA coverage — hardware architecture (MAC/SDP/PDP/CDP), TensorRT integration, supported layers, multi-engine scheduling, profiling, and production deployment patterns — see [**Orin Nano DLA Deep Dive**](../Orin-Nano-DLA-Deep-Dive/Guide.md).

Orin Nano 8GB includes **1 DLA (Deep Learning Accelerator)** capable of up to 10 TOPS (INT8).

### How DLA Accesses Memory

DLA uses the same unified DRAM but has its own SMMU stream:

```
DLA engine
   ↓
DLA IOVA
   ↓
SMMU (DLA stream context)
   ↓
Physical DRAM
```

### TensorRT DLA Execution

When TensorRT runs layers on DLA:

1. TensorRT allocates input/output buffers in DRAM
2. Maps them into DLA's SMMU context
3. Programs DLA registers (layer config, weights pointer, I/O pointers)
4. DLA executes the layer, reading weights and input from DRAM, writing output to DRAM
5. If next layer runs on GPU, the same output buffer is mapped into GPU's SMMU — zero-copy handoff

### DLA vs GPU Memory Trade-offs

| Aspect          | GPU                        | DLA                         |
|-----------------|----------------------------|-----------------------------|
| Memory bandwidth | High (shared with CPU)    | Lower (limited ports)       |
| Precision       | FP32, FP16, INT8           | FP16, INT8 only             |
| Layer support   | All TensorRT layers        | Subset (conv, pool, etc.)   |
| Power           | Higher                     | Much lower per TOPS         |

Running layers on DLA frees GPU for other tasks and reduces power consumption — critical for edge/battery systems.

---

## 13. TensorRT Engine Memory Management

TensorRT is the primary inference engine on Jetson. Understanding how it allocates and uses memory is essential for fitting models into the 8 GB budget.

### How TensorRT Uses Memory

```
TensorRT engine lifecycle:

1. Build phase (on workstation or Jetson):
   Model (ONNX/UFF) → TensorRT optimizer → serialized engine (.engine file)

2. Load phase (on Jetson at startup):
   Read .engine from disk → deserialize → allocate:
     ├── Weight memory:     model weights (pinned, read-only after load)
     ├── Activation memory: intermediate layer outputs (reused across layers)
     ├── Workspace memory:  scratch space for conv/GEMM algorithms
     └── I/O buffers:       input and output tensors

3. Inference phase (per frame):
   Write input → execute() → read output
   No new allocations — everything is preallocated
```

### Memory Breakdown for a Typical Model

```
YOLOv8-S (INT8, 640×640 input, batch=1):

  Component              Memory      Notes
  ──────────────────────────────────────────────────
  Weights (INT8)          ~5 MB      Quantized from ~22 MB FP32
  Activation buffers     ~12 MB      Largest intermediate tensor
  Workspace              ~30 MB      Convolution algorithm scratch
  I/O buffers             ~2 MB      Input image + output detections
  ──────────────────────────────────────────────────
  Total                  ~49 MB      Fits easily

ResNet-50 (FP16, 224×224, batch=1):

  Weights (FP16)         ~48 MB
  Activation buffers     ~25 MB
  Workspace              ~50 MB
  I/O buffers             ~1 MB
  ──────────────────────────────────────────────────
  Total                 ~124 MB

Llama 3.2 3B (INT4-AWQ, via TensorRT-LLM):

  Weights (INT4)       ~1500 MB      3B × 0.5 bytes
  KV cache (INT8)       ~110 MB      26 layers × 8 heads × 128 dim × 2048 ctx
  Activation buffers    ~200 MB      Attention + FFN intermediates
  Workspace             ~100 MB
  ──────────────────────────────────────────────────
  Total               ~1910 MB      Fits, but consumes ~2 GB of 5.5 GB available
```

### Workspace Memory — The Hidden Consumer

TensorRT tries multiple convolution algorithms during build and picks the fastest. Faster algorithms often need more workspace. You control the trade-off:

```cpp
config->setMemoryPoolLimit(MemoryPoolType::kWORKSPACE, 64 << 20);  // 64 MB max
// Larger workspace = TensorRT can try faster algorithms
// Smaller workspace = less memory used, possibly slower algorithms
```

**Jetson recommendation:** Set workspace to 64–128 MB. Going higher rarely helps on Orin Nano's smaller SMs.

### Activation Memory Reuse

TensorRT reuses activation buffers across layers. Layer 5's output buffer can be reused for Layer 10's output if Layer 5's data is no longer needed.

```
Layer 1 → [Buffer A] → Layer 2 → [Buffer B] → Layer 3 → [Buffer A] (reused!)
                                                           ↑
                                              Layer 1's data no longer needed
```

This is why TensorRT uses far less memory than naive PyTorch inference (which keeps all intermediate tensors alive until backward pass).

### Inspecting TensorRT Memory Usage

```python
import tensorrt as trt

# After building engine
engine = runtime.deserialize_cuda_engine(engine_data)

# Check memory
print(f"Device memory: {engine.device_memory_size / 1024**2:.1f} MB")
print(f"Layers: {engine.num_layers}")
print(f"Max batch: {engine.max_batch_size}")

# Per-layer memory (advanced)
inspector = engine.create_engine_inspector()
print(inspector.get_engine_information(trt.LayerInformationFormat.JSON))
```

---

## 14. Multi-Model and Multi-Engine Inference

Production Jetson systems often run multiple AI models simultaneously: object detection + classification + tracking, or detection + segmentation + LLM.

### Memory Planning for Multi-Model Pipelines

```
Example: Autonomous robot vision pipeline

  Camera → YOLOv8-N (detect) → ResNet-18 (classify) → DeepSORT (track)
                  ↓                     ↓                    ↓
              ~30 MB               ~25 MB                ~15 MB
              GPU + DLA            GPU                   CPU

Pipeline memory budget:
  Model weights:      30 + 25 + 15         = ~70 MB
  Activation buffers: 12 + 8 + 5           = ~25 MB
  Workspace:          30 + 20 + 0          = ~50 MB
  Camera buffers:     4 frames × 3 MB      = ~12 MB
  ───────────────────────────────────────────────────
  Total AI pipeline:                        ~157 MB

  + OS/kernel/CMA/CUDA overhead:           ~2500 MB
  ───────────────────────────────────────────────────
  Remaining from 8 GB:                     ~5343 MB  ← plenty of room
```

### GPU + DLA Split — Maximum Throughput

Running detection on DLA while classification runs on GPU achieves overlap — both engines execute in parallel on the same DRAM:

```
Timeline (overlapped):

  Frame N:   │ DLA: YOLO detect ──────│
             │ GPU: (idle)            │ GPU: ResNet classify ──│
             │                         │                        │
  Frame N+1: │                    DLA: YOLO detect ──────│     │
             │                         │ GPU: ResNet classify ──│

DLA and GPU read different parts of DRAM simultaneously.
Memory controller arbitrates bandwidth between them.
```

**Key constraint:** DLA and GPU compete for the same ~102 GB/s DRAM bandwidth. If both are bandwidth-saturated, total throughput drops. Monitor with `tegrastats`:

```bash
tegrastats --interval 500
# Look for: GR3D_FREQ (GPU utilization), EMC_FREQ (memory clock)
# If EMC is at 100%, you're bandwidth-limited — reduce model size or batch
```

### Multi-Engine TensorRT Contexts

```cpp
// Load two engines
auto det_engine = runtime->deserializeCudaEngine(det_data, det_size);
auto cls_engine = runtime->deserializeCudaEngine(cls_data, cls_size);

// Create execution contexts (share GPU, separate state)
auto det_ctx = det_engine->createExecutionContext();
auto cls_ctx = cls_engine->createExecutionContext();

// Run on separate CUDA streams for overlap
cudaStream_t det_stream, cls_stream;
cudaStreamCreate(&det_stream);
cudaStreamCreate(&cls_stream);

det_ctx->enqueueV2(det_bindings, det_stream, nullptr);
cls_ctx->enqueueV2(cls_bindings, cls_stream, nullptr);

// Both execute concurrently if resources allow
cudaStreamSynchronize(det_stream);
cudaStreamSynchronize(cls_stream);
```

---

## 15. LLM Memory Patterns on Unified Architecture

LLMs have unique memory patterns that interact with Jetson's unified architecture differently than vision models.

### Autoregressive Decode — The Memory Access Pattern

```
Prefill phase (process entire prompt):
  All tokens processed in parallel
  Memory pattern: large GEMM (batch = prompt_length × hidden_dim)
  Bandwidth usage: HIGH (loading full weight matrices)
  Compute utilization: GOOD (large batch amortizes weight loading)

Decode phase (generate one token at a time):
  One token generated per step
  Memory pattern: skinny GEMV (batch=1 × hidden_dim)
  Bandwidth usage: HIGH (still load full weight matrices for 1 token!)
  Compute utilization: TERRIBLE (~1% — almost all time spent loading weights)

This is why LLM decode is severely memory-bandwidth-bound on Jetson.
```

### Weight Loading Dominates on Jetson

```
Llama 3.2 3B INT4 decode — one token:

  Weight loading: 1.5 GB read from DRAM
  Computation:    3B × 2 FLOPs = 6 GFLOP
  Time to load:   1.5 GB / 102 GB/s = ~14.7 ms
  Time to compute: 6 GFLOP / 67 TOPS = ~0.09 ms
  ─────────────────────────────────────────────
  Decode time:    ~15 ms per token → ~67 tokens/sec

  Compute utilization: 0.09 / 14.7 = 0.6%

  → 99.5% of time is waiting for DRAM to deliver weights
  → Faster compute (more CUDA cores) would not help
  → Only bandwidth helps: smaller weights (INT4 > INT8 > FP16)
```

**This is fundamentally different from server GPUs:**

```
H100 with same model:
  Time to load:   1.5 GB / 3350 GB/s = ~0.45 ms
  → ~2,200 tokens/sec (bandwidth difference)

Jetson Orin Nano Super is ~33× slower for LLM decode purely due to bandwidth.
This is not fixable by software — it's physics.
```

### KV Cache in Unified Memory — The Advantage

On discrete GPUs, KV cache lives in GPU VRAM. If you need CPU post-processing of attention patterns (for debugging, interpretability), you must copy back over PCIe.

On Jetson, KV cache is in the same DRAM — CPU can inspect it directly:

```cpp
// Allocate KV cache with cudaMallocHost (zero-copy on Jetson)
float* kv_cache;
cudaMallocHost(&kv_cache, kv_cache_size);

// GPU writes during attention
attention_kernel<<<grid, block>>>(q, k, v, kv_cache, ...);
cudaDeviceSynchronize();

// CPU can read KV cache directly — no copy!
for (int l = 0; l < num_layers; l++) {
    printf("Layer %d, head 0, token 0: K=%.3f\n",
           l, kv_cache[l * kv_stride]);
}
```

### LLM Memory Growth During Conversation

```
KV cache grows with each generated token:

Token 1:    Model (1.5 GB) + KV (0.05 MB) = 1500 MB
Token 100:  Model (1.5 GB) + KV (5.3 MB)  = 1505 MB
Token 1000: Model (1.5 GB) + KV (53 MB)   = 1553 MB
Token 2048: Model (1.5 GB) + KV (109 MB)  = 1609 MB
Token 4096: Model (1.5 GB) + KV (218 MB)  = 1718 MB
Token 8192: Model (1.5 GB) + KV (436 MB)  = 1936 MB  ← danger zone on 8 GB

Monitor continuously:
  tegrastats --interval 1000 | grep -oP 'RAM \d+/\d+MB'
```

Set a hard context limit in production to prevent OOM:

```bash
# llama.cpp: cap context to prevent OOM
./llama-cli -m model.gguf -c 2048 --memory-f32 0  # INT8 KV cache
```

---

## 16. Inference Memory Optimization Strategies

### 16.1 Quantization Impact on Memory

Quantization is the single most effective memory optimization — every bit saved is bandwidth saved:

```
Same model, different precisions — memory and tokens/sec:

  Precision   Size     DRAM reads/token   Est. tokens/sec
  ─────────────────────────────────────────────────────────
  FP32        12 GB    12 GB/token        won't fit
  FP16         6 GB    6 GB/token         ~8 tok/s
  INT8         3 GB    3 GB/token         ~17 tok/s
  INT4         1.5 GB  1.5 GB/token       ~33 tok/s
  INT3         1.1 GB  1.1 GB/token       ~45 tok/s (quality degrades)

  Tokens/sec scales almost linearly with quantization level
  because decode is 99%+ memory-bandwidth-bound.
```

### 16.2 Weight Layout for Bandwidth

How weights are stored in memory affects bandwidth utilization:

```
Row-major (default):
  Weight matrix [4096 × 4096] stored as 4096 rows of 4096 elements
  For batch=1 GEMV: read entire matrix row by row
  Memory access: sequential → good for DRAM burst reads ✓

Blocked layout (TensorRT-LLM):
  Matrix split into tiles that fit in SM shared memory
  Each tile loaded once, used for multiple output elements
  Better reuse → fewer total DRAM reads ✓

Interleaved quantized layout:
  INT4 weights packed 2 per byte, interleaved for Tensor Core alignment
  Dequantize in registers during compute → no extra bandwidth ✓
```

TensorRT and llama.cpp handle layout optimization automatically. If writing custom kernels, layout matters enormously.

### 16.3 Shared Memory Tiling for AI Kernels

The same tiling principle from CUDA Section 3 applies to all AI kernels on Jetson:

```
Attention kernel without tiling:
  For each output element:
    Load Q row from DRAM (4096 × 2 bytes = 8 KB)
    Load K column from DRAM (context × 2 bytes)
    Compute dot product
    Load V column from DRAM
    Accumulate
  Total DRAM: O(seq² × d) — quadratic in sequence length

Attention kernel with tiling (FlashAttention):
  Load Q tile (64 × 128) into shared memory
  For each K/V tile:
    Load K tile into shared memory
    Load V tile into shared memory
    Compute partial attention in shared memory
    Accumulate in registers
  Write final output to DRAM
  Total DRAM: O(seq × d) — linear! (tiles reused within shared memory)
```

On Orin Nano with 48 KB shared memory per SM, typical tile sizes:
- Attention: Q tile 32×64, K/V tile 32×64
- GEMM: 64×64 tile (INT4 weights + FP16 activations)
- Convolution: 16×16 output tile

### 16.4 DMA-BUF for AI Vision Pipelines

The zero-copy path from camera to AI model is the most efficient pipeline on Jetson:

```
Optimal AI vision pipeline (zero-copy throughout):

  Camera → ISP → [DMA-BUF] → GPU preprocess → [same buffer] → TensorRT
                    ↑                                              ↓
              CMA allocation                              Detection output
              (done once at startup)                      in pinned memory
                                                               ↓
                                                          CPU post-process
                                                          (direct access, no copy)

  Total copies: ZERO
  Total DRAM bandwidth: only what compute needs (no wasted copy traffic)
```

Compare with naive pipeline:
```
  Camera → ISP → memcpy → CPU buffer → cudaMemcpy → GPU buffer → TensorRT → cudaMemcpy → CPU
  Total copies: 3 (each wastes ~3–12 MB × 30 FPS of bandwidth)
  Wasted bandwidth: ~360 MB/s for 1080p — 0.35% of total 102 GB/s
  For 4K: ~1.4 GB/s wasted — 2.7% of total bandwidth

  On bandwidth-starved Jetson, every percent counts.
```

### 16.5 Multi-Model Memory Sharing Patterns

When running multiple AI models, share buffers where possible:

```cpp
// Pre-allocate one buffer for the largest input
size_t max_input_size = std::max({yolo_input_size, resnet_input_size, seg_input_size});
void* shared_input;
cudaMallocHost(&shared_input, max_input_size);  // pinned, zero-copy

// Reuse for all models (they run sequentially)
preprocess_for_yolo(camera_frame, shared_input);
yolo_ctx->enqueueV2({shared_input, yolo_output}, stream, nullptr);

preprocess_for_resnet(crop, shared_input);  // reuse same buffer
resnet_ctx->enqueueV2({shared_input, resnet_output}, stream, nullptr);
```

**Memory saved:** instead of 3 separate input buffers (3 × 3 MB = 9 MB), use 1 × 3 MB = 3 MB. This adds up quickly with multiple models and multiple cameras.

---

## 17. OP-TEE and Secure Memory

Orin Nano supports ARM TrustZone with two execution worlds:

```
Normal World (Linux, CUDA, all userspace)
─────────────────────────────────────────
Secure World (OP-TEE Trusted OS)
```

### Secure Memory Carveout

A region of DRAM is carved out exclusively for the secure world:

* **Not mapped in Linux** — `cat /proc/iomem` will not show it
* **Protected by TrustZone** — hardware prevents normal-world access
* **Not accessible by SMMU** — even DMA from devices cannot reach it

Used for:

* DRM key storage and content decryption
* Secure boot chain validation at runtime
* Cryptographic services (hardware-accelerated)
* Secure storage (encrypted key material)

### Memory Impact

If the OP-TEE carveout is configured too large, Linux usable RAM shrinks. On a memory-constrained 8GB system, every MB counts. Default carveouts are typically 16–64MB.

### Interacting with OP-TEE

Linux communicates with OP-TEE via the TEE subsystem:

```bash
# Check if OP-TEE is running
ls /dev/tee*

# Typical devices
/dev/tee0        # TEE device
/dev/teepriv0    # Privileged TEE device
```

Applications use the OP-TEE client library (`libteec`) to call Trusted Applications (TAs) in the secure world.

---

## 18. Multi-Camera Memory Planning

Production Jetson systems often run 2–6 cameras simultaneously. Memory planning is critical.

### Per-Camera Memory Budget

For a single 1080p camera at 30 FPS:

| Component                  | Memory Per Frame | Frames Buffered | Total      |
|----------------------------|-----------------|-----------------|------------|
| CSI/VI capture buffer      | ~6MB (RAW10)    | 4 (ring)        | ~24MB      |
| ISP output (NV12)          | ~3MB            | 4 (ring)        | ~12MB      |
| CUDA inference input       | ~3MB            | 2               | ~6MB       |
| **Total per camera**       |                 |                 | **~42MB**  |

For 4K (3840x2160):

| Component                  | Memory Per Frame | Frames Buffered | Total      |
|----------------------------|-----------------|-----------------|------------|
| CSI/VI capture buffer      | ~24MB (RAW10)   | 4 (ring)        | ~96MB      |
| ISP output (NV12)          | ~12MB           | 4 (ring)        | ~48MB      |
| CUDA inference input       | ~12MB           | 2               | ~24MB      |
| **Total per camera**       |                 |                 | **~168MB** |

### System Memory Budget (4-camera 1080p Example)

| Component                    | Memory    |
|------------------------------|-----------|
| OS + kernel + services       | ~500MB    |
| Camera buffers (4 cameras)   | ~168MB    |
| CMA reserved                 | ~768MB    |
| TensorRT model (YOLOv8-S)    | ~100MB    |
| CUDA runtime + scratch       | ~200MB    |
| Firmware carveouts            | ~400MB    |
| **Remaining for userspace**  | **~5.5GB** |

With larger models or 4K cameras, this budget tightens significantly. Plan and measure before deployment.

### Buffer Pool Strategy

For production systems:

1. **Pre-allocate all buffers at startup** — avoid runtime allocation/free cycles
2. **Use fixed-size buffer pools** — NVMM pools or V4L2 REQBUFS with fixed count
3. **Pin buffers** — prevent kernel from swapping or migrating them
4. **Monitor fragmentation** — log CMA and buddy state periodically

---

## 19. Performance Monitoring and Profiling

### tegrastats

NVIDIA's real-time system monitor:

```bash
tegrastats --interval 1000
```

Output includes:

* RAM usage (used/total)
* CPU utilization per core
* GPU utilization percentage
* GPU frequency
* Temperature (CPU, GPU, board)
* Power consumption per rail

### /proc/meminfo

Key fields for Jetson memory analysis:

```bash
cat /proc/meminfo
```

| Field        | What It Tells You                              |
|--------------|------------------------------------------------|
| MemTotal     | Total Linux-visible RAM (after carveouts)      |
| MemAvailable | Estimated available memory for new allocations |
| CmaTotal     | Total CMA region size                          |
| CmaFree      | Free CMA memory                                |
| Slab         | Kernel slab allocator usage                    |
| Mapped       | Memory-mapped file pages                       |

### /proc/buddyinfo

Shows fragmentation per zone and order:

```bash
cat /proc/buddyinfo
```

Healthy system: numbers across all orders.
Fragmented system: high counts at order 0–2, zeros at order 6+.

### Nsight Systems

For profiling CUDA + camera + inference together:

```bash
nsys profile --trace=cuda,nvtx,osrt ./my_inference_app
```

Shows timeline of:

* CUDA kernel launches
* Memory allocations and transfers
* CPU/GPU synchronization points
* OS runtime events

### SMMU and DMA Debugging

```bash
# SMMU faults
dmesg | grep smmu

# IOMMU groups (which devices share SMMU context)
ls /sys/kernel/iommu_groups/*/devices/

# DMA-BUF usage
cat /sys/kernel/debug/dma_buf/bufinfo
```

---

## 20. Production Debug Checklist

When camera or inference randomly fails after hours of operation:

### Step 1 — Check Overall Memory

```bash
cat /proc/meminfo | grep -E "MemTotal|MemAvailable|CmaTotal|CmaFree"
```

If `MemAvailable` is very low, the system is under memory pressure. If `CmaFree` is low, camera buffers may fail to allocate.

### Step 2 — Check Fragmentation

```bash
cat /proc/buddyinfo
```

If high-order columns (order 6+) show `0`, memory is fragmented. Even with free memory, large contiguous allocations will fail.

### Step 3 — Check CMA Usage

```bash
grep Cma /proc/meminfo
```

Compare `CmaFree` to your expected per-frame allocation size. If `CmaFree` is less than the largest single allocation needed, allocation will fail.

### Step 4 — Check for SMMU Faults

```bash
dmesg | grep smmu
```

SMMU fault example:

```
arm-smmu 12000000.iommu: Unhandled context fault: iova=0x1234000, fsynr=0x11
```

This means a device accessed an unmapped IOVA — typically a use-after-free bug or incorrect DMA-BUF handling.

### Step 5 — Check for OOM Events

```bash
dmesg | grep -i "out of memory\|oom"
```

The OOM killer may have terminated a process. Check which process was killed and why.

### Step 6 — Check Thermal Throttling

```bash
cat /sys/devices/virtual/thermal/thermal_zone*/temp
cat /sys/devices/virtual/thermal/thermal_zone*/type
```

If temperature exceeds limits, the system throttles clocks. This can cause inference to miss deadlines, buffers to queue up, and memory to grow.

---

## 21. Common Production Issues and Solutions

### CMA Exhaustion

**Symptom:** `Failed to allocate buffer` after hours of operation.

**Cause:** CMA fragmentation from repeated alloc/free cycles.

**Solutions:**
* Pre-allocate buffer pools at startup; never free them
* Increase CMA size (see Section 8)
* Use NVMM buffer pools with fixed count
* Periodically trigger compaction: `echo 1 > /proc/sys/vm/compact_memory`

### SMMU Faults Under Load

**Symptom:** `Unhandled context fault` in dmesg, frames dropped.

**Cause:** Buffer freed while device still performing DMA, or race condition in DMA-BUF handling.

**Solutions:**
* Ensure proper synchronization before freeing DMA-BUF
* Check refcounting on shared buffers
* Use V4L2 QBUF/DQBUF properly (don't access buffer while queued)

### OOM Killer on Jetson

**Symptom:** Application killed unexpectedly; `dmesg` shows OOM.

**Cause:** Combined GPU + CPU + camera memory exceeds available RAM.

**Solutions:**
* Reduce model size (quantize to INT8, prune)
* Reduce camera buffer count or resolution
* Offload layers to DLA (frees GPU memory for other uses)
* Monitor memory budget and set process memory limits

### Thermal Throttling Causing Latency Spikes

**Symptom:** Inference FPS drops periodically; `tegrastats` shows clock reduction.

**Cause:** SoC temperature exceeds thermal limits.

**Solutions:**
* Add active cooling (fan, heatsink)
* Use `nvpmodel` to set appropriate power mode
* Reduce workload or duty cycle
* Check ambient temperature and enclosure ventilation

### Memory Leak in Long-Running Inference

**Symptom:** `MemAvailable` steadily decreases over hours/days.

**Cause:** CUDA memory not freed, DMA-BUF not closed, Python object accumulation.

**Solutions:**
* Use `torch.cuda.empty_cache()` periodically if using PyTorch
* Ensure every `cudaMalloc` has a matching `cudaFree`
* Close DMA-BUF file descriptors after use
* Profile with `cuda-memcheck` or Nsight Compute

---

## 22. References

* [NVIDIA Jetson Linux Developer Guide](https://docs.nvidia.com/jetson/archives/r36.4/DeveloperGuide/) — official documentation covering boot, memory, drivers
* [Jetson Orin Nano Datasheet](https://developer.nvidia.com/embedded/jetson-orin-nano) — hardware specifications
* [ARM SMMU Architecture Specification](https://developer.arm.com/documentation/ihi0070/) — SMMU v3 reference (SMMU v2 subset)
* [Linux CMA Documentation](https://www.kernel.org/doc/html/latest/mm/cma.html) — kernel CMA subsystem
* [Linux Buddy Allocator](https://www.kernel.org/doc/html/latest/mm/page_allocator.html) — page allocator internals
* [OP-TEE Documentation](https://optee.readthedocs.io/) — Trusted Execution Environment
* [GStreamer on Jetson](https://docs.nvidia.com/jetson/archives/r36.4/DeveloperGuide/SD/Multimedia/AcceleratedGstreamer.html) — hardware-accelerated multimedia pipelines
* [TensorRT DLA Documentation](https://docs.nvidia.com/deeplearning/tensorrt/developer-guide/) — DLA integration with TensorRT
* Main guide: [Nvidia Jetson Platform Guide](../Guide.md)
