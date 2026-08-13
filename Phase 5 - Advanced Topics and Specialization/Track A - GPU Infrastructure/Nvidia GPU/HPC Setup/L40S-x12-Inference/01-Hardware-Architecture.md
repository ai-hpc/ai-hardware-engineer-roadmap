# 01 — L40S Hardware Architecture

## 1. Die and Process

- **GPU die:** AD102 (Ada Lovelace architecture, TSMC 4N)
- **Same die as:** RTX 4090 (consumer), but different thermal/power/firmware profile
- **Transistors:** 76.3 billion
- **SMs:** 142 Streaming Multiprocessors
- **CUDA cores:** 18,176
- **Tensor Cores:** 4th generation (FP8, FP16, BF16, TF32, INT8, INT4)
- **FP8 TFLOPS:** ~733 TFLOPS (sparse), ~366 TFLOPS (dense)
- **BF16 TFLOPS:** ~366 TFLOPS (sparse), ~183 TFLOPS (dense)
- **TF32 TFLOPS:** ~183 TFLOPS (dense)

> The L40S is distinguished from L40 (no FP8) by adding FP8 Tensor Core support, making it suitable for current-generation LLM inference.

## 2. GDDR6 Memory Subsystem

| Property | L40S | A10 (prev gen) | A100 PCIe |
|---|---|---|---|
| Memory type | GDDR6 | GDDR6 | HBM2e |
| Capacity | 48 GB | 24 GB | 80 GB |
| Bandwidth | 864 GB/s | 600 GB/s | 1,935 GB/s |
| Interface | 384-bit | 384-bit | 5120-bit |

### GDDR6 vs HBM: Key Differences

```
GDDR6 (L40S):
  + Cheaper to manufacture (standard PCB stacking)
  + Higher capacity per dollar
  + Good bandwidth for inference (memory-bound decode)
  − ~5× less bandwidth than HBM3e
  − PCIe attachment (shared CPU-GPU bandwidth)
  − No on-package NVLink possible

HBM3e (H200):
  + Extreme bandwidth (4.8 TB/s per GPU)
  + On-package with NVSwitch for GPU-GPU transfers
  − Expensive (specialized packaging)
  − Fixed capacity tiers (80 GB, 141 GB)
```

For LLM **decode** (the throughput bottleneck), memory bandwidth is the critical metric. L40S achieves 864 GB/s vs H200's 4.8 TB/s — the gap is significant for memory-bound workloads.

### Memory Capacity Planning for 12x L40S

```
Total GPU memory: 12 × 48 GB = 576 GB

Model weight allocation (FP16):
  7B   model: 14 GB  → fits on 1 GPU (34 GB free for KV cache)
  13B  model: 26 GB  → fits on 1 GPU (22 GB free for KV cache)
  34B  model: 68 GB  → needs 2 GPUs (14 GB/GPU free)
  70B  model: 140 GB → needs 3-4 GPUs (depends on KV cache needs)
  180B model: 360 GB → needs 8-10 GPUs
```

## 3. PCIe Topology (No NVLink)

The L40S uses PCIe 4.0 x16 as its only host and GPU-to-GPU interconnect. This is the most important architectural constraint to understand.

### PCIe Bandwidth vs NVLink

```
PCIe 4.0 x16:   ~32 GB/s per direction (bidirectional: 64 GB/s)
NVLink 4.0:      900 GB/s bidirectional per GPU

Ratio: NVLink is 14× faster for GPU-to-GPU communication.
```

### 12-GPU PCIe Topology (Typical Server)

```
CPU 0 (socket 0)              CPU 1 (socket 1)
   |                               |
PCIe Root Complex 0          PCIe Root Complex 1
   |          |                |           |
Switch 0   Switch 1        Switch 2    Switch 3
  / \        / \              / \          / \
GPU0 GPU1  GPU2 GPU3       GPU4 GPU5   GPU6 GPU7
                                       |     |
                                      GPU8  GPU9
                                      GPU10 GPU11
```

### Verifying PCIe Topology

```bash
# Show full topology including NUMA and PCIe relationship
nvidia-smi topo -m

# Output (simplified):
#        GPU0 GPU1 GPU2 GPU3 ... CPU Affinity
# GPU0    X    SYS  SYS  SYS ...   0-15
# GPU1   SYS   X   SYS  SYS ...   0-15
# ...
# SYS = traverses PCIe through CPU NUMA node (highest latency)
# NODE = traverses PCIe within same NUMA node (medium latency)
# PHB = traverses PCIe host bridge (low latency)
# PXB = traverses PCIe switch (lowest latency, like NVLink)

# Measure actual P2P bandwidth
python -c "
import torch
a = torch.randn(1024*1024*256, device='cuda:0', dtype=torch.float16)  # 512 MB
b = torch.empty_like(a).to('cuda:1')
import time
for _ in range(5): b.copy_(a)
torch.cuda.synchronize()
t0 = time.perf_counter()
for _ in range(100): b.copy_(a)
torch.cuda.synchronize()
bw = 512e6 * 100 / (time.perf_counter() - t0) / 1e9
print(f'P2P bandwidth GPU0→GPU1: {bw:.1f} GB/s')
# Expected: 24-30 GB/s (PCIe 4.0, direct switch)
# Poor result: < 10 GB/s (traverses NUMA boundary)
"
```

## 4. L40S PCIe Form Factor Advantages

### Rack Density

```
2U server (typical):
  4 × L40S @ 350W = 1,400W total

4U server (dense GPU):
  8 × L40S @ 350W = 2,800W total

For 12 GPUs:
  Option A: 3 × 4U servers (4 GPUs each), cross-server via InfiniBand
  Option B: 1 × 6U or 8U super-dense chassis
  Option C: 2U + 4U combination

H100/H200 SXM reference:
  DGX H100: 8 GPUs, 10.2U, 10.2kW
  L40S equivalent 8 GPUs: ~4U, ~2.8kW
```

### Flexibility

- L40S can run in any PCIe 4.0 server (no SXM baseboard needed)
- Mix with CPUs, FPGAs, or networking cards in same chassis
- Standard power connectors (PCIe 16-pin, 600W cable)
- Replaceable individually (no SXM module replacement)

## 5. Key Features for Inference

### NVENC / NVDEC (Media Engines)

L40S includes 2× NVENC + 2× NVDEC per GPU — relevant for multimodal inference pipelines processing video.

### Ada Lovelace Shader Execution Reordering (SER)

SER dynamically reorders shader workloads to improve occupancy — primarily useful for graphics. For compute/AI, standard CUDA scheduling applies.

### ADA FP8 vs Hopper FP8

```
Ada (L40S) FP8:    FP8 Tensor Cores, E4M3 and E5M2 formats
Hopper (H200) FP8: FP8 + Transformer Engine for automated scaling
                   + hardware-accelerated amax tracking

For L40S: FP8 quantization must be done offline (PTQ)
          No hardware delayed scaling support
          Use GPTQ/AWQ for post-training quantization instead
```

## 6. Power and Cooling

| Property | Value |
|---|---|
| TDP per L40S | 350 W |
| 12-GPU system TDP | ~4,200 W (GPUs) |
| Cooling | Air-cooled (passive heatsink + server fans) |
| Required airflow | Front-to-back, 200+ CFM recommended |
| PCIe power connector | 16-pin ATX 3.0 (600W capable) |

### Thermal Management

```bash
# Monitor GPU temperatures and fan speed
nvidia-smi dmon -s pucvt -d 5 -i 0,1,2,3,4,5,6,7,8,9,10,11

# Set power limit (if thermal throttling occurs)
sudo nvidia-smi -pl 300 -i 0  # reduce to 300W for GPU 0

# Check throttling reasons
nvidia-smi -q -d PERFORMANCE | grep "Reason"
# "Active: Yes" under "SW Thermal Slowdown" means throttling
```

## References

- [L40S Datasheet](https://www.nvidia.com/en-us/data-center/l40s/)
- [Ada Lovelace Architecture Whitepaper](https://images.nvidia.com/akamai/marketing/documents/Ada-GPU-Architecture-Overview.pdf)
- [PCIe 4.0 Specification](https://pcisig.com/pcie-4.0)
- [NVIDIA L40S Deployment Guide](https://docs.nvidia.com/deeplearning/performance/dl-performance-gpu-background/index.html)
