# 01 — H200 Hardware Architecture

## 1. Die and Process

- **GPU die:** GH100 (Hopper architecture, TSMC 4N)
- **Transistors:** 80 billion
- **SMs:** 132 Streaming Multiprocessors
- **CUDA cores:** 16,896
- **Tensor Cores:** 4th generation (FP8, FP16, BF16, TF32, INT8, INT4)
- **FP8 peak:** ~3,958 TFLOPS (dense)
- **BF16 peak:** ~1,979 TFLOPS (dense)
- **TF32 peak:** ~989 TFLOPS

The H200 uses the same GH100 die as H100 but replaces HBM3 with HBM3e stacks for higher capacity and bandwidth.

## 2. HBM3e Memory Subsystem

| Property | H100 SXM5 | H200 SXM5 |
|---|---|---|
| Memory type | HBM3 | HBM3e |
| Capacity | 80 GB | 141 GB |
| Bandwidth | 3.35 TB/s | 4.8 TB/s |
| Stacks | 5 | 6 |

### Why HBM3e Matters for AI

- Larger KV caches for long-context LLM inference (fit 128K+ tokens in memory)
- Bigger model shards → fewer pipeline stages → less pipeline bubble overhead
- 43% more bandwidth → memory-bound ops (attention, embedding lookups) run faster

### Memory Access Best Practices

```python
# Profile actual HBM bandwidth utilization
import torch
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

# Always use BF16 for training (numerically stable, uses Tensor Cores)
model = model.to(torch.bfloat16)

# Pin CPU buffers for async H2D/D2H transfers
buffer = torch.zeros(size, pin_memory=True)
```

## 3. NVLink 4.0 and NVSwitch 3rd Gen

### NVLink 4.0 Specs

- **Links per GPU:** 18 NVLink 4.0 lanes
- **Bandwidth per link:** 50 GB/s bidirectional
- **Total GPU-to-GPU bandwidth:** 900 GB/s bidirectional per GPU

### NVSwitch 3rd Gen (8-GPU Node Topology)

An 8-GPU SXM5 node uses **four NVSwitch 3.0 chips** forming a full all-to-all mesh:

```
GPU0 ──┐
GPU1 ──┤
GPU2 ──┤   NVSwitch 0   ←→   NVSwitch 1
GPU3 ──┤         ↕               ↕
GPU4 ──┤   NVSwitch 2   ←→   NVSwitch 3
GPU5 ──┤
GPU6 ──┤
GPU7 ──┘

Every GPU has direct full-bandwidth path to every other GPU.
No multi-hop penalty unlike ring or tree topologies.
```

### Why Full Mesh Matters

- All-reduce across 8 GPUs stays at **full 900 GB/s** (no bottleneck GPUs)
- Tensor parallel all-reduce for attention heads completes in ~1 µs for typical sizes
- Point-to-point P2P transfers for pipeline parallelism are lossless

### Verifying NVLink in Practice

```bash
# Check NVLink topology
nvidia-smi topo -m

# Monitor NVLink traffic per GPU
nvidia-smi dmon -s u -d 1

# NCCL topology detection
NCCL_DEBUG=INFO torchrun --nproc_per_node=8 your_script.py 2>&1 | grep "NCCL"
```

## 4. SXM5 Baseboard and Host Connectivity

- **PCIe:** PCIe 5.0 x16 per GPU (via NVLink C2C bridge to CPU on Grace-Hopper)
- **NVMe:** Direct GPUDirect Storage paths to local NVMe over PCIe 5.0
- **Thermal:** Liquid-cooled SXM5 module; GPU junction temperature target < 83°C

### CPU-GPU Memory (Grace Hopper Superchip variant)

On GH200 (Grace + H200), CPU and GPU share a **unified 900 GB/s NVLink-C2C** fabric — CPU LPDDR5x and GPU HBM3e appear in the same address space. This eliminates PCIe bottlenecks for CPU-GPU data movement.

## 5. Power and Cooling

| Property | Value |
|---|---|
| TDP per H200 | 700 W |
| 8-GPU node TDP | ~5,600 W (GPUs only) |
| Cooling method | Direct liquid cooling (DLC) |
| Inlet coolant temp | ≤ 45°C recommended |

Design your rack PDU for at least **7.5 kW per node** (accounting for CPUs, NVSwitches, networking).

## 6. Key Architectural Features for AI

### Transformer Engine

The H200 includes a **Transformer Engine** that automatically selects FP8 or BF16 precision per layer:

```python
# PyTorch + Transformer Engine (TE)
import transformer_engine.pytorch as te

# Replace standard Linear with TE Linear — auto FP8
layer = te.Linear(in_features, out_features, bias=True)

# TE handles scaling factors, amax history, and E4M3/E5M2 selection
```

### In-Network Computing via NVSwitch

NVSwitch 3.0 supports **SHARP (Scalable Hierarchical Aggregation and Reduction Protocol)** in-network reductions — all-reduce operations are partially computed inside the switch fabric, reducing GPU cycles spent on communication.

## References

- [H200 Datasheet](https://www.nvidia.com/en-us/data-center/h200/)
- [Hopper Architecture Whitepaper](https://resources.nvidia.com/en-us-tensor-core/gtc22-whitepaper-hopper)
- [NVLink 4.0 Technical Blog](https://developer.nvidia.com/blog/nvidia-hopper-architecture-in-depth/)
- [Transformer Engine Documentation](https://docs.nvidia.com/deeplearning/transformer-engine/user-guide/)
