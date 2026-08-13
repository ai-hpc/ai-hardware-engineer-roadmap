# 04 — NCCL Configuration & Tuning

## 1. The Most Important Environment Variables

NCCL is configured entirely through environment variables. These affect performance more than almost any code change.

### Debugging and Logging

```bash
# Log level: VERSION < WARN < INFO < TRACE
export NCCL_DEBUG=INFO          # recommended for production setup
export NCCL_DEBUG=WARN          # quieter; only warnings and errors
export NCCL_DEBUG=TRACE         # very verbose; use only for debugging

# Filter debug output to specific subsystems
export NCCL_DEBUG_SUBSYS=ALL
export NCCL_DEBUG_SUBSYS=INIT   # topology detection and communicator setup
export NCCL_DEBUG_SUBSYS=GRAPH  # algorithm and ring/tree selection
export NCCL_DEBUG_SUBSYS=TUNING # algorithm performance tuning

# Write debug output to a file (one per rank)
export NCCL_DEBUG_FILE=/tmp/nccl_debug_%h_%p.txt
# %h = hostname, %p = process ID
```

**Reading NCCL_DEBUG=INFO output:**

```
NCCL INFO Bootstrap: Using [0] eth0:192.168.1.100<0>
NCCL INFO NCCL version 2.20.5+cuda12.2
NCCL INFO Channel 00/08 : 0 1 2 3 4 5 6 7
NCCL INFO Ring 00 : 0 -> 1 -> 2 -> 3 -> 4 -> 5 -> 6 -> 7 -> 0
NCCL INFO Trees [0] 3/-1/-1->0->-1|0->1/3/-1
NCCL INFO Setting affinity for GPU 0 to ffff
```

- `Channel 00/08`: 8 channels (parallel rings) for throughput
- `Ring 00`: the ring order (GPU IDs) for channel 0
- `Trees [0]`: the tree structure for the tree algorithm

---

### Algorithm and Protocol

```bash
# Force algorithm (NCCL auto-selects by default)
export NCCL_ALGO=Ring           # Ring AllReduce
export NCCL_ALGO=Tree           # Tree AllReduce
export NCCL_ALGO=CollNetDirect  # CollNet (in-network computing, requires switch support)
export NCCL_ALGO=CollNetChain   # CollNet chain variant
export NCCL_ALGO=NVLS           # NVLink SHARP (H100/H200 NVSwitch only)
export NCCL_ALGO=NVLSTree       # NVLink SHARP tree variant

# Force protocol
export NCCL_PROTO=Simple        # best for large messages (>512 KB)
export NCCL_PROTO=LL128         # medium messages
export NCCL_PROTO=LL            # latency-optimized for small messages

# Number of channels (parallel rings)
export NCCL_MIN_NCHANNELS=4     # minimum channels
export NCCL_MAX_NCHANNELS=8     # maximum channels (default: auto)
# More channels = higher bandwidth for large messages
# Too many channels = wasted SM resources for small messages
```

---

### P2P (GPU-to-GPU) Communication

```bash
# Use NVLink for P2P (strongly recommended for NVLink systems)
export NCCL_P2P_LEVEL=NVL      # NVLink only
export NCCL_P2P_LEVEL=SYS      # all P2P (NVLink + PCIe)
export NCCL_P2P_LEVEL=LOC      # same socket only
export NCCL_P2P_DISABLE=1      # disable P2P (fallback to shared memory or network)

# SHM (shared memory) transport for same-host fallback
export NCCL_SHM_DISABLE=0      # enable (default)
export NCCL_SHM_DISABLE=1      # disable (force network transport)

# Buffer size for SHM
export NCCL_BUFFSIZE=8388608   # 8 MB (default is 4 MB)
# Larger buffer = better bandwidth for large messages
# Smaller buffer = less memory used
```

---

### InfiniBand (Multi-Node)

```bash
# Select InfiniBand HCA (Host Channel Adapter)
export NCCL_IB_HCA=mlx5_0                    # specific HCA
export NCCL_IB_HCA=mlx5_0,mlx5_1            # multiple HCAs (bonding)
export NCCL_IB_HCA=^mlx5_2                  # exclude specific HCA

# Enable/disable InfiniBand
export NCCL_IB_DISABLE=0                     # enable IB (default)
export NCCL_IB_DISABLE=1                     # force Ethernet

# GPUDirect RDMA level
export NCCL_NET_GDR_LEVEL=0   # no GPUDirect RDMA
export NCCL_NET_GDR_LEVEL=1   # single-hop (same PCIe switch)
export NCCL_NET_GDR_LEVEL=2   # two-hop
export NCCL_NET_GDR_LEVEL=5   # any topology (recommended when IB is present)

# Traffic class (for QoS on IB fabric)
export NCCL_IB_TC=106         # DSCP traffic class
export NCCL_IB_SL=0           # service level (0-15)
export NCCL_IB_GID_INDEX=3    # GID index (3 = RoCEv2 for Ethernet-based RDMA)

# Timeout (seconds) for IB operations
export NCCL_IB_TIMEOUT=23     # 23 = ~8 seconds (2^23 × 4 ns)
```

---

### Timeout and Fault Tolerance

```bash
# Watchdog timeout for detecting hangs (seconds)
export NCCL_TIMEOUT=1800       # 30 minutes (default)
# Reduce for faster hang detection in production:
export NCCL_TIMEOUT=300        # 5 minutes

# Socket timeout for bootstrap
export NCCL_SOCKET_TIMEOUT=60  # 60 seconds (default)

# Async error handling
export NCCL_ASYNC_ERROR_HANDLING=1  # raise errors asynchronously
```

---

## 2. Topology-Specific Tuning Recipes

### Recipe: 8x H200 (NVSwitch, NVLink 4.0)

```bash
# Maximum performance on NVSwitch system
export NCCL_P2P_LEVEL=NVL
export NCCL_NET_GDR_LEVEL=5
export NCCL_ALGO=NVLS               # NVLink SHARP (in-switch reduction)
# or
export NCCL_ALGO=Tree               # Double binary tree (if NVLS unavailable)
export NCCL_MIN_NCHANNELS=8         # use all 8 channels
export NCCL_BUFFSIZE=16777216       # 16 MB buffer
export NCCL_DEBUG=WARN              # production verbosity
```

### Recipe: 12x L40S (PCIe, No NVLink)

```bash
# PCIe-constrained system
export NCCL_P2P_LEVEL=SYS          # use PCIe P2P
export NCCL_SHM_DISABLE=0          # use shared memory for same-host
export NCCL_ALGO=Ring              # Ring preferred for PCIe
export NCCL_MAX_NCHANNELS=4        # fewer channels (PCIe can't support many)
export NCCL_BUFFSIZE=8388608       # 8 MB
export NCCL_SOCKET_IFNAME=eth0     # correct network interface
```

### Recipe: Multi-Node with InfiniBand (HDR 200 Gb/s)

```bash
export NCCL_IB_HCA=mlx5_0,mlx5_1  # use both IB ports
export NCCL_NET_GDR_LEVEL=5        # GPUDirect RDMA
export NCCL_IB_DISABLE=0
export NCCL_IB_GID_INDEX=3         # RoCEv2
export NCCL_IB_TC=106
export NCCL_IB_SL=0
export NCCL_SOCKET_IFNAME=ib0
export NCCL_TIMEOUT=600            # longer timeout for large clusters

# Hierarchical: NVLink intra-node, IB inter-node
# NCCL detects this automatically with correct IB + NVLink configuration
```

---

## 3. Topology Detection and Verification

### Checking Detected Topology

```bash
# NCCL topology detection (set DEBUG=INFO before your job)
export NCCL_DEBUG=INFO
torchrun --nproc_per_node=8 python -c "
import torch.distributed as dist
dist.init_process_group('nccl')
import time; time.sleep(2)  # let NCCL print topology
dist.destroy_process_group()
" 2>&1 | grep -E "Ring|Tree|Channel|NVLink|IB"
```

Expected output for 8x H200:
```
NCCL INFO Channel 00/08 : 0 1 2 3 4 5 6 7     ← 8 parallel channels
NCCL INFO Ring 00 : 0 -> 1 -> 2 -> 3 -> 4 -> 5 -> 6 -> 7 -> 0
NCCL INFO Using NVLink  ← NVLink detected
NCCL INFO Trees [0] 3/-1/-1->0->-1|0->1/3/-1
```

Expected output for 8x L40S (PCIe):
```
NCCL INFO Channel 00/04 : 0 1 2 3 4 5 6 7     ← 4 channels (PCIe limited)
NCCL INFO Ring 00 : 0 -> 2 -> 4 -> 6 -> 1 -> 3 -> 5 -> 7 -> 0  ← non-sequential (PCIe-aware)
NCCL INFO Using shared memory for GPU-to-GPU  ← SHM fallback for same-host
```

### nvidia-smi Topology Check

```bash
# Full topology matrix
nvidia-smi topo -m

# Example for 8x H200:
#         GPU0 GPU1 GPU2 GPU3 GPU4 GPU5 GPU6 GPU7
# GPU0      X  NV18 NV18 NV18 NV18 NV18 NV18 NV18
# GPU1    NV18   X  NV18 NV18 NV18 NV18 NV18 NV18
# ...
# NV18 = 18 NVLink lanes = full NVLink 4.0 connectivity

# PCIe system:
#         GPU0 GPU1 GPU2 GPU3
# GPU0      X  PXB  PHB  SYS
# GPU1    PXB   X   PHB  SYS
# PXB = same PCIe switch (fast), PHB = host bridge (slower), SYS = cross NUMA (slow)
```

---

## 4. Bandwidth Tuning by Message Size

NCCL performance varies significantly with message size. Tune based on your gradient/parameter sizes.

```bash
# Find your typical gradient tensor sizes
python - <<'EOF'
from transformers import LlamaForCausalLM, LlamaConfig
import torch

config = LlamaConfig(hidden_size=8192, num_hidden_layers=80)
model = LlamaForCausalLM(config)

total_bytes = 0
for name, param in model.named_parameters():
    size_mb = param.numel() * 2 / 1e6  # BF16 = 2 bytes
    print(f"{name:60s}: {size_mb:.1f} MB")
    total_bytes += param.numel() * 2

print(f"\nTotal: {total_bytes/1e9:.1f} GB")
# This tells you the AllReduce size for DDP
# For ZeRO-3: each AllGather is parameter_size / world_size
EOF
```

### Tuning for Small Gradients (< 1 MB)

```bash
# Small gradients: prioritize latency
export NCCL_PROTO=LL          # low-latency protocol
export NCCL_ALGO=Tree         # tree has better latency (log N steps)
export NCCL_MAX_NCHANNELS=1   # single channel (less overhead)
```

### Tuning for Large Gradients (> 100 MB)

```bash
# Large gradients: maximize bandwidth
export NCCL_PROTO=Simple      # best bandwidth
export NCCL_ALGO=Ring         # bandwidth-optimal
export NCCL_MIN_NCHANNELS=8   # all channels for parallelism
export NCCL_BUFFSIZE=16777216  # 16 MB buffer
```

---

## 5. DDP Bucket Size Tuning

Bucket size controls how gradients are grouped for AllReduce. Larger buckets = fewer NCCL calls = better bandwidth utilization.

```python
# Default: 25 MB per bucket
# For large models on NVLink: increase significantly
model = DDP(
    model,
    device_ids=[rank],
    bucket_cap_mb=200,       # 200 MB per bucket — fewer, larger AllReduces
    gradient_as_bucket_view=True,  # avoid extra buffer copy
    static_graph=True,       # optimization if graph doesn't change (most training)
)

# Find optimal bucket size:
# Too small: many small NCCL calls → high overhead
# Too large: all-or-nothing sync → less overlap with backward
# Sweet spot: ~1-2 buckets covering the largest parameter groups
```

---

## 6. Profile NCCL Operations

```bash
# Profile with Nsight Systems
nsys profile \
    --trace=cuda,nvtx,nccl \        # include NCCL in trace
    --stats=true \
    --output=nccl_profile \
    torchrun --nproc_per_node=8 train.py

# View nccl_profile.nsys-rep in nsys-ui
# NCCL ops appear as colored blocks in the GPU timeline
# Look for:
#   ncclAllReduce → how long does it take?
#   GPU idle between AllReduce and next compute → communication gap
```

```python
# Mark NCCL calls with NVTX for easier profiling
import torch.cuda.nvtx as nvtx
import torch.distributed as dist

def profile_allreduce(tensor, name="grad"):
    nvtx.range_push(f"AllReduce:{name}")
    dist.all_reduce(tensor)
    torch.cuda.synchronize()
    nvtx.range_pop()
```

---

## References

- [NCCL Environment Variables Reference](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/env.html)
- [NCCL Tuning Guide](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/usage/collectives.html)
- [PyTorch DDP Performance Tuning](https://pytorch.org/tutorials/intermediate/ddp_tutorial.html)
- [NCCL Source: Algorithm Selection](https://github.com/NVIDIA/nccl/blob/master/src/graph/search.cc)
