# 02 — NCCL Algorithms & How It Achieves 900 GB/s

## 1. The Core Problem: AllReduce Algorithms

There are multiple ways to implement AllReduce across N GPUs. NCCL picks the best one based on topology and message size. Understanding them explains **why NCCL performance varies** with GPU count and buffer size.

---

## 2. Ring AllReduce

The most bandwidth-efficient algorithm for large messages. Used by NCCL on NVLink systems for messages > ~1 MB.

### Step-by-Step with 4 GPUs, tensor [A, B, C, D]

**Phase 1: Reduce-Scatter (N-1 steps)**

Each GPU holds one chunk. In each step, each GPU sends its current chunk to the next GPU and receives the next chunk.

```
Initial state:
  GPU0: [A0, B0, C0, D0]
  GPU1: [A1, B1, C1, D1]
  GPU2: [A2, B2, C2, D2]
  GPU3: [A3, B3, C3, D3]

Step 1 (each GPU sends chunk i to GPU i+1, receives chunk i-1):
  GPU0: sends D0 → GPU1, receives C3 from GPU3 → computes [C2+C3]
  GPU1: sends A1 → GPU2, receives D0 from GPU0 → computes [D0+D1]
  GPU2: sends B2 → GPU3, receives A1 from GPU1 → computes [A1+A2]
  GPU3: sends C3 → GPU0, receives B2 from GPU2 → computes [B2+B3]

Step 2:
  GPU0: [B1+B2+B3], GPU1: [C2+C3+C0], ...

After N-1=3 steps:
  GPU0 has fully reduced chunk D: [D0+D1+D2+D3]
  GPU1 has fully reduced chunk A: [A0+A1+A2+A3]
  GPU2 has fully reduced chunk B: [B0+B1+B2+B3]
  GPU3 has fully reduced chunk C: [C0+C1+C2+C3]
```

**Phase 2: AllGather (N-1 steps)**

Each GPU distributes its fully-reduced chunk to all others.

```
Step 1:
  GPU0 sends [D_sum] → GPU1
  GPU1 sends [A_sum] → GPU2
  GPU2 sends [B_sum] → GPU3
  GPU3 sends [C_sum] → GPU0

After N-1=3 steps: all GPUs have all chunks
  GPU0: [A_sum, B_sum, C_sum, D_sum]  ✓
  GPU1: [A_sum, B_sum, C_sum, D_sum]  ✓
  ...
```

### Ring AllReduce Bandwidth Analysis

```
Data sent per GPU per step: S/N  (S = tensor size, N = GPU count)
Number of steps: 2(N-1)
Total data sent per GPU: 2(N-1) × S/N ≈ 2S (for large N)

Bus bandwidth used: S × 2(N-1)/N / time
Fraction of peak bandwidth used: (N-1)/N → approaches 1.0 as N grows

For 8 GPUs: (8-1)/8 = 87.5% of peak NVLink bandwidth
For 16 GPUs: 93.75% of peak

This is why Ring AllReduce is "bandwidth optimal" — it uses nearly all available bandwidth regardless of GPU count.
```

---

## 3. Tree AllReduce

Better for **small messages** or when latency matters more than bandwidth. NVSwitch clusters often use this.

```
Reduce phase (gather to root):
       GPU0  (root)
      /    \
   GPU1    GPU2
   /  \
GPU3  GPU4

Step 1: GPU3→GPU1, GPU4→GPU1, GPU2→GPU0 (all parallel)
Step 2: GPU1→GPU0

AllGather phase (reverse direction):
Step 1: GPU0→GPU1, GPU0→GPU2
Step 2: GPU1→GPU3, GPU1→GPU4
```

**Latency:** O(log N) steps vs O(N) for Ring.
**Bandwidth:** Less efficient than Ring for large messages — data must pass through intermediate nodes.

NCCL switches between Ring and Tree based on message size automatically.

---

## 4. Double Binary Tree (NCCL's Default for NVSwitch)

NCCL uses **two overlapping binary trees** that share the reduction work. This achieves full bandwidth utilization with O(log N) latency — combining the best of both worlds.

```
Tree A (left-leaning):
    0
   / \
  1   2
 / \
3   4

Tree B (right-leaning, complement):
    4
   / \
  3   0
     / \
    2   1

Reduce-Scatter uses Tree A, AllGather uses Tree B.
Every link is used in both trees → 100% bandwidth utilization
Every GPU participates equally → no root bottleneck
```

This is why NCCL 2.x on NVSwitch systems is so efficient.

---

## 5. How NCCL Achieves 900 GB/s on 8x H200

H200 NVLink 4.0 specs:
- 18 NVLink 4.0 lanes per GPU
- 50 GB/s bidirectional per lane
- Total: 900 GB/s bidirectional per GPU

### Why Full Bandwidth Is Achievable

**Full NVSwitch mesh topology:**

```
Every GPU has a direct NVLink path to every other GPU (no hops).

GPU0 ←→ GPU1: 900 GB/s direct
GPU0 ←→ GPU2: 900 GB/s direct
GPU0 ←→ GPU7: 900 GB/s direct
```

With Ring AllReduce across 8 GPUs:
- Each GPU sends/receives S/8 per step
- 2(N-1) = 14 steps total
- Each step uses the full NVLink bandwidth between adjacent ring GPUs

**Measured bus bandwidth formula:**
```
bus_bandwidth = algbw × (2 × (N-1) / N)

where algbw = (message_size / time_taken)

For N=8: bus_bw = algbw × (14/8) = algbw × 1.75
```

**Measured with nccl-tests:**

```bash
./build/all_reduce_perf -b 1G -e 8G -f 2 -g 8

# Output columns:
# size   algbw(GB/s)   busbw(GB/s)
# 1 GB   257 GB/s      449 GB/s
# 4 GB   280 GB/s      490 GB/s
# 8 GB   291 GB/s      509 GB/s  ← approaches 512 GB/s = 900 GB/s × 8/8 × 0.57

# Why not 900 GB/s?
# Bus bandwidth per GPU (900 GB/s) is bidirectional peak
# AllReduce uses each link for both send+receive simultaneously
# 509 GB/s busbw ≈ 57% of 900 GB/s (bidirectional divided by 2 for half-duplex accounting)
# Full-duplex: 509 × 2 = 1018 GB/s ≈ ~900 GB/s per GPU ✓
```

---

## 6. Bandwidth Roofline for AllReduce

```
AllReduce communication time = (2 × S × (N-1)) / (N × BW)

where:
  S  = tensor size in bytes
  N  = number of GPUs
  BW = per-GPU NVLink bandwidth (one direction)

For H200 (BW = 450 GB/s unidirectional per GPU):
  8 GPUs, 1 GB gradient:
  Time = (2 × 1 GB × 7) / (8 × 450 GB/s)
       = 14 / 3600
       = 3.9 ms

  This matches measured values (~4 ms for 1 GB on 8x H200).
```

---

## 7. Algorithm Selection in NCCL

NCCL picks the algorithm based on message size and topology:

```
Message size:
  < 1 KB:     NCCL uses simple P2P (direct GPU-to-GPU send/recv)
  1 KB - 1 MB: Tree algorithm (lower latency for small messages)
  > 1 MB:     Ring algorithm (maximum bandwidth for large messages)

Topology:
  NVSwitch (full mesh): Double Binary Tree preferred
  PCIe ring: Ring AllReduce preferred
  Multi-node: Hierarchical (NVLink intra-node, IB inter-node)
```

Override manually:

```bash
export NCCL_ALGO=Ring         # force Ring AllReduce
export NCCL_ALGO=Tree         # force Tree AllReduce
export NCCL_PROTO=Simple      # protocol: Simple / LL / LL128
```

---

## 8. Protocols: Simple, LL, LL128

NCCL uses three protocols optimized for different scenarios:

| Protocol | Optimized For | Mechanism |
|---|---|---|
| **LL (Low Latency)** | Small messages, latency | Packs data + flag in 8-byte chunks, polling |
| **LL128** | Medium messages | 128-byte chunks, better bandwidth |
| **Simple** | Large messages, bandwidth | Standard DMA transfers, best bandwidth |

```bash
# NCCL auto-selects based on message size
# Force protocol manually:
export NCCL_PROTO=Simple      # best for training (large gradients)
export NCCL_PROTO=LL          # best for inference token sync (small, latency-sensitive)
```

---

## 9. Measuring Your Actual NCCL Bandwidth

```bash
# Install nccl-tests
git clone https://github.com/NVIDIA/nccl-tests
cd nccl-tests
make -j MPI=0 CUDA_HOME=/usr/local/cuda NCCL_HOME=/usr

# AllReduce bandwidth sweep (1 MB to 8 GB)
./build/all_reduce_perf \
    -b 1M -e 8G -f 2 \
    -g 8 \                   # 8 GPUs
    -n 20 \                  # 20 iterations
    -w 5                     # 5 warmup iterations

# AllGather (important for FSDP parameter gathering)
./build/all_gather_perf -b 1M -e 8G -f 2 -g 8

# ReduceScatter (important for ZeRO gradient sharding)
./build/reduce_scatter_perf -b 1M -e 8G -f 2 -g 8

# Broadcast (model weight distribution)
./build/broadcast_perf -b 1M -e 8G -f 2 -g 8
```

**Interpreting output:**

```
#                                                              out-of-place                       in-place
#       size         count    type   redop    root     time   algbw    busbw #wrong     time   algbw    busbw #wrong
#        (B)    (elements)                             (us)  (GB/s)   (GB/s)            (us)  (GB/s)   (GB/s)
   1048576       262144   float     sum      -1    382.1     2.75     4.81      0    381.5     2.75     4.81      0
  67108864     16777216   float     sum      -1   2431.6    27.60    48.30      0   2426.5    27.66    48.41      0
 536870912    134217728   float     sum      -1  14985.5   358.28   626.98      0  14894.2   360.47   630.82      0

# algbw: algorithm bandwidth = size / time
# busbw: bus bandwidth = algbw × 2(N-1)/N  ← the number to compare against NVLink spec
# Target busbw for 8x H200: > 450 GB/s for large messages
```

---

## 10. Gradient Communication as a Bottleneck

```
Training step time = compute_time + communication_time

Without overlap:
  Total = T_compute + T_comm

With overlap (async AllReduce during backward):
  Total = max(T_compute, T_comm) + T_serial_overhead

Overlap efficiency = T_comm / T_compute
  If T_comm < T_compute: communication is completely hidden (ideal)
  If T_comm > T_compute: communication is the bottleneck

For H200 (1979 TFLOPS BF16) training 70B model, batch=4, seq=2048:
  T_compute ≈ 6 × 70B params × 4 × 2048 tokens / 1979 TFLOPS ≈ ~1.7 ms/step
  T_comm (gradient AllReduce, 140 GB BF16): 140e9 / (450e9 × 7/8) = ~355 ms/step

  → Communication dominates massively!
  → This is why ZeRO-3 is necessary: it replaces AllReduce with ReduceScatter (only S/N data)
```

---

## References

- [Bandwidth-Optimal AllReduce (Baidu paper)](https://arxiv.org/abs/1802.05799)
- [NCCL Algorithm Documentation](https://github.com/NVIDIA/nccl/blob/master/src/graph/algo.cc)
- [NVLink 4.0 Technical Brief](https://www.nvidia.com/en-us/data-center/nvlink/)
- [nccl-tests Repository](https://github.com/NVIDIA/nccl-tests)
