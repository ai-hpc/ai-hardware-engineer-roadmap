# 07 — NCCL at Trillion-Parameter Scale

## 1. Why Trillion-Parameter Models Need Different NCCL Strategy

A 1T parameter model in BF16 weighs **2 TB**. With 8x H200 (1.1 TB total HBM3e), it doesn't even fit on a single node. Training it requires:

```
Problem: AllReduce a 2 TB gradient across 256 GPUs (32 nodes of 8x H200)
  Naive AllReduce: 2 TB × 2(255)/256 ≈ 4 TB per GPU needs to travel IB
  IB HDR bandwidth: 25 GB/s per port
  Time: 4 TB / 25 GB/s = 160 seconds per step ← completely unacceptable

Solution: Never do a 2 TB AllReduce.
  Use 3D parallelism so that each NCCL op is small and localized.
```

---

## 2. The 3D Parallelism + NCCL Architecture

### Dimension Definitions

```
Data Parallel (DP):
  Multiple identical model replicas, each on different GPUs
  Communication: AllReduce gradients (large, infrequent — once per step)
  Group size: typically 8-64 (one replica per DGX node or pod)

Tensor Parallel (TP):
  Single layer split across GPUs (column/row parallelism)
  Communication: AllReduce activations (small, very frequent — twice per layer)
  Group size: 8 (within one NVLink node — keep on NVLink!)

Pipeline Parallel (PP):
  Different layers on different GPUs/nodes
  Communication: P2P send/recv of activations (medium, sequential)
  Group size: 4-64 (across nodes — uses InfiniBand)
```

### Mapping to Hardware for 1T Model, 256 GPUs (32 nodes × 8 H200)

```
TP=8 (within each 8-GPU node)
PP=8 (8 pipeline stages, one per node)
DP=4 (4 data parallel replicas)
Total: 8 × 8 × 4 = 256 GPUs ✓

GPU assignment:
  Node  0: TP-group 0, PP-stage 0, DP-replica 0 (GPUs 0-7)
  Node  1: TP-group 1, PP-stage 1, DP-replica 0 (GPUs 8-15)
  ...
  Node  7: TP-group 7, PP-stage 7, DP-replica 0 (GPUs 56-63)
  Node  8: TP-group 0, PP-stage 0, DP-replica 1 (GPUs 64-71)
  ...
  Node 31: TP-group 7, PP-stage 7, DP-replica 3 (GPUs 248-255)
```

### NCCL Groups for 3D Parallelism

```python
import torch.distributed as dist

rank = dist.get_rank()        # global rank (0-255)
world_size = dist.get_world_size()   # 256

TP_SIZE = 8
PP_SIZE = 8
DP_SIZE = 4

# For rank r:
tp_rank = rank % TP_SIZE
pp_rank = (rank // TP_SIZE) % PP_SIZE
dp_rank = rank // (TP_SIZE * PP_SIZE)

# Create TP communicator (all ranks with same pp_rank and dp_rank)
tp_group = None
for pp in range(PP_SIZE):
    for dp in range(DP_SIZE):
        ranks = [tp + pp * TP_SIZE + dp * TP_SIZE * PP_SIZE
                 for tp in range(TP_SIZE)]
        group = dist.new_group(ranks=ranks)
        if pp_rank == pp and dp_rank == dp:
            tp_group = group   # my TP group

# Create DP communicator (all ranks with same tp_rank and pp_rank)
dp_group = None
for tp in range(TP_SIZE):
    for pp in range(PP_SIZE):
        ranks = [tp + pp * TP_SIZE + dp * TP_SIZE * PP_SIZE
                 for dp in range(DP_SIZE)]
        group = dist.new_group(ranks=ranks)
        if tp_rank == tp and pp_rank == pp:
            dp_group = group   # my DP group

# PP is handled via point-to-point (not a group communicator)
```

---

## 3. Tensor Parallelism NCCL Pattern (Frequent, Small, NVLink)

In a transformer layer, tensor parallelism creates this NCCL pattern:

```
Layer i, TP=8 (within one 8-GPU node):

Attention:
  [AllGather QKV input] → all GPUs get full hidden state
  [per-head attention computation, no NCCL]
  [ReduceScatter output] → each GPU gets 1/8 of output

  Actually in Megatron: input is replicated, output gets AllReduced
  AllReduce size: B × S × H × 2 bytes = 4 × 2048 × 8192 × 2 = 134 MB
  NVLink time: 134 MB / (450 GB/s) = 0.3 ms per AllReduce
  80 layers × 2 AllReduces = 160 × 0.3 ms = 48 ms total TP communication
```

With sequence parallelism (Megatron 3.x):

```
Input X [B, S, H] → split along sequence: each GPU gets [B, S/8, H]

Attention (column parallel):
  AllGather X from [B, S/8, H] → [B, S, H]   only when needed
  ReduceScatter output [B, S, H] → [B, S/8, H]  each GPU keeps shard

MLP (same pattern)

Benefit: activations are smaller (S/8 per GPU) → 8× less activation memory
         AllGather only materializes full sequence for GEMM
```

---

## 4. Pipeline Parallelism NCCL Pattern (Infrequent, Large, InfiniBand)

Pipeline parallel sends activations between stages. This is **point-to-point**, not collective.

```
Forward pass (4 pipeline stages, 4 micro-batches — 1F1B schedule):

Time →
Stage 0: F0  F1  F2  F3  B3  B2  B1  B0
Stage 1:     F0  F1  F2  F3  B3  B2  B1  B0
Stage 2:         F0  F1  F2  F3  B3  B2  B1  B0
Stage 3:             F0  F1  F2  F3  B3  B2  B1  B0

F = forward micro-batch, B = backward micro-batch
Send activation from Stage i to i+1 when forward completes
Send gradient from Stage i+1 to i when backward completes
```

```python
# Megatron pipeline P2P communication
from megatron.core.pipeline_parallel.schedules import recv_forward, send_forward

# Receive activation from previous stage
input_tensor = recv_forward(tensor_shape, dtype=torch.bfloat16)

# Compute layer
output = model_chunk(input_tensor)

# Send activation to next stage
send_forward(output)

# Activation tensor size:
# [micro_batch_size, seq_len, hidden_size]
# = [4, 2048, 8192] × 2 bytes = 134 MB per micro-batch
# Sent over InfiniBand: 134 MB / 25 GB/s = 5.4 ms per stage boundary
# 8 PP stages = 8 boundaries, but pipelined — not serialized
```

### Pipeline Bubble

```
Pipeline "bubble" = idle time while pipeline fills and drains

                Bubble →←    Bubble
Stage 0: F0 F1 F2 F3 [B3  B2  B1  B0]
Stage 1:    F0 F1 F2  F3  [B3  B2  B1] B0
Stage 2:       F0 F1  F2   F3  [B3  B2] B1 B0
Stage 3:          F0  F1   F2   F3  [B3] B2 B1 B0

Bubble fraction = (PP-1) / (num_micro_batches + PP - 1)
For PP=8, micro_batches=16: (8-1)/(16+7) = 7/23 = 30%
→ 30% of pipeline capacity wasted on bubbles

Minimize by: increasing micro_batch count (more gradient accumulation steps)
```

---

## 5. Data Parallel NCCL Pattern (Infrequent, Huge, InfiniBand)

With TP=8 and PP=8, each DP replica is a complete model. DP AllReduce happens once per step.

```
DP=4 replicas, each replica = 8 nodes (64 GPUs)
After all forward+backward passes within one replica:
  Each GPU holds its layer-shard's gradient (from FSDP/ZeRO-3)

With ZeRO-3 across DP dimension:
  ReduceScatter across DP groups: each node gets 1/4 of gradient shards
  Size per GPU: (full_gradient_size / world_size) × 2 bytes
  For 1T model: 2 TB / (32 nodes × 8 GPUs) = 7.8 GB per GPU

  AllReduce over IB: not 2 TB, but 7.8 GB per GPU
  IB time: 7.8 GB / 25 GB/s ≈ 312 ms per step (overlapped with backward)
```

---

## 6. Expert Parallelism — AllToAll for Mixture of Experts

MoE models (Mixtral 8×22B, GPT-MoE) use **AllToAll** — the most complex NCCL pattern.

```
Problem:
  Token X needs to be processed by Expert 3 (on GPU 5)
  Token Y needs to be processed by Expert 7 (on GPU 7)
  Each token is routed to a different GPU

AllToAll solution:
  All GPUs simultaneously exchange their routed tokens

Before AllToAll:
  GPU0 has: [tok0→E3, tok1→E1, tok2→E7, tok3→E0]
  GPU1 has: [tok4→E2, tok5→E0, tok6→E5, tok7→E3]
  ...

After AllToAll:
  GPU0 (Expert 0): receives [tok3, tok5, ...]  from wherever they came
  GPU1 (Expert 1): receives [tok1, ...]
  ...

Then each GPU runs its expert computation independently (no NCCL)

After expert computation, AllToAll again to return tokens to their original GPUs.
```

```python
# MoE AllToAll in PyTorch
def moe_forward(hidden_states, router):
    # 1. Route tokens to experts
    expert_indices, gate_scores = router(hidden_states)

    # 2. AllToAll: dispatch tokens to expert GPUs
    dispatched = torch.zeros(num_experts_per_gpu, capacity, hidden_size, device="cuda")
    dist.all_to_all_single(
        dispatched,          # output buffer (what this GPU receives)
        send_buffer,         # tokens to send to each GPU
        group=ep_group,      # expert parallel group
    )

    # 3. Compute expert on received tokens (local, no NCCL)
    expert_output = self.expert(dispatched)

    # 4. AllToAll: return processed tokens to their origin GPUs
    combined = torch.zeros_like(hidden_states)
    dist.all_to_all_single(
        combined,            # tokens returned to us
        expert_output,       # processed tokens to send back
        group=ep_group,
    )

    # 5. Combine gate scores with returned tokens
    return (combined * gate_scores.unsqueeze(-1)).sum(dim=1)
```

---

## 7. NCCL Communication Volume at Scale

### Per-Step Communication Budget for 1T Model (256 GPUs)

```
TP AllReduce (per layer, 80 layers, TP=8):
  Size: B × S × H × 2 = 4 × 2048 × 8192 × 2 = 134 MB
  Frequency: 2 per layer × 80 layers = 160
  Total per GPU: 160 × 134 MB = ~21 GB/step (NVLink, ~47 ms at 450 GB/s)

PP P2P activation send (per micro-batch, 7 boundaries, 16 micro-batches):
  Size: micro_batch × S × H × 2 = 4 × 2048 × 8192 × 2 = 134 MB
  Frequency: 7 boundaries × 16 micro-batches × 2 (fwd+bwd) = 224
  Total per GPU: 224 × 134 MB = ~30 GB/step (IB, ~1.2s at 25 GB/s, PIPELINED)

DP ReduceScatter + AllGather (once per step, ZeRO-3):
  Size: 2 TB / 256 GPUs × 2 = 15.6 GB per GPU
  Total: 15.6 GB × 2 = 31 GB/step (IB, ~1.2s at 25 GB/s)

Key insight: PP and DP both use IB and total ~62 GB. At 25 GB/s, that's ~2.5s of IB time.
But PP and DP overlap with each other and with compute.
Effective bottleneck: IB bandwidth shared between PP and DP.
```

### Optimization: Overlapping Everything

```
Ideal overlap for 1 step with 3D parallelism:

Timeline:
[TP AllReduce layer 0][TP AllReduce layer 1]...[TP AllReduce layer 79]
    (NVLink, fast, overlapped with layer compute)

[PP stage 0 forward][send activation to stage 1][PP stage 1 forward]...
                     ↑ IB, 5 ms, overlapped with compute of next stage

                    [DP AllReduce ongoing in background during backward]
                     ↑ IB, overlapped with later layers' backward

With perfect overlap: step_time ≈ max(compute, IB_PP, IB_DP)
                    ≈ compute_time (if compute dominates)
```

---

## 8. Production Monitoring at Scale

### Per-Step Timing Breakdown

```python
import time
import torch.distributed as dist

class DistributedTimer:
    def __init__(self):
        self.times = {}

    def start(self, name):
        torch.cuda.synchronize()
        self.times[name] = time.perf_counter()

    def stop(self, name):
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - self.times[name]
        # AllReduce timing across all ranks for global view
        t = torch.tensor(elapsed, device="cuda")
        dist.all_reduce(t, op=dist.ReduceOp.MAX)
        return t.item()

timer = DistributedTimer()

for step in range(total_steps):
    timer.start("forward")
    output = model(batch)
    fwd_time = timer.stop("forward")

    timer.start("backward")
    output.loss.backward()
    bwd_time = timer.stop("backward")

    timer.start("optimizer")
    optimizer.step()
    opt_time = timer.stop("optimizer")

    if dist.get_rank() == 0:
        total = fwd_time + bwd_time + opt_time
        # Estimate MFU
        flops = 6 * model_params * seq_len * batch_size
        mfu = flops / (total * peak_flops * world_size)
        print(f"Step {step}: fwd={fwd_time*1000:.0f}ms bwd={bwd_time*1000:.0f}ms "
              f"opt={opt_time*1000:.0f}ms MFU={mfu*100:.1f}%")
```

### NCCL Communication Profiling at Scale

```bash
# Profile a single training step at full scale
nsys profile \
    --trace=cuda,nvtx,nccl \
    --duration=30 \                  # capture 30 seconds
    --output=profile_rank%q{RANK} \ # separate file per rank
    torchrun ... train.py

# Analyze: which NCCL operation takes longest?
# Expected breakdown for well-tuned 1T run:
#   TP AllReduce: ~5% of step time (NVLink, fast)
#   PP P2P: ~15% (IB, partially overlapped)
#   DP AllReduce/ReduceScatter: ~10% (IB, overlapped with backward)
#   Compute: ~70%
# MFU target: ~40-50% (hard at this scale)
```

---

## 9. Real-World Examples

### GPT-4 (estimated): ~1.8T parameters, ~25,000 A100s

```
Estimated configuration:
  TP=8, PP=16, DP=~195
  MoE with 16 experts, EP=8 across DP dimension
  Data: ~13T tokens, 90-day training run
  NCCL handles: TP AllReduce, PP P2P, DP ReduceScatter/AllGather, MoE AllToAll
```

### Llama-3 405B: Meta's open-source large model

```
Training: ~16,384 H100 GPUs
  TP=8 (within node), PP=16 (across nodes), DP=128
  2 × 24,576 context length with ring attention
  Communication: standard 3D parallelism + sequence parallelism
  NCCL: ~48 ms TP per step (NVLink), ~400 ms DP per step (IB, overlapped)
  Total step time: ~800 ms, MFU: ~38%
```

---

## References

- [Megatron-LM: Efficient Training of Large Language Models](https://arxiv.org/abs/2104.04473)
- [GPipe: Efficient Training of Giant Neural Networks using Pipeline Parallelism](https://arxiv.org/abs/1811.06965)
- [Mixture of Experts at Scale (Fedus et al.)](https://arxiv.org/abs/2101.03961)
- [Llama 3 Technical Report](https://ai.meta.com/research/publications/the-llama-3-herd-of-models/)
- [Reducing Activation Recomputation with Sequence Parallelism](https://arxiv.org/abs/2205.05198)
- [ZeRO++: Memory Optimizations Towards Training Trillion Parameter Models](https://arxiv.org/abs/2306.10209)
