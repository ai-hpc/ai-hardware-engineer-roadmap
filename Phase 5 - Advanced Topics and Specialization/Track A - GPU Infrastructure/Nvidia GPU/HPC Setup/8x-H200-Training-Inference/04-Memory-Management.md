# 04 — Memory Management on 8x H200

## 1. Memory Budget Planning

### Training Memory Breakdown (per GPU, Mixed Precision BF16)

```
Model parameters:     P bytes    (BF16 = 2 bytes/param)
Gradients:            P bytes    (BF16)
Optimizer states:     4P bytes   (FP32 copies for AdamW: params + m + v)
Activations:          A bytes    (depends on sequence length, batch size)
─────────────────────────────────────────────────────────────────────────
Total (no ZeRO):      6P + A

With ZeRO-3 (8 GPUs): (6P + A) / 8  per GPU
```

For a 70B model:
- P = 70B × 2 bytes = 140 GB
- Optimizer = 70B × 12 bytes = 840 GB
- Total without ZeRO = 980 GB → needs ZeRO-3
- With ZeRO-3 across 8 GPUs: ~123 GB per GPU ✓

### Inference Memory Breakdown

```
Weights:          W bytes  (FP16 = 2 bytes/param)
KV Cache:         K bytes  (2 × layers × heads × head_dim × seq_len × batch × 2)
CUDA workspace:   ~2 GB
Activations:      ~batch × seq × hidden × 2 bytes
─────────────────────────────────────────────────────────────────────────
Total:            W + K + overhead
```

## 2. HBM3e Memory Allocation Strategy

```python
import torch

def get_memory_stats(device=0):
    t = torch.cuda.get_device_properties(device).total_memory / 1e9
    r = torch.cuda.memory_reserved(device) / 1e9
    a = torch.cuda.memory_allocated(device) / 1e9
    print(f"Total: {t:.1f} GB | Reserved: {r:.1f} GB | Allocated: {a:.1f} GB | Free: {t-r:.1f} GB")

# Monitor before and after model load
get_memory_stats()
model = load_model()
get_memory_stats()
```

### CUDA Memory Pool Configuration

```python
# Prevent memory fragmentation with expandable segments
import torch
torch.cuda.memory.set_per_process_memory_fraction(0.95)  # leave 5% for CUDA runtime

# Configure memory allocator
torch.cuda.set_per_process_memory_fraction(0.95, device=0)

# Environment variable approach (before import torch)
# PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,max_split_size_mb:512
```

```bash
# For vLLM — controls how much GPU memory the KV cache pool uses
export VLLM_GPU_MEMORY_UTILIZATION=0.90
```

## 3. KV Cache Deep Dive

### KV Cache Size Formula

```python
def kv_cache_size_gb(
    num_layers: int,
    num_kv_heads: int,
    head_dim: int,
    max_seq_len: int,
    max_batch_size: int,
    dtype_bytes: int = 2,  # FP16 = 2, FP8 = 1
) -> float:
    # 2 for K and V
    total_bytes = 2 * num_layers * num_kv_heads * head_dim * max_seq_len * max_batch_size * dtype_bytes
    return total_bytes / 1e9

# Llama-3 70B: 80 layers, 8 GQA kv-heads, head_dim=128
kv = kv_cache_size_gb(
    num_layers=80,
    num_kv_heads=8,         # GQA reduces from 64 heads to 8 KV heads
    head_dim=128,
    max_seq_len=8192,
    max_batch_size=32,
)
print(f"KV Cache: {kv:.1f} GB")  # ~42 GB with FP16
```

### Grouped Query Attention (GQA) Memory Savings

| Attention Type | KV Heads | KV Cache Size | Used By |
|---|---|---|---|
| MHA | 64 | 100% | GPT-3, early transformers |
| GQA | 8 | 12.5% | Llama-3, Mistral |
| MQA | 1 | 1.6% | PaLM, Falcon |

GQA is the best trade-off: near-MHA quality at much lower KV memory cost.

## 4. PagedAttention (vLLM)

```
Traditional KV cache allocation:
  Max sequence length is pre-allocated, even if unused.

  Seq A (actual 200 tokens, allocated 4096): [200 used ][3896 wasted]
  Seq B (actual 3800 tokens, allocated 4096): [3800 used][296 wasted ]

  → ~30–50% GPU memory waste → fewer concurrent sequences

PagedAttention:
  Memory divided into fixed-size "blocks" (16 tokens per block by default)

  Seq A: [Block 3][Block 7][Block 12]   (only 13 blocks used)
  Seq B: [Block 0][Block 1]...[Block 237] (238 blocks used)

  Blocks are allocated on demand, freed immediately when sequence ends
  → Near-zero waste → 2-4× more concurrent sequences
```

```python
# Tune block size for your workload
llm = LLM(
    model="...",
    block_size=16,              # 16 tokens per block (default)
    # 32 tokens/block = higher throughput, less flexibility
    # 8 tokens/block = more flexibility, more metadata overhead
)
```

## 5. Activation Checkpointing Trade-offs

```
Without checkpointing:
  Forward pass saves ALL activations for backward.
  Memory = O(layers × batch × seq × hidden)

With checkpointing (every k layers):
  Forward pass saves only checkpoint activations.
  Backward recomputes k layers when needed.
  Memory = O((layers/k) × batch × seq × hidden)
  Compute overhead = ~33% extra forward passes

For H200 (large memory): checkpoint only when nearing OOM
```

```python
# Selective checkpointing — only checkpoint expensive ops
import torch.utils.checkpoint as cp

class TransformerLayer(nn.Module):
    def forward(self, x):
        # Only checkpoint the attention (expensive memory-wise)
        attn_out = cp.checkpoint(self.attention, x, use_reentrant=False)
        # Don't checkpoint MLP (cheaper to store)
        return self.mlp(attn_out)
```

## 6. CPU Offloading Strategy

For H200, CPU offloading is rarely needed for 70B inference. But for training very large models:

```python
# DeepSpeed CPU offloading for optimizer states
{
  "zero_optimization": {
    "stage": 3,
    "offload_optimizer": {
      "device": "cpu",
      "pin_memory": true,        # pinned memory for fast H2D
      "buffer_count": 4,         # prefetch buffers
      "fast_init": false
    },
    "offload_param": {
      "device": "none"           # H200: keep params on GPU
    }
  }
}
```

### Bandwidth Requirement for CPU Offload

```
CPU RAM ↔ GPU: PCIe 5.0 x16 = ~64 GB/s (H200 host)
NVLink C2C (GH200): 900 GB/s (no PCIe bottleneck)

With PCIe 5.0: for a 70B model optimizer (840 GB FP32),
  full round-trip = 840 GB / 64 GB/s = ~13 seconds per update step
  → CPU offload is a LAST RESORT for PCIe-attached systems
```

## 7. Memory Profiling Tools

```bash
# PyTorch memory snapshot (detailed allocation trace)
import torch.cuda.memory as mem

torch.cuda.memory._record_memory_history(max_entries=100000)
# ... run workload ...
mem.memory._dump_snapshot("mem_snapshot.pickle")

# Visualize at: https://pytorch.org/memory_viz
```

```bash
# Nsight Systems — system-level memory timeline
nsys profile \
    --trace=cuda,nvtx,osrt \
    --gpu-metrics-device=all \
    --output=mem_profile \
    python train.py

# Nsight Compute — per-kernel memory analysis
ncu --metrics l1tex__t_bytes,dram__bytes \
    --target-processes all \
    python train.py
```

```bash
# nvidia-smi memory monitoring during training
watch -n 1 nvidia-smi --query-gpu=index,memory.used,memory.free,utilization.gpu \
    --format=csv,noheader,nounits
```

## 8. OOM Debugging Checklist

```
1. Check reserved vs allocated:
   torch.cuda.memory_reserved()  >> torch.cuda.memory_allocated() ?
   → Memory fragmentation: set PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

2. Unexpected allocations:
   → Use torch.cuda.memory_snapshot() to find surprise allocations

3. Gradient accumulation with DDP:
   → with model.no_sync(): inside accumulation loop prevents premature all-reduce

4. KV cache growing unbounded (inference):
   → Set max_model_len and max_num_seqs limits in vLLM
   → Check for missing attention mask / infinite generation loops

5. NCCL broadcast allocations:
   → NCCL uses ~300 MB per GPU; account for this in memory budget

6. torch.compile memory increase:
   → First iteration may OOM during graph compilation; use smaller batch for warmup
```

## References

- [PyTorch Memory Management](https://pytorch.org/docs/stable/notes/cuda.html#memory-management)
- [ZeRO Memory Optimization Paper](https://arxiv.org/abs/1910.02054)
- [vLLM PagedAttention Paper](https://arxiv.org/abs/2309.06180)
- [Reducing Activation Recomputation in Large Transformer Models](https://arxiv.org/abs/2205.05198)
