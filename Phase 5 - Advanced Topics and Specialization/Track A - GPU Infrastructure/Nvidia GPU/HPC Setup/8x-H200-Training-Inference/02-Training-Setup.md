# 02 — Training Setup on 8x H200

## 1. What Models Fit in 8x H200?

With 8 × 141 GB = **1,128 GB total HBM3e**, the single-node capacity for training is:

| Model Size | BF16 Weights | Optimizer (AdamW) | Activations (est.) | Fits? |
|---|---|---|---|---|
| 7B | 14 GB | 56 GB | ~10 GB | Yes (single GPU) |
| 13B | 26 GB | 104 GB | ~20 GB | Yes (2 GPUs) |
| 70B | 140 GB | 560 GB | ~80 GB | Yes (8 GPUs, ZeRO-3) |
| 180B | 360 GB | 1,440 GB | ~200 GB | Needs multi-node |
| 405B | 810 GB | 3,240 GB | ~400 GB | Needs multi-node |

> Rule of thumb: AdamW stores 4× model parameters (weights, gradients, m, v) — all in FP32.

## 2. Parallelism Strategy Selection

### Decision Tree for 8x H200 Single Node

```
Model params < 7B?
  └─ DDP (pure data parallel) — simple, near-linear scaling

7B – 70B?
  └─ FSDP with ZeRO-3 — shard weights, grads, optimizer states across 8 GPUs
     OR
  └─ Tensor Parallel (TP=8) — ideal for transformer MHA layers

70B+ on single node?
  └─ TP=8 + ZeRO-3 FSDP — combine both for maximum memory efficiency
     Multi-node: add Pipeline Parallelism (PP) across nodes
```

### 3D Parallelism at Scale

```
Total GPUs = DP × TP × PP

Single 8-GPU node:
  TP=8, DP=1, PP=1  → pure tensor parallel (best for inference-like single-node)
  TP=4, DP=2, PP=1  → tensor parallel + data parallel (training with 2 micro-batches)
  TP=2, DP=4, PP=1  → more data parallel replicas, smaller model shards
```

## 3. Data Parallelism with FSDP (PyTorch)

```python
import torch
import torch.distributed as dist
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import MixedPrecision, ShardingStrategy
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy

def setup(rank, world_size):
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)

def main(rank, world_size):
    setup(rank, world_size)

    # Mixed precision: BF16 params/gradients, FP32 optimizer states
    mp_policy = MixedPrecision(
        param_dtype=torch.bfloat16,
        reduce_dtype=torch.bfloat16,
        buffer_dtype=torch.bfloat16,
    )

    from transformers import LlamaForCausalLM
    from transformers.models.llama.modeling_llama import LlamaDecoderLayer

    model = LlamaForCausalLM.from_pretrained("meta-llama/Llama-2-70b-hf")

    # FULL_SHARD = ZeRO-3: shards params, grads, optimizer states
    model = FSDP(
        model,
        sharding_strategy=ShardingStrategy.FULL_SHARD,
        mixed_precision=mp_policy,
        auto_wrap_policy=transformer_auto_wrap_policy(
            transformer_layer_cls={LlamaDecoderLayer}
        ),
        device_id=rank,
        use_orig_params=True,  # required for torch.compile
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, fused=True)

    # Training loop
    for batch in dataloader:
        with torch.autocast("cuda", dtype=torch.bfloat16):
            loss = model(**batch).loss
        loss.backward()
        # Gradient clipping with FSDP
        model.clip_grad_norm_(max_norm=1.0)
        optimizer.step()
        optimizer.zero_grad()
```

**Launch:**
```bash
torchrun --nproc_per_node=8 \
         --master_addr=localhost \
         --master_port=29500 \
         train.py
```

## 4. Tensor Parallelism with Megatron-LM

Tensor parallelism splits individual weight matrices across GPUs — ideal for transformer attention and MLP layers.

### Column and Row Parallel Linear

```
MHA weight W_QKV [hidden, 3*hidden]:
  GPU0: W_QKV[:, 0:hidden//8*3]   → handles heads 0–11
  GPU1: W_QKV[:, hidden//8*3: ...]  → handles heads 12–23
  ...
  GPU7: last partition

All-reduce after row parallel (output projection).
```

```python
# Megatron-LM style tensor parallel — column parallel
from megatron.core.tensor_parallel import ColumnParallelLinear, RowParallelLinear

# Input: [seq, batch, hidden] replicated on all GPUs
# Output: [seq, batch, hidden/tp] — each GPU has a shard
qkv_proj = ColumnParallelLinear(
    input_size=hidden,
    output_size=3 * hidden,
    bias=False,
    gather_output=False,   # keep sharded for subsequent ops
)

# After per-head computation, reduce-scatter to reconstruct full output
out_proj = RowParallelLinear(
    input_size=hidden,
    output_size=hidden,
    bias=False,
    input_is_parallel=True,
)
```

### Starting a Megatron Training Run

```bash
python pretrain_gpt.py \
    --tensor-model-parallel-size 8 \
    --pipeline-model-parallel-size 1 \
    --num-layers 80 \
    --hidden-size 8192 \
    --num-attention-heads 64 \
    --seq-length 4096 \
    --micro-batch-size 1 \
    --global-batch-size 512 \
    --bf16 \
    --use-flash-attn \
    --use-distributed-optimizer
```

## 5. DeepSpeed ZeRO-3 with Offloading

```python
# deepspeed_config.json
{
  "zero_optimization": {
    "stage": 3,
    "offload_optimizer": {"device": "cpu", "pin_memory": true},
    "offload_param": {"device": "none"},  # keep params on GPU for H200
    "overlap_comm": true,
    "contiguous_gradients": true,
    "reduce_bucket_size": 5e8,
    "stage3_prefetch_bucket_size": 5e8,
    "stage3_param_persistence_threshold": 1e6
  },
  "bf16": {"enabled": true},
  "gradient_clipping": 1.0,
  "train_micro_batch_size_per_gpu": 4,
  "gradient_accumulation_steps": 16
}
```

```bash
deepspeed --num_gpus=8 train.py --deepspeed deepspeed_config.json
```

> **H200 Tip:** With 141 GB HBM3e per GPU, CPU offloading is rarely needed for models ≤70B. Disable `offload_optimizer` for better throughput — the extra VRAM handles optimizer states in GPU memory.

## 6. Flash Attention 3 (H200 Optimized)

FlashAttention-3 is specifically optimized for Hopper (H100/H200) with:
- Asynchronous WGMMA (warpgroup matrix multiply-accumulate)
- Software pipelining overlapping GEMM with softmax
- FP8 support for 2× throughput

```python
# Using FlashAttention-3 via PyTorch SDPA (PyTorch ≥2.2)
import torch.nn.functional as F

# Automatically uses FA3 on Hopper when available
out = F.scaled_dot_product_attention(
    q, k, v,
    attn_mask=None,
    dropout_p=0.0,
    is_causal=True,
    # scale=None  # defaults to 1/sqrt(head_dim)
)

# Or explicitly via flash_attn package
from flash_attn import flash_attn_func
out = flash_attn_func(q, k, v, causal=True, softmax_scale=None)
```

## 7. Gradient Checkpointing

For very deep models, recompute activations during backward pass to trade compute for memory:

```python
from torch.utils.checkpoint import checkpoint_sequential

# Recompute every n layers
n = 4
output = checkpoint_sequential(model.layers, n, input_tensor)

# With FSDP — enable activation checkpointing per transformer layer
from torch.distributed.fsdp.wrap import enable_wrap, wrap
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
    checkpoint_wrapper, CheckpointImpl
)

model = checkpoint_wrapper(model, checkpoint_impl=CheckpointImpl.NO_REENTRANT)
```

## 8. Training Monitoring

```bash
# Watch all 8 GPUs: utilization, memory, power, temp
nvidia-smi dmon -s pucvt -d 5

# Live GPU metrics with nvitop
pip install nvitop && nvitop

# Weights & Biases integration
import wandb
wandb.init(project="llm-training-h200")
wandb.log({"train/loss": loss, "train/mfu": mfu, "gpu/mem_gb": mem_gb})
```

### MFU (Model FLOP Utilization) Calculation

```python
def compute_mfu(model_params, seq_len, batch_size, elapsed_sec, dtype="bf16"):
    # 6 * N * D for a forward+backward pass (Chinchilla)
    flops_per_token = 6 * model_params
    total_tokens = seq_len * batch_size
    achieved_tflops = (flops_per_token * total_tokens) / elapsed_sec / 1e12

    # H200 BF16 peak: 1979 TFLOPS
    peak_tflops = 1979.0
    mfu = achieved_tflops / peak_tflops
    return mfu
```

Target MFU for well-tuned training on H200: **45–55%** (BF16, large batch).

## References

- [PyTorch FSDP Tutorial](https://pytorch.org/tutorials/intermediate/FSDP_tutorial.html)
- [DeepSpeed ZeRO Documentation](https://www.deepspeed.ai/docs/config-json/)
- [Megatron-LM Repository](https://github.com/NVIDIA/Megatron-LM)
- [FlashAttention-3 Paper](https://arxiv.org/abs/2407.08608)
- [Llama 3 Training Details (Meta)](https://ai.meta.com/research/publications/the-llama-3-herd-of-models/)
