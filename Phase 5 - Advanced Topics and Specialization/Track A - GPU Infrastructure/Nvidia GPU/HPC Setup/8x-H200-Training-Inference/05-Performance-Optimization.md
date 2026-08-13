# 05 — Performance Optimization on 8x H200

## 1. Performance Targets and Roofline

### H200 Hardware Limits

| Bound | Metric | H200 Value |
|---|---|---|
| Compute (BF16) | TFLOPS | 1,979 |
| Compute (FP8) | TFLOPS | 3,958 |
| Memory bandwidth | TB/s | 4.8 |
| NVLink bandwidth | GB/s bidirectional | 900 per GPU |
| PCIe 5.0 bandwidth | GB/s | ~128 (host) |

### Roofline Analysis

```
Arithmetic Intensity (AI) = FLOPs / Bytes transferred from HBM

If AI < Ridge Point:  → MEMORY BOUND → optimize data movement
If AI > Ridge Point:  → COMPUTE BOUND → optimize FLOPs efficiency

H200 Ridge Point = 1,979 TFLOPS / 4.8 TB/s ≈ 412 FLOP/Byte (BF16)

Operation AIs:
  GEMM (large):     ~1000+ FLOP/Byte → COMPUTE BOUND ✓ (targets Tensor Cores)
  Attention (small batch): ~10 FLOP/Byte → MEMORY BOUND (use Flash Attention)
  Element-wise ops: ~1-4 FLOP/Byte → MEMORY BOUND (fuse them)
  All-reduce:       ~0 FLOPs, pure bandwidth → NVLink BOUND
```

```bash
# Run roofline analysis with Nsight Compute
ncu --set roofline --target-processes all \
    --output roofline_report \
    python -c "import torch; a=torch.randn(4096,4096,device='cuda',dtype=torch.bfloat16); torch.mm(a,a)"

# Open in Nsight Compute GUI for visualization
ncu-ui roofline_report.ncu-rep
```

## 2. CUDA Graphs for Inference

Inference has a fixed computation graph (same shapes every step). CUDA Graphs eliminate CPU kernel launch overhead.

```python
import torch

# Standard inference: CPU launches each kernel individually
# ~5-50 µs overhead per kernel → adds up for many small kernels

# CUDA Graph: capture once, replay instantly
def setup_cuda_graph(model, sample_input):
    # Warmup (fills CUDA caches, etc.)
    with torch.cuda.stream(torch.cuda.Stream()):
        for _ in range(3):
            model(sample_input)

    # Capture
    g = torch.cuda.CUDAGraph()
    static_input = sample_input.clone()
    static_output = torch.zeros_like(model(static_input))

    with torch.cuda.graph(g):
        static_output = model(static_input)

    return g, static_input, static_output

def graph_inference(g, static_input, static_output, new_input):
    static_input.copy_(new_input)
    g.replay()
    return static_output.clone()

# Speedup: 1.5-3× for small batch sizes (< 32)
# Largest impact at batch_size=1 (pure latency mode)
```

### CUDA Graphs with Dynamic Shapes (Bucketing)

```python
# For variable sequence lengths, maintain a graph per "bucket"
BUCKETS = [128, 256, 512, 1024, 2048, 4096]

graphs = {}
for seq_len in BUCKETS:
    dummy_input = torch.zeros(1, seq_len, dtype=torch.long, device="cuda")
    g, si, so = setup_cuda_graph(model, dummy_input)
    graphs[seq_len] = (g, si, so)

def bucketed_inference(input_ids):
    seq_len = input_ids.shape[1]
    bucket = next(b for b in BUCKETS if b >= seq_len)
    padded = torch.nn.functional.pad(input_ids, (0, bucket - seq_len))
    g, si, so = graphs[bucket]
    return graph_inference(g, si, so, padded)
```

## 3. Kernel Fusion

Fusing multiple element-wise operations into a single kernel eliminates redundant HBM reads/writes.

### Manual Fusion with Triton

```python
import triton
import triton.language as tl

@triton.jit
def fused_gelu_add_kernel(
    x_ptr, bias_ptr, out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    x = tl.load(x_ptr + offsets, mask=mask)
    bias = tl.load(bias_ptr + offsets, mask=mask)

    # GELU approximation
    val = x + bias
    gelu = val * 0.5 * (1.0 + tl.math.erf(val / 1.41421356))

    tl.store(out_ptr + offsets, gelu, mask=mask)

# 1 HBM read + 1 HBM write instead of 3 reads + 2 writes for separate kernels
```

### torch.compile for Automatic Fusion

```python
import torch

@torch.compile(mode="max-autotune")  # most aggressive optimization
def forward(x, w, bias):
    h = torch.mm(x, w) + bias
    return torch.nn.functional.gelu(h)

# torch.compile uses Triton to auto-fuse element-wise ops after GEMM
# mode options: "default", "reduce-overhead" (CUDA graphs), "max-autotune"
```

## 4. Communication Optimization

### Overlap Compute and Communication

```python
# Without overlap:
# [ALL-REDUCE gradient][forward pass][ALL-REDUCE gradient]...
#  ← communication →  ← compute →  ← communication →

# With overlap (DDP default):
# [forward][backward for layer N][ALL-REDUCE layer N, while computing layer N-1]
# Communication is hidden behind backward computation

# FSDP with prefetching
model = FSDP(
    model,
    sharding_strategy=ShardingStrategy.FULL_SHARD,
    forward_prefetch=True,     # prefetch next layer's params during forward
    backward_prefetch=BackwardPrefetch.BACKWARD_PRE,  # prefetch during backward
    limit_all_gathers=True,    # limit concurrent all-gathers to avoid OOM
)
```

### Gradient Compression

```python
# PowerSGD: low-rank gradient compression for bandwidth reduction
from torch.distributed.algorithms.ddp_comm_hooks import powerSGD_hook as powerSGD

state = powerSGD.PowerSGDState(
    process_group=None,
    matrix_approximation_rank=32,   # rank of approximation (lower = more compression)
    warm_start=True,
)
model.register_comm_hook(state, powerSGD.batched_powerSGD_hook)
# Reduces gradient communication by 4-10× (with small quality trade-off)
```

### NCCL Tuning for H200

```bash
# Force NVLink for all inter-GPU communication
export NCCL_P2P_LEVEL=NVL
export NCCL_NET_GDR_LEVEL=5          # use GPUDirect RDMA where available
export NCCL_ALGO=Tree                 # Tree or Ring; Tree often better on NVSwitch
export NCCL_PROTO=Simple              # Simple/LL/LL128

# Tune all-reduce chunk size
export NCCL_BUFFSIZE=8388608          # 8 MB

# Profile NCCL operations
export NCCL_DEBUG=INFO
export NCCL_DEBUG_SUBSYS=ALL
```

## 5. Mixed Precision Best Practices

```python
# Preferred: BF16 (H200 native, stable numerics)
with torch.autocast("cuda", dtype=torch.bfloat16):
    loss = model(inputs)

# For FP8 (maximum throughput on H200):
import transformer_engine.pytorch as te
from transformer_engine.common.recipe import Format, DelayedScaling

fp8_recipe = DelayedScaling(
    margin=0,
    interval=1,
    fp8_format=Format.HYBRID,
    amax_history_len=16,
    amax_compute_algo="max",
)

with te.fp8_autocast(enabled=True, fp8_recipe=fp8_recipe):
    loss = model(inputs)

# FP8 gains: ~1.8× speedup on GEMM-heavy workloads vs BF16
```

## 6. torch.compile Optimization Modes

```python
# Progressive optimization levels:

# Level 1: Default — safe fusions, no kernel search
model = torch.compile(model, mode="default")

# Level 2: Reduce overhead — adds CUDA graphs automatically
model = torch.compile(model, mode="reduce-overhead")

# Level 3: Max autotune — searches best kernel configs (slow first run)
model = torch.compile(model, mode="max-autotune")

# Backend options:
# "inductor" (default): Triton-based fused kernels
# "cudagraphs": CUDA graph capture only
# "aot_eager": AOT Autograd without optimization (debugging)

# Disable for dynamic shapes (e.g., variable-length generation)
model = torch.compile(model, dynamic=True)
```

## 7. Profiling Workflow

### Step 1: High-Level Timeline (Nsight Systems)

```bash
nsys profile \
    --trace=cuda,nvtx,python \
    --gpu-metrics-device=all \
    --stats=true \
    --output=profile_run \
    torchrun --nproc_per_node=8 train.py

# View: nsys-ui profile_run.nsys-rep
# Look for: GPU idle gaps, PCIe transfers, NCCL synchronization
```

### Step 2: Kernel-Level Analysis (Nsight Compute)

```bash
ncu \
    --set detailed \
    --kernel-name "ampere_fp16_s884gemm" \
    --launch-count 5 \
    --target-processes all \
    --output kernel_analysis \
    python inference.py

# Key metrics:
# sm__throughput.avg.pct_of_peak_sustained_elapsed → SM utilization
# dram__throughput.avg.pct_of_peak_sustained_elapsed → HBM utilization
# l1tex__throughput → L1 hit rate
```

### Step 3: PyTorch Profiler Integration

```python
from torch.profiler import profile, record_function, ProfilerActivity

with profile(
    activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
    with_stack=True,
    record_shapes=True,
    profile_memory=True,
    schedule=torch.profiler.schedule(wait=1, warmup=3, active=5),
    on_trace_ready=torch.profiler.tensorboard_trace_handler("./prof"),
) as prof:
    for step, batch in enumerate(dataloader):
        with record_function("forward"):
            loss = model(**batch).loss
        with record_function("backward"):
            loss.backward()
        prof.step()

# Visualize in TensorBoard: tensorboard --logdir=./prof
```

## 8. Common Bottlenecks and Fixes

| Symptom | Root Cause | Fix |
|---|---|---|
| GPU util < 40% (training) | Data loading bottleneck | Increase DataLoader workers, use DALI |
| GPU util < 40% (inference) | Batch size too small | Increase batch size or use continuous batching |
| All-reduce > 30% of step time | Gradient sync dominates | Reduce DP replicas, increase gradient accumulation |
| MFU < 20% | Memory-bound kernels | Enable Flash Attention, fuse ops, use larger batch |
| OOM during compilation | torch.compile traces peak | Compile with `dynamic=True`, reduce batch for compile step |
| NVLink not used | NCCL falls back to PCIe | Set `NCCL_P2P_LEVEL=NVL`, verify `nvidia-smi topo -m` |

## References

- [Nsight Systems User Guide](https://docs.nvidia.com/nsight-systems/)
- [Nsight Compute User Guide](https://docs.nvidia.com/nsight-compute/)
- [torch.compile Documentation](https://pytorch.org/tutorials/intermediate/torch_compile_tutorial.html)
- [Triton Tutorial](https://triton-lang.org/main/getting-started/tutorials/)
- [NCCL Developer Guide](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/)
