# 04 — Kernel Fusion

## 1. The Problem: HBM Round-Trips Kill Performance

Every separate kernel must read inputs from HBM (High Bandwidth Memory) and write outputs back. When kernels are chained, this creates redundant memory traffic:

```
Unfused: LayerNorm → GELU → Dropout → Residual Add

  HBM → LayerNorm kernel → HBM
                           HBM → GELU kernel → HBM
                                               HBM → Dropout kernel → HBM
                                                                      HBM → Add kernel → HBM

HBM reads/writes:  8 total (4 reads + 4 writes)
Data size:         8 × seq_len × hidden × 2 bytes (BF16)

For seq=2048, hidden=8192, BF16:
  8 × 2048 × 8192 × 2 = 256 MB of HBM traffic

Fused:
  HBM → FusedLayerNormGeluDropoutAdd kernel → HBM

HBM reads/writes:  2 total (1 read + 1 write)
Data:              64 MB of HBM traffic

Speedup:           4× reduction in memory traffic
```

Fusion is the **most impactful single optimization** for memory-bound GPU kernels. Element-wise operations (add, multiply, activation functions, normalization) are almost always memory-bound — they do very little compute per byte loaded.

---

## 2. Arithmetic Intensity and the Roofline

To know whether fusion helps, compute **arithmetic intensity** (AI):

```
AI = FLOPs / Bytes of HBM traffic

Memory-bound region: AI < Ridge point (H200: ~412 FLOP/Byte for BF16)
Compute-bound region: AI > Ridge point

Element-wise ops:
  GELU:     ~8 FLOPs / 2 bytes = 4 FLOP/Byte  → deeply memory-bound
  LayerNorm: ~10 FLOPs / 2 bytes = 5 FLOP/Byte → deeply memory-bound
  Add:       ~1 FLOPs / 4 bytes = 0.25 FLOP/Byte → pure memory-bound

GEMM (matrix multiply):
  1024³ GEMM: 2 GFLOP / 8 MB = 256 FLOP/Byte → compute-bound
```

Fusing element-wise ops eliminates HBM round-trips, **raising effective AI** from 4 to 16+ FLOP/Byte — much closer to the roofline.

---

## 3. Manual Kernel Fusion in CUDA

### Before Fusion (3 Separate Kernels)

```cpp
// Kernel 1: GELU
__global__ void gelu_kernel(float* x, float* y, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        float v = x[i];
        y[i] = 0.5f * v * (1.0f + tanhf(0.7978845608f * (v + 0.044715f * v * v * v)));
    }
}

// Kernel 2: Dropout
__global__ void dropout_kernel(float* y, float* z, float p, unsigned int seed, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        curandState state;
        curand_init(seed, i, 0, &state);
        float r = curand_uniform(&state);
        z[i] = (r > p) ? y[i] / (1.0f - p) : 0.0f;
    }
}

// Kernel 3: Residual Add
__global__ void add_kernel(float* z, float* residual, float* out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = z[i] + residual[i];
}

// Calling them (3 HBM round-trips):
gelu_kernel<<<grid, block>>>(x, y, n);
dropout_kernel<<<grid, block>>>(y, z, p, seed, n);
add_kernel<<<grid, block>>>(z, residual, out, n);
```

### After Fusion (1 Kernel, 1 HBM Round-Trip)

```cpp
__global__ void fused_gelu_dropout_add(
    const float* __restrict__ x,
    const float* __restrict__ residual,
    float* __restrict__ out,
    float dropout_prob,
    unsigned int seed,
    int n
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    // Load once from HBM
    float v   = x[i];
    float res = residual[i];

    // GELU in registers (no HBM write)
    float gelu_val = 0.5f * v * (1.0f + tanhf(0.7978845608f * (v + 0.044715f * v * v * v)));

    // Dropout in registers (no HBM write)
    curandState state;
    curand_init(seed, i, 0, &state);
    float r = curand_uniform(&state);
    float dropped = (r > dropout_prob) ? gelu_val / (1.0f - dropout_prob) : 0.0f;

    // Residual add in registers (no HBM write)
    float result = dropped + res;

    // Write once to HBM
    out[i] = result;
}
```

Memory traffic reduced: 3 reads + 3 writes → 2 reads + 1 write = **5× fewer HBM accesses**.

---

## 4. Fusion with Triton (Pythonic High-Performance Kernels)

Triton is NVIDIA's Python-based kernel compiler. It produces optimized PTX without writing CUDA C++.

```python
import triton
import triton.language as tl

@triton.jit
def fused_layernorm_gelu_kernel(
    X_ptr,          # input
    W_ptr,          # layernorm weight (gamma)
    B_ptr,          # layernorm bias (beta)
    OUT_ptr,        # output
    n_cols,         # hidden dimension
    eps: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    # Each program (block) handles one row (one token)
    row = tl.program_id(0)
    X_row = X_ptr + row * n_cols

    # Load the entire row into SRAM (shared memory equivalent)
    offsets = tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_cols
    x = tl.load(X_row + offsets, mask=mask, other=0.0)

    # LayerNorm (all in registers/SRAM — no HBM round-trip)
    mean = tl.sum(x, axis=0) / n_cols
    x_centered = x - mean
    var = tl.sum(x_centered * x_centered, axis=0) / n_cols
    x_norm = x_centered / tl.sqrt(var + eps)

    # Scale and shift
    w = tl.load(W_ptr + offsets, mask=mask)
    b = tl.load(B_ptr + offsets, mask=mask)
    x_scaled = x_norm * w + b

    # GELU (still in registers)
    # Approximation: GELU(x) ≈ x * sigmoid(1.702 * x)
    gelu_out = x_scaled * tl.sigmoid(1.702 * x_scaled)

    # One write to HBM
    tl.store(OUT_ptr + row * n_cols + offsets, gelu_out, mask=mask)


def fused_layernorm_gelu(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor):
    M, N = x.shape
    output = torch.empty_like(x)
    BLOCK_SIZE = triton.next_power_of_2(N)

    fused_layernorm_gelu_kernel[(M,)](
        x, weight, bias, output,
        n_cols=N, eps=1e-5,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    return output
```

---

## 5. torch.compile — Automatic Fusion

`torch.compile` with the Inductor backend **automatically fuses element-wise operations** using Triton under the hood:

```python
import torch

# Unfused model — PyTorch executes each op as a separate kernel
def forward_unfused(x, residual, w, b):
    x = torch.nn.functional.layer_norm(x, [x.size(-1)], w, b)
    x = torch.nn.functional.gelu(x)
    x = torch.nn.functional.dropout(x, p=0.1, training=True)
    x = x + residual
    return x

# Fused model — torch.compile generates a single Triton kernel
forward_fused = torch.compile(forward_unfused, mode="max-autotune")

# First call: compiles + fuses (slow ~10-60s)
y = forward_fused(x, residual, w, b)

# Subsequent calls: single fused Triton kernel
y = forward_fused(x, residual, w, b)  # 3-5× faster than unfused
```

To see the generated Triton code:

```python
import torch._inductor.config as inductor_config
inductor_config.debug = True   # prints generated Triton kernels to stdout
torch.compile(fn)(x)
```

---

## 6. FlashAttention: The Most Impactful Fusion in AI

FlashAttention is the canonical example of kernel fusion in AI — it fuses the entire attention computation to avoid materializing the N×N attention matrix in HBM.

### Unfused Attention

```
Q, K, V ∈ R^(seq × head_dim)

Step 1: S = Q @ K^T              → [seq, seq] in HBM (HUGE for long seq)
Step 2: P = softmax(S / √d)      → read S from HBM, write P to HBM
Step 3: O = P @ V                → read P from HBM

HBM for S: seq² × 2 bytes
seq=8192: 8192² × 2 = 134 MB per head
64 heads: 8.6 GB per attention layer
80 layers: 688 GB of attention matrices!
→ Does not fit in HBM for long contexts
```

### FlashAttention (Fused)

```
Process Q, K, V in tiles that fit in SRAM (shared memory):

For each tile of Q:
  For each tile of K, V:
    Load tile_Q, tile_K, tile_V into SRAM (fast)
    Compute partial S_tile = tile_Q @ tile_K^T  (in SRAM)
    Compute partial softmax with running correction  (in SRAM)
    Accumulate partial O_tile += softmax × tile_V  (in SRAM)

Write final O to HBM (once per Q tile)

HBM traffic: O(seq × head_dim) instead of O(seq²)
→ 5-20× less HBM traffic for seq > 1024
```

```python
# PyTorch SDPA automatically uses FlashAttention when available
out = torch.nn.functional.scaled_dot_product_attention(
    q, k, v,
    attn_mask=None,
    dropout_p=0.0,
    is_causal=True,   # causal mask applied inside fused kernel
)
# → Single fused CUDA kernel (FlashAttention 2/3)
# → No N² HBM allocation
# → 2-4× faster than naive attention for seq > 1024
```

---

## 7. Common Fusion Patterns in LLM Inference

| Pattern | Unfused Kernels | Fused | Speedup |
|---|---|---|---|
| QKV projection | Linear + Bias + Split | FusedQKVLinear | 1.5× |
| Attention | QK matmul + Softmax + AV matmul | FlashAttention | 3–8× |
| MLP | Gate + SiLU + Mul + Up proj | SwiGLU fused | 2× |
| LayerNorm + linear | 2 kernels | FusedLinearLayerNorm | 1.4× |
| Sampling | Logits + TopK + Sample | FusedSampling | 2× |
| Embedding + RoPE | 2 kernels | FusedEmbeddingRoPE | 1.3× |

---

## 8. Profiling to Find Fusion Opportunities

```bash
# Step 1: Profile with Nsight Systems to see all kernels
nsys profile python inference.py
# Open in nsys-ui → GPU rows → find clusters of small kernels

# Step 2: Use PyTorch profiler to identify bottlenecks
from torch.profiler import profile, ProfilerActivity

with profile(
    activities=[ProfilerActivity.CUDA],
    record_shapes=True,
) as prof:
    model(input)

# Print top CUDA kernels by self_cuda_time_total
print(prof.key_averages().table(sort_by="self_cuda_time_total", row_limit=20))
```

**Fusion opportunity signals:**
- Many small kernels (< 1 ms each) in sequence
- Same input tensor read by multiple consecutive kernels
- Element-wise ops between two GEMMs
- Memory bandwidth near peak but SM compute utilization low

---

## 9. Operator Fusion in TensorRT-LLM

TRT-LLM has a dedicated fusion pass:

```bash
# Enable aggressive fusion during engine build
trtllm-build \
    --checkpoint_dir /ckpt \
    --output_dir /engine \
    --gemm_plugin bfloat16 \                     # fused GEMM+Bias
    --gpt_attention_plugin bfloat16 \            # FlashAttention fusion
    --context_fmha enable \                      # fused context phase attention
    --use_paged_context_fmha enable \            # paged FlashAttention
    --strongly_typed                             # type-specific fusion
```

The TRT-LLM engine applies:
- GEMM + bias fusion
- Layernorm + linear fusion
- Attention QKV fusion
- SwiGLU/GeGLU fusion
- RoPE embedding fusion

Result: **60–80% fewer kernel launches** than PyTorch eager mode.

---

## References

- [FlashAttention Paper](https://arxiv.org/abs/2205.14135)
- [FlashAttention-2 Paper](https://arxiv.org/abs/2307.08691)
- [OpenAI Triton](https://triton-lang.org/)
- [torch.compile Fusion](https://pytorch.org/tutorials/intermediate/torch_compile_tutorial.html)
- [TensorRT Kernel Fusion](https://docs.nvidia.com/deeplearning/tensorrt/developer-guide/index.html#fusion)
- [Nsight Compute Roofline](https://docs.nvidia.com/nsight-compute/NsightCompute/index.html#roofline-chart)
