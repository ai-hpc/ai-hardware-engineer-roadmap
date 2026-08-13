# 02 — Inference Optimization on L40S

## 1. Why L40S Inference is Different from H200

| Constraint | Impact | Mitigation |
|---|---|---|
| 864 GB/s vs 4.8 TB/s HBM | Decode is 5× more memory-bound | Larger batches, quantization |
| PCIe x16 for GPU-GPU | All-reduce is 14× slower | Minimize TP degree, use pipeline parallel |
| 48 GB per GPU | Smaller model shards | More aggressive quantization (INT4) |
| No NVLink | High latency tensor parallel | Prefer pipeline parallelism for multi-GPU |
| FP8 (no TE hardware scaling) | Manual quantization required | Use GPTQ/AWQ offline |

## 2. Quantization: Essential for L40S

Quantization is more important on L40S than H200 because:
1. Smaller per-GPU memory → larger models need more compression
2. Lower memory bandwidth → quantized ops improve memory-bound performance

### GPTQ (Post-Training Quantization)

```bash
# Install AutoGPTQ
pip install auto-gptq

# Quantize Llama-3 70B to INT4 (GPTQ)
python - <<'EOF'
from transformers import AutoTokenizer
from auto_gptq import AutoGPTQForCausalLM, BaseQuantizeConfig

model_name = "meta-llama/Llama-3-70b"
tokenizer = AutoTokenizer.from_pretrained(model_name)

quantize_config = BaseQuantizeConfig(
    bits=4,               # INT4 quantization
    group_size=128,       # quantization group size (128 is standard)
    damp_percent=0.01,
    desc_act=True,        # activation ordering (better quality)
)

model = AutoGPTQForCausalLM.from_pretrained(
    model_name,
    quantize_config=quantize_config,
    device_map="auto",
)

# Calibration data (128 samples, 2048 tokens each)
examples = [tokenizer("calibration text " * 200, return_tensors="pt")]
model.quantize(examples)
model.save_quantized("/models/llama-3-70b-gptq-int4")
EOF
```

INT4 GPTQ memory savings:
- FP16: 140 GB (70B model)
- INT4: ~35 GB (70B model) → fits on **1 L40S!** (with KV cache limits)

### AWQ (Activation-Aware Weight Quantization)

```bash
pip install autoawq

python - <<'EOF'
from awq import AutoAWQForCausalLM
from transformers import AutoTokenizer

model_path = "meta-llama/Llama-3-70b"
quant_path = "/models/llama-3-70b-awq-int4"

model = AutoAWQForCausalLM.from_pretrained(model_path, device_map="cuda:0")
tokenizer = AutoTokenizer.from_pretrained(model_path)

quant_config = {
    "zero_point": True,     # zero-point quantization (better quality)
    "q_group_size": 128,    # group size
    "w_bit": 4,             # INT4
    "version": "GEMM",      # GEMM or GEMV kernel
}

model.quantize(tokenizer, quant_config=quant_config)
model.save_quantized(quant_path)
tokenizer.save_pretrained(quant_path)
EOF
```

AWQ vs GPTQ comparison:
- AWQ: slightly better perplexity, faster inference (optimized GEMM kernels)
- GPTQ: more control, `desc_act=True` gives best quality
- Both: ~4× memory reduction vs FP16

### FP8 Static Quantization for L40S

```python
# For L40S, use offline FP8 quantization (no hardware TE scaling)
from transformers import AutoModelForCausalLM
import torch

model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-3-8b",
    torch_dtype=torch.float8_e4m3fn,  # requires PyTorch 2.1+
    device_map="cuda:0",
)
# Note: FP8 on Ada gives ~1.4× speedup vs FP16 (vs ~2× on Hopper with TE)
```

### Quantization Decision Guide

| Model Size | L40S Strategy | GPUs Needed | Notes |
|---|---|---|---|
| 7B | FP16 or BF16 | 1 | 14 GB, fast, no quality loss |
| 13B | FP16 | 1 | 26 GB, fits with small KV cache |
| 34B | INT8 or AWQ INT4 | 1-2 | INT8: 34 GB (1 GPU), INT4: 17 GB |
| 70B | AWQ/GPTQ INT4 | 1-2 | INT4: 35 GB (1 GPU), max context limited |
| 180B | GPTQ INT4 | 4-5 | 90 GB total, need multi-GPU |

## 3. vLLM Configuration for L40S

```python
from vllm import LLM, SamplingParams

# Single GPU, 7B model — standard deployment
llm = LLM(
    model="meta-llama/Llama-3-8b-instruct",
    dtype="bfloat16",
    max_model_len=8192,
    gpu_memory_utilization=0.90,
    max_num_seqs=128,
)

# Single GPU, 70B INT4 AWQ — fits on 1 L40S
llm = LLM(
    model="/models/llama-3-70b-awq-int4",
    quantization="awq",
    dtype="float16",
    max_model_len=4096,             # limit context due to 48 GB constraint
    gpu_memory_utilization=0.85,    # leave room for KV cache
    max_num_seqs=64,
)

# Multi-GPU, 70B BF16 — 2 L40S with TP=2
llm = LLM(
    model="meta-llama/Llama-3-70b-instruct",
    tensor_parallel_size=2,         # PCIe limited; keep TP low
    dtype="bfloat16",
    max_model_len=8192,
    gpu_memory_utilization=0.90,
)
```

### vLLM Tuning for PCIe Systems

```bash
# L40S-specific vLLM launch flags
python -m vllm.entrypoints.openai.api_server \
    --model meta-llama/Llama-3-8b-instruct \
    --dtype bfloat16 \
    --max-model-len 8192 \
    --gpu-memory-utilization 0.90 \
    --max-num-seqs 256 \
    --max-num-batched-tokens 32768 \
    --block-size 16 \
    --port 8000

# For GPTQ/AWQ quantized models
python -m vllm.entrypoints.openai.api_server \
    --model /models/llama-3-70b-awq \
    --quantization awq \
    --max-model-len 4096 \
    --gpu-memory-utilization 0.85 \
    --max-num-seqs 128
```

## 4. Continuous Batching Strategies

### L40S Optimal Batch Size

Unlike H200 where large batch sizes are preferable, L40S has tighter memory constraints:

```
7B model on L40S (48 GB total):
  Weights (BF16): 14 GB
  CUDA reserved:  ~2 GB
  Available KV:   ~32 GB

KV cache per token (Llama-3 8B, FP16):
  2 × 32 layers × 8 kv-heads × 128 head-dim × 2 bytes = 131 KB/token

Max concurrent tokens at BS=256, seq=256:
  256 × 256 = 65,536 tokens × 131 KB = ~8.6 GB → fits ✓

Sweet spot for L40S 7B: batch_size=128-256
```

```python
# Benchmark batch sizes to find throughput peak
import torch, time
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-3-8b",
    torch_dtype=torch.bfloat16,
    device_map="cuda:0",
)
model.eval()

for batch_size in [1, 4, 16, 32, 64, 128]:
    input_ids = torch.randint(0, 32000, (batch_size, 128), device="cuda:0")

    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        for _ in range(3): model(input_ids)  # warmup
        t0 = time.perf_counter()
        for _ in range(20): model(input_ids)
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - t0

    tps = batch_size * 128 * 20 / elapsed
    print(f"BS={batch_size:4d}: {tps:8.0f} tokens/s")
```

## 5. Speculative Decoding

Speculative decoding is especially effective on L40S because the decode step is highly memory-bandwidth bound:

```python
# L40S speculative decoding setup
llm = LLM(
    model="meta-llama/Llama-3-70b-instruct",   # target model (2 GPUs, BF16)
    speculative_model="meta-llama/Llama-3-8b-instruct",  # draft on 1 GPU
    num_speculative_tokens=5,
    tensor_parallel_size=2,
)
```

Alternative: use a tiny draft model (< 1B) for even larger speedups:

```python
llm = LLM(
    model="meta-llama/Llama-3-8b-instruct",
    speculative_model="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    num_speculative_tokens=6,
    speculative_max_model_len=4096,
)
# Typical speedup: 1.5-2.5x on L40S (memory-bound decode benefits most)
```

## 6. KV Cache Quantization

On L40S (48 GB), KV cache compression is critical for long contexts:

```python
# vLLM with FP8 KV cache
llm = LLM(
    model="meta-llama/Llama-3-8b-instruct",
    kv_cache_dtype="fp8",       # cuts KV cache memory by 50%
    max_model_len=32768,        # now supports 32K context on single L40S
)

# FP8 KV cache impact on L40S 7B:
# FP16 KV: 131 KB/token  → 32K context needs 4.2 GB (max ~240 batch sequences at 128 tokens)
# FP8 KV:  66 KB/token   → 32K context needs 2.1 GB (nearly 2× more sequences)
```

## 7. Flash Attention 2 (L40S)

L40S supports Flash Attention 2 (not FA3, which is Hopper-specific):

```bash
pip install flash-attn --no-build-isolation
```

```python
# Flash Attention 2 is automatic in PyTorch ≥ 2.2 via SDPA
import torch
# This automatically uses FA2 on L40S (Ada)
out = torch.nn.functional.scaled_dot_product_attention(q, k, v, is_causal=True)

# Memory savings: O(N) vs O(N²) for attention map
# Speed: 2-4× faster than naive attention for seq_len > 1024
```

## 8. Triton Inference Server Setup

For production multi-model deployment across 12 L40S GPUs:

```bash
# Model repository structure
model_repo/
├── llama-3-8b/
│   ├── config.pbtxt
│   └── 1/
│       └── model.py
├── llama-3-70b-awq/
│   ├── config.pbtxt
│   └── 1/
│       └── model.py
└── ensemble/
    └── config.pbtxt

# config.pbtxt for vLLM backend on Triton
name: "llama-3-8b"
backend: "vllm"
max_batch_size: 256

model_transaction_policy {
  decoupled: true  # streaming
}

instance_group [
  {
    count: 1
    kind: KIND_GPU
    gpus: [0]  # assign to GPU 0
  }
]

parameters {
  key: "model"
  value: { string_value: "meta-llama/Llama-3-8b-instruct" }
}
parameters {
  key: "dtype"
  value: { string_value: "bfloat16" }
}
```

```bash
# Launch Triton with 12 L40S GPUs
tritonserver \
    --model-repository=/model_repo \
    --backend-config=vllm,cmdline_args="--max-num-seqs 256" \
    --http-port 8000 \
    --grpc-port 8001 \
    --log-verbose 1
```

## References

- [AutoGPTQ](https://github.com/PanQiWei/AutoGPTQ)
- [AutoAWQ](https://github.com/casper-hansen/AutoAWQ)
- [vLLM Quantization Guide](https://docs.vllm.ai/en/latest/quantization/auto_awq.html)
- [FlashAttention-2 Paper](https://arxiv.org/abs/2307.08691)
- [Triton Inference Server + vLLM](https://github.com/triton-inference-server/vllm_backend)
- [Speculative Decoding Survey](https://arxiv.org/abs/2401.07851)
