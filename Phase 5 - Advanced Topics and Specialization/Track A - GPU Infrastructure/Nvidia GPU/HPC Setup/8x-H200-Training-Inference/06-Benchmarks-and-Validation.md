# 06 — Benchmarks & Validation on 8x H200

## 1. Standard Benchmark Suite

Run these benchmarks in order to validate hardware health, software stack, and end-to-end AI performance.

### Level 0: Hardware Health Check

```bash
# GPU info and clock state
nvidia-smi -q | grep -E "Product Name|Memory Total|Clocks|Power"

# Verify all 8 GPUs visible
nvidia-smi --query-gpu=index,name,memory.total --format=csv

# NVLink health
nvidia-smi nvlink --status -i 0   # repeat for 0-7
nvidia-smi nvlink --errorcounters -i 0

# Thermal stress — should stay < 83°C at 700W TDP
nvidia-smi dmon -s pucvt -d 2
```

### Level 1: GPU Compute (HPL / DGEMM)

```bash
# NVIDIA HPL benchmark (standard Top500 benchmark)
docker run --gpus all --rm nvcr.io/nvidia/hpc-benchmarks:23.10 \
    hpl.sh --dat /workspace/hpl-dgx-h100/HPL.dat

# Expected: ~50+ PFLOPS for 8x H200 at FP64
# (HPL tests FP64; AI workloads use BF16/FP8)

# Quick GEMM bandwidth test
python - <<'EOF'
import torch, time
M, N, K = 4096, 4096, 4096
a = torch.randn(M, K, device="cuda", dtype=torch.bfloat16)
b = torch.randn(K, N, device="cuda", dtype=torch.bfloat16)
for _ in range(10): torch.mm(a, b)  # warmup
torch.cuda.synchronize()
t0 = time.perf_counter()
for _ in range(100): torch.mm(a, b)
torch.cuda.synchronize()
t1 = time.perf_counter()
tflops = 2 * M * N * K * 100 / (t1 - t0) / 1e12
print(f"GEMM TFLOPS (BF16): {tflops:.0f}")
# Target: > 1700 TFLOPS (86% of 1979 peak)
EOF
```

### Level 2: NVLink Bandwidth

```bash
# NCCL all-reduce bandwidth test
git clone https://github.com/NVIDIA/nccl-tests
cd nccl-tests && make -j MPI=0 CUDA_HOME=/usr/local/cuda

# All-reduce across 8 GPUs
./build/all_reduce_perf -b 1G -e 8G -f 2 -g 8

# Expected: ~450+ GB/s bus bandwidth (out of 900 GB/s bidirectional NVLink)
# Lower numbers indicate NVLink issues or NCCL misconfiguration
```

### Level 3: Memory Bandwidth

```bash
python - <<'EOF'
import torch, time
N = 2 * 1024**3 // 2  # 2 GB of BF16 data
x = torch.randn(N, device="cuda", dtype=torch.bfloat16)
y = torch.empty_like(x)
for _ in range(5): y.copy_(x)  # warmup
torch.cuda.synchronize()
t0 = time.perf_counter()
for _ in range(100): y.copy_(x)
torch.cuda.synchronize()
t1 = time.perf_counter()
bw = (2 * N * 2 * 100) / (t1 - t0) / 1e12  # read + write, BF16 = 2 bytes
print(f"HBM Bandwidth: {bw:.2f} TB/s")
# Target: > 4.0 TB/s (83% of 4.8 TB/s peak)
EOF
```

## 2. Training Benchmarks

### MFU (Model FLOP Utilization)

```python
import torch, time
from transformers import LlamaForCausalLM, LlamaConfig

# Proxy model for benchmarking
config = LlamaConfig(
    hidden_size=8192, num_hidden_layers=80,
    num_attention_heads=64, num_key_value_heads=8,
    intermediate_size=28672, max_position_embeddings=4096,
)
model = LlamaForCausalLM(config).to("cuda:0").to(torch.bfloat16)

batch_size, seq_len = 4, 2048
inputs = torch.randint(0, 32000, (batch_size, seq_len), device="cuda:0")

# Warmup
for _ in range(3):
    loss = model(inputs, labels=inputs).loss
    loss.backward()
    model.zero_grad()
torch.cuda.synchronize()

# Benchmark
t0 = time.perf_counter()
N_STEPS = 20
for _ in range(N_STEPS):
    loss = model(inputs, labels=inputs).loss
    loss.backward()
    model.zero_grad()
torch.cuda.synchronize()
elapsed = time.perf_counter() - t0

# MFU calculation
params = sum(p.numel() for p in model.parameters())
flops_per_token = 6 * params   # forward + backward ≈ 3x forward
tokens_per_step = batch_size * seq_len
total_flops = flops_per_token * tokens_per_step * N_STEPS
achieved_tflops = total_flops / elapsed / 1e12
mfu = achieved_tflops / 1979.0  # H200 BF16 peak
print(f"Achieved: {achieved_tflops:.0f} TFLOPS | MFU: {mfu*100:.1f}%")
# Target single-GPU MFU: > 45%
```

### Multi-GPU Scaling Efficiency

```bash
# Run training script with 1, 2, 4, 8 GPUs and measure throughput
for N in 1 2 4 8; do
    torchrun --nproc_per_node=$N train_benchmark.py \
        --batch-size $((4 * N)) \
        --seq-len 2048 \
        --steps 50 \
        --output scaling_N${N}.json
done

# Expected scaling efficiency (vs linear):
# 2 GPUs: > 95%
# 4 GPUs: > 92%
# 8 GPUs: > 88%
```

### Throughput vs Batch Size Curve

| Batch Size | Tokens/s (Llama-70B, 8x H200, BF16) | GPU Mem Used |
|---|---|---|
| 1 | ~1,200 | ~200 GB |
| 4 | ~4,500 | ~240 GB |
| 16 | ~14,000 | ~400 GB |
| 64 | ~18,000 | ~900 GB |
| 128 | OOM (need FP8 or quant) | — |

## 3. Inference Benchmarks

### Latency vs Throughput Trade-off

```bash
# vLLM benchmark suite
python benchmarks/benchmark_serving.py \
    --model meta-llama/Llama-3-70b-instruct \
    --tensor-parallel-size 8 \
    --request-rate 10 \
    --num-prompts 500 \
    --input-len 512 \
    --output-len 256

# Sweep request rates
for RATE in 1 5 10 20 50 100; do
    python benchmarks/benchmark_serving.py \
        --request-rate $RATE \
        [other args] \
        --output-json rate_${RATE}.json
done
```

### Key Inference Metrics

| Metric | Definition | H200 Target (Llama-70B, TP=8) |
|---|---|---|
| **TTFT** | Time to first token (prefill) | < 100 ms (512 input tokens) |
| **TPOT** | Time per output token (decode) | < 20 ms/token |
| **Throughput** | Output tokens/second total | > 15,000 tokens/s at BS=64 |
| **ITL** | Inter-token latency (streaming) | < 30 ms P99 |
| **MBU** | Memory bandwidth utilization | > 75% (decode is memory-bound) |

### Decode MBU Calculation

```python
def compute_mbu(model_params_bytes, tokens_per_second, hbm_bw_tbs=4.8):
    """
    Decode is memory-bound: GPU loads all model weights per token generated.
    model_params_bytes: total model size in bytes (e.g., 70B * 2 for BF16 = 140e9)
    """
    bytes_per_token = model_params_bytes  # approximately load full model
    achieved_tb_s = bytes_per_token * tokens_per_second / 1e12
    mbu = achieved_tb_s / hbm_bw_tbs
    return mbu

# 70B BF16: 140 GB weights, 15000 tokens/s
mbu = compute_mbu(140e9, 15000)
print(f"MBU: {mbu*100:.1f}%")  # ~44% — room for improvement with larger batches
```

## 4. Validation Checklist

### Before Production Deployment

```
Hardware:
[ ] All 8 GPUs detected by nvidia-smi (driver version ≥ 550)
[ ] NVLink errors = 0 on all links (nvidia-smi nvlink --errorcounters)
[ ] GPU temperatures stable under load (< 83°C)
[ ] Power readings ~700W per GPU at full load
[ ] ECC errors = 0 (nvidia-smi --query-gpu=ecc.errors.uncorrected.volatile.total)

Software stack:
[ ] CUDA 12.3+, cuDNN 9+, NCCL 2.20+
[ ] PyTorch ≥ 2.3 (Flash Attention 3 support)
[ ] vLLM or TRT-LLM latest release
[ ] Transformer Engine installed for FP8

Networking:
[ ] NCCL all-reduce latency < 100 µs for 1 MB payload
[ ] NVLink bus bandwidth > 400 GB/s (nccl-tests all_reduce_perf)
[ ] No PCIe retrain errors in dmesg

Model serving:
[ ] Health check endpoint responds < 50 ms
[ ] TTFT within SLA for p50/p99
[ ] Memory utilization stable (no slow leak)
[ ] Graceful handling of max_model_len exceeded requests
```

### ECC and Error Monitoring

```bash
# Check for GPU hardware errors
nvidia-smi --query-gpu=gpu_name,ecc.errors.corrected.volatile.total,\
ecc.errors.uncorrected.volatile.total --format=csv

# Set ECC mode (requires reboot)
sudo nvidia-smi -e 1  # enable ECC (reduces memory by ~6%)
sudo nvidia-smi -e 0  # disable ECC (more memory, no correction)

# For training: ECC ON (data integrity > memory)
# For inference: ECC ON (production reliability)
```

## 5. Comparative Baselines

### H200 vs H100 vs A100

| Benchmark | A100 SXM4 80GB | H100 SXM5 80GB | H200 SXM5 141GB |
|---|---|---|---|
| BF16 TFLOPS | 312 | 989 | 1,979 |
| HBM Bandwidth | 2.0 TB/s | 3.35 TB/s | 4.8 TB/s |
| Llama-70B TTFT (512 in) | ~400 ms | ~150 ms | ~90 ms |
| Llama-70B throughput | ~5K tok/s | ~10K tok/s | ~18K tok/s |
| Max context (fits in VRAM) | ~32K tokens | ~32K tokens | ~128K tokens |

## References

- [NVIDIA H200 Performance Data](https://www.nvidia.com/en-us/data-center/h200/)
- [MLPerf Inference Results](https://mlcommons.org/benchmarks/inference-datacenter/)
- [nccl-tests Repository](https://github.com/NVIDIA/nccl-tests)
- [vLLM Benchmarks](https://github.com/vllm-project/vllm/tree/main/benchmarks)
- [LLM-Perf Benchmark Suite](https://github.com/huggingface/optimum-benchmark)
