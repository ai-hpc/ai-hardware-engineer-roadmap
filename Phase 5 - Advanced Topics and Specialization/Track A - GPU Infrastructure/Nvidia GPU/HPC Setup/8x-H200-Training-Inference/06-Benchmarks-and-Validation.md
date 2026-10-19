# 06 — Benchmark H200 with a reproducible evidence trail

**Outcome:** distinguish arithmetic correctness, hardware ceilings and service performance. Source review: 2026-10-18. **No H200 measurements are supplied here**; previous untraceable token-rate/latency tables are not baselines. The [CPU distributed reference](../NCCL-Deep-Dive/CPU-Collectives-Lab/Guide.md) supplies actual portable correctness evidence.

## Record the experiment

Capture GPU SKU/count, driver, CUDA/framework/library versions, model and tokenizer revisions, power/clock policy, topology, parallel groups, precision, input/output token distributions, batch/concurrency and warmup. Retain raw outputs and failures. Change one factor at a time and repeat runs to show variation.

```bash
nvidia-smi --query-gpu=index,name,memory.total,power.limit --format=csv
nvidia-smi topo -m
nvidia-smi nvlink --status -i 0
python -c 'import torch; print(torch.__version__, torch.version.cuda)'
```

Check vendor/site thermal limits and error counters; a universal 83°C threshold is not a server acceptance test. Do not toggle ECC during benchmarking. Hardware errors need the [diagnostic procedure](../NCCL-Deep-Dive/06-Debugging.md).

## Small correctness test, then a bounded GPU microbenchmark

The script below requires CUDA and BF16 support. It uses at most 256 MiB per copy buffer and chooses a smaller allocation from currently free memory. It measures **one GPU**, not the sum of eight GPU peaks. It was syntax-reviewed but not GPU-executed for this update.

```python
import json
import torch

if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
    raise SystemExit("A CUDA GPU with BF16 support is required")
torch.manual_seed(7)
device = torch.device("cuda:0")
torch.cuda.set_device(device)

# Check a small, non-square product before timing a larger shape.
a = torch.randn(37, 53, device=device, dtype=torch.bfloat16)
b = torch.randn(53, 19, device=device, dtype=torch.bfloat16)
actual = (a @ b).float()
reference = (a.cpu().double() @ b.cpu().double()).float().to(device)
relative_l2 = ((actual - reference).norm() / reference.norm()).item()
assert relative_l2 < .01, relative_l2

def mean_ms(operation, repeats=30):
    for _ in range(5): operation()
    torch.cuda.synchronize(device)
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(repeats): operation()
    end.record()
    end.synchronize()
    return start.elapsed_time(end) / repeats

M, N, K = 2048, 2048, 2048
a = torch.randn(M, K, device=device, dtype=torch.bfloat16)
b = torch.randn(K, N, device=device, dtype=torch.bfloat16)
c = torch.empty(M, N, device=device, dtype=torch.bfloat16)
gemm_ms = mean_ms(lambda: torch.mm(a, b, out=c))
free_bytes, _ = torch.cuda.mem_get_info(device)
# Two buffers consume at most 20% of reported free memory here.
buffer_bytes = min(256 * 1024**2, int(free_bytes * .10))
if buffer_bytes < 16 * 1024**2:
    raise SystemExit("Insufficient free memory for the copy experiment")
x = torch.full((buffer_bytes // 2,), 1.25, device=device, dtype=torch.bfloat16)
y = torch.empty_like(x)
copy_ms = mean_ms(lambda: y.copy_(x))
assert torch.equal(x, y)
print(json.dumps({
    "device": torch.cuda.get_device_name(device),
    "torch": torch.__version__, "cuda": torch.version.cuda,
    "gemm_relative_l2": relative_l2, "gemm_ms": gemm_ms,
    "dense_BF16_TFLOPs": 2 * M * N * K / (gemm_ms / 1000) / 1e12,
    "copy_buffer_bytes": x.numel() * x.element_size(), "copy_ms": copy_ms,
    "copy_read_plus_write_TBps": 2 * x.numel() * x.element_size() / (copy_ms / 1000) / 1e12
}, indent=2))
```

The copy metric counts **read plus write** traffic; a one-way GPU-to-GPU link copy instead counts payload bytes once. A working set smaller than cache can inflate an apparent memory-bandwidth figure. Sweep size, record whether it exceeds cache, and use profiling counters before calling it HBM bandwidth. Event timing excludes allocations and host transfer; end-to-end application measurements must include costs on their path.

The H200 SXM dense BF16 ceiling is about **989.5 TFLOP/s**, not the 1,979 sparse number. Eight GPUs' FP64 Tensor Core peaks total **0.536 PFLOP/s**, so a claimed 50 PFLOP/s FP64 HPL result is impossible for this configuration. HPL, HPL-AI and BF16 GEMM are different workloads. [H200 specification](https://www.nvidia.com/en-us/data-center/h200/).

## Validate communication separately

Use the [NCCL sweep](../NCCL-Deep-Dive/02-Algorithms-and-Bandwidth.md), save the nccl-tests commit, and require zero wrong values. Report both algorithm bandwidth and normalized bus bandwidth, operation size, rank count and in-place/out-of-place mode. A result below a guessed 450 GB/s threshold does not by itself diagnose a faulty link.

## Training throughput and MFU

Start with a model that fits and a working training script; do not instantiate a 70B FP32 model on one GPU as a “quick proxy.” Count **global valid tokens per optimizer step**, including all data-parallel workers and accumulation, once. For a dense model, a rough useful-compute estimate is `6P×global_tokens`. MFU divides this by elapsed step time times the number of GPUs times the **matching dense** arithmetic peak. State omitted attention/recompute costs and report achieved tokens/s alongside this approximation.

Record warm steady-state steps, optimizer behavior, quality/loss and peak memory on every rank. Report the slowest rank/end-to-end step. A fast result that silently processes fewer examples or omits optimizer work is a different benchmark.

## Inference: throughput and latency need a workload

Use a pinned installed runtime and its `--help`; benchmark interfaces evolve. Current vLLM exposes `vllm bench throughput` for offline execution and `vllm bench serve` for a running server. Set a local immutable model snapshot path after checking capacity and model support; no model is downloaded by the course update.

```bash
vllm --version
vllm bench throughput --help
vllm bench serve --help
# Example offline command after MODEL points to a prepared local snapshot:
vllm bench throughput --model "$MODEL" --input-len 512 --output-len 128 \
    --num-prompts 100 --tensor-parallel-size 8
```

For online tests, state arrival process/rate, concurrency, timeouts, warmup and draining behavior. Report success/error counts, TTFT, inter-token timing, completion latency and **goodput** meeting the task's quality/latency limits. Offline maximum tokens/s is not an online p99 guarantee. [vLLM throughput CLI](https://docs.vllm.ai/en/latest/cli/bench/throughput/), [serving benchmark CLI](https://docs.vllm.ai/en/latest/cli/bench/serve/).

A streaming network chunk is not a model token. Empty role events, partial JSON, multiple tokens per content delta and usage-only events break naive counters. Use the runtime's supported benchmark client and exact tokenizer/usage semantics; do not count lines containing `content`.

**Assessment:** submit one numerical reference check, one bounded compute/copy result if CUDA is available, a collective sweep, and a serving workload manifest. Explain why a faster throughput result can still fail the same robot/cloud service latency requirement. Keep hardware work explicitly unexecuted until raw results exist.
