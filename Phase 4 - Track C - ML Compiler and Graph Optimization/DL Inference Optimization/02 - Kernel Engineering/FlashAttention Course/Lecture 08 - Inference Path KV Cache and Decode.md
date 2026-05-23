# Lecture 8 — Inference Path: KV Cache, Decode, RoPE, GQA, Paged KV

**Parent:** [FlashAttention Course](Guide.md)

**One-line purpose:** Understand the decode-time variant of FlashAttention — `q_len = 1`, in-kernel KV cache update, paged KV, GQA / MQA, rotary, ALiBi, sliding window — and the shape of the APIs that Qwen / vLLM / TensorRT-LLM use to call it.

**Prerequisites:** Lectures 1–7. Familiarity with autoregressive LLM decoding.

**Artifact:** A microbenchmark of decode-step latency with and without paged KV, plus a small RoPE / GQA sanity check that confirms the in-kernel rotary matches a separate-rotary reference.

---

## Why it matters

Training is dominated by the *prefill / forward* shape (`B × N × H × D`). Inference, after the prompt is processed, is dominated by the *decode* shape (`B × 1 × H × D`) — one query token per batch element, attending over all previously-generated KV. The kernel design is different: you no longer have a `B_r × d` query tile, you have a single row, and the GPU's challenge is to fill its width while reading the KV cache once per step.

Serving stacks (vLLM, SGLang, TensorRT-LLM, NVIDIA Triton) live or die by this kernel. Single-request best-TPS on H200 is bounded by how cleanly you stream KV from HBM with the smallest possible overhead per token.

---

## Mental model

### Decode shape and what changes

| | Prefill | Decode |
|---|---------|--------|
| `q_len` | full prompt (e.g. 32k) | 1 |
| `kv_len` | same as q_len | grows by 1 per step |
| Bottleneck | compute (matmul-bound) | memory bandwidth (read KV + weights) |
| FA kernel | `flash_attn_func`, `flash_attn_varlen_func` | `flash_attn_with_kvcache` |

Because `q_len = 1`, you no longer have the `B_r × d` query tile that filled an SM. Instead, the kernel parallelises across **batch × head × KV-tile** and uses a much smaller `(1, B_c)` MMA shape. Tensor cores prefer larger matmuls, so decode is consistently below GEMM peak — typically 30–50% of peak FLOPs even when implemented perfectly.

### In-kernel KV cache update

Every decode step you compute new `k_new` and `v_new` for the current token and write them into the KV cache at position `cur_pos`. The fused inference kernel does this **inside the kernel**:

```
flash_attn_with_kvcache(q, k_cache, v_cache, k_new, v_new, cache_seqlens, ...)
```

There is no separate kv_append kernel launch — the same kernel that does attention also writes the new tokens. That saves a kernel launch per token. For a model with 80 layers and 32 ms/token, that is 80 launches saved × ~5 µs = 400 µs/token, or ~10% TPS at 50 TPS.

### Paged KV cache

For large-batch serving you cannot pre-allocate a contiguous KV tensor per request (memory waste from variable lengths). Instead, KV is split into fixed-size **pages** (e.g. 16 tokens × `H × D`) and a **page table** per request maps logical positions to physical pages.

The attention kernel reads from `k_cache[block_indices[i]]` rather than a contiguous slab — same compute, indirected loads. FlashInfer and FA's `with_kvcache` both support this; the page size is a kernel-time constant (typically 16, 32, or 64).

### MQA / GQA

- **MHA** (multi-head attention): one K and one V per Q head. KV size = `N × H × D`.
- **GQA** (grouped-query attention): K and V are shared across groups of Q heads. With `H_kv = H / G`, KV size = `N × H_kv × D`. Reduces KV bandwidth by `G`.
- **MQA** (multi-query attention): `H_kv = 1`. Biggest KV bandwidth saving, worst quality.

Qwen2.5 and Llama 3 use GQA with `G = 4` or `8`. The attention kernel handles GQA by broadcasting K/V across the group — you compute the same K·V tile against multiple Q heads.

### RoPE (rotary position embedding)

Each Q and K vector is rotated by a position-dependent matrix `R(pos)` before the dot product. Done correctly, it bakes positional information into the attention scores without an additive bias.

Two options:

1. **External rotary kernel** — call a separate kernel that rotates Q and K, then pass to attention.
2. **In-kernel rotary** — the attention kernel rotates Q and K as it loads them.

`flash_attn_with_kvcache` supports option 2 via `rotary_cos` / `rotary_sin` arguments. It saves a kernel launch per token. The cost is a few extra FMAs per element — negligible.

### ALiBi (Attention with Linear Biases)

Adds a per-head linear bias `-m_h · |i - j|` to the attention scores. No K matrix needed for position. FA supports it via the `alibi_slopes` argument. Not common in modern LLMs but used in MPT and some experimental architectures.

### Sliding-window attention

Each query attends only to the last `window` keys. `flash_attn_func(window_size=(left, right))` clips the attention range — same code path, just a tighter mask. Used in Mistral, some Gemma variants.

---

## Build it

### 1. Decode-step latency benchmark

```python
# decode_bench.py
import torch
import torch.cuda
from flash_attn import flash_attn_with_kvcache

B, H, D = 1, 32, 128
H_kv = 8     # GQA group=4
N_max = 32768

q = torch.randn(B, 1, H, D, device="cuda", dtype=torch.bfloat16)
k_cache = torch.randn(B, N_max, H_kv, D, device="cuda", dtype=torch.bfloat16)
v_cache = torch.randn_like(k_cache)
k_new = torch.randn(B, 1, H_kv, D, device="cuda", dtype=torch.bfloat16)
v_new = torch.randn_like(k_new)

def step(cur_len):
    cache_seqlens = torch.tensor([cur_len], device="cuda", dtype=torch.int32)
    return flash_attn_with_kvcache(
        q, k_cache, v_cache, k=k_new, v=v_new,
        cache_seqlens=cache_seqlens, causal=True,
    )

# Warmup
for L in [128, 1024, 4096, 16384]:
    _ = step(L)
torch.cuda.synchronize()

start = torch.cuda.Event(enable_timing=True)
end = torch.cuda.Event(enable_timing=True)
for L in [128, 1024, 4096, 16384, 32000]:
    times = []
    for _ in range(50):
        start.record(); _ = step(L); end.record()
        end.synchronize()
        times.append(start.elapsed_time(end))
    print(f"L={L:>5}  median={sorted(times)[25]:.3f} ms")
```

You should see decode latency grow roughly linearly with KV length (memory-bound: each step reads `L × H_kv × D × 2` bytes per layer). At `L = 16k` for 7B with `H_kv = 8, D = 128`, that is 32 MB per layer per step — for a 32-layer model that is ~1 GB per token. At 4.8 TB/s, that is ~200 µs of pure KV traffic per token.

### 2. Paged-vs-contiguous comparison (uses FlashInfer)

FlashInfer's `BatchDecodeWithPagedKVCacheWrapper` is the canonical paged decode API:

```python
import flashinfer
import torch

B, H, D, page_size, N = 1, 32, 128, 16, 4096
H_kv = 8
num_pages = N // page_size

q = torch.randn(B, H, D, device="cuda", dtype=torch.bfloat16)
k_cache = torch.randn(num_pages, page_size, H_kv, D, device="cuda", dtype=torch.bfloat16)
v_cache = torch.randn_like(k_cache)
indices = torch.arange(num_pages, device="cuda", dtype=torch.int32)
indptr  = torch.tensor([0, num_pages], device="cuda", dtype=torch.int32)
last_page_len = torch.tensor([page_size], device="cuda", dtype=torch.int32)

wrapper = flashinfer.BatchDecodeWithPagedKVCacheWrapper(
    torch.empty(128*1024*1024, dtype=torch.uint8, device="cuda"),
    kv_layout="NHD",
)
wrapper.plan(indptr, indices, last_page_len, H, H_kv, D, page_size, dtype=torch.bfloat16)
o = wrapper.run(q, (k_cache, v_cache))
print(o.shape)
```

Compare the per-step latency to the FA `with_kvcache` baseline you produced in step 1. Paged decode should be within 10–15% of contiguous decode on H100/H200; if not, the page-table indirection is your bottleneck and you tune page size.

### 3. RoPE / GQA sanity check

```python
# rope_gqa_check.py
# Verifies that in-kernel rotary in flash_attn_with_kvcache produces the same
# output as separately-applied rotary + attention.
```

Apply rotary externally (using `rotary_embedding` from your codebase or PyTorch), then call attention without `rotary_cos`/`rotary_sin`. Separately, call `flash_attn_with_kvcache(rotary_cos=..., rotary_sin=...)` without external rotary. Compare outputs at bf16 with `atol = 5e-3`. They must match.

---

## Use it in the real stack

- **vLLM**: `vllm/attention/backends/flash_attn.py` calls `flash_attn_with_kvcache` for decode, `flash_attn_varlen_func` for chunked prefill.
- **SGLang**: similar pattern, plus their own RadixAttention layer on top for prefix caching.
- **TensorRT-LLM**: uses its own attention kernels (fused with MQA/GQA + RoPE) but shape-wise identical to what FA provides.
- **The cacheon-sglang-miner repo we worked on**: see `cuda/src/kernels/attention_flashinfer.cu` for a hand-written wrapper around FlashInfer's paged decode. It is a worked example of the API in production.

Skim each one. The patterns repeat: prefill via varlen, decode via paged/with_kvcache, CUDA-graph capture around the decode step to amortise launch overhead.

---

## Measure it

For decode benchmarks, report:

- **TTFT** (time to first token) — prefill latency.
- **TPS** (tokens per second) for a long generation — 1 / decode-step time.
- **Achieved HBM bandwidth** at `L_max`: should be 70–90% of GPU peak. If lower, you have launch or scheduling overhead, not a memory-bound kernel.
- **KV-cache memory footprint** per token per layer: `2 · H_kv · D · dtype_bytes`. For 32k context this often dwarfs activation memory.

Always benchmark **after** capturing a CUDA graph if your serving stack uses graphs (most do). Pre-graph and post-graph numbers can differ by 2× for short sequences.

---

## Ship it

Drop into `flash-attn-course/`:

1. `decode_bench.py` and a `decode_latency.csv` over `L ∈ {128, 1k, 4k, 16k, 32k}`.
2. `paged_vs_contig.csv` comparing FA `with_kvcache` and FlashInfer paged decode at one shape.
3. `rope_gqa_check.py` with a passing tolerance report.

If you have those three, you can have an informed conversation about any inference-serving stack on the planet.

---

## Related pages

- [Lecture 7 — Backward pass and validation](Lecture%2007%20-%20Backward%20Pass%20and%20Validation.md)
- [Lecture 9 — Hopper / FA3 / FA4](Lecture%2009%20-%20Hopper%20FA3%20FA4.md)
- [DL Inference Runtimes and Deployment](../../05%20-%20Inference%20Runtimes%20and%20Deployment/Guide.md)
- FlashInfer: <https://github.com/flashinfer-ai/flashinfer>
