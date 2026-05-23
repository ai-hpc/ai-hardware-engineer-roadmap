# Lecture 5 — Repo Anatomy and Python / CUDA API

**Parent:** [FlashAttention Course](Guide.md)

**One-line purpose:** Map every important directory and entry point in `Dao-AILab/flash-attention` so you can trace any call from Python to the launched CUDA kernel without getting lost.

**Prerequisites:** Lectures 1–4. A local clone of `flash-attention` built against your CUDA toolkit.

**Artifact:** An annotated call trace from `flash_attn_func(...)` down to the CUDA kernel launch, plus a minimal local build recipe that survives a fresh checkout.

---

## Why it matters

The FA repo is now ~150 source files across Python, C++, and CUDA, with multiple kernel families (FA1 / FA2 / FA3), several APIs (`func`, `qkvpacked`, `varlen`, `with_kvcache`), and a complicated build that compiles different kernel families per CUDA arch. Without a map you will spend hours grepping; with a map you can land in the right file on the first try.

---

## Mental model

### Top-level layout

```
flash-attention/
├── flash_attn/                        # Python package
│   ├── flash_attn_interface.py        # the API users call
│   ├── flash_attn_triton.py           # Triton reference impl
│   ├── modules/                       # MHA / Block / Norm helpers
│   ├── ops/                           # rotary, layernorm, etc.
│   └── utils/                         # benchmarking, distributed helpers
├── csrc/
│   ├── flash_attn/                    # FA2 (Ampere/Ada) C++/CUDA
│   │   ├── flash_api.cpp              # pybind / TORCH_LIBRARY entry
│   │   └── src/
│   │       ├── flash_fwd_kernel.h     # forward kernel
│   │       ├── flash_bwd_kernel.h     # backward kernel
│   │       ├── flash_fwd_launch_template.h
│   │       ├── kernel_traits.h        # tile sizes per head_dim
│   │       ├── softmax.h              # rowmax/rowsum/rescale
│   │       └── mask.h                 # causal / sliding window / alibi
│   ├── flash_attn_hopper/             # FA3 (Hopper, sm90)
│   │   └── ...                        # WGMMA + TMA + warp-specialised
│   ├── ft_attention/                  # legacy faster-transformer attention
│   ├── layer_norm/                    # fused LayerNorm CUDA
│   ├── rotary/                        # rotary embedding CUDA
│   └── xentropy/                      # cross-entropy CUDA
├── tests/                             # pytest suites; correctness vs ref
├── benchmarks/                        # microbench scripts
└── setup.py                           # picks which kernels to compile per arch
```

The two directories you will spend the most time in are `flash_attn/` (the Python facing API) and `csrc/flash_attn/src/` (FA2 kernels) or `csrc/flash_attn_hopper/` (FA3 kernels).

### The two API styles

| API | Shapes | When to use |
|-----|--------|-------------|
| `flash_attn_func(q, k, v, ...)` | `[B, S, H, D]` | The simple, fixed-length training/eval path |
| `flash_attn_qkvpacked_func(qkv, ...)` | `[B, S, 3, H, D]` | When QKV is fused (saves a transpose for many trainers) |
| `flash_attn_varlen_func(q, k, v, cu_seqlens_q, cu_seqlens_k, ...)` | packed `[total_tokens, H, D]` + `[B+1]` indptr | Variable-length batches (e.g. vLLM-style packed sequences) |
| `flash_attn_with_kvcache(...)` | decode path: `[B, 1, H, D]` query, paged KV cache | Inference decode step |

All four end up calling the same kernels under different launch templates.

### How a Python call reaches the GPU

```
flash_attn_func(q, k, v, dropout_p, softmax_scale, causal=True)
  │
  ▼ flash_attn/flash_attn_interface.py
  │   flash_attn_func -> FlashAttnFunc.apply (autograd)
  │     forward(): calls torch.ops.flash_attn._flash_attn_forward(...)
  │
  ▼ torch.ops.flash_attn._flash_attn_forward
  │   bound via TORCH_LIBRARY in flash_attn/_C
  │
  ▼ csrc/flash_attn/flash_api.cpp
  │   mha_fwd(...) — argument parsing, dispatch on head_dim
  │
  ▼ csrc/flash_attn/src/flash_fwd_launch_template.h
  │   run_mha_fwd_<head_dim, is_causal, ...>() — kernel traits, grid, launch
  │
  ▼ csrc/flash_attn/src/flash_fwd_kernel.h
      flash_fwd_kernel<...><<<grid, block, smem>>>(params)
      // does the FA1/FA2 algorithm from Lecture 3
```

Backward goes through the parallel files (`flash_bwd_kernel.h`, `flash_bwd_launch_template.h`).

The repo is organised so that the **algorithm** lives in `flash_fwd_kernel.h`, the **shape dispatch** lives in `flash_fwd_launch_template.h`, the **tile sizes** live in `kernel_traits.h`, and the **API surface** lives in `flash_api.cpp`. Once you have that mental partition, finding things is fast.

### Where FA1 vs FA2 vs FA3 actually live

- **FA1:** historical, no longer in the main branch as a separate path. The FA2 codebase is "FA1 + better scheduling"; the FA1 algorithm is recoverable by reading the FA1 paper and matching it against the FA2 kernel.
- **FA2:** `csrc/flash_attn/`. Builds for sm80/sm86/sm89/sm90.
- **FA3:** `csrc/flash_attn_hopper/`. Builds only for sm90a. Uses CUTLASS / CuTe for WGMMA + TMA. Header layout intentionally mirrors FA2 so you can diff them.

### The build (why your first build will fail)

`setup.py` enumerates kernels by `(head_dim, is_causal, dropout, alibi)` and emits one `.cu` per combination. With all flags it produces hundreds of TUs and takes 30–60 minutes to compile. Use the `MAX_JOBS` and `FLASH_ATTN_FORCE_BUILD` env vars; for a workstation, `MAX_JOBS=4` is usually safe. If you only care about one head_dim, set `FLASH_ATTENTION_DISABLE_BACKWARD=TRUE` and patch `setup.py` to compile a single dim — your build time drops to a couple of minutes.

For this course's purposes you can use the pre-built wheel; you only need to *build* if you are going to patch a kernel (Lecture 10).

---

## Build it

### 1. Trace a call

In a Python REPL with FA installed:

```python
import torch, flash_attn
print(flash_attn.__file__)
print(flash_attn.flash_attn_interface.__file__)
```

Open the second file. Find `flash_attn_func`. Note that it calls `torch.ops.flash_attn._flash_attn_forward`. That symbol is registered from C++ — find where:

```
grep -RIn "_flash_attn_forward" csrc/
```

The hit lands in `csrc/flash_attn/flash_api.cpp`. Read the `mha_fwd` function top to bottom: argument validation → `set_params_fprop()` → `run_mha_fwd<...>()`. Follow the `run_mha_fwd` call to `flash_fwd_launch_template.h`.

Write down the chain as a single-page diagram. Save it as `flash_attn_call_trace.md`.

### 2. Local minimal build

```
git clone https://github.com/Dao-AILab/flash-attention.git
cd flash-attention
pip install ninja packaging
MAX_JOBS=4 pip install --no-build-isolation -e .
```

On a 16-core workstation with H100 this takes ~30–45 minutes the first time. If you hit OOM during nvcc compile, drop `MAX_JOBS` to 2.

Verify:

```python
import torch
from flash_attn import flash_attn_func
q = torch.randn(2, 1024, 8, 64, device="cuda", dtype=torch.bfloat16)
o = flash_attn_func(q, q, q, causal=True)
print(o.shape, o.dtype)
```

If the import works and the call returns, your build is good.

### 3. A "hello, kernel" patch

Find a `printf` near the entry of `flash_fwd_kernel.h` (or add one inside an `#if 0 ... #endif` block guarded by a constexpr `if (thread0()) printf(...)`). Rebuild. Run your test from step 2. If you see the printf, you have a working build → patch → test cycle. This is what you need to have for Lecture 10.

---

## Use it in the real stack

Three production examples that exercise these APIs:

- **vLLM** uses `flash_attn_with_kvcache` for paged decode and `flash_attn_varlen_func` for chunked prefill.
- **TransformerEngine** (NVIDIA) replaces SDPA with FA3 on Hopper via the `flash_attn_hopper` path.
- **PyTorch SDPA's "flash" backend** is FA2 bound through `torch._C._aten._scaled_dot_product_flash_attention`.

Open one of these and locate the call. For vLLM, search `vllm/attention/backends/flash_attn.py` — you will see `flash_attn_with_kvcache` and `flash_attn_varlen_func` used directly. This is exactly the API we cover in Lecture 8.

---

## Measure it

Use the official benchmark:

```
python benchmarks/benchmark_flash_attention.py --mode fwd --batch_size 2 --seqlen 4096 --nheads 16 --headdim 128
```

It prints time, TFLOPs, and HBM bandwidth for FA2 and PyTorch SDPA at the requested shape. Run a small sweep and save the CSV; you will use this baseline in Lecture 6 to validate your FA2 work-partitioning understanding.

---

## Ship it

In your `flash-attn-course/`:

1. `flash_attn_call_trace.md` — your annotated chain from `flash_attn_func` to the kernel launch.
2. `local_build_notes.md` — exact `pip install` line, total build time, any errors you hit and how you fixed them.
3. A "hello, kernel" patch you can show: a printf, a no-op MMA-count counter, or a benign tile-size override gated by an env var. Anything that proves you can edit and rebuild a kernel and have Python see the change.

These three artifacts are the prerequisite for every later lab.

---

## Related pages

- [Lecture 4 — GPU kernel performance basics](Lecture%2004%20-%20GPU%20Kernel%20Performance%20Basics.md)
- [Lecture 6 — FlashAttention-2](Lecture%2006%20-%20FlashAttention-2.md)
- [Lecture 8 — Inference path](Lecture%2008%20-%20Inference%20Path%20KV%20Cache%20and%20Decode.md)
