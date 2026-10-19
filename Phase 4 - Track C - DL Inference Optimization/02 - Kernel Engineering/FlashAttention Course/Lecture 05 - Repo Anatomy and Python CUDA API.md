# Lecture 5 — Trace a pinned API to the right kernel family

**Course:** [FlashAttention](Guide.md) · **Reviewed:** 18 October 2026

**Goal:** distinguish the distribution, import surface and implementation before building. **Artifact:** a source map at revision `965c9212e6be33c821ee9f2b461564c5c91a2503`, inspected on 18 October 2026. No FlashAttention GPU distribution was installed or executed in the course CPU environment.

## 1. Separate current package families

| Family | Source location in this revision | Public entry example | Evidence boundary |
|---|---|---|---|
| FA2 CUDA path | `flash_attn/flash_attn_interface.py`, `csrc/flash_attn/` | `from flash_attn import flash_attn_func` | Source reviewed |
| FA3 Hopper path | `hopper/` | `from flash_attn_3 import flash_attn_interface` | Source reviewed; separate build/package |
| FA4 CuTeDSL | `flash_attn/cute/` | `from flash_attn.cute import flash_attn_func` | Installable prerelease, source reviewed |
| PyTorch SDPA | PyTorch's selected backend | `torch.nn.functional.scaled_dot_product_attention` | CPU math actually executed |

PyPI metadata at review time lists `flash-attn` 2.8.3.post2 and `flash-attn-4` 4.0.0b35. A current source commit and a released distribution are distinct snapshots. The FA3 PyPI project reports 0.0.0; do not infer its source revision or compatibility from that placeholder version alone. Use the selected source/build record. [Pinned upstream README](https://github.com/Dao-AILab/flash-attention/blob/965c9212e6be33c821ee9f2b461564c5c91a2503/README.md) · [FA4 package metadata](https://pypi.org/pypi/flash-attn-4/json).

## 2. Read the actual FA2 call chain

At the pinned revision, the Python interface wraps an autograd function and a Python-registered custom operation. Its forward wrapper calls `flash_attn_gpu.fwd`, with `flash_attn_gpu` bound to the `flash_attn_2_cuda` extension for the selected NVIDIA path. C++ bindings and dispatch then select the launch/kernel. It is incorrect to search only C++ for the Python custom-op registration or assume a nonexistent `_C.pyi` proves the call chain.

Read these exact files and record the symbols you find:

1. [Python interface](https://github.com/Dao-AILab/flash-attention/blob/965c9212e6be33c821ee9f2b461564c5c91a2503/flash_attn/flash_attn_interface.py): `flash_attn_func`, autograd wrapper, `_flash_attn_forward` and `flash_attn_gpu.fwd`.
2. [C++ API](https://github.com/Dao-AILab/flash-attention/blob/965c9212e6be33c821ee9f2b461564c5c91a2503/csrc/flash_attn/flash_api.cpp): bindings, validation and forward dispatch.
3. [Forward launch template](https://github.com/Dao-AILab/flash-attention/blob/965c9212e6be33c821ee9f2b461564c5c91a2503/csrc/flash_attn/src/flash_fwd_launch_template.h): tile traits and launch choices.

The standard interface expects `[batch, sequence, heads, dim]`; PyTorch SDPA uses `[batch, heads, sequence, dim]`. A transpose is a layout operation, not permission to ignore stride and contiguity requirements. Varlen packed inputs have different sequence metadata; cache-update APIs mutate state and have separate contracts.

## 3. Audit support before installing

The pinned FA2 NVIDIA README specifies CUDA 12+, supported Ampere/Ada/Hopper devices, FP16/BF16 and stated head-dimension constraints. That is not a guarantee for every Jetson wheel, CUDA/PyTorch ABI or feature combination. FA3 and FA4 have their own hardware/build requirements.

For a hardware extension, create an isolated environment, record Python/PyTorch/CUDA/compiler/driver/GPU, then select a revision or wheel before installation. If building, choose a concurrency limit from available RAM; no fixed build time is promised. Read the actual setup file before inventing an environment flag. Run forward/backward correctness after import; import success alone only establishes that loading succeeded.

The source [benchmark script](https://github.com/Dao-AILab/flash-attention/blob/965c9212e6be33c821ee9f2b461564c5c91a2503/benchmarks/benchmark_flash_attention.py) uses in-file shape lists in this snapshot. It does not implement the old lesson's `--seqlen`/`--headdim` argument parser. The course instead supplies [sdpa_experiment.py](labs/sdpa_experiment.py), whose declared arguments are exercised on CPU. Its optional GPU path selects one backend and does not silently substitute a benchmark label.

## 4. Practice and answers

**Task:** choose one family and build a table of input shape, dtype, mask alignment, output, mutation and backward support from its actual interface. Then compare it with the CPU fixture contract.

**Answer check:** an API name shared by FA2, FA3 and FA4 does not establish identical optional arguments or return values. A PyTorch backend name does not establish that the separately installed Dao-AILab distribution was called. Capture versions and inspect/profiling evidence for the path actually used.
