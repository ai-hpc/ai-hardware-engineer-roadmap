# jetson-llm

Memory-first LLM inference runtime for NVIDIA Jetson Orin.

**Target hardware:** Orin Nano Super 8 GB (SM 8.7, 102 GB/s, 67 TOPS)
**Not supported:** x86, discrete GPUs, Windows, macOS — Jetson only.

---

## 🚀 Production implementation: `GeniePod/genie-ai-runtime` v1.0.0

The working, hardware-validated runtime — under active development as part of the GeniePod local home-AI stack — lives at:

### 👉 **[`GeniePod/genie-ai-runtime`](https://github.com/GeniePod/genie-ai-runtime)** &nbsp;·&nbsp; [release v1.0.0](https://github.com/GeniePod/genie-ai-runtime/releases/tag/v1.0.0)

This folder is the **original framework / scaffold** from which the production implementation was forked. It documents the initial architecture and the first-tokens bring-up plan. The production repo is where every subsequent kernel optimization, persistent-KV work, and production-hardening step happened.

### v1.0.0 verified on Jetson Orin Nano Super 8 GB (Qwen3-4B-Q4_K_M, 25 W MAXN SUPER)

| Workload | Number |
| -------- | ------ |
| Prefill (33-tok cold) | **38.0 tok/s** |
| Decode | **9.9 tok/s** |
| Cold TTFT | **877 ms** |
| Warm-turn TTFT (persistent KV, 67 % prefix) | **444 ms** |
| KV pool memory @ 1024 ctx | **74 MB** (INT8 default) |
| Model load — cold (NVMe) | **30 s** at 79 MB/s |
| Model load — warm (pagecache) | **1.3 s** at 1.8 GB/s |
| vs `llama-bench pp18 = 17.97 ± 0.65 tok/s` | **+115 % prefill** |

Output stays sensibly-identical to FP16 reference across the entire optimization path (FP16-ULP-bounded drift from tensor-core `mma.sync`; INT8-precision-floor drift from per-(layer, pos, kv_head) KV quantization).

### What the production repo adds beyond this scaffold

- **Paths A → I** (numbered optimization umbrellas), each with phase plans, per-PR verified-result tables on Jetson, and rollback discipline. See [`ROADMAP.md`](https://github.com/GeniePod/genie-ai-runtime/blob/main/ROADMAP.md) in the production repo for the full narrative.
- **Tensor-core MMQ Q4_K prefill GEMM** (Path E) — multi-warp cooperative dequant, `mma.sync.aligned.m16n8k16` on SM 8.7. +147 % prefill over the scalar path.
- **uint32-load decode GEMV** (Path C) — quad-byte coalesced weight loads + residual-fused output. +21 % decode.
- **Persistent KV cache** (Path F) — per-conversation save / hydrate with longest-prefix match, model fingerprint validation, LRU eviction at 1 GB cap. Warm-turn TTFT drops 48 %.
- **INT8 KV cache** (Path I, default) — per-(layer, pos, kv_head) absmax-scaled. 144 MB → 74 MB at 1024 ctx, sensibly-identical output.
- **OpenAI-compatible HTTP server** — built on cpp-httplib + nlohmann/json (the same libs llama.cpp's server uses), SSE streaming, Qwen3 reasoning split into `reasoning_content`, systemd unit + installer. Opt-in build (`-DJLLM_BUILD_SERVER=ON`); the engine ships as an embeddable library by default.
- **Production hardening** — `--version` flag, cold/warm load timing, OOM-guard prevents crashes, stability soak harness for 1000+ token × 100 iteration runs.

### Patterns that worked

Three patterns from the alpha-track that travel well to other inference-engine projects (worth stealing if you're doing similar work):

1. **Path-based umbrella issues with staged phases.** Every numbered Path = one GitHub umbrella issue with a phase table, risks named up front, and small per-phase PRs that each posted a verified-result comment on Jetson before merge. Made negative results (Path G's no-op attempts, Path C's split-K dead end, Path E's E2 microbenchmark) easy to absorb and roll back.
2. **"Sensibly-identical, not byte-identical" as the quality bar.** Once `mma.sync` reordered float accumulation, byte equality was off the table. Defining "FP16-ULP-bounded drift, character-equivalent on the canonical prompt" unlocked the whole tensor-core path. Same compromise enabled INT8 KV.
3. **Honest perf re-baselines.** Every release re-measured the prior release's number same-day with the same prompt and machine state. We never compared cross-day numbers — environmental noise had bitten us once and we baked the discipline in after.

---

## Why this matters (and why generic runtimes don't fit)

Existing runtimes are not designed for 8 GB unified memory shared with voice STT, TTS, denoise, and a Home Assistant container:

- **llama.cpp** — portable but generic CUDA kernels, no Jetson memory awareness. The runtime the production repo aims to replace inside GenieClaw.
- **TensorRT-LLM** — fast but datacenter-shaped (A100/H100), too heavy for Orin Nano's iGPU budget.
- **jetson-llm / `genie-ai-runtime`** — memory-first, power-aware, Orin SM 8.7-tuned CUDA kernels, pre-allocated KV/scratch pools, single binary, single GGUF, single shared-memory budget that fits alongside `whisper-server` and `genie-core`.

## Architecture (as originally scaffolded — see production repo for current state)

```
┌──────────────────────────────────────────────────────┐
│                   jetson-llm                          │
│                                                       │
│  Serving Layer (OpenAI-compatible REST API)           │
│    POST /v1/chat/completions | GET /health            │
│                                                       │
│  Engine (GGUF load → prefill → decode → sample)      │
│    transformer_layer() × N_layers per token           │
│    Memory guard + thermal check per token             │
│                                                       │
│  CUDA Kernels (SM 8.7 tuned)                          │
│    gemv_q4 | fused_rmsnorm | flash_attn | rope        │
│    swiglu | softmax | fp16↔int8                       │
│                                                       │
│  Memory Manager                                       │
│    MemoryBudget | OOMGuard | KVCachePool | ScratchPool│
│                                                       │
│  Jetson HAL                                           │
│    PowerState | ThermalState | LiveStats | JetsonInfo │
└──────────────────────────────────────────────────────┘
```

The production repo extends this with:
- Tensor-core MMQ Q4_K prefill GEMM (a new kernel that didn't exist in the scaffold)
- Persistent KV cache module (`src/persistence/`)
- cpp-httplib + nlohmann/json-based server (replacing the original raw-sockets server)
- `scripts/soak.sh` and `scripts/bench_load.sh` for stability + load-time validation

## How to use this folder

| You want to… | Go here |
| ------------ | ------- |
| Run, build, or contribute to the actual implementation | **[`GeniePod/genie-ai-runtime`](https://github.com/GeniePod/genie-ai-runtime)** |
| Read v1.0.0 release notes | [Release v1.0.0](https://github.com/GeniePod/genie-ai-runtime/releases/tag/v1.0.0) |
| Read the alpha-track narrative (Paths A→I, what we tried, what failed) | [`ROADMAP.md` (production repo)](https://github.com/GeniePod/genie-ai-runtime/blob/main/ROADMAP.md) |
| Read per-release verified numbers | [`CHANGELOG.md` (production repo)](https://github.com/GeniePod/genie-ai-runtime/blob/main/CHANGELOG.md) |
| Read the HTTP server reference | [`docs/server.md` (production repo)](https://github.com/GeniePod/genie-ai-runtime/blob/main/docs/server.md) |
| Read the original first-tokens roadmap (this folder's planning doc) | [`ROADMAP.md`](./ROADMAP.md) |
| Read the original test plan | [`TESTING.md`](./TESTING.md) |
| See the original scaffolded modules | [`src/`](./src/), [`include/`](./include/), [`tests/`](./tests/), [`scripts/`](./scripts/) |

## Original scaffold (preserved as the starting framework)

The rest of this README documents the framework as it was originally scaffolded for the AI Hardware Engineer Roadmap exercise. Everything below describes the **initial** module layout, completed components, and bug-tracking notes from the pre-v0.1 bring-up — preserved here as historical context for the roadmap. The production repo has moved well past it.

### Completed (as of initial scaffold, ✅)

| Component | Files | Lines | Status |
|-----------|-------|-------|--------|
| **Memory Manager** | budget.cpp, kv_cache.cpp, pool.cpp | ~300 | ✅ Implemented, tested |
| **Jetson HAL** | power.cpp, thermal.cpp, sysinfo.cpp | ~250 | ✅ Reads sysfs, tested |
| **CUDA Kernels** | 6 .cu files | ~500 | ✅ Implemented, 5 correctness tests pass |
| **GGUF Config Parser** | model.cpp (load_gguf_config) | ~80 | ✅ Reads model architecture |
| **GGUF Tensor Parser** | model.cpp (parse_tensor_infos) | ~150 | ✅ Parses tensor name/shape/offset |
| **Weight Mapping** | model.cpp (load_and_map_weights) | ~80 | ✅ Maps tensor names → struct pointers |
| **Tokenizer** | tokenizer.cpp | ~160 | ✅ Reads GGUF vocab, encode/decode |
| **Sampling** | sample.cpp | ~120 | ✅ Top-k, top-p, temperature, repeat penalty |
| **Transformer Forward** | decode.cpp (transformer_layer) | ~100 | ✅ Wires all 12 ops per layer |
| **Decode Loop** | decode.cpp (generate) | ~80 | ✅ Prefill + decode + streaming |
| **CLI** | main.cpp | ~120 | ✅ Interactive + single prompt + OOM pre-check |
| **HTTP Server** | http_server.cpp, main_server.cpp | ~250 | ✅ /health, /v1/chat/completions, /v1/models |
| **Scripts** | setup, bench, profile | ~180 | ✅ First-time setup, benchmark, nsys profiling |
| **Tests** | test_memory, test_kernels, test_model_load | ~280 | ✅ Memory, 5 kernel tests, 8 model load tests |
| **Total** | **30 files** | **~4,200** | (scaffold; see production repo for current ~10 k LOC) |

### Initial bugs that were fixed during bring-up (✅, historical)

| # | Bug | Fix applied |
|---|-----|-------------|
| 1 | GGUF KV skip miscalculated offsets | Rewrote with exact GGUF type sizes via `gguf_scalar_size()` helper; arrays of scalars skip in one `fseek` |
| 2 | Residual connection not chained | Added `vec_add()` kernel: `x2 = x + attn_proj`, then `x = x2 + ffn_out` |
| 3 | Embedding memcpy wrong direction | Changed to `cudaMemcpyDefault` (works for both host mmap and device memory) |
| 4 | Missing `<sys/mman.h>` include | Added `#include <sys/mman.h>` |
| 5 | CUDA graph body empty | Implemented full graph capture: all transformer layers + final norm + logit projection |
| 6 | Attention accumulator `acc[d%4]` | Replaced with per-dimension `s_out[head_dim]` in shared memory |
| 7 | FP16 logits, no FP32 | Added `fp16_to_fp32()` GPU kernel; convert on device before D2H copy |
| 8 | Tokenizer O(V×L) scan | Added `token_to_id_` hash map + `max_token_len_` for O(max_len) longest-match |

The Path A → I work in the production repo addresses an entirely different class of issues (kernel architecture, memory layout, correctness/perf tradeoffs at scale).

### Original v0.1 → v0.4 milestone roadmap (✅ delivered + extended)

```
v0.1 — First Tokens (DONE → see alpha.2 in production repo)
  ✅ All 8 bring-up bugs fixed
  ✅ Build on Jetson (cmake + make)
  ✅ test_model_load passes with Qwen3-4B Q4_K_M
  ✅ Generate coherent text

v0.2 — Benchmark Baseline (DONE → alpha.2 baseline established)
  ✅ bench.sh produces tok/s numbers
  ✅ Compared against llama-bench (17.97 tok/s pp18 baseline)

v0.3 — Performance Target (DONE → exceeded in alpha.8)
  ✅ >20% faster than stock llama.cpp on decode (decode: parity; prefill: +115%)
  ✅ CUDA graph replay verified working
  □ Memory-stable over 1000+ tokens (soak harness shipped, full run pending — issue #4)

v0.4 — Production Ready (DONE → v1.0.0)
  ✅ Chat template support + SSE streaming
  ✅ Multi-turn conversation (Path F persistent KV)
  ✅ Documented performance table across releases (CHANGELOG.md alpha.2 → v1.0.0)
  □ 24-hour stability test (issue #7, scheduled post-v1.0)
```

## License

MIT — same as the production repo, on purpose. The runtime is infrastructure that other projects should be able to embed cheaply.
